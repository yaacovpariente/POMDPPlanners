// Continuous Light-Dark POMDP native sampling hot path, built on the
// shared pomdp_native core.
//
// Light-dark is a 2-D navigation POMDP whose:
//
// * transition is a simple additive Gaussian: next_state = state + action + noise,
//   with no clipping. This fits the stock ``TransitionModelCpp<2>`` directly --
//   the subclass only overrides ``compute_mean_from_state`` to return
//   ``state + action``.
// * observation covariance is state-dependent: particles within ``beacon_radius``
//   of any beacon use a tighter ("near") Gaussian, all others use the ("far")
//   Gaussian. Post-sampling, observations are clipped to ``[0, grid_size]``.
//   This fits ``StateDependentObservationModelCpp<2>`` via an
//   ``is_near_next_state`` hook that scans the packed beacon positions.
//
// Only the ``NORMAL_NOISE`` observation model variant is ported to C++.
// The ``NORMAL_NOISE_NO_OBS_IN_DARK`` and ``DISTANCE_BASED`` variants
// return string ``"None"`` observations in the dark and stay on the
// Python / numpy path; see the Python shim for the dispatch split.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "pomdp_native/gaussian.hpp"
#include "pomdp_native/marshalling.hpp"
#include "pomdp_native/models.hpp"
#include "pomdp_native/rng.hpp"

namespace py = pybind11;

namespace {

constexpr std::size_t kLightDarkStateDim = 2;

// Flatten a beacons ndarray (expected shape (2, N)) into interleaved
// [x0, y0, x1, y1, ...] doubles for cache-friendly per-row tests. The
// Python-side storage is (2, N); we transpose on intake.
std::vector<double> flatten_beacons(const py::array_t<double> &beacons_in) {
    if (beacons_in.ndim() != 2) {
        throw std::invalid_argument("beacons must have shape (2, N)");
    }
    auto view = beacons_in.unchecked<2>();
    if (view.shape(0) != static_cast<py::ssize_t>(kLightDarkStateDim)) {
        throw std::invalid_argument("beacons must have shape (2, N)");
    }
    const auto n_beacons = static_cast<std::size_t>(view.shape(1));
    std::vector<double> packed;
    packed.reserve(n_beacons * kLightDarkStateDim);
    for (std::size_t j = 0; j < n_beacons; ++j) {
        packed.push_back(view(0, static_cast<py::ssize_t>(j)));
        packed.push_back(view(1, static_cast<py::ssize_t>(j)));
    }
    return packed;
}

class ContinuousLightDarkTransitionCpp
    : public pomdp_native::TransitionModelCpp<kLightDarkStateDim> {
  public:
    ContinuousLightDarkTransitionCpp(const py::object &state_obj,
                                     const py::array_t<double> &action_arr,
                                     const py::array_t<double> &covariance)
        : pomdp_native::TransitionModelCpp<kLightDarkStateDim>(
              pomdp_native::to_array<kLightDarkStateDim>(state_obj, "state"),
              py::reinterpret_borrow<py::object>(action_arr),
              pomdp_native::GaussianND<kLightDarkStateDim>::from_covariance(covariance)),
          action_vec_(pomdp_native::to_array<kLightDarkStateDim>(
              py::reinterpret_borrow<py::object>(action_arr), "action")) {}

    py::array_t<double> state_property() const {
        return pomdp_native::array_from_vector(state_.data(), state_.size());
    }
    py::array_t<double> action_property() const {
        return pomdp_native::array_from_vector(action_vec_.data(), action_vec_.size());
    }

    // Rewrite only the state field; covariance/Cholesky and action stay
    // frozen so the cached factor remains valid. Lets Python keep one
    // kernel per (env, action) instead of rebuilding for every call.
    void set_state(const py::object &state_obj) {
        state_ = pomdp_native::to_array<kLightDarkStateDim>(state_obj, "state");
    }

  protected:
    void compute_mean_from_state(const double *state, double *out) const override {
        for (std::size_t i = 0; i < kLightDarkStateDim; ++i) {
            out[i] = state[i] + action_vec_[i];
        }
    }

  private:
    std::array<double, kLightDarkStateDim> action_vec_;
};

class ContinuousLightDarkObservationCpp
    : public pomdp_native::StateDependentObservationModelCpp<kLightDarkStateDim> {
  public:
    ContinuousLightDarkObservationCpp(const py::object &next_state_obj,
                                      const py::array_t<double> &action_arr,
                                      const py::array_t<double> &covariance_near,
                                      const py::array_t<double> &covariance_far,
                                      const py::array_t<double> &beacons,
                                      double beacon_radius, double grid_size)
        : pomdp_native::StateDependentObservationModelCpp<kLightDarkStateDim>(
              pomdp_native::to_array<kLightDarkStateDim>(next_state_obj, "next_state"),
              py::reinterpret_borrow<py::object>(action_arr),
              pomdp_native::GaussianND<kLightDarkStateDim>::from_covariance(covariance_near),
              pomdp_native::GaussianND<kLightDarkStateDim>::from_covariance(covariance_far)),
          beacons_packed_(flatten_beacons(beacons)),
          beacon_radius_sq_(beacon_radius * beacon_radius),
          grid_size_(grid_size) {}

    py::array_t<double> next_state_property() const {
        return pomdp_native::array_from_vector(next_state_.data(), next_state_.size());
    }
    py::array_t<double> mean_property() const {
        return pomdp_native::array_from_vector(next_state_.data(), next_state_.size());
    }
    const py::object &action_property() const { return action_; }

    // Rewrite only the next_state field; near/far Choleskys, action, and
    // beacon config stay frozen so cached factors remain valid.
    void set_next_state(const py::object &next_state_obj) {
        next_state_ = pomdp_native::to_array<kLightDarkStateDim>(next_state_obj, "next_state");
    }

  protected:
    bool is_near_next_state(const double *next_state) const override {
        const std::size_t n_beacons = beacons_packed_.size() / kLightDarkStateDim;
        for (std::size_t j = 0; j < n_beacons; ++j) {
            const double bx = beacons_packed_[j * kLightDarkStateDim];
            const double by = beacons_packed_[j * kLightDarkStateDim + 1];
            const double dx = next_state[0] - bx;
            const double dy = next_state[1] - by;
            if (dx * dx + dy * dy <= beacon_radius_sq_) {
                return true;
            }
        }
        return false;
    }

    void post_sample_transform(double *sample) const override {
        for (std::size_t i = 0; i < kLightDarkStateDim; ++i) {
            sample[i] = std::clamp(sample[i], 0.0, grid_size_);
        }
    }

  private:
    std::vector<double> beacons_packed_;  // [x0, y0, x1, y1, ...]
    double beacon_radius_sq_;
    double grid_size_;
};

// ---------------------------------------------------------------------------
// Reward helper: Standard / DangerousStates model (STANDARD variant).
// Returns (reward, in_obstacle_range) as a std::pair.
// ---------------------------------------------------------------------------
static std::pair<double, bool> compute_reward_standard(
    const double *state, const double *action, const double *goal_state,
    const std::vector<double> &obstacles_packed,  // interleaved [x0,y0,x1,y1,...]
    double goal_state_radius, double obstacle_radius, double grid_size,
    double fuel_cost, double goal_reward, double obstacle_reward) {
    const double next_x = state[0] + action[0];
    const double next_y = state[1] + action[1];
    const double gx = next_x - goal_state[0];
    const double gy = next_y - goal_state[1];
    const double dist_to_goal = std::sqrt(gx * gx + gy * gy);
    double reward = -fuel_cost - dist_to_goal;

    if (dist_to_goal <= goal_state_radius) {
        return {reward + goal_reward, false};
    }

    const double obs_r_sq = obstacle_radius * obstacle_radius;
    const std::size_t n_obs = obstacles_packed.size() / kLightDarkStateDim;
    for (std::size_t j = 0; j < n_obs; ++j) {
        const double ox = next_x - obstacles_packed[j * 2];
        const double oy = next_y - obstacles_packed[j * 2 + 1];
        if (ox * ox + oy * oy <= obs_r_sq) {
            return {reward, true};
        }
    }

    if (next_x < 0.0 || next_y < 0.0 || next_x > grid_size || next_y > grid_size) {
        return {reward + obstacle_reward, false};
    }
    return {reward, false};
}

// ---------------------------------------------------------------------------
// Terminal check matching ContinuousLightDarkPOMDP.is_terminal.
// goal OR (is_obstacle_hit_terminal AND in any obstacle).
// ---------------------------------------------------------------------------
static bool is_terminal_cpp(
    const double *state, const double *goal_state,
    const std::vector<double> &obstacles_packed,
    double goal_state_radius, double obstacle_radius,
    bool is_obstacle_hit_terminal) {
    const double gx = state[0] - goal_state[0];
    const double gy = state[1] - goal_state[1];
    if (std::sqrt(gx * gx + gy * gy) <= goal_state_radius) {
        return true;
    }
    if (!is_obstacle_hit_terminal) {
        return false;
    }
    const double r_sq = obstacle_radius * obstacle_radius;
    const std::size_t n_obs = obstacles_packed.size() / kLightDarkStateDim;
    for (std::size_t j = 0; j < n_obs; ++j) {
        const double ox = state[0] - obstacles_packed[j * 2];
        const double oy = state[1] - obstacles_packed[j * 2 + 1];
        if (ox * ox + oy * oy <= r_sq) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Native simulate_rollout for STANDARD reward model.
//
// Walk a single random rollout from initial_state. At each depth:
//   1. check terminal — break if true
//   2. pick action_array[action_indices[depth]]
//   3. compute reward (STANDARD model, stochastic obstacle draw via C++ RNG)
//   4. step transition (same additive-Gaussian kernel as the transition class)
//   5. accumulate gamma^depth * reward
//
// action_array: shape (n_actions, 2)  — all action vectors pre-stacked
// action_indices: shape (max_depth,)  — pre-drawn integer action indices
// obstacles: interleaved [x0, y0, x1, y1, ...]  (flat 1-D array)
// covariance: shape (2, 2)  — state transition covariance
// ---------------------------------------------------------------------------
double simulate_rollout(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &initial_state,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &action_array,
    const py::array_t<int, py::array::c_style | py::array::forcecast> &action_indices,
    int max_depth,
    int start_depth,
    double discount_factor,
    // reward / terminal params
    const py::array_t<double, py::array::c_style | py::array::forcecast> &goal_state_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &obstacles_arr,
    double goal_state_radius,
    double obstacle_radius,
    double grid_size,
    double fuel_cost,
    double goal_reward,
    double obstacle_reward,
    double obstacle_hit_probability,
    bool is_obstacle_hit_terminal,
    // transition params
    const py::array_t<double> &covariance) {
    if (initial_state.ndim() != 1 || static_cast<std::size_t>(initial_state.shape(0)) != kLightDarkStateDim) {
        throw std::invalid_argument("initial_state must have shape (2,)");
    }
    if (action_array.ndim() != 2 || static_cast<std::size_t>(action_array.shape(1)) != kLightDarkStateDim) {
        throw std::invalid_argument("action_array must have shape (n_actions, 2)");
    }
    const int n_actions = static_cast<int>(action_array.shape(0));
    if (action_indices.ndim() != 1) {
        throw std::invalid_argument("action_indices must be 1-D");
    }
    const int n_indices = static_cast<int>(action_indices.shape(0));
    if (obstacles_arr.ndim() != 1) {
        throw std::invalid_argument("obstacles must be a flat 1-D array [x0,y0,x1,y1,...]");
    }

    // Unpack goal state
    auto gs_view = goal_state_arr.unchecked<1>();
    const double goal_state[kLightDarkStateDim] = {gs_view(0), gs_view(1)};

    // Unpack obstacles into a local vector
    const auto n_obs_flat = static_cast<std::size_t>(obstacles_arr.shape(0));
    auto obs_view = obstacles_arr.unchecked<1>();
    std::vector<double> obstacles_packed(n_obs_flat);
    for (std::size_t i = 0; i < n_obs_flat; ++i) {
        obstacles_packed[i] = obs_view(static_cast<py::ssize_t>(i));
    }

    // Build Gaussian noise kernel from covariance (Cholesky once)
    const auto noise = pomdp_native::GaussianND<kLightDarkStateDim>::from_covariance(covariance);

    // Copy initial state
    auto state_view = initial_state.unchecked<1>();
    double state[kLightDarkStateDim] = {state_view(0), state_view(1)};
    double next_state[kLightDarkStateDim];

    auto aa_view = action_array.unchecked<2>();
    auto ai_view = action_indices.unchecked<1>();

    pomdp_native::RNGState &rng = pomdp_native::default_rng();
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    double total = 0.0;
    double gamma_power = 1.0;
    int depth = start_depth;

    while (depth < max_depth) {
        if (is_terminal_cpp(state, goal_state, obstacles_packed, goal_state_radius, obstacle_radius,
                            is_obstacle_hit_terminal)) {
            break;
        }

        // Select action
        const int idx_slot = depth - start_depth;
        if (idx_slot >= n_indices) {
            break;
        }
        int ai = ai_view(static_cast<py::ssize_t>(idx_slot));
        if (ai < 0 || ai >= n_actions) {
            ai = ai % n_actions;
        }
        const double action_x = aa_view(static_cast<py::ssize_t>(ai), 0);
        const double action_y = aa_view(static_cast<py::ssize_t>(ai), 1);
        const double action[kLightDarkStateDim] = {action_x, action_y};

        // Compute reward (STANDARD model)
        auto [base_reward, in_obstacle_range] = compute_reward_standard(
            state, action, goal_state, obstacles_packed, goal_state_radius, obstacle_radius,
            grid_size, fuel_cost, goal_reward, obstacle_reward);

        double step_reward = base_reward;
        if (in_obstacle_range) {
            if (uniform01(rng.engine()) < obstacle_hit_probability) {
                step_reward += obstacle_reward;
            }
        }

        total += gamma_power * step_reward;
        gamma_power *= discount_factor;

        // Step transition: next_state = state + action + Gaussian noise
        const double mean[kLightDarkStateDim] = {state[0] + action_x, state[1] + action_y};
        noise.sample_into(next_state, mean, rng);
        state[0] = next_state[0];
        state[1] = next_state[1];
        ++depth;
    }

    return total;
}

}  // anonymous namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native (C++) sampling hot path for the Continuous Light-Dark POMDP.";

    m.def("set_seed", &pomdp_native::set_default_seed, py::arg("seed"),
          "Seed the module-level RNG used by sample().");

    m.def("simulate_rollout", &simulate_rollout,
          py::arg("initial_state"), py::arg("action_array"), py::arg("action_indices"),
          py::arg("max_depth"), py::arg("start_depth"), py::arg("discount_factor"),
          py::arg("goal_state"), py::arg("obstacles"), py::arg("goal_state_radius"),
          py::arg("obstacle_radius"), py::arg("grid_size"), py::arg("fuel_cost"),
          py::arg("goal_reward"), py::arg("obstacle_reward"),
          py::arg("obstacle_hit_probability"), py::arg("is_obstacle_hit_terminal"),
          py::arg("covariance"),
          "Native random rollout for the STANDARD reward model. "
          "Returns discounted return from initial_state. "
          "action_indices must be a pre-drawn array of int indices (shape (max_depth-start_depth,)). "
          "obstacles must be a flat 1-D array [x0, y0, x1, y1, ...].");

    py::class_<ContinuousLightDarkTransitionCpp>(m, "ContinuousLightDarkTransitionCpp")
        .def(py::init<const py::object &, const py::array_t<double> &,
                      const py::array_t<double> &>(),
             py::arg("state"), py::arg("action"), py::arg("covariance"))
        .def("sample", &ContinuousLightDarkTransitionCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &ContinuousLightDarkTransitionCpp::probability, py::arg("values"))
        .def("batch_sample", &ContinuousLightDarkTransitionCpp::batch_sample,
             py::arg("particles"))
        .def("set_state", &ContinuousLightDarkTransitionCpp::set_state, py::arg("state"))
        .def_property_readonly("state", &ContinuousLightDarkTransitionCpp::state_property)
        .def_property_readonly("action", &ContinuousLightDarkTransitionCpp::action_property);

    py::class_<ContinuousLightDarkObservationCpp>(m, "ContinuousLightDarkObservationCpp")
        .def(py::init<const py::object &, const py::array_t<double> &,
                      const py::array_t<double> &, const py::array_t<double> &,
                      const py::array_t<double> &, double, double>(),
             py::arg("next_state"), py::arg("action"), py::arg("covariance_near"),
             py::arg("covariance_far"), py::arg("beacons"), py::arg("beacon_radius"),
             py::arg("grid_size"))
        .def("sample", &ContinuousLightDarkObservationCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &ContinuousLightDarkObservationCpp::probability, py::arg("values"))
        .def("batch_log_likelihood",
             &ContinuousLightDarkObservationCpp::batch_log_likelihood,
             py::arg("next_particles"), py::arg("observation"))
        .def("set_next_state", &ContinuousLightDarkObservationCpp::set_next_state,
             py::arg("next_state"))
        .def_property_readonly("next_state",
                               &ContinuousLightDarkObservationCpp::next_state_property)
        .def_property_readonly("mean", &ContinuousLightDarkObservationCpp::mean_property)
        .def_property_readonly("action",
                               &ContinuousLightDarkObservationCpp::action_property);
}
