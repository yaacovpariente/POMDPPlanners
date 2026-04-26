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
#include <limits>
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

// ===========================================================================
// Discrete Light-Dark native helpers.
//
// The discrete env runs on integer-valued positions plus a small set of
// candidate observations (one per action direction + the no-noise case).
// The probability tables ``obs_probs_near`` / ``obs_probs_far`` are 1-D float
// arrays of length ``n_actions + 1`` (the last entry is the no-noise
// probability, matching the Python construction in
// DiscreteLightDarkPOMDP._precompute_sampling_tables).
//
// ``action_offsets`` is a (n_actions, 2) double array whose rows are the
// ``action_to_vector`` entries in self.actions order ([up, down, right, left]).
// ``obstacles_packed`` is the flat [x0, y0, x1, y1, ...] obstacle list.
// ``beacons_packed`` is the flat [x0, y0, x1, y1, ...] beacon list.
//
// All functions below match the Python semantics exactly:
//   - near-beacon predicate uses strict less-than against beacon_radius
//   - candidate observation matching uses exact float equality (positions
//     are integer-valued in the discrete env, so this is safe)
//   - log-prob uses np.log(p) for p>0 and -inf for p==0
// ===========================================================================

static bool discrete_is_near_beacon(double sx, double sy,
                                    const std::vector<double> &beacons_packed,
                                    double beacon_radius) {
    const double br_sq = beacon_radius * beacon_radius;
    const std::size_t n_beacons = beacons_packed.size() / kLightDarkStateDim;
    for (std::size_t j = 0; j < n_beacons; ++j) {
        const double bx = beacons_packed[j * kLightDarkStateDim];
        const double by = beacons_packed[j * kLightDarkStateDim + 1];
        const double dx = sx - bx;
        const double dy = sy - by;
        if (dx * dx + dy * dy < br_sq) {
            return true;
        }
    }
    return false;
}

// Match a single observation against the candidate list
// [ns + offset_0, ns + offset_1, ..., ns + offset_{n-1}, ns]
// and return the candidate index (or -1 if no exact match). Matches
// Python's ``np.array_equal`` over the per-dimension equality.
static int discrete_match_candidate(
    double obs_x, double obs_y, double ns_x, double ns_y,
    const double *action_offsets, std::size_t n_actions) {
    for (std::size_t k = 0; k < n_actions; ++k) {
        const double cx = ns_x + action_offsets[k * kLightDarkStateDim];
        const double cy = ns_y + action_offsets[k * kLightDarkStateDim + 1];
        if (cx == obs_x && cy == obs_y) {
            return static_cast<int>(k);
        }
    }
    if (ns_x == obs_x && ns_y == obs_y) {
        return static_cast<int>(n_actions);
    }
    return -1;
}

// Discrete is_terminal: goal-equality OR (in any obstacle).
// The Python is_terminal compares state against goal_state with np.all and
// scans obstacles with np.any-on-equality (no obstacle_hit_terminal toggle
// — discrete env always treats obstacle as terminal in is_terminal).
bool discrete_is_terminal(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &state,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &goal_state_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &obstacles_arr) {
    if (state.ndim() != 1 || static_cast<std::size_t>(state.shape(0)) != kLightDarkStateDim) {
        throw std::invalid_argument("state must have shape (2,)");
    }
    auto sv = state.unchecked<1>();
    auto gv = goal_state_arr.unchecked<1>();
    if (sv(0) == gv(0) && sv(1) == gv(1)) {
        return true;
    }
    if (obstacles_arr.ndim() != 1) {
        throw std::invalid_argument("obstacles must be a flat 1-D array");
    }
    auto ov = obstacles_arr.unchecked<1>();
    const std::size_t n_obs = static_cast<std::size_t>(ov.shape(0)) / kLightDarkStateDim;
    for (std::size_t j = 0; j < n_obs; ++j) {
        const double ox = ov(static_cast<py::ssize_t>(j * 2));
        const double oy = ov(static_cast<py::ssize_t>(j * 2 + 1));
        if (sv(0) == ox && sv(1) == oy) {
            return true;
        }
    }
    return false;
}

// Single-state observation log-probability for the NORMAL discrete observation
// model. ``observations`` may be 1-D (single obs) or 2-D (n, 2) — returns a
// 1-D float64 array of length 1 or n.
py::array_t<double> discrete_observation_log_prob(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &next_state,
    const py::object &observations,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &beacons_arr,
    double beacon_radius,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &obs_probs_near,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &obs_probs_far,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &action_offsets) {
    if (next_state.ndim() != 1 ||
        static_cast<std::size_t>(next_state.shape(0)) != kLightDarkStateDim) {
        throw std::invalid_argument("next_state must have shape (2,)");
    }
    if (action_offsets.ndim() != 2 ||
        static_cast<std::size_t>(action_offsets.shape(1)) != kLightDarkStateDim) {
        throw std::invalid_argument("action_offsets must have shape (n_actions, 2)");
    }
    if (beacons_arr.ndim() != 1) {
        throw std::invalid_argument("beacons must be a flat 1-D array");
    }
    if (obs_probs_near.ndim() != 1 || obs_probs_far.ndim() != 1) {
        throw std::invalid_argument("obs_probs_{near,far} must be 1-D arrays");
    }
    if (obs_probs_near.shape(0) != obs_probs_far.shape(0)) {
        throw std::invalid_argument("obs_probs_near and obs_probs_far must have equal length");
    }

    auto ns_view = next_state.unchecked<1>();
    const double ns_x = ns_view(0);
    const double ns_y = ns_view(1);

    auto beacons_view = beacons_arr.unchecked<1>();
    std::vector<double> beacons_packed(static_cast<std::size_t>(beacons_view.shape(0)));
    for (py::ssize_t j = 0; j < beacons_view.shape(0); ++j) {
        beacons_packed[static_cast<std::size_t>(j)] = beacons_view(j);
    }

    const bool near = discrete_is_near_beacon(ns_x, ns_y, beacons_packed, beacon_radius);
    auto probs_view = (near ? obs_probs_near : obs_probs_far).unchecked<1>();
    const std::size_t n_actions = static_cast<std::size_t>(action_offsets.shape(0));
    auto offsets_view = action_offsets.unchecked<2>();
    // Pack offsets into a flat buffer for cache-friendly per-row tests
    std::vector<double> offsets_packed(n_actions * kLightDarkStateDim);
    for (std::size_t k = 0; k < n_actions; ++k) {
        offsets_packed[k * kLightDarkStateDim] =
            offsets_view(static_cast<py::ssize_t>(k), 0);
        offsets_packed[k * kLightDarkStateDim + 1] =
            offsets_view(static_cast<py::ssize_t>(k), 1);
    }

    auto extract_observations =
        [](const py::object &obs_obj) -> std::pair<std::size_t, std::vector<double>> {
        auto arr =
            obs_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        std::vector<double> flat;
        std::size_t n_rows = 0;
        if (arr.ndim() == 1) {
            if (static_cast<std::size_t>(arr.shape(0)) != kLightDarkStateDim) {
                throw std::invalid_argument("1-D observations must have length 2");
            }
            auto u = arr.unchecked<1>();
            n_rows = 1;
            flat.push_back(u(0));
            flat.push_back(u(1));
        } else if (arr.ndim() == 2) {
            if (static_cast<std::size_t>(arr.shape(1)) != kLightDarkStateDim) {
                throw std::invalid_argument("2-D observations must have shape (n, 2)");
            }
            auto u = arr.unchecked<2>();
            n_rows = static_cast<std::size_t>(u.shape(0));
            flat.reserve(n_rows * kLightDarkStateDim);
            for (py::ssize_t i = 0; i < u.shape(0); ++i) {
                flat.push_back(u(i, 0));
                flat.push_back(u(i, 1));
            }
        } else {
            throw std::invalid_argument("observations must be 1-D or 2-D");
        }
        return {n_rows, std::move(flat)};
    };

    auto [n_rows, obs_flat] = extract_observations(observations);
    py::array_t<double> result(static_cast<py::ssize_t>(n_rows));
    auto out = result.mutable_unchecked<1>();
    for (std::size_t i = 0; i < n_rows; ++i) {
        const int idx = discrete_match_candidate(
            obs_flat[i * kLightDarkStateDim], obs_flat[i * kLightDarkStateDim + 1],
            ns_x, ns_y, offsets_packed.data(), n_actions);
        if (idx < 0) {
            out(static_cast<py::ssize_t>(i)) = -std::numeric_limits<double>::infinity();
        } else {
            const double p = probs_view(static_cast<py::ssize_t>(idx));
            out(static_cast<py::ssize_t>(i)) =
                (p > 0.0) ? std::log(p) : -std::numeric_limits<double>::infinity();
        }
    }
    return result;
}

// Per-state observation log-probability for the NORMAL discrete observation
// model. Inputs: next_states (n, 2), single observation (2,), plus beacons,
// obs prob tables, action offsets. Returns a 1-D float64 array of length n.
py::array_t<double> discrete_observation_log_prob_per_state(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &next_states,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &observation,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &beacons_arr,
    double beacon_radius,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &obs_probs_near,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &obs_probs_far,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &action_offsets) {
    if (next_states.ndim() != 2 ||
        static_cast<std::size_t>(next_states.shape(1)) != kLightDarkStateDim) {
        throw std::invalid_argument("next_states must have shape (n, 2)");
    }
    if (observation.ndim() != 1 ||
        static_cast<std::size_t>(observation.shape(0)) != kLightDarkStateDim) {
        throw std::invalid_argument("observation must have shape (2,)");
    }
    if (action_offsets.ndim() != 2 ||
        static_cast<std::size_t>(action_offsets.shape(1)) != kLightDarkStateDim) {
        throw std::invalid_argument("action_offsets must have shape (n_actions, 2)");
    }
    if (beacons_arr.ndim() != 1) {
        throw std::invalid_argument("beacons must be a flat 1-D array");
    }
    auto obs_view = observation.unchecked<1>();
    const double obs_x = obs_view(0);
    const double obs_y = obs_view(1);

    auto beacons_view = beacons_arr.unchecked<1>();
    std::vector<double> beacons_packed(static_cast<std::size_t>(beacons_view.shape(0)));
    for (py::ssize_t j = 0; j < beacons_view.shape(0); ++j) {
        beacons_packed[static_cast<std::size_t>(j)] = beacons_view(j);
    }

    auto probs_near_view = obs_probs_near.unchecked<1>();
    auto probs_far_view = obs_probs_far.unchecked<1>();
    const std::size_t n_actions = static_cast<std::size_t>(action_offsets.shape(0));
    auto offsets_view = action_offsets.unchecked<2>();
    std::vector<double> offsets_packed(n_actions * kLightDarkStateDim);
    for (std::size_t k = 0; k < n_actions; ++k) {
        offsets_packed[k * kLightDarkStateDim] =
            offsets_view(static_cast<py::ssize_t>(k), 0);
        offsets_packed[k * kLightDarkStateDim + 1] =
            offsets_view(static_cast<py::ssize_t>(k), 1);
    }

    const std::size_t n = static_cast<std::size_t>(next_states.shape(0));
    auto ns_view = next_states.unchecked<2>();
    py::array_t<double> result(static_cast<py::ssize_t>(n));
    auto out = result.mutable_unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) {
        const double ns_x = ns_view(static_cast<py::ssize_t>(i), 0);
        const double ns_y = ns_view(static_cast<py::ssize_t>(i), 1);
        const bool near = discrete_is_near_beacon(ns_x, ns_y, beacons_packed, beacon_radius);
        const int idx = discrete_match_candidate(
            obs_x, obs_y, ns_x, ns_y, offsets_packed.data(), n_actions);
        if (idx < 0) {
            out(static_cast<py::ssize_t>(i)) = -std::numeric_limits<double>::infinity();
            continue;
        }
        const double p = (near ? probs_near_view : probs_far_view)(static_cast<py::ssize_t>(idx));
        out(static_cast<py::ssize_t>(i)) =
            (p > 0.0) ? std::log(p) : -std::numeric_limits<double>::infinity();
    }
    return result;
}

// Discrete native rollout: sample a uniform-random rollout from initial_state
// using the C++ RNG (seeded via _native.set_seed). Each step:
//   1. terminal check (goal OR obstacle)
//   2. pick action_array[action_indices[depth]]
//   3. compute reward (deterministic: -fuel_cost - dist_to_goal, plus
//      goal_reward at goal, obstacle_reward on obstacle hit with probability
//      obstacle_hit_probability via C++ RNG, obstacle_reward on out-of-grid)
//   4. transition: with probability transition_error_prob the action fails
//      and a uniformly random different action vector is applied; otherwise
//      apply the chosen action vector
//   5. accumulate gamma^depth * reward
//
// Inputs:
//   action_array: shape (n_actions, 2) — action vectors stacked row-wise
//   action_indices: shape (max_depth - start_depth,) — pre-drawn rollout
//                   action indices; the per-step transition error draw and
//                   the obstacle-hit draw both use the C++ RNG.
//   obstacles: flat 1-D array [x0, y0, x1, y1, ...]
double discrete_simulate_rollout(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &initial_state,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &action_array,
    const py::array_t<int, py::array::c_style | py::array::forcecast> &action_indices,
    int max_depth, int start_depth, double discount_factor,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &goal_state_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &obstacles_arr,
    double grid_size, double fuel_cost, double goal_reward, double obstacle_reward,
    double obstacle_hit_probability, double transition_error_prob) {
    if (initial_state.ndim() != 1 ||
        static_cast<std::size_t>(initial_state.shape(0)) != kLightDarkStateDim) {
        throw std::invalid_argument("initial_state must have shape (2,)");
    }
    if (action_array.ndim() != 2 ||
        static_cast<std::size_t>(action_array.shape(1)) != kLightDarkStateDim) {
        throw std::invalid_argument("action_array must have shape (n_actions, 2)");
    }
    if (obstacles_arr.ndim() != 1) {
        throw std::invalid_argument("obstacles must be a flat 1-D array");
    }
    const int n_actions = static_cast<int>(action_array.shape(0));
    const int n_indices = static_cast<int>(action_indices.shape(0));

    auto gs_view = goal_state_arr.unchecked<1>();
    const double goal_x = gs_view(0);
    const double goal_y = gs_view(1);

    const auto n_obs_flat = static_cast<std::size_t>(obstacles_arr.shape(0));
    auto obs_view = obstacles_arr.unchecked<1>();
    std::vector<double> obstacles_packed(n_obs_flat);
    for (std::size_t i = 0; i < n_obs_flat; ++i) {
        obstacles_packed[i] = obs_view(static_cast<py::ssize_t>(i));
    }
    const std::size_t n_obstacles = n_obs_flat / kLightDarkStateDim;

    auto state_view = initial_state.unchecked<1>();
    double sx = state_view(0);
    double sy = state_view(1);

    auto aa_view = action_array.unchecked<2>();
    auto ai_view = action_indices.unchecked<1>();
    pomdp_native::RNGState &rng = pomdp_native::default_rng();
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    std::uniform_int_distribution<int> uniform_other(0, n_actions - 2);

    double total = 0.0;
    double gamma_power = 1.0;
    int depth = start_depth;

    while (depth < max_depth) {
        // Terminal: goal or any obstacle (discrete is_terminal does not
        // honor an is_obstacle_hit_terminal flag — obstacle is always
        // terminal in the Python is_terminal()).
        if (sx == goal_x && sy == goal_y) {
            break;
        }
        bool on_obstacle = false;
        for (std::size_t j = 0; j < n_obstacles; ++j) {
            if (sx == obstacles_packed[j * 2] && sy == obstacles_packed[j * 2 + 1]) {
                on_obstacle = true;
                break;
            }
        }
        if (on_obstacle) {
            break;
        }

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

        // Compute deterministic reward components — match
        // DiscreteLightDarkPOMDP._compute_reward_fast.
        const double nx = sx + action_x;
        const double ny = sy + action_y;
        const double dx = nx - goal_x;
        const double dy = ny - goal_y;
        double reward = -fuel_cost - std::sqrt(dx * dx + dy * dy);

        const bool reached_goal = (nx == goal_x && ny == goal_y);
        bool hit_obstacle_after_step = false;
        if (!reached_goal) {
            for (std::size_t j = 0; j < n_obstacles; ++j) {
                if (nx == obstacles_packed[j * 2] && ny == obstacles_packed[j * 2 + 1]) {
                    hit_obstacle_after_step = true;
                    break;
                }
            }
        }

        if (reached_goal) {
            reward += goal_reward;
        } else if (hit_obstacle_after_step) {
            if (uniform01(rng.engine()) < obstacle_hit_probability) {
                reward += obstacle_reward;
            }
        } else if (nx < 0.0 || ny < 0.0 || nx > grid_size || ny > grid_size) {
            reward += obstacle_reward;
        }

        total += gamma_power * reward;
        gamma_power *= discount_factor;

        // Stochastic transition: with probability transition_error_prob the
        // action fails and a different action vector is uniformly chosen.
        // Mirrors the Python sample_next_state semantics where probs[i] =
        // 1 - transition_error_prob and probs[k != i] = transition_error_prob
        // / (n_actions - 1).
        int chosen = ai;
        if (transition_error_prob > 0.0 &&
            uniform01(rng.engine()) < transition_error_prob) {
            int alt = uniform_other(rng.engine());
            if (alt >= ai) {
                ++alt;
            }
            chosen = alt;
        }
        sx += aa_view(static_cast<py::ssize_t>(chosen), 0);
        sy += aa_view(static_cast<py::ssize_t>(chosen), 1);
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

    m.def("discrete_is_terminal", &discrete_is_terminal,
          py::arg("state"), py::arg("goal_state"), py::arg("obstacles"),
          "Discrete-LD is_terminal: state-equals-goal OR state in any obstacle.");

    m.def("discrete_observation_log_prob", &discrete_observation_log_prob,
          py::arg("next_state"), py::arg("observations"), py::arg("beacons"),
          py::arg("beacon_radius"), py::arg("obs_probs_near"), py::arg("obs_probs_far"),
          py::arg("action_offsets"),
          "Single-state observation log-probability for the NORMAL discrete observation model.");

    m.def("discrete_observation_log_prob_per_state",
          &discrete_observation_log_prob_per_state,
          py::arg("next_states"), py::arg("observation"), py::arg("beacons"),
          py::arg("beacon_radius"), py::arg("obs_probs_near"), py::arg("obs_probs_far"),
          py::arg("action_offsets"),
          "Per-state observation log-probability for the NORMAL discrete observation model.");

    m.def("discrete_simulate_rollout", &discrete_simulate_rollout,
          py::arg("initial_state"), py::arg("action_array"), py::arg("action_indices"),
          py::arg("max_depth"), py::arg("start_depth"), py::arg("discount_factor"),
          py::arg("goal_state"), py::arg("obstacles"), py::arg("grid_size"),
          py::arg("fuel_cost"), py::arg("goal_reward"), py::arg("obstacle_reward"),
          py::arg("obstacle_hit_probability"), py::arg("transition_error_prob"),
          "Native random rollout for the discrete LightDark env. Uses the module-level "
          "C++ RNG for the obstacle-hit and transition-error draws; the per-step rollout "
          "action indices are pre-drawn on the Python side.");

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
