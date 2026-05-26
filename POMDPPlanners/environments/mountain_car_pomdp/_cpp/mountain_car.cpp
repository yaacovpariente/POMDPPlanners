// Copyright 2025 Yaacov Pariente
// SPDX-License-Identifier: MIT

// MountainCar POMDP native sampling hot path, built on the shared
// pomdp_native core (templated on the compile-time state dimension).
// MountainCar instantiates TransitionModelCpp<2> / ObservationModelCpp<2>
// so the Gaussian loops are fully unrolled by the compiler.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

#include "pomdp_native/gaussian.hpp"
#include "pomdp_native/marshalling.hpp"
#include "pomdp_native/models.hpp"
#include "pomdp_native/rng.hpp"

namespace py = pybind11;

namespace {

constexpr std::size_t kMountainCarStateDim = 2;

class MountainCarTransitionCpp : public pomdp_native::TransitionModelCpp<kMountainCarStateDim> {
  public:
    MountainCarTransitionCpp(const py::object &state_obj, int action, double power, double gravity,
                             double max_speed, double min_position, double max_position,
                             const py::array_t<double> &covariance)
        : pomdp_native::TransitionModelCpp<kMountainCarStateDim>(
              pomdp_native::to_array<kMountainCarStateDim>(state_obj, "state"),
              py::cast(action),
              pomdp_native::GaussianND<kMountainCarStateDim>::from_covariance(covariance)),
          action_int_(action),
          power_(power),
          gravity_(gravity),
          max_speed_(max_speed),
          min_position_(min_position),
          max_position_(max_position) {}

    py::tuple state_property() const { return py::make_tuple(state_[0], state_[1]); }
    int action_property() const { return action_int_; }
    double power_property() const { return power_; }
    double gravity_property() const { return gravity_; }
    double max_speed_property() const { return max_speed_; }
    double min_position_property() const { return min_position_; }
    double max_position_property() const { return max_position_; }

    py::array_t<double> compute_deterministic_next_state_py() const {
        double out[kMountainCarStateDim];
        compute_mean_from_state(state_.data(), out);
        return pomdp_native::array_from_vector(out, kMountainCarStateDim);
    }

    // Rewrite only the state field; env params (action, power, gravity,
    // max_speed, min_position, max_position, covariance) stay frozen so
    // Python can keep one kernel per (env, action) and reuse it across
    // calls. Mirrors SafeAntVelocityTransitionCpp::set_state.
    void set_state(const py::object &state_obj) {
        state_ = pomdp_native::to_array<kMountainCarStateDim>(state_obj, "state");
    }

  protected:
    void compute_mean_from_state(const double *state, double *out) const override {
        double v = state[1] + static_cast<double>(action_int_) * power_ +
                   std::cos(3.0 * state[0]) * (-gravity_);
        v = std::clamp(v, -max_speed_, max_speed_);
        double p = state[0] + v;
        p = std::clamp(p, min_position_, max_position_);
        if (p == min_position_ && v < 0.0) {
            v = 0.0;
        }
        out[0] = p;
        out[1] = v;
    }

    void post_sample_transform(double *sample) const override {
        double &p = sample[0];
        double &v = sample[1];
        v = std::clamp(v, -max_speed_, max_speed_);
        p = std::clamp(p, min_position_, max_position_);
        if (p == min_position_ && v < 0.0) {
            v = 0.0;
        }
    }

  private:
    int action_int_;
    double power_;
    double gravity_;
    double max_speed_;
    double min_position_;
    double max_position_;
};

class MountainCarObservationCpp : public pomdp_native::ObservationModelCpp<kMountainCarStateDim> {
  public:
    MountainCarObservationCpp(const py::object &next_state_obj, int action,
                              const py::array_t<double> &covariance)
        : pomdp_native::ObservationModelCpp<kMountainCarStateDim>(
              pomdp_native::to_array<kMountainCarStateDim>(next_state_obj, "next_state"),
              py::cast(action),
              pomdp_native::GaussianND<kMountainCarStateDim>::from_covariance(covariance)),
          action_int_(action) {}

    py::tuple next_state_property() const {
        return py::make_tuple(next_state_[0], next_state_[1]);
    }
    int action_property() const { return action_int_; }
    py::array_t<double> mean_property() const {
        return pomdp_native::array_from_vector(next_state_.data(), next_state_.size());
    }

    // Rewrite only the next_state field; env params (action, covariance)
    // stay frozen. Mirrors SafeAntVelocityObservationCpp::set_next_state.
    void set_next_state(const py::object &next_state_obj) {
        next_state_ =
            pomdp_native::to_array<kMountainCarStateDim>(next_state_obj, "next_state");
    }

  private:
    int action_int_;
};

// ---------------------------------------------------------------------------
// Deterministic MountainCar physics step. Writes 2 doubles to ``out``.
// ``state`` is [position, velocity]; ``action_int`` is -1, 0, or 1.
// ---------------------------------------------------------------------------
static void mountain_car_step(const double *state, double *out, int action_int,
                               double power, double gravity,
                               double max_speed, double min_position,
                               double max_position) noexcept {
    double v = state[1] + static_cast<double>(action_int) * power +
               std::cos(3.0 * state[0]) * (-gravity);
    v = std::clamp(v, -max_speed, max_speed);
    double p = state[0] + v;
    p = std::clamp(p, min_position, max_position);
    if (p == min_position && v < 0.0) {
        v = 0.0;
    }
    out[0] = p;
    out[1] = v;
}

// Returns true when position >= goal_position (terminal).
static bool mountain_car_is_terminal(const double *state,
                                      double goal_position) noexcept {
    return state[0] >= goal_position;
}

// ---------------------------------------------------------------------------
// Native simulate_rollout for MountainCar.
//
// actions_arr: shape (n_actions,) — the integer action values (e.g. [-1,0,1])
// action_indices: pre-drawn 1-D int32 indices into actions_arr, shape (n,)
// covariance: 2x2 state-transition covariance matrix
// ---------------------------------------------------------------------------
double mountain_car_simulate_rollout(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &initial_state,
    const py::array_t<int, py::array::c_style | py::array::forcecast> &actions_arr,
    const py::array_t<int, py::array::c_style | py::array::forcecast> &action_indices,
    int max_depth,
    int start_depth,
    double discount_factor,
    double power,
    double gravity,
    double max_speed,
    double min_position,
    double max_position,
    double goal_position,
    const py::array_t<double> &covariance) {
    if (initial_state.ndim() != 1 ||
        static_cast<std::size_t>(initial_state.shape(0)) != kMountainCarStateDim) {
        throw std::invalid_argument("initial_state must have shape (2,)");
    }
    if (action_indices.ndim() != 1) {
        throw std::invalid_argument("action_indices must be 1-D");
    }

    const int n_actions = static_cast<int>(actions_arr.shape(0));
    const auto noise =
        pomdp_native::GaussianND<kMountainCarStateDim>::from_covariance(covariance);

    auto state_view = initial_state.unchecked<1>();
    double state[kMountainCarStateDim] = {state_view(0), state_view(1)};
    double next_state[kMountainCarStateDim];

    auto ai_view = action_indices.unchecked<1>();
    auto av_view = actions_arr.unchecked<1>();
    const int n_indices = static_cast<int>(action_indices.shape(0));

    pomdp_native::RNGState &rng = pomdp_native::default_rng();

    double total = 0.0;
    double gamma_power = 1.0;
    int depth = start_depth;

    while (depth < max_depth) {
        if (mountain_car_is_terminal(state, goal_position)) {
            break;
        }

        const int idx_slot = depth - start_depth;
        if (idx_slot >= n_indices) {
            break;
        }
        int ai = ai_view(static_cast<py::ssize_t>(idx_slot));
        if (ai < 0 || ai >= n_actions) {
            ai = ((ai % n_actions) + n_actions) % n_actions;
        }
        const int action_int = av_view(static_cast<py::ssize_t>(ai));

        // Reward: -1.0 for non-terminal states
        total += gamma_power * (-1.0);
        gamma_power *= discount_factor;

        // Compute deterministic next state then add Gaussian noise
        mountain_car_step(state, next_state, action_int, power, gravity,
                          max_speed, min_position, max_position);
        noise.sample_into(state, next_state, rng);
        // Post-sample clamp (matches post_sample_transform)
        state[1] = std::clamp(state[1], -max_speed, max_speed);
        state[0] = std::clamp(state[0], min_position, max_position);
        if (state[0] == min_position && state[1] < 0.0) {
            state[1] = 0.0;
        }
        ++depth;
    }

    return total;
}

}  // anonymous namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native (C++) sampling hot path for MountainCar POMDP.";

    m.def("set_seed", &pomdp_native::set_default_seed, py::arg("seed"),
          "Seed the module-level RNG used by sample().");

    m.def("simulate_rollout", &mountain_car_simulate_rollout,
          py::arg("initial_state"), py::arg("actions"), py::arg("action_indices"),
          py::arg("max_depth"), py::arg("start_depth"), py::arg("discount_factor"),
          py::arg("power"), py::arg("gravity"), py::arg("max_speed"),
          py::arg("min_position"), py::arg("max_position"), py::arg("goal_position"),
          py::arg("covariance"),
          "Native random rollout for MountainCar. "
          "actions must be a 1-D int32 array of action values (e.g. [-1,0,1]). "
          "action_indices must be a pre-drawn 1-D int32 array of indices into actions. "
          "Returns discounted return from initial_state.");

    py::class_<MountainCarTransitionCpp>(m, "MountainCarTransitionCpp")
        .def(py::init<const py::object &, int, double, double, double, double, double,
                      const py::array_t<double> &>(),
             py::arg("state"), py::arg("action"), py::arg("power"), py::arg("gravity"),
             py::arg("max_speed"), py::arg("min_position"), py::arg("max_position"),
             py::arg("covariance"))
        .def("sample", &MountainCarTransitionCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &MountainCarTransitionCpp::probability, py::arg("values"))
        .def("batch_sample", &MountainCarTransitionCpp::batch_sample, py::arg("particles"))
        .def("set_state", &MountainCarTransitionCpp::set_state, py::arg("state"))
        .def("_compute_deterministic_next_state",
             &MountainCarTransitionCpp::compute_deterministic_next_state_py)
        .def_property_readonly("state", &MountainCarTransitionCpp::state_property)
        .def_property_readonly("action", &MountainCarTransitionCpp::action_property)
        .def_property_readonly("power", &MountainCarTransitionCpp::power_property)
        .def_property_readonly("gravity", &MountainCarTransitionCpp::gravity_property)
        .def_property_readonly("max_speed", &MountainCarTransitionCpp::max_speed_property)
        .def_property_readonly("min_position", &MountainCarTransitionCpp::min_position_property)
        .def_property_readonly("max_position", &MountainCarTransitionCpp::max_position_property);

    py::class_<MountainCarObservationCpp>(m, "MountainCarObservationCpp")
        .def(py::init<const py::object &, int, const py::array_t<double> &>(),
             py::arg("next_state"), py::arg("action"), py::arg("covariance"))
        .def("sample", &MountainCarObservationCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &MountainCarObservationCpp::probability, py::arg("values"))
        .def("batch_log_likelihood", &MountainCarObservationCpp::batch_log_likelihood,
             py::arg("next_particles"), py::arg("observation"))
        .def("set_next_state", &MountainCarObservationCpp::set_next_state,
             py::arg("next_state"))
        .def_property_readonly("next_state", &MountainCarObservationCpp::next_state_property)
        .def_property_readonly("action", &MountainCarObservationCpp::action_property)
        .def_property_readonly("mean", &MountainCarObservationCpp::mean_property);
}
