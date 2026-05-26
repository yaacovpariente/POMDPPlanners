// SPDX-License-Identifier: MIT

// Safety Ant Velocity POMDP native sampling hot path.
//
// The transition model does NOT fit the shared
// ``pomdp_native::TransitionModelCpp<Dim>`` Gaussian-around-deterministic-mean
// template: the noise is a uniformly-sampled force direction
// ``theta ~ Uniform(-pi, pi)`` that feeds into a deterministic damped-force
// integration step. We therefore implement ``SafeAntVelocityTransitionCpp``
// as a standalone class (mirroring the continuous-laser-tag approach) and
// draw angles from the shared module-level ``RNGState``.
//
// The observation model, by contrast, is an identity-mean diagonal Gaussian
// on the 4-D next state and reuses ``pomdp_native::ObservationModelCpp<4>``
// for ``sample`` / ``probability`` / ``batch_log_likelihood``.
//
// Physics (matches SafeAntVelocityStateTransition.sample in
// safety_ant_velocity_pomdp.py):
//
//   force_magnitude = force_scales[action] * max_force
//   theta           = Uniform(-pi, pi)
//   force           = force_magnitude * (cos theta, sin theta)
//   accel           = (force - damping * velocity) / mass
//   next_velocity   = velocity + accel * dt
//   next_position   = position + next_velocity * dt

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pomdp_native/gaussian.hpp"
#include "pomdp_native/marshalling.hpp"
#include "pomdp_native/models.hpp"
#include "pomdp_native/rng.hpp"

namespace py = pybind11;

namespace {

constexpr std::size_t kSafeAntStateDim = 4;
constexpr double kPi = 3.14159265358979323846;

// Free physics helper shared by the transition class and simulate_rollout.
inline void integrate_one_step_free(const double *state, double force_magnitude, double theta,
                                    double *out, double dt, double mass, double damping) {
    const double px = state[0];
    const double py_ = state[1];
    const double vx = state[2];
    const double vy = state[3];

    const double fx = force_magnitude * std::cos(theta);
    const double fy = force_magnitude * std::sin(theta);

    const double ax = (fx - damping * vx) / mass;
    const double ay = (fy - damping * vy) / mass;
    const double nvx = vx + ax * dt;
    const double nvy = vy + ay * dt;

    out[0] = px + nvx * dt;
    out[1] = py_ + nvy * dt;
    out[2] = nvx;
    out[3] = nvy;
}

class SafeAntVelocityTransitionCpp {
  public:
    SafeAntVelocityTransitionCpp(const py::object &state_obj, int action, double dt, double mass,
                                 double damping, double max_force,
                                 const py::array_t<double> &force_scales)
        : state_(pomdp_native::to_array<kSafeAntStateDim>(state_obj, "state")),
          action_(action),
          dt_(dt),
          mass_(mass),
          damping_(damping),
          max_force_(max_force),
          force_scales_(unpack_force_scales(force_scales)) {
        if (mass_ <= 0.0) {
            throw std::invalid_argument("mass must be > 0");
        }
        if (force_scales_.empty()) {
            throw std::invalid_argument("force_scales must be non-empty");
        }
        if (action_ < 0 || static_cast<std::size_t>(action_) >= force_scales_.size()) {
            throw std::invalid_argument("action out of range of force_scales");
        }
    }

    py::array_t<double> state_property() const {
        return pomdp_native::array_from_vector(state_.data(), state_.size());
    }
    int action_property() const { return action_; }
    double dt_property() const { return dt_; }
    double mass_property() const { return mass_; }
    double damping_property() const { return damping_; }
    double max_force_property() const { return max_force_; }
    py::array_t<double> force_scales_property() const {
        return pomdp_native::array_from_vector(force_scales_.data(), force_scales_.size());
    }

    // Rewrite only the stored state; physics scalars and force_scales stay
    // frozen so the kernel can be reused across calls with different states
    // without paying the per-call construction cost.
    void set_state(const py::object &state_obj) {
        state_ = pomdp_native::to_array<kSafeAntStateDim>(state_obj, "state");
    }

    py::list sample(int n_samples) const {
        if (n_samples < 0) {
            throw std::invalid_argument("n_samples must be non-negative");
        }
        const double force_magnitude = force_scales_[static_cast<std::size_t>(action_)] * max_force_;
        pomdp_native::RNGState &rng = pomdp_native::default_rng();
        std::uniform_real_distribution<double> uniform_angle(-kPi, kPi);

        py::list out;
        double buf[kSafeAntStateDim];  // NOLINT(modernize-avoid-c-arrays)
        for (int i = 0; i < n_samples; ++i) {
            const double theta = uniform_angle(rng.engine());
            integrate_one_step(state_.data(), force_magnitude, theta, buf);
            out.append(pomdp_native::array_from_vector(buf, kSafeAntStateDim));
        }
        return out;
    }

    py::array_t<double> batch_sample(
        py::array_t<double, py::array::c_style | py::array::forcecast> particles) const {
        if (particles.ndim() != 2 ||
            static_cast<std::size_t>(particles.shape(1)) != kSafeAntStateDim) {
            throw std::invalid_argument("particles must have shape (N, 4)");
        }
        const auto n_rows = static_cast<std::size_t>(particles.shape(0));
        auto particles_view = particles.unchecked<2>();

        auto out = py::array_t<double>(
            {static_cast<py::ssize_t>(n_rows), static_cast<py::ssize_t>(kSafeAntStateDim)});
        auto out_view = out.mutable_unchecked<2>();

        const double force_magnitude = force_scales_[static_cast<std::size_t>(action_)] * max_force_;
        pomdp_native::RNGState &rng = pomdp_native::default_rng();
        std::uniform_real_distribution<double> uniform_angle(-kPi, kPi);

        double state_row[kSafeAntStateDim];  // NOLINT(modernize-avoid-c-arrays)
        double buf[kSafeAntStateDim];        // NOLINT(modernize-avoid-c-arrays)
        for (std::size_t i = 0; i < n_rows; ++i) {
            for (std::size_t d = 0; d < kSafeAntStateDim; ++d) {
                state_row[d] = particles_view(static_cast<py::ssize_t>(i),
                                              static_cast<py::ssize_t>(d));
            }
            const double theta = uniform_angle(rng.engine());
            integrate_one_step(state_row, force_magnitude, theta, buf);
            for (std::size_t d = 0; d < kSafeAntStateDim; ++d) {
                out_view(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(d)) = buf[d];
            }
        }
        return out;
    }

  private:
    static std::vector<double> unpack_force_scales(const py::array_t<double> &arr) {
        if (arr.ndim() != 1) {
            throw std::invalid_argument("force_scales must be a 1-D ndarray");
        }
        const auto n = static_cast<std::size_t>(arr.shape(0));
        auto u = arr.unchecked<1>();
        std::vector<double> out;
        out.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            out.push_back(u(static_cast<py::ssize_t>(i)));
        }
        return out;
    }

    void integrate_one_step(const double *state, double force_magnitude, double theta,
                            double *out) const {
        integrate_one_step_free(state, force_magnitude, theta, out, dt_, mass_, damping_);
    }

    std::array<double, kSafeAntStateDim> state_;
    int action_;
    double dt_;
    double mass_;
    double damping_;
    double max_force_;
    std::vector<double> force_scales_;
};

class SafeAntVelocityObservationCpp
    : public pomdp_native::ObservationModelCpp<kSafeAntStateDim> {
  public:
    SafeAntVelocityObservationCpp(const py::object &next_state_obj, int action,
                                  const py::array_t<double> &covariance)
        : pomdp_native::ObservationModelCpp<kSafeAntStateDim>(
              pomdp_native::to_array<kSafeAntStateDim>(next_state_obj, "next_state"),
              py::cast(action),
              pomdp_native::GaussianND<kSafeAntStateDim>::from_covariance(covariance)),
          action_int_(action) {}

    py::array_t<double> next_state_property() const {
        return pomdp_native::array_from_vector(next_state_.data(), next_state_.size());
    }
    int action_property() const { return action_int_; }
    py::array_t<double> mean_property() const {
        return pomdp_native::array_from_vector(next_state_.data(), next_state_.size());
    }

    // Rewrite only the next_state; cached Cholesky factors and log-norm
    // constant inside the GaussianND noise stay valid across calls. Permits
    // reusing one kernel for many sample / probability / batch_log_likelihood
    // invocations.
    void set_next_state(const py::object &next_state_obj) {
        next_state_ = pomdp_native::to_array<kSafeAntStateDim>(next_state_obj, "next_state");
    }

  private:
    int action_int_;
};

// ---------------------------------------------------------------------------
// simulate_rollout: single random rollout in C++.
//
// Physics and reward match SafeAntVelocityPOMDP:
//   - force_magnitude = force_scales[action] * max_force
//   - theta ~ Uniform(-pi, pi)
//   - accel = (force - damping * velocity) / mass
//   - next_velocity = velocity + accel * dt
//   - next_position = position + next_velocity * dt
//
//   reward = speed * movement_reward_scale
//            + (safety_violation_penalty if speed > safe_velocity_threshold)
//
//   terminal = (speed > safe_velocity_threshold * 1.5)
//
// action_indices: pre-drawn int32 array of length (max_depth - start_depth).
// force_scales: 1-D float64 array, length n_actions.
// ---------------------------------------------------------------------------
double simulate_rollout(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &initial_state,
    const py::array_t<int, py::array::c_style | py::array::forcecast> &action_indices,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &force_scales,
    int max_depth,
    int start_depth,
    double discount_factor,
    double dt,
    double mass,
    double damping,
    double max_force,
    double safe_velocity_threshold,
    double safety_violation_penalty,
    double movement_reward_scale) {
    if (initial_state.ndim() != 1 ||
        static_cast<std::size_t>(initial_state.shape(0)) != kSafeAntStateDim) {
        throw std::invalid_argument("initial_state must have shape (4,)");
    }
    if (force_scales.ndim() != 1 || force_scales.shape(0) < 1) {
        throw std::invalid_argument("force_scales must be a non-empty 1-D array");
    }
    const int n_actions = static_cast<int>(force_scales.shape(0));
    if (action_indices.ndim() != 1) {
        throw std::invalid_argument("action_indices must be 1-D");
    }
    const int n_indices = static_cast<int>(action_indices.shape(0));

    auto state_view = initial_state.unchecked<1>();
    auto ai_view = action_indices.unchecked<1>();
    auto fs_view = force_scales.unchecked<1>();

    double state[kSafeAntStateDim] = {
        state_view(0), state_view(1), state_view(2), state_view(3)};

    const double terminal_threshold = safe_velocity_threshold * 1.5;

    pomdp_native::RNGState &rng = pomdp_native::default_rng();
    std::uniform_real_distribution<double> uniform_angle(-kPi, kPi);

    double total = 0.0;
    double gamma_power = 1.0;
    int depth = start_depth;

    while (depth < max_depth) {
        // Terminal check: speed > 1.5 * safe_velocity_threshold
        const double vx = state[2];
        const double vy = state[3];
        const double speed = std::sqrt(vx * vx + vy * vy);
        if (speed > terminal_threshold) {
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

        const double force_magnitude = fs_view(static_cast<py::ssize_t>(ai)) * max_force;
        const double theta = uniform_angle(rng.engine());

        // Step transition
        double next_state[kSafeAntStateDim];
        integrate_one_step_free(state, force_magnitude, theta, next_state,
                                dt, mass, damping);

        // Reward from next_state (matches Python: speed of next_state drives reward)
        const double nvx = next_state[2];
        const double nvy = next_state[3];
        const double next_speed = std::sqrt(nvx * nvx + nvy * nvy);
        double step_reward = next_speed * movement_reward_scale;
        if (next_speed > safe_velocity_threshold) {
            step_reward += safety_violation_penalty;
        }

        total += gamma_power * step_reward;
        gamma_power *= discount_factor;

        state[0] = next_state[0];
        state[1] = next_state[1];
        state[2] = next_state[2];
        state[3] = next_state[3];
        ++depth;
    }
    return total;
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native (C++) sampling hot path for Safety Ant Velocity POMDP.";

    m.def("set_seed", &pomdp_native::set_default_seed, py::arg("seed"),
          "Seed the module-level RNG used by sample() / batch_sample().");

    m.def("simulate_rollout", &simulate_rollout,
          py::arg("initial_state"), py::arg("action_indices"), py::arg("force_scales"),
          py::arg("max_depth"), py::arg("start_depth"), py::arg("discount_factor"),
          py::arg("dt"), py::arg("mass"), py::arg("damping"), py::arg("max_force"),
          py::arg("safe_velocity_threshold"), py::arg("safety_violation_penalty"),
          py::arg("movement_reward_scale"),
          "Native random rollout for SafeAntVelocityPOMDP. Returns discounted reward sum. "
          "action_indices must be a pre-drawn int32 array of length (max_depth - start_depth).");

    py::class_<SafeAntVelocityTransitionCpp>(m, "SafeAntVelocityTransitionCpp")
        .def(py::init<const py::object &, int, double, double, double, double,
                      const py::array_t<double> &>(),
             py::arg("state"), py::arg("action"), py::arg("dt"), py::arg("mass"),
             py::arg("damping"), py::arg("max_force"), py::arg("force_scales"))
        .def("sample", &SafeAntVelocityTransitionCpp::sample, py::arg("n_samples") = 1)
        .def("batch_sample", &SafeAntVelocityTransitionCpp::batch_sample, py::arg("particles"))
        .def("set_state", &SafeAntVelocityTransitionCpp::set_state, py::arg("state"))
        .def_property_readonly("state", &SafeAntVelocityTransitionCpp::state_property)
        .def_property_readonly("action", &SafeAntVelocityTransitionCpp::action_property)
        .def_property_readonly("dt", &SafeAntVelocityTransitionCpp::dt_property)
        .def_property_readonly("mass", &SafeAntVelocityTransitionCpp::mass_property)
        .def_property_readonly("damping", &SafeAntVelocityTransitionCpp::damping_property)
        .def_property_readonly("max_force", &SafeAntVelocityTransitionCpp::max_force_property)
        .def_property_readonly("force_scales", &SafeAntVelocityTransitionCpp::force_scales_property);

    py::class_<SafeAntVelocityObservationCpp>(m, "SafeAntVelocityObservationCpp")
        .def(py::init<const py::object &, int, const py::array_t<double> &>(),
             py::arg("next_state"), py::arg("action"), py::arg("covariance"))
        .def("sample", &SafeAntVelocityObservationCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &SafeAntVelocityObservationCpp::probability, py::arg("values"))
        .def("batch_log_likelihood", &SafeAntVelocityObservationCpp::batch_log_likelihood,
             py::arg("next_particles"), py::arg("observation"))
        .def("set_next_state", &SafeAntVelocityObservationCpp::set_next_state,
             py::arg("next_state"))
        .def_property_readonly("next_state", &SafeAntVelocityObservationCpp::next_state_property)
        .def_property_readonly("action", &SafeAntVelocityObservationCpp::action_property)
        .def_property_readonly("mean", &SafeAntVelocityObservationCpp::mean_property);
}
