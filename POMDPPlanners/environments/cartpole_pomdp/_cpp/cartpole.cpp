// CartPole POMDP native sampling hot path, built on the shared
// pomdp_native core (templated on the compile-time state dimension).
// CartPole instantiates TransitionModelCpp<4> / ObservationModelCpp<4>
// so the Gaussian loops are fully unrolled by the compiler.
//
// The transition model implements the classical cart-pole physics
// (see e.g. https://coneural.org/florian/papers/05_cart_pole.pdf):
//
//   temp      = (force + polemass_length * theta_dot^2 * sin(theta)) / total_mass
//   theta_acc = (gravity * sin(theta) - cos(theta) * temp)
//             / (length * (4/3 - masspole * cos(theta)^2 / total_mass))
//   x_acc     = temp - polemass_length * theta_acc * cos(theta) / total_mass
//
// and integrates one step with either plain Euler or semi-implicit Euler.
// No state clamping is applied -- terminal states are out-of-bounds
// positions/angles that the Python reward / is_terminal methods detect
// after the fact, matching the pre-port behavior. ``post_sample_transform``
// is therefore a no-op (inherited default).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "pomdp_native/gaussian.hpp"
#include "pomdp_native/marshalling.hpp"
#include "pomdp_native/models.hpp"
#include "pomdp_native/rng.hpp"

namespace py = pybind11;

namespace {

constexpr std::size_t kCartPoleStateDim = 4;
constexpr int kEulerIntegrator = 0;
constexpr int kSemiImplicitEulerIntegrator = 1;

// ---------------------------------------------------------------------------
// Deterministic CartPole physics step (Euler or semi-implicit Euler).
// Writes next state into ``out``. ``state`` has 4 elements:
// [x, x_dot, theta, theta_dot]. ``action_int`` must be 0 or 1.
// ---------------------------------------------------------------------------
static void cartpole_step(const double *state, double *out, int action_int,
                           double force_mag, double total_mass,
                           double polemass_length, double gravity, double length,
                           int integrator, double tau, double masspole) {
    const double x = state[0];
    const double x_dot = state[1];
    const double theta = state[2];
    const double theta_dot = state[3];

    const double force = (action_int == 1) ? force_mag : -force_mag;
    const double costheta = std::cos(theta);
    const double sintheta = std::sin(theta);

    const double temp =
        (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
    const double thetaacc =
        (gravity * sintheta - costheta * temp) /
        (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
    const double xacc = temp - polemass_length * thetaacc * costheta / total_mass;

    if (integrator == kEulerIntegrator) {
        out[0] = x + tau * x_dot;
        out[1] = x_dot + tau * xacc;
        out[2] = theta + tau * theta_dot;
        out[3] = theta_dot + tau * thetaacc;
    } else {
        const double next_x_dot = x_dot + tau * xacc;
        const double next_theta_dot = theta_dot + tau * thetaacc;
        out[0] = x + tau * next_x_dot;
        out[1] = next_x_dot;
        out[2] = theta + tau * next_theta_dot;
        out[3] = next_theta_dot;
    }
}

// Returns true when the cart-pole is in a terminal state.
static bool cartpole_is_terminal(const double *state, double x_threshold,
                                  double theta_threshold) noexcept {
    const double x = state[0];
    const double theta = state[2];
    return (x < -x_threshold || x > x_threshold ||
            theta < -theta_threshold || theta > theta_threshold);
}

// ---------------------------------------------------------------------------
// Native simulate_rollout for CartPole.
//
// action_indices: pre-drawn integer action indices (0 or 1), shape (n,)
// covariance: 4x4 state-transition covariance matrix
// ---------------------------------------------------------------------------
double cartpole_simulate_rollout(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &initial_state,
    const py::array_t<int, py::array::c_style | py::array::forcecast> &action_indices,
    int max_depth,
    int start_depth,
    double discount_factor,
    double force_mag,
    double total_mass,
    double polemass_length,
    double gravity,
    double length,
    int kinematics_integrator,
    double tau,
    double masspole,
    double x_threshold,
    double theta_threshold,
    const py::array_t<double> &covariance) {
    if (initial_state.ndim() != 1 ||
        static_cast<std::size_t>(initial_state.shape(0)) != kCartPoleStateDim) {
        throw std::invalid_argument("initial_state must have shape (4,)");
    }
    if (action_indices.ndim() != 1) {
        throw std::invalid_argument("action_indices must be 1-D");
    }

    const auto noise =
        pomdp_native::GaussianND<kCartPoleStateDim>::from_covariance(covariance);

    auto state_view = initial_state.unchecked<1>();
    double state[kCartPoleStateDim] = {state_view(0), state_view(1), state_view(2), state_view(3)};
    double next_state[kCartPoleStateDim];

    auto ai_view = action_indices.unchecked<1>();
    const int n_indices = static_cast<int>(action_indices.shape(0));

    pomdp_native::RNGState &rng = pomdp_native::default_rng();

    double total = 0.0;
    double gamma_power = 1.0;
    int depth = start_depth;

    while (depth < max_depth) {
        if (cartpole_is_terminal(state, x_threshold, theta_threshold)) {
            break;
        }

        const int idx_slot = depth - start_depth;
        if (idx_slot >= n_indices) {
            break;
        }
        const int action_int = ai_view(static_cast<py::ssize_t>(idx_slot));

        // Reward for current (non-terminal) state: always 1.0
        total += gamma_power * 1.0;
        gamma_power *= discount_factor;

        // Compute deterministic next state then add Gaussian noise
        cartpole_step(state, next_state, action_int, force_mag, total_mass,
                      polemass_length, gravity, length,
                      kinematics_integrator, tau, masspole);
        noise.sample_into(state, next_state, rng);
        ++depth;
    }

    return total;
}

int encode_integrator(const std::string &name) {
    if (name == "euler") {
        return kEulerIntegrator;
    }
    if (name == "semi-implicit euler") {
        return kSemiImplicitEulerIntegrator;
    }
    throw std::invalid_argument("kinematics_integrator must be 'euler' or 'semi-implicit euler'");
}

class CartPoleTransitionCpp : public pomdp_native::TransitionModelCpp<kCartPoleStateDim> {
  public:
    CartPoleTransitionCpp(const py::object &state_obj, int action, double force_mag,
                          double total_mass, double polemass_length, double gravity, double length,
                          const std::string &kinematics_integrator, double tau, double masspole,
                          const py::array_t<double> &covariance)
        : pomdp_native::TransitionModelCpp<kCartPoleStateDim>(
              pomdp_native::to_array<kCartPoleStateDim>(state_obj, "state"), py::cast(action),
              pomdp_native::GaussianND<kCartPoleStateDim>::from_covariance(covariance)),
          action_int_(action),
          force_mag_(force_mag),
          total_mass_(total_mass),
          polemass_length_(polemass_length),
          gravity_(gravity),
          length_(length),
          kinematics_integrator_name_(kinematics_integrator),
          kinematics_integrator_(encode_integrator(kinematics_integrator)),
          tau_(tau),
          masspole_(masspole) {}

    py::array_t<double> state_property() const {
        return pomdp_native::array_from_vector(state_.data(), state_.size());
    }
    int action_property() const { return action_int_; }
    double force_mag_property() const { return force_mag_; }
    double total_mass_property() const { return total_mass_; }
    double polemass_length_property() const { return polemass_length_; }
    double gravity_property() const { return gravity_; }
    double length_property() const { return length_; }
    std::string kinematics_integrator_property() const { return kinematics_integrator_name_; }
    double tau_property() const { return tau_; }
    double masspole_property() const { return masspole_; }

    py::array_t<double> compute_deterministic_next_state_py() const {
        double out[kCartPoleStateDim];
        compute_mean_from_state(state_.data(), out);
        return pomdp_native::array_from_vector(out, kCartPoleStateDim);
    }

  protected:
    void compute_mean_from_state(const double *state, double *out) const override {
        const double x = state[0];
        const double x_dot = state[1];
        const double theta = state[2];
        const double theta_dot = state[3];

        const double force = (action_int_ == 1) ? force_mag_ : -force_mag_;
        const double costheta = std::cos(theta);
        const double sintheta = std::sin(theta);

        const double temp =
            (force + polemass_length_ * theta_dot * theta_dot * sintheta) / total_mass_;
        const double thetaacc =
            (gravity_ * sintheta - costheta * temp) /
            (length_ * (4.0 / 3.0 - masspole_ * costheta * costheta / total_mass_));
        const double xacc = temp - polemass_length_ * thetaacc * costheta / total_mass_;

        double next_x;
        double next_x_dot;
        double next_theta;
        double next_theta_dot;
        if (kinematics_integrator_ == kEulerIntegrator) {
            next_x = x + tau_ * x_dot;
            next_x_dot = x_dot + tau_ * xacc;
            next_theta = theta + tau_ * theta_dot;
            next_theta_dot = theta_dot + tau_ * thetaacc;
        } else {  // semi-implicit euler
            next_x_dot = x_dot + tau_ * xacc;
            next_x = x + tau_ * next_x_dot;
            next_theta_dot = theta_dot + tau_ * thetaacc;
            next_theta = theta + tau_ * next_theta_dot;
        }
        out[0] = next_x;
        out[1] = next_x_dot;
        out[2] = next_theta;
        out[3] = next_theta_dot;
    }

  private:
    int action_int_;
    double force_mag_;
    double total_mass_;
    double polemass_length_;
    double gravity_;
    double length_;
    std::string kinematics_integrator_name_;
    int kinematics_integrator_;
    double tau_;
    double masspole_;
};

class CartPoleObservationCpp : public pomdp_native::ObservationModelCpp<kCartPoleStateDim> {
  public:
    CartPoleObservationCpp(const py::object &next_state_obj, int action,
                           const py::array_t<double> &covariance)
        : pomdp_native::ObservationModelCpp<kCartPoleStateDim>(
              pomdp_native::to_array<kCartPoleStateDim>(next_state_obj, "next_state"),
              py::cast(action),
              pomdp_native::GaussianND<kCartPoleStateDim>::from_covariance(covariance)),
          action_int_(action) {}

    py::array_t<double> next_state_property() const {
        return pomdp_native::array_from_vector(next_state_.data(), next_state_.size());
    }
    int action_property() const { return action_int_; }
    py::array_t<double> mean_property() const {
        return pomdp_native::array_from_vector(next_state_.data(), next_state_.size());
    }

  private:
    int action_int_;
};

}  // anonymous namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native (C++) sampling hot path for CartPole POMDP.";

    m.def("set_seed", &pomdp_native::set_default_seed, py::arg("seed"),
          "Seed the module-level RNG used by sample().");

    m.def("simulate_rollout", &cartpole_simulate_rollout,
          py::arg("initial_state"), py::arg("action_indices"),
          py::arg("max_depth"), py::arg("start_depth"), py::arg("discount_factor"),
          py::arg("force_mag"), py::arg("total_mass"), py::arg("polemass_length"),
          py::arg("gravity"), py::arg("length"), py::arg("kinematics_integrator"),
          py::arg("tau"), py::arg("masspole"),
          py::arg("x_threshold"), py::arg("theta_threshold"),
          py::arg("covariance"),
          "Native random rollout for CartPole. "
          "action_indices must be a pre-drawn 1-D int32 array. "
          "Returns discounted return from initial_state.");

    py::class_<CartPoleTransitionCpp>(m, "CartPoleTransitionCpp")
        .def(py::init<const py::object &, int, double, double, double, double, double,
                      const std::string &, double, double, const py::array_t<double> &>(),
             py::arg("state"), py::arg("action"), py::arg("force_mag"), py::arg("total_mass"),
             py::arg("polemass_length"), py::arg("gravity"), py::arg("length"),
             py::arg("kinematics_integrator"), py::arg("tau"), py::arg("masspole"),
             py::arg("covariance"))
        .def("sample", &CartPoleTransitionCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &CartPoleTransitionCpp::probability, py::arg("values"))
        .def("batch_sample", &CartPoleTransitionCpp::batch_sample, py::arg("particles"))
        .def("_compute_deterministic_next_state",
             &CartPoleTransitionCpp::compute_deterministic_next_state_py)
        .def_property_readonly("state", &CartPoleTransitionCpp::state_property)
        .def_property_readonly("action", &CartPoleTransitionCpp::action_property)
        .def_property_readonly("force_mag", &CartPoleTransitionCpp::force_mag_property)
        .def_property_readonly("total_mass", &CartPoleTransitionCpp::total_mass_property)
        .def_property_readonly("polemass_length", &CartPoleTransitionCpp::polemass_length_property)
        .def_property_readonly("gravity", &CartPoleTransitionCpp::gravity_property)
        .def_property_readonly("length", &CartPoleTransitionCpp::length_property)
        .def_property_readonly("kinematics_integrator",
                               &CartPoleTransitionCpp::kinematics_integrator_property)
        .def_property_readonly("tau", &CartPoleTransitionCpp::tau_property)
        .def_property_readonly("masspole", &CartPoleTransitionCpp::masspole_property);

    py::class_<CartPoleObservationCpp>(m, "CartPoleObservationCpp")
        .def(py::init<const py::object &, int, const py::array_t<double> &>(),
             py::arg("next_state"), py::arg("action"), py::arg("covariance"))
        .def("sample", &CartPoleObservationCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &CartPoleObservationCpp::probability, py::arg("values"))
        .def("batch_log_likelihood", &CartPoleObservationCpp::batch_log_likelihood,
             py::arg("next_particles"), py::arg("observation"))
        .def_property_readonly("next_state", &CartPoleObservationCpp::next_state_property)
        .def_property_readonly("action", &CartPoleObservationCpp::action_property)
        .def_property_readonly("mean", &CartPoleObservationCpp::mean_property);
}
