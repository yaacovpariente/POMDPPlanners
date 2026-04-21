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
#include <cstddef>
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

}  // anonymous namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native (C++) sampling hot path for the Continuous Light-Dark POMDP.";

    m.def("set_seed", &pomdp_native::set_default_seed, py::arg("seed"),
          "Seed the module-level RNG used by sample().");

    py::class_<ContinuousLightDarkTransitionCpp>(m, "ContinuousLightDarkTransitionCpp")
        .def(py::init<const py::object &, const py::array_t<double> &,
                      const py::array_t<double> &>(),
             py::arg("state"), py::arg("action"), py::arg("covariance"))
        .def("sample", &ContinuousLightDarkTransitionCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &ContinuousLightDarkTransitionCpp::probability, py::arg("values"))
        .def("batch_sample", &ContinuousLightDarkTransitionCpp::batch_sample,
             py::arg("particles"))
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
        .def_property_readonly("next_state",
                               &ContinuousLightDarkObservationCpp::next_state_property)
        .def_property_readonly("mean", &ContinuousLightDarkObservationCpp::mean_property)
        .def_property_readonly("action",
                               &ContinuousLightDarkObservationCpp::action_property);
}
