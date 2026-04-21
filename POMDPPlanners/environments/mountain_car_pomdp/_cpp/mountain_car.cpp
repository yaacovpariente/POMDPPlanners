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

  private:
    int action_int_;
};

}  // anonymous namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native (C++) sampling hot path for MountainCar POMDP.";

    m.def("set_seed", &pomdp_native::set_default_seed, py::arg("seed"),
          "Seed the module-level RNG used by sample().");

    py::class_<MountainCarTransitionCpp>(m, "MountainCarTransitionCpp")
        .def(py::init<const py::object &, int, double, double, double, double, double,
                      const py::array_t<double> &>(),
             py::arg("state"), py::arg("action"), py::arg("power"), py::arg("gravity"),
             py::arg("max_speed"), py::arg("min_position"), py::arg("max_position"),
             py::arg("covariance"))
        .def("sample", &MountainCarTransitionCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &MountainCarTransitionCpp::probability, py::arg("values"))
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
        .def_property_readonly("next_state", &MountainCarObservationCpp::next_state_property)
        .def_property_readonly("action", &MountainCarObservationCpp::action_property)
        .def_property_readonly("mean", &MountainCarObservationCpp::mean_property);
}
