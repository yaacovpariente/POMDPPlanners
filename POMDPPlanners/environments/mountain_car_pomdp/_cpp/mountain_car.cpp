// MountainCar POMDP native sampling hot path.
//
// Exposes two classes that implement the StateTransitionModel /
// ObservationModel contracts for the MountainCar environment with all of
// sample() / probability() executed in C++. The Python shim in
// mountain_car_pomdp.py inherits directly from these classes so there is no
// Python frame on the hot path between caller and math.
//
// Covariance is accepted as a 2x2 numpy array at construction time; the
// lower-triangular Cholesky factor and log-normalisation constant are
// precomputed. State is 2D so the linear algebra is hand-coded scalar ops
// (no Eigen dependency).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

constexpr double kLog2Pi = 1.8378770664093454835606594728112352797227949;  // log(2*pi)

std::mt19937_64 &global_rng() {
    static std::mt19937_64 rng{std::random_device{}()};
    return rng;
}

void set_seed(std::uint64_t seed) { global_rng().seed(seed); }

struct Gaussian2D {
    // Lower-triangular Cholesky factor L such that L * L^T = cov.
    // L = [[l00, 0], [l10, l11]]
    double l00;
    double l10;
    double l11;
    double log_normalization;  // -0.5 * (d * log(2*pi) + log|cov|)

    static Gaussian2D from_covariance(const py::array_t<double> &cov) {
        auto cov_unchecked = cov.unchecked<2>();
        if (cov_unchecked.shape(0) != 2 || cov_unchecked.shape(1) != 2) {
            throw std::invalid_argument(
                "covariance must be a 2x2 array for MountainCar noise");
        }
        const double a = cov_unchecked(0, 0);
        const double b = cov_unchecked(0, 1);
        const double c = cov_unchecked(1, 0);
        const double d = cov_unchecked(1, 1);

        if (std::abs(b - c) > 1e-12) {
            throw std::invalid_argument("covariance must be symmetric");
        }
        if (a <= 0.0) {
            throw std::invalid_argument("covariance[0,0] must be positive");
        }
        const double l00 = std::sqrt(a);
        const double l10 = b / l00;
        const double schur = d - l10 * l10;
        if (schur <= 0.0) {
            throw std::invalid_argument("covariance is not positive definite");
        }
        const double l11 = std::sqrt(schur);
        const double log_det = 2.0 * (std::log(l00) + std::log(l11));
        const double log_norm = -0.5 * (2.0 * kLog2Pi + log_det);
        return Gaussian2D{l00, l10, l11, log_norm};
    }

    // Sample z ~ N(0,I), then x = mean + L * z.
    void sample_into(std::array<double, 2> &out, const std::array<double, 2> &mean,
                     std::mt19937_64 &rng) const {
        std::normal_distribution<double> standard_normal(0.0, 1.0);
        const double z0 = standard_normal(rng);
        const double z1 = standard_normal(rng);
        out[0] = mean[0] + l00 * z0;
        out[1] = mean[1] + l10 * z0 + l11 * z1;
    }

    // log pdf via triangular solve: y = L^{-1} (x - mean); log pdf = log_norm - 0.5 * y^T y.
    double log_pdf(const std::array<double, 2> &x, const std::array<double, 2> &mean) const {
        const double dx0 = x[0] - mean[0];
        const double dx1 = x[1] - mean[1];
        const double y0 = dx0 / l00;
        const double y1 = (dx1 - l10 * y0) / l11;
        return log_normalization - 0.5 * (y0 * y0 + y1 * y1);
    }
};

std::array<double, 2> to_pair(const py::object &obj, const char *label) {
    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
        auto seq = obj.cast<py::sequence>();
        if (py::len(seq) != 2) {
            throw std::invalid_argument(std::string(label) + " must have length 2");
        }
        return {seq[0].cast<double>(), seq[1].cast<double>()};
    }
    if (py::isinstance<py::array>(obj)) {
        auto arr = obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        if (arr.ndim() != 1 || arr.shape(0) != 2) {
            throw std::invalid_argument(std::string(label) +
                                        " ndarray must be 1-D with length 2");
        }
        auto unchecked = arr.unchecked<1>();
        return {unchecked(0), unchecked(1)};
    }
    // Fall back to sequence protocol (e.g. numpy scalar tuple).
    auto seq = obj.cast<py::sequence>();
    if (py::len(seq) != 2) {
        throw std::invalid_argument(std::string(label) + " must have length 2");
    }
    return {seq[0].cast<double>(), seq[1].cast<double>()};
}

py::array_t<double> array_from_pair(const std::array<double, 2> &pair) {
    auto arr = py::array_t<double>(2);
    auto buf = arr.mutable_unchecked<1>();
    buf(0) = pair[0];
    buf(1) = pair[1];
    return arr;
}

class MountainCarTransitionCpp {
  public:
    MountainCarTransitionCpp(const py::object &state_obj, int action, double power, double gravity,
                             double max_speed, double min_position, double max_position,
                             const py::array_t<double> &covariance)
        : state_(to_pair(state_obj, "state")),
          action_(action),
          power_(power),
          gravity_(gravity),
          max_speed_(max_speed),
          min_position_(min_position),
          max_position_(max_position),
          noise_(Gaussian2D::from_covariance(covariance)) {}

    std::array<double, 2> compute_deterministic_next_state() const {
        double v = state_[1] + static_cast<double>(action_) * power_ +
                   std::cos(3.0 * state_[0]) * (-gravity_);
        v = std::clamp(v, -max_speed_, max_speed_);
        double p = state_[0] + v;
        p = std::clamp(p, min_position_, max_position_);
        if (p == min_position_ && v < 0.0) {
            v = 0.0;
        }
        return {p, v};
    }

    std::array<double, 2> clip_state(const std::array<double, 2> &s) const {
        double p = s[0];
        double v = s[1];
        v = std::clamp(v, -max_speed_, max_speed_);
        p = std::clamp(p, min_position_, max_position_);
        if (p == min_position_ && v < 0.0) {
            v = 0.0;
        }
        return {p, v};
    }

    py::list sample(int n_samples = 1) const {
        if (n_samples < 0) {
            throw std::invalid_argument("n_samples must be non-negative");
        }
        const auto det = compute_deterministic_next_state();
        const std::array<double, 2> zero_mean{0.0, 0.0};
        auto &rng = global_rng();
        py::list out;
        std::array<double, 2> noise{};
        for (int i = 0; i < n_samples; ++i) {
            noise_.sample_into(noise, zero_mean, rng);
            std::array<double, 2> s{det[0] + noise[0], det[1] + noise[1]};
            s = clip_state(s);
            out.append(array_from_pair(s));
        }
        return out;
    }

    py::array_t<double> probability(const py::object &values) const {
        const auto det = compute_deterministic_next_state();
        const auto rows = extract_2d_rows(values);
        const py::ssize_t n = static_cast<py::ssize_t>(rows.size());
        auto out = py::array_t<double>(n);
        auto out_buf = out.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < n; ++i) {
            out_buf(i) = std::exp(noise_.log_pdf(rows[static_cast<std::size_t>(i)], det));
        }
        return out;
    }

    py::tuple state_property() const { return py::make_tuple(state_[0], state_[1]); }
    int action_property() const { return action_; }
    double power_property() const { return power_; }
    double gravity_property() const { return gravity_; }
    double max_speed_property() const { return max_speed_; }
    double min_position_property() const { return min_position_; }
    double max_position_property() const { return max_position_; }

    py::array_t<double> compute_deterministic_next_state_py() const {
        return array_from_pair(compute_deterministic_next_state());
    }

  private:
    static std::vector<std::array<double, 2>> extract_2d_rows(const py::object &values) {
        std::vector<std::array<double, 2>> rows;
        if (py::isinstance<py::array>(values)) {
            auto arr = values.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
            if (arr.ndim() == 1) {
                if (arr.shape(0) != 2) {
                    throw std::invalid_argument(
                        "1-D values ndarray must have length 2");
                }
                auto u = arr.unchecked<1>();
                rows.push_back({u(0), u(1)});
                return rows;
            }
            if (arr.ndim() == 2) {
                if (arr.shape(1) != 2) {
                    throw std::invalid_argument(
                        "2-D values ndarray must have shape (n, 2)");
                }
                auto u = arr.unchecked<2>();
                rows.reserve(static_cast<std::size_t>(u.shape(0)));
                for (py::ssize_t i = 0; i < u.shape(0); ++i) {
                    rows.push_back({u(i, 0), u(i, 1)});
                }
                return rows;
            }
            throw std::invalid_argument("values ndarray must be 1-D or 2-D");
        }
        auto seq = values.cast<py::sequence>();
        rows.reserve(py::len(seq));
        for (py::ssize_t i = 0; i < py::len(seq); ++i) {
            rows.push_back(to_pair(seq[i].cast<py::object>(), "values element"));
        }
        return rows;
    }

    std::array<double, 2> state_;
    int action_;
    double power_;
    double gravity_;
    double max_speed_;
    double min_position_;
    double max_position_;
    Gaussian2D noise_;
};

class MountainCarObservationCpp {
  public:
    MountainCarObservationCpp(const py::object &next_state_obj, int action,
                              const py::array_t<double> &covariance)
        : next_state_(to_pair(next_state_obj, "next_state")),
          action_(action),
          noise_(Gaussian2D::from_covariance(covariance)) {}

    py::list sample(int n_samples = 1) const {
        if (n_samples < 0) {
            throw std::invalid_argument("n_samples must be non-negative");
        }
        auto &rng = global_rng();
        py::list out;
        std::array<double, 2> obs{};
        for (int i = 0; i < n_samples; ++i) {
            noise_.sample_into(obs, next_state_, rng);
            out.append(array_from_pair(obs));
        }
        return out;
    }

    py::array_t<double> probability(const py::object &values) const {
        const auto rows = extract_2d_rows(values);
        const py::ssize_t n = static_cast<py::ssize_t>(rows.size());
        auto out = py::array_t<double>(n);
        auto out_buf = out.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < n; ++i) {
            out_buf(i) = std::exp(noise_.log_pdf(rows[static_cast<std::size_t>(i)], next_state_));
        }
        return out;
    }

    py::tuple next_state_property() const {
        return py::make_tuple(next_state_[0], next_state_[1]);
    }
    int action_property() const { return action_; }
    py::array_t<double> mean_property() const { return array_from_pair(next_state_); }

  private:
    static std::vector<std::array<double, 2>> extract_2d_rows(const py::object &values) {
        std::vector<std::array<double, 2>> rows;
        if (py::isinstance<py::array>(values)) {
            auto arr = values.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
            if (arr.ndim() == 1) {
                if (arr.shape(0) != 2) {
                    throw std::invalid_argument("1-D values ndarray must have length 2");
                }
                auto u = arr.unchecked<1>();
                rows.push_back({u(0), u(1)});
                return rows;
            }
            if (arr.ndim() == 2) {
                if (arr.shape(1) != 2) {
                    throw std::invalid_argument("2-D values ndarray must have shape (n, 2)");
                }
                auto u = arr.unchecked<2>();
                rows.reserve(static_cast<std::size_t>(u.shape(0)));
                for (py::ssize_t i = 0; i < u.shape(0); ++i) {
                    rows.push_back({u(i, 0), u(i, 1)});
                }
                return rows;
            }
            throw std::invalid_argument("values ndarray must be 1-D or 2-D");
        }
        auto seq = values.cast<py::sequence>();
        rows.reserve(py::len(seq));
        for (py::ssize_t i = 0; i < py::len(seq); ++i) {
            rows.push_back(to_pair(seq[i].cast<py::object>(), "values element"));
        }
        return rows;
    }

    std::array<double, 2> next_state_;
    int action_;
    Gaussian2D noise_;
};

}  // anonymous namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native (C++) sampling hot path for MountainCar POMDP.";

    m.def("set_seed", &set_seed, py::arg("seed"),
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
