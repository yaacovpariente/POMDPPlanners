// ContinuousPush POMDP native sampling hot path, built on the shared
// pomdp_native core.
//
// Design note: Push's state is 6-D ([robot_xy, object_xy, target_xy]) but
// the additive Gaussian noise is only 2-D (robot displacement) and the
// object / target updates are deterministic contact-mechanics. The generic
// TransitionModelCpp<6> / ObservationModelCpp<6> templates assume a single
// Dim-dimensional Gaussian added to the full state vector; that doesn't
// match Push. Instead of bloating pomdp_native with a variant base, this
// extension builds directly on GaussianND<2> and writes its own sample /
// probability / batch_sample loop. The contact-geometry helpers are
// hand-translated from continuous_push_geometry.py into private member
// functions so the batch path is a single C++ round-trip per belief
// update. Parity tests (test_continuous_push_native_equivalence.py)
// cross-check bit-exact agreement with the Python reference.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "pomdp_native/gaussian.hpp"
#include "pomdp_native/marshalling.hpp"
#include "pomdp_native/rng.hpp"

namespace py = pybind11;

namespace {

constexpr std::size_t kPushStateDim = 6;
constexpr std::size_t kNoiseDim = 2;

// Load the (M, 4) obstacles ndarray into a flat row-major buffer. Works
// for M==0. Rows are ``(cx, cy, hx, hy)``.
std::vector<double> load_obstacles(const py::array_t<double> &obstacles) {
    if (obstacles.ndim() != 2 || obstacles.shape(1) != 4) {
        throw std::invalid_argument("obstacles must have shape (M, 4)");
    }
    const auto m = static_cast<std::size_t>(obstacles.shape(0));
    std::vector<double> out(m * 4);
    auto u = obstacles.unchecked<2>();
    for (std::size_t i = 0; i < m; ++i) {
        out[i * 4 + 0] = u(static_cast<py::ssize_t>(i), 0);
        out[i * 4 + 1] = u(static_cast<py::ssize_t>(i), 1);
        out[i * 4 + 2] = u(static_cast<py::ssize_t>(i), 2);
        out[i * 4 + 3] = u(static_cast<py::ssize_t>(i), 3);
    }
    return out;
}

class ContinuousPushTransitionCpp {
  public:
    ContinuousPushTransitionCpp(const py::object &state_obj, const py::object &action_obj,
                                double grid_size, double push_threshold,
                                double friction_coefficient, double max_push,
                                double robot_radius,
                                const py::array_t<double> &obstacles,
                                const py::array_t<double> &covariance)
        : state_(pomdp_native::to_array<kPushStateDim>(state_obj, "state")),
          action_(pomdp_native::to_array<kNoiseDim>(action_obj, "action")),
          grid_size_(grid_size),
          push_threshold_(push_threshold),
          friction_coefficient_(friction_coefficient),
          max_push_(max_push),
          robot_radius_(robot_radius),
          obstacles_(load_obstacles(obstacles)),
          n_obstacles_(obstacles_.size() / 4),
          noise_(pomdp_native::GaussianND<kNoiseDim>::from_covariance(covariance)) {}

    py::array_t<double> state_property() const {
        return pomdp_native::array_from_vector(state_.data(), kPushStateDim);
    }
    py::array_t<double> action_property() const {
        return pomdp_native::array_from_vector(action_.data(), kNoiseDim);
    }

    py::list sample(int n_samples) const {
        if (n_samples < 0) {
            throw std::invalid_argument("n_samples must be non-negative");
        }
        py::list out;
        pomdp_native::RNGState &rng = pomdp_native::default_rng();
        double row[kPushStateDim];  // NOLINT(modernize-avoid-c-arrays)
        for (int i = 0; i < n_samples; ++i) {
            sample_row_from_state(state_.data(), row, rng);
            out.append(pomdp_native::array_from_vector(row, kPushStateDim));
        }
        return out;
    }

    // probability evaluates p(next_state) = N(robot_next | robot_pos + action, cov).
    // Matches ContinuousPushStateTransitionModel.probability exactly.
    py::array_t<double> probability(const py::object &values) const {
        auto batch = pomdp_native::extract_rows_nd(values, kPushStateDim);
        double robot_mean[kNoiseDim];  // NOLINT(modernize-avoid-c-arrays)
        robot_mean[0] = state_[0] + action_[0];
        robot_mean[1] = state_[1] + action_[1];

        auto out = py::array_t<double>(static_cast<py::ssize_t>(batch.n));
        auto buf = out.mutable_unchecked<1>();
        for (std::size_t i = 0; i < batch.n; ++i) {
            const double *x = batch.flat.data() + i * kPushStateDim;
            double robot_x[kNoiseDim];  // NOLINT(modernize-avoid-c-arrays)
            robot_x[0] = x[0];
            robot_x[1] = x[1];
            buf(static_cast<py::ssize_t>(i)) = std::exp(noise_.log_pdf(robot_x, robot_mean));
        }
        return out;
    }

    // batch_sample: (N, 6) -> (N, 6). Row state is varied per particle;
    // the model's stored state_ is not read on this path.
    py::array_t<double> batch_sample(
        py::array_t<double, py::array::c_style | py::array::forcecast> particles) const {
        if (particles.ndim() != 2 ||
            static_cast<std::size_t>(particles.shape(1)) != kPushStateDim) {
            throw std::invalid_argument("particles must have shape (N, 6)");
        }
        const auto n_rows = static_cast<std::size_t>(particles.shape(0));
        auto particles_view = particles.template unchecked<2>();

        auto out = py::array_t<double>(
            {static_cast<py::ssize_t>(n_rows), static_cast<py::ssize_t>(kPushStateDim)});
        auto out_view = out.template mutable_unchecked<2>();

        pomdp_native::RNGState &rng = pomdp_native::default_rng();
        double state_row[kPushStateDim];  // NOLINT(modernize-avoid-c-arrays)
        double out_row[kPushStateDim];    // NOLINT(modernize-avoid-c-arrays)
        for (std::size_t i = 0; i < n_rows; ++i) {
            for (std::size_t d = 0; d < kPushStateDim; ++d) {
                state_row[d] = particles_view(static_cast<py::ssize_t>(i),
                                              static_cast<py::ssize_t>(d));
            }
            sample_row_from_state(state_row, out_row, rng);
            for (std::size_t d = 0; d < kPushStateDim; ++d) {
                out_view(static_cast<py::ssize_t>(i),
                         static_cast<py::ssize_t>(d)) = out_row[d];
            }
        }
        return out;
    }

  private:
    // Given a 6-D input state row, draw one full next-state row (writing
    // robot, object, target into ``out``). ``out`` may not alias ``state_row``.
    void sample_row_from_state(const double *state_row, double *out,
                               pomdp_native::RNGState &rng) const {
        // robot_new = state_row[0:2] + action + noise
        double robot_mean[kNoiseDim];  // NOLINT(modernize-avoid-c-arrays)
        robot_mean[0] = state_row[0] + action_[0];
        robot_mean[1] = state_row[1] + action_[1];
        double robot_sample[kNoiseDim];  // NOLINT(modernize-avoid-c-arrays)
        noise_.sample_into(robot_sample, robot_mean, rng);

        // Resolve circle-wall collisions, then clamp to grid.
        resolve_circle_wall_collision(robot_sample);
        clamp_circle_to_grid(robot_sample);

        // Apply deterministic push using the *post-noise* robot position.
        double object_in[kNoiseDim];   // NOLINT(modernize-avoid-c-arrays)
        double object_out[kNoiseDim];  // NOLINT(modernize-avoid-c-arrays)
        object_in[0] = state_row[2];
        object_in[1] = state_row[3];
        apply_push(robot_sample, object_in, object_out);

        out[0] = robot_sample[0];
        out[1] = robot_sample[1];
        out[2] = object_out[0];
        out[3] = object_out[1];
        // Target is unchanged.
        out[4] = state_row[4];
        out[5] = state_row[5];
    }

    // Geometry helpers hand-translated from continuous_push_geometry.py.
    // All mutate pos (a 2-D buffer) in place where applicable.

    void resolve_circle_wall_collision(double *pos) const {
        for (std::size_t i = 0; i < n_obstacles_; ++i) {
            resolve_single_circle_wall(pos, &obstacles_[i * 4]);
        }
    }

    void resolve_single_circle_wall(double *pos, const double *wall) const {
        const double cx = wall[0];
        const double cy = wall[1];
        const double hx = wall[2];
        const double hy = wall[3];
        const double closest_x = std::clamp(pos[0], cx - hx, cx + hx);
        const double closest_y = std::clamp(pos[1], cy - hy, cy + hy);
        const double dx = pos[0] - closest_x;
        const double dy = pos[1] - closest_y;
        const double dist_sq = dx * dx + dy * dy;
        const double r_sq = robot_radius_ * robot_radius_;
        if (dist_sq >= r_sq) {
            return;
        }
        const double dist = dist_sq > 1e-12 ? std::sqrt(dist_sq) : 0.0;
        if (dist < 1e-12) {
            const double pen_left = pos[0] - (cx - hx);
            const double pen_right = (cx + hx) - pos[0];
            const double pen_down = pos[1] - (cy - hy);
            const double pen_up = (cy + hy) - pos[1];
            double min_pen = pen_left;
            if (pen_right < min_pen) {
                min_pen = pen_right;
            }
            if (pen_down < min_pen) {
                min_pen = pen_down;
            }
            if (pen_up < min_pen) {
                min_pen = pen_up;
            }
            // Python chains elif; ties resolve to the earliest branch.
            if (min_pen == pen_left) {
                pos[0] = cx - hx - robot_radius_;
            } else if (min_pen == pen_right) {
                pos[0] = cx + hx + robot_radius_;
            } else if (min_pen == pen_down) {
                pos[1] = cy - hy - robot_radius_;
            } else {
                pos[1] = cy + hy + robot_radius_;
            }
        } else {
            const double overlap = robot_radius_ - dist;
            pos[0] += (dx / dist) * overlap;
            pos[1] += (dy / dist) * overlap;
        }
    }

    void clamp_circle_to_grid(double *pos) const {
        const double lo = robot_radius_;
        const double hi = grid_size_ - 1.0 - robot_radius_;
        pos[0] = std::clamp(pos[0], lo, hi);
        pos[1] = std::clamp(pos[1], lo, hi);
    }

    bool point_inside_aabb(const double *point, const double *wall) const {
        const double cx = wall[0];
        const double cy = wall[1];
        const double hx = wall[2];
        const double hy = wall[3];
        return (cx - hx) <= point[0] && point[0] <= (cx + hx) &&
               (cy - hy) <= point[1] && point[1] <= (cy + hy);
    }

    // Writes the next object position into out_obj, given the post-noise
    // robot position and the pre-step object position. Mirrors
    // ContinuousPushStateTransitionModel._apply_push precisely.
    void apply_push(const double *robot_pos, const double *obj_in, double *out_obj) const {
        out_obj[0] = obj_in[0];
        out_obj[1] = obj_in[1];
        const double dx = robot_pos[0] - obj_in[0];
        const double dy = robot_pos[1] - obj_in[1];
        const double dist_to_obj = std::sqrt(dx * dx + dy * dy);
        if (dist_to_obj >= push_threshold_) {
            return;
        }
        const double action_norm =
            std::sqrt(action_[0] * action_[0] + action_[1] * action_[1]);
        if (action_norm < 1e-12) {
            return;
        }
        const double dir_x = action_[0] / action_norm;
        const double dir_y = action_[1] / action_norm;
        const double force_mag =
            std::min(action_norm, max_push_) * (1.0 - friction_coefficient_);
        double intended[kNoiseDim];  // NOLINT(modernize-avoid-c-arrays)
        intended[0] = obj_in[0] + dir_x * force_mag;
        intended[1] = obj_in[1] + dir_y * force_mag;
        for (std::size_t i = 0; i < n_obstacles_; ++i) {
            if (point_inside_aabb(intended, &obstacles_[i * 4])) {
                return;  // Blocked, keep original.
            }
        }
        out_obj[0] = std::clamp(intended[0], 0.0, grid_size_ - 1.0);
        out_obj[1] = std::clamp(intended[1], 0.0, grid_size_ - 1.0);
    }

    std::array<double, kPushStateDim> state_;
    std::array<double, kNoiseDim> action_;
    double grid_size_;
    double push_threshold_;
    double friction_coefficient_;
    double max_push_;
    double robot_radius_;
    std::vector<double> obstacles_;  // (M*4,) row-major
    std::size_t n_obstacles_;
    pomdp_native::GaussianND<kNoiseDim> noise_;
};

class ContinuousPushObservationCpp {
  public:
    ContinuousPushObservationCpp(const py::object &next_state_obj, const py::object &action_obj,
                                 double observation_noise, double grid_size)
        : next_state_(pomdp_native::to_array<kPushStateDim>(next_state_obj, "next_state")),
          action_(pomdp_native::to_array<kNoiseDim>(action_obj, "action")),
          observation_noise_(observation_noise),
          grid_size_(grid_size),
          obs_variance_(observation_noise * observation_noise),
          obs_log_normalization_(-std::log(2.0 * M_PI * observation_noise * observation_noise)) {}

    py::array_t<double> next_state_property() const {
        return pomdp_native::array_from_vector(next_state_.data(), kPushStateDim);
    }
    py::array_t<double> action_property() const {
        return pomdp_native::array_from_vector(action_.data(), kNoiseDim);
    }

    // Samples full 6-D observations: robot and target are exact; object has
    // additive isotropic Gaussian noise clipped to the grid. Mirrors
    // ContinuousPushObservationModel.sample exactly.
    py::list sample(int n_samples) const {
        if (n_samples < 0) {
            throw std::invalid_argument("n_samples must be non-negative");
        }
        py::list out;
        pomdp_native::RNGState &rng = pomdp_native::default_rng();
        std::normal_distribution<double> standard_normal(0.0, 1.0);
        double row[kPushStateDim];  // NOLINT(modernize-avoid-c-arrays)
        for (int i = 0; i < n_samples; ++i) {
            row[0] = next_state_[0];
            row[1] = next_state_[1];
            // Python uses np.random.normal(0, sigma, size=2); each call consumes
            // one standard normal then scales by sigma. Reproduce that here using
            // the module RNG.
            const double nx = standard_normal(rng.engine()) * observation_noise_;
            const double ny = standard_normal(rng.engine()) * observation_noise_;
            double obj_x = next_state_[2] + nx;
            double obj_y = next_state_[3] + ny;
            obj_x = std::clamp(obj_x, 0.0, grid_size_ - 1.0);
            obj_y = std::clamp(obj_y, 0.0, grid_size_ - 1.0);
            row[2] = obj_x;
            row[3] = obj_y;
            row[4] = next_state_[4];
            row[5] = next_state_[5];
            out.append(pomdp_native::array_from_vector(row, kPushStateDim));
        }
        return out;
    }

    // probability(values): values rows are 6-D observations; use the
    // object-position slice (cols 2:4) against next_state_[2:4]. Matches
    // ContinuousPushObservationModel.probability exactly (isotropic 2-D
    // Gaussian on object position).
    py::array_t<double> probability(const py::object &values) const {
        auto batch = pomdp_native::extract_rows_nd(values, kPushStateDim);
        auto out = py::array_t<double>(static_cast<py::ssize_t>(batch.n));
        auto buf = out.mutable_unchecked<1>();
        const double normalization = 1.0 / (2.0 * M_PI * obs_variance_);
        for (std::size_t i = 0; i < batch.n; ++i) {
            const double *x = batch.flat.data() + i * kPushStateDim;
            const double dx = x[2] - next_state_[2];
            const double dy = x[3] - next_state_[3];
            const double log_prob = -0.5 * (dx * dx + dy * dy) / obs_variance_;
            buf(static_cast<py::ssize_t>(i)) = normalization * std::exp(log_prob);
        }
        return out;
    }

    // batch_log_likelihood: next_particles (N, 6), observation (6,) -> (N,).
    // Matches ContinuousPushVectorizedUpdater.batch_observation_log_likelihood
    // bit-for-bit: isotropic Gaussian log-pdf on object-position slice.
    py::array_t<double> batch_log_likelihood(
        py::array_t<double, py::array::c_style | py::array::forcecast> next_particles,
        py::array_t<double, py::array::c_style | py::array::forcecast> observation) const {
        if (next_particles.ndim() != 2 ||
            static_cast<std::size_t>(next_particles.shape(1)) != kPushStateDim) {
            throw std::invalid_argument("next_particles must have shape (N, 6)");
        }
        if (observation.ndim() != 1 ||
            static_cast<std::size_t>(observation.shape(0)) != kPushStateDim) {
            throw std::invalid_argument("observation must have shape (6,)");
        }
        const auto n_rows = static_cast<std::size_t>(next_particles.shape(0));
        auto particles_view = next_particles.template unchecked<2>();
        auto obs_view = observation.template unchecked<1>();

        const double obs_obj_x = obs_view(2);
        const double obs_obj_y = obs_view(3);

        auto out = py::array_t<double>(static_cast<py::ssize_t>(n_rows));
        auto out_view = out.mutable_unchecked<1>();
        for (std::size_t i = 0; i < n_rows; ++i) {
            const double dx = obs_obj_x - particles_view(static_cast<py::ssize_t>(i), 2);
            const double dy = obs_obj_y - particles_view(static_cast<py::ssize_t>(i), 3);
            out_view(static_cast<py::ssize_t>(i)) =
                obs_log_normalization_ - 0.5 * (dx * dx + dy * dy) / obs_variance_;
        }
        return out;
    }

  private:
    std::array<double, kPushStateDim> next_state_;
    std::array<double, kNoiseDim> action_;
    double observation_noise_;
    double grid_size_;
    double obs_variance_;
    double obs_log_normalization_;  // log(1 / (2 pi sigma^2))
};

}  // anonymous namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native (C++) sampling hot path for Continuous Push POMDP.";

    m.def("set_seed", &pomdp_native::set_default_seed, py::arg("seed"),
          "Seed the module-level RNG used by sample().");

    py::class_<ContinuousPushTransitionCpp>(m, "ContinuousPushTransitionCpp")
        .def(py::init<const py::object &, const py::object &, double, double, double, double,
                      double, const py::array_t<double> &, const py::array_t<double> &>(),
             py::arg("state"), py::arg("action"), py::arg("grid_size"),
             py::arg("push_threshold"), py::arg("friction_coefficient"),
             py::arg("max_push"), py::arg("robot_radius"), py::arg("obstacles"),
             py::arg("covariance"))
        .def("sample", &ContinuousPushTransitionCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &ContinuousPushTransitionCpp::probability, py::arg("values"))
        .def("batch_sample", &ContinuousPushTransitionCpp::batch_sample, py::arg("particles"))
        .def_property_readonly("state", &ContinuousPushTransitionCpp::state_property)
        .def_property_readonly("action", &ContinuousPushTransitionCpp::action_property);

    py::class_<ContinuousPushObservationCpp>(m, "ContinuousPushObservationCpp")
        .def(py::init<const py::object &, const py::object &, double, double>(),
             py::arg("next_state"), py::arg("action"), py::arg("observation_noise"),
             py::arg("grid_size"))
        .def("sample", &ContinuousPushObservationCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &ContinuousPushObservationCpp::probability, py::arg("values"))
        .def("batch_log_likelihood", &ContinuousPushObservationCpp::batch_log_likelihood,
             py::arg("next_particles"), py::arg("observation"))
        .def_property_readonly("next_state", &ContinuousPushObservationCpp::next_state_property)
        .def_property_readonly("action", &ContinuousPushObservationCpp::action_property);
}
