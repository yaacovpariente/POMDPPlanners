// SPDX-License-Identifier: MIT

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
#include <limits>
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

    // Rewrite only the state field; covariance/Cholesky, action, obstacles,
    // and geometry params stay frozen so the cached factor remains valid.
    void set_state(const py::object &state_obj) {
        state_ = pomdp_native::to_array<kPushStateDim>(state_obj, "state");
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

    // Hot-path entry: take the input state directly, draw one sample, and
    // return a single (6,) ndarray. Bypasses set_state + py::list wrapping +
    // list[0] indexing that the generic ``sample(n_samples=1)`` path incurs.
    py::array_t<double> sample_one(
        const py::array_t<double, py::array::c_style | py::array::forcecast> &state_in) const {
        if (state_in.ndim() != 1 ||
            static_cast<std::size_t>(state_in.shape(0)) != kPushStateDim) {
            throw std::invalid_argument("state must have shape (6,)");
        }
        auto sv = state_in.unchecked<1>();
        double src[kPushStateDim];  // NOLINT(modernize-avoid-c-arrays)
        for (std::size_t d = 0; d < kPushStateDim; ++d) {
            src[d] = sv(static_cast<py::ssize_t>(d));
        }
        double row[kPushStateDim];  // NOLINT(modernize-avoid-c-arrays)
        pomdp_native::RNGState &rng = pomdp_native::default_rng();
        sample_row_from_state(src, row, rng);
        return pomdp_native::array_from_vector(row, kPushStateDim);
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

    // Rewrite only the next_state field; observation_noise and grid_size
    // stay frozen so the cached normalizer/variance remain valid.
    void set_next_state(const py::object &next_state_obj) {
        next_state_ = pomdp_native::to_array<kPushStateDim>(next_state_obj, "next_state");
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

    // Hot-path single-sample entry: take the next_state directly, draw one
    // observation, return a single (6,) ndarray. Bypasses set_next_state +
    // py::list wrapping + list[0] indexing.
    py::array_t<double> sample_one(
        const py::array_t<double, py::array::c_style | py::array::forcecast> &next_state_in)
        const {
        if (next_state_in.ndim() != 1 ||
            static_cast<std::size_t>(next_state_in.shape(0)) != kPushStateDim) {
            throw std::invalid_argument("next_state must have shape (6,)");
        }
        auto sv = next_state_in.unchecked<1>();
        pomdp_native::RNGState &rng = pomdp_native::default_rng();
        std::normal_distribution<double> standard_normal(0.0, 1.0);
        double row[kPushStateDim];  // NOLINT(modernize-avoid-c-arrays)
        row[0] = sv(0);
        row[1] = sv(1);
        const double nx = standard_normal(rng.engine()) * observation_noise_;
        const double ny = standard_normal(rng.engine()) * observation_noise_;
        const double obj_x = std::clamp(sv(2) + nx, 0.0, grid_size_ - 1.0);
        const double obj_y = std::clamp(sv(3) + ny, 0.0, grid_size_ - 1.0);
        row[2] = obj_x;
        row[3] = obj_y;
        row[4] = sv(4);
        row[5] = sv(5);
        return pomdp_native::array_from_vector(row, kPushStateDim);
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

// ─── Discrete Push rollout kernel ────────────────────────────────────────────
//
// Ports the Python rollout loop in PushPOMDP.simulate_random_rollout / the
// transition in PushPOMDP._sample_one_next_state and the reward function in
// PushPOMDP._reward_from_next_state to a single C++ frame.  The caller
// pre-draws action indices in Python (via np.random.randint) and passes them
// here; the C++ layer only needs uniform-[0,1) draws for the transition-error
// coin flip (and the np.random.choice replacement for the error branch).
//
// Action index mapping (matches PushStateTransition._AVAILABLE_ACTIONS order):
//   0 -> up    (dx=0, dy=+1)
//   1 -> down  (dx=0, dy=-1)
//   2 -> right (dx=+1, dy=0)
//   3 -> left  (dx=-1, dy=0)

// dx/dy table indexed by action index 0..3.
constexpr std::array<std::array<int, 2>, 4> kDiscreteDXY = {
    {{{0, 1}},    // 0 = up
     {{0, -1}},   // 1 = down
     {{1, 0}},    // 2 = right
     {{-1, 0}}}   // 3 = left
};

// Error-action tables: for each intended action index i, the three OTHER
// indices in order (matches Python ``[a for a in actions if a != action]``).
constexpr std::array<std::array<int, 3>, 4> kErrorActions = {{
    {{1, 2, 3}},  // 0 intended -> error from {1,2,3}
    {{0, 2, 3}},  // 1 intended -> error from {0,2,3}
    {{0, 1, 3}},  // 2 intended -> error from {0,1,3}
    {{0, 1, 2}}   // 3 intended -> error from {0,1,2}
}};

// Load discrete obstacles: shape (M, 2) or empty. Returns flat (cx, cy) pairs.
std::vector<double> load_discrete_obstacles(const py::array_t<double> &obstacles) {
    if (obstacles.ndim() == 1 && obstacles.shape(0) == 0) {
        return {};
    }
    if (obstacles.ndim() != 2 || obstacles.shape(1) != 2) {
        throw std::invalid_argument("discrete obstacles must have shape (M, 2)");
    }
    const auto m = static_cast<std::size_t>(obstacles.shape(0));
    std::vector<double> out(m * 2);
    auto u = obstacles.unchecked<2>();
    for (std::size_t i = 0; i < m; ++i) {
        out[i * 2 + 0] = u(static_cast<py::ssize_t>(i), 0);
        out[i * 2 + 1] = u(static_cast<py::ssize_t>(i), 1);
    }
    return out;
}

// Load dangerous-area centres: shape (K, 2), (0, 2), or (0,) 1-D for empty.
// Returns flat (cx, cy) pairs.
std::vector<double> load_dangerous_areas(const py::array_t<double> &areas) {
    if (areas.ndim() == 1 && areas.shape(0) == 0) {
        return {};
    }
    if (areas.ndim() != 2 || areas.shape(1) != 2) {
        throw std::invalid_argument("dangerous_areas must have shape (K, 2)");
    }
    const auto k = static_cast<std::size_t>(areas.shape(0));
    std::vector<double> out(k * 2);
    auto u = areas.unchecked<2>();
    for (std::size_t i = 0; i < k; ++i) {
        out[i * 2 + 0] = u(static_cast<py::ssize_t>(i), 0);
        out[i * 2 + 1] = u(static_cast<py::ssize_t>(i), 1);
    }
    return out;
}

// Returns true if (px, py) is within dangerous_area_radius_sq of any centre.
static inline bool point_in_dangerous_areas(double px, double py,
                                            const std::vector<double> &areas,
                                            double radius_sq) noexcept {
    const std::size_t k = areas.size() / 2;
    for (std::size_t i = 0; i < k; ++i) {
        const double dx = px - areas[i * 2 + 0];
        const double dy = py - areas[i * 2 + 1];
        if (dx * dx + dy * dy <= radius_sq) {
            return true;
        }
    }
    return false;
}

// Returns true if (px, py) is within obstacle_radius_sq of any obstacle.
static inline bool discrete_collides(double px, double py,
                                     const std::vector<double> &obstacles,
                                     double obstacle_radius_sq) noexcept {
    const std::size_t n = obstacles.size() / 2;
    for (std::size_t i = 0; i < n; ++i) {
        const double ddx = px - obstacles[i * 2 + 0];
        const double ddy = py - obstacles[i * 2 + 1];
        if (ddx * ddx + ddy * ddy <= obstacle_radius_sq) {
            return true;
        }
    }
    return false;
}

// Reward-variant codes shared between the standalone batch kernels and the
// inline rollout reward paths. Mirrors the Python ``RewardModelType`` enum.
constexpr int kRewardVariantConstantHazardPenalty = 0;
constexpr int kRewardVariantHighVar = 1;
constexpr int kRewardVariantDistanceDecayedHazardPenalty = 2;

// Squared Euclidean distance from (px, py) to the nearest dangerous-area
// centre. Used by the Decaying variant to evaluate the exponential
// hit-probability without materialising sqrt twice.
static inline double dangerous_min_dist_sq(double px, double py_,
                                           const std::vector<double> &areas) noexcept {
    const std::size_t k = areas.size() / 2;
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < k; ++i) {
        const double dx = px - areas[i * 2 + 0];
        const double dy = py_ - areas[i * 2 + 1];
        const double d_sq = dx * dx + dy * dy;
        if (d_sq < best) {
            best = d_sq;
        }
    }
    return best;
}

// Per-row dangerous-area reward contribution. Branches on ``variant_code``:
// CONSTANT_HAZARD_PENALTY (with optional Bernoulli), ZERO_MEAN_HAZARD_SHOCK (``±penalty``
// 50/50 when in zone) and DISTANCE_DECAYED_HAZARD_PENALTY (penalty fires with
// ``exp(-min_dist / penalty_decay)`` over every row, no radius cutoff).
static inline double dangerous_contribution(int variant_code, bool in_zone,
                                            double penalty, double hit_prob,
                                            double min_dist_sq, double penalty_decay,
                                            pomdp_native::RNGState &rng) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    if (variant_code == kRewardVariantConstantHazardPenalty) {
        if (!in_zone) {
            return 0.0;
        }
        if (hit_prob >= 1.0 || uni(rng.engine()) < hit_prob) {
            return penalty;
        }
        return 0.0;
    }
    if (variant_code == kRewardVariantHighVar) {
        if (!in_zone) {
            return 0.0;
        }
        return uni(rng.engine()) < 0.5 ? penalty : -penalty;
    }
    // Decaying: every row, prob exp(-sqrt(min_dist_sq) / penalty_decay).
    const double u = uni(rng.engine());
    if (u <= 0.0) {
        return penalty;
    }
    const double log_u = std::log(u);
    if (penalty_decay * penalty_decay * log_u * log_u > min_dist_sq) {
        return penalty;
    }
    return 0.0;
}

// Reward configuration for the discrete rollout kernel. Mirrors the
// parameters consumed by ``push_reward_batch`` so the inline rollout reward
// stays bit-for-bit aligned with the standalone batch kernel (sample-mean
// parity across all three :class:`RewardModelType` variants).
struct DiscreteStepRewardConfig {
    double obstacle_penalty;
    double obstacle_hit_probability;
    double dangerous_area_radius_sq;
    double dangerous_area_penalty;
    double dangerous_area_hit_probability;
    int reward_variant_code;
    double penalty_decay;
};

// Per-step reward using the REALISED post-action robot position. Mirrors
// ``DiscretePushRewardModel.compute_reward`` (and its high-variance /
// decaying subclasses) so the rollout kernel matches the standalone
// ``push_reward_batch`` for every variant.
static double discrete_step_reward(double nrx, double nry, double nox, double noy,
                                   double tx, double ty,
                                   const std::vector<double> &obstacles,
                                   double obstacle_radius_sq,
                                   const std::vector<double> &dangerous_areas,
                                   const DiscreteStepRewardConfig &cfg,
                                   pomdp_native::RNGState &rng) {
    const double rdx = nox - tx;
    const double rdy = noy - ty;
    const double dist_to_target = std::sqrt(rdx * rdx + rdy * rdy);
    double reward = -dist_to_target;
    if (dist_to_target < 0.5) {
        reward += 100.0;
    }
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    if (!obstacles.empty() && discrete_collides(nrx, nry, obstacles, obstacle_radius_sq)) {
        if (cfg.obstacle_hit_probability >= 1.0 ||
            uni(rng.engine()) < cfg.obstacle_hit_probability) {
            reward += cfg.obstacle_penalty;
        }
    }
    if (!dangerous_areas.empty()) {
        const bool in_zone = point_in_dangerous_areas(nrx, nry, dangerous_areas,
                                                     cfg.dangerous_area_radius_sq);
        const double min_sq = (cfg.reward_variant_code == kRewardVariantDistanceDecayedHazardPenalty)
                                  ? dangerous_min_dist_sq(nrx, nry, dangerous_areas)
                                  : 0.0;
        reward += dangerous_contribution(cfg.reward_variant_code, in_zone,
                                         cfg.dangerous_area_penalty,
                                         cfg.dangerous_area_hit_probability,
                                         min_sq, cfg.penalty_decay, rng);
    }
    return reward;
}

// Apply one discrete Push step, writing the next state into the 6-element
// output array.  Returns the immediate reward.
//
// action_idx must be in [0, 3].  Uses the same semantics as
// PushPOMDP._sample_one_next_state and PushPOMDP._reward_from_next_state.
static double discrete_step(const double *state, int action_idx, double *next_state,
                             double grid_max, double push_threshold_sq,
                             double push_scale, double obstacle_radius_sq,
                             const std::vector<double> &obstacles,
                             const std::vector<double> &dangerous_areas,
                             const DiscreteStepRewardConfig &reward_cfg,
                             pomdp_native::RNGState &rng,
                             double transition_error_prob) {
    // Resolve action error ---------------------------------------------------
    int actual_idx = action_idx;
    if (transition_error_prob > 0.0) {
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        if (uni(rng.engine()) < transition_error_prob) {
            // Pick uniformly from the three other actions.
            std::uniform_int_distribution<int> pick(0, 2);
            actual_idx = kErrorActions[static_cast<std::size_t>(action_idx)]
                                      [static_cast<std::size_t>(pick(rng.engine()))];
        }
    }

    const int dx = kDiscreteDXY[static_cast<std::size_t>(actual_idx)][0];
    const int dy = kDiscreteDXY[static_cast<std::size_t>(actual_idx)][1];

    const double rx = state[0];
    const double ry = state[1];
    const double ox = state[2];
    const double oy = state[3];
    const double tx = state[4];
    const double ty = state[5];

    // Robot movement: intended position; blocked if colliding with obstacle. --
    const double irx = rx + static_cast<double>(dx);
    const double iry = ry + static_cast<double>(dy);
    double nrx;
    double nry;
    if (!obstacles.empty() && discrete_collides(irx, iry, obstacles, obstacle_radius_sq)) {
        nrx = rx;
        nry = ry;
    } else {
        nrx = irx;
        nry = iry;
    }

    // Object push: if robot is within push_threshold of object. ---------------
    const double ddx = nrx - ox;
    const double ddy = nry - oy;
    const double dist_sq = ddx * ddx + ddy * ddy;
    double nox;
    double noy;
    if (dist_sq < push_threshold_sq) {
        const double iox = ox + static_cast<double>(dx) * push_scale;
        const double ioy = oy + static_cast<double>(dy) * push_scale;
        if (!obstacles.empty() && discrete_collides(iox, ioy, obstacles, obstacle_radius_sq)) {
            nox = ox;
            noy = oy;
        } else {
            nox = iox;
            noy = ioy;
        }
    } else {
        nox = ox;
        noy = oy;
    }

    // Clamp to grid. ----------------------------------------------------------
    nrx = std::max(0.0, std::min(nrx, grid_max));
    nry = std::max(0.0, std::min(nry, grid_max));
    nox = std::max(0.0, std::min(nox, grid_max));
    noy = std::max(0.0, std::min(noy, grid_max));

    next_state[0] = nrx;
    next_state[1] = nry;
    next_state[2] = nox;
    next_state[3] = noy;
    next_state[4] = tx;
    next_state[5] = ty;

    // Reward (uses realised post-action robot position; matches the Python
    // ``DiscretePushRewardModel`` family and the variant-aware
    // ``push_reward_batch`` standalone kernel).
    return discrete_step_reward(nrx, nry, nox, noy, tx, ty, obstacles,
                                obstacle_radius_sq, dangerous_areas, reward_cfg,
                                rng);
}

// Is-terminal: (obj_x - tgt_x)^2 + (obj_y - tgt_y)^2 < 0.25.
static inline bool discrete_is_terminal(const double *state) noexcept {
    const double dx = state[2] - state[4];
    const double dy = state[3] - state[5];
    return (dx * dx + dy * dy) < 0.25;
}

// ── Discrete Push deterministic-transition kernel (per-action cache) ───────
//
// Mirrors PushPOMDP._compute_next_state_for_action exactly: a closed-form
// next-state computation given a (state, action) pair, with no RNG involved.
// The kernel is constructed once per action label (the resolved (dx, dy) is
// frozen at ctor time); the caller flips ``state_`` via set_state on each
// call and pulls compute_next_state(). compute_next_state_for_action(other)
// supports the error-action branch in transition_log_probability where the
// same kernel is used to evaluate alternative actions without rebuilding.

class PushDiscreteTransitionCpp {
  public:
    PushDiscreteTransitionCpp(
        py::array_t<double, py::array::c_style | py::array::forcecast> state,
        py::array_t<double, py::array::c_style | py::array::forcecast> action_dxdy,
        double grid_size, double push_threshold, double friction_coefficient,
        py::array_t<double, py::array::c_style | py::array::forcecast> obstacles_flat,
        int n_obstacles, double obstacle_radius)
        : grid_size_(grid_size),
          push_threshold_(push_threshold),
          friction_coefficient_(friction_coefficient),
          n_obstacles_(n_obstacles),
          obstacle_radius_(obstacle_radius),
          state_(kPushStateDim, 0.0) {
        load_action_(action_dxdy, action_dxdy_);
        load_obstacles_flat_(obstacles_flat);
        load_state_(state);
    }

    void set_state(
        py::array_t<double, py::array::c_style | py::array::forcecast> state) {
        load_state_(state);
    }

    py::array_t<double> compute_next_state() const {
        auto out = py::array_t<double>(static_cast<py::ssize_t>(kPushStateDim));
        compute_next_state_into_(action_dxdy_, out.mutable_data());
        return out;
    }

    py::array_t<double> compute_next_state_for_action(
        py::array_t<double, py::array::c_style | py::array::forcecast> action_dxdy) const {
        std::array<double, 2> override_action{};
        load_action_(action_dxdy, override_action);
        auto out = py::array_t<double>(static_cast<py::ssize_t>(kPushStateDim));
        compute_next_state_into_(override_action, out.mutable_data());
        return out;
    }

  private:
    static void load_action_(
        const py::array_t<double, py::array::c_style | py::array::forcecast> &action,
        std::array<double, 2> &dst) {
        if (action.ndim() != 1 || action.shape(0) != 2) {
            throw std::invalid_argument("action_dxdy must have shape (2,)");
        }
        auto av = action.unchecked<1>();
        dst[0] = av(0);
        dst[1] = av(1);
    }

    void load_obstacles_flat_(
        const py::array_t<double, py::array::c_style | py::array::forcecast> &obstacles_flat) {
        if (n_obstacles_ < 0) {
            throw std::invalid_argument("n_obstacles must be non-negative");
        }
        const auto expected = static_cast<std::size_t>(n_obstacles_) * 2;
        if (obstacles_flat.ndim() != 1 ||
            static_cast<std::size_t>(obstacles_flat.shape(0)) != expected) {
            throw std::invalid_argument("obstacles_flat must have shape (n_obstacles*2,)");
        }
        obstacles_flat_.assign(obstacles_flat.data(), obstacles_flat.data() + expected);
    }

    void load_state_(
        const py::array_t<double, py::array::c_style | py::array::forcecast> &state) {
        if (state.ndim() != 1 ||
            static_cast<std::size_t>(state.shape(0)) != kPushStateDim) {
            throw std::invalid_argument("state must have shape (6,)");
        }
        auto sv = state.unchecked<1>();
        for (std::size_t d = 0; d < kPushStateDim; ++d) {
            state_[d] = sv(static_cast<py::ssize_t>(d));
        }
    }

    bool collides_(double px, double py_) const noexcept {
        if (n_obstacles_ <= 0) {
            return false;
        }
        const double r_sq = obstacle_radius_ * obstacle_radius_;
        for (int i = 0; i < n_obstacles_; ++i) {
            const double dx = px - obstacles_flat_[static_cast<std::size_t>(i) * 2 + 0];
            const double dy = py_ - obstacles_flat_[static_cast<std::size_t>(i) * 2 + 1];
            if (dx * dx + dy * dy <= r_sq) {
                return true;
            }
        }
        return false;
    }

    void compute_next_state_into_(const std::array<double, 2> &dxy, double *out) const {
        const double dx = dxy[0];
        const double dy = dxy[1];

        const double rx = state_[0];
        const double ry = state_[1];
        const double ox = state_[2];
        const double oy = state_[3];
        const double tx = state_[4];
        const double ty = state_[5];

        // Intended robot move; obstacle collision blocks movement.
        const double irx = rx + dx;
        const double iry = ry + dy;
        double nrx;
        double nry;
        if (collides_(irx, iry)) {
            nrx = rx;
            nry = ry;
        } else {
            nrx = irx;
            nry = iry;
        }

        // Object push if robot within push_threshold of object.
        const double ddx = nrx - ox;
        const double ddy = nry - oy;
        const double dist_sq = ddx * ddx + ddy * ddy;
        const double thr_sq = push_threshold_ * push_threshold_;
        double nox;
        double noy;
        if (dist_sq < thr_sq) {
            const double push_scale = 1.0 - friction_coefficient_;
            const double iox = ox + dx * push_scale;
            const double ioy = oy + dy * push_scale;
            if (collides_(iox, ioy)) {
                nox = ox;
                noy = oy;
            } else {
                nox = iox;
                noy = ioy;
            }
        } else {
            nox = ox;
            noy = oy;
        }

        const double gmax = grid_size_ - 1.0;
        out[0] = std::max(0.0, std::min(nrx, gmax));
        out[1] = std::max(0.0, std::min(nry, gmax));
        out[2] = std::max(0.0, std::min(nox, gmax));
        out[3] = std::max(0.0, std::min(noy, gmax));
        out[4] = tx;
        out[5] = ty;
    }

    std::array<double, 2> action_dxdy_{};
    double grid_size_;
    double push_threshold_;
    double friction_coefficient_;
    std::vector<double> obstacles_flat_;
    int n_obstacles_;
    double obstacle_radius_;
    std::vector<double> state_;
};

// simulate_rollout_discrete: run a full rollout in one C++ frame.
//
// Parameters:
//   state         – initial 6-D state (numpy 1-D float64 array, length 6).
//   action_indices – pre-drawn action indices from Python (int32 / int64),
//                   length max_depth; only the first (max_depth - depth)
//                   entries that are actually consumed matter.
//   max_depth     – rollout horizon (same semantics as the Python override).
//   depth         – depth already consumed before this rollout (usually 0).
//   discount      – discount factor γ.
//   grid_size     – grid size (integer ≥ 1); grid_max = grid_size - 1.
//   push_threshold – scalar push threshold.
//   friction_coefficient – friction; push_scale = 1 - friction.
//   obstacles     – (M, 2) float64 array of obstacle centres; shape (0,) or
//                   (0, 2) for no obstacles.
//   obstacle_radius – obstacle radius for point-in-circle tests.
//   obstacle_penalty – negative penalty added to reward on robot collision.
//   transition_error_prob – probability of picking a random other action.
//
// Returns:
//   Discounted sum of immediate rewards (scalar float).
double simulate_rollout_discrete(
    py::array_t<double, py::array::c_style | py::array::forcecast> state_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> action_indices,
    int max_depth, int depth, double discount, double grid_size,
    double push_threshold, double friction_coefficient,
    py::array_t<double, py::array::c_style | py::array::forcecast> obstacles_arr,
    double obstacle_radius, double obstacle_penalty,
    py::array_t<double, py::array::c_style | py::array::forcecast> dangerous_areas_arr,
    double dangerous_area_radius, double dangerous_area_penalty,
    double transition_error_prob, double obstacle_hit_probability,
    double dangerous_area_hit_probability, int reward_variant_code,
    double penalty_decay) {
    if (state_arr.ndim() != 1 || state_arr.shape(0) != 6) {
        throw std::invalid_argument("state must be a 1-D array of length 6");
    }
    if (action_indices.ndim() != 1) {
        throw std::invalid_argument("action_indices must be 1-D");
    }
    const int n_actions = static_cast<int>(action_indices.shape(0));
    auto state_view = state_arr.unchecked<1>();
    auto idx_view = action_indices.unchecked<1>();

    // Pre-compute derived constants.
    const double grid_max = grid_size - 1.0;
    const double push_threshold_sq = push_threshold * push_threshold;
    const double push_scale = 1.0 - friction_coefficient;
    const double obstacle_radius_sq = obstacle_radius * obstacle_radius;
    const double dangerous_area_radius_sq = dangerous_area_radius * dangerous_area_radius;

    // Load obstacles (M, 2) flat.
    std::vector<double> obstacles;
    if (!(obstacles_arr.ndim() == 1 && obstacles_arr.shape(0) == 0)) {
        obstacles = load_discrete_obstacles(obstacles_arr);
    }

    // Load dangerous areas (K, 2) flat.
    std::vector<double> dangerous_areas = load_dangerous_areas(dangerous_areas_arr);

    // Copy initial state into a mutable buffer.
    double cur[6];  // NOLINT(modernize-avoid-c-arrays)
    double nxt[6];  // NOLINT(modernize-avoid-c-arrays)
    for (int d = 0; d < 6; ++d) {
        cur[d] = state_view(d);
    }

    pomdp_native::RNGState &rng = pomdp_native::default_rng();
    double total = 0.0;
    double gamma_power = 1.0;
    int action_cursor = 0;

    const DiscreteStepRewardConfig reward_cfg{
        obstacle_penalty,
        obstacle_hit_probability,
        dangerous_area_radius_sq,
        dangerous_area_penalty,
        dangerous_area_hit_probability,
        reward_variant_code,
        penalty_decay,
    };

    while (depth < max_depth && !discrete_is_terminal(cur)) {
        if (action_cursor >= n_actions) {
            break;  // Ran out of pre-drawn actions; stop cleanly.
        }
        const int action_idx = static_cast<int>(idx_view(action_cursor));
        ++action_cursor;

        const double r = discrete_step(cur, action_idx, nxt, grid_max,
                                       push_threshold_sq, push_scale,
                                       obstacle_radius_sq, obstacles,
                                       dangerous_areas, reward_cfg,
                                       rng, transition_error_prob);
        total += gamma_power * r;
        gamma_power *= discount;
        std::copy(nxt, nxt + 6, cur);
        ++depth;
    }
    return total;
}

// ── ContinuousPush native rollout (added by perf agent) ──
//
// cont_simulate_rollout walks one random rollout from initial_state entirely
// inside C++:
//   1. Terminal check: ||obj - target||_2 < 0.5
//   2. Pick action from pre-drawn action_indices
//   3. Sample next state (same geometry as ContinuousPushTransitionCpp)
//   4. Reward = -dist(next_obj, target) + 100*(dist<0.5)
//              + obstacle_penalty if state[:2]+action circles any obstacle AABB
//   5. Accumulate gamma^step * reward
//
// Static helpers replicate the private member functions of
// ContinuousPushTransitionCpp without requiring access to class internals.

static bool cont_circle_aabb_overlap(const double *pos, double robot_radius,
                                     const double *wall) noexcept {
    const double cx = wall[0];
    const double cy = wall[1];
    const double hx = wall[2];
    const double hy = wall[3];
    const double closest_x = std::clamp(pos[0], cx - hx, cx + hx);
    const double closest_y = std::clamp(pos[1], cy - hy, cy + hy);
    const double ddx = pos[0] - closest_x;
    const double ddy = pos[1] - closest_y;
    return (ddx * ddx + ddy * ddy) < (robot_radius * robot_radius);
}

static bool cont_is_terminal(const double *state) noexcept {
    const double dx = state[2] - state[4];
    const double dy = state[3] - state[5];
    return (dx * dx + dy * dy) < 0.25;  // 0.5^2
}

static void cont_resolve_single_circle_wall(double *pos, const double *wall,
                                            double robot_radius) noexcept {
    const double cx = wall[0];
    const double cy = wall[1];
    const double hx = wall[2];
    const double hy = wall[3];
    const double closest_x = std::clamp(pos[0], cx - hx, cx + hx);
    const double closest_y = std::clamp(pos[1], cy - hy, cy + hy);
    const double dx = pos[0] - closest_x;
    const double dy = pos[1] - closest_y;
    const double dist_sq = dx * dx + dy * dy;
    const double r_sq = robot_radius * robot_radius;
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
        if (min_pen == pen_left) {
            pos[0] = cx - hx - robot_radius;
        } else if (min_pen == pen_right) {
            pos[0] = cx + hx + robot_radius;
        } else if (min_pen == pen_down) {
            pos[1] = cy - hy - robot_radius;
        } else {
            pos[1] = cy + hy + robot_radius;
        }
    } else {
        const double overlap = robot_radius - dist;
        pos[0] += (dx / dist) * overlap;
        pos[1] += (dy / dist) * overlap;
    }
}

static bool cont_point_inside_aabb(const double *point, const double *wall) noexcept {
    const double cx = wall[0];
    const double cy = wall[1];
    const double hx = wall[2];
    const double hy = wall[3];
    return (cx - hx) <= point[0] && point[0] <= (cx + hx) &&
           (cy - hy) <= point[1] && point[1] <= (cy + hy);
}

// Sample one next-state row from (state_row, action, noise) into out_row.
static void cont_sample_next_state_row(
    const double *state_row, const double *action, double grid_size,
    double push_threshold, double friction_coefficient, double max_push,
    double robot_radius, const std::vector<double> &obs_flat,
    std::size_t n_obs, const pomdp_native::GaussianND<kNoiseDim> &noise,
    double *out_row, pomdp_native::RNGState &rng) {

    double robot_mean[kNoiseDim];  // NOLINT(modernize-avoid-c-arrays)
    robot_mean[0] = state_row[0] + action[0];
    robot_mean[1] = state_row[1] + action[1];
    double robot_sample[kNoiseDim];  // NOLINT(modernize-avoid-c-arrays)
    noise.sample_into(robot_sample, robot_mean, rng);

    for (std::size_t i = 0; i < n_obs; ++i) {
        cont_resolve_single_circle_wall(robot_sample, &obs_flat[i * 4], robot_radius);
    }
    const double lo = robot_radius;
    const double hi = grid_size - 1.0 - robot_radius;
    robot_sample[0] = std::clamp(robot_sample[0], lo, hi);
    robot_sample[1] = std::clamp(robot_sample[1], lo, hi);

    double object_out[kNoiseDim];  // NOLINT(modernize-avoid-c-arrays)
    object_out[0] = state_row[2];
    object_out[1] = state_row[3];

    const double ddx = robot_sample[0] - state_row[2];
    const double ddy = robot_sample[1] - state_row[3];
    const double dist_to_obj = std::sqrt(ddx * ddx + ddy * ddy);
    if (dist_to_obj < push_threshold) {
        const double action_norm = std::sqrt(action[0] * action[0] + action[1] * action[1]);
        if (action_norm >= 1e-12) {
            const double dir_x = action[0] / action_norm;
            const double dir_y = action[1] / action_norm;
            const double force_mag = std::min(action_norm, max_push) * (1.0 - friction_coefficient);
            double intended[kNoiseDim];  // NOLINT(modernize-avoid-c-arrays)
            intended[0] = state_row[2] + dir_x * force_mag;
            intended[1] = state_row[3] + dir_y * force_mag;
            bool blocked = false;
            for (std::size_t i = 0; i < n_obs; ++i) {
                if (cont_point_inside_aabb(intended, &obs_flat[i * 4])) {
                    blocked = true;
                    break;
                }
            }
            if (!blocked) {
                object_out[0] = std::clamp(intended[0], 0.0, grid_size - 1.0);
                object_out[1] = std::clamp(intended[1], 0.0, grid_size - 1.0);
            }
        }
    }

    out_row[0] = robot_sample[0];
    out_row[1] = robot_sample[1];
    out_row[2] = object_out[0];
    out_row[3] = object_out[1];
    out_row[4] = state_row[4];  // target unchanged
    out_row[5] = state_row[5];
}

// Per-step reward for the continuous rollout using the REALISED post-action
// robot position (next_state[:2]). Mirrors ``ContinuousPushRewardModel``
// (and its high-variance / decaying subclasses) so the rollout output stays
// in sample-mean parity with ``cont_push_reward_batch`` across every
// :class:`RewardModelType` variant.
static double cont_step_reward(const double *next_state,
                               const std::vector<double> &obs_flat,
                               std::size_t n_obs, double robot_radius,
                               const std::vector<double> &dangerous_flat,
                               const DiscreteStepRewardConfig &cfg,
                               pomdp_native::RNGState &rng) {
    const double rd = next_state[2] - next_state[4];
    const double rd2 = next_state[3] - next_state[5];
    const double dist_to_target = std::sqrt(rd * rd + rd2 * rd2);
    double reward = -dist_to_target;
    if (dist_to_target < 0.5) {
        reward += 100.0;
    }
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const double robot_after[2] = {  // NOLINT(modernize-avoid-c-arrays)
        next_state[0], next_state[1]};
    if (n_obs > 0) {
        bool collide = false;
        for (std::size_t i = 0; i < n_obs; ++i) {
            if (cont_circle_aabb_overlap(robot_after, robot_radius, &obs_flat[i * 4])) {
                collide = true;
                break;
            }
        }
        if (collide) {
            if (cfg.obstacle_hit_probability >= 1.0 ||
                uni(rng.engine()) < cfg.obstacle_hit_probability) {
                reward += cfg.obstacle_penalty;
            }
        }
    }
    if (!dangerous_flat.empty()) {
        const bool in_zone = point_in_dangerous_areas(robot_after[0], robot_after[1],
                                                     dangerous_flat,
                                                     cfg.dangerous_area_radius_sq);
        const double min_sq = (cfg.reward_variant_code == kRewardVariantDistanceDecayedHazardPenalty)
                                  ? dangerous_min_dist_sq(robot_after[0], robot_after[1],
                                                          dangerous_flat)
                                  : 0.0;
        reward += dangerous_contribution(cfg.reward_variant_code, in_zone,
                                         cfg.dangerous_area_penalty,
                                         cfg.dangerous_area_hit_probability,
                                         min_sq, cfg.penalty_decay, rng);
    }
    return reward;
}

double cont_simulate_rollout(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &initial_state,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &action_array,
    const py::array_t<int, py::array::c_style | py::array::forcecast> &action_indices,
    int max_depth, int start_depth, double discount_factor,
    double grid_size, double push_threshold, double friction_coefficient,
    double max_push, double robot_radius, double obstacle_penalty,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &obstacles,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dangerous_areas,
    double dangerous_area_radius, double dangerous_area_penalty,
    const py::array_t<double> &covariance,
    double obstacle_hit_probability, double dangerous_area_hit_probability,
    int reward_variant_code, double penalty_decay) {

    if (initial_state.ndim() != 1 ||
        static_cast<std::size_t>(initial_state.shape(0)) != kPushStateDim) {
        throw std::invalid_argument("initial_state must have shape (6,)");
    }
    if (action_array.ndim() != 2 ||
        static_cast<std::size_t>(action_array.shape(1)) != kNoiseDim) {
        throw std::invalid_argument("action_array must have shape (n_actions, 2)");
    }
    if (action_indices.ndim() != 1) {
        throw std::invalid_argument("action_indices must be 1-D");
    }
    if (obstacles.ndim() != 2 || obstacles.shape(1) != 4) {
        throw std::invalid_argument("obstacles must have shape (M, 4)");
    }

    const int n_actions = static_cast<int>(action_array.shape(0));
    const int n_indices = static_cast<int>(action_indices.shape(0));
    const auto n_obs = static_cast<std::size_t>(obstacles.shape(0));

    std::vector<double> obs_flat(n_obs * 4);
    {
        auto obs_v = obstacles.unchecked<2>();
        for (std::size_t i = 0; i < n_obs; ++i) {
            obs_flat[i * 4 + 0] = obs_v(static_cast<py::ssize_t>(i), 0);
            obs_flat[i * 4 + 1] = obs_v(static_cast<py::ssize_t>(i), 1);
            obs_flat[i * 4 + 2] = obs_v(static_cast<py::ssize_t>(i), 2);
            obs_flat[i * 4 + 3] = obs_v(static_cast<py::ssize_t>(i), 3);
        }
    }

    std::vector<double> dangerous_flat = load_dangerous_areas(dangerous_areas);
    const double dangerous_area_radius_sq = dangerous_area_radius * dangerous_area_radius;

    const auto noise = pomdp_native::GaussianND<kNoiseDim>::from_covariance(covariance);

    auto is_view = initial_state.unchecked<1>();
    double state[kPushStateDim];      // NOLINT(modernize-avoid-c-arrays)
    double next_state[kPushStateDim]; // NOLINT(modernize-avoid-c-arrays)
    for (std::size_t d = 0; d < kPushStateDim; ++d) {
        state[d] = is_view(static_cast<py::ssize_t>(d));
    }

    auto aa_view = action_array.unchecked<2>();
    auto ai_view = action_indices.unchecked<1>();
    pomdp_native::RNGState &rng = pomdp_native::default_rng();

    double total = 0.0;
    double gamma_power = 1.0;
    int depth = start_depth;

    const DiscreteStepRewardConfig reward_cfg{
        obstacle_penalty,
        obstacle_hit_probability,
        dangerous_area_radius_sq,
        dangerous_area_penalty,
        dangerous_area_hit_probability,
        reward_variant_code,
        penalty_decay,
    };

    while (depth < max_depth) {
        if (cont_is_terminal(state)) {
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
        const double action[kNoiseDim] = {  // NOLINT(modernize-avoid-c-arrays)
            aa_view(static_cast<py::ssize_t>(ai), 0),
            aa_view(static_cast<py::ssize_t>(ai), 1)};

        cont_sample_next_state_row(state, action, grid_size, push_threshold,
                                   friction_coefficient, max_push, robot_radius,
                                   obs_flat, n_obs, noise, next_state, rng);

        const double reward = cont_step_reward(next_state, obs_flat, n_obs,
                                               robot_radius, dangerous_flat,
                                               reward_cfg, rng);

        total += gamma_power * reward;
        gamma_power *= discount_factor;
        for (std::size_t d = 0; d < kPushStateDim; ++d) {
            state[d] = next_state[d];
        }
        ++depth;
    }

    return total;
}

// ── Discrete Push belief-update batch kernels ───────────────────────────────
//
// Native ports of PushVectorizedUpdater._transition_for_action and the
// observation log-likelihood path. Replaces the per-particle Python loops
// that dominate WeightedParticleBelief._update_weights when running PFT-DPW
// or POMCPOW on the discrete Push environment.
//
// Mirrors the LaserTag pattern (belief_batch_transition_discrete /
// belief_batch_obs_log_likelihood_discrete): single C++ round-trip per
// belief update, independent C++ RNG for the per-particle action-error coin
// flip (deterministic-otherwise transition; obs likelihood has no RNG).

// Apply one discrete Push action to all N particles deterministically.
// Reads particle rows in_data[i*6 .. i*6+5] and writes next-state rows to
// out_data with identical layout.
inline void discrete_apply_action_batch(int action_idx, std::size_t n,
                                         const double *in_data, double *out_data,
                                         double grid_max, double push_threshold_sq,
                                         double push_scale, double obstacle_radius_sq,
                                         const std::vector<double> &obstacles) noexcept {
    const int dx = kDiscreteDXY[static_cast<std::size_t>(action_idx)][0];
    const int dy = kDiscreteDXY[static_cast<std::size_t>(action_idx)][1];
    const bool has_obs = !obstacles.empty();
    for (std::size_t i = 0; i < n; ++i) {
        const double *src = in_data + i * 6;
        double *dst = out_data + i * 6;
        const double rx = src[0];
        const double ry = src[1];
        const double ox = src[2];
        const double oy = src[3];

        // Robot movement: blocked by obstacle.
        const double irx = rx + static_cast<double>(dx);
        const double iry = ry + static_cast<double>(dy);
        double nrx = irx;
        double nry = iry;
        if (has_obs && discrete_collides(irx, iry, obstacles, obstacle_radius_sq)) {
            nrx = rx;
            nry = ry;
        }

        // Object push: only if robot within push_threshold of object.
        const double ddx = nrx - ox;
        const double ddy = nry - oy;
        double nox = ox;
        double noy = oy;
        if ((ddx * ddx + ddy * ddy) < push_threshold_sq) {
            const double iox = ox + static_cast<double>(dx) * push_scale;
            const double ioy = oy + static_cast<double>(dy) * push_scale;
            if (!(has_obs && discrete_collides(iox, ioy, obstacles, obstacle_radius_sq))) {
                nox = iox;
                noy = ioy;
            }
        }

        // Clamp to grid.
        dst[0] = std::clamp(nrx, 0.0, grid_max);
        dst[1] = std::clamp(nry, 0.0, grid_max);
        dst[2] = std::clamp(nox, 0.0, grid_max);
        dst[3] = std::clamp(noy, 0.0, grid_max);
        // Target unchanged.
        dst[4] = src[4];
        dst[5] = src[5];
    }
}

// Native port of PushVectorizedUpdater.batch_transition.
//
// Parameters:
//   particles            : (N, 6) float64 input particles
//   action_idx           : intended action (0..3)
//   transition_error_prob: per-particle probability of executing a random
//                          other action instead of action_idx
//   obstacles_arr        : (M, 2) float64 obstacle centres, or empty (0,) /
//                          (0, 2) array for no obstacles
//   obstacle_radius      : collision radius for each obstacle
//   grid_size            : grid size; positions clipped to [0, grid_size-1]
//   push_threshold       : maximum robot-object distance for a push
//   friction_coefficient : friction reducing push force (push_scale = 1 - f)
//
// Returns: (N, 6) float64 array of next particles.
py::array_t<double> belief_batch_transition_discrete(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &particles,
    int action_idx, double transition_error_prob,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &obstacles_arr,
    double obstacle_radius, double grid_size, double push_threshold,
    double friction_coefficient) {
    if (particles.ndim() != 2 || particles.shape(1) != 6) {
        throw std::invalid_argument("particles must have shape (N, 6)");
    }
    if (action_idx < 0 || action_idx > 3) {
        throw std::invalid_argument("action_idx must be in {0,1,2,3}");
    }
    const auto n = static_cast<std::size_t>(particles.shape(0));
    auto out = py::array_t<double>({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(6)});
    if (n == 0) {
        return out;
    }

    // Normalize obstacles: accept (0,), (0,2), (M,2).
    std::vector<double> obstacles;
    if (!(obstacles_arr.ndim() == 1 && obstacles_arr.shape(0) == 0)) {
        obstacles = load_discrete_obstacles(obstacles_arr);
    }

    const double grid_max = grid_size - 1.0;
    const double push_threshold_sq = push_threshold * push_threshold;
    const double push_scale = 1.0 - friction_coefficient;
    const double obstacle_radius_sq = obstacle_radius * obstacle_radius;

    const double *in_data = particles.data();
    double *out_data = out.mutable_data();

    // Fast path: no error → single dispatch.
    if (transition_error_prob <= 0.0) {
        discrete_apply_action_batch(action_idx, n, in_data, out_data, grid_max,
                                     push_threshold_sq, push_scale,
                                     obstacle_radius_sq, obstacles);
        return out;
    }

    // Error path: compute candidates for all 4 actions, then per-particle
    // choose intended or one of the three other actions. Mirrors the Python
    // ``_batch_transition_with_error`` semantics.
    std::array<std::vector<double>, 4> candidates;
    for (int a = 0; a < 4; ++a) {
        candidates[static_cast<std::size_t>(a)].assign(n * 6, 0.0);
        discrete_apply_action_batch(a, n, in_data,
                                     candidates[static_cast<std::size_t>(a)].data(),
                                     grid_max, push_threshold_sq, push_scale,
                                     obstacle_radius_sq, obstacles);
    }

    pomdp_native::RNGState &rng = pomdp_native::default_rng();
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::uniform_int_distribution<int> err_choice(0, 2);
    const auto &err_table = kErrorActions[static_cast<std::size_t>(action_idx)];
    for (std::size_t i = 0; i < n; ++i) {
        const double err_u = uniform(rng.engine());
        int chosen = action_idx;
        if (err_u < transition_error_prob) {
            chosen = err_table[static_cast<std::size_t>(err_choice(rng.engine()))];
        }
        const double *src = candidates[static_cast<std::size_t>(chosen)].data() + i * 6;
        for (std::size_t d = 0; d < 6; ++d) {
            out_data[i * 6 + d] = src[d];
        }
    }
    return out;
}

// Native port of PushVectorizedUpdater.batch_observation_log_likelihood.
//
// Closed-form 2-D isotropic Gaussian log-pdf evaluated on the object-position
// slice (cols 2..3) of each particle. No RNG; bit-for-bit equivalent to the
// Python ``CovarianceParameterizedMultivariateNormal.log_pdf`` for diag(σ²).
//
// Parameters:
//   next_particles    : (N, 6) float64 input particles (post-transition)
//   observation       : (6,) float64 observation vector; only obs[2:4] used
//   observation_noise : standard deviation σ of the isotropic 2-D Gaussian
//
// Returns: (N,) float64 log-likelihoods.
py::array_t<double> belief_batch_obs_log_likelihood_discrete(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &next_particles,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &observation,
    double observation_noise) {
    if (next_particles.ndim() != 2 || next_particles.shape(1) != 6) {
        throw std::invalid_argument("next_particles must have shape (N, 6)");
    }
    if (observation.ndim() != 1 || observation.shape(0) != 6) {
        throw std::invalid_argument("observation must have shape (6,)");
    }
    const auto n = static_cast<std::size_t>(next_particles.shape(0));
    auto out = py::array_t<double>(static_cast<py::ssize_t>(n));
    if (n == 0) {
        return out;
    }

    const double *part_data = next_particles.data();
    const double *obs_data = observation.data();
    double *out_data = out.mutable_data();

    const double variance = observation_noise * observation_noise;
    const double inv_2var = 0.5 / variance;
    const double log_norm = -std::log(2.0 * M_PI * variance);
    const double obs_obj_x = obs_data[2];
    const double obs_obj_y = obs_data[3];

    for (std::size_t i = 0; i < n; ++i) {
        const double dx = part_data[i * 6 + 2] - obs_obj_x;
        const double dy = part_data[i * 6 + 3] - obs_obj_y;
        out_data[i] = log_norm - (dx * dx + dy * dy) * inv_2var;
    }
    return out;
}

// ── Single-step Push observation kernels (added by perf agent) ──────────────
//
// These free functions expose the Push observation log-likelihood without
// going through ContinuousPushObservationCpp's per-action kernel cache /
// set_next_state / batch_log_likelihood path. The math is the same isotropic
// 2-D Gaussian log-pdf on the object-position slice (cols 2..3) used by the
// existing kernel; this entry skips kernel-cache lookup and per-call
// ``np.ascontiguousarray`` overhead so the Python single-state hot path can
// shave ~10us per call vs the existing routing through ``batch_log_likelihood``.

py::array_t<double> push_observation_log_probability_step(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &next_state,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &observations,
    double observation_noise) {
    if (next_state.ndim() != 1 ||
        static_cast<std::size_t>(next_state.shape(0)) != kPushStateDim) {
        throw std::invalid_argument("next_state must have shape (6,)");
    }
    if (observations.ndim() != 2 ||
        static_cast<std::size_t>(observations.shape(1)) != kPushStateDim) {
        throw std::invalid_argument("observations must have shape (N, 6)");
    }
    auto sv = next_state.unchecked<1>();
    auto ov = observations.unchecked<2>();

    const double variance = observation_noise * observation_noise;
    const double inv_2var = 0.5 / variance;
    const double log_norm = -std::log(2.0 * M_PI * variance);

    const double mean_x = sv(2);
    const double mean_y = sv(3);

    const auto n = static_cast<std::size_t>(observations.shape(0));
    auto out = py::array_t<double>(static_cast<py::ssize_t>(n));
    auto out_view = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = ov(static_cast<py::ssize_t>(i), 2) - mean_x;
        const double dy = ov(static_cast<py::ssize_t>(i), 3) - mean_y;
        out_view(static_cast<py::ssize_t>(i)) = log_norm - (dx * dx + dy * dy) * inv_2var;
    }
    return out;
}

// ── Standalone reward-batch kernels (variant-aware) ─────────────────────────
//
// ``push_reward_batch`` / ``cont_push_reward_batch`` mirror the Python
// reward-model batch path bit-by-bit (sample-mean parity, not bit-exact):
//   * base term: ``-distance(next_obj, target)`` + ``+100`` goal bonus.
//   * obstacle penalty: deterministic point-in-circle (discrete) /
//     circle-vs-AABB overlap (continuous) on the REALISED post-action robot
//     position ``next_state[:2]`` with optional Bernoulli ``hit_probability``.
//   * dangerous-area penalty: variant-driven.
//        - reward_variant_code 0 (CONSTANT_HAZARD_PENALTY): per-row penalty when in zone,
//          optional Bernoulli ``dangerous_area_hit_probability``.
//        - reward_variant_code 1 (ZERO_MEAN_HAZARD_SHOCK): ``±penalty`` 50/50
//          when in zone; expectation is 0.
//        - reward_variant_code 2 (DISTANCE_DECAYED_HAZARD_PENALTY): penalty fires
//          with probability ``exp(-min_dist / penalty_decay)`` over every
//          row (no radius cutoff).
// All RNG draws come from ``pomdp_native::default_rng()``. The variant
// codes, ``dangerous_min_dist_sq`` and ``dangerous_contribution`` are
// defined earlier (near ``discrete_step_reward``) so they can be shared
// between the inline rollout reward and the standalone batch kernels.

py::array_t<double> push_reward_batch(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &states,
    int action_idx,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &next_states,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &obstacles,
    double obstacle_radius, double obstacle_penalty, double obstacle_hit_probability,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dangerous_areas,
    double dangerous_area_radius, double dangerous_area_penalty,
    double dangerous_area_hit_probability, int reward_variant_code,
    double penalty_decay) {
    (void)action_idx;  // reward depends on realised next_state, not the action
    if (next_states.ndim() != 2 || next_states.shape(1) != 6) {
        throw std::invalid_argument("next_states must have shape (N, 6)");
    }
    if (states.ndim() != 2 || states.shape(1) != 6) {
        throw std::invalid_argument("states must have shape (N, 6)");
    }
    const auto n = static_cast<std::size_t>(next_states.shape(0));
    auto out = py::array_t<double>(static_cast<py::ssize_t>(n));
    if (n == 0) {
        return out;
    }
    std::vector<double> obs_flat;
    if (!(obstacles.ndim() == 1 && obstacles.shape(0) == 0)) {
        obs_flat = load_discrete_obstacles(obstacles);
    }
    std::vector<double> areas_flat = load_dangerous_areas(dangerous_areas);
    const double obs_r_sq = obstacle_radius * obstacle_radius;
    const double area_r_sq = dangerous_area_radius * dangerous_area_radius;
    const double *next_data = next_states.data();
    double *out_data = out.mutable_data();
    pomdp_native::RNGState &rng = pomdp_native::default_rng();
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const bool has_areas = !areas_flat.empty();
    const bool has_obs = !obs_flat.empty();
    for (std::size_t i = 0; i < n; ++i) {
        const double rx = next_data[i * 6 + 0];
        const double ry = next_data[i * 6 + 1];
        const double ox = next_data[i * 6 + 2];
        const double oy = next_data[i * 6 + 3];
        const double tx = next_data[i * 6 + 4];
        const double ty = next_data[i * 6 + 5];
        const double dxg = ox - tx;
        const double dyg = oy - ty;
        const double dist = std::sqrt(dxg * dxg + dyg * dyg);
        double reward = -dist;
        if (dist < 0.5) {
            reward += 100.0;
        }
        if (has_obs && discrete_collides(rx, ry, obs_flat, obs_r_sq)) {
            if (obstacle_hit_probability >= 1.0 ||
                uni(rng.engine()) < obstacle_hit_probability) {
                reward += obstacle_penalty;
            }
        }
        if (has_areas) {
            const bool in_zone =
                point_in_dangerous_areas(rx, ry, areas_flat, area_r_sq);
            const double min_sq = (reward_variant_code == kRewardVariantDistanceDecayedHazardPenalty)
                                      ? dangerous_min_dist_sq(rx, ry, areas_flat)
                                      : 0.0;
            reward += dangerous_contribution(reward_variant_code, in_zone,
                                             dangerous_area_penalty,
                                             dangerous_area_hit_probability,
                                             min_sq, penalty_decay, rng);
        }
        out_data[i] = reward;
    }
    return out;
}

py::array_t<double> cont_push_reward_batch(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &states,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &action,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &next_states,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &obstacles,
    double robot_radius, double obstacle_penalty, double obstacle_hit_probability,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dangerous_areas,
    double dangerous_area_radius, double dangerous_area_penalty,
    double dangerous_area_hit_probability, int reward_variant_code,
    double penalty_decay) {
    (void)states;
    (void)action;
    if (next_states.ndim() != 2 || next_states.shape(1) != 6) {
        throw std::invalid_argument("next_states must have shape (N, 6)");
    }
    const auto n = static_cast<std::size_t>(next_states.shape(0));
    auto out = py::array_t<double>(static_cast<py::ssize_t>(n));
    if (n == 0) {
        return out;
    }
    std::vector<double> obs_flat;
    if (!(obstacles.ndim() == 1 && obstacles.shape(0) == 0)) {
        obs_flat = load_obstacles(obstacles);
    }
    std::vector<double> areas_flat = load_dangerous_areas(dangerous_areas);
    const double area_r_sq = dangerous_area_radius * dangerous_area_radius;
    const auto n_obs = obs_flat.size() / 4;
    const double *next_data = next_states.data();
    double *out_data = out.mutable_data();
    pomdp_native::RNGState &rng = pomdp_native::default_rng();
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const bool has_areas = !areas_flat.empty();
    for (std::size_t i = 0; i < n; ++i) {
        const double rx = next_data[i * 6 + 0];
        const double ry = next_data[i * 6 + 1];
        const double ox = next_data[i * 6 + 2];
        const double oy = next_data[i * 6 + 3];
        const double tx = next_data[i * 6 + 4];
        const double ty = next_data[i * 6 + 5];
        const double dxg = ox - tx;
        const double dyg = oy - ty;
        const double dist = std::sqrt(dxg * dxg + dyg * dyg);
        double reward = -dist;
        if (dist < 0.5) {
            reward += 100.0;
        }
        if (n_obs > 0) {
            const double pos[2] = {rx, ry};  // NOLINT(modernize-avoid-c-arrays)
            bool collide = false;
            for (std::size_t j = 0; j < n_obs; ++j) {
                if (cont_circle_aabb_overlap(pos, robot_radius, &obs_flat[j * 4])) {
                    collide = true;
                    break;
                }
            }
            if (collide) {
                if (obstacle_hit_probability >= 1.0 ||
                    uni(rng.engine()) < obstacle_hit_probability) {
                    reward += obstacle_penalty;
                }
            }
        }
        if (has_areas) {
            const bool in_zone =
                point_in_dangerous_areas(rx, ry, areas_flat, area_r_sq);
            const double min_sq = (reward_variant_code == kRewardVariantDistanceDecayedHazardPenalty)
                                      ? dangerous_min_dist_sq(rx, ry, areas_flat)
                                      : 0.0;
            reward += dangerous_contribution(reward_variant_code, in_zone,
                                             dangerous_area_penalty,
                                             dangerous_area_hit_probability,
                                             min_sq, penalty_decay, rng);
        }
        out_data[i] = reward;
    }
    return out;
}

}  // anonymous namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native (C++) sampling hot path for Continuous Push POMDP.";

    m.def("set_seed", &pomdp_native::set_default_seed, py::arg("seed"),
          "Seed the module-level RNG used by sample().");

    m.def("cont_simulate_rollout", &cont_simulate_rollout,
          py::arg("initial_state"), py::arg("action_array"), py::arg("action_indices"),
          py::arg("max_depth"), py::arg("start_depth"), py::arg("discount_factor"),
          py::arg("grid_size"), py::arg("push_threshold"), py::arg("friction_coefficient"),
          py::arg("max_push"), py::arg("robot_radius"), py::arg("obstacle_penalty"),
          py::arg("obstacles"), py::arg("dangerous_areas"),
          py::arg("dangerous_area_radius"), py::arg("dangerous_area_penalty"),
          py::arg("covariance"),
          py::arg("obstacle_hit_probability") = 1.0,
          py::arg("dangerous_area_hit_probability") = 1.0,
          py::arg("reward_variant_code") = 0,
          py::arg("penalty_decay") = 1.0,
          "Native random rollout for ContinuousPushPOMDP. "
          "Returns discounted return from initial_state. "
          "action_indices must be a pre-drawn int32 array of shape (steps_left,). "
          "obstacles must have shape (M, 4) with rows (cx, cy, hx, hy). "
          "dangerous_areas must have shape (K, 2) (or (0, 2) when empty); each "
          "row is the (x, y) centre of a circular dangerous area of radius "
          "``dangerous_area_radius``. Per-step reward consults the REALISED "
          "post-action robot position (next_state[:2]) for obstacle / danger "
          "tests, with Bernoulli ``*_hit_probability`` gating. "
          "``reward_variant_code`` selects the dangerous-area contract "
          "(0=CONSTANT_HAZARD_PENALTY, 1=ZERO_MEAN_HAZARD_SHOCK, 2=DISTANCE_DECAYED_HAZARD_PENALTY); "
          "``penalty_decay`` is the decay length used by variant 2.");

    m.def("simulate_rollout_discrete", &simulate_rollout_discrete,
          py::arg("state"), py::arg("action_indices"), py::arg("max_depth"),
          py::arg("depth"), py::arg("discount"), py::arg("grid_size"),
          py::arg("push_threshold"), py::arg("friction_coefficient"),
          py::arg("obstacles"), py::arg("obstacle_radius"),
          py::arg("obstacle_penalty"),
          py::arg("dangerous_areas"), py::arg("dangerous_area_radius"),
          py::arg("dangerous_area_penalty"),
          py::arg("transition_error_prob"),
          py::arg("obstacle_hit_probability") = 1.0,
          py::arg("dangerous_area_hit_probability") = 1.0,
          py::arg("reward_variant_code") = 0,
          py::arg("penalty_decay") = 1.0,
          "Run a full discrete Push rollout in C++. Returns discounted reward sum. "
          "dangerous_areas must have shape (K, 2) (or empty). Per-step reward "
          "consults the REALISED post-action robot position (next_state[:2]) "
          "for obstacle / danger tests, with Bernoulli ``*_hit_probability`` "
          "gating. ``reward_variant_code`` selects the dangerous-area contract "
          "(0=CONSTANT_HAZARD_PENALTY, 1=ZERO_MEAN_HAZARD_SHOCK, 2=DISTANCE_DECAYED_HAZARD_PENALTY); "
          "``penalty_decay`` is the decay length used by variant 2.");

    m.def("belief_batch_transition_discrete", &belief_batch_transition_discrete,
          py::arg("particles"), py::arg("action_idx"), py::arg("transition_error_prob"),
          py::arg("obstacles"), py::arg("obstacle_radius"), py::arg("grid_size"),
          py::arg("push_threshold"), py::arg("friction_coefficient"),
          "Native batch transition for the discrete Push belief updater. "
          "Applies action_idx to all (N, 6) particles in one C++ call. "
          "When transition_error_prob > 0, an independent C++ RNG decides "
          "per-particle which action actually executes (matches Python "
          "PushVectorizedUpdater._batch_transition_with_error semantics).");

    m.def("belief_batch_obs_log_likelihood_discrete",
          &belief_batch_obs_log_likelihood_discrete,
          py::arg("next_particles"), py::arg("observation"),
          py::arg("observation_noise"),
          "Native batch observation log-likelihood for the discrete Push "
          "belief updater. Returns the per-particle log N(obs[2:4] | "
          "particle[2:4], σ²·I_2) over all (N, 6) particles. "
          "Bit-for-bit equivalent to PushVectorizedUpdater."
          "batch_observation_log_likelihood (no RNG involved).");

    m.def("push_reward_batch", &push_reward_batch,
          py::arg("states"), py::arg("action_idx"), py::arg("next_states"),
          py::arg("obstacles"), py::arg("obstacle_radius"),
          py::arg("obstacle_penalty"), py::arg("obstacle_hit_probability"),
          py::arg("dangerous_areas"), py::arg("dangerous_area_radius"),
          py::arg("dangerous_area_penalty"),
          py::arg("dangerous_area_hit_probability"),
          py::arg("reward_variant_code"), py::arg("penalty_decay"),
          "Standalone variant-aware reward batch for PushPOMDP (discrete).\n\n"
          "Implements all three RewardModelType variants:\n"
          "  0 = CONSTANT_HAZARD_PENALTY: deterministic per-zone penalty (optional Bernoulli).\n"
          "  1 = ZERO_MEAN_HAZARD_SHOCK: ±penalty 50/50 when in zone.\n"
          "  2 = DISTANCE_DECAYED_HAZARD_PENALTY: penalty fires with probability\n"
          "      exp(-min_dist / penalty_decay) for every row (no radius cutoff).\n"
          "Obstacle penalty fires on next_state[:2] (realised position).");

    m.def("cont_push_reward_batch", &cont_push_reward_batch,
          py::arg("states"), py::arg("action"), py::arg("next_states"),
          py::arg("obstacles"), py::arg("robot_radius"),
          py::arg("obstacle_penalty"), py::arg("obstacle_hit_probability"),
          py::arg("dangerous_areas"), py::arg("dangerous_area_radius"),
          py::arg("dangerous_area_penalty"),
          py::arg("dangerous_area_hit_probability"),
          py::arg("reward_variant_code"), py::arg("penalty_decay"),
          "Standalone variant-aware reward batch for ContinuousPushPOMDP.\n\n"
          "Same three variants as push_reward_batch. Obstacle test is a\n"
          "circle-vs-AABB overlap on next_state[:2] with robot_radius;\n"
          "obstacles rows are (cx, cy, hx, hy).");

    py::class_<PushDiscreteTransitionCpp>(m, "PushDiscreteTransitionCpp")
        .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                      py::array_t<double, py::array::c_style | py::array::forcecast>,
                      double, double, double,
                      py::array_t<double, py::array::c_style | py::array::forcecast>,
                      int, double>(),
             py::arg("state"), py::arg("action_dxdy"), py::arg("grid_size"),
             py::arg("push_threshold"), py::arg("friction_coefficient"),
             py::arg("obstacles_flat"), py::arg("n_obstacles"),
             py::arg("obstacle_radius"))
        .def("set_state", &PushDiscreteTransitionCpp::set_state, py::arg("state"))
        .def("compute_next_state", &PushDiscreteTransitionCpp::compute_next_state)
        .def("compute_next_state_for_action",
             &PushDiscreteTransitionCpp::compute_next_state_for_action,
             py::arg("action_dxdy"));

    py::class_<ContinuousPushTransitionCpp>(m, "ContinuousPushTransitionCpp")
        .def(py::init<const py::object &, const py::object &, double, double, double, double,
                      double, const py::array_t<double> &, const py::array_t<double> &>(),
             py::arg("state"), py::arg("action"), py::arg("grid_size"),
             py::arg("push_threshold"), py::arg("friction_coefficient"),
             py::arg("max_push"), py::arg("robot_radius"), py::arg("obstacles"),
             py::arg("covariance"))
        .def("sample", &ContinuousPushTransitionCpp::sample, py::arg("n_samples") = 1)
        .def("sample_one", &ContinuousPushTransitionCpp::sample_one, py::arg("state"))
        .def("probability", &ContinuousPushTransitionCpp::probability, py::arg("values"))
        .def("batch_sample", &ContinuousPushTransitionCpp::batch_sample, py::arg("particles"))
        .def("set_state", &ContinuousPushTransitionCpp::set_state, py::arg("state"))
        .def_property_readonly("state", &ContinuousPushTransitionCpp::state_property)
        .def_property_readonly("action", &ContinuousPushTransitionCpp::action_property);

    m.def(
        "observation_log_probability_step", &push_observation_log_probability_step,
        py::arg("next_state"), py::arg("observations"), py::arg("observation_noise"),
        "Per-observation log-probability for ContinuousPushPOMDP.\n\n"
        "Single-step entry that mirrors\n"
        "ContinuousPushObservationCpp::batch_log_likelihood for one fixed\n"
        "next_state, without kernel-cache lookup or set_next_state overhead.\n"
        "``observations`` must be shape (N, 6) float64 (or (1, 6) for one\n"
        "observation). Returns a (N,) float64 array of log-probabilities.");

    py::class_<ContinuousPushObservationCpp>(m, "ContinuousPushObservationCpp")
        .def(py::init<const py::object &, const py::object &, double, double>(),
             py::arg("next_state"), py::arg("action"), py::arg("observation_noise"),
             py::arg("grid_size"))
        .def("sample", &ContinuousPushObservationCpp::sample, py::arg("n_samples") = 1)
        .def("sample_one", &ContinuousPushObservationCpp::sample_one, py::arg("next_state"))
        .def("probability", &ContinuousPushObservationCpp::probability, py::arg("values"))
        .def("batch_log_likelihood", &ContinuousPushObservationCpp::batch_log_likelihood,
             py::arg("next_particles"), py::arg("observation"))
        .def("set_next_state", &ContinuousPushObservationCpp::set_next_state,
             py::arg("next_state"))
        .def_property_readonly("next_state", &ContinuousPushObservationCpp::next_state_property)
        .def_property_readonly("action", &ContinuousPushObservationCpp::action_property);
}
