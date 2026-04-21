// Continuous LaserTag POMDP native sampling hot path.
//
// The continuous LaserTag transition and observation models do not fit the
// shared ``TransitionModelCpp<Dim>`` / ``ObservationModelCpp<Dim>`` base
// classes directly: state has dimension 5 (``[robot_x, robot_y, opp_x, opp_y,
// terminal_flag]``) but only two disjoint 2-D position slices carry Gaussian
// noise, and the observation mean is a geometry-based laser scan (not the
// identity on the next state). This extension therefore composes
// ``pomdp_native::GaussianND<2>`` and the module-level RNG explicitly instead
// of inheriting, while preserving the auto-dispatch protocol used by
// ``WeightedParticleBelief._update_weights``: both classes expose
// ``sample``, ``probability``, ``batch_sample`` (transition) and
// ``batch_log_likelihood`` (observation) with the same signatures and return
// types as the MountainCar port.
//
// The laser geometry (ray-AABB slab intersection, ray-circle intersection,
// circle-AABB wall-collision resolution, grid clamping) is ported from
// ``continuous_laser_tag_geometry.py`` to C++ so the belief-update hot path
// does not bounce back into Python.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pomdp_native/gaussian.hpp"
#include "pomdp_native/marshalling.hpp"
#include "pomdp_native/rng.hpp"

namespace py = pybind11;

namespace {

constexpr std::size_t kStateDim = 5;
constexpr std::size_t kObsDim = 8;
constexpr double kRayMax = 1e4;
constexpr double kParallelEps = 1e-12;
constexpr double kHitEps = 1e-9;

// 8 laser-ray unit direction vectors: N, NE, E, SE, S, SW, W, NW.
// Matches LASER_DIRECTIONS in continuous_laser_tag_geometry.py.
constexpr double kSqrt2Inv = 0.70710678118654752440;
constexpr std::array<std::array<double, 2>, kObsDim> kLaserDirections = {{
    {0.0, 1.0},
    {kSqrt2Inv, kSqrt2Inv},
    {1.0, 0.0},
    {kSqrt2Inv, -kSqrt2Inv},
    {0.0, -1.0},
    {-kSqrt2Inv, -kSqrt2Inv},
    {-1.0, 0.0},
    {-kSqrt2Inv, kSqrt2Inv},
}};

// A wall is stored as (cx, cy, hx, hy): AABB center and half-extents.
struct Wall {
    double cx;
    double cy;
    double hx;
    double hy;
};

std::vector<Wall> load_walls(const py::array_t<double> &walls_arr) {
    std::vector<Wall> walls;
    if (walls_arr.ndim() == 1 && walls_arr.shape(0) == 0) {
        return walls;
    }
    if (walls_arr.ndim() != 2 || walls_arr.shape(1) != 4) {
        throw std::invalid_argument("walls must have shape (M, 4)");
    }
    const auto m = static_cast<std::size_t>(walls_arr.shape(0));
    auto u = walls_arr.unchecked<2>();
    walls.reserve(m);
    for (std::size_t i = 0; i < m; ++i) {
        walls.push_back({u(static_cast<py::ssize_t>(i), 0), u(static_cast<py::ssize_t>(i), 1),
                         u(static_cast<py::ssize_t>(i), 2), u(static_cast<py::ssize_t>(i), 3)});
    }
    return walls;
}

// Single-ray slab intersection: returns min positive hit distance, or kRayMax
// if the ray does not intersect the AABB within (0, kRayMax].
double ray_single_aabb(double ox, double oy, double dx, double dy, const Wall &w) {
    const double min_x = w.cx - w.hx;
    const double max_x = w.cx + w.hx;
    const double min_y = w.cy - w.hy;
    const double max_y = w.cy + w.hy;

    double t_enter_x;
    double t_exit_x;
    if (std::abs(dx) > kParallelEps) {
        const double inv_dx = 1.0 / dx;
        double tx1 = (min_x - ox) * inv_dx;
        double tx2 = (max_x - ox) * inv_dx;
        t_enter_x = std::min(tx1, tx2);
        t_exit_x = std::max(tx1, tx2);
    } else {
        const bool inside_x = (ox >= min_x) && (ox <= max_x);
        t_enter_x = inside_x ? -std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::infinity();
        t_exit_x = inside_x ? std::numeric_limits<double>::infinity()
                            : -std::numeric_limits<double>::infinity();
    }

    double t_enter_y;
    double t_exit_y;
    if (std::abs(dy) > kParallelEps) {
        const double inv_dy = 1.0 / dy;
        double ty1 = (min_y - oy) * inv_dy;
        double ty2 = (max_y - oy) * inv_dy;
        t_enter_y = std::min(ty1, ty2);
        t_exit_y = std::max(ty1, ty2);
    } else {
        const bool inside_y = (oy >= min_y) && (oy <= max_y);
        t_enter_y = inside_y ? -std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::infinity();
        t_exit_y = inside_y ? std::numeric_limits<double>::infinity()
                            : -std::numeric_limits<double>::infinity();
    }

    const double t_min = std::max(t_enter_x, t_enter_y);
    const double t_max = std::min(t_exit_x, t_exit_y);
    const double hit_t = (t_min > 0.0) ? t_min : t_max;
    const bool valid = (t_max > std::max(t_min, 0.0)) && (hit_t > kHitEps);
    if (valid) {
        return hit_t;
    }
    return kRayMax;
}

double ray_walls_distance(double ox, double oy, double dx, double dy,
                          const std::vector<Wall> &walls) {
    double best = kRayMax;
    for (const auto &w : walls) {
        const double t = ray_single_aabb(ox, oy, dx, dy, w);
        if (t < best) {
            best = t;
        }
    }
    return best;
}

double ray_grid_boundary_distance(double ox, double oy, double dx, double dy,
                                  double grid_w, double grid_h) {
    double best = kRayMax;
    if (std::abs(dx) > kParallelEps) {
        const double t_left = -ox / dx;
        const double t_right = (grid_w - ox) / dx;
        if (t_left > kHitEps && t_left < best) {
            best = t_left;
        }
        if (t_right > kHitEps && t_right < best) {
            best = t_right;
        }
    }
    if (std::abs(dy) > kParallelEps) {
        const double t_bottom = -oy / dy;
        const double t_top = (grid_h - oy) / dy;
        if (t_bottom > kHitEps && t_bottom < best) {
            best = t_bottom;
        }
        if (t_top > kHitEps && t_top < best) {
            best = t_top;
        }
    }
    return best;
}

double ray_circle_distance(double ox, double oy, double dx, double dy, double cx, double cy,
                           double radius) {
    const double ocx = ox - cx;
    const double ocy = oy - cy;
    const double b = ocx * dx + ocy * dy;
    const double c = ocx * ocx + ocy * ocy - radius * radius;
    const double disc = b * b - c;
    if (disc < 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    const double sqrt_disc = std::sqrt(disc);
    const double t1 = -b - sqrt_disc;
    if (t1 > kHitEps) {
        return t1;
    }
    const double t2 = -b + sqrt_disc;
    if (t2 > kHitEps) {
        return t2;
    }
    return std::numeric_limits<double>::infinity();
}

// Computes the 8-direction laser measurement vector from a single robot
// position. Matches compute_laser_measurements in continuous_laser_tag_geometry.py.
void compute_laser_measurements(double robot_x, double robot_y, double opp_x, double opp_y,
                                double opponent_radius, const std::vector<Wall> &walls,
                                double grid_w, double grid_h, double *out) {
    for (std::size_t d = 0; d < kObsDim; ++d) {
        const double dx = kLaserDirections[d][0];
        const double dy = kLaserDirections[d][1];
        double best = ray_walls_distance(robot_x, robot_y, dx, dy, walls);
        const double boundary = ray_grid_boundary_distance(robot_x, robot_y, dx, dy, grid_w, grid_h);
        if (boundary < best) {
            best = boundary;
        }
        const double opp_d =
            ray_circle_distance(robot_x, robot_y, dx, dy, opp_x, opp_y, opponent_radius);
        if (opp_d < best) {
            best = opp_d;
        }
        out[d] = best;
    }
}

// Resolve collision between a circular entity at (x, y) with radius r and a
// single wall AABB. Mirrors _resolve_single_wall in the Python geometry
// module (circle-AABB minimum-translation-vector push-out).
void resolve_single_wall(double &x, double &y, double radius, const Wall &w) {
    const double closest_x = std::clamp(x, w.cx - w.hx, w.cx + w.hx);
    const double closest_y = std::clamp(y, w.cy - w.hy, w.cy + w.hy);
    const double dx = x - closest_x;
    const double dy = y - closest_y;
    const double dist_sq = dx * dx + dy * dy;
    const double r_sq = radius * radius;
    if (dist_sq >= r_sq) {
        return;
    }
    const double dist = (dist_sq > 1e-24) ? std::sqrt(dist_sq) : 0.0;
    if (dist < 1e-12) {
        const double pen_left = x - (w.cx - w.hx);
        const double pen_right = (w.cx + w.hx) - x;
        const double pen_down = y - (w.cy - w.hy);
        const double pen_up = (w.cy + w.hy) - y;
        const double min_pen = std::min({pen_left, pen_right, pen_down, pen_up});
        if (min_pen == pen_left) {
            x = w.cx - w.hx - radius;
        } else if (min_pen == pen_right) {
            x = w.cx + w.hx + radius;
        } else if (min_pen == pen_down) {
            y = w.cy - w.hy - radius;
        } else {
            y = w.cy + w.hy + radius;
        }
    } else {
        const double overlap = radius - dist;
        x += (dx / dist) * overlap;
        y += (dy / dist) * overlap;
    }
}

void resolve_walls(double &x, double &y, double radius, const std::vector<Wall> &walls) {
    for (const auto &w : walls) {
        resolve_single_wall(x, y, radius, w);
    }
}

void clamp_to_grid(double &x, double &y, double radius, double grid_w, double grid_h) {
    x = std::clamp(x, radius, grid_w - radius);
    y = std::clamp(y, radius, grid_h - radius);
}

// Shared environment geometry / physics parameters used by both the
// transition and observation models.
struct EnvParams {
    std::vector<Wall> walls;
    double grid_w;
    double grid_h;
    double robot_radius;
    double opponent_radius;
    double tag_radius;
    double pursuit_speed;
};

EnvParams make_env_params(const py::array_t<double> &walls_arr,
                          const py::array_t<double> &grid_size, double robot_radius,
                          double opponent_radius, double tag_radius, double pursuit_speed) {
    if (grid_size.ndim() != 1 || grid_size.shape(0) != 2) {
        throw std::invalid_argument("grid_size must have shape (2,)");
    }
    auto g = grid_size.unchecked<1>();
    EnvParams p;
    p.walls = load_walls(walls_arr);
    p.grid_w = g(0);
    p.grid_h = g(1);
    p.robot_radius = robot_radius;
    p.opponent_radius = opponent_radius;
    p.tag_radius = tag_radius;
    p.pursuit_speed = pursuit_speed;
    return p;
}

// Draw a 2-D Gaussian step around (mean_x, mean_y), push out of walls, and
// clamp to the grid. Centralizes the shared "Gaussian step + geometry" logic
// used by both the robot and opponent update paths.
void sample_move(double mean_x, double mean_y, double radius,
                 const pomdp_native::GaussianND<2> &noise, const EnvParams &env,
                 pomdp_native::RNGState &rng, double &out_x, double &out_y) {
    const double mean[2] = {mean_x, mean_y};
    double sample[2];
    noise.sample_into(sample, mean, rng);
    out_x = sample[0];
    out_y = sample[1];
    resolve_walls(out_x, out_y, radius, env.walls);
    clamp_to_grid(out_x, out_y, radius, env.grid_w, env.grid_h);
}

// Compute the pursuit target for the opponent: move ``pursuit_speed`` toward
// the robot, sampling a random unit vector when the two are coincident.
void opponent_pursuit_mean(double robot_x, double robot_y, double opp_x, double opp_y,
                           double pursuit_speed, pomdp_native::RNGState &rng, double &mean_x,
                           double &mean_y) {
    const double diff_x = robot_x - opp_x;
    const double diff_y = robot_y - opp_y;
    const double dist = std::hypot(diff_x, diff_y);
    double dir_x;
    double dir_y;
    if (dist < 1e-9) {
        std::normal_distribution<double> standard_normal(0.0, 1.0);
        dir_x = standard_normal(rng.engine());
        dir_y = standard_normal(rng.engine());
        const double dir_norm = std::max(std::hypot(dir_x, dir_y), 1e-9);
        dir_x /= dir_norm;
        dir_y /= dir_norm;
    } else {
        dir_x = diff_x / dist;
        dir_y = diff_y / dist;
    }
    mean_x = opp_x + pursuit_speed * dir_x;
    mean_y = opp_y + pursuit_speed * dir_y;
}

// Parse the 3-element action vector (dx, dy, tag_flag) from a Python object.
std::array<double, 3> parse_action(const py::object &action) {
    std::array<double, 3> out{};
    if (py::isinstance<py::array>(action)) {
        auto arr = action.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        if (arr.ndim() != 1) {
            throw std::invalid_argument("action ndarray must be 1-D");
        }
        const auto n = static_cast<std::size_t>(arr.shape(0));
        if (n < 2 || n > 3) {
            throw std::invalid_argument("action must have length 2 or 3");
        }
        auto u = arr.unchecked<1>();
        out[0] = u(0);
        out[1] = u(1);
        out[2] = (n == 3) ? u(2) : 0.0;
        return out;
    }
    auto seq = action.cast<py::sequence>();
    const auto n = static_cast<std::size_t>(py::len(seq));
    if (n < 2 || n > 3) {
        throw std::invalid_argument("action must have length 2 or 3");
    }
    out[0] = seq[0].cast<double>();
    out[1] = seq[1].cast<double>();
    out[2] = (n == 3) ? seq[2].cast<double>() : 0.0;
    return out;
}

// Extract a (5,) state row from a Python object (ndarray / tuple / list).
std::array<double, kStateDim> parse_state(const py::object &state) {
    return pomdp_native::to_array<kStateDim>(state, "state");
}

class ContinuousLaserTagTransitionCpp {
  public:
    ContinuousLaserTagTransitionCpp(const py::object &state_obj, const py::object &action_obj,
                                    const py::array_t<double> &robot_cov,
                                    const py::array_t<double> &opponent_cov, double pursuit_speed,
                                    const py::array_t<double> &walls_arr,
                                    const py::array_t<double> &grid_size, double robot_radius,
                                    double opponent_radius, double tag_radius)
        : state_(parse_state(state_obj)),
          action_(parse_action(action_obj)),
          robot_noise_(pomdp_native::GaussianND<2>::from_covariance(robot_cov)),
          opp_noise_(pomdp_native::GaussianND<2>::from_covariance(opponent_cov)),
          env_(make_env_params(walls_arr, grid_size, robot_radius, opponent_radius, tag_radius,
                               pursuit_speed)) {}

    py::list sample(int n_samples) const {
        if (n_samples < 0) {
            throw std::invalid_argument("n_samples must be non-negative");
        }
        py::list out;
        pomdp_native::RNGState &rng = pomdp_native::default_rng();
        double row[kStateDim];  // NOLINT(modernize-avoid-c-arrays)
        for (int i = 0; i < n_samples; ++i) {
            sample_into(state_.data(), row, rng);
            out.append(pomdp_native::array_from_vector(row, kStateDim));
        }
        return out;
    }

    // The state-transition density does not have a tractable closed form
    // (it's a Gaussian convolved with wall-collision / clamping projection).
    // Returning zeros preserves the pre-port Python placeholder contract.
    py::array_t<double> probability(const py::object &values) const {
        auto batch = pomdp_native::extract_rows_nd(values, kStateDim);
        auto out = py::array_t<double>(static_cast<py::ssize_t>(batch.n));
        auto buf = out.mutable_unchecked<1>();
        for (std::size_t i = 0; i < batch.n; ++i) {
            buf(static_cast<py::ssize_t>(i)) = 0.0;
        }
        return out;
    }

    py::array_t<double> batch_sample(
        const py::array_t<double, py::array::c_style | py::array::forcecast> &particles) const {
        if (particles.ndim() != 2 || particles.shape(1) != static_cast<py::ssize_t>(kStateDim)) {
            throw std::invalid_argument("particles must have shape (N, 5)");
        }
        const auto n_rows = static_cast<std::size_t>(particles.shape(0));
        auto in_view = particles.unchecked<2>();
        auto out = py::array_t<double>(
            {static_cast<py::ssize_t>(n_rows), static_cast<py::ssize_t>(kStateDim)});
        auto out_view = out.mutable_unchecked<2>();

        pomdp_native::RNGState &rng = pomdp_native::default_rng();
        double row[kStateDim];  // NOLINT(modernize-avoid-c-arrays)
        double next[kStateDim];  // NOLINT(modernize-avoid-c-arrays)
        for (std::size_t i = 0; i < n_rows; ++i) {
            for (std::size_t d = 0; d < kStateDim; ++d) {
                row[d] = in_view(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(d));
            }
            sample_into(row, next, rng);
            for (std::size_t d = 0; d < kStateDim; ++d) {
                out_view(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(d)) = next[d];
            }
        }
        return out;
    }

    // Properties (stored state / action exposed as Python tuples).
    py::tuple state_property() const {
        return py::make_tuple(state_[0], state_[1], state_[2], state_[3], state_[4]);
    }
    py::tuple action_property() const {
        return py::make_tuple(action_[0], action_[1], action_[2]);
    }

  private:
    void sample_into(const double *src, double *out, pomdp_native::RNGState &rng) const {
        // Terminal particles are absorbing: copy the state through unchanged.
        if (src[4] != 0.0) {
            for (std::size_t d = 0; d < kStateDim; ++d) {
                out[d] = src[d];
            }
            return;
        }
        const double robot_x = src[0];
        const double robot_y = src[1];
        const double opp_x = src[2];
        const double opp_y = src[3];
        const double tag_flag = action_[2];

        if (tag_flag > 0.5) {
            const double dx = robot_x - opp_x;
            const double dy = robot_y - opp_y;
            const double dist = std::hypot(dx, dy);
            if (dist <= env_.tag_radius) {
                out[0] = robot_x;
                out[1] = robot_y;
                out[2] = opp_x;
                out[3] = opp_y;
                out[4] = 1.0;
                return;
            }
            // Robot does not move on a tag; opponent still pursues.
            double mean_opp_x;
            double mean_opp_y;
            opponent_pursuit_mean(robot_x, robot_y, opp_x, opp_y, env_.pursuit_speed, rng,
                                  mean_opp_x, mean_opp_y);
            double new_opp_x;
            double new_opp_y;
            sample_move(mean_opp_x, mean_opp_y, env_.opponent_radius, opp_noise_, env_, rng,
                        new_opp_x, new_opp_y);
            out[0] = robot_x;
            out[1] = robot_y;
            out[2] = new_opp_x;
            out[3] = new_opp_y;
            out[4] = 0.0;
            return;
        }

        // Non-tag action: apply robot Gaussian move then opponent pursuit.
        double new_robot_x;
        double new_robot_y;
        sample_move(robot_x + action_[0], robot_y + action_[1], env_.robot_radius, robot_noise_,
                    env_, rng, new_robot_x, new_robot_y);

        double mean_opp_x;
        double mean_opp_y;
        opponent_pursuit_mean(new_robot_x, new_robot_y, opp_x, opp_y, env_.pursuit_speed, rng,
                              mean_opp_x, mean_opp_y);
        double new_opp_x;
        double new_opp_y;
        sample_move(mean_opp_x, mean_opp_y, env_.opponent_radius, opp_noise_, env_, rng, new_opp_x,
                    new_opp_y);

        out[0] = new_robot_x;
        out[1] = new_robot_y;
        out[2] = new_opp_x;
        out[3] = new_opp_y;
        out[4] = 0.0;
    }

    std::array<double, kStateDim> state_;
    std::array<double, 3> action_;
    pomdp_native::GaussianND<2> robot_noise_;
    pomdp_native::GaussianND<2> opp_noise_;
    EnvParams env_;
};

class ContinuousLaserTagObservationCpp {
  public:
    ContinuousLaserTagObservationCpp(const py::object &next_state_obj,
                                     const py::object &action_obj, double measurement_noise,
                                     const py::array_t<double> &walls_arr,
                                     const py::array_t<double> &grid_size, double opponent_radius)
        : next_state_(parse_state(next_state_obj)),
          action_(parse_action(action_obj)),
          measurement_noise_(measurement_noise),
          variance_(measurement_noise * measurement_noise),
          inv_2var_(0.5 / (measurement_noise * measurement_noise)),
          log_norm_1d_(-0.5 * std::log(2.0 * M_PI * measurement_noise * measurement_noise)),
          env_(make_env_params(walls_arr, grid_size, 0.0, opponent_radius, 0.0, 0.0)) {}

    py::list sample(int n_samples) const {
        if (n_samples < 0) {
            throw std::invalid_argument("n_samples must be non-negative");
        }
        py::list out;
        if (next_state_[4] != 0.0) {
            for (int i = 0; i < n_samples; ++i) {
                double terminal[kObsDim];  // NOLINT(modernize-avoid-c-arrays)
                for (std::size_t d = 0; d < kObsDim; ++d) {
                    terminal[d] = -1.0;
                }
                out.append(pomdp_native::array_from_vector(terminal, kObsDim));
            }
            return out;
        }

        double mean[kObsDim];  // NOLINT(modernize-avoid-c-arrays)
        compute_mean(next_state_.data(), mean);
        pomdp_native::RNGState &rng = pomdp_native::default_rng();
        std::normal_distribution<double> standard_normal(0.0, 1.0);
        for (int i = 0; i < n_samples; ++i) {
            double obs[kObsDim];  // NOLINT(modernize-avoid-c-arrays)
            for (std::size_t d = 0; d < kObsDim; ++d) {
                const double z = standard_normal(rng.engine());
                obs[d] = std::max(0.0, mean[d] + measurement_noise_ * z);
            }
            out.append(pomdp_native::array_from_vector(obs, kObsDim));
        }
        return out;
    }

    py::array_t<double> probability(const py::object &values) const {
        auto batch = pomdp_native::extract_rows_nd(values, kObsDim);
        auto out = py::array_t<double>(static_cast<py::ssize_t>(batch.n));
        auto buf = out.mutable_unchecked<1>();
        if (next_state_[4] != 0.0) {
            for (std::size_t i = 0; i < batch.n; ++i) {
                const double *obs = batch.flat.data() + i * kObsDim;
                bool matches_terminal = true;
                for (std::size_t d = 0; d < kObsDim; ++d) {
                    if (std::abs(obs[d] - (-1.0)) > 1e-8) {
                        matches_terminal = false;
                        break;
                    }
                }
                buf(static_cast<py::ssize_t>(i)) = matches_terminal ? 1.0 : 0.0;
            }
            return out;
        }
        double mean[kObsDim];  // NOLINT(modernize-avoid-c-arrays)
        compute_mean(next_state_.data(), mean);
        for (std::size_t i = 0; i < batch.n; ++i) {
            buf(static_cast<py::ssize_t>(i)) = std::exp(log_pdf(batch.flat.data() + i * kObsDim, mean));
        }
        return out;
    }

    py::array_t<double> batch_log_likelihood(
        const py::array_t<double, py::array::c_style | py::array::forcecast> &next_particles,
        const py::array_t<double, py::array::c_style | py::array::forcecast> &observation) const {
        if (next_particles.ndim() != 2 ||
            next_particles.shape(1) != static_cast<py::ssize_t>(kStateDim)) {
            throw std::invalid_argument("next_particles must have shape (N, 5)");
        }
        if (observation.ndim() != 1 || observation.shape(0) != static_cast<py::ssize_t>(kObsDim)) {
            throw std::invalid_argument("observation must have shape (8,)");
        }
        const auto n_rows = static_cast<std::size_t>(next_particles.shape(0));
        auto part_view = next_particles.unchecked<2>();
        auto obs_view = observation.unchecked<1>();

        double obs[kObsDim];  // NOLINT(modernize-avoid-c-arrays)
        bool obs_is_terminal = true;
        for (std::size_t d = 0; d < kObsDim; ++d) {
            obs[d] = obs_view(static_cast<py::ssize_t>(d));
            if (std::abs(obs[d] - (-1.0)) > 1e-8) {
                obs_is_terminal = false;
            }
        }

        auto out = py::array_t<double>(static_cast<py::ssize_t>(n_rows));
        auto buf = out.mutable_unchecked<1>();

        const double neg_inf = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < n_rows; ++i) {
            const double terminal_flag =
                part_view(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(4));
            if (terminal_flag != 0.0) {
                buf(static_cast<py::ssize_t>(i)) = obs_is_terminal ? 0.0 : neg_inf;
                continue;
            }
            if (obs_is_terminal) {
                buf(static_cast<py::ssize_t>(i)) = neg_inf;
                continue;
            }
            double state_row[kStateDim];  // NOLINT(modernize-avoid-c-arrays)
            for (std::size_t d = 0; d < kStateDim; ++d) {
                state_row[d] = part_view(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(d));
            }
            double mean[kObsDim];  // NOLINT(modernize-avoid-c-arrays)
            compute_mean(state_row, mean);
            buf(static_cast<py::ssize_t>(i)) = log_pdf(obs, mean);
        }
        return out;
    }

    py::tuple next_state_property() const {
        return py::make_tuple(next_state_[0], next_state_[1], next_state_[2], next_state_[3],
                              next_state_[4]);
    }
    py::tuple action_property() const {
        return py::make_tuple(action_[0], action_[1], action_[2]);
    }
    py::array_t<double> mean_property() const {
        double mean[kObsDim];  // NOLINT(modernize-avoid-c-arrays)
        if (next_state_[4] != 0.0) {
            for (std::size_t d = 0; d < kObsDim; ++d) {
                mean[d] = -1.0;
            }
        } else {
            compute_mean(next_state_.data(), mean);
        }
        return pomdp_native::array_from_vector(mean, kObsDim);
    }

  private:
    void compute_mean(const double *state, double *out) const {
        compute_laser_measurements(state[0], state[1], state[2], state[3], env_.opponent_radius,
                                   env_.walls, env_.grid_w, env_.grid_h, out);
    }

    double log_pdf(const double *obs, const double *mean) const {
        double sq = 0.0;
        for (std::size_t d = 0; d < kObsDim; ++d) {
            const double diff = obs[d] - mean[d];
            sq += diff * diff;
        }
        return static_cast<double>(kObsDim) * log_norm_1d_ - sq * inv_2var_;
    }

    std::array<double, kStateDim> next_state_;
    std::array<double, 3> action_;
    double measurement_noise_;
    double variance_;
    double inv_2var_;
    double log_norm_1d_;
    EnvParams env_;
};

}  // anonymous namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native (C++) sampling hot path for Continuous LaserTag POMDP.";

    m.def("set_seed", &pomdp_native::set_default_seed, py::arg("seed"),
          "Seed the module-level RNG used by sample() / batch_sample() / batch_log_likelihood().");

    py::class_<ContinuousLaserTagTransitionCpp>(m, "ContinuousLaserTagTransitionCpp")
        .def(py::init<const py::object &, const py::object &, const py::array_t<double> &,
                      const py::array_t<double> &, double, const py::array_t<double> &,
                      const py::array_t<double> &, double, double, double>(),
             py::arg("state"), py::arg("action"), py::arg("robot_covariance"),
             py::arg("opponent_covariance"), py::arg("pursuit_speed"), py::arg("walls"),
             py::arg("grid_size"), py::arg("robot_radius"), py::arg("opponent_radius"),
             py::arg("tag_radius"))
        .def("sample", &ContinuousLaserTagTransitionCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &ContinuousLaserTagTransitionCpp::probability, py::arg("values"))
        .def("batch_sample", &ContinuousLaserTagTransitionCpp::batch_sample, py::arg("particles"))
        .def_property_readonly("state", &ContinuousLaserTagTransitionCpp::state_property)
        .def_property_readonly("action", &ContinuousLaserTagTransitionCpp::action_property);

    py::class_<ContinuousLaserTagObservationCpp>(m, "ContinuousLaserTagObservationCpp")
        .def(py::init<const py::object &, const py::object &, double,
                      const py::array_t<double> &, const py::array_t<double> &, double>(),
             py::arg("next_state"), py::arg("action"), py::arg("measurement_noise"),
             py::arg("walls"), py::arg("grid_size"), py::arg("opponent_radius"))
        .def("sample", &ContinuousLaserTagObservationCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &ContinuousLaserTagObservationCpp::probability, py::arg("values"))
        .def("batch_log_likelihood", &ContinuousLaserTagObservationCpp::batch_log_likelihood,
             py::arg("next_particles"), py::arg("observation"))
        .def_property_readonly("next_state", &ContinuousLaserTagObservationCpp::next_state_property)
        .def_property_readonly("action", &ContinuousLaserTagObservationCpp::action_property)
        .def_property_readonly("mean", &ContinuousLaserTagObservationCpp::mean_property);
}
