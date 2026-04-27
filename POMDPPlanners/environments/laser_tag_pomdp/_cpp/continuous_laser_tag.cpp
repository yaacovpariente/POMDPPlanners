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
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>
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

    // Rewrite only the state field; covariance / Cholesky factors, action,
    // and env geometry stay frozen so cached Cholesky factors and the
    // pre-built wall list remain valid. Lets Python keep one kernel per
    // (env, action) instead of rebuilding for every call.
    void set_state(const py::object &state_obj) { state_ = parse_state(state_obj); }

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
    // Shared terminal-sentinel predicates so kernel.probability and
    // kernel.batch_log_likelihood agree on the contract:
    //   - next_state terminal (flag != 0) AND obs is the all--1 sentinel ->
    //     prob = 1.0 (log = 0.0): certainty.
    //   - next_state terminal XOR obs sentinel -> prob = 0.0 (log = -inf):
    //     impossible.
    //   - both non-terminal -> Gaussian PDF.
    static bool is_terminal_state(const double *state) { return state[4] != 0.0; }

    static bool is_terminal_sentinel_obs(const double *obs) {
        for (std::size_t d = 0; d < kObsDim; ++d) {
            if (std::abs(obs[d] - (-1.0)) > 1e-8) {
                return false;
            }
        }
        return true;
    }

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
        const bool state_is_terminal = is_terminal_state(next_state_.data());
        if (state_is_terminal) {
            for (std::size_t i = 0; i < batch.n; ++i) {
                const double *obs = batch.flat.data() + i * kObsDim;
                buf(static_cast<py::ssize_t>(i)) = is_terminal_sentinel_obs(obs) ? 1.0 : 0.0;
            }
            return out;
        }
        double mean[kObsDim];  // NOLINT(modernize-avoid-c-arrays)
        compute_mean(next_state_.data(), mean);
        for (std::size_t i = 0; i < batch.n; ++i) {
            const double *obs = batch.flat.data() + i * kObsDim;
            // Non-terminal next_state with the terminal sentinel observation
            // is impossible per the kernel contract; return 0.0 so the
            // Python wrapper's np.log(...) under errstate(divide="ignore")
            // produces -inf, matching batch_log_likelihood's explicit guard.
            if (is_terminal_sentinel_obs(obs)) {
                buf(static_cast<py::ssize_t>(i)) = 0.0;
                continue;
            }
            buf(static_cast<py::ssize_t>(i)) = std::exp(log_pdf(obs, mean));
        }
        return out;
    }

    // B1 fix: return ``log_pdf`` directly without the exp/log round-trip
    // performed by ``probability``. The exp(log_pdf) round-trip in the
    // legacy ``probability`` path silently collapses to 0.0 for
    // observations whose log_pdf is below the IEEE-754 double underflow
    // boundary (~ -745), making the Python ``np.log(probability(...))``
    // wrapper return -inf while the batched ``batch_log_likelihood``
    // path returns the finite log_pdf. This entry point lets the Python
    // wrapper preserve the finite log-likelihood for low-density obs.
    //
    // Terminal-sentinel handling mirrors ``probability``: when the
    // stored next_state is terminal, the only matching observation is
    // the all-(-1) sentinel — its prob is 1.0 (log = 0.0); any other
    // observation has prob 0.0 (log = -inf). This branch is the B2
    // surface and is intentionally left in lock-step with ``probability``
    // for now (B2 owns it in a separate fix).
    py::array_t<double> log_probability(const py::object &values) const {
        auto batch = pomdp_native::extract_rows_nd(values, kObsDim);
        auto out = py::array_t<double>(static_cast<py::ssize_t>(batch.n));
        auto buf = out.mutable_unchecked<1>();
        const double neg_inf = -std::numeric_limits<double>::infinity();
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
                buf(static_cast<py::ssize_t>(i)) = matches_terminal ? 0.0 : neg_inf;
            }
            return out;
        }
        double mean[kObsDim];  // NOLINT(modernize-avoid-c-arrays)
        compute_mean(next_state_.data(), mean);
        for (std::size_t i = 0; i < batch.n; ++i) {
            buf(static_cast<py::ssize_t>(i)) = log_pdf(batch.flat.data() + i * kObsDim, mean);
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
        for (std::size_t d = 0; d < kObsDim; ++d) {
            obs[d] = obs_view(static_cast<py::ssize_t>(d));
        }
        const bool obs_is_terminal = is_terminal_sentinel_obs(obs);

        auto out = py::array_t<double>(static_cast<py::ssize_t>(n_rows));
        auto buf = out.mutable_unchecked<1>();

        const double neg_inf = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < n_rows; ++i) {
            double state_row[kStateDim];  // NOLINT(modernize-avoid-c-arrays)
            for (std::size_t d = 0; d < kStateDim; ++d) {
                state_row[d] = part_view(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(d));
            }
            const bool state_is_terminal = is_terminal_state(state_row);
            if (state_is_terminal) {
                buf(static_cast<py::ssize_t>(i)) = obs_is_terminal ? 0.0 : neg_inf;
                continue;
            }
            if (obs_is_terminal) {
                buf(static_cast<py::ssize_t>(i)) = neg_inf;
                continue;
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

    // Rewrite only the next_state field; measurement_noise constants, action,
    // and env geometry (walls / grid / opponent radius) stay frozen so the
    // cached log-norm factor and pre-built wall list remain valid.
    void set_next_state(const py::object &next_state_obj) {
        next_state_ = parse_state(next_state_obj);
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

// Vectorised reward kernel.
//
// Mirrors ContinuousLaserTagPOMDP.reward / reward_batch in pure C++:
//   - terminal-flag rows yield 0.0 (live mask in Python collapsed here);
//   - on a tag action (action[2] > 0.5) live rows get +tag_reward when
//     ||robot - opp|| <= tag_radius, else -tag_penalty;
//   - all live rows pay -step_cost;
//   - any live row whose robot position lies within
//     ``dangerous_area_radius`` of any (dx, dy) pays -dangerous_area_penalty.
//
// dangerous_areas is shape (K, 2) (or empty (0, 2) / 1-D length-0); the
// per-particle inner loop replaces the per-area numpy ``np.sqrt(...)``
// + boolean-index assignment chain that dominated PFT-DPW reward time.
py::array_t<double> reward_batch(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &states,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &action,
    double tag_radius, double tag_reward, double tag_penalty, double step_cost,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dangerous_areas,
    double dangerous_area_radius, double dangerous_area_penalty) {
    if (states.ndim() != 2 || states.shape(1) != static_cast<py::ssize_t>(kStateDim)) {
        throw std::invalid_argument("states must have shape (N, 5)");
    }
    if (action.ndim() != 1 || (action.shape(0) != 2 && action.shape(0) != 3)) {
        throw std::invalid_argument("action must be a length-2 or length-3 1-D array");
    }
    auto action_view = action.unchecked<1>();
    const double tag_flag = (action.shape(0) == 3) ? action_view(2) : 0.0;
    const bool is_tag = tag_flag > 0.5;

    // dangerous_areas: accept (K, 2) or (0,) for empty.
    std::vector<std::pair<double, double>> areas;
    if (!(dangerous_areas.ndim() == 1 && dangerous_areas.shape(0) == 0)) {
        if (dangerous_areas.ndim() != 2 || dangerous_areas.shape(1) != 2) {
            throw std::invalid_argument("dangerous_areas must have shape (K, 2)");
        }
        const auto k = static_cast<std::size_t>(dangerous_areas.shape(0));
        areas.reserve(k);
        auto da_view = dangerous_areas.unchecked<2>();
        for (std::size_t i = 0; i < k; ++i) {
            areas.emplace_back(da_view(static_cast<py::ssize_t>(i), 0),
                               da_view(static_cast<py::ssize_t>(i), 1));
        }
    }
    const double r_sq = dangerous_area_radius * dangerous_area_radius;
    const double tag_radius_sq = tag_radius * tag_radius;

    const auto n = static_cast<std::size_t>(states.shape(0));
    auto state_view = states.unchecked<2>();
    auto out = py::array_t<double>(static_cast<py::ssize_t>(n));
    auto buf = out.mutable_unchecked<1>();

    for (std::size_t i = 0; i < n; ++i) {
        const py::ssize_t row = static_cast<py::ssize_t>(i);
        // Terminal particles contribute zero reward.
        if (state_view(row, 4) != 0.0) {
            buf(row) = 0.0;
            continue;
        }
        double r = -step_cost;
        if (is_tag) {
            const double rx = state_view(row, 0);
            const double ry = state_view(row, 1);
            const double ox = state_view(row, 2);
            const double oy = state_view(row, 3);
            const double dx = rx - ox;
            const double dy = ry - oy;
            const double dist_sq = dx * dx + dy * dy;
            r += (dist_sq <= tag_radius_sq) ? tag_reward : -tag_penalty;
        }
        if (!areas.empty()) {
            const double rx = state_view(row, 0);
            const double ry = state_view(row, 1);
            // Match the Python ``reward_batch`` semantics: accumulate one
            // penalty per matching area (the singular ``reward`` short-
            // circuits, but reward_batch is what we're replacing).
            for (const auto &area : areas) {
                const double ddx = rx - area.first;
                const double ddy = ry - area.second;
                if (ddx * ddx + ddy * ddy <= r_sq) {
                    r -= dangerous_area_penalty;
                }
            }
        }
        buf(row) = r;
    }
    return out;
}

// Vectorised reward kernel for the discrete LaserTagPOMDP.
//
// Mirrors LaserTagPOMDP._compute_reward_batch in pure C++:
//   - terminal-flag rows yield 0.0;
//   - on the tag action (action == 4): live rows get +tag_reward when the
//     robot grid cell equals the opponent grid cell, else -tag_penalty;
//   - on a movement action (0..3): live rows pay -step_cost;
//   - the intended-position penalty (-dangerous_area_penalty) is applied
//     when the intended cell hits a wall (in-bounds + walls hit) or lies
//     within ``dangerous_area_radius`` (Euclidean) of any dangerous-area
//     centre. For action == 4 the intended cell is the current robot cell;
//     for actions 0..3 it is the robot cell shifted by ``action_directions``.
//
// dangerous_areas: shape (D, 2) float64 (or (0, 2) / 1-D length-0 for empty).
// walls_flat: 1-D int64 array of (row, col) pairs flattened (length = 2 * n_walls).
// action_directions: shape (4, 2) int64 with row r = (dr, dc) for action r.
py::array_t<double> lasertag_discrete_reward_batch(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &states,
    int action,
    int rows, int cols,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast> &walls_flat,
    int n_walls,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dangerous_areas,
    int n_dangerous,
    double dangerous_area_radius,
    double dangerous_area_penalty,
    double tag_reward,
    double tag_penalty,
    double step_cost,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast> &action_directions) {
    if (states.ndim() != 2 || states.shape(1) != static_cast<py::ssize_t>(kStateDim)) {
        throw std::invalid_argument("states must have shape (N, 5)");
    }
    if (walls_flat.ndim() != 1 ||
        walls_flat.shape(0) != static_cast<py::ssize_t>(2 * n_walls)) {
        throw std::invalid_argument("walls_flat must be a 1-D int64 array of length 2 * n_walls");
    }
    if (action_directions.ndim() != 2 || action_directions.shape(0) != 4 ||
        action_directions.shape(1) != 2) {
        throw std::invalid_argument("action_directions must have shape (4, 2)");
    }

    // Pack walls into a hash set of row * cols + col for O(1) wall hits.
    auto walls_view = walls_flat.unchecked<1>();
    std::unordered_set<int64_t> wall_cells;
    wall_cells.reserve(static_cast<std::size_t>(n_walls) * 2 + 1);
    for (int i = 0; i < n_walls; ++i) {
        const int64_t wr = walls_view(static_cast<py::ssize_t>(2 * i));
        const int64_t wc = walls_view(static_cast<py::ssize_t>(2 * i + 1));
        wall_cells.insert(wr * static_cast<int64_t>(cols) + wc);
    }

    // Pack dangerous-area centres.
    std::vector<std::pair<double, double>> areas;
    if (n_dangerous > 0) {
        if (dangerous_areas.ndim() != 2 || dangerous_areas.shape(0) != n_dangerous ||
            dangerous_areas.shape(1) != 2) {
            throw std::invalid_argument("dangerous_areas must have shape (n_dangerous, 2)");
        }
        areas.reserve(static_cast<std::size_t>(n_dangerous));
        auto da_view = dangerous_areas.unchecked<2>();
        for (int i = 0; i < n_dangerous; ++i) {
            areas.emplace_back(da_view(static_cast<py::ssize_t>(i), 0),
                               da_view(static_cast<py::ssize_t>(i), 1));
        }
    }
    const double r_sq = dangerous_area_radius * dangerous_area_radius;

    // Resolve the action's grid delta. For action 4 (tag) the intended cell is
    // the current cell, so dr/dc are zero; for 0..3 they are looked up from
    // ``action_directions``.
    int64_t dr = 0;
    int64_t dc = 0;
    if (action >= 0 && action <= 3) {
        auto ad_view = action_directions.unchecked<2>();
        dr = ad_view(action, 0);
        dc = ad_view(action, 1);
    }
    const bool is_tag = (action == 4);

    const auto n = static_cast<std::size_t>(states.shape(0));
    auto state_view = states.unchecked<2>();
    auto out = py::array_t<double>(static_cast<py::ssize_t>(n));
    auto buf = out.mutable_unchecked<1>();

    for (std::size_t i = 0; i < n; ++i) {
        const py::ssize_t row = static_cast<py::ssize_t>(i);
        if (state_view(row, 4) != 0.0) {
            buf(row) = 0.0;
            continue;
        }

        const int64_t robot_r = static_cast<int64_t>(state_view(row, 0));
        const int64_t robot_c = static_cast<int64_t>(state_view(row, 1));

        double r;
        if (is_tag) {
            const int64_t opp_r = static_cast<int64_t>(state_view(row, 2));
            const int64_t opp_c = static_cast<int64_t>(state_view(row, 3));
            r = (robot_r == opp_r && robot_c == opp_c) ? tag_reward : -tag_penalty;
        } else {
            r = -step_cost;
        }

        const int64_t int_r = robot_r + dr;
        const int64_t int_c = robot_c + dc;

        bool penalty = false;
        if (int_r >= 0 && int_r < static_cast<int64_t>(rows) &&
            int_c >= 0 && int_c < static_cast<int64_t>(cols)) {
            const int64_t key = int_r * static_cast<int64_t>(cols) + int_c;
            if (wall_cells.find(key) != wall_cells.end()) {
                penalty = true;
            }
        }
        if (!penalty && !areas.empty()) {
            const double pr = static_cast<double>(int_r);
            const double pc = static_cast<double>(int_c);
            for (const auto &area : areas) {
                const double ddr = pr - area.first;
                const double ddc = pc - area.second;
                if (ddr * ddr + ddc * ddc <= r_sq) {
                    penalty = true;
                    break;
                }
            }
        }
        if (penalty) {
            r -= dangerous_area_penalty;
        }
        buf(row) = r;
    }
    return out;
}

// ── ContinuousLaserTag native rollout (added by perf agent) ──────────────────
//
// cont_simulate_rollout runs an entire random rollout in a single C++ frame.
// Python pre-samples the action sequence (shape (N, 3) float64); this function
// steps the environment for up to (max_depth - start_depth) steps, accumulating
// discounted reward, and returns early on a terminal state.
//
// The transition logic is shared with ContinuousLaserTagTransitionCpp::sample_into
// via cont_step_state (a module-level static helper). Reward is computed inline
// using the same formula as reward_batch.
// ─────────────────────────────────────────────────────────────────────────────

// One environment step: advance state in-place. Mirrors sample_into in
// ContinuousLaserTagTransitionCpp. Returns true if the resulting state is terminal.
static bool cont_step_state(double *state, const double *action, const EnvParams &env,
                             const pomdp_native::GaussianND<2> &robot_noise,
                             const pomdp_native::GaussianND<2> &opp_noise,
                             pomdp_native::RNGState &rng) {
    // Absorbing terminal: no change.
    if (state[4] != 0.0) {
        return true;
    }
    const double robot_x = state[0];
    const double robot_y = state[1];
    const double opp_x = state[2];
    const double opp_y = state[3];
    const double tag_flag = action[2];

    if (tag_flag > 0.5) {
        const double dx = robot_x - opp_x;
        const double dy = robot_y - opp_y;
        const double dist = std::hypot(dx, dy);
        if (dist <= env.tag_radius) {
            state[4] = 1.0;
            return true;
        }
        // Robot stays; opponent pursues.
        double mean_opp_x;
        double mean_opp_y;
        opponent_pursuit_mean(robot_x, robot_y, opp_x, opp_y, env.pursuit_speed, rng, mean_opp_x,
                              mean_opp_y);
        double new_opp_x;
        double new_opp_y;
        sample_move(mean_opp_x, mean_opp_y, env.opponent_radius, opp_noise, env, rng, new_opp_x,
                    new_opp_y);
        state[2] = new_opp_x;
        state[3] = new_opp_y;
        return false;
    }

    // Non-tag: robot moves, then opponent pursues.
    double new_robot_x;
    double new_robot_y;
    sample_move(robot_x + action[0], robot_y + action[1], env.robot_radius, robot_noise, env, rng,
                new_robot_x, new_robot_y);

    double mean_opp_x;
    double mean_opp_y;
    opponent_pursuit_mean(new_robot_x, new_robot_y, opp_x, opp_y, env.pursuit_speed, rng,
                          mean_opp_x, mean_opp_y);
    double new_opp_x;
    double new_opp_y;
    sample_move(mean_opp_x, mean_opp_y, env.opponent_radius, opp_noise, env, rng, new_opp_x,
                new_opp_y);

    state[0] = new_robot_x;
    state[1] = new_robot_y;
    state[2] = new_opp_x;
    state[3] = new_opp_y;
    state[4] = 0.0;
    return false;
}

// cont_simulate_rollout: run a full rollout in one C++ frame.
//
// Parameters:
//   initial_state      - shape (5,) float64 start state
//   actions_buffer     - shape (N, 3) float64 pre-sampled action sequence (N >= steps_left)
//   start_depth        - depth already consumed by the search tree
//   max_depth          - total rollout depth limit (inclusive, so steps = max_depth - start_depth)
//   discount_factor    - per-step gamma
//   robot_covariance   - (2, 2) float64 covariance for robot Gaussian noise
//   opponent_covariance- (2, 2) float64 covariance for opponent Gaussian noise
//   pursuit_speed      - opponent mean step magnitude
//   walls              - (M, 4) float64 AABB walls (cx, cy, hx, hy)
//   grid_size          - (2,) float64 [width, height]
//   robot_radius       - robot body radius
//   opponent_radius    - opponent body radius
//   tag_radius         - tagging success distance
//   tag_reward         - reward for a successful tag
//   tag_penalty        - penalty for a failed tag
//   step_cost          - cost per step
//   dangerous_areas    - (K, 2) float64 dangerous area centres, or empty (0,) array
//   dangerous_area_radius   - dangerous area radius
//   dangerous_area_penalty  - penalty per dangerous area step
//
// Returns: discounted sum of rewards (scalar double)
double cont_simulate_rollout(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &initial_state,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &actions_buffer,
    int start_depth, int max_depth, double discount_factor,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &robot_covariance,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &opponent_covariance,
    double pursuit_speed,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &walls,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &grid_size,
    double robot_radius, double opponent_radius, double tag_radius, double tag_reward,
    double tag_penalty, double step_cost,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dangerous_areas,
    double dangerous_area_radius, double dangerous_area_penalty) {
    // Validate shapes.
    if (initial_state.ndim() != 1 || initial_state.shape(0) != static_cast<py::ssize_t>(kStateDim)) {
        throw std::invalid_argument("initial_state must have shape (5,)");
    }
    const int steps_left = max_depth - start_depth;
    if (steps_left <= 0) {
        return 0.0;
    }
    if (actions_buffer.ndim() != 2 || actions_buffer.shape(1) != 3) {
        throw std::invalid_argument("actions_buffer must have shape (N, 3)");
    }
    const int n_actions = static_cast<int>(actions_buffer.shape(0));
    if (n_actions < steps_left) {
        throw std::invalid_argument("actions_buffer has fewer rows than steps_left");
    }

    // Build env params.
    py::array_t<double> grid_size_arr = grid_size;
    EnvParams env = make_env_params(walls, grid_size_arr, robot_radius, opponent_radius, tag_radius,
                                    pursuit_speed);

    // Build Gaussian noise models.
    pomdp_native::GaussianND<2> robot_noise = pomdp_native::GaussianND<2>::from_covariance(robot_covariance);
    pomdp_native::GaussianND<2> opp_noise = pomdp_native::GaussianND<2>::from_covariance(opponent_covariance);

    // Build dangerous areas vector.
    std::vector<std::pair<double, double>> danger_areas_vec;
    if (!(dangerous_areas.ndim() == 1 && dangerous_areas.shape(0) == 0)) {
        if (dangerous_areas.ndim() != 2 || dangerous_areas.shape(1) != 2) {
            throw std::invalid_argument("dangerous_areas must have shape (K, 2) or empty (0,)");
        }
        const auto k = static_cast<std::size_t>(dangerous_areas.shape(0));
        danger_areas_vec.reserve(k);
        auto da_view = dangerous_areas.unchecked<2>();
        for (std::size_t i = 0; i < k; ++i) {
            danger_areas_vec.emplace_back(da_view(static_cast<py::ssize_t>(i), 0),
                                          da_view(static_cast<py::ssize_t>(i), 1));
        }
    }

    // Copy initial state.
    double state[kStateDim];  // NOLINT(modernize-avoid-c-arrays)
    auto s_view = initial_state.unchecked<1>();
    for (std::size_t d = 0; d < kStateDim; ++d) {
        state[d] = s_view(static_cast<py::ssize_t>(d));
    }

    auto act_view = actions_buffer.unchecked<2>();
    pomdp_native::RNGState &rng = pomdp_native::default_rng();

    double total = 0.0;
    double gamma_power = 1.0;
    const double da_r_sq = dangerous_area_radius * dangerous_area_radius;

    for (int step = 0; step < steps_left; ++step) {
        // Check terminal before acting.
        if (state[4] != 0.0) {
            break;
        }
        const double action[3] = {act_view(step, 0), act_view(step, 1), act_view(step, 2)};  // NOLINT

        // Compute reward for current (state, action) before stepping.
        double r = -step_cost;
        const bool is_tag = action[2] > 0.5;
        if (is_tag) {
            const double dx = state[0] - state[2];
            const double dy = state[1] - state[3];
            const double dist_sq = dx * dx + dy * dy;
            const double tag_r_sq = tag_radius * tag_radius;
            r += (dist_sq <= tag_r_sq) ? tag_reward : -tag_penalty;
        }
        if (!danger_areas_vec.empty()) {
            for (const auto &area : danger_areas_vec) {
                const double ddx = state[0] - area.first;
                const double ddy = state[1] - area.second;
                if (ddx * ddx + ddy * ddy <= da_r_sq) {
                    r -= dangerous_area_penalty;
                }
            }
        }

        total += gamma_power * r;
        gamma_power *= discount_factor;

        // Step the state.
        cont_step_state(state, action, env, robot_noise, opp_noise, rng);
    }
    return total;
}

// ── Discrete LaserTag rollout kernel ────────────────────────────────────────
//
// Encodes the full transition / reward / terminal logic for the discrete
// LaserTagPOMDP and runs a complete random-action rollout in a single C++
// frame, eliminating all Python call overhead per step.
//
// The RNG used here is ``pomdp_native::default_rng()`` (mt19937_64). Because
// the Python hot-path uses NumPy's legacy mt19937 (32-bit), the random-variate
// sequences differ for the same integer seed; equivalence tests must therefore
// compare distributions (means over many trials), not individual values.

// Action → (drow, dcol).  Index 4 (Tag) has zero displacement.
static constexpr std::array<std::array<int, 2>, 5> kDiscActDirs = {{
    {{-1, 0}}, {{1, 0}}, {{0, 1}}, {{0, -1}}, {{0, 0}}}};

struct DiscreteEnvParams {
    int rows;
    int cols;
    // wall_grid[r * cols + c] == true  iff (r, c) is a wall.
    std::vector<bool> wall_grid;
    // dangerous areas: (row, col) pairs.
    std::vector<std::pair<double, double>> dangerous_areas;
    double dangerous_area_radius_sq;
    double dangerous_area_penalty;
    double tag_reward;
    double tag_penalty;
    double step_cost;
    double transition_error_prob;
};

DiscreteEnvParams make_discrete_env_params(
    int rows, int cols,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> &walls_flat,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dangerous_areas_arr,
    double dangerous_area_radius, double dangerous_area_penalty,
    double tag_reward, double tag_penalty, double step_cost,
    double transition_error_prob) {

    DiscreteEnvParams p;
    p.rows = rows;
    p.cols = cols;
    p.wall_grid.assign(static_cast<std::size_t>(rows * cols), false);

    // walls_flat: 1-D int64 array of length 2*M: [r0, c0, r1, c1, ...]
    if (walls_flat.ndim() != 1) {
        throw std::invalid_argument("walls_flat must be a 1-D int64 array");
    }
    const auto wlen = static_cast<std::size_t>(walls_flat.shape(0));
    if (wlen % 2 != 0) {
        throw std::invalid_argument("walls_flat length must be even");
    }
    auto wview = walls_flat.unchecked<1>();
    for (std::size_t i = 0; i < wlen; i += 2) {
        const int wr = static_cast<int>(wview(static_cast<py::ssize_t>(i)));
        const int wc = static_cast<int>(wview(static_cast<py::ssize_t>(i + 1)));
        if (wr >= 0 && wr < rows && wc >= 0 && wc < cols) {
            p.wall_grid[static_cast<std::size_t>(wr * cols + wc)] = true;
        }
    }

    // dangerous_areas: 2-D (K, 2) float64, or 1-D empty.
    if (!(dangerous_areas_arr.ndim() == 1 && dangerous_areas_arr.shape(0) == 0)) {
        if (dangerous_areas_arr.ndim() != 2 || dangerous_areas_arr.shape(1) != 2) {
            throw std::invalid_argument("dangerous_areas must have shape (K, 2) or be empty");
        }
        const auto k = static_cast<std::size_t>(dangerous_areas_arr.shape(0));
        auto da_view = dangerous_areas_arr.unchecked<2>();
        p.dangerous_areas.reserve(k);
        for (std::size_t i = 0; i < k; ++i) {
            p.dangerous_areas.emplace_back(
                da_view(static_cast<py::ssize_t>(i), 0),
                da_view(static_cast<py::ssize_t>(i), 1));
        }
    }

    p.dangerous_area_radius_sq = dangerous_area_radius * dangerous_area_radius;
    p.dangerous_area_penalty = dangerous_area_penalty;
    p.tag_reward = tag_reward;
    p.tag_penalty = tag_penalty;
    p.step_cost = step_cost;
    p.transition_error_prob = transition_error_prob;
    return p;
}

inline bool disc_is_valid(int r, int c, const DiscreteEnvParams &env) {
    if (r < 0 || r >= env.rows || c < 0 || c >= env.cols) {
        return false;
    }
    return !env.wall_grid[static_cast<std::size_t>(r * env.cols + c)];
}

inline bool disc_is_dangerous(int r, int c, const DiscreteEnvParams &env) {
    for (const auto &area : env.dangerous_areas) {
        const double dr = static_cast<double>(r) - area.first;
        const double dc = static_cast<double>(c) - area.second;
        if (dr * dr + dc * dc <= env.dangerous_area_radius_sq) {
            return true;
        }
    }
    return false;
}

double disc_reward(const double *state, int action, const DiscreteEnvParams &env) {
    if (state[4] != 0.0) {
        return 0.0;
    }
    const int robot_r = static_cast<int>(state[0]);
    const int robot_c = static_cast<int>(state[1]);
    const int opp_r   = static_cast<int>(state[2]);
    const int opp_c   = static_cast<int>(state[3]);

    double base;
    int int_r;
    int int_c;
    if (action == 4) {
        base = (robot_r == opp_r && robot_c == opp_c)
                   ? env.tag_reward
                   : -env.tag_penalty;
        int_r = robot_r;
        int_c = robot_c;
    } else {
        base = -env.step_cost;
        int_r = robot_r + kDiscActDirs[static_cast<std::size_t>(action)][0];
        int_c = robot_c + kDiscActDirs[static_cast<std::size_t>(action)][1];
    }

    const bool intended_in_bounds =
        (int_r >= 0 && int_r < env.rows && int_c >= 0 && int_c < env.cols);
    const bool is_wall =
        intended_in_bounds &&
        env.wall_grid[static_cast<std::size_t>(int_r * env.cols + int_c)];
    if (is_wall || disc_is_dangerous(int_r, int_c, env)) {
        base -= env.dangerous_area_penalty;
    }
    return base;
}

std::pair<int, int> disc_sample_opponent_move(
    int opp_r, int opp_c, int robot_r, int robot_c,
    const DiscreteEnvParams &env, pomdp_native::RNGState &rng) {

    struct Candidate {
        int r;
        int c;
        double prob;
    };
    std::array<Candidate, 5> cands{};
    std::size_t n_cands = 0;

    auto add_cand = [&](int r, int c, double prob) {
        if (prob > 0.0 && disc_is_valid(r, c, env)) {
            cands[n_cands++] = {r, c, prob};
        }
    };

    // x-moves (column direction, fixed row = opp_r)
    if (robot_c == opp_c) {
        add_cand(opp_r, opp_c + 1, 0.2);
        add_cand(opp_r, opp_c - 1, 0.2);
    } else {
        const int toward_c = (robot_c > opp_c) ? opp_c + 1 : opp_c - 1;
        add_cand(opp_r, toward_c, 0.4);
    }

    // y-moves (row direction, fixed col = opp_c)
    if (robot_r == opp_r) {
        add_cand(opp_r + 1, opp_c, 0.2);
        add_cand(opp_r - 1, opp_c, 0.2);
    } else {
        const int toward_r = (robot_r > opp_r) ? opp_r + 1 : opp_r - 1;
        add_cand(toward_r, opp_c, 0.4);
    }

    // Compute actual_total including the base stay probability 0.2.
    double actual_total = 0.2;
    for (std::size_t i = 0; i < n_cands; ++i) {
        actual_total += cands[i].prob;
    }
    const double stay_prob =
        (actual_total < 1.0) ? 0.2 + (1.0 - actual_total) : 0.2;
    cands[n_cands++] = {opp_r, opp_c, stay_prob};

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double u = uniform(rng.engine());
    double cum = 0.0;
    for (std::size_t i = 0; i < n_cands; ++i) {
        cum += cands[i].prob;
        if (u < cum) {
            return {cands[i].r, cands[i].c};
        }
    }
    return {cands[n_cands - 1].r, cands[n_cands - 1].c};
}

// Run a complete random-action rollout from initial_state for the discrete
// LaserTagPOMDP.  All RNG draws use ``pomdp_native::default_rng()``.
//
// Parameters:
//   initial_state           : (5,) float64 [robot_r, robot_c, opp_r, opp_c, terminal]
//   max_depth               : maximum total rollout depth
//   discount                : discount factor gamma
//   initial_depth           : depth already consumed by the search tree
//   rows, cols              : grid dimensions
//   walls_flat              : 1-D int64 array [r0,c0, r1,c1, ...] of wall cells
//   dangerous_areas         : (K,2) float64 area centers, or 1-D empty
//   dangerous_area_radius   : Euclidean radius of each dangerous area
//   dangerous_area_penalty  : penalty applied when intended position is in any area
//   tag_reward              : reward for a successful tag
//   tag_penalty             : penalty for a failed tag
//   step_cost               : cost per movement step
//   transition_error_prob   : probability of random movement error
//
// Returns: discounted sum of immediate rewards.
double simulate_rollout_discrete(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &initial_state,
    int max_depth, double discount, int initial_depth,
    int rows, int cols,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> &walls_flat,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dangerous_areas,
    double dangerous_area_radius, double dangerous_area_penalty,
    double tag_reward, double tag_penalty, double step_cost,
    double transition_error_prob) {

    if (initial_state.ndim() != 1 || initial_state.shape(0) != 5) {
        throw std::invalid_argument("initial_state must have shape (5,)");
    }

    const DiscreteEnvParams env = make_discrete_env_params(
        rows, cols, walls_flat, dangerous_areas,
        dangerous_area_radius, dangerous_area_penalty,
        tag_reward, tag_penalty, step_cost, transition_error_prob);

    pomdp_native::RNGState &rng = pomdp_native::default_rng();
    std::uniform_int_distribution<int> action_dist(0, 4);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    auto sv = initial_state.unchecked<1>();
    double state[5] = {sv(0), sv(1), sv(2), sv(3), sv(4)};  // NOLINT(modernize-avoid-c-arrays)

    double total = 0.0;
    double gamma_power = 1.0;
    int depth = initial_depth;

    while (depth < max_depth && state[4] == 0.0) {
        const int action = action_dist(rng.engine());

        // Reward computed on current state before transition.
        total += gamma_power * disc_reward(state, action, env);
        gamma_power *= discount;

        // Actual action (with possible error on movement actions).
        int actual_action = action;
        if (action != 4 && transition_error_prob > 0.0 &&
                uniform(rng.engine()) < transition_error_prob) {
            std::uniform_int_distribution<int> err_dist(0, 2);
            int err_idx = err_dist(rng.engine());
            int count = 0;
            for (int a = 0; a < 4; ++a) {
                if (a == action) {
                    continue;
                }
                if (count == err_idx) {
                    actual_action = a;
                    break;
                }
                ++count;
            }
        }

        // Robot next position.
        const int robot_r = static_cast<int>(state[0]);
        const int robot_c = static_cast<int>(state[1]);
        int new_robot_r = robot_r;
        int new_robot_c = robot_c;
        if (actual_action != 4) {
            const int cand_r = robot_r + kDiscActDirs[static_cast<std::size_t>(actual_action)][0];
            const int cand_c = robot_c + kDiscActDirs[static_cast<std::size_t>(actual_action)][1];
            if (disc_is_valid(cand_r, cand_c, env)) {
                new_robot_r = cand_r;
                new_robot_c = cand_c;
            }
        }

        const int opp_r = static_cast<int>(state[2]);
        const int opp_c = static_cast<int>(state[3]);

        // Successful tag: robot and opponent at same cell.
        if (actual_action == 4 && robot_r == opp_r && robot_c == opp_c) {
            state[0] = static_cast<double>(new_robot_r);
            state[1] = static_cast<double>(new_robot_c);
            state[4] = 1.0;
            break;
        }

        // Sample opponent move.
        auto [new_opp_r, new_opp_c] = disc_sample_opponent_move(
            opp_r, opp_c, new_robot_r, new_robot_c, env, rng);

        state[0] = static_cast<double>(new_robot_r);
        state[1] = static_cast<double>(new_robot_c);
        state[2] = static_cast<double>(new_opp_r);
        state[3] = static_cast<double>(new_opp_c);

        ++depth;
    }

    return total;
}

// ── Discrete LaserTag belief-update kernels ─────────────────────────────────
//
// Native ports of the four hot helpers in
// ``laser_tag_vectorized_updater.py``:
//   - _batch_is_valid (inlined into both kernels via valid_cell lookup)
//   - _batch_opponent_move (sampled inline in transition kernel)
//   - _batch_laser_measurements (inlined into obs kernel via wall_dist_table)
//   - _compute_opponent_distance_on_ray (inlined into obs kernel)
//
// The transition kernel preserves the Python helper's draw order: one bulk
// uniform per non-terminal-particle (opponent move), plus the
// transition-error sampling (one uniform per particle when probability > 0,
// then one categorical-of-3 per error-flagged particle). RNG state is the
// shared module-level mt19937_64; tests for the Python-level updater
// already only check distributions over many samples, not bit-by-bit RNG
// equality, so swapping NumPy's PRNG for the C++ RNG is safe.

// 8 laser-direction (drow, dcol) vectors, matching _LASER_DIRECTIONS in
// laser_tag_vectorized_updater.py.
constexpr std::array<std::array<int, 2>, 8> kBeliefLaserDirections = {{
    {{-1, 0}}, {{-1, 1}}, {{0, 1}}, {{1, 1}},
    {{1, 0}}, {{1, -1}}, {{0, -1}}, {{-1, -1}},
}};

// 5 action-direction (drow, dcol) vectors: N, S, E, W, Tag (Tag = no-op).
constexpr std::array<std::array<int, 2>, 5> kBeliefActionDirections = {{
    {{-1, 0}}, {{1, 0}}, {{0, 1}}, {{0, -1}}, {{0, 0}},
}};

inline bool belief_is_valid(int r, int c, int rows, int cols, const std::uint8_t *valid_cell) {
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        return false;
    }
    return valid_cell[r * cols + c] != 0;
}

// Compute the opponent's next position by sampling from the 5-way
// categorical distribution defined in
// LaserTagVectorizedUpdater._batch_opponent_move, using a single uniform
// draw u in [0, 1).  The 5 categorical bins (right, left, up, down, stay)
// match the Python cumulative thresholds (cum1, cum2, cum3, cum4, 1.0).
void belief_sample_opponent_move(int robot_r, int robot_c, int opp_r, int opp_c,
                                  int rows, int cols, const std::uint8_t *valid_cell,
                                  double u, int *out_r, int *out_c) {
    const bool right_valid = belief_is_valid(opp_r, opp_c + 1, rows, cols, valid_cell);
    const bool left_valid = belief_is_valid(opp_r, opp_c - 1, rows, cols, valid_cell);
    const bool up_valid = belief_is_valid(opp_r - 1, opp_c, rows, cols, valid_cell);
    const bool down_valid = belief_is_valid(opp_r + 1, opp_c, rows, cols, valid_cell);

    const bool same_col = (robot_c == opp_c);
    const bool same_row = (robot_r == opp_r);

    double right_prob = 0.0;
    if (same_col && right_valid) {
        right_prob = 0.2;
    } else if (robot_c > opp_c && right_valid) {
        right_prob = 0.4;
    }
    double left_prob = 0.0;
    if (same_col && left_valid) {
        left_prob = 0.2;
    } else if (robot_c < opp_c && left_valid) {
        left_prob = 0.4;
    }
    double up_prob = 0.0;
    if (same_row && up_valid) {
        up_prob = 0.2;
    } else if (robot_r < opp_r && up_valid) {
        up_prob = 0.4;
    }
    double down_prob = 0.0;
    if (same_row && down_valid) {
        down_prob = 0.2;
    } else if (robot_r > opp_r && down_valid) {
        down_prob = 0.4;
    }

    const double cum1 = right_prob;
    const double cum2 = cum1 + left_prob;
    const double cum3 = cum2 + up_prob;
    const double cum4 = cum3 + down_prob;

    if (u < cum1) {
        *out_r = opp_r;
        *out_c = opp_c + 1;
    } else if (u < cum2) {
        *out_r = opp_r;
        *out_c = opp_c - 1;
    } else if (u < cum3) {
        *out_r = opp_r - 1;
        *out_c = opp_c;
    } else if (u < cum4) {
        *out_r = opp_r + 1;
        *out_c = opp_c;
    } else {
        *out_r = opp_r;
        *out_c = opp_c;
    }
}

// Apply a single (intended) movement-or-tag action to all live particles.
//
// ``out`` must be a writable (N, 5) buffer (caller pre-fills it with the
// input particles).  Live (non-terminal) particles are updated in place.
// Uses one uniform draw per live particle for the opponent move.
void belief_apply_action(int action_idx, std::size_t n,
                          const double *particles, double *out,
                          int rows, int cols, const std::uint8_t *valid_cell,
                          pomdp_native::RNGState &rng) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const int dr = kBeliefActionDirections[static_cast<std::size_t>(action_idx)][0];
    const int dc = kBeliefActionDirections[static_cast<std::size_t>(action_idx)][1];

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t off = i * 5;
        // Copy through (terminal handled by caller; we still mirror values).
        out[off + 0] = particles[off + 0];
        out[off + 1] = particles[off + 1];
        out[off + 2] = particles[off + 2];
        out[off + 3] = particles[off + 3];
        out[off + 4] = particles[off + 4];

        if (particles[off + 4] != 0.0) {
            continue;
        }

        const int robot_r = static_cast<int>(particles[off + 0]);
        const int robot_c = static_cast<int>(particles[off + 1]);
        const int opp_r = static_cast<int>(particles[off + 2]);
        const int opp_c = static_cast<int>(particles[off + 3]);

        // Robot move: tag (action 4) leaves robot in place; movement
        // actions move iff the candidate cell is valid.
        int new_robot_r;
        int new_robot_c;
        if (action_idx == 4) {
            new_robot_r = robot_r;
            new_robot_c = robot_c;
        } else {
            const int cand_r = robot_r + dr;
            const int cand_c = robot_c + dc;
            if (belief_is_valid(cand_r, cand_c, rows, cols, valid_cell)) {
                new_robot_r = cand_r;
                new_robot_c = cand_c;
            } else {
                new_robot_r = robot_r;
                new_robot_c = robot_c;
            }
        }

        // Tag at opponent cell: terminal, opponent does NOT move.
        if (action_idx == 4 && new_robot_r == opp_r && new_robot_c == opp_c) {
            out[off + 0] = static_cast<double>(new_robot_r);
            out[off + 1] = static_cast<double>(new_robot_c);
            out[off + 2] = static_cast<double>(opp_r);
            out[off + 3] = static_cast<double>(opp_c);
            out[off + 4] = 1.0;
            continue;
        }

        const double u = uniform(rng.engine());
        int new_opp_r;
        int new_opp_c;
        belief_sample_opponent_move(new_robot_r, new_robot_c, opp_r, opp_c,
                                     rows, cols, valid_cell, u,
                                     &new_opp_r, &new_opp_c);

        out[off + 0] = static_cast<double>(new_robot_r);
        out[off + 1] = static_cast<double>(new_robot_c);
        out[off + 2] = static_cast<double>(new_opp_r);
        out[off + 3] = static_cast<double>(new_opp_c);
        out[off + 4] = 0.0;
    }
}

// Native port of LaserTagVectorizedUpdater.batch_transition.
//
// Parameters:
//   particles               : (N, 5) float64 input particles
//   action_idx              : intended movement action (0..4)
//   transition_error_prob   : per-particle probability of executing a random
//                             non-intended movement instead of action_idx
//                             (only applied for action_idx in {0,1,2,3})
//   valid_cell_flat         : (rows * cols,) uint8 boolean grid where 1
//                             indicates a non-wall cell
//   rows, cols              : grid dimensions
//
// Returns: (N, 5) float64 array of next particles.
py::array_t<double> belief_batch_transition_discrete(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &particles,
    int action_idx, double transition_error_prob,
    const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> &valid_cell_flat,
    int rows, int cols) {
    if (particles.ndim() != 2 || particles.shape(1) != 5) {
        throw std::invalid_argument("particles must have shape (N, 5)");
    }
    if (action_idx < 0 || action_idx > 4) {
        throw std::invalid_argument("action_idx must be in {0,1,2,3,4}");
    }
    if (valid_cell_flat.ndim() != 1 ||
        static_cast<int>(valid_cell_flat.shape(0)) != rows * cols) {
        throw std::invalid_argument("valid_cell_flat must have shape (rows*cols,)");
    }
    const auto n = static_cast<std::size_t>(particles.shape(0));
    auto out = py::array_t<double>({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(5)});
    if (n == 0) {
        return out;
    }

    auto in_view = particles.unchecked<2>();
    auto out_view = out.mutable_unchecked<2>();
    auto vc_view = valid_cell_flat.unchecked<1>();

    // Flatten input into a contiguous double buffer for the kernels (the
    // pybind unchecked accessor is already contiguous, so we just view
    // its underlying data through .data()).
    const double *in_data = particles.data();
    double *out_data = out.mutable_data();
    const std::uint8_t *vc_data = valid_cell_flat.data();

    pomdp_native::RNGState &rng = pomdp_native::default_rng();
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    // Fast path: tag action or zero error probability → single dispatch.
    if (action_idx == 4 || transition_error_prob <= 0.0) {
        belief_apply_action(action_idx, n, in_data, out_data, rows, cols, vc_data, rng);
        // Suppress unused-variable warnings from unused views.
        (void)in_view;
        (void)out_view;
        (void)vc_view;
        return out;
    }

    // Error path: per-particle Bernoulli. Allocate four candidate buffers
    // (one per error action) and select per particle.  This matches the
    // Python ``_batch_transition_with_error`` semantics.
    std::array<std::vector<double>, 4> candidates;
    for (int a = 0; a < 4; ++a) {
        candidates[static_cast<std::size_t>(a)].assign(n * 5, 0.0);
        belief_apply_action(a, n, in_data, candidates[static_cast<std::size_t>(a)].data(),
                            rows, cols, vc_data, rng);
    }

    // For each particle, decide which action was actually executed.
    // ``other_actions`` mirrors the Python list comprehension: actions in
    // {0,1,2,3} \ {intended}.  We index this per-particle when the error
    // flag fires.
    std::array<int, 3> other_actions{};
    {
        int idx = 0;
        for (int a = 0; a < 4; ++a) {
            if (a != action_idx) {
                other_actions[static_cast<std::size_t>(idx++)] = a;
            }
        }
    }

    std::uniform_int_distribution<int> err_choice(0, 2);
    for (std::size_t i = 0; i < n; ++i) {
        const double err_u = uniform(rng.engine());
        int chosen;
        if (err_u < transition_error_prob) {
            chosen = other_actions[static_cast<std::size_t>(err_choice(rng.engine()))];
        } else {
            chosen = action_idx;
        }
        const double *src = candidates[static_cast<std::size_t>(chosen)].data() + i * 5;
        for (std::size_t d = 0; d < 5; ++d) {
            out_data[i * 5 + d] = src[d];
        }
    }
    (void)in_view;
    (void)out_view;
    (void)vc_view;
    return out;
}

// Compute opponent distance on a single ray, matching the geometry of
// LaserTagVectorizedUpdater._compute_opponent_distance_on_ray.  Returns
// -1 if the opponent is not on this ray within the wall distance.
inline int opponent_distance_on_ray(int diff_r, int diff_c, int dr, int dc, int wall_dist) {
    if (dr != 0 && dc != 0) {
        // Diagonal ray: both coordinates must equal the same positive integer step.
        if (dr * diff_r <= 0 || dc * diff_c <= 0) {
            return -1;
        }
        const int step_r = diff_r / dr;
        const int step_c = diff_c / dc;
        if (step_r != step_c) {
            return -1;
        }
        if (diff_r != step_r * dr || diff_c != step_c * dc) {
            return -1;
        }
        if (step_r < 1 || (step_r - 1) > wall_dist) {
            return -1;
        }
        return step_r - 1;
    }
    if (dr != 0) {
        if (diff_c != 0) {
            return -1;
        }
        if (dr * diff_r <= 0) {
            return -1;
        }
        const int step = diff_r / dr;
        if (step < 1 || (step - 1) > wall_dist) {
            return -1;
        }
        return step - 1;
    }
    // dc != 0
    if (diff_r != 0) {
        return -1;
    }
    if (dc * diff_c <= 0) {
        return -1;
    }
    const int step = diff_c / dc;
    if (step < 1 || (step - 1) > wall_dist) {
        return -1;
    }
    return step - 1;
}

// Compute the 8-direction laser measurement vector for a single particle.
// ``wall_dist_table`` is a flat (rows * cols * 8) int32 buffer; for cell
// (r, c) and direction d the wall distance is at ``r * cols * 8 + c * 8 + d``.
inline void belief_compute_laser_measurements(int robot_r, int robot_c, int opp_r, int opp_c,
                                                const std::int32_t *wall_dist_table, int cols,
                                                double *out) {
    const int diff_r = opp_r - robot_r;
    const int diff_c = opp_c - robot_c;
    const std::int32_t *wall_row = wall_dist_table + (robot_r * cols + robot_c) * 8;
    for (std::size_t d = 0; d < 8; ++d) {
        const int wall_dist = static_cast<int>(wall_row[d]);
        const int dr = kBeliefLaserDirections[d][0];
        const int dc = kBeliefLaserDirections[d][1];
        const int opp_dist = opponent_distance_on_ray(diff_r, diff_c, dr, dc, wall_dist);
        if (opp_dist >= 0 && opp_dist < wall_dist) {
            out[d] = static_cast<double>(opp_dist);
        } else {
            out[d] = static_cast<double>(wall_dist);
        }
    }
}

// Native port of
// LaserTagVectorizedUpdater.batch_observation_log_likelihood.
//
// Parameters:
//   next_particles    : (N, 5) float64 input particles (post-transition)
//   observation       : (8,) float64 observation vector (terminal = all -1)
//   wall_dist_table_flat : (rows * cols * 8,) int32 precomputed table
//   rows, cols        : grid dimensions
//   log_norm_1d       : -0.5 * log(2 * pi * variance)
//   inv_2var          : 0.5 / variance
//
// Returns: (N,) float64 log-likelihoods. Matches the semantics of
// _batch_non_terminal_log_likelihood combined with the terminal handling
// in batch_observation_log_likelihood.
py::array_t<double> belief_batch_obs_log_likelihood_discrete(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &next_particles,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &observation,
    const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> &wall_dist_table_flat,
    int rows, int cols, double log_norm_1d, double inv_2var) {
    if (next_particles.ndim() != 2 || next_particles.shape(1) != 5) {
        throw std::invalid_argument("next_particles must have shape (N, 5)");
    }
    if (observation.ndim() != 1 || observation.shape(0) != 8) {
        throw std::invalid_argument("observation must have shape (8,)");
    }
    if (wall_dist_table_flat.ndim() != 1 ||
        static_cast<int>(wall_dist_table_flat.shape(0)) != rows * cols * 8) {
        throw std::invalid_argument(
            "wall_dist_table_flat must have shape (rows*cols*8,)");
    }

    const auto n = static_cast<std::size_t>(next_particles.shape(0));
    auto out = py::array_t<double>(static_cast<py::ssize_t>(n));
    if (n == 0) {
        return out;
    }

    const double *part_data = next_particles.data();
    const double *obs_data = observation.data();
    const std::int32_t *wall_data = wall_dist_table_flat.data();
    double *out_data = out.mutable_data();

    bool obs_is_terminal = true;
    for (std::size_t d = 0; d < 8; ++d) {
        if (obs_data[d] != -1.0) {
            obs_is_terminal = false;
            break;
        }
    }

    const double neg_inf = -std::numeric_limits<double>::infinity();
    const double obs_const = 8.0 * log_norm_1d;

    for (std::size_t i = 0; i < n; ++i) {
        const double terminal_flag = part_data[i * 5 + 4];
        if (obs_is_terminal) {
            out_data[i] = (terminal_flag == 1.0) ? 0.0 : neg_inf;
            continue;
        }
        if (terminal_flag == 1.0) {
            out_data[i] = neg_inf;
            continue;
        }
        const int robot_r = static_cast<int>(part_data[i * 5 + 0]);
        const int robot_c = static_cast<int>(part_data[i * 5 + 1]);
        const int opp_r = static_cast<int>(part_data[i * 5 + 2]);
        const int opp_c = static_cast<int>(part_data[i * 5 + 3]);

        double measurements[8];  // NOLINT(modernize-avoid-c-arrays)
        belief_compute_laser_measurements(robot_r, robot_c, opp_r, opp_c, wall_data, cols,
                                            measurements);

        double sq = 0.0;
        for (std::size_t d = 0; d < 8; ++d) {
            const double diff = obs_data[d] - measurements[d];
            sq += diff * diff;
        }
        out_data[i] = obs_const - sq * inv_2var;
    }
    return out;
}

// ── Discrete LaserTag single-step kernels ───────────────────────────────────
//
// These helpers expose the per-step transition / observation / observation-log
// probability math used by ``simulate_rollout_discrete`` so the Python single-
// state hot path (``sample_next_state``, ``sample_observation``,
// ``observation_log_probability``) can call into C++ without duplicating
// implementation logic.
//
// To preserve byte-for-byte numpy RNG reproducibility, the transition /
// observation kernels do NOT draw randomness internally; the Python caller
// pre-draws the required uniforms / normals using ``np.random.*`` and forwards
// the values here. C++ then deterministically computes the next state /
// observation using these pre-drawn samples.

// Cumulative sample: given probabilities (sum to 1) and a uniform draw u in
// [0, 1), return the index where u falls in the cumulative distribution.
// Mirrors numpy.random.choice behavior with size=1 and p=probs.
inline std::size_t cumulative_sample(const double *probs, std::size_t n, double u) {
    double cum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        cum += probs[i];
        if (u < cum) {
            return i;
        }
    }
    return n - 1;
}

// 8-direction laser distance scan from (robot_r, robot_c). Mirrors
// LaserTagPOMDP._laser_distance_inline / _LASER_DIRECTIONS.
// Output: out[0..7] in order N, NE, E, SE, S, SW, W, NW.
static constexpr std::array<std::array<int, 2>, 8> kDiscLaserDirs = {{
    {{-1, 0}}, {{-1, 1}}, {{0, 1}}, {{1, 1}}, {{1, 0}}, {{1, -1}}, {{0, -1}}, {{-1, -1}}}};

void disc_laser_measurements(int robot_r, int robot_c, int opp_r, int opp_c,
                             const DiscreteEnvParams &env, double *out) {
    for (std::size_t d = 0; d < 8; ++d) {
        const int dr = kDiscLaserDirs[d][0];
        const int dc = kDiscLaserDirs[d][1];
        int r = robot_r;
        int c = robot_c;
        double dist = 0.0;
        while (true) {
            r += dr;
            c += dc;
            dist += 1.0;
            if (r < 0 || r >= env.rows || c < 0 || c >= env.cols) {
                break;
            }
            if (env.wall_grid[static_cast<std::size_t>(r * env.cols + c)]) {
                break;
            }
            if (r == opp_r && c == opp_c) {
                break;
            }
        }
        out[d] = dist - 1.0;
    }
}

// Build the opponent-move probability table (positions and probabilities)
// after the robot has already moved. Mirrors
// LaserTagPOMDP._opponent_move_probabilities_inline. Returns the count.
std::size_t disc_opponent_move_table(int opp_r, int opp_c, int robot_r_after,
                                     int robot_c_after, const DiscreteEnvParams &env,
                                     int *out_rows, int *out_cols, double *out_probs) {
    std::size_t n = 0;

    auto try_add = [&](int r, int c, double prob) {
        if (prob > 0.0 && disc_is_valid(r, c, env)) {
            out_rows[n] = r;
            out_cols[n] = c;
            out_probs[n] = prob;
            ++n;
        }
    };

    // x-moves (column direction, fixed row = opp_r)
    if (robot_c_after == opp_c) {
        try_add(opp_r, opp_c + 1, 0.2);
        try_add(opp_r, opp_c - 1, 0.2);
    } else {
        const int toward_c = (robot_c_after > opp_c) ? opp_c + 1 : opp_c - 1;
        try_add(opp_r, toward_c, 0.4);
    }

    // y-moves (row direction, fixed col = opp_c)
    if (robot_r_after == opp_r) {
        try_add(opp_r + 1, opp_c, 0.2);
        try_add(opp_r - 1, opp_c, 0.2);
    } else {
        const int toward_r = (robot_r_after > opp_r) ? opp_r + 1 : opp_r - 1;
        try_add(toward_r, opp_c, 0.4);
    }

    // Stay action probability: 0.2 + slack from invalid moves.
    double actual_total = 0.2;
    for (std::size_t i = 0; i < n; ++i) {
        actual_total += out_probs[i];
    }
    const double stay_prob = (actual_total < 1.0) ? 0.2 + (1.0 - actual_total) : 0.2;
    out_rows[n] = opp_r;
    out_cols[n] = opp_c;
    out_probs[n] = stay_prob;
    ++n;
    return n;
}

// Compute the next-state for a single transition using a PRE-DRAWN uniform
// for the opponent move. The Python caller is responsible for resolving the
// actual_action (handling the optional transition error in numpy, preserving
// byte-identical RNG state with the original Python implementation), and for
// drawing ``opp_uniform`` via ``np.random.random()`` (except on the
// successful-tag short circuit where no opp draw is required).
py::array_t<double> lasertag_sample_next_state_step(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &state_arr,
    int actual_action, double opp_uniform, int rows, int cols,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> &walls_flat) {
    if (state_arr.ndim() != 1 || state_arr.shape(0) != 5) {
        throw std::invalid_argument("state must have shape (5,)");
    }
    if (actual_action < 0 || actual_action > 4) {
        throw std::invalid_argument("actual_action must be in [0, 4]");
    }
    auto sv = state_arr.unchecked<1>();

    // Build a lightweight env (no dangerous_areas / reward fields needed).
    DiscreteEnvParams env;
    env.rows = rows;
    env.cols = cols;
    env.wall_grid.assign(static_cast<std::size_t>(rows * cols), false);
    if (walls_flat.ndim() != 1) {
        throw std::invalid_argument("walls_flat must be 1-D");
    }
    const auto wlen = static_cast<std::size_t>(walls_flat.shape(0));
    if (wlen % 2 != 0) {
        throw std::invalid_argument("walls_flat length must be even");
    }
    auto wview = walls_flat.unchecked<1>();
    for (std::size_t i = 0; i < wlen; i += 2) {
        const int wr = static_cast<int>(wview(static_cast<py::ssize_t>(i)));
        const int wc = static_cast<int>(wview(static_cast<py::ssize_t>(i + 1)));
        if (wr >= 0 && wr < rows && wc >= 0 && wc < cols) {
            env.wall_grid[static_cast<std::size_t>(wr * cols + wc)] = true;
        }
    }

    const int robot_r = static_cast<int>(sv(0));
    const int robot_c = static_cast<int>(sv(1));
    const int opp_r = static_cast<int>(sv(2));
    const int opp_c = static_cast<int>(sv(3));

    // Robot's next position.
    int robot_next_r = robot_r;
    int robot_next_c = robot_c;
    if (actual_action != 4) {
        const int dr = kDiscActDirs[static_cast<std::size_t>(actual_action)][0];
        const int dc = kDiscActDirs[static_cast<std::size_t>(actual_action)][1];
        if (disc_is_valid(robot_r + dr, robot_c + dc, env)) {
            robot_next_r = robot_r + dr;
            robot_next_c = robot_c + dc;
        }
    }

    // Allocate output state.
    auto out = py::array_t<double>(5);
    auto ov = out.mutable_unchecked<1>();

    // Successful tag: terminal state, no opp draw.
    if (actual_action == 4 && robot_r == opp_r && robot_c == opp_c) {
        ov(0) = static_cast<double>(robot_next_r);
        ov(1) = static_cast<double>(robot_next_c);
        ov(2) = static_cast<double>(opp_r);
        ov(3) = static_cast<double>(opp_c);
        ov(4) = 1.0;
        return out;
    }

    // Sample opponent move via cumulative draw on the opp-move table.
    int opp_rows_buf[5];   // NOLINT(modernize-avoid-c-arrays)
    int opp_cols_buf[5];   // NOLINT(modernize-avoid-c-arrays)
    double opp_probs[5];   // NOLINT(modernize-avoid-c-arrays)
    const std::size_t n_opp = disc_opponent_move_table(
        opp_r, opp_c, robot_next_r, robot_next_c, env, opp_rows_buf, opp_cols_buf, opp_probs);
    const std::size_t pick = cumulative_sample(opp_probs, n_opp, opp_uniform);

    ov(0) = static_cast<double>(robot_next_r);
    ov(1) = static_cast<double>(robot_next_c);
    ov(2) = static_cast<double>(opp_rows_buf[pick]);
    ov(3) = static_cast<double>(opp_cols_buf[pick]);
    ov(4) = 0.0;
    return out;
}

// Compute the noisy 8-direction laser observation for a non-terminal next
// state, given a pre-drawn (8,) noise vector. Returns a tuple-compatible
// 8-element float64 ndarray. Caller must handle the terminal case.
py::array_t<double> lasertag_sample_observation_step(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &next_state_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &noise,
    int rows, int cols,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> &walls_flat) {
    if (next_state_arr.ndim() != 1 || next_state_arr.shape(0) != 5) {
        throw std::invalid_argument("next_state must have shape (5,)");
    }
    if (noise.ndim() != 1 || noise.shape(0) != 8) {
        throw std::invalid_argument("noise must have shape (8,)");
    }
    auto sv = next_state_arr.unchecked<1>();
    auto nv = noise.unchecked<1>();

    DiscreteEnvParams env;
    env.rows = rows;
    env.cols = cols;
    env.wall_grid.assign(static_cast<std::size_t>(rows * cols), false);
    auto wview = walls_flat.unchecked<1>();
    const auto wlen = static_cast<std::size_t>(walls_flat.shape(0));
    for (std::size_t i = 0; i < wlen; i += 2) {
        const int wr = static_cast<int>(wview(static_cast<py::ssize_t>(i)));
        const int wc = static_cast<int>(wview(static_cast<py::ssize_t>(i + 1)));
        if (wr >= 0 && wr < rows && wc >= 0 && wc < cols) {
            env.wall_grid[static_cast<std::size_t>(wr * cols + wc)] = true;
        }
    }

    const int robot_r = static_cast<int>(sv(0));
    const int robot_c = static_cast<int>(sv(1));
    const int opp_r = static_cast<int>(sv(2));
    const int opp_c = static_cast<int>(sv(3));

    double truth[8];  // NOLINT(modernize-avoid-c-arrays)
    disc_laser_measurements(robot_r, robot_c, opp_r, opp_c, env, truth);

    auto out = py::array_t<double>(8);
    auto ov = out.mutable_unchecked<1>();
    for (std::size_t d = 0; d < 8; ++d) {
        const double noisy = truth[d] + nv(static_cast<py::ssize_t>(d));
        ov(static_cast<py::ssize_t>(d)) = std::max(0.0, noisy);
    }
    return out;
}

// Single-state observation log-probability over a list/array of observations.
// Mirrors LaserTagPOMDP.observation_log_probability exactly.
py::array_t<double> lasertag_observation_log_probability_step(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &next_state_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &observations_arr,
    double measurement_noise, int rows, int cols,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> &walls_flat) {
    if (next_state_arr.ndim() != 1 || next_state_arr.shape(0) != 5) {
        throw std::invalid_argument("next_state must have shape (5,)");
    }
    if (observations_arr.ndim() != 2 || observations_arr.shape(1) != 8) {
        throw std::invalid_argument("observations must have shape (N, 8)");
    }
    auto sv = next_state_arr.unchecked<1>();
    auto ov = observations_arr.unchecked<2>();

    const auto n = static_cast<std::size_t>(observations_arr.shape(0));
    auto out = py::array_t<double>(static_cast<py::ssize_t>(n));
    auto out_view = out.mutable_unchecked<1>();

    const double neg_inf = -std::numeric_limits<double>::infinity();

    // Terminal next_state: observation must equal sentinel (-1, ..., -1).
    if (sv(4) != 0.0) {
        for (std::size_t i = 0; i < n; ++i) {
            bool matches = true;
            for (std::size_t d = 0; d < 8; ++d) {
                if (std::abs(ov(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(d)) - (-1.0)) >
                    1e-12) {
                    matches = false;
                    break;
                }
            }
            // log(1.0) = 0.0; log(0.0) = -inf
            out_view(static_cast<py::ssize_t>(i)) = matches ? 0.0 : neg_inf;
        }
        return out;
    }

    // Non-terminal: 8-dim Gaussian log-pdf with shared variance.
    DiscreteEnvParams env;
    env.rows = rows;
    env.cols = cols;
    env.wall_grid.assign(static_cast<std::size_t>(rows * cols), false);
    auto wview = walls_flat.unchecked<1>();
    const auto wlen = static_cast<std::size_t>(walls_flat.shape(0));
    for (std::size_t i = 0; i < wlen; i += 2) {
        const int wr = static_cast<int>(wview(static_cast<py::ssize_t>(i)));
        const int wc = static_cast<int>(wview(static_cast<py::ssize_t>(i + 1)));
        if (wr >= 0 && wr < rows && wc >= 0 && wc < cols) {
            env.wall_grid[static_cast<std::size_t>(wr * cols + wc)] = true;
        }
    }
    const int robot_r = static_cast<int>(sv(0));
    const int robot_c = static_cast<int>(sv(1));
    const int opp_r = static_cast<int>(sv(2));
    const int opp_c = static_cast<int>(sv(3));

    double truth[8];  // NOLINT(modernize-avoid-c-arrays)
    disc_laser_measurements(robot_r, robot_c, opp_r, opp_c, env, truth);

    const double variance = measurement_noise * measurement_noise;
    const double inv_2var = 0.5 / variance;
    const double log_norm_const = -0.5 * std::log(2.0 * M_PI * variance);

    for (std::size_t i = 0; i < n; ++i) {
        double log_prob = 0.0;
        bool any_negative = false;
        for (std::size_t d = 0; d < 8; ++d) {
            const double observed = ov(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(d));
            if (observed < 0.0) {
                any_negative = true;
                break;
            }
            const double diff = observed - truth[d];
            log_prob += log_norm_const - diff * diff * inv_2var;
        }
        out_view(static_cast<py::ssize_t>(i)) = any_negative ? neg_inf : log_prob;
    }
    return out;
}

}  // anonymous namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native (C++) sampling hot path for Continuous LaserTag POMDP.";

    m.def("set_seed", &pomdp_native::set_default_seed, py::arg("seed"),
          "Seed the module-level RNG used by sample() / batch_sample() / batch_log_likelihood().");

    m.def("reward_batch", &reward_batch, py::arg("states"), py::arg("action"),
          py::arg("tag_radius"), py::arg("tag_reward"), py::arg("tag_penalty"),
          py::arg("step_cost"), py::arg("dangerous_areas"), py::arg("dangerous_area_radius"),
          py::arg("dangerous_area_penalty"),
          "Vectorised reward computation: returns shape (N,) float64. See "
          "ContinuousLaserTagPOMDP.reward_batch for semantics.");

    m.def("lasertag_discrete_reward_batch", &lasertag_discrete_reward_batch,
          py::arg("states"), py::arg("action"),
          py::arg("rows"), py::arg("cols"),
          py::arg("walls_flat"), py::arg("n_walls"),
          py::arg("dangerous_areas"), py::arg("n_dangerous"),
          py::arg("dangerous_area_radius"), py::arg("dangerous_area_penalty"),
          py::arg("tag_reward"), py::arg("tag_penalty"), py::arg("step_cost"),
          py::arg("action_directions"),
          "Vectorised reward computation for the discrete LaserTagPOMDP. "
          "Returns shape (N,) float64. Mirrors LaserTagPOMDP._compute_reward_batch "
          "semantics: terminal rows yield 0, action 4 yields +tag_reward / "
          "-tag_penalty, actions 0..3 pay -step_cost, and the intended cell "
          "(robot + action_directions[action] for actions 0..3, robot for tag) "
          "incurs -dangerous_area_penalty when it hits a wall (in-bounds) or "
          "lies within dangerous_area_radius of any dangerous-area centre.");

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
        .def("set_state", &ContinuousLaserTagTransitionCpp::set_state, py::arg("state"))
        .def_property_readonly("state", &ContinuousLaserTagTransitionCpp::state_property)
        .def_property_readonly("action", &ContinuousLaserTagTransitionCpp::action_property);

    py::class_<ContinuousLaserTagObservationCpp>(m, "ContinuousLaserTagObservationCpp")
        .def(py::init<const py::object &, const py::object &, double,
                      const py::array_t<double> &, const py::array_t<double> &, double>(),
             py::arg("next_state"), py::arg("action"), py::arg("measurement_noise"),
             py::arg("walls"), py::arg("grid_size"), py::arg("opponent_radius"))
        .def("sample", &ContinuousLaserTagObservationCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &ContinuousLaserTagObservationCpp::probability, py::arg("values"))
        .def("log_probability", &ContinuousLaserTagObservationCpp::log_probability,
             py::arg("values"))
        .def("batch_log_likelihood", &ContinuousLaserTagObservationCpp::batch_log_likelihood,
             py::arg("next_particles"), py::arg("observation"))
        .def("set_next_state", &ContinuousLaserTagObservationCpp::set_next_state,
             py::arg("next_state"))
        .def_property_readonly("next_state", &ContinuousLaserTagObservationCpp::next_state_property)
        .def_property_readonly("action", &ContinuousLaserTagObservationCpp::action_property)
        .def_property_readonly("mean", &ContinuousLaserTagObservationCpp::mean_property);

    // ── ContinuousLaserTag native rollout binding (added by perf agent) ──────
    m.def(
        "cont_simulate_rollout", &cont_simulate_rollout,
        py::arg("initial_state"), py::arg("actions_buffer"), py::arg("start_depth"),
        py::arg("max_depth"), py::arg("discount_factor"), py::arg("robot_covariance"),
        py::arg("opponent_covariance"), py::arg("pursuit_speed"), py::arg("walls"),
        py::arg("grid_size"), py::arg("robot_radius"), py::arg("opponent_radius"),
        py::arg("tag_radius"), py::arg("tag_reward"), py::arg("tag_penalty"),
        py::arg("step_cost"), py::arg("dangerous_areas"), py::arg("dangerous_area_radius"),
        py::arg("dangerous_area_penalty"),
        "Run a full random rollout for ContinuousLaserTagPOMDP in one C++ frame.\n\n"
        "``actions_buffer`` must be shape (N, 3) float64 with N >= max_depth - start_depth.\n"
        "Returns the discounted sum of immediate rewards along the sampled trajectory.");

    // ── Discrete LaserTag native rollout binding ──────────────────────────────
    m.def(
        "simulate_rollout_discrete", &simulate_rollout_discrete,
        py::arg("initial_state"), py::arg("max_depth"), py::arg("discount"),
        py::arg("initial_depth"),
        py::arg("rows"), py::arg("cols"),
        py::arg("walls_flat"), py::arg("dangerous_areas"),
        py::arg("dangerous_area_radius"), py::arg("dangerous_area_penalty"),
        py::arg("tag_reward"), py::arg("tag_penalty"), py::arg("step_cost"),
        py::arg("transition_error_prob"),
        "Run a full random-action rollout for the discrete LaserTagPOMDP in one C++ frame.\n\n"
        "Actions are drawn uniformly from {0,1,2,3,4} using pomdp_native::default_rng().\n"
        "Seed via set_seed() before calling to obtain reproducible trajectories.\n"
        "Returns the discounted sum of immediate rewards along the sampled trajectory.");

    // ── Discrete LaserTag belief-update kernels ─────────────────────────────
    m.def(
        "belief_batch_transition_discrete", &belief_batch_transition_discrete,
        py::arg("particles"), py::arg("action_idx"), py::arg("transition_error_prob"),
        py::arg("valid_cell_flat"), py::arg("rows"), py::arg("cols"),
        "Native port of LaserTagVectorizedUpdater.batch_transition.\n\n"
        "Returns the (N, 5) float64 array of next particles.  Uses\n"
        "pomdp_native::default_rng(); seed via set_seed() for reproducibility.");

    m.def(
        "belief_batch_obs_log_likelihood_discrete",
        &belief_batch_obs_log_likelihood_discrete,
        py::arg("next_particles"), py::arg("observation"),
        py::arg("wall_dist_table_flat"), py::arg("rows"), py::arg("cols"),
        py::arg("log_norm_1d"), py::arg("inv_2var"),
        "Native port of\n"
        "LaserTagVectorizedUpdater.batch_observation_log_likelihood.\n\n"
        "Returns the (N,) float64 array of per-particle log-likelihoods.");

    // ── Discrete LaserTag single-step kernels ───────────────────────────────
    m.def(
        "sample_next_state_step", &lasertag_sample_next_state_step,
        py::arg("state"), py::arg("actual_action"), py::arg("opp_uniform"),
        py::arg("rows"), py::arg("cols"), py::arg("walls_flat"),
        "Single-step transition for the discrete LaserTagPOMDP.\n\n"
        "Caller must resolve the actual_action via numpy (handling the\n"
        "optional transition error). ``opp_uniform`` is a uniform [0,1) draw\n"
        "used to pick the opponent move; must be drawn with np.random.random()\n"
        "for byte-identical reproducibility against the original Python path.\n"
        "Returns a (5,) float64 ndarray.");

    m.def(
        "sample_observation_step", &lasertag_sample_observation_step,
        py::arg("next_state"), py::arg("noise"),
        py::arg("rows"), py::arg("cols"), py::arg("walls_flat"),
        "Single-step observation for the discrete LaserTagPOMDP.\n\n"
        "``noise`` is a length-8 float64 array of pre-drawn N(0, sigma) samples\n"
        "(typically np.random.normal(0, measurement_noise, size=8)).\n"
        "Returns the noisy 8-direction laser observation as a (8,) float64\n"
        "ndarray. Caller must check next_state[4] (terminal) before calling.");

    m.def(
        "observation_log_probability_step", &lasertag_observation_log_probability_step,
        py::arg("next_state"), py::arg("observations"), py::arg("measurement_noise"),
        py::arg("rows"), py::arg("cols"), py::arg("walls_flat"),
        "Per-observation log-probability for the discrete LaserTagPOMDP.\n\n"
        "``observations`` is a (N, 8) float64 array. Returns a (N,) float64\n"
        "array of log-probabilities, mirroring\n"
        "LaserTagPOMDP.observation_log_probability semantics (terminal next\n"
        "state requires the (-1, ..., -1) sentinel observation).");
}
