// SPDX-License-Identifier: MIT

// RockSample POMDP native sampling hot path.
//
// The RockSample transition is deterministic (no RNG) and the observation is
// a 3-way categorical over {none=0, good=1, bad=2} with a Bernoulli flip
// whose probability depends on the Euclidean distance between the robot and
// the queried rock via ``exp(-distance / sensor_efficiency)``. The state is
// ``[robot_row, robot_col, rock_0, ..., rock_{R-1}]`` with terminal sentinel
// ``[-1, -1, ...]``.
//
// This extension does not inherit from ``TransitionModelCpp<Dim>`` /
// ``ObservationModelCpp<Dim>`` because the state dimension is runtime-variable
// (depends on ``num_rocks``) and neither model is a Gaussian mean-shift. It
// preserves the auto-dispatch protocol used by
// ``WeightedParticleBelief._update_weights``: both classes expose the same
// ``sample`` / ``probability`` / ``batch_sample`` / ``batch_log_likelihood``
// signatures as the LaserTag port, plus a scalar integer observation code
// on ``batch_log_likelihood``.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "pomdp_native/marshalling.hpp"
#include "pomdp_native/rng.hpp"

namespace py = pybind11;

namespace {

// Observation codes must match the Python OBS_NONE / OBS_GOOD / OBS_BAD
// constants in ``rocksample_vectorized_updater.py``.
constexpr int kObsNone = 0;
constexpr int kObsGood = 1;
constexpr int kObsBad = 2;

// Defensive flooring constants applied symmetrically by ``probability`` /
// ``batch_log_likelihood`` so the env-API scalar (np.log(prob)) and batch
// (log-pdf) paths agree on the same floored value (~ -690.776) for
// impossible events. ``std::log`` is not constexpr in C++17 so the log
// constant is a hard-coded ``static const double`` matching
// ``std::log(kProbFloor)``.
constexpr double kProbFloor = 1e-300;
static const double kLogProbFloor = -690.7755278982137;  // == std::log(kProbFloor)

// Discrete action ids mirror rock_sample_pomdp.py:
//   0 = sample (pick up rock at robot position if any)
//   1 = move north (row -= 1, clamped to >= 0)
//   2 = move east  (col += 1; if col >= map_cols the episode terminates)
//   3 = move south (row += 1, clamped to map_rows - 1)
//   4 = move west  (col -= 1, clamped to >= 0)
//   5..4+num_rocks = check rock k (no movement, noisy Bernoulli sensor)
//   Anything else is silently treated as a "check on nonexistent rock":
//   no state change, observation is deterministic "none". This mirrors the
//   Python per-particle fallback branch.

struct EnvParams {
    int map_rows;
    int map_cols;
    int num_rocks;
    double sensor_efficiency;
    std::vector<std::int32_t> rock_rows;  // length num_rocks
    std::vector<std::int32_t> rock_cols;  // length num_rocks
};

EnvParams make_env_params(int map_rows, int map_cols, int num_rocks,
                          const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>
                              &rock_positions,
                          double sensor_efficiency) {
    if (map_rows <= 0 || map_cols <= 0) {
        throw std::invalid_argument("map_rows and map_cols must be positive");
    }
    if (num_rocks < 0) {
        throw std::invalid_argument("num_rocks must be non-negative");
    }
    if (sensor_efficiency <= 0.0) {
        throw std::invalid_argument("sensor_efficiency must be positive");
    }
    if (rock_positions.ndim() != 2 || rock_positions.shape(0) != num_rocks ||
        rock_positions.shape(1) != 2) {
        throw std::invalid_argument("rock_positions must have shape (num_rocks, 2)");
    }
    EnvParams env;
    env.map_rows = map_rows;
    env.map_cols = map_cols;
    env.num_rocks = num_rocks;
    env.sensor_efficiency = sensor_efficiency;
    env.rock_rows.reserve(static_cast<std::size_t>(num_rocks));
    env.rock_cols.reserve(static_cast<std::size_t>(num_rocks));
    auto u = rock_positions.unchecked<2>();
    for (py::ssize_t i = 0; i < num_rocks; ++i) {
        env.rock_rows.push_back(u(i, 0));
        env.rock_cols.push_back(u(i, 1));
    }
    return env;
}

// Parse a state row from a Python object without enforcing a fixed length.
// The per-particle constructor is lenient: some callers pass a state that
// does not match the configured ``num_rocks`` (e.g. observation tests that
// only exercise movement / sample actions, where rock slots are not read).
// Accepts ndarray / tuple / list forms; requires 1-D and length >= 2 so
// ``state[0]`` / ``state[1]`` (robot row / col) are always addressable.
std::vector<double> parse_state_flexible(const py::object &state_obj, const char *label) {
    std::vector<double> out;
    if (py::isinstance<py::array>(state_obj)) {
        auto arr = state_obj.cast<
            py::array_t<double, py::array::c_style | py::array::forcecast>>();
        if (arr.ndim() != 1 || arr.shape(0) < 2) {
            throw std::invalid_argument(std::string(label) +
                                        " must be a 1-D array with length >= 2");
        }
        const auto n = static_cast<std::size_t>(arr.shape(0));
        out.reserve(n);
        auto u = arr.unchecked<1>();
        for (std::size_t i = 0; i < n; ++i) {
            out.push_back(u(static_cast<py::ssize_t>(i)));
        }
        return out;
    }
    auto seq = state_obj.cast<py::sequence>();
    const std::size_t n = static_cast<std::size_t>(py::len(seq));
    if (n < 2) {
        throw std::invalid_argument(std::string(label) + " must have length >= 2");
    }
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(seq[static_cast<py::ssize_t>(i)].cast<double>());
    }
    return out;
}

// Terminal sentinel: robot_row == -1 && robot_col == -1. Matches
// ``RockSamplePOMDP.is_terminal``.
inline bool is_terminal_row(const double *row) {
    return static_cast<int>(row[0]) == -1 && static_cast<int>(row[1]) == -1;
}

// Deterministic transition: writes next-state into ``out`` (length state_dim).
// ``src`` is the current state row, also length state_dim. ``state_dim`` is
// the length of the state vector used by this call (per-particle callers
// may pass a state whose length does not match ``2 + env.num_rocks``; the
// batch path always passes exactly ``2 + env.num_rocks``). ``env`` supplies
// grid and rock geometry. ``action`` is the discrete action id.
void transition_into(const double *src, double *out, int action, const EnvParams &env,
                     std::size_t state_dim) {
    // Rocks default to passed-through (unchanged).
    for (std::size_t d = 0; d < state_dim; ++d) {
        out[d] = src[d];
    }

    // Terminal state is absorbing.
    if (is_terminal_row(src)) {
        return;
    }

    const int robot_row = static_cast<int>(src[0]);
    const int robot_col = static_cast<int>(src[1]);

    // Defensive parity with Python: if current col already >= map_cols we
    // are already "exiting"; transition to terminal with rocks preserved.
    if (robot_col >= env.map_cols) {
        out[0] = -1.0;
        out[1] = -1.0;
        return;
    }

    int new_row = robot_row;
    int new_col = robot_col;
    switch (action) {
        case 1:  // North
            new_row = robot_row - 1;
            if (new_row < 0) new_row = 0;
            break;
        case 2:  // East (unclamped to allow exit)
            new_col = robot_col + 1;
            break;
        case 3:  // South
            new_row = robot_row + 1;
            if (new_row > env.map_rows - 1) new_row = env.map_rows - 1;
            break;
        case 4:  // West
            new_col = robot_col - 1;
            if (new_col < 0) new_col = 0;
            break;
        case 0:  // Sample: flip the rock at the robot's position to bad (0.0)
            for (int i = 0; i < env.num_rocks; ++i) {
                if (env.rock_rows[static_cast<std::size_t>(i)] == robot_row &&
                    env.rock_cols[static_cast<std::size_t>(i)] == robot_col) {
                    // Only write if the rock slot actually exists in this
                    // state row; match the lenient Python per-particle
                    // behavior for mismatched state / num_rocks.
                    const std::size_t slot = static_cast<std::size_t>(2 + i);
                    if (slot < state_dim) {
                        out[slot] = 0.0;
                    }
                    break;
                }
            }
            break;
        default:  // Check actions and unknown ids: no position change.
            break;
    }

    if (new_col >= env.map_cols) {
        out[0] = -1.0;
        out[1] = -1.0;
        return;
    }
    out[0] = static_cast<double>(new_row);
    out[1] = static_cast<double>(new_col);
}

class RockSampleTransitionCpp {
  public:
    RockSampleTransitionCpp(
        const py::object &state_obj, int action, int map_rows, int map_cols, int num_rocks,
        const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> &rock_positions,
        double sensor_efficiency)
        : env_(make_env_params(map_rows, map_cols, num_rocks, rock_positions, sensor_efficiency)),
          state_(parse_state_flexible(state_obj, "state")),
          action_(action) {}

    py::list sample(int n_samples) const {
        if (n_samples < 0) {
            throw std::invalid_argument("n_samples must be non-negative");
        }
        const std::size_t state_dim = state_.size();
        std::vector<double> next(state_dim);
        transition_into(state_.data(), next.data(), action_, env_, state_dim);

        py::list out;
        for (int i = 0; i < n_samples; ++i) {
            out.append(pomdp_native::array_from_vector(next.data(), state_dim));
        }
        return out;
    }

    // Indicator density: 1.0 for rows equal to the deterministic next state,
    // ``kProbFloor`` otherwise. Symmetric defensive flooring keeps the
    // env-API scalar ``np.log(probs)`` from emitting ``-inf`` for
    // impossible candidates (matches the log-space floor applied in
    // ``batch_log_likelihood``).
    py::array_t<double> probability(const py::object &values) const {
        const std::size_t state_dim = state_.size();
        auto batch = pomdp_native::extract_rows_nd(values, state_dim);
        std::vector<double> next(state_dim);
        transition_into(state_.data(), next.data(), action_, env_, state_dim);

        auto out = py::array_t<double>(static_cast<py::ssize_t>(batch.n));
        auto buf = out.mutable_unchecked<1>();
        for (std::size_t i = 0; i < batch.n; ++i) {
            const double *row = batch.flat.data() + i * state_dim;
            bool equal = true;
            for (std::size_t d = 0; d < state_dim; ++d) {
                if (row[d] != next[d]) {
                    equal = false;
                    break;
                }
            }
            buf(static_cast<py::ssize_t>(i)) = equal ? 1.0 : kProbFloor;
        }
        return out;
    }

    py::array_t<double> batch_sample(
        const py::array_t<double, py::array::c_style | py::array::forcecast> &particles) const {
        // Batch path requires the canonical shape (N, 2 + num_rocks) so the
        // caller's particle array layout is unambiguous.
        const std::size_t batch_state_dim = static_cast<std::size_t>(2 + env_.num_rocks);
        if (particles.ndim() != 2 ||
            static_cast<std::size_t>(particles.shape(1)) != batch_state_dim) {
            throw std::invalid_argument("particles must have shape (N, 2 + num_rocks)");
        }
        const auto n_rows = static_cast<std::size_t>(particles.shape(0));
        auto out = py::array_t<double>(
            {static_cast<py::ssize_t>(n_rows), static_cast<py::ssize_t>(batch_state_dim)});

        const double *in_data = particles.data();
        double *out_data = out.mutable_data();

        // Step 1: bulk memcpy the whole particle block; most action paths
        // only mutate a few slots per row, so starting from the input is
        // cheaper than clearing and rewriting column-by-column.
        std::memcpy(out_data, in_data, n_rows * batch_state_dim * sizeof(double));

        // Step 2: action-specific vectorizable mutations. These mirror the
        // pre-port NumPy ``_apply_movement`` / ``_apply_sample`` /
        // ``_apply_exit`` kernels.
        batch_apply_action_(out_data, n_rows, batch_state_dim);
        return out;
    }

    py::tuple state_property() const {
        py::tuple t(state_.size());
        for (std::size_t d = 0; d < state_.size(); ++d) {
            t[d] = state_[d];
        }
        return t;
    }
    int action_property() const { return action_; }

    // Rewrite only the state field; env geometry, action, and rock positions
    // stay frozen. Lets Python keep one kernel per (env, action) instead of
    // rebuilding for every call. Mirrors ContinuousLaserTagTransitionCpp.
    void set_state(const py::object &state_obj) {
        state_ = parse_state_flexible(state_obj, "state");
    }

  private:
    // Action-specific batch mutator. Applies the same transition semantics
    // as the per-particle ``transition_into`` but in a cache-friendly
    // column-style loop that matches the pre-port NumPy vectorization.
    // ``data`` has already been memcpy'd from input particles so terminal
    // rows are automatically preserved and the read/write alias is safe.
    void batch_apply_action_(double *data, std::size_t n_rows,
                             std::size_t state_dim) const {
        const int map_rows = env_.map_rows;
        const int map_cols = env_.map_cols;
        switch (action_) {
            case 0: {  // Sample: flip rock at robot position
                for (int k = 0; k < env_.num_rocks; ++k) {
                    const double rr =
                        static_cast<double>(env_.rock_rows[static_cast<std::size_t>(k)]);
                    const double rc =
                        static_cast<double>(env_.rock_cols[static_cast<std::size_t>(k)]);
                    const std::size_t slot = static_cast<std::size_t>(2 + k);
                    for (std::size_t i = 0; i < n_rows; ++i) {
                        double *row = data + i * state_dim;
                        // Terminal is row[0] == -1 && row[1] == -1; skip.
                        if (row[0] < 0.0 && row[1] < 0.0) {
                            continue;
                        }
                        if (row[0] == rr && row[1] == rc) {
                            row[slot] = 0.0;
                        }
                    }
                }
                return;
            }
            case 1: {  // North: row -= 1, clamped >= 0
                for (std::size_t i = 0; i < n_rows; ++i) {
                    double *row = data + i * state_dim;
                    if (row[0] < 0.0 && row[1] < 0.0) {
                        continue;
                    }
                    double new_r = row[0] - 1.0;
                    if (new_r < 0.0) {
                        new_r = 0.0;
                    }
                    row[0] = new_r;
                }
                return;
            }
            case 2: {  // East: col += 1, exit if new_col >= map_cols
                const double exit_threshold = static_cast<double>(map_cols);
                for (std::size_t i = 0; i < n_rows; ++i) {
                    double *row = data + i * state_dim;
                    if (row[0] < 0.0 && row[1] < 0.0) {
                        continue;
                    }
                    const double new_c = row[1] + 1.0;
                    if (new_c >= exit_threshold) {
                        row[0] = -1.0;
                        row[1] = -1.0;
                    } else {
                        row[1] = new_c;
                    }
                }
                return;
            }
            case 3: {  // South: row += 1, clamped <= map_rows - 1
                const double max_r = static_cast<double>(map_rows - 1);
                for (std::size_t i = 0; i < n_rows; ++i) {
                    double *row = data + i * state_dim;
                    if (row[0] < 0.0 && row[1] < 0.0) {
                        continue;
                    }
                    double new_r = row[0] + 1.0;
                    if (new_r > max_r) {
                        new_r = max_r;
                    }
                    row[0] = new_r;
                }
                return;
            }
            case 4: {  // West: col -= 1, clamped >= 0
                for (std::size_t i = 0; i < n_rows; ++i) {
                    double *row = data + i * state_dim;
                    if (row[0] < 0.0 && row[1] < 0.0) {
                        continue;
                    }
                    double new_c = row[1] - 1.0;
                    if (new_c < 0.0) {
                        new_c = 0.0;
                    }
                    row[1] = new_c;
                }
                return;
            }
            default:  // Check (>= 5) or unknown: no state change.
                return;
        }
    }

    EnvParams env_;
    std::vector<double> state_;
    int action_;
};

class RockSampleObservationCpp {
  public:
    RockSampleObservationCpp(
        const py::object &next_state_obj, int action, int map_rows, int map_cols, int num_rocks,
        const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> &rock_positions,
        double sensor_efficiency)
        : env_(make_env_params(map_rows, map_cols, num_rocks, rock_positions, sensor_efficiency)),
          next_state_(parse_state_flexible(next_state_obj, "next_state")),
          action_(action) {}

    // Returns a Python list of int observation codes (0/1/2). The Python
    // shim translates these back to the public string API.
    py::list sample(int n_samples) const {
        if (n_samples < 0) {
            throw std::invalid_argument("n_samples must be non-negative");
        }
        py::list out;
        const int rock_idx = action_ - 5;
        const bool is_check = (action_ >= 5 && rock_idx < env_.num_rocks &&
                               static_cast<std::size_t>(2 + rock_idx) < next_state_.size());
        if (!is_check) {
            for (int i = 0; i < n_samples; ++i) {
                out.append(py::int_(kObsNone));
            }
            return out;
        }

        const double efficiency = check_efficiency(next_state_.data(), rock_idx);
        const bool rock_good =
            next_state_[static_cast<std::size_t>(2 + rock_idx)] > 0.5;
        const int correct = rock_good ? kObsGood : kObsBad;
        const int flipped = rock_good ? kObsBad : kObsGood;

        pomdp_native::RNGState &rng = pomdp_native::default_rng();
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        for (int i = 0; i < n_samples; ++i) {
            const double u = uniform(rng.engine());
            out.append(py::int_(u < efficiency ? correct : flipped));
        }
        return out;
    }

    // ``values`` is an integer array (or sequence) of observation codes.
    // Returns the per-element probability in linear (not log) space.
    py::array_t<double> probability(const py::object &values) const {
        auto codes_arr = values.cast<
            py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>>();
        if (codes_arr.ndim() != 1) {
            throw std::invalid_argument("values must be a 1-D array of int codes");
        }
        const auto n = static_cast<std::size_t>(codes_arr.shape(0));
        auto code_view = codes_arr.unchecked<1>();

        const int rock_idx = action_ - 5;
        const bool is_check = (action_ >= 5 && rock_idx < env_.num_rocks &&
                               static_cast<std::size_t>(2 + rock_idx) < next_state_.size());

        auto out = py::array_t<double>(static_cast<py::ssize_t>(n));
        auto buf = out.mutable_unchecked<1>();

        if (!is_check) {
            for (std::size_t i = 0; i < n; ++i) {
                buf(static_cast<py::ssize_t>(i)) =
                    code_view(static_cast<py::ssize_t>(i)) == kObsNone ? 1.0 : kProbFloor;
            }
            return out;
        }

        // Terminal-sentinel short-circuit: mirrors ``batch_log_likelihood``,
        // which treats ``[-1, -1, ...]`` rows as having log-floor likelihood
        // for any observation under a check action. Without this, the scalar
        // path would compute Bernoulli(efficiency) from the sentinel coords
        // and disagree with the batch path on terminal particles.
        if (next_state_.size() >= 2 && next_state_[0] < 0.0 && next_state_[1] < 0.0) {
            for (std::size_t i = 0; i < n; ++i) {
                buf(static_cast<py::ssize_t>(i)) = kProbFloor;
            }
            return out;
        }

        const double efficiency = check_efficiency(next_state_.data(), rock_idx);
        const bool rock_good =
            next_state_[static_cast<std::size_t>(2 + rock_idx)] > 0.5;
        const double p_good = rock_good ? efficiency : (1.0 - efficiency);
        const double p_bad = rock_good ? (1.0 - efficiency) : efficiency;
        for (std::size_t i = 0; i < n; ++i) {
            const int code = code_view(static_cast<py::ssize_t>(i));
            double p = 0.0;
            if (code == kObsNone) {
                p = 0.0;
            } else if (code == kObsGood) {
                p = p_good;
            } else if (code == kObsBad) {
                p = p_bad;
            }
            // Symmetric defensive flooring: keep impossible events from
            // emitting np.log(0) = -inf through the env API (mirrors the
            // log-floor in batch_log_likelihood).
            if (p < kProbFloor) {
                p = kProbFloor;
            }
            buf(static_cast<py::ssize_t>(i)) = p;
        }
        return out;
    }

    // Batch log-likelihood: one scalar observation code, N particles.
    // Semantics match the pre-port ``_log_ll_movement`` / ``_log_ll_check``:
    //   movement or invalid-check action: 0 if obs == NONE else -inf
    //     (terminal particles do NOT get special treatment for movement)
    //   valid check action:
    //     - live particle + OBS_GOOD/BAD: log Bernoulli
    //     - live particle + OBS_NONE:     -inf
    //     - terminal particle:            -inf (regardless of obs)
    py::array_t<double> batch_log_likelihood(
        const py::array_t<double, py::array::c_style | py::array::forcecast> &next_particles,
        int observation) const {
        const std::size_t state_dim = static_cast<std::size_t>(2 + env_.num_rocks);
        if (next_particles.ndim() != 2 ||
            static_cast<std::size_t>(next_particles.shape(1)) != state_dim) {
            throw std::invalid_argument("next_particles must have shape (N, 2 + num_rocks)");
        }
        const auto n_rows = static_cast<std::size_t>(next_particles.shape(0));
        auto out = py::array_t<double>(static_cast<py::ssize_t>(n_rows));
        double *buf = out.mutable_data();

        // Symmetric flooring: replace what used to be -inf for impossible
        // events with kLogProbFloor so the env-API batch path matches the
        // scalar (np.log(kernel.probability(...))) path which gets the same
        // floor through ``probability``'s kProbFloor clamp.
        const int rock_idx = action_ - 5;
        const bool is_check = (action_ >= 5 && rock_idx < env_.num_rocks);

        if (!is_check) {
            const double val = (observation == kObsNone) ? 0.0 : kLogProbFloor;
            for (std::size_t i = 0; i < n_rows; ++i) {
                buf[i] = val;
            }
            return out;
        }

        if (observation == kObsNone) {
            for (std::size_t i = 0; i < n_rows; ++i) {
                buf[i] = kLogProbFloor;
            }
            return out;
        }

        const std::int32_t rock_r = env_.rock_rows[static_cast<std::size_t>(rock_idx)];
        const std::int32_t rock_c = env_.rock_cols[static_cast<std::size_t>(rock_idx)];
        const double inv_sigma = 1.0 / env_.sensor_efficiency;
        const std::size_t rock_offset = static_cast<std::size_t>(2 + rock_idx);

        const double *data = next_particles.data();
        for (std::size_t i = 0; i < n_rows; ++i) {
            const double *row = data + i * state_dim;
            const double robot_row = row[0];
            const double robot_col = row[1];
            if (robot_row < 0.0 && robot_col < 0.0) {
                buf[i] = kLogProbFloor;
                continue;
            }
            const double dr = robot_row - static_cast<double>(rock_r);
            const double dc = robot_col - static_cast<double>(rock_c);
            const double distance = std::sqrt(dr * dr + dc * dc);
            const double efficiency = std::exp(-distance * inv_sigma);
            const bool rock_good = row[rock_offset] > 0.5;
            double prob;
            if (observation == kObsGood) {
                prob = rock_good ? efficiency : (1.0 - efficiency);
            } else {  // OBS_BAD
                prob = rock_good ? (1.0 - efficiency) : efficiency;
            }
            if (prob < kProbFloor) prob = kProbFloor;
            double log_prob = std::log(prob);
            if (log_prob < kLogProbFloor) {
                log_prob = kLogProbFloor;
            }
            buf[i] = log_prob;
        }
        return out;
    }

    py::tuple next_state_property() const {
        py::tuple t(next_state_.size());
        for (std::size_t d = 0; d < next_state_.size(); ++d) {
            t[d] = next_state_[d];
        }
        return t;
    }
    int action_property() const { return action_; }

    // Rewrite only the next_state field; env geometry, action, and rock
    // positions stay frozen. Mirrors ContinuousLaserTagObservationCpp.
    void set_next_state(const py::object &next_state_obj) {
        next_state_ = parse_state_flexible(next_state_obj, "next_state");
    }

  private:
    double check_efficiency(const double *state, int rock_idx) const {
        const std::int32_t rock_r = env_.rock_rows[static_cast<std::size_t>(rock_idx)];
        const std::int32_t rock_c = env_.rock_cols[static_cast<std::size_t>(rock_idx)];
        const double dr = state[0] - static_cast<double>(rock_r);
        const double dc = state[1] - static_cast<double>(rock_c);
        const double distance = std::sqrt(dr * dr + dc * dc);
        return std::exp(-distance / env_.sensor_efficiency);
    }

    EnvParams env_;
    std::vector<double> next_state_;
    int action_;
};

// ---------------------------------------------------------------------------
// Reward-variant codes mirror the Python ``RewardModelType`` enum used by
// :class:`RockSamplePOMDP`:
//
//   0 -> CONSTANT_HAZARD_PENALTY             (constant-probability dangerous-area penalty)
//   1 -> ZERO_MEAN_HAZARD_SHOCK (+/- 50/50 dangerous-area perturbation)
//   2 -> DISTANCE_DECAYED_HAZARD_PENALTY (exp(-min_dist/decay) hit probability)
// ---------------------------------------------------------------------------
constexpr int kRewardVariantConstantHazardPenalty = 0;
constexpr int kRewardVariantZeroMeanHazardShock = 1;
constexpr int kRewardVariantDistanceDecayedHazardPenalty = 2;

// Base reward shared across all variants: step / exit / sample / sense.
// ``state_row`` is the current state row (length state_dim) so this
// matches the Python ``compute_reward`` which uses the CURRENT robot
// position for the exit / sample / sense terms.
inline double base_reward_term(const double *state_row, int action, int map_cols,
                               int num_rocks, const std::int32_t *rock_rows,
                               const std::int32_t *rock_cols, std::size_t state_dim,
                               double step_penalty, double exit_reward,
                               double good_rock_reward, double bad_rock_penalty,
                               double sensor_use_penalty) {
    double r = step_penalty;
    const int robot_row = static_cast<int>(state_row[0]);
    const int robot_col = static_cast<int>(state_row[1]);
    if (action == 2 && robot_col == map_cols - 1) {
        r += exit_reward;
        return r;
    }
    if (action == 0) {
        for (int k = 0; k < num_rocks; ++k) {
            if (rock_rows[k] == robot_row && rock_cols[k] == robot_col) {
                const std::size_t slot = static_cast<std::size_t>(2 + k);
                if (slot < state_dim) {
                    r += (state_row[slot] > 0.5) ? good_rock_reward : bad_rock_penalty;
                }
                break;
            }
        }
    }
    if (action >= 5) {
        r += sensor_use_penalty;
    }
    return r;
}

// Minimum squared distance from ``(row, col)`` to any dangerous-area centre.
// ``dangerous_areas`` is a (K, 2) row-major float64 array.
inline double min_dist_to_danger_sq(double row, double col, const double *dangers,
                                    std::size_t k) {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < k; ++j) {
        const double dr = row - dangers[j * 2];
        const double dc = col - dangers[j * 2 + 1];
        const double d2 = dr * dr + dc * dc;
        if (d2 < best) best = d2;
    }
    return best;
}

// Compute the dangerous-area reward contribution for a single
// (next_row, next_col) under variant ``reward_variant_code``. ``k`` is the
// number of dangerous-area centres (zero means no contribution).
inline double dangerous_area_term(double next_row, double next_col,
                                  const double *dangers, std::size_t k,
                                  double dangerous_area_radius,
                                  double dangerous_area_penalty,
                                  double dangerous_area_hit_probability,
                                  int reward_variant_code, double penalty_decay) {
    if (k == 0) {
        return 0.0;
    }
    pomdp_native::RNGState &rng = pomdp_native::default_rng();
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    if (reward_variant_code == kRewardVariantDistanceDecayedHazardPenalty) {
        const double min_d = std::sqrt(min_dist_to_danger_sq(next_row, next_col, dangers, k));
        const double p = std::exp(-min_d / penalty_decay);
        return (uniform(rng.engine()) < p) ? dangerous_area_penalty : 0.0;
    }
    const double radius_sq = dangerous_area_radius * dangerous_area_radius;
    const double min_d2 = min_dist_to_danger_sq(next_row, next_col, dangers, k);
    if (min_d2 > radius_sq) {
        return 0.0;
    }
    if (reward_variant_code == kRewardVariantZeroMeanHazardShock) {
        return (uniform(rng.engine()) < 0.5) ? dangerous_area_penalty
                                             : -dangerous_area_penalty;
    }
    // CONSTANT_HAZARD_PENALTY: constant-probability hit
    if (dangerous_area_hit_probability >= 1.0 ||
        uniform(rng.engine()) < dangerous_area_hit_probability) {
        return dangerous_area_penalty;
    }
    return 0.0;
}

// Standalone reward batch kernel. Returns a (N,) float64 array where each
// entry is the per-row reward under the configured variant.
py::array_t<double> reward_batch(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &states,
    int action,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &next_states,
    int map_rows, int map_cols,
    const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>
        &rock_positions,
    double step_penalty, double bad_rock_penalty, double good_rock_reward,
    double sensor_use_penalty, double exit_reward,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dangerous_areas,
    double dangerous_area_radius, double dangerous_area_penalty,
    double dangerous_area_hit_probability, int reward_variant_code,
    double penalty_decay) {
    (void)map_rows;  // currently unused; kept for parity with the kernel API
    if (states.ndim() != 2 || states.shape(1) < 2) {
        throw std::invalid_argument("states must have shape (N, 2 + num_rocks)");
    }
    if (next_states.ndim() != 2 || next_states.shape(0) != states.shape(0) ||
        next_states.shape(1) < 2) {
        throw std::invalid_argument("next_states must have same shape as states");
    }
    if (rock_positions.ndim() != 2 || rock_positions.shape(1) != 2) {
        throw std::invalid_argument("rock_positions must have shape (num_rocks, 2)");
    }
    if (dangerous_areas.ndim() != 2 || dangerous_areas.shape(1) != 2) {
        throw std::invalid_argument("dangerous_areas must have shape (K, 2)");
    }

    const auto n_rows = static_cast<std::size_t>(states.shape(0));
    const auto state_dim = static_cast<std::size_t>(states.shape(1));
    const int num_rocks = static_cast<int>(rock_positions.shape(0));
    const std::size_t n_dangers = static_cast<std::size_t>(dangerous_areas.shape(0));

    std::vector<std::int32_t> rock_rows_local(static_cast<std::size_t>(num_rocks));
    std::vector<std::int32_t> rock_cols_local(static_cast<std::size_t>(num_rocks));
    auto rp_view = rock_positions.unchecked<2>();
    for (int k = 0; k < num_rocks; ++k) {
        rock_rows_local[static_cast<std::size_t>(k)] = rp_view(k, 0);
        rock_cols_local[static_cast<std::size_t>(k)] = rp_view(k, 1);
    }

    const double *states_data = states.data();
    const double *next_data = next_states.data();
    const std::size_t next_dim = static_cast<std::size_t>(next_states.shape(1));
    const double *dangers_data = (n_dangers > 0) ? dangerous_areas.data() : nullptr;

    auto out = py::array_t<double>(static_cast<py::ssize_t>(n_rows));
    double *out_data = out.mutable_data();

    for (std::size_t i = 0; i < n_rows; ++i) {
        const double *cur_row = states_data + i * state_dim;
        const double *nxt_row = next_data + i * next_dim;
        double r = base_reward_term(cur_row, action, map_cols, num_rocks,
                                    rock_rows_local.data(), rock_cols_local.data(),
                                    state_dim, step_penalty, exit_reward,
                                    good_rock_reward, bad_rock_penalty,
                                    sensor_use_penalty);
        // Skip dangerous-area term for the exit case (Python returns early).
        const int robot_col = static_cast<int>(cur_row[1]);
        const bool is_exit = (action == 2 && robot_col == map_cols - 1);
        if (!is_exit && n_dangers > 0) {
            r += dangerous_area_term(nxt_row[0], nxt_row[1], dangers_data, n_dangers,
                                     dangerous_area_radius, dangerous_area_penalty,
                                     dangerous_area_hit_probability,
                                     reward_variant_code, penalty_decay);
        }
        out_data[i] = r;
    }
    return out;
}

// ---------------------------------------------------------------------------
// simulate_rollout_discrete: run a full rollout from ``initial_state``.
//
// Each step:
//   1. check terminal (robot_row == -1, robot_col == -1) — break if true
//   2. pick action from action_indices
//   3. deterministic transition via transition_into
//   4. compute reward matching _reward_from_next_state (variant-aware
//      dangerous-area term)
//   5. accumulate gamma^depth * reward
//
// action_indices: pre-drawn int32 array of length (max_depth - start_depth).
// rock_positions_flat: interleaved [row0, col0, row1, col1, ...] (1-D int32).
// ---------------------------------------------------------------------------
double simulate_rollout_discrete(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &initial_state,
    const py::array_t<int, py::array::c_style | py::array::forcecast> &action_indices,
    const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>
        &rock_positions_flat,
    int max_depth,
    int start_depth,
    double discount_factor,
    int map_rows,
    int map_cols,
    int n_actions,
    double step_penalty,
    double exit_reward,
    double good_rock_reward,
    double bad_rock_penalty,
    double sensor_use_penalty,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dangerous_areas,
    double dangerous_area_radius,
    double dangerous_area_penalty,
    double dangerous_area_hit_probability,
    int reward_variant_code,
    double penalty_decay) {
    if (initial_state.ndim() != 1 || initial_state.shape(0) < 2) {
        throw std::invalid_argument("initial_state must be a 1-D array with length >= 2");
    }
    if (rock_positions_flat.ndim() != 1) {
        throw std::invalid_argument("rock_positions_flat must be 1-D");
    }
    if (action_indices.ndim() != 1) {
        throw std::invalid_argument("action_indices must be 1-D");
    }
    if (dangerous_areas.ndim() != 2 || dangerous_areas.shape(1) != 2) {
        throw std::invalid_argument("dangerous_areas must have shape (K, 2)");
    }
    const std::size_t n_dangers = static_cast<std::size_t>(dangerous_areas.shape(0));
    const double *dangers_data = (n_dangers > 0) ? dangerous_areas.data() : nullptr;
    const int n_indices = static_cast<int>(action_indices.shape(0));

    const std::size_t state_dim = static_cast<std::size_t>(initial_state.shape(0));
    const int num_rocks = static_cast<int>((state_dim >= 2) ? state_dim - 2 : 0);

    // Unpack rock positions from flat array
    const auto rp_size = static_cast<std::size_t>(rock_positions_flat.shape(0));
    auto rp_view = rock_positions_flat.unchecked<1>();
    std::vector<std::int32_t> rock_rows_local;
    std::vector<std::int32_t> rock_cols_local;
    rock_rows_local.reserve(static_cast<std::size_t>(num_rocks));
    rock_cols_local.reserve(static_cast<std::size_t>(num_rocks));
    for (std::size_t i = 0; i + 1 < rp_size; i += 2) {
        rock_rows_local.push_back(rp_view(static_cast<py::ssize_t>(i)));
        rock_cols_local.push_back(rp_view(static_cast<py::ssize_t>(i + 1)));
    }

    // Build a minimal EnvParams for transition_into
    EnvParams env_local;
    env_local.map_rows = map_rows;
    env_local.map_cols = map_cols;
    env_local.num_rocks = num_rocks;
    env_local.sensor_efficiency = 1.0;
    env_local.rock_rows = rock_rows_local;
    env_local.rock_cols = rock_cols_local;

    // Copy initial state into mutable buffer
    auto state_view = initial_state.unchecked<1>();
    std::vector<double> cur(state_dim), nxt(state_dim);
    for (std::size_t d = 0; d < state_dim; ++d) {
        cur[d] = state_view(static_cast<py::ssize_t>(d));
    }

    auto ai_view = action_indices.unchecked<1>();

    double total = 0.0;
    double gamma_power = 1.0;
    int depth = start_depth;

    while (depth < max_depth) {
        if (is_terminal_row(cur.data())) {
            break;
        }

        const int idx_slot = depth - start_depth;
        if (idx_slot >= n_indices) {
            break;
        }
        int ai = ai_view(static_cast<py::ssize_t>(idx_slot));
        if (n_actions > 0 && (ai < 0 || ai >= n_actions)) {
            ai = ((ai % n_actions) + n_actions) % n_actions;
        }

        // Deterministic transition
        transition_into(cur.data(), nxt.data(), ai, env_local, state_dim);

        // Reward: matches _reward_from_next_state with variant-aware
        // dangerous-area term.
        double r = base_reward_term(cur.data(), ai, map_cols, num_rocks,
                                    env_local.rock_rows.data(),
                                    env_local.rock_cols.data(), state_dim,
                                    step_penalty, exit_reward, good_rock_reward,
                                    bad_rock_penalty, sensor_use_penalty);
        const int robot_col_cur = static_cast<int>(cur[1]);
        const bool is_exit = (ai == 2 && robot_col_cur == map_cols - 1);
        if (!is_exit && n_dangers > 0) {
            r += dangerous_area_term(nxt[0], nxt[1], dangers_data, n_dangers,
                                     dangerous_area_radius, dangerous_area_penalty,
                                     dangerous_area_hit_probability,
                                     reward_variant_code, penalty_decay);
        }

        total += gamma_power * r;
        gamma_power *= discount_factor;
        std::swap(cur, nxt);
        ++depth;
    }
    return total;
}

}  // anonymous namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native (C++) sampling hot path for RockSample POMDP.";

    m.def("set_seed", &pomdp_native::set_default_seed, py::arg("seed"),
          "Seed the module-level RNG used by sample().");

    m.def("simulate_rollout_discrete", &simulate_rollout_discrete,
          py::arg("initial_state"), py::arg("action_indices"),
          py::arg("rock_positions_flat"), py::arg("max_depth"), py::arg("start_depth"),
          py::arg("discount_factor"), py::arg("map_rows"), py::arg("map_cols"),
          py::arg("n_actions"), py::arg("step_penalty"), py::arg("exit_reward"),
          py::arg("good_rock_reward"), py::arg("bad_rock_penalty"),
          py::arg("sensor_use_penalty"),
          py::arg("dangerous_areas"), py::arg("dangerous_area_radius"),
          py::arg("dangerous_area_penalty"),
          py::arg("dangerous_area_hit_probability"),
          py::arg("reward_variant_code"), py::arg("penalty_decay"),
          "Native random rollout for RockSamplePOMDP with variant-aware "
          "dangerous-area reward term. Returns discounted reward sum. "
          "action_indices must be a pre-drawn int32 array of length (max_depth-start_depth). "
          "rock_positions_flat is a 1-D int32 array [row0, col0, row1, col1, ...]. "
          "dangerous_areas is a (K, 2) float64 array (may be empty). "
          "reward_variant_code: 0=CONSTANT_HAZARD_PENALTY, 1=ZERO_MEAN_HAZARD_SHOCK, 2=DISTANCE_DECAYED_HAZARD_PENALTY.");

    m.def("reward_batch", &reward_batch,
          py::arg("states"), py::arg("action"), py::arg("next_states"),
          py::arg("map_rows"), py::arg("map_cols"), py::arg("rock_positions"),
          py::arg("step_penalty"), py::arg("bad_rock_penalty"),
          py::arg("good_rock_reward"), py::arg("sensor_use_penalty"),
          py::arg("exit_reward"), py::arg("dangerous_areas"),
          py::arg("dangerous_area_radius"), py::arg("dangerous_area_penalty"),
          py::arg("dangerous_area_hit_probability"),
          py::arg("reward_variant_code"), py::arg("penalty_decay"),
          "Variant-aware standalone batch reward kernel for RockSamplePOMDP. "
          "Returns a (N,) float64 array of rewards. "
          "reward_variant_code: 0=CONSTANT_HAZARD_PENALTY, 1=ZERO_MEAN_HAZARD_SHOCK, 2=DISTANCE_DECAYED_HAZARD_PENALTY.");

    py::class_<RockSampleTransitionCpp>(m, "RockSampleTransitionCpp")
        .def(py::init<const py::object &, int, int, int, int,
                      const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> &,
                      double>(),
             py::arg("state"), py::arg("action"), py::arg("map_rows"), py::arg("map_cols"),
             py::arg("num_rocks"), py::arg("rock_positions"), py::arg("sensor_efficiency"))
        .def("sample", &RockSampleTransitionCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &RockSampleTransitionCpp::probability, py::arg("values"))
        .def("batch_sample", &RockSampleTransitionCpp::batch_sample, py::arg("particles"))
        .def("set_state", &RockSampleTransitionCpp::set_state, py::arg("state"))
        .def_property_readonly("state", &RockSampleTransitionCpp::state_property)
        .def_property_readonly("action", &RockSampleTransitionCpp::action_property);

    py::class_<RockSampleObservationCpp>(m, "RockSampleObservationCpp")
        .def(py::init<const py::object &, int, int, int, int,
                      const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> &,
                      double>(),
             py::arg("next_state"), py::arg("action"), py::arg("map_rows"), py::arg("map_cols"),
             py::arg("num_rocks"), py::arg("rock_positions"), py::arg("sensor_efficiency"))
        .def("sample", &RockSampleObservationCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &RockSampleObservationCpp::probability, py::arg("values"))
        .def("batch_log_likelihood", &RockSampleObservationCpp::batch_log_likelihood,
             py::arg("next_particles"), py::arg("observation"))
        .def("set_next_state", &RockSampleObservationCpp::set_next_state,
             py::arg("next_state"))
        .def_property_readonly("next_state", &RockSampleObservationCpp::next_state_property)
        .def_property_readonly("action", &RockSampleObservationCpp::action_property);
}
