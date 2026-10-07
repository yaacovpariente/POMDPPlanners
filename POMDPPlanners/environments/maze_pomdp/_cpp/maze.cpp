// SPDX-License-Identifier: MIT

// Native hot path for the maze family: DiscreteMazePOMDP, ContinuousMazePOMDP
// and TMazePOMDP.
//
// The three environments share one task: a state ``[x, y, goal_side,
// cue_phase]``, a cue cell that arms a single noisy reading, two goal cells,
// and a reward that pays +goal_reward, -wrong_goal_penalty or -step_penalty.
// They differ only in how a move is resolved and how a position maps to a
// cell, so one model class serves all three, switched by ``mode``:
//
//   0  DiscreteMazePOMDP    cell = round-half-even(x, y); a one-cell move
//                           into a wall is refused; a non-walkable start cell
//                           raises KeyError, as the Python lookup table does.
//   1  TMazePOMDP           cell = truncate-toward-zero(x, y); a move into a
//                           wall leaves the position unchanged; the cue arms
//                           whenever the resulting cell is the cue cell.
//   2  ContinuousMazePOMDP  the displacement is clipped to max_step_size and
//                           the swept segment is walked cell by cell: the
//                           first wall refuses the move, the first goal stops
//                           it at its entry point, crossing the cue arms it.
//
// Every operation is a line-by-line port of the Python reference in
// ``maze_pomdp.py`` / ``t_maze_pomdp.py``, written so the floating-point
// results are bit-identical: the same expressions in the same order, and no
// fused multiply-add (see the pragma below). The equivalence tests compare
// the two with exact equality.
//
// Observation draws use this module's RNG (``set_seed``), not numpy's, as
// the other native ports do. The draw itself reproduces numpy's
// ``searchsorted(cumsum(probs), u)``, so a given uniform maps to the same
// label in both.
//
// The model is one object per environment, not one kernel per action as in
// the RockSample port: the continuous variant's actions are fresh float
// vectors on every call, and a per-action cache keyed on them is the
// ``id()``-recycling trap the env-implementation notes describe.

// Python evaluates ``a + b * c`` as two rounded operations. A compiler that
// contracts it into one fused multiply-add rounds once and can move a segment
// cut point by one ulp, which is enough to flip a cell-boundary test. Clang
// (Apple arm64) and GCC on targets with FMA (aarch64) both contract by
// default, so contraction is switched off for this file.
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pomdp_native/rng.hpp"

namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Int32Array = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using UInt8Array = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

// State layout and encodings, matching STATE_* / GOAL_* / CUE_* in Python.
constexpr std::size_t kStateWidth = 4;
constexpr std::size_t kStateX = 0;
constexpr std::size_t kStateY = 1;
constexpr std::size_t kStateGoal = 2;
constexpr std::size_t kStateCuePhase = 3;
constexpr double kGoalLeft = 0.0;
constexpr double kCueUnseen = 0.0;
constexpr double kCueEmitting = 1.0;
constexpr double kCueConsumed = 2.0;

// Observation codes: the index into OBSERVATIONS = (left_cue, right_cue, empty).
constexpr int kObsLeftCue = 0;
constexpr int kObsRightCue = 1;
constexpr int kObsEmpty = 2;

// Discrete action codes: the index into ACTIONS = (up, down, left, right).
constexpr int kActionOffsets[4][2] = {{0, 1}, {0, -1}, {-1, 0}, {1, 0}};

// Python's _CELL_TOLERANCE.
constexpr double kCellTolerance = 1e-9;

constexpr int kModeDiscreteMaze = 0;
constexpr int kModeTMaze = 1;
constexpr int kModeContinuousMaze = 2;

// Which goal a position is in.
constexpr int kNoGoal = 0;
constexpr int kLeftGoal = 1;
constexpr int kRightGoal = 2;

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

struct Cell {
    std::int64_t x;
    std::int64_t y;
    bool operator==(const Cell &other) const { return x == other.x && y == other.y; }
    bool operator!=(const Cell &other) const { return !(*this == other); }
};

// What one step's path did, before any of it is written into a state.
// Mirrors Python's StepOutcome.
struct Outcome {
    double x;
    double y;
    bool blocked;
    bool entered_cue;
    int goal;  // kNoGoal / kLeftGoal / kRightGoal
};

// The closed interval of cell indices along one axis whose closed unit
// interval contains ``value``. Mirrors Python's _boundary_span, including the
// left-to-right evaluation of ``value - 0.5 - tol``.
inline void boundary_span(double value, std::int64_t &lower, std::int64_t &upper) {
    lower = static_cast<std::int64_t>(std::ceil(value - 0.5 - kCellTolerance));
    upper = static_cast<std::int64_t>(std::floor(value + 0.5 + kCellTolerance));
}

// Parse one state row; the Python reference reads slots 0..3 of any 1-D input.
inline const double *state_ptr(const DoubleArray &state) {
    if (state.ndim() != 1 || state.shape(0) < static_cast<py::ssize_t>(kStateWidth)) {
        throw std::invalid_argument("state must be a 1-D array of length >= 4");
    }
    return state.data();
}

inline const double *batch_ptr(const DoubleArray &states, std::size_t &n_rows, const char *label) {
    if (states.ndim() == 1 && states.shape(0) == static_cast<py::ssize_t>(kStateWidth)) {
        n_rows = 1;
        return states.data();
    }
    if (states.ndim() != 2 || states.shape(1) != static_cast<py::ssize_t>(kStateWidth)) {
        throw std::invalid_argument(std::string(label) + " must have shape (N, 4)");
    }
    n_rows = static_cast<std::size_t>(states.shape(0));
    return states.data();
}

class MazeModelCpp {
  public:
    MazeModelCpp(int mode, const UInt8Array &walkable, std::int64_t origin_x,
                 std::int64_t origin_y, std::pair<std::int64_t, std::int64_t> cue_cell,
                 std::pair<std::int64_t, std::int64_t> left_goal_cell,
                 std::pair<std::int64_t, std::int64_t> right_goal_cell, double cue_accuracy,
                 double log_cue_accuracy, double log_cue_error, double goal_reward,
                 double wrong_goal_penalty, double step_penalty, double max_step_size,
                 const py::tuple &observation_labels)
        : mode_(mode),
          origin_x_(origin_x),
          origin_y_(origin_y),
          cue_{cue_cell.first, cue_cell.second},
          left_goal_{left_goal_cell.first, left_goal_cell.second},
          right_goal_{right_goal_cell.first, right_goal_cell.second},
          cue_accuracy_(cue_accuracy),
          log_cue_accuracy_(log_cue_accuracy),
          log_cue_error_(log_cue_error),
          goal_reward_(goal_reward),
          wrong_goal_penalty_(wrong_goal_penalty),
          step_penalty_(step_penalty),
          max_step_size_(max_step_size) {
        if (mode != kModeDiscreteMaze && mode != kModeTMaze && mode != kModeContinuousMaze) {
            throw std::invalid_argument("mode must be 0 (discrete), 1 (T-maze) or 2 (continuous)");
        }
        if (walkable.ndim() != 2) {
            throw std::invalid_argument("walkable must be a 2-D (nx, ny) array");
        }
        if (py::len(observation_labels) != 3) {
            throw std::invalid_argument("observation_labels must hold three labels");
        }
        nx_ = static_cast<std::int64_t>(walkable.shape(0));
        ny_ = static_cast<std::int64_t>(walkable.shape(1));
        walkable_.assign(walkable.data(), walkable.data() + walkable.size());
        for (std::size_t i = 0; i < 3; ++i) {
            labels_[i] = observation_labels[i];
        }
    }

    // ── Public API (bound to Python) ────────────────────────────────────

    bool is_terminal(const DoubleArray &state) const {
        const double *s = state_ptr(state);
        return terminal_row(s);
    }

    py::array_t<double> sample_next_state(const DoubleArray &state,
                                          const py::object &action) const {
        const double *s = state_ptr(state);
        const auto width = static_cast<std::size_t>(state.shape(0));
        const Action a = parse_action(action);
        if (terminal_row(s)) {
            // Absorbing: the Python reference returns a copy of the whole input.
            auto out = py::array_t<double>(static_cast<py::ssize_t>(width));
            std::copy(s, s + width, out.mutable_data());
            return out;
        }
        auto out = py::array_t<double>(static_cast<py::ssize_t>(kStateWidth));
        successor_live(s, a, out.mutable_data());
        return out;
    }

    py::array_t<double> batch_sample(const DoubleArray &states, const py::object &action) const {
        std::size_t n_rows = 0;
        const double *in = batch_ptr(states, n_rows, "states");
        const Action a = parse_action(action);
        auto out = py::array_t<double>(
            {static_cast<py::ssize_t>(n_rows), static_cast<py::ssize_t>(kStateWidth)});
        double *dst = out.mutable_data();
        for (std::size_t i = 0; i < n_rows; ++i) {
            successor_row(in + i * kStateWidth, a, dst + i * kStateWidth);
        }
        return out;
    }

    // One label when n_samples == 1, else a list, as the Python method returns.
    py::object sample_observation(const DoubleArray &next_state, int n_samples) const {
        if (n_samples < 0) {
            throw std::invalid_argument("n_samples must be non-negative");
        }
        const double *s = state_ptr(next_state);
        double cumulative[3];  // NOLINT(modernize-avoid-c-arrays)
        observation_cumulative(s, cumulative);
        pomdp_native::RNGState &rng = pomdp_native::default_rng();
        if (n_samples == 1) {
            return labels_[draw_code(cumulative, rng)];
        }
        py::list out;
        for (int i = 0; i < n_samples; ++i) {
            out.append(labels_[draw_code(cumulative, rng)]);
        }
        return out;
    }

    // Log Z(o | s') for a list of observation codes against one state. Codes
    // outside 0..2 stand for labels outside the alphabet and score -inf.
    py::array_t<double> observation_log_probability(const DoubleArray &next_state,
                                                    const Int32Array &codes) const {
        const double *s = state_ptr(next_state);
        if (codes.ndim() != 1) {
            throw std::invalid_argument("codes must be a 1-D int array");
        }
        const auto n = static_cast<std::size_t>(codes.shape(0));
        auto out = py::array_t<double>(static_cast<py::ssize_t>(n));
        double *dst = out.mutable_data();
        const std::int32_t *src = codes.data();
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = log_likelihood(s, src[i]);
        }
        return out;
    }

    // Log Z(o | s') of one observation code against many states.
    py::array_t<double> batch_log_likelihood(const DoubleArray &next_states,
                                             int observation) const {
        std::size_t n_rows = 0;
        const double *in = batch_ptr(next_states, n_rows, "next_states");
        auto out = py::array_t<double>(static_cast<py::ssize_t>(n_rows));
        double *dst = out.mutable_data();
        for (std::size_t i = 0; i < n_rows; ++i) {
            dst[i] = log_likelihood(in + i * kStateWidth, observation);
        }
        return out;
    }

    double reward(const DoubleArray &state, const py::object &action,
                  const py::object &next_state) const {
        const double *s = state_ptr(state);
        if (!next_state.is_none()) {
            const DoubleArray ns = next_state.cast<DoubleArray>();
            return reward_from_successor(s, state_ptr(ns));
        }
        double successor[kStateWidth];  // NOLINT(modernize-avoid-c-arrays)
        successor_row(s, parse_action(action), successor);
        return reward_from_successor(s, successor);
    }

    py::array_t<double> reward_batch(const DoubleArray &states, const py::object &action,
                                     const py::object &next_states) const {
        std::size_t n_rows = 0;
        const double *in = batch_ptr(states, n_rows, "states");
        auto out = py::array_t<double>(static_cast<py::ssize_t>(n_rows));
        double *dst = out.mutable_data();
        if (!next_states.is_none()) {
            const DoubleArray ns_arr = next_states.cast<DoubleArray>();
            std::size_t n_next = 0;
            const double *ns = batch_ptr(ns_arr, n_next, "next_states");
            if (n_next != n_rows) {
                throw std::invalid_argument("next_states must have as many rows as states");
            }
            for (std::size_t i = 0; i < n_rows; ++i) {
                dst[i] = reward_from_successor(in + i * kStateWidth, ns + i * kStateWidth);
            }
            return out;
        }
        const Action a = parse_action(action);
        double successor[kStateWidth];  // NOLINT(modernize-avoid-c-arrays)
        for (std::size_t i = 0; i < n_rows; ++i) {
            successor_row(in + i * kStateWidth, a, successor);
            dst[i] = reward_from_successor(in + i * kStateWidth, successor);
        }
        return out;
    }

    // Fused G(s, a): (next_state, observation label, reward). Same draws, in
    // the same order, as sample_next_state -> sample_observation -> reward.
    py::tuple sample_next_step(const DoubleArray &state, const py::object &action) const {
        const double *s = state_ptr(state);
        const auto width = static_cast<std::size_t>(state.shape(0));
        const Action a = parse_action(action);
        py::array_t<double> next;
        if (terminal_row(s)) {
            next = py::array_t<double>(static_cast<py::ssize_t>(width));
            std::copy(s, s + width, next.mutable_data());
        } else {
            next = py::array_t<double>(static_cast<py::ssize_t>(kStateWidth));
            successor_live(s, a, next.mutable_data());
        }
        const double *ns = next.data();
        double cumulative[3];  // NOLINT(modernize-avoid-c-arrays)
        observation_cumulative(ns, cumulative);
        const int code = draw_code(cumulative, pomdp_native::default_rng());
        const double r = reward_from_successor(s, ns);
        return py::make_tuple(next, labels_[code], r);
    }

    // Whether the move is refused by a wall (DiscreteMaze / ContinuousMaze
    // step_info). The caller has already excluded terminal states.
    bool blocked(const DoubleArray &state, const py::object &action) const {
        const double *s = state_ptr(state);
        return execute(s, parse_action(action)).blocked;
    }

    // Random rollout from ``initial_state`` over pre-drawn actions. Matches
    // ``python_random_rollout``: stop at max_depth or a terminal state, and
    // fold the rewards from the end as ``r + gamma * rest`` so the sum is
    // rounded exactly as the recursive Python version rounds it.
    //
    // ``actions`` is an int32 vector of action codes for the discrete modes and
    // a (K, 2) float64 array of displacements for the continuous mode.
    double simulate_rollout(const DoubleArray &initial_state, const py::object &actions,
                            int max_depth, int start_depth, double discount_factor) const {
        const double *s0 = state_ptr(initial_state);
        double cur[kStateWidth];   // NOLINT(modernize-avoid-c-arrays)
        double next[kStateWidth];  // NOLINT(modernize-avoid-c-arrays)
        std::copy(s0, s0 + kStateWidth, cur);

        std::vector<Action> plan;
        if (mode_ == kModeContinuousMaze) {
            const DoubleArray arr = actions.cast<DoubleArray>();
            if (arr.ndim() != 2 || arr.shape(1) != 2) {
                throw std::invalid_argument("continuous actions must have shape (K, 2)");
            }
            const auto k = static_cast<std::size_t>(arr.shape(0));
            plan.reserve(k);
            for (std::size_t i = 0; i < k; ++i) {
                plan.push_back(Action{kVectorAction, arr.data()[2 * i], arr.data()[2 * i + 1]});
            }
        } else {
            const Int32Array arr = actions.cast<Int32Array>();
            if (arr.ndim() != 1) {
                throw std::invalid_argument("discrete actions must be a 1-D int array");
            }
            const auto k = static_cast<std::size_t>(arr.shape(0));
            plan.reserve(k);
            for (std::size_t i = 0; i < k; ++i) {
                plan.push_back(discrete_action(arr.data()[i]));
            }
        }

        std::vector<double> rewards;
        int depth = start_depth;
        while (depth < max_depth && !terminal_row(cur)) {
            const auto slot = static_cast<std::size_t>(depth - start_depth);
            if (slot >= plan.size()) {
                throw std::invalid_argument("fewer pre-drawn actions than rollout steps");
            }
            successor_live(cur, plan[slot], next);
            rewards.push_back(reward_from_successor(cur, next));
            std::copy(next, next + kStateWidth, cur);
            ++depth;
        }
        double value = 0.0;
        for (auto it = rewards.rbegin(); it != rewards.rend(); ++it) {
            value = *it + discount_factor * value;
        }
        return value;
    }

    int mode() const { return mode_; }

  private:
    // A resolved action: a direction code for the discrete modes, a clipped
    // displacement for the continuous one.
    struct Action {
        int code;  // 0..3 for a direction, kVectorAction, or kBadVectorAction
        double dx;
        double dy;
    };
    static constexpr int kVectorAction = -1;
    static constexpr int kBadVectorAction = -2;

    // An out-of-range code stands for a label outside ACTIONS. It is accepted
    // here and rejected in ``execute``, because the Python reference only
    // looks the action up for a non-terminal state.
    static Action discrete_action(int code) { return Action{code, 0.0, 0.0}; }

    static void check_action_code(const Action &a) {
        if (a.code < 0 || a.code > 3) {
            throw py::key_error("action is not one of up, down, left, right");
        }
    }

    // Mirrors ContinuousMazePOMDP.clip_action.
    Action clip_action(double x0, double x1) const {
        if (!std::isfinite(x0) || !std::isfinite(x1)) {
            throw std::invalid_argument("action must contain only finite values");
        }
        const double magnitude = std::sqrt(x0 * x0 + x1 * x1);
        if (magnitude > max_step_size_) {
            const double scale = max_step_size_ / magnitude;
            return Action{kVectorAction, x0 * scale, x1 * scale};
        }
        return Action{kVectorAction, x0, x1};
    }

    Action parse_action(const py::object &action) const {
        if (mode_ != kModeContinuousMaze) {
            return discrete_action(action.cast<int>());
        }
        // Kept raw and validated in ``execute``: the Python reference never
        // inspects the action of a terminal state.
        const DoubleArray vec = action.cast<DoubleArray>();
        if (vec.size() != 2) {
            return Action{kBadVectorAction, 0.0, 0.0};
        }
        return Action{kVectorAction, vec.data()[0], vec.data()[1]};
    }

    bool walkable(const Cell &c) const {
        const std::int64_t ix = c.x - origin_x_;
        const std::int64_t iy = c.y - origin_y_;
        if (ix < 0 || iy < 0 || ix >= nx_ || iy >= ny_) {
            return false;
        }
        return walkable_[static_cast<std::size_t>(ix * ny_ + iy)] != 0;
    }

    int goal_of_cell(const Cell &c) const {
        if (c == left_goal_) return kLeftGoal;
        if (c == right_goal_) return kRightGoal;
        return kNoGoal;
    }

    // The goal holding (x, y): any cell of the closed-square cover for the
    // continuous geometry, the truncated cell for the T-maze.
    int goal_at(double x, double y) const {
        if (mode_ == kModeTMaze) {
            return goal_of_cell(Cell{static_cast<std::int64_t>(x), static_cast<std::int64_t>(y)});
        }
        std::int64_t c0 = 0, c1 = 0, r0 = 0, r1 = 0;
        boundary_span(x, c0, c1);
        boundary_span(y, r0, r1);
        for (std::int64_t cx = c0; cx <= c1; ++cx) {
            for (std::int64_t cy = r0; cy <= r1; ++cy) {
                const int g = goal_of_cell(Cell{cx, cy});
                if (g != kNoGoal) return g;
            }
        }
        return kNoGoal;
    }

    bool terminal_row(const double *s) const { return goal_at(s[kStateX], s[kStateY]) != kNoGoal; }

    // ── Movement ─────────────────────────────────────────────────────────

    Outcome execute(const double *s, const Action &a) const {
        if (mode_ == kModeContinuousMaze) {
            if (a.code == kBadVectorAction) {
                throw std::invalid_argument("action must be a 2-vector");
            }
            const Action clipped = clip_action(a.dx, a.dy);
            const double sx = s[kStateX];
            const double sy = s[kStateY];
            return walk_segment(sx, sy, sx + clipped.dx, sy + clipped.dy);
        }
        if (mode_ == kModeTMaze) {
            check_action_code(a);
            const int *offset = kActionOffsets[a.code];
            const Cell here{static_cast<std::int64_t>(s[kStateX]),
                            static_cast<std::int64_t>(s[kStateY])};
            const Cell candidate{here.x + offset[0], here.y + offset[1]};
            const Cell landed = walkable(candidate) ? candidate : here;
            // The T-maze arms the cue on landing in the cue cell, a refused
            // move included; ``blocked`` is not read for the T-maze.
            return Outcome{static_cast<double>(landed.x), static_cast<double>(landed.y),
                           landed == here, landed == cue_, goal_of_cell(landed)};
        }
        // Discrete maze: Python's ``self._transitions[cell][action]``.
        const Cell here{static_cast<std::int64_t>(std::nearbyint(s[kStateX])),
                        static_cast<std::int64_t>(std::nearbyint(s[kStateY]))};
        if (!walkable(here)) {
            throw py::key_error("(" + std::to_string(here.x) + ", " + std::to_string(here.y) +
                                ")");
        }
        check_action_code(a);
        const int *offset = kActionOffsets[a.code];
        const Cell candidate{here.x + offset[0], here.y + offset[1]};
        if (!walkable(candidate)) {
            return Outcome{static_cast<double>(here.x), static_cast<double>(here.y), true, false,
                           kNoGoal};
        }
        return Outcome{static_cast<double>(candidate.x), static_cast<double>(candidate.y), false,
                       candidate == cue_, goal_of_cell(candidate)};
    }

    // Append every cell of the closed-square cover of (x, y) not yet in
    // ``cells``, stamped with ``time``. Mirrors ``earliest.setdefault``.
    static void set_default_cells(double x, double y, double time,
                                  std::vector<std::pair<Cell, double>> &cells) {
        std::int64_t c0 = 0, c1 = 0, r0 = 0, r1 = 0;
        boundary_span(x, c0, c1);
        boundary_span(y, r0, r1);
        for (std::int64_t cx = c0; cx <= c1; ++cx) {
            for (std::int64_t cy = r0; cy <= r1; ++cy) {
                const Cell c{cx, cy};
                bool seen = false;
                for (const auto &entry : cells) {
                    if (entry.first == c) {
                        seen = true;
                        break;
                    }
                }
                if (!seen) cells.emplace_back(c, time);
            }
        }
    }

    // Mirrors BaseMazePOMDP._crossed_cells: (cell, entry time) pairs, sorted
    // by time with ties kept in insertion order (Python's sort is stable).
    static std::vector<std::pair<Cell, double>> crossed_cells(double sx, double sy, double ex,
                                                              double ey) {
        const double dx = ex - sx;
        const double dy = ey - sy;
        std::vector<double> cuts{0.0, 1.0};
        const double origins[2] = {sx, sy};  // NOLINT(modernize-avoid-c-arrays)
        const double deltas[2] = {dx, dy};   // NOLINT(modernize-avoid-c-arrays)
        for (int axis = 0; axis < 2; ++axis) {
            const double origin = origins[axis];
            const double delta = deltas[axis];
            if (delta == 0.0) continue;
            const double far = origin + delta;
            const double low = std::min(origin, far);
            const double high = std::max(origin, far);
            const auto first = static_cast<std::int64_t>(std::floor(low + 0.5));
            const auto last = static_cast<std::int64_t>(std::ceil(high + 0.5));
            for (std::int64_t index = first; index <= last; ++index) {
                const double time = (static_cast<double>(index) - 0.5 - origin) / delta;
                if (0.0 < time && time < 1.0) cuts.push_back(time);
            }
        }
        std::sort(cuts.begin(), cuts.end());
        cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());

        std::vector<std::pair<Cell, double>> earliest;
        for (std::size_t i = 0; i < cuts.size(); ++i) {
            const double time = cuts[i];
            set_default_cells(sx + dx * time, sy + dy * time, time, earliest);
            if (i + 1 < cuts.size()) {
                const double middle = 0.5 * (time + cuts[i + 1]);
                set_default_cells(sx + dx * middle, sy + dy * middle, time, earliest);
            }
        }
        std::stable_sort(earliest.begin(), earliest.end(),
                         [](const auto &a, const auto &b) { return a.second < b.second; });
        return earliest;
    }

    // Mirrors BaseMazePOMDP._walk_segment.
    Outcome walk_segment(double sx, double sy, double ex, double ey) const {
        std::vector<Cell> standing;
        {
            std::int64_t c0 = 0, c1 = 0, r0 = 0, r1 = 0;
            boundary_span(sx, c0, c1);
            boundary_span(sy, r0, r1);
            for (std::int64_t cx = c0; cx <= c1; ++cx) {
                for (std::int64_t cy = r0; cy <= r1; ++cy) standing.push_back(Cell{cx, cy});
            }
        }
        const auto crossed = crossed_cells(sx, sy, ex, ey);
        bool entered_cue = false;
        std::size_t i = 0;
        while (i < crossed.size()) {
            const double time = crossed[i].second;
            std::size_t j = i;
            bool any_wall = false;
            bool any_cue = false;
            int reached = kNoGoal;
            bool any_new = false;
            for (; j < crossed.size() && crossed[j].second == time; ++j) {
                const Cell &c = crossed[j].first;
                if (std::find(standing.begin(), standing.end(), c) != standing.end()) continue;
                any_new = true;
                if (!walkable(c)) any_wall = true;
                if (c == cue_) any_cue = true;
                if (reached == kNoGoal) reached = goal_of_cell(c);
            }
            i = j;
            if (!any_new) continue;
            if (any_wall) return Outcome{sx, sy, true, false, kNoGoal};
            if (any_cue) entered_cue = true;
            if (reached != kNoGoal) {
                return Outcome{sx + (ex - sx) * time, sy + (ey - sy) * time, false, entered_cue,
                               reached};
            }
        }
        return Outcome{ex, ey, false, entered_cue, kNoGoal};
    }

    static double next_cue_phase(double phase, bool entered_cue) {
        if (phase == kCueEmitting) return kCueConsumed;
        if (phase == kCueUnseen && entered_cue) return kCueEmitting;
        return phase;
    }

    // Successor of a state known to be non-terminal.
    void successor_live(const double *s, const Action &a, double *out) const {
        const Outcome o = execute(s, a);
        out[kStateX] = o.x;
        out[kStateY] = o.y;
        out[kStateGoal] = s[kStateGoal];
        out[kStateCuePhase] = next_cue_phase(s[kStateCuePhase], o.entered_cue);
    }

    void successor_row(const double *s, const Action &a, double *out) const {
        if (terminal_row(s)) {
            std::copy(s, s + kStateWidth, out);
            return;
        }
        successor_live(s, a, out);
    }

    // ── Reward and observation ───────────────────────────────────────────

    double reward_from_successor(const double *s, const double *successor) const {
        if (terminal_row(s)) return 0.0;
        const int reached = goal_at(successor[kStateX], successor[kStateY]);
        if (reached == kNoGoal) return -step_penalty_;
        const int paying = (s[kStateGoal] == kGoalLeft) ? kLeftGoal : kRightGoal;
        return reached == paying ? goal_reward_ : -wrong_goal_penalty_;
    }

    // np.cumsum of _observation_probs, summed in the same order.
    void observation_cumulative(const double *s, double *cumulative) const {
        double p[3] = {0.0, 0.0, 0.0};  // NOLINT(modernize-avoid-c-arrays)
        if (s[kStateCuePhase] != kCueEmitting) {
            p[kObsEmpty] = 1.0;
        } else if (s[kStateGoal] == kGoalLeft) {
            p[kObsLeftCue] = cue_accuracy_;
            p[kObsRightCue] = 1.0 - cue_accuracy_;
        } else {
            p[kObsRightCue] = cue_accuracy_;
            p[kObsLeftCue] = 1.0 - cue_accuracy_;
        }
        cumulative[0] = p[0];
        cumulative[1] = cumulative[0] + p[1];
        cumulative[2] = cumulative[1] + p[2];
    }

    // np.searchsorted(cumulative, u) (side="left"), clamped to the last label.
    static int draw_code(const double *cumulative, pomdp_native::RNGState &rng) {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const double u = uniform(rng.engine());
        int index = 0;
        for (int i = 0; i < 3; ++i) {
            if (cumulative[i] < u) ++index;
        }
        return std::min(index, 2);
    }

    // Mirrors observation_log_probability_per_state for one row.
    double log_likelihood(const double *s, int code) const {
        const bool emitting = s[kStateCuePhase] == kCueEmitting;
        if (code == kObsEmpty) return emitting ? kNegInf : 0.0;
        if (code != kObsLeftCue && code != kObsRightCue) return kNegInf;
        if (!emitting) return kNegInf;
        const bool goal_is_left = s[kStateGoal] == kGoalLeft;
        const bool matches = (code == kObsLeftCue) ? goal_is_left : !goal_is_left;
        return matches ? log_cue_accuracy_ : log_cue_error_;
    }

    int mode_;
    std::int64_t origin_x_;
    std::int64_t origin_y_;
    std::int64_t nx_ = 0;
    std::int64_t ny_ = 0;
    std::vector<std::uint8_t> walkable_;
    Cell cue_;
    Cell left_goal_;
    Cell right_goal_;
    double cue_accuracy_;
    double log_cue_accuracy_;
    double log_cue_error_;
    double goal_reward_;
    double wrong_goal_penalty_;
    double step_penalty_;
    double max_step_size_;
    py::object labels_[3];  // NOLINT(modernize-avoid-c-arrays)
};

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native (C++) hot path for the maze POMDPs (discrete, continuous, T-maze).";

    m.def("set_seed", &pomdp_native::set_default_seed, py::arg("seed"),
          "Seed the module-level RNG used by the observation sampler.");

    py::class_<MazeModelCpp>(m, "MazeModelCpp")
        .def(py::init<int, const UInt8Array &, std::int64_t, std::int64_t,
                      std::pair<std::int64_t, std::int64_t>, std::pair<std::int64_t, std::int64_t>,
                      std::pair<std::int64_t, std::int64_t>, double, double, double, double,
                      double, double, double, const py::tuple &>(),
             py::arg("mode"), py::arg("walkable"), py::arg("origin_x"), py::arg("origin_y"),
             py::arg("cue_cell"), py::arg("left_goal_cell"), py::arg("right_goal_cell"),
             py::arg("cue_accuracy"), py::arg("log_cue_accuracy"), py::arg("log_cue_error"),
             py::arg("goal_reward"), py::arg("wrong_goal_penalty"), py::arg("step_penalty"),
             py::arg("max_step_size"), py::arg("observation_labels"))
        .def("is_terminal", &MazeModelCpp::is_terminal, py::arg("state"))
        .def("sample_next_state", &MazeModelCpp::sample_next_state, py::arg("state"),
             py::arg("action"))
        .def("batch_sample", &MazeModelCpp::batch_sample, py::arg("states"), py::arg("action"))
        .def("sample_observation", &MazeModelCpp::sample_observation, py::arg("next_state"),
             py::arg("n_samples") = 1)
        .def("observation_log_probability", &MazeModelCpp::observation_log_probability,
             py::arg("next_state"), py::arg("codes"))
        .def("batch_log_likelihood", &MazeModelCpp::batch_log_likelihood,
             py::arg("next_states"), py::arg("observation"))
        .def("reward", &MazeModelCpp::reward, py::arg("state"), py::arg("action"),
             py::arg("next_state") = py::none())
        .def("reward_batch", &MazeModelCpp::reward_batch, py::arg("states"), py::arg("action"),
             py::arg("next_states") = py::none())
        .def("sample_next_step", &MazeModelCpp::sample_next_step, py::arg("state"),
             py::arg("action"))
        .def("blocked", &MazeModelCpp::blocked, py::arg("state"), py::arg("action"))
        .def("simulate_rollout", &MazeModelCpp::simulate_rollout, py::arg("initial_state"),
             py::arg("actions"), py::arg("max_depth"), py::arg("start_depth"),
             py::arg("discount_factor"))
        .def_property_readonly("mode", &MazeModelCpp::mode);
}
