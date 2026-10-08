// SPDX-License-Identifier: MIT

// Firefighting POMDP native extension.
//
// One model object per environment holds the world (grid, obstacles, depot)
// and every probability and cost. The action is a call argument rather than a
// constructor argument: a joint action is only a base-5 decode, and a kernel
// per action would mean a cache of 5 ** num_firefighters objects for nothing.
//
// Every method here mirrors one method of FirefightingPOMDP in
// firefighting_pomdp.py, which stays the reference:
//   sample_next_state           <- _transition
//   transition_log_probability  <- _successor_log_probability
//   sample_observation          <- _draw_observation
//   observation_log_probability <- observation_log_probability
//   reward / reward_batch       <- reward
//   is_terminal                 <- is_terminal
// The draws come from the module-local RNG (pomdp_native/rng.hpp), not from
// numpy, so a seeded run gives the same distribution as the Python code but
// not the same sequence.
//
// State layout (float64):
//   [step, (row, col, tank, health) x N, wind_direction, wind_strength, cells]
// Observation layout (float64):
//   [(row, col, tank, health) x N, cells], unseen cells reported as -1.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pomdp_native/rng.hpp"

namespace py = pybind11;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

constexpr int kUnburnt = 0;
constexpr int kSmoldering = 1;
constexpr int kBurning = 2;
constexpr int kBurnt = 3;
constexpr int kWet = 4;
constexpr int kNumCategories = 5;
constexpr int kNumFirefighterActions = 5;
constexpr int kSuppress = 4;
constexpr int kFieldWidth = 4;
constexpr int kFirefighterOffset = 1;
constexpr int kWindHigh = 1;
constexpr double kUnknown = -1.0;
constexpr int kOffsets[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
constexpr int kHeatDamage[kNumCategories] = {0, 1, 2, 0, 0};
const double kNegInf = -std::numeric_limits<double>::infinity();

// numpy's rint and Python's round both round half to even, which is what
// nearbyint does under the default rounding mode.
inline long to_int(double value) { return static_cast<long>(std::nearbyint(value)); }

inline bool is_alight(int category) {
    return category == kSmoldering || category == kBurning;
}

// Category lookups clamp out-of-range codes to "no effect" instead of reading
// past a table; a valid state never holds one.
inline int heat_damage(int category) {
    return (category >= 0 && category < kNumCategories) ? kHeatDamage[category] : 0;
}

struct Firefighter {
    long row;
    long col;
    long tank;
    long health;
};

class FirefightingModelCpp {
  public:
    FirefightingModelCpp(int num_rows, int num_cols, int num_firefighters,
                         py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>
                             obstacle_mask,
                         int depot_row, int depot_col, int max_tank,
                         int sensing_radius, double observation_error_probability,
                         double slip_probability, double spread_probability, double wind_gain_low,
                         double wind_gain_high, double crosswind_attenuation,
                         double growth_probability, double burnout_probability,
                         double suppression_probability_unburnt,
                         double suppression_probability_smoldering,
                         double suppression_probability_burning, int max_steps,
                         double success_reward, double step_cost, double smoldering_cell_cost,
                         double burning_cell_cost, double burnt_cell_cost, double damage_cost,
                         double water_cost, bool is_all_firefighters_disabled_terminal)
        : rows_(num_rows),
          cols_(num_cols),
          n_(num_firefighters),
          cells_(num_rows * num_cols),
          depot_row_(depot_row),
          depot_col_(depot_col),
          max_tank_(max_tank),
          rho_(sensing_radius),
          error_(observation_error_probability),
          slip_(slip_probability),
          spread_(spread_probability),
          gain_low_(wind_gain_low),
          gain_high_(wind_gain_high),
          crosswind_attenuation_(crosswind_attenuation),
          growth_(growth_probability),
          burnout_(burnout_probability),
          max_steps_(max_steps),
          success_reward_(success_reward),
          step_cost_(step_cost),
          smoldering_cost_(smoldering_cell_cost),
          burning_cost_(burning_cell_cost),
          burnt_cost_(burnt_cell_cost),
          damage_cost_(damage_cost),
          water_cost_(water_cost),
          disabled_terminal_(is_all_firefighters_disabled_terminal) {
        if (rows_ < 1 || cols_ < 1 || n_ < 1) {
            throw std::invalid_argument("grid and firefighter count must be positive");
        }
        if (obstacle_mask.size() != cells_) {
            throw std::invalid_argument("obstacle_mask must have num_rows * num_cols entries");
        }
        obstacle_.assign(obstacle_mask.data(), obstacle_mask.data() + cells_);
        suppression_[kUnburnt] = suppression_probability_unburnt;
        suppression_[kSmoldering] = suppression_probability_smoldering;
        suppression_[kBurning] = suppression_probability_burning;
        suppression_[kBurnt] = 0.0;
        suppression_[kWet] = 0.0;
        num_actions_ = 1;
        for (int i = 0; i < n_; ++i) {
            num_actions_ *= kNumFirefighterActions;
        }
        wind_direction_index_ = kFirefighterOffset + kFieldWidth * n_;
        wind_strength_index_ = wind_direction_index_ + 1;
        fire_offset_ = wind_strength_index_ + 1;
        state_size_ = fire_offset_ + cells_;
        observation_size_ = kFieldWidth * n_ + cells_;
    }

    // -- single-state entry points ------------------------------------

    py::array_t<double> sample_next_state(const Array &state, long action) {
        const double *s = checked_state(state, "state");
        py::array_t<double> out(state_size_);
        step(s, action, out.mutable_data());
        return out;
    }

    py::array_t<double> sample_next_states(const Array &state, long action, long n_samples) {
        const double *s = checked_state(state, "state");
        if (n_samples < 1) {
            throw std::invalid_argument("n_samples must be at least 1");
        }
        py::array_t<double> out({static_cast<py::ssize_t>(n_samples),
                                 static_cast<py::ssize_t>(state_size_)});
        double *o = out.mutable_data();
        for (long i = 0; i < n_samples; ++i) {
            step(s, action, o + i * state_size_);
        }
        return out;
    }

    py::array_t<double> sample_next_state_batch(const Array &states, long action) {
        const double *s = checked_rows(states, state_size_, "states");
        const py::ssize_t count = states.shape(0);
        py::array_t<double> out({count, static_cast<py::ssize_t>(state_size_)});
        double *o = out.mutable_data();
        for (py::ssize_t i = 0; i < count; ++i) {
            step(s + i * state_size_, action, o + i * state_size_);
        }
        return out;
    }

    py::array_t<double> transition_log_probability(const Array &state, long action,
                                                   const Array &candidates) {
        const double *s = checked_state(state, "state");
        const auto [count, width] = rows_of(candidates);
        py::array_t<double> out(count);
        double *o = out.mutable_data();
        const double *c = candidates.data();
        for (py::ssize_t i = 0; i < count; ++i) {
            o[i] = width == state_size_ ? successor_log_probability(s, action, c + i * width)
                                        : kNegInf;
        }
        return out;
    }

    py::array_t<double> sample_observation(const Array &next_state) {
        const double *s = checked_state(next_state, "next_state");
        py::array_t<double> out(observation_size_);
        draw_observation(s, out.mutable_data());
        return out;
    }

    py::array_t<double> sample_observations(const Array &next_state, long n_samples) {
        const double *s = checked_state(next_state, "next_state");
        if (n_samples < 1) {
            throw std::invalid_argument("n_samples must be at least 1");
        }
        py::array_t<double> out({static_cast<py::ssize_t>(n_samples),
                                 static_cast<py::ssize_t>(observation_size_)});
        double *o = out.mutable_data();
        for (long i = 0; i < n_samples; ++i) {
            draw_observation(s, o + i * observation_size_);
        }
        return out;
    }

    py::array_t<double> observation_log_probability(const Array &next_state,
                                                    const Array &observations) {
        const double *s = checked_state(next_state, "next_state");
        const auto [count, width] = rows_of(observations);
        py::array_t<double> out(count);
        double *o = out.mutable_data();
        const double *c = observations.data();
        visible_mask(s, visible_);
        for (py::ssize_t i = 0; i < count; ++i) {
            o[i] = width == observation_size_ ? observation_score(s, visible_, c + i * width)
                                              : kNegInf;
        }
        return out;
    }

    double reward(const Array &state, long action, const Array &next_state) {
        return reward_raw(checked_state(state, "state"), action,
                          checked_state(next_state, "next_state"));
    }

    py::array_t<double> reward_batch(const Array &states, long action, const Array &next_states) {
        const double *s = checked_rows(states, state_size_, "states");
        const double *ns = checked_rows(next_states, state_size_, "next_states");
        if (states.shape(0) != next_states.shape(0)) {
            throw std::invalid_argument("states and next_states must have the same length");
        }
        const py::ssize_t count = states.shape(0);
        py::array_t<double> out(count);
        double *o = out.mutable_data();
        for (py::ssize_t i = 0; i < count; ++i) {
            o[i] = reward_raw(s + i * state_size_, action, ns + i * state_size_);
        }
        return out;
    }

    bool is_terminal(const Array &state) {
        return terminal_raw(checked_state(state, "state"));
    }

    py::tuple sample_next_step(const Array &state, long action) {
        const double *s = checked_state(state, "state");
        py::array_t<double> next(state_size_);
        double *ns = next.mutable_data();
        step(s, action, ns);
        py::array_t<double> observation(observation_size_);
        draw_observation(ns, observation.mutable_data());
        const double r = reward_raw(s, action, ns);
        return py::make_tuple(next, observation, r);
    }

    // Random rollout with pre-drawn joint actions. Mirrors
    // python_random_rollout: stop at max_depth or at a terminal state, and
    // score each step against the successor it actually drew.
    double simulate_rollout(const Array &initial_state,
                            const py::array_t<std::int64_t, py::array::c_style |
                                                                py::array::forcecast> &actions,
                            long max_depth, long start_depth, double discount_factor) {
        const double *s0 = checked_state(initial_state, "initial_state");
        const long steps = max_depth - start_depth;
        if (steps <= 0) {
            return 0.0;
        }
        if (actions.size() < steps) {
            throw std::invalid_argument("actions must hold max_depth - start_depth entries");
        }
        const std::int64_t *a = actions.data();
        rollout_a_.assign(s0, s0 + state_size_);
        rollout_b_.resize(state_size_);
        double *current = rollout_a_.data();
        double *next = rollout_b_.data();
        double total = 0.0;
        double discount = 1.0;
        for (long d = 0; d < steps; ++d) {
            if (terminal_raw(current)) {
                break;
            }
            step(current, static_cast<long>(a[d]), next);
            total += discount * reward_raw(current, static_cast<long>(a[d]), next);
            discount *= discount_factor;
            std::swap(current, next);
        }
        return total;
    }

    int state_size() const { return state_size_; }
    int observation_size() const { return observation_size_; }
    long num_actions() const { return num_actions_; }

  private:
    // -- marshalling --------------------------------------------------

    const double *checked_state(const Array &array, const char *label) const {
        if (array.ndim() != 1 || array.shape(0) != state_size_) {
            throw std::invalid_argument(std::string(label) + " must be 1-D with length " +
                                        std::to_string(state_size_));
        }
        return array.data();
    }

    static const double *checked_rows(const Array &array, int width, const char *label) {
        if (array.ndim() != 2 || array.shape(1) != width) {
            throw std::invalid_argument(std::string(label) + " must be 2-D with " +
                                        std::to_string(width) + " columns");
        }
        return array.data();
    }

    // A 1-D candidate is one row, as np.atleast_2d makes it in Python.
    static std::pair<py::ssize_t, py::ssize_t> rows_of(const Array &array) {
        if (array.ndim() == 1) {
            return {1, array.shape(0)};
        }
        if (array.ndim() == 2) {
            return {array.shape(0), array.shape(1)};
        }
        throw std::invalid_argument("candidates must be 1-D or 2-D");
    }

    // -- state decoding -----------------------------------------------

    const std::vector<int> &decode(long action) {
        if (action < 0 || action >= num_actions_) {
            throw std::invalid_argument("joint action must be in [0, " +
                                        std::to_string(num_actions_) + "), got " +
                                        std::to_string(action));
        }
        actions_.resize(n_);
        long joint = action;
        for (int i = 0; i < n_; ++i) {
            actions_[i] = static_cast<int>(joint % kNumFirefighterActions);
            joint /= kNumFirefighterActions;
        }
        return actions_;
    }

    void read_firefighters(const double *s, std::vector<Firefighter> &out) const {
        out.resize(n_);
        for (int i = 0; i < n_; ++i) {
            const double *f = s + kFirefighterOffset + kFieldWidth * i;
            out[i] = {to_int(f[0]), to_int(f[1]), to_int(f[2]), to_int(f[3])};
        }
    }

    // The transition and its density index the fire map at each firefighter's
    // cell, so a firefighter off the grid would read outside it. The Python
    // reference raises IndexError there; this raises ValueError.
    void check_on_grid(const std::vector<Firefighter> &ff) const {
        for (const Firefighter &f : ff) {
            if (f.row < 0 || f.row >= rows_ || f.col < 0 || f.col >= cols_) {
                throw std::invalid_argument("firefighter at (" + std::to_string(f.row) + ", " +
                                            std::to_string(f.col) + ") is off the grid");
            }
        }
    }

    void read_fire(const double *s, std::vector<int> &out) const {
        out.resize(cells_);
        for (int k = 0; k < cells_; ++k) {
            out[k] = static_cast<int>(to_int(s[fire_offset_ + k]));
        }
    }

    bool admissible(long row, long col, const std::vector<int> &fire) const {
        if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
            return false;
        }
        const long k = row * cols_ + col;
        return !obstacle_[k] && fire[k] != kBurnt;
    }

    // Writes each firefighter's admissible target into targets_ (row, col),
    // or (-1, -1) when the firefighter is disabled, suppressing, or blocked.
    void move_targets(const std::vector<Firefighter> &ff, const std::vector<int> &acts,
                      const std::vector<int> &fire) {
        targets_.resize(2 * n_);
        for (int i = 0; i < n_; ++i) {
            targets_[2 * i] = -1;
            targets_[2 * i + 1] = -1;
            if (ff[i].health <= 0 || acts[i] == kSuppress) {
                continue;
            }
            const long row = ff[i].row + kOffsets[acts[i]][0];
            const long col = ff[i].col + kOffsets[acts[i]][1];
            if (admissible(row, col, fire)) {
                targets_[2 * i] = row;
                targets_[2 * i + 1] = col;
            }
        }
    }

    bool sprays(const Firefighter &f, int act) const {
        return f.health > 0 && act == kSuppress && f.tank > 0;
    }

    void coverage_counts(const std::vector<long> &positions, const std::vector<Firefighter> &ff,
                         const std::vector<int> &acts) {
        counts_.assign(cells_, 0);
        for (int i = 0; i < n_; ++i) {
            if (!sprays(ff[i], acts[i])) {
                continue;
            }
            const long row = positions[2 * i];
            const long col = positions[2 * i + 1];
            if (row >= 0 && row < rows_ && col >= 0 && col < cols_) {
                counts_[row * cols_ + col] += 1;
            }
            for (const auto &offset : kOffsets) {
                const long r = row + offset[0];
                const long c = col + offset[1];
                if (r >= 0 && r < rows_ && c >= 0 && c < cols_) {
                    counts_[r * cols_ + c] += 1;
                }
            }
        }
    }

    double soak_probability(int category, int count) const {
        const double per_spray =
            (category >= 0 && category < kNumCategories) ? suppression_[category] : 0.0;
        return 1.0 - std::pow(1.0 - per_spray, static_cast<double>(count));
    }

    // One minus the product, over alight four-neighbours of ``k`` in ``fire``,
    // of the per-neighbour survival. Multiplied in DIRECTION_OFFSETS order so
    // the double is the one numpy computes.
    double ignition_probability(int k, const std::vector<int> &fire, int direction,
                                int strength) const {
        const double gain = strength == kWindHigh ? gain_high_ : gain_low_;
        const double downwind_rate = std::min(1.0, spread_ * gain);
        const double crosswind_rate = spread_ * (1.0 - crosswind_attenuation_);
        const int row = k / cols_;
        const int col = k % cols_;
        double survive = 1.0;
        for (int code = 0; code < 4; ++code) {
            const int r = row - kOffsets[code][0];
            const int c = col - kOffsets[code][1];
            if (r < 0 || r >= rows_ || c < 0 || c >= cols_) {
                continue;
            }
            if (is_alight(fire[r * cols_ + c])) {
                const double rate = code == direction ? downwind_rate : crosswind_rate;
                survive = survive * (1.0 - rate);
            }
        }
        return 1.0 - survive;
    }

    // -- transition ---------------------------------------------------

    void step(const double *s, long action, double *out) {
        const std::vector<int> &acts = decode(action);
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        auto &engine = pomdp_native::default_rng().engine();

        read_firefighters(s, ff_);
        check_on_grid(ff_);
        read_fire(s, fire_);
        const int direction = static_cast<int>(to_int(s[wind_direction_index_]));
        const int strength = static_cast<int>(to_int(s[wind_strength_index_]));

        // 1. Motion; the slip draw is taken only where a move could succeed.
        move_targets(ff_, acts, fire_);
        positions_.resize(2 * n_);
        for (int i = 0; i < n_; ++i) {
            positions_[2 * i] = ff_[i].row;
            positions_[2 * i + 1] = ff_[i].col;
            if (targets_[2 * i] < 0) {
                continue;
            }
            if (unif(engine) >= slip_) {
                positions_[2 * i] = targets_[2 * i];
                positions_[2 * i + 1] = targets_[2 * i + 1];
            }
        }

        // 2. Suppression, the tank, then the depot refill overriding the cost.
        tanks_.resize(n_);
        bool any_sprayer = false;
        for (int i = 0; i < n_; ++i) {
            tanks_[i] = ff_[i].tank;
            if (sprays(ff_[i], acts[i])) {
                any_sprayer = true;
            }
        }
        if (any_sprayer) {
            coverage_counts(positions_, ff_, acts);
            for (int k = 0; k < cells_; ++k) {
                if (counts_[k] <= 0) {
                    continue;
                }
                const double p = soak_probability(fire_[k], counts_[k]);
                if (unif(engine) < p) {
                    fire_[k] = kWet;
                }
            }
            for (int i = 0; i < n_; ++i) {
                if (sprays(ff_[i], acts[i])) {
                    tanks_[i] -= 1;
                }
            }
        }
        for (int i = 0; i < n_; ++i) {
            if (positions_[2 * i] == depot_row_ && positions_[2 * i + 1] == depot_col_) {
                tanks_[i] = max_tank_;
            }
        }

        // 3. Spread, read from the post-suppression map.
        after_ = fire_;
        for (int k = 0; k < cells_; ++k) {
            if (after_[k] != kUnburnt || obstacle_[k]) {
                continue;
            }
            const double p = ignition_probability(k, after_, direction, strength);
            if (p > 0.0 && unif(engine) < p) {
                fire_[k] = kSmoldering;
            }
        }

        // 4. Growth and burnout, on cells already alight before the spread.
        for (int k = 0; k < cells_; ++k) {
            if (after_[k] == kSmoldering) {
                if (unif(engine) < growth_) {
                    fire_[k] = kBurning;
                }
            } else if (after_[k] == kBurning) {
                if (unif(engine) < burnout_) {
                    fire_[k] = kBurnt;
                }
            }
        }

        // 5. Heat damage on the final map, 6. bookkeeping.
        std::copy(s, s + state_size_, out);
        out[0] = s[0] + 1.0;
        for (int i = 0; i < n_; ++i) {
            const long cell = positions_[2 * i] * cols_ + positions_[2 * i + 1];
            const long damage = heat_damage(fire_[cell]);
            const long health = std::max<long>(0, ff_[i].health - damage);
            double *f = out + kFirefighterOffset + kFieldWidth * i;
            f[0] = static_cast<double>(positions_[2 * i]);
            f[1] = static_cast<double>(positions_[2 * i + 1]);
            f[2] = static_cast<double>(tanks_[i]);
            f[3] = static_cast<double>(health);
        }
        for (int k = 0; k < cells_; ++k) {
            out[fire_offset_ + k] = static_cast<double>(fire_[k]);
        }
    }

    // Exact log-probability of one candidate successor; mirrors
    // FirefightingPOMDP._successor_log_probability line for line.
    double successor_log_probability(const double *s, long action, const double *candidate) {
        for (int j = 0; j < state_size_; ++j) {
            if (!std::isfinite(candidate[j])) {
                return kNegInf;
            }
        }
        if (candidate[0] != s[0] + 1.0) {
            return kNegInf;
        }
        if (to_int(candidate[wind_direction_index_]) != to_int(s[wind_direction_index_]) ||
            to_int(candidate[wind_strength_index_]) != to_int(s[wind_strength_index_])) {
            return kNegInf;
        }
        const std::vector<int> &acts = decode(action);
        read_firefighters(s, ff_);
        check_on_grid(ff_);
        read_firefighters(candidate, next_ff_);
        read_fire(s, fire_);
        read_fire(candidate, next_fire_);

        // 1. Motion.
        double log_probability = 0.0;
        move_targets(ff_, acts, fire_);
        positions_.resize(2 * n_);
        for (int i = 0; i < n_; ++i) {
            positions_[2 * i] = next_ff_[i].row;
            positions_[2 * i + 1] = next_ff_[i].col;
            const bool stayed = next_ff_[i].row == ff_[i].row && next_ff_[i].col == ff_[i].col;
            if (targets_[2 * i] < 0) {
                if (!stayed) {
                    return kNegInf;
                }
                continue;
            }
            if (next_ff_[i].row == targets_[2 * i] && next_ff_[i].col == targets_[2 * i + 1]) {
                log_probability += std::log1p(-slip_);
            } else if (stayed) {
                log_probability += slip_ > 0.0 ? std::log(slip_) : kNegInf;
            } else {
                return kNegInf;
            }
        }
        if (!std::isfinite(log_probability)) {
            return kNegInf;
        }

        // 2. Suppression and the deterministic tank rule.
        coverage_counts(positions_, ff_, acts);
        for (int i = 0; i < n_; ++i) {
            long expected = ff_[i].tank;
            if (sprays(ff_[i], acts[i])) {
                expected -= 1;
            }
            if (positions_[2 * i] == depot_row_ && positions_[2 * i + 1] == depot_col_) {
                expected = max_tank_;
            }
            if (next_ff_[i].tank != expected) {
                return kNegInf;
            }
        }

        // 3-4. Per-cell map law over the recovered post-suppression map.
        after_ = fire_;
        for (int k = 0; k < cells_; ++k) {
            const int before = fire_[k];
            const int after = next_fire_[k];
            if (before == kBurnt || before == kWet) {
                if (after != before) {
                    return kNegInf;
                }
                continue;
            }
            const double soak = soak_probability(before, counts_[k]);
            if (after == kWet) {
                log_probability += std::log(soak);
                after_[k] = kWet;
                continue;
            }
            log_probability += std::log1p(-soak);
            if (!std::isfinite(log_probability)) {
                return kNegInf;
            }
            if (before == kUnburnt && after != kUnburnt && after != kSmoldering) {
                return kNegInf;
            }
            if (before == kSmoldering && after != kSmoldering && after != kBurning) {
                return kNegInf;
            }
            if (before == kBurning && after != kBurning && after != kBurnt) {
                return kNegInf;
            }
        }

        const int direction = static_cast<int>(to_int(s[wind_direction_index_]));
        const int strength = static_cast<int>(to_int(s[wind_strength_index_]));
        for (int k = 0; k < cells_; ++k) {
            const int before = fire_[k];
            const int after = next_fire_[k];
            if (after == kWet || before == kBurnt || before == kWet) {
                continue;
            }
            double probability;
            if (before == kUnburnt) {
                if (obstacle_[k]) {
                    if (after != kUnburnt) {
                        return kNegInf;
                    }
                    continue;
                }
                const double ignition = ignition_probability(k, after_, direction, strength);
                probability = after == kSmoldering ? ignition : 1.0 - ignition;
            } else if (before == kSmoldering) {
                probability = after == kBurning ? growth_ : 1.0 - growth_;
            } else {
                probability = after == kBurnt ? burnout_ : 1.0 - burnout_;
            }
            if (probability <= 0.0) {
                return kNegInf;
            }
            log_probability += std::log(probability);
        }

        // 5. Heat damage is deterministic given the final map and the poses.
        for (int i = 0; i < n_; ++i) {
            const long cell = positions_[2 * i] * cols_ + positions_[2 * i + 1];
            const long damage = heat_damage(next_fire_[cell]);
            if (next_ff_[i].health != std::max<long>(0, ff_[i].health - damage)) {
                return kNegInf;
            }
        }
        return log_probability;
    }

    // -- observation --------------------------------------------------

    void visible_mask(const double *s, std::vector<std::uint8_t> &visible) {
        read_firefighters(s, obs_ff_);
        visible.assign(cells_, 0);
        for (int i = 0; i < n_; ++i) {
            if (obs_ff_[i].health <= 0) {
                continue;
            }
            const long r0 = std::max<long>(0, obs_ff_[i].row - rho_);
            const long r1 = std::min<long>(rows_ - 1, obs_ff_[i].row + rho_);
            const long c0 = std::max<long>(0, obs_ff_[i].col - rho_);
            const long c1 = std::min<long>(cols_ - 1, obs_ff_[i].col + rho_);
            for (long r = r0; r <= r1; ++r) {
                for (long c = c0; c <= c1; ++c) {
                    visible[r * cols_ + c] = 1;
                }
            }
        }
    }

    void draw_observation(const double *s, double *out) {
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        std::uniform_int_distribution<int> offset(1, kNumCategories - 1);
        auto &engine = pomdp_native::default_rng().engine();
        visible_mask(s, visible_);
        std::copy(s + kFirefighterOffset, s + kFirefighterOffset + kFieldWidth * n_, out);
        double *reported = out + kFieldWidth * n_;
        for (int k = 0; k < cells_; ++k) {
            if (!visible_[k]) {
                reported[k] = kUnknown;
                continue;
            }
            const long truth = to_int(s[fire_offset_ + k]);
            if (unif(engine) < error_) {
                // A uniform offset of 1..4 lands on each wrong category with
                // equal probability, as in the Python sampler. Python's ``%``
                // is a floor modulo, so a negative sum wraps the same way here.
                long value = (truth + offset(engine)) % kNumCategories;
                if (value < 0) {
                    value += kNumCategories;
                }
                reported[k] = static_cast<double>(value);
            } else {
                reported[k] = static_cast<double>(truth);
            }
        }
    }

    double observation_score(const double *s, const std::vector<std::uint8_t> &visible,
                             const double *candidate) const {
        const int fields = kFieldWidth * n_;
        for (int j = 0; j < fields; ++j) {
            if (!(candidate[j] == s[kFirefighterOffset + j])) {
                return kNegInf;
            }
        }
        const double *reported = candidate + fields;
        long matches = 0;
        long mismatches = 0;
        for (int k = 0; k < cells_; ++k) {
            if (!visible[k]) {
                if (reported[k] != kUnknown) {
                    return kNegInf;
                }
            }
        }
        for (int k = 0; k < cells_; ++k) {
            if (!visible[k]) {
                continue;
            }
            const double value = reported[k];
            if (value < 0.0 || value >= kNumCategories || std::fmod(value, 1.0) != 0.0) {
                return kNegInf;
            }
            if (value == static_cast<double>(to_int(s[fire_offset_ + k]))) {
                ++matches;
            } else {
                ++mismatches;
            }
        }
        // Each term only when its count is non-zero: 0 * -inf is NaN at the
        // two legal error endpoints, as in the Python method.
        const double log_correct = std::log1p(-error_);
        const double log_wrong = std::log(error_ / (kNumCategories - 1));
        double score = 0.0;
        if (matches) {
            score += static_cast<double>(matches) * log_correct;
        }
        if (mismatches) {
            score += static_cast<double>(mismatches) * log_wrong;
        }
        return score;
    }

    // -- reward and termination ---------------------------------------

    double reward_raw(const double *s, long action, const double *ns) {
        const std::vector<int> &acts = decode(action);
        long smoldering = 0;
        long burning = 0;
        long newly_burnt = 0;
        for (int k = 0; k < cells_; ++k) {
            const long before = to_int(s[fire_offset_ + k]);
            const long after = to_int(ns[fire_offset_ + k]);
            smoldering += after == kSmoldering;
            burning += after == kBurning;
            newly_burnt += (after == kBurnt) && (before != kBurnt);
        }
        long health_lost = 0;
        long sprays_used = 0;
        for (int i = 0; i < n_; ++i) {
            const double *f = s + kFirefighterOffset + kFieldWidth * i;
            const double *g = ns + kFirefighterOffset + kFieldWidth * i;
            health_lost += to_int(f[3]) - to_int(g[3]);
            const Firefighter before{to_int(f[0]), to_int(f[1]), to_int(f[2]), to_int(f[3])};
            sprays_used += sprays(before, acts[i]);
        }
        double r = -step_cost_;
        r -= smoldering_cost_ * static_cast<double>(smoldering);
        r -= burning_cost_ * static_cast<double>(burning);
        r -= burnt_cost_ * static_cast<double>(newly_burnt);
        r -= damage_cost_ * static_cast<double>(health_lost);
        r -= water_cost_ * static_cast<double>(sprays_used);
        if (smoldering == 0 && burning == 0) {
            r += success_reward_;
        }
        return r;
    }

    bool terminal_raw(const double *s) const {
        bool any_alight = false;
        for (int k = 0; k < cells_ && !any_alight; ++k) {
            any_alight = is_alight(static_cast<int>(to_int(s[fire_offset_ + k])));
        }
        if (!any_alight) {
            return true;
        }
        if (disabled_terminal_) {
            bool any_live = false;
            for (int i = 0; i < n_ && !any_live; ++i) {
                any_live = to_int(s[kFirefighterOffset + kFieldWidth * i + 3]) > 0;
            }
            if (!any_live) {
                return true;
            }
        }
        return to_int(s[0]) >= max_steps_;
    }

    // -- configuration ------------------------------------------------
    int rows_;
    int cols_;
    int n_;
    int cells_;
    long depot_row_;
    long depot_col_;
    long max_tank_;
    long rho_;
    double error_;
    double slip_;
    double spread_;
    double gain_low_;
    double gain_high_;
    double crosswind_attenuation_;
    double growth_;
    double burnout_;
    long max_steps_;
    double success_reward_;
    double step_cost_;
    double smoldering_cost_;
    double burning_cost_;
    double burnt_cost_;
    double damage_cost_;
    double water_cost_;
    bool disabled_terminal_;
    double suppression_[kNumCategories];
    std::vector<std::uint8_t> obstacle_;
    long num_actions_;
    int wind_direction_index_;
    int wind_strength_index_;
    int fire_offset_;
    int state_size_;
    int observation_size_;

    // -- scratch buffers, reused across calls (the GIL serialises them) --
    std::vector<int> actions_;
    std::vector<Firefighter> ff_;
    std::vector<Firefighter> next_ff_;
    std::vector<Firefighter> obs_ff_;
    std::vector<int> fire_;
    std::vector<int> next_fire_;
    std::vector<int> after_;
    std::vector<int> counts_;
    std::vector<long> targets_;
    std::vector<long> positions_;
    std::vector<long> tanks_;
    std::vector<std::uint8_t> visible_;
    std::vector<double> rollout_a_;
    std::vector<double> rollout_b_;
};

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Firefighting POMDP native C++ kernels (pomdp_native).";

    m.def(
        "set_seed", [](std::uint64_t seed) { pomdp_native::set_default_seed(seed); },
        py::arg("seed"), "Seed the module-local RNG every sampling entry point draws from.");

    py::class_<FirefightingModelCpp>(m, "FirefightingModelCpp")
        .def(py::init<int, int, int,
                      py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>, int,
                      int, int, int, double, double, double, double, double, double, double,
                      double, double, double, double, int, double, double, double, double, double,
                      double, double, bool>(),
             py::arg("num_rows"), py::arg("num_cols"), py::arg("num_firefighters"),
             py::arg("obstacle_mask"), py::arg("depot_row"), py::arg("depot_col"),
             py::arg("max_tank"), py::arg("sensing_radius"),
             py::arg("observation_error_probability"), py::arg("slip_probability"),
             py::arg("spread_probability"), py::arg("wind_gain_low"), py::arg("wind_gain_high"),
             py::arg("crosswind_attenuation"), py::arg("growth_probability"),
             py::arg("burnout_probability"), py::arg("suppression_probability_unburnt"),
             py::arg("suppression_probability_smoldering"),
             py::arg("suppression_probability_burning"), py::arg("max_steps"),
             py::arg("success_reward"), py::arg("step_cost"), py::arg("smoldering_cell_cost"),
             py::arg("burning_cell_cost"), py::arg("burnt_cell_cost"), py::arg("damage_cost"),
             py::arg("water_cost"), py::arg("is_all_firefighters_disabled_terminal"))
        .def("sample_next_state", &FirefightingModelCpp::sample_next_state, py::arg("state"),
             py::arg("action"))
        .def("sample_next_states", &FirefightingModelCpp::sample_next_states, py::arg("state"),
             py::arg("action"), py::arg("n_samples"))
        .def("sample_next_state_batch", &FirefightingModelCpp::sample_next_state_batch,
             py::arg("states"), py::arg("action"))
        .def("transition_log_probability", &FirefightingModelCpp::transition_log_probability,
             py::arg("state"), py::arg("action"), py::arg("next_states"))
        .def("sample_observation", &FirefightingModelCpp::sample_observation,
             py::arg("next_state"))
        .def("sample_observations", &FirefightingModelCpp::sample_observations,
             py::arg("next_state"), py::arg("n_samples"))
        .def("observation_log_probability", &FirefightingModelCpp::observation_log_probability,
             py::arg("next_state"), py::arg("observations"))
        .def("reward", &FirefightingModelCpp::reward, py::arg("state"), py::arg("action"),
             py::arg("next_state"))
        .def("reward_batch", &FirefightingModelCpp::reward_batch, py::arg("states"),
             py::arg("action"), py::arg("next_states"))
        .def("is_terminal", &FirefightingModelCpp::is_terminal, py::arg("state"))
        .def("sample_next_step", &FirefightingModelCpp::sample_next_step, py::arg("state"),
             py::arg("action"))
        .def("simulate_rollout", &FirefightingModelCpp::simulate_rollout,
             py::arg("initial_state"), py::arg("actions"), py::arg("max_depth"),
             py::arg("start_depth"), py::arg("discount_factor"))
        .def_property_readonly("state_size", &FirefightingModelCpp::state_size)
        .def_property_readonly("observation_size", &FirefightingModelCpp::observation_size)
        .def_property_readonly("num_actions", &FirefightingModelCpp::num_actions);
}
