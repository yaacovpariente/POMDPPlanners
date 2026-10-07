// SPDX-License-Identifier: MIT

// Chicheck Invaders POMDP native hot path.
//
// A line-by-line port of the scalar model in ``chicheck_invaders_pomdp.py``:
// the step (ship move, hitscan shot, dive coins, flock move, arrivals), the
// transition log-probability, the two-sensor observation sampler and its
// log-likelihood, the reward, the terminal check, and a random rollout.
//
// State layout (length 4 + 5 N)::
//
//     [ step | ship column | cooldown | ship hit
//       | chicken i: column, row, direction, mode, alive ]
//
// Observation layout (length 1 + 5 N, partial mode)::
//
//     [ own-column reading
//       | chicken i: camera reported, camera offset, radar reported,
//                    radar rows, radar drop ]
//
// In full mode the observation is the state itself.
//
// Every deterministic quantity here is computed with the same operations in
// the same order as the Python reference, so transitions, rewards, terminal
// checks and the transition log-probability agree exactly. The rounded-normal
// likelihood calls ``erf`` / ``erfc`` from the C++ standard library where the
// Python reference calls SciPy's ``ndtr``; the two can differ in the last bit,
// so the observation log-likelihood agrees to rounding rather than exactly.
//
// The random draws come from this module's own RNG (``set_seed``), not from
// ``np.random``. The laws are the same as the Python reference -- one uniform
// dive coin per slot per step, a standard normal per noisy reading, one
// uniform per detection and per drop-flag flip -- but the streams differ.
//
// The state dimension depends on ``num_chickens``, so this extension does not
// use the fixed-dimension ``TransitionModelCpp<Dim>`` base, the same choice
// the RockSample and PacMan ports made.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "pomdp_native/rng.hpp"

// Clang fuses ``a * b + c`` into one fused multiply-add on ARM by default,
// which rounds once where Python rounds twice. That changes the last bit of a
// discounted return and can move the radar's ``dx * dx + dy * dy <= r * r``
// test at the disc edge, so contraction is off for this file. GCC already
// leaves it off under ``-std=c++17``.
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;

// Mirrors ``IMPOSSIBLE_LOG_PROBABILITY`` in chicheck_invaders_pomdp.py.
constexpr double kImpossibleLogProbability = -1e18;

// Layout constants mirror chicheck_invaders_schema.py.
constexpr std::size_t kStepIndex = 0;
constexpr std::size_t kShipColumnIndex = 1;
constexpr std::size_t kCooldownIndex = 2;
constexpr std::size_t kShipHitIndex = 3;
constexpr std::size_t kShipWidth = 4;
constexpr std::size_t kChickenWidth = 5;
constexpr std::size_t kChickenColumn = 0;
constexpr std::size_t kChickenRow = 1;
constexpr std::size_t kChickenDirection = 2;
constexpr std::size_t kChickenMode = 3;
constexpr std::size_t kChickenAlive = 4;
constexpr double kModePatrol = 0.0;
constexpr double kModeDive = 1.0;

constexpr std::size_t kObservedShipColumnIndex = 0;
constexpr std::size_t kObservationShipWidth = 1;
constexpr std::size_t kObservationChickenWidth = 5;
constexpr std::size_t kObservedCameraReported = 0;
constexpr std::size_t kObservedCameraOffset = 1;
constexpr std::size_t kObservedRadarReported = 2;
constexpr std::size_t kObservedRadarRows = 3;
constexpr std::size_t kObservedRadarDrop = 4;

// Action indices mirror ``ChicheckInvadersAction``.
constexpr int kActionLeft = 1;
constexpr int kActionRight = 2;
constexpr int kActionFire = 3;
constexpr int kNumActions = 4;

// Python's ``int(round(x))``: round half to even, then truncate to an integer.
// ``std::nearbyint`` rounds half to even under the default rounding mode.
inline long python_round(double value) { return static_cast<long>(std::nearbyint(value)); }

// Python's ``int(x)`` on a float: truncation toward zero.
inline long python_int(double value) { return static_cast<long>(value); }

// SciPy's ``ndtr`` (Cephes), written with the same branches so the two differ
// only where the platform ``erf`` / ``erfc`` differ from Cephes' own.
inline double ndtr(double a) {
    if (std::isnan(a)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    constexpr double kSqrt1_2 = 0.70710678118654752440;
    const double x = a * kSqrt1_2;
    const double z = std::fabs(x);
    if (z < kSqrt1_2) {
        return 0.5 + 0.5 * std::erf(x);
    }
    double y = 0.5 * std::erfc(z);
    if (x > 0.0) {
        y = 1.0 - y;
    }
    return y;
}

// ``rounded_normal_pmf`` for one reading.
inline double rounded_normal_pmf(double reading, double mean, double std_dev) {
    if (std_dev <= 0.0) {
        return reading == mean ? 1.0 : 0.0;
    }
    const double upper = (reading + 0.5 - mean) / std_dev;
    const double lower = (reading - 0.5 - mean) / std_dev;
    const double mass = lower > 0.0 ? ndtr(-lower) - ndtr(-upper) : ndtr(upper) - ndtr(lower);
    return std::max(mass, 0.0);
}

// ``ChicheckInvadersPOMDP._log``.
inline double floored_log(double value) {
    return value > 0.0 ? std::log(value) : kImpossibleLogProbability;
}

inline double uniform01(std::mt19937_64 &engine) {
    return std::uniform_real_distribution<double>(0.0, 1.0)(engine);
}

inline double standard_normal(std::mt19937_64 &engine) {
    return std::normal_distribution<double>(0.0, 1.0)(engine);
}

// ``sample_rounded_normal``.
inline double sample_rounded_normal(double mean, double std_dev, std::mt19937_64 &engine) {
    if (std_dev <= 0.0) {
        return static_cast<double>(python_round(mean));
    }
    return static_cast<double>(python_round(mean + std_dev * standard_normal(engine)));
}

// Read a 1-D float64 array of a fixed length into ``out``.
void read_vector(const DoubleArray &array, std::size_t expected, const char *label,
                 std::vector<double> &out) {
    if (array.ndim() != 1 || static_cast<std::size_t>(array.shape(0)) != expected) {
        throw std::invalid_argument(std::string(label) + " must be 1-D with length " +
                                    std::to_string(expected));
    }
    const double *data = array.data();
    out.assign(data, data + expected);
}

// Check a 2-D float64 array has ``expected`` columns; return its row count.
std::size_t check_rows(const DoubleArray &array, std::size_t expected, const char *label) {
    if (array.ndim() != 2 || static_cast<std::size_t>(array.shape(1)) != expected) {
        throw std::invalid_argument(std::string(label) + " must have shape (N, " +
                                    std::to_string(expected) + ")");
    }
    return static_cast<std::size_t>(array.shape(0));
}

py::array_t<double> to_numpy(const std::vector<double> &values) {
    py::array_t<double> out(static_cast<py::ssize_t>(values.size()));
    std::copy(values.begin(), values.end(), out.mutable_data());
    return out;
}

// The environment's configuration and every model the port covers, written
// against raw pointers so the scalar and batched entry points share one body.
class ChicheckInvadersConfigCpp {
  public:
    // One argument per ChicheckInvadersPOMDP constructor parameter that the
    // dynamics, the sensors or the reward read.
    ChicheckInvadersConfigCpp(int num_columns, int num_rows, int num_chickens, int fire_cooldown,
                              double dive_probability, double camera_detection_probability,
                              double radar_detection_probability, double ship_column_noise_std,
                              double camera_offset_noise_std, double radar_range_noise_std,
                              double drop_flag_error_probability, double camera_slope,
                              double radar_radius, bool full_observation, double kill_reward,
                              double shot_cost, double step_cost, double ship_hit_penalty,
                              double clear_reward, int max_steps)
        : num_columns_(num_columns),
          num_rows_(num_rows),
          num_chickens_(num_chickens),
          fire_cooldown_(fire_cooldown),
          dive_probability_(dive_probability),
          camera_detection_probability_(camera_detection_probability),
          radar_detection_probability_(radar_detection_probability),
          ship_column_noise_std_(ship_column_noise_std),
          camera_offset_noise_std_(camera_offset_noise_std),
          radar_range_noise_std_(radar_range_noise_std),
          drop_flag_error_probability_(drop_flag_error_probability),
          camera_slope_(camera_slope),
          radar_radius_(radar_radius),
          full_observation_(full_observation),
          kill_reward_(kill_reward),
          shot_cost_(shot_cost),
          step_cost_(step_cost),
          ship_hit_penalty_(ship_hit_penalty),
          clear_reward_(clear_reward),
          max_steps_(max_steps) {
        if (num_chickens < 1 || num_columns < 2 || num_rows < 2) {
            throw std::invalid_argument("invalid Chicheck Invaders geometry");
        }
    }

    std::size_t state_size() const {
        return kShipWidth + kChickenWidth * static_cast<std::size_t>(num_chickens_);
    }

    std::size_t observation_size() const {
        if (full_observation_) {
            return state_size();
        }
        return kObservationShipWidth +
               kObservationChickenWidth * static_cast<std::size_t>(num_chickens_);
    }

    int num_chickens() const { return num_chickens_; }

    // -- state accessors --------------------------------------------------

    static const double *chicken(const double *state, int index) {
        return state + kShipWidth + kChickenWidth * static_cast<std::size_t>(index);
    }

    static double *chicken(double *state, int index) {
        return state + kShipWidth + kChickenWidth * static_cast<std::size_t>(index);
    }

    int live_chicken_count(const double *state) const {
        int count = 0;
        for (int i = 0; i < num_chickens_; ++i) {
            if (chicken(state, i)[kChickenAlive] > 0.0) {
                ++count;
            }
        }
        return count;
    }

    static bool fires(const double *state, int action) {
        return action == kActionFire && state[kCooldownIndex] <= 0.0;
    }

    int shot_target(const double *state) const {
        const long column = python_round(state[kShipColumnIndex]);
        int target = -1;
        double lowest = std::numeric_limits<double>::infinity();
        for (int i = 0; i < num_chickens_; ++i) {
            const double *slot = chicken(state, i);
            if (slot[kChickenAlive] <= 0.0) {
                continue;
            }
            if (python_int(slot[kChickenColumn]) != column) {
                continue;
            }
            const double row = slot[kChickenRow];
            if (row < 1.0) {
                continue;
            }
            if (row < lowest) {
                target = i;
                lowest = row;
            }
        }
        return target;
    }

    long moved_ship_column(const double *state, int action) const {
        long delta = 0;
        if (action == kActionLeft) {
            delta = -1;
        } else if (action == kActionRight) {
            delta = 1;
        }
        const long moved = python_round(state[kShipColumnIndex]) + delta;
        return std::min(std::max(moved, 0L), static_cast<long>(num_columns_ - 1));
    }

    // -- dynamics -----------------------------------------------------------

    // ``_step_once``. ``coins`` holds one dive coin per slot; ``out`` must not
    // alias ``state``.
    void step(const double *state, int action, const char *coins, double *out) const {
        const std::size_t size = state_size();
        std::copy(state, state + size, out);
        out[kStepIndex] = state[kStepIndex] + 1.0;

        const bool fired = fires(state, action);
        const long ship_column = moved_ship_column(state, action);
        out[kShipColumnIndex] = static_cast<double>(ship_column);
        out[kCooldownIndex] = fired ? static_cast<double>(fire_cooldown_)
                                    : std::max(state[kCooldownIndex] - 1.0, 0.0);

        if (fired) {
            const int target = shot_target(out);
            if (target >= 0) {
                chicken(out, target)[kChickenAlive] = 0.0;
            }
        }

        const double right_wall = static_cast<double>(num_columns_ - 1);
        for (int i = 0; i < num_chickens_; ++i) {
            double *slot = chicken(out, i);
            if (!(slot[kChickenAlive] > 0.0)) {
                continue;
            }
            if (slot[kChickenMode] == kModePatrol && coins[i] != 0) {
                slot[kChickenMode] = kModeDive;
            }
            if (slot[kChickenMode] == kModeDive) {
                slot[kChickenRow] -= 1.0;
                continue;
            }
            const double column = slot[kChickenColumn];
            double direction = slot[kChickenDirection];
            const double stepped = column + direction;
            if (stepped < 0.0 || stepped > right_wall) {
                direction = -direction;
            }
            slot[kChickenDirection] = direction;
            slot[kChickenColumn] = column + direction;
        }

        // ``_resolve_arrivals``.
        for (int i = 0; i < num_chickens_; ++i) {
            double *slot = chicken(out, i);
            if (slot[kChickenAlive] <= 0.0 || slot[kChickenRow] > 0.0) {
                continue;
            }
            if (python_int(slot[kChickenColumn]) == ship_column) {
                slot[kChickenRow] = 0.0;
                out[kShipHitIndex] = 1.0;
                continue;
            }
            slot[kChickenRow] = static_cast<double>(num_rows_ - 1);
            slot[kChickenMode] = kModePatrol;
        }
    }

    // ``_draw_dive_switches``: one coin per slot, alive or not.
    void draw_coins(std::mt19937_64 &engine, std::vector<char> &coins) const {
        coins.resize(static_cast<std::size_t>(num_chickens_));
        for (int i = 0; i < num_chickens_; ++i) {
            coins[static_cast<std::size_t>(i)] = uniform01(engine) < dive_probability_ ? 1 : 0;
        }
    }

    void sample_step(const double *state, int action, std::mt19937_64 &engine,
                     std::vector<char> &coins, double *out) const {
        draw_coins(engine, coins);
        step(state, action, coins.data(), out);
    }

    // ``coin_eligible_slots``.
    void coin_eligible(const double *state, int action, std::vector<char> &eligible) const {
        eligible.resize(static_cast<std::size_t>(num_chickens_));
        for (int i = 0; i < num_chickens_; ++i) {
            const double *slot = chicken(state, i);
            eligible[static_cast<std::size_t>(i)] =
                (slot[kChickenAlive] > 0.0 && slot[kChickenMode] == kModePatrol) ? 1 : 0;
        }
        if (fires(state, action)) {
            const int target = shot_target(state);
            if (target >= 0) {
                eligible[static_cast<std::size_t>(target)] = 0;
            }
        }
    }

    // ``_implied_dive_switches``.
    void implied_switches(const double *before, const double *candidate,
                          std::vector<char> &switches) const {
        switches.resize(static_cast<std::size_t>(num_chickens_));
        const double top_row = static_cast<double>(num_rows_ - 1);
        for (int i = 0; i < num_chickens_; ++i) {
            const double *prior = chicken(before, i);
            const double *after = chicken(candidate, i);
            const bool was_patrolling = prior[kChickenMode] == kModePatrol;
            const bool now_diving = after[kChickenMode] == kModeDive;
            const bool pulled_up = after[kChickenAlive] > 0.0 &&
                                   after[kChickenMode] == kModePatrol &&
                                   prior[kChickenRow] == 1.0 && after[kChickenRow] == top_row &&
                                   after[kChickenColumn] == prior[kChickenColumn];
            switches[static_cast<std::size_t>(i)] =
                (was_patrolling && (now_diving || pulled_up)) ? 1 : 0;
        }
    }

    // ``_dive_coin_log_probability``.
    double coin_log_probability(const std::vector<char> &eligible,
                                const std::vector<char> &switches) const {
        const auto n = static_cast<std::size_t>(num_chickens_);
        if (dive_probability_ <= 0.0) {
            for (std::size_t i = 0; i < n; ++i) {
                if (switches[i] && eligible[i]) {
                    return kImpossibleLogProbability;
                }
            }
            return 0.0;
        }
        if (dive_probability_ >= 1.0) {
            for (std::size_t i = 0; i < n; ++i) {
                if (eligible[i] && !switches[i]) {
                    return kImpossibleLogProbability;
                }
            }
            return 0.0;
        }
        long switched = 0;
        long eligible_count = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (eligible[i]) {
                ++eligible_count;
                if (switches[i]) {
                    ++switched;
                }
            }
        }
        const long held = eligible_count - switched;
        return static_cast<double>(switched) * std::log(dive_probability_) +
               static_cast<double>(held) * std::log(1.0 - dive_probability_);
    }

    // ``transition_log_probability`` for one well-formed candidate.
    double transition_log_probability(const double *state, int action, const double *candidate,
                                      const std::vector<char> &eligible,
                                      std::vector<char> &switches,
                                      std::vector<double> &rebuilt) const {
        implied_switches(state, candidate, switches);
        rebuilt.resize(state_size());
        step(state, action, switches.data(), rebuilt.data());
        for (std::size_t d = 0; d < rebuilt.size(); ++d) {
            if (!(rebuilt[d] == candidate[d])) {
                return kImpossibleLogProbability;
            }
        }
        return coin_log_probability(eligible, switches);
    }

    // -- observations -------------------------------------------------------

    bool camera_sees(double column_offset, double row) const {
        return std::fabs(column_offset) <= camera_slope_ * row;
    }

    bool radar_sees(double column_offset, double row) const {
        return column_offset * column_offset + row * row <= radar_radius_ * radar_radius_;
    }

    // ``_sample_observation_once``.
    void sample_observation(const double *next_state, std::mt19937_64 &engine,
                            double *out) const {
        if (full_observation_) {
            std::copy(next_state, next_state + state_size(), out);
            return;
        }
        std::fill(out, out + observation_size(), 0.0);
        out[kObservedShipColumnIndex] =
            sample_rounded_normal(next_state[kShipColumnIndex], ship_column_noise_std_, engine);
        for (int i = 0; i < num_chickens_; ++i) {
            const double *slot = chicken(next_state, i);
            const bool alive = slot[kChickenAlive] > 0.0;
            const double column_offset = slot[kChickenColumn] - next_state[kShipColumnIndex];
            const double row = slot[kChickenRow];
            double *reading = out + kObservationShipWidth +
                              kObservationChickenWidth * static_cast<std::size_t>(i);
            if (alive && camera_sees(column_offset, row) &&
                uniform01(engine) < camera_detection_probability_) {
                reading[kObservedCameraReported] = 1.0;
                reading[kObservedCameraOffset] =
                    sample_rounded_normal(column_offset, camera_offset_noise_std_, engine);
            }
            if (alive && radar_sees(column_offset, row) &&
                uniform01(engine) < radar_detection_probability_) {
                reading[kObservedRadarReported] = 1.0;
                reading[kObservedRadarRows] =
                    sample_rounded_normal(row, radar_range_noise_std_, engine);
                const double true_drop = slot[kChickenMode] == kModeDive ? -1.0 : 0.0;
                const bool flipped = uniform01(engine) < drop_flag_error_probability_;
                reading[kObservedRadarDrop] = flipped ? (-1.0 - true_drop) : true_drop;
            }
        }
    }

    double camera_log_factor(bool reported, double reading, bool in_reach, double truth) const {
        if (!in_reach) {
            return reported ? kImpossibleLogProbability : 0.0;
        }
        if (!reported) {
            return floored_log(1.0 - camera_detection_probability_);
        }
        return floored_log(camera_detection_probability_) +
               floored_log(rounded_normal_pmf(reading, truth, camera_offset_noise_std_));
    }

    double radar_log_factor(bool reported, double rows, double drop, bool in_reach, double truth,
                            double true_drop) const {
        if (!in_reach) {
            return reported ? kImpossibleLogProbability : 0.0;
        }
        if (!reported) {
            return floored_log(1.0 - radar_detection_probability_);
        }
        const double flag = drop == true_drop ? 1.0 - drop_flag_error_probability_
                                              : drop_flag_error_probability_;
        return floored_log(radar_detection_probability_) +
               floored_log(rounded_normal_pmf(rows, truth, radar_range_noise_std_)) +
               floored_log(flag);
    }

    // ``_observation_log_likelihood`` for an observation of the right length.
    double observation_log_likelihood(const double *next_state, const double *observation) const {
        if (full_observation_) {
            for (std::size_t d = 0; d < state_size(); ++d) {
                if (!(next_state[d] == observation[d])) {
                    return kImpossibleLogProbability;
                }
            }
            return 0.0;
        }
        double total = floored_log(rounded_normal_pmf(observation[kObservedShipColumnIndex],
                                                      next_state[kShipColumnIndex],
                                                      ship_column_noise_std_));
        for (int i = 0; i < num_chickens_; ++i) {
            const double *slot = chicken(next_state, i);
            const bool alive = slot[kChickenAlive] > 0.0;
            const double column_offset = slot[kChickenColumn] - next_state[kShipColumnIndex];
            const double row = slot[kChickenRow];
            const double *reading = observation + kObservationShipWidth +
                                    kObservationChickenWidth * static_cast<std::size_t>(i);
            total += camera_log_factor(reading[kObservedCameraReported] > 0.0,
                                       reading[kObservedCameraOffset],
                                       alive && camera_sees(column_offset, row), column_offset);
            total += radar_log_factor(reading[kObservedRadarReported] > 0.0,
                                      reading[kObservedRadarRows], reading[kObservedRadarDrop],
                                      alive && radar_sees(column_offset, row), row,
                                      slot[kChickenMode] == kModeDive ? -1.0 : 0.0);
            if (total <= kImpossibleLogProbability) {
                return kImpossibleLogProbability;
            }
        }
        return total;
    }

    // -- reward and terminal ------------------------------------------------

    // ``reward``. ``next_state`` may be null, which is the no-successor branch.
    double reward(const double *state, int action, const double *next_state) const {
        const bool fired = fires(state, action);
        double charged = -step_cost_ - (fired ? shot_cost_ : 0.0);
        if (next_state == nullptr) {
            if (!fired || shot_target(state) < 0) {
                return charged;
            }
            charged += kill_reward_;
            if (live_chicken_count(state) == 1) {
                charged += clear_reward_;
            }
            return charged;
        }
        const int before = live_chicken_count(state);
        const int after = live_chicken_count(next_state);
        double total = charged + kill_reward_ * static_cast<double>(before - after);
        const bool was_hit = state[kShipHitIndex] > 0.0;
        const bool is_hit = next_state[kShipHitIndex] > 0.0;
        if (is_hit && !was_hit) {
            total -= ship_hit_penalty_;
        }
        if (after == 0 && before > 0) {
            total += clear_reward_;
        }
        return total;
    }

    bool is_terminal(const double *state) const {
        if (state[kShipHitIndex] > 0.0) {
            return true;
        }
        if (python_round(state[kStepIndex]) >= max_steps_) {
            return true;
        }
        return live_chicken_count(state) == 0;
    }

  private:
    int num_columns_;
    int num_rows_;
    int num_chickens_;
    int fire_cooldown_;
    double dive_probability_;
    double camera_detection_probability_;
    double radar_detection_probability_;
    double ship_column_noise_std_;
    double camera_offset_noise_std_;
    double radar_range_noise_std_;
    double drop_flag_error_probability_;
    double camera_slope_;
    double radar_radius_;
    bool full_observation_;
    double kill_reward_;
    double shot_cost_;
    double step_cost_;
    double ship_hit_penalty_;
    double clear_reward_;
    int max_steps_;
};

using Config = ChicheckInvadersConfigCpp;

// Transition kernel for one action, with the RockSample / PacMan surface:
// ``set_state`` + ``sample`` / ``log_probability`` for the stored state, and
// ``batch_sample`` over a particle array.
class ChicheckInvadersTransitionCpp {
  public:
    ChicheckInvadersTransitionCpp(const DoubleArray &state, int action, const Config &config)
        : config_(config), action_(action) {
        if (action < 0 || action >= kNumActions) {
            throw std::invalid_argument("action must be in [0, 4)");
        }
        read_vector(state, config_.state_size(), "state", state_);
    }

    void set_state(const DoubleArray &state) {
        read_vector(state, config_.state_size(), "state", state_);
    }

    py::list sample(int n_samples) {
        if (n_samples < 0) {
            throw std::invalid_argument("n_samples must be non-negative");
        }
        auto &engine = pomdp_native::default_rng().engine();
        py::list out;
        for (int i = 0; i < n_samples; ++i) {
            py::array_t<double> successor(static_cast<py::ssize_t>(config_.state_size()));
            config_.sample_step(state_.data(), action_, engine, coins_, successor.mutable_data());
            out.append(successor);
        }
        return out;
    }

    // Sample one successor of ``state`` without storing it: the env's hot path.
    py::array_t<double> sample_from(const DoubleArray &state) {
        const std::size_t size = config_.state_size();
        if (state.ndim() != 1 || static_cast<std::size_t>(state.shape(0)) != size) {
            throw std::invalid_argument("state must be 1-D with length " + std::to_string(size));
        }
        py::array_t<double> successor(static_cast<py::ssize_t>(size));
        config_.sample_step(state.data(), action_, pomdp_native::default_rng().engine(), coins_,
                            successor.mutable_data());
        return successor;
    }

    py::array_t<double> log_probability(const DoubleArray &next_states) {
        const std::size_t n = check_rows(next_states, config_.state_size(), "next_states");
        py::array_t<double> out(static_cast<py::ssize_t>(n));
        double *scores = out.mutable_data();
        config_.coin_eligible(state_.data(), action_, eligible_);
        const double *rows = next_states.data();
        for (std::size_t i = 0; i < n; ++i) {
            scores[i] = config_.transition_log_probability(
                state_.data(), action_, rows + i * config_.state_size(), eligible_, switches_,
                rebuilt_);
        }
        return out;
    }

    py::array_t<double> batch_sample(const DoubleArray &particles) {
        const std::size_t size = config_.state_size();
        const std::size_t n = check_rows(particles, size, "particles");
        py::array_t<double> out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(size)});
        double *dst = out.mutable_data();
        const double *src = particles.data();
        auto &engine = pomdp_native::default_rng().engine();
        for (std::size_t i = 0; i < n; ++i) {
            config_.sample_step(src + i * size, action_, engine, coins_, dst + i * size);
        }
        return out;
    }

    py::array_t<double> state_property() const { return to_numpy(state_); }
    int action_property() const { return action_; }

  private:
    Config config_;
    int action_;
    std::vector<double> state_;
    std::vector<char> coins_;
    std::vector<char> eligible_;
    std::vector<char> switches_;
    std::vector<double> rebuilt_;
};

// Observation kernel. The reading does not depend on the action, so one
// kernel serves every action.
class ChicheckInvadersObservationCpp {
  public:
    ChicheckInvadersObservationCpp(const DoubleArray &next_state, const Config &config)
        : config_(config) {
        read_vector(next_state, config_.state_size(), "next_state", next_state_);
    }

    void set_next_state(const DoubleArray &next_state) {
        read_vector(next_state, config_.state_size(), "next_state", next_state_);
    }

    py::list sample(int n_samples) const {
        if (n_samples < 0) {
            throw std::invalid_argument("n_samples must be non-negative");
        }
        auto &engine = pomdp_native::default_rng().engine();
        py::list out;
        for (int i = 0; i < n_samples; ++i) {
            py::array_t<double> reading(static_cast<py::ssize_t>(config_.observation_size()));
            config_.sample_observation(next_state_.data(), engine, reading.mutable_data());
            out.append(reading);
        }
        return out;
    }

    // Sample one reading of ``next_state`` without storing it: the env's hot path.
    py::array_t<double> sample_from(const DoubleArray &next_state) const {
        if (next_state.ndim() != 1 ||
            static_cast<std::size_t>(next_state.shape(0)) != config_.state_size()) {
            throw std::invalid_argument("next_state must be 1-D with length " +
                                        std::to_string(config_.state_size()));
        }
        py::array_t<double> reading(static_cast<py::ssize_t>(config_.observation_size()));
        config_.sample_observation(next_state.data(), pomdp_native::default_rng().engine(),
                                   reading.mutable_data());
        return reading;
    }

    // Log-likelihood of each candidate reading under the stored successor.
    py::array_t<double> log_probability(const DoubleArray &observations) const {
        const std::size_t n = check_rows(observations, config_.observation_size(), "observations");
        py::array_t<double> out(static_cast<py::ssize_t>(n));
        double *scores = out.mutable_data();
        const double *rows = observations.data();
        for (std::size_t i = 0; i < n; ++i) {
            scores[i] = config_.observation_log_likelihood(
                next_state_.data(), rows + i * config_.observation_size());
        }
        return out;
    }

    // Log-likelihood of one reading under each candidate successor.
    py::array_t<double> batch_log_likelihood(const DoubleArray &next_particles,
                                             const DoubleArray &observation) const {
        const std::size_t size = config_.state_size();
        const std::size_t n = check_rows(next_particles, size, "next_particles");
        if (observation.ndim() != 1 ||
            static_cast<std::size_t>(observation.shape(0)) != config_.observation_size()) {
            throw std::invalid_argument("observation must be 1-D with length " +
                                        std::to_string(config_.observation_size()));
        }
        py::array_t<double> out(static_cast<py::ssize_t>(n));
        double *scores = out.mutable_data();
        const double *rows = next_particles.data();
        for (std::size_t i = 0; i < n; ++i) {
            scores[i] = config_.observation_log_likelihood(rows + i * size, observation.data());
        }
        return out;
    }

    py::array_t<double> next_state_property() const { return to_numpy(next_state_); }

  private:
    Config config_;
    std::vector<double> next_state_;
};

void check_state(const DoubleArray &state, const Config &config) {
    if (state.ndim() != 1 || static_cast<std::size_t>(state.shape(0)) != config.state_size()) {
        throw std::invalid_argument("state must be 1-D with length " +
                                    std::to_string(config.state_size()));
    }
}

double reward(const Config &config, const DoubleArray &state, int action,
              const py::object &next_state) {
    check_state(state, config);
    if (next_state.is_none()) {
        return config.reward(state.data(), action, nullptr);
    }
    auto successor = next_state.cast<DoubleArray>();
    check_state(successor, config);
    return config.reward(state.data(), action, successor.data());
}

py::array_t<double> reward_batch(const Config &config, const DoubleArray &states, int action,
                                 const py::object &next_states) {
    const std::size_t size = config.state_size();
    const std::size_t n = check_rows(states, size, "states");
    py::array_t<double> out(static_cast<py::ssize_t>(n));
    double *rewards = out.mutable_data();
    const double *src = states.data();
    if (next_states.is_none()) {
        for (std::size_t i = 0; i < n; ++i) {
            rewards[i] = config.reward(src + i * size, action, nullptr);
        }
        return out;
    }
    auto successors = next_states.cast<DoubleArray>();
    if (check_rows(successors, size, "next_states") != n) {
        throw std::invalid_argument("states and next_states must have the same row count");
    }
    const double *dst = successors.data();
    for (std::size_t i = 0; i < n; ++i) {
        rewards[i] = config.reward(src + i * size, action, dst + i * size);
    }
    return out;
}

bool is_terminal(const Config &config, const DoubleArray &state) {
    check_state(state, config);
    return config.is_terminal(state.data());
}

// ``Environment.sample_next_step`` in one call: successor, reading of the
// successor, and the reward scored against that successor, drawn in that order.
py::tuple sample_next_step(const Config &config, const DoubleArray &state, int action) {
    check_state(state, config);
    if (action < 0 || action >= kNumActions) {
        throw std::invalid_argument("action must be in [0, 4)");
    }
    auto &engine = pomdp_native::default_rng().engine();
    std::vector<char> coins;
    py::array_t<double> successor(static_cast<py::ssize_t>(config.state_size()));
    config.sample_step(state.data(), action, engine, coins, successor.mutable_data());
    py::array_t<double> reading(static_cast<py::ssize_t>(config.observation_size()));
    config.sample_observation(successor.data(), engine, reading.mutable_data());
    const double reward_value = config.reward(state.data(), action, successor.data());
    return py::make_tuple(successor, reading, reward_value);
}

// Random rollout: step with the pre-drawn actions until the actions run out
// or the state is terminal, then fold the rewards as ``r + gamma * rest``
// from the end, the same order ``python_random_rollout``'s recursion adds
// them in.
double simulate_rollout(const Config &config, const DoubleArray &state,
                        const IntArray &action_indices, double discount_factor) {
    check_state(state, config);
    if (action_indices.ndim() != 1) {
        throw std::invalid_argument("action_indices must be 1-D");
    }
    const std::size_t size = config.state_size();
    const auto steps = static_cast<std::size_t>(action_indices.shape(0));
    const std::int32_t *actions = action_indices.data();
    std::vector<double> current(state.data(), state.data() + size);
    std::vector<double> successor(size);
    std::vector<double> rewards;
    rewards.reserve(steps);
    std::vector<char> coins;
    auto &engine = pomdp_native::default_rng().engine();
    for (std::size_t k = 0; k < steps; ++k) {
        if (config.is_terminal(current.data())) {
            break;
        }
        const int action = static_cast<int>(actions[k]);
        if (action < 0 || action >= kNumActions) {
            throw std::invalid_argument("action index out of range");
        }
        config.sample_step(current.data(), action, engine, coins, successor.data());
        rewards.push_back(config.reward(current.data(), action, successor.data()));
        current.swap(successor);
    }
    double total = 0.0;
    for (auto it = rewards.rbegin(); it != rewards.rend(); ++it) {
        total = *it + discount_factor * total;
    }
    return total;
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native (C++) hot path for the Chicheck Invaders POMDP.";

    m.def("set_seed", &pomdp_native::set_default_seed, py::arg("seed"),
          "Seed the module-level RNG used by every sampler in this module.");

    py::class_<Config>(m, "ChicheckInvadersConfigCpp")
        .def(py::init<int, int, int, int, double, double, double, double, double, double, double,
                      double, double, bool, double, double, double, double, double, int>(),
             py::arg("num_columns"), py::arg("num_rows"), py::arg("num_chickens"),
             py::arg("fire_cooldown"), py::arg("dive_probability"),
             py::arg("camera_detection_probability"), py::arg("radar_detection_probability"),
             py::arg("ship_column_noise_std"), py::arg("camera_offset_noise_std"),
             py::arg("radar_range_noise_std"), py::arg("drop_flag_error_probability"),
             py::arg("camera_slope"), py::arg("radar_radius"), py::arg("full_observation"),
             py::arg("kill_reward"), py::arg("shot_cost"), py::arg("step_cost"),
             py::arg("ship_hit_penalty"), py::arg("clear_reward"), py::arg("max_steps"))
        .def_property_readonly("state_size", &Config::state_size)
        .def_property_readonly("observation_size", &Config::observation_size);

    m.def("reward", &reward, py::arg("config"), py::arg("state"), py::arg("action"),
          py::arg("next_state") = py::none(),
          "Scalar reward; next_state=None scores the no-successor branch.");
    m.def("reward_batch", &reward_batch, py::arg("config"), py::arg("states"), py::arg("action"),
          py::arg("next_states") = py::none(),
          "Reward of each row of states under one action, shape (N,).");
    m.def("is_terminal", &is_terminal, py::arg("config"), py::arg("state"),
          "Flock cleared, ship hit, or step budget spent.");
    m.def("sample_next_step", &sample_next_step, py::arg("config"), py::arg("state"),
          py::arg("action"), "(next_state, observation, reward) for one sampled step.");
    m.def("simulate_rollout", &simulate_rollout, py::arg("config"), py::arg("state"),
          py::arg("action_indices"), py::arg("discount_factor"),
          "Discounted return of a rollout along pre-drawn int32 action indices.");

    py::class_<ChicheckInvadersTransitionCpp>(m, "ChicheckInvadersTransitionCpp")
        .def(py::init<const DoubleArray &, int, const Config &>(), py::arg("state"),
             py::arg("action"), py::arg("config"))
        .def("set_state", &ChicheckInvadersTransitionCpp::set_state, py::arg("state"))
        .def("sample", &ChicheckInvadersTransitionCpp::sample, py::arg("n_samples") = 1)
        .def("sample_from", &ChicheckInvadersTransitionCpp::sample_from, py::arg("state"))
        .def("log_probability", &ChicheckInvadersTransitionCpp::log_probability,
             py::arg("next_states"))
        .def("batch_sample", &ChicheckInvadersTransitionCpp::batch_sample, py::arg("particles"))
        .def_property_readonly("state", &ChicheckInvadersTransitionCpp::state_property)
        .def_property_readonly("action", &ChicheckInvadersTransitionCpp::action_property);

    py::class_<ChicheckInvadersObservationCpp>(m, "ChicheckInvadersObservationCpp")
        .def(py::init<const DoubleArray &, const Config &>(), py::arg("next_state"),
             py::arg("config"))
        .def("set_next_state", &ChicheckInvadersObservationCpp::set_next_state,
             py::arg("next_state"))
        .def("sample", &ChicheckInvadersObservationCpp::sample, py::arg("n_samples") = 1)
        .def("sample_from", &ChicheckInvadersObservationCpp::sample_from, py::arg("next_state"))
        .def("log_probability", &ChicheckInvadersObservationCpp::log_probability,
             py::arg("observations"))
        .def("batch_log_likelihood", &ChicheckInvadersObservationCpp::batch_log_likelihood,
             py::arg("next_particles"), py::arg("observation"))
        .def_property_readonly("next_state", &ChicheckInvadersObservationCpp::next_state_property);
}
