// SPDX-License-Identifier: MIT

// Battleship POMDP native hot path.
//
// The state is a float64 row ``[occupancy | probed]``, each half ``num_cells``
// long. Probing cell ``a`` sets ``probed[a]`` and changes nothing else; the
// observation is HIT (1) when ``occupancy[a]`` is set and MISS (0) otherwise;
// the reward is ``hit_reward`` when the probed cell is occupied and was not
// probed before, and ``-miss_penalty`` otherwise. A state is terminal when
// every occupied cell is probed. Nothing here draws randomness, so the module
// has no RNG and no ``set_seed``: the rollout kernel takes its actions as a
// pre-drawn array from numpy.
//
// Every comparison against a flag uses the same threshold as the Python
// methods in ``battleship_pomdp.py`` (``> 0.5`` for set, ``<= 0.5`` where the
// Python batch path uses it), so a non-binary or NaN entry is classified the
// same way on both sides.
//
// Unlike RockSample, the kernels are module-level functions rather than
// per-action kernel objects. Each one is a pure function of
// ``(state, action, num_cells)``, so a kernel object would hold nothing but
// the state, and the ``set_state`` + ``sample`` pair it needs costs two
// Python-to-C++ calls where one function call does the work.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

// Must match HIT / MISS in battleship_pomdp.py.
constexpr int kHit = 1;
constexpr int kMiss = 0;

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

void check_num_cells(int num_cells) {
    if (num_cells < 1) {
        throw std::invalid_argument("num_cells must be at least 1");
    }
}

// Reject an action outside the board. Python indexes ``state[num_cells + a]``,
// which raises IndexError past the end; pybind11 maps std::out_of_range to
// IndexError, so the error type matches.
void check_action(int action, int num_cells) {
    if (action < 0 || action >= num_cells) {
        throw std::out_of_range("action " + std::to_string(action) + " is not a cell in [0, " +
                                std::to_string(num_cells) + ")");
    }
}

// A 1-D state that holds at least the probe flag of cell ``action``.
const double *state_row(const DoubleArray &state, int action, int num_cells, const char *label,
                        std::size_t *length) {
    if (state.ndim() != 1) {
        throw std::invalid_argument(std::string(label) + " must be a 1-D array");
    }
    const auto n = static_cast<std::size_t>(state.shape(0));
    if (n < static_cast<std::size_t>(num_cells + action + 1)) {
        throw std::out_of_range(std::string(label) + " of length " + std::to_string(n) +
                                " has no probe flag for cell " + std::to_string(action));
    }
    *length = n;
    return state.data();
}

// A 1-D state of exactly ``2 * num_cells`` entries.
const double *full_state_row(const DoubleArray &state, int num_cells, const char *label) {
    if (state.ndim() != 1 || state.shape(0) != static_cast<py::ssize_t>(2 * num_cells)) {
        throw std::invalid_argument(std::string(label) + " must be a 1-D array of length " +
                                    std::to_string(2 * num_cells));
    }
    return state.data();
}

inline double reward_of(const double *row, int action, int num_cells, double hit_reward,
                        double miss_penalty) {
    const bool holds_ship = row[action] > 0.5;
    const bool already_probed = row[num_cells + action] > 0.5;
    return (holds_ship && !already_probed) ? hit_reward : -miss_penalty;
}

inline bool is_terminal_row(const double *row, int num_cells) {
    for (int c = 0; c < num_cells; ++c) {
        if (row[c] > 0.5 && row[num_cells + c] <= 0.5) {
            return false;
        }
    }
    return true;
}

py::array_t<double> copy_row(const double *src, std::size_t n) {
    py::array_t<double> out(static_cast<py::ssize_t>(n));
    std::memcpy(out.mutable_data(), src, n * sizeof(double));
    return out;
}

py::array_t<double> sample_next_state(const DoubleArray &state, int action, int num_cells) {
    check_num_cells(num_cells);
    check_action(action, num_cells);
    std::size_t n = 0;
    const double *row = state_row(state, action, num_cells, "state", &n);
    auto out = copy_row(row, n);
    out.mutable_data()[num_cells + action] = 1.0;
    return out;
}

py::array_t<double> sample_next_state_batch(const DoubleArray &states, int action,
                                            int num_cells) {
    check_num_cells(num_cells);
    check_action(action, num_cells);
    if (states.ndim() != 2) {
        throw std::invalid_argument("states must be a 2-D array");
    }
    const auto n_rows = static_cast<std::size_t>(states.shape(0));
    const auto width = static_cast<std::size_t>(states.shape(1));
    if (width < static_cast<std::size_t>(num_cells + action + 1)) {
        throw std::out_of_range("states rows have no probe flag for cell " +
                                std::to_string(action));
    }
    py::array_t<double> out(
        {static_cast<py::ssize_t>(n_rows), static_cast<py::ssize_t>(width)});
    double *dst = out.mutable_data();
    if (n_rows > 0) {
        std::memcpy(dst, states.data(), n_rows * width * sizeof(double));
    }
    const std::size_t slot = static_cast<std::size_t>(num_cells + action);
    for (std::size_t i = 0; i < n_rows; ++i) {
        dst[i * width + slot] = 1.0;
    }
    return out;
}

// 0.0 for each candidate within 0.5 of the realised successor on every entry,
// -inf otherwise; the same test as ``np.all(np.abs(c - e) < 0.5, axis=1)``.
py::array_t<double> transition_log_probability(const DoubleArray &state, int action,
                                               int num_cells, const DoubleArray &next_states) {
    check_num_cells(num_cells);
    check_action(action, num_cells);
    std::size_t n = 0;
    const double *row = state_row(state, action, num_cells, "state", &n);
    std::vector<double> expected(row, row + n);
    expected[static_cast<std::size_t>(num_cells + action)] = 1.0;

    std::size_t n_rows = 0;
    if (next_states.ndim() == 1 && static_cast<std::size_t>(next_states.shape(0)) == n) {
        n_rows = 1;
    } else if (next_states.ndim() == 2 && static_cast<std::size_t>(next_states.shape(1)) == n) {
        n_rows = static_cast<std::size_t>(next_states.shape(0));
    } else {
        throw std::invalid_argument("next_states must be a state or an (N, " +
                                    std::to_string(n) + ") array of states");
    }
    const double *cand = next_states.data();
    py::array_t<double> out(static_cast<py::ssize_t>(n_rows));
    double *buf = out.mutable_data();
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n_rows; ++i) {
        const double *c = cand + i * n;
        bool match = true;
        for (std::size_t d = 0; d < n; ++d) {
            if (!(std::fabs(c[d] - expected[d]) < 0.5)) {
                match = false;
                break;
            }
        }
        buf[i] = match ? 0.0 : neg_inf;
    }
    return out;
}

int sample_observation(const DoubleArray &next_state, int action, int num_cells) {
    check_num_cells(num_cells);
    check_action(action, num_cells);
    if (next_state.ndim() != 1 || next_state.shape(0) <= static_cast<py::ssize_t>(action)) {
        throw std::out_of_range("next_state has no occupancy entry for cell " +
                                std::to_string(action));
    }
    return next_state.data()[action] > 0.5 ? kHit : kMiss;
}

// ``observations`` is read flattened, as Python's ``np.asarray(...).ravel()``.
py::array_t<double> observation_log_probability(const DoubleArray &next_state, int action,
                                                int num_cells,
                                                const DoubleArray &observations) {
    const double truth = static_cast<double>(sample_observation(next_state, action, num_cells));
    const auto k = static_cast<std::size_t>(observations.size());
    const double *obs = observations.data();
    py::array_t<double> out(static_cast<py::ssize_t>(k));
    double *buf = out.mutable_data();
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < k; ++i) {
        buf[i] = (obs[i] == truth) ? 0.0 : neg_inf;
    }
    return out;
}

double reward(const DoubleArray &state, int action, int num_cells, double hit_reward,
              double miss_penalty) {
    check_num_cells(num_cells);
    check_action(action, num_cells);
    std::size_t n = 0;
    const double *row = state_row(state, action, num_cells, "state", &n);
    return reward_of(row, action, num_cells, hit_reward, miss_penalty);
}

// Batch reward. Mirrors the Python batch expression
// ``(s[:, a] > 0.5) & (s[:, num_cells + a] <= 0.5)``, which differs from the
// scalar ``not (s[num_cells + a] > 0.5)`` only for a NaN probe flag.
py::array_t<double> reward_batch(const DoubleArray &states, int action, int num_cells,
                                 double hit_reward, double miss_penalty) {
    check_num_cells(num_cells);
    check_action(action, num_cells);
    if (states.ndim() != 2) {
        throw std::invalid_argument("states must be a 2-D array");
    }
    const auto n_rows = static_cast<std::size_t>(states.shape(0));
    const auto width = static_cast<std::size_t>(states.shape(1));
    if (width < static_cast<std::size_t>(num_cells + action + 1)) {
        throw std::out_of_range("states rows have no probe flag for cell " +
                                std::to_string(action));
    }
    const double *data = states.data();
    py::array_t<double> out(static_cast<py::ssize_t>(n_rows));
    double *buf = out.mutable_data();
    const std::size_t probe_slot = static_cast<std::size_t>(num_cells + action);
    for (std::size_t i = 0; i < n_rows; ++i) {
        const double *row = data + i * width;
        const bool is_new_hit = row[action] > 0.5 && row[probe_slot] <= 0.5;
        buf[i] = is_new_hit ? hit_reward : -miss_penalty;
    }
    return out;
}

bool is_terminal(const DoubleArray &state, int num_cells) {
    check_num_cells(num_cells);
    return is_terminal_row(full_state_row(state, num_cells, "state"), num_cells);
}

// One call for ``sample_next_state`` + ``sample_observation`` + ``reward``.
py::tuple sample_next_step(const DoubleArray &state, int action, int num_cells,
                           double hit_reward, double miss_penalty) {
    check_num_cells(num_cells);
    check_action(action, num_cells);
    std::size_t n = 0;
    const double *row = state_row(state, action, num_cells, "state", &n);
    const double r = reward_of(row, action, num_cells, hit_reward, miss_penalty);
    const int observation = row[action] > 0.5 ? kHit : kMiss;
    auto next_state = copy_row(row, n);
    next_state.mutable_data()[num_cells + action] = 1.0;
    return py::make_tuple(next_state, py::int_(observation), py::float_(r));
}

// Random rollout from ``initial_state`` with pre-drawn actions.
//
// Matches ``python_random_rollout``: stop at ``max_depth`` or at a terminal
// state; otherwise take action ``action_indices[depth - start_depth]``. The
// discounted sum is accumulated from the last step backwards,
// ``v = r_k + gamma * v``, which is the order the recursive Python rollout
// adds in, so the two return bit-identical values for the same actions.
double simulate_rollout_discrete(const DoubleArray &initial_state,
                                 const py::array_t<std::int32_t, py::array::c_style |
                                                                     py::array::forcecast>
                                     &action_indices,
                                 int max_depth, int start_depth, double discount_factor,
                                 int num_cells, double hit_reward, double miss_penalty) {
    check_num_cells(num_cells);
    const double *init = full_state_row(initial_state, num_cells, "initial_state");
    if (action_indices.ndim() != 1) {
        throw std::invalid_argument("action_indices must be 1-D");
    }
    const auto n_indices = static_cast<std::size_t>(action_indices.shape(0));
    const std::int32_t *actions = action_indices.data();

    const std::size_t width = static_cast<std::size_t>(2 * num_cells);
    std::vector<double> cur(init, init + width);
    std::vector<double> rewards;
    if (max_depth > start_depth) {
        rewards.reserve(static_cast<std::size_t>(max_depth - start_depth));
    }
    for (int depth = start_depth; depth < max_depth; ++depth) {
        if (is_terminal_row(cur.data(), num_cells)) {
            break;
        }
        const std::size_t slot = static_cast<std::size_t>(depth - start_depth);
        if (slot >= n_indices) {
            throw std::invalid_argument("action_indices is shorter than max_depth - start_depth");
        }
        const int a = static_cast<int>(actions[slot]);
        check_action(a, num_cells);
        rewards.push_back(reward_of(cur.data(), a, num_cells, hit_reward, miss_penalty));
        cur[static_cast<std::size_t>(num_cells + a)] = 1.0;
    }
    // The product goes through a volatile so the compiler cannot fuse
    // ``r + gamma * v`` into one fused multiply-add: clang does that at -O3 on
    // arm64, and the single rounding of an FMA differs from Python's two
    // roundings in the last bit.
    double total = 0.0;
    for (auto it = rewards.rbegin(); it != rewards.rend(); ++it) {
        volatile double discounted = discount_factor * total;
        total = *it + discounted;
    }
    return total;
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native (C++) hot path for the Battleship POMDP.";

    m.def("sample_next_state", &sample_next_state, py::arg("state"), py::arg("action"),
          py::arg("num_cells"), "Copy of ``state`` with the probe flag of ``action`` set.");
    m.def("sample_next_state_batch", &sample_next_state_batch, py::arg("states"),
          py::arg("action"), py::arg("num_cells"),
          "Apply one probe to every row of an (N, D) array.");
    m.def("transition_log_probability", &transition_log_probability, py::arg("state"),
          py::arg("action"), py::arg("num_cells"), py::arg("next_states"),
          "0.0 for the realised successor, -inf for any other candidate.");
    m.def("sample_observation", &sample_observation, py::arg("next_state"), py::arg("action"),
          py::arg("num_cells"), "HIT (1) if the probed cell is occupied, else MISS (0).");
    m.def("observation_log_probability", &observation_log_probability,
          py::arg("next_state"), py::arg("action"), py::arg("num_cells"),
          py::arg("observations"),
          "0.0 for the observation the sensor emits, -inf for any other.");
    m.def("reward", &reward, py::arg("state"), py::arg("action"), py::arg("num_cells"),
          py::arg("hit_reward"), py::arg("miss_penalty"), "Reward of one probe.");
    m.def("reward_batch", &reward_batch, py::arg("states"), py::arg("action"),
          py::arg("num_cells"), py::arg("hit_reward"), py::arg("miss_penalty"),
          "Reward of one probe for every row of an (N, D) array.");
    m.def("is_terminal", &is_terminal, py::arg("state"), py::arg("num_cells"),
          "True when every occupied cell has been probed.");
    m.def("sample_next_step", &sample_next_step, py::arg("state"), py::arg("action"),
          py::arg("num_cells"), py::arg("hit_reward"), py::arg("miss_penalty"),
          "(next_state, observation, reward) in one call.");
    m.def("simulate_rollout_discrete", &simulate_rollout_discrete, py::arg("initial_state"),
          py::arg("action_indices"), py::arg("max_depth"), py::arg("start_depth"),
          py::arg("discount_factor"), py::arg("num_cells"), py::arg("hit_reward"),
          py::arg("miss_penalty"),
          "Discounted return of a rollout with pre-drawn actions; stops at a terminal "
          "state or at max_depth.");
}
