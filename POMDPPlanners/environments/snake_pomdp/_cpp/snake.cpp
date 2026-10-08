// SPDX-License-Identifier: MIT

// Snake POMDP native kernels.
//
// Mirrors ``SnakePOMDP`` in ``snake_pomdp.py`` rule for rule. The state is a
// float64 vector
//   [status, length, steps_since_food, food_row, food_col, r0, c0, r1, c1, ...]
// of width ``5 + 2 * target_length``, with -1 in unused body slots and in the
// food slots of a won episode. The body update is deterministic; the only
// random part of a transition is where the food respawns after an eat that
// does not end the episode, drawn uniformly over the cells the new body leaves
// free. The observation is a tuple of ints
//   (1, scent, seen_row, seen_col, r0, c0, r1, c1, ...)
// for a live state and (0,) for every terminal state.
//
// Every Python ``int(round(x))`` read of a state slot is ``std::nearbyint``
// here, which rounds half to even under the default floating-point mode, as
// Python's ``round`` does. ``std::round`` rounds half away from zero and would
// disagree on a hand-built state holding 0.5.
//
// The state width depends on ``target_length``, so these classes do not
// inherit from ``TransitionModelCpp<Dim>`` / ``ObservationModelCpp<Dim>``.
// They compose the module-local RNG (pomdp_native/rng.hpp) directly, so
// ``set_seed`` on this module is the stream every draw comes from.
//
// Log-probabilities of impossible events are ``-inf``, as in the Python code.
// The other ports floor them at about -690.8; doing that here would change
// what ``observation_log_probability`` returns.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "pomdp_native/rng.hpp"

namespace py = pybind11;

namespace {

constexpr int kStatusIndex = 0;
constexpr int kLengthIndex = 1;
constexpr int kStepsIndex = 2;
constexpr int kFoodRowIndex = 3;
constexpr int kFoodColIndex = 4;
constexpr int kBodyOffset = 5;
constexpr double kEmpty = -1.0;

constexpr int kRunning = 0;
constexpr int kWall = 1;
constexpr int kSelf = 2;
constexpr int kStarvation = 3;
constexpr int kWin = 4;

constexpr int kTurnLeft = 0;
constexpr int kTurnRight = 2;

constexpr int kObservationLive = 1;
constexpr int kObservationTerminal = 0;
constexpr int kNumQuadrants = 4;

// Clockwise headings as (row, col) steps; matches DIRECTIONS in Python.
constexpr int kDirRow[4] = {-1, 0, 1, 0};
constexpr int kDirCol[4] = {0, 1, 0, -1};

const double kNegInf = -std::numeric_limits<double>::infinity();

using CArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

inline int py_round(double value) { return static_cast<int>(std::nearbyint(value)); }

inline bool state_is_terminal(const double *state) {
    return py_round(state[kStatusIndex]) != kRunning;
}

// Geometry and limits shared by every kernel.
struct SnakeParams {
    int grid_size = 0;
    int target_length = 0;
    int starvation_limit = 0;
    int window_radius = 0;
    double detection_probability = 0.0;
    double scent_accuracy = 0.0;

    int state_size() const { return kBodyOffset + 2 * target_length; }
    bool in_grid(int row, int col) const {
        return row >= 0 && row < grid_size && col >= 0 && col < grid_size;
    }
};

// The deterministic half of one transition, as SnakePOMDP.transition_outcome.
struct Outcome {
    std::vector<int> rows;  // new body, head first
    std::vector<int> cols;
    bool eat = false;
    int counter = 0;
    int termination = kRunning;
    bool food_present = false;
    int food_row = -1;
    int food_col = -1;
};

void read_body(const double *state, std::vector<int> &rows, std::vector<int> &cols) {
    const int length = py_round(state[kLengthIndex]);
    rows.resize(static_cast<std::size_t>(std::max(length, 0)));
    cols.resize(rows.size());
    for (int i = 0; i < length; ++i) {
        rows[i] = py_round(state[kBodyOffset + 2 * i]);
        cols[i] = py_round(state[kBodyOffset + 2 * i + 1]);
    }
}

void resolve(const SnakeParams &params, const double *state, int action, Outcome &out) {
    thread_local std::vector<int> body_rows;
    thread_local std::vector<int> body_cols;
    read_body(state, body_rows, body_cols);
    const std::size_t length = body_rows.size();
    if (length < 2) {
        throw py::value_error("a snake shorter than two cells has no heading");
    }
    const int head_row = body_rows[0];
    const int head_col = body_cols[0];
    const int d_row = head_row - body_rows[1];
    const int d_col = head_col - body_cols[1];
    int heading = -1;
    for (int i = 0; i < 4; ++i) {
        if (kDirRow[i] == d_row && kDirCol[i] == d_col) {
            heading = i;
        }
    }
    if (heading < 0) {
        throw py::value_error("the heading is not one of the four axis steps");
    }
    if (action < 0 || action > 2) {
        throw py::value_error("action must be one of 0, 1, 2, got " + std::to_string(action));
    }
    if (action == kTurnLeft) {
        heading = (heading + 3) % 4;
    } else if (action == kTurnRight) {
        heading = (heading + 1) % 4;
    }
    const int target_row = head_row + kDirRow[heading];
    const int target_col = head_col + kDirCol[heading];

    out.food_row = py_round(state[kFoodRowIndex]);
    out.food_col = py_round(state[kFoodColIndex]);
    out.food_present = !(out.food_row < 0 && out.food_col < 0);
    out.eat = out.food_present && target_row == out.food_row && target_col == out.food_col;

    // Eating keeps the tail; otherwise the tail cell is released.
    const std::size_t kept = out.eat ? length : length - 1;
    out.rows.resize(kept + 1);
    out.cols.resize(kept + 1);
    out.rows[0] = target_row;
    out.cols[0] = target_col;
    for (std::size_t i = 0; i < kept; ++i) {
        out.rows[i + 1] = body_rows[i];
        out.cols[i + 1] = body_cols[i];
    }
    out.counter = out.eat ? 0 : py_round(state[kStepsIndex]) + 1;

    // Wall, then self, then the win, then starvation.
    if (!params.in_grid(target_row, target_col)) {
        out.termination = kWall;
        return;
    }
    for (std::size_t i = 1; i < out.rows.size(); ++i) {
        if (out.rows[i] == target_row && out.cols[i] == target_col) {
            out.termination = kSelf;
            return;
        }
    }
    if (static_cast<int>(out.rows.size()) >= params.target_length) {
        out.termination = kWin;
        return;
    }
    if (out.counter >= params.starvation_limit) {
        out.termination = kStarvation;
        return;
    }
    out.termination = kRunning;
}

inline bool respawns(const Outcome &outcome) {
    return outcome.eat && outcome.termination == kRunning;
}

double outcome_reward(const Outcome &outcome) {
    if (outcome.eat) {
        return 1.0;
    }
    if (outcome.termination == kWall || outcome.termination == kSelf ||
        outcome.termination == kStarvation) {
        return -1.0;
    }
    return 0.0;
}

double reward_of(const SnakeParams &params, const double *state, int action) {
    if (state_is_terminal(state)) {
        return 0.0;
    }
    thread_local Outcome outcome;
    resolve(params, state, action, outcome);
    return outcome_reward(outcome);
}

// Flat indices of the in-grid cells ``outcome``'s body leaves free, ascending.
void free_cells(const SnakeParams &params, const Outcome &outcome, std::vector<int> &cells) {
    const int num_cells = params.grid_size * params.grid_size;
    thread_local std::vector<char> occupied;
    occupied.assign(static_cast<std::size_t>(num_cells), 0);
    for (std::size_t i = 0; i < outcome.rows.size(); ++i) {
        if (params.in_grid(outcome.rows[i], outcome.cols[i])) {
            occupied[outcome.rows[i] * params.grid_size + outcome.cols[i]] = 1;
        }
    }
    cells.clear();
    for (int cell = 0; cell < num_cells; ++cell) {
        if (!occupied[cell]) {
            cells.push_back(cell);
        }
    }
}

// Write the successor (SnakePOMDP._successor) into ``out``. ``food_cell`` is
// the flat index of the respawned food, or -1 when nothing respawned.
void write_successor(const SnakeParams &params, const Outcome &outcome, int food_cell,
                     double *out) {
    const int body_length = static_cast<int>(outcome.rows.size());
    if (body_length > params.target_length) {
        throw py::value_error("body of " + std::to_string(body_length) +
                              " cells exceeds target_length " +
                              std::to_string(params.target_length));
    }
    const int width = params.state_size();
    for (int i = 0; i < width; ++i) {
        out[i] = kEmpty;
    }
    out[kStatusIndex] = static_cast<double>(outcome.termination);
    out[kLengthIndex] = static_cast<double>(body_length);
    out[kStepsIndex] = static_cast<double>(outcome.counter);

    bool has_food = false;
    int food_row = -1;
    int food_col = -1;
    if (outcome.termination == kWall || outcome.termination == kSelf) {
        has_food = outcome.food_present;
        food_row = outcome.food_row;
        food_col = outcome.food_col;
    } else if (outcome.termination == kWin) {
        has_food = false;
    } else if (outcome.eat) {
        if (food_cell >= 0) {
            has_food = true;
            food_row = food_cell / params.grid_size;
            food_col = food_cell % params.grid_size;
        }
    } else {
        has_food = outcome.food_present;
        food_row = outcome.food_row;
        food_col = outcome.food_col;
    }
    if (has_food) {
        out[kFoodRowIndex] = static_cast<double>(food_row);
        out[kFoodColIndex] = static_cast<double>(food_col);
    }
    for (int i = 0; i < body_length; ++i) {
        out[kBodyOffset + 2 * i] = static_cast<double>(outcome.rows[i]);
        out[kBodyOffset + 2 * i + 1] = static_cast<double>(outcome.cols[i]);
    }
}

// One transition draw from ``state`` into ``out`` (width state_size()).
void sample_transition(const SnakeParams &params, const double *state, int action, double *out,
                       std::mt19937_64 &engine) {
    const int width = params.state_size();
    if (state_is_terminal(state)) {
        for (int i = 0; i < width; ++i) {
            out[i] = state[i];
        }
        return;
    }
    thread_local Outcome outcome;
    resolve(params, state, action, outcome);
    int food_cell = -1;
    if (respawns(outcome)) {
        thread_local std::vector<int> cells;
        free_cells(params, outcome, cells);
        if (cells.empty()) {
            throw py::value_error("the body fills the grid, so the food has nowhere to respawn");
        }
        std::uniform_int_distribution<int> pick(0, static_cast<int>(cells.size()) - 1);
        food_cell = cells[static_cast<std::size_t>(pick(engine))];
    }
    write_successor(params, outcome, food_cell, out);
}

// Same as sample_transition, but also returns the reward for (state, action).
double sample_transition_with_reward(const SnakeParams &params, const double *state, int action,
                                     double *out, std::mt19937_64 &engine) {
    const int width = params.state_size();
    if (state_is_terminal(state)) {
        for (int i = 0; i < width; ++i) {
            out[i] = state[i];
        }
        return 0.0;
    }
    thread_local Outcome outcome;
    resolve(params, state, action, outcome);
    int food_cell = -1;
    if (respawns(outcome)) {
        thread_local std::vector<int> cells;
        free_cells(params, outcome, cells);
        if (cells.empty()) {
            throw py::value_error("the body fills the grid, so the food has nowhere to respawn");
        }
        std::uniform_int_distribution<int> pick(0, static_cast<int>(cells.size()) - 1);
        food_cell = cells[static_cast<std::size_t>(pick(engine))];
    }
    write_successor(params, outcome, food_cell, out);
    return outcome_reward(outcome);
}

// What the sensor needs from a live next state.
struct SensorView {
    std::vector<int> rows;
    std::vector<int> cols;
    int food_row = 0;
    int food_col = 0;
    bool inside = false;
    double scent_probability[kNumQuadrants] = {0.0, 0.0, 0.0, 0.0};
};

void read_sensor(const SnakeParams &params, const double *next_state, SensorView &view) {
    read_body(next_state, view.rows, view.cols);
    if (view.rows.empty()) {
        throw py::index_error("a live Snake state needs a head to observe from");
    }
    const int food_row = py_round(next_state[kFoodRowIndex]);
    const int food_col = py_round(next_state[kFoodColIndex]);
    if (food_row < 0 && food_col < 0) {
        throw py::value_error("a non-terminal Snake state always carries a food cell");
    }
    view.food_row = food_row;
    view.food_col = food_col;
    const int d_row = food_row - view.rows[0];
    const int d_col = food_col - view.cols[0];
    view.inside = params.in_grid(food_row, food_col) && std::abs(d_row) <= params.window_radius &&
                  std::abs(d_col) <= params.window_radius;

    // quadrants_for_offset: NE=0, NW=1, SE=2, SW=3.
    if (d_row == 0 && d_col == 0) {
        throw py::value_error("the food cell can never coincide with the head cell");
    }
    bool compatible[kNumQuadrants];
    int n_compatible = 0;
    for (int q = 0; q < kNumQuadrants; ++q) {
        const bool north = (q == 0 || q == 1);
        const bool east = (q == 0 || q == 2);
        const bool vertical_ok = d_row < 0 ? north : (d_row > 0 ? !north : true);
        const bool horizontal_ok = d_col > 0 ? east : (d_col < 0 ? !east : true);
        compatible[q] = vertical_ok && horizontal_ok;
        n_compatible += compatible[q] ? 1 : 0;
    }
    const double wrong = (1.0 - params.scent_accuracy) / (kNumQuadrants - n_compatible);
    const double right = params.scent_accuracy / n_compatible;
    for (int q = 0; q < kNumQuadrants; ++q) {
        view.scent_probability[q] = compatible[q] ? right : wrong;
    }
}

// numpy's ``choice(p=...)``: normalised cumulative sum, then the first bin
// whose upper edge exceeds the uniform draw.
int draw_quadrant(const double *probabilities, std::mt19937_64 &engine) {
    double cdf[kNumQuadrants];
    double running = 0.0;
    for (int q = 0; q < kNumQuadrants; ++q) {
        running += probabilities[q];
        cdf[q] = running;
    }
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double u = uniform(engine) * cdf[kNumQuadrants - 1];
    for (int q = 0; q < kNumQuadrants; ++q) {
        if (cdf[q] > u) {
            return q;
        }
    }
    return kNumQuadrants - 1;
}

py::tuple terminal_observation() {
    py::tuple out(1);
    out[0] = py::int_(kObservationTerminal);
    return out;
}

py::tuple draw_observation(const SnakeParams &params, const SensorView &view,
                           std::mt19937_64 &engine) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    // The detection draw happens only when the food is inside the window, as
    // the Python ``inside and random() < p`` short-circuit does.
    const bool detected = view.inside && uniform(engine) < params.detection_probability;
    const int scent = draw_quadrant(view.scent_probability, engine);
    const std::size_t length = view.rows.size();
    py::tuple out(4 + 2 * length);
    out[0] = py::int_(kObservationLive);
    out[1] = py::int_(scent);
    out[2] = py::int_(detected ? view.food_row : -1);
    out[3] = py::int_(detected ? view.food_col : -1);
    for (std::size_t i = 0; i < length; ++i) {
        out[4 + 2 * i] = py::int_(view.rows[i]);
        out[5 + 2 * i] = py::int_(view.cols[i]);
    }
    return out;
}

py::tuple sample_observation_one(const SnakeParams &params, const double *next_state,
                                 std::mt19937_64 &engine) {
    if (state_is_terminal(next_state)) {
        return terminal_observation();
    }
    thread_local SensorView view;
    read_sensor(params, next_state, view);
    return draw_observation(params, view, engine);
}

// int(value) for every element of one candidate reading.
void read_reading(const py::handle &reading, std::vector<long long> &flat) {
    flat.clear();
    for (const py::handle item : py::reinterpret_borrow<py::iterable>(reading)) {
        flat.push_back(py::int_(py::reinterpret_borrow<py::object>(item)).cast<long long>());
    }
}

// SnakePOMDP.observation_log_probability for one candidate under a live state.
double live_log_likelihood(const SnakeParams &params, const SensorView &view,
                           const std::vector<long long> &flat) {
    if (flat.empty() || flat[0] != kObservationLive) {
        return kNegInf;
    }
    // decode_observation reads flat[1], flat[2], flat[3] only when something
    // was seen, and the body pairs from index 4; a reading too short for what
    // it reads raises IndexError there, so it does here.
    if (flat.size() < 3 || (flat[2] >= 0 && flat.size() < 4) ||
        (flat.size() > 4 && (flat.size() - 4) % 2 != 0)) {
        throw py::index_error("tuple index out of range");
    }
    const long long scent = flat[1];
    const bool seen = flat[2] >= 0;
    const std::size_t observed_length = flat.size() > 4 ? (flat.size() - 4) / 2 : 0;
    if (observed_length != view.rows.size()) {
        return kNegInf;
    }
    for (std::size_t i = 0; i < observed_length; ++i) {
        if (flat[4 + 2 * i] != view.rows[i] || flat[5 + 2 * i] != view.cols[i]) {
            return kNegInf;
        }
    }
    if (scent < 0 || scent >= kNumQuadrants) {
        return kNegInf;
    }
    double sighting = 0.0;
    if (!seen) {
        sighting = view.inside ? std::log1p(-params.detection_probability) : 0.0;
    } else if (view.inside && flat[2] == view.food_row && flat[3] == view.food_col) {
        sighting = std::log(params.detection_probability);
    } else {
        return kNegInf;  // no false positives
    }
    return sighting + std::log(view.scent_probability[scent]);
}

SnakeParams make_transition_params(int grid_size, int target_length, int starvation_limit) {
    SnakeParams params;
    params.grid_size = grid_size;
    params.target_length = target_length;
    params.starvation_limit = starvation_limit;
    return params;
}

SnakeParams make_full_params(int grid_size, int target_length, int starvation_limit,
                             int window_radius, double detection_probability,
                             double scent_accuracy) {
    SnakeParams params = make_transition_params(grid_size, target_length, starvation_limit);
    params.window_radius = window_radius;
    params.detection_probability = detection_probability;
    params.scent_accuracy = scent_accuracy;
    return params;
}

std::vector<double> state_vector(const CArray &state, int width, const char *label) {
    if (state.ndim() != 1 || state.shape(0) != width) {
        throw py::value_error(std::string(label) + " must be a 1-D array of width " +
                              std::to_string(width));
    }
    return std::vector<double>(state.data(), state.data() + width);
}

const double *rows_of(const CArray &rows, int width, const char *label, std::size_t &n_rows) {
    if (rows.ndim() != 2 || rows.shape(1) != width) {
        throw py::value_error(std::string(label) + " must have shape (N, " +
                              std::to_string(width) + ")");
    }
    n_rows = static_cast<std::size_t>(rows.shape(0));
    return rows.data();
}

class SnakeTransitionCpp {
  public:
    SnakeTransitionCpp(const CArray &state, int action, int grid_size, int target_length,
                       int starvation_limit)
        : params_(make_transition_params(grid_size, target_length, starvation_limit)),
          action_(action) {
        state_ = state_vector(state, params_.state_size(), "state");
    }

    void set_state(const CArray &state) {
        state_ = state_vector(state, params_.state_size(), "state");
    }

    py::list sample(int n_samples) const {
        if (n_samples < 0) {
            throw py::value_error("n_samples must be non-negative");
        }
        const int width = params_.state_size();
        auto &engine = pomdp_native::default_rng().engine();
        py::list out;
        for (int i = 0; i < n_samples; ++i) {
            py::array_t<double> row(width);
            sample_transition(params_, state_.data(), action_, row.mutable_data(), engine);
            out.append(row);
        }
        return out;
    }

    // log P(candidate | state, action) per row, as transition_log_probability.
    py::array_t<double> log_probability(const CArray &candidates) const {
        const int width = params_.state_size();
        std::size_t n_rows = 0;
        const double *rows = rows_of(candidates, width, "next_states", n_rows);
        py::array_t<double> out(static_cast<py::ssize_t>(n_rows));
        double *scores = out.mutable_data();

        if (state_is_terminal(state_.data())) {
            for (std::size_t i = 0; i < n_rows; ++i) {
                scores[i] = rows_equal(rows + i * width, state_.data(), 0, width) ? 0.0 : kNegInf;
            }
            return out;
        }
        Outcome outcome;
        resolve(params_, state_.data(), action_, outcome);
        if (!respawns(outcome)) {
            std::vector<double> expected(static_cast<std::size_t>(width));
            write_successor(params_, outcome, -1, expected.data());
            for (std::size_t i = 0; i < n_rows; ++i) {
                scores[i] = rows_equal(rows + i * width, expected.data(), 0, width) ? 0.0 : kNegInf;
            }
            return out;
        }
        // Every respawn cell gives the same successor except for the food
        // slots, so a candidate scores -log(#free) when it matches the rest
        // exactly and its food slots name a free cell.
        std::vector<int> cells;
        free_cells(params_, outcome, cells);
        std::vector<char> is_free(static_cast<std::size_t>(params_.grid_size * params_.grid_size), 0);
        for (int cell : cells) {
            is_free[cell] = 1;
        }
        std::vector<double> expected(static_cast<std::size_t>(width));
        write_successor(params_, outcome, cells.empty() ? -1 : cells[0], expected.data());
        const double log_probability = -std::log(static_cast<double>(cells.size()));
        for (std::size_t i = 0; i < n_rows; ++i) {
            const double *row = rows + i * width;
            scores[i] = kNegInf;
            if (!rows_equal(row, expected.data(), 0, kFoodRowIndex) ||
                !rows_equal(row, expected.data(), kBodyOffset, width)) {
                continue;
            }
            const double food_row = row[kFoodRowIndex];
            const double food_col = row[kFoodColIndex];
            if (food_row != std::floor(food_row) || food_col != std::floor(food_col)) {
                continue;
            }
            if (!(food_row >= 0.0 && food_row < params_.grid_size && food_col >= 0.0 &&
                  food_col < params_.grid_size)) {
                continue;
            }
            const int cell =
                static_cast<int>(food_row) * params_.grid_size + static_cast<int>(food_col);
            if (is_free[cell]) {
                scores[i] = log_probability;
            }
        }
        return out;
    }

    py::array_t<double> probability(const CArray &candidates) const {
        py::array_t<double> logs = log_probability(candidates);
        double *values = logs.mutable_data();
        for (py::ssize_t i = 0; i < logs.shape(0); ++i) {
            values[i] = std::exp(values[i]);
        }
        return logs;
    }

    // One successor per input row under this kernel's action.
    py::array_t<double> batch_sample(const CArray &particles) const {
        const int width = params_.state_size();
        std::size_t n_rows = 0;
        const double *rows = rows_of(particles, width, "particles", n_rows);
        py::array_t<double> out({static_cast<py::ssize_t>(n_rows), static_cast<py::ssize_t>(width)});
        double *buffer = out.mutable_data();
        auto &engine = pomdp_native::default_rng().engine();
        for (std::size_t i = 0; i < n_rows; ++i) {
            sample_transition(params_, rows + i * width, action_, buffer + i * width, engine);
        }
        return out;
    }

    py::array_t<double> state_property() const {
        py::array_t<double> out(static_cast<py::ssize_t>(state_.size()));
        std::copy(state_.begin(), state_.end(), out.mutable_data());
        return out;
    }
    int action_property() const { return action_; }

  private:
    static bool rows_equal(const double *a, const double *b, int begin, int end) {
        for (int i = begin; i < end; ++i) {
            if (!(a[i] == b[i])) {
                return false;
            }
        }
        return true;
    }

    SnakeParams params_;
    int action_;
    std::vector<double> state_;
};

class SnakeObservationCpp {
  public:
    SnakeObservationCpp(const CArray &next_state, int action, int grid_size, int target_length,
                        int window_radius, double detection_probability, double scent_accuracy)
        : params_(make_full_params(grid_size, target_length, 1, window_radius,
                                   detection_probability, scent_accuracy)),
          action_(action) {
        next_state_ = state_vector(next_state, params_.state_size(), "next_state");
    }

    void set_next_state(const CArray &next_state) {
        next_state_ = state_vector(next_state, params_.state_size(), "next_state");
    }

    py::list sample(int n_samples) const {
        if (n_samples < 0) {
            throw py::value_error("n_samples must be non-negative");
        }
        auto &engine = pomdp_native::default_rng().engine();
        py::list out;
        if (state_is_terminal(next_state_.data())) {
            for (int i = 0; i < n_samples; ++i) {
                out.append(terminal_observation());
            }
            return out;
        }
        SensorView view;
        read_sensor(params_, next_state_.data(), view);
        for (int i = 0; i < n_samples; ++i) {
            out.append(draw_observation(params_, view, engine));
        }
        return out;
    }

    // log P(reading | next_state) per candidate, as observation_log_probability.
    // ``readings`` is a sequence of readings; telling one bare reading from a
    // sequence of them is left to the Python caller (_as_observation_list).
    py::array_t<double> log_probability(const py::sequence &readings) const {
        const auto n_readings = static_cast<py::ssize_t>(readings.size());
        py::array_t<double> out(n_readings);
        double *scores = out.mutable_data();
        std::vector<long long> flat;

        if (state_is_terminal(next_state_.data())) {
            for (py::ssize_t i = 0; i < n_readings; ++i) {
                read_reading(readings[static_cast<std::size_t>(i)], flat);
                const bool is_terminal_reading =
                    flat.size() == 1 && flat[0] == kObservationTerminal;
                scores[i] = is_terminal_reading ? 0.0 : kNegInf;
            }
            return out;
        }
        SensorView view;
        read_sensor(params_, next_state_.data(), view);
        for (py::ssize_t i = 0; i < n_readings; ++i) {
            read_reading(readings[static_cast<std::size_t>(i)], flat);
            scores[i] = live_log_likelihood(params_, view, flat);
        }
        return out;
    }

    py::array_t<double> probability(const py::sequence &readings) const {
        py::array_t<double> logs = log_probability(readings);
        double *values = logs.mutable_data();
        for (py::ssize_t i = 0; i < logs.shape(0); ++i) {
            values[i] = std::exp(values[i]);
        }
        return logs;
    }

    // log P(observation | row) for each next-state row.
    py::array_t<double> batch_log_likelihood(const CArray &next_particles,
                                             const py::object &observation) const {
        const int width = params_.state_size();
        std::size_t n_rows = 0;
        const double *rows = rows_of(next_particles, width, "next_particles", n_rows);
        std::vector<long long> flat;
        read_reading(observation, flat);
        py::array_t<double> out(static_cast<py::ssize_t>(n_rows));
        double *scores = out.mutable_data();
        SensorView view;
        for (std::size_t i = 0; i < n_rows; ++i) {
            const double *row = rows + i * width;
            if (state_is_terminal(row)) {
                scores[i] = (flat.size() == 1 && flat[0] == kObservationTerminal) ? 0.0 : kNegInf;
                continue;
            }
            read_sensor(params_, row, view);
            scores[i] = live_log_likelihood(params_, view, flat);
        }
        return out;
    }

    py::array_t<double> next_state_property() const {
        py::array_t<double> out(static_cast<py::ssize_t>(next_state_.size()));
        std::copy(next_state_.begin(), next_state_.end(), out.mutable_data());
        return out;
    }
    int action_property() const { return action_; }

  private:
    SnakeParams params_;
    int action_;
    std::vector<double> next_state_;
};

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Snake POMDP native C++ kernels (pomdp_native).";

    m.def(
        "set_seed", [](std::uint64_t seed) { pomdp_native::set_default_seed(seed); },
        py::arg("seed"), "Seed the module-local RNG used by every sampler in this module.");

    py::class_<SnakeTransitionCpp>(m, "SnakeTransitionCpp")
        .def(py::init<const CArray &, int, int, int, int>(), py::arg("state"), py::arg("action"),
             py::arg("grid_size"), py::arg("target_length"), py::arg("starvation_limit"))
        .def("sample", &SnakeTransitionCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &SnakeTransitionCpp::probability, py::arg("values"))
        .def("log_probability", &SnakeTransitionCpp::log_probability, py::arg("values"))
        .def("batch_sample", &SnakeTransitionCpp::batch_sample, py::arg("particles"))
        .def("set_state", &SnakeTransitionCpp::set_state, py::arg("state"))
        .def_property_readonly("state", &SnakeTransitionCpp::state_property)
        .def_property_readonly("action", &SnakeTransitionCpp::action_property);

    py::class_<SnakeObservationCpp>(m, "SnakeObservationCpp")
        .def(py::init<const CArray &, int, int, int, int, double, double>(),
             py::arg("next_state"), py::arg("action"), py::arg("grid_size"),
             py::arg("target_length"), py::arg("window_radius"),
             py::arg("detection_probability"), py::arg("scent_accuracy"))
        .def("sample", &SnakeObservationCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &SnakeObservationCpp::probability, py::arg("values"))
        .def("log_probability", &SnakeObservationCpp::log_probability, py::arg("values"))
        .def("batch_log_likelihood", &SnakeObservationCpp::batch_log_likelihood,
             py::arg("next_particles"), py::arg("observation"))
        .def("set_next_state", &SnakeObservationCpp::set_next_state, py::arg("next_state"))
        .def_property_readonly("next_state", &SnakeObservationCpp::next_state_property)
        .def_property_readonly("action", &SnakeObservationCpp::action_property);

    m.def(
        "sample_next_step",
        [](const CArray &state, int action, int grid_size, int target_length,
           int starvation_limit, int window_radius, double detection_probability,
           double scent_accuracy) -> py::tuple {
            const SnakeParams params =
                make_full_params(grid_size, target_length, starvation_limit, window_radius,
                                 detection_probability, scent_accuracy);
            const int width = params.state_size();
            if (state.ndim() != 1 || state.shape(0) != width) {
                throw py::value_error("state must be a 1-D array of width " +
                                      std::to_string(width));
            }
            auto &engine = pomdp_native::default_rng().engine();
            py::array_t<double> next_state(width);
            double *buffer = next_state.mutable_data();
            const double reward =
                sample_transition_with_reward(params, state.data(), action, buffer, engine);
            py::tuple observation = sample_observation_one(params, buffer, engine);
            return py::make_tuple(next_state, observation, reward);
        },
        py::arg("state"), py::arg("action"), py::arg("grid_size"), py::arg("target_length"),
        py::arg("starvation_limit"), py::arg("window_radius"), py::arg("detection_probability"),
        py::arg("scent_accuracy"),
        "One (next_state, observation, reward) draw. Same RNG draws, in the same order, as "
        "a transition sample followed by an observation sample.");

    m.def(
        "reward",
        [](const CArray &state, int action, int grid_size, int target_length,
           int starvation_limit) -> double {
            const SnakeParams params =
                make_transition_params(grid_size, target_length, starvation_limit);
            const int width = params.state_size();
            if (state.ndim() != 1 || state.shape(0) != width) {
                throw py::value_error("state must be a 1-D array of width " +
                                      std::to_string(width));
            }
            return reward_of(params, state.data(), action);
        },
        py::arg("state"), py::arg("action"), py::arg("grid_size"), py::arg("target_length"),
        py::arg("starvation_limit"),
        "Reward of one (state, action): +1 for eating, -1 for dying, 0 otherwise.");

    m.def(
        "reward_batch",
        [](const CArray &states, int action, int grid_size, int target_length,
           int starvation_limit) -> py::array_t<double> {
            const SnakeParams params =
                make_transition_params(grid_size, target_length, starvation_limit);
            const int width = params.state_size();
            std::size_t n_rows = 0;
            const double *rows = rows_of(states, width, "states", n_rows);
            py::array_t<double> out(static_cast<py::ssize_t>(n_rows));
            double *rewards = out.mutable_data();
            for (std::size_t i = 0; i < n_rows; ++i) {
                rewards[i] = reward_of(params, rows + i * width, action);
            }
            return out;
        },
        py::arg("states"), py::arg("action"), py::arg("grid_size"), py::arg("target_length"),
        py::arg("starvation_limit"),
        "Reward of one action against each state row. Returns (N,) float64.");

    m.def(
        "simulate_rollout_discrete",
        [](const CArray &initial_state, const py::array_t<std::int32_t, py::array::c_style |
                                                                            py::array::forcecast>
                                            &action_indices,
           int max_depth, int start_depth, double discount_factor, int grid_size,
           int target_length, int starvation_limit) -> double {
            const SnakeParams params =
                make_transition_params(grid_size, target_length, starvation_limit);
            const int width = params.state_size();
            std::vector<double> state = state_vector(initial_state, width, "initial_state");
            std::vector<double> next(static_cast<std::size_t>(width));
            const int steps = max_depth - start_depth;
            if (steps > 0 && action_indices.size() < steps) {
                throw py::value_error("action_indices must hold max_depth - start_depth actions");
            }
            const std::int32_t *actions = action_indices.data();
            auto &engine = pomdp_native::default_rng().engine();
            double total = 0.0;
            double discount = 1.0;
            for (int step = 0; step < steps; ++step) {
                if (state_is_terminal(state.data())) {
                    break;
                }
                const double reward = sample_transition_with_reward(
                    params, state.data(), static_cast<int>(actions[step]), next.data(), engine);
                total += discount * reward;
                discount *= discount_factor;
                state.swap(next);
            }
            return total;
        },
        py::arg("initial_state"), py::arg("action_indices"), py::arg("max_depth"),
        py::arg("start_depth"), py::arg("discount_factor"), py::arg("grid_size"),
        py::arg("target_length"), py::arg("starvation_limit"),
        "Random rollout from initial_state with pre-drawn action indices; returns the "
        "discounted sum of rewards.");
}
