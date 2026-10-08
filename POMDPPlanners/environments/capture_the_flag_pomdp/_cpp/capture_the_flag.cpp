// SPDX-License-Identifier: MIT

// CaptureTheFlag POMDP native extension.
//
// A line-by-line port of the scalar model in ``capture_the_flag_pomdp.py``:
// the per-player move distributions, the red role policy, the ordered
// pick-up / tagging / scoring stages, the range and flag-detector observation
// model, the reward, and a random rollout. The Python methods stay in the
// environment as the reference (``_python_*``), and the native-equivalence
// tests compare the two.
//
// Structure differs from the per-action kernel classes of the other ports on
// purpose. The configuration has about thirty fields and the action is a
// plain integer, so one ``CaptureTheFlagModelCpp`` object holds the
// configuration and every method takes ``(state, action)``. A hot-path call is
// then one pybind11 call instead of ``set_state`` followed by ``sample``.
//
// Exactness. Every probability, likelihood and reward is computed in the same
// operation order as the Python code, so the two agree bit for bit:
//   * Probabilities are accumulated in the order the Python dicts accumulate
//     them, and joint outcomes are enumerated in ``_cartesian`` order.
//   * Likelihoods are returned as probabilities, not logs; the Python caller
//     applies ``np.log`` exactly as the reference does. ``std::log`` and
//     numpy's log are not guaranteed to agree in the last bit.
//   * Floating-point contraction (fused multiply-add) is turned off below.
//     Clang contracts ``a * b - c * d`` by default on ARM64, which would
//     change the last bit of a reward. GCC under ``-std=c++17`` (what
//     Pybind11Extension passes) already defaults to ``-ffp-contract=off``.
//
// Randomness comes from the module-local RNG (pomdp_native/rng.hpp), seeded
// by ``set_seed``. Samples follow the same distributions as the Python
// sampler but are not the same draws, because the Python sampler draws from
// ``np.random``.

#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pomdp_native/rng.hpp"

#ifdef __clang__
#pragma clang fp contract(off)
#endif

namespace py = pybind11;

namespace {

// Per-player action ids, matching capture_the_flag_pomdp_utils.
constexpr int kPlayerActions = 6;
constexpr int kActionScan = 5;
constexpr int kDx[kPlayerActions] = {0, 1, 0, -1, 0, 0};
constexpr int kDy[kPlayerActions] = {1, 0, -1, 0, 0, 0};
// The two slip directions per move action (N, E, S, W), in PERPENDICULAR order.
constexpr int kPerpendicular[4][2] = {{1, 3}, {0, 2}, {1, 3}, {0, 2}};
// Neighbour order of ``_neighbours``: (0, 1), (1, 0), (0, -1), (-1, 0).
constexpr int kNeighbourDx[4] = {0, 1, 0, -1};
constexpr int kNeighbourDy[4] = {1, 0, -1, 0};
// A player has at most five outcomes: stay plus four neighbours.
constexpr int kMaxOptions = 5;
// Fixed scratch size per team. The environment falls back to the Python
// reference for larger teams rather than this file allocating per call.
constexpr int kMaxPlayers = 64;

struct Cell {
    int x;
    int y;
};

inline bool same_cell(Cell a, Cell b) { return a.x == b.x && a.y == b.y; }
inline bool cell_less(Cell a, Cell b) { return a.x < b.x || (a.x == b.x && a.y < b.y); }
inline int manhattan(Cell a, Cell b) { return std::abs(a.x - b.x) + std::abs(a.y - b.y); }

// A small ``(cell, probability)`` table that accumulates like the Python dict
// ``outcomes[cell] = outcomes.get(cell, 0.0) + weight`` and sorts like
// ``sorted(outcomes.items())``.
struct Options {
    Cell cell[kMaxOptions];
    double prob[kMaxOptions];
    int n = 0;

    void add(Cell c, double weight) {
        for (int k = 0; k < n; ++k) {
            if (same_cell(cell[k], c)) {
                prob[k] += weight;
                return;
            }
        }
        cell[n] = c;
        prob[n] = 0.0 + weight;
        ++n;
    }

    void sort() {
        for (int a = 1; a < n; ++a) {
            Cell c = cell[a];
            double p = prob[a];
            int b = a - 1;
            while (b >= 0 && cell_less(c, cell[b])) {
                cell[b + 1] = cell[b];
                prob[b + 1] = prob[b];
                --b;
            }
            cell[b + 1] = c;
            prob[b + 1] = p;
        }
    }

    // ``_draw_cell``: one option needs no draw; otherwise walk the CDF.
    Cell draw(pomdp_native::RNGState &rng) const {
        if (n == 1) {
            return cell[0];
        }
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const double u = uniform(rng.engine());
        double cumulative = 0.0;
        for (int k = 0; k < n; ++k) {
            cumulative += prob[k];
            if (u < cumulative) {
                return cell[k];
            }
        }
        return cell[n - 1];
    }
};

// One range reading's distribution, sorted by value, as ``_range_probabilities``
// builds it and ``_inverse_cdf`` walks it.
struct RangeTable {
    int value[3];
    double prob[3];
    int n = 0;

    double get(int v) const {
        for (int k = 0; k < n; ++k) {
            if (value[k] == v) {
                return prob[k];
            }
        }
        return 0.0;
    }
};

inline int py_floor_div(long long a, long long b, long long *mod) {
    long long q = a / b;
    long long r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) {
        r += b;
        q -= 1;
    }
    *mod = r;
    return static_cast<int>(q);
}

struct Scratch {
    Cell blue[kMaxPlayers];
    Cell red[kMaxPlayers];
    int actions[kMaxPlayers];
};

class CaptureTheFlagModelCpp {
  public:
    // pylint-style argument list mirrors the Python constructor.
    CaptureTheFlagModelCpp(int width, int height, int midline,
                           const std::vector<std::pair<int, int>> &trees, int n_blue, int n_red,
                           int n_red_defenders, std::pair<int, int> blue_base,
                           std::pair<int, int> red_base, std::pair<int, int> blue_flag_cell,
                           const std::vector<std::pair<int, int>> &red_flag_candidates,
                           double slip_probability, double range_error_probability,
                           double red_pursuit_probability, int red_alert_radius, int freeze_steps,
                           int tagger_cooldown_steps, double detector_half_distance_move,
                           double detector_half_distance_scan, int score_to_win,
                           double capture_reward, double concede_penalty, double tagged_penalty,
                           double tag_reward, double pickup_reward, double move_cost,
                           double scan_cost)
        : width_(width),
          height_(height),
          midline_(midline),
          n_blue_(n_blue),
          n_red_(n_red),
          n_red_defenders_(n_red_defenders),
          blue_base_{blue_base.first, blue_base.second},
          red_base_{red_base.first, red_base.second},
          blue_flag_cell_{blue_flag_cell.first, blue_flag_cell.second},
          slip_(slip_probability),
          range_error_(range_error_probability),
          pursuit_(red_pursuit_probability),
          alert_radius_(red_alert_radius),
          freeze_steps_(freeze_steps),
          cooldown_steps_(tagger_cooldown_steps),
          half_move_(detector_half_distance_move),
          half_scan_(detector_half_distance_scan),
          score_to_win_(score_to_win),
          capture_reward_(capture_reward),
          concede_penalty_(concede_penalty),
          tagged_penalty_(tagged_penalty),
          tag_reward_(tag_reward),
          pickup_reward_(pickup_reward),
          move_cost_(move_cost),
          scan_cost_(scan_cost) {
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("grid dimensions must be positive");
        }
        if (n_blue < 1 || n_red < 1 || n_blue > kMaxPlayers || n_red > kMaxPlayers) {
            throw std::invalid_argument("team sizes must be in [1, " +
                                        std::to_string(kMaxPlayers) + "]");
        }
        if (red_flag_candidates.empty()) {
            throw std::invalid_argument("at least one red flag candidate is required");
        }
        tree_mask_.assign(static_cast<std::size_t>(width) * height, 0);
        for (const auto &t : trees) {
            if (t.first >= 0 && t.first < width && t.second >= 0 && t.second < height) {
                tree_mask_[static_cast<std::size_t>(t.first) * height + t.second] = 1;
            }
        }
        for (const auto &c : red_flag_candidates) {
            candidates_.push_back(Cell{c.first, c.second});
        }
        for (const Cell &flag : candidates_) {
            guard_posts_.push_back(compute_guard_post(flag));
        }

        red_pos_ = 2 * n_blue;
        flag_cell_ = 2 * n_blue + 2 * n_red;
        carrier_red_flag_ = flag_cell_ + 1;
        carrier_blue_flag_ = flag_cell_ + 2;
        freeze_blue_ = flag_cell_ + 3;
        freeze_red_ = freeze_blue_ + n_blue;
        cooldown_blue_ = freeze_red_ + n_red;
        cooldown_red_ = cooldown_blue_ + n_blue;
        score_blue_ = cooldown_red_ + n_red;
        score_red_ = score_blue_ + 1;
        state_size_ = score_red_ + 1;
        observation_size_ = 2 * n_blue + n_blue * n_red + n_blue + 2 + n_blue + 2;
        max_range_ = width + height - 2;
        n_actions_ = 1;
        for (int i = 0; i < n_blue; ++i) {
            n_actions_ *= kPlayerActions;
        }
    }

    int state_size() const { return state_size_; }
    int observation_size() const { return observation_size_; }
    int n_actions() const { return static_cast<int>(n_actions_); }

    // ----------------------------------------------------------- field

    bool in_bounds(Cell c) const { return c.x >= 0 && c.x < width_ && c.y >= 0 && c.y < height_; }

    bool is_free(Cell c) const {
        return in_bounds(c) && tree_mask_[static_cast<std::size_t>(c.x) * height_ + c.y] == 0;
    }

    bool is_blue_half(Cell c) const { return c.x < midline_; }
    bool is_red_half(Cell c) const { return c.x > midline_; }

    // ``_neighbours``: the cell itself, then its free 4-neighbours.
    int neighbours(Cell c, Cell out[kMaxOptions]) const {
        int n = 0;
        out[n++] = c;
        for (int k = 0; k < 4; ++k) {
            Cell candidate{c.x + kNeighbourDx[k], c.y + kNeighbourDy[k]};
            if (is_free(candidate)) {
                out[n++] = candidate;
            }
        }
        return n;
    }

    Cell compute_guard_post(Cell flag) const {
        Cell options[kMaxOptions];
        const int n = neighbours(flag, options);
        bool found = false;
        Cell best{0, 0};
        int best_distance = 0;
        for (int k = 0; k < n; ++k) {
            if (same_cell(options[k], flag)) {
                continue;
            }
            const int d = manhattan(options[k], red_base_);
            if (!found || d < best_distance ||
                (d == best_distance && cell_less(options[k], best))) {
                best = options[k];
                best_distance = d;
                found = true;
            }
        }
        return found ? best : flag;
    }

    // --------------------------------------------------------- reading state

    static int as_int(double v) { return static_cast<int>(v); }

    // ``red_flag_candidates[int(state[flag_cell])]``, with Python's negative
    // index wrap and IndexError for an index outside the list.
    int flag_index(const double *s) const {
        int idx = as_int(s[flag_cell_]);
        const int k = static_cast<int>(candidates_.size());
        if (idx < 0) {
            idx += k;
        }
        if (idx < 0 || idx >= k) {
            throw py::index_error("red flag candidate index out of range");
        }
        return idx;
    }

    void read_cells(const double *s, Cell *blue, Cell *red) const {
        for (int i = 0; i < n_blue_; ++i) {
            blue[i] = Cell{as_int(s[2 * i]), as_int(s[2 * i + 1])};
        }
        for (int j = 0; j < n_red_; ++j) {
            red[j] = Cell{as_int(s[red_pos_ + 2 * j]), as_int(s[red_pos_ + 2 * j + 1])};
        }
    }

    // ``decode_joint_action`` with Python's floor modulo, so any integer
    // decodes exactly as the reference decodes it.
    void decode(long long action, int *out) const {
        long long remaining = action;
        for (int i = 0; i < n_blue_; ++i) {
            long long digit = 0;
            remaining = py_floor_div(remaining, kPlayerActions, &digit);
            out[i] = static_cast<int>(digit);
        }
    }

    bool is_terminal(const double *s) const {
        return as_int(s[score_blue_]) >= score_to_win_ || as_int(s[score_red_]) >= score_to_win_;
    }

    // ----------------------------------------------------- transition model

    Options blue_move(Cell cell, int player_action, bool frozen) const {
        Options out;
        if (frozen || player_action < 0 || player_action > 3) {
            out.add(cell, 1.0);
            return out;
        }
        const int moves[3] = {player_action, kPerpendicular[player_action][0],
                              kPerpendicular[player_action][1]};
        const double weights[3] = {1.0 - slip_, slip_ / 2.0, slip_ / 2.0};
        for (int k = 0; k < 3; ++k) {
            if (weights[k] == 0.0) {
                continue;
            }
            Cell target{cell.x + kDx[moves[k]], cell.y + kDy[moves[k]]};
            out.add(is_free(target) ? target : cell, weights[k]);
        }
        out.sort();
        return out;
    }

    Cell red_target(int j, const Cell *blue, Cell red_cell, int flag_idx,
                    int carrier_blue_flag) const {
        if (carrier_blue_flag == j + 1) {
            return red_base_;
        }
        if (j >= n_red_defenders_) {  // RedRole.ATTACK
            return blue_flag_cell_;
        }
        bool found = false;
        Cell nearest{0, 0};
        int nearest_distance = 0;
        for (int i = 0; i < n_blue_; ++i) {
            if (!is_red_half(blue[i])) {
                continue;
            }
            const int d = manhattan(red_cell, blue[i]);
            if (!found || d < nearest_distance ||
                (d == nearest_distance && cell_less(blue[i], nearest))) {
                nearest = blue[i];
                nearest_distance = d;
                found = true;
            }
        }
        if (found && nearest_distance <= alert_radius_) {
            return nearest;
        }
        return guard_posts_[static_cast<std::size_t>(flag_idx)];
    }

    Options red_move(Cell red_cell, Cell target, bool frozen) const {
        Options out;
        if (frozen) {
            out.add(red_cell, 1.0);
            return out;
        }
        Cell options[kMaxOptions];
        const int n = neighbours(red_cell, options);
        int best_distance = manhattan(options[0], target);
        for (int k = 1; k < n; ++k) {
            const int d = manhattan(options[k], target);
            if (d < best_distance) {
                best_distance = d;
            }
        }
        int n_closing = 0;
        for (int k = 0; k < n; ++k) {
            if (manhattan(options[k], target) == best_distance) {
                ++n_closing;
            }
        }
        const double uniform_share = (1.0 - pursuit_) / static_cast<double>(n);
        for (int k = 0; k < n; ++k) {
            out.add(options[k], uniform_share);
        }
        const double closing_share = pursuit_ / static_cast<double>(n_closing);
        for (int k = 0; k < n; ++k) {
            if (manhattan(options[k], target) == best_distance) {
                out.add(options[k], closing_share);
            }
        }
        out.sort();
        return out;
    }

    // ``_apply_deterministic_stages``. ``blue`` and ``red`` are modified in
    // place (tagged players are sent home), as the Python lists are.
    void apply_stages(const double *s, Cell *blue, Cell *red, double *out) const {
        std::fill(out, out + state_size_, 0.0);
        out[flag_cell_] = s[flag_cell_];

        int freeze_b[kMaxPlayers];
        int freeze_r[kMaxPlayers];
        int cool_b[kMaxPlayers];
        int cool_r[kMaxPlayers];
        bool was_frozen_b[kMaxPlayers];
        bool was_frozen_r[kMaxPlayers];
        bool tagged_b[kMaxPlayers];
        bool tagged_r[kMaxPlayers];
        for (int i = 0; i < n_blue_; ++i) {
            const int f = as_int(s[freeze_blue_ + i]);
            freeze_b[i] = f - 1 > 0 ? f - 1 : 0;
            const int c = as_int(s[cooldown_blue_ + i]);
            cool_b[i] = c - 1 > 0 ? c - 1 : 0;
            was_frozen_b[i] = f > 0;
            tagged_b[i] = false;
        }
        for (int j = 0; j < n_red_; ++j) {
            const int f = as_int(s[freeze_red_ + j]);
            freeze_r[j] = f - 1 > 0 ? f - 1 : 0;
            const int c = as_int(s[cooldown_red_ + j]);
            cool_r[j] = c - 1 > 0 ? c - 1 : 0;
            was_frozen_r[j] = f > 0;
            tagged_r[j] = false;
        }

        int carrier_red_flag = as_int(s[carrier_red_flag_]);
        int carrier_blue_flag = as_int(s[carrier_blue_flag_]);
        const Cell flag = candidates_[static_cast<std::size_t>(flag_index(s))];

        // Pick-up. Lowest index wins a tie.
        if (carrier_red_flag == 0) {
            for (int i = 0; i < n_blue_; ++i) {
                if (same_cell(blue[i], flag) && !was_frozen_b[i]) {
                    carrier_red_flag = i + 1;
                    break;
                }
            }
        }
        if (carrier_blue_flag == 0) {
            for (int j = 0; j < n_red_; ++j) {
                if (same_cell(red[j], blue_flag_cell_) && !was_frozen_r[j]) {
                    carrier_blue_flag = j + 1;
                    break;
                }
            }
        }

        // Tagging. Red tags first, then blue, as in the reference.
        for (int j = 0; j < n_red_; ++j) {
            if (was_frozen_r[j] || as_int(s[cooldown_red_ + j]) > 0) {
                continue;
            }
            for (int i = 0; i < n_blue_; ++i) {
                if (!same_cell(blue[i], red[j]) || !is_red_half(blue[i])) {
                    continue;
                }
                if (was_frozen_b[i] || tagged_b[i]) {
                    continue;
                }
                blue[i] = blue_base_;
                freeze_b[i] = freeze_steps_;
                cool_r[j] = cooldown_steps_;
                if (carrier_red_flag == i + 1) {
                    carrier_red_flag = 0;
                }
                tagged_b[i] = true;
                break;
            }
        }
        for (int i = 0; i < n_blue_; ++i) {
            if (was_frozen_b[i] || tagged_b[i]) {
                continue;
            }
            if (as_int(s[cooldown_blue_ + i]) > 0) {
                continue;
            }
            for (int j = 0; j < n_red_; ++j) {
                if (!same_cell(red[j], blue[i]) || !is_blue_half(red[j])) {
                    continue;
                }
                if (was_frozen_r[j] || tagged_r[j]) {
                    continue;
                }
                red[j] = red_base_;
                freeze_r[j] = freeze_steps_;
                cool_b[i] = cooldown_steps_;
                if (carrier_blue_flag == j + 1) {
                    carrier_blue_flag = 0;
                }
                tagged_r[j] = true;
                break;
            }
        }

        // Scoring, both sides judged against the same carrier ids.
        int score_blue = as_int(s[score_blue_]);
        int score_red = as_int(s[score_red_]);
        const bool blue_scores = carrier_red_flag != 0 &&
                                 same_cell(blue[carrier_red_flag - 1], blue_base_) &&
                                 carrier_blue_flag == 0;
        const bool red_scores = carrier_blue_flag != 0 &&
                                same_cell(red[carrier_blue_flag - 1], red_base_) &&
                                carrier_red_flag == 0;
        if (blue_scores) {
            score_blue += 1;
            carrier_red_flag = 0;
        }
        if (red_scores) {
            score_red += 1;
            carrier_blue_flag = 0;
        }

        for (int i = 0; i < n_blue_; ++i) {
            out[2 * i] = static_cast<double>(blue[i].x);
            out[2 * i + 1] = static_cast<double>(blue[i].y);
            out[freeze_blue_ + i] = static_cast<double>(freeze_b[i]);
            out[cooldown_blue_ + i] = static_cast<double>(cool_b[i]);
        }
        for (int j = 0; j < n_red_; ++j) {
            out[red_pos_ + 2 * j] = static_cast<double>(red[j].x);
            out[red_pos_ + 2 * j + 1] = static_cast<double>(red[j].y);
            out[freeze_red_ + j] = static_cast<double>(freeze_r[j]);
            out[cooldown_red_ + j] = static_cast<double>(cool_r[j]);
        }
        out[carrier_red_flag_] = static_cast<double>(carrier_red_flag);
        out[carrier_blue_flag_] = static_cast<double>(carrier_blue_flag);
        out[score_blue_] = static_cast<double>(score_blue);
        out[score_red_] = static_cast<double>(score_red);
    }

    // ``sample_next_state`` for one draw, including the terminal copy.
    void draw_successor(const double *s, long long action, pomdp_native::RNGState &rng,
                        Scratch &scratch, double *out) const {
        if (is_terminal(s)) {
            std::memcpy(out, s, sizeof(double) * static_cast<std::size_t>(state_size_));
            return;
        }
        Cell current_blue[kMaxPlayers];
        read_cells(s, current_blue, scratch.red);
        decode(action, scratch.actions);
        for (int i = 0; i < n_blue_; ++i) {
            scratch.blue[i] = blue_move(current_blue[i], scratch.actions[i],
                                        as_int(s[freeze_blue_ + i]) > 0)
                                  .draw(rng);
        }
        const int flag_idx = flag_index(s);
        const int carrier_blue_flag = as_int(s[carrier_blue_flag_]);
        Cell drawn_red[kMaxPlayers];
        for (int j = 0; j < n_red_; ++j) {
            const Cell target =
                red_target(j, scratch.blue, scratch.red[j], flag_idx, carrier_blue_flag);
            drawn_red[j] = red_move(scratch.red[j], target, as_int(s[freeze_red_ + j]) > 0)
                               .draw(rng);
        }
        apply_stages(s, scratch.blue, drawn_red, out);
    }

    // ``_successor_distribution``: distinct successors in first-seen order and
    // their probabilities, summed in the reference's enumeration order.
    void successor_distribution(const double *s, long long action,
                                std::vector<double> &states_flat,
                                std::vector<double> &probs) const {
        states_flat.clear();
        probs.clear();
        Cell blue[kMaxPlayers];
        Cell red[kMaxPlayers];
        int actions[kMaxPlayers];
        read_cells(s, blue, red);
        decode(action, actions);
        const int flag_idx = flag_index(s);
        const int carrier_blue_flag = as_int(s[carrier_blue_flag_]);

        std::vector<Options> blue_options(static_cast<std::size_t>(n_blue_));
        for (int i = 0; i < n_blue_; ++i) {
            blue_options[i] = blue_move(blue[i], actions[i], as_int(s[freeze_blue_ + i]) > 0);
        }

        std::unordered_map<std::string, std::size_t> index;
        std::vector<double> successor(static_cast<std::size_t>(state_size_));
        Cell blue_combo[kMaxPlayers];
        Cell red_combo[kMaxPlayers];
        Cell blue_work[kMaxPlayers];
        std::vector<Options> red_options(static_cast<std::size_t>(n_red_));

        // Odometer over the players with player 0 the slowest digit, which is
        // the order ``_cartesian`` expands combinations in.
        std::vector<int> blue_digit(static_cast<std::size_t>(n_blue_), 0);
        std::vector<int> red_digit(static_cast<std::size_t>(n_red_), 0);

        auto advance = [](std::vector<int> &digit, const std::vector<Options> &opts) {
            for (int p = static_cast<int>(digit.size()) - 1; p >= 0; --p) {
                if (++digit[p] < opts[p].n) {
                    return true;
                }
                digit[p] = 0;
            }
            return false;
        };
        auto combo_weight = [](const std::vector<int> &digit, const std::vector<Options> &opts,
                               Cell *cells, double *prob) {
            double p = 1.0;
            for (std::size_t k = 0; k < digit.size(); ++k) {
                const double w = opts[k].prob[digit[k]];
                if (!(w > 0.0)) {
                    return false;
                }
                cells[k] = opts[k].cell[digit[k]];
                p = p * w;
            }
            *prob = p;
            return true;
        };

        do {
            double blue_prob = 0.0;
            if (!combo_weight(blue_digit, blue_options, blue_combo, &blue_prob)) {
                continue;
            }
            for (int j = 0; j < n_red_; ++j) {
                const Cell target = red_target(j, blue_combo, red[j], flag_idx, carrier_blue_flag);
                red_options[j] = red_move(red[j], target, as_int(s[freeze_red_ + j]) > 0);
            }
            std::fill(red_digit.begin(), red_digit.end(), 0);
            do {
                double red_prob = 0.0;
                if (!combo_weight(red_digit, red_options, red_combo, &red_prob)) {
                    continue;
                }
                for (int i = 0; i < n_blue_; ++i) {
                    blue_work[i] = blue_combo[i];
                }
                apply_stages(s, blue_work, red_combo, successor.data());
                std::string key(reinterpret_cast<const char *>(successor.data()),
                                sizeof(double) * successor.size());
                auto found = index.find(key);
                if (found == index.end()) {
                    index.emplace(std::move(key), probs.size());
                    probs.push_back(0.0 + blue_prob * red_prob);
                    states_flat.insert(states_flat.end(), successor.begin(), successor.end());
                } else {
                    probs[found->second] += blue_prob * red_prob;
                }
            } while (advance(red_digit, red_options));
        } while (advance(blue_digit, blue_options));
    }

    // ---------------------------------------------------- observation model

    double detection_probability(int distance, int player_action) const {
        const double half = player_action == kActionScan ? half_scan_ : half_move_;
        return 0.5 * (1.0 + std::pow(2.0, -static_cast<double>(distance) / half));
    }

    RangeTable range_table(int true_distance) const {
        RangeTable unsorted;
        const int values[3] = {true_distance, true_distance - 1, true_distance + 1};
        const double weights[3] = {1.0 - range_error_, range_error_ / 2.0, range_error_ / 2.0};
        for (int k = 0; k < 3; ++k) {
            if (weights[k] == 0.0) {
                continue;
            }
            int clipped = values[k] < 0 ? 0 : values[k];
            clipped = clipped < max_range_ ? clipped : max_range_;
            bool merged = false;
            for (int m = 0; m < unsorted.n; ++m) {
                if (unsorted.value[m] == clipped) {
                    unsorted.prob[m] += weights[k];
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                unsorted.value[unsorted.n] = clipped;
                unsorted.prob[unsorted.n] = 0.0 + weights[k];
                ++unsorted.n;
            }
        }
        // Sort by value for the inverse CDF.
        for (int a = 1; a < unsorted.n; ++a) {
            const int v = unsorted.value[a];
            const double p = unsorted.prob[a];
            int b = a - 1;
            while (b >= 0 && unsorted.value[b] > v) {
                unsorted.value[b + 1] = unsorted.value[b];
                unsorted.prob[b + 1] = unsorted.prob[b];
                --b;
            }
            unsorted.value[b + 1] = v;
            unsorted.prob[b + 1] = p;
        }
        return unsorted;
    }

    void sample_observation_into(const double *ns, long long action, pomdp_native::RNGState &rng,
                                 double *out) const {
        if (is_terminal(ns)) {
            for (int k = 0; k < observation_size_; ++k) {
                out[k] = -1.0;
            }
            return;
        }
        Cell blue[kMaxPlayers];
        Cell red[kMaxPlayers];
        int actions[kMaxPlayers];
        read_cells(ns, blue, red);
        decode(action, actions);
        const Cell flag = candidates_[static_cast<std::size_t>(flag_index(ns))];
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        int k = 0;
        for (int i = 0; i < n_blue_; ++i) {
            out[k++] = static_cast<double>(blue[i].x);
            out[k++] = static_cast<double>(blue[i].y);
        }
        for (int i = 0; i < n_blue_; ++i) {
            for (int j = 0; j < n_red_; ++j) {
                const RangeTable table = range_table(manhattan(blue[i], red[j]));
                const double u = uniform(rng.engine());
                double cumulative = 0.0;
                int value = table.value[table.n - 1];
                for (int m = 0; m < table.n; ++m) {
                    cumulative += table.prob[m];
                    if (u < cumulative) {
                        value = table.value[m];
                        break;
                    }
                }
                out[k++] = static_cast<double>(value);
            }
        }
        for (int i = 0; i < n_blue_; ++i) {
            const double p = detection_probability(manhattan(blue[i], flag), actions[i]);
            out[k++] = uniform(rng.engine()) < p ? 1.0 : 0.0;
        }
        k = write_suffix(ns, out, k);
    }

    int write_suffix(const double *ns, double *out, int k) const {
        out[k++] = ns[carrier_red_flag_];
        out[k++] = ns[carrier_blue_flag_] != 0.0 ? 1.0 : 0.0;
        for (int i = 0; i < n_blue_; ++i) {
            out[k++] = ns[freeze_blue_ + i];
        }
        out[k++] = ns[score_blue_];
        out[k++] = ns[score_red_];
        return k;
    }

    // Per-next-state tables reused across candidate observations.
    struct ObservationContext {
        bool terminal = false;
        std::vector<double> prefix;
        std::vector<double> suffix;
        std::vector<RangeTable> ranges;
        std::vector<double> detection;
    };

    ObservationContext observation_context(const double *ns, long long action) const {
        ObservationContext ctx;
        if (is_terminal(ns)) {
            ctx.terminal = true;
            return ctx;
        }
        Cell blue[kMaxPlayers];
        Cell red[kMaxPlayers];
        int actions[kMaxPlayers];
        read_cells(ns, blue, red);
        decode(action, actions);
        const Cell flag = candidates_[static_cast<std::size_t>(flag_index(ns))];
        for (int i = 0; i < n_blue_; ++i) {
            ctx.prefix.push_back(static_cast<double>(blue[i].x));
            ctx.prefix.push_back(static_cast<double>(blue[i].y));
        }
        for (int i = 0; i < n_blue_; ++i) {
            for (int j = 0; j < n_red_; ++j) {
                ctx.ranges.push_back(range_table(manhattan(blue[i], red[j])));
            }
            ctx.detection.push_back(detection_probability(manhattan(blue[i], flag), actions[i]));
        }
        ctx.suffix.resize(static_cast<std::size_t>(n_blue_ + 4));
        write_suffix(ns, ctx.suffix.data(), 0);
        return ctx;
    }

    // ``observation_log_probability`` before the log, for one candidate.
    double observation_probability_one(const ObservationContext &ctx, const double *values,
                                       std::size_t size) const {
        if (ctx.terminal) {
            if (size != static_cast<std::size_t>(observation_size_)) {
                return 0.0;
            }
            for (std::size_t k = 0; k < size; ++k) {
                if (!(values[k] == -1.0)) {
                    return 0.0;
                }
            }
            return 1.0;
        }
        if (size != static_cast<std::size_t>(observation_size_)) {
            return 0.0;
        }
        const std::size_t n_prefix = ctx.prefix.size();
        const std::size_t n_suffix = ctx.suffix.size();
        for (std::size_t k = 0; k < n_prefix; ++k) {
            if (!(values[k] == ctx.prefix[k])) {
                return 0.0;
            }
        }
        for (std::size_t k = 0; k < n_suffix; ++k) {
            if (!(values[size - n_suffix + k] == ctx.suffix[k])) {
                return 0.0;
            }
        }
        double probability = 1.0;
        const std::size_t n_pairs = ctx.ranges.size();
        for (std::size_t pair = 0; pair < n_pairs; ++pair) {
            const double reading = values[n_prefix + pair];
            if (reading != std::floor(reading)) {
                return 0.0;
            }
            // Outside [0, max_range] no table holds the value; checking first
            // keeps the int conversion defined.
            if (reading < 0.0 || reading > static_cast<double>(max_range_)) {
                probability *= 0.0;
                continue;
            }
            probability *= ctx.ranges[pair].get(static_cast<int>(reading));
        }
        if (probability == 0.0) {
            return 0.0;
        }
        for (int i = 0; i < n_blue_; ++i) {
            const double bit = values[n_prefix + n_pairs + static_cast<std::size_t>(i)];
            if (bit != 0.0 && bit != 1.0) {
                return 0.0;
            }
            probability *= bit == 1.0 ? ctx.detection[i] : 1.0 - ctx.detection[i];
        }
        return probability;
    }

    // ------------------------------------------------------------ reward

    double reward_one(const double *s, long long action, const double *ns) const {
        if (is_terminal(s)) {
            return 0.0;
        }
        const int score_delta_blue = as_int(ns[score_blue_]) - as_int(s[score_blue_]);
        const int score_delta_red = as_int(ns[score_red_]) - as_int(s[score_red_]);
        int suffered = 0;
        for (int i = 0; i < n_blue_; ++i) {
            if (as_int(s[freeze_blue_ + i]) == 0 && as_int(ns[freeze_blue_ + i]) == freeze_steps_) {
                ++suffered;
            }
        }
        int inflicted = 0;
        for (int j = 0; j < n_red_; ++j) {
            if (as_int(s[freeze_red_ + j]) == 0 && as_int(ns[freeze_red_ + j]) == freeze_steps_) {
                ++inflicted;
            }
        }
        const bool picked_up = as_int(s[carrier_red_flag_]) == 0 && as_int(ns[carrier_red_flag_]) != 0;
        int actions[kMaxPlayers];
        decode(action, actions);
        double action_cost = 0.0;
        for (int i = 0; i < n_blue_; ++i) {
            action_cost += actions[i] == kActionScan ? scan_cost_ : move_cost_;
        }
        return capture_reward_ * static_cast<double>(score_delta_blue) -
               concede_penalty_ * static_cast<double>(score_delta_red) -
               tagged_penalty_ * static_cast<double>(suffered) +
               tag_reward_ * static_cast<double>(inflicted) +
               pickup_reward_ * (picked_up ? 1.0 : 0.0) - action_cost;
    }

    // ------------------------------------------------------------ bindings

    const double *checked_state(const py::array_t<double, py::array::c_style | py::array::forcecast>
                                    &arr,
                                const char *label) const {
        if (arr.ndim() != 1 || arr.shape(0) != state_size_) {
            throw std::invalid_argument(std::string(label) + " must be 1-D with length " +
                                        std::to_string(state_size_));
        }
        return arr.data();
    }

    using StateArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

    py::object sample_next_state(const StateArray &state, long long action, int n_samples) const {
        if (n_samples < 1) {
            throw std::invalid_argument("n_samples must be positive");
        }
        const double *s = checked_state(state, "state");
        auto &rng = pomdp_native::default_rng();
        Scratch scratch;
        if (n_samples == 1) {
            py::array_t<double> out(static_cast<py::ssize_t>(state_size_));
            draw_successor(s, action, rng, scratch, out.mutable_data());
            return std::move(out);
        }
        py::list samples;
        for (int k = 0; k < n_samples; ++k) {
            py::array_t<double> out(static_cast<py::ssize_t>(state_size_));
            draw_successor(s, action, rng, scratch, out.mutable_data());
            samples.append(std::move(out));
        }
        return std::move(samples);
    }

    py::array_t<double> batch_sample(const py::array_t<double, py::array::c_style |
                                                                   py::array::forcecast> &states,
                                     long long action) const {
        if (states.ndim() != 2 || states.shape(1) != state_size_) {
            throw std::invalid_argument("states must have shape (N, state_size)");
        }
        const py::ssize_t n = states.shape(0);
        py::array_t<double> out({n, static_cast<py::ssize_t>(state_size_)});
        const double *in = states.data();
        double *dst = out.mutable_data();
        auto &rng = pomdp_native::default_rng();
        Scratch scratch;
        for (py::ssize_t r = 0; r < n; ++r) {
            draw_successor(in + r * state_size_, action, rng, scratch, dst + r * state_size_);
        }
        return out;
    }

    py::array_t<double> transition_probability(const StateArray &state, long long action,
                                               const py::object &next_states) const {
        const double *s = checked_state(state, "state");
        std::vector<std::vector<double>> candidates = rows_of(next_states);
        py::array_t<double> out(static_cast<py::ssize_t>(candidates.size()));
        double *dst = out.mutable_data();
        if (is_terminal(s)) {
            for (std::size_t r = 0; r < candidates.size(); ++r) {
                const auto &c = candidates[r];
                bool equal = c.size() == static_cast<std::size_t>(state_size_);
                for (std::size_t k = 0; equal && k < c.size(); ++k) {
                    equal = c[k] == s[k];
                }
                dst[r] = equal ? 1.0 : 0.0;
            }
            return out;
        }
        std::vector<double> flat;
        std::vector<double> probs;
        successor_distribution(s, action, flat, probs);
        const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(state_size_);
        for (std::size_t r = 0; r < candidates.size(); ++r) {
            const auto &c = candidates[r];
            double p = 0.0;
            if (c.size() == static_cast<std::size_t>(state_size_)) {
                for (std::size_t m = 0; m < probs.size(); ++m) {
                    if (std::memcmp(flat.data() + m * state_size_, c.data(), bytes) == 0) {
                        p = probs[m];
                        break;
                    }
                }
            }
            dst[r] = p;
        }
        return out;
    }

    py::tuple successor_distribution_py(const StateArray &state, long long action) const {
        const double *s = checked_state(state, "state");
        std::vector<double> flat;
        std::vector<double> probs;
        successor_distribution(s, action, flat, probs);
        const auto n = static_cast<py::ssize_t>(probs.size());
        py::array_t<double> states({n, static_cast<py::ssize_t>(state_size_)});
        std::memcpy(states.mutable_data(), flat.data(), sizeof(double) * flat.size());
        py::array_t<double> weights(n);
        std::memcpy(weights.mutable_data(), probs.data(), sizeof(double) * probs.size());
        return py::make_tuple(states, weights);
    }

    py::tuple make_observation(const double *values) const {
        PyObject *tuple = PyTuple_New(observation_size_);
        if (tuple == nullptr) {
            throw py::error_already_set();
        }
        for (int k = 0; k < observation_size_; ++k) {
            PyObject *item = PyFloat_FromDouble(values[k]);
            if (item == nullptr) {
                Py_DECREF(tuple);
                throw py::error_already_set();
            }
            PyTuple_SET_ITEM(tuple, k, item);
        }
        return py::reinterpret_steal<py::tuple>(tuple);
    }

    py::object sample_observation(const StateArray &next_state, long long action,
                                  int n_samples) const {
        if (n_samples < 1) {
            throw std::invalid_argument("n_samples must be positive");
        }
        const double *ns = checked_state(next_state, "next_state");
        auto &rng = pomdp_native::default_rng();
        std::vector<double> buf(static_cast<std::size_t>(observation_size_));
        if (n_samples == 1) {
            sample_observation_into(ns, action, rng, buf.data());
            return make_observation(buf.data());
        }
        py::list samples;
        for (int k = 0; k < n_samples; ++k) {
            sample_observation_into(ns, action, rng, buf.data());
            samples.append(make_observation(buf.data()));
        }
        return std::move(samples);
    }

    // One observation as flat doubles: ``np.asarray(o, dtype=float64).ravel()``.
    static std::vector<double> observation_values(const py::handle &obj) {
        std::vector<double> values;
        PyObject *raw = obj.ptr();
        if (PyTuple_Check(raw) || PyList_Check(raw)) {
            const Py_ssize_t n = PySequence_Fast_GET_SIZE(raw);
            PyObject **items = PySequence_Fast_ITEMS(raw);
            values.reserve(static_cast<std::size_t>(n));
            bool flat = true;
            for (Py_ssize_t k = 0; k < n; ++k) {
                PyObject *item = items[k];
                if (PyFloat_Check(item)) {
                    values.push_back(PyFloat_AS_DOUBLE(item));
                } else if (PyLong_Check(item)) {
                    const double v = PyLong_AsDouble(item);
                    if (v == -1.0 && PyErr_Occurred()) {
                        PyErr_Clear();
                        flat = false;
                        break;
                    }
                    values.push_back(v);
                } else {
                    flat = false;
                    break;
                }
            }
            if (flat) {
                return values;
            }
            values.clear();
        }
        auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obj);
        if (!arr) {
            throw py::error_already_set();
        }
        const double *data = arr.data();
        values.assign(data, data + arr.size());
        return values;
    }

    static std::vector<std::vector<double>> rows_of(const py::object &rows) {
        std::vector<std::vector<double>> out;
        for (const py::handle &row : rows) {
            out.push_back(observation_values(row));
        }
        return out;
    }

    py::array_t<double> observation_probability(const StateArray &next_state, long long action,
                                                const py::object &observations) const {
        const double *ns = checked_state(next_state, "next_state");
        const ObservationContext ctx = observation_context(ns, action);
        const auto rows = rows_of(observations);
        py::array_t<double> out(static_cast<py::ssize_t>(rows.size()));
        double *dst = out.mutable_data();
        for (std::size_t r = 0; r < rows.size(); ++r) {
            dst[r] = observation_probability_one(ctx, rows[r].data(), rows[r].size());
        }
        return out;
    }

    py::array_t<double> batch_observation_probability(
        const py::array_t<double, py::array::c_style | py::array::forcecast> &next_states,
        long long action, const py::object &observation) const {
        if (next_states.ndim() != 2 || next_states.shape(1) != state_size_) {
            throw std::invalid_argument("next_states must have shape (N, state_size)");
        }
        const std::vector<double> values = observation_values(observation);
        const py::ssize_t n = next_states.shape(0);
        py::array_t<double> out(n);
        double *dst = out.mutable_data();
        const double *in = next_states.data();
        for (py::ssize_t r = 0; r < n; ++r) {
            const ObservationContext ctx = observation_context(in + r * state_size_, action);
            dst[r] = observation_probability_one(ctx, values.data(), values.size());
        }
        return out;
    }

    double reward(const StateArray &state, long long action, const StateArray &next_state) const {
        return reward_one(checked_state(state, "state"), action,
                          checked_state(next_state, "next_state"));
    }

    py::array_t<double> reward_batch(
        const py::array_t<double, py::array::c_style | py::array::forcecast> &states,
        long long action,
        const py::array_t<double, py::array::c_style | py::array::forcecast> &next_states) const {
        if (states.ndim() != 2 || states.shape(1) != state_size_ || next_states.ndim() != 2 ||
            next_states.shape(1) != state_size_) {
            throw std::invalid_argument("states and next_states must have shape (N, state_size)");
        }
        // ``zip`` in the reference stops at the shorter of the two.
        const py::ssize_t n = std::min(states.shape(0), next_states.shape(0));
        py::array_t<double> out(n);
        double *dst = out.mutable_data();
        for (py::ssize_t r = 0; r < n; ++r) {
            dst[r] = reward_one(states.data() + r * state_size_, action,
                                next_states.data() + r * state_size_);
        }
        return out;
    }

    // ``Environment.sample_next_step``: successor, observation and reward in
    // one call.
    py::tuple sample_next_step(const StateArray &state, long long action) const {
        const double *s = checked_state(state, "state");
        auto &rng = pomdp_native::default_rng();
        Scratch scratch;
        py::array_t<double> next_state(static_cast<py::ssize_t>(state_size_));
        double *ns = next_state.mutable_data();
        draw_successor(s, action, rng, scratch, ns);
        std::vector<double> buf(static_cast<std::size_t>(observation_size_));
        sample_observation_into(ns, action, rng, buf.data());
        const double r = reward_one(s, action, ns);
        return py::make_tuple(next_state, make_observation(buf.data()), r);
    }

    // Random rollout with pre-drawn joint actions. Stops at a terminal state
    // or after ``len(action_indices)`` steps, and returns the discounted sum
    // ``sum_k discount**k * r_k``.
    double simulate_rollout(const StateArray &state,
                            const py::array_t<std::int64_t, py::array::c_style |
                                                                py::array::forcecast> &actions,
                            double discount_factor) const {
        const double *s = checked_state(state, "state");
        std::vector<double> current(s, s + state_size_);
        std::vector<double> next(static_cast<std::size_t>(state_size_));
        auto &rng = pomdp_native::default_rng();
        Scratch scratch;
        const std::int64_t *acts = actions.data();
        const py::ssize_t steps = actions.size();
        double total = 0.0;
        double discount = 1.0;
        for (py::ssize_t k = 0; k < steps; ++k) {
            if (is_terminal(current.data())) {
                break;
            }
            draw_successor(current.data(), acts[k], rng, scratch, next.data());
            total += discount * reward_one(current.data(), acts[k], next.data());
            discount *= discount_factor;
            current.swap(next);
        }
        return total;
    }

  private:
    int width_;
    int height_;
    int midline_;
    int n_blue_;
    int n_red_;
    int n_red_defenders_;
    Cell blue_base_;
    Cell red_base_;
    Cell blue_flag_cell_;
    double slip_;
    double range_error_;
    double pursuit_;
    int alert_radius_;
    int freeze_steps_;
    int cooldown_steps_;
    double half_move_;
    double half_scan_;
    int score_to_win_;
    double capture_reward_;
    double concede_penalty_;
    double tagged_penalty_;
    double tag_reward_;
    double pickup_reward_;
    double move_cost_;
    double scan_cost_;

    std::vector<std::uint8_t> tree_mask_;
    std::vector<Cell> candidates_;
    std::vector<Cell> guard_posts_;

    int red_pos_ = 0;
    int flag_cell_ = 0;
    int carrier_red_flag_ = 0;
    int carrier_blue_flag_ = 0;
    int freeze_blue_ = 0;
    int freeze_red_ = 0;
    int cooldown_blue_ = 0;
    int cooldown_red_ = 0;
    int score_blue_ = 0;
    int score_red_ = 0;
    int state_size_ = 0;
    int observation_size_ = 0;
    int max_range_ = 0;
    long long n_actions_ = 1;
};

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "CaptureTheFlag POMDP native C++ kernels (pomdp_native).";

    m.def(
        "set_seed", [](std::uint64_t seed) { pomdp_native::set_default_seed(seed); },
        py::arg("seed"), "Seed the module-local RNG used by every sampling entry point.");

    m.attr("MAX_PLAYERS_PER_TEAM") = kMaxPlayers;

    using Model = CaptureTheFlagModelCpp;
    py::class_<Model>(m, "CaptureTheFlagModelCpp")
        .def(py::init<int, int, int, const std::vector<std::pair<int, int>> &, int, int, int,
                      std::pair<int, int>, std::pair<int, int>, std::pair<int, int>,
                      const std::vector<std::pair<int, int>> &, double, double, double, int, int,
                      int, double, double, int, double, double, double, double, double, double,
                      double>(),
             py::arg("width"), py::arg("height"), py::arg("midline"), py::arg("trees"),
             py::arg("n_blue"), py::arg("n_red"), py::arg("n_red_defenders"),
             py::arg("blue_base"), py::arg("red_base"), py::arg("blue_flag_cell"),
             py::arg("red_flag_candidates"), py::arg("slip_probability"),
             py::arg("range_error_probability"), py::arg("red_pursuit_probability"),
             py::arg("red_alert_radius"), py::arg("freeze_steps"),
             py::arg("tagger_cooldown_steps"), py::arg("detector_half_distance_move"),
             py::arg("detector_half_distance_scan"), py::arg("score_to_win"),
             py::arg("capture_reward"), py::arg("concede_penalty"), py::arg("tagged_penalty"),
             py::arg("tag_reward"), py::arg("pickup_reward"), py::arg("move_cost"),
             py::arg("scan_cost"))
        .def_property_readonly("state_size", &Model::state_size)
        .def_property_readonly("observation_size", &Model::observation_size)
        .def_property_readonly("n_actions", &Model::n_actions)
        .def("is_terminal",
             [](const Model &self, const Model::StateArray &state) {
                 return self.is_terminal(self.checked_state(state, "state"));
             },
             py::arg("state"))
        .def("sample_next_state", &Model::sample_next_state, py::arg("state"), py::arg("action"),
             py::arg("n_samples") = 1)
        .def("batch_sample", &Model::batch_sample, py::arg("states"), py::arg("action"))
        .def("transition_probability", &Model::transition_probability, py::arg("state"),
             py::arg("action"), py::arg("next_states"))
        .def("successor_distribution", &Model::successor_distribution_py, py::arg("state"),
             py::arg("action"))
        .def("sample_observation", &Model::sample_observation, py::arg("next_state"),
             py::arg("action"), py::arg("n_samples") = 1)
        .def("observation_probability", &Model::observation_probability, py::arg("next_state"),
             py::arg("action"), py::arg("observations"))
        .def("batch_observation_probability", &Model::batch_observation_probability,
             py::arg("next_states"), py::arg("action"), py::arg("observation"))
        .def("reward", &Model::reward, py::arg("state"), py::arg("action"),
             py::arg("next_state"))
        .def("reward_batch", &Model::reward_batch, py::arg("states"), py::arg("action"),
             py::arg("next_states"))
        .def("sample_next_step", &Model::sample_next_step, py::arg("state"), py::arg("action"))
        .def("simulate_rollout", &Model::simulate_rollout, py::arg("state"),
             py::arg("action_indices"), py::arg("discount_factor"));
}
