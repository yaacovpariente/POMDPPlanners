// PacMan POMDP native extension.
//
// Mirrors the structural pattern used by the continuous LaserTag port: state
// dimension is env-parameterized (depends on num_ghosts and num_pellets), so
// this extension does NOT inherit from pomdp_native::TransitionModelCpp<Dim>.
// It composes the module-local RNG (pomdp_native/rng.hpp) and the python/numpy
// marshalling helpers (pomdp_native/marshalling.hpp) directly.
//
// State layout (float64, row-major):
//   [pac_row, pac_col, g0_row, g0_col, ..., pellet_mask[0..P-1], score, terminal]
// Built and read from Python via PacManPOMDP.make_state / get_* methods; the
// C++ side works off explicit index constants passed to each class's ctor.

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pomdp_native/marshalling.hpp"
#include "pomdp_native/rng.hpp"

namespace py = pybind11;

namespace {

constexpr int kNumMoves = 5;  // N, E, S, W, Stay  (matches pacman_grid_utils)

enum class GhostCoord : int {
    Independent = 0,
    Coordinated = 1,
    Mixed = 2,
};

enum class GhostStrategy : int {
    Aggressive = 0,
    Patrol = 1,
    Ambush = 2,
};

// Non-owning view into a 4-D int32 neighbor table of shape (R, C, 5, 2).
// Stored as a flat buffer with strides; the Python-side array is contiguous
// so strides come straight from shape (rows * cols * NUM_MOVES * 2 step=1).
struct NeighborTable {
    const std::int32_t *data;
    int rows;
    int cols;
    // Flat index into the (rows, cols, 5, 2) buffer.
    inline std::int32_t row_of(int r, int c, int move) const {
        return data[((r * cols + c) * kNumMoves + move) * 2];
    }
    inline std::int32_t col_of(int r, int c, int move) const {
        return data[((r * cols + c) * kNumMoves + move) * 2 + 1];
    }
};

struct NeighborValidity {
    const std::uint8_t *data;  // bool stored as uint8
    int rows;
    int cols;
    inline bool valid(int r, int c, int move) const {
        return data[(r * cols + c) * kNumMoves + move] != 0;
    }
};

struct PelletTable {
    const std::int32_t *data;  // (P, 2) row-major
    int num_pellets;
    inline std::int32_t row(int p) const { return data[p * 2]; }
    inline std::int32_t col(int p) const { return data[p * 2 + 1]; }
    // Returns -1 if no registered pellet matches.
    inline int index_of(int r, int c) const {
        for (int p = 0; p < num_pellets; ++p) {
            if (data[p * 2] == r && data[p * 2 + 1] == c) {
                return p;
            }
        }
        return -1;
    }
};

// ---------------------------------------------------------------------------
// EnvParams: immutable per-instance parameters owned by the transition class.
// ---------------------------------------------------------------------------
struct TransitionEnv {
    int maze_rows;
    int maze_cols;
    int num_ghosts;
    int num_pellets;
    int state_dim;
    double ghost_aggressiveness;
    double pellet_reward;
    GhostCoord ghost_coordination;
    std::vector<GhostStrategy> ghost_strategies;  // size num_ghosts
    NeighborTable neighbor_table;
    NeighborValidity neighbor_validity;
    PelletTable pellet_positions;
    int idx_pac_row;
    int idx_pac_col;
    int idx_ghosts_start;
    int idx_pellets_start;
    int idx_pellets_end;
    int idx_score;
    int idx_terminal;
    // Mutable int32 buffer (num_ghosts,): patrol direction index per ghost.
    // Passed in from the env; mutated in place to match the Python behavior.
    std::int32_t *patrol_dir_state;
};

// ---------------------------------------------------------------------------
// Helpers: valid moves, manhattan, argmax, softmax sampling.
// ---------------------------------------------------------------------------

// Collect indices of valid moves (0..4) from a given (r, c); returns count.
inline int collect_valid_moves(const TransitionEnv &env, int r, int c, int out_moves[kNumMoves]) {
    int count = 0;
    for (int m = 0; m < kNumMoves; ++m) {
        if (env.neighbor_validity.valid(r, c, m)) {
            out_moves[count++] = m;
        }
    }
    return count;
}

inline int manhattan(int r1, int c1, int r2, int c2) {
    return std::abs(r1 - r2) + std::abs(c1 - c2);
}

// Sample an index from the provided move list by softmax over distances to
// the target (closer = better; negative distance / temperature). Mirrors
// Python `_independent_ghost_move_probability` for the aggressive branch.
inline int softmax_move_toward(const TransitionEnv &env, const int moves[], int n, int ghost_r,
                               int ghost_c, int target_r, int target_c, std::mt19937_64 &rng) {
    double scores[kNumMoves];
    double max_score = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        const int m = moves[i];
        const int nr = env.neighbor_table.row_of(ghost_r, ghost_c, m);
        const int nc = env.neighbor_table.col_of(ghost_r, ghost_c, m);
        scores[i] = -static_cast<double>(manhattan(nr, nc, target_r, target_c)) /
                    env.ghost_aggressiveness;
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        scores[i] = std::exp(scores[i] - max_score);
        sum += scores[i];
    }
    std::uniform_real_distribution<double> unif(0.0, sum);
    double u = unif(rng);
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += scores[i];
        if (u < acc) {
            return i;
        }
    }
    return n - 1;
}

// Compute the full softmax probability distribution of moves for the
// aggressive branch. Returns move distribution pd[n] normalized.
inline void softmax_move_probabilities(const TransitionEnv &env, const int moves[], int n,
                                       int ghost_r, int ghost_c, int pacman_r, int pacman_c,
                                       double out[kNumMoves]) {
    double max_score = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        const int m = moves[i];
        const int nr = env.neighbor_table.row_of(ghost_r, ghost_c, m);
        const int nc = env.neighbor_table.col_of(ghost_r, ghost_c, m);
        out[i] = -static_cast<double>(manhattan(nr, nc, pacman_r, pacman_c)) /
                 env.ghost_aggressiveness;
        if (out[i] > max_score) {
            max_score = out[i];
        }
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        out[i] = std::exp(out[i] - max_score);
        sum += out[i];
    }
    for (int i = 0; i < n; ++i) {
        out[i] /= sum;
    }
}

// ---------------------------------------------------------------------------
// Ghost movement — per-strategy deterministic or stochastic move.
// Return value is (new_r, new_c) packed into a pair.
// ---------------------------------------------------------------------------

struct Pos {
    int r;
    int c;
};

// Uniform-random from the valid moves list.
inline Pos uniform_random_move(const TransitionEnv &env, int r, int c, const int moves[], int n,
                               std::mt19937_64 &rng) {
    std::uniform_int_distribution<int> pick(0, n - 1);
    const int m = moves[pick(rng)];
    return {env.neighbor_table.row_of(r, c, m), env.neighbor_table.col_of(r, c, m)};
}

// Aggressive: softmax-sample toward pacman_pos (matches Python `_move_single_ghost`
// default branch).
inline Pos move_aggressive(const TransitionEnv &env, int ghost_r, int ghost_c, int pacman_r,
                           int pacman_c, std::mt19937_64 &rng) {
    int moves[kNumMoves];
    const int n = collect_valid_moves(env, ghost_r, ghost_c, moves);
    if (n == 0) {
        return {ghost_r, ghost_c};
    }
    const int i = softmax_move_toward(env, moves, n, ghost_r, ghost_c, pacman_r, pacman_c, rng);
    const int m = moves[i];
    return {env.neighbor_table.row_of(ghost_r, ghost_c, m),
            env.neighbor_table.col_of(ghost_r, ghost_c, m)};
}

// Patrol: walk in current direction if valid, else rotate CW by one and pick
// uniformly from valid moves. Mutates `patrol_dir_state[ghost_id]`.
inline Pos move_patrol(const TransitionEnv &env, int ghost_r, int ghost_c, int ghost_id,
                       std::mt19937_64 &rng) {
    // Python direction list: [(0, -1), (1, 0), (0, 1), (-1, 0)]  (N, E, S, W by that mapping)
    // Translating: current_dir advances (dr, dc) from ghost_pos; if
    // (ghost_r + dr, ghost_c + dc) is in the possible_moves list we take it;
    // else we rotate to (current_dir + 1) % 4 and uniform-sample.
    static constexpr int kPatrolDR[4] = {0, 1, 0, -1};
    static constexpr int kPatrolDC[4] = {-1, 0, 1, 0};
    int &dir_ref = env.patrol_dir_state[ghost_id];
    const int dr = kPatrolDR[dir_ref];
    const int dc = kPatrolDC[dir_ref];
    const int target_r = ghost_r + dr;
    const int target_c = ghost_c + dc;
    int moves[kNumMoves];
    const int n = collect_valid_moves(env, ghost_r, ghost_c, moves);
    if (n == 0) {
        return {ghost_r, ghost_c};
    }
    // Does the target_r, target_c coincide with one of the valid-move results?
    for (int i = 0; i < n; ++i) {
        const int m = moves[i];
        const int nr = env.neighbor_table.row_of(ghost_r, ghost_c, m);
        const int nc = env.neighbor_table.col_of(ghost_r, ghost_c, m);
        if (nr == target_r && nc == target_c) {
            return {nr, nc};
        }
    }
    dir_ref = (dir_ref + 1) % 4;
    return uniform_random_move(env, ghost_r, ghost_c, moves, n, rng);
}

// Ambush: deterministic argmin over a piecewise score penalizing distances
// outside [2, 4] by +10 (Python `_move_ambush_ghost`).
inline Pos move_ambush(const TransitionEnv &env, int ghost_r, int ghost_c, int pacman_r,
                       int pacman_c) {
    int moves[kNumMoves];
    const int n = collect_valid_moves(env, ghost_r, ghost_c, moves);
    if (n == 0) {
        return {ghost_r, ghost_c};
    }
    double best_score = std::numeric_limits<double>::infinity();
    int best_r = ghost_r;
    int best_c = ghost_c;
    for (int i = 0; i < n; ++i) {
        const int m = moves[i];
        const int nr = env.neighbor_table.row_of(ghost_r, ghost_c, m);
        const int nc = env.neighbor_table.col_of(ghost_r, ghost_c, m);
        const int dist = manhattan(nr, nc, pacman_r, pacman_c);
        const double score =
            (dist >= 2 && dist <= 4) ? static_cast<double>(dist) : static_cast<double>(dist + 10);
        if (score < best_score) {
            best_score = score;
            best_r = nr;
            best_c = nc;
        }
    }
    return {best_r, best_c};
}

// Move-toward-target: deterministic argmin over manhattan distance.
inline Pos move_toward_target(const TransitionEnv &env, int ghost_r, int ghost_c, int target_r,
                              int target_c) {
    int moves[kNumMoves];
    const int n = collect_valid_moves(env, ghost_r, ghost_c, moves);
    if (n == 0) {
        return {ghost_r, ghost_c};
    }
    int best_r = ghost_r;
    int best_c = ghost_c;
    int best_dist = std::numeric_limits<int>::max();
    for (int i = 0; i < n; ++i) {
        const int m = moves[i];
        const int nr = env.neighbor_table.row_of(ghost_r, ghost_c, m);
        const int nc = env.neighbor_table.col_of(ghost_r, ghost_c, m);
        const int dist = manhattan(nr, nc, target_r, target_c);
        if (dist < best_dist) {
            best_dist = dist;
            best_r = nr;
            best_c = nc;
        }
    }
    return {best_r, best_c};
}

// Predict where PacMan is likely to escape: argmax over pacman's 4 non-stay
// moves of the minimum distance to any ghost other than `current_ghost_id`.
inline Pos predict_pacman_escape_route(const TransitionEnv &env, int pacman_r, int pacman_c,
                                       const std::vector<int> &ghost_rs,
                                       const std::vector<int> &ghost_cs, int current_ghost_id) {
    int best_r = pacman_r;
    int best_c = pacman_c;
    int best_min_dist = -1;
    // Iterate 4 non-stay moves (0..3) — the Python helper excludes Stay.
    for (int m = 0; m < 4; ++m) {
        if (!env.neighbor_validity.valid(pacman_r, pacman_c, m)) {
            continue;
        }
        const int nr = env.neighbor_table.row_of(pacman_r, pacman_c, m);
        const int nc = env.neighbor_table.col_of(pacman_r, pacman_c, m);
        int min_dist = std::numeric_limits<int>::max();
        for (std::size_t i = 0; i < ghost_rs.size(); ++i) {
            if (static_cast<int>(i) == current_ghost_id) {
                continue;
            }
            const int d = manhattan(nr, nc, ghost_rs[i], ghost_cs[i]);
            if (d < min_dist) {
                min_dist = d;
            }
        }
        if (min_dist > best_min_dist) {
            best_min_dist = min_dist;
            best_r = nr;
            best_c = nc;
        }
    }
    return {best_r, best_c};
}

// Coordinated: ghost_id==0 chases pacman directly; other ghosts move toward
// the predicted escape position (deterministic).
inline Pos move_coordinated(const TransitionEnv &env, int ghost_r, int ghost_c, int ghost_id,
                            int pacman_r, int pacman_c, const std::vector<int> &all_ghost_rs,
                            const std::vector<int> &all_ghost_cs) {
    int target_r;
    int target_c;
    if (ghost_id == 0) {
        target_r = pacman_r;
        target_c = pacman_c;
    } else {
        const Pos target =
            predict_pacman_escape_route(env, pacman_r, pacman_c, all_ghost_rs, all_ghost_cs,
                                        ghost_id);
        target_r = target.r;
        target_c = target.c;
    }
    return move_toward_target(env, ghost_r, ghost_c, target_r, target_c);
}

// Apply a ghost's movement strategy (respecting coordination mode).
inline Pos move_one_ghost(const TransitionEnv &env, int ghost_id, int ghost_r, int ghost_c,
                          int pacman_r, int pacman_c, const std::vector<int> &all_ghost_rs,
                          const std::vector<int> &all_ghost_cs, std::mt19937_64 &rng) {
    const GhostCoord coord = env.ghost_coordination;
    const bool use_coord_branch =
        (coord == GhostCoord::Coordinated) ||
        (coord == GhostCoord::Mixed && (ghost_id % 2 == 0));
    if (use_coord_branch) {
        return move_coordinated(env, ghost_r, ghost_c, ghost_id, pacman_r, pacman_c, all_ghost_rs,
                                all_ghost_cs);
    }
    // Independent branch — dispatch on per-ghost strategy.
    const GhostStrategy strat = env.ghost_strategies[static_cast<std::size_t>(ghost_id)];
    switch (strat) {
        case GhostStrategy::Patrol:
            return move_patrol(env, ghost_r, ghost_c, ghost_id, rng);
        case GhostStrategy::Ambush:
            return move_ambush(env, ghost_r, ghost_c, pacman_r, pacman_c);
        case GhostStrategy::Aggressive:
        default:
            return move_aggressive(env, ghost_r, ghost_c, pacman_r, pacman_c, rng);
    }
}

// ---------------------------------------------------------------------------
// Transition core: compute next state array given current state + action.
// Mutates `next_state` in place (caller supplies a copy of the input state).
// ---------------------------------------------------------------------------
void apply_transition(const TransitionEnv &env, int action, double *next_state,
                      std::mt19937_64 &rng) {
    if (next_state[env.idx_terminal] > 0.5) {
        return;  // Terminal is absorbing.
    }
    const int pac_r = static_cast<int>(next_state[env.idx_pac_row]);
    const int pac_c = static_cast<int>(next_state[env.idx_pac_col]);

    // Move pacman via precomputed neighbor table (invalid moves keep position).
    int new_pac_r;
    int new_pac_c;
    if (action >= 0 && action < kNumMoves) {
        new_pac_r = env.neighbor_table.row_of(pac_r, pac_c, action);
        new_pac_c = env.neighbor_table.col_of(pac_r, pac_c, action);
    } else {
        new_pac_r = pac_r;
        new_pac_c = pac_c;
    }

    // Collect current ghost positions (needed for coordinated/mixed).
    std::vector<int> ghost_rs(static_cast<std::size_t>(env.num_ghosts));
    std::vector<int> ghost_cs(static_cast<std::size_t>(env.num_ghosts));
    for (int g = 0; g < env.num_ghosts; ++g) {
        ghost_rs[g] = static_cast<int>(next_state[env.idx_ghosts_start + 2 * g]);
        ghost_cs[g] = static_cast<int>(next_state[env.idx_ghosts_start + 2 * g + 1]);
    }

    // Move each ghost and write back to the state array.
    for (int g = 0; g < env.num_ghosts; ++g) {
        const Pos next = move_one_ghost(env, g, ghost_rs[g], ghost_cs[g], pac_r, pac_c, ghost_rs,
                                        ghost_cs, rng);
        next_state[env.idx_ghosts_start + 2 * g] = static_cast<double>(next.r);
        next_state[env.idx_ghosts_start + 2 * g + 1] = static_cast<double>(next.c);
    }

    // Pacman pos update.
    next_state[env.idx_pac_row] = static_cast<double>(new_pac_r);
    next_state[env.idx_pac_col] = static_cast<double>(new_pac_c);

    // Collision check — terminal if pacman is at any ghost position.
    for (int g = 0; g < env.num_ghosts; ++g) {
        const int gr = static_cast<int>(next_state[env.idx_ghosts_start + 2 * g]);
        const int gc = static_cast<int>(next_state[env.idx_ghosts_start + 2 * g + 1]);
        if (gr == new_pac_r && gc == new_pac_c) {
            next_state[env.idx_terminal] = 1.0;
            return;
        }
    }

    // Pellet collection.
    const int pellet_idx = env.pellet_positions.index_of(new_pac_r, new_pac_c);
    if (pellet_idx >= 0 && next_state[env.idx_pellets_start + pellet_idx] > 0.5) {
        next_state[env.idx_pellets_start + pellet_idx] = 0.0;
        next_state[env.idx_score] += env.pellet_reward;
    }

    // Terminal if no pellets remain (unchanged by collection branch — matches
    // ndarray-state refactor's unconditional post-move check).
    bool any_active = false;
    for (int p = 0; p < env.num_pellets; ++p) {
        if (next_state[env.idx_pellets_start + p] > 0.5) {
            any_active = true;
            break;
        }
    }
    if (!any_active) {
        next_state[env.idx_terminal] = 1.0;
    }
}

// ---------------------------------------------------------------------------
// Transition probability helpers (exact Python parity).
// ---------------------------------------------------------------------------

// Probability of a single ghost moving from current to target under its
// assigned strategy (Python `_single_ghost_move_probability`).
double single_ghost_move_probability(const TransitionEnv &env, int ghost_id, int cur_r, int cur_c,
                                     int tgt_r, int tgt_c, int pacman_r, int pacman_c) {
    int moves[kNumMoves];
    const int n = collect_valid_moves(env, cur_r, cur_c, moves);
    if (n == 0) {
        // No valid moves — probability concentrated on (cur_r, cur_c) if
        // target equals current.
        return (tgt_r == cur_r && tgt_c == cur_c) ? 1.0 : 0.0;
    }
    int target_idx = -1;
    for (int i = 0; i < n; ++i) {
        const int m = moves[i];
        const int nr = env.neighbor_table.row_of(cur_r, cur_c, m);
        const int nc = env.neighbor_table.col_of(cur_r, cur_c, m);
        if (nr == tgt_r && nc == tgt_c) {
            target_idx = i;
            break;
        }
    }
    if (target_idx < 0) {
        return 0.0;
    }
    // Dispatch on coordination mode — Python's `_single_ghost_move_probability`.
    const GhostCoord coord = env.ghost_coordination;
    const bool use_coord_branch =
        (coord == GhostCoord::Coordinated) ||
        (coord == GhostCoord::Mixed && (ghost_id % 2 == 0));
    if (use_coord_branch) {
        // `_coordinated_ghost_move_probability` — uniform over possible moves.
        return 1.0 / static_cast<double>(n);
    }
    // Independent branch.
    const GhostStrategy strat = env.ghost_strategies[static_cast<std::size_t>(ghost_id)];
    if (strat == GhostStrategy::Patrol || strat == GhostStrategy::Ambush) {
        return 1.0 / static_cast<double>(n);
    }
    // Aggressive — softmax over Manhattan distance to pacman.
    double probs[kNumMoves];
    softmax_move_probabilities(env, moves, n, cur_r, cur_c, pacman_r, pacman_c, probs);
    return probs[target_idx];
}

// ---------------------------------------------------------------------------
// PacManTransitionCpp — the Python-facing class.
// ---------------------------------------------------------------------------
class PacManTransitionCpp {
  public:
    PacManTransitionCpp(py::array_t<double> state, int action, int maze_rows, int maze_cols,
                        py::array_t<std::int32_t> neighbor_table,
                        py::array_t<std::uint8_t> neighbor_validity,
                        py::array_t<std::int32_t> pellet_positions, double ghost_aggressiveness,
                        int ghost_coordination_code,
                        py::array_t<std::int32_t> ghost_strategy_codes, int num_ghosts,
                        int num_pellets, double pellet_reward, int idx_pac_row, int idx_pac_col,
                        int idx_ghosts_start, int idx_pellets_start, int idx_pellets_end,
                        int idx_score, int idx_terminal,
                        py::array_t<std::int32_t> patrol_dir_state)
        : state_array_(state),
          action_(action),
          neighbor_table_array_(neighbor_table),
          neighbor_validity_array_(neighbor_validity),
          pellet_positions_array_(pellet_positions),
          ghost_strategy_codes_array_(ghost_strategy_codes),
          patrol_dir_state_array_(patrol_dir_state) {
        // Unpack neighbor_table shape (R, C, 5, 2).
        if (neighbor_table.ndim() != 4 || neighbor_table.shape(0) != maze_rows ||
            neighbor_table.shape(1) != maze_cols || neighbor_table.shape(2) != kNumMoves ||
            neighbor_table.shape(3) != 2) {
            throw std::invalid_argument("neighbor_table must have shape (R, C, 5, 2)");
        }
        if (neighbor_validity.ndim() != 3 || neighbor_validity.shape(0) != maze_rows ||
            neighbor_validity.shape(1) != maze_cols || neighbor_validity.shape(2) != kNumMoves) {
            throw std::invalid_argument("neighbor_validity must have shape (R, C, 5)");
        }
        if (pellet_positions.ndim() != 2 || pellet_positions.shape(0) != num_pellets ||
            pellet_positions.shape(1) != 2) {
            throw std::invalid_argument("pellet_positions must have shape (P, 2)");
        }
        if (ghost_strategy_codes.ndim() != 1 || ghost_strategy_codes.shape(0) != num_ghosts) {
            throw std::invalid_argument("ghost_strategy_codes must have shape (num_ghosts,)");
        }
        if (patrol_dir_state.ndim() != 1 || patrol_dir_state.shape(0) != num_ghosts) {
            throw std::invalid_argument("patrol_dir_state must have shape (num_ghosts,)");
        }
        if (state.ndim() != 1) {
            throw std::invalid_argument("state must be 1-D");
        }

        env_.maze_rows = maze_rows;
        env_.maze_cols = maze_cols;
        env_.num_ghosts = num_ghosts;
        env_.num_pellets = num_pellets;
        env_.state_dim = static_cast<int>(state.shape(0));
        env_.ghost_aggressiveness = ghost_aggressiveness;
        env_.pellet_reward = pellet_reward;
        env_.ghost_coordination = static_cast<GhostCoord>(ghost_coordination_code);
        env_.neighbor_table = {neighbor_table.data(), maze_rows, maze_cols};
        env_.neighbor_validity = {neighbor_validity.data(), maze_rows, maze_cols};
        env_.pellet_positions = {pellet_positions.data(), num_pellets};
        env_.idx_pac_row = idx_pac_row;
        env_.idx_pac_col = idx_pac_col;
        env_.idx_ghosts_start = idx_ghosts_start;
        env_.idx_pellets_start = idx_pellets_start;
        env_.idx_pellets_end = idx_pellets_end;
        env_.idx_score = idx_score;
        env_.idx_terminal = idx_terminal;

        env_.ghost_strategies.resize(static_cast<std::size_t>(num_ghosts));
        auto codes = ghost_strategy_codes.unchecked<1>();
        for (int g = 0; g < num_ghosts; ++g) {
            env_.ghost_strategies[g] = static_cast<GhostStrategy>(codes(g));
        }
        env_.patrol_dir_state = patrol_dir_state.mutable_data();
    }

    // sample(n_samples=1) -> List[np.ndarray(state_dim,)]
    py::list sample(int n_samples) const {
        py::list out;
        // Reference base = a copy of the input state, transitioned once.
        auto base = state_copy_as_array();
        auto buf = base.mutable_data();
        apply_transition(env_, action_, buf, pomdp_native::default_rng().engine());
        out.append(base);
        for (int i = 1; i < n_samples; ++i) {
            auto extra = state_copy_as_array();
            auto eb = extra.mutable_data();
            apply_transition(env_, action_, eb, pomdp_native::default_rng().engine());
            out.append(extra);
        }
        return out;
    }

    // probability(values) -> np.ndarray[float64]
    // Ported from Python's _transition_probability_for_candidate loop.
    py::array_t<double> probability(py::object values) const {
        // Read current state cheaply.
        auto state_u = state_array_.unchecked<1>();
        const bool terminal = state_u(env_.idx_terminal) > 0.5;
        const int pac_r = static_cast<int>(state_u(env_.idx_pac_row));
        const int pac_c = static_cast<int>(state_u(env_.idx_pac_col));

        // Extract candidates as row-major batch.
        auto batch = pomdp_native::extract_rows_nd(values, static_cast<std::size_t>(env_.state_dim));
        const std::size_t n = batch.n;
        auto out = py::array_t<double>(static_cast<py::ssize_t>(n));
        auto obuf = out.mutable_unchecked<1>();

        if (terminal) {
            // Only identity transitions have non-zero probability.
            std::vector<double> current(env_.state_dim);
            for (int d = 0; d < env_.state_dim; ++d) {
                current[d] = state_u(d);
            }
            for (std::size_t i = 0; i < n; ++i) {
                bool equal = true;
                for (int d = 0; d < env_.state_dim; ++d) {
                    if (batch.flat[i * env_.state_dim + d] != current[d]) {
                        equal = false;
                        break;
                    }
                }
                obuf(static_cast<py::ssize_t>(i)) = equal ? 1.0 : 0.0;
            }
            return out;
        }

        // Determine pacman_next_pos under the current action.
        int new_pac_r = pac_r;
        int new_pac_c = pac_c;
        if (action_ >= 0 && action_ < kNumMoves) {
            new_pac_r = env_.neighbor_table.row_of(pac_r, pac_c, action_);
            new_pac_c = env_.neighbor_table.col_of(pac_r, pac_c, action_);
        }

        // Current pellet set + score for pellet/score validity.
        std::vector<bool> cur_active(static_cast<std::size_t>(env_.num_pellets), false);
        for (int p = 0; p < env_.num_pellets; ++p) {
            cur_active[p] = state_u(env_.idx_pellets_start + p) > 0.5;
        }
        const double cur_score = state_u(env_.idx_score);

        // Score + pellet mask after pacman moves (before ghost movement).
        double expected_score = cur_score;
        std::vector<bool> expected_active = cur_active;
        const int collected_pellet_idx = env_.pellet_positions.index_of(new_pac_r, new_pac_c);
        bool pellet_collected_in_expected = false;
        if (collected_pellet_idx >= 0 && cur_active[collected_pellet_idx]) {
            expected_active[collected_pellet_idx] = false;
            expected_score += env_.pellet_reward;
            pellet_collected_in_expected = true;
        }

        double total = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double *row = &batch.flat[i * env_.state_dim];
            // 1. Pacman pos check.
            if (static_cast<int>(row[env_.idx_pac_row]) != new_pac_r ||
                static_cast<int>(row[env_.idx_pac_col]) != new_pac_c) {
                obuf(static_cast<py::ssize_t>(i)) = 0.0;
                continue;
            }
            // 2. Ghost transition probability (product over ghosts).
            double ghost_prob = 1.0;
            for (int g = 0; g < env_.num_ghosts; ++g) {
                const int cur_gr = static_cast<int>(state_u(env_.idx_ghosts_start + 2 * g));
                const int cur_gc = static_cast<int>(state_u(env_.idx_ghosts_start + 2 * g + 1));
                const int tgt_gr = static_cast<int>(row[env_.idx_ghosts_start + 2 * g]);
                const int tgt_gc = static_cast<int>(row[env_.idx_ghosts_start + 2 * g + 1]);
                ghost_prob *= single_ghost_move_probability(env_, g, cur_gr, cur_gc, tgt_gr, tgt_gc,
                                                             pac_r, pac_c);
                if (ghost_prob == 0.0) {
                    break;
                }
            }
            if (ghost_prob == 0.0) {
                obuf(static_cast<py::ssize_t>(i)) = 0.0;
                continue;
            }
            // 3. Pellet configuration validity.
            bool pellet_ok = true;
            for (int p = 0; p < env_.num_pellets; ++p) {
                const bool cand_active = row[env_.idx_pellets_start + p] > 0.5;
                if (cand_active != expected_active[p]) {
                    pellet_ok = false;
                    break;
                }
            }
            if (!pellet_ok || row[env_.idx_score] != expected_score) {
                obuf(static_cast<py::ssize_t>(i)) = 0.0;
                continue;
            }
            // 4. Terminal validity — collision OR all pellets collected.
            bool collision = false;
            for (int g = 0; g < env_.num_ghosts; ++g) {
                const int tgt_gr = static_cast<int>(row[env_.idx_ghosts_start + 2 * g]);
                const int tgt_gc = static_cast<int>(row[env_.idx_ghosts_start + 2 * g + 1]);
                if (tgt_gr == new_pac_r && tgt_gc == new_pac_c) {
                    collision = true;
                    break;
                }
            }
            bool any_active = false;
            for (int p = 0; p < env_.num_pellets; ++p) {
                if (row[env_.idx_pellets_start + p] > 0.5) {
                    any_active = true;
                    break;
                }
            }
            const bool all_pellets_collected = !any_active;
            const bool expected_terminal = collision || all_pellets_collected;
            const bool cand_terminal = row[env_.idx_terminal] > 0.5;
            if (cand_terminal != expected_terminal) {
                obuf(static_cast<py::ssize_t>(i)) = 0.0;
                continue;
            }
            (void)pellet_collected_in_expected;
            obuf(static_cast<py::ssize_t>(i)) = ghost_prob;
            total += ghost_prob;
        }
        // Normalize.
        if (total > 0.0) {
            for (std::size_t i = 0; i < n; ++i) {
                obuf(static_cast<py::ssize_t>(i)) /= total;
            }
        }
        return out;
    }

    // batch_sample(particles) -> np.ndarray[(N, state_dim), float64]
    py::array_t<double> batch_sample(py::array_t<double> particles) const {
        if (particles.ndim() != 2 || particles.shape(1) != env_.state_dim) {
            throw std::invalid_argument("particles must have shape (N, state_dim)");
        }
        const std::size_t n = static_cast<std::size_t>(particles.shape(0));
        auto out = py::array_t<double>({static_cast<py::ssize_t>(n),
                                        static_cast<py::ssize_t>(env_.state_dim)});
        auto src = particles.unchecked<2>();
        auto dst = out.mutable_unchecked<2>();
        auto &rng = pomdp_native::default_rng().engine();
        std::vector<double> scratch(static_cast<std::size_t>(env_.state_dim));
        for (std::size_t i = 0; i < n; ++i) {
            for (int d = 0; d < env_.state_dim; ++d) {
                scratch[d] = src(static_cast<py::ssize_t>(i), d);
            }
            apply_transition(env_, action_, scratch.data(), rng);
            for (int d = 0; d < env_.state_dim; ++d) {
                dst(static_cast<py::ssize_t>(i), d) = scratch[d];
            }
        }
        return out;
    }

    py::array_t<double> state_property() const { return state_array_; }
    int action_property() const { return action_; }

  private:
    py::array_t<double> state_copy_as_array() const {
        auto arr = py::array_t<double>(static_cast<py::ssize_t>(env_.state_dim));
        auto src = state_array_.unchecked<1>();
        auto dst = arr.mutable_unchecked<1>();
        for (int d = 0; d < env_.state_dim; ++d) {
            dst(static_cast<py::ssize_t>(d)) = src(static_cast<py::ssize_t>(d));
        }
        return arr;
    }

    // Hold the Python-owned arrays for lifetime management (views inside
    // env_ alias into their buffers).
    py::array_t<double> state_array_;
    int action_;
    py::array_t<std::int32_t> neighbor_table_array_;
    py::array_t<std::uint8_t> neighbor_validity_array_;
    py::array_t<std::int32_t> pellet_positions_array_;
    py::array_t<std::int32_t> ghost_strategy_codes_array_;
    py::array_t<std::int32_t> patrol_dir_state_array_;
    TransitionEnv env_{};
};

}  // anonymous namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "PacMan POMDP native C++ kernels (pomdp_native).";

    m.def(
        "set_seed",
        [](std::uint64_t seed) { pomdp_native::set_default_seed(seed); },
        py::arg("seed"),
        "Seed the module-local RNG used by sample()/batch entry points.");

    py::class_<PacManTransitionCpp>(m, "PacManTransitionCpp")
        .def(py::init<py::array_t<double>, int, int, int, py::array_t<std::int32_t>,
                      py::array_t<std::uint8_t>, py::array_t<std::int32_t>, double, int,
                      py::array_t<std::int32_t>, int, int, double, int, int, int, int, int, int,
                      int, py::array_t<std::int32_t>>(),
             py::arg("state"), py::arg("action"), py::arg("maze_rows"),
             py::arg("maze_cols"), py::arg("neighbor_table"), py::arg("neighbor_validity"),
             py::arg("pellet_positions"), py::arg("ghost_aggressiveness"),
             py::arg("ghost_coordination_code"), py::arg("ghost_strategy_codes"),
             py::arg("num_ghosts"), py::arg("num_pellets"), py::arg("pellet_reward"),
             py::arg("idx_pac_row"), py::arg("idx_pac_col"), py::arg("idx_ghosts_start"),
             py::arg("idx_pellets_start"), py::arg("idx_pellets_end"), py::arg("idx_score"),
             py::arg("idx_terminal"), py::arg("patrol_dir_state"))
        .def("sample", &PacManTransitionCpp::sample, py::arg("n_samples") = 1)
        .def("probability", &PacManTransitionCpp::probability, py::arg("values"))
        .def("batch_sample", &PacManTransitionCpp::batch_sample, py::arg("particles"))
        .def_property_readonly("state", &PacManTransitionCpp::state_property)
        .def_property_readonly("action", &PacManTransitionCpp::action_property);
}
