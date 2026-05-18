"""PacMan-specific Numba-JIT kernels.

Single-pass batched reward kernel that mirrors the numpy implementation in
:class:`PacManRewardModel.compute_reward_batch` but fuses the
terminal / collision / pellet / win / dangerous-area work into one loop
over rows. The numpy path issues ~15-20 separate operator calls per
invocation; folding them into one ``@njit`` kernel eliminates the
per-call dispatch overhead and the temporary allocations.

Conventions
-----------
- All array inputs are contiguous ``float64`` / ``int32`` / ``int64`` /
  ``uint8`` as declared on the kernel signature.
- ``next_states`` is always passed as a real array. When the caller has
  no realised next-state batch, pass a zero-row sentinel
  (``np.empty((0, state_dim), dtype=np.float64)``) and set
  ``has_next_states`` to ``False``; the kernel skips the collision
  contribution in that case.
- No Python objects, no RNG: reward is deterministic given inputs.
"""

import numpy as np
from numba import njit

# pylint: disable=no-value-for-parameter,not-an-iterable,too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches


@njit(cache=True)  # type: ignore[misc]
def compute_reward_batch_kernel(
    states: np.ndarray,
    action: int,
    next_states: np.ndarray,
    has_next_states: bool,
    neighbor_table: np.ndarray,
    pellet_positions: np.ndarray,
    dangerous_areas: np.ndarray,
    num_ghosts: int,
    step_penalty: float,
    ghost_collision_penalty: float,
    pellet_reward: float,
    win_reward: float,
    dangerous_area_penalty: float,
    dangerous_area_radius: float,
    idx_pac_row: int,
    idx_pac_col: int,
    idx_ghosts_start: int,
    idx_pellets_start: int,
    idx_terminal: int,
) -> np.ndarray:
    """Fused per-row reward kernel.

    Replicates the numpy implementation exactly: terminal -> 0.0,
    otherwise step_penalty + (optional) collision + dangerous-area +
    pellet + win-bonus contributions. The pellet-mask iteration bound
    is derived from ``pellet_positions.shape[0]``; no separate
    ``idx_pellets_end`` parameter is needed.
    """
    n_rows = states.shape[0]
    n_pellets = pellet_positions.shape[0]
    n_dangerous = dangerous_areas.shape[0]
    radius_sq = dangerous_area_radius * dangerous_area_radius
    rewards = np.zeros(n_rows, dtype=np.float64)

    for i in range(n_rows):
        if states[i, idx_terminal] > 0.5:
            continue

        reward = step_penalty
        pac_row = int(states[i, idx_pac_row])
        pac_col = int(states[i, idx_pac_col])
        new_pac_row = int(neighbor_table[pac_row, pac_col, action, 0])
        new_pac_col = int(neighbor_table[pac_row, pac_col, action, 1])

        # Collision detection mirrors the C++ rollout kernel
        # (_cpp/pacman.cpp lines 1435-1438): same-cell OR pacman-ghost
        # swap. The swap arm fires when a ghost previously at pacman's
        # new cell ends up at pacman's old cell — the transition writes
        # the ghost back to the old pacman position and sets terminal=1.
        # Only one penalty per step (break after first hit) to match C++.
        if has_next_states and num_ghosts > 0:
            for g in range(num_ghosts):
                g_row = int(next_states[i, idx_ghosts_start + 2 * g])
                g_col = int(next_states[i, idx_ghosts_start + 2 * g + 1])
                prev_g_row = int(states[i, idx_ghosts_start + 2 * g])
                prev_g_col = int(states[i, idx_ghosts_start + 2 * g + 1])
                same_cell = g_row == new_pac_row and g_col == new_pac_col
                swap = (
                    prev_g_row == new_pac_row
                    and prev_g_col == new_pac_col
                    and g_row == pac_row
                    and g_col == pac_col
                )
                if same_cell or swap:
                    reward += ghost_collision_penalty
                    break

        if n_dangerous > 0:
            for d in range(n_dangerous):
                dr = new_pac_row - dangerous_areas[d, 0]
                dc = new_pac_col - dangerous_areas[d, 1]
                if dr * dr + dc * dc <= radius_sq:
                    reward -= dangerous_area_penalty
                    break

        if n_pellets > 0:
            collected_idx = -1
            for p in range(n_pellets):
                if (
                    new_pac_row == pellet_positions[p, 0]
                    and new_pac_col == pellet_positions[p, 1]
                    and states[i, idx_pellets_start + p] > 0.5
                ):
                    collected_idx = p
                    break

            if collected_idx >= 0:
                reward += pellet_reward
                remaining = 0
                for p in range(n_pellets):
                    if p == collected_idx:
                        continue
                    if states[i, idx_pellets_start + p] > 0.5:
                        remaining += 1
                if remaining == 0:
                    reward += win_reward

        rewards[i] = reward

    return rewards


@njit(cache=True)  # type: ignore[misc]
def compute_reward_scalar_kernel(
    state: np.ndarray,
    next_state: np.ndarray,
    dangerous_areas: np.ndarray,
    num_ghosts: int,
    n_pellets: int,
    step_penalty: float,
    ghost_collision_penalty: float,
    pellet_reward: float,
    win_reward: float,
    dangerous_area_penalty: float,
    dangerous_area_radius: float,
    idx_pac_row: int,
    idx_pac_col: int,
    idx_ghosts_start: int,
    idx_pellets_start: int,
    idx_score: int,
    idx_terminal: int,
) -> float:
    """Single-row scalar reward kernel mirroring the Python implementation.

    Reads the realised pacman / ghost positions directly from
    ``next_state`` (matching the Python scalar — pacman moves
    deterministically, so the neighbor-table lookup the batch kernel
    uses is redundant here). Pellet pickup is detected via the
    ``next_state.score > state.score`` invariant the transition kernel
    upholds; win-bonus fires only when ``next_state`` is terminal with
    no remaining pellets.
    """
    if state[idx_terminal] > 0.5:
        return 0.0
    reward = step_penalty
    next_pac_row = int(next_state[idx_pac_row])
    next_pac_col = int(next_state[idx_pac_col])
    prev_pac_row = int(state[idx_pac_row])
    prev_pac_col = int(state[idx_pac_col])

    # Collision detection mirrors the C++ rollout kernel (_cpp/pacman.cpp
    # lines 1435-1438): same-cell OR pacman-ghost swap. The swap arm fires
    # when a ghost previously at pacman's new cell ends up on pacman's old
    # cell — the transition kernel then writes the ghost at the old pacman
    # position and sets terminal=1 (see apply_transition lines 454-457).
    # Penalty is credited at most once per step (break after first hit) to
    # match the C++ break.
    for g in range(num_ghosts):
        g_row = int(next_state[idx_ghosts_start + 2 * g])
        g_col = int(next_state[idx_ghosts_start + 2 * g + 1])
        prev_g_row = int(state[idx_ghosts_start + 2 * g])
        prev_g_col = int(state[idx_ghosts_start + 2 * g + 1])
        same_cell = g_row == next_pac_row and g_col == next_pac_col
        swap = (
            prev_g_row == next_pac_row
            and prev_g_col == next_pac_col
            and g_row == prev_pac_row
            and g_col == prev_pac_col
        )
        if same_cell or swap:
            reward += ghost_collision_penalty
            break

    if next_state[idx_score] > state[idx_score]:
        reward += pellet_reward

    n_dangerous = dangerous_areas.shape[0]
    if n_dangerous > 0:
        radius_sq = dangerous_area_radius * dangerous_area_radius
        for d in range(n_dangerous):
            dr = next_pac_row - dangerous_areas[d, 0]
            dc = next_pac_col - dangerous_areas[d, 1]
            if dr * dr + dc * dc <= radius_sq:
                reward -= dangerous_area_penalty
                break

    if next_state[idx_terminal] > 0.5 and n_pellets > 0:
        all_collected = True
        for p in range(n_pellets):
            if next_state[idx_pellets_start + p] > 0.5:
                all_collected = False
                break
        if all_collected:
            reward += win_reward

    return reward
