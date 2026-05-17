"""Reward models for the PacMan POMDP.

Mirrors the abstract-base / concrete-subclass layout used by
:mod:`POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models`
and
:mod:`POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_utils.laser_tag_reward_models`,
so further PacMan reward variants can be added without growing the env class.

The reward model owns all the parameters and pre-built buffers that the
reward computation needs (state-layout indices, pellet positions,
dangerous areas, scalar penalties / bonuses, and a callable that returns
the env's lazily-built neighbor table). The environment retains its own
copies of these values for the transition / observation paths and
delegates ``reward()`` / ``reward_batch()`` to the model.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import numpy as np


class BasePacManRewardModel(ABC):
    """Abstract reward model for PacMan POMDP variants."""

    @abstractmethod
    def compute_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> float:
        """Return the scalar reward for ``(state, action, next_state)``."""

    @abstractmethod
    def compute_reward_batch(
        self,
        states: np.ndarray,
        action: int,
        next_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the per-row reward for a batch of states under a single action."""


class PacManRewardModel(BasePacManRewardModel):
    """Standard PacMan reward model.

    Reward structure:
        * Per-step: ``step_penalty`` baseline applied to every non-terminal
          transition.
        * Pellet collection: ``+pellet_reward`` when ``next_state`` records
          a score increase versus ``state`` (scalar path) or when the
          realised next position lands on an active pellet cell (batch
          path).
        * Ghost collision: ``+ghost_collision_penalty`` when any ghost in
          ``next_state`` occupies PacMan's realised next cell.
        * Dangerous area: ``-dangerous_area_penalty`` when PacMan's
          realised next position lies inside any configured circular
          hazard zone (squared-distance check against
          ``dangerous_area_radius``).
        * Win bonus: ``+win_reward`` when the transition consumes the
          last remaining pellet (terminal with empty pellet mask in the
          scalar path; final-pellet pickup in the batch path).

    Terminal states return ``0.0`` reward.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        *,
        num_ghosts: int,
        pellet_positions_arr: np.ndarray,
        dangerous_areas: List[Tuple[int, int]],
        dangerous_areas_arr: np.ndarray,
        dangerous_area_radius: float,
        dangerous_area_penalty: float,
        step_penalty: float,
        ghost_collision_penalty: float,
        pellet_reward: float,
        win_reward: float,
        idx_pac_row: int,
        idx_pac_col: int,
        idx_ghosts_start: int,
        idx_pellets_start: int,
        idx_pellets_end: int,
        idx_score: int,
        idx_terminal: int,
        neighbor_table_getter: Callable[[], np.ndarray],
    ):
        self.num_ghosts = num_ghosts
        self._pellet_positions_arr = pellet_positions_arr
        self.dangerous_areas = dangerous_areas
        self._dangerous_areas_arr = dangerous_areas_arr
        self.dangerous_area_radius = float(dangerous_area_radius)
        self.dangerous_area_penalty = float(dangerous_area_penalty)
        self.step_penalty = float(step_penalty)
        self.ghost_collision_penalty = float(ghost_collision_penalty)
        self.pellet_reward = float(pellet_reward)
        self.win_reward = float(win_reward)
        self._idx_pac_row = idx_pac_row
        self._idx_pac_col = idx_pac_col
        self._idx_ghosts_start = idx_ghosts_start
        self._idx_pellets_start = idx_pellets_start
        self._idx_pellets_end = idx_pellets_end
        self._idx_score = idx_score
        self._idx_terminal = idx_terminal
        self._neighbor_table_getter = neighbor_table_getter

    def _is_in_dangerous_area(self, position: Tuple[int, int]) -> bool:
        # Squared-distance check matches laser_tag's discrete implementation
        # and avoids a sqrt per call on a hot path.
        if not self.dangerous_areas:
            return False
        pos_row, pos_col = position
        radius_sq = self.dangerous_area_radius * self.dangerous_area_radius
        for danger_row, danger_col in self.dangerous_areas:
            dr = pos_row - danger_row
            dc = pos_col - danger_col
            if dr * dr + dc * dc <= radius_sq:
                return True
        return False

    def compute_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> float:
        del action  # Realised post-transition state already encodes the choice.
        if state[self._idx_terminal] > 0.5:
            return 0.0

        total_reward = self.step_penalty

        next_pac_row = int(next_state[self._idx_pac_row])
        next_pac_col = int(next_state[self._idx_pac_col])
        for g in range(self.num_ghosts):
            g_row = int(next_state[self._idx_ghosts_start + 2 * g])
            g_col = int(next_state[self._idx_ghosts_start + 2 * g + 1])
            if next_pac_row == g_row and next_pac_col == g_col:
                total_reward += self.ghost_collision_penalty
                break

        if next_state[self._idx_score] > state[self._idx_score]:
            total_reward += self.pellet_reward

        if self._is_in_dangerous_area((next_pac_row, next_pac_col)):
            total_reward -= self.dangerous_area_penalty

        if next_state[self._idx_terminal] > 0.5:
            pellet_mask = next_state[self._idx_pellets_start : self._idx_pellets_end]
            if not np.any(pellet_mask > 0.5):
                total_reward += self.win_reward

        return total_reward

    def compute_reward_batch(
        self,
        states: np.ndarray,
        action: int,
        next_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Single-pass vectorised reward kernel. Without ``next_states`` the
        # ghost-collision penalty is excluded because it depends on the
        # stochastic ghost transition; when supplied (caller already
        # realised the batch transition) the penalty is included against
        # those realised draws. Patrol-direction state is left alone here
        # — it is mutated only inside the C++ transition kernel.
        terminal = states[:, self._idx_terminal] > 0.5
        rewards = np.where(terminal, 0.0, self.step_penalty)

        # Vectorised neighbor-table lookup: (rows, cols, action) -> next pos.
        pac_rows = states[:, self._idx_pac_row].astype(np.int32)
        pac_cols = states[:, self._idx_pac_col].astype(np.int32)
        new_positions = self._neighbor_table_getter()[pac_rows, pac_cols, action]
        new_pac_rows = new_positions[:, 0]
        new_pac_cols = new_positions[:, 1]

        rewards = self._add_collision_penalty_batch(
            rewards, terminal, new_pac_rows, new_pac_cols, next_states
        )

        rewards = self._add_dangerous_area_penalty_batch(
            rewards, terminal, new_pac_rows, new_pac_cols
        )

        if self._pellet_positions_arr.shape[0] == 0:
            # Degenerate config (env constructed with no pellets): there is
            # nothing to collect and therefore nothing to "win". Returning
            # rewards alone (step penalty for non-terminal, zero for
            # terminal, plus optional collision) avoids the prior bug that
            # paid out ``win_reward`` on every non-terminal step against
            # an empty pellet set.
            return rewards

        pellet_mask = states[:, self._idx_pellets_start : self._idx_pellets_end]
        pellet_pos = self._pellet_positions_arr  # (P, 2) int32, static.

        # Broadcast (N, 1) vs (1, P): which pellet (if any) sits on the
        # target cell, gated on it currently being active.
        pos_match = (new_pac_rows[:, None] == pellet_pos[None, :, 0]) & (
            new_pac_cols[:, None] == pellet_pos[None, :, 1]
        )
        active_match = pos_match & (pellet_mask > 0.5)
        collected = active_match.any(axis=1)

        remaining_after = pellet_mask.sum(axis=1) - collected.astype(np.float64)
        all_collected = collected & (remaining_after < 0.5)

        rewards += np.where(~terminal & collected, self.pellet_reward, 0.0)
        rewards += np.where(~terminal & all_collected, self.win_reward, 0.0)
        return rewards

    def _add_collision_penalty_batch(
        self,
        rewards: np.ndarray,
        terminal: np.ndarray,
        new_pac_rows: np.ndarray,
        new_pac_cols: np.ndarray,
        next_states: Optional[np.ndarray],
    ) -> np.ndarray:
        if next_states is None or self.num_ghosts <= 0:
            return rewards
        # Use realised post-transition ghost positions; mark a collision
        # for any ghost that ends on pacman's new cell.
        collision = np.zeros(rewards.shape, dtype=bool)
        for g in range(self.num_ghosts):
            g_rows = next_states[:, self._idx_ghosts_start + 2 * g].astype(np.int32)
            g_cols = next_states[:, self._idx_ghosts_start + 2 * g + 1].astype(np.int32)
            collision |= (g_rows == new_pac_rows) & (g_cols == new_pac_cols)
        return rewards + np.where(~terminal & collision, self.ghost_collision_penalty, 0.0)

    def _add_dangerous_area_penalty_batch(
        self,
        rewards: np.ndarray,
        terminal: np.ndarray,
        new_pac_rows: np.ndarray,
        new_pac_cols: np.ndarray,
    ) -> np.ndarray:
        if self._dangerous_areas_arr.shape[0] == 0:
            return rewards
        # Vectorised squared-distance check across all configured zones;
        # ``any`` collapses the (N, D) matrix to a per-particle mask. Uses
        # squared distance to stay consistent with ``_is_in_dangerous_area``.
        centers = self._dangerous_areas_arr  # (D, 2) float64
        dr = new_pac_rows[:, None] - centers[:, 0]
        dc = new_pac_cols[:, None] - centers[:, 1]
        radius_sq = self.dangerous_area_radius * self.dangerous_area_radius
        in_danger = (dr * dr + dc * dc <= radius_sq).any(axis=1)
        return rewards + np.where(~terminal & in_danger, -self.dangerous_area_penalty, 0.0)
