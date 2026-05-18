"""Reward models for the PacMan POMDP.

Mirrors the abstract-base / concrete-subclass layout used by
:mod:`POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models`
and
:mod:`POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_utils.laser_tag_reward_models`,
so further PacMan reward variants can be added without growing the env class.

Three concrete variants are provided. They share all of the non-dangerous-
area scoring (step / collision / pellet / win) via the
``PacManRewardModel`` base; each variant only customises the dangerous-
area contribution:

* :class:`PacManRewardModel` (CONSTANT_HAZARD_PENALTY): deterministic
  ``-dangerous_area_penalty`` whenever the realised next pacman position
  lies inside any configured circular hazard zone (squared-distance
  check against ``dangerous_area_radius``).
* :class:`PacManZeroMeanHazardShockRewardModel` (ZERO_MEAN_HAZARD_SHOCK):
  ``±dangerous_area_penalty`` 50/50 in-zone — zero expected
  contribution, high variance.
* :class:`PacManDistanceDecayedHazardPenaltyRewardModel`
  (DISTANCE_DECAYED_HAZARD_PENALTY): ``-dangerous_area_penalty`` is applied
  with probability ``exp(-min_dist / penalty_decay)`` based on the
  *closest* zone centre — no radius cutoff.

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

from POMDPPlanners.environments.environment_utils.dangerous_areas_kernels import (
    decaying_prob_penalty_batch_kernel,
    decaying_prob_penalty_kernel,
)
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_utils.numba_kernels import (
    compute_reward_batch_kernel,
    compute_reward_scalar_kernel,
)


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

    Subclasses override :meth:`_dangerous_area_contribution_scalar` and
    :meth:`_apply_dangerous_area_contribution_batch` to express different
    stochastic penalty models; the rest of the scoring is identical.
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

        # (2, D) C-contiguous view of dangerous-area centres for the generic
        # decaying-prob kernels (which expect axis 0 to hold the two coordinate
        # components). Empty (2, 0) when no zones are configured.
        if self._dangerous_areas_arr.shape[0] > 0:
            self._dangerous_centers_xy: np.ndarray = np.ascontiguousarray(
                self._dangerous_areas_arr.T
            )
        else:
            self._dangerous_centers_xy = np.empty((2, 0), dtype=np.float64)
        # (0, 2) sentinel passed to the fused kernel by subclasses that want
        # to compute the dangerous-area contribution in Python instead.
        self._empty_dangerous_areas: np.ndarray = np.empty((0, 2), dtype=np.float64)
        # Square-distance comparison cached so the per-row danger check
        # never sqrts on the hot path.
        self._dangerous_radius_sq: float = float(dangerous_area_radius) * float(
            dangerous_area_radius
        )

    def compute_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> float:
        del action  # Realised post-transition state already encodes the choice.
        return float(
            compute_reward_scalar_kernel(
                np.ascontiguousarray(state, dtype=np.float64),
                np.ascontiguousarray(next_state, dtype=np.float64),
                self._dangerous_areas_arr,
                int(self.num_ghosts),
                int(self._pellet_positions_arr.shape[0]),
                float(self.step_penalty),
                float(self.ghost_collision_penalty),
                float(self.pellet_reward),
                float(self.win_reward),
                float(self.dangerous_area_penalty),
                float(self.dangerous_area_radius),
                int(self._idx_pac_row),
                int(self._idx_pac_col),
                int(self._idx_ghosts_start),
                int(self._idx_pellets_start),
                int(self._idx_score),
                int(self._idx_terminal),
            )
        )

    def compute_reward_batch(
        self,
        states: np.ndarray,
        action: int,
        next_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Single-pass fused reward kernel. The Numba kernel fuses the
        # terminal / collision / dangerous-area / pellet / win
        # contributions into one loop over rows, replacing ~15-20 numpy
        # operator calls. Without ``next_states`` the ghost-collision
        # penalty is excluded because it depends on the stochastic ghost
        # transition; when supplied (caller already realised the batch
        # transition) the penalty is included against those realised
        # draws. Patrol-direction state is left alone here — it is
        # mutated only inside the C++ transition kernel.
        states_arr = np.ascontiguousarray(states, dtype=np.float64)
        if next_states is None:
            next_states_arr = np.empty((0, states_arr.shape[1]), dtype=np.float64)
            has_next_states = False
        else:
            next_states_arr = np.ascontiguousarray(next_states, dtype=np.float64)
            has_next_states = True

        return compute_reward_batch_kernel(
            states_arr,
            int(action),
            next_states_arr,
            has_next_states,
            np.ascontiguousarray(self._neighbor_table_getter(), dtype=np.int32),
            self._pellet_positions_arr,
            self._dangerous_areas_arr,
            int(self.num_ghosts),
            float(self.step_penalty),
            float(self.ghost_collision_penalty),
            float(self.pellet_reward),
            float(self.win_reward),
            float(self.dangerous_area_penalty),
            float(self.dangerous_area_radius),
            int(self._idx_pac_row),
            int(self._idx_pac_col),
            int(self._idx_ghosts_start),
            int(self._idx_pellets_start),
            int(self._idx_terminal),
        )

    # ------------------------------------------------------------------
    # Helpers reused by HV / Decaying subclasses (kept private so the
    # public API stays the abstract ``compute_reward`` / ``compute_reward_batch``).
    # ------------------------------------------------------------------

    def _compute_base_reward_scalar(self, state: np.ndarray, next_state: np.ndarray) -> float:
        # Reward without the dangerous-area contribution. Achieved by passing
        # the ``(0, 2)`` sentinel so the fused kernel's danger loop is skipped
        # (its first check is ``if n_dangerous > 0``).
        return float(
            compute_reward_scalar_kernel(
                np.ascontiguousarray(state, dtype=np.float64),
                np.ascontiguousarray(next_state, dtype=np.float64),
                self._empty_dangerous_areas,
                int(self.num_ghosts),
                int(self._pellet_positions_arr.shape[0]),
                float(self.step_penalty),
                float(self.ghost_collision_penalty),
                float(self.pellet_reward),
                float(self.win_reward),
                0.0,
                0.0,
                int(self._idx_pac_row),
                int(self._idx_pac_col),
                int(self._idx_ghosts_start),
                int(self._idx_pellets_start),
                int(self._idx_score),
                int(self._idx_terminal),
            )
        )

    def _compute_base_reward_batch(
        self,
        states: np.ndarray,
        action: int,
        next_states: Optional[np.ndarray],
    ) -> np.ndarray:
        # Same fused-kernel call as :meth:`compute_reward_batch` but with the
        # dangerous-area arg zeroed so the variant can add its own contribution.
        states_arr = np.ascontiguousarray(states, dtype=np.float64)
        if next_states is None:
            next_states_arr = np.empty((0, states_arr.shape[1]), dtype=np.float64)
            has_next_states = False
        else:
            next_states_arr = np.ascontiguousarray(next_states, dtype=np.float64)
            has_next_states = True
        return compute_reward_batch_kernel(
            states_arr,
            int(action),
            next_states_arr,
            has_next_states,
            np.ascontiguousarray(self._neighbor_table_getter(), dtype=np.int32),
            self._pellet_positions_arr,
            self._empty_dangerous_areas,
            int(self.num_ghosts),
            float(self.step_penalty),
            float(self.ghost_collision_penalty),
            float(self.pellet_reward),
            float(self.win_reward),
            0.0,
            0.0,
            int(self._idx_pac_row),
            int(self._idx_pac_col),
            int(self._idx_ghosts_start),
            int(self._idx_pellets_start),
            int(self._idx_terminal),
        )

    def _is_in_dangerous_area(self, position: Tuple[int, int]) -> bool:
        if self._dangerous_areas_arr.shape[0] == 0:
            return False
        pos_r, pos_c = position
        for d in range(self._dangerous_areas_arr.shape[0]):
            dr = pos_r - self._dangerous_areas_arr[d, 0]
            dc = pos_c - self._dangerous_areas_arr[d, 1]
            if dr * dr + dc * dc <= self._dangerous_radius_sq:
                return True
        return False

    def _compute_next_pacman_positions_batch(
        self, states: np.ndarray, action: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Derive realised next pacman (row, col) via the neighbor table —
        # matches the fused batch kernel exactly so variant contributions
        # score against the same realised position the base reward used.
        neighbor_table = self._neighbor_table_getter()
        pac_rows = states[:, self._idx_pac_row].astype(np.int64)
        pac_cols = states[:, self._idx_pac_col].astype(np.int64)
        new_rows = neighbor_table[pac_rows, pac_cols, int(action), 0].astype(np.int64)
        new_cols = neighbor_table[pac_rows, pac_cols, int(action), 1].astype(np.int64)
        return new_rows, new_cols

    def _compute_in_zone_mask_batch(
        self, new_pac_rows: np.ndarray, new_pac_cols: np.ndarray
    ) -> np.ndarray:
        n = new_pac_rows.shape[0]
        if self._dangerous_areas_arr.shape[0] == 0:
            return np.zeros(n, dtype=bool)
        in_zone = np.zeros(n, dtype=bool)
        centers = self._dangerous_areas_arr  # (D, 2)
        for d in range(centers.shape[0]):
            dr = new_pac_rows - centers[d, 0]
            dc = new_pac_cols - centers[d, 1]
            in_zone |= (dr * dr + dc * dc) <= self._dangerous_radius_sq
        return in_zone


class PacManZeroMeanHazardShockRewardModel(PacManRewardModel):
    """ZERO_MEAN_HAZARD_SHOCK variant.

    Replaces the deterministic in-zone penalty with a 50/50 split between
    ``+dangerous_area_penalty`` and ``-dangerous_area_penalty`` whenever
    the realised next pacman position is in any dangerous area. Expected
    contribution is ``0``; variance is ``dangerous_area_penalty**2``.
    Mirrors
    :class:`~POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models.ContinuousLDZeroMeanHazardShockRewardModel`
    and
    :class:`~POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_utils.laser_tag_reward_models.LaserTagZeroMeanHazardShockRewardModel`.
    All non-dangerous-area scoring (collision / pellet / win / step
    penalty) is identical to the standard model.
    """

    def compute_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> float:
        del action  # Realised post-transition state already encodes the choice.
        if state[self._idx_terminal] > 0.5:
            return 0.0
        base = self._compute_base_reward_scalar(state, next_state)
        position = (
            int(next_state[self._idx_pac_row]),
            int(next_state[self._idx_pac_col]),
        )
        return base + self._dangerous_area_contribution_scalar(position)

    def _dangerous_area_contribution_scalar(self, position: Tuple[int, int]) -> float:
        if not self._is_in_dangerous_area(position):
            return 0.0
        if np.random.random() < 0.5:
            return self.dangerous_area_penalty
        return -self.dangerous_area_penalty

    def compute_reward_batch(
        self,
        states: np.ndarray,
        action: int,
        next_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        states_arr = np.ascontiguousarray(states, dtype=np.float64)
        next_states_arr = (
            None if next_states is None else np.ascontiguousarray(next_states, dtype=np.float64)
        )
        rewards = self._compute_base_reward_batch(states_arr, action, next_states_arr)
        if self._dangerous_areas_arr.shape[0] == 0:
            return rewards
        new_pac_rows, new_pac_cols = self._compute_next_pacman_positions_batch(states_arr, action)
        in_zone = self._compute_in_zone_mask_batch(new_pac_rows, new_pac_cols)
        # Terminal rows return 0 reward (already zeroed by the base kernel) and
        # must not pick up a stochastic danger contribution either.
        terminal = states_arr[:, self._idx_terminal] > 0.5
        in_zone &= ~terminal
        in_zone_indices = np.flatnonzero(in_zone)
        if in_zone_indices.size == 0:
            return rewards
        # ``signs`` matches the per-row independent ±penalty draw. Single
        # batched ``np.random.random`` keeps seed semantics aligned with the
        # light-dark / laser-tag HV batch paths.
        coins = np.random.random(in_zone_indices.size)
        signs = np.where(coins < 0.5, 1.0, -1.0)
        rewards[in_zone_indices] += signs * self.dangerous_area_penalty
        return rewards


class PacManDistanceDecayedHazardPenaltyRewardModel(PacManRewardModel):
    """DISTANCE_DECAYED_HAZARD_PENALTY variant.

    Penalty is applied with probability
    ``exp(-min_dist / penalty_decay)`` where ``min_dist`` is the
    Euclidean distance from the realised next pacman position to the
    *closest* dangerous-area centre. No radius cutoff — every step
    risks some (vanishingly small at large distance) penalty. Each
    call draws one uniform regardless of distance, matching the
    existing decaying-prob kernel and the light-dark / laser-tag
    analogous reward models. ``dangerous_area_radius`` is **ignored**
    in this model.
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
        penalty_decay: float,
    ):
        super().__init__(
            num_ghosts=num_ghosts,
            pellet_positions_arr=pellet_positions_arr,
            dangerous_areas=dangerous_areas,
            dangerous_areas_arr=dangerous_areas_arr,
            dangerous_area_radius=dangerous_area_radius,
            dangerous_area_penalty=dangerous_area_penalty,
            step_penalty=step_penalty,
            ghost_collision_penalty=ghost_collision_penalty,
            pellet_reward=pellet_reward,
            win_reward=win_reward,
            idx_pac_row=idx_pac_row,
            idx_pac_col=idx_pac_col,
            idx_ghosts_start=idx_ghosts_start,
            idx_pellets_start=idx_pellets_start,
            idx_pellets_end=idx_pellets_end,
            idx_score=idx_score,
            idx_terminal=idx_terminal,
            neighbor_table_getter=neighbor_table_getter,
        )
        if penalty_decay <= 0.0:
            raise ValueError("penalty_decay must be strictly positive")
        self.penalty_decay = float(penalty_decay)

    def compute_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> float:
        del action  # Realised post-transition state already encodes the choice.
        if state[self._idx_terminal] > 0.5:
            return 0.0
        base = self._compute_base_reward_scalar(state, next_state)
        position = (
            int(next_state[self._idx_pac_row]),
            int(next_state[self._idx_pac_col]),
        )
        return base + self._dangerous_area_contribution_scalar(position)

    def _dangerous_area_contribution_scalar(self, position: Tuple[int, int]) -> float:
        if self._dangerous_centers_xy.shape[1] == 0:
            return 0.0
        point = np.array([float(position[0]), float(position[1])])
        return float(
            decaying_prob_penalty_kernel(
                point,
                self._dangerous_centers_xy,
                -self.dangerous_area_penalty,
                self.penalty_decay,
                float(np.random.random()),
            )
        )

    def compute_reward_batch(
        self,
        states: np.ndarray,
        action: int,
        next_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        states_arr = np.ascontiguousarray(states, dtype=np.float64)
        next_states_arr = (
            None if next_states is None else np.ascontiguousarray(next_states, dtype=np.float64)
        )
        rewards = self._compute_base_reward_batch(states_arr, action, next_states_arr)
        if self._dangerous_centers_xy.shape[1] == 0:
            return rewards
        new_pac_rows, new_pac_cols = self._compute_next_pacman_positions_batch(states_arr, action)
        n = new_pac_rows.shape[0]
        points = np.empty((n, 2), dtype=np.float64)
        points[:, 0] = new_pac_rows
        points[:, 1] = new_pac_cols
        # Decay penalty is applied to *every* non-terminal row (no radius
        # cutoff). One uniform draw per row keeps seed semantics aligned
        # with the light-dark / laser-tag Decaying batch paths.
        uniforms = np.random.random(n)
        contributions = decaying_prob_penalty_batch_kernel(
            points,
            self._dangerous_centers_xy,
            -self.dangerous_area_penalty,
            self.penalty_decay,
            uniforms,
        )
        terminal = states_arr[:, self._idx_terminal] > 0.5
        contributions[terminal] = 0.0
        rewards += contributions
        return rewards
