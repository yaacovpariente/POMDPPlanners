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
