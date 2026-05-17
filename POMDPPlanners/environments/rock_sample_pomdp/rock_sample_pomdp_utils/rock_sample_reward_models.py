"""Reward models for the RockSample POMDP.

Mirrors the abstract-base / concrete-subclass layout used by
:mod:`POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models`
and
:mod:`POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_utils.laser_tag_reward_models`,
so that further RockSample reward variants can be added without growing
the env class.

The reward model owns the parameters that the reward computation needs
(rock positions, dangerous areas, per-action rewards). The environment
retains its own parameter copies for the transition / observation paths
and delegates ``reward()`` / ``reward_batch()`` to the model.
"""

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class BaseRockSampleRewardModel(ABC):
    """Abstract reward model for RockSample POMDP variants."""

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


class RockSampleRewardModel(BaseRockSampleRewardModel):
    """Standard RockSample reward model.

    Reward structure:
        * Exit (action ``2`` East from the rightmost column): ``+exit_reward``.
        * Sample (action ``0``) at a rock cell: ``+good_rock_reward`` if the
          rock is good, ``+bad_rock_penalty`` if it is bad.
        * Check actions (``>= 5``): ``+sensor_use_penalty``.
        * Per-step: ``+step_penalty`` baseline applied to every action.
        * Dangerous area: ``+dangerous_area_penalty`` is added whenever the
          *realised* next robot position lies inside a dangerous area, gated
          by a per-call Bernoulli with probability
          ``dangerous_area_hit_probability`` (deterministic when ``== 1.0``).

    Note that this model uses the additive convention: pass a *negative*
    ``dangerous_area_penalty`` to penalise danger entry.
    """

    def __init__(
        self,
        map_size: Tuple[int, int],
        rock_positions: List[Tuple[int, int]],
        step_penalty: float,
        bad_rock_penalty: float,
        good_rock_reward: float,
        sensor_use_penalty: float,
        exit_reward: float,
        dangerous_areas: List[Tuple[int, int]],
        dangerous_area_radius: float,
        dangerous_area_penalty: float,
        dangerous_area_hit_probability: float,
    ):
        self.map_size = map_size
        self.rock_positions = rock_positions
        self.step_penalty = step_penalty
        self.bad_rock_penalty = bad_rock_penalty
        self.good_rock_reward = good_rock_reward
        self.sensor_use_penalty = sensor_use_penalty
        self.exit_reward = exit_reward
        self.dangerous_areas = dangerous_areas
        self.dangerous_area_radius = dangerous_area_radius
        self.dangerous_area_penalty = dangerous_area_penalty
        self.dangerous_area_hit_probability = float(dangerous_area_hit_probability)

    def _is_in_dangerous_area(self, position: Tuple[int, int]) -> bool:
        """Check if a position is within any dangerous area."""
        if not self.dangerous_areas:
            return False

        pos_row, pos_col = position
        for danger_row, danger_col in self.dangerous_areas:
            distance = math.sqrt((pos_row - danger_row) ** 2 + (pos_col - danger_col) ** 2)
            if distance <= self.dangerous_area_radius:
                return True

        return False

    def compute_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> float:
        total_reward = self.step_penalty

        robot_row, robot_col = int(state[0]), int(state[1])
        if action == 2 and robot_col == self.map_size[1] - 1:
            total_reward += self.exit_reward
            return total_reward

        if action == 0:
            for i, rock_pos in enumerate(self.rock_positions):
                if (robot_row, robot_col) == rock_pos:
                    if bool(state[2 + i] > 0.5):
                        total_reward += self.good_rock_reward
                    else:
                        total_reward += self.bad_rock_penalty
                    break

        if action >= 5:
            total_reward += self.sensor_use_penalty

        next_robot_pos = (int(next_state[0]), int(next_state[1]))
        if self._is_in_dangerous_area(next_robot_pos):
            if (
                self.dangerous_area_hit_probability >= 1.0
                or np.random.random() < self.dangerous_area_hit_probability
            ):
                total_reward += self.dangerous_area_penalty

        return total_reward

    def compute_reward_batch(
        self,
        states: np.ndarray,
        action: int,
        next_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = states.shape[0]
        rewards = np.full(n, self.step_penalty, dtype=np.float64)
        map_cols = self.map_size[1]

        if action == 2:
            exits = states[:, 1].astype(int) == (map_cols - 1)
            terminal = (states[:, 0] < 0) & (states[:, 1] < 0)
            rewards[exits | terminal] += self.exit_reward
            return rewards

        if action == 0:
            robot_rows = states[:, 0].astype(int)
            robot_cols = states[:, 1].astype(int)
            for i, (rr, rc) in enumerate(self.rock_positions):
                at_rock = (robot_rows == rr) & (robot_cols == rc)
                if not np.any(at_rock):
                    continue
                rock_slot = 2 + i
                rock_good = states[:, rock_slot] > 0.5
                rewards[at_rock & rock_good] += self.good_rock_reward
                rewards[at_rock & ~rock_good] += self.bad_rock_penalty

        if action >= 5:
            rewards += self.sensor_use_penalty

        if not self.dangerous_areas:
            return rewards

        if next_states is not None:
            next_robot_rows = next_states[:, 0].astype(int)
            next_robot_cols = next_states[:, 1].astype(int)
        else:
            next_robot_rows, next_robot_cols = self._closed_form_next_robot_pos(states, action)
        deterministic = self.dangerous_area_hit_probability >= 1.0
        for j in range(n):
            if not self._is_in_dangerous_area((next_robot_rows[j], next_robot_cols[j])):
                continue
            if deterministic or np.random.random() < self.dangerous_area_hit_probability:
                rewards[j] += self.dangerous_area_penalty

        return rewards

    def _closed_form_next_robot_pos(
        self, states: np.ndarray, action: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Closed-form post-transition robot position for the dangerous-area
        # check. RockSample movement actions translate by ``(dr, dc)`` and
        # clip to map bounds; sample / sense actions leave position
        # unchanged. Terminal-sentinel rows (state[:, 0] < 0) keep their
        # negative coordinates so they never match a dangerous-area cell,
        # mirroring the batch-kernel semantics for that path.
        robot_rows = states[:, 0].astype(int)
        robot_cols = states[:, 1].astype(int)
        if action == 1:
            dr, dc = -1, 0
        elif action == 3:
            dr, dc = 1, 0
        elif action == 4:
            dr, dc = 0, -1
        else:
            dr, dc = 0, 0
        new_rows = robot_rows + dr
        new_cols = robot_cols + dc
        terminal = (states[:, 0] < 0) & (states[:, 1] < 0)
        new_rows = np.clip(new_rows, 0, self.map_size[0] - 1)
        new_cols = np.clip(new_cols, 0, self.map_size[1] - 1)
        new_rows[terminal] = robot_rows[terminal]
        new_cols[terminal] = robot_cols[terminal]
        return new_rows, new_cols
