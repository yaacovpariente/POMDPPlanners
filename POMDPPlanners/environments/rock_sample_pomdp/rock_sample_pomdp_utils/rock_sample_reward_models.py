"""Reward models for the RockSample POMDP.

Mirrors the abstract-base / concrete-subclass layout used by
:mod:`POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models`
and
:mod:`POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_utils.laser_tag_reward_models`,
so that further RockSample reward variants can be added without growing
the env class.

Three concrete variants are provided. They share all of the
non-dangerous-area scoring (exit, sample, sense, step-penalty) via the
``RockSampleRewardModel`` base; each variant only customises the
dangerous-area contribution:

* :class:`RockSampleRewardModel` (STANDARD): constant-probability
  penalty when the realised next position is in any dangerous area.
* :class:`RockSampleHighVarianceRewardModel` (HIGH_VARIANCE_STATES):
  ``±dangerous_area_penalty`` 50/50 in-zone — zero expected
  contribution, high variance.
* :class:`RockSampleDecayingHitProbabilityRewardModel`
  (DECAYING_HIT_PROBABILITY): penalty applied with probability
  ``exp(-min_dist / penalty_decay)`` based on the *closest* zone
  centre — no radius cutoff.
"""

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from POMDPPlanners.environments.environment_utils.dangerous_areas_kernels import (
    decaying_prob_penalty_batch_kernel,
    decaying_prob_penalty_kernel,
    membership_within_radius_batch_kernel,
)


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

    Subclasses override :meth:`_dangerous_area_contribution_scalar` and
    :meth:`_apply_dangerous_area_contribution_batch` to express different
    stochastic penalty models; the rest of the scoring is identical.
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

        # Pre-built (2, D) C-contiguous float64 array of dangerous-area centres
        # consumed by the njit kernels so the hot path avoids per-call array
        # construction. Squared radius cached so the membership kernel takes
        # ``radius_sq`` directly.
        if self.dangerous_areas:
            self._dangerous_areas_xy: np.ndarray = np.ascontiguousarray(
                np.asarray(self.dangerous_areas, dtype=np.float64).T
            )
        else:
            self._dangerous_areas_xy = np.empty((2, 0), dtype=np.float64)
        self._dangerous_area_radius_sq: float = float(
            self.dangerous_area_radius * self.dangerous_area_radius
        )

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
        total_reward += self._dangerous_area_contribution_scalar(next_robot_pos)

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

        exits_mask: Optional[np.ndarray] = None
        if action == 2:
            exits_mask = states[:, 1].astype(int) == (map_cols - 1)
            rewards[exits_mask] += self.exit_reward

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

        self._apply_dangerous_area_contribution_batch(
            rewards, next_robot_rows, next_robot_cols, skip_mask=exits_mask
        )

        return rewards

    def _dangerous_area_contribution_scalar(self, next_robot_pos: Tuple[int, int]) -> float:
        """Constant-probability penalty: ``+dangerous_area_penalty`` in-zone
        with probability ``dangerous_area_hit_probability`` (deterministic
        when ``== 1.0``), otherwise ``0.0``.
        """
        if not self._is_in_dangerous_area(next_robot_pos):
            return 0.0
        if (
            self.dangerous_area_hit_probability >= 1.0
            or np.random.random() < self.dangerous_area_hit_probability
        ):
            return self.dangerous_area_penalty
        return 0.0

    def _apply_dangerous_area_contribution_batch(
        self,
        rewards: np.ndarray,
        next_robot_rows: np.ndarray,
        next_robot_cols: np.ndarray,
        skip_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Vectorised constant-probability dangerous-area contribution.

        RNG stays in Python so seeded behaviour matches the per-row scalar
        loop bit-for-bit: deterministic case draws zero uniforms;
        stochastic case draws exactly ``in_zone.sum()`` uniforms, in
        ascending row index order. ``skip_mask`` rows (e.g. East exits)
        are excluded from both the membership check and the RNG draws so
        the scalar early-return contract is preserved.
        """
        in_zone = self._compute_in_zone_mask(next_robot_rows, next_robot_cols)
        if skip_mask is not None:
            in_zone &= ~skip_mask
        if self.dangerous_area_hit_probability >= 1.0:
            rewards[in_zone] += self.dangerous_area_penalty
        else:
            in_zone_indices = np.flatnonzero(in_zone)
            coins = np.random.random(in_zone_indices.size)
            hits = coins < self.dangerous_area_hit_probability
            rewards[in_zone_indices[hits]] += self.dangerous_area_penalty

    def _compute_in_zone_mask(
        self, next_robot_rows: np.ndarray, next_robot_cols: np.ndarray
    ) -> np.ndarray:
        # Single allocation + cast-on-assign beats column_stack(astype, astype),
        # which materialises two extra intermediate float64 arrays before
        # concatenating them.
        points = np.empty((next_robot_rows.shape[0], 2), dtype=np.float64)
        points[:, 0] = next_robot_rows
        points[:, 1] = next_robot_cols
        return membership_within_radius_batch_kernel(
            points,
            self._dangerous_areas_xy,
            self._dangerous_area_radius_sq,
        )

    def _closed_form_next_robot_pos(
        self, states: np.ndarray, action: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Closed-form post-transition robot position for the dangerous-area
        # check. RockSample movement actions translate by ``(dr, dc)`` and
        # clip to map bounds; sample / sense actions leave position
        # unchanged. East (action=2) on a non-rightmost row moves +1 col;
        # rightmost-column rows exit and are handled by the exit-mask in
        # ``compute_reward_batch`` so this method returns the clipped
        # position (rightmost-column rows stay put after clipping).
        # Terminal-sentinel rows (state[:, 0] < 0) keep their negative
        # coordinates so they never match a dangerous-area cell,
        # mirroring the batch-kernel semantics for that path.
        robot_rows = states[:, 0].astype(int)
        robot_cols = states[:, 1].astype(int)
        if action == 1:
            dr, dc = -1, 0
        elif action == 2:
            dr, dc = 0, 1
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


class RockSampleHighVarianceRewardModel(RockSampleRewardModel):
    """HIGH_VARIANCE_STATES variant.

    Replaces the constant-probability penalty with a 50/50 split between
    ``+dangerous_area_penalty`` and ``-dangerous_area_penalty`` whenever
    the realised next position is in any dangerous area. Expected
    contribution is ``0``; variance is ``dangerous_area_penalty**2``.
    Suitable for benchmarking risk-sensitive planners against
    expected-value planners on identical means.

    ``dangerous_area_hit_probability`` is **ignored** in this model.
    """

    def _dangerous_area_contribution_scalar(self, next_robot_pos: Tuple[int, int]) -> float:
        if not self._is_in_dangerous_area(next_robot_pos):
            return 0.0
        if np.random.random() < 0.5:
            return self.dangerous_area_penalty
        return -self.dangerous_area_penalty

    def _apply_dangerous_area_contribution_batch(
        self,
        rewards: np.ndarray,
        next_robot_rows: np.ndarray,
        next_robot_cols: np.ndarray,
        skip_mask: Optional[np.ndarray] = None,
    ) -> None:
        in_zone = self._compute_in_zone_mask(next_robot_rows, next_robot_cols)
        if skip_mask is not None:
            in_zone &= ~skip_mask
        in_zone_indices = np.flatnonzero(in_zone)
        coins = np.random.random(in_zone_indices.size)
        signs = np.where(coins < 0.5, 1.0, -1.0)
        rewards[in_zone_indices] += signs * self.dangerous_area_penalty


class RockSampleDecayingHitProbabilityRewardModel(RockSampleRewardModel):
    """DECAYING_HIT_PROBABILITY variant.

    Penalty is applied with probability
    ``exp(-min_dist / penalty_decay)`` where ``min_dist`` is the
    Euclidean distance from the realised next position to the *closest*
    dangerous-area centre. No radius cutoff — every step risks some
    (vanishingly small at large distance) penalty. Each call draws one
    uniform regardless of distance, matching the existing decaying-prob
    kernel and light-dark's analogous reward model.

    ``dangerous_area_radius`` and ``dangerous_area_hit_probability`` are
    **ignored** in this model.
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
        penalty_decay: float,
    ):
        super().__init__(
            map_size=map_size,
            rock_positions=rock_positions,
            step_penalty=step_penalty,
            bad_rock_penalty=bad_rock_penalty,
            good_rock_reward=good_rock_reward,
            sensor_use_penalty=sensor_use_penalty,
            exit_reward=exit_reward,
            dangerous_areas=dangerous_areas,
            dangerous_area_radius=dangerous_area_radius,
            dangerous_area_penalty=dangerous_area_penalty,
            dangerous_area_hit_probability=dangerous_area_hit_probability,
        )
        if penalty_decay <= 0.0:
            raise ValueError("penalty_decay must be positive")
        self.penalty_decay = float(penalty_decay)

    def _dangerous_area_contribution_scalar(self, next_robot_pos: Tuple[int, int]) -> float:
        if not self.dangerous_areas:
            return 0.0
        point = np.asarray(next_robot_pos, dtype=np.float64)
        uniform = float(np.random.random())
        return float(
            decaying_prob_penalty_kernel(
                point,
                self._dangerous_areas_xy,
                self.dangerous_area_penalty,
                self.penalty_decay,
                uniform,
            )
        )

    def _apply_dangerous_area_contribution_batch(
        self,
        rewards: np.ndarray,
        next_robot_rows: np.ndarray,
        next_robot_cols: np.ndarray,
        skip_mask: Optional[np.ndarray] = None,
    ) -> None:
        # Single allocation + cast-on-assign beats column_stack(astype, astype),
        # which materialises two extra intermediate float64 arrays before
        # concatenating them.
        points = np.empty((next_robot_rows.shape[0], 2), dtype=np.float64)
        points[:, 0] = next_robot_rows
        points[:, 1] = next_robot_cols
        uniforms = np.random.random(points.shape[0])
        contributions = decaying_prob_penalty_batch_kernel(
            points,
            self._dangerous_areas_xy,
            self.dangerous_area_penalty,
            self.penalty_decay,
            uniforms,
        )
        if skip_mask is not None:
            contributions = np.where(skip_mask, 0.0, contributions)
        rewards += contributions
