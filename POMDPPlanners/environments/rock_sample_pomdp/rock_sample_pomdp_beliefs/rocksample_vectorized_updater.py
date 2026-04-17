"""Vectorized particle belief updater for RockSample POMDP.

This module provides a NumPy-vectorized implementation of particle belief
updates for the RockSample environment, eliminating Python-level loops
over individual particles.

Classes:
    RockSampleVectorizedUpdater: Batched transition and observation
        log-likelihood for RockSample states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
        RockSamplePOMDP,
    )

# Observation encoding: "none" -> 0, "good" -> 1, "bad" -> 2
OBS_NONE = 0
OBS_GOOD = 1
OBS_BAD = 2


class RockSampleVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Vectorized particle belief updater for the RockSample POMDP.

    Stores precomputed environment parameters and performs all-particle
    transitions and observation log-likelihood evaluations using NumPy
    operations.  State layout per particle is
    ``[robot_row, robot_col, rock_0_quality, ..., rock_{R-1}_quality]``.

    Attributes:
        map_rows: Number of grid rows.
        map_cols: Number of grid columns.
        num_rocks: Number of rocks in the environment.
        rock_positions: Array of shape (R, 2) with rock (row, col) positions.
        sensor_efficiency: Sensor noise parameter (higher = less noise).
    """

    def __init__(
        self,
        map_rows: int,
        map_cols: int,
        num_rocks: int,
        rock_positions: np.ndarray,
        sensor_efficiency: float,
    ):
        self.map_rows = map_rows
        self.map_cols = map_cols
        self.num_rocks = num_rocks
        self.rock_positions = rock_positions
        self.sensor_efficiency = sensor_efficiency

    @classmethod
    def from_environment(cls, env: RockSamplePOMDP) -> RockSampleVectorizedUpdater:
        """Construct an updater from a RockSamplePOMDP instance."""
        rock_pos = np.array(env.rock_positions, dtype=np.int32)
        return cls(
            map_rows=env.map_size[0],
            map_cols=env.map_size[1],
            num_rocks=len(env.rock_positions),
            rock_positions=rock_pos,
            sensor_efficiency=env.sensor_efficiency,
        )

    # ------------------------------------------------------------------
    # Public interface (ABC)
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Transition all particles for the given action.

        Args:
            particles: Array of shape (N, 2 + num_rocks).
            action: Scalar action index.

        Returns:
            Next-state particles of shape (N, 2 + num_rocks).
        """
        action_idx = int(action)
        result = particles.copy()

        live = self._live_mask(result)
        if not np.any(live):
            return result

        if action_idx == 0:
            self._apply_sample(result, live)
        elif 1 <= action_idx <= 4:
            self._apply_movement(result, live, action_idx)
        # action >= 5 (check): no state change

        self._apply_exit(result, live)
        return result

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> np.ndarray:
        """Compute observation log-likelihoods for all particles.

        Args:
            next_particles: Array of shape (N, 2 + num_rocks).
            action: Scalar action index.
            observation: Integer-encoded observation (0=none, 1=good, 2=bad).

        Returns:
            Log-likelihoods of shape (N,).
        """
        action_idx = int(action)
        obs_int = int(np.asarray(observation).item())
        n_particles = next_particles.shape[0]

        if action_idx <= 4:
            return self._log_ll_movement(n_particles, obs_int)

        rock_idx = action_idx - 5
        if rock_idx >= self.num_rocks:
            return self._log_ll_movement(n_particles, obs_int)

        if obs_int == OBS_NONE:
            return np.full(n_particles, -np.inf)

        return self._log_ll_check(next_particles, rock_idx, obs_int)

    @property
    def config_id(self) -> str:
        """Return a deterministic identifier for this updater configuration."""
        cfg = {
            "class": "RockSampleVectorizedUpdater",
            "map_rows": self.map_rows,
            "map_cols": self.map_cols,
            "num_rocks": self.num_rocks,
            "rock_positions": self.rock_positions.tolist(),
            "sensor_efficiency": self.sensor_efficiency,
        }
        return config_to_id(cfg)

    # ------------------------------------------------------------------
    # Private helpers — transition
    # ------------------------------------------------------------------

    @staticmethod
    def _live_mask(particles: np.ndarray) -> np.ndarray:
        return (particles[:, 0] >= 0) | (particles[:, 1] >= 0)

    def _apply_movement(self, result: np.ndarray, live: np.ndarray, action_idx: int) -> None:
        if action_idx == 1:  # North
            result[live, 0] = np.maximum(0, result[live, 0] - 1)
        elif action_idx == 2:  # East
            result[live, 1] = result[live, 1] + 1
        elif action_idx == 3:  # South
            result[live, 0] = np.minimum(self.map_rows - 1, result[live, 0] + 1)
        elif action_idx == 4:  # West
            result[live, 1] = np.maximum(0, result[live, 1] - 1)

    def _apply_sample(self, result: np.ndarray, live: np.ndarray) -> None:
        robot_row = result[:, 0].astype(np.int32)
        robot_col = result[:, 1].astype(np.int32)
        for i in range(self.num_rocks):
            rock_r, rock_c = self.rock_positions[i]
            at_rock = live & (robot_row == rock_r) & (robot_col == rock_c)
            result[at_rock, 2 + i] = 0.0

    def _apply_exit(self, result: np.ndarray, live: np.ndarray) -> None:
        exited = live & (result[:, 1] >= self.map_cols)
        result[exited, 0] = -1.0
        result[exited, 1] = -1.0

    # ------------------------------------------------------------------
    # Private helpers — observation log-likelihood
    # ------------------------------------------------------------------

    @staticmethod
    def _log_ll_movement(n_particles: int, obs_int: int) -> np.ndarray:
        if obs_int == OBS_NONE:
            return np.zeros(n_particles)
        return np.full(n_particles, -np.inf)

    def _log_ll_check(
        self,
        next_particles: np.ndarray,
        rock_idx: int,
        obs_int: int,
    ) -> np.ndarray:
        rock_r, rock_c = self.rock_positions[rock_idx]
        robot_row = next_particles[:, 0]
        robot_col = next_particles[:, 1]

        distance = np.sqrt((robot_row - rock_r) ** 2 + (robot_col - rock_c) ** 2)
        efficiency = np.exp(-distance / self.sensor_efficiency)

        is_good = next_particles[:, 2 + rock_idx] > 0.5

        if obs_int == OBS_GOOD:
            prob = np.where(is_good, efficiency, 1.0 - efficiency)
        else:  # OBS_BAD
            prob = np.where(is_good, 1.0 - efficiency, efficiency)

        prob = np.maximum(prob, 1e-300)
        log_ll = np.log(prob)

        is_terminal = (next_particles[:, 0] < 0) & (next_particles[:, 1] < 0)
        log_ll[is_terminal] = -np.inf

        return log_ll
