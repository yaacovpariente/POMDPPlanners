# SPDX-License-Identifier: MIT

"""Vectorized particle belief updater for RockSample POMDP.

This module provides the batched transition and observation log-likelihood
entry points used by :class:`VectorizedWeightedParticleBelief`. Both hot
paths delegate to the native C++ extension
(``POMDPPlanners.environments.rock_sample_pomdp._native``); this file is a
thin Python adapter that owns the stored environment parameters and the
observation string-to-int convention.

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
from POMDPPlanners.environments.rock_sample_pomdp import _native
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

    Stores precomputed environment parameters and dispatches batched
    transitions and observation log-likelihood evaluations to the native
    C++ extension. State layout per particle is
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
        self.rock_positions = np.asarray(rock_positions, dtype=np.int32)
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
        action_idx = int(np.asarray(action).item())
        particles_arr = np.asarray(particles, dtype=float)
        ref_state = self._reference_state(particles_arr)
        transition = _native.RockSampleTransitionCpp(
            state=ref_state,
            action=action_idx,
            map_rows=self.map_rows,
            map_cols=self.map_cols,
            num_rocks=self.num_rocks,
            rock_positions=self.rock_positions,
            sensor_efficiency=self.sensor_efficiency,
        )
        return transition.batch_sample(particles_arr)

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
        action_idx = int(np.asarray(action).item())
        obs_int = int(np.asarray(observation).item())
        next_arr = np.asarray(next_particles, dtype=float)
        ref_state = self._reference_state(next_arr)
        obs_model = _native.RockSampleObservationCpp(
            next_state=ref_state,
            action=action_idx,
            map_rows=self.map_rows,
            map_cols=self.map_cols,
            num_rocks=self.num_rocks,
            rock_positions=self.rock_positions,
            sensor_efficiency=self.sensor_efficiency,
        )
        return obs_model.batch_log_likelihood(next_arr, obs_int)

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
    # Private helpers
    # ------------------------------------------------------------------

    def _reference_state(self, particles_arr: np.ndarray) -> np.ndarray:
        # The C++ constructor requires a state row; it is only read by the
        # per-particle sample() / probability() entry points, not by the
        # batch entry points. An empty-particles batch call is not expected
        # on the hot path, but we still provide a valid shape to avoid
        # touching uninitialized memory inside C++.
        if particles_arr.shape[0] > 0:
            return particles_arr[0]
        return np.zeros(2 + self.num_rocks, dtype=float)
