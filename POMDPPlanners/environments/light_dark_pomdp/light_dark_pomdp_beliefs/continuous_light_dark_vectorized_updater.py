"""Vectorized particle belief updater for the Continuous Light-Dark POMDP.

This module implements a concrete
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
that performs batched state transitions and observation log-likelihood
evaluations for the Continuous Light-Dark environment, replacing
per-particle Python loops with NumPy array operations.

Classes:
    ContinuousLightDarkVectorizedUpdater: Batched updater for the
        Continuous Light-Dark POMDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.utils.config_to_id import config_to_id
from POMDPPlanners.utils.multivariate_normal import (
    CovarianceParameterizedMultivariateNormal,
)

if TYPE_CHECKING:
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        ContinuousLightDarkPOMDP,
    )


class ContinuousLightDarkVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Vectorized particle belief updater for the Continuous Light-Dark POMDP.

    Performs all-particle transitions and observation log-likelihood
    evaluations using vectorized NumPy operations, replacing per-particle
    Python loops with batched array operations.

    Attributes:
        state_transition_dist: Noise distribution for state transitions.
        obs_dist_near_beacon: Observation distribution when near a beacon.
        obs_dist_far_from_beacon: Observation distribution when far from beacons.
        beacons: Beacon positions as a (2, n_beacons) array.
        beacon_radius: Distance threshold for beacon proximity.
        grid_size: Grid boundary for the environment.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        ...     ContinuousLightDarkPOMDP,
        ... )
        >>> env = ContinuousLightDarkPOMDP(discount_factor=0.95)
        >>> updater = ContinuousLightDarkVectorizedUpdater.from_environment(env)
        >>> particles = np.random.rand(50, 2) * 10
        >>> action = np.array([1.0, 0.0])
        >>> next_p = updater.batch_transition(particles, action)
        >>> next_p.shape
        (50, 2)
        >>> obs = np.array([5.0, 5.0])
        >>> ll = updater.batch_observation_log_likelihood(next_p, action, obs)
        >>> ll.shape
        (50,)
    """

    def __init__(
        self,
        state_transition_dist: CovarianceParameterizedMultivariateNormal,
        obs_dist_near_beacon: CovarianceParameterizedMultivariateNormal,
        obs_dist_far_from_beacon: CovarianceParameterizedMultivariateNormal,
        beacons: np.ndarray,
        beacon_radius: float,
        grid_size: int,
    ):
        """Initialize the vectorized updater.

        Args:
            state_transition_dist: Pre-built MVN for transition noise.
            obs_dist_near_beacon: Pre-built MVN for near-beacon observations.
            obs_dist_far_from_beacon: Pre-built MVN for far-from-beacon observations.
            beacons: Beacon positions as a (2, n_beacons) array.
            beacon_radius: Distance threshold for near-beacon classification.
            grid_size: Grid boundary used by the environment.
        """
        self.state_transition_dist = state_transition_dist
        self.obs_dist_near_beacon = obs_dist_near_beacon
        self.obs_dist_far_from_beacon = obs_dist_far_from_beacon
        self.beacons = np.asarray(beacons, dtype=float)
        self.beacon_radius = beacon_radius
        self.grid_size = grid_size

    @classmethod
    def from_environment(
        cls, env: "ContinuousLightDarkPOMDP"
    ) -> "ContinuousLightDarkVectorizedUpdater":
        """Construct an updater from a ContinuousLightDarkPOMDP instance.

        Args:
            env: Environment to extract distribution parameters from.

        Returns:
            A new ``ContinuousLightDarkVectorizedUpdater`` instance.
        """
        # pylint: disable=protected-access
        return cls(
            state_transition_dist=env._state_transition_dist,
            obs_dist_near_beacon=env._obs_dist_near_beacon,
            obs_dist_far_from_beacon=env._obs_dist_far_from_beacon,
            beacons=env.beacons,
            beacon_radius=env.beacon_radius,
            grid_size=env.grid_size,
        )
        # pylint: enable=protected-access

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=float).ravel()
        n = particles.shape[0]
        noise = self.state_transition_dist.sample(mean=np.zeros(particles.shape[1]), n_samples=n)
        return particles + action + noise

    def batch_observation_log_likelihood(
        self, next_particles: np.ndarray, action: np.ndarray, observation: np.ndarray
    ) -> np.ndarray:
        observation = np.asarray(observation, dtype=float).ravel()
        near_mask = self._batch_near_beacon(next_particles)
        n = next_particles.shape[0]
        log_likelihoods = np.empty(n)

        near_idx = np.where(near_mask)[0]
        far_idx = np.where(~near_mask)[0]

        if near_idx.size > 0:
            log_likelihoods[near_idx] = self.obs_dist_near_beacon.log_pdf(
                next_particles[near_idx], observation
            )
        if far_idx.size > 0:
            log_likelihoods[far_idx] = self.obs_dist_far_from_beacon.log_pdf(
                next_particles[far_idx], observation
            )

        return log_likelihoods

    @property
    def config_id(self) -> str:
        config_dict = {
            "class": "ContinuousLightDarkVectorizedUpdater",
            "state_transition_cov": self.state_transition_dist.covariance.tolist(),
            "obs_cov_near": self.obs_dist_near_beacon.covariance.tolist(),
            "obs_cov_far": self.obs_dist_far_from_beacon.covariance.tolist(),
            "beacons": self.beacons.tolist(),
            "beacon_radius": self.beacon_radius,
            "grid_size": self.grid_size,
        }
        return config_to_id(config_dict)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _batch_near_beacon(self, particles: np.ndarray) -> np.ndarray:
        # particles: (N, 2), beacons: (2, B)
        # diff: (N, 2, B)
        diff = particles[:, :, np.newaxis] - self.beacons[np.newaxis, :, :]
        # distances: (N, B)
        distances = np.linalg.norm(diff, axis=1)
        # min distance per particle: (N,)
        min_dist = distances.min(axis=1)
        return min_dist <= self.beacon_radius
