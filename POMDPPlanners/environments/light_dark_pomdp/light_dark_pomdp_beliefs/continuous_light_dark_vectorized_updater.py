"""Vectorized particle belief updater for the Continuous Light-Dark POMDP.

This module implements a concrete
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
that performs batched state transitions and observation log-likelihood
evaluations for the Continuous Light-Dark environment, replacing
per-particle Python loops with NumPy array operations.

Because the state dimension is always 2, all linear-algebra operations
(Cholesky sampling, Mahalanobis distance, beacon distance) are expanded
into closed-form scalar arithmetic at init time, avoiding per-call
``solve_triangular`` / ``np.linalg.norm`` dispatch overhead.

Classes:
    ContinuousLightDarkVectorizedUpdater: Batched updater for the
        Continuous Light-Dark POMDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.core.environment import SpaceType
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
        action_to_vector: Optional[Dict[str, np.ndarray]] = None,
    ):
        """Initialize the vectorized updater.

        Args:
            state_transition_dist: Pre-built MVN for transition noise.
            obs_dist_near_beacon: Pre-built MVN for near-beacon observations.
            obs_dist_far_from_beacon: Pre-built MVN for far-from-beacon observations.
            beacons: Beacon positions as a (2, n_beacons) array.
            beacon_radius: Distance threshold for near-beacon classification.
            grid_size: Grid boundary used by the environment.
            action_to_vector: Optional mapping from string action names to
                numeric vectors. When provided, string actions are converted
                before the batch transition.
        """
        self.state_transition_dist = state_transition_dist
        self.obs_dist_near_beacon = obs_dist_near_beacon
        self.obs_dist_far_from_beacon = obs_dist_far_from_beacon
        self.beacons = np.asarray(beacons, dtype=float)
        self.beacon_radius = beacon_radius
        self.beacon_radius_sq = beacon_radius * beacon_radius
        self.grid_size = grid_size
        self._action_to_vector = action_to_vector

        self._precompute_cholesky_scalars(state_transition_dist)
        self._precompute_precision_scalars(obs_dist_near_beacon, obs_dist_far_from_beacon)
        self._precompute_beacon_arrays()

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
        has_discrete_actions = (
            hasattr(env, "space_info") and env.space_info.action_space == SpaceType.DISCRETE
        )
        action_map = getattr(env, "action_to_vector", None) if has_discrete_actions else None
        return cls(
            state_transition_dist=env._state_transition_dist,
            obs_dist_near_beacon=env._obs_dist_near_beacon,
            obs_dist_far_from_beacon=env._obs_dist_far_from_beacon,
            beacons=env.beacons,
            beacon_radius=env.beacon_radius,
            grid_size=env.grid_size,
            action_to_vector=action_map,
        )
        # pylint: enable=protected-access

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        action = self._resolve_action(action)
        n = particles.shape[0]
        z = np.random.standard_normal((n, 2))
        noise_x = z[:, 0] * self._trans_L00
        noise_y = z[:, 0] * self._trans_L10 + z[:, 1] * self._trans_L11
        result = particles + action
        result[:, 0] += noise_x
        result[:, 1] += noise_y
        return result

    def batch_observation_log_likelihood(
        self, next_particles: np.ndarray, action: np.ndarray, observation: np.ndarray
    ) -> np.ndarray:
        observation = np.asarray(observation, dtype=float).ravel()
        px = next_particles[:, 0]
        py = next_particles[:, 1]

        near_mask = self._batch_near_beacon_scalar(px, py)
        dx = px - observation[0]
        dy = py - observation[1]

        maha_near = (
            self._obs_near_P00 * dx**2 + self._obs_near_2P01 * dx * dy + self._obs_near_P11 * dy**2
        )
        maha_far = (
            self._obs_far_P00 * dx**2 + self._obs_far_2P01 * dx * dy + self._obs_far_P11 * dy**2
        )

        return np.where(
            near_mask,
            self._obs_near_log_norm - 0.5 * maha_near,
            self._obs_far_log_norm - 0.5 * maha_far,
        )

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
        if self._action_to_vector is not None:
            config_dict["action_to_vector"] = {
                k: v.tolist() for k, v in self._action_to_vector.items()
            }
        return config_to_id(config_dict)

    # ------------------------------------------------------------------
    # Private helpers — pre-computation
    # ------------------------------------------------------------------

    def _precompute_cholesky_scalars(self, dist):
        L = dist._cholesky_L  # pylint: disable=protected-access
        self._trans_L00 = float(L[0, 0])
        self._trans_L10 = float(L[1, 0])
        self._trans_L11 = float(L[1, 1])

    def _precompute_precision_scalars(self, near_dist, far_dist):
        P_near = np.linalg.inv(near_dist.covariance)
        self._obs_near_P00 = float(P_near[0, 0])
        self._obs_near_2P01 = float(2.0 * P_near[0, 1])
        self._obs_near_P11 = float(P_near[1, 1])
        self._obs_near_log_norm = float(
            near_dist._log_normalization  # pylint: disable=protected-access
        )

        P_far = np.linalg.inv(far_dist.covariance)
        self._obs_far_P00 = float(P_far[0, 0])
        self._obs_far_2P01 = float(2.0 * P_far[0, 1])
        self._obs_far_P11 = float(P_far[1, 1])
        self._obs_far_log_norm = float(
            far_dist._log_normalization  # pylint: disable=protected-access
        )

    def _precompute_beacon_arrays(self):
        self._beacon_x = np.ascontiguousarray(self.beacons[0])
        self._beacon_y = np.ascontiguousarray(self.beacons[1])

    # ------------------------------------------------------------------
    # Private helpers — action resolution
    # ------------------------------------------------------------------

    def _resolve_action(self, action) -> np.ndarray:
        if self._action_to_vector is not None and isinstance(action, str):
            return np.asarray(self._action_to_vector[action], dtype=float).ravel()
        return np.asarray(action, dtype=float).ravel()

    # ------------------------------------------------------------------
    # Private helpers — runtime
    # ------------------------------------------------------------------

    def _batch_near_beacon_scalar(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        bx = px[:, np.newaxis] - self._beacon_x
        by = py[:, np.newaxis] - self._beacon_y
        sq_distances = bx**2 + by**2
        return sq_distances.min(axis=1) <= self.beacon_radius_sq
