"""Vectorized particle belief updaters for the Continuous Light-Dark POMDP.

This module implements concrete
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
subclasses that perform batched state transitions and observation
log-likelihood evaluations for the Continuous Light-Dark environment,
replacing per-particle Python loops with NumPy array operations.

Because the state dimension is always 2, all linear-algebra operations
(Cholesky sampling, Mahalanobis distance, beacon distance) are expanded
into closed-form scalar arithmetic at init time, avoiding per-call
``solve_triangular`` / ``np.linalg.norm`` dispatch overhead.

Three updaters correspond to the three
:class:`~POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp.ObservationModelType`
variants:

Classes:
    ContinuousLightDarkVectorizedUpdater: ``NORMAL_NOISE`` — always
        returns Gaussian log-likelihoods with binary near/far noise.
    ContinuousLightDarkNoObsInDarkVectorizedUpdater:
        ``NORMAL_NOISE_NO_OBS_IN_DARK`` — handles ``"None"``
        observations when far from beacons.
    ContinuousLightDarkDistanceBasedVectorizedUpdater:
        ``DISTANCE_BASED`` — handles ``"None"`` observations when
        beyond beacon radius.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.environments.light_dark_pomdp import (
    _native,  # pylint: disable=no-name-in-module
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

        # Cached covariance arrays for the native batch entry points. The
        # MVN objects' ``covariance`` property returns a copy, so we pay
        # the allocation once instead of on every batch call.
        self._state_transition_cov: np.ndarray = state_transition_dist.covariance
        self._obs_cov_near: np.ndarray = obs_dist_near_beacon.covariance
        self._obs_cov_far: np.ndarray = obs_dist_far_from_beacon.covariance

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
        # Delegate to the native C++ batch sampler so both this path and the
        # per-particle ContinuousLightDarkStateTransitionModel.sample()
        # share the same C++ RNG (closes the cross-path divergence that
        # used to fail the equivalence tests).
        action_vec = self._resolve_action(action)
        transition = _native.ContinuousLightDarkTransitionCpp(
            state=particles[0],
            action=action_vec,
            covariance=self._state_transition_cov,
        )
        return transition.batch_sample(particles)

    def batch_observation_log_likelihood(
        self, next_particles: np.ndarray, action: np.ndarray, observation: np.ndarray
    ) -> np.ndarray:
        # Delegate to the native C++ observation log-likelihood. The
        # per-row near/far decision is made inside C++ using the same
        # beacon-distance test as the per-particle model.
        observation_arr = np.asarray(observation, dtype=float).ravel()
        action_vec = self._resolve_action(action)
        obs_model = _native.ContinuousLightDarkObservationCpp(
            next_state=next_particles[0],
            action=action_vec,
            covariance_near=self._obs_cov_near,
            covariance_far=self._obs_cov_far,
            beacons=self.beacons,
            beacon_radius=float(self.beacon_radius),
            grid_size=float(self.grid_size),
        )
        return obs_model.batch_log_likelihood(next_particles, observation_arr)

    @property
    def config_id(self) -> str:
        return config_to_id(self._build_config_dict("ContinuousLightDarkVectorizedUpdater"))

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

    # ------------------------------------------------------------------
    # Config-id helpers used by subclasses
    # ------------------------------------------------------------------

    def _build_config_dict(self, class_name: str) -> dict:
        config_dict: dict = {
            "class": class_name,
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
        return config_dict


class ContinuousLightDarkNoObsInDarkVectorizedUpdater(
    ContinuousLightDarkVectorizedUpdater,
):
    """Vectorized updater for the ``NORMAL_NOISE_NO_OBS_IN_DARK`` observation model.

    Particles far from all beacons receive ``"None"`` observations
    (log-likelihood 0 for ``"None"``, ``-inf`` for any array observation).
    Particles near a beacon use the near-beacon Gaussian distribution.

    Inherits transition logic and all precomputation from
    :class:`ContinuousLightDarkVectorizedUpdater`.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        ...     ContinuousLightDarkPOMDP, ObservationModelType,
        ... )
        >>> env = ContinuousLightDarkPOMDP(
        ...     discount_factor=0.95,
        ...     observation_model_type=ObservationModelType.NORMAL_NOISE_NO_OBS_IN_DARK,
        ... )
        >>> updater = ContinuousLightDarkNoObsInDarkVectorizedUpdater.from_environment(env)
        >>> particles = np.random.rand(50, 2) * 10
        >>> action = np.array([1.0, 0.0])
        >>> next_p = updater.batch_transition(particles, action)
        >>> ll = updater.batch_observation_log_likelihood(next_p, action, "None")
        >>> ll.shape
        (50,)
    """

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,  # noqa: ARG002
        observation: Union[np.ndarray, str],
    ) -> np.ndarray:
        px = next_particles[:, 0]
        py = next_particles[:, 1]
        near_mask = self._batch_near_beacon_scalar(px, py)

        if isinstance(observation, str):
            return np.where(near_mask, -np.inf, 0.0)

        # For array observations, use the active distribution (near or far)
        # matching the non-vectorized model's behavior.
        return self._log_lik_near_far(px, py, near_mask, observation)

    @property
    def config_id(self) -> str:
        return config_to_id(
            self._build_config_dict("ContinuousLightDarkNoObsInDarkVectorizedUpdater")
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_lik_near_far(
        self,
        px: np.ndarray,
        py: np.ndarray,
        near_mask: np.ndarray,
        observation: np.ndarray,
    ) -> np.ndarray:
        obs = np.asarray(observation, dtype=float).ravel()
        dx = px - obs[0]
        dy = py - obs[1]
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


class ContinuousLightDarkDistanceBasedVectorizedUpdater(
    ContinuousLightDarkVectorizedUpdater,
):
    """Vectorized updater for the ``DISTANCE_BASED`` observation model.

    Unlike :class:`ContinuousLightDarkNoObsInDarkVectorizedUpdater`,
    this model assigns probability 0 (``-inf`` in log space) to array
    observations when the particle is beyond ``beacon_radius`` from all
    beacons, matching the non-vectorized
    :class:`ContinuousLightDarkDistanceBasedObservationModel` behaviour.

    Inherits transition logic from
    :class:`ContinuousLightDarkVectorizedUpdater`.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        ...     ContinuousLightDarkPOMDP, ObservationModelType,
        ... )
        >>> env = ContinuousLightDarkPOMDP(
        ...     discount_factor=0.95,
        ...     observation_model_type=ObservationModelType.DISTANCE_BASED,
        ... )
        >>> updater = ContinuousLightDarkDistanceBasedVectorizedUpdater.from_environment(env)
        >>> particles = np.random.rand(50, 2) * 10
        >>> action = np.array([1.0, 0.0])
        >>> next_p = updater.batch_transition(particles, action)
        >>> ll = updater.batch_observation_log_likelihood(next_p, action, "None")
        >>> ll.shape
        (50,)
    """

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,  # noqa: ARG002
        observation: Union[np.ndarray, str],
    ) -> np.ndarray:
        px = next_particles[:, 0]
        py = next_particles[:, 1]
        near_mask = self._batch_near_beacon_scalar(px, py)

        if isinstance(observation, str):
            return np.where(near_mask, -np.inf, 0.0)

        # Array observations: near-beacon uses near dist, far returns -inf
        obs = np.asarray(observation, dtype=float).ravel()
        dx = px - obs[0]
        dy = py - obs[1]
        maha_near = (
            self._obs_near_P00 * dx**2 + self._obs_near_2P01 * dx * dy + self._obs_near_P11 * dy**2
        )
        return np.where(
            near_mask,
            self._obs_near_log_norm - 0.5 * maha_near,
            -np.inf,
        )

    @property
    def config_id(self) -> str:
        return config_to_id(
            self._build_config_dict("ContinuousLightDarkDistanceBasedVectorizedUpdater")
        )
