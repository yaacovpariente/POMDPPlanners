# SPDX-License-Identifier: MIT

"""Vectorized particle belief updater for the Safety Ant Velocity POMDP.

This module implements a concrete
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
that performs batched state transitions and observation log-likelihood
evaluations for the Safety Ant Velocity environment by delegating to the
native C++ ``_native`` extension's batch entry points.

Classes:
    SafetyAntVelocityVectorizedUpdater: Batched updater for the
        Safety Ant Velocity POMDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.environments.safety_ant_velocity_pomdp import _native
from POMDPPlanners.utils.config_to_id import config_to_id
from POMDPPlanners.utils.multivariate_normal import (
    CovarianceParameterizedMultivariateNormal,
)

if TYPE_CHECKING:
    from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_pomdp import (
        SafeAntVelocityPOMDP,
    )


class SafetyAntVelocityVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Vectorized particle belief updater for the Safety Ant Velocity POMDP.

    Performs all-particle transitions and observation log-likelihood
    evaluations by delegating to the native ``_native`` C++ extension's
    batch entry points (``SafeAntVelocityTransitionCpp.batch_sample`` and
    ``SafeAntVelocityObservationCpp.batch_log_likelihood``).

    The transition is stochastic because force direction is uniformly
    random, so ``batch_transition`` samples N independent force directions
    for N particles (drawn from the module-level C++ RNG). Observations
    follow a diagonal Gaussian constructed from the environment's position
    and velocity noise parameters.

    Attributes:
        obs_dist: Observation noise distribution (4-D diagonal Gaussian).
        dt: Integration time step.
        mass: Agent mass.
        damping: Damping coefficient opposing velocity.
        max_force: Maximum force magnitude.
        force_scales: Force scaling factors for each discrete action.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.safety_ant_velocity_pomdp import (
        ...     SafeAntVelocityPOMDP,
        ...     _native,
        ... )
        >>> _native.set_seed(42)
        >>> env = SafeAntVelocityPOMDP(discount_factor=0.99)
        >>> updater = SafetyAntVelocityVectorizedUpdater.from_environment(env)
        >>> particles = np.column_stack([
        ...     np.random.uniform(-1, 1, (50, 2)),
        ...     np.zeros((50, 2)),
        ... ])
        >>> action = 2
        >>> next_p = updater.batch_transition(particles, action)
        >>> next_p.shape
        (50, 4)
        >>> obs = np.array([0.0, 0.0, 0.0, 0.0])
        >>> ll = updater.batch_observation_log_likelihood(next_p, action, obs)
        >>> ll.shape
        (50,)
    """

    def __init__(
        self,
        obs_dist: CovarianceParameterizedMultivariateNormal,
        dt: float,
        mass: float,
        damping: float,
        max_force: float,
        force_scales: np.ndarray,
    ):
        """Initialize the vectorized updater.

        Args:
            obs_dist: Pre-built MVN for observation noise.
            dt: Integration time step.
            mass: Agent mass.
            damping: Damping coefficient.
            max_force: Maximum force magnitude.
            force_scales: Array of force scaling factors per discrete action.
        """
        self.obs_dist = obs_dist
        self.dt = dt
        self.mass = mass
        self.damping = damping
        self.max_force = max_force
        self.force_scales = np.asarray(force_scales, dtype=float)
        self._obs_covariance = np.asarray(obs_dist.covariance, dtype=float)
        # Per-action kernel caches: same pattern as the env wrapper. Avoids
        # reconstructing the C++ kernel (Cholesky factor + parameter unpack)
        # on every batch_transition / batch_observation_log_likelihood call.
        self._trans_kernel_cache: Dict[int, Any] = {}
        self._obs_kernel_cache: Dict[int, Any] = {}

    @classmethod
    def from_environment(cls, env: "SafeAntVelocityPOMDP") -> "SafetyAntVelocityVectorizedUpdater":
        """Construct an updater from a SafeAntVelocityPOMDP instance.

        Args:
            env: Environment to extract parameters from.

        Returns:
            A new ``SafetyAntVelocityVectorizedUpdater`` instance.
        """
        cov = np.diag(
            [
                env.position_noise**2,
                env.position_noise**2,
                env.velocity_noise**2,
                env.velocity_noise**2,
            ]
        )
        obs_dist = CovarianceParameterizedMultivariateNormal(cov)
        return cls(
            obs_dist=obs_dist,
            dt=env.dt,
            mass=env.mass,
            damping=env.damping,
            max_force=env.max_force,
            force_scales=np.array([0.0, 0.33, 0.67, 1.0]),
        )

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        # Drop unpicklable pybind11 kernel handles before serialization;
        # __setstate__ recreates the empty caches and they re-fill lazily.
        state = self.__dict__.copy()
        state["_trans_kernel_cache"] = {}
        state["_obs_kernel_cache"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        vars(self).update(state)
        self._trans_kernel_cache = {}
        self._obs_kernel_cache = {}

    def _get_trans_kernel(self, action: int) -> Any:
        kernel = self._trans_kernel_cache.get(action)
        if kernel is None:
            kernel = _native.SafeAntVelocityTransitionCpp(
                state=np.zeros(4, dtype=np.float64),
                action=action,
                dt=self.dt,
                mass=self.mass,
                damping=self.damping,
                max_force=self.max_force,
                force_scales=self.force_scales,
            )
            self._trans_kernel_cache[action] = kernel
        return kernel

    def _get_obs_kernel(self, action: int) -> Any:
        kernel = self._obs_kernel_cache.get(action)
        if kernel is None:
            kernel = _native.SafeAntVelocityObservationCpp(
                next_state=np.zeros(4, dtype=np.float64),
                action=action,
                covariance=self._obs_covariance,
            )
            self._obs_kernel_cache[action] = kernel
        return kernel

    def batch_transition(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        # batch_sample reads per-row state from the input array, so the
        # kernel's stored state is irrelevant — no set_state needed.
        kernel = self._get_trans_kernel(int(action))
        return kernel.batch_sample(particles)

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> np.ndarray:
        observation_arr = np.asarray(observation, dtype=float).ravel()
        # batch_log_likelihood reads per-row from next_particles; no
        # set_next_state needed.
        kernel = self._get_obs_kernel(int(action))
        return kernel.batch_log_likelihood(next_particles, observation_arr)

    @property
    def config_id(self) -> str:
        config_dict = {
            "class": "SafetyAntVelocityVectorizedUpdater",
            "obs_cov": self.obs_dist.covariance.tolist(),
            "dt": self.dt,
            "mass": self.mass,
            "damping": self.damping,
            "max_force": self.max_force,
            "force_scales": self.force_scales.tolist(),
        }
        return config_to_id(config_dict)
