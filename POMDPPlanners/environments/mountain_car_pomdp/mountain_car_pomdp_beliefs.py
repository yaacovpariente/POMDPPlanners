"""Vectorized particle belief updater for the Mountain Car POMDP.

This module implements a concrete
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
that performs batched state transitions and observation log-likelihood
evaluations for the Mountain Car environment, replacing per-particle
Python loops with NumPy array operations.

Classes:
    MountainCarVectorizedUpdater: Batched updater for the Mountain Car POMDP.

Functions:
    create_mountain_car_belief: Factory producing a configured belief for MountainCarPOMDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from POMDPPlanners.core.belief.belief_utils import get_initial_belief
from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp_gaussian_beliefs import (
    GaussianBeliefUpdaterType,
    create_mountain_car_gaussian_belief,
)
from POMDPPlanners.utils.belief_factory import BeliefType
from POMDPPlanners.utils.config_to_id import config_to_id
from POMDPPlanners.utils.multivariate_normal import (
    CovarianceParameterizedMultivariateNormal,
)

if TYPE_CHECKING:
    from POMDPPlanners.core.belief.base_belief import Belief
    from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP


class MountainCarVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Vectorized particle belief updater for the Mountain Car POMDP.

    Performs all-particle transitions and observation log-likelihood
    evaluations using vectorized NumPy operations, replacing per-particle
    Python loops with batched array operations.

    ``batch_transition`` applies the deterministic cart physics to all
    particles, then adds a per-particle Gaussian process-noise sample drawn
    from ``state_transition_dist`` (mirroring
    :meth:`MountainCarTransition.sample`), and finally re-applies the
    position/velocity clipping and wall-stop boundary rule. Observations
    follow a single Gaussian centred on the true state.

    Attributes:
        state_transition_dist: Process-noise distribution added after the
            deterministic physics step.
        obs_dist: Observation noise distribution.
        power: Engine power scaling factor.
        gravity: Gravitational force constant.
        max_speed: Maximum velocity magnitude.
        min_position: Minimum position boundary.
        max_position: Maximum position boundary.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
        >>> env = MountainCarPOMDP(discount_factor=0.99)
        >>> updater = MountainCarVectorizedUpdater.from_environment(env)
        >>> particles = np.column_stack([
        ...     np.random.uniform(-0.6, -0.4, 50),
        ...     np.zeros(50),
        ... ])
        >>> action = 1
        >>> next_p = updater.batch_transition(particles, action)
        >>> next_p.shape
        (50, 2)
        >>> obs = np.array([-0.5, 0.0])
        >>> ll = updater.batch_observation_log_likelihood(next_p, action, obs)
        >>> ll.shape
        (50,)
    """

    def __init__(
        self,
        state_transition_dist: CovarianceParameterizedMultivariateNormal,
        obs_dist: CovarianceParameterizedMultivariateNormal,
        power: float,
        gravity: float,
        max_speed: float,
        min_position: float,
        max_position: float,
    ):
        """Initialize the vectorized updater.

        Args:
            state_transition_dist: Pre-built MVN for process noise added on
                top of the deterministic next-state physics.
            obs_dist: Pre-built MVN for observation noise.
            power: Engine power scaling factor.
            gravity: Gravitational force constant.
            max_speed: Maximum velocity magnitude.
            min_position: Minimum position boundary.
            max_position: Maximum position boundary.
        """
        self.state_transition_dist = state_transition_dist
        self.obs_dist = obs_dist
        self.power = power
        self.gravity = gravity
        self.max_speed = max_speed
        self.min_position = min_position
        self.max_position = max_position

    @classmethod
    def from_environment(cls, env: "MountainCarPOMDP") -> "MountainCarVectorizedUpdater":
        """Construct an updater from a MountainCarPOMDP instance.

        Args:
            env: Environment to extract parameters from.

        Returns:
            A new ``MountainCarVectorizedUpdater`` instance.
        """
        # pylint: disable=protected-access
        return cls(
            state_transition_dist=env._state_transition_dist,
            obs_dist=env._obs_dist,
            power=env.power,
            gravity=env.gravity,
            max_speed=env.max_speed,
            min_position=env.min_position,
            max_position=env.max_position,
        )
        # pylint: enable=protected-access

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        # particles: (N, 2) = [position, velocity]
        position, velocity = self._deterministic_next_state(particles, action)
        position, velocity = self._add_transition_noise(position, velocity, particles.shape[0])
        position, velocity = self._apply_clipping_and_wall_stop(position, velocity)
        return np.column_stack([position, velocity])

    def _deterministic_next_state(
        self, particles: np.ndarray, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        position = particles[:, 0]
        velocity = particles[:, 1]
        velocity = velocity + action * self.power + np.cos(3.0 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position = position + velocity
        return self._apply_clipping_and_wall_stop(position, velocity)

    def _add_transition_noise(
        self, position: np.ndarray, velocity: np.ndarray, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        noise = self.state_transition_dist.sample(np.zeros(2), n_samples=n)
        return position + noise[:, 0], velocity + noise[:, 1]

    def _apply_clipping_and_wall_stop(
        self, position: np.ndarray, velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position = np.clip(position, self.min_position, self.max_position)
        at_min = position == self.min_position
        velocity = np.where(at_min & (velocity < 0), 0.0, velocity)
        return position, velocity

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> np.ndarray:
        observation = np.asarray(observation, dtype=float).ravel()
        return self.obs_dist.log_pdf(next_particles, observation)

    @property
    def config_id(self) -> str:
        config_dict = {
            "class": "MountainCarVectorizedUpdater",
            "state_transition_cov": self.state_transition_dist.covariance.tolist(),
            "obs_cov": self.obs_dist.covariance.tolist(),
            "power": self.power,
            "gravity": self.gravity,
            "max_speed": self.max_speed,
            "min_position": self.min_position,
            "max_position": self.max_position,
        }
        return config_to_id(config_dict)


# ---------------------------------------------------------------------------
# Belief factory
# ---------------------------------------------------------------------------

_SUPPORTED_TYPES = {BeliefType.PARTICLE, BeliefType.VECTORIZED_PARTICLE, BeliefType.GAUSSIAN}
_DEFAULT_TYPE = BeliefType.VECTORIZED_PARTICLE


def create_mountain_car_belief(
    env: "MountainCarPOMDP",
    belief_type: BeliefType = _DEFAULT_TYPE,
    n_particles: int = 200,
    **kwargs: Any,
) -> "Belief":
    """Create a ready-to-use belief for the Mountain Car POMDP.

    For ``BeliefType.GAUSSIAN``, the following keyword arguments are
    forwarded to
    :func:`create_mountain_car_gaussian_belief`:

    - ``updater_type`` (:class:`GaussianBeliefUpdaterType`): defaults to
      ``GaussianBeliefUpdaterType.UKF``.
    - ``initial_covariance`` (``np.ndarray``): defaults to
      ``np.diag([0.2**2 / 12, 1e-4])``.
    - ``process_noise_scale`` (``float``): defaults to ``1e-4``.

    Args:
        env: MountainCarPOMDP environment instance.
        belief_type: Desired belief representation.
            Defaults to ``BeliefType.VECTORIZED_PARTICLE``.
        n_particles: Number of particles (ignored for GAUSSIAN).
            Defaults to 200.
        **kwargs: Extra arguments forwarded to the Gaussian factory.

    Returns:
        A configured :class:`Belief` object.

    Raises:
        ValueError: If *belief_type* is not supported.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
        >>> env = MountainCarPOMDP(discount_factor=0.99)
        >>> belief = create_mountain_car_belief(env, n_particles=50)
        >>> belief.sample().shape
        (2,)
    """
    if belief_type not in _SUPPORTED_TYPES:
        raise ValueError(
            f"MountainCarPOMDP does not support {belief_type}. " f"Supported: {_SUPPORTED_TYPES}"
        )
    if belief_type == BeliefType.PARTICLE:
        return _create_particle_belief(env, n_particles)
    if belief_type == BeliefType.GAUSSIAN:
        return _create_gaussian_belief(env, **kwargs)
    return _create_vectorized_belief(env, n_particles)


def _create_particle_belief(env: "MountainCarPOMDP", n_particles: int) -> "Belief":
    return get_initial_belief(env, n_particles)


def _create_gaussian_belief(env: "MountainCarPOMDP", **kwargs: Any) -> "Belief":
    updater_type = kwargs.pop("updater_type", GaussianBeliefUpdaterType.UKF)
    initial_covariance = kwargs.pop("initial_covariance", None)
    process_noise_scale = kwargs.pop("process_noise_scale", 1e-4)
    return create_mountain_car_gaussian_belief(
        env=env,
        updater_type=updater_type,
        initial_covariance=initial_covariance,
        process_noise_scale=process_noise_scale,
    )


def _create_vectorized_belief(env: "MountainCarPOMDP", n_particles: int) -> "Belief":
    updater = MountainCarVectorizedUpdater.from_environment(env)
    particles = np.array(env.initial_state_dist().sample(n_samples=n_particles))
    log_weights = np.log(np.ones(n_particles) / n_particles)
    return VectorizedWeightedParticleBelief(particles, log_weights, updater)
