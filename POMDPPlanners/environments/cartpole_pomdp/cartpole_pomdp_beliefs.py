# SPDX-License-Identifier: MIT

"""Vectorized particle belief updater for the CartPole POMDP.

This module implements a concrete
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
that performs batched state transitions and observation log-likelihood
evaluations for the CartPole environment, replacing per-particle Python
loops with NumPy array operations.

Classes:
    CartPoleVectorizedUpdater: Batched updater for the CartPole POMDP.

Functions:
    create_cartpole_belief: Factory producing a configured belief for CartPolePOMDP.
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
from POMDPPlanners.environments.cartpole_pomdp import _native
from POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp_gaussian_beliefs import (
    GaussianBeliefUpdaterType,
    create_cartpole_gaussian_belief,
)
from POMDPPlanners.utils.belief_factory import BeliefType
from POMDPPlanners.utils.config_to_id import config_to_id
from POMDPPlanners.utils.multivariate_normal import (
    CovarianceParameterizedMultivariateNormal,
)

if TYPE_CHECKING:
    from POMDPPlanners.core.belief.base_belief import Belief
    from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP


class CartPoleVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Vectorized particle belief updater for the CartPole POMDP.

    Performs all-particle transitions and observation log-likelihood
    evaluations using vectorized NumPy operations, replacing per-particle
    Python loops with batched array operations.

    ``batch_transition`` applies the deterministic cart-pole physics to
    all particles and then adds a per-particle Gaussian process-noise
    sample drawn from ``state_transition_dist`` (mirroring
    :meth:`CartPoleTransition.sample`).  Observations follow a single
    Gaussian centred on the true state.

    Attributes:
        state_transition_dist: Process-noise distribution added after the
            deterministic physics step.
        obs_dist: Observation noise distribution.
        force_mag: Magnitude of force applied to the cart.
        gravity: Gravitational acceleration constant.
        masscart: Mass of the cart.
        masspole: Mass of the pole.
        total_mass: Combined mass of cart and pole.
        length: Half the pole's length.
        polemass_length: Pole mass times pole half-length.
        tau: Integration time step.
        kinematics_integrator: Integration method ("euler" or "semi-implicit euler").

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
        >>> noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
        >>> env = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov)
        >>> updater = CartPoleVectorizedUpdater.from_environment(env)
        >>> particles = np.random.uniform(-0.05, 0.05, (50, 4))
        >>> action = 1
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
        state_transition_dist: CovarianceParameterizedMultivariateNormal,
        obs_dist: CovarianceParameterizedMultivariateNormal,
        force_mag: float,
        gravity: float,
        masscart: float,
        masspole: float,
        total_mass: float,
        length: float,
        polemass_length: float,
        tau: float,
        kinematics_integrator: str,
    ):
        """Initialize the vectorized updater.

        Args:
            state_transition_dist: Pre-built MVN for process noise added on
                top of the deterministic next-state physics.
            obs_dist: Pre-built MVN for observation noise.
            force_mag: Magnitude of applied force.
            gravity: Gravitational acceleration constant.
            masscart: Mass of the cart.
            masspole: Mass of the pole.
            total_mass: Combined mass of cart and pole.
            length: Half the pole's length.
            polemass_length: Pole mass times pole half-length.
            tau: Integration time step.
            kinematics_integrator: Integration method.
        """
        self.state_transition_dist = state_transition_dist
        self.obs_dist = obs_dist
        self.force_mag = force_mag
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = total_mass
        self.length = length
        self.polemass_length = polemass_length
        self.tau = tau
        self.kinematics_integrator = kinematics_integrator
        # Cached covariance arrays for the native batch entry points. The
        # MVN objects' ``covariance`` property returns a copy, so we pay the
        # allocation once instead of on every batch_transition call.
        self._state_transition_cov: np.ndarray = state_transition_dist.covariance
        self._obs_cov: np.ndarray = obs_dist.covariance

    @classmethod
    def from_environment(cls, env: "CartPolePOMDP") -> "CartPoleVectorizedUpdater":
        """Construct an updater from a CartPolePOMDP instance.

        Args:
            env: Environment to extract parameters from.

        Returns:
            A new ``CartPoleVectorizedUpdater`` instance.
        """
        # pylint: disable=protected-access
        return cls(
            state_transition_dist=env._state_transition_dist,
            obs_dist=env._obs_dist,
            force_mag=env.force_mag,
            gravity=env.gravity,
            masscart=env.masscart,
            masspole=env.masspole,
            total_mass=env.total_mass,
            length=env.length,
            polemass_length=env.polemass_length,
            tau=env.tau,
            kinematics_integrator=env.kinematics_integrator,
        )
        # pylint: enable=protected-access

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        # particles: (N, 4) = [x, x_dot, theta, theta_dot]. Delegates to the
        # native C++ batch sampler so both this path and the per-particle
        # env.sample_next_state() (which also routes through
        # _native.CartPoleTransitionCpp) share the same C++ RNG. The
        # `state=particles[0]` passed to the ctor is unused on the batch
        # path; only the ctor signature requires it.
        transition = _native.CartPoleTransitionCpp(
            state=particles[0],
            action=int(action),
            force_mag=self.force_mag,
            total_mass=self.total_mass,
            polemass_length=self.polemass_length,
            gravity=self.gravity,
            length=self.length,
            kinematics_integrator=self.kinematics_integrator,
            tau=self.tau,
            masspole=self.masspole,
            covariance=self._state_transition_cov,
        )
        return transition.batch_sample(particles)

    def _deterministic_next_state(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        # Retained as a public-via-tests helper; mirrors the C++
        # deterministic path so unit tests of physics don't depend on
        # constructing a native object.
        force = self.force_mag if action == 1 else -self.force_mag

        theta = particles[:, 2]
        theta_dot = particles[:, 3]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * cos_theta**2 / self.total_mass)
        )
        x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass

        next_particles = np.empty_like(particles)
        if self.kinematics_integrator == "euler":
            next_particles[:, 0] = particles[:, 0] + self.tau * particles[:, 1]
            next_particles[:, 1] = particles[:, 1] + self.tau * x_acc
            next_particles[:, 2] = particles[:, 2] + self.tau * particles[:, 3]
            next_particles[:, 3] = particles[:, 3] + self.tau * theta_acc
        else:  # semi-implicit euler
            next_particles[:, 1] = particles[:, 1] + self.tau * x_acc
            next_particles[:, 0] = particles[:, 0] + self.tau * next_particles[:, 1]
            next_particles[:, 3] = particles[:, 3] + self.tau * theta_acc
            next_particles[:, 2] = particles[:, 2] + self.tau * next_particles[:, 3]

        return next_particles

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> np.ndarray:
        observation_arr = np.asarray(observation, dtype=float).ravel()
        # Delegate to the native C++ observation likelihood. `next_state`
        # passed to the ctor is unused on the batch path.
        obs_model = _native.CartPoleObservationCpp(
            next_state=next_particles[0],
            action=int(action),
            covariance=self._obs_cov,
        )
        return obs_model.batch_log_likelihood(next_particles, observation_arr)

    @property
    def config_id(self) -> str:
        config_dict = {
            "class": "CartPoleVectorizedUpdater",
            "state_transition_cov": self.state_transition_dist.covariance.tolist(),
            "obs_cov": self.obs_dist.covariance.tolist(),
            "force_mag": self.force_mag,
            "gravity": self.gravity,
            "masscart": self.masscart,
            "masspole": self.masspole,
            "total_mass": self.total_mass,
            "length": self.length,
            "polemass_length": self.polemass_length,
            "tau": self.tau,
            "kinematics_integrator": self.kinematics_integrator,
        }
        return config_to_id(config_dict)


# ---------------------------------------------------------------------------
# Belief factory
# ---------------------------------------------------------------------------

_SUPPORTED_TYPES = {BeliefType.PARTICLE, BeliefType.VECTORIZED_PARTICLE, BeliefType.GAUSSIAN}
_DEFAULT_TYPE = BeliefType.VECTORIZED_PARTICLE


def create_cartpole_belief(
    env: "CartPolePOMDP",
    belief_type: BeliefType = _DEFAULT_TYPE,
    n_particles: int = 200,
    **kwargs: Any,
) -> "Belief":
    """Create a ready-to-use belief for the CartPole POMDP.

    For ``BeliefType.GAUSSIAN``, the following keyword arguments are
    forwarded to
    :func:`create_cartpole_gaussian_belief`:

    - ``updater_type`` (:class:`GaussianBeliefUpdaterType`): defaults to
      ``GaussianBeliefUpdaterType.UKF``.
    - ``initial_covariance`` (``np.ndarray``): defaults to
      ``np.eye(4) * (0.1**2 / 12)``.
    - ``process_noise_scale`` (``float``): defaults to ``1e-4``.

    Args:
        env: CartPolePOMDP environment instance.
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
        >>> from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
        >>> env = CartPolePOMDP(discount_factor=0.99,
        ...                    noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))
        >>> belief = create_cartpole_belief(env, n_particles=50)
        >>> belief.sample().shape
        (4,)
    """
    if belief_type not in _SUPPORTED_TYPES:
        raise ValueError(
            f"CartPolePOMDP does not support {belief_type}. " f"Supported: {_SUPPORTED_TYPES}"
        )
    if belief_type == BeliefType.PARTICLE:
        return _create_particle_belief(env, n_particles)
    if belief_type == BeliefType.GAUSSIAN:
        return _create_gaussian_belief(env, **kwargs)
    return _create_vectorized_belief(env, n_particles)


def _create_particle_belief(env: "CartPolePOMDP", n_particles: int) -> "Belief":
    return get_initial_belief(env, n_particles)


def _create_gaussian_belief(env: "CartPolePOMDP", **kwargs: Any) -> "Belief":
    updater_type = kwargs.pop("updater_type", GaussianBeliefUpdaterType.UKF)
    initial_covariance = kwargs.pop("initial_covariance", None)
    process_noise_scale = kwargs.pop("process_noise_scale", 1e-4)
    return create_cartpole_gaussian_belief(
        env=env,
        updater_type=updater_type,
        initial_covariance=initial_covariance,
        process_noise_scale=process_noise_scale,
    )


def _create_vectorized_belief(env: "CartPolePOMDP", n_particles: int) -> "Belief":
    updater = CartPoleVectorizedUpdater.from_environment(env)
    particles = np.array(env.initial_state_dist().sample(n_samples=n_particles))
    log_weights = np.log(np.ones(n_particles) / n_particles)
    return VectorizedWeightedParticleBelief(particles, log_weights, updater)
