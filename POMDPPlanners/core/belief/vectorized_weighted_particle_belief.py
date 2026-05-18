"""Vectorized weighted particle belief state representation.

This module provides a weighted particle filter that stores particles as a
NumPy array and delegates updates to a
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`,
eliminating Python-level loops over individual particles.

Classes:
    VectorizedWeightedParticleBelief: Vectorized weighted particle filter.
"""

from typing import Any, Optional

import numpy as np

from POMDPPlanners.core.belief.base_belief import Belief
from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.utils.config_to_id import config_to_id


class VectorizedWeightedParticleBelief(Belief):
    """Vectorized weighted particle filter for POMDP belief states.

    Stores particles as a 2-D NumPy array of shape ``(N, d)`` and performs
    all update operations via a
    :class:`VectorizedParticleBeliefUpdater`, so the entire
    predict-reweight-resample cycle runs without Python loops over particles.

    Attributes:
        particles: Particle array of shape (N, d).
        log_weights: Log-weights of shape (N,).
        normalized_weights: Probability weights of shape (N,).
        updater: Vectorized updater instance.
        resampling: Whether automatic ESS-based resampling is enabled.
        ess_factor: Fraction of N used as the ESS threshold.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>>
        >>> # Create a trivial identity updater for demonstration
        >>> from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
        ...     VectorizedParticleBeliefUpdater,
        ... )
        >>> class IdentityUpdater(VectorizedParticleBeliefUpdater):
        ...     def batch_transition(self, particles, action):
        ...         return particles + action
        ...     def batch_observation_log_likelihood(self, next_particles, action, observation):
        ...         return np.zeros(len(next_particles))
        ...     @property
        ...     def config_id(self):
        ...         return "identity"
        >>>
        >>> particles = np.random.randn(20, 2)
        >>> log_w = np.log(np.ones(20) / 20)
        >>> belief = VectorizedWeightedParticleBelief(
        ...     particles=particles,
        ...     log_weights=log_w,
        ...     updater=IdentityUpdater(),
        ...     resampling=True,
        ... )
        >>> state = belief.sample()
        >>> state.shape
        (2,)
        >>> new_belief = belief.update(
        ...     action=np.array([1.0, 0.0]),
        ...     observation=np.array([0.5, 0.5]),
        ...     pomdp=None,
        ... )
        >>> new_belief.particles.shape
        (20, 2)
    """

    def __init__(
        self,
        particles: np.ndarray,
        log_weights: np.ndarray,
        updater: VectorizedParticleBeliefUpdater,
        resampling: bool = False,
        ess_factor: float = 0.5,
    ):
        """Initialize vectorized weighted particle belief.

        Args:
            particles: Array of shape (N, d).
            log_weights: Array of shape (N,).
            updater: A :class:`VectorizedParticleBeliefUpdater` instance.
            resampling: Enable ESS-based resampling. Defaults to False.
            ess_factor: ESS threshold as a fraction of N. Defaults to 0.5.

        Raises:
            TypeError: If particles or log_weights are not numpy arrays.
            ValueError: If shapes are inconsistent or weights are invalid.
        """
        self._validate_init_args(particles, log_weights, resampling)

        self.particles = particles
        self.log_weights = log_weights
        self.normalized_weights = self._normalize(log_weights)
        self._cumulative_weights = np.cumsum(self.normalized_weights)
        self.updater = updater
        self.resampling = resampling
        self.ess_factor = ess_factor
        self.ess_threshold = len(particles) * ess_factor

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sample(self) -> np.ndarray:
        """Sample a state from the belief.

        Returns:
            A state vector of shape (d,).
        """
        idx = np.searchsorted(self._cumulative_weights, np.random.random())
        return self.particles[idx]

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Optional[Environment] = None,
        state: Optional[Any] = None,
    ) -> "VectorizedWeightedParticleBelief":
        """Update belief using the vectorized updater.

        Args:
            action: Action that was executed.
            observation: Observation that was received.
            pomdp: Unused. Kept for interface compatibility.
            state: Ignored.

        Returns:
            New VectorizedWeightedParticleBelief with updated particles and weights.
        """
        observation = np.asarray(observation, dtype=float)

        next_particles = self.updater.batch_transition(self.particles, action)
        log_likelihoods = self.updater.batch_observation_log_likelihood(
            next_particles, action, observation
        )
        next_log_weights = self.log_weights + log_likelihoods

        if self.resampling:
            next_particles, next_log_weights = self._resample(next_particles, next_log_weights)

        return VectorizedWeightedParticleBelief(
            particles=next_particles,
            log_weights=next_log_weights,
            updater=self.updater,
            resampling=self.resampling,
            ess_factor=self.ess_factor,
        )

    def to_unique_support_distribution(self) -> DiscreteDistribution:
        """Convert the belief to a DiscreteDistribution with unique particles.

        Returns:
            DiscreteDistribution: A distribution where each particle appears only once,
            with its probability being the sum of all its occurrences in the original
            belief. Probabilities are L1-normalized to sum to 1.0. If the belief has
            no particles, an empty DiscreteDistribution is returned.
        """
        if self.particles.shape[0] == 0:
            return self._empty_discrete_distribution()

        unique_particles, inverse_indices = np.unique(self.particles, axis=0, return_inverse=True)
        inverse_indices = np.asarray(inverse_indices).ravel()
        combined_weights = np.zeros(unique_particles.shape[0], dtype=float)
        np.add.at(combined_weights, inverse_indices, self.normalized_weights)

        total = combined_weights.sum()
        if total > 0.0:
            probabilities = combined_weights / total
        else:
            probabilities = np.full(unique_particles.shape[0], 1.0 / unique_particles.shape[0])

        values = [unique_particles[i] for i in range(unique_particles.shape[0])]
        return DiscreteDistribution(values=values, probs=probabilities)

    @staticmethod
    def _empty_discrete_distribution() -> DiscreteDistribution:
        # DiscreteDistribution.__init__ rejects probs that do not sum to 1.0, so we
        # build the empty case directly. Sampling/probability methods are not
        # meaningful on an empty belief; consumers gate on len(values).
        empty = DiscreteDistribution.__new__(DiscreteDistribution)
        empty.values = []
        empty.probs = np.array([], dtype=float)
        empty._cumprobs = np.array([], dtype=float)  # pylint: disable=protected-access
        return empty

    @property
    def n_particles(self) -> int:
        """Return the number of particles."""
        return self.particles.shape[0]

    @property
    def dim(self) -> int:
        """Return the state dimensionality."""
        return self.particles.shape[1]

    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on belief configuration."""
        config_dict = {
            "particles": self.particles.tolist(),
            "log_weights": self.log_weights.tolist(),
            "resampling": self.resampling,
            "ess_factor": self.ess_factor,
            "updater": self.updater.config_id,
        }
        return config_to_id(config_dict)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_init_args(
        particles: np.ndarray,
        log_weights: np.ndarray,
        resampling: bool,
    ) -> None:
        if not isinstance(particles, np.ndarray):
            raise TypeError("particles must be a numpy.ndarray")
        if not isinstance(log_weights, np.ndarray):
            raise TypeError("log_weights must be a numpy.ndarray")
        if particles.ndim != 2:
            raise ValueError(f"particles must be 2-D, got {particles.ndim}-D")
        if log_weights.ndim != 1:
            raise ValueError(f"log_weights must be 1-D, got {log_weights.ndim}-D")
        if particles.shape[0] != log_weights.shape[0]:
            raise ValueError(
                f"particles has {particles.shape[0]} rows but "
                f"log_weights has {log_weights.shape[0]} entries"
            )
        if not isinstance(resampling, bool):
            raise TypeError("resampling must be a boolean")
        if np.any(np.isnan(log_weights)) or np.any(np.isposinf(log_weights)):
            raise ValueError("log_weights must not contain NaN or +inf values")

    @staticmethod
    def _normalize(log_weights: np.ndarray) -> np.ndarray:
        if np.all(np.isneginf(log_weights)):
            return np.ones(len(log_weights)) / len(log_weights)
        shifted = log_weights - np.max(log_weights)
        weights = np.exp(shifted)
        return weights / weights.sum()

    def _resample(self, particles: np.ndarray, log_weights: np.ndarray) -> tuple:
        normalized = self._normalize(log_weights)
        ess = 1.0 / np.sum(normalized**2)
        if ess < self.ess_threshold:
            indices = np.random.choice(len(particles), size=len(particles), p=normalized)
            resampled_particles = particles[indices]
            uniform_log_w = np.full(len(particles), -np.log(len(particles)))
            return resampled_particles, uniform_log_w
        return particles, log_weights
