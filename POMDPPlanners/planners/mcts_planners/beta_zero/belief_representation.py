"""Belief-to-feature-vector representations for BetaZero.

This module provides abstractions for converting POMDP belief states into
fixed-size feature vectors suitable for neural network input. The default
implementation extracts particle statistics (mean and standard deviation).

Reference:
    Moss, R. J., Corso, A., Caers, J., & Kochenderfer, M. J. (2024). BetaZero:
    Belief-State Planning for Long-Horizon POMDPs using Learned Approximations.
    Reinforcement Learning Conference (RLC).

Classes:
    BeliefRepresentation: Abstract base for belief feature extraction
    ParticleMeanStdRepresentation: Default φ(b) = [mean(particles), std(particles)]
"""

from abc import ABC, abstractmethod
import numpy as np

from POMDPPlanners.core.belief import (
    Belief,
    GaussianBelief,
    WeightedParticleBelief,
    WeightedParticleBeliefStateUpdate,
)


class BeliefRepresentation(ABC):
    """Abstract base class for mapping beliefs to fixed-size feature vectors.

    Subclasses define how a POMDP belief state is compressed into a
    numerical vector φ(b) ∈ ℝ^d that can be fed to a neural network.

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    @abstractmethod
    def __call__(self, belief: Belief) -> np.ndarray:
        """Convert a belief into a feature vector.

        Args:
            belief: The belief state to represent.

        Returns:
            Feature vector of shape ``(feature_dim,)``.
        """

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensionality of the output feature vector."""


class ParticleMeanStdRepresentation(BeliefRepresentation):
    """Default belief representation: φ(b) = [mean(particles), std(particles)].

    For a state space of dimension *d*, the output is a vector of length 2·d
    formed by concatenating the (weighted) mean and standard deviation of the
    belief's particle set.

    Supported belief types:
    - ``WeightedParticleBelief`` / ``WeightedParticleBeliefStateUpdate``:
      uses normalised weights for statistics.
    - ``GaussianBelief``: extracts mean and diagonal of covariance.
    - Any other ``Belief`` subclass: falls back to sampling 100 particles.

    Args:
        state_dim: Dimensionality of the state space.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.core.belief import WeightedParticleBelief
        >>> from POMDPPlanners.planners.mcts_planners.beta_zero.belief_representation import (
        ...     ParticleMeanStdRepresentation,
        ... )
        >>>
        >>> particles = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        >>> log_weights = np.log([0.2, 0.5, 0.3])
        >>> belief = WeightedParticleBelief(particles, log_weights)
        >>> rep = ParticleMeanStdRepresentation(state_dim=2)
        >>> features = rep(belief)
        >>> features.shape
        (4,)
        >>> rep.feature_dim
        4
    """

    def __init__(self, state_dim: int):
        self._state_dim = state_dim

    @property
    def feature_dim(self) -> int:
        return 2 * self._state_dim

    def __call__(self, belief: Belief) -> np.ndarray:
        if isinstance(belief, WeightedParticleBelief):
            return self._from_weighted_particle_belief(belief)
        if isinstance(belief, WeightedParticleBeliefStateUpdate):
            return self._from_weighted_particle_belief_state_update(belief)
        if isinstance(belief, GaussianBelief):
            return self._from_gaussian(belief)
        return self._from_samples(belief)

    def _from_weighted_particle_belief(
        self,
        belief: WeightedParticleBelief,
    ) -> np.ndarray:
        try:
            particles = np.asarray(belief.particles, dtype=np.float64)
        except (ValueError, TypeError):
            return self._from_samples(belief)
        if particles.ndim == 1:
            particles = particles.reshape(-1, 1)
        weights = np.asarray(belief.normalized_weights, dtype=np.float64)
        mean = np.average(particles, weights=weights, axis=0)
        var = np.average((particles - mean) ** 2, weights=weights, axis=0)
        std = np.sqrt(np.maximum(var, 0.0))
        return np.concatenate([mean, std]).astype(np.float32)

    def _from_weighted_particle_belief_state_update(
        self,
        belief: WeightedParticleBeliefStateUpdate,
    ) -> np.ndarray:
        try:
            particles = np.asarray(belief.particles, dtype=np.float64)
        except (ValueError, TypeError):
            return self._from_samples(belief)
        if particles.ndim == 1:
            particles = particles.reshape(-1, 1)
        raw_weights = np.asarray(belief.weights, dtype=np.float64)
        total = raw_weights.sum()
        weights = raw_weights / total if total > 0 else np.ones(len(raw_weights)) / len(raw_weights)
        mean = np.average(particles, weights=weights, axis=0)
        var = np.average((particles - mean) ** 2, weights=weights, axis=0)
        std = np.sqrt(np.maximum(var, 0.0))
        return np.concatenate([mean, std]).astype(np.float32)

    def _from_gaussian(self, belief: GaussianBelief) -> np.ndarray:
        mean = np.asarray(belief.mean, dtype=np.float32)
        std = np.sqrt(np.maximum(np.diag(belief.covariance), 0.0)).astype(np.float32)
        return np.concatenate([mean, std])

    def _from_samples(self, belief: Belief, n_samples: int = 100) -> np.ndarray:
        raw_samples = [belief.sample() for _ in range(n_samples)]
        try:
            samples = np.array(raw_samples, dtype=np.float64)
        except (ValueError, TypeError):
            # Non-numeric states: return zero features
            return np.zeros(self.feature_dim, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)
        return np.concatenate([mean, std]).astype(np.float32)
