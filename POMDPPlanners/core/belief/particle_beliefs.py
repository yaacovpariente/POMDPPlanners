"""Particle-based belief state representations for POMDP environments.

This module provides particle filter implementations for approximate belief
tracking, including both weighted and unweighted variants, with support for
reinvigoration and incremental state accumulation.

Classes:
    UnweightedParticleBelief: Uniform particle filter for discrete observation spaces
    WeightedParticleBelief: Weighted particle filter for continuous observation spaces
    WeightedParticleBeliefReinvigoration: Extended weighted filter with reinvigoration
    WeightedParticleBeliefStateUpdate: Incremental weighted particle belief for online learning
    UnweightedParticleBeliefStateUpdate: Incremental unweighted particle belief

Functions:
    get_unique_support: Extract unique particles and their combined probabilities
"""

import bisect
import random
from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.belief.base_belief import Belief
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.utils.config_to_id import config_to_id


class UnweightedParticleBelief(Belief):
    """Unweighted particle belief implementation.

    This class implements a particle filter with uniform particles.
    """

    def __init__(self, particles: list, reinvigoration_fraction=0.2):
        """Initialize unweighted particle belief.

        Args:
            particles: List of particles representing the belief state
            reinvigoration_fraction: Fraction of particles to reinvigorate
        """
        self.num_particles = len(particles)
        self.reinvigoration_fraction = reinvigoration_fraction
        self.particles = particles

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        state: Optional[Any] = None,
    ) -> "UnweightedParticleBelief":
        """Update belief with action-observation pair."""
        new_particles = []

        for _ in range(self.num_particles):
            # Sample a particle and simulate its transition
            s = random.choice(self.particles)
            next_s = pomdp.sample_next_state(state=s, action=action)
            obs = pomdp.sample_observation(next_state=next_s, action=action)
            if pomdp.is_equal_observation(obs, observation):
                new_particles.append(next_s)

        # Reinvigorate if degeneracy detected (too few new particles)
        if len(new_particles) < self.reinvigoration_fraction * self.num_particles:
            num_new = int(self.reinvigoration_fraction * self.num_particles)
            reinvigorated = [
                self.reinvigorate(action=action, observation=observation, pomdp=pomdp)
                for _ in range(max(0, num_new))
            ]
            new_particles += reinvigorated

        # Replenish to full count
        while len(new_particles) < self.num_particles:
            new_particles.append(random.choice(new_particles))

        self.particles = new_particles
        return self

    def reinvigorate(self, action: Any, observation: Any, pomdp: Environment):
        """Simulate a new particle that matches the action-observation pair."""
        # Try sampling from initial state and simulate until match
        while True:
            s = self._reinvigoration_pertubation(
                action=action, observation=observation, pomdp=pomdp
            )
            next_s = pomdp.sample_next_state(state=s, action=action)
            obs = pomdp.sample_observation(next_state=next_s, action=action)
            if pomdp.is_equal_observation(obs, observation):
                return next_s

    def sample(self):
        """Sample a particle from the belief."""
        return random.choice(self.particles)

    @abstractmethod
    def _reinvigoration_pertubation(self, action: Any, observation: Any, pomdp: Environment) -> Any:
        """Implement perturbation for reinvigoration in specific environment."""


class WeightedParticleBelief(Belief):
    """Weighted particle filter implementation for POMDP belief states.

    This class implements a particle filter with weighted particles, suitable
    for continuous observation spaces. It supports automatic resampling based
    on effective sample size to maintain particle diversity.

    Attributes:
        particles: List of state particles representing the belief
        log_weights: Log-weights of particles in log space for numerical stability
        normalized_weights: Normalized probability weights (computed automatically)
        resampling: Whether automatic resampling is enabled
        ess_factor: Effective sample size factor for resampling threshold
        ess_threshold: Computed threshold for triggering resampling
        eps: Small epsilon value for numerical stability in weight updates

    Example:
        >>> import numpy as np
        >>> # Create belief with 10 particles (smaller for testing)
        >>> particles = [[x, y] for x, y in zip(np.random.randn(10), np.random.randn(10))]
        >>> log_weights = np.log(np.ones(10) / 10)  # Uniform weights

        >>> belief = WeightedParticleBelief(
        ...     particles=particles,
        ...     log_weights=log_weights,
        ...     resampling=True,
        ...     ess_factor=0.5
        ... )
        >>> # Sample a state from belief
        >>> state = belief.sample()
        >>> len(state) == 2  # [x, y] coordinate
        True
    """

    def __init__(
        self,
        particles: List[Any],
        log_weights: np.ndarray,
        resampling: bool = False,
        ess_factor: float = 0.5,
    ):
        """Initialize weighted particle belief.

        Args:
            particles: List of state particles
            log_weights: Log-weights for particles (must sum to 1 in probability space)
            resampling: Enable automatic resampling when ESS drops. Defaults to False.
            ess_factor: Effective sample size threshold factor (0 < ess_factor <= 1). Defaults to 0.5.

        Raises:
            TypeError: If particles is not a list or log_weights is not a numpy array
            ValueError: If particles and weights have different lengths, or weights are invalid
        """
        if not isinstance(particles, list):
            raise TypeError("particles must be a list")
        if not isinstance(log_weights, np.ndarray):
            raise TypeError("log_weights must be a numpy.ndarray")
        if len(particles) != len(log_weights):
            raise ValueError("particles and log_weights must have the same length")
        if not isinstance(resampling, bool):
            raise TypeError("resampling must be a boolean")
        if not np.any(log_weights != 0):
            raise ValueError("At least one log_weight must be nonzero")
        if not np.all(np.isfinite(log_weights)):
            raise ValueError("log_weights must be finite numbers (not Inf, -Inf, or NaN)")

        self.particles: List[Any] = particles
        self.log_weights: np.ndarray = log_weights
        # First subtract max for numerical stability, then normalize to sum to 1
        self.normalized_weights: np.ndarray = np.exp(self.log_weights - np.max(self.log_weights))
        self.normalized_weights = self.normalized_weights / np.sum(self.normalized_weights)
        self.resampling = resampling
        self.ess_factor = ess_factor
        self.ess_threshold = len(particles) * ess_factor

        self.eps = 1e-10

        # Lazy cumulative-weight CDF. Built on first `sample()` call and
        # cached thereafter; skipped entirely for beliefs that are never
        # sampled (e.g. intermediate update() instances used only for
        # weight propagation).
        self._cdf: Optional[List[float]] = None

    def to_dict(self) -> dict:
        """Convert the belief to a dictionary for serialization.

        Returns:
            dict: A dictionary containing all necessary fields for deserialization.
        """
        return {
            "particles": self.particles,
            "log_weights": self.log_weights.tolist(),
            "resampling": self.resampling,
            "ess_factor": self.ess_factor,
        }

    def to_unique_support_distribution(self) -> "DiscreteDistribution":
        """Convert the belief to a DiscreteDistribution with unique particles.

        Returns:
            DiscreteDistribution: A distribution where each particle appears only once,
            with its probability being the sum of all its occurrences in the original belief.
        """
        unique_particles, probabilities = get_unique_support(
            self.particles, self.normalized_weights
        )
        return DiscreteDistribution(values=unique_particles, probs=probabilities)

    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on belief configuration."""

        def serialize_value(value):
            """Serialize values in a deterministic way."""
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            if isinstance(value, dict):
                return {str(k): serialize_value(v) for k, v in sorted(value.items())}
            if hasattr(value, "__dict__"):
                return serialize_value(value.__dict__)
            return str(value)

        # Create a list of particle-weight pairs to maintain order
        particle_weight_pairs = []
        for particle, weight in zip(self.particles, self.log_weights):
            # Convert particle to a serializable format if needed
            if isinstance(particle, np.ndarray):
                particle = particle.tolist()
            particle_weight_pairs.append((serialize_value(particle), float(weight)))

        # Sort particle-weight pairs to make config_id invariant to order
        particle_weight_pairs.sort(key=lambda x: (str(x[0]), x[1]))

        config_dict = {
            "particle_weight_pairs": particle_weight_pairs,
            "resampling": self.resampling,
            "ess_factor": self.ess_factor,
        }
        config_dict = dict(sorted(config_dict.items()))
        return config_to_id(config_dict)

    def _resample(
        self, particles: List[Any], log_weights: np.ndarray
    ) -> Tuple[List[Any], np.ndarray]:
        normalized_weights = np.exp(log_weights - np.max(log_weights))
        normalized_weights = normalized_weights / np.sum(normalized_weights)

        effective_sample_size = 1 / np.sum(np.square(normalized_weights))
        if effective_sample_size < self.ess_threshold:
            sampled_indexes = random.choices(
                range(len(particles)),
                weights=normalized_weights,
                k=len(particles),
            )
            resampled_particles = [particles[i] for i in sampled_indexes]
            new_log_weights = np.log(np.ones(len(resampled_particles)) / len(resampled_particles))
            return resampled_particles, new_log_weights
        return particles, log_weights

    def _update_weights(
        self, action: Any, observation: Any, pomdp: Environment
    ) -> Tuple[List[Any], np.ndarray]:
        # Fast path: if the env's transition model exposes a native batch
        # entry point, process all particles in a single C++ call. When the
        # observation model also exposes batch_log_likelihood, the
        # per-particle Python loop is eliminated entirely and the belief
        # update becomes O(1) Python round trips instead of O(N).
        transition_model = pomdp.state_transition_model(state=self.particles[0], action=action)
        if hasattr(transition_model, "batch_sample"):
            return self._update_weights_batch(
                action=action,
                observation=observation,
                pomdp=pomdp,
                transition_model=transition_model,
            )

        # Fallback: per-particle Python loop. Byte-identical to pre-Layer-2
        # behaviour for envs without native batch support.
        next_particles = [
            pomdp.sample_next_state(state=particle, action=action) for particle in self.particles
        ]
        probs = np.array(
            [
                pomdp.observation_model(next_state=next_particle, action=action).probability(
                    [observation]
                )[0]
                for next_particle in next_particles
            ]
        )

        next_log_weights = self.log_weights + np.log(self.eps + probs)

        return next_particles, next_log_weights

    def _update_weights_batch(
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        transition_model: Any,
    ) -> Tuple[List[Any], np.ndarray]:
        particles_arr = np.asarray(self.particles, dtype=float)
        next_arr = transition_model.batch_sample(particles_arr)
        next_particles: List[Any] = [next_arr[i] for i in range(len(next_arr))]

        obs_model = pomdp.observation_model(next_state=next_arr[0], action=action)
        if hasattr(obs_model, "batch_log_likelihood"):
            observation_arr = np.asarray(observation, dtype=float).ravel()
            # pyright cannot see through the hasattr narrowing onto a
            # concrete native subclass (e.g. MountainCarObservationCpp);
            # the runtime hasattr guarantees the call is safe.
            log_ll = obs_model.batch_log_likelihood(  # type: ignore[attr-defined]
                next_arr, observation_arr
            )
            # No eps clamp on the native batch path -- log_pdf is always
            # finite for Gaussian observation models, matching the existing
            # VectorizedWeightedParticleBelief contract.
            next_log_weights = self.log_weights + log_ll
            return next_particles, next_log_weights

        probs = np.array(
            [
                pomdp.observation_model(next_state=next_particle, action=action).probability(
                    [observation]
                )[0]
                for next_particle in next_particles
            ]
        )
        next_log_weights = self.log_weights + np.log(self.eps + probs)
        return next_particles, next_log_weights

    def update(
        self,
        action,
        observation,
        pomdp: Environment,
        state: Optional[Any] = None,
    ) -> "WeightedParticleBelief":
        """Update belief with action-observation pair."""
        next_particles, next_log_weights = self._update_weights(
            action=action, observation=observation, pomdp=pomdp
        )

        if self.resampling:
            next_particles, next_log_weights = self._resample(next_particles, next_log_weights)

        return WeightedParticleBelief(
            particles=next_particles,
            log_weights=next_log_weights,
            resampling=self.resampling,
        )

    def sample(self):
        """Sample a particle from the belief."""
        cdf = self._cdf
        if cdf is None:
            cdf = np.cumsum(self.normalized_weights).tolist()
            self._cdf = cdf
        target = random.random() * cdf[-1]
        idx = bisect.bisect_left(cdf, target)
        if idx >= len(self.particles):
            idx = len(self.particles) - 1
        return self.particles[idx]


class WeightedParticleBeliefReinvigoration(WeightedParticleBelief, ABC):
    """Weighted particle belief with reinvigoration capability."""

    def __init__(
        self,
        particles: List[Any],
        log_weights: np.ndarray,
        resampling: bool = True,
        ess_factor: float = 0.5,
        reinvigoration_fraction: float = 0.2,
    ):
        """Initialize weighted particle belief with reinvigoration."""
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            resampling=resampling,
            ess_factor=ess_factor,
        )

        self.reinvigoration_fraction = reinvigoration_fraction

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        state: Optional[Any] = None,
    ) -> "WeightedParticleBelief":
        """Update belief with reinvigoration."""
        belief_after_weights = super().update(
            action=action, observation=observation, pomdp=pomdp, state=state
        )

        belief_reinvigorated = self.reinvigorate(
            action=action, observation=observation, pomdp=pomdp, belief=belief_after_weights
        )

        # Ensure return type is WeightedParticleBelief
        assert isinstance(belief_reinvigorated, WeightedParticleBelief)
        return belief_reinvigorated

    @abstractmethod
    def reinvigorate(
        self, action: Any, observation: Any, pomdp: Environment, belief: Belief
    ) -> Belief:
        """Implement reinvigoration for specific POMDP environment."""


class WeightedParticleBeliefStateUpdate(Belief):
    """Incremental weighted particle belief for online state estimation.

    This class implements a lightweight belief representation that incrementally
    accumulates state particles with associated observation likelihood weights.
    It is designed for online learning and planning algorithms that build beliefs
    by sequentially adding individual state samples rather than maintaining a
    fixed-size particle set.

    Unlike traditional particle filters that maintain a fixed number of particles
    through resampling, WeightedParticleBeliefStateUpdate grows dynamically by
    accumulating particles with observation-based weights. This makes it particularly
    suitable for Monte Carlo Tree Search (MCTS) algorithms where beliefs are built
    incrementally during tree expansion.

    Key Features:
    - **Incremental Accumulation**: Add particles one-by-one without resampling
    - **Observation Weighting**: Each particle weighted by observation likelihood
    - **Efficient Updates**: Both in-place and immutable update operations
    - **Weighted Sampling**: Sample states proportionally to observation evidence
    - **Memory Efficient**: No fixed particle budget, grows as needed
    - **Deterministic Config ID**: Order-invariant identification for caching

    Mathematical Foundation:
    The belief represents a discrete probability distribution where each particle
    s_i has weight w_i = P(o|s_i,a), the observation likelihood. The probability
    of state s is proportional to the sum of weights for all particles with that state:

    P(s|o,a) ∝ Σ_{i: s_i=s} w_i

    Attributes:
        particles: List of state particles representing possible world states
        weights: List of observation likelihood weights for each particle
        weights_sum: Running sum of all weights for efficient normalization

    Example:
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> env = TigerPOMDP(discount_factor=0.95)
        >>> belief = WeightedParticleBeliefStateUpdate(particles=[], weights=[])
        >>> belief.inplace_update("listen", "hear_left", env, "tiger_left")
        >>> belief.inplace_update("listen", "hear_left", env, "tiger_right")
        >>> sampled_state = belief.sample()
        >>> sampled_state in ["tiger_left", "tiger_right"]
        True
    """

    def __init__(self, particles: Optional[list] = None, weights: Optional[list] = None):
        """Initialize weighted particle belief.

        Creates a belief state with given particles and weights. Empty lists
        create an empty belief that can be populated incrementally.

        Args:
            particles: List of state particles. Defaults to empty list.
            weights: List of observation likelihood weights corresponding to particles.
                Must have same length as particles. Defaults to empty list.

        Raises:
            ValueError: If particles and weights have different lengths.

        Note:
            When weights is empty, weights_sum is automatically set to 0.
            This handles the case where an empty belief is initialized.
        """
        if particles is None:
            particles = []
        if weights is None:
            weights = []

        if len(particles) != len(weights):
            raise ValueError("particles and weights must have the same length")

        self.particles: list = list(particles)
        self.weights: list = list(weights)
        self.weights_sum = sum(weights) if weights else 0
        # Cumulative-weight CDF, maintained in sync with `weights`. Lets
        # `sample()` use O(log K) bisect instead of an O(K) renormalization
        # pass each call. Updated incrementally in `inplace_update`.
        self._cdf: List[float] = []
        running = 0.0
        for w in self.weights:
            running += float(w)
            self._cdf.append(running)

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        state: Optional[Any] = None,
    ) -> "WeightedParticleBeliefStateUpdate":
        """Create new belief by adding a state particle with observation weight.

        This method creates a new belief instance without modifying the current one.
        The new particle's weight is computed as the observation likelihood given
        the state and action using the environment's observation model.

        Args:
            action: Action that was executed to reach the state
            observation: Observation received after executing the action
            pomdp: Environment providing the observation model for weight computation
            state: State particle to add to the belief. If None, no particle is added.

        Returns:
            New WeightedParticleBeliefStateUpdate instance with the additional particle
            and updated weights.

        """
        if state is None:
            raise ValueError("state cannot be None")

        new_particles = self.particles + [state]
        observation_probability = pomdp.observation_model(
            next_state=state, action=action
        ).probability([observation])[0]
        new_weights = self.weights + [
            float(
                observation_probability.item()
                if hasattr(observation_probability, "item")
                else observation_probability
            )
        ]
        new_belief = WeightedParticleBeliefStateUpdate(particles=new_particles, weights=new_weights)

        return new_belief

    def inplace_update(
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        state: Optional[Any] = None,
    ) -> None:
        """Add a state particle with observation weight to current belief.

        This method modifies the current belief in-place by appending a new particle
        and its corresponding observation likelihood weight. The weight is computed
        using the environment's observation model and efficiently updates the
        running weight sum.

        Args:
            action: Action that was executed to reach the state
            observation: Observation received after executing the action
            pomdp: Environment providing the observation model for weight computation
            state: State particle to add to the belief. If None, no particle is added.

        """
        if state is None:
            raise ValueError("state cannot be None")
        if action is None:
            raise ValueError("action cannot be None")
        if observation is None:
            raise ValueError("observation cannot be None")
        if not isinstance(pomdp, Environment):
            raise TypeError("pomdp must be an instance of Environment")

        self.particles.append(state)
        observation_probability = pomdp.observation_model(
            next_state=state, action=action
        ).probability([observation])[0]
        weight = float(observation_probability)
        self.weights.append(weight)
        self.weights_sum = float(self.weights_sum) + weight
        # Maintain CDF for O(log K) sampling.
        prev_total = self._cdf[-1] if self._cdf else 0.0
        self._cdf.append(prev_total + weight)

    def sample(self) -> Any:
        """Sample a state from the current belief distribution.

        Returns:
            A state sampled according to the belief's probability distribution

        Raises:
            ValueError: If belief is empty or has zero weights.
        """
        if not self.particles or self.weights_sum == 0:
            raise ValueError("Cannot sample from empty or unnormalized belief")

        # O(log K) sample via bisect on the maintained CDF (was an O(K)
        # renormalization + random.choices call).
        target = random.random() * self._cdf[-1]
        idx = bisect.bisect_left(self._cdf, target)
        if idx >= len(self.particles):
            idx = len(self.particles) - 1
        particle = self.particles[idx]

        # Defensive programming: ensure particle is a numpy array if it's a list
        # (fix for 'list' object has no attribute 'shape' error)
        if isinstance(particle, list):
            particle = np.array(particle)

        return particle

    def to_unique_support_distribution(self) -> "DiscreteDistribution":
        """Convert the belief to a DiscreteDistribution with unique particles.

        Returns:
            DiscreteDistribution: A distribution where each particle appears only once,
            with its probability being the sum of all its occurrences in the original belief.
        """
        # Convert weights list to numpy array for get_unique_support
        weights_arr = np.array(self.weights, dtype=float)
        unique_particles, probabilities = get_unique_support(self.particles, weights_arr)
        return DiscreteDistribution(values=unique_particles, probs=probabilities)

    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on belief configuration.

        This implementation ensures that config_id is invariant to the order
        of particles and weights, similar to WeightedParticleBelief.
        """

        def serialize_value(value):
            """Serialize values in a deterministic way."""
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            if isinstance(value, dict):
                return {str(k): serialize_value(v) for k, v in sorted(value.items())}
            if hasattr(value, "__dict__"):
                return serialize_value(value.__dict__)
            return str(value)

        # Create a list of particle-weight pairs to maintain order
        particle_weight_pairs = []
        for particle, weight in zip(self.particles, self.weights):
            # Convert particle to a serializable format if needed
            if isinstance(particle, np.ndarray):
                particle = particle.tolist()
            particle_weight_pairs.append((serialize_value(particle), float(weight)))

        # Sort particle-weight pairs to make config_id invariant to order
        particle_weight_pairs.sort(key=lambda x: (str(x[0]), x[1]))

        config_dict = {
            "particle_weight_pairs": particle_weight_pairs,
            "weights_sum": self.weights_sum,
        }
        config_dict = dict(sorted(config_dict.items()))
        return config_to_id(config_dict)


class UnweightedParticleBeliefStateUpdate(Belief):
    r"""Uniform particle belief for incremental state accumulation.

    This class implements a lightweight belief representation that maintains
    a collection of state particles with uniform probability distribution.
    Unlike weighted particle filters, all particles contribute equally to
    the belief state, making it suitable for discrete observation spaces
    where observation likelihoods are binary (match/no-match) rather than
    continuous probability distributions.

    UnweightedParticleBeliefStateUpdate is designed for online learning and
    planning algorithms that incrementally accumulate particles during tree
    expansion or sequential state estimation. It provides both in-place and
    immutable update operations for different algorithmic requirements.

    Key Features:
    - **Uniform Weighting**: All particles have equal probability weight
    - **Incremental Accumulation**: Add particles one-by-one without resampling
    - **Memory Efficient**: No weight storage, minimal memory overhead
    - **Fast Sampling**: Simple uniform random sampling from particle set
    - **Efficient Updates**: Both in-place and immutable update operations
    - **Deterministic Config ID**: Order-invariant identification for caching

    Mathematical Foundation:
    The belief represents a discrete uniform distribution over accumulated particles.
    Each particle has equal probability 1/N where N is the total number of particles.
    For particles with the same state value, the probability is proportional to
    their count:

    P(s) = count(s) / N = |{i: s_i = s}| / |particles|

    This makes it ideal for discrete observation models where observations either
    match a state (probability 1) or don't match (probability 0).

    Attributes:
        particles: List of state particles, each with uniform probability
        weights_sum: Total number of particles (equivalent to uniform weight sum)

    Example:
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> env = TigerPOMDP(discount_factor=0.95)
        >>> belief = UnweightedParticleBeliefStateUpdate(particles=[])
        >>> belief.inplace_update("listen", "hear_left", env, "tiger_left")
        >>> belief.inplace_update("listen", "hear_right", env, "tiger_right")
        >>> sampled_state = belief.sample()
        >>> sampled_state in ["tiger_left", "tiger_right"]
        True
    """

    def __init__(self, particles: Optional[list] = None):
        """Initialize unweighted particle belief.

        Creates a belief state with uniform probability distribution over
        the provided particles. Each particle has equal weight 1/N where
        N is the number of particles.

        Args:
            particles: List of state particles with uniform weights.
                Defaults to empty list for incremental construction.
        """
        if particles is None:
            particles = []
        self.particles = particles
        self.weights_sum = len(particles)

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        state: Optional[Any] = None,
    ) -> "UnweightedParticleBeliefStateUpdate":
        """Create new belief by adding a state particle with uniform weight.

        This method creates a new belief instance without modifying the current one.
        Unlike weighted beliefs, all particles (including the new one) have equal
        probability in the resulting belief distribution.

        Args:
            action: Action that was executed to reach the state (not used for weighting)
            observation: Observation received after executing the action (not used for weighting)
            pomdp: Environment instance (not used for uniform weighting)
            state: State particle to add to the belief. If None, no particle is added.

        Returns:
            New UnweightedParticleBeliefStateUpdate instance with the additional particle.

        """
        new_particles = self.particles + [state]
        return UnweightedParticleBeliefStateUpdate(particles=new_particles)

    def sample(self) -> Any:
        """Sample a state uniformly from the current belief distribution.

        Returns:
            A state sampled uniformly from the particle set

        Raises:
            IndexError: If belief is empty (no particles to sample from)
        """
        return random.choice(self.particles)

    def inplace_update(
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        state: Optional[Any] = None,
    ) -> None:
        """Add a state particle with uniform weight to current belief.

        This method modifies the current belief in-place by appending a new particle.
        Unlike weighted beliefs, no observation likelihood computation is performed;
        the new particle simply joins the uniform distribution.

        Args:
            action: Action that was executed to reach the state (not used for weighting)
            observation: Observation received after executing the action (not used for weighting)
            pomdp: Environment instance (not used for uniform weighting)
            state: State particle to add to the belief. If None, no particle is added.
        """
        self.particles.append(state)
        self.weights_sum += 1

    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on belief configuration.

        This implementation ensures that config_id is invariant to the order
        of particles by sorting them.
        """

        def serialize_value(value):
            """Serialize values in a deterministic way."""
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            if isinstance(value, dict):
                return {str(k): serialize_value(v) for k, v in sorted(value.items())}
            if hasattr(value, "__dict__"):
                return serialize_value(value.__dict__)
            return str(value)

        # Convert particles to serializable format and sort them
        serialized_particles = []
        for particle in self.particles:
            # Convert particle to a serializable format if needed
            if isinstance(particle, np.ndarray):
                particle = particle.tolist()
            serialized_particles.append(serialize_value(particle))

        # Sort particles to make config_id invariant to order
        serialized_particles.sort(key=str)

        config_dict = {
            "particles": serialized_particles,
            "weights_sum": self.weights_sum,
        }
        config_dict = dict(sorted(config_dict.items()))
        return config_to_id(config_dict)


def get_unique_support(
    particles: List[Any], probabilities: np.ndarray
) -> Tuple[List[Any], np.ndarray]:
    """Extract unique particles and their combined probabilities.

    This function takes a list of particles and their associated probabilities,
    combines probabilities for duplicate particles, and returns unique particles
    with normalized probabilities.

    Args:
        particles: List of particles of any type
        probabilities: Array of probabilities/weights corresponding to each particle

    Returns:
        Tuple containing:
            - List of unique particles (preserving original types)
            - Normalized numpy array of probabilities summing to 1

    Example:
        >>> particles = [1, 2, 1, 3, 2]
        >>> probs = np.array([0.2, 0.3, 0.1, 0.2, 0.2])
        >>> unique_particles, unique_probs = get_unique_support(particles, probs)
        >>> unique_particles  # [1, 2, 3]
        [1, 2, 3]
        >>> float(np.sum(unique_probs))  # Should be 1.0
        1.0
    """

    def _make_hashable(value):
        # Try to hash directly first - if it works, return as-is
        try:
            hash(value)
            return value
        except TypeError:
            # Not hashable, need to convert
            pass

        # Convert based on type
        if isinstance(value, np.ndarray):
            # Recursively convert nested arrays to tuples
            return tuple(_make_hashable(item) for item in value.tolist())
        if isinstance(value, (list, tuple)):
            # Convert lists to tuples, recursively handle nested structures
            return tuple(_make_hashable(item) for item in value)
        # Shouldn't reach here if hash check worked, but fallback
        return value

    # Create dictionaries to store unique particles and their combined probabilities
    # We need to store both the hashable key and the original particle
    unique_particles_dict: Dict[Hashable, Tuple[Any, float]] = {}

    # Iterate through particles and their probabilities
    for particle, prob in zip(particles, probabilities):
        # Convert particle to hashable key
        particle_key: Hashable = _make_hashable(particle)

        # Add or update the probability for this particle
        if particle_key in unique_particles_dict:
            original_particle, combined_prob = unique_particles_dict[particle_key]
            unique_particles_dict[particle_key] = (original_particle, combined_prob + prob)
        else:
            unique_particles_dict[particle_key] = (particle, prob)

    # Convert back to original particle types and create arrays
    unique_particles_list: List[Any] = []
    combined_probs: List[float] = []
    for particle_key, (original_particle, prob) in unique_particles_dict.items():
        unique_particles_list.append(original_particle)
        combined_probs.append(float(prob))

    # Convert to numpy array and normalize to ensure sum is exactly 1
    probabilities_arr = np.array(combined_probs, dtype=float)
    if len(probabilities_arr) > 0:
        probabilities_arr = probabilities_arr / np.sum(probabilities_arr)  # Normalize to sum to 1

    return unique_particles_list, probabilities_arr
