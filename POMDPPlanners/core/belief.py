"""Module for POMDP belief state representations.

This module provides belief state abstractions for POMDP environments, including
particle filter implementations for approximate belief tracking and belief update
mechanisms.

Classes:
    Belief: Abstract base class for belief representations
    WeightedParticleBelief: Weighted particle filter for continuous observation spaces
    UnweightedParticleBelief: Uniform particle filter for discrete observation spaces
    WeightedParticleBeliefReinvigoration: Extended weighted filter with reinvigoration
    WeightedParticleBeliefStateUpdate: Incremental weighted particle belief for online learning

Functions:
    sample_next_belief: Simulate one step of belief evolution
    get_initial_belief: Create initial belief from environment's initial distribution
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Iterable, Dict, Hashable, List, Union

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.utils.config_to_id import config_to_id


class Belief(ABC):
    """Abstract base class for POMDP belief state representations.

    This class defines the interface for belief states in POMDP environments.
    Belief states represent probability distributions over the state space,
    capturing the agent's uncertainty about the current state.

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses must implement the update() and sample() methods.
    """

    @classmethod
    def from_config(cls, config):
        """Create a belief instance from configuration.

        Factory method that dynamically creates belief instances based on
        configuration objects specifying the class name and parameters.

        Args:
            config: Configuration object with class_name and params attributes

        Returns:
            New belief instance of the specified type

        Raises:
            ValueError: If the specified belief class is not found
        """

        # Get all subclasses of Belief recursively
        def get_all_subclasses(c):
            subclasses = c.__subclasses__()
            for subclass in subclasses:
                subclasses.extend(get_all_subclasses(subclass))
            return subclasses

        all_subclasses = get_all_subclasses(cls)
        for subclass in all_subclasses:
            if subclass.__name__ == config.class_name:
                return subclass(**config.params)
        raise ValueError(f"Belief class '{config.class_name}' not found")

    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on belief configuration."""

        def serialize_value(value):
            """Serialize values in a deterministic way."""
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (str, int, float, bool)):
                return value
            elif isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            elif isinstance(value, dict):
                return {str(k): serialize_value(v) for k, v in sorted(value.items())}
            elif hasattr(value, "__dict__"):
                return serialize_value(value.__dict__)
            return str(value)

        config_dict = {}
        for key, value in self.__dict__.items():
            if key.startswith("_") or callable(value):
                continue
            config_dict[key] = serialize_value(value)
        config_dict = dict(sorted(config_dict.items()))
        return config_to_id(config_dict)

    def __hash__(self) -> int:
        """Make the belief hashable by using its config_id."""
        return hash(self.config_id)

    def __eq__(self, other: object) -> bool:
        """Define equality based on config_id."""
        if not isinstance(other, Belief):
            return NotImplemented
        return self.config_id == other.config_id

    def inplace_update(
        self, action: Any, observation: Any, pomdp: Environment, state: Optional[Any] = None
    ) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        state: Optional[Any] = None,
    ) -> "Belief":
        """Update belief given an action-observation pair.

        Performs Bayesian belief update using the environment's transition
        and observation models.

        Args:
            action: Action that was executed
            observation: Observation that was received
            pomdp: Environment providing transition and observation models

        Returns:
            Updated belief state reflecting the new information

        Note:
            Subclasses must implement this method according to their
            specific belief representation and update strategy.
        """
        pass

    @abstractmethod
    def sample(self) -> Any:
        """Sample a state from the current belief distribution.

        Returns:
            A state sampled according to the belief's probability distribution

        Note:
            Subclasses must implement this method to enable state sampling
            for planning and simulation purposes.
        """
        pass


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
            next_s = pomdp.state_transition_model(state=s, action=action).sample()[0]
            obs = pomdp.observation_model(next_state=next_s, action=action).sample()[0]
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
            next_s = pomdp.state_transition_model(state=s, action=action).sample()[0]
            obs = pomdp.observation_model(next_state=next_s, action=action).sample()[0]
            if pomdp.is_equal_observation(obs, observation):
                return next_s

    def sample(self):
        """Sample a particle from the belief."""
        return random.choice(self.particles)

    @abstractmethod
    def _reinvigoration_pertubation(self, action: Any, observation: Any, pomdp: Environment) -> Any:
        """Implement perturbation for reinvigoration in specific environment."""
        pass


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

        >>> # Verify belief creation
        >>> len(belief.particles) == 10
        True
        >>> len(belief.log_weights) == 10
        True
        >>> belief.resampling
        True

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
        # Create a dictionary to store unique particles and their combined weights
        unique_particles: Dict[Hashable, float] = {}

        # Iterate through particles and their weights
        for particle, weight in zip(self.particles, self.normalized_weights):
            # Convert particle to tuple if it's a numpy array for hashability
            if isinstance(particle, np.ndarray):
                particle_key: Hashable = tuple(particle.tolist())
            else:
                particle_key = particle  # assume hashable

            # Add or update the weight for this particle
            if particle_key in unique_particles:
                unique_particles[particle_key] += weight
            else:
                unique_particles[particle_key] = weight

        # Convert back to original particle types and create arrays
        particles: List[Any] = []
        weights: List[float] = []
        for particle_key, weight in unique_particles.items():
            if isinstance(particle_key, tuple):
                particles.append(np.array(particle_key))
            else:
                particles.append(particle_key)
            weights.append(float(weight))

        # Convert to numpy array and normalize to ensure sum is exactly 1
        weights_arr = np.array(weights, dtype=float)
        weights_arr = weights_arr / np.sum(weights_arr)  # Normalize to sum to 1

        return DiscreteDistribution(values=particles, probs=weights_arr)

    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on belief configuration."""

        def serialize_value(value):
            """Serialize values in a deterministic way."""
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (str, int, float, bool)):
                return value
            elif isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            elif isinstance(value, dict):
                return {str(k): serialize_value(v) for k, v in sorted(value.items())}
            elif hasattr(value, "__dict__"):
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
        """Resample particles based on their weights if effective sample size is below threshold.

        Args:
            particles: List of particles to potentially resample
            log_weights: Log weights of the particles

        Returns:
            Tuple containing:
            - Resampled particles (or original if no resampling needed)
            - New log weights (or original if no resampling needed)
        """
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
        next_particles = [
            pomdp.state_transition_model(state=particle, action=action).sample()[0]
            for particle in self.particles
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
        idx = np.random.choice(len(self.particles), p=self.normalized_weights)
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
        pass


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

    Examples:
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> # Create environment and empty belief
        >>> env = TigerPOMDP(discount_factor=0.95)
        >>> belief = WeightedParticleBeliefStateUpdate(particles=[], weights=[])

        >>> # Add states incrementally with observations
        >>> belief.inplace_update("listen", "hear_left", env, "tiger_left")
        >>> belief.inplace_update("listen", "hear_left", env, "tiger_right")
        >>> belief.inplace_update("listen", "hear_left", env, "tiger_left")

        >>> # Verify belief construction
        >>> len(belief.particles) == 3
        True
        >>> bool(belief.weights_sum > 0)
        True
        >>> "tiger_left" in belief.particles
        True

        >>> # Sample weighted by observation probabilities
        >>> sampled_state = belief.sample()
        >>> sampled_state in ["tiger_left", "tiger_right"]
        True

        >>> import numpy as np
        >>> from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
        >>> # Create continuous state environment
        >>> noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
        >>> env = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov)

        >>> # Start with single particle
        >>> initial_state = np.array([0.0, 0.0, 0.1, 0.0])
        >>> belief = WeightedParticleBeliefStateUpdate([initial_state], [1.0])
        >>> len(belief.particles) == 1
        True

        >>> # Create child belief with new observation
        >>> action = 1  # Apply force right
        >>> obs = np.array([0.1, 0.0, 0.08, 0.0])
        >>> next_state = env.state_transition_model(initial_state, action).sample()[0]
        >>> child_belief = belief.update(action, obs, env, next_state)

        >>> # Verify immutable update
        >>> len(belief.particles) == 1  # Original unchanged
        True
        >>> len(child_belief.particles) == 2  # New belief has additional particle
        True
        >>> bool(child_belief.weights[-1] > 0)  # New particle has positive weight
        True

        >>> from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
        >>> env = SanityPOMDP(discount_factor=0.95)

        >>> # Strategy 1: In-place updates (memory efficient)
        >>> belief_inplace = WeightedParticleBeliefStateUpdate([], [])
        >>> states_observations = [(0, 0), (1, 0), (0, 0)]

        >>> for state, obs in states_observations:
        ...     belief_inplace.inplace_update("action", obs, env, state)

        >>> len(belief_inplace.particles) == 3
        True

        >>> # Strategy 2: Immutable updates (functional style)
        >>> belief_immutable = WeightedParticleBeliefStateUpdate([], [])
        >>> for state, obs in states_observations:
        ...     belief_immutable = belief_immutable.update("action", obs, env, state)

        >>> len(belief_immutable.particles) == 3
        True

        >>> # Both should have same final state
        >>> len(belief_inplace.particles) == len(belief_immutable.particles)
        True

        Monte Carlo Tree Search integration::

            from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            from POMDPPlanners.core.belief import WeightedParticleBeliefStateUpdate
            from POMDPPlanners.core.tree import BeliefNode, ActionNode
            import random

            env = TigerPOMDP(discount_factor=0.95)

            # Root belief node with initial particles
            root_belief = WeightedParticleBeliefStateUpdate(
                particles=["tiger_left", "tiger_right"],
                weights=[0.5, 0.5]
            )
            root_node = BeliefNode(belief=root_belief)

            # Simulate MCTS expansion
            action = "listen"
            possible_observations = ["hear_left", "hear_right"]

            # Create action node
            action_node = ActionNode(action=action, parent=root_node)

            # For each possible observation, create belief child
            for observation in possible_observations:
                # Sample particles and create child belief
                child_belief = WeightedParticleBeliefStateUpdate([], [])

                # Add particles based on transition model
                for _ in range(5):  # Multiple particles per observation
                    parent_state = root_belief.sample()
                    next_state = env.state_transition_model(parent_state, action).sample()[0]
                    child_belief.inplace_update(action, observation, env, next_state)

                # Create belief node for tree
                belief_node = BeliefNode(belief=child_belief, observation=observation, parent=action_node)
                print(f"Child belief for {observation}: {len(child_belief.particles)} particles")

        Weighted sampling and state estimation::

            from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            from POMDPPlanners.core.belief import WeightedParticleBeliefStateUpdate
            import collections

            env = TigerPOMDP(discount_factor=0.95)
            belief = WeightedParticleBeliefStateUpdate([], [])

            # Add strongly biased evidence for tiger_left
            evidence_sets = [
                ("tiger_left", "hear_left", 5),    # Strong evidence for left
                ("tiger_right", "hear_left", 2),   # Weak evidence for left from right
                ("tiger_left", "hear_right", 1),   # Weak evidence for right from left
                ("tiger_right", "hear_right", 3),  # Medium evidence for right
            ]

            for state, obs, count in evidence_sets:
                for _ in range(count):
                    belief.inplace_update("listen", obs, env, state)

            # Analyze sampling distribution
            samples = [belief.sample() for _ in range(1000)]
            sample_counts = collections.Counter(samples)

            print("Sampling distribution:")
            for state, count in sample_counts.items():
                probability = count / 1000
                print(f"  {state}: {probability:.3f}")

            # Should strongly favor tiger_left due to evidence weighting

        Configuration ID and caching::

            from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
            from POMDPPlanners.core.belief import WeightedParticleBeliefStateUpdate

            env = SanityPOMDP(discount_factor=0.95)

            # Create two beliefs with same particles in different orders
            belief1 = WeightedParticleBeliefStateUpdate([0, 1, 0], [0.8, 0.2, 0.6])
            belief2 = WeightedParticleBeliefStateUpdate([1, 0, 0], [0.2, 0.8, 0.6])

            # Config IDs should be equal (order-invariant)
            print(f"Belief 1 config ID: {belief1.config_id}")
            print(f"Belief 2 config ID: {belief2.config_id}")
            print(f"IDs match: {belief1.config_id == belief2.config_id}")

            # Useful for caching in planning algorithms
            belief_cache = {belief1.config_id: "cached_result"}

            if belief2.config_id in belief_cache:
                print("Cache hit! Reusing computation.")

        Advanced: Custom particle types::

            import numpy as np
            from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
            from POMDPPlanners.core.belief import WeightedParticleBeliefStateUpdate

            # Works with any particle type - numpy arrays, custom objects, etc.
            noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
            env = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov)

            # Numpy array particles
            particles = [
                np.array([0.0, 0.0, 0.1, 0.0]),
                np.array([0.1, 0.0, 0.08, 0.0]),
                np.array([-0.1, 0.0, 0.12, 0.0])
            ]
            weights = [0.4, 0.35, 0.25]

            belief = WeightedParticleBeliefStateUpdate(particles, weights)

            # Add more complex state
            complex_state = np.array([0.05, 0.1, 0.09, -0.05])
            action = np.array([1])  # Force right
            observation = np.array([0.06, 0.1, 0.088, -0.05])

            new_belief = belief.update(action, observation, env, complex_state)
            sampled_state = new_belief.sample()

            print(f"Sampled state shape: {sampled_state.shape}")
            print(f"Sampled state: {sampled_state}")
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

        Example:
            Creating a new belief with an additional particle::\

                # Original belief with 2 particles
                belief = WeightedParticleBeliefStateUpdate(
                    particles=["state1", "state2"],
                    weights=[0.7, 0.3]
                )

                # Create new belief with additional particle
                new_belief = belief.update(
                    action="action1",
                    observation="obs1",
                    pomdp=environment,
                    state="state3"
                )

                # Original belief unchanged, new belief has 3 particles
                assert len(belief.particles) == 2
                assert len(new_belief.particles) == 3
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

        Example:
            Incrementally building a belief::\

                belief = WeightedParticleBeliefStateUpdate([], [])

                # Add particles one by one
                belief.inplace_update("listen", "hear_left", env, "tiger_left")
                belief.inplace_update("listen", "hear_right", env, "tiger_right")
                belief.inplace_update("listen", "hear_left", env, "tiger_left")

                # Belief now contains 3 particles with observation-based weights
                assert len(belief.particles) == 3
                assert belief.weights_sum > 0
        """
        if state is None:
            raise ValueError("state cannot be None")

        self.particles.append(state)
        observation_probability = pomdp.observation_model(
            next_state=state, action=action
        ).probability([observation])[0]
        self.weights.append(float(observation_probability))
        self.weights_sum = float(self.weights_sum) + float(observation_probability)

    def sample(self) -> Any:
        """Sample a state from the current belief distribution.

        Returns:
            A state sampled according to the belief's probability distribution

        Raises:
            ValueError: If belief is empty or has zero weights.
        """
        if not self.particles or self.weights_sum == 0:
            raise ValueError("Cannot sample from empty or unnormalized belief")

        # Normalize weights for sampling
        normalized_weights = [w / float(self.weights_sum) for w in self.weights]

        # Sample based on normalized weights
        return random.choices(self.particles, weights=normalized_weights, k=1)[0]

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
            elif isinstance(value, (str, int, float, bool)):
                return value
            elif isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            elif isinstance(value, dict):
                return {str(k): serialize_value(v) for k, v in sorted(value.items())}
            elif hasattr(value, "__dict__"):
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

    Examples:
        Basic uniform belief construction::

            from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            from POMDPPlanners.core.belief import UnweightedParticleBeliefStateUpdate

            # Create environment and empty uniform belief
            env = TigerPOMDP(discount_factor=0.95)
            belief = UnweightedParticleBeliefStateUpdate(particles=[])

            # Add states uniformly (all have equal probability)
            belief.inplace_update("listen", "hear_left", env, "tiger_left")
            belief.inplace_update("listen", "hear_right", env, "tiger_right")
            belief.inplace_update("listen", "hear_left", env, "tiger_left")

            print(f"Belief has {len(belief.particles)} particles")
            print(f"Weights sum: {belief.weights_sum}")  # Equal to number of particles

            # Sample uniformly from particles
            sampled_state = belief.sample()  # Each "tiger_left" has 2/3 prob, "tiger_right" has 1/3

        Monte Carlo Tree Search with uniform beliefs::

            from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
            from POMDPPlanners.core.belief import UnweightedParticleBeliefStateUpdate
            from POMDPPlanners.core.tree import BeliefNode, ActionNode
            import random

            env = SanityPOMDP(discount_factor=0.95)

            # Root belief with uniform initial distribution
            root_belief = UnweightedParticleBeliefStateUpdate(
                particles=[0, 1, 0, 1, 0]  # More 0s than 1s, but all weighted equally
            )
            root_node = BeliefNode(belief=root_belief)

            # Simulate MCTS node expansion
            action = 0  # Good action
            possible_observations = [0, 1]  # Discrete observations

            # Create action node
            action_node = ActionNode(action=action, parent=root_node)

            # For each observation, accumulate child belief
            for observation in possible_observations:
                child_belief = UnweightedParticleBeliefStateUpdate([])

                # Add particles uniformly based on environment dynamics
                for _ in range(10):  # Multiple simulations
                    parent_state = root_belief.sample()
                    next_state = env.state_transition_model(parent_state, action).sample()[0]
                    child_belief.inplace_update(action, observation, env, next_state)

                # Create belief node
                belief_node = BeliefNode(belief=child_belief, observation=observation, parent=action_node)
                print(f"Child for obs {observation}: {len(child_belief.particles)} particles")

        Comparing weighted vs unweighted belief updates::

            from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
            from POMDPPlanners.core.belief import (
                WeightedParticleBeliefStateUpdate, UnweightedParticleBeliefStateUpdate
            )
            import collections

            env = SanityPOMDP(discount_factor=0.95)

            # Same particle sequence for both belief types
            states = [0, 1, 0, 1, 0, 1, 0]  # More 0s (good states) than 1s (bad states)
            observations = [0, 1, 0, 0, 1, 0, 1]
            action = 0

            # Weighted belief considers observation likelihoods
            weighted_belief = WeightedParticleBeliefStateUpdate([], [])
            for state, obs in zip(states, observations):
                weighted_belief.inplace_update(action, obs, env, state)

            # Unweighted belief treats all particles equally
            unweighted_belief = UnweightedParticleBeliefStateUpdate([])
            for state, obs in zip(states, observations):
                unweighted_belief.inplace_update(action, obs, env, state)

            # Compare sampling distributions
            print("Weighted belief sampling:")
            weighted_samples = [weighted_belief.sample() for _ in range(1000)]
            weighted_counts = collections.Counter(weighted_samples)
            for state, count in sorted(weighted_counts.items()):
                print(f"  State {state}: {count/1000:.3f}")

            print("\\nUnweighted belief sampling:")
            unweighted_samples = [unweighted_belief.sample() for _ in range(1000)]
            unweighted_counts = collections.Counter(unweighted_samples)
            for state, count in sorted(unweighted_counts.items()):
                print(f"  State {state}: {count/1000:.3f}")

        Discrete observation filtering::

            from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            from POMDPPlanners.core.belief import UnweightedParticleBeliefStateUpdate
            import collections

            env = TigerPOMDP(discount_factor=0.95)
            belief = UnweightedParticleBeliefStateUpdate([])

            # Simulate multiple observations (discrete: hear_left or hear_right)
            observation_sequence = [
                ("tiger_left", "hear_left"),    # Consistent evidence
                ("tiger_left", "hear_left"),    # More consistent evidence
                ("tiger_right", "hear_left"),   # Inconsistent evidence
                ("tiger_left", "hear_right"),   # Inconsistent evidence
                ("tiger_left", "hear_left"),    # Back to consistent
                ("tiger_right", "hear_right"),  # Consistent for right
            ]

            for state, obs in observation_sequence:
                belief.inplace_update("listen", obs, env, state)

            print(f"Final belief has {len(belief.particles)} particles")

            # Analyze uniform distribution over accumulated particles
            samples = [belief.sample() for _ in range(1000)]
            sample_counts = collections.Counter(samples)

            print("Uniform belief distribution:")
            for state, count in sample_counts.items():
                probability = count / 1000
                particle_count = belief.particles.count(state)
                expected_prob = particle_count / len(belief.particles)
                print(f"  {state}: {probability:.3f} (expected: {expected_prob:.3f})")

        Immutable belief trees for planning::

            import numpy as np
            from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
            from POMDPPlanners.core.belief import UnweightedParticleBeliefStateUpdate

            # Create continuous state environment
            noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
            env = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov)

            # Start with uniform belief over multiple initial states
            initial_states = [
                np.array([0.0, 0.0, 0.1, 0.0]),   # Balanced pole
                np.array([0.1, 0.0, 0.08, 0.0]),  # Slightly right
                np.array([-0.1, 0.0, 0.12, 0.0])  # Slightly left
            ]

            root_belief = UnweightedParticleBeliefStateUpdate(initial_states)

            # Generate child beliefs for different actions (functional style)
            actions = [0, 1]  # Push left or right
            child_beliefs = {}

            for action in actions:
                child_belief = UnweightedParticleBeliefStateUpdate([])

                # Generate next states uniformly
                for _ in range(5):  # Multiple rollouts per action
                    current_state = root_belief.sample()
                    next_state = env.state_transition_model(current_state, action).sample()[0]
                    # For simplicity, assume observation equals next state (fully observable case)
                    child_belief = child_belief.update(action, next_state, env, next_state)

                child_beliefs[action] = child_belief
                print(f"Child belief for action {action}: {len(child_belief.particles)} particles")

            # All child beliefs maintain uniform distribution over their particles
            for action, child_belief in child_beliefs.items():
                sample = child_belief.sample()
                print(f"Sample from action {action} belief: {sample[:2]}")  # First 2 components

        Memory-efficient particle accumulation::

            from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
            from POMDPPlanners.core.belief import (
                WeightedParticleBeliefStateUpdate, UnweightedParticleBeliefStateUpdate
            )
            import sys

            env = SanityPOMDP(discount_factor=0.95)

            # Compare memory usage between weighted and unweighted beliefs
            weighted_belief = WeightedParticleBeliefStateUpdate([], [])
            unweighted_belief = UnweightedParticleBeliefStateUpdate([])

            # Add many particles
            states = [0, 1] * 1000  # 2000 particles
            for state in states:
                weighted_belief.inplace_update("action", 0, env, state)
                unweighted_belief.inplace_update("action", 0, env, state)

            # Check memory usage (unweighted should use less memory)
            weighted_size = sys.getsizeof(weighted_belief.particles) + sys.getsizeof(weighted_belief.weights)
            unweighted_size = sys.getsizeof(unweighted_belief.particles)

            print(f"Weighted belief memory: {weighted_size} bytes")
            print(f"Unweighted belief memory: {unweighted_size} bytes")
            print(f"Memory savings: {((weighted_size - unweighted_size) / weighted_size * 100):.1f}%")

            # Both have same number of particles but unweighted saves memory on weights
            assert len(weighted_belief.particles) == len(unweighted_belief.particles)

        Configuration caching and equality::

            from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
            from POMDPPlanners.core.belief import UnweightedParticleBeliefStateUpdate

            env = SanityPOMDP(discount_factor=0.95)

            # Create two beliefs with same particles in different orders
            belief1 = UnweightedParticleBeliefStateUpdate([0, 1, 0, 1, 0])
            belief2 = UnweightedParticleBeliefStateUpdate([1, 0, 1, 0, 0])

            # Config IDs should be equal (order-invariant)
            print(f"Belief 1 config ID: {belief1.config_id[:8]}...")
            print(f"Belief 2 config ID: {belief2.config_id[:8]}...")
            print(f"IDs match: {belief1.config_id == belief2.config_id}")

            # Test belief equality
            print(f"Beliefs equal: {belief1 == belief2}")

            # Useful for caching in tree search algorithms
            belief_cache = {belief1: "cached_computation"}

            if belief2 in belief_cache:
                print("Cache hit! Beliefs are equivalent.")
            else:
                print("Cache miss - beliefs differ.")

        Large-scale particle accumulation::

            from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            from POMDPPlanners.core.belief import UnweightedParticleBeliefStateUpdate
            import time
            import collections

            env = TigerPOMDP(discount_factor=0.95)
            belief = UnweightedParticleBeliefStateUpdate([])

            # Time large-scale particle addition
            start_time = time.time()

            # Add 10000 particles uniformly
            states = ["tiger_left", "tiger_right"]
            observations = ["hear_left", "hear_right"]

            for i in range(10000):
                state = states[i % 2]  # Alternate between states
                obs = observations[i % 2]  # Alternate between observations
                belief.inplace_update("listen", obs, env, state)

            end_time = time.time()

            print(f"Added {len(belief.particles)} particles in {end_time - start_time:.3f} seconds")
            print(f"Rate: {len(belief.particles) / (end_time - start_time):.0f} particles/second")

            # Verify uniform distribution
            samples = [belief.sample() for _ in range(1000)]
            sample_counts = collections.Counter(samples)

            for state, count in sample_counts.items():
                print(f"{state}: {count/1000:.3f} (expected: 0.5)")
    """

    def __init__(self, particles: list = []):
        """Initialize unweighted particle belief.

        Creates a belief state with uniform probability distribution over
        the provided particles. Each particle has equal weight 1/N where
        N is the number of particles.

        Args:
            particles: List of state particles with uniform weights.
                Defaults to empty list for incremental construction.

        Example:
            Creating beliefs with different initialization strategies::\

                # Empty belief for incremental construction
                empty_belief = UnweightedParticleBeliefStateUpdate([])

                # Pre-populated belief with uniform distribution
                states = ["state1", "state2", "state1", "state3"]
                belief = UnweightedParticleBeliefStateUpdate(states)

                # Each state has probability proportional to its count
                # state1: 2/4 = 0.5, state2: 1/4 = 0.25, state3: 1/4 = 0.25
        """
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

        Example:
            Creating new beliefs immutably::\

                # Original belief with 3 particles
                belief = UnweightedParticleBeliefStateUpdate(["state1", "state2", "state1"])

                # Create new belief with additional particle
                new_belief = belief.update("action", "obs", environment, "state3")

                # Original belief unchanged, new belief has 4 particles
                assert len(belief.particles) == 3
                assert len(new_belief.particles) == 4

                # All particles have equal probability in new belief
                # state1: 2/4 = 0.5, state2: 1/4 = 0.25, state3: 1/4 = 0.25
        """
        new_particles = self.particles + [state]
        return UnweightedParticleBeliefStateUpdate(particles=new_particles)

    def sample(self) -> Any:
        """Sample a state uniformly from the current belief distribution.

        Returns:
            A state sampled uniformly from the particle set

        Raises:
            IndexError: If belief is empty (no particles to sample from)

        Example:
            Uniform sampling from particle belief::\

                # Create belief with repeated particles
                particles = ["A", "B", "A", "A", "C"]
                belief = UnweightedParticleBeliefStateUpdate(particles)

                # Sample multiple times to see distribution
                samples = [belief.sample() for _ in range(1000)]
                from collections import Counter
                counts = Counter(samples)

                # Should approximate: A: 60%, B: 20%, C: 20%
                for state, count in counts.items():
                    print(f"{state}: {count/1000:.2f}")
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

        Example:
            Incrementally building uniform beliefs::\

                belief = UnweightedParticleBeliefStateUpdate([])

                # Add particles one by one - all have equal weight
                belief.inplace_update("action1", "obs1", env, "state_A")
                belief.inplace_update("action2", "obs2", env, "state_B")
                belief.inplace_update("action3", "obs3", env, "state_A")

                # Belief now has 3 particles: ["state_A", "state_B", "state_A"]
                # Sampling probabilities: state_A = 2/3, state_B = 1/3
                assert len(belief.particles) == 3
                assert belief.weights_sum == 3  # Equal to particle count
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
            elif isinstance(value, (str, int, float, bool)):
                return value
            elif isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            elif isinstance(value, dict):
                return {str(k): serialize_value(v) for k, v in sorted(value.items())}
            elif hasattr(value, "__dict__"):
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


def sample_next_belief(belief: Belief, action: Any, pomdp: "Environment") -> Tuple[Belief, Any]:
    """Simulate one step of belief evolution.

    This function samples a state from the current belief, simulates the
    environment dynamics, and updates the belief with the resulting observation.

    Args:
        belief: Current belief state
        action: Action to execute
        pomdp: Environment providing dynamics models

    Returns:
        Tuple containing:
            - Updated belief after incorporating the observation
            - Observation that was generated
    """
    state = belief.sample()
    next_state = pomdp.state_transition_model(state=state, action=action).sample()[0]
    observation = pomdp.observation_model(next_state=next_state, action=action).sample()[0]

    next_belief = belief.update(action=action, observation=observation, pomdp=pomdp)

    return next_belief, observation


def get_initial_belief(
    pomdp: Environment, n_particles: int, resampling: bool = True
) -> WeightedParticleBelief:
    """Create initial belief from environment's initial state distribution.

    Args:
        pomdp: Environment to get initial distribution from
        n_particles: Number of particles to generate for the belief
        resampling: Enable resampling in the created belief. Defaults to True.

    Returns:
        WeightedParticleBelief with uniform weights over initial states

    Raises:
        TypeError: If n_particles is not an integer
        ValueError: If n_particles is not positive
    """
    if not isinstance(n_particles, int):
        raise TypeError("n_particles must be an integer")
    if n_particles <= 0:
        raise ValueError("n_particles must be greater than 0")

    particles = pomdp.initial_state_dist().sample(n_samples=n_particles)
    log_weights = np.log(np.ones(n_particles) / n_particles)

    return WeightedParticleBelief(
        particles=particles, log_weights=log_weights, resampling=resampling
    )


def is_terminal_particle_belief(
    belief: Union[
        WeightedParticleBelief,
        WeightedParticleBeliefStateUpdate,
        UnweightedParticleBeliefStateUpdate,
    ],
    env: Environment,
) -> bool:
    """Check if the belief is terminal."""
    return all(env.is_terminal(particle) for particle in belief.particles)


def is_terminal_belief(belief: Belief, env: Environment) -> bool:
    """Check if the belief is terminal."""
    if isinstance(
        belief,
        (
            WeightedParticleBelief,
            WeightedParticleBeliefStateUpdate,
            UnweightedParticleBeliefStateUpdate,
        ),
    ):
        return is_terminal_particle_belief(belief=belief, env=env)
    else:
        raise NotImplementedError("is_terminal_belief is not implemented for this belief type")
