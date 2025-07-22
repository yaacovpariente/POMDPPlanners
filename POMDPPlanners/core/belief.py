from abc import ABC, abstractmethod
from typing import Tuple, Any
import random

import numpy as np

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.utils.config_to_id import config_to_id

class Belief(ABC):
    @classmethod
    def from_config(cls, config):
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
            """Helper function to serialize values in a deterministic way."""
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (str, int, float, bool)):
                return value
            elif isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            elif isinstance(value, dict):
                return {str(k): serialize_value(v) for k, v in sorted(value.items())}
            elif hasattr(value, '__dict__'):
                return serialize_value(value.__dict__)
            else:
                return str(value)
        
        config_dict = {}
        for key, value in self.__dict__.items():
            if key.startswith('_') or callable(value):
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

    @abstractmethod
    def update(self, action, observation, pomdp: Environment) -> "Belief":
        pass

    @abstractmethod
    def sample(self):
        pass

class UnweightedParticleBelief(Belief):
    def __init__(self, particles: list, reinvigoration_fraction=0.2):
        self.num_particles = len(particles)
        self.reinvigoration_fraction = reinvigoration_fraction
        self.particles = particles

    def update(self, action: Any, observation: Any, pomdp: Environment, is_reinvigorate: bool = False) -> "UnweightedParticleBelief":
        new_particles = []

        for _ in range(self.num_particles):
            # Sample a particle and simulate its transition
            s = random.choice(self.particles)
            next_s = pomdp.state_transition_model(state=s, action=action).sample()[0]
            obs = pomdp.observation_model(next_state=next_s, action=action).sample()[0]
            if pomdp.is_equal_observation(obs, observation):
                new_particles.append(next_s)

        # Reinvigorate if degeneracy detected (too few new particles)
        if is_reinvigorate:
            if len(new_particles) < self.reinvigoration_fraction * self.num_particles:
                num_new = int(self.reinvigoration_fraction * self.num_particles)
                reinvigorated = [self.reinvigorate(action=action, observation=observation, pomdp=pomdp) for _ in range(num_new)]
                new_particles += reinvigorated

        # Replenish to full count
        while len(new_particles) < self.num_particles:
            new_particles.append(random.choice(new_particles))

        self.particles = new_particles

    def reinvigorate(self, action: Any, observation: Any, pomdp: Environment):
        """Simulate a new particle that matches the action-observation pair."""
        # Try sampling from initial state and simulate until match
        while True:
            s = self._reinvigoration_pertubation(action=action, observation=observation, pomdp=pomdp)
            next_s = pomdp.state_transition_model(state=s, action=action).sample()[0]
            obs = pomdp.observation_model(next_state=next_s, action=action).sample()[0]
            if pomdp.is_equal_observation(obs, observation):
                return next_s

    def sample(self):
        return random.choice(self.particles)
    
    @abstractmethod
    def _reinvigoration_pertubation(self, action: Any, observation: Any, pomdp: Environment) -> Any:
        """This method should be implemented specifically for each environment."""
        pass

class WeightedParticleBelief(Belief):
    def __init__(
        self, particles: list, log_weights: np.ndarray, resampling: bool = False, ess_threshold: float = 0.5
    ):
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

        self.particles = particles
        self.log_weights = log_weights
        # First subtract max for numerical stability, then normalize to sum to 1
        self.normalized_weights = np.exp(self.log_weights - np.max(self.log_weights))
        self.normalized_weights = self.normalized_weights / np.sum(
            self.normalized_weights
        )
        self.resampling = resampling
        self.ess_threshold = ess_threshold

        self.eps = 1e-10

    def to_dict(self) -> dict:
        """Convert the belief to a dictionary for serialization.
        
        Returns:
            dict: A dictionary containing all necessary fields for deserialization.
        """
        return {
            'particles': self.particles,
            'log_weights': self.log_weights.tolist(),
            'resampling': self.resampling,
            'ess_threshold': self.ess_threshold
        }

    def to_unique_support_distribution(self) -> "DiscreteDistribution":
        """Convert the belief to a DiscreteDistribution with unique particles.
        
        Returns:
            DiscreteDistribution: A distribution where each particle appears only once,
            with its probability being the sum of all its occurrences in the original belief.
        """
        
        # Create a dictionary to store unique particles and their combined weights
        unique_particles = {}
        
        # Iterate through particles and their weights
        for particle, weight in zip(self.particles, self.normalized_weights):
            # Convert particle to tuple if it's a numpy array for hashability
            if isinstance(particle, np.ndarray):
                particle_key = tuple(particle.tolist())
            else:
                particle_key = particle
                
            # Add or update the weight for this particle
            if particle_key in unique_particles:
                unique_particles[particle_key] += weight
            else:
                unique_particles[particle_key] = weight
        
        # Convert back to original particle types and create arrays
        particles = []
        weights = []
        for particle_key, weight in unique_particles.items():
            if isinstance(particle_key, tuple):
                particles.append(np.array(particle_key))
            else:
                particles.append(particle_key)
            weights.append(weight)
            
        # Convert to numpy array and normalize to ensure sum is exactly 1
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize to sum to 1
        
        return DiscreteDistribution(values=particles, probs=weights)

    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on belief configuration."""
        def serialize_value(value):
            """Helper function to serialize values in a deterministic way."""
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (str, int, float, bool)):
                return value
            elif isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            elif isinstance(value, dict):
                return {str(k): serialize_value(v) for k, v in sorted(value.items())}
            elif hasattr(value, '__dict__'):
                return serialize_value(value.__dict__)
            else:
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
            'particle_weight_pairs': particle_weight_pairs,
            'resampling': self.resampling,
            'ess_threshold': self.ess_threshold
        }
        config_dict = dict(sorted(config_dict.items()))
        return config_to_id(config_dict)

    def _resample(self, particles: list, log_weights: np.ndarray) -> Tuple[list, np.ndarray]:
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
    
    def _update_weights(self, action: Any, observation: Any, pomdp: Environment) -> Tuple[np.ndarray, np.ndarray]:
        next_particles = [
            pomdp.state_transition_model(state=particle, action=action).sample()[0]
            for particle in self.particles
        ]
        probs = np.array([
            pomdp.observation_model(
                next_state=next_particle, action=action
            ).probability([observation])[0]
            for next_particle in next_particles
        ])
        
        next_log_weights = self.log_weights + np.log(self.eps + probs)

        return next_particles, next_log_weights

    def update(self, action, observation, pomdp: Environment) -> "WeightedParticleBelief":
        next_particles, next_log_weights = self._update_weights(action=action, observation=observation, pomdp=pomdp)
        
        if self.resampling:
            next_particles, next_log_weights = self._resample(next_particles, next_log_weights)
                
        return WeightedParticleBelief(
            particles=next_particles,
            log_weights=next_log_weights,
            resampling=self.resampling,
        )

    def sample(self):
        idx = np.random.choice(len(self.particles), p=self.normalized_weights)
        return self.particles[idx]

class WeightedParticleBeliefReinvigoration(WeightedParticleBelief, ABC):
    def __init__(self, particles: list, log_weights: np.ndarray, resampling: bool = True, ess_threshold: float = 0.5, reinvigoration_fraction: float = 0.2):
        super().__init__(
            particles=particles, 
            log_weights=log_weights, 
            resampling=resampling, 
            ess_threshold=ess_threshold
        )
        
        self.reinvigoration_fraction = reinvigoration_fraction
        
    def update(self, action, observation, pomdp: Environment) -> "WeightedParticleBelief":
        belief = super().update(
            action=action,
            observation=observation,
            pomdp=pomdp
        )
        
        belief = self.reinvigorate(
            action=action,
            observation=observation,
            pomdp=pomdp,
            belief=belief
        )
        
        return belief
    
    @abstractmethod
    def reinvigorate(self, action: Any, observation: Any, pomdp: Environment, belief: Belief) -> Belief:
        """This function should be implemented for a specific POMDP environment."""
        pass
        

def sample_next_belief(
    belief: Belief, action: Any, pomdp: "Environment"
) -> Tuple[Belief, Any]:
    state = belief.sample()
    next_state = pomdp.state_transition_model(state=state, action=action).sample()[0]
    observation = pomdp.observation_model(next_state=next_state, action=action).sample()[0]

    next_belief = belief.update(action=action, observation=observation, pomdp=pomdp)

    return next_belief, observation


def get_initial_belief(
    pomdp: Environment, n_particles: int, resampling: bool = True
) -> Belief:
    if not isinstance(n_particles, int):
        raise TypeError("n_particles must be an integer")
    if n_particles <= 0:
        raise ValueError("n_particles must be greater than 0")

    particles = pomdp.initial_state_dist().sample(n_samples=n_particles)
    log_weights = np.log(np.ones(n_particles) / n_particles)

    return WeightedParticleBelief(
        particles=particles, log_weights=log_weights, resampling=resampling
    )
