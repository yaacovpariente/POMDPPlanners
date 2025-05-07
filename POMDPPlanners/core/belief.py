from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import random
import hashlib
import json

import numpy as np

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.distributions import DiscreteDistribution

class Belief(ABC):
    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on belief configuration."""
        config_dict = {}
        
        # Include all public attributes that aren't callables
        for key, value in self.__dict__.items():
            if key.startswith('_') or callable(value):
                continue
                
            # Handle numpy arrays
            if isinstance(value, np.ndarray):
                config_dict[key] = value.tolist()
            # Handle basic Python types
            elif isinstance(value, (str, int, float, bool, list, tuple)):
                config_dict[key] = value
            # Handle dictionaries
            elif isinstance(value, dict):
                # Only include serializable values
                serializable_dict = {}
                for k, v in value.items():
                    if isinstance(v, (str, int, float, bool, list, tuple)):
                        serializable_dict[k] = v
                    elif isinstance(v, np.ndarray):
                        serializable_dict[k] = v.tolist()
                config_dict[key] = serializable_dict
                
        # Sort dictionary to ensure consistent ordering
        config_dict = dict(sorted(config_dict.items()))
        
        # Create a deterministic string representation and hash it
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def __hash__(self) -> int:
        """Make the belief hashable by using its config_id."""
        return int(self.config_id, 16)  # Convert hex string to integer

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
            next_s = pomdp.state_transition_model(state=s, action=action).sample()
            obs = pomdp.observation_model(next_state=next_s, action=action).sample()
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
            next_s = pomdp.state_transition_model(state=s, action=action).sample()
            obs = pomdp.observation_model(next_state=next_s, action=action).sample()
            if pomdp.is_equal_observation(obs, observation):
                return next_s

    def sample(self):
        return random.choice(self.particles)
    
    @abstractmethod
    def _reinvigoration_pertubation(self, action: Any, observation: Any, pomdp: Environment) -> Any:
        """This method should be implemented specifically for each environment."""
        pass

# class UnweightedParticleBelief(Belief):
#     def __init__(self, particles: list):
#         assert isinstance(particles, list)
#         self.particles = particles

#     def update(self, action, observation, pomdp: Environment) -> "UnweightedParticleBelief":
#         state = self.sample()
#         next_state = pomdp.state_transition_model(state=state, action=action).sample()
#         next_observation = pomdp.observation_model(next_state=next_state, action=action).sample()
                

#     def sample(self):
#         idx = random.randint(0, len(self.particles))
#         return self.particles[idx]

class WeightedParticleBelief(Belief):
    def __init__(
        self, particles: list, log_weights: np.ndarray, resampling: bool = False, ess_threshold: float = 0.5
    ):
        assert isinstance(particles, list)
        assert isinstance(log_weights, np.ndarray)
        assert len(particles) == len(log_weights)
        assert isinstance(resampling, bool)
        assert sum(log_weights != 0) > 0
        assert np.all(
            np.isfinite(log_weights)
        ), "log_weights must be finite numbers (not Inf, -Inf, or NaN)"

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
        config_dict = {}
        
        # Create a list of particle-weight pairs to maintain order
        particle_weight_pairs = []
        for particle, weight in zip(self.particles, self.log_weights):
            # Convert particle to a serializable format if needed
            if isinstance(particle, np.ndarray):
                particle = particle.tolist()
            particle_weight_pairs.append((particle, float(weight)))
        
        # Sort particle-weight pairs to make config_id invariant to order
        particle_weight_pairs.sort(key=lambda x: (str(x[0]), x[1]))
        
        config_dict['particle_weight_pairs'] = particle_weight_pairs
        config_dict['resampling'] = self.resampling
        config_dict['ess_threshold'] = self.ess_threshold
        
        # Sort dictionary to ensure consistent ordering
        config_dict = dict(sorted(config_dict.items()))
        
        # Create a deterministic string representation and hash it
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def update(self, action, observation, pomdp: Environment) -> "WeightedParticleBelief":
        next_particles = [
            pomdp.state_transition_model(state=particle, action=action).sample()
            for particle in self.particles
        ]
        next_log_weights = self.log_weights + np.log(
            self.eps
            + np.array(
                [
                    pomdp.observation_model(
                        next_state=next_particle, action=action
                    ).probability(observation)
                    for next_particle in next_particles
                ]
            )
        )

        if self.resampling:
            effective_sample_size = 1 / np.sum(np.square(self.normalized_weights))
            if effective_sample_size < self.ess_threshold:
                normalized_next_weights = np.exp(
                    next_log_weights - np.max(next_log_weights)
                )
                normalized_next_weights = normalized_next_weights / np.sum(
                    normalized_next_weights
                )
                sampled_indexes = random.choices(
                    range(len(next_particles)),
                    weights=normalized_next_weights,
                    k=len(next_particles),
                )
                state_sample = [next_particles[i] for i in sampled_indexes]
                next_log_weights = np.log(np.ones(len(state_sample)) / len(state_sample))
                next_particles = state_sample
                
        return WeightedParticleBelief(
            particles=next_particles,
            log_weights=next_log_weights,
            resampling=self.resampling,
        )

    def sample(self):
        idx = np.random.choice(len(self.particles), p=self.normalized_weights)
        return self.particles[idx]

class WeightedParticleBeliefReinvigoration(WeightedParticleBelief, ABC):
    def __init__(self, particles: list, log_weights: np.ndarray, resampling: bool = False, ess_threshold: float = 0.5, reinvigoration_fraction: float = 0.2):
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
    next_state = pomdp.state_transition_model(state=state, action=action).sample()
    observation = pomdp.observation_model(next_state=next_state, action=action).sample()

    next_belief = belief.update(action=action, observation=observation, pomdp=pomdp)

    return next_belief, observation


def get_initial_belief(
    pomdp: Environment, n_particles: int, resampling: bool = False
) -> Belief:
    assert isinstance(n_particles, int)
    assert n_particles > 0

    particles = [pomdp.initial_state_dist().sample() for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)

    return WeightedParticleBelief(
        particles=particles, log_weights=log_weights, resampling=resampling
    )
