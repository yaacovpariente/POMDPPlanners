from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import random

import numpy as np

from POMDPPlanners.core.environment import Environment

class Belief(ABC):
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
