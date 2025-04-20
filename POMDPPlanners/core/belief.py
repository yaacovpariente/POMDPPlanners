from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import random

import numpy as np

from POMDPPlanners.core.environment import Environment

class Belief(ABC):
    @abstractmethod
    def update(self, action, observation, pomdp: Environment) -> 'Belief':
        pass
    
    @abstractmethod
    def sample(self):
        pass
    
    
class ParticleBelief(Belief):
    def __init__(self, particles: list, log_weights: np.ndarray, resampling: bool = False):
        assert isinstance(particles, list)
        assert isinstance(log_weights, np.ndarray)
        assert len(particles) == len(log_weights)
        assert isinstance(resampling, bool)
        assert sum(log_weights != 0) > 0
        assert np.all(np.isfinite(log_weights)), "log_weights must be finite numbers (not Inf, -Inf, or NaN)"
        
        self.particles = particles
        self.log_weights = log_weights
        # First subtract max for numerical stability, then normalize to sum to 1
        self.normalized_weights = np.exp(self.log_weights - np.max(self.log_weights))
        self.normalized_weights = self.normalized_weights / np.sum(self.normalized_weights)
        self.resampling = resampling
        
        self.eps = 1e-10

    def update(self, action, observation, pomdp: Environment) -> 'ParticleBelief':
        next_particles = [pomdp.state_transition_model(state=particle, action=action).sample() for particle in self.particles]
        next_log_weights = self.log_weights + np.log(self.eps + np.array([
            pomdp.observation_model(next_state=next_particle, action=action).probability(observation) 
            for next_particle in next_particles
        ]))

        if self.resampling:
            normalized_next_weights = np.exp(next_log_weights - np.max(next_log_weights))
            normalized_next_weights = normalized_next_weights / np.sum(normalized_next_weights)
            sampled_indexes = random.choices(range(len(next_particles)), weights=normalized_next_weights, k=len(next_particles))
            state_sample = [next_particles[i] for i in sampled_indexes]
            next_log_weights = np.log(np.ones(len(state_sample)) / len(state_sample))
        else:
            state_sample = next_particles
        
        return ParticleBelief(
            particles=state_sample, 
            log_weights=next_log_weights,
            resampling=self.resampling
        )
    
    def sample(self):
        idx = np.random.choice(len(self.particles), p=self.normalized_weights)
        return self.particles[idx]

def sample_next_belief(belief: Belief, action: Any, pomdp: 'Environment') -> Tuple[Belief, Any]:
    state = belief.sample()
    next_state = pomdp.state_transition_model(state=state, action=action).sample()
    observation = pomdp.observation_model(next_state=next_state, action=action).sample()

    next_belief = belief.update(action=action, observation=observation, pomdp=pomdp)
    
    return next_belief, observation

def get_initial_belief(pomdp: Environment, n_particles: int, resampling: bool = False) -> Belief:
    assert isinstance(n_particles, int)
    assert n_particles > 0
    
    particles = [pomdp.initial_state_dist().sample() for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)
    
    return ParticleBelief(particles=particles, log_weights=log_weights, resampling=resampling)