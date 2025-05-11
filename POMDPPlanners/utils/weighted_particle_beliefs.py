from typing import Any

import numpy as np

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.belief import WeightedParticleBeliefReinvigoration

class WeightedParticleBeliefDiscreteLightDark(WeightedParticleBeliefReinvigoration):
    def __init__(
        self, 
        particles: list, 
        log_weights: np.ndarray, 
        resampling: bool = False, 
        ess_threshold: float = 0.5,
        reinvigoration_fraction: float = 0.2
    ):
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            resampling=resampling,
            ess_threshold=ess_threshold
        )
        
        self.reinvigoration_fraction = reinvigoration_fraction
        self.action_to_vector = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "right": np.array([1, 0]),
            "left": np.array([-1, 0]),
        }
        
        self.actions = ["up", "down", "right", "left"]

    def reinvigorate(self, action: Any, observation: Any, pomdp: Environment, belief: WeightedParticleBeliefReinvigoration) -> WeightedParticleBeliefReinvigoration:
        effective_sample_size = 1 / np.sum(np.square(self.normalized_weights))
        
        if effective_sample_size > self.ess_threshold:
            states = [observation + self.action_to_vector[action] for action in self.actions]
            states.append(observation)
            
            n_reinvigorate = int(self.reinvigoration_fraction * len(belief.particles))
            reinvigorate_indices = np.random.choice(len(states), size=n_reinvigorate, replace=True)
            reinvigorated_states = [states[i] for i in reinvigorate_indices]
            
            replace_indices = np.random.choice(len(belief.particles), size=n_reinvigorate, replace=True)
            belief.particles[:len(replace_indices)] = reinvigorated_states
                
        return belief
    
class WeightedParticleBeliefDiscreteLightDarkFullCoverage(WeightedParticleBeliefReinvigoration):
    def __init__(
        self, 
        particles: list, 
        log_weights: np.ndarray, 
        ess_threshold: float = 0.5,
        reinvigoration_fraction: float = 0.05
    ):
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            resampling=False, # resampling is done in only the reinvigorate method
            ess_threshold=ess_threshold
        )
        
        self.reinvigoration_particles_weights_sum = reinvigoration_fraction
        
        self.action_to_vector = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "right": np.array([1, 0]),
            "left": np.array([-1, 0]),
        }
        
        self.actions = ["up", "down", "right", "left"]

    def reinvigorate(self, action: Any, observation: Any, pomdp: Environment, belief: WeightedParticleBeliefReinvigoration) -> WeightedParticleBeliefReinvigoration:
        self.particles, self.log_weights = self._resample(particles=self.particles, log_weights=self.log_weights)
            
        states = [observation + self.action_to_vector[action] for action in self.actions]
        states.append(observation)
        
        n_states = len(states)
        self.particles[-n_states:] = states
        
        return self

class WeightedParticleBeliefContinuousLightDarkFullCoverage(WeightedParticleBeliefReinvigoration):
    def __init__(
        self, 
        particles: list, 
        log_weights: np.ndarray, 
        ess_threshold: float = 0.5,
        reinvigoration_fraction: float = 0.05,
        reinvigoration_cov_matrix: np.ndarray = np.eye(2)
    ):
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            resampling=False, # resampling is done in only the reinvigorate method
            ess_threshold=ess_threshold
        )
        
        self.reinvigoration_particles_weights_sum = reinvigoration_fraction
        self.reinvigoration_cov_matrix = reinvigoration_cov_matrix
        
        self.action_to_vector = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "right": np.array([1, 0]),
            "left": np.array([-1, 0]),
        }
        
        self.actions = ["up", "down", "right", "left"]

    def reinvigorate(self, action: Any, observation: Any, pomdp: Environment, belief: WeightedParticleBeliefReinvigoration) -> WeightedParticleBeliefReinvigoration:
        self.particles, self.log_weights = self._resample(particles=self.particles, log_weights=self.log_weights)
            
        states = [observation + self.action_to_vector[action] for action in self.actions]
        states.append(observation)
        
        n_reinvigorate = int(self.reinvigoration_particles_weights_sum * len(self.particles))
        
        if n_reinvigorate > 0:  # Only proceed if we need to reinvigorate
            # Sample centers for each particle to reinvigorate
            center_indices = np.random.randint(0, len(states), size=n_reinvigorate)
            centers = np.array([states[i] for i in center_indices])
            
            # Sample all particles at once
            reinvigorated_states = np.random.multivariate_normal(
                mean=np.zeros(2),  # We'll add the centers after sampling
                cov=self.reinvigoration_cov_matrix,
                size=n_reinvigorate
            ) + centers
            
            # Clip to ensure within grid bounds
            reinvigorated_states = np.clip(reinvigorated_states, 0, pomdp.grid_size)
            
            self.particles[-n_reinvigorate:] = reinvigorated_states.tolist()
        
        return self
