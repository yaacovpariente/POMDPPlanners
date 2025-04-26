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
            n_reinvigorate = int(self.reinvigoration_fraction * len(belief.particles))
            reinvigorate_indices = np.random.choice(len(states), size=n_reinvigorate, replace=True)
            reinvigorated_states = [states[i] for i in reinvigorate_indices]
            
            replace_indices = np.random.choice(len(belief.particles), size=n_reinvigorate, replace=True)
            belief.particles[:len(replace_indices)] = reinvigorated_states
                
        return belief