from typing import Any, List
import numpy as np

from POMDPPlanners.core.environment import DiscreteActionsEnvironment
from POMDPPlanners.core.distributions import Distribution

class SanityStateTransitionModel(Distribution):
    def __init__(self, state: int, action: int):
        super().__init__()
        self.state = state
        self.action = action
        
    def sample(self) -> int:
        # Action 0 always leads to state 0 (good state)
        # Action 1 always leads to state 1 (bad state)
        return 0 if self.action == 0 else 1

    def probability(self, next_state: int) -> float:
        if self.action == 0:
            return 1.0 if next_state == 0 else 0.0
        else:  # action == 1
            return 1.0 if next_state == 1 else 0.0

class SanityObservationModel(Distribution):
    def __init__(self, next_state: int, action: int):
        super().__init__()
        self.next_state = next_state
        self.action = action
        
    def sample(self) -> int:
        # Observation always matches the state
        return self.next_state

    def probability(self, observation: int) -> float:
        return 1.0 if observation == self.next_state else 0.0

class SanityInitialStateDist(Distribution):
    def sample(self) -> int:
        # Always start in good state (0)
        return 0

    def probability(self, state: int) -> float:
        return 1.0 if state == 0 else 0.0

class SanityInitialObservationDist(Distribution):
    def sample(self) -> int:
        # Initial observation always matches initial state (0)
        return 0

    def probability(self, observation: int) -> float:
        return 1.0 if observation == 0 else 0.0

class SanityPOMDP(DiscreteActionsEnvironment):
    def __init__(self):
        super().__init__(discount_factor=0.95, name="SanityPOMDP")
        
    def state_transition_model(self, state: int, action: int) -> SanityStateTransitionModel:
        return SanityStateTransitionModel(state, action)
    
    def observation_model(self, next_state: int, action: int) -> SanityObservationModel:
        return SanityObservationModel(next_state, action)
    
    def reward(self, state: int, action: int) -> float:
        # Higher reward for being in good state (0)
        return 1.0 if state == 0 else 0.0
    
    def is_terminal(self, state: int) -> bool:
        return False  # No terminal states
    
    def initial_state_dist(self) -> SanityInitialStateDist:
        return SanityInitialStateDist()
    
    def initial_observation_dist(self) -> SanityInitialObservationDist:
        return SanityInitialObservationDist()
    
    def get_actions(self) -> List[int]:
        return [0, 1]  # Two actions: 0 (good) and 1 (bad)
    
    def is_equal_observation(self, observation1: int, observation2: int) -> bool:
        return observation1 == observation2
