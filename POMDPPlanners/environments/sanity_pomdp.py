from typing import Any, List, Optional
from pathlib import Path
import numpy as np

from POMDPPlanners.core.environment import DiscreteActionsEnvironment, SpaceInfo, SpaceType
from POMDPPlanners.core.distributions import Distribution

class SanityStateTransitionModel(Distribution):
    def __init__(self, state: int, action: int):
        super().__init__()
        self.state = state
        self.action = action
        
    def sample(self, n_samples: int = 1) -> List[int]:
        # Action 0 always leads to state 0 (good state)
        # Action 1 always leads to state 1 (bad state)
        next_state = 0 if self.action == 0 else 1
        return [next_state] * n_samples

    def probability(self, values: List[int]) -> np.ndarray:
        result = np.zeros(len(values))
        expected_next_state = 0 if self.action == 0 else 1
        for i, value in enumerate(values):
            if value == expected_next_state:
                result[i] = 1.0
        return result

class SanityObservationModel(Distribution):
    def __init__(self, next_state: int, action: int):
        super().__init__()
        self.next_state = next_state
        self.action = action
        
    def sample(self, n_samples: int = 1) -> List[int]:
        # Observation always matches the state
        return [self.next_state] * n_samples

    def probability(self, values: List[int]) -> np.ndarray:
        result = np.zeros(len(values))
        for i, value in enumerate(values):
            if value == self.next_state:
                result[i] = 1.0
        return result

class SanityInitialStateDist(Distribution):
    def sample(self, n_samples: int = 1) -> List[int]:
        # Always start in good state (0)
        return [0] * n_samples

    def probability(self, values: List[int]) -> np.ndarray:
        result = np.zeros(len(values))
        for i, value in enumerate(values):
            if value == 0:
                result[i] = 1.0
        return result

class SanityInitialObservationDist(Distribution):
    def sample(self, n_samples: int = 1) -> List[int]:
        # Initial observation always matches initial state (0)
        return [0] * n_samples

    def probability(self, values: List[int]) -> np.ndarray:
        result = np.zeros(len(values))
        for i, value in enumerate(values):
            if value == 0:
                result[i] = 1.0
        return result

class SanityPOMDP(DiscreteActionsEnvironment):
    def __init__(self, discount_factor: float = 0.95, output_dir: Optional[Path] = None, debug: bool = False):
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,  # Binary action space
            observation_space=SpaceType.DISCRETE  # Binary observation space
        )
        super().__init__(discount_factor=discount_factor, name="SanityPOMDP", space_info=space_info, output_dir=output_dir, debug=debug)
        
    def state_transition_model(self, state: int, action: int) -> SanityStateTransitionModel:
        return SanityStateTransitionModel(state, action)
    
    def observation_model(self, next_state: int, action: int) -> SanityObservationModel:
        return SanityObservationModel(next_state, action)
    
    def reward(self, state: int, action: int) -> float:
        # Higher reward for being in good state (0)
        next_state = self.state_transition_model(state, action).sample()[0]
        return 1.0 if next_state == 0 else 0.0
    
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
