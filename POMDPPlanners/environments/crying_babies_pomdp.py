from typing import List, Any
import numpy as np

from POMDPPlanners.core.environment import DiscreteActionsEnvironment
from POMDPPlanners.core.distributions import Distribution

class BabiesTransition(Distribution):
    def __init__(self, state: tuple, action: str, num_babies: int):
        self.state = state
        self.action = action
        self.num_babies = num_babies
        
    def sample(self) -> tuple:
        new_state = list(self.state)
        
        # Babies get hungry with probability 0.1 each time step
        for i in range(self.num_babies):
            if np.random.random() < 0.1:
                new_state[i] = True
        
        # Feeding a baby makes it not hungry
        if self.action == 'feed_baby1':
            new_state[0] = False
        elif self.action == 'feed_baby2':
            new_state[1] = False
            
        return tuple(new_state)
    
    def probability(self, value):
        # Simplified probability calculation
        # In reality, this would need to account for all possible transitions
        return 1.0

class BabiesObservation(Distribution):
    def __init__(self, state: tuple, action: str, observations: List[str]):
        self.state = state
        self.action = action
        self.observations = observations
        
    def sample(self) -> str:
        # Hungry babies cry with probability 0.8
        cries = []
        for i, hungry in enumerate(self.state):
            if hungry and np.random.random() < 0.8:
                cries.append(i)
        
        if len(cries) == 0:
            return 'cry_none'
        elif len(cries) == 1:
            return 'cry_baby1' if cries[0] == 0 else 'cry_baby2'
        else:
            return 'cry_both'
        
    def probability(self, value):
        # Simplified probability calculation
        # In reality, this would need to account for all possible observations
        return 1.0

class CryingBabiesPOMDP(DiscreteActionsEnvironment):
    def __init__(self, discount_factor: float, num_babies: int = 2):
        super().__init__(discount_factor)
        self.num_babies = num_babies
        # States are represented as tuples of booleans, where True means baby is hungry
        self.states = [(b1, b2) for b1 in [True, False] for b2 in [True, False]]
        self.actions = ['feed_baby1', 'feed_baby2', 'do_nothing']
        self.observations = ['cry_baby1', 'cry_baby2', 'cry_both', 'cry_none']
        
    def state_transition_model(self, state: tuple, action: str) -> Distribution:
        return BabiesTransition(state, action, self.num_babies)

    def observation_model(self, state: tuple, action: str) -> Distribution:
        return BabiesObservation(state, action, self.observations)

    def reward(self, state: tuple, action: str) -> float:
        reward = 0.0
        
        # Calculate the state after the action
        new_state = list(state)
        if action == 'feed_baby1':
            new_state[0] = False
        elif action == 'feed_baby2':
            new_state[1] = False
            
        # Penalty for each hungry baby in the new state
        for hungry in new_state:
            if hungry:
                reward -= 5.0
                
        # Cost of feeding
        if action != 'do_nothing':
            reward -= 1.0
            
        return reward

    def is_terminal(self, state: tuple) -> bool:
        return False  # Game continues indefinitely

    def initial_state_dist(self) -> Distribution:
        class InitialState(Distribution):
            def __init__(self, states: List[tuple]):
                self.states = states
                
            def sample(self) -> tuple:
                # Start with both babies not hungry
                return (False, False)
                
        return InitialState(self.states)

    def initial_observation_dist(self) -> Distribution:
        class InitialObservation(Distribution):
            def sample(self) -> str:
                return 'cry_none'
                
        return InitialObservation()

    def get_actions(self) -> List[Any]:
        return self.actions
