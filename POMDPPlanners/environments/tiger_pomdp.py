from typing import List, Any
import numpy as np
from POMDPPlanners.core.environment import DiscreteActionsEnvironment, ObservationModel, StateTransitionModel
from POMDPPlanners.core.distributions import Distribution, DiscreteDistribution

STATES = ['tiger_left', 'tiger_right']
ACTIONS = ['listen', 'open_left', 'open_right']
OBSERVATIONS = ['hear_left', 'hear_right', 'hear_nothing']


class TigerTransition(StateTransitionModel):
    def __init__(self, state: str, action: str):
        self.state = state
        self.action = action
        
    def sample(self) -> str:
        # State only changes when opening a door
        if self.action == 'open_left' or self.action == 'open_right':
            # After opening a door, tiger is randomly placed behind either door
            return np.random.choice(STATES)
        return self.state
    
    def probability(self, next_state: Any) -> float:
        if self.action == 'open_left' or self.action == 'open_right':
            return 0.5
        return 1.0
                
class TigerObservation(ObservationModel):
    def __init__(self, next_state: str, action: str):
        self.next_state = next_state
        self.action = action
        
    def sample(self) -> str:
        if self.action == 'listen':
            # Listen action is 85% accurate
            if self.next_state == 'tiger_left':
                return 'hear_left' if np.random.random() < 0.85 else 'hear_right'
            else:
                return 'hear_right' if np.random.random() < 0.85 else 'hear_left'
        else:
            # When opening a door, observation is random (hearing nothing)
            return 'hear_nothing'
        
    def probability(self, next_observation: Any) -> float:
        if self.action == 'listen':
            if self.next_state == 'tiger_left':
                return 0.85 if next_observation == 'hear_left' else 0.15
            else:
                return 0.85 if next_observation == 'hear_right' else 0.15
            
        if next_observation == 'hear_nothing':
            return 1.0
        else:
            raise ValueError(f"Invalid observation: {next_observation}")


class TigerPOMDP(DiscreteActionsEnvironment):
    def __init__(self, discount_factor: float):
        super().__init__(discount_factor)
        self.states = STATES
        self.actions = ACTIONS
        self.observations = OBSERVATIONS
        
    def state_transition_model(self, state: str, action: str) -> Distribution:
        return TigerTransition(state=state, action=action)

    def observation_model(self, state: str, action: str) -> Distribution:                    
        return TigerObservation(next_state=state, action=action)

    def reward(self, state: str, action: str) -> float:
        if action == 'listen':
            return -1.0  # Cost of listening
        elif action == 'open_left':
            if state == 'tiger_left':
                return -100.0  # Opening door with tiger
            else:
                return 10.0  # Opening door with treasure
        else:  # open_right
            if state == 'tiger_right':
                return -100.0  # Opening door with tiger
            else:
                return 10.0  # Opening door with treasure

    def is_terminal(self, state: str) -> bool:
        # Game ends when a door is opened
        return False  # Since we handle terminal states in the transition model

    def initial_state_dist(self) -> Distribution:
        return DiscreteDistribution(
            values=STATES,
            probs=np.ones(len(STATES)) / len(STATES)
        )

    def initial_observation_dist(self) -> Distribution:
        return DiscreteDistribution(
            values=['hear_nothing'],
            probs=[1.0]
        )

    def get_actions(self) -> List[Any]:
        return self.actions
