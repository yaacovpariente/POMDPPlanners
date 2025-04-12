from typing import Any, List, Tuple
from abc import ABC, abstractmethod

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.simulation import StepData

class ObservationModel(Distribution, ABC):
    def __init__(self, next_state, action):
        self.next_state = next_state
        self.action = action
        
    @abstractmethod
    def sample(self) -> Any:
        pass
    
    def probability(self, next_observation: Any) -> float:
        pass

class StateTransitionModel(Distribution, ABC):
    def __init__(self, state, action):
        self.state = state
        self.action = action

    @abstractmethod
    def sample(self) -> Any:
        pass

    def probability(self, next_state: Any) -> float:
        pass

class Environment(ABC):
    def __init__(self, discount_factor: float):
        self.discount_factor = discount_factor
        
    @abstractmethod
    def state_transition_model(self, state: Any, action: Any) -> StateTransitionModel:
        pass

    @abstractmethod
    def observation_model(self, state: Any, action: Any) -> ObservationModel:
        pass

    @abstractmethod
    def reward(self, state: Any, action: Any) -> float:
        pass

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        pass

    @abstractmethod
    def initial_state_dist(self) -> Distribution:
        pass
    
    @abstractmethod
    def initial_observation_dist(self) -> Distribution:
        pass
    
    def sample_next_step(self, state: Any, action: Any) -> Tuple[Any, Any, float]:
        next_state = self.state_transition_model(state=state, action=action).sample()
        next_observation = self.observation_model(state=next_state, action=action).sample()
        reward = self.reward(state=state, action=action)
        
        return next_state, next_observation, reward
    
    def get_history_artifacts(self, history: List[StepData]) -> None:
        pass

class DiscreteActionsEnvironment(Environment):
    def __init__(self, discount_factor: float):
        super().__init__(discount_factor)
        
    @abstractmethod
    def state_transition_model(self, state: Any, action: Any) -> StateTransitionModel:
        pass

    @abstractmethod
    def observation_model(self, state: Any, action: Any) -> ObservationModel:
        pass

    @abstractmethod
    def reward(self, state: Any, action: Any) -> float:
        pass

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        pass

    @abstractmethod
    def initial_state_dist(self) -> Distribution:
        pass
    
    @abstractmethod
    def initial_observation_dist(self) -> Distribution:
        pass
    
    @abstractmethod
    def get_actions(self) -> List[Any]:
        pass
    