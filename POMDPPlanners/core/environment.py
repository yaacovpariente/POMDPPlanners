from typing import Any, List, Tuple
import gymnasium as gym
from abc import ABC, abstractmethod

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.simulation import StepData

class ObservationModel(Distribution, ABC):
    def __init__(self, next_state, action):
        self.next_state = next_state
        self.action = action
        
    @abstractmethod
    def sample(self) -> Any:
        pass
    
    @abstractmethod
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
        next_state, reward, done, info = self.env.step(action)
        return 

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
    
    
class GymEnvStateTransition(StateTransitionModel):
    def __init__(self, env: gym.Env, state: Any, action: Any):
        super().__init__(state, action)
        self.env = env
        
    def sample(self) -> Any:
        next_state, reward, done, info = self.env.step(self.action)
        return next_state

class GymEnvNormalNoiseObservation(ObservationModel):
    def __init__(self, env: gym.Env, state: Any, action: Any, cov_matrix: np.ndarray):
        super().__init__(state, action)
        self.env = env
        self.cov_matrix = cov_matrix
        self.noise_dist = np.random.multivariate_normal(mean=self.state, cov=self.cov_matrix)

    def sample(self) -> Any:
        return self.noise_dist.sample()
    
    def probability(self, next_observation: Any) -> float:
        return self.noise_dist.pdf(next_observation)
class GymEnvironmentWrapper(Environment):
    def __init__(self, env: gym.Env, cov_matrix: np.ndarray):
        super().__init__(discount_factor=env.spec.timestep_limit / env.spec.max_episode_steps)
        self.env = env
        self.env.reset()
        
        self.cov_matrix = cov_matrix
        
    def state_transition_model(self, state: Any, action: Any) -> StateTransitionModel:
        return GymEnvStateTransition(self.env, state, action)
    
    def observation_model(self, state: Any, action: Any) -> ObservationModel:
        return GymEnvNormalNoiseObservation(self.env, state, action, self.cov_matrix)
    
    def reward(self, state: Any, action: Any) -> float:
        next_state, reward, done, info = self.env.step(action)
        return reward
    
    def is_terminal(self, state: Any) -> bool:
        next_state, reward, done, info = self.env.step(action)
    
    def initial_state_dist(self) -> Distribution:
        pass
    
    def initial_observation_dist(self) -> Distribution:
        pass
    
    def get_actions(self) -> List[Any]:
        pass

