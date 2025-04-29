from typing import Any, List, Tuple
from pathlib import Path
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
    def __init__(self, discount_factor: float, name: str):
        self.discount_factor = discount_factor
        self.name = name

    def __eq__(self, other):
        if not isinstance(other, Environment):
            return False
        if self.__class__ != other.__class__:
            return False

        def _compare_values(v1, v2):
            """Helper function to compare values, handling numpy arrays specially."""
            if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
                if not (isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray)):
                    return False
                return np.array_equal(v1, v2)
            elif isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
                if len(v1) != len(v2):
                    return False
                return all(_compare_values(x1, x2) for x1, x2 in zip(v1, v2))
            elif isinstance(v1, dict) and isinstance(v2, dict):
                if v1.keys() != v2.keys():
                    return False
                return all(_compare_values(v1[k], v2[k]) for k in v1)
            else:
                return v1 == v2

        # Compare all public attributes (excluding callables and private)
        for key, value in self.__dict__.items():
            if key.startswith('_') or callable(value):
                continue
            if not hasattr(other, key):
                return False
            other_value = getattr(other, key)
            if not _compare_values(value, other_value):
                return False

        # Check for any attributes in other that aren't in self
        for key in other.__dict__:
            if key.startswith('_') or callable(getattr(other, key)):
                continue
            if not hasattr(self, key):
                return False

        return True

    @abstractmethod
    def state_transition_model(self, state: Any, action: Any) -> StateTransitionModel:
        pass

    @abstractmethod
    def observation_model(self, next_state: Any, action: Any) -> ObservationModel:
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
    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        pass

    def sample_next_step(self, state: Any, action: Any) -> Tuple[Any, Any, float]:
        next_state = self.state_transition_model(state=state, action=action).sample()
        next_observation = self.observation_model(
            next_state=next_state, action=action
        ).sample()
        reward = self.reward(state=state, action=action)

        return next_state, next_observation, reward

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        pass


class DiscreteActionsEnvironment(Environment):
    def __init__(self, discount_factor: float, name: str):
        super().__init__(discount_factor=discount_factor, name=name)

    @abstractmethod
    def state_transition_model(self, state: Any, action: Any) -> StateTransitionModel:
        pass

    @abstractmethod
    def observation_model(self, next_state: Any, action: Any) -> ObservationModel:
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

    @abstractmethod
    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        pass

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        pass


# class GymEnvStateTransition(StateTransitionModel):
#     def __init__(self, env: gym.Env, state: Any, action: Any):
#         super().__init__(state, action)
#         self.env = env

#     def sample(self) -> Any:
#         next_state, reward, done, info = self.env.step(self.action)
#         return next_state

# class GymEnvNormalNoiseObservation(ObservationModel):
#     def __init__(self, env: gym.Env, state: Any, action: Any, cov_matrix: np.ndarray):
#         super().__init__(state, action)
#         self.env = env
#         self.cov_matrix = cov_matrix
#         self.noise_dist = np.random.multivariate_normal(mean=self.state, cov=self.cov_matrix)

#     def sample(self) -> Any:
#         return self.noise_dist.sample()

#     def probability(self, next_observation: Any) -> float:
#         return self.noise_dist.pdf(next_observation)
# class GymEnvironmentWrapper(Environment):
#     def __init__(self, env: gym.Env, cov_matrix: np.ndarray):
#         super().__init__()
#         self.env = env
#         self.env.reset()

#         self.cov_matrix = cov_matrix

#     def state_transition_model(self, state: Any, action: Any) -> StateTransitionModel:
#         return GymEnvStateTransition(self.env, state, action)

#     def observation_model(self, state: Any, action: Any) -> ObservationModel:
#         return GymEnvNormalNoiseObservation(self.env, state, action, self.cov_matrix)

#     def reward(self, state: Any, action: Any) -> float:
#         next_state, reward, done, info = self.env.step(action)
#         return reward

#     def is_terminal(self, state: Any) -> bool:
#         next_state, reward, done, info = self.env.step(action)

#     def initial_state_dist(self) -> Distribution:
#         pass

#     def initial_observation_dist(self) -> Distribution:
#         pass

#     def get_actions(self) -> List[Any]:
#         pass
