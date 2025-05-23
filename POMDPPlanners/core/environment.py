from typing import Any, List, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.core.simulation import MetricValue
from POMDPPlanners.utils.config_to_id import config_to_id

class SpaceType(Enum):
    """Enum representing different types of spaces in the environment."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MIXED = "mixed"

@dataclass
class SpaceInfo:
    """Class containing information about the environment's spaces."""
    action_space: SpaceType
    observation_space: SpaceType

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
    def __init__(self, discount_factor: float, name: str, space_info: SpaceInfo):
        self.discount_factor = discount_factor
        self.name = name
        self.space_info = space_info

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

    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on environment configuration."""
        def serialize_value(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (str, int, float, bool)):
                return value
            elif isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            elif isinstance(value, dict):
                return {str(k): serialize_value(v) for k, v in sorted(value.items())}
            elif isinstance(value, SpaceInfo):
                return {
                    'action_space': serialize_value(value.action_space),
                    'observation_space': serialize_value(value.observation_space)
                }
            elif isinstance(value, Enum):
                return value.value
            elif hasattr(value, '__dict__'):
                return serialize_value(value.__dict__)
            else:
                return str(value)
        config_dict = {}
        for key, value in self.__dict__.items():
            if key.startswith('_') or callable(value):
                continue
            config_dict[key] = serialize_value(value)
        config_dict = dict(sorted(config_dict.items()))
        return config_to_id(config_dict)

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
    
    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        return []


class DiscreteActionsEnvironment(Environment):
    def __init__(self, discount_factor: float, name: str, space_info: SpaceInfo):
        super().__init__(discount_factor=discount_factor, name=name, space_info=space_info)

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


class EnvironmentGenerator(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_environment(self) -> Environment:
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
