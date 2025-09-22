"""Module for POMDP environment abstractions.

This module provides the foundational classes and interfaces for defining
POMDP environments, including abstract base classes for state transitions,
observation models, and reward functions.

Classes:
    Environment: Abstract base class for POMDP environments
    DiscreteActionsEnvironment: Specialized for discrete action spaces
    ObservationModel: Abstract observation model interface
    StateTransitionModel: Abstract state transition interface
    EnvironmentGenerator: Factory pattern for environment creation
    SpaceType: Enumeration for action/observation space types
    SpaceInfo: Data class containing space type information
"""

from typing import Any, List, Tuple, Optional
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
import logging

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.core.simulation import MetricValue
from POMDPPlanners.utils.config_to_id import config_to_id
from POMDPPlanners.utils.logger import get_logger


class SpaceType(Enum):
    """Enumeration for categorizing action and observation spaces.

    This enum is used to classify the mathematical structure of action
    and observation spaces in POMDP environments.

    Attributes:
        DISCRETE: Finite, countable spaces (e.g., {0, 1, 2, ...})
        CONTINUOUS: Real-valued continuous spaces (e.g., R^n)
        MIXED: Combination of discrete and continuous elements
    """

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MIXED = "mixed"


@dataclass
class SpaceInfo:
    """Data class containing space type information for an environment.

    This class encapsulates the space type classifications for both
    actions and observations in a POMDP environment.

    Attributes:
        action_space: The type of action space (discrete, continuous, or mixed)
        observation_space: The type of observation space (discrete, continuous, or mixed)

    Example:
        Creating space info for different environment types::

            # Discrete actions, continuous observations
            space_info = SpaceInfo(
                action_space=SpaceType.DISCRETE,
                observation_space=SpaceType.CONTINUOUS
            )

            # Both discrete
            discrete_space = SpaceInfo(
                action_space=SpaceType.DISCRETE,
                observation_space=SpaceType.DISCRETE
            )

            # Mixed space types
            mixed_space = SpaceInfo(
                action_space=SpaceType.MIXED,
                observation_space=SpaceType.CONTINUOUS
            )
    """

    action_space: SpaceType
    observation_space: SpaceType


class ObservationModel(Distribution, ABC):
    """Abstract base class for POMDP observation models.

    This class defines the interface for observation models that generate
    observations given a next state and action. Inherits from Distribution
    to provide sampling and probability calculation capabilities.

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses must implement the sample() method.

    Attributes:
        next_state: The state after taking an action
        action: The action that was taken
    """

    def __init__(self, next_state: Any, action: Any):
        """Initialize the observation model.

        Args:
            next_state: The resulting state after taking an action
            action: The action that was executed
        """
        self.next_state = next_state
        self.action = action

    @abstractmethod
    def sample(self, n_samples: int = 1) -> List[Any]:
        """Sample observations from the observation model.

        Args:
            n_samples: Number of observation samples to generate. Defaults to 1.

        Returns:
            List of sampled observations of length n_samples.

        Note:
            Subclasses must implement this method according to their
            specific observation generation logic.
        """
        pass

    def probability(self, values: List[Any]) -> np.ndarray:
        """Calculate observation probabilities for given values.

        Args:
            values: List of observation values to calculate probabilities for

        Returns:
            Array of probabilities corresponding to the input values

        Raises:
            NotImplementedError: This method is not implemented by default.
                Subclasses should override if probability calculation is needed.
        """
        raise NotImplementedError(
            "The method is not implemented for this observation model."
        )


class StateTransitionModel(Distribution, ABC):
    """Abstract base class for POMDP state transition models.

    This class defines the interface for state transition models that generate
    next states given a current state and action. Inherits from Distribution
    to provide sampling and probability calculation capabilities.

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses must implement the sample() method.

    Attributes:
        state: The current state
        action: The action to be taken
    """

    def __init__(self, state: Any, action: Any):
        """Initialize the state transition model.

        Args:
            state: The current state
            action: The action to be executed from the current state
        """
        self.state = state
        self.action = action

    @abstractmethod
    def sample(self, n_samples: int = 1) -> List[Any]:
        """Sample next states from the transition model.

        Args:
            n_samples: Number of next state samples to generate. Defaults to 1.

        Returns:
            List of sampled next states of length n_samples.

        Note:
            Subclasses must implement this method according to their
            specific state transition dynamics.
        """
        pass

    def probability(self, values: List[Any]) -> np.ndarray:
        """Calculate transition probabilities for given next states.

        Args:
            values: List of next state values to calculate probabilities for

        Returns:
            Array of transition probabilities corresponding to the input values

        Raises:
            NotImplementedError: This method is not implemented by default.
                Subclasses should override if probability calculation is needed.
        """
        raise NotImplementedError(
            "The method is not implemented for this state transition model."
        )


class Environment(ABC):
    """Abstract base class for POMDP environments.

    This is the core abstract class that all POMDP environments must inherit from.
    It defines the essential interface for POMDP environments including state
    transitions, observations, rewards, and terminal conditions.

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses must implement all abstract methods.

    Attributes:
        discount_factor: Discount factor for future rewards
        name: Environment identifier string
        space_info: Information about action and observation space types
        reward_range: Optional tuple containing (min_reward, max_reward)
        output_dir: Optional directory for logging output
        debug: Flag to enable debug logging
    """

    def __init__(
        self,
        discount_factor: float,
        name: str,
        space_info: SpaceInfo,
        reward_range: Optional[Tuple[float, float]] = None,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the POMDP environment.

        Args:
            discount_factor: Discount factor for future rewards (0 < discount_factor <= 1)
            name: Unique identifier for the environment
            space_info: Information about action and observation space types
            reward_range: Optional tuple containing (min_reward, max_reward) for the environment.
                Defaults to None. If provided, will be validated.
            output_dir: Optional directory for logging output. Defaults to None.
            debug: Enable debug logging. Defaults to False.
            use_queue_logger: Whether to use queue-based logging. Defaults to True.
        """
        self.discount_factor = discount_factor
        self.name = name
        self.space_info = space_info
        self.reward_range = self._validate_reward_range(reward_range)
        self.output_dir = output_dir
        self.debug = debug
        self.use_queue_logger = use_queue_logger

        self.logger.info(
            f"Initializing {self.name} environment with discount factor {self.discount_factor}"
        )
        self.logger.debug(
            f"Space info: action_space={self.space_info.action_space}, observation_space={self.space_info.observation_space}"
        )
        if self.reward_range is not None:
            self.logger.debug(f"Reward range: {self.reward_range}")

    def _validate_reward_range(
        self, reward_range: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """Validate reward_range if provided.

        Args:
            reward_range: Optional tuple containing (min_reward, max_reward)

        Returns:
            Validated reward_range tuple or None if input was None

        Raises:
            ValueError: If reward_range structure or values are invalid
            TypeError: If reward_range values are not numeric
        """
        if reward_range is None:
            return None

        # Validate structure
        if not isinstance(reward_range, tuple) or len(reward_range) != 2:
            raise ValueError("reward_range must be a tuple of exactly two float values")

        min_reward, max_reward = reward_range

        # Check that both values are numeric (float or int)
        if not isinstance(min_reward, (int, float)) or not isinstance(
            max_reward, (int, float)
        ):
            raise TypeError("reward_range values must be numeric (int or float)")

        # Convert to float to ensure consistency
        min_reward, max_reward = float(min_reward), float(max_reward)

        # Check for NaN values
        if np.isnan(min_reward) or np.isnan(max_reward):
            raise ValueError("reward_range values cannot be NaN")

        # Check that min_reward <= max_reward (allowing inf values)
        if min_reward > max_reward:
            raise ValueError(
                f"reward_range minimum ({min_reward}) must be less than or equal to maximum ({max_reward})"
            )

        return (min_reward, max_reward)

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this environment.

        The logger is implemented as a property to maintain pickle compatibility,
        as logger objects cannot be pickled directly.

        Returns:
            Configured logger instance with hierarchical naming
        """
        return get_logger(
            name=f"environment.{self.name}",
            output_dir=self.output_dir,
            debug=self.debug,
            use_queue=self.use_queue_logger,
        )

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
            if key.startswith("_") or callable(value):
                continue
            if not hasattr(other, key):
                return False
            other_value = getattr(other, key)
            if not _compare_values(value, other_value):
                return False

        # Check for any attributes in other that aren't in self
        for key in other.__dict__:
            if key.startswith("_") or callable(getattr(other, key)):
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
                    "action_space": serialize_value(value.action_space),
                    "observation_space": serialize_value(value.observation_space),
                }
            elif isinstance(value, Enum):
                return value.value
            elif hasattr(value, "__dict__"):
                # Skip logger objects
                if isinstance(value, logging.Logger):
                    return None
                return serialize_value(value.__dict__)
            else:
                return str(value)

        config_dict = {}
        for key, value in self.__dict__.items():
            # Skip logger and private attributes
            if (
                key.startswith("_")
                or callable(value)
                or isinstance(value, logging.Logger)
            ):
                continue
            serialized_value = serialize_value(value)
            if serialized_value is not None:  # Skip None values (like logger)
                config_dict[key] = serialized_value
        config_dict = dict(sorted(config_dict.items()))
        return config_to_id(config_dict)

    def __hash__(self) -> int:
        return hash(self.config_id)

    @abstractmethod
    def state_transition_model(self, state: Any, action: Any) -> StateTransitionModel:
        """Get the state transition model for a given state-action pair.

        Args:
            state: Current state
            action: Action to be executed

        Returns:
            State transition model that can sample next states

        Note:
            Subclasses must implement this method to define state dynamics.
        """
        pass

    @abstractmethod
    def observation_model(self, next_state: Any, action: Any) -> ObservationModel:
        """Get the observation model for a given next state and action.

        Args:
            next_state: The resulting state after taking an action
            action: The action that was executed

        Returns:
            Observation model that can sample observations

        Note:
            Subclasses must implement this method to define observation generation.
        """
        pass

    @abstractmethod
    def reward(self, state: Any, action: Any) -> float:
        """Calculate the immediate reward for a state-action pair.

        Args:
            state: Current state
            action: Action executed from the state

        Returns:
            Immediate reward value

        Note:
            Subclasses must implement this method to define reward structure.
        """
        pass

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        """Check if a state is terminal.

        Args:
            state: State to check for terminal condition

        Returns:
            True if the state is terminal, False otherwise

        Note:
            Subclasses must implement this method to define terminal conditions.
        """
        pass

    @abstractmethod
    def initial_state_dist(self) -> Distribution:
        """Get the initial state distribution.

        Returns:
            Distribution over initial states

        Note:
            Subclasses must implement this method to define the starting distribution.
        """
        pass

    @abstractmethod
    def initial_observation_dist(self) -> Distribution:
        """Get the initial observation distribution.

        Returns:
            Distribution over initial observations

        Note:
            Subclasses must implement this method to define initial observations.
        """
        pass

    @abstractmethod
    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        """Check if two observations are equal.

        Args:
            observation1: First observation to compare
            observation2: Second observation to compare

        Returns:
            True if observations are considered equal, False otherwise

        Note:
            Subclasses must implement this method to define observation equality.
            This is particularly important for discrete observation spaces.
        """
        pass

    def sample_next_step(self, state: Any, action: Any) -> Tuple[Any, Any, float]:
        """Sample a complete state transition step.

        This convenience method combines state transition, observation generation,
        and reward calculation in a single operation.

        Args:
            state: Current state
            action: Action to execute

        Returns:
            Tuple containing:
                - next_state: Sampled next state
                - next_observation: Sampled observation
                - reward: Immediate reward
        """
        next_state = self.state_transition_model(state=state, action=action).sample()[0]
        next_observation = self.observation_model(
            next_state=next_state, action=action
        ).sample()[0]
        reward = self.reward(state=state, action=action)

        return next_state, next_observation, reward

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Cache visualization data for an episode history.

        This method can be overridden by subclasses to provide environment-specific
        visualization caching capabilities.

        Args:
            history: List of step data from an episode
            cache_path: Path where visualization data should be cached
        """
        pass

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        """Compute environment-specific metrics from episode histories.

        This method can be overridden by subclasses to provide custom
        metric calculations beyond standard return and episode length.

        Args:
            histories: List of episode histories to analyze

        Returns:
            List of computed metrics with confidence intervals
        """
        return []


class DiscreteActionsEnvironment(Environment):
    """Abstract base class for POMDP environments with discrete action spaces.

    This class extends the base Environment class with additional functionality
    specific to environments that have finite, enumerable action sets.

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses must implement all abstract methods from Environment plus
        the get_actions() method.
    """

    def __init__(
        self,
        discount_factor: float,
        name: str,
        space_info: SpaceInfo,
        reward_range: Optional[Tuple[float, float]] = None,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the discrete actions environment.

        Args:
            discount_factor: Discount factor for future rewards (0 < discount_factor <= 1)
            name: Unique identifier for the environment
            space_info: Information about action and observation space types
            reward_range: Optional tuple containing (min_reward, max_reward) for the environment.
                Defaults to None. If provided, will be validated.
            output_dir: Optional directory for logging output. Defaults to None.
            debug: Enable debug logging. Defaults to False.
        """
        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=reward_range,
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )
        self.logger.debug("Initialized DiscreteActionsEnvironment")

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
        """Get all possible actions in the discrete action space.

        Returns:
            List containing all valid actions that can be executed

        Note:
            Subclasses must implement this method to enumerate all possible actions.
            This is used by planning algorithms that need to iterate over actions.
        """
        pass

    @abstractmethod
    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        pass


class EnvironmentGenerator(ABC):
    """Abstract base class for environment generators.

    This class implements the factory pattern for creating environment instances.
    It's useful for generating environments with randomized parameters or
    for creating multiple environment variants.

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses must implement the generate_environment() method.

    Attributes:
        name: Identifier for the generator
    """

    def __init__(self, name: str):
        """Initialize the environment generator.

        Args:
            name: Unique identifier for this generator
        """
        self.name = name

    @abstractmethod
    def generate_environment(self) -> Environment:
        """Generate a new environment instance.

        Returns:
            Newly created environment instance

        Note:
            Subclasses must implement this method to define environment creation logic.
            This may involve randomization, parameter sampling, or deterministic generation.
        """
        pass
