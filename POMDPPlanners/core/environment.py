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

import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.serialization import (
    deserialize_value as deserialize_value_base,
    register_deserializer,
    register_serializer,
    serialize_value as serialize_value_base,
)
from POMDPPlanners.utils.config_to_id import config_to_id
from POMDPPlanners.utils.logger import get_logger

if TYPE_CHECKING:
    from POMDPPlanners.core.simulation import History, MetricValue, StepData


def _serialize_space_info(space_info: Any) -> dict:
    """Serialize SpaceInfo to plain dict without type markers.

    Maintains backward compatibility with existing saved environments.
    Format: {"action_space": "discrete", "observation_space": "continuous"}

    Args:
        space_info: SpaceInfo instance to serialize

    Returns:
        Plain dict with action_space and observation_space string values
    """
    return {
        "action_space": space_info.action_space.value,
        "observation_space": space_info.observation_space.value,
    }


def _deserialize_space_info(data: dict) -> Any:
    """Deserialize SpaceInfo from plain dict format.

    Handles dicts with action_space and observation_space keys without
    requiring __type__ markers for backward compatibility.

    Args:
        data: Dict with action_space and observation_space keys

    Returns:
        SpaceInfo instance

    Raises:
        ValueError: If data cannot be deserialized to SpaceInfo
    """
    if isinstance(data, dict) and "action_space" in data and "observation_space" in data:
        # Import SpaceType here to avoid circular dependency
        return SpaceInfo(
            action_space=SpaceType(data["action_space"]),
            observation_space=SpaceType(data["observation_space"]),
        )
    raise ValueError(f"Cannot deserialize SpaceInfo from {data}")


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
        Creating space info for different environment types:

        >>> # Discrete actions, continuous observations
        >>> space_info = SpaceInfo(
        ...     action_space=SpaceType.DISCRETE,
        ...     observation_space=SpaceType.CONTINUOUS
        ... )
    """

    action_space: SpaceType
    observation_space: SpaceType


# Register SpaceInfo serialization handlers at module load time
# This enables centralized serialization system to handle SpaceInfo automatically
register_serializer(SpaceInfo, _serialize_space_info)
register_deserializer(SpaceInfo, _deserialize_space_info)


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
        raise NotImplementedError("The method is not implemented for this observation model.")


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
        raise NotImplementedError("The method is not implemented for this state transition model.")


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
            "Initializing %s environment with discount factor %s", self.name, self.discount_factor
        )
        self.logger.debug(
            "Space info: action_space=%s, observation_space=%s",
            self.space_info.action_space,
            self.space_info.observation_space,
        )
        if self.reward_range is not None:
            self.logger.debug("Reward range: %s", self.reward_range)

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
        if not isinstance(min_reward, (int, float)) or not isinstance(max_reward, (int, float)):
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

        def _compare_values(v1, v2):  # pylint: disable=too-many-return-statements
            """Helper function to compare values, handling numpy arrays specially."""
            if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
                if not (isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray)):
                    return False
                return np.array_equal(v1, v2)
            if isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
                if len(v1) != len(v2):
                    return False
                return all(_compare_values(x1, x2) for x1, x2 in zip(v1, v2))
            if isinstance(v1, dict) and isinstance(v2, dict):
                if v1.keys() != v2.keys():
                    return False
                return all(_compare_values(v1[k], v2[k]) for k in v1)
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
        """Generate a deterministic identifier based on environment configuration.

        Note:
            Uses custom serialization logic (not centralized serialize_value) to ensure:
            - Deterministic dict key ordering for consistent hashing
            - Compact format without __type__ markers
            - Recursive handling of nested objects
            Changing this serialization format would invalidate all cached results.
        """

        def serialize_value(value):  # pylint: disable=too-many-return-statements
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            if isinstance(value, dict):
                return {str(k): serialize_value(v) for k, v in sorted(value.items())}
            if isinstance(value, SpaceInfo):
                return {
                    "action_space": serialize_value(value.action_space),
                    "observation_space": serialize_value(value.observation_space),
                }
            if isinstance(value, Enum):
                return value.value
            if hasattr(value, "__dict__"):
                # Skip logger objects
                if isinstance(value, logging.Logger):
                    return None
                return serialize_value(value.__dict__)
            return str(value)

        config_dict = {}
        for key, value in self.__dict__.items():
            # Skip logger and private attributes
            if key.startswith("_") or callable(value) or isinstance(value, logging.Logger):
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

    def reward_batch(self, states: Union[np.ndarray, Sequence[Any]], action: Any) -> np.ndarray:
        """Calculate rewards for a batch of states given a single action.

        Provides a loop-based default that subclasses can override with
        vectorized numpy implementations for better performance.

        Args:
            states: Sequence of states of length ``N``.
            action: Action executed from each state.

        Returns:
            1-D array of reward values with shape ``(N,)``.
        """
        return np.array([self.reward(states[i], action) for i in range(len(states))])

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

    @abstractmethod
    def initial_state_dist(self) -> Distribution:
        """Get the initial state distribution.

        Returns:
            Distribution over initial states

        Note:
            Subclasses must implement this method to define the starting distribution.
        """

    @abstractmethod
    def initial_observation_dist(self) -> Distribution:
        """Get the initial observation distribution.

        Returns:
            Distribution over initial observations

        Note:
            Subclasses must implement this method to define initial observations.
        """

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

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> Any:
        """Sample one or more next states for ``(state, action)``.

        Hot-path entry point used by MCTS planners and by particle filters.
        The default delegates to
        ``state_transition_model(state, action).sample(n_samples)``; subclasses
        may override to skip the per-call wrapper allocation while preserving
        the same RNG draw sequence.

        Args:
            state: Current state.
            action: Action to execute.
            n_samples: Number of samples to draw. Defaults to 1.

        Returns:
            When ``n_samples == 1``: a single next state of the env's
            native type.
            When ``n_samples > 1``: an array-like of length ``n_samples``.
            Numeric envs return ``np.ndarray`` of shape ``(n_samples, *dim)``;
            structured envs (Tiger, Pacman, Sanity) return a
            ``List[T]`` of length ``n_samples``.
        """
        samples = self.state_transition_model(state=state, action=action).sample(n_samples)
        if n_samples == 1:
            return samples[0]
        return samples

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        """Sample one or more observations for ``(next_state, action)``.

        Hot-path entry point used by MCTS planners and by particle filters.
        The default delegates to
        ``observation_model(next_state, action).sample(n_samples)``;
        subclasses may override to skip the per-call wrapper allocation while
        preserving the same RNG draw sequence.

        Args:
            next_state: State after the action was executed.
            action: Action that was executed.
            n_samples: Number of samples to draw. Defaults to 1.

        Returns:
            When ``n_samples == 1``: a single observation of the env's
            native type.
            When ``n_samples > 1``: an array-like of length ``n_samples``.
            Numeric envs return ``np.ndarray`` of shape ``(n_samples, *dim)``;
            structured envs return a ``List[T]`` of length ``n_samples``.
        """
        samples = self.observation_model(next_state=next_state, action=action).sample(n_samples)
        if n_samples == 1:
            return samples[0]
        return samples

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        """Log-probability of each candidate next state under ``(state, action)``.

        The default delegates to
        ``np.log(state_transition_model(state, action).probability(next_states))``;
        subclasses may override for vectorized native paths.

        Args:
            state: Current state.
            action: Action that was executed.
            next_states: A sequence (length N) or batch ndarray (shape ``(N, *dim)``)
                of candidate next states.

        Returns:
            ndarray of shape ``(N,)`` with log-probabilities (or log-PDFs for
            continuous envs).
        """
        probs = np.asarray(
            self.state_transition_model(state=state, action=action).probability(next_states)
        )
        return np.log(probs + 1e-300)

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        """Log-probability of each candidate observation under ``(next_state, action)``.

        The default delegates to
        ``np.log(observation_model(next_state, action).probability(observations))``;
        subclasses may override for vectorized native paths.

        Args:
            next_state: State after the action was executed.
            action: Action that was executed.
            observations: A sequence (length N) or batch ndarray (shape ``(N, *dim)``)
                of candidate observations.

        Returns:
            ndarray of shape ``(N,)`` with log-probabilities (or log-PDFs for
            continuous envs).
        """
        probs = np.asarray(
            self.observation_model(next_state=next_state, action=action).probability(observations)
        )
        return np.log(probs + 1e-300)

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
        next_state = self.sample_next_state(state=state, action=action)
        next_observation = self.sample_observation(next_state=next_state, action=action)
        reward = self.reward(state=state, action=action)

        return next_state, next_observation, reward

    def cache_visualization(self, history: "List[StepData]", cache_path: Path) -> None:
        """Cache visualization data for an episode history.

        This method can be overridden by subclasses to provide environment-specific
        visualization caching capabilities.

        Args:
            history: List of step data from an episode
            cache_path: Path where visualization data should be cached
        """

    def get_metric_names(self) -> List[str]:
        """Get names of environment-specific metrics.

        This method returns the names of custom metrics that this environment
        computes in the compute_metrics() method. It enables users to discover
        what metrics are available for hyperparameter optimization.

        Returns:
            List of metric names that this environment produces.
            Default implementation returns empty list for environments without custom metrics.

        Note:
            Subclasses that override compute_metrics() should also override this method
            to return the names of metrics they produce. Use an Enum to ensure consistency
            between the names returned here and the names used in compute_metrics().
        """
        return []

    def compute_metrics(
        self, histories: "List[History]"
    ) -> "List[MetricValue]":  # pylint: disable=unused-argument
        """Compute environment-specific metrics from episode histories.

        This method can be overridden by subclasses to provide custom
        metric calculations beyond standard return and episode length.

        Args:
            histories: List of episode histories to analyze

        Returns:
            List of computed metrics with confidence intervals
        """
        return []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize environment to dictionary format.

        Extracts environment class information and constructor parameters
        to enable JSON serialization and reconstruction.

        Returns:
            Dictionary with structure:
                - class: Full class path (module.ClassName)
                - module: Module name
                - params: Constructor parameters
                - config_id: Deterministic configuration identifier

        Example:
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> env = TigerPOMDP(discount_factor=0.95)
            >>> env_dict = env.to_dict()
            >>> 'class' in env_dict and 'params' in env_dict
            True

        Note:
            Uses centralized serialization system with registered SpaceInfo handler.
        """
        # Get environment class information
        env_class = self.__class__
        env_module = env_class.__module__
        env_class_name = env_class.__name__

        # Extract constructor parameters
        sig = inspect.signature(env_class.__init__)
        params = {}

        for param_name, _ in sig.parameters.items():
            if param_name == "self":
                continue
            if hasattr(self, param_name):
                value = getattr(self, param_name)
                # Use centralized serialization (SpaceInfo handled by registered handler)
                serialized_value = serialize_value_base(value)
                if serialized_value is not None:  # Skip None values (like logger)
                    params[param_name] = serialized_value

        return {
            "class": f"{env_module}.{env_class_name}",
            "module": env_module,
            "params": params,
            "config_id": self.config_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Environment":
        """Reconstruct environment from dictionary.

        Dynamically imports the environment class and instantiates it
        with the saved parameters.

        Args:
            data: Dictionary containing environment serialization data
                with keys: class, module, params, config_id

        Returns:
            Reconstructed environment instance

        Raises:
            ImportError: If environment class cannot be imported
            ValueError: If required data fields are missing
            TypeError: If parameters are invalid for environment constructor

        Example:
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> env = TigerPOMDP(discount_factor=0.95)
            >>> env_dict = env.to_dict()
            >>> reconstructed_env = Environment.from_dict(env_dict)
            >>> reconstructed_env.discount_factor
            0.95
        """

        def deserialize_value(
            value, target_type, param_name=""
        ):  # pylint: disable=too-many-branches
            """Deserialize value with environment-specific handling.

            Handles environment-specific patterns before delegating to centralized system:
            - List[Tuple[...]] / Set[Tuple[...]] for obstacles, rock positions
            - Matrix parameters (covariance matrices) with parameter name detection

            Note:
                SpaceInfo is handled automatically by registered handler in centralized system.
            """
            # Unwrap Optional[T] types first
            unwrapped_type = target_type
            if hasattr(target_type, "__origin__") and target_type.__origin__ is Union:
                # Get non-None type from Optional
                # pylint: disable=unidiomatic-typecheck
                args = [arg for arg in target_type.__args__ if arg is not type(None)]
                if args:
                    unwrapped_type = args[0]

            # Environment-specific pattern: List[Tuple[...]] and Set[Tuple[...]]
            # Used by PushPOMDP (obstacles) and RockSamplePOMDP (rock_positions)
            # Handles multiple serialized formats for compatibility
            if hasattr(unwrapped_type, "__origin__"):
                if unwrapped_type.__origin__ in (list, set):
                    # Check if the element type is a tuple
                    args = getattr(unwrapped_type, "__args__", ())
                    if args and hasattr(args[0], "__origin__") and args[0].__origin__ is tuple:
                        # This is List[Tuple[...]] or Set[Tuple[...]]
                        if isinstance(value, list) and value:
                            # Format 1: Tuple markers like {'__type__': 'tuple', 'values': [x, y]}
                            if isinstance(value[0], dict) and value[0].get("__type__") == "tuple":
                                return [deserialize_value_base(elem, None) for elem in value]

                        # First deserialize the value (might be ndarray marker or plain list)
                        deserialized = deserialize_value_base(value, None)

                        # Format 2: NumPy array shape (2, N) → [(x1,y1), (x2,y2), ...]
                        if isinstance(deserialized, np.ndarray):
                            if deserialized.ndim == 2 and deserialized.shape[0] == 2:
                                return list(zip(deserialized[0], deserialized[1]))
                        # Format 3: 2D list [[x1,x2,...], [y1,y2,...]] → [(x1,y1), ...]
                        elif isinstance(deserialized, list) and deserialized:
                            if len(deserialized) == 2 and isinstance(deserialized[0], list):
                                return list(zip(deserialized[0], deserialized[1]))

            # Environment-specific pattern: Matrix parameter name detection
            # Ensures covariance matrices are always numpy arrays
            matrix_param_names = [
                "noise_cov",
                "_cov",
                "cov_matrix",
                "state_transition_cov_matrix",
                "observation_cov_matrix",
            ]
            if any(name in param_name.lower() for name in matrix_param_names):
                result = deserialize_value_base(value, target_type)
                if not isinstance(result, np.ndarray):
                    result = np.array(result)
                return result

            # Handle numpy array type annotations
            if target_type == np.ndarray or (
                hasattr(target_type, "__name__") and "ndarray" in target_type.__name__
            ):
                result = deserialize_value_base(value, target_type)
                if not isinstance(result, np.ndarray):
                    result = np.array(result)
                return result

            # Delegate to centralized deserialization for all other types
            return deserialize_value_base(value, target_type)

        # Validate required fields
        if "class" not in data or "module" not in data or "params" not in data:
            raise ValueError("Environment data missing required fields: class, module, or params")

        # Import environment class dynamically
        module_name = data["module"]
        class_name = data["class"].split(".")[-1]

        try:
            module = importlib.import_module(module_name)
            env_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import environment class {data['class']}: {str(e)}"
            ) from e

        # Deserialize parameters with type hints
        sig = inspect.signature(env_class.__init__)
        params = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name in data["params"]:
                value = data["params"][param_name]
                # Try to deserialize with type annotation if available
                if param.annotation != inspect.Parameter.empty:
                    value = deserialize_value(value, param.annotation, param_name)
                else:
                    value = deserialize_value(value, type(value), param_name)
                params[param_name] = value

        # Reconstruct environment
        try:
            return env_class(**params)
        except TypeError as e:
            raise TypeError(
                f"Failed to construct {class_name} with params {params}: {str(e)}"
            ) from e


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
