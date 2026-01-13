"""Module for POMDP policy abstractions and execution tracking.

This module provides the foundational interface for POMDP policies, including
abstract base classes for policy implementations and data structures for
tracking policy execution and performance metrics.

Classes:
    Policy: Abstract base class for all POMDP policies
    PolicySpaceInfo: Space type information for policy compatibility
    PolicyInfoVariable: Named tuple for policy execution metrics
    PolicyRunData: Container for policy execution information
"""

import importlib
import inspect
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

from POMDPPlanners.utils.config_to_id import config_to_id, NumpyEncoder
from POMDPPlanners.utils.logger import get_logger

from POMDPPlanners.core.environment import Environment, SpaceType

if TYPE_CHECKING:
    from POMDPPlanners.core.belief import Belief

    # from POMDPPlanners.core.environment import Environment, SpaceType


@dataclass
class PolicySpaceInfo:
    """Data class containing space type requirements for policy compatibility.

    This class specifies the action and observation space types that a policy
    is designed to work with, enabling compatibility checking with environments.

    Attributes:
        action_space: Required action space type (discrete, continuous, or mixed)
        observation_space: Required observation space type (discrete, continuous, or mixed)
    """

    action_space: "SpaceType"
    observation_space: "SpaceType"


class PolicyInfoVariable(NamedTuple):
    """Named tuple for storing policy execution metrics.

    This structure stores key-value pairs of policy performance metrics
    that are collected during policy execution.

    Attributes:
        name: Descriptive name of the metric (e.g., "nodes_expanded", "planning_time")
        value: Numeric value of the metric
    """

    name: str
    value: Union[float, int]


class PolicyRunData(NamedTuple):
    """Container for policy execution information and metrics.

    This class aggregates all the information collected during a policy's
    action selection process, including performance metrics and execution details.

    Attributes:
        info_variables: List of policy-specific metrics and performance data
    """

    info_variables: List[PolicyInfoVariable]


# Module-level helper functions for Policy save/load


def _serialize_value(value: Any) -> Any:
    """Serialize value for JSON compatibility.

    Args:
        value: Value to serialize

    Returns:
        JSON-serializable representation of value
    """
    if value is None:
        return None
    elif isinstance(value, Path):
        return str(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.integer, np.floating)):
        return value.item()
    elif isinstance(value, Enum):
        return value.value
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    elif isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    elif isinstance(value, logging.Logger):
        return None  # Skip loggers
    else:
        # For unknown types, try to convert to string
        return str(value)


def _deserialize_value(value: Any, target_type: type) -> Any:
    """Deserialize value to target type.

    Args:
        value: Serialized value
        target_type: Target type for deserialization

    Returns:
        Value converted to target type
    """
    if value is None:
        return None

    # Handle Path objects
    if target_type == Path:
        return Path(value) if value is not None else None

    # Handle Optional types
    if hasattr(target_type, "__origin__") and target_type.__origin__ is Union:
        # Get the non-None type from Optional[T]
        args = [arg for arg in target_type.__args__ if arg is not type(None)]
        if args:
            return _deserialize_value(value, args[0])

    return value


def _extract_constructor_params(policy: "Policy") -> Dict[str, Any]:
    """Extract constructor parameters from policy instance.

    Uses inspect.signature() to discover constructor parameters and walks
    through the class hierarchy to capture all parameters.

    Args:
        policy: Policy instance to extract parameters from

    Returns:
        Dictionary of parameter names to values
    """
    params = {}

    # Walk through class hierarchy (Policy → PathSimulationPolicy → Concrete)
    for cls in inspect.getmro(policy.__class__):
        if cls == object or not issubclass(cls, Policy):
            break

        sig = inspect.signature(cls.__init__)
        for param_name, _ in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name == "environment":
                # Skip environment - handle separately
                continue

            # Get current value from instance
            if hasattr(policy, param_name):
                value = getattr(policy, param_name)
                # Skip action_sampler - handle separately
                if param_name == "action_sampler":
                    continue
                params[param_name] = _serialize_value(value)

    return params


def _serialize_action_sampler(action_sampler: Any) -> Dict[str, Any]:
    """Serialize ActionSampler object.

    Uses ActionSampler's __getstate__ method for serialization.

    Args:
        action_sampler: ActionSampler instance to serialize

    Returns:
        Dictionary with class info and state
    """
    return {
        "class": f"{action_sampler.__class__.__module__}.{action_sampler.__class__.__name__}",
        "module": action_sampler.__class__.__module__,
        "state": action_sampler.__getstate__(),
    }


def _deserialize_action_sampler(sampler_data: Dict[str, Any]) -> Any:
    """Reconstruct ActionSampler from serialized data.

    Args:
        sampler_data: Dictionary containing ActionSampler class and state

    Returns:
        Reconstructed ActionSampler instance
    """
    # Import ActionSampler class
    module_name = sampler_data["module"]
    class_name = sampler_data["class"].split(".")[-1]

    module = importlib.import_module(module_name)
    sampler_class = getattr(module, class_name)

    # Use __reduce__ pattern: create instance, then restore state
    sampler = sampler_class.__new__(sampler_class)
    sampler.__setstate__(sampler_data["state"])

    return sampler


def _get_default_filepath(policy: "Policy", base_dir: Path = Path("saved_policies")) -> Path:
    """Generate default filepath for saving policy.

    Args:
        policy: Policy instance
        base_dir: Base directory for saved policies

    Returns:
        Path with structure: {base_dir}/{env_name}/{policy_class}/{policy_name}_{timestamp}.json
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    env_name = policy.environment.name
    policy_class_name = policy.__class__.__name__
    policy_name = policy.name

    filepath = base_dir / env_name / policy_class_name / f"{policy_name}_{timestamp}.json"
    return filepath


def _get_package_version() -> str:
    """Get POMDPPlanners package version.

    Returns:
        Package version string or "unknown"
    """
    try:
        import pkg_resources

        return pkg_resources.get_distribution("POMDPPlanners").version
    except Exception:  # pylint: disable=broad-exception-caught
        # Catch all exceptions to ensure function always returns a version string
        return "unknown"


class Policy(ABC):
    """Abstract base class for POMDP policies.

    This class defines the interface for POMDP policies that select actions
    based on belief states. All concrete policy implementations must inherit
    from this class and implement the action selection and space information methods.

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses must implement the action() and get_space_info() methods.

    Attributes:
        environment: The POMDP environment this policy operates in
        discount_factor: Discount factor for future rewards
        name: Unique identifier for the policy
        log_path: Optional directory for logging output
        debug: Flag to enable debug logging
    """

    def __init__(
        self,
        environment: "Environment",
        discount_factor: float,
        name: str,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the POMDP policy.

        Args:
            environment: Environment that this policy will operate in
            discount_factor: Discount factor for future rewards (0 < discount_factor <= 1)
            name: Unique identifier for this policy instance
            log_path: Optional directory for logging output. Defaults to None.
            debug: Enable debug logging. Defaults to False.
        """
        self.environment = environment
        self.discount_factor = discount_factor
        self.name = name
        self.log_path = log_path
        self.debug = debug
        self.use_queue_logger = use_queue_logger

        self._verify_environment_compatibility()

        # Initialize logger with the policy's name and user-specified settings
        self.logger.info("Initialized policy: %s (debug=%s)", self.name, self.debug)

    def _verify_environment_compatibility(self) -> None:
        """Verify that the policy is compatible with the environment."""
        policy_space_info = self.get_space_info()
        environment_space_info = self.environment.space_info

        if (
            policy_space_info.action_space == SpaceType.DISCRETE
            and environment_space_info.action_space in [SpaceType.CONTINUOUS, SpaceType.MIXED]
        ):
            raise ValueError(
                f"Policy {self.name} is not compatible with the environment {self.environment.name} because the policy assumes discrete action space and the environment assumes continuous action space"
            )

        if (
            policy_space_info.observation_space == SpaceType.DISCRETE
            and environment_space_info.observation_space in [SpaceType.CONTINUOUS, SpaceType.MIXED]
        ):
            raise ValueError(
                f"Policy {self.name} is not compatible with the environment {self.environment.name} because the policy assumes discrete observation space and the environment assumes continuous observation space"
            )

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this policy.

        The logger is implemented as a property to maintain pickle compatibility,
        as logger objects cannot be pickled directly.

        Returns:
            Configured logger instance with hierarchical naming
        """
        return get_logger(
            name=f"policy.{self.name}",
            level=logging.INFO,
            output_dir=self.log_path,
            debug=self.debug,
            use_queue=self.use_queue_logger,
        )

    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on policy configuration."""

        def serialize_value(value):
            """Helper function to serialize values in a deterministic way."""
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (str, int, float, bool)):
                return value
            elif isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            elif isinstance(value, dict):
                return {str(k): serialize_value(v) for k, v in sorted(value.items())}
            elif isinstance(value, logging.Logger):
                # Exclude logger from config serialization
                return None
            elif hasattr(value, "__dict__"):
                # Skip logger objects in __dict__ to avoid recursion
                if isinstance(value, logging.Logger):
                    return serialize_value(value.name)
                return serialize_value(value.__dict__)
            else:
                return str(value)

        config_dict = {}
        for key, value in self.__dict__.items():
            if key.startswith("_") or callable(value):
                continue
            config_dict[key] = serialize_value(value)
        config_dict = dict(sorted(config_dict.items()))
        config_dict["environment"] = self.environment.config_id

        return config_to_id(config_dict)

    def __hash__(self) -> int:
        return hash(self.config_id)

    @abstractmethod
    def action(self, belief: "Belief") -> Tuple[List[Any], PolicyRunData]:
        """Select action(s) based on the current belief state.

        This is the core method that implements the policy's decision-making logic.
        It takes a belief state and returns the selected action(s) along with
        execution information and performance metrics.

        Args:
            belief: Current belief state representing uncertainty over states

        Returns:
            Tuple containing:
                - List of selected actions (typically single action, but supports multiple)
                - PolicyRunData with execution metrics and performance information

        Note:
            Subclasses must implement this method with their specific planning
            or decision-making algorithm.
        """
        pass

    @classmethod
    @abstractmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        """Get space type requirements for this policy class.

        This class method specifies what types of action and observation spaces
        this policy implementation can handle, enabling compatibility checking
        with environments.

        Returns:
            PolicySpaceInfo specifying required action and observation space types

        Note:
            Subclasses must implement this method to declare their space compatibility.
            This is used for validation when pairing policies with environments.
        """
        pass

    @classmethod
    @abstractmethod
    def get_info_variable_names(cls) -> List[str]:
        """Get names of policy info variables that this policy produces.

        This class method returns the names of metrics and performance data
        that the policy tracks during execution via PolicyInfoVariable objects.
        It enables users to discover what metrics are available for hyperparameter
        optimization before running simulations.

        Returns:
            List of info variable names that this policy produces during action selection

        Note:
            Subclasses must implement this method to declare what metrics they track.
            Use an Enum to ensure consistency between the names returned here and
            the names used when creating PolicyInfoVariable objects in the action() method.
        """
        pass

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Save policy configuration to JSON file.

        Saves only constructor parameters needed to reconstruct the policy,
        not the full internal state. This enables human-readable policy
        configurations that can be versioned, inspected, and modified.

        Args:
            filepath: Path where to save the policy configuration.
                If None, uses default location:
                saved_policies/{env_name}/{policy_class}/{policy_name}_{timestamp}.json

        Returns:
            Path where policy was saved

        Raises:
            ValueError: If policy parameters cannot be serialized
            IOError: If file cannot be written

        Example:
            >>> from POMDPPlanners.environments import TigerPOMDP
            >>> from POMDPPlanners.planners import POMCP
            >>> env = TigerPOMDP(discount_factor=0.95)
            >>> planner = POMCP(environment=env, discount_factor=0.95,
            ...                 depth=10, exploration_constant=1.0,
            ...                 name="test", n_simulations=100)
            >>> # Save with default path
            >>> filepath = planner.save()
            >>> # Or save to custom path
            >>> filepath = planner.save("my_policy.json")
        """
        if filepath is None:
            filepath = _get_default_filepath(self)

        filepath = Path(filepath)

        try:
            # Extract policy parameters
            params = _extract_constructor_params(self)

            # Serialize environment
            env_data = self.environment.to_dict()

            # Build save dictionary
            save_data = {
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "pomdpplanners_version": _get_package_version(),
                    "policy_class": f"{self.__class__.__module__}.{self.__class__.__name__}",
                    "policy_config_id": self.config_id,
                    "format_version": "1.0",
                },
                "environment": env_data,
                "policy": {"params": params},
            }

            # Handle action_sampler if present
            action_sampler = getattr(self, "action_sampler", None)
            if action_sampler is not None:
                save_data["action_sampler"] = _serialize_action_sampler(action_sampler)

            # Write to file with NumpyEncoder
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, cls=NumpyEncoder)

            return filepath

        except Exception as e:
            raise ValueError(f"Failed to save policy: {str(e)}") from e

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Policy":
        """Load policy configuration from JSON file.

        Reconstructs policy instance from saved constructor parameters.
        Creates both the environment and policy from the saved configuration.

        Args:
            filepath: Path to the saved policy configuration file

        Returns:
            Reconstructed policy instance

        Raises:
            FileNotFoundError: If filepath does not exist
            ValueError: If JSON format is invalid or unsupported
            ImportError: If policy/environment classes cannot be imported

        Example:
            >>> from POMDPPlanners.planners import POMCP
            >>> # Load policy from file
            >>> planner = POMCP.load("saved_policies/TigerPOMDP/POMCP/test_20260113_103045.json")
            >>> # Verify reconstruction
            >>> print(planner.depth)  # 10
            >>> print(planner.exploration_constant)  # 1.0
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Policy file not found: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate format version
            format_version = data.get("metadata", {}).get("format_version")
            if format_version != "1.0":
                raise ValueError(f"Unsupported format version: {format_version}")

            # Get policy class
            policy_class_path = data["metadata"]["policy_class"]
            module_name = ".".join(policy_class_path.split(".")[:-1])
            class_name = policy_class_path.split(".")[-1]

            module = importlib.import_module(module_name)
            policy_class = getattr(module, class_name)

            # Reconstruct environment
            environment = Environment.from_dict(data["environment"])

            # Get policy parameters
            policy_params = data["policy"]["params"].copy()
            policy_params["environment"] = environment

            # Handle action_sampler if present
            if "action_sampler" in data:
                sampler = _deserialize_action_sampler(data["action_sampler"])
                policy_params["action_sampler"] = sampler

            # Deserialize parameter types
            sig = inspect.signature(policy_class.__init__)
            for param_name, param in sig.parameters.items():
                if param_name in policy_params and param.annotation != inspect.Parameter.empty:
                    policy_params[param_name] = _deserialize_value(
                        policy_params[param_name], param.annotation
                    )

            # Construct policy
            policy = policy_class(**policy_params)

            # Optionally warn about config_id mismatch
            loaded_config_id = policy.config_id
            saved_config_id = data["metadata"].get("policy_config_id")

            if loaded_config_id != saved_config_id:
                import warnings

                warnings.warn(
                    f"Loaded policy config_id ({loaded_config_id}) differs from saved "
                    f"config_id ({saved_config_id}). This may indicate parameter mismatch."
                )

            return policy

        except Exception as e:
            raise ValueError(f"Failed to load policy from {filepath}: {str(e)}") from e
