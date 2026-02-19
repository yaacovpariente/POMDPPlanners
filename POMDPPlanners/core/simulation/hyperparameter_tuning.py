from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.core.belief import Belief
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.policy import Policy
    from POMDPPlanners.core.policy import PolicySpaceInfo


class CategoricalHyperParameter(NamedTuple):
    choices: list[Any]
    name: str

    def id(self) -> str:
        return config_to_id(
            {
                "choices": self.choices,
                "name": self.name,
            }
        )

    def __hash__(self) -> int:
        return hash(self.id())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CategoricalHyperParameter):
            return False

        return self.id() == other.id()


class NumericalHyperParameter(NamedTuple):
    low: Union[int, float]
    high: Union[int, float]
    name: str

    def id(self) -> str:
        return config_to_id(
            {
                "low": self.low,
                "high": self.high,
                "name": self.name,
            }
        )

    def __hash__(self) -> int:
        return hash(self.id())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NumericalHyperParameter):
            return False

        return self.id() == other.id()


HyperParameterFeature = Union[CategoricalHyperParameter, NumericalHyperParameter]


class HyperParameterOptimizationDirection(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True)
class HyperParamPlannerConfig:
    policy_cls: Type["Policy"]
    hyper_parameters: Sequence[HyperParameterFeature]
    constant_parameters: Dict[str, Any]
    training_hyper_parameters: Sequence[HyperParameterFeature] = ()
    training_constant_parameters: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Validate input types and hyperparameter names."""
        # Frozen dataclass requires object.__setattr__ for defaults
        if self.training_constant_parameters is None:
            object.__setattr__(self, "training_constant_parameters", {})
        self._validate_types()
        self._validate_hyperparameter_names()

    def _validate_types(self) -> None:
        """Verify input types are correct."""
        if not isinstance(self.policy_cls, type):
            raise TypeError(
                f"policy_cls must be a class type, got {type(self.policy_cls).__name__}"
            )

        # Validate hyper_parameters is a sequence (list or tuple)
        if not isinstance(self.hyper_parameters, (list, tuple)):
            raise TypeError(
                f"hyper_parameters must be a Sequence (list or tuple), got {type(self.hyper_parameters).__name__}"
            )

        # Validate each hyperparameter is of correct type
        for i, param in enumerate(self.hyper_parameters):
            if not isinstance(param, (CategoricalHyperParameter, NumericalHyperParameter)):
                raise TypeError(
                    f"hyper_parameters[{i}] must be either CategoricalHyperParameter or NumericalHyperParameter, "
                    f"got {type(param).__name__}"
                )

        # Validate constant_parameters is a dict
        if not isinstance(self.constant_parameters, dict):
            raise TypeError(
                f"constant_parameters must be a dict, got {type(self.constant_parameters).__name__}"
            )

        # Validate training_hyper_parameters
        if not isinstance(self.training_hyper_parameters, (list, tuple)):
            raise TypeError(
                f"training_hyper_parameters must be a Sequence, "
                f"got {type(self.training_hyper_parameters).__name__}"
            )
        for i, param in enumerate(self.training_hyper_parameters):
            if not isinstance(param, (CategoricalHyperParameter, NumericalHyperParameter)):
                raise TypeError(
                    f"training_hyper_parameters[{i}] must be a HyperParameterFeature, "
                    f"got {type(param).__name__}"
                )

        if not isinstance(self.training_constant_parameters, dict):
            raise TypeError(
                f"training_constant_parameters must be a dict, "
                f"got {type(self.training_constant_parameters).__name__}"
            )

    def _validate_hyperparameter_names(self) -> None:
        """Verify all hyperparameters correspond to policy class constructor parameters."""
        try:
            sig = signature(self.policy_cls.__init__)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot inspect signature of policy_cls {self.policy_cls.__name__}: {e}"
            ) from e

        # Get valid parameter names from the policy class constructor
        # Exclude 'self' as it's implicit
        valid_param_names = set(sig.parameters.keys()) - {"self"}

        # Check each hyperparameter
        for param in self.hyper_parameters:
            if param.name not in valid_param_names:
                raise ValueError(
                    f"Hyperparameter '{param.name}' is not a valid parameter of "
                    f"{self.policy_cls.__name__}.__init__(). "
                    f"Valid parameters are: {sorted(valid_param_names)}"
                )

        # Check constant parameters too
        for param_name in self.constant_parameters.keys():
            if param_name not in valid_param_names:
                raise ValueError(
                    f"Constant parameter '{param_name}' is not a valid parameter of "
                    f"{self.policy_cls.__name__}.__init__(). "
                    f"Valid parameters are: {sorted(valid_param_names)}"
                )

        # Validate training hyperparameters against PolicyTrainer.__init__
        if self.training_hyper_parameters or self.training_constant_parameters:
            self._validate_training_parameter_names()

    def _validate_training_parameter_names(self) -> None:
        from POMDPPlanners.training.policy_trainer import (  # pylint: disable=import-outside-toplevel
            PolicyTrainer,
        )

        try:
            sig = signature(PolicyTrainer.__init__)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot inspect PolicyTrainer signature: {e}") from e

        valid_names = set(sig.parameters.keys()) - {"self"}
        for param in self.training_hyper_parameters:
            if param.name not in valid_names:
                raise ValueError(
                    f"Training hyperparameter '{param.name}' is not a valid parameter of "
                    f"PolicyTrainer.__init__(). Valid parameters are: {sorted(valid_names)}"
                )
        for param_name in self.training_constant_parameters.keys():
            if param_name not in valid_names:
                raise ValueError(
                    f"Training constant parameter '{param_name}' is not a valid parameter of "
                    f"PolicyTrainer.__init__(). Valid parameters are: {sorted(valid_names)}"
                )

    @property
    def config_id(self) -> str:
        return config_to_id(
            {
                "policy_cls": self.policy_cls.__name__,
                "hyper_parameters": sorted([param.id() for param in self.hyper_parameters]),
                "constant_parameters": self.constant_parameters,
                "training_hyper_parameters": sorted(
                    [param.id() for param in self.training_hyper_parameters]
                ),
                "training_constant_parameters": self.training_constant_parameters,
            }
        )


class ParameterToOptimizeMapper(ABC):
    @abstractmethod
    def generate(
        self, environment: "Environment", policy_cls: Optional[Type["Policy"]] = None
    ) -> List[Tuple[str, HyperParameterOptimizationDirection]]:
        pass


@dataclass(frozen=True)
class HyperParameterRunParams:
    """Configuration parameters for hyperparameter optimization runs.

    This frozen dataclass contains all parameters needed to configure and execute
    a hyperparameter optimization run. Input validation is performed at construction
    time to ensure all parameters are valid before optimization begins.

    Attributes:
        environment: POMDP environment instance to optimize policies for
        belief: Initial belief state for the environment
        hyper_param_planner_config: Configuration defining policy class, hyperparameters,
            and constant parameters for optimization
        num_episodes: Number of episodes to run per trial (must be positive)
        num_steps: Maximum number of steps per episode (must be positive)
        n_trials: Number of optimization trials to execute (must be positive)
        parameters_to_optimize: List of (metric_name, direction) tuples specifying
            which metrics to optimize and in which direction (maximize/minimize)

    Raises:
        ValueError: If any numerical parameter is non-positive, if hyperparameters
            or parameters_to_optimize are empty, or if metric names are invalid
        TypeError: If environment, belief, or policy_cls have incorrect types
    """

    environment: "Environment"
    belief: "Belief"
    hyper_param_planner_config: HyperParamPlannerConfig
    num_episodes: int
    num_steps: int
    n_trials: int
    parameters_to_optimize: List[Tuple[str, HyperParameterOptimizationDirection]]

    def __post_init__(self) -> None:
        """Validate all parameters at construction time."""
        # Import validation dependencies at runtime to avoid circular imports
        from POMDPPlanners.core.environment import (
            Environment,
        )  # pylint: disable=import-outside-toplevel
        from POMDPPlanners.core.belief import Belief  # pylint: disable=import-outside-toplevel
        from POMDPPlanners.core.policy import Policy  # pylint: disable=import-outside-toplevel
        from POMDPPlanners.simulations.simulation_statistics import (  # pylint: disable=import-outside-toplevel
            get_metric_names_from_environment_policy_pair,
        )

        # Validate numerical parameters
        if self.num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {self.num_episodes}")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if self.n_trials <= 0:
            raise ValueError(f"n_trials must be positive, got {self.n_trials}")

        # Validate types
        if not isinstance(self.environment, Environment):
            raise TypeError(
                f"environment must be an Environment instance, "
                f"got {type(self.environment).__name__}"
            )
        if not isinstance(self.belief, Belief):
            raise TypeError(f"belief must be a Belief instance, got {type(self.belief).__name__}")
        if not (
            isinstance(self.hyper_param_planner_config.policy_cls, type)
            and issubclass(self.hyper_param_planner_config.policy_cls, Policy)
        ):
            raise TypeError(
                f"policy_cls must be a Policy subclass, got {self.hyper_param_planner_config.policy_cls}"
            )

        # Validate non-empty collections
        if not isinstance(self.hyper_param_planner_config.hyper_parameters, (list, tuple)):
            raise TypeError(
                f"hyper_parameters must be list or tuple, got {type(self.hyper_param_planner_config.hyper_parameters).__name__}"
            )
        if not self.hyper_param_planner_config.hyper_parameters:
            raise ValueError("hyper_parameters cannot be empty")

        if not isinstance(self.parameters_to_optimize, list):
            raise TypeError(
                f"parameters_to_optimize must be list, got {type(self.parameters_to_optimize).__name__}"
            )
        if not self.parameters_to_optimize:
            raise ValueError("parameters_to_optimize cannot be empty")

        # Validate metric names
        policy_cls = self.hyper_param_planner_config.policy_cls
        available_metrics = get_metric_names_from_environment_policy_pair(
            self.environment, policy_cls
        )
        for metric_name, _ in self.parameters_to_optimize:
            if metric_name not in available_metrics:
                raise ValueError(
                    f"Invalid metric name '{metric_name}' in parameters_to_optimize. "
                    f"Available metrics for {self.environment.__class__.__name__} "
                    f"with {policy_cls.__name__}: {available_metrics}"
                )

    @property
    def config_id(self) -> str:
        return config_to_id(
            {
                "environment": self.environment.config_id,
                "belief": self.belief.config_id,
                "hyper_param_planner_config": self.hyper_param_planner_config.config_id,
                "num_episodes": self.num_episodes,
                "num_steps": self.num_steps,
                "n_trials": self.n_trials,
                "parameters_to_optimize": self.parameters_to_optimize,
            }
        )


@dataclass(frozen=True)
class OptimizedPolicyResult:
    """Result of hyperparameter optimization containing the optimized policy and metrics.

    This frozen dataclass contains all information about a completed hyperparameter
    optimization run, including the optimized policy, chosen hyperparameters, and
    achieved metric values. Input validation is performed at construction time.

    Attributes:
        environment: POMDP environment instance used for optimization
        policy: Optimized policy instance with best hyperparameters
        chosen_hyper_parameters: Dictionary of hyperparameter names to chosen values
        num_episodes: Number of episodes run per trial (must be positive)
        num_steps: Maximum number of steps per episode (must be positive)
        parameters_to_optimize: List of (metric_name, direction) tuples that were optimized
        optimized_metric_values: Dictionary mapping metric names to achieved values
            (None if metric value not found)

    Raises:
        ValueError: If num_episodes or num_steps are non-positive, if chosen_hyper_parameters
            or parameters_to_optimize are empty, or if metric names are invalid
        TypeError: If environment, policy types are incorrect, or if data structures
            have wrong types
    """

    environment: "Environment"
    policy: "Policy"
    chosen_hyper_parameters: dict
    num_episodes: int
    num_steps: int
    parameters_to_optimize: List[Tuple[str, HyperParameterOptimizationDirection]]
    optimized_metric_values: Dict[
        str, Optional[float]
    ]  # Actual metric values achieved (None if not found)

    def __post_init__(self) -> None:  # pylint: disable=too-many-branches
        """Validate all parameters at construction time."""
        # Import validation dependencies at runtime to avoid circular imports
        from POMDPPlanners.core.environment import (
            Environment,
        )  # pylint: disable=import-outside-toplevel
        from POMDPPlanners.core.policy import Policy  # pylint: disable=import-outside-toplevel
        from POMDPPlanners.simulations.simulation_statistics import (  # pylint: disable=import-outside-toplevel
            get_metric_names_from_environment_policy_pair,
        )

        # Validate numerical parameters
        if self.num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {self.num_episodes}")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")

        # Validate types
        if not isinstance(self.environment, Environment):
            raise TypeError(
                f"environment must be an Environment instance, "
                f"got {type(self.environment).__name__}"
            )
        if not isinstance(self.policy, Policy):
            raise TypeError(f"policy must be a Policy instance, got {type(self.policy).__name__}")

        # Validate non-empty collections
        if not isinstance(self.chosen_hyper_parameters, dict):
            raise TypeError(
                f"chosen_hyper_parameters must be a dict, got {type(self.chosen_hyper_parameters).__name__}"
            )
        if not self.chosen_hyper_parameters:
            raise ValueError("chosen_hyper_parameters dict cannot be empty")

        if not isinstance(self.parameters_to_optimize, list):
            raise TypeError(
                f"parameters_to_optimize must be list, got {type(self.parameters_to_optimize).__name__}"
            )
        if not self.parameters_to_optimize:
            raise ValueError("parameters_to_optimize cannot be empty")

        # Validate each parameter_to_optimize tuple
        for i, param_tuple in enumerate(self.parameters_to_optimize):
            if not isinstance(param_tuple, tuple) or len(param_tuple) != 2:
                raise TypeError(
                    f"parameters_to_optimize[{i}] must be a tuple of length 2, "
                    f"got {type(param_tuple).__name__} with length {len(param_tuple) if isinstance(param_tuple, tuple) else 'N/A'}"
                )
            metric_name, direction = param_tuple
            if not isinstance(metric_name, str):
                raise TypeError(
                    f"parameters_to_optimize[{i}][0] (metric_name) must be a str, "
                    f"got {type(metric_name).__name__}"
                )
            if not isinstance(direction, HyperParameterOptimizationDirection):
                raise TypeError(
                    f"parameters_to_optimize[{i}][1] (direction) must be a "
                    f"HyperParameterOptimizationDirection, got {type(direction).__name__}"
                )

        # Validate optimized_metric_values is a dict (can be empty in some cases)
        if not isinstance(self.optimized_metric_values, dict):
            raise TypeError(
                f"optimized_metric_values must be a dict, "
                f"got {type(self.optimized_metric_values).__name__}"
            )

        # Validate metric names against available metrics
        policy_cls = type(self.policy)
        available_metrics = get_metric_names_from_environment_policy_pair(
            self.environment, policy_cls
        )

        for metric_name, _ in self.parameters_to_optimize:
            if metric_name not in available_metrics:
                raise ValueError(
                    f"Invalid metric name '{metric_name}' in parameters_to_optimize. "
                    f"Available metrics for {self.environment.__class__.__name__} "
                    f"with {policy_cls.__name__}: {available_metrics}"
                )

        # Validate that optimized_metric_values contains all optimized metrics
        optimized_metric_names = {name for name, _ in self.parameters_to_optimize}
        for metric_name in optimized_metric_names:
            if metric_name not in self.optimized_metric_values:
                raise ValueError(
                    f"Metric '{metric_name}' in parameters_to_optimize is missing "
                    f"from optimized_metric_values. Expected keys: {optimized_metric_names}, "
                    f"got: {set(self.optimized_metric_values.keys())}"
                )

        # Validate that optimized_metric_values doesn't have extra metrics
        for metric_name in self.optimized_metric_values.keys():
            if metric_name not in optimized_metric_names:
                raise ValueError(
                    f"Metric '{metric_name}' in optimized_metric_values was not "
                    f"in parameters_to_optimize. Expected keys: {optimized_metric_names}, "
                    f"got: {set(self.optimized_metric_values.keys())}"
                )


class HyperParamPlannerConfigGenerator(ABC):
    @abstractmethod
    def generate(self, environment: "Environment") -> HyperParamPlannerConfig:
        pass

    @abstractmethod
    def get_planner_space_info(self) -> "PolicySpaceInfo":
        pass
