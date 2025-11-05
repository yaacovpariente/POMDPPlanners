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

    def __post_init__(self) -> None:
        """Validate input types and hyperparameter names."""
        self._validate_types()
        self._validate_hyperparameter_names()

    def _validate_types(self) -> None:
        """Verify input types are correct."""
        if not isinstance(self.policy_cls, type):
            raise TypeError(
                f"policy_cls must be a class type, got {type(self.policy_cls).__name__}"
            )

        if not isinstance(self.hyper_parameters, (list, tuple)):
            raise TypeError(
                f"hyper_parameters must be a Sequence (list or tuple), got {type(self.hyper_parameters).__name__}"
            )

        for i, param in enumerate(self.hyper_parameters):
            if not isinstance(param, (CategoricalHyperParameter, NumericalHyperParameter)):
                raise TypeError(
                    f"hyper_parameters[{i}] must be either CategoricalHyperParameter or NumericalHyperParameter, "
                    f"got {type(param).__name__}"
                )

        if not isinstance(self.constant_parameters, dict):
            raise TypeError(
                f"constant_parameters must be a dict, got {type(self.constant_parameters).__name__}"
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

    @property
    def config_id(self) -> str:
        return config_to_id(
            {
                "policy_cls": self.policy_cls.__name__,
                "hyper_parameters": sorted([param.id() for param in self.hyper_parameters]),
                "constant_parameters": self.constant_parameters,
            }
        )


class ParameterToOptimizeMapper(ABC):
    @abstractmethod
    def generate(
        self, environment: "Environment", policy_cls: Optional[Type["Policy"]] = None
    ) -> List[Tuple[str, HyperParameterOptimizationDirection]]:
        pass


class HyperParameterRunParams(NamedTuple):
    environment: "Environment"
    belief: "Belief"
    hyper_param_planner_config: HyperParamPlannerConfig
    num_episodes: int
    num_steps: int
    n_trials: int
    parameters_to_optimize: List[Tuple[str, HyperParameterOptimizationDirection]]

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


class OptimizedPolicyResult(NamedTuple):
    environment: "Environment"
    policy: "Policy"
    chosen_hyper_parameters: dict
    num_episodes: int
    num_steps: int
    parameters_to_optimize: List[Tuple[str, HyperParameterOptimizationDirection]]
    optimized_metric_values: Dict[
        str, Optional[float]
    ]  # Actual metric values achieved (None if not found)


class HyperParamPlannerConfigGenerator(ABC):
    @abstractmethod
    def generate(self, environment: "Environment") -> HyperParamPlannerConfig:
        pass

    @abstractmethod
    def get_planner_space_info(self) -> "PolicySpaceInfo":
        pass
