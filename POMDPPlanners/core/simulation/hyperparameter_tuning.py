from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

if TYPE_CHECKING:
    from POMDPPlanners.core.belief import Belief
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.policy import Policy
    from POMDPPlanners.core.policy import PolicySpaceInfo


class CategoricalHyperParameter(NamedTuple):
    choices: list[Any]
    name: str


class NumericalHyperParameter(NamedTuple):
    low: Union[int, float]
    high: Union[int, float]
    name: str


HyperParameterFeature = Union[CategoricalHyperParameter, NumericalHyperParameter]


class HyperParameterOptimizationDirection(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True)
class HyperParamPlannerConfig:
    policy_cls: Type["Policy"]
    hyper_parameters: Sequence[HyperParameterFeature]
    constant_parameters: Dict[str, Any]


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
