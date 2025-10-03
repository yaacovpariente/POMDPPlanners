from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, NamedTuple, Sequence, Type, Union

if TYPE_CHECKING:
    from POMDPPlanners.core.belief import Belief
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.policy import Policy


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


class HyperParamPlannerConfig:
    def __init__(
        self,
        policy_cls: type,
        hyper_parameters: Sequence[HyperParameterFeature],
        constant_parameters: Dict[str, Any],
    ):
        self.policy_cls = policy_cls
        self.hyper_parameters = hyper_parameters
        self.constant_parameters = constant_parameters


class HyperParameterRunParams(NamedTuple):
    environment: "Environment"
    belief: "Belief"
    hyper_param_planner_config: HyperParamPlannerConfig
    num_episodes: int
    num_steps: int
    n_trials: int
    direction: HyperParameterOptimizationDirection
    parameter_to_optimize: str


class OptimizedPolicyResult(NamedTuple):
    environment: "Environment"
    policy: "Policy"
    chosen_hyper_parameters: dict
    num_episodes: int
    num_steps: int
    direction: HyperParameterOptimizationDirection
    parameter_to_optimize: str
