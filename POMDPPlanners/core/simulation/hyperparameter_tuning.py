from abc import ABC, abstractmethod
from typing import Any, NamedTuple, Union, Type, List, Literal, TYPE_CHECKING
from pathlib import Path
from enum import Enum

if TYPE_CHECKING:
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.policy import Policy
    from POMDPPlanners.core.belief import Belief


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


class HyperParameterRunParams(NamedTuple):
    environment: "Environment"
    belief: "Belief"
    policy_cls: Type["Policy"]
    hyper_parameters: List[HyperParameterFeature]
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
