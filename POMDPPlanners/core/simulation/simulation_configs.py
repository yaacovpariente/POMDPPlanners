from abc import ABC, abstractmethod
from dataclasses import dataclass
from ossaudiodev import SNDCTL_SEQ_SYNC
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, List

from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterFeature,
    HyperParameterRunParams,
)

if TYPE_CHECKING:
    from POMDPPlanners.core.belief import Belief
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.policy import Policy


@dataclass(frozen=True)
class EnvironmentRunParams:
    environment: "Environment"
    belief: "Belief"
    policies: Sequence["Policy"]
    num_episodes: int
    num_steps: int


class EvaluationExperimentConfigCreator(ABC):
    @abstractmethod
    def get_experiment_configs(self) -> Sequence[EnvironmentRunParams]:
        pass


class HyperparameterOptimizationExperimentConfigCreator(ABC):
    @abstractmethod
    def get_experiment_configs(self) -> Sequence[HyperParameterRunParams]:
        pass
