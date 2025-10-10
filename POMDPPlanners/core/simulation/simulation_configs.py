from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    def _get_experiment_configs(self) -> Sequence[HyperParameterRunParams]:
        pass

    def get_experiment_configs(self) -> List[HyperParameterRunParams]:
        configs = self._get_experiment_configs()
        config_ids = set([config.config_id for config in configs])

        if len(config_ids) != len(configs):
            raise ValueError("Duplicate configs found")

        return list(configs)
