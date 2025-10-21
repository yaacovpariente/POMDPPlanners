from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, List
from POMDPPlanners.utils.config_to_id import config_to_id

from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterRunParams,
)

if TYPE_CHECKING:
    from POMDPPlanners.core.belief import Belief
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.policy import Policy, PolicySpaceInfo


@dataclass(frozen=True)
class EnvironmentRunParams:
    environment: "Environment"
    belief: "Belief"
    policies: Sequence["Policy"]
    num_episodes: int
    num_steps: int

    @property
    def config_id(self) -> str:
        return config_to_id(
            {
                "environment": self.environment.config_id,
                "belief": self.belief.config_id,
                "policies": sorted([policy.config_id for policy in self.policies]),
                "num_episodes": self.num_episodes,
                "num_steps": self.num_steps,
            }
        )

    def __hash__(self) -> int:
        return hash(self.config_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnvironmentRunParams):
            return False
        return self.config_id == other.config_id


class PlannerGenerator(ABC):
    @abstractmethod
    def generate(self, environment: "Environment") -> "Policy":
        pass

    @abstractmethod
    def get_planner_space_info(self) -> "PolicySpaceInfo":
        pass


class EvaluationExperimentConfigCreator(ABC):
    @abstractmethod
    def _get_experiment_configs(self) -> Sequence[EnvironmentRunParams]:
        pass

    def get_experiment_configs(self) -> List[EnvironmentRunParams]:
        configs = self._get_experiment_configs()
        config_ids = set([config.config_id for config in configs])

        if len(config_ids) != len(configs):
            raise ValueError("Duplicate configs found")

        return list(configs)


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
