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


def _get_environment_run_params_validation_dependencies():
    """Import validation dependencies at runtime to avoid circular imports.

    Returns:
        Tuple containing Environment, Belief, Policy classes
    """
    from POMDPPlanners.core.environment import (
        Environment,
    )  # pylint: disable=import-outside-toplevel
    from POMDPPlanners.core.belief import Belief  # pylint: disable=import-outside-toplevel
    from POMDPPlanners.core.policy import Policy  # pylint: disable=import-outside-toplevel

    return Environment, Belief, Policy


@dataclass(frozen=True)
class EnvironmentRunParams:
    """Configuration parameters for environment evaluation runs.

    This frozen dataclass contains all parameters needed to configure and execute
    an environment evaluation run. Input validation is performed at construction
    time to ensure all parameters are valid before execution begins.

    Attributes:
        environment: POMDP environment instance to evaluate policies in
        belief: Initial belief state for the environment
        policies: Sequence of policy instances to evaluate (must be non-empty)
        num_episodes: Number of episodes to run per policy (must be positive)
        num_steps: Maximum number of steps per episode (must be positive)

    Raises:
        ValueError: If any numerical parameter is non-positive or if policies list is empty
        TypeError: If environment, belief, or any policy has incorrect type
    """

    environment: "Environment"
    belief: "Belief"
    policies: Sequence["Policy"]
    num_episodes: int
    num_steps: int

    def __post_init__(self) -> None:
        """Validate all parameters at construction time."""
        # Import validation dependencies at runtime to avoid circular imports
        Environment, Belief, Policy = _get_environment_run_params_validation_dependencies()

        # Validate numerical parameters are integers
        if not isinstance(self.num_episodes, int):
            raise TypeError(
                f"num_episodes must be an integer, got {type(self.num_episodes).__name__}"
            )

        if not isinstance(self.num_steps, int):
            raise TypeError(f"num_steps must be an integer, got {type(self.num_steps).__name__}")

        # Validate numerical parameters are positive
        if self.num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {self.num_episodes}")

        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")

        # Validate environment
        if not isinstance(self.environment, Environment):
            raise TypeError(
                f"environment must be an Environment instance, "
                f"got {type(self.environment).__name__}"
            )

        # Validate belief
        if not isinstance(self.belief, Belief):
            raise TypeError(f"belief must be a Belief instance, got {type(self.belief).__name__}")

        # Validate policies is not empty
        if not self.policies:
            raise ValueError("policies list cannot be empty")

        # Validate all policies are Policy instances
        for idx, policy in enumerate(self.policies):
            if not isinstance(policy, Policy):
                raise TypeError(
                    f"policies[{idx}] must be a Policy instance, " f"got {type(policy).__name__}"
                )

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
        config_ids = {config.config_id for config in configs}

        if len(config_ids) != len(configs):
            raise ValueError("Duplicate configs found")

        return list(configs)


class HyperparameterOptimizationExperimentConfigCreator(ABC):
    @abstractmethod
    def _get_experiment_configs(self) -> Sequence[HyperParameterRunParams]:
        pass

    def get_experiment_configs(self) -> List[HyperParameterRunParams]:
        configs = self._get_experiment_configs()
        config_ids = {config.config_id for config in configs}

        if len(config_ids) != len(configs):
            raise ValueError("Duplicate configs found")

        return list(configs)
