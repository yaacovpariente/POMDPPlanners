from typing import TYPE_CHECKING, Optional
from pathlib import Path
from dataclasses import dataclass

from POMDPPlanners.core.simulation.hyperparameter_tuning import HyperParameterFeature, CategoricalHyperParameter, NumericalHyperParameter


if TYPE_CHECKING:
    from POMDPPlanners.core.belief import Belief
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.policy import Policy

@dataclass(frozen=True)
class EnvironmentRunParams:
    environment: 'Environment'
    belief: 'Belief'
    policies: list['Policy']
    num_episodes: int
    num_steps: int


@dataclass(frozen=True)
class HyperParameterRunParams:
    environment: 'Environment'
    belief: 'Belief'
    policies: list['Policy']
    num_episodes: int
    num_steps: int
    hyper_parameters: list[HyperParameterFeature]
    b_trials: int
    n_jobs: int
    cache_visualizations: bool
    cache_dir_path: Path
    experiment_name: str
    mlflow_tracking_uri: Optional[str] = None
    mlruns_path: Optional[Path] = None