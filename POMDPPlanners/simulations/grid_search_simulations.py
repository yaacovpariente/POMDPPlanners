"""Grid (linear) search hyperparameter optimizer.

Sibling of :mod:`POMDPPlanners.simulations.hyper_parameter_tuning_simulations`
that swaps Optuna's adaptive search for an exhaustive Cartesian-product
sweep over user-supplied value lists. Both optimizers share the
:class:`BaseHyperParameterOptimizer` infrastructure (cache directory,
MLflow setup and logging, final-evaluation rerun, artifact serialization)
— they differ only in how :class:`HyperParameterRunParams` is mapped onto
:class:`SimulationTask` instances.

For each input config, this module emits **one**
:class:`GridSearchTuningTask` that internally fans out
``len(combos) * num_episodes`` per-episode tasks through its own
:class:`JoblibTaskManager`, achieving full FLAT parallelism over both
the grid axis and the episode axis.

Usable hyperparameter types are :class:`CategoricalHyperParameter` and
:class:`NumericalGridSpec`. A bare :class:`NumericalHyperParameter` is
rejected — grid search needs an explicit value list.
"""

from typing import List, Tuple

from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    CategoricalHyperParameter,
    HyperParameterRunParams,
    NumericalGridSpec,
)
from POMDPPlanners.core.simulation.tasks import SimulationTask
from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import (
    BaseHyperParameterOptimizer,
)
from POMDPPlanners.simulations.simulations_deployment.tasks.grid_search_tuning_task import (
    GridSearchTuningTask,
)
from POMDPPlanners.utils.logger import get_logger

logger = get_logger(__name__)


class GridSearchOptimizer(BaseHyperParameterOptimizer):
    """Exhaustive grid (linear) search over an explicit set of values per axis.

    Drop-in alternative to :class:`HyperParameterOptimizer`: same
    constructor signature (inherited), same ``optimize(configs)`` contract,
    same MLflow logging shape, same
    :class:`~POMDPPlanners.core.simulation.hyperparameter_tuning.OptimizedPolicyResult`
    output. Choose this optimizer when you want to evaluate every
    combination of a small number of value lists rather than letting
    Optuna's TPE/CMA-ES sample adaptively.

    Each input :class:`HyperParameterRunParams` produces one
    :class:`GridSearchTuningTask`, which internally runs
    ``len(combinations) * num_episodes`` episodes in a single parallel
    batch (FLAT parallelism). Combinations share matched-pairs episode
    seeds so head-to-head comparisons have lower variance.
    """

    def _create_tasks(
        self, configs: List[HyperParameterRunParams]
    ) -> Tuple[List[SimulationTask], List[str]]:
        tasks: List[SimulationTask] = []
        task_identifiers: List[str] = []
        for config in configs:
            self._validate_grid_hyperparameters(config)
            task = GridSearchTuningTask(
                environment=config.environment,
                belief=config.belief,
                policy_cls=config.hyper_param_planner_config.policy_cls,
                hyper_parameters=config.hyper_param_planner_config.hyper_parameters,  # type: ignore[arg-type]
                constant_parameters=config.hyper_param_planner_config.constant_parameters,
                num_episodes=config.num_episodes,
                num_steps=config.num_steps,
                parameters_to_optimize=config.parameters_to_optimize,
                cache_dir=self.cache_dir_path,
                n_jobs=self.n_jobs,
                confidence_interval_level=self.confidence_interval_level,
                alpha=self.alpha,
            )
            tasks.append(task)
            task_identifiers.append(task.get_config_id())
        return tasks, task_identifiers

    @staticmethod
    def _validate_grid_hyperparameters(config: HyperParameterRunParams) -> None:
        """Reject a config whose hyperparameters aren't grid-compatible.

        The shared :class:`HyperParamPlannerConfig` accepts both Optuna and
        grid hyperparameter types so the same config dataclass can target
        either optimizer; the per-optimizer validation lives here.
        """
        for param in config.hyper_param_planner_config.hyper_parameters:
            if not isinstance(param, (CategoricalHyperParameter, NumericalGridSpec)):
                raise TypeError(
                    f"GridSearchOptimizer requires CategoricalHyperParameter or "
                    f"NumericalGridSpec for every hyperparameter; "
                    f"got {type(param).__name__} for {param.name!r}. "
                    "Switch to NumericalGridSpec(low, high, n_points, name, scale) "
                    "or use HyperParameterOptimizer (Optuna) instead."
                )
