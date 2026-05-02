"""Grid (linear) search hyperparameter tuning task.

Mirrors the structure of :class:`HyperParameterTuningSimulationTask` but
replaces Optuna's adaptive search with a deterministic Cartesian-product
sweep over user-supplied value lists. Every ``(combination, episode_index)``
pair runs as one independent :class:`EpisodeSimulationTask` submitted to an
inner :class:`JoblibTaskManager` in a single batch — full FLAT parallelism.

Episodes are seeded **matched-pairs** style: ``seed = base_seed +
episode_index``, identical across combinations for a given ``episode_index``.
This shares trajectory noise across combos so head-to-head comparisons
have lower variance.
"""

import tempfile
import time
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import (
    History,
    SimulationTask,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    GridHyperParameterFeature,
    HyperParameterOptimizationDirection,
    OptimizedPolicyResult,
    compute_pareto_scores,
    expand_grid,
)
from POMDPPlanners.simulations.simulation_statistics import (
    compute_statistics_environment_policy_pair,
)
from POMDPPlanners.simulations.simulations_deployment.cache_dbs import DiskCacheDB
from POMDPPlanners.simulations.simulations_deployment.task_managers import JoblibTaskManager
from POMDPPlanners.simulations.simulations_deployment.tasks.episode_simulation_task import (
    EpisodeSimulationTask,
)
from POMDPPlanners.utils.config_to_id import config_to_id
from POMDPPlanners.utils.logger import get_logger


class GridSearchTuningTask(SimulationTask):
    """Run a grid search over an explicit list of hyperparameter values.

    The Cartesian product of every axis defined in ``hyper_parameters`` is
    evaluated by simulating ``num_episodes`` matched-pairs episodes per
    combination. The combination with the best aggregated score (z-score
    normalization across the metrics named in ``parameters_to_optimize``,
    sign-flipped for ``MINIMIZE``) wins.

    Submits all ``len(combos) * num_episodes`` episode tasks to an inner
    :class:`JoblibTaskManager` in one call, so wall-clock parallelism is
    only bounded by ``n_jobs`` — the relationship between the number of
    combinations and the number of episodes per combination is irrelevant.
    """

    def __init__(
        self,
        environment: Environment,
        belief: Belief,
        policy_cls: Type[Policy],
        hyper_parameters: Sequence[GridHyperParameterFeature],
        constant_parameters: Dict[str, Any],
        num_episodes: int,
        num_steps: int,
        parameters_to_optimize: List[Tuple[str, HyperParameterOptimizationDirection]],
        cache_dir: Union[Path, str, None] = None,
        n_jobs: int = 1,
        confidence_interval_level: float = 0.95,
        alpha: float = 0.1,
        base_seed: int = 42,
        experiment_name: str = "grid_search_optimization",
    ):
        self._validate_inputs(
            environment=environment,
            belief=belief,
            policy_cls=policy_cls,
            hyper_parameters=hyper_parameters,
            constant_parameters=constant_parameters,
            num_episodes=num_episodes,
            num_steps=num_steps,
            parameters_to_optimize=parameters_to_optimize,
            n_jobs=n_jobs,
            confidence_interval_level=confidence_interval_level,
            alpha=alpha,
        )

        self.environment = environment
        self.belief = belief
        self.policy_cls = policy_cls
        self.hyper_parameters = hyper_parameters
        self.constant_parameters = constant_parameters
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.parameters_to_optimize = parameters_to_optimize
        self.cache_dir: Optional[str] = str(cache_dir) if cache_dir is not None else None
        self.n_jobs = n_jobs
        self.confidence_interval_level = confidence_interval_level
        self.alpha = alpha
        self.base_seed = base_seed
        self.experiment_name = experiment_name

        self.logger = get_logger(name=f"grid_search.{policy_cls.__name__}")
        self._last_optimization_result: Optional[OptimizedPolicyResult] = None
        self._last_optimization_metadata: Optional[Dict[str, Any]] = None

    @staticmethod
    def _validate_inputs(  # pylint: disable=too-many-branches
        environment: Environment,
        belief: Belief,
        policy_cls: Type[Policy],
        hyper_parameters: Sequence[GridHyperParameterFeature],
        constant_parameters: Dict[str, Any],
        num_episodes: int,
        num_steps: int,
        parameters_to_optimize: List[Tuple[str, HyperParameterOptimizationDirection]],
        n_jobs: int,
        confidence_interval_level: float,
        alpha: float,
    ) -> None:
        if not isinstance(environment, Environment):
            raise TypeError(f"environment must be an Environment, got {type(environment).__name__}")
        if not isinstance(belief, Belief):
            raise TypeError(f"belief must be a Belief, got {type(belief).__name__}")
        if not isinstance(policy_cls, type):
            raise TypeError(f"policy_cls must be a class, got {type(policy_cls).__name__}")
        if not isinstance(hyper_parameters, Sequence) or len(hyper_parameters) == 0:
            raise ValueError("hyper_parameters must be a non-empty sequence")
        if not isinstance(constant_parameters, dict):
            raise TypeError("constant_parameters must be a dict")
        if not isinstance(num_episodes, int) or num_episodes <= 0:
            raise ValueError(f"num_episodes must be a positive int, got {num_episodes}")
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError(f"num_steps must be a positive int, got {num_steps}")
        if not isinstance(parameters_to_optimize, list) or len(parameters_to_optimize) == 0:
            raise ValueError("parameters_to_optimize must be a non-empty list")
        if not isinstance(n_jobs, int) or (n_jobs <= 0 and n_jobs != -1):
            raise ValueError(f"n_jobs must be a positive int or -1, got {n_jobs}")
        if not 0.0 < confidence_interval_level < 1.0:
            raise ValueError(
                f"confidence_interval_level must be in (0, 1), got {confidence_interval_level}"
            )
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    def run(self) -> Optional[OptimizedPolicyResult]:
        start_time = time.time()
        try:
            combos = expand_grid(self.hyper_parameters)
            if not combos:
                raise ValueError("expand_grid returned no combinations")

            self.logger.info(
                "Starting grid search: %d combinations x %d episodes = %d episode runs",
                len(combos),
                self.num_episodes,
                len(combos) * self.num_episodes,
            )

            policies = self._build_policies(combos)
            histories_by_combo = self._run_all_episodes(policies)
            metric_table = self._aggregate_combo_metrics(histories_by_combo)
            best_combo_idx, scores = self._select_best_combo(metric_table)

            optimization_time = time.time() - start_time
            result = self._build_result(
                winning_combo=combos[best_combo_idx],
                winning_policy=policies[best_combo_idx],
                winning_metrics=metric_table[best_combo_idx],
            )
            self._record_metadata(
                best_combo_idx=best_combo_idx,
                metric_table=metric_table,
                scores=scores,
                n_combos=len(combos),
                optimization_time=optimization_time,
            )
            self._last_optimization_result = result
            return result

        except Exception:  # pylint: disable=broad-exception-caught
            self.logger.exception("Grid search task failed")
            return None

    def _build_policies(self, combos: List[Dict[str, Any]]) -> List[Policy]:
        policies: List[Policy] = []
        for combo in combos:
            kwargs: Dict[str, Any] = {"environment": self.environment}
            kwargs.update(self.constant_parameters)
            kwargs.update(combo)
            policies.append(self.policy_cls(**kwargs))
        return policies

    def _run_all_episodes(self, policies: List[Policy]) -> Dict[int, List[History]]:
        tasks: List[SimulationTask] = []
        identifiers: List[Tuple[int, int]] = []
        for combo_idx, policy in enumerate(policies):
            for episode_idx in range(self.num_episodes):
                tasks.append(
                    EpisodeSimulationTask(
                        environment=self.environment,
                        policy=policy,
                        initial_belief=deepcopy(self.belief),
                        num_steps=self.num_steps,
                        episode_id=episode_idx,
                        seed=self.base_seed + episode_idx,
                        episode_number=episode_idx,
                        cache_dir=self.cache_dir,
                        debug=False,
                        console_output=False,
                    )
                )
                identifiers.append((combo_idx, episode_idx))

        with ExitStack() as stack:
            base_cache_dir = self._resolve_inner_cache_dir(stack)
            cache_db = DiskCacheDB(cache_dir=str(Path(base_cache_dir) / "episodes_db"))
            manager = stack.enter_context(
                JoblibTaskManager(
                    cache_db=cache_db,
                    n_jobs=self.n_jobs,
                    cache_dir=str(Path(base_cache_dir) / "episodes_joblib"),
                    console_output=False,
                    no_logs=True,
                )
            )
            histories, returned_identifiers = manager.run_tasks(tasks, identifiers)

        grouped: Dict[int, List[History]] = {idx: [] for idx in range(len(policies))}
        for identifier, history in zip(returned_identifiers, histories):
            combo_idx, _ = identifier
            grouped[combo_idx].append(history)
        return grouped

    def _resolve_inner_cache_dir(self, stack: ExitStack) -> str:
        """Return a directory for the inner task manager.

        Uses ``self.cache_dir`` when configured (cross-run caching) or a
        :class:`tempfile.TemporaryDirectory` registered with ``stack`` when
        not (single-run, auto-cleaned).
        """
        if self.cache_dir is not None:
            target = Path(self.cache_dir) / "grid_search" / self.get_config_id()
            target.mkdir(parents=True, exist_ok=True)
            return str(target)
        # Lifetime tied to the caller's ExitStack — a `with` here would close
        # the directory before returning its path.
        # pylint: disable-next=consider-using-with
        tmp = stack.enter_context(tempfile.TemporaryDirectory(prefix="grid_search_"))
        return tmp

    def _aggregate_combo_metrics(
        self, histories_by_combo: Dict[int, List[History]]
    ) -> Dict[int, Dict[str, float]]:
        required = {name for name, _ in self.parameters_to_optimize}
        metric_table: Dict[int, Dict[str, float]] = {}
        for combo_idx, histories in histories_by_combo.items():
            if not histories:
                self.logger.warning(
                    "Combo %d had no successful episodes — excluding from scoring", combo_idx
                )
                continue
            statistics = compute_statistics_environment_policy_pair(
                env=self.environment,
                histories=histories,
                alpha=self.alpha,
                confidence_interval_level=self.confidence_interval_level,
            )
            row: Dict[str, float] = {
                metric.name: metric.value for metric in statistics if metric.name in required
            }
            missing = required - set(row.keys())
            if missing:
                raise ValueError(
                    f"Combo {combo_idx} missing required metrics {sorted(missing)} "
                    f"in computed statistics"
                )
            metric_table[combo_idx] = row

        if not metric_table:
            raise ValueError("No combination produced any successful episode")
        return metric_table

    def _select_best_combo(
        self, metric_table: Dict[int, Dict[str, float]]
    ) -> Tuple[int, Dict[int, float]]:
        scores = compute_pareto_scores(metric_table, self.parameters_to_optimize)
        best_combo_idx = max(scores, key=lambda k: scores[k])
        return best_combo_idx, dict(scores)

    def _build_result(
        self,
        winning_combo: Dict[str, Any],
        winning_policy: Policy,
        winning_metrics: Dict[str, float],
    ) -> OptimizedPolicyResult:
        return OptimizedPolicyResult(
            environment=self.environment,
            policy=winning_policy,
            chosen_hyper_parameters=dict(winning_combo),
            num_episodes=self.num_episodes,
            num_steps=self.num_steps,
            parameters_to_optimize=list(self.parameters_to_optimize),
            optimized_metric_values=dict(winning_metrics),
        )

    def _record_metadata(
        self,
        best_combo_idx: int,
        metric_table: Dict[int, Dict[str, float]],
        scores: Dict[int, float],
        n_combos: int,
        optimization_time: float,
    ) -> None:
        best_metrics = metric_table[best_combo_idx]
        self._last_optimization_metadata = {
            "best_pareto_score": float(scores[best_combo_idx]),
            "best_trial_metrics": dict(best_metrics),
            "best_trial_number": int(best_combo_idx),
            "best_trial_statistics": [],
            "n_trials": int(n_combos),
            "optimization_time": float(optimization_time),
            "config_id": self.get_config_id(),
            "all_pareto_scores": {int(k): float(v) for k, v in scores.items()},
        }

    def get_optimization_metadata(self) -> Optional[Dict[str, Any]]:
        return self._last_optimization_metadata

    def get_config_id(self) -> str:
        return config_to_id(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "GridSearchTuningTask",
            "environment": self.environment.config_id,
            "belief": self.belief.config_id,
            "policy_cls": self.policy_cls.__name__,
            "hyper_parameters": [param.id() for param in self.hyper_parameters],
            "constant_parameters": self.constant_parameters,
            "num_episodes": self.num_episodes,
            "num_steps": self.num_steps,
            "parameters_to_optimize": [
                (name, direction.value) for name, direction in self.parameters_to_optimize
            ],
            "base_seed": self.base_seed,
        }

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("logger", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        vars(self).update(state)
        self.logger = get_logger(name=f"grid_search.{self.policy_cls.__name__}")
