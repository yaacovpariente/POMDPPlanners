# SPDX-License-Identifier: MIT

"""Abstract base class for POMDP simulation frameworks."""

import cProfile
import io
import pstats
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import mlflow
import pandas as pd

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import (
    EnvironmentRunParams,
    ExperimentVisualizer,
    MetricValue,
    SimulationTask,
)
from POMDPPlanners.simulations.simulation_statistics import (
    metrics_dict_to_dataframe,
)
from POMDPPlanners.simulations.simulations_deployment.run_progress import (
    NotificationConfig,
    Notifier,
    build_notifier,
    install_signal_handlers,
)
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
    TaskManagerConfig,
)
from POMDPPlanners.simulations.simulations_deployment.tasks import (
    EnvironmentVisualizationTask,
)
from POMDPPlanners.simulations.simulator.episode_returns_visualizer import (
    EpisodeReturnsVisualizer,
)
from POMDPPlanners.utils.logger import cleanup_all_loggers, get_logger
from POMDPPlanners.utils.visualization import (
    plot_policies_comparison_on_environment,
)


def _validate_environment_policy_comparison_parameters(
    environment_run_params: List[EnvironmentRunParams],
    alpha: float,
    confidence_interval_level: float,
    n_jobs: int,
    cache_visualizations: bool,
) -> None:
    """Validate input parameters for environment-policy comparison methods.

    Args:
        environment_run_params: List of environment run parameters
        alpha: Alpha value for risk metrics (must be between 0 and 1)
        confidence_interval_level: Confidence level for statistics (must be between 0 and 1, exclusive)
        n_jobs: Number of parallel jobs (positive integer or -1)
        cache_visualizations: Whether to cache visualizations

    Raises:
        ValueError: If any parameter has invalid type or value
    """
    if not isinstance(environment_run_params, list):
        raise ValueError("environment_run_params must be a list")
    if not all(isinstance(param, EnvironmentRunParams) for param in environment_run_params):
        raise ValueError("All elements in environment_run_params must be EnvironmentRunParams")
    if not isinstance(alpha, float):
        raise ValueError("alpha must be a float")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if not isinstance(confidence_interval_level, float):
        raise ValueError("confidence_interval_level must be a float")
    if not 0 < confidence_interval_level < 1:
        raise ValueError("confidence_interval_level must be between 0 and 1")
    if not isinstance(n_jobs, int):
        raise ValueError("n_jobs must be an integer")
    if not (n_jobs > 0 or n_jobs == -1):
        raise ValueError("n_jobs must be a positive integer or -1")
    if not isinstance(cache_visualizations, bool):
        raise ValueError("cache_visualizations must be a boolean")


def _build_policies_by_env(
    environment_run_params: List[EnvironmentRunParams],
) -> Dict[str, List[Policy]]:
    """Build a dict mapping environment name to its policies.

    Used to assemble visualization tasks without capturing the simulator's
    ``self`` in a closure (which would drag the live task manager into the
    pickle stream sent to workers).
    """
    return {params.environment.name: list(params.policies) for params in environment_run_params}


class BaseSimulator(ABC):
    """Abstract base class for POMDP simulation frameworks.

    This class provides a foundation for running large-scale POMDP simulations with
    support for parallel execution, experiment tracking, and result analysis. It handles
    the orchestration of multiple environment-policy combinations and provides infrastructure
    for caching, logging, and visualization.

    Key Features:
    - Parallel simulation execution using configurable task managers (Dask, Joblib, PBS)
    - MLflow integration for experiment tracking and artifact logging
    - Statistical analysis with confidence intervals and risk metrics
    - Pluggable visualization strategy via :class:`ExperimentVisualizer`
    - Flexible caching strategies and performance profiling

    Attributes:
        cache_dir_path: Path for storing simulation results and artifacts
        experiment_name: Name of the MLflow experiment for tracking
        debug: Whether debug logging is enabled
        task_manager: Task manager instance for handling parallel execution
        visualizer: Experiment-level visualization strategy

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses must implement the abstract methods _create_simulation_tasks
        and _compute_metrics.
    """

    def __init__(
        self,
        task_manager_config: TaskManagerConfig,
        cache_dir_path: Optional[Path] = None,
        experiment_name: str = "POMDP_Planning_Comparison",
        debug: bool = False,
        enable_profiling: bool = False,
        profiling_output_limit: int = 50,
        use_queue_logger: bool = False,
        console_output: bool = True,
        no_logs: bool = False,
        visualizer: Optional[ExperimentVisualizer] = None,
        notification_config: Optional[NotificationConfig] = None,
    ):
        """Initialize the simulator.

        Args:
            task_manager_config: Configuration object for task manager creation
            cache_dir_path: Path to store results and log files. Set to None to disable file logging.
            experiment_name: Name of the MLFlow experiment
            debug: Whether to enable debug logging
            enable_profiling: Whether to enable cProfile profiling
            profiling_output_limit: Maximum number of functions to show in profiling output (default: 50)
            use_queue_logger: Whether to use queue-based logging
            console_output: Whether to print logs to console (default: True)
            no_logs: Whether to disable all logging (default: False)
            visualizer: Strategy for rendering aggregated experiment artifacts after
                each comparison run. Defaults to :class:`EpisodeReturnsVisualizer`.
            notification_config: Optional :class:`NotificationConfig` controlling
                Slack + progress-DB notifications. When ``None``, defaults to
                :meth:`NotificationConfig.disabled`, meaning no notifications
                will fire. The ``SimulationsAPI`` layer is responsible for
                building a config from env vars and threading it in; direct
                callers (tests, workflows) get silent behaviour by default.
        """
        self.cache_dir_path = cache_dir_path
        self.experiment_name = experiment_name
        self.debug = debug
        self.enable_profiling = enable_profiling
        self.profiling_output_limit = profiling_output_limit
        self.profiler: Optional[cProfile.Profile] = None
        self.use_queue_logger = use_queue_logger
        self.visualizer: ExperimentVisualizer = visualizer or EpisodeReturnsVisualizer()

        self.logger = get_logger(
            name=f"simulator.{experiment_name}",
            debug=debug,
            output_dir=cache_dir_path if not no_logs else None,
            use_queue=use_queue_logger,
            console_output=console_output if not no_logs else False,
        )

        cache_dir = str(cache_dir_path) if cache_dir_path else None
        self.task_manager = task_manager_config.create_task_manager(cache_dir=cache_dir)

        if cache_dir_path is not None:
            self._setup_mlflow_tracking()

        self.notification_config: NotificationConfig = (
            notification_config
            if notification_config is not None
            else NotificationConfig.disabled()
        )
        self._run_id: str = uuid.uuid4().hex
        self._notifier: Notifier = build_notifier(
            experiment_name=self.experiment_name,
            config=self.notification_config,
            logger=self.logger,
        )
        self._uninstall_signals: Callable[[], None] = install_signal_handlers(
            self._notifier, self._run_id
        )
        self.task_manager.set_progress_callback(
            lambda: self._notifier.episode_completed(self._run_id)
        )

    def __getstate__(self) -> Dict:
        # The signal-handler uninstall callable is a closure created by
        # :func:`install_signal_handlers` and is therefore not picklable.
        # Drop it on pickle; restore as a no-op on unpickle. Unpickled
        # simulators are not the live process that installed the handlers,
        # so they have nothing to uninstall.
        state = self.__dict__.copy()
        state["_uninstall_signals"] = None
        return state

    def __setstate__(self, state: Dict) -> None:
        for key, value in state.items():
            setattr(self, key, value)
        if getattr(self, "_uninstall_signals", None) is None:
            self._uninstall_signals = lambda: None

    def _setup_mlflow_tracking(self) -> None:
        """Configure MLFlow tracking."""
        if self.cache_dir_path is None:
            self.cache_dir_path = Path.cwd()

        self.cache_dir_path.mkdir(parents=True, exist_ok=True)

        mlruns_path = self.cache_dir_path / "mlruns"
        mlruns_path.mkdir(parents=True, exist_ok=True)

        trash_path = mlruns_path / ".trash"
        trash_path.mkdir(parents=True, exist_ok=True)

        tracking_uri = f"file://{mlruns_path.absolute()}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        self.task_manager.__exit__(exc_type, exc_val, exc_tb)

        try:
            if exc_type is None:
                self._notifier.run_finished(self._run_id)
            else:
                self._notifier.run_failed(self._run_id, f"{exc_type.__name__}: {exc_val}")
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        try:
            self._uninstall_signals()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        try:
            cleanup_all_loggers()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    def compare_multiple_environments_policies(
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float = 0.1,
        confidence_interval_level: float = 0.95,
        n_jobs: int = 1,
        cache_visualizations: bool = True,
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Compare multiple policies on multiple environments with optional profiling.

        Args:
            environment_run_params: List of environment run parameters
            alpha: Alpha value for statistics computation
            confidence_interval_level: Confidence level for statistics
            n_jobs: Number of parallel jobs
            cache_visualizations: Whether to cache per-episode environment visualizations

        Returns:
            Tuple of results dictionary and statistics DataFrame
        """
        self._notifier.run_started(
            self._run_id,
            metadata={
                "num_environments": len(environment_run_params),
                "environment_names": [params.environment.name for params in environment_run_params],
                "n_jobs": n_jobs,
            },
        )

        if self.enable_profiling:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            self.logger.info("Profiling enabled - starting cProfile")

        try:
            return self._compare_multiple_environments_policies(
                environment_run_params=environment_run_params,
                alpha=alpha,
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs,
                cache_visualizations=cache_visualizations,
            )
        finally:
            if self.enable_profiling and self.profiler:
                self.profiler.disable()
                self._log_profiling_results()

    def _log_profiling_results(self) -> None:
        if not self.profiler:
            return

        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(self.profiling_output_limit)

        profiling_output = s.getvalue()
        self.logger.info("Profiling results (all functions):\n%s", profiling_output)

        if self.cache_dir_path:
            profiling_file = self.cache_dir_path / "profiling_results.txt"
            with open(profiling_file, "w", encoding="utf-8") as f:
                ps = pstats.Stats(self.profiler, stream=f).sort_stats("cumulative")
                ps.print_stats(self.profiling_output_limit)
            self.logger.info("Detailed profiling results saved to: %s", profiling_file)

    def cleanup_mlflow_runs(self) -> None:
        """Clean up any active MLflow runs.

        Ensures any active MLflow runs are properly ended. Useful for cleanup
        in tests or when the simulator is used outside of a context manager.
        """
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    def _compare_multiple_environments_policies(
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float = 0.1,
        confidence_interval_level: float = 0.95,
        n_jobs: int = 1,
        cache_visualizations: bool = True,
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        _validate_environment_policy_comparison_parameters(
            environment_run_params=environment_run_params,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
            cache_visualizations=cache_visualizations,
        )

        self._log_comparison_overview(environment_run_params)

        active_run = mlflow.active_run()
        nested = active_run is not None
        with mlflow.start_run(run_name="environment_policy_comparison", nested=nested):
            self._log_mlflow_comparison_parameters(
                alpha=alpha,
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs,
                cache_visualizations=cache_visualizations,
                environment_run_params=environment_run_params,
            )

            results, metrics = self._run_simulations_and_compute_metrics(
                environment_run_params=environment_run_params,
                alpha=alpha,
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs,
            )

            merged_df = self._create_statistics_and_policy_dataframes(
                metrics=metrics, environment_run_params=environment_run_params
            )

            self._log_comparison_data_to_mlflow(merged_df, environment_run_params)

            self._create_and_log_environment_visualizations(
                results=results,
                environment_run_params=environment_run_params,
                cache_visualizations=cache_visualizations,
                n_jobs=n_jobs,
            )

        return results, merged_df

    def _log_comparison_overview(self, environment_run_params: List[EnvironmentRunParams]) -> None:
        env_algo_info = "\n".join(
            [
                f"Environment: {run_params.environment.name} - Algorithms: {[p.name for p in run_params.policies]}"
                for run_params in environment_run_params
            ]
        )
        self.logger.info("Running comparison with:\n%s", env_algo_info)

    def _log_mlflow_comparison_parameters(
        self,
        alpha: float,
        confidence_interval_level: float,
        n_jobs: int,
        cache_visualizations: bool,
        environment_run_params: List[EnvironmentRunParams],
    ) -> None:
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("confidence_interval_level", confidence_interval_level)
        mlflow.log_param("n_jobs", n_jobs)
        mlflow.log_param("cache_visualizations", cache_visualizations)
        mlflow.log_param("num_environments", len(environment_run_params))

        for i, params in enumerate(environment_run_params):
            env_prefix = f"env_{i}"
            mlflow.log_param(f"{env_prefix}_name", params.environment.name)
            mlflow.log_param(f"{env_prefix}_num_episodes", params.num_episodes)
            mlflow.log_param(f"{env_prefix}_num_steps", params.num_steps)
            mlflow.log_param(f"{env_prefix}_num_policies", len(params.policies))

            for j, policy in enumerate(params.policies):
                policy_prefix = f"{env_prefix}_policy_{j}"
                mlflow.log_param(f"{policy_prefix}_name", policy.name)
                mlflow.log_param(f"{policy_prefix}_type", policy.__class__.__name__)

    def _run_simulations_and_compute_metrics(
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float,
        n_jobs: int,
    ) -> Tuple[Dict[str, Dict[str, list]], Dict[str, Dict[str, List[MetricValue]]]]:
        results: Dict[str, Dict[str, list]] = (
            self.simulate_multiple_environments_and_policies_parallel(
                environment_run_params=environment_run_params,
                alpha=alpha,
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs,
            )
        )

        metrics: Dict[str, Dict[str, List[MetricValue]]] = self._compute_metrics(
            results=results,
            environment_run_params=environment_run_params,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
        )

        self._log_metrics_to_mlflow(metrics)

        return results, metrics

    def _create_statistics_and_policy_dataframes(
        self,
        metrics: Dict[str, Dict[str, List[MetricValue]]],
        environment_run_params: List[EnvironmentRunParams],
    ) -> pd.DataFrame:
        statistics_df = metrics_dict_to_dataframe(metrics_dict=metrics)

        policy_configs_df = self._create_policy_configurations_df(
            [
                (run_params.environment, run_params.belief, run_params.policies)
                for run_params in environment_run_params
            ]
        )
        return pd.merge(statistics_df, policy_configs_df, on=["environment", "policy"])

    def _log_comparison_data_to_mlflow(
        self,
        merged_df: pd.DataFrame,
        environment_run_params: List[EnvironmentRunParams],
    ) -> None:
        mlflow.log_table(merged_df, "statistics/comparison_results.json")

        policy_configs_df = self._create_policy_configurations_df(
            [
                (run_params.environment, run_params.belief, run_params.policies)
                for run_params in environment_run_params
            ]
        )
        mlflow.log_table(policy_configs_df, "statistics/policy_configurations.json")

    def _generate_and_log_comparison_plots(
        self, metrics: Dict[str, Dict[str, List[MetricValue]]]
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_plots_dir_str:
            temp_plots_dir = Path(temp_plots_dir_str)
            plot_policies_comparison_on_environment(
                metrics_dict=metrics, cache_dir_path=temp_plots_dir
            )
            for item in temp_plots_dir.iterdir():
                mlflow.log_artifact(str(item), "policy_comparison_plots")

    def _create_and_log_environment_visualizations(
        self,
        results: Dict[str, Dict[str, list]],
        environment_run_params: List[EnvironmentRunParams],
        cache_visualizations: bool,
        n_jobs: int,  # pylint: disable=unused-argument
    ) -> None:
        """Build per-environment visualization tasks and dispatch via the task manager.

        Each ``EnvironmentVisualizationTask`` runs on a worker, renders into a
        worker-local scratch directory, and returns a ``Dict[str, bytes]``
        keyed by POSIX-style relative path. The parent process materializes
        those bytes locally and uploads them to MLflow. No filesystem path
        crosses the wire — workers and clients can run on different OSes.
        """
        if not results:
            return

        viz_cache_dir = self.cache_dir_path or (Path.cwd() / ".cache")
        viz_cache_dir.mkdir(parents=True, exist_ok=True)

        env_lookup = {
            params.environment.name: params.environment for params in environment_run_params
        }
        policies_by_env = _build_policies_by_env(environment_run_params)

        tasks: List[SimulationTask] = []
        identifiers: List[str] = []
        for env_name, policy_results in results.items():
            policies_for_env = [p for p in policies_by_env[env_name] if p.name in policy_results]
            tasks.append(
                EnvironmentVisualizationTask(
                    visualizer=self.visualizer,
                    env_name=env_name,
                    environment=env_lookup[env_name],
                    policy_results=policy_results,
                    policies=policies_for_env,
                    cache_visualizations=cache_visualizations,
                )
            )
            identifiers.append(env_name)

        artifact_bundles, returned_env_names = self.task_manager.run_tasks(tasks, identifiers)

        for env_name, artifacts in zip(returned_env_names, artifact_bundles):
            if not artifacts:
                continue
            env_dir = viz_cache_dir / "viz_artifacts" / env_name
            env_dir.mkdir(parents=True, exist_ok=True)
            for rel_key, data in artifacts.items():
                target = env_dir / Path(rel_key)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            for item in env_dir.iterdir():
                mlflow.log_artifact(str(item), env_name)

        if self.cache_dir_path:
            viz_artifacts_dir = self.cache_dir_path / "viz_artifacts"
            if viz_artifacts_dir.exists():
                shutil.rmtree(viz_artifacts_dir, ignore_errors=True)

    def _create_policy_configurations_df(
        self,
        environment_belief_policy_tuples: List[Tuple[Environment, Belief, Sequence[Policy]]],
    ) -> pd.DataFrame:
        """Create a DataFrame containing policy configurations for all environment-policy pairs."""
        policy_configs = []
        for env, _, policies in environment_belief_policy_tuples:
            for policy in policies:
                policy_params = {
                    key: value
                    for key, value in policy.__dict__.items()
                    if isinstance(value, (int, float, str, bool))
                }
                row_data = {
                    "environment": env.name,
                    "policy": policy.name,
                    "policy_type": policy.__class__.__name__,
                    **policy_params,
                }
                policy_configs.append(row_data)

        return pd.DataFrame(policy_configs)

    def _organize_simulation_results(
        self,
        results_list: list,
        environment_belief_policy_tuples: List[Tuple[Environment, Belief, Sequence[Policy]]],
        num_episodes: int,
        task_identifiers: List[Tuple[str, str]],
    ) -> Dict[str, Dict[str, list]]:
        """Organize simulation results by environment and policy using task identifiers."""
        self.logger.info("Organizing results by environment and policy")

        results: Dict[str, Dict[str, list]] = {}
        for env, _, policies in environment_belief_policy_tuples:
            results[env.name] = {policy.name: [] for policy in policies}

        for result, (env_name, policy_name) in zip(results_list, task_identifiers):
            if env_name in results and policy_name in results[env_name]:
                results[env_name][policy_name].append(result)
                self.logger.debug("Added result for %s with %s", env_name, policy_name)
            else:
                self.logger.warning(
                    "Received result for unknown env-policy pair: %s, %s",
                    env_name,
                    policy_name,
                )

        for env, _, policies in environment_belief_policy_tuples:
            for policy in policies:
                result = results[env.name][policy.name]
                if len(result) != num_episodes:
                    self.logger.warning(
                        "Policy %s in environment %s has %s results, expected %s",
                        policy.name,
                        env.name,
                        len(result),
                        num_episodes,
                    )

        return results

    def simulate_multiple_environments_and_policies_parallel(
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float = 0.95,
        n_jobs: int = 1,
    ) -> Dict[str, Dict[str, list]]:
        """Simulate multiple policies on multiple environments in parallel."""
        self._validate_parallel_simulation_inputs(
            environment_run_params=environment_run_params,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
        )

        simulation_tasks, task_identifiers = self._create_simulation_tasks(
            environment_run_params=environment_run_params,
        )

        results_list, task_identifiers = self.task_manager.run_tasks(
            simulation_tasks, task_identifiers
        )

        if len(results_list) != len(task_identifiers):
            raise ValueError("results_list and task_identifiers must have the same length")
        if len(results_list) == 0:
            raise ValueError("All tasks failed.")

        return self._organize_simulation_results(
            results_list=results_list,
            environment_belief_policy_tuples=[
                (params.environment, params.belief, params.policies)
                for params in environment_run_params
            ],
            num_episodes=environment_run_params[0].num_episodes,
            task_identifiers=task_identifiers,
        )

    def _validate_parallel_simulation_inputs(  # pylint: disable=too-many-branches
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float,
        n_jobs: int,
    ) -> None:
        if not isinstance(environment_run_params, list):
            raise ValueError("environment_run_params must be a list")
        if len(environment_run_params) == 0:
            raise ValueError("environment_run_params cannot be empty")
        for params in environment_run_params:
            if not isinstance(params, EnvironmentRunParams):
                raise ValueError(f"Expected EnvironmentRunParams, got {type(params)}")
            if not isinstance(params.environment, Environment):
                raise ValueError(f"Expected Environment, got {type(params.environment)}")
            if not isinstance(params.belief, Belief):
                raise ValueError(f"Expected Belief, got {type(params.belief)}")
            if not isinstance(params.policies, list):
                raise ValueError(f"Expected list of policies, got {type(params.policies)}")
            if len(params.policies) == 0:
                raise ValueError("Policy list cannot be empty")
            for policy in params.policies:
                if not isinstance(policy, Policy):
                    raise ValueError(f"Expected Policy, got {type(policy)}")
            if not (isinstance(params.num_episodes, int) and params.num_episodes > 0):
                raise ValueError("num_episodes must be a positive integer")
            if not (isinstance(params.num_steps, int) and params.num_steps > 0):
                raise ValueError("num_steps must be a positive integer")
        env_names = [params.environment.name for params in environment_run_params]
        if len(env_names) != len(set(env_names)):
            raise ValueError("All environments must have unique names")
        all_policies = [policy for params in environment_run_params for policy in params.policies]
        policy_names = [policy.name for policy in all_policies]
        if len(policy_names) != len(set(policy_names)):
            raise ValueError("All policies must have unique names")
        if not isinstance(alpha, float):
            raise ValueError("alpha must be a float")
        if not isinstance(confidence_interval_level, float):
            raise ValueError("confidence_interval_level must be a float")
        if not 0 <= confidence_interval_level <= 1:
            raise ValueError("confidence_interval_level must be between 0 and 1")
        if not (isinstance(n_jobs, int) and (n_jobs > 0 or n_jobs == -1)):
            raise ValueError("n_jobs must be a positive integer or -1")

    @abstractmethod
    def _create_simulation_tasks(
        self,
        environment_run_params: List[EnvironmentRunParams],
    ) -> Tuple[List[SimulationTask], List[Tuple[str, str]]]:
        """Create list of simulation tasks with deterministic ordering."""

    @abstractmethod
    def _compute_metrics(
        self,
        results: Dict[str, Dict[str, list]],
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float,
    ) -> Dict[str, Dict[str, List[MetricValue]]]:
        """Compute metrics for the simulation results."""

    def _log_metrics_to_mlflow(self, metrics: Dict[str, Dict[str, List[MetricValue]]]) -> None:
        """Log all metrics to MLflow for tracking and comparison."""
        self.logger.info("Logging metrics to MLflow")

        for env_name, policy_metrics_dict in metrics.items():
            for policy_name, metric_list in policy_metrics_dict.items():
                metric_prefix = f"{env_name}_{policy_name}"

                for metric in metric_list:
                    mlflow.log_metric(f"{metric_prefix}_{metric.name}", metric.value)
                    mlflow.log_metric(
                        f"{metric_prefix}_{metric.name}_ci_lower",
                        metric.lower_confidence_bound,
                    )
                    mlflow.log_metric(
                        f"{metric_prefix}_{metric.name}_ci_upper",
                        metric.upper_confidence_bound,
                    )
                    ci_width = metric.upper_confidence_bound - metric.lower_confidence_bound
                    mlflow.log_metric(f"{metric_prefix}_{metric.name}_ci_width", ci_width)

        self.logger.info("Logged metrics for %s environments", len(metrics))
