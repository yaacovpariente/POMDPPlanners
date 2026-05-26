# SPDX-License-Identifier: MIT

"""Hyperparameter optimization module for POMDP policies.

This module provides tools for optimizing hyperparameters of POMDP policies using
Optuna optimization framework with MLFlow experiment tracking. It supports batch
optimization of multiple environment-policy configurations with comprehensive
experiment tracking and result analysis.

The module integrates with the POMDPPlanners simulation framework to run parallel
episodes and compute performance statistics for hyperparameter evaluation.

Key Features:
    - Advanced optimization algorithms via w (TPE, CMA-ES, etc.)
    - Parallel episode execution with caching support via JoblibTaskManager
    - Comprehensive MLFlow experiment tracking and visualization
    - Support for both categorical and numerical hyperparameters
    - Batch optimization of multiple configurations
    - Statistical analysis with confidence intervals
    - Automatic MLflow logging of parameters, metrics, and artifacts
    - Task-based execution architecture for scalability

Classes:
    HyperParameterOptimizer: Main class for conducting hyperparameter optimization studies

Type Aliases:
    HyperParameterFeature: Union type for categorical or numerical hyperparameter definitions

Example:
    Setting up hyperparameter optimization components:

    >>> from pathlib import Path
    >>> from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import (
    ...     HyperParameterOptimizer
    ... )
    >>> from POMDPPlanners.core.simulation import (
    ...     NumericalHyperParameter, CategoricalHyperParameter
    ... )
    >>> from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    ...     HyperParameterRunParams, HyperParameterOptimizationDirection
    ... )
    >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
    >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
    >>> from POMDPPlanners.core.belief import get_initial_belief

    >>> # Set up optimization
    >>> import tempfile
    >>> import shutil
    >>> cache_path = Path(tempfile.mkdtemp())
    >>> optimizer = HyperParameterOptimizer(
    ...     cache_dir_path=cache_path,
    ...     experiment_name="POMCP_Tiger_Test",
    ...     n_jobs=1,                      # Single job for testing
    ...     confidence_interval_level=0.95,
    ...     alpha=0.05
    ... )
    >>> optimizer.experiment_name
    'POMCP_Tiger_Test'
    >>> optimizer.n_jobs
    1
    >>> shutil.rmtree(cache_path, ignore_errors=True)

    >>> # Create test environment and belief
    >>> tiger_env = TigerPOMDP(discount_factor=0.95)
    >>> tiger_env.name
    'TigerPOMDP'
    >>> initial_belief = get_initial_belief(tiger_env, n_particles=10)  # Reduced for testing
    >>> len(initial_belief.particles)
    10

    >>> # Test hyperparameter definitions (note: order is low, high, name for Numerical)
    >>> numerical_param = NumericalHyperParameter(0.1, 10.0, "exploration_constant")
    >>> numerical_param.name
    'exploration_constant'
    >>> numerical_param.low
    0.1
    >>> numerical_param.high
    10.0

    >>> # For categorical: order is choices, name
    >>> categorical_param = CategoricalHyperParameter(["tpe", "random"], "algorithm")
    >>> categorical_param.name
    'algorithm'
    >>> len(categorical_param.choices)
    2

    >>> # Test configuration creation (simplified for testing)
    >>> from POMDPPlanners.core.simulation.hyperparameter_tuning import HyperParamPlannerConfig
    >>> planner_config = HyperParamPlannerConfig(
    ...     policy_cls=POMCP,
    ...     hyper_parameters=[numerical_param],
    ...     constant_parameters={"depth": 5}
    ... )
    >>> config = HyperParameterRunParams(
    ...     environment=tiger_env,
    ...     belief=initial_belief,
    ...     hyper_param_planner_config=planner_config,
    ...     num_episodes=2,        # Reduced for testing
    ...     num_steps=5,           # Reduced for testing
    ...     n_trials=3,            # Reduced for testing
    ...     parameters_to_optimize=[
    ...         ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
    ...     ]
    ... )
    >>> config.num_episodes
    2
    >>> config.n_trials
    3
    >>> config.parameters_to_optimize[0][1].value
    'maximize'

Note:
    This module requires Optuna and MLFlow to be installed. The optimization process
    can be computationally intensive for complex policies and large parameter spaces.

    Key Implementation Details:
    - Uses JoblibTaskManager for task execution and caching
    - All optimization runs are automatically tracked in MLflow
    - Requires explicit belief initialization (not generated automatically)
    - n_trials parameter is mandatory in HyperParameterRunParams
    - Results include optimized policies with their chosen hyperparameters
"""

import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import mlflow

from POMDPPlanners.core.simulation import (
    CategoricalHyperParameter,
    EnvironmentRunParams,
    NumericalHyperParameter,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterRunParams,
    OptimizedPolicyResult,
    ParallelizationLevel,
)
from POMDPPlanners.simulations.simulations_deployment.run_progress import (
    NotificationConfig,
    Notifier,
    build_notifier,
    install_signal_handlers,
)
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
    JoblibConfig,
    TaskManagerConfig,
)
from POMDPPlanners.simulations.simulations_deployment.tasks.hyper_parameter_tuning_simulation_task import (
    HyperParameterTuningSimulationTask,
)
from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.utils.logger import cleanup_all_loggers, get_logger

logger = get_logger(__name__)


class HyperParameterOptimizer:
    """Hyperparameter optimization for POMDP policies using Optuna and MLFlow.

    This class provides a complete framework for optimizing POMDP policy hyperparameters
    using Optuna's advanced optimization algorithms with integrated MLFlow experiment
    tracking. It supports batch optimization of multiple configurations with comprehensive
    result logging and analysis.

    The optimizer integrates with the JoblibTaskManager framework to leverage parallel
    computation and caching capabilities for efficient hyperparameter search across
    large parameter spaces. All optimization runs are automatically tracked in MLflow
    with detailed parameter logging, metrics recording, and artifact storage.

    Attributes:
        cache_dir_path: Directory for storing optimization results and cached data
        experiment_name: Name of the MLFlow experiment for tracking
        n_jobs: Number of parallel jobs for episode execution
        confidence_interval_level: Statistical confidence level for metrics
        alpha: Significance level for statistical tests (between 0.0 and 1.0).
            Used for hypothesis testing and confidence interval calculations.
            Defaults to 0.05 for 5% significance level.
        mlflow_tracking_uri: File URI for MLFlow tracking (always local file storage)
        mlruns_path: Path to MLFlow experiment tracking directory on local filesystem
        task_manager: JoblibTaskManager instance for task execution and caching

    Example:
        Creating and configuring hyperparameter optimizer:

        >>> from pathlib import Path
        >>> from POMDPPlanners.core.simulation import NumericalHyperParameter
        >>> from POMDPPlanners.core.simulation.hyperparameter_tuning import (
        ...     HyperParameterRunParams, HyperParameterOptimizationDirection
        ... )
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        >>> from POMDPPlanners.core.belief import get_initial_belief

        >>> # Test optimizer configuration
        >>> import tempfile
        >>> import shutil
        >>> cache_path = Path(tempfile.mkdtemp())
        >>> optimizer = HyperParameterOptimizer(
        ...     cache_dir_path=cache_path,
        ...     experiment_name="POMCP_Tuning_Test",
        ...     n_jobs=1,                   # Single job for testing
        ...     confidence_interval_level=0.95,
        ...     alpha=0.05
        ... )
        >>> optimizer.experiment_name
        'POMCP_Tuning_Test'
        >>> optimizer.confidence_interval_level
        0.95
        >>> optimizer.alpha
        0.05
        >>> shutil.rmtree(cache_path, ignore_errors=True)

        >>> # Create test environment and belief
        >>> env = TigerPOMDP(discount_factor=0.95)
        >>> initial_belief = get_initial_belief(env, n_particles=10)  # Reduced for testing

        >>> # Test configuration creation
        >>> from POMDPPlanners.core.simulation.hyperparameter_tuning import HyperParamPlannerConfig
        >>> planner_config = HyperParamPlannerConfig(
        ...     policy_cls=POMCP,
        ...     hyper_parameters=[
        ...         NumericalHyperParameter(0.1, 10.0, "exploration_constant"),
        ...     ],
        ...     constant_parameters={"depth": 5, "n_simulations": 200}
        ... )
        >>> config = HyperParameterRunParams(
        ...     environment=env,
        ...     belief=initial_belief,
        ...     hyper_param_planner_config=planner_config,
        ...     num_episodes=50,          # Reduced for testing
        ...     num_steps=5,             # Reduced for testing
        ...     n_trials=2,              # Reduced for testing
        ...     parameters_to_optimize=[
        ...         ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
        ...     ]
        ... )
        >>> len(config.hyper_param_planner_config.hyper_parameters)
        1
        >>> config.hyper_param_planner_config.hyper_parameters[0].name
        'exploration_constant'
        >>> config.parameters_to_optimize[0][0]
        'average_return'
    """

    def __init__(
        self,
        cache_dir_path: Path,
        experiment_name: str = "POMDP_Parameter_Optimization",
        task_manager_config: TaskManagerConfig = JoblibConfig(n_jobs=1),
        n_jobs: int = 1,
        confidence_interval_level: float = 0.95,
        alpha: float = 0.05,
        mlflow_tracking_uri: Optional[Path] = None,
        use_queue_logger: bool = False,
        parallelization_level: ParallelizationLevel = ParallelizationLevel.OPTUNA_TRIALS,
        notification_config: Optional[NotificationConfig] = None,
    ):
        """Initialize the hyperparameter optimizer.

        Sets up MLFlow experiment tracking, creates the JoblibTaskManager instance
        for task execution and caching, and configures optimization parameters.

        Args:
            cache_dir_path: Directory path for storing optimization results,
                MLFlow experiments, and simulation cache. Will be created if
                it doesn't exist.
            experiment_name: Name for the MLFlow experiment. Used for organizing
                optimization runs and tracking results. Defaults to
                "POMDP_Parameter_Optimization".
            n_jobs: Number of parallel jobs for episode execution. Higher values
                can significantly speed up optimization but require more system
                resources. Defaults to 1 for single-threaded execution.
            confidence_interval_level: Statistical confidence level for metrics
                computation (between 0.0 and 1.0). Used for computing confidence
                intervals in performance statistics. Defaults to 0.95 for 95%
                confidence intervals.
            alpha: Significance level for statistical tests (between 0.0 and 1.0).
                Used for hypothesis testing and confidence interval calculations.
                Defaults to 0.05 for 5% significance level.
            mlflow_tracking_uri: Path to custom MLFlow tracking directory on local machine.
                If None (default), uses cache_dir_path/mlruns for local file storage.
                If provided, must be a Path object pointing to the desired MLflow
                tracking directory on the local filesystem.
            use_queue_logger: Whether to use queue-based logging. Defaults to True.
            parallelization_level: Controls where parallelization is applied.
                OPTUNA_TRIALS (default) parallelizes across Optuna trials while
                running episodes sequentially. EPISODES parallelizes across
                episodes within each trial while running trials sequentially.
        Raises:
            TypeError: If cache_dir_path is not a Path object
        """
        self.cache_dir_path = cache_dir_path
        self.experiment_name = experiment_name
        self.task_manager_config = task_manager_config
        self.n_jobs = n_jobs
        self.confidence_interval_level = confidence_interval_level
        self.alpha = alpha
        self.use_queue_logger = use_queue_logger
        self.parallelization_level = parallelization_level
        # Set up MLFlow tracking
        if mlflow_tracking_uri is None:
            # Use local file storage in cache_dir_path/mlruns
            self.mlruns_path = cache_dir_path / "mlruns"
        else:
            # Use user-provided tracking path
            self.mlruns_path = Path(mlflow_tracking_uri)

        # Create the mlruns directory and set up MLflow
        self.mlruns_path.mkdir(parents=True, exist_ok=True)
        self.mlflow_tracking_uri: str = f"file://{self.mlruns_path.absolute()}"
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.task_manager = self.task_manager_config.create_task_manager(
            cache_dir=str(self.cache_dir_path / "task_manager_cache")
        )

        self.notification_config: NotificationConfig = (
            notification_config
            if notification_config is not None
            else NotificationConfig.disabled()
        )
        self._run_id: str = uuid.uuid4().hex
        self._notifier: Notifier = build_notifier(
            experiment_name=self.experiment_name,
            config=self.notification_config,
            logger=logger,
        )
        self._uninstall_signals: Callable[[], None] = install_signal_handlers(
            self._notifier, self._run_id
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.cleanup()

    def __getstate__(self) -> Dict:
        # The signal-handler uninstall callable is a closure and not
        # picklable; drop it. The unpickled optimizer is not the live
        # process that installed the handlers, so the no-op restore is
        # correct.
        state = self.__dict__.copy()
        state["_uninstall_signals"] = None
        return state

    def __setstate__(self, state: Dict) -> None:
        for key, value in state.items():
            setattr(self, key, value)
        if getattr(self, "_uninstall_signals", None) is None:
            self._uninstall_signals = lambda: None

    def cleanup(self):
        """Clean up resources including loggers and task managers."""
        # Clean up task manager
        if hasattr(self, "task_manager"):
            try:
                self.task_manager.__exit__(None, None, None)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        # Clean up any active MLflow runs
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        # Uninstall SIGTERM/SIGINT handlers installed at __init__ time.
        try:
            uninstall = getattr(self, "_uninstall_signals", None)
            if callable(uninstall):
                uninstall()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        # Clean up logger resources to prevent hanging
        try:
            cleanup_all_loggers()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    def _create_tasks(
        self, configs: List[HyperParameterRunParams]
    ) -> Tuple[List[HyperParameterTuningSimulationTask], List[str]]:
        total_trials = sum(config.n_trials for config in configs)
        tasks = []
        task_identifiers = []
        running_offset = 0
        for config in configs:
            task = HyperParameterTuningSimulationTask(
                environment=config.environment,
                belief=config.belief,
                policy_cls=config.hyper_param_planner_config.policy_cls,
                hyper_parameters=config.hyper_param_planner_config.hyper_parameters,
                constant_parameters=config.hyper_param_planner_config.constant_parameters,
                num_episodes=config.num_episodes,
                num_steps=config.num_steps,
                parameters_to_optimize=config.parameters_to_optimize,
                n_trials=config.n_trials,
                cache_dir=self.cache_dir_path,
                debug=False,
                console_output=True,
                n_jobs=self.n_jobs,
                confidence_interval_level=self.confidence_interval_level,
                alpha=self.alpha,
                parallelization_level=self.parallelization_level,
                experiment_name=self.experiment_name,
                parent_run_id=self._run_id,
                webhook_url=(
                    self.notification_config.webhook_url
                    if self.notification_config.is_active()
                    else None
                ),
                trial_interval=self.notification_config.trial_interval,
                trial_offset=running_offset,
                total_trials_globally=total_trials,
                progress_db_path=self.notification_config.progress_db_path,
            )
            tasks.append(task)
            task_identifiers.append(task.get_config_id())
            running_offset += config.n_trials

        return tasks, task_identifiers

    def _validate_optimization_configs(self, configs: List[HyperParameterRunParams]) -> None:
        """Validate optimization configurations before execution.

        Performs basic validation of the configs list structure. Individual config
        validation is now handled by HyperParameterRunParams.__post_init__ at
        construction time.

        Args:
            configs: List of hyperparameter run configurations to validate

        Raises:
            ValueError: If configs list is empty
            TypeError: If any element is not a HyperParameterRunParams instance
        """
        # Check configs is not empty
        if not configs:
            raise ValueError("configs list cannot be empty")

        # Validate each configuration type
        for idx, config in enumerate(configs):
            if not isinstance(config, HyperParameterRunParams):
                raise TypeError(
                    f"Configuration at index {idx} is not a HyperParameterRunParams instance. "
                    f"Got type: {type(config).__name__}"
                )

    def optimize(self, configs: List[HyperParameterRunParams]) -> List[OptimizedPolicyResult]:
        """Optimize hyperparameters for multiple configurations.

        This method provides the main interface for hyperparameter optimization by accepting
        a list of HyperParameterRunParams and returning OptimizedPolicyResult objects.
        It delegates to the task-based optimization infrastructure for execution.

        All optimization runs and results are tracked in MLflow with comprehensive
        parameter logging, metrics recording, and artifact storage. The method automatically
        creates MLflow runs for batch-level tracking and individual configuration tracking.

        Args:
            configs: List of hyperparameter run configurations, each containing environment,
                policy class, hyperparameter ranges, and optimization settings. Each config
                must include the required n_trials parameter.

        Returns:
            List of OptimizedPolicyResult objects containing the optimized policies
            and their parameters for each input configuration.

        Raises:
            ValueError: If any configuration contains invalid parameters
            TypeError: If policy classes are not Policy subclasses
            RuntimeError: If optimization fails for any configuration

        Note:
            This method automatically handles MLflow experiment tracking and creates
            nested runs for each configuration. All results are logged with comprehensive
            metadata including hyperparameter ranges, optimization metrics, and final
            evaluation statistics.
        """
        if not configs:
            return []

        # Validate input configurations before starting optimization
        self._validate_optimization_configs(configs)

        logger.info(
            "Starting optimization for %s configurations using stub interface with MLflow tracking",
            len(configs),
        )

        total_trials = sum(config.n_trials for config in configs)
        self._notifier.run_started(
            self._run_id,
            metadata={
                "num_configs": len(configs),
                "total_trials": total_trials,
                "trial_notification_interval": self.notification_config.trial_interval,
                "n_jobs": self.n_jobs,
            },
        )

        try:
            # Prepare MLflow session and execute optimization tasks
            self._prepare_mlflow_session()

            with mlflow.start_run(run_name=f"optimize_batch_{len(configs)}_configs"):
                # Log batch-level information
                self._log_batch_level_parameters(configs)

                # Execute optimization tasks
                task_results, tasks = self._execute_optimization_tasks(configs)

                # Process results and log to MLflow
                results = self._process_task_results_with_mlflow_logging(
                    configs, task_results, tasks
                )

                # Log batch-level summary
                self._log_batch_level_summary(configs, results)

            logger.info(
                "All %s configurations optimized successfully with MLflow tracking",
                len(configs),
            )
        except Exception as exc:
            try:
                self._notifier.run_failed(self._run_id, f"{type(exc).__name__}: {exc}")
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            raise

        try:
            self._notifier.run_finished(self._run_id)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        return results

    def _prepare_mlflow_session(self) -> None:
        """Prepare MLflow session by ending any active run."""
        active_run = mlflow.active_run()
        if active_run is not None:
            mlflow.end_run()

    def _log_batch_level_parameters(self, configs: List[HyperParameterRunParams]) -> None:
        # Log batch-level parameters
        mlflow.log_param("num_configurations", len(configs))
        mlflow.log_param("batch_method", "stub_interface_optimize")

        # Log summary of all configurations
        config_summary = {
            "environments": [config.environment.__class__.__name__ for config in configs],
            "policies": [
                config.hyper_param_planner_config.policy_cls.__name__ for config in configs
            ],
            "parameters_to_optimize": [
                [
                    (param_name, direction.value)
                    for param_name, direction in config.parameters_to_optimize
                ]
                for config in configs
            ],
        }
        mlflow.log_dict(config_summary, "batch_configuration_summary.json")

    def _execute_optimization_tasks(
        self, configs: List[HyperParameterRunParams]
    ) -> Tuple[List[OptimizedPolicyResult], List[HyperParameterTuningSimulationTask]]:
        tasks, task_identifiers = self._create_tasks(configs)

        with self.task_manager:
            task_results, task_identifiers = self.task_manager.run_tasks(
                tasks=tasks, task_identifiers=task_identifiers  # type: ignore
            )

        return task_results, tasks

    def _process_task_results_with_mlflow_logging(
        self,
        configs: List[HyperParameterRunParams],
        task_results: List[OptimizedPolicyResult],
        tasks: List[HyperParameterTuningSimulationTask],
    ) -> List[OptimizedPolicyResult]:
        results = []

        # Match successful task results with their configs and tasks
        successful_configs_with_index = self._match_successful_results_with_configs(
            task_results, configs, tasks
        )

        # Process each successful result
        for original_index, config, task, task_result in successful_configs_with_index:
            logger.info(
                "Processing results for configuration %s/%s: %s with %s",
                original_index + 1,
                len(configs),
                config.environment.__class__.__name__,
                config.hyper_param_planner_config.policy_cls.__name__,
            )

            # Log this configuration's results to MLflow
            optimization_result = self._log_single_configuration_results(
                original_index, config, task, task_result
            )

            if optimization_result:
                results.append(optimization_result)

        return results

    def _match_successful_results_with_configs(
        self, task_results: List, configs: List[HyperParameterRunParams], tasks: List
    ) -> List[tuple]:
        successful_configs_with_index = []
        for i, task_result in enumerate(task_results):
            if task_result is not None and i < len(configs) and i < len(tasks):
                successful_configs_with_index.append((i, configs[i], tasks[i], task_result))
        return successful_configs_with_index

    def _log_single_configuration_results(
        self,
        original_index: int,
        config: HyperParameterRunParams,
        task: "HyperParameterTuningSimulationTask",
        task_result: "OptimizedPolicyResult",
    ) -> "OptimizedPolicyResult":  # type: ignore
        # Get parameters as a string for logging
        _params_str = ", ".join(  # pylint: disable=unused-variable
            [
                f"{param_name}({direction.value})"
                for param_name, direction in config.parameters_to_optimize
            ]
        )

        # Start nested run for this configuration
        with mlflow.start_run(
            run_name=f"config_{original_index+1}_{config.environment.__class__.__name__}_{config.hyper_param_planner_config.policy_cls.__name__}",
            nested=True,
        ):
            try:
                # Log configuration parameters
                all_params = self._prepare_configuration_parameters(original_index, config)
                mlflow.log_params(all_params)

                # Process and log optimization results
                self._log_optimization_results(task_result, task)

                # Run and log final evaluation
                self._run_and_log_final_evaluation(config, task_result, all_params, original_index)

                # Log success and return result
                best_value = self._get_best_value_from_task(task)
                logger.info(
                    "Configuration %s optimized successfully. Best value: %s",
                    original_index + 1,
                    best_value,
                )
                return task_result

            except Exception as e:
                # Log failure metrics
                self._log_configuration_failure(original_index, e)
                raise RuntimeError(
                    f"Failed to optimize configuration {original_index+1}: {e}"
                ) from e

    def _prepare_configuration_parameters(
        self,
        original_index: int,
        config: HyperParameterRunParams,
    ) -> dict:
        # Log configuration parameters
        config_params = {
            "config_index": original_index + 1,
            "environment_type": config.environment.__class__.__name__,
            "policy_type": config.hyper_param_planner_config.policy_cls.__name__,
            "num_episodes": config.num_episodes,
            "num_steps": config.num_steps,
            "parameters_to_optimize": str(
                [
                    (param_name, direction.value)
                    for param_name, direction in config.parameters_to_optimize
                ]
            ),
            "n_trials": config.n_trials,
        }

        # Log environment-specific parameters
        env_params = {
            f"env_{key}": value
            for key, value in config.environment.__dict__.items()
            if isinstance(value, (int, float, str, bool))
        }

        # Log hyperparameter ranges
        param_ranges_info = {}
        for param in config.hyper_param_planner_config.hyper_parameters:
            if isinstance(param, CategoricalHyperParameter):
                param_ranges_info[f"param_range_{param.name}"] = f"choices: {param.choices}"
            elif isinstance(param, NumericalHyperParameter):
                param_ranges_info[f"param_range_{param.name}"] = f"{param.low}-{param.high}"

        # Log constant parameters (non-optimized policy parameters)
        constant_params = {
            f"constant_{key}": value
            for key, value in config.hyper_param_planner_config.constant_parameters.items()
            if isinstance(value, (int, float, str, bool))
        }

        # Combine all parameters
        return {**config_params, **env_params, **param_ranges_info, **constant_params}

    def _extract_all_policy_parameters(self, optimization_result: "OptimizedPolicyResult") -> dict:
        """Extract all parameters from the optimized policy for comprehensive MLflow logging.

        This method extracts all policy parameters including:
        - Policy name and type
        - All policy attributes (instance variables)
        - Constant parameters that were used during policy creation
        - Environment reference information

        Args:
            optimization_result: The OptimizedPolicyResult containing the optimized policy

        Returns:
            dict: Dictionary of all policy parameters suitable for MLflow logging
        """
        policy = optimization_result.policy
        policy_params = {}

        # Log policy basic information
        policy_params["policy_name"] = getattr(policy, "name", "unnamed")
        policy_params["policy_class"] = policy.__class__.__name__

        # Log all policy attributes (instance variables)
        for attr_name, attr_value in policy.__dict__.items():
            # Skip private attributes and complex objects
            if not attr_name.startswith("_") and isinstance(attr_value, (int, float, str, bool)):
                policy_params[f"policy_{attr_name}"] = attr_value
            elif not attr_name.startswith("_") and isinstance(attr_value, (list, tuple)):
                # Log simple lists/tuples as strings
                if all(isinstance(item, (int, float, str, bool)) for item in attr_value):
                    policy_params[f"policy_{attr_name}"] = str(attr_value)

        # Log environment information
        env = optimization_result.environment
        policy_params["env_name"] = getattr(env, "name", "unnamed")
        policy_params["env_class"] = env.__class__.__name__

        # Log environment attributes that are simple types
        for attr_name, attr_value in env.__dict__.items():
            if not attr_name.startswith("_") and isinstance(attr_value, (int, float, str, bool)):
                policy_params[f"env_{attr_name}"] = attr_value

        return policy_params

    def _log_optimization_results(
        self,
        optimization_result: "OptimizedPolicyResult",
        task: "HyperParameterTuningSimulationTask",
    ) -> None:
        if optimization_result is not None:
            # Log optimization results (best hyperparameters)
            mlflow.log_params(
                {f"best_{k}": v for k, v in optimization_result.chosen_hyper_parameters.items()}
            )

            # Log all policy parameters (including constant parameters and policy attributes)
            all_policy_params = self._extract_all_policy_parameters(optimization_result)
            mlflow.log_params(all_policy_params)

            # Get additional metadata from the task if available
            task_metadata = task.get_optimization_metadata()
            if task_metadata:
                mlflow.log_metric("best_pareto_score", task_metadata["best_pareto_score"])
                mlflow.log_metric("optimization_time", task_metadata["optimization_time"])
                mlflow.log_metric("n_trials_executed", task_metadata["n_trials"])
                if task_metadata["best_trial_number"] is not None:
                    mlflow.log_metric("best_trial_number", task_metadata["best_trial_number"])

                # Log individual metric values from best trial
                if "best_trial_metrics" in task_metadata:
                    for metric_name, metric_value in task_metadata["best_trial_metrics"].items():
                        if metric_value is not None:
                            mlflow.log_metric(f"best_trial_{metric_name}", metric_value)

            mlflow.log_metric("optimization_success", 1.0)
        else:
            # Task failed - log failure metrics
            mlflow.log_metric("optimization_success", 0.0)
            mlflow.log_param("error_message", "Task execution returned None")

    def _run_and_log_final_evaluation(
        self,
        config: HyperParameterRunParams,
        optimization_result: "OptimizedPolicyResult",
        all_params: dict,
        original_index: int,
    ) -> None:
        task_manager_config = JoblibConfig(n_jobs=self.n_jobs)
        simulator = POMDPSimulator(
            task_manager_config=task_manager_config,
            cache_dir_path=None,
            experiment_name=f"optimization_results_config_{original_index+1}",
        )

        env_run_params = [
            EnvironmentRunParams(
                environment=config.environment,
                belief=config.belief,
                policies=[optimization_result.policy],
                num_episodes=config.num_episodes,
                num_steps=config.num_steps,
            )
        ]

        # Use the simulator's _run_simulations_and_compute_metrics method
        _, metrics = (
            simulator._run_simulations_and_compute_metrics(  # pylint: disable=protected-access
                environment_run_params=env_run_params,
                alpha=self.alpha,
                confidence_interval_level=self.confidence_interval_level,
                n_jobs=self.n_jobs,
            )
        )

        # Extract final statistics from metrics
        final_statistics = metrics[config.environment.name][optimization_result.policy.name]

        # Log final evaluation metrics with final_ prefix to distinguish from optimization metrics
        for metric in final_statistics:
            mlflow.log_metric(f"final_{metric.name}", metric.value)
            mlflow.log_metric(f"final_{metric.name}_lower_ci", metric.lower_confidence_bound)
            mlflow.log_metric(f"final_{metric.name}_upper_ci", metric.upper_confidence_bound)

        # Save detailed results as artifacts
        self._save_detailed_results_as_artifacts(
            optimization_result, config, all_params, final_statistics, original_index
        )

    def _save_detailed_results_as_artifacts(
        self,
        optimization_result: "OptimizedPolicyResult",
        config: HyperParameterRunParams,
        all_params: dict,
        final_statistics: List,
        original_index: int,
    ) -> None:
        # Get task metadata for best value
        task_metadata = getattr(optimization_result, "_task_metadata", None)

        results_data = {
            "configuration_index": original_index + 1,
            "best_parameters": optimization_result.chosen_hyper_parameters,
            "best_pareto_score": task_metadata["best_pareto_score"] if task_metadata else "unknown",
            "best_trial_metrics": (
                task_metadata.get("best_trial_metrics", {}) if task_metadata else {}
            ),
            "hyperparameter_ranges": [
                param._asdict() for param in config.hyper_param_planner_config.hyper_parameters
            ],
            "final_statistics": [metric._asdict() for metric in final_statistics],
            "configuration_params": all_params,
            "optimization_metadata": task_metadata,
        }
        mlflow.log_dict(results_data, f"optimization_results_config_{original_index+1}.json")

        # Log the planner's chosen configuration as a separate artifact
        planner_config = {
            "planner_type": config.hyper_param_planner_config.policy_cls.__name__,
            "chosen_hyper_parameters": optimization_result.chosen_hyper_parameters,
            "constant_parameters": (
                config.hyper_param_planner_config.constant_parameters
                if hasattr(config, "constant_parameters")
                else {}
            ),
            "environment_type": config.environment.__class__.__name__,
            "parameters_to_optimize": [
                (param_name, direction.value)
                for param_name, direction in config.parameters_to_optimize
            ],
            "best_pareto_score": (
                task_metadata.get("best_pareto_score") if task_metadata else "unknown"
            ),
            "best_trial_metrics": (
                task_metadata.get("best_trial_metrics") if task_metadata else "unknown"
            ),
            "all_policy_parameters": self._extract_all_policy_parameters(optimization_result),
            "policy_creation_params": {
                "policy_name": getattr(optimization_result.policy, "name", "unnamed"),
                "policy_class": optimization_result.policy.__class__.__name__,
                "environment_name": getattr(optimization_result.environment, "name", "unnamed"),
                "environment_class": optimization_result.environment.__class__.__name__,
            },
        }
        mlflow.log_dict(planner_config, f"planner_chosen_config_{original_index+1}.json")

    def _get_best_value_from_task(self, task: "HyperParameterTuningSimulationTask") -> str:
        task_metadata = task.get_optimization_metadata()
        if task_metadata:
            # Format as "pareto_score: X (metrics: {...})"
            pareto_score = task_metadata.get("best_pareto_score", "unknown")
            metrics = task_metadata.get("best_trial_metrics", {})
            return f"pareto_score: {pareto_score}, metrics: {metrics}"
        return "unknown"

    def _log_configuration_failure(self, original_index: int, exception: Exception) -> None:
        mlflow.log_metric("optimization_success", 0.0)
        mlflow.log_param("error_message", str(exception))
        logger.error("Optimization failed for configuration %s: %s", original_index + 1, exception)

    def _log_batch_level_summary(
        self, configs: List[HyperParameterRunParams], results: List
    ) -> None:
        # Log batch-level summary metrics
        mlflow.log_metric("batch_success_rate", len(results) / len(configs))
        mlflow.log_metric("batch_completed_configs", len(results))
        mlflow.log_metric("batch_failed_configs", len(configs) - len(results))

        # Create batch summary
        batch_summary = {
            "total_configurations": len(configs),
            "successful_configurations": len(results),
            "failed_configurations": len(configs) - len(results),
            "success_rate": len(results) / len(configs),
            "environment_types": list(
                set(config.environment.__class__.__name__ for config in configs)
            ),
            "policy_types": list(
                set(config.hyper_param_planner_config.policy_cls.__name__ for config in configs)
            ),
        }
        mlflow.log_dict(batch_summary, "batch_optimization_summary.json")
