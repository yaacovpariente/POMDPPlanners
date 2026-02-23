"""Workflow classes for optimization and evaluation.

This module provides class-based workflows for running hyperparameter optimization
followed by policy evaluation in different execution environments.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import pandas as pd

from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterRunParams,
    OptimizedPolicyResult,
)
from POMDPPlanners.core.simulation import EnvironmentRunParams

if TYPE_CHECKING:
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.belief import Belief
    from POMDPPlanners.core.policy import Policy
from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import (
    HyperParameterOptimizer,
)  # pylint: disable=wrong-import-position
from POMDPPlanners.simulations.simulator import (
    POMDPSimulator,
)  # pylint: disable=wrong-import-position
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (  # pylint: disable=wrong-import-position
    DaskConfig,
    JoblibConfig,
    PBSConfig,
    TaskManagerConfig,
)
from POMDPPlanners.utils.logger import get_logger  # pylint: disable=wrong-import-position

logger = get_logger(__name__)


class OptimizationEvaluationWorkflow(ABC):
    """Base class for optimization and evaluation workflows.

    This class encapsulates common parameters and logic for running
    hyperparameter optimization followed by policy evaluation.

    Attributes:
        cache_dir: Directory for caching results.
        experiment_name: Name of the experiment.
        evaluation_episodes: Number of episodes for evaluation.
        evaluation_steps: Maximum steps per episode for evaluation.
        evaluation_n_jobs: Number of parallel jobs for evaluation.
        confidence_interval_level: Confidence level for statistical intervals.
        alpha: Significance level for statistical tests.
        debug: Enable debug mode.
        verbose: Enable verbose logging.
        cache_visualizations: Whether to cache visualization outputs.
    """

    def __init__(
        self,
        cache_dir: Path,
        experiment_name: str,
        optimization_n_jobs: int = 1,
        evaluation_episodes: int = 3,
        evaluation_steps: int = 6,
        evaluation_n_jobs: int = 1,
        confidence_interval_level: float = 0.95,
        alpha: float = 0.05,
        debug: bool = False,
        verbose: bool = True,
        cache_visualizations: bool = True,
    ):
        """Initialize the workflow with common parameters.

        Args:
            cache_dir: Directory for caching results.
            experiment_name: Name of the experiment.
            optimization_n_jobs: Number of parallel jobs for optimization.
            evaluation_episodes: Number of episodes for evaluation.
            evaluation_steps: Maximum steps per episode for evaluation.
            evaluation_n_jobs: Number of parallel jobs for evaluation.
            confidence_interval_level: Confidence level for intervals.
            alpha: Significance level for statistical tests.
            debug: Enable debug mode.
            verbose: Enable verbose logging.
            cache_visualizations: Whether to cache visualizations.
        """
        self.cache_dir = cache_dir
        self.experiment_name = experiment_name
        self.evaluation_episodes = evaluation_episodes
        self.evaluation_steps = evaluation_steps
        self.optimization_n_jobs = optimization_n_jobs
        self.evaluation_n_jobs = evaluation_n_jobs
        self.confidence_interval_level = confidence_interval_level
        self.alpha = alpha
        self.debug = debug
        self.verbose = verbose
        self.cache_visualizations = cache_visualizations

    @abstractmethod
    def _get_task_manager_config_hyperparameter(self) -> TaskManagerConfig:
        """Get task manager configuration for hyperparameter optimization.

        Returns:
            Task manager configuration for optimization phase.
        """

    @abstractmethod
    def _get_task_manager_config_evaluation(self) -> TaskManagerConfig:
        """Get task manager configuration for evaluation.

        Returns:
            Task manager configuration for evaluation phase.
        """

    def _validate_configs(self, configs: List[HyperParameterRunParams]) -> None:
        """Validate workflow configurations before execution.

        This function performs workflow-level validation to ensure the configs
        list is properly structured. Individual config validation is handled by
        HyperParameterRunParams.__post_init__() at construction time.

        Args:
            configs: List of hyperparameter run configurations to validate

        Raises:
            ValueError: If configs list is empty
            TypeError: If any element is not a HyperParameterRunParams instance

        Note:
            Detailed parameter validation (num_episodes, num_steps, environment,
            belief, policy_cls, hyper_parameters, parameters_to_optimize, metric
            names) is performed by HyperParameterRunParams at construction time.
        """
        # Check configs is not empty
        if not configs:
            raise ValueError("configs list cannot be empty")

        # Validate each element is a HyperParameterRunParams instance
        for idx, config in enumerate(configs):
            if not isinstance(config, HyperParameterRunParams):
                raise TypeError(
                    f"Configuration at index {idx} is not a HyperParameterRunParams instance. "
                    f"Got type: {type(config).__name__}"
                )

    def optimize_and_evaluate(
        self,
        configs: List[HyperParameterRunParams],
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Optimize and evaluate multiple POMDP planners.

        Args:
            configs: List of hyperparameter run configurations.

        Returns:
            Tuple of results dictionary and DataFrame.

        Raises:
            ValueError: If any configuration contains invalid parameters
            TypeError: If policy classes are not Policy subclasses or configs have wrong types
        """
        # Validate configurations before starting workflow
        self._validate_configs(configs)

        if self.verbose:
            logger.info("Optimizing %d environments and policies", len(configs))

        if self.debug:
            logger.debug("Optimizing %d environments and policies", len(configs))

        optimizer = HyperParameterOptimizer(
            cache_dir_path=self.cache_dir,
            experiment_name=self.experiment_name,
            task_manager_config=self._get_task_manager_config_hyperparameter(),
            n_jobs=self.optimization_n_jobs,
            confidence_interval_level=self.confidence_interval_level,
            alpha=self.alpha,
        )

        optimization_results: List[OptimizedPolicyResult] = optimizer.optimize(configs)

        # Aggregate policies by environment
        env_to_results: Dict[str, Tuple[Environment, Belief, List[Policy]]] = {}
        for i, result in enumerate(optimization_results):
            env_id = result.environment.config_id
            if env_id not in env_to_results:
                env_to_results[env_id] = (result.environment, configs[i].belief, [])
            env_to_results[env_id][2].append(result.policy)

        optimization_results_organized = list(env_to_results.values())

        chosen_planners_eval_configs = [
            EnvironmentRunParams(
                environment=env,
                belief=belief,
                policies=policies,
                num_episodes=self.evaluation_episodes,
                num_steps=self.evaluation_steps,
            )
            for env, belief, policies in optimization_results_organized
        ]

        with POMDPSimulator(
            task_manager_config=self._get_task_manager_config_evaluation(),
            cache_dir_path=self.cache_dir,
            experiment_name=self.experiment_name,
            debug=self.debug,
        ) as simulator:
            results = simulator.compare_multiple_environments_policies(
                environment_run_params=chosen_planners_eval_configs,
                alpha=self.alpha,
                confidence_interval_level=self.confidence_interval_level,
                n_jobs=self.evaluation_n_jobs,
                cache_visualizations=self.cache_visualizations,
            )

        return results


class OptimizationEvaluationLocalWorkflow(OptimizationEvaluationWorkflow):
    """Workflow for local execution using Joblib parallelization.

    Attributes:
        optimization_n_jobs: Number of parallel jobs for optimization (-1 uses all cores).
        All other attributes inherited from OptimizationEvaluationWorkflow.

    Example:
        >>> from pathlib import Path
        >>> workflow = OptimizationEvaluationLocalWorkflow(
        ...     cache_dir=Path("./results"),
        ...     experiment_name="My_Experiment",
        ...     optimization_n_jobs=-1,
        ... )
    """

    def __init__(
        self,
        cache_dir: Path,
        experiment_name: str,
        optimization_n_jobs: int = -1,
        evaluation_episodes: int = 2,
        evaluation_steps: int = 6,
        evaluation_n_jobs: int = 1,
        confidence_interval_level: float = 0.95,
        alpha: float = 0.05,
        debug: bool = False,
        verbose: bool = True,
        cache_visualizations: bool = True,
    ):
        """Initialize local workflow.

        Args:
            cache_dir: Directory for caching results.
            experiment_name: Name of the experiment.
            optimization_n_jobs: Number of parallel jobs for optimization.
            evaluation_episodes: Number of episodes for evaluation.
            evaluation_steps: Maximum steps per episode for evaluation.
            evaluation_n_jobs: Number of parallel jobs for evaluation.
            confidence_interval_level: Confidence level for intervals.
            alpha: Significance level for statistical tests.
            debug: Enable debug mode.
            verbose: Enable verbose logging.
            cache_visualizations: Whether to cache visualizations.
        """
        super().__init__(
            cache_dir=cache_dir,
            experiment_name=experiment_name,
            optimization_n_jobs=optimization_n_jobs,
            evaluation_episodes=evaluation_episodes,
            evaluation_steps=evaluation_steps,
            evaluation_n_jobs=evaluation_n_jobs,
            confidence_interval_level=confidence_interval_level,
            alpha=alpha,
            debug=debug,
            verbose=verbose,
            cache_visualizations=cache_visualizations,
        )

    def _get_task_manager_config_hyperparameter(self) -> TaskManagerConfig:
        """Get Joblib task manager for hyperparameter optimization."""
        # n_jobs of this config is responsible for the parallelization of multiple optimization tasks.
        # In local execution, we only have one optimization task, as the parallelization is done by Optuna package using self.n_jobs
        return JoblibConfig(n_jobs=1)

    def _get_task_manager_config_evaluation(self) -> TaskManagerConfig:
        """Get Joblib task manager for evaluation."""
        return JoblibConfig(n_jobs=self.evaluation_n_jobs)


class OptimizationEvaluationDaskWorkflow(OptimizationEvaluationWorkflow):
    """Workflow for Dask distributed execution.

    Attributes:
        n_workers: Number of Dask workers.
        scheduler_address: Optional address of existing Dask scheduler.
        cache_size: Size of cache in bytes.
        clear_cache_on_start: Whether to clear cache at startup.
        All other attributes inherited from OptimizationEvaluationWorkflow.

    Example:
        >>> from pathlib import Path
        >>> workflow = OptimizationEvaluationDaskWorkflow(
        ...     cache_dir=Path("./results"),
        ...     experiment_name="Dask_Experiment",
        ...     n_workers=4,
        ... )
    """

    def __init__(
        self,
        cache_dir: Path,
        experiment_name: str,
        n_workers: int = 4,
        scheduler_address: Optional[str] = None,
        cache_size: int = int(2e9),
        clear_cache_on_start: bool = False,
        evaluation_episodes: int = 2,
        evaluation_steps: int = 6,
        evaluation_n_jobs: int = 1,
        confidence_interval_level: float = 0.95,
        alpha: float = 0.05,
        debug: bool = False,
        verbose: bool = True,
        cache_visualizations: bool = True,
    ):
        """Initialize Dask workflow.

        Args:
            cache_dir: Directory for caching results.
            experiment_name: Name of the experiment.
            n_workers: Number of Dask worker processes.
            scheduler_address: Address of existing Dask scheduler (None for local cluster).
            cache_size: Size of cache in bytes.
            clear_cache_on_start: Whether to clear cache at startup.
            evaluation_episodes: Number of episodes for evaluation.
            evaluation_steps: Maximum steps per episode for evaluation.
            evaluation_n_jobs: Number of parallel jobs for evaluation.
            confidence_interval_level: Confidence level for intervals.
            alpha: Significance level for statistical tests.
            debug: Enable debug mode.
            verbose: Enable verbose logging.
            cache_visualizations: Whether to cache visualizations.
        """
        self.n_workers = n_workers
        self.scheduler_address = scheduler_address
        self.cache_size = cache_size
        self.clear_cache_on_start = clear_cache_on_start

        super().__init__(
            cache_dir=cache_dir,
            experiment_name=experiment_name,
            optimization_n_jobs=n_workers,
            evaluation_episodes=evaluation_episodes,
            evaluation_steps=evaluation_steps,
            evaluation_n_jobs=evaluation_n_jobs,
            confidence_interval_level=confidence_interval_level,
            alpha=alpha,
            debug=debug,
            verbose=verbose,
            cache_visualizations=cache_visualizations,
        )

    def _get_task_manager_config_hyperparameter(self) -> TaskManagerConfig:
        """Get Dask task manager for hyperparameter optimization."""
        return DaskConfig(
            n_workers=self.n_workers,
            scheduler_address=self.scheduler_address,
            cache_size=self.cache_size,
            clear_cache_on_start=self.clear_cache_on_start,
        )

    def _get_task_manager_config_evaluation(self) -> TaskManagerConfig:
        """Get Dask task manager for evaluation."""
        return DaskConfig(
            n_workers=self.n_workers,
            scheduler_address=self.scheduler_address,
            cache_size=self.cache_size,
            clear_cache_on_start=False,  # Don't clear cache during evaluation
        )


class OptimizationEvaluationPBSWorkflow(OptimizationEvaluationWorkflow):
    """Workflow for PBS cluster execution.

    Attributes:
        queue: PBS queue name.
        n_workers: Number of PBS workers.
        cores: Cores per worker.
        memory: Memory per worker.
        processes: Processes per worker.
        walltime: Maximum walltime.
        job_extra: Additional PBS job parameters.
        All other attributes inherited from OptimizationEvaluationWorkflow.

    Example:
        >>> from pathlib import Path
        >>> workflow = OptimizationEvaluationPBSWorkflow(
        ...     cache_dir=Path("./results"),
        ...     experiment_name="Cluster_Experiment",
        ...     queue="short",
        ...     n_workers=10,
        ...     cores=4,
        ...     memory="8GB",
        ... )
        >>> workflow.n_workers
        10
    """

    def __init__(
        self,
        cache_dir: Path,
        experiment_name: str,
        queue: str = "short",
        n_workers: int = 4,
        cores: int = 1,
        memory: str = "4GB",
        processes: int = 1,
        walltime: str = "03:00:00",
        job_extra: Optional[List[str]] = None,
        evaluation_episodes: int = 2,
        evaluation_steps: int = 6,
        evaluation_n_jobs: int = 1,
        is_risk_averse: bool = False,  # pylint: disable=unused-argument
        confidence_interval_level: float = 0.95,
        alpha: float = 0.05,
        debug: bool = False,
        verbose: bool = True,
        cache_visualizations: bool = True,
        enable_dashboard: bool = True,
        dashboard_address: str = "0.0.0.0",
        dashboard_port: int = 8787,
        dashboard_prefix: Optional[str] = None,
    ):
        """Initialize PBS workflow.

        Args:
            cache_dir: Directory for caching results.
            experiment_name: Name of the experiment.
            queue: PBS queue name.
            n_workers: Number of PBS workers.
            cores: Cores to allocate per worker.
            memory: Memory per worker.
            processes: Processes per worker.
            walltime: Maximum walltime.
            job_extra: Additional PBS job parameters.
            evaluation_episodes: Number of episodes for evaluation.
            evaluation_steps: Maximum steps per episode for evaluation.
            evaluation_n_jobs: Number of parallel jobs for evaluation.
            is_risk_averse: Whether to run risk-averse benchmark.
            confidence_interval_level: Confidence level for intervals.
            alpha: Significance level for statistical tests.
            debug: Enable debug mode.
            verbose: Enable verbose logging.
            cache_visualizations: Whether to cache visualizations.
            enable_dashboard: Whether to enable the Dask dashboard.
            dashboard_address: Address to bind the dashboard to.
            dashboard_port: Port for the Dask dashboard.
            dashboard_prefix: URL prefix for dashboard (useful with reverse proxies).
        """
        self.queue = queue
        self.n_workers = n_workers
        self.cores = cores
        self.memory = memory
        self.processes = processes
        self.walltime = walltime
        self.job_extra = job_extra
        self.enable_dashboard = enable_dashboard
        self.dashboard_address = dashboard_address
        self.dashboard_port = dashboard_port
        self.dashboard_prefix = dashboard_prefix

        super().__init__(
            cache_dir=cache_dir,
            experiment_name=experiment_name,
            optimization_n_jobs=cores,
            evaluation_episodes=evaluation_episodes,
            evaluation_steps=evaluation_steps,
            evaluation_n_jobs=evaluation_n_jobs,
            confidence_interval_level=confidence_interval_level,
            alpha=alpha,
            debug=debug,
            verbose=verbose,
            cache_visualizations=cache_visualizations,
        )

    def _get_task_manager_config_hyperparameter(self) -> TaskManagerConfig:
        """Get PBS task manager for hyperparameter optimization."""
        return PBSConfig(
            queue=self.queue,
            n_workers=self.n_workers,
            cores=self.cores,
            memory=self.memory,
            processes=1,
            walltime=self.walltime,
            job_extra=self.job_extra,
            enable_dashboard=self.enable_dashboard,
            dashboard_address=self.dashboard_address,
            dashboard_port=self.dashboard_port,
            dashboard_prefix=self.dashboard_prefix,
        )

    def _get_task_manager_config_evaluation(self) -> TaskManagerConfig:
        """Get PBS task manager for evaluation."""
        return PBSConfig(
            queue=self.queue,
            n_workers=self.n_workers,
            cores=self.cores,
            memory=self.memory,
            processes=self.processes,
            walltime=self.walltime,
            job_extra=self.job_extra,
            enable_dashboard=self.enable_dashboard,
            dashboard_address=self.dashboard_address,
            dashboard_port=self.dashboard_port,
            dashboard_prefix=self.dashboard_prefix,
        )
