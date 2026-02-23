# pylint: disable=fixme
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List

from POMDPPlanners.core.simulation.simulation_configs import EnvironmentRunParams
from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
    TaskManagerConfig,
    JoblibConfig,
    DaskConfig,
    PBSConfig,
)


class PlannerEvaluationWorkflow(ABC):
    def __init__(
        self,
        experiment_name: str,
        cache_dir_path: Optional[Path],
        debug: bool = False,
        n_jobs: int = 1,
        enable_profiling: bool = False,
        verbose: bool = True,
        alpha: float = 0.1,
        confidence_interval_level: float = 0.95,
        cache_visualizations: bool = True,
    ):
        self.cache_dir_path = cache_dir_path
        self.experiment_name = experiment_name
        self.debug = debug
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.enable_profiling = enable_profiling
        self.alpha = alpha
        self.confidence_interval_level = confidence_interval_level
        self.cache_visualizations = cache_visualizations

    def evaluate(self, configs: List[EnvironmentRunParams]):
        self._validate_configs(configs)
        simulator = POMDPSimulator(
            cache_dir_path=self.cache_dir_path,
            experiment_name=self.experiment_name,
            debug=self.debug,
            task_console_output=False,
            enable_profiling=self.enable_profiling,
            task_manager_config=self._get_task_manager_config(),
        )
        return simulator.compare_multiple_environments_policies(
            environment_run_params=configs,
            alpha=self.alpha,
            confidence_interval_level=self.confidence_interval_level,
            n_jobs=self.n_jobs,
            cache_visualizations=self.cache_visualizations,
        )

    def _validate_configs(self, configs: List[EnvironmentRunParams]) -> None:
        """Validate evaluation configurations before execution.

        Performs basic validation of the configs list structure. Individual config
        validation is now handled by EnvironmentRunParams.__post_init__ at
        construction time.

        Args:
            configs: List of EnvironmentRunParams to validate.

        Raises:
            ValueError: If configs list is empty
            TypeError: If configs is not a list or any element is not EnvironmentRunParams
        """
        # Check configs is not empty
        if not configs:
            raise ValueError("configs list cannot be empty")

        # Check configs is a list
        if not isinstance(configs, list):
            raise TypeError(f"configs must be a list, got {type(configs).__name__}")

        # Validate each configuration type
        for idx, config in enumerate(configs):
            if not isinstance(config, EnvironmentRunParams):
                raise TypeError(
                    f"Configuration at index {idx}: Expected EnvironmentRunParams, "
                    f"got {type(config).__name__}"
                )

    @abstractmethod
    def _get_task_manager_config(self) -> TaskManagerConfig:
        pass


class PlannerEvaluationLocalWorkflow(PlannerEvaluationWorkflow):
    """Workflow for local planner evaluation using Joblib parallelization.

    Attributes:
        All attributes inherited from PlannerEvaluationWorkflow.

    Example:
        >>> from pathlib import Path
        >>> workflow = PlannerEvaluationLocalWorkflow(
        ...     cache_dir_path=Path("./results"),
        ...     experiment_name="Local_Evaluation",
        ...     n_jobs=-1,
        ... )
    """

    def _get_task_manager_config(self) -> TaskManagerConfig:
        return JoblibConfig(n_jobs=self.n_jobs)


class PlannerEvaluationDaskWorkflow(PlannerEvaluationWorkflow):
    """Workflow for Dask distributed planner evaluation.

    Attributes:
        n_workers: Number of Dask workers.
        scheduler_address: Optional address of existing Dask scheduler.
        cache_size: Size of cache in bytes.
        clear_cache_on_start: Whether to clear cache at startup.
        All other attributes inherited from PlannerEvaluationWorkflow.

    Example:
        >>> from pathlib import Path
        >>> workflow = PlannerEvaluationDaskWorkflow(
        ...     cache_dir_path=Path("./results"),
        ...     experiment_name="Dask_Evaluation",
        ...     n_workers=4,
        ... )
    """

    def __init__(
        self,
        cache_dir_path: Path,
        experiment_name: str,
        n_workers: int = 4,
        scheduler_address: Optional[str] = None,
        cache_size: int = int(2e9),
        clear_cache_on_start: bool = False,
        debug: bool = False,
        n_jobs: int = 1,
        enable_profiling: bool = False,
        verbose: bool = True,
        alpha: float = 0.1,
        confidence_interval_level: float = 0.95,
        cache_visualizations: bool = True,
    ):
        """Initialize Dask workflow.

        Args:
            cache_dir_path: Directory for caching results.
            experiment_name: Name of the experiment.
            n_workers: Number of Dask worker processes.
            scheduler_address: Address of existing Dask scheduler (None for local cluster).
            cache_size: Size of cache in bytes.
            clear_cache_on_start: Whether to clear cache at startup.
            debug: Enable debug mode.
            enable_profiling: Enable profiling.
            verbose: Enable verbose logging.
            alpha: Significance level for statistical tests.
            confidence_interval_level: Confidence level for intervals.
            cache_visualizations: Whether to cache visualizations.
        """
        self.n_workers = n_workers
        self.scheduler_address = scheduler_address
        self.cache_size = cache_size
        self.clear_cache_on_start = clear_cache_on_start

        super().__init__(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=debug,
            n_jobs=n_workers,  # TODO: bug here.
            enable_profiling=enable_profiling,
            verbose=verbose,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            cache_visualizations=cache_visualizations,
        )

    def _get_task_manager_config(self) -> TaskManagerConfig:
        return DaskConfig(
            n_workers=self.n_workers,
            scheduler_address=self.scheduler_address,
            cache_size=self.cache_size,
            clear_cache_on_start=self.clear_cache_on_start,
        )


class PlannerEvaluationPBSWorkflow(PlannerEvaluationWorkflow):
    """Workflow for PBS cluster planner evaluation.

    Attributes:
        queue: PBS queue name.
        n_workers: Number of PBS workers.
        cores: Cores per worker.
        memory: Memory per worker.
        processes: Processes per worker.
        walltime: Maximum walltime.
        job_extra: Additional PBS job parameters.
        enable_dashboard: Whether to enable the Dask dashboard.
        dashboard_address: Address to bind the dashboard to.
        dashboard_port: Port for the Dask dashboard.
        dashboard_prefix: URL prefix for dashboard (useful with reverse proxies).
        All other attributes inherited from PlannerEvaluationWorkflow.

    Example:
        >>> from pathlib import Path
        >>> workflow = PlannerEvaluationPBSWorkflow(
        ...     cache_dir_path=Path("./results"),
        ...     experiment_name="PBS_Evaluation",
        ...     queue="short",
        ...     n_workers=10,
        ...     cores=4,
        ...     memory="8GB",
        ... )
    """

    def __init__(
        self,
        cache_dir_path: Path,
        experiment_name: str,
        queue: str = "short",
        n_workers: int = 4,
        cores: int = 1,
        memory: str = "4GB",
        processes: int = 1,
        walltime: str = "03:00:00",
        job_extra: Optional[List[str]] = None,
        debug: bool = False,
        enable_profiling: bool = False,
        verbose: bool = True,
        alpha: float = 0.1,
        confidence_interval_level: float = 0.95,
        cache_visualizations: bool = True,
        enable_dashboard: bool = True,
        dashboard_address: str = "0.0.0.0",
        dashboard_port: int = 8787,
        dashboard_prefix: Optional[str] = None,
    ):
        """Initialize PBS workflow.

        Args:
            cache_dir_path: Directory for caching results.
            experiment_name: Name of the experiment.
            queue: PBS queue name.
            n_workers: Number of PBS workers.
            cores: Cores to allocate per worker.
            memory: Memory per worker.
            processes: Processes per worker.
            walltime: Maximum walltime.
            job_extra: Additional PBS job parameters.
            debug: Enable debug mode.
            enable_profiling: Enable profiling.
            verbose: Enable verbose logging.
            alpha: Significance level for statistical tests.
            confidence_interval_level: Confidence level for intervals.
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
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=debug,
            n_jobs=cores,  # TODO: bug here.
            enable_profiling=enable_profiling,
            verbose=verbose,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            cache_visualizations=cache_visualizations,
        )

    def _get_task_manager_config(self) -> TaskManagerConfig:
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
