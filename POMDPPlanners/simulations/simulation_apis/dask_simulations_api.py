# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.core.simulation import EnvironmentRunParams
from POMDPPlanners.core.simulation.simulation_configs import PlannerGenerator
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterRunParams,
    HyperParamPlannerConfigGenerator,
    OptimizedPolicyResult,
    ParallelizationLevel,
)
from POMDPPlanners.simulations.workflows.hyperparameter_tuning_evaluation_workflows import (
    OptimizationEvaluationDaskWorkflow,
)
from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import (
    HyperParameterOptimizer,
)
from POMDPPlanners.simulations.simulations_deployment.run_progress import (
    NotificationConfig,
)
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
    DaskConfig,
)
from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.configs.experiment_configs import (
    PolicyHyperparameterOptimizationExperimentConfigCreator,
    AllHyperparameterBenchmarksExperimentConfigCreator,
    AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator,
)
from POMDPPlanners.utils.logger import get_logger
from POMDPPlanners.simulations.simulation_apis.simulations_api_interface import (
    SimulationsAPIInterface,
)
from POMDPPlanners.simulations.workflows.planner_evaluation_workflow import (
    PlannerEvaluationDaskWorkflow,
)


class DaskSimulationsAPI(SimulationsAPIInterface):
    """High-level API for running POMDP simulation experiments using Dask.

    This class provides a simplified interface for running POMDP simulations with
    distributed execution using Dask for distributed computing. It wraps the
    POMDPSimulator class and provides convenient methods for remote simulation
    workflows with Dask cluster support.

    Key Features:
    - Remote execution using Dask for distributed computation
    - Support for existing Dask clusters or automatic local cluster creation
    - Debug mode with reduced episodes for quick testing
    - Automatic profiling and performance analysis
    - MLflow experiment tracking and result management

    Example:
        >>> from pathlib import Path
        >>> from POMDPPlanners.simulations.simulation_apis.dask_simulations_api import DaskSimulationsAPI
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> from POMDPPlanners.core.simulation import EnvironmentRunParams
        >>> # Initialize the API
        >>> api = DaskSimulationsAPI(debug=False)
        >>> # Create environment
        >>> tiger = TigerPOMDP(discount_factor=0.95)
        >>> tiger.name
        'TigerPOMDP'
        >>> # Create policy
        >>> policy = POMCP(
        ...     environment=tiger,
        ...     discount_factor=0.95,
        ...     depth=5,
        ...     exploration_constant=1.0,
        ...     name="POMCP_Distributed",
        ...     n_simulations=20
        ... )
        >>> policy.name
        'POMCP_Distributed'
        >>> # Configure simulation parameters
        >>> env_params = EnvironmentRunParams(
        ...     environment=tiger,
        ...     belief=get_initial_belief(tiger, n_particles=10),
        ...     policies=[policy],
        ...     num_episodes=2,
        ...     num_steps=3
        ... )
        >>> env_params.num_episodes
        2
    """

    def __init__(
        self,
        cache_dir_path: Optional[Path] = None,
        debug: bool = False,
        scheduler_address: Optional[str] = None,
        cache_size: int = int(2e9),
        notification_config: Optional[NotificationConfig] = None,
    ):
        """Initialize the DaskSimulationsAPI.

        Args:
            cache_dir_path: Optional path for storing simulation results and logs
            debug: Whether to enable debug-level logging output
            notification_config: Optional :class:`NotificationConfig`. When
                ``None``, defaults to :meth:`NotificationConfig.from_env`.
        """
        self.logger = get_logger(
            name="dask_simulations_api", output_dir=cache_dir_path, debug=debug
        )
        self.notification_config: NotificationConfig = (
            notification_config
            if notification_config is not None
            else NotificationConfig.from_env()
        )
        self.logger.info("Initialized DaskSimulationsAPI")

        self.scheduler_address = scheduler_address
        self.cache_size = cache_size

    def run_multiple_environments_and_policies(
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float,
        experiment_name: str = "POMDP_Planning_Comparison",
        debug: bool = False,
        scheduler_address: Optional[str] = None,
        n_jobs: int = -1,
        cache_dir_path: Optional[Path] = None,
        clear_cache_on_start: bool = False,
        enable_profiling: bool = False,
        profiling_output_limit: int = 50,
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Run POMDP simulations remotely using Dask for distributed execution.

        This method executes POMDP simulations across multiple machines using Dask
        for distributed computing. It's ideal for large-scale experiments that require
        significant computational resources and can benefit from distributed processing.

        Args:
            environment_run_params: List of environment configurations for simulation.
                Each configuration specifies an environment, belief state, policies,
                number of episodes, and number of steps per episode.
            alpha: Statistical significance level for confidence intervals (e.g., 0.05 for 95% CI).
                Used for computing risk metrics like Conditional Value at Risk (CVaR).
            confidence_interval_level: Confidence level for statistical analysis (e.g., 0.95).
                Determines the width of confidence intervals for performance metrics.
            experiment_name: Name for the experiment and MLflow tracking. Used to organize
                results and enable comparison across different experimental runs.
            debug: Whether to enable debug-level logging output. When True, provides
                detailed information about simulation progress and internal operations.
            scheduler_address: Address of the Dask scheduler for distributed execution.
                If None, creates a local Dask cluster. Format: "tcp://scheduler-ip:port".
            n_jobs: Number of worker processes for distributed execution. Use -1 to use
                all available workers, or specify a positive integer for a specific number.
            cache_dir_path: Optional path for storing simulation results, logs, and artifacts.
                If None, results are stored in the current working directory.
            clear_cache_on_start: Whether to clear existing cache before starting simulation.
                Useful for ensuring clean runs when debugging or testing.
            enable_profiling: Whether to enable performance profiling using cProfile.
                Generates detailed timing information for optimization analysis.
            profiling_output_limit: Maximum number of profiling entries to display
                when profiling is enabled. Helps focus on the most time-consuming operations.

        Returns:
            Tuple containing:
                - Dict[str, Dict[str, list]]: Raw simulation results organized by environment
                  name, then policy name, containing lists of History objects for each episode.
                - pd.DataFrame: Statistical summary with confidence intervals, performance
                  metrics, and policy configuration details for analysis and comparison.

        Example:
            Running a distributed simulation with a remote Dask cluster:

            >>> from pathlib import Path
            >>> from POMDPPlanners.simulations.simulation_apis.dask_simulations_api import DaskSimulationsAPI
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
            >>> from POMDPPlanners.core.belief import get_initial_belief
            >>> from POMDPPlanners.core.simulation import EnvironmentRunParams
            >>> # Initialize the API
            >>> api = DaskSimulationsAPI(debug=False)
            >>> # Create environment and policy
            >>> tiger = TigerPOMDP(discount_factor=0.95)
            >>> policy = POMCP(
            ...     environment=tiger,
            ...     discount_factor=0.95,
            ...     depth=5,
            ...     exploration_constant=1.0,
            ...     name="POMCP_Distributed",
            ...     n_simulations=20
            ... )
            >>> # Configure simulation parameters for distributed execution
            >>> environment_run_params = [
            ...     EnvironmentRunParams(
            ...         environment=tiger,
            ...         belief=get_initial_belief(tiger, n_particles=10),
            ...         policies=[policy],
            ...         num_episodes=2,
            ...         num_steps=3
            ...     )
            ... ]
            >>> # Run distributed simulation (skip actual execution)
            >>> results, statistics_df = api.run_multiple_environments_and_policies(
            ...     environment_run_params=environment_run_params,
            ...     alpha=0.05,
            ...     confidence_interval_level=0.95,
            ...     experiment_name="Distributed_Tiger_Study",
            ...     scheduler_address=None,  # Local cluster
            ...     n_jobs=1,
            ...     enable_profiling=False
            ... ) # doctest: +SKIP
            >>> # Check simulation results
            >>> len(statistics_df) >= 1  # doctest: +SKIP
            True
        """
        self.logger.info(
            "Starting simulation run with %d environment configurations",
            len(environment_run_params),
        )
        self.logger.debug(
            "Parameters: alpha=%s, confidence_interval=%s, n_jobs=%s",
            alpha,
            confidence_interval_level,
            n_jobs,
        )

        task_manager_config = DaskConfig(
            n_workers=n_jobs,
            scheduler_address=scheduler_address,
            clear_cache_on_start=clear_cache_on_start,
        )

        with POMDPSimulator(
            task_manager_config=task_manager_config,
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=debug,
            enable_profiling=enable_profiling,
            profiling_output_limit=profiling_output_limit,
            notification_config=self.notification_config,
        ) as simulator:
            self.logger.info("Running simulation comparison")
            results = simulator.compare_multiple_environments_policies(
                environment_run_params=environment_run_params,
                alpha=alpha,
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs,
                cache_visualizations=True,
            )
            self.logger.info("Simulation run completed")
            return results

    def run_all_benchmark_environments_on_planner_generators(
        self,
        generators: Sequence[PlannerGenerator],
        n_particles: int = 30,
        num_episodes: int = 10,
        num_steps: int = 20,
        alpha: float = 0.1,
        confidence_interval_level: float = 0.95,
        experiment_name: str = "All_Benchmark_Environments_On_Planner_Generators",
        n_jobs: int = -1,
        cache_dir_path: Optional[Path] = None,
        clear_cache_on_start: bool = False,
        enable_profiling: bool = False,
        profiling_output_limit: int = 50,
        cache_visualizations: bool = True,
        is_risk_averse: bool = False,
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Run all benchmark environments on planner generators using Dask.

        This method runs evaluation experiments across all compatible benchmark environments
        for a given set of planner generators using Dask for distributed execution.

        Args:
            generators: Sequence of planner generators to evaluate.
            n_particles: Number of particles for belief representation.
            num_episodes: Number of episodes for evaluation.
            num_steps: Maximum steps per episode for evaluation.
            alpha: Significance level for statistical tests.
            confidence_interval_level: Confidence level for intervals.
            experiment_name: Name for the experiment.
            n_jobs: Number of parallel jobs for execution.
            cache_dir_path: Optional path for storing results.
            clear_cache_on_start: Whether to clear cache at startup.
            enable_profiling: Whether to enable performance profiling.
            profiling_output_limit: Maximum number of profiling entries to display.
            cache_visualizations: Whether to cache visualizations.
            scheduler_address: Address of existing Dask scheduler (None for local cluster).
            cache_size: Size of Dask cache in bytes.
            is_risk_averse: Whether to run risk-averse benchmark.
        Returns:
            Tuple of results dictionary and DataFrame.

        Raises:
            ValueError: If generators list is empty or contains invalid objects.
        """
        if len(generators) == 0:
            raise ValueError("generators list cannot be empty")

        if not all(isinstance(gen, PlannerGenerator) for gen in generators):
            raise ValueError("generators list must contain only PlannerGenerator objects")

        if cache_dir_path is None:
            cache_dir_path = Path("./all_benchmark_environments_on_planner_generators_dask_results")

        creator = AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator(
            generators=generators,
            n_particles=n_particles,
            num_episodes=num_episodes,
            num_steps=num_steps,
            is_risk_averse=is_risk_averse,
        )

        workflow = PlannerEvaluationDaskWorkflow(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            n_workers=n_jobs,
            scheduler_address=self.scheduler_address,
            cache_size=self.cache_size,
            clear_cache_on_start=clear_cache_on_start,
            debug=False,
            n_jobs=n_jobs,
            enable_profiling=enable_profiling,
            verbose=True,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            cache_visualizations=cache_visualizations,
        )
        configs = creator.get_experiment_configs()
        return workflow.evaluate(configs)

    def run_hyperparameter_optimization(
        self,
        environment_run_params: List[HyperParameterRunParams],
        experiment_name: str = "POMDP_Hyperparameter_Optimization",
        n_jobs: int = -1,
        cache_dir_path: Optional[Path] = None,
        clear_cache_on_start: bool = False,
        debug: bool = False,
        confidence_interval_level: float = 0.95,
        alpha: float = 0.05,
        use_queue_logger: bool = False,
        parallelization_level: ParallelizationLevel = ParallelizationLevel.OPTUNA_TRIALS,
        scheduler_address: Optional[str] = None,
        cache_size: int = int(2e9),
    ) -> List[OptimizedPolicyResult]:
        """Run hyperparameter optimization for POMDP policies using Optuna with Dask.

        This method provides a high-level interface for hyperparameter optimization
        using Dask for distributed execution. It wraps the HyperParameterOptimizer
        class and supports optimization of multiple environment-policy configurations
        with comprehensive MLflow tracking and statistical analysis.

        Args:
            environment_run_params: List of HyperParameterRunParams configurations.
            experiment_name: Name for the MLflow experiment tracking.
            n_jobs: Number of parallel workers for Dask execution.
            cache_dir_path: Optional path for storing optimization results.
            clear_cache_on_start: Whether to clear existing cache before starting.
            debug: Whether to enable debug-level logging output.
            confidence_interval_level: Confidence level for statistical analysis.
            alpha: Significance level for statistical tests.
            use_queue_logger: Whether to use queue-based logging.
            scheduler_address: Address of existing Dask scheduler (None for local cluster).
            cache_size: Size of Dask cache in bytes.

        Returns:
            List[OptimizedPolicyResult]: List of optimization results.

        Raises:
            ValueError: If any configuration contains invalid parameters.
            RuntimeError: If optimization fails for any configuration.
        """
        self.logger.info(
            "Starting hyperparameter optimization for %d configurations",
            len(environment_run_params),
        )
        self.logger.debug(
            "Parameters: experiment_name=%s, n_jobs=%s, confidence_interval=%s, alpha=%s",
            experiment_name,
            n_jobs,
            confidence_interval_level,
            alpha,
        )

        # Set up cache directory
        if cache_dir_path is None:
            cache_dir_path = Path("./hyperparameter_optimization_results")

        cache_dir_path.mkdir(parents=True, exist_ok=True)

        task_manager_config = DaskConfig(
            n_workers=n_jobs,
            scheduler_address=scheduler_address,
            cache_size=cache_size,
            clear_cache_on_start=clear_cache_on_start,
        )

        optimizer = HyperParameterOptimizer(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            n_jobs=n_jobs,
            task_manager_config=task_manager_config,
            confidence_interval_level=confidence_interval_level,
            alpha=alpha,
            use_queue_logger=use_queue_logger,
            parallelization_level=parallelization_level,
            notification_config=self.notification_config,
        )

        try:
            self.logger.info("Running hyperparameter optimization with Dask")
            results = optimizer.optimize(environment_run_params)

            self.logger.info(
                "Hyperparameter optimization completed successfully. "
                "Optimized %d out of %d configurations",
                len(results),
                len(environment_run_params),
            )

            for i, result in enumerate(results):
                self.logger.info(
                    "Configuration %d: %s with %s - Best parameters: %s",
                    i + 1,
                    result.environment.__class__.__name__,
                    result.policy.__class__.__name__,
                    result.chosen_hyper_parameters,
                )

            return results

        except Exception as e:
            self.logger.error("Hyperparameter optimization failed: %s", e)
            raise RuntimeError(f"Hyperparameter optimization failed: {e}") from e

        finally:
            try:
                optimizer.cleanup()
            except Exception as cleanup_error:  # pylint: disable=broad-exception-caught
                self.logger.warning("Error during optimizer cleanup: %s", cleanup_error)

    def run_hyperparameter_tuning_experiment_with_benchmarks(
        self,
        generators: Sequence[HyperParamPlannerConfigGenerator],
        particles: int = 30,
        num_episodes: int = 10,
        num_steps: int = 20,
        n_trials: int = 100,
        discount_factor: float = 0.95,
        time_out_in_seconds: float = 3.0,
        evaluation_episodes: int = 3,
        evaluation_steps: int = 6,
        evaluation_n_jobs: int = 1,
        optimization_n_jobs: int = -1,
        is_risk_averse: bool = False,
        confidence_interval_level: float = 0.95,
        alpha: float = 0.05,
        cache_dir_path: Optional[Path] = None,
        experiment_name: str = "Comprehensive_Benchmark",
        debug: bool = False,
        cache_visualizations: bool = True,
        scheduler_address: Optional[str] = None,
        cache_size: int = int(2e9),
        clear_cache_on_start: bool = False,
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Run comprehensive benchmark with hyperparameter optimization using Dask.

        This method runs hyperparameter optimization followed by policy evaluation
        using Dask for distributed execution. It optimizes for average return across
        all configured environments and benchmark planners.

        Args:
            generators: Hyperparameter configuration generators list.
            particles: Number of particles for belief representation.
            num_episodes: Number of episodes for optimization.
            num_steps: Maximum steps per episode for optimization.
            n_trials: Number of optimization trials.
            discount_factor: Discount factor for the MDP.
            time_out_in_seconds: Timeout for planner execution.
            evaluation_episodes: Number of episodes for evaluation.
            evaluation_steps: Maximum steps per episode for evaluation.
            evaluation_n_jobs: Number of parallel jobs for evaluation.
            optimization_n_jobs: Number of parallel workers for optimization.
            is_risk_averse: Whether to run risk-averse benchmark.
            confidence_interval_level: Confidence level for intervals.
            alpha: Significance level for statistical tests.
            cache_dir_path: Optional path for storing results.
            experiment_name: Name for the experiment.
            debug: Enable debug mode.
            cache_visualizations: Whether to cache visualizations.
            scheduler_address: Address of existing Dask scheduler (None for local cluster).
            cache_size: Size of Dask cache in bytes.
            clear_cache_on_start: Whether to clear cache at startup.

        Returns:
            Tuple of results dictionary and DataFrame.
        """
        self.logger.info("Starting comprehensive benchmark with Dask execution")

        if cache_dir_path is None:
            cache_dir_path = Path("./comprehensive_benchmark_results")

        creator = PolicyHyperparameterOptimizationExperimentConfigCreator(
            generators=generators,
            particles=particles,
            num_episodes=num_episodes,
            num_steps=num_steps,
            n_trials=n_trials,
            discount_factor=discount_factor,
            time_out_in_seconds=time_out_in_seconds,
            is_risk_averse=is_risk_averse,
        )
        configs = creator.get_experiment_configs()

        workflow = OptimizationEvaluationDaskWorkflow(
            cache_dir=cache_dir_path,
            experiment_name=experiment_name,
            n_workers=optimization_n_jobs,
            scheduler_address=scheduler_address,
            cache_size=cache_size,
            clear_cache_on_start=clear_cache_on_start,
            evaluation_episodes=evaluation_episodes,
            evaluation_steps=evaluation_steps,
            evaluation_n_jobs=evaluation_n_jobs,
            confidence_interval_level=confidence_interval_level,
            alpha=alpha,
            debug=debug,
            verbose=True,
            cache_visualizations=cache_visualizations,
        )

        return workflow.optimize_and_evaluate(configs)

    def run_optimize_and_evaluate(
        self,
        configs: List[HyperParameterRunParams],
        evaluation_episodes: int = 100,
        evaluation_steps: int = 100,
        evaluation_n_jobs: int = 1,
        optimization_n_jobs: int = -1,
        confidence_interval_level: float = 0.95,
        alpha: float = 0.05,
        cache_dir_path: Optional[Path] = None,
        experiment_name: str = "Optimize_And_Evaluate",
        debug: bool = False,
        cache_visualizations: bool = True,
        scheduler_address: Optional[str] = None,
        cache_size: int = int(2e9),
        clear_cache_on_start: bool = False,
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Run hyperparameter optimization and evaluation using Dask.

        This method runs hyperparameter optimization for the provided configurations,
        then evaluates the optimized policies using Dask for distributed execution.

        Args:
            configs: List of hyperparameter run configurations.
            evaluation_episodes: Number of episodes for evaluation.
            evaluation_steps: Maximum steps per episode for evaluation.
            evaluation_n_jobs: Number of parallel jobs for evaluation.
            optimization_n_jobs: Number of parallel workers for optimization.
            confidence_interval_level: Confidence level for intervals.
            alpha: Significance level for statistical tests.
            cache_dir_path: Optional path for storing results.
            experiment_name: Name for the experiment.
            debug: Enable debug mode.
            cache_visualizations: Whether to cache visualizations.
            scheduler_address: Address of existing Dask scheduler (None for local cluster).
            cache_size: Size of Dask cache in bytes.
            clear_cache_on_start: Whether to clear cache at startup.

        Returns:
            Tuple of results dictionary and DataFrame.

        Raises:
            ValueError: If configs list is empty.
        """
        self.logger.info("Starting optimize and evaluate workflow with Dask execution")

        if not configs:
            raise ValueError("configs list cannot be empty")

        if cache_dir_path is None:
            cache_dir_path = Path("./optimize_and_evaluate_results")

        workflow = OptimizationEvaluationDaskWorkflow(
            cache_dir=cache_dir_path,
            experiment_name=experiment_name,
            n_workers=optimization_n_jobs,
            scheduler_address=scheduler_address,
            cache_size=cache_size,
            clear_cache_on_start=clear_cache_on_start,
            evaluation_episodes=evaluation_episodes,
            evaluation_steps=evaluation_steps,
            evaluation_n_jobs=evaluation_n_jobs,
            confidence_interval_level=confidence_interval_level,
            alpha=alpha,
            debug=debug,
            verbose=True,
            cache_visualizations=cache_visualizations,
        )

        return workflow.optimize_and_evaluate(configs)

    def run_all_hyperparameter_benchmarks(
        self,
        policy_space_info: PolicySpaceInfo,
        particles: int = 30,
        num_episodes: int = 10,
        num_steps: int = 20,
        n_trials: int = 100,
        discount_factor: float = 0.95,
        time_out_in_seconds: float = 3.0,
        evaluation_episodes: int = 3,
        evaluation_steps: int = 6,
        evaluation_n_jobs: int = 1,
        optimization_n_jobs: int = -1,
        is_risk_averse: bool = False,
        confidence_interval_level: float = 0.95,
        alpha: float = 0.05,
        cache_dir_path: Optional[Path] = None,
        experiment_name: str = "All_Hyperparameter_Benchmarks",
        debug: bool = False,
        cache_visualizations: bool = True,
        scheduler_address: Optional[str] = None,
        cache_size: int = int(2e9),
        clear_cache_on_start: bool = False,
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Run all hyperparameter benchmarks with optimization using Dask.

        This method runs hyperparameter optimization for all compatible environments
        and planners for a given policy space, followed by evaluation using Dask
        for distributed execution.

        Args:
            policy_space_info: Policy space information specifying action and observation
                space types for compatibility matching.
            particles: Number of particles for belief representation.
            num_episodes: Number of episodes for optimization.
            num_steps: Maximum steps per episode for optimization.
            n_trials: Number of optimization trials.
            discount_factor: Discount factor for the MDP.
            time_out_in_seconds: Timeout for planner execution.
            evaluation_episodes: Number of episodes for evaluation.
            evaluation_steps: Maximum steps per episode for evaluation.
            evaluation_n_jobs: Number of parallel jobs for evaluation.
            optimization_n_jobs: Number of parallel workers for optimization.
            is_risk_averse: Whether to run risk-averse benchmark.
            confidence_interval_level: Confidence level for intervals.
            alpha: Significance level for statistical tests.
            cache_dir_path: Optional path for storing results.
            experiment_name: Name for the experiment.
            debug: Enable debug mode.
            cache_visualizations: Whether to cache visualizations.
            scheduler_address: Address of existing Dask scheduler (None for local cluster).
            cache_size: Size of Dask cache in bytes.
            clear_cache_on_start: Whether to clear cache at startup.

        Returns:
            Tuple of results dictionary and DataFrame.
        """
        self.logger.info("Starting all hyperparameter benchmarks with Dask execution")

        if cache_dir_path is None:
            cache_dir_path = Path("./all_hyperparameter_benchmarks_results")

        creator = AllHyperparameterBenchmarksExperimentConfigCreator(
            policy_space_info=policy_space_info,
            particles=particles,
            num_episodes=num_episodes,
            num_steps=num_steps,
            n_trials=n_trials,
            discount_factor=discount_factor,
            time_out_in_seconds=time_out_in_seconds,
            is_risk_averse=is_risk_averse,
            debug=debug,
        )
        configs = creator.get_experiment_configs()

        workflow = OptimizationEvaluationDaskWorkflow(
            cache_dir=cache_dir_path,
            experiment_name=experiment_name,
            n_workers=optimization_n_jobs,
            scheduler_address=scheduler_address,
            cache_size=cache_size,
            clear_cache_on_start=clear_cache_on_start,
            evaluation_episodes=evaluation_episodes,
            evaluation_steps=evaluation_steps,
            evaluation_n_jobs=evaluation_n_jobs,
            confidence_interval_level=confidence_interval_level,
            alpha=alpha,
            debug=debug,
            verbose=True,
            cache_visualizations=cache_visualizations,
        )

        return workflow.optimize_and_evaluate(configs)
