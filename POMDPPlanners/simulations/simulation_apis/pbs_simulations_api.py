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
    OptimizationEvaluationPBSWorkflow,
)
from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import (
    HyperParameterOptimizer,
)
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import PBSConfig
from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.simulations.simulation_apis.simulations_api_interface import (
    SimulationsAPIInterface,
)
from POMDPPlanners.configs.experiment_configs import (
    PolicyHyperparameterOptimizationExperimentConfigCreator,
    AllHyperparameterBenchmarksExperimentConfigCreator,
    AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator,
)
from POMDPPlanners.utils.logger import get_logger
from POMDPPlanners.simulations.workflows.planner_evaluation_workflow import (
    PlannerEvaluationPBSWorkflow,
)


class PBSSimulationsAPI(SimulationsAPIInterface):
    """High-level API for running POMDP simulations on PBS clusters.

    This class provides a simplified interface for running POMDP simulations on
    PBS (Portable Batch System) clusters, commonly used in high-performance computing
    environments. It automatically manages job submission, scaling, and distributed
    execution across cluster nodes.

    Key Features:
    - PBS cluster execution for large-scale experiments
    - Automatic job submission and scaling
    - Dask dashboard support for monitoring
    - Configurable resource allocation (cores, memory, walltime)
    - MLflow experiment tracking and result management

    Example:
        >>> from pathlib import Path
        >>> from POMDPPlanners.simulations.simulation_apis.pbs_simulations_api import PBSSimulationsAPI
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> from POMDPPlanners.core.simulation import EnvironmentRunParams
        >>> # Initialize the PBS API with cluster configuration
        >>> api = PBSSimulationsAPI(
        ...     queue="batch",
        ...     n_workers=4,
        ...     cores=2,
        ...     memory="8GB",
        ...     walltime="02:00:00",
        ...     debug=False
        ... )
        >>> # Create environment
        >>> tiger = TigerPOMDP(discount_factor=0.95)
        >>> # Create policy
        >>> policy = POMCP(
        ...     environment=tiger,
        ...     discount_factor=0.95,
        ...     depth=5,
        ...     exploration_constant=1.0,
        ...     name="POMCP_Cluster",
        ...     n_simulations=20
        ... )
        >>> # Configure simulation parameters
        >>> env_params = EnvironmentRunParams(
        ...     environment=tiger,
        ...     belief=get_initial_belief(tiger, n_particles=10),
        ...     policies=[policy],
        ...     num_episodes=100,
        ...     num_steps=50
        ... )
    """

    def __init__(
        self,
        queue: str,
        n_workers: int = 4,
        cores: int = 1,
        memory: str = "4GB",
        processes: int = 1,
        walltime: str = "01:00:00",
        job_extra: Optional[List[str]] = None,
        enable_dashboard: bool = True,
        dashboard_address: str = "0.0.0.0",
        dashboard_port: int = 8787,
        dashboard_prefix: Optional[str] = None,
        cache_dir_path: Optional[Path] = None,
        debug: bool = False,
    ):
        """Initialize the PBSSimulationsAPI.

        Args:
            queue: PBS queue name to submit jobs to
            n_workers: Number of worker jobs to submit to PBS cluster
            cores: Number of CPU cores per PBS job
            memory: Memory allocation per PBS job (e.g., "4GB", "8GB")
            processes: Number of processes per PBS job
            walltime: Maximum runtime per job in HH:MM:SS format
            job_extra: Additional PBS directives as list of strings
            enable_dashboard: Whether to enable the Dask dashboard for monitoring
            dashboard_address: Address to bind the dashboard to (e.g., "0.0.0.0", "127.0.0.1")
            dashboard_port: Port for the Dask dashboard (default: 8787)
            dashboard_prefix: URL prefix for dashboard (useful with reverse proxies)
            cache_dir_path: Optional path for storing simulation results and logs
            debug: Whether to enable debug-level logging output
        """
        # Store PBS-specific configuration
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
        self.cache_dir_path = cache_dir_path
        self.debug = debug

        self.logger = get_logger(name="pbs_simulations_api", output_dir=cache_dir_path, debug=debug)
        self.logger.info("Initialized PBSSimulationsAPI")
        self.logger.debug(
            "PBS Config: queue=%s, n_workers=%s, cores=%s, memory=%s, walltime=%s",
            queue,
            n_workers,
            cores,
            memory,
            walltime,
        )

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
        """Run simulations using PBS cluster computing.

        This method executes POMDP simulations on a PBS (Portable Batch System) cluster,
        which is commonly used in high-performance computing environments. It automatically
        manages job submission, scaling, and distributed execution across cluster nodes.

        Args:
            environment_run_params: List of environment configurations for simulation
            alpha: Statistical significance level for confidence intervals (e.g., 0.05 for 95% CI)
            confidence_interval_level: Confidence level for statistical analysis (e.g., 0.95)
            experiment_name: Name for the experiment and MLflow tracking
            debug: Whether to enable debug-level logging output
            n_jobs: Number of parallel jobs for simulation execution. Use -1 to use
                all available workers (n_workers × processes), or specify a positive
                integer for a specific number of parallel jobs.
            cache_dir_path: Optional path for storing simulation results and logs
            clear_cache_on_start: Whether to clear cache before starting simulation
            enable_profiling: Whether to enable performance profiling
            profiling_output_limit: Maximum number of profiling entries to display

        Returns:
            Tuple containing:
                - Dict[str, Dict[str, list]]: Raw simulation results organized by environment and policy
                - pd.DataFrame: Statistical summary with confidence intervals and performance metrics

        Example:
            Running a large-scale simulation study on PBS cluster:

            >>> from pathlib import Path
            >>> from POMDPPlanners.simulations.simulation_apis.pbs_simulations_api import PBSSimulationsAPI
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
            >>> from POMDPPlanners.core.belief import get_initial_belief
            >>> from POMDPPlanners.core.simulation import EnvironmentRunParams
            >>> # Initialize the PBS API
            >>> api = PBSSimulationsAPI(
            ...     queue="batch",
            ...     n_workers=4,
            ...     cores=2,
            ...     memory="8GB",
            ...     walltime="02:00:00"
            ... )
            >>> # Create environment and policy
            >>> tiger = TigerPOMDP(discount_factor=0.95)
            >>> policy = POMCP(
            ...     environment=tiger,
            ...     discount_factor=0.95,
            ...     depth=5,
            ...     exploration_constant=1.0,
            ...     name="POMCP_ClusterTest",
            ...     n_simulations=20
            ... )
            >>> # Configure simulation for cluster execution
            >>> environment_run_params = [
            ...     EnvironmentRunParams(
            ...         environment=tiger,
            ...         belief=get_initial_belief(tiger, n_particles=10),
            ...         policies=[policy],
            ...         num_episodes=100,
            ...         num_steps=50
            ...     )
            ... ]
            >>> # Run on PBS cluster (skip actual execution)
            >>> results, statistics_df = api.run_multiple_environments_and_policies_pbs_run(
            ...     environment_run_params=environment_run_params,
            ...     alpha=0.05,
            ...     confidence_interval_level=0.95,
            ...     experiment_name="Large_Scale_Tiger_Study",
            ...     enable_profiling=False
            ... ) # doctest: +SKIP
            >>> # Check simulation results
            >>> len(statistics_df) >= 1  # doctest: +SKIP
            True
        """
        self.logger.info(
            "Starting PBS cluster simulation with %d environment configurations",
            len(environment_run_params),
        )
        self.logger.debug(
            "Simulation Parameters: alpha=%s, confidence_interval=%s",
            alpha,
            confidence_interval_level,
        )

        task_manager_config = PBSConfig(
            queue=self.queue,
            n_workers=self.n_workers,
            cores=self.cores,
            memory=self.memory,
            processes=self.processes,
            walltime=self.walltime,
            job_extra=self.job_extra,
            clear_cache_on_start=clear_cache_on_start,
            enable_dashboard=self.enable_dashboard,
            dashboard_address=self.dashboard_address,
            dashboard_port=self.dashboard_port,
            dashboard_prefix=self.dashboard_prefix,
        )

        with POMDPSimulator(
            task_manager_config=task_manager_config,
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=debug,
            enable_profiling=enable_profiling,
            profiling_output_limit=profiling_output_limit,
            use_queue_logger=True,
        ) as simulator:
            self.logger.info("Running PBS cluster simulation comparison")
            results = simulator.compare_multiple_environments_policies(
                environment_run_params=environment_run_params,
                alpha=alpha,
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs,
                cache_visualizations=True,
            )
            self.logger.info("PBS cluster simulation run completed")
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
        """Run all benchmark environments on planner generators using PBS cluster.

        This method runs evaluation experiments across all compatible benchmark environments
        for a given set of planner generators using PBS cluster computing.

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
            cache_dir_path = Path("./all_benchmark_environments_on_planner_generators_pbs_results")

        creator = AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator(
            generators=generators,
            n_particles=n_particles,
            num_episodes=num_episodes,
            num_steps=num_steps,
            is_risk_averse=is_risk_averse,
        )

        workflow = PlannerEvaluationPBSWorkflow(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            queue=self.queue,
            n_workers=self.n_workers,
            cores=self.cores,
            memory=self.memory,
            processes=self.processes,
            walltime=self.walltime,
            job_extra=self.job_extra,
            debug=False,
            enable_profiling=enable_profiling,
            verbose=True,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            cache_visualizations=cache_visualizations,
            enable_dashboard=self.enable_dashboard,
            dashboard_address=self.dashboard_address,
            dashboard_port=self.dashboard_port,
            dashboard_prefix=self.dashboard_prefix,
        )
        return workflow.evaluate(creator.get_experiment_configs())

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
    ) -> List[OptimizedPolicyResult]:
        """Run hyperparameter optimization for POMDP policies using Optuna on PBS cluster.

        This method provides hyperparameter optimization on PBS cluster infrastructure,
        utilizing distributed computing resources for parallel trial evaluation.

        Args:
            environment_run_params: List of HyperParameterRunParams configurations,
                each specifying an environment, policy class, hyperparameter ranges,
                and optimization settings.
            experiment_name: Name for the MLflow experiment tracking.
            n_jobs: Number of parallel jobs for episode execution. Use -1 to use
                all available workers (n_workers × processes).
            cache_dir_path: Optional path for storing optimization results and logs.
            clear_cache_on_start: Whether to clear existing cache before starting.
            debug: Whether to enable debug-level logging output.
            confidence_interval_level: Confidence level for statistical analysis.
            alpha: Significance level for statistical tests.
            use_queue_logger: Whether to use queue-based logging for distributed execution.

        Returns:
            List[OptimizedPolicyResult]: List of optimization results with optimized
                policies and their best hyperparameters.
        """
        self.logger.info(
            "Starting hyperparameter optimization on PBS cluster with %d configurations",
            len(environment_run_params),
        )

        if cache_dir_path is None:
            cache_dir_path = Path("./hyperparameter_optimization_pbs_results")

        task_manager_config = PBSConfig(
            queue=self.queue,
            n_workers=self.n_workers,
            cores=self.cores,
            memory=self.memory,
            processes=self.processes,
            walltime=self.walltime,
            job_extra=self.job_extra,
            clear_cache_on_start=clear_cache_on_start,
            enable_dashboard=self.enable_dashboard,
            dashboard_address=self.dashboard_address,
            dashboard_port=self.dashboard_port,
            dashboard_prefix=self.dashboard_prefix,
        )

        optimizer = HyperParameterOptimizer(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            task_manager_config=task_manager_config,
            n_jobs=n_jobs,
            confidence_interval_level=confidence_interval_level,
            alpha=alpha,
            use_queue_logger=use_queue_logger,
            parallelization_level=parallelization_level,
        )

        self.logger.info("Running PBS cluster hyperparameter optimization")
        results = optimizer.optimize(environment_run_params)
        self.logger.info("PBS cluster hyperparameter optimization completed")
        return results

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
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Run comprehensive benchmark with hyperparameter optimization on PBS cluster.

        This method runs hyperparameter optimization followed by policy evaluation
        using PBS cluster computing. It optimizes for average return across
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
            optimization_n_jobs: Number of parallel jobs for optimization (-1 uses all cores).
            is_risk_averse: Whether to run risk-averse benchmark.
            confidence_interval_level: Confidence level for intervals.
            alpha: Significance level for statistical tests.
            cache_dir_path: Optional path for storing results.
            experiment_name: Name for the experiment.
            debug: Enable debug mode.
            cache_visualizations: Whether to cache visualizations.

        Returns:
            Tuple of results dictionary and DataFrame.
        """
        self.logger.info("Starting comprehensive benchmark with PBS cluster execution")

        if cache_dir_path is None:
            cache_dir_path = Path("./comprehensive_benchmark_pbs_results")

        creator = PolicyHyperparameterOptimizationExperimentConfigCreator(
            generators=generators,
            particles=particles,
            num_episodes=num_episodes,
            num_steps=num_steps,
            n_trials=n_trials,
            is_risk_averse=is_risk_averse,
            discount_factor=discount_factor,
            time_out_in_seconds=time_out_in_seconds,
        )
        configs = creator.get_experiment_configs()

        workflow = OptimizationEvaluationPBSWorkflow(
            cache_dir=cache_dir_path,
            experiment_name=experiment_name,
            queue=self.queue,
            n_workers=self.n_workers,
            cores=self.cores,
            memory=self.memory,
            processes=self.processes,
            walltime=self.walltime,
            job_extra=self.job_extra,
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
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Run hyperparameter optimization and evaluation on PBS cluster.

        This method runs hyperparameter optimization for the provided configurations,
        then evaluates the optimized policies using PBS cluster computing.

        Args:
            configs: List of hyperparameter run configurations.
            evaluation_episodes: Number of episodes for evaluation.
            evaluation_steps: Maximum steps per episode for evaluation.
            evaluation_n_jobs: Number of parallel jobs for evaluation.
            optimization_n_jobs: Number of parallel jobs for optimization (-1 uses all cores).
            confidence_interval_level: Confidence level for intervals.
            alpha: Significance level for statistical tests.
            cache_dir_path: Optional path for storing results.
            experiment_name: Name for the experiment.
            debug: Enable debug mode.
            cache_visualizations: Whether to cache visualizations.

        Returns:
            Tuple of results dictionary and DataFrame.

        Example:
            >>> from pathlib import Path
            >>> from POMDPPlanners.simulations.simulation_apis.pbs_simulations_api import PBSSimulationsAPI
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
            >>> from POMDPPlanners.core.belief import get_initial_belief
            >>> from POMDPPlanners.core.simulation import NumericalHyperParameter
            >>> from POMDPPlanners.core.simulation.hyperparameter_tuning import (
            ...     HyperParameterRunParams,
            ...     HyperParameterOptimizationDirection,
            ...     HyperParamPlannerConfig
            ... )
            >>> # Initialize the PBS API
            >>> api = PBSSimulationsAPI(
            ...     queue="test_queue",
            ...     n_workers=4,
            ...     cores=2,
            ...     memory="8GB",
            ...     walltime="02:00:00"
            ... )
            >>> # Create environment and initial belief
            >>> tiger = TigerPOMDP(discount_factor=0.95)
            >>> initial_belief = get_initial_belief(tiger, n_particles=10)
            >>> # Define hyperparameter optimization configurations
            >>> planner_config = HyperParamPlannerConfig(
            ...     policy_cls=POMCP,
            ...     hyper_parameters=[
            ...         NumericalHyperParameter(0.1, 2.0, "exploration_constant"),
            ...         NumericalHyperParameter(10, 50, "n_simulations")
            ...     ],
            ...     constant_parameters={
            ...         "discount_factor": 0.95,
            ...         "name": "OptimizedPOMCP",
            ...         "depth": 5
            ...     }
            ... )
            >>> optimization_configs = [
            ...     HyperParameterRunParams(
            ...         environment=tiger,
            ...         belief=initial_belief,
            ...         hyper_param_planner_config=planner_config,
            ...         num_episodes=2,
            ...         num_steps=3,
            ...         n_trials=3,
            ...         parameters_to_optimize=[
            ...             ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ...         ]
            ...     )
            ... ]
            >>> # Run optimization and evaluation on PBS cluster
            >>> results, stats_df = api.run_optimize_and_evaluate_pbs(
            ...     configs=optimization_configs,
            ...     evaluation_episodes=5,
            ...     evaluation_steps=10,
            ...     evaluation_n_jobs=1,
            ...     debug=False
            ... ) # doctest: +SKIP
            >>> len(stats_df) >= 1  # doctest: +SKIP
            True
        """
        self.logger.info("Starting optimize and evaluate workflow with PBS cluster execution")

        if cache_dir_path is None:
            cache_dir_path = Path("./optimize_and_evaluate_pbs_results")

        # Extract parameters from first config
        if not configs:
            raise ValueError("configs list cannot be empty")

        workflow = OptimizationEvaluationPBSWorkflow(
            cache_dir=cache_dir_path,
            experiment_name=experiment_name,
            queue=self.queue,
            n_workers=self.n_workers,
            cores=self.cores,
            memory=self.memory,
            processes=self.processes,
            walltime=self.walltime,
            job_extra=self.job_extra,
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
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Run all hyperparameter benchmarks with optimization on PBS cluster.

        This method runs hyperparameter optimization for all compatible environments
        and planners for a given policy space, followed by evaluation using PBS
        cluster computing.

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
            optimization_n_jobs: Number of parallel jobs for optimization (-1 uses all cores).
            is_risk_averse: Whether to run risk-averse benchmark.
            confidence_interval_level: Confidence level for intervals.
            alpha: Significance level for statistical tests.
            cache_dir_path: Optional path for storing results.
            experiment_name: Name for the experiment.
            debug: Enable debug mode.
            cache_visualizations: Whether to cache visualizations.

        Returns:
            Tuple of results dictionary and DataFrame.
        """
        self.logger.info("Starting all hyperparameter benchmarks with PBS cluster execution")

        if cache_dir_path is None:
            cache_dir_path = Path("./all_hyperparameter_benchmarks_pbs_results")

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

        workflow = OptimizationEvaluationPBSWorkflow(
            cache_dir=cache_dir_path,
            experiment_name=experiment_name,
            queue=self.queue,
            n_workers=self.n_workers,
            cores=self.cores,
            memory=self.memory,
            processes=self.processes,
            walltime=self.walltime,
            job_extra=self.job_extra,
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
