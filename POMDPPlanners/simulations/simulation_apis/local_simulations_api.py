import importlib
import inspect
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from POMDPPlanners.utils.hyperparameter_tuning_and_eval import HyperParamPlannerConfig

import pandas as pd

from POMDPPlanners.core.belief import Belief, get_initial_belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy, PolicySpaceInfo
from POMDPPlanners.core.simulation import (
    CategoricalHyperParameter,
    EnvironmentRunParams,
    NumericalHyperParameter,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterOptimizationDirection,
    HyperParameterRunParams,
    HyperParamPlannerConfigGenerator,
    OptimizedPolicyResult,
)
from POMDPPlanners.simulations.planner_evaluation_workflow import (
    PlannerEvaluationLocalWorkflow,
)
from POMDPPlanners.simulations.hyperparameter_tuning_evaluation_workflows import (
    OptimizationEvaluationLocalWorkflow,
)
from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import (
    HyperParameterOptimizer,
)
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
    JoblibConfig,
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


from POMDPPlanners.core.simulation.simulation_configs import PlannerGenerator


class LocalSimulationsAPI(SimulationsAPIInterface):
    """High-level API for running POMDP simulation experiments locally.

    This class provides a simplified interface for running POMDP simulations with
    local execution using Joblib for single-machine parallelization. It wraps the
    POMDPSimulator class and provides convenient methods for common simulation
    workflows without distributed computing requirements.

    Key Features:
    - Local execution using Joblib for single-machine parallelization
    - Debug mode with reduced episodes for quick testing
    - Automatic profiling and performance analysis
    - MLflow experiment tracking and result management
    - Hyperparameter optimization with Optuna

    Example:
        >>> from pathlib import Path
        >>> from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> from POMDPPlanners.core.simulation import EnvironmentRunParams
        >>> # Initialize the API
        >>> api = LocalSimulationsAPI(debug=True)
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
        ...     name="POMCP_Tiger",
        ...     n_simulations=20
        ... )
        >>> policy.name
        'POMCP_Tiger'
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

    def __init__(self, cache_dir_path: Optional[Path] = None, debug: bool = False):
        """Initialize the LocalSimulationsAPI.

        Args:
            cache_dir_path: Optional path for storing simulation results and logs
            debug: Whether to enable debug-level logging output
        """
        self.logger = get_logger(
            name="local_simulations_api", output_dir=cache_dir_path, debug=debug
        )
        self.logger.info("Initialized LocalSimulationsAPI")

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
        """Run POMDP simulations locally using Joblib for parallel execution.

        This method executes POMDP simulations on a single machine using Joblib
        for parallel processing. It's ideal for development, testing, and small to
        medium-scale experiments that can be completed on a single workstation.

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
            scheduler_address: This parameter is ignored by LocalSimulationsAPI. It exists
                for interface compatibility with DaskSimulationsAPI.
            n_jobs: Number of parallel jobs for execution. Use -1 to use all available
                CPU cores, or specify a positive integer for a specific number of cores.
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
            Running a local simulation with multiple environments and policies:

            >>> from pathlib import Path
            >>> from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
            >>> from POMDPPlanners.core.belief import get_initial_belief
            >>> from POMDPPlanners.core.simulation import EnvironmentRunParams
            >>> # Initialize the API
            >>> api = LocalSimulationsAPI(debug=True)
            >>> # Create environment and policy
            >>> tiger = TigerPOMDP(discount_factor=0.95)
            >>> policy = POMCP(
            ...     environment=tiger,
            ...     discount_factor=0.95,
            ...     depth=5,
            ...     exploration_constant=1.0,
            ...     name="POMCP_Local",
            ...     n_simulations=20
            ... )
            >>> # Configure simulation parameters
            >>> environment_run_params = [
            ...     EnvironmentRunParams(
            ...         environment=tiger,
            ...         belief=get_initial_belief(tiger, n_particles=10),
            ...         policies=[policy],
            ...         num_episodes=2,
            ...         num_steps=3
            ...     )
            ... ]
            >>> # Run local simulation
            >>> results, statistics_df = api.run_multiple_environments_and_policies(
            ...     environment_run_params=environment_run_params,
            ...     alpha=0.05,
            ...     confidence_interval_level=0.95,
            ...     experiment_name="Local_Tiger_Study",
            ...     n_jobs=1,
            ...     enable_profiling=False
            ... ) # doctest: +SKIP
            >>> # Check simulation results
            >>> len(statistics_df) >= 1  # doctest: +SKIP
            True
        """
        self.logger.info(
            f"Starting simulation run with {len(environment_run_params)} environment configurations"
        )
        self.logger.debug(
            f"Parameters: alpha={alpha}, confidence_interval={confidence_interval_level}, n_jobs={n_jobs}"
        )

        task_manager_config = JoblibConfig(n_jobs=n_jobs, clear_cache_on_start=clear_cache_on_start)

        with POMDPSimulator(
            task_manager_config=task_manager_config,
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=debug,
            enable_profiling=enable_profiling,
            profiling_output_limit=profiling_output_limit,
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

    def run_multiple_environments_and_policies_with_initial_debug_run(
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float,
        experiment_name: str = "POMDP_Planning_Comparison",
        n_jobs: int = -1,
        cache_dir_path: Optional[Path] = None,
        clear_cache_on_start: bool = False,
        enable_profiling: bool = False,
        profiling_output_limit: int = 50,
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Run POMDP simulations with an initial debug run for validation.

        This method executes POMDP simulations in two phases: first a quick debug run
        with reduced episodes and steps to validate the configuration, followed by the
        full simulation run. This approach helps catch configuration errors early and
        provides confidence that the full simulation will complete successfully.

        The debug run uses the same environment and policy configurations but with
        significantly reduced computational requirements (2 episodes, 2 steps each).
        Both runs are tracked separately in MLflow for comparison and analysis.

        Args:
            environment_run_params: List of environment configurations for simulation.
                Each configuration specifies an environment, belief state, policies,
                number of episodes, and number of steps per episode.
            alpha: Statistical significance level for confidence intervals (e.g., 0.05 for 95% CI).
                Used for computing risk metrics like Conditional Value at Risk (CVaR).
            confidence_interval_level: Confidence level for statistical analysis (e.g., 0.95).
                Determines the width of confidence intervals for performance metrics.
            experiment_name: Name for the main experiment and MLflow tracking. The debug
                run will use "{experiment_name}_debug_run" as its experiment name.
            n_jobs: Number of parallel jobs for execution. Use -1 to use all available
                CPU cores, or specify a positive integer for a specific number of cores.
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
                - Dict[str, Dict[str, list]]: Raw simulation results from the main run,
                  organized by environment name, then policy name, containing lists of
                  History objects for each episode.
                - pd.DataFrame: Statistical summary with confidence intervals, performance
                  metrics, and policy configuration details for analysis and comparison.

        Example:
            Running a simulation with initial debug validation:

            >>> from pathlib import Path
            >>> from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
            >>> from POMDPPlanners.core.belief import get_initial_belief
            >>> from POMDPPlanners.core.simulation import EnvironmentRunParams
            >>> # Initialize the API
            >>> api = LocalSimulationsAPI(debug=True)
            >>> # Create environment and policy
            >>> tiger = TigerPOMDP(discount_factor=0.95)
            >>> policy = POMCP(
            ...     environment=tiger,
            ...     discount_factor=0.95,
            ...     depth=5,
            ...     exploration_constant=1.0,
            ...     name="POMCP_DebugValidated",
            ...     n_simulations=20
            ... )
            >>> # Configure simulation parameters
            >>> environment_run_params = [
            ...     EnvironmentRunParams(
            ...         environment=tiger,
            ...         belief=get_initial_belief(tiger, n_particles=10),
            ...         policies=[policy],
            ...         num_episodes=3,  # Small number for testing
            ...         num_steps=2      # Small number for testing
            ...     )
            ... ]
            >>> # Run simulation with debug validation
            >>> results, statistics_df = api.run_multiple_environments_and_policies_with_initial_debug_run(
            ...     environment_run_params=environment_run_params,
            ...     alpha=0.05,
            ...     confidence_interval_level=0.95,
            ...     experiment_name="Debug_Validated_Tiger_Study",
            ...     n_jobs=1,
            ...     enable_profiling=False
            ... ) # doctest: +SKIP
            >>> # Check simulation results
            >>> len(statistics_df) >= 1  # doctest: +SKIP
            True
        """
        self.logger.info("Starting simulation run with initial debug run")
        self.logger.debug(
            f"Parameters: alpha={alpha}, confidence_interval={confidence_interval_level}, n_jobs={n_jobs}"
        )

        # Create debug configurations
        environment_run_params_debug = [
            EnvironmentRunParams(
                environment=config.environment,
                belief=config.belief,
                policies=config.policies,
                num_episodes=2,
                num_steps=2,
            )
            for config in environment_run_params
        ]
        self.logger.info(
            f"Created debug configurations with {len(environment_run_params_debug)} environments"
        )

        # Run debug simulation with separate experiment name to avoid conflicts
        debug_experiment_name = f"{experiment_name}_debug_run"
        self.logger.info("Starting debug simulation run")
        debug_task_manager_config = JoblibConfig(
            n_jobs=n_jobs, clear_cache_on_start=clear_cache_on_start
        )

        with POMDPSimulator(
            task_manager_config=debug_task_manager_config,
            cache_dir_path=cache_dir_path,
            experiment_name=debug_experiment_name,
            debug=True,
            enable_profiling=enable_profiling,
            profiling_output_limit=profiling_output_limit,
        ) as simulator_debug:
            simulator_debug.compare_multiple_environments_policies(
                environment_run_params=environment_run_params_debug,
                alpha=alpha,
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs,
                cache_visualizations=True,
            )
        self.logger.info("Debug simulation run completed")

        # Run main simulation
        self.logger.info("Starting main simulation run")
        main_task_manager_config = JoblibConfig(
            n_jobs=n_jobs, clear_cache_on_start=clear_cache_on_start
        )

        with POMDPSimulator(
            task_manager_config=main_task_manager_config,
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=False,
            enable_profiling=enable_profiling,
            profiling_output_limit=profiling_output_limit,
        ) as simulator:
            results = simulator.compare_multiple_environments_policies(
                environment_run_params=environment_run_params,
                alpha=alpha,
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs,
                cache_visualizations=True,
            )
        self.logger.info("Main simulation run completed!")
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
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        if len(generators) == 0:
            raise ValueError("generators list cannot be empty")

        if not all(isinstance(gen, PlannerGenerator) for gen in generators):
            raise ValueError("generators list must contain only PlannerGenerator objects")

        creator = AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator(
            generators=generators,
            n_particles=n_particles,
            num_episodes=num_episodes,
            num_steps=num_steps,
        )

        configs = creator.get_experiment_configs()
        workflow = PlannerEvaluationLocalWorkflow(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
            enable_profiling=enable_profiling,
            verbose=True,
            cache_visualizations=cache_visualizations,
        )
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
    ) -> List[OptimizedPolicyResult]:
        """Run hyperparameter optimization for POMDP policies using Optuna.

        This method provides a high-level interface for hyperparameter optimization
        by wrapping the HyperParameterOptimizer class. It supports optimization
        of multiple environment-policy configurations with comprehensive MLflow
        tracking and statistical analysis.

        The optimization uses Optuna's advanced algorithms (TPE, CMA-ES, etc.) to
        efficiently search the hyperparameter space and find optimal configurations
        for POMDP policies.

        Args:
            environment_run_params: List of HyperParameterRunParams configurations,
                each specifying an environment, policy class, hyperparameter ranges,
                and optimization settings. Each configuration must include the
                required n_trials parameter.
            experiment_name: Name for the MLflow experiment tracking. Used to organize
                optimization runs and enable comparison across different experiments.
            n_jobs: Number of parallel jobs for episode execution. Use -1 to use all
                available CPU cores, or specify a positive integer for a specific
                number of cores.
            cache_dir_path: Optional path for storing optimization results, logs,
                and MLflow artifacts. If None, results are stored in the current
                working directory.
            clear_cache_on_start: Whether to clear existing cache before starting
                optimization. Useful for ensuring clean runs when debugging or testing.
            debug: Whether to enable debug-level logging output. When True, provides
                detailed information about optimization progress and internal operations.
            confidence_interval_level: Confidence level for statistical analysis
                (between 0.0 and 1.0). Used for computing confidence intervals in
                performance statistics. Defaults to 0.95 for 95% confidence intervals.
            alpha: Significance level for statistical tests (between 0.0 and 1.0).
                Used for hypothesis testing and confidence interval calculations.
                Defaults to 0.05 for 5% significance level.
            use_queue_logger: Whether to use queue-based logging for distributed
                execution scenarios. Defaults to False for local execution.

        Returns:
            List[OptimizedPolicyResult]: List of optimization results, each containing
                the optimized policy with its best hyperparameters, environment reference,
                and optimization metadata for each input configuration.

        Raises:
            ValueError: If any configuration contains invalid parameters or missing
                required fields like n_trials.
            TypeError: If policy classes are not Policy subclasses.
            RuntimeError: If optimization fails for any configuration.

        Example:
            Running hyperparameter optimization for POMCP on Tiger POMDP:

            >>> from pathlib import Path
            >>> from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
            >>> from POMDPPlanners.core.belief import get_initial_belief
            >>> from POMDPPlanners.core.simulation import (
            ...     NumericalHyperParameter, CategoricalHyperParameter
            ... )
            >>> from POMDPPlanners.core.simulation.hyperparameter_tuning import (
            ...     HyperParameterRunParams, HyperParameterOptimizationDirection,
            ...     HyperParamPlannerConfig
            ... )
            >>> # Initialize the API
            >>> api = LocalSimulationsAPI(debug=True)
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
            ...         num_episodes=2,       # Small for testing
            ...         num_steps=3,          # Small for testing
            ...         n_trials=3,          # Small number for testing
            ...         parameters_to_optimize=[
            ...             ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ...         ]
            ...     )
            ... ]
            >>> # Run hyperparameter optimization
            >>> results = api.run_hyperparameter_optimization(
            ...     environment_run_params=optimization_configs,
            ...     experiment_name="Tiger_POMCP_Optimization",
            ...     n_jobs=1,
            ...     debug=True
            ... ) # doctest: +SKIP
            >>> # Check optimization results
            >>> len(results) >= 1  # doctest: +SKIP
            True

        Note:
            This method requires Optuna and MLflow to be installed. The optimization
            process can be computationally intensive for complex policies and large
            parameter spaces. All optimization runs are automatically tracked in
            MLflow with comprehensive parameter logging, metrics recording, and
            artifact storage.

            Key Implementation Details:
            - Uses HyperParameterOptimizer internally for optimization logic
            - All optimization runs are automatically tracked in MLflow
            - Requires explicit belief initialization (not generated automatically)
            - n_trials parameter is mandatory in each HyperParameterRunParams
            - Results include optimized policies with their chosen hyperparameters
            - Supports both numerical and categorical hyperparameters
        """
        self.logger.info(
            f"Starting hyperparameter optimization for {len(environment_run_params)} configurations"
        )
        self.logger.debug(
            f"Parameters: experiment_name={experiment_name}, n_jobs={n_jobs}, "
            f"confidence_interval={confidence_interval_level}, alpha={alpha}"
        )

        # Set up cache directory
        if cache_dir_path is None:
            cache_dir_path = Path("./hyperparameter_optimization_results")

        # Create cache directory if it doesn't exist
        cache_dir_path.mkdir(parents=True, exist_ok=True)

        task_manager_config = JoblibConfig(n_jobs=1)

        # Initialize the hyperparameter optimizer
        optimizer = HyperParameterOptimizer(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            n_jobs=n_jobs,
            task_manager_config=task_manager_config,
            confidence_interval_level=confidence_interval_level,
            alpha=alpha,
            use_queue_logger=use_queue_logger,
        )

        try:
            # Run optimization
            self.logger.info("Running hyperparameter optimization")
            results = optimizer.optimize(environment_run_params)

            self.logger.info(
                f"Hyperparameter optimization completed successfully. "
                f"Optimized {len(results)} out of {len(environment_run_params)} configurations"
            )

            # Log summary of results
            for i, result in enumerate(results):
                self.logger.info(
                    f"Configuration {i+1}: {result.environment.__class__.__name__} "
                    f"with {result.policy.__class__.__name__} - "
                    f"Best parameters: {result.chosen_hyper_parameters}"
                )

            return results

        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {e}")
            raise RuntimeError(f"Hyperparameter optimization failed: {e}") from e

        finally:
            # Clean up optimizer resources
            try:
                optimizer.cleanup()
            except Exception as cleanup_error:
                self.logger.warning(f"Error during optimizer cleanup: {cleanup_error}")

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
        """Run comprehensive benchmark with hyperparameter optimization locally.

        This method runs hyperparameter optimization followed by policy evaluation
        using local Joblib parallelization. It optimizes for average return across
        all configured environments and benchmark planners.

        Args:
            generators: Hyperparameter configuration generators list.
            particles: Number of particles for belief representation.
            num_episodes: Number of episodes for optimization.
            num_steps: Maximum steps per episode for optimization.
            n_trials: Number of optimization trials.
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
        self.logger.info("Starting comprehensive benchmark with local execution")

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
            debug=debug,
        )
        configs = creator.get_experiment_configs()

        workflow = OptimizationEvaluationLocalWorkflow(
            cache_dir=cache_dir_path,
            experiment_name=experiment_name,
            optimization_n_jobs=optimization_n_jobs,
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
        """Run hyperparameter optimization and evaluation locally.

        This method runs hyperparameter optimization for the provided configurations,
        then evaluates the optimized policies using local Joblib parallelization.

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
            >>> from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
            >>> from POMDPPlanners.core.belief import get_initial_belief
            >>> from POMDPPlanners.core.simulation import NumericalHyperParameter
            >>> from POMDPPlanners.core.simulation.hyperparameter_tuning import (
            ...     HyperParameterRunParams,
            ...     HyperParameterOptimizationDirection,
            ...     HyperParamPlannerConfig
            ... )
            >>> # Initialize the API
            >>> api = LocalSimulationsAPI(debug=True)
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
            >>> # Run optimization and evaluation
            >>> results, stats_df = api.run_optimize_and_evaluate(
            ...     configs=optimization_configs,
            ...     evaluation_episodes=5,
            ...     evaluation_steps=10,
            ...     evaluation_n_jobs=1,
            ...     optimization_n_jobs=1,
            ...     debug=True
            ... ) # doctest: +SKIP
            >>> len(stats_df) >= 1  # doctest: +SKIP
            True
        """
        self.logger.info("Starting optimize and evaluate workflow with local execution")

        if cache_dir_path is None:
            cache_dir_path = Path("./optimize_and_evaluate_results")

        # Extract parameters from first config
        if not configs:
            raise ValueError("configs list cannot be empty")

        workflow = OptimizationEvaluationLocalWorkflow(
            cache_dir=cache_dir_path,
            experiment_name=experiment_name,
            optimization_n_jobs=optimization_n_jobs,
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
        """Run all hyperparameter benchmarks with optimization locally.

        This method runs hyperparameter optimization for all compatible environments
        and planners for a given policy space, followed by evaluation using local
        Joblib parallelization.

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
        self.logger.info("Starting all hyperparameter benchmarks with local execution")

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

        workflow = OptimizationEvaluationLocalWorkflow(
            cache_dir=cache_dir_path,
            experiment_name=experiment_name,
            optimization_n_jobs=optimization_n_jobs,
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
