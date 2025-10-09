from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.core.simulation import EnvironmentRunParams
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterRunParams,
    HyperParamPlannerConfigGenerator,
    OptimizedPolicyResult,
)
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
    DaskConfig,
)
from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.utils.logger import get_logger
from POMDPPlanners.simulations.simulation_apis.simulations_api_interface import (
    SimulationsAPIInterface,
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

    def __init__(self, cache_dir_path: Optional[Path] = None, debug: bool = False):
        """Initialize the DaskSimulationsAPI.

        Args:
            cache_dir_path: Optional path for storing simulation results and logs
            debug: Whether to enable debug-level logging output
        """
        self.logger = get_logger(
            name="dask_simulations_api", output_dir=cache_dir_path, debug=debug
        )
        self.logger.info("Initialized DaskSimulationsAPI")

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
            f"Starting simulation run with {len(environment_run_params)} environment configurations"
        )
        self.logger.debug(
            f"Parameters: alpha={alpha}, confidence_interval={confidence_interval_level}, n_jobs={n_jobs}"
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

        Note:
            This method is not yet implemented for Dask distributed execution.
            Use LocalSimulationsAPI for hyperparameter optimization.

        Raises:
            NotImplementedError: This method is not implemented for DaskSimulationsAPI.
        """
        raise NotImplementedError(
            "Hyperparameter optimization is not yet implemented for Dask distributed execution. "
            "Use LocalSimulationsAPI for hyperparameter optimization."
        )

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
        """Run comprehensive benchmark with hyperparameter optimization.

        Note:
            This method is not yet implemented for Dask distributed execution.
            Use LocalSimulationsAPI for hyperparameter tuning experiments.

        Raises:
            NotImplementedError: This method is not implemented for DaskSimulationsAPI.
        """
        raise NotImplementedError(
            "Hyperparameter tuning experiment with benchmarks is not yet implemented for Dask distributed execution. "
            "Use LocalSimulationsAPI for hyperparameter tuning experiments."
        )

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
        """Run hyperparameter optimization and evaluation.

        Note:
            This method is not yet implemented for Dask distributed execution.
            Use LocalSimulationsAPI for optimize and evaluate workflows.

        Raises:
            NotImplementedError: This method is not implemented for DaskSimulationsAPI.
        """
        raise NotImplementedError(
            "Optimize and evaluate workflow is not yet implemented for Dask distributed execution. "
            "Use LocalSimulationsAPI for optimize and evaluate workflows."
        )

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
        """Run all hyperparameter benchmarks with optimization.

        Note:
            This method is not yet implemented for Dask distributed execution.
            Use LocalSimulationsAPI for all hyperparameter benchmarks.

        Raises:
            NotImplementedError: This method is not implemented for DaskSimulationsAPI.
        """
        raise NotImplementedError(
            "All hyperparameter benchmarks is not yet implemented for Dask distributed execution. "
            "Use LocalSimulationsAPI for all hyperparameter benchmarks."
        )
