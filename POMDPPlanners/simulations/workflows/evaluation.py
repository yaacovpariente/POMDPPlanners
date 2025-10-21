"""Policy evaluation workflows."""

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

from POMDPPlanners.core.simulation import EnvironmentRunParams
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
    JoblibConfig,
    PBSConfig,
)
from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.utils.logger import get_logger

logger = get_logger(__name__)


def run_multiple_environments_and_policies_local_run(
    environment_run_params: List[EnvironmentRunParams],
    alpha: float,
    confidence_interval_level: float,
    experiment_name: str = "POMDP_Planning_Comparison",
    debug: bool = False,
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
        >>> results, statistics_df = api.run_multiple_environments_and_policies_local_run(
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
    logger.info(
        "Starting simulation run with %s environment configurations", len(environment_run_params)
    )
    logger.debug(
        "Parameters: alpha=%s, confidence_interval=%s, n_jobs=%s",
        alpha,
        confidence_interval_level,
        n_jobs,
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
        logger.info("Running simulation comparison")
        results = simulator.compare_multiple_environments_policies(
            environment_run_params=environment_run_params,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
            cache_visualizations=True,
        )
        logger.info("Simulation run completed")
        return results


def run_multiple_environments_and_policies_pbs_run(
    environment_run_params: List[EnvironmentRunParams],
    alpha: float,
    confidence_interval_level: float,
    queue: str,
    experiment_name: str = "POMDP_Planning_Comparison",
    debug: bool = False,
    n_workers: int = 4,
    cores: int = 1,
    memory: str = "4GB",
    processes: int = 1,
    walltime: str = "01:00:00",
    job_extra: Optional[List[str]] = None,
    n_jobs: int = -1,
    cache_dir_path: Optional[Path] = None,
    clear_cache_on_start: bool = False,
    enable_profiling: bool = False,
    profiling_output_limit: int = 50,
    enable_dashboard: bool = True,
    dashboard_address: str = "0.0.0.0",
    dashboard_port: int = 8787,
    dashboard_prefix: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
    """Run simulations using PBS cluster computing.

    This method executes POMDP simulations on a PBS (Portable Batch System) cluster,
    which is commonly used in high-performance computing environments. It automatically
    manages job submission, scaling, and distributed execution across cluster nodes.

    Args:
        environment_run_params: List of environment configurations for simulation
        alpha: Statistical significance level for confidence intervals (e.g., 0.05 for 95% CI)
        confidence_interval_level: Confidence level for statistical analysis (e.g., 0.95)
        queue: PBS queue name to submit jobs to
        experiment_name: Name for the experiment and MLflow tracking
        debug: Whether to enable debug-level logging output
        n_workers: Number of worker jobs to submit to PBS cluster
        cores: Number of CPU cores per PBS job
        memory: Memory allocation per PBS job (e.g., "4GB", "8GB")
        processes: Number of processes per PBS job
        walltime: Maximum runtime per job in HH:MM:SS format
        job_extra: Additional PBS directives as list of strings
        n_jobs: Number of parallel jobs for simulation execution. Use -1 to use
            all available workers (n_workers × processes), or specify a positive
            integer for a specific number of parallel jobs.
        cache_dir_path: Optional path for storing simulation results and logs
        clear_cache_on_start: Whether to clear cache before starting simulation
        enable_profiling: Whether to enable performance profiling
        profiling_output_limit: Maximum number of profiling entries to display
        enable_dashboard: Whether to enable the Dask dashboard for monitoring
        dashboard_address: Address to bind the dashboard to (e.g., "0.0.0.0", "127.0.0.1")
        dashboard_port: Port for the Dask dashboard (default: 8787)
        dashboard_prefix: URL prefix for dashboard (useful with reverse proxies)

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
        >>> # Initialize the API
        >>> api = PBSSimulationsAPI(queue="test_queue", debug=False)
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
        ...         num_episodes=2,  # Small number for testing
        ...         num_steps=3
        ...     )
        ... ]
        >>> # Run on PBS cluster (skip actual execution)
        >>> results, statistics_df = api.run_multiple_environments_and_policies_pbs_run(
        ...     environment_run_params=environment_run_params,
        ...     alpha=0.05,
        ...     confidence_interval_level=0.95,
        ...     queue="test_queue",
        ...     experiment_name="Small_Scale_Tiger_Study",
        ...     n_workers=2,
        ...     cores=1,
        ...     memory="4GB",
        ...     walltime="00:30:00",
        ...     enable_profiling=False
        ... ) # doctest: +SKIP
        >>> # Check simulation results
        >>> len(statistics_df) >= 1  # doctest: +SKIP
        True
    """
    logger.info(
        "Starting PBS cluster simulation with %s environment configurations",
        len(environment_run_params),
    )
    logger.debug(
        "PBS Parameters: queue=%s, n_workers=%s, cores=%s, memory=%s, walltime=%s",
        queue,
        n_workers,
        cores,
        memory,
        walltime,
    )
    logger.debug(
        "Simulation Parameters: alpha=%s, confidence_interval=%s", alpha, confidence_interval_level
    )

    task_manager_config = PBSConfig(
        queue=queue,
        n_workers=n_workers,
        cores=cores,
        memory=memory,
        processes=processes,
        walltime=walltime,
        job_extra=job_extra,
        clear_cache_on_start=clear_cache_on_start,
        enable_dashboard=enable_dashboard,
        dashboard_address=dashboard_address,
        dashboard_port=dashboard_port,
        dashboard_prefix=dashboard_prefix,
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
        logger.info("Running PBS cluster simulation comparison")
        results = simulator.compare_multiple_environments_policies(
            environment_run_params=environment_run_params,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
            cache_visualizations=True,
        )
        logger.info("PBS cluster simulation run completed")
        return results
