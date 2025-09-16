import os
import importlib
import inspect
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import Belief, get_initial_belief
from POMDPPlanners.core.simulation import EnvironmentRunParams
from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
    JoblibConfig,
    DaskConfig,
    PBSConfig,
)
from POMDPPlanners.utils.logger import get_logger


class SimulationsAPI:
    """High-level API for running POMDP simulation experiments.

    This class provides a simplified interface for running POMDP simulations with
    different execution modes (local vs remote) and configuration options. It wraps
    the POMDPSimulator class and provides convenient methods for common simulation
    workflows.

    Key Features:
    - Local execution using Joblib for single-machine parallelization
    - Remote execution using Dask for distributed computation
    - Debug mode with reduced episodes for quick testing
    - Automatic profiling and performance analysis
    - MLflow experiment tracking and result management

    Example:
        Running a comprehensive algorithm comparison study::

            from pathlib import Path
            from POMDPPlanners.simulations.simulations_api import SimulationsAPI
            from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
            from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
            from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
            from POMDPPlanners.core.belief import get_initial_belief
            from POMDPPlanners.core.simulation import EnvironmentRunParams
            import numpy as np

            # Initialize the API
            api = SimulationsAPI(
                cache_dir_path=Path("./simulation_results"),
                debug=True  # Enable debug logging
            )

            # Create environments
            tiger = TigerPOMDP(discount_factor=0.95)
            cartpole = CartPolePOMDP(
                discount_factor=0.99,
                noise_cov=np.diag([0.1, 0.1, 0.1, 0.1])
            )

            # Create policies for each environment
            tiger_policies = [
                POMCP(
                    environment=tiger,
                    discount_factor=0.95,
                    depth=10,
                    exploration_constant=1.0,
                    name="POMCP_Tiger",
                    n_simulations=1000
                ),
                StandardSparseSamplingDiscreteActionsPlanner(
                    environment=tiger,
                    branching_factor=5,
                    depth=5,
                    name="SparseSampling_Tiger"
                )
            ]

            cartpole_policies = [
                POMCP(
                    environment=cartpole,
                    discount_factor=0.99,
                    depth=8,
                    exploration_constant=1.0,
                    name="POMCP_CartPole",
                    n_simulations=500
                )
            ]

            # Configure simulation parameters
            environment_run_params = [
                EnvironmentRunParams(
                    environment=tiger,
                    belief=get_initial_belief(tiger, n_particles=1000),
                    policies=tiger_policies,
                    num_episodes=200,
                    num_steps=25
                ),
                EnvironmentRunParams(
                    environment=cartpole,
                    belief=get_initial_belief(cartpole, n_particles=500),
                    policies=cartpole_policies,
                    num_episodes=100,
                    num_steps=50
                )
            ]

            # Run simulation with initial debug run
            results, statistics_df = api.run_multiple_environments_and_policies_local_run_with_initial_debug_run(
                environment_run_params=environment_run_params,
                alpha=0.05,
                confidence_interval_level=0.95,
                experiment_name="Multi_Environment_Comparison",
                n_jobs=-1,  # Use all available cores
                enable_profiling=True
            )

            # Analyze results
            print("\\nSimulation Results Summary:")
            print(f"Environments tested: {statistics_df['environment'].unique()}")
            print(f"Policies tested: {statistics_df['policy'].unique()}")
            print(f"Total configurations: {len(statistics_df)}")

            # Compare policies within each environment
            for env_name in statistics_df['environment'].unique():
                env_stats = statistics_df[statistics_df['environment'] == env_name]
                print(f"\\n{env_name} Results:")
                for policy_name in env_stats['policy'].unique():
                    policy_stats = env_stats[env_stats['policy'] == policy_name]
                    avg_return = policy_stats['average_return'].iloc[0]
                    ci_lower = policy_stats['average_return_ci_lower'].iloc[0]
                    ci_upper = policy_stats['average_return_ci_upper'].iloc[0]
                    print(f"  {policy_name}: {avg_return:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
    """

    def __init__(self, cache_dir_path: Path = None, debug: bool = False):
        """Initialize the SimulationsAPI.

        Args:
            cache_dir_path: Optional path for storing simulation results and logs
            debug: Whether to enable debug-level logging output
        """
        self.logger = get_logger(
            name="simulations_api", output_dir=cache_dir_path, debug=debug
        )
        self.logger.info("Initialized SimulationsAPI")

    def run_multiple_environments_and_policies_local_run(
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float,
        experiment_name: str = "POMDP_Planning_Comparison",
        debug: bool = False,
        n_jobs: int = -1,
        cache_dir_path: Path = None,
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
            Running a local simulation with multiple environments and policies::

                from pathlib import Path
                from POMDPPlanners.simulations.simulations_api import SimulationsAPI
                from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
                from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
                from POMDPPlanners.core.belief import get_initial_belief
                from POMDPPlanners.core.simulation import EnvironmentRunParams

                # Initialize the API
                api = SimulationsAPI(
                    cache_dir_path=Path("./local_results"),
                    debug=True
                )

                # Create environment and policies
                tiger = TigerPOMDP(discount_factor=0.95)
                policies = [
                    POMCP(
                        environment=tiger,
                        discount_factor=0.95,
                        depth=10,
                        exploration_constant=1.0,
                        name="POMCP_Local",
                        n_simulations=1000
                    )
                ]

                # Configure simulation parameters
                environment_run_params = [
                    EnvironmentRunParams(
                        environment=tiger,
                        belief=get_initial_belief(tiger, n_particles=1000),
                        policies=policies,
                        num_episodes=100,
                        num_steps=20
                    )
                ]

                # Run local simulation
                results, statistics_df = api.run_multiple_environments_and_policies_local_run(
                    environment_run_params=environment_run_params,
                    alpha=0.05,
                    confidence_interval_level=0.95,
                    experiment_name="Local_Tiger_Study",
                    n_jobs=4,  # Use 4 CPU cores
                    enable_profiling=True
                )

                print(f"Simulation completed with {len(statistics_df)} configurations")
        """
        self.logger.info(
            f"Starting simulation run with {len(environment_run_params)} environment configurations"
        )
        self.logger.debug(
            f"Parameters: alpha={alpha}, confidence_interval={confidence_interval_level}, n_jobs={n_jobs}"
        )

        task_manager_config = JoblibConfig(
            n_jobs=n_jobs, clear_cache_on_start=clear_cache_on_start
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

    def run_multiple_environments_and_policies_remote_run(
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float,
        experiment_name: str = "POMDP_Planning_Comparison",
        debug: bool = False,
        scheduler_address: str = None,
        n_jobs: int = -1,
        cache_dir_path: Path = None,
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
            Running a distributed simulation with a remote Dask cluster::

                from pathlib import Path
                from POMDPPlanners.simulations.simulations_api import SimulationsAPI
                from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
                from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
                from POMDPPlanners.core.belief import get_initial_belief
                from POMDPPlanners.core.simulation import EnvironmentRunParams

                # Initialize the API
                api = SimulationsAPI(
                    cache_dir_path=Path("./distributed_results"),
                    debug=False
                )

                # Create environment and policies
                tiger = TigerPOMDP(discount_factor=0.95)
                policies = [
                    POMCP(
                        environment=tiger,
                        discount_factor=0.95,
                        depth=15,
                        exploration_constant=1.0,
                        name="POMCP_Distributed",
                        n_simulations=2000
                    )
                ]

                # Configure simulation parameters for distributed execution
                environment_run_params = [
                    EnvironmentRunParams(
                        environment=tiger,
                        belief=get_initial_belief(tiger, n_particles=2000),
                        policies=policies,
                        num_episodes=500,  # Large number for distributed processing
                        num_steps=30
                    )
                ]

                # Run distributed simulation
                results, statistics_df = api.run_multiple_environments_and_policies_remote_run(
                    environment_run_params=environment_run_params,
                    alpha=0.01,  # 99% confidence intervals
                    confidence_interval_level=0.99,
                    experiment_name="Distributed_Tiger_Study",
                    scheduler_address="tcp://192.168.1.100:8786",  # Remote scheduler
                    n_jobs=8,  # Use 8 workers
                    enable_profiling=True
                )

                print(f"Distributed simulation completed with {len(statistics_df)} configurations")
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

    def run_multiple_environments_and_policies_local_run_with_initial_debug_run(
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float,
        experiment_name: str = "POMDP_Planning_Comparison",
        n_jobs: int = -1,
        cache_dir_path: Path = None,
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
            Running a simulation with initial debug validation::

                from pathlib import Path
                from POMDPPlanners.simulations.simulations_api import SimulationsAPI
                from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
                from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
                from POMDPPlanners.core.belief import get_initial_belief
                from POMDPPlanners.core.simulation import EnvironmentRunParams

                # Initialize the API
                api = SimulationsAPI(
                    cache_dir_path=Path("./debug_validated_results"),
                    debug=True
                )

                # Create environment and policies
                tiger = TigerPOMDP(discount_factor=0.95)
                policies = [
                    POMCP(
                        environment=tiger,
                        discount_factor=0.95,
                        depth=10,
                        exploration_constant=1.0,
                        name="POMCP_DebugValidated",
                        n_simulations=1000
                    )
                ]

                # Configure simulation parameters
                environment_run_params = [
                    EnvironmentRunParams(
                        environment=tiger,
                        belief=get_initial_belief(tiger, n_particles=1000),
                        policies=policies,
                        num_episodes=200,  # Full simulation episodes
                        num_steps=25       # Full simulation steps
                    )
                ]

                # Run simulation with debug validation
                results, statistics_df = api.run_multiple_environments_and_policies_local_run_with_initial_debug_run(
                    environment_run_params=environment_run_params,
                    alpha=0.05,
                    confidence_interval_level=0.95,
                    experiment_name="Debug_Validated_Tiger_Study",
                    n_jobs=4,
                    enable_profiling=True
                )

                print(f"Debug-validated simulation completed with {len(statistics_df)} configurations")
                print("Check MLflow for both debug and main experiment results")
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

    def run_multiple_environments_and_policies_pbs_run(
        self,
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
        job_extra: List[str] = None,
        cache_dir_path: Path = None,
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
            queue: PBS queue name to submit jobs to
            experiment_name: Name for the experiment and MLflow tracking
            debug: Whether to enable debug-level logging output
            n_workers: Number of worker jobs to submit to PBS cluster
            cores: Number of CPU cores per PBS job
            memory: Memory allocation per PBS job (e.g., "4GB", "8GB")
            processes: Number of processes per PBS job
            walltime: Maximum runtime per job in HH:MM:SS format
            job_extra: Additional PBS directives as list of strings
            cache_dir_path: Optional path for storing simulation results and logs
            clear_cache_on_start: Whether to clear cache before starting simulation
            enable_profiling: Whether to enable performance profiling
            profiling_output_limit: Maximum number of profiling entries to display

        Returns:
            Tuple containing:
                - Dict[str, Dict[str, list]]: Raw simulation results organized by environment and policy
                - pd.DataFrame: Statistical summary with confidence intervals and performance metrics

        Example:
            Running a large-scale simulation study on PBS cluster::

                from pathlib import Path
                from POMDPPlanners.simulations.simulations_api import SimulationsAPI
                from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
                from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
                from POMDPPlanners.core.belief import get_initial_belief
                from POMDPPlanners.core.simulation import EnvironmentRunParams

                # Initialize the API
                api = SimulationsAPI(
                    cache_dir_path=Path("./cluster_results"),
                    debug=False
                )

                # Create environment and policy
                tiger = TigerPOMDP(discount_factor=0.95)
                policy = POMCP(
                    environment=tiger,
                    discount_factor=0.95,
                    depth=15,
                    exploration_constant=1.0,
                    name="POMCP_ClusterTest",
                    n_simulations=2000
                )

                # Configure simulation for cluster execution
                environment_run_params = [
                    EnvironmentRunParams(
                        environment=tiger,
                        belief=get_initial_belief(tiger, n_particles=2000),
                        policies=[policy],
                        num_episodes=1000,  # Large number for cluster
                        num_steps=50
                    )
                ]

                # Run on PBS cluster with custom configuration
                results, statistics_df = api.run_multiple_environments_and_policies_pbs_run(
                    environment_run_params=environment_run_params,
                    alpha=0.01,  # 99% confidence intervals
                    confidence_interval_level=0.99,
                    queue="gpu_queue",  # PBS queue name
                    experiment_name="Large_Scale_Tiger_Study",
                    n_workers=16,  # Submit 16 jobs to cluster
                    cores=4,  # 4 cores per job
                    memory="16GB",  # 16GB per job
                    walltime="04:00:00",  # 4 hour time limit
                    job_extra=["#PBS -l feature=gpu", "#PBS -m ae"],  # GPU nodes, email notifications
                    enable_profiling=True
                )

                print(f"Cluster simulation completed with {len(statistics_df)} configurations")
        """
        self.logger.info(
            f"Starting PBS cluster simulation with {len(environment_run_params)} environment configurations"
        )
        self.logger.debug(
            f"PBS Parameters: queue={queue}, n_workers={n_workers}, cores={cores}, memory={memory}, walltime={walltime}"
        )
        self.logger.debug(
            f"Simulation Parameters: alpha={alpha}, confidence_interval={confidence_interval_level}"
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
                n_jobs=n_workers,  # Use n_workers for consistency
                cache_visualizations=True,
            )
            self.logger.info("PBS cluster simulation run completed")
            return results
