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
from POMDPPlanners.simulations.simulations_deployment.task_managers import TaskManagerType
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
            name="simulations_api",
            output_dir=cache_dir_path,
            debug=debug
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
        self.logger.info(f"Starting simulation run with {len(environment_run_params)} environment configurations")
        self.logger.debug(f"Parameters: alpha={alpha}, confidence_interval={confidence_interval_level}, n_jobs={n_jobs}")
        
        with POMDPSimulator(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=debug,
            task_manager_type=TaskManagerType.JOBLIB,
            n_jobs=n_jobs,
            clear_cache_on_start=clear_cache_on_start,
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
        self.logger.info(f"Starting simulation run with {len(environment_run_params)} environment configurations")
        self.logger.debug(f"Parameters: alpha={alpha}, confidence_interval={confidence_interval_level}, n_jobs={n_jobs}")
        
        with POMDPSimulator(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=debug,
            task_manager_type=TaskManagerType.DASK,
            n_jobs=n_jobs,
            scheduler_address=scheduler_address,
            clear_cache_on_start=clear_cache_on_start,
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
        self.logger.info("Starting simulation run with initial debug run")
        self.logger.debug(f"Parameters: alpha={alpha}, confidence_interval={confidence_interval_level}, n_jobs={n_jobs}")
        
        # Create debug configurations
        environment_run_params_debug = [
            EnvironmentRunParams(
                environment=config.environment,
                belief=config.belief,
                policies=config.policies,
                num_episodes=2,
                num_steps=2
            )
            for config in environment_run_params
        ]
        self.logger.info(f"Created debug configurations with {len(environment_run_params_debug)} environments")
        
        # Run debug simulation with separate experiment name to avoid conflicts
        debug_experiment_name = f"{experiment_name}_debug_run"
        self.logger.info("Starting debug simulation run")
        with POMDPSimulator(
            cache_dir_path=cache_dir_path,
            experiment_name=debug_experiment_name,
            debug=True,
            task_manager_type=TaskManagerType.JOBLIB,
            n_jobs=n_jobs,
            clear_cache_on_start=clear_cache_on_start,
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
        with POMDPSimulator(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=False,
            task_manager_type=TaskManagerType.JOBLIB,
            n_jobs=n_jobs,
            clear_cache_on_start=clear_cache_on_start,
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
