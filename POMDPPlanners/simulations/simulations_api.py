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
    def __init__(self, cache_dir_path: Path = None, debug: bool = False):
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
        profiling_stats_count: int = 50,
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        self.logger.info(f"Starting simulation run with {len(environment_run_params)} environment configurations")
        self.logger.debug(f"Parameters: alpha={alpha}, confidence_interval={confidence_interval_level}, n_jobs={n_jobs}")
        
        simulator = POMDPSimulator(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=debug,
            task_manager_type=TaskManagerType.JOBLIB,
            n_jobs=n_jobs,
            clear_cache_on_start=clear_cache_on_start,
            enable_profiling=enable_profiling,
            profiling_stats_count=profiling_stats_count
        )
        
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
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        self.logger.info(f"Starting simulation run with {len(environment_run_params)} environment configurations")
        self.logger.debug(f"Parameters: alpha={alpha}, confidence_interval={confidence_interval_level}, n_jobs={n_jobs}")
        
        simulator = POMDPSimulator(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=debug,
            task_manager_type=TaskManagerType.DASK,
            n_jobs=n_jobs,
            scheduler_address=scheduler_address,
            clear_cache_on_start=clear_cache_on_start,
        )
        
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
        
        # Run debug simulation
        self.logger.info("Starting debug simulation run")
        simulator_debug = POMDPSimulator(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=True,
            task_manager_type=TaskManagerType.JOBLIB,
            n_jobs=n_jobs,
            clear_cache_on_start=clear_cache_on_start,
        )
        
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
        simulator = POMDPSimulator(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=False,
            task_manager_type=TaskManagerType.JOBLIB,
            n_jobs=n_jobs,
            clear_cache_on_start=clear_cache_on_start,
        )
        
        results = simulator.compare_multiple_environments_policies(
            environment_run_params=environment_run_params,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
            cache_visualizations=True,
        )
        self.logger.info("Main simulation run completed!")
        return results
