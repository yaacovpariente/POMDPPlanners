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

class SimulationsAPI:
    def __init__(self):
        pass

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
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        simulator = POMDPSimulator(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=debug,
            task_manager_type=TaskManagerType.JOBLIB,
            n_jobs=n_jobs,
            clear_cache_on_start=clear_cache_on_start,
        )
        return simulator.compare_multiple_environments_policies(
            environment_run_params=environment_run_params,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
            cache_visualizations=True,
        )
        
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
        
        simulator = POMDPSimulator(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=False,
            task_manager_type=TaskManagerType.JOBLIB,
            n_jobs=n_jobs,
            clear_cache_on_start=clear_cache_on_start,
        )
        
        return simulator.compare_multiple_environments_policies(
            environment_run_params=environment_run_params,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
            cache_visualizations=True,
        )
