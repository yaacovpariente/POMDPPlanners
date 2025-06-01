# from typing import List, Type, Tuple, Dict, Optional, Union
# import copy
# from time import time
# from pathlib import Path
# import mlflow
# import json
# import os
# import logging
# from joblib import Parallel, delayed
# from multiprocessing import cpu_count
# import optuna
# import pandas as pd
# import numpy as np
# import tempfile
# from tqdm import tqdm
# from typing import Union, List, Tuple, Type
# from POMDPPlanners.core.environment import Environment
# from POMDPPlanners.core.policy import Policy
# from POMDPPlanners.core.belief import Belief, get_initial_belief
# from POMDPPlanners.core.simulation import (
#     History,
#     StepData,
#     CategoricalHyperParameter,
#     NumericalHyperParameter,
#     MetricValue,
# )
# from POMDPPlanners.simulations.simulation_statistics import compute_statistics_environment_policy_pair, compute_statistics_environments_policies_comparison
# from POMDPPlanners.utils.visualization import plot_metrics_comparison, plot_discounted_returns_histogram, plot_environment_policy_pair_comparison, plot_discounted_returns_histogram_multiple_policies
# from POMDPPlanners.utils.logger import get_logger

# logger = get_logger(__name__)

# HyperParameterFeatures = Union[CategoricalHyperParameter, NumericalHyperParameter]

# class HyperParameterOptimizer:
#     """A class for optimizing hyperparameters of POMDP policies using Optuna."""
    
#     def __init__(
#         self,
#         cache_dir_path: Path,
#         experiment_name: str = "POMDP_Parameter_Optimization",
#         n_jobs: int = 1,
#         confidence_interval_level: float = 0.95,
#     ):
#         """Initialize the hyperparameter optimizer.
        
#         Args:
#             cache_dir_path: Path to store optimization results and cache
#             experiment_name: Name of the MLFlow experiment
#             n_jobs: Number of parallel jobs for simulation
#             confidence_interval_level: Confidence level for statistics
#         """
#         self.cache_dir_path = cache_dir_path
#         self.experiment_name = experiment_name
#         self.n_jobs = n_jobs
#         self.confidence_interval_level = confidence_interval_level
        
#         # Set up MLFlow tracking
#         self.mlruns_path = cache_dir_path / "mlruns"
#         self.mlruns_path.mkdir(parents=True, exist_ok=True)
#         mlflow.set_tracking_uri(str(self.mlruns_path))
#         mlflow.set_experiment(experiment_name)

#     def run_multiple_episodes(
#         self,
#         environment: Environment,
#         policy: Policy,
#         initial_belief: Belief,
#         num_episodes: int,
#         num_steps: int,
#         scheduler_address: Optional[str] = None,
#     ) -> List[History]:
#         """Run multiple episodes in parallel."""
#         logger.info(f"Starting {num_episodes} episodes with {num_steps} steps each using {self.n_jobs} jobs")
#         logger.info(f"Environment: {environment.name}, Policy: {policy.name}")
        
#         assert isinstance(environment, Environment)
#         assert isinstance(policy, Policy)
#         assert isinstance(initial_belief, Belief)
#         if scheduler_address is not None:
#             assert isinstance(scheduler_address, str)

#         assert num_episodes > 0
#         assert num_steps > 0

#         # Create a list of arguments for each episode
#         episode_args = [
#             (environment, policy, initial_belief, num_steps, i, hash(f"{environment.name}_{policy.name}_{i}") % (2**32), scheduler_address, self.cache_dir_path) 
#             for i in range(num_episodes)
#         ]

#         # Run episodes in parallel using joblib with progress bar
#         logger.info(f"Running episodes in parallel for {environment.name} with {policy.name}")
#         histories = Parallel(n_jobs=self.n_jobs)(
#             delayed(self._run_and_cache_episode)(*args) for args in tqdm(
#                 episode_args,
#                 total=num_episodes,
#                 desc=f"Running episodes for {environment.name} with {policy.name}",
#                 unit="episode"
#             )
#         )
#         logger.info(f"All episodes completed for {environment.name} with {policy.name}")

#         return histories

#     def simulation(
#         self,
#         environment: Environment,
#         policy: Policy,
#         initial_belief: Belief,
#         num_episodes: int,
#         num_steps: int,
#         alpha: float,
#         scheduler_address: Optional[str] = None,
#     ) -> Tuple[List[History], List[MetricValue]]:
#         """Run a simulation and compute statistics."""
#         logger.info(f"Starting simulation with {num_episodes} episodes, {num_steps} steps, "
#                     f"alpha={alpha}, confidence_interval={self.confidence_interval_level}")
        
#         assert isinstance(environment, Environment)
#         assert isinstance(policy, Policy)
#         assert isinstance(initial_belief, Belief)
#         assert isinstance(num_episodes, int)
#         assert isinstance(num_steps, int)
#         if scheduler_address is not None:
#             assert isinstance(scheduler_address, str)

#         assert 1 >= self.confidence_interval_level >= 0
#         assert num_episodes > 0
#         assert num_steps > 0

#         histories = self.run_multiple_episodes(
#             environment=environment,
#             policy=policy,
#             initial_belief=initial_belief,
#             num_episodes=num_episodes,
#             num_steps=num_steps,
#             scheduler_address=scheduler_address,
#         )

#         logger.info("Computing statistics from simulation results")
#         statistics = compute_statistics_environment_policy_pair(
#             env=environment,
#             histories=histories,
#             alpha=alpha,
#             confidence_interval_level=self.confidence_interval_level,
#         )
#         logger.info("Statistics computation completed")

#         return histories, statistics

#     def optimize_policy_parameters(
#         self,
#         environment: Environment,
#         policy_class: Type[Policy],
#         param_ranges: List[HyperParameterFeatures],
#         num_episodes: int,
#         num_steps: int,
#         n_particles: int,
#         parameter_to_optimize: str = "average_return",
#         direction: str = "maximize",
#         n_trials: int = 100,
#     ) -> Tuple[dict, float, List[History]]:
#         """Optimize policy parameters using Optuna."""
#         assert isinstance(environment, Environment)
#         assert isinstance(policy_class, type)
#         assert issubclass(policy_class, Policy)
#         assert isinstance(param_ranges, list)
#         assert all(isinstance(param, (CategoricalHyperParameter, NumericalHyperParameter)) for param in param_ranges)
#         assert isinstance(num_episodes, int)
#         assert isinstance(num_steps, int)
#         assert isinstance(n_particles, int)
#         assert isinstance(parameter_to_optimize, str)
#         assert isinstance(n_trials, int)

#         assert num_episodes > 0
#         assert num_steps > 0
#         assert n_particles > 0
#         assert direction in ["maximize", "minimize"]

#         def evaluation_function(policy: Policy, trial: optuna.Trial) -> float:
#             initial_belief = get_initial_belief(pomdp=environment, n_particles=n_particles)
#             histories, statistics = self.simulation(
#                 environment=environment,
#                 policy=policy,
#                 initial_belief=initial_belief,
#                 num_episodes=num_episodes,
#                 num_steps=num_steps,
#                 alpha=0.05,  # Fixed alpha for optimization
#                 scheduler_address=None,
#             )

#             # Store histories as trial user attributes
#             trial.set_user_attr("histories", histories)

#             # Get the current value
#             for metric in statistics:
#                 if metric.name == parameter_to_optimize:
#                     return metric.value

#             raise ValueError(f"Parameter {parameter_to_optimize} not found in statistics")

#         def objective(trial: optuna.Trial) -> float:
#             # Create parameters dictionary from hyperparameters
#             policy_params = {"environment": environment}

#             # Add optimization parameters
#             for param in param_ranges:
#                 if isinstance(param, CategoricalHyperParameter):
#                     policy_params[param.name] = trial.suggest_categorical(
#                         param.name, param.choices
#                     )
#                 elif isinstance(param, NumericalHyperParameter):
#                     if isinstance(param.low, float):
#                         policy_params[param.name] = trial.suggest_float(
#                             param.name, param.low, param.high
#                         )
#                     else:
#                         policy_params[param.name] = trial.suggest_int(
#                             param.name, param.low, param.high
#                         )

#             # Create policy instance with suggested parameters
#             policy = policy_class(**policy_params)

#             # Evaluate policy and return objective value
#             return evaluation_function(policy, trial)

#         # Create and run the study
#         study = optuna.create_study(direction=direction)
#         study.optimize(objective, n_trials=n_trials)

#         # Get best parameters and value
#         best_params = study.best_params
#         best_value = study.best_value

#         # Get histories from the best trial
#         best_trial = study.best_trial
#         histories = best_trial.user_attrs["histories"]

#         return best_params, best_value, histories

#     def optimize_multiple_environments(
#         self,
#         environment_policy_pairs: List[
#             Tuple[Environment, Tuple[Type[Policy], List[HyperParameterFeatures]]]
#         ],
#         num_episodes: int,
#         num_steps: int,
#         n_particles: int,
#         parameter_to_optimize: str = "average_return",
#         direction: str = "maximize",
#         n_trials: int = 100,
#     ) -> Tuple[List[Tuple[dict, float, List[History]]], pd.DataFrame]:
#         """Optimize policy parameters for multiple environment-policy pairs."""
#         assert isinstance(environment_policy_pairs, List)
#         assert all(
#             isinstance(pair, tuple) and len(pair) == 2 for pair in environment_policy_pairs
#         )
#         assert all(isinstance(env, Environment) for env, _ in environment_policy_pairs)
#         assert all(
#             isinstance(policy_config, tuple) and len(policy_config) == 2
#             for _, policy_config in environment_policy_pairs
#         )
#         assert all(
#             isinstance(policy_class, type) and issubclass(policy_class, Policy)
#             for _, (policy_class, _) in environment_policy_pairs
#         )
#         assert all(
#             isinstance(param_ranges, list)
#             and all(isinstance(param, (CategoricalHyperParameter, NumericalHyperParameter)) for param in param_ranges)
#             for _, (_, param_ranges) in environment_policy_pairs
#         )

#         results = []
#         planner_statistics = []
#         aggregated_data = []

#         for pair_idx, (environment, (policy_class, param_ranges)) in enumerate(
#             environment_policy_pairs
#         ):
#             # End any active run safely
#             active_run = mlflow.active_run()
#             if active_run is not None:
#                 mlflow.end_run()

#             # Start a new MLFlow run for each environment-policy combination
#             with mlflow.start_run(
#                 run_name=f"{environment.__class__.__name__}_{policy_class.__name__}_{pair_idx}"
#             ):
#                 # Log common parameters
#                 common_params = {
#                     "environment_type": environment.__class__.__name__,
#                     "policy_type": policy_class.__name__,
#                     "num_episodes": num_episodes,
#                     "num_steps": num_steps,
#                     "n_particles": n_particles,
#                     "parameter_to_optimize": parameter_to_optimize,
#                     "direction": direction,
#                     "n_trials": n_trials,
#                     "confidence_interval_level": self.confidence_interval_level,
#                 }

#                 # Log environment-specific parameters
#                 env_params = {
#                     f"env_{key}": value
#                     for key, value in environment.__dict__.items()
#                     if isinstance(value, (int, float, str, bool))
#                 }

#                 # Log policy-specific parameter ranges
#                 policy_param_ranges = {}
#                 for param in param_ranges:
#                     if isinstance(param, CategoricalHyperParameter):
#                         policy_param_ranges[f"param_range_{param.name}"] = (
#                             f"choices: {param.choices}"
#                         )
#                     elif isinstance(param, NumericalHyperParameter):
#                         policy_param_ranges[f"param_range_{param.name}"] = (
#                             f"{param.low}-{param.high}"
#                         )

#                 # Combine all parameters
#                 all_params = {**common_params, **env_params, **policy_param_ranges}
#                 mlflow.log_params(all_params)

#                 # Run optimization
#                 best_params, best_value, histories = self.optimize_policy_parameters(
#                     environment=environment,
#                     policy_class=policy_class,
#                     param_ranges=param_ranges,
#                     num_episodes=num_episodes,
#                     num_steps=num_steps,
#                     n_particles=n_particles,
#                     parameter_to_optimize=parameter_to_optimize,
#                     direction=direction,
#                     n_trials=n_trials,
#                 )

#                 # Get statistics from the best trial
#                 initial_belief = get_initial_belief(
#                     pomdp=environment, n_particles=n_particles
#                 )
#                 _, statistics = self.simulation(
#                     environment=environment,
#                     policy=policy_class(environment=environment, **best_params),
#                     initial_belief=initial_belief,
#                     num_episodes=num_episodes,
#                     num_steps=num_steps,
#                     alpha=0.05,  # Fixed alpha for evaluation
#                 )

#                 # Log best parameters and value
#                 mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
#                 mlflow.log_metric("best_value", best_value)

#                 # Save the full results as a JSON artifact
#                 results_data = {
#                     "best_parameters": best_params,
#                     "best_value": best_value,
#                     "parameters": all_params,
#                     "param_ranges": [param._asdict() for param in param_ranges],
#                     "statistics": [metric._asdict() for metric in statistics],
#                 }
#                 mlflow.log_dict(results_data, f"optimization_results_pair_{pair_idx}.json")

#                 results.append((best_params, best_value, histories))
#                 planner_statistics.append(statistics)

#                 # Aggregate data for DataFrame
#                 run_data = {
#                     **all_params,
#                     **{f"best_{k}": v for k, v in best_params.items()},
#                     "best_value": best_value,
#                     **{metric.name: metric.value for metric in statistics},
#                 }
#                 aggregated_data.append(run_data)

#         # Convert aggregated data to DataFrame
#         df = pd.DataFrame(aggregated_data)

#         # Log DataFrame as table artifact through MLFlow
#         mlflow.log_table(df, "optimization_results.json")

#         # Plot statistics comparison
#         with tempfile.TemporaryDirectory() as temp_dir:
#             temp_dir_path = Path(temp_dir)
#             plot_metrics_comparison(
#                 statistics=planner_statistics,
#                 environments=[env for env, (policy_class, _) in environment_policy_pairs],
#                 policies=[policy_class for _, (policy_class, _) in environment_policy_pairs],
#                 cache_dir_path=temp_dir_path,
#             )

#             # Log all generated plots as artifacts
#             for plot_file in temp_dir_path.glob("*.png"):
#                 mlflow.log_artifact(str(plot_file), "statistics_plots")

#         # End any active run safely before returning
#         active_run = mlflow.active_run()
#         if active_run is not None:
#             mlflow.end_run()

#         return results, df

#     def _run_and_cache_episode(
#         self,
#         environment: Environment,
#         policy: Policy,
#         initial_belief: Belief,
#         num_steps: int,
#         episode_idx: int,
#         episode_hash: int,
#         scheduler_address: Optional[str],
#         cache_dir_path: Path,
#     ) -> History:
#         """Run a single episode and cache the results."""
#         # TODO: Implement episode caching logic
#         # This is a placeholder for the actual implementation
#         raise NotImplementedError("Episode caching not yet implemented")
