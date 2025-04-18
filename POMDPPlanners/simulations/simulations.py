from typing import List, Type, Tuple
import copy
from time import time
from pathlib import Path
import mlflow
import json
import os
from joblib import Parallel, delayed
import optuna
import pandas as pd
import numpy as np
import tempfile

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import Belief, get_initial_belief
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.simulations.simulation_statistics import compute_statistics
from POMDPPlanners.utils.visualization import plot_statistics_comparison

def run_multiple_episodes(
    environment: Environment, 
    policy: Policy, 
    initial_belief: Belief, 
    discount_factor: float, 
    num_episodes: int, 
    num_steps: int,
    n_jobs: int = 1
) -> List[History]:
    assert isinstance(environment, Environment)
    assert isinstance(policy, Policy)
    assert isinstance(initial_belief, Belief)
    assert isinstance(discount_factor, float)
    
    assert 1 >= discount_factor >= 0
    assert num_episodes > 0
    assert num_steps > 0
    
    # Create a list of arguments for each episode
    episode_args = [(environment, policy, initial_belief, discount_factor, num_steps) for _ in range(num_episodes)]
    
    # Run episodes in parallel using joblib
    histories = Parallel(n_jobs=n_jobs)(delayed(run_episode)(*args) for args in episode_args)
    
    return histories

def run_episode(
    environment: Environment, 
    policy: Policy, 
    initial_belief: Belief, 
    discount_factor: float, 
    num_steps: int
) -> History:
    assert isinstance(environment, Environment)
    assert isinstance(policy, Policy)
    assert isinstance(initial_belief, Belief)
    assert isinstance(discount_factor, float)
    
    assert 1 >= discount_factor >= 0
    assert num_steps > 0
    
    average_state_sampling_time = 0.
    average_action_time = 0.
    average_observation_time = 0.
    average_belief_update_time = 0.
    average_reward_time = 0.
    
    belief = copy.deepcopy(initial_belief)
    state = belief.sample()
    
    history = []
    for i in range(1, num_steps + 1):
        action_start_time = time()
        action = policy.action(belief)
        action_time = time() - action_start_time
        average_action_time = (average_action_time * (i - 1) + action_time) / i

        reward_start_time = time()
        reward = environment.reward(state, action)
        reward_time = time() - reward_start_time
        average_reward_time = (average_reward_time * (i - 1) + reward_time) / i
        
        state_sampling_start_time = time()
        next_state = environment.state_transition_model(state, action).sample()
        state_sampling_time = time() - state_sampling_start_time
        average_state_sampling_time = (average_state_sampling_time * (i - 1) + state_sampling_time) / i

        observation_start_time = time()
        observation = environment.observation_model(next_state, action).sample()
        observation_time = time() - observation_start_time
        average_observation_time = (average_observation_time * (i - 1) + observation_time) / i
        
        history.append(
            StepData(
                state=state, 
                action=action, 
                next_state=next_state, 
                observation=observation, 
                reward=reward
            )
        )
        
        belief_update_start_time = time()
        belief = belief.update(action=action, observation=observation, pomdp=environment)
        belief_update_time = time() - belief_update_start_time
        average_belief_update_time = (average_belief_update_time * (i - 1) + belief_update_time) / i

        state = next_state
    
    return History(
        history=history, 
        discount_factor=discount_factor, 
        average_state_sampling_time=average_state_sampling_time, 
        average_action_time=average_action_time, 
        average_observation_time=average_observation_time, 
        average_belief_update_time=average_belief_update_time,
        average_reward_time=average_reward_time
    )

def simulation(
    environment: Environment,
    policy: Policy,
    initial_belief: Belief,
    discount_factor: float,
    num_episodes: int,
    num_steps: int,
    alpha: float,
    confidence_interval_level: float = 0.95,
    n_jobs: int = 1
) -> dict:
    assert isinstance(environment, Environment)
    assert isinstance(policy, Policy)
    assert isinstance(initial_belief, Belief)
    assert isinstance(discount_factor, float)
    assert isinstance(num_episodes, int)
    assert isinstance(num_steps, int)
    assert isinstance(confidence_interval_level, float)
    
    assert 1 >= confidence_interval_level >= 0
    assert num_episodes > 0
    assert num_steps > 0
    
    histories = run_multiple_episodes(
        environment=environment, 
        policy=policy, 
        initial_belief=initial_belief, 
        discount_factor=discount_factor, 
        num_episodes=num_episodes, 
        num_steps=num_steps,
        n_jobs=n_jobs
    )
    
    statistics = compute_statistics(
        histories=histories, 
        alpha=alpha, 
        confidence_interval_level=confidence_interval_level
    )
    
    return histories, statistics

def compare_planners(
    environment_policy_pairs: List[Tuple[Environment, Policy]],
    discount_factor: float,
    num_episodes: int,
    num_steps: int,
    alpha: float,
    n_particles: int,
    cache_dir_path: Path,
    experiment_name: str = "POMDP_Planning_Comparison",
    confidence_interval_level: float = 0.95,
    n_jobs: int = 1
) -> Tuple[List[dict], pd.DataFrame]:
    assert isinstance(environment_policy_pairs, List)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in environment_policy_pairs)
    assert all(isinstance(env, Environment) for env, _ in environment_policy_pairs)
    assert all(isinstance(policy, Policy) for _, policy in environment_policy_pairs)
    assert isinstance(discount_factor, float)
    assert isinstance(num_episodes, int)
    assert isinstance(num_steps, int)
    assert isinstance(alpha, float)
    assert isinstance(n_particles, int)
    assert isinstance(cache_dir_path, Path)
    assert isinstance(confidence_interval_level, float)
    
    assert num_episodes > 0
    assert num_steps > 0
    assert n_particles > 0
    assert 1 >= discount_factor >= 0

    # Set up MLFlow tracking with proper file scheme for Windows
    mlruns_path = cache_dir_path / "mlruns"
    mlruns_path.mkdir(parents=True, exist_ok=True)
    # Use empty string for default local storage
    tracking_uri = mlruns_path
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    planner_statistics = []
    aggregated_data = []

    for pair_idx, (environment, policy) in enumerate(environment_policy_pairs):
        # End any active run safely
        active_run = mlflow.active_run()
        if active_run is not None:
            mlflow.end_run()
        
        # Start a new MLFlow run for each environment-policy combination
        with mlflow.start_run(run_name=f"{environment.__class__.__name__}_{policy.__class__.__name__}_{pair_idx}"):
            # Log common parameters
            common_params = {
                "environment_type": environment.__class__.__name__,
                "policy_type": policy.__class__.__name__,
                "discount_factor": discount_factor,
                "num_episodes": num_episodes,
                "num_steps": num_steps,
                "alpha": alpha,
                "n_particles": n_particles,
                "confidence_interval_level": confidence_interval_level
            }
            
            # Log environment-specific parameters
            env_params = {
                f"env_{key}": value 
                for key, value in environment.__dict__.items() 
                if isinstance(value, (int, float, str, bool))
            }
            
            # Log policy-specific parameters
            policy_params = {
                f"policy_{key}": value 
                for key, value in policy.__dict__.items() 
                if isinstance(value, (int, float, str, bool))
            }
            
            # Combine all parameters
            all_params = {
                **common_params,
                **env_params,
                **policy_params
            }
            
            mlflow.log_params(all_params)
            
            initial_belief = get_initial_belief(pomdp=environment, n_particles=n_particles)
            histories, statistics = simulation(
                environment=environment,
                policy=policy,
                initial_belief=initial_belief,
                discount_factor=discount_factor,
                num_episodes=num_episodes,
                num_steps=num_steps,
                alpha=alpha,
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs
            )
            
            # Log metrics from statistics
            for metric_name, (metric_value, confidence_interval) in statistics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)
                    # Log confidence interval bounds
                    mlflow.log_metric(f"{metric_name}_ci_lower", confidence_interval[0])
                    mlflow.log_metric(f"{metric_name}_ci_upper", confidence_interval[1])
                elif isinstance(metric_value, dict):
                    # For nested statistics, flatten them
                    for sub_name, (sub_value, sub_ci) in metric_value.items():
                        if isinstance(sub_value, (int, float)):
                            mlflow.log_metric(f"{metric_name}_{sub_name}", sub_value)
                            mlflow.log_metric(f"{metric_name}_{sub_name}_ci_lower", sub_ci[0])
                            mlflow.log_metric(f"{metric_name}_{sub_name}_ci_upper", sub_ci[1])
            
            # Save the full statistics as a JSON artifact
            full_data = {
                "statistics": statistics,
                "parameters": all_params
            }
            mlflow.log_dict(full_data, "statistics/full_data.json")
            
            planner_statistics.append(statistics)
            
            # Aggregate data for DataFrame
            run_data = {
                **all_params,
                **statistics
            }
            aggregated_data.append(run_data)
    
    # Convert aggregated data to DataFrame
    df = pd.DataFrame(aggregated_data)
    
    # Log DataFrame as table artifact
    mlflow.log_table(df, "dataframes/planner_comparison.json")
    
    # Plot statistics comparison
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        plot_statistics_comparison(
            statistics=planner_statistics,
            environments=[env for env, _ in environment_policy_pairs],
            policies=[policy for _, policy in environment_policy_pairs],
            cache_dir_path=temp_dir_path
        )
        
        # Log all generated plots as artifacts
        for plot_file in temp_dir_path.glob("*.png"):
            mlflow.log_artifact(str(plot_file), "statistics_plots")
    
    # End any active run safely before returning
    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.end_run()
    
    return planner_statistics, df

def create_policy_optimization_objective(
    policy_class: Type[Policy], 
    param_ranges: dict, 
    evaluation_function: callable,
    environment: Environment = None,
    discount_factor: float = None
) -> callable:
    """
    Creates an Optuna objective function for optimizing policy parameters.
    
    Args:
        policy_class: The Policy class to optimize
        param_ranges: Dictionary defining parameter ranges and types
        evaluation_function: Function that takes a policy and returns a score to optimize
        environment: The environment to use for policy creation (if required)
        discount_factor: The discount factor to use for policy creation (if required)
        
    Returns:
        An objective function suitable for Optuna optimization
    """
    assert isinstance(policy_class, type)
    assert issubclass(policy_class, Policy)
    assert isinstance(param_ranges, dict)
    assert callable(evaluation_function)
    if environment is not None:
        assert isinstance(environment, Environment)
    if discount_factor is not None:
        assert isinstance(discount_factor, float)
        assert 0 <= discount_factor <= 1
    
    def objective(trial):
        # Create parameters dictionary from ranges
        policy_params = {}
        
        # Add required parameters first if provided
        if environment is not None:
            policy_params['environment'] = environment
        if discount_factor is not None:
            policy_params['discount_factor'] = discount_factor
            
        # Add optimization parameters
        for param_name, param_config in param_ranges.items():
            if param_config['type'] == 'int':
                policy_params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config['low'], 
                    param_config['high']
                )
            elif param_config['type'] == 'float':
                policy_params[param_name] = trial.suggest_float(
                    param_name, 
                    param_config['low'], 
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_config['type'] == 'categorical':
                policy_params[param_name] = trial.suggest_categorical(
                    param_name, 
                    param_config['choices']
                )
        
        # Create policy instance with suggested parameters
        policy = policy_class(**policy_params)
        
        # Evaluate policy and return objective value
        result = evaluation_function(policy)
        # Handle tuple return values (e.g., from simulation statistics)
        if isinstance(result, tuple):
            return result[0]  # Return the mean value
        return result
    
    return objective

def optimize_policy_parameters_with_optuna(
    environment: Environment,
    policy_class: Type[Policy],
    param_ranges: dict,  # TODO: Change to list of HyperParameterSearch
    discount_factor: float,
    num_episodes: int,
    num_steps: int,
    n_particles: int,
    cache_dir_path: Path,
    parameter_to_optimize: str = "average_return",
    direction: str = "maximize",
    n_trials: int = 100,
    confidence_interval_level: float = 0.95,
    n_jobs: int = 1
) -> Tuple[dict, float, List[History]]:
    """
    Optimizes policy parameters using Optuna.
    
    Args:
        environment: The environment to evaluate policies in
        policy_class: The Policy class to optimize
        param_ranges: Dictionary defining parameter ranges and types
        discount_factor: Discount factor for reward calculation
        num_episodes: Number of episodes to run per evaluation
        num_steps: Number of steps per episode
        n_particles: Number of particles for belief representation
        cache_dir_path: Path to store optimization results
        parameter_to_optimize: Name of the statistic to optimize
        n_trials: Number of optimization trials
        confidence_interval_level: Confidence level for statistics
        
    Returns:
        Tuple containing:
        - Dictionary of best parameters found
        - Best objective value achieved
        - List of histories from the best trial
    """
    assert isinstance(environment, Environment)
    assert isinstance(policy_class, type)
    assert issubclass(policy_class, Policy)
    assert isinstance(param_ranges, dict)
    assert isinstance(discount_factor, float)
    assert isinstance(num_episodes, int)
    assert isinstance(num_steps, int)
    assert isinstance(n_particles, int)
    assert isinstance(cache_dir_path, Path)
    assert isinstance(parameter_to_optimize, str)
    assert isinstance(n_trials, int)
    assert isinstance(confidence_interval_level, float)
    
    assert num_episodes > 0
    assert num_steps > 0
    assert n_particles > 0
    assert 0 <= discount_factor <= 1
    assert 0 <= confidence_interval_level <= 1
    
    def evaluation_function(policy: Policy, trial: optuna.Trial) -> float:
        initial_belief = get_initial_belief(pomdp=environment, n_particles=n_particles)
        histories, statistics = simulation(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            discount_factor=discount_factor,
            num_episodes=num_episodes,
            num_steps=num_steps,
            alpha=0.05,  # Fixed alpha for optimization
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs
        )
        
        # Store histories as trial user attributes
        trial.set_user_attr('histories', histories)
        
        # Get the current value
        current_value = statistics[parameter_to_optimize]
        if isinstance(current_value, tuple):
            current_value = current_value[0]  # Get the mean value
            
        return current_value
    
    def objective(trial: optuna.Trial) -> float:
        # Create parameters dictionary from ranges
        policy_params = {}
        
        # Add required parameters first
        policy_params['environment'] = environment
        policy_params['discount_factor'] = discount_factor
            
        # Add optimization parameters
        for param_name, param_config in param_ranges.items():
            if param_config['type'] == 'int':
                policy_params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config['low'], 
                    param_config['high']
                )
            elif param_config['type'] == 'float':
                policy_params[param_name] = trial.suggest_float(
                    param_name, 
                    param_config['low'], 
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_config['type'] == 'categorical':
                policy_params[param_name] = trial.suggest_categorical(
                    param_name, 
                    param_config['choices']
                )
        
        # Create policy instance with suggested parameters
        policy = policy_class(**policy_params)
        
        # Evaluate policy and return objective value
        return evaluation_function(policy, trial)
    
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    
    # Get histories from the best trial
    best_histories = study.best_trial.user_attrs['histories']
    
    # Remove environment and discount_factor from best params as they were not optimized
    best_params = {k: v for k, v in study.best_trial.params.items() 
                  if k not in ['environment', 'discount_factor']}
    
    return best_params, study.best_value, best_histories

def optimize_policy_parameters_for_multiple_environments(
    environment_policy_pairs: List[Tuple[Environment, Tuple[Type[Policy], dict]]],
    discount_factor: float,
    num_episodes: int,
    num_steps: int,
    n_particles: int,
    cache_dir_path: Path,
    parameter_to_optimize: str = "average_return",
    direction: str = "maximize",
    n_trials: int = 100,
    confidence_interval_level: float = 0.95,
    experiment_name: str = "POMDP_Parameter_Optimization",
    n_jobs: int = 1
) -> Tuple[List[Tuple[dict, float, List[History]]], pd.DataFrame]:
    assert isinstance(environment_policy_pairs, List)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in environment_policy_pairs)
    assert all(isinstance(env, Environment) for env, _ in environment_policy_pairs)
    assert all(isinstance(policy_config, tuple) and len(policy_config) == 2 for _, policy_config in environment_policy_pairs)
    assert all(isinstance(policy_class, type) and issubclass(policy_class, Policy) for _, (policy_class, _) in environment_policy_pairs)
    assert all(isinstance(param_ranges, dict) for _, (_, param_ranges) in environment_policy_pairs)
    assert isinstance(discount_factor, float)
    assert isinstance(num_episodes, int)
    assert isinstance(num_steps, int)
    assert isinstance(n_particles, int)
    assert isinstance(cache_dir_path, Path)
    assert isinstance(parameter_to_optimize, str)
    assert isinstance(direction, str)
    assert isinstance(n_trials, int)
    assert isinstance(confidence_interval_level, float)
    assert isinstance(experiment_name, str)
    
    assert num_episodes > 0
    assert num_steps > 0
    assert n_particles > 0
    assert 0 <= discount_factor <= 1
    assert 0 <= confidence_interval_level <= 1
    assert direction in ["maximize", "minimize"]
    
    # Set up MLFlow tracking
    mlruns_path = cache_dir_path / "mlruns"
    mlruns_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = mlruns_path
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    results = []
    aggregated_data = []
    planner_statistics = []
    
    for pair_idx, (environment, (policy_class, param_ranges)) in enumerate(environment_policy_pairs):
        # End any active run safely
        active_run = mlflow.active_run()
        if active_run is not None:
            mlflow.end_run()
        
        # Start a new MLFlow run for each environment-policy combination
        with mlflow.start_run(run_name=f"pair_{pair_idx}"):
            # Log common parameters
            common_params = {
                "environment_type": environment.__class__.__name__,
                "policy_type": policy_class.__name__,
                "discount_factor": discount_factor,
                "num_episodes": num_episodes,
                "num_steps": num_steps,
                "n_particles": n_particles,
                "parameter_to_optimize": parameter_to_optimize,
                "direction": direction,
                "n_trials": n_trials,
                "confidence_interval_level": confidence_interval_level
            }
            
            # Log environment-specific parameters
            env_params = {
                f"env_{key}": value 
                for key, value in environment.__dict__.items() 
                if isinstance(value, (int, float, str, bool))
            }
            
            # Log policy-specific parameter ranges
            policy_param_ranges = {
                f"param_range_{param_name}": f"{param_config['low']}-{param_config['high']}"
                for param_name, param_config in param_ranges.items()
            }
            
            # Combine all parameters
            all_params = {
                **common_params,
                **env_params,
                **policy_param_ranges
            }
            
            mlflow.log_params(all_params)
            
            # Run optimization
            best_params, best_value, histories = optimize_policy_parameters_with_optuna(
                environment=environment,
                policy_class=policy_class,
                param_ranges=param_ranges,
                discount_factor=discount_factor,
                num_episodes=num_episodes,
                num_steps=num_steps,
                n_particles=n_particles,
                cache_dir_path=cache_dir_path,
                parameter_to_optimize=parameter_to_optimize,
                direction=direction,
                n_trials=n_trials,
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs
            )
            
            # Get statistics from the best trial
            initial_belief = get_initial_belief(pomdp=environment, n_particles=n_particles)
            _, statistics = simulation(
                environment=environment,
                policy=policy_class(environment=environment, discount_factor=discount_factor, **best_params),
                initial_belief=initial_belief,
                discount_factor=discount_factor,
                num_episodes=num_episodes,
                num_steps=num_steps,
                alpha=0.05,  # Fixed alpha for evaluation
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs
            )
            
            # Log best parameters and value
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_value", best_value)
            
            # Save the full results as a JSON artifact using log_dict
            results_data = {
                "best_parameters": best_params,
                "best_value": best_value,
                "parameters": all_params,
                "param_ranges": param_ranges,
                "statistics": statistics
            }
            mlflow.log_dict(results_data, f"optimization_results_pair_{pair_idx}.json")
            
            results.append((best_params, best_value, histories))
            planner_statistics.append(statistics)
            
            # Aggregate data for DataFrame
            run_data = {
                **all_params,
                **{f"best_{k}": v for k, v in best_params.items()},
                "best_value": best_value,
                **statistics
            }
            aggregated_data.append(run_data)
    
    # Convert aggregated data to DataFrame
    df = pd.DataFrame(aggregated_data)
    
    # Log DataFrame as table artifact through MLFlow
    mlflow.log_table(df, "optimization_results.json")
    
    # Plot statistics comparison
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        plot_statistics_comparison(
            statistics=planner_statistics,
            environments=[env for env, _ in environment_policy_pairs],
            policies=[policy_class for _, (policy_class, _) in environment_policy_pairs],
            cache_dir_path=temp_dir_path
        )
        
        # Log all generated plots as artifacts
        for plot_file in temp_dir_path.glob("*.png"):
            mlflow.log_artifact(str(plot_file), "statistics_plots")
    
    # End any active run safely before returning
    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.end_run()
    
    return results, df
