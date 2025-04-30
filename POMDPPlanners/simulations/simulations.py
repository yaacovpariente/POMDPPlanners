from typing import List, Type, Tuple, Dict
import copy
from time import time
from pathlib import Path
import mlflow
import json
import os
import logging
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import optuna
import pandas as pd
import numpy as np
import tempfile
from tqdm import tqdm
from typing import Union, List, Tuple, Type
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import Belief, get_initial_belief
from POMDPPlanners.core.simulation import (
    History,
    StepData,
    CategoricalHyperParameter,
    NumericalHyperParameter,
    MetricValue,
)
from POMDPPlanners.simulations.simulation_statistics import compute_statistics_environment_policy_pair, compute_statistics_environments_policies_comparison
from POMDPPlanners.utils.visualization import plot_metrics_comparison, plot_discounted_returns_histogram, plot_environment_policy_pair_comparison, plot_discounted_returns_histogram_multiple_policies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

HyperParameterFeatures = Union[CategoricalHyperParameter, NumericalHyperParameter]


def run_multiple_episodes(
    environment: Environment,
    policy: Policy,
    initial_belief: Belief,
    num_episodes: int,
    num_steps: int,
    n_jobs: int = 1,
) -> List[History]:
    logger.info(f"Starting {num_episodes} episodes with {num_steps} steps each using {n_jobs} jobs")
    logger.info(f"Environment: {environment.name}, Policy: {policy.name}")
    assert isinstance(environment, Environment)
    assert isinstance(policy, Policy)
    assert isinstance(initial_belief, Belief)

    assert num_episodes > 0
    assert num_steps > 0

    # Create a list of arguments for each episode
    episode_args = [
        (environment, policy, initial_belief, num_steps) for _ in range(num_episodes)
    ]

    # Run episodes in parallel using joblib with progress bar
    logger.info(f"Running episodes in parallel for {environment.name} with {policy.name}")
    histories = Parallel(n_jobs=n_jobs)(
        delayed(run_episode)(*args) for args in tqdm(
            episode_args,
            total=num_episodes,
            desc=f"Running episodes for {environment.name} with {policy.name}",
            unit="episode"
        )
    )
    logger.info(f"All episodes completed for {environment.name} with {policy.name}")

    return histories


def run_episode(
    environment: Environment, policy: Policy, initial_belief: Belief, num_steps: int
) -> History:
    logger.debug(f"Starting episode with {num_steps} steps")
    
    assert isinstance(environment, Environment)
    assert isinstance(policy, Policy)
    assert isinstance(initial_belief, Belief)
    assert num_steps > 0

    average_state_sampling_time = 0.0
    average_action_time = 0.0
    average_observation_time = 0.0
    average_belief_update_time = 0.0
    average_reward_time = 0.0
    actual_num_steps = 0
    reach_terminal_state = False

    belief = copy.deepcopy(initial_belief)
    state = belief.sample()

    history = []
    for i in range(1, num_steps + 1):
        # TODO: add count of terminal states.
        if environment.is_terminal(state=state):
            reach_terminal_state = True
            break

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
        average_state_sampling_time = (
            average_state_sampling_time * (i - 1) + state_sampling_time
        ) / i

        observation_start_time = time()
        observation = environment.observation_model(next_state, action).sample()
        observation_time = time() - observation_start_time
        average_observation_time = (
            average_observation_time * (i - 1) + observation_time
        ) / i

        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=reward,
            )
        )

        belief_update_start_time = time()
        belief = belief.update(
            action=action, observation=observation, pomdp=environment
        )
        belief_update_time = time() - belief_update_start_time
        average_belief_update_time = (
            average_belief_update_time * (i - 1) + belief_update_time
        ) / i

        actual_num_steps += 1
        state = next_state
        
    logger.debug(f"Episode completed with average times: action={average_action_time:.4f}s, "
                f"reward={average_reward_time:.4f}s, state_sampling={average_state_sampling_time:.4f}s, "
                f"observation={average_observation_time:.4f}s, belief_update={average_belief_update_time:.4f}s")

    return History(
        history=history,
        discount_factor=environment.discount_factor,
        average_state_sampling_time=average_state_sampling_time,
        average_action_time=average_action_time,
        average_observation_time=average_observation_time,
        average_belief_update_time=average_belief_update_time,
        average_reward_time=average_reward_time,
        actual_num_steps=actual_num_steps,
        reach_terminal_state=reach_terminal_state,
    )


def simulation(
    environment: Environment,
    policy: Policy,
    initial_belief: Belief,
    num_episodes: int,
    num_steps: int,
    alpha: float,
    confidence_interval_level: float = 0.95,
    n_jobs: int = 1,
) -> Tuple[List[History], List[MetricValue]]:
    logger.info(f"Starting simulation with {num_episodes} episodes, {num_steps} steps, "
                f"alpha={alpha}, confidence_interval={confidence_interval_level}")
    assert isinstance(environment, Environment)
    assert isinstance(policy, Policy)
    assert isinstance(initial_belief, Belief)
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
        num_episodes=num_episodes,
        num_steps=num_steps,
        n_jobs=n_jobs,
    )

    logger.info("Computing statistics from simulation results")
    statistics = compute_statistics_environment_policy_pair(
        env=environment,
        histories=histories,
        alpha=alpha,
        confidence_interval_level=confidence_interval_level,
    )
    logger.info("Statistics computation completed")

    return histories, statistics


def simulate_multiple_policies(
    environment: Environment,
    policies: List[Policy],
    initial_belief: Belief,
    num_episodes: int,
    num_steps: int,
    alpha: float,
    confidence_interval_level: float = 0.95,
    n_jobs: int = 1,
) -> Tuple[Dict[str, List[History]], List[List[MetricValue]]]:
    """
    Simulate multiple policies on a single environment.

    Args:
        environment: The environment to evaluate policies in
        policies: List of policies to evaluate
        initial_belief: Initial belief state
        num_episodes: Number of episodes to run per policy
        num_steps: Number of steps per episode
        alpha: Alpha value for statistics computation
        confidence_interval_level: Confidence level for statistics
        n_jobs: Number of parallel jobs for simulation

    Returns:
        Tuple containing:
        - Dictionary mapping policy names to their histories
    """
    assert isinstance(environment, Environment)
    assert isinstance(policies, list)
    assert all(isinstance(policy, Policy) for policy in policies)
    assert isinstance(initial_belief, Belief)
    assert isinstance(num_episodes, int)
    assert isinstance(num_steps, int)
    assert isinstance(alpha, float)
    assert isinstance(confidence_interval_level, float)
    assert isinstance(n_jobs, int)

    policy_names = [policy.name for policy in policies]
    assert len(policy_names) == len(set(policy_names)), "All policies must have unique names"

    assert num_episodes > 0
    assert num_steps > 0
    assert 0 <= confidence_interval_level <= 1

    all_histories = {}
    all_statistics = []

    for policy in policies:
        # Run simulation
        histories, statistics = simulation(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            num_episodes=num_episodes,
            num_steps=num_steps,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
        )

        all_histories[policy.name] = histories
        all_statistics.append(statistics)

    return all_histories


def simulate_multiple_environments_and_policies(
    environments: List[Tuple[Environment, Belief]],
    policies: List[Policy],
    num_episodes: int,
    num_steps: int,
    alpha: float,
    confidence_interval_level: float = 0.95,
    n_jobs: int = 1,
) -> Dict[str, Dict[str, List[History]]]:
    """
    Simulate multiple policies on multiple environments.

    Args:
        environments: List of tuples containing (environment, initial_belief)
        policies: List of policies to evaluate
        num_episodes: Number of episodes to run per policy
        num_steps: Number of steps per episode
        alpha: Alpha value for statistics computation
        confidence_interval_level: Confidence level for statistics
        n_jobs: Number of parallel jobs for simulation

    Returns:
        Dictionary mapping environment names to dictionaries of policy histories
    """
    assert isinstance(environments, list)
    assert all(isinstance(env, tuple) and len(env) == 2 for env in environments)
    assert all(isinstance(env, Environment) and isinstance(belief, Belief) for env, belief in environments)
    assert isinstance(policies, list)
    assert all(isinstance(policy, Policy) for policy in policies)
    assert isinstance(num_episodes, int)
    assert isinstance(num_steps, int)
    assert isinstance(alpha, float)
    assert isinstance(confidence_interval_level, float)
    assert isinstance(n_jobs, int)

    # Verify unique environment names
    env_names = [env.name for env, _ in environments]
    assert len(env_names) == len(set(env_names)), "All environments must have unique names"

    # Verify unique policy names
    policy_names = [policy.name for policy in policies]
    assert len(policy_names) == len(set(policy_names)), "All policies must have unique names"

    assert num_episodes > 0
    assert num_steps > 0
    assert 0 <= confidence_interval_level <= 1

    results = {}
    for environment, initial_belief in environments:
        # Run simulations for all policies on this environment
        policy_histories = simulate_multiple_policies(
            environment=environment,
            policies=policies,
            initial_belief=initial_belief,
            num_episodes=num_episodes,
            num_steps=num_steps,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
        )
        results[environment.name] = policy_histories

    return results


def compare_multiple_environments_policies(
    environment_belief_policy_tuples: List[Tuple[Environment, Belief, List[Policy]]],
    num_episodes: int,
    num_steps: int,
    alpha: float,
    confidence_interval_level: float = 0.95,
    n_jobs: int = 1,
    cache_dir_path: Path = None,
    experiment_name: str = "POMDP_Planning_Comparison",
    cache_visualizations: bool = True,
) -> Tuple[Dict[str, Dict[str, List[History]]], pd.DataFrame]:
    """
    Compare multiple policies on multiple environments and cache results in MLFlow.
    
    Args:
        environment_belief_policy_tuples: List of tuples containing (environment, belief, policies)
        num_episodes: Number of episodes to run per policy
        num_steps: Number of steps per episode
        alpha: Alpha value for statistics computation
        confidence_interval_level: Confidence level for statistics
        n_jobs: Number of parallel jobs for simulation
        cache_dir_path: Path to store results (if None, uses current directory)
        experiment_name: Name of the MLFlow experiment
        
    Returns:
        Tuple containing:
        - Dictionary mapping environment names to dictionaries of policy histories
        - DataFrame with statistics for all environment-policy pairs
    """
    assert isinstance(environment_belief_policy_tuples, list)
    assert all(isinstance(tup, tuple) and len(tup) == 3 for tup in environment_belief_policy_tuples)
    assert all(isinstance(env, Environment) and isinstance(belief, Belief) and isinstance(policies, list) 
              for env, belief, policies in environment_belief_policy_tuples)
    assert all(all(isinstance(policy, Policy) for policy in policies) 
              for _, _, policies in environment_belief_policy_tuples)
    assert isinstance(num_episodes, int)
    assert isinstance(num_steps, int)
    assert isinstance(alpha, float)
    assert isinstance(confidence_interval_level, float)
    assert isinstance(n_jobs, int)
    if cache_dir_path is not None:
        assert isinstance(cache_dir_path, Path)
    assert isinstance(experiment_name, str)

    # Verify unique environment names
    env_names = [env.name for env, _, _ in environment_belief_policy_tuples]
    assert len(env_names) == len(set(env_names)), "All environments must have unique names"

    # Verify unique policy names across all environments
    all_policies = [policy for _, _, policies in environment_belief_policy_tuples for policy in policies]
    policy_names = [policy.name for policy in all_policies]
    assert len(policy_names) == len(set(policy_names)), "All policies must have unique names"

    assert num_episodes > 0
    assert num_steps > 0
    assert 0 <= confidence_interval_level <= 1

    # Set up MLFlow tracking
    if cache_dir_path is None:
        cache_dir_path = Path.cwd()
    mlruns_path = cache_dir_path / "mlruns"
    mlruns_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = mlruns_path
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLFlow tracking set up at {tracking_uri}")

    # Cache results in MLFlow
    with mlflow.start_run(run_name="environment_policy_comparison"):
        # Run simulations
        histories = {}
        for environment, initial_belief, policies in environment_belief_policy_tuples:
            # Run simulations for policies on this environment
            policy_histories = simulate_multiple_policies(
                environment=environment,
                policies=policies,
                initial_belief=initial_belief,
                num_episodes=num_episodes,
                num_steps=num_steps,
                alpha=alpha,
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs,
            )
            histories[environment.name] = policy_histories

        # Compute statistics
        statistics_df = compute_statistics_environments_policies_comparison(
            histories=histories,
            environments=[env for env, _, _ in environment_belief_policy_tuples],
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
        )

        # Log statistics DataFrame
        mlflow.log_table(statistics_df, "statistics/comparison_results.json")

        # Create results directory
        results_dir = cache_dir_path / "results"
        results_dir.mkdir(exist_ok=True)
            
        for env_name, policy_histories_dict in histories.items():
            environment = next(env for env, _, _ in environment_belief_policy_tuples if env.name == env_name)
            env_dir = results_dir / env_name
            env_dir.mkdir(exist_ok=True)
            
            for policy_name, policy_histories in policy_histories_dict.items():
                policy = next(policy for _, _, policies in environment_belief_policy_tuples 
                            for policy in policies if policy.name == policy_name)
                
                # Create policy directory
                policy_dir = env_dir / policy_name
                policy_dir.mkdir(exist_ok=True)
                
                # Create plots directory
                plots_dir = policy_dir / "plots"
                plots_dir.mkdir(exist_ok=True)
                
                # Create and save plots
                plot_path = plots_dir / "discounted_returns_histogram.png"
                plot_discounted_returns_histogram(
                    histories=policy_histories,
                    policy=policy,
                    environment=environment,
                    cache_path=plot_path
                )
                
                # Create visualizations directory
                viz_dir = policy_dir / "visualizations"
                viz_dir.mkdir(exist_ok=True)
                
                if cache_visualizations:
                    # Cache visualizations for each history
                    for episode_idx, history in enumerate(policy_histories):
                        file_name = f"agent_path_{episode_idx}.gif"
                        cache_path = viz_dir / file_name
                        
                        try:
                            environment.cache_visualization(
                                history=history,
                                cache_path=cache_path
                            )
                        except Exception as e:
                            logger.warning(f"Visualization failed for episode {episode_idx}: {str(e)}")
                            continue
            
            # Create comparison plot for all policies in this environment
            comparison_plot_path = env_dir / "policy_comparison_histogram.png"
            # Get the policies that correspond to the histories in policy_histories_dict
            policies_for_plot = [policy for policy in policies if policy.name in policy_histories_dict]
            plot_discounted_returns_histogram_multiple_policies(
                histories=policy_histories_dict,
                policies=policies_for_plot,
                environment=environment,
                cache_path=comparison_plot_path
            )
        
        # Log all generated plots and visualizations as artifacts
        mlflow.log_artifact(str(results_dir), ".")

    return histories, statistics_df


def _process_environment_policy_pair(
    pair_idx: int,
    environment_policy_pair: Tuple[Environment, Policy, Belief],
    num_episodes: int,
    num_steps: int,
    alpha: float,
    confidence_interval_level: float,
    n_jobs: int,
    experiment_name: str,
) -> Tuple[List[MetricValue], dict]:
    """Process a single environment-policy pair and return its statistics and run data."""
    environment, policy, initial_belief = environment_policy_pair
    
    # Start a new MLFlow run for each environment-policy combination
    with mlflow.start_run(
        run_name=f"{environment.__class__.__name__}_{policy.__class__.__name__}_{pair_idx}"
    ):
        # Log common parameters
        common_params = {
            "environment_type": environment.__class__.__name__,
            "policy_type": policy.__class__.__name__,
            "num_episodes": num_episodes,
            "num_steps": num_steps,
            "alpha": alpha,
            "confidence_interval_level": confidence_interval_level,
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
        all_params = {**common_params, **env_params, **policy_params}

        mlflow.log_params(all_params)

        # Run simulation
        histories, statistics = simulation(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            num_episodes=num_episodes,
            num_steps=num_steps,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
        )

        # Log metrics from statistics
        for metric in statistics:
            mlflow.log_metric(metric.name, metric.value)
            mlflow.log_metric(f"{metric.name}_ci_lower", metric.lower_confidence_bound)
            mlflow.log_metric(f"{metric.name}_ci_upper", metric.upper_confidence_bound)

        # Save the full statistics as a JSON artifact
        full_data = {
            "statistics": [metric._asdict() for metric in statistics],
            "parameters": all_params,
        }
        mlflow.log_dict(full_data, "statistics/full_data.json")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Create and save return histogram
            histogram_path = temp_dir_path / "returns_histogram.png"
            plot_discounted_returns_histogram(histories, histogram_path)
            mlflow.log_artifact(str(histogram_path), "visualization/returns_histogram.png")

            for episode_idx, history in enumerate(histories):
                episode_dir = temp_dir_path / f"episode_{episode_idx}"
                file_name = f"agent_path_{episode_idx}.gif"
                cache_path = episode_dir / file_name
                episode_dir.mkdir(exist_ok=True)

                try:
                    environment.cache_visualization(
                        history=history, cache_path=cache_path
                    )
                    if cache_path.exists():
                        mlflow.log_artifact(
                            str(cache_path), f"visualization/{file_name}"
                        )
                except Exception as e:
                    logger.warning(f"Visualization failed for episode {episode_idx}: {str(e)}")
                    pass

        # Aggregate data for DataFrame
        run_data = {
            **all_params,
            **{metric.name: metric.value for metric in statistics},
        }

        return statistics, run_data


def compare_planners(
    environment_policy_pairs: List[Tuple[Environment, Policy, Belief]],
    num_episodes: int,
    num_steps: int,
    alpha: float,
    n_particles: int,
    cache_dir_path: Path,
    resampling: bool = True,
    experiment_name: str = "POMDP_Planning_Comparison",
    confidence_interval_level: float = 0.95,
    n_jobs: int = 1,
) -> Tuple[List[List[MetricValue]], pd.DataFrame]:
    logger.info(f"Starting planner comparison with {len(environment_policy_pairs)} pairs, "
                f"{num_episodes} episodes, {num_steps} steps")
    assert isinstance(environment_policy_pairs, List)
    assert len(environment_policy_pairs) > 0

    for env, pol, belief in environment_policy_pairs:
        assert isinstance(env, Environment)
        assert isinstance(pol, Policy)
        assert isinstance(belief, Belief)

    assert isinstance(num_episodes, int)
    assert isinstance(num_steps, int)
    assert isinstance(alpha, float)
    assert isinstance(n_particles, int)
    assert isinstance(cache_dir_path, Path)
    assert isinstance(confidence_interval_level, float)
    assert isinstance(resampling, bool)

    assert num_episodes > 0
    assert num_steps > 0
    assert n_particles > 0

    # Set up MLFlow tracking with proper file scheme for Windows
    mlruns_path = cache_dir_path / "mlruns"
    mlruns_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = mlruns_path
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLFlow tracking set up at {tracking_uri}")

    # Process environment-policy pairs in parallel
    cpu_number = cpu_count()
    if n_jobs == -1:
        n_jobs = cpu_number
    else:
        if cpu_number < n_jobs:
            n_jobs = cpu_number

    if num_episodes > n_jobs:
        n_jobs_pairs = 1
        n_jobs_process = cpu_number
    else:
        n_jobs_pairs = min(len(environment_policy_pairs), cpu_number)
        n_jobs_process = int(cpu_number / n_jobs_pairs)
    
    logger.info(f"Using {n_jobs_pairs} jobs for parallel processing of environment-policy pairs")
    logger.info(f"Using {n_jobs_process} jobs for internal processing within each pair")
    
    results = Parallel(n_jobs=n_jobs_pairs)(
        delayed(_process_environment_policy_pair)(
            pair_idx=pair_idx,
            environment_policy_pair=pair,
            num_episodes=num_episodes,
            num_steps=num_steps,
            alpha=alpha,
            n_particles=n_particles,
            cache_dir_path=cache_dir_path,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs_process,  # Use single job for each pair's internal processing
            experiment_name=experiment_name,
        )
        for pair_idx, pair in enumerate(environment_policy_pairs)
    )

    # Unpack results
    planner_statistics, aggregated_data = zip(*results)

    # Convert aggregated data to DataFrame
    df = pd.DataFrame(aggregated_data)
    logger.info("Created DataFrame with aggregated results")

    with mlflow.start_run(run_name="stats and plots"):
        # Log DataFrame as table artifact
        mlflow.log_table(df, "dataframes/planner_comparison.json")

        # Plot statistics comparison
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            logger.info("Generating statistics comparison plots")
            plot_metrics_comparison(
                statistics=planner_statistics,
                environments=[env for env, _, _ in environment_policy_pairs],
                policies=[policy for _, policy, _ in environment_policy_pairs],
                cache_dir_path=temp_dir_path,
            )

            # Log all generated plots as artifacts
            for plot_file in (temp_dir_path / "plots").glob("*.png"):
                mlflow.log_artifact(
                    str(plot_file), f"statistics_plots/{str(plot_file.name)}"
                )

    # End any active run safely before returning
    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.end_run()

    logger.info("Planner comparison completed successfully")
    return planner_statistics, df


def create_policy_optimization_objective(
    policy_class: Type[Policy],
    param_ranges: dict,
    evaluation_function: callable,
    environment: Environment = None,
    discount_factor: float = None,
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
            policy_params["environment"] = environment
        if discount_factor is not None:
            policy_params["discount_factor"] = discount_factor

        # Add optimization parameters
        for param_name, param_config in param_ranges.items():
            if param_config["type"] == "int":
                policy_params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "float":
                policy_params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_config["type"] == "categorical":
                policy_params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
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
    param_ranges: List[HyperParameterFeatures],
    num_episodes: int,
    num_steps: int,
    n_particles: int,
    cache_dir_path: Path,
    parameter_to_optimize: str = "average_return",
    direction: str = "maximize",
    n_trials: int = 100,
    confidence_interval_level: float = 0.95,
    n_jobs: int = 1,
) -> Tuple[dict, float, List[History]]:
    """
    Optimizes policy parameters using Optuna.

    Args:
        environment: The environment to evaluate policies in
        policy_class: The Policy class to optimize
        param_ranges: List of hyperparameters to optimize
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
    assert isinstance(param_ranges, list)
    assert all(isinstance(param, HyperParameterFeatures) for param in param_ranges)
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
    assert 0 <= confidence_interval_level <= 1

    def evaluation_function(policy: Policy, trial: optuna.Trial) -> float:
        initial_belief = get_initial_belief(pomdp=environment, n_particles=n_particles)
        histories, statistics = simulation(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            num_episodes=num_episodes,
            num_steps=num_steps,
            alpha=0.05,  # Fixed alpha for optimization
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
        )

        # Store histories as trial user attributes
        trial.set_user_attr("histories", histories)

        # Get the current value
        for metric in statistics:
            if metric.name == parameter_to_optimize:
                return metric.value

        raise ValueError(f"Parameter {parameter_to_optimize} not found in statistics")

    def objective(trial: optuna.Trial) -> float:
        # Create parameters dictionary from hyperparameters
        policy_params = {}

        # Add required parameters first
        policy_params["environment"] = environment

        # Add optimization parameters
        for param in param_ranges:
            if isinstance(param, CategoricalHyperParameter):
                policy_params[param.name] = trial.suggest_categorical(
                    param.name, param.choices
                )
            elif isinstance(param, NumericalHyperParameter):
                if isinstance(param.low, float):
                    policy_params[param.name] = trial.suggest_float(
                        param.name, param.low, param.high
                    )
                else:
                    policy_params[param.name] = trial.suggest_int(
                        param.name, param.low, param.high
                    )

        # Create policy instance with suggested parameters
        policy = policy_class(**policy_params)

        # Evaluate policy and return objective value
        return evaluation_function(policy, trial)

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    # Get histories from the best trial-
    best_histories = study.best_trial.user_attrs["histories"]

    # Remove environment from best params as it was not optimized
    best_params = {
        k: v for k, v in study.best_trial.params.items() if k not in ["environment"]
    }

    return best_params, study.best_value, best_histories


def optimize_policy_parameters_for_multiple_environments(
    environment_policy_pairs: List[
        Tuple[Environment, Tuple[Type[Policy], List[HyperParameterFeatures]]]
    ],
    num_episodes: int,
    num_steps: int,
    n_particles: int,
    cache_dir_path: Path,
    parameter_to_optimize: str = "average_return",
    direction: str = "maximize",
    n_trials: int = 100,
    confidence_interval_level: float = 0.95,
    experiment_name: str = "POMDP_Parameter_Optimization",
    n_jobs: int = 1,
) -> Tuple[List[Tuple[dict, float, List[History]]], pd.DataFrame]:
    assert isinstance(environment_policy_pairs, List)
    assert all(
        isinstance(pair, tuple) and len(pair) == 2 for pair in environment_policy_pairs
    )
    assert all(isinstance(env, Environment) for env, _ in environment_policy_pairs)
    assert all(
        isinstance(policy_config, tuple) and len(policy_config) == 2
        for _, policy_config in environment_policy_pairs
    )
    assert all(
        isinstance(policy_class, type) and issubclass(policy_class, Policy)
        for _, (policy_class, _) in environment_policy_pairs
    )
    assert all(
        isinstance(param_ranges, list)
        and all(isinstance(param, HyperParameterFeatures) for param in param_ranges)
        for _, (_, param_ranges) in environment_policy_pairs
    )
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
    assert 0 <= confidence_interval_level <= 1
    assert direction in ["maximize", "minimize"]

    # Set up MLFlow tracking
    mlruns_path = cache_dir_path / "mlruns"
    mlruns_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = mlruns_path
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    results = []
    planner_statistics = []
    aggregated_data = []

    for pair_idx, (environment, (policy_class, param_ranges)) in enumerate(
        environment_policy_pairs
    ):
        # End any active run safely
        active_run = mlflow.active_run()
        if active_run is not None:
            mlflow.end_run()

        # Start a new MLFlow run for each environment-policy combination
        with mlflow.start_run(
            run_name=f"{environment.__class__.__name__}_{policy_class.__name__}_{pair_idx}"
        ):
            # Log common parameters
            common_params = {
                "environment_type": environment.__class__.__name__,
                "policy_type": policy_class.__name__,
                "num_episodes": num_episodes,
                "num_steps": num_steps,
                "n_particles": n_particles,
                "parameter_to_optimize": parameter_to_optimize,
                "direction": direction,
                "n_trials": n_trials,
                "confidence_interval_level": confidence_interval_level,
            }

            # Log environment-specific parameters
            env_params = {
                f"env_{key}": value
                for key, value in environment.__dict__.items()
                if isinstance(value, (int, float, str, bool))
            }

            # Log policy-specific parameter ranges
            policy_param_ranges = {}
            for param in param_ranges:
                if isinstance(param, CategoricalHyperParameter):
                    policy_param_ranges[f"param_range_{param.name}"] = (
                        f"choices: {param.choices}"
                    )
                elif isinstance(param, NumericalHyperParameter):
                    policy_param_ranges[f"param_range_{param.name}"] = (
                        f"{param.low}-{param.high}"
                    )

            # Combine all parameters
            all_params = {**common_params, **env_params, **policy_param_ranges}

            mlflow.log_params(all_params)

            # Run optimization
            best_params, best_value, histories = optimize_policy_parameters_with_optuna(
                environment=environment,
                policy_class=policy_class,
                param_ranges=param_ranges,
                num_episodes=num_episodes,
                num_steps=num_steps,
                n_particles=n_particles,
                cache_dir_path=cache_dir_path,
                parameter_to_optimize=parameter_to_optimize,
                direction=direction,
                n_trials=n_trials,
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs,
            )

            # Get statistics from the best trial
            initial_belief = get_initial_belief(
                pomdp=environment, n_particles=n_particles
            )
            _, statistics = simulation(
                environment=environment,
                policy=policy_class(environment=environment, **best_params),
                initial_belief=initial_belief,
                num_episodes=num_episodes,
                num_steps=num_steps,
                alpha=0.05,  # Fixed alpha for evaluation
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs,
            )

            # Log best parameters and value
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_value", best_value)

            # Save the full results as a JSON artifact using log_dict
            results_data = {
                "best_parameters": best_params,
                "best_value": best_value,
                "parameters": all_params,
                "param_ranges": [param._asdict() for param in param_ranges],
                "statistics": [metric._asdict() for metric in statistics],
            }
            mlflow.log_dict(results_data, f"optimization_results_pair_{pair_idx}.json")

            results.append((best_params, best_value, histories))
            planner_statistics.append(statistics)

            # Aggregate data for DataFrame
            run_data = {
                **all_params,
                **{f"best_{k}": v for k, v in best_params.items()},
                "best_value": best_value,
                **{metric.name: metric.value for metric in statistics},
            }
            aggregated_data.append(run_data)

    # Convert aggregated data to DataFrame
    df = pd.DataFrame(aggregated_data)

    # Log DataFrame as table artifact through MLFlow
    mlflow.log_table(df, "optimization_results.json")

    # Plot statistics comparison
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        plot_metrics_comparison(
            statistics=planner_statistics,
            environments=[env for env, (policy_class, _) in environment_policy_pairs],
            policies=[policy_class for _, (policy_class, _) in environment_policy_pairs],
            cache_dir_path=temp_dir_path,
        )

        # Log all generated plots as artifacts
        for plot_file in temp_dir_path.glob("*.png"):
            mlflow.log_artifact(str(plot_file), "statistics_plots")

    # End any active run safely before returning
    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.end_run()

    return results, df
