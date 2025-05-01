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
            history.append(
                StepData(
                    state=state,
                    action=None,
                    next_state=None,
                    observation=None,
                    reward=None,
                )
            )

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
    environment_belief_policy_tuples: List[Tuple[Environment, Belief, List[Policy]]],
    num_episodes: int,
    num_steps: int,
    alpha: float,
    confidence_interval_level: float = 0.95,
    n_jobs: int = 1,
) -> Dict[str, Dict[str, List[History]]]:
    """
    Simulate multiple policies on multiple environments.

    Args:
        environment_belief_policy_tuples: List of tuples containing (environment, initial_belief, policies)
        num_episodes: Number of episodes to run per policy
        num_steps: Number of steps per episode
        alpha: Alpha value for statistics computation
        confidence_interval_level: Confidence level for statistics
        n_jobs: Number of parallel jobs for simulation

    Returns:
        Dictionary mapping environment names to dictionaries of policy histories
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

    results = {}
    for environment, initial_belief, policies in environment_belief_policy_tuples:
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
        cache_visualizations: Whether to cache visualizations
        
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
        histories = simulate_multiple_environments_and_policies(
            environment_belief_policy_tuples=environment_belief_policy_tuples,
            num_episodes=num_episodes,
            num_steps=num_steps,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
        )
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
            policies_for_plot = [policy for _, _, policies in environment_belief_policy_tuples 
                               for policy in policies if policy.name in policy_histories_dict]
            plot_discounted_returns_histogram_multiple_policies(
                histories=policy_histories_dict,
                policies=policies_for_plot,
                environment=environment,
                cache_path=comparison_plot_path
            )
        
        # Log all generated plots and visualizations as artifacts
        mlflow.log_artifact(str(results_dir), ".")

    return histories, statistics_df