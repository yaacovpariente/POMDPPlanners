from typing import List, Type, Tuple, Dict
from typing import NamedTuple
import copy
from time import time
from pathlib import Path
import mlflow

from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
import pandas as pd

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.simulation import (
    History,
    StepData,
    EnvironmentRunParams
)
from POMDPPlanners.simulations.simulation_statistics import compute_statistics_environments_policies_comparison
from POMDPPlanners.simulations.simulations_deployment import LocalSimulationDeployment, RemoteRaySimulationDeployment, DeploymentType, SimulationDeployment
from POMDPPlanners.utils.visualization import plot_discounted_returns_histogram, plot_discounted_returns_histogram_multiple_policies
from POMDPPlanners.utils.logger import logger


def run_episode(
    environment: Environment, policy: Policy, initial_belief: Belief, num_steps: int,
) -> History:
    """Run a single episode without caching.
    
    Args:
        environment: The environment to run the episode in
        policy: The policy to use
        initial_belief: Initial belief state
        num_steps: Number of steps to run
        
    Returns:
        History object containing episode data
    """
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
        if environment.is_terminal(state=state):
            reach_terminal_state = True
            history.append(
                StepData(
                    state=state,
                    action=None,
                    next_state=None,
                    observation=None,
                    reward=None,
                    belief=belief,
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
                belief=belief,
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


def validate_episode_inputs(
    environment: Environment,
    policy: Policy,
    initial_belief: Belief,
    num_steps: int,
    cache_dir_path: Path = None,
    episode_id: int = None,
    seed: int = None,
) -> None:
    """Validate inputs for episode simulation.
    
    Args:
        environment: The environment to run the episode in
        policy: The policy to use
        initial_belief: Initial belief state
        num_steps: Number of steps to run
        cache_dir_path: Directory to store cache files (optional)
        episode_id: Unique identifier for this episode
        seed: Random seed for deterministic execution
        
    Raises:
        AssertionError: If any of the inputs are invalid
    """
    assert isinstance(environment, Environment), "environment must be an Environment instance"
    assert isinstance(policy, Policy), "policy must be a Policy instance"
    assert isinstance(initial_belief, Belief), "initial_belief must be a Belief instance"
    assert isinstance(num_steps, int) and num_steps > 0, "num_steps must be a positive integer"
    assert isinstance(episode_id, int), "episode_id must be an integer"
    assert isinstance(seed, int), "seed must be an integer"
    if cache_dir_path is not None:
        assert isinstance(cache_dir_path, Path), "cache_dir_path must be a Path instance"

def run_and_cache_episode(
    environment: Environment,
    policy: Policy,
    initial_belief: Belief,
    num_steps: int,
    episode_id: int,
    seed: int,
    simulation_deployment: SimulationDeployment,
    cache_dir_path: Path = None,
) -> History:
    """Run an episode with optional caching support and deterministic seed.
    
    Args:
        environment: The environment to run the episode in
        policy: The policy to use
        initial_belief: Initial belief state
        num_steps: Number of steps to run
        episode_id: Unique identifier for this episode
        seed: Random seed for deterministic execution
        cache_dir_path: Directory to store cache files (optional)
        
    Returns:
        History object containing episode data
    """
    # Input validation
    validate_episode_inputs(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
        cache_dir_path=cache_dir_path,
        episode_id=episode_id,
        seed=seed
    )

    if cache_dir_path is not None:
        general_config = {'episode_id': episode_id, 'num_steps': num_steps, 'seed': seed}

        # Try to load from cache
        try:
            cached_history = simulation_deployment.load_episode_simulation_results(
                environment=environment,
                policy=policy,
                initial_belief=initial_belief,
                cache_dir_path=cache_dir_path,
                general_config=general_config
            )
            if len(cached_history) > 0:
                logger.info(f"Loaded episode {episode_id} from cache for environment {environment.name} and policy {policy.name}")
                return cached_history[0]
        except Exception as e:
            logger.warning(f"Failed to load episode {episode_id} from cache for environment {environment.name} and policy {policy.name}: {str(e)}")

    # Set random seed
    state = np.random.get_state()
    np.random.seed(seed)

    try:
        # Run the episode
        result = run_episode(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            num_steps=num_steps,
        )

        # Cache the result if caching is enabled
        if cache_dir_path is not None:
            try:
                simulation_deployment.save_episode_simulation_results(
                    environment=environment,
                    policy=policy,
                    initial_belief=initial_belief,
                    results=[result],
                    cache_dir_path=cache_dir_path,
                    general_config=general_config
                )
                logger.debug(f"Cached episode {episode_id}")
            except Exception as e:
                logger.warning(f"Failed to cache episode {episode_id}: {str(e)}")

        return result
    finally:
        # Restore random state
        np.random.set_state(state)

def validate_parallel_simulation_inputs(
    environment_run_params: List[EnvironmentRunParams],
    alpha: float,
    confidence_interval_level: float,
    n_jobs: int
) -> None:
    """Validate all input parameters for parallel simulation.
    
    Args:
        environment_run_params: List of EnvironmentRunParams containing environment, belief, policies and run parameters
        alpha: Alpha value for statistics computation
        confidence_interval_level: Confidence level for statistics
        n_jobs: Number of parallel jobs for simulation
        
    Raises:
        AssertionError: If any of the inputs are invalid
    """
    assert isinstance(environment_run_params, list), "environment_run_params must be a list"
    assert len(environment_run_params) > 0, "environment_run_params cannot be empty"
    
    for params in environment_run_params:
        assert isinstance(params, EnvironmentRunParams), f"Expected EnvironmentRunParams, got {type(params)}"
        assert isinstance(params.environment, Environment), f"Expected Environment, got {type(params.environment)}"
        assert isinstance(params.belief, Belief), f"Expected Belief, got {type(params.belief)}"
        assert isinstance(params.policies, list), f"Expected list of policies, got {type(params.policies)}"
        assert len(params.policies) > 0, "Policy list cannot be empty"
        for policy in params.policies:
            assert isinstance(policy, Policy), f"Expected Policy, got {type(policy)}"
        assert isinstance(params.num_episodes, int) and params.num_episodes > 0, "num_episodes must be a positive integer"
        assert isinstance(params.num_steps, int) and params.num_steps > 0, "num_steps must be a positive integer"

    # Verify unique environment names
    env_names = [params.environment.name for params in environment_run_params]
    assert len(env_names) == len(set(env_names)), "All environments must have unique names"

    # Verify unique policy names across all environments
    all_policies = [policy for params in environment_run_params for policy in params.policies]
    policy_names = [policy.name for policy in all_policies]
    assert len(policy_names) == len(set(policy_names)), "All policies must have unique names"

    assert isinstance(alpha, float), "alpha must be a float"
    assert isinstance(confidence_interval_level, float), "confidence_interval_level must be a float"
    assert 0 <= confidence_interval_level <= 1, "confidence_interval_level must be between 0 and 1"
    assert isinstance(n_jobs, int) and n_jobs > 0, "n_jobs must be a positive integer"

def create_simulation_tasks(
    environment_run_params: List[EnvironmentRunParams],
    cache_dir_path: Path,
    simulation_deployment: SimulationDeployment
) -> List[Dict]:
    """Create list of simulation tasks with deterministic ordering.
    
    Args:
        environment_run_params: List of EnvironmentRunParams containing environment, belief, policies and run parameters
        cache_dir_path: Path to store results
        simulation_deployment: The deployment strategy to use
        
    Returns:
        List of dictionaries containing simulation task parameters
    """
    simulation_tasks = []
    total_tasks = 0
    
    for params in environment_run_params:
        for policy in params.policies:
            for episode_id in range(params.num_episodes):
                seed = hash(f"{params.environment.name}_{policy.name}_{episode_id}") % (2**32)
                simulation_tasks.append({
                    'environment': params.environment,
                    'policy': policy,
                    'initial_belief': params.belief,
                    'num_steps': params.num_steps,
                    'episode_id': episode_id,
                    'seed': seed,
                    'cache_dir_path': cache_dir_path,
                    'simulation_deployment': simulation_deployment
                })
                total_tasks += 1

    simulation_tasks.sort(key=lambda x: x['seed'])
    logger.info(f"Created {total_tasks} simulation tasks across {len(set(params.environment.name for params in environment_run_params))} "
                f"environments and {len(set(p.name for params in environment_run_params for p in params.policies))} policies")
    
    return simulation_tasks

def execute_parallel_simulations(
    simulation_tasks: List[Dict],
    n_jobs: int,
    simulation_deployment: SimulationDeployment
) -> List[History]:
    """Execute simulation tasks in parallel.
    
    Args:
        simulation_tasks: List of dictionaries containing simulation task parameters
        n_jobs: Number of parallel jobs for simulation
        simulation_deployment: The deployment strategy to use
        
    Returns:
        List of History objects containing simulation results
    """
    logger.info(f"Starting parallel execution using {n_jobs} jobs")
    start_time = time()
    
    histories_list = Parallel(n_jobs=n_jobs)(
        delayed(run_and_cache_episode)(
            simulation_deployment=simulation_deployment,
            **task
        ) for task in tqdm(
            simulation_tasks,
            total=len(simulation_tasks),
            desc="Running parallel simulations",
            unit="episode"
        )
    )
    
    end_time = time()
    logger.info(f"Parallel execution completed in {end_time - start_time:.2f} seconds")
    
    return histories_list

def organize_simulation_results(
    histories_list: List[History],
    environment_belief_policy_tuples: List[Tuple[Environment, Belief, List[Policy]]],
    num_episodes: int
) -> Dict[str, Dict[str, List[History]]]:
    """Organize simulation results by environment and policy.
    
    Args:
        histories_list: List of History objects containing simulation results
        environment_belief_policy_tuples: List of tuples containing (environment, belief, policies)
        num_episodes: Number of episodes per policy
        
    Returns:
        Dictionary mapping environment names to dictionaries of policy histories
    """
    logger.info("Organizing results by environment and policy")
    results = {}
    current_idx = 0
    
    for environment, _, policies in environment_belief_policy_tuples:
        env_results = {}
        for policy in policies:
            policy_histories = histories_list[current_idx:current_idx + num_episodes]
            env_results[policy.name] = policy_histories
            current_idx += num_episodes
            logger.debug(f"Processed {len(policy_histories)} histories for {environment.name} with {policy.name}")
        results[environment.name] = env_results
    
    return results

def simulate_multiple_environments_and_policies_parallel(
    environment_run_params: List[EnvironmentRunParams],
    alpha: float,
    confidence_interval_level: float = 0.95,
    n_jobs: int = 1,
    cache_dir_path: Path = None,
    deployment_type: DeploymentType = DeploymentType.LOCAL
) -> Dict[str, Dict[str, List[History]]]:
    """Simulate multiple policies on multiple environments in parallel at the episode level.

    Args:
        environment_run_params: List of EnvironmentRunParams containing environment, belief, policies and run parameters
        alpha: Alpha value for statistics computation
        confidence_interval_level: Confidence level for statistics
        n_jobs: Number of parallel jobs for simulation
        cache_dir_path: Path to store results (if None, uses current directory)
        deployment_type: Type of deployment to use for simulations

    Returns:
        Dictionary mapping environment names to dictionaries of policy histories
    """
    # Validate inputs
    validate_parallel_simulation_inputs(
        environment_run_params=environment_run_params,
        alpha=alpha,
        confidence_interval_level=confidence_interval_level,
        n_jobs=n_jobs
    )

    # Initialize deployment based on type
    if deployment_type == DeploymentType.LOCAL:
        simulation_deployment = LocalSimulationDeployment(n_jobs=n_jobs)
    elif deployment_type == DeploymentType.REMOTE_RAY:
        simulation_deployment = RemoteRaySimulationDeployment(num_cpus=n_jobs)
    else:
        raise ValueError(f"Unsupported deployment type: {deployment_type}")

    # Create simulation tasks
    simulation_tasks = create_simulation_tasks(
        environment_run_params=environment_run_params,
        cache_dir_path=cache_dir_path,
        simulation_deployment=simulation_deployment
    )

    # Execute simulations
    try:
        histories_list = simulation_deployment.run_multiple_episodes(
            func=run_and_cache_episode,
            episode_configs=simulation_tasks
        )
    except Exception as e:
        logger.error(f"Error running simulations: {str(e)}")
        raise e
    finally:
        # Close Redis connection if using remote Ray deployment
        if deployment_type == DeploymentType.REMOTE_RAY:
            del simulation_deployment
        
    # Organize and return results
    return organize_simulation_results(
        histories_list=histories_list,
        environment_belief_policy_tuples=[(params.environment, params.belief, params.policies) for params in environment_run_params],
        num_episodes=environment_run_params[0].num_episodes
    )

def validate_comparison_inputs(
    environment_run_params: List[EnvironmentRunParams],
    confidence_interval_level: float,
    cache_dir_path: Path = None,
    experiment_name: str = None
) -> None:
    """Validate all input parameters for the comparison function.
    
    Args:
        environment_run_params: List of EnvironmentRunParams containing environment, belief, policies and run parameters
        confidence_interval_level: Confidence level for statistics
        cache_dir_path: Path to store results (if None, uses current directory)
        experiment_name: Name of the MLFlow experiment
        
    Raises:
        AssertionError: If any of the inputs are invalid
    """
    assert isinstance(environment_run_params, list), "environment_run_params must be a list"
    assert len(environment_run_params) > 0, "environment_run_params cannot be empty"
    
    for params in environment_run_params:
        assert isinstance(params, EnvironmentRunParams), f"Expected EnvironmentRunParams, got {type(params)}"
        assert isinstance(params.environment, Environment), f"Expected Environment, got {type(params.environment)}"
        assert isinstance(params.belief, Belief), f"Expected Belief, got {type(params.belief)}"
        assert isinstance(params.policies, list), f"Expected list of policies, got {type(params.policies)}"
        assert len(params.policies) > 0, "Policy list cannot be empty"
        for policy in params.policies:
            assert isinstance(policy, Policy), f"Expected Policy, got {type(policy)}"
        assert isinstance(params.num_episodes, int) and params.num_episodes > 0, "num_episodes must be a positive integer"
        assert isinstance(params.num_steps, int) and params.num_steps > 0, "num_steps must be a positive integer"

    # Verify unique environment names
    env_names = [params.environment.name for params in environment_run_params]
    assert len(env_names) == len(set(env_names)), "All environments must have unique names"

    # Verify unique policy names across all environments
    all_policies = [policy for params in environment_run_params for policy in params.policies]
    policy_names = [policy.name for policy in all_policies]
    assert len(policy_names) == len(set(policy_names)), "All policies must have unique names"

    assert isinstance(confidence_interval_level, float)
    assert 0 <= confidence_interval_level <= 1, "confidence_interval_level must be between 0 and 1"
    
    if cache_dir_path is not None:
        assert isinstance(cache_dir_path, Path), "cache_dir_path must be a Path instance"
    if experiment_name is not None:
        assert isinstance(experiment_name, str), "experiment_name must be a string"

def setup_mlflow_tracking(cache_dir_path: Path, experiment_name: str) -> Path:
    """Configure MLFlow tracking and return tracking URI."""
    if cache_dir_path is None:
        cache_dir_path = Path.cwd()
    mlruns_path = cache_dir_path / "mlruns"
    mlruns_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = mlruns_path
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri

def create_environment_visualizations(
    env_name: str,
    environment: Environment,
    policy_histories_dict: Dict[str, List[History]],
    policies: List[Policy],
    results_dir: Path,
    cache_visualizations: bool
) -> None:
    """Create and save visualizations for a specific environment."""
    env_dir = results_dir / env_name
    env_dir.mkdir(exist_ok=True)
    
    for policy_name, policy_histories in policy_histories_dict.items():
        policy = next(p for p in policies if p.name == policy_name)
        create_policy_visualizations(
            policy=policy,
            environment=environment,
            policy_histories=policy_histories,
            env_dir=env_dir,
            cache_visualizations=cache_visualizations
        )
    
    # Create comparison plot
    comparison_plot_path = env_dir / "policy_comparison_histogram.png"
    plot_discounted_returns_histogram_multiple_policies(
        histories=policy_histories_dict,
        policies=policies,
        environment=environment,
        cache_path=comparison_plot_path
    )

def create_policy_visualizations(
    policy: Policy,
    environment: Environment,
    policy_histories: List[History],
    env_dir: Path,
    cache_visualizations: bool
) -> None:
    """Create and save visualizations for a specific policy."""
    policy_dir = env_dir / policy.name
    policy_dir.mkdir(exist_ok=True)
    
    # Create plots
    plots_dir = policy_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / "discounted_returns_histogram.png"
    plot_discounted_returns_histogram(
        histories=policy_histories,
        policy=policy,
        environment=environment,
        cache_path=plot_path
    )
    
    if cache_visualizations:
        cache_episode_visualizations(
            environment=environment,
            policy_histories=policy_histories,
            policy_dir=policy_dir
        )

def cache_episode_visualizations(
    environment: Environment,
    policy_histories: List[History],
    policy_dir: Path
) -> None:
    """Cache visualizations for individual episodes."""
    viz_dir = policy_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
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

def create_policy_configurations_df(
    environment_belief_policy_tuples: List[Tuple[Environment, Belief, List[Policy]]]
) -> pd.DataFrame:
    """Create a DataFrame containing policy configurations for all environment-policy pairs.
    
    Args:
        environment_belief_policy_tuples: List of tuples containing (environment, belief, policies)
        
    Returns:
        DataFrame where each row represents an environment-policy pair and columns are configuration parameters
    """
    policy_configs = []
    for env, belief, policies in environment_belief_policy_tuples:
        for policy in policies:
            # Get policy parameters
            policy_params = {
                key: value 
                for key, value in policy.__dict__.items()
                if isinstance(value, (int, float, str, bool))
            }
            
            # Create row with environment and policy info
            row_data = {
                'environment': env.name,
                'policy': policy.name,
                'policy_type': policy.__class__.__name__,
                **policy_params  # Add all policy parameters
            }
            policy_configs.append(row_data)
    
    return pd.DataFrame(policy_configs)



def compare_multiple_environments_policies(
    environment_run_params: List[EnvironmentRunParams],
    alpha: float,
    confidence_interval_level: float = 0.95,
    n_jobs: int = 1,
    cache_dir_path: Path = None,
    experiment_name: str = "POMDP_Planning_Comparison",
    cache_visualizations: bool = True,
    deployment_type: DeploymentType = DeploymentType.LOCAL
) -> Tuple[Dict[str, Dict[str, List[History]]], pd.DataFrame]:
    """Compare multiple policies on multiple environments and cache results in MLFlow.
    
    Args:
        environment_run_params: List of EnvironmentRunParams containing environment, belief, policies and run parameters
        alpha: Alpha value for statistics computation
        confidence_interval_level: Confidence level for statistics
        n_jobs: Number of parallel jobs for simulation
        cache_dir_path: Path to store results (if None, uses current directory)
        experiment_name: Name of the MLFlow experiment
        cache_visualizations: Whether to cache visualizations
        deployment_type: Type of deployment to use for simulations

    Returns:
        Tuple containing:
        - Dictionary mapping environment names to dictionaries of policy histories
        - DataFrame with statistics and policy configurations
    """
    # Validate inputs
    validate_comparison_inputs(
        environment_run_params=environment_run_params,
        confidence_interval_level=confidence_interval_level,
        cache_dir_path=cache_dir_path,
        experiment_name=experiment_name
    )

    # Setup MLFlow
    tracking_uri = setup_mlflow_tracking(cache_dir_path, experiment_name)
    logger.info(f"MLFlow tracking set up at {tracking_uri}")

    # Run main comparison
    with mlflow.start_run(run_name="environment_policy_comparison"):
        # Run simulations
        histories = simulate_multiple_environments_and_policies_parallel(
            environment_run_params=environment_run_params,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
            cache_dir_path=cache_dir_path,
            deployment_type=deployment_type
        )
        
        # Compute statistics
        statistics_df = compute_statistics_environments_policies_comparison(
            histories=histories,
            environments=[params.environment for params in environment_run_params],
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
        )

        # Create and merge policy configurations
        policy_configs_df = create_policy_configurations_df([(params.environment, params.belief, params.policies) for params in environment_run_params])
        merged_df = pd.merge(
            statistics_df,
            policy_configs_df,
            on=['environment', 'policy']
        )

        # Log statistics and configurations
        mlflow.log_table(merged_df, "statistics/comparison_results.json")
        mlflow.log_table(policy_configs_df, "statistics/policy_configurations.json")

        # Create results directory and visualizations
        results_dir = cache_dir_path / "results"
        results_dir.mkdir(exist_ok=True)
        
        for env_name, policy_histories_dict in histories.items():
            environment = next(params.environment for params in environment_run_params if params.environment.name == env_name)
            policies = [p for params in environment_run_params for p in params.policies if p.name in policy_histories_dict]
            
            create_environment_visualizations(
                env_name=env_name,
                environment=environment,
                policy_histories_dict=policy_histories_dict,
                policies=policies,
                results_dir=results_dir,
                cache_visualizations=cache_visualizations
            )
        
        # Log all artifacts
        mlflow.log_artifact(str(results_dir), ".")

    return histories, merged_df
