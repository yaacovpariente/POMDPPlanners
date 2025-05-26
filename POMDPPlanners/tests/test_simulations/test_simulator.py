import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import mlflow
import json
import os
import pandas as pd
from unittest.mock import patch

from POMDPPlanners.core.simulation import MetricValue
from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import DiscreteLightDarkPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.simulation import History, StepData, EnvironmentRunParams
from POMDPPlanners.simulations.simulations_deployment import LocalSimulationDeployment, DeploymentType


@pytest.fixture
def temp_cache_dir():
    # Create a temporary directory for MLFlow cache
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    # Ensure the directory exists and is empty
    if temp_path.exists():
        shutil.rmtree(temp_path)
    temp_path.mkdir(parents=True, exist_ok=True)
    yield temp_path
    # Cleanup
    try:
        if temp_path.exists():
            # Force close any open file handles
            import gc
            gc.collect()
            # Try to remove the directory
            shutil.rmtree(temp_path, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Failed to clean up temporary directory {temp_path}: {e}")


@pytest.fixture
def simulator(temp_cache_dir):
    return POMDPSimulator(
        cache_dir_path=temp_cache_dir,
        experiment_name="Test_Experiment",
        debug=True
    )


def test_simulator_initialization():
    """Test that the simulator initializes correctly."""
    # Test with default parameters
    simulator = POMDPSimulator()
    assert simulator.cache_dir_path is None
    assert simulator.experiment_name == "POMDP_Planning_Comparison"
    
    # Test with custom parameters
    cache_dir = Path("/tmp/test_cache")
    simulator = POMDPSimulator(
        cache_dir_path=cache_dir,
        experiment_name="Custom_Experiment",
        debug=True
    )
    assert simulator.cache_dir_path == cache_dir
    assert simulator.experiment_name == "Custom_Experiment"


def test_run_episode_returns_history(simulator):
    """Test that run_episode returns a valid History object."""
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=3)
    num_steps = 5

    # Execute
    history = simulator.run_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
    )

    # Assert
    assert isinstance(history, History)
    assert len(history.history) == num_steps


def test_run_episode_timing_statistics(simulator):
    """Test that run_episode records timing statistics correctly."""
    # Setup
    environment = DiscreteLightDarkPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=100)
    num_steps = 5

    # Execute
    history = simulator.run_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
    )

    # Assert timing statistics are positive and reasonable
    assert history.average_state_sampling_time >= 0
    assert history.average_action_time >= 0
    assert history.average_observation_time >= 0
    assert history.average_belief_update_time >= 0
    assert history.average_reward_time >= 0


def test_run_episode_valid_transitions(simulator):
    """Test that run_episode produces valid state transitions and observations."""
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=3)
    num_steps = 5

    # Execute
    history = simulator.run_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
    )

    # Assert valid state transitions and observations
    for step in history.history:
        assert step.state in ["tiger_left", "tiger_right"]
        assert step.action in ["listen", "open_left", "open_right"]
        assert step.next_state in ["tiger_left", "tiger_right"]
        assert step.observation in ["hear_left", "hear_right", "hear_nothing"]
        assert isinstance(step.reward, float)


def test_run_and_cache_episode_caching(simulator):
    """Test that run_and_cache_episode correctly caches and retrieves results."""
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=3)
    num_steps = 5
    episode_id = 1
    seed = 42
    simulation_deployment = LocalSimulationDeployment(n_jobs=1)

    # First run - should execute and cache
    history1 = simulator.run_and_cache_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
        episode_id=episode_id,
        seed=seed,
        simulation_deployment=simulation_deployment
    )

    # Second run - should load from cache
    history2 = simulator.run_and_cache_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
        episode_id=episode_id,
        seed=seed,
        simulation_deployment=simulation_deployment
    )

    # Assert histories are identical
    assert history1.history == history2.history
    assert history1.discount_factor == history2.discount_factor
    assert history1.average_state_sampling_time == history2.average_state_sampling_time
    assert history1.average_action_time == history2.average_action_time
    assert history1.average_observation_time == history2.average_observation_time
    assert history1.average_belief_update_time == history2.average_belief_update_time
    assert history1.average_reward_time == history2.average_reward_time
    assert history1.actual_num_steps == history2.actual_num_steps
    assert history1.reach_terminal_state == history2.reach_terminal_state


def test_simulate_multiple_environments_and_policies_parallel(simulator):
    """Test parallel simulation of multiple environments and policies."""
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=3)
    num_episodes = 2
    num_steps = 3

    # Execute
    results = simulator.simulate_multiple_environments_and_policies_parallel(
        environment_run_params=[
            EnvironmentRunParams(
                environment=environment,
                belief=initial_belief,
                policies=[policy],
                num_episodes=num_episodes,
                num_steps=num_steps
            )
        ],
        alpha=0.1,
        deployment_type=DeploymentType.LOCAL
    )

    # Assert
    assert isinstance(results, dict)
    assert environment.name in results
    assert policy.name in results[environment.name]
    assert len(results[environment.name][policy.name]) == num_episodes

    # Check each history
    for history in results[environment.name][policy.name]:
        assert isinstance(history, History)
        assert len(history.history) == num_steps
        assert history.actual_num_steps == num_steps
        assert isinstance(history.reach_terminal_state, bool)


def test_compare_multiple_environments_policies(simulator):
    """Test comparison of multiple environments and policies."""
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=3)
    num_episodes = 2
    num_steps = 3

    # Execute
    histories, merged_df = simulator.compare_multiple_environments_policies(
        environment_run_params=[
            EnvironmentRunParams(
                environment=environment,
                belief=initial_belief,
                policies=[policy],
                num_episodes=num_episodes,
                num_steps=num_steps
            )
        ],
        alpha=0.1,
        deployment_type=DeploymentType.LOCAL
    )

    # Assert
    assert isinstance(histories, dict)
    assert isinstance(merged_df, pd.DataFrame)
    
    # Check histories
    assert environment.name in histories
    assert policy.name in histories[environment.name]
    assert len(histories[environment.name][policy.name]) == num_episodes
    
    # Check DataFrame
    assert 'environment' in merged_df.columns
    assert 'policy' in merged_df.columns
    assert 'average_return' in merged_df.columns
    assert 'depth' in merged_df.columns
    assert 'exploration_constant' in merged_df.columns
    assert 'n_simulations' in merged_df.columns
    
    # Check values
    row = merged_df.iloc[0]
    assert row['environment'] == environment.name
    assert row['policy'] == policy.name
    assert row['depth'] == 3
    assert row['exploration_constant'] == 1.0
    assert row['n_simulations'] == 2


def test_parallel_execution_maintains_statistical_properties(simulator):
    """Test that parallel execution maintains statistical properties."""
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=3)
    num_episodes = 4
    num_steps = 3

    # Run with different numbers of jobs
    results_1job = simulator.simulate_multiple_environments_and_policies_parallel(
        environment_run_params=[
            EnvironmentRunParams(
                environment=environment,
                belief=initial_belief,
                policies=[policy],
                num_episodes=num_episodes,
                num_steps=num_steps
            )
        ],
        alpha=0.1,
        n_jobs=1,
        deployment_type=DeploymentType.LOCAL
    )

    results_2jobs = simulator.simulate_multiple_environments_and_policies_parallel(
        environment_run_params=[
            EnvironmentRunParams(
                environment=environment,
                belief=initial_belief,
                policies=[policy],
                num_episodes=num_episodes,
                num_steps=num_steps
            )
        ],
        alpha=0.1,
        n_jobs=2,
        deployment_type=DeploymentType.LOCAL
    )

    # Compare aggregate statistics
    def compute_statistics(results):
        stats = {
            'total_episodes': 0,
            'total_steps': 0,
            'total_listens': 0,
            'total_opens': 0,
            'total_reward': 0.0,
            'rewards_per_episode': [],
            'listens_per_episode': [],
            'opens_per_episode': []
        }
        
        for env_name, env_results in results.items():
            for policy_name, histories in env_results.items():
                stats['total_episodes'] += len(histories)
                for history in histories:
                    episode_reward = 0.0
                    episode_listens = 0
                    episode_opens = 0
                    stats['total_steps'] += len(history.history)
                    for step in history.history:
                        if step.action == 'listen':
                            stats['total_listens'] += 1
                            episode_listens += 1
                        elif step.action in ['open_left', 'open_right']:
                            stats['total_opens'] += 1
                            episode_opens += 1
                        episode_reward += step.reward
                        stats['total_reward'] += step.reward
                    stats['rewards_per_episode'].append(episode_reward)
                    stats['listens_per_episode'].append(episode_listens)
                    stats['opens_per_episode'].append(episode_opens)
        
        return stats

    stats_1job = compute_statistics(results_1job)
    stats_2jobs = compute_statistics(results_2jobs)

    # Assert structural properties that should be identical
    assert stats_1job['total_episodes'] == stats_2jobs['total_episodes'] == num_episodes
    assert stats_1job['total_steps'] == stats_2jobs['total_steps'] == num_episodes * num_steps
    assert stats_1job['total_listens'] + stats_1job['total_opens'] == stats_1job['total_steps']
    assert stats_2jobs['total_listens'] + stats_2jobs['total_opens'] == stats_2jobs['total_steps']
    
    # Check that the distribution of actions is reasonable
    for stats in [stats_1job, stats_2jobs]:
        assert all(listens + opens > 0 for listens, opens 
                  in zip(stats['listens_per_episode'], stats['opens_per_episode']))
        assert all(listens + opens == num_steps for listens, opens 
                  in zip(stats['listens_per_episode'], stats['opens_per_episode']))
        
        min_possible_reward = -100.0 * num_steps
        max_possible_reward = 10.0 * num_steps
        
        for i, r in enumerate(stats['rewards_per_episode']):
            assert min_possible_reward <= r <= max_possible_reward, \
                f"Episode {i} reward {r} outside bounds [{min_possible_reward}, {max_possible_reward}]"
    
    # The average reward per episode should be roughly similar
    avg_reward_1job = stats_1job['total_reward'] / stats_1job['total_episodes']
    avg_reward_2jobs = stats_2jobs['total_reward'] / stats_2jobs['total_episodes']
    reward_difference = abs(avg_reward_1job - avg_reward_2jobs)
    max_reward_difference = 100.0  # Increased tolerance due to high variance in rewards
    assert reward_difference < max_reward_difference, \
        f"Average reward difference {reward_difference} exceeds maximum allowed {max_reward_difference}"


def test_visualization_creation(simulator):
    """Test that visualizations are created correctly."""
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=3)
    num_episodes = 2
    num_steps = 3

    # Execute comparison with visualizations
    histories, _ = simulator.compare_multiple_environments_policies(
        environment_run_params=[
            EnvironmentRunParams(
                environment=environment,
                belief=initial_belief,
                policies=[policy],
                num_episodes=num_episodes,
                num_steps=num_steps
            )
        ],
        alpha=0.1,
        cache_visualizations=True,
        deployment_type=DeploymentType.LOCAL
    )

    # Check that visualization directories were created
    results_dir = simulator.cache_dir_path / "results"
    env_dir = results_dir / environment.name
    policy_dir = env_dir / policy.name
    
    assert results_dir.exists()
    assert env_dir.exists()
    assert policy_dir.exists()
    
    # Check for plot files
    plots_dir = policy_dir / "plots"
    assert plots_dir.exists()
    assert (plots_dir / "discounted_returns_histogram.png").exists()
    
    # Check for comparison plot
    assert (env_dir / "policy_comparison_histogram.png").exists()
    
    # Check for visualization directory
    viz_dir = policy_dir / "visualizations"
    assert viz_dir.exists() 