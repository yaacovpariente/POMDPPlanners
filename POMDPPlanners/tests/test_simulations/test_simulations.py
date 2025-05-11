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
from POMDPPlanners.simulations.simulations import (
    run_episode,
    run_and_cache_episode,
    simulate_multiple_environments_and_policies_parallel,
    create_policy_configurations_df,
    compare_multiple_environments_policies,
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import DiscreteLightDarkPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.simulations.simulations_deployment import LocalSimulationDeployment, DeploymentType


def test_run_episode_returns_history():
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
    history = run_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
    )

    # Assert
    assert isinstance(history, History)
    assert len(history.history) == num_steps


def test_run_episode_timing_statistics():
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
    history = run_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
    )

    # Assert timing statistics are positive and reasonable
    assert (
        history.average_state_sampling_time >= 0
    )  # Can be 0 if state transitions are deterministic
    # assert history.average_action_time > 0
    assert (
        history.average_observation_time >= 0
    )  # Can be 0 if observations are deterministic
    assert history.average_belief_update_time > 0
    assert history.average_reward_time >= 0  # Can be 0 if rewards are deterministic


def test_run_episode_valid_transitions():
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
    history = run_episode(
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


def test_run_episode_reward_calculation():
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
    history = run_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
    )

    # Assert reward values are correct based on TigerPOMDP rules
    for step in history.history:
        if step.action == "listen":
            assert step.reward == -1.0
        elif step.action == "open_left":
            assert step.reward in [-100.0, 10.0]
        elif step.action == "open_right":
            assert step.reward in [-100.0, 10.0]


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
def simulation_deployment():
    return LocalSimulationDeployment(n_jobs=1)


def test_run_and_cache_episode_returns_history(temp_cache_dir, simulation_deployment):
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

    # Execute
    history = run_and_cache_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
        cache_dir_path=temp_cache_dir,
        episode_id=episode_id,
        seed=seed,
        simulation_deployment=simulation_deployment
    )

    # Assert
    assert isinstance(history, History)
    assert len(history.history) == num_steps


def test_run_and_cache_episode_caching(temp_cache_dir, simulation_deployment):
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

    # First run - should execute and cache
    history1 = run_and_cache_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
        cache_dir_path=temp_cache_dir,
        episode_id=episode_id,
        seed=seed,
        simulation_deployment=simulation_deployment
    )

    # Second run - should load from cache
    history2 = run_and_cache_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
        cache_dir_path=temp_cache_dir,
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


def test_run_and_cache_episode_different_episode_ids(temp_cache_dir, simulation_deployment):
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

    # Run two different episodes
    history1 = run_and_cache_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
        cache_dir_path=temp_cache_dir,
        episode_id=1,
        seed=42,
        simulation_deployment=simulation_deployment
    )

    history2 = run_and_cache_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
        cache_dir_path=temp_cache_dir,
        episode_id=2,
        seed=43,
        simulation_deployment=simulation_deployment
    )

    # Assert histories are different (due to different seeds)
    assert history1.history != history2.history


def test_run_and_cache_episode_parameter_validation(simulation_deployment):
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
    temp_dir = Path("/tmp/test_cache")

    # Test invalid parameters
    with pytest.raises(AssertionError):
        run_and_cache_episode(
            environment="not_an_environment",
            policy=policy,
            initial_belief=initial_belief,
            num_steps=5,
            cache_dir_path=temp_dir,
            episode_id=1,
            seed=42,
            simulation_deployment=simulation_deployment
        )

    with pytest.raises(AssertionError):
        run_and_cache_episode(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            num_steps=5,
            cache_dir_path=temp_dir,
            episode_id=None,  # Invalid episode_id
            seed=42,
            simulation_deployment=simulation_deployment
        )

    with pytest.raises(AssertionError):
        run_and_cache_episode(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            num_steps=0,  # Invalid num_steps
            cache_dir_path=temp_dir,
            episode_id=1,
            seed=42,
            simulation_deployment=simulation_deployment
        )


def test_simulate_multiple_environments_and_policies_parallel_parameter_validation():
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

    # Test invalid parameters
    with pytest.raises(AssertionError):
        simulate_multiple_environments_and_policies_parallel(
            environment_belief_policy_tuples=[("not_an_environment", initial_belief, [policy])],
            num_episodes=10,
            num_steps=5,
            alpha=0.1,
        )

    with pytest.raises(AssertionError):
        simulate_multiple_environments_and_policies_parallel(
            environment_belief_policy_tuples=[],  # Empty list
            num_episodes=10,
            num_steps=5,
            alpha=0.1,
        )

    with pytest.raises(AssertionError):
        simulate_multiple_environments_and_policies_parallel(
            environment_belief_policy_tuples=[(environment, initial_belief, [policy])],
            num_episodes=0,  # Invalid num_episodes
            num_steps=5,
            alpha=0.1,
        )

    with pytest.raises(AssertionError):
        simulate_multiple_environments_and_policies_parallel(
            environment_belief_policy_tuples=[(environment, initial_belief, [policy])],
            num_episodes=10,
            num_steps=5,
            alpha=0.1,
            confidence_interval_level=1.5,  # Invalid confidence interval
        )


def test_simulate_multiple_environments_and_policies_parallel_single_environment_policy(temp_cache_dir):
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
    results = simulate_multiple_environments_and_policies_parallel(
        environment_belief_policy_tuples=[(environment, initial_belief, [policy])],
        num_episodes=num_episodes,
        num_steps=num_steps,
        alpha=0.1,
        cache_dir_path=temp_cache_dir,
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


def test_simulate_multiple_environments_and_policies_parallel_multiple_environments_policies(temp_cache_dir):
    # Setup
    environment1 = TigerPOMDP(discount_factor=0.95, name="TigerPOMDP_095")
    environment2 = TigerPOMDP(discount_factor=0.99, name="TigerPOMDP_099")
    policy1 = POMCP(
        environment=environment1,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy1",
        n_simulations=2
    )
    policy2 = POMCP(
        environment=environment2,
        discount_factor=0.99,
        depth=4,
        exploration_constant=1.5,
        name="TestPolicy2",
        n_simulations=2
    )
    initial_belief1 = get_initial_belief(environment1, n_particles=3)
    initial_belief2 = get_initial_belief(environment2, n_particles=3)
    num_episodes = 2
    num_steps = 3

    # Execute
    results = simulate_multiple_environments_and_policies_parallel(
        environment_belief_policy_tuples=[
            (environment1, initial_belief1, [policy1]),
            (environment2, initial_belief2, [policy2])
        ],
        num_episodes=num_episodes,
        num_steps=num_steps,
        alpha=0.1,
        cache_dir_path=temp_cache_dir,
        deployment_type=DeploymentType.LOCAL
    )

    # Assert
    assert isinstance(results, dict)
    assert len(results) == 2  # Two environments
    assert environment1.name in results
    assert environment2.name in results

    # Check first environment
    assert policy1.name in results[environment1.name]
    assert len(results[environment1.name][policy1.name]) == num_episodes
    for history in results[environment1.name][policy1.name]:
        assert isinstance(history, History)
        assert len(history.history) == num_steps

    # Check second environment
    assert policy2.name in results[environment2.name]
    assert len(results[environment2.name][policy2.name]) == num_episodes
    for history in results[environment2.name][policy2.name]:
        assert isinstance(history, History)
        assert len(history.history) == num_steps


def test_simulate_multiple_environments_and_policies_parallel_caching(temp_cache_dir):
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

    # First run - should execute and cache
    results1 = simulate_multiple_environments_and_policies_parallel(
        environment_belief_policy_tuples=[(environment, initial_belief, [policy])],
        num_episodes=num_episodes,
        num_steps=num_steps,
        alpha=0.1,
        cache_dir_path=temp_cache_dir,
        deployment_type=DeploymentType.LOCAL
    )

    # Second run - should load from cache
    results2 = simulate_multiple_environments_and_policies_parallel(
        environment_belief_policy_tuples=[(environment, initial_belief, [policy])],
        num_episodes=num_episodes,
        num_steps=num_steps,
        alpha=0.1,
        cache_dir_path=temp_cache_dir,
        deployment_type=DeploymentType.LOCAL
    )

    # Assert results are identical
    assert results1 == results2


def test_parallel_execution_maintains_statistical_properties():
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
    results_1job = simulate_multiple_environments_and_policies_parallel(
        environment_belief_policy_tuples=[(environment, initial_belief, [policy])],
        num_episodes=num_episodes,
        num_steps=num_steps,
        alpha=0.1,
        n_jobs=1,
        deployment_type=DeploymentType.LOCAL
    )

    results_2jobs = simulate_multiple_environments_and_policies_parallel(
        environment_belief_policy_tuples=[(environment, initial_belief, [policy])],
        num_episodes=num_episodes,
        num_steps=num_steps,
        alpha=0.1,
        n_jobs=2,
        deployment_type=DeploymentType.LOCAL
    )

    # Compare aggregate statistics instead of exact histories
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

    # Print out rewards for debugging
    print("\nRewards from 1 job:", stats_1job['rewards_per_episode'])
    print("Rewards from 2 jobs:", stats_2jobs['rewards_per_episode'])

    # Assert structural properties that should be identical
    assert stats_1job['total_episodes'] == stats_2jobs['total_episodes'] == num_episodes
    assert stats_1job['total_steps'] == stats_2jobs['total_steps'] == num_episodes * num_steps
    assert stats_1job['total_listens'] + stats_1job['total_opens'] == stats_1job['total_steps']
    assert stats_2jobs['total_listens'] + stats_2jobs['total_opens'] == stats_2jobs['total_steps']
    
    # Check that the distribution of actions is reasonable
    for stats in [stats_1job, stats_2jobs]:
        # Each episode should have at least one action
        assert all(listens + opens > 0 for listens, opens 
                  in zip(stats['listens_per_episode'], stats['opens_per_episode']))
        
        # Each episode should have steps matching the num_steps parameter
        assert all(listens + opens == num_steps for listens, opens 
                  in zip(stats['listens_per_episode'], stats['opens_per_episode']))
        
        # Calculate theoretical bounds for episode rewards
        # Worst case: All steps open wrong door (-100 * num_steps) = -300
        # Best case: All steps open right door (+10 * num_steps) = +30
        min_possible_reward = -100.0 * num_steps
        max_possible_reward = 10.0 * num_steps
        
        # Check each episode's rewards
        for i, r in enumerate(stats['rewards_per_episode']):
            assert min_possible_reward <= r <= max_possible_reward, \
                f"Episode {i} reward {r} outside bounds [{min_possible_reward}, {max_possible_reward}]"
    
    # The average reward per episode should be roughly similar
    # We use a large tolerance since the policy is stochastic and rewards have high variance
    avg_reward_1job = stats_1job['total_reward'] / stats_1job['total_episodes']
    avg_reward_2jobs = stats_2jobs['total_reward'] / stats_2jobs['total_episodes']
    reward_difference = abs(avg_reward_1job - avg_reward_2jobs)
    max_reward_difference = 100.0  # Increased tolerance due to high variance in rewards
    assert reward_difference < max_reward_difference, \
        f"Average reward difference {reward_difference} exceeds maximum allowed {max_reward_difference}"


def test_create_policy_configurations_df():
    """Test the creation of policy configurations DataFrame."""
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

    # Execute
    df = create_policy_configurations_df([(environment, initial_belief, [policy])])

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1  # One row for one policy
    assert 'environment' in df.columns
    assert 'policy' in df.columns
    assert 'policy_type' in df.columns
    assert df.iloc[0]['environment'] == environment.name
    assert df.iloc[0]['policy'] == policy.name
    assert df.iloc[0]['policy_type'] == policy.__class__.__name__
    
    # Check that policy parameters are included
    assert 'depth' in df.columns
    assert 'exploration_constant' in df.columns
    assert 'n_simulations' in df.columns
    assert df.iloc[0]['depth'] == 3
    assert df.iloc[0]['exploration_constant'] == 1.0
    assert df.iloc[0]['n_simulations'] == 2


def test_create_policy_configurations_df_multiple_policies():
    """Test the creation of policy configurations DataFrame with multiple policies."""
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy1 = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy1",
        n_simulations=2
    )
    policy2 = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=4,
        exploration_constant=1.5,
        name="TestPolicy2",
        n_simulations=3
    )
    initial_belief = get_initial_belief(environment, n_particles=3)

    # Execute
    df = create_policy_configurations_df([(environment, initial_belief, [policy1, policy2])])

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # Two rows for two policies
    assert 'environment' in df.columns
    assert 'policy' in df.columns
    assert 'policy_type' in df.columns
    
    # Check first policy
    policy1_row = df[df['policy'] == policy1.name].iloc[0]
    assert policy1_row['environment'] == environment.name
    assert policy1_row['policy_type'] == policy1.__class__.__name__
    assert policy1_row['depth'] == 3
    assert policy1_row['exploration_constant'] == 1.0
    assert policy1_row['n_simulations'] == 2
    
    # Check second policy
    policy2_row = df[df['policy'] == policy2.name].iloc[0]
    assert policy2_row['environment'] == environment.name
    assert policy2_row['policy_type'] == policy2.__class__.__name__
    assert policy2_row['depth'] == 4
    assert policy2_row['exploration_constant'] == 1.5
    assert policy2_row['n_simulations'] == 3


def test_compare_multiple_environments_policies_with_configurations(temp_cache_dir):
    """Test that policy configurations are correctly included in the comparison results."""
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
    _, merged_df = compare_multiple_environments_policies(
        environment_belief_policy_tuples=[(environment, initial_belief, [policy])],
        num_episodes=num_episodes,
        num_steps=num_steps,
        alpha=0.1,
        cache_dir_path=temp_cache_dir,
        deployment_type=DeploymentType.LOCAL
    )

    # Assert
    assert isinstance(merged_df, pd.DataFrame)
    
    # Check that both statistics and policy configurations are present
    assert 'environment' in merged_df.columns
    assert 'policy' in merged_df.columns
    assert 'average_return' in merged_df.columns  # Statistics column
    assert 'depth' in merged_df.columns  # Policy configuration column
    assert 'exploration_constant' in merged_df.columns  # Policy configuration column
    assert 'n_simulations' in merged_df.columns  # Policy configuration column
    
    # Check values
    row = merged_df.iloc[0]
    assert row['environment'] == environment.name
    assert row['policy'] == policy.name
    assert row['depth'] == 3
    assert row['exploration_constant'] == 1.0
    assert row['n_simulations'] == 2


def test_compare_multiple_environments_policies_configurations_multiple_policies(temp_cache_dir):
    """Test policy configurations with multiple environments and policies."""
    # Setup
    environment1 = TigerPOMDP(discount_factor=0.95, name="TigerPOMDP_095")
    environment2 = TigerPOMDP(discount_factor=0.99, name="TigerPOMDP_099")
    policy1 = POMCP(
        environment=environment1,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy1",
        n_simulations=2
    )
    policy2 = POMCP(
        environment=environment2,
        discount_factor=0.99,
        depth=4,
        exploration_constant=1.5,
        name="TestPolicy2",
        n_simulations=3
    )
    initial_belief1 = get_initial_belief(environment1, n_particles=3)
    initial_belief2 = get_initial_belief(environment2, n_particles=3)
    num_episodes = 2
    num_steps = 3

    # Execute
    _, merged_df = compare_multiple_environments_policies(
        environment_belief_policy_tuples=[
            (environment1, initial_belief1, [policy1]),
            (environment2, initial_belief2, [policy2])
        ],
        num_episodes=num_episodes,
        num_steps=num_steps,
        alpha=0.1,
        cache_dir_path=temp_cache_dir,
        deployment_type=DeploymentType.LOCAL
    )

    # Assert
    assert isinstance(merged_df, pd.DataFrame)
    assert len(merged_df) == 2  # Two rows for two environment-policy pairs
    
    # Check first environment-policy pair
    row1 = merged_df[merged_df['policy'] == policy1.name].iloc[0]
    assert row1['environment'] == environment1.name
    assert row1['depth'] == 3
    assert row1['exploration_constant'] == 1.0
    assert row1['n_simulations'] == 2
    
    # Check second environment-policy pair
    row2 = merged_df[merged_df['policy'] == policy2.name].iloc[0]
    assert row2['environment'] == environment2.name
    assert row2['depth'] == 4
    assert row2['exploration_constant'] == 1.5
    assert row2['n_simulations'] == 3
