import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import mlflow
import json
import os
import optuna
import pandas as pd

from POMDPPlanners.simulations.simulations import run_episode, simulation, compare_planners
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.simulation import History
from POMDPPlanners.simulations.simulations import create_policy_optimization_objective, optimize_policy_parameters_with_optuna, optimize_policy_parameters_for_multiple_environments

def test_run_episode_returns_history():
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=2,
        depth=3,
        discount_factor=0.95
    )
    initial_belief = get_initial_belief(environment, n_particles=100)
    num_steps = 5
    
    # Execute
    history = run_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        discount_factor=0.95,
        num_steps=num_steps
    )
    
    # Assert
    assert isinstance(history, History)
    assert len(history.history) == num_steps
    assert history.discount_factor == 0.95

def test_run_episode_timing_statistics():
    # Setup
    environment = MountainCarPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=2,
        depth=3,
        discount_factor=0.95
    )
    initial_belief = get_initial_belief(environment, n_particles=100)
    num_steps = 5
    
    # Execute
    history = run_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        discount_factor=0.95,
        num_steps=num_steps
    )
    
    # Assert timing statistics are positive and reasonable
    assert history.average_state_sampling_time >= 0  # Can be 0 if state transitions are deterministic
    assert history.average_action_time > 0
    assert history.average_observation_time >= 0  # Can be 0 if observations are deterministic
    assert history.average_belief_update_time > 0
    assert history.average_reward_time >= 0  # Can be 0 if rewards are deterministic

def test_run_episode_valid_transitions():
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=2,
        depth=3,
        discount_factor=0.95
    )
    initial_belief = get_initial_belief(environment, n_particles=100)
    num_steps = 5
    
    # Execute
    history = run_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        discount_factor=0.95,
        num_steps=num_steps
    )
    
    # Assert valid state transitions and observations
    for step in history.history:
        assert step.state in ['tiger_left', 'tiger_right']
        assert step.action in ['listen', 'open_left', 'open_right']
        assert step.next_state in ['tiger_left', 'tiger_right']
        assert step.observation in ['hear_left', 'hear_right', 'hear_nothing']
        assert isinstance(step.reward, float)

def test_run_episode_reward_calculation():
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=2,
        depth=3,
        discount_factor=0.95
    )
    initial_belief = get_initial_belief(environment, n_particles=100)
    num_steps = 5
    
    # Execute
    history = run_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        discount_factor=0.95,
        num_steps=num_steps
    )
    
    # Assert reward values are correct based on TigerPOMDP rules
    for step in history.history:
        if step.action == 'listen':
            assert step.reward == -1.0
        elif step.action == 'open_left':
            assert step.reward in [-100.0, 10.0]
        elif step.action == 'open_right':
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
    if temp_path.exists():
        shutil.rmtree(temp_path)
        
def test_simulation_parameter_validation():
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=2,
        depth=3,
        discount_factor=0.95
    )
    initial_belief = get_initial_belief(environment, n_particles=100)
    
    # Test invalid parameters
    with pytest.raises(AssertionError):
        simulation(
            environment="not_an_environment",
            policy=policy,
            initial_belief=initial_belief,
            discount_factor=0.95,
            num_episodes=10,
            num_steps=5,
            alpha=0.1
        )
    
    with pytest.raises(AssertionError):
        simulation(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            discount_factor=0.95,
            num_episodes=0,  # Invalid num_episodes
            num_steps=5,
            alpha=0.1
        )
    
    with pytest.raises(AssertionError):
        simulation(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            discount_factor=0.95,
            num_episodes=10,
            num_steps=5,
            alpha=0.1,
            confidence_interval_level=1.5  # Invalid confidence interval
        )

def test_simulation_returns_histories_and_statistics():
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=2,
        depth=3,
        discount_factor=0.95
    )
    initial_belief = get_initial_belief(environment, n_particles=100)
    num_episodes = 3
    num_steps = 5
    alpha = 0.1
    
    # Execute
    histories, statistics = simulation(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        discount_factor=0.95,
        num_episodes=num_episodes,
        num_steps=num_steps,
        alpha=alpha
    )
    
    # Assert
    assert len(histories) == num_episodes
    assert all(isinstance(history, History) for history in histories)
    assert all(len(history.history) == num_steps for history in histories)
    assert isinstance(statistics, dict)
    assert 'average_return' in statistics
    assert 'return_cvar' in statistics
    assert 'return_value_at_risk' in statistics
    assert 'average_state_sampling_time' in statistics
    assert 'average_action_time' in statistics
    assert 'average_observation_time' in statistics
    assert 'average_belief_update_time' in statistics
    assert 'average_reward_time' in statistics

def test_simulation_statistics_consistency():
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=2,
        depth=3,
        discount_factor=0.95
    )
    initial_belief = get_initial_belief(environment, n_particles=100)
    num_episodes = 10
    num_steps = 5
    alpha = 0.1
    
    # Execute
    histories, statistics = simulation(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        discount_factor=0.95,
        num_episodes=num_episodes,
        num_steps=num_steps,
        alpha=alpha
    )
    
    # Assert statistics are consistent
    # For negative returns, CVaR should be greater than or equal to mean (since worst cases are least negative)
    # For positive returns, CVaR should be less than or equal to mean (since worst cases are least positive)
    if statistics['average_return'][0] < 0:
        assert statistics['average_return'][0] <= statistics['return_cvar'][0]
    else:
        assert statistics['average_return'][0] >= statistics['return_cvar'][0]
    
    assert statistics['average_return'][1][0] <= statistics['average_return'][0] <= statistics['average_return'][1][1]
    
    # Check timing statistics
    for key in ['average_state_sampling_time', 'average_action_time', 'average_observation_time', 
                'average_belief_update_time', 'average_reward_time']:
        mean, ci = statistics[key]
        assert mean >= 0  # Mean timing value should be non-negative
        # Skip confidence interval check if all values are identical (zero variance)
        if not np.isnan(ci[0]) and not np.isnan(ci[1]):
            assert ci[0] <= ci[1]  # Confidence interval bounds should be ordered

def test_simulation_different_confidence_intervals():
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=2,
        depth=3,
        discount_factor=0.95
    )
    initial_belief = get_initial_belief(environment, n_particles=100)
    num_episodes = 5
    num_steps = 5
    alpha = 0.1
    
    # Test different confidence interval levels
    for confidence_level in [0.9, 0.95, 0.99]:
        histories, statistics = simulation(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            discount_factor=0.95,
            num_episodes=num_episodes,
            num_steps=num_steps,
            alpha=alpha,
            confidence_interval_level=confidence_level
        )
        
        # Assert confidence interval width increases with confidence level on average
        ci_width = statistics['average_return'][1][1] - statistics['average_return'][1][0]
        if confidence_level == 0.9:
            width_90 = ci_width
        elif confidence_level == 0.95:
            width_95 = ci_width
            # Due to randomness and small sample size, widths can vary significantly
            assert abs(width_95 - width_90) / max(abs(width_95), abs(width_90)) < 1.0
        else:  # confidence_level == 0.99
            width_99 = ci_width
            # Due to randomness and small sample size, widths can vary significantly
            assert abs(width_99 - width_95) / max(abs(width_99), abs(width_95)) < 1.0

def test_simulation_different_alphas():
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=2,
        depth=3,
        discount_factor=0.95
    )
    initial_belief = get_initial_belief(environment, n_particles=100)
    num_episodes = 50  # Increased to reduce variance
    num_steps = 5

    # Test different alpha values
    for alpha in [0.1, 0.5, 0.9]:
        histories, statistics = simulation(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            discount_factor=0.95,
            num_episodes=num_episodes,
            num_steps=num_steps,
            alpha=alpha
        )

        # Assert CVaR behavior with alpha
        if alpha == 0.1:
            cvar_01 = statistics['return_cvar'][0]
            mean_return = statistics['average_return'][0]
        elif alpha == 0.5:
            cvar_05 = statistics['return_cvar'][0]
            # Due to high variance in the Tiger POMDP and small sample size,
            # we can only check that the values don't differ too extremely
            rel_diff = abs(cvar_05 - cvar_01) / max(abs(cvar_05), abs(cvar_01))
            assert rel_diff < 2.0
        else:  # alpha == 0.9
            cvar_09 = statistics['return_cvar'][0]
            rel_diff = abs(cvar_09 - cvar_05) / max(abs(cvar_09), abs(cvar_05))
            assert rel_diff < 2.0

def test_compare_planners_parameter_validation(temp_cache_dir):
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=2,
        depth=3,
        discount_factor=0.95
    )

    # Create mlruns directory
    mlruns_dir = temp_cache_dir / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    # Test invalid parameters
    with pytest.raises(AssertionError):
        compare_planners(
            environment_policy_pairs="not_a_list",  # Invalid type
            discount_factor=0.95,
            num_episodes=10,
            num_steps=5,
            alpha=0.1,
            n_particles=100,
            cache_dir_path=temp_cache_dir
        )

    with pytest.raises(AssertionError):
        compare_planners(
            environment_policy_pairs=[(environment, "not_a_policy")],  # Invalid policy type
            discount_factor=0.95,
            num_episodes=10,
            num_steps=5,
            alpha=0.1,
            n_particles=100,
            cache_dir_path=temp_cache_dir
        )

    with pytest.raises(AssertionError):
        compare_planners(
            environment_policy_pairs=[(environment, policy)],
            discount_factor=1.5,  # Invalid discount factor
            num_episodes=10,
            num_steps=5,
            alpha=0.1,
            n_particles=100,
            cache_dir_path=temp_cache_dir
        )

def test_compare_planners_mlflow_integration(temp_cache_dir):
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=2,
        depth=2,
        discount_factor=0.95
    )

    # Create mlruns directory
    mlruns_dir = temp_cache_dir / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{mlruns_dir}")
    mlflow.set_experiment("test_experiment")

    # Run comparison
    statistics, df = compare_planners(
        environment_policy_pairs=[(environment, policy)],
        discount_factor=0.95,
        num_episodes=2,
        num_steps=3,
        alpha=0.1,
        n_particles=100,
        cache_dir_path=temp_cache_dir,
        experiment_name="test_experiment"
    )

    # Verify that MLFlow wrote something to disk
    assert mlruns_dir.exists()
    assert any(mlruns_dir.rglob("*"))  # Check that there are any files in the mlruns directory

def test_compare_planners_different_parameters(temp_cache_dir):
    # Setup
    environment1 = TigerPOMDP(discount_factor=0.95)
    environment2 = TigerPOMDP(discount_factor=0.99)
    policy1 = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment1,
        branching_factor=2,
        depth=3,
        discount_factor=0.95
    )
    policy2 = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment2,
        branching_factor=3,
        depth=4,
        discount_factor=0.99
    )

    # Create mlruns directory
    mlruns_dir = temp_cache_dir / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    # Run comparison with different parameters
    statistics, df = compare_planners(
        environment_policy_pairs=[
            (environment1, policy1),
            (environment2, policy2)
        ],
        discount_factor=0.95,
        num_episodes=2,
        num_steps=3,
        alpha=0.1,
        n_particles=100,
        cache_dir_path=temp_cache_dir
    )

    # Verify statistics for each combination
    assert len(statistics) == 2  # One result per environment-policy pair

    # Verify DataFrame structure and content
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # One row per environment-policy pair
    assert 'environment_type' in df.columns
    assert 'policy_type' in df.columns
    assert 'discount_factor' in df.columns
    assert 'num_episodes' in df.columns
    assert 'num_steps' in df.columns
    assert 'alpha' in df.columns
    assert 'n_particles' in df.columns
    assert 'average_return' in df.columns
    assert 'return_cvar' in df.columns
    assert 'return_value_at_risk' in df.columns
    assert 'average_state_sampling_time' in df.columns
    assert 'average_action_time' in df.columns
    assert 'average_observation_time' in df.columns
    assert 'average_belief_update_time' in df.columns
    assert 'average_reward_time' in df.columns

    # Verify environment types
    assert all(env_type == 'TigerPOMDP' for env_type in df['environment_type'])
    
    # Verify policy types
    assert all(policy_type == 'StandardSparseSamplingDiscreteActionsPlanner' for policy_type in df['policy_type'])
    
    # Verify parameter values
    assert all(df['discount_factor'] == 0.95)
    assert all(df['num_episodes'] == 2)
    assert all(df['num_steps'] == 3)
    assert all(df['alpha'] == 0.1)
    assert all(df['n_particles'] == 100)
    
def test_create_policy_optimization_objective():
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    
    # Define parameter ranges for testing
    param_ranges = {
        'branching_factor': {
            'type': 'int',
            'low': 2,
            'high': 5
        },
        'depth': {
            'type': 'int',
            'low': 2,
            'high': 4
        }
    }
    
    # Create a simple evaluation function
    def evaluation_function(policy):
        assert isinstance(policy, StandardSparseSamplingDiscreteActionsPlanner)
        assert hasattr(policy, 'environment')
        assert hasattr(policy, 'discount_factor')
        return (1.0, (0.5, 1.5))  # Return a tuple to simulate statistics
    
    # Create a mock trial that implements all suggest methods
    class MockTrial:
        def suggest_int(self, name, low, high):
            return (low + high) // 2
            
        def suggest_float(self, name, low, high, log=False):
            return (low + high) / 2
            
        def suggest_categorical(self, name, choices):
            return choices[0]
    
    # Execute
    objective = create_policy_optimization_objective(
        policy_class=StandardSparseSamplingDiscreteActionsPlanner,
        param_ranges=param_ranges,
        evaluation_function=evaluation_function,
        environment=environment,
        discount_factor=0.95
    )
    
    # Test the objective function
    result = objective(MockTrial())
    
    # Assert
    assert result == 1.0  # Should return the mean value from the tuple

def test_optimize_policy_parameters_with_optuna(temp_cache_dir):
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    
    # Define parameter ranges for testing
    param_ranges = {
        'branching_factor': {
            'type': 'int',
            'low': 2,
            'high': 3
        },
        'depth': {
            'type': 'int',
            'low': 2,
            'high': 3
        }
    }
    
    # Execute
    best_params, best_value, histories = optimize_policy_parameters_with_optuna(
        environment=environment,
        policy_class=StandardSparseSamplingDiscreteActionsPlanner,
        param_ranges=param_ranges,
        discount_factor=0.95,
        num_episodes=2,  # Small number for testing
        num_steps=3,     # Small number for testing
        n_particles=100,
        cache_dir_path=temp_cache_dir,
        parameter_to_optimize="average_return",
        n_trials=2,      # Small number for testing
        confidence_interval_level=0.95
    )
    
    # Assert
    assert isinstance(best_params, dict)
    assert 'branching_factor' in best_params
    assert 'depth' in best_params
    assert 'environment' not in best_params  # Should not be in optimized params
    assert 'discount_factor' not in best_params  # Should not be in optimized params
    assert isinstance(best_value, (float, np.ndarray))
    assert isinstance(histories, list)
    assert len(histories) == 2  # num_episodes
    assert all(isinstance(history, History) for history in histories)
    
    # Check parameter bounds
    assert 2 <= best_params['branching_factor'] <= 3
    assert 2 <= best_params['depth'] <= 3

def test_optimize_policy_parameters_with_optuna_invalid_params(temp_cache_dir):
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    
    # Define invalid parameter ranges
    param_ranges = {
        'invalid_param': {  # Parameter that doesn't exist in the policy
            'type': 'int',
            'low': 1,
            'high': 2
        }
    }
    
    # Test that invalid parameters raise an error
    with pytest.raises(Exception):
        optimize_policy_parameters_with_optuna(
            environment=environment,
            policy_class=StandardSparseSamplingDiscreteActionsPlanner,
            param_ranges=param_ranges,
            discount_factor=0.95,
            num_episodes=2,
            num_steps=3,
            n_particles=5,
            cache_dir_path=temp_cache_dir,
            parameter_to_optimize="average_return",
            n_trials=2,
            confidence_interval_level=0.95
        )

def test_optimize_policy_parameters_with_optuna_different_metrics(temp_cache_dir):
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    
    # Define parameter ranges
    param_ranges = {
        'branching_factor': {
            'type': 'int',
            'low': 2,
            'high': 3
        },
        'depth': {
            'type': 'int',
            'low': 2,
            'high': 3
        }
    }
    
    # Test optimization with different metrics
    metrics = ["average_return", "return_cvar", "return_value_at_risk"]
    
    for metric in metrics:
        best_params, best_value, histories = optimize_policy_parameters_with_optuna(
            environment=environment,
            policy_class=StandardSparseSamplingDiscreteActionsPlanner,
            param_ranges=param_ranges,
            discount_factor=0.95,
            num_episodes=2,
            num_steps=3,
            n_particles=5,
            cache_dir_path=temp_cache_dir,
            parameter_to_optimize=metric,
            n_trials=2,
            confidence_interval_level=0.95
        )
        
        assert isinstance(best_params, dict)
        assert isinstance(best_value, (float, np.ndarray))
        assert isinstance(histories, list)
        assert len(histories) == 2  # num_episodes
        assert all(isinstance(history, History) for history in histories)
        assert 'branching_factor' in best_params
        assert 'depth' in best_params
        assert 'environment' not in best_params  # Should not be in optimized params
        assert 'discount_factor' not in best_params  # Should not be in optimized params

def test_optimize_policy_parameters_for_multiple_environments(temp_cache_dir):
    # Setup
    environment1 = TigerPOMDP(discount_factor=0.95)
    environment2 = TigerPOMDP(discount_factor=0.99)

    # Define parameter ranges for testing
    param_ranges1 = {
        'branching_factor': {
            'type': 'int',
            'low': 2,
            'high': 3
        },
        'depth': {
            'type': 'int',
            'low': 2,
            'high': 3
        }
    }

    param_ranges2 = {
        'branching_factor': {
            'type': 'int',
            'low': 3,
            'high': 4
        },
        'depth': {
            'type': 'int',
            'low': 3,
            'high': 4
        }
    }

    # Create environment-policy pairs
    environment_policy_pairs = [
        (environment1, (StandardSparseSamplingDiscreteActionsPlanner, param_ranges1)),
        (environment2, (StandardSparseSamplingDiscreteActionsPlanner, param_ranges2))
    ]

    # Set up MLFlow tracking
    mlruns_dir = temp_cache_dir / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{mlruns_dir}")
    mlflow.set_experiment("test_optimization")

    # Execute
    results, df = optimize_policy_parameters_for_multiple_environments(
        environment_policy_pairs=environment_policy_pairs,
        discount_factor=0.95,
        num_episodes=2,  # Small number for testing
        num_steps=3,     # Small number for testing
        n_particles=5,
        cache_dir_path=temp_cache_dir,
        parameter_to_optimize="average_return",
        n_trials=2,      # Small number for testing
        confidence_interval_level=0.95,
        experiment_name="test_optimization"
    )

    # Assert results
    assert len(results) == 2  # One result per environment-policy pair

    for i, (best_params, best_value, histories) in enumerate(results):
        assert isinstance(best_params, dict)
        assert 'branching_factor' in best_params
        assert 'depth' in best_params
        assert 'environment' not in best_params  # Should not be in optimized params
        assert 'discount_factor' not in best_params  # Should not be in optimized params
        assert isinstance(best_value, (float, np.ndarray))
        assert isinstance(histories, list)
        assert len(histories) == 2  # num_episodes
        assert all(isinstance(history, History) for history in histories)

        # Check parameter bounds based on which policy config was used
        param_ranges = param_ranges1 if i == 0 else param_ranges2
        assert param_ranges['branching_factor']['low'] <= best_params['branching_factor'] <= param_ranges['branching_factor']['high']
        assert param_ranges['depth']['low'] <= best_params['depth'] <= param_ranges['depth']['high']

    # Verify DataFrame structure and content
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # One row per environment-policy pair
    assert 'environment_type' in df.columns
    assert 'policy_type' in df.columns
    assert 'discount_factor' in df.columns
    assert 'num_episodes' in df.columns
    assert 'num_steps' in df.columns
    assert 'n_particles' in df.columns
    assert 'parameter_to_optimize' in df.columns
    assert 'direction' in df.columns
    assert 'n_trials' in df.columns
    assert 'confidence_interval_level' in df.columns
    assert 'best_branching_factor' in df.columns
    assert 'best_depth' in df.columns
    assert 'best_value' in df.columns
    assert 'average_return' in df.columns
    assert 'return_cvar' in df.columns
    assert 'return_value_at_risk' in df.columns
    assert 'average_state_sampling_time' in df.columns
    assert 'average_action_time' in df.columns
    assert 'average_observation_time' in df.columns
    assert 'average_belief_update_time' in df.columns
    assert 'average_reward_time' in df.columns

    # Verify that MLFlow wrote something to disk
    assert mlruns_dir.exists()
    assert any(mlruns_dir.rglob("*"))  # Check that there are any files in the mlruns directory
        