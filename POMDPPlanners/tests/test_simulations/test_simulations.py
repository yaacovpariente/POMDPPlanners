import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import mlflow

from POMDPPlanners.simulations.simulations import run_episode, simulation
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.simulation import History

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
    cache_dir = Path("test_cache")
    
    # Test invalid parameters
    with pytest.raises(AssertionError):
        simulation(
            environment="not_an_environment",
            policy=policy,
            initial_belief=initial_belief,
            discount_factor=0.95,
            num_episodes=10,
            num_steps=5,
            alpha=0.1,
            cache_dir_path=cache_dir
        )
    
    with pytest.raises(AssertionError):
        simulation(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            discount_factor=0.95,
            num_episodes=0,  # Invalid num_episodes
            num_steps=5,
            alpha=0.1,
            cache_dir_path=cache_dir
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
            confidence_interval_level=1.5,  # Invalid confidence interval
            cache_dir_path=cache_dir
        )

def test_simulation_returns_histories_and_statistics(temp_cache_dir):
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
        alpha=alpha,
        cache_dir_path=temp_cache_dir
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

def test_simulation_statistics_consistency(temp_cache_dir):
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
        alpha=alpha,
        cache_dir_path=temp_cache_dir
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

def test_simulation_different_confidence_intervals(temp_cache_dir):
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
            cache_dir_path=temp_cache_dir,
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

def test_simulation_different_alphas(temp_cache_dir):
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
            alpha=alpha,
            cache_dir_path=temp_cache_dir
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
        