import numpy as np
import pytest
from pathlib import Path

from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import ContinuousLightDarkPOMDP
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import ObservationModel


@pytest.fixture
def base_light_dark_environment() -> ContinuousLightDarkPOMDP:
    """Fixture providing a base ContinuousLightDarkPOMDP environment for comparison."""
    return ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        state_transition_cov_matrix=np.eye(2),
        observation_cov_matrix=np.eye(2),
        obstacle_hit_probability=0.2,
        obstacle_reward=-10.0,
        goal_reward=10.0,
        fuel_cost=2.0,
        grid_size=11,
        goal_state_radius=1.5,
        beacon_radius=1.0,
        obstacle_radius=1.5,
    )


class TestContinuousLightDarkPOMDPEquality:
    """Test suite for ContinuousLightDarkPOMDP equality comparisons."""
    
    def test_same_discount_factor(self, base_light_dark_environment: ContinuousLightDarkPOMDP):
        """Test that ContinuousLightDarkPOMDPs with same discount factor are equal."""
        other_env = ContinuousLightDarkPOMDP(
            discount_factor=0.95,
            state_transition_cov_matrix=np.eye(2),
            observation_cov_matrix=np.eye(2),
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            goal_state_radius=1.5,
            beacon_radius=1.0,
            obstacle_radius=1.5,
        )
        assert base_light_dark_environment == other_env
        assert other_env == base_light_dark_environment  # Test symmetry
    
    def test_different_discount_factor(self, base_light_dark_environment: ContinuousLightDarkPOMDP):
        """Test that ContinuousLightDarkPOMDPs with different discount factors are not equal."""
        other_env = ContinuousLightDarkPOMDP(
            discount_factor=0.8,
            state_transition_cov_matrix=np.eye(2),
            observation_cov_matrix=np.eye(2),
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            goal_state_radius=1.5,
            beacon_radius=1.0,
            obstacle_radius=1.5,
        )
        assert base_light_dark_environment != other_env
        assert other_env != base_light_dark_environment  # Test symmetry
    
    def test_different_covariance_matrices(self, base_light_dark_environment: ContinuousLightDarkPOMDP):
        """Test that ContinuousLightDarkPOMDPs with different covariance matrices are not equal."""
        other_env = ContinuousLightDarkPOMDP(
            discount_factor=0.95,
            state_transition_cov_matrix=2 * np.eye(2),  # Different state transition covariance
            observation_cov_matrix=np.eye(2),
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            goal_state_radius=1.5,
            beacon_radius=1.0,
            obstacle_radius=1.5,
        )
        assert base_light_dark_environment != other_env
        
        other_env = ContinuousLightDarkPOMDP(
            discount_factor=0.95,
            state_transition_cov_matrix=np.eye(2),
            observation_cov_matrix=2 * np.eye(2),  # Different observation covariance
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            goal_state_radius=1.5,
            beacon_radius=1.0,
            obstacle_radius=1.5,
        )
        assert base_light_dark_environment != other_env
    
    def test_different_radii(self, base_light_dark_environment: ContinuousLightDarkPOMDP):
        """Test that ContinuousLightDarkPOMDPs with different radii are not equal."""
        other_env = ContinuousLightDarkPOMDP(
            discount_factor=0.95,
            state_transition_cov_matrix=np.eye(2),
            observation_cov_matrix=np.eye(2),
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            goal_state_radius=2.0,  # Different goal radius
            beacon_radius=1.0,
            obstacle_radius=1.5,
        )
        assert base_light_dark_environment != other_env
        
        other_env = ContinuousLightDarkPOMDP(
            discount_factor=0.95,
            state_transition_cov_matrix=np.eye(2),
            observation_cov_matrix=np.eye(2),
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            goal_state_radius=1.5,
            beacon_radius=2.0,  # Different beacon radius
            obstacle_radius=1.5,
        )
        assert base_light_dark_environment != other_env
        
        other_env = ContinuousLightDarkPOMDP(
            discount_factor=0.95,
            state_transition_cov_matrix=np.eye(2),
            observation_cov_matrix=np.eye(2),
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            goal_state_radius=1.5,
            beacon_radius=1.0,
            obstacle_radius=2.0,  # Different obstacle radius
        )
        assert base_light_dark_environment != other_env


def test_initialization():
    """Test initialization with default parameters"""
    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        state_transition_cov_matrix=np.eye(2),
        observation_cov_matrix=np.eye(2),
        obstacle_hit_probability=0.2,
        obstacle_reward=-10.0,
        goal_reward=10.0,
        fuel_cost=2.0,
        grid_size=11,
        goal_state_radius=1.5,
        beacon_radius=1.0,
        obstacle_radius=1.5,
    )

    # Check default parameters
    assert np.array_equal(env.state_transition_cov_matrix, np.eye(2))
    assert np.array_equal(env.observation_cov_matrix, np.eye(2))
    assert env.obstacle_hit_probability == 0.2
    assert env.obstacle_reward == -10.0
    assert env.goal_reward == 10.0
    assert env.fuel_cost == 2.0
    assert env.grid_size == 11
    assert env.goal_state_radius == 1.5
    assert env.beacon_radius == 1.0
    assert env.obstacle_radius == 1.5

    # Check default state positions
    assert np.array_equal(env.goal_state, np.array([10, 5]))
    assert np.array_equal(env.start_state, np.array([0, 5]))

    # Check default beacons
    expected_beacons = np.array(
        [[0, 0, 0, 5, 5, 5, 10, 10, 10], [0, 5, 10, 0, 5, 10, 0, 5, 10]]
    )
    assert np.array_equal(env.beacons, expected_beacons)

    # Check default obstacles
    expected_obstacles = np.array([[3, 7], [5, 5]])
    assert np.array_equal(env.obstacles, expected_obstacles)


def test_state_transition_model():
    """Test state transition model"""
    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        state_transition_cov_matrix=np.eye(2) * 0.1  # Small noise
    )
    state = np.array([5, 5])

    # Test state transition
    dist = env.state_transition_model(state, "up")
    assert isinstance(dist, DiscreteDistribution)
    
    # Check that the mean of the distribution is correct
    next_state = dist.values[0]
    expected_next_state = state + np.array([0, 1])  # up action
    assert np.allclose(next_state, expected_next_state, atol=0.5)  # Allow for some noise


def test_observation_model():
    """Test observation model"""
    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        observation_cov_matrix=np.eye(2) * 0.1  # Small noise
    )
    state = np.array([5, 5])

    # Test observation model
    dist = env.observation_model(state, "up")
    assert isinstance(dist, DiscreteDistribution)
    
    # Check that the observation is close to the true state
    observation = dist.values[0]
    assert np.allclose(observation, state, atol=0.5)  # Allow for some noise


def test_reward_function():
    """Test reward function"""
    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        obstacle_hit_probability=1.0,
        goal_state_radius=1.5,
        obstacle_radius=1.5
    )

    # Test goal state reward (state within goal radius)
    state = np.array([9, 5])
    reward = env.reward(state, "right")  # Moves closer to goal
    assert reward > 0  # Should be positive due to goal reward

    # Test obstacle hit reward (state within obstacle radius)
    state = np.array([2, 5])
    reward = env.reward(state, "right")  # Moves closer to obstacle
    assert reward < 0  # Should be negative due to obstacle reward

    # Test normal movement reward
    state = np.array([1, 1])
    reward = env.reward(state, "right")
    assert reward < 0  # Should be negative due to fuel cost and distance to goal


def test_is_terminal():
    """Test terminal state detection"""
    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        goal_state_radius=1.5,
        obstacle_radius=1.5
    )

    # Test goal state (within radius)
    assert env.is_terminal(env.goal_state)
    assert env.is_terminal(env.goal_state + np.array([1, 0]))  # Within radius

    # Test obstacle state (within radius)
    assert env.is_terminal(env.obstacles[:, 0])  # First obstacle
    assert env.is_terminal(env.obstacles[:, 0] + np.array([1, 0]))  # Within radius

    # Test out of grid state
    assert env.is_terminal(np.array([-1, 5]))
    assert env.is_terminal(np.array([12, 5]))  # grid_size + 1

    # Test non-terminal state
    assert not env.is_terminal(np.array([1, 1]))


def test_compute_metrics():
    """Test computation of metrics for different simulation histories"""
    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        goal_state_radius=1.5,
        obstacle_radius=1.5
    )
    
    # Create test histories
    from POMDPPlanners.core.simulation import History, StepData
    from POMDPPlanners.core.belief import WeightedParticleBelief
    
    # Create a simple belief for testing
    def create_test_belief(state):
        return WeightedParticleBelief(
            particles=[state],
            log_weights=np.array([1.0]),
            resampling=False
        )
    
    # History 1: Reaches goal (within radius)
    history1 = History([
        StepData(
            state=np.array([0, 5]),
            action="right",
            next_state=np.array([1, 5]),
            observation=np.array([1, 5]),
            reward=-2.0,
            belief=create_test_belief(np.array([0, 5]))
        ),
        StepData(
            state=np.array([9, 5]),  # Within goal radius
            action="right",
            next_state=np.array([10, 5]),
            observation=np.array([10, 5]),
            reward=8.0,
            belief=create_test_belief(np.array([9, 5]))
        ),
    ], discount_factor=0.95, average_state_sampling_time=0.0, average_action_time=0.0, average_observation_time=0.0, average_belief_update_time=0.0, average_reward_time=0.0, actual_num_steps=2, reach_terminal_state=True)
    
    # History 2: Hits obstacle (within radius)
    history2 = History([
        StepData(
            state=np.array([0, 5]),
            action="right",
            next_state=np.array([1, 5]),
            observation=np.array([1, 5]),
            reward=-2.0,
            belief=create_test_belief(np.array([0, 5]))
        ),
        StepData(
            state=np.array([2, 5]),
            action="right",
            next_state=np.array([3, 5]),
            observation=np.array([3, 5]),
            reward=-12.0,  # Within obstacle radius
            belief=create_test_belief(np.array([2, 5]))
        ),
    ], discount_factor=0.95, average_state_sampling_time=0.0, average_action_time=0.0, average_observation_time=0.0, average_belief_update_time=0.0, average_reward_time=0.0, actual_num_steps=2, reach_terminal_state=True)
    
    # Compute metrics
    metrics = env.compute_metrics([history1, history2])
    
    # Convert metrics to dictionary for easier access
    metrics_dict = {metric.name: metric for metric in metrics}
    
    # Test goal reaching rate
    assert "goal_reaching_rate" in metrics_dict
    goal_rate = metrics_dict["goal_reaching_rate"]
    assert goal_rate.value == 0.5  # 1 out of 2 histories reach goal
    assert goal_rate.lower_confidence_bound <= goal_rate.value <= goal_rate.upper_confidence_bound
    
    # Test obstacle hit rate
    assert "obstacle_hit_rate" in metrics_dict
    obstacle_rate = metrics_dict["obstacle_hit_rate"]
    assert obstacle_rate.value == 0.5  # 1 out of 2 histories hits obstacle
    assert obstacle_rate.lower_confidence_bound <= obstacle_rate.value <= obstacle_rate.upper_confidence_bound
