import numpy as np
import pytest
from pathlib import Path

from POMDPPlanners.environments.discrete_light_dark_pomdp import DiscreteLightDarkPOMDP
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import ObservationModel


def test_initialization():
    """Test initialization with default parameters"""
    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        transition_error_prob=0.05,
        observation_error_prob=0.05,
        obstacle_hit_probability=0.2,
        obstacle_reward=-10.0,
        goal_reward=10.0,
        fuel_cost=2.0,
        grid_size=11,
        is_stochastic_reward=True,
    )

    # Check default parameters
    assert env.transition_error_prob == 0.05
    assert env.observation_error_prob == 0.05
    assert env.obstacle_hit_probability == 0.2
    assert env.obstacle_reward == -10.0
    assert env.goal_reward == 10.0
    assert env.fuel_cost == 2.0
    assert env.grid_size == 11
    assert env.beacon_radius == 1.0

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
    env = DiscreteLightDarkPOMDP(discount_factor=0.95, transition_error_prob=0.1)
    state = np.array([5, 5])

    # Test 'up' action
    dist = env.state_transition_model(state, "up")
    assert isinstance(dist, DiscreteDistribution)

    # Check probabilities
    # The implementation uses the order: up, down, right, left
    # For 'up' action, the first position (index 0) should have high probability
    expected_probs = np.array(
        [0.9, 0.1 / 3, 0.1 / 3, 0.1 / 3]
    )  # 0.9 for correct action, 0.1/3 for others
    assert np.allclose(dist.probs, expected_probs)

    # Check values
    expected_values = [
        state + np.array([0, 1]),  # up
        state + np.array([0, -1]),  # down
        state + np.array([1, 0]),  # right
        state + np.array([-1, 0]),  # left
    ]
    assert all(np.array_equal(v1, v2) for v1, v2 in zip(dist.values, expected_values))


def test_observation_model():
    """Test observation model"""
    env = DiscreteLightDarkPOMDP(discount_factor=0.95, observation_error_prob=0.1)
    state = np.array([5, 5])

    # Test observation model
    dist = env.observation_model(state, "up")
    assert isinstance(dist, ObservationModel)

    # Check that the correct state has highest probability
    correct_state_idx = (
        len(dist.distribution.values) - 1
    )  # Last value is the correct state
    assert (
        dist.distribution.probs[correct_state_idx] > 0.5
    )  # Should be 1 - observation_error_prob


def test_reward_function():
    """Test reward function"""
    env = DiscreteLightDarkPOMDP(discount_factor=0.95)

    # Test goal state reward
    state = np.array([9, 5])
    reward = env.reward(state, "right")  # Moves to [10, 5] which is goal
    assert reward == env.goal_reward - env.fuel_cost

    # Test obstacle hit reward
    state = np.array([2, 5])
    reward = env.reward(state, "right")  # Moves to [3, 5] which is obstacle
    assert reward == env.obstacle_reward - env.fuel_cost - np.linalg.norm(np.array([3, 5]) - env.goal_state)

    # Test normal movement reward
    state = np.array([1, 1])
    reward = env.reward(state, "right")
    assert reward == -env.fuel_cost - np.linalg.norm(np.array([2, 1]) - env.goal_state)


def test_is_terminal():
    """Test terminal state detection"""
    env = DiscreteLightDarkPOMDP(discount_factor=0.95)

    # Test goal state
    assert env.is_terminal(env.goal_state)

    # Test obstacle state
    assert env.is_terminal(env.obstacles[:, 0])  # First obstacle

    # Test out of grid state
    assert env.is_terminal(np.array([-1, 5]))
    assert env.is_terminal(np.array([12, 5]))  # grid_size + 1

    # Test non-terminal state
    assert not env.is_terminal(np.array([1, 1]))


def test_initial_distributions():
    """Test initial state and observation distributions"""
    env = DiscreteLightDarkPOMDP(discount_factor=0.95)

    # Test initial state distribution
    state_dist = env.initial_state_dist()
    assert isinstance(state_dist, DiscreteDistribution)
    assert np.array_equal(state_dist.values[0], env.start_state)
    assert state_dist.probs[0] == 1.0

    # Test initial observation distribution
    obs_dist = env.initial_observation_dist()
    assert isinstance(obs_dist, DiscreteDistribution)
    assert obs_dist.values[0] == 1.0
    assert obs_dist.probs[0] == 1.0


def test_get_actions():
    """Test action list retrieval"""
    env = DiscreteLightDarkPOMDP(discount_factor=0.95)
    actions = env.get_actions()
    assert set(actions) == {"up", "down", "right", "left"}


def test_visualize_path(tmp_path):
    """Test path visualization"""
    env = DiscreteLightDarkPOMDP(discount_factor=0.95)
    path = [np.array([0, 5]), np.array([1, 5]), np.array([2, 5]), np.array([3, 5])]

    # Test visualization with temporary path
    cache_path = tmp_path / "test_animation.gif"
    env.visualize_path(path, cache_path)

    # Verify file was created
    assert cache_path.exists()
