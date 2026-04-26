"""Tests for Continuous Light Dark POMDP environment.

This module tests the Continuous Light Dark POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

import random
import time

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from POMDPPlanners.configs.environment_configs import (
    RiskAverseEnvironmentConfigsAPI,
)
from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.policy import PolicyRunData
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    ContinuousLightDarkPOMDPDiscreteActions,
    ObservationModelType,
    RewardModelType,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models import (
    ContinuousLightDarkDecayingHitProbabilityRewardModel,
    ContinuousLDDangerousStatesRewardModel,
)


# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


@pytest.fixture
def base_light_dark_environment() -> ContinuousLightDarkPOMDPDiscreteActions:
    """Fixture providing a base ContinuousLightDarkPOMDP environment for comparison."""
    return ContinuousLightDarkPOMDPDiscreteActions(
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


@pytest.fixture
def base_continuous_light_dark_pomdp() -> ContinuousLightDarkPOMDP:
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


@pytest.fixture
def pomdp():
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

    def test_same_discount_factor(
        self, base_light_dark_environment: ContinuousLightDarkPOMDPDiscreteActions
    ):
        """Test that ContinuousLightDarkPOMDPs with same discount factor are equal."""
        other_env = ContinuousLightDarkPOMDPDiscreteActions(
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

    def test_different_discount_factor(
        self, base_light_dark_environment: ContinuousLightDarkPOMDPDiscreteActions
    ):
        """Test that ContinuousLightDarkPOMDPs with different discount factors are not equal."""
        other_env = ContinuousLightDarkPOMDPDiscreteActions(
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

    def test_different_covariance_matrices(
        self, base_light_dark_environment: ContinuousLightDarkPOMDPDiscreteActions
    ):
        """Test that ContinuousLightDarkPOMDPs with different covariance matrices are not equal."""
        other_env = ContinuousLightDarkPOMDPDiscreteActions(
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

        other_env = ContinuousLightDarkPOMDPDiscreteActions(
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

    def test_different_radii(
        self, base_light_dark_environment: ContinuousLightDarkPOMDPDiscreteActions
    ):
        """Test that ContinuousLightDarkPOMDPs with different radii are not equal."""
        other_env = ContinuousLightDarkPOMDPDiscreteActions(
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

        other_env = ContinuousLightDarkPOMDPDiscreteActions(
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

        other_env = ContinuousLightDarkPOMDPDiscreteActions(
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
    env = ContinuousLightDarkPOMDPDiscreteActions(
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
    expected_beacons = np.array([[0, 0, 0, 5, 5, 5, 10, 10, 10], [0, 5, 10, 0, 5, 10, 0, 5, 10]])
    assert np.array_equal(env.beacons, expected_beacons)

    # Check default obstacles (now 2xN format like beacons)
    expected_obstacles = np.array([[3, 5], [7, 5]])
    assert np.array_equal(env.obstacles, expected_obstacles)


def test_beacons_and_obstacles_array_structure():
    """Test that beacons and obstacles are numpy arrays with correct shapes after initialization."""
    env = ContinuousLightDarkPOMDPDiscreteActions(
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

    # Test beacons structure
    assert isinstance(env.beacons, np.ndarray), "beacons should be a numpy array"
    assert env.beacons.ndim == 2, "beacons should be 2-dimensional"
    assert env.beacons.shape[0] == 2, "beacons should have 2 rows (x and y coordinates)"
    assert env.beacons.shape[1] == 9, "beacons should have 9 columns"

    # Test obstacles structure
    assert isinstance(env.obstacles, np.ndarray), "obstacles should be a numpy array"
    assert env.obstacles.ndim == 2, "obstacles should be 2-dimensional"
    assert env.obstacles.shape[1] == 2, "obstacles should have 2 columns (x and y coordinates)"
    assert env.obstacles.shape[0] == 2, "obstacles should have 2 rows"


def test_state_transition_model(pomdp):
    # Test state transition via the env-level sample API.
    state = np.array([0.0, 0.0])
    action = np.array([0.0, 0.0])
    next_state = pomdp.sample_next_state(state, action)
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (2,)


def test_observation_model(pomdp):
    # Test observation model via the env-level sample API.
    state = np.array([0.0, 0.0])
    action = np.array([0.0, 0.0])
    obs = pomdp.sample_observation(state, action)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)


def test_sample_next_step(pomdp):
    # Test sample_next_step method
    state = np.array([0.0, 0.0])
    action = np.array([0.0, 0.0])
    next_state, observation, reward = pomdp.sample_next_step(state, action)

    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (2,)
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (2,)
    assert isinstance(reward, float)


def test_initial_state_distribution(pomdp):
    # Test initial state distribution
    dist = pomdp.initial_state_dist()
    state = dist.sample()[0]
    assert isinstance(state, np.ndarray)
    assert state.shape == (2,)


def test_reward_function():
    """Test reward function"""
    env = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95,
        obstacle_hit_probability=1.0,
        goal_state_radius=1.5,
        obstacle_radius=1.5,
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
    env = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95, goal_state_radius=1.5, obstacle_radius=1.5
    )

    # Test goal state (within radius)
    assert env.is_terminal(env.goal_state)
    assert env.is_terminal(env.goal_state + np.array([1, 0]))  # Within radius

    # Test obstacle state (within radius)
    assert env.is_terminal(env.obstacles[:, 0])  # First obstacle
    assert env.is_terminal(env.obstacles[:, 0] + np.array([1, 0]))  # Within radius

    # Test out of grid state (no longer terminal)
    assert not env.is_terminal(np.array([-1, 5]))
    assert not env.is_terminal(np.array([12, 5]))  # grid_size + 1

    # Test non-terminal state
    assert not env.is_terminal(np.array([1, 1]))


def test_reward_range():
    """Test that reward range is correctly calculated.

    Purpose: Validates that ContinuousLightDarkPOMDPDiscreteActions calculates reward range based on environment parameters

    Given: A ContinuousLightDarkPOMDPDiscreteActions environment with specific parameters
    When: Environment reward_range attribute is checked
    Then: Returns calculated range based on maximum distance to goal and reward parameters

    Test type: unit
    """
    env = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95,
        obstacle_reward=-15.0,
        goal_reward=25.0,
        fuel_cost=2.0,
        grid_size=11,
    )

    # Expected calculation for STANDARD reward model:
    # Maximum distance to goal is diagonal of grid: sqrt(2) * grid_size
    max_distance_to_goal = np.sqrt(2) * 11  # grid_size=11
    # Min: -fuel_cost - max_distance + obstacle_reward
    expected_min = -2.0 - max_distance_to_goal + (-15.0)
    # Max: -fuel_cost + goal_reward
    expected_max = -2.0 + 25.0

    expected_reward_range = (expected_min, expected_max)
    assert env.reward_range == expected_reward_range

    # Test with another environment instance with different parameters
    env2 = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.99,
        obstacle_reward=-50.0,
        goal_reward=100.0,
        fuel_cost=3.0,
        grid_size=15,
    )

    # Calculate expected range for different parameters
    max_distance2 = np.sqrt(2) * 15  # grid_size=15
    expected_min2 = -3.0 - max_distance2 + (-50.0)
    expected_max2 = -3.0 + 100.0
    expected_reward_range2 = (expected_min2, expected_max2)

    assert env2.reward_range == expected_reward_range2


def test_compute_metrics():
    """Test computation of metrics for different simulation histories"""
    env = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95, goal_state_radius=1.5, obstacle_radius=1.5
    )

    # Create test histories
    # Create a simple belief for testing
    def create_test_belief(state):
        return WeightedParticleBelief(
            particles=[state], log_weights=np.array([1.0]), resampling=False
        )

    # History 1: Reaches goal (within radius)
    history1 = History(
        [
            StepData(
                state=np.array([0, 5]),
                action="right",
                next_state=np.array([1, 5]),
                observation=np.array([1, 5]),
                reward=-2.0,
                belief=create_test_belief(np.array([0, 5])),
            ),
            StepData(
                state=np.array([9, 5]),  # Within goal radius
                action="right",
                next_state=np.array([10, 5]),
                observation=np.array([10, 5]),
                reward=8.0,
                belief=create_test_belief(np.array([9, 5])),
            ),
        ],
        discount_factor=0.95,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=2,
        reach_terminal_state=True,
        policy_run_data=[PolicyRunData(info_variables=[])],
    )

    # History 2: Hits obstacle (within radius)
    history2 = History(
        [
            StepData(
                state=np.array([0, 5]),
                action="right",
                next_state=np.array([1, 5]),
                observation=np.array([1, 5]),
                reward=-2.0,
                belief=create_test_belief(np.array([0, 5])),
            ),
            StepData(
                state=np.array([4, 5]),
                action="right",
                next_state=np.array([5, 5]),
                observation=np.array([5, 5]),
                reward=-12.0,  # Within obstacle radius (obstacle at (5, 5) with radius 1.5)
                belief=create_test_belief(np.array([4, 5])),
            ),
        ],
        discount_factor=0.95,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=2,
        reach_terminal_state=True,
        policy_run_data=[PolicyRunData(info_variables=[])],
    )

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
    assert (
        obstacle_rate.lower_confidence_bound
        <= obstacle_rate.value
        <= obstacle_rate.upper_confidence_bound
    )


def test_continuous_light_dark_pomdp_initialization(base_continuous_light_dark_pomdp):
    env = base_continuous_light_dark_pomdp
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
    assert np.array_equal(env.goal_state, np.array([10, 5]))
    assert np.array_equal(env.start_state, np.array([0, 5]))
    expected_beacons = np.array([[0, 0, 0, 5, 5, 5, 10, 10, 10], [0, 5, 10, 0, 5, 10, 0, 5, 10]])
    assert np.array_equal(env.beacons, expected_beacons)
    expected_obstacles = np.array([[3, 5], [7, 5]])
    assert np.array_equal(env.obstacles, expected_obstacles)


def test_continuous_light_dark_pomdp_state_transition_model(
    base_continuous_light_dark_pomdp,
):
    np.random.seed(42)
    env = base_continuous_light_dark_pomdp
    state = np.array([5, 5])
    action = np.array([0, 1])

    next_state = env.sample_next_state(state, action)
    expected_next_state = state + action
    # Allow for noise in state transition (3 standard deviations)
    assert np.allclose(next_state, expected_next_state, atol=3.0)


def test_continuous_light_dark_pomdp_observation_model(
    base_continuous_light_dark_pomdp,
):
    np.random.seed(42)
    env = base_continuous_light_dark_pomdp
    next_state = np.array([5, 5])
    action = np.array([0, 1])

    observation = env.sample_observation(next_state, action)
    # Allow for noise in observation (3 standard deviations)
    assert np.allclose(observation, next_state, atol=3.0)


def test_continuous_light_dark_pomdp_reward(base_continuous_light_dark_pomdp):
    env = base_continuous_light_dark_pomdp
    # Goal state
    state = np.array([9, 5])
    action = np.array([1, 0])
    reward = env.reward(state, action)
    assert reward > 0
    # Obstacle hit
    state = np.array([2, 5])
    action = np.array([1, 0])
    reward = env.reward(state, action)
    assert reward < 0
    # Normal movement
    state = np.array([1, 1])
    action = np.array([1, 0])
    reward = env.reward(state, action)
    assert reward < 0


def test_continuous_light_dark_pomdp_is_terminal(base_continuous_light_dark_pomdp):
    env = base_continuous_light_dark_pomdp
    # Goal state
    assert env.is_terminal(env.goal_state)
    assert env.is_terminal(env.goal_state + np.array([1, 0]))
    # Obstacle state
    assert env.is_terminal(env.obstacles[:, 0])
    assert env.is_terminal(env.obstacles[:, 0] + np.array([1, 0]))
    # Out of grid (no longer terminal)
    assert not env.is_terminal(np.array([-1, 5]))
    assert not env.is_terminal(np.array([12, 5]))
    # Non-terminal
    assert not env.is_terminal(np.array([1, 1]))


def test_continuous_light_dark_pomdp_compute_metrics(base_continuous_light_dark_pomdp):
    env = base_continuous_light_dark_pomdp

    def create_test_belief(state):
        return WeightedParticleBelief(
            particles=[state], log_weights=np.array([1.0]), resampling=False
        )

    history1 = History(
        [
            StepData(
                state=np.array([0, 5]),
                action=np.array([1, 0]),
                next_state=np.array([1, 5]),
                observation=np.array([1, 5]),
                reward=-2.0,
                belief=create_test_belief(np.array([0, 5])),
            ),
            StepData(
                state=np.array([9, 5]),
                action=np.array([1, 0]),
                next_state=np.array([10, 5]),
                observation=np.array([10, 5]),
                reward=8.0,
                belief=create_test_belief(np.array([9, 5])),
            ),
        ],
        discount_factor=0.95,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=2,
        reach_terminal_state=True,
        policy_run_data=[PolicyRunData(info_variables=[])],
    )
    history2 = History(
        [
            StepData(
                state=np.array([0, 5]),
                action=np.array([1, 0]),
                next_state=np.array([1, 5]),
                observation=np.array([1, 5]),
                reward=-2.0,
                belief=create_test_belief(np.array([0, 5])),
            ),
            StepData(
                state=np.array([4, 5]),
                action=np.array([1, 0]),
                next_state=np.array([5, 5]),
                observation=np.array([5, 5]),
                reward=-12.0,  # Within obstacle radius (obstacle at (5, 5) with radius 1.5)
                belief=create_test_belief(np.array([4, 5])),
            ),
        ],
        discount_factor=0.95,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=2,
        reach_terminal_state=True,
        policy_run_data=[PolicyRunData(info_variables=[])],
    )
    metrics = env.compute_metrics([history1, history2])
    metrics_dict = {metric.name: metric for metric in metrics}
    assert "goal_reaching_rate" in metrics_dict
    goal_rate = metrics_dict["goal_reaching_rate"]
    assert goal_rate.value == 0.5
    assert goal_rate.lower_confidence_bound <= goal_rate.value <= goal_rate.upper_confidence_bound
    assert "obstacle_hit_rate" in metrics_dict
    obstacle_rate = metrics_dict["obstacle_hit_rate"]
    assert obstacle_rate.value == 0.5
    assert (
        obstacle_rate.lower_confidence_bound
        <= obstacle_rate.value
        <= obstacle_rate.upper_confidence_bound
    )


def test_decaying_hit_probability_reward_model():
    """Test that the environment uses the decaying hit probability reward model when specified."""
    env = ContinuousLightDarkPOMDPDiscreteActions(
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
        reward_model_type=RewardModelType.DECAYING_HIT_PROBABILITY,
        penalty_decay=0.5,
    )
    assert isinstance(env.reward_model, ContinuousLightDarkDecayingHitProbabilityRewardModel)


def test_dangerous_states_reward_model():
    """Test that the environment uses the dangerous states reward model when specified."""
    env = ContinuousLightDarkPOMDPDiscreteActions(
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
        reward_model_type=RewardModelType.DANGEROUS_STATES,
    )

    # Test that the correct reward model is initialized
    assert isinstance(env.reward_model, ContinuousLDDangerousStatesRewardModel)

    # Test that the reward model produces both positive and negative rewards near obstacles
    state = np.array([4, 5])  # Near obstacle at (5, 5) - distance 1.0
    action = "right"  # Use string action instead of numpy array

    # Run multiple times to ensure we get both positive and negative rewards
    rewards = [env.reward(state, action) for _ in range(100)]
    positive_rewards = [r for r in rewards if r > 0]
    negative_rewards = [r for r in rewards if r < 0]

    # Check that we get both positive and negative rewards
    assert len(positive_rewards) > 0, "Should get some positive rewards"
    assert len(negative_rewards) > 0, "Should get some negative rewards"
    # Note: We do not check the mean reward, as the environment's reward structure includes other penalties.


def test_single_obstacle_reward_behavior():
    """Test reward behavior with a single obstacle and radius 1.

    Purpose: Validates that ContinuousLightDarkPOMDPDiscreteActions correctly applies obstacle rewards
    based on obstacle radius when states are within or outside the obstacle range

    Given: A ContinuousLightDarkPOMDPDiscreteActions environment with one obstacle at (5,5) and radius 1.0
    When: Reward is calculated for states at the obstacle center, within radius, and outside radius
    Then:
        - States at obstacle center get high obstacle reward (negative)
        - States within obstacle radius get obstacle reward (negative)
        - States outside obstacle radius get normal movement reward (lower than obstacle reward)

    Test type: unit
    """
    # Create environment with single obstacle at (5,5) and radius 1.0
    single_obstacle = [(5.0, 5.0)]

    env = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95,
        state_transition_cov_matrix=np.eye(2),
        observation_cov_matrix=np.eye(2),
        obstacle_hit_probability=1.0,  # Always hit obstacle when in range
        obstacle_reward=-15.0,  # High negative reward for obstacle hits
        goal_reward=10.0,
        fuel_cost=2.0,
        grid_size=11,
        goal_state_radius=1.5,
        beacon_radius=1.0,
        obstacle_radius=1.0,  # Radius of 1.0
        obstacles=single_obstacle,
    )

    # Test state exactly at obstacle center (5, 5)
    obstacle_center_state = np.array([5.0, 5.0])
    action = "up"  # Use "up" action which moves to (5,6) - still within obstacle radius

    # Calculate reward multiple times to account for stochasticity
    obstacle_rewards = []
    for _ in range(100):
        reward = env.reward(obstacle_center_state, action)
        obstacle_rewards.append(reward)

    # The reward should be consistently high (negative) when moving within obstacle radius
    # Base reward: -fuel_cost - distance_to_goal + obstacle_reward
    # Moving from (5,5) to (5,6): -2.0 - ||[5,6] - [10,5]|| + (-15.0) = -2.0 - 5.1 + (-15.0) ≈ -22.1
    expected_obstacle_reward = (
        -2.0 - np.linalg.norm(np.array([5.0, 6.0]) - np.array([10, 5])) + (-15.0)
    )
    assert all(
        abs(r - expected_obstacle_reward) < 0.0001 for r in obstacle_rewards
    ), f"Obstacle center rewards should be around {expected_obstacle_reward}, got {obstacle_rewards[:5]}"

    # Test state outside obstacle radius (7, 5) - distance 2 from center
    outside_radius_state = np.array([7.0, 5.0])
    outside_radius_rewards = []
    for _ in range(100):
        reward = env.reward(outside_radius_state, action)
        outside_radius_rewards.append(reward)

    # Should get normal movement reward (no obstacle penalty)
    expected_outside_radius_reward = -2.0 - np.linalg.norm(np.array([7.0, 6.0]) - np.array([10, 5]))
    assert all(
        abs(r - expected_outside_radius_reward) < 0.1 for r in outside_radius_rewards
    ), f"Outside radius rewards should be around {expected_outside_radius_reward}, got {outside_radius_rewards[:5]}"

    # Verify that obstacle rewards are lower (more negative) than normal movement rewards
    avg_obstacle_reward = np.mean(obstacle_rewards)
    avg_outside_reward = np.mean(outside_radius_rewards)

    assert (
        avg_obstacle_reward < avg_outside_reward
    ), f"Obstacle rewards ({avg_obstacle_reward:.2f}) should be lower than normal rewards ({avg_outside_reward:.2f})"

    # Verify the obstacle is properly positioned (now 2xN format)
    assert env.obstacles.shape == (2, 1), "Should have 2 coordinates for 1 obstacle"
    assert np.array_equal(env.obstacles[:, 0], [5, 5]), "Obstacle should be at (5, 5)"
    assert env.obstacle_radius == 1.0, "Obstacle radius should be 1.0"

    # Print debug information
    print(f"Obstacle center rewards (first 5): {obstacle_rewards[:5]}")
    print(f"Outside radius rewards (first 5): {outside_radius_rewards[:5]}")
    print(f"Average obstacle reward: {avg_obstacle_reward:.2f}")
    print(f"Average outside reward: {avg_outside_reward:.2f}")
    print(f"Expected obstacle reward: {expected_obstacle_reward:.2f}")
    print(f"Expected outside reward: {expected_outside_radius_reward:.2f}")


class TestVisualizePath:
    """Test suite for visualize_path function."""

    def test_visualize_path_creates_gif_file(self, base_light_dark_environment, tmp_path):
        """Test that visualize_path creates a GIF file at the specified cache path.

        Purpose: Validates that visualize_path successfully creates a GIF animation file

        Given: A light-dark environment, simple agent path, belief path, actions, and valid cache path
        When: visualize_path is called with the test data
        Then: A GIF file is created at the specified cache path and has non-zero size

        Test type: unit
        """
        env = base_light_dark_environment

        # Create simple test path
        path = [
            np.array([0, 5]),  # Start state
            np.array([1, 5]),  # Move right
            np.array([2, 5]),  # Move right again
            np.array([3, 5]),  # Final position
        ]

        # Create simple belief distributions
        belief_path = [
            DiscreteDistribution(
                values=[np.array([0, 5]), np.array([0, 4])], probs=np.array([0.8, 0.2])
            ),
            DiscreteDistribution(
                values=[np.array([1, 5]), np.array([1, 4])], probs=np.array([0.9, 0.1])
            ),
            DiscreteDistribution(
                values=[np.array([2, 5]), np.array([2, 4])],
                probs=np.array([0.95, 0.05]),
            ),
            DiscreteDistribution(values=[np.array([3, 5])], probs=np.array([1.0])),
        ]

        # Create actions
        actions = ["right", "right", "right"]

        # Create cache path
        cache_path = tmp_path / "test_visualization.gif"

        # Call visualize_path
        env.visualize_path(path, belief_path, actions, cache_path)

        # Verify file was created
        assert cache_path.exists(), "GIF file should be created"
        assert cache_path.stat().st_size > 0, "GIF file should have non-zero size"

        # Verify it's a valid GIF file by checking magic bytes
        with open(cache_path, "rb") as f:
            magic = f.read(6)
            assert magic in [
                b"GIF87a",
                b"GIF89a",
            ], "File should have valid GIF magic bytes"

    def test_visualize_path_with_invalid_cache_path_type(self, base_light_dark_environment):
        """Test that visualize_path raises TypeError for invalid cache_path type.

        Purpose: Validates proper error handling when cache_path is not a Path object

        Given: A light-dark environment and cache_path as string instead of Path object
        When: visualize_path is called with invalid cache_path type
        Then: TypeError is raised with appropriate error message

        Test type: unit
        """
        env = base_light_dark_environment

        path = [np.array([0, 5])]
        belief_path = [DiscreteDistribution(values=[np.array([0, 5])], probs=np.array([1.0]))]
        actions = []

        # Test with string instead of Path
        with pytest.raises(TypeError, match="cache_path must be a Path object"):
            env.visualize_path(path, belief_path, actions, "not_a_path.gif")

    def test_visualize_path_with_invalid_cache_path_extension(
        self, base_light_dark_environment, tmp_path
    ):
        """Test that visualize_path raises ValueError for non-GIF cache path.

        Purpose: Validates proper error handling when cache_path doesn't end with .gif

        Given: A light-dark environment and cache_path without .gif extension
        When: visualize_path is called with invalid file extension
        Then: ValueError is raised with appropriate error message

        Test type: unit
        """
        env = base_light_dark_environment

        path = [np.array([0, 5])]
        belief_path = [DiscreteDistribution(values=[np.array([0, 5])], probs=np.array([1.0]))]
        actions = []

        # Test with .png extension
        cache_path = tmp_path / "test_visualization.png"
        with pytest.raises(ValueError, match="cache_path must end with .gif"):
            env.visualize_path(path, belief_path, actions, cache_path)

    def test_visualize_path_with_empty_path(self, base_light_dark_environment, tmp_path):
        """Test that visualize_path handles empty path gracefully.

        Purpose: Validates robustness when provided with empty input data

        Given: A light-dark environment with empty path, belief_path, and actions
        When: visualize_path is called with empty data
        Then: IndexError is raised due to matplotlib limitation with empty frames

        Test type: unit
        """
        env = base_light_dark_environment

        path = []
        belief_path = []
        actions = []
        cache_path = tmp_path / "empty_visualization.gif"

        # Empty path causes matplotlib to fail with IndexError
        with pytest.raises(IndexError, match="list index out of range"):
            env.visualize_path(path, belief_path, actions, cache_path)

    def test_visualize_path_with_complex_belief_distributions(
        self, base_light_dark_environment, tmp_path
    ):
        """Test visualize_path with complex belief distributions containing multiple particles.

        Purpose: Validates visualization handles complex belief states with many particles

        Given: A light-dark environment with path and belief distributions having many particles
        When: visualize_path is called with complex belief data
        Then: GIF file is created successfully and belief particles are properly visualized

        Test type: unit
        """
        env = base_light_dark_environment

        # Create path with more complex movement
        path = [
            np.array([0, 5]),
            np.array([1, 4]),
            np.array([2, 4]),
            np.array([3, 5]),
            np.array([4, 5]),
        ]

        # Create complex belief distributions with multiple particles
        belief_path = [
            DiscreteDistribution(
                values=[
                    np.array([0, 5]),
                    np.array([0, 4]),
                    np.array([1, 5]),
                    np.array([0, 6]),
                ],
                probs=np.array([0.4, 0.3, 0.2, 0.1]),
            ),
            DiscreteDistribution(
                values=[
                    np.array([1, 4]),
                    np.array([1, 3]),
                    np.array([2, 4]),
                    np.array([1, 5]),
                ],
                probs=np.array([0.5, 0.2, 0.2, 0.1]),
            ),
            DiscreteDistribution(
                values=[np.array([2, 4]), np.array([2, 3]), np.array([3, 4])],
                probs=np.array([0.6, 0.25, 0.15]),
            ),
            DiscreteDistribution(
                values=[np.array([3, 5]), np.array([3, 4])], probs=np.array([0.8, 0.2])
            ),
            DiscreteDistribution(values=[np.array([4, 5])], probs=np.array([1.0])),
        ]

        actions = ["down", "right", "up", "right"]
        cache_path = tmp_path / "complex_visualization.gif"

        env.visualize_path(path, belief_path, actions, cache_path)

        # Verify successful creation
        assert cache_path.exists()
        assert cache_path.stat().st_size > 0

    def test_visualize_path_with_continuous_actions(
        self, base_continuous_light_dark_pomdp, tmp_path
    ):
        """Test visualize_path with continuous actions (numpy arrays).

        Purpose: Validates visualization works with continuous action spaces

        Given: A continuous light-dark environment with numpy array actions
        When: visualize_path is called with continuous actions
        Then: GIF file is created successfully with proper action arrows

        Test type: unit
        """
        env = base_continuous_light_dark_pomdp

        path = [np.array([0, 5]), np.array([1.5, 4.5]), np.array([3.2, 4.8])]

        belief_path = [
            DiscreteDistribution(values=[np.array([0, 5])], probs=np.array([1.0])),
            DiscreteDistribution(values=[np.array([1.5, 4.5])], probs=np.array([1.0])),
            DiscreteDistribution(values=[np.array([3.2, 4.8])], probs=np.array([1.0])),
        ]

        # Use continuous actions (numpy arrays)
        actions = [np.array([1.5, -0.5]), np.array([1.7, 0.3])]
        cache_path = tmp_path / "continuous_actions_visualization.gif"

        env.visualize_path(path, belief_path, actions, cache_path)

        assert cache_path.exists()
        assert cache_path.stat().st_size > 0

    def test_visualize_path_caching_behavior(self, base_light_dark_environment, tmp_path):
        """Test that visualize_path caching works correctly by overwriting existing files.

        Purpose: Validates that visualization caching works properly by overwriting files

        Given: A light-dark environment and an existing GIF file at the cache path
        When: visualize_path is called multiple times with the same cache path
        Then: File is successfully overwritten each time and contains different content

        Test type: unit
        """
        env = base_light_dark_environment
        cache_path = tmp_path / "cached_visualization.gif"

        # First visualization with simple path
        path1 = [np.array([0, 5]), np.array([1, 5])]
        belief_path1 = [
            DiscreteDistribution(values=[np.array([0, 5])], probs=np.array([1.0])),
            DiscreteDistribution(values=[np.array([1, 5])], probs=np.array([1.0])),
        ]
        actions1 = ["right"]

        env.visualize_path(path1, belief_path1, actions1, cache_path)

        # Verify first file exists
        assert cache_path.exists()
        first_size = cache_path.stat().st_size
        first_mtime = cache_path.stat().st_mtime

        # Small delay to ensure different modification time

        time.sleep(0.1)

        # Second visualization with different path (should overwrite)
        path2 = [
            np.array([0, 5]),
            np.array([1, 5]),
            np.array([2, 5]),
            np.array([3, 5]),
            np.array([4, 5]),
        ]
        belief_path2 = [
            DiscreteDistribution(values=[np.array([i, 5])], probs=np.array([1.0])) for i in range(5)
        ]
        actions2 = ["right", "right", "right", "right"]

        env.visualize_path(path2, belief_path2, actions2, cache_path)

        # Verify file was updated
        assert cache_path.exists()
        second_size = cache_path.stat().st_size
        second_mtime = cache_path.stat().st_mtime

        # File should have been modified (different size or modification time)
        assert (
            second_mtime > first_mtime or second_size != first_size
        ), "Cache file should be updated with new visualization"

    def test_visualize_path_cache_directory_creation(self, base_light_dark_environment, tmp_path):
        """Test that visualize_path requires parent directories to exist.

        Purpose: Validates that visualization requires existing parent directories

        Given: A light-dark environment and cache path with non-existent parent directories
        When: visualize_path is called with nested cache path
        Then: FileNotFoundError is raised due to missing parent directories

        Test type: unit
        """
        env = base_light_dark_environment

        # Create nested path that doesn't exist
        nested_cache_path = tmp_path / "visualizations" / "experiments" / "test_run.gif"

        # Ensure parent directories don't exist initially
        assert not nested_cache_path.parent.exists()

        path = [np.array([0, 5]), np.array([1, 5])]
        belief_path = [
            DiscreteDistribution(values=[np.array([0, 5])], probs=np.array([1.0])),
            DiscreteDistribution(values=[np.array([1, 5])], probs=np.array([1.0])),
        ]
        actions = ["right"]

        # Should raise FileNotFoundError due to missing parent directories
        with pytest.raises(FileNotFoundError):
            env.visualize_path(path, belief_path, actions, nested_cache_path)

        # Now create parent directories and try again
        nested_cache_path.parent.mkdir(parents=True)
        env.visualize_path(path, belief_path, actions, nested_cache_path)

        # Verify file was created successfully
        assert (
            nested_cache_path.exists()
        ), "GIF file should be created when parent directories exist"
        assert nested_cache_path.stat().st_size > 0

    def test_visualize_path_with_mismatched_lengths(self, base_light_dark_environment, tmp_path):
        """Test visualize_path behavior with mismatched path, belief_path, and actions lengths.

        Purpose: Validates robustness when input arrays have different lengths

        Given: A light-dark environment with path, belief_path, and actions of different lengths
        When: visualize_path is called with mismatched array lengths
        Then: Function completes without error, handling missing data gracefully

        Test type: unit
        """
        env = base_light_dark_environment

        # Intentionally create mismatched lengths
        path = [
            np.array([0, 5]),
            np.array([1, 5]),
            np.array([2, 5]),
            np.array([3, 5]),
        ]  # 4 states
        belief_path = [
            DiscreteDistribution(values=[np.array([0, 5])], probs=np.array([1.0])),
            DiscreteDistribution(
                values=[np.array([1, 5])], probs=np.array([1.0])
            ),  # Only 2 beliefs
        ]
        actions = ["right"]  # Only 1 action

        cache_path = tmp_path / "mismatched_lengths.gif"

        # Should handle gracefully without throwing exceptions
        env.visualize_path(path, belief_path, actions, cache_path)

        assert cache_path.exists()
        assert cache_path.stat().st_size > 0

    def test_visualize_path_belief_particles_with_zero_probabilities(
        self, base_light_dark_environment, tmp_path
    ):
        """Test visualize_path with belief distributions containing zero probabilities.

        Purpose: Validates handling of belief particles with zero or very small probabilities

        Given: A light-dark environment with belief distributions containing zero probabilities
        When: visualize_path is called with zero-probability particles
        Then: Visualization handles zero probabilities correctly without errors

        Test type: unit
        """
        env = base_light_dark_environment

        path = [np.array([0, 5]), np.array([1, 5])]

        # Create belief with zero probabilities
        belief_path = [
            DiscreteDistribution(
                values=[np.array([0, 5]), np.array([0, 4]), np.array([1, 5])],
                probs=np.array([0.8, 0.0, 0.2]),  # Middle particle has zero probability
            ),
            DiscreteDistribution(
                values=[np.array([1, 5]), np.array([1, 4])],
                probs=np.array([1.0, 0.0]),  # Second particle has zero probability
            ),
        ]

        actions = ["right"]
        cache_path = tmp_path / "zero_prob_particles.gif"

        env.visualize_path(path, belief_path, actions, cache_path)

        assert cache_path.exists()
        assert cache_path.stat().st_size > 0

    def test_visualize_path_preserves_existing_cache_structure(
        self, base_light_dark_environment, tmp_path
    ):
        """Test that visualize_path preserves existing cache directory structure.

        Purpose: Validates that caching doesn't interfere with existing directory structure

        Given: A light-dark environment and existing cache directory with other files
        When: visualize_path is called to save in existing directory
        Then: New GIF is saved without affecting other files in the cache directory

        Test type: integration
        """
        env = base_light_dark_environment

        # Create existing cache structure
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create some existing files
        existing_file1 = cache_dir / "existing1.txt"
        existing_file2 = cache_dir / "existing2.json"
        existing_file1.write_text("existing content 1")
        existing_file2.write_text('{"existing": "content"}')

        # Store original content and timestamps
        original_content1 = existing_file1.read_text()
        original_content2 = existing_file2.read_text()
        original_mtime1 = existing_file1.stat().st_mtime
        original_mtime2 = existing_file2.stat().st_mtime

        # Create visualization in same directory
        path = [np.array([0, 5]), np.array([1, 5])]
        belief_path = [
            DiscreteDistribution(values=[np.array([0, 5])], probs=np.array([1.0])),
            DiscreteDistribution(values=[np.array([1, 5])], probs=np.array([1.0])),
        ]
        actions = ["right"]
        cache_path = cache_dir / "new_visualization.gif"

        env.visualize_path(path, belief_path, actions, cache_path)

        # Verify new file was created
        assert cache_path.exists()
        assert cache_path.stat().st_size > 0

        # Verify existing files were not modified
        assert existing_file1.read_text() == original_content1
        assert existing_file2.read_text() == original_content2
        assert existing_file1.stat().st_mtime == original_mtime1
        assert existing_file2.stat().st_mtime == original_mtime2


def test_beacon_proximity_observation_covariance_changes():
    """Test that observation log-probability density rises near a beacon.

    Purpose: Validates that the env's observation log-prob uses the
    reduced near-beacon covariance when the next-state is within
    beacon_radius.

    Given: A continuous light-dark env with beacons at (0,0), (5,5), (10,10)
        and beacon_radius=1.0.
    When: env.observation_log_probability(next_state, action, [next_state]) is
        called for a near-beacon next_state and a far-from-beacon next_state.
    Then: env.is_state_near_beacon agrees, and the near-beacon log-prob (at
        the mean) is strictly greater than the far-from-beacon log-prob (at
        the mean) thanks to the reduced covariance.

    Test type: unit
    """
    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        observation_cov_matrix=np.eye(2) * 4.0,
        beacons=[(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)],
        beacon_radius=1.0,
    )
    action = np.array([0.0, 0.0])

    near_state = np.array([0.5, 0.5])  # Distance sqrt(0.5) < 1.0 from (0,0)
    far_state = np.array([2.5, 2.5])  # Distance > 1.0 from all beacons

    assert env.is_state_near_beacon(near_state) is True
    assert env.is_state_near_beacon(far_state) is False

    # Density at the mean should be higher for the near-beacon distribution
    # (covariance is half).
    near_logp = env.observation_log_probability(near_state, action, [near_state])[0]
    far_logp = env.observation_log_probability(far_state, action, [far_state])[0]
    assert near_logp > far_logp

    # Boundary: exactly beacon_radius IS near
    # (any_point_within_radius_kernel uses ``<= radius_sq``).
    boundary_state = np.array([1.0, 0.0])
    assert env.is_state_near_beacon(boundary_state) is True


def test_beacon_proximity_with_multiple_beacons():
    """Test beacon proximity detection with multiple beacons.

    Purpose: Validates that proximity detection works correctly with multiple beacons

    Given: A continuous light-dark env with beacons at (0,0), (5,5), (10,10) and
        beacon_radius=1.5.
    When: env.is_state_near_beacon is queried for states close to each beacon
        and a state equidistant between two beacons but outside both radii.
    Then: Proximity is detected correctly per beacon and rejected for the
        equidistant point that is outside any beacon's radius.

    Test type: unit
    """
    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        observation_cov_matrix=np.eye(2) * 2.0,
        beacons=[(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)],
        beacon_radius=1.5,
    )

    near_first_beacon = np.array([1.0, 1.0])  # Distance sqrt(2) ≈ 1.41 < 1.5 from (0,0)
    near_middle_beacon = np.array([5.0, 6.0])  # Distance 1.0 < 1.5 from (5,5)
    near_last_beacon = np.array([9.0, 10.0])  # Distance 1.0 < 1.5 from (10,10)
    equidistant_state = np.array([2.5, 2.5])  # Far from all beacons

    assert env.is_state_near_beacon(near_first_beacon) is True
    assert env.is_state_near_beacon(near_middle_beacon) is True
    assert env.is_state_near_beacon(near_last_beacon) is True
    assert env.is_state_near_beacon(equidistant_state) is False


def test_observation_model_probability_function():
    """Test the env's observation log-probability for NORMAL_NOISE.

    Purpose: Validates that the env's observation log-probability function
    matches the expected multivariate-normal log-pdf values when far from
    all beacons (the far-beacon distribution applies).

    Given: A ContinuousLightDarkPOMDP env with observation_cov_matrix=0.25*I
        and beacons placed far from the next_state used for evaluation.
    When: env.observation_log_probability is computed for observation values
        equal to and offset from the next_state mean.
    Then: log-probabilities match np.log(multivariate_normal.pdf(...))
        within 1e-10 relative tolerance and density decreases with distance.

    Test type: unit
    """
    next_state = np.array([5.0, 3.0])
    action = np.array([0.0, 0.0])
    observation_cov_matrix = np.eye(2) * 0.25
    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        observation_cov_matrix=observation_cov_matrix,
        beacons=[(0.0, 0.0), (10.0, 10.0)],
        beacon_radius=1.0,
    )

    # Single value at the mean.
    log_p = env.observation_log_probability(next_state, action, [next_state])
    expected_log = float(
        np.log(
            multivariate_normal.pdf(
                next_state, mean=next_state, cov=observation_cov_matrix  # type: ignore
            )
        )
    )
    assert isinstance(log_p, np.ndarray)
    assert len(log_p) == 1
    assert np.isclose(log_p[0], expected_log, rtol=1e-10)

    # Multiple values.
    observation_values = [
        np.array([5.0, 3.0]),
        np.array([5.5, 3.5]),
        np.array([4.0, 2.0]),
    ]
    log_p_multi = env.observation_log_probability(next_state, action, observation_values)
    expected_log_multi = np.log(
        multivariate_normal.pdf(
            observation_values, mean=next_state, cov=observation_cov_matrix  # type: ignore
        )
    )
    assert len(log_p_multi) == 3
    assert np.allclose(log_p_multi, expected_log_multi, rtol=1e-10)

    # Density decreases with distance from mean.
    close_obs = np.array([5.1, 3.1])
    far_obs = np.array([6.0, 4.0])
    close_log = env.observation_log_probability(next_state, action, [close_obs])[0]
    far_log = env.observation_log_probability(next_state, action, [far_obs])[0]
    assert close_log > far_log


def test_observation_model_probability_with_beacon_proximity():
    """Test that the env's near-beacon log-prob density exceeds the far-beacon.

    Purpose: Validates that the near-beacon distribution (lower covariance)
    yields a strictly higher log-pdf at the mean than the far-beacon
    distribution.

    Given: A ContinuousLightDarkPOMDP env with two beacons and beacon_radius=1.0.
    When: env.observation_log_probability is evaluated at the mean for a
        near-beacon next_state and a far-from-beacon next_state.
    Then: The near-beacon log-prob is strictly greater (lower covariance =>
        higher peak density at the mean).

    Test type: unit
    """
    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        observation_cov_matrix=np.eye(2),
        beacons=[(0.0, 0.0), (5.0, 5.0)],
        beacon_radius=1.0,
    )
    action = np.array([0.0, 0.0])
    near_state = np.array([0.5, 0.5])
    far_state = np.array([7.0, 7.0])

    near_log = env.observation_log_probability(near_state, action, [near_state])[0]
    far_log = env.observation_log_probability(far_state, action, [far_state])[0]
    assert near_log > far_log


def test_observation_model_probability_edge_cases():
    """Test the env observation log-prob with edge case inputs.

    Purpose: Validates robust handling of edge cases (single observation,
    grid boundaries, very small covariance) via the env API.

    Given: A ContinuousLightDarkPOMDP env and a far-from-beacon next_state.
    When: env.observation_log_probability is evaluated on single, boundary,
        and very-small-covariance inputs.
    Then: All return finite log-probabilities, with density at the mean
        much higher than slightly off the mean for very small covariance.

    Test type: unit
    """
    next_state = np.array([5.0, 5.0])
    action = np.array([0.0, 0.0])
    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        observation_cov_matrix=np.eye(2) * 0.1,
        beacons=[(2.0, 2.0), (8.0, 8.0)],
        beacon_radius=1.0,
    )

    # Single observation at the mean.
    single_log = env.observation_log_probability(next_state, action, [next_state])
    assert isinstance(single_log, np.ndarray)
    assert len(single_log) == 1
    assert np.isfinite(single_log[0])

    # Observations at grid boundaries — all return finite log-probs.
    boundary_observations = [
        np.array([0.0, 0.0]),
        np.array([11.0, 11.0]),
        np.array([0.0, 11.0]),
        np.array([11.0, 0.0]),
    ]
    boundary_log = env.observation_log_probability(next_state, action, boundary_observations)
    assert len(boundary_log) == 4
    assert all(np.isfinite(boundary_log))

    # Very small covariance — density at the mean is much higher than off.
    small_env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        observation_cov_matrix=np.eye(2) * 1e-6,
        beacons=[(2.0, 2.0), (8.0, 8.0)],
        beacon_radius=1.0,
    )
    exact_log = small_env.observation_log_probability(next_state, action, [next_state])[0]
    offset_log = small_env.observation_log_probability(
        next_state, action, [np.array([5.01, 5.01])]
    )[0]
    assert exact_log > offset_log


def test_start_state_not_in_obstacle_radius():
    """Test that start state is not within obstacle radius causing immediate termination.

    Purpose: Validates that environment configurations don't place start state within obstacle radius

    Given: A continuous light-dark POMDP environment with obstacles and start state
    When: Environment is initialized with default or configured parameters
    Then: Start state is not within obstacle radius of any obstacle (not terminal at start)

    Test type: unit
    """
    # Test default environment configuration
    env = ContinuousLightDarkPOMDPDiscreteActions(
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

    # Check that start state is not terminal (not in obstacle radius)
    assert not env.is_terminal(
        env.start_state
    ), f"Start state {env.start_state} should not be terminal (within obstacle radius)"

    # Explicitly check distance to each obstacle
    start_state = env.start_state
    for i in range(env.obstacles.shape[1]):
        obstacle_pos = env.obstacles[:, i]
        distance = np.linalg.norm(start_state - obstacle_pos)
        assert distance > env.obstacle_radius, (
            f"Start state {start_state} is too close to obstacle {i} at {obstacle_pos} "
            f"(distance {distance:.2f} <= radius {env.obstacle_radius})"
        )


def test_risk_averse_environment_config_start_state_validity():
    """Test that RiskAverseEnvironmentConfigsAPI doesn't create invalid start states.

    Purpose: Validates that environment configuration APIs create valid non-terminal start states

    Given: A RiskAverseEnvironmentConfigsAPI configuration
    When: Continuous light-dark POMDP environment is created
    Then: Start state is not within obstacle radius and episode can proceed

    Test type: integration
    """
    # Test the configuration that was causing the issue in visualization example
    env_configs = RiskAverseEnvironmentConfigsAPI(discount_factor=0.95)
    (
        light_dark_env,
        _initial_belief,  # pylint: disable=unused-variable
    ) = env_configs.continuous_observations_discrete_actions_light_dark_pomdp_config(n_particles=50)

    # Narrow to the concrete type so attribute access type-checks.
    assert isinstance(light_dark_env, ContinuousLightDarkPOMDPDiscreteActions)
    light_dark_env_typed = light_dark_env

    # Check that start state is not terminal
    assert not light_dark_env_typed.is_terminal(
        light_dark_env_typed.start_state
    ), f"RiskAverseEnvironmentConfigsAPI created terminal start state {light_dark_env_typed.start_state}"

    # Explicitly check distance to each obstacle
    start_state = light_dark_env_typed.start_state
    for i in range(light_dark_env_typed.obstacles.shape[1]):
        obstacle_pos = light_dark_env_typed.obstacles[:, i]
        distance = np.linalg.norm(start_state - obstacle_pos)
        assert distance > light_dark_env_typed.obstacle_radius, (
            f"RiskAverseEnvironmentConfigsAPI: Start state {start_state} too close to obstacle {i} "
            f"at {obstacle_pos} (distance {distance:.2f} <= radius {light_dark_env_typed.obstacle_radius})"
        )

    # The main test has passed - start state is not terminal and obstacles are far enough away
    # This is sufficient to catch the original issue where start state was within obstacle radius
    print("✓ RiskAverseEnvironmentConfigsAPI configuration is valid - start state not terminal")


def test_environment_configuration_obstacle_placement():
    """Test that environment configurations have reasonable obstacle placement.

    Purpose: Validates that obstacles don't block all paths or create impossible scenarios

    Given: Various continuous light-dark POMDP environment configurations
    When: Environment is initialized with different parameters
    Then: Start and goal states are accessible and obstacles don't create impossible scenarios

    Test type: unit
    """
    # Test with different grid sizes using appropriate beacon and obstacle placements
    test_configs = [
        {
            "grid_size": 5,
            "start_state": np.array([0, 2]),
            "goal_state": np.array([4, 2]),
            "beacons": [(0, 0), (4, 0), (0, 4), (4, 4)],
            "obstacles": [(2, 1), (2, 3)],
        },
        {
            "grid_size": 11,
            "start_state": np.array([0, 5]),
            "goal_state": np.array([10, 5]),
            "beacons": [(0, 0), (0, 10), (10, 0), (10, 10)],
            "obstacles": [(3, 5), (7, 5)],
        },
        {
            "grid_size": 15,
            "start_state": np.array([0, 7]),
            "goal_state": np.array([14, 7]),
            "beacons": [(0, 0), (0, 14), (14, 0), (14, 14)],
            "obstacles": [(5, 7), (10, 7)],
        },
    ]

    for config in test_configs:
        env = ContinuousLightDarkPOMDPDiscreteActions(
            discount_factor=0.95,
            state_transition_cov_matrix=np.eye(2),
            observation_cov_matrix=np.eye(2),
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=config["grid_size"],
            goal_state_radius=1.5,
            beacon_radius=1.0,
            obstacle_radius=1.5,
            start_state=config["start_state"],
            goal_state=config["goal_state"],
            beacons=config["beacons"],
            obstacles=config["obstacles"],
        )

        # Basic sanity checks
        assert not env.is_terminal(
            env.start_state
        ), f"Start state terminal for grid size {config['grid_size']}"

        # Goal should not be terminal due to obstacles (only due to reaching goal)
        goal_state = env.goal_state
        if env.is_terminal(goal_state):
            # If goal is terminal, it should be because we're in the goal radius, not obstacle radius
            goal_distance = np.linalg.norm(goal_state - goal_state)  # Should be 0
            assert (
                goal_distance <= env.goal_state_radius
            ), f"Goal state terminal for wrong reason in grid size {config['grid_size']}"

        # Start and goal should be sufficiently far apart to make the problem meaningful
        start_goal_distance = np.linalg.norm(env.start_state - env.goal_state)
        assert (
            start_goal_distance > env.goal_state_radius
        ), f"Start and goal too close for grid size {config['grid_size']}"


def test_continuous_light_dark_sample_next_state_shapes_and_distribution():
    """Test that env.sample_next_state returns the right shapes and is approximately centered.

    Purpose: Validates the env-API state-transition sampling produces a
    (2,) array for n_samples=1 and a (N, 2) array for n_samples>1, with
    samples approximately centred around state+action.

    Given: A ContinuousLightDarkPOMDP env with small state-transition covariance
        and a (state, action) pair.
    When: env.sample_next_state is called with n_samples in {1, 5, 100}.
    Then: Output shapes are (2,) / (5, 2) / (100, 2); empirical mean is
        within 0.5 of state+action; samples are not all identical.

    Test type: unit
    """
    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        state_transition_cov_matrix=np.eye(2) * 0.1,
    )
    state = np.array([2.0, 3.0])
    action = np.array([1.0, -0.5])

    single = env.sample_next_state(state, action, n_samples=1)
    assert isinstance(single, np.ndarray)
    assert single.shape == (2,)

    five = env.sample_next_state(state, action, n_samples=5)
    assert isinstance(five, np.ndarray)
    assert five.shape == (5, 2)

    many = env.sample_next_state(state, action, n_samples=100)
    assert many.shape == (100, 2)

    # Empirical mean is close to state+action.
    expected_mean = state + action
    actual_mean = many.mean(axis=0)
    assert np.allclose(actual_mean, expected_mean, atol=0.5)

    # Samples are not all identical.
    unique = {tuple(row) for row in many}
    assert len(unique) > 1


def test_normal_noise_observation_model():
    """Test the env's NORMAL_NOISE observation sampling always returns ndarray observations.

    Purpose: Validates that NORMAL_NOISE observations are sampled via
    env.sample_observation and never produce a "None" sentinel.

    Given: A ContinuousLightDarkPOMDPDiscreteActions env configured with
        ObservationModelType.NORMAL_NOISE.
    When: env.sample_observation is called for several samples.
    Then: All returned observations are ndarrays of shape (2,).

    Test type: unit
    """
    env = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95,
        observation_model_type=ObservationModelType.NORMAL_NOISE,
    )
    assert env.observation_model_type == ObservationModelType.NORMAL_NOISE

    state = np.array([5.0, 5.0])
    action = "up"
    observations = env.sample_observation(state, action, n_samples=10)
    assert isinstance(observations, np.ndarray)
    assert observations.shape == (10, 2)


def test_normal_noise_no_obs_in_dark_observation_model():
    """Test the env's NORMAL_NOISE_NO_OBS_IN_DARK sampling near vs far beacon.

    Purpose: Validates that the NORMAL_NOISE_NO_OBS_IN_DARK env path returns
    real observations only when the next_state is within beacon_radius and
    returns "None" sentinels otherwise.

    Given: A ContinuousLightDarkPOMDPDiscreteActions env with
        ObservationModelType.NORMAL_NOISE_NO_OBS_IN_DARK.
    When: env.sample_observation is called for a near-beacon and far-beacon
        next_state.
    Then: Near-beacon returns ndarray observations; far-beacon returns
        ["None"] * n_samples.

    Test type: unit
    """
    env = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95,
        observation_model_type=ObservationModelType.NORMAL_NOISE_NO_OBS_IN_DARK,
    )
    assert env.observation_model_type == ObservationModelType.NORMAL_NOISE_NO_OBS_IN_DARK
    action = "up"

    # Near beacon — real observations.
    state_near = np.array([0.5, 0.5])  # Within beacon_radius=1.0 of (0,0)
    observations_near = env.sample_observation(state_near, action, n_samples=10)
    assert all(isinstance(obs, np.ndarray) for obs in observations_near)

    # Far from beacon — "None" sentinels.
    state_far = np.array([3.0, 3.0])
    observations_far = env.sample_observation(state_far, action, n_samples=10)
    assert all(obs == "None" for obs in observations_far)


def test_default_observation_model_type():
    """Test that the default observation model type is NORMAL_NOISE.

    Purpose: Validates backward-compatibility default observation_model_type.

    Given: A ContinuousLightDarkPOMDPDiscreteActions env with no explicit
        observation_model_type.
    When: env.observation_model_type is read and env.sample_observation is
        called.
    Then: Default type is NORMAL_NOISE and observations are ndarrays.

    Test type: unit
    """
    env = ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)
    assert env.observation_model_type == ObservationModelType.NORMAL_NOISE

    obs = env.sample_observation(np.array([5.0, 5.0]), "up", n_samples=1)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)


def test_observation_model_type_equality():
    """Test that environments with different observation model types are not equal."""
    env1 = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95,
        observation_model_type=ObservationModelType.NORMAL_NOISE,
    )
    env2 = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95,
        observation_model_type=ObservationModelType.NORMAL_NOISE_NO_OBS_IN_DARK,
    )
    assert env1 != env2, "Environments with different observation model types should not be equal"

    # Test distance-based vs others
    env3 = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95,
        observation_model_type=ObservationModelType.DISTANCE_BASED,
    )
    assert env1 != env3, "Distance-based should not equal normal noise"
    assert env2 != env3, "Distance-based should not equal no obs in dark"


def test_continuous_light_dark_pomdp_observation_model_type():
    """Test NO_OBS_IN_DARK observation type on ContinuousLightDarkPOMDP (continuous actions).

    Purpose: Validates that the continuous-action variant of the env honours
    NORMAL_NOISE_NO_OBS_IN_DARK and returns "None" when far from beacons.

    Given: A ContinuousLightDarkPOMDP env with NORMAL_NOISE_NO_OBS_IN_DARK
        and a far-from-beacon next_state.
    When: env.sample_observation(next_state, action, n_samples=5) is called.
    Then: All five observations are the "None" sentinel.

    Test type: unit
    """
    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        observation_model_type=ObservationModelType.NORMAL_NOISE_NO_OBS_IN_DARK,
    )
    assert env.observation_model_type == ObservationModelType.NORMAL_NOISE_NO_OBS_IN_DARK

    state = np.array([3.0, 3.0])  # far from beacons
    action = np.array([0.0, 0.0])
    observations = env.sample_observation(state, action, n_samples=5)
    assert all(obs == "None" for obs in observations)


def test_distance_based_observation_model():
    """Test the env's DISTANCE_BASED sampling near, far, and at the radius boundary.

    Purpose: Validates that DISTANCE_BASED returns real observations when
    min_distance_to_beacon <= beacon_radius and "None" sentinels otherwise
    (the predicate is ``min_distance > beacon_radius`` for the "None"
    branch, so exactly at the radius still produces real observations).

    Given: A ContinuousLightDarkPOMDPDiscreteActions env with
        ObservationModelType.DISTANCE_BASED and beacon_radius=1.0.
    When: env.sample_observation is called for a near-beacon, far-beacon,
        and exactly-at-beacon-radius next_state.
    Then: Near and at-radius cases return real ndarray observations; the
        far case returns "None" sentinels. ``env.is_state_near_beacon``
        agrees on near vs far.

    Test type: unit
    """
    env = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95,
        observation_model_type=ObservationModelType.DISTANCE_BASED,
    )
    assert env.observation_model_type == ObservationModelType.DISTANCE_BASED
    action = "up"

    # Near beacon — strict-less-than predicate for is_state_near_beacon.
    state_near = np.array([0.5, 0.0])  # 0.5 from beacon (0,0)
    observations_near = env.sample_observation(state_near, action, n_samples=10)
    assert all(isinstance(obs, np.ndarray) for obs in observations_near)
    assert env.is_state_near_beacon(state_near) is True

    # Far from any beacon.
    state_far = np.array([3.0, 3.0])
    observations_far = env.sample_observation(state_far, action, n_samples=10)
    assert all(obs == "None" for obs in observations_far)
    assert env.is_state_near_beacon(state_far) is False

    # Exactly at beacon_radius: predicate min_distance > beacon_radius is
    # False so observations are still returned (the "None" branch is gated
    # by strict-greater-than).
    state_at_radius = np.array([1.0, 0.0])
    observations_at_radius = env.sample_observation(state_at_radius, action, n_samples=10)
    assert all(isinstance(obs, np.ndarray) for obs in observations_at_radius)


def test_distance_based_observation_model_binary_selection():
    """Test DISTANCE_BASED partitions states by beacon_radius.

    Purpose: Validates that env.sample_observation honours the binary
    ``min_distance > beacon_radius`` split on the DISTANCE_BASED model
    across a range of distances from a single beacon.

    Given: A ContinuousLightDarkPOMDPDiscreteActions env with one beacon at
        the origin and beacon_radius=2.0.
    When: env.sample_observation is called for states placed at distances
        within and beyond beacon_radius along the x-axis.
    Then: All near distances yield real ndarray observations; all far
        distances yield "None" sentinels.

    Test type: unit
    """
    env = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95,
        observation_cov_matrix=np.eye(2) * 2.0,
        beacon_radius=2.0,
        beacons=[(0.0, 0.0)],
        observation_model_type=ObservationModelType.DISTANCE_BASED,
    )
    action = "up"

    # All inside beacon_radius — observations are real arrays.
    for d in [0.0, 0.5, 1.0, 1.5, 2.0]:
        state = np.array([d, 0.0])
        observations = env.sample_observation(state, action, n_samples=4)
        assert all(isinstance(obs, np.ndarray) for obs in observations)

    # All beyond beacon_radius — observations are "None".
    for d in [2.1, 3.0, 5.0]:
        state = np.array([d, 0.0])
        observations = env.sample_observation(state, action, n_samples=4)
        assert all(obs == "None" for obs in observations)


def test_distance_based_observation_model_probability():
    """Test DISTANCE_BASED log-probability for "None" and real observations.

    Purpose: Validates env.observation_log_probability semantics for the
    DISTANCE_BASED model: "None" carries all the mass when far (log 0),
    real observations are impossible when far (-inf), and the inverse
    holds when near (real obs have finite log-density, "None" is -inf).

    Given: A ContinuousLightDarkPOMDPDiscreteActions env with DISTANCE_BASED
        and beacon_radius=1.0.
    When: env.observation_log_probability is queried with "None" and a real
        observation value for near and far next_state.
    Then: prob(None|far)=1 (log 0); prob(None|near)=0 (-inf); prob(real|near)
        is finite; prob(real|far)=0 (-inf).

    Test type: unit
    """
    env = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95,
        beacon_radius=1.0,
        observation_model_type=ObservationModelType.DISTANCE_BASED,
    )
    action = "up"

    state_far = np.array([3.0, 3.0])
    state_near = np.array([0.5, 0.5])

    # "None" probability: 1 when far, 0 when near.
    log_none_far = env.observation_log_probability(state_far, action, ["None"])[0]
    log_none_near = env.observation_log_probability(state_near, action, ["None"])[0]
    assert np.isclose(log_none_far, 0.0)  # log(1)
    assert log_none_near == -np.inf  # log(0)

    # Real observation: finite when near, -inf when far.
    obs_value = np.array([0.5, 0.5])
    log_real_near = env.observation_log_probability(state_near, action, [obs_value])[0]
    log_real_far = env.observation_log_probability(state_far, action, [obs_value])[0]
    assert np.isfinite(log_real_near)
    assert log_real_far == -np.inf


def test_reward_batch_continuous_matches_scalar(base_continuous_light_dark_pomdp):
    """Test that ContinuousLightDarkPOMDP reward_batch matches per-element reward.

    Purpose: Validates vectorized reward_batch gives same results as scalar reward with same seed

    Given: A ContinuousLightDarkPOMDP environment and 100 random states
    When: reward_batch and scalar reward calls are made with identical seeds
    Then: Both produce identical reward arrays and output shape is (N,)

    Test type: unit
    """
    env = base_continuous_light_dark_pomdp
    states = np.random.RandomState(0).uniform(0, 10, (100, 2))
    action = np.array([0.5, 0.5])

    np.random.seed(99)
    batch_rewards = env.reward_batch(states, action)
    assert batch_rewards.shape == (100,)

    np.random.seed(99)
    expected = np.array([env.reward(states[i], action) for i in range(100)])
    np.testing.assert_allclose(batch_rewards, expected)


def test_reward_batch_discrete_actions_matches_scalar(base_light_dark_environment):
    """Test that ContinuousLightDarkPOMDPDiscreteActions reward_batch matches scalar reward.

    Purpose: Validates vectorized reward_batch for discrete-action variant

    Given: A ContinuousLightDarkPOMDPDiscreteActions environment and 100 random states
    When: reward_batch and scalar reward calls are made with identical seeds for string action "up"
    Then: Both produce identical reward arrays and output shape is (N,)

    Test type: unit
    """
    env = base_light_dark_environment
    states = np.random.RandomState(0).uniform(0, 10, (100, 2))
    action = "up"

    np.random.seed(99)
    batch_rewards = env.reward_batch(states, action)
    assert batch_rewards.shape == (100,)

    np.random.seed(99)
    expected = np.array([env.reward(states[i], action) for i in range(100)])
    np.testing.assert_allclose(batch_rewards, expected)


# ---------------------------------------------------------------------------
# Batch sampling and log-probability API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [1, 5, 100])
def test_continuous_sample_next_state_n_samples_shapes(n_samples):
    """env.sample_next_state honours n_samples and returns the right shapes.

    Purpose: Validates the shape contract of env.sample_next_state on the
    continuous-action variant.

    Given: A ContinuousLightDarkPOMDP env and a (state, action) pair.
    When: env.sample_next_state(state, action, n_samples=N) is called for
        N in {1, 5, 100}.
    Then: For n_samples=1 a (2,) ndarray; for n_samples>1 a (N, 2) ndarray.

    Test type: unit
    """
    env = ContinuousLightDarkPOMDP(discount_factor=0.95)
    state = np.array([3.0, 4.0])
    action = np.array([1.0, 0.5])

    direct = env.sample_next_state(state, action, n_samples=n_samples)
    if n_samples == 1:
        assert isinstance(direct, np.ndarray)
        assert direct.shape == (2,)
    else:
        assert isinstance(direct, np.ndarray)
        assert direct.shape == (n_samples, 2)


@pytest.mark.parametrize("n_samples", [1, 5, 100])
def test_continuous_sample_observation_n_samples_shapes(n_samples):
    """env.sample_observation honours n_samples for NORMAL_NOISE.

    Purpose: Validates the shape contract of env.sample_observation on the
    continuous-action variant under the NORMAL_NOISE model type.

    Given: A ContinuousLightDarkPOMDP env and a (next_state, action) pair
        near a beacon (NORMAL_NOISE default).
    When: env.sample_observation(ns, a, n_samples=N) is called for
        N in {1, 5, 100}.
    Then: For n_samples=1 a (2,) ndarray; for n_samples>1 a (N, 2) ndarray.

    Test type: unit
    """
    env = ContinuousLightDarkPOMDP(discount_factor=0.95)
    next_state = np.array([5.0, 5.0])  # near beacon
    action = np.array([0.0, 0.0])

    direct = env.sample_observation(next_state, action, n_samples=n_samples)
    if n_samples == 1:
        assert isinstance(direct, np.ndarray)
        assert direct.shape == (2,)
    else:
        assert isinstance(direct, np.ndarray)
        assert direct.shape == (n_samples, 2)


def test_continuous_sample_observation_n_samples_for_non_normal_noise():
    """env.sample_observation honours n_samples for non-NORMAL_NOISE types.

    Purpose: Validates that NORMAL_NOISE_NO_OBS_IN_DARK and DISTANCE_BASED
    observation paths return lists of length n_samples whose elements are
    either ndarray observations (near beacon) or "None" sentinels.

    Given: Two ContinuousLightDarkPOMDP envs, one per non-NORMAL_NOISE type,
        and a near-beacon next_state.
    When: env.sample_observation(next_state, action, n_samples=4) is called.
    Then: Returned list has length 4 with each element either an ndarray of
        shape (2,) or the "None" sentinel.

    Test type: unit
    """
    for obs_type in (
        ObservationModelType.NORMAL_NOISE_NO_OBS_IN_DARK,
        ObservationModelType.DISTANCE_BASED,
    ):
        env = ContinuousLightDarkPOMDP(
            discount_factor=0.95,
            observation_model_type=obs_type,
        )
        next_state = np.array([5.0, 5.0])
        action = np.array([0.0, 0.0])
        direct = env.sample_observation(next_state, action, n_samples=4)
        assert len(direct) == 4
        for value in direct:
            assert isinstance(value, np.ndarray) or value == "None"
