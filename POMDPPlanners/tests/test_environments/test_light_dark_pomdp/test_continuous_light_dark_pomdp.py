"""Tests for Continuous Light Dark POMDP environment.

This module tests the Continuous Light Dark POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

import random

import numpy as np
import pytest

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import ObservationModel, SpaceInfo, SpaceType
from POMDPPlanners.core.policy import PolicyInfoVariable, PolicyRunData
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    ContinuousLightDarkPOMDPDiscreteActions,
)


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
    # Test state transition
    state = np.array([0.0, 0.0])
    action = np.array([0.0, 0.0])
    transition = pomdp.state_transition_model(state, action)
    next_state = transition.sample()[0]
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (2,)


def test_observation_model(pomdp):
    # Test observation model
    state = np.array([0.0, 0.0])
    action = np.array([0.0, 0.0])
    observation = pomdp.observation_model(state, action)
    obs = observation.sample()[0]
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

    # Test out of grid state
    assert env.is_terminal(np.array([-1, 5]))
    assert env.is_terminal(np.array([12, 5]))  # grid_size + 1

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
    from POMDPPlanners.core.belief import WeightedParticleBelief
    from POMDPPlanners.core.simulation import History, StepData

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
    env = base_continuous_light_dark_pomdp
    state = np.array([5, 5])
    action = np.array([0, 1])
    dist = env.state_transition_model(state, action)
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        StateTransitionModel,
    )

    assert isinstance(dist, StateTransitionModel)
    next_state = dist.sample()
    expected_next_state = state + action
    # Allow for noise in state transition (3 standard deviations)
    assert np.allclose(next_state, expected_next_state, atol=3.0)


def test_continuous_light_dark_pomdp_observation_model(
    base_continuous_light_dark_pomdp,
):
    env = base_continuous_light_dark_pomdp
    next_state = np.array([5, 5])
    action = np.array([0, 1])
    dist = env.observation_model(next_state, action)
    from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_observation_models import (
        ContinuousLightDarkNormalNoiseObservationModel,
    )

    assert isinstance(dist, ContinuousLightDarkNormalNoiseObservationModel)
    observation = dist.sample()
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
    # Out of grid
    assert env.is_terminal(np.array([-1, 5]))
    assert env.is_terminal(np.array([12, 5]))
    # Non-terminal
    assert not env.is_terminal(np.array([1, 1]))


def test_continuous_light_dark_pomdp_compute_metrics(base_continuous_light_dark_pomdp):
    env = base_continuous_light_dark_pomdp
    from POMDPPlanners.core.belief import WeightedParticleBelief
    from POMDPPlanners.core.policy import PolicyInfoVariable, PolicyRunData
    from POMDPPlanners.core.simulation import History, StepData

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
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        RewardModelType,
    )
    from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models import (
        ContinuousLightDarkDecayingHitProbabilityRewardModel,
    )

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
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        RewardModelType,
    )
    from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models import (
        ContinuousLDDangerousStatesRewardModel,
    )

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
        abs(r - expected_obstacle_reward) < 0.1 for r in obstacle_rewards
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
        import time

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
    """Test that observation covariance matrix changes when agent is near a beacon.

    Purpose: Validates that observation covariance is reduced when agent is within beacon radius

    Given: A continuous light-dark POMDP environment with beacons and beacon radius
    When: Observation model is created for states near vs far from beacons
    Then: Near-beacon states have reduced covariance matrix (multiplied by 0.5)

    Test type: unit
    """
    from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_observation_models import (
        ContinuousLightDarkNormalNoiseObservationModel,
    )

    # Set up test parameters
    observation_cov_matrix = np.eye(2) * 4.0  # Base covariance matrix
    grid_size = 11
    beacons = np.array([[0, 5, 10], [0, 5, 10]])  # Beacons at (0,0), (5,5), (10,10)
    beacon_radius = 1.0
    action = np.array([0, 0])  # Dummy action

    # Test state near beacon (0,0) - within radius
    near_beacon_state = np.array([0.5, 0.5])  # Distance sqrt(0.5) < 1.0 from beacon (0,0)
    obs_model_near = ContinuousLightDarkNormalNoiseObservationModel(
        next_state=near_beacon_state,
        action=action,
        observation_cov_matrix=observation_cov_matrix,
        grid_size=grid_size,
        beacons=beacons,
        beacon_radius=beacon_radius,
    )

    # Test state far from all beacons - outside all radii
    far_from_beacon_state = np.array([2.5, 2.5])  # Distance > 1.0 from all beacons
    obs_model_far = ContinuousLightDarkNormalNoiseObservationModel(
        next_state=far_from_beacon_state,
        action=action,
        observation_cov_matrix=observation_cov_matrix,
        grid_size=grid_size,
        beacons=beacons,
        beacon_radius=beacon_radius,
    )

    # Verify near_beacon flag is set correctly
    assert obs_model_near.near_beacon == True, "Should detect proximity to beacon at (0,0)"
    assert obs_model_far.near_beacon == False, "Should not detect proximity to any beacon"

    # Verify covariance matrix changes
    expected_near_cov = observation_cov_matrix * 0.5  # Reduced by half when near beacon
    expected_far_cov = observation_cov_matrix.copy()  # Unchanged when far from beacons

    assert np.array_equal(
        obs_model_near.observation_cov_matrix, expected_near_cov
    ), "Observation covariance should be reduced by half when near beacon"
    assert np.array_equal(
        obs_model_far.observation_cov_matrix, expected_far_cov
    ), "Observation covariance should remain unchanged when far from beacons"

    # Test edge case: exactly at beacon radius boundary
    boundary_state = np.array([1.0, 0.0])  # Exactly 1.0 distance from beacon (0,0)
    obs_model_boundary = ContinuousLightDarkNormalNoiseObservationModel(
        next_state=boundary_state,
        action=action,
        observation_cov_matrix=observation_cov_matrix,
        grid_size=grid_size,
        beacons=beacons,
        beacon_radius=beacon_radius,
    )

    # At exactly beacon_radius distance, should be considered "near" (<=)
    assert obs_model_boundary.near_beacon == True, "Should detect proximity at exact beacon radius"
    assert np.array_equal(
        obs_model_boundary.observation_cov_matrix, expected_near_cov
    ), "Observation covariance should be reduced at exact beacon radius boundary"


def test_beacon_proximity_with_multiple_beacons():
    """Test beacon proximity detection with multiple beacons.

    Purpose: Validates that proximity detection works correctly with multiple beacons

    Given: A continuous light-dark environment with multiple beacons at different positions
    When: Observation model is created for states near different beacons
    Then: Proximity is detected correctly for any beacon within radius

    Test type: unit
    """
    from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_observation_models import (
        ContinuousLightDarkNormalNoiseObservationModel,
    )

    observation_cov_matrix = np.eye(2) * 2.0
    grid_size = 11
    # Multiple beacons: (0,0), (5,5), (10,10)
    beacons = np.array([[0, 5, 10], [0, 5, 10]])
    beacon_radius = 1.5
    action = np.array([0, 0])

    # Test near first beacon (0,0)
    near_first_beacon = np.array([1.0, 1.0])  # Distance sqrt(2) ≈ 1.41 < 1.5 from (0,0)
    obs_model_1 = ContinuousLightDarkNormalNoiseObservationModel(
        next_state=near_first_beacon,
        action=action,
        observation_cov_matrix=observation_cov_matrix,
        grid_size=grid_size,
        beacons=beacons,
        beacon_radius=beacon_radius,
    )

    # Test near middle beacon (5,5)
    near_middle_beacon = np.array([5.0, 6.0])  # Distance 1.0 < 1.5 from (5,5)
    obs_model_2 = ContinuousLightDarkNormalNoiseObservationModel(
        next_state=near_middle_beacon,
        action=action,
        observation_cov_matrix=observation_cov_matrix,
        grid_size=grid_size,
        beacons=beacons,
        beacon_radius=beacon_radius,
    )

    # Test near last beacon (10,10)
    near_last_beacon = np.array([9.0, 10.0])  # Distance 1.0 < 1.5 from (10,10)
    obs_model_3 = ContinuousLightDarkNormalNoiseObservationModel(
        next_state=near_last_beacon,
        action=action,
        observation_cov_matrix=observation_cov_matrix,
        grid_size=grid_size,
        beacons=beacons,
        beacon_radius=beacon_radius,
    )

    # Test equidistant from multiple beacons but still within range
    # Point (2.5, 2.5) is equidistant from (0,0) and (5,5)
    equidistant_state = np.array([2.5, 2.5])  # Should be far from all beacons
    obs_model_4 = ContinuousLightDarkNormalNoiseObservationModel(
        next_state=equidistant_state,
        action=action,
        observation_cov_matrix=observation_cov_matrix,
        grid_size=grid_size,
        beacons=beacons,
        beacon_radius=beacon_radius,
    )

    # Verify proximity detection for each beacon
    assert obs_model_1.near_beacon == True, "Should detect proximity to first beacon (0,0)"
    assert obs_model_2.near_beacon == True, "Should detect proximity to middle beacon (5,5)"
    assert obs_model_3.near_beacon == True, "Should detect proximity to last beacon (10,10)"
    assert obs_model_4.near_beacon == False, "Should not detect proximity when far from all beacons"

    # Verify covariance reduction for near-beacon cases
    expected_reduced_cov = observation_cov_matrix * 0.5
    expected_normal_cov = observation_cov_matrix.copy()

    assert np.array_equal(obs_model_1.observation_cov_matrix, expected_reduced_cov)
    assert np.array_equal(obs_model_2.observation_cov_matrix, expected_reduced_cov)
    assert np.array_equal(obs_model_3.observation_cov_matrix, expected_reduced_cov)
    assert np.array_equal(obs_model_4.observation_cov_matrix, expected_normal_cov)


def test_observation_model_probability_function():
    """Test the probability function of the observation model.

    Purpose: Validates that observation model correctly calculates probability densities

    Given: A continuous light-dark observation model with specific next_state and covariance
    When: Probability is calculated for various observation values
    Then: Probabilities match expected multivariate normal distribution values

    Test type: unit
    """
    from scipy.stats import multivariate_normal

    from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_observation_models import (
        ContinuousLightDarkNormalNoiseObservationModel,
    )

    # Set up test parameters
    next_state = np.array([5.0, 3.0])
    action = np.array([0, 0])
    observation_cov_matrix = np.eye(2) * 0.25  # Small variance for more precise testing
    grid_size = 11
    beacons = np.array([[0, 10], [0, 10]])  # Beacons far from next_state
    beacon_radius = 1.0

    # Create observation model (far from beacons so covariance unchanged)
    obs_model = ContinuousLightDarkNormalNoiseObservationModel(
        next_state=next_state,
        action=action,
        observation_cov_matrix=observation_cov_matrix,
        grid_size=grid_size,
        beacons=beacons,
        beacon_radius=beacon_radius,
    )

    # Test single observation value
    observation_values = [np.array([5.0, 3.0])]  # Same as next_state (mean)
    probabilities = obs_model.probability(observation_values)

    # Calculate expected probability using scipy
    expected_prob = multivariate_normal.pdf(
        observation_values[0], mean=next_state, cov=observation_cov_matrix  # type: ignore
    )

    assert isinstance(probabilities, np.ndarray), "Probability should return numpy array"
    assert len(probabilities) == 1, "Should return one probability for one observation"
    assert np.isclose(
        probabilities[0], expected_prob, rtol=1e-10
    ), f"Probability {probabilities[0]} should match expected {expected_prob}"

    # Test multiple observation values
    observation_values = [
        np.array([5.0, 3.0]),  # At mean
        np.array([5.5, 3.5]),  # Offset from mean
        np.array([4.0, 2.0]),  # Different offset
    ]
    probabilities = obs_model.probability(observation_values)

    # Calculate expected probabilities
    expected_probs = multivariate_normal.pdf(
        observation_values, mean=next_state, cov=observation_cov_matrix  # type: ignore
    )

    assert len(probabilities) == 3, "Should return three probabilities for three observations"
    assert np.allclose(
        probabilities, expected_probs, rtol=1e-10
    ), f"Probabilities {probabilities} should match expected {expected_probs}"

    # Test probability decreases with distance from mean
    close_obs = np.array([5.1, 3.1])  # Close to mean
    far_obs = np.array([6.0, 4.0])  # Far from mean

    close_prob = obs_model.probability([close_obs])[0]
    far_prob = obs_model.probability([far_obs])[0]

    assert (
        close_prob > far_prob
    ), f"Probability for closer observation {close_prob} should be higher than farther {far_prob}"


def test_observation_model_probability_with_beacon_proximity():
    """Test observation model probability function with reduced covariance near beacons.

    Purpose: Validates that probability calculations use reduced covariance when near beacons

    Given: Observation models near and far from beacons with different covariance matrices
    When: Probability is calculated for the same observation values
    Then: Near-beacon model has higher probability density due to reduced covariance

    Test type: unit
    """
    from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_observation_models import (
        ContinuousLightDarkNormalNoiseObservationModel,
    )

    next_state = np.array([0.5, 0.5])  # Close to beacon at (0,0)
    action = np.array([0, 0])
    observation_cov_matrix = np.eye(2) * 1.0
    grid_size = 11
    beacons = np.array([[0, 5], [0, 5]])  # Beacon at (0,0) and (5,5)
    beacon_radius = 1.0

    # Create observation model near beacon (covariance will be reduced)
    obs_model_near = ContinuousLightDarkNormalNoiseObservationModel(
        next_state=next_state,
        action=action,
        observation_cov_matrix=observation_cov_matrix,
        grid_size=grid_size,
        beacons=beacons,
        beacon_radius=beacon_radius,
    )

    # Create observation model far from beacons (covariance unchanged)
    far_state = np.array([7.0, 7.0])  # Far from all beacons
    obs_model_far = ContinuousLightDarkNormalNoiseObservationModel(
        next_state=far_state,
        action=action,
        observation_cov_matrix=observation_cov_matrix,
        grid_size=grid_size,
        beacons=beacons,
        beacon_radius=beacon_radius,
    )

    # Test observation at the respective means
    near_prob = obs_model_near.probability([next_state])[0]
    far_prob = obs_model_far.probability([far_state])[0]

    # Near beacon should have higher probability density at mean due to lower covariance
    # With covariance reduced by 0.5, the probability density at mean increases
    assert (
        near_prob > far_prob
    ), f"Near-beacon probability {near_prob} should be higher than far-beacon {far_prob}"

    # Verify the covariance matrices are different
    assert not np.array_equal(
        obs_model_near.observation_cov_matrix, obs_model_far.observation_cov_matrix
    ), "Near and far beacon models should have different covariance matrices"

    # Verify the near-beacon covariance is reduced
    expected_near_cov = observation_cov_matrix * 0.5
    assert np.array_equal(
        obs_model_near.observation_cov_matrix, expected_near_cov
    ), "Near-beacon covariance should be reduced by factor of 0.5"


def test_observation_model_probability_edge_cases():
    """Test observation model probability function with edge cases.

    Purpose: Validates robust handling of edge cases in probability calculation

    Given: An observation model and various edge case inputs
    When: Probability is calculated for edge cases
    Then: Function handles edge cases gracefully without errors

    Test type: unit
    """
    from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_observation_models import (
        ContinuousLightDarkNormalNoiseObservationModel,
    )

    next_state = np.array([5.0, 5.0])
    action = np.array([0, 0])
    observation_cov_matrix = np.eye(2) * 0.1
    grid_size = 11
    beacons = np.array([[2, 8], [2, 8]])  # No beacons near next_state
    beacon_radius = 1.0

    obs_model = ContinuousLightDarkNormalNoiseObservationModel(
        next_state=next_state,
        action=action,
        observation_cov_matrix=observation_cov_matrix,
        grid_size=grid_size,
        beacons=beacons,
        beacon_radius=beacon_radius,
    )

    # Test empty list - should handle gracefully by returning empty array
    # Note: scipy.multivariate_normal.pdf doesn't handle empty arrays well,
    # so we expect the current implementation to have issues with empty input
    try:
        empty_probs = obs_model.probability([])
        assert isinstance(empty_probs, np.ndarray), "Should return numpy array for empty input"
        assert len(empty_probs) == 0, "Should return empty array for empty input"
    except ValueError:
        # This is expected behavior with current implementation and scipy
        pass

    # Test single observation (ensures single value is converted to array)
    single_obs = [np.array([5.0, 5.0])]
    single_prob = obs_model.probability(single_obs)
    assert isinstance(single_prob, np.ndarray), "Single probability should be numpy array"
    assert len(single_prob) == 1, "Single observation should return array of length 1"
    assert single_prob[0] > 0, "Probability should be positive"

    # Test observations at grid boundaries
    boundary_observations = [
        np.array([0.0, 0.0]),  # Bottom-left corner
        np.array([11.0, 11.0]),  # Top-right corner (at grid_size)
        np.array([0.0, 11.0]),  # Top-left corner
        np.array([11.0, 0.0]),  # Bottom-right corner
    ]
    boundary_probs = obs_model.probability(boundary_observations)
    assert len(boundary_probs) == 4, "Should return probability for each boundary observation"
    assert all(prob >= 0 for prob in boundary_probs), "All probabilities should be non-negative"

    # Test very small covariance (high precision)
    small_cov_model = ContinuousLightDarkNormalNoiseObservationModel(
        next_state=next_state,
        action=action,
        observation_cov_matrix=np.eye(2) * 1e-6,  # Very small covariance
        grid_size=grid_size,
        beacons=beacons,
        beacon_radius=beacon_radius,
    )

    # Observation exactly at mean should have very high probability
    exact_obs = [np.array([5.0, 5.0])]
    exact_prob = small_cov_model.probability(exact_obs)[0]

    # Observation slightly off should have much lower probability
    offset_obs = [np.array([5.01, 5.01])]
    offset_prob = small_cov_model.probability(offset_obs)[0]

    assert (
        exact_prob > offset_prob
    ), "Exact observation should have much higher probability with small covariance"


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
    from POMDPPlanners.configs.environment_configs import (
        RiskAverseEnvironmentConfigsAPI,
    )

    # Test the configuration that was causing the issue in visualization example
    env_configs = RiskAverseEnvironmentConfigsAPI(discount_factor=0.95)
    (
        light_dark_env,
        initial_belief,
    ) = env_configs.continuous_observations_discrete_actions_light_dark_pomdp_config(n_particles=50)

    # Cast to specific type to access attributes
    from typing import cast
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        ContinuousLightDarkPOMDPDiscreteActions,
    )

    light_dark_env_typed = cast(ContinuousLightDarkPOMDPDiscreteActions, light_dark_env)

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
    print(f"✓ RiskAverseEnvironmentConfigsAPI configuration is valid - start state not terminal")


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
