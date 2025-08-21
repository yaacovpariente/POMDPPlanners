import numpy as np
import pytest
from pathlib import Path
import random

from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import ContinuousLightDarkPOMDPDiscreteActions, ContinuousLightDarkPOMDP
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import ObservationModel, SpaceInfo, SpaceType
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.policy import PolicyRunData, PolicyInfoVariable

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
    
    def test_same_discount_factor(self, base_light_dark_environment: ContinuousLightDarkPOMDPDiscreteActions):
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
    
    def test_different_discount_factor(self, base_light_dark_environment: ContinuousLightDarkPOMDPDiscreteActions):
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
    
    def test_different_covariance_matrices(self, base_light_dark_environment: ContinuousLightDarkPOMDPDiscreteActions):
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
    
    def test_different_radii(self, base_light_dark_environment: ContinuousLightDarkPOMDPDiscreteActions):
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
    expected_beacons = np.array(
        [[0, 0, 0, 5, 5, 5, 10, 10, 10], [0, 5, 10, 0, 5, 10, 0, 5, 10]]
    )
    assert np.array_equal(env.beacons, expected_beacons)

    # Check default obstacles
    expected_obstacles = np.array([[3, 7], [5, 5]])
    assert np.array_equal(env.obstacles, expected_obstacles)


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
    env = ContinuousLightDarkPOMDPDiscreteActions(
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
    env = ContinuousLightDarkPOMDPDiscreteActions(
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
    ], discount_factor=0.95, average_state_sampling_time=0.0, average_action_time=0.0, average_observation_time=0.0, average_belief_update_time=0.0, average_reward_time=0.0, actual_num_steps=2, reach_terminal_state=True, policy_run_data=PolicyRunData(info_variables=[]))
    
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
    ], discount_factor=0.95, average_state_sampling_time=0.0, average_action_time=0.0, average_observation_time=0.0, average_belief_update_time=0.0, average_reward_time=0.0, actual_num_steps=2, reach_terminal_state=True, policy_run_data=PolicyRunData(info_variables=[]))
    
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
    expected_beacons = np.array(
        [[0, 0, 0, 5, 5, 5, 10, 10, 10], [0, 5, 10, 0, 5, 10, 0, 5, 10]]
    )
    assert np.array_equal(env.beacons, expected_beacons)
    expected_obstacles = np.array([[3, 7], [5, 5]])
    assert np.array_equal(env.obstacles, expected_obstacles)


def test_continuous_light_dark_pomdp_state_transition_model(base_continuous_light_dark_pomdp):
    env = base_continuous_light_dark_pomdp
    state = np.array([5, 5])
    action = np.array([0, 1])
    dist = env.state_transition_model(state, action)
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import StateTransitionModel
    assert isinstance(dist, StateTransitionModel)
    next_state = dist.sample()
    expected_next_state = state + action
    # Allow for noise in state transition (3 standard deviations)
    assert np.allclose(next_state, expected_next_state, atol=3.0)


def test_continuous_light_dark_pomdp_observation_model(base_continuous_light_dark_pomdp):
    env = base_continuous_light_dark_pomdp
    next_state = np.array([5, 5])
    action = np.array([0, 1])
    dist = env.observation_model(next_state, action)
    from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_observation_models import ContinuousLightDarkNormalNoiseObservationModel
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
    from POMDPPlanners.core.simulation import History, StepData
    from POMDPPlanners.core.belief import WeightedParticleBelief
    from POMDPPlanners.core.policy import PolicyRunData, PolicyInfoVariable
    def create_test_belief(state):
        return WeightedParticleBelief(
            particles=[state],
            log_weights=np.array([1.0]),
            resampling=False
        )
    history1 = History([
        StepData(
            state=np.array([0, 5]),
            action=np.array([1, 0]),
            next_state=np.array([1, 5]),
            observation=np.array([1, 5]),
            reward=-2.0,
            belief=create_test_belief(np.array([0, 5]))
        ),
        StepData(
            state=np.array([9, 5]),
            action=np.array([1, 0]),
            next_state=np.array([10, 5]),
            observation=np.array([10, 5]),
            reward=8.0,
            belief=create_test_belief(np.array([9, 5]))
        ),
    ], discount_factor=0.95, average_state_sampling_time=0.0, average_action_time=0.0, average_observation_time=0.0, average_belief_update_time=0.0, average_reward_time=0.0, actual_num_steps=2, reach_terminal_state=True, policy_run_data=PolicyRunData(info_variables=[]))
    history2 = History([
        StepData(
            state=np.array([0, 5]),
            action=np.array([1, 0]),
            next_state=np.array([1, 5]),
            observation=np.array([1, 5]),
            reward=-2.0,
            belief=create_test_belief(np.array([0, 5]))
        ),
        StepData(
            state=np.array([2, 5]),
            action=np.array([1, 0]),
            next_state=np.array([3, 5]),
            observation=np.array([3, 5]),
            reward=-12.0,
            belief=create_test_belief(np.array([2, 5]))
        ),
    ], discount_factor=0.95, average_state_sampling_time=0.0, average_action_time=0.0, average_observation_time=0.0, average_belief_update_time=0.0, average_reward_time=0.0, actual_num_steps=2, reach_terminal_state=True, policy_run_data=PolicyRunData(info_variables=[]))
    metrics = env.compute_metrics([history1, history2])
    metrics_dict = {metric.name: metric for metric in metrics}
    assert "goal_reaching_rate" in metrics_dict
    goal_rate = metrics_dict["goal_reaching_rate"]
    assert goal_rate.value == 0.5
    assert goal_rate.lower_confidence_bound <= goal_rate.value <= goal_rate.upper_confidence_bound
    assert "obstacle_hit_rate" in metrics_dict
    obstacle_rate = metrics_dict["obstacle_hit_rate"]
    assert obstacle_rate.value == 0.5
    assert obstacle_rate.lower_confidence_bound <= obstacle_rate.value <= obstacle_rate.upper_confidence_bound


def test_decaying_hit_probability_reward_model():
    """Test that the environment uses the decaying hit probability reward model when specified."""
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import RewardModelType
    from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models import ContinuousLightDarkDecayingHitProbabilityRewardModel

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
        penalty_decay=0.5
    )
    assert isinstance(env.reward_model, ContinuousLightDarkDecayingHitProbabilityRewardModel)


def test_dangerous_states_reward_model():
    """Test that the environment uses the dangerous states reward model when specified."""
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import RewardModelType
    from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models import ContinuousLDDangerousStatesRewardModel

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
        reward_model_type=RewardModelType.DANGEROUS_STATES
    )
    
    # Test that the correct reward model is initialized
    assert isinstance(env.reward_model, ContinuousLDDangerousStatesRewardModel)
    
    # Test that the reward model produces both positive and negative rewards near obstacles
    state = np.array([2, 5])  # Near an obstacle
    action = "right"  # Use string action instead of numpy array
    
    # Run multiple times to ensure we get both positive and negative rewards
    rewards = [env.reward(state, action) for _ in range(100)]
    positive_rewards = [r for r in rewards if r > 0]
    negative_rewards = [r for r in rewards if r < 0]
    
    # Check that we get both positive and negative rewards
    assert len(positive_rewards) > 0, "Should get some positive rewards"
    assert len(negative_rewards) > 0, "Should get some negative rewards"
    # Note: We do not check the mean reward, as the environment's reward structure includes other penalties.


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
            np.array([3, 5])   # Final position
        ]
        
        # Create simple belief distributions
        belief_path = [
            DiscreteDistribution(values=[np.array([0, 5]), np.array([0, 4])], probs=np.array([0.8, 0.2])),
            DiscreteDistribution(values=[np.array([1, 5]), np.array([1, 4])], probs=np.array([0.9, 0.1])),
            DiscreteDistribution(values=[np.array([2, 5]), np.array([2, 4])], probs=np.array([0.95, 0.05])),
            DiscreteDistribution(values=[np.array([3, 5])], probs=np.array([1.0]))
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
        with open(cache_path, 'rb') as f:
            magic = f.read(6)
            assert magic in [b'GIF87a', b'GIF89a'], "File should have valid GIF magic bytes"
    
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
    
    def test_visualize_path_with_invalid_cache_path_extension(self, base_light_dark_environment, tmp_path):
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
    
    def test_visualize_path_with_complex_belief_distributions(self, base_light_dark_environment, tmp_path):
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
            np.array([4, 5])
        ]
        
        # Create complex belief distributions with multiple particles
        belief_path = [
            DiscreteDistribution(
                values=[np.array([0, 5]), np.array([0, 4]), np.array([1, 5]), np.array([0, 6])], 
                probs=np.array([0.4, 0.3, 0.2, 0.1])
            ),
            DiscreteDistribution(
                values=[np.array([1, 4]), np.array([1, 3]), np.array([2, 4]), np.array([1, 5])], 
                probs=np.array([0.5, 0.2, 0.2, 0.1])
            ),
            DiscreteDistribution(
                values=[np.array([2, 4]), np.array([2, 3]), np.array([3, 4])], 
                probs=np.array([0.6, 0.25, 0.15])
            ),
            DiscreteDistribution(
                values=[np.array([3, 5]), np.array([3, 4])], 
                probs=np.array([0.8, 0.2])
            ),
            DiscreteDistribution(
                values=[np.array([4, 5])], 
                probs=np.array([1.0])
            )
        ]
        
        actions = ["down", "right", "up", "right"]
        cache_path = tmp_path / "complex_visualization.gif"
        
        env.visualize_path(path, belief_path, actions, cache_path)
        
        # Verify successful creation
        assert cache_path.exists()
        assert cache_path.stat().st_size > 0
    
    def test_visualize_path_with_continuous_actions(self, base_continuous_light_dark_pomdp, tmp_path):
        """Test visualize_path with continuous actions (numpy arrays).
        
        Purpose: Validates visualization works with continuous action spaces
        
        Given: A continuous light-dark environment with numpy array actions
        When: visualize_path is called with continuous actions
        Then: GIF file is created successfully with proper action arrows
        
        Test type: unit
        """
        env = base_continuous_light_dark_pomdp
        
        path = [
            np.array([0, 5]),
            np.array([1.5, 4.5]),
            np.array([3.2, 4.8])
        ]
        
        belief_path = [
            DiscreteDistribution(values=[np.array([0, 5])], probs=np.array([1.0])),
            DiscreteDistribution(values=[np.array([1.5, 4.5])], probs=np.array([1.0])),
            DiscreteDistribution(values=[np.array([3.2, 4.8])], probs=np.array([1.0]))
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
            DiscreteDistribution(values=[np.array([1, 5])], probs=np.array([1.0]))
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
            np.array([4, 5])
        ]
        belief_path2 = [
            DiscreteDistribution(values=[np.array([i, 5])], probs=np.array([1.0])) 
            for i in range(5)
        ]
        actions2 = ["right", "right", "right", "right"]
        
        env.visualize_path(path2, belief_path2, actions2, cache_path)
        
        # Verify file was updated
        assert cache_path.exists()
        second_size = cache_path.stat().st_size
        second_mtime = cache_path.stat().st_mtime
        
        # File should have been modified (different size or modification time)
        assert second_mtime > first_mtime or second_size != first_size, \
            "Cache file should be updated with new visualization"
    
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
            DiscreteDistribution(values=[np.array([1, 5])], probs=np.array([1.0]))
        ]
        actions = ["right"]
        
        # Should raise FileNotFoundError due to missing parent directories
        with pytest.raises(FileNotFoundError):
            env.visualize_path(path, belief_path, actions, nested_cache_path)
        
        # Now create parent directories and try again
        nested_cache_path.parent.mkdir(parents=True)
        env.visualize_path(path, belief_path, actions, nested_cache_path)
        
        # Verify file was created successfully
        assert nested_cache_path.exists(), "GIF file should be created when parent directories exist"
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
        path = [np.array([0, 5]), np.array([1, 5]), np.array([2, 5]), np.array([3, 5])]  # 4 states
        belief_path = [
            DiscreteDistribution(values=[np.array([0, 5])], probs=np.array([1.0])),
            DiscreteDistribution(values=[np.array([1, 5])], probs=np.array([1.0]))  # Only 2 beliefs
        ]
        actions = ["right"]  # Only 1 action
        
        cache_path = tmp_path / "mismatched_lengths.gif"
        
        # Should handle gracefully without throwing exceptions
        env.visualize_path(path, belief_path, actions, cache_path)
        
        assert cache_path.exists()
        assert cache_path.stat().st_size > 0
    
    def test_visualize_path_belief_particles_with_zero_probabilities(self, base_light_dark_environment, tmp_path):
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
                probs=np.array([0.8, 0.0, 0.2])  # Middle particle has zero probability
            ),
            DiscreteDistribution(
                values=[np.array([1, 5]), np.array([1, 4])], 
                probs=np.array([1.0, 0.0])  # Second particle has zero probability
            )
        ]
        
        actions = ["right"]
        cache_path = tmp_path / "zero_prob_particles.gif"
        
        env.visualize_path(path, belief_path, actions, cache_path)
        
        assert cache_path.exists()
        assert cache_path.stat().st_size > 0
    
    def test_visualize_path_preserves_existing_cache_structure(self, base_light_dark_environment, tmp_path):
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
            DiscreteDistribution(values=[np.array([1, 5])], probs=np.array([1.0]))
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
