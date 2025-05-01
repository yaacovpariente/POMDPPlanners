import numpy as np
import pytest
from pathlib import Path

from POMDPPlanners.environments.discrete_light_dark_pomdp import DiscreteLightDarkPOMDP
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import ObservationModel


@pytest.fixture
def base_light_dark_environment() -> DiscreteLightDarkPOMDP:
    """Fixture providing a base DiscreteLightDarkPOMDP environment for comparison."""
    return DiscreteLightDarkPOMDP(
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


class TestDiscreteLightDarkPOMDPEquality:
    """Test suite for DiscreteLightDarkPOMDP equality comparisons."""
    
    def test_same_discount_factor(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with same discount factor are equal."""
        other_env = DiscreteLightDarkPOMDP(
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
        assert base_light_dark_environment == other_env
        assert other_env == base_light_dark_environment  # Test symmetry
    
    def test_different_discount_factor(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different discount factors are not equal."""
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.8,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment != other_env
        assert other_env != base_light_dark_environment  # Test symmetry
    
    def test_different_transition_error(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different transition error probabilities are not equal."""
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.1,  # Different transition error
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment != other_env
    
    def test_different_observation_error(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different observation error probabilities are not equal."""
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.1,  # Different observation error
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment != other_env
    
    def test_different_beacons(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different beacon positions are not equal."""
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            beacons=np.array([[1, 1, 1, 6, 6, 6, 11, 11, 11], [1, 6, 11, 1, 6, 11, 1, 6, 11]]),  # Different beacons
        )
        assert base_light_dark_environment != other_env
    
    def test_different_obstacles(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different obstacle positions are not equal."""
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            obstacles=np.array([[4, 8], [6, 6]]),  # Different obstacles
        )
        assert base_light_dark_environment != other_env
    
    def test_different_goal_state(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different goal states are not equal."""
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            goal_state=np.array([9, 4]),  # Different goal state
        )
        assert base_light_dark_environment != other_env
    
    def test_different_start_state(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different start states are not equal."""
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            start_state=np.array([1, 4]),  # Different start state
        )
        assert base_light_dark_environment != other_env
    
    def test_different_rewards(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different rewards are not equal."""
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-20.0,  # Different obstacle reward
            goal_reward=20.0,  # Different goal reward
            fuel_cost=3.0,  # Different fuel cost
            grid_size=11,
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment != other_env
    
    def test_different_grid_size(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different grid sizes are not equal."""
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=15,  # Different grid size
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment != other_env
    
    def test_different_beacon_radius(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different beacon radii are not equal."""
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            beacon_radius=2.0,  # Different beacon radius
        )
        assert base_light_dark_environment != other_env
    
    def test_different_stochastic_reward(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different stochastic reward settings are not equal."""
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=False,  # Different stochastic reward setting
        )
        assert base_light_dark_environment != other_env
    
    def test_comparison_with_non_environment(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test comparison with non-Environment objects."""
        assert base_light_dark_environment != "not an environment"
        assert base_light_dark_environment != 42
        assert base_light_dark_environment != None
    
    def test_missing_attributes(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test equality when attributes are missing."""
        other_env = DiscreteLightDarkPOMDP(
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
        delattr(other_env, 'beacons')
        assert base_light_dark_environment != other_env
        
        other_env = DiscreteLightDarkPOMDP(
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
        delattr(other_env, 'obstacles')
        assert base_light_dark_environment != other_env
    
    def test_deep_copy_equality(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that a deep copy of DiscreteLightDarkPOMDP is equal to original."""
        import copy
        copied_env = copy.deepcopy(base_light_dark_environment)
        assert copied_env == base_light_dark_environment
        assert base_light_dark_environment == copied_env  # Test symmetry


class TestDiscreteLightDarkPOMDPConfigId:
    """Test suite for DiscreteLightDarkPOMDP config_id functionality."""
    
    def test_config_id_consistency(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that config_id is consistent for identical environments."""
        other_env = DiscreteLightDarkPOMDP(
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
        assert base_light_dark_environment.config_id == other_env.config_id
    
    def test_config_id_different_discount_factor(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that config_id changes with different discount factor."""
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.8,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment.config_id != other_env.config_id
    
    def test_config_id_different_parameters(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that config_id changes with different parameters."""
        # Test different transition error
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.1,  # Different
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment.config_id != other_env.config_id
        
        # Test different observation error
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.1,  # Different
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment.config_id != other_env.config_id
        
        # Test different beacons
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            beacons=np.array([[1, 1, 1, 6, 6, 6, 11, 11, 11], [1, 6, 11, 1, 6, 11, 1, 6, 11]]),  # Different
        )
        assert base_light_dark_environment.config_id != other_env.config_id
    
    def test_config_id_format(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that config_id is a valid SHA-256 hash."""
        config_id = base_light_dark_environment.config_id
        assert isinstance(config_id, str)
        assert len(config_id) == 64  # SHA-256 hash length
        assert all(c in '0123456789abcdef' for c in config_id)  # Valid hex characters
    
    def test_config_id_deterministic(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that config_id is deterministic (same input always produces same output)."""
        config_id1 = base_light_dark_environment.config_id
        config_id2 = base_light_dark_environment.config_id
        assert config_id1 == config_id2
    
    def test_config_id_order_invariance(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that config_id is invariant to the order of beacons and obstacles."""
        # Create environment with same beacons but in different order
        beacons_reordered = base_light_dark_environment.beacons.copy()
        # Swap first and last beacon columns
        beacons_reordered[:, [0, -1]] = beacons_reordered[:, [-1, 0]]
        
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            beacons=beacons_reordered
        )
        assert base_light_dark_environment.config_id == other_env.config_id
        
        # Create environment with same obstacles but in different order
        obstacles_reordered = base_light_dark_environment.obstacles.copy()
        # Swap the two obstacle positions
        obstacles_reordered[:, [0, 1]] = obstacles_reordered[:, [1, 0]]
        
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            obstacles=obstacles_reordered
        )
        assert base_light_dark_environment.config_id == other_env.config_id
        
        # Test both beacons and obstacles reordered together
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            beacons=beacons_reordered,
            obstacles=obstacles_reordered
        )
        assert base_light_dark_environment.config_id == other_env.config_id


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


def test_compute_metrics():
    """Test computation of metrics for different simulation histories"""
    env = DiscreteLightDarkPOMDP(discount_factor=0.95)
    
    # Create test histories
    from POMDPPlanners.core.simulation import History, StepData
    
    # History 1: Reaches goal in 3 steps
    history1 = History([
        StepData(state=np.array([0, 5]), action="right", next_state=np.array([1, 5]), observation=np.array([1, 5]), reward=-2.0),
        StepData(state=np.array([1, 5]), action="right", next_state=np.array([2, 5]), observation=np.array([2, 5]), reward=-2.0),
        StepData(state=np.array([2, 5]), action="right", next_state=np.array([3, 5]), observation=np.array([3, 5]), reward=-2.0),
        StepData(state=np.array([10, 5]), action="right", next_state=np.array([10, 5]), observation=np.array([10, 5]), reward=8.0),  # Goal state
    ], discount_factor=0.95, average_state_sampling_time=0.0, average_action_time=0.0, average_observation_time=0.0, average_belief_update_time=0.0, average_reward_time=0.0, actual_num_steps=4, reach_terminal_state=True)
    
    # History 2: Hits obstacle
    history2 = History([
        StepData(state=np.array([0, 5]), action="right", next_state=np.array([1, 5]), observation=np.array([1, 5]), reward=-2.0),
        StepData(state=np.array([1, 5]), action="right", next_state=np.array([2, 5]), observation=np.array([2, 5]), reward=-2.0),
        StepData(state=np.array([3, 5]), action="right", next_state=np.array([3, 5]), observation=np.array([3, 5]), reward=-12.0),  # Obstacle state
    ], discount_factor=0.95, average_state_sampling_time=0.0, average_action_time=0.0, average_observation_time=0.0, average_belief_update_time=0.0, average_reward_time=0.0, actual_num_steps=3, reach_terminal_state=True)
    
    # History 3: Reaches goal in 5 steps, avoiding obstacle by going up
    history3 = History([
        StepData(state=np.array([0, 5]), action="right", next_state=np.array([1, 5]), observation=np.array([1, 5]), reward=-2.0),
        StepData(state=np.array([1, 5]), action="right", next_state=np.array([2, 5]), observation=np.array([2, 5]), reward=-2.0),
        StepData(state=np.array([2, 5]), action="up", next_state=np.array([2, 6]), observation=np.array([2, 6]), reward=-2.0),
        StepData(state=np.array([2, 6]), action="right", next_state=np.array([3, 6]), observation=np.array([3, 6]), reward=-2.0),
        StepData(state=np.array([3, 6]), action="right", next_state=np.array([4, 6]), observation=np.array([4, 6]), reward=-2.0),
        StepData(state=np.array([10, 5]), action="right", next_state=np.array([10, 5]), observation=np.array([10, 5]), reward=8.0),  # Goal state
    ], discount_factor=0.95, average_state_sampling_time=0.0, average_action_time=0.0, average_observation_time=0.0, average_belief_update_time=0.0, average_reward_time=0.0, actual_num_steps=6, reach_terminal_state=True)
    
    # Compute metrics
    metrics = env.compute_metrics([history1, history2, history3])
    
    # Convert metrics to dictionary for easier access
    metrics_dict = {metric.name: metric for metric in metrics}
    
    # Test goal reaching rate
    assert "goal_reaching_rate" in metrics_dict
    goal_rate = metrics_dict["goal_reaching_rate"]
    assert goal_rate.value == 2/3  # 2 out of 3 histories reach goal
    assert goal_rate.lower_confidence_bound <= goal_rate.value <= goal_rate.upper_confidence_bound
    
    # Test obstacle hit rate
    assert "obstacle_hit_rate" in metrics_dict
    obstacle_rate = metrics_dict["obstacle_hit_rate"]
    assert obstacle_rate.value == 1/3  # 1 out of 3 histories hits obstacle
    assert obstacle_rate.lower_confidence_bound <= obstacle_rate.value <= obstacle_rate.upper_confidence_bound
