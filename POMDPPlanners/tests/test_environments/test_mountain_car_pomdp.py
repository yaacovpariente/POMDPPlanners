import pytest
import numpy as np
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.core.simulation import StepData


def test_mountain_car_initialization():
    pomdp = MountainCarPOMDP(discount_factor=0.95)
    assert pomdp.min_position == -1.2
    assert pomdp.max_position == 0.6
    assert pomdp.max_speed == 0.07
    assert pomdp.goal_position == 0.5
    assert pomdp.power == 0.001
    assert pomdp.gravity == 0.0025
    assert len(pomdp.actions) == 3
    assert -1 in pomdp.actions
    assert 0 in pomdp.actions
    assert 1 in pomdp.actions


def test_mountain_car_state_transition():
    pomdp = MountainCarPOMDP(discount_factor=0.95)

    # Test left acceleration
    state = (0.0, 0.0)
    transition = pomdp.state_transition_model(state, -1)
    new_state = transition.sample()
    assert isinstance(new_state, tuple)
    assert len(new_state) == 2
    assert isinstance(new_state[0], float)
    assert isinstance(new_state[1], float)
    assert new_state[0] <= pomdp.max_position
    assert new_state[0] >= pomdp.min_position
    assert abs(new_state[1]) <= pomdp.max_speed

    # Test right acceleration
    state = (0.0, 0.0)
    transition = pomdp.state_transition_model(state, 1)
    new_state = transition.sample()
    assert new_state[0] <= pomdp.max_position
    assert new_state[0] >= pomdp.min_position
    assert abs(new_state[1]) <= pomdp.max_speed

    # Test no acceleration
    state = (0.0, 0.0)
    transition = pomdp.state_transition_model(state, 0)
    new_state = transition.sample()
    assert new_state[0] <= pomdp.max_position
    assert new_state[0] >= pomdp.min_position
    assert abs(new_state[1]) <= pomdp.max_speed


def test_mountain_car_observation():
    pomdp = MountainCarPOMDP(discount_factor=0.95)

    # Test observation with known state
    state = (0.0, 0.0)
    observation = pomdp.observation_model(state, 0).sample()
    assert isinstance(observation, tuple)
    assert len(observation) == 2
    assert isinstance(observation[0], float)
    assert isinstance(observation[1], float)

    # Test observation noise
    state = (0.0, 0.0)
    observations = [pomdp.observation_model(state, 0).sample() for _ in range(100)]
    positions = [obs[0] for obs in observations]
    velocities = [obs[1] for obs in observations]

    # Check that observations are noisy
    assert np.std(positions) > 0
    assert np.std(velocities) > 0


def test_mountain_car_reward():
    pomdp = MountainCarPOMDP(discount_factor=0.95)

    # Test reward when not at goal
    state = (0.0, 0.0)
    reward = pomdp.reward(state, 0)
    assert reward == -1.0

    # Test reward when at goal
    state = (pomdp.goal_position, 0.0)
    reward = pomdp.reward(state, 0)
    assert reward == 0.0

    # Test reward when past goal
    state = (pomdp.goal_position + 0.1, 0.0)
    reward = pomdp.reward(state, 0)
    assert reward == 0.0


def test_mountain_car_terminal():
    pomdp = MountainCarPOMDP(discount_factor=0.95)

    # Test non-terminal state
    state = (0.0, 0.0)
    assert not pomdp.is_terminal(state)

    # Test terminal state
    state = (pomdp.goal_position, 0.0)
    assert pomdp.is_terminal(state)

    # Test state past goal
    state = (pomdp.goal_position + 0.1, 0.0)
    assert pomdp.is_terminal(state)


def test_mountain_car_initial_state():
    pomdp = MountainCarPOMDP(discount_factor=0.95)
    initial_state = pomdp.initial_state_dist().sample()

    assert isinstance(initial_state, tuple)
    assert len(initial_state) == 2
    assert isinstance(initial_state[0], float)
    assert isinstance(initial_state[1], float)
    assert initial_state[0] >= -0.6
    assert initial_state[0] <= -0.4
    assert initial_state[1] == 0.0


def test_mountain_car_initial_observation():
    pomdp = MountainCarPOMDP(discount_factor=0.95)
    initial_observation = pomdp.initial_observation_dist().sample()

    assert isinstance(initial_observation, tuple)
    assert len(initial_observation) == 2
    assert initial_observation[0] == 0.0
    assert initial_observation[1] == 0.0


def test_mountain_car_actions():
    pomdp = MountainCarPOMDP(discount_factor=0.95)
    actions = pomdp.get_actions()

    assert len(actions) == 3
    assert -1 in actions
    assert 0 in actions
    assert 1 in actions


def test_mountain_car_state_bounds():
    pomdp = MountainCarPOMDP(discount_factor=0.95)

    # Test position bounds
    state = (pomdp.min_position - 0.1, 0.0)
    transition = pomdp.state_transition_model(state, 0).sample()
    assert transition[0] >= pomdp.min_position

    state = (pomdp.max_position + 0.1, 0.0)
    transition = pomdp.state_transition_model(state, 0).sample()
    assert transition[0] <= pomdp.max_position

    # Test velocity bounds
    state = (0.0, pomdp.max_speed + 0.1)
    transition = pomdp.state_transition_model(state, 0).sample()
    assert abs(transition[1]) <= pomdp.max_speed

    state = (0.0, -pomdp.max_speed - 0.1)
    transition = pomdp.state_transition_model(state, 0).sample()
    assert abs(transition[1]) <= pomdp.max_speed


@pytest.fixture
def base_mountain_car_environment() -> MountainCarPOMDP:
    """Fixture providing a base MountainCarPOMDP environment for comparison."""
    return MountainCarPOMDP(discount_factor=0.95)


class TestMountainCarPOMDPEquality:
    """Test suite for MountainCarPOMDP equality comparisons."""
    
    def test_same_discount_factor(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that MountainCarPOMDPs with same discount factor are equal."""
        other_env = MountainCarPOMDP(discount_factor=0.95)
        assert base_mountain_car_environment == other_env
        assert other_env == base_mountain_car_environment  # Test symmetry
    
    def test_different_discount_factor(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that MountainCarPOMDPs with different discount factors are not equal."""
        other_env = MountainCarPOMDP(discount_factor=0.8)
        assert base_mountain_car_environment != other_env
        assert other_env != base_mountain_car_environment  # Test symmetry
    
    def test_different_parameters(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that MountainCarPOMDPs with different parameters are not equal."""
        # Create a copy and modify parameters
        other_env = MountainCarPOMDP(discount_factor=0.95)
        other_env.min_position = -1.0  # Different from -1.2
        assert base_mountain_car_environment != other_env
        
        other_env = MountainCarPOMDP(discount_factor=0.95)
        other_env.max_position = 0.7  # Different from 0.6
        assert base_mountain_car_environment != other_env
        
        other_env = MountainCarPOMDP(discount_factor=0.95)
        other_env.max_speed = 0.08  # Different from 0.07
        assert base_mountain_car_environment != other_env
        
        other_env = MountainCarPOMDP(discount_factor=0.95)
        other_env.goal_position = 0.6  # Different from 0.5
        assert base_mountain_car_environment != other_env
        
        other_env = MountainCarPOMDP(discount_factor=0.95)
        other_env.power = 0.002  # Different from 0.001
        assert base_mountain_car_environment != other_env
        
        other_env = MountainCarPOMDP(discount_factor=0.95)
        other_env.gravity = 0.003  # Different from 0.0025
        assert base_mountain_car_environment != other_env
    
    def test_different_noise_parameters(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that MountainCarPOMDPs with different noise parameters are not equal."""
        other_env = MountainCarPOMDP(discount_factor=0.95)
        other_env.position_noise = 0.2  # Different from 0.1
        assert base_mountain_car_environment != other_env
        
        other_env = MountainCarPOMDP(discount_factor=0.95)
        other_env.velocity_noise = 0.02  # Different from 0.01
        assert base_mountain_car_environment != other_env
        
        # Test covariance matrix changes
        other_env = MountainCarPOMDP(discount_factor=0.95)
        other_env.cov_matrix = np.array([[0.2**2, 0], [0, 0.02**2]])  # Different from original
        assert base_mountain_car_environment != other_env
    
    def test_different_actions(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that MountainCarPOMDPs with different actions are not equal."""
        other_env = MountainCarPOMDP(discount_factor=0.95)
        other_env.actions = [-1, 0, 1, 2]  # Different from [-1, 0, 1]
        assert base_mountain_car_environment != other_env
    
    def test_comparison_with_non_environment(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test comparison with non-Environment objects."""
        assert base_mountain_car_environment != "not an environment"
        assert base_mountain_car_environment != 42
        assert base_mountain_car_environment != None
    
    def test_missing_attributes(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test equality when attributes are missing."""
        other_env = MountainCarPOMDP(discount_factor=0.95)
        delattr(other_env, 'min_position')
        assert base_mountain_car_environment != other_env
        
        other_env = MountainCarPOMDP(discount_factor=0.95)
        delattr(other_env, 'cov_matrix')
        assert base_mountain_car_environment != other_env
    
    def test_deep_copy_equality(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that a deep copy of MountainCarPOMDP is equal to original."""
        import copy
        copied_env = copy.deepcopy(base_mountain_car_environment)
        assert copied_env == base_mountain_car_environment
        assert base_mountain_car_environment == copied_env  # Test symmetry
