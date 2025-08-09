import pytest
import numpy as np
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.core.simulation import StepData


def test_mountain_car_initialization():
    """Test mountain car initialization.
    
    Purpose: Validates proper initialization of mountain car 
    
    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes
    
    Test type: unit
    """
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


def test_state_transition_model(base_mountain_car_environment):
    """Test state transition model.
    
    Purpose: Validates state transition model
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Test state transition
    state = np.array([0.0, 0.0])
    action = 0
    transition = base_mountain_car_environment.state_transition_model(state, action)
    new_state = transition.sample()[0]
    assert isinstance(new_state, np.ndarray)
    assert new_state.shape == (2,)

    # Test with different action
    action = 1
    transition = base_mountain_car_environment.state_transition_model(state, action)
    new_state = transition.sample()[0]
    assert isinstance(new_state, np.ndarray)
    assert new_state.shape == (2,)

    # Test with negative action
    action = -1
    transition = base_mountain_car_environment.state_transition_model(state, action)
    new_state = transition.sample()[0]
    assert isinstance(new_state, np.ndarray)
    assert new_state.shape == (2,)


def test_observation_model(base_mountain_car_environment):
    """Test observation model.
    
    Purpose: Validates observation model
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Test observation model
    state = np.array([0.0, 0.0])
    action = 0
    observation = base_mountain_car_environment.observation_model(state, action)
    obs = observation.sample()[0]
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)

    # Test multiple observations
    observations = base_mountain_car_environment.observation_model(state, action).sample(n_samples=100)
    assert len(observations) == 100
    assert all(isinstance(obs, np.ndarray) and obs.shape == (2,) for obs in observations)


def test_initial_state_distribution(base_mountain_car_environment):
    """Test initial state distribution.
    
    Purpose: Validates initial state distribution
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Test initial state distribution
    dist = base_mountain_car_environment.initial_state_dist()
    initial_state = dist.sample()[0]
    assert isinstance(initial_state, np.ndarray)
    assert initial_state.shape == (2,)


def test_initial_observation_distribution(base_mountain_car_environment):
    """Test initial observation distribution.
    
    Purpose: Validates initial observation distribution
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Test initial observation distribution
    dist = base_mountain_car_environment.initial_observation_dist()
    initial_observation = dist.sample()[0]
    assert isinstance(initial_observation, np.ndarray)
    assert initial_observation.shape == (2,)


def test_sample_next_step(base_mountain_car_environment):
    """Test sample next step.
    
    Purpose: Validates sampling behavior for  next step
    
    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution
    
    Test type: unit
    """
    # Test sample_next_step method
    state = np.array([0.0, 0.0])
    action = 0
    next_state, observation, reward = base_mountain_car_environment.sample_next_step(state, action)
    
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (2,)
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (2,)
    assert isinstance(reward, float)

    # Test with different action
    action = 1
    next_state, observation, reward = base_mountain_car_environment.sample_next_step(state, action)
    
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (2,)
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (2,)
    assert isinstance(reward, float)

    # Test with negative action
    action = -1
    next_state, observation, reward = base_mountain_car_environment.sample_next_step(state, action)
    
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (2,)
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (2,)
    assert isinstance(reward, float)


def test_mountain_car_reward():
    """Test mountain car reward.
    
    Purpose: Validates mountain car reward
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
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
    """Test mountain car terminal.
    
    Purpose: Validates mountain car terminal
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
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


def test_mountain_car_actions():
    """Test mountain car actions.
    
    Purpose: Validates mountain car actions
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    pomdp = MountainCarPOMDP(discount_factor=0.95)
    actions = pomdp.get_actions()

    assert len(actions) == 3
    assert -1 in actions
    assert 0 in actions
    assert 1 in actions


def test_mountain_car_state_bounds():
    """Test mountain car state bounds.
    
    Purpose: Validates mountain car state bounds
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    pomdp = MountainCarPOMDP(discount_factor=0.95)

    # Test position bounds
    state = (pomdp.min_position - 0.1, 0.0)
    transition = pomdp.state_transition_model(state, 0).sample()[0]
    assert transition[0] >= pomdp.min_position

    state = (pomdp.max_position + 0.1, 0.0)
    transition = pomdp.state_transition_model(state, 0).sample()[0]
    assert transition[0] <= pomdp.max_position

    # Test velocity bounds
    state = (0.0, pomdp.max_speed + 0.1)
    transition = pomdp.state_transition_model(state, 0).sample()[0]
    assert abs(transition[1]) <= pomdp.max_speed

    state = (0.0, -pomdp.max_speed - 0.1)
    transition = pomdp.state_transition_model(state, 0).sample()[0]
    assert abs(transition[1]) <= pomdp.max_speed


@pytest.fixture
def base_mountain_car_environment() -> MountainCarPOMDP:
    """Test that MountainCarPOMDPs with same discount factor are equal.
    
    Purpose: Validates same discount factor
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    return MountainCarPOMDP(discount_factor=0.95)


class TestMountainCarPOMDPEquality:
    """Test that MountainCarPOMDPs with different discount factors are not equal.
    
    Purpose: Validates different discount factor
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    
    def test_same_discount_factor(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that MountainCarPOMDPs with same discount factor are equal."""
        other_env = MountainCarPOMDP(discount_factor=0.95)
        assert base_mountain_car_environment == other_env
        assert other_env == base_mountain_car_environment  # Test symmetry
    
    def test_different_discount_factor(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that MountainCarPOMDPs with different parameters are not equal.
    
    Purpose: Validates different parameters
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
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
        """Test that MountainCarPOMDPs with different noise parameters are not equal.
    
    Purpose: Validates different noise parameters
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
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
        
        """Test that MountainCarPOMDPs with different actions are not equal.
    
    Purpose: Validates different actions
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
        other_env.velocity_noise = 0.02  # Different from 0.01
        assert base_mountain_car_environment != other_env
        
        # Test covariance matrix changes
        other_env = MountainCarPOMDP(discount_factor=0.95)
        other_env.cov_matrix = np.array([[0.2**2, 0], [0, 0.02**2]])  # Different from original
        assert base_mountain_car_environment != other_env
    
    def test_different_actions(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test comparison with non-Environment objects.
    
    Purpose: Validates comparison with non environment
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
        other_env = MountainCarPOMDP(discount_factor=0.95)
        other_env.actions = [-1, 0, 1, 2]  # Different from [-1, 0, 1]
        assert base_mountain_car_environment != other_env
    
    def test_comparison_with_non_environment(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test equality when attributes are missing.
    
    Purpose: Validates missing attributes
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
        assert base_mountain_car_environment != "not an environment"
        assert base_mountain_car_environment != 42
        assert base_mountain_car_environment != None
    
    def test_missing_attributes(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that a deep copy of MountainCarPOMDP is equal to original.
    
    Purpose: Validates equality comparison for deep copy 
    
    Given: Objects with same or different configurations
    When: Equality comparison is performed
    Then: Objects are correctly identified as equal or unequal
    
    Test type: unit
    """
        other_env = MountainCarPOMDP(discount_factor=0.95)
        delattr(other_env, 'min_position')
        assert base_mountain_car_environment != other_env
        
        other_env = MountainCarPOMDP(discount_factor=0.95)
        delattr(other_env, 'cov_matrix')
        assert base_mountain_car_environment != other_env
    
    def test_deep_copy_equality(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that config_id is consistent for identical environments.
    
    Purpose: Validates config_id behavior for  consistency
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
        import copy
        copied_env = copy.deepcopy(base_mountain_car_environment)
        assert copied_env == base_mountain_car_environment
        assert base_mountain_car_environment == copied_env  # Test symmetry


class TestMountainCarPOMDPConfigId:
    """Test that config_id changes with different discount factor.
    
    Purpose: Validates config_id behavior for  different discount factor
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
    
    def test_config_id_consistency(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that config_id changes with different parameters.
    
    Purpose: Validates config_id behavior for  different parameters
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
        other_env = MountainCarPOMDP(discount_factor=0.95)
        assert base_mountain_car_environment.config_id == other_env.config_id
    
    def test_config_id_different_discount_factor(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that config_id changes with different discount factor."""
        other_env = MountainCarPOMDP(discount_factor=0.8)
        assert base_mountain_car_environment.config_id != other_env.config_id
    
    def test_config_id_different_parameters(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that config_id changes with different parameters."""
        # Test different position noise
        other_env = MountainCarPOMDP(discount_factor=0.95)
        other_env.position_noise = 0.2  # Different from 0.1
        assert base_mountain_car_environment.config_id != other_env.config_id
        
        # Test different velocity noise
        other_env = MountainCarPOMDP(discount_factor=0.95)
        other_env.velocity_noise = 0.02  # Different from 0.01
        assert base_mountain_car_environment.config_id != other_env.config_id
        
        """Test that config_id is a valid SHA-256 hash.
    
    Purpose: Validates config_id behavior for  format
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
        other_env = MountainCarPOMDP(discount_factor=0.95)
        other_env.power = 0.002  # Different from 0.001
        assert base_mountain_car_environment.config_id != other_env.config_id
        
        other_env = MountainCarPOMDP(discount_factor=0.95)
        other_env.gravity = 0.003  # Different from 0.0025
        assert base_mountain_car_environment.config_id != other_env.config_id
    
    def test_config_id_format(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that config_id is deterministic (same input always produces same output).
    
    Purpose: Validates config_id behavior for  deterministic
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
        config_id = base_mountain_car_environment.config_id
        assert isinstance(config_id, str)
        assert len(config_id) == 64  # SHA-256 hash length
        assert all(c in '0123456789abcdef' for c in config_id)  # Valid hex characters
    
    def test_config_id_deterministic(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that config_id is deterministic (same input always produces same output)."""
        config_id1 = base_mountain_car_environment.config_id
        config_id2 = base_mountain_car_environment.config_id
        assert config_id1 == config_id2
