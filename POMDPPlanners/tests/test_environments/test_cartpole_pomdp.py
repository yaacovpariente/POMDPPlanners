import numpy as np
import pytest
from POMDPPlanners.environments.cartpole_pomdp import (
    CartPolePOMDP,
    CartPoleStateTransition,
    CartPoleObservation,
    CartPoleInitialStateDistribution,
)

np.random.seed(42)

@pytest.fixture
def base_cartpole_environment() -> CartPolePOMDP:
    """Fixture providing a base CartPolePOMDP environment for comparison."""
    noise_cov = np.eye(4) * 0.1
    return CartPolePOMDP(discount_factor=0.95, noise_cov=noise_cov)


class TestCartPolePOMDPEquality:
    """Test suite for CartPolePOMDP equality comparisons."""
    
    def test_same_discount_factor(self, base_cartpole_environment: CartPolePOMDP):
        """Test that CartPolePOMDPs with same discount factor are equal.
    
    Purpose: Validates same discount factor
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=base_cartpole_environment.noise_cov
        )
        assert base_cartpole_environment == other_env
        assert other_env == base_cartpole_environment  # Test symmetry
    
    def test_different_discount_factor(self, base_cartpole_environment: CartPolePOMDP):
        """Test that CartPolePOMDPs with different discount factors are not equal.
    
    Purpose: Validates different discount factor
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
        other_env = CartPolePOMDP(
            discount_factor=0.8,
            noise_cov=base_cartpole_environment.noise_cov
        )
        assert base_cartpole_environment != other_env
        assert other_env != base_cartpole_environment  # Test symmetry
    
    def test_different_noise_covariance(self, base_cartpole_environment: CartPolePOMDP):
        """Test that CartPolePOMDPs with different noise covariance are not equal.
    
    Purpose: Validates different noise covariance
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=np.eye(4) * 0.2  # Different noise covariance
        )
        assert base_cartpole_environment != other_env
        assert other_env != base_cartpole_environment  # Test symmetry
    
    def test_different_physical_parameters(self, base_cartpole_environment: CartPolePOMDP):
        """Test that CartPolePOMDPs with different physical parameters are not equal.
    
    Purpose: Validates different physical parameters
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=base_cartpole_environment.noise_cov
        )
        other_env.gravity = 10.0  # Different gravity
        assert base_cartpole_environment != other_env
        
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=base_cartpole_environment.noise_cov
        )
        other_env.masscart = 2.0  # Different cart mass
        assert base_cartpole_environment != other_env
    
    def test_comparison_with_non_environment(self, base_cartpole_environment: CartPolePOMDP):
        """Test comparison with non-Environment objects.
    
    Purpose: Validates comparison with non environment
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
        assert base_cartpole_environment != "not an environment"
        assert base_cartpole_environment != 42
        assert base_cartpole_environment != None
    
    def test_missing_attributes(self, base_cartpole_environment: CartPolePOMDP):
        """Test equality when attributes are missing.
    
    Purpose: Validates missing attributes
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=base_cartpole_environment.noise_cov
        )
        delattr(other_env, 'gravity')
        assert base_cartpole_environment != other_env
        
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=base_cartpole_environment.noise_cov
        )
        delattr(other_env, 'masscart')
        assert base_cartpole_environment != other_env
    
    def test_deep_copy_equality(self, base_cartpole_environment: CartPolePOMDP):
        """Test that a deep copy of CartPolePOMDP is equal to original.
    
    Purpose: Validates equality comparison for deep copy 
    
    Given: Objects with same or different configurations
    When: Equality comparison is performed
    Then: Objects are correctly identified as equal or unequal
    
    Test type: unit
    """
        import copy
        copied_env = copy.deepcopy(base_cartpole_environment)
        assert copied_env == base_cartpole_environment
        assert base_cartpole_environment == copied_env  # Test symmetry


class TestCartPolePOMDPConfigId:
    """Test suite for CartPolePOMDP config_id functionality."""
    
    def test_config_id_consistency(self, base_cartpole_environment: CartPolePOMDP):
        """Test that config_id is consistent for identical environments.
    
    Purpose: Validates config_id behavior for  consistency
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=base_cartpole_environment.noise_cov
        )
        assert base_cartpole_environment.config_id == other_env.config_id
    
    def test_config_id_different_discount_factor(self, base_cartpole_environment: CartPolePOMDP):
        """Test that config_id changes with different discount factor.
    
    Purpose: Validates config_id behavior for  different discount factor
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
        other_env = CartPolePOMDP(
            discount_factor=0.8,
            noise_cov=base_cartpole_environment.noise_cov
        )
        assert base_cartpole_environment.config_id != other_env.config_id
    
    def test_config_id_different_noise_covariance(self, base_cartpole_environment: CartPolePOMDP):
        """Test that config_id changes with different noise covariance.
    
    Purpose: Validates config_id behavior for  different noise covariance
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=np.eye(4) * 0.2  # Different noise covariance
        )
        assert base_cartpole_environment.config_id != other_env.config_id
    
    def test_config_id_different_physical_parameters(self, base_cartpole_environment: CartPolePOMDP):
        """Test that config_id changes with different physical parameters.
    
    Purpose: Validates config_id behavior for  different physical parameters
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=base_cartpole_environment.noise_cov
        )
        other_env.gravity = 10.0  # Different gravity
        assert base_cartpole_environment.config_id != other_env.config_id
        
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=base_cartpole_environment.noise_cov
        )
        other_env.masscart = 2.0  # Different cart mass
        assert base_cartpole_environment.config_id != other_env.config_id
    
    def test_config_id_format(self, base_cartpole_environment: CartPolePOMDP):
        """Test that config_id is a valid SHA-256 hash.
    
    Purpose: Validates config_id behavior for  format
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
        config_id = base_cartpole_environment.config_id
        assert isinstance(config_id, str)
        assert len(config_id) == 64  # SHA-256 hash length
        assert all(c in '0123456789abcdef' for c in config_id)  # Valid hex characters
    
    def test_config_id_deterministic(self, base_cartpole_environment: CartPolePOMDP):
        """Test that config_id is deterministic (same input always produces same output).
    
    Purpose: Validates config_id behavior for  deterministic
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
        config_id1 = base_cartpole_environment.config_id
        config_id2 = base_cartpole_environment.config_id
        assert config_id1 == config_id2


def test_state_transition_model(base_cartpole_environment):
    """Test state transition model.
    
    Purpose: Validates state transition model
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Test state transition
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = 0
    transition = base_cartpole_environment.state_transition_model(state, action)
    next_state = transition.sample()[0]
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (4,)


def test_observation_model(base_cartpole_environment):
    """Test observation model.
    
    Purpose: Validates observation model
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Test observation model
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = 0
    observation = base_cartpole_environment.observation_model(state, action)
    obs = observation.sample()[0]
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)


def test_initial_state_distribution(base_cartpole_environment):
    """Test initial state distribution.
    
    Purpose: Validates initial state distribution
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Test initial state distribution
    dist = base_cartpole_environment.initial_state_dist()
    state = dist.sample()[0]
    assert isinstance(state, np.ndarray)
    assert state.shape == (4,)


def test_cartpole_pomdp_initialization():
    """Test cartpole pomdp initialization.
    
    Purpose: Validates proper initialization of cartpole pomdp 
    
    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes
    
    Test type: unit
    """
    # Test POMDP initialization
    noise_cov = np.eye(4) * 0.1

    env = CartPolePOMDP(discount_factor=0.95, noise_cov=noise_cov)

    # Verify parameters
    assert np.array_equal(env.noise_cov, noise_cov)
    assert env.gravity == 9.8
    assert env.masscart == 1.0
    assert env.masspole == 0.1
    assert env.length == 0.5


def test_cartpole_pomdp_reward():
    """Test cartpole pomdp reward.
    
    Purpose: Validates cartpole pomdp reward
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Test reward function
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=np.eye(4) * 0.1)

    # Test non-terminal state
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = np.array([0])
    reward = env.reward(state, action)
    assert reward == 1.0

    # Test terminal state (pole angle too large)
    state = np.array([0.0, 0.0, 0.3, 0.0])  # theta > theta_threshold
    reward = env.reward(state, action)
    assert reward == 0.0


def test_cartpole_pomdp_terminal():
    """Test cartpole pomdp terminal.
    
    Purpose: Validates cartpole pomdp terminal
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Test terminal state detection
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=np.eye(4) * 0.1)

    # Test non-terminal state
    state = np.array([0.0, 0.0, 0.0, 0.0])
    assert not env.is_terminal(state)

    # Test terminal state (cart position too far)
    state = np.array([2.5, 0.0, 0.0, 0.0])  # x > x_threshold
    assert env.is_terminal(state)

    # Test terminal state (pole angle too large)
    state = np.array([0.0, 0.0, 0.3, 0.0])  # theta > theta_threshold
    assert env.is_terminal(state)


def test_cartpole_pomdp_models():
    """Test cartpole pomdp models.
    
    Purpose: Validates cartpole pomdp models
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Test model creation
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=np.eye(4) * 0.1)
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = np.array([0])

    # Test state transition model
    transition_model = env.state_transition_model(state, action)
    assert isinstance(transition_model, CartPoleStateTransition)

    # Test observation model
    observation_model = env.observation_model(state, action)
    assert isinstance(observation_model, CartPoleObservation)

    # Test initial state distribution
    initial_dist = env.initial_state_dist()
    assert isinstance(initial_dist, CartPoleInitialStateDistribution)

    # Test initial observation distribution
    initial_obs_dist = env.initial_observation_dist()
    assert isinstance(initial_obs_dist, CartPoleInitialStateDistribution)
