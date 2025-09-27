"""Tests for CartPole POMDP environment.

This module tests the CartPole POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

import random

import numpy as np
import pytest

from POMDPPlanners.environments.cartpole_pomdp import (
    CartPoleInitialStateDistribution,
    CartPoleObservation,
    CartPolePOMDP,
    CartPoleStateTransition,
)

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


@pytest.fixture
def base_cartpole_environment() -> CartPolePOMDP:
    """Fixture providing a base CartPolePOMDP environment for comparison."""
    noise_cov = np.eye(4) * 0.1
    return CartPolePOMDP(discount_factor=0.95, noise_cov=noise_cov)


class TestCartPolePOMDPEquality:
    """Test suite for CartPolePOMDP equality comparisons."""

    def test_same_discount_factor(self, base_cartpole_environment: CartPolePOMDP):
        """Test that CartPolePOMDPs with same discount factor are equal.

        Purpose: Validates that CartPolePOMDP equality comparison works correctly for identical discount factors

        Given: Two CartPolePOMDP environments with identical discount_factor=0.95 and noise_cov
        When: Equality comparison is performed between the environments
        Then: Both environments are equal to each other, demonstrating symmetry of equality operator

        Test type: unit
        """
        other_env = CartPolePOMDP(
            discount_factor=0.95, noise_cov=base_cartpole_environment.noise_cov
        )
        assert base_cartpole_environment == other_env
        assert other_env == base_cartpole_environment  # Test symmetry

    def test_different_discount_factor(self, base_cartpole_environment: CartPolePOMDP):
        """Test that CartPolePOMDPs with different discount factors are not equal.

        Purpose: Validates that CartPolePOMDP equality comparison correctly identifies different discount factors

        Given: Two CartPolePOMDP environments with different discount factors (0.95 vs 0.8) but same noise_cov
        When: Equality comparison is performed between the environments
        Then: Environments are not equal to each other, demonstrating symmetry of inequality operator

        Test type: unit
        """
        other_env = CartPolePOMDP(
            discount_factor=0.8, noise_cov=base_cartpole_environment.noise_cov
        )
        assert base_cartpole_environment != other_env
        assert other_env != base_cartpole_environment  # Test symmetry

    def test_different_noise_covariance(self, base_cartpole_environment: CartPolePOMDP):
        """Test that CartPolePOMDPs with different noise covariance are not equal.

        Purpose: Validates that CartPolePOMDP equality comparison correctly identifies different noise covariance matrices

        Given: Two CartPolePOMDP environments with same discount_factor=0.95 but different noise_cov (0.1 vs 0.2)
        When: Equality comparison is performed between the environments
        Then: Environments are not equal to each other, demonstrating that noise covariance affects equality

        Test type: unit
        """
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=np.eye(4) * 0.2,  # Different noise covariance
        )
        assert base_cartpole_environment != other_env
        assert other_env != base_cartpole_environment  # Test symmetry

    def test_different_physical_parameters(self, base_cartpole_environment: CartPolePOMDP):
        """Test that CartPolePOMDPs with different physical parameters are not equal.

        Purpose: Validates that CartPolePOMDP equality comparison correctly identifies different physical parameters

        Given: Two CartPolePOMDP environments with same discount_factor and noise_cov but different gravity (9.8 vs 10.0) and cart mass (1.0 vs 2.0)
        When: Equality comparison is performed between the environments
        Then: Environments are not equal to each other, demonstrating that physical parameters affect equality

        Test type: unit
        """
        other_env = CartPolePOMDP(
            discount_factor=0.95, noise_cov=base_cartpole_environment.noise_cov
        )
        other_env.gravity = 10.0  # Different gravity
        assert base_cartpole_environment != other_env

        other_env = CartPolePOMDP(
            discount_factor=0.95, noise_cov=base_cartpole_environment.noise_cov
        )
        other_env.masscart = 2.0  # Different cart mass
        assert base_cartpole_environment != other_env

    def test_comparison_with_non_environment(self, base_cartpole_environment: CartPolePOMDP):
        """Test comparison with non-Environment objects.

        Purpose: Validates that CartPolePOMDP equality comparison correctly handles non-Environment objects

        Given: A CartPolePOMDP environment and various non-Environment objects (string, integer, None)
        When: Equality comparison is performed between environment and non-Environment objects
        Then: All comparisons return False, demonstrating proper type checking in equality operator

        Test type: unit
        """
        assert base_cartpole_environment != "not an environment"
        assert base_cartpole_environment != 42
        assert base_cartpole_environment != None

    def test_missing_attributes(self, base_cartpole_environment: CartPolePOMDP):
        """Test equality when attributes are missing.

        Purpose: Validates that CartPolePOMDP equality comparison correctly handles missing attributes

        Given: A CartPolePOMDP environment and another environment with missing 'gravity' or 'masscart' attributes
        When: Equality comparison is performed between environments with missing attributes
        Then: Environments are not equal, demonstrating that missing attributes affect equality comparison

        Test type: unit
        """
        other_env = CartPolePOMDP(
            discount_factor=0.95, noise_cov=base_cartpole_environment.noise_cov
        )
        delattr(other_env, "gravity")
        assert base_cartpole_environment != other_env

        other_env = CartPolePOMDP(
            discount_factor=0.95, noise_cov=base_cartpole_environment.noise_cov
        )
        delattr(other_env, "masscart")
        assert base_cartpole_environment != other_env

    def test_deep_copy_equality(self, base_cartpole_environment: CartPolePOMDP):
        """Test that a deep copy of CartPolePOMDP is equal to original.

        Purpose: Validates that CartPolePOMDP equality comparison works correctly with deep copies

        Given: A CartPolePOMDP environment and its deep copy with identical attributes
        When: Equality comparison is performed between original and deep copy
        Then: Both environments are equal to each other, demonstrating symmetry and deep copy integrity

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

        Purpose: Validates that CartPolePOMDP config_id generates consistent identifiers for identical configurations

        Given: Two CartPolePOMDP environments with identical discount_factor=0.95 and noise_cov
        When: Config IDs are generated for both environments
        Then: Both environments have identical config_ids, demonstrating consistency for same configuration

        Test type: configuration
        """
        other_env = CartPolePOMDP(
            discount_factor=0.95, noise_cov=base_cartpole_environment.noise_cov
        )
        assert base_cartpole_environment.config_id == other_env.config_id

    def test_config_id_different_discount_factor(self, base_cartpole_environment: CartPolePOMDP):
        """Test that config_id changes with different discount factor.

        Purpose: Validates that CartPolePOMDP config_id generates different identifiers for different discount factors

        Given: Two CartPolePOMDP environments with different discount factors (0.95 vs 0.8) but same noise_cov
        When: Config IDs are generated for both environments
        Then: Environments have different config_ids, demonstrating uniqueness for different configurations

        Test type: configuration
        """
        other_env = CartPolePOMDP(
            discount_factor=0.8, noise_cov=base_cartpole_environment.noise_cov
        )
        assert base_cartpole_environment.config_id != other_env.config_id

    def test_config_id_different_noise_covariance(self, base_cartpole_environment: CartPolePOMDP):
        """Test that config_id changes with different noise covariance.

        Purpose: Validates that CartPolePOMDP config_id generates different identifiers for different noise covariance matrices

        Given: Two CartPolePOMDP environments with same discount_factor=0.95 but different noise_cov (0.1 vs 0.2)
        When: Config IDs are generated for both environments
        Then: Environments have different config_ids, demonstrating uniqueness for different noise configurations

        Test type: configuration
        """
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=np.eye(4) * 0.2,  # Different noise covariance
        )
        assert base_cartpole_environment.config_id != other_env.config_id

    def test_config_id_different_physical_parameters(
        self, base_cartpole_environment: CartPolePOMDP
    ):
        """Test that config_id changes with different physical parameters.

        Purpose: Validates that CartPolePOMDP config_id generates different identifiers for different physical parameters

        Given: Two CartPolePOMDP environments with same discount_factor and noise_cov but different gravity (9.8 vs 10.0) and cart mass (1.0 vs 2.0)
        When: Config IDs are generated for both environments
        Then: Environments have different config_ids, demonstrating uniqueness for different physical configurations

        Test type: configuration
        """
        other_env = CartPolePOMDP(
            discount_factor=0.95, noise_cov=base_cartpole_environment.noise_cov
        )
        other_env.gravity = 10.0  # Different gravity
        assert base_cartpole_environment.config_id != other_env.config_id

        other_env = CartPolePOMDP(
            discount_factor=0.95, noise_cov=base_cartpole_environment.noise_cov
        )
        other_env.masscart = 2.0  # Different cart mass
        assert base_cartpole_environment.config_id != other_env.config_id

    def test_config_id_format(self, base_cartpole_environment: CartPolePOMDP):
        """Test that config_id is a valid SHA-256 hash.

        Purpose: Validates that CartPolePOMDP config_id generates properly formatted SHA-256 hash identifiers

        Given: A CartPolePOMDP environment with specific configuration
        When: Config ID is generated for the environment
        Then: Returns a 64-character string containing only valid hexadecimal characters (0-9, a-f)

        Test type: configuration
        """
        config_id = base_cartpole_environment.config_id
        assert isinstance(config_id, str)
        assert len(config_id) == 64  # SHA-256 hash length
        assert all(c in "0123456789abcdef" for c in config_id)  # Valid hex characters

    def test_config_id_deterministic(self, base_cartpole_environment: CartPolePOMDP):
        """Test that config_id is deterministic (same input always produces same output).

        Purpose: Validates that CartPolePOMDP config_id generates deterministic identifiers for identical configurations

        Given: A CartPolePOMDP environment with specific configuration
        When: Config ID is generated multiple times for the same environment
        Then: All generated config_ids are identical, demonstrating deterministic behavior

        Test type: configuration
        """
        config_id1 = base_cartpole_environment.config_id
        config_id2 = base_cartpole_environment.config_id
        assert config_id1 == config_id2


def test_state_transition_model(base_cartpole_environment):
    """Test state transition model.

    Purpose: Validates that CartPolePOMDP state transition model works correctly

    Given: A CartPolePOMDP environment and initial state [0.0, 0.0, 0.0, 0.0] with action 0
    When: State transition model is created and next state is sampled
    Then: Returns a 4D numpy array representing the next state after applying the action

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

    Purpose: Validates that CartPolePOMDP observation model works correctly

    Given: A CartPolePOMDP environment and state [0.0, 0.0, 0.0, 0.0] with action 0
    When: Observation model is created and observation is sampled
    Then: Returns a 4D numpy array representing the noisy observation of the state

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

    Purpose: Validates that CartPolePOMDP initial state distribution works correctly

    Given: A CartPolePOMDP environment
    When: Initial state distribution is created and state is sampled
    Then: Returns a 4D numpy array representing the initial state of the cart-pole system

    Test type: unit
    """
    # Test initial state distribution
    dist = base_cartpole_environment.initial_state_dist()
    state = dist.sample()[0]
    assert isinstance(state, np.ndarray)
    assert state.shape == (4,)


def test_cartpole_pomdp_initialization():
    """Test cartpole pomdp initialization.

    Purpose: Validates that CartPolePOMDP initializes correctly with specified parameters

    Given: Constructor parameters including discount_factor=0.95 and noise_cov matrix
    When: CartPolePOMDP environment is created
    Then: Environment has correct attributes: noise_cov matches input, gravity=9.8, masscart=1.0, masspole=0.1, length=0.5

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

    Purpose: Validates that CartPolePOMDP reward function works correctly for different states

    Given: A CartPolePOMDP environment and different states (non-terminal vs terminal)
    When: Reward function is called with state and action
    Then: Returns 1.0 for non-terminal states and 0.0 for terminal states (pole angle too large)

    Test type: unit
    """
    # Test reward function
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=np.eye(4) * 0.1)

    # Test non-terminal state
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = 0
    reward = env.reward(state, action)
    assert reward == 1.0

    # Test terminal state (pole angle too large)
    state = np.array([0.0, 0.0, 0.3, 0.0])  # theta > theta_threshold
    reward = env.reward(state, action)
    assert reward == 0.0


def test_cartpole_pomdp_terminal():
    """Test cartpole pomdp terminal.

    Purpose: Validates that CartPolePOMDP terminal state detection works correctly

    Given: A CartPolePOMDP environment and different states (non-terminal vs terminal)
    When: is_terminal method is called with various states
    Then: Returns False for non-terminal states and True for terminal states (cart position too far or pole angle too large)

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

    Purpose: Validates that CartPolePOMDP model creation methods work correctly

    Given: A CartPolePOMDP environment and state [0.0, 0.0, 0.0, 0.0] with action [0]
    When: Various model creation methods are called (state_transition_model, observation_model, initial_state_dist, initial_observation_dist)
    Then: Returns correct model types: CartPoleStateTransition, CartPoleObservation, and CartPoleInitialStateDistribution

    Test type: unit
    """
    # Test model creation
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=np.eye(4) * 0.1)
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = 0

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
