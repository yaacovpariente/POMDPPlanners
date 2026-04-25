"""Tests for CartPole POMDP environment.

This module tests the CartPole POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

import copy
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
        assert base_cartpole_environment is not None

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
    When: env.sample_next_state is called
    Then: Returns a 4D numpy array representing the next state after applying the action

    Test type: unit
    """
    # Test state transition
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = 0
    next_state = base_cartpole_environment.sample_next_state(state=state, action=action)
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (4,)


def test_state_transition_produces_varying_samples(base_cartpole_environment):
    """Test that state transition with noise produces varying samples.

    Purpose: Validates that the stochastic state transition model produces
    different samples due to Gaussian process noise

    Given: A CartPolePOMDP environment and initial state [0.0, 0.0, 0.1, 0.0] with action 1
    When: Multiple samples are drawn via env.sample_next_state
    Then: Not all samples are identical, confirming stochastic behavior

    Test type: unit
    """
    state = np.array([0.0, 0.0, 0.1, 0.0])
    action = 1
    samples = base_cartpole_environment.sample_next_state(state=state, action=action, n_samples=50)

    assert len(samples) == 50
    assert all(s.shape == (4,) for s in samples)

    # With noise, samples should not all be identical
    sample_array = np.array(samples)
    assert not np.all(
        sample_array == sample_array[0]
    ), "All samples are identical — noise is not being applied"


def test_state_transition_noise_magnitude(base_cartpole_environment):
    """Test that state transition noise magnitude is reasonable.

    Purpose: Validates that noisy samples cluster around the deterministic next state

    Given: A CartPolePOMDP environment with default noise covariance
    When: Many samples are drawn from the state transition model
    Then: Sample mean is close to deterministic next state and standard deviations
    match the expected noise levels from the covariance matrix

    Test type: unit
    """
    state = np.array([0.0, 0.0, 0.1, 0.0])
    action = 1
    transition = base_cartpole_environment.state_transition_model(state, action)
    samples = np.array(transition.sample(n_samples=5000))

    # The mean of many samples should be close to the deterministic next state
    # pylint: disable=protected-access
    deterministic = transition._compute_deterministic_next_state()
    sample_mean = samples.mean(axis=0)
    np.testing.assert_allclose(sample_mean, deterministic, atol=0.01)

    # Standard deviations should roughly match sqrt of diagonal of cov matrix
    expected_std = np.sqrt(np.diag(base_cartpole_environment.state_transition_cov))
    sample_std = samples.std(axis=0)
    np.testing.assert_allclose(sample_std, expected_std, rtol=0.3)


def test_state_transition_default_covariance():
    """Test that default state transition covariance is used when not specified.

    Purpose: Validates that CartPolePOMDP uses the default state transition
    covariance matrix when none is provided

    Given: A CartPolePOMDP environment created without explicit state_transition_cov
    When: The state_transition_cov attribute is checked
    Then: It matches the class-level DEFAULT_STATE_TRANSITION_COV

    Test type: unit
    """
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=np.eye(4) * 0.1)
    np.testing.assert_array_equal(
        env.state_transition_cov, CartPolePOMDP.DEFAULT_STATE_TRANSITION_COV
    )


def test_state_transition_custom_covariance():
    """Test that custom state transition covariance is applied correctly.

    Purpose: Validates that a custom state transition covariance matrix is
    stored and used when explicitly provided

    Given: A CartPolePOMDP environment created with a custom state_transition_cov
    When: The state_transition_cov attribute is checked
    Then: It matches the custom covariance matrix provided at construction

    Test type: unit
    """
    custom_cov = np.diag([1e-3, 1e-3, 1e-4, 1e-3])
    env = CartPolePOMDP(
        discount_factor=0.95,
        noise_cov=np.eye(4) * 0.1,
        state_transition_cov=custom_cov,
    )
    np.testing.assert_array_equal(env.state_transition_cov, custom_cov)


def test_observation_model(base_cartpole_environment):
    """Test observation model.

    Purpose: Validates that CartPolePOMDP observation sampling works correctly

    Given: A CartPolePOMDP environment and next_state [0.0, 0.0, 0.0, 0.0] with action 0
    When: env.sample_observation is called
    Then: Returns a 4D numpy array representing the noisy observation of the state

    Test type: unit
    """
    # Test observation model
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = 0
    obs = base_cartpole_environment.sample_observation(next_state=state, action=action)
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


def test_cartpole_observation_model_probability_shape_single_observation():
    """Test that observation_log_probability returns correct shape for single observation.

    Purpose: Validates that env.observation_log_probability() returns a 1D array of scalars,
    not a 2D array, when given a single observation

    Given: A CartPolePOMDP env with diag(0.1) noise_cov, next_state [0.1, 0.05, 0.02, -0.1]
        and a single observation
    When: env.observation_log_probability is called with a list containing one observation
    Then: Returns 1D numpy array with shape (1,) containing a scalar log-probability value

    Test type: unit
    """
    # ARRANGE: Create env with desired noise cov
    true_state = np.array([0.1, 0.05, 0.02, -0.1])
    action = 1
    noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=noise_cov)

    # Create single observation
    observation = np.array([0.12, 0.06, 0.025, -0.09])

    # ACT: Get probability via env-level API
    log_probs = env.observation_log_probability(
        next_state=true_state, action=action, observations=[observation]
    )
    probs = np.exp(log_probs)

    # ASSERT: Check shape and type
    assert isinstance(probs, np.ndarray), "probability should return numpy array"
    assert (
        probs.ndim == 1
    ), f"probability should return 1D array, got {probs.ndim}D array with shape {probs.shape}"
    assert probs.shape == (1,), f"shape (1,) expected, got {probs.shape}"
    assert np.isscalar(probs[0]), f"Individual probability should be scalar, got {type(probs[0])}"
    assert probs[0] > 0.0, f"Probability density should be positive, got {probs[0]}"


def test_cartpole_observation_model_probability_shape_multiple_observations():
    """Test that observation_log_probability returns correct shape for multiple observations.

    Purpose: Validates that env.observation_log_probability() returns a 1D array of scalars
    when given multiple observations, with length matching number of observations

    Given: A CartPolePOMDP env with diag(0.1) noise_cov, next_state [0.1, 0.05, 0.02, -0.1]
        and three observations
    When: env.observation_log_probability is called with a list containing three observations
    Then: Returns 1D numpy array with shape (3,) containing scalar log-probability values

    Test type: unit
    """
    # ARRANGE: Create env with desired noise cov
    true_state = np.array([0.1, 0.05, 0.02, -0.1])
    action = 1
    noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=noise_cov)

    # Create multiple observations
    observations = [
        np.array([0.12, 0.06, 0.025, -0.09]),
        np.array([0.08, 0.04, 0.015, -0.11]),
        np.array([0.11, 0.055, 0.022, -0.095]),
    ]

    # ACT: Get probabilities via env-level API
    probs = np.exp(
        env.observation_log_probability(
            next_state=true_state, action=action, observations=observations
        )
    )

    # ASSERT: Check shape and type
    assert isinstance(probs, np.ndarray), "probability should return numpy array"
    assert (
        probs.ndim == 1
    ), f"probability should return 1D array, got {probs.ndim}D array with shape {probs.shape}"
    assert probs.shape == (3,), f"shape (3,) expected, got {probs.shape}"

    # Check each probability density is a scalar and positive
    for i, prob in enumerate(probs):
        assert np.isscalar(prob), f"Individual probability[{i}] should be scalar, got {type(prob)}"
        assert isinstance(
            prob, (int, float, np.floating)
        ), f"Probability density[{i}] should be numeric, got {type(prob)}"
        assert prob > 0.0, f"Probability density[{i}] should be positive, got {prob}"


def test_cartpole_observation_model_probability_empty_list():
    """Test that observation_log_probability handles empty observation list correctly.

    Purpose: Validates that env.observation_log_probability() returns empty 1D array for empty input

    Given: A CartPolePOMDP env with diag(0.1) noise_cov and an empty list of observations
    When: env.observation_log_probability is called with empty list
    Then: Returns empty 1D numpy array with shape (0,)

    Test type: unit
    """
    # ARRANGE: Create env with desired noise cov
    true_state = np.array([0.1, 0.05, 0.02, -0.1])
    action = 1
    noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=noise_cov)

    # ACT: Get probability for empty list via env-level API
    probs = np.exp(
        env.observation_log_probability(next_state=true_state, action=action, observations=[])
    )

    # ASSERT: Check shape
    assert isinstance(probs, np.ndarray), "probability should return numpy array"
    assert probs.ndim == 1, f"probability should return 1D array, got {probs.ndim}D array"
    assert probs.shape == (0,), f"shape (0,) expected, got {probs.shape}"


def test_cartpole_observation_model_probability_values_reasonable():
    """Test that observation_log_probability values are reasonable for noisy observations.

    Purpose: Validates that env.observation_log_probability computes reasonable probability values
    based on Gaussian noise model, with closer observations having higher probability

    Given: A CartPolePOMDP env with diag(0.1) noise_cov and observations at different
        distances from true state
    When: env.observation_log_probability is called with close and far observations
    Then: Closer observations have higher probability than distant observations

    Test type: unit
    """
    # ARRANGE: Create env with desired noise cov
    true_state = np.array([0.1, 0.05, 0.02, -0.1])
    action = 1
    noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=noise_cov)

    # Create observations: one close to true state, one far
    close_obs = true_state + np.array([0.01, 0.01, 0.01, 0.01])  # Small deviation
    far_obs = true_state + np.array([1.0, 1.0, 1.0, 1.0])  # Large deviation

    # ACT: Get probabilities via env-level API
    probs = np.exp(
        env.observation_log_probability(
            next_state=true_state, action=action, observations=[close_obs, far_obs]
        )
    )

    # ASSERT: Close observation should have higher probability
    assert probs[0] > probs[1], f"Close prob ({probs[0]}) should exceed far prob ({probs[1]})"

    # Both should be positive (Gaussian has non-zero probability everywhere)
    assert probs[0] > 0.0, "Close observation should have positive probability"
    assert probs[1] > 0.0, "Far observation should have positive (but smaller) probability"


def test_get_metric_names():
    """Test that get_metric_names returns goal_reaching_rate.

    Purpose: Validates that CartPolePOMDP returns the correct metric names

    Given: A CartPolePOMDP environment
    When: get_metric_names is called
    Then: Returns list containing "goal_reaching_rate"

    Test type: unit
    """
    noise_cov = np.eye(4) * 0.1
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=noise_cov)
    metric_names = env.get_metric_names()
    assert "goal_reaching_rate" in metric_names
    assert len(metric_names) == 1


def test_compute_metrics_goal_reaching():
    """Test computation of goal-reaching metrics for different simulation histories.

    Purpose: Validates that CartPolePOMDP computes goal-reaching rate correctly

    Given: Three simulation histories - 2 completing successfully (no crash), 1 crashing
    When: compute_metrics analyzes the simulation histories
    Then: Returns goal_reaching_rate=2/3 with confidence bounds

    Test type: unit
    """
    from POMDPPlanners.core.policy import PolicyRunData
    from POMDPPlanners.core.simulation import History, StepData

    noise_cov = np.eye(4) * 0.1
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=noise_cov)

    # Create a simple belief for testing
    def create_test_belief(state):
        from POMDPPlanners.core.belief import WeightedParticleBelief

        return WeightedParticleBelief(
            particles=[state], log_weights=np.array([1.0]), resampling=False
        )

    # History 1: Completes successfully (no terminal state reached)
    history1 = History(
        [
            StepData(
                state=np.array([0.0, 0.0, 0.0, 0.0]),
                action=0,
                next_state=np.array([0.0, 0.0, 0.0, 0.0]),
                observation=np.array([0.0, 0.0, 0.0, 0.0]),
                reward=1.0,
                belief=create_test_belief(np.array([0.0, 0.0, 0.0, 0.0])),
            ),
            StepData(
                state=np.array([0.0, 0.0, 0.0, 0.0]),
                action=1,
                next_state=np.array([0.0, 0.0, 0.0, 0.0]),
                observation=np.array([0.0, 0.0, 0.0, 0.0]),
                reward=1.0,
                belief=create_test_belief(np.array([0.0, 0.0, 0.0, 0.0])),
            ),
        ],
        discount_factor=0.95,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=2,
        reach_terminal_state=False,
        policy_run_data=[PolicyRunData(info_variables=[])],
    )

    # History 2: Completes successfully (no terminal state reached)
    history2 = History(
        [
            StepData(
                state=np.array([0.0, 0.0, 0.0, 0.0]),
                action=1,
                next_state=np.array([0.0, 0.0, 0.0, 0.0]),
                observation=np.array([0.0, 0.0, 0.0, 0.0]),
                reward=1.0,
                belief=create_test_belief(np.array([0.0, 0.0, 0.0, 0.0])),
            ),
        ],
        discount_factor=0.95,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=1,
        reach_terminal_state=False,
        policy_run_data=[PolicyRunData(info_variables=[])],
    )

    # History 3: Crashes (reaches terminal state - pole angle too large)
    history3 = History(
        [
            StepData(
                state=np.array([0.0, 0.0, 0.0, 0.0]),
                action=0,
                next_state=np.array([0.0, 0.0, 0.0, 0.0]),
                observation=np.array([0.0, 0.0, 0.0, 0.0]),
                reward=1.0,
                belief=create_test_belief(np.array([0.0, 0.0, 0.0, 0.0])),
            ),
            StepData(
                state=np.array([0.0, 0.0, 0.3, 0.0]),  # Terminal state (theta > threshold)
                action=0,
                next_state=np.array([0.0, 0.0, 0.3, 0.0]),
                observation=np.array([0.0, 0.0, 0.3, 0.0]),
                reward=0.0,
                belief=create_test_belief(np.array([0.0, 0.0, 0.3, 0.0])),
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
    metrics = env.compute_metrics([history1, history2, history3])

    # Convert metrics to dictionary for easier access
    metrics_dict = {metric.name: metric for metric in metrics}

    # Test goal reaching rate
    assert "goal_reaching_rate" in metrics_dict
    goal_rate = metrics_dict["goal_reaching_rate"]
    assert goal_rate.value == 2 / 3  # 2 out of 3 histories complete successfully
    assert goal_rate.lower_confidence_bound <= goal_rate.value <= goal_rate.upper_confidence_bound


def test_reward_batch_matches_scalar_reward():
    """Test that reward_batch returns results consistent with scalar reward.

    Purpose: Validates that the vectorized reward_batch gives identical outputs
    to calling reward() individually for each state.

    Given: A CartPolePOMDP environment and an array of 100 random states
    When: reward_batch is called with the state array
    Then: Output shape is (N,) and values match element-wise reward() calls exactly

    Test type: unit
    """
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))
    np.random.seed(42)
    states = np.random.randn(100, 4)
    action = 1

    batch_rewards = env.reward_batch(states, action)

    assert batch_rewards.shape == (100,)
    expected = np.array([env.reward(states[i], action) for i in range(100)])
    np.testing.assert_array_equal(batch_rewards, expected)

    # Also test with N=1
    single = env.reward_batch(states[:1], action)
    assert single.shape == (1,)
    assert single[0] == env.reward(states[0], action)


def test_sample_next_state_rng_pinned_equivalence(base_cartpole_environment):
    """Test that sample_next_state matches state_transition_model().sample()[0] under fixed RNG.

    Purpose: Validates that the sample_next_state override produces byte-identical results
        to the wrapper-based path when both use the same native RNG seed.

    Given: A CartPolePOMDP environment and (state, action) pairs covering both actions and
        a range of state vectors near and away from the equilibrium, with the native module
        RNG and Python np.random/random seeded identically before each pair of draws
    When: A sample is drawn through state_transition_model(s, a).sample()[0] and again
        through env.sample_next_state(s, a) after re-seeding
    Then: The two draws produce arrays equal element-wise across all combinations

    Test type: unit
    """
    from POMDPPlanners.environments.cartpole_pomdp import (  # pylint: disable=import-outside-toplevel
        _native,
    )

    env = base_cartpole_environment
    cases = [
        (np.array([0.0, 0.0, 0.0, 0.0]), 0),
        (np.array([0.0, 0.0, 0.0, 0.0]), 1),
        (np.array([0.1, 0.05, 0.02, -0.1]), 1),
        (np.array([-0.1, -0.05, -0.02, 0.1]), 0),
        (np.array([0.5, 0.2, 0.05, 0.0]), 1),
    ]
    for state, action in cases:
        _native.set_seed(2024)
        np.random.seed(2024)
        random.seed(2024)
        wrapper_sample = env.state_transition_model(state=state, action=action).sample()[0]
        _native.set_seed(2024)
        np.random.seed(2024)
        random.seed(2024)
        direct_sample = env.sample_next_state(state=state, action=action)
        np.testing.assert_array_equal(
            wrapper_sample,
            direct_sample,
            err_msg=f"sample_next_state mismatch for ({state.tolist()}, {action})",
        )


def test_sample_observation_rng_pinned_equivalence(base_cartpole_environment):
    """Test that sample_observation matches observation_model().sample()[0] under fixed RNG.

    Purpose: Validates that the sample_observation override produces byte-identical results
        to the wrapper-based path when both use the same native RNG seed.

    Given: A CartPolePOMDP environment and (next_state, action) pairs covering both actions
        across a range of state vectors, with the native module RNG and Python
        np.random/random seeded identically before each pair of draws
    When: An observation is drawn through observation_model(ns, a).sample()[0] and again
        through env.sample_observation(ns, a) after re-seeding
    Then: The two draws produce arrays equal element-wise across all combinations

    Test type: unit
    """
    from POMDPPlanners.environments.cartpole_pomdp import (  # pylint: disable=import-outside-toplevel
        _native,
    )

    env = base_cartpole_environment
    cases = [
        (np.array([0.0, 0.0, 0.0, 0.0]), 0),
        (np.array([0.0, 0.0, 0.0, 0.0]), 1),
        (np.array([0.1, 0.05, 0.02, -0.1]), 1),
        (np.array([-0.1, -0.05, -0.02, 0.1]), 0),
        (np.array([0.5, 0.2, 0.05, 0.0]), 1),
    ]
    for next_state, action in cases:
        _native.set_seed(99)
        np.random.seed(99)
        random.seed(99)
        wrapper_obs = env.observation_model(next_state=next_state, action=action).sample()[0]
        _native.set_seed(99)
        np.random.seed(99)
        random.seed(99)
        direct_obs = env.sample_observation(next_state=next_state, action=action)
        np.testing.assert_array_equal(
            wrapper_obs,
            direct_obs,
            err_msg=f"sample_observation mismatch for ({next_state.tolist()}, {action})",
        )


def test_sample_next_state_n_samples_equivalence(base_cartpole_environment):
    """Test sample_next_state with n>1 matches state_transition_model().sample(n) under fixed RNG.

    Purpose: Validates that the n_samples-aware sample_next_state override produces
        byte-identical batched results to the wrapper-based path when both use the same
        native RNG seed.

    Given: A CartPolePOMDP environment and (state, action) pairs covering both actions
        and a range of state vectors near and away from equilibrium, with the native
        module RNG seeded identically before each pair of draws, for n in {1, 5, 100}
    When: A batch is drawn through state_transition_model(s, a).sample(n) and again
        through env.sample_next_state(s, a, n_samples=n) after re-seeding
    Then: The two batches are equal element-wise across all combinations and all n

    Test type: unit
    """
    from POMDPPlanners.environments.cartpole_pomdp import (  # pylint: disable=import-outside-toplevel
        _native,
    )

    env = base_cartpole_environment
    cases = [
        (np.array([0.0, 0.0, 0.0, 0.0]), 0),
        (np.array([0.0, 0.0, 0.0, 0.0]), 1),
        (np.array([0.1, 0.05, 0.02, -0.1]), 1),
        (np.array([-0.1, -0.05, -0.02, 0.1]), 0),
    ]
    for n in (1, 5, 100):
        for state, action in cases:
            _native.set_seed(2024)
            np.random.seed(2024)
            random.seed(2024)
            wrapper_samples = env.state_transition_model(state=state, action=action).sample(n)
            _native.set_seed(2024)
            np.random.seed(2024)
            random.seed(2024)
            direct_samples = env.sample_next_state(state=state, action=action, n_samples=n)
            wrapper_arr = np.asarray(wrapper_samples).reshape(n, -1)
            direct_arr = np.asarray(direct_samples).reshape(n, -1)
            np.testing.assert_array_equal(
                wrapper_arr,
                direct_arr,
                err_msg=f"sample_next_state n={n} mismatch for ({state.tolist()}, {action})",
            )


def test_sample_observation_n_samples_equivalence(base_cartpole_environment):
    """Test sample_observation with n>1 matches observation_model().sample(n) under fixed RNG.

    Purpose: Validates that the n_samples-aware sample_observation override produces
        byte-identical batched results to the wrapper-based path when both use the same
        native RNG seed.

    Given: A CartPolePOMDP environment and (next_state, action) pairs covering both
        actions across a range of state vectors, with the native module RNG seeded
        identically, for n in {1, 5, 100}
    When: A batch is drawn through observation_model(ns, a).sample(n) and again through
        env.sample_observation(ns, a, n_samples=n) after re-seeding
    Then: The two batches are equal element-wise across all combinations and all n

    Test type: unit
    """
    from POMDPPlanners.environments.cartpole_pomdp import (  # pylint: disable=import-outside-toplevel
        _native,
    )

    env = base_cartpole_environment
    cases = [
        (np.array([0.0, 0.0, 0.0, 0.0]), 0),
        (np.array([0.0, 0.0, 0.0, 0.0]), 1),
        (np.array([0.1, 0.05, 0.02, -0.1]), 1),
        (np.array([-0.1, -0.05, -0.02, 0.1]), 0),
    ]
    for n in (1, 5, 100):
        for next_state, action in cases:
            _native.set_seed(99)
            np.random.seed(99)
            random.seed(99)
            wrapper_samples = env.observation_model(next_state=next_state, action=action).sample(n)
            _native.set_seed(99)
            np.random.seed(99)
            random.seed(99)
            direct_samples = env.sample_observation(
                next_state=next_state, action=action, n_samples=n
            )
            wrapper_arr = np.asarray(wrapper_samples).reshape(n, -1)
            direct_arr = np.asarray(direct_samples).reshape(n, -1)
            np.testing.assert_array_equal(
                wrapper_arr,
                direct_arr,
                err_msg=(
                    f"sample_observation n={n} mismatch for " f"({next_state.tolist()}, {action})"
                ),
            )


def test_transition_log_probability_equivalence(base_cartpole_environment):
    """Test transition_log_probability matches np.log(probability) from the wrapper.

    Purpose: Validates that transition_log_probability returns log-PDFs equivalent
        (within fp tolerance) to applying np.log to the wrapper-based probability path.

    Given: A CartPolePOMDP environment and (state, action) pairs covering both actions,
        plus a batch of candidate next-state arrays
    When: Log-probabilities are computed via env.transition_log_probability(s, a, vals)
        and via np.log(env.state_transition_model(s, a).probability(vals) + 1e-300)
    Then: The two ndarrays are equal element-wise within fp tolerance

    Test type: unit
    """
    env = base_cartpole_environment
    candidate_states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.001, 0.002, 0.05, 0.0],
            [0.05, 0.0, 0.02, 0.01],
            [-0.02, 0.01, -0.005, 0.0],
            [1.0, 0.5, 0.1, 0.2],
        ]
    )
    cases = [
        (np.array([0.0, 0.0, 0.0, 0.0]), 0),
        (np.array([0.0, 0.0, 0.0, 0.0]), 1),
        (np.array([0.1, 0.05, 0.02, -0.1]), 1),
        (np.array([-0.1, -0.05, -0.02, 0.1]), 0),
    ]
    for state, action in cases:
        direct = env.transition_log_probability(
            state=state, action=action, next_states=candidate_states
        )
        wrapper_probs = np.asarray(
            env.state_transition_model(state=state, action=action).probability(candidate_states)
        )
        ref = np.log(wrapper_probs + 1e-300)
        np.testing.assert_allclose(
            direct,
            ref,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"transition_log_probability mismatch for ({state.tolist()}, {action})",
        )


def test_observation_log_probability_equivalence(base_cartpole_environment):
    """Test observation_log_probability matches np.log(probability) from the wrapper.

    Purpose: Validates that observation_log_probability returns log-PDFs equivalent
        (within fp tolerance) to applying np.log to the wrapper-based probability path.

    Given: A CartPolePOMDP environment and (next_state, action) pairs covering both
        actions, plus a batch of candidate observation arrays
    When: Log-probabilities are computed via env.observation_log_probability(ns, a, obs)
        and via np.log(env.observation_model(ns, a).probability(obs) + 1e-300)
    Then: The two ndarrays are equal element-wise within fp tolerance

    Test type: unit
    """
    env = base_cartpole_environment
    candidate_obs = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.05, 0.02, 0.01, 0.0],
            [-0.05, -0.02, -0.01, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 0.0, 0.1, -0.1],
        ]
    )
    cases = [
        (np.array([0.0, 0.0, 0.0, 0.0]), 0),
        (np.array([0.0, 0.0, 0.0, 0.0]), 1),
        (np.array([0.1, 0.05, 0.02, -0.1]), 1),
        (np.array([-0.1, -0.05, -0.02, 0.1]), 0),
    ]
    for next_state, action in cases:
        direct = env.observation_log_probability(
            next_state=next_state, action=action, observations=candidate_obs
        )
        wrapper_probs = np.asarray(
            env.observation_model(next_state=next_state, action=action).probability(candidate_obs)
        )
        ref = np.log(wrapper_probs + 1e-300)
        np.testing.assert_allclose(
            direct,
            ref,
            rtol=1e-12,
            atol=1e-12,
            err_msg=(
                f"observation_log_probability mismatch for " f"({next_state.tolist()}, {action})"
            ),
        )
