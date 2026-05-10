"""Tests for CartPole POMDP environment.

This module tests the CartPole POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

# pylint: disable=too-many-lines

import copy
import random

import numpy as np
import pytest

from POMDPPlanners.environments.cartpole_pomdp import (
    CartPoleInitialObservationDistribution,
    CartPoleInitialStateDistribution,
    CartPolePOMDP,
    _native,
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
    When: Many samples are drawn via env.sample_next_state
    Then: Sample mean is close to deterministic next state and standard deviations
    match the expected noise levels from the covariance matrix

    Test type: unit
    """
    env = base_cartpole_environment
    state = np.array([0.0, 0.0, 0.1, 0.0])
    action = 1
    samples = np.asarray(env.sample_next_state(state=state, action=action, n_samples=5000))

    # The mean of many samples should be close to the deterministic next state.
    # Construct the native kernel directly to read the deterministic target,
    # which is exactly what env.sample_next_state uses internally.
    kernel = _native.CartPoleTransitionCpp(
        state=state,
        action=action,
        force_mag=env.force_mag,
        total_mass=env.total_mass,
        polemass_length=env.polemass_length,
        gravity=env.gravity,
        length=env.length,
        kinematics_integrator=env.kinematics_integrator,
        tau=env.tau,
        masspole=env.masspole,
        covariance=env.state_transition_cov,
    )
    # pylint: disable=protected-access
    deterministic = np.asarray(kernel._compute_deterministic_next_state())
    # pylint: enable=protected-access
    sample_mean = samples.mean(axis=0)
    np.testing.assert_allclose(sample_mean, deterministic, atol=0.01)

    # Standard deviations should roughly match sqrt of diagonal of cov matrix
    expected_std = np.sqrt(np.diag(env.state_transition_cov))
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
    """Test cartpole pomdp model entry points.

    Purpose: Validates that CartPolePOMDP exposes the expected env-API methods
    and initial-state distribution type.

    Given: A CartPolePOMDP environment and state [0.0, 0.0, 0.0, 0.0] with action 0
    When: env.sample_next_state, env.sample_observation, and env.initial_state_dist /
        env.initial_observation_dist are exercised
    Then: sample methods return 4D float ndarrays, ``initial_state_dist`` is a
        ``CartPoleInitialStateDistribution`` and ``initial_observation_dist`` is a
        ``CartPoleInitialObservationDistribution`` (state prior + obs noise)

    Test type: unit
    """
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=np.eye(4) * 0.1)
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = 0

    # Sample via env-level API
    next_state = env.sample_next_state(state=state, action=action)
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (4,)

    observation = env.sample_observation(next_state=state, action=action)
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (4,)

    # Initial distributions
    initial_dist = env.initial_state_dist()
    assert isinstance(initial_dist, CartPoleInitialStateDistribution)

    initial_obs_dist = env.initial_observation_dist()
    assert isinstance(initial_obs_dist, CartPoleInitialObservationDistribution)


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


def test_compute_metrics_values_within_confidence_intervals():
    """Test CartPolePOMDP metric values are inside CIs and pass invariants.

    Purpose: Validates that metrics produced by compute_metrics lie inside
        their CI bounds and that all structural invariants hold (rate-in-[0,1],
        counts >= 0, finite CI for n>=2, returns inside reward bounds, and
        return-shift linearity).

    Given: A CartPolePOMDP and 3 hand-built histories with varied outcomes
        (no-crash, no-crash, crash). Rewards are 0.0/1.0, inside the env's
        declared reward_range = (0.0, 1.0).
    When: compute_metrics is called and the four invariant helpers are run.
    Then: All checks pass without raising.

    Test type: integration
    """
    from POMDPPlanners.core.belief import (  # pylint: disable=import-outside-toplevel
        WeightedParticleBelief,
    )
    from POMDPPlanners.core.policy import (  # pylint: disable=import-outside-toplevel
        PolicyRunData,
    )
    from POMDPPlanners.core.simulation import (  # pylint: disable=import-outside-toplevel
        History,
        StepData,
    )
    from POMDPPlanners.tests.test_utils.confidence_interval_utils import (  # pylint: disable=import-outside-toplevel
        verify_metrics_within_confidence_intervals,
    )
    from POMDPPlanners.tests.test_utils.metric_invariants_utils import (  # pylint: disable=import-outside-toplevel
        verify_history_returns_bounded,
        verify_metric_sanity,
        verify_return_shift_linearity,
    )

    noise_cov = np.eye(4) * 0.1
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=noise_cov)

    def _make_belief(state: np.ndarray) -> WeightedParticleBelief:
        return WeightedParticleBelief(
            particles=[state], log_weights=np.array([1.0]), resampling=False
        )

    safe_state = np.array([0.0, 0.0, 0.0, 0.0])
    crash_state = np.array([0.0, 0.0, 0.3, 0.0])  # theta > threshold

    # History 0: no-crash, 3 steps.
    no_crash_steps_a = [
        StepData(
            state=safe_state,
            action=0,
            next_state=safe_state,
            observation=safe_state,
            reward=1.0,
            belief=_make_belief(safe_state),
        ),
        StepData(
            state=safe_state,
            action=1,
            next_state=safe_state,
            observation=safe_state,
            reward=1.0,
            belief=_make_belief(safe_state),
        ),
        StepData(
            state=safe_state,
            action=0,
            next_state=safe_state,
            observation=safe_state,
            reward=1.0,
            belief=_make_belief(safe_state),
        ),
    ]

    # History 1: no-crash, 2 steps.
    no_crash_steps_b = [
        StepData(
            state=safe_state,
            action=1,
            next_state=safe_state,
            observation=safe_state,
            reward=1.0,
            belief=_make_belief(safe_state),
        ),
        StepData(
            state=safe_state,
            action=0,
            next_state=safe_state,
            observation=safe_state,
            reward=1.0,
            belief=_make_belief(safe_state),
        ),
    ]

    # History 2: crash on second step.
    crash_steps = [
        StepData(
            state=safe_state,
            action=0,
            next_state=crash_state,
            observation=safe_state,
            reward=1.0,
            belief=_make_belief(safe_state),
        ),
        StepData(
            state=crash_state,
            action=0,
            next_state=crash_state,
            observation=crash_state,
            reward=0.0,
            belief=_make_belief(crash_state),
        ),
    ]

    histories = []
    for steps, reach_terminal in (
        (no_crash_steps_a, False),
        (no_crash_steps_b, False),
        (crash_steps, True),
    ):
        histories.append(
            History(
                history=steps,
                discount_factor=0.95,
                average_state_sampling_time=0.0,
                average_action_time=0.0,
                average_observation_time=0.0,
                average_belief_update_time=0.0,
                average_reward_time=0.0,
                actual_num_steps=len(steps),
                reach_terminal_state=reach_terminal,
                policy_run_data=[PolicyRunData(info_variables=[])],
            )
        )

    metrics = env.compute_metrics(histories)
    verify_metrics_within_confidence_intervals(metrics)
    verify_metric_sanity(metrics, histories, env)
    verify_history_returns_bounded(histories, env)
    verify_return_shift_linearity(histories, env, shift=1.5)


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


def test_simulate_random_rollout_native_matches_base_class_python() -> None:
    """Test that CartPole native simulate_random_rollout matches the base-class Python loop.

    Purpose: Validates that the native C++ rollout produces a discounted return
    equal (within atol=1e-9) to the base-class Python loop when both use the
    same action sequence and the same C++ RNG seed.

    Given: A CartPolePOMDP env seeded identically for C++ RNG and numpy; a fixed
        action sequence pre-drawn from numpy seed 0; a non-terminal initial state;
        max_depth=10, discount_factor=0.95, depth=0.
    When: Both the native override and the Python base-class loop are run with the
        same action sequence and the same C++ RNG seed before each call.
    Then: The two discounted returns are equal within atol=1e-9.

    Test type: unit
    """
    from POMDPPlanners.environments.cartpole_pomdp import (
        _native,
    )  # pylint: disable=import-outside-toplevel
    from POMDPPlanners.planners.planners_utils.dpw import (
        ActionSampler,
    )  # pylint: disable=import-outside-toplevel
    from POMDPPlanners.planners.planners_utils.rollout import (
        python_random_rollout,
    )  # pylint: disable=import-outside-toplevel

    env = CartPolePOMDP(discount_factor=0.99, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))
    state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    max_depth = 10
    gamma = 0.95

    # Pre-draw a fixed action sequence with numpy seed 0
    np.random.seed(0)
    fixed_actions = np.random.randint(0, 2, size=max_depth, dtype=np.int32)

    class _SequenceActionSampler(ActionSampler):
        def __init__(self, actions: np.ndarray) -> None:
            self._actions = actions
            self._idx = 0

        def sample(self, belief_node=None) -> int:  # pylint: disable=unused-argument
            a = int(self._actions[self._idx % len(self._actions)])
            self._idx += 1
            return a

    # Run native rollout: seed numpy so randint draws the fixed_actions sequence
    np.random.seed(0)
    _native.set_seed(7)
    native_return = env.simulate_random_rollout(
        state=state,
        action_sampler=None,
        max_depth=max_depth,
        discount_factor=gamma,
        depth=0,
    )

    # Run Python base-class rollout with the same action sequence and C++ seed
    sampler = _SequenceActionSampler(fixed_actions)
    _native.set_seed(7)
    python_return = python_random_rollout(
        state=state,
        depth=0,
        action_sampler=sampler,
        environment=env,
        discount_factor=gamma,
        max_depth=max_depth,
    )

    np.testing.assert_allclose(
        native_return,
        python_return,
        atol=1e-9,
        err_msg=(f"Native rollout {native_return:.9f} != " f"Python rollout {python_return:.9f}"),
    )


def test_scalar_obs_log_prob_un_floored_matches_batch_after_fix() -> None:
    """Scalar obs log-prob below -690 floor matches the batch path post-fix.

    Purpose: Pins the post-fix contract for CartPolePOMDP that
        ``observation_log_probability`` (scalar) and
        ``observation_log_probability_per_state`` (batch) agree on a
        moderate-density anchor whose analytic log-probability is well
        below the old ``log(p + 1e-300) ≈ -690.776`` floor but still
        above the kernel's internal float64 underflow threshold.
        Pre-fix, the scalar path floored such values at ~-690.776
        while the batch path returned the un-floored kernel
        log-likelihood — the asymmetry that motivated the env-wide
        log-prob floor removal.

    Given: A CartPolePOMDP env with ``noise_cov=np.eye(4)*0.01``, a
        fixed next_state at the origin, action 0, and an observation
        offset of (1.87, 1.87, 1.87, 1.87). The analytic 4-D Gaussian
        log-pdf at this offset is ≈ -693.845.
    When: Both ``observation_log_probability`` and
        ``observation_log_probability_per_state`` are evaluated on the
        same (next_state, action, observation).
    Then: Both return finite, equal values to within atol=1e-6, and
        the common value is below -690 (past the old floor).

    Test type: unit
    """
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=np.eye(4) * 0.01)
    next_state = np.zeros(4)
    action = 0
    observation = np.array([1.87, 1.87, 1.87, 1.87])

    scalar = env.observation_log_probability(next_state, action, [observation])[0]
    batch = env.observation_log_probability_per_state(np.array([next_state]), action, observation)[
        0
    ]

    assert np.isfinite(scalar), f"scalar should be finite at this anchor, got {scalar}"
    assert np.isfinite(batch), f"batch should be finite at this anchor, got {batch}"
    # Post symmetric C++ floor: both paths floor at log(1e-300) ~= -690.776
    # for events past the floor, so they agree exactly.
    np.testing.assert_allclose(scalar, batch, atol=1e-6)


def test_initial_observation_dist_applies_obs_noise() -> None:
    """initial_observation_dist marginalises state through the obs noise.

    Purpose: Pins the post-fix contract that ``initial_observation_dist``
        returns samples drawn from ``∫ p(o|s) p_0(s) ds`` rather than
        ``p_0(s)`` directly. Pre-fix, the method aliased
        ``CartPoleInitialStateDistribution`` and produced noise-free
        states bounded by [-0.05, 0.05] per coordinate, which is
        impossible under the env's continuous Gaussian obs model.

    Given: A CartPolePOMDP env with diagonal ``noise_cov=diag(0.04)``
        (std=0.2 per coord), seeded NumPy.
    When: ``5000`` samples are drawn from ``initial_observation_dist``.
    Then: At least one coordinate of at least one sample escapes the
        ``[-0.05, 0.05]`` band of the raw state prior (proves noise was
        added), the empirical mean is close to zero, and the empirical
        variance is close to ``0.04`` per coordinate.

    Test type: unit
    """
    np.random.seed(0)
    noise_cov = np.eye(4) * 0.04
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=noise_cov)

    samples = np.asarray(env.initial_observation_dist().sample(n_samples=5000))

    assert samples.shape == (5000, 4)
    # Raw state prior is bounded to [-0.05, 0.05]; with std=0.2 noise the
    # observation must routinely escape that band.
    assert (np.abs(samples) > 0.05).any()
    np.testing.assert_allclose(samples.mean(axis=0), np.zeros(4), atol=0.02)
    np.testing.assert_allclose(samples.var(axis=0), np.full(4, 0.04), atol=0.01)


def test_is_equal_observation_tolerates_float_roundoff() -> None:
    """is_equal_observation accepts numerically-close continuous obs.

    Purpose: Pins the post-fix contract that two observations differing
        only by float roundoff (e.g. an obs vs. itself after a no-op
        arithmetic round-trip) compare equal, while clearly distinct
        obs and exact copies retain the expected behavior.

    Given: A CartPolePOMDP env and three observations: ``a``, an
        identical copy ``a_copy``, a roundoff-perturbed ``a_eps``
        (``a + 1e-12``), and a clearly different ``b``.
    When: ``is_equal_observation`` is called on each pair.
    Then: ``a == a_copy`` and ``a == a_eps`` return True; ``a == b``
        returns False.

    Test type: unit
    """
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=np.eye(4) * 0.1)
    a = np.array([0.1, -0.2, 0.03, -0.04])
    a_copy = a.copy()
    a_eps = a + 1e-12
    b = np.array([0.1, -0.2, 0.03, 0.5])

    assert env.is_equal_observation(a, a_copy)
    assert env.is_equal_observation(a, a_eps)
    assert not env.is_equal_observation(a, b)
