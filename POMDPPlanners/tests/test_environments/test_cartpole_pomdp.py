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
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal

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


def test_cartpole_observation_model_probability_shape_single_observation():
    """Test that observation model probability returns correct shape for single observation.

    Purpose: Validates that CartPoleObservation.probability() returns a 1D array of scalars,
    not a 2D array, when given a single observation

    Given: A CartPoleObservation model with state [0.1, 0.05, 0.02, -0.1] and a single observation
    When: probability() method is called with a list containing one observation
    Then: Returns 1D numpy array with shape (1,) containing a scalar probability value

    Test type: unit
    """
    # ARRANGE: Create observation model
    true_state = np.array([0.1, 0.05, 0.02, -0.1])
    action = 1
    noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
    obs_dist = CovarianceParameterizedMultivariateNormal(noise_cov)
    obs_model = CartPoleObservation(next_state=true_state, action=action, obs_dist=obs_dist)

    # Create single observation
    observation = np.array([0.12, 0.06, 0.025, -0.09])

    # ACT: Get probability
    probs = obs_model.probability([observation])

    # ASSERT: Check shape and type
    assert isinstance(probs, np.ndarray), "probability() should return numpy array"
    assert (
        probs.ndim == 1
    ), f"probability() should return 1D array, got {probs.ndim}D array with shape {probs.shape}"
    assert probs.shape == (1,), f"probability([obs]) should have shape (1,), got {probs.shape}"
    assert np.isscalar(probs[0]), f"Individual probability should be scalar, got {type(probs[0])}"
    assert probs[0] > 0.0, f"Probability density should be positive, got {probs[0]}"


def test_cartpole_observation_model_probability_shape_multiple_observations():
    """Test that observation model probability returns correct shape for multiple observations.

    Purpose: Validates that CartPoleObservation.probability() returns a 1D array of scalars
    when given multiple observations, with length matching number of observations

    Given: A CartPoleObservation model with state [0.1, 0.05, 0.02, -0.1] and three observations
    When: probability() method is called with a list containing three observations
    Then: Returns 1D numpy array with shape (3,) containing scalar probability values

    Test type: unit
    """
    # ARRANGE: Create observation model
    true_state = np.array([0.1, 0.05, 0.02, -0.1])
    action = 1
    noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
    obs_dist = CovarianceParameterizedMultivariateNormal(noise_cov)
    obs_model = CartPoleObservation(next_state=true_state, action=action, obs_dist=obs_dist)

    # Create multiple observations
    observations = [
        np.array([0.12, 0.06, 0.025, -0.09]),
        np.array([0.08, 0.04, 0.015, -0.11]),
        np.array([0.11, 0.055, 0.022, -0.095]),
    ]

    # ACT: Get probabilities
    probs = obs_model.probability(observations)

    # ASSERT: Check shape and type
    assert isinstance(probs, np.ndarray), "probability() should return numpy array"
    assert (
        probs.ndim == 1
    ), f"probability() should return 1D array, got {probs.ndim}D array with shape {probs.shape}"
    assert probs.shape == (3,), f"probability(3 obs) should have shape (3,), got {probs.shape}"

    # Check each probability density is a scalar and positive
    for i, prob in enumerate(probs):
        assert np.isscalar(prob), f"Individual probability[{i}] should be scalar, got {type(prob)}"
        assert isinstance(
            prob, (int, float, np.floating)
        ), f"Probability density[{i}] should be numeric, got {type(prob)}"
        assert prob > 0.0, f"Probability density[{i}] should be positive, got {prob}"


def test_cartpole_observation_model_probability_empty_list():
    """Test that observation model probability handles empty observation list correctly.

    Purpose: Validates that CartPoleObservation.probability() returns empty 1D array for empty input

    Given: A CartPoleObservation model and an empty list of observations
    When: probability() method is called with empty list
    Then: Returns empty 1D numpy array with shape (0,)

    Test type: unit
    """
    # ARRANGE: Create observation model
    true_state = np.array([0.1, 0.05, 0.02, -0.1])
    action = 1
    noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
    obs_dist = CovarianceParameterizedMultivariateNormal(noise_cov)
    obs_model = CartPoleObservation(next_state=true_state, action=action, obs_dist=obs_dist)

    # ACT: Get probability for empty list
    probs = obs_model.probability([])

    # ASSERT: Check shape
    assert isinstance(probs, np.ndarray), "probability() should return numpy array"
    assert probs.ndim == 1, f"probability() should return 1D array, got {probs.ndim}D array"
    assert probs.shape == (0,), f"probability([]) should have shape (0,), got {probs.shape}"


def test_cartpole_observation_model_probability_values_reasonable():
    """Test that observation model probability values are reasonable for noisy observations.

    Purpose: Validates that CartPoleObservation.probability() computes reasonable probability values
    based on Gaussian noise model, with closer observations having higher probability

    Given: A CartPoleObservation model and observations at different distances from true state
    When: probability() method is called with observations close to and far from true state
    Then: Closer observations have higher probability than distant observations

    Test type: unit
    """
    # ARRANGE: Create observation model
    true_state = np.array([0.1, 0.05, 0.02, -0.1])
    action = 1
    noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
    obs_dist = CovarianceParameterizedMultivariateNormal(noise_cov)
    obs_model = CartPoleObservation(next_state=true_state, action=action, obs_dist=obs_dist)

    # Create observations: one close to true state, one far
    close_obs = true_state + np.array([0.01, 0.01, 0.01, 0.01])  # Small deviation
    far_obs = true_state + np.array([1.0, 1.0, 1.0, 1.0])  # Large deviation

    # ACT: Get probabilities
    probs = obs_model.probability([close_obs, far_obs])

    # ASSERT: Close observation should have higher probability
    assert (
        probs[0] > probs[1]
    ), f"Close observation prob ({probs[0]}) should be higher than far observation prob ({probs[1]})"

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
