"""Tests for MountainCar POMDP environment.

This module tests the MountainCar POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

import copy
import random

import numpy as np
import pytest
import scipy.stats

from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


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
    """Test state transition model for different actions.

    Purpose: Validates that state transition model correctly updates car position and velocity for different actions

    Given: MountainCarPOMDP environment and test state [0.0, 0.0] with actions -1, 0, and 1
    When: env.sample_next_state is called for each action
    Then: All next states have correct 2D shape representing [position, velocity] and change based on applied action

    Test type: unit
    """
    # Test state transition
    state = np.array([0.0, 0.0])
    action = 0
    new_state = base_mountain_car_environment.sample_next_state(state=state, action=action)
    assert isinstance(new_state, np.ndarray)
    assert new_state.shape == (2,)

    # Test with different action
    action = 1
    new_state = base_mountain_car_environment.sample_next_state(state=state, action=action)
    assert isinstance(new_state, np.ndarray)
    assert new_state.shape == (2,)

    # Test with negative action
    action = -1
    new_state = base_mountain_car_environment.sample_next_state(state=state, action=action)
    assert isinstance(new_state, np.ndarray)
    assert new_state.shape == (2,)


def test_state_transition_produces_varying_samples(base_mountain_car_environment):
    """Test that state transition with noise produces varying samples.

    Purpose: Validates that the stochastic state transition model produces
    different samples due to Gaussian process noise

    Given: A MountainCarPOMDP environment and initial state [-0.5, 0.0] with action 1
    When: Multiple samples are drawn via env.sample_next_state
    Then: Not all samples are identical, confirming stochastic behavior

    Test type: unit
    """
    state = np.array([-0.5, 0.0])
    action = 1
    samples = base_mountain_car_environment.sample_next_state(
        state=state, action=action, n_samples=50
    )

    assert len(samples) == 50
    assert all(s.shape == (2,) for s in samples)

    sample_array = np.array(samples)
    assert not np.all(
        sample_array == sample_array[0]
    ), "All samples are identical — noise is not being applied"


def test_state_transition_noise_magnitude(base_mountain_car_environment):
    """Test that state transition noise magnitude is reasonable.

    Purpose: Validates that noisy samples cluster around the deterministic next state

    Given: A MountainCarPOMDP environment with default noise covariance
    When: Many samples are drawn via env.sample_next_state
    Then: Sample mean is close to deterministic next state computed via the native
    kernel and standard deviations match the expected noise levels from the
    covariance matrix

    Test type: unit
    """
    # pylint: disable=import-outside-toplevel
    from POMDPPlanners.environments.mountain_car_pomdp import _native

    env = base_mountain_car_environment
    state = np.array([-0.5, 0.0])
    action = 1
    samples = np.array(env.sample_next_state(state=state, action=action, n_samples=5000))

    deterministic_kernel = _native.MountainCarTransitionCpp(
        state=state,
        action=action,
        power=env.power,
        gravity=env.gravity,
        max_speed=env.max_speed,
        min_position=env.min_position,
        max_position=env.max_position,
        covariance=env.state_transition_cov,
    )
    # pylint: disable=protected-access
    deterministic = deterministic_kernel._compute_deterministic_next_state()
    sample_mean = samples.mean(axis=0)
    np.testing.assert_allclose(sample_mean, deterministic, atol=0.005)

    expected_std = np.sqrt(np.diag(env.state_transition_cov))
    sample_std = samples.std(axis=0)
    np.testing.assert_allclose(sample_std, expected_std, rtol=0.3)


def test_state_transition_default_covariance():
    """Test that default state transition covariance is used when not specified.

    Purpose: Validates that MountainCarPOMDP uses the default state transition
    covariance matrix when none is provided

    Given: A MountainCarPOMDP environment created without explicit state_transition_cov
    When: The state_transition_cov attribute is checked
    Then: It matches the class-level DEFAULT_STATE_TRANSITION_COV

    Test type: unit
    """
    env = MountainCarPOMDP(discount_factor=0.95)
    np.testing.assert_array_equal(
        env.state_transition_cov, MountainCarPOMDP.DEFAULT_STATE_TRANSITION_COV
    )


def test_state_transition_custom_covariance():
    """Test that custom state transition covariance is applied correctly.

    Purpose: Validates that a custom state transition covariance matrix is
    stored and used when explicitly provided

    Given: A MountainCarPOMDP environment created with a custom state_transition_cov
    When: The state_transition_cov attribute is checked
    Then: It matches the custom covariance matrix provided at construction

    Test type: unit
    """
    custom_cov = np.diag([1e-4, 1e-5])
    env = MountainCarPOMDP(
        discount_factor=0.95,
        state_transition_cov=custom_cov,
    )
    np.testing.assert_array_equal(env.state_transition_cov, custom_cov)


def test_observation_model(base_mountain_car_environment):
    """Test observation model with noise addition.

    Purpose: Validates that observation model correctly adds noise to position and velocity measurements

    Given: MountainCarPOMDP environment and test state [0.0, 0.0] with action 0
    When: env.sample_observation is called
    Then: All observations have correct 2D shape and contain noise in both position and velocity components

    Test type: unit
    """
    # Test observation model
    state = np.array([0.0, 0.0])
    action = 0
    obs = base_mountain_car_environment.sample_observation(next_state=state, action=action)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)

    # Test multiple observations
    observations = base_mountain_car_environment.sample_observation(
        next_state=state, action=action, n_samples=100
    )
    assert len(observations) == 100
    assert all(isinstance(obs, np.ndarray) and obs.shape == (2,) for obs in observations)


def test_observation_model_probability_single_observation(base_mountain_car_environment):
    """Test observation log-probability with single observation.

    Purpose: Validates that observation_log_probability correctly handles single observation input

    Given: MountainCarPOMDP environment with specific next_state and action
    When: env.observation_log_probability is called with a single observation
    Then: Returns numpy array with shape (1,) and exponentiated probability is positive

    Test type: unit
    """
    # Test single observation probability
    state = np.array([0.0, 0.0])
    action = 0

    # Create a single observation close to true state
    observation = np.array([0.05, 0.01])  # Small deviation from true state [0.0, 0.0]

    # Get probability via env-level API
    log_probabilities = base_mountain_car_environment.observation_log_probability(
        next_state=state, action=action, observations=[observation]
    )
    probabilities = np.exp(log_probabilities)

    # Verify output type and shape
    assert isinstance(probabilities, np.ndarray), "Probability should return numpy array"
    assert probabilities.shape == (
        1,
    ), f"Single observation should return shape (1,), got {probabilities.shape}"
    assert probabilities[0] > 0.0, "Probability should be positive"
    assert isinstance(probabilities[0], (float, np.floating)), "Probability value should be float"


def test_observation_model_probability_multiple_observations(base_mountain_car_environment):
    """Test observation log-probability with multiple observations.

    Purpose: Validates that observation_log_probability correctly handles multiple observations

    Given: MountainCarPOMDP environment with specific next_state and action
    When: env.observation_log_probability is called with multiple observations
    Then: Returns numpy array with shape (n,) of decreasing probabilities with distance

    Test type: unit
    """
    # Test multiple observations probability
    state = np.array([0.0, 0.0])
    action = 0

    # Create multiple observations at different distances from true state
    observations = [
        np.array([0.0, 0.0]),  # Exactly at true state (highest probability)
        np.array([0.05, 0.01]),  # Close to true state
        np.array([0.1, 0.02]),  # Medium distance
        np.array([0.5, 0.1]),  # Far from true state (lowest probability)
    ]

    # Get probabilities via env-level API
    log_probabilities = base_mountain_car_environment.observation_log_probability(
        next_state=state, action=action, observations=observations
    )
    probabilities = np.exp(log_probabilities)

    # Verify output type and shape
    assert isinstance(probabilities, np.ndarray), "Probability should return numpy array"
    assert probabilities.shape == (
        4,
    ), f"Four observations should return shape (4,), got {probabilities.shape}"

    # All probabilities should be positive
    assert np.all(probabilities > 0.0), "All probabilities should be positive"

    # Probabilities should decrease with distance from true state
    assert probabilities[0] > probabilities[1], "Probability at true state should be highest"
    assert probabilities[1] > probabilities[2], "Closer observation should have higher probability"
    assert probabilities[2] > probabilities[3], "Closer observation should have higher probability"


def test_observation_model_probability_mathematical_correctness(base_mountain_car_environment):
    """Test observation_log_probability mathematical correctness.

    Purpose: Validates that observation_log_probability computes correct probability
    densities matching the multivariate normal distribution centered at next_state

    Given: MountainCarPOMDP environment with known covariance matrix and true next_state
    When: env.observation_log_probability is called with specific observations
    Then: Exponentiated log-probabilities match scipy multivariate_normal.pdf calculations

    Test type: unit
    """
    env = base_mountain_car_environment
    true_state = np.array([0.0, 0.0])
    action = 0
    cov_matrix = env.cov_matrix

    # Test with observation at true state (should have highest probability)
    observation_at_mean = true_state.copy()
    prob_at_mean = float(
        np.exp(
            env.observation_log_probability(
                next_state=true_state, action=action, observations=[observation_at_mean]
            )
        )[0]
    )

    expected_prob_at_mean = scipy.stats.multivariate_normal.pdf(
        observation_at_mean, mean=true_state, cov=cov_matrix
    )

    assert np.isclose(
        prob_at_mean, expected_prob_at_mean, rtol=1e-10
    ), f"Probability at mean {prob_at_mean} should match expected {expected_prob_at_mean}"

    # Test with observation away from mean
    observation_offset = true_state + np.array([0.1, 0.01])
    prob_offset = float(
        np.exp(
            env.observation_log_probability(
                next_state=true_state, action=action, observations=[observation_offset]
            )
        )[0]
    )

    expected_prob_offset = scipy.stats.multivariate_normal.pdf(
        observation_offset, mean=true_state, cov=cov_matrix
    )

    assert np.isclose(
        prob_offset, expected_prob_offset, rtol=1e-10
    ), f"Probability at offset {prob_offset} should match expected {expected_prob_offset}"

    # Verify that probability at mean is higher than probability at offset
    assert prob_at_mean > prob_offset, "Probability at mean should be higher than at offset"


def test_observation_model_probability_edge_cases(base_mountain_car_environment):
    """Test observation log-probability edge cases.

    Purpose: Validates that observation_log_probability handles edge cases correctly

    Given: MountainCarPOMDP environment with specific next_state and action
    When: env.observation_log_probability is called with edge cases (empty list, extreme values)
    Then: Function handles edge cases gracefully and returns appropriate results

    Test type: unit
    """
    state = np.array([0.0, 0.0])
    action = 0

    # Test empty list (should return empty array)
    empty_log_probs = base_mountain_car_environment.observation_log_probability(
        next_state=state, action=action, observations=[]
    )
    empty_probs = np.exp(empty_log_probs)
    assert isinstance(empty_probs, np.ndarray), "Empty list should return numpy array"
    assert empty_probs.shape == (
        0,
    ), f"Empty list should return shape (0,), got {empty_probs.shape}"

    # Test with extreme observation values (should still return positive probabilities)
    extreme_observations = [
        np.array([10.0, 10.0]),  # Very far from true state
        np.array([-10.0, -10.0]),  # Very far in opposite direction
        np.array([0.0, 0.0]),  # At true state
    ]

    extreme_log_probs = base_mountain_car_environment.observation_log_probability(
        next_state=state, action=action, observations=extreme_observations
    )
    extreme_probs = np.exp(extreme_log_probs)
    assert isinstance(extreme_probs, np.ndarray), "Extreme values should return numpy array"
    assert extreme_probs.shape == (
        3,
    ), f"Three extreme observations should return shape (3,), got {extreme_probs.shape}"

    # All probabilities should be non-negative (Gaussian has non-zero probability everywhere, but can be numerically zero)
    assert np.all(
        extreme_probs >= 0.0
    ), "All probabilities should be non-negative even for extreme values"

    # Probability at true state should be highest
    assert (
        extreme_probs[2] > extreme_probs[0]
    ), "Probability at true state should be higher than extreme offset"
    assert (
        extreme_probs[2] > extreme_probs[1]
    ), "Probability at true state should be higher than extreme offset"


def test_observation_model_probability_batch_consistency(base_mountain_car_environment):
    """Test observation log-probability batch consistency.

    Purpose: Validates that observation_log_probability produces consistent results
    when called multiple times with same inputs

    Given: MountainCarPOMDP environment with specific next_state and action
    When: env.observation_log_probability is called multiple times with identical observations
    Then: All calls return identical probability values, demonstrating deterministic behavior

    Test type: unit
    """
    state = np.array([0.0, 0.0])
    action = 0

    # Create test observations
    observations = [
        np.array([0.0, 0.0]),
        np.array([0.1, 0.01]),
        np.array([-0.05, -0.005]),
    ]

    # Call probability function multiple times via env-level API
    probs1 = np.exp(
        base_mountain_car_environment.observation_log_probability(
            next_state=state, action=action, observations=observations
        )
    )
    probs2 = np.exp(
        base_mountain_car_environment.observation_log_probability(
            next_state=state, action=action, observations=observations
        )
    )
    probs3 = np.exp(
        base_mountain_car_environment.observation_log_probability(
            next_state=state, action=action, observations=observations
        )
    )

    # All results should be identical
    assert np.array_equal(probs1, probs2), "Multiple calls should return identical results"
    assert np.array_equal(probs2, probs3), "Multiple calls should return identical results"
    assert np.array_equal(probs1, probs3), "Multiple calls should return identical results"


def test_observation_model_probability_different_states(base_mountain_car_environment):
    """Test observation log-probability with different true states.

    Purpose: Validates that observation_log_probability works correctly
    with different true states and maintains proper probability relationships

    Given: MountainCarPOMDP environment with different true next_states
    When: env.observation_log_probability is called with observations relative to each true state
    Then: Probabilities are highest for observations closest to their respective true states

    Test type: unit
    """
    # Test with different true states
    states = [
        np.array([0.0, 0.0]),
        np.array([0.2, 0.01]),
        np.array([-0.3, -0.02]),
    ]

    for state in states:
        action = 0

        # Create observations at different distances from this true state
        observations = [
            state.copy(),  # At true state
            state + np.array([0.05, 0.005]),  # Close to true state
            state + np.array([0.2, 0.02]),  # Far from true state
        ]

        probabilities = np.exp(
            base_mountain_car_environment.observation_log_probability(
                next_state=state, action=action, observations=observations
            )
        )

        # Verify shape and type
        assert isinstance(probabilities, np.ndarray), "Should return numpy array"
        assert probabilities.shape == (3,), f"Should return shape (3,), got {probabilities.shape}"

        # Verify probability relationships
        assert probabilities[0] > probabilities[1], "Probability at true state should be highest"
        assert (
            probabilities[1] > probabilities[2]
        ), "Closer observation should have higher probability"

        # All probabilities should be positive
        assert np.all(probabilities > 0.0), "All probabilities should be positive"


def test_initial_state_distribution(base_mountain_car_environment):
    """Test initial state distribution sampling.

    Purpose: Validates that initial state distribution generates valid 2D states

    Given: MountainCarPOMDP environment
    When: Initial state distribution is sampled
    Then: Returns valid initial state with correct 2D shape representing [position, velocity]

    Test type: unit
    """
    # Test initial state distribution
    dist = base_mountain_car_environment.initial_state_dist()
    initial_state = dist.sample()[0]
    assert isinstance(initial_state, np.ndarray)
    assert initial_state.shape == (2,)


def test_initial_observation_distribution(base_mountain_car_environment):
    """Test initial observation distribution sampling.

    Purpose: Validates that initial observation distribution generates valid 2D observations

    Given: MountainCarPOMDP environment
    When: Initial observation distribution is sampled
    Then: Returns valid initial observation with correct 2D shape representing [position, velocity]

    Test type: unit
    """
    # Test initial observation distribution
    dist = base_mountain_car_environment.initial_observation_dist()
    initial_observation = dist.sample()[0]
    assert isinstance(initial_observation, np.ndarray)
    assert initial_observation.shape == (2,)


def test_sample_next_step(base_mountain_car_environment):
    """Test complete environment step simulation.

    Purpose: Validates that sample_next_step correctly simulates car dynamics, observations, and rewards

    Given: MountainCarPOMDP environment, initial state [0.0, 0.0], and different actions (-1, 0, 1)
    When: sample_next_step is called for each action
    Then: Returns valid next_state, observation, and reward with correct types, shapes, and physics-based state changes

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
    """Test reward function for different car positions.

    Purpose: Validates that reward function correctly penalizes non-goal positions and rewards goal achievement

    Given: MountainCarPOMDP environment with goal_position=0.5
    When: Reward is calculated for positions below goal (0.0), at goal (0.5), and past goal (0.6)
    Then: Below goal returns -1.0, at goal returns 0.0, past goal returns 0.0

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
    """Test terminal state detection based on goal position.

    Purpose: Validates that environment correctly identifies terminal states when car reaches or passes goal

    Given: MountainCarPOMDP environment with goal_position=0.5
    When: is_terminal is called for positions below goal (0.0), at goal (0.5), and past goal (0.6)
    Then: Below goal returns False, at goal returns True, past goal returns True

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
    """Test action space retrieval.

    Purpose: Validates that environment provides correct discrete action space for car control

    Given: MountainCarPOMDP environment
    When: get_actions method is called
    Then: Returns list of 3 actions [-1, 0, 1] representing left, no action, and right respectively

    Test type: unit
    """
    pomdp = MountainCarPOMDP(discount_factor=0.95)
    actions = pomdp.get_actions()

    assert len(actions) == 3
    assert -1 in actions
    assert 0 in actions
    assert 1 in actions


def test_reward_range():
    """Test that reward range is correctly set.

    Purpose: Validates that MountainCarPOMDP has the correct reward range parameters

    Given: A MountainCarPOMDP environment with default configuration
    When: Environment reward_range attribute is checked
    Then: Returns (-1.0, 0.0) representing the minimum (per-step penalty) and maximum (goal achievement) rewards

    Test type: unit
    """
    pomdp = MountainCarPOMDP(discount_factor=0.95)
    assert pomdp.reward_range == (-1.0, 0.0)

    # Verify the actual rewards match the range
    non_goal_state = (0.0, 0.0)  # Below goal position
    goal_state = (pomdp.goal_position, 0.0)  # At goal position
    past_goal_state = (pomdp.goal_position + 0.1, 0.0)  # Past goal position

    # Test all possible rewards
    non_goal_reward = pomdp.reward(non_goal_state, 0)
    goal_reward = pomdp.reward(goal_state, 0)
    past_goal_reward = pomdp.reward(past_goal_state, 0)

    min_reward = min(non_goal_reward, goal_reward, past_goal_reward)
    max_reward = max(non_goal_reward, goal_reward, past_goal_reward)

    assert pomdp.reward_range is not None
    assert min_reward == pomdp.reward_range[0]  # Should be -1.0
    assert max_reward == pomdp.reward_range[1]  # Should be 0.0


def test_mountain_car_state_bounds():
    """Test state boundary enforcement in transitions.

    Purpose: Validates that state transitions respect position and velocity bounds defined by environment parameters

    Given: MountainCarPOMDP environment with min_position=-1.2, max_position=0.6, max_speed=0.07
    When: State transitions are sampled from states outside bounds
    Then: Resulting states are clamped to valid ranges: position within [-1.2, 0.6], velocity within [-0.07, 0.07]

    Test type: unit
    """
    pomdp = MountainCarPOMDP(discount_factor=0.95)

    # Test position bounds
    state = (pomdp.min_position - 0.1, 0.0)
    transition = pomdp.sample_next_state(state=state, action=0)
    assert transition[0] >= pomdp.min_position

    state = (pomdp.max_position + 0.1, 0.0)
    transition = pomdp.sample_next_state(state=state, action=0)
    assert transition[0] <= pomdp.max_position

    # Test velocity bounds
    state = (0.0, pomdp.max_speed + 0.1)
    transition = pomdp.sample_next_state(state=state, action=0)
    assert abs(transition[1]) <= pomdp.max_speed

    state = (0.0, -pomdp.max_speed - 0.1)
    transition = pomdp.sample_next_state(state=state, action=0)
    assert abs(transition[1]) <= pomdp.max_speed


@pytest.fixture
def base_mountain_car_environment() -> MountainCarPOMDP:
    """Create base MountainCarPOMDP environment for testing.

    Purpose: Provides a consistent test environment with standard parameters

    Given: Standard discount factor of 0.95
    When: Fixture is used in tests
    Then: Returns MountainCarPOMDP instance with min_position=-1.2, max_position=0.6, max_speed=0.07, goal_position=0.5

    Test type: fixture
    """
    return MountainCarPOMDP(discount_factor=0.95)


class TestMountainCarPOMDPEquality:
    """Test that MountainCarPOMDPs with different configurations are not equal.

    Purpose: Validates that environment equality comparison correctly identifies different configurations

    Given: MountainCarPOMDP environments with various parameter differences
    When: Equality comparison is performed
    Then: Environments with different parameters are correctly identified as unequal

    Test type: unit
    """

    def test_same_discount_factor(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that MountainCarPOMDPs with same discount factor are equal.

        Purpose: Validates that environment equality comparison works correctly for identical configurations

        Given: Two MountainCarPOMDP environments with identical discount_factor=0.95
        When: Equality comparison is performed
        Then: Both environments are equal, confirming symmetry of equality relation

        Test type: unit
        """
        other_env = MountainCarPOMDP(discount_factor=0.95)
        assert base_mountain_car_environment == other_env
        assert other_env == base_mountain_car_environment  # Test symmetry

    def test_different_discount_factor(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that MountainCarPOMDPs with different discount factors are not equal.

        Purpose: Validates that environment equality comparison correctly identifies different configurations

        Given: MountainCarPOMDP with discount_factor=0.95 and another with discount_factor=0.8
        When: Equality comparison is performed
        Then: Environments are not equal, confirming symmetry of inequality relation

        Test type: unit
        """
        other_env = MountainCarPOMDP(discount_factor=0.8)
        assert base_mountain_car_environment != other_env
        assert other_env != base_mountain_car_environment  # Test symmetry

    def test_different_parameters(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that MountainCarPOMDPs with different parameters are not equal.

        Purpose: Validates that environment equality comparison detects parameter differences

        Given: MountainCarPOMDP with standard parameters and modified versions with different min_position, max_position, max_speed, goal_position
        When: Equality comparison is performed
        Then: Modified environments are not equal to base environment, confirming parameter sensitivity

        Test type: unit
        """
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

    def test_different_noise_parameters(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that MountainCarPOMDPs with different noise parameters are not equal.

        Purpose: Validates that environment equality comparison detects noise parameter differences

        Given: MountainCarPOMDP with standard parameters and modified versions with different power, gravity, position_noise, velocity_noise
        When: Equality comparison is performed
        Then: Modified environments are not equal to base environment, confirming noise parameter sensitivity

        Test type: unit
        """
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
        """Test that MountainCarPOMDPs with different actions are not equal.

        Purpose: Validates that environment equality comparison detects action space differences

        Given: MountainCarPOMDP with standard actions [-1, 0, 1] and modified version with actions [-1, 0, 1, 2]
        When: Equality comparison is performed
        Then: Modified environment is not equal to base environment, confirming action space sensitivity

        Test type: unit
        """
        other_env = MountainCarPOMDP(discount_factor=0.95)
        other_env.actions = [-1, 0, 1, 2]  # Different from [-1, 0, 1]
        assert base_mountain_car_environment != other_env

    def test_comparison_with_non_environment(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test comparison with non-Environment objects.

        Purpose: Validates that environment equality comparison handles non-environment objects correctly

        Given: MountainCarPOMDP environment and non-environment objects (string, integer, None)
        When: Equality comparison is performed
        Then: All comparisons return False, confirming proper type checking

        Test type: unit
        """
        assert base_mountain_car_environment != "not an environment"
        assert base_mountain_car_environment != 42
        assert base_mountain_car_environment is not None

    def test_missing_attributes(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test equality when attributes are missing.

        Purpose: Validates that environment equality comparison handles missing attributes gracefully

        Given: MountainCarPOMDP environment and modified version with deleted attributes (min_position, cov_matrix)
        When: Equality comparison is performed
        Then: Modified environment is not equal to base environment, confirming attribute completeness requirement

        Test type: unit
        """
        other_env = MountainCarPOMDP(discount_factor=0.95)
        delattr(other_env, "min_position")
        assert base_mountain_car_environment != other_env

        other_env = MountainCarPOMDP(discount_factor=0.95)
        delattr(other_env, "cov_matrix")
        assert base_mountain_car_environment != other_env

    def test_deep_copy_equality(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that a deep copy of MountainCarPOMDP is equal to original.

        Purpose: Validates that deep copying preserves environment equality

        Given: MountainCarPOMDP environment and its deep copy
        When: Equality comparison is performed between original and copy
        Then: Both environments are equal, confirming deep copy preserves all attributes

        Test type: unit
        """
        copied_env = copy.deepcopy(base_mountain_car_environment)
        assert copied_env == base_mountain_car_environment
        assert base_mountain_car_environment == copied_env  # Test symmetry


class TestMountainCarPOMDPConfigId:
    """Test that config_id changes with different configurations.

    Purpose: Validates that config_id generates unique identifiers for different MountainCarPOMDP configurations

    Given: MountainCarPOMDP environments with various parameter differences
    When: Config IDs are generated and compared
    Then: Different configurations have different config_ids, demonstrating uniqueness and consistency

    Test type: configuration
    """

    def test_config_id_consistency(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that config_id is consistent for identical environments.

        Purpose: Validates that config_id generates consistent identifiers for identical configurations

        Given: Two MountainCarPOMDP environments with identical parameters
        When: Config IDs are generated for both environments
        Then: Both environments have identical config_ids, demonstrating consistency for same configuration

        Test type: configuration
        """
        other_env = MountainCarPOMDP(discount_factor=0.95)
        assert base_mountain_car_environment.config_id == other_env.config_id

    def test_config_id_different_discount_factor(
        self, base_mountain_car_environment: MountainCarPOMDP
    ):
        """Test that config_id changes with different discount factor.

        Purpose: Validates that config_id generates different identifiers for different discount factors

        Given: MountainCarPOMDP with discount_factor=0.95 and another with discount_factor=0.8
        When: Config IDs are generated for both environments
        Then: Environments have different config_ids, demonstrating uniqueness for different configurations

        Test type: configuration
        """
        other_env = MountainCarPOMDP(discount_factor=0.8)
        assert base_mountain_car_environment.config_id != other_env.config_id

    def test_config_id_different_parameters(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that config_id changes with different parameters.

        Purpose: Validates that config_id generates different identifiers for different parameter configurations

        Given: MountainCarPOMDP with standard parameters and modified versions with different noise parameters
        When: Config IDs are generated for all environments
        Then: Modified environments have different config_ids from base environment, demonstrating parameter sensitivity

        Test type: configuration
        """
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
        """Test that config_id is a valid SHA-256 hash.

        Purpose: Validates that config_id generates properly formatted SHA-256 hash identifiers

        Given: MountainCarPOMDP environment with specific configuration
        When: Config ID is generated for the environment
        Then: Returns a 64-character string containing only valid hexadecimal characters (0-9, a-f)

        Test type: configuration
        """
        config_id = base_mountain_car_environment.config_id
        assert isinstance(config_id, str)
        assert len(config_id) == 64  # SHA-256 hash length
        assert all(c in "0123456789abcdef" for c in config_id)  # Valid hex characters

    def test_config_id_deterministic(self, base_mountain_car_environment: MountainCarPOMDP):
        """Test that config_id is deterministic (same input always produces same output).

        Purpose: Validates that config_id generates deterministic identifiers for identical configurations

        Given: MountainCarPOMDP environment with specific configuration
        When: Config ID is generated multiple times for the same environment
        Then: All generated config_ids are identical, demonstrating deterministic behavior

        Test type: configuration
        """
        config_id1 = base_mountain_car_environment.config_id
        config_id2 = base_mountain_car_environment.config_id
        assert config_id1 == config_id2


def test_get_metric_names():
    """Test that get_metric_names returns goal_reaching_rate.

    Purpose: Validates that MountainCarPOMDP returns the correct metric names

    Given: A MountainCarPOMDP environment
    When: get_metric_names is called
    Then: Returns list containing "goal_reaching_rate"

    Test type: unit
    """
    pomdp = MountainCarPOMDP(discount_factor=0.95)
    metric_names = pomdp.get_metric_names()
    assert "goal_reaching_rate" in metric_names
    assert len(metric_names) == 1


def test_compute_metrics_goal_reaching():
    """Test computation of goal-reaching metrics for different simulation histories.

    Purpose: Validates that MountainCarPOMDP computes goal-reaching rate correctly

    Given: Three simulation histories - 2 reaching goal, 1 not reaching goal
    When: compute_metrics analyzes the simulation histories
    Then: Returns goal_reaching_rate=2/3 with confidence bounds

    Test type: unit
    """
    from POMDPPlanners.core.policy import PolicyRunData
    from POMDPPlanners.core.simulation import History, StepData

    pomdp = MountainCarPOMDP(discount_factor=0.95)

    # Create a simple belief for testing
    def create_test_belief(state):
        from POMDPPlanners.core.belief import WeightedParticleBelief

        return WeightedParticleBelief(
            particles=[state], log_weights=np.array([1.0]), resampling=False
        )

    # History 1: Reaches goal (position >= 0.5)
    history1 = History(
        [
            StepData(
                state=np.array([0.0, 0.0]),
                action=1,
                next_state=np.array([0.1, 0.01]),
                observation=np.array([0.1, 0.01]),
                reward=-1.0,
                belief=create_test_belief(np.array([0.0, 0.0])),
            ),
            StepData(
                state=np.array([0.5, 0.02]),  # At goal position
                action=1,
                next_state=np.array([0.6, 0.03]),
                observation=np.array([0.6, 0.03]),
                reward=0.0,
                belief=create_test_belief(np.array([0.5, 0.02])),
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

    # History 2: Reaches goal (position > 0.5)
    history2 = History(
        [
            StepData(
                state=np.array([0.0, 0.0]),
                action=1,
                next_state=np.array([0.1, 0.01]),
                observation=np.array([0.1, 0.01]),
                reward=-1.0,
                belief=create_test_belief(np.array([0.0, 0.0])),
            ),
            StepData(
                state=np.array([0.6, 0.02]),  # Past goal position
                action=1,
                next_state=np.array([0.7, 0.03]),
                observation=np.array([0.7, 0.03]),
                reward=0.0,
                belief=create_test_belief(np.array([0.6, 0.02])),
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

    # History 3: Does not reach goal (position < 0.5)
    history3 = History(
        [
            StepData(
                state=np.array([0.0, 0.0]),
                action=-1,
                next_state=np.array([-0.1, -0.01]),
                observation=np.array([-0.1, -0.01]),
                reward=-1.0,
                belief=create_test_belief(np.array([0.0, 0.0])),
            ),
            StepData(
                state=np.array([-0.2, -0.02]),  # Far from goal
                action=-1,
                next_state=np.array([-0.3, -0.03]),
                observation=np.array([-0.3, -0.03]),
                reward=-1.0,
                belief=create_test_belief(np.array([-0.2, -0.02])),
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

    # Compute metrics
    metrics = pomdp.compute_metrics([history1, history2, history3])

    # Convert metrics to dictionary for easier access
    metrics_dict = {metric.name: metric for metric in metrics}

    # Test goal reaching rate
    assert "goal_reaching_rate" in metrics_dict
    goal_rate = metrics_dict["goal_reaching_rate"]
    assert goal_rate.value == 2 / 3  # 2 out of 3 histories reach goal
    assert goal_rate.lower_confidence_bound <= goal_rate.value <= goal_rate.upper_confidence_bound


def test_reward_batch_matches_scalar_reward():
    """Test that reward_batch returns results consistent with scalar reward.

    Purpose: Validates that the vectorized reward_batch gives identical outputs
    to calling reward() individually for each state.

    Given: A MountainCarPOMDP environment and an array of states with mixed goal/non-goal positions
    When: reward_batch is called with the state array
    Then: Output shape is (N,) and values match element-wise reward() calls exactly

    Test type: unit
    """
    env = MountainCarPOMDP(discount_factor=0.95)
    np.random.seed(42)
    # Create states with positions spanning below and above goal_position
    states = np.column_stack(
        [
            np.random.uniform(-1.2, 0.7, 100),
            np.random.uniform(-0.07, 0.07, 100),
        ]
    )
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
    """Test that MountainCar native simulate_random_rollout matches the base-class Python loop.

    Purpose: Validates that the native C++ rollout produces a discounted return
    equal (within atol=1e-9) to the base-class Python loop when both use the
    same action sequence and the same C++ RNG seed.

    Given: A MountainCarPOMDP env seeded identically for C++ RNG and numpy; a fixed
        action sequence pre-drawn from numpy seed 0; a non-terminal initial state;
        max_depth=10, discount_factor=0.95, depth=0.
    When: Both the native override and the Python base-class loop are run with the
        same action sequence and the same C++ RNG seed before each call.
    Then: The two discounted returns are equal within atol=1e-9.

    Test type: unit
    """
    from POMDPPlanners.environments.mountain_car_pomdp import (
        _native,
    )  # pylint: disable=import-outside-toplevel
    from POMDPPlanners.planners.planners_utils.dpw import (
        ActionSampler,
    )  # pylint: disable=import-outside-toplevel
    from POMDPPlanners.planners.planners_utils.rollout import (
        python_random_rollout,
    )  # pylint: disable=import-outside-toplevel

    env = MountainCarPOMDP(discount_factor=0.99)
    state = np.array([-0.5, 0.0], dtype=np.float64)
    max_depth = 10
    gamma = 0.95

    # Pre-draw a fixed action sequence (indices into [-1, 0, 1]) with numpy seed 0
    np.random.seed(0)
    fixed_action_indices = np.random.randint(0, 3, size=max_depth, dtype=np.int32)
    fixed_actions = [env.actions[int(i)] for i in fixed_action_indices]

    class _SequenceActionSampler(ActionSampler):
        def __init__(self, actions: list) -> None:
            self._actions = actions
            self._idx = 0

        def sample(self, belief_node=None) -> int:  # pylint: disable=unused-argument
            a = self._actions[self._idx % len(self._actions)]
            self._idx += 1
            return a

    # Run native rollout: seed numpy so randint draws the same index sequence
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

    Purpose: Pins the post-fix contract for MountainCarPOMDP that
        ``observation_log_probability`` (scalar) and
        ``observation_log_probability_per_state`` (batch) agree on a
        moderate-density anchor whose analytic log-probability is well
        below the old ``log(p + 1e-300) ≈ -690.776`` floor but still
        above the kernel's internal float64 underflow threshold.
        Pre-fix, the scalar path floored such values at ~-690.776 while
        the batch path returned the un-floored kernel log-likelihood —
        the asymmetry that motivated the env-wide log-prob floor
        removal.

    Given: A default MountainCarPOMDP env (position_noise=0.1,
        velocity_noise=0.01), a fixed next_state at the origin, action 0,
        and an observation at (3.78, 0.0). The analytic 2-D Gaussian
        log-pdf at this offset is ≈ -709.350.
    When: Both ``observation_log_probability`` and
        ``observation_log_probability_per_state`` are evaluated on the
        same (next_state, action, observation).
    Then: Both return finite, equal values to within atol=1e-6, and the
        common value is below -700 (past the old floor).

    Test type: unit
    """
    env = MountainCarPOMDP(discount_factor=0.95)
    next_state = np.array([0.0, 0.0])
    action = 0
    observation = np.array([3.78, 0.0])

    scalar = env.observation_log_probability(next_state, action, [observation])[0]
    batch = env.observation_log_probability_per_state(np.array([next_state]), action, observation)[
        0
    ]

    assert np.isfinite(scalar), f"scalar should be finite at this anchor, got {scalar}"
    assert np.isfinite(batch), f"batch should be finite at this anchor, got {batch}"
    assert (
        scalar < -700.0
    ), f"anchor must be below the old -690.776 floor to exercise the fix; got {scalar}"
    np.testing.assert_allclose(scalar, batch, atol=1e-6)
