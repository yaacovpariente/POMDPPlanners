"""Tests for MountainCar POMDP environment.

This module tests the MountainCar POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

import pytest
import numpy as np
import random
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.core.simulation import StepData

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
    When: State transition model is called and next states are sampled
    Then: All next states have correct 2D shape representing [position, velocity] and change based on applied action

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
    """Test observation model with noise addition.

    Purpose: Validates that observation model correctly adds noise to position and velocity measurements

    Given: MountainCarPOMDP environment and test state [0.0, 0.0] with action 0
    When: Observation model is called and observations are sampled
    Then: All observations have correct 2D shape and contain noise in both position and velocity components

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
    observations = base_mountain_car_environment.observation_model(
        state, action
    ).sample(n_samples=100)
    assert len(observations) == 100
    assert all(
        isinstance(obs, np.ndarray) and obs.shape == (2,) for obs in observations
    )


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
    next_state, observation, reward = base_mountain_car_environment.sample_next_step(
        state, action
    )

    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (2,)
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (2,)
    assert isinstance(reward, float)

    # Test with different action
    action = 1
    next_state, observation, reward = base_mountain_car_environment.sample_next_step(
        state, action
    )

    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (2,)
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (2,)
    assert isinstance(reward, float)

    # Test with negative action
    action = -1
    next_state, observation, reward = base_mountain_car_environment.sample_next_step(
        state, action
    )

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

    def test_same_discount_factor(
        self, base_mountain_car_environment: MountainCarPOMDP
    ):
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

    def test_different_discount_factor(
        self, base_mountain_car_environment: MountainCarPOMDP
    ):
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

    def test_different_parameters(
        self, base_mountain_car_environment: MountainCarPOMDP
    ):
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

    def test_different_noise_parameters(
        self, base_mountain_car_environment: MountainCarPOMDP
    ):
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
        other_env.cov_matrix = np.array(
            [[0.2**2, 0], [0, 0.02**2]]
        )  # Different from original
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

    def test_comparison_with_non_environment(
        self, base_mountain_car_environment: MountainCarPOMDP
    ):
        """Test comparison with non-Environment objects.

        Purpose: Validates that environment equality comparison handles non-environment objects correctly

        Given: MountainCarPOMDP environment and non-environment objects (string, integer, None)
        When: Equality comparison is performed
        Then: All comparisons return False, confirming proper type checking

        Test type: unit
        """
        assert base_mountain_car_environment != "not an environment"
        assert base_mountain_car_environment != 42
        assert base_mountain_car_environment != None

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
        import copy

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

    def test_config_id_consistency(
        self, base_mountain_car_environment: MountainCarPOMDP
    ):
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

    def test_config_id_different_parameters(
        self, base_mountain_car_environment: MountainCarPOMDP
    ):
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

    def test_config_id_deterministic(
        self, base_mountain_car_environment: MountainCarPOMDP
    ):
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
