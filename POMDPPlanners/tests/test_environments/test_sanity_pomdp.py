"""Tests for Sanity POMDP environment.

This module tests the Sanity POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

import pytest
import numpy as np
import random
from POMDPPlanners.environments.sanity_pomdp import (
    SanityPOMDP,
    SanityStateTransitionModel,
    SanityObservationModel,
    SanityInitialStateDist,
    SanityInitialObservationDist,
)
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.core.policy import PolicyRunData, PolicyInfoVariable

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


@pytest.fixture
def sanity_pomdp():
    return SanityPOMDP(discount_factor=0.95)


@pytest.fixture
def sanity_pomdp_debug():
    return SanityPOMDP(discount_factor=0.95, debug=True)


class TestSanityPOMDPInitialization:
    """Test suite for SanityPOMDP initialization."""

    def test_initialization(self, sanity_pomdp):
        """Test SanityPOMDP environment initialization with default parameters.

        Purpose: Validates SanityPOMDP environment initializes correctly with default discount factor and debug settings

        Given: SanityPOMDP constructed with default parameters (discount_factor=0.95, debug=False)
        When: Environment instance is created
        Then: Discount factor, name, space types, and debug mode are correctly set

        Test type: unit
        """
        assert sanity_pomdp.discount_factor == 0.95
        assert sanity_pomdp.name == "SanityPOMDP"
        assert sanity_pomdp.space_info.action_space.value == "discrete"
        assert sanity_pomdp.space_info.observation_space.value == "discrete"
        assert sanity_pomdp.debug is False

    def test_initialization_with_debug(self, sanity_pomdp_debug):
        """Test initialization with debug mode.

        Purpose: Validates that SanityPOMDP can be initialized with debug mode enabled

        Given: SanityPOMDP constructor with debug=True parameter
        When: Environment instance is created with debug mode
        Then: Environment has debug attribute set to True, enabling debug functionality

        Test type: unit
        """
        assert sanity_pomdp_debug.debug is True

    def test_initialization_with_output_dir(self):
        """Test initialization with output directory.

        Purpose: Validates that SanityPOMDP can be initialized with custom output directory

        Given: SanityPOMDP constructor with output_dir parameter pointing to /tmp/test_output
        When: Environment instance is created with output directory
        Then: Environment has output_dir attribute set to the specified path

        Test type: unit
        """
        from pathlib import Path

        output_dir = Path("/tmp/test_output")
        env = SanityPOMDP(discount_factor=0.95, output_dir=output_dir)
        assert env.output_dir == output_dir


class TestSanityPOMDPEquality:
    """Test suite for SanityPOMDP equality comparisons."""

    def test_same_environment_equality(self, sanity_pomdp):
        """Test that identical environments are equal.

        Purpose: Validates equality comparison for same environment

        Given: Objects with same or different configurations
        When: Equality comparison is performed
        Then: Objects are correctly identified as equal or unequal

        Test type: unit
        """
        other_env = SanityPOMDP(discount_factor=0.95)
        assert sanity_pomdp == other_env
        assert other_env == sanity_pomdp  # Test symmetry

    def test_different_discount_factor(self, sanity_pomdp):
        """Test that environments with different discount factors are not equal.

        Purpose: Validates that environment equality comparison correctly identifies different discount factors

        Given: SanityPOMDP with discount_factor=0.95 and another with discount_factor=0.8
        When: Equality comparison is performed
        Then: Environments are not equal, confirming discount factor sensitivity

        Test type: unit
        """
        other_env = SanityPOMDP(discount_factor=0.8)
        assert sanity_pomdp != other_env
        assert other_env != sanity_pomdp  # Test symmetry

    def test_different_debug_mode(self, sanity_pomdp):
        """Test that environments with different debug modes are not equal.

        Purpose: Validates that environment equality comparison correctly identifies different debug modes

        Given: SanityPOMDP with debug=False and another with debug=True
        When: Equality comparison is performed
        Then: Environments are not equal, confirming debug mode sensitivity

        Test type: unit
        """
        other_env = SanityPOMDP(discount_factor=0.95, debug=True)
        assert sanity_pomdp != other_env
        assert other_env != sanity_pomdp  # Test symmetry

    def test_comparison_with_non_environment(self, sanity_pomdp):
        """Test comparison with non-Environment objects.

        Purpose: Validates that environment equality comparison handles non-environment objects correctly

        Given: SanityPOMDP environment and non-environment objects (string, integer, None)
        When: Equality comparison is performed
        Then: All comparisons return False, confirming proper type checking

        Test type: unit
        """
        assert sanity_pomdp != "not an environment"
        assert sanity_pomdp != 42
        assert sanity_pomdp != None


class TestSanityPOMDPConfigId:
    """Test suite for SanityPOMDP config_id functionality."""

    def test_config_id_consistency(self, sanity_pomdp):
        """Test that config_id is consistent for identical environments.

        Purpose: Validates that SanityPOMDP config_id generates consistent identifiers for identical configurations

        Given: Two SanityPOMDP environments with identical discount_factor=0.95 and debug=False
        When: Config IDs are generated for both environments
        Then: Both environments have identical config_ids, demonstrating consistency for same configuration

        Test type: configuration
        """
        other_env = SanityPOMDP(discount_factor=0.95)
        assert sanity_pomdp.config_id == other_env.config_id

    def test_config_id_different_discount_factor(self, sanity_pomdp):
        """Test that config_id changes with different discount factors.

        Purpose: Validates that SanityPOMDP config_id generates different identifiers for different discount factors

        Given: Two SanityPOMDP environments with different discount factors (0.95 vs 0.8)
        When: Config IDs are generated for both environments
        Then: Environments have different config_ids, demonstrating uniqueness for different configurations

        Test type: configuration
        """
        other_env = SanityPOMDP(discount_factor=0.8)
        assert sanity_pomdp.config_id != other_env.config_id

    def test_config_id_different_debug_mode(self, sanity_pomdp):
        """Test that config_id changes with different debug modes.

        Purpose: Validates that SanityPOMDP config_id generates different identifiers for different debug modes

        Given: Two SanityPOMDP environments with different debug modes (False vs True)
        When: Config IDs are generated for both environments
        Then: Environments have different config_ids, demonstrating uniqueness for different debug configurations

        Test type: configuration
        """
        other_env = SanityPOMDP(discount_factor=0.95, debug=True)
        assert sanity_pomdp.config_id != other_env.config_id

    def test_config_id_format(self, sanity_pomdp):
        """Test that config_id is a valid SHA-256 hash.

        Purpose: Validates that SanityPOMDP config_id generates properly formatted SHA-256 hash identifiers

        Given: A SanityPOMDP environment with specific configuration
        When: Config ID is generated for the environment
        Then: Returns a 64-character string containing only valid hexadecimal characters (0-9, a-f)

        Test type: configuration
        """
        config_id = sanity_pomdp.config_id
        assert isinstance(config_id, str)
        assert len(config_id) == 64  # SHA-256 hash length
        assert all(c in "0123456789abcdef" for c in config_id)  # Valid hex characters

    def test_config_id_deterministic(self, sanity_pomdp):
        """Test that config_id is deterministic (same input always produces same output).

        Purpose: Validates that SanityPOMDP config_id generates deterministic identifiers for identical configurations

        Given: A SanityPOMDP environment with specific configuration
        When: Config ID is generated multiple times for the same environment
        Then: All generated config_ids are identical, demonstrating deterministic behavior

        Test type: configuration
        """
        config_id1 = sanity_pomdp.config_id
        config_id2 = sanity_pomdp.config_id
        assert config_id1 == config_id2


class TestSanityPOMDPActions:
    """Test suite for SanityPOMDP action-related functionality."""

    def test_get_actions(self, sanity_pomdp):
        """Test that get_actions returns the correct actions.

        Purpose: Validates that SanityPOMDP get_actions method returns the correct action space

        Given: A SanityPOMDP environment with default configuration
        When: get_actions method is called
        Then: Returns list [0, 1] representing the two available actions in the environment

        Test type: unit
        """
        actions = sanity_pomdp.get_actions()
        assert actions == [0, 1]
        assert len(actions) == 2


class TestSanityPOMDPReward:
    """Test suite for SanityPOMDP reward-related functionality."""

    def test_reward_range(self, sanity_pomdp):
        """Test that reward range is correctly set.

        Purpose: Validates that SanityPOMDP has the correct reward range parameters

        Given: A SanityPOMDP environment with default configuration
        When: Environment reward_range attribute is checked
        Then: Returns (0.0, 1.0) representing the minimum (bad state) and maximum (good state) rewards

        Test type: unit
        """
        assert sanity_pomdp.reward_range == (0.0, 1.0)

        # Verify the actual rewards match the range
        # Test reward based on next state (after state transition)
        reward_from_state_0_action_0 = sanity_pomdp.reward(
            state=0, action=0
        )  # Action 0 leads to state 0 -> reward 1.0
        reward_from_state_0_action_1 = sanity_pomdp.reward(
            state=0, action=1
        )  # Action 1 leads to state 1 -> reward 0.0
        reward_from_state_1_action_0 = sanity_pomdp.reward(
            state=1, action=0
        )  # Action 0 leads to state 0 -> reward 1.0
        reward_from_state_1_action_1 = sanity_pomdp.reward(
            state=1, action=1
        )  # Action 1 leads to state 1 -> reward 0.0

        all_rewards = [
            reward_from_state_0_action_0,
            reward_from_state_0_action_1,
            reward_from_state_1_action_0,
            reward_from_state_1_action_1,
        ]

        min_reward = min(all_rewards)
        max_reward = max(all_rewards)

        assert min_reward == sanity_pomdp.reward_range[0]  # Should be 0.0
        assert max_reward == sanity_pomdp.reward_range[1]  # Should be 1.0


class TestSanityStateTransitionModel:
    """Test suite for SanityStateTransitionModel."""

    def test_initialization(self):
        """Test SanityStateTransitionModel initialization with specific state-action pair.

        Purpose: Validates SanityStateTransitionModel initializes correctly with specified state and action parameters

        Given: State transition model created with state=0 and action=1 parameters
        When: SanityStateTransitionModel instance is constructed
        Then: Model stores correct state and action values for deterministic transitions

        Test type: unit
        """
        model = SanityStateTransitionModel(state=0, action=1)
        assert model.state == 0
        assert model.action == 1

    def test_sample_action_0(self):
        """Test sampling with action 0 (should always lead to state 0).

        Purpose: Validates state transition sampling with action 0 produces deterministic state 0 transitions

        Given: SanityStateTransitionModel configured with initial state=0 and action=0
        When: Sample method is called for 10 samples
        Then: All samples return state 0 (deterministic transition for action 0)

        Test type: unit
        """
        model = SanityStateTransitionModel(state=0, action=0)
        samples = model.sample(n_samples=10)
        assert all(s == 0 for s in samples)
        assert len(samples) == 10

    def test_sample_action_1(self):
        """Test sampling with action 1 (should always lead to state 1).

        Purpose: Validates state transition sampling with action 1 produces deterministic state 1 transitions

        Given: SanityStateTransitionModel configured with initial state=0 and action=1
        When: Sample method is called for 10 samples
        Then: All samples return state 1 (deterministic transition for action 1)

        Test type: unit
        """
        model = SanityStateTransitionModel(state=0, action=1)
        samples = model.sample(n_samples=10)
        assert all(s == 1 for s in samples)
        assert len(samples) == 10

    def test_sample_different_states(self):
        """Test that state doesn't affect transition (only action matters).

        Purpose: Validates sampling behavior for  different states

        Given: Configured object with sampling capabilities
        When: Sample method is called
        Then: Valid samples are returned according to distribution

        Test type: unit
        """
        # Action 0 should always lead to state 0 regardless of current state
        for state in [0, 1]:
            model = SanityStateTransitionModel(state=state, action=0)
            samples = model.sample(n_samples=5)
            assert all(s == 0 for s in samples)

        # Action 1 should always lead to state 1 regardless of current state
        for state in [0, 1]:
            model = SanityStateTransitionModel(state=state, action=1)
            samples = model.sample(n_samples=5)
            assert all(s == 1 for s in samples)

    def test_probability_action_0(self):
        """Test probability calculation for action 0.

        Purpose: Validates state transition probabilities for action 0 (deterministically leads to state 0)

        Given: SanityStateTransitionModel with action=0 and test values [0,1,0,1]
        When: Probability method is called with the test values
        Then: Returns [1.0,0.0,1.0,0.0] probabilities (1.0 for state 0, 0.0 for state 1)

        Test type: unit
        """
        model = SanityStateTransitionModel(state=0, action=0)
        values = [0, 1, 0, 1]
        probs = model.probability(values)
        expected = np.array([1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(probs, expected)

    def test_probability_action_1(self):
        """Test probability calculation for action 1.

        Purpose: Validates state transition probabilities for action 1 (deterministically leads to state 1)

        Given: SanityStateTransitionModel with action=1 and test values [0,1,0,1]
        When: Probability method is called with the test values
        Then: Returns [0.0,1.0,0.0,1.0] probabilities (0.0 for state 0, 1.0 for state 1)

        Test type: unit
        """
        model = SanityStateTransitionModel(state=0, action=1)
        values = [0, 1, 0, 1]
        probs = model.probability(values)
        expected = np.array([0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(probs, expected)


class TestSanityObservationModel:
    """Test suite for SanityObservationModel."""

    def test_initialization(self):
        """Test SanityObservationModel initialization with specific state-action pair.

        Purpose: Validates SanityObservationModel initializes correctly with next_state and action parameters

        Given: Observation model created with next_state=1 and action=0 parameters
        When: SanityObservationModel instance is constructed
        Then: Model stores correct next_state and action values for deterministic observations

        Test type: unit
        """
        model = SanityObservationModel(next_state=1, action=0)
        assert model.next_state == 1
        assert model.action == 0

    def test_sample_state_0(self):
        """Test sampling with state 0 (should always observe 0).

        Purpose: Validates observation sampling from state 0 produces deterministic observation 0

        Given: SanityObservationModel configured with next_state=0 and action=0
        When: Sample method is called for 10 samples
        Then: All samples return observation 0 (deterministic observation for state 0)

        Test type: unit
        """
        model = SanityObservationModel(next_state=0, action=0)
        samples = model.sample(n_samples=10)
        assert all(s == 0 for s in samples)
        assert len(samples) == 10

    def test_sample_state_1(self):
        """Test sampling with state 1 (should always observe 1).

        Purpose: Validates observation sampling from state 1 produces deterministic observation 1

        Given: SanityObservationModel configured with next_state=1 and action=0
        When: Sample method is called for 10 samples
        Then: All samples return observation 1 (deterministic observation for state 1)

        Test type: unit
        """
        model = SanityObservationModel(next_state=1, action=0)
        samples = model.sample(n_samples=10)
        assert all(s == 1 for s in samples)
        assert len(samples) == 10

    def test_sample_different_actions(self):
        """Test that action doesn't affect observation (only state matters).

        Purpose: Validates sampling behavior for  different actions

        Given: Configured object with sampling capabilities
        When: Sample method is called
        Then: Valid samples are returned according to distribution

        Test type: unit
        """
        # State 0 should always give observation 0 regardless of action
        for action in [0, 1]:
            model = SanityObservationModel(next_state=0, action=action)
            samples = model.sample(n_samples=5)
            assert all(s == 0 for s in samples)

        # State 1 should always give observation 1 regardless of action
        for action in [0, 1]:
            model = SanityObservationModel(next_state=1, action=action)
            samples = model.sample(n_samples=5)
            assert all(s == 1 for s in samples)

    def test_probability_state_0(self):
        """Test probability calculation for state 0.

        Purpose: Validates observation probabilities for state 0 (deterministically observes observation 0)

        Given: SanityObservationModel with next_state=0 and test values [0,1,0,1]
        When: Probability method is called with the test values
        Then: Returns [1.0,0.0,1.0,0.0] probabilities (1.0 for obs 0, 0.0 for obs 1)

        Test type: unit
        """
        model = SanityObservationModel(next_state=0, action=0)
        values = [0, 1, 0, 1]
        probs = model.probability(values)
        expected = np.array([1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(probs, expected)

    def test_probability_state_1(self):
        """Test probability calculation for state 1.

        Purpose: Validates observation probabilities for state 1 (deterministically observes observation 1)

        Given: SanityObservationModel with next_state=1 and test values [0,1,0,1]
        When: Probability method is called with the test values
        Then: Returns [0.0,1.0,0.0,1.0] probabilities (0.0 for obs 0, 1.0 for obs 1)

        Test type: unit
        """
        model = SanityObservationModel(next_state=1, action=0)
        values = [0, 1, 0, 1]
        probs = model.probability(values)
        expected = np.array([0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(probs, expected)


class TestSanityInitialStateDist:
    """Test suite for SanityInitialStateDist."""

    def test_sample(self):
        """Test that initial state distribution always returns state 0.

        Purpose: Validates initial state distribution sampling produces deterministic state 0 samples

        Given: SanityInitialStateDist configured for deterministic initial state
        When: Sample method is called for 10 samples
        Then: All samples return state 0 (deterministic initial state distribution)

        Test type: unit
        """
        dist = SanityInitialStateDist()
        samples = dist.sample(n_samples=10)
        assert all(s == 0 for s in samples)
        assert len(samples) == 10

    def test_probability(self):
        """Test probability calculation for initial state distribution.

        Purpose: Validates initial state distribution probabilities (1.0 for state 0, 0.0 for state 1)

        Given: SanityInitialStateDist and test values [0,1,0,1]
        When: Probability method is called with the test values
        Then: Returns [1.0,0.0,1.0,0.0] probabilities reflecting deterministic state 0 distribution

        Test type: unit
        """
        dist = SanityInitialStateDist()
        values = [0, 1, 0, 1]
        probs = dist.probability(values)
        expected = np.array([1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(probs, expected)


class TestSanityInitialObservationDist:
    """Test suite for SanityInitialObservationDist."""

    def test_sample(self):
        """Test that initial observation distribution always returns observation 0.

        Purpose: Validates initial observation distribution sampling produces deterministic observation 0 samples

        Given: SanityInitialObservationDist configured for deterministic initial observation
        When: Sample method is called for 10 samples
        Then: All samples return observation 0 (deterministic initial observation distribution)

        Test type: unit
        """
        dist = SanityInitialObservationDist()
        samples = dist.sample(n_samples=10)
        assert all(s == 0 for s in samples)
        assert len(samples) == 10

    def test_probability(self):
        """Test probability calculation for initial observation distribution.

        Purpose: Validates initial observation distribution probabilities (1.0 for obs 0, 0.0 for obs 1)

        Given: SanityInitialObservationDist and test values [0,1,0,1]
        When: Probability method is called with the test values
        Then: Returns [1.0,0.0,1.0,0.0] probabilities reflecting deterministic observation 0 distribution

        Test type: unit
        """
        dist = SanityInitialObservationDist()
        values = [0, 1, 0, 1]
        probs = dist.probability(values)
        expected = np.array([1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(probs, expected)


class TestSanityPOMDPModels:
    """Test suite for SanityPOMDP model creation."""

    def test_state_transition_model(self, sanity_pomdp):
        """Test state transition model creation.

        Purpose: Validates that SanityPOMDP can create proper state transition models

        Given: A SanityPOMDP environment and state=0, action=1 parameters
        When: state_transition_model method is called with the parameters
        Then: Returns SanityStateTransitionModel instance with correct state and action attributes

        Test type: unit
        """
        model = sanity_pomdp.state_transition_model(state=0, action=1)
        assert isinstance(model, SanityStateTransitionModel)
        assert model.state == 0
        assert model.action == 1

    def test_observation_model(self, sanity_pomdp):
        """Test observation model creation.

        Purpose: Validates that SanityPOMDP can create proper observation models

        Given: A SanityPOMDP environment and next_state=1, action=0 parameters
        When: observation_model method is called with the parameters
        Then: Returns SanityObservationModel instance with correct next_state and action attributes

        Test type: unit
        """
        model = sanity_pomdp.observation_model(next_state=1, action=0)
        assert isinstance(model, SanityObservationModel)
        assert model.next_state == 1
        assert model.action == 0

    def test_initial_state_dist(self, sanity_pomdp):
        """Test initial state distribution creation.

        Purpose: Validates that SanityPOMDP can create proper initial state distributions

        Given: A SanityPOMDP environment
        When: initial_state_dist method is called
        Then: Returns SanityInitialStateDist instance for generating initial states

        Test type: unit
        """
        dist = sanity_pomdp.initial_state_dist()
        assert isinstance(dist, SanityInitialStateDist)

    def test_initial_observation_dist(self, sanity_pomdp):
        """Test initial observation distribution creation.

        Purpose: Validates that SanityPOMDP can create proper initial observation distributions

        Given: A SanityPOMDP environment
        When: initial_observation_dist method is called
        Then: Returns SanityInitialObservationDist instance for generating initial observations

        Test type: unit
        """
        dist = sanity_pomdp.initial_observation_dist()
        assert isinstance(dist, SanityInitialObservationDist)


class TestSanityPOMDPTerminal:
    """Test suite for SanityPOMDP terminal state detection."""

    def test_is_terminal(self, sanity_pomdp):
        """Test that no states are terminal.

        Purpose: Validates that SanityPOMDP correctly identifies that no states are terminal

        Given: A SanityPOMDP environment and states [0, 1]
        When: is_terminal method is called for each state
        Then: All states return False, confirming that SanityPOMDP has no terminal states

        Test type: unit
        """
        for state in [0, 1]:
            assert not sanity_pomdp.is_terminal(state)


class TestSanityPOMDPObservationEquality:
    """Test suite for SanityPOMDP observation equality."""

    def test_is_equal_observation(self, sanity_pomdp):
        """Test observation equality comparison.

        Purpose: Validates that SanityPOMDP observation equality comparison works correctly

        Given: A SanityPOMDP environment and observation pairs (0,0), (1,1), (0,1), (1,0)
        When: is_equal_observation method is called for each observation pair
        Then: Returns True for identical observations (0,0) and (1,1), False for different observations (0,1) and (1,0)

        Test type: unit
        """
        assert sanity_pomdp.is_equal_observation(0, 0)
        assert sanity_pomdp.is_equal_observation(1, 1)
        assert not sanity_pomdp.is_equal_observation(0, 1)
        assert not sanity_pomdp.is_equal_observation(1, 0)


class TestSanityPOMDPSampleNextStep:
    """Test suite for SanityPOMDP sample_next_step functionality."""

    def test_sample_next_step_action_0(self, sanity_pomdp):
        """Test sample_next_step with action 0.

        Purpose: Validates environment step sampling with action 0 produces deterministic transition to state 0

        Given: SanityPOMDP environment and initial state=0, action=0
        When: sample_next_step method is called
        Then: Returns next_state=0, observation=0, reward=1.0 (deterministic action 0 outcome)

        Test type: unit
        """
        next_state, next_observation, reward = sanity_pomdp.sample_next_step(state=0, action=0)
        assert next_state == 0  # Action 0 leads to state 0
        assert next_observation == 0  # State 0 gives observation 0
        assert reward == 1.0  # State 0 gives reward 1.0

    def test_sample_next_step_action_1(self, sanity_pomdp):
        """Test sample_next_step with action 1.

        Purpose: Validates environment step sampling with action 1 produces deterministic transition to state 1

        Given: SanityPOMDP environment and initial state=0, action=1
        When: sample_next_step method is called
        Then: Returns next_state=1, observation=1, reward=0.0 (deterministic action 1 outcome)

        Test type: unit
        """
        next_state, next_observation, reward = sanity_pomdp.sample_next_step(state=0, action=1)
        assert next_state == 1  # Action 1 leads to state 1
        assert next_observation == 1  # State 1 gives observation 1
        assert reward == 0.0  # State 1 gives reward 0.0

    def test_sample_next_step_from_state_1(self, sanity_pomdp):
        """Test sample_next_step starting from state 1.

        Purpose: Validates that environment step sampling works correctly from different initial states

        Given: SanityPOMDP environment and initial state=1, action=0
        When: sample_next_step method is called
        Then: Returns next_state=0, observation=0, reward=1.0 (action 0 always leads to state 0 regardless of current state)

        Test type: unit
        """
        next_state, next_observation, reward = sanity_pomdp.sample_next_step(state=1, action=0)
        assert next_state == 0  # Action 0 leads to state 0
        assert next_observation == 0  # State 0 gives observation 0
        assert reward == 1.0  # State 1 gives reward 0.0 (reward is based on current state)


class TestSanityPOMDPMetrics:
    """Test suite for SanityPOMDP compute_metrics functionality."""

    def test_compute_metrics_empty_histories(self, sanity_pomdp):
        """Test metrics computation with empty histories.

        Purpose: Validates that SanityPOMDP compute_metrics handles empty history lists correctly

        Given: A SanityPOMDP environment and empty history list []
        When: compute_metrics method is called with empty histories
        Then: Returns empty list [], confirming that no metrics are computed for empty input

        Test type: unit
        """
        metrics = sanity_pomdp.compute_metrics([])
        assert metrics == []

    def test_compute_metrics_with_histories(self, sanity_pomdp):
        """Test metrics computation with sample histories.

        Purpose: Validates that SanityPOMDP compute_metrics handles non-empty history lists correctly

        Given: A SanityPOMDP environment and sample history with 2 steps (action 0 and action 1)
        When: compute_metrics method is called with the sample history
        Then: Returns empty list [], confirming that SanityPOMDP doesn't implement custom metrics

        Test type: unit
        """
        # Create a simple history
        steps = [
            StepData(state=0, action=0, next_state=0, observation=0, reward=1.0, belief=None),
            StepData(state=0, action=1, next_state=1, observation=1, reward=0.0, belief=None),
        ]
        history = History(
            history=steps,
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=2,
            reach_terminal_state=False,
            policy_run_data=PolicyRunData(info_variables=[]),
        )

        metrics = sanity_pomdp.compute_metrics([history])
        assert metrics == []  # SanityPOMDP doesn't implement custom metrics


class TestSanityPOMDPIntegration:
    """Integration tests for SanityPOMDP."""

    def test_full_episode_simulation(self, sanity_pomdp):
        """Test a full episode simulation.

        Purpose: Validates that SanityPOMDP can simulate complete episodes with proper state transitions and rewards

        Given: A SanityPOMDP environment with initial state and observation distributions
        When: Episode simulation is performed with actions 0 and 1
        Then: Action 0 leads to state 0 with reward 1.0, action 1 leads to state 1 with reward 0.0, demonstrating deterministic behavior

        Test type: unit
        """
        # Start with initial state
        initial_state_dist = sanity_pomdp.initial_state_dist()
        initial_obs_dist = sanity_pomdp.initial_observation_dist()

        state = initial_state_dist.sample()[0]
        observation = initial_obs_dist.sample()[0]

        assert state == 0
        assert observation == 0

        # Take action 0 (should lead to good state)
        next_state, next_observation, reward = sanity_pomdp.sample_next_step(state, 0)
        assert next_state == 0
        assert next_observation == 0
        assert reward == 1.0

        # Take action 1 (should lead to bad state)
        next_state, next_observation, reward = sanity_pomdp.sample_next_step(state, 1)
        assert next_state == 1
        assert next_observation == 1
        assert reward == 0.0

    def test_deterministic_behavior(self, sanity_pomdp):
        """Test that the environment behaves deterministically.

        Purpose: Validates that SanityPOMDP produces consistent, deterministic results across multiple samples

        Given: A SanityPOMDP environment and specific state-action pairs
        When: sample_next_step is called multiple times with identical parameters
        Then: All calls return identical results, confirming deterministic behavior for both action 0 and action 1

        Test type: unit
        """
        # Test multiple samples with same parameters
        for _ in range(10):
            next_state, next_observation, reward = sanity_pomdp.sample_next_step(0, 0)
            assert next_state == 0
            assert next_observation == 0
            assert reward == 1.0

        for _ in range(10):
            next_state, next_observation, reward = sanity_pomdp.sample_next_step(0, 1)
            assert next_state == 1
            assert next_observation == 1
            assert reward == 0.0
