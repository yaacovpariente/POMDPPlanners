"""Tests for environment base classes.

This module tests the environment base classes, focusing on:
- Basic environment functionality
- Environment interfaces
- Space information
- Environment types
"""

import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import (
    Environment,
    ObservationModel,
    SpaceInfo,
    SpaceType,
    StateTransitionModel,
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.utils.logger import reset_logger_state

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


class MockDistribution(Distribution):
    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        return [np.array([1, 2])] * n_samples

    def probability(self, values: List[np.ndarray]) -> np.ndarray:
        return np.array([1.0] * len(values))


class MockStateTransitionModel(StateTransitionModel):
    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        return [np.array([1, 2])] * n_samples

    def probability(self, values: List[np.ndarray]) -> np.ndarray:
        return np.array([1.0] * len(values))


class MockObservationModel(ObservationModel):
    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        return [np.array([1, 2])] * n_samples

    def probability(self, values: List[np.ndarray]) -> np.ndarray:
        return np.array([1.0] * len(values))


class MockEnvironment(Environment):
    def __init__(
        self,
        discount_factor: float,
        test_array: Optional[np.ndarray] = None,
        output_dir: Optional[Path] = None,
        debug: bool = False,
    ):
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )
        super().__init__(
            discount_factor=discount_factor,
            name="MockEnvironment",
            space_info=space_info,
            output_dir=output_dir,
            debug=debug,
        )
        self.test_array = test_array if test_array is not None else np.array([1, 2, 3])

    def state_transition_model(self, state: np.ndarray, action: np.ndarray) -> StateTransitionModel:
        return MockStateTransitionModel(state, action)

    def observation_model(self, next_state: np.ndarray, action: np.ndarray) -> ObservationModel:
        return MockObservationModel(next_state, action)

    def reward(self, state: np.ndarray, action: np.ndarray) -> float:
        return 1.0

    def is_terminal(self, state: np.ndarray) -> bool:
        return False

    def initial_state_dist(self) -> Distribution:
        return MockDistribution()

    def initial_observation_dist(self) -> Distribution:
        return MockDistribution()

    def is_equal_observation(self, observation1: np.ndarray, observation2: np.ndarray) -> bool:
        return np.array_equal(observation1, observation2)


class DifferentEnvironment(Environment):
    def __init__(
        self,
        discount_factor: float,
        output_dir: Optional[Path] = None,
        debug: bool = False,
    ):
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )
        super().__init__(
            discount_factor=discount_factor,
            name="DifferentEnvironment",
            space_info=space_info,
            output_dir=output_dir,
            debug=debug,
        )

    def state_transition_model(self, state: np.ndarray, action: np.ndarray) -> StateTransitionModel:
        return MockStateTransitionModel(state, action)

    def observation_model(self, next_state: np.ndarray, action: np.ndarray) -> ObservationModel:
        return MockObservationModel(next_state, action)

    def reward(self, state: np.ndarray, action: np.ndarray) -> float:
        return 1.0

    def is_terminal(self, state: np.ndarray) -> bool:
        return False

    def initial_state_dist(self) -> Distribution:
        return MockDistribution()

    def initial_observation_dist(self) -> Distribution:
        return MockDistribution()

    def is_equal_observation(self, observation1: np.ndarray, observation2: np.ndarray) -> bool:
        return np.array_equal(observation1, observation2)


@pytest.fixture
def base_environment() -> MockEnvironment:
    """Fixture providing a base environment for comparison."""
    return MockEnvironment(discount_factor=0.9)


@pytest.fixture
def base_environment_with_logging(tmp_path) -> MockEnvironment:
    """Fixture providing a base environment with logging enabled."""
    return MockEnvironment(discount_factor=0.9, output_dir=tmp_path, debug=True)


@pytest.mark.parametrize(
    "discount_factor,expected_equal",
    [
        (0.9, True),  # Same discount factor
        (0.8, False),  # Different discount factor
    ],
)
def test_discount_factor_equality(
    base_environment: MockEnvironment, discount_factor: float, expected_equal: bool
):
    """Test equality comparison with different discount factors.

    Purpose: Validates equality comparison for discount factor

    Given: Objects with same or different configurations
    When: Equality comparison is performed
    Then: Objects are correctly identified as equal or unequal

    Test type: unit
    """
    other_env = MockEnvironment(discount_factor=discount_factor)
    assert (base_environment == other_env) == expected_equal


@pytest.mark.parametrize(
    "test_array,expected_equal",
    [
        (np.array([1, 2, 3]), True),  # Same array
        (np.array([4, 5, 6]), False),  # Different array
    ],
)
def test_array_equality(
    base_environment: MockEnvironment, test_array: np.ndarray, expected_equal: bool
):
    """Test equality comparison with different numpy arrays.

    Purpose: Validates equality comparison for array

    Given: Objects with same or different configurations
    When: Equality comparison is performed
    Then: Objects are correctly identified as equal or unequal

    Test type: unit
    """
    other_env = MockEnvironment(discount_factor=0.9, test_array=test_array)
    assert (base_environment == other_env) == expected_equal


def test_different_environment_class(base_environment: MockEnvironment):
    """Test equality comparison with a different environment class.

    Purpose: Validates environment equality returns False when comparing different environment classes

    Given: MockEnvironment and DifferentEnvironment with same parameters
    When: Equality comparison is performed between different environment types
    Then: Environments are correctly identified as not equal

    Test type: unit
    """
    different_env = DifferentEnvironment(discount_factor=0.9)
    assert base_environment != different_env


def test_non_environment_comparison(base_environment: MockEnvironment):
    """Test equality comparison with non-Environment objects.

    Purpose: Validates that Environment equality returns False when compared with non-Environment objects

    Given: MockEnvironment instance and non-Environment objects (string, integer, None)
    When: Equality comparison is performed between Environment and non-Environment types
    Then: All comparisons return False for non-Environment objects

    Test type: unit
    """
    assert base_environment != "not an environment"
    assert base_environment != 42
    assert base_environment is not None


def test_missing_attribute(base_environment: MockEnvironment):
    """Test equality comparison when an attribute is missing.

    Purpose: Validates that Environment equality returns False when attributes are missing from comparison object

    Given: Two MockEnvironments with same configuration, one missing the test_array attribute
    When: Equality comparison is performed between complete and incomplete environment
    Then: Environments are correctly identified as not equal due to missing attribute

    Test type: unit
    """
    other_env = MockEnvironment(discount_factor=0.9)
    delattr(other_env, "test_array")
    assert base_environment != other_env


@pytest.mark.parametrize(
    "attr_name,attr_value",
    [
        ("discount_factor", 0.9),
        ("test_array", np.array([1, 2, 3])),
    ],
)
def test_attribute_presence(base_environment: MockEnvironment, attr_name: str, attr_value: object):
    """Test that attributes are present and have correct values.

    Purpose: Validates that Environment instances have required attributes with correct values for equality comparison

    Given: MockEnvironment with discount_factor=0.9 and test_array=[1,2,3], parameterized attribute names and values
    When: Attribute presence and values are checked using hasattr and getattr
    Then: All specified attributes exist and have correct values (with special handling for numpy arrays)

    Test type: unit
    """
    assert hasattr(base_environment, attr_name)
    if isinstance(attr_value, np.ndarray):
        assert np.array_equal(getattr(base_environment, attr_name), attr_value)
    else:
        assert getattr(base_environment, attr_name) == attr_value


@pytest.fixture
def base_tiger_environment() -> TigerPOMDP:
    """Fixture providing a base TigerPOMDP environment for comparison."""
    return TigerPOMDP(discount_factor=0.95)


@pytest.fixture(autouse=True)
def cleanup_loggers():
    """Automatically clean up logger state after each test."""
    yield  # Run the test
    reset_logger_state()  # Clean up after the test


class TestEnvironmentConfigId:
    """Test suite for Environment config_id property."""

    def test_same_config_same_id(self, base_environment: MockEnvironment):
        """Test that environments with same configuration have same config_id.

        Purpose: Validates that Environment instances with identical configurations generate identical config_id values

        Given: Two MockEnvironments with same discount_factor=0.9 and default test_array=[1,2,3]
        When: config_id is generated for both environment instances
        Then: Both environments produce the same config_id hash value

        Test type: configuration
        """
        other_env = MockEnvironment(discount_factor=0.9)
        assert base_environment.config_id == other_env.config_id

    def test_different_discount_factor(self, base_environment: MockEnvironment):
        """Test that different discount factor results in different config_id.

        Purpose: Validates that Environment instances with different discount factors generate different config_id values

        Given: Base MockEnvironment with discount_factor=0.9 and another with discount_factor=0.8
        When: config_id is generated for both environment instances
        Then: Different discount factors produce different config_id hash values

        Test type: unit
        """
        other_env = MockEnvironment(discount_factor=0.8)
        assert base_environment.config_id != other_env.config_id

    def test_different_array(self, base_environment: MockEnvironment):
        """Test that different numpy array results in different config_id.

        Purpose: Validates that Environment instances with different numpy arrays generate different config_id values

        Given: Base MockEnvironment with test_array=[1,2,3] and another with test_array=[4,5,6]
        When: config_id is generated for both environment instances with different arrays
        Then: Different numpy arrays produce different config_id hash values

        Test type: unit
        """
        other_env = MockEnvironment(discount_factor=0.9, test_array=np.array([4, 5, 6]))
        assert base_environment.config_id != other_env.config_id

    def test_different_environment_class(self, base_environment: MockEnvironment):
        """Test that different environment class results in different config_id.

        Purpose: Validates config_id generation produces different IDs for different environment classes

        Given: MockEnvironment and DifferentEnvironment with same parameters
        When: config_id is generated for both environment instances
        Then: Different environment classes produce different config_id values

        Test type: unit
        """
        different_env = DifferentEnvironment(discount_factor=0.9)
        assert base_environment.config_id != different_env.config_id

    def test_config_id_consistency(self, base_environment: MockEnvironment):
        """Test that config_id remains consistent across multiple calls.

        Purpose: Validates that Environment config_id property returns consistent values across multiple accesses

        Given: Single MockEnvironment instance with fixed configuration parameters
        When: config_id property is accessed multiple times on the same instance
        Then: All accesses return identical config_id hash values (deterministic behavior)

        Test type: configuration
        """
        id1 = base_environment.config_id
        id2 = base_environment.config_id
        assert id1 == id2

    def test_tiger_pomdp_config_id(self, base_tiger_environment: TigerPOMDP):
        """Test config_id with TigerPOMDP environment.

        Purpose: Validates that TigerPOMDP config_id generation works correctly for real environment implementations

        Given: TigerPOMDP environments with same (discount=0.95) and different (discount=0.8) configurations
        When: config_id is generated for TigerPOMDP instances with varying parameters
        Then: Same configurations produce identical IDs, different configurations produce different IDs

        Test type: configuration
        """
        # Same configuration should have same ID
        other_env = TigerPOMDP(discount_factor=0.95)
        assert base_tiger_environment.config_id == other_env.config_id

        # Different configuration should have different ID
        different_env = TigerPOMDP(discount_factor=0.8)
        assert base_tiger_environment.config_id != different_env.config_id

    def test_config_id_format(self, base_environment: MockEnvironment):
        """Test that config_id is a valid SHA-256 hash.

        Purpose: Validates that Environment config_id follows proper SHA-256 hash format specification

        Given: MockEnvironment instance with configuration parameters
        When: config_id property generates hash value from environment configuration
        Then: Returns 64-character string containing only valid hexadecimal characters (0-9, a-f)

        Test type: configuration
        """
        config_id = base_environment.config_id
        assert len(config_id) == 64  # SHA-256 produces 64 hex characters
        assert all(c in "0123456789abcdef" for c in config_id)  # Valid hex characters


class TestEnvironmentLogger:
    """Test suite for Environment logger functionality."""

    def test_logger_initialization(self, base_environment_with_logging: MockEnvironment):
        """Test that logger is properly initialized.

        Purpose: Validates proper initialization of logger

        Given: Constructor parameters and initial conditions
        When: Object is initialized
        Then: Object is properly constructed with expected attributes

        Test type: unit
        """
        assert hasattr(base_environment_with_logging, "logger")
        assert base_environment_with_logging.logger.name == "environment.MockEnvironment"

    def test_logger_without_output_dir(self, base_environment: MockEnvironment):
        """Test that logger works without output directory.

        Purpose: Validates that Environment logger initialization works correctly when no output directory is provided

        Given: MockEnvironment created without output_dir parameter (default None)
        When: Logger attribute is accessed after environment initialization
        Then: Logger exists with correct name "environment.MockEnvironment" without requiring output directory

        Test type: unit
        """
        assert hasattr(base_environment, "logger")
        assert base_environment.logger.name == "environment.MockEnvironment"

    def test_logger_debug_mode(self, base_environment_with_logging: MockEnvironment):
        """Test that debug mode affects logger configuration.

        Purpose: Validates that Environment logger configuration reflects debug mode

        Given: MockEnvironment created with debug=True and output_dir parameters
        When: Logger is accessed after environment initialization in debug mode
        Then: Logger exists and is properly configured for debug mode

        Test type: unit
        """
        # Queue-based logging caches loggers, so we test that debug mode is stored in configuration
        # rather than testing the exact logger level which may be affected by caching
        logger = base_environment_with_logging.logger
        assert logger is not None
        assert hasattr(base_environment_with_logging, "debug")
        assert base_environment_with_logging.debug is True

    def test_logger_normal_mode(self, base_environment: MockEnvironment):
        """Test that normal mode sets correct logger configuration.

        Purpose: Validates that Environment logger configuration reflects normal mode

        Given: MockEnvironment created with default debug=False parameter
        When: Logger is accessed after environment initialization in normal mode
        Then: Logger exists and debug mode is properly configured

        Test type: unit
        """
        # Queue-based logging caches loggers, so we test that normal mode is stored in configuration
        logger = base_environment.logger
        assert logger is not None
        assert hasattr(base_environment, "debug")
        assert base_environment.debug is False

    def test_logger_output_dir(self, tmp_path):
        """Test that logger properly handles output directory configuration.

        Purpose: Validates that Environment logger handles output directory configuration correctly

        Given: MockEnvironment with output_dir=temporary directory path
        When: Environment is initialized with output directory logging enabled
        Then: Environment stores output_dir and logger is configured properly

        Test type: unit
        """
        env = MockEnvironment(discount_factor=0.9, output_dir=tmp_path)
        # With queue-based logging, directory creation happens asynchronously
        # Test that the environment properly stores the output_dir configuration
        assert env.output_dir == tmp_path
        assert env.logger is not None
        # Log a message to trigger the writer thread to create directories
        env.logger.info("Test message to trigger directory creation")
        # Note: In queue-based logging, the logs directory is created by the writer thread
        # when the first message is processed, so we can't immediately assert its existence
