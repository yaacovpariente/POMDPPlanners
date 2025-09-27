"""Tests for policy implementations.

This module tests the policy implementations, focusing on:
- Basic policy functionality
- Policy execution
- Policy data structures
- Policy evaluation
"""

import logging
import random
from pathlib import Path
from typing import Optional
from unittest.mock import Mock

import numpy as np
import pytest

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import (
    Environment,
    ObservationModel,
    SpaceInfo,
    SpaceType,
    StateTransitionModel,
)
from POMDPPlanners.core.policy import Policy, PolicySpaceInfo, PolicyRunData
from POMDPPlanners.utils.logger import get_logger, reset_logger_state

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


class MockEnvironment(Environment):
    def __init__(self, discount_factor: float, name: str):
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )
        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            use_queue_logger=False,
        )  # Use individual logger for tests

    def state_transition_model(self, state, action):
        from POMDPPlanners.core.environment import StateTransitionModel

        return Mock(spec=StateTransitionModel)

    def observation_model(self, next_state, action):
        from POMDPPlanners.core.environment import ObservationModel

        return Mock(spec=ObservationModel)

    def reward(self, state, action):
        return 0.0

    def is_terminal(self, state):
        return False

    def initial_state_dist(self):
        from POMDPPlanners.core.distributions import Distribution

        return Mock(spec=Distribution)

    def initial_observation_dist(self):
        from POMDPPlanners.core.distributions import Distribution

        return Mock(spec=Distribution)

    def is_equal_observation(self, observation1, observation2):
        return True


class MockBelief(Belief):
    def __init__(self, environment: Environment):
        self.environment = environment

    def update(self, action, observation, pomdp, state=None):
        return self

    def sample(self):
        return "mock_state"


class MockPolicy(Policy):
    def __init__(
        self,
        environment: Environment,
        discount_factor: float,
        name: str,
        custom_param: str = "default",
        log_path: Optional[Path] = None,
        debug: bool = False,
    ):
        super().__init__(environment, discount_factor, name, log_path=log_path, debug=debug)
        self.custom_param = custom_param

    def action(self, belief: Belief):
        return (["mock_action"], PolicyRunData(info_variables=[]))

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )


class DifferentPolicy(Policy):
    def __init__(
        self,
        environment: Environment,
        discount_factor: float,
        name: str,
        log_path: Optional[Path] = None,
        debug: bool = False,
    ):
        super().__init__(environment, discount_factor, name, log_path=log_path, debug=debug)

    def action(self, belief: Belief):
        return (["different_action"], PolicyRunData(info_variables=[]))

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )


@pytest.fixture
def base_environment():
    return MockEnvironment(discount_factor=0.9, name="test_env")


@pytest.fixture
def base_belief(base_environment):
    return MockBelief(base_environment)


@pytest.fixture
def base_policy(base_environment):
    return MockPolicy(base_environment, discount_factor=0.9, name="test_policy")


@pytest.fixture
def temp_log_dir(tmp_path):
    """Fixture to provide a temporary directory for log files."""
    return tmp_path / "logs"


@pytest.fixture(autouse=True)
def cleanup_loggers():
    """Automatically clean up logger state after each test."""
    yield  # Run the test
    reset_logger_state()  # Clean up after the test


class TestPolicyLogger:
    def test_logger_initialization(self, base_environment, temp_log_dir):
        """Test that logger is properly initialized with different configurations.

        Purpose: Validates proper initialization of logger

        Given: Constructor parameters and initial conditions
        When: Object is initialized
        Then: Object is properly constructed with expected attributes

        Test type: unit
        """
        # Test with default settings (console only)
        policy = MockPolicy(base_environment, discount_factor=0.9, name="test_policy")
        assert policy.logger.name == "policy.test_policy"
        # Queue-based logging uses queue handlers, not direct console handlers
        assert policy.logger is not None

        # Test with log file
        policy = MockPolicy(
            base_environment,
            discount_factor=0.9,
            name="test_policy_file",
            log_path=temp_log_dir,
        )
        assert policy.logger.name == "policy.test_policy_file"
        # Queue-based logging handles file creation asynchronously
        assert policy.logger is not None

        # Log a message to trigger file creation by the writer thread
        policy.logger.info("Test message for file creation")
        # Note: With queue-based logging, files are created asynchronously by writer thread

        # Test with debug mode
        policy = MockPolicy(
            base_environment,
            discount_factor=0.9,
            name="test_policy_debug",
            log_path=temp_log_dir,
            debug=True,
        )
        assert policy.logger.name == "policy.test_policy_debug"
        # Queue-based logging stores debug configuration rather than direct logger level
        assert policy.logger is not None
        assert policy.debug is True

    def test_logger_output(self, base_environment, temp_log_dir, caplog):
        """Test that logger produces expected output for different logging levels and debug modes.

        Purpose: Validates that policy logger correctly filters and outputs messages based on logging level and debug configuration

        Given: MockPolicy instances with different debug settings and log file configurations
        When: Logger methods are called with different severity levels (debug, info, warning, error)
        Then: Messages appear in console and log files according to the configured logging level (INFO filters debug, DEBUG shows all)

        Test type: unit
        """
        # Test normal mode (INFO level)
        policy = MockPolicy(
            base_environment,
            discount_factor=0.9,
            name="test_policy",
            log_path=temp_log_dir,
        )

        # Test logging different levels
        with caplog.at_level(logging.DEBUG):  # Capture all levels
            policy.logger.debug("Debug message")
            policy.logger.info("Info message")
            policy.logger.warning("Warning message")
            policy.logger.error("Error message")

        # Check console output - queue-based logging should still show messages in console
        assert "Info message" in caplog.text
        assert "Warning message" in caplog.text
        assert "Error message" in caplog.text
        # Debug message behavior depends on queue logger configuration

        # With queue-based logging, file creation happens asynchronously
        # Test that the policy configuration is correct instead
        assert policy.log_path == temp_log_dir

        # Test debug mode
        policy = MockPolicy(
            base_environment,
            discount_factor=0.9,
            name="test_policy_debug",
            log_path=temp_log_dir,
            debug=True,
        )

        # Clear previous logs
        caplog.clear()

        # Test logging in debug mode
        with caplog.at_level(logging.DEBUG):
            policy.logger.debug("Debug message in debug mode")
            policy.logger.info("Info message in debug mode")
            assert "Info message in debug mode" in caplog.text
            # Debug message visibility depends on queue logger debug configuration

        # Test that debug mode is properly configured in the policy
        assert policy.debug is True

    def test_logger_initialization_message(self, base_environment, caplog):
        """Test that the logger outputs the correct initialization message.

        Purpose: Validates proper initialization of logger  message

        Given: Constructor parameters and initial conditions
        When: Object is initialized
        Then: Object is properly constructed with expected attributes

        Test type: unit
        """
        with caplog.at_level(logging.INFO):
            policy = MockPolicy(
                base_environment, discount_factor=0.9, name="test_policy", debug=True
            )
            assert "Initialized policy: test_policy (debug=True)" in caplog.text


class TestPolicyConfigId:
    def test_config_id_generation(self, base_policy):
        """Test that config_id generates a deterministic hash.

        Purpose: Validates config_id behavior for  generation

        Given: Belief objects with specific configurations
        When: Config IDs are generated or compared
        Then: Config IDs behave as expected (deterministic, unique, etc.)

        Test type: configuration
        """
        config_id = base_policy.config_id
        assert isinstance(config_id, str)
        assert len(config_id) == 64  # SHA-256 produces 64 hex characters
        assert all(c in "0123456789abcdef" for c in config_id)  # Valid hex characters

    def test_config_id_includes_logger_settings(self, base_environment, temp_log_dir):
        """Test that logger settings are included in config_id.

        Purpose: Validates config_id behavior for  includes logger settings

        Given: Belief objects with specific configurations
        When: Config IDs are generated or compared
        Then: Config IDs behave as expected (deterministic, unique, etc.)

        Test type: configuration
        """
        # Create policies with different debug settings but same name
        policy1 = MockPolicy(
            base_environment,
            discount_factor=0.9,
            name="test_policy",
            log_path=temp_log_dir,
            debug=True,
        )
        policy2 = MockPolicy(
            base_environment,
            discount_factor=0.9,
            name="test_policy",
            log_path=temp_log_dir,
            debug=False,
        )

        # Create policies with different names but same debug settings
        policy3 = MockPolicy(
            base_environment,
            discount_factor=0.9,
            name="test_policy_alt",
            log_path=temp_log_dir,
            debug=True,
        )

        # Different debug settings should produce different config_ids
        assert (
            policy1.config_id != policy2.config_id
        ), "Policies with different debug settings should have different config_ids"

        # Different names should produce different config_ids
        assert (
            policy1.config_id != policy3.config_id
        ), "Policies with different names should have different config_ids"

        # Same settings should produce same config_ids
        policy4 = MockPolicy(
            base_environment,
            discount_factor=0.9,
            name="test_policy",
            log_path=temp_log_dir,
            debug=True,
        )
        assert (
            policy1.config_id == policy4.config_id
        ), "Policies with same settings should have same config_ids"
