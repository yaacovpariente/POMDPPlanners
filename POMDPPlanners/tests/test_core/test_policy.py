"""Tests for the Policy class."""

import pytest
import numpy as np
from pathlib import Path
import logging
import random
from POMDPPlanners.core.policy import Policy, PolicySpaceInfo
from POMDPPlanners.core.environment import Environment, StateTransitionModel, ObservationModel, SpaceInfo, SpaceType
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.utils.logger import get_logger

np.random.seed(42)
random.seed(42)

class MockEnvironment(Environment):
    def __init__(self, discount_factor: float, name: str):
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.DISCRETE
        )
        super().__init__(discount_factor=discount_factor, name=name, space_info=space_info)
    
    def state_transition_model(self, state, action):
        pass
    
    def observation_model(self, next_state, action):
        pass
    
    def reward(self, state, action):
        pass
    
    def is_terminal(self, state):
        pass
    
    def initial_state_dist(self):
        pass
    
    def initial_observation_dist(self):
        pass
    
    def is_equal_observation(self, observation1, observation2):
        pass

class MockBelief(Belief):
    def __init__(self, environment: Environment):
        super().__init__(environment)
    
    def update(self, action, observation):
        pass
    
    def sample(self):
        pass

class MockPolicy(Policy):
    def __init__(
        self, 
        environment: Environment, 
        discount_factor: float, 
        name: str, 
        custom_param: str = "default",
        log_path: Path = None,
        debug: bool = False
    ):
        super().__init__(environment, discount_factor, name, log_path=log_path, debug=debug)
        self.custom_param = custom_param
    
    def action(self, belief: Belief):
        pass

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.DISCRETE
        )

class DifferentPolicy(Policy):
    def __init__(
        self, 
        environment: Environment, 
        discount_factor: float, 
        name: str,
        log_path: Path = None,
        debug: bool = False
    ):
        super().__init__(environment, discount_factor, name, log_path=log_path, debug=debug)
    
    def action(self, belief: Belief):
        pass

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.DISCRETE
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
        assert policy.logger.level == logging.INFO
        assert len(policy.logger.handlers) == 1  # Only console handler
        assert any(isinstance(h, logging.StreamHandler) for h in policy.logger.handlers)
        
        # Test with log file
        policy = MockPolicy(
            base_environment, 
            discount_factor=0.9, 
            name="test_policy",
            log_path=temp_log_dir
        )
        assert policy.logger.name == "policy.test_policy"
        assert len(policy.logger.handlers) == 2  # Console and file handlers
        assert any(isinstance(h, logging.FileHandler) for h in policy.logger.handlers)
        assert any(isinstance(h, logging.StreamHandler) for h in policy.logger.handlers)
        
        # Verify log file was created - note the nested logs directory
        log_files = list(temp_log_dir.glob("logs/*.log"))
        assert len(log_files) > 0
        assert any("test_policy" in log_file.name for log_file in log_files)
        
        # Test with debug mode
        policy = MockPolicy(
            base_environment, 
            discount_factor=0.9, 
            name="test_policy_debug",
            log_path=temp_log_dir,
            debug=True
        )
        assert policy.logger.name == "policy.test_policy_debug"
        assert policy.logger.level == logging.DEBUG
        assert len(policy.logger.handlers) == 2  # Console and file handlers

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
            log_path=temp_log_dir
        )
        
        # Test logging different levels
        with caplog.at_level(logging.DEBUG):  # Capture all levels
            policy.logger.debug("Debug message")
            policy.logger.info("Info message")
            policy.logger.warning("Warning message")
            policy.logger.error("Error message")
        
        # Check console output - debug should not appear in normal mode
        assert "Debug message" not in caplog.text
        assert "Info message" in caplog.text
        assert "Warning message" in caplog.text
        assert "Error message" in caplog.text
        
        # Verify log file content - note the nested logs directory
        log_files = list(temp_log_dir.glob("logs/*.log"))
        assert len(log_files) > 0
        log_content = log_files[0].read_text()
        assert "Debug message" not in log_content  # Debug not in file either
        assert "Info message" in log_content
        assert "Warning message" in log_content
        assert "Error message" in log_content
        
        # Test debug mode
        policy = MockPolicy(
            base_environment, 
            discount_factor=0.9, 
            name="test_policy_debug",
            log_path=temp_log_dir,
            debug=True
        )
        
        # Clear previous logs
        caplog.clear()
        
        # Test logging in debug mode
        with caplog.at_level(logging.DEBUG):
            policy.logger.debug("Debug message in debug mode")
            policy.logger.info("Info message in debug mode")
            assert "Debug message in debug mode" in caplog.text
            assert "Info message in debug mode" in caplog.text
            
        # Verify debug messages appear in log file - note the nested logs directory
        log_files = list(temp_log_dir.glob("logs/*debug*.log"))
        assert len(log_files) > 0
        log_content = log_files[0].read_text()
        assert "Debug message in debug mode" in log_content

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
                base_environment,
                discount_factor=0.9,
                name="test_policy",
                debug=True
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
        assert all(c in '0123456789abcdef' for c in config_id)  # Valid hex characters

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
            debug=True
        )
        policy2 = MockPolicy(
            base_environment, 
            discount_factor=0.9, 
            name="test_policy",
            log_path=temp_log_dir,
            debug=False
        )
        
        # Create policies with different names but same debug settings
        policy3 = MockPolicy(
            base_environment, 
            discount_factor=0.9, 
            name="test_policy_alt",
            log_path=temp_log_dir,
            debug=True
        )
        
        # Different debug settings should produce different config_ids
        assert policy1.config_id != policy2.config_id, "Policies with different debug settings should have different config_ids"
        
        # Different names should produce different config_ids
        assert policy1.config_id != policy3.config_id, "Policies with different names should have different config_ids"
        
        # Same settings should produce same config_ids
        policy4 = MockPolicy(
            base_environment, 
            discount_factor=0.9, 
            name="test_policy",
            log_path=temp_log_dir,
            debug=True
        )
        assert policy1.config_id == policy4.config_id, "Policies with same settings should have same config_ids"
