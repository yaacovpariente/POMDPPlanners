import numpy as np
import pytest
from typing import Optional, Any, List
from pathlib import Path
from POMDPPlanners.core.environment import (
    Environment,
    StateTransitionModel,
    ObservationModel,
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType
)
from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

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
    def __init__(self, discount_factor: float, test_array: Optional[np.ndarray] = None, 
                 output_dir: Optional[Path] = None, debug: bool = False):
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.DISCRETE
        )
        super().__init__(discount_factor=discount_factor, name="MockEnvironment", 
                        space_info=space_info, output_dir=output_dir, debug=debug)
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
    def __init__(self, discount_factor: float, output_dir: Optional[Path] = None, debug: bool = False):
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.DISCRETE
        )
        super().__init__(discount_factor=discount_factor, name="DifferentEnvironment", 
                        space_info=space_info, output_dir=output_dir, debug=debug)
    
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

@pytest.mark.parametrize("discount_factor,expected_equal", [
    (0.9, True),   # Same discount factor
    (0.8, False),  # Different discount factor
])
def test_discount_factor_equality(base_environment: MockEnvironment, 
                                discount_factor: float, 
                                expected_equal: bool):
    """Test equality comparison with different discount factors."""
    other_env = MockEnvironment(discount_factor=discount_factor)
    assert (base_environment == other_env) == expected_equal

@pytest.mark.parametrize("test_array,expected_equal", [
    (np.array([1, 2, 3]), True),   # Same array
    (np.array([4, 5, 6]), False),  # Different array
])
def test_array_equality(base_environment: MockEnvironment, 
                       test_array: np.ndarray, 
                       expected_equal: bool):
    """Test equality comparison with different numpy arrays."""
    other_env = MockEnvironment(discount_factor=0.9, test_array=test_array)
    assert (base_environment == other_env) == expected_equal

def test_different_environment_class(base_environment: MockEnvironment):
    """Test equality comparison with a different environment class."""
    different_env = DifferentEnvironment(discount_factor=0.9)
    assert base_environment != different_env

def test_non_environment_comparison(base_environment: MockEnvironment):
    """Test equality comparison with non-Environment objects."""
    assert base_environment != "not an environment"
    assert base_environment != 42
    assert base_environment != None

def test_missing_attribute(base_environment: MockEnvironment):
    """Test equality comparison when an attribute is missing."""
    other_env = MockEnvironment(discount_factor=0.9)
    delattr(other_env, 'test_array')
    assert base_environment != other_env

@pytest.mark.parametrize("attr_name,attr_value", [
    ("discount_factor", 0.9),
    ("test_array", np.array([1, 2, 3])),
])
def test_attribute_presence(base_environment: MockEnvironment, 
                          attr_name: str, 
                          attr_value: object):
    """Test that attributes are present and have correct values."""
    assert hasattr(base_environment, attr_name)
    if isinstance(attr_value, np.ndarray):
        assert np.array_equal(getattr(base_environment, attr_name), attr_value)
    else:
        assert getattr(base_environment, attr_name) == attr_value

@pytest.fixture
def base_tiger_environment() -> TigerPOMDP:
    """Fixture providing a base TigerPOMDP environment for comparison."""
    return TigerPOMDP(discount_factor=0.95)

class TestEnvironmentConfigId:
    """Test suite for Environment config_id property."""
    
    def test_same_config_same_id(self, base_environment: MockEnvironment):
        """Test that environments with same configuration have same config_id."""
        other_env = MockEnvironment(discount_factor=0.9)
        assert base_environment.config_id == other_env.config_id
    
    def test_different_discount_factor(self, base_environment: MockEnvironment):
        """Test that different discount factor results in different config_id."""
        other_env = MockEnvironment(discount_factor=0.8)
        assert base_environment.config_id != other_env.config_id
    
    def test_different_array(self, base_environment: MockEnvironment):
        """Test that different numpy array results in different config_id."""
        other_env = MockEnvironment(discount_factor=0.9, test_array=np.array([4, 5, 6]))
        assert base_environment.config_id != other_env.config_id
    
    def test_different_environment_class(self, base_environment: MockEnvironment):
        """Test that different environment class results in different config_id."""
        different_env = DifferentEnvironment(discount_factor=0.9)
        assert base_environment.config_id != different_env.config_id
    
    def test_config_id_consistency(self, base_environment: MockEnvironment):
        """Test that config_id remains consistent across multiple calls."""
        id1 = base_environment.config_id
        id2 = base_environment.config_id
        assert id1 == id2
    
    def test_tiger_pomdp_config_id(self, base_tiger_environment: TigerPOMDP):
        """Test config_id with TigerPOMDP environment."""
        # Same configuration should have same ID
        other_env = TigerPOMDP(discount_factor=0.95)
        assert base_tiger_environment.config_id == other_env.config_id
        
        # Different configuration should have different ID
        different_env = TigerPOMDP(discount_factor=0.8)
        assert base_tiger_environment.config_id != different_env.config_id
    
    def test_config_id_format(self, base_environment: MockEnvironment):
        """Test that config_id is a valid SHA-256 hash."""
        config_id = base_environment.config_id
        assert len(config_id) == 64  # SHA-256 produces 64 hex characters
        assert all(c in '0123456789abcdef' for c in config_id)  # Valid hex characters

class TestEnvironmentLogger:
    """Test suite for Environment logger functionality."""
    
    def test_logger_initialization(self, base_environment_with_logging: MockEnvironment):
        """Test that logger is properly initialized."""
        assert hasattr(base_environment_with_logging, 'logger')
        assert base_environment_with_logging.logger.name == "environment.MockEnvironment"
    
    def test_logger_without_output_dir(self, base_environment: MockEnvironment):
        """Test that logger works without output directory."""
        assert hasattr(base_environment, 'logger')
        assert base_environment.logger.name == "environment.MockEnvironment"
    
    def test_logger_debug_mode(self, base_environment_with_logging: MockEnvironment):
        """Test that debug mode affects logger level."""
        assert base_environment_with_logging.logger.level <= 10  # DEBUG level is 10
    
    def test_logger_normal_mode(self, base_environment: MockEnvironment):
        """Test that normal mode sets correct logger level."""
        assert base_environment.logger.level == 20  # INFO level is 20
    
    def test_logger_output_dir(self, tmp_path):
        """Test that logger creates log files in output directory."""
        env = MockEnvironment(discount_factor=0.9, output_dir=tmp_path)
        logs_dir = tmp_path / "logs"
        assert logs_dir.exists()
        log_files = list(logs_dir.glob("*.log"))
        assert len(log_files) > 0
