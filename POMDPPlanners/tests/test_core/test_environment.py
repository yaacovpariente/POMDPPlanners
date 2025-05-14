import numpy as np
import pytest
from typing import Optional
from POMDPPlanners.core.environment import Environment, StateTransitionModel, ObservationModel, DiscreteActionsEnvironment
from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP, STATES, ACTIONS, OBSERVATIONS
from POMDPPlanners.core.config_types import EnvironmentConfig

class MockDistribution(Distribution):
    def sample(self) -> np.ndarray:
        return np.array([1, 2])
    
    def probability(self, value: np.ndarray) -> float:
        return 1.0

class MockStateTransitionModel(StateTransitionModel):
    def sample(self) -> np.ndarray:
        return np.array([1, 2])
    
    def probability(self, next_state: np.ndarray) -> float:
        return 1.0

class MockObservationModel(ObservationModel):
    def sample(self) -> np.ndarray:
        return np.array([3, 4])
    
    def probability(self, next_observation: np.ndarray) -> float:
        return 1.0

class MockEnvironment(Environment):
    def __init__(self, discount_factor: float, test_array: Optional[np.ndarray] = None):
        super().__init__(discount_factor=discount_factor, name="MockEnvironment")
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
    def __init__(self, discount_factor: float):
        super().__init__(discount_factor=discount_factor, name="DifferentEnvironment")
    
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

class TestTigerPOMDPEquality:
    """Test suite for TigerPOMDP equality comparisons."""
    
    def test_same_discount_factor(self, base_tiger_environment: TigerPOMDP):
        """Test that TigerPOMDPs with same discount factor are equal."""
        other_env = TigerPOMDP(discount_factor=0.95)  # Same as base
        assert base_tiger_environment == other_env
        assert other_env == base_tiger_environment  # Test symmetry
    
    def test_different_discount_factor(self, base_tiger_environment: TigerPOMDP):
        """Test that TigerPOMDPs with different discount factors are not equal."""
        other_env = TigerPOMDP(discount_factor=0.8)  # Different from base
        assert base_tiger_environment != other_env
        assert other_env != base_tiger_environment  # Test symmetry
    
    def test_comparison_with_mock_environment(self, base_tiger_environment: TigerPOMDP):
        """Test that TigerPOMDP is not equal to other Environment types."""
        mock_env = MockEnvironment(discount_factor=0.95)
        assert base_tiger_environment != mock_env
        assert mock_env != base_tiger_environment  # Test symmetry
    
    @pytest.mark.parametrize("invalid_value", [None, 42, "tiger", []])
    def test_comparison_with_non_environment(self, base_tiger_environment: TigerPOMDP, invalid_value):
        """Test comparison with non-Environment objects."""
        assert base_tiger_environment != invalid_value
    
    def test_missing_attributes(self, base_tiger_environment: TigerPOMDP):
        """Test equality when attributes are missing."""
        other_env = TigerPOMDP(discount_factor=0.95)
        delattr(other_env, 'states')
        assert base_tiger_environment != other_env
        
        other_env = TigerPOMDP(discount_factor=0.95)
        delattr(other_env, 'actions')
        assert base_tiger_environment != other_env
        
        other_env = TigerPOMDP(discount_factor=0.95)
        delattr(other_env, 'observations')
        assert base_tiger_environment != other_env
    
    def test_modified_tiger_environment(self, base_tiger_environment: TigerPOMDP):
        """Test equality after modifying environment attributes."""
        modified_env = TigerPOMDP(discount_factor=0.95)
        
        # Test modifying each attribute
        modifications = [
            ('states', ['new_state1', 'new_state2']),
            ('actions', ['new_action1', 'new_action2']),
            ('observations', ['new_obs1', 'new_obs2']),
            ('discount_factor', 0.8)
        ]
        
        for attr, new_value in modifications:
            # Create fresh environment for each test
            test_env = TigerPOMDP(discount_factor=0.95)
            assert test_env == base_tiger_environment  # Should be equal initially
            
            # Modify attribute
            setattr(test_env, attr, new_value)
            assert test_env != base_tiger_environment  # Should not be equal after modification
    
    def test_deep_copy_equality(self, base_tiger_environment: TigerPOMDP):
        """Test that a deep copy of TigerPOMDP is equal to original."""
        import copy
        copied_env = copy.deepcopy(base_tiger_environment)
        assert copied_env == base_tiger_environment
        assert base_tiger_environment == copied_env  # Test symmetry

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

class TestEnvironmentConfig:
    """Test suite for Environment configuration functionality."""
    
    def test_from_config_valid(self, base_environment: MockEnvironment):
        """Test creating an environment from valid config."""
        config = EnvironmentConfig(
            class_name="MockEnvironment",
            params={
                "discount_factor": 0.9,
                "test_array": np.array([1, 2, 3])
            }
        )
        
        env = Environment.from_config(config)
        assert isinstance(env, MockEnvironment)
        assert env.discount_factor == config.params["discount_factor"]
        assert np.array_equal(env.test_array, config.params["test_array"])
    
    def test_from_config_invalid_class(self):
        """Test creating an environment with invalid class name."""
        config = EnvironmentConfig(
            class_name="NonExistentEnvironment",
            params={
                "discount_factor": 0.9,
                "test_array": np.array([1, 2, 3])
            }
        )
        
        with pytest.raises(ValueError, match="Environment class 'NonExistentEnvironment' not found"):
            Environment.from_config(config)
    
    def test_from_config_missing_required_params(self):
        """Test creating an environment with missing required parameters."""
        config = EnvironmentConfig(
            class_name="MockEnvironment",
            params={
                # Missing discount_factor
                "test_array": np.array([1, 2, 3])
            }
        )
        
        with pytest.raises(TypeError):
            Environment.from_config(config)
    
    def test_from_config_default_params(self):
        """Test creating an environment with default parameters."""
        config = EnvironmentConfig(
            class_name="MockEnvironment",
            params={
                "discount_factor": 0.9
                # test_array will use default value
            }
        )
        
        env = Environment.from_config(config)
        assert np.array_equal(env.test_array, np.array([1, 2, 3]))
    
    def test_from_config_different_environment(self):
        """Test creating a different environment type from config."""
        config = EnvironmentConfig(
            class_name="DifferentEnvironment",
            params={
                "discount_factor": 0.9
            }
        )
        
        env = Environment.from_config(config)
        assert isinstance(env, DifferentEnvironment)
        assert env.discount_factor == config.params["discount_factor"]

class TestEnvironmentConfigCreation:
    """Test suite for creating all available environments from config."""
    
    @pytest.mark.parametrize("env_name,env_class,required_params", [
        ("SanityPOMDP", "SanityPOMDP", {}),  # SanityPOMDP doesn't require discount_factor
        ("TigerPOMDP", "TigerPOMDP", {"discount_factor": 0.95}),
        ("CartpolePOMDP", "CartpolePOMDP", {"discount_factor": 0.99}),
        ("MountainCarPOMDP", "MountainCarPOMDP", {"discount_factor": 0.99}),
        ("PushPOMDP", "PushPOMDP", {"discount_factor": 0.99}),
        ("SafetyAntVelocityPOMDP", "SafetyAntVelocityPOMDP", {"discount_factor": 0.99}),
        ("LightDarkPOMDP", "LightDarkPOMDP", {"discount_factor": 0.99})
    ])
    def test_environment_creation(self, env_name: str, env_class: str, required_params: dict):
        """Test creating each environment type from config."""
        # Create config with required parameters
        config = EnvironmentConfig(
            class_name=env_class,
            params=required_params
        )
        
        try:
            # Create environment from config
            env = Environment.from_config(config)
            
            # Verify environment was created correctly
            assert env is not None
            assert env.__class__.__name__ == env_class
            assert isinstance(env, Environment)
            
            # Test basic environment functionality
            assert hasattr(env, 'name')
            assert env.name == env_class
            
            # Test that required methods exist
            assert hasattr(env, 'state_transition_model')
            assert hasattr(env, 'observation_model')
            assert hasattr(env, 'reward')
            assert hasattr(env, 'is_terminal')
            assert hasattr(env, 'initial_state_dist')
            assert hasattr(env, 'initial_observation_dist')
            assert hasattr(env, 'is_equal_observation')
            
            # For discrete action environments, test get_actions
            if isinstance(env, DiscreteActionsEnvironment):
                assert hasattr(env, 'get_actions')
                actions = env.get_actions()
                assert isinstance(actions, list)
                assert len(actions) > 0
                
        except ValueError as e:
            if "not found" in str(e):
                pytest.skip(f"Environment class {env_class} not found")
            else:
                raise
    
    def test_environment_creation_with_params(self):
        """Test creating environments with specific parameters."""
        # Test TigerPOMDP with specific parameters
        tiger_config = EnvironmentConfig(
            class_name="TigerPOMDP",
            params={
                "discount_factor": 0.95,
                "name": "TestTiger"
            }
        )
        tiger_env = Environment.from_config(tiger_config)
        assert tiger_env.discount_factor == 0.95
        assert tiger_env.name == "TestTiger"
        
        # Test CartpolePOMDP with specific parameters
        cartpole_config = EnvironmentConfig(
            class_name="CartpolePOMDP",
            params={
                "discount_factor": 0.99,
                "name": "TestCartpole"
            }
        )
        try:
            cartpole_env = Environment.from_config(cartpole_config)
            assert cartpole_env.discount_factor == 0.99
            assert cartpole_env.name == "TestCartpole"
        except ValueError as e:
            if "not found" in str(e):
                pytest.skip("CartpolePOMDP environment not found")
            else:
                raise
    
    def test_environment_creation_invalid_params(self):
        """Test creating environments with invalid parameters."""
        # Test with invalid discount factor
        config = EnvironmentConfig(
            class_name="TigerPOMDP",  # Use TigerPOMDP instead of SanityPOMDP
            params={
                "discount_factor": 2.0  # Invalid: should be between 0 and 1
            }
        )
        with pytest.raises(ValueError):
            Environment.from_config(config)
        
        # Test with missing required parameters
        config = EnvironmentConfig(
            class_name="TigerPOMDP",
            params={}  # Missing required parameters
        )
        with pytest.raises(TypeError):
            Environment.from_config(config)
    
    def test_environment_creation_invalid_class(self):
        """Test creating environment with invalid class name."""
        config = EnvironmentConfig(
            class_name="NonExistentEnvironment",
            params={}
        )
        with pytest.raises(ValueError, match="Environment class 'NonExistentEnvironment' not found"):
            Environment.from_config(config)
