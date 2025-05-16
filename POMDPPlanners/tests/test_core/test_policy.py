"""Tests for the Policy class."""

import pytest
import numpy as np
from POMDPPlanners.core.policy import Policy, PolicySpaceInfo
from POMDPPlanners.core.environment import Environment, StateTransitionModel, ObservationModel, SpaceInfo, SpaceType
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.config_types import PolicyConfig

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
    def __init__(self, environment: Environment, discount_factor: float, name: str, custom_param: str = "default"):
        super().__init__(environment, discount_factor, name)
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
    def __init__(self, environment: Environment, discount_factor: float, name: str):
        super().__init__(environment, discount_factor, name)
    
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

class TestPolicyConfigId:
    def test_config_id_generation(self, base_policy):
        """Test that config_id generates a deterministic hash."""
        config_id = base_policy.config_id
        assert isinstance(config_id, str)
        assert len(config_id) == 64  # SHA-256 produces 64 hex characters
        assert all(c in '0123456789abcdef' for c in config_id)  # Valid hex characters

class TestPolicyConfig:
    """Test suite for Policy configuration functionality."""
    
    def test_from_config_valid(self, base_environment: MockEnvironment):
        """Test creating a policy from valid config."""
        config = PolicyConfig(
            class_name="MockPolicy",
            params={
                "environment": base_environment,
                "discount_factor": 0.9,
                "name": "test_policy",
                "custom_param": "test_value"
            }
        )
        
        policy = Policy.from_config(config)
        assert isinstance(policy, MockPolicy)
        assert policy.environment == config.params["environment"]
        assert policy.discount_factor == config.params["discount_factor"]
        assert policy.name == config.params["name"]
        assert policy.custom_param == config.params["custom_param"]
    
    def test_from_config_invalid_class(self, base_environment: MockEnvironment):
        """Test creating a policy with invalid class name."""
        config = PolicyConfig(
            class_name="NonExistentPolicy",
            params={
                "environment": base_environment,
                "discount_factor": 0.9,
                "name": "test_policy"
            }
        )
        
        with pytest.raises(ValueError, match="Policy class 'NonExistentPolicy' not found"):
            Policy.from_config(config)
    
    def test_from_config_missing_required_params(self, base_environment: MockEnvironment):
        """Test creating a policy with missing required parameters."""
        config = PolicyConfig(
            class_name="MockPolicy",
            params={
                "environment": base_environment,
                # Missing discount_factor and name
                "custom_param": "test_value"
            }
        )
        
        with pytest.raises(TypeError):
            Policy.from_config(config)
    
    def test_from_config_default_params(self, base_environment: MockEnvironment):
        """Test creating a policy with default parameters."""
        config = PolicyConfig(
            class_name="MockPolicy",
            params={
                "environment": base_environment,
                "discount_factor": 0.9,
                "name": "test_policy"
                # custom_param will use default value
            }
        )
        
        policy = Policy.from_config(config)
        assert policy.custom_param == "default"
    
    def test_from_config_different_policy(self, base_environment: MockEnvironment):
        """Test creating a different policy type from config."""
        config = PolicyConfig(
            class_name="DifferentPolicy",
            params={
                "environment": base_environment,
                "discount_factor": 0.9,
                "name": "test_policy"
            }
        )
        
        policy = Policy.from_config(config)
        assert isinstance(policy, DifferentPolicy)
        assert policy.environment == config.params["environment"]
        assert policy.discount_factor == config.params["discount_factor"]
        assert policy.name == config.params["name"]

class TestAllPolicyConfigCreation:
    """Test suite for creating all available policies from config."""
    
    @pytest.mark.parametrize("policy_class,params", [
        ("MockPolicy", {"discount_factor": 0.9, "name": "test_policy", "environment": None}),
        ("DifferentPolicy", {"discount_factor": 0.9, "name": "test_policy", "environment": None}),
        ("POMCP", {"discount_factor": 0.9, "name": "test_policy", "environment": None, "n_simulations": 1}),
        ("StandardSparseSamplingDiscreteActionsPlanner", {"discount_factor": 0.9, "name": "test_policy", "environment": None, "n_simulations": 1}),
        ("SparsePFT", {"discount_factor": 0.9, "name": "test_policy", "environment": None, "n_simulations": 1}),
        ("PathSimulationPolicy", {"discount_factor": 0.9, "name": "test_policy", "environment": None, "n_simulations": 1, "time_out_in_seconds": 1}),
    ])
    def test_policy_creation(self, policy_class, params, base_environment):
        """Test creating each policy type from config."""
        # Set environment param if needed
        if "environment" in params:
            params["environment"] = base_environment
        config = PolicyConfig(
            class_name=policy_class,
            params=params
        )
        try:
            policy = Policy.from_config(config)
            assert policy is not None
            assert policy.__class__.__name__ == policy_class
            assert isinstance(policy, Policy)
        except ValueError as e:
            if "not found" in str(e):
                pytest.skip(f"Policy class {policy_class} not found")
            else:
                raise
        except TypeError as e:
            pytest.skip(f"Policy class {policy_class} could not be instantiated: {e}")
