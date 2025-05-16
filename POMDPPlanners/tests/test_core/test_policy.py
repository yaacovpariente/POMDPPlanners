"""Tests for the Policy class."""

import pytest
import numpy as np
from POMDPPlanners.core.policy import Policy, PolicySpaceInfo
from POMDPPlanners.core.environment import Environment, StateTransitionModel, ObservationModel, SpaceInfo, SpaceType
from POMDPPlanners.core.belief import Belief

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
