import pytest
import numpy as np
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP


@pytest.fixture
def tiger_pomdp():
    return TigerPOMDP(discount_factor=0.95)


def test_initialization(tiger_pomdp):
    assert tiger_pomdp.discount_factor == 0.95
    assert set(tiger_pomdp.states) == {"tiger_left", "tiger_right"}
    assert set(tiger_pomdp.actions) == {"listen", "open_left", "open_right"}
    assert set(tiger_pomdp.observations) == {"hear_left", "hear_right", "hear_nothing"}


def test_get_actions(tiger_pomdp):
    actions = tiger_pomdp.get_actions()
    assert len(actions) == 3
    assert set(actions) == {"listen", "open_left", "open_right"}


def test_initial_state_distribution(tiger_pomdp):
    dist = tiger_pomdp.initial_state_dist()
    # Sample multiple times to ensure we get both states
    samples = [dist.sample() for _ in range(100)]
    assert all(s in tiger_pomdp.states for s in samples)
    assert len(set(samples)) == 2  # Should get both states


def test_initial_observation_distribution(tiger_pomdp):
    dist = tiger_pomdp.initial_observation_dist()
    samples = [dist.sample() for _ in range(10)]
    assert all(s == "hear_nothing" for s in samples)


def test_state_transition_listen(tiger_pomdp):
    # Listening shouldn't change the state
    for state in tiger_pomdp.states:
        dist = tiger_pomdp.state_transition_model(state, "listen")
        samples = [dist.sample() for _ in range(10)]
        assert all(s == state for s in samples)


def test_state_transition_open_door(tiger_pomdp):
    # Opening a door should randomly place tiger behind either door
    for action in ["open_left", "open_right"]:
        dist = tiger_pomdp.state_transition_model("tiger_left", action)
        samples = [dist.sample() for _ in range(100)]
        assert all(s in tiger_pomdp.states for s in samples)
        assert len(set(samples)) == 2  # Should get both states


def test_observation_model_listen(tiger_pomdp):
    # Test listen action with both states
    for state in tiger_pomdp.states:
        dist = tiger_pomdp.observation_model(state, "listen")
        samples = [dist.sample() for _ in range(100)]
        assert all(s in tiger_pomdp.observations for s in samples)

        # Should mostly get correct observation
        expected_obs = "hear_left" if state == "tiger_left" else "hear_right"
        correct_count = sum(1 for s in samples if s == expected_obs)
        assert correct_count > 70  # Should be around 85% correct


def test_observation_model_open_door(tiger_pomdp):
    # Opening a door should always give 'hear_nothing'
    for state in tiger_pomdp.states:
        for action in ["open_left", "open_right"]:
            dist = tiger_pomdp.observation_model(state, action)
            samples = [dist.sample() for _ in range(10)]
            assert all(s == "hear_nothing" for s in samples)


def test_reward_func_listen(tiger_pomdp):
    # Listening should always give -1 reward
    for state in tiger_pomdp.states:
        assert tiger_pomdp.reward(state, "listen") == -1.0


def test_reward_func_open_door(tiger_pomdp):
    # Test opening doors with tiger
    assert tiger_pomdp.reward("tiger_left", "open_left") == -100.0
    assert tiger_pomdp.reward("tiger_right", "open_right") == -100.0

    # Test opening doors with treasure
    assert tiger_pomdp.reward("tiger_right", "open_left") == 10.0
    assert tiger_pomdp.reward("tiger_left", "open_right") == 10.0


def test_is_terminal(tiger_pomdp):
    # Currently always returns False
    for state in tiger_pomdp.states:
        assert not tiger_pomdp.is_terminal(state)


class TestTigerPOMDPConfigId:
    """Test suite for TigerPOMDP config_id functionality."""
    
    def test_config_id_consistency(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id is consistent for identical environments."""
        other_env = TigerPOMDP(discount_factor=0.95)
        assert tiger_pomdp.config_id == other_env.config_id
    
    def test_config_id_different_discount_factor(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id changes with different discount factor."""
        other_env = TigerPOMDP(discount_factor=0.8)
        assert tiger_pomdp.config_id != other_env.config_id
    
    def test_config_id_different_states(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id changes with different states."""
        other_env = TigerPOMDP(discount_factor=0.95)
        other_env.states = ["tiger_left", "tiger_right", "tiger_middle"]  # Different states
        assert tiger_pomdp.config_id != other_env.config_id
    
    def test_config_id_different_actions(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id changes with different actions."""
        other_env = TigerPOMDP(discount_factor=0.95)
        other_env.actions = ["listen", "open_left", "open_right", "wait"]  # Different actions
        assert tiger_pomdp.config_id != other_env.config_id
    
    def test_config_id_different_observations(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id changes with different observations."""
        other_env = TigerPOMDP(discount_factor=0.95)
        other_env.observations = ["hear_left", "hear_right", "hear_nothing", "hear_both"]  # Different observations
        assert tiger_pomdp.config_id != other_env.config_id
    
    def test_config_id_format(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id is a valid SHA-256 hash."""
        config_id = tiger_pomdp.config_id
        assert isinstance(config_id, str)
        assert len(config_id) == 64  # SHA-256 hash length
        assert all(c in '0123456789abcdef' for c in config_id)  # Valid hex characters
    
    def test_config_id_deterministic(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id is deterministic (same input always produces same output)."""
        config_id1 = tiger_pomdp.config_id
        config_id2 = tiger_pomdp.config_id
        assert config_id1 == config_id2
