import pytest
import numpy as np
from POMDPPlanners.environments.sanity_pomdp import (
    SanityPOMDP,
    SanityStateTransitionModel,
    SanityObservationModel,
    SanityInitialStateDist,
    SanityInitialObservationDist
)
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.core.policy import PolicyRunData, PolicyInfoVariable


@pytest.fixture
def sanity_pomdp():
    return SanityPOMDP(discount_factor=0.95)


@pytest.fixture
def sanity_pomdp_debug():
    return SanityPOMDP(discount_factor=0.95, debug=True)


class TestSanityPOMDPInitialization:
    """Test suite for SanityPOMDP initialization."""
    
    def test_initialization(self, sanity_pomdp):
        """Test basic initialization."""
        assert sanity_pomdp.discount_factor == 0.95
        assert sanity_pomdp.name == "SanityPOMDP"
        assert sanity_pomdp.space_info.action_space.value == "discrete"
        assert sanity_pomdp.space_info.observation_space.value == "discrete"
        assert sanity_pomdp.debug is False
    
    def test_initialization_with_debug(self, sanity_pomdp_debug):
        """Test initialization with debug mode."""
        assert sanity_pomdp_debug.debug is True
    
    def test_initialization_with_output_dir(self):
        """Test initialization with output directory."""
        from pathlib import Path
        output_dir = Path("/tmp/test_output")
        env = SanityPOMDP(discount_factor=0.95, output_dir=output_dir)
        assert env.output_dir == output_dir


class TestSanityPOMDPEquality:
    """Test suite for SanityPOMDP equality comparisons."""
    
    def test_same_environment_equality(self, sanity_pomdp):
        """Test that identical environments are equal."""
        other_env = SanityPOMDP(discount_factor=0.95)
        assert sanity_pomdp == other_env
        assert other_env == sanity_pomdp  # Test symmetry
    
    def test_different_discount_factor(self, sanity_pomdp):
        """Test that environments with different discount factors are not equal."""
        other_env = SanityPOMDP(discount_factor=0.8)
        assert sanity_pomdp != other_env
        assert other_env != sanity_pomdp  # Test symmetry
    
    def test_different_debug_mode(self, sanity_pomdp):
        """Test that environments with different debug modes are not equal."""
        other_env = SanityPOMDP(discount_factor=0.95, debug=True)
        assert sanity_pomdp != other_env
        assert other_env != sanity_pomdp  # Test symmetry
    
    def test_comparison_with_non_environment(self, sanity_pomdp):
        """Test comparison with non-Environment objects."""
        assert sanity_pomdp != "not an environment"
        assert sanity_pomdp != 42
        assert sanity_pomdp != None


class TestSanityPOMDPConfigId:
    """Test suite for SanityPOMDP config_id functionality."""
    
    def test_config_id_consistency(self, sanity_pomdp):
        """Test that config_id is consistent for identical environments."""
        other_env = SanityPOMDP(discount_factor=0.95)
        assert sanity_pomdp.config_id == other_env.config_id
    
    def test_config_id_different_discount_factor(self, sanity_pomdp):
        """Test that config_id changes with different discount factor."""
        other_env = SanityPOMDP(discount_factor=0.8)
        assert sanity_pomdp.config_id != other_env.config_id
    
    def test_config_id_different_debug_mode(self, sanity_pomdp):
        """Test that config_id changes with different debug mode."""
        other_env = SanityPOMDP(discount_factor=0.95, debug=True)
        assert sanity_pomdp.config_id != other_env.config_id
    
    def test_config_id_format(self, sanity_pomdp):
        """Test that config_id is a valid SHA-256 hash."""
        config_id = sanity_pomdp.config_id
        assert isinstance(config_id, str)
        assert len(config_id) == 64  # SHA-256 hash length
        assert all(c in '0123456789abcdef' for c in config_id)  # Valid hex characters
    
    def test_config_id_deterministic(self, sanity_pomdp):
        """Test that config_id is deterministic (same input always produces same output)."""
        config_id1 = sanity_pomdp.config_id
        config_id2 = sanity_pomdp.config_id
        assert config_id1 == config_id2


class TestSanityPOMDPActions:
    """Test suite for SanityPOMDP action-related functionality."""
    
    def test_get_actions(self, sanity_pomdp):
        """Test that get_actions returns the correct actions."""
        actions = sanity_pomdp.get_actions()
        assert actions == [0, 1]
        assert len(actions) == 2


class TestSanityStateTransitionModel:
    """Test suite for SanityStateTransitionModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = SanityStateTransitionModel(state=0, action=1)
        assert model.state == 0
        assert model.action == 1
    
    def test_sample_action_0(self):
        """Test sampling with action 0 (should always lead to state 0)."""
        model = SanityStateTransitionModel(state=0, action=0)
        samples = model.sample(n_samples=10)
        assert all(s == 0 for s in samples)
        assert len(samples) == 10
    
    def test_sample_action_1(self):
        """Test sampling with action 1 (should always lead to state 1)."""
        model = SanityStateTransitionModel(state=0, action=1)
        samples = model.sample(n_samples=10)
        assert all(s == 1 for s in samples)
        assert len(samples) == 10
    
    def test_sample_different_states(self):
        """Test that state doesn't affect transition (only action matters)."""
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
        """Test probability calculation for action 0."""
        model = SanityStateTransitionModel(state=0, action=0)
        values = [0, 1, 0, 1]
        probs = model.probability(values)
        expected = np.array([1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(probs, expected)
    
    def test_probability_action_1(self):
        """Test probability calculation for action 1."""
        model = SanityStateTransitionModel(state=0, action=1)
        values = [0, 1, 0, 1]
        probs = model.probability(values)
        expected = np.array([0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(probs, expected)


class TestSanityObservationModel:
    """Test suite for SanityObservationModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = SanityObservationModel(next_state=1, action=0)
        assert model.next_state == 1
        assert model.action == 0
    
    def test_sample_state_0(self):
        """Test sampling with state 0 (should always observe 0)."""
        model = SanityObservationModel(next_state=0, action=0)
        samples = model.sample(n_samples=10)
        assert all(s == 0 for s in samples)
        assert len(samples) == 10
    
    def test_sample_state_1(self):
        """Test sampling with state 1 (should always observe 1)."""
        model = SanityObservationModel(next_state=1, action=0)
        samples = model.sample(n_samples=10)
        assert all(s == 1 for s in samples)
        assert len(samples) == 10
    
    def test_sample_different_actions(self):
        """Test that action doesn't affect observation (only state matters)."""
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
        """Test probability calculation for state 0."""
        model = SanityObservationModel(next_state=0, action=0)
        values = [0, 1, 0, 1]
        probs = model.probability(values)
        expected = np.array([1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(probs, expected)
    
    def test_probability_state_1(self):
        """Test probability calculation for state 1."""
        model = SanityObservationModel(next_state=1, action=0)
        values = [0, 1, 0, 1]
        probs = model.probability(values)
        expected = np.array([0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(probs, expected)


class TestSanityInitialStateDist:
    """Test suite for SanityInitialStateDist."""
    
    def test_sample(self):
        """Test that initial state distribution always returns state 0."""
        dist = SanityInitialStateDist()
        samples = dist.sample(n_samples=10)
        assert all(s == 0 for s in samples)
        assert len(samples) == 10
    
    def test_probability(self):
        """Test probability calculation for initial state distribution."""
        dist = SanityInitialStateDist()
        values = [0, 1, 0, 1]
        probs = dist.probability(values)
        expected = np.array([1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(probs, expected)


class TestSanityInitialObservationDist:
    """Test suite for SanityInitialObservationDist."""
    
    def test_sample(self):
        """Test that initial observation distribution always returns observation 0."""
        dist = SanityInitialObservationDist()
        samples = dist.sample(n_samples=10)
        assert all(s == 0 for s in samples)
        assert len(samples) == 10
    
    def test_probability(self):
        """Test probability calculation for initial observation distribution."""
        dist = SanityInitialObservationDist()
        values = [0, 1, 0, 1]
        probs = dist.probability(values)
        expected = np.array([1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(probs, expected)


class TestSanityPOMDPModels:
    """Test suite for SanityPOMDP model creation."""
    
    def test_state_transition_model(self, sanity_pomdp):
        """Test state transition model creation."""
        model = sanity_pomdp.state_transition_model(state=0, action=1)
        assert isinstance(model, SanityStateTransitionModel)
        assert model.state == 0
        assert model.action == 1
    
    def test_observation_model(self, sanity_pomdp):
        """Test observation model creation."""
        model = sanity_pomdp.observation_model(next_state=1, action=0)
        assert isinstance(model, SanityObservationModel)
        assert model.next_state == 1
        assert model.action == 0
    
    def test_initial_state_dist(self, sanity_pomdp):
        """Test initial state distribution creation."""
        dist = sanity_pomdp.initial_state_dist()
        assert isinstance(dist, SanityInitialStateDist)
    
    def test_initial_observation_dist(self, sanity_pomdp):
        """Test initial observation distribution creation."""
        dist = sanity_pomdp.initial_observation_dist()
        assert isinstance(dist, SanityInitialObservationDist)


class TestSanityPOMDPReward:
    """Test suite for SanityPOMDP reward function."""
    
    def test_reward_state_0(self, sanity_pomdp):
        """Test reward for state 0 (should be 1.0)."""
        for action in [0, 1]:
            reward = sanity_pomdp.reward(state=0, action=action)
            assert reward == 1.0
    
    def test_reward_state_1(self, sanity_pomdp):
        """Test reward for state 1 (should be 0.0)."""
        for action in [0, 1]:
            reward = sanity_pomdp.reward(state=1, action=action)
            assert reward == 0.0


class TestSanityPOMDPTerminal:
    """Test suite for SanityPOMDP terminal state detection."""
    
    def test_is_terminal(self, sanity_pomdp):
        """Test that no states are terminal."""
        for state in [0, 1]:
            assert not sanity_pomdp.is_terminal(state)


class TestSanityPOMDPObservationEquality:
    """Test suite for SanityPOMDP observation equality."""
    
    def test_is_equal_observation(self, sanity_pomdp):
        """Test observation equality comparison."""
        assert sanity_pomdp.is_equal_observation(0, 0)
        assert sanity_pomdp.is_equal_observation(1, 1)
        assert not sanity_pomdp.is_equal_observation(0, 1)
        assert not sanity_pomdp.is_equal_observation(1, 0)


class TestSanityPOMDPSampleNextStep:
    """Test suite for SanityPOMDP sample_next_step functionality."""
    
    def test_sample_next_step_action_0(self, sanity_pomdp):
        """Test sample_next_step with action 0."""
        next_state, next_observation, reward = sanity_pomdp.sample_next_step(state=0, action=0)
        assert next_state == 0  # Action 0 leads to state 0
        assert next_observation == 0  # State 0 gives observation 0
        assert reward == 1.0  # State 0 gives reward 1.0
    
    def test_sample_next_step_action_1(self, sanity_pomdp):
        """Test sample_next_step with action 1."""
        next_state, next_observation, reward = sanity_pomdp.sample_next_step(state=0, action=1)
        assert next_state == 1  # Action 1 leads to state 1
        assert next_observation == 1  # State 1 gives observation 1
        assert reward == 0.0  # State 1 gives reward 0.0
    
    def test_sample_next_step_from_state_1(self, sanity_pomdp):
        """Test sample_next_step starting from state 1."""
        next_state, next_observation, reward = sanity_pomdp.sample_next_step(state=1, action=0)
        assert next_state == 0  # Action 0 leads to state 0
        assert next_observation == 0  # State 0 gives observation 0
        assert reward == 0.0  # State 1 gives reward 0.0 (reward is based on current state)


class TestSanityPOMDPMetrics:
    """Test suite for SanityPOMDP compute_metrics functionality."""
    
    def test_compute_metrics_empty_histories(self, sanity_pomdp):
        """Test metrics computation with empty histories."""
        metrics = sanity_pomdp.compute_metrics([])
        assert metrics == []
    
    def test_compute_metrics_with_histories(self, sanity_pomdp):
        """Test metrics computation with sample histories."""
        # Create a simple history
        steps = [
            StepData(
                state=0,
                action=0,
                next_state=0,
                observation=0,
                reward=1.0,
                belief=None
            ),
            StepData(
                state=0,
                action=1,
                next_state=1,
                observation=1,
                reward=0.0,
                belief=None
            )
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
            policy_run_data=PolicyRunData(info_variables=[])
        )
        
        metrics = sanity_pomdp.compute_metrics([history])
        assert metrics == []  # SanityPOMDP doesn't implement custom metrics


class TestSanityPOMDPIntegration:
    """Integration tests for SanityPOMDP."""
    
    def test_full_episode_simulation(self, sanity_pomdp):
        """Test a full episode simulation."""
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
        """Test that the environment behaves deterministically."""
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