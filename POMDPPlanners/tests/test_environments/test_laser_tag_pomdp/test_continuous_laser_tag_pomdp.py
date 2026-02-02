"""Tests for the Continuous LaserTag POMDP environment.

Tests cover both ContinuousLaserTagPOMDP and
ContinuousLaserTagPOMDPDiscreteActions, including state transition,
observation model, reward, terminal conditions, metrics, and
registry integration.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
    ContinuousLaserTagPOMDP,
    ContinuousLaserTagPOMDPDiscreteActions,
)


@pytest.fixture
def env():
    """Continuous-action environment with no walls for simpler testing."""
    return ContinuousLaserTagPOMDP(
        discount_factor=0.95,
        walls=[],
        dangerous_areas=[],
    )


@pytest.fixture
def env_discrete():
    """Discrete-action environment with no walls for simpler testing."""
    return ContinuousLaserTagPOMDPDiscreteActions(
        discount_factor=0.95,
        walls=[],
        dangerous_areas=[],
    )


@pytest.fixture
def env_default():
    """Environment with default configuration (includes walls)."""
    return ContinuousLaserTagPOMDP(discount_factor=0.95)


@pytest.fixture
def env_discrete_default():
    """Discrete-action environment with default configuration."""
    return ContinuousLaserTagPOMDPDiscreteActions(discount_factor=0.95)


class TestContinuousLaserTagPOMDPInit:
    """Tests for ContinuousLaserTagPOMDP initialization."""

    def test_valid_initialization(self, env):
        """Test basic environment creation.

        Purpose: Validates that the environment initializes correctly.

        Given: Valid constructor parameters with no walls.
        When: ContinuousLaserTagPOMDP is instantiated.
        Then: The environment has correct attribute values.

        Test type: unit
        """
        assert env.discount_factor == 0.95
        assert env.name == "ContinuousLaserTagPOMDP"
        assert env.tag_reward == 10.0
        assert env.tag_penalty == 10.0
        assert env.step_cost == 1.0

    def test_invalid_discount_raises(self):
        """Test that invalid discount factor raises ValueError.

        Purpose: Validates input validation.

        Given: A discount factor outside [0, 1].
        When: ContinuousLaserTagPOMDP is constructed.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="discount_factor"):
            ContinuousLaserTagPOMDP(discount_factor=1.5)

    def test_default_walls(self, env_default):
        """Test that default walls are loaded.

        Purpose: Validates default wall configuration.

        Given: No walls parameter provided.
        When: Environment is created with defaults.
        Then: Walls array is non-empty.

        Test type: unit
        """
        assert env_default.walls.shape[0] > 0
        assert env_default.walls.shape[1] == 4


class TestStateTransitionModel:
    """Tests for the ContinuousLaserTagStateTransitionModel."""

    def test_sample_returns_correct_shape(self, env):
        """Test that transition samples have shape (5,).

        Purpose: Validates transition sample output shape.

        Given: A non-terminal state and a movement action.
        When: Transition model samples are generated.
        Then: Each sample has shape (5,).

        Test type: unit
        """
        np.random.seed(42)
        state = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        action = np.array([1.0, 0.0, 0.0])
        model = env.state_transition_model(state, action)
        samples = model.sample(n_samples=5)
        assert len(samples) == 5
        for s in samples:
            assert s.shape == (5,)

    def test_terminal_state_unchanged(self, env):
        """Test that terminal states are not changed.

        Purpose: Validates that terminal states remain terminal.

        Given: A terminal state.
        When: Transition model samples are generated.
        Then: The returned state is unchanged.

        Test type: unit
        """
        np.random.seed(42)
        state = np.array([5.0, 3.0, 8.0, 5.0, 1.0])
        action = np.array([1.0, 0.0, 0.0])
        model = env.state_transition_model(state, action)
        samples = model.sample(n_samples=3)
        for s in samples:
            np.testing.assert_array_equal(s, state)

    def test_tag_action_succeeds_when_close(self, env):
        """Test that tag succeeds when robot and opponent are close.

        Purpose: Validates tag success condition.

        Given: Robot and opponent at near-identical positions.
        When: Tag action is executed.
        Then: The next state should be terminal.

        Test type: unit
        """
        np.random.seed(42)
        state = np.array([5.0, 3.0, 5.0, 3.0, 0.0])
        action = np.array([0.0, 0.0, 1.0])  # tag
        model = env.state_transition_model(state, action)
        # With very close positions, tag should succeed
        samples = model.sample(n_samples=10)
        terminal_count = sum(1 for s in samples if bool(s[4]))
        assert terminal_count > 0

    def test_movement_changes_robot_position(self, env):
        """Test that movement action changes robot position.

        Purpose: Validates that robot moves with action.

        Given: A non-terminal state and movement action.
        When: Multiple samples are generated.
        Then: Robot positions differ from original (with noise).

        Test type: unit
        """
        np.random.seed(42)
        state = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        action = np.array([1.0, 0.0, 0.0])
        model = env.state_transition_model(state, action)
        samples = model.sample(n_samples=20)
        # Mean robot x should be approximately 6.0
        mean_x = np.mean([s[0] for s in samples])
        assert abs(mean_x - 6.0) < 1.0


class TestObservationModel:
    """Tests for the ContinuousLaserTagObservationModel."""

    def test_sample_returns_correct_shape(self, env):
        """Test that observation samples have shape (8,).

        Purpose: Validates observation output shape.

        Given: A non-terminal state.
        When: Observation model generates samples.
        Then: Each sample has shape (8,).

        Test type: unit
        """
        np.random.seed(42)
        state = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        action = np.array([1.0, 0.0, 0.0])
        model = env.observation_model(state, action)
        samples = model.sample(n_samples=5)
        assert len(samples) == 5
        for obs in samples:
            assert obs.shape == (8,)

    def test_terminal_observation(self, env):
        """Test that terminal state gives terminal observation.

        Purpose: Validates terminal observation is all -1.

        Given: A terminal state.
        When: Observation model generates samples.
        Then: All observations are np.full(8, -1.0).

        Test type: unit
        """
        state = np.array([5.0, 3.0, 8.0, 5.0, 1.0])
        action = np.array([1.0, 0.0, 0.0])
        model = env.observation_model(state, action)
        samples = model.sample(n_samples=3)
        for obs in samples:
            np.testing.assert_array_equal(obs, np.full(8, -1.0))

    def test_observations_non_negative(self, env):
        """Test that non-terminal observations are non-negative.

        Purpose: Validates that clamped measurements are >= 0.

        Given: A non-terminal state.
        When: Observations are sampled.
        Then: All values are >= 0.

        Test type: unit
        """
        np.random.seed(42)
        state = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        action = np.array([1.0, 0.0, 0.0])
        model = env.observation_model(state, action)
        samples = model.sample(n_samples=20)
        for obs in samples:
            assert np.all(obs >= 0)


class TestReward:
    """Tests for the reward function."""

    def test_terminal_state_zero_reward(self, env):
        """Test that terminal states give zero reward.

        Purpose: Validates terminal reward.

        Given: A terminal state.
        When: reward() is called.
        Then: Returns 0.0.

        Test type: unit
        """
        state = np.array([5.0, 3.0, 8.0, 5.0, 1.0])
        action = np.array([1.0, 0.0, 0.0])
        assert env.reward(state, action) == 0.0

    def test_movement_gives_step_cost(self, env):
        """Test that movement-only action gives -step_cost.

        Purpose: Validates step cost for movement actions.

        Given: A non-terminal state and movement action (no tag).
        When: reward() is called.
        Then: Returns -step_cost.

        Test type: unit
        """
        state = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        action = np.array([1.0, 0.0, 0.0])
        r = env.reward(state, action)
        assert r == -env.step_cost

    def test_successful_tag_gives_tag_reward(self, env):
        """Test that successful tag gives positive reward.

        Purpose: Validates tag reward when robot is at opponent position.

        Given: Robot and opponent at same position, tag action.
        When: reward() is called.
        Then: Returns tag_reward - step_cost.

        Test type: unit
        """
        state = np.array([5.0, 3.0, 5.0, 3.0, 0.0])
        action = np.array([0.0, 0.0, 1.0])
        r = env.reward(state, action)
        assert r == env.tag_reward - env.step_cost

    def test_failed_tag_gives_penalty(self, env):
        """Test that failed tag gives penalty.

        Purpose: Validates tag penalty when robot is far from opponent.

        Given: Robot far from opponent, tag action.
        When: reward() is called.
        Then: Returns -tag_penalty - step_cost.

        Test type: unit
        """
        state = np.array([1.0, 1.0, 9.0, 5.0, 0.0])
        action = np.array([0.0, 0.0, 1.0])
        r = env.reward(state, action)
        assert r == -env.tag_penalty - env.step_cost

    def test_dangerous_area_penalty(self):
        """Test that dangerous area adds penalty.

        Purpose: Validates dangerous area penalty application.

        Given: Robot in a dangerous area.
        When: reward() is called.
        Then: Reward includes dangerous area penalty.

        Test type: unit
        """
        env = ContinuousLaserTagPOMDP(
            discount_factor=0.95,
            walls=[],
            dangerous_areas=[(5.0, 3.0)],
            dangerous_area_radius=1.0,
            dangerous_area_penalty=5.0,
        )
        state = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        action = np.array([1.0, 0.0, 0.0])
        r = env.reward(state, action)
        assert r == -env.step_cost - 5.0


class TestRewardBatch:
    """Tests for reward_batch."""

    def test_batch_shape(self, env):
        """Test that reward_batch returns correct shape.

        Purpose: Validates batch reward output shape.

        Given: N states.
        When: reward_batch() is called.
        Then: Returns array of shape (N,).

        Test type: unit
        """
        states = np.array(
            [
                [5.0, 3.0, 8.0, 5.0, 0.0],
                [5.0, 3.0, 5.0, 3.0, 0.0],
                [5.0, 3.0, 8.0, 5.0, 1.0],
            ]
        )
        action = np.array([0.0, 0.0, 1.0])
        rewards = env.reward_batch(states, action)
        assert rewards.shape == (3,)

    def test_batch_terminal_zero(self, env):
        """Test that terminal states in batch give zero reward.

        Purpose: Validates batch terminal handling.

        Given: A batch with one terminal state.
        When: reward_batch() is called.
        Then: Terminal state gets 0.0 reward.

        Test type: unit
        """
        states = np.array([[5.0, 3.0, 8.0, 5.0, 1.0]])
        action = np.array([1.0, 0.0, 0.0])
        rewards = env.reward_batch(states, action)
        assert rewards[0] == 0.0


class TestIsTerminal:
    """Tests for the is_terminal method."""

    def test_non_terminal(self, env):
        """Test non-terminal state.

        Purpose: Validates non-terminal detection.

        Given: A state with terminal flag 0.
        When: is_terminal() is called.
        Then: Returns False.

        Test type: unit
        """
        state = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        assert env.is_terminal(state) is False

    def test_terminal(self, env):
        """Test terminal state.

        Purpose: Validates terminal detection.

        Given: A state with terminal flag 1.
        When: is_terminal() is called.
        Then: Returns True.

        Test type: unit
        """
        state = np.array([5.0, 3.0, 8.0, 5.0, 1.0])
        assert env.is_terminal(state) is True


class TestInitialStateDist:
    """Tests for initial state distribution."""

    def test_samples_valid_states(self, env):
        """Test that initial state samples are valid.

        Purpose: Validates initial state distribution produces valid states.

        Given: A continuous LaserTag environment.
        When: initial_state_dist().sample() is called.
        Then: States have shape (5,), are non-terminal, and positions are in grid.

        Test type: unit
        """
        np.random.seed(42)
        dist = env.initial_state_dist()
        samples = dist.sample(n_samples=20)
        for s in samples:
            assert s.shape == (5,)
            assert s[4] == 0.0  # non-terminal
            assert 0 <= s[0] <= env.grid_size[0]
            assert 0 <= s[1] <= env.grid_size[1]
            assert 0 <= s[2] <= env.grid_size[0]
            assert 0 <= s[3] <= env.grid_size[1]

    def test_fixed_initial_state(self):
        """Test that fixed initial state is returned.

        Purpose: Validates fixed initial_state parameter.

        Given: An environment with a fixed initial state.
        When: initial_state_dist().sample() is called.
        Then: Returns the fixed state.

        Test type: unit
        """
        fixed = np.array([2.0, 3.0, 8.0, 5.0, 0.0])
        env = ContinuousLaserTagPOMDP(
            discount_factor=0.95,
            walls=[],
            initial_state=fixed,
        )
        dist = env.initial_state_dist()
        samples = dist.sample(n_samples=5)
        for s in samples:
            np.testing.assert_array_equal(s, fixed)


class TestSampleNextStep:
    """Tests for the sample_next_step convenience method."""

    def test_returns_tuple_of_three(self, env):
        """Test sample_next_step returns (next_state, observation, reward).

        Purpose: Validates the convenience method output.

        Given: A valid state and action.
        When: sample_next_step() is called.
        Then: Returns a tuple of (ndarray(5), ndarray(8), float).

        Test type: integration
        """
        np.random.seed(42)
        state = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        action = np.array([1.0, 0.0, 0.0])
        ns, obs, r = env.sample_next_step(state, action)
        assert ns.shape == (5,)
        assert obs.shape == (8,)
        assert isinstance(r, float)


class TestDiscreteActionsVariant:
    """Tests for ContinuousLaserTagPOMDPDiscreteActions."""

    def test_get_actions(self, env_discrete):
        """Test that get_actions returns the correct set.

        Purpose: Validates discrete action list.

        Given: A discrete-action environment.
        When: get_actions() is called.
        Then: Returns ["up", "down", "right", "left", "tag"].

        Test type: unit
        """
        actions = env_discrete.get_actions()
        assert actions == ["up", "down", "right", "left", "tag"]

    def test_sample_next_step_with_string_action(self, env_discrete):
        """Test that string actions work with sample_next_step.

        Purpose: Validates discrete action conversion.

        Given: A valid state and a string action.
        When: sample_next_step() is called.
        Then: Returns valid (next_state, observation, reward).

        Test type: integration
        """
        np.random.seed(42)
        state = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        ns, obs, r = env_discrete.sample_next_step(state, "right")
        assert ns.shape == (5,)
        assert obs.shape == (8,)
        assert isinstance(r, float)

    def test_tag_action(self, env_discrete):
        """Test the 'tag' discrete action.

        Purpose: Validates that tag action maps to [0, 0, 1].

        Given: Robot and opponent at same position.
        When: 'tag' action is used.
        Then: Reward includes tag_reward.

        Test type: unit
        """
        state = np.array([5.0, 3.0, 5.0, 3.0, 0.0])
        r = env_discrete.reward(state, "tag")
        assert r == env_discrete.tag_reward - env_discrete.step_cost

    def test_reward_batch_with_string_action(self, env_discrete):
        """Test reward_batch with discrete actions.

        Purpose: Validates batch reward with string actions.

        Given: A batch of states and a string action.
        When: reward_batch() is called.
        Then: Returns correct shape and values.

        Test type: unit
        """
        states = np.array(
            [
                [5.0, 3.0, 8.0, 5.0, 0.0],
                [5.0, 3.0, 8.0, 5.0, 1.0],
            ]
        )
        rewards = env_discrete.reward_batch(states, "right")
        assert rewards.shape == (2,)
        assert rewards[1] == 0.0  # terminal


class TestMetrics:
    """Tests for metric computation."""

    def test_get_metric_names(self, env):
        """Test that metric names are returned.

        Purpose: Validates metric name list.

        Given: An environment.
        When: get_metric_names() is called.
        Then: Returns a non-empty list of strings.

        Test type: unit
        """
        names = env.get_metric_names()
        assert len(names) > 0
        assert "tag_success_rate" in names

    def test_compute_metrics_empty(self, env):
        """Test compute_metrics with empty histories.

        Purpose: Validates empty input handling.

        Given: An empty history list.
        When: compute_metrics() is called.
        Then: Returns empty list.

        Test type: unit
        """
        assert env.compute_metrics([]) == []


class TestIsEqualObservation:
    """Tests for observation comparison."""

    def test_equal_observations(self, env):
        """Test that equal observations are detected.

        Purpose: Validates observation equality check.

        Given: Two identical observations.
        When: is_equal_observation() is called.
        Then: Returns True.

        Test type: unit
        """
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        assert env.is_equal_observation(obs, obs.copy())

    def test_different_observations(self, env):
        """Test that different observations are not equal.

        Purpose: Validates observation inequality check.

        Given: Two different observations.
        When: is_equal_observation() is called.
        Then: Returns False.

        Test type: unit
        """
        obs1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        obs2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0])
        assert not env.is_equal_observation(obs1, obs2)


class TestEnvironmentRegistry:
    """Tests for environment registry integration."""

    def test_continuous_laser_tag_in_registry(self):
        """Test that ContinuousLaserTagPOMDP is in the registry.

        Purpose: Validates registry integration.

        Given: The environment registry.
        When: Checked for ContinuousLaserTagPOMDP.
        Then: Both variants are present.

        Test type: integration
        """
        from POMDPPlanners.environments import ENVIRONMENT_REGISTRY

        assert "ContinuousLaserTagPOMDP" in ENVIRONMENT_REGISTRY
        assert "ContinuousLaserTagPOMDPDiscreteActions" in ENVIRONMENT_REGISTRY

    def test_get_environment_factory(self):
        """Test that get_environment works for the new environments.

        Purpose: Validates factory function integration.

        Given: The get_environment factory.
        When: Called with "ContinuousLaserTagPOMDP".
        Then: Returns a ContinuousLaserTagPOMDP instance.

        Test type: integration
        """
        from POMDPPlanners.environments import get_environment

        env = get_environment("ContinuousLaserTagPOMDP", discount_factor=0.95)
        assert isinstance(env, ContinuousLaserTagPOMDP)

        env_d = get_environment("ContinuousLaserTagPOMDPDiscreteActions", discount_factor=0.95)
        assert isinstance(env_d, ContinuousLaserTagPOMDPDiscreteActions)
