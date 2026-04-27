"""Tests for the Continuous LaserTag POMDP environment.

Tests cover both ContinuousLaserTagPOMDP and
ContinuousLaserTagPOMDPDiscreteActions, including state transition,
observation model, reward, terminal conditions, metrics, and
registry integration.
"""

import math

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.policy import PolicyRunData
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.laser_tag_pomdp import _native
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
    ContinuousLaserTagPOMDP,
    ContinuousLaserTagPOMDPDiscreteActions,
)
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.planners.planners_utils.rollout import python_random_rollout
from POMDPPlanners.tests.test_utils.confidence_interval_utils import (
    verify_metrics_within_confidence_intervals,
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


class TestSampleNextState:
    """Tests for env.sample_next_state behaviour."""

    def test_sample_returns_correct_shape(self, env):
        """Test that transition samples have shape (5,).

        Purpose: Validates transition sample output shape.

        Given: A non-terminal state and a movement action.
        When: env.sample_next_state is called with n_samples=5.
        Then: Each sample has shape (5,).

        Test type: unit
        """
        np.random.seed(42)
        state = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        action = np.array([1.0, 0.0, 0.0])
        samples = env.sample_next_state(state, action, n_samples=5)
        assert len(samples) == 5
        for s in samples:
            assert s.shape == (5,)

    def test_terminal_state_unchanged(self, env):
        """Test that terminal states are not changed.

        Purpose: Validates that terminal states remain terminal.

        Given: A terminal state.
        When: env.sample_next_state is called.
        Then: The returned state is unchanged.

        Test type: unit
        """
        np.random.seed(42)
        state = np.array([5.0, 3.0, 8.0, 5.0, 1.0])
        action = np.array([1.0, 0.0, 0.0])
        samples = env.sample_next_state(state, action, n_samples=3)
        for s in samples:
            np.testing.assert_array_equal(s, state)

    def test_tag_action_succeeds_when_close(self, env):
        """Test that tag succeeds when robot and opponent are close.

        Purpose: Validates tag success condition.

        Given: Robot and opponent at near-identical positions.
        When: Tag action is executed via env.sample_next_state.
        Then: At least one of the next states is terminal.

        Test type: unit
        """
        np.random.seed(42)
        state = np.array([5.0, 3.0, 5.0, 3.0, 0.0])
        action = np.array([0.0, 0.0, 1.0])  # tag
        samples = env.sample_next_state(state, action, n_samples=10)
        terminal_count = sum(1 for s in samples if bool(s[4]))
        assert terminal_count > 0

    def test_movement_changes_robot_position(self, env):
        """Test that movement action changes robot position.

        Purpose: Validates that robot moves with action.

        Given: A non-terminal state and movement action.
        When: Multiple samples are generated via env.sample_next_state.
        Then: Mean robot x position is approximately the original + action[0].

        Test type: unit
        """
        np.random.seed(42)
        state = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        action = np.array([1.0, 0.0, 0.0])
        samples = env.sample_next_state(state, action, n_samples=20)
        # Mean robot x should be approximately 6.0
        mean_x = np.mean([s[0] for s in samples])
        assert abs(mean_x - 6.0) < 1.0


class TestSampleObservation:
    """Tests for env.sample_observation behaviour."""

    def test_sample_returns_correct_shape(self, env):
        """Test that observation samples have shape (8,).

        Purpose: Validates observation output shape.

        Given: A non-terminal state.
        When: env.sample_observation is called with n_samples=5.
        Then: Each sample has shape (8,).

        Test type: unit
        """
        np.random.seed(42)
        state = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        action = np.array([1.0, 0.0, 0.0])
        samples = env.sample_observation(state, action, n_samples=5)
        assert len(samples) == 5
        for obs in samples:
            assert obs.shape == (8,)

    def test_terminal_observation(self, env):
        """Test that terminal state gives terminal observation.

        Purpose: Validates terminal observation is all -1.

        Given: A terminal state.
        When: env.sample_observation is called.
        Then: All observations are np.full(8, -1.0).

        Test type: unit
        """
        state = np.array([5.0, 3.0, 8.0, 5.0, 1.0])
        action = np.array([1.0, 0.0, 0.0])
        samples = env.sample_observation(state, action, n_samples=3)
        for obs in samples:
            np.testing.assert_array_equal(obs, np.full(8, -1.0))

    def test_observations_non_negative(self, env):
        """Test that non-terminal observations are non-negative.

        Purpose: Validates that clamped measurements are >= 0.

        Given: A non-terminal state.
        When: Observations are sampled via env.sample_observation.
        Then: All values are >= 0.

        Test type: unit
        """
        np.random.seed(42)
        state = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        action = np.array([1.0, 0.0, 0.0])
        samples = env.sample_observation(state, action, n_samples=20)
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

    def test_discrete_sample_next_state_translates_action(self, env_discrete):
        """Test that discrete sample_next_state translates string action to vector.

        Purpose: Validates that the discrete-actions subclass routes string actions
            through action_to_vector before sampling, producing a valid next state.

        Given: A discrete-action environment, a non-terminal state, and the "up"
            string action.
        When: env.sample_next_state(state, "up") is called.
        Then: The returned next state has shape (5,) and is non-terminal, and the
            mean robot y-coordinate over many samples is approximately state[1] + 1
            (since "up" maps to [0, 1, 0]).

        Test type: unit
        """
        np.random.seed(42)
        state = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        ns = env_discrete.sample_next_state(state, "up")
        assert ns.shape == (5,)
        samples = env_discrete.sample_next_state(state, "up", n_samples=200)
        mean_y = np.mean([s[1] for s in samples])
        assert abs(mean_y - 4.0) < 0.5

    def test_discrete_sample_observation_translates_action(self, env_discrete):
        """Test that discrete sample_observation translates string action to vector.

        Purpose: Validates that the discrete-actions subclass routes string actions
            through action_to_vector when generating observations.

        Given: A discrete-action environment, a non-terminal state, and the "right"
            string action.
        When: env.sample_observation(state, "right") is called.
        Then: The returned observation has shape (8,) and all values are >= 0.

        Test type: unit
        """
        np.random.seed(42)
        state = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        obs = env_discrete.sample_observation(state, "right")
        assert obs.shape == (8,)
        assert np.all(obs >= 0)


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

    def test_compute_metrics_discrete_actions(self, env_discrete):
        """Test compute_metrics with discrete string actions.

        Purpose: Validates that compute_metrics handles string actions from
        ContinuousLaserTagPOMDPDiscreteActions without raising an error.

        Given: A ContinuousLaserTagPOMDPDiscreteActions environment and a history
            containing discrete string actions (e.g. 'left', 'tag').
        When: compute_metrics() is called on the history.
        Then: Returns a non-empty list of MetricValue without raising ValueError.

        Test type: unit
        """
        np.random.seed(42)
        initial_state = env_discrete.initial_state_dist().sample()[0]
        dummy_belief = WeightedParticleBelief(
            particles=[initial_state], log_weights=np.array([1.0])
        )

        steps = []
        state = initial_state
        for action in ["left", "right", "tag"]:
            next_state, obs, reward = env_discrete.sample_next_step(state, action)
            steps.append(
                StepData(
                    state=state,
                    action=action,
                    next_state=next_state,
                    observation=obs,
                    reward=reward,
                    belief=dummy_belief,
                )
            )
            state = next_state

        history = History(
            history=steps,
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=len(steps),
            reach_terminal_state=False,
            policy_run_data=[PolicyRunData(info_variables=[])],
        )

        metrics = env_discrete.compute_metrics([history])
        assert len(metrics) > 0
        metrics_dict = {m.name: m for m in metrics}
        assert "tag_success_rate" in metrics_dict

    def test_compute_metrics_with_episode_data(self, env):
        """Test compute_metrics with a multi-step continuous-action history.

        Purpose: Validates that compute_metrics returns all 7 metric names with
        finite float values when given a history with continuous numeric actions.

        Given: A ContinuousLaserTagPOMDP environment and a History containing
            movement steps and a successful tag step.
        When: compute_metrics() is called on the history.
        Then: All 7 expected metric names are present and values are finite.

        Test type: integration
        """
        np.random.seed(42)
        dummy_belief = WeightedParticleBelief(
            particles=[np.array([5.0, 3.0, 8.0, 5.0, 0.0])],
            log_weights=np.array([1.0]),
        )

        steps = []
        # Movement step
        state = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        action = np.array([1.0, 0.0, 0.0])
        ns, obs, r = env.sample_next_step(state, action)
        steps.append(
            StepData(
                state=state,
                action=action,
                next_state=ns,
                observation=obs,
                reward=r,
                belief=dummy_belief,
            )
        )

        # Another movement step
        state = ns
        action = np.array([0.0, 1.0, 0.0])
        ns, obs, r = env.sample_next_step(state, action)
        steps.append(
            StepData(
                state=state,
                action=action,
                next_state=ns,
                observation=obs,
                reward=r,
                belief=dummy_belief,
            )
        )

        # Successful tag: place robot at opponent position, then tag
        state = np.array([5.0, 3.0, 5.0, 3.0, 0.0])
        action = np.array([0.0, 0.0, 1.0])
        r = env.reward(state, action)
        ns = np.array([5.0, 3.0, 5.0, 3.0, 1.0])
        obs = np.full(8, -1.0)
        steps.append(
            StepData(
                state=state,
                action=action,
                next_state=ns,
                observation=obs,
                reward=r,
                belief=dummy_belief,
            )
        )

        history = History(
            history=steps,
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=len(steps),
            reach_terminal_state=True,
            policy_run_data=[PolicyRunData(info_variables=[])],
        )

        metrics = env.compute_metrics([history])
        expected_names = env.get_metric_names()
        assert len(expected_names) == 7
        metrics_dict = {m.name: m for m in metrics}
        for name in expected_names:
            assert name in metrics_dict, f"Missing metric: {name}"
            assert np.isfinite(
                metrics_dict[name].value
            ), f"Non-finite value for {name}: {metrics_dict[name].value}"

    def test_compute_metrics_discrete_actions_tag_detection(self, env_discrete):
        """Test that discrete 'tag' action is correctly detected as a failed tag.

        Purpose: Validates that _count_episode_metrics in the discrete variant
        correctly converts the 'tag' string action to [0, 0, 1] and detects
        the tag flag, resulting in average_failed_tag_attempts > 0.

        Given: A ContinuousLaserTagPOMDPDiscreteActions environment and a
            History where the robot is far from the opponent and uses 'tag'.
        When: compute_metrics() is called on the history.
        Then: average_failed_tag_attempts metric value is > 0.

        Test type: unit
        """
        np.random.seed(42)
        dummy_belief = WeightedParticleBelief(
            particles=[np.array([1.0, 1.0, 9.0, 5.0, 0.0])],
            log_weights=np.array([1.0]),
        )

        # Robot far from opponent — tag will fail
        state = np.array([1.0, 1.0, 9.0, 5.0, 0.0])
        ns, obs, r = env_discrete.sample_next_step(state, "tag")
        steps = [
            StepData(
                state=state,
                action="tag",
                next_state=ns,
                observation=obs,
                reward=r,
                belief=dummy_belief,
            ),
        ]

        history = History(
            history=steps,
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=1,
            reach_terminal_state=False,
            policy_run_data=[PolicyRunData(info_variables=[])],
        )

        metrics = env_discrete.compute_metrics([history])
        metrics_dict = {m.name: m for m in metrics}
        assert metrics_dict["average_failed_tag_attempts"].value > 0

    def test_compute_metrics_dangerous_area_steps(self):
        """Test that dangerous area steps are counted in metrics.

        Purpose: Validates that average_dangerous_area_steps is > 0 when the
        robot state is inside a known dangerous area.

        Given: An environment with a dangerous area at (5.0, 3.0) with radius 1.0
            and a History where the robot is at that position.
        When: compute_metrics() is called.
        Then: average_dangerous_area_steps > 0.

        Test type: unit
        """
        np.random.seed(42)
        env = ContinuousLaserTagPOMDP(
            discount_factor=0.95,
            walls=[],
            dangerous_areas=[(5.0, 3.0)],
            dangerous_area_radius=1.0,
            dangerous_area_penalty=5.0,
        )
        dummy_belief = WeightedParticleBelief(
            particles=[np.array([5.0, 3.0, 8.0, 5.0, 0.0])],
            log_weights=np.array([1.0]),
        )

        # Robot inside the dangerous area at (5.0, 3.0)
        state = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        action = np.array([0.5, 0.0, 0.0])
        ns, obs, r = env.sample_next_step(state, action)
        steps = [
            StepData(
                state=state,
                action=action,
                next_state=ns,
                observation=obs,
                reward=r,
                belief=dummy_belief,
            ),
        ]

        history = History(
            history=steps,
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=1,
            reach_terminal_state=False,
            policy_run_data=[PolicyRunData(info_variables=[])],
        )

        metrics = env.compute_metrics([history])
        metrics_dict = {m.name: m for m in metrics}
        assert metrics_dict["average_dangerous_area_steps"].value > 0

    def test_compute_metrics_multiple_episodes(self, env):
        """Test compute_metrics averages across multiple episodes with CIs.

        Purpose: Validates that metric values are averages across episodes and
        that confidence bounds are finite (not -inf/inf) when multiple episodes
        are provided.

        Given: A ContinuousLaserTagPOMDP environment with 3 episodes having
            different outcomes (movement only, failed tag, successful tag).
        When: compute_metrics() is called with all 3 histories.
        Then: Metric values are averages, and confidence bounds are finite.

        Test type: integration
        """
        np.random.seed(42)
        dummy_belief = WeightedParticleBelief(
            particles=[np.array([5.0, 3.0, 8.0, 5.0, 0.0])],
            log_weights=np.array([1.0]),
        )

        def _make_history(steps, terminal):
            return History(
                history=steps,
                discount_factor=0.95,
                average_state_sampling_time=0.0,
                average_action_time=0.0,
                average_observation_time=0.0,
                average_belief_update_time=0.0,
                average_reward_time=0.0,
                actual_num_steps=len(steps),
                reach_terminal_state=terminal,
                policy_run_data=[PolicyRunData(info_variables=[])],
            )

        # Episode 1: movement only (2 steps)
        s1 = np.array([5.0, 3.0, 8.0, 5.0, 0.0])
        a1 = np.array([1.0, 0.0, 0.0])
        ns1, obs1, r1 = env.sample_next_step(s1, a1)
        s2 = ns1
        a2 = np.array([0.0, 1.0, 0.0])
        ns2, obs2, r2 = env.sample_next_step(s2, a2)
        h1 = _make_history(
            [
                StepData(s1, a1, ns1, obs1, r1, dummy_belief),
                StepData(s2, a2, ns2, obs2, r2, dummy_belief),
            ],
            terminal=False,
        )

        # Episode 2: failed tag (robot far from opponent)
        s3 = np.array([1.0, 1.0, 9.0, 5.0, 0.0])
        a3 = np.array([0.0, 0.0, 1.0])
        r3 = env.reward(s3, a3)  # failed tag → negative reward
        ns3, obs3, _ = env.sample_next_step(s3, a3)
        h2 = _make_history(
            [
                StepData(s3, a3, ns3, obs3, r3, dummy_belief),
            ],
            terminal=False,
        )

        # Episode 3: successful tag (robot at opponent)
        s4 = np.array([5.0, 3.0, 5.0, 3.0, 0.0])
        a4 = np.array([0.0, 0.0, 1.0])
        r4 = env.reward(s4, a4)  # successful tag → positive reward
        ns4 = np.array([5.0, 3.0, 5.0, 3.0, 1.0])
        obs4 = np.full(8, -1.0)
        h3 = _make_history(
            [
                StepData(s4, a4, ns4, obs4, r4, dummy_belief),
            ],
            terminal=True,
        )

        metrics = env.compute_metrics([h1, h2, h3])
        assert len(metrics) == 7

        metrics_dict = {m.name: m for m in metrics}

        # Values should be averages
        avg_length = metrics_dict["average_episode_length"].value
        assert avg_length == pytest.approx((2 + 1 + 1) / 3.0)

        # With 3 episodes, confidence bounds should be finite (not -inf/inf)
        verify_metrics_within_confidence_intervals(metrics)
        for m in metrics:
            assert np.isfinite(
                m.lower_confidence_bound
            ), f"{m.name} has non-finite lower bound: {m.lower_confidence_bound}"
            assert np.isfinite(
                m.upper_confidence_bound
            ), f"{m.name} has non-finite upper bound: {m.upper_confidence_bound}"


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


class TestObservationLogProbabilityTerminal:
    """Tests for observation_log_probability terminal-state semantics."""

    def test_observation_log_probability_terminal_state(self) -> None:
        """observation_log_probability handles terminal next_state (-inf for non-terminal obs).

        Purpose: Validates the terminal-sentinel branch of observation_log_probability
            returns log(1)=0 for the terminal observation and -inf otherwise.

        Given: A ContinuousLaserTagPOMDP and a terminal next_state.
        When: Querying log-prob for the terminal sentinel and an arbitrary non-terminal obs.
        Then: First entry is 0.0, second is -inf.

        Test type: unit
        """
        env = ContinuousLaserTagPOMDP(discount_factor=0.95, walls=[], dangerous_areas=[])
        terminal_state = np.array([3.0, 3.0, 8.0, 5.0, 1.0])
        action = np.array([0.0, 0.0, 1.0])
        terminal_obs = np.full(8, -1.0)
        non_terminal_obs = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])

        actual = env.observation_log_probability(
            terminal_state, action, [terminal_obs, non_terminal_obs]
        )

        assert actual.shape == (2,)
        assert actual[0] == 0.0
        assert np.isneginf(actual[1])


class TestContinuousLaserTagNativeRollout:
    """Tests for the native cont_simulate_rollout entry point."""

    def test_continuous_lasertag_native_rollout_matches_python(self) -> None:
        """Native cont_simulate_rollout return matches python_random_rollout.

        Purpose: Validates that cont_simulate_rollout in the C++ _native extension
            produces a discounted return numerically identical to the Python
            python_random_rollout baseline when given the same RNG
            seed and action sequence.

        Given: A ContinuousLaserTagPOMDP with no walls, a fixed initial state, a
            deterministic action sampler that replays a pre-drawn action sequence,
            and the module-level _native RNG seeded to the same value before each
            call.
        When: We compute the rollout return via the Python base-class loop (using
            the deterministic action sampler to ensure both paths see the same
            action sequence) and via env.simulate_random_rollout (which delegates
            to cont_simulate_rollout after pre-sampling actions).
        Then: The two return values agree within atol=1e-9.

        Test type: unit
        """
        env = ContinuousLaserTagPOMDP(
            discount_factor=0.95,
            walls=[],
            dangerous_areas=[],
        )
        state = np.array([3.0, 3.0, 8.0, 5.0, 0.0])
        max_depth = 10
        depth = 2
        steps_left = max_depth - depth

        # Pre-draw the action sequence that both paths will use.
        np.random.seed(0)
        raw_actions = [
            np.ascontiguousarray(
                np.array(
                    [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.0], dtype=np.float64
                )
            )
            for _ in range(steps_left)
        ]

        # A deterministic action sampler that replays the pre-drawn actions.
        class _ReplayActionSampler(ActionSampler):
            def __init__(self, actions):
                self._actions = list(actions)
                self._idx = 0

            def sample(self, belief_node=None):  # pylint: disable=unused-argument
                act = self._actions[self._idx % len(self._actions)]
                self._idx += 1
                return act

        # Python baseline: call python_random_rollout directly.
        _native.set_seed(0)
        python_result = python_random_rollout(
            state=state,
            depth=depth,
            action_sampler=_ReplayActionSampler(raw_actions),
            environment=env,
            discount_factor=env.discount_factor,
            max_depth=max_depth,
        )

        # Native path: call env.simulate_random_rollout (delegates to cont_simulate_rollout).
        _native.set_seed(0)
        native_result = env.simulate_random_rollout(
            state=state,
            action_sampler=_ReplayActionSampler(raw_actions),
            max_depth=max_depth,
            discount_factor=env.discount_factor,
            depth=depth,
        )

        np.testing.assert_allclose(native_result, python_result, atol=1e-9)

    def test_continuous_lasertag_native_rollout_terminal_state_returns_zero(self) -> None:
        """Native rollout returns 0.0 for a terminal initial state.

        Purpose: Validates the early-exit path when the initial state is terminal.

        Given: A ContinuousLaserTagPOMDP and a terminal state (terminal_flag=1).
        When: simulate_random_rollout is called.
        Then: Returns 0.0.

        Test type: unit
        """

        class _DummySampler:
            def sample(self, belief_node=None):  # pylint: disable=unused-argument
                return np.array([1.0, 0.0, 0.0])

        env = ContinuousLaserTagPOMDP(discount_factor=0.95, walls=[], dangerous_areas=[])
        terminal_state = np.array([5.0, 3.0, 5.0, 3.0, 1.0])
        result = env.simulate_random_rollout(
            state=terminal_state,
            action_sampler=_DummySampler(),
            max_depth=10,
            discount_factor=env.discount_factor,
            depth=0,
        )
        assert result == 0.0

    def test_continuous_lasertag_native_rollout_zero_depth_returns_zero(self) -> None:
        """Native rollout returns 0.0 when depth equals max_depth.

        Purpose: Validates the early-exit path when no steps remain.

        Given: A ContinuousLaserTagPOMDP and depth == max_depth.
        When: simulate_random_rollout is called.
        Then: Returns 0.0.

        Test type: unit
        """

        class _DummySampler:
            def sample(self, belief_node=None):  # pylint: disable=unused-argument
                return np.array([1.0, 0.0, 0.0])

        env = ContinuousLaserTagPOMDP(discount_factor=0.95, walls=[], dangerous_areas=[])
        state = np.array([3.0, 3.0, 8.0, 5.0, 0.0])
        result = env.simulate_random_rollout(
            state=state,
            action_sampler=_DummySampler(),
            max_depth=5,
            discount_factor=env.discount_factor,
            depth=5,
        )
        assert result == 0.0

    def test_discrete_actions_native_rollout_matches_python(self) -> None:
        """DiscreteActions variant native rollout matches Python baseline.

        Purpose: Validates that ContinuousLaserTagPOMDPDiscreteActions.simulate_random_rollout
            produces the same result as the Python loop for a deterministic action
            sequence.

        Given: A ContinuousLaserTagPOMDPDiscreteActions, a fixed initial state, a
            deterministic string-action sampler, and identical _native RNG seeds.
        When: Both Python and native paths are called.
        Then: The two return values agree within atol=1e-9.

        Test type: unit
        """
        env = ContinuousLaserTagPOMDPDiscreteActions(
            discount_factor=0.95,
            walls=[],
            dangerous_areas=[],
        )
        state = np.array([3.0, 3.0, 8.0, 5.0, 0.0])
        max_depth = 8
        depth = 1
        steps_left = max_depth - depth

        action_sequence = ["right", "up", "left", "down", "tag", "right", "up"] * 10
        action_sequence = action_sequence[:steps_left]

        class _DiscreteSampler(ActionSampler):
            def __init__(self, actions):
                self._actions = list(actions)
                self._idx = 0

            def sample(self, belief_node=None):  # pylint: disable=unused-argument
                act = self._actions[self._idx % len(self._actions)]
                self._idx += 1
                return act

        _native.set_seed(7)
        python_result = python_random_rollout(
            state=state,
            depth=depth,
            action_sampler=_DiscreteSampler(action_sequence),
            environment=env,
            discount_factor=env.discount_factor,
            max_depth=max_depth,
        )

        _native.set_seed(7)
        native_result = env.simulate_random_rollout(
            state=state,
            action_sampler=_DiscreteSampler(action_sequence),
            max_depth=max_depth,
            discount_factor=env.discount_factor,
            depth=depth,
        )

        np.testing.assert_allclose(native_result, python_result, atol=1e-9)


class TestObservationLogProbabilityLowDensityB1:
    """Regression tests for B1 — scalar/batch log-prob underflow asymmetry."""

    def test_scalar_log_probability_low_density_obs_matches_batch(self) -> None:
        """Scalar observation_log_probability matches batch path on a low-density obs.

        Purpose: Regression for B1. The scalar ``observation_log_probability``
            previously round-tripped through ``np.exp(log_pdf)`` inside the C++
            ``probability`` kernel and then took ``np.log`` in Python; for an
            observation whose ``log_pdf`` is well below IEEE-754 double-precision
            ``exp`` underflow (~ -745), ``exp(log_pdf)`` collapses to ``0.0`` and
            the wrapper returned ``-inf``. Meanwhile the batched
            ``observation_log_probability_per_state`` returns ``log_pdf``
            directly via ``batch_log_likelihood``, never paying the round-trip.
            After the fix, the scalar path must return the same finite,
            large-negative value as the batch path.

        Given: A ContinuousLaserTagPOMDP with no walls, a non-terminal
            next_state, and a far-from-mean (non-terminal-sentinel) observation
            chosen so that ``log_pdf`` is approximately -1e6 — well past the
            ``exp`` underflow boundary.
        When: We compute the log-probability of that single observation via
            both the scalar ``env.observation_log_probability`` (passing a
            list-of-one) and the batch
            ``env.observation_log_probability_per_state`` (passing a
            list-of-one next_state).
        Then: The scalar value is finite (not ``-inf``), is large and negative
            (less than ``-1e5``), and agrees with the batch value within
            ``atol=1e-6``.

        Test type: unit
        """
        env = ContinuousLaserTagPOMDP(discount_factor=0.95, walls=[], dangerous_areas=[])
        action = np.array([1.0, 0.0, 0.0])
        next_state = np.array([3.0, 3.0, 8.0, 5.0, 0.0])

        # Compute the kernel mean so we can place the observation far from it
        # without depending on internal layout. log_pdf for an 8-D Gaussian
        # with sigma=1 is 8 * log_norm_1d_ - 0.5 * sum_sq_diff. With per-dim
        # diff = 500, sum_sq_diff = 8 * 250000 = 2e6, so log_pdf is ~ -1e6.
        kernel = env._get_obs_kernel(action)  # pylint: disable=protected-access
        kernel.set_next_state(next_state)
        mean = np.asarray(kernel.mean, dtype=np.float64)
        far_obs = mean + 500.0
        # Sanity: this is *not* the terminal sentinel (np.full(8, -1.0)),
        # so we are exercising B1 (low-density Gaussian) and not B2.
        assert not np.allclose(far_obs, -1.0)

        scalar_log_probs = env.observation_log_probability(next_state, action, [far_obs])
        batch_log_probs = env.observation_log_probability_per_state(
            np.asarray([next_state], dtype=np.float64), action, far_obs
        )

        assert scalar_log_probs.shape == (1,)
        assert batch_log_probs.shape == (1,)
        assert np.isfinite(
            scalar_log_probs[0]
        ), f"scalar log-prob underflowed to {scalar_log_probs[0]} — B1 not fixed"
        assert scalar_log_probs[0] < -1e5
        np.testing.assert_allclose(scalar_log_probs[0], batch_log_probs[0], atol=1e-6)

    def test_scalar_log_probability_single_low_density_obs_matches_batch(self) -> None:
        """observation_log_probability_single returns finite value on low-density obs.

        Purpose: Regression for B1 on the scalar fast-path used by POMCPOW's
            ``WeightedParticleBeliefStateUpdate``. Pre-fix, this method called
            ``kernel.probability([obs])[0]`` and clamped to ``-math.inf`` when
            ``prob <= 0.0``, masking the underflow. Post-fix, it must return a
            finite log-prob agreeing with the batch path.

        Given: A ContinuousLaserTagPOMDP with no walls, a non-terminal
            next_state, and a far-from-mean (non-terminal-sentinel) observation
            with ``log_pdf`` well past the ``exp`` underflow boundary.
        When: We compute the log-probability via
            ``env.observation_log_probability_single`` and via
            ``env.observation_log_probability_per_state`` (single-row batch).
        Then: The scalar value is finite, large negative, and agrees with the
            batch value within ``atol=1e-6``.

        Test type: unit
        """
        env = ContinuousLaserTagPOMDP(discount_factor=0.95, walls=[], dangerous_areas=[])
        action = np.array([1.0, 0.0, 0.0])
        next_state = np.array([3.0, 3.0, 8.0, 5.0, 0.0])

        kernel = env._get_obs_kernel(action)  # pylint: disable=protected-access
        kernel.set_next_state(next_state)
        mean = np.asarray(kernel.mean, dtype=np.float64)
        far_obs = mean + 500.0
        assert not np.allclose(far_obs, -1.0)

        scalar_value = env.observation_log_probability_single(next_state, action, far_obs)
        batch_log_probs = env.observation_log_probability_per_state(
            np.asarray([next_state], dtype=np.float64), action, far_obs
        )

        assert math.isfinite(
            scalar_value
        ), f"scalar single log-prob underflowed to {scalar_value} — B1 not fixed"
        assert scalar_value < -1e5
        np.testing.assert_allclose(scalar_value, batch_log_probs[0], atol=1e-6)
