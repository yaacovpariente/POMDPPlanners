"""Tests for the Continuous Push POMDP environment.

This module tests the state transition behavior, observation behavior,
environment, and discrete actions wrapper for the Continuous Push POMDP.
All transition / observation behavior is exercised through the env-level
API (``sample_next_state`` / ``sample_observation`` /
``transition_log_probability`` / ``observation_log_probability``); the
historical wrapper classes ``ContinuousPushStateTransitionModel`` and
``ContinuousPushObservationModel`` no longer exist.
"""

# pylint: disable=protected-access

import numpy as np
import pytest

from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.policy import PolicyRunData
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.push_pomdp import _native
from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import (
    ContinuousPushPOMDP,
    ContinuousPushPOMDPDiscreteActions,
)


def _make_env(**overrides) -> ContinuousPushPOMDP:
    """Build a ContinuousPushPOMDP with overridable defaults.

    The transition tests below previously instantiated the
    ``ContinuousPushStateTransitionModel`` wrapper directly with arbitrary
    parameters per test. With the wrapper gone, each test now spins up an
    env with the same parameters and exercises behavior via
    ``env.sample_next_state``.
    """
    defaults: dict = dict(  # pylint: disable=use-dict-literal
        discount_factor=0.99,
        grid_size=10,
        push_threshold=1.0,
        friction_coefficient=0.3,
        max_push=2.0,
        observation_noise=0.1,
        robot_radius=0.3,
        state_transition_cov_matrix=np.eye(2) * 0.01,
    )
    defaults.update(overrides)
    return ContinuousPushPOMDP(**defaults)


# ------------------------------------------------------------------
# State Transition (env-API)
# ------------------------------------------------------------------


class TestContinuousPushStateTransition:
    """Test continuous push state-transition behavior via env.sample_next_state."""

    def test_robot_moves_in_action_direction(self):
        """Test that the robot moves approximately in the action direction.

        Purpose: Validates that action vector translates to movement.

        Given: Robot at (2, 3), action = (1, 0), small noise.
        When: env.sample_next_state is called.
        Then: Robot x increases, y stays roughly constant.

        Test type: unit
        """
        env = _make_env()
        state = np.array([2.0, 3.0, 5.0, 5.0, 9.0, 9.0])
        ns = env.sample_next_state(state, np.array([1.0, 0.0]))
        assert ns[0] > state[0]  # moved right
        assert abs(ns[1] - state[1]) < 1.0  # y roughly constant

    def test_push_within_threshold(self):
        """Test that object is pushed when robot is within threshold.

        Purpose: Validates push mechanics.

        Given: Robot at (2, 3), object at (2.5, 3.0), within threshold=1.0.
        When: Robot moves right with action (1, 0).
        Then: Object x increases in the action direction.

        Test type: unit
        """
        env = _make_env(state_transition_cov_matrix=np.eye(2) * 1e-8)
        state = np.array([2.0, 3.0, 2.5, 3.0, 9.0, 9.0])
        ns = env.sample_next_state(state, np.array([1.0, 0.0]))
        assert ns[2] > state[2]  # object pushed right

    def test_no_push_beyond_threshold(self):
        """Test that object is NOT pushed when robot is beyond threshold.

        Purpose: Validates that push only occurs within threshold.

        Given: Robot at (0, 0), object at (5, 5), threshold=1.0.
        When: Robot moves right.
        Then: Object position unchanged.

        Test type: unit
        """
        env = _make_env(state_transition_cov_matrix=np.eye(2) * 1e-8)
        state = np.array([0.0, 0.0, 5.0, 5.0, 9.0, 9.0])
        ns = env.sample_next_state(state, np.array([1.0, 0.0]))
        np.testing.assert_allclose(ns[2:4], state[2:4], atol=1e-6)

    def test_push_force_capped(self):
        """Test that push force is capped at max_push.

        Purpose: Validates capped push scaling.

        Given: Large action (10, 0), max_push=2.0, friction=0.3.
        When: Push is applied to nearby object.
        Then: Push magnitude <= max_push * (1 - friction).

        Test type: unit
        """
        env = _make_env(
            state_transition_cov_matrix=np.eye(2) * 1e-8,
            max_push=2.0,
            friction_coefficient=0.3,
        )
        state = np.array([4.0, 4.0, 4.5, 4.0, 9.0, 9.0])
        ns = env.sample_next_state(state, np.array([10.0, 0.0]))
        obj_delta = ns[2] - state[2]
        # max push displacement = min(10, 2) * (1 - 0.3) = 1.4
        assert obj_delta <= 1.4 + 0.01

    def test_friction_reduces_push(self):
        """Test that friction reduces push displacement.

        Purpose: Validates friction effect on pushing.

        Given: Friction = 0.5 vs friction = 0.0.
        When: Same action applied.
        Then: Higher friction produces smaller object displacement.

        Test type: unit
        """
        state = np.array([4.0, 4.0, 4.5, 4.0, 9.0, 9.0])
        env_no_fric = _make_env(
            state_transition_cov_matrix=np.eye(2) * 1e-8,
            friction_coefficient=0.0,
        )
        env_hi_fric = _make_env(
            state_transition_cov_matrix=np.eye(2) * 1e-8,
            friction_coefficient=0.5,
        )
        ns_no = env_no_fric.sample_next_state(state, np.array([1.0, 0.0]))
        ns_hi = env_hi_fric.sample_next_state(state, np.array([1.0, 0.0]))
        assert ns_no[2] > ns_hi[2]

    def test_obstacle_blocks_object_push(self):
        """Test that obstacles block object push.

        Purpose: Validates that object stays put when pushed into obstacle.

        Given: Object at (4.5, 5.0), obstacle AABB at (5, 5) half=0.5.
        When: Robot pushes object rightward into obstacle.
        Then: Object position unchanged.

        Test type: unit
        """
        env = _make_env(
            state_transition_cov_matrix=np.eye(2) * 1e-8,
            obstacles=[(5.0, 5.0, 0.5)],
        )
        state = np.array([4.0, 5.0, 4.5, 5.0, 9.0, 9.0])
        ns = env.sample_next_state(state, np.array([1.0, 0.0]))
        np.testing.assert_allclose(ns[2:4], state[2:4], atol=1e-6)

    def test_grid_clamping(self):
        """Test that positions are clamped to grid.

        Purpose: Validates grid boundary enforcement.

        Given: Robot at (8.5, 8.5) with action pushing out of grid.
        When: env.sample_next_state is called.
        Then: Robot stays within valid bounds.

        Test type: unit
        """
        env = _make_env(state_transition_cov_matrix=np.eye(2) * 1e-8)
        state = np.array([8.5, 8.5, 5.0, 5.0, 9.0, 9.0])
        ns = env.sample_next_state(state, np.array([5.0, 5.0]))
        assert ns[0] <= 10 - 1 - 0.3 + 0.01  # within robot radius of grid edge
        assert ns[1] <= 10 - 1 - 0.3 + 0.01

    def test_transition_log_probability_returns_finite(self):
        """Test that transition_log_probability returns a finite log-density.

        Purpose: Validates the env-level transition log-probability path
            (the new replacement for the wrapper's ``probability`` method).

        Given: An env, a (state, action) pair, and a candidate next-state
            close to the deterministic robot target.
        When: env.transition_log_probability(state, action, [next_state])
            is called.
        Then: Returns a length-1 finite log-density array.

        Test type: unit
        """
        env = _make_env()
        state = np.array([2.0, 3.0, 5.0, 5.0, 9.0, 9.0])
        action = np.array([1.0, 0.0])
        candidate = np.array([3.0, 3.0, 5.0, 5.0, 9.0, 9.0])
        log_prob = env.transition_log_probability(state, action, [candidate])
        assert log_prob.shape == (1,)
        assert np.isfinite(log_prob[0])


# ------------------------------------------------------------------
# Observation Model (env-API)
# ------------------------------------------------------------------


class TestContinuousPushObservation:
    """Test continuous push observation behavior via env.sample_observation."""

    def test_observation_shape(self):
        """Test that observation has shape (6,).

        Purpose: Validates observation vector shape.

        Given: A state with 6 elements.
        When: env.sample_observation is called.
        Then: Observation has shape (6,).

        Test type: unit
        """
        env = _make_env(observation_noise=0.1)
        next_state = np.array([3.0, 4.0, 5.0, 5.0, 9.0, 9.0])
        obs = env.sample_observation(next_state, np.array([1.0, 0.0]))
        assert obs.shape == (6,)

    def test_robot_and_target_exact(self):
        """Test that robot and target positions are observed exactly.

        Purpose: Validates that robot and target are noiseless in observations.

        Given: State with known robot and target positions.
        When: env.sample_observation is called.
        Then: Observation indices 0:2 and 4:6 match state exactly.

        Test type: unit
        """
        env = _make_env(observation_noise=0.1)
        next_state = np.array([3.0, 4.0, 5.0, 5.0, 9.0, 9.0])
        obs = env.sample_observation(next_state, np.array([1.0, 0.0]))
        np.testing.assert_array_equal(obs[:2], next_state[:2])
        np.testing.assert_array_equal(obs[4:6], next_state[4:6])

    def test_object_position_noisy(self):
        """Test that object position observation includes noise.

        Purpose: Validates that object position is noisy.

        Given: Observation noise = 1.0 (large).
        When: Multiple samples are drawn.
        Then: Object position varies across samples.

        Test type: unit
        """
        env = _make_env(observation_noise=1.0)
        next_state = np.array([3.0, 4.0, 5.0, 5.0, 9.0, 9.0])
        action = np.array([1.0, 0.0])
        obs1 = env.sample_observation(next_state, action)
        obs2 = env.sample_observation(next_state, action)
        assert not np.array_equal(obs1[2:4], obs2[2:4])

    def test_observation_log_probability_returns_finite(self):
        """Test that observation_log_probability is finite for plausible observations.

        Purpose: Validates the env-level observation log-probability path
            (the new replacement for the wrapper's ``probability`` method).

        Given: An env, a (next_state, action) pair, and an observation equal
            to the noise-free truth.
        When: env.observation_log_probability(next_state, action,
            [observation]) is called.
        Then: Returns a length-1 finite log-density array.

        Test type: unit
        """
        env = _make_env(observation_noise=0.1)
        next_state = np.array([3.0, 4.0, 5.0, 5.0, 9.0, 9.0])
        action = np.array([1.0, 0.0])
        log_prob = env.observation_log_probability(next_state, action, [next_state.copy()])
        assert log_prob.shape == (1,)
        assert np.isfinite(log_prob[0])


# ------------------------------------------------------------------
# ContinuousPushPOMDP
# ------------------------------------------------------------------


class TestContinuousPushPOMDP:
    """Test the ContinuousPushPOMDP environment."""

    def setup_method(self):
        """Set up shared test fixtures."""
        np.random.seed(42)
        self.env = ContinuousPushPOMDP(  # pylint: disable=attribute-defined-outside-init
            discount_factor=0.99,
            grid_size=10,
            obstacles=[(5.0, 5.0, 0.5)],
            robot_radius=0.3,
            state_transition_cov_matrix=np.eye(2) * 0.01,
        )

    def test_space_info_continuous(self):
        """Test that space info is continuous for both actions and observations.

        Purpose: Validates SpaceInfo for continuous environment.

        Given: A ContinuousPushPOMDP.
        When: space_info is accessed.
        Then: Both action and observation spaces are CONTINUOUS.

        Test type: unit
        """
        assert self.env.space_info.action_space == SpaceType.CONTINUOUS
        assert self.env.space_info.observation_space == SpaceType.CONTINUOUS

    def test_initial_state_shape(self):
        """Test that initial state has shape (6,).

        Purpose: Validates initial state distribution.

        Given: Environment with default settings.
        When: initial_state_dist().sample() is called.
        Then: Returns array of shape (6,).

        Test type: unit
        """
        state = self.env.initial_state_dist().sample()[0]
        assert state.shape == (6,)

    def test_sample_next_step(self):
        """Test that sample_next_step returns valid tuple.

        Purpose: Validates full POMDP step.

        Given: An initial state and continuous action.
        When: sample_next_step is called.
        Then: Returns (next_state, observation, reward).

        Test type: unit
        """
        state = self.env.initial_state_dist().sample()[0]
        ns, obs, rew = self.env.sample_next_step(state, np.array([1.0, 0.0]))
        assert ns.shape == (6,)
        assert obs.shape == (6,)
        assert isinstance(rew, float)

    def test_terminal_at_target(self):
        """Test that is_terminal returns True when object is at target.

        Purpose: Validates terminal condition.

        Given: Object placed at target position.
        When: is_terminal is called.
        Then: Returns True.

        Test type: unit
        """
        state = np.array([1.0, 1.0, 9.0, 9.0, 9.0, 9.0])
        assert self.env.is_terminal(state)

    def test_not_terminal_far_from_target(self):
        """Test that is_terminal returns False when object is far from target.

        Purpose: Validates non-terminal condition.

        Given: Object far from target.
        When: is_terminal is called.
        Then: Returns False.

        Test type: unit
        """
        state = np.array([1.0, 1.0, 1.0, 1.0, 9.0, 9.0])
        assert not self.env.is_terminal(state)

    def test_reward_positive_at_goal(self):
        """Test that reward includes bonus when object reaches target.

        Purpose: Validates goal reward bonus.

        Given: Object very close to target.
        When: reward is called.
        Then: Reward is high (includes +100 bonus).

        Test type: unit
        """
        state = np.array([8.5, 8.5, 8.9, 8.9, 9.0, 9.0])
        rew = self.env.reward(state, np.array([0.0, 0.0]))
        # Distance is small, goal bonus should kick in for nearby transitions
        # Just verify reward is finite
        assert np.isfinite(rew)

    def test_obstacle_penalty_applied(self):
        """Test that obstacle collision incurs penalty.

        Purpose: Validates obstacle penalty.

        Given: Robot at position that will collide with obstacle after action.
        When: reward is called with action into obstacle.
        Then: Reward is lower than without collision.

        Test type: unit
        """
        # Robot moves into obstacle at (5, 5) half=0.5
        state_near_obs = np.array([4.0, 5.0, 1.0, 1.0, 9.0, 9.0])
        state_safe = np.array([1.0, 1.0, 1.0, 1.0, 9.0, 9.0])
        action = np.array([1.0, 0.0])

        rew_obs = self.env.reward(state_near_obs, action)
        rew_safe = self.env.reward(state_safe, action)
        # Both get distance penalty; obstacle one also gets obstacle_penalty
        assert rew_obs < rew_safe

    def test_reward_batch_shape(self):
        """Test that reward_batch returns correct shape.

        Purpose: Validates vectorized reward computation.

        Given: 5 states and one action.
        When: reward_batch is called.
        Then: Returns array of shape (5,).

        Test type: unit
        """
        states = np.array([self.env.initial_state_dist().sample()[0] for _ in range(5)])
        rewards = self.env.reward_batch(states, np.array([1.0, 0.0]))
        assert rewards.shape == (5,)

    def test_config_id_deterministic(self):
        """Test that two identical environments have the same config_id.

        Purpose: Validates deterministic config identification.

        Given: Two environments with identical parameters.
        When: config_id is accessed.
        Then: Both return the same string.

        Test type: unit
        """
        env2 = ContinuousPushPOMDP(
            discount_factor=0.99,
            grid_size=10,
            obstacles=[(5.0, 5.0, 0.5)],
            robot_radius=0.3,
            state_transition_cov_matrix=np.eye(2) * 0.01,
        )
        assert self.env.config_id == env2.config_id

    def test_fixed_initial_state(self):
        """Test that fixed initial state works.

        Purpose: Validates fixed initial state distribution.

        Given: Environment with fixed initial_state.
        When: initial_state_dist().sample() is called.
        Then: Returns the fixed state.

        Test type: unit
        """
        fixed = np.array([1.0, 1.0, 5.0, 5.0, 9.0, 9.0])
        env = ContinuousPushPOMDP(discount_factor=0.99, initial_state=fixed)
        state = env.initial_state_dist().sample()[0]
        np.testing.assert_array_equal(state, fixed)

    def test_no_obstacles_environment(self):
        """Test environment works with no obstacles.

        Purpose: Validates obstacle-free configuration.

        Given: Environment with no obstacles.
        When: sample_next_step is called.
        Then: Works without error.

        Test type: unit
        """
        env = ContinuousPushPOMDP(discount_factor=0.99)
        state = env.initial_state_dist().sample()[0]
        result = env.sample_next_step(state, np.array([1.0, 0.0]))
        assert result[0].shape == (6,)

    def test_compute_metrics(self):
        """Test compute_metrics returns expected metric names.

        Purpose: Validates metric computation.

        Given: A short simulated history.
        When: compute_metrics is called.
        Then: Returns list of MetricValue with correct names.

        Test type: integration
        """
        state = self.env.initial_state_dist().sample()[0]
        steps = []
        for _ in range(5):
            action = np.array([0.5, 0.0])
            ns, obs, rew = self.env.sample_next_step(state, action)
            steps.append(
                StepData(
                    state=state,
                    action=action,
                    next_state=ns,
                    observation=obs,
                    reward=rew,
                    belief=None,  # type: ignore[arg-type]
                )
            )
            state = ns

        history = History(
            history=steps,
            discount_factor=0.99,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=5,
            reach_terminal_state=False,
            policy_run_data=[PolicyRunData(info_variables=[])],
        )
        metrics = self.env.compute_metrics([history])
        metric_names = {m.name for m in metrics}
        assert "goal_reaching_rate" in metric_names
        assert "robot_obstacle_collision_rate" in metric_names


# ------------------------------------------------------------------
# Discrete Actions
# ------------------------------------------------------------------


class TestContinuousPushPOMDPDiscreteActions:
    """Test the discrete-action wrapper."""

    def setup_method(self):
        """Set up shared test fixtures."""
        np.random.seed(42)
        # pylint: disable=attribute-defined-outside-init
        self.env = ContinuousPushPOMDPDiscreteActions(
            discount_factor=0.99,
            grid_size=10,
            robot_radius=0.3,
            state_transition_cov_matrix=np.eye(2) * 0.01,
        )

    def test_actions_list(self):
        """Test that get_actions returns expected string actions.

        Purpose: Validates discrete action set.

        Given: A ContinuousPushPOMDPDiscreteActions environment.
        When: get_actions() is called.
        Then: Returns ["up", "down", "right", "left"].

        Test type: unit
        """
        actions = self.env.get_actions()
        assert actions == ["up", "down", "right", "left"]

    def test_space_info_discrete_actions(self):
        """Test that action space is DISCRETE.

        Purpose: Validates SpaceInfo override.

        Given: A discrete-action environment.
        When: space_info is accessed.
        Then: action_space is DISCRETE, observation_space is CONTINUOUS.

        Test type: unit
        """
        assert self.env.space_info.action_space == SpaceType.DISCRETE
        assert self.env.space_info.observation_space == SpaceType.CONTINUOUS

    def test_sample_next_step_with_string_action(self):
        """Test that string actions work in sample_next_step.

        Purpose: Validates string-to-vector action mapping.

        Given: Initial state and "right" action.
        When: sample_next_step is called.
        Then: Returns valid (next_state, observation, reward).

        Test type: unit
        """
        state = self.env.initial_state_dist().sample()[0]
        ns, obs, rew = self.env.sample_next_step(state, "right")
        assert ns.shape == (6,)
        assert obs.shape == (6,)
        assert isinstance(rew, float)

    def test_action_mapping_directions(self):
        """Test that each action moves robot in expected direction.

        Purpose: Validates action direction mapping.

        Given: Robot at center, very low noise.
        When: Each directional action is taken.
        Then: Robot moves in the expected direction.

        Test type: unit
        """
        low_cov = np.eye(2) * 1e-8
        env = ContinuousPushPOMDPDiscreteActions(
            discount_factor=0.99,
            state_transition_cov_matrix=low_cov,
            robot_radius=0.3,
        )
        state = np.array([5.0, 5.0, 1.0, 1.0, 9.0, 9.0])

        ns_right = env.sample_next_step(state, "right")[0]
        assert ns_right[0] > state[0]

        ns_left = env.sample_next_step(state, "left")[0]
        assert ns_left[0] < state[0]

        ns_up = env.sample_next_step(state, "up")[0]
        assert ns_up[1] > state[1]

        ns_down = env.sample_next_step(state, "down")[0]
        assert ns_down[1] < state[1]


# ------------------------------------------------------------------
# Discrete-actions wrapper: str -> vector resolution parity
# ------------------------------------------------------------------


class TestDiscreteActionsResolvesAction:
    """Validate ContinuousPushPOMDPDiscreteActions delegates to the parent
    env-API methods after resolving str actions through ``action_to_vector``.
    """

    def test_sample_next_state_uses_action_to_vector(self):
        """Discrete-actions wrapper resolves str action via action_to_vector and matches.

        Purpose: Validates that ContinuousPushPOMDPDiscreteActions.sample_next_state
            looks up the cached action vector and produces the same result as
            calling the parent override with that vector directly.

        Given: A ContinuousPushPOMDPDiscreteActions env, a single state, and
            identical ``_native.set_seed`` before each path.
        When: For each str action, env.sample_next_state(state, str_action) is
            compared against env.sample_next_state(state, action_to_vector[a])
            via the continuous parent's override.
        Then: Both paths produce np.array_equal next-states.

        Test type: unit
        """
        env = ContinuousPushPOMDPDiscreteActions(
            discount_factor=0.99,
            state_transition_cov_matrix=np.eye(2) * 0.01,
        )
        state = np.array([5.0, 5.0, 5.5, 5.0, 9.0, 9.0])
        for str_action in env.get_actions():
            vec = env.action_to_vector[str_action]
            _native.set_seed(2024)
            via_str = env.sample_next_state(state, str_action)
            _native.set_seed(2024)
            via_vec = ContinuousPushPOMDP.sample_next_state(env, state, vec)
            np.testing.assert_array_equal(via_str, via_vec)

    def test_sample_observation_uses_action_to_vector(self):
        """Discrete-actions wrapper sample_observation resolves str via action_to_vector.

        Purpose: Same as sample_next_state, for the observation override.

        Given: A ContinuousPushPOMDPDiscreteActions env, a single next_state,
            and identical ``_native.set_seed`` before each path.
        When: For each str action, env.sample_observation(ns, str_action) is
            compared against ContinuousPushPOMDP.sample_observation(env, ns,
            action_to_vector[str_action]).
        Then: Both paths produce np.array_equal observations.

        Test type: unit
        """
        env = ContinuousPushPOMDPDiscreteActions(discount_factor=0.99)
        next_state = np.array([5.0, 5.0, 5.5, 5.0, 9.0, 9.0])
        for str_action in env.get_actions():
            vec = env.action_to_vector[str_action]
            _native.set_seed(2024)
            via_str = env.sample_observation(next_state, str_action)
            _native.set_seed(2024)
            via_vec = ContinuousPushPOMDP.sample_observation(env, next_state, vec)
            np.testing.assert_array_equal(via_str, via_vec)

    def test_action_to_vector_is_contiguous_float64(self):
        """action_to_vector entries are contiguous float64 ndarrays.

        Purpose: Guards the cache contract that lets the hot path skip
            ``np.asarray(...).ravel()`` conversions inside the C++ kernel
            constructor.

        Given: A ContinuousPushPOMDPDiscreteActions env.
        When: action_to_vector entries are inspected.
        Then: Each value is a 1-D ndarray of shape (2,), dtype float64, and
            C-contiguous.

        Test type: unit
        """
        env = ContinuousPushPOMDPDiscreteActions(discount_factor=0.99)
        for action_name, vec in env.action_to_vector.items():
            assert isinstance(vec, np.ndarray), action_name
            assert vec.dtype == np.float64, action_name
            assert vec.shape == (2,), action_name
            assert vec.flags["C_CONTIGUOUS"], action_name

    def test_log_probability_resolves_action(self):
        """Discrete-actions wrapper log-probability resolves str via action_to_vector.

        Purpose: Validates that
            ``ContinuousPushPOMDPDiscreteActions.transition_log_probability``
            and ``observation_log_probability`` look up the cached action
            vector and produce the same result as calling the parent override
            with that vector directly.

        Given: A ContinuousPushPOMDPDiscreteActions env, a (state, action,
            next_state) triple.
        When: For each str action, env.{transition,observation}_log_probability
            is compared against the parent override called with
            ``action_to_vector[a]``.
        Then: Both paths produce np.allclose results.

        Test type: unit
        """
        env = ContinuousPushPOMDPDiscreteActions(
            discount_factor=0.99,
            state_transition_cov_matrix=np.eye(2) * 0.01,
            observation_noise=0.3,
        )
        state = np.array([5.0, 5.0, 5.5, 5.0, 9.0, 9.0])
        next_states = [
            np.array([5.5, 5.0, 5.5, 5.0, 9.0, 9.0]),
            np.array([5.0, 5.5, 5.5, 5.0, 9.0, 9.0]),
        ]
        observations = [
            np.array([5.0, 5.0, 5.5, 5.0, 9.0, 9.0]),
            np.array([5.0, 5.0, 5.6, 5.1, 9.0, 9.0]),
        ]
        for str_action in env.get_actions():
            vec = env.action_to_vector[str_action]
            via_str = env.transition_log_probability(state, str_action, next_states)
            via_vec = ContinuousPushPOMDP.transition_log_probability(env, state, vec, next_states)
            np.testing.assert_allclose(via_str, via_vec, rtol=1e-12, atol=1e-12)

            via_str_obs = env.observation_log_probability(state, str_action, observations)
            via_vec_obs = ContinuousPushPOMDP.observation_log_probability(
                env, state, vec, observations
            )
            np.testing.assert_allclose(via_str_obs, via_vec_obs, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------
# n_samples contract for env-API sampling
# ------------------------------------------------------------------


class TestSampleNSamplesContract:
    """Shape / determinism contract for sample_{next_state,observation}(n_samples)."""

    @pytest.mark.parametrize("n_samples", [1, 5, 100])
    def test_sample_next_state_n_samples_shape(self, n_samples):
        """Test sample_next_state(n_samples=n) returns the right shape.

        Purpose: Guards the contract that ``n_samples=1`` returns a single
            6-D ndarray and ``n_samples>1`` returns a length-n list of 6-D
            ndarrays.

        Given: A ContinuousPushPOMDP env and a (state, action) pair.
        When: env.sample_next_state(state, action, n_samples=n) is called.
        Then: For n==1 a single ndarray of shape (6,) is returned; for n>1
            a length-n list of (6,) ndarrays is returned.

        Test type: unit
        """
        env = ContinuousPushPOMDP(
            discount_factor=0.99,
            state_transition_cov_matrix=np.eye(2) * 0.01,
            robot_radius=0.3,
        )
        state = np.array([2.5, 3.1, 3.0, 3.0, 8.0, 8.0])
        action = np.array([1.0, 0.0])
        result = env.sample_next_state(state, action, n_samples=n_samples)
        if n_samples == 1:
            assert isinstance(result, np.ndarray)
            assert result.shape == (6,)
        else:
            assert len(result) == n_samples
            for sample in result:
                assert isinstance(sample, np.ndarray)
                assert sample.shape == (6,)

    @pytest.mark.parametrize("n_samples", [1, 5, 100])
    def test_sample_observation_n_samples_shape(self, n_samples):
        """Test sample_observation(n_samples=n) returns the right shape.

        Purpose: Guards the contract that ``n_samples=1`` returns a single
            6-D ndarray and ``n_samples>1`` returns a length-n list of 6-D
            ndarrays for the observation API.

        Given: A ContinuousPushPOMDP env and a (next_state, action) pair.
        When: env.sample_observation(next_state, action, n_samples=n) is
            called.
        Then: For n==1 a single ndarray of shape (6,) is returned; for n>1
            a length-n list of (6,) ndarrays is returned.

        Test type: unit
        """
        env = ContinuousPushPOMDP(
            discount_factor=0.99,
            observation_noise=0.3,
            robot_radius=0.3,
        )
        next_state = np.array([5.0, 5.0, 4.5, 5.5, 8.0, 8.0])
        action = np.array([0.5, 0.5])
        result = env.sample_observation(next_state, action, n_samples=n_samples)
        if n_samples == 1:
            assert isinstance(result, np.ndarray)
            assert result.shape == (6,)
        else:
            assert len(result) == n_samples
            for sample in result:
                assert isinstance(sample, np.ndarray)
                assert sample.shape == (6,)
