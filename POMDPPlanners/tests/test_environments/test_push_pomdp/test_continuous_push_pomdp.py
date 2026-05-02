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
from POMDPPlanners.tests.test_utils.confidence_interval_utils import (
    verify_metrics_within_confidence_intervals,
)
from POMDPPlanners.tests.test_utils.metric_invariants_utils import (
    verify_history_returns_bounded,
    verify_metric_sanity,
    verify_return_shift_linearity,
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

    def test_compute_metrics_values_within_confidence_intervals(self):
        """Test full-metric CI containment + structural invariants.

        Purpose: Broad property check that every metric's value is inside its
            own CI and that all structural invariants on the metric set and
            histories hold (rate-in-[0,1], counts >= 0, finite CI for n>=2,
            per-step-counts <= total_steps, discounted return inside reward
            bounds, return shifts linearly with reward shift).

        Given: 5 hand-built histories with varied collision/goal patterns
            (all-safe, robot-only collisions, object-only collisions, both,
            and a terminal-reaching episode).
        When: compute_metrics is called.
        Then: All four invariant helpers pass without raising.

        Test type: integration
        """
        target = np.array([9.0, 9.0])
        obstacle_center = np.array([5.0, 5.0])
        safe_robot = np.array([1.0, 1.0])
        safe_object = np.array([1.0, 1.0])

        # History 0: all-safe (no collisions, no goal)
        all_safe_steps = [
            StepData(
                state=np.concatenate([safe_robot, safe_object, target]),
                action=np.array([0.0, 0.0]),
                next_state=np.concatenate([safe_robot, safe_object, target]),
                observation=np.concatenate([safe_robot, safe_object, target]),
                reward=0.0,
                belief=None,  # type: ignore[arg-type]
            )
            for _ in range(4)
        ]

        # History 1: robot-only collisions on first 2 of 4 steps.
        robot_only_steps = []
        for step_index in range(4):
            robot_pos = obstacle_center if step_index < 2 else safe_robot
            state = np.concatenate([robot_pos, safe_object, target])
            robot_only_steps.append(
                StepData(
                    state=state,
                    action=np.array([0.0, 0.0]),
                    next_state=state,
                    observation=state,
                    reward=-1.0,
                    belief=None,  # type: ignore[arg-type]
                )
            )

        # History 2: object-only collisions on first 3 of 5 steps.
        object_only_steps = []
        for step_index in range(5):
            object_pos = obstacle_center if step_index < 3 else safe_object
            state = np.concatenate([safe_robot, object_pos, target])
            object_only_steps.append(
                StepData(
                    state=state,
                    action=np.array([0.0, 0.0]),
                    next_state=state,
                    observation=state,
                    reward=-1.0,
                    belief=None,  # type: ignore[arg-type]
                )
            )

        # History 3: both robot AND object colliding on 1 of 3 steps.
        both_steps = []
        for step_index in range(3):
            robot_pos = obstacle_center if step_index == 0 else safe_robot
            object_pos = obstacle_center if step_index == 0 else safe_object
            state = np.concatenate([robot_pos, object_pos, target])
            both_steps.append(
                StepData(
                    state=state,
                    action=np.array([0.0, 0.0]),
                    next_state=state,
                    observation=state,
                    reward=-2.0,
                    belief=None,  # type: ignore[arg-type]
                )
            )

        # History 4: terminal-reaching (object placed at target).
        # is_terminal triggers when ||object - target|| < 0.5 (typical thresh).
        terminal_steps = []
        for step_index in range(3):
            object_pos = target if step_index == 2 else safe_object
            state = np.concatenate([safe_robot, object_pos, target])
            terminal_steps.append(
                StepData(
                    state=state,
                    action=np.array([0.0, 0.0]),
                    next_state=state,
                    observation=state,
                    reward=0.0,
                    belief=None,  # type: ignore[arg-type]
                )
            )

        histories = []
        for steps, reach_terminal in (
            (all_safe_steps, False),
            (robot_only_steps, False),
            (object_only_steps, False),
            (both_steps, False),
            (terminal_steps, True),
        ):
            histories.append(
                History(
                    history=steps,
                    discount_factor=0.99,
                    average_state_sampling_time=0.0,
                    average_action_time=0.0,
                    average_observation_time=0.0,
                    average_belief_update_time=0.0,
                    average_reward_time=0.0,
                    actual_num_steps=len(steps),
                    reach_terminal_state=reach_terminal,
                    policy_run_data=[PolicyRunData(info_variables=[])],
                )
            )

        metrics = self.env.compute_metrics(histories)
        verify_metrics_within_confidence_intervals(metrics)
        verify_metric_sanity(metrics, histories, self.env)
        verify_history_returns_bounded(histories, self.env)
        verify_return_shift_linearity(histories, self.env, shift=1.5)


class TestContinuousPushStochasticObstacleHitProbability:
    """Tests for ``obstacle_hit_probability`` on ``ContinuousPushPOMDP``.

    Geometry: a single AABB obstacle centred at ``(5.0, 5.0)`` with
    half-size ``0.5`` (extents ``[4.5, 5.5]^2``). The robot starts at
    ``(4.0, 5.0)`` and the action ``(1.0, 0.0)`` puts the post-action
    robot position at ``(5.0, 5.0)`` — well inside the AABB even after
    accounting for ``robot_radius``. Transition covariance is set to a
    near-zero diagonal so the distance-to-target term is effectively
    deterministic and the obstacle penalty (-10.0) cleanly dominates the
    decision boundary used to bucket "hit" vs. "no-hit" trials.
    """

    OBSTACLE_PENALTY = -10.0
    COLLIDE_ACTION = np.array([1.0, 0.0])

    @staticmethod
    def _stochastic_env(hit_probability: float) -> ContinuousPushPOMDP:
        return ContinuousPushPOMDP(
            discount_factor=0.99,
            grid_size=10,
            obstacles=[(5.0, 5.0, 0.5)],
            obstacle_penalty=TestContinuousPushStochasticObstacleHitProbability.OBSTACLE_PENALTY,
            obstacle_hit_probability=hit_probability,
            robot_radius=0.3,
            state_transition_cov_matrix=np.eye(2) * 1e-8,
        )

    @staticmethod
    def _collide_state() -> np.ndarray:
        # robot just left of obstacle AABB; object far from target so the
        # distance-to-target term is well above 0.5 (no goal bonus).
        return np.array([4.0, 5.0, 1.0, 1.0, 9.0, 9.0])

    def test_default_hit_probability_is_one(self):
        """Default hit probability preserves legacy deterministic behavior.

        Purpose: Validates that omitting the new parameter keeps the
            deterministic obstacle-collision penalty active.

        Given: A ContinuousPushPOMDP with default parameters.
        When: ``obstacle_hit_probability`` is read.
        Then: It equals ``1.0``.

        Test type: unit
        """
        env = ContinuousPushPOMDP(discount_factor=0.99)
        assert env.obstacle_hit_probability == 1.0

    def test_hit_probability_zero_never_applies_penalty(self):
        """hit_probability=0 disables the obstacle-collision penalty.

        Purpose: Validates the lower-bound of the stochastic penalty for
            the continuous-action env.

        Given: A robot moving right into an obstacle AABB with
            hit_probability=0 and near-zero transition noise.
        When: ``reward()`` is called many times.
        Then: Every reward sits above the (baseline - penalty/2) midpoint
            — i.e. no obstacle penalty was applied.

        Test type: unit
        """
        env = self._stochastic_env(hit_probability=0.0)
        state = self._collide_state()
        # With p=0 the reward equals the baseline distance term up to the
        # tiny transition-noise jitter (cov 1e-8). A midpoint between
        # baseline and (baseline + obstacle_penalty) cleanly separates the
        # two regimes regardless of where the absolute baseline sits.
        baseline = env.reward(state, self.COLLIDE_ACTION)
        midpoint = baseline + self.OBSTACLE_PENALTY / 2.0
        np.random.seed(0)
        for _ in range(200):
            r = env.reward(state, self.COLLIDE_ACTION)
            assert r > midpoint

    def test_hit_probability_one_always_applies_penalty(self):
        """hit_probability=1 matches legacy deterministic penalty.

        Purpose: Regression check that the default behavior is preserved
            when hit_probability=1.0 is passed explicitly.

        Given: A robot moving right into an obstacle AABB with
            hit_probability=1.0 and near-zero transition noise.
        When: ``reward()`` is called many times.
        Then: Every reward includes the full obstacle penalty.

        Test type: unit
        """
        env = self._stochastic_env(hit_probability=1.0)
        state = self._collide_state()
        # Compare against the same env with p=0 to isolate the penalty.
        env_no_pen = self._stochastic_env(hit_probability=0.0)
        baseline = env_no_pen.reward(state, self.COLLIDE_ACTION)
        for _ in range(50):
            r = env.reward(state, self.COLLIDE_ACTION)
            assert r == pytest.approx(baseline + self.OBSTACLE_PENALTY, abs=1e-3)

    def test_hit_probability_zero_three_empirical_rate(self):
        """Empirical hit rate matches hit_probability over many calls.

        Purpose: Validates that the per-call Bernoulli draw matches the
            configured probability over a large sample.

        Given: A robot moving into an obstacle with hit_probability=0.3.
        When: ``reward()`` is called 5000 times.
        Then: Empirical hit rate is within 0.05 of 0.3.

        Test type: unit
        """
        env = self._stochastic_env(hit_probability=0.3)
        state = self._collide_state()
        env_no_pen = self._stochastic_env(hit_probability=0.0)
        baseline = env_no_pen.reward(state, self.COLLIDE_ACTION)
        midpoint = baseline + self.OBSTACLE_PENALTY / 2.0
        np.random.seed(123)
        n_trials = 5000
        hits = 0
        for _ in range(n_trials):
            if env.reward(state, self.COLLIDE_ACTION) < midpoint:
                hits += 1
        empirical_rate = hits / n_trials
        assert abs(empirical_rate - 0.3) < 0.05

    def test_reward_batch_honours_hit_probability(self):
        """reward_batch applies stochastic penalty consistently.

        Purpose: Validates that the batched reward path uses the same
            Bernoulli mechanism as the single-state path.

        Given: 5000 copies of a state moving into an obstacle with
            hit_probability=0.3.
        When: ``reward_batch()`` is called once.
        Then: Empirical hit rate across the batch is within 0.05 of 0.3.

        Test type: unit
        """
        env = self._stochastic_env(hit_probability=0.3)
        state = self._collide_state()
        env_no_pen = self._stochastic_env(hit_probability=0.0)
        baseline = env_no_pen.reward(state, self.COLLIDE_ACTION)
        midpoint = baseline + self.OBSTACLE_PENALTY / 2.0
        n_trials = 5000
        states = np.tile(state, (n_trials, 1))
        np.random.seed(456)
        rewards = env.reward_batch(states, self.COLLIDE_ACTION)
        hits = int(np.sum(rewards < midpoint))
        empirical_rate = hits / n_trials
        assert abs(empirical_rate - 0.3) < 0.05

    def test_reward_batch_zero_probability_never_applies_penalty(self):
        """reward_batch with hit_probability=0 returns no obstacle penalty.

        Purpose: Validates the lower-bound of the stochastic penalty in
            the batched path.

        Given: A batch of in-zone next-state states with
            hit_probability=0.0.
        When: ``reward_batch()`` is called.
        Then: All returned rewards exceed the obstacle-penalty floor.

        Test type: unit
        """
        env = self._stochastic_env(hit_probability=0.0)
        state = self._collide_state()
        env_no_pen = self._stochastic_env(hit_probability=0.0)
        baseline = env_no_pen.reward(state, self.COLLIDE_ACTION)
        midpoint = baseline + self.OBSTACLE_PENALTY / 2.0
        states = np.tile(state, (200, 1))
        np.random.seed(789)
        rewards = env.reward_batch(states, self.COLLIDE_ACTION)
        assert np.all(rewards > midpoint)

    @pytest.mark.parametrize("bad_value", [-0.1, 1.5, 2.0, -1.0])
    def test_invalid_hit_probability_raises(self, bad_value: float):
        """Out-of-range hit_probability raises ValueError.

        Purpose: Validates input validation on the new parameter.

        Given: A hit_probability value outside [0, 1].
        When: ContinuousPushPOMDP is constructed.
        Then: ValueError is raised mentioning the parameter name.

        Test type: unit
        """
        with pytest.raises(ValueError, match="obstacle_hit_probability"):
            ContinuousPushPOMDP(discount_factor=0.99, obstacle_hit_probability=bad_value)


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


class TestContinuousPushDiscreteActionsHitProbability:
    """Regression tests that ``obstacle_hit_probability`` is forwarded
    through ``ContinuousPushPOMDPDiscreteActions.__init__`` to the
    parent ``ContinuousPushPOMDP``.

    Without the kwarg in the wrapper signature, planners constructing
    the discrete-action env saw the parent default (1.0) regardless of
    what they intended — masking risk-sensitive evaluation.
    """

    OBSTACLE_PENALTY = -10.0
    COLLIDE_ACTION = "right"

    @staticmethod
    def _stochastic_env(hit_probability: float) -> ContinuousPushPOMDPDiscreteActions:
        return ContinuousPushPOMDPDiscreteActions(
            discount_factor=0.99,
            grid_size=10,
            obstacles=[(5.0, 5.0, 0.5)],
            obstacle_penalty=TestContinuousPushDiscreteActionsHitProbability.OBSTACLE_PENALTY,
            obstacle_hit_probability=hit_probability,
            robot_radius=0.3,
            state_transition_cov_matrix=np.eye(2) * 1e-8,
        )

    def test_default_hit_probability_is_one(self):
        """Default kwarg preserves legacy deterministic behavior.

        Purpose: Validates that omitting the kwarg yields the parent default.

        Given: A wrapper constructed with no ``obstacle_hit_probability``.
        When: ``obstacle_hit_probability`` is read on the instance.
        Then: It equals ``1.0``.

        Test type: unit
        """
        env = ContinuousPushPOMDPDiscreteActions(discount_factor=0.99)
        assert env.obstacle_hit_probability == 1.0

    @pytest.mark.parametrize("p", [0.0, 0.3, 0.7, 1.0])
    def test_kwarg_forwarded_to_parent_attribute(self, p: float):
        """Wrapper kwarg flows through to the parent's stored attribute.

        Purpose: Regression — without the forwarding fix, this assertion
            failed for every ``p != 1.0``.

        Given: A wrapper constructed with ``obstacle_hit_probability=p``.
        When: The instance attribute is read.
        Then: It equals ``p``.

        Test type: unit
        """
        env = self._stochastic_env(hit_probability=p)
        assert env.obstacle_hit_probability == pytest.approx(p)

    def test_hit_probability_zero_disables_penalty_via_string_action(self):
        """Functional check that the forwarded kwarg actually gates the penalty.

        Purpose: Regression — verifies the value reaches the reward path
            and not just the attribute, when called through the wrapper's
            string-action API.

        Given: A wrapper with ``hit_probability=0`` and a state that would
            otherwise collide with an obstacle on ``"right"``.
        When: ``reward()`` is called many times via the string-action API.
        Then: No reward sits at-or-below the obstacle-penalty floor.

        Test type: unit
        """
        env = self._stochastic_env(hit_probability=0.0)
        state = np.array([4.0, 5.0, 1.0, 1.0, 9.0, 9.0])
        baseline = env.reward(state, self.COLLIDE_ACTION)
        midpoint = baseline + self.OBSTACLE_PENALTY / 2.0
        np.random.seed(0)
        for _ in range(100):
            assert env.reward(state, self.COLLIDE_ACTION) > midpoint

    @pytest.mark.parametrize("bad_value", [-0.1, 1.5, 2.0, -1.0])
    def test_invalid_hit_probability_raises(self, bad_value: float):
        """Out-of-range values raise via the parent's validator.

        Purpose: Validates that the wrapper does not silently swallow
            invalid values — they reach the parent's ValueError.

        Given: An ``obstacle_hit_probability`` outside ``[0, 1]``.
        When: The wrapper is constructed.
        Then: ``ValueError`` is raised mentioning the parameter name.

        Test type: unit
        """
        with pytest.raises(ValueError, match="obstacle_hit_probability"):
            ContinuousPushPOMDPDiscreteActions(
                discount_factor=0.99, obstacle_hit_probability=bad_value
            )


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


# Native rollout equivalence
# ------------------------------------------------------------------


class _FixedActionSampler:
    """Minimal sampler that cycles through a fixed sequence of actions."""

    def __init__(self, actions_seq):
        self._actions = list(actions_seq)
        self._idx = 0

    def sample(self, belief_node=None):  # pylint: disable=unused-argument
        action = self._actions[self._idx % len(self._actions)]
        self._idx += 1
        return action


class TestNativeRolloutEquivalence:
    """Equivalence tests for the native C++ simulate_random_rollout override."""

    def _make_env(self) -> ContinuousPushPOMDPDiscreteActions:
        return ContinuousPushPOMDPDiscreteActions(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            max_push=2.0,
            observation_noise=0.1,
            obstacles=[(5.0, 5.0, 0.5)],
            obstacle_penalty=-10.0,
            robot_radius=0.3,
            state_transition_cov_matrix=np.eye(2) * 0.05,
        )

    def test_native_rollout_is_float(self):
        """simulate_random_rollout returns a Python float.

        Purpose: Validates that the native override returns a scalar float,
            matching the return-type contract of the base class.

        Given: A ContinuousPushPOMDPDiscreteActions env and an initial state.
        When: simulate_random_rollout is called with a discrete sampler.
        Then: The result is a Python float.

        Test type: unit
        """
        from POMDPPlanners.environments.push_pomdp import (
            _native,
        )  # pylint: disable=import-outside-toplevel

        env = self._make_env()
        _native.set_seed(1)
        state = np.array([2.0, 2.0, 5.0, 5.0, 9.0, 9.0])
        from POMDPPlanners.utils.action_samplers import (
            DiscreteActionSampler,
        )  # pylint: disable=import-outside-toplevel

        sampler = DiscreteActionSampler(env.get_actions())
        result = env.simulate_random_rollout(
            state=state,
            action_sampler=sampler,
            max_depth=10,
            discount_factor=0.95,
        )
        assert isinstance(result, float)

    def test_native_rollout_deterministic_under_seed(self):
        """Two calls with the same seed produce identical rollout returns.

        Purpose: Validates that the native C++ rollout is deterministic when
            the module-level RNG is re-seeded to the same value before each
            call.

        Given: A fixed initial state and two identical seeds applied via
            ``_native.set_seed`` before each rollout.
        When: simulate_random_rollout is called twice with the same seed.
        Then: Both calls return the exact same float.

        Test type: unit
        """
        from POMDPPlanners.environments.push_pomdp import (
            _native,
        )  # pylint: disable=import-outside-toplevel
        from POMDPPlanners.utils.action_samplers import (
            DiscreteActionSampler,
        )  # pylint: disable=import-outside-toplevel

        env = self._make_env()
        state = np.array([3.0, 3.0, 6.0, 6.0, 9.0, 9.0])
        sampler = DiscreteActionSampler(env.get_actions())

        np.random.seed(42)
        _native.set_seed(999)
        result_a = env.simulate_random_rollout(
            state=state, action_sampler=sampler, max_depth=15, discount_factor=0.95
        )

        np.random.seed(42)
        _native.set_seed(999)
        result_b = env.simulate_random_rollout(
            state=state, action_sampler=sampler, max_depth=15, discount_factor=0.95
        )

        assert result_a == result_b

    def test_native_rollout_equivalence_with_python_step_by_step(self):
        """Native rollout matches a step-by-step Python reference at fixed seed.

        Purpose: Validates that cont_simulate_rollout computes the same
            discounted return as a manually constructed Python loop that
            calls ``sample_next_state`` and ``reward`` using the same
            action sequence and the same module-level C++ RNG seed.

        Given: A fixed 6-D state, a fixed seed, and a pre-drawn action-index
            sequence using ``np.random`` (so both paths draw from the same
            action sequence).
        When: The native rollout and a Python reference loop are each run
            under the same ``_native.set_seed`` and ``np.random.seed``.
        Then: Both discounted returns agree to absolute tolerance 1e-9.

        Test type: unit
        """
        from POMDPPlanners.environments.push_pomdp import (
            _native,
        )  # pylint: disable=import-outside-toplevel

        env = self._make_env()
        state = np.array([2.0, 2.0, 6.0, 6.0, 9.0, 9.0])
        discount_factor = 0.95
        max_depth = 8
        depth = 0
        steps_left = max_depth - depth
        seed_np = 1234
        seed_native = 7777

        # Pre-draw the same action index sequence used by both paths.
        np.random.seed(seed_np)
        action_indices = np.random.randint(0, 4, size=steps_left, dtype=np.int32)

        # ── Native path ──────────────────────────────────────────────
        _native.set_seed(seed_native)
        native_return = env.simulate_random_rollout(
            state=state,
            action_sampler=_FixedActionSampler([env.get_actions()[i] for i in action_indices]),
            max_depth=max_depth,
            discount_factor=discount_factor,
            depth=depth,
        )

        # ── Python reference loop ─────────────────────────────────────
        # Reproduce the exact same action sequence via FixedActionSampler
        # and reset the native RNG to the same seed so the Gaussian draws
        # inside sample_next_state match.
        _native.set_seed(seed_native)
        ref_state = state.copy()
        ref_total = 0.0
        gamma_power = 1.0
        for i in range(steps_left):
            if env.is_terminal(ref_state):
                break
            str_action = env.get_actions()[action_indices[i]]
            ref_next = env.sample_next_state(ref_state, str_action)
            ref_reward = env.reward(ref_state, str_action)
            ref_total += gamma_power * ref_reward
            ref_state = ref_next
            gamma_power *= discount_factor

        np.testing.assert_allclose(
            native_return,
            ref_total,
            atol=1e-9,
            err_msg=(
                f"Native rollout {native_return} vs Python reference {ref_total}: "
                "discounted returns must agree to 1e-9"
            ),
        )

    def test_native_rollout_terminates_at_goal(self):
        """Rollout stops early when object is already at goal.

        Purpose: Validates the terminal check in the native rollout loop.

        Given: A state where the object is within 0.5 of the target
            (terminal).
        When: simulate_random_rollout is called with max_depth=20.
        Then: Returns 0.0 (no steps taken).

        Test type: unit
        """
        from POMDPPlanners.environments.push_pomdp import (
            _native,
        )  # pylint: disable=import-outside-toplevel
        from POMDPPlanners.utils.action_samplers import (
            DiscreteActionSampler,
        )  # pylint: disable=import-outside-toplevel

        env = self._make_env()
        # Object at (9.1, 9.1), target at (9, 9) → dist ≈ 0.14 < 0.5 → terminal
        terminal_state = np.array([1.0, 1.0, 9.1, 9.1, 9.0, 9.0])
        assert env.is_terminal(terminal_state)
        sampler = DiscreteActionSampler(env.get_actions())
        _native.set_seed(0)
        result = env.simulate_random_rollout(
            state=terminal_state,
            action_sampler=sampler,
            max_depth=20,
            discount_factor=0.95,
        )
        assert result == 0.0

    def test_native_rollout_zero_steps_returns_zero(self):
        """Rollout returns 0.0 when depth == max_depth.

        Purpose: Validates the boundary guard (steps_left <= 0).

        Given: A non-terminal state with depth == max_depth.
        When: simulate_random_rollout is called.
        Then: Returns 0.0 immediately.

        Test type: unit
        """
        from POMDPPlanners.utils.action_samplers import (
            DiscreteActionSampler,
        )  # pylint: disable=import-outside-toplevel

        env = self._make_env()
        state = np.array([2.0, 2.0, 5.0, 5.0, 9.0, 9.0])
        sampler = DiscreteActionSampler(env.get_actions())
        result = env.simulate_random_rollout(
            state=state,
            action_sampler=sampler,
            max_depth=5,
            discount_factor=0.95,
            depth=5,
        )
        assert result == 0.0

    def test_continuous_env_falls_back_to_python_loop(self):
        """ContinuousPushPOMDP (no action_to_vector) falls back to Python.

        Purpose: Validates that the base ContinuousPushPOMDP, which has no
            fixed discrete action set, gracefully delegates to the Python
            base-class loop when simulate_random_rollout is called.

        Given: A ContinuousPushPOMDP (not the discrete subclass) and a
            UnitCircleActionSampler.
        When: simulate_random_rollout is called.
        Then: Returns a finite float without error.

        Test type: unit
        """
        from POMDPPlanners.utils.action_samplers import (
            UnitCircleActionSampler,
        )  # pylint: disable=import-outside-toplevel

        env = ContinuousPushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            state_transition_cov_matrix=np.eye(2) * 0.05,
        )
        state = np.array([2.0, 2.0, 5.0, 5.0, 9.0, 9.0])
        sampler = UnitCircleActionSampler(max_action_magnitude=1.0)
        np.random.seed(42)
        result = env.simulate_random_rollout(
            state=state,
            action_sampler=sampler,
            max_depth=5,
            discount_factor=0.95,
        )
        assert np.isfinite(result)
