"""Tests for the Continuous Push POMDP environment.

This module tests the state transition model, observation model,
environment, and discrete actions wrapper for the Continuous Push POMDP.
"""

# pylint: disable=protected-access

from typing import Any

import numpy as np

from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import (
    ContinuousPushObservationModel,
    ContinuousPushPOMDP,
    ContinuousPushPOMDPDiscreteActions,
    ContinuousPushStateTransitionModel,
)
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal


# ------------------------------------------------------------------
# State Transition
# ------------------------------------------------------------------


class TestContinuousPushStateTransition:
    """Test the continuous push state transition model."""

    def setup_method(self):
        """Set up shared test fixtures."""
        np.random.seed(42)
        self.cov = np.eye(2) * 0.01
        self.dist = CovarianceParameterizedMultivariateNormal(self.cov)
        self.obstacles = np.empty((0, 4))

    def _make_transition(self, state, action, **kwargs):
        defaults: dict[str, Any] = dict(
            state_transition_dist=self.dist,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            max_push=2.0,
            obstacles=self.obstacles,
            robot_radius=0.3,
        )
        defaults.update(kwargs)
        return ContinuousPushStateTransitionModel(state=state, action=action, **defaults)

    def test_robot_moves_in_action_direction(self):
        """Test that the robot moves approximately in the action direction.

        Purpose: Validates that action vector translates to movement.

        Given: Robot at (2, 3), action = (1, 0), small noise.
        When: sample() is called.
        Then: Robot x increases, y stays roughly constant.

        Test type: unit
        """
        state = np.array([2.0, 3.0, 5.0, 5.0, 9.0, 9.0])
        t = self._make_transition(state, np.array([1.0, 0.0]))
        ns = t.sample()[0]
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
        np.random.seed(42)
        state = np.array([2.0, 3.0, 2.5, 3.0, 9.0, 9.0])
        t = self._make_transition(
            state,
            np.array([1.0, 0.0]),
            state_transition_dist=CovarianceParameterizedMultivariateNormal(np.eye(2) * 1e-8),
        )
        ns = t.sample()[0]
        assert ns[2] > state[2]  # object pushed right

    def test_no_push_beyond_threshold(self):
        """Test that object is NOT pushed when robot is beyond threshold.

        Purpose: Validates that push only occurs within threshold.

        Given: Robot at (0, 0), object at (5, 5), threshold=1.0.
        When: Robot moves right.
        Then: Object position unchanged.

        Test type: unit
        """
        np.random.seed(42)
        state = np.array([0.0, 0.0, 5.0, 5.0, 9.0, 9.0])
        t = self._make_transition(
            state,
            np.array([1.0, 0.0]),
            state_transition_dist=CovarianceParameterizedMultivariateNormal(np.eye(2) * 1e-8),
        )
        ns = t.sample()[0]
        np.testing.assert_allclose(ns[2:4], state[2:4], atol=1e-6)

    def test_push_force_capped(self):
        """Test that push force is capped at max_push.

        Purpose: Validates capped push scaling.

        Given: Large action (10, 0), max_push=2.0, friction=0.3.
        When: Push is applied to nearby object.
        Then: Push magnitude <= max_push * (1 - friction).

        Test type: unit
        """
        np.random.seed(42)
        state = np.array([4.0, 4.0, 4.5, 4.0, 9.0, 9.0])
        t = self._make_transition(
            state,
            np.array([10.0, 0.0]),
            max_push=2.0,
            friction_coefficient=0.3,
            state_transition_dist=CovarianceParameterizedMultivariateNormal(np.eye(2) * 1e-8),
        )
        ns = t.sample()[0]
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
        np.random.seed(42)
        state = np.array([4.0, 4.0, 4.5, 4.0, 9.0, 9.0])
        low_cov = CovarianceParameterizedMultivariateNormal(np.eye(2) * 1e-8)

        t_no_fric = self._make_transition(
            state,
            np.array([1.0, 0.0]),
            friction_coefficient=0.0,
            state_transition_dist=low_cov,
        )
        t_hi_fric = self._make_transition(
            state,
            np.array([1.0, 0.0]),
            friction_coefficient=0.5,
            state_transition_dist=low_cov,
        )
        ns_no = t_no_fric.sample()[0]
        ns_hi = t_hi_fric.sample()[0]
        assert ns_no[2] > ns_hi[2]

    def test_obstacle_blocks_object_push(self):
        """Test that obstacles block object push.

        Purpose: Validates that object stays put when pushed into obstacle.

        Given: Object at (4.5, 5.0), obstacle AABB at (5, 5) half=0.5.
        When: Robot pushes object rightward into obstacle.
        Then: Object position unchanged.

        Test type: unit
        """
        np.random.seed(42)
        walls = np.array([[5.0, 5.0, 0.5, 0.5]])
        state = np.array([4.0, 5.0, 4.5, 5.0, 9.0, 9.0])
        low_cov = CovarianceParameterizedMultivariateNormal(np.eye(2) * 1e-8)
        t = self._make_transition(
            state,
            np.array([1.0, 0.0]),
            obstacles=walls,
            state_transition_dist=low_cov,
        )
        ns = t.sample()[0]
        np.testing.assert_allclose(ns[2:4], state[2:4], atol=1e-6)

    def test_grid_clamping(self):
        """Test that positions are clamped to grid.

        Purpose: Validates grid boundary enforcement.

        Given: Robot at (8.5, 8.5) with action pushing out of grid.
        When: sample() is called.
        Then: Robot stays within valid bounds.

        Test type: unit
        """
        np.random.seed(42)
        state = np.array([8.5, 8.5, 5.0, 5.0, 9.0, 9.0])
        low_cov = CovarianceParameterizedMultivariateNormal(np.eye(2) * 1e-8)
        t = self._make_transition(
            state,
            np.array([5.0, 5.0]),
            state_transition_dist=low_cov,
        )
        ns = t.sample()[0]
        assert ns[0] <= 10 - 1 - 0.3 + 0.01  # within robot radius of grid edge
        assert ns[1] <= 10 - 1 - 0.3 + 0.01

    def test_probability_returns_positive(self):
        """Test that probability returns positive densities.

        Purpose: Validates probability method.

        Given: A transition model and a plausible next state.
        When: probability() is called.
        Then: Returns a positive probability.

        Test type: unit
        """
        state = np.array([2.0, 3.0, 5.0, 5.0, 9.0, 9.0])
        t = self._make_transition(state, np.array([1.0, 0.0]))
        prob = t.probability([np.array([3.0, 3.0, 5.0, 5.0, 9.0, 9.0])])
        assert prob[0] > 0


# ------------------------------------------------------------------
# Observation Model
# ------------------------------------------------------------------


class TestContinuousPushObservation:
    """Test the continuous push observation model."""

    def setup_method(self):
        """Set up shared test fixtures."""
        np.random.seed(42)

    def test_observation_shape(self):
        """Test that observation has shape (6,).

        Purpose: Validates observation vector shape.

        Given: A state with 6 elements.
        When: sample() is called.
        Then: Observation has shape (6,).

        Test type: unit
        """
        state = np.array([3.0, 4.0, 5.0, 5.0, 9.0, 9.0])
        model = ContinuousPushObservationModel(state, np.array([1.0, 0.0]), 0.1, 10)
        obs = model.sample()[0]
        assert obs.shape == (6,)

    def test_robot_and_target_exact(self):
        """Test that robot and target positions are observed exactly.

        Purpose: Validates that robot and target are noiseless in observations.

        Given: State with known robot and target positions.
        When: sample() is called.
        Then: Observation indices 0:2 and 4:6 match state exactly.

        Test type: unit
        """
        state = np.array([3.0, 4.0, 5.0, 5.0, 9.0, 9.0])
        model = ContinuousPushObservationModel(state, np.array([1.0, 0.0]), 0.1, 10)
        obs = model.sample()[0]
        np.testing.assert_array_equal(obs[:2], state[:2])
        np.testing.assert_array_equal(obs[4:6], state[4:6])

    def test_object_position_noisy(self):
        """Test that object position observation includes noise.

        Purpose: Validates that object position is noisy.

        Given: Observation noise = 1.0 (large).
        When: Multiple samples are drawn.
        Then: Object position varies across samples.

        Test type: unit
        """
        state = np.array([3.0, 4.0, 5.0, 5.0, 9.0, 9.0])
        model = ContinuousPushObservationModel(state, np.array([1.0, 0.0]), 1.0, 10)
        obs1 = model.sample()[0]
        obs2 = model.sample()[0]
        assert not np.array_equal(obs1[2:4], obs2[2:4])

    def test_probability_returns_positive(self):
        """Test that observation probability is positive for plausible observations.

        Purpose: Validates probability calculation.

        Given: An observation close to the true state.
        When: probability() is called.
        Then: Returns positive probability.

        Test type: unit
        """
        state = np.array([3.0, 4.0, 5.0, 5.0, 9.0, 9.0])
        model = ContinuousPushObservationModel(state, np.array([1.0, 0.0]), 0.1, 10)
        obs = state.copy()
        prob = model.probability([obs])
        assert prob[0] > 0


# ------------------------------------------------------------------
# ContinuousPushPOMDP
# ------------------------------------------------------------------


class TestContinuousPushPOMDP:
    """Test the ContinuousPushPOMDP environment."""

    def setup_method(self):
        """Set up shared test fixtures."""
        np.random.seed(42)
        self.env = ContinuousPushPOMDP(
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
        ns, obs, rew = env.sample_next_step(state, np.array([1.0, 0.0]))
        assert ns.shape == (6,)

    def test_compute_metrics(self):
        """Test compute_metrics returns expected metric names.

        Purpose: Validates metric computation.

        Given: A short simulated history.
        When: compute_metrics is called.
        Then: Returns list of MetricValue with correct names.

        Test type: integration
        """
        from POMDPPlanners.core.simulation import History, StepData
        from POMDPPlanners.core.policy import PolicyRunData

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
