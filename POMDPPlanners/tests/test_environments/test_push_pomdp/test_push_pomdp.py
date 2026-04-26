"""Tests for Push POMDP environment.

This module tests the Push POMDP environment, focusing on:
- Basic environment functionality
- Obstacle collision penalties
- Reward verification between obstacle and non-obstacle states
"""

# pylint: disable=protected-access  # Tests need to access protected members

import random

import numpy as np
import pytest

from POMDPPlanners.environments.push_pomdp import PushPOMDP, _native as push_native

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


class TestPushPOMDP:
    """Test cases for Push POMDP environment."""

    def setup_method(self):
        """Set up test environment before each test method."""
        # Create Push POMDP environment with obstacles
        self.env = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            observation_noise=0.1,
            obstacles=[(3.0, 3.0), (7.0, 7.0)],  # Two obstacles
            obstacle_radius=0.5,
            obstacle_penalty=-10.0,
        )

        # Test positions
        self.obstacle_pos = np.array([3.0, 3.0])  # In obstacle
        self.safe_pos = np.array([1.0, 1.0])  # Safe area
        self.target_pos = np.array([9.0, 9.0])  # Target position
        self.object_pos = np.array([2.0, 2.0])  # Object position

    def test_obstacle_collision_detection(self):
        """Test that obstacle collision detection works correctly."""
        # Position in obstacle
        assert self.env._is_colliding_with_obstacle(np.array([2.0, 3.0]), "right") is True
        assert (
            self.env._is_colliding_with_obstacle(np.array([4.2, 3.2]), "left") is True
        )  # Within radius
        assert (
            self.env._is_colliding_with_obstacle(np.array([3.3, 4.1]), "down") is True
        )  # Within radius

        # Position outside obstacle
        assert self.env._is_colliding_with_obstacle(np.array([3.0, 3.0]), "up") is False
        assert self.env._is_colliding_with_obstacle(np.array([3.0, 3.0]), "down") is False
        assert (
            self.env._is_colliding_with_obstacle(np.array([3.6, 3.6]), "left") is False
        )  # Outside radius

    def test_reward_difference_obstacle_vs_safe(self):
        """Test that rewards for actions leading to obstacle collision are lower than safe actions."""
        # Create state where robot action will lead to obstacle collision
        # Robot at (2.5, 3.0), obstacle at (3.0, 3.0) with radius 0.5
        # Action "right" will move to (3.5, 3.0), distance = 0.5, exactly at boundary (collision)
        will_collide_state = np.concatenate(
            [
                np.array([2.5, 3.0]),  # Robot position before action
                self.object_pos,  # Object position
                self.target_pos,  # Target position
            ]
        )

        # Create state where robot action will NOT lead to obstacle collision
        safe_state = np.concatenate(
            [
                self.safe_pos,  # Robot in safe area
                self.object_pos,  # Object position
                self.target_pos,  # Target position
            ]
        )

        # Test movement action ("right")
        # For will_collide_state: (2.5, 3.0) + "right" = (3.5, 3.0), distance from (3.0, 3.0) = 0.5 <= 0.5 (collision!)
        # For safe_state: (1.0, 1.0) + "right" = (2.0, 1.0), no collision
        will_collide_reward = self.env.reward(will_collide_state, action="right")
        safe_reward = self.env.reward(safe_state, action="right")

        # Action leading to collision should get obstacle penalty (-10.0) in addition to distance reward
        # Safe action should only get distance reward

        # The reward for collision action should be lower than safe action
        assert (
            will_collide_reward < safe_reward
        ), f"Collision action reward ({will_collide_reward}) should be < safe action reward ({safe_reward})"

        # Both should be negative (due to distance to target)
        assert will_collide_reward < 0
        assert safe_reward < 0

    def test_robot_obstacle_collision_penalty(self):
        """Test that actions leading to robot collision with obstacles apply penalty."""
        # Create state where action will lead to collision
        # Robot at (2.0, 3.0), obstacle at (3.0, 3.0) with radius 0.5
        # Action "right" will move to (3.0, 3.0), distance = 0.0 (collision!)
        will_collide_state = np.concatenate(
            [
                np.array([2.0, 3.0]),  # Robot position before action
                self.object_pos,  # Object position
                self.target_pos,  # Target position
            ]
        )

        # Create state where action will NOT lead to collision
        # Robot at (3.6, 3.6), action "up" moves to (3.6, 4.6)
        # Distance from (3.0, 3.0) = sqrt((3.6-3.0)^2 + (4.6-3.0)^2) = sqrt(0.36 + 2.56) ≈ 1.71 > 0.5 (no collision)
        no_collision_state = np.concatenate(
            [
                np.array([3.6, 3.6]),  # Robot position before action
                self.object_pos,  # Object position
                self.target_pos,  # Target position
            ]
        )

        # Test actions that will/won't lead to collision
        will_collide_reward = self.env.reward(will_collide_state, action="right")
        no_collision_reward = self.env.reward(no_collision_state, action="up")

        # Action leading to collision should get lower reward due to penalty
        assert (
            will_collide_reward < no_collision_reward
        ), f"Collision action reward ({will_collide_reward}) should be < no collision action reward ({no_collision_reward})"

        # Calculate expected penalties (both states have same object position, so same distance to target)
        distance_to_target = np.linalg.norm(self.object_pos - self.target_pos)

        # Collision action should have distance component + penalty
        expected_collision_reward = -distance_to_target + self.env.obstacle_penalty
        # No collision action should only have distance component
        expected_no_collision_reward = -distance_to_target

        # Verify the penalty is applied correctly
        assert (
            abs(will_collide_reward - expected_collision_reward) < 1e-6
        ), f"Expected collision reward {expected_collision_reward}, got {will_collide_reward}"
        assert (
            abs(no_collision_reward - expected_no_collision_reward) < 1e-6
        ), f"Expected no collision reward {expected_no_collision_reward}, got {no_collision_reward}"

    def test_object_obstacle_collision_penalty(self):
        """Test that object collision with obstacles does not apply penalty (only robot collision does)."""
        # Create state with object in obstacle (robot is safe)
        object_obstacle_state = np.concatenate(
            [
                self.safe_pos,  # Robot in safe area
                self.obstacle_pos,  # Object in obstacle
                self.target_pos,  # Target position
            ]
        )

        # Create state with object just outside obstacle
        object_near_obstacle_state = np.concatenate(
            [
                self.safe_pos,  # Robot in safe area
                np.array([3.6, 3.6]),  # Object just outside obstacle radius
                self.target_pos,  # Target position
            ]
        )

        # Test same action on both states
        object_obstacle_reward = self.env.reward(object_obstacle_state, action="up")
        object_near_obstacle_reward = self.env.reward(object_near_obstacle_state, action="up")

        # Calculate expected rewards (no penalty applied since robot is safe in both cases)
        distance_to_target_obstacle = np.linalg.norm(self.obstacle_pos - self.target_pos)
        distance_to_target_near = np.linalg.norm(np.array([3.6, 3.6]) - self.target_pos)

        # Both should have only distance component (no penalty since robot is not in obstacle)
        expected_obstacle_reward = -distance_to_target_obstacle
        expected_near_reward = -distance_to_target_near

        # Verify the rewards are calculated correctly (no penalty for object-only collision)
        assert (
            abs(object_obstacle_reward - expected_obstacle_reward) < 1e-6
        ), f"Expected object obstacle reward {expected_obstacle_reward}, got {object_obstacle_reward}"
        assert (
            abs(object_near_obstacle_reward - expected_near_reward) < 1e-6
        ), f"Expected object near obstacle reward {expected_near_reward}, got {object_near_obstacle_reward}"

    def test_both_robot_and_object_obstacle_collision(self):
        """Test that only robot action collision applies penalty (not object position in obstacle)."""
        # Create state where robot action will lead to collision, object is in obstacle
        # Robot at (2.0, 3.0), action "right" moves to (3.0, 3.0) - collision!
        # Object at (7.0, 7.0) which is in the second obstacle - but this shouldn't affect penalty
        robot_collision_state = np.concatenate(
            [
                np.array([2.0, 3.0]),  # Robot position before action
                np.array([7.0, 7.0]),  # Object in different obstacle
                self.target_pos,  # Target position
            ]
        )

        # Create state where robot action will NOT lead to collision, object is safe
        # Robot at (1.0, 1.0), action "up" moves to (1.0, 2.0) - no collision
        # Object at (2.0, 2.0) which is in safe area
        both_safe_state = np.concatenate(
            [
                self.safe_pos,  # Robot in safe area
                np.array([2.0, 2.0]),  # Object in safe area
                self.target_pos,  # Target position
            ]
        )

        # Test actions on both states
        robot_collision_reward = self.env.reward(robot_collision_state, action="right")
        both_safe_reward = self.env.reward(both_safe_state, action="up")

        # Robot action leading to collision should get lower reward due to penalty
        # Note: robot_collision_state has object closer to target, so without penalty it would have higher reward
        # But with penalty, it should be lower
        assert (
            robot_collision_reward < both_safe_reward
        ), f"Robot collision reward ({robot_collision_reward}) should be < both safe reward ({both_safe_reward})"

        # Calculate expected penalties
        distance_to_target_robot_collision = np.linalg.norm(np.array([7.0, 7.0]) - self.target_pos)
        distance_to_target_safe = np.linalg.norm(np.array([2.0, 2.0]) - self.target_pos)

        # Only robot action collision applies penalty (not object position in obstacle)
        expected_robot_collision_reward = (
            -distance_to_target_robot_collision + self.env.obstacle_penalty
        )
        expected_safe_reward = -distance_to_target_safe

        # Verify the penalties are applied correctly (only one penalty for robot action collision)
        assert (
            abs(robot_collision_reward - expected_robot_collision_reward) < 1e-6
        ), f"Expected robot collision reward {expected_robot_collision_reward}, got {robot_collision_reward}"
        assert (
            abs(both_safe_reward - expected_safe_reward) < 1e-6
        ), f"Expected both safe reward {expected_safe_reward}, got {both_safe_reward}"

    def test_environment_initialization(self):
        """Test that Push POMDP environment initializes correctly."""
        assert self.env.name == "PushPOMDP"
        assert self.env.discount_factor == 0.95
        assert self.env.grid_size == 10
        assert self.env.push_threshold == 1.0
        assert self.env.friction_coefficient == 0.3
        assert self.env.observation_noise == 0.1
        assert self.env.obstacle_penalty == -10.0
        assert len(self.env.obstacles) == 2
        assert (3.0, 3.0) in self.env.obstacles
        assert (7.0, 7.0) in self.env.obstacles
        assert self.env.obstacle_radius == 0.5

    def test_action_space(self):
        """Test that action space is correct."""
        actions = self.env.get_actions()
        expected_actions = ["up", "down", "right", "left"]

        assert actions == expected_actions
        assert len(actions) == 4

    def test_initial_state_distribution(self):
        """Test initial state distribution."""
        initial_dist = self.env.initial_state_dist()

        # Sample some initial states to test
        sample_states = initial_dist.sample(n_samples=5)

        # Should have valid states
        assert len(sample_states) == 5

        # All states should be numpy arrays with correct shape
        for state in sample_states:
            assert isinstance(state, np.ndarray)
            assert state.shape == (6,)  # [robot_x, robot_y, object_x, object_y, target_x, target_y]
            assert not self.env.is_terminal(state)  # Initial states should not be terminal

            # Verify positions are within bounds
            assert np.all(state >= 0)
            assert np.all(state < self.env.grid_size)

            # Verify minimum distance from target (object should be at least 2.0 units from target)
            object_pos = state[2:4]
            target_pos = state[4:6]
            distance = np.linalg.norm(object_pos - target_pos)
            assert distance >= 2.0  # Minimum distance constraint

    def test_sample_next_state_uses_env_parameters(self):
        """Test that env.sample_next_state honors env physics parameters.

        Purpose: Validates that the env-API sample_next_state respects the
            env's grid_size, push_threshold, and friction_coefficient when
            computing transitions.

        Given: A PushPOMDP with known parameters and a (state, action) pair
            where the robot sits one cell to the left of the object so the
            object will be pushed.
        When: env.sample_next_state(state, action) is called.
        Then: The robot moves by one unit, the object is pushed by
            ``1 - friction_coefficient``, and the result is clipped within
            ``[0, grid_size - 1]``.

        Test type: unit
        """
        # Robot directly left of object so a "right" action both moves the
        # robot and pushes the object (push threshold is 1.0).
        test_state = np.array([1.0, 1.0, 2.0, 1.0, 9.0, 9.0])
        next_state = self.env.sample_next_state(test_state, "right")

        assert isinstance(next_state, np.ndarray)
        assert next_state.shape == (6,)
        # Robot moved one unit right.
        assert np.isclose(next_state[0], 2.0)
        # Object pushed right by (1 - friction_coefficient) = 0.7.
        assert np.isclose(next_state[2], 2.0 + (1.0 - self.env.friction_coefficient))
        # Target unchanged.
        assert np.array_equal(next_state[4:], test_state[4:])
        # All positions within grid bounds.
        assert np.all(next_state >= 0)
        assert np.all(next_state < self.env.grid_size)

    def test_sample_observation_uses_env_parameters(self):
        """Test that env.sample_observation honors env observation parameters.

        Purpose: Validates that the env-API sample_observation respects the
            env's observation_noise and grid_size, exactly mirroring the
            object position (with noise) and clamping to bounds.

        Given: A PushPOMDP and a known next_state.
        When: env.sample_observation(next_state, action) is called.
        Then: The observation has shape (6,), the robot/target slices match
            the state exactly, and the object slice is within grid bounds.

        Test type: unit
        """
        next_state = np.array([2.0, 1.0, 2.0, 2.0, 9.0, 9.0])
        observation = self.env.sample_observation(next_state, "right")

        assert isinstance(observation, np.ndarray)
        assert observation.shape == (6,)
        # Robot and target observed exactly.
        assert np.array_equal(observation[:2], next_state[:2])
        assert np.array_equal(observation[4:], next_state[4:])
        # Object position within grid bounds.
        assert 0.0 <= observation[2] < self.env.grid_size
        assert 0.0 <= observation[3] < self.env.grid_size

    def test_terminal_condition(self):
        """Test terminal condition detection."""
        # State with object far from target
        far_state = np.array([1.0, 1.0, 2.0, 2.0, 9.0, 9.0])
        assert not self.env.is_terminal(far_state)

        # State with object close to target
        close_state = np.array([1.0, 1.0, 8.9, 8.9, 9.0, 9.0])
        assert self.env.is_terminal(close_state)

        # State with object at target
        at_target_state = np.array([1.0, 1.0, 9.0, 9.0, 9.0, 9.0])
        assert self.env.is_terminal(at_target_state)

    def test_state_transition(self):
        """Test state transition with pushing behavior."""
        # Robot directly left of object; push_threshold of 2.0 ensures pushing.
        state = np.array([5.0, 5.0, 6.0, 5.0, 9.0, 9.0])
        action = "right"
        env = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=2.0,
            friction_coefficient=0.3,
        )

        next_state = env.sample_next_state(state, action)

        # Verify state dimensions
        assert next_state.shape == (6,)

        # Verify robot moved right
        assert next_state[0] > state[0]  # robot_x increased

        # Verify object was pushed (since robot was close enough)
        assert next_state[2] > state[2]  # object_x increased

        # Verify target position unchanged
        assert np.array_equal(next_state[4:], state[4:])

        # Verify positions within bounds
        assert np.all(next_state >= 0)
        assert np.all(next_state < env.grid_size)

    def test_state_transition_no_push(self):
        """Test state transition without pushing when robot is too far."""
        # Robot far from object so the push threshold gates out the push.
        state = np.array([1.0, 1.0, 8.0, 8.0, 9.0, 9.0])
        action = "right"
        env = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
        )

        next_state = env.sample_next_state(state, action)

        # Verify robot moved
        assert next_state[0] > state[0]

        # Verify object didn't move (too far from robot)
        assert np.array_equal(next_state[2:4], state[2:4])

    def test_observation_model_functionality(self):
        """Test observation model functionality with noise."""
        # Build an env whose observation parameters drive the env-API call.
        env = PushPOMDP(discount_factor=0.95, grid_size=10, observation_noise=0.1)
        state = np.array([5.0, 5.0, 4.0, 5.0, 9.0, 9.0])

        observation = env.sample_observation(state, "right")

        # Verify observation dimensions
        assert observation.shape == (6,)

        # Verify robot position is exact
        assert np.array_equal(observation[:2], state[:2])

        # Verify object position has noise
        assert not np.array_equal(observation[2:4], state[2:4])

        # Verify target position is exact
        assert np.array_equal(observation[4:], state[4:])

    def test_reward_function(self):
        """Test reward function behavior."""
        # Test reward for object far from target
        state_far = np.array([5.0, 5.0, 5.0, 5.0, 9.0, 9.0])
        reward_far = self.env.reward(state_far, "right")
        assert reward_far < 0  # Negative reward for distance

        # Test reward for object near target
        state_near = np.array([8.0, 8.0, 8.5, 8.5, 9.0, 9.0])
        reward_near = self.env.reward(state_near, "right")
        assert reward_near > reward_far  # Higher reward for being closer

        # Test reward for object at target
        state_at_target = np.array([9.0, 9.0, 9.0, 9.0, 9.0, 9.0])
        reward_at_target = self.env.reward(state_at_target, "right")
        assert reward_at_target > 0  # Positive reward for reaching target

    def test_reward_range(self):
        """Test that reward range is correctly calculated."""
        # Test with default grid size (10)
        env = PushPOMDP(discount_factor=0.95, grid_size=10)

        # Expected calculation from PushPOMDP constructor:
        # Maximum distance is diagonal from corner to corner: sqrt(2) * (grid_size - 1)
        # For grid_size=10: sqrt(2) * 9 ≈ 12.73
        max_distance = np.sqrt(2) * (10 - 1)
        expected_min = -max_distance  # Worst case: maximum distance to target
        expected_max = 100.0  # Best case: at target with bonus reward

        assert env.reward_range == (expected_min, expected_max)

        # Test with different grid size
        env2 = PushPOMDP(discount_factor=0.95, grid_size=25)

        max_distance2 = np.sqrt(2) * (25 - 1)  # sqrt(2) * 24 ≈ 33.94
        expected_min2 = -max_distance2
        expected_max2 = 100.0

        assert env2.reward_range == (expected_min2, expected_max2)

        # Verify the reward range bounds with actual rewards
        # Maximum distance in grid is from (0,0) to (grid_size-1, grid_size-1)
        max_distance = np.sqrt((9) ** 2 + (9) ** 2)  # Diagonal distance for 10x10 grid

        # State with maximum distance (object at origin, target at far corner)
        worst_state = np.array([0.0, 0.0, 0.0, 0.0, 9.0, 9.0])
        worst_reward = env.reward(worst_state, "right")

        # State at target (object reaches target for bonus reward)
        best_state = np.array([9.0, 9.0, 8.9, 8.9, 9.0, 9.0])  # Just under 0.5 distance
        best_reward = env.reward(best_state, "right")

        # Verify rewards are within expected range
        assert env.reward_range is not None
        assert worst_reward >= env.reward_range[0]
        assert best_reward <= env.reward_range[1]

    def test_is_equal_observation(self):
        """Test observation equality comparison."""
        # Test equal observations
        obs1 = np.array([1.0, 1.0, 2.0, 2.0, 9.0, 9.0])
        obs2 = np.array([1.0, 1.0, 2.0, 2.0, 9.0, 9.0])
        assert self.env.is_equal_observation(obs1, obs2)

        # Test different observations
        obs3 = np.array([1.0, 1.0, 2.1, 2.0, 9.0, 9.0])
        assert not self.env.is_equal_observation(obs1, obs3)

    def test_sample_next_step(self):
        """Test sample next step functionality."""
        state = np.array([5.0, 5.0, 4.0, 5.0, 9.0, 9.0])
        action = "right"

        next_state, observation, reward = self.env.sample_next_step(state, action)

        # Verify return types and shapes
        assert isinstance(next_state, np.ndarray)
        assert isinstance(observation, np.ndarray)
        assert isinstance(reward, float)
        assert next_state.shape == (6,)
        assert observation.shape == (6,)

        # Verify state transition
        assert next_state[0] > state[0]  # Robot moved right

        # Verify observation has noise on object position
        assert not np.array_equal(observation[2:4], next_state[2:4])

        # Verify reward is calculated
        assert reward == self.env.reward(next_state, action)

    def test_environment_equality(self):
        """Test environment equality comparison."""
        # Create two identical environments
        env1 = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            observation_noise=0.1,
        )
        env2 = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            observation_noise=0.1,
        )

        # Test equality
        assert env1 == env2

        # Test inequality with different parameters
        env3 = PushPOMDP(
            discount_factor=0.9,  # Different discount factor
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            observation_noise=0.1,
        )
        assert env1 != env3

    def test_config_id(self):
        """Test config_id behavior."""
        # Create two environments with same parameters
        env1 = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            observation_noise=0.1,
        )
        env2 = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            observation_noise=0.1,
        )

        # Test same config_id for identical environments
        assert env1.config_id == env2.config_id

        # Test different config_id for different environments
        env3 = PushPOMDP(
            discount_factor=0.9,  # Different discount factor
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            observation_noise=0.1,
        )
        assert env1.config_id != env3.config_id

    def test_observation_never_empty_from_sample(self):
        """Test that env.sample_observation never produces empty observations.

        Purpose: Validates that sample_observation always produces correctly
            shaped observations and that the resulting observations have a
            finite, non-negative log-probability under
            ``env.observation_log_probability``.

        Given: A PushPOMDP environment.
        When: Multiple observations are sampled via env.sample_observation.
        Then: All observations have shape (6,), are finite, and yield finite
            log-probabilities under env.observation_log_probability.

        Test type: unit
        """
        env = PushPOMDP(discount_factor=0.95, grid_size=10, observation_noise=0.1)
        next_state = np.array([5.0, 5.0, 4.0, 5.0, 9.0, 9.0])

        # Sample many observations to check consistency.
        for _ in range(20):
            observations = env.sample_observation(next_state, "right", n_samples=5)

            assert len(observations) == 5, "Should return requested number of observations"

            for obs in observations:
                assert isinstance(obs, np.ndarray), "Each observation should be numpy array"
                assert obs.shape == (6,), f"Expected shape (6,), got {obs.shape}"
                assert np.all(np.isfinite(obs)), "All observation values should be finite"

                # Test that this observation can be evaluated under the env API.
                try:
                    log_probs = env.observation_log_probability(next_state, "right", [obs])
                    assert len(log_probs) == 1, "Should return one log-probability"
                    log_prob = log_probs[0]
                    assert np.isfinite(log_prob), "Log-probability should be finite"
                except Exception as exc:  # pylint: disable=broad-except
                    pytest.fail(
                        f"Valid observation {obs} with shape {obs.shape} "
                        f"failed log-probability calculation: {exc}"
                    )

    def test_sample_next_step_observation_never_empty(self):
        """Test that sample_next_step never produces empty observations.

        Purpose: Validates that the high-level sample_next_step method produces valid observations

        Given: A valid PushPOMDP environment and various states
        When: sample_next_step is called multiple times
        Then: All returned observations should be properly shaped and non-empty

        Test type: integration
        """
        # Test with various initial states
        test_states = [
            np.array([1.0, 1.0, 2.0, 2.0, 9.0, 9.0]),  # Normal state
            np.array([0.0, 0.0, 1.0, 1.0, 9.0, 9.0]),  # Corner state
            np.array([9.0, 9.0, 8.0, 8.0, 9.0, 9.0]),  # Near target
            np.array([5.0, 5.0, 5.5, 5.5, 9.0, 9.0]),  # Close robot-object
        ]

        actions = self.env.get_actions()

        for state in test_states:
            for action in actions:
                # Call sample_next_step multiple times to check consistency
                for _ in range(5):
                    next_state, observation, reward = self.env.sample_next_step(state, action)

                    # Check observation properties
                    assert isinstance(observation, np.ndarray), "Observation should be numpy array"
                    assert observation.shape == (
                        6,
                    ), f"Expected observation shape (6,), got {observation.shape}"
                    assert len(observation) > 0, "Observation should not be empty"
                    assert np.all(
                        np.isfinite(observation)
                    ), "All observation values should be finite"

                    # Verify observation can be evaluated under env-API log-prob.
                    try:
                        log_probs = self.env.observation_log_probability(
                            next_state, action, [observation]
                        )
                        assert len(log_probs) == 1, "Should return one log-probability"
                        log_prob = log_probs[0]
                        assert np.isfinite(log_prob), "Log-probability should be finite"
                    except Exception as exc:  # pylint: disable=broad-except
                        pytest.fail(
                            f"sample_next_step produced invalid observation "
                            f"{observation} with shape {observation.shape}: {exc}"
                        )

    def test_state_transition_probability_deterministic(self):
        """Test that env.transition_log_probability correctly identifies deterministic transitions.

        Purpose: Validates that ``np.exp(env.transition_log_probability(...))``
            returns 1.0 for the correct next state and 0.0 for others when
            transition_error_prob=0.

        Given: A PushPOMDP without obstacles or transition error and a known
            (state, action) pair.
        When: env.transition_log_probability is queried with various
            candidate next states.
        Then: Only the actual sampled next state has probability 1.0; all
            others have probability 0.0.

        Test type: unit
        """
        state = np.array([2.0, 2.0, 3.0, 3.0, 9.0, 9.0])
        action = "right"
        env = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            obstacles=[],
            obstacle_radius=0.5,
        )

        # Get the actual next state via the env API.
        actual_next_state = env.sample_next_state(state, action)

        # Test probability for actual next state.
        prob_actual = np.exp(env.transition_log_probability(state, action, [actual_next_state]))
        assert len(prob_actual) == 1
        assert prob_actual[0] == 1.0, "Actual next state should have probability 1.0"

        # Test probability for different states.
        different_state1 = np.array([2.0, 2.0, 3.0, 3.0, 9.0, 9.0])  # Same as initial
        different_state2 = np.array([5.0, 5.0, 5.0, 5.0, 9.0, 9.0])  # Completely different
        different_state3 = actual_next_state + np.array(
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
        )  # Slightly off

        prob_diff1 = np.exp(env.transition_log_probability(state, action, [different_state1]))
        prob_diff2 = np.exp(env.transition_log_probability(state, action, [different_state2]))
        prob_diff3 = np.exp(env.transition_log_probability(state, action, [different_state3]))

        assert prob_diff1[0] == 0.0, "Different state should have probability 0.0"
        assert prob_diff2[0] == 0.0, "Different state should have probability 0.0"
        assert prob_diff3[0] == 0.0, "Slightly different state should have probability 0.0"

    def test_state_transition_probability_with_obstacles(self):
        """Test state transition probability when robot collides with obstacles.

        Purpose: Validates that probability() correctly handles obstacle collision scenarios

        Given: A state where robot movement will be blocked by an obstacle
        When: The probability method is called with potential next states
        Then: Only the state where robot stays in place should have probability 1.0

        Test type: unit
        """
        obstacles = [(3.0, 3.0)]
        state = np.array([2.5, 3.0, 1.0, 1.0, 9.0, 9.0])  # Robot just left of obstacle
        action = "right"  # Will collide with obstacle
        env = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            obstacles=obstacles,
            obstacle_radius=0.5,
        )

        # Get the actual next state (robot should stay in place due to collision).
        actual_next_state = env.sample_next_state(state, action)

        # Verify robot didn't move (stayed at same position due to collision).
        assert np.allclose(
            actual_next_state[:2], state[:2]
        ), "Robot should stay in place when colliding with obstacle"

        # Test probability for actual next state.
        prob_actual = np.exp(env.transition_log_probability(state, action, [actual_next_state]))
        assert prob_actual[0] == 1.0, "Actual next state should have probability 1.0"

        # Test probability for state where robot would have moved (if no obstacle).
        hypothetical_moved_state = state.copy()
        hypothetical_moved_state[0] += 1.0  # Robot moved right
        prob_moved = np.exp(
            env.transition_log_probability(state, action, [hypothetical_moved_state])
        )
        assert prob_moved[0] == 0.0, "State with robot moved should have probability 0.0"

    def test_state_transition_probability_pushing_object(self):
        """Test state transition probability when robot pushes object.

        Purpose: Validates that probability() correctly handles object pushing with friction

        Given: A state where robot is close enough to push the object
        When: The probability method is called with potential next states
        Then: Only the state with correct push dynamics should have probability 1.0

        Test type: unit
        """
        state = np.array([2.0, 2.0, 2.5, 2.0, 9.0, 9.0])  # Robot close to object
        action = "right"
        friction = 0.5
        env = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=1.0,  # Robot is within push threshold
            friction_coefficient=friction,
            obstacles=[],
            obstacle_radius=0.5,
        )

        # Get the actual next state with push dynamics.
        actual_next_state = env.sample_next_state(state, action)

        # Verify both robot and object moved.
        assert actual_next_state[0] > state[0], "Robot should have moved right"
        assert actual_next_state[2] > state[2], "Object should have been pushed right"

        # Verify friction was applied (object moves less than robot).
        expected_object_movement = 1.0 * (1 - friction)  # movement * (1 - friction)
        actual_object_movement = actual_next_state[2] - state[2]
        assert np.isclose(
            actual_object_movement, expected_object_movement, atol=0.01
        ), "Object movement should account for friction"

        # Test probability for actual next state.
        prob_actual = np.exp(env.transition_log_probability(state, action, [actual_next_state]))
        assert prob_actual[0] == 1.0, "Actual next state should have probability 1.0"

        # Test probability for state where object didn't move.
        no_push_state = state.copy()
        no_push_state[0] += 1.0  # Only robot moved
        prob_no_push = np.exp(env.transition_log_probability(state, action, [no_push_state]))
        assert prob_no_push[0] == 0.0, "State without object push should have probability 0.0"

    def test_state_transition_probability_boundary_clipping(self):
        """Test state transition probability with boundary clipping.

        Purpose: Validates that probability() correctly handles grid boundary constraints

        Given: A state where robot is at grid boundary
        When: Robot attempts to move beyond boundary
        Then: Only the state with clipped position should have probability 1.0

        Test type: unit
        """
        state = np.array([0.0, 0.0, 2.0, 2.0, 9.0, 9.0])  # Robot at corner
        action = "left"  # Try to move beyond boundary
        env = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            obstacles=[],
            obstacle_radius=0.5,
        )

        # Get actual next state (should be clipped at boundary).
        actual_next_state = env.sample_next_state(state, action)

        # Verify robot position was clipped at boundary.
        assert actual_next_state[0] == 0.0, "Robot x-position should be clipped at 0"
        assert actual_next_state[1] == 0.0, "Robot y-position should remain at 0"

        # Test probability for actual next state.
        prob_actual = np.exp(env.transition_log_probability(state, action, [actual_next_state]))
        assert prob_actual[0] == 1.0, "Actual next state should have probability 1.0"

        # Test probability for state with negative position (if no clipping).
        hypothetical_negative_state = state.copy()
        hypothetical_negative_state[0] = -1.0  # Beyond boundary
        prob_negative = np.exp(
            env.transition_log_probability(state, action, [hypothetical_negative_state])
        )
        assert prob_negative[0] == 0.0, "State beyond boundary should have probability 0.0"

    def test_state_transition_probability_multiple_states(self):
        """Test state transition probability with multiple candidate states.

        Purpose: Validates that probability() correctly handles batch evaluation of multiple states

        Given: A state transition and a list of multiple potential next states
        When: The probability method is called with the list
        Then: Should return probability array with 1.0 only for the correct state

        Test type: unit
        """
        state = np.array([5.0, 5.0, 6.0, 5.0, 9.0, 9.0])
        action = "up"
        env = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            obstacles=[],
            obstacle_radius=0.5,
        )

        # Get actual next state.
        actual_next_state = env.sample_next_state(state, action)

        # Create list of candidate states (only one should be correct).
        candidate_states = [
            np.array([5.0, 6.0, 6.0, 5.7, 9.0, 9.0]),  # Possible next state
            actual_next_state,  # The correct next state
            np.array([5.0, 5.0, 6.0, 5.0, 9.0, 9.0]),  # Initial state (shouldn't match)
            np.array([4.0, 5.0, 5.0, 5.0, 9.0, 9.0]),  # Different state
        ]

        # Test probability for all candidates.
        probs = np.exp(env.transition_log_probability(state, action, candidate_states))

        # Verify output shape.
        assert len(probs) == 4, "Should return probability for each candidate"

        # Verify only the actual next state has probability 1.0.
        assert probs[1] == 1.0, "Actual next state should have probability 1.0"
        assert probs[0] == 0.0, "Incorrect state should have probability 0.0"
        assert probs[2] == 0.0, "Initial state should have probability 0.0"
        assert probs[3] == 0.0, "Different state should have probability 0.0"

        # Verify probabilities sum to 1.0 (since only one state is possible).
        assert np.sum(probs) == 1.0, "Probabilities should sum to 1.0"

    def test_fixed_initial_state_returns_exact_state(self):
        """Test that fixed initial state distribution returns the exact specified state.

        Purpose: Validates that providing initial_state parameter creates a deterministic
                 distribution that always returns the same state.

        Given: A PushPOMDP environment initialized with a fixed initial_state
        When: Sampling from initial_state_dist multiple times
        Then: All samples should be identical to the provided initial_state

        Test type: unit
        """
        fixed_state = np.array([2.0, 3.0, 5.0, 5.0, 9.0, 9.0])
        env = PushPOMDP(discount_factor=0.95, initial_state=fixed_state)

        # Sample multiple times
        samples = [env.initial_state_dist().sample()[0] for _ in range(5)]

        for sample in samples:
            assert np.array_equal(
                sample, fixed_state
            ), f"Sample {sample} should equal fixed state {fixed_state}"

    def test_fixed_initial_state_n_samples(self):
        """Test that fixed initial state distribution returns correct number of samples.

        Purpose: Validates that n_samples parameter works correctly with fixed state.

        Given: A PushPOMDP environment initialized with a fixed initial_state
        When: Calling sample with n_samples > 1
        Then: All returned samples should be identical to the provided initial_state

        Test type: unit
        """
        fixed_state = np.array([1.0, 2.0, 4.0, 4.0, 9.0, 9.0])
        env = PushPOMDP(discount_factor=0.95, initial_state=fixed_state)

        samples = env.initial_state_dist().sample(n_samples=10)

        assert len(samples) == 10, "Should return exactly 10 samples"
        for i, sample in enumerate(samples):
            assert np.array_equal(sample, fixed_state), f"Sample {i} should equal fixed state"

    def test_fixed_initial_state_is_copied(self):
        """Test that fixed initial state samples are independent copies.

        Purpose: Validates that modifying a sampled state doesn't affect other samples.

        Given: A PushPOMDP environment initialized with a fixed initial_state
        When: Sampling and modifying one sample
        Then: Other samples and future samples should remain unchanged

        Test type: unit
        """
        fixed_state = np.array([2.0, 3.0, 5.0, 5.0, 9.0, 9.0])
        env = PushPOMDP(discount_factor=0.95, initial_state=fixed_state)

        sample1 = env.initial_state_dist().sample()[0]
        sample1[0] = 999.0  # Modify the sample

        sample2 = env.initial_state_dist().sample()[0]

        assert sample2[0] == 2.0, "Second sample should be unaffected by modification"
        assert np.array_equal(sample2, fixed_state), "Second sample should equal original"

    def test_fixed_initial_state_observation_dist(self):
        """Test that initial_observation_dist also uses fixed state.

        Purpose: Validates that initial_observation_dist returns the same fixed state.

        Given: A PushPOMDP environment initialized with a fixed initial_state
        When: Sampling from initial_observation_dist
        Then: The returned observation should match the fixed state

        Test type: unit
        """
        fixed_state = np.array([3.0, 4.0, 6.0, 6.0, 9.0, 9.0])
        env = PushPOMDP(discount_factor=0.95, initial_state=fixed_state)

        observation = env.initial_observation_dist().sample()[0]

        assert np.array_equal(
            observation, fixed_state
        ), "Initial observation should equal fixed state"

    def test_no_initial_state_returns_random_states(self):
        """Test that without initial_state, distribution returns random states.

        Purpose: Validates that default behavior (no initial_state) produces random states.

        Given: A PushPOMDP environment without initial_state parameter
        When: Sampling multiple times from initial_state_dist
        Then: At least some samples should be different from each other

        Test type: unit
        """
        env = PushPOMDP(discount_factor=0.95)

        samples = [env.initial_state_dist().sample()[0] for _ in range(10)]

        # Check that not all samples are identical (random behavior)
        all_same = all(np.array_equal(samples[0], s) for s in samples[1:])
        assert not all_same, "Random initial states should produce different samples"

    def test_random_initial_state_avoids_obstacles(self):
        """Test that random initial state distribution avoids obstacle positions.

        Purpose: Validates that randomly generated states don't place robot or object
                 in obstacle positions.

        Given: A PushPOMDP environment with obstacles and no fixed initial_state
        When: Sampling multiple initial states
        Then: Neither robot nor object should be within obstacle radius

        Test type: unit
        """
        obstacles = [(3.0, 3.0), (7.0, 7.0)]
        env = PushPOMDP(
            discount_factor=0.95,
            obstacles=obstacles,
            obstacle_radius=0.5,
        )

        for _ in range(20):
            state = env.initial_state_dist().sample()[0]
            robot_pos = state[:2]
            object_pos = state[2:4]

            assert not env._is_colliding_with_obstacle(
                robot_pos
            ), f"Robot at {robot_pos} should not be in obstacle"
            assert not env._is_colliding_with_obstacle(
                object_pos
            ), f"Object at {object_pos} should not be in obstacle"

    def test_random_initial_state_object_away_from_target(self):
        """Test that random initial state places object away from target.

        Purpose: Validates that randomly generated states place the object at least
                 2.0 units away from the target position.

        Given: A PushPOMDP environment without fixed initial_state
        When: Sampling multiple initial states
        Then: Object should be at least 2.0 units from target position

        Test type: unit
        """
        env = PushPOMDP(discount_factor=0.95, grid_size=10)

        for _ in range(20):
            state = env.initial_state_dist().sample()[0]
            object_pos = state[2:4]
            target_pos = state[4:6]

            distance = np.linalg.norm(object_pos - target_pos)
            assert (
                distance >= 2.0
            ), f"Object at {object_pos} should be >= 2.0 units from target {target_pos}"

    def test_state_transition_all_actions_deterministic(self):
        """Test env.sample_next_state and env.transition_log_probability for all actions with transition_error_prob=0.

        Purpose: Validates that all actions (up, down, left, right) execute correctly in deterministic mode

        Given: A PushPOMDP with transition_error_prob=0 and hardcoded states
        When: Each action is executed
        Then: Resulting state should exactly match hardcoded expected state

        Test type: unit
        """
        # Hardcoded initial state: robot at (5.0, 5.0), object at (5.0, 5.0), target at (9.0, 9.0).
        # Robot and object are at same position, so pushing will occur.
        initial_state = np.array([5.0, 5.0, 5.0, 5.0, 9.0, 9.0])
        env = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=2.0,  # Large enough so robot stays within threshold after moving
            friction_coefficient=0.3,
            obstacles=[],
            obstacle_radius=0.5,
            transition_error_prob=0.0,  # Explicitly set to 0 for deterministic behavior
        )

        # Hardcoded expected states for each action.
        # Robot moves by 1.0, object moves by 1.0 * (1 - 0.3) = 0.7.
        expected_states = {
            "up": np.array([5.0, 6.0, 5.0, 5.7, 9.0, 9.0]),  # Robot up, object pushed up
            "down": np.array([5.0, 4.0, 5.0, 4.3, 9.0, 9.0]),  # Robot down, object pushed down
            "right": np.array([6.0, 5.0, 5.7, 5.0, 9.0, 9.0]),  # Robot right, object pushed right
            "left": np.array([4.0, 5.0, 4.3, 5.0, 9.0, 9.0]),  # Robot left, object pushed left
        }

        for action_name, expected_state in expected_states.items():
            # Test sample_next_state - should return exact expected state.
            next_state = env.sample_next_state(initial_state.copy(), action_name)
            assert np.array_equal(
                next_state, expected_state
            ), f"Action '{action_name}': Expected {expected_state}, got {next_state}"

            # Test transition_log_probability - should give probability 1.0 for expected state.
            prob_expected = np.exp(
                env.transition_log_probability(initial_state, action_name, [expected_state])
            )
            assert len(prob_expected) == 1
            assert prob_expected[0] == 1.0, (
                f"Action '{action_name}': Expected state should have probability 1.0, "
                f"got {prob_expected[0]}"
            )

            # Test transition_log_probability for initial state (should be 0.0).
            prob_initial = np.exp(
                env.transition_log_probability(initial_state, action_name, [initial_state])
            )
            assert prob_initial[0] == 0.0, (
                f"Action '{action_name}': Initial state should have probability 0.0, "
                f"got {prob_initial[0]}"
            )


class TestPushPOMDPStateTransition:
    """Test cases for PushPOMDP state transition methods."""

    def test_state_transition_all_actions_no_push_deterministic(self):
        """Test state transition for all actions when robot is too far to push object.

        Purpose: Validates that all actions work correctly when robot cannot push object

        Given: A PushPOMDP with robot far from object and transition_error_prob=0
        When: Each action is executed
        Then: Resulting state should exactly match hardcoded expected state (robot moves, object doesn't)

        Test type: unit
        """
        # Hardcoded initial state: robot at (1.0, 1.0), object at (8.0, 8.0), target at (9.0, 9.0).
        # Robot is far from object, so no pushing will occur.
        initial_state = np.array([1.0, 1.0, 8.0, 8.0, 9.0, 9.0])
        env = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            obstacles=[],
            obstacle_radius=0.5,
            transition_error_prob=0.0,  # Explicitly set to 0 for deterministic behavior
        )

        # Hardcoded expected states for each action.
        # Robot moves by 1.0, object stays at (8.0, 8.0).
        expected_states = {
            "up": np.array([1.0, 2.0, 8.0, 8.0, 9.0, 9.0]),  # Robot up, object unchanged
            "down": np.array([1.0, 0.0, 8.0, 8.0, 9.0, 9.0]),  # Robot down, object unchanged
            "right": np.array([2.0, 1.0, 8.0, 8.0, 9.0, 9.0]),  # Robot right, object unchanged
            "left": np.array([0.0, 1.0, 8.0, 8.0, 9.0, 9.0]),  # Robot left, object unchanged
        }

        for action_name, expected_state in expected_states.items():
            # Test sample_next_state - should return exact expected state.
            next_state = env.sample_next_state(initial_state.copy(), action_name)
            assert np.array_equal(
                next_state, expected_state
            ), f"Action '{action_name}': Expected {expected_state}, got {next_state}"

            # Test transition_log_probability - should give probability 1.0 for expected state.
            prob_expected = np.exp(
                env.transition_log_probability(initial_state, action_name, [expected_state])
            )
            assert prob_expected[0] == 1.0, (
                f"Action '{action_name}': Expected state should have probability 1.0, "
                f"got {prob_expected[0]}"
            )

            # Test transition_log_probability for initial state (should be 0.0).
            prob_initial = np.exp(
                env.transition_log_probability(initial_state, action_name, [initial_state])
            )
            assert prob_initial[0] == 0.0, (
                f"Action '{action_name}': Initial state should have probability 0.0, "
                f"got {prob_initial[0]}"
            )


class TestSampleNextStepEquivalence:
    """Test that the optimized sample_next_step produces identical results to the base class."""

    def test_sample_next_step_matches_base_class(self):
        """Test optimized sample_next_step agrees with base Environment.sample_next_step.

        Purpose: Validates that the inlined sample_next_step override produces
        identical results to the original base class implementation.

        Given: A PushPOMDP environment and valid (state, action) pairs
        When: Both the optimized and base class sample_next_step are called
              with the same numpy RNG seed
        Then: next_state, observation, and reward are identical

        Test type: unit
        """
        from POMDPPlanners.core.environment import Environment

        env = PushPOMDP(discount_factor=0.95)
        state = env.initial_state_dist().sample()[0]

        for action in env.get_actions():
            for _ in range(20):
                seed = np.random.randint(0, 2**31)

                np.random.seed(seed)
                opt_next, opt_obs, opt_reward = env.sample_next_step(state, action)

                np.random.seed(seed)
                base_next, base_obs, base_reward = Environment.sample_next_step(env, state, action)

                np.testing.assert_array_almost_equal(
                    opt_next,
                    base_next,
                    decimal=10,
                    err_msg=f"next_state mismatch for action={action}, seed={seed}",
                )
                np.testing.assert_array_almost_equal(
                    opt_obs,
                    base_obs,
                    decimal=10,
                    err_msg=f"observation mismatch for action={action}, seed={seed}",
                )
                assert abs(opt_reward - base_reward) < 1e-10, (
                    f"reward mismatch for action={action}, seed={seed}: "
                    f"{opt_reward} != {base_reward}"
                )


class TestRewardBatchMatchesScalarLoop:
    """Test that reward_batch produces distributional results equivalent to scalar reward()."""

    def test_push_reward_batch_matches_scalar_loop(self):
        """Test that reward_batch summary statistics match the scalar-loop equivalent.

        Purpose: Validates that the vectorised reward_batch override produces
            rewards drawn from the same distribution as the scalar reward() loop.
            Both paths internally re-sample a fresh next state, so exact value
            equality per-particle is not expected; instead we compare the
            empirical mean and std over a large particle set (N=500) and verify
            they agree within 5% relative on mean and 10% relative on std.

        Given: A PushPOMDP(discount_factor=0.95) with obstacles.  32 initial
            particles sampled from the initial-state distribution with seed 0.
            For each of the four actions, a large particle set of 500 copies of
            the same state is evaluated so the per-particle stochasticity
            averages out.

        When: reward_batch(particles, action) is called once (with np.random
            seeded to 123 before the call) to obtain a batch of 500 rewards.
            Separately, scalar env.reward(s, action) is called 500 times with
            the same seed (123 then incremented per-call) to obtain 500 scalar
            rewards.

        Then: For every action, abs(mean_batch - mean_scalar) / max(|mean_scalar|, 1e-6)
            < 0.05 and abs(std_batch - std_scalar) / max(std_scalar, 1e-6) < 0.10.

        Test type: unit
        """
        _N_PARTICLES = 500
        _MEAN_TOL = 0.05
        _STD_TOL = 0.10

        env = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            obstacles=[(3.0, 3.0), (7.0, 7.0)],
            obstacle_radius=0.5,
            obstacle_penalty=-10.0,
        )

        np.random.seed(0)
        random.seed(0)
        initial_state = env.initial_state_dist().sample()[0]

        # Tile the same state to get a large particle array
        particles = np.tile(initial_state, (_N_PARTICLES, 1))

        for action in env.get_actions():
            # --- batch path ---
            np.random.seed(123)
            random.seed(123)
            batch_rewards = env.reward_batch(particles, action)

            # --- scalar loop path (independent draws, same seed base) ---
            np.random.seed(123)
            random.seed(123)
            scalar_rewards = np.array(
                [env.reward(initial_state, action) for _ in range(_N_PARTICLES)]
            )

            mean_batch = float(np.mean(batch_rewards))
            mean_scalar = float(np.mean(scalar_rewards))
            std_batch = float(np.std(batch_rewards))
            std_scalar = float(np.std(scalar_rewards))

            mean_ref = max(abs(mean_scalar), 1e-6)
            std_ref = max(std_scalar, 1e-6)

            assert abs(mean_batch - mean_scalar) / mean_ref < _MEAN_TOL, (
                f"action={action}: mean_batch={mean_batch:.4f}, mean_scalar={mean_scalar:.4f}, "
                f"rel_diff={abs(mean_batch - mean_scalar) / mean_ref:.4f} >= {_MEAN_TOL}"
            )
            assert abs(std_batch - std_scalar) / std_ref < _STD_TOL, (
                f"action={action}: std_batch={std_batch:.4f}, std_scalar={std_scalar:.4f}, "
                f"rel_diff={abs(std_batch - std_scalar) / std_ref:.4f} >= {_STD_TOL}"
            )

    def test_push_reward_batch_shape_and_dtype(self):
        """Test that reward_batch returns a correctly shaped float64 array.

        Purpose: Validates the output shape, dtype, and basic finite-ness of reward_batch.

        Given: A PushPOMDP and a batch of 16 particles.
        When: reward_batch is called for each action.
        Then: Output has shape (16,), dtype float64, and all finite values.

        Test type: unit
        """
        env = PushPOMDP(discount_factor=0.95)
        np.random.seed(7)
        particles = env.initial_state_dist().sample(n_samples=16)
        particles_arr = np.array(particles)

        for action in env.get_actions():
            result = env.reward_batch(particles_arr, action)
            assert result.shape == (16,), f"Expected shape (16,) for action={action}"
            assert result.dtype == np.float64, f"Expected float64 dtype for action={action}"
            assert np.all(np.isfinite(result)), f"Expected finite rewards for action={action}"

    def test_push_reward_batch_with_obstacles_applies_penalty(self):
        """Test that reward_batch correctly applies obstacle penalty when robot would collide.

        Purpose: Validates that the vectorised obstacle-penalty branch in reward_batch
            produces penalty values identical to the scalar reward() path.

        Given: A PushPOMDP with one obstacle at (3, 3).  States where the robot
            will collide when moving 'right' (robot at (2, 3)) and states where
            there is no collision (robot at (1, 1)).

        When: reward_batch is called on a homogeneous batch of collision /
            no-collision states.  The next states are deterministic (no friction,
            no transition error) so exact value equality holds.

        Then: Collision-state batch rewards equal the scalar reward for that
            state (same next-state drawn from a deterministic transition).
            No-collision batch rewards have no obstacle penalty component.

        Test type: unit
        """
        env = PushPOMDP(
            discount_factor=0.95,
            obstacles=[(3.0, 3.0)],
            obstacle_radius=0.5,
            obstacle_penalty=-10.0,
            transition_error_prob=0.0,
            friction_coefficient=0.0,
        )

        # State where 'right' triggers penalty: robot at (2,3), intended=(3,3) in obstacle
        collision_state = np.array([2.0, 3.0, 5.0, 5.0, 9.0, 9.0])
        no_collision_state = np.array([1.0, 1.0, 5.0, 5.0, 9.0, 9.0])

        np.random.seed(0)
        collision_batch = env.reward_batch(np.tile(collision_state, (10, 1)), "right")
        np.random.seed(0)
        collision_scalar = env.reward(collision_state, "right")

        # All rewards in the batch should equal the scalar (deterministic transition)
        np.testing.assert_allclose(collision_batch, collision_scalar, rtol=1e-10)

        np.random.seed(1)
        no_col_batch = env.reward_batch(np.tile(no_collision_state, (10, 1)), "right")
        np.random.seed(1)
        no_col_scalar = env.reward(no_collision_state, "right")

        np.testing.assert_allclose(no_col_batch, no_col_scalar, rtol=1e-10)

        # Collision rewards should be lower than no-collision rewards
        assert np.mean(collision_batch) < np.mean(no_col_batch)


class TestPushNativeSimulateRollout:
    """Tests for the native C++ simulate_rollout_discrete entry point."""

    def test_push_native_simulate_rollout_matches_python(self):
        """Native C++ rollout must produce the same discounted return as the Python reference.

        Purpose: Validates that simulate_rollout_discrete in C++ is a byte-exact
            port of PushPOMDP._python_simulate_random_rollout for deterministic
            (transition_error_prob=0) rollouts with a fixed action sequence.

        Given: A PushPOMDP with transition_error_prob=0 and a fixed initial state.
            A deterministic action sequence is pre-drawn and shared between the
            Python and native paths so that both see the same action at each step.

        When: Both paths execute the rollout from the same state with the same
            action sequence and environment parameters.

        Then: The discounted returns agree to within atol=1e-9, confirming that
            the C++ transition, reward, and terminal-check kernels are faithful
            ports of the Python originals.

        Test type: integration
        """
        _ACTIONS = ["up", "down", "right", "left"]

        env = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            observation_noise=0.1,
            obstacles=[(3.0, 3.0), (7.0, 7.0)],
            obstacle_radius=0.5,
            obstacle_penalty=-10.0,
            transition_error_prob=0.0,
        )

        initial_state = np.array([2.0, 3.0, 2.5, 3.1, 9.0, 9.0], dtype=np.float64)
        max_depth = 20
        depth = 0

        # Pre-draw action indices that will be shared between both paths.
        np.random.seed(123)
        action_indices = np.random.randint(0, 4, size=max_depth, dtype=np.int64)
        action_strings = [_ACTIONS[i] for i in action_indices]

        # Python reference path (uses the pre-drawn action string list).
        python_return = env._python_simulate_random_rollout(
            state=initial_state,
            actions=action_strings,
            max_depth=max_depth,
            discount_factor=env.discount_factor,
            depth=depth,
        )

        # Native C++ path: seed must NOT be set here — with transition_error_prob=0
        # the C++ kernel makes zero RNG calls, so the result is fully determined
        # by the action_indices array.
        obs_arr = env._get_native_rollout_obstacles()
        state_arr = np.asarray(initial_state, dtype=np.float64)
        native_return = float(
            push_native.simulate_rollout_discrete(
                state=state_arr,
                action_indices=action_indices,
                max_depth=max_depth,
                depth=depth,
                discount=env.discount_factor,
                grid_size=float(env.grid_size),
                push_threshold=float(env.push_threshold),
                friction_coefficient=float(env.friction_coefficient),
                obstacles=obs_arr,
                obstacle_radius=float(env.obstacle_radius),
                obstacle_penalty=float(env.obstacle_penalty),
                transition_error_prob=float(env.transition_error_prob),
            )
        )

        np.testing.assert_allclose(
            native_return,
            python_return,
            atol=1e-9,
            err_msg=(
                f"Native rollout ({native_return:.15f}) does not match "
                f"Python reference ({python_return:.15f})."
            ),
        )

    def test_push_native_simulate_rollout_no_obstacles(self):
        """Native rollout with no obstacles matches Python reference.

        Purpose: Validates the obstacle-free code path in simulate_rollout_discrete.

        Given: A PushPOMDP with no obstacles and transition_error_prob=0.
        When: Both paths execute from the same state with the same action sequence.
        Then: Returns match to within atol=1e-9.

        Test type: integration
        """
        _ACTIONS = ["up", "down", "right", "left"]

        env = PushPOMDP(
            discount_factor=0.99,
            grid_size=10,
            push_threshold=1.5,
            friction_coefficient=0.2,
            observation_noise=0.1,
            obstacles=None,
            obstacle_radius=0.5,
            obstacle_penalty=-5.0,
            transition_error_prob=0.0,
        )

        initial_state = np.array([0.0, 0.0, 5.0, 5.0, 9.0, 9.0], dtype=np.float64)
        max_depth = 15

        np.random.seed(999)
        action_indices = np.random.randint(0, 4, size=max_depth, dtype=np.int64)
        action_strings = [_ACTIONS[i] for i in action_indices]

        python_return = env._python_simulate_random_rollout(
            state=initial_state,
            actions=action_strings,
            max_depth=max_depth,
            discount_factor=env.discount_factor,
        )

        obs_arr = env._get_native_rollout_obstacles()
        native_return = float(
            push_native.simulate_rollout_discrete(
                state=np.asarray(initial_state, dtype=np.float64),
                action_indices=action_indices,
                max_depth=max_depth,
                depth=0,
                discount=env.discount_factor,
                grid_size=float(env.grid_size),
                push_threshold=float(env.push_threshold),
                friction_coefficient=float(env.friction_coefficient),
                obstacles=obs_arr,
                obstacle_radius=float(env.obstacle_radius),
                obstacle_penalty=float(env.obstacle_penalty),
                transition_error_prob=float(env.transition_error_prob),
            )
        )

        np.testing.assert_allclose(
            native_return,
            python_return,
            atol=1e-9,
            err_msg=(
                f"Native rollout no-obstacles ({native_return:.15f}) != "
                f"Python ({python_return:.15f})"
            ),
        )

    def test_push_native_rollout_terminal_state_returns_zero(self):
        """Rollout from a terminal state returns zero immediately.

        Purpose: Validates that simulate_rollout_discrete correctly detects a
            terminal initial state and returns 0.0 without stepping.

        Given: A PushPOMDP and a state where the object is already at the target.
        When: simulate_random_rollout is called from the terminal state.
        Then: The return value is 0.0.

        Test type: unit
        """
        env = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            observation_noise=0.1,
            transition_error_prob=0.0,
        )

        # Object is at the target: (9-9)^2 + (9-9)^2 = 0 < 0.25, terminal.
        terminal_state = np.array([1.0, 1.0, 9.0, 9.0, 9.0, 9.0], dtype=np.float64)
        assert env.is_terminal(terminal_state)

        class _DummySampler:
            def sample(self) -> str:
                return "up"

        result = env.simulate_random_rollout(
            state=terminal_state,
            action_sampler=_DummySampler(),
            max_depth=20,
            discount_factor=env.discount_factor,
        )
        assert result == 0.0
