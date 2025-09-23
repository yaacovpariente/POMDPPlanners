"""Tests for Push POMDP environment.

This module tests the Push POMDP environment, focusing on:
- Basic environment functionality
- Obstacle collision penalties
- Reward verification between obstacle and non-obstacle states
"""

import random
from pathlib import Path

import numpy as np
import pytest

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)

from POMDPPlanners.environments.push_pomdp import (
    PushObservation,
    PushPOMDP,
    PushStateTransition,
)


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
        assert self.env._is_colliding_with_obstacle(np.array([3.0, 3.0])) == True
        assert self.env._is_colliding_with_obstacle(np.array([3.2, 3.2])) == True  # Within radius
        assert self.env._is_colliding_with_obstacle(np.array([2.8, 3.1])) == True  # Within radius

        # Position outside obstacle
        assert self.env._is_colliding_with_obstacle(np.array([1.0, 1.0])) == False
        assert self.env._is_colliding_with_obstacle(np.array([5.0, 5.0])) == False
        assert self.env._is_colliding_with_obstacle(np.array([3.6, 3.6])) == False  # Outside radius

    def test_reward_difference_obstacle_vs_safe(self):
        """Test that rewards in obstacle states are lower than in safe states."""
        # Create states with robot in different areas
        obstacle_state = np.concatenate(
            [
                self.obstacle_pos,  # Robot in obstacle
                self.object_pos,  # Object position
                self.target_pos,  # Target position
            ]
        )

        safe_state = np.concatenate(
            [
                self.safe_pos,  # Robot in safe area
                self.object_pos,  # Object position
                self.target_pos,  # Target position
            ]
        )

        # Test movement action ("right")
        obstacle_reward = self.env.reward(obstacle_state, action="right")
        safe_reward = self.env.reward(safe_state, action="right")

        # Robot in obstacle should get obstacle penalty (-10.0) in addition to distance reward
        # Robot in safe area should only get distance reward

        # The reward in obstacle state should be lower than safe state
        assert (
            obstacle_reward < safe_reward
        ), f"Obstacle state reward ({obstacle_reward}) should be < safe state reward ({safe_reward})"

        # Both should be negative (due to distance to target)
        assert obstacle_reward < 0
        assert safe_reward < 0

    def test_robot_obstacle_collision_penalty(self):
        """Test that robot collision with obstacles applies penalty."""
        # Create state with robot in obstacle
        obstacle_state = np.concatenate(
            [
                self.obstacle_pos,  # Robot in obstacle
                self.object_pos,  # Object position
                self.target_pos,  # Target position
            ]
        )

        # Create state with robot just outside obstacle
        near_obstacle_state = np.concatenate(
            [
                np.array([3.6, 3.6]),  # Robot just outside obstacle radius
                self.object_pos,  # Object position
                self.target_pos,  # Target position
            ]
        )

        # Test same action on both states
        obstacle_reward = self.env.reward(obstacle_state, action="up")
        near_obstacle_reward = self.env.reward(near_obstacle_state, action="up")

        # Robot in obstacle should get lower reward due to penalty
        assert (
            obstacle_reward < near_obstacle_reward
        ), f"Robot in obstacle reward ({obstacle_reward}) should be < near obstacle reward ({near_obstacle_reward})"

        # Calculate expected penalty
        distance_to_target_obstacle = np.linalg.norm(self.object_pos - self.target_pos)
        distance_to_target_near = np.linalg.norm(self.object_pos - self.target_pos)

        # Both should have same distance component, but obstacle state has additional penalty
        expected_obstacle_reward = -distance_to_target_obstacle + self.env.obstacle_penalty
        expected_near_reward = -distance_to_target_near

        # Verify the penalty is applied correctly
        assert (
            abs(obstacle_reward - expected_obstacle_reward) < 1e-6
        ), f"Expected obstacle reward {expected_obstacle_reward}, got {obstacle_reward}"
        assert (
            abs(near_obstacle_reward - expected_near_reward) < 1e-6
        ), f"Expected near obstacle reward {expected_near_reward}, got {near_obstacle_reward}"

    def test_object_obstacle_collision_penalty(self):
        """Test that object collision with obstacles applies penalty."""
        # Create state with object in obstacle
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

        # Object in obstacle should get lower reward due to penalty
        assert (
            object_obstacle_reward < object_near_obstacle_reward
        ), f"Object in obstacle reward ({object_obstacle_reward}) should be < near obstacle reward ({object_near_obstacle_reward})"

        # Calculate expected penalty
        distance_to_target_obstacle = np.linalg.norm(self.obstacle_pos - self.target_pos)
        distance_to_target_near = np.linalg.norm(np.array([3.6, 3.6]) - self.target_pos)

        # Both should have distance component, but obstacle state has additional penalty
        expected_obstacle_reward = -distance_to_target_obstacle + self.env.obstacle_penalty
        expected_near_reward = -distance_to_target_near

        # Verify the penalty is applied correctly
        assert (
            abs(object_obstacle_reward - expected_obstacle_reward) < 1e-6
        ), f"Expected object obstacle reward {expected_obstacle_reward}, got {object_obstacle_reward}"
        assert (
            abs(object_near_obstacle_reward - expected_near_reward) < 1e-6
        ), f"Expected object near obstacle reward {expected_near_reward}, got {object_near_obstacle_reward}"

    def test_both_robot_and_object_obstacle_collision(self):
        """Test that both robot and object collisions apply penalties."""
        # Create state with both robot and object in obstacles
        both_obstacle_state = np.concatenate(
            [
                self.obstacle_pos,  # Robot in obstacle
                np.array([7.0, 7.0]),  # Object in different obstacle
                self.target_pos,  # Target position
            ]
        )

        # Create state with both robot and object in safe areas
        both_safe_state = np.concatenate(
            [
                self.safe_pos,  # Robot in safe area
                np.array([2.0, 2.0]),  # Object in safe area
                self.target_pos,  # Target position
            ]
        )

        # Test same action on both states
        both_obstacle_reward = self.env.reward(both_obstacle_state, action="up")
        both_safe_reward = self.env.reward(both_safe_state, action="up")

        # Both obstacle state should get lower reward due to double penalty
        assert (
            both_obstacle_reward < both_safe_reward
        ), f"Both obstacle reward ({both_obstacle_reward}) should be < both safe reward ({both_safe_reward})"

        # Calculate expected penalties
        distance_to_target_obstacle = np.linalg.norm(np.array([7.0, 7.0]) - self.target_pos)
        distance_to_target_safe = np.linalg.norm(np.array([2.0, 2.0]) - self.target_pos)

        # Both obstacle state should have double penalty
        expected_obstacle_reward = -distance_to_target_obstacle + 2 * self.env.obstacle_penalty
        expected_safe_reward = -distance_to_target_safe

        # Verify the penalties are applied correctly
        assert (
            abs(both_obstacle_reward - expected_obstacle_reward) < 1e-6
        ), f"Expected both obstacle reward {expected_obstacle_reward}, got {both_obstacle_reward}"
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

    def test_state_transition_model(self):
        """Test state transition model creation and functionality."""
        # Create a test state
        test_state = np.array([1.0, 1.0, 2.0, 2.0, 9.0, 9.0])

        # Get transition model
        transition_model = self.env.state_transition_model(test_state, "right")

        assert isinstance(transition_model, PushStateTransition)
        assert transition_model.grid_size == self.env.grid_size
        assert transition_model.push_threshold == self.env.push_threshold
        assert transition_model.friction_coefficient == self.env.friction_coefficient

    def test_observation_model(self):
        """Test observation model creation and functionality."""
        # Create a test next state
        test_next_state = np.array([2.0, 1.0, 2.0, 2.0, 9.0, 9.0])

        # Get observation model
        obs_model = self.env.observation_model(test_next_state, "right")

        assert isinstance(obs_model, PushObservation)
        assert obs_model.observation_noise == self.env.observation_noise
        assert obs_model.grid_size == self.env.grid_size

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
        # Test state transition with known parameters
        state = np.array([5.0, 5.0, 6.0, 5.0, 9.0, 9.0])  # Robot to the left of object
        action = "right"
        grid_size = 10
        push_threshold = 2.0  # Increased threshold to ensure pushing
        friction_coefficient = 0.3

        transition = PushStateTransition(
            state=state,
            action=action,
            grid_size=grid_size,
            push_threshold=push_threshold,
            friction_coefficient=friction_coefficient,
        )

        next_state = transition.sample()[0]  # Get first element from list

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
        assert np.all(next_state < grid_size)

    def test_state_transition_no_push(self):
        """Test state transition without pushing when robot is too far."""
        # Test state transition when robot is too far to push
        state = np.array([1.0, 1.0, 8.0, 8.0, 9.0, 9.0])  # Robot far from object
        action = "right"
        grid_size = 10
        push_threshold = 1.0
        friction_coefficient = 0.3

        transition = PushStateTransition(
            state=state,
            action=action,
            grid_size=grid_size,
            push_threshold=push_threshold,
            friction_coefficient=friction_coefficient,
        )

        next_state = transition.sample()[0]  # Get first element from list

        # Verify robot moved
        assert next_state[0] > state[0]

        # Verify object didn't move (too far from robot)
        assert np.array_equal(next_state[2:4], state[2:4])

    def test_observation_model_functionality(self):
        """Test observation model functionality with noise."""
        # Test observation model
        state = np.array([5.0, 5.0, 4.0, 5.0, 9.0, 9.0])
        action = "right"
        observation_noise = 0.1
        grid_size = 10

        observation_model = PushObservation(
            next_state=state,
            action=action,
            observation_noise=observation_noise,
            grid_size=grid_size,
        )

        observation = observation_model.sample()[0]  # Get first element from list

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

    def test_observation_model_empty_observation_error(self):
        """Test that observation model properly handles invalid empty observations.

        Purpose: Validates that the observation probability method handles invalid inputs gracefully

        Given: A valid PushObservation model and an empty observation array
        When: The probability method is called with an empty observation
        Then: A descriptive error should be raised explaining the invalid observation format

        Test type: unit
        """
        # Create observation model
        state = np.array([5.0, 5.0, 4.0, 5.0, 9.0, 9.0])
        obs_model = PushObservation(
            next_state=state,
            action="right",
            observation_noise=0.1,
            grid_size=10,
        )

        # Test with empty observation (this is the error case we're debugging)
        empty_observation = np.array([])

        with pytest.raises(ValueError) as exc_info:
            obs_model.probability([empty_observation])  # Now expects list

        # The error should mention expected array format
        assert "Expected non-empty numpy array observation" in str(exc_info.value)

    def test_observation_model_invalid_observation_shapes(self):
        """Test observation model with various invalid observation shapes.

        Purpose: Validates that observation model handles all invalid observation shapes properly

        Given: A valid PushObservation model and observations with incorrect shapes
        When: The probability method is called with malformed observations
        Then: Appropriate errors should be raised for each invalid shape

        Test type: unit
        """
        # Create observation model
        state = np.array([5.0, 5.0, 4.0, 5.0, 9.0, 9.0])
        obs_model = PushObservation(
            next_state=state,
            action="right",
            observation_noise=0.1,
            grid_size=10,
        )

        # Test with various invalid observation shapes
        invalid_observations = [
            np.array([]),  # Empty array (shape (0,))
            np.array([1.0]),  # Too short (shape (1,))
            np.array([1.0, 2.0, 3.0]),  # Too short (shape (3,))
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # Too short (shape (5,))
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),  # Too long (shape (7,))
        ]

        for i, invalid_obs in enumerate(invalid_observations):
            with pytest.raises(ValueError) as exc_info:
                obs_model.probability([invalid_obs])  # Now expects list

            print(
                f"Invalid observation {i} shape {invalid_obs.shape} correctly raised: {type(exc_info.value).__name__}"
            )

    def test_observation_never_empty_from_sample(self):
        """Test that observation model never produces empty observations from sample method.

        Purpose: Validates that the sample method always produces correctly shaped observations

        Given: A valid PushObservation model
        When: Multiple observations are sampled
        Then: All observations should have shape (6,) and valid content

        Test type: unit
        """
        # Create observation model
        state = np.array([5.0, 5.0, 4.0, 5.0, 9.0, 9.0])
        obs_model = PushObservation(
            next_state=state,
            action="right",
            observation_noise=0.1,
            grid_size=10,
        )

        # Sample many observations to check consistency
        for _ in range(20):
            observations = obs_model.sample(n_samples=5)

            assert len(observations) == 5, "Should return requested number of observations"

            for obs in observations:
                assert isinstance(obs, np.ndarray), "Each observation should be numpy array"
                assert obs.shape == (6,), f"Expected shape (6,), got {obs.shape}"
                assert len(obs) > 0, "Observation should not be empty"
                assert np.all(np.isfinite(obs)), "All observation values should be finite"

                # Test that this observation can be used in probability calculation
                try:
                    probs = obs_model.probability([obs])  # Now expects list
                    assert len(probs) == 1, "Should return one probability"
                    prob = probs[0]
                    assert np.isfinite(prob), "Probability should be finite"
                    assert prob >= 0.0, "Probability should be non-negative"
                except Exception as e:
                    pytest.fail(
                        f"Valid observation {obs} with shape {obs.shape} failed probability calculation: {e}"
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

                    # Verify observation can be used in probability calculation
                    obs_model = self.env.observation_model(next_state, action)
                    try:
                        probs = obs_model.probability([observation])  # Now expects list
                        assert len(probs) == 1, "Should return one probability"
                        prob = probs[0]
                        assert np.isfinite(prob), "Probability should be finite"
                        assert prob >= 0.0, "Probability should be non-negative"
                    except Exception as e:
                        pytest.fail(
                            f"sample_next_step produced invalid observation {observation} with shape {observation.shape}: {e}"
                        )
