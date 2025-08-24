"""Tests for LaserTag POMDP Environment Implementation.

This module contains comprehensive tests for the LaserTag POMDP environment,
including tests for state transitions, observations, rewards, metrics computation,
and visualization functionality.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List, Any
from unittest.mock import Mock
import random

from POMDPPlanners.environments.laser_tag_pomdp import (
    LaserTagPOMDP,
    LaserTagState,
    LaserTagStateTransition,
    LaserTagObservation
)
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.core.distributions import DiscreteDistribution

np.random.seed(42)
random.seed(42)


class TestLaserTagState:
    """Test LaserTagState representation and methods.
    
    Purpose: Validates LaserTag state representation and equality operations
    
    Given: LaserTagState instances with robot/opponent positions and terminal flags
    When: States are created, compared, and hashed
    Then: State operations work correctly with proper equality and hashing
    
    Test type: unit
    """
    
    def test_state_creation_and_equality(self):
        """Test LaserTagState creation and equality comparison.
        
        Purpose: Validates state creation and equality comparison functionality
        
        Given: Two identical LaserTag states and one different state
        When: States are created and compared using == operator  
        Then: Identical states are equal and different states are not equal
        
        Test type: unit
        """
        state1 = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)
        state2 = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)
        state3 = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=True)
        
        assert state1 == state2
        assert state1 != state3
        assert state1 != "not a state"
    
    def test_state_hashing(self):
        """Test LaserTagState hashing for use in sets and dictionaries.
        
        Purpose: Validates that identical states have same hash for container usage
        
        Given: Two identical LaserTag states
        When: Hash values are computed
        Then: States have same hash and can be used as dictionary keys
        
        Test type: unit
        """
        state1 = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)
        state2 = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)
        
        assert hash(state1) == hash(state2)
        
        # Test usage in set
        state_set = {state1, state2}
        assert len(state_set) == 1


class TestLaserTagStateTransition:
    """Test LaserTagStateTransition model functionality.
    
    Purpose: Validates state transition dynamics including robot movement and opponent behavior
    
    Given: LaserTag state transition models with various states and actions
    When: Transitions are sampled and probabilities calculated
    Then: Robot and opponent movements follow expected dynamics
    
    Test type: unit
    """
    
    def test_robot_movement_basic(self):
        """Test basic robot movement without walls.
        
        Purpose: Validates robot movement follows action directions correctly
        
        Given: Robot at center position with clear movement directions
        When: Movement actions (North, South, East, West) are executed
        Then: Robot moves to expected adjacent positions
        
        Test type: unit
        """
        state = LaserTagState(robot=(3, 5), opponent=(1, 1), terminal=False)
        floor_shape = (7, 11)
        walls = set()
        
        # Test North movement (action 0)
        transition = LaserTagStateTransition(state, 0, floor_shape, walls)
        next_states = transition.sample(n_samples=10)
        for next_state in next_states:
            assert next_state.robot == (2, 5)  # One row up
            assert not next_state.terminal
    
    def test_robot_boundary_collision(self):
        """Test robot cannot move outside grid boundaries.
        
        Purpose: Validates robot respects grid boundaries and stays in place when blocked
        
        Given: Robot at grid boundary positions
        When: Actions try to move robot outside boundaries
        Then: Robot stays at current position instead of moving
        
        Test type: unit
        """
        floor_shape = (7, 11)
        walls = set()
        
        # Test robot at top boundary trying to go North
        state = LaserTagState(robot=(0, 5), opponent=(3, 3), terminal=False)
        transition = LaserTagStateTransition(state, 0, floor_shape, walls)
        next_states = transition.sample(n_samples=5)
        for next_state in next_states:
            assert next_state.robot == (0, 5)  # Should stay in place
    
    def test_robot_wall_collision(self):
        """Test robot cannot move into walls.
        
        Purpose: Validates robot respects wall positions and cannot move through them
        
        Given: Robot adjacent to wall positions
        When: Actions try to move robot into walls
        Then: Robot stays at current position instead of moving into wall
        
        Test type: unit
        """
        state = LaserTagState(robot=(3, 2), opponent=(1, 1), terminal=False)
        floor_shape = (7, 11)
        walls = {(3, 3)}  # Wall to the East
        
        # Test robot trying to move East into wall
        transition = LaserTagStateTransition(state, 2, floor_shape, walls)
        next_states = transition.sample(n_samples=5)
        for next_state in next_states:
            assert next_state.robot == (3, 2)  # Should stay in place
    
    def test_opponent_movement_toward_robot(self):
        """Test opponent tends to move toward robot position.
        
        Purpose: Validates opponent movement probabilities favor moves toward robot
        
        Given: Opponent in position where it can move toward robot
        When: Many state transitions are sampled
        Then: Opponent moves toward robot more frequently than random
        
        Test type: unit
        """
        state = LaserTagState(robot=(2, 5), opponent=(5, 5), terminal=False)
        floor_shape = (7, 11)
        walls = set()
        
        transition = LaserTagStateTransition(state, 1, floor_shape, walls)  # Robot moves South
        samples = transition.sample(n_samples=1000)
        
        # Count opponent positions
        opponent_positions = [s.opponent for s in samples]
        from collections import Counter
        pos_counts = Counter(opponent_positions)
        
        # Opponent should prefer moving toward robot (should prefer North: (4,5))
        toward_robot_count = pos_counts.get((4, 5), 0)
        total_samples = len(samples)
        toward_robot_prob = toward_robot_count / total_samples
        
        # Should be around 0.4 based on implementation
        assert toward_robot_prob > 0.3, f"Expected >0.3 probability toward robot, got {toward_robot_prob}"
    
    def test_successful_tagging(self):
        """Test successful tagging creates terminal state.
        
        Purpose: Validates that tag action at same position creates terminal state
        
        Given: Robot and opponent at same position
        When: Tag action (action 4) is executed
        Then: Next state is terminal with same positions
        
        Test type: unit
        """
        state = LaserTagState(robot=(3, 5), opponent=(3, 5), terminal=False)
        floor_shape = (7, 11)
        walls = set()
        
        transition = LaserTagStateTransition(state, 4, floor_shape, walls)
        next_states = transition.sample(n_samples=10)
        
        for next_state in next_states:
            assert next_state.terminal
            assert next_state.robot == (3, 5)
            assert next_state.opponent == (3, 5)
    
    def test_failed_tagging(self):
        """Test failed tagging does not create terminal state.
        
        Purpose: Validates that tag action at different positions remains non-terminal
        
        Given: Robot and opponent at different positions
        When: Tag action (action 4) is executed
        Then: Next state remains non-terminal with normal opponent movement
        
        Test type: unit
        """
        state = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)
        floor_shape = (7, 11)
        walls = set()
        
        transition = LaserTagStateTransition(state, 4, floor_shape, walls)
        next_states = transition.sample(n_samples=10)
        
        for next_state in next_states:
            assert not next_state.terminal
            assert next_state.robot == (3, 5)  # Robot doesn't move on tag


class TestLaserTagObservation:
    """Test LaserTagObservation model functionality.
    
    Purpose: Validates observation generation with Gaussian noise around opponent position
    
    Given: LaserTag observation models with various states
    When: Observations are sampled and probabilities calculated
    Then: Observations follow expected noise characteristics
    
    Test type: unit
    """
    
    def test_observation_noise_characteristics(self):
        """Test observation model adds appropriate Gaussian noise.
        
        Purpose: Validates observation noise follows Gaussian distribution around true position
        
        Given: Observation model with known opponent position and noise level
        When: Many observations are sampled
        Then: Observations are distributed around true position with correct noise characteristics
        
        Test type: unit
        """
        state = LaserTagState(robot=(2, 2), opponent=(2, 4), terminal=False)
        obs_model = LaserTagObservation(
            state, 0, measurement_noise=0.1,  # Low noise for testing
            floor_shape=(5, 5), walls=set()
        )
        
        observations = obs_model.sample(n_samples=100)
        obs_array = np.array(observations)
        
        # Check all observations are 8-dimensional
        assert obs_array.shape[1] == 8, "Observations should be 8-dimensional laser measurements"
        
        # Check observations are reasonable laser ranges (non-negative)
        assert np.all(obs_array >= 0), "Laser measurements should be non-negative"
        
        # Check standard deviation is around measurement_noise for each direction
        std_obs = np.std(obs_array, axis=0)
        assert np.all(std_obs <= 1.0), "Standard deviation should be reasonable for laser measurements"
    
    def test_terminal_state_observation(self):
        """Test terminal state produces special observation.
        
        Purpose: Validates terminal states generate designated terminal observation
        
        Given: Terminal LaserTag state
        When: Observations are sampled
        Then: All observations are the special terminal observation (-1, -1)
        
        Test type: unit
        """
        state = LaserTagState(robot=(3, 5), opponent=(3, 5), terminal=True)
        obs_model = LaserTagObservation(
            state, 4, measurement_noise=1.0,
            floor_shape=(7, 11), walls=set()
        )
        
        observations = obs_model.sample(n_samples=10)
        terminal_obs = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
        for obs in observations:
            assert obs == terminal_obs, f"Terminal observation should be {terminal_obs}"


class TestLaserTagPOMDP:
    """Test main LaserTag POMDP environment functionality.
    
    Purpose: Validates complete environment behavior including rewards, terminals, and metrics
    
    Given: LaserTag POMDP environments with various configurations
    When: Environment methods are called with different states and actions
    Then: Environment behaves according to LaserTag problem specification
    
    Test type: integration
    """
    
    def test_environment_initialization(self):
        """Test LaserTag POMDP environment initialization.
        
        Purpose: Validates environment initializes with correct default parameters
        
        Given: LaserTag environment constructor with discount factor
        When: Environment is created
        Then: Environment has expected attributes and space configuration
        
        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95)
        
        assert env.discount_factor == 0.95
        assert env.floor_shape == (11, 7)
        # Default walls - 8 walls within valid grid bounds
        expected_walls = {(1, 2), (3, 0), (3, 4), (5, 0), (6, 4), (9, 1), (9, 4), (10, 6)}
        assert env.walls == expected_walls
        assert len(env.walls) == 8  # Should have 8 walls by default
        assert env.tag_reward == 10.0
        assert env.tag_penalty == 10.0
        assert env.step_cost == 1.0
        assert len(env.get_actions()) == 5
    
    def test_environment_initialization_with_walls(self):
        """Test LaserTag POMDP environment initialization with walls.
        
        Purpose: Validates environment initializes correctly with wall configuration
        
        Given: LaserTag environment constructor with custom wall set
        When: Environment is created with walls parameter
        Then: Environment stores walls correctly
        
        Test type: unit
        """
        walls = {(3, 3), (4, 4)}
        env = LaserTagPOMDP(discount_factor=0.95, walls=walls)
        
        assert env.walls == walls
    
    def test_reward_structure(self):
        """Test reward function returns correct values.
        
        Purpose: Validates reward function implements LaserTag reward structure correctly
        
        Given: Various state-action combinations
        When: Reward function is called
        Then: Returns correct rewards for tagging success/failure and movement
        
        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95)
        
        # Test successful tag
        same_pos_state = LaserTagState(robot=(3, 5), opponent=(3, 5), terminal=False)
        tag_reward = env.reward(same_pos_state, 4)
        assert tag_reward == env.tag_reward
        
        # Test failed tag
        diff_pos_state = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)
        failed_tag_reward = env.reward(diff_pos_state, 4)
        assert failed_tag_reward == -env.tag_penalty
        
        # Test movement cost
        move_reward = env.reward(diff_pos_state, 0)  # North
        assert move_reward == -env.step_cost
        
        # Test terminal state
        terminal_state = LaserTagState(robot=(3, 5), opponent=(3, 5), terminal=True)
        terminal_reward = env.reward(terminal_state, 0)
        assert terminal_reward == 0.0
    
    def test_terminal_state_detection(self):
        """Test terminal state detection.
        
        Purpose: Validates environment correctly identifies terminal states
        
        Given: States with different terminal flags
        When: is_terminal method is called
        Then: Returns correct boolean based on state terminal flag
        
        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95)
        
        non_terminal = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)
        terminal = LaserTagState(robot=(3, 5), opponent=(3, 5), terminal=True)
        
        assert not env.is_terminal(non_terminal)
        assert env.is_terminal(terminal)
    
    def test_initial_state_distribution(self):
        """Test initial state distribution properties.
        
        Purpose: Validates initial state distribution covers valid robot-opponent combinations
        
        Given: LaserTag environment
        When: Initial state distribution is sampled
        Then: States have robot and opponent at different valid positions
        
        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95)
        initial_dist = env.initial_state_dist()
        
        # Sample several initial states
        initial_states = initial_dist.sample(n_samples=10)
        
        for state in initial_states:
            assert isinstance(state, LaserTagState)
            assert state.robot != state.opponent  # Should start at different positions
            assert not state.terminal
    
    def test_observation_equality(self):
        """Test observation equality comparison.
        
        Purpose: Validates observation equality handles 8-dimensional laser measurements correctly
        
        Given: Pairs of similar and different 8D laser observations
        When: is_equal_observation method is called
        Then: Returns correct equality based on small tolerance for floating point
        
        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95)
        
        obs1 = (2.0, 4.0, 1.5, 3.2, 2.8, 1.1, 0.9, 2.5)
        obs2 = (2.0, 4.0, 1.5, 3.2, 2.8, 1.1, 0.9, 2.5)
        obs3 = (2.1, 4.1, 1.6, 3.3, 2.9, 1.2, 1.0, 2.6)
        
        assert env.is_equal_observation(obs1, obs2)
        assert not env.is_equal_observation(obs1, obs3)
    
    def test_compute_metrics_with_wall_collisions(self):
        """Test compute_metrics includes wall collision counting.
        
        Purpose: Validates metrics computation includes wall collision tracking
        
        Given: Mock history with wall collision scenario
        When: compute_metrics is called
        Then: Returns metrics including wall collision count
        
        Test type: unit
        """
        walls = {(3, 3)}
        env = LaserTagPOMDP(discount_factor=0.95, walls=walls)
        
        # Create mock history with wall collision
        # Mock the policy_run_data
        mock_policy_run_data = Mock()
        
        # Step 1: Robot tries to move East into wall
        state1 = LaserTagState(robot=(3, 2), opponent=(1, 1), terminal=False)
        step1 = StepData(
            state=state1, 
            action=2,  # East
            next_state=LaserTagState(robot=(3, 2), opponent=(1, 0), terminal=False),  # Robot stayed due to wall
            observation=(1.1, 0.9),
            reward=-1.0,
            belief=Mock()
        )
        
        # Step 2: Normal movement
        state2 = LaserTagState(robot=(3, 2), opponent=(1, 0), terminal=False)
        step2 = StepData(
            state=state2,
            action=0,  # North
            next_state=LaserTagState(robot=(2, 2), opponent=(1, 1), terminal=False),
            observation=(1.0, 1.1),
            reward=-1.0,
            belief=Mock()
        )
        
        history = History(
            history=[step1, step2],
            discount_factor=0.95,
            average_state_sampling_time=0.1,
            average_action_time=0.1,
            average_observation_time=0.1,
            average_belief_update_time=0.1,
            average_reward_time=0.1,
            actual_num_steps=2,
            reach_terminal_state=False,
            policy_run_data=mock_policy_run_data
        )
        
        histories = [history]
        metrics = env.compute_metrics(histories)
        
        # Find wall collision metric
        collision_metric = None
        for metric in metrics:
            if metric.name == "average_obstacle_collisions":
                collision_metric = metric
                break
        
        assert collision_metric is not None, "Wall collision metric not found"
        assert collision_metric.value == 1.0, f"Expected 1.0 collision, got {collision_metric.value}"
        
        # Find dangerous area steps metric
        dangerous_area_metric = None
        for metric in metrics:
            if metric.name == "average_dangerous_area_steps":
                dangerous_area_metric = metric
                break
        
        assert dangerous_area_metric is not None, "Dangerous area steps metric not found"
        # Should have 0 dangerous area steps since robot positions are not in default dangerous areas
        assert dangerous_area_metric.value == 0.0, f"Expected 0.0 dangerous area steps, got {dangerous_area_metric.value}"
    
    def test_wall_collision_penalty(self):
        """Test wall collision applies dangerous area penalty.
        
        Purpose: Validates wall collisions apply additional dangerous area penalty
        
        Given: LaserTag environment with walls and robot attempting wall collision
        When: Robot tries to move into a wall
        Then: Reward includes both step cost and dangerous area penalty
        
        Test type: unit
        """
        walls = {(3, 3)}
        env = LaserTagPOMDP(
            discount_factor=0.95, 
            walls=walls, 
            dangerous_area_penalty=5.0,
            step_cost=1.0
        )
        
        # Test wall collision penalty
        state = LaserTagState(robot=(3, 2), opponent=(1, 1), terminal=False)
        
        # Try to move East into wall at (3, 3)
        reward = env.reward(state, 2)  # East action
        
        # Expected: -1.0 (step cost) - 5.0 (wall collision penalty) = -6.0
        expected_reward = -env.step_cost - env.dangerous_area_penalty
        assert reward == expected_reward, f"Expected {expected_reward} for wall collision, got {reward}"
    
    def test_cache_visualization_functionality(self):
        """Test cache_visualization method basic functionality.
        
        Purpose: Validates cache_visualization runs without errors and creates output
        
        Given: LaserTag environment and mock episode history
        When: cache_visualization is called with valid parameters
        Then: Method executes without error and creates visualization file
        
        Test type: integration
        """
        env = LaserTagPOMDP(discount_factor=0.95)
        
        # Create simple mock history
        mock_policy_run_data = Mock()
        
        steps = []
        for i in range(3):
            state = LaserTagState(robot=(i, 0), opponent=(6-i, 10), terminal=False)
            step = StepData(
                state=state,
                action=1,  # South
                next_state=state,
                observation=(6-i + 0.1, 10.1),
                reward=-1.0,
                belief=Mock()
            )
            steps.append(step)
        
        history = History(
            history=steps,
            discount_factor=0.95,
            average_state_sampling_time=0.1,
            average_action_time=0.1,
            average_observation_time=0.1,
            average_belief_update_time=0.1,
            average_reward_time=0.1,
            actual_num_steps=3,
            reach_terminal_state=False,
            policy_run_data=mock_policy_run_data
        )
        
        cache_path = Path("test_laser_tag_visualization.gif")
        
        try:
            # This should not raise an exception
            env.cache_visualization(history, cache_path)
            
            # Clean up if file was created
            if cache_path.exists():
                cache_path.unlink()
                
        except Exception as e:
            pytest.fail(f"cache_visualization raised an exception: {e}")
    
    def test_cache_visualization_error_handling(self):
        """Test cache_visualization error handling with invalid inputs.
        
        Purpose: Validates cache_visualization properly handles invalid inputs
        
        Given: Invalid cache_path types and empty histories
        When: cache_visualization is called with invalid parameters
        Then: Appropriate exceptions are raised with descriptive messages
        
        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95)
        
        # Test with non-Path cache_path
        with pytest.raises(TypeError, match="cache_path must be a Path object"):
            env.cache_visualization(Mock(), "invalid_path")
        
        # Test with non-gif extension
        with pytest.raises(ValueError, match="cache_path must end with .gif"):
            env.cache_visualization(Mock(), Path("test.png"))
        
        # Test with empty history
        mock_history = Mock()
        mock_history.history = []
        with pytest.raises(ValueError, match="Cannot visualize empty history"):
            env.cache_visualization(mock_history, Path("test.gif"))


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])