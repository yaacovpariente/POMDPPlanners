"""Tests for Light Dark POMDP reward models.

This module tests the reward models from light_dark_reward_models.py, focusing on:
- Base reward model functionality
- Continuous light dark reward model with obstacles
- Dangerous states reward model with high variance
- Decaying hit probability reward model
"""

import pytest
import numpy as np
import random

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)

from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models import (
    BaseLightDarkRewardModel,
    ContinuousLightDarkRewardModel,
    ContinuousLDDangerousStatesRewardModel,
    ContinuousLightDarkDecayingHitProbabilityRewardModel
)


class TestBaseLightDarkRewardModel:
    """Test cases for base reward model."""
    
    def test_abstract_class_cannot_be_instantiated(self):
        """Test that abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseLightDarkRewardModel()
    
    def test_input_validation(self):
        """Test input validation for state and action shapes."""
        # Create a concrete implementation for testing
        class ConcreteRewardModel(BaseLightDarkRewardModel):
            def _compute_reward(self, state: np.ndarray, action: np.ndarray) -> float:
                return 0.0
        
        model = ConcreteRewardModel()
        
        # Valid inputs
        valid_state = np.array([1.0, 2.0])
        valid_action = np.array([0.5, -0.3])
        
        # Should not raise error
        reward = model.compute_reward(valid_state, valid_action)
        assert reward == 0.0
        
        # Invalid state shape
        invalid_state = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="state must be a 2D vector"):
            model.compute_reward(invalid_state, valid_action)
        
        # Invalid action shape
        invalid_action = np.array([1.0])
        with pytest.raises(ValueError, match="action must be a 2D vector"):
            model.compute_reward(valid_state, invalid_action)


class TestContinuousLightDarkRewardModel:
    """Test cases for continuous light dark reward model."""
    
    def setup_method(self):
        """Set up test environment before each test method."""
        self.goal_state = np.array([8.0, 8.0])
        self.obstacles = np.array([[3.0, 3.0], [6.0, 6.0]]).T
        self.goal_state_radius = 1.0
        self.obstacle_radius = 1.0
        self.grid_size = 10
        self.obstacle_hit_probability = 0.3
        self.obstacle_reward = -10.0
        self.goal_reward = 100.0
        self.fuel_cost = 1.0
        
        self.model = ContinuousLightDarkRewardModel(
            goal_state=self.goal_state,
            obstacles=self.obstacles,
            goal_state_radius=self.goal_state_radius,
            obstacle_radius=self.obstacle_radius,
            grid_size=self.grid_size,
            obstacle_hit_probability=self.obstacle_hit_probability,
            obstacle_reward=self.obstacle_reward,
            goal_reward=self.goal_reward,
            fuel_cost=self.fuel_cost
        )
    
    def test_goal_state_reward(self):
        """Test reward when next state is in goal region."""
        # State that will reach goal with action
        state = np.array([7.5, 7.5])
        action = np.array([0.6, 0.6])  # Will reach goal
        
        reward = self.model.compute_reward(state, action)
        
        # Should get goal reward minus fuel cost and distance penalty
        expected_reward = -self.fuel_cost + self.goal_reward - np.linalg.norm(
            state + action - self.goal_state
        )
        
        assert abs(reward - expected_reward) < 1e-6
    
    def test_obstacle_collision_reward(self):
        """Test reward when next state collides with obstacle."""
        # State that will hit obstacle with action
        state = np.array([2.5, 2.5])
        action = np.array([0.6, 0.6])  # Will hit obstacle
        
        # Since obstacle reward is probabilistic, run multiple times
        rewards = []
        for _ in range(100):
            reward = self.model.compute_reward(state, action)
            rewards.append(reward)
        
        # Should get obstacle reward with some probability
        obstacle_rewards = [r for r in rewards if r < -5.0]  # Obstacle penalty applied
        no_obstacle_rewards = [r for r in rewards if r > -5.0]  # No obstacle penalty
        
        # At least obstacle penalties should occur
        assert len(obstacle_rewards) > 0
        
        # Check if no-obstacle case occurs (depends on probability)
        if len(no_obstacle_rewards) > 0:
            # Both cases occur
            assert True
        else:
            # Only obstacle penalties occur (valid for high probability)
            assert len(obstacle_rewards) == len(rewards)
    
    def test_out_of_grid_reward(self):
        """Test reward when next state is out of grid."""
        # State that will go out of grid with action
        state = np.array([9.5, 5.0])
        action = np.array([1.0, 0.0])  # Will go out of grid
        
        reward = self.model.compute_reward(state, action)
        
        # Should get obstacle reward (out of grid penalty) minus fuel cost and distance
        expected_reward = -self.fuel_cost + self.obstacle_reward - np.linalg.norm(
            state + action - self.goal_state
        )
        
        assert abs(reward - expected_reward) < 1e-6
    
    def test_normal_movement_reward(self):
        """Test reward for normal movement without special conditions."""
        # State that will move normally
        state = np.array([4.0, 4.0])
        action = np.array([0.5, 0.5])
        
        reward = self.model.compute_reward(state, action)
        
        # Should only get fuel cost and distance penalty
        expected_reward = -self.fuel_cost - np.linalg.norm(
            state + action - self.goal_state
        )
        
        assert abs(reward - expected_reward) < 1e-6
    
    def test_obstacle_detection_logic(self):
        """Test that obstacle detection logic works correctly."""
        # Test positions near obstacles
        near_obstacle = np.array([3.5, 3.5])
        # Check if this position would be detected as in obstacle range
        distances = np.linalg.norm(near_obstacle.reshape(-1, 1) - self.obstacles, axis=0)
        is_near = np.any(distances <= self.obstacle_radius)
        assert is_near
        
        # Test positions far from obstacles
        far_from_obstacle = np.array([1.0, 1.0])
        distances = np.linalg.norm(far_from_obstacle.reshape(-1, 1) - self.obstacles, axis=0)
        is_far = np.any(distances <= self.obstacle_radius)
        assert not is_far
    
    def test_goal_detection_logic(self):
        """Test that goal detection logic works correctly."""
        # Test position in goal
        in_goal = np.array([8.0, 8.0])
        distance_to_goal = np.linalg.norm(in_goal - self.goal_state)
        is_in_goal = distance_to_goal <= self.goal_state_radius
        assert is_in_goal
        
        # Test position near goal but not in it
        near_goal = np.array([9.5, 9.5])
        distance_to_goal = np.linalg.norm(near_goal - self.goal_state)
        is_in_goal = distance_to_goal <= self.goal_state_radius
        assert not is_in_goal
    
    def test_grid_boundary_detection_logic(self):
        """Test that grid boundary detection logic works correctly."""
        # Test position inside grid
        inside_grid = np.array([5.0, 5.0])
        is_outside = np.any(inside_grid < 0) or np.any(inside_grid > self.grid_size)
        assert not is_outside
        
        # Test position outside grid
        outside_grid = np.array([10.5, 5.0])
        is_outside = np.any(outside_grid < 0) or np.any(outside_grid > self.grid_size)
        assert is_outside
        
        # Test position at boundary
        at_boundary = np.array([0.0, 0.0])
        is_outside = np.any(at_boundary < 0) or np.any(at_boundary > self.grid_size)
        assert not is_outside


class TestContinuousLDDangerousStatesRewardModel:
    """Test cases for dangerous states reward model."""
    
    def setup_method(self):
        """Set up test environment before each test method."""
        self.goal_state = np.array([8.0, 8.0])
        self.obstacles = np.array([[3.0, 3.0], [6.0, 6.0]]).T
        self.goal_state_radius = 1.0
        self.obstacle_radius = 1.0
        self.grid_size = 10
        self.obstacle_hit_probability = 0.3
        self.obstacle_reward = -10.0
        self.goal_reward = 100.0
        self.fuel_cost = 1.0
        
        self.model = ContinuousLDDangerousStatesRewardModel(
            goal_state=self.goal_state,
            obstacles=self.obstacles,
            goal_state_radius=self.goal_state_radius,
            obstacle_radius=self.obstacle_radius,
            grid_size=self.grid_size,
            obstacle_hit_probability=self.obstacle_hit_probability,
            obstacle_reward=self.obstacle_reward,
            goal_reward=self.goal_reward,
            fuel_cost=self.fuel_cost
        )
    
    def test_obstacle_reward_high_variance(self):
        """Test that obstacle rewards have high variance (penalty or bonus)."""
        # State that will hit obstacle
        state = np.array([2.5, 2.5])
        action = np.array([0.6, 0.6])
        
        # Run multiple times to capture variance
        rewards = []
        for _ in range(100):
            reward = self.model.compute_reward(state, action)
            rewards.append(reward)
        
        # Should get both positive and negative obstacle rewards
        positive_obstacle_rewards = [r for r in rewards if r > -5.0]  # Bonus applied
        negative_obstacle_rewards = [r for r in rewards if r < -15.0]  # Penalty applied
        
        # Both cases should occur due to 50/50 probability
        assert len(positive_obstacle_rewards) > 0
        assert len(negative_obstacle_rewards) > 0
        
        # Calculate variance
        reward_variance = np.var(rewards)
        
        # Variance should be high due to random penalty/bonus
        assert reward_variance > 50.0  # High variance threshold
    
    def test_obstacle_reward_probability_distribution(self):
        """Test that obstacle rewards follow 50/50 probability distribution."""
        # State that will hit obstacle
        state = np.array([2.5, 2.5])
        action = np.array([0.6, 0.6])
        
        # Run many times to get accurate probability estimate
        rewards = []
        for _ in range(1000):
            reward = self.model.compute_reward(state, action)
            rewards.append(reward)
        
        # Count penalty vs bonus cases
        penalty_count = sum(1 for r in rewards if r < -15.0)
        bonus_count = sum(1 for r in rewards if r > -5.0)
        
        # Should be roughly 50/50 (within reasonable tolerance)
        penalty_ratio = penalty_count / len(rewards)
        bonus_ratio = bonus_count / len(rewards)
        
        assert 0.4 < penalty_ratio < 0.6, f"Penalty ratio {penalty_ratio} not close to 0.5"
        assert 0.4 < bonus_ratio < 0.6, f"Bonus ratio {bonus_ratio} not close to 0.5"
    
    def test_inheritance_from_base(self):
        """Test that dangerous states model inherits correctly from base."""
        # Should have all the same methods as base class
        assert hasattr(self.model, 'compute_reward')
        assert hasattr(self.model, '_compute_reward')
        assert hasattr(self.model, '_obstacle_reward')
        
        # Should override _obstacle_reward method
        base_method = ContinuousLightDarkRewardModel._obstacle_reward
        derived_method = self.model._obstacle_reward
        
        assert base_method != derived_method


class TestContinuousLightDarkDecayingHitProbabilityRewardModel:
    """Test cases for decaying hit probability reward model."""
    
    def setup_method(self):
        """Set up test environment before each test method."""
        self.goal_state = np.array([8.0, 8.0])
        self.obstacles = np.array([[3.0, 3.0], [6.0, 6.0]]).T
        self.goal_state_radius = 1.0
        self.obstacle_radius = 1.0
        self.grid_size = 10
        self.obstacle_hit_probability = 0.3
        self.obstacle_reward = -10.0
        self.goal_reward = 100.0
        self.fuel_cost = 1.0
        self.penalty_decay = 2.0
        
        self.model = ContinuousLightDarkDecayingHitProbabilityRewardModel(
            goal_state=self.goal_state,
            obstacles=self.obstacles,
            goal_state_radius=self.goal_state_radius,
            obstacle_radius=self.obstacle_radius,
            grid_size=self.grid_size,
            obstacle_hit_probability=self.obstacle_hit_probability,
            obstacle_reward=self.obstacle_reward,
            goal_reward=self.goal_reward,
            fuel_cost=self.fuel_cost,
            penalty_decay=self.penalty_decay
        )
    
    def test_decaying_obstacle_probability(self):
        """Test that obstacle hit probability decreases with distance."""
        # Test positions at different distances from obstacles
        # Use positions that are clearly at different distances from obstacles
        close_to_obstacle = np.array([3.1, 3.1])  # Very close to obstacle at (3,3)
        medium_distance = np.array([5.0, 5.0])    # Medium distance from obstacles
        far_from_obstacle = np.array([8.0, 8.0])  # Far from obstacles
        
        # Calculate theoretical probabilities based on exponential decay
        close_distance = np.min(np.linalg.norm(close_to_obstacle.reshape(-1, 1) - self.obstacles, axis=0))
        medium_distance_val = np.min(np.linalg.norm(medium_distance.reshape(-1, 1) - self.obstacles, axis=0))
        far_distance = np.min(np.linalg.norm(far_from_obstacle.reshape(-1, 1) - self.obstacles, axis=0))
        
        # Print distances for debugging
        print(f"Distances: close={close_distance:.3f}, medium={medium_distance_val:.3f}, far={far_distance:.3f}")
        
        # Theoretical probabilities using exponential decay
        close_prob_theoretical = np.exp(-close_distance / self.penalty_decay)
        medium_prob_theoretical = np.exp(-medium_distance_val / self.penalty_decay)
        far_prob_theoretical = np.exp(-far_distance / self.penalty_decay)
        
        # Verify theoretical ordering: closer = higher probability
        assert close_prob_theoretical > medium_prob_theoretical, \
            f"Close theoretical probability {close_prob_theoretical:.3f} should be > medium {medium_prob_theoretical:.3f}"
        assert medium_prob_theoretical > far_prob_theoretical, \
            f"Medium theoretical probability {medium_prob_theoretical:.3f} should be > far {far_prob_theoretical:.3f}"
        
        # Also verify that the model's empirical probabilities are reasonable
        # Run multiple times to get probability estimates
        close_rewards = []
        medium_rewards = []
        far_rewards = []
        
        for _ in range(100):
            close_rewards.append(self.model._obstacle_reward(close_to_obstacle))
            medium_rewards.append(self.model._obstacle_reward(medium_distance))
            far_rewards.append(self.model._obstacle_reward(far_from_obstacle))
        
        # Count penalty applications
        close_penalties = sum(1 for r in close_rewards if r < 0)
        medium_penalties = sum(1 for r in medium_rewards if r < 0)
        far_penalties = sum(1 for r in far_rewards if r < 0)
        
        # Empirical probabilities
        close_prob_empirical = close_penalties / len(close_rewards)
        medium_prob_empirical = medium_penalties / len(medium_rewards)
        far_prob_empirical = far_penalties / len(far_rewards)
        
        # Verify empirical probabilities are close to theoretical (within reasonable tolerance)
        assert abs(close_prob_empirical - close_prob_theoretical) < 0.2, \
            f"Close empirical {close_prob_empirical:.3f} should be close to theoretical {close_prob_theoretical:.3f}"
        assert abs(medium_prob_empirical - medium_prob_theoretical) < 0.2, \
            f"Medium empirical {medium_prob_empirical:.3f} should be close to theoretical {medium_prob_theoretical:.3f}"
        assert abs(far_prob_empirical - far_prob_theoretical) < 0.2, \
            f"Far empirical {far_prob_empirical:.3f} should be close to theoretical {far_prob_theoretical:.3f}"
    
    def test_penalty_decay_parameter_effect(self):
        """Test that penalty_decay parameter affects probability decay rate."""
        # Create model with different decay rates
        fast_decay_model = ContinuousLightDarkDecayingHitProbabilityRewardModel(
            goal_state=self.goal_state,
            obstacles=self.obstacles,
            goal_state_radius=self.goal_state_radius,
            obstacle_radius=self.obstacle_radius,
            grid_size=self.grid_size,
            obstacle_hit_probability=self.obstacle_hit_probability,
            obstacle_reward=self.obstacle_reward,
            goal_reward=self.goal_reward,
            fuel_cost=self.fuel_cost,
            penalty_decay=1.0  # Fast decay
        )
        
        slow_decay_model = ContinuousLightDarkDecayingHitProbabilityRewardModel(
            goal_state=self.goal_state,
            obstacles=self.obstacles,
            goal_state_radius=self.goal_state_radius,
            obstacle_radius=self.obstacle_radius,
            grid_size=self.grid_size,
            obstacle_hit_probability=self.obstacle_hit_probability,
            obstacle_reward=self.obstacle_reward,
            goal_reward=self.goal_reward,
            fuel_cost=self.fuel_cost,
            penalty_decay=5.0  # Slow decay
        )
        
        # Test at medium distance
        test_pos = np.array([4.0, 4.0])
        
        # Run multiple times to get probability estimates
        fast_decay_rewards = []
        slow_decay_rewards = []
        
        for _ in range(100):
            fast_decay_rewards.append(fast_decay_model._obstacle_reward(test_pos))
            slow_decay_rewards.append(slow_decay_model._obstacle_reward(test_pos))
        
        # Count penalty applications
        fast_decay_penalties = sum(1 for r in fast_decay_rewards if r < 0)
        slow_decay_penalties = sum(1 for r in slow_decay_rewards if r < 0)
        
        # Fast decay should have fewer penalties at medium distance
        fast_decay_prob = fast_decay_penalties / len(fast_decay_rewards)
        slow_decay_prob = slow_decay_penalties / len(slow_decay_rewards)
        
        assert fast_decay_prob < slow_decay_prob, \
            f"Fast decay probability {fast_decay_prob} should be < slow decay {slow_decay_prob}"
    
    def test_obstacle_distance_calculation(self):
        """Test that obstacle distance calculation works correctly."""
        # Test position
        test_pos = np.array([4.0, 4.0])
        
        # Calculate expected minimum distance to obstacles
        distances = []
        for i in range(self.obstacles.shape[1]):
            obstacle_pos = self.obstacles[:, i]
            distance = np.linalg.norm(test_pos - obstacle_pos)
            distances.append(distance)
        
        expected_min_distance = min(distances)
        
        # The model should use this minimum distance for probability calculation
        # Run multiple times to verify the pattern
        penalties = []
        for _ in range(100):
            reward = self.model._obstacle_reward(test_pos)
            if reward < 0:
                penalties.append(1)
            else:
                penalties.append(0)
        
        # Calculate empirical probability
        empirical_prob = np.mean(penalties)
        
        # Theoretical probability based on exponential decay
        theoretical_prob = np.exp(-expected_min_distance / self.penalty_decay)
        
        # Should be reasonably close (within 20% due to randomness)
        assert abs(empirical_prob - theoretical_prob) < 0.2, \
            f"Empirical prob {empirical_prob} should be close to theoretical {theoretical_prob}"
    
    def test_reward_structure(self):
        """Test that reward structure includes all expected components."""
        # Test normal movement
        state = np.array([4.0, 4.0])
        action = np.array([0.5, 0.5])
        
        reward = self.model.compute_reward(state, action)
        
        # Should include: fuel cost + distance penalty + potential obstacle reward
        base_reward = -self.fuel_cost - np.linalg.norm(state + action - self.goal_state)
        
        # The obstacle reward component is probabilistic, so we can't test exact value
        # But we can verify the structure - reward should be <= base_reward (obstacle penalty makes it worse)
        assert reward <= base_reward, f"Reward {reward} should be <= base_reward {base_reward}"
        
        # Test goal state
        goal_state = np.array([7.5, 7.5])
        goal_action = np.array([0.6, 0.6])
        
        goal_reward = self.model.compute_reward(goal_state, goal_action)
        
        # Should get goal reward
        assert goal_reward > 0  # Goal reward should be positive
    
    def test_inheritance_and_overrides(self):
        """Test that decaying model inherits correctly and overrides methods."""
        # Should have all base methods
        assert hasattr(self.model, 'compute_reward')
        assert hasattr(self.model, '_compute_reward')
        assert hasattr(self.model, '_obstacle_reward')
        
        # Should have penalty_decay attribute
        assert hasattr(self.model, 'penalty_decay')
        assert self.model.penalty_decay == self.penalty_decay
        
        # Should override _obstacle_reward method
        base_method = ContinuousLightDarkRewardModel._obstacle_reward
        derived_method = self.model._obstacle_reward
        
        assert base_method != derived_method
