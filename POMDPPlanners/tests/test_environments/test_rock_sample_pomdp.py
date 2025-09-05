"""Tests for RockSample POMDP environment.

This module tests the RockSample POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

import pytest
import numpy as np
import random
from pathlib import Path
from unittest.mock import patch
import tempfile

from POMDPPlanners.environments.rock_sample_pomdp import (
    RockSamplePOMDP,
    RockSampleState,
    RockSampleStateTransitionModel,
    RockSampleObservationModel,
    create_random_rock_sample
)
from POMDPPlanners.core.simulation import History, StepData

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


class TestRockSampleState:
    """Test RockSampleState data class."""
    
    def test_state_creation_valid(self):
        """Test valid state creation with robot position and rock states.
        
        Purpose: Validates that RockSampleState can be created with valid parameters
        
        Given: Valid robot position tuple and rock states tuple
        When: RockSampleState is instantiated
        Then: State is created successfully with correct attributes
        
        Test type: unit
        """
        robot_pos = (2, 3)
        rocks = (True, False, True)
        state = RockSampleState(robot_pos, rocks)
        
        assert state.robot_pos == robot_pos
        assert state.rocks == rocks
        assert isinstance(state.rocks, tuple)
    
    def test_state_immutable(self):
        """Test that state is immutable after creation.
        
        Purpose: Validates that RockSampleState is properly frozen/immutable
        
        Given: A created RockSampleState instance
        When: Attempting to modify state attributes
        Then: AttributeError is raised preventing modification
        
        Test type: unit
        """
        state = RockSampleState((1, 1), (True, False))
        
        with pytest.raises(AttributeError):
            state.robot_pos = (2, 2)
        
        with pytest.raises(AttributeError):
            state.rocks = (False, True)
    
    def test_state_validation_invalid_robot_pos(self):
        """Test state validation with invalid robot position.
        
        Purpose: Validates proper error handling for malformed robot positions
        
        Given: Invalid robot position (not tuple of two integers)
        When: RockSampleState is instantiated
        Then: ValueError is raised with descriptive message
        
        Test type: unit
        """
        with pytest.raises(ValueError, match="robot_pos must be a tuple of two integers"):
            RockSampleState([1, 2], (True,))
        
        with pytest.raises(ValueError, match="robot_pos must be a tuple of two integers"):
            RockSampleState((1, 2, 3), (True,))
    
    def test_state_validation_invalid_rocks(self):
        """Test state validation with invalid rock states.
        
        Purpose: Validates proper error handling for malformed rock states
        
        Given: Invalid rock states (not tuple)
        When: RockSampleState is instantiated
        Then: ValueError is raised with descriptive message
        
        Test type: unit
        """
        with pytest.raises(ValueError, match="rocks must be a tuple of booleans"):
            RockSampleState((1, 2), [True, False])
    
    def test_state_equality(self):
        """Test state equality comparison.
        
        Purpose: Validates that identical states are considered equal
        
        Given: Two RockSampleState instances with identical attributes
        When: States are compared for equality
        Then: States are equal and have same hash
        
        Test type: unit
        """
        state1 = RockSampleState((1, 2), (True, False))
        state2 = RockSampleState((1, 2), (True, False))
        state3 = RockSampleState((2, 1), (True, False))
        
        assert state1 == state2
        assert state1 != state3
        assert hash(state1) == hash(state2)


class TestRockSamplePOMDP:
    """Test RockSamplePOMDP environment."""
    
    def test_pomdp_creation_default_params(self):
        """Test POMDP creation with default parameters.
        
        Purpose: Validates that POMDP can be created with default parameters
        
        Given: No parameters specified
        When: RockSamplePOMDP is instantiated
        Then: POMDP is created with expected default values
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        
        assert pomdp.map_size == (5, 5)
        assert pomdp.rock_positions == [(0, 0), (2, 2), (3, 3)]
        assert pomdp.init_pos == (0, 0)
        assert pomdp.sensor_efficiency == 10.0
        assert pomdp.bad_rock_penalty == -10.0
        assert pomdp.good_rock_reward == 10.0
        assert pomdp.step_penalty == 0.0
        assert pomdp.sensor_use_penalty == 0.0
        assert pomdp.exit_reward == 10.0
        assert pomdp.discount_factor == 0.95
    
    def test_pomdp_creation_custom_params(self):
        """Test POMDP creation with custom parameters.
        
        Purpose: Validates that POMDP accepts custom configuration parameters
        
        Given: Custom parameters for map size, rocks, and rewards
        When: RockSamplePOMDP is instantiated with custom parameters
        Then: POMDP is created with specified custom values
        
        Test type: unit
        """
        custom_rocks = [(1, 1), (2, 3), (4, 4)]
        pomdp = RockSamplePOMDP(
            map_size=(7, 8),
            rock_positions=custom_rocks,
            init_pos=(1, 0),
            sensor_efficiency=15.0,
            bad_rock_penalty=-20.0,
            good_rock_reward=15.0,
            discount_factor=0.9
        )
        
        assert pomdp.map_size == (7, 8)
        assert pomdp.rock_positions == custom_rocks
        assert pomdp.init_pos == (1, 0)
        assert pomdp.sensor_efficiency == 15.0
        assert pomdp.bad_rock_penalty == -20.0
        assert pomdp.good_rock_reward == 15.0
        assert pomdp.discount_factor == 0.9
    
    def test_pomdp_validation_invalid_map_size(self):
        """Test parameter validation for invalid map size.
        
        Purpose: Validates proper error handling for invalid map dimensions
        
        Given: Non-positive map size values
        When: RockSamplePOMDP is instantiated
        Then: ValueError is raised for invalid dimensions
        
        Test type: unit
        """
        with pytest.raises(ValueError, match="Map size must be positive"):
            RockSamplePOMDP(map_size=(0, 5))
        
        with pytest.raises(ValueError, match="Map size must be positive"):
            RockSamplePOMDP(map_size=(5, -1))
    
    def test_pomdp_validation_empty_rocks(self):
        """Test parameter validation for empty rock positions.
        
        Purpose: Validates that empty rock list is allowed (creates environment with no rocks)
        
        Given: Empty rock positions list
        When: RockSamplePOMDP is instantiated
        Then: POMDP is created successfully with no rocks
        
        Test type: unit
        """
        # Empty rocks should be allowed - creates environment with no rocks
        pomdp = RockSamplePOMDP(rock_positions=[])
        assert pomdp.rock_positions == []  # Should remain empty as requested
        assert len(pomdp.get_actions()) == 5  # Only basic actions, no check actions
        
        # Verify it doesn't use default rocks when explicitly set to empty
        assert len(pomdp.action_names) == 5
    
    def test_pomdp_validation_rocks_out_of_bounds(self):
        """Test parameter validation for rocks outside map bounds.
        
        Purpose: Validates proper error handling for invalid rock positions
        
        Given: Rock positions outside map boundaries
        When: RockSamplePOMDP is instantiated
        Then: ValueError is raised for out-of-bounds positions
        
        Test type: unit
        """
        with pytest.raises(ValueError, match="Rock position .* is outside map bounds"):
            RockSamplePOMDP(map_size=(3, 3), rock_positions=[(0, 0), (3, 3)])
        
        with pytest.raises(ValueError, match="Rock position .* is outside map bounds"):
            RockSamplePOMDP(map_size=(5, 5), rock_positions=[(2, 2), (-1, 3)])
    
    def test_pomdp_validation_init_pos_out_of_bounds(self):
        """Test parameter validation for initial position outside bounds.
        
        Purpose: Validates proper error handling for invalid initial position
        
        Given: Initial position outside map boundaries
        When: RockSamplePOMDP is instantiated
        Then: ValueError is raised for out-of-bounds initial position
        
        Test type: unit
        """
        with pytest.raises(ValueError, match="Initial position .* is outside map bounds"):
            RockSamplePOMDP(map_size=(3, 3), rock_positions=[], init_pos=(3, 2))
        
        with pytest.raises(ValueError, match="Initial position .* is outside map bounds"):
            RockSamplePOMDP(map_size=(5, 5), rock_positions=[], init_pos=(2, -1))
    
    def test_get_actions(self):
        """Test action enumeration.
        
        Purpose: Validates that all expected actions are available
        
        Given: A RockSamplePOMDP with 3 rocks
        When: get_actions() is called
        Then: Returns list with correct number of actions (5 basic + 3 check actions)
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP(rock_positions=[(0, 0), (2, 2), (3, 3)])
        actions = pomdp.get_actions()
        
        expected_actions = [0, 1, 2, 3, 4, 5, 6, 7]  # 5 basic + 3 check
        assert actions == expected_actions
        assert len(pomdp.action_names) == 8
        assert pomdp.action_names[0] == "sample"
        assert pomdp.action_names[1] == "north"
        assert pomdp.action_names[5] == "check_rock_0"
    
    def test_initial_state_distribution(self):
        """Test initial state distribution generation.
        
        Purpose: Validates that initial state distribution covers all rock configurations
        
        Given: A RockSamplePOMDP with 2 rocks
        When: initial_state_dist() is called
        Then: Returns distribution with 2^2=4 equally probable states
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP(
            map_size=(3, 3),
            rock_positions=[(0, 0), (1, 1)],
            init_pos=(0, 0)
        )
        dist = pomdp.initial_state_dist()
        
        assert len(dist.values) == 4  # 2^2 possible rock configurations
        assert np.allclose(dist.probs, [0.25] * 4)
        
        # Check all states have correct initial position
        for state in dist.values:
            assert state.robot_pos == (0, 0)
    
    def test_initial_observation_distribution(self):
        """Test initial observation distribution.
        
        Purpose: Validates that initial observation is always 'none'
        
        Given: Any RockSamplePOMDP instance
        When: initial_observation_dist() is called
        Then: Returns distribution with single 'none' observation
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        dist = pomdp.initial_observation_dist()
        
        assert len(dist.values) == 1
        assert dist.values[0] == "none"
        assert dist.probs[0] == 1.0
    
    def test_is_equal_observation(self):
        """Test observation equality comparison.
        
        Purpose: Validates proper observation equality checking
        
        Given: Various observation values
        When: is_equal_observation() is called
        Then: Returns correct equality results
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        
        assert pomdp.is_equal_observation("good", "good")
        assert pomdp.is_equal_observation("bad", "bad")
        assert pomdp.is_equal_observation("none", "none")
        assert not pomdp.is_equal_observation("good", "bad")
        assert not pomdp.is_equal_observation("none", "good")
    
    def test_is_terminal(self):
        """Test terminal state detection.
        
        Purpose: Validates that terminal states are correctly identified
        
        Given: States with different robot positions
        When: is_terminal() is called
        Then: Returns True only for terminal position (-1, -1)
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        
        normal_state = RockSampleState((2, 3), (True, False, True))
        terminal_state = RockSampleState((-1, -1), (True, False, True))
        
        assert not pomdp.is_terminal(normal_state)
        assert pomdp.is_terminal(terminal_state)


class TestStateTransitionModel:
    """Test state transition model."""
    
    def test_transition_movement_north(self):
        """Test north movement transition.
        
        Purpose: Validates correct robot movement north within bounds
        
        Given: Robot at position (2, 1) with north action
        When: State transition is sampled
        Then: Robot moves to position (1, 1) and rocks remain unchanged
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = RockSampleState((2, 1), (True, False, True))
        transition = RockSampleStateTransitionModel(state, 1, pomdp)  # North
        
        next_state = transition.sample()[0]
        assert next_state.robot_pos == (1, 1)
        assert next_state.rocks == (True, False, True)
    
    def test_transition_movement_north_boundary(self):
        """Test north movement at boundary.
        
        Purpose: Validates that robot cannot move beyond north boundary
        
        Given: Robot at top boundary (0, 1) with north action
        When: State transition is sampled
        Then: Robot remains at (0, 1) due to boundary constraint
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = RockSampleState((0, 1), (True, False, True))
        transition = RockSampleStateTransitionModel(state, 1, pomdp)  # North
        
        next_state = transition.sample()[0]
        assert next_state.robot_pos == (0, 1)  # Blocked by boundary
    
    def test_transition_movement_east(self):
        """Test east movement transition.
        
        Purpose: Validates correct robot movement east within bounds
        
        Given: Robot at position (1, 2) with east action
        When: State transition is sampled
        Then: Robot moves to position (1, 3) and rocks remain unchanged
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = RockSampleState((1, 2), (True, False, True))
        transition = RockSampleStateTransitionModel(state, 2, pomdp)  # East
        
        next_state = transition.sample()[0]
        assert next_state.robot_pos == (1, 3)
        assert next_state.rocks == (True, False, True)
    
    def test_transition_movement_east_to_exit(self):
        """Test east movement leading to exit.
        
        Purpose: Validates terminal state transition when exiting map
        
        Given: Robot at right boundary (1, 4) in 5x5 map with east action
        When: State transition is sampled
        Then: Robot moves to terminal position (-1, -1)
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()  # Default 5x5 map
        state = RockSampleState((1, 4), (True, False, True))  # Right boundary
        transition = RockSampleStateTransitionModel(state, 2, pomdp)  # East
        
        next_state = transition.sample()[0]
        assert next_state.robot_pos == (-1, -1)  # Terminal state
    
    def test_transition_movement_south(self):
        """Test south movement transition.
        
        Purpose: Validates correct robot movement south within bounds
        
        Given: Robot at position (1, 2) with south action
        When: State transition is sampled
        Then: Robot moves to position (2, 2) and rocks remain unchanged
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = RockSampleState((1, 2), (True, False, True))
        transition = RockSampleStateTransitionModel(state, 3, pomdp)  # South
        
        next_state = transition.sample()[0]
        assert next_state.robot_pos == (2, 2)
        assert next_state.rocks == (True, False, True)
    
    def test_transition_movement_south_boundary(self):
        """Test south movement at boundary.
        
        Purpose: Validates that robot cannot move beyond south boundary
        
        Given: Robot at bottom boundary (4, 1) in 5x5 map with south action
        When: State transition is sampled
        Then: Robot remains at (4, 1) due to boundary constraint
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()  # Default 5x5 map
        state = RockSampleState((4, 1), (True, False, True))  # Bottom boundary
        transition = RockSampleStateTransitionModel(state, 3, pomdp)  # South
        
        next_state = transition.sample()[0]
        assert next_state.robot_pos == (4, 1)  # Blocked by boundary
    
    def test_transition_movement_west(self):
        """Test west movement transition.
        
        Purpose: Validates correct robot movement west within bounds
        
        Given: Robot at position (2, 3) with west action
        When: State transition is sampled
        Then: Robot moves to position (2, 2) and rocks remain unchanged
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = RockSampleState((2, 3), (True, False, True))
        transition = RockSampleStateTransitionModel(state, 4, pomdp)  # West
        
        next_state = transition.sample()[0]
        assert next_state.robot_pos == (2, 2)
        assert next_state.rocks == (True, False, True)
    
    def test_transition_movement_west_boundary(self):
        """Test west movement at boundary.
        
        Purpose: Validates that robot cannot move beyond west boundary
        
        Given: Robot at left boundary (2, 0) with west action
        When: State transition is sampled
        Then: Robot remains at (2, 0) due to boundary constraint
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = RockSampleState((2, 0), (True, False, True))
        transition = RockSampleStateTransitionModel(state, 4, pomdp)  # West
        
        next_state = transition.sample()[0]
        assert next_state.robot_pos == (2, 0)  # Blocked by boundary
    
    def test_transition_sample_at_rock_position(self):
        """Test sampling action at rock position.
        
        Purpose: Validates that sampling at rock position changes rock state
        
        Given: Robot at rock position (0, 0) with good rock and sample action
        When: State transition is sampled
        Then: Robot stays at (0, 0) and rock becomes bad
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP(rock_positions=[(0, 0), (2, 2), (3, 3)])
        state = RockSampleState((0, 0), (True, False, True))  # At rock 0
        transition = RockSampleStateTransitionModel(state, 0, pomdp)  # Sample
        
        next_state = transition.sample()[0]
        assert next_state.robot_pos == (0, 0)  # Stay in place
        assert next_state.rocks == (False, False, True)  # Rock 0 becomes bad
    
    def test_transition_sample_not_at_rock(self):
        """Test sampling action not at rock position.
        
        Purpose: Validates that sampling away from rocks has no effect on rock states
        
        Given: Robot at position not matching any rock with sample action
        When: State transition is sampled
        Then: Robot stays in place and all rocks remain unchanged
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP(rock_positions=[(0, 0), (2, 2), (3, 3)])
        state = RockSampleState((1, 1), (True, False, True))  # Not at any rock
        transition = RockSampleStateTransitionModel(state, 0, pomdp)  # Sample
        
        next_state = transition.sample()[0]
        assert next_state.robot_pos == (1, 1)
        assert next_state.rocks == (True, False, True)  # No change
    
    def test_transition_check_action(self):
        """Test check action transition.
        
        Purpose: Validates that check actions keep robot in place
        
        Given: Robot at any position with check action
        When: State transition is sampled
        Then: Robot remains at same position and rocks unchanged
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = RockSampleState((2, 1), (True, False, True))
        transition = RockSampleStateTransitionModel(state, 5, pomdp)  # Check rock 0
        
        next_state = transition.sample()[0]
        assert next_state.robot_pos == (2, 1)  # Stay in place
        assert next_state.rocks == (True, False, True)  # No change


class TestObservationModel:
    """Test observation model."""
    
    def test_observation_movement_action(self):
        """Test observation for movement actions.
        
        Purpose: Validates that movement actions always produce 'none' observation
        
        Given: Any state with movement action (1-4)
        When: Observation is sampled
        Then: Always returns 'none' observation
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = RockSampleState((1, 1), (True, False))
        
        for action in [1, 2, 3, 4]:  # North, East, South, West
            obs_model = RockSampleObservationModel(state, action, pomdp)
            observation = obs_model.sample()[0]
            assert observation == "none"
    
    def test_observation_sample_action(self):
        """Test observation for sample action.
        
        Purpose: Validates that sample action always produces 'none' observation
        
        Given: Any state with sample action
        When: Observation is sampled
        Then: Returns 'none' observation
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = RockSampleState((1, 1), (True, False))
        obs_model = RockSampleObservationModel(state, 0, pomdp)  # Sample
        
        observation = obs_model.sample()[0]
        assert observation == "none"
    
    def test_observation_check_action_close_distance(self):
        """Test observation for check action at close distance.
        
        Purpose: Validates high accuracy sensor readings at close range
        
        Given: Robot very close to rock with check action and high sensor efficiency
        When: Observation probabilities are calculated
        Then: Correct observation has high probability
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP(
            rock_positions=[(1, 1), (3, 3)],
            sensor_efficiency=50.0  # High efficiency
        )
        state = RockSampleState((1, 1), (True, False))  # At rock 0 position
        obs_model = RockSampleObservationModel(state, 5, pomdp)  # Check rock 0
        
        # Should have high probability of correct observation
        probs = obs_model.probability(["good", "bad", "none"])
        assert probs[0] > 0.9  # High probability of "good"
        assert probs[1] < 0.1  # Low probability of "bad"
        assert probs[2] == 0.0  # No probability of "none"
    
    def test_observation_check_action_far_distance(self):
        """Test observation for check action at far distance.
        
        Purpose: Validates reduced accuracy sensor readings at far range
        
        Given: Robot far from rock with check action and very low sensor efficiency
        When: Observation probabilities are calculated
        Then: Observation probabilities are closer to random
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP(
            rock_positions=[(0, 0), (1, 2)],  # Closer rocks
            sensor_efficiency=2.0  # Moderate efficiency for reasonable uncertainty
        )
        state = RockSampleState((0, 0), (True, False))  # At rock 0, check rock 1 at (1,2)
        obs_model = RockSampleObservationModel(state, 6, pomdp)  # Check rock 1
        
        probs = obs_model.probability(["good", "bad", "none"])
        # Distance = sqrt((0-1)^2 + (0-2)^2) = sqrt(5) ≈ 2.24
        # Efficiency = exp(-2.24/2.0) ≈ 0.33
        # For bad rock: P(good) = 1-0.33 = 0.67, P(bad) = 0.33
        assert 0.6 < probs[0] < 0.8  # P(good|bad_rock) should be high due to low efficiency
        assert 0.2 < probs[1] < 0.4  # P(bad|bad_rock) should be low
        assert probs[2] == 0.0
    
    def test_observation_check_invalid_rock(self):
        """Test observation for check action on invalid rock index.
        
        Purpose: Validates proper handling of invalid rock check actions
        
        Given: Check action for rock index beyond available rocks
        When: Observation is sampled
        Then: Returns 'none' observation
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP(rock_positions=[(0, 0), (2, 2)])  # Only 2 rocks
        state = RockSampleState((1, 1), (True, False))
        obs_model = RockSampleObservationModel(state, 7, pomdp)  # Check rock 2 (invalid)
        
        observation = obs_model.sample()[0]
        assert observation == "none"
    
    def test_observation_probability_calculation(self):
        """Test observation probability calculation accuracy.
        
        Purpose: Validates mathematical correctness of sensor probability model
        
        Given: Known robot and rock positions with defined sensor efficiency
        When: Observation probabilities are calculated
        Then: Probabilities sum to 1 and match expected sensor model
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP(
            rock_positions=[(2, 2)],
            sensor_efficiency=10.0
        )
        state = RockSampleState((1, 1), (True,))  # Good rock at distance sqrt(2)
        obs_model = RockSampleObservationModel(state, 5, pomdp)  # Check rock 0
        
        probs = obs_model.probability(["good", "bad", "none"])
        
        # Probabilities should sum to 1 (excluding 'none')
        assert abs(probs[0] + probs[1] - 1.0) < 1e-10
        assert probs[2] == 0.0
        
        # For good rock, "good" observation should be more likely
        assert probs[0] > probs[1]


class TestRewardFunction:
    """Test reward function."""
    
    def test_reward_basic_movement(self):
        """Test reward for basic movement actions.
        
        Purpose: Validates that movement actions incur only step penalty
        
        Given: Robot at any position with movement action
        When: Reward is calculated
        Then: Returns only the step penalty
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP(step_penalty=-1.0)
        state = RockSampleState((2, 2), (True, False))
        
        for action in [1, 2, 3, 4]:  # Movement actions
            reward = pomdp.reward(state, action)
            assert reward == -1.0
    
    def test_reward_exit_action(self):
        """Test reward for exiting the grid.
        
        Purpose: Validates that exiting through right boundary gives exit reward
        
        Given: Robot at right edge with east action
        When: Reward is calculated
        Then: Returns step penalty plus exit reward
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP(
            step_penalty=-1.0,
            exit_reward=10.0
        )
        state = RockSampleState((2, 4), (True, False))  # At right edge of 5x5 map
        
        reward = pomdp.reward(state, 2)  # East action
        assert reward == -1.0 + 10.0  # step_penalty + exit_reward
    
    def test_reward_sample_good_rock(self):
        """Test reward for sampling good rock.
        
        Purpose: Validates reward for successfully sampling good rock
        
        Given: Robot at good rock position with sample action
        When: Reward is calculated
        Then: Returns step penalty plus good rock reward
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP(
            rock_positions=[(1, 1), (3, 3)],
            step_penalty=-0.5,
            good_rock_reward=15.0
        )
        state = RockSampleState((1, 1), (True, False))  # At good rock
        
        reward = pomdp.reward(state, 0)  # Sample action
        assert reward == -0.5 + 15.0
    
    def test_reward_sample_bad_rock(self):
        """Test reward for sampling bad rock.
        
        Purpose: Validates penalty for sampling bad rock
        
        Given: Robot at bad rock position with sample action
        When: Reward is calculated
        Then: Returns step penalty plus bad rock penalty
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP(
            rock_positions=[(1, 1), (3, 3)],
            step_penalty=-0.5,
            bad_rock_penalty=-20.0
        )
        state = RockSampleState((1, 1), (False, True))  # At bad rock
        
        reward = pomdp.reward(state, 0)  # Sample action
        assert reward == -0.5 + (-20.0)
    
    def test_reward_sample_not_at_rock(self):
        """Test reward for sampling not at rock position.
        
        Purpose: Validates that sampling away from rocks gives only step penalty
        
        Given: Robot not at any rock position with sample action
        When: Reward is calculated
        Then: Returns only step penalty
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP(
            rock_positions=[(0, 0), (3, 3)],
            step_penalty=-1.5
        )
        state = RockSampleState((1, 1), (True, False))  # Not at any rock
        
        reward = pomdp.reward(state, 0)  # Sample action
        assert reward == -1.5
    
    def test_reward_sensor_use(self):
        """Test reward for using sensor.
        
        Purpose: Validates that sensor use incurs additional penalty
        
        Given: Any state with check action and sensor use penalty
        When: Reward is calculated
        Then: Returns step penalty plus sensor use penalty
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP(
            step_penalty=-1.0,
            sensor_use_penalty=-2.0
        )
        state = RockSampleState((2, 2), (True, False))
        
        reward = pomdp.reward(state, 5)  # Check rock 0
        assert reward == -1.0 + (-2.0)
    
    def test_reward_range(self):
        """Test that reward range is correctly calculated.
        
        Purpose: Validates that RockSamplePOMDP reward range is properly calculated based on environment parameters
        
        Given: A RockSamplePOMDP environment with specific penalty/reward parameters
        When: Environment reward_range attribute is checked
        Then: Returns range based on min_reward (step+bad_rock+sensor penalties) and max_reward (step+exit reward)
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP(
            step_penalty=-1.0,
            bad_rock_penalty=-5.0,
            good_rock_reward=8.0,
            exit_reward=12.0,
            sensor_use_penalty=-0.5
        )
        
        # Expected calculations from RockSamplePOMDP constructor:
        # min_reward = step_penalty + bad_rock_penalty + sensor_use_penalty = -1.0 + (-5.0) + (-0.5) = -6.5
        # max_reward = step_penalty + exit_reward = -1.0 + 12.0 = 11.0
        expected_min = -6.5
        expected_max = 11.0
        
        assert pomdp.reward_range == (expected_min, expected_max)
        
        # Verify with different parameters
        pomdp2 = RockSamplePOMDP(
            step_penalty=-0.5,
            bad_rock_penalty=-10.0,
            exit_reward=20.0,
            sensor_use_penalty=-1.0
        )
        
        expected_min2 = -0.5 + (-10.0) + (-1.0)  # -11.5
        expected_max2 = -0.5 + 20.0  # 19.5
        
        assert pomdp2.reward_range == (expected_min2, expected_max2)


class TestMetricsComputation:
    """Test metrics computation."""
    
    def test_compute_metrics_empty_histories(self):
        """Test metrics computation with empty histories.
        
        Purpose: Validates proper handling of empty history list
        
        Given: Empty list of histories
        When: compute_metrics() is called
        Then: Returns empty list of metrics
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        metrics = pomdp.compute_metrics([])
        assert metrics == []
    
    def test_compute_metrics_rocks_sampled(self):
        """Test computation of rocks sampled metric.
        
        Purpose: Validates accurate counting of sampling actions across episodes
        
        Given: Histories with various sampling actions
        When: compute_metrics() is called
        Then: Returns correct average rocks sampled metric
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        
        # Create mock histories
        histories = []
        for num_samples in [1, 2, 3]:
            steps = []
            state = RockSampleState((0, 0), (True, True))
            
            # Add steps with sample actions
            for i in range(num_samples):
                step = StepData(
                    state=state,
                    action=0,  # Sample action
                    next_state=state,
                    observation="none",
                    reward=10.0,
                    belief=None
                )
                steps.append(step)
            
            # Add non-sample action
            step = StepData(
                state=state,
                action=1,  # Movement action
                next_state=state,
                observation="none", 
                reward=-1.0,
                belief=None
            )
            steps.append(step)
            
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
                policy_run_data=None
            )
            histories.append(history)
        
        metrics = pomdp.compute_metrics(histories)
        
        # Find rocks sampled metric
        rocks_metric = next((m for m in metrics if m.name == "avg_rocks_sampled"), None)
        assert rocks_metric is not None
        assert rocks_metric.value == 2.0  # (1+2+3)/3
    
    def test_compute_metrics_exit_success_rate(self):
        """Test computation of exit success rate metric.
        
        Purpose: Validates accurate calculation of episode exit success rate
        
        Given: Histories with mix of terminal and non-terminal episodes
        When: compute_metrics() is called
        Then: Returns correct exit success rate metric
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        
        histories = []
        
        # Successful exit history
        terminal_state = RockSampleState((-1, -1), (True, False))
        success_step = StepData(
            state=terminal_state,
            action=None,
            next_state=terminal_state,
            observation="none",
            reward=10.0,
            belief=None
        )
        success_history = History(
            history=[success_step],
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=1,
            reach_terminal_state=True,
            policy_run_data=None
        )
        histories.append(success_history)
        
        # Failed (non-exit) history
        normal_state = RockSampleState((2, 2), (True, False))
        fail_step = StepData(
            state=normal_state,
            action=1,
            next_state=normal_state,
            observation="none",
            reward=-1.0,
            belief=None
        )
        fail_history = History(
            history=[fail_step],
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=1,
            reach_terminal_state=False,
            policy_run_data=None
        )
        histories.append(fail_history)
        
        metrics = pomdp.compute_metrics(histories)
        
        # Find exit success rate metric
        exit_metric = next((m for m in metrics if m.name == "exit_success_rate"), None)
        assert exit_metric is not None
        assert exit_metric.value == 0.5  # 1/2 successful exits


class TestVisualization:
    """Test visualization functionality."""
    
    def test_visualize_path_parameter_validation(self):
        """Test visualization parameter validation.
        
        Purpose: Validates proper parameter validation for visualization
        
        Given: Invalid cache_path parameters
        When: visualize_path() is called
        Then: Appropriate errors are raised
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = RockSampleState((0, 0), (True, False))
        
        with pytest.raises(TypeError, match="cache_path must be a Path object"):
            pomdp.visualize_path([state], [1], "invalid_path")
        
        with pytest.raises(ValueError, match="cache_path must end with .gif"):
            pomdp.visualize_path([state], [1], Path("test.png"))
    
    def test_cache_visualization_empty_history(self):
        """Test visualization caching with empty history.
        
        Purpose: Validates proper error handling for empty history
        
        Given: Empty history
        When: cache_visualization() is called
        Then: ValueError is raised
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        empty_history = History(
            history=[],
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=0,
            reach_terminal_state=False,
            policy_run_data=None
        )
        
        with pytest.raises(ValueError, match="Cannot visualize empty history"):
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_path = Path(temp_dir) / "test.gif"
                pomdp.cache_visualization(empty_history.history, cache_path)
    
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.animation.FuncAnimation.save')
    def test_cache_visualization_success(self, mock_save, mock_close):
        """Test successful visualization caching.
        
        Purpose: Validates successful generation and caching of visualization
        
        Given: Valid history with steps
        When: cache_visualization() is called
        Then: Visualization is generated and saved without errors
        
        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        
        # Create history with steps
        state1 = RockSampleState((0, 0), (True, False))
        state2 = RockSampleState((0, 1), (True, False))
        
        step1 = StepData(
            state=state1,
            action=2,  # East
            next_state=state2,
            observation="none",
            reward=-1.0,
            belief=None
        )
        step2 = StepData(
            state=state2,
            action=None,
            next_state=state2,
            observation="none",
            reward=0.0,
            belief=None
        )
        
        history = History(
            history=[step1, step2],
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=2,
            reach_terminal_state=False,
            policy_run_data=None
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "test.gif"
            
            # Should not raise any exceptions
            pomdp.cache_visualization(history.history, cache_path)
            
            # Verify mocks were called
            mock_save.assert_called_once()
            mock_close.assert_called_once()


class TestRandomRockSample:
    """Test random RockSample creation utility."""
    
    def test_create_random_rock_sample_default(self):
        """Test random RockSample creation with default parameters.
        
        Purpose: Validates random environment generation with default settings
        
        Given: No parameters specified
        When: create_random_rock_sample() is called
        Then: Returns RockSamplePOMDP with expected default configuration
        
        Test type: unit
        """
        pomdp = create_random_rock_sample()
        
        assert pomdp.map_size == (7, 7)
        assert len(pomdp.rock_positions) == 8
        assert pomdp.init_pos == (0, 0)
        
        # Check all rock positions are within bounds
        for pos in pomdp.rock_positions:
            assert 0 <= pos[0] < 7
            assert 0 <= pos[1] < 7
    
    def test_create_random_rock_sample_custom_params(self):
        """Test random RockSample creation with custom parameters.
        
        Purpose: Validates random environment generation with custom settings
        
        Given: Custom map size, rock count, and seed
        When: create_random_rock_sample() is called
        Then: Returns RockSamplePOMDP with specified configuration
        
        Test type: unit
        """
        pomdp = create_random_rock_sample(map_size=5, num_rocks=4, seed=42)
        
        assert pomdp.map_size == (5, 5)
        assert len(pomdp.rock_positions) == 4
        
        # Check reproducibility with same seed
        pomdp2 = create_random_rock_sample(map_size=5, num_rocks=4, seed=42)
        assert pomdp.rock_positions == pomdp2.rock_positions
    
    def test_create_random_rock_sample_too_many_rocks(self):
        """Test random creation with more rocks than grid cells.
        
        Purpose: Validates proper handling when requested rocks exceed grid capacity
        
        Given: More rocks requested than available grid positions
        When: create_random_rock_sample() is called
        Then: Places maximum possible rocks without error
        
        Test type: unit
        """
        pomdp = create_random_rock_sample(map_size=2, num_rocks=10, seed=123)
        
        assert pomdp.map_size == (2, 2)
        assert len(pomdp.rock_positions) == 4  # Maximum for 2x2 grid
        
        # Ensure no duplicate positions
        assert len(set(pomdp.rock_positions)) == len(pomdp.rock_positions)


class TestIntegration:
    """Integration tests for complete POMDP functionality."""
    
    def test_complete_episode_simulation(self):
        """Test complete episode simulation.
        
        Purpose: Validates end-to-end functionality through complete episode
        
        Given: RockSamplePOMDP environment and sequence of actions
        When: Full episode is simulated step by step
        Then: All transitions, observations, and rewards work correctly
        
        Test type: integration
        """
        pomdp = RockSamplePOMDP(
            map_size=(3, 3),
            rock_positions=[(0, 0), (2, 2)],
            init_pos=(0, 0)
        )
        
        # Start with initial state
        initial_dist = pomdp.initial_state_dist()
        state = RockSampleState((0, 0), (True, False))  # Specific initial state
        
        # Action sequence: check rock 0, sample, move east three times (exit)
        actions = [4, 0, 2, 2, 2]  # check_rock_0, sample, east, east, east
        
        current_state = state
        total_reward = 0.0
        
        for i, action in enumerate(actions):
            # Sample next step
            next_state, observation, reward = pomdp.sample_next_step(current_state, action)
            total_reward += reward
            
            # Verify step is valid
            assert isinstance(next_state, RockSampleState)
            assert isinstance(observation, str)
            assert isinstance(reward, (int, float))
            
            current_state = next_state
            
            if pomdp.is_terminal(current_state):
                break
        
        # Should have received rewards for sampling good rock and exiting
        assert total_reward > 0  # Good rock + exit rewards exceed step penalties
        assert pomdp.is_terminal(current_state)  # Should reach terminal state
    
    def test_sensor_accuracy_validation(self):
        """Test sensor accuracy matches expected behavior.
        
        Purpose: Validates sensor model produces statistically correct observations
        
        Given: Known robot and rock positions with multiple sensor readings
        When: Many observations are sampled from check actions  
        Then: Observation frequencies match expected sensor efficiency
        
        Test type: integration
        """
        # High efficiency sensor for deterministic behavior
        pomdp = RockSamplePOMDP(
            rock_positions=[(1, 1)],
            sensor_efficiency=100.0  # Very high efficiency
        )
        
        # Robot at rock position - should get very accurate readings
        state = RockSampleState((1, 1), (True,))  # Good rock
        obs_model = RockSampleObservationModel(state, 5, pomdp)  # Check rock 0
        
        # Sample many observations
        observations = obs_model.sample(100)
        good_count = sum(1 for obs in observations if obs == "good")
        
        # With very high efficiency and zero distance, should be very accurate
        assert good_count > 90  # Should be > 90% accurate
    
    def test_state_space_coverage(self):
        """Test that initial state distribution covers expected state space.
        
        Purpose: Validates initial state distribution completeness
        
        Given: RockSamplePOMDP with known number of rocks
        When: Initial state distribution is generated
        Then: All possible rock configurations are represented
        
        Test type: integration
        """
        pomdp = RockSamplePOMDP(
            rock_positions=[(0, 0), (1, 1)],
            init_pos=(0, 0)
        )
        
        initial_dist = pomdp.initial_state_dist()
        
        # Should have 2^2 = 4 states
        assert len(initial_dist.values) == 4
        
        # Extract rock configurations
        rock_configs = set()
        for state in initial_dist.values:
            assert state.robot_pos == (0, 0)  # All start at same position
            rock_configs.add(state.rocks)
        
        expected_configs = {
            (True, True),
            (True, False), 
            (False, True),
            (False, False)
        }
        
        assert rock_configs == expected_configs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])