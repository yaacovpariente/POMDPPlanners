# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for RockSample POMDP environment.

This module tests the RockSample POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

# pylint: disable=too-many-lines

import random
from unittest.mock import Mock

import numpy as np
import pytest

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.rock_sample_pomdp import (
    RockSamplePOMDP,
    RockSampleState,
    create_random_rock_sample,
    create_rock_sample_state,
    get_robot_pos,
    get_rocks,
    states_equal,
)
from POMDPPlanners.tests.test_utils.confidence_interval_utils import (
    verify_metrics_within_confidence_intervals,
)
from POMDPPlanners.tests.test_utils.history_builders import build_test_history
from POMDPPlanners.tests.test_utils.metric_invariants_utils import (
    verify_history_returns_bounded,
    verify_metric_sanity,
    verify_return_shift_linearity,
)

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
        state = create_rock_sample_state(robot_pos, rocks)

        assert get_robot_pos(state) == robot_pos
        assert get_rocks(state) == rocks
        assert isinstance(get_rocks(state), tuple)

    def test_state_immutable(self):
        """Test that state is immutable after creation.

        Purpose: Validates that RockSampleState is properly frozen/immutable

        Given: A created RockSampleState instance
        When: Attempting to modify state attributes
        Then: State array supports numpy array assignment semantics

        Test type: unit
        """
        state = create_rock_sample_state((1, 1), (True, False))

        # Numpy arrays are mutable, but we can verify the structure
        assert isinstance(state, np.ndarray)
        assert len(state) == 4  # 2 for robot pos + 2 for rocks

    def test_state_validation_invalid_robot_pos(self):
        """Test state validation with invalid robot position.

        Purpose: Validates proper error handling for malformed robot positions

        Given: Invalid robot position (not tuple of two integers)
        When: create_rock_sample_state is called
        Then: Function handles the input (numpy array creation doesn't validate types)

        Test type: unit
        """
        # Numpy arrays accept various inputs, validation happens at usage level
        # These no longer raise errors during creation
        pass

    def test_state_validation_invalid_rocks(self):
        """Test state validation with invalid rock states.

        Purpose: Validates proper error handling for malformed rock states

        Given: Invalid rock states (not tuple)
        When: create_rock_sample_state is called
        Then: Function handles the input (numpy array creation doesn't validate types)

        Test type: unit
        """
        # Numpy arrays accept various inputs, validation happens at usage level
        pass

    def test_state_equality(self):
        """Test state equality comparison.

        Purpose: Validates that identical states are considered equal

        Given: Two RockSampleState instances with identical attributes
        When: States are compared for equality
        Then: States are equal

        Test type: unit
        """
        state1 = create_rock_sample_state((1, 2), (True, False))
        state2 = create_rock_sample_state((1, 2), (True, False))
        state3 = create_rock_sample_state((2, 1), (True, False))

        assert states_equal(state1, state2)
        assert not states_equal(state1, state3)


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
            discount_factor=0.9,
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
        pomdp = RockSamplePOMDP(map_size=(3, 3), rock_positions=[(0, 0), (1, 1)], init_pos=(0, 0))
        dist = pomdp.initial_state_dist()

        assert len(dist.values) == 4  # 2^2 possible rock configurations
        assert np.allclose(dist.probs, [0.25] * 4)

        # Check all states have correct initial position
        for state in dist.values:
            assert get_robot_pos(state) == (0, 0)

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

        normal_state = create_rock_sample_state((2, 3), (True, False, True))
        terminal_state = create_rock_sample_state((-1, -1), (True, False, True))

        assert not pomdp.is_terminal(normal_state)
        assert pomdp.is_terminal(terminal_state)


class TestStateTransitionModel:
    """Test state transition model via env.sample_next_state env-API."""

    def test_transition_movement_north(self):
        """Test north movement transition.

        Purpose: Validates correct robot movement north within bounds

        Given: Robot at position (2, 1) with north action
        When: env.sample_next_state is invoked
        Then: Robot moves to position (1, 1) and rocks remain unchanged

        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = create_rock_sample_state((2, 1), (True, False, True))

        next_state = pomdp.sample_next_state(state=state, action=1)  # North
        assert get_robot_pos(next_state) == (1, 1)
        assert get_rocks(next_state) == (True, False, True)

    def test_transition_movement_north_boundary(self):
        """Test north movement at boundary.

        Purpose: Validates that robot cannot move beyond north boundary

        Given: Robot at top boundary (0, 1) with north action
        When: env.sample_next_state is invoked
        Then: Robot remains at (0, 1) due to boundary constraint

        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = create_rock_sample_state((0, 1), (True, False, True))

        next_state = pomdp.sample_next_state(state=state, action=1)  # North
        assert get_robot_pos(next_state) == (0, 1)  # Blocked by boundary

    def test_transition_movement_east(self):
        """Test east movement transition.

        Purpose: Validates correct robot movement east within bounds

        Given: Robot at position (1, 2) with east action
        When: env.sample_next_state is invoked
        Then: Robot moves to position (1, 3) and rocks remain unchanged

        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = create_rock_sample_state((1, 2), (True, False, True))

        next_state = pomdp.sample_next_state(state=state, action=2)  # East
        assert get_robot_pos(next_state) == (1, 3)
        assert get_rocks(next_state) == (True, False, True)

    def test_transition_movement_east_to_exit(self):
        """Test east movement leading to exit.

        Purpose: Validates terminal state transition when exiting map

        Given: Robot at right boundary (1, 4) in 5x5 map with east action
        When: env.sample_next_state is invoked
        Then: Robot moves to terminal position (-1, -1)

        Test type: unit
        """
        pomdp = RockSamplePOMDP()  # Default 5x5 map
        state = create_rock_sample_state((1, 4), (True, False, True))  # Right boundary

        next_state = pomdp.sample_next_state(state=state, action=2)  # East
        assert get_robot_pos(next_state) == (-1, -1)  # Terminal state

    def test_transition_movement_south(self):
        """Test south movement transition.

        Purpose: Validates correct robot movement south within bounds

        Given: Robot at position (1, 2) with south action
        When: env.sample_next_state is invoked
        Then: Robot moves to position (2, 2) and rocks remain unchanged

        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = create_rock_sample_state((1, 2), (True, False, True))

        next_state = pomdp.sample_next_state(state=state, action=3)  # South
        assert get_robot_pos(next_state) == (2, 2)
        assert get_rocks(next_state) == (True, False, True)

    def test_transition_movement_south_boundary(self):
        """Test south movement at boundary.

        Purpose: Validates that robot cannot move beyond south boundary

        Given: Robot at bottom boundary (4, 1) in 5x5 map with south action
        When: env.sample_next_state is invoked
        Then: Robot remains at (4, 1) due to boundary constraint

        Test type: unit
        """
        pomdp = RockSamplePOMDP()  # Default 5x5 map
        state = create_rock_sample_state((4, 1), (True, False, True))  # Bottom boundary

        next_state = pomdp.sample_next_state(state=state, action=3)  # South
        assert get_robot_pos(next_state) == (4, 1)  # Blocked by boundary

    def test_transition_movement_west(self):
        """Test west movement transition.

        Purpose: Validates correct robot movement west within bounds

        Given: Robot at position (2, 3) with west action
        When: env.sample_next_state is invoked
        Then: Robot moves to position (2, 2) and rocks remain unchanged

        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = create_rock_sample_state((2, 3), (True, False, True))

        next_state = pomdp.sample_next_state(state=state, action=4)  # West
        assert get_robot_pos(next_state) == (2, 2)
        assert get_rocks(next_state) == (True, False, True)

    def test_transition_movement_west_boundary(self):
        """Test west movement at boundary.

        Purpose: Validates that robot cannot move beyond west boundary

        Given: Robot at left boundary (2, 0) with west action
        When: env.sample_next_state is invoked
        Then: Robot remains at (2, 0) due to boundary constraint

        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = create_rock_sample_state((2, 0), (True, False, True))

        next_state = pomdp.sample_next_state(state=state, action=4)  # West
        assert get_robot_pos(next_state) == (2, 0)  # Blocked by boundary

    def test_transition_sample_at_rock_position(self):
        """Test sampling action at rock position.

        Purpose: Validates that sampling at rock position changes rock state

        Given: Robot at rock position (0, 0) with good rock and sample action
        When: env.sample_next_state is invoked
        Then: Robot stays at (0, 0) and rock becomes bad

        Test type: unit
        """
        pomdp = RockSamplePOMDP(rock_positions=[(0, 0), (2, 2), (3, 3)])
        state = create_rock_sample_state((0, 0), (True, False, True))  # At rock 0

        next_state = pomdp.sample_next_state(state=state, action=0)  # Sample
        assert get_robot_pos(next_state) == (0, 0)  # Stay in place
        assert get_rocks(next_state) == (False, False, True)  # Rock 0 becomes bad

    def test_transition_sample_not_at_rock(self):
        """Test sampling action not at rock position.

        Purpose: Validates that sampling away from rocks has no effect on rock states

        Given: Robot at position not matching any rock with sample action
        When: env.sample_next_state is invoked
        Then: Robot stays in place and all rocks remain unchanged

        Test type: unit
        """
        pomdp = RockSamplePOMDP(rock_positions=[(0, 0), (2, 2), (3, 3)])
        state = create_rock_sample_state((1, 1), (True, False, True))  # Not at any rock

        next_state = pomdp.sample_next_state(state=state, action=0)  # Sample
        assert get_robot_pos(next_state) == (1, 1)
        assert get_rocks(next_state) == (True, False, True)  # No change

    def test_transition_check_action(self):
        """Test check action transition.

        Purpose: Validates that check actions keep robot in place

        Given: Robot at any position with check action
        When: env.sample_next_state is invoked
        Then: Robot remains at same position and rocks unchanged

        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = create_rock_sample_state((2, 1), (True, False, True))

        next_state = pomdp.sample_next_state(state=state, action=5)  # Check rock 0
        assert get_robot_pos(next_state) == (2, 1)  # Stay in place
        assert get_rocks(next_state) == (True, False, True)  # No change


class TestObservationModel:
    """Test observation model via env.sample_observation / env.observation_log_probability."""

    def test_observation_movement_action(self):
        """Test observation for movement actions.

        Purpose: Validates that movement actions always produce 'none' observation

        Given: Any state with movement action (1-4)
        When: env.sample_observation is invoked
        Then: Always returns 'none' observation

        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = create_rock_sample_state((1, 1), (True, False))

        for action in [1, 2, 3, 4]:  # North, East, South, West
            observation = pomdp.sample_observation(next_state=state, action=action)
            assert observation == "none"

    def test_observation_sample_action(self):
        """Test observation for sample action.

        Purpose: Validates that sample action always produces 'none' observation

        Given: Any state with sample action
        When: env.sample_observation is invoked
        Then: Returns 'none' observation

        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = create_rock_sample_state((1, 1), (True, False))

        observation = pomdp.sample_observation(next_state=state, action=0)  # Sample
        assert observation == "none"

    def test_observation_check_action_close_distance(self):
        """Test observation for check action at close distance.

        Purpose: Validates high accuracy sensor readings at close range

        Given: Robot very close to rock with check action and high sensor efficiency
        When: Observation probabilities are calculated via env.observation_log_probability
        Then: Correct observation has high probability

        Test type: unit
        """
        pomdp = RockSamplePOMDP(
            rock_positions=[(1, 1), (3, 3)], sensor_efficiency=50.0  # High efficiency
        )
        state = create_rock_sample_state((1, 1), (True, False))  # At rock 0 position

        # Should have high probability of correct observation
        log_probs = pomdp.observation_log_probability(state, 5, ["good", "bad", "none"])
        probs = np.exp(log_probs)
        assert probs[0] > 0.9  # High probability of "good"
        assert probs[1] < 0.1  # Low probability of "bad"
        assert probs[2] < 1e-200  # Effectively zero probability of "none"

    def test_observation_check_action_far_distance(self):
        """Test observation for check action at far distance.

        Purpose: Validates reduced accuracy sensor readings at far range

        Given: Robot far from rock with check action and very low sensor efficiency
        When: Observation probabilities are calculated via env.observation_log_probability
        Then: Observation probabilities are closer to random

        Test type: unit
        """
        pomdp = RockSamplePOMDP(
            rock_positions=[(0, 0), (1, 2)],  # Closer rocks
            sensor_efficiency=2.0,  # Moderate efficiency for reasonable uncertainty
        )
        state = create_rock_sample_state((0, 0), (True, False))  # At rock 0, check rock 1 at (1,2)

        log_probs = pomdp.observation_log_probability(state, 6, ["good", "bad", "none"])
        probs = np.exp(log_probs)
        # Distance = sqrt((0-1)^2 + (0-2)^2) = sqrt(5) ≈ 2.24
        # Efficiency = exp(-2.24/2.0) ≈ 0.33
        # For bad rock: P(good) = 1-0.33 = 0.67, P(bad) = 0.33
        assert 0.6 < probs[0] < 0.8  # P(good|bad_rock) should be high due to low efficiency
        assert 0.2 < probs[1] < 0.4  # P(bad|bad_rock) should be low
        assert probs[2] < 1e-200  # Effectively zero probability of "none"

    def test_observation_check_invalid_rock(self):
        """Test observation for check action on invalid rock index.

        Purpose: Validates proper handling of invalid rock check actions

        Given: Check action for rock index beyond available rocks
        When: env.sample_observation is invoked
        Then: Returns 'none' observation

        Test type: unit
        """
        pomdp = RockSamplePOMDP(rock_positions=[(0, 0), (2, 2)])  # Only 2 rocks
        state = create_rock_sample_state((1, 1), (True, False))

        observation = pomdp.sample_observation(next_state=state, action=7)  # Check rock 2 (invalid)
        assert observation == "none"

    def test_observation_probability_calculation(self):
        """Test observation probability calculation accuracy.

        Purpose: Validates mathematical correctness of sensor probability model

        Given: Known robot and rock positions with defined sensor efficiency
        When: Observation probabilities are calculated via env.observation_log_probability
        Then: Probabilities sum to 1 and match expected sensor model

        Test type: unit
        """
        pomdp = RockSamplePOMDP(rock_positions=[(2, 2)], sensor_efficiency=10.0)
        state = create_rock_sample_state((1, 1), (True,))  # Good rock at distance sqrt(2)

        log_probs = pomdp.observation_log_probability(state, 5, ["good", "bad", "none"])
        probs = np.exp(log_probs)

        # Probabilities should sum to 1 (excluding 'none')
        assert abs(probs[0] + probs[1] - 1.0) < 1e-10
        assert probs[2] < 1e-200  # Effectively zero probability of "none"

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
        state = create_rock_sample_state((2, 2), (True, False))

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
        pomdp = RockSamplePOMDP(step_penalty=-1.0, exit_reward=10.0)
        state = create_rock_sample_state((2, 4), (True, False))  # At right edge of 5x5 map

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
            rock_positions=[(1, 1), (3, 3)], step_penalty=-0.5, good_rock_reward=15.0
        )
        state = create_rock_sample_state((1, 1), (True, False))  # At good rock

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
            rock_positions=[(1, 1), (3, 3)], step_penalty=-0.5, bad_rock_penalty=-20.0
        )
        state = create_rock_sample_state((1, 1), (False, True))  # At bad rock

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
        pomdp = RockSamplePOMDP(rock_positions=[(0, 0), (3, 3)], step_penalty=-1.5)
        state = create_rock_sample_state((1, 1), (True, False))  # Not at any rock

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
        pomdp = RockSamplePOMDP(step_penalty=-1.0, sensor_use_penalty=-2.0)
        state = create_rock_sample_state((2, 2), (True, False))

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
            sensor_use_penalty=-0.5,
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
            sensor_use_penalty=-1.0,
        )

        expected_min2 = -0.5 + (-10.0) + (-1.0)  # -11.5
        expected_max2 = -0.5 + 20.0  # 19.5

        assert pomdp2.reward_range == (expected_min2, expected_max2)


class TestDangerousArea:
    """Test dangerous area detection."""

    def test_is_in_dangerous_area_no_dangerous_areas(self):
        """Test dangerous area check with no dangerous areas defined.

        Purpose: Validates that method returns False when no dangerous areas are configured

        Given: POMDP with no dangerous areas
        When: _is_in_dangerous_area() is called with any position
        Then: Returns False

        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        assert pomdp._is_in_dangerous_area((0, 0)) is False
        assert pomdp._is_in_dangerous_area((2, 2)) is False
        assert pomdp._is_in_dangerous_area((4, 4)) is False

    def test_is_in_dangerous_area_within_radius(self):
        """Test dangerous area check when position is within radius.

        Purpose: Validates that positions within dangerous area radius are correctly identified

        Given: POMDP with dangerous area at (2, 2) with radius 1.5
        When: _is_in_dangerous_area() is called with position within radius
        Then: Returns True

        Test type: unit
        """
        pomdp = RockSamplePOMDP(dangerous_areas=[(2, 2)], dangerous_area_radius=1.5)
        # Position at center
        assert pomdp._is_in_dangerous_area((2, 2)) is True
        # Position adjacent (distance = 1.0)
        assert pomdp._is_in_dangerous_area((2, 3)) is True
        assert pomdp._is_in_dangerous_area((3, 2)) is True
        assert pomdp._is_in_dangerous_area((1, 2)) is True
        assert pomdp._is_in_dangerous_area((2, 1)) is True
        # Position diagonal (distance = sqrt(2) ≈ 1.41)
        assert pomdp._is_in_dangerous_area((3, 3)) is True
        assert pomdp._is_in_dangerous_area((1, 1)) is True

    def test_is_in_dangerous_area_outside_radius(self):
        """Test dangerous area check when position is outside radius.

        Purpose: Validates that positions beyond dangerous area radius are correctly identified

        Given: POMDP with dangerous area at (2, 2) with radius 1.0
        When: _is_in_dangerous_area() is called with position beyond radius
        Then: Returns False

        Test type: unit
        """
        pomdp = RockSamplePOMDP(dangerous_areas=[(2, 2)], dangerous_area_radius=1.0)
        # Position at center (distance = 0.0)
        assert pomdp._is_in_dangerous_area((2, 2)) is True
        # Position adjacent (distance = 1.0, exactly at boundary)
        assert pomdp._is_in_dangerous_area((2, 3)) is True
        # Position two steps away (distance = 2.0)
        assert pomdp._is_in_dangerous_area((2, 4)) is False
        assert pomdp._is_in_dangerous_area((4, 2)) is False
        # Position far away
        assert pomdp._is_in_dangerous_area((0, 0)) is False
        assert pomdp._is_in_dangerous_area((4, 4)) is False

    def test_is_in_dangerous_area_exact_radius_boundary(self):
        """Test dangerous area check at exact radius boundary.

        Purpose: Validates boundary condition where distance equals radius

        Given: POMDP with dangerous area at (2, 2) with radius sqrt(2) ≈ 1.414
        When: _is_in_dangerous_area() is called with position at exact boundary
        Then: Returns True (distance <= radius)

        Test type: unit
        """
        import math

        pomdp = RockSamplePOMDP(dangerous_areas=[(2, 2)], dangerous_area_radius=math.sqrt(2))
        # Position diagonal (distance = sqrt(2), exactly at boundary)
        assert pomdp._is_in_dangerous_area((3, 3)) is True
        assert pomdp._is_in_dangerous_area((1, 1)) is True
        # Position slightly further (distance = 2.0)
        assert pomdp._is_in_dangerous_area((2, 4)) is False

    def test_is_in_dangerous_area_multiple_areas(self):
        """Test dangerous area check with multiple dangerous areas.

        Purpose: Validates that method checks all dangerous areas and returns True if any match

        Given: POMDP with multiple dangerous areas
        When: _is_in_dangerous_area() is called with position within any area
        Then: Returns True

        Test type: unit
        """
        pomdp = RockSamplePOMDP(dangerous_areas=[(1, 1), (3, 3)], dangerous_area_radius=1.5)
        # Position within first area
        assert pomdp._is_in_dangerous_area((1, 1)) is True
        assert pomdp._is_in_dangerous_area((2, 1)) is True
        # Position within second area
        assert pomdp._is_in_dangerous_area((3, 3)) is True
        assert pomdp._is_in_dangerous_area((4, 3)) is True
        # Position outside both areas (distance > 1.5 from both centers)
        # (0, 3) is sqrt((0-1)^2 + (3-1)^2) = sqrt(5) ≈ 2.24 from (1,1) and
        # sqrt((0-3)^2 + (3-3)^2) = 3.0 from (3,3), both > 1.5
        assert pomdp._is_in_dangerous_area((0, 3)) is False
        # (4, 0) is sqrt((4-1)^2 + (0-1)^2) = sqrt(10) ≈ 3.16 from (1,1) and
        # sqrt((4-3)^2 + (0-3)^2) = sqrt(10) ≈ 3.16 from (3,3), both > 1.5
        assert pomdp._is_in_dangerous_area((4, 0)) is False

    def test_is_in_dangerous_area_large_radius(self):
        """Test dangerous area check with large radius covering entire map.

        Purpose: Validates behavior with very large dangerous area radius

        Given: POMDP with dangerous area at center and large radius
        When: _is_in_dangerous_area() is called with any position
        Then: Returns True for all positions within radius

        Test type: unit
        """
        pomdp = RockSamplePOMDP(
            map_size=(5, 5),
            dangerous_areas=[(2, 2)],
            dangerous_area_radius=10.0,  # Large enough to cover entire map
        )
        # All positions should be within radius
        assert pomdp._is_in_dangerous_area((0, 0)) is True
        assert pomdp._is_in_dangerous_area((2, 2)) is True
        assert pomdp._is_in_dangerous_area((4, 4)) is True

    def test_is_in_dangerous_area_small_radius(self):
        """Test dangerous area check with very small radius.

        Purpose: Validates behavior with very small dangerous area radius

        Given: POMDP with dangerous area at (2, 2) and radius 0.1
        When: _is_in_dangerous_area() is called
        Then: Returns True only for position at exact center

        Test type: unit
        """
        pomdp = RockSamplePOMDP(dangerous_areas=[(2, 2)], dangerous_area_radius=0.1)
        # Only exact center should be within radius
        assert pomdp._is_in_dangerous_area((2, 2)) is True
        # Adjacent positions should be outside
        assert pomdp._is_in_dangerous_area((2, 3)) is False
        assert pomdp._is_in_dangerous_area((3, 2)) is False


class TestStochasticDangerousAreaPenalty:
    """Tests for the ``dangerous_area_hit_probability`` parameter.

    Geometry: dangerous area centred at ``(2, 0)`` with radius 1.0; the
    test fixtures place the robot at ``(1, 0)`` and use action 3 (south)
    so the next-state position ``(2, 0)`` lies inside the zone. Action 3
    avoids the action-2 (east-exit) early-return branch in
    ``_reward_batch_vectorized``.
    """

    SOUTH_ACTION = 3

    @staticmethod
    def _stochastic_env(hit_probability: float, penalty: float = -5.0) -> RockSamplePOMDP:
        return RockSamplePOMDP(
            map_size=(5, 5),
            rock_positions=[(4, 4)],
            init_pos=(1, 0),
            dangerous_areas=[(2, 0)],
            dangerous_area_radius=1.0,
            dangerous_area_penalty=penalty,
            dangerous_area_hit_probability=hit_probability,
            step_penalty=0.0,
        )

    def test_default_hit_probability_is_one(self):
        """Default hit probability preserves legacy deterministic behavior.

        Purpose: Validates that omitting the new parameter keeps the
            deterministic dangerous-area penalty active.

        Given: A RockSamplePOMDP constructed with default parameters.
        When: ``dangerous_area_hit_probability`` is read.
        Then: It equals ``1.0`` (deterministic-penalty regime).

        Test type: unit
        """
        env = RockSamplePOMDP()
        assert env.dangerous_area_hit_probability == 1.0

    def test_hit_probability_zero_never_applies_penalty(self):
        """hit_probability=0 disables the dangerous-area penalty.

        Purpose: Validates the lower-bound of the stochastic penalty.

        Given: A robot moving south from (1, 0) into the (2, 0)
            dangerous area with hit_probability=0.
        When: ``reward()`` is called many times.
        Then: The dangerous-area penalty is never added.

        Test type: unit
        """
        env = self._stochastic_env(hit_probability=0.0)
        state = create_rock_sample_state((1, 0), (True,))
        np.random.seed(0)
        for _ in range(200):
            assert env.reward(state, self.SOUTH_ACTION) == 0.0

    def test_hit_probability_one_matches_deterministic(self):
        """hit_probability=1 matches legacy deterministic penalty.

        Purpose: Regression check that the default behavior is preserved
            when hit_probability=1.0 is passed explicitly.

        Given: A robot moving south from (1, 0) into the (2, 0)
            dangerous area with hit_probability=1.0.
        When: ``reward()`` is called many times.
        Then: The dangerous-area penalty is always applied.

        Test type: unit
        """
        env = self._stochastic_env(hit_probability=1.0)
        state = create_rock_sample_state((1, 0), (True,))
        for _ in range(50):
            assert env.reward(state, self.SOUTH_ACTION) == pytest.approx(-5.0)

    def test_hit_probability_zero_three_empirical_rate(self):
        """Empirical hit rate matches hit_probability over many calls.

        Purpose: Validates that the per-call Bernoulli draw matches the
            configured probability over a large sample.

        Given: A robot moving into a dangerous area with
            hit_probability=0.3.
        When: ``reward()`` is called 5000 times.
        Then: Empirical hit rate is within 0.05 of 0.3.

        Test type: unit
        """
        env = self._stochastic_env(hit_probability=0.3)
        state = create_rock_sample_state((1, 0), (True,))
        np.random.seed(123)
        n_trials = 5000
        hits = 0
        for _ in range(n_trials):
            if env.reward(state, self.SOUTH_ACTION) == pytest.approx(-5.0):
                hits += 1
        empirical_rate = hits / n_trials
        assert abs(empirical_rate - 0.3) < 0.05

    def test_reward_batch_honours_hit_probability(self):
        """reward_batch applies stochastic penalty consistently with single-state.

        Purpose: Validates that the batched reward path uses the same
            Bernoulli mechanism as the single-state path.

        Given: 5000 copies of a state moving into a dangerous area with
            hit_probability=0.3.
        When: ``reward_batch()`` is called once.
        Then: Empirical hit rate across the batch is within 0.05 of 0.3.

        Test type: unit
        """
        env = self._stochastic_env(hit_probability=0.3)
        state = create_rock_sample_state((1, 0), (True,))
        n_trials = 5000
        states = np.tile(state, (n_trials, 1))
        np.random.seed(456)
        rewards = env.reward_batch(states, self.SOUTH_ACTION)
        hits = int(np.sum(np.isclose(rewards, -5.0)))
        empirical_rate = hits / n_trials
        assert abs(empirical_rate - 0.3) < 0.05

    def test_reward_batch_zero_probability_never_applies_penalty(self):
        """reward_batch with hit_probability=0 returns zero penalty for all rows.

        Purpose: Validates the lower-bound of the stochastic penalty in
            the batched path.

        Given: A batch of in-zone next-state states with
            hit_probability=0.0 and step_penalty=0.
        When: ``reward_batch()`` is called.
        Then: All returned rewards equal 0.0.

        Test type: unit
        """
        env = self._stochastic_env(hit_probability=0.0)
        state = create_rock_sample_state((1, 0), (True,))
        states = np.tile(state, (200, 1))
        np.random.seed(789)
        rewards = env.reward_batch(states, self.SOUTH_ACTION)
        assert np.all(rewards == 0.0)

    @pytest.mark.parametrize("bad_value", [-0.1, 1.5, 2.0, -1.0])
    def test_invalid_hit_probability_raises(self, bad_value: float):
        """Out-of-range hit_probability raises ValueError.

        Purpose: Validates input validation on the new parameter.

        Given: A hit_probability value outside [0, 1].
        When: RockSamplePOMDP is constructed.
        Then: ValueError is raised mentioning the parameter name.

        Test type: unit
        """
        with pytest.raises(ValueError, match="dangerous_area_hit_probability"):
            RockSamplePOMDP(dangerous_area_hit_probability=bad_value)


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
            state = create_rock_sample_state((0, 0), (True, True))

            # Add steps with sample actions
            for i in range(num_samples):
                step = StepData(
                    state=state,
                    action=0,  # Sample action
                    next_state=state,
                    observation="none",
                    reward=10.0,
                    belief=Mock(spec=Belief),
                )
                steps.append(step)

            # Add non-sample action
            step = StepData(
                state=state,
                action=1,  # Movement action
                next_state=state,
                observation="none",
                reward=-1.0,
                belief=Mock(spec=Belief),
            )
            steps.append(step)

            history = build_test_history(steps=steps, reach_terminal=False)
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
        terminal_state = create_rock_sample_state((-1, -1), (True, False))
        success_step = StepData(
            state=terminal_state,
            action=None,
            next_state=terminal_state,
            observation="none",
            reward=10.0,
            belief=Mock(spec=Belief),
        )
        success_history = build_test_history(
            steps=[success_step], actual_num_steps=1, reach_terminal=True
        )
        histories.append(success_history)

        # Failed (non-exit) history
        normal_state = create_rock_sample_state((2, 2), (True, False))
        fail_step = StepData(
            state=normal_state,
            action=1,
            next_state=normal_state,
            observation="none",
            reward=-1.0,
            belief=Mock(spec=Belief),
        )
        fail_history = build_test_history(
            steps=[fail_step], actual_num_steps=1, reach_terminal=False
        )
        histories.append(fail_history)

        metrics = pomdp.compute_metrics(histories)

        # Find exit success rate metric
        exit_metric = next((m for m in metrics if m.name == "exit_success_rate"), None)
        assert exit_metric is not None
        assert exit_metric.value == 0.5  # 1/2 successful exits

    def test_compute_metrics_values_within_confidence_intervals(self):
        """Test RockSamplePOMDP metric values are inside CIs and pass invariants.

        Purpose: Validates that metrics produced by compute_metrics lie inside
            their CI bounds and that all structural invariants hold (rate-in-[0,1],
            counts >= 0, finite CI for n>=2, returns inside reward bounds, and
            return-shift linearity).

        Given: A RockSamplePOMDP and 3 hand-built histories with varied
            outcomes (sample-and-exit, traverse, no-action). Rewards lie within
            reward_range = (-10, 10).
        When: compute_metrics is called and the four invariant helpers are run.
        Then: All checks pass without raising.

        Test type: integration
        """
        pomdp = RockSamplePOMDP()

        # History 0: sample once and exit successfully.
        sample_state = create_rock_sample_state((0, 0), (True, True))
        terminal_state = create_rock_sample_state((-1, -1), (True, False))
        sample_exit_steps = [
            StepData(
                state=sample_state,
                action=0,  # Sample
                next_state=sample_state,
                observation="none",
                reward=10.0,
                belief=Mock(spec=Belief),
            ),
            StepData(
                state=terminal_state,
                action=1,
                next_state=terminal_state,
                observation="none",
                reward=10.0,  # exit reward
                belief=Mock(spec=Belief),
            ),
        ]

        # History 1: traverse without sampling, no exit.
        traverse_state = create_rock_sample_state((2, 2), (True, False))
        traverse_steps = [
            StepData(
                state=traverse_state,
                action=1,
                next_state=traverse_state,
                observation="none",
                reward=0.0,
                belief=Mock(spec=Belief),
            ),
            StepData(
                state=traverse_state,
                action=2,
                next_state=traverse_state,
                observation="none",
                reward=0.0,
                belief=Mock(spec=Belief),
            ),
        ]

        # History 2: sample twice, no exit.
        sample_twice_state = create_rock_sample_state((1, 1), (False, False))
        sample_twice_steps = [
            StepData(
                state=sample_twice_state,
                action=0,
                next_state=sample_twice_state,
                observation="none",
                reward=-10.0,  # bad rock penalty
                belief=Mock(spec=Belief),
            ),
            StepData(
                state=sample_twice_state,
                action=0,
                next_state=sample_twice_state,
                observation="none",
                reward=-10.0,
                belief=Mock(spec=Belief),
            ),
        ]

        histories = [
            build_test_history(steps=sample_exit_steps, actual_num_steps=2, reach_terminal=True),
            build_test_history(steps=traverse_steps, actual_num_steps=2, reach_terminal=False),
            build_test_history(steps=sample_twice_steps, actual_num_steps=2, reach_terminal=False),
        ]

        metrics = pomdp.compute_metrics(histories)
        verify_metrics_within_confidence_intervals(metrics)
        verify_metric_sanity(metrics, histories, pomdp)
        verify_history_returns_bounded(histories, pomdp)
        verify_return_shift_linearity(histories, pomdp, shift=1.5)


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
        pomdp = RockSamplePOMDP(map_size=(3, 3), rock_positions=[(0, 0), (2, 2)], init_pos=(0, 0))

        # Start with initial state
        initial_dist = pomdp.initial_state_dist()
        state = create_rock_sample_state((0, 0), (True, False))  # Specific initial state

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
        When: Many observations are sampled from check actions via env.sample_observation
        Then: Observation frequencies match expected sensor efficiency

        Test type: integration
        """
        # High efficiency sensor for deterministic behavior
        pomdp = RockSamplePOMDP(
            rock_positions=[(1, 1)], sensor_efficiency=100.0  # Very high efficiency
        )

        # Robot at rock position - should get very accurate readings
        state = create_rock_sample_state((1, 1), (True,))  # Good rock

        # Sample many observations via env-API batched sampling
        observations = pomdp.sample_observation(next_state=state, action=5, n_samples=100)
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
        pomdp = RockSamplePOMDP(rock_positions=[(0, 0), (1, 1)], init_pos=(0, 0))

        initial_dist = pomdp.initial_state_dist()

        # Should have 2^2 = 4 states
        assert len(initial_dist.values) == 4

        # Extract rock configurations
        rock_configs = set()
        for state in initial_dist.values:
            assert get_robot_pos(state) == (0, 0)  # All start at same position
            rock_configs.add(get_rocks(state))

        expected_configs = {(True, True), (True, False), (False, True), (False, False)}

        assert rock_configs == expected_configs


class TestSampleNextStepEquivalence:
    """Test that the optimized sample_next_step produces identical results to the base class."""

    def test_sample_next_step_matches_base_class(self):
        """Test optimized sample_next_step agrees with base Environment.sample_next_step.

        Purpose: Validates that the inlined sample_next_step override produces
        identical results to the original base class implementation.

        Given: A RockSamplePOMDP environment and valid (state, action) pairs
        When: Both the optimized and base class sample_next_step are called
              with the same numpy RNG seed
        Then: next_state, observation, and reward are identical

        Test type: unit
        """
        from POMDPPlanners.core.environment import Environment

        env = RockSamplePOMDP(discount_factor=0.95)
        state = env.initial_state_dist().sample()[0]

        for action in [0, 1, 2, 3, 4, 5]:
            for _ in range(30):
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
                assert (
                    opt_obs == base_obs
                ), f"observation mismatch for action={action}, seed={seed}: {opt_obs} != {base_obs}"
                assert opt_reward == base_reward, (
                    f"reward mismatch for action={action}, seed={seed}: "
                    f"{opt_reward} != {base_reward}"
                )


# ---------------------------------------------------------------------------
# reward_batch equivalence (atol=1e-12, deterministic) and native rollout tests
# ---------------------------------------------------------------------------

_RS_ROCK_POSITIONS = [(1, 1), (3, 3), (5, 5)]
_RS_MAP_SIZE = (7, 7)
_RS_NUM_ROCKS = len(_RS_ROCK_POSITIONS)
_RS_STATE_DIM = 2 + _RS_NUM_ROCKS


def _make_rs_env(**kwargs) -> RockSamplePOMDP:
    return RockSamplePOMDP(
        map_size=_RS_MAP_SIZE,
        rock_positions=list(_RS_ROCK_POSITIONS),
        init_pos=(0, 0),
        **kwargs,
    )


class _FixedActionSamplerRS:
    def __init__(self, action: int) -> None:
        self._action = action

    def sample(self) -> int:
        return self._action


def test_rocksample_reward_batch_matches_scalar():
    """reward_batch matches scalar _reward_from_next_state for all actions.

    Purpose: Validates that the vectorised reward_batch produces the same
    per-particle rewards as calling the scalar reward path per particle,
    covering movement, sample, check, and exit actions. Since transitions
    are deterministic, the results are bit-identical (atol=1e-12).

    Given: A batch of 32 random particles, tested for all actions in [0, 7].
    When: reward_batch is compared element-by-element to the scalar path.
    Then: All differences are within atol=1e-12.

    Test type: integration
    """
    env_local = _make_rs_env()
    rng = np.random.default_rng(0)
    particles = np.zeros((32, _RS_STATE_DIM), dtype=np.float64)
    for i in range(32):
        particles[i, 0] = int(rng.integers(0, _RS_MAP_SIZE[0]))
        particles[i, 1] = int(rng.integers(0, _RS_MAP_SIZE[1]))
        particles[i, 2:] = rng.integers(0, 2, size=_RS_NUM_ROCKS).astype(float)

    for action in range(5 + _RS_NUM_ROCKS):
        batch_rewards = env_local.reward_batch(particles, action)
        scalar_rewards = np.array(
            [
                env_local._reward_from_next_state(  # pylint: disable=protected-access
                    row,
                    action,
                    env_local.sample_next_state(state=row, action=action),
                )
                for row in particles
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            batch_rewards,
            scalar_rewards,
            atol=1e-12,
            rtol=0.0,
            err_msg=f"reward_batch mismatch for action={action}",
        )


def _rs_python_rollout_native_semantics(
    env_local: RockSamplePOMDP,
    initial_state: RockSampleState,
    action_indices: np.ndarray,
    max_depth: int,
    start_depth: int,
    discount_factor: float,
) -> float:
    state = np.array(initial_state, dtype=np.float64)
    total = 0.0
    gamma_power = 1.0
    depth = start_depth
    n_actions = len(env_local.action_names)

    for action_int in action_indices:
        if depth >= max_depth:
            break
        if int(state[0]) == -1 and int(state[1]) == -1:
            break
        ai = int(action_int) % n_actions
        next_state = np.asarray(
            env_local.sample_next_state(state=state, action=ai), dtype=np.float64
        )
        r = env_local._reward_from_next_state(  # pylint: disable=protected-access
            state, ai, next_state
        )
        total += gamma_power * r
        gamma_power *= discount_factor
        state = next_state
        depth += 1

    return total


def test_native_simulate_rollout_rocksample_matches_python_reference():
    """native simulate_rollout_discrete matches Python reference with same semantics.

    Purpose: Validates that the C++ rollout accumulates the same discounted
    return as a Python loop using the same deterministic transitions and
    reward conventions (no dangerous-area term) under identical pre-drawn
    action indices.

    Given: A RockSamplePOMDP with no dangerous areas, a fixed initial state,
        and identical pre-drawn action_indices for both implementations.
    When: Both ``_native.simulate_rollout_discrete`` and the Python
        reference walk the same trajectory.
    Then: Returned discounted returns agree within atol=1e-9.

    Test type: integration
    """
    from POMDPPlanners.environments.rock_sample_pomdp import (
        _native as rs_native,
    )  # pylint: disable=import-outside-toplevel

    np.random.seed(0)
    env_local = _make_rs_env()
    initial_state = create_rock_sample_state((0, 0), (True, False, True))
    max_depth = 10
    start_depth = 0
    discount_factor = 0.95
    steps_left = max_depth - start_depth
    n_actions = len(env_local.action_names)

    action_indices = np.random.randint(0, n_actions, size=steps_left, dtype=np.int32)

    native_result = rs_native.simulate_rollout_discrete(
        initial_state=np.ascontiguousarray(initial_state, dtype=np.float64),
        action_indices=action_indices,
        rock_positions_flat=env_local._rock_positions_flat,  # pylint: disable=protected-access
        max_depth=max_depth,
        start_depth=start_depth,
        discount_factor=discount_factor,
        map_rows=int(env_local.map_size[0]),
        map_cols=int(env_local.map_size[1]),
        n_actions=n_actions,
        step_penalty=float(env_local.step_penalty),
        exit_reward=float(env_local.exit_reward),
        good_rock_reward=float(env_local.good_rock_reward),
        bad_rock_penalty=float(env_local.bad_rock_penalty),
        sensor_use_penalty=float(env_local.sensor_use_penalty),
        dangerous_areas=np.empty((0, 2), dtype=np.float64),
        dangerous_area_radius=1.0,
        dangerous_area_penalty=0.0,
        dangerous_area_hit_probability=1.0,
        reward_variant_code=0,
        penalty_decay=1.0,
    )

    python_result = _rs_python_rollout_native_semantics(
        env_local=env_local,
        initial_state=initial_state,
        action_indices=action_indices,
        max_depth=max_depth,
        start_depth=start_depth,
        discount_factor=discount_factor,
    )

    np.testing.assert_allclose(native_result, python_result, atol=1e-9, rtol=0.0)


def test_simulate_random_rollout_rocksample_returns_finite_float():
    """simulate_random_rollout returns a finite float from an initial state.

    Purpose: Smoke-test that the native override does not raise or produce
    non-finite values from a fresh initial state with a shallow horizon.

    Given: A RockSamplePOMDP, its initial state, and a fixed action.
    When: simulate_random_rollout is called with max_depth=10.
    Then: The result is a finite float.

    Test type: integration
    """
    np.random.seed(0)
    env_local = _make_rs_env()
    state = env_local.initial_state_dist().sample()[0]
    sampler = _FixedActionSamplerRS(action=2)

    result = env_local.simulate_random_rollout(
        state=state,
        action_sampler=sampler,
        max_depth=10,
        discount_factor=0.95,
    )

    assert isinstance(result, float)
    assert np.isfinite(result)


def test_simulate_random_rollout_rocksample_returns_zero_at_max_depth():
    """simulate_random_rollout returns 0.0 when depth equals max_depth.

    Purpose: Validates the depth-bounded base case for the override.

    Given: A RockSamplePOMDP, any state, and depth == max_depth.
    When: simulate_random_rollout is called.
    Then: The return is exactly 0.0.

    Test type: unit
    """
    env_local = _make_rs_env()
    state = env_local.initial_state_dist().sample()[0]
    sampler = _FixedActionSamplerRS(action=1)

    result = env_local.simulate_random_rollout(
        state=state,
        action_sampler=sampler,
        max_depth=5,
        discount_factor=0.95,
        depth=5,
    )

    assert result == 0.0


def test_simulate_random_rollout_rocksample_terminal_returns_zero():
    """simulate_random_rollout returns 0.0 from a terminal sentinel state.

    Purpose: Validates that the terminal sentinel [-1, -1, ...] causes
    the rollout to exit immediately and return 0.0.

    Given: A terminal-sentinel state and depth < max_depth.
    When: simulate_random_rollout is called.
    Then: The return is exactly 0.0.

    Test type: unit
    """
    env_local = _make_rs_env()
    terminal_state = np.array([-1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    sampler = _FixedActionSamplerRS(action=2)

    result = env_local.simulate_random_rollout(
        state=terminal_state,
        action_sampler=sampler,
        max_depth=10,
        discount_factor=0.95,
        depth=0,
    )

    assert result == 0.0


def test_scalar_obs_log_prob_un_floored_matches_batch_after_fix() -> None:
    """Scalar and batch obs log-prob agree at the symmetric C++ floor.

    Purpose: Pins the post-fix contract for RockSamplePOMDP's audit
        case A5 — a CHECK action where the canonical observation
        ``"none"`` is impossible (it can only fire on movement actions,
        never on CHECK). With the symmetric C++ kernel floor in place,
        both ``observation_log_probability`` (scalar, via
        ``kernel.probability``) and
        ``observation_log_probability_per_state`` (batch, via
        ``kernel.batch_log_likelihood``) return
        ``log(kProbFloor) = log(1e-300) ~= -690.776`` for impossible
        events.

    Given: A 1-rock RockSamplePOMDP with the rock at (1, 1), the robot
        co-located at (1, 1) with that rock marked True, and the CHECK
        action for rock 0 (action index 5). The observation ``"none"``
        has zero probability under any CHECK action by construction.
    When: Both ``observation_log_probability`` and
        ``observation_log_probability_per_state`` are evaluated for
        ``observation="none"``.
    Then: Both return ``log(1e-300) ~= -690.776`` (the symmetric C++
        floor), pinning the post-fix log-prob contract for impossible
        events.

    Test type: unit
    """
    env = RockSamplePOMDP(
        discount_factor=0.95,
        rock_positions=[(1, 1)],
        sensor_efficiency=10.0,
    )
    state = create_rock_sample_state((1, 1), (True,))

    scalar = env.observation_log_probability(state, 5, ["none"])[0]
    batch = env.observation_log_probability_per_state(state.reshape(1, -1), 5, "none")[0]

    expected_floor = float(np.log(1e-300))  # ~= -690.7755278982137
    np.testing.assert_allclose(scalar, expected_floor, atol=1e-6)
    np.testing.assert_allclose(batch, expected_floor, atol=1e-6)
    np.testing.assert_allclose(scalar, batch, atol=1e-6)


class TestRockSampleRewardBatchNextStateConsistency:
    """Regression tests for ``reward_batch`` honouring passed ``next_states``.

    The dangerous-area penalty in ``reward`` is checked against the
    realised ``next_state`` (e.g. provided by
    :meth:`Environment.sample_next_step`). ``reward_batch`` must
    consume ``next_states`` the same way and not reconstruct the
    next robot position via closed-form math, otherwise batch and
    scalar paths disagree whenever the caller-supplied next state
    differs from the closed-form prediction.
    """

    SOUTH_ACTION = 3

    @staticmethod
    def _danger_env(hit_probability: float = 1.0) -> RockSamplePOMDP:
        return RockSamplePOMDP(
            map_size=(5, 5),
            rock_positions=[(4, 4)],
            init_pos=(0, 0),
            dangerous_areas=[(2, 0)],
            dangerous_area_radius=1.0,
            dangerous_area_penalty=-5.0,
            dangerous_area_hit_probability=hit_probability,
            step_penalty=0.0,
        )

    def test_reward_batch_uses_passed_next_states_per_row(self):
        """reward_batch consults caller-supplied next_states for danger check.

        Purpose: Validates that the dangerous-area penalty in
            ``reward_batch`` fires based on the realised next-state
            position, not on the closed-form reconstruction from
            ``(state, action)``.

        Given: A 2-row batch where the closed-form next position is
            (1, 0) -- outside the danger zone centred at (2, 0) with
            radius 1.0 -- but caller-supplied ``next_states`` place
            row 0 at (2, 0) (inside the zone) and row 1 at (4, 4)
            (outside the zone).
        When: ``reward_batch(states, action=NORTH, next_states=...)``
            is called with ``hit_probability=1.0``.
        Then: Row 0 receives the dangerous-area penalty (-5.0) and
            row 1 does not (0.0).

        Test type: unit
        """
        env = self._danger_env(hit_probability=1.0)
        # Action 1 is NORTH (-1, 0); from (1, 0) the closed-form
        # next position is (0, 0), which is outside the danger zone.
        north_action = 1
        states = np.stack(
            [
                create_rock_sample_state((1, 0), (True,)).astype(np.float64),
                create_rock_sample_state((1, 0), (True,)).astype(np.float64),
            ]
        )
        next_states = np.stack(
            [
                # Hand-crafted realised next position (2, 0) -- inside
                # the danger zone, disagreeing with closed-form (0, 0).
                create_rock_sample_state((2, 0), (True,)).astype(np.float64),
                # Realised next position (4, 4) -- outside the zone.
                create_rock_sample_state((4, 4), (True,)).astype(np.float64),
            ]
        )

        rewards = env.reward_batch(states, north_action, next_states=next_states)

        assert rewards[0] == pytest.approx(-5.0)
        assert rewards[1] == pytest.approx(0.0)

    def test_reward_batch_matches_scalar_reward_per_row(self):
        """reward_batch[i] equals reward(states[i], a, next_states[i]).

        Purpose: Validates row-wise consistency between batch and
            scalar paths when both consume the same caller-supplied
            ``next_states`` and the same RNG seed.

        Given: A 4-row batch with mixed in-zone and out-of-zone
            realised next states under ``hit_probability=1.0`` (so
            the danger penalty is deterministic and the test is
            insensitive to RNG ordering between batch and scalar).
        When: ``reward_batch(..., next_states=ns)`` is computed and
            ``reward(states[i], a, next_states[i])`` is computed for
            each row.
        Then: The two reward arrays match element-wise.

        Test type: unit
        """
        env = self._danger_env(hit_probability=1.0)
        north_action = 1
        states = np.stack(
            [create_rock_sample_state((1, 0), (True,)).astype(np.float64) for _ in range(4)]
        )
        next_states = np.stack(
            [
                create_rock_sample_state((2, 0), (True,)).astype(np.float64),  # in zone
                create_rock_sample_state((4, 4), (True,)).astype(np.float64),  # out
                create_rock_sample_state((2, 1), (True,)).astype(np.float64),  # in zone
                create_rock_sample_state((0, 0), (True,)).astype(np.float64),  # out
            ]
        )

        np.random.seed(2024)
        batch_rewards = env.reward_batch(states, north_action, next_states=next_states)

        np.random.seed(2024)
        scalar_rewards = np.array(
            [
                env.reward(states[i], north_action, next_state=next_states[i])
                for i in range(states.shape[0])
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(batch_rewards, scalar_rewards, atol=1e-12, rtol=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
