"""Tests for LaserTag POMDP environment.

This module tests the LaserTag POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

import random
from collections import Counter

import numpy as np
import pytest

from typing import List

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.policy import Policy, PolicyRunData, PolicySpaceInfo
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.laser_tag_pomdp import LaserTagPOMDP
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.simulations.episodes import run_episode
from POMDPPlanners.tests.test_utils.confidence_interval_utils import (
    verify_metrics_within_confidence_intervals,
)
from POMDPPlanners.utils.logger import get_logger

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


class RandomPolicy(Policy):
    """Simple random policy for testing purposes."""

    def __init__(self, environment, name="RandomPolicy"):
        """Initialize random policy."""
        super().__init__(
            environment=environment,
            discount_factor=environment.discount_factor,
            name=name,
        )
        self.actions = environment.get_actions()

    def action(self, belief):
        """Return random action."""
        action = np.random.choice(self.actions)
        return [action], PolicyRunData(info_variables=[])

    @classmethod
    def get_space_info(cls):
        """Return space info from environment."""
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.CONTINUOUS
        )

    @classmethod
    def get_info_variable_names(cls) -> List[str]:
        """Return info variable names (none for random policy)."""
        return []


class TestLaserTagState:
    """Test LaserTagState representation and methods.

    Purpose: Validates LaserTag state representation and equality operations

    Given: LaserTagState instances with robot/opponent positions and terminal flags
    When: States are created, compared, and hashed
    Then: State operations work correctly with proper equality and hashing

    Test type: unit
    """

    def test_state_creation_and_equality(self):
        """Test LaserTag state creation and equality comparison.

        Purpose: Validates state creation and equality comparison functionality

        Given: Two identical LaserTag states and one different state as numpy arrays
        When: States are created and compared using np.array_equal
        Then: Identical states are equal and different states are not equal

        Test type: unit
        """
        state1 = np.array([3.0, 5.0, 2.0, 4.0, 0.0])
        state2 = np.array([3.0, 5.0, 2.0, 4.0, 0.0])
        state3 = np.array([3.0, 5.0, 2.0, 4.0, 1.0])

        assert np.array_equal(state1, state2)
        assert not np.array_equal(state1, state3)

    def test_state_structure(self):
        """Test LaserTag state vector structure.

        Purpose: Validates state vector structure and element access

        Given: A LaserTag state as numpy array
        When: Elements are accessed by index
        Then: Correct values are returned for robot pos, opponent pos, and terminal flag

        Test type: unit
        """
        state = np.array([3.0, 5.0, 2.0, 4.0, 0.0])

        # Test robot position access
        assert int(state[0]) == 3
        assert int(state[1]) == 5

        # Test opponent position access
        assert int(state[2]) == 2
        assert int(state[3]) == 4

        # Test terminal flag
        assert bool(state[4]) == False


def _make_env(
    *,
    floor_shape=(7, 11),
    walls=None,
    transition_error_prob=0.0,
    measurement_noise=1.0,
):
    """Build a LaserTagPOMDP with empty dangerous areas for transition/observation tests."""
    return LaserTagPOMDP(
        discount_factor=0.95,
        floor_shape=floor_shape,
        walls=walls if walls is not None else set(),
        dangerous_areas=set(),
        measurement_noise=measurement_noise,
        transition_error_prob=transition_error_prob,
    )


def _transition_probabilities(env, state, action, next_states):
    """Convenience: env.transition_log_probability -> probability list."""
    log_probs = env.transition_log_probability(state, action, next_states)
    return np.exp(log_probs)


def _observation_probabilities(env, next_state, action, observations):
    """Convenience: env.observation_log_probability -> probability list."""
    log_probs = env.observation_log_probability(next_state, action, observations)
    return np.exp(log_probs)


class TestLaserTagStateTransition:
    """Test LaserTag state transition dynamics via the env API.

    Purpose: Validates state transition dynamics including robot movement and opponent behavior

    Given: LaserTagPOMDP environments with various floors, walls, and actions
    When: Transitions are sampled via env.sample_next_state and probabilities are
        evaluated via env.transition_log_probability
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
        env = _make_env(floor_shape=(7, 11))
        state = np.array([3.0, 5.0, 1.0, 1.0, 0.0])

        next_states = env.sample_next_state(state, 0, n_samples=10)
        for next_state in next_states:
            assert (int(next_state[0]), int(next_state[1])) == (2, 5)  # One row up
            assert not bool(next_state[4])

    def test_robot_boundary_collision(self):
        """Test robot cannot move outside grid boundaries.

        Purpose: Validates robot respects grid boundaries and stays in place when blocked

        Given: Robot at grid boundary positions
        When: Actions try to move robot outside boundaries
        Then: Robot stays at current position instead of moving

        Test type: unit
        """
        env = _make_env(floor_shape=(7, 11))

        # Robot at top boundary trying to go North
        state = np.array([0.0, 5.0, 3.0, 3.0, 0.0])
        next_states = env.sample_next_state(state, 0, n_samples=5)
        for next_state in next_states:
            assert (int(next_state[0]), int(next_state[1])) == (0, 5)  # Should stay in place

    def test_robot_wall_collision(self):
        """Test robot cannot move into walls.

        Purpose: Validates robot respects wall positions and cannot move through them

        Given: Robot adjacent to wall positions
        When: Actions try to move robot into walls
        Then: Robot stays at current position instead of moving into wall

        Test type: unit
        """
        env = _make_env(floor_shape=(7, 11), walls={(3, 3)})  # Wall to the East
        state = np.array([3.0, 2.0, 1.0, 1.0, 0.0])

        next_states = env.sample_next_state(state, 2, n_samples=5)
        for next_state in next_states:
            assert (int(next_state[0]), int(next_state[1])) == (3, 2)  # Should stay in place

    def test_opponent_movement_toward_robot(self):
        """Test opponent tends to move toward robot position.

        Purpose: Validates opponent movement probabilities favor moves toward robot

        Given: Opponent in position where it can move toward robot
        When: Many state transitions are sampled
        Then: Opponent moves toward robot more frequently than random

        Test type: unit
        """
        env = _make_env(floor_shape=(7, 11))
        state = np.array([2.0, 5.0, 5.0, 5.0, 0.0])

        # Robot moves South (action 1)
        samples = env.sample_next_state(state, 1, n_samples=1000)

        # Count opponent positions
        opponent_positions = [(int(s[2]), int(s[3])) for s in samples]
        pos_counts = Counter(opponent_positions)

        # Opponent should prefer moving toward robot (should prefer North: (4,5))
        toward_robot_count = pos_counts.get((4, 5), 0)
        total_samples = len(samples)
        toward_robot_prob = toward_robot_count / total_samples

        # Should be around 0.4 based on implementation
        assert (
            toward_robot_prob > 0.3
        ), f"Expected >0.3 probability toward robot, got {toward_robot_prob}"

    def test_successful_tagging(self):
        """Test successful tagging creates terminal state.

        Purpose: Validates that tag action at same position creates terminal state

        Given: Robot and opponent at same position
        When: Tag action (action 4) is executed
        Then: Next state is terminal with same positions

        Test type: unit
        """
        env = _make_env(floor_shape=(7, 11))
        state = np.array([3.0, 5.0, 3.0, 5.0, 0.0])

        next_states = env.sample_next_state(state, 4, n_samples=10)
        for next_state in next_states:
            assert bool(next_state[4])
            assert (int(next_state[0]), int(next_state[1])) == (3, 5)
            assert (int(next_state[2]), int(next_state[3])) == (3, 5)

    def test_failed_tagging(self):
        """Test failed tagging does not create terminal state.

        Purpose: Validates that tag action at different positions remains non-terminal

        Given: Robot and opponent at different positions
        When: Tag action (action 4) is executed
        Then: Next state remains non-terminal with normal opponent movement

        Test type: unit
        """
        env = _make_env(floor_shape=(7, 11))
        state = np.array([3.0, 5.0, 2.0, 4.0, 0.0])

        next_states = env.sample_next_state(state, 4, n_samples=10)
        for next_state in next_states:
            assert not bool(next_state[4])
            assert (int(next_state[0]), int(next_state[1])) == (3, 5)  # Robot doesn't move on tag

    def test_probability_successful_tag(self):
        """Test probability calculation for successful tag transition.

        Purpose: Validates probability assignment for terminal state after successful tag

        Given: Robot and opponent at same position with tag action
        When: env.transition_log_probability is called with candidate states
        Then: Returns probability 1.0 for correct terminal state and 0.0 for others

        Test type: unit
        """
        env = _make_env(floor_shape=(7, 11))
        state = np.array([3.0, 5.0, 3.0, 5.0, 0.0])

        correct_terminal = np.array([3.0, 5.0, 3.0, 5.0, 1.0])
        wrong_terminal = np.array([3.0, 4.0, 3.0, 4.0, 1.0])
        non_terminal = np.array([3.0, 5.0, 3.0, 5.0, 0.0])

        test_states = [correct_terminal, wrong_terminal, non_terminal]
        probs = _transition_probabilities(env, state, 4, test_states)

        assert probs[0] == 1.0, f"Expected 1.0 for correct terminal state, got {probs[0]}"
        assert probs[1] == 0.0, f"Expected 0.0 for wrong terminal state, got {probs[1]}"
        assert probs[2] == 0.0, f"Expected 0.0 for non-terminal state, got {probs[2]}"

    def test_probability_regular_transition(self):
        """Test probability calculation for regular state transitions.

        Purpose: Validates probability distribution over opponent positions during normal movement

        Given: State with robot and opponent at different positions with movement action
        When: env.transition_log_probability is called with various next states
        Then: Returns correct probabilities based on opponent movement model

        Test type: unit
        """
        env = _make_env(floor_shape=(7, 11))
        state = np.array([3.0, 5.0, 5.0, 5.0, 0.0])

        # Robot moves South (action 1), so next robot position is (4, 5)
        # Opponent at (5, 5) should prefer moving toward robot at (4, 5)
        # Expected moves: North to (4, 5) with prob 0.4, stay at (5, 5) with prob 0.2
        next_state_north = np.array([4.0, 5.0, 4.0, 5.0, 0.0])  # Opponent moves North
        next_state_stay = np.array([4.0, 5.0, 5.0, 5.0, 0.0])  # Opponent stays
        next_state_south = np.array([4.0, 5.0, 6.0, 5.0, 0.0])  # Opponent moves South
        next_state_wrong_robot = np.array([3.0, 5.0, 4.0, 5.0, 0.0])  # Wrong robot position

        test_states = [next_state_north, next_state_stay, next_state_south, next_state_wrong_robot]
        probs = _transition_probabilities(env, state, 1, test_states)

        assert all(p >= 0 for p in probs), "All probabilities should be non-negative"
        assert (
            probs[0] > 0
        ), f"Expected positive probability for opponent moving North, got {probs[0]}"
        assert probs[1] > 0, f"Expected positive probability for opponent staying, got {probs[1]}"
        assert (
            probs[2] == 0.0
        ), f"Expected 0.0 probability for opponent moving South (away), got {probs[2]}"
        assert probs[3] == 0.0, f"Expected 0.0 for wrong robot position, got {probs[3]}"
        assert (
            0.35 <= probs[0] <= 0.45
        ), f"Expected ~0.4 probability for North movement, got {probs[0]}"

    def test_probability_normalization_with_walls(self):
        """Test probability normalization when opponent movement is blocked by walls.

        Purpose: Validates that blocked moves redistribute probability to staying

        Given: Opponent near walls that block some movement directions
        When: env.transition_log_probability is called with possible next states
        Then: Blocked moves have zero probability and staying absorbs extra probability

        Test type: unit
        """
        env = _make_env(floor_shape=(7, 11), walls={(3, 4), (4, 3)})
        state = np.array([3.0, 5.0, 3.0, 3.0, 0.0])

        # Robot moves North (action 0), so next robot position is (2, 5)
        # East move (3, 4) is blocked, South move (4, 3) is blocked
        # North move (2, 3) and West move (3, 2) are valid
        next_state_stay = np.array([2.0, 5.0, 3.0, 3.0, 0.0])
        next_state_north = np.array([2.0, 5.0, 2.0, 3.0, 0.0])
        next_state_east_blocked = np.array([2.0, 5.0, 3.0, 4.0, 0.0])

        test_states = [next_state_stay, next_state_north, next_state_east_blocked]
        probs = _transition_probabilities(env, state, 0, test_states)

        assert (
            probs[0] > 0.2
        ), f"Expected stay probability > 0.2 due to blocked moves, got {probs[0]}"
        assert probs[1] >= 0, f"Expected non-negative probability for North, got {probs[1]}"
        assert probs[2] == 0.0, f"Expected 0.0 for blocked East move, got {probs[2]}"

        # Total probability should sum to 1.0
        all_possible_positions = [(3, 3), (2, 3), (3, 2)]
        all_states = [
            np.array([2.0, 5.0, float(pos[0]), float(pos[1]), 0.0])
            for pos in all_possible_positions
        ]
        all_probs = _transition_probabilities(env, state, 0, all_states)
        total_prob = sum(all_probs)
        assert (
            abs(total_prob - 1.0) < 0.01
        ), f"Expected probabilities to sum to ~1.0, got {total_prob}"

    def test_transition_error_probability_tag_action_no_errors(self):
        """Test that Tag action always executes correctly regardless of error probability.

        Purpose: Validates Tag action is never affected by transition error probability

        Given: Environment with Tag action and non-zero error probability
        When: Transitions are sampled
        Then: Tag action always executes correctly (no random action selection)

        Test type: unit
        """
        np.random.seed(42)  # For reproducibility
        env = _make_env(floor_shape=(7, 11), transition_error_prob=0.9)
        state = np.array([3.0, 5.0, 3.0, 5.0, 0.0])  # Robot and opponent at same position

        next_states = env.sample_next_state(state, 4, n_samples=100)
        for next_state in next_states:
            assert bool(next_state[4]) is True, "Tag action should result in terminal state"
            assert (int(next_state[0]), int(next_state[1])) == (
                3,
                5,
            ), "Robot should stay in place for Tag action"

    def test_transition_error_probability_zero_deterministic(self):
        """Test that error_prob=0.0 maintains deterministic behavior.

        Purpose: Validates backward compatibility with deterministic transitions

        Given: Environment with error_prob=0.0
        When: Movement actions are executed
        Then: Robot always moves according to intended action

        Test type: unit
        """
        np.random.seed(42)  # For reproducibility
        env = _make_env(floor_shape=(7, 11), transition_error_prob=0.0)
        state = np.array([3.0, 5.0, 1.0, 1.0, 0.0])

        # North movement should always succeed
        next_states = env.sample_next_state(state, 0, n_samples=100)
        for next_state in next_states:
            assert (int(next_state[0]), int(next_state[1])) == (
                2,
                5,
            ), "With error_prob=0.0, robot should always move North"

    def test_transition_error_probability_movement_actions(self):
        """Test that movement actions can have errors with non-zero probability.

        Purpose: Validates stochastic action execution for movement actions

        Given: Environment with error_prob > 0.0 and movement action
        When: Transitions are sampled many times
        Then: Robot sometimes executes different actions than intended

        Test type: unit
        """
        np.random.seed(42)  # For reproducibility
        env = _make_env(floor_shape=(7, 11), transition_error_prob=0.8)
        state = np.array([3.0, 5.0, 1.0, 1.0, 0.0])

        # North movement (action 0) with high error probability
        next_states = env.sample_next_state(state, 0, n_samples=1000)
        robot_positions = [(int(ns[0]), int(ns[1])) for ns in next_states]
        position_counts = Counter(robot_positions)

        # Tag action (4) should NOT be in error selection, so robot should always move
        # Verify robot rarely stays in place from Tag error
        assert (3, 5) not in position_counts or position_counts[
            (3, 5)
        ] == 0, "Robot should not stay in place - Tag action not included in error selection"

        intended_count = position_counts.get((2, 5), 0)
        total_samples = sum(position_counts.values())

        assert intended_count < total_samples, (
            f"With error_prob=0.8, not all samples should be intended action. "
            f"Got {intended_count}/{total_samples} intended actions."
        )

        # Verify at least one error action occurs (with high error prob, this should happen)
        error_positions = {(4, 5), (3, 6), (3, 4)}  # South, East, West
        assert any(
            pos in position_counts for pos in error_positions
        ), f"At least one error action should occur. Got positions: {list(position_counts.keys())}"

    def test_transition_error_probability_error_actions_exclude_tag(self):
        """Test that error actions only include movement actions, not Tag.

        Purpose: Validates Tag action is excluded from error action selection

        Given: Environment with movement action and high error probability
        When: Errors occur
        Then: Only movement actions (0-3) are selected as errors, never Tag (4)

        Test type: unit
        """
        np.random.seed(42)  # For reproducibility
        env = _make_env(floor_shape=(7, 11), transition_error_prob=0.99)
        state = np.array([3.0, 5.0, 1.0, 1.0, 0.0])

        # East movement (action 2) with very high error probability
        next_states = env.sample_next_state(state, 2, n_samples=1000)
        robot_positions = [(int(ns[0]), int(ns[1])) for ns in next_states]

        # Robot should never stay in place (which would happen if Tag was selected)
        # East action from (3,5) would go to (3,6); error actions would be (2,5)/(4,5)/(3,4)
        # Tag would keep robot at (3,5) - this should NOT happen
        assert (3, 5) not in Counter(robot_positions) or Counter(robot_positions)[
            (3, 5)
        ] == 0, "Robot should not stay in place - Tag action should not be in error selection"

    def test_transition_error_probability_probability_calculation(self):
        """Test that transition_log_probability accounts for error probability.

        Purpose: Validates probability calculations include all possible action outcomes

        Given: Environment with error probability and movement action
        When: env.transition_log_probability is called for various next states
        Then: Probabilities correctly account for intended and error actions

        Test type: unit
        """
        env = _make_env(floor_shape=(7, 11), transition_error_prob=0.5)
        state = np.array([3.0, 5.0, 1.0, 1.0, 0.0])

        # North action (0). Intended -> (2,5). Errors -> (4,5)/(3,6)/(3,4).
        next_state_north = np.array([2.0, 5.0, 1.0, 1.0, 0.0])  # Intended action
        next_state_south = np.array([4.0, 5.0, 1.0, 1.0, 0.0])  # Error action
        next_state_east = np.array([3.0, 6.0, 1.0, 1.0, 0.0])  # Error action
        next_state_west = np.array([3.0, 4.0, 1.0, 1.0, 0.0])  # Error action
        next_state_wrong = np.array([5.0, 5.0, 1.0, 1.0, 0.0])  # Impossible state

        test_states = [
            next_state_north,
            next_state_south,
            next_state_east,
            next_state_west,
            next_state_wrong,
        ]
        probs = _transition_probabilities(env, state, 0, test_states)

        assert probs[0] > 0, "Intended action should have positive probability"
        assert probs[1] > 0, "Error action (South) should have positive probability"
        assert probs[2] > 0, "Error action (East) should have positive probability"
        assert probs[3] > 0, "Error action (West) should have positive probability"
        assert probs[4] == 0.0, "Impossible state should have zero probability"

    def test_transition_error_probability_tag_probability_deterministic(self):
        """Test that Tag action probability is deterministic regardless of error_prob.

        Purpose: Validates Tag action probability calculation is not affected by error_prob

        Given: Environment with Tag action and high error probability
        When: env.transition_log_probability is called
        Then: Tag action probabilities are deterministic (1.0 or 0.0)

        Test type: unit
        """
        env = _make_env(floor_shape=(7, 11), transition_error_prob=0.9)
        state = np.array([3.0, 5.0, 3.0, 5.0, 0.0])  # Robot and opponent at same position

        correct_state = np.array([3.0, 5.0, 3.0, 5.0, 1.0])
        wrong_state = np.array([3.0, 5.0, 3.0, 5.0, 0.0])

        probs = _transition_probabilities(env, state, 4, [correct_state, wrong_state])
        assert probs[0] == 1.0, "Correct terminal state should have probability 1.0"
        assert probs[1] == 0.0, "Wrong state should have probability 0.0"

    def test_state_transition_all_actions_deterministic(self):
        """Test sample/probability for all movement actions with transition_error_prob=0.

        Purpose: Validates that all movement actions (0-3) execute correctly in deterministic
            mode and robot moves to expected positions

        Given: A LaserTagPOMDP with transition_error_prob=0 and hardcoded states
        When: Each movement action is executed via env.sample_next_state and probability via
            env.transition_log_probability
        Then: Robot moves to expected adjacent positions and the resulting next state has
            positive probability

        Test type: unit
        """
        # Robot at center (5, 5), opponent at (3, 3); large grid with no walls
        initial_state = np.array([5.0, 5.0, 3.0, 3.0, 0.0])
        env = _make_env(floor_shape=(11, 11), transition_error_prob=0.0)

        expected_robot_positions = {
            0: (4, 5),  # North
            1: (6, 5),  # South
            2: (5, 6),  # East
            3: (5, 4),  # West
        }

        for action, expected_robot_pos in expected_robot_positions.items():
            next_state = env.sample_next_state(initial_state.copy(), action)
            actual_robot_pos = (int(next_state[0]), int(next_state[1]))
            assert (
                actual_robot_pos == expected_robot_pos
            ), f"Action {action}: Expected robot at {expected_robot_pos}, got {actual_robot_pos}"

            opponent_pos = (int(next_state[2]), int(next_state[3]))
            assert 0 <= opponent_pos[0] < 11, "Opponent row out of bounds"
            assert 0 <= opponent_pos[1] < 11, "Opponent col out of bounds"
            assert not bool(next_state[4]), "State should be non-terminal"

            prob_actual = _transition_probabilities(env, initial_state, action, [next_state])
            assert prob_actual[0] > 0.0, (
                f"Action {action}: Actual next state should have positive probability, "
                f"got {prob_actual[0]}"
            )

            # Test probability for state with wrong robot position (should be 0.0)
            wrong_robot_state = np.array(
                [
                    initial_state[0],
                    initial_state[1],
                    next_state[2],
                    next_state[3],
                    0.0,
                ]
            )
            prob_wrong = _transition_probabilities(env, initial_state, action, [wrong_robot_state])
            assert prob_wrong[0] == 0.0, (
                f"Action {action}: State with wrong robot position should have probability 0.0, "
                f"got {prob_wrong[0]}"
            )

            # Robot position should always be the same (deterministic) across multiple samples
            samples = env.sample_next_state(initial_state.copy(), action, n_samples=5)
            assert len(samples) == 5
            for i, sample in enumerate(samples):
                sample_robot_pos = (int(sample[0]), int(sample[1]))
                assert sample_robot_pos == expected_robot_pos, (
                    f"Action {action}: Sample {i} robot should be at {expected_robot_pos}, "
                    f"got {sample_robot_pos}"
                )

    def test_state_transition_tag_action_deterministic(self):
        """Test state transition for Tag action when robot and opponent are at same position.

        Purpose: Validates Tag action sampling and probability when robot and opponent are
            colocated

        Given: A LaserTagPOMDP with Tag action and robot/opponent at same position
        When: Tag action is executed via env.sample_next_state and env.transition_log_probability
        Then: Resulting state is terminal with robot and opponent at same position

        Test type: unit
        """
        initial_state = np.array([5.0, 5.0, 5.0, 5.0, 0.0])
        env = _make_env(floor_shape=(11, 11), transition_error_prob=0.0)
        action = 4

        expected_state = np.array([5.0, 5.0, 5.0, 5.0, 1.0])

        next_state = env.sample_next_state(initial_state.copy(), action)
        assert np.array_equal(
            next_state, expected_state
        ), f"Tag action: Expected {expected_state}, got {next_state}"

        prob_expected = _transition_probabilities(env, initial_state, action, [expected_state])
        assert (
            prob_expected[0] == 1.0
        ), f"Tag action: Expected state should have probability 1.0, got {prob_expected[0]}"

        non_terminal_state = np.array([5.0, 5.0, 5.0, 5.0, 0.0])
        prob_non_terminal = _transition_probabilities(
            env, initial_state, action, [non_terminal_state]
        )
        assert prob_non_terminal[0] == 0.0, (
            f"Tag action: Non-terminal state should have probability 0.0, "
            f"got {prob_non_terminal[0]}"
        )

        prob_initial = _transition_probabilities(env, initial_state, action, [initial_state])
        assert (
            prob_initial[0] == 0.0
        ), f"Tag action: Initial state should have probability 0.0, got {prob_initial[0]}"

        samples = env.sample_next_state(initial_state.copy(), action, n_samples=5)
        assert len(samples) == 5
        for i, sample in enumerate(samples):
            assert np.array_equal(
                sample, expected_state
            ), f"Tag action: Sample {i} should equal expected state"

    def test_state_transition_all_actions_boundary_handling(self):
        """Test state transition for all actions when robot is at grid boundaries.

        Purpose: Validates that all actions work correctly when robot is at grid boundaries

        Given: A LaserTagPOMDP with robot at grid boundaries and transition_error_prob=0
        When: Each action is executed via env.sample_next_state
        Then: Robot stays in place when action would move outside boundaries

        Test type: unit
        """
        floor_shape = (7, 7)
        env = _make_env(floor_shape=floor_shape, transition_error_prob=0.0)

        test_cases = [
            ((0, 3), 0, (0, 3)),  # Top boundary, try North - should stay
            ((6, 3), 1, (6, 3)),  # Bottom boundary, try South - should stay
            ((3, 6), 2, (3, 6)),  # Right boundary, try East - should stay
            ((3, 0), 3, (3, 0)),  # Left boundary, try West - should stay
        ]

        for (robot_row, robot_col), action, expected_robot_pos in test_cases:
            initial_state = np.array([float(robot_row), float(robot_col), 2.0, 2.0, 0.0])

            next_state = env.sample_next_state(initial_state.copy(), action)
            actual_robot_pos = (int(next_state[0]), int(next_state[1]))

            assert actual_robot_pos == expected_robot_pos, (
                f"Action {action} at boundary ({robot_row}, {robot_col}): "
                f"Expected robot at {expected_robot_pos}, got {actual_robot_pos}"
            )

            prob_actual = _transition_probabilities(env, initial_state, action, [next_state])
            assert prob_actual[0] > 0.0, (
                f"Action {action} at boundary: Actual next state should have positive probability, "
                f"got {prob_actual[0]}"
            )

            outside_boundary_state = next_state.copy()
            if action == 0:  # North
                outside_boundary_state[0] = -1.0
            elif action == 1:  # South
                outside_boundary_state[0] = floor_shape[0]
            elif action == 2:  # East
                outside_boundary_state[1] = floor_shape[1]
            elif action == 3:  # West
                outside_boundary_state[1] = -1.0

            prob_outside = _transition_probabilities(
                env, initial_state, action, [outside_boundary_state]
            )
            assert prob_outside[0] == 0.0, (
                f"Action {action} at boundary: State with robot outside boundary should have "
                f"probability 0.0, got {prob_outside[0]}"
            )


class TestLaserTagPOMDPTransitionError:
    """Test LaserTagPOMDP transition error probability parameter.

    Purpose: Validates transition_error_prob parameter in LaserTagPOMDP environment

    Given: LaserTagPOMDP instances with various transition_error_prob values
    When: Environment is used for state transitions
    Then: Error probability affects robot movement as expected

    Test type: unit
    """

    def test_transition_error_probability_parameter_validation(self):
        """Test that transition_error_prob parameter is validated.

        Purpose: Validates parameter validation for transition_error_prob

        Given: Invalid transition_error_prob values
        When: LaserTagPOMDP is initialized
        Then: ValueError is raised for invalid values

        Test type: unit
        """
        # Test negative value
        with pytest.raises(ValueError, match="transition_error_prob must be between 0 and 1"):
            LaserTagPOMDP(discount_factor=0.95, transition_error_prob=-0.1)

        # Test value > 1.0
        with pytest.raises(ValueError, match="transition_error_prob must be between 0 and 1"):
            LaserTagPOMDP(discount_factor=0.95, transition_error_prob=1.1)

        # Test valid values should not raise
        LaserTagPOMDP(discount_factor=0.95, transition_error_prob=0.0)
        LaserTagPOMDP(discount_factor=0.95, transition_error_prob=0.5)
        LaserTagPOMDP(discount_factor=0.95, transition_error_prob=1.0)

    def test_transition_error_probability_stored_on_environment(self):
        """Test that transition_error_prob is stored on the environment.

        Purpose: Validates parameter is accessible on the env instance

        Given: LaserTagPOMDP with transition_error_prob
        When: The attribute is read
        Then: Environment exposes the configured value

        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95, transition_error_prob=0.3)
        assert env.transition_error_prob == 0.3

    def test_transition_error_probability_default_zero(self):
        """Test that default transition_error_prob is 0.0 (backward compatibility).

        Purpose: Validates backward compatibility with default deterministic behavior

        Given: LaserTagPOMDP without specifying transition_error_prob
        When: Movement actions are sampled
        Then: transition_error_prob defaults to 0.0 and the robot always executes the
            intended action

        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95, dangerous_areas=set())
        assert env.transition_error_prob == 0.0, "Default transition_error_prob should be 0.0"

        np.random.seed(42)
        state = np.array([3.0, 5.0, 1.0, 1.0, 0.0])
        next_states = env.sample_next_state(state, 0, n_samples=50)
        for ns in next_states:
            assert (int(ns[0]), int(ns[1])) == (
                2,
                5,
            ), "Default error_prob should leave robot deterministically moving North"

    def test_transition_error_probability_affects_simulation(self):
        """Test that transition_error_prob affects actual simulation behavior.

        Purpose: Validates error probability affects robot movement in simulations

        Given: Environment with non-zero error probability
        When: Multiple transitions are sampled
        Then: Robot sometimes executes different actions than intended

        Test type: integration
        """
        np.random.seed(42)  # For reproducibility
        env = LaserTagPOMDP(discount_factor=0.95, transition_error_prob=0.7)
        state = np.array([3.0, 5.0, 1.0, 1.0, 0.0])

        # Sample many transitions with North action (0) via the env-level API.
        next_states = env.sample_next_state(state, 0, n_samples=1000)
        robot_positions = [(int(ns[0]), int(ns[1])) for ns in next_states]

        position_counts = Counter(robot_positions)

        # With error_prob=0.7, we should see mostly error actions
        # Verify that we don't always get the intended action (indicating stochasticity)
        intended_count = position_counts.get((2, 5), 0)
        total_samples = sum(position_counts.values())

        # With error_prob=0.7, we expect ~30% intended, ~70% errors
        # But with fixed seed, we might get deterministic results
        # So we just verify that stochastic behavior is working (not all intended)
        assert intended_count < total_samples, (
            f"With error_prob=0.7, not all samples should be intended action. "
            f"Got {intended_count}/{total_samples} intended actions."
        )

        # Verify at least one error position occurs (with high error prob, this should happen)
        error_positions = {(4, 5), (3, 6), (3, 4)}  # South, East, West
        assert any(
            pos in position_counts for pos in error_positions
        ), f"At least one error position should occur. Got positions: {list(position_counts.keys())}"


class TestLaserTagObservation:
    """Test LaserTag observation generation via the env API.

    Purpose: Validates observation generation with Gaussian noise around laser ranges

    Given: LaserTagPOMDP environments with various states and noise levels
    When: Observations are sampled via env.sample_observation and probabilities are
        evaluated via env.observation_log_probability
    Then: Observations follow expected noise characteristics

    Test type: unit
    """

    def test_observation_noise_characteristics(self):
        """Test observation generation adds appropriate Gaussian noise.

        Purpose: Validates observation noise is bounded by the configured measurement_noise

        Given: Environment with low measurement noise and small grid
        When: Many observations are sampled via env.sample_observation
        Then: Observations are 8-D, non-negative, and have bounded standard deviation

        Test type: unit
        """
        env = _make_env(floor_shape=(5, 5), measurement_noise=0.1)
        next_state = np.array([2.0, 2.0, 2.0, 4.0, 0.0])

        observations = env.sample_observation(next_state, 0, n_samples=100)
        obs_array = np.array(observations)

        assert obs_array.shape[1] == 8, "Observations should be 8-dimensional laser measurements"
        assert np.all(obs_array >= 0), "Laser measurements should be non-negative"

        std_obs = np.std(obs_array, axis=0)
        assert np.all(
            std_obs <= 1.0
        ), "Standard deviation should be reasonable for laser measurements"

    def test_terminal_state_observation(self):
        """Test terminal state produces special observation.

        Purpose: Validates terminal states generate designated terminal observation

        Given: Terminal LaserTag state
        When: Observations are sampled via env.sample_observation
        Then: All observations are the special terminal observation tuple

        Test type: unit
        """
        env = _make_env(floor_shape=(7, 11))
        terminal_state = np.array([3.0, 5.0, 3.0, 5.0, 1.0])

        observations = env.sample_observation(terminal_state, 4, n_samples=10)
        terminal_obs = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
        for obs in observations:
            assert tuple(obs) == terminal_obs, f"Terminal observation should be {terminal_obs}"

    def test_probability_terminal_observation(self):
        """Test probability calculation for terminal state observations.

        Purpose: Validates probability is 1.0 for terminal observation and 0.0 for others

        Given: Terminal LaserTag state
        When: env.observation_log_probability is called with terminal and non-terminal obs
        Then: Returns 1.0 for terminal observation and 0.0 for others

        Test type: unit
        """
        env = _make_env(floor_shape=(7, 11))
        terminal_state = np.array([3.0, 5.0, 3.0, 5.0, 1.0])

        terminal_obs = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
        non_terminal_obs = (1.0, 2.0, 3.0, 1.5, 2.5, 1.2, 0.8, 2.1)
        wrong_terminal_obs = (-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0)

        test_observations = [terminal_obs, non_terminal_obs, wrong_terminal_obs]
        probs = _observation_probabilities(env, terminal_state, 4, test_observations)

        assert probs[0] == 1.0, f"Expected 1.0 for terminal observation, got {probs[0]}"
        assert probs[1] == 0.0, f"Expected 0.0 for non-terminal observation, got {probs[1]}"
        assert probs[2] == 0.0, f"Expected 0.0 for wrong terminal observation, got {probs[2]}"

    def test_probability_non_terminal_observation(self):
        """Test probability calculation for non-terminal state observations.

        Purpose: Validates Gaussian probability density calculation for laser measurements

        Given: Non-terminal state with known laser measurements
        When: env.observation_log_probability is called with observations near true measurements
        Then: Returns higher probability for observations closer to true measurements

        Test type: unit
        """
        env = _make_env(floor_shape=(7, 7), measurement_noise=1.0)
        next_state = np.array([3.0, 3.0, 5.0, 5.0, 0.0])

        np.random.seed(42)
        true_sample = env.sample_observation(next_state, 0)

        exact_obs = tuple(true_sample)
        close_obs = tuple(m + 0.1 for m in true_sample)
        far_obs = tuple(m + 5.0 for m in true_sample)

        test_observations = [exact_obs, close_obs, far_obs]
        probs = _observation_probabilities(env, next_state, 0, test_observations)

        assert probs[0] > 0, f"Expected positive probability for exact observation, got {probs[0]}"
        assert (
            probs[0] > probs[1] > probs[2]
        ), f"Expected exact > close > far, got {probs[0]}, {probs[1]}, {probs[2]}"
        assert all(p >= 0 for p in probs), "All probabilities should be non-negative"

    def test_probability_with_walls(self):
        """Test probability calculation accounts for walls in laser measurements.

        Purpose: Validates laser measurements consider walls as obstacles

        Given: Robot position with walls in specific directions
        When: env.observation_log_probability is called on a sampled observation
        Then: Returns positive probability for the matching observation

        Test type: unit
        """
        env = _make_env(floor_shape=(7, 7), walls={(3, 5)}, measurement_noise=0.5)
        next_state = np.array([3.0, 3.0, 5.0, 5.0, 0.0])

        np.random.seed(42)
        sampled_obs = env.sample_observation(next_state, 0)
        probs = _observation_probabilities(env, next_state, 0, [sampled_obs])
        assert (
            probs[0] > 0
        ), f"Expected positive probability for matching observation, got {probs[0]}"

    def test_probability_gaussian_properties(self):
        """Test that probability follows Gaussian distribution properties.

        Purpose: Validates observation probability follows expected Gaussian characteristics

        Given: Environment with measurement_noise=1.0
        When: env.observation_log_probability is called with observations at various
            distances from a sampled baseline
        Then: Probabilities follow a monotone Gaussian decay pattern

        Test type: unit
        """
        env = _make_env(floor_shape=(7, 7), measurement_noise=1.0)
        next_state = np.array([3.0, 3.0, 5.0, 5.0, 0.0])

        np.random.seed(42)
        base_sample = env.sample_observation(next_state, 0)

        observations_at_distances = []
        for distance in [0.0, 0.5, 1.0, 2.0, 3.0]:
            obs = tuple(base_sample[i] + (distance if i == 0 else 0.0) for i in range(8))
            observations_at_distances.append(obs)

        probs = _observation_probabilities(env, next_state, 0, observations_at_distances)

        for i in range(len(probs) - 1):
            assert (
                probs[i] >= probs[i + 1]
            ), f"Probability should decrease with distance: {probs[i]} >= {probs[i+1]} at index {i}"

        assert all(p > 0 for p in probs), f"All probabilities should be positive, got {probs}"


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
        expected_walls = {
            (1, 2),
            (3, 0),
            (3, 4),
            (5, 0),
            (6, 4),
            (9, 1),
            (9, 4),
            (10, 6),
        }
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
        # Create environment with no dangerous areas to get deterministic rewards
        env = LaserTagPOMDP(discount_factor=0.95, dangerous_areas=set())

        # Test successful tag - using position (1, 1) which is not in default dangerous areas
        same_pos_state = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        tag_reward = env.reward(same_pos_state, 4)
        assert tag_reward == env.tag_reward

        # Test failed tag
        diff_pos_state = np.array([1.0, 1.0, 2.0, 4.0, 0.0])
        failed_tag_reward = env.reward(diff_pos_state, 4)
        assert failed_tag_reward == -env.tag_penalty

        # Test movement cost
        move_reward = env.reward(diff_pos_state, 0)  # North
        assert move_reward == -env.step_cost

        # Test terminal state
        terminal_state = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
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

        non_terminal = np.array([3.0, 5.0, 2.0, 4.0, 0.0])
        terminal = np.array([3.0, 5.0, 3.0, 5.0, 1.0])

        assert not env.is_terminal(non_terminal)
        assert env.is_terminal(terminal)

    def test_reward_range(self):
        """Test that reward range is correctly calculated.

        Purpose: Validates that LaserTagPOMDP reward range is properly calculated based on tag rewards and penalties

        Given: A LaserTagPOMDP environment with specific tag reward and penalty parameters
        When: Environment reward_range attribute is checked
        Then: Returns range based on worst case (-tag_penalty) and best case (tag_reward)

        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95, tag_reward=15.0, tag_penalty=20.0, step_cost=2.0)

        # Expected calculation from LaserTagPOMDP constructor:
        # reward_range=(-tag_penalty, tag_reward) = (-20.0, 15.0)
        expected_min = -20.0  # Failed tag penalty
        expected_max = 15.0  # Successful tag reward

        assert env.reward_range == (expected_min, expected_max)

        # Test with different parameters
        env2 = LaserTagPOMDP(
            discount_factor=0.95, tag_reward=50.0, tag_penalty=100.0, step_cost=1.0
        )

        expected_min2 = -100.0  # Failed tag penalty
        expected_max2 = 50.0  # Successful tag reward

        assert env2.reward_range == (expected_min2, expected_max2)

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
            assert isinstance(state, np.ndarray) and len(state) == 5
            assert (int(state[0]), int(state[1])) != (
                int(state[2]),
                int(state[3]),
            )  # Should start at different positions
            assert not bool(state[4])

    def test_initial_state_functionality(self):
        """Test initial_state parameter functionality.

        Purpose: Validates that initial_state parameter correctly sets initial state distribution

        Given: LaserTag environment with initial_state parameter set or None
        When: Initial state distribution is sampled
        Then: Returns uniform distribution when None, or single state with probability 1.0 when set

        Test type: unit
        """
        # Test 1: When initial_state is None (default), should return uniform distribution
        env_default = LaserTagPOMDP(discount_factor=0.95)
        assert env_default.initial_state is None

        initial_dist_default = env_default.initial_state_dist()
        initial_states_default = initial_dist_default.sample(n_samples=20)

        # Should get different states (not all the same)
        unique_states = set(
            tuple(state) for state in initial_states_default
        )  # Convert to tuple for hashing
        assert (
            len(unique_states) > 1
        ), "Default distribution should return multiple different states"

        # Test 2: When initial_state is set, should return that state with probability 1.0
        start_state = np.array([2.0, 3.0, 5.0, 6.0, 0.0])
        env_custom = LaserTagPOMDP(discount_factor=0.95, initial_state=start_state)
        assert env_custom.initial_state is not None
        assert np.array_equal(env_custom.initial_state, start_state)

        initial_dist_custom = env_custom.initial_state_dist()
        initial_states_custom = initial_dist_custom.sample(n_samples=50)

        # All sampled states should be the same as start_state
        for state in initial_states_custom:
            assert np.array_equal(
                state, start_state
            ), f"Expected all states to be {start_state}, got {state}"

        # Test probability: should be 1.0 for the start state
        probs = initial_dist_custom.probability([start_state])
        assert probs[0] == 1.0, f"Expected probability 1.0 for start state, got {probs[0]}"

        # Test probability: should be 0.0 for any other state
        other_state = np.array([1.0, 1.0, 2.0, 2.0, 0.0])
        probs_other = initial_dist_custom.probability([other_state])
        assert (
            probs_other[0] == 0.0
        ), f"Expected probability 0.0 for other state, got {probs_other[0]}"

        # Test 3: Verify that the distribution has only one state
        # Check that it's a DiscreteDistribution and verify its structure
        assert isinstance(
            initial_dist_custom, DiscreteDistribution
        ), "Initial state distribution should be DiscreteDistribution"
        assert (
            len(initial_dist_custom.values) == 1
        ), "Custom distribution should have only one state"
        assert np.array_equal(
            initial_dist_custom.values[0], start_state
        ), "Distribution should contain the start state"

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
        # Create a simple belief for testing
        dummy_particles = [np.array([3.0, 2.0, 1.0, 1.0, 0.0])]
        dummy_log_weights = np.array([-0.1])  # Small non-zero log weight
        test_belief = WeightedParticleBelief(
            particles=dummy_particles, log_weights=dummy_log_weights
        )

        # Step 1: Robot tries to move East into wall
        state1 = np.array([3.0, 2.0, 1.0, 1.0, 0.0])
        step1 = StepData(
            state=state1,
            action=2,  # East
            next_state=np.array([3.0, 2.0, 1.0, 0.0, 0.0]),  # Robot stayed due to wall
            observation=(
                1.1,
                0.9,
                1.2,
                0.8,
                1.0,
                1.1,
                0.9,
                1.3,
            ),  # 8D laser observation
            reward=-1.0,
            belief=test_belief,
        )

        # Step 2: Normal movement
        state2 = np.array([3.0, 2.0, 1.0, 0.0, 0.0])
        step2 = StepData(
            state=state2,
            action=0,  # North
            next_state=np.array([2.0, 2.0, 1.0, 1.0, 0.0]),
            observation=(
                1.0,
                1.1,
                1.05,
                1.15,
                1.02,
                1.08,
                1.12,
                1.04,
            ),  # 8D laser observation
            reward=-1.0,
            belief=test_belief,
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
            policy_run_data=[PolicyRunData(info_variables=[])],
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
        assert (
            collision_metric.value == 1.0
        ), f"Expected 1.0 collision, got {collision_metric.value}"

        # Find dangerous area steps metric
        dangerous_area_metric = None
        for metric in metrics:
            if metric.name == "average_dangerous_area_steps":
                dangerous_area_metric = metric
                break

        assert dangerous_area_metric is not None, "Dangerous area steps metric not found"
        # Should have 0 dangerous area steps since robot positions are not in default dangerous areas
        assert (
            dangerous_area_metric.value == 0.0
        ), f"Expected 0.0 dangerous area steps, got {dangerous_area_metric.value}"

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
            discount_factor=0.95, walls=walls, dangerous_area_penalty=5.0, step_cost=1.0
        )

        # Test wall collision penalty
        state = np.array([3.0, 2.0, 1.0, 1.0, 0.0])

        # Try to move East into wall at (3, 3)
        reward = env.reward(state, 2)  # East action

        # Expected: -1.0 (step cost) - 5.0 (wall collision penalty) = -6.0
        expected_reward = -env.step_cost - env.dangerous_area_penalty
        assert (
            reward == expected_reward
        ), f"Expected {expected_reward} for wall collision, got {reward}"

    def test_reward_difference_wall_collision_vs_safe(self):
        """Test that rewards for actions leading to wall collision are lower than safe actions.

        Purpose: Validates that wall collision actions receive lower rewards than safe actions

        Given: LaserTag environment with walls and two states (one leading to collision, one safe)
        When: Same action is executed in both states
        Then: Reward for collision action should be lower than safe action reward

        Test type: unit
        """
        walls = {(3, 3)}
        env = LaserTagPOMDP(
            discount_factor=0.95, walls=walls, dangerous_area_penalty=5.0, step_cost=1.0
        )

        # Create state where robot action will lead to wall collision
        # Robot at (3, 2), wall at (3, 3), action East (2) will try to move to (3, 3) - collision!
        will_collide_state = np.array([3.0, 2.0, 1.0, 1.0, 0.0])

        # Create state where robot action will NOT lead to wall collision
        # Robot at (1, 1), action East (2) will move to (1, 2) - no collision
        safe_state = np.array([1.0, 1.0, 1.0, 1.0, 0.0])

        # Test movement action (East = 2)
        # For will_collide_state: (3, 2) + East = (3, 3) which is a wall - collision!
        # For safe_state: (1, 1) + East = (1, 2) which is safe - no collision
        will_collide_reward = env.reward(will_collide_state, 2)  # East action
        safe_reward = env.reward(safe_state, 2)  # East action

        # Action leading to collision should get dangerous_area_penalty (-5.0) in addition to step_cost (-1.0)
        # Safe action should only get step_cost (-1.0)

        # The reward for collision action should be lower than safe action
        assert (
            will_collide_reward < safe_reward
        ), f"Collision action reward ({will_collide_reward}) should be < safe action reward ({safe_reward})"

        # Both should be negative (due to step cost)
        assert (
            will_collide_reward < 0
        ), f"Collision reward should be negative, got {will_collide_reward}"
        assert safe_reward < 0, f"Safe reward should be negative, got {safe_reward}"

        # Verify the difference is exactly the dangerous_area_penalty
        expected_difference = env.dangerous_area_penalty
        actual_difference = safe_reward - will_collide_reward
        assert abs(actual_difference - expected_difference) < 1e-6, (
            f"Reward difference should be {expected_difference} (dangerous_area_penalty), "
            f"got {actual_difference}"
        )

    def test_compute_metrics_with_simulator_generated_history(self):
        """Test compute_metrics with realistic history generated using environment simulation.

        Purpose: Validates compute_metrics function with realistic episode data generated by environment simulation

        Given: LaserTag environment and manually generated episode using environment step sampling
        When: Episode histories are generated and compute_metrics is called
        Then: All expected metrics are returned with valid values and proper types

        Test type: integration
        """
        # Create LaserTag environment with known configuration
        env = LaserTagPOMDP(
            discount_factor=0.95,
            tag_reward=10.0,
            tag_penalty=10.0,
            step_cost=1.0,
            measurement_noise=0.1,
            dangerous_areas={(3, 3), (4, 4)},  # Add some dangerous areas
        )

        # Generate realistic episode histories by running environment simulation
        histories = []
        np.random.seed(42)  # For reproducibility

        for episode_idx in range(3):  # Generate 3 episodes
            # Initialize episode
            initial_state = env.initial_state_dist().sample()[0]
            current_state = initial_state
            steps = []

            # Run episode for up to 15 steps
            for step_idx in range(15):
                # Choose random action (simulate random policy)
                action = np.random.choice(env.get_actions())

                # Sample next step using environment
                next_state, observation, reward = env.sample_next_step(current_state, action)

                # Create a belief for testing
                dummy_particles = [current_state]
                dummy_log_weights = np.array([-0.1])  # Small non-zero log weight
                test_belief = WeightedParticleBelief(
                    particles=dummy_particles, log_weights=dummy_log_weights
                )

                # Create step data
                step = StepData(
                    state=current_state,
                    action=action,
                    next_state=next_state,
                    observation=observation,
                    reward=reward,
                    belief=test_belief,
                )
                steps.append(step)

                # Update current state
                current_state = next_state

                # End episode if terminal or successful tag
                if env.is_terminal(next_state) or (action == 4 and reward > 0):
                    break

            # Create history object
            history = History(
                history=steps,
                discount_factor=env.discount_factor,
                average_state_sampling_time=0.01,
                average_action_time=0.01,
                average_observation_time=0.01,
                average_belief_update_time=0.01,
                average_reward_time=0.01,
                actual_num_steps=len(steps),
                reach_terminal_state=env.is_terminal(current_state),
                policy_run_data=[PolicyRunData(info_variables=[])],
            )
            histories.append(history)

        # Test compute_metrics with generated histories
        metrics = env.compute_metrics(histories)

        # Validate that we get the expected LaserTag-specific metrics
        metric_names = {metric.name for metric in metrics}
        expected_metrics = {
            "tag_success_rate",
            "average_episode_length",
            "average_failed_tag_attempts",
            "average_obstacle_collisions",
            "average_dangerous_area_steps",
        }

        assert expected_metrics.issubset(
            metric_names
        ), f"Missing metrics: {expected_metrics - metric_names}"

        # Validate metric value types and ranges
        for metric in metrics:
            assert isinstance(
                metric.value, (int, float)
            ), f"Metric {metric.name} value must be numeric"
            assert (
                metric.value >= 0
            ), f"Metric {metric.name} value must be non-negative: {metric.value}"
            assert hasattr(
                metric, "lower_confidence_bound"
            ), f"Metric {metric.name} missing lower confidence bound"
            assert hasattr(
                metric, "upper_confidence_bound"
            ), f"Metric {metric.name} missing upper confidence bound"

            # Validate specific metric ranges
            if metric.name == "tag_success_rate":
                assert (
                    0.0 <= metric.value <= 1.0
                ), f"Tag success rate must be between 0 and 1: {metric.value}"
            elif metric.name == "average_episode_length":
                assert (
                    metric.value <= 15
                ), f"Average episode length should not exceed max: {metric.value}"

        # Test that histories contain realistic LaserTag data
        for history in histories:
            assert len(history.history) > 0, "History should contain steps"

            for step in history.history:
                assert (
                    isinstance(step.state, np.ndarray) and len(step.state) == 5
                ), "State should be LaserTagState"
                assert (
                    step.action in env.get_actions()
                ), f"Action {step.action} not in valid actions"
                assert isinstance(step.reward, (int, float)), "Reward should be numeric"
                assert len(step.observation) == 8, "LaserTag observation should be 8-dimensional"

                # Validate robot and opponent positions are within grid bounds
                robot_pos = (int(step.state[0]), int(step.state[1]))
                opponent_pos = (int(step.state[2]), int(step.state[3]))
                assert (
                    0 <= robot_pos[0] < env.floor_shape[0]
                ), f"Robot row {robot_pos[0]} out of bounds"
                assert (
                    0 <= robot_pos[1] < env.floor_shape[1]
                ), f"Robot col {robot_pos[1]} out of bounds"
                assert (
                    0 <= opponent_pos[0] < env.floor_shape[0]
                ), f"Opponent row {opponent_pos[0]} out of bounds"
                assert (
                    0 <= opponent_pos[1] < env.floor_shape[1]
                ), f"Opponent col {opponent_pos[1]} out of bounds"

    def test_compute_metrics_handles_none_rewards(self):
        """Test compute_metrics properly handles None reward values without crashing.

        Purpose: Validates that compute_metrics handles None reward values gracefully without TypeError

        Given: Episode history with some steps containing None rewards
        When: compute_metrics is called on the history
        Then: Metrics are computed without error and None rewards are ignored

        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95)

        # Create test belief
        dummy_particles = [np.array([3.0, 5.0, 2.0, 4.0, 0.0])]
        dummy_log_weights = np.array([-0.1])  # Small non-zero log weight
        test_belief = WeightedParticleBelief(
            particles=dummy_particles, log_weights=dummy_log_weights
        )

        # Step 1: Normal reward
        step1 = StepData(
            state=np.array([3.0, 5.0, 2.0, 4.0, 0.0]),
            action=0,  # North
            next_state=np.array([2.0, 5.0, 1.0, 4.0, 0.0]),
            observation=(1.0, 2.0, 3.0, 1.5, 2.5, 1.2, 0.8, 2.1),
            reward=-1.0,
            belief=test_belief,
        )

        # Step 2: None reward (this should not crash)
        step2 = StepData(
            state=np.array([2.0, 5.0, 1.0, 4.0, 0.0]),
            action=4,  # Tag action
            next_state=np.array([2.0, 5.0, 0.0, 4.0, 0.0]),
            observation=(1.1, 2.1, 3.1, 1.6, 2.6, 1.3, 0.9, 2.2),
            reward=None,  # This is the problematic case
            belief=test_belief,
        )

        # Step 3: Successful tag with positive reward
        step3 = StepData(
            state=np.array([2.0, 5.0, 0.0, 4.0, 0.0]),
            action=4,  # Tag action
            next_state=np.array([2.0, 5.0, 2.0, 5.0, 1.0]),
            observation=(
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
            ),  # Terminal obs
            reward=10.0,  # Successful tag
            belief=test_belief,
        )

        history = History(
            history=[step1, step2, step3],
            discount_factor=env.discount_factor,
            average_state_sampling_time=0.01,
            average_action_time=0.01,
            average_observation_time=0.01,
            average_belief_update_time=0.01,
            average_reward_time=0.01,
            actual_num_steps=3,
            reach_terminal_state=True,
            policy_run_data=[PolicyRunData(info_variables=[])],
        )

        # This should not raise any exceptions
        try:
            metrics = env.compute_metrics([history])

            # Validate metrics were computed
            metric_names = {metric.name for metric in metrics}
            expected_metrics = {
                "tag_success_rate",
                "average_episode_length",
                "average_failed_tag_attempts",
                "average_obstacle_collisions",
                "average_dangerous_area_steps",
            }

            assert expected_metrics.issubset(
                metric_names
            ), f"Missing metrics: {expected_metrics - metric_names}"

            # Find success rate metric
            success_rate_metric = None
            for metric in metrics:
                if metric.name == "tag_success_rate":
                    success_rate_metric = metric
                    break

            assert success_rate_metric is not None, "tag_success_rate metric not found"
            # Should be 1.0 since last step has positive reward (successful tag)
            assert (
                success_rate_metric.value == 1.0
            ), f"Expected success rate 1.0, got {success_rate_metric.value}"

            # Find failed tag attempts metric
            failed_tags_metric = None
            for metric in metrics:
                if metric.name == "average_failed_tag_attempts":
                    failed_tags_metric = metric
                    break

            assert failed_tags_metric is not None, "average_failed_tag_attempts metric not found"
            # Should be 0.0 since None reward is ignored (not counted as failed tag)
            assert (
                failed_tags_metric.value == 0.0
            ), f"Expected 0.0 failed tags, got {failed_tags_metric.value}"

        except Exception as e:
            pytest.fail(f"compute_metrics should handle None rewards gracefully, but raised: {e}")


def test_metrics_confidence_intervals():
    """Test that metric values fall within their confidence intervals using realistic episodes.

    Purpose: Validates that LaserTagPOMDP compute_metrics returns metric values within their confidence bounds

    Given: A LaserTagPOMDP environment and realistic episode histories generated using run_episode
    When: compute_metrics is called with the histories
    Then: Each metric value falls within its lower_confidence_bound and upper_confidence_bound

    Test type: integration
    """
    # Create a LaserTagPOMDP environment with smaller grid for faster testing
    env = LaserTagPOMDP(
        discount_factor=0.95,
        floor_shape=(7, 7),  # Smaller grid for faster episodes
        walls={(2, 2), (3, 4), (5, 1)},  # Some walls for variety
        dangerous_areas={(1, 3), (4, 4)},  # Some dangerous areas
    )

    # Create random policy and initial belief
    policy = RandomPolicy(env, name="TestRandomPolicy")

    # Create initial belief using particles from initial state distribution
    initial_state = env.initial_state_dist().sample()[0]
    particles = [initial_state] * 50  # Simple particle belief
    log_weights = np.full(50, -np.log(50))  # Uniform weights
    initial_belief = WeightedParticleBelief(
        particles=particles, log_weights=log_weights, resampling=False
    )

    # Create logger for episode runs
    logger = get_logger(name="confidence_test", debug=False)

    # Generate realistic episode histories using run_episode
    histories = []
    np.random.seed(42)  # For reproducible test

    for episode_idx in range(20):  # Use enough episodes for meaningful statistics
        # Run episode with varying episode lengths
        max_steps = 5 + (episode_idx % 8)  # Vary episode length 5-12 steps

        history = run_episode(
            environment=env,
            policy=policy,
            initial_belief=initial_belief,
            num_steps=max_steps,
            logger=logger,
        )

        histories.append(history)

    # Ensure we have at least some histories
    assert len(histories) > 0, "No histories were generated"

    # Compute metrics using the realistic histories
    metrics = env.compute_metrics(histories)

    # Verify that metrics were computed
    assert len(metrics) > 0, "No metrics were computed"

    # Use generic confidence interval verification
    verify_metrics_within_confidence_intervals(metrics)


class TestLaserTagRewardBatch:
    """Tests for the vectorised reward_batch override on LaserTagPOMDP.

    Purpose: Validates that reward_batch produces results numerically identical to
        the scalar reward() loop and handles edge cases correctly.

    Given: A LaserTagPOMDP instance with default parameters.
    When: reward_batch is called with various particle sets and actions.
    Then: Results match the scalar loop within floating-point tolerance and corner
        cases (terminal states, out-of-bounds intended positions) are handled.

    Test type: unit
    """

    def _make_env(self) -> LaserTagPOMDP:
        return LaserTagPOMDP(discount_factor=0.95)

    def _sample_states(self, env: LaserTagPOMDP, n: int, seed: int = 0) -> np.ndarray:
        np.random.seed(seed)
        dist = env.initial_state_dist()
        states = np.array([dist.sample()[0] for _ in range(n)], dtype=np.float64)
        return states

    def test_lasertag_reward_batch_matches_scalar_loop(self) -> None:
        """reward_batch agrees with the scalar loop for 64 particles × all 5 actions.

        Purpose: Validates numerical equivalence between reward_batch and the
            per-particle Python loop that calls scalar reward() for every action.

        Given: 64 non-terminal particle states sampled from the initial distribution.
        When: reward_batch(states, action) and [env.reward(s, action) for s in states]
            are computed for each action in {0, 1, 2, 3, 4}.
        Then: Arrays agree with atol=1e-12.

        Test type: unit
        """
        env = self._make_env()
        states = self._sample_states(env, 64, seed=42)

        for action in env.get_actions():
            batch = env.reward_batch(states, action)
            scalar = np.array([env.reward(s, action) for s in states], dtype=np.float64)
            np.testing.assert_allclose(
                batch,
                scalar,
                atol=1e-12,
                err_msg=f"Mismatch for action={action}",
            )

    def test_lasertag_reward_batch_terminal_states_return_zero(self) -> None:
        """reward_batch returns 0 for every particle whose terminal flag is set.

        Purpose: Validates that terminal particles always yield reward 0 regardless
            of robot/opponent positions or the chosen action.

        Given: A batch of 20 states with terminal flag (index 4) set to 1.0.
        When: reward_batch is called with each action in {0, 1, 2, 3, 4}.
        Then: All returned rewards are exactly 0.0.

        Test type: unit
        """
        env = self._make_env()
        states = self._sample_states(env, 20, seed=7)
        states[:, 4] = 1.0  # Force all particles to terminal

        for action in env.get_actions():
            batch = env.reward_batch(states, action)
            assert np.all(batch == 0.0), (
                f"Expected all-zero rewards for terminal states with action={action}, "
                f"got {batch}"
            )

    def test_lasertag_reward_batch_handles_out_of_bounds_intended(self) -> None:
        """reward_batch matches scalar reward when the intended move exits the grid.

        Purpose: Validates that an intended position outside the floor grid is treated
            identically by reward_batch and scalar reward (no crash, correct penalty
            logic — OOB positions are not in self.walls so no wall penalty applies).

        Given: A robot placed at the top row (row=0) attempting to move North (action=0),
            which would yield intended row=-1 (out of bounds).
        When: reward_batch is called with this set of particles.
        Then: reward_batch result equals the scalar reward result for each particle.

        Test type: unit
        """
        env = self._make_env()
        # Robot at row=0, various cols; opponent somewhere else; non-terminal.
        n = 10
        states = np.zeros((n, 5), dtype=np.float64)
        states[:, 0] = 0.0  # robot row = 0 → North moves OOB
        states[:, 1] = np.arange(n, dtype=np.float64) % env.floor_shape[1]
        states[:, 2] = 5.0  # opponent row
        states[:, 3] = 3.0  # opponent col
        states[:, 4] = 0.0  # non-terminal

        action = 0  # North — intended row = -1 (OOB)
        batch = env.reward_batch(states, action)
        scalar = np.array([env.reward(s, action) for s in states], dtype=np.float64)
        np.testing.assert_allclose(
            batch,
            scalar,
            atol=1e-12,
            err_msg="Mismatch for OOB-intended-position particles",
        )


class _UniformActionSampler(ActionSampler):
    """Action sampler that draws uniformly from {0, 1, 2, 3, 4}."""

    def sample(self, belief_node=None) -> int:
        del belief_node
        return int(np.random.choice([0, 1, 2, 3, 4]))


def _python_rollout(
    env: LaserTagPOMDP,
    state: np.ndarray,
    max_depth: int,
    discount_factor: float,
    depth: int = 0,
) -> float:
    """Pure Python rollout that uses numpy RNG (bypasses the native override)."""
    from POMDPPlanners.planners.planners_utils.rollout import (
        python_random_rollout,
    )  # pylint: disable=import-outside-toplevel

    return python_random_rollout(
        state=state,
        depth=depth,
        action_sampler=_UniformActionSampler(),
        environment=env,
        discount_factor=discount_factor,
        max_depth=max_depth,
    )


class TestNativeDiscreteRollout:
    """Tests for the native C++ discrete LaserTag rollout kernel."""

    def test_lasertag_native_simulate_rollout_matches_python(self) -> None:
        """Native C++ rollout produces the same return distribution as the Python loop.

        Purpose: Validates that simulate_rollout_discrete in the C++ extension implements
            the same stochastic transition / reward / terminal logic as the Python
            ``Environment.simulate_random_rollout`` base-class loop, so that replacing
            the Python loop with the C++ kernel does not change the algorithm's value
            estimates in expectation.

        Given: A LaserTagPOMDP with default walls and dangerous areas, a fixed
            initial state, max_depth=15, discount=0.95, and 500 independent
            rollout trials.
        When: 500 trials are run with both the pure Python path (NumPy RNG) and
            the native C++ path (mt19937_64 RNG), each seeded independently before
            the batch.
        Then: The Monte-Carlo means of the two return distributions agree within
            0.5 standard deviations of the Python distribution.

        Test type: integration
        """
        env = LaserTagPOMDP(discount_factor=0.95)
        state = np.array([0.0, 0.0, 6.0, 5.0, 0.0])
        max_depth = 15
        discount = 0.95
        n_trials = 500

        # Python path (base-class loop, uses numpy RNG).
        np.random.seed(0)
        python_returns = np.array(
            [_python_rollout(env, state, max_depth, discount) for _ in range(n_trials)]
        )

        # Native C++ path (uses pomdp_native::default_rng, mt19937_64).
        from POMDPPlanners.environments.laser_tag_pomdp import (
            _native,
        )  # pylint: disable=import-outside-toplevel

        _native.set_seed(0)
        native_returns = np.array(
            [
                env.simulate_random_rollout(
                    state=state,
                    action_sampler=_UniformActionSampler(),
                    max_depth=max_depth,
                    discount_factor=discount,
                )
                for _ in range(n_trials)
            ]
        )

        py_mean = float(np.mean(python_returns))
        nat_mean = float(np.mean(native_returns))
        scale = float(np.std(python_returns)) + 1e-6
        diff = abs(nat_mean - py_mean)
        assert diff < 0.5 * scale, (
            f"Native mean {nat_mean:.3f} differs from Python mean {py_mean:.3f} "
            f"by {diff:.3f} (scale={scale:.3f}). C++ and Python rollout kernels "
            "implement different distributions."
        )

    def test_lasertag_native_rollout_returns_finite_float(self) -> None:
        """Native rollout returns a finite float from a valid initial state.

        Purpose: Smoke-test that simulate_rollout_discrete does not raise and
            returns a finite value.

        Given: A default LaserTagPOMDP, a non-terminal state, max_depth=10.
        When: simulate_rollout_discrete is called via the LaserTagPOMDP override.
        Then: Result is a finite float.

        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95)
        state = np.array([0.0, 0.0, 6.0, 5.0, 0.0])
        from POMDPPlanners.environments.laser_tag_pomdp import (
            _native,
        )  # pylint: disable=import-outside-toplevel

        _native.set_seed(7)
        result = env.simulate_random_rollout(
            state=state,
            action_sampler=_UniformActionSampler(),
            max_depth=10,
            discount_factor=0.95,
        )
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_lasertag_native_rollout_terminal_state_returns_zero(self) -> None:
        """Native rollout returns 0.0 when the initial state is already terminal.

        Purpose: Validates that the terminal short-circuit in the C++ kernel
            works correctly — no further steps are taken.

        Given: A LaserTagPOMDP and a terminal state (state[4] == 1.0).
        When: simulate_rollout_discrete is called with the terminal state.
        Then: Returns exactly 0.0.

        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95)
        terminal_state = np.array([3.0, 2.0, 3.0, 2.0, 1.0])
        from POMDPPlanners.environments.laser_tag_pomdp import (
            _native,
        )  # pylint: disable=import-outside-toplevel

        _native.set_seed(0)
        result = env.simulate_random_rollout(
            state=terminal_state,
            action_sampler=_UniformActionSampler(),
            max_depth=20,
            discount_factor=0.95,
        )
        assert result == 0.0

    def test_lasertag_native_rollout_respects_max_depth(self) -> None:
        """Native rollout accumulates at most max_depth steps.

        Purpose: Verifies that the C++ kernel stops after max_depth steps even
            when no terminal state is reached, by checking that the return
            magnitude is bounded by max possible per-step reward times geometric sum.

        Given: A non-terminal state, max_depth=5, discount=1.0.
        When: simulate_rollout_discrete is called.
        Then: |return| <= max_step_abs_reward * max_depth.

        Test type: unit
        """
        env = LaserTagPOMDP(
            discount_factor=0.95,
            tag_reward=10.0,
            tag_penalty=10.0,
            step_cost=1.0,
            dangerous_area_penalty=5.0,
        )
        state = np.array([0.0, 0.0, 6.0, 5.0, 0.0])
        max_depth = 5
        from POMDPPlanners.environments.laser_tag_pomdp import (
            _native,
        )  # pylint: disable=import-outside-toplevel

        _native.set_seed(99)
        result = env.simulate_random_rollout(
            state=state,
            action_sampler=_UniformActionSampler(),
            max_depth=max_depth,
            discount_factor=1.0,
        )
        max_abs = (10.0 + 5.0) * max_depth  # tag_reward + penalty, 5 steps
        assert abs(result) <= max_abs + 1e-9


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
