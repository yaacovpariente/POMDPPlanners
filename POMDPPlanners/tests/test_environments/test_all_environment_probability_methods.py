"""Tests for probability methods across all POMDP environments.

This module validates that all state transition models correctly implement
the probability() method by comparing computed probabilities against empirical
sampling distributions.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import (
    LaserTagPOMDP,
    LaserTagStateTransition,
)
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import (
    PacManPOMDP,
    PacManState,
    PacManStateTransitionModel,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp import (
    PushPOMDP,
    PushStateTransition,
)
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP, SanityStateTransitionModel
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP, TigerStateTransition
from POMDPPlanners.tests.test_utils.test_probability_utils import (
    validate_probability_matches_empirical_distribution,
)


class TestTigerPOMDPProbability:
    """Test probability method for Tiger POMDP."""

    def test_tiger_listen_action_probability(self):
        """Test Tiger POMDP probability method with listen action.

        Purpose: Validates that probability() method matches empirical distribution for listen action

        Given: Tiger POMDP with listen action (deterministic state transition)
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        transition = TigerStateTransition(state="tiger_left", action="listen")
        results = validate_probability_matches_empirical_distribution(
            transition, num_samples=1000, max_js_divergence=0.01
        )

        # Listen action should be deterministic (state doesn't change)
        assert results["probabilities_normalized"]
        assert results["distance"] < 0.01  # Should be nearly perfect
        assert results["num_unique_states"] == 1  # Deterministic transition

    def test_tiger_open_door_action_probability(self):
        """Test Tiger POMDP probability method with open door action.

        Purpose: Validates that probability() method matches empirical distribution for open action

        Given: Tiger POMDP with open_left action (stochastic state transition)
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        transition = TigerStateTransition(state="tiger_left", action="open_left")
        results = validate_probability_matches_empirical_distribution(
            transition, num_samples=1000, max_js_divergence=0.05
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.05
        assert results["num_unique_states"] == 2  # Two possible tiger positions


class TestPacManPOMDPProbability:
    """Test probability method for PacMan POMDP."""

    def test_pacman_basic_movement_probability(self):
        """Test PacMan POMDP probability method with basic movement.

        Purpose: Validates that probability() method matches empirical distribution

        Given: PacMan POMDP with simple configuration and movement action
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        pomdp = PacManPOMDP(
            maze_size=(5, 5),
            walls=set(),
            initial_pellets=[(2, 2)],
            initial_pacman_pos=(0, 0),
            num_ghosts=1,
            initial_ghost_positions=[(3, 3)],
            ghost_aggressiveness=2.0,
        )

        state = PacManState(
            pacman_pos=(0, 0), ghost_positions=((3, 3),), pellets=((2, 2),), score=0
        )

        transition = PacManStateTransitionModel(state, action=1, pomdp=pomdp)  # Move east
        results = validate_probability_matches_empirical_distribution(
            transition, num_samples=1000, max_js_divergence=0.05
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.05


class TestPushPOMDPProbability:
    """Test probability method for Push POMDP."""

    def test_push_movement_probability(self):
        """Test Push POMDP probability method with movement action.

        Purpose: Validates that probability() method matches empirical distribution

        Given: Push POMDP with basic configuration and movement action
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        pomdp = PushPOMDP(discount_factor=0.95, grid_size=5)

        # Get initial state
        initial_state = pomdp.initial_state_dist().sample()[0]

        # Get a valid action
        actions = pomdp.get_actions()
        action = actions[1]  # East

        transition = PushStateTransition(
            state=initial_state,
            action=action,
            grid_size=pomdp.grid_size,
            push_threshold=pomdp.push_threshold,
            friction_coefficient=pomdp.friction_coefficient,
            obstacles=pomdp.obstacles,
            obstacle_radius=pomdp.obstacle_radius,
        )
        results = validate_probability_matches_empirical_distribution(
            transition, num_samples=1000, max_js_divergence=0.10
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.10  # More tolerance for continuous states


class TestLaserTagPOMDPProbability:
    """Test probability method for LaserTag POMDP."""

    def test_laser_tag_movement_probability(self):
        """Test LaserTag POMDP probability method with movement action.

        Purpose: Validates that probability() method matches empirical distribution

        Given: LaserTag POMDP with basic configuration and movement action
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        pomdp = LaserTagPOMDP(discount_factor=0.95)

        # Get initial state
        initial_state = pomdp.initial_state_dist().sample()[0]

        # Get a valid action
        actions = pomdp.get_actions()
        action = actions[1]  # East

        transition = LaserTagStateTransition(
            state=initial_state,
            action=action,
            floor_shape=pomdp.floor_shape,
            walls=pomdp.walls,
        )
        results = validate_probability_matches_empirical_distribution(
            transition, num_samples=1000, max_js_divergence=0.05
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.05


class TestSanityPOMDPProbability:
    """Test probability method for Sanity POMDP."""

    def test_sanity_transition_probability(self):
        """Test Sanity POMDP probability method.

        Purpose: Validates that probability() method matches empirical distribution

        Given: Sanity POMDP with basic configuration and action
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        pomdp = SanityPOMDP()

        # Get initial state
        initial_state = pomdp.initial_state_dist().sample()[0]

        # Get a valid action
        actions = pomdp.get_actions()
        action = actions[0]

        transition = SanityStateTransitionModel(initial_state, action)
        results = validate_probability_matches_empirical_distribution(
            transition, num_samples=1000, max_js_divergence=0.05
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.05
