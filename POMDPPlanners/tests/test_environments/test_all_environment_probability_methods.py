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
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
    RockSamplePOMDP,
    RockSampleState,
    RockSampleStateTransitionModel,
)
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP, SanityStateTransitionModel
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP, TigerStateTransition
from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP, CartPoleStateTransition
from POMDPPlanners.environments.mountain_car_pomdp import (
    MountainCarPOMDP,
    MountainCarTransition,
)
from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_pomdp import (
    SafeAntVelocityPOMDP,
    SafeAntVelocityStateTransition,
)
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


class TestRockSamplePOMDPProbability:
    """Test probability method for RockSample POMDP."""

    def test_rock_sample_movement_probability(self):
        """Test RockSample POMDP probability method with movement action.

        Purpose: Validates that probability() method matches empirical distribution

        Given: RockSample POMDP with basic configuration and movement action
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        pomdp = RockSamplePOMDP(map_size=(5, 5))

        # Get initial state
        initial_state = pomdp.initial_state_dist().sample()[0]

        # Get a valid action (movement action - deterministic)
        actions = pomdp.get_actions()
        action = actions[1]  # North

        transition = RockSampleStateTransitionModel(state=initial_state, action=action, pomdp=pomdp)
        results = validate_probability_matches_empirical_distribution(
            transition, num_samples=1000, max_js_divergence=0.01
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.01  # Should be nearly perfect (deterministic)
        assert results["num_unique_states"] == 1  # Deterministic transition

    def test_rock_sample_sample_action_probability(self):
        """Test RockSample POMDP probability method with sample action.

        Purpose: Validates that probability() method works correctly for sample actions

        Given: RockSample POMDP with robot at a rock position
        When: Executing sample action and comparing probabilities to empirical distribution
        Then: Probabilities match within tolerance (deterministic transition)

        Test type: unit
        """
        # Create environment with known rock positions
        rock_positions = [(0, 0), (2, 2)]
        pomdp = RockSamplePOMDP(map_size=(5, 5), rock_positions=rock_positions, init_pos=(0, 0))

        # Create state where robot is at a rock position
        initial_state = RockSampleState(robot_pos=(0, 0), rocks=(True, True))

        # Sample action
        action = 0

        transition = RockSampleStateTransitionModel(state=initial_state, action=action, pomdp=pomdp)
        results = validate_probability_matches_empirical_distribution(
            transition, num_samples=1000, max_js_divergence=0.01
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.01  # Deterministic
        assert results["num_unique_states"] == 1  # Only one possible next state


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


class TestCartPolePOMDPProbability:
    """Test probability method for CartPole POMDP."""

    def test_cartpole_transition_probability(self):
        """Test CartPole POMDP probability method with physics-based transition.

        Purpose: Validates that probability() method matches empirical distribution

        Given: CartPole POMDP with basic configuration and action
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
        pomdp = CartPolePOMDP(discount_factor=0.95, noise_cov=noise_cov)

        # Get initial state
        initial_state = pomdp.initial_state_dist().sample()[0]

        # Get a valid action
        actions = pomdp.get_actions()
        action = actions[0]  # Left force

        transition = CartPoleStateTransition(
            state=initial_state,
            action=action,
            force_mag=pomdp.force_mag,
            total_mass=pomdp.total_mass,
            polemass_length=pomdp.polemass_length,
            gravity=pomdp.gravity,
            length=pomdp.length,
            kinematics_integrator=pomdp.kinematics_integrator,
            tau=pomdp.tau,
            masspole=pomdp.masspole,
        )
        results = validate_probability_matches_empirical_distribution(
            transition, num_samples=1000, max_js_divergence=0.01
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.01  # Should be nearly perfect (deterministic)
        assert results["num_unique_states"] == 1  # Deterministic transition


class TestMountainCarPOMDPProbability:
    """Test probability method for Mountain Car POMDP."""

    def test_mountain_car_transition_probability(self):
        """Test Mountain Car POMDP probability method with physics-based transition.

        Purpose: Validates that probability() method matches empirical distribution

        Given: Mountain Car POMDP with basic configuration and action
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        pomdp = MountainCarPOMDP(discount_factor=0.95)

        # Get initial state
        initial_state = pomdp.initial_state_dist().sample()[0]

        # Get a valid action
        actions = pomdp.get_actions()
        action = actions[0]  # Reverse

        transition = MountainCarTransition(
            state=initial_state,
            action=action,
            power=pomdp.power,
            gravity=pomdp.gravity,
            max_speed=pomdp.max_speed,
            min_position=pomdp.min_position,
            max_position=pomdp.max_position,
        )
        results = validate_probability_matches_empirical_distribution(
            transition, num_samples=1000, max_js_divergence=0.01
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.01  # Should be nearly perfect (deterministic)
        assert results["num_unique_states"] == 1  # Deterministic transition


class TestSafetyAntVelocityPOMDPProbability:
    """Test probability method for Safety Ant Velocity POMDP."""

    def test_safety_ant_zero_force_probability(self):
        """Test Safety Ant Velocity POMDP probability method with zero force action.

        Purpose: Validates that probability() method matches empirical distribution for zero force

        Given: Safety Ant Velocity POMDP with zero force action (deterministic transition)
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        pomdp = SafeAntVelocityPOMDP(discount_factor=0.95)

        # Get initial state
        initial_state = pomdp.initial_state_dist().sample()[0]

        # Get zero force action
        action = 0  # No force

        transition = SafeAntVelocityStateTransition(
            state=initial_state,
            action=action,
            dt=pomdp.dt,
            mass=pomdp.mass,
            damping=pomdp.damping,
            max_force=pomdp.max_force,
        )
        results = validate_probability_matches_empirical_distribution(
            transition, num_samples=1000, max_js_divergence=0.01
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.01  # Should be nearly perfect (deterministic)
        assert results["num_unique_states"] == 1  # Deterministic transition

    @pytest.mark.skip(
        reason="SafetyAntVelocity with non-zero force has truly continuous distribution "
        "(uniform over a ring due to random force direction). Discrete empirical sampling "
        "validation is not appropriate for continuous distributions."
    )
    def test_safety_ant_nonzero_force_probability(self):
        """Test Safety Ant Velocity POMDP probability method with non-zero force action.

        Purpose: Validates that probability() method matches empirical distribution for stochastic force

        Given: Safety Ant Velocity POMDP with non-zero force action (stochastic transition)
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit

        Note: This test is skipped because the non-zero force action results in a truly
        continuous distribution (uniform over a ring in state space due to random force
        direction). The current validation approach using discrete empirical sampling is
        not suitable for continuous distributions, as every sample is unique and the
        empirical distribution becomes uniform.
        """
        pomdp = SafeAntVelocityPOMDP(discount_factor=0.95)

        # Get initial state
        initial_state = pomdp.initial_state_dist().sample()[0]

        # Get non-zero force action
        action = 2  # Medium force

        transition = SafeAntVelocityStateTransition(
            state=initial_state,
            action=action,
            dt=pomdp.dt,
            mass=pomdp.mass,
            damping=pomdp.damping,
            max_force=pomdp.max_force,
        )
        results = validate_probability_matches_empirical_distribution(
            transition, num_samples=1000, max_js_divergence=0.15
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.15  # More tolerance for stochastic transitions
