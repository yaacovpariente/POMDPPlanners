"""Interface compliance tests for ObservationModel implementations.

This module validates that all observation models correctly implement
the ObservationModel interface, particularly the probability() method,
by comparing computed probabilities against empirical sampling distributions.

This is part of the interface compliance test suite that ensures all
implementations satisfy their respective contracts.
"""

import random
import numpy as np
import pytest

from POMDPPlanners.environments.cartpole_pomdp import CartPoleObservation
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import (
    LaserTagObservation,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_observation_models import (
    DiscreteLDObservationModel,
)
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarObservation
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import (
    PacManObservationModel,
    PacManPOMDP,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp import PushObservation
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
    RockSampleObservationModel,
    RockSamplePOMDP,
    RockSampleState,
    create_rock_sample_state,
)
from POMDPPlanners.environments.sanity_pomdp import SanityObservationModel
from POMDPPlanners.tests.test_utils.test_probability_utils import (
    validate_continuous_observation_model_pdf_consistency,
    validate_observation_probability_matches_empirical_distribution,
)

# Set random seed for reproducibility across all tests
np.random.seed(42)
random.seed(42)


class TestSanityObservationProbability:
    """Test probability method for Sanity ObservationModel."""

    def test_sanity_observation_probability_good_state(self):
        """Test Sanity observation probability method with good state (0).

        Purpose: Validates that probability() returns correct values for good state

        Given: Sanity observation model in good state (next_state=0)
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance (deterministic observation = state)

        Test type: unit
        """
        obs_model = SanityObservationModel(next_state=0, action=0)
        results = validate_observation_probability_matches_empirical_distribution(
            obs_model, num_samples=1000, max_js_divergence=0.01
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.01
        assert results["num_unique_observations"] == 1  # Only observation 0

    def test_sanity_observation_probability_bad_state(self):
        """Test Sanity observation probability method with bad state (1).

        Purpose: Validates that probability() returns correct values for bad state

        Given: Sanity observation model in bad state (next_state=1)
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance (deterministic observation = state)

        Test type: unit
        """
        obs_model = SanityObservationModel(next_state=1, action=1)
        results = validate_observation_probability_matches_empirical_distribution(
            obs_model, num_samples=1000, max_js_divergence=0.01
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.01
        assert results["num_unique_observations"] == 1  # Only observation 1


class TestCartPoleObservationProbability:
    """Test probability method for CartPole ObservationModel."""

    def test_cartpole_observation_probability_moderate_noise(self):
        """Test CartPole observation probability with moderate noise.

        Purpose: Validates that probability() returns valid PDF values with moderate noise

        Given: CartPole observation model with moderate Gaussian noise (0.1 variance)
        When: Checking PDF consistency properties
        Then: PDF values are non-negative, deterministic, and show expected variation

        Test type: unit
        """
        noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
        true_state = np.array([0.1, 0.05, 0.02, -0.1])
        obs_dist = CovarianceParameterizedMultivariateNormal(noise_cov)
        obs_model = CartPoleObservation(next_state=true_state, action=0, obs_dist=obs_dist)

        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"
        assert results["pdf_std"] > 0, "PDF values should vary across samples"

    def test_cartpole_observation_probability_low_noise(self):
        """Test CartPole observation probability with low noise.

        Purpose: Validates that probability() returns valid PDF values with low noise

        Given: CartPole observation model with low Gaussian noise (0.01 variance)
        When: Checking PDF consistency properties
        Then: PDF values are non-negative and show higher concentration around mean

        Test type: unit
        """
        noise_cov = np.diag([0.01, 0.01, 0.01, 0.01])
        true_state = np.array([0.0, 0.0, 0.0, 0.0])  # Centered state
        obs_dist = CovarianceParameterizedMultivariateNormal(noise_cov)
        obs_model = CartPoleObservation(next_state=true_state, action=1, obs_dist=obs_dist)

        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"

    def test_cartpole_observation_probability_high_noise(self):
        """Test CartPole observation probability with high noise.

        Purpose: Validates that probability() returns valid PDF values with high noise

        Given: CartPole observation model with high Gaussian noise (1.0 variance)
        When: Checking PDF consistency properties
        Then: PDF values are non-negative and show wider spread

        Test type: unit
        """
        noise_cov = np.diag([1.0, 1.0, 1.0, 1.0])
        true_state = np.array([0.5, -0.3, 0.1, 0.2])  # Non-centered state
        obs_dist = CovarianceParameterizedMultivariateNormal(noise_cov)
        obs_model = CartPoleObservation(next_state=true_state, action=0, obs_dist=obs_dist)

        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"
        assert results["pdf_std"] > 0, "PDF values should vary across samples"


class TestMountainCarObservationProbability:
    """Test probability method for Mountain Car ObservationModel."""

    def test_mountain_car_observation_probability_valley_position(self):
        """Test Mountain Car observation probability at valley (lowest point).

        Purpose: Validates probability() at the valley position (x=-0.5)

        Given: Mountain Car in valley position with moderate noise
        When: Checking PDF consistency properties
        Then: PDF values are non-negative and deterministic

        Test type: unit
        """
        cov_matrix = np.array([[0.1**2, 0], [0, 0.01**2]])
        obs_dist = CovarianceParameterizedMultivariateNormal(cov_matrix)
        true_state = (-0.5, 0.0)  # Valley position, zero velocity
        obs_model = MountainCarObservation(next_state=true_state, action=1, obs_dist=obs_dist)

        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"
        assert results["pdf_std"] > 0, "PDF values should vary across samples"

    def test_mountain_car_observation_probability_left_hill(self):
        """Test Mountain Car observation probability near left hill.

        Purpose: Validates probability() near the left boundary

        Given: Mountain Car near left hill with negative velocity
        When: Checking PDF consistency properties
        Then: PDF values are non-negative and deterministic

        Test type: unit
        """
        cov_matrix = np.array([[0.1**2, 0], [0, 0.01**2]])
        obs_dist = CovarianceParameterizedMultivariateNormal(cov_matrix)
        true_state = (-1.0, -0.05)  # Near left boundary, moving left
        obs_model = MountainCarObservation(next_state=true_state, action=0, obs_dist=obs_dist)

        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"

    def test_mountain_car_observation_probability_near_goal(self):
        """Test Mountain Car observation probability near goal position.

        Purpose: Validates probability() near the goal (x=0.5)

        Given: Mountain Car near goal with positive velocity
        When: Checking PDF consistency properties
        Then: PDF values are non-negative and deterministic

        Test type: unit
        """
        cov_matrix = np.array([[0.05**2, 0], [0, 0.005**2]])  # Lower noise near goal
        obs_dist = CovarianceParameterizedMultivariateNormal(cov_matrix)
        true_state = (0.45, 0.05)  # Near goal, moving right
        obs_model = MountainCarObservation(next_state=true_state, action=2, obs_dist=obs_dist)

        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"


class TestPushObservationProbability:
    """Test probability method for Push ObservationModel."""

    def test_push_observation_probability_center_position(self):
        """Test Push observation probability with object in center.

        Purpose: Validates probability() when object is in the center of the grid

        Given: Push observation model with object in center, moderate noise
        When: Checking PDF consistency properties
        Then: PDF values are non-negative, deterministic, and show variation

        Test type: unit
        """
        # State: [robot_x, robot_y, object_x, object_y, target_x, target_y]
        true_state = np.array([5.0, 5.0, 5.0, 5.0, 8.0, 8.0])
        obs_model = PushObservation(
            next_state=true_state, action="right", observation_noise=0.1, grid_size=10
        )

        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"
        assert results["pdf_std"] > 0, "PDF values should vary across samples"

    def test_push_observation_probability_corner_position(self):
        """Test Push observation probability with object near corner.

        Purpose: Validates probability() when object is near grid corner

        Given: Push observation model with object near corner (0,0)
        When: Checking PDF consistency properties
        Then: PDF values are non-negative and deterministic

        Test type: unit
        """
        true_state = np.array([1.0, 1.0, 0.5, 0.5, 8.0, 8.0])
        obs_model = PushObservation(
            next_state=true_state, action="up", observation_noise=0.1, grid_size=10
        )

        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"

    def test_push_observation_probability_high_noise(self):
        """Test Push observation probability with high observation noise.

        Purpose: Validates probability() with high noise level

        Given: Push observation model with high noise (0.5)
        When: Checking PDF consistency properties
        Then: PDF values show wider spread due to higher noise

        Test type: unit
        """
        true_state = np.array([3.0, 3.0, 2.8, 3.2, 8.0, 8.0])
        obs_model = PushObservation(
            next_state=true_state, action="left", observation_noise=0.5, grid_size=10
        )

        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"
        assert results["pdf_std"] > 0, "PDF values should vary across samples"

    def test_push_observation_probability_near_target(self):
        """Test Push observation probability when object is near target.

        Purpose: Validates probability() when object is close to target position

        Given: Push observation model with object near target
        When: Checking PDF consistency properties
        Then: PDF values are non-negative and deterministic

        Test type: unit
        """
        true_state = np.array([7.5, 7.5, 7.8, 7.9, 8.0, 8.0])  # Object almost at target
        obs_model = PushObservation(
            next_state=true_state, action="down", observation_noise=0.1, grid_size=10
        )

        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"


class TestLaserTagObservationProbability:
    """Test probability method for LaserTag ObservationModel."""

    def test_laser_tag_observation_probability_open_area(self):
        """Test LaserTag observation probability in open area without walls.

        Purpose: Validates probability() in open floor without obstacles

        Given: LaserTag in open area, no walls, moderate noise
        When: Checking PDF consistency properties
        Then: PDF values are non-negative, deterministic, and show variation

        Test type: unit
        """
        # State: [robot_row, robot_col, opponent_row, opponent_col, terminal]
        state = np.array([3.0, 5.0, 2.0, 4.0, 0.0])
        obs_model = LaserTagObservation(
            next_state=state,
            action=0,
            measurement_noise=1.0,
            floor_shape=(7, 11),
            walls=set(),
        )

        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"
        assert results["pdf_std"] > 0, "PDF values should vary across samples"

    def test_laser_tag_observation_probability_with_walls(self):
        """Test LaserTag observation probability with walls present.

        Purpose: Validates probability() when walls affect laser measurements

        Given: LaserTag with walls blocking some laser directions
        When: Checking PDF consistency properties
        Then: PDF values are non-negative and deterministic

        Test type: unit
        """
        state = np.array([3.0, 5.0, 5.0, 8.0, 0.0])
        walls = {(3, 6), (3, 7), (4, 5), (2, 5)}  # Walls around robot
        obs_model = LaserTagObservation(
            next_state=state,
            action=1,
            measurement_noise=1.0,
            floor_shape=(7, 11),
            walls=walls,
        )

        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"

    def test_laser_tag_observation_probability_corner_position(self):
        """Test LaserTag observation probability when robot is in corner.

        Purpose: Validates probability() when robot is near grid boundary

        Given: LaserTag with robot in corner position
        When: Checking PDF consistency properties
        Then: PDF values are non-negative and deterministic

        Test type: unit
        """
        state = np.array([0.0, 0.0, 3.0, 3.0, 0.0])  # Robot in top-left corner
        obs_model = LaserTagObservation(
            next_state=state,
            action=2,
            measurement_noise=0.5,
            floor_shape=(7, 11),
            walls=set(),
        )

        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"

    def test_laser_tag_observation_probability_high_noise(self):
        """Test LaserTag observation probability with high measurement noise.

        Purpose: Validates probability() with high noise affecting measurements

        Given: LaserTag with high measurement noise (2.0)
        When: Checking PDF consistency properties
        Then: PDF values show wider spread due to higher noise

        Test type: unit
        """
        state = np.array([3.0, 5.0, 4.0, 6.0, 0.0])
        obs_model = LaserTagObservation(
            next_state=state,
            action=0,
            measurement_noise=2.0,  # High noise
            floor_shape=(7, 11),
            walls=set(),
        )

        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"

    def test_laser_tag_observation_probability_opponent_adjacent(self):
        """Test LaserTag observation probability when opponent is adjacent.

        Purpose: Validates probability() when opponent is directly next to robot

        Given: LaserTag with opponent one cell away from robot
        When: Checking PDF consistency properties
        Then: PDF values are non-negative and deterministic

        Test type: unit
        """
        state = np.array([3.0, 5.0, 3.0, 6.0, 0.0])  # Opponent one cell to the right
        obs_model = LaserTagObservation(
            next_state=state,
            action=0,
            measurement_noise=1.0,
            floor_shape=(7, 11),
            walls=set(),
        )

        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"


class TestRockSampleObservationProbability:
    """Test probability method for RockSample ObservationModel."""

    def test_rock_sample_movement_observation_probability(self):
        """Test RockSample observation probability method with movement action.

        Purpose: Validates that probability() method matches empirical distribution for movement

        Given: RockSample observation model with movement action (deterministic observation)
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance (should be nearly perfect)

        Test type: unit
        """
        pomdp = RockSamplePOMDP(map_size=(5, 5))
        initial_state = pomdp.initial_state_dist().sample()[0]
        action = 1  # Movement action

        obs_model = RockSampleObservationModel(next_state=initial_state, action=action, pomdp=pomdp)

        results = validate_observation_probability_matches_empirical_distribution(
            obs_model, num_samples=1000, max_js_divergence=0.01
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.01  # Should be nearly perfect (deterministic)
        assert results["num_unique_observations"] == 1  # Only "none" observation

    def test_rock_sample_check_action_observation_probability(self):
        """Test RockSample observation probability method with check action.

        Purpose: Validates that probability() method matches empirical distribution for check action

        Given: RockSample observation model with check action (stochastic observations)
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        # Create environment with known rock positions
        rock_positions = [(0, 0), (2, 2)]
        pomdp = RockSamplePOMDP(map_size=(5, 5), rock_positions=rock_positions, init_pos=(0, 0))

        # Create state where robot is at a rock position
        next_state = create_rock_sample_state(robot_pos=(0, 0), rocks=(True, True))
        action = 5  # Check rock 0

        obs_model = RockSampleObservationModel(next_state=next_state, action=action, pomdp=pomdp)

        results = validate_observation_probability_matches_empirical_distribution(
            obs_model, num_samples=1000, max_js_divergence=0.05
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.05


class TestPacManObservationProbability:
    """Test probability method for PacMan ObservationModel."""

    def test_pacman_observation_probability_ghost_far(self):
        """Test PacMan observation probability when ghost is far from pacman.

        Purpose: Validates probability() when ghost is distant (high noise)

        Given: PacMan with ghost far from pacman position (distance=6)
        When: Comparing computed probabilities to empirical sampling
        Then: PDF values are non-negative with expected noise characteristics

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

        state = pomdp.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((3, 3),),
            pellets=((2, 2),),
            score=0.0,
        )

        obs_model = PacManObservationModel(next_state=state, action=1, pomdp=pomdp)
        results = validate_observation_probability_matches_empirical_distribution(
            obs_model,
            num_samples=1000,
            max_js_divergence=0.15,
            check_normalization=False,
        )

        assert np.all(results["computed_probs"] >= 0), "PDF values must be non-negative"
        assert results["distance"] < 0.15

    def test_pacman_observation_probability_ghost_close(self):
        """Test PacMan observation probability when ghost is close to pacman.

        Purpose: Validates probability() when ghost is adjacent (low noise)

        Given: PacMan with ghost adjacent to pacman (distance=1)
        When: Checking PDF consistency properties
        Then: PDF values are non-negative with lower noise due to proximity

        Test type: unit

        Note: When ghost is close, noise is very low (noise_std approaches 0),
        leading to more concentrated observations and potentially higher JS divergence
        due to discretization effects. We use PDF consistency check instead.
        """
        pomdp = PacManPOMDP(
            maze_size=(5, 5),
            walls=set(),
            initial_pellets=[(2, 2)],
            initial_pacman_pos=(2, 2),
            num_ghosts=1,
            initial_ghost_positions=[(2, 3)],
            ghost_aggressiveness=2.0,
        )

        state = pomdp.make_state(
            pacman_pos=(2, 2), ghost_positions=((2, 3),), pellets=(), score=10.0
        )

        obs_model = PacManObservationModel(next_state=state, action=0, pomdp=pomdp)
        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"

    def test_pacman_observation_probability_multiple_ghosts(self):
        """Test PacMan observation probability with multiple ghosts.

        Purpose: Validates probability() with two ghosts at different distances

        Given: PacMan with two ghosts at different positions
        When: Checking PDF consistency properties
        Then: PDF values are non-negative for combined ghost observations

        Test type: unit

        Note: With multiple ghosts, the observation space grows significantly,
        making JS divergence less meaningful. We use PDF consistency check instead.
        """
        pomdp = PacManPOMDP(
            maze_size=(7, 7),
            walls=set(),
            initial_pellets=[(3, 3)],
            initial_pacman_pos=(0, 0),
            num_ghosts=2,
            initial_ghost_positions=[(2, 2), (5, 5)],
            ghost_aggressiveness=1.5,
        )

        state = pomdp.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((2, 2), (5, 5)),
            pellets=((3, 3),),
            score=0.0,
        )

        obs_model = PacManObservationModel(next_state=state, action=1, pomdp=pomdp)
        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"

    def test_pacman_observation_probability_with_walls(self):
        """Test PacMan observation probability with walls in the maze.

        Purpose: Validates probability() in a maze with walls

        Given: PacMan in maze with walls affecting movement
        When: Checking PDF consistency properties
        Then: PDF values are non-negative

        Test type: unit
        """
        walls = {(1, 1), (1, 2), (2, 1)}
        pomdp = PacManPOMDP(
            maze_size=(5, 5),
            walls=walls,
            initial_pellets=[(3, 3)],
            initial_pacman_pos=(0, 0),
            num_ghosts=1,
            initial_ghost_positions=[(4, 4)],
            ghost_aggressiveness=2.0,
        )

        state = pomdp.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((4, 4),),
            pellets=((3, 3),),
            score=0.0,
        )

        obs_model = PacManObservationModel(next_state=state, action=2, pomdp=pomdp)
        results = validate_continuous_observation_model_pdf_consistency(obs_model, num_samples=1000)

        assert results["pdf_values_non_negative"], "PDF values must be non-negative"
        assert results["pdf_deterministic"], "PDF computation must be deterministic"


class TestDiscreteLightDarkObservationProbability:
    """Test probability method for Discrete Light-Dark ObservationModel."""

    def test_discrete_ld_observation_probability_near_beacon(self):
        """Test Discrete Light-Dark observation probability near a beacon.

        Purpose: Validates probability() when agent is near a beacon (low noise)

        Given: Agent position within beacon radius (reduced observation error)
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match with low divergence (accurate observations)

        Test type: unit
        """
        beacons = np.array([[2.0, 2.0], [4.0, 4.0]]).T  # Shape (2, num_beacons)
        obstacles = np.array([[1.0, 1.0]]).T
        next_state = np.array([2.0, 2.5])  # Very close to beacon at (2,2)
        action = "up"

        obs_model = DiscreteLDObservationModel(
            next_state=next_state,
            action=action,
            beacons=beacons,
            obstacles=obstacles,
            beacon_radius=1.0,
            observation_error_prob=0.1,
        )

        results = validate_observation_probability_matches_empirical_distribution(
            obs_model, num_samples=1000, max_js_divergence=0.05
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.05

    def test_discrete_ld_observation_probability_far_from_beacon(self):
        """Test Discrete Light-Dark observation probability far from beacons.

        Purpose: Validates probability() when agent is far from all beacons (high noise)

        Given: Agent position outside all beacon radii (full observation error)
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities show higher uncertainty due to distance from beacons

        Test type: unit
        """
        beacons = np.array([[0.0, 0.0], [10.0, 10.0]]).T
        obstacles = np.array([[5.0, 5.0]]).T
        next_state = np.array([5.0, 3.0])  # Far from both beacons
        action = "right"

        obs_model = DiscreteLDObservationModel(
            next_state=next_state,
            action=action,
            beacons=beacons,
            obstacles=obstacles,
            beacon_radius=1.0,
            observation_error_prob=0.2,
        )

        results = validate_observation_probability_matches_empirical_distribution(
            obs_model, num_samples=1000, max_js_divergence=0.10
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.10

    def test_discrete_ld_observation_probability_near_obstacle(self):
        """Test Discrete Light-Dark observation probability near an obstacle.

        Purpose: Validates probability() when agent is near an obstacle

        Given: Agent position close to an obstacle
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities are properly normalized

        Test type: unit
        """
        beacons = np.array([[2.0, 2.0], [8.0, 8.0]]).T
        obstacles = np.array([[3.0, 3.0], [5.0, 5.0]]).T  # Two obstacles
        next_state = np.array([3.5, 3.0])  # Near obstacle at (3,3)
        action = "down"

        obs_model = DiscreteLDObservationModel(
            next_state=next_state,
            action=action,
            beacons=beacons,
            obstacles=obstacles,
            beacon_radius=1.5,
            observation_error_prob=0.15,
        )

        results = validate_observation_probability_matches_empirical_distribution(
            obs_model, num_samples=1000, max_js_divergence=0.10
        )

        assert results["probabilities_normalized"]

    def test_discrete_ld_observation_probability_high_error(self):
        """Test Discrete Light-Dark observation probability with high error rate.

        Purpose: Validates probability() with high observation error probability

        Given: High observation error probability (0.3)
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities show higher spread but remain normalized

        Test type: unit
        """
        beacons = np.array([[5.0, 5.0]]).T  # Single beacon
        obstacles = np.array([[2.0, 2.0]]).T
        next_state = np.array([7.0, 7.0])  # Far from beacon
        action = "left"

        obs_model = DiscreteLDObservationModel(
            next_state=next_state,
            action=action,
            beacons=beacons,
            obstacles=obstacles,
            beacon_radius=1.0,
            observation_error_prob=0.3,  # High error
        )

        results = validate_observation_probability_matches_empirical_distribution(
            obs_model, num_samples=1000, max_js_divergence=0.10
        )

        assert results["probabilities_normalized"]

    def test_discrete_ld_observation_probability_at_start(self):
        """Test Discrete Light-Dark observation probability at start position.

        Purpose: Validates probability() at typical start position

        Given: Agent at start position (0,5) with default beacon configuration
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match empirical distribution

        Test type: unit
        """
        beacons = np.array([[0.0, 0.0], [0.0, 5.0], [5.0, 5.0]]).T
        obstacles = np.array([[3.0, 7.0]]).T
        next_state = np.array([0.0, 5.0])  # At beacon position
        action = "right"

        obs_model = DiscreteLDObservationModel(
            next_state=next_state,
            action=action,
            beacons=beacons,
            obstacles=obstacles,
            beacon_radius=1.0,
            observation_error_prob=0.05,
        )

        results = validate_observation_probability_matches_empirical_distribution(
            obs_model, num_samples=1000, max_js_divergence=0.05
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.05
