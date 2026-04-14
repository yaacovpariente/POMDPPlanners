"""Tests for PacMan vectorized particle belief updater."""

# pylint: disable=protected-access

import numpy as np
import pytest

from POMDPPlanners.environments.pacman_pomdp import PacManPOMDP, PacManState
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_beliefs.pacman_vectorized_updater import (
    PacManVectorizedUpdater,
)


@pytest.fixture()
def simple_env():
    return PacManPOMDP(
        maze_size=(5, 5),
        walls={(2, 2)},
        initial_pellets=[(1, 1), (3, 3)],
        initial_pacman_pos=(0, 0),
        num_ghosts=1,
        initial_ghost_positions=[(4, 4)],
        ghost_aggressiveness=2.0,
        ghost_coordination="independent",
        discount_factor=0.95,
    )


@pytest.fixture()
def updater(simple_env):
    return PacManVectorizedUpdater.from_environment(simple_env)


@pytest.fixture()
def sample_particles(simple_env):
    np.random.seed(42)
    states = simple_env.initial_state_dist().sample(n_samples=10)
    return simple_env.states_to_array(states)


class TestFromEnvironmentConstruction:
    def test_creates_without_error(self, simple_env):
        """Test that from_environment constructs an updater successfully.

        Purpose: Validates updater creation from environment.

        Given: A configured PacManPOMDP environment.
        When: PacManVectorizedUpdater.from_environment is called.
        Then: An updater instance is created without errors.

        Test type: unit
        """
        updater = PacManVectorizedUpdater.from_environment(simple_env)
        assert updater.maze_size == (5, 5)
        assert updater.num_ghosts == 1
        assert updater.num_pellets == 2
        assert updater.state_dim == simple_env._state_dim


class TestBatchTransition:
    def test_shape_preserved(self, updater, sample_particles):
        """Test that batch_transition preserves particle array shape.

        Purpose: Validates output shape matches input shape.

        Given: Particles array of shape (10, d).
        When: batch_transition is called with action 2 (South).
        Then: Output shape is (10, d).

        Test type: unit
        """
        result = updater.batch_transition(sample_particles, 2)
        assert result.shape == sample_particles.shape

    def test_terminal_unchanged(self, updater, simple_env):
        """Test that terminal particles remain unchanged after transition.

        Purpose: Validates terminal particle preservation.

        Given: A particle array where some particles are terminal.
        When: batch_transition is called.
        Then: Terminal particles are identical to input.

        Test type: unit
        """
        state = PacManState(
            pacman_pos=(0, 0),
            ghost_positions=((4, 4),),
            pellets=((1, 1), (3, 3)),
            score=0,
            terminal=True,
        )
        arr = simple_env.state_to_array(state).reshape(1, -1)
        result = updater.batch_transition(arr, 2)
        np.testing.assert_array_equal(result, arr)

    def test_pacman_moves_correctly(self, updater, simple_env):
        """Test that PacMan moves to correct position on valid move.

        Purpose: Validates pacman movement direction.

        Given: PacMan at (0,0) in a 5x5 maze.
        When: Action South (2) is applied.
        Then: PacMan moves to (1,0).

        Test type: unit
        """
        state = PacManState(
            pacman_pos=(0, 0),
            ghost_positions=((4, 4),),
            pellets=((1, 1), (3, 3)),
            score=0,
            terminal=False,
        )
        np.random.seed(42)
        arr = simple_env.state_to_array(state).reshape(1, -1)
        result = updater.batch_transition(arr, 2)  # South
        assert result[0, updater._idx_pac_row] == 1
        assert result[0, updater._idx_pac_col] == 0

    def test_pellet_collection(self, updater, simple_env):
        """Test that pellet bitmask flips and score increments on collection.

        Purpose: Validates pellet collection mechanics.

        Given: PacMan at (1,0) with pellet at (1,1).
        When: Action East (1) moves PacMan onto the pellet.
        Then: Pellet bitmask flips to 0 and score increments.

        Test type: unit
        """
        state = PacManState(
            pacman_pos=(1, 0),
            ghost_positions=((4, 4),),
            pellets=((1, 1), (3, 3)),
            score=0,
            terminal=False,
        )
        np.random.seed(42)
        arr = simple_env.state_to_array(state).reshape(1, -1)
        result = updater.batch_transition(arr, 1)  # East -> (1,1)
        # Pellet at index 0 ((1,1)) should be collected
        assert result[0, updater._idx_pellets_start] == 0.0
        assert result[0, updater._idx_score] > arr[0, updater._idx_score]

    def test_ghost_collision_sets_terminal(self, updater, simple_env):
        """Test that ghost collision sets terminal flag.

        Purpose: Validates collision detection in vectorized transition.

        Given: PacMan at (3,4) with ghost at (4,4).
        When: Action South (2) moves PacMan to ghost position.
        Then: Terminal flag is set to 1.

        Test type: unit
        """
        state = PacManState(
            pacman_pos=(3, 4),
            ghost_positions=((4, 4),),
            pellets=((1, 1), (3, 3)),
            score=0,
            terminal=False,
        )
        np.random.seed(123)
        arr = simple_env.state_to_array(state).reshape(1, -1)
        # Move south to (4,4) where ghost might be
        result = updater.batch_transition(arr, 2)
        # PacMan should be at (4,4)
        assert result[0, updater._idx_pac_row] == 4
        assert result[0, updater._idx_pac_col] == 4
        # If ghost stayed at (4,4), should be terminal
        ghost_r = result[0, updater._idx_ghosts_start]
        ghost_c = result[0, updater._idx_ghosts_start + 1]
        if ghost_r == 4 and ghost_c == 4:
            assert result[0, updater._idx_terminal] == 1.0


class TestBatchObservationLogLikelihood:
    def test_shape(self, updater, sample_particles):
        """Test output shape of batch_observation_log_likelihood.

        Purpose: Validates output shape is (N,).

        Given: 10 particles and an observation.
        When: batch_observation_log_likelihood is called.
        Then: Output shape is (10,).

        Test type: unit
        """
        obs = np.array([4.0, 4.0])
        ll = updater.batch_observation_log_likelihood(sample_particles, 0, obs)
        assert ll.shape == (10,)

    def test_terminal_observation(self, updater, simple_env):
        """Test terminal particle handling in log-likelihood.

        Purpose: Validates terminal particles get correct log-likelihood.

        Given: A terminal particle and a terminal observation (-1, -1).
        When: batch_observation_log_likelihood is called.
        Then: Terminal particle gets 0.0 log-likelihood.

        Test type: unit
        """
        state = PacManState(
            pacman_pos=(0, 0),
            ghost_positions=((4, 4),),
            pellets=(),
            score=20,
            terminal=True,
        )
        arr = simple_env.state_to_array(state).reshape(1, -1)
        obs = np.array([-1.0, -1.0])
        ll = updater.batch_observation_log_likelihood(arr, 0, obs)
        assert ll[0] == 0.0

    def test_closer_ghost_higher_precision(self, updater, simple_env):
        """Test that closer ghosts yield higher log-likelihood for accurate obs.

        Purpose: Validates distance-dependent observation noise.

        Given: Two particles — one with ghost close to PacMan, one far.
        When: Observation matches the true ghost position exactly.
        Then: Close-ghost particle has higher log-likelihood.

        Test type: unit
        """
        close_state = PacManState(
            pacman_pos=(0, 0),
            ghost_positions=((0, 1),),
            pellets=((1, 1), (3, 3)),
            score=0,
            terminal=False,
        )
        far_state = PacManState(
            pacman_pos=(0, 0),
            ghost_positions=((4, 4),),
            pellets=((1, 1), (3, 3)),
            score=0,
            terminal=False,
        )
        close_arr = simple_env.state_to_array(close_state).reshape(1, -1)
        far_arr = simple_env.state_to_array(far_state).reshape(1, -1)
        particles = np.vstack([close_arr, far_arr])

        # Observation matches close ghost exactly
        obs = np.array([0.0, 1.0])
        ll = updater.batch_observation_log_likelihood(particles, 0, obs)
        assert ll[0] > ll[1]


class TestConfigId:
    def test_deterministic(self, simple_env):
        """Test that config_id is deterministic.

        Purpose: Validates reproducible config_id generation.

        Given: Two updaters from the same environment.
        When: config_id is accessed.
        Then: Both return the same string.

        Test type: unit
        """
        u1 = PacManVectorizedUpdater.from_environment(simple_env)
        u2 = PacManVectorizedUpdater.from_environment(simple_env)
        assert u1.config_id == u2.config_id
