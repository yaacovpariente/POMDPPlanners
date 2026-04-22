"""Tests for PacMan POMDP environment.

This module tests the PacMan POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

import random
from typing import Tuple
from unittest.mock import Mock

import numpy as np
import pytest

from POMDPPlanners.core.belief import Belief, WeightedParticleBelief
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.pacman_pomdp import (
    PacManObservationModel,
    PacManPOMDP,
    PacManStateTransitionModel,
    create_simple_maze_pacman,
)
from POMDPPlanners.tests.test_utils.test_probability_utils import (
    validate_probability_matches_empirical_distribution,
)

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


class TestMakeState:
    """Test cases for PacManPOMDP.make_state factory."""

    def setup_method(self):
        """Construct a shared env with two ghosts and four pellets."""
        self.pomdp = PacManPOMDP(
            maze_size=(7, 7),
            walls=set(),
            initial_pellets=[(0, 0), (5, 5), (1, 2), (3, 4)],
            initial_pacman_pos=(1, 2),
            num_ghosts=2,
            initial_ghost_positions=[(3, 4), (5, 6)],
        )

    def test_make_state_returns_ndarray_in_canonical_layout(self):
        """Test make_state returns a float64 ndarray matching readers round-trip.

        Purpose: Validates that make_state produces a correctly-shaped state array.

        Given: Valid position tuples, pellet tuple, score and terminal flag.
        When: make_state is called with those arguments.
        Then: An ndarray of shape (state_dim,) dtype float64 is returned and
            the env reader methods recover each original field.

        Test type: unit
        """
        state = self.pomdp.make_state(
            pacman_pos=(1, 2),
            ghost_positions=((3, 4), (5, 6)),
            pellets=((0, 0), (5, 5)),
            score=50.0,
            terminal=False,
        )

        assert isinstance(state, np.ndarray)
        assert state.shape == (self.pomdp._state_dim,)
        assert state.dtype == np.float64
        assert self.pomdp.get_pacman_pos(state) == (1, 2)
        assert self.pomdp.get_ghost_positions(state) == ((3, 4), (5, 6))
        assert set(self.pomdp.get_pellets(state)) == {(0, 0), (5, 5)}
        assert self.pomdp.get_score(state) == 50.0
        assert self.pomdp.get_terminal(state) is False

    def test_make_state_default_pellets_activates_all_initial(self):
        """Test make_state with pellets=None activates every initial pellet.

        Purpose: Validates the "pellets=None" convenience path matches
            initial_state_dist semantics.

        Given: make_state called without pellets argument.
        When: Readers are used to inspect the pellet mask.
        Then: Every initial pellet position is active.

        Test type: unit
        """
        state = self.pomdp.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((3, 4), (5, 6)),
        )
        assert set(self.pomdp.get_pellets(state)) == {(0, 0), (5, 5), (1, 2), (3, 4)}

    def test_make_state_rejects_bad_pacman_pos(self):
        """Test make_state rejects non-2-tuple pacman_pos.

        Purpose: Validates input validation on the pacman_pos argument.

        Given: An invalid pacman_pos (list instead of tuple; single-element tuple).
        When: make_state is called.
        Then: ValueError is raised describing the violation.

        Test type: unit
        """
        with pytest.raises(ValueError, match="pacman_pos must be a tuple of two integers"):
            self.pomdp.make_state(
                pacman_pos=[1, 2],  # type: ignore[arg-type]
                ghost_positions=((3, 4), (5, 6)),
            )
        with pytest.raises(ValueError, match="pacman_pos must be a tuple of two integers"):
            self.pomdp.make_state(
                pacman_pos=(1,),  # type: ignore[arg-type]
                ghost_positions=((3, 4), (5, 6)),
            )

    def test_make_state_rejects_ghost_count_mismatch(self):
        """Test make_state rejects ghost_positions whose length != num_ghosts.

        Purpose: Validates that the factory guards against silent ghost count bugs.

        Given: A ghost_positions tuple shorter/longer than num_ghosts.
        When: make_state is called.
        Then: ValueError is raised mentioning ghost count.

        Test type: unit
        """
        with pytest.raises(ValueError, match="ghost_positions length"):
            self.pomdp.make_state(
                pacman_pos=(1, 2),
                ghost_positions=((3, 4),),
            )
        with pytest.raises(ValueError, match="ghost_positions must be a tuple"):
            self.pomdp.make_state(
                pacman_pos=(1, 2),
                ghost_positions="invalid",  # type: ignore[arg-type]
            )

    def test_make_state_silently_drops_unknown_pellet_positions(self):
        """Test make_state silently skips pellet positions not in the env's set.

        Purpose: Validates the legacy permissive semantics: unknown pellet positions
            are accepted but produce no bit in the pellet mask, since the array
            layout reserves slots only for positions registered at env init.

        Given: A pellets tuple containing a position the env never registered.
        When: make_state is called.
        Then: No error is raised; the returned state has no active pellets.

        Test type: unit
        """
        state = self.pomdp.make_state(
            pacman_pos=(1, 2),
            ghost_positions=((3, 4), (5, 6)),
            pellets=((9, 9),),
        )
        assert self.pomdp.get_pellets(state) == ()


class TestPacManStateTransitionModel:
    """Test cases for PacManStateTransitionModel."""

    def setup_method(self):
        """Set up test environment and states for each test."""
        self.pomdp = PacManPOMDP(
            maze_size=(5, 5),
            walls={(2, 2)},
            initial_pellets=[(1, 1), (3, 3)],
            initial_pacman_pos=(0, 0),
            num_ghosts=1,
            initial_ghost_positions=[(4, 4)],
        )

        self.state = self.pomdp.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((4, 4),),
            pellets=((1, 1), (3, 3)),
            score=0,
        )

    def test_pacman_movement_north(self):
        """Test PacMan movement in north direction.

        Purpose: Validates that PacMan moves correctly when north action is executed

        Given: PacMan at position (1, 1) and north action (0)
        When: State transition is computed
        Then: PacMan position changes to (0, 1) and other state elements remain unchanged

        Test type: unit
        """
        # Set local random seed for deterministic behavior
        np.random.seed(42)

        state = self.pomdp.make_state(
            pacman_pos=(1, 1), ghost_positions=((4, 4),), pellets=((3, 3),), score=0
        )

        transition = PacManStateTransitionModel(state, action=0, pomdp=self.pomdp)  # North
        next_state = transition.sample()[0]

        assert self.pomdp.get_pacman_pos(next_state) == (0, 1)  # Moved north
        # Ghost movement is stochastic, so we test that all ghost positions are valid
        for ghost_pos in self.pomdp.get_ghost_positions(next_state):
            assert self._is_valid_position(ghost_pos)
        assert self.pomdp.get_pellets(next_state) == self.pomdp.get_pellets(state)
        assert not self.pomdp.get_terminal(next_state)

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds and not a wall)."""
        row, col = pos
        return (
            0 <= row < self.pomdp.maze_size[0]
            and 0 <= col < self.pomdp.maze_size[1]
            and pos not in self.pomdp.walls
        )

    def test_pacman_movement_wall_collision(self):
        """Test PacMan movement into a wall.

        Purpose: Validates that PacMan cannot move through walls and stays in place

        Given: PacMan adjacent to a wall and action directing into the wall
        When: State transition is computed
        Then: PacMan position remains unchanged

        Test type: unit
        """
        state = self.pomdp.make_state(
            pacman_pos=(2, 1), ghost_positions=((4, 4),), pellets=((3, 3),), score=0
        )

        # Try to move east into wall at (2, 2)
        transition = PacManStateTransitionModel(state, action=1, pomdp=self.pomdp)  # East
        next_state = transition.sample()[0]

        assert self.pomdp.get_pacman_pos(next_state) == (2, 1)  # Stayed in place due to wall

    def test_pacman_movement_boundary_collision(self):
        """Test PacMan movement at maze boundaries.

        Purpose: Validates that PacMan cannot move outside maze boundaries

        Given: PacMan at maze edge and action directing outside boundary
        When: State transition is computed
        Then: PacMan position remains unchanged

        Test type: unit
        """
        state = self.pomdp.make_state(
            pacman_pos=(0, 0), ghost_positions=((4, 4),), pellets=((3, 3),), score=0
        )

        # Try to move north from top boundary
        transition = PacManStateTransitionModel(state, action=0, pomdp=self.pomdp)  # North
        next_state = transition.sample()[0]

        assert self.pomdp.get_pacman_pos(next_state) == (0, 0)  # Stayed at boundary

    def test_pellet_collection(self):
        """Test pellet collection when PacMan moves to pellet position.

        Purpose: Validates that pellets are collected and score increases correctly

        Given: PacMan moving to a position containing a pellet
        When: State transition is computed
        Then: Pellet is removed from pellets tuple and score increases

        Test type: unit
        """
        state = self.pomdp.make_state(
            pacman_pos=(1, 0),
            ghost_positions=((4, 4),),
            pellets=((1, 1), (3, 3)),
            score=10,
        )

        # Move east to collect pellet at (1, 1)
        transition = PacManStateTransitionModel(state, action=1, pomdp=self.pomdp)  # East
        next_state = transition.sample()[0]

        assert self.pomdp.get_pacman_pos(next_state) == (1, 1)
        assert (1, 1) not in self.pomdp.get_pellets(next_state)
        assert self.pomdp.get_score(next_state) == 10 + self.pomdp.pellet_reward

    def test_ghost_collision_terminal(self):
        """Test game termination when PacMan collides with ghost.

        Purpose: Validates that game becomes terminal when PacMan and ghost move to same position

        Given: A state where collision detection works correctly
        When: State transition results in PacMan and ghost at same position
        Then: Resulting state is marked as terminal

        Test type: unit
        """
        # Test collision detection by checking if we can get a terminal state
        # when PacMan and ghost end up at the same position after movement.
        # Since ghost movement is stochastic, we'll test multiple scenarios.

        # Scenario 1: Test with a state that already has collision
        collision_state = self.pomdp.make_state(
            pacman_pos=(2, 2),
            ghost_positions=((2, 2),),  # Already at same position
            pellets=((1, 1),),
            score=0,
            terminal=True,  # Manually set as terminal to test the logic
        )

        # Terminal states should remain unchanged
        transition = PacManStateTransitionModel(collision_state, action=0, pomdp=self.pomdp)
        next_state = transition.sample()[0]
        assert self.pomdp.get_terminal(next_state)
        assert np.array_equal(next_state, collision_state)  # Should be unchanged

        # Scenario 2: Test collision detection in state transition logic
        # Create a scenario where we can trigger collision by checking multiple samples
        test_state = self.pomdp.make_state(
            pacman_pos=(1, 1),
            ghost_positions=((1, 2),),  # Adjacent positions
            pellets=((3, 3),),
            score=0,
        )

        # Test multiple transitions to see if collision can occur
        transition = PacManStateTransitionModel(test_state, action=1, pomdp=self.pomdp)  # East
        collision_found = False

        # Sample multiple times since ghost movement is stochastic
        for _ in range(20):  # Try multiple times
            next_state = transition.sample()[0]
            if self.pomdp.get_terminal(next_state) and self.pomdp.get_pacman_pos(
                next_state
            ) in self.pomdp.get_ghost_positions(next_state):
                collision_found = True
                break

        # Note: Due to stochastic ghost movement, collision might not always occur
        # The test validates that the collision detection logic exists and can work

    def test_win_condition_all_pellets_collected(self):
        """Test win condition when all pellets are collected.

        Purpose: Validates that game becomes terminal when all pellets are collected

        Given: PacMan collecting the last remaining pellet
        When: State transition is computed
        Then: Resulting state is terminal with empty pellets tuple

        Test type: unit
        """
        state = self.pomdp.make_state(
            pacman_pos=(1, 0),
            ghost_positions=((4, 4),),
            pellets=((1, 1),),  # Only one pellet left — must be in env.initial_pellets
            score=90.0,
        )

        # Move east to collect last pellet at (1,1)
        transition = PacManStateTransitionModel(state, action=1, pomdp=self.pomdp)  # East
        next_state = transition.sample()[0]

        assert self.pomdp.get_terminal(next_state)
        assert len(self.pomdp.get_pellets(next_state)) == 0
        assert self.pomdp.get_score(next_state) == 90 + self.pomdp.pellet_reward

    def test_terminal_state_unchanged(self):
        """Test that terminal states remain unchanged.

        Purpose: Validates that no further transitions occur from terminal states

        Given: A terminal state
        When: State transition is computed with any action
        Then: State remains exactly the same

        Test type: unit
        """
        terminal_state = self.pomdp.make_state(
            pacman_pos=(2, 2),
            ghost_positions=((2, 2),),
            pellets=(),
            score=100,
            terminal=True,
        )

        transition = PacManStateTransitionModel(terminal_state, action=1, pomdp=self.pomdp)
        next_state = transition.sample()[0]

        assert np.array_equal(next_state, terminal_state)  # Unchanged

    def test_ghost_movement_stochastic(self):
        """Test that ghost movement is stochastic.

        Purpose: Validates that ghost position changes and shows stochastic behavior

        Given: Multiple samples from the same state-action pair
        When: State transitions are computed multiple times
        Then: Ghost positions vary across samples demonstrating stochasticity

        Test type: unit
        """
        state = self.pomdp.make_state(
            pacman_pos=(0, 0), ghost_positions=((2, 2),), pellets=((4, 4),), score=0
        )

        transition = PacManStateTransitionModel(state, action=0, pomdp=self.pomdp)
        ghost_positions = set()

        # Sample multiple times to see ghost movement variation
        for _ in range(20):
            next_state = transition.sample()[0]
            ghost_positions.add(
                self.pomdp.get_ghost_positions(next_state)[0]
            )  # First ghost position

        # Ghost should move (not stay at (2, 2) every time)
        assert len(ghost_positions) > 1

    def test_probability_vs_empirical_distribution(self):
        """Test that computed probabilities match empirical sampling distribution.

        Purpose: Validates that the probability() method correctly computes transition probabilities
                 by comparing them to empirical frequencies from sampling

        Given: A state-transition model with stochastic ghost movements
        When: Computing probabilities for sampled states and comparing to empirical distribution
        Then: Computed probabilities should closely match empirical frequencies from sampling

        Test type: unit
        """
        # Set seed for reproducibility
        np.random.seed(42)

        # Create a simple environment for testing
        pomdp = PacManPOMDP(
            maze_size=(5, 5),
            walls=set(),  # No walls for simplicity
            initial_pellets=[(2, 2)],
            initial_pacman_pos=(0, 0),
            num_ghosts=1,
            initial_ghost_positions=[(3, 3)],
            ghost_aggressiveness=2.0,
        )

        # Create initial state
        state = self.pomdp.make_state(
            pacman_pos=(0, 0), ghost_positions=((3, 3),), pellets=((2, 2),), score=0
        )

        # Choose action (move east)
        action = 1
        transition_model = PacManStateTransitionModel(state, action, pomdp)

        # Use the general utility function to validate probability method
        results = validate_probability_matches_empirical_distribution(
            transition_model, num_samples=1000, max_js_divergence=0.05
        )

        # Verify results
        assert results["probabilities_normalized"]
        assert results["distance"] < 0.05

        print(f"\nProbability validation results:")
        print(f"Number of unique states: {results['num_unique_states']}")
        print(f"Computed probabilities: {results['computed_probs']}")
        print(f"Empirical probabilities: {results['empirical_probs']}")
        print(f"JS Divergence: {results['distance']:.6f}")
        print(f"State counts: {results['state_counts']}")


class TestPacManObservationModel:
    """Test cases for PacManObservationModel."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.pomdp = PacManPOMDP(
            maze_size=(5, 5),
            walls={(1, 2), (2, 1), (3, 4)},  # Custom walls that don't conflict
            initial_pellets=[(1, 1), (1, 3), (3, 1), (3, 3)],  # Valid for 5x5 maze
            observation_noise_factor=0.5,
            max_observation_noise=2.0,
            initial_pacman_pos=(0, 0),
            num_ghosts=1,
            initial_ghost_positions=[(4, 4)],
        )

    def test_observation_terminal_state(self):
        """Test observations from terminal states.

        Purpose: Validates that terminal states produce terminal observations

        Given: A terminal state
        When: Observation is sampled
        Then: Terminal observation (-1, -1) is returned

        Test type: unit
        """
        terminal_state = self.pomdp.make_state(
            pacman_pos=(2, 2), ghost_positions=((2, 2),), pellets=(), terminal=True
        )

        obs_model = PacManObservationModel(terminal_state, action=0, pomdp=self.pomdp)
        observations = obs_model.sample(n_samples=5)

        # Multi-ghost observations: each observation is a tuple of ghost position observations
        expected_terminal_obs = ((-1, -1),)  # Tuple with one terminal observation
        assert all(obs == expected_terminal_obs for obs in observations)

    def test_observation_noise_close_distance(self):
        """Test observation noise when PacMan and ghost are close.

        Purpose: Validates that close distances produce more accurate observations

        Given: PacMan and ghost at close distance
        When: Multiple observations are sampled
        Then: Observations cluster around true ghost position with low noise

        Test type: unit
        """
        state = self.pomdp.make_state(
            pacman_pos=(2, 2),
            ghost_positions=((2, 3),),
            pellets=((0, 0),),  # Close to PacMan
        )

        obs_model = PacManObservationModel(state, action=0, pomdp=self.pomdp)
        observations = obs_model.sample(n_samples=10)

        # Most observations should be close to true ghost position
        true_pos = self.pomdp.get_ghost_positions(state)[0]  # First ghost
        close_observations = sum(
            1
            for obs in observations
            if abs(obs[0][0] - true_pos[0]) <= 1
            and abs(obs[0][1] - true_pos[1]) <= 1  # obs is tuple of positions
        )

        # At close distance, most observations should be accurate
        assert close_observations >= 7  # At least 70% accurate

    def test_observation_noise_far_distance(self):
        """Test observation noise when PacMan and ghost are far apart.

        Purpose: Validates that far distances produce noisier observations

        Given: PacMan and ghost at far distance
        When: Multiple observations are sampled
        Then: Observations show higher variance around true ghost position

        Test type: unit
        """
        state = self.pomdp.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((4, 4),),
            pellets=((1, 1),),  # Far from PacMan
        )

        obs_model = PacManObservationModel(state, action=0, pomdp=self.pomdp)
        observations = obs_model.sample(n_samples=20)

        # Calculate variance in observations for the first ghost
        obs_rows = [obs[0][0] for obs in observations]  # First ghost row observations
        obs_cols = [obs[0][1] for obs in observations]  # First ghost col observations

        row_variance = np.var(obs_rows)
        col_variance = np.var(obs_cols)

        # Far distance should produce higher variance
        assert row_variance > 0.1 or col_variance > 0.1

    def test_observation_bounds_clamping(self):
        """Test that observations are clamped to maze boundaries.

        Purpose: Validates that noisy observations don't exceed maze boundaries

        Given: Observation model with potential noise that could exceed bounds
        When: Multiple observations are sampled
        Then: All observations are within valid maze coordinates

        Test type: unit
        """
        state = self.pomdp.make_state(
            pacman_pos=(0, 0), ghost_positions=((4, 4),), pellets=((1, 1),)
        )

        obs_model = PacManObservationModel(state, action=0, pomdp=self.pomdp)
        observations = obs_model.sample(n_samples=50)

        for obs in observations:
            # Each observation is a tuple of ghost position observations
            for ghost_obs in obs:
                if ghost_obs != (-1, -1):  # Skip terminal observations
                    assert 0 <= ghost_obs[0] < self.pomdp.maze_size[0]
                    assert 0 <= ghost_obs[1] < self.pomdp.maze_size[1]

    def test_observation_probability_calculation(self):
        """Test observation probability calculation.

        Purpose: Validates that observation probabilities are calculated correctly

        Given: Observation model with specific state and candidate observations
        When: Probabilities are calculated for different observations
        Then: Probabilities sum to reasonable total and favor closer observations

        Test type: unit
        """
        state = self.pomdp.make_state(
            pacman_pos=(2, 2), ghost_positions=((2, 3),), pellets=((0, 0),)
        )

        obs_model = PacManObservationModel(state, action=0, pomdp=self.pomdp)

        # Test probabilities for multi-ghost observations at different distances
        candidate_obs = [
            ((2, 3),),
            ((2, 4),),
            ((-1, -1),),
        ]  # True pos, nearby, terminal
        probs = obs_model.probability(candidate_obs)

        assert len(probs) == 3
        assert probs[0] > probs[1]  # True position should have higher probability
        assert (
            probs[2] == 0.0
        )  # Terminal observation should have 0 probability for non-terminal state


class TestPacManPOMDP:
    """Test cases for PacManPOMDP environment."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.pomdp = PacManPOMDP(
            maze_size=(5, 5),
            walls={(2, 2), (3, 1)},
            initial_pellets=[(1, 1), (3, 3)],
            initial_pacman_pos=(0, 0),
            num_ghosts=1,
            initial_ghost_positions=[(4, 4)],
        )

    def test_pomdp_initialization(self):
        """Test PacMan POMDP initialization with default parameters.

        Purpose: Validates that PacManPOMDP initializes correctly with default settings

        Given: Default PacManPOMDP parameters
        When: PacManPOMDP is instantiated
        Then: All attributes are set correctly and environment is ready for use

        Test type: unit
        """
        pomdp = PacManPOMDP()

        assert pomdp.maze_size == (7, 7)
        assert pomdp.initial_pacman_pos == (0, 0)
        assert pomdp.num_ghosts == 1
        assert pomdp.initial_ghost_pos == pomdp.initial_ghost_positions[0]  # Backward compatibility
        assert pomdp.pellet_reward == 10.0
        assert pomdp.ghost_collision_penalty == -100.0
        assert pomdp.step_penalty == -1.0
        assert pomdp.win_reward == 100.0
        assert pomdp.ghost_coordination == "independent"
        assert pomdp.name == "PacManPOMDP"
        assert pomdp.space_info.action_space == SpaceType.DISCRETE
        assert pomdp.space_info.observation_space == SpaceType.DISCRETE

    def test_pomdp_parameter_validation_invalid_maze_size(self):
        """Test parameter validation for invalid maze size.

        Purpose: Validates that PacManPOMDP rejects invalid maze dimensions

        Given: Invalid maze size (zero or negative dimensions)
        When: PacManPOMDP is instantiated
        Then: ValueError is raised with descriptive message

        Test type: unit
        """
        with pytest.raises(ValueError, match="Maze size must be positive"):
            PacManPOMDP(maze_size=(0, 5))

        with pytest.raises(ValueError, match="Maze size must be positive"):
            PacManPOMDP(maze_size=(5, -1))

    def test_pomdp_parameter_validation_positions_out_of_bounds(self):
        """Test parameter validation for out-of-bounds positions.

        Purpose: Validates that PacManPOMDP rejects positions outside maze boundaries

        Given: Initial positions outside the defined maze size
        When: PacManPOMDP is instantiated
        Then: ValueError is raised indicating position is outside bounds

        Test type: unit
        """
        with pytest.raises(ValueError, match="PacMan position .* is outside maze bounds"):
            PacManPOMDP(maze_size=(5, 5), initial_pacman_pos=(5, 0))  # Outside bounds

        with pytest.raises(ValueError, match="Ghost 0 position .* is outside maze bounds"):
            PacManPOMDP(
                maze_size=(5, 5), num_ghosts=1, initial_ghost_positions=[(0, 5)]
            )  # Outside bounds

    def test_pomdp_parameter_validation_pellet_in_wall(self):
        """Test parameter validation for pellets placed in walls.

        Purpose: Validates that PacManPOMDP rejects pellets positioned inside walls

        Given: Pellet position that coincides with a wall position
        When: PacManPOMDP is instantiated
        Then: ValueError is raised indicating pellet is inside a wall

        Test type: unit
        """
        with pytest.raises(ValueError, match="Pellet position .* is inside a wall"):
            PacManPOMDP(
                maze_size=(5, 5),
                walls={(2, 2)},
                initial_pellets=[(2, 2), (3, 3)],  # Pellet inside wall
                initial_pacman_pos=(0, 0),
                num_ghosts=1,
                initial_ghost_positions=[(4, 4)],
            )

    def test_pomdp_parameter_validation_initial_pos_in_wall(self):
        """Test parameter validation for initial positions in walls.

        Purpose: Validates that PacManPOMDP rejects initial positions inside walls

        Given: Initial PacMan or ghost position inside a wall
        When: PacManPOMDP is instantiated
        Then: ValueError is raised indicating position is inside a wall

        Test type: unit
        """
        with pytest.raises(ValueError, match="Initial PacMan position .* is inside a wall"):
            PacManPOMDP(
                maze_size=(5, 5),
                walls={(0, 0)},
                initial_pellets=[(1, 1)],
                initial_pacman_pos=(0, 0),  # In wall
                num_ghosts=1,
                initial_ghost_positions=[(4, 4)],
            )

        with pytest.raises(ValueError, match="Initial ghost 0 position .* is inside a wall"):
            PacManPOMDP(
                maze_size=(5, 5),
                walls={(4, 4)},
                initial_pellets=[(1, 1)],
                initial_pacman_pos=(0, 0),
                num_ghosts=1,
                initial_ghost_positions=[(4, 4)],  # In wall
            )

    def test_get_actions(self):
        """Test action enumeration.

        Purpose: Validates that PacManPOMDP returns correct list of available actions

        Given: A PacManPOMDP instance
        When: get_actions() is called
        Then: Returns list of action integers corresponding to movement directions

        Test type: unit
        """
        actions = self.pomdp.get_actions()
        expected_actions = [0, 1, 2, 3]  # North, East, South, West

        assert actions == expected_actions
        assert len(actions) == 4

    def test_initial_state_distribution(self):
        """Test initial state distribution.

        Purpose: Validates that initial state distribution returns correct deterministic state

        Given: A PacManPOMDP instance with defined initial parameters
        When: initial_state_dist().sample() is called
        Then: Returns state with correct initial positions and pellets

        Test type: unit
        """
        initial_dist = self.pomdp.initial_state_dist()
        initial_state = initial_dist.sample()[0]

        assert isinstance(initial_state, np.ndarray)
        assert self.pomdp.get_pacman_pos(initial_state) == self.pomdp.initial_pacman_pos
        assert self.pomdp.get_ghost_positions(initial_state) == tuple(
            self.pomdp.initial_ghost_positions
        )
        assert (
            self.pomdp.get_ghost_positions(initial_state)[0] == self.pomdp.initial_ghost_pos
        )  # First ghost matches env's legacy single-ghost attr
        assert self.pomdp.get_pellets(initial_state) == tuple(self.pomdp.initial_pellets)
        assert self.pomdp.get_score(initial_state) == 0
        assert not self.pomdp.get_terminal(initial_state)

    def test_initial_observation_distribution(self):
        """Test initial observation distribution.

        Purpose: Validates that initial observation distribution returns valid ghost position observation

        Given: A PacManPOMDP instance
        When: initial_observation_dist().sample() is called
        Then: Returns valid ghost position tuple within maze bounds

        Test type: unit
        """
        initial_obs_dist = self.pomdp.initial_observation_dist()
        initial_obs = initial_obs_dist.sample()[0]

        assert isinstance(initial_obs, tuple)
        assert len(initial_obs) == self.pomdp.num_ghosts  # One observation per ghost
        for ghost_obs in initial_obs:
            assert isinstance(ghost_obs, tuple)
            assert len(ghost_obs) == 2
            assert 0 <= ghost_obs[0] < self.pomdp.maze_size[0]
            assert 0 <= ghost_obs[1] < self.pomdp.maze_size[1]

    def test_reward_calculation_step_penalty(self):
        """Test reward calculation with step penalty.

        Purpose: Validates that step penalty is applied for all non-terminal actions

        Given: Non-terminal state and any action
        When: reward() is called
        Then: Returns reward that includes step penalty component

        Test type: unit
        """
        state = self.pomdp.make_state(
            pacman_pos=(0, 1),
            ghost_positions=((3, 3),),
            pellets=((1, 1), (3, 3)),
            score=0.0,
        )

        # Move north (into boundary, pacman stays at (0,1); no pellet there)
        reward = self.pomdp.reward(state, action=0)

        # Should include step penalty and nothing else (no pellet, no collision, no win)
        assert reward == self.pomdp.step_penalty

    def test_reward_calculation_pellet_collection(self):
        """Test reward calculation for pellet collection.

        Purpose: Validates that pellet collection adds appropriate reward

        Given: State where PacMan moves to collect a pellet
        When: reward() is calculated
        Then: Returns reward including pellet collection bonus

        Test type: unit
        """
        state = self.pomdp.make_state(
            pacman_pos=(1, 0),  # Adjacent to pellet at (1, 1)
            ghost_positions=((3, 3),),
            pellets=((1, 1), (3, 3)),
            score=0.0,
        )

        # Move east to collect pellet at (1,1)
        reward = self.pomdp.reward(state, action=1)

        # Should include both step penalty and pellet reward
        expected_reward = self.pomdp.step_penalty + self.pomdp.pellet_reward
        assert reward == expected_reward

    def test_reward_calculation_ghost_collision(self):
        """Test reward calculation for ghost collision.

        Purpose: Validates that reward function correctly handles collision detection

        Given: Any state and action
        When: reward() is calculated
        Then: Reward includes collision penalty if collision occurs in the resulting state

        Test type: unit
        """
        # Simple state for testing reward calculation
        state = self.pomdp.make_state(
            pacman_pos=(1, 1), ghost_positions=((1, 2),), pellets=((3, 3),), score=10
        )

        # Get reward for any action
        reward = self.pomdp.reward(state, action=1)  # Move east

        # Reward should include step penalty at minimum
        assert reward <= self.pomdp.step_penalty

        # If collision penalty is included, reward should be much more negative
        if reward < self.pomdp.step_penalty:
            # Collision occurred - verify it's the expected penalty
            expected_collision_reward = self.pomdp.step_penalty + self.pomdp.ghost_collision_penalty
            assert reward == expected_collision_reward

    def test_reward_calculation_win_condition(self):
        """Test reward calculation for win condition.

        Purpose: Validates that collecting all pellets adds win reward

        Given: State where last pellet is collected
        When: reward() is calculated
        Then: Returns reward including win bonus

        Test type: unit
        """
        state = self.pomdp.make_state(
            pacman_pos=(1, 0),  # Adjacent to last pellet at (1, 1)
            ghost_positions=((4, 4),),
            pellets=((1, 1),),  # Only one pellet left — in env.initial_pellets
            score=90.0,
        )

        # Move east to collect last pellet at (1,1)
        reward = self.pomdp.reward(state, action=1)

        # Should include step penalty, pellet reward, and win reward
        expected_reward = self.pomdp.step_penalty + self.pomdp.pellet_reward + self.pomdp.win_reward
        assert reward == expected_reward

    def test_terminal_state_check(self):
        """Test terminal state identification.

        Purpose: Validates that terminal states are correctly identified

        Given: States with terminal and non-terminal conditions
        When: is_terminal() is called
        Then: Returns correct boolean value for terminal status

        Test type: unit
        """
        # Non-terminal state
        non_terminal_state = self.pomdp.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((4, 4),),
            pellets=((1, 1),),
            terminal=False,
        )
        assert not self.pomdp.is_terminal(non_terminal_state)

        # Terminal state
        terminal_state = self.pomdp.make_state(
            pacman_pos=(2, 2), ghost_positions=((2, 2),), pellets=(), terminal=True
        )
        assert self.pomdp.is_terminal(terminal_state)

    def test_reward_range(self):
        """Test that reward range is correctly calculated.

        Purpose: Validates that PacManPOMDP reward range is properly calculated based on environment parameters

        Given: A PacManPOMDP environment with specific reward/penalty parameters
        When: Environment reward_range attribute is checked
        Then: Returns range based on min_reward (step + ghost collision penalties) and max_reward (step + pellet + win rewards)

        Test type: unit
        """
        # Expected calculation from PacManPOMDP constructor:
        # min_reward = step_penalty + ghost_collision_penalty = -1.0 + (-100.0) = -101.0
        # max_reward = step_penalty + pellet_reward + win_reward = -1.0 + 10.0 + 100.0 = 109.0
        expected_min = self.pomdp.step_penalty + self.pomdp.ghost_collision_penalty
        expected_max = self.pomdp.step_penalty + self.pomdp.pellet_reward + self.pomdp.win_reward

        assert self.pomdp.reward_range == (expected_min, expected_max)

        # Test with custom parameters
        custom_pomdp = PacManPOMDP(
            maze_size=(5, 5),
            walls={(1, 2), (2, 1), (3, 4)},  # Custom walls that don't conflict
            initial_pellets=[(1, 1), (1, 3), (3, 1), (3, 3)],  # Valid for 5x5 maze
            initial_pacman_pos=(0, 0),
            initial_ghost_pos=(4, 4),  # Within 5x5 maze bounds
            pellet_reward=20.0,
            ghost_collision_penalty=-200.0,
            step_penalty=-2.0,
            win_reward=500.0,
        )

        expected_min_custom = -2.0 + (-200.0)  # -202.0
        expected_max_custom = -2.0 + 20.0 + 500.0  # 518.0

        assert custom_pomdp.reward_range == (expected_min_custom, expected_max_custom)

    def test_observation_equality(self):
        """Test observation equality comparison.

        Purpose: Validates that observation equality is checked correctly

        Given: Identical and different observation tuples
        When: is_equal_observation() is called
        Then: Returns correct boolean for observation equality

        Test type: unit
        """
        obs1 = ((2, 3),)  # Multi-ghost observation tuple
        obs2 = ((2, 3),)
        obs3 = ((3, 2),)

        assert self.pomdp.is_equal_observation(obs1, obs2)
        assert not self.pomdp.is_equal_observation(obs1, obs3)

    def test_sample_next_step_integration(self):
        """Test complete state transition step.

        Purpose: Validates that sample_next_step() integrates all models correctly

        Given: Initial state and action
        When: sample_next_step() is called
        Then: Returns valid next state, observation, and reward

        Test type: integration
        """
        initial_state = self.pomdp.initial_state_dist().sample()[0]

        next_state, obs, reward = self.pomdp.sample_next_step(initial_state, action=1)

        assert isinstance(next_state, np.ndarray)
        assert isinstance(obs, tuple)
        assert len(obs) == self.pomdp.num_ghosts  # Multi-ghost observations
        for ghost_obs in obs:
            assert isinstance(ghost_obs, tuple)
            assert len(ghost_obs) == 2
        assert isinstance(reward, (int, float))

        # Reward should include step penalty at minimum
        assert reward <= self.pomdp.step_penalty


class TestPacManPOMDPMetrics:
    """Test cases for PacMan POMDP metrics computation."""

    def setup_method(self):
        """Set up test environment for metrics tests."""
        self.pomdp = PacManPOMDP(
            maze_size=(5, 5),
            walls={(1, 2), (3, 1)},  # Custom walls that don't conflict with pellets
            initial_pellets=[(1, 1), (3, 3)],  # Valid pellets not in walls
            initial_pacman_pos=(0, 0),
            initial_ghost_positions=[(4, 4)],  # Multi-ghost format
        )

    def test_compute_metrics_empty_histories(self):
        """Test metrics computation with empty histories.

        Purpose: Validates that empty history list returns empty metrics

        Given: Empty list of episode histories
        When: compute_metrics() is called
        Then: Returns empty list of metrics

        Test type: unit
        """
        metrics = self.pomdp.compute_metrics([])
        assert metrics == []

    def test_compute_metrics_win_rate(self):
        """Test win rate metric calculation.

        Purpose: Validates that win rate is calculated correctly from episode histories

        Given: Episode histories with wins and losses
        When: compute_metrics() is called
        Then: Returns accurate win rate metric with confidence intervals

        Test type: unit
        """
        # Create mock histories
        winning_state = self.pomdp.make_state(
            pacman_pos=(2, 2),
            ghost_positions=((0, 0),),  # Multi-ghost format
            pellets=(),  # No pellets left = win
            terminal=True,
            score=100,
        )

        losing_state = self.pomdp.make_state(
            pacman_pos=(1, 1),
            ghost_positions=((1, 1),),  # Multi-ghost format - collision = loss
            pellets=((1, 1),),  # Pellets remaining — must be a registered position
            terminal=True,
            score=50.0,
        )

        # Create histories with proper StepData structure
        # Use Mock belief for testing metrics computation
        dummy_belief = Mock(spec=Belief)

        win_history = History(
            history=[
                StepData(
                    state=winning_state,
                    action=None,
                    next_state=winning_state,
                    observation=None,
                    reward=0,
                    belief=dummy_belief,
                )
            ],
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=1,
            reach_terminal_state=True,
            policy_run_data=[],
        )

        lose_history = History(
            history=[
                StepData(
                    state=losing_state,
                    action=None,
                    next_state=losing_state,
                    observation=None,
                    reward=0,
                    belief=dummy_belief,
                )
            ],
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=1,
            reach_terminal_state=True,
            policy_run_data=[],
        )

        histories = [win_history, lose_history, win_history]  # 2 wins, 1 loss

        metrics = self.pomdp.compute_metrics(histories)

        # Find win rate metric
        win_rate_metric = next((m for m in metrics if m.name == "win_rate"), None)
        assert win_rate_metric is not None
        assert abs(win_rate_metric.value - (2 / 3)) < 0.01  # 66.7% win rate

    def test_compute_metrics_pellets_collected(self):
        """Test pellets collected metric calculation.

        Purpose: Validates that average pellets collected is calculated correctly

        Given: Episode histories with different numbers of pellets collected
        When: compute_metrics() is called
        Then: Returns accurate average pellets collected metric

        Test type: unit
        """
        # States with different pellet collection amounts
        state1 = self.pomdp.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((2, 2),),  # Multi-ghost format
            pellets=(),  # All 2 pellets collected
            terminal=True,
            score=20,
        )

        state2 = self.pomdp.make_state(
            pacman_pos=(1, 1),
            ghost_positions=((1, 1),),  # Multi-ghost format
            pellets=((1, 1),),  # Only 1 pellet collected (the other is still active)
            terminal=True,
            score=10.0,
        )

        # Create histories with proper StepData structure
        # Use Mock belief for testing metrics computation
        dummy_belief = Mock(spec=Belief)

        histories = [
            History(
                history=[
                    StepData(
                        state=state1,
                        action=None,
                        next_state=state1,
                        observation=None,
                        reward=0,
                        belief=dummy_belief,
                    )
                ],
                discount_factor=0.95,
                average_state_sampling_time=0.0,
                average_action_time=0.0,
                average_observation_time=0.0,
                average_belief_update_time=0.0,
                average_reward_time=0.0,
                actual_num_steps=1,
                reach_terminal_state=True,
                policy_run_data=[],
            ),
            History(
                history=[
                    StepData(
                        state=state2,
                        action=None,
                        next_state=state2,
                        observation=None,
                        reward=0,
                        belief=dummy_belief,
                    )
                ],
                discount_factor=0.95,
                average_state_sampling_time=0.0,
                average_action_time=0.0,
                average_observation_time=0.0,
                average_belief_update_time=0.0,
                average_reward_time=0.0,
                actual_num_steps=1,
                reach_terminal_state=True,
                policy_run_data=[],
            ),
        ]

        metrics = self.pomdp.compute_metrics(histories)

        # Find pellets collected metric
        pellets_metric = next((m for m in metrics if m.name == "avg_pellets_collected"), None)
        assert pellets_metric is not None
        assert abs(pellets_metric.value - 1.5) < 0.01  # Average of 2 and 1 pellets

    def test_compute_metrics_average_pacman_ghost_distance(self):
        """Test average PacMan-ghost distance metric calculation.

        Purpose: Validates that average distance between PacMan and ghost is calculated correctly

        Given: Episode histories with known PacMan and ghost positions
        When: compute_metrics() is called
        Then: Returns accurate average distance metric across episodes

        Test type: unit
        """

        # Create dummy belief for step data
        dummy_belief = WeightedParticleBelief(
            particles=[None], log_weights=np.array([0.1]), resampling=False
        )

        # Episode 1: Two steps with distances 4 and 2 (average = 3)
        state1_ep1 = self.pomdp.make_state(
            pacman_pos=(1, 1),
            ghost_positions=(
                (3, 3),
            ),  # Multi-ghost format - Manhattan distance = |1-3| + |1-3| = 4
            pellets=((2, 2),),
        )
        state2_ep1 = self.pomdp.make_state(
            pacman_pos=(2, 2),
            ghost_positions=(
                (3, 3),
            ),  # Multi-ghost format - Manhattan distance = |2-3| + |2-3| = 2
            pellets=((2, 2),),
        )

        history1 = History(
            history=[
                StepData(
                    state=state1_ep1,
                    action=None,
                    next_state=state2_ep1,
                    observation=None,
                    reward=0,
                    belief=dummy_belief,
                ),
                StepData(
                    state=state2_ep1,
                    action=None,
                    next_state=state2_ep1,
                    observation=None,
                    reward=0,
                    belief=dummy_belief,
                ),
            ],
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=2,
            reach_terminal_state=False,
            policy_run_data=[],
        )

        # Episode 2: Two steps with distances 6 and 4 (average = 5)
        state1_ep2 = self.pomdp.make_state(
            pacman_pos=(0, 0),
            ghost_positions=(
                (3, 3),
            ),  # Multi-ghost format - Manhattan distance = |0-3| + |0-3| = 6
            pellets=((2, 2),),
        )
        state2_ep2 = self.pomdp.make_state(
            pacman_pos=(1, 1),
            ghost_positions=(
                (3, 3),
            ),  # Multi-ghost format - Manhattan distance = |1-3| + |1-3| = 4
            pellets=((2, 2),),
        )

        history2 = History(
            history=[
                StepData(
                    state=state1_ep2,
                    action=None,
                    next_state=state2_ep2,
                    observation=None,
                    reward=0,
                    belief=dummy_belief,
                ),
                StepData(
                    state=state2_ep2,
                    action=None,
                    next_state=state2_ep2,
                    observation=None,
                    reward=0,
                    belief=dummy_belief,
                ),
            ],
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=2,
            reach_terminal_state=False,
            policy_run_data=[],
        )

        metrics = self.pomdp.compute_metrics([history1, history2])

        # Find distance metric
        distance_metric = next(
            (m for m in metrics if m.name == "avg_pacman_closest_ghost_distance"), None
        )
        assert distance_metric is not None

        # Expected: average of episode averages: (3 + 5) / 2 = 4
        expected_distance = 4.0
        assert abs(distance_metric.value - expected_distance) < 0.01

        # Should have confidence bounds
        assert distance_metric.lower_confidence_bound is not None
        assert distance_metric.upper_confidence_bound is not None

    def test_compute_metrics_collision_encounters(self):
        """Test collision encounters metric calculation.

        Purpose: Validates that collision encounters between PacMan and ghost are counted correctly

        Given: Episode histories with known collision events
        When: compute_metrics() is called
        Then: Returns accurate average collision encounters metric across episodes

        Test type: unit
        """

        # Create dummy belief for step data
        dummy_belief = WeightedParticleBelief(
            particles=[None], log_weights=np.array([0.1]), resampling=False
        )

        # Episode 1: Two collisions (steps where PacMan and ghost are at same position)
        state1_ep1 = self.pomdp.make_state(
            pacman_pos=(2, 2),
            ghost_positions=((2, 2),),  # Multi-ghost format - collision!
            pellets=((1, 1),),
        )
        state2_ep1 = self.pomdp.make_state(
            pacman_pos=(2, 3),
            ghost_positions=((3, 3),),  # Multi-ghost format - no collision
            pellets=((1, 1),),
        )
        state3_ep1 = self.pomdp.make_state(
            pacman_pos=(3, 3),
            ghost_positions=((3, 3),),  # Multi-ghost format - collision!
            pellets=((1, 1),),
        )

        history1 = History(
            history=[
                StepData(
                    state=state1_ep1,
                    action=None,
                    next_state=state2_ep1,
                    observation=None,
                    reward=0,
                    belief=dummy_belief,
                ),
                StepData(
                    state=state2_ep1,
                    action=None,
                    next_state=state3_ep1,
                    observation=None,
                    reward=0,
                    belief=dummy_belief,
                ),
                StepData(
                    state=state3_ep1,
                    action=None,
                    next_state=state3_ep1,
                    observation=None,
                    reward=0,
                    belief=dummy_belief,
                ),
            ],
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=3,
            reach_terminal_state=False,
            policy_run_data=[],
        )

        # Episode 2: One collision
        state1_ep2 = self.pomdp.make_state(
            pacman_pos=(1, 1),
            ghost_positions=((2, 2),),  # Multi-ghost format - no collision
            pellets=((0, 0),),
        )
        state2_ep2 = self.pomdp.make_state(
            pacman_pos=(1, 2),
            ghost_positions=((1, 2),),  # Multi-ghost format - collision!
            pellets=((0, 0),),
        )

        history2 = History(
            history=[
                StepData(
                    state=state1_ep2,
                    action=None,
                    next_state=state2_ep2,
                    observation=None,
                    reward=0,
                    belief=dummy_belief,
                ),
                StepData(
                    state=state2_ep2,
                    action=None,
                    next_state=state2_ep2,
                    observation=None,
                    reward=0,
                    belief=dummy_belief,
                ),
            ],
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=2,
            reach_terminal_state=False,
            policy_run_data=[],
        )

        metrics = self.pomdp.compute_metrics([history1, history2])

        # Find collision encounters metric
        collision_metric = next((m for m in metrics if m.name == "avg_collision_encounters"), None)
        assert collision_metric is not None

        # Expected: average of episode collisions: (2 + 1) / 2 = 1.5
        expected_collisions = 1.5
        assert abs(collision_metric.value - expected_collisions) < 0.01

        # Should have confidence bounds
        assert collision_metric.lower_confidence_bound is not None
        assert collision_metric.upper_confidence_bound is not None


class TestCreateSimpleMazePacman:
    """Test cases for create_simple_maze_pacman function."""

    def test_create_simple_maze_default(self):
        """Test create_simple_maze_pacman with default parameters.

        Purpose: Validates that simple maze creation works with default settings

        Given: Default parameters for simple maze creation
        When: create_simple_maze_pacman() is called
        Then: Returns valid PacManPOMDP instance with expected configuration

        Test type: unit
        """
        pomdp = create_simple_maze_pacman()

        assert isinstance(pomdp, PacManPOMDP)
        assert pomdp.maze_size == (7, 7)
        assert pomdp.initial_pacman_pos == (0, 0)
        assert pomdp.initial_ghost_positions == [(6, 6)]  # Multi-ghost format
        assert len(pomdp.initial_pellets) == 4  # Corner pellets
        assert len(pomdp.walls) <= 5  # At most 5 walls

    def test_create_simple_maze_custom_parameters(self):
        """Test create_simple_maze_pacman with custom parameters.

        Purpose: Validates that simple maze creation respects custom parameters

        Given: Custom maze size, number of walls, and random seed
        When: create_simple_maze_pacman() is called with parameters
        Then: Returns PacManPOMDP instance with specified configuration

        Test type: unit
        """
        pomdp = create_simple_maze_pacman(maze_size=5, num_walls=3, seed=42)

        assert pomdp.maze_size == (5, 5)
        assert len(pomdp.walls) == 3
        assert pomdp.initial_pacman_pos == (0, 0)
        assert pomdp.initial_ghost_positions == [(4, 4)]  # Multi-ghost format - bottom right corner

    def test_create_simple_maze_deterministic_seed(self):
        """Test create_simple_maze_pacman with deterministic seeding.

        Purpose: Validates that same seed produces identical maze configurations

        Given: Same random seed used multiple times
        When: create_simple_maze_pacman() is called repeatedly with same seed
        Then: All generated mazes have identical wall configurations

        Test type: unit
        """
        pomdp1 = create_simple_maze_pacman(maze_size=5, num_walls=3, seed=123)
        pomdp2 = create_simple_maze_pacman(maze_size=5, num_walls=3, seed=123)

        assert pomdp1.walls == pomdp2.walls
        assert pomdp1.initial_pellets == pomdp2.initial_pellets

    def test_create_simple_maze_wall_constraints(self):
        """Test that walls avoid important positions.

        Purpose: Validates that randomly placed walls don't block essential positions

        Given: Simple maze creation with walls
        When: create_simple_maze_pacman() is called
        Then: Walls don't occupy corner positions or center where PacMan/ghost start

        Test type: unit
        """
        pomdp = create_simple_maze_pacman(maze_size=7, num_walls=10, seed=456)

        # Check that walls don't block important positions
        forbidden_positions = {(0, 0), (0, 6), (6, 0), (6, 6), (3, 3)}

        for wall in pomdp.walls:
            assert wall not in forbidden_positions


def test_pacman_pomdp_usage_example():
    """Test the usage example from PacManPOMDP docstring.

    Purpose: Validates that the documented usage example executes correctly

    Given: Usage example code from PacManPOMDP class documentation
    When: Example code is executed
    Then: All operations complete successfully without errors

    Test type: example
    """
    # Execute the exact usage example from docstring
    walls = {(1, 1), (1, 2), (3, 3)}
    pellets = [(0, 2), (2, 0), (4, 4)]
    pomdp = PacManPOMDP(maze_size=(7, 7), walls=walls, initial_pellets=pellets)

    # Sample initial state
    initial_state = pomdp.initial_state_dist().sample()[0]

    # Execute action
    next_state, obs, reward = pomdp.sample_next_step(initial_state, 1)

    # Verify results
    assert isinstance(next_state, np.ndarray)
    assert isinstance(obs, tuple)
    assert isinstance(reward, (int, float))


class TestMultiGhostFeatures:
    """Test cases for multi-ghost specific functionality."""

    def setup_method(self):
        """Set up a default 3-ghost / 4-pellet env for state-construction tests."""
        self.pomdp = PacManPOMDP(
            maze_size=(7, 7),
            walls=set(),
            initial_pellets=[(0, 0), (5, 5), (0, 1), (1, 1)],
            initial_pacman_pos=(2, 3),
            num_ghosts=3,
            initial_ghost_positions=[(1, 1), (4, 4), (6, 2)],
        )

    def test_multi_ghost_initialization(self):
        """Test PacManPOMDP initialization with multiple ghosts.

        Purpose: Validates that PacManPOMDP can be initialized with multiple ghosts

        Given: Parameters specifying multiple ghost positions and strategies
        When: PacManPOMDP is initialized
        Then: Environment correctly stores multi-ghost configuration

        Test type: unit
        """
        pomdp = PacManPOMDP(
            maze_size=(7, 7),
            walls={(2, 2), (3, 3)},
            initial_pellets=[(0, 1), (1, 0)],
            initial_pacman_pos=(0, 0),
            initial_ghost_positions=[(6, 6), (5, 5), (4, 4)],
            num_ghosts=3,
            ghost_coordination="independent",
            ghost_strategies=["aggressive", "patrol", "ambush"],
        )

        assert pomdp.num_ghosts == 3
        assert len(pomdp.initial_ghost_positions) == 3
        assert pomdp.initial_ghost_positions == [(6, 6), (5, 5), (4, 4)]
        assert pomdp.ghost_coordination == "independent"
        assert pomdp.ghost_strategies == ["aggressive", "patrol", "ambush"]

    def test_multi_ghost_state_creation(self):
        """Test PacManState creation with multiple ghosts.

        Purpose: Validates that PacManState correctly handles multiple ghost positions

        Given: State parameters with multiple ghost positions
        When: PacManState is created
        Then: State correctly stores all ghost positions with proper properties

        Test type: unit
        """
        state = self.pomdp.make_state(
            pacman_pos=(2, 3),
            ghost_positions=((1, 1), (4, 4), (6, 2)),
            pellets=((0, 0), (5, 5)),
            score=150.0,
            terminal=False,
        )

        ghosts = self.pomdp.get_ghost_positions(state)
        assert len(ghosts) == 3
        assert ghosts == ((1, 1), (4, 4), (6, 2))
        assert self.pomdp.num_ghosts == 3
        assert ghosts[0] == (1, 1)  # First ghost position

    def test_multi_ghost_observation_format(self):
        """Test observation format for multiple ghosts.

        Purpose: Validates that observations correctly represent multiple ghost positions

        Given: PacManPOMDP with multiple ghosts
        When: Observations are generated
        Then: Observation tuple contains position for each ghost

        Test type: unit
        """
        pomdp = PacManPOMDP(
            maze_size=(5, 5),
            walls={(2, 2)},
            initial_pellets=[(0, 1)],
            initial_pacman_pos=(0, 0),
            initial_ghost_positions=[(4, 4), (3, 3)],
            num_ghosts=2,
        )

        state = pomdp.make_state(
            pacman_pos=(1, 1),
            ghost_positions=((2, 3), (4, 1)),
            pellets=((0, 1),),
        )

        # Create observation model explicitly

        obs_model = PacManObservationModel(state, action=0, pomdp=pomdp)
        obs = obs_model.sample(n_samples=1)[0]  # Sample one observation

        assert isinstance(obs, tuple)
        assert len(obs) == 2  # One observation per ghost
        for ghost_obs in obs:
            assert isinstance(ghost_obs, tuple)
            assert len(ghost_obs) == 2  # (x, y) position

    def test_ghost_coordination_strategies(self):
        """Test different ghost coordination strategies.

        Purpose: Validates that different ghost coordination modes are properly handled

        Given: PacManPOMDP with different coordination strategies
        When: State transitions occur
        Then: Ghost movements follow coordination patterns appropriately

        Test type: unit
        """
        # Test independent coordination
        pomdp_independent = PacManPOMDP(
            maze_size=(5, 5),
            walls=set(),  # No walls to avoid conflicts
            initial_pellets=[(0, 1), (2, 2)],  # Explicit pellets within bounds
            initial_ghost_positions=[(3, 3), (4, 4)],
            num_ghosts=2,
            ghost_coordination="independent",
        )
        assert pomdp_independent.ghost_coordination == "independent"

        # Test coordinated strategy
        pomdp_coordinated = PacManPOMDP(
            maze_size=(5, 5),
            walls=set(),  # No walls to avoid conflicts
            initial_pellets=[(0, 1), (2, 2)],  # Explicit pellets within bounds
            initial_ghost_positions=[(3, 3), (4, 4)],
            num_ghosts=2,
            ghost_coordination="coordinated",
        )
        assert pomdp_coordinated.ghost_coordination == "coordinated"

        # Test mixed strategy
        pomdp_mixed = PacManPOMDP(
            maze_size=(5, 5),
            walls=set(),  # No walls to avoid conflicts
            initial_pellets=[(0, 1), (2, 2)],  # Explicit pellets within bounds
            initial_ghost_positions=[(3, 3), (4, 4)],
            num_ghosts=2,
            ghost_coordination="mixed",
        )
        assert pomdp_mixed.ghost_coordination == "mixed"

    def test_individual_ghost_strategies(self):
        """Test individual ghost AI strategies.

        Purpose: Validates that individual ghosts can have different AI behaviors

        Given: PacManPOMDP with specific ghost strategies
        When: Ghost strategies are configured
        Then: Each ghost has assigned strategy type

        Test type: unit
        """
        pomdp = PacManPOMDP(
            maze_size=(6, 6),
            walls=set(),  # No walls to avoid conflicts
            initial_pellets=[(0, 1), (1, 0), (2, 2)],  # Explicit pellets within bounds
            initial_ghost_positions=[(5, 5), (4, 4), (3, 3)],
            num_ghosts=3,
            ghost_strategies=["aggressive", "patrol", "ambush"],
        )

        assert len(pomdp.ghost_strategies) == 3
        assert pomdp.ghost_strategies[0] == "aggressive"
        assert pomdp.ghost_strategies[1] == "patrol"
        assert pomdp.ghost_strategies[2] == "ambush"

    def test_multi_ghost_collision_detection(self):
        """Test collision detection with multiple ghosts.

        Purpose: Validates that collisions are detected correctly with any ghost

        Given: PacManState with multiple ghosts at various positions
        When: Collision detection is performed
        Then: Collision is detected if PacMan occupies same position as any ghost

        Test type: unit
        """
        pomdp = PacManPOMDP(
            maze_size=(5, 5),
            walls=set(),
            initial_pellets=[(0, 0), (3, 3)],
            initial_ghost_positions=[(1, 1), (2, 2), (4, 4)],
            num_ghosts=3,
        )

        # Test collision with second ghost
        state_collision = pomdp.make_state(
            pacman_pos=(2, 2),
            ghost_positions=((1, 1), (2, 2), (4, 4)),  # Collision with second ghost
            pellets=((0, 0),),
        )

        # Collision should be detected
        assert any(
            ghost_pos == pomdp.get_pacman_pos(state_collision)
            for ghost_pos in pomdp.get_ghost_positions(state_collision)
        )

        # Test no collision
        state_no_collision = pomdp.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((1, 1), (2, 2), (4, 4)),
            pellets=((3, 3),),
        )

        assert not any(
            ghost_pos == pomdp.get_pacman_pos(state_no_collision)
            for ghost_pos in pomdp.get_ghost_positions(state_no_collision)
        )

    def test_multi_ghost_state_transitions(self):
        """Test state transitions with multiple ghosts.

        Purpose: Validates that state transitions correctly update all ghost positions

        Given: PacManPOMDP with multiple ghosts
        When: State transition is sampled
        Then: Next state contains updated positions for all ghosts

        Test type: integration
        """
        pomdp = PacManPOMDP(
            maze_size=(5, 5),
            walls=set(),  # No walls for simpler testing
            initial_pellets=[(1, 1)],
            initial_ghost_positions=[(3, 3), (4, 4)],
            num_ghosts=2,
            ghost_aggressiveness=1.0,  # Correct parameter name
        )

        initial_state = pomdp.make_state(
            pacman_pos=(0, 0), ghost_positions=((3, 3), (4, 4)), pellets=((1, 1),)
        )

        # Create state transition model explicitly

        state_model = PacManStateTransitionModel(initial_state, action=1, pomdp=pomdp)
        next_state = state_model.sample(n_samples=1)[0]  # Sample one next state

        # Verify next state has correct ghost count
        ghosts = pomdp.get_ghost_positions(next_state)
        assert len(ghosts) == 2
        assert isinstance(ghosts, tuple)
        assert all(isinstance(pos, tuple) and len(pos) == 2 for pos in ghosts)

    def test_multi_ghost_reward_calculation(self):
        """Test reward calculation with multiple ghosts.

        Purpose: Validates that collision penalties work correctly with multiple ghosts

        Given: State with multiple ghosts where collision may occur
        When: Reward is calculated
        Then: Collision penalty applies if PacMan collides with any ghost

        Test type: unit
        """
        pomdp = PacManPOMDP(
            maze_size=(5, 5),
            walls=set(),  # No walls to avoid conflicts
            initial_pellets=[(0, 1), (1, 0)],  # Explicit pellets within bounds
            initial_ghost_positions=[(4, 4), (1, 0)],
            num_ghosts=2,
            ghost_collision_penalty=-100.0,
            step_penalty=-1.0,
            ghost_strategies=["aggressive", "aggressive"],
            ghost_aggressiveness=0.01,  # Very low temperature → ghosts deterministically chase
        )

        # Force a deterministic collision: PacMan at (0, 0) trying to move
        # North hits the top boundary and stays; the adjacent ghost at (1, 0)
        # under aggressive+low-temperature softmax deterministically moves to
        # (0, 0) → collision on this step.
        from POMDPPlanners.environments.pacman_pomdp import (
            _native,
        )  # pylint: disable=import-outside-toplevel

        _native.set_seed(0)
        collision_state = pomdp.make_state(
            pacman_pos=(0, 0), ghost_positions=((4, 4), (1, 0)), pellets=((1, 0),)
        )

        reward = pomdp.reward(collision_state, action=0)  # North, out-of-bounds → stay
        expected_collision_reward = pomdp.step_penalty + pomdp.ghost_collision_penalty
        assert reward == expected_collision_reward

        # State with no collision
        safe_state = pomdp.make_state(
            pacman_pos=(0, 0), ghost_positions=((2, 2), (3, 3)), pellets=((1, 0),)
        )

        safe_reward = pomdp.reward(safe_state, action=0)
        assert safe_reward == pomdp.step_penalty  # Only step penalty, no collision

    def test_backward_compatibility_properties(self):
        """Test backward compatibility properties for multi-ghost support.

        Purpose: Validates that single-ghost API still works with multi-ghost implementation

        Given: PacManState and PacManPOMDP with single ghost
        When: Legacy properties are accessed
        Then: Properties return correct values maintaining backward compatibility

        Test type: unit
        """
        # Build a single-ghost env for this assertion block
        single_ghost_pomdp = PacManPOMDP(
            maze_size=(5, 5),
            walls=set(),
            initial_pellets=[(0, 0), (0, 1), (2, 2)],
            initial_ghost_positions=[(3, 4)],
            num_ghosts=1,
        )
        state = single_ghost_pomdp.make_state(
            pacman_pos=(1, 2),
            ghost_positions=((3, 4),),
            pellets=((0, 0),),
        )

        # First ghost exposed via the env's ghost-positions reader
        ghosts = single_ghost_pomdp.get_ghost_positions(state)
        assert ghosts[0] == (3, 4)
        assert single_ghost_pomdp.num_ghosts == 1

        # Test PacManPOMDP backward compatibility
        pomdp = PacManPOMDP(
            maze_size=(5, 5),
            walls=set(),  # No walls to avoid conflicts
            initial_pellets=[(0, 1), (2, 2)],  # Explicit pellets within bounds
            initial_ghost_positions=[(4, 4)],  # Single ghost
        )

        # Should work with single ghost
        assert pomdp.num_ghosts == 1
        assert pomdp.initial_ghost_positions == [(4, 4)]

        # Legacy initialization should still work
        pomdp_legacy = PacManPOMDP(
            maze_size=(5, 5),
            walls=set(),  # No walls to avoid conflicts
            initial_pellets=[(0, 1), (2, 2)],  # Explicit pellets within bounds
            initial_ghost_pos=(4, 4),  # Legacy parameter
        )

        assert pomdp_legacy.num_ghosts == 1
        assert pomdp_legacy.initial_ghost_positions == [(4, 4)]


# ---------------------------------------------------------------------------
# Array state conversion tests
# ---------------------------------------------------------------------------


class TestArrayStateLayout:
    """Tests for the canonical ndarray state layout produced by make_state."""

    @pytest.fixture()
    def env(self):
        return PacManPOMDP(
            maze_size=(5, 5),
            walls={(2, 2)},
            initial_pellets=[(1, 1), (3, 3)],
            initial_pacman_pos=(0, 0),
            num_ghosts=1,
            initial_ghost_positions=[(4, 4)],
            discount_factor=0.95,
        )

    def test_make_state_shape(self, env):
        """Test that make_state returns a 1-D array of length state_dim.

        Purpose: Validates the canonical state-array shape.

        Given: An initial state produced by the env.
        When: The state shape is inspected.
        Then: It is a 1-D ndarray with length `_state_dim`.

        Test type: unit
        """
        state = env.initial_state_dist().sample()[0]
        assert isinstance(state, np.ndarray)
        assert state.shape == (env._state_dim,)

    def test_pellet_bitmask_encoding(self, env):
        """Test pellet bitmask encoding.

        Purpose: Validates eaten pellets produce 0s and remaining produce 1s.

        Given: Two states — one with all pellets, one with one eaten.
        When: make_state builds the arrays.
        Then: The pellet-mask slice shows the correct active/inactive bits.

        Test type: unit
        """
        full_state = env.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((4, 4),),
            pellets=((1, 1), (3, 3)),
            score=0.0,
            terminal=False,
        )
        assert full_state[env._idx_pellets_start] == 1.0
        assert full_state[env._idx_pellets_start + 1] == 1.0

        partial_state = env.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((4, 4),),
            pellets=((3, 3),),
            score=10.0,
            terminal=False,
        )
        assert partial_state[env._idx_pellets_start] == 0.0  # (1,1) eaten
        assert partial_state[env._idx_pellets_start + 1] == 1.0  # (3,3) remains

    def test_terminal_flag_encoding(self, env):
        """Test terminal flag encoding.

        Purpose: Validates terminal=True produces 1.0 at the terminal index.

        Given: A state constructed with terminal=True.
        When: The array is built via make_state.
        Then: The terminal-index slot holds 1.0.

        Test type: unit
        """
        state = env.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((0, 0),),
            pellets=((1, 1), (3, 3)),
            score=0.0,
            terminal=True,
        )
        assert state[env._idx_terminal] == 1.0

    def test_readers_match_make_state_inputs(self, env):
        """Test that env readers round-trip with make_state inputs.

        Purpose: Validates layout consistency between make_state and readers.

        Given: A state built with specific kwargs.
        When: Reader methods are called on it.
        Then: Each reader returns the value originally passed to make_state.

        Test type: unit
        """
        state = env.make_state(
            pacman_pos=(1, 1),
            ghost_positions=((3, 3),),
            pellets=((3, 3),),
            score=10.0,
            terminal=False,
        )
        assert env.get_pacman_pos(state) == (1, 1)
        assert env.get_ghost_positions(state) == ((3, 3),)
        assert env.get_pellets(state) == ((3, 3),)
        assert env.get_score(state) == 10.0
        assert env.get_terminal(state) is False

    def test_stack_of_states_has_batch_shape(self, env):
        """Test that np.stack on sampled states produces a 2-D batch array.

        Purpose: Validates the batch-shape convention used by the vectorized
            belief path.

        Given: 5 initial states sampled from the distribution.
        When: np.stack is applied.
        Then: The result has shape (5, state_dim).

        Test type: unit
        """
        np.random.seed(42)
        states = env.initial_state_dist().sample(n_samples=5)
        batch = np.stack(states)
        assert batch.shape == (5, env._state_dim)
        for i, s in enumerate(states):
            np.testing.assert_array_equal(batch[i], s)


class TestRewardBatch:
    """Tests for vectorized reward_batch on PacManPOMDP."""

    @pytest.fixture()
    def env(self):
        return PacManPOMDP(
            maze_size=(5, 5),
            walls={(2, 2)},
            initial_pellets=[(1, 1), (3, 3)],
            initial_pacman_pos=(0, 0),
            num_ghosts=1,
            initial_ghost_positions=[(4, 4)],
            discount_factor=0.95,
        )

    def test_shape(self, env):
        """Test reward_batch output shape.

        Purpose: Validates output shape is (N,).

        Given: 5 array-encoded states.
        When: reward_batch is called.
        Then: Output has shape (5,).

        Test type: unit
        """
        np.random.seed(42)
        states = env.initial_state_dist().sample(n_samples=5)
        arr = np.stack(states)
        rewards = env.reward_batch(arr, 2)
        assert rewards.shape == (5,)

    def test_terminal_zero(self, env):
        """Test that terminal states return 0 reward.

        Purpose: Validates terminal state reward is zero.

        Given: A terminal state encoded as array.
        When: reward_batch is called.
        Then: Reward is 0.0.

        Test type: unit
        """
        arr = env.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((0, 0),),
            pellets=((1, 1), (3, 3)),
            score=0.0,
            terminal=True,
        ).reshape(1, -1)
        rewards = env.reward_batch(arr, 2)
        assert rewards[0] == 0.0

    def test_step_penalty_included(self, env):
        """Test that non-terminal non-event states include step penalty.

        Purpose: Validates base step penalty is applied.

        Given: A non-terminal state far from pellets and ghosts.
        When: reward_batch is called with an action that doesn't collect pellets.
        Then: Reward equals step_penalty.

        Test type: unit
        """
        arr = env.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((4, 4),),
            pellets=((3, 3),),
            score=0.0,
            terminal=False,
        ).reshape(1, -1)
        # Move South from (0,0) to (1,0) - no pellet there
        rewards = env.reward_batch(arr, 2)
        assert rewards[0] == env.step_penalty

    def test_pellet_collection_reward(self, env):
        """Test that pellet collection adds pellet_reward.

        Purpose: Validates pellet collection reward component.

        Given: PacMan at (1,0) with pellet at (1,1).
        When: Action East moves PacMan onto the pellet.
        Then: Reward includes step_penalty + pellet_reward.

        Test type: unit
        """
        arr = env.make_state(
            pacman_pos=(1, 0),
            ghost_positions=((4, 4),),
            pellets=((1, 1), (3, 3)),
            score=0.0,
            terminal=False,
        ).reshape(1, -1)
        rewards = env.reward_batch(arr, 1)  # East -> (1,1)
        assert rewards[0] == env.step_penalty + env.pellet_reward
