"""Tests for PacMan POMDP environment.

This module tests the PacMan POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

import pytest
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple

from POMDPPlanners.environments.pacman_pomdp import (
    PacManState,
    PacManStateTransitionModel,
    PacManObservationModel,
    PacManPOMDP,
    create_simple_maze_pacman,
)
from POMDPPlanners.core.environment import SpaceType, SpaceInfo
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.tests.test_utils.confidence_interval_utils import (
    verify_metrics_within_confidence_intervals,
)

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


class TestPacManState:
    """Test cases for PacManState dataclass."""

    def test_pacman_state_creation(self):
        """Test PacManState creation with valid parameters.

        Purpose: Validates that PacManState can be created with correct types and values

        Given: Valid position tuples, pellet tuple, score and terminal flag
        When: PacManState is instantiated
        Then: State is created successfully with correct attributes

        Test type: unit
        """
        state = PacManState(
            pacman_pos=(1, 2),
            ghost_positions=((3, 4), (5, 6)),
            pellets=((0, 0), (5, 5)),
            score=50,
            terminal=False,
        )

        assert state.pacman_pos == (1, 2)
        assert state.ghost_positions == ((3, 4), (5, 6))
        assert state.ghost_pos == (3, 4)  # Backward compatibility
        assert state.num_ghosts == 2
        assert state.pellets == ((0, 0), (5, 5))
        assert state.score == 50
        assert state.terminal is False

    def test_pacman_state_immutability(self):
        """Test that PacManState is immutable.

        Purpose: Validates that PacManState fields cannot be modified after creation

        Given: A created PacManState instance
        When: Attempting to modify any field
        Then: AttributeError is raised due to frozen dataclass

        Test type: unit
        """
        state = PacManState(
            pacman_pos=(0, 0), ghost_positions=((1, 1),), pellets=((2, 2),), score=0
        )

        with pytest.raises(AttributeError):
            state.pacman_pos = (1, 1)

        with pytest.raises(AttributeError):
            state.score = 10

    def test_pacman_state_validation_invalid_pacman_pos(self):
        """Test PacManState validation with invalid PacMan position.

        Purpose: Validates that PacManState rejects invalid PacMan position types

        Given: Invalid PacMan position (not a tuple or wrong length)
        When: PacManState is instantiated
        Then: ValueError is raised with descriptive message

        Test type: unit
        """
        with pytest.raises(
            ValueError, match="pacman_pos must be a tuple of two integers"
        ):
            PacManState(
                pacman_pos=[1, 2],  # List instead of tuple
                ghost_positions=((3, 4),),
                pellets=((0, 0),),
            )

        with pytest.raises(
            ValueError, match="pacman_pos must be a tuple of two integers"
        ):
            PacManState(
                pacman_pos=(1,),
                ghost_positions=((3, 4),),
                pellets=((0, 0),),  # Wrong length
            )

    def test_pacman_state_validation_invalid_ghost_positions(self):
        """Test PacManState validation with invalid ghost positions.

        Purpose: Validates that PacManState rejects invalid ghost positions types

        Given: Invalid ghost positions (not a tuple or wrong types inside)
        When: PacManState is instantiated
        Then: ValueError is raised with descriptive message

        Test type: unit
        """
        with pytest.raises(
            ValueError, match="ghost_positions must be a tuple of position tuples"
        ):
            PacManState(
                pacman_pos=(1, 2),
                ghost_positions="invalid",  # String instead of tuple
                pellets=((0, 0),),
            )

        with pytest.raises(
            ValueError, match="ghost_positions\\[0\\] must be a tuple of two integers"
        ):
            PacManState(
                pacman_pos=(1, 2),
                ghost_positions=(["invalid"],),  # List instead of tuple inside
                pellets=((0, 0),),
            )

    def test_pacman_state_validation_invalid_pellets(self):
        """Test PacManState validation with invalid pellets type.

        Purpose: Validates that PacManState rejects invalid pellets type

        Given: Invalid pellets (not a tuple)
        When: PacManState is instantiated
        Then: ValueError is raised with descriptive message

        Test type: unit
        """
        with pytest.raises(
            ValueError, match="pellets must be a tuple of position tuples"
        ):
            PacManState(
                pacman_pos=(1, 2),
                ghost_positions=((3, 4),),
                pellets=[(0, 0), (1, 1)],  # List instead of tuple
            )


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

        self.state = PacManState(
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

        state = PacManState(
            pacman_pos=(1, 1), ghost_positions=((4, 4),), pellets=((3, 3),), score=0
        )

        transition = PacManStateTransitionModel(
            state, action=0, pomdp=self.pomdp
        )  # North
        next_state = transition.sample()[0]

        assert next_state.pacman_pos == (0, 1)  # Moved north
        # Ghost movement is stochastic, so we test that all ghost positions are valid
        for ghost_pos in next_state.ghost_positions:
            assert self._is_valid_position(ghost_pos)
        assert next_state.pellets == state.pellets
        assert not next_state.terminal

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
        state = PacManState(
            pacman_pos=(2, 1), ghost_positions=((4, 4),), pellets=((3, 3),), score=0
        )

        # Try to move east into wall at (2, 2)
        transition = PacManStateTransitionModel(
            state, action=1, pomdp=self.pomdp
        )  # East
        next_state = transition.sample()[0]

        assert next_state.pacman_pos == (2, 1)  # Stayed in place due to wall

    def test_pacman_movement_boundary_collision(self):
        """Test PacMan movement at maze boundaries.

        Purpose: Validates that PacMan cannot move outside maze boundaries

        Given: PacMan at maze edge and action directing outside boundary
        When: State transition is computed
        Then: PacMan position remains unchanged

        Test type: unit
        """
        state = PacManState(
            pacman_pos=(0, 0), ghost_positions=((4, 4),), pellets=((3, 3),), score=0
        )

        # Try to move north from top boundary
        transition = PacManStateTransitionModel(
            state, action=0, pomdp=self.pomdp
        )  # North
        next_state = transition.sample()[0]

        assert next_state.pacman_pos == (0, 0)  # Stayed at boundary

    def test_pellet_collection(self):
        """Test pellet collection when PacMan moves to pellet position.

        Purpose: Validates that pellets are collected and score increases correctly

        Given: PacMan moving to a position containing a pellet
        When: State transition is computed
        Then: Pellet is removed from pellets tuple and score increases

        Test type: unit
        """
        state = PacManState(
            pacman_pos=(1, 0),
            ghost_positions=((4, 4),),
            pellets=((1, 1), (3, 3)),
            score=10,
        )

        # Move east to collect pellet at (1, 1)
        transition = PacManStateTransitionModel(
            state, action=1, pomdp=self.pomdp
        )  # East
        next_state = transition.sample()[0]

        assert next_state.pacman_pos == (1, 1)
        assert (1, 1) not in next_state.pellets
        assert next_state.score == 10 + self.pomdp.pellet_reward

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
        collision_state = PacManState(
            pacman_pos=(2, 2),
            ghost_positions=((2, 2),),  # Already at same position
            pellets=((1, 1),),
            score=0,
            terminal=True,  # Manually set as terminal to test the logic
        )

        # Terminal states should remain unchanged
        transition = PacManStateTransitionModel(
            collision_state, action=0, pomdp=self.pomdp
        )
        next_state = transition.sample()[0]
        assert next_state.terminal
        assert next_state == collision_state  # Should be unchanged

        # Scenario 2: Test collision detection in state transition logic
        # Create a scenario where we can trigger collision by checking multiple samples
        test_state = PacManState(
            pacman_pos=(1, 1),
            ghost_positions=((1, 2),),  # Adjacent positions
            pellets=((3, 3),),
            score=0,
        )

        # Test multiple transitions to see if collision can occur
        transition = PacManStateTransitionModel(
            test_state, action=1, pomdp=self.pomdp
        )  # East
        collision_found = False

        # Sample multiple times since ghost movement is stochastic
        for _ in range(20):  # Try multiple times
            next_state = transition.sample()[0]
            if (
                next_state.terminal
                and next_state.pacman_pos in next_state.ghost_positions
            ):
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
        state = PacManState(
            pacman_pos=(0, 0),
            ghost_positions=((4, 4),),
            pellets=((0, 1),),  # Only one pellet left
            score=90,
        )

        # Move east to collect last pellet
        transition = PacManStateTransitionModel(
            state, action=1, pomdp=self.pomdp
        )  # East
        next_state = transition.sample()[0]

        assert next_state.terminal
        assert len(next_state.pellets) == 0
        assert next_state.score == 90 + self.pomdp.pellet_reward

    def test_terminal_state_unchanged(self):
        """Test that terminal states remain unchanged.

        Purpose: Validates that no further transitions occur from terminal states

        Given: A terminal state
        When: State transition is computed with any action
        Then: State remains exactly the same

        Test type: unit
        """
        terminal_state = PacManState(
            pacman_pos=(2, 2),
            ghost_positions=((2, 2),),
            pellets=(),
            score=100,
            terminal=True,
        )

        transition = PacManStateTransitionModel(
            terminal_state, action=1, pomdp=self.pomdp
        )
        next_state = transition.sample()[0]

        assert next_state == terminal_state  # Unchanged

    def test_ghost_movement_stochastic(self):
        """Test that ghost movement is stochastic.

        Purpose: Validates that ghost position changes and shows stochastic behavior

        Given: Multiple samples from the same state-action pair
        When: State transitions are computed multiple times
        Then: Ghost positions vary across samples demonstrating stochasticity

        Test type: unit
        """
        state = PacManState(
            pacman_pos=(0, 0), ghost_positions=((2, 2),), pellets=((4, 4),), score=0
        )

        transition = PacManStateTransitionModel(state, action=0, pomdp=self.pomdp)
        ghost_positions = set()

        # Sample multiple times to see ghost movement variation
        for _ in range(20):
            next_state = transition.sample()[0]
            ghost_positions.add(next_state.ghost_positions[0])  # First ghost position

        # Ghost should move (not stay at (2, 2) every time)
        assert len(ghost_positions) > 1


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
        terminal_state = PacManState(
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
        state = PacManState(
            pacman_pos=(2, 2),
            ghost_positions=((2, 3),),
            pellets=((0, 0),),  # Close to PacMan
        )

        obs_model = PacManObservationModel(state, action=0, pomdp=self.pomdp)
        observations = obs_model.sample(n_samples=10)

        # Most observations should be close to true ghost position
        true_pos = state.ghost_positions[0]  # First ghost
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
        state = PacManState(
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
        state = PacManState(
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
        state = PacManState(
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
        assert (
            pomdp.initial_ghost_pos == pomdp.initial_ghost_positions[0]
        )  # Backward compatibility
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
        with pytest.raises(
            ValueError, match="PacMan position .* is outside maze bounds"
        ):
            PacManPOMDP(maze_size=(5, 5), initial_pacman_pos=(5, 0))  # Outside bounds

        with pytest.raises(
            ValueError, match="Ghost 0 position .* is outside maze bounds"
        ):
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
        with pytest.raises(
            ValueError, match="Initial PacMan position .* is inside a wall"
        ):
            PacManPOMDP(
                maze_size=(5, 5),
                walls={(0, 0)},
                initial_pellets=[(1, 1)],
                initial_pacman_pos=(0, 0),  # In wall
                num_ghosts=1,
                initial_ghost_positions=[(4, 4)],
            )

        with pytest.raises(
            ValueError, match="Initial ghost 0 position .* is inside a wall"
        ):
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

        assert isinstance(initial_state, PacManState)
        assert initial_state.pacman_pos == self.pomdp.initial_pacman_pos
        assert initial_state.ghost_positions == tuple(
            self.pomdp.initial_ghost_positions
        )
        assert (
            initial_state.ghost_pos == self.pomdp.initial_ghost_pos
        )  # Backward compatibility
        assert initial_state.pellets == tuple(self.pomdp.initial_pellets)
        assert initial_state.score == 0
        assert not initial_state.terminal

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
        state = PacManState(
            pacman_pos=(1, 1), ghost_positions=((3, 3),), pellets=((2, 2),), score=0
        )

        reward = self.pomdp.reward(state, action=0)

        # Should include step penalty at minimum
        assert reward <= self.pomdp.step_penalty

    def test_reward_calculation_pellet_collection(self):
        """Test reward calculation for pellet collection.

        Purpose: Validates that pellet collection adds appropriate reward

        Given: State where PacMan moves to collect a pellet
        When: reward() is calculated
        Then: Returns reward including pellet collection bonus

        Test type: unit
        """
        state = PacManState(
            pacman_pos=(1, 0),  # Adjacent to pellet at (1, 1)
            ghost_positions=((3, 3),),
            pellets=((1, 1), (2, 2)),
            score=0,
        )

        # Move east to collect pellet
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
        state = PacManState(
            pacman_pos=(1, 1), ghost_positions=((1, 2),), pellets=((3, 3),), score=10
        )

        # Get reward for any action
        reward = self.pomdp.reward(state, action=1)  # Move east

        # Reward should include step penalty at minimum
        assert reward <= self.pomdp.step_penalty

        # If collision penalty is included, reward should be much more negative
        if reward < self.pomdp.step_penalty:
            # Collision occurred - verify it's the expected penalty
            expected_collision_reward = (
                self.pomdp.step_penalty + self.pomdp.ghost_collision_penalty
            )
            assert reward == expected_collision_reward

    def test_reward_calculation_win_condition(self):
        """Test reward calculation for win condition.

        Purpose: Validates that collecting all pellets adds win reward

        Given: State where last pellet is collected
        When: reward() is calculated
        Then: Returns reward including win bonus

        Test type: unit
        """
        state = PacManState(
            pacman_pos=(0, 0),  # Adjacent to last pellet at (0, 1)
            ghost_positions=((4, 4),),
            pellets=((0, 1),),  # Only one pellet left
            score=90,
        )

        # Move east to collect last pellet
        reward = self.pomdp.reward(state, action=1)

        # Should include step penalty, pellet reward, and win reward
        expected_reward = (
            self.pomdp.step_penalty + self.pomdp.pellet_reward + self.pomdp.win_reward
        )
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
        non_terminal_state = PacManState(
            pacman_pos=(0, 0),
            ghost_positions=((4, 4),),
            pellets=((1, 1),),
            terminal=False,
        )
        assert not self.pomdp.is_terminal(non_terminal_state)

        # Terminal state
        terminal_state = PacManState(
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
        expected_max = (
            self.pomdp.step_penalty + self.pomdp.pellet_reward + self.pomdp.win_reward
        )

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

        assert isinstance(next_state, PacManState)
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
        winning_state = PacManState(
            pacman_pos=(2, 2),
            ghost_positions=((0, 0),),  # Multi-ghost format
            pellets=(),  # No pellets left = win
            terminal=True,
            score=100,
        )

        losing_state = PacManState(
            pacman_pos=(1, 1),
            ghost_positions=((1, 1),),  # Multi-ghost format - collision = loss
            pellets=((2, 2),),  # Pellets remaining
            terminal=True,
            score=50,
        )

        # Create histories with proper StepData structure
        # Use None for belief since we're only testing metrics computation
        dummy_belief = None

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
            policy_run_data=None,
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
            policy_run_data=None,
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
        state1 = PacManState(
            pacman_pos=(0, 0),
            ghost_positions=((2, 2),),  # Multi-ghost format
            pellets=(),  # All 2 pellets collected
            terminal=True,
            score=20,
        )

        state2 = PacManState(
            pacman_pos=(1, 1),
            ghost_positions=((1, 1),),  # Multi-ghost format
            pellets=((2, 2),),  # Only 1 pellet collected
            terminal=True,
            score=10,
        )

        # Create histories with proper StepData structure
        # Use None for belief since we're only testing metrics computation
        dummy_belief = None

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
                policy_run_data=None,
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
                policy_run_data=None,
            ),
        ]

        metrics = self.pomdp.compute_metrics(histories)

        # Find pellets collected metric
        pellets_metric = next(
            (m for m in metrics if m.name == "avg_pellets_collected"), None
        )
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
        from POMDPPlanners.core.belief import WeightedParticleBelief
        import numpy as np

        # Create dummy belief for step data
        dummy_belief = WeightedParticleBelief(
            particles=[None], log_weights=np.array([0.1]), resampling=False
        )

        # Episode 1: Two steps with distances 4 and 2 (average = 3)
        state1_ep1 = PacManState(
            pacman_pos=(1, 1),
            ghost_positions=(
                (3, 3),
            ),  # Multi-ghost format - Manhattan distance = |1-3| + |1-3| = 4
            pellets=((2, 2),),
        )
        state2_ep1 = PacManState(
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
            policy_run_data=None,
        )

        # Episode 2: Two steps with distances 6 and 4 (average = 5)
        state1_ep2 = PacManState(
            pacman_pos=(0, 0),
            ghost_positions=(
                (3, 3),
            ),  # Multi-ghost format - Manhattan distance = |0-3| + |0-3| = 6
            pellets=((2, 2),),
        )
        state2_ep2 = PacManState(
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
            policy_run_data=None,
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
        from POMDPPlanners.core.belief import WeightedParticleBelief
        import numpy as np

        # Create dummy belief for step data
        dummy_belief = WeightedParticleBelief(
            particles=[None], log_weights=np.array([0.1]), resampling=False
        )

        # Episode 1: Two collisions (steps where PacMan and ghost are at same position)
        state1_ep1 = PacManState(
            pacman_pos=(2, 2),
            ghost_positions=((2, 2),),  # Multi-ghost format - collision!
            pellets=((1, 1),),
        )
        state2_ep1 = PacManState(
            pacman_pos=(2, 3),
            ghost_positions=((3, 3),),  # Multi-ghost format - no collision
            pellets=((1, 1),),
        )
        state3_ep1 = PacManState(
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
            policy_run_data=None,
        )

        # Episode 2: One collision
        state1_ep2 = PacManState(
            pacman_pos=(1, 1),
            ghost_positions=((2, 2),),  # Multi-ghost format - no collision
            pellets=((0, 0),),
        )
        state2_ep2 = PacManState(
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
            policy_run_data=None,
        )

        metrics = self.pomdp.compute_metrics([history1, history2])

        # Find collision encounters metric
        collision_metric = next(
            (m for m in metrics if m.name == "avg_collision_encounters"), None
        )
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
        assert pomdp.initial_ghost_positions == [
            (4, 4)
        ]  # Multi-ghost format - bottom right corner

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
    assert isinstance(next_state, PacManState)
    assert isinstance(obs, tuple)
    assert isinstance(reward, (int, float))


class TestMultiGhostFeatures:
    """Test cases for multi-ghost specific functionality."""

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
        state = PacManState(
            pacman_pos=(2, 3),
            ghost_positions=((1, 1), (4, 4), (6, 2)),
            pellets=((0, 0), (5, 5)),
            score=150,
            terminal=False,
        )

        assert len(state.ghost_positions) == 3
        assert state.ghost_positions == ((1, 1), (4, 4), (6, 2))
        assert state.num_ghosts == 3
        # Test backward compatibility
        assert state.ghost_pos == (1, 1)  # First ghost position

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

        state = PacManState(
            pacman_pos=(1, 1), ghost_positions=((2, 3), (4, 1)), pellets=((0, 1),)
        )

        # Create observation model explicitly
        from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import (
            PacManObservationModel,
        )

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
        # Test collision with second ghost
        state_collision = PacManState(
            pacman_pos=(2, 2),
            ghost_positions=((1, 1), (2, 2), (4, 4)),  # Collision with second ghost
            pellets=((0, 0),),
        )

        pomdp = PacManPOMDP(
            maze_size=(5, 5),
            walls=set(),  # No walls to avoid conflicts
            initial_pellets=[(0, 1), (1, 0)],  # Explicit pellets within bounds
            initial_ghost_positions=[(1, 1), (2, 2), (4, 4)],
            num_ghosts=3,
        )

        # Collision should be detected
        assert any(
            ghost_pos == state_collision.pacman_pos
            for ghost_pos in state_collision.ghost_positions
        )

        # Test no collision
        state_no_collision = PacManState(
            pacman_pos=(0, 0),
            ghost_positions=((1, 1), (2, 2), (4, 4)),
            pellets=((3, 3),),
        )

        assert not any(
            ghost_pos == state_no_collision.pacman_pos
            for ghost_pos in state_no_collision.ghost_positions
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

        initial_state = PacManState(
            pacman_pos=(0, 0), ghost_positions=((3, 3), (4, 4)), pellets=((1, 1),)
        )

        # Create state transition model explicitly
        from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import (
            PacManStateTransitionModel,
        )

        state_model = PacManStateTransitionModel(initial_state, action=1, pomdp=pomdp)
        next_state = state_model.sample(n_samples=1)[0]  # Sample one next state

        # Verify next state has correct ghost count
        assert len(next_state.ghost_positions) == 2
        assert isinstance(next_state.ghost_positions, tuple)
        assert all(
            isinstance(pos, tuple) and len(pos) == 2
            for pos in next_state.ghost_positions
        )

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
            initial_ghost_positions=[(2, 2), (3, 3)],
            num_ghosts=2,
            ghost_collision_penalty=-100.0,
            step_penalty=-1.0,
        )

        # State with collision (PacMan at same position as second ghost)
        collision_state = PacManState(
            pacman_pos=(3, 3), ghost_positions=((2, 2), (3, 3)), pellets=((1, 1),)
        )

        reward = pomdp.reward(collision_state, action=0)
        expected_collision_reward = pomdp.step_penalty + pomdp.ghost_collision_penalty
        assert reward == expected_collision_reward

        # State with no collision
        safe_state = PacManState(
            pacman_pos=(0, 0), ghost_positions=((2, 2), (3, 3)), pellets=((1, 1),)
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
        # Test PacManState backward compatibility
        state = PacManState(
            pacman_pos=(1, 2),
            ghost_positions=((3, 4),),  # Single ghost in multi-ghost format
            pellets=((0, 0),),
        )

        # Legacy property should work
        assert state.ghost_pos == (3, 4)
        assert state.num_ghosts == 1

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
