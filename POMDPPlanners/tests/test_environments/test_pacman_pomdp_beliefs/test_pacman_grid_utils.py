# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for PacMan grid navigation utility functions."""

import numpy as np

from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_beliefs.pacman_grid_utils import (
    precompute_neighbor_table,
    precompute_neighbor_validity,
    precompute_valid_cell_mask,
)


def _small_maze():
    maze_size = (5, 5)
    walls = {(1, 1), (2, 2)}
    return maze_size, walls


class TestValidCellMask:
    def test_excludes_walls(self):
        """Test that wall cells are marked False in the mask.

        Purpose: Validates wall cells are excluded from valid mask.

        Given: A 5x5 maze with walls at (1,1) and (2,2).
        When: precompute_valid_cell_mask is called.
        Then: Wall positions are False, others are True.

        Test type: unit
        """
        maze_size, walls = _small_maze()
        mask = precompute_valid_cell_mask(maze_size, walls)
        assert mask.shape == (5, 5)
        assert not mask[1, 1]
        assert not mask[2, 2]
        assert mask[0, 0]
        assert mask[4, 4]

    def test_shape(self):
        """Test output shape matches maze dimensions.

        Purpose: Validates output shape correctness.

        Given: A 5x5 maze.
        When: precompute_valid_cell_mask is called.
        Then: Output shape is (5, 5).

        Test type: unit
        """
        maze_size, walls = _small_maze()
        mask = precompute_valid_cell_mask(maze_size, walls)
        assert mask.shape == maze_size


class TestNeighborTable:
    def test_wall_blocked_move_stays(self):
        """Test that moving into a wall keeps position unchanged.

        Purpose: Validates wall collision handling in neighbor table.

        Given: A 5x5 maze with a wall at (1,1).
        When: Looking up a move from (1,0) East (which would go to (1,1)).
        Then: The result position stays at (1,0).

        Test type: unit
        """
        maze_size, walls = _small_maze()
        mask = precompute_valid_cell_mask(maze_size, walls)
        table = precompute_neighbor_table(maze_size, mask)
        # Move East from (1,0) -> should hit wall at (1,1), stay at (1,0)
        result = table[1, 0, 1]  # East
        assert result[0] == 1 and result[1] == 0

    def test_valid_move_updates_position(self):
        """Test that a valid move correctly updates position.

        Purpose: Validates position update for valid moves.

        Given: A 5x5 maze.
        When: Moving South from (0,0).
        Then: Position updates to (1,0).

        Test type: unit
        """
        maze_size, walls = _small_maze()
        mask = precompute_valid_cell_mask(maze_size, walls)
        table = precompute_neighbor_table(maze_size, mask)
        # Move South from (0,0) -> goes to (1,0) which is valid
        result = table[0, 0, 2]  # South
        assert result[0] == 1 and result[1] == 0

    def test_out_of_bounds_stays(self):
        """Test that moving out of bounds keeps position unchanged.

        Purpose: Validates boundary handling.

        Given: A 5x5 maze.
        When: Moving North from (0,0) which would go out of bounds.
        Then: Position stays at (0,0).

        Test type: unit
        """
        maze_size, walls = _small_maze()
        mask = precompute_valid_cell_mask(maze_size, walls)
        table = precompute_neighbor_table(maze_size, mask)
        result = table[0, 0, 0]  # North from (0,0)
        assert result[0] == 0 and result[1] == 0


class TestNeighborValidity:
    def test_stay_always_valid(self):
        """Test that the stay move is always valid.

        Purpose: Validates that move index 4 (stay) is always True.

        Given: A 5x5 maze.
        When: Checking validity of stay move for all cells.
        Then: Stay move (index 4) is True everywhere.

        Test type: unit
        """
        maze_size, walls = _small_maze()
        mask = precompute_valid_cell_mask(maze_size, walls)
        validity = precompute_neighbor_validity(maze_size, mask)
        assert np.all(validity[:, :, 4])

    def test_wall_blocked_move_invalid(self):
        """Test that moves into walls are marked invalid.

        Purpose: Validates wall-blocked moves are False.

        Given: A 5x5 maze with a wall at (1,1).
        When: Checking East move from (1,0).
        Then: East move is invalid.

        Test type: unit
        """
        maze_size, walls = _small_maze()
        mask = precompute_valid_cell_mask(maze_size, walls)
        validity = precompute_neighbor_validity(maze_size, mask)
        assert not validity[1, 0, 1]  # East from (1,0) -> wall at (1,1)
