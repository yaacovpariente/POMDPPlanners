# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Precomputed grid navigation tables for vectorized PacMan operations.

This module provides utility functions that precompute lookup tables for
maze navigation, enabling O(1) position lookups via NumPy fancy indexing
instead of per-particle Python branching.

Functions:
    precompute_valid_cell_mask: Boolean grid marking non-wall cells.
    precompute_neighbor_table: Result position for each cell and move.
    precompute_neighbor_validity: Boolean mask of valid moves per cell.
"""

from typing import Set, Tuple

import numpy as np

# Move indices: 0=North, 1=East, 2=South, 3=West, 4=Stay
_DIRECTION_OFFSETS = np.array(
    [
        [-1, 0],  # North
        [0, 1],  # East
        [1, 0],  # South
        [0, -1],  # West
        [0, 0],  # Stay
    ],
    dtype=np.int32,
)

NUM_MOVES = 5


def precompute_valid_cell_mask(
    maze_size: Tuple[int, int],
    walls: Set[Tuple[int, int]],
) -> np.ndarray:
    """Create boolean grid where True means the cell is not a wall.

    Args:
        maze_size: Grid dimensions as (rows, cols).
        walls: Set of wall positions as (row, col) tuples.

    Returns:
        Boolean array of shape ``(rows, cols)``.
    """
    rows, cols = maze_size
    mask = np.ones((rows, cols), dtype=bool)
    for r, c in walls:
        if 0 <= r < rows and 0 <= c < cols:
            mask[r, c] = False
    return mask


def precompute_neighbor_table(
    maze_size: Tuple[int, int],
    valid_cell_mask: np.ndarray,
) -> np.ndarray:
    """Precompute resulting position for each cell and each of 5 moves.

    For each cell and move direction, stores the resulting position.
    If the move would go out of bounds or into a wall, the position
    stays the same (i.e., the agent does not move).

    Args:
        maze_size: Grid dimensions as (rows, cols).
        valid_cell_mask: Boolean array of shape ``(rows, cols)``
            from :func:`precompute_valid_cell_mask`.

    Returns:
        Integer array of shape ``(rows, cols, 5, 2)`` where
        ``result[r, c, move_idx]`` gives ``[new_row, new_col]``.
    """
    rows, cols = maze_size
    table = np.zeros((rows, cols, NUM_MOVES, 2), dtype=np.int32)

    for r in range(rows):
        for c in range(cols):
            for m in range(NUM_MOVES):
                nr = r + _DIRECTION_OFFSETS[m, 0]
                nc = c + _DIRECTION_OFFSETS[m, 1]
                if 0 <= nr < rows and 0 <= nc < cols and valid_cell_mask[nr, nc]:
                    table[r, c, m, 0] = nr
                    table[r, c, m, 1] = nc
                else:
                    table[r, c, m, 0] = r
                    table[r, c, m, 1] = c
    return table


def precompute_neighbor_validity(
    maze_size: Tuple[int, int],
    valid_cell_mask: np.ndarray,
) -> np.ndarray:
    """Precompute which moves are valid from each cell.

    A move is valid if the resulting position is in-bounds and not a wall.
    The stay move (index 4) is always valid.

    Args:
        maze_size: Grid dimensions as (rows, cols).
        valid_cell_mask: Boolean array of shape ``(rows, cols)``.

    Returns:
        Boolean array of shape ``(rows, cols, 5)`` where
        ``result[r, c, move_idx]`` is True if the move leads to a valid cell.
    """
    rows, cols = maze_size
    validity = np.zeros((rows, cols, NUM_MOVES), dtype=bool)

    for r in range(rows):
        for c in range(cols):
            for m in range(NUM_MOVES):
                nr = r + _DIRECTION_OFFSETS[m, 0]
                nc = c + _DIRECTION_OFFSETS[m, 1]
                if 0 <= nr < rows and 0 <= nc < cols and valid_cell_mask[nr, nc]:
                    validity[r, c, m] = True
                else:
                    validity[r, c, m] = False
            # Stay is always valid
            validity[r, c, 4] = True
    return validity
