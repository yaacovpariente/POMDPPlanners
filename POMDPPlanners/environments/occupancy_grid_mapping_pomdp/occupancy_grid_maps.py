# SPDX-License-Identifier: MIT

"""The map prior an occupancy-grid mapping episode draws its world from.

The hidden state of this environment is the true map, so where that map comes
from *is* the problem definition. It is drawn afresh for each episode from the
prior implemented here rather than fixed once in the environment's
configuration, and that is deliberate: with one map baked into the config every
belief particle would carry the same world, nothing would be hidden, and the
task would collapse from a POMDP into an MDP over the robot's own log-odds
bookkeeping. Within an episode the drawn map never changes, which is the sense
in which the robot faces a *fixed* unknown environment.

The prior is a small structured one -- an optional wall around the outside plus
a handful of axis-aligned rectangular blocks in the interior -- rather than
independent per-cell coin flips. Independent cells make almost every map a
featureless speckle with no corridors, no rooms and no occlusion, and occlusion
is the thing that makes an exploration problem an exploration problem.

Classes:
    OccupancyGridInitialStateDistribution: The per-episode map prior, lifted
        into the environment's state layout.

Functions:
    sample_occupancy_map: Draw one true occupancy grid.
"""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from POMDPPlanners.core.distributions import Distribution


def sample_occupancy_map(
    num_rows: int,
    num_cols: int,
    num_obstacles: int,
    max_obstacle_size: int,
    has_boundary_wall: bool,
    keep_free: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Draw one true occupancy grid from the prior.

    Draws from the global ``np.random`` stream, which is the convention every
    environment here follows and what makes ``np.random.seed`` reproduce a run.

    Args:
        num_rows: Grid rows.
        num_cols: Grid columns.
        num_obstacles: Number of rectangular blocks to place in the interior.
        max_obstacle_size: Largest side length, in cells, of one block. Each
            side is drawn uniformly from ``1..max_obstacle_size``.
        has_boundary_wall: Whether the outermost ring of cells is occupied. A
            walled grid is a closed world, so every beam that does not run out
            of range terminates on something and the map is fully resolvable in
            principle.
        keep_free: Cells that must stay free whatever the blocks land on --
            the robot's start cell and its neighbours, so an episode cannot
            begin inside a wall or immediately boxed in.

    Returns:
        ``(num_rows, num_cols)`` ``float64`` array, ``1.0`` where occupied.
    """
    occupancy = np.zeros((int(num_rows), int(num_cols)), dtype=np.float64)

    if has_boundary_wall:
        occupancy[0, :] = 1.0
        occupancy[-1, :] = 1.0
        occupancy[:, 0] = 1.0
        occupancy[:, -1] = 1.0

    margin = 1 if has_boundary_wall else 0
    low_row, high_row = margin, int(num_rows) - margin
    low_col, high_col = margin, int(num_cols) - margin
    if high_row > low_row and high_col > low_col:
        for _ in range(int(num_obstacles)):
            height = int(np.random.randint(1, int(max_obstacle_size) + 1))
            width = int(np.random.randint(1, int(max_obstacle_size) + 1))
            top = int(np.random.randint(low_row, high_row))
            left = int(np.random.randint(low_col, high_col))
            occupancy[top : min(top + height, high_row), left : min(left + width, high_col)] = 1.0

    for row, col in keep_free:
        occupancy[int(row), int(col)] = 0.0

    return occupancy


class OccupancyGridInitialStateDistribution(Distribution):
    """The environment's initial state distribution: a map prior, nothing else.

    The robot's start pose is known and identical in every draw, and the
    log-odds map starts at zero everywhere -- one unknown cell, ``p = 0.5``. All
    the randomness is in the true map, which is exactly the hidden state.

    Attributes:
        state_size: Length of one state vector.
        map_offset: Index at which the true-map block starts.
        num_rows: Grid rows.
        num_cols: Grid columns.
        start_row: Robot's start row.
        start_col: Robot's start column.
        start_heading: Robot's start heading index.
        num_obstacles: Blocks placed per draw.
        max_obstacle_size: Largest block side length.
        has_boundary_wall: Whether the outer ring is walled.
        keep_free: Cells forced free in every draw.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        start_row: int,
        start_col: int,
        start_heading: int,
        num_obstacles: int,
        max_obstacle_size: int,
        has_boundary_wall: bool,
        keep_free: Sequence[Tuple[int, int]],
        state_size: int,
        map_offset: int,
    ):
        """Initialize the distribution.

        Args:
            num_rows: Grid rows.
            num_cols: Grid columns.
            start_row: Robot's start row.
            start_col: Robot's start column.
            start_heading: Robot's start heading index.
            num_obstacles: Blocks placed per draw.
            max_obstacle_size: Largest block side length.
            has_boundary_wall: Whether the outer ring is walled.
            keep_free: Cells forced free in every draw.
            state_size: Length of one state vector.
            map_offset: Index at which the true-map block starts.
        """
        self.num_rows = int(num_rows)
        self.num_cols = int(num_cols)
        self.start_row = int(start_row)
        self.start_col = int(start_col)
        self.start_heading = int(start_heading)
        self.num_obstacles = int(num_obstacles)
        self.max_obstacle_size = int(max_obstacle_size)
        self.has_boundary_wall = bool(has_boundary_wall)
        self.keep_free = tuple((int(r), int(c)) for r, c in keep_free)
        self.state_size = int(state_size)
        self.map_offset = int(map_offset)

    def sample(self, n_samples: int = 1) -> List[Any]:
        """Draw ``n_samples`` independent initial states.

        Args:
            n_samples: How many states to draw. Defaults to 1.

        Returns:
            A list of ``float64`` state arrays, each its own buffer so a caller
            mutating one particle cannot corrupt another.
        """
        num_cells = self.num_rows * self.num_cols
        states: List[Any] = []
        for _ in range(int(n_samples)):
            state = np.zeros(self.state_size, dtype=np.float64)
            state[1] = float(self.start_row)
            state[2] = float(self.start_col)
            state[3] = float(self.start_heading)
            state[self.map_offset : self.map_offset + num_cells] = sample_occupancy_map(
                num_rows=self.num_rows,
                num_cols=self.num_cols,
                num_obstacles=self.num_obstacles,
                max_obstacle_size=self.max_obstacle_size,
                has_boundary_wall=self.has_boundary_wall,
                keep_free=self.keep_free,
            ).ravel()
            states.append(state)
        return states

    def probability(self, values: List[Any]) -> np.ndarray:
        """Not available: the map prior has no closed form.

        Args:
            values: Candidate initial states.

        Raises:
            NotImplementedError: Always. Blocks are placed independently and may
                overlap, may be clipped at the interior boundary, and are then
                overwritten by ``keep_free``, so several placement sequences
                produce the same grid and the prior mass of a map is a sum over
                an unenumerated set. Returning a plausible-looking wrong number
                here would silently corrupt any importance weight built on it;
                nothing in the environment or in the particle-filter path calls
                this, which is why leaving it unimplemented costs nothing.
        """
        raise NotImplementedError(
            "OccupancyGridInitialStateDistribution has no closed-form density: "
            "overlapping, clipped and overwritten blocks make several placement "
            "sequences produce the same map."
        )


def resolve_keep_free(
    start_row: int, start_col: int, num_rows: int, num_cols: int
) -> Tuple[Tuple[int, int], ...]:
    """Cells the map prior must leave free: the start cell and its neighbours.

    Args:
        start_row: Robot's start row.
        start_col: Robot's start column.
        num_rows: Grid rows.
        num_cols: Grid columns.

    Returns:
        In-grid ``(row, col)`` pairs, start cell first.
    """
    candidates = [
        (int(start_row), int(start_col)),
        (int(start_row) - 1, int(start_col)),
        (int(start_row) + 1, int(start_col)),
        (int(start_row), int(start_col) - 1),
        (int(start_row), int(start_col) + 1),
    ]
    return tuple(
        (row, col)
        for row, col in candidates
        if 0 <= row < int(num_rows) and 0 <= col < int(num_cols)
    )


def default_start_cell(num_rows: int, num_cols: int) -> Tuple[int, int]:
    """The grid's centre cell, which is where an episode starts by default.

    Args:
        num_rows: Grid rows.
        num_cols: Grid columns.

    Returns:
        The ``(row, col)`` centre cell.
    """
    return int(num_rows) // 2, int(num_cols) // 2


def optional_int(value: Optional[int], fallback: int) -> int:
    """Return ``value`` when it is given, else ``fallback``.

    Args:
        value: An optional integer.
        fallback: Value used when ``value`` is ``None``.

    Returns:
        The resolved integer.
    """
    return int(fallback) if value is None else int(value)
