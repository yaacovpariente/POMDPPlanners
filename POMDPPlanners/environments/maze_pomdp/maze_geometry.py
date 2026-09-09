# SPDX-License-Identifier: MIT

"""Generated maze geometry, shared by the discrete and the continuous maze POMDPs.

The T-Maze's difficulty knob is the length of a straight corridor, which buys
distance but not decisions. This module builds a real maze instead: a seeded
depth-first carving with a configurable number of extra walls knocked out, so the
map has branches, dead ends and loops, and its size is the difficulty knob.

Coordinates
    Grid cells ``(x, y)`` with ``x`` running right from 0 and ``y`` running *up*
    from 0, matching the T-Maze's convention. A cell is either walkable or wall.
    The outermost ring of the grid is always wall.

Dimensions
    ``width`` and ``height`` count **grid cells**, drawn cells included. Both must
    be odd — the maze alternates cell, wall, cell, wall along each axis, so an even
    dimension would leave a half wall at one edge. ``width >= 7`` and
    ``height >= 9`` are the smallest sizes with room for a start tail, three maze
    rows and two separated goal corners.

    Worked sizes: small ``7 x 9``, medium ``11 x 13``, large ``17 x 19``.

Layout
    Row 0 is wall. The **start cell** sits at ``(centre_column, 1)`` and the **cue
    cell** directly above it at ``(centre_column, 2)``. The maze proper occupies
    rows 3 upward, and the cue cell is its only connection to the start. That makes
    the cue a cut point: every route out of the start crosses it exactly once, so
    reaching either goal requires reading the cue.

    The two **goal cells** are the top-left and top-right corners of the maze,
    ``(1, height - 2)`` and ``(width - 2, height - 2)``.

Guarantees, checked at construction rather than assumed
    * every walkable cell reaches every other one — no isolated chamber;
    * removing the cue cell leaves the start alone, so the cue is unavoidable;
    * each goal is reachable from the start without stepping on the other goal;
    * the two goals' shortest-path distances from the start are within a tolerance
      of each other, so neither corner is the obvious cheap answer.

    The generator retries against derived sub-seeds until all four hold, and keeps
    the most balanced layout it saw if none does. It never draws from the global
    numpy random stream: it owns a ``numpy.random.Generator`` keyed on the seed, so
    building a maze cannot shift the draws a simulation is reproducing.

Classes:
    MazeGeometry: One generated layout.

Example:
    >>> geometry = MazeGeometry(width=7, height=9, seed=0)
    >>> geometry.start_cell, geometry.cue_cell
    ((3, 1), (3, 2))
    >>> geometry.left_goal_cell, geometry.right_goal_cell
    ((1, 7), (5, 7))
    >>> MazeGeometry(width=7, height=9, seed=0).walkable == geometry.walkable
    True
"""

from collections import deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

Cell = Tuple[int, int]

# Smallest grid that still has a start tail, three maze rows and two separated
# goal corners. Anything smaller degenerates into a corridor.
MIN_WIDTH = 7
MIN_HEIGHT = 9

# How many derived sub-seeds the generator will try before settling for the most
# balanced layout it saw. Sixty-four is far more than any observed size needs and
# still costs under a millisecond.
_MAX_ATTEMPTS = 64

# Four-neighbourhood, in a fixed order so the carving is reproducible.
_NEIGHBOUR_OFFSETS: Tuple[Cell, ...] = ((0, 1), (0, -1), (-1, 0), (1, 0))


class MazeGeometry:
    """One generated maze layout.

    Attributes:
        width: Grid columns, odd and at least :data:`MIN_WIDTH`.
        height: Grid rows, odd and at least :data:`MIN_HEIGHT`.
        seed: The seed the layout was carved from.
        loop_fraction: Fraction of the interior walls left over by the carving that
            were knocked out to create alternate routes.
        walkable: Every walkable grid cell.
        start_cell: The fixed cell every episode begins in.
        cue_cell: The one cell whose entry emits the episode's cue reading.
        left_goal_cell: The goal at the top-left corner.
        right_goal_cell: The goal at the top-right corner.
    """

    def __init__(
        self, width: int = 7, height: int = 9, seed: int = 0, loop_fraction: float = 0.15
    ) -> None:
        """Carve a maze.

        Args:
            width: Grid columns. Must be odd and at least :data:`MIN_WIDTH`.
            height: Grid rows. Must be odd and at least :data:`MIN_HEIGHT`.
            seed: Seed for the private generator that carves the maze.
            loop_fraction: Fraction of the walls the carving left standing between
                two maze cells that are knocked out afterwards, in ``[0.0, 1.0]``.
                ``0.0`` gives a perfect maze — exactly one route between any two
                cells; larger values give alternate routes. Defaults to 0.15.

        Raises:
            ValueError: If a dimension is even or below its minimum, or
                ``loop_fraction`` is outside ``[0.0, 1.0]``.
            RuntimeError: If no attempt produced a layout satisfying the
                connectivity and reachability guarantees. Balance is the only
                guarantee the generator will settle for missing.
        """
        if not isinstance(width, (int, np.integer)):
            raise ValueError(f"width must be an integer, got {width!r}.")
        if not isinstance(height, (int, np.integer)):
            raise ValueError(f"height must be an integer, got {height!r}.")
        if not isinstance(seed, (int, np.integer)):
            raise ValueError(f"seed must be an integer, got {seed!r}.")
        if width < MIN_WIDTH or width % 2 == 0:
            raise ValueError(f"width must be odd and at least {MIN_WIDTH}, got {width}.")
        if height < MIN_HEIGHT or height % 2 == 0:
            raise ValueError(f"height must be odd and at least {MIN_HEIGHT}, got {height}.")
        if not 0.0 <= loop_fraction <= 1.0:
            raise ValueError(f"loop_fraction must lie in [0.0, 1.0], got {loop_fraction}.")

        self.width = int(width)
        self.height = int(height)
        self.seed = int(seed)
        self.loop_fraction = float(loop_fraction)

        self._columns = (self.width - 1) // 2
        self._rows = (self.height - 3) // 2
        centre_column = 2 * (self._columns // 2) + 1

        self.start_cell: Cell = (centre_column, 1)
        self.cue_cell: Cell = (centre_column, 2)
        self.left_goal_cell: Cell = (1, self.height - 2)
        self.right_goal_cell: Cell = (self.width - 2, self.height - 2)

        self.walkable: FrozenSet[Cell] = self._carve_best_attempt()

    # Carving
    def _carve_best_attempt(self) -> FrozenSet[Cell]:
        """Carve against derived sub-seeds until the guarantees hold.

        Returns:
            The walkable cells of the first layout that satisfies every guarantee,
            or of the most balanced valid layout seen if none did.

        Raises:
            RuntimeError: If no attempt satisfied connectivity and reachability.
        """
        sequence = np.random.SeedSequence(self.seed)
        best: Optional[FrozenSet[Cell]] = None
        best_imbalance = np.inf
        for child in sequence.spawn(_MAX_ATTEMPTS):
            walkable = self._carve_one(np.random.default_rng(child))
            imbalance = self._imbalance(walkable)
            if imbalance is None:
                continue
            if imbalance < best_imbalance:
                best, best_imbalance = walkable, imbalance
            if imbalance <= self._balance_tolerance(walkable):
                return walkable
        if best is None:
            raise RuntimeError(
                f"No {self.width}x{self.height} layout satisfied the maze guarantees in "
                f"{_MAX_ATTEMPTS} attempts at seed {self.seed}."
            )
        return best

    def _carve_one(self, rng: np.random.Generator) -> FrozenSet[Cell]:
        """Carve one candidate layout with ``rng``.

        A randomized depth-first walk over the maze cells produces a spanning tree,
        then ``loop_fraction`` of the walls it left standing between two adjacent
        maze cells are knocked out. Removing tree edges is what turns a perfect
        maze into one with alternate routes; the depth-first walk is what gives it
        long corridors and dead ends rather than a uniform braid.

        Args:
            rng: The private generator this attempt draws from.

        Returns:
            The walkable grid cells.
        """
        entry = (self._columns // 2, 0)
        visited: Set[Cell] = {entry}
        carved: Set[Cell] = set()
        stack: List[Cell] = [entry]
        while stack:
            column, row = stack[-1]
            candidates = [
                (column + dx, row + dy)
                for dx, dy in _NEIGHBOUR_OFFSETS
                if 0 <= column + dx < self._columns
                and 0 <= row + dy < self._rows
                and (column + dx, row + dy) not in visited
            ]
            if not candidates:
                stack.pop()
                continue
            chosen = candidates[int(rng.integers(len(candidates)))]
            carved.add(self._wall_between(stack[-1], chosen))
            visited.add(chosen)
            stack.append(chosen)

        standing = sorted(self._interior_walls() - carved)
        if standing and self.loop_fraction > 0.0:
            order = rng.permutation(len(standing))
            knock_out = int(self.loop_fraction * len(standing))
            carved.update(standing[int(index)] for index in order[:knock_out])

        walkable = {
            self._maze_cell(column, row)
            for column in range(self._columns)
            for row in range(self._rows)
        }
        walkable.update(carved)
        walkable.add(self.start_cell)
        walkable.add(self.cue_cell)
        return frozenset(walkable)

    def _maze_cell(self, column: int, row: int) -> Cell:
        """The grid cell holding maze cell ``(column, row)``."""
        return (2 * column + 1, 2 * row + 3)

    def _wall_between(self, first: Cell, second: Cell) -> Cell:
        """The grid cell separating two adjacent maze cells."""
        one, two = self._maze_cell(*first), self._maze_cell(*second)
        return ((one[0] + two[0]) // 2, (one[1] + two[1]) // 2)

    def _interior_walls(self) -> Set[Cell]:
        """Every grid cell that separates two adjacent maze cells."""
        walls: Set[Cell] = set()
        for column in range(self._columns):
            for row in range(self._rows):
                if column + 1 < self._columns:
                    walls.add(
                        self._wall_between((column, row), (column + 1, row))
                    )
                if row + 1 < self._rows:
                    walls.add(
                        self._wall_between((column, row), (column, row + 1))
                    )
        return walls

    # Guarantees
    def _imbalance(self, walkable: FrozenSet[Cell]) -> Optional[float]:
        """How unequal the two goals' distances are, or ``None`` if invalid.

        Args:
            walkable: The candidate layout's walkable cells.

        Returns:
            ``abs(d_left - d_right)`` when the layout satisfies connectivity, the
            cue cut point and both goal-reachability guarantees; ``None`` otherwise.
        """
        distances = _distances_from(self.start_cell, walkable)
        if len(distances) != len(walkable):
            return None
        if _distances_from(self.start_cell, walkable - {self.cue_cell}).keys() != {self.start_cell}:
            return None
        for goal, blocked in (
            (self.left_goal_cell, self.right_goal_cell),
            (self.right_goal_cell, self.left_goal_cell),
        ):
            if goal not in _distances_from(self.start_cell, walkable - {blocked}):
                return None
        return abs(distances[self.left_goal_cell] - distances[self.right_goal_cell])

    def _balance_tolerance(self, walkable: FrozenSet[Cell]) -> float:
        """How far apart the two goal distances may be and still count as balanced.

        A fixed tolerance would be unreachable on a large maze and trivially met on
        a small one, so it scales with the distances themselves, with a floor of two
        steps because a one-step difference is inside the grid's own parity.

        Args:
            walkable: The candidate layout's walkable cells.

        Returns:
            The largest acceptable difference between the two goal distances.
        """
        distances = _distances_from(self.start_cell, walkable)
        mean = 0.5 * (
            distances[self.left_goal_cell] + distances[self.right_goal_cell]
        )
        return max(2.0, 0.2 * mean)

    # Read-only views
    @property
    def goal_cells(self) -> Tuple[Cell, Cell]:
        """The two terminal cells, left first."""
        return (self.left_goal_cell, self.right_goal_cell)

    def is_walkable(self, cell: Cell) -> bool:
        """Whether ``cell`` is inside the walkable region."""
        return cell in self.walkable

    def shortest_path_lengths(self) -> Dict[Cell, int]:
        """Steps from the start cell to every walkable cell."""
        return _distances_from(self.start_cell, self.walkable)

    def neighbours(self, cell: Cell) -> List[Cell]:
        """The walkable four-neighbours of ``cell``, in a fixed order."""
        return [
            (cell[0] + dx, cell[1] + dy)
            for dx, dy in _NEIGHBOUR_OFFSETS
            if (cell[0] + dx, cell[1] + dy) in self.walkable
        ]

    def dead_end_cells(self) -> List[Cell]:
        """Walkable cells with exactly one walkable neighbour."""
        return sorted(cell for cell in self.walkable if len(self.neighbours(cell)) == 1)

    def branch_cells(self) -> List[Cell]:
        """Walkable cells with three or more walkable neighbours."""
        return sorted(cell for cell in self.walkable if len(self.neighbours(cell)) >= 3)

    def loop_count(self) -> int:
        """Independent cycles in the walkable graph — its cyclomatic number.

        Zero means a perfect maze with exactly one route between any two cells;
        each additional cycle is one alternate route.
        """
        edges = sum(len(self.neighbours(cell)) for cell in self.walkable) // 2
        return edges - len(self.walkable) + 1

    def __repr__(self) -> str:
        return (
            f"MazeGeometry(width={self.width}, height={self.height}, seed={self.seed}, "
            f"loop_fraction={self.loop_fraction})"
        )


def _distances_from(source: Cell, walkable: FrozenSet[Cell]) -> Dict[Cell, int]:
    """Breadth-first step distances from ``source`` within ``walkable``.

    Args:
        source: The cell to search from.
        walkable: The cells the search may stand on.

    Returns:
        One entry per reachable cell. Empty if ``source`` is not walkable.
    """
    if source not in walkable:
        return {}
    distances: Dict[Cell, int] = {source: 0}
    queue: deque = deque([source])
    while queue:
        cell = queue.popleft()
        for dx, dy in _NEIGHBOUR_OFFSETS:
            neighbour = (cell[0] + dx, cell[1] + dy)
            if neighbour in walkable and neighbour not in distances:
                distances[neighbour] = distances[cell] + 1
                queue.append(neighbour)
    return distances


__all__ = ["Cell", "MIN_HEIGHT", "MIN_WIDTH", "MazeGeometry"]
