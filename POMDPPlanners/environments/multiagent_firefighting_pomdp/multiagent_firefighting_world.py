# SPDX-License-Identifier: MIT

"""The world a multi-agent firefighting episode is played on, and its prior.

This module holds everything about the firefighting POMDP that is *not* the
transition, observation or reward law: the cell categories, the per-robot
actions, the eight hidden wind values, the default obstacle blob / depot /
robot start cells, and the initial state distribution.

Keeping it separate from the environment matters for one reason in particular.
The hidden state of this environment is the wind *and* the fire, and where both
come from at reset is part of the problem definition rather than an
implementation detail: a wind fixed in the configuration would be observable by
inspection, and a fire always starting in the same cell would let a planner
memorise the answer instead of inferring it.

Classes:
    FireCategory: The five per-cell categories.
    FirefightingAction: The five per-robot actions.
    WindDirection: The four wind directions.
    WindStrength: The two wind strengths.
    MultiAgentFirefightingInitialStateDistribution: The reset distribution.

Functions:
    default_obstacle_cells: The default obstacle blob for a grid.
    default_depot_cell: The default depot cell for a grid.
    default_robot_start_cells: The default robot start cells for a grid.
"""

from enum import IntEnum
from math import comb
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from POMDPPlanners.core.distributions import Distribution


class FireCategory(IntEnum):
    """What one cell of the world currently is.

    ``BURNT`` and ``WET`` are absorbing: no rule in the transition maps either
    back into ``{UNBURNT, SMOLDERING, BURNING}``. That is what makes "no cell
    is alight" a genuine terminal state rather than a moment that can be
    undone, and therefore what makes the task completion metric meaningful.

    Attributes:
        UNBURNT: Has fuel, not alight. Can ignite, or be pre-wetted.
        SMOLDERING: Alight at intensity 1. Costs 1 health per step to stand in.
        BURNING: Alight at intensity 2. Costs 2 health per step to stand in.
        BURNT: Fuel consumed, inert. Absorbing, and not enterable.
        WET: Soaked, cannot reignite. Absorbing, and enterable.
    """

    UNBURNT = 0
    SMOLDERING = 1
    BURNING = 2
    BURNT = 3
    WET = 4


#: Index of the step counter inside a state vector.
STEP_INDEX = 0
#: Index at which the per-robot block starts.
ROBOT_OFFSET = 1
#: Scalars per robot inside that block: row, column, tank, health.
ROBOT_FIELD_WIDTH = 4

#: Number of cell categories. Used by the observation confusion matrix, which
#: spreads its error mass over the ``NUM_CATEGORIES - 1`` wrong categories.
NUM_CATEGORIES = len(FireCategory)


class FirefightingAction(IntEnum):
    """One robot's action. The joint action is these in base 5.

    Attributes:
        NORTH: Attempt to move one cell north (decreasing row).
        EAST: Attempt to move one cell east (increasing column).
        SOUTH: Attempt to move one cell south (increasing row).
        WEST: Attempt to move one cell west (decreasing column).
        SUPPRESS: Spray this robot's own cell and its four neighbours.
    """

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    SUPPRESS = 4


#: Number of per-robot actions, and therefore the base the joint action is
#: written in: the joint action space has ``NUM_ROBOT_ACTIONS ** num_robots``
#: members.
NUM_ROBOT_ACTIONS = len(FirefightingAction)


class WindDirection(IntEnum):
    """The direction the hidden wind blows *towards*.

    A cell whose alight neighbour sits directly upwind of it -- that is, whose
    offset from that neighbour equals this direction's offset -- catches at the
    boosted rate. Every other neighbour is attenuated.

    Attributes:
        NORTH: Blows towards decreasing row.
        EAST: Blows towards increasing column.
        SOUTH: Blows towards increasing row.
        WEST: Blows towards decreasing column.
    """

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class WindStrength(IntEnum):
    """How hard the hidden wind blows.

    Only the strength half of the wind sets the downwind gain, which is why a
    strong wind of unknown direction is a different inference problem from a
    weak one: the belief over the eight wind values need not collapse to a
    point for a planner to act well.

    Attributes:
        LOW: The weaker gain.
        HIGH: The stronger gain.
    """

    LOW = 0
    HIGH = 1


#: Number of wind values: four directions crossed with two strengths.
NUM_WIND_VALUES = len(WindDirection) * len(WindStrength)

#: Cell offsets of the four compass directions, indexed by the shared integer
#: code that :class:`FirefightingAction` and :class:`WindDirection` both use.
#: One table because a move north and a wind blowing north are the same offset,
#: and two tables would be two places for that to drift.
DIRECTION_OFFSETS: Tuple[Tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))

#: Health lost per step standing on a cell of each category. With the default
#: ``max_health`` of 3 a robot survives one burning step and is disabled by the
#: second, which is what makes fighting from an adjacent cell the intended play
#: rather than a nicety.
HEAT_DAMAGE: Tuple[int, ...] = (0, 1, 2, 0, 0)

#: The most health one robot can lose in one step, i.e. the damage of standing
#: in a ``BURNING`` cell. Named because the declared reward range needs it.
MAX_HEAT_DAMAGE_PER_STEP = max(HEAT_DAMAGE)


def default_obstacle_cells(num_rows: int, num_cols: int) -> List[Tuple[int, int]]:
    """Return the default obstacle blob for a ``num_rows`` x ``num_cols`` grid.

    One small square block just past the middle of the grid, clipped at the
    edges. A blob rather than a scatter, and one rather than several, because
    the obstacles are here to stop the grid being trivially open -- to make a
    robot's route to the far side of a fire cost something -- not to turn the
    task into a maze.

    Args:
        num_rows: Grid rows.
        num_cols: Grid columns.

    Returns:
        The obstacle cells, in row-major order. Empty on a grid too small to
        hold the block.
    """
    top = int(num_rows) // 2
    left = int(num_cols) // 2
    return [
        (row, col)
        for row in range(top, min(top + 2, int(num_rows)))
        for col in range(left, min(left + 2, int(num_cols)))
    ]


def default_depot_cell(
    num_rows: int, num_cols: int, obstacle_cells: Sequence[Tuple[int, int]]
) -> Tuple[int, int]:
    """Return the default depot for a grid: the first non-obstacle cell.

    In row-major order that is the north-west corner unless something is
    standing on it. A corner is deliberate: the depot has to be far enough from
    where fires usually start that a refill trip costs real time, which is what
    makes the tank a constraint rather than a formality.

    Args:
        num_rows: Grid rows.
        num_cols: Grid columns.
        obstacle_cells: Cells the depot may not be placed on.

    Returns:
        The depot cell.

    Raises:
        ValueError: If every cell of the grid is an obstacle.
    """
    blocked = {(int(row), int(col)) for row, col in obstacle_cells}
    for row in range(int(num_rows)):
        for col in range(int(num_cols)):
            if (row, col) not in blocked:
                return (row, col)
    raise ValueError("every cell is an obstacle, so there is nowhere to put the depot")


def default_robot_start_cells(
    num_rows: int,
    num_cols: int,
    num_robots: int,
    obstacle_cells: Sequence[Tuple[int, int]],
    depot_cell: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Return the default start cells: the first free cells a quarter in.

    The scan begins at ``(num_rows // 4, num_cols // 4)``, wraps in row-major
    order, and skips obstacles and the depot. On the default 10x10 grid that
    puts the robots at ``(2, 2)`` and ``(2, 3)``: a short trip from the
    north-west depot, so refilling is a real but affordable interruption, and
    not so central that they start on top of every fire.

    The robots start adjacent to each other rather than spread out. Spreading
    out buys information -- two footprints that do not overlap see twice as
    much -- and starting them apart would hand a planner that benefit for free
    instead of making it choose to take it.

    Args:
        num_rows: Grid rows.
        num_cols: Grid columns.
        num_robots: How many start cells to return.
        obstacle_cells: Cells a robot may not start on.
        depot_cell: The depot, kept clear so no robot starts with a free refill.

    Returns:
        ``num_robots`` distinct start cells.

    Raises:
        ValueError: If the grid has fewer free cells than robots.
    """
    num_cells = int(num_rows) * int(num_cols)
    blocked = {(int(row), int(col)) for row, col in obstacle_cells}
    blocked.add((int(depot_cell[0]), int(depot_cell[1])))
    start = (int(num_rows) // 4) * int(num_cols) + int(num_cols) // 4
    cells: List[Tuple[int, int]] = []
    for offset in range(num_cells):
        index = (start + offset) % num_cells
        cell = (index // int(num_cols), index % int(num_cols))
        if cell not in blocked:
            cells.append(cell)
            if len(cells) == int(num_robots):
                return cells
    raise ValueError(
        f"a {num_rows}x{num_cols} grid with {len(blocked)} blocked cells cannot "
        f"seat {num_robots} robots"
    )


class MultiAgentFirefightingInitialStateDistribution(Distribution):
    """The reset distribution: a uniform wind and a uniformly placed fire.

    Everything else is fixed and known -- the step counter is 0, the robots are
    at their configured start cells, and every tank and every health bar is
    full. All the randomness is in the two things the robots cannot see at
    reset: which of the eight winds is blowing, and which cells are already
    alight.

    Every draw has at least one alight cell, so the goal is never satisfied at
    ``t = 0`` and the success reward cannot be collected for free.

    Attributes:
        state_size: Length of one state vector.
        fire_offset: Index at which the fire-map block starts.
        wind_direction_index: Index of the wind direction field.
        wind_strength_index: Index of the wind strength field.
        robot_start_cells: The robots' fixed start cells.
        max_tank: Tank capacity, which every robot starts at.
        max_health: Health capacity, which every robot starts at.
        num_initial_fires: How many cells are alight at reset.
        ignitable_indices: Flat indices of the cells a fire may start in.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        state_size: int,
        fire_offset: int,
        wind_direction_index: int,
        wind_strength_index: int,
        robot_start_cells: Sequence[Tuple[int, int]],
        max_tank: int,
        max_health: int,
        num_initial_fires: int,
        ignitable_indices: Sequence[int],
        num_cols: int,
    ):
        """Initialize the reset distribution.

        Args:
            state_size: Length of one state vector.
            fire_offset: Index at which the fire-map block starts.
            wind_direction_index: Index of the wind direction field.
            wind_strength_index: Index of the wind strength field.
            robot_start_cells: The robots' fixed start cells.
            max_tank: Tank capacity, which every robot starts at.
            max_health: Health capacity, which every robot starts at.
            num_initial_fires: How many cells are alight at reset.
            ignitable_indices: Flat indices of the cells a fire may start in,
                i.e. every non-obstacle cell.
            num_cols: Grid columns, used to flatten the robot start cells.
        """
        self.state_size = int(state_size)
        self.fire_offset = int(fire_offset)
        self.wind_direction_index = int(wind_direction_index)
        self.wind_strength_index = int(wind_strength_index)
        self.robot_start_cells = tuple((int(r), int(c)) for r, c in robot_start_cells)
        self.max_tank = int(max_tank)
        self.max_health = int(max_health)
        self.num_initial_fires = int(num_initial_fires)
        self.ignitable_indices = np.asarray(ignitable_indices, dtype=np.int64)
        self.num_cols = int(num_cols)

    def sample(self, n_samples: int = 1) -> List[Any]:
        """Draw ``n_samples`` independent initial states.

        Draws from the global ``np.random`` stream, which is the convention
        every environment here follows and what makes ``np.random.seed``
        reproduce a run.

        Args:
            n_samples: How many states to draw. Defaults to 1.

        Returns:
            A list of ``float64`` state arrays, each its own buffer so a caller
            mutating one particle cannot corrupt another.
        """
        states: List[Any] = []
        for _ in range(int(n_samples)):
            state = np.zeros(self.state_size, dtype=np.float64)
            for robot, (row, col) in enumerate(self.robot_start_cells):
                base = ROBOT_OFFSET + ROBOT_FIELD_WIDTH * robot
                state[base] = float(row)
                state[base + 1] = float(col)
                state[base + 2] = float(self.max_tank)
                state[base + 3] = float(self.max_health)
            state[self.wind_direction_index] = float(np.random.randint(len(WindDirection)))
            state[self.wind_strength_index] = float(np.random.randint(len(WindStrength)))
            ignited = np.random.choice(
                self.ignitable_indices, size=self.num_initial_fires, replace=False
            )
            state[self.fire_offset + ignited] = float(FireCategory.BURNING)
            states.append(state)
        return states

    def probability(self, values: List[Any]) -> np.ndarray:
        """Exact prior mass of each candidate initial state.

        The prior is a product of two uniforms -- one over the eight winds and
        one over the unordered sets of ``num_initial_fires`` ignitable cells --
        so unlike a placement prior with overlapping draws it has a closed
        form and is worth reporting rather than refusing.

        Args:
            values: Candidate initial states.

        Returns:
            One probability per candidate; ``0.0`` for anything this
            distribution cannot produce.
        """
        wind_count = float(len(WindDirection) * len(WindStrength))
        fire_sets = float(comb(int(self.ignitable_indices.size), self.num_initial_fires))
        mass = 1.0 / (wind_count * fire_sets)
        ignitable = set(int(index) for index in self.ignitable_indices)

        probabilities = np.zeros(len(values), dtype=np.float64)
        for position, candidate in enumerate(values):
            state = np.asarray(candidate, dtype=np.float64)
            if state.shape != (self.state_size,):
                continue
            if state[STEP_INDEX] != 0.0:
                continue
            expected_robots = []
            for row, col in self.robot_start_cells:
                expected_robots.extend(
                    [float(row), float(col), float(self.max_tank), float(self.max_health)]
                )
            robot_block = state[
                ROBOT_OFFSET : ROBOT_OFFSET + ROBOT_FIELD_WIDTH * len(self.robot_start_cells)
            ]
            if not np.array_equal(robot_block, expected_robots):
                continue
            if not 0 <= state[self.wind_direction_index] < len(WindDirection):
                continue
            if not 0 <= state[self.wind_strength_index] < len(WindStrength):
                continue
            fire = state[self.fire_offset :]
            alight = np.flatnonzero(fire == float(FireCategory.BURNING))
            if alight.size != self.num_initial_fires:
                continue
            if not set(int(index) for index in alight) <= ignitable:
                continue
            others = np.delete(fire, alight)
            if not np.all(others == float(FireCategory.UNBURNT)):
                continue
            probabilities[position] = mass
        return probabilities


def resolve_cells(
    cells: Optional[Sequence[Sequence[int]]],
    fallback: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Normalise a cell-sequence constructor argument to a list of int pairs.

    ``None`` means "use the default", which is not the same as an empty
    sequence: a caller that genuinely wants no obstacles passes ``[]`` and gets
    an open grid. Conflating the two is the shape of bug the environment API
    contract calls out, so the two branches are written apart here rather than
    behind a truthiness test.

    Args:
        cells: The argument as given, or ``None`` for the default.
        fallback: The default to use when ``cells`` is ``None``.

    Returns:
        The cells as a list of ``(row, col)`` int tuples.
    """
    chosen = fallback if cells is None else cells
    return [(int(row), int(col)) for row, col in chosen]
