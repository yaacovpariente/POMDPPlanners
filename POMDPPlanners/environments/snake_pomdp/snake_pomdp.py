# SPDX-License-Identifier: MIT

"""Snake as a POMDP: a known body chasing a hidden apple.

The classic arcade game, with the one change that turns it into a partially
observable problem: the snake does not know where the food is. Its own body is
observed exactly -- the actions are deterministic, so the agent could compute
the body from its own history even if nothing reported it -- and all the
uncertainty therefore lives in a single hidden cell, the food.

Two sensors say something about that cell:

* a 5x5 vision window centred on the head, which reports the food's exact cell
  when the food is inside it, and does so only ``detection_probability`` of the
  time. There are no false positives, so a sighting is conclusive;
* a scent, which names one of the four diagonal quadrants relative to the head
  and is correct with probability ``scent_accuracy``. It is available on every
  step, at any distance, and is the only signal the agent has while the food is
  outside the window.

Because the body is known and the food is one cell, the belief is a categorical
distribution over the grid. That is small enough to track exactly, which is what
:class:`~POMDPPlanners.environments.snake_pomdp.snake_belief.SnakeBelief` does;
a generic particle filter would work too, but it collapses to a point mass the
moment the window reports a sighting and then has to survive the next respawn on
a resampled set.

Ported from the MDP version in `snake-rl <https://github.com/DragonWarrior15/snake-rl>`_,
which is fully observable and rewards the same events.

The walls are not cells of the state. They sit just outside the ``grid_size``
playable cells, so a head that leaves the grid has hit a wall; the renderer
draws them as a border around the playable area.

Classes:
    SnakeAction: The three actions.
    SnakeQuadrant: The four scent quadrants.
    SnakeTermination: Why an episode ended.
    SnakeStepChannel: Per-step measurement channels.
    SnakePOMDPMetrics: Metric names.
    SnakePOMDP: The environment.
"""

# pylint: disable=too-many-lines  # one environment, its sensors and its metrics

from enum import Enum, IntEnum
from pathlib import Path
from collections.abc import Hashable, Sequence as AbcSequence
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.step_info_metrics import (
    EpisodeReduction,
    StepInfoMetric,
)


class SnakeAction(IntEnum):
    """The snake's three actions, relative to its current heading.

    Reversing is not offered, which is what makes the action set relative
    rather than absolute: a snake three cells long that turned back on itself
    would walk into its own neck on every step.

    Attributes:
        TURN_LEFT: Rotate the heading 90 degrees counter-clockwise, then move.
        GO_STRAIGHT: Keep the heading, then move.
        TURN_RIGHT: Rotate the heading 90 degrees clockwise, then move.
    """

    TURN_LEFT = 0
    GO_STRAIGHT = 1
    TURN_RIGHT = 2


class SnakeQuadrant(IntEnum):
    """The four diagonal quadrants the scent can name.

    A quadrant is compatible with a food offset when every non-zero component
    of the offset agrees with it. Food sharing the head's row or column is
    therefore compatible with two quadrants, not one.

    Attributes:
        NORTH_EAST: Food above and/or to the right of the head.
        NORTH_WEST: Food above and/or to the left.
        SOUTH_EAST: Food below and/or to the right.
        SOUTH_WEST: Food below and/or to the left.
    """

    NORTH_EAST = 0
    NORTH_WEST = 1
    SOUTH_EAST = 2
    SOUTH_WEST = 3


class SnakeTermination(IntEnum):
    """Why an episode ended, stored in the state's first slot.

    Attributes:
        RUNNING: Not a terminal state.
        WALL: The head left the grid.
        SELF: The head entered a cell the body still occupies.
        STARVATION: ``starvation_limit`` steps passed with no food eaten.
        WIN: The snake reached ``target_length``.
    """

    RUNNING = 0
    WALL = 1
    SELF = 2
    STARVATION = 3
    WIN = 4


class SnakeStepChannel(Enum):
    """Per-step measurement channels reported by :meth:`SnakePOMDP.step_info`."""

    TARGET_LENGTH_REACHED = "target_length_reached"
    EPISODE_UNFINISHED = "episode_unfinished"
    EPISODE_FAILURE = "episode_failure"
    RECORDED_STEP = "recorded_step"
    FOOD_EATEN = "food_eaten"
    MAX_LENGTH = "max_length"
    WALL_DEATH = "wall_death"
    SELF_DEATH = "self_death"
    STARVATION_DEATH = "starvation_death"
    STEPS_SINCE_FOOD = "steps_since_food"


class SnakePOMDPMetrics(Enum):
    """Metric names for the Snake POMDP environment."""

    TASK_COMPLETION_RATE = "task_completion_rate"
    ENDED_BY_GOAL = "ended_by_goal"
    ENDED_BY_FAILURE = "ended_by_failure"
    ENDED_BY_TIMEOUT = "ended_by_timeout"
    AVERAGE_EPISODE_LENGTH = "average_episode_length"
    AVERAGE_FOOD_EATEN = "average_food_eaten"
    MAX_SNAKE_LENGTH = "max_snake_length"
    WALL_DEATH_RATE = "wall_death_rate"
    SELF_DEATH_RATE = "self_death_rate"
    STARVATION_DEATH_RATE = "starvation_death_rate"
    MAX_STEPS_SINCE_FOOD = "max_steps_since_food"


#: Absolute headings, in clockwise order, as ``(row, col)`` steps. Row 0 is the
#: top of the grid, so north decreases the row.
DIRECTIONS: Tuple[Tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))

#: Index of the termination reason inside a state vector.
STATUS_INDEX = 0
#: Index of the snake's length inside a state vector.
LENGTH_INDEX = 1
#: Index of the steps-since-food counter inside a state vector.
STEPS_SINCE_FOOD_INDEX = 2
#: Index of the food row inside a state vector.
FOOD_ROW_INDEX = 3
#: Index of the food column inside a state vector.
FOOD_COL_INDEX = 4
#: First index of the body block inside a state vector.
BODY_OFFSET = 5

#: Value stored in an unused body slot, and in the food slots of a won episode
#: where no food was respawned. No reachable cell uses it in both coordinates.
EMPTY = -1.0

#: Observation tag for an ordinary reading.
OBSERVATION_LIVE = 1
#: Observation tag for the terminal state's single fixed reading.
OBSERVATION_TERMINAL = 0
#: The one observation every terminal state emits. It carries no body, no
#: sighting and no scent: once the episode is over there is nothing left to
#: infer, and giving the terminal states distinct readings would let a planner
#: tell a win from a wall hit through the sensor rather than through the reward.
TERMINAL_OBSERVATION: Tuple[int, ...] = (OBSERVATION_TERMINAL,)

#: Type alias for a Snake state.
SnakeState = np.ndarray
#: Type alias for a Snake observation.
SnakeObservation = Tuple[int, ...]


def quadrants_for_offset(delta_row: int, delta_col: int) -> Tuple[int, ...]:
    """Return the quadrants compatible with a food offset from the head.

    Args:
        delta_row: Food row minus head row.
        delta_col: Food column minus head column.

    Returns:
        One quadrant when both components are non-zero, two when exactly one
        is zero.

    Raises:
        ValueError: If both components are zero. The food is never on the head:
            it is drawn from the cells the body does not occupy, and a step that
            moves the head onto it is an eat, which respawns it elsewhere.
    """
    if delta_row == 0 and delta_col == 0:
        raise ValueError("the food cell can never coincide with the head cell")
    north = [SnakeQuadrant.NORTH_EAST, SnakeQuadrant.NORTH_WEST]
    south = [SnakeQuadrant.SOUTH_EAST, SnakeQuadrant.SOUTH_WEST]
    east = [SnakeQuadrant.NORTH_EAST, SnakeQuadrant.SOUTH_EAST]
    west = [SnakeQuadrant.NORTH_WEST, SnakeQuadrant.SOUTH_WEST]
    vertical = north if delta_row < 0 else south if delta_row > 0 else north + south
    horizontal = east if delta_col > 0 else west if delta_col < 0 else east + west
    compatible = [int(quadrant) for quadrant in vertical if quadrant in horizontal]
    return tuple(sorted(compatible))


def create_snake_state(
    body: Sequence[Tuple[int, int]],
    food: Optional[Tuple[int, int]],
    steps_since_food: int = 0,
    target_length: int = 10,
    status: int = int(SnakeTermination.RUNNING),
) -> SnakeState:
    """Build a Snake state vector from its parts.

    Exists so tests and pinned scenarios can construct a specific situation
    without reproducing the layout arithmetic, which is the kind of duplication
    that goes stale silently when the layout changes.

    Args:
        body: Ordered body cells, head first.
        food: The food cell, or ``None`` for a won episode where none was
            respawned.
        steps_since_food: Steps taken since the last food. Defaults to 0.
        target_length: The environment's target length, which fixes the state
            vector's width. Defaults to 10.
        status: Termination reason. Defaults to ``SnakeTermination.RUNNING``.

    Returns:
        A ``float64`` state vector of length ``5 + 2 * target_length``.

    Raises:
        ValueError: If the body is longer than ``target_length``.
    """
    cells = [(int(row), int(col)) for row, col in body]
    if len(cells) > int(target_length):
        raise ValueError(f"body of {len(cells)} cells exceeds target_length {int(target_length)}")
    state = np.full(BODY_OFFSET + 2 * int(target_length), EMPTY, dtype=np.float64)
    state[STATUS_INDEX] = float(int(status))
    state[LENGTH_INDEX] = float(len(cells))
    state[STEPS_SINCE_FOOD_INDEX] = float(int(steps_since_food))
    if food is not None:
        state[FOOD_ROW_INDEX] = float(int(food[0]))
        state[FOOD_COL_INDEX] = float(int(food[1]))
    for index, (row, col) in enumerate(cells):
        state[BODY_OFFSET + 2 * index] = float(row)
        state[BODY_OFFSET + 2 * index + 1] = float(col)
    return state


class SnakeInitialStateDistribution(Distribution):
    """The fixed starting snake with the food uniform over the free cells.

    The body is deterministic -- head at the grid centre, three cells long,
    extending west, so the heading is east -- and the only thing drawn is the
    food cell, uniformly over the cells the body does not occupy. That makes
    this distribution exactly the prior the belief starts from.
    """

    def __init__(self, grid_size: int, target_length: int):
        """Initialize the distribution.

        Args:
            grid_size: Side length of the playable grid.
            target_length: The environment's target length, which fixes the
                state vector's width.
        """
        self.grid_size = int(grid_size)
        self.target_length = int(target_length)

    @property
    def body(self) -> Tuple[Tuple[int, int], ...]:
        """The starting body, head first."""
        centre = self.grid_size // 2
        return ((centre, centre), (centre, centre - 1), (centre, centre - 2))

    def free_cells(self) -> np.ndarray:
        """Flat indices of the cells the starting body leaves free.

        Off-grid body cells are skipped rather than folded into a flat index.
        ``SnakePOMDP.free_cells`` skips them too, and the two answers have to
        agree: this one decides where the food really spawns and the other
        decides where the belief thinks it can be, so a divergence is a prior
        that is wrong about a cell with nothing to report it. The environment's
        ``grid_size >= 4`` check means no reachable body has an off-grid cell,
        so this is the second lock on the same door rather than a live case.
        """
        occupied = {
            row * self.grid_size + col
            for row, col in self.body
            if 0 <= row < self.grid_size and 0 <= col < self.grid_size
        }
        return np.array(
            [cell for cell in range(self.grid_size**2) if cell not in occupied],
            dtype=np.int64,
        )

    def sample(self, n_samples: int = 1) -> List[Any]:
        """Draw ``n_samples`` initial states.

        Args:
            n_samples: How many states to draw. Defaults to 1.

        Returns:
            A list of ``float64`` state vectors.
        """
        free = self.free_cells()
        drawn = free[np.random.randint(0, free.size, size=int(n_samples))]
        body = self.body
        return [
            create_snake_state(
                body=body,
                food=(int(cell) // self.grid_size, int(cell) % self.grid_size),
                steps_since_food=0,
                target_length=self.target_length,
            )
            for cell in drawn
        ]

    def probability(self, values: List[Any]) -> np.ndarray:
        """Probability of each candidate initial state.

        Args:
            values: Candidate states.

        Returns:
            ``float64`` array of probabilities. Anything other than the fixed
            starting body with an unprobed counter and a free food cell has
            probability zero.
        """
        free = set(int(cell) for cell in self.free_cells())
        reference = create_snake_state(
            body=self.body, food=(0, 0), target_length=self.target_length
        )
        probs = np.zeros(len(values), dtype=np.float64)
        for index, value in enumerate(values):
            state = np.asarray(value, dtype=np.float64)
            if state.shape != reference.shape:
                continue
            if not np.array_equal(state[BODY_OFFSET:], reference[BODY_OFFSET:]):
                continue
            if state[STATUS_INDEX] != 0.0 or state[STEPS_SINCE_FOOD_INDEX] != 0.0:
                continue
            if state[LENGTH_INDEX] != reference[LENGTH_INDEX]:
                continue
            cell = int(state[FOOD_ROW_INDEX]) * self.grid_size + int(state[FOOD_COL_INDEX])
            if cell in free:
                probs[index] = 1.0 / len(free)
        return probs


class SnakePOMDP(DiscreteActionsEnvironment):  # pylint: disable=too-many-public-methods
    """Grow the snake to ``target_length`` without hitting a wall, itself or the clock.

    Dynamics:
        The body update is deterministic. The action turns the heading, the head
        steps into the next cell, and the tail is released unless the step ate
        the food -- which is what makes the snake grow. Stepping into the cell
        the tail has just left is legal; stepping into the tail while eating is
        not, because an eaten step keeps the tail where it is. The only random
        part of a transition is where the food respawns after it is eaten:
        uniformly over the cells the new body does not occupy.

    Observation model:
        The body is reported exactly. The 5x5 window around the head reports
        the food's cell with probability ``detection_probability`` when the food
        is inside it and nothing when it is not -- there are no false positives.
        The scent names a diagonal quadrant, correct with probability
        ``scent_accuracy``; when the food shares the head's row or column two
        quadrants are correct, and they split that probability between them.
        Every terminal state emits one fixed reading instead.

    Reward:
        ``+1`` for eating, ``-1`` for dying to a wall, to itself or to
        starvation, ``0`` otherwise. Winning happens by eating, so it pays the
        ``+1`` and nothing more. Eating and dying cannot both happen on one
        step: the food is never on a body cell, so a step that reaches it
        neither leaves the grid nor lands on the body.

    Terminal:
        A wall hit, a self hit, ``starvation_limit`` steps with no food, or
        ``target_length`` reached. The four are distinguished in the state so
        the metrics can report why an episode ended, but the agent's sensor
        reports the same reading for all of them.

    Attributes:
        grid_size: Side length of the playable grid.
        target_length: Body length that counts as completing the task.
        window_radius: Chebyshev radius of the vision window.
        detection_probability: Chance of seeing food that is inside the window.
        scent_accuracy: Chance the scent names a correct quadrant.
        starvation_limit: Steps without food before the snake starves.
        state_size: Width of a state vector.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> env = SnakePOMDP()
        >>> state = env.initial_state_dist().sample()[0]
        >>> env.is_terminal(state)
        False
        >>> env.snake_length(state)
        3
        >>> next_state, observation, reward = env.sample_next_step(state, 1)
        >>> observation[0]
        1
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        grid_size: int = 12,
        target_length: int = 10,
        window_radius: int = 2,
        detection_probability: float = 0.9,
        scent_accuracy: float = 0.7,
        starvation_limit: Optional[int] = None,
        discount_factor: float = 0.98,
        name: str = "Snake",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the Snake POMDP.

        Args:
            grid_size: Side length of the playable grid. Defaults to 12, which
                leaves 144 cells: large enough that the food is usually outside
                the window and the scent is the only signal, small enough that
                the exact belief is a 144-element vector.
            target_length: Body length that ends the episode as a win. Defaults
                to 10, so seven food items have to be found.
            window_radius: Chebyshev radius of the vision window. Defaults to 2,
                giving the 5x5 window the visualization draws.
            detection_probability: Chance of reporting food that is inside the
                window. Defaults to 0.9. Below 1 the agent cannot treat a silent
                window as proof the food is elsewhere, which is what keeps the
                window's absence of evidence weak evidence rather than a
                certainty.
            scent_accuracy: Chance the scent names a quadrant compatible with
                the food. Defaults to 0.7. At 0.25 the scent would be pure
                noise and at 1.0 it would localise the food to a quadrant in one
                step; 0.7 makes several readings worth accumulating.
            starvation_limit: Steps without food before the snake starves.
                Defaults to ``None``, which is ``2 * grid_size ** 2`` -- enough
                to cross the grid and search a good part of it, so starving is
                a real failure of the search rather than a step budget in
                disguise.
            discount_factor: Discount factor. Defaults to 0.98. Reaching the
                target takes hundreds of steps at ``grid_size=12``, so a
                shorter horizon would flatten the difference between finding
                food soon and finding it eventually.
            name: Environment name. Defaults to ``"Snake"``.
            output_dir: Output directory for logging. Defaults to ``None``.
            debug: Enable debug logging. Defaults to ``False``.
            use_queue_logger: Whether to use queue-based logging.

        Raises:
            ValueError: If the geometry, the sensor settings or the limits are
                outside their valid ranges.
        """
        # Four, not three. The starting body runs west from the centre column
        # ``grid_size // 2`` and occupies three columns, so a 3-wide grid puts
        # its tail at column -1 -- outside the grid, which the state's own
        # invariant forbids. That was not a loud failure: the tail's flat index
        # wrapped onto a playable cell, so the belief's prior put mass on a cell
        # the food could never spawn in and nothing raised.
        if grid_size < 4:
            raise ValueError(
                "grid_size must be at least 4: the starting body is three cells running "
                f"west from column grid_size // 2, and needs all three inside the grid, "
                f"got {grid_size}"
            )
        if target_length <= 3:
            raise ValueError(
                "target_length must exceed the starting length of 3, so that at least "
                f"one food item has to be found, got {target_length}"
            )
        if target_length > grid_size**2:
            raise ValueError(
                f"target_length {target_length} does not fit in a {grid_size}x{grid_size} grid"
            )
        if window_radius < 0:
            raise ValueError(f"window_radius must be non-negative, got {window_radius}")
        if not 0.0 <= detection_probability <= 1.0:
            raise ValueError(
                f"detection_probability must be in [0, 1], got {detection_probability}"
            )
        if not 0.0 <= scent_accuracy <= 1.0:
            raise ValueError(f"scent_accuracy must be in [0, 1], got {scent_accuracy}")

        resolved_starvation = (
            2 * int(grid_size) ** 2 if starvation_limit is None else int(starvation_limit)
        )
        if resolved_starvation < 1:
            raise ValueError(f"starvation_limit must be at least 1, got {resolved_starvation}")

        # Exactly one of the three outcomes fires per step: the snake eats, it
        # dies, or neither. Eating and dying cannot stack -- the food is never
        # on a body cell, so the step that reaches it cannot be a wall or self
        # hit -- and nothing else pays anything. The bound therefore holds for
        # every grid, every target length and every sensor setting, and is the
        # tighter pair rather than a sum.
        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
            ),
            reward_range=(-1.0, 1.0),
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        self.grid_size = int(grid_size)
        self.target_length = int(target_length)
        self.window_radius = int(window_radius)
        self.detection_probability = float(detection_probability)
        self.scent_accuracy = float(scent_accuracy)
        self.starvation_limit = resolved_starvation

        self.num_cells = self.grid_size**2
        self.state_size = BODY_OFFSET + 2 * self.target_length

    # -- state accessors ------------------------------------------------

    def snake_length(self, state: SnakeState) -> int:
        """Return the number of body cells in ``state``."""
        return int(round(float(np.asarray(state, dtype=np.float64)[LENGTH_INDEX])))

    def body(self, state: SnakeState) -> Tuple[Tuple[int, int], ...]:
        """Return the ordered body cells of ``state``, head first."""
        values = np.asarray(state, dtype=np.float64)
        length = self.snake_length(state)
        return tuple(
            (int(round(values[BODY_OFFSET + 2 * i])), int(round(values[BODY_OFFSET + 2 * i + 1])))
            for i in range(length)
        )

    def food(self, state: SnakeState) -> Optional[Tuple[int, int]]:
        """Return the food cell of ``state``, or ``None`` when there is none."""
        values = np.asarray(state, dtype=np.float64)
        row = int(round(values[FOOD_ROW_INDEX]))
        col = int(round(values[FOOD_COL_INDEX]))
        if row < 0 and col < 0:
            return None
        return (row, col)

    def steps_since_food(self, state: SnakeState) -> int:
        """Return the steps-since-food counter of ``state``."""
        return int(round(float(np.asarray(state, dtype=np.float64)[STEPS_SINCE_FOOD_INDEX])))

    def termination(self, state: SnakeState) -> SnakeTermination:
        """Return why ``state`` is terminal, or ``RUNNING`` when it is not."""
        return SnakeTermination(
            int(round(float(np.asarray(state, dtype=np.float64)[STATUS_INDEX])))
        )

    def heading(self, state: SnakeState) -> Tuple[int, int]:
        """Return the heading of ``state`` as a ``(row, col)`` step.

        The heading is not stored: it is the direction from the second body
        cell to the head, which is the only direction the head can have arrived
        from.

        Args:
            state: A state vector.

        Returns:
            One of :data:`DIRECTIONS`.

        Raises:
            ValueError: If the body is shorter than two cells, which no
                reachable state has.
        """
        cells = self.body(state)
        if len(cells) < 2:
            raise ValueError("a snake shorter than two cells has no heading")
        return (cells[0][0] - cells[1][0], cells[0][1] - cells[1][1])

    def in_grid(self, cell: Tuple[int, int]) -> bool:
        """Whether ``cell`` is one of the playable cells."""
        return 0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size

    def _running_food(self, state: SnakeState) -> Tuple[int, int]:
        """Return the food cell of a non-terminal ``state``.

        Args:
            state: A non-terminal state vector.

        Returns:
            The food cell.

        Raises:
            ValueError: If the state carries no food. The only state without
                one is the won state, which is terminal, so this means the
                caller built a state by hand that the dynamics cannot produce.
        """
        food = self.food(state)
        if food is None:
            raise ValueError("a non-terminal Snake state always carries a food cell")
        return food

    def window_cells(self, head: Tuple[int, int]) -> Tuple[Tuple[int, int], ...]:
        """Return the in-grid cells within the vision window around ``head``."""
        radius = self.window_radius
        return tuple(
            (row, col)
            for row in range(head[0] - radius, head[0] + radius + 1)
            for col in range(head[1] - radius, head[1] + radius + 1)
            if self.in_grid((row, col))
        )

    # -- dynamics -------------------------------------------------------

    def get_actions(self) -> List[int]:
        """Return the three actions: turn left, go straight, turn right."""
        return [int(action) for action in SnakeAction]

    def turn(self, heading: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Return the heading after applying ``action`` to ``heading``.

        Args:
            heading: The current heading, one of :data:`DIRECTIONS`.
            action: A :class:`SnakeAction` value.

        Returns:
            The new heading.

        Raises:
            ValueError: If ``heading`` is not an axis step or ``action`` is not
                one of the three actions.
        """
        heading = (int(heading[0]), int(heading[1]))
        if heading not in DIRECTIONS:
            raise ValueError(f"heading {heading} is not one of {DIRECTIONS}")
        action = int(action)
        if action not in (int(a) for a in SnakeAction):
            raise ValueError(f"action must be one of 0, 1, 2, got {action}")
        index = DIRECTIONS.index(heading)
        if action == int(SnakeAction.TURN_LEFT):
            return DIRECTIONS[(index - 1) % len(DIRECTIONS)]
        if action == int(SnakeAction.TURN_RIGHT):
            return DIRECTIONS[(index + 1) % len(DIRECTIONS)]
        return heading

    def transition_outcome(
        self, state: SnakeState, action: int
    ) -> Tuple[Tuple[Tuple[int, int], ...], bool, int, SnakeTermination]:
        """Resolve the deterministic half of one transition.

        Everything except where the food respawns is settled here: the new
        body, whether the step ate, the new counter and the termination reason.
        The reward and the metrics both read this, so the rules are stated once.

        Args:
            state: The state the step is taken from. Must not be terminal.
            action: A :class:`SnakeAction` value.

        Returns:
            ``(new_body, eat, steps_since_food, termination)``. ``new_body``
            keeps the head the step moved into even when that cell is off the
            grid or inside the body, so a renderer can mark where the snake
            died.
        """
        cells = self.body(state)
        head = cells[0]
        new_heading = self.turn(self.heading(state), action)
        target = (head[0] + new_heading[0], head[1] + new_heading[1])
        eat = target == self.food(state)

        # Eating keeps the tail where it is; otherwise the tail cell is
        # released, which is what makes the cell it has just left safe to enter.
        new_body = (target,) + (cells if eat else cells[:-1])

        counter = 0 if eat else self.steps_since_food(state) + 1

        # Wall and self are checked first, then the win, then starvation --
        # a snake that reaches its target length by walking into a wall has
        # still hit the wall.
        if not self.in_grid(target):
            return new_body, eat, counter, SnakeTermination.WALL
        if target in new_body[1:]:
            return new_body, eat, counter, SnakeTermination.SELF
        if len(new_body) >= self.target_length:
            return new_body, eat, counter, SnakeTermination.WIN
        if counter >= self.starvation_limit:
            return new_body, eat, counter, SnakeTermination.STARVATION
        return new_body, eat, counter, SnakeTermination.RUNNING

    def free_cells(self, body: Sequence[Tuple[int, int]]) -> np.ndarray:
        """Flat indices of the grid cells ``body`` does not occupy."""
        occupied = {
            int(row) * self.grid_size + int(col)
            for row, col in body
            if self.in_grid((int(row), int(col)))
        }
        return np.array(
            [cell for cell in range(self.num_cells) if cell not in occupied], dtype=np.int64
        )

    def _successor(
        self,
        new_body: Tuple[Tuple[int, int], ...],
        eat: bool,
        counter: int,
        termination: SnakeTermination,
        previous_food: Optional[Tuple[int, int]],
        food_cell: Optional[int] = None,
    ) -> SnakeState:
        """Assemble one successor state from a resolved transition.

        Args:
            new_body: The body after the step.
            eat: Whether the step ate the food.
            counter: The new steps-since-food value.
            termination: The termination reason.
            previous_food: The food cell before the step.
            food_cell: Flat index of the respawned food, when one was drawn.

        Returns:
            A ``float64`` state vector.
        """
        if termination in (SnakeTermination.WALL, SnakeTermination.SELF):
            # The head is off the grid or inside the body, so the body is not a
            # legal snake any more. It is kept so the renderer can mark the
            # cell the snake died in; nothing reads it as a live body, because
            # the state is terminal and every terminal state emits one reading.
            food = previous_food
        elif termination is SnakeTermination.WIN:
            # Nothing respawns: the episode is over at the moment the target
            # length is reached, so there is no food cell to name.
            food = None
        elif eat:
            food = (
                (int(food_cell) // self.grid_size, int(food_cell) % self.grid_size)
                if food_cell is not None
                else None
            )
        else:
            food = previous_food
        return create_snake_state(
            body=new_body,
            food=food,
            steps_since_food=counter,
            target_length=self.target_length,
            status=int(termination),
        )

    def sample_next_state(self, state: SnakeState, action: int, n_samples: int = 1) -> Any:
        """Move the snake, and respawn the food when it was eaten.

        Args:
            state: Current state.
            action: A :class:`SnakeAction` value.
            n_samples: How many successor samples to return. Defaults to 1.

        Returns:
            A single ``float64`` state when ``n_samples == 1``, else an
            ``(n_samples, state_size)`` ``float64`` array.

        Raises:
            ValueError: If the new body leaves no free cell for the food, which
                cannot happen while ``target_length`` fits in the grid.
        """
        count = int(n_samples)
        if self.is_terminal(state):
            absorbing = np.array(state, dtype=np.float64, copy=True)
            return absorbing if count == 1 else np.tile(absorbing, (count, 1))

        new_body, eat, counter, termination = self.transition_outcome(state, int(action))
        previous_food = self.food(state)

        respawns = eat and termination is SnakeTermination.RUNNING
        if not respawns:
            successor = self._successor(new_body, eat, counter, termination, previous_food)
            return successor if count == 1 else np.tile(successor, (count, 1))

        free = self.free_cells(new_body)
        if free.size == 0:
            raise ValueError("the body fills the grid, so the food has nowhere to respawn")
        drawn = free[np.random.randint(0, free.size, size=count)]
        samples = [
            self._successor(new_body, eat, counter, termination, previous_food, int(cell))
            for cell in drawn
        ]
        return samples[0] if count == 1 else np.asarray(samples, dtype=np.float64)

    def sample_next_state_batch(self, states: Any, action: int) -> np.ndarray:
        """Move every input snake under one action.

        Args:
            states: ``(N, state_size)`` array-like of particles.
            action: The action shared by every particle.

        Returns:
            ``(N, state_size)`` ``float64`` array. The dtype matches
            :meth:`sample_next_state` so a particle filter mixing the two paths
            cannot silently change particle precision.
        """
        rows = np.asarray(states, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        return np.asarray(
            [self.sample_next_state(state=row, action=action) for row in rows], dtype=np.float64
        )

    def transition_log_probability(
        self, state: SnakeState, action: int, next_states: Any
    ) -> np.ndarray:
        """Log-probability of each candidate successor.

        The body update is deterministic, so a candidate scores ``-inf`` unless
        it agrees with it exactly. When the step ate and the episode continues,
        the food respawns uniformly over the free cells and every consistent
        candidate shares that probability.

        Args:
            state: Current state.
            action: A :class:`SnakeAction` value.
            next_states: Candidate successors.

        Returns:
            ``(N,)`` ``float64`` array of log-probabilities.
        """
        candidates = np.asarray(next_states, dtype=np.float64)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
        scores = np.full(len(candidates), -np.inf, dtype=np.float64)

        if self.is_terminal(state):
            expected = np.asarray(state, dtype=np.float64)
            return np.where(np.all(candidates == expected, axis=1), 0.0, -np.inf)

        new_body, eat, counter, termination = self.transition_outcome(state, int(action))
        previous_food = self.food(state)
        if not (eat and termination is SnakeTermination.RUNNING):
            expected = self._successor(new_body, eat, counter, termination, previous_food)
            return np.where(np.all(candidates == expected, axis=1), 0.0, -np.inf)

        free = self.free_cells(new_body)
        log_probability = -float(np.log(free.size))
        for cell in free:
            expected = self._successor(
                new_body, eat, counter, termination, previous_food, int(cell)
            )
            scores[np.all(candidates == expected, axis=1)] = log_probability
        return scores

    # -- observations ---------------------------------------------------

    def encode_observation_tuple(
        self,
        body: Sequence[Tuple[int, int]],
        seen: Optional[Tuple[int, int]],
        scent: int,
    ) -> SnakeObservation:
        """Pack one reading into the flat integer tuple the agent receives.

        Args:
            body: The observed body, head first.
            seen: The sighted food cell, or ``None``.
            scent: The reported quadrant.

        Returns:
            ``(OBSERVATION_LIVE, scent, seen_row, seen_col, r0, c0, r1, c1, ...)``
            with ``-1, -1`` standing for "nothing seen".
        """
        seen_row, seen_col = (-1, -1) if seen is None else (int(seen[0]), int(seen[1]))
        flat: List[int] = [OBSERVATION_LIVE, int(scent), seen_row, seen_col]
        for row, col in body:
            flat.extend((int(row), int(col)))
        return tuple(flat)

    def decode_observation(
        self, observation: SnakeObservation
    ) -> Tuple[Tuple[Tuple[int, int], ...], Optional[Tuple[int, int]], int]:
        """Unpack a reading into ``(body, seen, scent)``.

        Args:
            observation: A reading produced by this environment.

        Returns:
            The observed body, the sighted cell or ``None``, and the quadrant.

        Raises:
            ValueError: If ``observation`` is the terminal reading, which has
                no body, sighting or scent to unpack.
        """
        flat = tuple(int(value) for value in observation)
        if not flat or flat[0] != OBSERVATION_LIVE:
            raise ValueError("the terminal observation carries no body, sighting or scent")
        scent = flat[1]
        seen = None if flat[2] < 0 else (flat[2], flat[3])
        cells = tuple((flat[i], flat[i + 1]) for i in range(4, len(flat), 2))
        return cells, seen, scent

    def scent_probabilities(self, head: Tuple[int, int], food: Tuple[int, int]) -> np.ndarray:
        """Probability of each quadrant given the head and the food cell.

        Args:
            head: The head cell.
            food: The food cell.

        Returns:
            ``(4,)`` ``float64`` array indexed by :class:`SnakeQuadrant`. The
            compatible quadrants share ``scent_accuracy`` between them and the
            rest share what is left, so the four always sum to one -- including
            the tie case, where the food shares the head's row or column and two
            quadrants are compatible.
        """
        compatible = quadrants_for_offset(food[0] - head[0], food[1] - head[1])
        probabilities = np.empty(len(SnakeQuadrant), dtype=np.float64)
        wrong = len(SnakeQuadrant) - len(compatible)
        probabilities[:] = (1.0 - self.scent_accuracy) / wrong
        for quadrant in compatible:
            probabilities[quadrant] = self.scent_accuracy / len(compatible)
        return probabilities

    def sample_observation(self, next_state: SnakeState, action: int, n_samples: int = 1) -> Any:
        """Report the body, a possible sighting and a noisy scent.

        Args:
            next_state: The post-transition state.
            action: Unused; the reading depends on the next state alone.
            n_samples: How many readings to draw. Defaults to 1.

        Returns:
            One observation tuple when ``n_samples == 1``, else a list of them.
        """
        del action
        count = int(n_samples)
        if self.is_terminal(next_state):
            return TERMINAL_OBSERVATION if count == 1 else [TERMINAL_OBSERVATION] * count

        cells = self.body(next_state)
        head = cells[0]
        food = self._running_food(next_state)
        inside = food in self.window_cells(head)
        scent_probabilities = self.scent_probabilities(head, food)

        readings: List[SnakeObservation] = []
        for _ in range(count):
            detected = inside and bool(np.random.random() < self.detection_probability)
            quadrant = int(np.random.choice(len(SnakeQuadrant), p=scent_probabilities))
            readings.append(
                self.encode_observation_tuple(cells, food if detected else None, quadrant)
            )
        return readings[0] if count == 1 else readings

    def observation_log_probability(
        self, next_state: SnakeState, action: int, observations: Any
    ) -> np.ndarray:
        """Log-likelihood of each candidate reading under ``next_state``.

        The three parts of a reading are independent given the state, so their
        log-likelihoods add: the body is a point mass, the sighting is a
        Bernoulli draw with no false positives, and the scent is the categorical
        law :meth:`scent_probabilities` returns.

        Args:
            next_state: The post-transition state.
            action: Unused.
            observations: Candidate readings, or one reading.

        Returns:
            ``(N,)`` ``float64`` array of log-likelihoods.
        """
        del action
        candidates = _as_observation_list(observations)
        scores = np.full(len(candidates), -np.inf, dtype=np.float64)

        if self.is_terminal(next_state):
            for index, candidate in enumerate(candidates):
                scores[index] = 0.0 if tuple(candidate) == TERMINAL_OBSERVATION else -np.inf
            return scores

        cells = self.body(next_state)
        head = cells[0]
        food = self._running_food(next_state)
        inside = food in self.window_cells(head)
        scent_probabilities = self.scent_probabilities(head, food)

        with np.errstate(divide="ignore"):
            log_scent = np.log(scent_probabilities)
        for index, candidate in enumerate(candidates):
            flat = tuple(int(value) for value in candidate)
            if not flat or flat[0] != OBSERVATION_LIVE:
                continue
            observed_body, seen, scent = self.decode_observation(flat)
            if observed_body != cells or not 0 <= scent < len(SnakeQuadrant):
                continue
            if seen is None:
                sighting = np.log1p(-self.detection_probability) if inside else 0.0
            elif inside and seen == food:
                sighting = float(np.log(self.detection_probability))
            else:
                # No false positives: a sighting of anything but the food, or
                # any sighting at all while the food is outside the window, is
                # a reading this sensor cannot produce.
                continue
            scores[index] = float(sighting) + float(log_scent[scent])
        return scores

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        """Check whether two readings are the same tuple of integers."""
        return tuple(int(value) for value in observation1) == tuple(
            int(value) for value in observation2
        )

    def hash_observation(self, observation: Any) -> Hashable:
        """Return a hashable key agreeing with :meth:`is_equal_observation`."""
        return tuple(int(value) for value in observation)

    def hash_action(self, action: Any) -> Hashable:
        """Return a hashable key for an action (already an int)."""
        return int(action)

    # -- reward ---------------------------------------------------------

    def reward(self, state: SnakeState, action: int, next_state: Any = None) -> float:
        """Score one step: ``+1`` for eating, ``-1`` for dying, ``0`` otherwise.

        The reward is a pure function of ``(state, action)``. Eating and dying
        are both settled by the deterministic half of the transition, and the
        only random part -- where the food respawns -- cannot change either, so
        ``next_state`` is accepted and ignored and
        :attr:`reward_requires_next_state` stays ``False``.

        Args:
            state: The state the step is taken from.
            action: A :class:`SnakeAction` value.
            next_state: Unused.

        Returns:
            The immediate reward.
        """
        del next_state
        if self.is_terminal(state):
            return 0.0
        _, eat, _, termination = self.transition_outcome(state, int(action))
        if eat:
            return 1.0
        if termination in (
            SnakeTermination.WALL,
            SnakeTermination.SELF,
            SnakeTermination.STARVATION,
        ):
            return -1.0
        return 0.0

    def reward_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: int,
        next_states: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    ) -> np.ndarray:
        """Score one action against a batch of states.

        Args:
            states: ``(N, state_size)`` array-like of states.
            action: A :class:`SnakeAction` value.
            next_states: Unused; see :meth:`reward`.

        Returns:
            ``(N,)`` ``float64`` array agreeing element-wise with :meth:`reward`.
        """
        del next_states
        rows = np.asarray(states, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        return np.array([self.reward(row, action) for row in rows], dtype=np.float64)

    # -- terminal / initial ---------------------------------------------

    def is_terminal(self, state: SnakeState) -> bool:
        """Whether ``state`` is one of the four terminal states."""
        return int(round(float(np.asarray(state, dtype=np.float64)[STATUS_INDEX]))) != int(
            SnakeTermination.RUNNING
        )

    def initial_state_dist(self) -> SnakeInitialStateDistribution:
        """The fixed starting snake with the food uniform over the free cells.

        The return type is narrowed from ``Distribution`` so callers can reach
        :attr:`SnakeInitialStateDistribution.body` and
        :meth:`SnakeInitialStateDistribution.free_cells` -- the belief's prior
        is built from both, and casting at every call site would be noise.
        """
        return SnakeInitialStateDistribution(
            grid_size=self.grid_size, target_length=self.target_length
        )

    def initial_observation_dist(self) -> DiscreteDistribution:
        """The pre-step reading, which carries no information.

        Returns:
            A point mass on the terminal reading's tag with no body, sighting
            or scent. Nothing conditions on it: the belief starts from the
            initial state distribution's own uniform prior over the free cells,
            and the first real reading arrives after the first action.
        """
        return DiscreteDistribution(values=[TERMINAL_OBSERVATION], probs=np.array([1.0]))

    # -- metrics --------------------------------------------------------

    def step_info(self, state: Any, action: Any, next_state: Any) -> Dict[str, float]:
        """Report the per-step channels this environment's metrics are built on.

        Draws no randomness: the eat and death channels re-run the deterministic
        half of the transition, and everything else is read off a state.

        The progress channels are read from ``next_state`` when there is one,
        for the reason Battleship documents: the episode runner checks its step
        budget before it checks terminality, so an episode whose final allowed
        step reaches the target length records no terminal bookkeeping step, and
        reading progress from ``state`` alone would score that win as a timeout.

        Args:
            state: The state the step was taken from, or the final state on the
                terminal bookkeeping step.
            action: The action taken, or ``None`` on the terminal step.
            next_state: The realised successor, or ``None`` on the terminal step.

        Returns:
            The channels named by :class:`SnakeStepChannel`. The eat channel
            reports ``0.0`` on the terminal step, where no action was taken.
        """
        eaten = 0.0
        if action is not None and next_state is not None and not self.is_terminal(state):
            _, eat, _, _ = self.transition_outcome(state, int(action))
            eaten = float(eat)

        current = state if next_state is None else next_state
        termination = self.termination(current)
        won = float(termination is SnakeTermination.WIN)
        return {
            SnakeStepChannel.TARGET_LENGTH_REACHED.value: won,
            # The three end-reason channels partition the episode between them,
            # so "unfinished" is its own channel rather than the complement of
            # the win. Deriving the timeout as ``1 - won`` would count a wall
            # death as a timeout as well as a failure, and the three rates would
            # sum to more than one.
            SnakeStepChannel.EPISODE_UNFINISHED.value: float(
                termination is SnakeTermination.RUNNING
            ),
            SnakeStepChannel.EPISODE_FAILURE.value: float(
                termination
                in (
                    SnakeTermination.WALL,
                    SnakeTermination.SELF,
                    SnakeTermination.STARVATION,
                )
            ),
            SnakeStepChannel.RECORDED_STEP.value: 1.0,
            SnakeStepChannel.FOOD_EATEN.value: eaten,
            SnakeStepChannel.MAX_LENGTH.value: float(self.snake_length(current)),
            SnakeStepChannel.WALL_DEATH.value: float(termination is SnakeTermination.WALL),
            SnakeStepChannel.SELF_DEATH.value: float(termination is SnakeTermination.SELF),
            SnakeStepChannel.STARVATION_DEATH.value: float(
                termination is SnakeTermination.STARVATION
            ),
            SnakeStepChannel.STEPS_SINCE_FOOD.value: float(self.steps_since_food(current)),
        }

    def get_metric_specs(self) -> List[StepInfoMetric]:
        """Declare the Snake metrics.

        Completion reduces with ``ANY``: reaching the target length is something
        that happens once and ends the episode, not a condition every step has
        to hold. ``ended_by_*`` reduce with ``LAST`` -- a won episode's last
        recorded step is the terminal bookkeeping step, a dead one's is the
        death, and an episode that simply ran out of runner steps is neither.

        The dangers here are the three ways to die. Each reduces with ``LAST``
        rather than ``SUM``, so each is a *rate* over episodes: a death happens
        at most once and ends the episode, and summing a state-derived channel
        would double-count it, because the death state is recorded twice --
        once as a step's successor and again as the terminal bookkeeping step.
        ``max_steps_since_food`` is the one severity metric that means
        something, since it is the only danger signal an episode that did not
        die reports at all.

        Returns:
            One spec per metric named in :class:`SnakePOMDPMetrics`.
        """
        return [
            StepInfoMetric(
                name=SnakePOMDPMetrics.TASK_COMPLETION_RATE.value,
                channel=SnakeStepChannel.TARGET_LENGTH_REACHED.value,
                per_episode=EpisodeReduction.ANY,
            ),
            StepInfoMetric(
                name=SnakePOMDPMetrics.ENDED_BY_GOAL.value,
                channel=SnakeStepChannel.TARGET_LENGTH_REACHED.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=SnakePOMDPMetrics.ENDED_BY_FAILURE.value,
                channel=SnakeStepChannel.EPISODE_FAILURE.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=SnakePOMDPMetrics.ENDED_BY_TIMEOUT.value,
                channel=SnakeStepChannel.EPISODE_UNFINISHED.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=SnakePOMDPMetrics.AVERAGE_EPISODE_LENGTH.value,
                channel=SnakeStepChannel.RECORDED_STEP.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=SnakePOMDPMetrics.AVERAGE_FOOD_EATEN.value,
                channel=SnakeStepChannel.FOOD_EATEN.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=SnakePOMDPMetrics.MAX_SNAKE_LENGTH.value,
                channel=SnakeStepChannel.MAX_LENGTH.value,
                per_episode=EpisodeReduction.MAX,
            ),
            StepInfoMetric(
                name=SnakePOMDPMetrics.WALL_DEATH_RATE.value,
                channel=SnakeStepChannel.WALL_DEATH.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=SnakePOMDPMetrics.SELF_DEATH_RATE.value,
                channel=SnakeStepChannel.SELF_DEATH.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=SnakePOMDPMetrics.STARVATION_DEATH_RATE.value,
                channel=SnakeStepChannel.STARVATION_DEATH.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=SnakePOMDPMetrics.MAX_STEPS_SINCE_FOOD.value,
                channel=SnakeStepChannel.STEPS_SINCE_FOOD.value,
                per_episode=EpisodeReduction.MAX,
            ),
        ]

    # -- visualization --------------------------------------------------

    def cache_visualization(
        self, history: List[StepData], output_dir: Path, episode_index: int
    ) -> None:
        """Write the episode's animated GIF into ``output_dir``.

        Args:
            history: Episode history.
            output_dir: Directory to write into.
            episode_index: Zero-based episode index, used to name the file.
        """
        # Imported lazily: matplotlib is heavy and every parallel worker imports
        # this module, while almost none of them render anything.
        # pylint: disable-next=import-outside-toplevel
        from POMDPPlanners.environments.snake_pomdp.snake_visualizer import SnakeVisualizer

        cache_path = output_dir / f"snake_board_{episode_index}.gif"
        SnakeVisualizer(self).create_visualization(history, cache_path)


def _as_observation_list(observations: Any) -> List[SnakeObservation]:
    """Return ``observations`` as a list of readings.

    A reading is itself a sequence of integers, so a bare reading and a list of
    readings are both sequences and have to be told apart by their first
    element. Without this, scoring one observation would be read as scoring its
    first integer.

    Args:
        observations: One reading, or a sequence of them.

    Returns:
        A list of readings.
    """
    if (
        isinstance(observations, tuple)
        and observations
        and isinstance(observations[0], (int, np.integer))
    ):
        return [observations]
    if isinstance(observations, AbcSequence) and not isinstance(observations, (str, bytes)):
        return [tuple(int(value) for value in reading) for reading in observations]
    return [tuple(int(value) for value in observations)]
