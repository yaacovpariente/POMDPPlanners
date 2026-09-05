# SPDX-License-Identifier: MIT

"""Generated-maze POMDP: the T-Maze's memory task on a real maze, in two movement models.

The agent starts at the bottom of a generated maze. One step up is the **cue cell**,
which reveals — once, noisily — which of two distant corner goals pays. The cue cell
is the only way out of the start, so it cannot be skipped; everything past it is a
maze of branches, dead ends and loops in which every observation is identical. The
agent has to carry the reading through that maze and act on it at the far end.

This is the same task the T-Maze poses, with size as the difficulty knob instead of
corridor length. On the T there is one decision, at the junction. Here the size of
the map sets how long the memory has to survive and how many wrong turns are
available, and both goals are reachable by meaningfully different routes.

Two environments share one map:

* :class:`DiscreteMazePOMDP` — integer cell positions, four one-cell moves.
* :class:`ContinuousMazePOMDP` — real positions, a bounded displacement vector per
  step, collision checked along the whole swept path.

Given the same ``maze_width``, ``maze_height``, ``maze_seed`` and ``loop_fraction``
the two see byte-identical geometry. See
:mod:`POMDPPlanners.environments.t_maze_pomdp.maze_geometry` for the layout, its
units and the guarantees it checks.

What is hidden
    Only which goal pays. The map is known and static, the start is fixed and known,
    and movement is deterministic. There is no localization observation, no motion
    or sensor noise beyond the cue's, and no hazard. A planner that tracks a belief
    over one bit and remembers it beats one that does not; nothing else is being
    tested.

State
    ``[x, y, goal_side, cue_phase]`` as a float64 array.

    * ``x``, ``y`` — position in grid coordinates, ``y`` up. Integer-valued in the
      discrete variant, real in the continuous one.
    * ``goal_side`` — :data:`GOAL_LEFT` (0.0) or :data:`GOAL_RIGHT` (1.0). Drawn
      uniformly once per episode and never changed by a transition.
    * ``cue_phase`` — :data:`CUE_UNSEEN` -> :data:`CUE_EMITTING` ->
      :data:`CUE_CONSUMED`. Carried *in the state* rather than as a flag on the
      environment, because a planner resamples transitions from arbitrary states out
      of order and an episode flag on ``self`` would be written by the search as well
      as by the world.

Actions
    Discrete: ``"up"``, ``"down"``, ``"left"``, ``"right"``, one cell each.
    Continuous: a 2-vector ``[dx, dy]``. A vector longer than ``max_step_size`` is
    **scaled down** to that length rather than rejected, so no sampler can produce an
    illegal action; a zero vector is legal and costs a step. Movement is
    deterministic in both.

The event rule
    A step's events are read off the cells its path crosses, in the order the path
    crosses them, ignoring any cell the agent was already standing in when the step
    began. Walking that list:

    * the first **wall** cell refuses the whole move — the agent stays exactly where
      it was and pays the step cost;
    * otherwise the first **goal** cell ends the step: the agent stops at the point
      where its path first entered that cell, and the state is terminal;
    * otherwise, if the **cue** cell is crossed and the cue has not been read, the
      cue is armed and the agent completes the move.

    One rule, used by the transition, the observation model, the reward and the
    metrics, in both variants. In the discrete variant the path is one cell centre to
    the next and the rule collapses to a cell lookup; in the continuous one it is
    what stops a long displacement jumping a wall, slipping through the corner point
    where two walls meet, or passing straight through a goal.

Observations
    ``"left_cue"``, ``"right_cue"``, ``"empty"``, the T-Maze's alphabet. A state with
    ``cue_phase == CUE_EMITTING`` names the true goal side with probability
    ``cue_accuracy`` and contradicts it otherwise; every other state reads ``"empty"``
    with probability 1. The cue is emitted by the step that crosses the cue cell and
    consumed by the next action whatever it is, so exactly one reading exists per
    episode — a revisit, a stay, or a bump into a wall while standing on the cue cell
    never produces a second one.

Reward
    ``+goal_reward`` for entering the goal the hidden side names,
    ``-wrong_goal_penalty`` for entering the other, ``-step_penalty`` for every other
    action including a refused move, and ``0`` from an absorbing terminal state. The
    terminal payout *replaces* the step cost rather than stacking with it.

Task completion means entering the **correct** goal. Entering the wrong one also ends
the episode and is reported separately, because a planner that guesses at the far end
and one that never gets there fail for opposite reasons.

Classes:
    BaseMazePOMDP: Everything the two variants share.
    DiscreteMazePOMDP: Cell positions, four one-cell moves.
    ContinuousMazePOMDP: Real positions, bounded displacement vectors.
    MazeStepChannel: Per-step measurement channel names.
    MazeMetric: Episode-level metric names.

Example:
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> small = DiscreteMazePOMDP(discount_factor=0.95)
    >>> small.geometry.width, small.geometry.height
    (7, 9)
    >>> medium = DiscreteMazePOMDP(
    ...     discount_factor=0.95, maze_width=11, maze_height=13, maze_seed=4
    ... )
    >>> medium.geometry.width, medium.geometry.height
    (11, 13)
    >>> continuous = ContinuousMazePOMDP(
    ...     discount_factor=0.95, maze_width=11, maze_height=13,
    ...     maze_seed=4, max_step_size=0.75,
    ... )
    >>> continuous.geometry.walkable == medium.geometry.walkable
    True
    >>> state = small.initial_state_dist().sample()[0]
    >>> next_state, observation, reward = small.sample_next_step(state, "up")
    >>> observation in ("left_cue", "right_cue")
    True
    >>> float(reward)
    -1.0
"""

import math
from collections.abc import Hashable
from enum import Enum
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    Environment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.step_info_metrics import EpisodeReduction, StepInfoMetric
from POMDPPlanners.environments.t_maze_pomdp.maze_geometry import Cell, MazeGeometry

# State slot indices. Deliberately the same layout the T-Maze uses, so a reader who
# knows one knows the other and the two can share a mental model.
STATE_X = 0
STATE_Y = 1
STATE_GOAL = 2
STATE_CUE_PHASE = 3
STATE_WIDTH = 4

# Goal-side encodings.
GOAL_LEFT = 0.0
GOAL_RIGHT = 1.0

# Cue delivery phases.
CUE_UNSEEN = 0.0
CUE_EMITTING = 1.0
CUE_CONSUMED = 2.0

# Observation alphabet. The same three labels the T-Maze uses, on purpose: the task
# is the same and results on the two should read against each other. There is no
# wall observation — it would leak position information the task is not about.
OBSERVATION_LEFT_CUE = "left_cue"
OBSERVATION_RIGHT_CUE = "right_cue"
OBSERVATION_EMPTY = "empty"
OBSERVATIONS: Tuple[str, ...] = (
    OBSERVATION_LEFT_CUE,
    OBSERVATION_RIGHT_CUE,
    OBSERVATION_EMPTY,
)

# Discrete actions and their (dx, dy) offsets.
ACTION_UP = "up"
ACTION_DOWN = "down"
ACTION_LEFT = "left"
ACTION_RIGHT = "right"
ACTION_OFFSETS: Dict[str, Cell] = {
    ACTION_UP: (0, 1),
    ACTION_DOWN: (0, -1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}
ACTIONS: Tuple[str, ...] = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)

# A cell owns the closed square of side 1 centred on it. The tolerance widens that
# square by a nanometre of grid so a point computed as 3.4999999996 still counts as
# having reached the boundary at 3.5. Without it a segment stopped exactly on a
# goal's edge could round to just outside the goal and fail to be terminal. It errs
# towards *counting* a touch, which is the conservative direction for walls too.
_CELL_TOLERANCE = 1e-9


class MazeStepChannel(Enum):
    """Per-step measurement channels written to ``StepData.info``."""

    CORRECT_GOAL = "correct_goal"
    WRONG_GOAL = "wrong_goal"
    ENDED_BY_GOAL = "ended_by_goal"
    ENDED_BY_FAILURE = "ended_by_failure"
    ENDED_BY_TIMEOUT = "ended_by_timeout"
    RECORDED_STEP = "recorded_step"
    WALL_COLLISION = "wall_collision"


class MazeMetric(Enum):
    """Episode-level metric names reported by ``compute_metrics``.

    The names match the T-Maze's on purpose: same task, same quantities, so a table
    of results can hold both.
    """

    TASK_COMPLETION_RATE = "task_completion_rate"
    WRONG_GOAL_RATE = "wrong_goal_rate"
    ENDED_BY_GOAL_RATE = "ended_by_goal_rate"
    ENDED_BY_FAILURE_RATE = "ended_by_failure_rate"
    ENDED_BY_TIMEOUT_RATE = "ended_by_timeout_rate"
    AVERAGE_EPISODE_LENGTH = "average_episode_length"
    AVERAGE_WALL_COLLISIONS = "average_wall_collisions"


def create_maze_state(
    position: Tuple[float, float], goal_side: float, cue_phase: float = CUE_UNSEEN
) -> np.ndarray:
    """Build a maze state array.

    Args:
        position: ``(x, y)`` in grid coordinates.
        goal_side: :data:`GOAL_LEFT` or :data:`GOAL_RIGHT`.
        cue_phase: One of :data:`CUE_UNSEEN`, :data:`CUE_EMITTING`,
            :data:`CUE_CONSUMED`. Defaults to :data:`CUE_UNSEEN`.

    Returns:
        The state as a float64 array ``[x, y, goal_side, cue_phase]``.
    """
    return np.array(
        [float(position[0]), float(position[1]), float(goal_side), float(cue_phase)],
        dtype=np.float64,
    )


class StepOutcome:
    """What one step's path did, before any of it is written into a state.

    Attributes:
        position: Where the agent ends up.
        blocked: Whether the move was refused by a wall.
        entered_cue: Whether the path crossed the cue cell.
        goal_cell: The goal cell the path entered, or ``None``.
    """

    __slots__ = ("position", "blocked", "entered_cue", "goal_cell")

    def __init__(
        self,
        position: Tuple[float, float],
        blocked: bool,
        entered_cue: bool,
        goal_cell: Optional[Cell],
    ) -> None:
        self.position = position
        self.blocked = blocked
        self.entered_cue = entered_cue
        self.goal_cell = goal_cell


class BaseMazePOMDP(Environment):
    """Everything the discrete and continuous maze variants share.

    Subclasses supply :meth:`_execute`, which turns a state and an action into a
    :class:`StepOutcome`. The cue phase, the reward, the observation model, the
    metrics and the visualization hook are written once here and are identical in
    both, which is what makes the two variants the same task under two movement
    models rather than two environments that resemble each other.

    Attributes:
        maze_width: Grid columns of the generated map.
        maze_height: Grid rows of the generated map.
        maze_seed: Seed the map was carved from.
        loop_fraction: Fraction of leftover interior walls knocked out for loops.
        cue_accuracy: Probability the cue names the true goal side.
        goal_reward: Paid for entering the correct goal.
        wrong_goal_penalty: Magnitude of the penalty for entering the other goal.
        step_penalty: Magnitude of the per-action cost on non-terminal actions.
    """

    def __init__(
        self,
        discount_factor: float,
        name: str,
        space_info: SpaceInfo,
        maze_width: int,
        maze_height: int,
        maze_seed: int,
        loop_fraction: float,
        cue_accuracy: float,
        goal_reward: float,
        wrong_goal_penalty: float,
        step_penalty: float,
        output_dir: Optional[Path],
        debug: bool,
        use_queue_logger: bool,
    ) -> None:
        """Initialize the shared part of a maze environment.

        Args:
            discount_factor: Discount factor for future rewards.
            name: Environment name.
            space_info: Action and observation space types.
            maze_width: Grid columns; odd, at least 7.
            maze_height: Grid rows; odd, at least 9.
            maze_seed: Seed for the private generator that carves the map.
            loop_fraction: Fraction of leftover interior walls knocked out.
            cue_accuracy: Probability the cue names the true goal side, in
                ``[0.5, 1.0]``.
            goal_reward: Reward for entering the correct goal.
            wrong_goal_penalty: Magnitude of the wrong-goal penalty.
            step_penalty: Magnitude of the per-action cost.
            output_dir: Optional directory for logging output.
            debug: Enable debug logging.
            use_queue_logger: Whether to use queue-based logging.

        Raises:
            ValueError: If ``cue_accuracy`` is outside ``[0.5, 1.0]``. A cue below
                0.5 is the same task with the labels swapped, and accepting it would
                make two different configs mean one environment.
        """
        if not 0.5 <= cue_accuracy <= 1.0:
            raise ValueError(f"cue_accuracy must lie in [0.5, 1.0], got {cue_accuracy}.")
        for label, value in (
            ("goal_reward", goal_reward),
            ("wrong_goal_penalty", wrong_goal_penalty),
            ("step_penalty", step_penalty),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be a finite non-negative magnitude, got {value}.")

        # Public, so every parameter that shapes the map, the collisions or the
        # actions reaches ``config_id`` and ``to_dict``. The generated geometry
        # itself is private below: it is a pure function of these four, and hashing
        # it would only add ways for the identity to drift.
        geometry = MazeGeometry(
            width=maze_width,
            height=maze_height,
            seed=maze_seed,
            loop_fraction=loop_fraction,
        )
        self.maze_width = geometry.width
        self.maze_height = geometry.height
        self.maze_seed = geometry.seed
        self.loop_fraction = geometry.loop_fraction
        self.cue_accuracy = float(cue_accuracy)
        self.goal_reward = float(goal_reward)
        self.wrong_goal_penalty = float(wrong_goal_penalty)
        self.step_penalty = float(step_penalty)

        # Every reward the environment can produce, enumerated rather than
        # estimated: +goal_reward on the correct goal, -wrong_goal_penalty on the
        # other, -step_penalty on every other action, and 0.0 from an absorbing
        # terminal state. The terminal payouts replace the step cost rather than
        # stacking with it, so no two terms ever add. The 0.0 keeps the bound honest
        # when a caller sets a penalty to zero or a negative reward.
        min_reward = min(0.0, -wrong_goal_penalty, -step_penalty, goal_reward)
        max_reward = max(0.0, goal_reward, -step_penalty, -wrong_goal_penalty)

        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=(min_reward, max_reward),
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        # Built once, in the constructor rather than on first use: a lazily
        # memoized attribute would change ``config_id`` the moment the environment
        # was used, and ``config_id`` is a cache key.
        self._geometry = geometry
        self._walkable = self._geometry.walkable
        self._cue_cell = self._geometry.cue_cell
        self._start_cell = self._geometry.start_cell
        self._left_goal_cell = self._geometry.left_goal_cell
        self._right_goal_cell = self._geometry.right_goal_cell
        self._goal_cells = frozenset(self._geometry.goal_cells)

    # Geometry
    @property
    def geometry(self) -> MazeGeometry:
        """The generated layout both variants are built on."""
        return self._geometry

    @property
    def start_cell(self) -> Cell:
        """The fixed cell every episode starts from."""
        return self._start_cell

    @property
    def cue_cell(self) -> Cell:
        """The one cell whose crossing emits the episode's cue reading."""
        return self._cue_cell

    @property
    def left_goal_cell(self) -> Cell:
        """The terminal cell at the top-left corner."""
        return self._left_goal_cell

    @property
    def right_goal_cell(self) -> Cell:
        """The terminal cell at the top-right corner."""
        return self._right_goal_cell

    @property
    def walkable_cells(self) -> Any:
        """Every walkable grid cell, as ``(x, y)`` pairs."""
        return self._walkable

    def goal_cell(self, goal_side: float) -> Cell:
        """The goal cell that pays the goal reward for ``goal_side``."""
        return (
            self._left_goal_cell
            if float(goal_side) == GOAL_LEFT
            else self._right_goal_cell
        )

    # The event rule
    def _cells_containing(self, x: float, y: float) -> Tuple[Cell, ...]:
        """Every grid cell whose closed unit square contains ``(x, y)``.

        A point strictly inside a cell yields one cell; a point on an edge yields the
        two cells sharing it; a point on a corner yields all four. That is what makes
        a diagonal move through the corner point between two walls illegal rather
        than a way through.
        """
        columns = _boundary_span(x)
        rows = _boundary_span(y)
        return tuple((column, row) for column in columns for row in rows)

    def _crossed_cells(
        self, start: Tuple[float, float], end: Tuple[float, float]
    ) -> List[Tuple[float, Cell]]:
        """Cells the closed segment ``start -> end`` meets, with the time it meets them.

        The segment is cut at every cell boundary it crosses. Between two consecutive
        cuts it lies in exactly one cell, found from the midpoint; at a cut it may
        touch two or four, found from the cut point itself. That covers the whole
        segment exactly, with no sampling and no step size to get wrong.

        Args:
            start: The point the step begins at.
            end: The point it would end at.

        Returns:
            ``(entry_time, cell)`` pairs sorted by entry time, each cell listed once
            at the earliest time the segment is inside it.
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        cuts = {0.0, 1.0}
        for origin, delta in ((start[0], dx), (start[1], dy)):
            if delta == 0.0:
                continue
            low, high = sorted((origin, origin + delta))
            first = math.floor(low + 0.5)
            last = math.ceil(high + 0.5)
            for index in range(first, last + 1):
                time = (index - 0.5 - origin) / delta
                if 0.0 < time < 1.0:
                    cuts.add(time)

        ordered = sorted(cuts)
        earliest: Dict[Cell, float] = {}
        for index, time in enumerate(ordered):
            point = (start[0] + dx * time, start[1] + dy * time)
            for cell in self._cells_containing(*point):
                earliest.setdefault(cell, time)
            if index + 1 < len(ordered):
                middle = 0.5 * (time + ordered[index + 1])
                point = (start[0] + dx * middle, start[1] + dy * middle)
                for cell in self._cells_containing(*point):
                    earliest.setdefault(cell, time)
        return sorted(((time, cell) for cell, time in earliest.items()), key=lambda pair: pair[0])

    def _walk_segment(
        self, start: Tuple[float, float], end: Tuple[float, float]
    ) -> StepOutcome:
        """Apply the event rule to one straight move.

        Args:
            start: The point the step begins at.
            end: The point it would end at if nothing interrupted it.

        Returns:
            Where the agent actually ends up and what the path triggered.
        """
        standing_in = set(self._cells_containing(*start))
        entered_cue = False
        for time, group in groupby(self._crossed_cells(start, end), key=lambda pair: pair[0]):
            # Cells met at the same instant — the corner point of a diagonal move
            # touches up to four — must be judged together, walls first. Judging
            # them one at a time in dictionary order let a goal cell sharing that
            # corner with a wall end the move before the wall was ever checked,
            # which squeezed the agent through the corner and into the goal.
            cells = [cell for _, cell in group if cell not in standing_in]
            if not cells:
                continue
            if any(cell not in self._walkable for cell in cells):
                return StepOutcome(start, True, False, None)
            if any(cell == self._cue_cell for cell in cells):
                entered_cue = True
            reached = next((cell for cell in cells if cell in self._goal_cells), None)
            if reached is not None:
                stop = (
                    start[0] + (end[0] - start[0]) * time,
                    start[1] + (end[1] - start[1]) * time,
                )
                return StepOutcome(stop, False, entered_cue, reached)
        return StepOutcome(end, False, entered_cue, None)

    def _execute(self, state_array: np.ndarray, action: Any) -> StepOutcome:
        """Resolve ``action`` from ``state_array``. Implemented per movement model."""
        raise NotImplementedError

    # Core dynamics
    def _next_cue_phase(self, phase: float, outcome: StepOutcome) -> float:
        """Advance the cue's delivery phase across one transition.

        An emitting cue is consumed by whatever action follows it, including one
        refused by a wall that leaves the agent standing on the cue cell — that is
        what makes the cue single-use rather than re-readable by staying put.
        """
        if phase == CUE_EMITTING:
            return CUE_CONSUMED
        if phase == CUE_UNSEEN and outcome.entered_cue:
            return CUE_EMITTING
        return phase

    def _successor(self, state: Any, action: Any) -> np.ndarray:
        """The single deterministic successor of ``(state, action)``."""
        state_array = np.asarray(state, dtype=np.float64)
        if self.is_terminal(state_array):
            # Absorbing: a goal keeps its own state forever, so an over-long episode
            # cannot walk back out of a terminal state or be paid twice.
            return state_array.copy()
        outcome = self._execute(state_array, action)
        return np.array(
            [
                float(outcome.position[0]),
                float(outcome.position[1]),
                float(state_array[STATE_GOAL]),
                self._next_cue_phase(float(state_array[STATE_CUE_PHASE]), outcome),
            ],
            dtype=np.float64,
        )

    def is_terminal(self, state: Any) -> bool:
        """Whether ``state``'s position lies in either goal cell."""
        state_array = np.asarray(state, dtype=np.float64)
        return self._goal_cell_at(
            float(state_array[STATE_X]), float(state_array[STATE_Y])
        ) is not None

    def _goal_cell_at(self, x: float, y: float) -> Optional[Cell]:
        """The goal cell holding ``(x, y)``, or ``None``."""
        for cell in self._cells_containing(x, y):
            if cell in self._goal_cells:
                return cell
        return None

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> Any:
        """Sample successors. Transitions are deterministic, so all samples agree."""
        successor = self._successor(state, action)
        if n_samples == 1:
            return successor
        return np.repeat(successor[np.newaxis, :], n_samples, axis=0)

    def sample_next_state_batch(self, states: Any, action: Any) -> np.ndarray:
        """Batch transition. Returns float64, matching :meth:`sample_next_state`.

        Returning the caller's dtype here would hand a particle filter int64 particles
        from the batch path and float64 ones from the single path, and a belief mixing
        the two would silently truncate half of them.
        """
        states_array = np.asarray(states, dtype=np.float64)
        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)
        return np.stack([self._successor(row, action) for row in states_array], axis=0)

    def transition_log_probability(
        self, state: Any, action: Any, next_states: Any
    ) -> np.ndarray:
        """Log ``T(s' | s, a)``: 0 for the one successor, ``-inf`` everywhere else."""
        successor = self._successor(state, action)
        candidates = np.asarray(next_states, dtype=np.float64)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
        matches = np.all(np.isclose(candidates, successor[np.newaxis, :]), axis=1)
        return np.where(matches, 0.0, -np.inf)

    # Observation model
    def _observation_probs(self, next_state: Any) -> np.ndarray:
        """``P(o | s')`` over :data:`OBSERVATIONS`, in that order.

        Normalised for every reachable state and accuracy in ``[0.5, 1.0]``:
        an emitting cue puts all mass on the two cue readings; every other state
        puts all of it on ``"empty"``.
        """
        state_array = np.asarray(next_state, dtype=np.float64)
        probs = np.zeros(len(OBSERVATIONS), dtype=np.float64)
        if float(state_array[STATE_CUE_PHASE]) != CUE_EMITTING:
            probs[2] = 1.0
            return probs
        if float(state_array[STATE_GOAL]) == GOAL_LEFT:
            probs[0] = self.cue_accuracy
            probs[1] = 1.0 - self.cue_accuracy
        else:
            probs[1] = self.cue_accuracy
            probs[0] = 1.0 - self.cue_accuracy
        return probs

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        """Draw an observation from ``P(o | s')``. ``action`` does not enter it.

        Inlined rather than routed through a freshly built ``DiscreteDistribution``:
        this runs once per node expansion inside a tree search on a wall-clock
        budget. The draw is the distribution's own — one ``np.random.rand`` per
        sample, in order, against the cumulative probabilities — so the RNG stream is
        unchanged.
        """
        del action
        cumulative = np.cumsum(self._observation_probs(next_state))
        last = len(OBSERVATIONS) - 1
        if n_samples == 1:
            return OBSERVATIONS[min(int(np.searchsorted(cumulative, np.random.rand())), last)]
        indices = np.clip(np.searchsorted(cumulative, np.random.rand(n_samples)), 0, last)
        return [OBSERVATIONS[index] for index in indices]

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        """Log ``Z(o | s')`` for each observation in ``observations``."""
        del action
        probs = self._observation_probs(next_state)
        lookup = {name: float(probs[index]) for index, name in enumerate(OBSERVATIONS)}
        values = [observations] if isinstance(observations, str) else list(observations)
        out = np.full(len(values), -np.inf, dtype=np.float64)
        for index, value in enumerate(values):
            probability = lookup.get(value, 0.0)
            if probability > 0.0:
                out[index] = float(np.log(probability))
        return out

    def observation_log_probability_per_state(
        self, next_states: Any, action: Any, observation: Any
    ) -> np.ndarray:
        """Log ``Z(o | s')`` of one observation against many candidate states.

        Vectorised over the particles rather than looped: this is the particle
        filter's reweighting step, so it runs once per belief update over every
        particle, and the likelihood is a two-way choice numpy can express directly.
        """
        del action
        states_array = np.asarray(next_states, dtype=np.float64)
        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)

        emitting = states_array[:, STATE_CUE_PHASE] == CUE_EMITTING
        if observation == OBSERVATION_EMPTY:
            return np.where(emitting, -np.inf, 0.0)
        if observation not in (OBSERVATION_LEFT_CUE, OBSERVATION_RIGHT_CUE):
            return np.full(len(states_array), -np.inf, dtype=np.float64)

        names_left = observation == OBSERVATION_LEFT_CUE
        goal_is_left = states_array[:, STATE_GOAL] == GOAL_LEFT
        matches = goal_is_left if names_left else ~goal_is_left
        with np.errstate(divide="ignore"):
            log_accuracy = float(np.log(self.cue_accuracy))
            log_error = float(np.log(1.0 - self.cue_accuracy))
        return np.where(emitting, np.where(matches, log_accuracy, log_error), -np.inf)

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        """Observations are labels, so equality is string equality."""
        return observation1 == observation2

    def hash_observation(self, observation: Any) -> Hashable:
        """The label itself is already a hashable key."""
        return observation

    # Reward
    def _reward_from_successor(
        self, state_array: np.ndarray, successor: np.ndarray
    ) -> float:
        if self.is_terminal(state_array):
            return 0.0
        reached = self._goal_cell_at(float(successor[STATE_X]), float(successor[STATE_Y]))
        if reached is None:
            return float(-self.step_penalty)
        if reached == self.goal_cell(float(state_array[STATE_GOAL])):
            return float(self.goal_reward)
        return float(-self.wrong_goal_penalty)

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        """Immediate reward for ``(state, action)``.

        Transitions are deterministic, so ``next_state`` carries no information the
        environment cannot recompute; it is used when supplied only to stay
        consistent with the driver that threads it through.
        """
        state_array = np.asarray(state, dtype=np.float64)
        successor = (
            self._successor(state_array, action)
            if next_state is None
            else np.asarray(next_state, dtype=np.float64)
        )
        return self._reward_from_successor(state_array, successor)

    def reward_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: Any,
        next_states: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    ) -> np.ndarray:
        """Rewards for many states under one action."""
        states_array = np.asarray(states, dtype=np.float64)
        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)
        if next_states is None:
            successors = self.sample_next_state_batch(states_array, action)
        else:
            successors = np.asarray(next_states, dtype=np.float64)
            if successors.ndim == 1:
                successors = successors.reshape(1, -1)
        return np.array(
            [
                self._reward_from_successor(state_row, successor_row)
                for state_row, successor_row in zip(states_array, successors)
            ],
            dtype=np.float64,
        )

    # Initial distributions
    def initial_state_dist(self) -> Distribution:
        """Uniform over the two goal sides, at the start cell with the cue unseen."""
        return DiscreteDistribution(
            values=[
                create_maze_state(self._start_cell, GOAL_LEFT, CUE_UNSEEN),
                create_maze_state(self._start_cell, GOAL_RIGHT, CUE_UNSEEN),
            ],
            probs=np.array([0.5, 0.5], dtype=np.float64),
        )

    def initial_observation_dist(self) -> Distribution:
        """Always ``"empty"``.

        The cue is emitted by *crossing* the cue cell and the runner only observes
        after an action, so putting a reading here would hand the agent the answer
        before it had moved.
        """
        return DiscreteDistribution(
            values=[OBSERVATION_EMPTY], probs=np.array([1.0], dtype=np.float64)
        )

    # Metrics
    def step_info(self, state: Any, action: Any, next_state: Any) -> Dict[str, float]:
        """Report the per-step channels for one transition.

        Every channel is a pure function of the arguments — no draw, no state kept on
        ``self``. The wall-collision channel re-resolves the move through the same
        deterministic event rule used by the transition instead of comparing positions,
        because in the continuous variant a legal zero-length action also leaves the
        position unchanged and is not a collision.

        The episode-end channels are read off the *outcome* of the step: the realised
        successor when there is one, and ``state`` itself on the terminal bookkeeping
        call, where ``action`` and ``next_state`` are both ``None``. Reading them off
        ``state`` alone would report a timeout on the very step that entered a goal.

        Args:
            state: The state the step was taken from, or the final state.
            action: The action taken, or ``None`` on the terminal step.
            next_state: The realised successor, or ``None`` on the terminal step.

        Returns:
            One entry per :class:`MazeStepChannel`.
        """
        state_array = np.asarray(state, dtype=np.float64)
        outcome_state = (
            state_array if next_state is None else np.asarray(next_state, dtype=np.float64)
        )
        reached = self._goal_cell_at(
            float(outcome_state[STATE_X]), float(outcome_state[STATE_Y])
        )
        goal_cell = self.goal_cell(float(state_array[STATE_GOAL]))

        at_correct = float(reached is not None and reached == goal_cell)
        at_wrong = float(reached is not None and reached != goal_cell)
        collided = 0.0
        if action is not None and next_state is not None and not self.is_terminal(state_array):
            collided = float(self._execute(state_array, action).blocked)
        return {
            MazeStepChannel.CORRECT_GOAL.value: at_correct,
            MazeStepChannel.WRONG_GOAL.value: at_wrong,
            MazeStepChannel.ENDED_BY_GOAL.value: at_correct,
            MazeStepChannel.ENDED_BY_FAILURE.value: at_wrong,
            MazeStepChannel.ENDED_BY_TIMEOUT.value: float(at_correct == 0.0 and at_wrong == 0.0),
            MazeStepChannel.RECORDED_STEP.value: 1.0,
            MazeStepChannel.WALL_COLLISION.value: collided,
        }

    def get_metric_specs(self) -> List[StepInfoMetric]:
        """Declare the episode metrics derived from the per-step channels.

        Completion reduces with ``ANY`` because the task is to *reach* something:
        once the correct goal is entered the episode is over and no later step can
        undo it.

        Reaching the wrong goal is reported separately from timing out because the
        two call for opposite fixes — a planner that guesses at the far end needs a
        better belief, one that times out needs a longer horizon — and a completion
        rate alone cannot tell them apart.

        There is no severity metric. The only bad outcome is entering the wrong goal,
        which happens at most once per episode and always costs the same; a "worst
        moment" number would restate the count.

        Returns:
            One spec per metric, in the order ``compute_metrics`` reports them.
        """
        return [
            StepInfoMetric(
                name=MazeMetric.TASK_COMPLETION_RATE.value,
                channel=MazeStepChannel.CORRECT_GOAL.value,
                per_episode=EpisodeReduction.ANY,
            ),
            StepInfoMetric(
                name=MazeMetric.WRONG_GOAL_RATE.value,
                channel=MazeStepChannel.WRONG_GOAL.value,
                per_episode=EpisodeReduction.ANY,
            ),
            StepInfoMetric(
                name=MazeMetric.ENDED_BY_GOAL_RATE.value,
                channel=MazeStepChannel.ENDED_BY_GOAL.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=MazeMetric.ENDED_BY_FAILURE_RATE.value,
                channel=MazeStepChannel.ENDED_BY_FAILURE.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=MazeMetric.ENDED_BY_TIMEOUT_RATE.value,
                channel=MazeStepChannel.ENDED_BY_TIMEOUT.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=MazeMetric.AVERAGE_EPISODE_LENGTH.value,
                channel=MazeStepChannel.RECORDED_STEP.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=MazeMetric.AVERAGE_WALL_COLLISIONS.value,
                channel=MazeStepChannel.WALL_COLLISION.value,
                per_episode=EpisodeReduction.SUM,
            ),
        ]

    # Visualization
    @property
    def draws_cell_guides(self) -> bool:
        """Whether the renderer should rule cell guides inside the walkable cells.

        True where positions are cells and the guides say what a step is worth; false
        where positions are real and a grid would suggest quantization that is not
        present.
        """
        return True

    def cache_visualization(
        self, history: List[StepData], output_dir: Path, episode_index: int
    ) -> None:
        """Write a GIF of one episode, belief included.

        Args:
            history: The episode's step records.
            output_dir: Directory the ``.gif`` is written into.
            episode_index: Zero-based episode index, used to name the file.
        """
        # Imported here so the environment can be constructed and planned on without
        # matplotlib installed, matching how the other grid environments defer it.
        # pylint: disable=import-outside-toplevel
        from POMDPPlanners.environments.t_maze_pomdp.maze_visualizer import (
            MazeVisualizer,
        )

        MazeVisualizer(self).create_visualization(
            history, output_dir / f"agent_path_{episode_index}.gif"
        )


class DiscreteMazePOMDP(BaseMazePOMDP, DiscreteActionsEnvironment):
    """Maze POMDP with cell positions and four one-cell moves.

    Example:
        >>> env = DiscreteMazePOMDP(discount_factor=0.95, maze_width=11, maze_height=13)
        >>> env.start_cell
        (5, 1)
        >>> env.left_goal_cell, env.right_goal_cell
        ((1, 11), (9, 11))
    """

    def __init__(
        self,
        discount_factor: float = 0.95,
        maze_width: int = 7,
        maze_height: int = 9,
        maze_seed: int = 0,
        loop_fraction: float = 0.15,
        cue_accuracy: float = 0.9,
        goal_reward: float = 10.0,
        wrong_goal_penalty: float = 10.0,
        step_penalty: float = 1.0,
        name: str = "DiscreteMazePOMDP",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ) -> None:
        """Initialize the discrete maze.

        Args:
            discount_factor: Discount factor for future rewards. Defaults to 0.95.
            maze_width: Grid columns. Odd, at least 7. Defaults to 7.
            maze_height: Grid rows. Odd, at least 9. Defaults to 9.
            maze_seed: Seed the map is carved from. The generator owns a private
                ``numpy.random.Generator``, so changing this does not move the global
                random stream a simulation is reproducing. Defaults to 0.
            loop_fraction: Fraction of the walls the carving left standing between
                two maze cells that are knocked out, giving alternate routes. 0.0 is
                a perfect maze. Defaults to 0.15.
            cue_accuracy: Probability the cue names the true goal side, in
                ``[0.5, 1.0]``. 1.0 is allowed and turns the task into pure memory.
                Defaults to 0.9.
            goal_reward: Reward for entering the correct goal. Defaults to 10.0.
            wrong_goal_penalty: Magnitude of the penalty for entering the other goal.
                Defaults to 10.0.
            step_penalty: Magnitude of the per-action cost, wall collisions included.
                Defaults to 1.0.
            name: Environment name. Defaults to ``"DiscreteMazePOMDP"``.
            output_dir: Optional directory for logging output.
            debug: Enable debug logging.
            use_queue_logger: Whether to use queue-based logging.
        """
        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
            ),
            maze_width=maze_width,
            maze_height=maze_height,
            maze_seed=maze_seed,
            loop_fraction=loop_fraction,
            cue_accuracy=cue_accuracy,
            goal_reward=goal_reward,
            wrong_goal_penalty=wrong_goal_penalty,
            step_penalty=step_penalty,
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )
        # One lookup per (cell, action), built once. Private, so it stays out of
        # ``config_id``: it is a pure function of the geometry, which is itself a
        # pure function of the public parameters.
        self._transitions: Dict[Cell, Dict[str, Cell]] = {
            cell: {
                action: (cell[0] + offset[0], cell[1] + offset[1])
                if (cell[0] + offset[0], cell[1] + offset[1]) in self._walkable
                else cell
                for action, offset in ACTION_OFFSETS.items()
            }
            for cell in self._walkable
        }

    def get_actions(self) -> List[str]:
        """The four movement actions, in a fixed order."""
        return list(ACTIONS)

    def hash_action(self, action: Any) -> Hashable:
        """The action label itself is already a hashable key."""
        return action

    def _execute(self, state_array: np.ndarray, action: Any) -> StepOutcome:
        """Resolve one cell step.

        A cell step crosses exactly the cell it lands in, so the shared event rule
        collapses to a lookup: a wall refuses the move, a goal ends the step, the cue
        cell arms the cue.
        """
        cell = (int(round(state_array[STATE_X])), int(round(state_array[STATE_Y])))
        landed = self._transitions[cell][action]
        if landed == cell:
            return StepOutcome((float(cell[0]), float(cell[1])), True, False, None)
        return StepOutcome(
            (float(landed[0]), float(landed[1])),
            False,
            landed == self._cue_cell,
            landed if landed in self._goal_cells else None,
        )


class ContinuousMazePOMDP(BaseMazePOMDP):
    """Maze POMDP with real positions and a bounded displacement vector per step.

    The map, the cue, the goals, the reward and the metrics are the discrete
    variant's. Only how the agent moves differs: a step is a straight segment, and it
    is legal only if the *whole* segment stays inside the walkable region.

    Attributes:
        max_step_size: The longest displacement one action may cover.

    Example:
        >>> import numpy as np
        >>> env = ContinuousMazePOMDP(discount_factor=0.95)
        >>> env.start_cell
        (3, 1)
        >>> state = env.initial_state_dist().sample()[0]
        >>> float(env.sample_next_state(state, np.array([0.0, 0.4]))[1])
        1.4
    """

    def __init__(
        self,
        discount_factor: float = 0.95,
        maze_width: int = 7,
        maze_height: int = 9,
        maze_seed: int = 0,
        loop_fraction: float = 0.15,
        max_step_size: float = 1.0,
        cue_accuracy: float = 0.9,
        goal_reward: float = 10.0,
        wrong_goal_penalty: float = 10.0,
        step_penalty: float = 1.0,
        name: str = "ContinuousMazePOMDP",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ) -> None:
        """Initialize the continuous maze.

        Args:
            discount_factor: Discount factor for future rewards. Defaults to 0.95.
            maze_width: Grid columns. Odd, at least 7. Defaults to 7.
            maze_height: Grid rows. Odd, at least 9. Defaults to 9.
            maze_seed: Seed the map is carved from. Defaults to 0.
            loop_fraction: Fraction of leftover interior walls knocked out for loops.
                Defaults to 0.15.
            max_step_size: The longest displacement one action may cover, in grid
                cells. An action longer than this is scaled down to it rather than
                rejected, so no sampler can produce an illegal action. The default
                1.0 is one discrete step, which makes the two variants comparable.
                Must be positive.
            cue_accuracy: Probability the cue names the true goal side, in
                ``[0.5, 1.0]``. Defaults to 0.9.
            goal_reward: Reward for entering the correct goal. Defaults to 10.0.
            wrong_goal_penalty: Magnitude of the penalty for the other goal.
                Defaults to 10.0.
            step_penalty: Magnitude of the per-action cost, refused moves included.
                Defaults to 1.0.
            name: Environment name. Defaults to ``"ContinuousMazePOMDP"``.
            output_dir: Optional directory for logging output.
            debug: Enable debug logging.
            use_queue_logger: Whether to use queue-based logging.

        Raises:
            ValueError: If ``max_step_size`` is not positive.
        """
        if not np.isfinite(max_step_size) or max_step_size <= 0.0:
            raise ValueError(f"max_step_size must be positive, got {max_step_size}.")
        self.max_step_size = float(max_step_size)
        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=SpaceInfo(
                action_space=SpaceType.CONTINUOUS, observation_space=SpaceType.DISCRETE
            ),
            maze_width=maze_width,
            maze_height=maze_height,
            maze_seed=maze_seed,
            loop_fraction=loop_fraction,
            cue_accuracy=cue_accuracy,
            goal_reward=goal_reward,
            wrong_goal_penalty=wrong_goal_penalty,
            step_penalty=step_penalty,
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

    @property
    def draws_cell_guides(self) -> bool:
        """False: positions are real here, and a grid would suggest a quantization."""
        return False

    def hash_action(self, action: Any) -> Hashable:
        """Bytes of the action vector, which match ``np.array_equal`` for one shape."""
        return np.ascontiguousarray(action, dtype=np.float64).tobytes()

    def clip_action(self, action: Any) -> np.ndarray:
        """The displacement ``action`` actually applies, at most ``max_step_size`` long.

        Scaling rather than rejecting keeps the action space closed under any sampler:
        a planner drawing from a disc of the wrong radius still produces legal moves,
        and the direction it chose is preserved.

        Args:
            action: A 2-vector displacement.

        Returns:
            The same direction, with magnitude capped at ``max_step_size``.

        Raises:
            ValueError: If ``action`` is not a 2-vector.
        """
        vector = np.asarray(action, dtype=np.float64).reshape(-1)
        if vector.shape != (2,):
            raise ValueError(f"action must be a 2-vector, got shape {np.shape(action)}.")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"action must contain only finite values, got {vector}.")
        magnitude = float(np.linalg.norm(vector))
        if magnitude > self.max_step_size:
            return vector * (self.max_step_size / magnitude)
        return vector

    def _execute(self, state_array: np.ndarray, action: Any) -> StepOutcome:
        """Resolve one displacement against the whole swept segment."""
        displacement = self.clip_action(action)
        start = (float(state_array[STATE_X]), float(state_array[STATE_Y]))
        end = (start[0] + float(displacement[0]), start[1] + float(displacement[1]))
        return self._walk_segment(start, end)


def _boundary_span(value: float) -> Tuple[int, ...]:
    """The cell indices along one axis whose closed unit interval contains ``value``.

    One index when the value is inside a cell, two when it sits on the boundary
    between them. :data:`_CELL_TOLERANCE` widens each interval so a coordinate that
    floating point put a nanometre short of a boundary still counts as on it.
    """
    lower = math.ceil(value - 0.5 - _CELL_TOLERANCE)
    upper = math.floor(value + 0.5 + _CELL_TOLERANCE)
    return tuple(range(lower, upper + 1))


__all__ = [
    "ACTIONS",
    "BaseMazePOMDP",
    "CUE_CONSUMED",
    "CUE_EMITTING",
    "CUE_UNSEEN",
    "ContinuousMazePOMDP",
    "DiscreteMazePOMDP",
    "GOAL_LEFT",
    "GOAL_RIGHT",
    "MazeMetric",
    "MazeStepChannel",
    "OBSERVATIONS",
    "OBSERVATION_EMPTY",
    "OBSERVATION_LEFT_CUE",
    "OBSERVATION_RIGHT_CUE",
    "STATE_CUE_PHASE",
    "STATE_GOAL",
    "STATE_WIDTH",
    "STATE_X",
    "STATE_Y",
    "StepOutcome",
    "create_maze_state",
]
