# SPDX-License-Identifier: MIT

"""Vectorized particle belief updaters for the generated-maze POMDPs.

The maze hides one bit -- which goal pays -- behind a deterministic map, so a
belief update is cheap per particle and the Python loop around it is the whole
cost. These updaters do the loop's work in NumPy instead: one table lookup per
particle for the discrete map, one swept-segment test per particle for the
continuous one, and a two-way ``np.where`` for the cue likelihood.

Both updaters reproduce the environment's own event rule rather than
approximating it, because the belief is what the planner searches against and a
belief that walks through walls the world refuses would be searching a
different maze. The discrete rule collapses to a lookup: the environment itself
precomputes one successor per (cell, action). The continuous rule is the
segment test spelled out as array algebra -- see
:meth:`ContinuousMazeVectorizedUpdater.batch_transition`.

Observations are the three labels ``"left_cue"``, ``"right_cue"`` and
``"empty"``; the updaters take them integer-encoded, and
:class:`~POMDPPlanners.environments.maze_pomdp.maze_pomdp_beliefs.maze_belief_factory.MazeVectorizedWeightedParticleBelief`
does the encoding.

Classes:
    BaseMazeVectorizedUpdater: The shared cue phase, cue likelihood and identity.
    DiscreteMazeVectorizedUpdater: One-cell moves, by lookup table.
    ContinuousMazeVectorizedUpdater: Swept-segment moves, by slab intersection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.environments.maze_pomdp.maze_pomdp import (
    ACTIONS,
    CUE_CONSUMED,
    CUE_EMITTING,
    CUE_UNSEEN,
    GOAL_LEFT,
    OBSERVATIONS,
    STATE_CUE_PHASE,
    STATE_GOAL,
    STATE_X,
    STATE_Y,
)
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.environments.maze_pomdp.maze_pomdp import (
        BaseMazePOMDP,
        ContinuousMazePOMDP,
        DiscreteMazePOMDP,
    )

# Observation encoding, in the environment's own ``OBSERVATIONS`` order.
OBS_LEFT_CUE = 0
OBS_RIGHT_CUE = 1
OBS_EMPTY = 2

# The environment's cell tolerance, mirrored rather than imported: a cell owns the
# closed unit square around it, widened by this much so a coordinate that floating
# point put a hair short of a boundary still counts as on it.
_CELL_TOLERANCE = 1e-9


class BaseMazeVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Shared map arrays, cue bookkeeping and observation likelihood.

    The subclasses differ only in how a step moves a particle. Everything that
    follows from the move -- whether the cue armed, how its phase advances, what
    a reading says about the hidden goal side -- is the same in both.

    Attributes:
        width: Grid columns.
        height: Grid rows.
        cue_accuracy: Probability the cue names the true goal side.
        cue_cell: The one cell whose crossing arms the cue.
        goal_cells: The two terminal cells, as a ``(2, 2)`` int array.
        walkable: ``(width, height)`` bool array, indexed ``[x, y]``.
    """

    def __init__(
        self,
        width: int,
        height: int,
        walkable: np.ndarray,
        cue_cell: Tuple[int, int],
        goal_cells: Tuple[Tuple[int, int], Tuple[int, int]],
        cue_accuracy: float,
    ) -> None:
        """Initialize the shared part of a maze updater.

        Args:
            width: Grid columns.
            height: Grid rows.
            walkable: ``(width, height)`` bool array, indexed ``[x, y]``.
            cue_cell: The cue cell as ``(x, y)``.
            goal_cells: The two goal cells as ``((x, y), (x, y))``.
            cue_accuracy: Probability the cue names the true goal side.
        """
        self.width = int(width)
        self.height = int(height)
        self.walkable = np.asarray(walkable, dtype=bool)
        self.cue_cell = (int(cue_cell[0]), int(cue_cell[1]))
        self.goal_cells = np.asarray(goal_cells, dtype=np.int64)
        self.cue_accuracy = float(cue_accuracy)

        self._goal_mask = np.zeros((self.width, self.height), dtype=bool)
        self._goal_mask[self.goal_cells[:, 0], self.goal_cells[:, 1]] = True

        with np.errstate(divide="ignore"):
            self._log_accuracy = float(np.log(self.cue_accuracy))
            self._log_error = float(np.log(1.0 - self.cue_accuracy))

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: Any,
        observation: Any,
    ) -> np.ndarray:
        """Log ``Z(o | s')`` for every particle at once.

        Args:
            next_particles: ``(N, 4)`` transitioned particles.
            action: Unused; the cue does not depend on what was done.
            observation: The integer-encoded reading.

        Returns:
            ``(N,)`` log-likelihoods.
        """
        del action
        particles = np.asarray(next_particles, dtype=np.float64)
        code = int(np.asarray(observation).ravel()[0])

        emitting = particles[:, STATE_CUE_PHASE] == CUE_EMITTING
        if code == OBS_EMPTY:
            # Only an emitting cue can produce anything but silence, so silence
            # rules those particles out entirely rather than merely disfavouring
            # them.
            return np.where(emitting, -np.inf, 0.0)
        if code not in (OBS_LEFT_CUE, OBS_RIGHT_CUE):
            return np.full(particles.shape[0], -np.inf, dtype=np.float64)

        goal_is_left = particles[:, STATE_GOAL] == GOAL_LEFT
        matches = goal_is_left if code == OBS_LEFT_CUE else ~goal_is_left
        return np.where(
            emitting, np.where(matches, self._log_accuracy, self._log_error), -np.inf
        )

    @property
    def config_id(self) -> str:
        """Return a deterministic identifier for this updater configuration."""
        return config_to_id(self._config_dict())

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _config_dict(self) -> Dict[str, Any]:
        """The identity every maze updater carries, before its own additions."""
        return {
            "class": type(self).__name__,
            "width": self.width,
            "height": self.height,
            # The walkable region is what a wrong map would differ in, and it is
            # not implied by anything else stored here: two seeds that produce the
            # same size produce different corridors.
            "walkable": np.flatnonzero(self.walkable.ravel()).tolist(),
            "cue_cell": list(self.cue_cell),
            "goal_cells": self.goal_cells.tolist(),
            "cue_accuracy": self.cue_accuracy,
        }

    def _advance_cue_phase(self, phase: np.ndarray, entered_cue: np.ndarray) -> np.ndarray:
        """Advance the cue phase for every particle.

        An emitting cue is consumed by whatever action follows it, including one a
        wall refused, which is what makes the cue single-use rather than
        re-readable by standing still.

        Args:
            phase: ``(N,)`` current cue phases.
            entered_cue: ``(N,)`` whether this step crossed into the cue cell.

        Returns:
            ``(N,)`` next cue phases.
        """
        armed = (phase == CUE_UNSEEN) & entered_cue
        return np.where(phase == CUE_EMITTING, CUE_CONSUMED, np.where(armed, CUE_EMITTING, phase))

    def _terminal_mask(self, particles: np.ndarray) -> np.ndarray:
        """Which particles already stand in a goal cell, and so never move again.

        A position on a cell boundary belongs to both cells that share it, so the
        test asks whether *any* cell containing the point is a goal rather than
        rounding to one of them. Rounding is the wrong question at exactly the
        positions a continuous step produces most often: it stops on the goal's
        edge.
        """
        out = np.zeros(particles.shape[0], dtype=bool)
        for x in _boundary_span_batch(particles[:, STATE_X]):
            for y in _boundary_span_batch(particles[:, STATE_Y]):
                inside = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
                hit = np.zeros(particles.shape[0], dtype=bool)
                hit[inside] = self._goal_mask[x[inside], y[inside]]
                out |= hit
        return out

    def _assemble(
        self,
        particles: np.ndarray,
        position: np.ndarray,
        entered_cue: np.ndarray,
        terminal: np.ndarray,
    ) -> np.ndarray:
        """Build the successor array, leaving terminal particles untouched.

        Args:
            particles: ``(N, 4)`` current particles.
            position: ``(N, 2)`` positions the step reached.
            entered_cue: ``(N,)`` whether the step crossed into the cue cell.
            terminal: ``(N,)`` particles that were already in a goal.

        Returns:
            ``(N, 4)`` successors.
        """
        out = np.empty_like(particles)
        out[:, STATE_X] = position[:, 0]
        out[:, STATE_Y] = position[:, 1]
        out[:, STATE_GOAL] = particles[:, STATE_GOAL]
        out[:, STATE_CUE_PHASE] = self._advance_cue_phase(
            particles[:, STATE_CUE_PHASE], entered_cue
        )
        # Absorbing: a goal keeps its own state forever, so an over-long rollout
        # cannot walk a particle back out of a terminal state.
        out[terminal] = particles[terminal]
        return out

    @staticmethod
    def _empty_like_particles(particles: np.ndarray) -> np.ndarray:
        return np.asarray(particles, dtype=np.float64).reshape(-1, 4)


class DiscreteMazeVectorizedUpdater(BaseMazeVectorizedUpdater):
    """Batched updater for :class:`DiscreteMazePOMDP`.

    A one-cell step crosses exactly the cell it lands in, so the environment's
    event rule is a lookup and the batch is one fancy-index into it.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.maze_pomdp.maze_pomdp import DiscreteMazePOMDP
        >>> env = DiscreteMazePOMDP(discount_factor=0.95)
        >>> updater = DiscreteMazeVectorizedUpdater.from_environment(env)
        >>> particles = np.stack(env.initial_state_dist().sample(n_samples=4))
        >>> updater.batch_transition(particles, "up").shape
        (4, 4)
    """

    def __init__(
        self,
        width: int,
        height: int,
        walkable: np.ndarray,
        cue_cell: Tuple[int, int],
        goal_cells: Tuple[Tuple[int, int], Tuple[int, int]],
        cue_accuracy: float,
    ) -> None:
        super().__init__(width, height, walkable, cue_cell, goal_cells, cue_accuracy)
        self._actions = list(ACTIONS)
        self._next_x, self._next_y = self._build_transition_tables()

    @classmethod
    def from_environment(cls, env: "DiscreteMazePOMDP") -> "DiscreteMazeVectorizedUpdater":
        """Construct an updater from a :class:`DiscreteMazePOMDP` instance."""
        return cls(
            width=env.maze_width,
            height=env.maze_height,
            walkable=_walkable_array(env),
            cue_cell=env.cue_cell,
            goal_cells=(env.left_goal_cell, env.right_goal_cell),
            cue_accuracy=env.cue_accuracy,
        )

    def batch_transition(self, particles: np.ndarray, action: Any) -> np.ndarray:
        """Move every particle one cell.

        Args:
            particles: ``(N, 4)`` particles.
            action: One of ``"up"``, ``"down"``, ``"left"``, ``"right"``.

        Returns:
            ``(N, 4)`` successors.
        """
        states = self._empty_like_particles(particles)
        if states.shape[0] == 0:
            return states
        action_index = self._actions.index(action if isinstance(action, str) else str(action))

        x = np.clip(np.rint(states[:, STATE_X]).astype(np.int64), 0, self.width - 1)
        y = np.clip(np.rint(states[:, STATE_Y]).astype(np.int64), 0, self.height - 1)
        landed_x = self._next_x[x, y, action_index]
        landed_y = self._next_y[x, y, action_index]

        blocked = (landed_x == x) & (landed_y == y)
        entered_cue = (
            (landed_x == self.cue_cell[0]) & (landed_y == self.cue_cell[1]) & ~blocked
        )
        position = np.stack([landed_x, landed_y], axis=1).astype(np.float64)
        return self._assemble(states, position, entered_cue, self._terminal_mask(states))

    def _build_transition_tables(self) -> Tuple[np.ndarray, np.ndarray]:
        """One successor cell per (cell, action), as two ``(W, H, 4)`` int arrays.

        A move into a wall, or off the grid, leaves the cell unchanged -- the same
        rule the environment applies, and the same one a wall cell gets, which
        keeps an out-of-region particle (nothing produces one, but nothing rules
        one out either) from indexing outside the table.
        """
        xs = np.arange(self.width)[:, None]
        ys = np.arange(self.height)[None, :]
        next_x = np.empty((self.width, self.height, len(self._actions)), dtype=np.int64)
        next_y = np.empty_like(next_x)
        offsets = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
        for index, action in enumerate(self._actions):
            dx, dy = offsets[action]
            target_x = np.broadcast_to(xs + dx, (self.width, self.height))
            target_y = np.broadcast_to(ys + dy, (self.width, self.height))
            inside = (
                (target_x >= 0)
                & (target_x < self.width)
                & (target_y >= 0)
                & (target_y < self.height)
            )
            legal = np.zeros((self.width, self.height), dtype=bool)
            legal[inside] = self.walkable[target_x[inside], target_y[inside]]
            next_x[:, :, index] = np.where(legal, target_x, np.broadcast_to(xs, legal.shape))
            next_y[:, :, index] = np.where(legal, target_y, np.broadcast_to(ys, legal.shape))
        return next_x, next_y


class ContinuousMazeVectorizedUpdater(BaseMazeVectorizedUpdater):
    """Batched updater for :class:`ContinuousMazePOMDP`.

    Positions agree with the environment's own to within its cell tolerance
    rather than exactly. The environment stops a step at the cut time where the
    segment crosses a cell's true boundary; this updater stops it at the
    tolerance-widened boundary the same code uses to decide membership, which is
    at most a nanometre of grid earlier. Both points are inside the goal square
    under the same tolerance, so the step is terminal either way.

    Attributes:
        max_step_size: The longest displacement one action may cover.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.maze_pomdp.maze_pomdp import ContinuousMazePOMDP
        >>> env = ContinuousMazePOMDP(discount_factor=0.95)
        >>> updater = ContinuousMazeVectorizedUpdater.from_environment(env)
        >>> particles = np.stack(env.initial_state_dist().sample(n_samples=4))
        >>> updater.batch_transition(particles, np.array([0.0, 0.4])).shape
        (4, 4)
    """

    def __init__(
        self,
        width: int,
        height: int,
        walkable: np.ndarray,
        cue_cell: Tuple[int, int],
        goal_cells: Tuple[Tuple[int, int], Tuple[int, int]],
        cue_accuracy: float,
        max_step_size: float,
    ) -> None:
        super().__init__(width, height, walkable, cue_cell, goal_cells, cue_accuracy)
        self.max_step_size = float(max_step_size)

    @classmethod
    def from_environment(cls, env: "ContinuousMazePOMDP") -> "ContinuousMazeVectorizedUpdater":
        """Construct an updater from a :class:`ContinuousMazePOMDP` instance."""
        return cls(
            width=env.maze_width,
            height=env.maze_height,
            walkable=_walkable_array(env),
            cue_cell=env.cue_cell,
            goal_cells=(env.left_goal_cell, env.right_goal_cell),
            cue_accuracy=env.cue_accuracy,
            max_step_size=env.max_step_size,
        )

    def batch_transition(self, particles: np.ndarray, action: Any) -> np.ndarray:
        """Sweep every particle's segment against the map.

        The environment walks the cells a segment meets in the order it meets
        them: the first wall refuses the whole move, the first goal stops it
        there, and the cue cell arms the cue on the way past. That order is
        recovered here without sorting anything. For each candidate cell the
        segment's entry time is the largest of its two slab entry times, so three
        reductions -- the earliest wall, the earliest goal, the earliest cue --
        decide the step. Walls are judged before goals inside one group of
        simultaneous cells, which is why the wall comparison is inclusive.

        Args:
            particles: ``(N, 4)`` particles.
            action: A 2-vector displacement, scaled down to ``max_step_size`` if
                longer, exactly as the environment scales it.

        Returns:
            ``(N, 4)`` successors.
        """
        states = self._empty_like_particles(particles)
        if states.shape[0] == 0:
            return states
        delta = self._clip_action(action)
        start = states[:, (STATE_X, STATE_Y)]
        end = start + delta

        wall_time, goal_time, cue_time = self._event_times(start, delta)

        # ``isfinite`` first: a path that meets neither a wall nor a goal leaves
        # both times at infinity, and ``inf <= inf`` would read an open corridor
        # as a wall.
        blocked = np.isfinite(wall_time) & (wall_time <= goal_time + _CELL_TOLERANCE)
        stopped = np.isfinite(goal_time) & ~blocked

        position = np.where(blocked[:, None], start, end)
        # Zeroed where no goal was met: an infinite stop time times a zero
        # displacement component is a NaN, and it would reach the output through
        # the ``where`` below on a path that never took it.
        stop_time = np.where(stopped, goal_time, 0.0)
        stop_point = start + stop_time[:, None] * delta
        position = np.where(stopped[:, None], stop_point, position)

        entered_cue = np.where(
            blocked,
            False,
            np.where(stopped, cue_time <= goal_time + _CELL_TOLERANCE, np.isfinite(cue_time)),
        )
        return self._assemble(states, position, entered_cue, self._terminal_mask(states))

    @property
    def config_id(self) -> str:
        """Return a deterministic identifier for this updater configuration."""
        config = self._config_dict()
        config["max_step_size"] = self.max_step_size
        return config_to_id(config)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clip_action(self, action: Any) -> np.ndarray:
        vector = np.asarray(action, dtype=np.float64).reshape(-1)
        if vector.shape != (2,):
            raise ValueError(f"action must be a 2-vector, got shape {np.shape(action)}.")
        magnitude = float(np.linalg.norm(vector))
        if magnitude > self.max_step_size:
            return vector * (self.max_step_size / magnitude)
        return vector

    def _candidate_offsets(self, delta: np.ndarray) -> np.ndarray:
        """Cell offsets from a particle's own cell that its segment could reach.

        A segment of length at most ``max_step_size`` cannot leave the box of
        cells this far away, so the enumeration is exhaustive rather than a
        heuristic. One extra ring covers the half-cell the start point may sit
        off-centre by.
        """
        reach_x = int(np.ceil(abs(delta[0]) + 0.5)) + 1
        reach_y = int(np.ceil(abs(delta[1]) + 0.5)) + 1
        grid_x, grid_y = np.meshgrid(
            np.arange(-reach_x, reach_x + 1), np.arange(-reach_y, reach_y + 1), indexing="ij"
        )
        return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    def _event_times(
        self, start: np.ndarray, delta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Earliest time the segment meets a wall, a goal and the cue cell.

        Cells the particle already stands in are excluded: the environment
        ignores them, which is what lets an agent stand on the cue cell without
        re-reading it and step out of a cell that is about to be judged again.

        Args:
            start: ``(N, 2)`` segment starts.
            delta: The shared ``(2,)`` displacement.

        Returns:
            ``(wall_time, goal_time, cue_time)``, each ``(N,)`` and ``inf`` where
            the segment meets no such cell.
        """
        offsets = self._candidate_offsets(delta)
        base = np.rint(start).astype(np.int64)
        cells = base[:, None, :] + offsets[None, :, :]  # (N, K, 2)

        enter, exit_time = _slab_times(start, delta, cells)
        met = enter <= exit_time + _CELL_TOLERANCE
        standing = met & (enter <= _CELL_TOLERANCE)
        candidate = met & ~standing

        inside = (
            (cells[:, :, 0] >= 0)
            & (cells[:, :, 0] < self.width)
            & (cells[:, :, 1] >= 0)
            & (cells[:, :, 1] < self.height)
        )
        flat_x = np.clip(cells[:, :, 0], 0, self.width - 1)
        flat_y = np.clip(cells[:, :, 1], 0, self.height - 1)
        is_wall = ~(inside & self.walkable[flat_x, flat_y])
        is_goal = inside & self._goal_mask[flat_x, flat_y]
        is_cue = (cells[:, :, 0] == self.cue_cell[0]) & (cells[:, :, 1] == self.cue_cell[1])

        def earliest(selector: np.ndarray) -> np.ndarray:
            times = np.where(candidate & selector, enter, np.inf)
            return times.min(axis=1)

        return earliest(is_wall), earliest(is_goal), earliest(is_cue)


def _slab_times(
    start: np.ndarray, delta: np.ndarray, cells: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Entry and exit times of each segment through each cell's closed square.

    Args:
        start: ``(N, 2)`` segment starts.
        delta: The shared ``(2,)`` displacement.
        cells: ``(N, K, 2)`` candidate cells.

    Returns:
        ``(enter, exit)``, both ``(N, K)`` and clipped to ``[0, 1]``. A cell the
        segment misses comes back with ``enter > exit``.
    """
    enter = np.zeros(cells.shape[:2], dtype=np.float64)
    exit_time = np.ones(cells.shape[:2], dtype=np.float64)
    for axis in (0, 1):
        low = cells[:, :, axis] - 0.5 - _CELL_TOLERANCE
        high = cells[:, :, axis] + 0.5 + _CELL_TOLERANCE
        origin = start[:, axis][:, None]
        if delta[axis] == 0.0:
            # No motion on this axis: the cell is either in the slab for the whole
            # segment or never, and "never" has to survive the min/max below.
            misses = (origin < low) | (origin > high)
            enter = np.where(misses, np.inf, enter)
            exit_time = np.where(misses, -np.inf, exit_time)
            continue
        first = (low - origin) / delta[axis]
        second = (high - origin) / delta[axis]
        enter = np.maximum(enter, np.minimum(first, second))
        exit_time = np.minimum(exit_time, np.maximum(first, second))
    return enter, exit_time


def _boundary_span_batch(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The one or two cell indices along one axis whose closed interval holds each value.

    The environment's ``_boundary_span`` returns a variable-length tuple per
    point; the batched form returns both ends of that range, which are equal when
    the point is strictly inside a cell. Callers test both, so the duplicate is
    harmless.

    Args:
        values: ``(N,)`` coordinates along one axis.

    Returns:
        ``(lower, upper)`` index arrays, each ``(N,)``.
    """
    lower = np.ceil(values - 0.5 - _CELL_TOLERANCE).astype(np.int64)
    upper = np.floor(values + 0.5 + _CELL_TOLERANCE).astype(np.int64)
    return lower, upper


def _walkable_array(env: "BaseMazePOMDP") -> np.ndarray:
    """The environment's walkable cell set as a ``(width, height)`` bool array."""
    walkable = np.zeros((env.maze_width, env.maze_height), dtype=bool)
    for cell in env.walkable_cells:
        walkable[int(cell[0]), int(cell[1])] = True
    return walkable


__all__ = [
    "OBSERVATIONS",
    "OBS_EMPTY",
    "OBS_LEFT_CUE",
    "OBS_RIGHT_CUE",
    "BaseMazeVectorizedUpdater",
    "ContinuousMazeVectorizedUpdater",
    "DiscreteMazeVectorizedUpdater",
]
