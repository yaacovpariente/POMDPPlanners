# SPDX-License-Identifier: MIT

"""Vectorized particle belief for the Snake POMDP.

The batched twin of
:class:`~POMDPPlanners.environments.snake_pomdp.snake_belief.SnakeBelief`. The
exact belief carries a categorical over the one hidden cell; this one carries
particles and updates them through the
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
interface, so a vectorized planner can hold it.

Both halves of the update are the environment's own model, spelled out as array
algebra: :meth:`SnakeVectorizedUpdater.batch_transition` turns, moves, eats,
grows and respawns every particle at once, and
:meth:`SnakeVectorizedUpdater.batch_observation_log_likelihood` scores the body,
the sighting and the scent the way
:meth:`~POMDPPlanners.environments.snake_pomdp.snake_pomdp.SnakePOMDP.observation_log_probability`
does.

One thing the generic filter cannot survive here is kept: the vision window has
no false positives, so a sighting rules out every food cell but one, and a
particle set holding none of them would have every weight floored and resample
impossible cells back over the whole set. When that happens -- every particle
ruled out by the reading -- :meth:`SnakeVectorizedWeightedParticleBelief.update`
redraws the food from the cells the reading itself allows instead of from the
dead particles. A sighting names one cell, so the redraw is exact; with no
sighting the free cells are drawn in proportion to the scent likelihood, which
is the posterior implied by that step alone.

Classes:
    SnakeVectorizedUpdater: Batched transition and observation likelihood.
    SnakeVectorizedWeightedParticleBelief: The belief, with the redraw.

Functions:
    create_snake_belief: Factory used by the top-level belief factory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

from POMDPPlanners.core.belief.belief_utils import get_initial_belief
from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.snake_pomdp.snake_pomdp import (
    BODY_OFFSET,
    DIRECTIONS,
    EMPTY,
    FOOD_COL_INDEX,
    FOOD_ROW_INDEX,
    LENGTH_INDEX,
    OBSERVATION_LIVE,
    STATUS_INDEX,
    STEPS_SINCE_FOOD_INDEX,
    SnakeAction,
    SnakeQuadrant,
    SnakeTermination,
)
from POMDPPlanners.utils.belief_factory import BeliefType
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.core.belief.base_belief import Belief
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.environments.snake_pomdp.snake_pomdp import SnakePOMDP

#: Headings as an array, in the environment's clockwise order.
_DIRECTIONS = np.asarray(DIRECTIONS, dtype=np.int64)

#: How each action rotates the heading index in that clockwise order.
_ACTION_ROTATION = {
    int(SnakeAction.TURN_LEFT): -1,
    int(SnakeAction.GO_STRAIGHT): 0,
    int(SnakeAction.TURN_RIGHT): 1,
}

#: Quadrant membership as two masks over :class:`SnakeQuadrant`, so the
#: compatible set for an offset is one boolean ``and`` instead of a lookup.
_IS_NORTH = np.array([True, True, False, False])
_IS_EAST = np.array([True, False, True, False])


class SnakeVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Batched transition and observation likelihood for the Snake POMDP.

    Attributes:
        grid_size: Side length of the playable grid.
        target_length: Length that wins the episode, and the body block's width.
        starvation_limit: Steps without food that end the episode.
        window_radius: Half-width of the vision window.
        detection_probability: Chance the window reports food inside it.
        scent_accuracy: Chance the scent names a compatible quadrant.
    """

    def __init__(
        self,
        grid_size: int,
        target_length: int,
        starvation_limit: int,
        window_radius: int,
        detection_probability: float,
        scent_accuracy: float,
    ) -> None:
        """Initialize the updater.

        Args:
            grid_size: Side length of the playable grid.
            target_length: Length that wins the episode.
            starvation_limit: Steps without food that end the episode.
            window_radius: Half-width of the vision window.
            detection_probability: Chance the window reports food inside it.
            scent_accuracy: Chance the scent names a compatible quadrant.
        """
        self.grid_size = int(grid_size)
        self.target_length = int(target_length)
        self.starvation_limit = int(starvation_limit)
        self.window_radius = int(window_radius)
        self.detection_probability = float(detection_probability)
        self.scent_accuracy = float(scent_accuracy)
        self.num_cells = self.grid_size * self.grid_size
        self.state_width = BODY_OFFSET + 2 * self.target_length

    @classmethod
    def from_environment(cls, env: "SnakePOMDP") -> "SnakeVectorizedUpdater":
        """Construct an updater from a :class:`SnakePOMDP` instance."""
        return cls(
            grid_size=env.grid_size,
            target_length=env.target_length,
            starvation_limit=env.starvation_limit,
            window_radius=env.window_radius,
            detection_probability=env.detection_probability,
            scent_accuracy=env.scent_accuracy,
        )

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: Any) -> np.ndarray:
        """Step every particle under one action.

        The deterministic half -- turn, move, eat, grow, die -- is array algebra
        over the whole particle set. The stochastic half is the food respawn,
        which is uniform over each particle's own free cells and so is drawn per
        particle, still without a Python loop.

        Args:
            particles: ``(N, state_width)`` particles.
            action: A :class:`SnakeAction` value.

        Returns:
            ``(N, state_width)`` successors. Terminal particles come back
            unchanged, because a finished episode is absorbing.
        """
        states = np.array(particles, dtype=np.float64, copy=True).reshape(-1, self.state_width)
        if states.shape[0] == 0:
            return states

        running = states[:, STATUS_INDEX] == int(SnakeTermination.RUNNING)
        if not np.any(running):
            return states

        live = states[running]
        length = np.rint(live[:, LENGTH_INDEX]).astype(np.int64)
        body = live[:, BODY_OFFSET:].reshape(len(live), self.target_length, 2)

        target = self._targets(body, int(np.asarray(action).ravel()[0]))
        food = live[:, (FOOD_ROW_INDEX, FOOD_COL_INDEX)]
        eat = np.all(np.rint(target) == np.rint(food), axis=1)

        new_length = length + eat
        new_body = self._shift_body(body, target, new_length)
        counter = np.where(eat, 0.0, live[:, STEPS_SINCE_FOOD_INDEX] + 1.0)
        status = self._status(target, new_body, new_length, counter)

        successors = np.empty_like(live)
        successors[:, STATUS_INDEX] = status
        successors[:, LENGTH_INDEX] = new_length
        successors[:, STEPS_SINCE_FOOD_INDEX] = counter
        successors[:, BODY_OFFSET:] = new_body.reshape(len(live), -1)
        successors[:, (FOOD_ROW_INDEX, FOOD_COL_INDEX)] = self._next_food(
            food, eat, status, new_body, new_length
        )

        states[running] = successors
        return states

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: Any,
        observation: Any,
    ) -> np.ndarray:
        """Score one reading against every particle.

        The three parts of a reading are independent given the state, so their
        log-likelihoods add: the body is a point mass, the sighting is a
        Bernoulli draw with no false positives, and the scent is a categorical
        over the four quadrants.

        Args:
            next_particles: ``(N, state_width)`` transitioned particles.
            action: Unused; the reading depends on the next state alone.
            observation: A reading, as the flat tuple the environment emits.

        Returns:
            ``(N,)`` log-likelihoods.
        """
        del action
        particles = np.asarray(next_particles, dtype=np.float64).reshape(-1, self.state_width)
        flat = np.asarray(observation, dtype=np.float64).ravel()
        terminal = particles[:, STATUS_INDEX] != int(SnakeTermination.RUNNING)

        if flat.size == 0 or int(round(float(flat[0]))) != OBSERVATION_LIVE:
            # The terminal reading is the only thing a finished episode emits,
            # and it is all it emits, so it is a point mass on those particles.
            return np.where(terminal, 0.0, -np.inf)

        scores = np.full(particles.shape[0], -np.inf, dtype=np.float64)
        live = ~terminal
        if not np.any(live):
            return scores

        scent = int(round(float(flat[1])))
        seen = None if flat[2] < 0 else (int(round(float(flat[2]))), int(round(float(flat[3]))))
        observed_body = flat[4:]

        candidates = particles[live]
        length = np.rint(candidates[:, LENGTH_INDEX]).astype(np.int64)
        body_matches = (length == observed_body.size // 2) & np.all(
            np.rint(candidates[:, BODY_OFFSET : BODY_OFFSET + observed_body.size])
            == np.rint(observed_body),
            axis=1,
        )

        head = candidates[:, BODY_OFFSET : BODY_OFFSET + 2]
        food = candidates[:, (FOOD_ROW_INDEX, FOOD_COL_INDEX)]
        inside = np.all(np.abs(food - head) <= self.window_radius, axis=1)

        sighting = self._sighting_log_likelihood(inside, food, seen)
        scent_log = self._scent_log_likelihood(head, food, scent)

        scores[np.flatnonzero(live)] = np.where(body_matches, sighting + scent_log, -np.inf)
        return scores

    @property
    def config_id(self) -> str:
        """Return a deterministic identifier for this updater configuration."""
        return config_to_id(
            {
                "class": "SnakeVectorizedUpdater",
                "grid_size": self.grid_size,
                "target_length": self.target_length,
                "starvation_limit": self.starvation_limit,
                "window_radius": self.window_radius,
                "detection_probability": self.detection_probability,
                "scent_accuracy": self.scent_accuracy,
            }
        )

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    def _targets(self, body: np.ndarray, action: int) -> np.ndarray:
        """The cell each particle's head moves into.

        The heading is not stored: it is the step from the second body cell to
        the head, which is the only direction the head can have arrived from.
        """
        head = body[:, 0, :]
        heading = np.rint(head - body[:, 1, :]).astype(np.int64)
        index = np.where(
            heading[:, 0] == -1,
            0,
            np.where(heading[:, 1] == 1, 1, np.where(heading[:, 0] == 1, 2, 3)),
        )
        rotated = (index + _ACTION_ROTATION[int(action)]) % len(DIRECTIONS)
        return head + _DIRECTIONS[rotated]

    def _shift_body(
        self, body: np.ndarray, target: np.ndarray, new_length: np.ndarray
    ) -> np.ndarray:
        """Push the new head on and drop whatever falls past the new length.

        Eating keeps the tail where it is; otherwise the tail cell is released,
        which is what makes the cell it has just left safe to enter. Both cases
        are the same shift -- only the length that survives it differs.
        """
        shifted = np.empty_like(body)
        shifted[:, 0, :] = target
        shifted[:, 1:, :] = body[:, :-1, :]
        slots = np.arange(self.target_length)[None, :]
        used = slots < new_length[:, None]
        return np.where(used[:, :, None], shifted, EMPTY)

    def _status(
        self,
        target: np.ndarray,
        new_body: np.ndarray,
        new_length: np.ndarray,
        counter: np.ndarray,
    ) -> np.ndarray:
        """Why each particle's episode ended, in the environment's own order.

        Wall and self are checked first, then the win, then starvation: a snake
        that reaches its target length by walking into a wall has still hit the
        wall.
        """
        wall = np.any((target < 0) | (target >= self.grid_size), axis=1)

        # ``new_body[1:]`` for each particle, masked to the slots it actually
        # uses. A slot past the length holds EMPTY, which no target can equal.
        tail = new_body[:, 1:, :]
        slots = np.arange(1, self.target_length)[None, :]
        occupied = slots < new_length[:, None]
        hits_self = np.any(
            occupied & np.all(np.rint(tail) == np.rint(target)[:, None, :], axis=2), axis=1
        )

        win = new_length >= self.target_length
        starved = counter >= self.starvation_limit
        return np.where(
            wall,
            int(SnakeTermination.WALL),
            np.where(
                hits_self,
                int(SnakeTermination.SELF),
                np.where(
                    win,
                    int(SnakeTermination.WIN),
                    np.where(
                        starved, int(SnakeTermination.STARVATION), int(SnakeTermination.RUNNING)
                    ),
                ),
            ),
        ).astype(np.float64)

    def _next_food(
        self,
        food: np.ndarray,
        eat: np.ndarray,
        status: np.ndarray,
        new_body: np.ndarray,
        new_length: np.ndarray,
    ) -> np.ndarray:
        """Where the food is after the step.

        It respawns only when the step ate and the episode continues. A win
        names no cell -- the episode is over the moment the target length is
        reached -- and a death keeps the cell it had, which nothing reads.
        """
        respawns = eat & (status == int(SnakeTermination.RUNNING))
        out = np.array(food, dtype=np.float64, copy=True)
        out[status == int(SnakeTermination.WIN)] = EMPTY

        rows = np.flatnonzero(respawns)
        if rows.size:
            drawn = self._draw_free_cells(new_body[rows], new_length[rows])
            out[rows, 0] = drawn // self.grid_size
            out[rows, 1] = drawn % self.grid_size
        return out

    def _draw_free_cells(self, bodies: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Draw one free cell per particle, uniformly over the cells it leaves free.

        Args:
            bodies: ``(M, target_length, 2)`` bodies.
            lengths: ``(M,)`` how many of each body's slots are in use.

        Returns:
            ``(M,)`` flat cell indices.

        Raises:
            ValueError: If a body fills the grid, so the food has nowhere to
                respawn. The environment's ``grid_size`` check rules that out.
        """
        free = self._free_mask(bodies, lengths)
        counts = free.sum(axis=1)
        if np.any(counts == 0):
            raise ValueError("the body fills the grid, so the food has nowhere to respawn")
        rank = (np.random.random(len(free)) * counts).astype(np.int64) + 1
        # The rank-th free cell of each row, found by counting free cells along
        # it: one cumulative sum instead of a per-particle search.
        reached = (np.cumsum(free, axis=1) >= rank[:, None]) & free
        return np.argmax(reached, axis=1)

    def _free_mask(self, bodies: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """``(M, num_cells)`` mask of the cells each body leaves free.

        Off-grid body cells are skipped rather than folded into a flat index, the
        way the environment's own ``free_cells`` skips them.
        """
        slots = np.arange(self.target_length)[None, :]
        used = slots < lengths[:, None]
        rows = np.rint(bodies[:, :, 0]).astype(np.int64)
        cols = np.rint(bodies[:, :, 1]).astype(np.int64)
        in_grid = (
            (rows >= 0) & (rows < self.grid_size) & (cols >= 0) & (cols < self.grid_size) & used
        )
        free = np.ones((len(bodies), self.num_cells), dtype=bool)
        flat = np.clip(rows * self.grid_size + cols, 0, self.num_cells - 1)
        particle_index = np.broadcast_to(np.arange(len(bodies))[:, None], flat.shape)
        free[particle_index[in_grid], flat[in_grid]] = False
        return free

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _sighting_log_likelihood(
        self, inside: np.ndarray, food: np.ndarray, seen: Optional[Tuple[int, int]]
    ) -> np.ndarray:
        """Log-likelihood of the sighting half of a reading.

        There are no false positives, so a sighting of anything but the
        particle's own food cell, or any sighting while that cell is outside the
        window, is a reading the sensor cannot produce.
        """
        if seen is None:
            return np.where(inside, np.log1p(-self.detection_probability), 0.0)
        matches = inside & np.all(np.rint(food) == np.asarray(seen, dtype=np.float64), axis=1)
        with np.errstate(divide="ignore"):
            return np.where(matches, float(np.log(self.detection_probability)), -np.inf)

    def _scent_log_likelihood(self, head: np.ndarray, food: np.ndarray, scent: int) -> np.ndarray:
        """Log-likelihood of the reported quadrant under each particle.

        A quadrant is compatible when every non-zero component of the offset
        agrees with it, so food sharing the head's row or column is compatible
        with two quadrants and they split the accuracy between them.
        """
        if not 0 <= scent < len(SnakeQuadrant):
            return np.full(len(head), -np.inf, dtype=np.float64)

        delta = np.rint(food - head)
        vertical_ok = np.where(
            delta[:, 0] < 0, _IS_NORTH[scent], np.where(delta[:, 0] > 0, ~_IS_NORTH[scent], True)
        )
        horizontal_ok = np.where(
            delta[:, 1] > 0, _IS_EAST[scent], np.where(delta[:, 1] < 0, ~_IS_EAST[scent], True)
        )
        compatible = vertical_ok & horizontal_ok
        # Two compatible quadrants exactly when one offset component is zero.
        n_compatible = np.where((delta[:, 0] == 0) | (delta[:, 1] == 0), 2.0, 1.0)
        n_wrong = len(SnakeQuadrant) - n_compatible

        with np.errstate(divide="ignore"):
            hit = np.log(self.scent_accuracy / n_compatible)
            miss = np.log((1.0 - self.scent_accuracy) / n_wrong)
        return np.where(compatible, hit, miss)

    # ------------------------------------------------------------------
    # Redraw support
    # ------------------------------------------------------------------

    def redraw_food(self, particles: np.ndarray, observation: Any) -> np.ndarray:
        """Resample the food cell of every particle from one reading alone.

        Used when the reading has ruled out every particle the belief held. The
        body is known exactly, so the only thing to redraw is the food cell, and
        the reading says everything there is to say about it: a sighting names
        one cell, and otherwise the free cells are drawn in proportion to the
        scent likelihood.

        Args:
            particles: ``(N, state_width)`` particles to redraw the food of.
            observation: The reading that ruled them out.

        Returns:
            ``(N, state_width)`` particles with fresh food cells.
        """
        out = np.array(particles, dtype=np.float64, copy=True)
        flat = np.asarray(observation, dtype=np.float64).ravel()
        if flat.size == 0 or int(round(float(flat[0]))) != OBSERVATION_LIVE:
            return out

        if flat[2] >= 0:
            out[:, FOOD_ROW_INDEX] = float(np.rint(flat[2]))
            out[:, FOOD_COL_INDEX] = float(np.rint(flat[3]))
            return out

        head = np.rint(out[0, BODY_OFFSET : BODY_OFFSET + 2]).astype(np.int64)
        lengths = np.rint(out[:, LENGTH_INDEX]).astype(np.int64)
        bodies = out[:, BODY_OFFSET:].reshape(len(out), self.target_length, 2)
        free = self._free_mask(bodies, lengths)

        weights = self._cell_scent_weights(head, int(round(float(flat[1]))))
        drawn = self._draw_weighted_cells(free, weights)
        out[:, FOOD_ROW_INDEX] = drawn // self.grid_size
        out[:, FOOD_COL_INDEX] = drawn % self.grid_size
        return out

    def _cell_scent_weights(self, head: np.ndarray, scent: int) -> np.ndarray:
        """``(num_cells,)`` scent likelihood of every cell, given the head."""
        rows, cols = np.divmod(np.arange(self.num_cells), self.grid_size)
        cells = np.stack([rows, cols], axis=1).astype(np.float64)
        heads = np.broadcast_to(head.astype(np.float64), cells.shape)
        weights = np.exp(self._scent_log_likelihood(heads, cells, scent))
        # The head's own cell has no offset and so no quadrant; the body never
        # holds the food, and the free mask drops it before these weights are used.
        weights[head[0] * self.grid_size + head[1]] = 0.0
        return weights

    def _draw_weighted_cells(self, free: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Draw one cell per row, from the free cells, in proportion to ``weights``."""
        masked = free * weights[None, :]
        totals = masked.sum(axis=1, keepdims=True)
        # A reading that makes every free cell impossible cannot happen -- some
        # quadrant always has positive probability -- but a uniform fallback
        # keeps a hand-built belief from dividing by zero.
        masked = np.where(totals > 0.0, masked, free.astype(np.float64))
        cumulative = np.cumsum(masked, axis=1)
        draws = np.random.random(len(free))[:, None] * cumulative[:, -1:]
        return np.argmax(cumulative >= draws, axis=1)


class SnakeVectorizedWeightedParticleBelief(VectorizedWeightedParticleBelief):
    """Vectorized particle belief that redraws when a sighting rules everything out."""

    updater: SnakeVectorizedUpdater

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Optional["Environment"] = None,
        state: Optional[Any] = None,
    ) -> "SnakeVectorizedWeightedParticleBelief":
        """Move the particles, reweight them, and redraw if the reading killed them all.

        Args:
            action: The action that was executed.
            observation: The reading the environment reported.
            pomdp: Unused; the updater owns everything the update needs.
            state: Ignored, so the true state cannot leak into the belief.

        Returns:
            The posterior belief.
        """
        del pomdp, state
        next_particles = self.updater.batch_transition(self.particles, action)
        log_likelihoods = self.updater.batch_observation_log_likelihood(
            next_particles, action, observation
        )
        next_log_weights = self.log_weights + log_likelihoods

        if not np.any(np.isfinite(next_log_weights)):
            # Every particle is impossible under this reading. Resampling would
            # spread those impossible food cells over the whole set and the
            # belief would go on meaning nothing; the redraw puts the particles
            # back on the cells the reading allows.
            next_particles = self.updater.redraw_food(next_particles, observation)
            next_log_weights = np.full(
                len(next_particles), -float(np.log(len(next_particles))), dtype=np.float64
            )
        elif self.resampling:
            next_particles, next_log_weights = self._resample(next_particles, next_log_weights)

        return SnakeVectorizedWeightedParticleBelief(
            particles=next_particles,
            log_weights=next_log_weights,
            updater=self.updater,
            resampling=self.resampling,
            ess_factor=self.ess_factor,
        )


def create_snake_belief(
    env: "SnakePOMDP",
    belief_type: BeliefType = BeliefType.VECTORIZED_PARTICLE,
    n_particles: int = 200,
    **kwargs: Any,
) -> "Belief":
    """Create a belief for the Snake POMDP.

    Args:
        env: The environment.
        belief_type: ``PARTICLE`` or ``VECTORIZED_PARTICLE``.
        n_particles: Number of particles. Defaults to 200.
        **kwargs: Reserved for future use.

    Returns:
        A configured belief.

    Raises:
        ValueError: If *belief_type* is not supported.
    """
    del kwargs
    if belief_type == BeliefType.PARTICLE:
        return get_initial_belief(env, n_particles)
    if belief_type != BeliefType.VECTORIZED_PARTICLE:
        raise ValueError(f"SnakePOMDP does not support belief type {belief_type!r}")

    updater = SnakeVectorizedUpdater.from_environment(env)
    particles = np.stack(env.initial_state_dist().sample(n_samples=n_particles))
    log_weights = np.full(n_particles, -float(np.log(n_particles)), dtype=np.float64)
    return SnakeVectorizedWeightedParticleBelief(
        particles=particles,
        log_weights=log_weights,
        updater=updater,
        resampling=True,
    )


__all__ = [
    "SnakeVectorizedUpdater",
    "SnakeVectorizedWeightedParticleBelief",
    "create_snake_belief",
]
