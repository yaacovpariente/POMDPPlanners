# SPDX-License-Identifier: MIT

"""Vectorized particle belief for the CaptureTheFlag POMDP.

Blue cannot see the red players and does not know which candidate cell holds
the red flag, and it infers both from the same reading: a noisy range from every
blue player to every red player, plus a per-player flag detector. Because every
blue player measures every red player, a particle's weight is a product of
``n_blue * n_red`` range factors and ``n_blue`` detector factors -- so a scalar
filter spends its whole update inside nested Python loops.

This updater keeps those loops, but over *players* rather than particles: the
team sizes are small and fixed, and the particle set is the axis that grows.
Every step below is one array operation across all particles at once, and the
player loops mirror the environment's own, in the same order, so the two stay
readable side by side. The order is the semantics here -- pick-up before
tagging, both scoring conditions judged against the same carrier ids, counters
decremented before any stage sets a fresh one -- and a copy that reorders them
is a different game.

Classes:
    CaptureTheFlagVectorizedUpdater: Batched transition and likelihood.
    CaptureTheFlagVectorizedBelief: The belief, which conditions on the
        exactly-observed half of a reading rather than weighting by it.

Functions:
    create_capture_the_flag_belief: Factory used by the top-level belief factory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

import numpy as np

from POMDPPlanners.core.belief.belief_utils import get_initial_belief
from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_pomdp_utils import (
    ACTION_DELTAS,
    ACTION_SCAN,
    PERPENDICULAR,
    RedRole,
    StateLayout,
    decode_joint_action,
)
from POMDPPlanners.utils.belief_factory import BeliefType
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.core.belief.base_belief import Belief
    from POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_pomdp import (
        CaptureTheFlagPOMDP,
    )

#: The sentinel every terminal state emits, mirrored from the environment.
_TERMINAL_OBSERVATION_VALUE = -1.0

#: The five cells a red player may step to, itself first, in the environment's
#: own neighbour order.
_RED_STEPS = ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0))


class CaptureTheFlagVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Batched transition and observation likelihood for CaptureTheFlag.

    Attributes:
        layout: Index offsets into the flat state vector.
        n_blue: Blue team size.
        n_red: Red team size.
        grid_size: Field dimensions as ``(width, height)``.
        free: ``(width, height)`` bool array of the cells a player may stand on.
    """

    # pylint: disable=too-many-instance-attributes  # one game's rules, spelled out

    def __init__(self, env: "CaptureTheFlagPOMDP") -> None:
        """Initialize the updater from an environment.

        Built from the environment rather than from a long parameter list: this
        model reads two dozen rule parameters, and naming them twice would be
        two dozen chances for the copy to drift.

        Args:
            env: The environment whose rules to reproduce.
        """
        self.layout: StateLayout = env.layout
        self.n_blue = int(env.n_blue)
        self.n_red = int(env.n_red)
        self.grid_size: Tuple[int, int] = (int(env.grid_size[0]), int(env.grid_size[1]))
        self.midline = int(env.midline)
        self.blue_base = tuple(int(value) for value in env.blue_base)
        self.red_base = tuple(int(value) for value in env.red_base)
        self.blue_flag_cell = tuple(int(value) for value in env.blue_flag_cell)
        self.red_flag_candidates = [
            (int(cell[0]), int(cell[1])) for cell in env.red_flag_candidates
        ]
        self.red_roles = list(env.red_roles)
        self.red_alert_radius = int(env.red_alert_radius)
        self.red_pursuit_probability = float(env.red_pursuit_probability)
        self.slip_probability = float(env.slip_probability)
        self.freeze_steps = int(env.freeze_steps)
        self.tagger_cooldown_steps = int(env.tagger_cooldown_steps)
        self.score_to_win = int(env.score_to_win)
        self.range_error_probability = float(env.range_error_probability)
        self.max_range = int(env.max_range)
        self.detector_half_distance_move = float(env.detector_half_distance_move)
        self.detector_half_distance_scan = float(env.detector_half_distance_scan)
        self.observation_size = int(env.observation_size)

        self.free = np.zeros(self.grid_size, dtype=bool)
        for cell in env.free_cells():
            self.free[int(cell[0]), int(cell[1])] = True
        # One guard post per flag candidate, precomputed: it is a pure function
        # of the candidate, and the environment computes it the same way.
        self.guard_posts = np.array(
            [env.guard_post(cell) for cell in self.red_flag_candidates], dtype=np.int64
        )
        self.flag_candidates = np.asarray(self.red_flag_candidates, dtype=np.int64)

    @classmethod
    def from_environment(cls, env: "CaptureTheFlagPOMDP") -> "CaptureTheFlagVectorizedUpdater":
        """Construct an updater from a :class:`CaptureTheFlagPOMDP` instance."""
        return cls(env)

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: Any) -> np.ndarray:
        """Advance every particle one step under one joint action.

        Blue moves first, then red picks a target from where blue has landed and
        moves, then the deterministic stages resolve pick-up, tagging, scoring
        and the counters.

        Args:
            particles: ``(N, state_size)`` particles.
            action: The joint action id.

        Returns:
            ``(N, state_size)`` successors. Terminal particles come back
            unchanged.
        """
        states = np.asarray(particles, dtype=np.float64).reshape(-1, self.layout.size)
        if states.shape[0] == 0:
            return np.array(states, copy=True)
        player_actions = decode_joint_action(int(np.asarray(action).ravel()[0]), self.n_blue)

        blue_cells = self._draw_blue_cells(states, player_actions)
        red_cells = self._draw_red_cells(states, blue_cells)
        successors = self._apply_deterministic_stages(states, blue_cells, red_cells)

        terminal = self._terminal_mask(states)
        successors[terminal] = states[terminal]
        return successors

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: Any,
        observation: Any,
    ) -> np.ndarray:
        """Score one reading against every particle.

        The exactly-observed components -- the blue positions, the carrier flags,
        the freezes and the scores -- act as a delta factor: a particle that
        disagrees with them is impossible, not merely unlikely. What remains is
        the product of the range readings and the flag detectors.

        Args:
            next_particles: ``(N, state_size)`` transitioned particles.
            action: The joint action taken to reach them, which sets each
                player's detector range.
            observation: The reading.

        Returns:
            ``(N,)`` log-likelihoods.
        """
        states = np.asarray(next_particles, dtype=np.float64).reshape(-1, self.layout.size)
        values = np.asarray(observation, dtype=np.float64).ravel()
        terminal = self._terminal_mask(states)

        if values.size != self.observation_size:
            return np.full(states.shape[0], -np.inf, dtype=np.float64)
        if np.all(values == _TERMINAL_OBSERVATION_VALUE):
            # The sentinel is emitted by terminal states and only by them.
            return np.where(terminal, 0.0, -np.inf)

        exact = self._exact_components_match(states, self._blue_cells(states), values)
        return np.where(
            exact & ~terminal, self.noisy_log_likelihood(states, action, values), -np.inf
        )

    def noisy_log_likelihood(
        self, next_particles: np.ndarray, action: Any, observation: Any
    ) -> np.ndarray:
        """Score only the noisy half of a reading: the ranges and the detectors.

        The exactly-observed half is a delta factor, so it contributes either
        zero or ``-inf``, and :class:`CaptureTheFlagVectorizedBelief` handles it
        by conditioning rather than by weighting. This entry point is what it
        weights with; :meth:`batch_observation_log_likelihood` is the full
        likelihood and multiplies the delta factor back in.

        Args:
            next_particles: ``(N, state_size)`` transitioned particles.
            action: The joint action taken to reach them.
            observation: The reading.

        Returns:
            ``(N,)`` log-likelihoods of the noisy components alone.
        """
        states = np.asarray(next_particles, dtype=np.float64).reshape(-1, self.layout.size)
        values = np.asarray(observation, dtype=np.float64).ravel()
        player_actions = decode_joint_action(int(np.asarray(action).ravel()[0]), self.n_blue)
        blue = self._blue_cells(states)
        red = self._red_cells(states)
        flag = self._flag_cells(states)

        total = np.zeros(states.shape[0], dtype=np.float64)
        offset = 2 * self.n_blue
        for i in range(self.n_blue):
            for j in range(self.n_red):
                reading = values[offset]
                offset += 1
                distance = _manhattan(blue[:, i, :], red[:, j, :])
                total = total + _log(self._range_probability(distance, reading))
        for i in range(self.n_blue):
            bit = values[offset]
            offset += 1
            probability = self._detection_probability(
                _manhattan(blue[:, i, :], flag), player_actions[i]
            )
            if bit not in (0.0, 1.0):
                return np.full(states.shape[0], -np.inf, dtype=np.float64)
            total = total + _log(probability if bit == 1.0 else 1.0 - probability)
        return total

    @property
    def config_id(self) -> str:
        """Return a deterministic identifier for this updater configuration."""
        return config_to_id(
            {
                "class": "CaptureTheFlagVectorizedUpdater",
                "grid_size": list(self.grid_size),
                "midline": self.midline,
                "free_cells": np.flatnonzero(self.free.ravel()).tolist(),
                "n_blue": self.n_blue,
                "n_red": self.n_red,
                "blue_base": list(self.blue_base),
                "red_base": list(self.red_base),
                "blue_flag_cell": list(self.blue_flag_cell),
                "red_flag_candidates": self.flag_candidates.tolist(),
                "red_roles": [role.value for role in self.red_roles],
                "red_alert_radius": self.red_alert_radius,
                "red_pursuit_probability": self.red_pursuit_probability,
                "slip_probability": self.slip_probability,
                "freeze_steps": self.freeze_steps,
                "tagger_cooldown_steps": self.tagger_cooldown_steps,
                "score_to_win": self.score_to_win,
                "range_error_probability": self.range_error_probability,
                "max_range": self.max_range,
                "detector_half_distance_move": self.detector_half_distance_move,
                "detector_half_distance_scan": self.detector_half_distance_scan,
            }
        )

    # ------------------------------------------------------------------
    # State views
    # ------------------------------------------------------------------

    def _blue_cells(self, states: np.ndarray) -> np.ndarray:
        """``(N, n_blue, 2)`` blue positions."""
        base = self.layout.blue_pos
        return states[:, base : base + 2 * self.n_blue].reshape(-1, self.n_blue, 2)

    def _red_cells(self, states: np.ndarray) -> np.ndarray:
        """``(N, n_red, 2)`` red positions."""
        base = self.layout.red_pos
        return states[:, base : base + 2 * self.n_red].reshape(-1, self.n_red, 2)

    def _flag_cells(self, states: np.ndarray) -> np.ndarray:
        """``(N, 2)`` red flag home cell of each particle."""
        return self.flag_candidates[states[:, self.layout.flag_cell].astype(np.int64)]

    def _terminal_mask(self, states: np.ndarray) -> np.ndarray:
        """Which particles have already been won."""
        return (states[:, self.layout.score_blue] >= self.score_to_win) | (
            states[:, self.layout.score_red] >= self.score_to_win
        )

    def _is_free(self, cells: np.ndarray) -> np.ndarray:
        """Whether each ``(..., 2)`` cell is on the field and not a tree."""
        x = cells[..., 0].astype(np.int64)
        y = cells[..., 1].astype(np.int64)
        inside = (x >= 0) & (x < self.grid_size[0]) & (y >= 0) & (y < self.grid_size[1])
        return (
            inside
            & self.free[np.clip(x, 0, self.grid_size[0] - 1), np.clip(y, 0, self.grid_size[1] - 1)]
        )

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def _draw_blue_cells(self, states: np.ndarray, player_actions: Tuple[int, ...]) -> np.ndarray:
        """Where each blue player lands, drawn per particle.

        A move slips to either perpendicular direction with probability
        ``slip_probability / 2``; a move into a tree or off the field leaves the
        player where it stood. A frozen player does not move at all.
        """
        cells = np.array(self._blue_cells(states), dtype=np.float64, copy=True)
        for i, player_action in enumerate(player_actions):
            if player_action not in PERPENDICULAR:
                continue
            frozen = states[:, self.layout.freeze_blue + i] > 0.0
            left, right = PERPENDICULAR[player_action]
            draws = np.random.random(states.shape[0])
            chosen = np.where(
                draws < 1.0 - self.slip_probability,
                player_action,
                np.where(draws < 1.0 - self.slip_probability / 2.0, left, right),
            )
            deltas = np.array([ACTION_DELTAS[int(value)] for value in chosen], dtype=np.float64)
            target = cells[:, i, :] + deltas
            blocked = ~self._is_free(target) | frozen
            cells[:, i, :] = np.where(blocked[:, None], cells[:, i, :], target)
        return cells

    def _draw_red_cells(self, states: np.ndarray, blue_cells: np.ndarray) -> np.ndarray:
        """Where each red player lands, drawn per particle.

        A red player takes one of its free neighbours or stays put: with
        probability ``red_pursuit_probability`` it takes one that closes on its
        target, and otherwise it picks uniformly among all of them.
        """
        cells = np.array(self._red_cells(states), dtype=np.float64, copy=True)
        flag = self._flag_cells(states)
        carrier_blue = states[:, self.layout.carrier_blue_flag].astype(np.int64)

        for j in range(self.n_red):
            frozen = states[:, self.layout.freeze_red + j] > 0.0
            target = self._red_target(j, blue_cells, cells[:, j, :], flag, carrier_blue, states)
            options = cells[:, j, None, :] + np.asarray(_RED_STEPS, dtype=np.float64)[None, :, :]
            legal = self._is_free(options)
            # The cell itself is always an option, tree or not: the environment
            # lists it before testing any neighbour.
            legal[:, 0] = True

            distances = _manhattan(options, target[:, None, :])
            best = np.where(legal, distances, np.iinfo(np.int64).max).min(axis=1, keepdims=True)
            closing = legal & (distances == best)

            weights = legal * (
                (1.0 - self.red_pursuit_probability) / legal.sum(axis=1, keepdims=True)
            )
            weights = weights + closing * (
                self.red_pursuit_probability / closing.sum(axis=1, keepdims=True)
            )
            picked = _draw_index(weights)
            drawn = np.take_along_axis(options, picked[:, None, None], axis=1)[:, 0, :]
            cells[:, j, :] = np.where(frozen[:, None], cells[:, j, :], drawn)
        return cells

    def _red_target(
        self,
        red_index: int,
        blue_cells: np.ndarray,
        red_cell: np.ndarray,
        flag: np.ndarray,
        carrier_blue: np.ndarray,
        states: np.ndarray,
    ) -> np.ndarray:
        """The cell one red player is heading for, per particle.

        Carrying beats the role: a defender that picked the blue flag up must
        still run it home, or the flag sits in the red half forever and red can
        never score.
        """
        del states
        count = blue_cells.shape[0]
        if self.red_roles[red_index] is RedRole.ATTACK:
            role_target = np.broadcast_to(
                np.asarray(self.blue_flag_cell, dtype=np.float64), (count, 2)
            )
        else:
            role_target = self._defender_target(blue_cells, red_cell, flag)
        carrying = (carrier_blue == red_index + 1)[:, None]
        return np.where(carrying, np.asarray(self.red_base, dtype=np.float64)[None, :], role_target)

    def _defender_target(
        self, blue_cells: np.ndarray, red_cell: np.ndarray, flag: np.ndarray
    ) -> np.ndarray:
        """A defender's target: the nearest close intruder, else the guard post.

        Ties are broken by distance first and then by the cell itself, the way
        the environment's ``min`` over ``(distance, cell)`` breaks them. The
        ordering is packed into one integer key so the whole batch can be argmin'd
        at once.
        """
        intruder = blue_cells[:, :, 0] > self.midline
        distance = _manhattan(blue_cells, red_cell[:, None, :])
        height = self.grid_size[1]
        cell_key = blue_cells[:, :, 0].astype(np.int64) * height + blue_cells[:, :, 1].astype(
            np.int64
        )
        key = distance * (self.grid_size[0] * height) + cell_key
        key = np.where(intruder, key, np.iinfo(np.int64).max)

        nearest = np.argmin(key, axis=1)
        chosen = np.take_along_axis(blue_cells, nearest[:, None, None], axis=1)[:, 0, :]
        chosen_distance = np.take_along_axis(distance, nearest[:, None], axis=1)[:, 0]
        chase = np.any(intruder, axis=1) & (chosen_distance <= self.red_alert_radius)

        posts = self.guard_posts[_flag_index(flag, self.flag_candidates)].astype(np.float64)
        return np.where(chase[:, None], chosen, posts)

    # ------------------------------------------------------------------
    # Deterministic stages
    # ------------------------------------------------------------------

    def _apply_deterministic_stages(
        self, states: np.ndarray, blue_cells: np.ndarray, red_cells: np.ndarray
    ) -> np.ndarray:
        """Resolve pick-up, tagging, scoring and the counters, for every particle.

        Pick-up runs before tagging, so a player tagged on the flag cell has
        already taken the flag and therefore drops it. Both scoring conditions
        are judged against the same pre-scoring carrier ids, which keeps blue
        scoring and red scoring mutually exclusive rather than letting whichever
        is resolved first enable the other.
        """
        layout = self.layout
        successors = np.zeros_like(states)
        successors[:, layout.flag_cell] = states[:, layout.flag_cell]

        freeze_blue = np.maximum(
            states[:, layout.freeze_blue : layout.freeze_blue + self.n_blue] - 1.0, 0.0
        )
        freeze_red = np.maximum(
            states[:, layout.freeze_red : layout.freeze_red + self.n_red] - 1.0, 0.0
        )
        cooldown_blue = np.maximum(
            states[:, layout.cooldown_blue : layout.cooldown_blue + self.n_blue] - 1.0, 0.0
        )
        cooldown_red = np.maximum(
            states[:, layout.cooldown_red : layout.cooldown_red + self.n_red] - 1.0, 0.0
        )
        was_frozen_blue = states[:, layout.freeze_blue : layout.freeze_blue + self.n_blue] > 0.0
        was_frozen_red = states[:, layout.freeze_red : layout.freeze_red + self.n_red] > 0.0

        carrier_red_flag = states[:, layout.carrier_red_flag].astype(np.int64)
        carrier_blue_flag = states[:, layout.carrier_blue_flag].astype(np.int64)
        flag = self._flag_cells(states)

        carrier_red_flag = self._pick_up(
            carrier_red_flag, blue_cells, flag.astype(np.float64), was_frozen_blue
        )
        carrier_blue_flag = self._pick_up(
            carrier_blue_flag,
            red_cells,
            np.broadcast_to(
                np.asarray(self.blue_flag_cell, dtype=np.float64), (states.shape[0], 2)
            ),
            was_frozen_red,
        )

        carrier_red_flag, carrier_blue_flag = self._tagging(
            blue_cells,
            red_cells,
            states,
            was_frozen_blue,
            was_frozen_red,
            freeze_blue,
            freeze_red,
            cooldown_blue,
            cooldown_red,
            carrier_red_flag,
            carrier_blue_flag,
        )

        score_blue = states[:, layout.score_blue]
        score_red = states[:, layout.score_red]
        blue_scores = (
            (carrier_red_flag != 0)
            & _at_cell(blue_cells, carrier_red_flag - 1, self.blue_base)
            & (carrier_blue_flag == 0)
        )
        red_scores = (
            (carrier_blue_flag != 0)
            & _at_cell(red_cells, carrier_blue_flag - 1, self.red_base)
            & (carrier_red_flag == 0)
        )
        score_blue = score_blue + blue_scores
        score_red = score_red + red_scores
        carrier_red_flag = np.where(blue_scores, 0, carrier_red_flag)
        carrier_blue_flag = np.where(red_scores, 0, carrier_blue_flag)

        successors[:, layout.blue_pos : layout.blue_pos + 2 * self.n_blue] = blue_cells.reshape(
            len(states), -1
        )
        successors[:, layout.red_pos : layout.red_pos + 2 * self.n_red] = red_cells.reshape(
            len(states), -1
        )
        successors[:, layout.carrier_red_flag] = carrier_red_flag
        successors[:, layout.carrier_blue_flag] = carrier_blue_flag
        successors[:, layout.freeze_blue : layout.freeze_blue + self.n_blue] = freeze_blue
        successors[:, layout.freeze_red : layout.freeze_red + self.n_red] = freeze_red
        successors[:, layout.cooldown_blue : layout.cooldown_blue + self.n_blue] = cooldown_blue
        successors[:, layout.cooldown_red : layout.cooldown_red + self.n_red] = cooldown_red
        successors[:, layout.score_blue] = score_blue
        successors[:, layout.score_red] = score_red
        return successors

    def _pick_up(
        self,
        carrier: np.ndarray,
        cells: np.ndarray,
        flag_cell: np.ndarray,
        was_frozen: np.ndarray,
    ) -> np.ndarray:
        """Give the flag to the lowest-indexed unfrozen player standing on it."""
        updated = np.array(carrier, copy=True)
        for index in range(cells.shape[1]):
            on_flag = np.all(cells[:, index, :] == flag_cell, axis=1) & ~was_frozen[:, index]
            updated = np.where((updated == 0) & on_flag, index + 1, updated)
        return updated

    # pylint: disable-next=too-many-arguments,too-many-locals
    def _tagging(
        self,
        blue_cells: np.ndarray,
        red_cells: np.ndarray,
        states: np.ndarray,
        was_frozen_blue: np.ndarray,
        was_frozen_red: np.ndarray,
        freeze_blue: np.ndarray,
        freeze_red: np.ndarray,
        cooldown_blue: np.ndarray,
        cooldown_red: np.ndarray,
        carrier_red_flag: np.ndarray,
        carrier_blue_flag: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Send tagged players home, in the environment's own order.

        Red tags first, then blue, and each tagger tags at most one opponent --
        the environment breaks out of its inner loop. The "already tagged this
        step" guards matter: a player tagged earlier in the stage has already
        been teleported to its own base, and without the guard it would be
        tagged again from a cell it never walked through.

        The arrays are modified in place; the carrier ids are returned because
        they are scalars per particle rather than slots in an array.
        """
        layout = self.layout
        tagged_blue = np.zeros(blue_cells.shape[:2], dtype=bool)
        tagged_red = np.zeros(red_cells.shape[:2], dtype=bool)

        for j in range(self.n_red):
            free_tagger = ~was_frozen_red[:, j] & (states[:, layout.cooldown_red + j] <= 0.0)
            done = np.zeros(len(states), dtype=bool)
            for i in range(self.n_blue):
                same_cell = np.all(blue_cells[:, i, :] == red_cells[:, j, :], axis=1)
                in_red_half = blue_cells[:, i, 0] > self.midline
                tags = (
                    free_tagger
                    & ~done
                    & same_cell
                    & in_red_half
                    & ~was_frozen_blue[:, i]
                    & ~tagged_blue[:, i]
                )
                blue_cells[tags, i, :] = np.asarray(self.blue_base, dtype=np.float64)
                freeze_blue[tags, i] = float(self.freeze_steps)
                cooldown_red[tags, j] = float(self.tagger_cooldown_steps)
                carrier_red_flag = np.where(tags & (carrier_red_flag == i + 1), 0, carrier_red_flag)
                tagged_blue[:, i] |= tags
                done |= tags

        for i in range(self.n_blue):
            free_tagger = (
                ~was_frozen_blue[:, i]
                & ~tagged_blue[:, i]
                & (states[:, layout.cooldown_blue + i] <= 0.0)
            )
            done = np.zeros(len(states), dtype=bool)
            for j in range(self.n_red):
                same_cell = np.all(red_cells[:, j, :] == blue_cells[:, i, :], axis=1)
                in_blue_half = red_cells[:, j, 0] < self.midline
                tags = (
                    free_tagger
                    & ~done
                    & same_cell
                    & in_blue_half
                    & ~was_frozen_red[:, j]
                    & ~tagged_red[:, j]
                )
                red_cells[tags, j, :] = np.asarray(self.red_base, dtype=np.float64)
                freeze_red[tags, j] = float(self.freeze_steps)
                cooldown_blue[tags, i] = float(self.tagger_cooldown_steps)
                carrier_blue_flag = np.where(
                    tags & (carrier_blue_flag == j + 1), 0, carrier_blue_flag
                )
                tagged_red[:, j] |= tags
                done |= tags

        return carrier_red_flag, carrier_blue_flag

    # ------------------------------------------------------------------
    # Observation model
    # ------------------------------------------------------------------

    def _exact_components_match(
        self, states: np.ndarray, blue: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """Whether each particle agrees with everything the reading states exactly."""
        layout = self.layout
        matches = np.all(blue.reshape(len(states), -1) == values[: 2 * self.n_blue], axis=1)

        suffix = values[-(self.n_blue + 4) :]
        matches &= states[:, layout.carrier_red_flag] == suffix[0]
        matches &= (states[:, layout.carrier_blue_flag] != 0.0) == (suffix[1] != 0.0)
        for i in range(self.n_blue):
            matches &= states[:, layout.freeze_blue + i] == suffix[2 + i]
        matches &= states[:, layout.score_blue] == suffix[-2]
        matches &= states[:, layout.score_red] == suffix[-1]
        return matches

    def _range_probability(self, distance: np.ndarray, reading: float) -> np.ndarray:
        """Probability of one range reading under each particle's true distance.

        Readings are clipped to ``[0, max_range]`` and the mass that would fall
        outside is folded back onto the end point, so a reading at either
        extreme collects both its own weight and the weight of the value it
        stands in for.
        """
        if reading != np.floor(reading):
            # The sampler emits integers only, so a fractional reading is
            # impossible rather than approximate.
            return np.zeros_like(distance, dtype=np.float64)
        probability = np.zeros(distance.shape, dtype=np.float64)
        for offset, weight in (
            (0, 1.0 - self.range_error_probability),
            (-1, self.range_error_probability / 2.0),
            (1, self.range_error_probability / 2.0),
        ):
            clipped = np.clip(distance + offset, 0, self.max_range)
            probability += np.where(clipped == int(reading), weight, 0.0)
        return probability

    def _detection_probability(self, distance: np.ndarray, player_action: int) -> np.ndarray:
        """Probability the flag detector fires, per particle."""
        half = (
            self.detector_half_distance_scan
            if player_action == ACTION_SCAN
            else self.detector_half_distance_move
        )
        return 0.5 * (1.0 + 2.0 ** (-distance / half))


class CaptureTheFlagVectorizedBelief(VectorizedWeightedParticleBelief):
    """Vectorized belief that conditions on what blue sees exactly.

    Blue's own positions, the carrier ids, its freeze counters and both scores
    come back from the sensor without noise, so the posterior puts all its mass
    on the values the reading states. A plain particle filter has to *wait* for
    that: it samples blue's slip, red's move and the flag candidate together,
    then floors every particle whose blue players slipped differently from the
    real ones -- which, with two blue players and a one-in-ten slip, is most of
    them on most steps, and the weights that survive say more about who guessed
    the slip than about where red is.

    This belief conditions instead. After the transition it writes the observed
    components onto every particle and weights only the noisy half -- the ranges
    and the flag detectors -- which is the half that actually carries
    information about the hidden state.

    The approximation is real and worth naming: red's move was drawn against the
    blue cells the particle sampled, not against the observed ones, so a
    particle whose blue player slipped has a red player that chased a cell blue
    never stood on. The two differ by at most one cell, and every alternative
    considered costs more than that: waiting for agreement gives a belief that
    is uniform over particles already known to be wrong, which looks exactly
    like a healthy prior and is not.

    What this belief does *not* do is track the flag candidate exactly. The
    range likelihood is sharp -- four readings, each with a one-cell error bar --
    so a candidate's weight is an average over however many red hypotheses its
    particles happen to hold, and that average is noisy. Measured over twelve
    15-step episodes on the default field it puts a mean weight of about 0.7 on
    the true candidate at 200 particles and about 0.8 at 400, converging
    outright in most episodes and over-committing to a wrong candidate in a
    couple. Marginalizing red out analytically would fix that, and is a larger
    design than this one.
    """

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Any = None,
        state: Any = None,
    ) -> "CaptureTheFlagVectorizedBelief":
        """Step, condition on the exactly-observed components, reweight, resample.

        Args:
            action: The joint action that was executed.
            observation: This step's reading.
            pomdp: Unused; the updater carries the rules.
            state: Ignored, so the true state cannot leak into the belief.

        Returns:
            The posterior belief.
        """
        del pomdp, state
        updater: CaptureTheFlagVectorizedUpdater = self.updater
        values = np.asarray(observation, dtype=np.float64).ravel()
        next_particles = updater.batch_transition(self.particles, action)

        if values.size == updater.observation_size and not np.all(
            values == _TERMINAL_OBSERVATION_VALUE
        ):
            next_particles = _write_observed_components(next_particles, values, updater)
            log_likelihoods = updater.noisy_log_likelihood(next_particles, action, values)
        else:
            log_likelihoods = updater.batch_observation_log_likelihood(
                next_particles, action, values
            )

        next_particles, next_log_weights = self._stratified_update(next_particles, log_likelihoods)

        return CaptureTheFlagVectorizedBelief(
            particles=next_particles,
            log_weights=next_log_weights,
            updater=self.updater,
            resampling=self.resampling,
            ess_factor=self.ess_factor,
        )

    def _stratified_update(
        self, particles: np.ndarray, log_likelihoods: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reweight and resample inside each flag candidate, never across them.

        The flag candidate is a *static* hidden variable: nothing in the
        transition ever moves a particle from one candidate to another. Ordinary
        resampling therefore deletes candidates permanently -- one unlucky step
        takes the last particle carrying the true one, and no later reading can
        bring it back, because there is nothing left to resample from. Measured
        on the default field with 200 particles, that is not a rare event: the
        filter commits to a candidate on the first step and stays there whether
        or not it is right.

        Stratifying fixes it without inventing evidence. Each candidate keeps its
        own particles and its own share of the posterior; resampling happens
        inside a candidate, so a candidate's particle count never changes and
        only its weight moves. A candidate whose particles are *all* ruled out
        carries its weight forward unchanged rather than dropping to zero: that
        reading says none of its fifty red hypotheses explain the step, which is
        evidence about those hypotheses, not proof about the candidate.

        Args:
            particles: ``(N, state_size)`` transitioned particles.
            log_likelihoods: ``(N,)`` log-likelihoods of this step's reading.

        Returns:
            ``(particles, log_weights)`` after the stratified update.
        """
        updater: CaptureTheFlagVectorizedUpdater = self.updater
        candidate = particles[:, updater.layout.flag_cell].astype(np.int64)
        weights = self.normalized_weights * np.exp(
            log_likelihoods - np.max(log_likelihoods[np.isfinite(log_likelihoods)], initial=0.0)
        )

        resampled = np.array(particles, dtype=np.float64, copy=True)
        posterior = np.zeros(len(particles), dtype=np.float64)
        for index in range(len(updater.red_flag_candidates)):
            rows = np.flatnonzero(candidate == index)
            if rows.size == 0:
                continue
            prior_mass = float(self.normalized_weights[rows].sum())
            group = weights[rows]
            total = float(group.sum())
            if total <= 0.0:
                posterior[rows] = prior_mass / rows.size
                continue
            if self.resampling:
                drawn = np.random.choice(rows, size=rows.size, p=group / total)
                resampled[rows] = particles[drawn]
            posterior[rows] = total / rows.size

        posterior = posterior / posterior.sum()
        with np.errstate(divide="ignore"):
            return resampled, np.log(posterior)


def _write_observed_components(
    particles: np.ndarray, values: np.ndarray, updater: CaptureTheFlagVectorizedUpdater
) -> np.ndarray:
    """Overwrite every particle's exactly-observed components with the reading.

    Args:
        particles: ``(N, state_size)`` particles.
        values: The reading.
        updater: The updater, for the layout and the team sizes.

    Returns:
        ``(N, state_size)`` particles agreeing with the reading on everything it
        reports exactly. The red positions and the flag candidate -- the hidden
        state -- are untouched.
    """
    layout = updater.layout
    conditioned = np.array(particles, dtype=np.float64, copy=True)
    conditioned[:, layout.blue_pos : layout.blue_pos + 2 * updater.n_blue] = values[
        : 2 * updater.n_blue
    ]

    suffix = values[-(updater.n_blue + 4) :]
    conditioned[:, layout.carrier_red_flag] = suffix[0]
    # The reading reports only *whether* red carries the blue flag, not which
    # red player does, so the carrier id is left alone when it agrees with the
    # bit and cleared when it does not.
    carrying = conditioned[:, layout.carrier_blue_flag] != 0.0
    conditioned[:, layout.carrier_blue_flag] = np.where(
        suffix[1] != 0.0,
        np.where(carrying, conditioned[:, layout.carrier_blue_flag], 1.0),
        0.0,
    )
    for index in range(updater.n_blue):
        conditioned[:, layout.freeze_blue + index] = suffix[2 + index]
    conditioned[:, layout.score_blue] = suffix[-2]
    conditioned[:, layout.score_red] = suffix[-1]
    return conditioned


def _manhattan(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Grid L1 distance between two broadcastable ``(..., 2)`` cell arrays."""
    return np.abs(left - right).sum(axis=-1).astype(np.int64)


def _flag_index(flag: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Which candidate each particle's flag cell is, as an index."""
    matches = np.all(flag[:, None, :] == candidates[None, :, :], axis=2)
    return np.argmax(matches, axis=1)


def _at_cell(cells: np.ndarray, index: np.ndarray, cell: Tuple[int, int]) -> np.ndarray:
    """Whether the ``index``-th player of each particle stands on ``cell``.

    An index of ``-1`` means "nobody is carrying", which the caller has already
    excluded with its own mask; it is clipped here so the gather stays in range.
    """
    safe = np.clip(index, 0, cells.shape[1] - 1)
    chosen = np.take_along_axis(cells, safe[:, None, None], axis=1)[:, 0, :]
    return np.all(chosen == np.asarray(cell, dtype=np.float64), axis=1)


def _draw_index(weights: np.ndarray) -> np.ndarray:
    """Draw one column index per row, in proportion to ``weights``."""
    cumulative = np.cumsum(weights, axis=1)
    draws = np.random.random(len(weights))[:, None] * cumulative[:, -1:]
    return np.argmax(cumulative > draws, axis=1)


def _log(values: np.ndarray) -> np.ndarray:
    """Natural log, with zero mapped to ``-inf`` rather than a warning."""
    with np.errstate(divide="ignore"):
        return np.log(values)


def create_capture_the_flag_belief(
    env: "CaptureTheFlagPOMDP",
    belief_type: BeliefType = BeliefType.VECTORIZED_PARTICLE,
    n_particles: int = 200,
    **kwargs: Any,
) -> "Belief":
    """Create a belief for the CaptureTheFlag POMDP.

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
        raise ValueError(f"CaptureTheFlagPOMDP does not support belief type {belief_type!r}")

    particles = np.stack(env.initial_state_dist().sample(n_samples=n_particles))
    log_weights = np.full(n_particles, -float(np.log(n_particles)), dtype=np.float64)
    return CaptureTheFlagVectorizedBelief(
        particles=particles,
        log_weights=log_weights,
        updater=CaptureTheFlagVectorizedUpdater.from_environment(env),
        resampling=True,
    )


__all__ = [
    "CaptureTheFlagVectorizedBelief",
    "CaptureTheFlagVectorizedUpdater",
    "create_capture_the_flag_belief",
]
