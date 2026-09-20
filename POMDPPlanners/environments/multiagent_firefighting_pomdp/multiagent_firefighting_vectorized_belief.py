# SPDX-License-Identifier: MIT

"""Vectorized particle belief for the multi-agent firefighting POMDP.

The robots see the exact poses, tanks and healths of all of them, plus a noisy
category for every cell within sensing range of a live robot. What they never
see is the wind, which is drawn once at reset and held for the episode: it has
to be inferred from which neighbours catch. So the hidden state is one static
eight-valued variable and the categories of the cells nobody is looking at, and
a belief here is a filter over whole fire maps.

That makes the per-particle cost a grid, not a scalar, which is why this
updater exists: every stage of the transition -- motion, suppression, spread,
growth and burnout, heat damage -- runs over the particle axis and the grid at
once, in the order the environment fixes. The order is the model. Suppression
before spread is what lets a robot stop a front by soaking the cell ahead of it
in the same step, and growth on the cells that were alight *before* the spread
is what stops a cell igniting and growing to burning inside one step.

Two things the belief does beyond the plain filter, both for the same reason --
the reading is sharp enough to empty a naive particle set:

* the poses, tanks and healths are reported exactly and depend on hidden state
  (heat damage is read off a cell the robot may not have seen), so a particle
  that disagrees with them is impossible. :class:`FirefightingVectorizedBelief`
  writes the reported values onto every particle instead of weighting by them;
* the wind is static, so ordinary resampling deletes wind values permanently --
  one unlucky step takes the last particle carrying the true one and no later
  evidence can bring it back. Resampling therefore happens inside a wind value,
  never across.

Classes:
    FirefightingVectorizedUpdater: Batched transition and likelihood.
    FirefightingVectorizedBelief: The belief, conditioned and wind-stratified.

Functions:
    create_firefighting_belief: Factory used by the top-level belief factory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Tuple

import numpy as np

from POMDPPlanners.core.belief.belief_utils import get_initial_belief
from POMDPPlanners.core.belief.running_episode_conditioning import (
    condition_log_weights_on_a_running_episode,
)
from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.multiagent_firefighting_pomdp.multiagent_firefighting_pomdp import (
    UNKNOWN_CATEGORY,
)
from POMDPPlanners.environments.multiagent_firefighting_pomdp.multiagent_firefighting_world import (
    DIRECTION_OFFSETS,
    HEAT_DAMAGE,
    NUM_CATEGORIES,
    ROBOT_FIELD_WIDTH,
    ROBOT_OFFSET,
    STEP_INDEX,
    FireCategory,
    FirefightingAction,
    WindStrength,
)
from POMDPPlanners.utils.belief_factory import BeliefType
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.core.belief.base_belief import Belief
    from POMDPPlanners.environments.multiagent_firefighting_pomdp.multiagent_firefighting_pomdp import (
        MultiAgentFirefightingPOMDP,
    )


class FirefightingVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Batched transition and observation likelihood for the firefighting POMDP.

    Attributes:
        num_rows: Grid rows.
        num_cols: Grid columns.
        num_robots: Robots on the grid.
        sensing_radius: Chebyshev radius each live robot sees.
        observation_error_probability: Chance a visible cell is misreported.
    """

    # pylint: disable=too-many-instance-attributes  # one world's rules, spelled out

    def __init__(self, env: "MultiAgentFirefightingPOMDP") -> None:
        """Initialize the updater from an environment.

        Built from the environment rather than from a parameter list: the model
        reads two dozen rule parameters and several derived tables, and naming
        them twice would be that many chances for the copy to drift.

        Args:
            env: The environment whose rules to reproduce.
        """
        self.num_rows = int(env.num_rows)
        self.num_cols = int(env.num_cols)
        self.num_cells = int(env.num_cells)
        self.num_robots = int(env.num_robots)
        self.max_tank = int(env.max_tank)
        self.depot_cell = (int(env.depot_cell[0]), int(env.depot_cell[1]))
        self.slip_probability = float(env.slip_probability)
        self.growth_probability = float(env.growth_probability)
        self.burnout_probability = float(env.burnout_probability)
        self.spread_probability = float(env.spread_probability)
        self.crosswind_attenuation = float(env.crosswind_attenuation)
        self.wind_gain_low = float(env.wind_gain_low)
        self.wind_gain_high = float(env.wind_gain_high)
        self.sensing_radius = int(env.sensing_radius)
        self.observation_error_probability = float(env.observation_error_probability)
        self.wind_direction_index = int(env.wind_direction_index)
        self.wind_strength_index = int(env.wind_strength_index)
        self.fire_offset = int(env.fire_offset)
        self.state_size = int(env.state_size)
        self.observation_size = int(env.observation_size)
        self.robot_block = ROBOT_FIELD_WIDTH * self.num_robots
        self.is_all_robots_disabled_terminal = bool(env.is_all_robots_disabled_terminal)

        self.obstacle_mask = np.asarray(env.obstacle_mask, dtype=bool)
        self._action_table = np.asarray(
            env._action_table, dtype=np.int64
        )  # pylint: disable=protected-access
        self._suppression_probabilities = np.asarray(
            env._suppression_probabilities, dtype=np.float64  # pylint: disable=protected-access
        )
        self._heat_damage = np.asarray(HEAT_DAMAGE, dtype=np.int64)

    @classmethod
    def from_environment(
        cls, env: "MultiAgentFirefightingPOMDP"
    ) -> "FirefightingVectorizedUpdater":
        """Construct an updater from a :class:`MultiAgentFirefightingPOMDP`."""
        return cls(env)

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: Any) -> np.ndarray:
        """Advance every particle one step under one joint action.

        Args:
            particles: ``(N, state_size)`` particles.
            action: The joint action id.

        Returns:
            ``(N, state_size)`` successors. The wind is copied unchanged, which
            is what makes it identifiable from a whole episode's spread pattern.
        """
        states = np.asarray(particles, dtype=np.float64).reshape(-1, self.state_size)
        successors = np.array(states, copy=True)
        if states.shape[0] == 0:
            return successors
        actions = self._action_table[int(np.asarray(action).ravel()[0])]

        robots = self.robot_block_view(states)
        fire = self.fire_maps(states)

        positions = self._move(robots, actions, fire)
        tanks = self._suppress(robots, actions, positions, fire)
        fire_after_suppression = np.array(fire, copy=True)
        self._spread(fire, fire_after_suppression, states)
        self._grow_and_burn_out(fire, fire_after_suppression)
        healths = self._heat(robots, positions, fire)

        successors[:, STEP_INDEX] = states[:, STEP_INDEX] + 1.0
        block = np.empty((len(states), self.num_robots, ROBOT_FIELD_WIDTH), dtype=np.float64)
        block[:, :, 0] = positions[:, :, 0]
        block[:, :, 1] = positions[:, :, 1]
        block[:, :, 2] = tanks
        block[:, :, 3] = healths
        successors[:, ROBOT_OFFSET : ROBOT_OFFSET + self.robot_block] = block.reshape(
            len(states), -1
        )
        successors[:, self.fire_offset :] = fire.reshape(len(states), -1).astype(np.float64)
        return successors

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: Any,
        observation: Any,
    ) -> np.ndarray:
        """Score one reading against every particle.

        The poses, tanks and healths are reported exactly, so any mismatch is
        ``-inf``. Every visible cell contributes its confusion probability, and
        a cell nobody is looking at contributes nothing -- but a *category*
        reported for such a cell is impossible, because the unknown marker is
        deterministic given the poses.

        Args:
            next_particles: ``(N, state_size)`` transitioned particles.
            action: Ignored; the reading depends on the successor alone.
            observation: The reading.

        Returns:
            ``(N,)`` log-likelihoods.
        """
        del action
        states = np.asarray(next_particles, dtype=np.float64).reshape(-1, self.state_size)
        values = np.asarray(observation, dtype=np.float64).ravel()
        if values.size != self.observation_size:
            return np.full(states.shape[0], -np.inf, dtype=np.float64)

        exact = np.all(
            states[:, ROBOT_OFFSET : ROBOT_OFFSET + self.robot_block] == values[: self.robot_block],
            axis=1,
        )
        return np.where(exact, self.cell_log_likelihood(states, values), -np.inf)

    def cell_log_likelihood(self, next_particles: np.ndarray, observation: Any) -> np.ndarray:
        """Score only the cell categories of a reading.

        The exactly-reported robot fields are a delta factor, so they contribute
        either zero or ``-inf``, and :class:`FirefightingVectorizedBelief`
        handles them by conditioning rather than by weighting. This is what it
        weights with.

        Args:
            next_particles: ``(N, state_size)`` transitioned particles.
            observation: The reading.

        Returns:
            ``(N,)`` log-likelihoods of the cell reports alone.
        """
        states = np.asarray(next_particles, dtype=np.float64).reshape(-1, self.state_size)
        values = np.asarray(observation, dtype=np.float64).ravel()
        reported = values[self.robot_block :]
        truth = np.rint(states[:, self.fire_offset :]).astype(np.int64)

        # The visible set follows from the poses, which every particle shares
        # once the belief has conditioned on them -- but it is recomputed per
        # particle so this entry point is also correct for a caller that has not.
        visible = self.visible_masks(states)
        seen = reported[None, :] != UNKNOWN_CATEGORY

        # A category reported where nobody is looking, or the unknown marker
        # where somebody is, contradicts the poses outright.
        impossible = np.any(seen != visible, axis=1)
        legal = np.all(
            (reported == UNKNOWN_CATEGORY)
            | ((reported >= 0) & (reported < NUM_CATEGORIES) & (reported % 1.0 == 0.0))
        )

        error = self.observation_error_probability
        with np.errstate(divide="ignore"):
            log_correct = float(np.log1p(-error))
            log_wrong = float(np.log(error / (NUM_CATEGORIES - 1)))

        matches = np.count_nonzero(visible & seen & (truth == np.rint(reported)[None, :]), axis=1)
        mismatches = np.count_nonzero(visible & seen, axis=1) - matches
        # Each term is added only when its count is non-zero. Both endpoints of
        # ``observation_error_probability`` are legal: at 0 the wrong-category
        # log is ``-inf`` and at 1 the correct-category log is, and ``0 * -inf``
        # is a NaN that would silently poison a particle weight.
        scores = np.where(matches > 0, matches * log_correct, 0.0) + np.where(
            mismatches > 0, mismatches * log_wrong, 0.0
        )
        if not legal:
            return np.full(states.shape[0], -np.inf, dtype=np.float64)
        return np.where(impossible, -np.inf, scores)

    def ruled_out_by_a_running_episode(self, next_particles: np.ndarray) -> np.ndarray:
        """Which particles the robots being asked to act again have ruled out.

        This environment needs the conditioning because "the fire is out" is
        terminal *and* invisible. A robot sees only the cells inside its
        sensing radius; every other cell reports ``UNKNOWN``, so a reading
        taken while three cells burn out of sight is identical to one taken
        over a grid that is already cold. Burnt and wet cells never re-ignite,
        so a particle whose last alight cell goes out can never come back, and
        :meth:`cell_log_likelihood` has nothing to contradict it with. Measured
        over random episodes on the pinned environment, up to 1.0000 of the
        belief's weight sat on "the fire is out" while it was still burning,
        and once there it stayed for the rest of the episode.

        The step limit is deliberately left out. Every particle carries the
        same counter, so it rules out all of them or none, and a factor
        identical across particles cancels in the posterior; conditioning on it
        would empty the belief on the final step for nothing.

        Args:
            next_particles: ``(N, state_size)`` transitioned particles.

        Returns:
            A boolean mask over ``next_particles``.
        """
        fire = self.fire_maps(next_particles).reshape(len(next_particles), -1)
        alight = (fire == int(FireCategory.SMOLDERING)) | (fire == int(FireCategory.BURNING))
        ruled_out = ~np.any(alight, axis=1)
        if self.is_all_robots_disabled_terminal:
            health = self.robot_block_view(next_particles)[:, :, 3]
            ruled_out = ruled_out | ~np.any(health > 0, axis=1)
        return ruled_out

    @property
    def config_id(self) -> str:
        """Return a deterministic identifier for this updater configuration."""
        return config_to_id(
            {
                "class": "FirefightingVectorizedUpdater",
                "is_all_robots_disabled_terminal": self.is_all_robots_disabled_terminal,
                "num_rows": self.num_rows,
                "num_cols": self.num_cols,
                "num_robots": self.num_robots,
                "max_tank": self.max_tank,
                "depot_cell": list(self.depot_cell),
                "obstacles": np.flatnonzero(self.obstacle_mask.ravel()).tolist(),
                "slip_probability": self.slip_probability,
                "growth_probability": self.growth_probability,
                "burnout_probability": self.burnout_probability,
                "spread_probability": self.spread_probability,
                "crosswind_attenuation": self.crosswind_attenuation,
                "wind_gain_low": self.wind_gain_low,
                "wind_gain_high": self.wind_gain_high,
                "sensing_radius": self.sensing_radius,
                "observation_error_probability": self.observation_error_probability,
                "suppression_probabilities": self._suppression_probabilities.tolist(),
            }
        )

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def robot_block_view(self, states: np.ndarray) -> np.ndarray:
        """``(N, num_robots, 4)`` robot fields. A copy, not a view."""
        return states[:, ROBOT_OFFSET : ROBOT_OFFSET + self.robot_block].reshape(
            -1, self.num_robots, ROBOT_FIELD_WIDTH
        )

    def fire_maps(self, states: np.ndarray) -> np.ndarray:
        """``(N, num_rows, num_cols)`` integer fire maps. A copy, not a view."""
        return (
            np.rint(states[:, self.fire_offset :])
            .astype(np.int64)
            .reshape(-1, self.num_rows, self.num_cols)
        )

    def visible_masks(self, states: np.ndarray) -> np.ndarray:
        """``(N, num_cells)`` mask of the cells some live robot can see.

        Two robots standing together see barely more than one, so spreading out
        is what buys information. A disabled robot sees nothing.
        """
        robots = self.robot_block_view(states)
        rows = np.arange(self.num_rows)[None, :, None]
        cols = np.arange(self.num_cols)[None, None, :]
        visible = np.zeros((len(states), self.num_rows, self.num_cols), dtype=bool)
        for robot in range(self.num_robots):
            live = robots[:, robot, 3] > 0
            near_row = np.abs(rows - robots[:, robot, 0][:, None, None]) <= self.sensing_radius
            near_col = np.abs(cols - robots[:, robot, 1][:, None, None]) <= self.sensing_radius
            visible |= live[:, None, None] & near_row & near_col
        return visible.reshape(len(states), -1)

    # ------------------------------------------------------------------
    # Transition stages
    # ------------------------------------------------------------------

    def _move(self, robots: np.ndarray, actions: np.ndarray, fire: np.ndarray) -> np.ndarray:
        """Stage 1: motion, resolved against the pre-step fire map.

        A disabled robot, a suppressing robot and a robot whose target is
        inadmissible all stay put; the slip draw is taken only where a move
        could succeed.
        """
        positions = np.array(robots[:, :, :2], copy=True)
        for robot in range(self.num_robots):
            action = int(actions[robot])
            if action == int(FirefightingAction.SUPPRESS):
                continue
            offset = DIRECTION_OFFSETS[action]
            target = positions[:, robot, :] + np.asarray(offset, dtype=np.float64)
            admissible = self._is_admissible(target, fire) & (robots[:, robot, 3] > 0)
            moved = admissible & (np.random.random(len(positions)) >= self.slip_probability)
            positions[:, robot, :] = np.where(moved[:, None], target, positions[:, robot, :])
        return positions

    def _is_admissible(self, target: np.ndarray, fire: np.ndarray) -> np.ndarray:
        """Whether each particle's target cell may be entered."""
        rows = np.rint(target[:, 0]).astype(np.int64)
        cols = np.rint(target[:, 1]).astype(np.int64)
        inside = (rows >= 0) & (rows < self.num_rows) & (cols >= 0) & (cols < self.num_cols)
        safe_rows = np.clip(rows, 0, self.num_rows - 1)
        safe_cols = np.clip(cols, 0, self.num_cols - 1)
        free = ~self.obstacle_mask[safe_rows, safe_cols]
        unburnt = fire[np.arange(len(fire)), safe_rows, safe_cols] != int(FireCategory.BURNT)
        return inside & free & unburnt

    def _suppress(
        self,
        robots: np.ndarray,
        actions: np.ndarray,
        positions: np.ndarray,
        fire: np.ndarray,
    ) -> np.ndarray:
        """Stage 2: suppression, then the tank, with a refill overriding the cost.

        Each spraying robot covers its own cell and its four neighbours and the
        counts add, so two robots covering one cell each get an independent
        attempt at soaking it. A robot that sprays and steps into the depot on
        the same step ends the step full.
        """
        tanks = np.array(robots[:, :, 2], copy=True)
        counts = np.zeros((len(fire), self.num_rows, self.num_cols), dtype=np.int64)
        for robot in range(self.num_robots):
            if int(actions[robot]) != int(FirefightingAction.SUPPRESS):
                continue
            sprays = (robots[:, robot, 3] > 0) & (robots[:, robot, 2] > 0)
            if not np.any(sprays):
                continue
            rows = np.rint(positions[:, robot, 0]).astype(np.int64)
            cols = np.rint(positions[:, robot, 1]).astype(np.int64)
            for offset_row, offset_col in ((0, 0),) + tuple(DIRECTION_OFFSETS):
                target_row = rows + offset_row
                target_col = cols + offset_col
                inside = (
                    (target_row >= 0)
                    & (target_row < self.num_rows)
                    & (target_col >= 0)
                    & (target_col < self.num_cols)
                )
                hit = np.flatnonzero(sprays & inside)
                np.add.at(counts, (hit, target_row[hit], target_col[hit]), 1)
            tanks[:, robot] = np.where(sprays, tanks[:, robot] - 1.0, tanks[:, robot])

        probability = 1.0 - np.power(1.0 - self._suppression_probabilities[fire], counts)
        soaked = (counts > 0) & (np.random.random(fire.shape) < probability)
        fire[soaked] = int(FireCategory.WET)

        at_depot = (np.rint(positions[:, :, 0]) == self.depot_cell[0]) & (
            np.rint(positions[:, :, 1]) == self.depot_cell[1]
        )
        return np.where(at_depot, float(self.max_tank), tanks)

    def _spread(
        self, fire: np.ndarray, fire_after_suppression: np.ndarray, states: np.ndarray
    ) -> None:
        """Stage 3: spread, read from the map that survived the hoses.

        A cell catches unless *every* alight neighbour fails to ignite it, so
        the per-cell probability is one minus a product over the alight
        four-neighbours: the one sitting directly upwind contributes the boosted
        rate and the other three the attenuated one.
        """
        direction = np.rint(states[:, self.wind_direction_index]).astype(np.int64)
        strength = np.rint(states[:, self.wind_strength_index]).astype(np.int64)
        gain = np.where(strength == int(WindStrength.HIGH), self.wind_gain_high, self.wind_gain_low)
        downwind_rate = np.minimum(1.0, self.spread_probability * gain)
        crosswind_rate = self.spread_probability * (1.0 - self.crosswind_attenuation)

        alight = (fire_after_suppression == int(FireCategory.SMOLDERING)) | (
            fire_after_suppression == int(FireCategory.BURNING)
        )
        survive = np.ones(fire.shape, dtype=np.float64)
        for code, (offset_row, offset_col) in enumerate(DIRECTION_OFFSETS):
            shifted = np.zeros_like(alight)
            rows = slice(max(0, offset_row), self.num_rows + min(0, offset_row))
            cols = slice(max(0, offset_col), self.num_cols + min(0, offset_col))
            source_rows = slice(max(0, -offset_row), self.num_rows + min(0, -offset_row))
            source_cols = slice(max(0, -offset_col), self.num_cols + min(0, -offset_col))
            shifted[:, rows, cols] = alight[:, source_rows, source_cols]
            rate = np.where(direction == code, downwind_rate, crosswind_rate)[:, None, None]
            survive = np.where(shifted, survive * (1.0 - rate), survive)

        ignition = 1.0 - survive
        ignitable = (fire == int(FireCategory.UNBURNT)) & ~self.obstacle_mask[None, :, :]
        ignited = ignitable & (ignition > 0.0) & (np.random.random(fire.shape) < ignition)
        fire[ignited] = int(FireCategory.SMOLDERING)

    def _grow_and_burn_out(self, fire: np.ndarray, fire_after_suppression: np.ndarray) -> None:
        """Stage 4: growth and burnout, on the cells alight before the spread."""
        smoldering = fire_after_suppression == int(FireCategory.SMOLDERING)
        grown = smoldering & (np.random.random(fire.shape) < self.growth_probability)
        fire[grown] = int(FireCategory.BURNING)

        burning = fire_after_suppression == int(FireCategory.BURNING)
        burnt = burning & (np.random.random(fire.shape) < self.burnout_probability)
        fire[burnt] = int(FireCategory.BURNT)

    def _heat(self, robots: np.ndarray, positions: np.ndarray, fire: np.ndarray) -> np.ndarray:
        """Stage 5: heat damage, read off the final map at each robot's final cell."""
        healths = np.array(robots[:, :, 3], copy=True)
        index = np.arange(len(fire))
        for robot in range(self.num_robots):
            rows = np.rint(positions[:, robot, 0]).astype(np.int64)
            cols = np.rint(positions[:, robot, 1]).astype(np.int64)
            damage = self._heat_damage[fire[index, rows, cols]]
            healths[:, robot] = np.maximum(healths[:, robot] - damage, 0.0)
        return healths


class FirefightingVectorizedBelief(VectorizedWeightedParticleBelief):
    """Vectorized belief that conditions on the robot fields and stratifies by wind.

    See the module docstring for why both are needed. In short: the robot fields
    are reported without noise but depend on hidden state, so weighting by them
    empties the particle set; and the wind never changes, so resampling across
    wind values throws away hypotheses that no later evidence can restore.
    """

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Any = None,
        state: Any = None,
    ) -> "FirefightingVectorizedBelief":
        """Step, condition on the reported robot fields, reweight, resample by wind.

        Args:
            action: The joint action that was executed.
            observation: This step's reading.
            pomdp: Unused; the updater carries the rules.
            state: Never read, so the true state cannot leak into the belief.
                Its *presence* is read: only the episode driver passes it, and
                that is what makes this a real filter step rather than a
                planner's hypothetical, which is the difference between the
                episode having continued being evidence and its being one of
                the outcomes the search still has to weigh. See
                :class:`VectorizedWeightedParticleBelief`.

        Returns:
            The posterior belief.
        """
        del pomdp
        updater: FirefightingVectorizedUpdater = self.updater
        values = np.asarray(observation, dtype=np.float64).ravel()
        next_particles = updater.batch_transition(self.particles, action)

        if values.size == updater.observation_size:
            next_particles[:, ROBOT_OFFSET : ROBOT_OFFSET + updater.robot_block] = values[
                : updater.robot_block
            ]
            log_likelihoods = updater.cell_log_likelihood(next_particles, values)
        else:
            log_likelihoods = updater.batch_observation_log_likelihood(
                next_particles, action, values
            )

        if state is not None:
            # Conditioned on the likelihoods rather than on the posterior: the
            # stratified update below carries a wind value forward untouched
            # when all of its particles score zero, so a stratum whose every
            # particle had gone cold would otherwise keep its full weight and
            # the conditioning would be undone one stratum at a time.
            ruled_out = updater.ruled_out_by_a_running_episode(next_particles)
            # Per wind value, not over the population: _stratified_update
            # carries a wind value forward at its prior weight when all of its
            # particles score zero, so flooring a whole stratum preserves it
            # intact and the conditioning is undone one stratum at a time.
            # A stratum with nothing left to condition on is rebuilt instead.
            for rows in self._wind_strata(next_particles):
                if ruled_out[rows].all():
                    self._reignite_out_of_sight(next_particles, values, rows)
                    ruled_out[rows] = False
            log_likelihoods, _ = condition_log_weights_on_a_running_episode(
                log_likelihoods, ruled_out
            )

        particles, log_weights = self._stratified_update(next_particles, log_likelihoods)
        return FirefightingVectorizedBelief(
            particles=particles,
            log_weights=log_weights,
            updater=self.updater,
            resampling=self.resampling,
            ess_factor=self.ess_factor,
        )

    def _wind_strata(self, particles: np.ndarray) -> List[np.ndarray]:
        """Row indices of each wind value present, the unit this belief resamples in.

        Args:
            particles: ``(N, state_size)`` particles.

        Returns:
            One index array per distinct wind value.
        """
        updater: FirefightingVectorizedUpdater = self.updater
        wind = np.rint(particles[:, updater.wind_direction_index]).astype(np.int64) * 2 + np.rint(
            particles[:, updater.wind_strength_index]
        ).astype(np.int64)
        return [np.flatnonzero(wind == value) for value in np.unique(wind)]

    def _reignite_out_of_sight(
        self, particles: np.ndarray, observation: np.ndarray, rows: np.ndarray
    ) -> None:
        """Put the fire back somewhere nobody is looking, in place.

        Called for a wind value whose every particle has gone cold. Flooring
        those particles would not help: :meth:`_stratified_update` carries a
        wind value forward at its prior weight precisely when all of its
        particles score zero, so the stratum would survive intact and stay
        certain the fire is out. Dropping the stratum instead is worse -- the
        wind never changes, so a wind value deleted once can never be restored,
        which is the reason this belief stratifies at all.

        The repair is the move the reading itself licenses: the robots report a
        category only for the cells they can see, so a cell they cannot see and
        that is not absorbing may be alight, and exactly that is what the step
        being taken says of at least one cell. Each particle gets one such cell
        set smoldering, drawn uniformly from its own admissible cells. Cells
        the reading names are left alone -- they are the only hard evidence the
        step produced -- and so are burnt and wet cells, which no rule maps
        back into being alight.

        A particle with no admissible cell is left as it is. Every cell it
        could hold fire in is either visible and cold or absorbing, so there is
        nothing consistent to re-ignite.

        Args:
            particles: ``(N, state_size)`` particles, modified in place.
            observation: This step's reading.
            rows: The rows to repair.
        """
        updater: FirefightingVectorizedUpdater = self.updater
        fire = particles[:, updater.fire_offset :]
        visible = updater.visible_masks(particles)
        reported = (
            observation[updater.robot_block :]
            if observation.size == updater.observation_size
            else np.full(updater.num_cells, UNKNOWN_CATEGORY)
        )
        unreported = (reported == UNKNOWN_CATEGORY)[None, :]
        absorbing = (np.rint(fire) == int(FireCategory.BURNT)) | (
            np.rint(fire) == int(FireCategory.WET)
        )
        admissible = (~visible | unreported) & ~absorbing

        for row in rows:
            choices = np.flatnonzero(admissible[row])
            if choices.size:
                fire[row, np.random.choice(choices)] = float(FireCategory.SMOLDERING)

    def _stratified_update(
        self, particles: np.ndarray, log_likelihoods: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reweight and resample inside each wind value, never across them.

        A wind value whose particles are *all* ruled out carries its weight
        forward unchanged rather than dropping to zero: that reading says none
        of its fire maps explain the step, which is evidence about those maps,
        not proof about the wind.

        Args:
            particles: ``(N, state_size)`` transitioned particles.
            log_likelihoods: ``(N,)`` log-likelihoods of this step's reading.

        Returns:
            ``(particles, log_weights)`` after the stratified update.
        """
        updater: FirefightingVectorizedUpdater = self.updater
        wind = np.rint(particles[:, updater.wind_direction_index]).astype(np.int64) * 2 + np.rint(
            particles[:, updater.wind_strength_index]
        ).astype(np.int64)
        finite = log_likelihoods[np.isfinite(log_likelihoods)]
        shift = float(np.max(finite)) if finite.size else 0.0
        weights = self.normalized_weights * np.exp(log_likelihoods - shift)

        resampled = np.array(particles, dtype=np.float64, copy=True)
        posterior = np.zeros(len(particles), dtype=np.float64)
        for value in np.unique(wind):
            rows = np.flatnonzero(wind == value)
            total = float(weights[rows].sum())
            if total <= 0.0:
                posterior[rows] = float(self.normalized_weights[rows].sum()) / rows.size
                continue
            if self.resampling:
                drawn = np.random.choice(rows, size=rows.size, p=weights[rows] / total)
                resampled[rows] = particles[drawn]
            posterior[rows] = total / rows.size

        posterior = posterior / posterior.sum()
        with np.errstate(divide="ignore"):
            return resampled, np.log(posterior)


def create_firefighting_belief(
    env: "MultiAgentFirefightingPOMDP",
    belief_type: BeliefType = BeliefType.VECTORIZED_PARTICLE,
    n_particles: int = 200,
    **kwargs: Any,
) -> "Belief":
    """Create a belief for the multi-agent firefighting POMDP.

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
        raise ValueError(
            f"MultiAgentFirefightingPOMDP does not support belief type {belief_type!r}"
        )

    particles = np.stack(env.initial_state_dist().sample(n_samples=n_particles))
    log_weights = np.full(n_particles, -float(np.log(n_particles)), dtype=np.float64)
    return FirefightingVectorizedBelief(
        particles=particles,
        log_weights=log_weights,
        updater=FirefightingVectorizedUpdater.from_environment(env),
        resampling=True,
    )


__all__ = [
    "FirefightingVectorizedBelief",
    "FirefightingVectorizedUpdater",
    "create_firefighting_belief",
]
