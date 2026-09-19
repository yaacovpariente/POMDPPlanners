# SPDX-License-Identifier: MIT

"""Vectorized particle belief updater for the occupancy-grid mapping POMDP.

The scalar filter in :mod:`..occupancy_grid_mapping_belief` calls the
environment once per particle for the predictive score and once per particle
for the map update. Both loops are pure array work over whole-map particles,
so this module does them for all particles in one call:

* :meth:`OccupancyGridMappingVectorizedUpdater.batch_predictive_log_likelihood`
  casts one batched scan over every hidden map and scores the observed ranges
  under the environment's range law and the exact motion probability. It is
  the batched form of ``predictive_observation_log_probability``. The range
  law is selected by ``range_noise_model``: the unbounded Gaussian, or the
  normal truncated to ``[0, +inf)`` whose normaliser depends on each beam's
  nominal range and is therefore scored per beam and per particle.
* :meth:`OccupancyGridMappingVectorizedUpdater.batch_state_from_observation`
  installs the observed pose and scan and applies the inverse sensor update.
  The inverse-sensor delta depends only on the observation and the observed
  pose, never on the hidden map, so it is computed once and added to every
  particle. It is the batched form of ``state_from_observation``.

The two methods of the shared updater interface are also implemented, as the
batched forms of ``sample_next_state`` and ``observation_log_probability``:
a generative transition that draws motion and a fresh noisy scan, and the
point-mass likelihood on the stored scan. They make the updater a faithful
batched copy of the environment's generative model, which is what the shared
equivalence tests check. The conditional filter does not use them, because a
freshly drawn scan matches the observed one with probability zero.

Classes:
    OccupancyGridMappingVectorizedUpdater: Batched updater for the environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_update_rules import (
    OccupancyUpdateRule,
    default_update_rule,
    non_default_update_rule_id,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_sensor import (
    HEADING_STEPS,
    NUM_HEADINGS,
    RangeNoiseModel,
    build_ray_templates,
    quadrature_ranges,
    resolve_range_noise_model,
    sample_ranges,
    scan_log_density,
)
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_pomdp import (
        OccupancyGridMappingPOMDP,
    )

# State layout, duplicated from the environment module rather than imported
# from it so this module does not import the environment at load time.
_STEP_INDEX = 0
_ROW_INDEX = 1
_COL_INDEX = 2
_HEADING_INDEX = 3
_POSE_WIDTH = 4
_FORWARD = 0
_TURN_LEFT = 1
_TURN_RIGHT = 2

_HEADING_STEPS = np.asarray(HEADING_STEPS, dtype=np.int64)


class OccupancyGridMappingVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Batched motion, ray casting and inverse-sensor mapping over map particles.

    Attributes:
        num_rows: Grid rows.
        num_cols: Grid columns.
        num_beams: Beams per scan.
        max_range_cells: Sensor range in cell widths.
        range_noise_std_cells: Per-beam range noise standard deviation.
        range_noise_model: Which per-beam range law that deviation parametrises.
        move_failure_probability: Chance an otherwise-valid forward move fails.
        update_rule: How an observed scan changes a particle's map.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import (
        ...     OccupancyGridMappingPOMDP,
        ... )
        >>> env = OccupancyGridMappingPOMDP()
        >>> updater = OccupancyGridMappingVectorizedUpdater.from_environment(env)
        >>> particles = np.asarray(env.initial_state_dist().sample(8))
        >>> successor = env.sample_next_state(particles[0], 0)
        >>> observation = env.sample_observation(successor, 0)
        >>> updater.batch_predictive_log_likelihood(particles, 0, observation).shape
        (8,)
        >>> updater.batch_state_from_observation(particles, observation).shape
        (8, 228)
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        num_beams: int,
        field_of_view_degrees: float,
        max_range_cells: float,
        range_noise_std_cells: float,
        free_log_odds: float,
        occupied_log_odds: float,
        log_odds_clamp: float,
        move_failure_probability: float,
        sensor_contract_version: int = 2,
        range_noise_model: Union[RangeNoiseModel, str] = RangeNoiseModel.GAUSSIAN,
        update_rule: Optional[OccupancyUpdateRule] = None,
    ):
        """Initialize the updater from the environment's sensor and motion settings.

        Args:
            num_rows: Grid rows.
            num_cols: Grid columns.
            num_beams: Beams per scan.
            field_of_view_degrees: Angular width of the fan, centred on the heading.
            max_range_cells: Sensor range in cell widths.
            range_noise_std_cells: Per-beam range noise standard deviation. Must
                be positive.
            free_log_odds: Increment for a cell a beam passes through. Negative.
            occupied_log_odds: Increment for the cell a beam stops in. Positive.
            log_odds_clamp: Symmetric bound on accumulated log-odds.
            move_failure_probability: Chance an otherwise-valid forward move fails.
            sensor_contract_version: The environment's sensor contract version,
                carried into the identifier so a contract change invalidates caches.
            range_noise_model: Which per-beam range law ``range_noise_std_cells``
                parametrises. Defaults to the unbounded Gaussian, the law every
                updater built before the option existed used. Accepts the
                member or its string value.
            update_rule: How an observed scan changes a particle's map.
                Defaults to ``None``, which rebuilds the environment's original
                rule from ``free_log_odds``, ``occupied_log_odds`` and
                ``log_odds_clamp``. Pass the environment's own rule so the
                batched kernels and the scalar transition cannot disagree;
                :meth:`from_environment` does exactly that.

        Raises:
            ValueError: If ``range_noise_std_cells`` is not positive, or
                ``range_noise_model`` names no known law.
        """
        if range_noise_std_cells <= 0.0:
            raise ValueError(f"range_noise_std_cells must be positive, got {range_noise_std_cells}")
        self.range_noise_model = resolve_range_noise_model(range_noise_model)
        self.num_rows = int(num_rows)
        self.num_cols = int(num_cols)
        self.num_beams = int(num_beams)
        self.field_of_view_degrees = float(field_of_view_degrees)
        self.max_range_cells = float(max_range_cells)
        self.range_noise_std_cells = float(range_noise_std_cells)
        self.free_log_odds = float(free_log_odds)
        self.occupied_log_odds = float(occupied_log_odds)
        self.log_odds_clamp = float(log_odds_clamp)
        self.move_failure_probability = float(move_failure_probability)
        self.sensor_contract_version = int(sensor_contract_version)
        self.update_rule = (
            default_update_rule(self.free_log_odds, self.occupied_log_odds, self.log_odds_clamp)
            if update_rule is None
            else update_rule
        )
        # Read-only from here on, for the same reason as in the environment:
        # ``config_id`` below takes the rule's identifier once per call, but a
        # belief cached under it would outlive an in-place parameter change.
        self.update_rule.freeze()

        self.num_cells = self.num_rows * self.num_cols
        self.map_offset = _POSE_WIDTH
        self.log_odds_offset = _POSE_WIDTH + self.num_cells
        self.scan_offset = _POSE_WIDTH + 2 * self.num_cells
        self.state_size = self.scan_offset + self.num_beams
        self._ray_templates = build_ray_templates(
            num_beams=self.num_beams,
            field_of_view_degrees=self.field_of_view_degrees,
            max_range_cells=self.max_range_cells,
        )

    @classmethod
    def from_environment(
        cls, env: "OccupancyGridMappingPOMDP"
    ) -> "OccupancyGridMappingVectorizedUpdater":
        """Construct an updater that matches ``env``'s sensor and motion model.

        Args:
            env: The environment to copy the settings from.

        Returns:
            A new updater.
        """
        return cls(
            num_rows=env.num_rows,
            num_cols=env.num_cols,
            num_beams=env.num_beams,
            field_of_view_degrees=env.field_of_view_degrees,
            max_range_cells=env.max_range_cells,
            range_noise_std_cells=env.range_noise_std_cells,
            free_log_odds=env.free_log_odds,
            occupied_log_odds=env.occupied_log_odds,
            log_odds_clamp=env.log_odds_clamp,
            move_failure_probability=env.move_failure_probability,
            sensor_contract_version=env.sensor_contract_version,
            range_noise_model=env.range_noise_model,
            update_rule=env.update_rule,
        )

    # ------------------------------------------------------------------
    # Conditional filter kernels
    # ------------------------------------------------------------------

    def batch_predictive_log_likelihood(
        self, particles: np.ndarray, action: int, observation: np.ndarray
    ) -> np.ndarray:
        """Score ``p(observation | particle, action)`` for every particle.

        Integrates the discrete motion outcomes and scores the observed ranges
        under the selected range law against each particle's hidden map,
        before the observation is stored. Matches the environment's scalar
        ``predictive_observation_log_probability`` particle by particle.

        Args:
            particles: State array of shape ``(N, state_size)``.
            action: The action taken.
            observation: ``[row, col, heading, ranges...]``.

        Returns:
            Log-scores of shape ``(N,)``. ``-inf`` where the observed pose is
            unreachable from the particle, or for a malformed observation.
        """
        particles = np.asarray(particles, dtype=np.float64)
        observation = np.asarray(observation, dtype=np.float64)
        count = particles.shape[0]
        if observation.shape != (3 + self.num_beams,) or not np.all(np.isfinite(observation)):
            return np.full(count, -np.inf)
        maps = self._maps(particles)
        scores = np.full(count, -np.inf)
        for rows, cols, headings, probabilities in self._motion_outcomes(particles, action):
            matched = (
                (probabilities > 0.0)
                & (rows.astype(np.float64) == observation[0])
                & (cols.astype(np.float64) == observation[1])
                & (headings.astype(np.float64) == observation[2])
            )
            if not np.any(matched):
                continue
            index = np.flatnonzero(matched)
            nominal = self._nominal_scans(maps[index], rows[index], cols[index], headings[index])
            # Under truncation the normaliser is a function of each beam's
            # nominal range, so it differs across particles and cannot be
            # hoisted into one constant; the helper handles both laws.
            log_density = scan_log_density(
                observation[3:][None, :],
                nominal,
                self.range_noise_std_cells,
                self.range_noise_model,
            )
            scores[index] = np.logaddexp(scores[index], np.log(probabilities[index]) + log_density)
        return scores

    def batch_state_from_observation(
        self, particles: np.ndarray, observation: np.ndarray
    ) -> np.ndarray:
        """Install the observed pose and scan and apply the map update to every particle.

        Matches the environment's scalar ``state_from_observation`` particle by
        particle, including its validation.

        Args:
            particles: State array of shape ``(N, state_size)``.
            observation: ``[row, col, heading, ranges...]``.

        Returns:
            Successor array of shape ``(N, state_size)``.

        Raises:
            ValueError: If the observation is malformed or its pose is not an
                in-grid integer pose.
        """
        particles = np.asarray(particles, dtype=np.float64)
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != (3 + self.num_beams,) or not np.all(np.isfinite(observation)):
            raise ValueError("observation must contain finite pose and ranges")
        row, col, heading = observation[:3]
        if (
            not np.array_equal(observation[:3], np.round(observation[:3]))
            or not 0 <= row < self.num_rows
            or not 0 <= col < self.num_cols
            or not 0 <= heading < NUM_HEADINGS
        ):
            raise ValueError("observation pose must be an in-grid integer pose")
        row, col, heading = int(row), int(col), int(heading)
        end = self.log_odds_offset + self.num_cells
        updated = self.update_rule.shared_update_log_odds(
            particles[:, self.log_odds_offset : end],
            row,
            col,
            heading,
            observation[3:],
            self._ray_templates,
            self.num_rows,
            self.num_cols,
            self.max_range_cells,
        )
        return self._install(particles, observation[None, :3], observation[None, 3:], updated)

    def batch_expected_reward(
        self,
        particles: np.ndarray,
        action: int,
        noise_points: np.ndarray,
        step_cost: float,
    ) -> np.ndarray:
        """Expected entropy reduction of one step from every particle, minus the step cost.

        The batched form of the environment's ``reward`` without a successor:
        for each motion outcome and each fixed noise point, install the noisy
        nominal scan and measure the map entropy afterwards, then average.
        Every hypothetical scan of every particle goes through one batched
        inverse-sensor call, so the cost is a handful of array operations
        rather than ``N * len(noise_points)`` environment calls.

        Args:
            particles: State array of shape ``(N, state_size)``.
            action: The action taken.
            noise_points: ``(K, num_beams)`` unit-variance integration points.
            step_cost: Constant charge per step.

        Returns:
            Expected rewards of shape ``(N,)``.
        """
        particles = np.asarray(particles, dtype=np.float64)
        noise_points = np.asarray(noise_points, dtype=np.float64)
        count = particles.shape[0]
        num_points = noise_points.shape[0]
        end = self.log_odds_offset + self.num_cells
        log_odds = particles[:, self.log_odds_offset : end]
        before = self.entropy_bits_rows(log_odds)
        maps = self._maps(particles)
        after = np.zeros(count)
        for rows, cols, headings, probabilities in self._motion_outcomes(particles, action):
            live = np.flatnonzero(probabilities > 0.0)
            if live.size == 0:
                continue
            nominal = self._nominal_scans(maps[live], rows[live], cols[live], headings[live])
            # Lay the K noisy scans of each particle out as K * n independent
            # particles, so one inverse-sensor call covers all of them. The
            # unit-normal points are mapped through the selected law, so under
            # truncation the integral averages over scans that law can produce.
            ranges = quadrature_ranges(
                nominal[None, :, :],
                noise_points[:, None, :],
                self.range_noise_std_cells,
                self.range_noise_model,
            ).reshape(num_points * len(live), self.num_beams)
            updated = self._updated_log_odds(
                np.tile(log_odds[live], (num_points, 1)),
                ranges,
                np.tile(rows[live], num_points),
                np.tile(cols[live], num_points),
                np.tile(headings[live], num_points),
            )
            entropies = self.entropy_bits_rows(updated).reshape(num_points, len(live))
            after[live] += probabilities[live] * entropies.sum(axis=0)
        return before - after / num_points - float(step_cost)

    @staticmethod
    def entropy_bits_rows(log_odds: np.ndarray) -> np.ndarray:
        """Summed binary entropy of each row of log-odds, in bits.

        Same numerics as the scalar ``grid_entropy_bits``: stable sigmoid,
        probabilities clipped away from 0 and 1, base-2 logs.
        """
        values = np.asarray(log_odds, dtype=np.float64)
        positive = values >= 0.0
        exp_negative_abs = np.exp(-np.abs(values))
        probability = np.where(
            positive, 1.0 / (1.0 + exp_negative_abs), exp_negative_abs / (1.0 + exp_negative_abs)
        )
        clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
        per_cell = -(clipped * np.log2(clipped) + (1.0 - clipped) * np.log2(1.0 - clipped))
        return per_cell.sum(axis=-1)

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface (generative model)
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: Any) -> np.ndarray:
        """Draw motion and a fresh noisy scan for every particle, then map it.

        The batched form of the environment's ``sample_next_state``. Draws from
        the global NumPy stream: one uniform per particle that has two motion
        outcomes, then ``(N, num_beams)`` range draws under the selected law.

        Args:
            particles: State array of shape ``(N, state_size)``.
            action: The action taken, as an int or a 0-d array.

        Returns:
            Successor array of shape ``(N, state_size)``.
        """
        particles = np.asarray(particles, dtype=np.float64)
        count = particles.shape[0]
        outcomes = self._motion_outcomes(particles, action)
        rows, cols, headings, _ = outcomes[0]
        if len(outcomes) == 2:
            stay_rows, stay_cols, stay_headings, stay_probabilities = outcomes[1]
            uncertain = stay_probabilities > 0.0
            if np.any(uncertain):
                draws = np.random.random(int(np.count_nonzero(uncertain)))
                stay = np.zeros(count, dtype=bool)
                stay[uncertain] = draws >= 1.0 - stay_probabilities[uncertain]
                rows = np.where(stay, stay_rows, rows)
                cols = np.where(stay, stay_cols, cols)
                headings = np.where(stay, stay_headings, headings)
        nominal = self._nominal_scans(self._maps(particles), rows, cols, headings)
        ranges = sample_ranges(nominal, self.range_noise_std_cells, self.range_noise_model)
        poses = np.stack([rows, cols, headings], axis=1).astype(np.float64)
        end = self.log_odds_offset + self.num_cells
        updated = self._updated_log_odds(
            particles[:, self.log_odds_offset : end], ranges, rows, cols, headings
        )
        return self._install(particles, poses, ranges, updated)

    def batch_observation_log_likelihood(
        self, next_particles: np.ndarray, action: Any, observation: np.ndarray
    ) -> np.ndarray:
        """Point mass on each particle's exact pose and stored scan.

        The batched form of the environment's ``observation_log_probability``.

        Args:
            next_particles: Successor array of shape ``(N, state_size)``.
            action: Unused; the observation depends only on the successor.
            observation: ``[row, col, heading, ranges...]``.

        Returns:
            ``0.0`` where the observation equals the particle's revealed
            reading, ``-inf`` elsewhere. Shape ``(N,)``.
        """
        del action
        next_particles = np.asarray(next_particles, dtype=np.float64)
        observation = np.asarray(observation, dtype=np.float64)
        count = next_particles.shape[0]
        if observation.shape != (3 + self.num_beams,):
            return np.full(count, -np.inf)
        rows, cols, headings = self._poses(next_particles)
        expected = np.concatenate(
            [
                np.stack([rows, cols, headings], axis=1).astype(np.float64),
                next_particles[:, self.scan_offset :],
            ],
            axis=1,
        )
        return np.where(np.all(expected == observation[None, :], axis=1), 0.0, -np.inf)

    @property
    def config_id(self) -> str:
        """Deterministic identifier of the sensor, motion and mapping settings."""
        rule_id = non_default_update_rule_id(
            self.update_rule, self.free_log_odds, self.occupied_log_odds, self.log_odds_clamp
        )
        return config_to_id(
            {
                "class": type(self).__name__,
                "num_rows": self.num_rows,
                "num_cols": self.num_cols,
                "num_beams": self.num_beams,
                "field_of_view_degrees": self.field_of_view_degrees,
                "max_range_cells": self.max_range_cells,
                "range_noise_std_cells": self.range_noise_std_cells,
                "range_noise_model": self.range_noise_model.value,
                "free_log_odds": self.free_log_odds,
                "occupied_log_odds": self.occupied_log_odds,
                "log_odds_clamp": self.log_odds_clamp,
                "move_failure_probability": self.move_failure_probability,
                "sensor_contract_version": self.sensor_contract_version,
                # Absent for the original rule, which the three log-odds
                # settings above already describe in full, so every belief
                # cached before rules became pluggable still matches.
                **({} if rule_id is None else {"update_rule": rule_id}),
            }
        )

    # ------------------------------------------------------------------
    # Batched geometry
    # ------------------------------------------------------------------

    def _maps(self, particles: np.ndarray) -> np.ndarray:
        """The hidden true maps as an ``(N, num_rows, num_cols)`` view."""
        return particles[:, self.map_offset : self.map_offset + self.num_cells].reshape(
            particles.shape[0], self.num_rows, self.num_cols
        )

    @staticmethod
    def _poses(particles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integer ``(rows, cols, headings)`` of every particle, rounded like ``pose``."""
        rows = np.rint(particles[:, _ROW_INDEX]).astype(np.int64)
        cols = np.rint(particles[:, _COL_INDEX]).astype(np.int64)
        headings = np.rint(particles[:, _HEADING_INDEX]).astype(np.int64) % NUM_HEADINGS
        return rows, cols, headings

    def _next_poses(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        headings: np.ndarray,
        action: int,
        maps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Where each particle's robot ends up, and whether a forward move was blocked."""
        action = int(action)
        count = rows.shape[0]
        if action == _TURN_LEFT:
            return rows, cols, (headings - 1) % NUM_HEADINGS, np.zeros(count, dtype=bool)
        if action == _TURN_RIGHT:
            return rows, cols, (headings + 1) % NUM_HEADINGS, np.zeros(count, dtype=bool)
        steps = _HEADING_STEPS[headings]
        target_rows = rows + steps[:, 0]
        target_cols = cols + steps[:, 1]
        inside = (
            (target_rows >= 0)
            & (target_rows < self.num_rows)
            & (target_cols >= 0)
            & (target_cols < self.num_cols)
        )
        safe_rows = np.where(inside, target_rows, 0)
        safe_cols = np.where(inside, target_cols, 0)
        occupied = maps[np.arange(count), safe_rows, safe_cols] != 0
        blocked = ~inside | occupied
        return (
            np.where(blocked, rows, target_rows),
            np.where(blocked, cols, target_cols),
            headings,
            blocked,
        )

    def _motion_outcomes(
        self, particles: np.ndarray, action: int
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Per-particle motion outcomes as ``(rows, cols, headings, probabilities)``.

        The first entry is the attempted pose; a second entry, present only when
        moves can fail, is the pose the robot keeps on failure. A particle whose
        robot did not move has probability 1 on the first entry and 0 on the
        second, which matches the scalar environment's single-outcome case.
        """
        rows, cols, headings = self._poses(particles)
        next_rows, next_cols, next_headings, _ = self._next_poses(
            rows, cols, headings, action, self._maps(particles)
        )
        if self.move_failure_probability <= 0.0:
            return [(next_rows, next_cols, next_headings, np.ones(rows.shape[0]))]
        moved = (next_rows != rows) | (next_cols != cols)
        failure = np.where(moved, self.move_failure_probability, 0.0)
        return [
            (next_rows, next_cols, next_headings, 1.0 - failure),
            (rows, cols, headings, failure),
        ]

    def _nominal_scans(
        self, maps: np.ndarray, rows: np.ndarray, cols: np.ndarray, headings: np.ndarray
    ) -> np.ndarray:
        """Noise-free ranges from each particle's pose against its own map.

        The batched form of ``cast_scan``, grouped by heading because each
        heading has its own ray template.
        """
        count = maps.shape[0]
        ranges = np.full((count, self.num_beams), self.max_range_cells, dtype=np.float64)
        for heading in np.unique(headings):
            index = np.flatnonzero(headings == heading)
            offsets, template_ranges = self._ray_templates[int(heading)]
            length = offsets.shape[1]
            ray_rows = offsets[None, :, :, 0] + rows[index, None, None]
            ray_cols = offsets[None, :, :, 1] + cols[index, None, None]
            inside = (
                (ray_rows >= 0)
                & (ray_rows < self.num_rows)
                & (ray_cols >= 0)
                & (ray_cols < self.num_cols)
            )
            safe_rows = np.where(inside, ray_rows, 0)
            safe_cols = np.where(inside, ray_cols, 0)
            occupied = (maps[index[:, None, None], safe_rows, safe_cols] != 0) & inside
            stops = occupied | ~inside
            any_stop = stops.any(axis=2)
            stop_slot = np.where(any_stop, np.argmax(stops, axis=2), length)
            clipped = np.minimum(stop_slot, length - 1)
            hit = any_stop & np.take_along_axis(occupied, clipped[:, :, None], axis=2)[:, :, 0]
            hit_ranges = np.take_along_axis(
                np.broadcast_to(template_ranges[None], (len(index), self.num_beams, length)),
                clipped[:, :, None],
                axis=2,
            )[:, :, 0]
            ranges[index] = np.where(hit, hit_ranges, self.max_range_cells)
        return ranges

    def _updated_log_odds(
        self,
        log_odds: np.ndarray,
        observed_ranges: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        headings: np.ndarray,
    ) -> np.ndarray:
        """Apply the map update rule to one scan per particle.

        Used by the generative transition and by the expected-reward integral,
        where every particle has its own reading. Delegates to the environment's
        rule, so the batched path cannot drift from the scalar one.

        Args:
            log_odds: ``(N, num_cells)`` previous log-odds.
            observed_ranges: ``(N, num_beams)`` measured ranges.
            rows: ``(N,)`` robot rows.
            cols: ``(N,)`` robot columns.
            headings: ``(N,)`` heading indices.

        Returns:
            ``(N, num_cells)`` next log-odds.
        """
        return self.update_rule.batch_update_log_odds(
            log_odds,
            rows,
            cols,
            headings,
            observed_ranges,
            self._ray_templates,
            self.num_rows,
            self.num_cols,
            self.max_range_cells,
        )

    def _install(
        self, particles: np.ndarray, poses: np.ndarray, ranges: np.ndarray, log_odds: np.ndarray
    ) -> np.ndarray:
        """Advance the step and write the pose, the scan and the updated map.

        The map arrives already updated and clamped, because the clamp belongs
        to the update rule rather than to this bookkeeping. ``poses`` and
        ``ranges`` broadcast over the particle axis, so one observation shared
        by every particle and one per particle both go through here.
        """
        successors = particles.copy()
        successors[:, _STEP_INDEX] += 1
        successors[:, _ROW_INDEX : _ROW_INDEX + 3] = poses
        successors[:, self.scan_offset :] = ranges
        end = self.log_odds_offset + self.num_cells
        successors[:, self.log_odds_offset : end] = log_odds
        return successors
