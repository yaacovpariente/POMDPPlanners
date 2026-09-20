# SPDX-License-Identifier: MIT

"""Vectorized particle belief for the Chicheck Invaders POMDP.

The batched twin of
:class:`~POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_belief.ChicheckInvadersBelief`.
The scalar filter loops over particles and then over chicken slots inside each
one, which is two nested Python loops per belief update on an environment whose
whole difficulty is that every chicken contributes a likelihood factor on every
step -- reported or not. Here both loops are array axes: particles down, chicken
slots across.

Everything the scalar belief does, this one does:

* the transition, in the environment's own order -- move the ship, resolve the
  hitscan shot, flip the dive coins, move the flock, settle whatever reached
  row 0;
* the likelihood, with a factor for every chicken, because silence from a
  chicken a particle places well inside a sensor is evidence against that
  particle;
* and the reinvigoration, which is what keeps the filter from dying. A
  weight-only filter cannot invent a hypothesis it never held, and two things
  here need one invented: the opening placement is drawn from more cells than a
  few hundred particles cover, and a chicken outside both sensors for several
  steps drifts away from every particle that guessed wrong. So a fraction of the
  particles have their *unreported* chickens re-drawn each step, a wipe-out
  rebuilds the flock from the reading itself, and a fully observable episode
  collapses the belief onto the state it was handed.

Classes:
    ChicheckInvadersVectorizedUpdater: Batched transition and likelihood.
    ChicheckInvadersVectorizedBelief: The belief, with the reinvigoration.

Functions:
    create_chicheck_invaders_vectorized_belief: Factory for the belief factory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from POMDPPlanners.core.belief.running_episode_conditioning import (
    condition_log_weights_on_a_running_episode,
)
from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_pomdp import (
    IMPOSSIBLE_LOG_PROBABILITY,
    ChicheckInvadersAction,
    ObservationMode,
)
from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_schema import (
    CHICKEN_ALIVE,
    CHICKEN_COLUMN,
    CHICKEN_DIRECTION,
    CHICKEN_MODE,
    CHICKEN_ROW,
    CHICKEN_WIDTH,
    COOLDOWN_INDEX,
    MODE_DIVE,
    MODE_PATROL,
    OBSERVATION_CHICKEN_WIDTH,
    OBSERVATION_SHIP_WIDTH,
    OBSERVED_CAMERA_OFFSET,
    OBSERVED_CAMERA_REPORTED,
    OBSERVED_RADAR_DROP,
    OBSERVED_RADAR_REPORTED,
    OBSERVED_RADAR_ROWS,
    OBSERVED_SHIP_COLUMN_INDEX,
    SHIP_COLUMN_INDEX,
    SHIP_HIT_INDEX,
    SHIP_WIDTH,
    STEP_INDEX,
    ruled_out_by_a_running_episode,
)
from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_sensors import (
    camera_sees,
    radar_sees,
    rounded_normal_pmf,
)
from POMDPPlanners.utils.belief_factory import BeliefType
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.core.belief.base_belief import Belief
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_pomdp import (
        ChicheckInvadersPOMDP,
    )


class ChicheckInvadersVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Batched transition and observation likelihood for Chicheck Invaders.

    Attributes:
        num_chickens: Chicken slots per particle.
        num_columns: Grid width.
        num_rows: Grid height.
        dive_probability: Chance a patrolling chicken switches to a dive.
        fire_cooldown: Steps the gun needs between shots.
        observation_mode: Whether the environment hands out the state or a
            sensor reading.
    """

    # pylint: disable=too-many-instance-attributes  # one sensor model, spelled out

    def __init__(
        self,
        num_chickens: int,
        num_columns: int,
        num_rows: int,
        dive_probability: float,
        fire_cooldown: int,
        observation_mode: ObservationMode,
        camera_slope: float,
        radar_radius: float,
        camera_detection_probability: float,
        radar_detection_probability: float,
        camera_offset_noise_std: float,
        radar_range_noise_std: float,
        ship_column_noise_std: float,
        drop_flag_error_probability: float,
    ) -> None:
        """Initialize the updater from the environment's parameters.

        Args:
            num_chickens: Chicken slots per particle.
            num_columns: Grid width.
            num_rows: Grid height.
            dive_probability: Chance a patrolling chicken starts diving.
            fire_cooldown: Steps the gun needs between shots.
            observation_mode: Full or partial observability.
            camera_slope: Half-slope of the camera's upward cone.
            radar_radius: Radius of the radar's disc.
            camera_detection_probability: Chance the camera reports a chicken
                inside its cone.
            radar_detection_probability: Chance the radar reports a chicken
                inside its disc.
            camera_offset_noise_std: Noise on the reported column offset.
            radar_range_noise_std: Noise on the reported row distance.
            ship_column_noise_std: Noise on the ship's own column reading.
            drop_flag_error_probability: Chance the drop flag is wrong.
        """
        self.num_chickens = int(num_chickens)
        self.num_columns = int(num_columns)
        self.num_rows = int(num_rows)
        self.dive_probability = float(dive_probability)
        self.fire_cooldown = int(fire_cooldown)
        self.observation_mode = observation_mode
        self.camera_slope = float(camera_slope)
        self.radar_radius = float(radar_radius)
        self.camera_detection_probability = float(camera_detection_probability)
        self.radar_detection_probability = float(radar_detection_probability)
        self.camera_offset_noise_std = float(camera_offset_noise_std)
        self.radar_range_noise_std = float(radar_range_noise_std)
        self.ship_column_noise_std = float(ship_column_noise_std)
        self.drop_flag_error_probability = float(drop_flag_error_probability)
        self.observation_size = (
            OBSERVATION_SHIP_WIDTH + OBSERVATION_CHICKEN_WIDTH * self.num_chickens
        )

    @classmethod
    def from_environment(cls, env: "ChicheckInvadersPOMDP") -> "ChicheckInvadersVectorizedUpdater":
        """Construct an updater from a :class:`ChicheckInvadersPOMDP` instance."""
        return cls(
            num_chickens=env.num_chickens,
            num_columns=env.num_columns,
            num_rows=env.num_rows,
            dive_probability=env.dive_probability,
            fire_cooldown=env.fire_cooldown,
            observation_mode=env.observation_mode,
            camera_slope=env.camera_slope,
            radar_radius=env.radar_radius,
            camera_detection_probability=env.camera_detection_probability,
            radar_detection_probability=env.radar_detection_probability,
            camera_offset_noise_std=env.camera_offset_noise_std,
            radar_range_noise_std=env.radar_range_noise_std,
            ship_column_noise_std=env.ship_column_noise_std,
            drop_flag_error_probability=env.drop_flag_error_probability,
        )

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: Any) -> np.ndarray:
        """Advance every particle one step under one action.

        The order is the environment's: move the ship, resolve the shot, flip
        the dive coins, move the chickens, then settle whatever reached row 0.
        The shot is resolved before the flock moves, so the ship hits what it
        aimed at rather than where the flock ends up.

        Args:
            particles: ``(N, state_size)`` particles.
            action: A :class:`ChicheckInvadersAction` value.

        Returns:
            ``(N, state_size)`` successors.
        """
        successors = np.array(particles, dtype=np.float64, copy=True)
        if successors.shape[0] == 0:
            return successors
        action_index = int(np.asarray(action).ravel()[0])

        successors[:, STEP_INDEX] += 1.0
        fired = self._fires(successors, action_index)
        ship_column = self._moved_ship_column(successors, action_index)
        successors[:, SHIP_COLUMN_INDEX] = ship_column
        successors[:, COOLDOWN_INDEX] = np.where(
            fired,
            float(self.fire_cooldown),
            np.maximum(successors[:, COOLDOWN_INDEX] - 1.0, 0.0),
        )

        # ``flock_view`` reshapes a non-contiguous column slice, so it is a copy
        # rather than a view and the three in-place steps below have to be
        # written back. Treating it as a view leaves every transition silently
        # unapplied.
        flock = self.flock_view(successors)
        self._resolve_shot(flock, fired, ship_column)
        self._move_flock(flock)
        self._resolve_arrivals(flock, ship_column, successors)
        successors[:, SHIP_WIDTH:] = flock.reshape(len(successors), -1)
        return successors

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: Any,
        observation: Any,
    ) -> np.ndarray:
        """Score one reading against every particle.

        Every chicken contributes a factor whether it was reported or not:
        silence from a chicken a particle places well inside a sensor's reach is
        evidence against that particle, and dropping the unreported slots would
        throw that evidence away.

        Args:
            next_particles: ``(N, state_size)`` transitioned particles.
            action: Ignored; the reading depends on the successor alone.
            observation: The reading.

        Returns:
            ``(N,)`` log-likelihoods, floored at
            :data:`IMPOSSIBLE_LOG_PROBABILITY` the way the environment floors
            them.
        """
        del action
        particles = np.asarray(next_particles, dtype=np.float64)
        reading = np.asarray(observation, dtype=np.float64).ravel()

        if self.observation_mode is ObservationMode.FULL:
            matches = (particles.shape[1] == reading.size) and np.all(
                particles == reading[None, :], axis=1
            )
            return np.where(matches, 0.0, IMPOSSIBLE_LOG_PROBABILITY)
        if reading.size != self.observation_size:
            return np.full(particles.shape[0], IMPOSSIBLE_LOG_PROBABILITY, dtype=np.float64)

        offsets, rows, alive = self._offsets(particles)
        camera = alive & camera_sees(offsets, rows, self.camera_slope)
        radar = alive & radar_sees(offsets, rows, self.radar_radius)
        modes = self.flock_view(particles)[:, :, CHICKEN_MODE]

        slots = reading[OBSERVATION_SHIP_WIDTH:].reshape(
            self.num_chickens, OBSERVATION_CHICKEN_WIDTH
        )
        total = _log(
            rounded_normal_pmf(
                reading[OBSERVED_SHIP_COLUMN_INDEX],
                particles[:, SHIP_COLUMN_INDEX],
                self.ship_column_noise_std,
            )
        )
        total = total + self._camera_log_factor(slots, camera, offsets).sum(axis=1)
        total = total + self._radar_log_factor(slots, radar, rows, modes).sum(axis=1)
        return np.where(total <= IMPOSSIBLE_LOG_PROBABILITY, IMPOSSIBLE_LOG_PROBABILITY, total)

    def ruled_out_by_a_running_episode(self, next_particles: np.ndarray) -> np.ndarray:
        """Which particles the ship being asked to act again has ruled out.

        Nothing in the reading names the ship-hit flag: it carries the ship's
        column and what the two sensors found, and that is all. So the sensor
        likelihood cannot tell a live world from a lost one, and a hit state is
        absorbing -- the flag never clears and the chicken that set it parks on
        row 0 -- so a particle that drifts in is never removed and never moves
        again. Measured over the golden action sequence, the belief went
        entirely certain the ship was already destroyed in 8 of 20 seeds at 60
        particles, 5 of 20 at 400 and 3 of 20 at 2000, while the real episode
        ran on.

        A cleared flock is ruled out for the same reason and by the same
        evidence: it too ends the episode, and it too varies between particles.

        The step limit is deliberately left out. Every particle carries the
        same counter, so it rules out all of them or none, and a factor
        identical across particles cancels in the posterior; flooring on it
        would empty the belief on the final step for nothing.

        Args:
            next_particles: ``(N, state_size)`` transitioned particles.

        Returns:
            A boolean mask over ``next_particles``.
        """
        return ruled_out_by_a_running_episode(next_particles, self.num_chickens)

    @property
    def config_id(self) -> str:
        """Return a deterministic identifier for this updater configuration."""
        return config_to_id(
            {
                "class": "ChicheckInvadersVectorizedUpdater",
                "num_chickens": self.num_chickens,
                "num_columns": self.num_columns,
                "num_rows": self.num_rows,
                "dive_probability": self.dive_probability,
                "fire_cooldown": self.fire_cooldown,
                "observation_mode": self.observation_mode.value,
                "camera_slope": self.camera_slope,
                "radar_radius": self.radar_radius,
                "camera_detection_probability": self.camera_detection_probability,
                "radar_detection_probability": self.radar_detection_probability,
                "camera_offset_noise_std": self.camera_offset_noise_std,
                "radar_range_noise_std": self.radar_range_noise_std,
                "ship_column_noise_std": self.ship_column_noise_std,
                "drop_flag_error_probability": self.drop_flag_error_probability,
            }
        )

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def flock_view(self, particles: np.ndarray) -> np.ndarray:
        """The ``(N, num_chickens, 5)`` chicken block of every particle.

        A copy, not a view: the block is a non-contiguous column slice, so the
        reshape cannot alias the particles. Callers that modify it write it back.
        """
        return particles[:, SHIP_WIDTH:].reshape(-1, self.num_chickens, CHICKEN_WIDTH)

    def _offsets(self, particles: np.ndarray) -> tuple:
        """``(column offsets, row distances, alive mask)``, each ``(N, num_chickens)``."""
        flock = self.flock_view(particles)
        offsets = flock[:, :, CHICKEN_COLUMN] - particles[:, SHIP_COLUMN_INDEX][:, None]
        return offsets, flock[:, :, CHICKEN_ROW], flock[:, :, CHICKEN_ALIVE] > 0.0

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    def _fires(self, particles: np.ndarray, action: int) -> np.ndarray:
        """Whether the gun actually discharges for each particle.

        The cooldown is the only thing that can block it, and it is read from
        the state, so two particles that disagree about it answer differently.
        """
        if action != int(ChicheckInvadersAction.FIRE):
            return np.zeros(particles.shape[0], dtype=bool)
        return particles[:, COOLDOWN_INDEX] <= 0.0

    def _moved_ship_column(self, particles: np.ndarray, action: int) -> np.ndarray:
        """Where the ship ends up, clamped at the walls."""
        delta = 0.0
        if action == int(ChicheckInvadersAction.LEFT):
            delta = -1.0
        elif action == int(ChicheckInvadersAction.RIGHT):
            delta = 1.0
        return np.clip(particles[:, SHIP_COLUMN_INDEX] + delta, 0.0, float(self.num_columns - 1))

    def _resolve_shot(self, flock: np.ndarray, fired: np.ndarray, ship_column: np.ndarray) -> None:
        """Kill the lowest live chicken in the ship's column, in place.

        Lowest rather than nearest-by-slot, so the outcome cannot depend on the
        order the flock happens to be stored in. Ties go to the lower slot, which
        ``argmin`` gives for free.
        """
        if not np.any(fired):
            return
        in_column = (
            (flock[:, :, CHICKEN_ALIVE] > 0.0)
            & (
                flock[:, :, CHICKEN_COLUMN].astype(np.int64)
                == ship_column.astype(np.int64)[:, None]
            )
            & (flock[:, :, CHICKEN_ROW] >= 1.0)
        )
        hittable = np.where(in_column, flock[:, :, CHICKEN_ROW], np.inf)
        target = np.argmin(hittable, axis=1)
        hits = fired & np.any(in_column, axis=1)
        rows = np.flatnonzero(hits)
        flock[rows, target[rows], CHICKEN_ALIVE] = 0.0

    def _move_flock(self, flock: np.ndarray) -> None:
        """Flip the dive coins, then drop the divers and walk the patrollers.

        One coin per slot per particle, alive or not, so the number of random
        numbers a step consumes never depends on how the episode is going.
        """
        coins = np.random.random(flock.shape[:2]) < self.dive_probability
        alive = flock[:, :, CHICKEN_ALIVE] > 0.0
        switching = alive & (flock[:, :, CHICKEN_MODE] == MODE_PATROL) & coins
        flock[:, :, CHICKEN_MODE] = np.where(switching, MODE_DIVE, flock[:, :, CHICKEN_MODE])

        diving = alive & (flock[:, :, CHICKEN_MODE] == MODE_DIVE)
        patrolling = alive & ~diving
        flock[:, :, CHICKEN_ROW] = np.where(
            diving, flock[:, :, CHICKEN_ROW] - 1.0, flock[:, :, CHICKEN_ROW]
        )

        columns = flock[:, :, CHICKEN_COLUMN]
        directions = flock[:, :, CHICKEN_DIRECTION]
        stepped = columns + directions
        # Bouncing reverses the direction *and then* takes the step, so a chicken
        # at a wall turns and moves in one step rather than standing still
        # against it for one.
        bounced = np.where(
            (stepped < 0.0) | (stepped > float(self.num_columns - 1)), -directions, directions
        )
        flock[:, :, CHICKEN_DIRECTION] = np.where(patrolling, bounced, directions)
        flock[:, :, CHICKEN_COLUMN] = np.where(patrolling, columns + bounced, columns)

    def _resolve_arrivals(
        self, flock: np.ndarray, ship_column: np.ndarray, successors: np.ndarray
    ) -> None:
        """Settle every live chicken that reached row 0, in place.

        One in the ship's column destroys it and ends the episode. One anywhere
        else pulls up: back to patrolling at the top row, same column, same
        direction.
        """
        arrived = (flock[:, :, CHICKEN_ALIVE] > 0.0) & (flock[:, :, CHICKEN_ROW] <= 0.0)
        in_column = (
            flock[:, :, CHICKEN_COLUMN].astype(np.int64) == ship_column.astype(np.int64)[:, None]
        )

        struck = arrived & in_column
        flock[:, :, CHICKEN_ROW] = np.where(struck, 0.0, flock[:, :, CHICKEN_ROW])
        successors[:, SHIP_HIT_INDEX] = np.where(
            np.any(struck, axis=1), 1.0, successors[:, SHIP_HIT_INDEX]
        )

        pulled_up = arrived & ~in_column
        flock[:, :, CHICKEN_ROW] = np.where(
            pulled_up, float(self.num_rows - 1), flock[:, :, CHICKEN_ROW]
        )
        flock[:, :, CHICKEN_MODE] = np.where(pulled_up, MODE_PATROL, flock[:, :, CHICKEN_MODE])

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _camera_log_factor(
        self, slots: np.ndarray, in_reach: np.ndarray, truth: np.ndarray
    ) -> np.ndarray:
        """``(N, num_chickens)`` camera factors, in log space."""
        reported = slots[:, OBSERVED_CAMERA_REPORTED][None, :] > 0.0
        readings = slots[:, OBSERVED_CAMERA_OFFSET][None, :]

        seen = _log(self.camera_detection_probability) + _log(
            rounded_normal_pmf(readings, truth, self.camera_offset_noise_std)
        )
        silent = np.full_like(truth, _log(1.0 - self.camera_detection_probability))
        # A report from a chicken the particle places outside the cone is a
        # reading this sensor cannot produce, not an unlikely one.
        out_of_reach = np.where(reported, IMPOSSIBLE_LOG_PROBABILITY, 0.0)
        return np.where(in_reach, np.where(reported, seen, silent), out_of_reach)

    def _radar_log_factor(
        self, slots: np.ndarray, in_reach: np.ndarray, truth: np.ndarray, modes: np.ndarray
    ) -> np.ndarray:
        """``(N, num_chickens)`` radar factors, in log space."""
        reported = slots[:, OBSERVED_RADAR_REPORTED][None, :] > 0.0
        rows = slots[:, OBSERVED_RADAR_ROWS][None, :]
        drop = slots[:, OBSERVED_RADAR_DROP][None, :]

        true_drop = np.where(modes == MODE_DIVE, -1.0, 0.0)
        flag = np.where(
            drop == true_drop,
            1.0 - self.drop_flag_error_probability,
            self.drop_flag_error_probability,
        )
        seen = (
            _log(self.radar_detection_probability)
            + _log(rounded_normal_pmf(rows, truth, self.radar_range_noise_std))
            + _log(flag)
        )
        silent = np.full_like(truth, _log(1.0 - self.radar_detection_probability))
        out_of_reach = np.where(reported, IMPOSSIBLE_LOG_PROBABILITY, 0.0)
        return np.where(in_reach, np.where(reported, seen, silent), out_of_reach)


class ChicheckInvadersVectorizedBelief(VectorizedWeightedParticleBelief):
    """Vectorized particle belief that re-draws the chickens it cannot see.

    Attributes:
        reinvigoration_fraction: Fraction of particles whose unreported chickens
            are re-drawn each step. Small on purpose: a refreshed particle
            carries no evidence from the steps before it, so refreshing too many
            throws away the filter's memory to buy diversity it may not need.
    """

    def __init__(
        self,
        particles: np.ndarray,
        log_weights: np.ndarray,
        updater: ChicheckInvadersVectorizedUpdater,
        resampling: bool = True,
        ess_factor: float = 0.5,
        reinvigoration_fraction: float = 0.1,
    ) -> None:
        """Initialize the belief.

        Args:
            particles: ``(N, state_size)`` particles.
            log_weights: One log-weight per particle.
            updater: The batched updater.
            resampling: Enable ESS-based resampling. Defaults to ``True``.
            ess_factor: ESS threshold as a fraction of N. Defaults to 0.5.
            reinvigoration_fraction: Fraction of particles refreshed each step.
                Defaults to 0.1.
        """
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            updater=updater,
            resampling=resampling,
            ess_factor=ess_factor,
        )
        self.reinvigoration_fraction = float(reinvigoration_fraction)

    @property
    def config_id(self) -> str:
        """Identity covering the particles and the refresh configuration.

        Two beliefs holding identical particles but refreshing different
        fractions of them behave differently on the next step, so the fraction
        belongs in the identity.
        """
        return config_to_id(
            {
                "particles": super().config_id,
                "reinvigoration_fraction": self.reinvigoration_fraction,
            }
        )

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Optional["Environment"] = None,
        state: Optional[Any] = None,
    ) -> "ChicheckInvadersVectorizedBelief":
        """Step, reweight, resample, then re-draw what the sensors did not report.

        Args:
            action: The action that was executed.
            observation: This step's reading.
            pomdp: Unused; the updater carries the observation mode and the
                sensor model.
            state: Never read, so the true state cannot leak into the belief.
                Its *presence* is read: only the episode driver passes it, and
                that is what separates a real filter step -- where the ship
                being asked to act again is evidence it was not destroyed --
                from a planner's hypothetical, where the ship's destruction is
                one of the outcomes the search still has to be able to reach.
                See :class:`VectorizedWeightedParticleBelief`.

        Returns:
            The posterior belief, of this class, so the refresh repeats next
            step.
        """
        del pomdp
        reading = np.asarray(observation, dtype=np.float64).ravel()
        next_particles = self.updater.batch_transition(self.particles, action)
        log_likelihoods = self.updater.batch_observation_log_likelihood(
            next_particles, action, reading
        )
        ruled_out = np.zeros(len(next_particles), dtype=bool)
        if state is not None:
            ruled_out = self.updater.ruled_out_by_a_running_episode(next_particles)
            # Folded into the likelihoods, not just the weights: _reinvigorate
            # reads them to decide whether the population needs rebuilding, and
            # a particle the running episode has ruled out must not count as
            # one that can still explain the reading.
            log_likelihoods, _ = condition_log_weights_on_a_running_episode(
                log_likelihoods, ruled_out
            )
        next_log_weights = self.log_weights + log_likelihoods

        if self.resampling:
            next_particles, next_log_weights = self._resample(next_particles, next_log_weights)

        particles, log_weights = self._reinvigorate(
            next_particles,
            next_log_weights,
            log_likelihoods,
            reading,
            ruled_out,
            state is not None,
        )
        return ChicheckInvadersVectorizedBelief(
            particles=particles,
            log_weights=log_weights,
            updater=self.updater,
            resampling=self.resampling,
            ess_factor=self.ess_factor,
            reinvigoration_fraction=self.reinvigoration_fraction,
        )

    # ------------------------------------------------------------------
    # Reinvigoration
    # ------------------------------------------------------------------

    def _reinvigorate(
        self,
        particles: np.ndarray,
        log_weights: np.ndarray,
        log_likelihoods: np.ndarray,
        reading: np.ndarray,
        ruled_out: np.ndarray,
        episode_is_running: bool,
    ) -> tuple:
        """Collapse onto the truth, rebuild after a wipe-out, or re-draw a few.

        Three cases, in order of how much of the belief they replace. A fully
        observable episode is a point mass on what was observed. A step where
        the particles carry nothing left to resample rebuilds the flock from
        the reading. Every other step re-draws the unreported chickens of a
        small fraction of the particles.

        There are two ways to carry nothing left to resample, and both land in
        the rebuild. The reading may be impossible under every particle. Or the
        running episode may have ruled every particle out -- the conditioning
        cannot floor those, because flooring them all leaves a belief supported
        on nothing, so it stands aside and the rebuild takes over instead.

        Either way, on a real step of the episode the rebuilt particles are
        revived. The rebuild re-seats the flock but carries the ship-hit flag
        forward, and it hands every particle the same uniform weight -- so a
        rebuild that kept the flag would reinstate the very hypothesis the
        conditioning had just floored, at full weight. That is where the
        conditioning alone left up to 0.8167 of the belief on terminal
        particles at 60 particles.

        Args:
            particles: The transitioned particles.
            log_weights: Their log-weights.
            log_likelihoods: This step's conditioned log-likelihoods.
            reading: This step's observation.
            ruled_out: Mask of the particles the running episode ruled out.
            episode_is_running: Whether this is a real step of the episode
                rather than a planner's hypothetical. Only then may a rebuilt
                particle be revived; inside a search the ship's destruction is
                an outcome the tree has to be able to reach.

        Returns:
            ``(particles, log_weights)``.
        """
        updater: ChicheckInvadersVectorizedUpdater = self.updater
        count = len(particles)
        if updater.observation_mode is ObservationMode.FULL:
            return np.tile(reading, (count, 1)), _uniform_log_weights(count)

        if ruled_out.all() or np.all(log_likelihoods <= IMPOSSIBLE_LOG_PROBABILITY):
            # A normalised weight vector looks identical whether every particle
            # was plausible or every particle was impossible, so this is checked
            # rather than inferred from the weights. A population the running
            # episode has emptied is checked separately, because the
            # conditioning leaves those likelihoods untouched: it refuses to
            # floor every particle, so nothing in the weights records it.
            rebuilt = self._rebuild_from_reading(particles, reading)
            if episode_is_running:
                self._revive(rebuilt)
            return rebuilt, _uniform_log_weights(count)

        reported = _reported_slots(reading, updater)
        chosen_count = int(round(self.reinvigoration_fraction * count))
        if chosen_count <= 0 or np.all(reported):
            return particles, log_weights

        chosen = np.random.choice(count, size=min(chosen_count, count), replace=False)
        refreshed = np.array(particles, dtype=np.float64, copy=True)
        self._refresh(refreshed, chosen, reported)
        return refreshed, log_weights

    def _refresh(self, particles: np.ndarray, chosen: np.ndarray, reported: np.ndarray) -> None:
        """Jitter the unreported live chickens of the chosen particles, in place."""
        updater: ChicheckInvadersVectorizedUpdater = self.updater
        flock = updater.flock_view(particles)[chosen]
        touchable = (~reported)[None, :] & (flock[:, :, CHICKEN_ALIVE] > 0.0)

        jitter = np.random.randint(-1, 2, size=flock.shape[:2] + (2,)).astype(np.float64)
        # Clamped rather than wrapped, and never onto row 0: a particle placed on
        # the ship's own row asserts the episode has already been lost, which no
        # jitter should be able to claim.
        columns = np.clip(
            flock[:, :, CHICKEN_COLUMN] + jitter[:, :, 0], 0.0, float(updater.num_columns - 1)
        )
        rows = np.clip(flock[:, :, CHICKEN_ROW] + jitter[:, :, 1], 1.0, float(updater.num_rows - 1))
        flock[:, :, CHICKEN_COLUMN] = np.where(touchable, columns, flock[:, :, CHICKEN_COLUMN])
        flock[:, :, CHICKEN_ROW] = np.where(touchable, rows, flock[:, :, CHICKEN_ROW])
        flock[:, :, CHICKEN_DIRECTION] = np.where(
            touchable, _random_directions(flock.shape[:2]), flock[:, :, CHICKEN_DIRECTION]
        )
        flock[:, :, CHICKEN_MODE] = np.where(
            touchable,
            _random_modes(flock.shape[:2], updater.dive_probability),
            flock[:, :, CHICKEN_MODE],
        )
        particles[chosen, SHIP_WIDTH:] = flock.reshape(len(chosen), -1)

    def _revive(self, particles: np.ndarray) -> None:
        """Undo, in place, whatever made a rebuilt particle terminal.

        The rebuild re-seats the flock but carries the ship-hit flag forward,
        and it leaves a slot the particle believes dead dead -- so a population
        rebuilt because the running episode ruled it out would still read as
        "the ship is already destroyed" and the repair would do nothing. The
        step being taken says otherwise on both counts: the flag is cleared,
        and a particle whose every slot is dead has one brought back, out of
        reach of both sensors so it does not contradict their silence.

        Args:
            particles: ``(N, state_size)`` rebuilt particles, modified in place.
        """
        updater: ChicheckInvadersVectorizedUpdater = self.updater
        particles[:, SHIP_HIT_INDEX] = 0.0

        flock = updater.flock_view(particles)
        empty = np.flatnonzero(~np.any(flock[:, :, CHICKEN_ALIVE] > 0.0, axis=1))
        if not empty.size:
            return
        ships = np.rint(particles[:, SHIP_COLUMN_INDEX])
        hidden_columns, hidden_rows = self._hidden_cells(ships, flock.shape[:2])
        slot = np.random.randint(0, updater.num_chickens, size=empty.size)
        flock[empty, slot, CHICKEN_ALIVE] = 1.0
        flock[empty, slot, CHICKEN_COLUMN] = hidden_columns[empty, slot]
        flock[empty, slot, CHICKEN_ROW] = hidden_rows[empty, slot]
        particles[:, SHIP_WIDTH:] = flock.reshape(len(particles), -1)

    def _rebuild_from_reading(self, particles: np.ndarray, reading: np.ndarray) -> np.ndarray:
        """Re-seat every particle's flock so it could have produced the reading.

        A reported chicken is placed where the reading puts it: the camera gives
        its column offset from the ship, the radar its row and whether it is
        dropping. Half a report places the half it gives and re-draws the other.
        A chicken nothing reported is placed *outside* both sensors, since
        silence is exactly the evidence that it is out of reach.
        """
        updater: ChicheckInvadersVectorizedUpdater = self.updater
        if reading.size != updater.observation_size:
            return particles

        rebuilt = np.array(particles, dtype=np.float64, copy=True)
        flock = updater.flock_view(rebuilt)
        ships = np.rint(rebuilt[:, SHIP_COLUMN_INDEX])
        slots = reading[OBSERVATION_SHIP_WIDTH:].reshape(
            updater.num_chickens, OBSERVATION_CHICKEN_WIDTH
        )
        saw_camera = (slots[:, OBSERVED_CAMERA_REPORTED] > 0.0)[None, :]
        saw_radar = (slots[:, OBSERVED_RADAR_REPORTED] > 0.0)[None, :]
        silent = ~(saw_camera | saw_radar)

        columns = np.where(
            saw_camera,
            np.clip(
                ships[:, None] + slots[:, OBSERVED_CAMERA_OFFSET][None, :],
                0.0,
                float(updater.num_columns - 1),
            ),
            np.random.randint(0, updater.num_columns, size=flock.shape[:2]).astype(np.float64),
        )
        rows = np.where(
            saw_radar,
            np.clip(slots[:, OBSERVED_RADAR_ROWS][None, :], 1.0, float(updater.num_rows - 1)),
            np.random.randint(1, updater.num_rows, size=flock.shape[:2]).astype(np.float64),
        )
        modes = np.where(
            saw_radar,
            np.where(slots[:, OBSERVED_RADAR_DROP][None, :] < 0.0, MODE_DIVE, MODE_PATROL),
            _random_modes(flock.shape[:2], updater.dive_probability),
        )

        hidden_columns, hidden_rows = self._hidden_cells(ships, flock.shape[:2])
        # A dead chicken is silent too, and nothing in the reading separates the
        # two, so a slot the particle already believes dead is left dead.
        alive = flock[:, :, CHICKEN_ALIVE] > 0.0
        reseat_silent = silent & alive

        flock[:, :, CHICKEN_COLUMN] = np.where(
            reseat_silent,
            hidden_columns,
            np.where(silent, flock[:, :, CHICKEN_COLUMN], columns),
        )
        flock[:, :, CHICKEN_ROW] = np.where(
            reseat_silent, hidden_rows, np.where(silent, flock[:, :, CHICKEN_ROW], rows)
        )
        flock[:, :, CHICKEN_MODE] = np.where(
            silent,
            np.where(
                reseat_silent,
                _random_modes(flock.shape[:2], updater.dive_probability),
                flock[:, :, CHICKEN_MODE],
            ),
            modes,
        )
        flock[:, :, CHICKEN_DIRECTION] = np.where(
            silent & ~alive,
            flock[:, :, CHICKEN_DIRECTION],
            _random_directions(flock.shape[:2]),
        )
        flock[:, :, CHICKEN_ALIVE] = np.where(silent, flock[:, :, CHICKEN_ALIVE], 1.0)

        rebuilt[:, SHIP_WIDTH:] = flock.reshape(len(rebuilt), -1)
        return rebuilt

    def _hidden_cells(self, ships: np.ndarray, shape: tuple) -> tuple:
        """Draw a cell neither sensor covers, for every particle and slot.

        When the sensors between them cover the whole grid there is no such
        cell, and the chicken is placed anywhere: silence then says nothing a
        position could satisfy, and pretending otherwise would be inventing
        evidence.
        """
        updater: ChicheckInvadersVectorizedUpdater = self.updater
        grid_columns, grid_rows = np.meshgrid(
            np.arange(updater.num_columns, dtype=np.float64),
            np.arange(1, updater.num_rows, dtype=np.float64),
            indexing="ij",
        )
        cell_columns = grid_columns.ravel()
        cell_rows = grid_rows.ravel()

        offsets = cell_columns[None, :] - ships[:, None]
        rows = np.broadcast_to(cell_rows[None, :], offsets.shape)
        hidden = ~camera_sees(offsets, rows, updater.camera_slope) & ~radar_sees(
            offsets, rows, updater.radar_radius
        )
        allowed = np.where(np.any(hidden, axis=1, keepdims=True), hidden, True)

        cumulative = np.cumsum(allowed, axis=1)
        totals = cumulative[:, -1]
        picks = (np.random.random(shape) * totals[:, None]).astype(np.int64)
        # ``searchsorted`` per particle would need a loop; one comparison against
        # the cumulative counts gives the same index for the whole batch.
        index = np.argmax(cumulative[:, None, :] > picks[:, :, None], axis=2)
        return cell_columns[index], cell_rows[index]


def _log(value: Any) -> np.ndarray:
    """Natural log, with zero mapped to the impossible floor."""
    values = np.asarray(value, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(values)
    return np.where(values > 0.0, logs, IMPOSSIBLE_LOG_PROBABILITY)


def _uniform_log_weights(count: int) -> np.ndarray:
    """Equal log-weights for ``count`` particles."""
    return np.full(int(count), -float(np.log(int(count))), dtype=np.float64)


def _random_directions(shape: tuple) -> np.ndarray:
    """A fresh left/right patrol direction per slot."""
    return np.where(np.random.random(shape) < 0.5, -1.0, 1.0)


def _random_modes(shape: tuple, dive_probability: float) -> np.ndarray:
    """A fresh patrol/dive mode per slot, drawn from the environment's dive rate."""
    return np.where(np.random.random(shape) < dive_probability, MODE_DIVE, MODE_PATROL)


def _reported_slots(reading: np.ndarray, updater: ChicheckInvadersVectorizedUpdater) -> np.ndarray:
    """Which chicken slots this step's reading located, and must be left alone."""
    if reading.size != updater.observation_size:
        return np.zeros(updater.num_chickens, dtype=bool)
    slots = reading[OBSERVATION_SHIP_WIDTH:].reshape(
        updater.num_chickens, OBSERVATION_CHICKEN_WIDTH
    )
    return (slots[:, OBSERVED_CAMERA_REPORTED] > 0.0) | (slots[:, OBSERVED_RADAR_REPORTED] > 0.0)


def create_chicheck_invaders_vectorized_belief(
    env: "ChicheckInvadersPOMDP",
    belief_type: Optional[BeliefType] = None,
    n_particles: int = 200,
    reinvigoration_fraction: float = 0.1,
    **kwargs: Any,
) -> "Belief":
    """Build the initial vectorized belief for a Chicheck Invaders environment.

    Args:
        env: The environment to draw particles from.
        belief_type: ``PARTICLE`` falls back to the scalar filter; anything else
            builds this one. Defaults to ``None``, which builds this one.
        n_particles: Number of particles. Defaults to 200.
        reinvigoration_fraction: Fraction of particles refreshed each step.
            Defaults to 0.1.
        **kwargs: Reserved for future use.

    Returns:
        A uniformly weighted belief over ``n_particles`` draws from the
        environment's initial state distribution.

    Raises:
        ValueError: If *belief_type* is one this environment does not support.
    """
    del kwargs
    if belief_type == BeliefType.PARTICLE:
        # Imported here rather than at module scope: the scalar belief module
        # and this one are both pulled in by the package's ``__init__``, and a
        # top-level import between them is a cycle.
        # pylint: disable-next=import-outside-toplevel
        from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_belief import (
            create_chicheck_invaders_belief,
        )

        return create_chicheck_invaders_belief(
            env, n_particles=n_particles, reinvigoration_fraction=reinvigoration_fraction
        )
    if belief_type not in (None, BeliefType.VECTORIZED_PARTICLE):
        raise ValueError(f"ChicheckInvadersPOMDP does not support belief type {belief_type!r}")

    particles = np.stack(env.initial_state_dist().sample(n_samples=int(n_particles)))
    return ChicheckInvadersVectorizedBelief(
        particles=particles,
        log_weights=_uniform_log_weights(int(n_particles)),
        updater=ChicheckInvadersVectorizedUpdater.from_environment(env),
        reinvigoration_fraction=reinvigoration_fraction,
    )


__all__ = [
    "ChicheckInvadersVectorizedBelief",
    "ChicheckInvadersVectorizedUpdater",
    "create_chicheck_invaders_vectorized_belief",
]
