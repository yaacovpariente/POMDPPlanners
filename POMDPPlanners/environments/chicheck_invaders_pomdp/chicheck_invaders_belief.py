# SPDX-License-Identifier: MIT

"""A weighted particle belief that can recover after the flock surprises it.

The ordinary weighted particle filter already does the interesting half of the
job here, and it does it through the likelihood rather than through any special
code: a chicken that a particle places well inside the camera cone, on a step
where the camera reported nothing, costs that particle a factor of
``1 - p_cam``. Silence is evidence, so particles that put chickens where they
would have been seen die off on their own.

What a weight-only filter cannot do is *invent* a hypothesis it never held. Two
ways that bites in this world:

* the flock's opening placement is drawn from a large set of cells, and a few
  hundred particles cover only a slice of it;
* a chicken outside both sensors for several steps drifts, and the particles
  that happened to guess its direction wrong never come back, because nothing
  in the transition moves a particle from one patrol phase to another.

:class:`ChicheckInvadersBelief` answers both with the same move: after the usual
reweight and resample, a fraction of the particles have the *unobserved*
chickens re-drawn -- a fresh direction, a fresh mode, and a one-cell jitter of
the position -- while every chicken the sensors just reported is left exactly as
the weights found it. Perturbing a chicken the camera just located would throw
away the only hard information the step produced.

Classes:
    ChicheckInvadersBelief: Weighted particle belief with flock reinvigoration.

Functions:
    create_chicheck_invaders_belief: Build the initial belief for an environment.
"""

from typing import Any, List, Optional

import numpy as np

from POMDPPlanners.core.belief.particle_beliefs import (
    WeightedParticleBelief,
    WeightedParticleBeliefReinvigoration,
)
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_pomdp import (
    IMPOSSIBLE_LOG_PROBABILITY,
    ChicheckInvadersPOMDP,
    ObservationMode,
)
from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_schema import (
    CHICKEN_ALIVE,
    CHICKEN_COLUMN,
    CHICKEN_DIRECTION,
    CHICKEN_MODE,
    CHICKEN_ROW,
    MODE_DIVE,
    MODE_PATROL,
    OBSERVATION_CHICKEN_WIDTH,
    OBSERVATION_SHIP_WIDTH,
    OBSERVED_CAMERA_OFFSET,
    OBSERVED_CAMERA_REPORTED,
    OBSERVED_RADAR_DROP,
    OBSERVED_RADAR_REPORTED,
    OBSERVED_RADAR_ROWS,
    SHIP_COLUMN_INDEX,
    chicken_slots,
)
from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_sensors import (
    camera_sees,
    radar_sees,
)
from POMDPPlanners.utils.config_to_id import config_to_id


class ChicheckInvadersBelief(WeightedParticleBeliefReinvigoration):
    """Weighted particle belief that re-draws the chickens it cannot see.

    Attributes:
        num_chickens: Number of chicken slots each particle carries.
        num_columns: Grid width, used to keep a jittered column on the grid.
        num_rows: Grid height, used to keep a jittered row above the ship.
        dive_probability: The environment's dive rate, which is also the prior a
            re-drawn chicken's mode is sampled from.
        reinvigoration_fraction: Fraction of particles refreshed each step.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        particles: List[Any],
        log_weights: np.ndarray,
        num_chickens: int,
        num_columns: int,
        num_rows: int,
        dive_probability: float,
        resampling: bool = True,
        ess_factor: float = 0.5,
        reinvigoration_fraction: float = 0.1,
    ):
        """Initialize the belief.

        Args:
            particles: State particles.
            log_weights: Log-weights for the particles.
            num_chickens: Number of chicken slots per particle.
            num_columns: Grid width.
            num_rows: Grid height.
            dive_probability: Prior a re-drawn chicken is diving.
            resampling: Enable automatic resampling when ESS drops. Defaults to
                ``True``.
            ess_factor: Effective-sample-size threshold factor. Defaults to 0.5.
            reinvigoration_fraction: Fraction of particles whose unobserved
                chickens are re-drawn each step. Defaults to 0.1 -- small on
                purpose, because a refreshed particle carries no evidence from
                the steps before it, so refreshing too many throws away the
                filter's memory to buy diversity it may not need.
        """
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            resampling=resampling,
            ess_factor=ess_factor,
            reinvigoration_fraction=reinvigoration_fraction,
        )
        self.num_chickens = int(num_chickens)
        self.num_columns = int(num_columns)
        self.num_rows = int(num_rows)
        self.dive_probability = float(dive_probability)

    @property
    def config_id(self) -> str:
        """Identifier covering the particles and the refresh configuration.

        Two beliefs holding identical particles but refreshing different
        fractions of them behave differently on the next step, so the fraction
        belongs in the identity. The observation does not: it is one step's
        reading rather than configuration, and folding it in would give an
        identity that changed every step and defeated the caching it exists for.
        """
        return config_to_id(
            {
                "particles": super().config_id,
                "num_chickens": self.num_chickens,
                "num_columns": self.num_columns,
                "num_rows": self.num_rows,
                "dive_probability": self.dive_probability,
                "reinvigoration_fraction": self.reinvigoration_fraction,
            }
        )

    def reinvigorate(  # type: ignore[override]
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        belief: "WeightedParticleBelief",
    ) -> "ChicheckInvadersBelief":
        """Collapse onto the truth, rebuild after a wipe-out, or re-draw a few.

        Three cases, in order of how much of the belief they replace.

        **Fully observable.** The observation *is* the state, and the likelihood
        is one only on an exact array match, so every particle drawn from the
        flock prior floors. ``WeightedParticleBelief`` then normalises an
        all-floor vector to a uniform one, which makes the belief a uniform
        distribution over particles that are all known to be wrong -- and the
        wrongness is invisible, because uniform weights are what a healthy prior
        looks like too. Since ``ChicheckInvadersPOMDP[fully_observable]`` is a
        registered environment, that would leave the advertised baseline running
        on a permanently meaningless belief. The right belief for a fully
        observable world is a point mass on what was observed, so that is what
        this installs.

        **Every weight floored in the partial mode.** The same
        normalise-to-uniform path is reachable whenever no particle can explain
        the reading -- easiest to hit under the noiseless preset, where a single
        contradicted slot is enough. Resampling cannot help, because it draws
        from the particles that are already wrong. The particles are therefore
        rebuilt from the reading itself, which is the only information left.

        **Otherwise.** The ordinary refresh: a fraction of the particles have
        their *unreported* chickens re-drawn, and chickens the sensors just
        located are left exactly as the weights found them.

        Args:
            action: Ignored; the refresh is about the flock, not the ship.
            observation: This step's reading.
            pomdp: The environment, read for its observation mode and its
                likelihood.
            belief: The belief after the ordinary reweight and resample.

        Returns:
            A belief of this class, so the refresh repeats next step.
        """
        del action
        if self._is_fully_observable(pomdp):
            return self._successor(
                [
                    np.array(observation, dtype=np.float64, copy=True)
                    for _ in range(len(belief.particles))
                ],
                self._uniform_log_weights(len(belief.particles)),
                belief,
            )

        particles = [np.array(p, dtype=np.float64, copy=True) for p in belief.particles]
        untouched = self._reported_slots(observation, pomdp)

        if self._every_particle_contradicts(particles, observation, pomdp):
            for particle in particles:
                self._rebuild_from_reading(particle, observation, pomdp)
            return self._successor(particles, self._uniform_log_weights(len(particles)), belief)

        count = int(round(self.reinvigoration_fraction * len(particles)))
        if count > 0 and not np.all(untouched):
            chosen = np.random.choice(
                len(particles), size=min(count, len(particles)), replace=False
            )
            for index in chosen:
                self._refresh(particles[int(index)], untouched)
        return self._successor(
            particles, np.array(belief.log_weights, dtype=np.float64, copy=True), belief
        )

    @staticmethod
    def _uniform_log_weights(count: int) -> np.ndarray:
        """Equal log-weights for ``count`` particles.

        Any constant is uniform, since log-weights are unnormalised. The
        constant is 1.0 rather than 0.0 because ``WeightedParticleBelief``
        rejects an all-zero vector outright -- it cannot tell "uniform" from
        "nothing was ever weighted".
        """
        return np.full(int(count), 1.0, dtype=np.float64)

    def _successor(
        self,
        particles: List[Any],
        log_weights: np.ndarray,
        belief: "WeightedParticleBelief",
    ) -> "ChicheckInvadersBelief":
        """Wrap ``particles`` back up as a belief of this class."""
        return ChicheckInvadersBelief(
            particles=particles,
            log_weights=log_weights,
            num_chickens=self.num_chickens,
            num_columns=self.num_columns,
            num_rows=self.num_rows,
            dive_probability=self.dive_probability,
            resampling=belief.resampling,
            ess_factor=belief.ess_factor,
            reinvigoration_fraction=self.reinvigoration_fraction,
        )

    @staticmethod
    def _is_fully_observable(pomdp: Environment) -> bool:
        """Whether ``pomdp`` hands out the state as its observation."""
        return (
            isinstance(pomdp, ChicheckInvadersPOMDP)
            and pomdp.observation_mode is ObservationMode.FULL
        )

    def _every_particle_contradicts(
        self, particles: List[Any], observation: Any, pomdp: Environment
    ) -> bool:
        """Whether this step's reading is impossible under every particle.

        Costs one extra likelihood pass over the particles per step. That is
        real, and it buys the one thing the filter cannot otherwise see: a
        normalised weight vector looks identical whether every particle was
        plausible or every particle was impossible.
        """
        if not isinstance(pomdp, ChicheckInvadersPOMDP) or not particles:
            return False
        scores = pomdp.observation_log_probability_per_state(particles, None, observation)
        return bool(np.all(np.asarray(scores) <= IMPOSSIBLE_LOG_PROBABILITY))

    def _rebuild_from_reading(
        self, particle: np.ndarray, observation: Any, pomdp: Environment
    ) -> None:
        """Re-seat one particle's flock so that it could have produced the reading.

        A reported chicken is placed where the reading puts it: the camera gives
        its column offset from the ship, the radar its row and whether it is
        dropping. Half a report places the half it gives and re-draws the other.
        A chicken nothing reported is placed *outside* both sensors, since
        silence is exactly the evidence that it is out of reach.
        """
        reading = np.asarray(observation, dtype=np.float64)
        expected = OBSERVATION_SHIP_WIDTH + OBSERVATION_CHICKEN_WIDTH * self.num_chickens
        if reading.shape != (expected,):
            return
        ship = int(round(float(particle[SHIP_COLUMN_INDEX])))
        flock = chicken_slots(particle, self.num_chickens)
        for index in range(self.num_chickens):
            base = OBSERVATION_SHIP_WIDTH + OBSERVATION_CHICKEN_WIDTH * index
            saw_camera = reading[base + OBSERVED_CAMERA_REPORTED] > 0.0
            saw_radar = reading[base + OBSERVED_RADAR_REPORTED] > 0.0
            if not saw_camera and not saw_radar:
                # A dead chicken is silent too, and nothing in the reading
                # separates the two, so a slot the particle already believes
                # dead is left dead.
                if flock[index, CHICKEN_ALIVE] > 0.0:
                    self._seat_out_of_reach(flock[index], ship, pomdp)
                continue
            flock[index, CHICKEN_ALIVE] = 1.0
            if saw_camera:
                column = ship + float(reading[base + OBSERVED_CAMERA_OFFSET])
                flock[index, CHICKEN_COLUMN] = min(max(column, 0.0), float(self.num_columns - 1))
            else:
                flock[index, CHICKEN_COLUMN] = float(np.random.randint(self.num_columns))
            if saw_radar:
                row = float(reading[base + OBSERVED_RADAR_ROWS])
                flock[index, CHICKEN_ROW] = min(max(row, 1.0), float(self.num_rows - 1))
                flock[index, CHICKEN_MODE] = (
                    MODE_DIVE if reading[base + OBSERVED_RADAR_DROP] < 0.0 else MODE_PATROL
                )
            else:
                flock[index, CHICKEN_ROW] = float(np.random.randint(1, self.num_rows))
                flock[index, CHICKEN_MODE] = (
                    MODE_DIVE if np.random.random() < self.dive_probability else MODE_PATROL
                )
            flock[index, CHICKEN_DIRECTION] = -1.0 if np.random.random() < 0.5 else 1.0

    def _seat_out_of_reach(self, slot: np.ndarray, ship: int, pomdp: Environment) -> None:
        """Place one silent chicken on a cell neither sensor covers, if one exists.

        When the sensors between them cover the whole grid there is no such
        cell, and the chicken is placed anywhere: silence then says nothing a
        position could satisfy, and pretending otherwise would be inventing
        evidence.
        """
        cells = [
            (column, row) for row in range(1, self.num_rows) for column in range(self.num_columns)
        ]
        if isinstance(pomdp, ChicheckInvadersPOMDP):
            offsets = np.array([column - ship for column, _ in cells], dtype=np.float64)
            rows = np.array([row for _, row in cells], dtype=np.float64)
            hidden = ~camera_sees(offsets, rows, pomdp.camera_slope) & ~radar_sees(
                offsets, rows, pomdp.radar_radius
            )
            if np.any(hidden):
                cells = [cell for cell, is_hidden in zip(cells, hidden) if is_hidden]
        column, row = cells[int(np.random.randint(len(cells)))]
        slot[CHICKEN_COLUMN] = float(column)
        slot[CHICKEN_ROW] = float(row)
        slot[CHICKEN_DIRECTION] = -1.0 if np.random.random() < 0.5 else 1.0
        slot[CHICKEN_MODE] = (
            MODE_DIVE if np.random.random() < self.dive_probability else MODE_PATROL
        )

    def _reported_slots(self, observation: Any, pomdp: Environment) -> np.ndarray:
        """Which chicken slots this step's reading located, and must be left alone."""
        if self._is_fully_observable(pomdp):
            # Nothing is hidden, so nothing may be perturbed. The caller never
            # reaches this in that mode -- it installs a point mass instead --
            # but the answer is still the honest one.
            return np.ones(self.num_chickens, dtype=bool)
        reading = np.asarray(observation, dtype=np.float64)
        reported = np.zeros(self.num_chickens, dtype=bool)
        expected = OBSERVATION_SHIP_WIDTH + OBSERVATION_CHICKEN_WIDTH * self.num_chickens
        if reading.shape != (expected,):
            return reported
        for index in range(self.num_chickens):
            base = OBSERVATION_SHIP_WIDTH + OBSERVATION_CHICKEN_WIDTH * index
            reported[index] = (
                reading[base + OBSERVED_CAMERA_REPORTED] > 0.0
                or reading[base + OBSERVED_RADAR_REPORTED] > 0.0
            )
        return reported

    def _refresh(self, particle: np.ndarray, untouched: np.ndarray) -> None:
        """Jitter one particle's unreported live chickens, in place."""
        flock = chicken_slots(particle, self.num_chickens)
        for index in range(self.num_chickens):
            if untouched[index] or flock[index, CHICKEN_ALIVE] <= 0.0:
                continue
            column = flock[index, CHICKEN_COLUMN] + float(np.random.randint(-1, 2))
            row = flock[index, CHICKEN_ROW] + float(np.random.randint(-1, 2))
            # Clamped rather than wrapped, and never onto row 0: a particle
            # placed on the ship's own row asserts the episode has already been
            # lost, which no jitter should be able to claim.
            flock[index, CHICKEN_COLUMN] = min(max(column, 0.0), float(self.num_columns - 1))
            flock[index, CHICKEN_ROW] = min(max(row, 1.0), float(self.num_rows - 1))
            flock[index, CHICKEN_DIRECTION] = -1.0 if np.random.random() < 0.5 else 1.0
            flock[index, CHICKEN_MODE] = (
                MODE_DIVE if np.random.random() < self.dive_probability else MODE_PATROL
            )


def create_chicheck_invaders_belief(
    env: ChicheckInvadersPOMDP,
    belief_type: Optional[Any] = None,
    n_particles: int = 200,
    reinvigoration_fraction: float = 0.1,
) -> ChicheckInvadersBelief:
    """Build the initial belief for a Chicheck Invaders environment.

    Args:
        env: The environment to draw particles from.
        belief_type: Accepted for the belief-factory signature and ignored;
            this environment has one filter.
        n_particles: Number of particles. Defaults to 200.
        reinvigoration_fraction: Fraction of particles refreshed each step.
            Defaults to 0.1.

    Returns:
        A uniformly weighted belief over ``n_particles`` draws from the
        environment's initial state distribution.
    """
    del belief_type
    particles = env.initial_state_dist().sample(n_samples=int(n_particles))
    return ChicheckInvadersBelief(
        particles=particles,
        log_weights=np.log(np.full(int(n_particles), 1.0 / int(n_particles))),
        num_chickens=env.num_chickens,
        num_columns=env.num_columns,
        num_rows=env.num_rows,
        dive_probability=env.dive_probability,
        reinvigoration_fraction=reinvigoration_fraction,
    )
