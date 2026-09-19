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

:class:`CrazyChickenBelief` answers both with the same move: after the usual
reweight and resample, a fraction of the particles have the *unobserved*
chickens re-drawn -- a fresh direction, a fresh mode, and a one-cell jitter of
the position -- while every chicken the sensors just reported is left exactly as
the weights found it. Perturbing a chicken the camera just located would throw
away the only hard information the step produced.

Classes:
    CrazyChickenBelief: Weighted particle belief with flock reinvigoration.

Functions:
    create_crazy_chicken_belief: Build the initial belief for an environment.
"""

from typing import Any, List, Optional

import numpy as np

from POMDPPlanners.core.belief.particle_beliefs import (
    WeightedParticleBelief,
    WeightedParticleBeliefReinvigoration,
)
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.environments.crazy_chicken_pomdp.crazy_chicken_pomdp import (
    CrazyChickenPOMDP,
    ObservationMode,
)
from POMDPPlanners.environments.crazy_chicken_pomdp.crazy_chicken_schema import (
    CHICKEN_ALIVE,
    CHICKEN_COLUMN,
    CHICKEN_DIRECTION,
    CHICKEN_MODE,
    CHICKEN_ROW,
    MODE_DIVE,
    MODE_PATROL,
    OBSERVATION_CHICKEN_WIDTH,
    OBSERVATION_SHIP_WIDTH,
    OBSERVED_CAMERA_REPORTED,
    OBSERVED_RADAR_REPORTED,
    chicken_slots,
)
from POMDPPlanners.utils.config_to_id import config_to_id


class CrazyChickenBelief(WeightedParticleBeliefReinvigoration):
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
    ) -> "CrazyChickenBelief":
        """Re-draw the unreported chickens in a fraction of the particles.

        Args:
            action: Ignored; the refresh is about the flock, not the ship.
            observation: This step's reading, which says which chicken slots
                were reported and are therefore left alone.
            pomdp: The environment, read for its observation mode.
            belief: The belief after the ordinary reweight and resample.

        Returns:
            A belief of this class, so the refresh repeats next step.
        """
        del action
        particles = [np.array(p, dtype=np.float64, copy=True) for p in belief.particles]
        untouched = self._reported_slots(observation, pomdp)
        count = int(round(self.reinvigoration_fraction * len(particles)))
        if count > 0 and not np.all(untouched):
            chosen = np.random.choice(
                len(particles), size=min(count, len(particles)), replace=False
            )
            for index in chosen:
                self._refresh(particles[int(index)], untouched)
        return CrazyChickenBelief(
            particles=particles,
            log_weights=np.array(belief.log_weights, dtype=np.float64, copy=True),
            num_chickens=self.num_chickens,
            num_columns=self.num_columns,
            num_rows=self.num_rows,
            dive_probability=self.dive_probability,
            resampling=belief.resampling,
            ess_factor=belief.ess_factor,
            reinvigoration_fraction=self.reinvigoration_fraction,
        )

    def _reported_slots(self, observation: Any, pomdp: Environment) -> np.ndarray:
        """Which chicken slots this step's reading located, and must be left alone."""
        if isinstance(pomdp, CrazyChickenPOMDP) and pomdp.observation_mode is ObservationMode.FULL:
            # Nothing is hidden, so nothing may be perturbed.
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


def create_crazy_chicken_belief(
    env: CrazyChickenPOMDP,
    belief_type: Optional[Any] = None,
    n_particles: int = 200,
    reinvigoration_fraction: float = 0.1,
) -> CrazyChickenBelief:
    """Build the initial belief for a Crazy Chicken environment.

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
    return CrazyChickenBelief(
        particles=particles,
        log_weights=np.log(np.full(int(n_particles), 1.0 / int(n_particles))),
        num_chickens=env.num_chickens,
        num_columns=env.num_columns,
        num_rows=env.num_rows,
        dive_probability=env.dive_probability,
        reinvigoration_fraction=reinvigoration_fraction,
    )
