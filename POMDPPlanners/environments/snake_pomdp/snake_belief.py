# SPDX-License-Identifier: MIT

"""Exact belief over the Snake food cell.

Everything in a Snake state except the food is a deterministic function of the
agent's own action history, and the observation reports it exactly. What is left
is one hidden cell, so the belief is a categorical distribution over the grid --
small enough to carry exactly rather than sample.

Carrying it exactly is not only cheaper than a particle filter here, it is more
robust. The vision window has no false positives, so a sighting rules out every
cell but one; a generic filter that happened to hold no particle on that cell
would have every weight floored to ``eps``, resample impossible cells back over
the whole set, and go on planning against food positions the sensor has already
excluded. Nothing raises. The belief just stops meaning anything.

The update follows the environment's own model:

* when the length grew, the agent has seen the snake eat, so the food it was
  tracking is gone and a fresh one was drawn uniformly over the cells the new
  body leaves free -- the prior restarts from that uniform;
* otherwise the previous distribution carries over, with the cells the new body
  now occupies set to zero (only the new head cell can be one of them, and it is
  ruled out precisely because the snake did *not* eat);

and either prior is then multiplied by the likelihood of the sighting and the
scent and normalised.

Particles are redrawn from the posterior on every update, so they are i.i.d.
draws from it and every one of them is a legal state.

Classes:
    SnakeBelief: The exact belief.
"""

from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.environments.snake_pomdp.snake_pomdp import (
    OBSERVATION_LIVE,
    SnakeTermination,
    create_snake_state,
)
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.environments.snake_pomdp.snake_pomdp import SnakePOMDP


def _as_snake(pomdp: Environment) -> "SnakePOMDP":
    """Narrow the base ``Environment`` the Belief API hands us to a Snake one.

    The ``Belief`` interface types this argument as ``Environment`` and every
    caller passes whatever environment the belief is held over, so narrowing the
    parameter itself would be a Liskov violation. The cast is checked once, at
    construction, by :meth:`SnakeBelief.from_environment`.

    Args:
        pomdp: The environment supplied by the caller.

    Returns:
        The same object, typed as a Snake environment.
    """
    return cast("SnakePOMDP", pomdp)


def _uniform_log_weights(n_particles: int) -> np.ndarray:
    """Return finite, equal log-weights for ``n_particles`` particles.

    ``log(1/n)`` is the honest value, but for ``n == 1`` it is exactly zero and
    :class:`WeightedParticleBelief` rejects an all-zero weight vector. Weights
    are normalised by subtracting their maximum, so any constant works.

    Args:
        n_particles: Number of particles.

    Returns:
        ``(n_particles,)`` ``float64`` array of equal, non-zero log-weights.
    """
    if n_particles == 1:
        return np.array([-1.0], dtype=np.float64)
    return np.full(n_particles, -float(np.log(n_particles)), dtype=np.float64)


class SnakeBelief(WeightedParticleBelief):
    """Particle belief backed by an exact categorical over the food cell.

    Subclasses :class:`WeightedParticleBelief` rather than :class:`Belief`
    directly so that everything downstream that dispatches on belief type -- the
    expected-reward helper, the terminal-belief check, the arena tree -- keeps
    working unchanged. The particles and their uniform weights are a genuine
    weighted particle belief; only :meth:`update` is replaced, because the
    generic weight-and-resample update is the part that is lossy here.

    Attributes:
        particles: ``(n_particles, state_size)`` ``float64`` array of states,
            drawn from the exact posterior and sharing one body and counter.
        log_weights: Uniform log-weights. The redraw already applies the
            posterior, so re-weighting on top of it would apply the same
            evidence twice.
        food_probabilities: ``(grid_size ** 2,)`` posterior over the food cell,
            in row-major order. This is the belief; the particles are draws
            from it.
    """

    def __init__(
        self,
        particles: Any,
        log_weights: np.ndarray,
        resampling: bool = False,
        ess_factor: float = 0.5,
        food_probabilities: Optional[np.ndarray] = None,
    ):
        """Initialize the belief.

        Args:
            particles: State particles, all sharing one body and counter.
            log_weights: Log-weights, one per particle.
            resampling: Kept for interface compatibility; this belief redraws
                from the exact posterior every update, so weight-based
                resampling never applies. Defaults to ``False``.
            ess_factor: Kept for interface compatibility.
            food_probabilities: The exact posterior over the food cell. Defaults
                to ``None``, which falls back to the particle histogram. The
                fallback is a Monte Carlo summary rather than the exact
                distribution, and exists only so a belief rebuilt from its
                particles alone (an unpickled one, say) still updates rather
                than failing.
        """
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            resampling=resampling,
            ess_factor=ess_factor,
        )
        self.food_probabilities: Optional[np.ndarray] = (
            None if food_probabilities is None else np.asarray(food_probabilities, dtype=np.float64)
        )

    def to_dict(self) -> dict:
        """Preserve the exact posterior across a serialization round trip.

        The inherited ``to_dict`` carries the particles and their weights, which
        is everything a generic particle belief is. It is not everything this
        one is: the exact distribution the particles were drawn from cannot be
        recovered from a finite sample of it, so a serialized belief would come
        back as a Monte Carlo summary of itself and the visualization's belief
        layer would lose its resolution.

        This follows the convention ``OccupancyGridMappingBelief`` sets. It is
        only half a fix: ``History.from_dict`` reconstructs the literal
        ``WeightedParticleBelief`` and no subclass, so a belief read back out of
        a serialized history is still a plain dict. Keeping the field here is
        what makes that a framework gap rather than lost data.

        Returns:
            The inherited fields plus the exact posterior over the food cell.
        """
        result = super().to_dict()
        result["food_probabilities"] = (
            None if self.food_probabilities is None else self.food_probabilities.tolist()
        )
        return result

    @classmethod
    def from_environment(cls, pomdp: Environment, n_particles: int = 100) -> "SnakeBelief":
        """Build the prior belief for ``pomdp``.

        Args:
            pomdp: A :class:`SnakePOMDP`.
            n_particles: How many particles to carry. Defaults to 100.

        Returns:
            The uniform-over-free-cells prior with the starting body.

        Raises:
            TypeError: If ``pomdp`` is not a Snake environment.
            ValueError: If ``n_particles`` is not positive.
        """
        if not hasattr(pomdp, "scent_probabilities"):
            raise TypeError(f"SnakeBelief needs a SnakePOMDP, got {type(pomdp).__name__}")
        if n_particles <= 0:
            raise ValueError(f"n_particles must be positive, got {n_particles}")
        env = _as_snake(pomdp)
        distribution = env.initial_state_dist()
        body = distribution.body  # type: ignore[attr-defined]
        probabilities = np.zeros(env.num_cells, dtype=np.float64)
        free = env.free_cells(body)
        probabilities[free] = 1.0 / free.size
        return cls(
            particles=_draw_particles(env, probabilities, body, 0, n_particles),
            log_weights=_uniform_log_weights(n_particles),
            food_probabilities=probabilities,
        )

    @property
    def config_id(self) -> str:
        """Identity of this belief, distinct from a generic particle belief's.

        The inherited identity is built from the particles and their weights
        alone, with no class in it. That is enough to tell two particle beliefs
        apart, but not enough to tell *this* belief from a
        :class:`WeightedParticleBelief` holding the same particles -- and the
        episode result cache keys on the initial belief's ``config_id``, so the
        two would share cached episodes. Comparing the exact belief against a
        generic filter is the first thing anyone will want to do on this
        environment, and it is exactly the comparison that collision would
        silently answer from the wrong run.

        Returns:
            The inherited identity, qualified by this class and by the exact
            distribution the particles were drawn from.
        """
        return config_to_id(
            {
                "class": f"{type(self).__module__}.{type(self).__qualname__}",
                "particles": super().config_id,
                "food_probabilities": (
                    [] if self.food_probabilities is None else self.food_probabilities.tolist()
                ),
            }
        )

    def marginal(self, pomdp: Environment) -> np.ndarray:
        """Per-cell posterior probability that the food is in that cell.

        Args:
            pomdp: The Snake environment the belief is held over.

        Returns:
            ``(grid_size, grid_size)`` ``float64`` array summing to one, or to
            zero for a belief whose episode has ended and whose particles carry
            no food.
        """
        env = _as_snake(pomdp)
        return self._flat_marginal(env).reshape(env.grid_size, env.grid_size)

    def _flat_marginal(self, env: "SnakePOMDP") -> np.ndarray:
        """Row-major posterior over the food cell, exact when it is carried."""
        if self.food_probabilities is not None:
            return self.food_probabilities
        histogram = np.zeros(env.num_cells, dtype=np.float64)
        weights = np.asarray(self.normalized_weights, dtype=np.float64)
        for particle, weight in zip(self.particles, weights):
            food = env.food(particle)
            if food is not None:
                histogram[food[0] * env.grid_size + food[1]] += float(weight)
        total = float(histogram.sum())
        return histogram / total if total > 0.0 else histogram

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        state: Optional[Any] = None,
    ) -> "WeightedParticleBelief":
        """Condition on one reading and redraw the particles.

        Args:
            action: The action just taken.
            observation: The reading it produced.
            pomdp: The Snake environment.
            state: Unused; the true state must not leak into the belief.

        Returns:
            The exact posterior belief. The terminal reading names no body,
            sighting or scent, but it is still evidence -- it rules out every
            food cell whose step would have left the episode running -- so it
            is conditioned on rather than propagated blindly.

        Raises:
            ValueError: If no food cell is consistent with the reading. That
                cannot happen while the belief is conditioned on readings this
                environment produced, so it means the belief and the world have
                come apart. It is raised rather than worked around because a
                belief supported on nothing is exactly the failure this class
                exists to prevent.
        """
        del state
        env = _as_snake(pomdp)
        flat = tuple(int(value) for value in observation)
        reference = np.asarray(self.particles[0], dtype=np.float64)
        if not flat or flat[0] != OBSERVATION_LIVE:
            if env.is_terminal(reference):
                # Already absorbing: every particle maps to itself, which is
                # exactly what the generic update does.
                return super().update(
                    action=action, observation=observation, pomdp=pomdp, state=None
                )
            return self._terminal_update(int(action), env, reference)

        new_body, seen, scent = env.decode_observation(flat)
        grew = len(new_body) > env.snake_length(reference)
        counter = 0 if grew else env.steps_since_food(reference) + 1

        free = env.free_cells(new_body)
        prior = np.zeros(env.num_cells, dtype=np.float64)
        if grew:
            # The tracked food was eaten and a fresh one was drawn uniformly
            # over the cells the new body leaves free.
            prior[free] = 1.0 / free.size
        else:
            prior[free] = self._flat_marginal(env)[free]

        posterior = prior * _likelihood(env, new_body[0], free, seen, scent, env.num_cells)
        total = float(posterior.sum())
        if total <= 0.0:
            raise ValueError(
                "no food cell is consistent with the Snake reading "
                f"(seen={seen}, scent={scent}); the belief and the world disagree"
            )
        posterior /= total

        n_particles = len(self.particles)
        return SnakeBelief(
            particles=_draw_particles(env, posterior, new_body, counter, n_particles),
            log_weights=_uniform_log_weights(n_particles),
            food_probabilities=posterior,
        )

    def _terminal_update(
        self, action: int, env: "SnakePOMDP", reference: np.ndarray
    ) -> "SnakeBelief":
        """Condition a live belief on the reading that says the episode ended.

        Deferring this to the generic update is wrong, and quietly so. That
        update propagates every particle and weights it by the observation
        likelihood, but a particle whose successor is still running scores
        ``-inf`` against the terminal reading and, with resampling off, is kept
        at floor weight rather than dropped. The belief then holds a majority of
        states the reading has ruled out and does not read as terminal, so a
        planner goes on expanding a branch whose episode is over.

        It only bites on a win. Walls, self-collisions and starvation do not
        depend on where the food is, so every particle ends together and the
        generic update happens to be right; winning requires eating, so only
        there do the particles disagree.

        The reading is evidence like any other: it keeps exactly the food cells
        whose step would have ended the episode. Each survivor has one
        deterministic successor -- a terminal step never respawns food -- so the
        posterior transfers to the successors unchanged.

        Args:
            action: The action just taken.
            env: The Snake environment.
            reference: Any live particle; all of them share body and counter.

        Returns:
            A belief over terminal states only.

        Raises:
            ValueError: If no food cell would have ended the episode.
        """
        body = env.body(reference)
        counter = env.steps_since_food(reference)
        prior = self._flat_marginal(env)

        successors = []
        masses = []
        for index in np.flatnonzero(prior > 0.0):
            candidate = create_snake_state(
                body=body,
                food=(int(index) // env.grid_size, int(index) % env.grid_size),
                steps_since_food=counter,
                target_length=env.target_length,
                status=int(SnakeTermination.RUNNING),
            )
            if env.transition_outcome(candidate, action)[3] is SnakeTermination.RUNNING:
                continue
            successors.append(env.sample_next_state(state=candidate, action=action))
            masses.append(float(prior[index]))

        total = float(sum(masses))
        if total <= 0.0:
            raise ValueError(
                "the reading says the episode ended, but no food cell the belief "
                "holds would have ended it; the belief and the world disagree"
            )

        n_particles = len(self.particles)
        drawn = np.random.choice(len(successors), size=n_particles, p=np.asarray(masses) / total)
        return SnakeBelief(
            particles=np.asarray([successors[int(i)] for i in drawn], dtype=np.float64),
            log_weights=_uniform_log_weights(n_particles),
        )


def _likelihood(
    env: "SnakePOMDP",
    head: tuple,
    free: np.ndarray,
    seen: Optional[tuple],
    scent: int,
    num_cells: int,
) -> np.ndarray:
    """Likelihood of one reading's sighting and scent, per candidate food cell.

    Args:
        env: The Snake environment.
        head: The head cell of the observed body.
        free: Flat indices of the cells the food could be in.
        seen: The sighted cell, or ``None``.
        scent: The reported quadrant.
        num_cells: Number of grid cells.

    Returns:
        ``(num_cells,)`` ``float64`` array, zero outside ``free``.
    """
    window = set(env.window_cells(head))
    likelihood = np.zeros(num_cells, dtype=np.float64)
    for index in free:
        cell = (int(index) // env.grid_size, int(index) % env.grid_size)
        inside = cell in window
        if seen is None:
            sighting = 1.0 - env.detection_probability if inside else 1.0
        elif inside and cell == seen:
            sighting = env.detection_probability
        else:
            # No false positives, so a sighting is conclusive: any other cell
            # has likelihood zero for this reading.
            continue
        likelihood[index] = sighting * float(env.scent_probabilities(head, cell)[scent])
    return likelihood


def _draw_particles(
    env: "SnakePOMDP",
    probabilities: np.ndarray,
    body: tuple,
    steps_since_food: int,
    n_particles: int,
) -> np.ndarray:
    """Draw ``n_particles`` states from the posterior over the food cell.

    Drawn with replacement: the posterior is a distribution over cells, and
    i.i.d. draws from it are what a particle belief is supposed to be. Sampling
    without replacement would refuse to represent a posterior narrower than the
    particle count -- exactly the case after a sighting, where one cell holds
    all the mass.

    Args:
        env: The Snake environment.
        probabilities: The posterior over the food cell, row-major.
        body: The body every drawn state shares.
        steps_since_food: The counter every drawn state shares.
        n_particles: How many particles to draw.

    Returns:
        ``(n_particles, state_size)`` ``float64`` array.
    """
    drawn = np.random.choice(env.num_cells, size=int(n_particles), p=probabilities)
    return np.asarray(
        [
            create_snake_state(
                body=body,
                food=(int(cell) // env.grid_size, int(cell) % env.grid_size),
                steps_since_food=steps_since_food,
                target_length=env.target_length,
                status=int(SnakeTermination.RUNNING),
            )
            for cell in drawn
        ],
        dtype=np.float64,
    )
