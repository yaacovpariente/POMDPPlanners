# SPDX-License-Identifier: MIT

"""The exact belief over the Snake food cell.

The belief is the part of this environment that a generic particle filter gets
wrong quietly, so these tests check the posterior itself rather than only that
an update runs: that a sighting collapses it, that a respawn restarts it, and
that it agrees with a brute-force application of Bayes' rule over the 49 cells
of a small grid.
"""

from typing import cast

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.belief.belief_utils import is_terminal_belief
from POMDPPlanners.environments.snake_pomdp.snake_belief import SnakeBelief
from POMDPPlanners.environments.snake_pomdp.snake_pomdp import (
    TERMINAL_OBSERVATION,
    SnakeAction,
    SnakePOMDP,
    SnakeQuadrant,
    SnakeTermination,
    create_snake_state,
)


def build_env(**overrides):
    """Return a small Snake environment, with overrides applied."""
    kwargs = {
        "grid_size": 7,
        "target_length": 6,
        "window_radius": 2,
        "detection_probability": 0.9,
        "scent_accuracy": 0.7,
        "starvation_limit": 8,
        "discount_factor": 0.98,
    }
    kwargs.update(overrides)
    return SnakePOMDP(**kwargs)


def marginal_of(belief, env):
    """Read the exact marginal off a belief the update typed as the base class.

    ``SnakeBelief.update`` is declared to return ``WeightedParticleBelief``,
    because the terminal reading falls back to the generic update. Every call
    here is on a live reading, which returns a ``SnakeBelief``.
    """
    return cast(SnakeBelief, belief).marginal(env)


def belief_over(env, body, probabilities, steps_since_food=0, n_particles=64):
    """Build a belief with a hand-written posterior over the food cell."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    np.random.seed(0)
    cells = np.random.choice(env.num_cells, size=n_particles, p=probabilities)
    particles = np.asarray(
        [
            create_snake_state(
                body=body,
                food=(int(cell) // env.grid_size, int(cell) % env.grid_size),
                steps_since_food=steps_since_food,
                target_length=env.target_length,
            )
            for cell in cells
        ],
        dtype=np.float64,
    )
    return SnakeBelief(
        particles=particles,
        log_weights=np.full(n_particles, -float(np.log(n_particles))),
        food_probabilities=probabilities,
    )


def brute_force_posterior(env, prior, body, seen, scent):
    """Apply Bayes' rule cell by cell, using only the environment's own model.

    Written independently of the belief so the two can disagree: it scores each
    candidate food cell by building the state it implies and asking the
    environment for the reading's likelihood, which is the same question the
    planner's generic filter would ask.
    """
    posterior = np.zeros(env.num_cells, dtype=np.float64)
    reading = env.encode_observation_tuple(body, seen, scent)
    for cell in range(env.num_cells):
        if prior[cell] <= 0.0:
            continue
        food = (cell // env.grid_size, cell % env.grid_size)
        if food in body:
            continue
        state = create_snake_state(body=body, food=food, target_length=env.target_length)
        likelihood = float(np.exp(env.observation_log_probability(state, 0, [reading])[0]))
        posterior[cell] = prior[cell] * likelihood
    total = posterior.sum()
    return posterior / total if total > 0 else posterior


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------


def test_the_prior_is_uniform_over_the_cells_the_starting_body_leaves_free():
    """The belief starts from the initial state distribution, not from nothing.

    Test type: unit
    """
    env = build_env()
    belief = SnakeBelief.from_environment(env, n_particles=32)
    marginal = marginal_of(belief, env)
    body = env.initial_state_dist().body
    free = env.num_cells - len(body)
    assert marginal.sum() == pytest.approx(1.0)
    for row, col in body:
        assert marginal[row, col] == 0.0
    assert np.isclose(marginal[marginal > 0], 1.0 / free).all()


def test_the_prior_refuses_a_foreign_environment_and_a_useless_particle_count():
    """Constructor mistakes fail at construction rather than at the first update.

    Test type: unit
    """
    with pytest.raises(TypeError, match="SnakePOMDP"):
        SnakeBelief.from_environment(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_particles"):
        SnakeBelief.from_environment(build_env(), n_particles=0)


def test_the_belief_has_its_own_identity():
    """A generic particle belief holding the same particles is not this belief.

    Purpose: The episode result cache keys on the initial belief's
        ``config_id``. Sharing one with ``WeightedParticleBelief`` would answer
        an exact-versus-generic comparison from the wrong run.

    Test type: unit
    """
    env = build_env()
    belief = SnakeBelief.from_environment(env, n_particles=16)
    generic = WeightedParticleBelief(particles=belief.particles, log_weights=belief.log_weights)
    assert belief.config_id != generic.config_id


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------


def test_a_sighting_collapses_the_belief_onto_one_cell():
    """The window has no false positives, so a sighting is conclusive.

    Test type: unit
    """
    env = build_env()
    body = ((3, 3), (3, 2), (3, 1))
    prior = np.zeros(env.num_cells)
    free = env.free_cells(body)
    prior[free] = 1.0 / free.size
    belief = belief_over(env, body, prior)

    moved = ((3, 4), (3, 3), (3, 2))
    reading = env.encode_observation_tuple(moved, (4, 5), int(SnakeQuadrant.SOUTH_EAST))
    updated = belief.update(int(SnakeAction.GO_STRAIGHT), reading, env)
    marginal = marginal_of(updated, env)
    assert marginal[4, 5] == pytest.approx(1.0)
    assert marginal.sum() == pytest.approx(1.0)
    assert all(env.food(particle) == (4, 5) for particle in updated.particles)


def test_a_silent_window_makes_cells_inside_it_less_likely_without_excluding_them():
    """Absence of evidence is weak evidence, because detection is below one.

    Purpose: With ``detection_probability`` at 1 a silent window would rule the
        cells out. At 0.9 it must not, and a belief that zeroed them would
        eventually be supported on nothing.

    Test type: unit
    """
    env = build_env(detection_probability=0.9, scent_accuracy=0.25)
    body = ((3, 3), (3, 2), (3, 1))
    prior = np.zeros(env.num_cells)
    free = env.free_cells(body)
    prior[free] = 1.0 / free.size
    belief = belief_over(env, body, prior)

    moved = ((3, 4), (3, 3), (3, 2))
    reading = env.encode_observation_tuple(moved, None, int(SnakeQuadrant.SOUTH_EAST))
    marginal = marginal_of(belief.update(int(SnakeAction.GO_STRAIGHT), reading, env), env)
    inside = marginal[4, 5]
    outside = marginal[0, 0]
    assert 0.0 < inside < outside
    assert inside / outside == pytest.approx(0.1, abs=1e-9)


def test_the_new_head_cell_is_ruled_out_when_nothing_was_eaten():
    """Not eating proves the food is not where the head has just arrived.

    Purpose: This is the only way a non-eating step removes mass, and it is the
        one line of the formal model that looks redundant and is not.

    Test type: unit
    """
    env = build_env(detection_probability=0.0, scent_accuracy=0.25)
    body = ((3, 3), (3, 2), (3, 1))
    prior = np.zeros(env.num_cells)
    free = env.free_cells(body)
    prior[free] = 1.0 / free.size
    belief = belief_over(env, body, prior)
    assert marginal_of(belief, env)[3, 4] > 0.0

    moved = ((3, 4), (3, 3), (3, 2))
    reading = env.encode_observation_tuple(moved, None, int(SnakeQuadrant.SOUTH_EAST))
    marginal = marginal_of(belief.update(int(SnakeAction.GO_STRAIGHT), reading, env), env)
    assert marginal[3, 4] == 0.0
    assert marginal.sum() == pytest.approx(1.0)


def test_growing_restarts_the_prior_uniform_over_the_new_body_free_cells():
    """Eating replaces the food, so everything learnt about the old one is gone.

    Purpose: The respawn is the one place the belief must *discard* evidence.
        Carrying the old posterior through would leave the planner certain
        about a food item that no longer exists.
    Given: Two beliefs with the same body and wildly different posteriors --
        one certain the food is straight ahead, one uniform -- and the same
        reading reporting a grown body.
    When: Both update.
    Then: They land on the same posterior, which is the uniform respawn prior
        conditioned on the reading and zero on every body cell.

    Test type: unit
    """
    env = build_env(detection_probability=0.0, scent_accuracy=0.25)
    body = ((3, 3), (3, 2), (3, 1))
    certain = np.zeros(env.num_cells)
    certain[3 * env.grid_size + 4] = 1.0  # certain the food is straight ahead
    free_before = env.free_cells(body)
    uniform = np.zeros(env.num_cells)
    uniform[free_before] = 1.0 / free_before.size

    grown = ((3, 4), (3, 3), (3, 2), (3, 1))
    reading = env.encode_observation_tuple(grown, None, int(SnakeQuadrant.SOUTH_EAST))
    free_after = env.free_cells(grown)
    respawn_prior = np.zeros(env.num_cells)
    respawn_prior[free_after] = 1.0 / free_after.size
    expected = brute_force_posterior(env, respawn_prior, grown, None, int(SnakeQuadrant.SOUTH_EAST))

    for prior in (certain, uniform):
        updated = belief_over(env, body, prior, steps_since_food=5).update(
            int(SnakeAction.GO_STRAIGHT), reading, env
        )
        marginal = marginal_of(updated, env)
        assert marginal.sum() == pytest.approx(1.0)
        assert np.allclose(marginal.ravel(), expected)
        for row, col in grown:
            assert marginal[row, col] == 0.0
        assert env.steps_since_food(updated.particles[0]) == 0


def test_the_counter_advances_with_the_observed_length():
    """The counter is deterministic given the observed lengths, so it is tracked.

    Test type: unit
    """
    env = build_env(scent_accuracy=0.25, detection_probability=0.0)
    body = ((3, 3), (3, 2), (3, 1))
    prior = np.zeros(env.num_cells)
    free = env.free_cells(body)
    prior[free] = 1.0 / free.size
    belief = belief_over(env, body, prior, steps_since_food=4)
    moved = ((3, 4), (3, 3), (3, 2))
    reading = env.encode_observation_tuple(moved, None, int(SnakeQuadrant.SOUTH_EAST))
    updated = belief.update(int(SnakeAction.GO_STRAIGHT), reading, env)
    assert env.steps_since_food(updated.particles[0]) == 5


@pytest.mark.parametrize(
    "seen, scent",
    [
        (None, int(SnakeQuadrant.NORTH_EAST)),
        (None, int(SnakeQuadrant.SOUTH_WEST)),
        ((4, 5), int(SnakeQuadrant.SOUTH_EAST)),
        (None, int(SnakeQuadrant.NORTH_WEST)),
    ],
)
def test_the_update_matches_bayes_rule_applied_cell_by_cell(seen, scent):
    """The belief is the posterior, not an approximation of it.

    Purpose: The update is written in terms of the environment's parts rather
        than by calling its likelihood, for speed. This test closes that gap by
        recomputing the posterior from the likelihood the world actually uses.
    Given: A lopsided prior and a reading, on a grid small enough to enumerate.
    When: The belief updates and a brute-force Bayes pass runs on the same input.
    Then: The two posteriors agree to floating-point tolerance.

    Test type: integration
    """
    env = build_env()
    body = ((3, 3), (3, 2), (3, 1))
    prior = np.zeros(env.num_cells)
    free = env.free_cells(body)
    weights = np.linspace(1.0, 4.0, free.size)
    prior[free] = weights / weights.sum()
    belief = belief_over(env, body, prior)

    moved = ((3, 4), (3, 3), (3, 2))
    reading = env.encode_observation_tuple(moved, seen, scent)
    updated = belief.update(int(SnakeAction.GO_STRAIGHT), reading, env)
    expected = brute_force_posterior(env, prior, moved, seen, scent)
    assert np.allclose(marginal_of(updated, env).ravel(), expected)


def test_an_impossible_reading_is_refused_rather_than_absorbed():
    """A belief supported on nothing is the failure this class exists to prevent.

    Test type: unit
    """
    env = build_env()
    body = ((3, 3), (3, 2), (3, 1))
    prior = np.zeros(env.num_cells)
    prior[0] = 1.0  # certain the food is at (0, 0)
    belief = belief_over(env, body, prior)
    moved = ((3, 4), (3, 3), (3, 2))
    reading = env.encode_observation_tuple(moved, (4, 5), int(SnakeQuadrant.SOUTH_EAST))
    with pytest.raises(ValueError, match="disagree"):
        belief.update(int(SnakeAction.GO_STRAIGHT), reading, env)


def test_the_terminal_reading_leaves_every_particle_terminal():
    """A belief that reads as still running keeps a planner expanding a dead branch.

    Purpose: The generic update propagates each particle and weights it by the
        observation likelihood, but with resampling off a particle whose
        successor is still running survives at floor weight instead of being
        dropped. The belief then reads as live. Asserting only the returned
        class, as this test once did, does not catch that.

    Test type: unit
    """
    env = build_env()
    # Head at (0, 3) heading east; turning left points north, off the grid.
    body = ((0, 3), (0, 2), (0, 1))
    prior = np.zeros(env.num_cells)
    free = env.free_cells(body)
    prior[free] = 1.0 / free.size
    belief = belief_over(env, body, prior, n_particles=16)

    np.random.seed(0)
    updated = belief.update(int(SnakeAction.TURN_LEFT), TERMINAL_OBSERVATION, env)
    assert isinstance(updated, WeightedParticleBelief)
    assert all(env.is_terminal(particle) for particle in updated.particles)
    assert is_terminal_belief(updated, env)
    assert {env.termination(particle) for particle in updated.particles} == {SnakeTermination.WALL}


def test_a_winning_step_ends_the_belief_even_though_only_one_food_cell_wins():
    """The win is the case the generic update gets wrong, because eating splits the particles.

    Purpose: Walls, self-collisions and starvation do not depend on the food, so
        every particle ends together and any update looks right. Winning
        requires eating, so exactly one food cell ends the episode and the rest
        keep running -- this is where a propagate-and-weight update leaves live
        particles behind.

    Test type: unit
    """
    env = build_env(target_length=4, starvation_limit=100)
    body = ((3, 3), (3, 2), (3, 1))
    prior = np.full(env.num_cells, 1.0 / (env.num_cells - len(body)))
    for row, col in body:
        prior[row * env.grid_size + col] = 0.0
    prior /= prior.sum()
    belief = belief_over(env, body, prior, n_particles=64)

    # Going straight moves the head to (3, 4). Only food there is eaten, and
    # eating takes the body to length 4, which is the target.
    np.random.seed(0)
    updated = belief.update(int(SnakeAction.GO_STRAIGHT), TERMINAL_OBSERVATION, env)

    assert is_terminal_belief(updated, env)
    assert all(env.is_terminal(particle) for particle in updated.particles)
    assert {env.termination(particle) for particle in updated.particles} == {SnakeTermination.WIN}
    assert all(len(env.body(particle)) == env.target_length for particle in updated.particles)


def test_a_terminal_reading_no_food_cell_explains_is_refused():
    """The terminal reading is evidence, so it can contradict the belief like any other.

    Test type: unit
    """
    env = build_env(target_length=6, starvation_limit=100)
    body = ((3, 3), (3, 2), (3, 1))
    prior = np.zeros(env.num_cells)
    prior[0 * env.grid_size + 0] = 1.0  # food at (0, 0), far from the head
    belief = belief_over(env, body, prior)
    # Going straight moves the head to (3, 4): no wall, no self, no win at
    # length 3 of 6, and the counter is nowhere near the starvation limit. So
    # nothing the belief holds could have ended this episode.
    with pytest.raises(ValueError, match="disagree"):
        belief.update(int(SnakeAction.GO_STRAIGHT), TERMINAL_OBSERVATION, env)


def test_a_belief_rebuilt_from_its_particles_alone_still_updates():
    """The particle histogram is the documented fallback, not a silent failure.

    Purpose: A belief that crossed a process boundary without its exact
        distribution must degrade to a Monte Carlo summary rather than raise.

    Test type: unit
    """
    env = build_env(detection_probability=0.0, scent_accuracy=0.25)
    belief = SnakeBelief.from_environment(env, n_particles=256)
    stripped = SnakeBelief(particles=belief.particles, log_weights=belief.log_weights)
    body = env.initial_state_dist().body
    moved = ((body[0][0], body[0][1] + 1), body[0], body[1])
    reading = env.encode_observation_tuple(moved, None, int(SnakeQuadrant.SOUTH_EAST))
    updated = stripped.update(int(SnakeAction.GO_STRAIGHT), reading, env)
    assert marginal_of(updated, env).sum() == pytest.approx(1.0)


def test_particles_are_legal_states_drawn_from_the_posterior():
    """Every particle is a state the world could be in, with the observed body.

    Test type: integration
    """
    env = build_env()
    belief = SnakeBelief.from_environment(env, n_particles=200)
    body = env.initial_state_dist().body
    moved = ((body[0][0], body[0][1] + 1), body[0], body[1])
    reading = env.encode_observation_tuple(moved, None, int(SnakeQuadrant.SOUTH_EAST))
    np.random.seed(0)
    updated = belief.update(int(SnakeAction.GO_STRAIGHT), reading, env)
    for particle in updated.particles:
        assert env.body(particle) == moved
        assert env.food(particle) not in moved
        assert not env.is_terminal(particle)


def test_serialization_preserves_the_exact_posterior():
    """The distribution survives ``to_dict``; the particles alone would not.

    Purpose: A finite sample cannot be inverted back into the distribution it
        came from, so a belief serialized without this field returns as a Monte
        Carlo summary of itself and the visualization's belief layer loses its
        resolution.

    Test type: unit
    """
    env = build_env()
    belief = SnakeBelief.from_environment(env, n_particles=8)
    stored = belief.to_dict()
    assert "food_probabilities" in stored
    assert np.allclose(np.asarray(stored["food_probabilities"]), belief.food_probabilities)
    rebuilt = SnakeBelief(
        particles=np.asarray(stored["particles"], dtype=np.float64),
        log_weights=np.asarray(stored["log_weights"], dtype=np.float64),
        food_probabilities=np.asarray(stored["food_probabilities"], dtype=np.float64),
    )
    assert np.allclose(rebuilt.marginal(env), belief.marginal(env))
