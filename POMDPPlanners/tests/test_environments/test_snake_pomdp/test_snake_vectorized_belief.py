# SPDX-License-Identifier: MIT

"""Tests for the Snake vectorized particle belief.

The updater reimplements the environment's transition and sensors in NumPy, so
most of what follows is parity: the batched step must agree with the scalar one
on the deterministic half, and its food respawns must land where the
environment would let them. The rest covers the redraw, which is what keeps a
sighting -- a reading with no false positives -- from emptying the filter.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.snake_pomdp import (
    SnakeAction,
    SnakePOMDP,
    SnakeTermination,
    SnakeVectorizedUpdater,
    SnakeVectorizedWeightedParticleBelief,
    create_snake_state,
)
from POMDPPlanners.utils.belief_factory import BeliefType, create_environment_belief


@pytest.fixture
def env():
    return SnakePOMDP(discount_factor=0.95)


@pytest.fixture
def updater(env):
    return SnakeVectorizedUpdater.from_environment(env)


def _reachable_states(env, count, rng):
    """States gathered by walking random actions out of the prior."""
    states = []
    for state in env.initial_state_dist().sample(n_samples=count):
        for _ in range(int(rng.integers(0, 6))):
            if env.is_terminal(state):
                break
            state = env.sample_next_state(state, int(rng.integers(0, 3)))
        states.append(np.asarray(state, dtype=np.float64))
    return np.stack(states)


class TestSnakeVectorizedUpdater:
    def test_body_and_bookkeeping_match_the_environment(self, env, updater):
        """Test that the batched step moves the snake exactly as the world does.

        Purpose: Everything but the food respawn is deterministic, so it is
        checkable outright -- and it is the half that decides when the snake
        dies.

        Given: Reachable states and every action.
        When: Each action is applied through both paths.
        Then: The status, length, counter and body agree exactly.

        Test type: unit
        """
        states = _reachable_states(env, 120, np.random.default_rng(0))

        for action in [int(value) for value in SnakeAction]:
            batched = updater.batch_transition(states, action)
            for index, state in enumerate(states):
                scalar = env.sample_next_state(state, action)
                np.testing.assert_array_equal(
                    batched[index][:3], scalar[:3], err_msg=f"action {action}, row {index}"
                )
                np.testing.assert_array_equal(
                    batched[index][5:], scalar[5:], err_msg=f"action {action}, row {index}"
                )

    def test_respawned_food_lands_on_a_free_cell(self, env, updater):
        """Test that eating respawns the food where the environment allows.

        Purpose: The respawn is the one stochastic part of the transition, so
        it cannot be compared value for value -- only its support can.

        Given: Reachable states and every action.
        When: The batched step is taken.
        Then: A step that ate and continued puts the food on a cell the new
            body leaves free; a win names no cell; anything else keeps the
            cell it had.

        Test type: unit
        """
        np.random.seed(7)
        states = _reachable_states(env, 80, np.random.default_rng(1))

        for action in [int(value) for value in SnakeAction]:
            batched = updater.batch_transition(states, action)
            for index, state in enumerate(states):
                if env.is_terminal(state):
                    np.testing.assert_array_equal(batched[index], state)
                    continue
                new_body, ate, _, termination = env.transition_outcome(state, action)
                food = (int(batched[index][3]), int(batched[index][4]))
                if termination is SnakeTermination.WIN:
                    assert food == (-1, -1)
                elif ate and termination is SnakeTermination.RUNNING:
                    free = set(env.free_cells(new_body).tolist())
                    assert food[0] * env.grid_size + food[1] in free
                else:
                    assert food == (int(state[3]), int(state[4]))

    def test_observation_log_likelihood_matches_the_environment(self, env, updater):
        """Test that the batched likelihood equals the environment's.

        Purpose: The reading has three independent parts and the belief
        multiplies all three; a disagreement in any of them reweights the food
        distribution wrongly.

        Given: Reachable states and readings drawn from several of them.
        When: Each reading is scored through both paths.
        Then: The log-likelihoods agree, infinities included.

        Test type: unit
        """
        states = _reachable_states(env, 60, np.random.default_rng(2))
        np.random.seed(3)

        for source in states[:20]:
            observation = env.sample_observation(source, 0)
            batched = updater.batch_observation_log_likelihood(
                states, 0, np.asarray(observation, dtype=np.float64)
            )
            scalar = np.array(
                [env.observation_log_probability(row, 0, [observation])[0] for row in states]
            )
            np.testing.assert_allclose(batched, scalar, atol=1e-12)

    def test_terminal_reading_is_a_point_mass_on_terminal_states(self, env, updater):
        """Test that the terminal reading rules out every live state.

        Purpose: Every terminal state emits one fixed reading and nothing else
        emits it, so it separates the two halves of the particle set outright.

        Given: A live state and a dead one.
        When: The terminal reading is scored against both.
        Then: The dead state scores zero and the live one ``-inf``.

        Test type: unit
        """
        live = env.initial_state_dist().sample()[0]
        dead = create_snake_state(
            body=[(2, 2), (2, 1)],
            food=(4, 4),
            target_length=env.target_length,
            status=int(SnakeTermination.WALL),
        )
        states = np.stack([live, dead])

        scores = updater.batch_observation_log_likelihood(states, 0, np.array([0.0]))

        assert np.isneginf(scores[0])
        assert scores[1] == 0.0


class TestSnakeVectorizedBelief:
    def test_default_belief_is_the_vectorized_one(self, env):
        """Test that the factory hands back the batched belief by default.

        Purpose: Registration is what makes the updater reachable.

        Given: A Snake environment.
        When: The top-level factory is asked for its belief.
        Then: The vectorized belief comes back.

        Test type: unit
        """
        assert isinstance(
            create_environment_belief(env, n_particles=32),
            SnakeVectorizedWeightedParticleBelief,
        )

    def test_particle_type_still_available(self, env):
        """Test that the generic particle belief is still selectable.

        Purpose: The vectorized belief is the default, not the only option.

        Given: A Snake environment.
        When: ``BeliefType.PARTICLE`` is requested.
        Then: Something other than the vectorized belief comes back.

        Test type: unit
        """
        belief = create_environment_belief(env, belief_type=BeliefType.PARTICLE, n_particles=32)
        assert not isinstance(belief, SnakeVectorizedWeightedParticleBelief)

    def test_a_sighting_puts_every_particle_on_the_sighted_cell(self, env):
        """Test that a sighting collapses the belief onto one cell.

        Purpose: The window has no false positives, so a sighting is
        conclusive. A filter that held no particle there would floor every
        weight and resample impossible cells instead.

        Given: A belief whose particles are all on the wrong food cell.
        When: A reading that sights the food elsewhere arrives.
        Then: Every particle carries the sighted cell.

        Test type: integration
        """
        np.random.seed(0)
        belief = create_environment_belief(env, n_particles=32)
        state = env.initial_state_dist().sample()[0]
        head = env.body(state)[0]
        sighted = (head[0] + 1, head[1] + 1)
        belief.particles[:, 3] = float(head[0] - 2)
        belief.particles[:, 4] = float(head[1] - 2)

        reading = env.encode_observation_tuple(env.body(state), sighted, 0)
        updated = belief.update(int(SnakeAction.GO_STRAIGHT), reading, env)

        assert np.all(updated.particles[:, 3] == sighted[0])
        assert np.all(updated.particles[:, 4] == sighted[1])

    def test_belief_survives_an_episode(self, env):
        """Test that the filter keeps a usable belief for a whole episode.

        Purpose: The failure this belief guards against is silent -- weights
        that normalise to uniform over particles already ruled out look exactly
        like a healthy prior.

        Given: A belief and an episode of random actions.
        When: Every reading is conditioned on.
        Then: The particle count holds and no weight is NaN.

        Test type: integration
        """
        np.random.seed(0)
        belief = create_environment_belief(env, n_particles=100)
        state = env.initial_state_dist().sample()[0]

        for _ in range(25):
            if env.is_terminal(state):
                break
            action = int(np.random.randint(0, 3))
            next_state = env.sample_next_state(state, action)
            belief = belief.update(action, env.sample_observation(next_state, action), env)
            state = next_state

        assert belief.particles.shape[0] == 100
        assert not np.any(np.isnan(belief.log_weights))
