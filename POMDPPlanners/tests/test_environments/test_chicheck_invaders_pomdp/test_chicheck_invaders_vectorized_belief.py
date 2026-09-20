# SPDX-License-Identifier: MIT

"""Tests for the Chicheck Invaders vectorized particle belief.

The batched updater is a copy of the environment's step and sensor model, so
the parity tests here are the ones that keep the copy honest. The transition is
stochastic only in the dive coins, and the environment can say which coins a
given successor implies -- so the batched successor is replayed through the
scalar step with those coins and must come back byte for byte.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.chicheck_invaders_pomdp import (
    ChicheckInvadersAction,
    ChicheckInvadersPOMDP,
    ChicheckInvadersVectorizedBelief,
    ChicheckInvadersVectorizedUpdater,
    ObservationMode,
    chicken_slots,
    noiseless_preset,
)
from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_pomdp import (
    IMPOSSIBLE_LOG_PROBABILITY,
)
from POMDPPlanners.utils.belief_factory import BeliefType, create_environment_belief


@pytest.fixture
def env():
    return ChicheckInvadersPOMDP(discount_factor=0.95)


@pytest.fixture
def updater(env):
    return ChicheckInvadersVectorizedUpdater.from_environment(env)


def _reachable_states(env, count, rng):
    states = []
    for state in env.initial_state_dist().sample(n_samples=count):
        for _ in range(int(rng.integers(0, 8))):
            state = env.sample_next_state(state, int(rng.integers(0, 4)))
        states.append(np.asarray(state, dtype=np.float64))
    return np.stack(states)


class TestChicheckInvadersVectorizedUpdater:
    def test_batch_transition_replays_through_the_scalar_step(self, env, updater):
        """Test that each batched successor is one the scalar step produces.

        Purpose: The only randomness is the dive coins, and the environment can
        recover which coins a successor implies. Replaying them through the
        scalar step turns a stochastic transition into an exact comparison.

        Given: Reachable states and every action.
        When: The batched step is taken and its coins replayed scalar-side.
        Then: The two successors are identical.

        Test type: unit
        """
        states = _reachable_states(env, 50, np.random.default_rng(0))

        for action in [int(value) for value in ChicheckInvadersAction]:
            batched = updater.batch_transition(states, action)
            for index, state in enumerate(states):
                before = chicken_slots(state, env.num_chickens)
                # pylint: disable-next=protected-access
                switches = env._implied_dive_switches(before, batched[index])
                assert switches is not None, f"malformed successor for action {action}"
                # pylint: disable-next=protected-access
                scalar = env._step_once(state, action, switches=switches)
                np.testing.assert_array_equal(
                    batched[index], scalar, err_msg=f"action {action}, row {index}"
                )

    def test_observation_log_likelihood_matches_the_environment(self, env, updater):
        """Test that the batched likelihood equals the environment's.

        Purpose: Every chicken contributes a factor whether it was reported or
        not, so a copy that dropped the silent ones would quietly stop using
        silence as evidence.

        Given: Reachable states and readings drawn from several of them.
        When: Each reading is scored through both paths.
        Then: The scores agree, and the impossible floor lands on the same
            particles.

        Test type: unit
        """
        states = _reachable_states(env, 40, np.random.default_rng(1))
        np.random.seed(2)

        for source in states[:15]:
            observation = env.sample_observation(source, 0)
            batched = updater.batch_observation_log_likelihood(states, 0, observation)
            scalar = np.asarray(env.observation_log_probability_per_state(states, 0, observation))
            floored = scalar <= IMPOSSIBLE_LOG_PROBABILITY
            np.testing.assert_array_equal(batched <= IMPOSSIBLE_LOG_PROBABILITY, floored)
            np.testing.assert_allclose(batched[~floored], scalar[~floored], atol=1e-9)

    def test_config_id_separates_flock_sizes(self, env, updater):
        """Test that the identity covers the flock and the sensors.

        Purpose: ``config_id`` is a cache key; a key blind to the flock size
        would serve one world's cached beliefs for another.

        Given: Two updaters on one environment and one on a larger flock.
        When: Their identities are compared.
        Then: The pair agrees and the larger flock does not.

        Test type: unit
        """
        same = ChicheckInvadersVectorizedUpdater.from_environment(env)
        other = ChicheckInvadersVectorizedUpdater.from_environment(
            ChicheckInvadersPOMDP(discount_factor=0.95, num_chickens=env.num_chickens + 1)
        )

        assert updater.config_id == same.config_id
        assert updater.config_id != other.config_id


class TestChicheckInvadersVectorizedBelief:
    def test_default_belief_is_the_vectorized_one(self, env):
        """Test that the factory hands back the batched belief by default.

        Purpose: Registration is what makes the updater reachable.

        Given: A Chicheck Invaders environment.
        When: The top-level factory is asked for its belief.
        Then: The vectorized belief comes back.

        Test type: unit
        """
        assert isinstance(
            create_environment_belief(env, n_particles=32), ChicheckInvadersVectorizedBelief
        )

    def test_scalar_filter_still_available(self, env):
        """Test that the scalar filter is still selectable.

        Purpose: The two filters are meant to be comparable; the batched one
        being the default must not remove the other.

        Given: A Chicheck Invaders environment.
        When: ``BeliefType.PARTICLE`` is requested.
        Then: Something other than the vectorized belief comes back.

        Test type: unit
        """
        belief = create_environment_belief(env, belief_type=BeliefType.PARTICLE, n_particles=32)
        assert not isinstance(belief, ChicheckInvadersVectorizedBelief)

    def test_fully_observable_belief_collapses_onto_the_state(self):
        """Test that a fully observable episode gives a point mass.

        Purpose: In that mode the observation *is* the state and the likelihood
        is one only on an exact match, so every particle from the flock prior
        floors -- and a floored vector normalises to uniform, which looks
        exactly like a healthy prior.

        Given: A fully observable environment and its prior belief.
        When: One step is taken and its reading conditioned on.
        Then: Every particle equals the observed state.

        Test type: integration
        """
        env = ChicheckInvadersPOMDP(discount_factor=0.95, observation_mode=ObservationMode.FULL)
        np.random.seed(1)
        belief = create_environment_belief(env, n_particles=32)
        state = env.initial_state_dist().sample()[0]

        next_state = env.sample_next_state(state, 0)
        observation = env.sample_observation(next_state, 0)
        belief = belief.update(0, observation, env)

        np.testing.assert_array_equal(
            belief.particles, np.tile(np.asarray(observation, dtype=np.float64), (32, 1))
        )

    def test_noiseless_preset_keeps_consistent_particles(self):
        """Test that the belief rebuilds when a sharp reading rules everything out.

        Purpose: Under the noiseless preset a single contradicted slot floors a
        particle, so the whole set can die in one step. Resampling cannot help
        -- it draws from particles that are already wrong.

        Given: The noiseless preset and several steps of an episode.
        When: Each reading is conditioned on.
        Then: Every particle can still explain the last reading.

        Test type: integration
        """
        env = ChicheckInvadersPOMDP(discount_factor=0.95, **noiseless_preset())
        np.random.seed(2)
        belief = create_environment_belief(env, n_particles=64)
        state = env.initial_state_dist().sample()[0]

        scores = np.zeros(64)
        for _ in range(6):
            action = int(np.random.randint(0, 4))
            next_state = env.sample_next_state(state, action)
            observation = env.sample_observation(next_state, action)
            belief = belief.update(action, observation, env)
            scores = np.asarray(
                env.observation_log_probability_per_state(list(belief.particles), None, observation)
            )
            state = next_state

        assert np.all(scores > IMPOSSIBLE_LOG_PROBABILITY)

    def test_config_id_covers_the_refresh_fraction(self, env):
        """Test that two beliefs refreshing differently have different identities.

        Purpose: They hold the same particles but behave differently on the
        next step, and ``config_id`` keys the episode result cache.

        Given: Two beliefs with identical particles and different fractions.
        When: Their identities are compared.
        Then: They differ.

        Test type: unit
        """
        np.random.seed(0)
        particles = np.stack(env.initial_state_dist().sample(n_samples=16))
        log_weights = np.full(16, -float(np.log(16)))
        updater = ChicheckInvadersVectorizedUpdater.from_environment(env)

        first = ChicheckInvadersVectorizedBelief(
            particles=particles,
            log_weights=log_weights,
            updater=updater,
            reinvigoration_fraction=0.1,
        )
        second = ChicheckInvadersVectorizedBelief(
            particles=particles,
            log_weights=log_weights,
            updater=updater,
            reinvigoration_fraction=0.3,
        )

        assert first.config_id != second.config_id
