# SPDX-License-Identifier: MIT

"""Tests for the multi-agent firefighting vectorized particle belief.

The fire is stochastic in four separate places, so the batched transition is
checked two ways: against the environment's own step under a preset that turns
every draw off, and against its transition probability on the full model, where
only reachability is checkable. The observation model is deterministic given
the state and is compared directly.
"""

from typing import cast

import numpy as np
import pytest

from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.multiagent_firefighting_pomdp import (
    FirefightingVectorizedBelief,
    FirefightingVectorizedUpdater,
    MultiAgentFirefightingPOMDP,
)
from POMDPPlanners.utils.belief_factory import BeliefType, create_environment_belief


@pytest.fixture
def env():
    return MultiAgentFirefightingPOMDP(discount_factor=0.95)


@pytest.fixture
def updater(env):
    return FirefightingVectorizedUpdater.from_environment(env)


def _reachable_states(env, count, rng):
    states = []
    for state in env.initial_state_dist().sample(n_samples=count):
        for _ in range(int(rng.integers(0, 5))):
            state = env.sample_next_state(state, int(rng.integers(0, env.num_actions)))
        states.append(np.asarray(state, dtype=np.float64))
    return np.stack(states)


class TestFirefightingVectorizedUpdater:
    def test_deterministic_preset_matches_the_environment(self):
        """Test that the batched step equals the scalar one with every draw off.

        Purpose: Six stages resolve in a fixed order, and the order is the
        model -- suppression before spread, growth only on cells alight before
        the spread. With the draws switched off the whole procedure becomes
        comparable value for value, which is the only test that pins the order.

        Given: An environment with no slip, spread, growth or burnout and
            certain suppression.
        When: Reachable states are stepped through both paths.
        Then: The successors are identical.

        Test type: unit
        """
        quiet = MultiAgentFirefightingPOMDP(
            discount_factor=0.95,
            slip_probability=0.0,
            growth_probability=0.0,
            burnout_probability=0.0,
            spread_probability=0.0,
            suppression_probability_smoldering=1.0,
            suppression_probability_burning=1.0,
        )
        updater = FirefightingVectorizedUpdater.from_environment(quiet)
        rng = np.random.default_rng(0)
        states = _reachable_states(quiet, 40, rng)

        for action in rng.integers(0, quiet.num_actions, size=10):
            np.testing.assert_array_equal(
                updater.batch_transition(states, int(action)),
                np.stack([quiet.sample_next_state(row, int(action)) for row in states]),
                err_msg=f"action {action}",
            )

    def test_every_sampled_successor_is_reachable(self, env, updater):
        """Test that the batched step never invents a successor.

        Purpose: With the draws switched on, only the support is checkable --
        but a stage applied in the wrong order, or to the wrong map, produces
        states the world cannot reach.

        Given: Reachable states and several joint actions.
        When: The batched step is taken.
        Then: Every successor has positive probability under the environment.

        Test type: unit
        """
        rng = np.random.default_rng(1)
        states = _reachable_states(env, 30, rng)

        for action in rng.integers(0, env.num_actions, size=6):
            batched = updater.batch_transition(states, int(action))
            for index in range(0, len(states), 5):
                score = env.transition_log_probability(
                    states[index], int(action), [batched[index]]
                )[0]
                assert np.isfinite(
                    score
                ), f"action {action}, row {index}: successor has zero probability"

    def test_observation_log_likelihood_matches_the_environment(self, env, updater):
        """Test that the batched likelihood equals the environment's.

        Purpose: The reading is a delta on the robot fields and a confusion
        matrix on every visible cell. Both halves decide which fire maps
        survive, and a mismatch in either reweights the wind posterior.

        Given: Reachable states and readings drawn from several of them.
        When: Each reading is scored through both paths.
        Then: The log-likelihoods agree, infinities included.

        Test type: unit
        """
        rng = np.random.default_rng(2)
        states = _reachable_states(env, 30, rng)
        np.random.seed(3)

        for source in states[:12]:
            observation = env.sample_observation(source, 0)
            batched = updater.batch_observation_log_likelihood(states, 0, observation)
            scalar = np.array(
                [env.observation_log_probability(row, 0, [observation])[0] for row in states]
            )
            np.testing.assert_array_equal(np.isfinite(batched), np.isfinite(scalar))
            finite = np.isfinite(scalar)
            np.testing.assert_allclose(batched[finite], scalar[finite], atol=1e-9)

    def test_wind_is_carried_unchanged(self, env, updater):
        """Test that a step never moves the wind.

        Purpose: The wind is drawn once at reset and held for the episode --
        that is what makes it identifiable from a whole episode's spread
        pattern, and what makes it worth stratifying the belief by.

        Given: Reachable states covering several wind values.
        When: A step is taken.
        Then: Both wind fields come back unchanged.

        Test type: unit
        """
        states = _reachable_states(env, 20, np.random.default_rng(4))

        successors = updater.batch_transition(states, 0)

        np.testing.assert_array_equal(
            successors[:, [env.wind_direction_index, env.wind_strength_index]],
            states[:, [env.wind_direction_index, env.wind_strength_index]],
        )


class TestFirefightingVectorizedBelief:
    def test_default_belief_is_the_vectorized_one(self, env):
        """Test that the factory hands back the batched belief by default.

        Purpose: Registration is what makes the updater reachable; without it
        this environment falls back to a generic filter that dies on the
        exactly-reported robot fields.

        Given: A firefighting environment.
        When: The top-level factory is asked for its belief.
        Then: The vectorized belief comes back.

        Test type: unit
        """
        assert isinstance(
            create_environment_belief(env, n_particles=32), FirefightingVectorizedBelief
        )

    def test_particle_type_still_available(self, env):
        """Test that the generic particle belief is still selectable.

        Purpose: The vectorized belief is the default, not the only option.

        Given: A firefighting environment.
        When: ``BeliefType.PARTICLE`` is requested.
        Then: Something other than the vectorized belief comes back.

        Test type: unit
        """
        belief = create_environment_belief(env, belief_type=BeliefType.PARTICLE, n_particles=32)
        assert not isinstance(belief, FirefightingVectorizedBelief)

    def test_update_conditions_on_the_reported_robot_fields(self, env):
        """Test that every particle adopts the poses, tanks and healths reported.

        Purpose: Those come back without noise but depend on hidden state --
        heat damage is read off a cell the robot may not have seen -- so
        weighting by them empties the particle set instead of informing it.

        Given: The prior belief and one step of an episode.
        When: The reading is conditioned on.
        Then: Every particle carries the reported robot fields.

        Test type: integration
        """
        np.random.seed(0)
        belief = cast(
            VectorizedWeightedParticleBelief,
            create_environment_belief(env, n_particles=64),
        )
        state = env.initial_state_dist().sample()[0]

        next_state = env.sample_next_state(state, 0)
        observation = np.asarray(env.sample_observation(next_state, 0), dtype=np.float64)
        belief = belief.update(0, observation, env)

        width = 4 * env.num_robots
        fields = belief.particles[:, 1 : 1 + width]
        np.testing.assert_array_equal(fields, np.tile(observation[:width], (64, 1)))

    def test_no_wind_value_is_resampled_away(self, env):
        """Test that every wind value keeps its particles through an episode.

        Purpose: The wind never changes, so a value resampled to zero particles
        can never come back, however clearly later spread favours it.

        Given: The prior belief, which covers the wind values it drew.
        When: Ten steps of an episode are conditioned on.
        Then: Every wind value holds the particles it started with.

        Test type: integration
        """
        np.random.seed(0)
        belief = cast(
            VectorizedWeightedParticleBelief,
            create_environment_belief(env, n_particles=200),
        )
        keys = (
            belief.particles[:, env.wind_direction_index] * 2
            + belief.particles[:, env.wind_strength_index]
        )
        before = np.bincount(keys.astype(int), minlength=8)
        state = env.initial_state_dist().sample()[0]

        for _ in range(10):
            action = int(np.random.randint(0, env.num_actions))
            next_state = env.sample_next_state(state, action)
            belief = belief.update(action, env.sample_observation(next_state, action), env)
            state = next_state
            if env.is_terminal(state):
                break

        keys = (
            belief.particles[:, env.wind_direction_index] * 2
            + belief.particles[:, env.wind_strength_index]
        )
        np.testing.assert_array_equal(np.bincount(keys.astype(int), minlength=8), before)

    def test_belief_learns_something_about_the_wind(self, env):
        """Test that the belief beats its own prior on the hidden wind.

        Purpose: The wind is the whole inference problem here, and it is only
        identifiable through which neighbours catch -- so with random actions
        the evidence arrives slowly. The mean over episodes is what is
        meaningful, not any single run.

        Given: Six 30-step episodes with 400 particles and random actions.
        When: Every reading is conditioned on.
        Then: The mean weight on the true wind beats the 0.125 prior.

        Test type: integration
        """
        masses = []
        for seed in range(6):
            np.random.seed(seed)
            belief = cast(
                VectorizedWeightedParticleBelief,
                create_environment_belief(env, n_particles=400),
            )
            state = env.initial_state_dist().sample()[0]
            truth = (
                int(round(state[env.wind_direction_index])),
                int(round(state[env.wind_strength_index])),
            )
            for _ in range(30):
                action = int(np.random.randint(0, env.num_actions))
                next_state = env.sample_next_state(state, action)
                belief = belief.update(action, env.sample_observation(next_state, action), env)
                state = next_state
                if env.is_terminal(state):
                    break
            same = (np.rint(belief.particles[:, env.wind_direction_index]) == truth[0]) & (
                np.rint(belief.particles[:, env.wind_strength_index]) == truth[1]
            )
            masses.append(float(belief.normalized_weights[same].sum()))

        assert np.mean(masses) > 0.125, f"mean weight on the true wind was {np.mean(masses)}"
