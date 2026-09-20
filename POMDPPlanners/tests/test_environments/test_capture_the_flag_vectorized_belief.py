# SPDX-License-Identifier: MIT

"""Tests for the CaptureTheFlag vectorized particle belief.

The transition here is stochastic in both teams' moves, so it cannot be
compared value for value against the environment. It is checked two ways
instead: every successor the batched sampler produces must be one the
environment gives positive probability, and over many draws the frequencies
must match the environment's own enumerated distribution. The observation
model is deterministic given the state and is compared directly.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.capture_the_flag_pomdp import (
    CaptureTheFlagPOMDP,
    CaptureTheFlagVectorizedBelief,
    CaptureTheFlagVectorizedUpdater,
)
from POMDPPlanners.utils.belief_factory import BeliefType, create_environment_belief


@pytest.fixture
def env():
    return CaptureTheFlagPOMDP(discount_factor=0.95)


@pytest.fixture
def updater(env):
    return CaptureTheFlagVectorizedUpdater.from_environment(env)


def _reachable_states(env, count, rng):
    states = []
    for state in env.initial_state_dist().sample(n_samples=count):
        for _ in range(int(rng.integers(0, 6))):
            state = env.sample_next_state(state, int(rng.integers(0, len(env.get_actions()))))
        states.append(np.asarray(state, dtype=np.float64))
    return np.stack(states)


class TestCaptureTheFlagVectorizedUpdater:
    def test_every_sampled_successor_is_reachable(self, env, updater):
        """Test that the batched step never invents a successor.

        Purpose: The stages -- pick-up, tagging, scoring, counters -- are
        order-dependent, and a copy that reorders them produces states the
        world cannot reach. That is invisible to a frequency test on any single
        state but not to this one.

        Given: Reachable states and several joint actions.
        When: The batched step is taken.
        Then: Every successor has positive probability under the environment.

        Test type: unit
        """
        rng = np.random.default_rng(0)
        states = _reachable_states(env, 40, rng)

        for action in rng.integers(0, len(env.get_actions()), size=8):
            batched = updater.batch_transition(states, int(action))
            for index in range(0, len(states), 5):
                score = env.transition_log_probability(
                    states[index], int(action), [batched[index]]
                )[0]
                assert np.isfinite(
                    score
                ), f"action {action}, row {index}: successor has zero probability"

    def test_sampled_frequencies_match_the_enumerated_distribution(self, env, updater):
        """Test that the batched sampler draws from the right distribution.

        Purpose: Reachability alone would pass a sampler that always took the
        most likely branch. This checks the weights too.

        Given: One state, one action, and the environment's enumeration of
            every successor with its probability.
        When: Ten thousand batched successors are drawn from that state.
        Then: Each successor's frequency is within 0.02 of its probability, and
            nothing outside the support appears.

        Test type: unit
        """
        np.random.seed(0)
        state = env.initial_state_dist().sample()[0]
        action = 0
        # pylint: disable-next=protected-access
        successors, probabilities = env._successor_distribution(state, action)
        expected = {
            successor.tobytes(): probability
            for successor, probability in zip(successors, probabilities)
        }

        drawn = updater.batch_transition(np.tile(state, (10000, 1)), action)
        counts: dict = {}
        for row in drawn:
            counts[row.tobytes()] = counts.get(row.tobytes(), 0) + 1

        assert not set(counts) - set(expected), "sampler produced unreachable successors"
        for key, probability in expected.items():
            assert abs(counts.get(key, 0) / len(drawn) - probability) < 0.02

    def test_observation_log_likelihood_matches_the_environment(self, env, updater):
        """Test that the batched likelihood equals the environment's.

        Purpose: Every blue player measures every red player, so the weight is
        a product of many factors; one wrong factor is enough to reweight the
        flag posterior the wrong way.

        Given: Reachable states and readings drawn from several of them.
        When: Each reading is scored through both paths.
        Then: The log-likelihoods agree, infinities included.

        Test type: unit
        """
        rng = np.random.default_rng(1)
        states = _reachable_states(env, 40, rng)
        np.random.seed(2)

        for _ in range(10):
            action = int(rng.integers(0, len(env.get_actions())))
            source = states[int(rng.integers(0, len(states)))]
            observation = env.sample_observation(source, action)
            batched = updater.batch_observation_log_likelihood(states, action, observation)
            scalar = np.array(
                [env.observation_log_probability(row, action, [observation])[0] for row in states]
            )
            np.testing.assert_array_equal(np.isfinite(batched), np.isfinite(scalar))
            finite = np.isfinite(scalar)
            np.testing.assert_allclose(batched[finite], scalar[finite], atol=1e-9)

    def test_terminal_particles_are_absorbing(self, env, updater):
        """Test that a won game never steps again.

        Purpose: An over-long rollout must not walk a particle back out of a
        terminal state.

        Given: A particle whose blue score has reached the winning score.
        When: Every action is applied.
        Then: The particle comes back unchanged.

        Test type: unit
        """
        state = np.asarray(env.initial_state_dist().sample()[0], dtype=np.float64)
        state[env.layout.score_blue] = float(env.score_to_win)
        states = state[np.newaxis, :]

        for action in range(min(6, len(env.get_actions()))):
            np.testing.assert_array_equal(updater.batch_transition(states, action), states)


class TestCaptureTheFlagVectorizedBelief:
    def test_default_belief_is_the_vectorized_one(self, env):
        """Test that the factory hands back the batched belief by default.

        Purpose: Registration is what makes the updater reachable; without it
        this environment falls back to a generic filter that dies on the
        exactly-observed components.

        Given: A CaptureTheFlag environment.
        When: The top-level factory is asked for its belief.
        Then: The vectorized belief comes back.

        Test type: unit
        """
        assert isinstance(
            create_environment_belief(env, n_particles=32), CaptureTheFlagVectorizedBelief
        )

    def test_particle_type_still_available(self, env):
        """Test that the generic particle belief is still selectable.

        Purpose: The vectorized belief is the default, not the only option.

        Given: A CaptureTheFlag environment.
        When: ``BeliefType.PARTICLE`` is requested.
        Then: Something other than the vectorized belief comes back.

        Test type: unit
        """
        belief = create_environment_belief(env, belief_type=BeliefType.PARTICLE, n_particles=32)
        assert not isinstance(belief, CaptureTheFlagVectorizedBelief)

    def test_update_conditions_on_the_observed_blue_positions(self, env):
        """Test that every particle adopts the blue positions the sensor reported.

        Purpose: Blue's own positions come back without noise, so the posterior
        puts all its mass on them. Weighting by them instead would floor every
        particle whose blue player slipped differently -- most of them, most
        steps.

        Given: The prior belief and one step of an episode.
        When: The reading is conditioned on.
        Then: Every particle carries the reported blue positions.

        Test type: integration
        """
        np.random.seed(0)
        belief = create_environment_belief(env, n_particles=64)
        state = env.initial_state_dist().sample()[0]
        action = 0

        next_state = env.sample_next_state(state, action)
        observation = np.asarray(env.sample_observation(next_state, action), dtype=np.float64)
        belief = belief.update(action, observation, env)

        reported = observation[: 2 * env.n_blue]
        positions = belief.particles[:, env.layout.blue_pos : env.layout.blue_pos + 2 * env.n_blue]
        np.testing.assert_array_equal(positions, np.tile(reported, (64, 1)))

    def test_no_flag_candidate_is_resampled_away(self, env):
        """Test that every candidate keeps its particles through an episode.

        Purpose: The flag candidate is static -- nothing in the transition ever
        moves a particle from one candidate to another -- so a candidate
        resampled to zero particles can never come back, however strongly later
        readings favour it.

        Given: The prior belief, which covers every candidate.
        When: Ten steps of an episode are conditioned on.
        Then: Every candidate still holds the particles it started with.

        Test type: integration
        """
        np.random.seed(0)
        belief = create_environment_belief(env, n_particles=200)
        before = np.bincount(
            belief.particles[:, env.layout.flag_cell].astype(int),
            minlength=len(env.red_flag_candidates),
        )
        state = env.initial_state_dist().sample()[0]

        for _ in range(10):
            action = int(np.random.randint(0, len(env.get_actions())))
            next_state = env.sample_next_state(state, action)
            belief = belief.update(action, env.sample_observation(next_state, action), env)
            state = next_state

        after = np.bincount(
            belief.particles[:, env.layout.flag_cell].astype(int),
            minlength=len(env.red_flag_candidates),
        )
        np.testing.assert_array_equal(after, before)

    def test_belief_finds_the_flag_more_often_than_not(self, env):
        """Test that the belief actually learns which candidate holds the flag.

        Purpose: The tests above check that nothing is broken; this one checks
        the belief is worth having. The range likelihood is sharp and averaged
        over a finite set of red hypotheses, so it converges in most episodes
        and over-commits in a few -- the mean is what is meaningful, not any
        single run.

        Given: Six 15-step episodes with 400 particles.
        When: Every reading is conditioned on.
        Then: The mean weight on the true candidate beats the 0.25 prior by a
            wide margin.

        Test type: integration
        """
        masses = []
        for seed in range(6):
            np.random.seed(seed)
            belief = create_environment_belief(env, n_particles=400)
            state = env.initial_state_dist().sample()[0]
            truth = int(state[env.layout.flag_cell])
            for _ in range(15):
                action = int(np.random.randint(0, len(env.get_actions())))
                next_state = env.sample_next_state(state, action)
                belief = belief.update(action, env.sample_observation(next_state, action), env)
                state = next_state
            on_truth = belief.particles[:, env.layout.flag_cell] == truth
            masses.append(float(belief.normalized_weights[on_truth].sum()))

        assert np.mean(masses) > 0.5, f"mean weight on the true candidate was {np.mean(masses)}"
