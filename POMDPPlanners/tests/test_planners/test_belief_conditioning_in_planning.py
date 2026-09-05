# SPDX-License-Identifier: MIT

"""Belief conditioning as the planners use it, with hand-computed posteriors.

Every belief-space planner in this repository advances its belief with
``Belief.update(action, observation, pomdp)`` and then samples from the result,
so the posterior weights are part of each planner's arithmetic even though no
planner computes them itself. The planner suites never check them: they run
against Tiger or Light-Dark, where the posterior cannot be worked out by eye.

These tests use a two-state example whose prior and likelihoods are chosen so
the posterior is a number a reader can verify. They also cover the degenerate
cases the skill's boundary list names — a likelihood that kills one particle,
equal likelihoods that change nothing, and a belief with a single particle —
and check that a planner's search does not write through to the belief object
its caller still holds.
"""

# pylint: disable=protected-access

import math
import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.tests.test_planners.planner_fixtures import (
    MISMATCH_LOG_PROB,
    NEXT,
    ROOT,
    ChainEnv,
    two_state_belief,
)


np.random.seed(42)
random.seed(42)

TOL = 1e-9


class _LikelihoodEnv(ChainEnv):
    """Chain whose observation likelihood is an explicit per-state table.

    The chain's own likelihood is a match/no-match indicator, which cannot
    produce a posterior other than a point mass. This subclass replaces it with
    a table, so ``P(o | s)`` is whatever the test says it is and the posterior
    is a short calculation.

    Transitions are the identity here, not the chain's successor map, so the
    prior carries into the update unchanged and the posterior is purely the
    likelihood's doing.
    """

    #: ``observation -> {state: probability}``.
    LIKELIHOODS = {
        "o1": {ROOT: 0.8, NEXT: 0.2},
        "o2": {ROOT: 0.5, NEXT: 0.5},
        "o3": {ROOT: 1.0, NEXT: 0.0},
    }

    def sample_next_state(self, state, action, n_samples: int = 1):
        self.transition_calls.append((state, action))
        return state if n_samples == 1 else [state] * n_samples

    def sample_observation(self, next_state, action, n_samples: int = 1):
        # The chain's own sampler returns the state, which is not a key in
        # LIKELIHOODS. Emitting a fixed observation keeps this environment's
        # only source of randomness out of the belief tests, which is the point
        # of the fixture: the posterior is then a function of the table alone.
        del next_state, action
        return "o1" if n_samples == 1 else ["o1"] * n_samples

    def sample_next_state_batch(self, states, action):
        del action
        return list(states)

    def observation_log_probability_per_state(self, next_states, action, observation):
        del action
        table = self.LIKELIHOODS[observation]
        return np.array(
            [
                math.log(table[state]) if table[state] > 0.0 else MISMATCH_LOG_PROB
                for state in next_states
            ],
            dtype=np.float64,
        )

    def observation_log_probability(self, next_state, action, observations):
        del action
        return np.array(
            [
                math.log(self.LIKELIHOODS[obs][next_state])
                if self.LIKELIHOODS[obs][next_state] > 0.0
                else MISMATCH_LOG_PROB
                for obs in observations
            ],
            dtype=np.float64,
        )


def _prior(first_weight=0.5):
    return two_state_belief(ROOT, NEXT, first_weight=first_weight)


def _posterior_weights(belief):
    return np.asarray(belief.normalized_weights, dtype=np.float64)


def test_equal_prior_and_likelihoods_0_8_and_0_2_give_those_posterior_weights():
    """The textbook case: an equal prior leaves the posterior equal to the likelihood.

    Purpose: The canonical hand-computable Bayes step. With prior 0.5 each and
        likelihoods 0.8 and 0.2, the unnormalised posterior is 0.4 and 0.1, and
        normalising gives 0.8 and 0.2.

    Given: A two-particle belief on ``root`` and ``next`` at 0.5 each, and an
        observation whose likelihood is 0.8 for ``root`` and 0.2 for ``next``.
    When: ``update`` runs with resampling off, so the weights are exact.
    Then: The normalised posterior weights are 0.8 and 0.2.

    Test type: unit
    """
    env = _LikelihoodEnv(discount_factor=0.5)
    posterior = _prior().update(action="a", observation="o1", pomdp=env)

    np.testing.assert_allclose(_posterior_weights(posterior), [0.8, 0.2], atol=TOL)


def test_an_uneven_prior_is_combined_with_the_likelihood():
    """The prior is not discarded: posterior is proportional to prior times likelihood.

    Purpose: A test using only an equal prior cannot tell "posterior equals
        likelihood" from "posterior equals prior times likelihood". This one
        can: with prior 0.25 and 0.75 and likelihoods 0.8 and 0.2, the
        unnormalised posterior is 0.2 and 0.15, which normalises to 4/7 and
        3/7 — a very different answer from the likelihood alone.

    Test type: unit
    """
    env = _LikelihoodEnv(discount_factor=0.5)
    posterior = _prior(first_weight=0.25).update(action="a", observation="o1", pomdp=env)

    np.testing.assert_allclose(_posterior_weights(posterior), [4 / 7, 3 / 7], atol=TOL)


def test_a_uniform_likelihood_leaves_the_prior_unchanged():
    """An uninformative observation must not move the belief.

    Purpose: A likelihood that is the same for every state carries no
        information; any change in the posterior would be a bug in the
        normalisation rather than a Bayes update.

    Test type: unit
    """
    env = _LikelihoodEnv(discount_factor=0.5)
    posterior = _prior(first_weight=0.25).update(action="a", observation="o2", pomdp=env)

    np.testing.assert_allclose(_posterior_weights(posterior), [0.25, 0.75], atol=TOL)


def test_a_zero_likelihood_particle_is_numerically_eliminated():
    """A particle the observation rules out keeps effectively no weight.

    Purpose: The degenerate-weight case. A zero likelihood is represented as a
        large negative log weight rather than negative infinity — the belief
        rejects non-finite log weights outright — so what has to be checked is
        that the surviving particle takes essentially all the mass and that no
        NaN appears.

    Given: An equal prior and an observation whose likelihood for ``next`` is 0.
    When: ``update`` runs.
    Then: ``root`` holds essentially all the weight, ``next`` essentially none,
        the weights still sum to 1, and every entry is finite.

    Test type: unit
    """
    env = _LikelihoodEnv(discount_factor=0.5)
    posterior = _prior().update(action="a", observation="o3", pomdp=env)

    weights = _posterior_weights(posterior)
    assert np.all(np.isfinite(weights)), f"posterior weights are not finite: {weights}"
    assert weights.sum() == pytest.approx(1.0, abs=TOL)
    assert weights[0] > 1.0 - 1e-12, (
        f"the surviving particle holds only {weights[0]} of the mass after the alternative was "
        "ruled out"
    )
    assert weights[1] < 1e-12


def test_a_single_particle_belief_stays_a_point_mass():
    """A one-particle belief normalises to weight 1 whatever its log weight.

    Purpose: Single-particle beliefs are everywhere in these tests and in
        POMCP's own tree nodes, so their normalisation is load-bearing. The
        scale of the log weight must not matter: -40 and +5 both normalise to
        1, and a normalisation that forgot to subtract the maximum would
        overflow or underflow on one of them.

    ``0.0`` is deliberately not in this list; see the test below.

    Test type: unit
    """
    for log_weight in (-1.0, 5.0, -40.0, 700.0):
        belief = WeightedParticleBelief(particles=[ROOT], log_weights=np.array([log_weight]))
        np.testing.assert_allclose(_posterior_weights(belief), [1.0], atol=TOL)
        assert belief.sample() == ROOT


def test_a_lone_zero_log_weight_is_rejected_by_the_all_zero_guard():
    """A single particle with log weight exactly 0 is refused at construction.

    Purpose: This is a boundary worth writing down rather than working around.
        ``WeightedParticleBelief`` guards against an all-zero log-weight vector
        because such a vector carries no information about the relative weights
        and normally means a caller forgot to fill it in. A one-particle belief
        with log weight 0 is caught by the same guard, even though it would
        normalise perfectly well to 1.

        The guard is therefore stricter than it strictly needs to be for the
        single-particle case. That is the shipped contract and it is what the
        planners are written against — every fixture in this suite uses a
        non-zero log weight — so it is pinned here rather than changed. A
        caller hitting it gets a clear message, not a silent wrong answer.

    Test type: unit
    """
    with pytest.raises(ValueError, match="nonzero"):
        WeightedParticleBelief(particles=[ROOT], log_weights=np.array([0.0]))


def test_an_all_zero_log_weight_vector_is_rejected():
    """The documented response to a degenerate weight vector is an error.

    Purpose: The skill asks what happens on zero total weight. Here the answer
        is an explicit rejection at construction rather than a silent uniform
        fallback, and pinning it stops a later change from making the failure
        silent.

    Test type: unit
    """
    with pytest.raises((ValueError, AssertionError)):
        WeightedParticleBelief(particles=[ROOT, NEXT], log_weights=np.array([0.0, 0.0]))


def test_a_non_finite_log_weight_is_rejected():
    """An infinite or NaN log weight fails at construction.

    Test type: unit
    """
    for bad in (np.inf, -np.inf, np.nan):
        with pytest.raises((ValueError, AssertionError)):
            WeightedParticleBelief(particles=[ROOT, NEXT], log_weights=np.array([bad, -1.0]))


def test_update_returns_a_new_belief_and_leaves_the_prior_alone():
    """Conditioning is not an in-place edit of the caller's belief.

    Purpose: The belief-space planners call ``update`` on a node's belief many
        times per search. If it mutated its receiver, every one of those calls
        would compound onto the same object and the tree's beliefs would all
        collapse to the last observation seen.

    Given: A prior whose weights are recorded before the update.
    When: ``update`` runs.
    Then: A different object comes back, and the prior's particles and log
        weights are unchanged.

    Test type: unit
    """
    env = _LikelihoodEnv(discount_factor=0.5)
    prior = _prior()
    before_particles = list(prior.particles)
    before_log_weights = np.array(prior.log_weights, copy=True)

    posterior = prior.update(action="a", observation="o1", pomdp=env)

    assert posterior is not prior
    assert prior.particles == before_particles
    np.testing.assert_array_equal(np.asarray(prior.log_weights), before_log_weights)


def test_two_successive_observations_compose_multiplicatively():
    """Conditioning twice equals conditioning once on the product of likelihoods.

    Purpose: A planner descending two levels conditions twice. The composed
        posterior must be proportional to prior times both likelihoods:
        starting equal, two ``o1`` observations give unnormalised
        ``0.5*0.8*0.8 = 0.32`` and ``0.5*0.2*0.2 = 0.02``, which normalise to
        16/17 and 1/17.

    Test type: unit
    """
    env = _LikelihoodEnv(discount_factor=0.5)
    once = _prior().update(action="a", observation="o1", pomdp=env)
    twice = once.update(action="a", observation="o1", pomdp=env)

    np.testing.assert_allclose(_posterior_weights(twice), [16 / 17, 1 / 17], atol=TOL)


def test_a_planner_search_does_not_write_through_to_the_callers_belief():
    """A whole PFT-DPW search leaves the belief object it was handed unchanged.

    Purpose: The composition of everything above. The planner conditions the
        belief repeatedly while building its tree; the caller's object must
        come back byte-identical, because the episode runner keeps using it.

    Given: A PFT-DPW search over the likelihood environment.
    When: ``action()`` runs a fixed number of simulations.
    Then: The caller's belief has the same particles and the same log weights.

    Test type: unit
    """
    from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
    from POMDPPlanners.tests.test_planners.planner_fixtures import FixedActionSampler

    np.random.seed(808)
    random.seed(808)
    env = _LikelihoodEnv(discount_factor=0.5)
    planner = PFT_DPW(
        environment=env,
        discount_factor=0.5,
        depth=2,
        name="pft_belief_immutability",
        action_sampler=FixedActionSampler(["a", "b"]),
        k_a=2.0,
        alpha_a=0.5,
        k_o=2.0,
        alpha_o=0.5,
        exploration_constant=1.0,
        n_simulations=6,
    )
    belief = _prior()
    before_particles = list(belief.particles)
    before_log_weights = np.array(belief.log_weights, copy=True)

    planner.action(belief)

    assert (
        belief.particles == before_particles
    ), f"the caller's particles became {belief.particles}; planning must not write through"
    np.testing.assert_array_equal(np.asarray(belief.log_weights), before_log_weights)
