# SPDX-License-Identifier: MIT

"""Tests for conditioning a particle filter on the episode still running.

What these pin is the helper's contract and, just as importantly, where it is
allowed to run. The conditioning is correct evidence at execution time and a
falsehood inside a planner's search tree, so the split between the two is
tested as deliberately as the arithmetic.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief.running_episode_conditioning import (
    RULED_OUT_LOG_MARGIN,
    condition_log_weights_on_a_running_episode,
)
from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)


class _StillUpdater(VectorizedParticleBeliefUpdater):
    """An updater that moves nothing and learns nothing.

    Both hot paths are inert, so any weight difference a test sees comes from
    the conditioning and from nothing else.
    """

    def __init__(self, ruled_out=None):
        self._ruled_out = ruled_out

    def batch_transition(self, particles, action):
        del action
        return particles

    def batch_observation_log_likelihood(self, next_particles, action, observation):
        del action, observation
        return np.zeros(len(next_particles))

    def ruled_out_by_a_running_episode(self, next_particles):
        del next_particles
        return self._ruled_out

    @property
    def config_id(self) -> str:
        return "still"


def _belief(updater, n_particles=4):
    return VectorizedWeightedParticleBelief(
        particles=np.zeros((n_particles, 2)),
        log_weights=np.log(np.ones(n_particles) / n_particles),
        updater=updater,
        resampling=False,
    )


def test_a_ruled_out_particle_is_floored_below_every_survivor():
    """The floor has to beat the worst particle the step still allows.

    Purpose: A fixed sentinel will not do. Environments that return raw
        log-likelihoods hand a contradicted particle a large negative number
        of its own, so a constant floor would let a ruled-out particle tie
        with -- and on a bad step outrank -- a particle the episode is still
        allowed to be in. Anchoring to the worst survivor is what keeps the
        ordering right at any likelihood scale.

    Given: Three particles, the middle one ruled out, with spread weights.
    When: The weights are conditioned.
    Then: The ruled-out particle sits below the worst survivor, and no
        survivor's weight is touched.

    Test type: unit
    """
    log_weights = np.array([-1.0, -2.0, -50.0])
    ruled_out = np.array([False, True, False])

    conditioned, acted = condition_log_weights_on_a_running_episode(log_weights, ruled_out)

    assert acted
    assert conditioned[1] == pytest.approx(-50.0 + RULED_OUT_LOG_MARGIN)
    assert conditioned[0] == -1.0 and conditioned[2] == -50.0


@pytest.mark.parametrize(
    "ruled_out",
    [np.array([False, False, False]), np.array([True, True, True])],
    ids=["none_ruled_out", "every_particle_ruled_out"],
)
def test_nothing_is_floored_when_the_mask_separates_no_particles(ruled_out):
    """A mask that hits everything or nothing carries no information.

    Purpose: Flooring every particle leaves a belief supported on nothing,
        which cannot be normalised or resampled, and flooring none is a
        pointless copy. Both are the caller's cue to repair the population
        itself, so the helper says it did not act rather than quietly
        returning something unusable.

    Given: A mask that is all-true or all-false.
    When: The weights are conditioned.
    Then: They come back untouched, with ``acted`` false.

    Test type: unit
    """
    log_weights = np.array([-1.0, -2.0, -3.0])

    conditioned, acted = condition_log_weights_on_a_running_episode(log_weights, ruled_out)

    assert not acted
    assert np.array_equal(conditioned, log_weights)


def test_ruled_out_particles_stay_below_survivors_that_are_all_impossible():
    """The all--inf population must not resurrect what was just ruled out.

    Purpose: When no particle explains the reading, every log-weight is
        ``-inf``, and the normaliser's documented fallback for that is a
        *uniform* distribution -- which would hand the resample the very
        particles this function ruled out. Measured on Discrete Light-Dark,
        that revived roughly 180 of 400 terminal particles at full weight on
        exactly the steps where the reading was impossible.

    Given: Every particle at ``-inf``, half of them ruled out.
    When: The weights are conditioned.
    Then: The survivors normalise to a positive share and the ruled-out ones
        to none.

    Test type: unit
    """
    log_weights = np.full(4, -np.inf)
    ruled_out = np.array([True, True, False, False])

    conditioned, acted = condition_log_weights_on_a_running_episode(log_weights, ruled_out)

    assert acted
    normalized = np.exp(conditioned - conditioned.max())
    normalized = normalized / normalized.sum()
    assert normalized[ruled_out].sum() == pytest.approx(0.0)
    assert normalized[~ruled_out].sum() == pytest.approx(1.0)


def test_an_updater_that_does_not_opt_in_keeps_its_weights():
    """Silence is the default, for every belief in the package.

    Purpose: Five environments needed this; the rest did not, and a blanket
        change to the shared update would alter every environment and every
        planner at once on the strength of those five. The base updater
        returns no mask, and a belief built on it must be bit-identical to
        what it was before the hook existed.

    Given: An updater that never overrides the hook, and a real step.
    When: The belief is updated with a state.
    Then: The log-weights are exactly the prior plus the likelihood.

    Test type: unit
    """

    class _Plain(_StillUpdater):
        ruled_out_by_a_running_episode = (
            VectorizedParticleBeliefUpdater.ruled_out_by_a_running_episode
        )

    belief = _belief(_Plain())

    updated = belief.update(
        action=np.zeros(2), observation=np.zeros(2), pomdp=None, state=np.zeros(2)
    )

    assert np.array_equal(updated.log_weights, belief.log_weights)


def test_a_planners_update_is_not_conditioned_but_the_drivers_is():
    """Only a step that really happened is evidence the episode continued.

    Purpose: This is the whole reason the conditioning keys on ``state``. A
        planner expanding its tree is asking what happens *if* it acts, and
        termination is one of the answers -- ``is_terminal_belief`` is how
        SparsePFT and ICVaR-PFT-DPW stop growing a branch, and PFT-DPW samples
        a particle for the same test. Stripping terminal particles in there
        would tell the search that hazards never end an episode and that a
        goal once reached keeps paying, which is the worst possible error on
        the hazard-terminal environments this exists for. The episode driver
        is the only caller that passes ``state``.

    Given: One belief and an updater that rules out its first two particles.
    When: It is updated once without a state and once with one.
    Then: Only the second is conditioned.

    Test type: unit
    """
    ruled_out = np.array([True, True, False, False])
    belief = _belief(_StillUpdater(ruled_out))

    in_tree = belief.update(action=np.zeros(2), observation=np.zeros(2), pomdp=None, state=None)
    at_execution = belief.update(
        action=np.zeros(2), observation=np.zeros(2), pomdp=None, state=np.zeros(2)
    )

    assert np.array_equal(in_tree.log_weights, belief.log_weights), (
        "a planner's belief update must be left alone: the search has to be able to "
        "reach a terminal belief"
    )
    assert at_execution.normalized_weights[ruled_out].sum() == pytest.approx(0.0)
