# SPDX-License-Identifier: MIT

"""Tests for the Battleship vectorized particle belief.

The batched updater is a copy of the environment's transition and sensor, so
the first tests here are parity tests against them. The rest are about the
property the belief exists for: Battleship's sensor is deterministic, so a
plain weight-and-resample filter fills with layouts the probes have already
ruled out, and it does that silently. These check that this one does not.
"""

from typing import cast

import numpy as np
import pytest

from POMDPPlanners.environments.battleship_pomdp import (
    HIT,
    MISS,
    BattleshipBelief,
    BattleshipPOMDP,
    BattleshipVectorizedUpdater,
    BattleshipVectorizedWeightedParticleBelief,
)
from POMDPPlanners.utils.belief_factory import BeliefType, create_environment_belief


@pytest.fixture
def env():
    return BattleshipPOMDP(discount_factor=0.99)


@pytest.fixture
def updater(env):
    return BattleshipVectorizedUpdater.from_environment(env)


def _probe_sequence(env, cells):
    """Walk a true board through a list of probes, yielding (cell, reading)."""
    np.random.seed(0)
    state = env.initial_state_dist().sample()[0]
    for cell in cells:
        next_state = env.sample_next_state(state, cell)
        yield cell, env.sample_observation(next_state, cell)
        state = next_state


class TestBattleshipVectorizedUpdater:
    def test_batch_transition_matches_environment(self, env, updater):
        """Test that the batched probe equals the environment's own.

        Purpose: The updater duplicates the transition; this is the test that
        the duplicate has not drifted.

        Given: Particles drawn from the prior.
        When: Several cells are probed through both paths.
        Then: The successor arrays are identical.

        Test type: unit
        """
        np.random.seed(0)
        states = np.stack(env.initial_state_dist().sample(n_samples=50))

        for cell in (0, 7, 13, 24):
            np.testing.assert_array_equal(
                updater.batch_transition(states, cell),
                env.sample_next_state_batch(states, cell),
                err_msg=f"cell {cell}",
            )

    @pytest.mark.parametrize("reading", [HIT, MISS])
    def test_observation_log_likelihood_matches_environment(self, env, updater, reading):
        """Test that the batched likelihood equals the environment's.

        Purpose: The sensor cannot lie here, so the likelihood is a hard filter
        and a disagreement would rule out the wrong half of the boards.

        Given: Probed particles and both possible readings.
        When: Each reading is scored through both paths.
        Then: The log-likelihoods agree, infinities included.

        Test type: unit
        """
        np.random.seed(0)
        states = np.stack(env.initial_state_dist().sample(n_samples=40))
        cell = 11
        probed = updater.batch_transition(states, cell)

        batched = updater.batch_observation_log_likelihood(probed, cell, reading)
        scalar = np.array(
            [env.observation_log_probability(row, cell, [reading])[0] for row in probed]
        )
        np.testing.assert_array_equal(batched, scalar)

    def test_config_id_separates_fleets(self, env, updater):
        """Test that the identity covers the board geometry.

        Purpose: ``config_id`` is a cache key, and a key blind to the fleet
        would serve one board's cached beliefs for another.

        Given: Two updaters on one environment and one on a different fleet.
        When: Their identities are compared.
        Then: The pair agrees and the different fleet does not.

        Test type: unit
        """
        same = BattleshipVectorizedUpdater.from_environment(env)
        other = BattleshipVectorizedUpdater.from_environment(
            BattleshipPOMDP(discount_factor=0.99, ship_lengths=(3, 2))
        )

        assert updater.config_id == same.config_id
        assert updater.config_id != other.config_id


class TestBattleshipVectorizedBelief:
    def test_default_belief_is_the_vectorized_one(self, env):
        """Test that the factory hands back the batched belief by default.

        Purpose: Registration is what makes the updater reachable; without it
        the environment silently falls back to the generic filter.

        Given: A Battleship environment.
        When: The top-level factory is asked for its belief.
        Then: The vectorized belief comes back.

        Test type: unit
        """
        belief = create_environment_belief(env, n_particles=32)
        assert isinstance(belief, BattleshipVectorizedWeightedParticleBelief)

    def test_particle_type_still_available(self, env):
        """Test that the generic particle belief is still selectable.

        Purpose: The exact belief is the default, not the only option;
        comparing against a generic filter has to stay possible.

        Given: A Battleship environment.
        When: ``BeliefType.PARTICLE`` is requested.
        Then: Something other than the vectorized belief comes back.

        Test type: unit
        """
        belief = create_environment_belief(env, belief_type=BeliefType.PARTICLE, n_particles=32)
        assert not isinstance(belief, BattleshipVectorizedWeightedParticleBelief)

    def test_every_particle_stays_a_legal_fleet(self, env):
        """Test that ten probes leave every particle a legal placement.

        Purpose: This is what the redraw buys. A resampling filter would be
        holding boards the probes ruled out, and the weights would look healthy
        while it did.

        Given: The prior belief and a run of probes against a real board.
        When: Each reading is conditioned on.
        Then: Every particle's occupancy half is a row of the layout table.

        Test type: integration
        """
        belief = cast(
            BattleshipVectorizedWeightedParticleBelief,
            create_environment_belief(env, n_particles=64),
        )
        for cell, reading in _probe_sequence(env, [0, 5, 12, 20, 3, 14, 21, 7, 18, 24]):
            belief = belief.update(cell, reading, env)

        masks = env.layouts.masks.astype(np.float64)
        for particle in belief.particles:
            board = particle[: env.num_cells]
            assert np.any(np.all(masks == board, axis=1)), "particle is not a legal fleet"

    def test_marginal_matches_the_exact_belief(self, env):
        """Test that the batched belief and the scalar exact one agree exactly.

        Purpose: The two are meant to carry the same posterior. If they
        disagree, one of them is wrong and only a comparison would say so.

        Given: The same probe sequence fed to both beliefs.
        When: Their per-cell occupancy marginals are compared.
        Then: They are identical.

        Test type: integration
        """
        vectorized = cast(
            BattleshipVectorizedWeightedParticleBelief,
            create_environment_belief(env, n_particles=64),
        )
        exact = BattleshipBelief.from_environment(env, n_particles=64)
        for cell, reading in _probe_sequence(env, [0, 5, 12, 20, 3, 14, 21]):
            vectorized = vectorized.update(cell, reading, env)
            exact = exact.update(cell, reading, env)

        np.testing.assert_array_equal(
            vectorized.occupancy_marginal(env), exact.occupancy_marginal(env)
        )

    def test_contradictory_reading_raises(self, env):
        """Test that a reading no layout explains raises rather than resets.

        Purpose: A belief supported on nothing is the failure this class exists
        to prevent. Quietly reinitialising would hide the cause -- another
        board's observations, or a mismatched fleet.

        Given: A belief that has probed a cell and seen a miss.
        When: The same cell is probed again and reported as a hit.
        Then: A ValueError names the cell and the reading.

        Test type: unit
        """
        np.random.seed(0)
        belief = create_environment_belief(env, n_particles=16)
        cell = 0
        belief = belief.update(cell, MISS, env)

        with pytest.raises(ValueError, match="no legal Battleship layout"):
            belief.update(cell, HIT, env)

    def test_repeat_probe_is_a_no_op(self, env):
        """Test that probing a resolved cell again changes nothing.

        Purpose: The same evidence must not be applied twice, and a repeat
        probe is the easy way to do that by accident.

        Given: A belief that has probed one cell.
        When: The same cell is probed again with the same reading.
        Then: The consistent-layout set is unchanged.

        Test type: unit
        """
        np.random.seed(0)
        belief = cast(
            BattleshipVectorizedWeightedParticleBelief,
            create_environment_belief(env, n_particles=32),
        )
        belief = belief.update(0, MISS, env)
        before = belief.consistent_indices.copy()

        after = belief.update(0, MISS, env)

        np.testing.assert_array_equal(after.consistent_indices, before)
