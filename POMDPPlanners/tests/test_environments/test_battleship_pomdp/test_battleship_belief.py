# SPDX-License-Identifier: MIT

"""Tests for the exact Battleship belief.

Battleship's observations are deterministic, which is what makes the belief the
delicate part of this environment. A generic particle filter degrades silently
here — every particle drawn from the prior eventually contradicts the readings,
the eps floor keeps the weights finite, and resampling spreads impossible boards
back across the belief with nothing raising. These tests pin the three
properties that failure mode violates: the belief stays exactly the prior
restricted to consistent layouts, every particle is always a legal fleet, and it
never runs out of support.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.environments.battleship_pomdp import (
    BattleshipBelief,
    BattleshipPOMDP,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import battleship_pinned_kwargs


@pytest.fixture(name="env")
def _env() -> BattleshipPOMDP:
    return BattleshipPOMDP(discount_factor=0.99, **battleship_pinned_kwargs())


def _probe_until_done(env, belief, true_state, max_steps=25):
    """Run a greedy-marginal prober, returning the trace of beliefs and states."""
    trace = []
    for _ in range(max_steps):
        unprobed = ~env.probed(true_state)
        scores = np.where(unprobed, belief.occupancy_marginal(env), -1.0)
        action = int(np.argmax(scores))
        next_state, observation, _ = env.sample_next_step(true_state, action)
        belief = belief.update(action=action, observation=observation, pomdp=env)
        true_state = next_state
        trace.append((action, observation, belief, true_state))
        if env.is_terminal(true_state):
            break
    return trace


class TestPrior:
    """The starting belief is the uniform prior over legal layouts."""

    def test_prior_covers_every_legal_layout(self, env: BattleshipPOMDP) -> None:
        """Purpose: a prior missing layouts rules out boards the world can produce.

        Given: a fresh belief
        When: its consistent set is read
        Then: it is every row of the layout table

        Test type: unit
        """
        belief = BattleshipBelief.from_environment(env, n_particles=16)
        assert belief.consistent_indices(env).size == env.layouts.num_layouts

    def test_prior_marginal_is_symmetric(self, env: BattleshipPOMDP) -> None:
        """Purpose: a uniform prior on a square board has the board's symmetries.

        An asymmetric prior would mean the enumeration favours one orientation
        or one edge, which no correct enumeration of a square board can.

        Given: a fresh belief's per-cell marginal
        When: it is compared with its own transpose and flips
        Then: they agree

        Test type: unit
        """
        marginal = BattleshipBelief.from_environment(env, n_particles=8).occupancy_marginal(env)
        grid = marginal.reshape(env.board_size, env.board_size)
        assert np.allclose(grid, grid.T)
        assert np.allclose(grid, np.flipud(grid))
        assert np.allclose(grid, np.fliplr(grid))

    def test_prior_marginal_sums_to_the_fleet_size(self, env: BattleshipPOMDP) -> None:
        """Purpose: the marginals must integrate to the number of ship cells.

        Given: the prior marginal
        When: summed over the board
        Then: it equals the fleet's cell count

        Test type: unit
        """
        marginal = BattleshipBelief.from_environment(env, n_particles=8).occupancy_marginal(env)
        assert marginal.sum() == pytest.approx(float(env.num_ship_cells))

    def test_particles_are_legal_layouts_with_nothing_probed(self, env: BattleshipPOMDP) -> None:
        """Purpose: every particle must be a board the world could have dealt.

        Given: a fresh belief
        When: its particles are inspected
        Then: each occupies the fleet's cell count and has an empty probe half

        Test type: unit
        """
        belief = BattleshipBelief.from_environment(env, n_particles=64)
        particles = np.asarray(belief.particles)
        assert particles.shape == (64, 2 * env.num_cells)
        assert np.all(particles[:, : env.num_cells].sum(axis=1) == env.num_ship_cells)
        assert not np.any(particles[:, env.num_cells :])


class TestExactness:
    """The posterior is the prior restricted to the consistent layouts."""

    def test_update_matches_a_brute_force_filter(self, env: BattleshipPOMDP) -> None:
        """Purpose: the incremental filter must equal filtering from scratch.

        Given: a board and a fixed probe sequence
        When: the belief is updated step by step
        Then: at every step its consistent set equals the set recomputed from
              the whole revealed history in one pass

        Test type: unit
        """
        np.random.seed(3)
        belief = BattleshipBelief.from_environment(env, n_particles=32)
        true_state = belief.sample()

        revealed_cells: list = []
        revealed_values: list = []
        for action in (0, 6, 12, 18, 24, 2, 7, 13):
            _, observation, _ = env.sample_next_step(true_state, action)
            belief = belief.update(action=action, observation=observation, pomdp=env)
            true_state = env.sample_next_state(true_state, action)

            revealed_cells.append(action)
            revealed_values.append(int(observation))
            expected = env.layouts.consistent_indices(
                np.array(revealed_cells), np.array(revealed_values, dtype=np.uint8)
            )
            assert np.array_equal(belief.consistent_indices(env), expected)

    def test_the_true_layout_is_never_ruled_out(self, env: BattleshipPOMDP) -> None:
        """Purpose: a filter that drops the truth is worse than no filter.

        Given: a board drawn from the prior and probed to completion
        When: the belief is updated after every probe
        Then: the true occupancy remains among the consistent layouts throughout

        Test type: unit
        """
        np.random.seed(11)
        belief = BattleshipBelief.from_environment(env, n_particles=32)
        true_state = belief.sample()
        truth = env.occupancy(true_state).astype(np.uint8)

        for _, _, belief, _ in _probe_until_done(env, belief, true_state):
            surviving = env.layouts.masks[belief.consistent_indices(env)]
            assert np.any(np.all(surviving == truth, axis=1))

    def test_probed_cells_are_resolved_in_every_particle(self, env: BattleshipPOMDP) -> None:
        """Purpose: a particle disagreeing with a reading is an impossible board.

        This is exactly what a generic particle filter loses here, so it is
        checked on every step rather than only at the end.

        Given: a probed board
        When: the belief is updated after every probe
        Then: every particle agrees with the truth on every probed cell

        Test type: unit
        """
        np.random.seed(5)
        belief = BattleshipBelief.from_environment(env, n_particles=48)
        true_state = belief.sample()

        for _, _, belief, state in _probe_until_done(env, belief, true_state):
            particles = np.asarray(belief.particles)
            probed = env.probed(state)
            truth = env.occupancy(state)
            assert np.array_equal(particles[:, env.num_cells :] > 0.5, np.tile(probed, (len(particles), 1)))
            assert np.all((particles[:, : env.num_cells] > 0.5)[:, probed] == truth[probed])

    def test_every_particle_stays_a_legal_fleet(self, env: BattleshipPOMDP) -> None:
        """Purpose: the belief must never invent a board the fleet cannot make.

        Given: a probed board
        When: the belief is updated after every probe
        Then: every particle's occupancy is a row of the legal-layout table

        Test type: unit
        """
        np.random.seed(17)
        belief = BattleshipBelief.from_environment(env, n_particles=24)
        true_state = belief.sample()
        legal = {row.tobytes() for row in env.layouts.masks}

        for _, _, belief, _ in _probe_until_done(env, belief, true_state):
            occupancy = (np.asarray(belief.particles)[:, : env.num_cells] > 0.5).astype(np.uint8)
            assert all(row.tobytes() in legal for row in occupancy)

    def test_belief_never_depletes(self, env: BattleshipPOMDP) -> None:
        """Purpose: depletion is the failure this belief exists to prevent.

        Given: twenty boards probed to completion
        When: the consistent set is measured after every probe
        Then: it is never empty, and the particle count never shrinks

        Test type: integration
        """
        for seed in range(20):
            np.random.seed(seed)
            belief = BattleshipBelief.from_environment(env, n_particles=32)
            true_state = belief.sample()
            for _, _, belief, _ in _probe_until_done(env, belief, true_state):
                assert belief.consistent_indices(env).size > 0
                assert len(belief.particles) == 32

    def test_a_fully_probed_board_pins_the_layout(self, env: BattleshipPOMDP) -> None:
        """Purpose: with everything revealed the belief must be a point mass.

        Given: a board with every cell probed
        When: the marginal is read
        Then: it is exactly the true occupancy

        Test type: unit
        """
        np.random.seed(2)
        belief = BattleshipBelief.from_environment(env, n_particles=16)
        true_state = belief.sample()
        for action in range(env.num_cells):
            _, observation, _ = env.sample_next_step(true_state, action)
            belief = belief.update(action=action, observation=observation, pomdp=env)
            true_state = env.sample_next_state(true_state, action)
        assert np.array_equal(
            belief.occupancy_marginal(env), env.occupancy(true_state).astype(np.float64)
        )

    def test_repeat_probe_does_not_reapply_the_evidence(self, env: BattleshipPOMDP) -> None:
        """Purpose: re-probing must be a no-op on the belief, not a second update.

        Given: a belief that has already conditioned on cell 0
        When: cell 0 is probed again with the same reading
        Then: the consistent set is unchanged

        Test type: unit
        """
        np.random.seed(4)
        belief = BattleshipBelief.from_environment(env, n_particles=16)
        true_state = belief.sample()
        _, observation, _ = env.sample_next_step(true_state, 0)

        once = belief.update(action=0, observation=observation, pomdp=env)
        twice = once.update(action=0, observation=observation, pomdp=env)
        assert np.array_equal(once.consistent_indices(env), twice.consistent_indices(env))


class TestRobustness:
    """Behaviour at the edges, and the interfaces other code relies on."""

    def test_it_is_a_weighted_particle_belief(self, env: BattleshipPOMDP) -> None:
        """Purpose: downstream code dispatches on belief type.

        The expected-reward helper and the terminal-belief check both branch on
        WeightedParticleBelief, so anything that is not one is silently routed
        down a different path.

        Given: a Battleship belief
        When: its type is checked
        Then: it is a WeightedParticleBelief with uniform normalized weights

        Test type: unit
        """
        belief = BattleshipBelief.from_environment(env, n_particles=8)
        assert isinstance(belief, WeightedParticleBelief)
        assert np.allclose(belief.normalized_weights, 1.0 / 8.0)

    def test_a_single_particle_belief_is_accepted(self, env: BattleshipPOMDP) -> None:
        """Purpose: log(1/1) is zero, which the base class rejects outright.

        Given: a one-particle belief
        When: it is built and updated
        Then: neither raises

        Test type: unit
        """
        np.random.seed(1)
        belief = BattleshipBelief.from_environment(env, n_particles=1)
        true_state = belief.sample()
        _, observation, _ = env.sample_next_step(true_state, 0)
        assert len(belief.update(action=0, observation=observation, pomdp=env).particles) == 1

    def test_consistent_set_is_recovered_from_the_particles(self, env: BattleshipPOMDP) -> None:
        """Purpose: the belief must survive pickling and rebuilding.

        The consistent set is a cache, so a belief rebuilt from its particles
        alone (as a worker process receives it) must recompute the same set
        rather than silently starting from the prior.

        Given: a belief after several probes
        When: a new belief is built from its particles alone
        Then: the two agree on the consistent set

        Test type: unit
        """
        np.random.seed(9)
        belief = BattleshipBelief.from_environment(env, n_particles=32)
        true_state = belief.sample()
        for action in (0, 6, 12, 18):
            _, observation, _ = env.sample_next_step(true_state, action)
            belief = belief.update(action=action, observation=observation, pomdp=env)
            true_state = env.sample_next_state(true_state, action)

        rebuilt = BattleshipBelief(
            particles=np.asarray(belief.particles), log_weights=belief.log_weights.copy()
        )
        assert np.array_equal(rebuilt.consistent_indices(env), belief.consistent_indices(env))

    def test_config_id_differs_from_a_generic_particle_belief(self, env) -> None:
        """Purpose: the episode result cache keys on the initial belief's config_id.

        The inherited identity is built from particles and weights with no class
        in it, so this belief and a plain WeightedParticleBelief holding the same
        particles would collide and share cached episodes. Comparing the exact
        belief against a generic filter is the first experiment anyone will run
        on this environment, and it is precisely the one that collision would
        answer from the wrong run.

        Given: a Battleship belief and a plain particle belief over its particles
        When: their config ids are compared
        Then: they differ, and the Battleship one is stable across rebuilds

        Test type: unit
        """
        np.random.seed(13)
        belief = BattleshipBelief.from_environment(env, n_particles=8)
        particles = np.asarray(belief.particles)
        generic = WeightedParticleBelief(
            particles=list(particles), log_weights=belief.log_weights.copy()
        )
        rebuilt = BattleshipBelief(particles=particles.copy(), log_weights=belief.log_weights.copy())

        assert belief.config_id != generic.config_id
        assert belief.config_id == rebuilt.config_id

    def test_consistent_set_is_recomputed_for_a_different_geometry(self, env) -> None:
        """Purpose: cached row indices mean nothing against another layout table.

        The cache holds row numbers into one environment's table. Queried against
        a table built for a different board, an in-range row would be read as a
        legal fleet rather than raising — a wrong belief with no symptom.

        Given: a belief whose consistent set was computed for the default board
        When: it is queried against a same-shaped board with touching forbidden
        Then: the set is recomputed against that board's table, and every index
              is in range for it

        Test type: unit
        """
        np.random.seed(23)
        strict_env = BattleshipPOMDP(discount_factor=0.99, allow_adjacent_ships=False)
        belief = BattleshipBelief.from_environment(env, n_particles=8)

        permissive_indices = belief.consistent_indices(env)
        strict_indices = belief.consistent_indices(strict_env)

        assert strict_indices.size == strict_env.layouts.num_layouts
        assert strict_indices.size != permissive_indices.size
        assert int(strict_indices.max()) < strict_env.layouts.num_layouts

    def test_a_contradictory_observation_raises(self, env: BattleshipPOMDP) -> None:
        """Purpose: an empty belief must fail loudly, not quietly reinitialise.

        Given: a belief that has already seen cell 0's true value
        When: it is fed the opposite reading for cell 0
        Then: a ValueError says the belief and the world disagree

        Test type: unit
        """
        np.random.seed(6)
        belief = BattleshipBelief.from_environment(env, n_particles=16)
        true_state = belief.sample()
        _, observation, _ = env.sample_next_step(true_state, 0)
        belief = belief.update(action=0, observation=observation, pomdp=env)

        with pytest.raises(ValueError, match="disagree"):
            belief.update(action=0, observation=1 - int(observation), pomdp=env)

    def test_from_environment_rejects_a_foreign_environment(self) -> None:
        """Purpose: a wrong environment would silently produce a meaningless belief.

        Given: an environment with no layout table
        When: a Battleship belief is requested for it
        Then: a TypeError says so

        Test type: unit
        """
        with pytest.raises(TypeError, match="BattleshipPOMDP"):
            BattleshipBelief.from_environment(object(), n_particles=4)  # type: ignore[arg-type]
