# SPDX-License-Identifier: MIT

"""Tests for PacMan vectorized particle belief updater."""

# pylint: disable=protected-access

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.pacman_pomdp import (
    PacManPOMDP,
    _native,
)  # pylint: disable=no-name-in-module
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_beliefs.pacman_vectorized_updater import (
    PacManVectorizedUpdater,
)
from POMDPPlanners.tests.test_core.test_belief.belief_equivalence_utils import (
    assert_sample_distributions_match,
    assert_update_particles_match,
    assert_update_weights_match,
)
from POMDPPlanners.tests.test_core.test_belief.vectorized_updater_test_utils import (
    assert_batch_obs_log_likelihood_matches_loop,
    assert_batch_transition_matches_loop,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import pacman_pinned_kwargs


def _make_aligned_beliefs(env, updater, n_particles=50):
    """Create baseline + vectorized beliefs with identical initial particles."""
    np.random.seed(42)
    states = env.initial_state_dist().sample(n_samples=n_particles)
    particles_array = np.stack(states)
    particles_list = [particles_array[i].copy() for i in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)

    base = WeightedParticleBelief(
        particles=particles_list,
        log_weights=log_weights.copy(),
        resampling=False,
    )
    vec = VectorizedWeightedParticleBelief(
        particles=particles_array.copy(),
        log_weights=log_weights.copy(),
        updater=updater,
        resampling=False,
    )
    return base, vec


def _make_particle_to_array(env):  # pylint: disable=unused-argument
    """Particles are ndarrays; identity converter kept for helper-signature parity."""

    def convert(particle):
        return particle

    return convert


@pytest.fixture()
def simple_env():
    return PacManPOMDP(
        discount_factor=0.95,
        **pacman_pinned_kwargs(
            maze_size=(5, 5),
            walls={(2, 2)},
            initial_pellets=[(1, 1), (3, 3)],
            initial_pacman_pos=(0, 0),
            num_ghosts=1,
            initial_ghost_positions=[(4, 4)],
            ghost_aggressiveness=2.0,
            ghost_coordination="independent",
        ),
    )


@pytest.fixture()
def updater(simple_env):
    return PacManVectorizedUpdater.from_environment(simple_env)


@pytest.fixture()
def sample_particles(simple_env):
    np.random.seed(42)
    states = simple_env.initial_state_dist().sample(n_samples=10)
    return np.stack(states)


class TestFromEnvironmentConstruction:
    def test_creates_without_error(self, simple_env):
        """Test that from_environment constructs an updater successfully.

        Purpose: Validates updater creation from environment.

        Given: A configured PacManPOMDP environment.
        When: PacManVectorizedUpdater.from_environment is called.
        Then: An updater instance is created without errors.

        Test type: unit
        """
        updater = PacManVectorizedUpdater.from_environment(simple_env)
        assert updater.maze_size == (5, 5)
        assert updater.num_ghosts == 1
        assert updater.num_pellets == 2
        assert updater.state_dim == simple_env._state_dim


class TestBatchTransition:
    def test_shape_preserved(self, updater, sample_particles):
        """Test that batch_transition preserves particle array shape.

        Purpose: Validates output shape matches input shape.

        Given: Particles array of shape (10, d).
        When: batch_transition is called with action 2 (South).
        Then: Output shape is (10, d).

        Test type: unit
        """
        result = updater.batch_transition(sample_particles, 2)
        assert result.shape == sample_particles.shape

    def test_terminal_unchanged(self, updater, simple_env):
        """Test that terminal particles remain unchanged after transition.

        Purpose: Validates terminal particle preservation.

        Given: A particle array where some particles are terminal.
        When: batch_transition is called.
        Then: Terminal particles are identical to input.

        Test type: unit
        """
        arr = simple_env.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((4, 4),),
            pellets=((1, 1), (3, 3)),
            score=0.0,
            terminal=True,
        ).reshape(1, -1)
        result = updater.batch_transition(arr, 2)
        np.testing.assert_array_equal(result, arr)

    def test_pacman_moves_correctly(self, updater, simple_env):
        """Test that PacMan moves to correct position on valid move.

        Purpose: Validates pacman movement direction.

        Given: PacMan at (0,0) in a 5x5 maze.
        When: Action South (2) is applied.
        Then: PacMan moves to (1,0).

        Test type: unit
        """
        np.random.seed(42)
        arr = simple_env.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((4, 4),),
            pellets=((1, 1), (3, 3)),
            score=0.0,
            terminal=False,
        ).reshape(1, -1)
        result = updater.batch_transition(arr, 2)  # South
        assert result[0, updater._idx_pac_row] == 1
        assert result[0, updater._idx_pac_col] == 0

    def test_pellet_collection(self, updater, simple_env):
        """Test that pellet bitmask flips and score increments on collection.

        Purpose: Validates pellet collection mechanics.

        Given: PacMan at (1,0) with pellet at (1,1).
        When: Action East (1) moves PacMan onto the pellet.
        Then: Pellet bitmask flips to 0 and score increments.

        Test type: unit
        """
        np.random.seed(42)
        arr = simple_env.make_state(
            pacman_pos=(1, 0),
            ghost_positions=((4, 4),),
            pellets=((1, 1), (3, 3)),
            score=0.0,
            terminal=False,
        ).reshape(1, -1)
        result = updater.batch_transition(arr, 1)  # East -> (1,1)
        # Pellet at index 0 ((1,1)) should be collected
        assert result[0, updater._idx_pellets_start] == 0.0
        assert result[0, updater._idx_score] > arr[0, updater._idx_score]

    def test_ghost_collision_sets_terminal(self, updater, simple_env):
        """Test that ghost collision sets terminal flag.

        Purpose: Validates collision detection in vectorized transition.

        Given: PacMan at (3,4) with ghost at (4,4).
        When: Action South (2) moves PacMan to ghost position.
        Then: Terminal flag is set to 1.

        Test type: unit
        """
        np.random.seed(123)
        arr = simple_env.make_state(
            pacman_pos=(3, 4),
            ghost_positions=((4, 4),),
            pellets=((1, 1), (3, 3)),
            score=0.0,
            terminal=False,
        ).reshape(1, -1)
        # Move south to (4,4) where ghost might be
        result = updater.batch_transition(arr, 2)
        # PacMan should be at (4,4)
        assert result[0, updater._idx_pac_row] == 4
        assert result[0, updater._idx_pac_col] == 4
        # If ghost stayed at (4,4), should be terminal
        ghost_r = result[0, updater._idx_ghosts_start]
        ghost_c = result[0, updater._idx_ghosts_start + 1]
        if ghost_r == 4 and ghost_c == 4:
            assert result[0, updater._idx_terminal] == 1.0


class TestBatchObservationLogLikelihood:
    def test_shape(self, updater, sample_particles):
        """Test output shape of batch_observation_log_likelihood.

        Purpose: Validates output shape is (N,).

        Given: 10 particles and an observation.
        When: batch_observation_log_likelihood is called.
        Then: Output shape is (10,).

        Test type: unit
        """
        obs = np.array([4.0, 4.0])
        ll = updater.batch_observation_log_likelihood(sample_particles, 0, obs)
        assert ll.shape == (10,)

    def test_terminal_observation(self, updater, simple_env):
        """Test terminal particle handling in log-likelihood.

        Purpose: Validates terminal particles get correct log-likelihood.

        Given: A terminal particle and a terminal observation (-1, -1).
        When: batch_observation_log_likelihood is called.
        Then: Terminal particle gets 0.0 log-likelihood.

        Test type: unit
        """
        arr = simple_env.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((4, 4),),
            pellets=(),
            score=20.0,
            terminal=True,
        ).reshape(1, -1)
        obs = np.array([-1.0, -1.0])
        ll = updater.batch_observation_log_likelihood(arr, 0, obs)
        assert ll[0] == 0.0

    def test_closer_ghost_higher_precision(self, updater, simple_env):
        """Test that closer ghosts yield higher log-likelihood for accurate obs.

        Purpose: Validates distance-dependent observation noise.

        Given: Two particles — one with ghost close to PacMan, one far.
        When: Observation matches the true ghost position exactly.
        Then: Close-ghost particle has higher log-likelihood.

        Test type: unit
        """
        close_arr = simple_env.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((0, 1),),
            pellets=((1, 1), (3, 3)),
            score=0.0,
            terminal=False,
        ).reshape(1, -1)
        far_arr = simple_env.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((4, 4),),
            pellets=((1, 1), (3, 3)),
            score=0.0,
            terminal=False,
        ).reshape(1, -1)
        particles = np.vstack([close_arr, far_arr])

        # Observation matches close ghost exactly
        obs = np.array([0.0, 1.0])
        ll = updater.batch_observation_log_likelihood(particles, 0, obs)
        assert ll[0] > ll[1]


# ---------------------------------------------------------------------------
# Equivalence test: vectorized vs per-particle loop
# ---------------------------------------------------------------------------


class TestEquivalenceWithPerParticleLoop:
    def test_batch_transition_matches_per_particle_loop(self, simple_env, updater):
        """Test vectorized batch_transition matches per-particle state_transition_model.

        Purpose: Verifies that batch_transition produces the same results as
                 calling the environment's state_transition_model per particle
                 with the same random seed.

        Given: A set of particles sampled from the initial distribution.
        When: batch_transition is called, and the same transitions are computed
              per-particle using the environment's state_transition_model.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(123)
        states = simple_env.initial_state_dist().sample(n_samples=20)
        particles = np.stack(states)

        def per_particle_fn(particle, action):
            return simple_env.sample_next_state(state=particle, action=action)

        for action in range(4):
            assert_batch_transition_matches_loop(
                updater=updater,
                particles=particles,
                action=action,
                per_particle_transition_fn=per_particle_fn,
                seed=999,
                seed_fn=_native.set_seed,
                err_msg=f"Mismatch for action {action}",
            )

    def test_batch_obs_log_likelihood_matches_per_particle_loop(self, simple_env, updater):
        """Test vectorized log-likelihood matches per-particle observation_model.probability.

        Purpose: Verifies that batch_observation_log_likelihood matches the
                 per-particle observation probability from the environment.

        Given: A set of particles and a ghost observation.
        When: batch_observation_log_likelihood is called, and per-particle
              log(observation_model.probability) is computed.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(42)
        states = simple_env.initial_state_dist().sample(n_samples=20)
        particles = np.stack(states)
        obs_array = np.array([3.0, 3.0])
        obs_tuple = ((3, 3),)

        def per_particle_ll_fn(particle, action, _observation):
            log_probs = simple_env.observation_log_probability(
                next_state=particle, action=action, observations=[obs_tuple]
            )
            return float(log_probs[0])

        assert_batch_obs_log_likelihood_matches_loop(
            updater=updater,
            particles=particles,
            action=0,
            observation=obs_array,
            per_particle_ll_fn=per_particle_ll_fn,
        )


class TestBeliefEquivalenceWithBaseline:
    def test_update_particles_match(self, simple_env, updater):
        """Test vectorized belief update produces identical next particles.

        Purpose: Validates that VectorizedWeightedParticleBelief.update and
            WeightedParticleBelief.update agree on next-state particles. The
            baseline belief stores particles as ndarray rows on initialization
            but WeightedParticleBelief.update stores returned PacManState
            instances after the step, so particle comparison is routed
            through env.state_to_array via particle_to_array.

        Given: 50 aligned particles in both beliefs.
        When: Both beliefs are updated with action=0 and a fixed observation.
        Then: Next-particle arrays agree within floating-point tolerance.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(simple_env, updater)
        obs = ((3, 3),)
        assert_update_particles_match(
            base=base,
            vec=vec,
            action=0,
            observation=obs,
            pomdp=simple_env,
            seed=999,
            seed_fn=_native.set_seed,
            particle_to_array=_make_particle_to_array(simple_env),
        )

    def test_update_weights_match(self, simple_env, updater):
        """Test vectorized and baseline beliefs produce identical normalized weights.

        Purpose: Validates observation-reweighting consistency on PacMan's
            ghost-position observations.

        Given: 50 aligned particles.
        When: Both beliefs are updated with action=0 under a shared seed.
        Then: Normalized weights agree within 1e-6 L-infinity.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(simple_env, updater)
        obs = ((3, 3),)
        assert_update_weights_match(
            base=base,
            vec=vec,
            action=0,
            observation=obs,
            pomdp=simple_env,
            atol=1e-6,
            seed=999,
            seed_fn=_native.set_seed,
        )

    def test_sample_distributions_match_post_update(self, simple_env, updater):
        """Test sample() on both beliefs draws unbiased from normalized_weights.

        Purpose: Validates sample() unbiasedness and cross-belief agreement.
            PacMan particles duplicate heavily (small grid), so aggregation by
            particle identity inside the helper handles discrete duplicates.

        Given: 50 aligned particles; one update step seeded identically.
        When: 20,000 samples are drawn from each belief.
        Then: Empirical histograms agree and each matches its normalized_weights.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(simple_env, updater)
        obs = ((3, 3),)
        _native.set_seed(999)
        np.random.seed(999)
        vec = vec.update(action=0, observation=obs, pomdp=simple_env)
        _native.set_seed(999)
        np.random.seed(999)
        base = base.update(action=0, observation=obs, pomdp=simple_env)

        assert_sample_distributions_match(
            base=base,
            vec=vec,
            n_samples=20_000,
            tol=0.03,
            atol_weights=0.03,
            seed=400,
            particle_to_array=_make_particle_to_array(simple_env),
        )


class TestConfigId:
    def test_deterministic(self, simple_env):
        """Test that config_id is deterministic.

        Purpose: Validates reproducible config_id generation.

        Given: Two updaters from the same environment.
        When: config_id is accessed.
        Then: Both return the same string.

        Test type: unit
        """
        u1 = PacManVectorizedUpdater.from_environment(simple_env)
        u2 = PacManVectorizedUpdater.from_environment(simple_env)
        assert u1.config_id == u2.config_id

    def test_slip_probability_changes_config_id(self, simple_env):
        """Test that differing slip_probability yields a distinct config_id.

        Purpose: Validates slip_probability participates in the cache key so
        beliefs built with different slip are not conflated.

        Given: Two updaters from the same layout, slip 0.0 vs 0.9.
        When: config_id is accessed on each.
        Then: The two config_id strings differ.

        Test type: unit
        """
        u0 = PacManVectorizedUpdater.from_environment(simple_env)
        simple_env.slip_probability = 0.9
        u9 = PacManVectorizedUpdater.from_environment(simple_env)
        assert u0.config_id != u9.config_id


def _tiled_particles(env, pacman_pos, n_particles):
    """Replicate a single non-terminal start state into ``n_particles`` rows."""
    state = env.make_state(
        pacman_pos=pacman_pos,
        ghost_positions=((4, 4),),
        pellets=((1, 1), (3, 3)),
        score=0.0,
        terminal=False,
    )
    return np.tile(np.asarray(state, dtype=float), (n_particles, 1))


class TestSlipAwareTransition:
    """Regression coverage: the belief transition must model motion slip.

    The stochastic PacMan variant injects motion *slip* (with probability
    ``slip_probability`` an intended move is replaced by a uniformly random
    action). If the vectorized updater ignores slip, every particle takes the
    intended move and the planner optimizes against a deterministic model that
    does not contain the environment's stochasticity.
    """

    def test_batch_transition_depends_on_slip_probability(self, simple_env):
        """Test that batch_transition outcomes depend on slip_probability.

        Purpose: Validates that the belief transition is not deterministic when
        slip_probability > 0 (the core bug: slip invisible to the planner).

        Given: Two updaters over an identical layout, slip 0.0 vs 0.9, and a
            batch of identical non-terminal particles at an interior cell.
        When: batch_transition is applied with a single move action to both.
        Then: The slip-free updater sends every particle to one PacMan cell,
            while the slippery updater scatters PacMan across several cells, and
            the two destination sets differ.

        Test type: unit
        """
        u0 = PacManVectorizedUpdater.from_environment(simple_env)
        simple_env.slip_probability = 0.9
        u9 = PacManVectorizedUpdater.from_environment(simple_env)

        parts = _tiled_particles(simple_env, (2, 1), 3000)
        cols = [u0._idx_pac_row, u0._idx_pac_col]

        np.random.seed(0)
        _native.set_seed(0)
        out0 = u0.batch_transition(parts.copy(), np.asarray(2))  # South
        np.random.seed(0)
        _native.set_seed(0)
        out9 = u9.batch_transition(parts.copy(), np.asarray(2))  # South

        dests0 = np.unique(out0[:, cols], axis=0)
        dests9 = np.unique(out9[:, cols], axis=0)
        assert len(dests0) == 1
        assert len(dests9) > 1
        assert not np.array_equal(np.sort(dests0, axis=0), np.sort(dests9, axis=0))

    def test_batch_transition_slip_mixture_fraction(self, simple_env):
        """Test that the slipped transition follows the slip mixture law.

        Purpose: Validates the acceptance criterion that the fraction of
        particles taking the intended move is ~ 1 - slip * (1 - 1/n_actions).

        Given: PacMan at interior cell (1,1) where all five actions lead to
            distinct in-bounds cells, and a slip-0.9 updater (5 actions).
        When: batch_transition is applied with the East action to a large
            batch of identical particles.
        Then: The fraction of particles reaching the intended East cell (1,2)
            is close to 1 - 0.9 * (4/5) = 0.28.

        Test type: unit
        """
        simple_env.slip_probability = 0.9
        updater = PacManVectorizedUpdater.from_environment(simple_env)
        n_particles = 8000
        parts = _tiled_particles(simple_env, (1, 1), n_particles)

        np.random.seed(1)
        _native.set_seed(1)
        out = updater.batch_transition(parts, np.asarray(1))  # East -> intended (1, 2)

        at_intended = np.mean(
            (out[:, updater._idx_pac_row] == 1) & (out[:, updater._idx_pac_col] == 2)
        )
        expected = 1.0 - 0.9 * (1.0 - 1.0 / 5.0)
        assert abs(at_intended - expected) < 0.03
