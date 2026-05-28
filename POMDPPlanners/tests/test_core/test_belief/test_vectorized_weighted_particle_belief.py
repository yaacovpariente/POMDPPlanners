# SPDX-License-Identifier: MIT

"""Tests for VectorizedWeightedParticleBelief and VectorizedParticleBeliefUpdater.

This module tests the vectorized weighted particle belief using a lightweight
mock updater that requires no environment dependencies.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.tests.test_utils.metric_invariants_utils import (
    verify_belief_invariants,
)


# ---------------------------------------------------------------------------
# Mock updater (identity transition, uniform log-likelihood)
# ---------------------------------------------------------------------------


class _MockUpdater(VectorizedParticleBeliefUpdater):
    """Deterministic updater: transition adds action, uniform log-likelihood."""

    def batch_transition(self, particles, action):
        return particles + action

    def batch_observation_log_likelihood(self, next_particles, action, observation):
        return np.zeros(len(next_particles))

    @property
    def config_id(self):
        return "mock"


class _NonUniformMockUpdater(VectorizedParticleBeliefUpdater):
    """Updater that assigns higher log-likelihood to particles near the observation."""

    def batch_transition(self, particles, action):
        return particles + action

    def batch_observation_log_likelihood(self, next_particles, action, observation):
        distances = np.linalg.norm(next_particles - observation, axis=1)
        return -distances

    @property
    def config_id(self):
        return "nonuniform_mock"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def uniform_belief():
    np.random.seed(0)
    n, d = 100, 2
    particles = np.random.randn(n, d)
    log_weights = np.full(n, -np.log(n))
    return VectorizedWeightedParticleBelief(
        particles=particles,
        log_weights=log_weights,
        updater=_MockUpdater(),
        resampling=False,
    )


# ---------------------------------------------------------------------------
# Construction validation tests
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_valid_construction(self, uniform_belief):
        """Test that a valid belief can be constructed.

        Purpose: Validates basic construction of VectorizedWeightedParticleBelief.

        Given: Valid particles array, uniform log weights, and a mock updater.
        When: The belief is constructed.
        Then: All attributes are set correctly and no errors are raised.

        Test type: unit
        """
        assert uniform_belief.n_particles == 100
        assert uniform_belief.dim == 2
        assert uniform_belief.particles.shape == (100, 2)
        assert uniform_belief.log_weights.shape == (100,)
        np.testing.assert_allclose(uniform_belief.normalized_weights.sum(), 1.0, atol=1e-12)
        verify_belief_invariants(uniform_belief, expected_n_particles=100)

    def test_particles_must_be_ndarray(self):
        """Test that non-ndarray particles raise TypeError.

        Purpose: Validates type checking on particles argument.

        Given: A list (not ndarray) is passed as particles.
        When: Construction is attempted.
        Then: TypeError is raised.

        Test type: unit
        """
        with pytest.raises(TypeError, match="particles must be a numpy.ndarray"):
            VectorizedWeightedParticleBelief(
                particles=[[1, 2], [3, 4]],  # type: ignore[arg-type]
                log_weights=np.zeros(2),
                updater=_MockUpdater(),
            )

    def test_log_weights_must_be_ndarray(self):
        """Test that non-ndarray log_weights raise TypeError.

        Purpose: Validates type checking on log_weights argument.

        Given: A list (not ndarray) is passed as log_weights.
        When: Construction is attempted.
        Then: TypeError is raised.

        Test type: unit
        """
        with pytest.raises(TypeError, match="log_weights must be a numpy.ndarray"):
            VectorizedWeightedParticleBelief(
                particles=np.zeros((2, 2)),
                log_weights=[0.0, 0.0],  # type: ignore[arg-type]
                updater=_MockUpdater(),
            )

    def test_particles_must_be_2d(self):
        """Test that 1-D particles raise ValueError.

        Purpose: Validates dimensionality checking on particles.

        Given: A 1-D array is passed as particles.
        When: Construction is attempted.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="particles must be 2-D"):
            VectorizedWeightedParticleBelief(
                particles=np.zeros(5),
                log_weights=np.zeros(5),
                updater=_MockUpdater(),
            )

    def test_shape_mismatch_raises(self):
        """Test that mismatched particle/weight lengths raise ValueError.

        Purpose: Validates consistency checking between particles and log_weights.

        Given: particles has 3 rows but log_weights has 2 entries.
        When: Construction is attempted.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="particles has 3 rows"):
            VectorizedWeightedParticleBelief(
                particles=np.zeros((3, 2)),
                log_weights=np.zeros(2),
                updater=_MockUpdater(),
            )

    def test_non_finite_log_weights_raise(self):
        """Test that NaN log_weights raise ValueError.

        Purpose: Validates that NaN values in log_weights are rejected.
        Note: -inf is allowed (represents zero-weight particles in log-space).

        Given: log_weights contain NaN.
        When: Construction is attempted.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="log_weights must not contain NaN or"):
            VectorizedWeightedParticleBelief(
                particles=np.zeros((2, 2)),
                log_weights=np.array([0.0, np.nan]),
                updater=_MockUpdater(),
            )


# ---------------------------------------------------------------------------
# Sampling tests
# ---------------------------------------------------------------------------


class TestSampling:
    def test_sample_returns_correct_shape(self, uniform_belief):
        """Test that sample returns a vector of the correct dimensionality.

        Purpose: Validates that sample() returns a 1-D array with d elements.

        Given: A belief with 2-D particles.
        When: A single sample is drawn.
        Then: The result has shape (2,).

        Test type: unit
        """
        state = uniform_belief.sample()
        assert state.shape == (2,)

    def test_sample_is_from_particle_set(self, uniform_belief):
        """Test that sampled state is one of the existing particles.

        Purpose: Validates that sample() picks from the stored particles.

        Given: A belief with known particles.
        When: Multiple samples are drawn.
        Then: Each sample matches a row in the particles array.

        Test type: unit
        """
        np.random.seed(1)
        for _ in range(50):
            state = uniform_belief.sample()
            diffs = np.linalg.norm(uniform_belief.particles - state, axis=1)
            assert np.min(diffs) < 1e-12


# ---------------------------------------------------------------------------
# Update tests
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_returns_new_belief(self, uniform_belief):
        """Test that update produces a new VectorizedWeightedParticleBelief.

        Purpose: Validates the return type and immutability of update.

        Given: A valid belief.
        When: update() is called.
        Then: A new VectorizedWeightedParticleBelief is returned.

        Test type: unit
        """
        new_belief = uniform_belief.update(
            action=np.array([1.0, 0.0]),
            observation=np.zeros(2),
            pomdp=None,
        )
        assert isinstance(new_belief, VectorizedWeightedParticleBelief)
        assert new_belief is not uniform_belief
        verify_belief_invariants(new_belief, expected_n_particles=uniform_belief.n_particles)

    def test_update_transitions_particles(self, uniform_belief):
        """Test that update transitions particles via the updater.

        Purpose: Validates that particles are shifted by the action (mock updater adds action).

        Given: A belief and a mock updater that adds action to particles.
        When: update() is called with action [1, 0].
        Then: New particles equal old particles + [1, 0].

        Test type: unit
        """
        action = np.array([1.0, 0.0])
        new_belief = uniform_belief.update(action=action, observation=np.zeros(2), pomdp=None)
        expected = uniform_belief.particles + action
        np.testing.assert_allclose(new_belief.particles, expected)

    def test_update_preserves_particle_count(self, uniform_belief):
        """Test that update preserves the number of particles.

        Purpose: Validates that N is unchanged after update.

        Given: A belief with 100 particles.
        When: update() is called.
        Then: The resulting belief still has 100 particles.

        Test type: unit
        """
        new_belief = uniform_belief.update(action=np.zeros(2), observation=np.zeros(2), pomdp=None)
        assert new_belief.n_particles == uniform_belief.n_particles
        verify_belief_invariants(new_belief, expected_n_particles=uniform_belief.n_particles)


# ---------------------------------------------------------------------------
# Resampling tests
# ---------------------------------------------------------------------------


class TestResampling:
    def test_resampling_triggers_on_low_ess(self):
        """Test that resampling triggers when ESS drops below threshold.

        Purpose: Validates ESS-based resampling with highly skewed weights.

        Given: A belief with one dominant weight and resampling enabled.
        When: update() is called with the non-uniform mock updater.
        Then: After update, weights are uniform (resampled).

        Test type: unit
        """
        np.random.seed(42)
        n, d = 50, 2
        particles = np.random.randn(n, d) * 5
        log_w = np.full(n, -np.log(n))

        updater = _NonUniformMockUpdater()
        belief = VectorizedWeightedParticleBelief(
            particles=particles,
            log_weights=log_w,
            updater=updater,
            resampling=True,
            ess_factor=0.5,
        )
        new_belief = belief.update(
            action=np.zeros(2),
            observation=np.array([0.0, 0.0]),
            pomdp=None,
        )
        # After resampling, weights should be uniform
        np.testing.assert_allclose(
            new_belief.normalized_weights,
            np.ones(n) / n,
            atol=1e-12,
        )
        verify_belief_invariants(new_belief, expected_n_particles=n)

    def test_no_resampling_when_disabled(self, uniform_belief):
        """Test that resampling does not occur when disabled.

        Purpose: Validates that resampling=False prevents resampling.

        Given: A belief with resampling=False and a non-uniform updater.
        When: update() is called with non-uniform log-likelihoods.
        Then: Weights are not reset to uniform.

        Test type: unit
        """
        np.random.seed(42)
        particles = np.random.randn(50, 2) * 5
        log_w = np.full(50, -np.log(50))
        updater = _NonUniformMockUpdater()
        belief = VectorizedWeightedParticleBelief(
            particles=particles,
            log_weights=log_w,
            updater=updater,
            resampling=False,
        )
        new_belief = belief.update(
            action=np.zeros(2),
            observation=np.array([0.0, 0.0]),
            pomdp=None,
        )
        # Weights should NOT be uniform since resampling is disabled
        assert not np.allclose(
            new_belief.normalized_weights,
            np.ones(50) / 50,
        )

    def test_resampling_collapses_to_dominant_state(self):
        """Test that resampling discards near-zero-weight particles in favor of the dominant state.

        Purpose: Validates particle survival bias under degenerate weights — when one
        particle holds essentially all the probability mass and all others are
        negligible, the post-resample particle set should be dominated by copies of
        the high-weight particle's state (this is the expected outcome of multinomial
        resampling on a degenerate weight distribution).

        Given: A belief with 100 distinct particles where particle 0 has log-weight 0.0
            and all others have log-weight -100.0 (relative weight ~e^-100, negligible),
            paired with a uniform-likelihood mock updater and a zero action so the
            transition is the identity.
        When: update() is called with resampling enabled (ess_factor=0.5).
        Then: At least 95% of post-resample particles are exact copies of the dominant
            particle's state, and the post-resample weights are uniform.

        Test type: unit
        """
        np.random.seed(42)
        n, d = 100, 2
        # 100 distinct particles so we can identify which survives resampling.
        particles = np.arange(n * d, dtype=float).reshape(n, d)
        dominant_state = particles[0].copy()

        # Degenerate log-weights: one near 0, the rest at -100 (~e^-100, negligible).
        log_weights = np.full(n, -100.0)
        log_weights[0] = 0.0

        belief = VectorizedWeightedParticleBelief(
            particles=particles,
            log_weights=log_weights,
            updater=_MockUpdater(),
            resampling=True,
            ess_factor=0.5,
        )

        new_belief = belief.update(
            action=np.zeros(d),
            observation=np.array([0.0, 0.0]),
            pomdp=None,
        )

        matches = np.all(new_belief.particles == dominant_state, axis=1)
        assert matches.sum() >= int(0.95 * n)
        np.testing.assert_allclose(
            new_belief.normalized_weights,
            np.ones(n) / n,
            atol=1e-12,
        )
        verify_belief_invariants(new_belief, expected_n_particles=n)


# ---------------------------------------------------------------------------
# Config ID tests
# ---------------------------------------------------------------------------


class TestConfigId:
    def test_config_id_deterministic(self, uniform_belief):
        """Test that config_id is deterministic for identical beliefs.

        Purpose: Validates config_id reproducibility.

        Given: Two identically-constructed beliefs.
        When: config_id is computed for both.
        Then: The IDs are equal.

        Test type: unit
        """
        belief_copy = VectorizedWeightedParticleBelief(
            particles=uniform_belief.particles.copy(),
            log_weights=uniform_belief.log_weights.copy(),
            updater=uniform_belief.updater,
            resampling=uniform_belief.resampling,
            ess_factor=uniform_belief.ess_factor,
        )
        assert uniform_belief.config_id == belief_copy.config_id

    def test_config_id_changes_with_particles(self, uniform_belief):
        """Test that config_id changes when particles differ.

        Purpose: Validates that different particles produce different IDs.

        Given: Two beliefs with different particles.
        When: config_id is computed for both.
        Then: The IDs differ.

        Test type: unit
        """
        other = VectorizedWeightedParticleBelief(
            particles=uniform_belief.particles + 1.0,
            log_weights=uniform_belief.log_weights.copy(),
            updater=uniform_belief.updater,
        )
        assert uniform_belief.config_id != other.config_id


# ---------------------------------------------------------------------------
# Unique-support distribution tests
# ---------------------------------------------------------------------------


def _pair_set(values, probs):
    return {(tuple(np.asarray(v).tolist()), round(float(p), 12)) for v, p in zip(values, probs)}


class TestToUniqueSupportDistribution:
    def test_to_unique_support_distribution_collapses_duplicates(self):
        """Test that duplicate particles are collapsed and weights are aggregated.

        Purpose: Validates the full contract of to_unique_support_distribution:
            empty-belief handling, duplicate collapsing, defensive L1
            normalization, and equivalence with WeightedParticleBelief.

        Given: Four sub-cases — (1) an empty belief, (2) a belief with duplicate
            particles, (3) a belief with unnormalized weights, and (4) the same
            particle/weight data wrapped in a WeightedParticleBelief.
        When: to_unique_support_distribution() is called on each.
        Then: (1) returns an empty DiscreteDistribution, (2) collapses
            duplicates and sums their weights, (3) probabilities sum to exactly
            1.0, and (4) produces the same unique-support distribution (as a
            set of (particle, probability) pairs) as WeightedParticleBelief.

        Test type: unit
        """
        # Sub-case 1: empty belief
        empty_belief = VectorizedWeightedParticleBelief(
            particles=np.zeros((0, 2)),
            log_weights=np.zeros(0),
            updater=_MockUpdater(),
        )
        empty_dist = empty_belief.to_unique_support_distribution()
        assert isinstance(empty_dist, DiscreteDistribution)
        assert len(empty_dist.values) == 0
        assert empty_dist.probs.shape == (0,)

        # Sub-case 2: duplicates collapse and their weights sum.
        # Particles 0 and 2 are identical; particles 1 and 3 are identical.
        particles = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        # Uniform weights -> two duplicate-groups of 2 collapse to 0.4 each,
        # singleton stays at 0.2.
        log_weights = np.full(5, -np.log(5))
        belief = VectorizedWeightedParticleBelief(
            particles=particles,
            log_weights=log_weights,
            updater=_MockUpdater(),
        )
        dist = belief.to_unique_support_distribution()
        assert isinstance(dist, DiscreteDistribution)
        assert len(dist.values) == 3
        for value in dist.values:
            assert isinstance(value, np.ndarray)
            assert value.shape == (2,)
        expected = {
            ((1.0, 2.0), 0.4),
            ((3.0, 4.0), 0.4),
            ((5.0, 6.0), 0.2),
        }
        assert _pair_set(dist.values, dist.probs) == expected

        # Sub-case 3: unnormalized weights -> probs still sum to exactly 1.0.
        unnormalized_log_weights = np.array(
            [np.log(2.0), np.log(3.0), np.log(2.0), np.log(3.0), np.log(5.0)]
        )
        belief_unnorm = VectorizedWeightedParticleBelief(
            particles=particles,
            log_weights=unnormalized_log_weights,
            updater=_MockUpdater(),
        )
        dist_unnorm = belief_unnorm.to_unique_support_distribution()
        assert float(dist_unnorm.probs.sum()) == pytest.approx(1.0, abs=1e-12)

        # Sub-case 4: equivalence with WeightedParticleBelief on the same data.
        ref_belief = WeightedParticleBelief(
            particles=[row.copy() for row in particles],
            log_weights=log_weights.copy(),
        )
        ref_dist = ref_belief.to_unique_support_distribution()
        assert _pair_set(dist.values, dist.probs) == _pair_set(ref_dist.values, ref_dist.probs)
