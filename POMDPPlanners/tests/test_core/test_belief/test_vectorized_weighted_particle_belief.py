"""Tests for VectorizedWeightedParticleBelief and VectorizedParticleBeliefUpdater.

This module tests the vectorized weighted particle belief using a lightweight
mock updater that requires no environment dependencies.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
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
        """Test that non-finite log_weights raise ValueError.

        Purpose: Validates finiteness checking on log_weights.

        Given: log_weights contain -inf.
        When: Construction is attempted.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="log_weights must be finite"):
            VectorizedWeightedParticleBelief(
                particles=np.zeros((2, 2)),
                log_weights=np.array([0.0, -np.inf]),
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
