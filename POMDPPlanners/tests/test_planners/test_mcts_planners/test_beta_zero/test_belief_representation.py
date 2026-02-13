"""Tests for ParticleMeanStdRepresentation belief-to-feature conversion.

This module verifies that the ParticleMeanStdRepresentation class correctly
extracts mean and standard deviation feature vectors from various belief types
including WeightedParticleBelief, GaussianBelief, and fallback sampling.
"""

from typing import Any, Optional

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief, GaussianBelief, Belief
from POMDPPlanners.core.belief.gaussian_belief_updaters import LinearKalmanFilterUpdater
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.planners.mcts_planners.beta_zero.belief_representation import (
    ParticleMeanStdRepresentation,
)

np.random.seed(42)


class _DummyBelief(Belief):
    """Minimal concrete Belief subclass for fallback testing."""

    def __init__(self, values):
        self._values = values

    def update(self, action, observation, pomdp, state=None):
        return self

    def sample(self):
        return self._values[np.random.randint(len(self._values))]


def test_weighted_mean_std_computed_correctly():
    """Test weighted mean and std computation with known particles and weights.

    Purpose: Validates that ParticleMeanStdRepresentation computes the correct
        weighted mean and weighted standard deviation for a 1D particle set.

    Given: Particles [1, 3, 5] with normalized weights [0.2, 0.5, 0.3]
    When: The representation is called on a WeightedParticleBelief
    Then: The resulting feature vector matches the manually computed
        weighted mean and weighted standard deviation

    Test type: unit
    """
    particles = [1.0, 3.0, 5.0]
    weights = np.array([0.2, 0.5, 0.3])
    log_weights = np.log(weights)

    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)
    rep = ParticleMeanStdRepresentation(state_dim=1)
    features = rep(belief)

    # Manual computation
    w = weights / weights.sum()
    expected_mean = np.average(particles, weights=w)
    expected_var = np.average((np.array(particles) - expected_mean) ** 2, weights=w)
    expected_std = np.sqrt(expected_var)

    assert features.shape == (2,)
    np.testing.assert_allclose(features[0], expected_mean, atol=1e-5)
    np.testing.assert_allclose(features[1], expected_std, atol=1e-5)


def test_feature_dim_equals_2_times_state_dim():
    """Test that feature_dim property returns 2 * state_dim.

    Purpose: Validates that the feature dimensionality is correctly reported
        as twice the state dimensionality.

    Given: A ParticleMeanStdRepresentation with state_dim=3
    When: The feature_dim property is accessed
    Then: It returns 6

    Test type: unit
    """
    rep = ParticleMeanStdRepresentation(state_dim=3)
    assert rep.feature_dim == 6


def test_works_with_weighted_particle_belief():
    """Test feature extraction from a WeightedParticleBelief.

    Purpose: Validates that the representation correctly processes a
        WeightedParticleBelief and returns a feature vector of the right shape.

    Given: A WeightedParticleBelief with 5 two-dimensional particles
    When: The representation is called on the belief
    Then: The output has shape (4,) and dtype float32

    Test type: unit
    """
    particles = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [2.0, 3.0], [4.0, 5.0]]
    log_weights = np.log(np.array([0.1, 0.2, 0.3, 0.15, 0.25]))
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    rep = ParticleMeanStdRepresentation(state_dim=2)
    features = rep(belief)

    assert features.shape == (4,)
    assert features.dtype == np.float32


def test_works_with_gaussian_belief():
    """Test feature extraction from a GaussianBelief.

    Purpose: Validates that for a GaussianBelief the features match
        [mean, sqrt(diag(covariance))].

    Given: A GaussianBelief with mean=[1.0, 2.0] and covariance=diag(4.0, 9.0)
    When: The representation is called on the belief
    Then: The features are [1.0, 2.0, 2.0, 3.0]

    Test type: unit
    """
    updater = LinearKalmanFilterUpdater(
        A=np.eye(2),
        B=np.zeros((2, 1)),
        H=np.eye(2),
        Q=0.1 * np.eye(2),
        R=0.5 * np.eye(2),
    )
    mean = np.array([1.0, 2.0])
    covariance = np.diag([4.0, 9.0])
    belief = GaussianBelief(mean=mean, covariance=covariance, updater=updater)

    rep = ParticleMeanStdRepresentation(state_dim=2)
    features = rep(belief)

    expected = np.array([1.0, 2.0, 2.0, 3.0], dtype=np.float32)
    assert features.shape == (4,)
    np.testing.assert_allclose(features, expected, atol=1e-5)


def test_falls_back_to_samples_for_unknown_belief():
    """Test fallback to sampling for an unknown Belief subclass.

    Purpose: Validates that when a Belief subclass is not a recognized type
        (WeightedParticleBelief, WeightedParticleBeliefStateUpdate, or
        GaussianBelief), the representation falls back to sampling and
        still produces a correctly shaped feature vector.

    Given: A custom Belief subclass that returns 1D numeric samples
    When: The representation is called on this unknown belief type
    Then: The output has the correct shape (2 * state_dim,) and finite values

    Test type: unit
    """
    values = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    belief = _DummyBelief(values)

    rep = ParticleMeanStdRepresentation(state_dim=2)
    features = rep(belief)

    assert features.shape == (4,)
    assert np.all(np.isfinite(features))
    assert features.dtype == np.float32


def test_multidimensional_states():
    """Test feature extraction with 2D state particles.

    Purpose: Validates correct handling of multidimensional state particles,
        ensuring the output has shape (2 * state_dim,).

    Given: A WeightedParticleBelief with particles [[1,2],[3,4]] in 2D state space
    When: The representation is called on the belief
    Then: The output has shape (4,)

    Test type: unit
    """
    particles = [[1.0, 2.0], [3.0, 4.0]]
    log_weights = np.log(np.array([0.5, 0.5]))
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    rep = ParticleMeanStdRepresentation(state_dim=2)
    features = rep(belief)

    assert features.shape == (4,)

    # With equal weights: mean = [2.0, 3.0], std = [1.0, 1.0]
    expected_mean = np.array([2.0, 3.0])
    expected_std = np.array([1.0, 1.0])
    np.testing.assert_allclose(features[:2], expected_mean, atol=1e-5)
    np.testing.assert_allclose(features[2:], expected_std, atol=1e-5)


def test_non_numeric_states_return_zeros():
    """Test graceful fallback to zeros for non-numeric state particles.

    Purpose: Validates that when particles are non-numeric (e.g., strings),
        the representation gracefully returns a zero feature vector instead
        of raising an error.

    Given: A custom Belief subclass whose sample() returns string values
    When: The representation is called on this belief
    Then: The output is a zero vector of shape (2 * state_dim,)

    Test type: unit
    """
    belief = _DummyBelief(["state_a", "state_b", "state_c"])

    rep = ParticleMeanStdRepresentation(state_dim=1)
    features = rep(belief)

    assert features.shape == (2,)
    np.testing.assert_array_equal(features, np.zeros(2, dtype=np.float32))
