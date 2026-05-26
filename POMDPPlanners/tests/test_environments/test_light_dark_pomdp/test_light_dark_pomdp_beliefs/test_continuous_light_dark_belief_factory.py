# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for the Continuous Light-Dark belief factory.

This module tests :func:`create_continuous_light_dark_belief` from
:mod:`POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_beliefs`.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief.gaussian_belief import GaussianBelief
from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_beliefs import (
    GaussianBeliefUpdaterType,
    create_continuous_light_dark_belief,
)
from POMDPPlanners.utils.belief_factory import BeliefType


@pytest.fixture
def env():
    return ContinuousLightDarkPOMDP(discount_factor=0.95)


class TestCreateContinuousLightDarkBelief:
    def test_default_returns_vectorized(self, env):
        """Test default belief type is VECTORIZED_PARTICLE.

        Purpose: Validates the default belief type returned by the factory.

        Given: A ContinuousLightDarkPOMDP environment.
        When: create_continuous_light_dark_belief is called with default args.
        Then: A VectorizedWeightedParticleBelief is returned.

        Test type: unit
        """
        belief = create_continuous_light_dark_belief(env, n_particles=50)
        assert isinstance(belief, VectorizedWeightedParticleBelief)

    def test_particle_type_returns_weighted(self, env):
        """Test PARTICLE type returns WeightedParticleBelief.

        Purpose: Validates the PARTICLE belief creation path.

        Given: A ContinuousLightDarkPOMDP environment.
        When: create_continuous_light_dark_belief is called with PARTICLE.
        Then: A WeightedParticleBelief is returned.

        Test type: unit
        """
        belief = create_continuous_light_dark_belief(
            env, belief_type=BeliefType.PARTICLE, n_particles=50
        )
        assert isinstance(belief, WeightedParticleBelief)

    def test_gaussian_type_returns_gaussian(self, env):
        """Test GAUSSIAN type returns GaussianBelief.

        Purpose: Validates the GAUSSIAN belief creation path.

        Given: A ContinuousLightDarkPOMDP environment.
        When: create_continuous_light_dark_belief is called with GAUSSIAN.
        Then: A GaussianBelief is returned.

        Test type: unit
        """
        belief = create_continuous_light_dark_belief(env, belief_type=BeliefType.GAUSSIAN)
        assert isinstance(belief, GaussianBelief)

    def test_gaussian_with_ukf(self, env):
        """Test GAUSSIAN with UKF updater type.

        Purpose: Validates that the default Gaussian uses UKF.

        Given: A ContinuousLightDarkPOMDP environment.
        When: create_continuous_light_dark_belief is called with GAUSSIAN.
        Then: A GaussianBelief with correct mean shape is returned.

        Test type: unit
        """
        belief = create_continuous_light_dark_belief(env, belief_type=BeliefType.GAUSSIAN)
        assert isinstance(belief, GaussianBelief)
        assert belief.mean.shape == (2,)

    def test_gaussian_with_linear_kalman(self, env):
        """Test GAUSSIAN with explicit LINEAR_KALMAN updater.

        Purpose: Validates passing updater_type kwarg.

        Given: A ContinuousLightDarkPOMDP environment.
        When: create_continuous_light_dark_belief is called with GAUSSIAN
            and updater_type=LINEAR_KALMAN.
        Then: A GaussianBelief is returned.

        Test type: unit
        """
        belief = create_continuous_light_dark_belief(
            env,
            belief_type=BeliefType.GAUSSIAN,
            updater_type=GaussianBeliefUpdaterType.LINEAR_KALMAN,
        )
        assert isinstance(belief, GaussianBelief)

    def test_gaussian_with_custom_covariance(self, env):
        """Test GAUSSIAN with custom initial covariance.

        Purpose: Validates passing initial_covariance kwarg.

        Given: A ContinuousLightDarkPOMDP environment.
        When: create_continuous_light_dark_belief is called with GAUSSIAN
            and a custom initial_covariance.
        Then: A GaussianBelief is returned with the given covariance.

        Test type: unit
        """
        cov = np.eye(2) * 10.0
        belief = create_continuous_light_dark_belief(
            env,
            belief_type=BeliefType.GAUSSIAN,
            initial_covariance=cov,
        )
        assert isinstance(belief, GaussianBelief)
        np.testing.assert_array_equal(belief.covariance, cov)

    def test_unsupported_type_raises(self, env):
        """Test unsupported belief type raises ValueError.

        Purpose: Validates error handling for unsupported belief types.

        Given: A ContinuousLightDarkPOMDP environment.
        When: create_continuous_light_dark_belief is called with
            GAUSSIAN_MIXTURE.
        Then: A ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="does not support"):
            create_continuous_light_dark_belief(env, belief_type=BeliefType.GAUSSIAN_MIXTURE)

    def test_n_particles_respected(self, env):
        """Test that n_particles is respected for vectorized belief.

        Purpose: Validates particle count in vectorized belief.

        Given: A ContinuousLightDarkPOMDP environment.
        When: create_continuous_light_dark_belief is called with n_particles=30.
        Then: The belief has exactly 30 particles.

        Test type: unit
        """
        belief = create_continuous_light_dark_belief(env, n_particles=30)
        assert isinstance(belief, VectorizedWeightedParticleBelief)
        assert belief.n_particles == 30

    def test_sample_shape_vectorized(self, env):
        """Test that vectorized belief sample has the correct shape.

        Purpose: Validates end-to-end usability for vectorized beliefs.

        Given: A ContinuousLightDarkPOMDP environment and a vectorized belief.
        When: sample() is called.
        Then: The sample has shape (2,).

        Test type: integration
        """
        np.random.seed(42)
        belief = create_continuous_light_dark_belief(env, n_particles=50)
        assert belief.sample().shape == (2,)

    def test_sample_shape_gaussian(self, env):
        """Test that Gaussian belief sample has the correct shape.

        Purpose: Validates end-to-end usability for Gaussian beliefs.

        Given: A ContinuousLightDarkPOMDP environment and a Gaussian belief.
        When: sample() is called.
        Then: The sample has shape (2,).

        Test type: integration
        """
        np.random.seed(42)
        belief = create_continuous_light_dark_belief(env, belief_type=BeliefType.GAUSSIAN)
        assert belief.sample().shape == (2,)
