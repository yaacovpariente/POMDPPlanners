# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for the top-level belief factory.

This module tests :func:`create_environment_belief` and :class:`BeliefType`
from :mod:`POMDPPlanners.utils.belief_factory`.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.tests.test_utils.metric_invariants_utils import (
    verify_belief_invariants,
)
from POMDPPlanners.utils.belief_factory import BeliefType, create_environment_belief


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiger_env():
    from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

    return TigerPOMDP(discount_factor=0.95)


@pytest.fixture
def cartpole_env():
    from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP

    return CartPolePOMDP(discount_factor=0.99, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))


@pytest.fixture
def mountain_car_env():
    from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP

    return MountainCarPOMDP(discount_factor=0.99)


@pytest.fixture
def light_dark_env():
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        ContinuousLightDarkPOMDP,
    )

    return ContinuousLightDarkPOMDP(discount_factor=0.95)


# ---------------------------------------------------------------------------
# BeliefType enum
# ---------------------------------------------------------------------------


class TestBeliefType:
    def test_enum_values(self):
        """Test that BeliefType enum contains expected members.

        Purpose: Validates the BeliefType enum has all required members.

        Given: The BeliefType enum.
        When: Members are accessed by name.
        Then: All four belief types exist with correct string values.

        Test type: unit
        """
        assert BeliefType.PARTICLE.value == "particle"
        assert BeliefType.VECTORIZED_PARTICLE.value == "vectorized_particle"
        assert BeliefType.GAUSSIAN.value == "gaussian"
        assert BeliefType.GAUSSIAN_MIXTURE.value == "gaussian_mixture"

    def test_enum_member_count(self):
        """Test that BeliefType enum has exactly four members.

        Purpose: Validates no extra or missing enum members.

        Given: The BeliefType enum.
        When: The members are counted.
        Then: There are exactly four members.

        Test type: unit
        """
        assert len(BeliefType) == 4


# ---------------------------------------------------------------------------
# Fallback environments (no custom factory)
# ---------------------------------------------------------------------------


class TestFallbackEnvironments:
    def test_tiger_default_returns_particle_belief(self, tiger_env):
        """Test that Tiger POMDP defaults to WeightedParticleBelief.

        Purpose: Validates the fallback path for environments without
        a registered per-env factory.

        Given: A TigerPOMDP environment.
        When: create_environment_belief is called with no belief_type.
        Then: A WeightedParticleBelief is returned.

        Test type: unit
        """
        belief = create_environment_belief(tiger_env, n_particles=50)
        assert isinstance(belief, WeightedParticleBelief)
        verify_belief_invariants(belief, expected_n_particles=50)

    def test_tiger_explicit_particle(self, tiger_env):
        """Test that Tiger POMDP accepts BeliefType.PARTICLE.

        Purpose: Validates explicit PARTICLE type selection for fallback envs.

        Given: A TigerPOMDP environment.
        When: create_environment_belief is called with PARTICLE.
        Then: A WeightedParticleBelief is returned.

        Test type: unit
        """
        belief = create_environment_belief(
            tiger_env, belief_type=BeliefType.PARTICLE, n_particles=50
        )
        assert isinstance(belief, WeightedParticleBelief)

    def test_tiger_unsupported_type_raises(self, tiger_env):
        """Test that Tiger POMDP rejects unsupported belief types.

        Purpose: Validates error handling for environments that only
        support PARTICLE beliefs.

        Given: A TigerPOMDP environment.
        When: create_environment_belief is called with VECTORIZED_PARTICLE.
        Then: A ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="does not have a custom belief factory"):
            create_environment_belief(tiger_env, belief_type=BeliefType.VECTORIZED_PARTICLE)

    def test_tiger_sample_returns_valid_state(self, tiger_env):
        """Test that the fallback belief produces valid samples.

        Purpose: Validates end-to-end usability of the fallback belief.

        Given: A TigerPOMDP environment.
        When: A belief is created and sampled.
        Then: The sample is a valid tiger state.

        Test type: integration
        """
        np.random.seed(42)
        belief = create_environment_belief(tiger_env, n_particles=50)
        sample = belief.sample()
        assert sample in ["tiger-left", "tiger-right", "tiger_left", "tiger_right"]


# ---------------------------------------------------------------------------
# Registered environments
# ---------------------------------------------------------------------------


class TestCartPole:
    def test_default_returns_vectorized(self, cartpole_env):
        """Test CartPole default is VectorizedWeightedParticleBelief.

        Purpose: Validates the default belief type for CartPolePOMDP.

        Given: A CartPolePOMDP environment.
        When: create_environment_belief is called with no belief_type.
        Then: A VectorizedWeightedParticleBelief is returned.

        Test type: unit
        """
        belief = create_environment_belief(cartpole_env, n_particles=50)
        assert isinstance(belief, VectorizedWeightedParticleBelief)
        verify_belief_invariants(belief, expected_n_particles=50)

    def test_particle_returns_weighted(self, cartpole_env):
        """Test CartPole PARTICLE type returns WeightedParticleBelief.

        Purpose: Validates the PARTICLE path for CartPolePOMDP.

        Given: A CartPolePOMDP environment.
        When: create_environment_belief is called with PARTICLE.
        Then: A WeightedParticleBelief is returned.

        Test type: unit
        """
        belief = create_environment_belief(
            cartpole_env, belief_type=BeliefType.PARTICLE, n_particles=50
        )
        assert isinstance(belief, WeightedParticleBelief)

    def test_unsupported_type_raises(self, cartpole_env):
        """Test CartPole rejects unsupported belief types.

        Purpose: Validates error handling for unsupported types.

        Given: A CartPolePOMDP environment.
        When: create_environment_belief is called with GAUSSIAN_MIXTURE.
        Then: A ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="does not support"):
            create_environment_belief(cartpole_env, belief_type=BeliefType.GAUSSIAN_MIXTURE)

    def test_sample_shape(self, cartpole_env):
        """Test CartPole belief sample has correct shape.

        Purpose: Validates end-to-end usability of the CartPole belief.

        Given: A CartPolePOMDP environment.
        When: A belief is created and sampled.
        Then: The sample has shape (4,).

        Test type: integration
        """
        np.random.seed(42)
        belief = create_environment_belief(cartpole_env, n_particles=50)
        assert belief.sample().shape == (4,)

    def test_n_particles_respected(self, cartpole_env):
        """Test that n_particles parameter is respected.

        Purpose: Validates that the particle count is correctly set.

        Given: A CartPolePOMDP environment.
        When: create_environment_belief is called with n_particles=30.
        Then: The belief has exactly 30 particles.

        Test type: unit
        """
        belief = create_environment_belief(cartpole_env, n_particles=30)
        assert isinstance(belief, VectorizedWeightedParticleBelief)
        assert belief.n_particles == 30


class TestMountainCar:
    def test_default_returns_vectorized(self, mountain_car_env):
        """Test MountainCar default is VectorizedWeightedParticleBelief.

        Purpose: Validates the default belief type for MountainCarPOMDP.

        Given: A MountainCarPOMDP environment.
        When: create_environment_belief is called with no belief_type.
        Then: A VectorizedWeightedParticleBelief is returned.

        Test type: unit
        """
        belief = create_environment_belief(mountain_car_env, n_particles=50)
        assert isinstance(belief, VectorizedWeightedParticleBelief)
        verify_belief_invariants(belief, expected_n_particles=50)

    def test_sample_shape(self, mountain_car_env):
        """Test MountainCar belief sample has correct shape.

        Purpose: Validates end-to-end usability of the MountainCar belief.

        Given: A MountainCarPOMDP environment.
        When: A belief is created and sampled.
        Then: The sample has shape (2,).

        Test type: integration
        """
        np.random.seed(42)
        belief = create_environment_belief(mountain_car_env, n_particles=50)
        assert belief.sample().shape == (2,)


class TestContinuousLightDark:
    def test_default_returns_vectorized(self, light_dark_env):
        """Test Light-Dark default is VectorizedWeightedParticleBelief.

        Purpose: Validates the default belief type for ContinuousLightDarkPOMDP.

        Given: A ContinuousLightDarkPOMDP environment.
        When: create_environment_belief is called with no belief_type.
        Then: A VectorizedWeightedParticleBelief is returned.

        Test type: unit
        """
        belief = create_environment_belief(light_dark_env, n_particles=50)
        assert isinstance(belief, VectorizedWeightedParticleBelief)
        verify_belief_invariants(belief, expected_n_particles=50)

    def test_gaussian_returns_gaussian_belief(self, light_dark_env):
        """Test Light-Dark GAUSSIAN type returns GaussianBelief.

        Purpose: Validates the GAUSSIAN path for ContinuousLightDarkPOMDP.

        Given: A ContinuousLightDarkPOMDP environment.
        When: create_environment_belief is called with GAUSSIAN.
        Then: A GaussianBelief is returned.

        Test type: unit
        """
        from POMDPPlanners.core.belief.gaussian_belief import GaussianBelief

        belief = create_environment_belief(light_dark_env, belief_type=BeliefType.GAUSSIAN)
        assert isinstance(belief, GaussianBelief)

    def test_gaussian_with_custom_updater(self, light_dark_env):
        """Test Light-Dark GAUSSIAN with explicit updater_type.

        Purpose: Validates that kwargs are forwarded to the Gaussian factory.

        Given: A ContinuousLightDarkPOMDP environment.
        When: create_environment_belief is called with GAUSSIAN and
            updater_type=LINEAR_KALMAN.
        Then: A GaussianBelief is returned with correct mean shape.

        Test type: integration
        """
        from POMDPPlanners.core.belief.gaussian_belief import GaussianBelief
        from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_beliefs import (
            GaussianBeliefUpdaterType,
        )

        belief = create_environment_belief(
            light_dark_env,
            belief_type=BeliefType.GAUSSIAN,
            updater_type=GaussianBeliefUpdaterType.LINEAR_KALMAN,
            initial_covariance=np.eye(2) * 3.0,
        )
        assert isinstance(belief, GaussianBelief)
        assert belief.mean.shape == (2,)

    def test_unsupported_type_raises(self, light_dark_env):
        """Test Light-Dark rejects unsupported belief types.

        Purpose: Validates error handling for unsupported types.

        Given: A ContinuousLightDarkPOMDP environment.
        When: create_environment_belief is called with GAUSSIAN_MIXTURE.
        Then: A ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="does not support"):
            create_environment_belief(light_dark_env, belief_type=BeliefType.GAUSSIAN_MIXTURE)
