# SPDX-License-Identifier: MIT

"""Tests for the CartPole belief factory.

This module tests :func:`create_cartpole_belief` from
:mod:`POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp_beliefs`.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp_beliefs import create_cartpole_belief
from POMDPPlanners.utils.belief_factory import BeliefType


@pytest.fixture
def env():
    return CartPolePOMDP(discount_factor=0.99, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))


class TestCreateCartpoleBelief:
    def test_default_returns_vectorized(self, env):
        """Test default belief type is VECTORIZED_PARTICLE.

        Purpose: Validates the default belief type returned by the factory.

        Given: A CartPolePOMDP environment.
        When: create_cartpole_belief is called with default arguments.
        Then: A VectorizedWeightedParticleBelief is returned.

        Test type: unit
        """
        belief = create_cartpole_belief(env, n_particles=50)
        assert isinstance(belief, VectorizedWeightedParticleBelief)

    def test_particle_type_returns_weighted(self, env):
        """Test PARTICLE type returns WeightedParticleBelief.

        Purpose: Validates the PARTICLE belief creation path.

        Given: A CartPolePOMDP environment.
        When: create_cartpole_belief is called with BeliefType.PARTICLE.
        Then: A WeightedParticleBelief is returned.

        Test type: unit
        """
        belief = create_cartpole_belief(env, belief_type=BeliefType.PARTICLE, n_particles=50)
        assert isinstance(belief, WeightedParticleBelief)

    def test_vectorized_type_returns_vectorized(self, env):
        """Test explicit VECTORIZED_PARTICLE type.

        Purpose: Validates explicit VECTORIZED_PARTICLE selection.

        Given: A CartPolePOMDP environment.
        When: create_cartpole_belief is called with VECTORIZED_PARTICLE.
        Then: A VectorizedWeightedParticleBelief is returned.

        Test type: unit
        """
        belief = create_cartpole_belief(
            env, belief_type=BeliefType.VECTORIZED_PARTICLE, n_particles=50
        )
        assert isinstance(belief, VectorizedWeightedParticleBelief)

    def test_unsupported_type_raises(self, env):
        """Test unsupported belief type raises ValueError.

        Purpose: Validates error handling for unsupported belief types.

        Given: A CartPolePOMDP environment.
        When: create_cartpole_belief is called with BeliefType.GAUSSIAN_MIXTURE.
        Then: A ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="does not support"):
            create_cartpole_belief(env, belief_type=BeliefType.GAUSSIAN_MIXTURE)

    def test_n_particles_respected_vectorized(self, env):
        """Test that n_particles is respected for vectorized belief.

        Purpose: Validates particle count in vectorized belief.

        Given: A CartPolePOMDP environment.
        When: create_cartpole_belief is called with n_particles=30.
        Then: The belief has exactly 30 particles.

        Test type: unit
        """
        belief = create_cartpole_belief(env, n_particles=30)
        assert isinstance(belief, VectorizedWeightedParticleBelief)
        assert belief.n_particles == 30

    def test_sample_shape(self, env):
        """Test that belief sample has the correct shape.

        Purpose: Validates end-to-end usability.

        Given: A CartPolePOMDP environment and a created belief.
        When: sample() is called.
        Then: The sample has shape (4,).

        Test type: integration
        """
        np.random.seed(42)
        belief = create_cartpole_belief(env, n_particles=50)
        assert belief.sample().shape == (4,)
