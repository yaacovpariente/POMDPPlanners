"""Tests for the Push belief factory.

This module tests :func:`create_push_belief` from
:mod:`POMDPPlanners.environments.push_pomdp.push_pomdp_beliefs`.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP
from POMDPPlanners.environments.push_pomdp.push_pomdp_beliefs import (
    create_push_belief,
)
from POMDPPlanners.utils.belief_factory import BeliefType


@pytest.fixture
def env():
    return PushPOMDP(discount_factor=0.95)


class TestCreatePushBelief:
    def test_default_returns_vectorized(self, env):
        """Test default belief type is VECTORIZED_PARTICLE.

        Purpose: Validates the default belief type returned by the factory.

        Given: A PushPOMDP environment.
        When: create_push_belief is called with default arguments.
        Then: A VectorizedWeightedParticleBelief is returned.

        Test type: unit
        """
        belief = create_push_belief(env, n_particles=50)
        assert isinstance(belief, VectorizedWeightedParticleBelief)

    def test_particle_type_returns_weighted(self, env):
        """Test PARTICLE type returns WeightedParticleBelief.

        Purpose: Validates the PARTICLE belief creation path.

        Given: A PushPOMDP environment.
        When: create_push_belief is called with BeliefType.PARTICLE.
        Then: A WeightedParticleBelief is returned.

        Test type: unit
        """
        belief = create_push_belief(env, belief_type=BeliefType.PARTICLE, n_particles=50)
        assert isinstance(belief, WeightedParticleBelief)

    def test_unsupported_type_raises(self, env):
        """Test unsupported belief type raises ValueError.

        Purpose: Validates error handling for unsupported belief types.

        Given: A PushPOMDP environment.
        When: create_push_belief is called with BeliefType.GAUSSIAN.
        Then: A ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="does not support"):
            create_push_belief(env, belief_type=BeliefType.GAUSSIAN)

    def test_n_particles_respected(self, env):
        """Test that n_particles is respected for vectorized belief.

        Purpose: Validates particle count in vectorized belief.

        Given: A PushPOMDP environment.
        When: create_push_belief is called with n_particles=30.
        Then: The belief has exactly 30 particles.

        Test type: unit
        """
        belief = create_push_belief(env, n_particles=30)
        assert isinstance(belief, VectorizedWeightedParticleBelief)
        assert belief.n_particles == 30

    def test_sample_shape(self, env):
        """Test that belief sample has the correct shape.

        Purpose: Validates end-to-end usability.

        Given: A PushPOMDP environment and a created belief.
        When: sample() is called.
        Then: The sample has shape (6,).

        Test type: integration
        """
        np.random.seed(42)
        belief = create_push_belief(env, n_particles=50)
        assert belief.sample().shape == (6,)
