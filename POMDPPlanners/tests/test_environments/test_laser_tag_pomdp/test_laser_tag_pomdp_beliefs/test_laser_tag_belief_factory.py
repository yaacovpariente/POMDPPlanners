# SPDX-License-Identifier: MIT

"""Tests for the LaserTag belief factory.

This module tests :func:`create_laser_tag_belief` from
:mod:`POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_beliefs`.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import LaserTagPOMDP
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_beliefs import (
    create_laser_tag_belief,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    laser_tag_pinned_kwargs as _lt_pinned_kwargs,
)
from POMDPPlanners.utils.belief_factory import BeliefType


@pytest.fixture
def env():
    return LaserTagPOMDP(discount_factor=0.95, **_lt_pinned_kwargs())


class TestCreateLaserTagBelief:
    def test_default_returns_vectorized(self, env):
        """Test default belief type is VECTORIZED_PARTICLE.

        Purpose: Validates the default belief type returned by the factory.

        Given: A LaserTagPOMDP environment.
        When: create_laser_tag_belief is called with default arguments.
        Then: A VectorizedWeightedParticleBelief is returned.

        Test type: unit
        """
        belief = create_laser_tag_belief(env, n_particles=50)
        assert isinstance(belief, VectorizedWeightedParticleBelief)

    def test_particle_type_returns_weighted(self, env):
        """Test PARTICLE type returns WeightedParticleBelief.

        Purpose: Validates the PARTICLE belief creation path.

        Given: A LaserTagPOMDP environment.
        When: create_laser_tag_belief is called with BeliefType.PARTICLE.
        Then: A WeightedParticleBelief is returned.

        Test type: unit
        """
        belief = create_laser_tag_belief(env, belief_type=BeliefType.PARTICLE, n_particles=50)
        assert isinstance(belief, WeightedParticleBelief)

    def test_unsupported_type_raises(self, env):
        """Test unsupported belief type raises ValueError.

        Purpose: Validates error handling for unsupported belief types.

        Given: A LaserTagPOMDP environment.
        When: create_laser_tag_belief is called with BeliefType.GAUSSIAN.
        Then: A ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="does not support"):
            create_laser_tag_belief(env, belief_type=BeliefType.GAUSSIAN)

    def test_n_particles_respected(self, env):
        """Test that n_particles is respected for vectorized belief.

        Purpose: Validates particle count in vectorized belief.

        Given: A LaserTagPOMDP environment.
        When: create_laser_tag_belief is called with n_particles=30.
        Then: The belief has exactly 30 particles.

        Test type: unit
        """
        belief = create_laser_tag_belief(env, n_particles=30)
        assert isinstance(belief, VectorizedWeightedParticleBelief)
        assert belief.n_particles == 30

    def test_sample_shape(self, env):
        """Test that belief sample has the correct shape.

        Purpose: Validates end-to-end usability.

        Given: A LaserTagPOMDP environment and a created belief.
        When: sample() is called.
        Then: The sample has shape (5,).

        Test type: integration
        """
        np.random.seed(42)
        belief = create_laser_tag_belief(env, n_particles=50)
        assert belief.sample().shape == (5,)
