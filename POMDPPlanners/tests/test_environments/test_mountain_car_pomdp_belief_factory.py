# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for the Mountain Car belief factory.

This module tests :func:`create_mountain_car_belief` from
:mod:`POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp_beliefs`.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp_beliefs import (
    create_mountain_car_belief,
)
from POMDPPlanners.utils.belief_factory import BeliefType


@pytest.fixture
def env():
    return MountainCarPOMDP(discount_factor=0.99)


class TestCreateMountainCarBelief:
    def test_default_returns_vectorized(self, env):
        """Test default belief type is VECTORIZED_PARTICLE.

        Purpose: Validates the default belief type returned by the factory.

        Given: A MountainCarPOMDP environment.
        When: create_mountain_car_belief is called with default arguments.
        Then: A VectorizedWeightedParticleBelief is returned.

        Test type: unit
        """
        belief = create_mountain_car_belief(env, n_particles=50)
        assert isinstance(belief, VectorizedWeightedParticleBelief)

    def test_particle_type_returns_weighted(self, env):
        """Test PARTICLE type returns WeightedParticleBelief.

        Purpose: Validates the PARTICLE belief creation path.

        Given: A MountainCarPOMDP environment.
        When: create_mountain_car_belief is called with BeliefType.PARTICLE.
        Then: A WeightedParticleBelief is returned.

        Test type: unit
        """
        belief = create_mountain_car_belief(env, belief_type=BeliefType.PARTICLE, n_particles=50)
        assert isinstance(belief, WeightedParticleBelief)

    def test_unsupported_type_raises(self, env):
        """Test unsupported belief type raises ValueError.

        Purpose: Validates error handling for unsupported belief types.

        Given: A MountainCarPOMDP environment.
        When: create_mountain_car_belief is called with BeliefType.GAUSSIAN_MIXTURE.
        Then: A ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="does not support"):
            create_mountain_car_belief(env, belief_type=BeliefType.GAUSSIAN_MIXTURE)

    def test_n_particles_respected(self, env):
        """Test that n_particles is respected for vectorized belief.

        Purpose: Validates particle count in vectorized belief.

        Given: A MountainCarPOMDP environment.
        When: create_mountain_car_belief is called with n_particles=30.
        Then: The belief has exactly 30 particles.

        Test type: unit
        """
        belief = create_mountain_car_belief(env, n_particles=30)
        assert isinstance(belief, VectorizedWeightedParticleBelief)
        assert belief.n_particles == 30

    def test_sample_shape(self, env):
        """Test that belief sample has the correct shape.

        Purpose: Validates end-to-end usability.

        Given: A MountainCarPOMDP environment and a created belief.
        When: sample() is called.
        Then: The sample has shape (2,).

        Test type: integration
        """
        np.random.seed(42)
        belief = create_mountain_car_belief(env, n_particles=50)
        assert belief.sample().shape == (2,)
