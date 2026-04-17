"""Tests for RockSample belief factory."""

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.environments.rock_sample_pomdp import RockSamplePOMDP
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp_beliefs.rocksample_belief_factory import (
    RockSampleVectorizedWeightedParticleBelief,
    create_rocksample_belief,
)
from POMDPPlanners.utils.belief_factory import BeliefType, create_environment_belief


@pytest.fixture()
def simple_env():
    return RockSamplePOMDP(
        map_size=(5, 5),
        rock_positions=[(1, 1), (3, 3)],
        init_pos=(2, 2),
        sensor_efficiency=10.0,
        discount_factor=0.95,
    )


class TestCreateVectorizedBelief:
    def test_returns_vectorized_type(self, simple_env):
        """Test that VECTORIZED_PARTICLE returns the custom subclass.

        Purpose: Validates correct return type for vectorized belief.

        Given: A RockSamplePOMDP environment.
        When: create_rocksample_belief is called with VECTORIZED_PARTICLE.
        Then: Returns RockSampleVectorizedWeightedParticleBelief.

        Test type: unit
        """
        belief = create_rocksample_belief(
            simple_env, belief_type=BeliefType.VECTORIZED_PARTICLE, n_particles=50
        )
        assert isinstance(belief, RockSampleVectorizedWeightedParticleBelief)

    def test_correct_shapes(self, simple_env):
        """Test that particle and weight arrays have correct shapes.

        Purpose: Validates dimensional correctness of belief arrays.

        Given: A RockSamplePOMDP with 2 rocks.
        When: Vectorized belief is created with 100 particles.
        Then: particles shape is (100, 4), log_weights shape is (100,).

        Test type: unit
        """
        belief = create_rocksample_belief(
            simple_env, belief_type=BeliefType.VECTORIZED_PARTICLE, n_particles=100
        )
        assert isinstance(belief, RockSampleVectorizedWeightedParticleBelief)
        assert belief.particles.shape == (100, 4)  # 2 pos + 2 rocks
        assert belief.log_weights.shape == (100,)


class TestCreateParticleBelief:
    def test_returns_particle_type(self, simple_env):
        """Test that PARTICLE returns a WeightedParticleBelief.

        Purpose: Validates correct return type for standard particle belief.

        Given: A RockSamplePOMDP environment.
        When: create_rocksample_belief is called with PARTICLE type.
        Then: Returns WeightedParticleBelief.

        Test type: unit
        """
        belief = create_rocksample_belief(
            simple_env, belief_type=BeliefType.PARTICLE, n_particles=50
        )
        assert isinstance(belief, WeightedParticleBelief)


class TestUnsupportedType:
    def test_gaussian_raises_value_error(self, simple_env):
        """Test that unsupported belief types raise ValueError.

        Purpose: Validates error handling for unsupported types.

        Given: A RockSamplePOMDP environment.
        When: create_rocksample_belief is called with GAUSSIAN.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="does not support"):
            create_rocksample_belief(simple_env, belief_type=BeliefType.GAUSSIAN, n_particles=50)


class TestRegistryDispatches:
    def test_global_factory_dispatches_to_rocksample(self, simple_env):
        """Test that the global belief factory dispatches to RockSample factory.

        Purpose: Validates global registry integration.

        Given: A RockSamplePOMDP environment.
        When: create_environment_belief is called.
        Then: Returns RockSampleVectorizedWeightedParticleBelief (default).

        Test type: integration
        """
        np.random.seed(42)
        belief = create_environment_belief(simple_env, n_particles=50)
        assert isinstance(belief, RockSampleVectorizedWeightedParticleBelief)
