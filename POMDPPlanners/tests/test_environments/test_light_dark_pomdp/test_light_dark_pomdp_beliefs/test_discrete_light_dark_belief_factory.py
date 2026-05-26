# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for the Discrete Light-Dark belief factory.

This module tests that create_discrete_light_dark_belief correctly dispatches
to the appropriate belief type and observation-model-specific updater.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
    ObservationModelType,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_beliefs import (
    DiscreteLightDarkDistanceBasedVectorizedUpdater,
    DiscreteLightDarkNoObsInDarkVectorizedUpdater,
    DiscreteLightDarkVectorizedUpdater,
    create_discrete_light_dark_belief,
)
from POMDPPlanners.utils.belief_factory import BeliefType


@pytest.fixture
def env():
    return DiscreteLightDarkPOMDP(discount_factor=0.95)


class TestCreateDiscreteLightDarkBelief:
    def test_default_type_is_vectorized_particle(self, env):
        """Test that default belief type is VECTORIZED_PARTICLE.

        Purpose: Validates the default factory behavior.

        Given: A DiscreteLightDarkPOMDP instance.
        When: create_discrete_light_dark_belief is called with defaults.
        Then: A VectorizedWeightedParticleBelief is returned.

        Test type: unit
        """
        np.random.seed(42)
        belief = create_discrete_light_dark_belief(env)
        assert isinstance(belief, VectorizedWeightedParticleBelief)

    def test_particle_type_returns_weighted_particle_belief(self, env):
        """Test that PARTICLE type returns WeightedParticleBelief.

        Purpose: Validates PARTICLE type dispatch.

        Given: A DiscreteLightDarkPOMDP instance.
        When: create_discrete_light_dark_belief is called with PARTICLE.
        Then: A WeightedParticleBelief is returned.

        Test type: unit
        """
        np.random.seed(42)
        belief = create_discrete_light_dark_belief(env, belief_type=BeliefType.PARTICLE)
        assert isinstance(belief, WeightedParticleBelief)

    def test_unsupported_type_raises_value_error(self, env):
        """Test that unsupported belief type raises ValueError.

        Purpose: Validates error handling for unsupported types.

        Given: A DiscreteLightDarkPOMDP instance.
        When: create_discrete_light_dark_belief is called with GAUSSIAN.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="does not support"):
            create_discrete_light_dark_belief(env, belief_type=BeliefType.GAUSSIAN)

    def test_n_particles_respected(self, env):
        """Test that n_particles parameter is respected.

        Purpose: Validates that the particle count is correct.

        Given: A DiscreteLightDarkPOMDP instance and n_particles=50.
        When: create_discrete_light_dark_belief is called.
        Then: The belief has 50 particles.

        Test type: unit
        """
        np.random.seed(42)
        belief = create_discrete_light_dark_belief(env, n_particles=50)
        assert isinstance(belief, VectorizedWeightedParticleBelief)
        assert belief.particles.shape[0] == 50


class TestObservationModelTypeDispatch:
    def test_normal_obs_model_uses_base_updater(self):
        """Test that NORMAL obs model produces base updater.

        Purpose: Validates updater dispatch for NORMAL observation model.

        Given: A DiscreteLightDarkPOMDP with NORMAL observation model.
        When: Vectorized belief is created.
        Then: The updater is a DiscreteLightDarkVectorizedUpdater.

        Test type: unit
        """
        np.random.seed(42)
        env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            observation_model_type=ObservationModelType.NORMAL,
        )
        belief = create_discrete_light_dark_belief(env)
        assert isinstance(belief, VectorizedWeightedParticleBelief)
        assert isinstance(belief.updater, DiscreteLightDarkVectorizedUpdater)
        assert not isinstance(belief.updater, DiscreteLightDarkNoObsInDarkVectorizedUpdater)

    def test_no_obs_in_dark_uses_correct_updater(self):
        """Test that NO_OBS_IN_DARK obs model produces correct updater.

        Purpose: Validates updater dispatch for NO_OBS_IN_DARK observation model.

        Given: A DiscreteLightDarkPOMDP with NO_OBS_IN_DARK observation model.
        When: Vectorized belief is created.
        Then: The updater is a DiscreteLightDarkNoObsInDarkVectorizedUpdater.

        Test type: unit
        """
        np.random.seed(42)
        env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            observation_model_type=ObservationModelType.NO_OBS_IN_DARK,
        )
        belief = create_discrete_light_dark_belief(env)
        assert isinstance(belief, VectorizedWeightedParticleBelief)
        assert isinstance(belief.updater, DiscreteLightDarkNoObsInDarkVectorizedUpdater)

    def test_distance_based_uses_correct_updater(self):
        """Test that DISTANCE_BASED obs model produces correct updater.

        Purpose: Validates updater dispatch for DISTANCE_BASED observation model.

        Given: A DiscreteLightDarkPOMDP with DISTANCE_BASED observation model.
        When: Vectorized belief is created.
        Then: The updater is a DiscreteLightDarkDistanceBasedVectorizedUpdater.

        Test type: unit
        """
        np.random.seed(42)
        env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            observation_model_type=ObservationModelType.DISTANCE_BASED,
        )
        belief = create_discrete_light_dark_belief(env)
        assert isinstance(belief, VectorizedWeightedParticleBelief)
        assert isinstance(belief.updater, DiscreteLightDarkDistanceBasedVectorizedUpdater)
