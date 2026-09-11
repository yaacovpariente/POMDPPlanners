# SPDX-License-Identifier: MIT

"""Tests for the occupancy-grid mapping belief factory and its registry entry."""

import numpy as np
import pytest

from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import (
    OccupancyGridMappingBelief,
    OccupancyGridMappingPOMDP,
    OccupancyGridMappingVectorizedBelief,
    create_occupancy_grid_mapping_belief,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    occupancy_grid_mapping_pinned_kwargs,
)
from POMDPPlanners.utils.belief_factory import BeliefType, create_environment_belief


@pytest.fixture
def env():
    return OccupancyGridMappingPOMDP(
        discount_factor=0.95, **occupancy_grid_mapping_pinned_kwargs(num_rows=8, num_cols=8)
    )


class TestEnvironmentFactory:
    def test_default_is_the_vectorized_filter(self, env):
        """Purpose: the batched filter is the one runs should get without asking.

        Test type: unit
        """
        belief = create_occupancy_grid_mapping_belief(env, n_particles=6)
        assert isinstance(belief, OccupancyGridMappingVectorizedBelief)
        assert belief.n_particles == 6
        assert belief.particles.shape == (6, env.state_size)

    def test_particle_is_the_scalar_whole_map_filter(self, env):
        """Purpose: ``PARTICLE`` must give the environment's own conditional
        filter, never the generic bootstrap filter, which cannot condition on
        a stored continuous scan.

        Test type: unit
        """
        belief = create_occupancy_grid_mapping_belief(env, BeliefType.PARTICLE, n_particles=6)
        assert isinstance(belief, OccupancyGridMappingBelief)
        assert len(belief.particles) == 6

    @pytest.mark.parametrize("belief_type", [BeliefType.GAUSSIAN, BeliefType.GAUSSIAN_MIXTURE])
    def test_other_types_are_rejected(self, env, belief_type):
        """Purpose: a whole-map prior has no Gaussian form to offer.

        Test type: unit
        """
        with pytest.raises(ValueError, match="does not support"):
            create_occupancy_grid_mapping_belief(env, belief_type)

    def test_both_filters_start_from_the_same_prior(self, env):
        """Purpose: the choice of filter must not change the maps an episode starts with.

        Test type: unit
        """
        np.random.seed(8)
        scalar = create_occupancy_grid_mapping_belief(env, BeliefType.PARTICLE, n_particles=5)
        np.random.seed(8)
        vectorized = create_occupancy_grid_mapping_belief(env, n_particles=5)
        assert isinstance(scalar, OccupancyGridMappingBelief)
        assert isinstance(vectorized, OccupancyGridMappingVectorizedBelief)
        np.testing.assert_array_equal(np.asarray(scalar.particles), vectorized.particles)


class TestTopLevelRegistry:
    def test_top_level_factory_routes_to_the_environment_factory(self, env):
        """Purpose: runs pick their belief through the top-level factory, so the
        environment must be registered there with the vectorized default.

        Test type: unit
        """
        assert isinstance(
            create_environment_belief(env, n_particles=4), OccupancyGridMappingVectorizedBelief
        )
        assert isinstance(
            create_environment_belief(env, BeliefType.PARTICLE, n_particles=4),
            OccupancyGridMappingBelief,
        )
        with pytest.raises(ValueError, match="does not support"):
            create_environment_belief(env, BeliefType.GAUSSIAN)
