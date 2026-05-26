# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for PacMan belief factory."""

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.pacman_pomdp import PacManPOMDP
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_beliefs.pacman_belief_factory import (
    create_pacman_belief,
)
from POMDPPlanners.utils.belief_factory import BeliefType, create_environment_belief


@pytest.fixture()
def env():
    return PacManPOMDP(
        maze_size=(5, 5),
        walls={(2, 2)},
        initial_pellets=[(1, 1), (3, 3)],
        initial_pacman_pos=(0, 0),
        num_ghosts=1,
        initial_ghost_positions=[(4, 4)],
        discount_factor=0.95,
    )


class TestCreateVectorizedBelief:
    def test_returns_vectorized_type(self, env):
        """Test that vectorized belief type is returned.

        Purpose: Validates factory creates VectorizedWeightedParticleBelief.

        Given: A PacManPOMDP environment.
        When: create_pacman_belief is called with VECTORIZED_PARTICLE type.
        Then: A VectorizedWeightedParticleBelief is returned.

        Test type: unit
        """
        np.random.seed(42)
        belief = create_pacman_belief(env, n_particles=50)
        assert isinstance(belief, VectorizedWeightedParticleBelief)

    def test_correct_shapes(self, env):
        """Test that vectorized belief has correct particle shapes.

        Purpose: Validates particle array dimensions.

        Given: A PacManPOMDP environment.
        When: Vectorized belief is created with 50 particles.
        Then: Particles shape is (50, state_dim).

        Test type: unit
        """
        np.random.seed(42)
        belief = create_pacman_belief(env, n_particles=50)
        assert isinstance(belief, VectorizedWeightedParticleBelief)
        assert belief.particles.shape == (50, env._state_dim)  # pylint: disable=protected-access
        assert belief.log_weights.shape == (50,)


class TestCreateParticleBelief:
    def test_returns_particle_type(self, env):
        """Test that particle belief type is returned.

        Purpose: Validates factory creates WeightedParticleBelief for PARTICLE type.

        Given: A PacManPOMDP environment.
        When: create_pacman_belief is called with PARTICLE type.
        Then: A WeightedParticleBelief is returned.

        Test type: unit
        """
        np.random.seed(42)
        belief = create_pacman_belief(env, belief_type=BeliefType.PARTICLE, n_particles=50)
        assert isinstance(belief, WeightedParticleBelief)


class TestUnsupportedType:
    def test_gaussian_raises_value_error(self, env):
        """Test that unsupported belief type raises ValueError.

        Purpose: Validates error handling for unsupported types.

        Given: A PacManPOMDP environment.
        When: create_pacman_belief is called with GAUSSIAN type.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="does not support"):
            create_pacman_belief(env, belief_type=BeliefType.GAUSSIAN)


class TestRegistryDispatches:
    def test_create_environment_belief_works(self, env):
        """Test that the global factory dispatches to PacMan factory.

        Purpose: Validates registry-based dispatch.

        Given: A PacManPOMDP environment registered in the factory.
        When: create_environment_belief is called.
        Then: A VectorizedWeightedParticleBelief is returned.

        Test type: integration
        """
        np.random.seed(42)
        belief = create_environment_belief(env, n_particles=50)
        assert isinstance(belief, VectorizedWeightedParticleBelief)
