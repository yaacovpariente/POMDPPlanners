"""Tests for the Continuous Push belief factory.

This module tests :func:`create_continuous_push_belief` from
:mod:`POMDPPlanners.environments.push_pomdp.push_pomdp_beliefs.continuous_push_belief_factory`.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import (
    ContinuousPushPOMDP,
    ContinuousPushPOMDPDiscreteActions,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp_beliefs.continuous_push_belief_factory import (
    create_continuous_push_belief,
)
from POMDPPlanners.utils.belief_factory import BeliefType, create_environment_belief


class TestContinuousPushBeliefFactory:
    """Test the belief factory for continuous push POMDP."""

    def setup_method(self):
        """Set up shared test fixtures."""
        np.random.seed(42)
        self.env = ContinuousPushPOMDP(
            discount_factor=0.99,
            state_transition_cov_matrix=np.eye(2) * 0.01,
            robot_radius=0.3,
        )

    def test_create_vectorized_belief(self):
        """Test creating vectorized particle belief.

        Purpose: Validates default belief creation (VECTORIZED_PARTICLE).

        Given: A ContinuousPushPOMDP environment.
        When: create_continuous_push_belief is called with default type.
        Then: Returns a VectorizedWeightedParticleBelief.

        Test type: unit
        """
        from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
            VectorizedWeightedParticleBelief,
        )

        belief = create_continuous_push_belief(self.env, n_particles=50)
        assert isinstance(belief, VectorizedWeightedParticleBelief)

    def test_create_particle_belief(self):
        """Test creating standard particle belief.

        Purpose: Validates PARTICLE belief creation.

        Given: A ContinuousPushPOMDP environment.
        When: create_continuous_push_belief is called with PARTICLE type.
        Then: Returns a Belief object.

        Test type: unit
        """
        belief = create_continuous_push_belief(
            self.env, belief_type=BeliefType.PARTICLE, n_particles=50
        )
        assert belief is not None

    def test_unsupported_type_raises(self):
        """Test that unsupported belief type raises ValueError.

        Purpose: Validates error handling for unsupported types.

        Given: A ContinuousPushPOMDP environment.
        When: create_continuous_push_belief is called with GAUSSIAN type.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="does not support"):
            create_continuous_push_belief(self.env, belief_type=BeliefType.GAUSSIAN)

    def test_registry_integration(self):
        """Test that belief factory registry dispatches correctly.

        Purpose: Validates top-level registry integration.

        Given: A ContinuousPushPOMDP environment.
        When: create_environment_belief is called.
        Then: Returns a valid belief.

        Test type: integration
        """
        belief = create_environment_belief(self.env, n_particles=50)
        assert belief is not None

    def test_registry_integration_discrete(self):
        """Test registry integration with discrete wrapper.

        Purpose: Validates top-level registry for discrete variant.

        Given: A ContinuousPushPOMDPDiscreteActions environment.
        When: create_environment_belief is called.
        Then: Returns a valid belief.

        Test type: integration
        """
        env_d = ContinuousPushPOMDPDiscreteActions(
            discount_factor=0.99,
            state_transition_cov_matrix=np.eye(2) * 0.01,
            robot_radius=0.3,
        )
        belief = create_environment_belief(env_d, n_particles=50)
        assert belief is not None
