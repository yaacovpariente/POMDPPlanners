"""Tests for the Continuous LaserTag belief factory.

Tests cover creation of vectorized and particle beliefs for both
continuous and discrete-action environment variants.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
    ContinuousLaserTagPOMDP,
    ContinuousLaserTagPOMDPDiscreteActions,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_beliefs.continuous_laser_tag_belief_factory import (
    create_continuous_laser_tag_belief,
)
from POMDPPlanners.utils.belief_factory import BeliefType, create_environment_belief


@pytest.fixture
def env():
    np.random.seed(42)
    return ContinuousLaserTagPOMDP(discount_factor=0.95, walls=[])


@pytest.fixture
def env_discrete():
    np.random.seed(42)
    return ContinuousLaserTagPOMDPDiscreteActions(discount_factor=0.95, walls=[])


class TestCreateContinuousLaserTagBelief:
    """Tests for the create_continuous_laser_tag_belief factory."""

    def test_vectorized_belief_creation(self, env):
        """Test that vectorized particle belief is created.

        Purpose: Validates vectorized belief creation.

        Given: A continuous LaserTag environment.
        When: Factory is called with VECTORIZED_PARTICLE type.
        Then: Returns a VectorizedWeightedParticleBelief.

        Test type: unit
        """
        np.random.seed(42)
        belief = create_continuous_laser_tag_belief(
            env,
            belief_type=BeliefType.VECTORIZED_PARTICLE,
            n_particles=50,
        )
        assert belief is not None
        sample = belief.sample()
        assert isinstance(sample, np.ndarray)
        assert sample.shape == (5,)

    def test_particle_belief_creation(self, env):
        """Test that standard particle belief is created.

        Purpose: Validates standard particle belief creation.

        Given: A continuous LaserTag environment.
        When: Factory is called with PARTICLE type.
        Then: Returns a valid belief object.

        Test type: unit
        """
        np.random.seed(42)
        belief = create_continuous_laser_tag_belief(
            env,
            belief_type=BeliefType.PARTICLE,
            n_particles=50,
        )
        assert belief is not None

    def test_unsupported_type_raises(self, env):
        """Test that unsupported belief type raises ValueError.

        Purpose: Validates error on unsupported belief type.

        Given: A continuous LaserTag environment.
        When: Factory is called with GAUSSIAN type.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="does not support"):
            create_continuous_laser_tag_belief(
                env,
                belief_type=BeliefType.GAUSSIAN,
            )

    def test_discrete_env_vectorized(self, env_discrete):
        """Test vectorized belief for discrete-action variant.

        Purpose: Validates factory works with discrete-action environment.

        Given: A discrete-action continuous LaserTag environment.
        When: Factory is called.
        Then: Returns a valid belief.

        Test type: unit
        """
        np.random.seed(42)
        belief = create_continuous_laser_tag_belief(env_discrete, n_particles=50)
        assert belief is not None


class TestBeliefFactoryRegistry:
    """Tests for top-level belief factory integration."""

    def test_create_environment_belief_continuous(self, env):
        """Test that create_environment_belief works for ContinuousLaserTagPOMDP.

        Purpose: Validates top-level factory registry integration.

        Given: A ContinuousLaserTagPOMDP instance.
        When: create_environment_belief is called.
        Then: Returns a valid belief.

        Test type: integration
        """
        np.random.seed(42)
        belief = create_environment_belief(env, n_particles=50)
        assert belief is not None

    def test_create_environment_belief_discrete(self, env_discrete):
        """Test that create_environment_belief works for discrete variant.

        Purpose: Validates registry for discrete-action variant.

        Given: A ContinuousLaserTagPOMDPDiscreteActions instance.
        When: create_environment_belief is called.
        Then: Returns a valid belief.

        Test type: integration
        """
        np.random.seed(42)
        belief = create_environment_belief(env_discrete, n_particles=50)
        assert belief is not None
