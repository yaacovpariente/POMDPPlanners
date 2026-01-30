"""Tests for Belief abstract base class.

This module tests the Belief ABC, focusing on:
- config_id generation and determinism
- Hash and equality semantics
- from_config factory method
"""

import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import (
    Belief,
    WeightedParticleBelief,
)
from POMDPPlanners.core.config_types import BeliefConfig
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.utils.weighted_particle_beliefs import (
    WeightedParticleBeliefContinuousLightDarkFullCoverage,
    WeightedParticleBeliefDiscreteLightDark,
    WeightedParticleBeliefDiscreteLightDarkFullCoverage,
    WeightedParticleBeliefSanityPOMDP,
)

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


def create_belief(env, config):
    """Create a belief from configuration."""
    if config.class_name == "WeightedParticleBelief":
        particles = env.initial_state_dist().sample(n_samples=config.params.get("n_particles", 10))
        log_weights = np.log(np.ones(len(particles)) / len(particles))
        return WeightedParticleBelief(
            particles=particles,
            log_weights=log_weights,
            resampling=config.params.get("resampling", True),
            ess_factor=config.params.get("ess_factor", 0.5),
        )
    elif config.class_name == "WeightedParticleBeliefDiscreteLightDark":
        particles = env.initial_state_dist().sample(n_samples=config.params.get("n_particles", 10))
        log_weights = np.log(np.ones(len(particles)) / len(particles))
        return WeightedParticleBeliefDiscreteLightDark(
            particles=particles,
            log_weights=log_weights,
            resampling=config.params.get("resampling", True),
            ess_factor=config.params.get("ess_factor", 0.5),
            reinvigoration_fraction=config.params.get("reinvigoration_fraction", 0.2),
        )
    elif config.class_name == "WeightedParticleBeliefDiscreteLightDarkFullCoverage":
        particles = env.initial_state_dist().sample(n_samples=config.params.get("n_particles", 10))
        log_weights = np.log(np.ones(len(particles)) / len(particles))
        return WeightedParticleBeliefDiscreteLightDarkFullCoverage(
            particles=particles,
            log_weights=log_weights,
            ess_factor=config.params.get("ess_factor", 0.5),
            reinvigoration_fraction=config.params.get("reinvigoration_fraction", 0.05),
        )
    elif config.class_name == "WeightedParticleBeliefContinuousLightDarkFullCoverage":
        particles = env.initial_state_dist().sample(n_samples=config.params.get("n_particles", 10))
        log_weights = np.log(np.ones(len(particles)) / len(particles))
        return WeightedParticleBeliefContinuousLightDarkFullCoverage(
            particles=particles,
            log_weights=log_weights,
            ess_factor=config.params.get("ess_factor", 0.5),
            reinvigoration_fraction=config.params.get("reinvigoration_fraction", 0.05),
            reinvigoration_cov_matrix=config.params.get("reinvigoration_cov_matrix", np.eye(2)),
        )
    elif config.class_name == "WeightedParticleBeliefSanityPOMDP":
        particles = env.initial_state_dist().sample(n_samples=config.params.get("n_particles", 10))
        log_weights = np.log(np.ones(len(particles)) / len(particles))
        return WeightedParticleBeliefSanityPOMDP(
            particles=particles,
            log_weights=log_weights,
            resampling=config.params.get("resampling", True),
            ess_factor=config.params.get("ess_factor", 0.5),
            reinvigoration_fraction=config.params.get("reinvigoration_fraction", 0.2),
        )
    else:
        raise ValueError(f"Unknown belief class: {config.class_name}")


def test_belief_config_id_identical_values_produces_same_hash():
    """
    Purpose: Validates that identical belief objects produce the same config_id hash for caching

    Given: Two TestBelief instances initialized with identical values (42)
    When: Config IDs are generated for both belief objects
    Then: Both beliefs produce identical config_id values for proper cache functionality

    Test type: unit
    """

    # ARRANGE: Define test belief class and create identical instances
    class TestBelief(Belief):
        def __init__(self, value):
            self.value = value

        def update(self, action, observation, pomdp, state=None):
            return self

        def sample(self):
            return self.value

    identical_value = 42
    belief1 = TestBelief(value=identical_value)
    belief2 = TestBelief(value=identical_value)

    # ACT: Generate config IDs for both beliefs
    config_id1 = belief1.config_id
    config_id2 = belief2.config_id

    # ASSERT: Verify identical beliefs produce same config_id
    assert config_id1 == config_id2
    assert isinstance(config_id1, str)
    assert len(config_id1) > 0


def test_belief_config_id_different_values_produces_different_hash():
    """
    Purpose: Ensures belief objects with different values generate unique config_id hashes

    Given: Two TestBelief instances with different values (42 and 43)
    When: Config IDs are computed for both belief objects
    Then: The beliefs produce different config_id values to prevent cache collisions

    Test type: unit
    """

    class TestBelief(Belief):
        def __init__(self, value):
            self.value = value

        def update(self, action, observation, pomdp, state=None):
            return self

        def sample(self):
            return self.value

    # ARRANGE: Create belief instances with different values
    value1, value2 = 42, 43
    belief1 = TestBelief(value=value1)
    belief2 = TestBelief(value=value2)

    # ACT: Generate config IDs for different beliefs
    config_id1 = belief1.config_id
    config_id2 = belief2.config_id

    # ASSERT: Verify different beliefs produce different config_ids
    assert config_id1 != config_id2
    assert isinstance(config_id1, str)
    assert isinstance(config_id2, str)


def test_belief_config_id_numpy_arrays_handles_identical_content():
    """
    Purpose: Verifies config_id generation works correctly with numpy array belief values

    Given: Two TestBelief instances containing identical numpy arrays [1, 2, 3]
    When: Config IDs are generated for both numpy-based beliefs
    Then: Both beliefs produce identical config_id values despite being separate array objects

    Test type: unit
    """

    class TestBelief(Belief):
        def __init__(self, value):
            self.value = value

        def update(self, action, observation, pomdp, state=None):
            return self

        def sample(self):
            return self.value

    # ARRANGE: Create beliefs with identical numpy array content
    array_content = [1, 2, 3]
    belief1 = TestBelief(value=np.array(array_content))
    belief2 = TestBelief(value=np.array(array_content))

    # ACT: Generate config IDs for numpy array beliefs
    config_id1 = belief1.config_id
    config_id2 = belief2.config_id

    # ASSERT: Verify identical array content produces same config_id
    assert config_id1 == config_id2
    assert not np.array_equal(belief1.value is belief2.value, True)  # Different objects
    assert np.array_equal(belief1.value, belief2.value)  # Same content


def test_belief_objects_usable_as_dictionary_keys_and_set_members():
    """
    Purpose: Validates belief objects can be used as dictionary keys and set members for caching

    Given: Two identical TestBelief instances with same value (42)
    When: Beliefs are used in sets and as dictionary keys
    Then: Set deduplicates identical beliefs and dictionary access works with both objects

    Test type: unit
    """

    class TestBelief(Belief):
        def __init__(self, value):
            self.value = value

        def update(self, action, observation, pomdp, state=None):
            return self

        def sample(self):
            return self.value

    # ARRANGE: Create identical belief instances for hashability testing

    test_value = 42
    belief1 = TestBelief(value=test_value)
    belief2 = TestBelief(value=test_value)

    # ACT: Use beliefs in set and dictionary operations
    belief_set = {belief1, belief2}
    belief_dict = {belief1: "cached_value"}
    dict_access_result = belief_dict[belief2]

    # ASSERT: Verify hashable behavior works correctly
    assert len(belief_set) == 1  # Identical beliefs deduplicated in set
    assert dict_access_result == "cached_value"  # Dictionary access works with equivalent belief
    assert belief1 in belief_set
    assert belief2 in belief_set


def test_belief_equality():
    """Test belief equality."""

    class TestBelief(Belief):
        def __init__(self, value):
            self.value = value

        def update(self, action, observation, pomdp, state=None):
            return self

        def sample(self):
            return self.value

    # Create two identical beliefs
    belief1 = TestBelief(value=42)
    belief2 = TestBelief(value=42)

    # Test equality
    assert belief1 == belief2
    assert belief2 == belief1

    # Test inequality with different values
    belief3 = TestBelief(value=43)
    assert belief1 != belief3


def test_belief_equality_with_different_types():
    """Test equality comparison with different types."""

    class TestBelief(Belief):
        def __init__(self, value):
            self.value = value

        def update(self, action, observation, pomdp, state=None):
            return self

        def sample(self):
            return self.value

    belief = TestBelief(value=42)

    # Should return NotImplemented for non-Belief types
    assert belief.__eq__(42) == NotImplemented
    assert belief.__eq__("not a belief") == NotImplemented


def test_create_belief_from_config_basic():
    config = BeliefConfig(
        class_name="WeightedParticleBelief",
        params={"n_particles": 5, "resampling": True, "ess_factor": 0.5},
    )
    env = TigerPOMDP(discount_factor=0.95)
    belief = create_belief(env, config)
    assert isinstance(belief, WeightedParticleBelief)
    assert len(belief.particles) == 5
    assert np.isclose(np.sum(np.exp(belief.log_weights - np.max(belief.log_weights))), 5)
    assert all(p in env.states for p in belief.particles)
    assert belief.resampling is True
    assert np.isclose(
        belief.ess_threshold, 2.5
    )  # ess_threshold = len(particles) * ess_factor = 5 * 0.5


def test_create_belief_particles_and_weights():
    config = BeliefConfig(
        class_name="WeightedParticleBelief",
        params={"n_particles": 3, "resampling": False, "ess_factor": 0.1},
    )
    env = TigerPOMDP(discount_factor=0.95)
    belief = create_belief(env, config)
    assert len(belief.particles) == 3
    assert all(p in env.states for p in belief.particles)
    assert np.allclose(np.exp(belief.log_weights - np.max(belief.log_weights)), np.ones(3))
    assert belief.resampling is False
    assert np.isclose(
        belief.ess_threshold, 0.3
    )  # ess_threshold = len(particles) * ess_factor = 3 * 0.1


def test_reinvigoration_discrete_light_dark():
    config = BeliefConfig(
        class_name="WeightedParticleBeliefDiscreteLightDark",
        params={
            "n_particles": 5,
            "resampling": True,
            "ess_factor": 0.5,
            "reinvigoration_fraction": 0.2,
        },
    )
    env = TigerPOMDP(discount_factor=0.95)
    belief = create_belief(env, config)
    assert isinstance(belief, WeightedParticleBeliefDiscreteLightDark)
    # Call reinvigorate with dummy action and observation
    reinvigorated = belief.reinvigorate(
        action="up", observation=np.array([0, 0]), pomdp=env, belief=belief
    )
    assert reinvigorated is belief


def test_reinvigoration_discrete_light_dark_full_coverage():
    config = BeliefConfig(
        class_name="WeightedParticleBeliefDiscreteLightDarkFullCoverage",
        params={"n_particles": 5, "ess_factor": 0.5, "reinvigoration_fraction": 0.05},
    )
    env = TigerPOMDP(discount_factor=0.95)
    belief = create_belief(env, config)
    assert isinstance(belief, WeightedParticleBeliefDiscreteLightDarkFullCoverage)
    # Call reinvigorate with dummy action and observation
    reinvigorated = belief.reinvigorate(
        action="up", observation=np.array([0, 0]), pomdp=env, belief=belief
    )
    assert reinvigorated is belief


def test_reinvigoration_continuous_light_dark_full_coverage():
    config = BeliefConfig(
        class_name="WeightedParticleBeliefContinuousLightDarkFullCoverage",
        params={
            "n_particles": 5,
            "ess_factor": 0.5,
            "reinvigoration_fraction": 0.05,
            "reinvigoration_cov_matrix": np.eye(2),
        },
    )
    env = TigerPOMDP(discount_factor=0.95)
    belief = create_belief(env, config)
    assert isinstance(belief, WeightedParticleBeliefContinuousLightDarkFullCoverage)
    # Call reinvigorate with dummy action and observation
    reinvigorated = belief.reinvigorate(
        action="up", observation=np.array([0, 0]), pomdp=env, belief=belief
    )
    assert reinvigorated is belief


def test_reinvigoration_sanity_pomdp():
    config = BeliefConfig(
        class_name="WeightedParticleBeliefSanityPOMDP",
        params={
            "n_particles": 5,
            "resampling": True,
            "ess_factor": 0.5,
            "reinvigoration_fraction": 0.2,
        },
    )
    env = TigerPOMDP(discount_factor=0.95)
    belief = create_belief(env, config)
    assert isinstance(belief, WeightedParticleBeliefSanityPOMDP)
    # Call reinvigorate with dummy action and observation
    reinvigorated = belief.reinvigorate(
        action="up", observation=np.array([0, 0]), pomdp=env, belief=belief
    )
    assert isinstance(reinvigorated, WeightedParticleBelief)
