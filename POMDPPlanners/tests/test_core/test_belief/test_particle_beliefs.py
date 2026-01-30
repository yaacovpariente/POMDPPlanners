"""Tests for particle belief implementations.

This module tests the particle belief implementations, focusing on:
- WeightedParticleBelief config_id, equality, update, sampling
- WeightedParticleBeliefStateUpdate initialization, update, inplace_update, sampling, config_id
- UnweightedParticleBeliefStateUpdate initialization, update, inplace_update, sampling, config_id
- get_unique_support function
- Usage example tests for all particle belief classes
"""

import collections
import random
import time

import numpy as np
import pytest

from POMDPPlanners.core.belief import (
    Belief,
    UnweightedParticleBeliefStateUpdate,
    WeightedParticleBelief,
    WeightedParticleBeliefStateUpdate,
    get_unique_support,
)
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


def test_weighted_particle_belief_config_id_deterministic():
    # Create two identical beliefs
    particles1 = [1, 2, 3]
    log_weights1 = np.array([0.1, 0.2, 0.3])
    belief1 = WeightedParticleBelief(particles=particles1, log_weights=log_weights1)

    particles2 = [1, 2, 3]
    log_weights2 = np.array([0.1, 0.2, 0.3])
    belief2 = WeightedParticleBelief(particles=particles2, log_weights=log_weights2)

    # Config IDs should be identical for identical beliefs
    assert belief1.config_id == belief2.config_id


def test_weighted_particle_belief_config_id_changes_with_particles():
    # Create two beliefs with different particles
    particles1 = [1, 2, 3]
    log_weights1 = np.array([0.1, 0.2, 0.3])
    belief1 = WeightedParticleBelief(particles=particles1, log_weights=log_weights1)

    particles2 = [1, 2, 4]  # Different particle
    log_weights2 = np.array([0.1, 0.2, 0.3])
    belief2 = WeightedParticleBelief(particles=particles2, log_weights=log_weights2)

    # Config IDs should be different
    assert belief1.config_id != belief2.config_id


def test_weighted_particle_belief_config_id_changes_with_weights():
    # Create two beliefs with different weights
    particles1 = [1, 2, 3]
    log_weights1 = np.array([0.1, 0.2, 0.3])
    belief1 = WeightedParticleBelief(particles=particles1, log_weights=log_weights1)

    particles2 = [1, 2, 3]
    log_weights2 = np.array([0.1, 0.2, 0.4])  # Different weight
    belief2 = WeightedParticleBelief(particles=particles2, log_weights=log_weights2)

    # Config IDs should be different
    assert belief1.config_id != belief2.config_id


def test_weighted_particle_belief_config_id_with_numpy_particles():
    # Test with numpy array particles
    particles1 = [np.array([1, 2]), np.array([3, 4])]
    log_weights1 = np.array([0.1, 0.2])
    belief1 = WeightedParticleBelief(particles=particles1, log_weights=log_weights1)

    particles2 = [np.array([1, 2]), np.array([3, 4])]
    log_weights2 = np.array([0.1, 0.2])
    belief2 = WeightedParticleBelief(particles=particles2, log_weights=log_weights2)

    # Config IDs should be identical for identical numpy array particles
    assert belief1.config_id == belief2.config_id


def test_weighted_particle_belief_config_id_with_mixed_particles():
    # Test with mixed type particles
    particles1 = [1, np.array([2, 3]), "test"]
    log_weights1 = np.array([0.1, 0.2, 0.3])
    belief1 = WeightedParticleBelief(particles=particles1, log_weights=log_weights1)

    particles2 = [1, np.array([2, 3]), "test"]
    log_weights2 = np.array([0.1, 0.2, 0.3])
    belief2 = WeightedParticleBelief(particles=particles2, log_weights=log_weights2)

    # Config IDs should be identical for identical mixed type particles
    assert belief1.config_id == belief2.config_id


def test_weighted_particle_belief_config_id_with_resampling():
    # Test that resampling parameter affects config_id
    particles = [1, 2, 3]
    log_weights = np.array([0.1, 0.2, 0.3])

    belief1 = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=True)
    belief2 = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)

    # Config IDs should be different due to different resampling settings
    assert belief1.config_id != belief2.config_id


def test_weighted_particle_belief_config_id_with_different_number_order():
    # Test with number particles in different order
    particles1 = [1, 2, 3]
    log_weights1 = np.array([0.1, 0.2, 0.3])
    belief1 = WeightedParticleBelief(particles=particles1, log_weights=log_weights1)

    # Same particles and weights but in different order
    particles2 = [3, 1, 2]
    log_weights2 = np.array([0.3, 0.1, 0.2])  # Weights reordered to match particles
    belief2 = WeightedParticleBelief(particles=particles2, log_weights=log_weights2)

    # Config IDs should be identical since they represent the same belief
    assert belief1.config_id == belief2.config_id


def test_weighted_particle_belief_config_id_with_different_numpy_order():
    # Test with numpy array particles in different order
    particles1 = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    log_weights1 = np.array([0.1, 0.2, 0.3])
    belief1 = WeightedParticleBelief(particles=particles1, log_weights=log_weights1)

    # Same particles and weights but in different order
    particles2 = [np.array([5, 6]), np.array([1, 2]), np.array([3, 4])]
    log_weights2 = np.array([0.3, 0.1, 0.2])  # Weights reordered to match particles
    belief2 = WeightedParticleBelief(particles=particles2, log_weights=log_weights2)

    # Config IDs should be identical since they represent the same belief
    assert belief1.config_id == belief2.config_id


def test_weighted_particle_belief_hashable():
    # Create two identical beliefs
    particles1 = [1, 2, 3]
    log_weights1 = np.array([0.1, 0.2, 0.3])
    belief1 = WeightedParticleBelief(particles=particles1, log_weights=log_weights1)

    particles2 = [1, 2, 3]
    log_weights2 = np.array([0.1, 0.2, 0.3])
    belief2 = WeightedParticleBelief(particles=particles2, log_weights=log_weights2)

    # Test that beliefs can be used in a set
    belief_set = {belief1, belief2}
    assert len(belief_set) == 1  # Should only have one unique belief

    # Test that beliefs can be used as dictionary keys
    belief_dict = {belief1: "value1"}
    assert belief_dict[belief2] == "value1"  # Should be able to access using belief2


def test_weighted_particle_belief_equality():
    # Create two identical beliefs
    particles1 = [1, 2, 3]
    log_weights1 = np.array([0.1, 0.2, 0.3])
    belief1 = WeightedParticleBelief(particles=particles1, log_weights=log_weights1)

    particles2 = [1, 2, 3]
    log_weights2 = np.array([0.1, 0.2, 0.3])
    belief2 = WeightedParticleBelief(particles=particles2, log_weights=log_weights2)

    # Test equality
    assert belief1 == belief2
    assert belief2 == belief1

    # Test inequality with different particles
    particles3 = [1, 2, 4]
    belief3 = WeightedParticleBelief(particles=particles3, log_weights=log_weights1)
    assert belief1 != belief3

    # Test inequality with different weights
    log_weights3 = np.array([0.1, 0.2, 0.4])
    belief4 = WeightedParticleBelief(particles=particles1, log_weights=log_weights3)
    assert belief1 != belief4


def test_weighted_particle_belief_equality_with_different_order():
    # Test equality with particles and weights in different order
    particles1 = [1, 2, 3]
    log_weights1 = np.array([0.1, 0.2, 0.3])
    belief1 = WeightedParticleBelief(particles=particles1, log_weights=log_weights1)

    # Same particles and weights but in different order
    particles2 = [3, 1, 2]
    log_weights2 = np.array([0.3, 0.1, 0.2])  # Weights reordered to match particles
    belief2 = WeightedParticleBelief(particles=particles2, log_weights=log_weights2)

    # Should be equal since they represent the same belief
    assert belief1 == belief2
    assert belief2 == belief1


def test_weighted_particle_belief_equality_with_numpy_particles():
    # Test equality with numpy array particles
    particles1 = [np.array([1, 2]), np.array([3, 4])]
    log_weights1 = np.array([0.1, 0.2])
    belief1 = WeightedParticleBelief(particles=particles1, log_weights=log_weights1)

    particles2 = [np.array([1, 2]), np.array([3, 4])]
    log_weights2 = np.array([0.1, 0.2])
    belief2 = WeightedParticleBelief(particles=particles2, log_weights=log_weights2)

    # Should be equal
    assert belief1 == belief2
    assert belief2 == belief1

    # Test inequality with different numpy arrays
    particles3 = [np.array([1, 2]), np.array([3, 5])]  # Different value
    belief3 = WeightedParticleBelief(particles=particles3, log_weights=log_weights1)
    assert belief1 != belief3


def test_weighted_particle_belief_equality_with_resampling():
    # Test that resampling parameter affects equality
    particles = [1, 2, 3]
    log_weights = np.array([0.1, 0.2, 0.3])

    belief1 = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=True)
    belief2 = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)

    # Should not be equal due to different resampling settings
    assert belief1 != belief2


def test_weighted_particle_belief_equality_with_different_types():
    # Test equality comparison with different types
    particles = [1, 2, 3]
    log_weights = np.array([0.1, 0.2, 0.3])
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    # Should return NotImplemented for non-Belief types
    assert belief.__eq__(42) == NotImplemented
    assert belief.__eq__("not a belief") == NotImplemented


def test_to_DiscreteDistribution_basic():
    """Test basic conversion to DiscreteDistribution with simple particles."""
    particles = [1, 2, 3]
    log_weights = np.array([0.1, 0.2, 0.3])
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    distribution = belief.to_unique_support_distribution()

    # Check that we have the same number of unique particles
    assert len(distribution.values) == 3
    # Check that weights sum to 1
    assert np.isclose(np.sum(distribution.probs), 1.0)
    # Check that all particles are present
    assert set(distribution.values) == set(particles)


def test_to_DiscreteDistribution_duplicate_particles():
    """Test conversion with duplicate particles that should be combined."""
    particles = [1, 1, 2, 2, 2, 3]  # Duplicate particles
    log_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    distribution = belief.to_unique_support_distribution()

    # Check that we have only unique particles
    assert len(distribution.values) == 3
    # Check that weights sum to 1
    assert np.isclose(np.sum(distribution.probs), 1.0)
    # Check that all unique particles are present
    assert set(distribution.values) == {1, 2, 3}

    # Find the weights for each particle
    particle_weights = {}
    for value, prob in zip(distribution.values, distribution.probs):
        particle_weights[value] = prob

    # Check that weights for duplicate particles are combined correctly
    # The weights should be proportional to the sum of their log weights
    assert particle_weights[1] > 0  # Combined weight of 0.1 and 0.2
    assert particle_weights[2] > 0  # Combined weight of 0.3, 0.4, and 0.5
    assert particle_weights[3] > 0  # Weight of 0.6


def test_to_DiscreteDistribution_numpy_particles():
    """Test conversion with numpy array particles."""
    particles = [np.array([1, 2]), np.array([1, 2]), np.array([3, 4])]  # Duplicate
    log_weights = np.array([0.1, 0.2, 0.3])
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    distribution = belief.to_unique_support_distribution()

    # Check that we have only unique particles
    assert len(distribution.values) == 2
    # Check that weights sum to 1
    assert np.isclose(np.sum(distribution.probs), 1.0)

    # Check that numpy arrays are preserved and duplicates are combined
    found_12 = False
    found_34 = False
    for value, prob in zip(distribution.values, distribution.probs):
        if np.array_equal(value, np.array([1, 2])):
            found_12 = True
            assert prob > 0  # Combined weight of 0.1 and 0.2
        elif np.array_equal(value, np.array([3, 4])):
            found_34 = True
            assert prob > 0  # Weight of 0.3

    assert found_12 and found_34


def test_to_DiscreteDistribution_mixed_particles():
    """Test conversion with mixed type particles."""
    particles = [
        1,
        "test",
        np.array([1, 2]),
        1,  # Duplicate
        "test",  # Duplicate
        np.array([1, 2]),  # Duplicate
    ]
    log_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    distribution = belief.to_unique_support_distribution()

    # Check that we have only unique particles
    assert len(distribution.values) == 3
    # Check that weights sum to 1
    assert np.isclose(np.sum(distribution.probs), 1.0)

    # Check that all unique particles are present with correct types
    found_int = False
    found_str = False
    found_array = False

    for value, prob in zip(distribution.values, distribution.probs):
        if isinstance(value, int) and value == 1:
            found_int = True
            assert prob > 0  # Combined weight of 0.1 and 0.4
        elif isinstance(value, str) and value == "test":
            found_str = True
            assert prob > 0  # Combined weight of 0.2 and 0.5
        elif isinstance(value, np.ndarray) and np.array_equal(value, np.array([1, 2])):
            found_array = True
            assert prob > 0  # Combined weight of 0.3 and 0.6

    assert found_int and found_str and found_array


def test_to_DiscreteDistribution_weight_normalization():
    """Test that weights are properly normalized in the output distribution."""
    particles = [1, 2, 3]
    log_weights = np.array([1.0, 2.0, 3.0])  # Large log weights
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    distribution = belief.to_unique_support_distribution()

    # Check that weights sum to 1
    assert np.isclose(np.sum(distribution.probs), 1.0)
    # Check that all weights are positive
    assert np.all(distribution.probs > 0)
    # Check that weights are properly normalized (larger log weight = larger probability)
    assert distribution.probs[2] > distribution.probs[1] > distribution.probs[0]


# Tests for get_unique_support function
def test_get_unique_support_basic():
    """Test basic get_unique_support functionality with simple particles."""
    particles = [1, 2, 3]
    probabilities = np.array([0.2, 0.3, 0.5])

    unique_particles, unique_probs = get_unique_support(particles, probabilities)

    # Check that we have the same number of unique particles
    assert len(unique_particles) == 3
    # Check that probabilities sum to 1
    assert np.isclose(np.sum(unique_probs), 1.0)
    # Check that all particles are present
    assert set(unique_particles) == set(particles)
    # Check that probabilities match (may be reordered)
    assert np.allclose(sorted(unique_probs), sorted(probabilities))


def test_get_unique_support_duplicate_particles():
    """Test get_unique_support with duplicate particles that should be combined."""
    particles = [1, 1, 2, 2, 2, 3]  # Duplicate particles
    probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    unique_particles, unique_probs = get_unique_support(particles, probabilities)

    # Check that we have only unique particles
    assert len(unique_particles) == 3
    # Check that probabilities sum to 1
    assert np.isclose(np.sum(unique_probs), 1.0)
    # Check that all unique particles are present
    assert set(unique_particles) == {1, 2, 3}

    # Find the probabilities for each particle
    particle_probs = {}
    for value, prob in zip(unique_particles, unique_probs):
        particle_probs[value] = prob

    # Check that probabilities for duplicate particles are combined correctly
    # Particle 1: 0.1 + 0.2 = 0.3, normalized
    # Particle 2: 0.3 + 0.4 + 0.5 = 1.2, normalized
    # Particle 3: 0.6, normalized
    # Total sum before normalization: 0.3 + 1.2 + 0.6 = 2.1
    assert np.isclose(particle_probs[1], 0.3 / 2.1)
    assert np.isclose(particle_probs[2], 1.2 / 2.1)
    assert np.isclose(particle_probs[3], 0.6 / 2.1)


def test_get_unique_support_numpy_particles():
    """Test get_unique_support with numpy array particles."""
    particles = [np.array([1, 2]), np.array([1, 2]), np.array([3, 4])]  # Duplicate
    probabilities = np.array([0.1, 0.2, 0.3])

    unique_particles, unique_probs = get_unique_support(particles, probabilities)

    # Check that we have only unique particles
    assert len(unique_particles) == 2
    # Check that probabilities sum to 1
    assert np.isclose(np.sum(unique_probs), 1.0)

    # Check that numpy arrays are preserved and duplicates are combined
    found_12 = False
    found_34 = False
    for value, prob in zip(unique_particles, unique_probs):
        if np.array_equal(value, np.array([1, 2])):
            found_12 = True
            assert np.isclose(prob, (0.1 + 0.2) / 0.6)  # Combined and normalized
        elif np.array_equal(value, np.array([3, 4])):
            found_34 = True
            assert np.isclose(prob, 0.3 / 0.6)  # Normalized

    assert found_12 and found_34


def test_get_unique_support_mixed_particles():
    """Test get_unique_support with mixed type particles."""
    particles = [
        1,
        "test",
        np.array([1, 2]),
        1,  # Duplicate
        "test",  # Duplicate
        np.array([1, 2]),  # Duplicate
    ]
    probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    unique_particles, unique_probs = get_unique_support(particles, probabilities)

    # Check that we have only unique particles
    assert len(unique_particles) == 3
    # Check that probabilities sum to 1
    assert np.isclose(np.sum(unique_probs), 1.0)

    # Check that all unique particles are present with correct types
    found_int = False
    found_str = False
    found_array = False

    for value, prob in zip(unique_particles, unique_probs):
        if isinstance(value, int) and value == 1:
            found_int = True
            assert np.isclose(prob, (0.1 + 0.4) / 2.1)  # Combined and normalized
        elif isinstance(value, str) and value == "test":
            found_str = True
            assert np.isclose(prob, (0.2 + 0.5) / 2.1)  # Combined and normalized
        elif isinstance(value, np.ndarray) and np.array_equal(value, np.array([1, 2])):
            found_array = True
            assert np.isclose(prob, (0.3 + 0.6) / 2.1)  # Combined and normalized

    assert found_int and found_str and found_array


def test_get_unique_support_normalization():
    """Test that get_unique_support properly normalizes probabilities."""
    particles = [1, 2, 3]
    probabilities = np.array([2.0, 4.0, 2.0])  # Unnormalized (sum = 8.0)

    unique_particles, unique_probs = get_unique_support(particles, probabilities)

    # Check that probabilities sum to 1
    assert np.isclose(np.sum(unique_probs), 1.0)
    # Check that all probabilities are positive
    assert np.all(unique_probs > 0)
    # Check that proportions are preserved: 2:4:2 = 1:2:1
    # After normalization: 0.25, 0.5, 0.25
    expected_probs = np.array([0.25, 0.5, 0.25])
    assert np.allclose(sorted(unique_probs), sorted(expected_probs))


def test_get_unique_support_empty():
    """Test get_unique_support with empty input."""
    particles = []
    probabilities = np.array([])

    unique_particles, unique_probs = get_unique_support(particles, probabilities)

    assert len(unique_particles) == 0
    assert len(unique_probs) == 0
    # Empty arrays should have sum of 0 (no normalization performed)
    assert np.sum(unique_probs) == 0.0


def test_get_unique_support_single_particle():
    """Test get_unique_support with single particle."""
    particles = [42]
    probabilities = np.array([0.5])

    unique_particles, unique_probs = get_unique_support(particles, probabilities)

    assert len(unique_particles) == 1
    assert unique_particles[0] == 42
    assert np.isclose(unique_probs[0], 1.0)  # Normalized to 1.0


def test_get_unique_support_all_duplicates():
    """Test get_unique_support when all particles are duplicates."""
    particles = [1, 1, 1, 1]
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])

    unique_particles, unique_probs = get_unique_support(particles, probabilities)

    assert len(unique_particles) == 1
    assert unique_particles[0] == 1
    assert np.isclose(unique_probs[0], 1.0)  # All combined and normalized


def test_get_unique_support_order_independent():
    """Test that get_unique_support produces same result regardless of input order."""
    particles1 = [1, 2, 1, 3, 2]
    probabilities1 = np.array([0.2, 0.3, 0.1, 0.2, 0.2])

    particles2 = [3, 1, 2, 2, 1]  # Same particles, different order
    probabilities2 = np.array([0.2, 0.2, 0.2, 0.3, 0.1])  # Reordered to match

    unique_particles1, unique_probs1 = get_unique_support(particles1, probabilities1)
    unique_particles2, unique_probs2 = get_unique_support(particles2, probabilities2)

    # Should have same unique particles (order may differ)
    assert set(unique_particles1) == set(unique_particles2)

    # Probabilities should match when sorted by particle
    particle_probs1 = {p: prob for p, prob in zip(unique_particles1, unique_probs1)}
    particle_probs2 = {p: prob for p, prob in zip(unique_particles2, unique_probs2)}

    for particle in particle_probs1:
        assert np.isclose(particle_probs1[particle], particle_probs2[particle])


def test_get_unique_support_zero_probabilities():
    """Test get_unique_support with zero probabilities."""
    particles = [1, 2, 3]
    probabilities = np.array([0.0, 0.5, 0.5])

    unique_particles, unique_probs = get_unique_support(particles, probabilities)

    # Zero probability particle should be included (it's in the unique particles)
    # After normalization: 0.0 becomes 0.0, 0.5 and 0.5 become 0.5 each
    # But wait - normalization would divide by sum, so 0.0 + 0.5 + 0.5 = 1.0
    # After normalization by 1.0: [0.0, 0.5, 0.5]
    assert len(unique_particles) == 3  # All particles should be included
    assert np.isclose(np.sum(unique_probs), 1.0)

    # Find probabilities
    particle_probs = {p: prob for p, prob in zip(unique_particles, unique_probs)}
    assert np.isclose(particle_probs[1], 0.0)
    assert np.isclose(particle_probs[2], 0.5)
    assert np.isclose(particle_probs[3], 0.5)


def test_get_unique_support_very_small_probabilities():
    """Test get_unique_support with very small probability values."""
    particles = [1, 2, 3]
    probabilities = np.array([1e-10, 0.5, 0.5 - 1e-10])

    unique_particles, unique_probs = get_unique_support(particles, probabilities)

    assert len(unique_particles) == 3
    assert np.isclose(np.sum(unique_probs), 1.0)


def test_get_unique_support_complex_numpy_arrays():
    """Test get_unique_support with multi-dimensional numpy arrays."""
    particles = [
        np.array([[1, 2], [3, 4]]),
        np.array([[1, 2], [3, 4]]),  # Duplicate
        np.array([[5, 6], [7, 8]]),
    ]
    probabilities = np.array([0.3, 0.2, 0.5])

    unique_particles, unique_probs = get_unique_support(particles, probabilities)

    assert len(unique_particles) == 2
    assert np.isclose(np.sum(unique_probs), 1.0)

    # Check that duplicates are combined
    found_1234 = False
    found_5678 = False
    for value, prob in zip(unique_particles, unique_probs):
        if np.array_equal(value, np.array([[1, 2], [3, 4]])):
            found_1234 = True
            assert np.isclose(prob, (0.3 + 0.2) / 1.0)  # Combined and normalized
        elif np.array_equal(value, np.array([[5, 6], [7, 8]])):
            found_5678 = True
            assert np.isclose(prob, 0.5 / 1.0)

    assert found_1234 and found_5678


def test_get_unique_support_return_types():
    """Test that get_unique_support returns correct types."""
    particles = [1, 2, 3]
    probabilities = np.array([0.3, 0.3, 0.4])

    unique_particles, unique_probs = get_unique_support(particles, probabilities)

    # Check return types
    assert isinstance(unique_particles, list)
    assert isinstance(unique_probs, np.ndarray)
    assert unique_probs.dtype == float
    assert len(unique_particles) == len(unique_probs)


def test_get_unique_support_preserves_original_types():
    """Test that get_unique_support preserves original particle types."""
    particles = [
        "string_particle",
        42,  # int
        3.14,  # float
        np.array([1, 2]),  # numpy array
        ("tuple", "particle"),  # tuple
    ]
    probabilities = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    unique_particles, unique_probs = get_unique_support(particles, probabilities)

    # Check that types are preserved
    assert len(unique_particles) == 5
    assert any(isinstance(p, str) and p == "string_particle" for p in unique_particles)
    assert any(isinstance(p, int) and p == 42 for p in unique_particles)
    assert any(isinstance(p, float) and p == 3.14 for p in unique_particles)
    assert any(
        isinstance(p, np.ndarray) and np.array_equal(p, np.array([1, 2])) for p in unique_particles
    )
    assert any(isinstance(p, tuple) and p == ("tuple", "particle") for p in unique_particles)


# Tests for WeightedParticleBelief.update() method
def test_belief_update_basic():
    """Test basic belief update functionality."""
    # Create a simple environment and belief
    env = SanityPOMDP()
    particles = [0, 1, 0, 1]  # Mix of states
    log_weights = np.array([0.1, 0.1, 0.1, 0.1])  # Non-zero weights
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)

    # Perform an update
    action = 0
    observation = 0
    updated_belief = belief.update(action=action, observation=observation, pomdp=env)

    # Check that we get a new belief object
    assert updated_belief is not belief
    assert isinstance(updated_belief, WeightedParticleBelief)

    # Check that we have the same number of particles
    assert len(updated_belief.particles) == len(belief.particles)

    # Check that weights are still valid (not all zero)
    assert np.any(updated_belief.log_weights > -np.inf)


def test_belief_update_with_resampling():
    """Test belief update with resampling enabled."""
    env = SanityPOMDP()
    particles = [0, 1, 0, 1]
    log_weights = np.array([0.1, 0.1, 0.1, 0.1])  # Non-zero weights
    belief = WeightedParticleBelief(
        particles=particles, log_weights=log_weights, resampling=True, ess_factor=0.5
    )

    action = 0
    observation = 0
    updated_belief = belief.update(action=action, observation=observation, pomdp=env)

    # Check that we get a new belief object
    assert updated_belief is not belief
    assert isinstance(updated_belief, WeightedParticleBelief)

    # Check that we have the same number of particles
    assert len(updated_belief.particles) == len(belief.particles)

    # Check that weights are normalized after resampling
    normalized_weights = np.exp(updated_belief.log_weights - np.max(updated_belief.log_weights))
    assert np.isclose(np.sum(normalized_weights), len(updated_belief.particles))


def test_belief_update_state_transitions():
    """Test that belief update correctly handles state transitions."""

    env = SanityPOMDP()
    particles = [0, 0, 0]  # All particles start in state 0
    log_weights = np.array([0.1, 0.1, 0.1])  # Non-zero weights
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)

    action = 0
    observation = 0
    updated_belief = belief.update(action=action, observation=observation, pomdp=env)

    # Check that particles have been updated (state transitions occurred)
    # In SanityPOMDP, action 0 from state 0 should transition to state 0
    assert all(p == 0 for p in updated_belief.particles)


def test_belief_update_observation_probabilities():
    """Test that belief update correctly computes observation probabilities."""

    env = SanityPOMDP()
    particles = [0, 1, 0, 1]
    log_weights = np.array([0.1, 0.1, 0.1, 0.1])  # Non-zero weights
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)

    action = 0
    observation = 0
    updated_belief = belief.update(action=action, observation=observation, pomdp=env)

    # Check that weights have been updated based on observation probabilities
    # The weights should reflect how likely each particle's next state is to produce the observation
    assert not np.array_equal(updated_belief.log_weights, belief.log_weights)


def test_belief_update_with_tiger_pomdp():
    """Test belief update with TigerPOMDP environment."""
    env = TigerPOMDP(discount_factor=0.95)
    particles = ["tiger_left", "tiger_right", "tiger_left"]
    log_weights = np.array([0.1, 0.1, 0.1])  # Non-zero weights
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)

    action = "listen"
    observation = "hear_left"
    updated_belief = belief.update(action=action, observation=observation, pomdp=env)

    # Check that we get a valid updated belief
    assert isinstance(updated_belief, WeightedParticleBelief)
    assert len(updated_belief.particles) == len(belief.particles)

    # Check that weights have been updated
    assert not np.array_equal(updated_belief.log_weights, belief.log_weights)


def test_belief_update_preserves_particle_count():
    """Test that belief update preserves the number of particles."""

    env = SanityPOMDP()
    n_particles = 10
    particles = [0] * n_particles
    log_weights = np.ones(n_particles) * 0.1  # Non-zero weights
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)

    action = 0
    observation = 0
    updated_belief = belief.update(action=action, observation=observation, pomdp=env)

    assert len(updated_belief.particles) == n_particles


def test_belief_update_weight_normalization():
    """Test that belief update properly handles weight normalization."""

    env = SanityPOMDP()
    particles = [0, 1, 0, 1]
    log_weights = np.array([1.0, 2.0, 3.0, 4.0])  # Unequal weights
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)

    action = 0
    observation = 0
    updated_belief = belief.update(action=action, observation=observation, pomdp=env)

    # Check that weights are finite
    assert np.all(np.isfinite(updated_belief.log_weights))

    # Check that at least some weights are different from original
    assert not np.array_equal(updated_belief.log_weights, belief.log_weights)


def test_belief_update_with_extreme_weights():
    """Test belief update with extreme weight values."""

    env = SanityPOMDP()
    particles = [0, 1, 0, 1]
    log_weights = np.array([-1000.0, -1000.0, -1000.0, -1000.0])  # Very small weights
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)

    action = 0
    observation = 0
    updated_belief = belief.update(action=action, observation=observation, pomdp=env)

    # Check that update doesn't crash with extreme weights
    assert isinstance(updated_belief, WeightedParticleBelief)
    assert len(updated_belief.particles) == len(belief.particles)


def test_belief_update_consistency():
    """Test that belief update produces consistent results."""

    env = SanityPOMDP()
    particles = [0, 1, 0, 1]
    log_weights = np.array([0.1, 0.1, 0.1, 0.1])  # Non-zero weights
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)

    action = 0
    observation = 0

    # Perform two identical updates
    updated_belief1 = belief.update(action=action, observation=observation, pomdp=env)
    updated_belief2 = belief.update(action=action, observation=observation, pomdp=env)

    # Results should be consistent (same particle count, similar weight structure)
    assert len(updated_belief1.particles) == len(updated_belief2.particles)
    assert len(updated_belief1.log_weights) == len(updated_belief2.log_weights)


def test_belief_update_with_different_actions():
    """Test that belief update behaves differently for different actions."""

    env = SanityPOMDP()
    particles = [0, 1, 0, 1]
    log_weights = np.array([0.1, 0.1, 0.1, 0.1])  # Non-zero weights
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)

    observation = 0

    # Update with action 0
    updated_belief1 = belief.update(action=0, observation=observation, pomdp=env)

    # Update with action 1
    updated_belief2 = belief.update(action=1, observation=observation, pomdp=env)

    # Results should be different for different actions
    # (At least the particles should be different due to different state transitions)
    assert updated_belief1.particles != updated_belief2.particles or not np.array_equal(
        updated_belief1.log_weights, updated_belief2.log_weights
    )


# Tests for WeightedParticleBeliefStateUpdate class
def test_weighted_particle_belief_state_update_initialization_empty():
    """Test WeightedParticleBeliefStateUpdate initialization with empty lists."""
    belief = WeightedParticleBeliefStateUpdate()

    assert belief.particles == []
    assert belief.weights == []
    assert belief.weights_sum == 0


def test_weighted_particle_belief_state_update_mutable_default_arguments():
    """Test that WeightedParticleBeliefStateUpdate doesn't have mutable default argument issues.

    Purpose: Validates that multiple empty belief instances don't share references to the same list objects

    Given: Multiple WeightedParticleBeliefStateUpdate instances created with default empty initialization
    When: Instances are created and their particle/weight lists are checked for reference sharing
    Then: Each instance has its own separate empty lists, preventing mutable default argument issues

    Test type: unit
    """
    belief1 = WeightedParticleBeliefStateUpdate()
    belief2 = WeightedParticleBeliefStateUpdate()

    # Verify that each instance has its own separate list objects
    assert belief1.particles is not belief2.particles, "Particles lists should not share references"
    assert belief1.weights is not belief2.weights, "Weights lists should not share references"

    # Both should still be empty
    assert belief1.particles == []
    assert belief1.weights == []
    assert belief2.particles == []
    assert belief2.weights == []


def test_weighted_particle_belief_state_update_initialization_with_data():
    """Test WeightedParticleBeliefStateUpdate initialization with particles and weights."""
    particles = ["state1", "state2", "state3"]
    weights = [0.3, 0.5, 0.2]

    belief = WeightedParticleBeliefStateUpdate(particles=particles, weights=weights)

    assert belief.particles == particles
    assert belief.weights == weights
    assert belief.weights_sum == sum(weights)


def test_weighted_particle_belief_state_update_initialization_weight_sum():
    """Test that weights_sum is calculated correctly during initialization."""
    particles = [1, 2, 3]
    weights = [0.1, 0.4, 0.5]

    belief = WeightedParticleBeliefStateUpdate(particles=particles, weights=weights)

    assert np.isclose(belief.weights_sum, 1.0)


def test_weighted_particle_belief_state_update_inplace_update():
    """Test inplace_update method adds state and weight correctly."""

    env = SanityPOMDP()
    belief = WeightedParticleBeliefStateUpdate(particles=[0], weights=[0.5])
    initial_sum = belief.weights_sum

    # Add a new state
    new_state = 1
    action = 0
    observation = 1

    belief.inplace_update(action=action, observation=observation, pomdp=env, state=new_state)

    # Check that the state was added
    assert new_state in belief.particles
    assert len(belief.particles) == 2
    assert len(belief.weights) == 2

    # Check that weights_sum was updated
    assert belief.weights_sum > initial_sum
    assert np.isclose(belief.weights_sum, sum(belief.weights))


def test_weighted_particle_belief_state_update_inplace_update_observation_probability():
    """Test that inplace_update correctly computes observation probability."""

    env = SanityPOMDP()
    belief = WeightedParticleBeliefStateUpdate()

    state = 0
    action = 0
    observation = 0  # Should have high probability for state 0 in SanityPOMDP

    belief.inplace_update(action=action, observation=observation, pomdp=env, state=state)

    # Check that the observation probability was computed and added
    assert len(belief.weights) == 1
    assert belief.weights[0] > 0  # Should be positive probability

    # For SanityPOMDP, observation equals state, so probability should be 1.0
    assert np.isclose(belief.weights[0], 1.0)


def test_weighted_particle_belief_state_update_multiple_inplace_updates():
    """Test multiple inplace_updates accumulate correctly."""

    env = SanityPOMDP()
    belief = WeightedParticleBeliefStateUpdate(particles=[], weights=[])

    states = [0, 1, 0]
    action = 0
    observation = 0

    for state in states:
        belief.inplace_update(action=action, observation=observation, pomdp=env, state=state)

    assert len(belief.particles) == 3
    assert len(belief.weights) == 3
    assert belief.particles == states
    assert np.isclose(belief.weights_sum, sum(belief.weights))


def test_weighted_particle_belief_state_update_update_returns_new_instance():
    """Test that update method returns a new instance."""

    env = SanityPOMDP()
    original_belief = WeightedParticleBeliefStateUpdate(particles=[0], weights=[0.5])

    action = 0
    observation = 0
    state = 1

    updated_belief = original_belief.update(
        action=action, observation=observation, pomdp=env, state=state
    )

    # Should return a new instance
    assert updated_belief is not original_belief
    assert isinstance(updated_belief, WeightedParticleBeliefStateUpdate)

    # Original belief should remain unchanged
    assert len(original_belief.particles) == 1
    assert len(original_belief.weights) == 1

    # New belief should have the additional state
    assert len(updated_belief.particles) == 2
    assert len(updated_belief.weights) == 2


def test_weighted_particle_belief_state_update_update_preserves_original_data():
    """Test that update method preserves original particles and weights."""

    env = SanityPOMDP()
    original_particles = [0, 1]
    original_weights = [0.3, 0.7]
    original_belief = WeightedParticleBeliefStateUpdate(
        particles=original_particles, weights=original_weights
    )

    action = 0
    observation = 0
    state = 0

    updated_belief = original_belief.update(
        action=action, observation=observation, pomdp=env, state=state
    )

    # Check that original data is preserved in new belief
    assert updated_belief.particles[:2] == original_particles
    assert updated_belief.weights[:2] == original_weights

    # Check that new data is added
    assert updated_belief.particles[2] == state
    assert len(updated_belief.particles) == 3
    assert len(updated_belief.weights) == 3


def test_weighted_particle_belief_state_update_inplace_vs_update_comparison():
    """Test that inplace_update modifies the belief in-place while update returns a new belief."""

    env = SanityPOMDP()

    # Test 1: inplace_update modifies the original belief
    belief1 = WeightedParticleBeliefStateUpdate(particles=[0], weights=[0.5])
    original_particles1 = belief1.particles.copy()
    original_weights1 = belief1.weights.copy()
    original_weights_sum1 = belief1.weights_sum

    # Perform inplace_update with matching observation (state=1, observation=1)
    belief1.inplace_update(action=0, observation=1, pomdp=env, state=1)

    # Verify that the original belief was modified in-place
    assert len(belief1.particles) == 2
    assert len(belief1.weights) == 2
    assert belief1.particles[0] == 0  # Original particle preserved
    assert belief1.particles[1] == 1  # New particle added
    assert belief1.weights[0] == 0.5  # Original weight preserved
    assert belief1.weights[1] > 0  # New weight added (should be 1.0 for matching observation)
    assert belief1.weights_sum > original_weights_sum1  # Weight sum updated

    # Test 2: update returns a new belief without modifying the original
    belief2 = WeightedParticleBeliefStateUpdate(particles=[0], weights=[0.5])
    original_particles2 = belief2.particles.copy()
    original_weights2 = belief2.weights.copy()
    original_weights_sum2 = belief2.weights_sum

    # Perform update with matching observation (state=1, observation=1)
    updated_belief2 = belief2.update(action=0, observation=1, pomdp=env, state=1)

    # Verify that the original belief was NOT modified
    assert len(belief2.particles) == 1  # Original belief unchanged
    assert len(belief2.weights) == 1  # Original belief unchanged
    assert belief2.particles == original_particles2  # Original particles unchanged
    assert belief2.weights == original_weights2  # Original weights unchanged
    assert belief2.weights_sum == original_weights_sum2  # Original weight sum unchanged

    # Verify that the returned belief is a new instance with the additional data
    assert updated_belief2 is not belief2  # Different object
    assert len(updated_belief2.particles) == 2  # New belief has additional particle
    assert len(updated_belief2.weights) == 2  # New belief has additional weight
    assert updated_belief2.particles[0] == 0  # Original particle preserved
    assert updated_belief2.particles[1] == 1  # New particle added
    assert updated_belief2.weights[0] == 0.5  # Original weight preserved
    assert (
        updated_belief2.weights[1] > 0
    )  # New weight added (should be 1.0 for matching observation)
    assert updated_belief2.weights_sum > original_weights_sum2  # Weight sum updated

    # Test 3: Multiple operations to ensure consistency
    belief3 = WeightedParticleBeliefStateUpdate(particles=[], weights=[])

    # Use inplace_update multiple times with matching observations
    belief3.inplace_update(action=0, observation=0, pomdp=env, state=0)
    belief3.inplace_update(action=0, observation=1, pomdp=env, state=1)
    belief3.inplace_update(action=0, observation=0, pomdp=env, state=0)

    # Verify belief3 was modified in-place
    assert len(belief3.particles) == 3
    assert len(belief3.weights) == 3
    assert belief3.particles == [0, 1, 0]

    # Create a new belief using update method
    belief4 = WeightedParticleBeliefStateUpdate(particles=[], weights=[])
    updated_belief4 = belief4.update(action=0, observation=0, pomdp=env, state=0)
    updated_belief4 = updated_belief4.update(action=0, observation=1, pomdp=env, state=1)
    updated_belief4 = updated_belief4.update(action=0, observation=0, pomdp=env, state=0)

    # Verify belief4 was not modified
    assert len(belief4.particles) == 0
    assert len(belief4.weights) == 0

    # Verify updated_belief4 has the expected data
    assert len(updated_belief4.particles) == 3
    assert len(updated_belief4.weights) == 3
    assert updated_belief4.particles == [0, 1, 0]


def test_weighted_particle_belief_state_update_sample_basic():
    """Test basic sampling functionality."""
    particles = ["state1", "state2", "state3"]
    weights = [0.1, 0.8, 0.1]  # state2 should be most likely

    belief = WeightedParticleBeliefStateUpdate(particles=particles, weights=weights)

    # Sample multiple times to test distribution
    samples = [belief.sample() for _ in range(100)]

    # Check that all samples are valid particles
    for sample in samples:
        assert sample in particles

    # state2 should be sampled most frequently due to highest weight
    state2_count = samples.count("state2")
    assert state2_count > 50  # Should be more than 50% due to weight 0.8


def test_weighted_particle_belief_state_update_sample_uniform_weights():
    """Test sampling with uniform weights."""
    particles = [1, 2, 3, 4]
    weights = [0.25, 0.25, 0.25, 0.25]  # Uniform weights

    belief = WeightedParticleBeliefStateUpdate(particles=particles, weights=weights)

    # Sample multiple times
    samples = [belief.sample() for _ in range(100)]

    # Check that all particles appear in samples
    unique_samples = set(samples)
    assert unique_samples == set(particles)

    # Check that distribution is roughly uniform (within reasonable bounds)
    for particle in particles:
        count = samples.count(particle)
        assert 10 <= count <= 40  # Should be roughly 25 ± 15


def test_weighted_particle_belief_state_update_sample_single_particle():
    """Test sampling with a single particle."""
    particles = ["only_state"]
    weights = [1.0]

    belief = WeightedParticleBeliefStateUpdate(particles=particles, weights=weights)

    # Should always return the only particle
    for _ in range(10):
        sample = belief.sample()
        assert sample == "only_state"


def test_weighted_particle_belief_state_update_sample_empty_belief_raises_error():
    """Test that sampling from empty belief raises ValueError."""
    belief = WeightedParticleBeliefStateUpdate([], [])

    with pytest.raises(ValueError, match="Cannot sample from empty or unnormalized belief"):
        belief.sample()


def test_weighted_particle_belief_state_update_sample_zero_weights_raises_error():
    """Test that sampling with zero weights raises ValueError."""
    particles = ["state1", "state2"]
    weights = [0.0, 0.0]  # All zero weights

    belief = WeightedParticleBeliefStateUpdate(particles=particles, weights=weights)

    with pytest.raises(ValueError, match="Cannot sample from empty or unnormalized belief"):
        belief.sample()


def test_weighted_particle_belief_state_update_sample_empty_particles_raises_error():
    """Test that creating belief with mismatched particles and weights raises ValueError."""
    particles = []
    weights = [0.5]  # Non-empty weights but empty particles

    # Should raise ValueError during initialization due to length mismatch
    with pytest.raises(ValueError, match="particles and weights must have the same length"):
        WeightedParticleBeliefStateUpdate(particles=particles, weights=weights)


def test_weighted_particle_belief_state_update_sample_normalization():
    """Test that sampling works correctly with unnormalized weights."""
    particles = ["state1", "state2", "state3"]
    weights = [2.0, 8.0, 2.0]  # Unnormalized (sum = 12), but proportions are 1:4:1

    belief = WeightedParticleBeliefStateUpdate(particles=particles, weights=weights)

    # Sample multiple times
    samples = [belief.sample() for _ in range(120)]

    # Check proportions (state2 should be ~4x more frequent than others)
    state1_count = samples.count("state1")
    state2_count = samples.count("state2")
    state3_count = samples.count("state3")

    # state2 should be most frequent
    assert state2_count > state1_count
    assert state2_count > state3_count

    # Approximate ratio check (allowing for randomness)
    assert 60 <= state2_count <= 100  # Should be roughly 4/6 = 2/3 of samples


def test_weighted_particle_belief_state_update_with_tiger_pomdp():
    """Test WeightedParticleBeliefStateUpdate integration with TigerPOMDP."""
    env = TigerPOMDP(discount_factor=0.95)

    # Start with empty belief
    belief = WeightedParticleBeliefStateUpdate([], [])

    # Add some states
    states = ["tiger_left", "tiger_right", "tiger_left"]
    action = "listen"
    observation = "hear_left"

    for state in states:
        belief.inplace_update(action=action, observation=observation, pomdp=env, state=state)

    # Check that all states were added
    assert len(belief.particles) == 3
    assert belief.particles == states

    # Check that weights reflect observation probabilities
    assert all(w > 0 for w in belief.weights)  # All should be positive

    # Sample should work
    sample = belief.sample()
    assert sample in ["tiger_left", "tiger_right"]


def test_weighted_particle_belief_state_update_with_tiger_pomdp_different_observations():
    """Test WeightedParticleBeliefStateUpdate with different observations in TigerPOMDP."""
    env = TigerPOMDP(discount_factor=0.95)

    # Create two beliefs with different observations
    belief1 = WeightedParticleBeliefStateUpdate([], [])
    belief2 = WeightedParticleBeliefStateUpdate([], [])

    state = "tiger_left"
    action = "listen"

    # Update with observation that matches the state
    belief1.inplace_update(action=action, observation="hear_left", pomdp=env, state=state)

    # Update with observation that doesn't match the state
    belief2.inplace_update(action=action, observation="hear_right", pomdp=env, state=state)

    # The matching observation should have higher probability
    assert belief1.weights[0] > belief2.weights[0]


def test_weighted_particle_belief_state_update_inheritance():
    """Test that WeightedParticleBeliefStateUpdate inherits from Belief."""
    belief = WeightedParticleBeliefStateUpdate()

    assert isinstance(belief, Belief)


def test_weighted_particle_belief_state_update_config_id():
    """Test that WeightedParticleBeliefStateUpdate has a config_id property."""
    particles = [1, 2, 3]
    weights = [0.3, 0.4, 0.3]

    belief = WeightedParticleBeliefStateUpdate(particles=particles, weights=weights)

    # Should have a config_id
    assert hasattr(belief, "config_id")
    assert isinstance(belief.config_id, str)


def test_weighted_particle_belief_state_update_config_id_deterministic():
    """Test that identical beliefs have identical config_ids."""
    particles1 = [1, 2, 3]
    weights1 = [0.3, 0.4, 0.3]
    belief1 = WeightedParticleBeliefStateUpdate(particles=particles1, weights=weights1)

    particles2 = [1, 2, 3]
    weights2 = [0.3, 0.4, 0.3]
    belief2 = WeightedParticleBeliefStateUpdate(particles=particles2, weights=weights2)

    assert belief1.config_id == belief2.config_id


def test_weighted_particle_belief_state_update_config_id_changes_with_particles():
    """Test that config_id changes when particles change."""
    particles1 = [1, 2, 3]
    weights = [0.3, 0.4, 0.3]
    belief1 = WeightedParticleBeliefStateUpdate(particles=particles1, weights=weights)

    particles2 = [1, 2, 4]  # Different particle
    belief2 = WeightedParticleBeliefStateUpdate(particles=particles2, weights=weights)

    assert belief1.config_id != belief2.config_id


def test_weighted_particle_belief_state_update_config_id_changes_with_weights():
    """Test that config_id changes when weights change."""
    particles = [1, 2, 3]
    weights1 = [0.3, 0.4, 0.3]
    belief1 = WeightedParticleBeliefStateUpdate(particles=particles, weights=weights1)

    weights2 = [0.3, 0.5, 0.2]  # Different weights
    belief2 = WeightedParticleBeliefStateUpdate(particles=particles, weights=weights2)

    assert belief1.config_id != belief2.config_id


def test_weighted_particle_belief_state_update_config_id_with_numpy_particles():
    """Test config_id with numpy array particles."""
    particles1 = [np.array([1, 2]), np.array([3, 4])]
    weights1 = [0.3, 0.7]
    belief1 = WeightedParticleBeliefStateUpdate(particles=particles1, weights=weights1)

    particles2 = [np.array([1, 2]), np.array([3, 4])]
    weights2 = [0.3, 0.7]
    belief2 = WeightedParticleBeliefStateUpdate(particles=particles2, weights=weights2)

    # Config IDs should be identical for identical numpy array particles
    assert belief1.config_id == belief2.config_id


def test_weighted_particle_belief_state_update_config_id_with_mixed_particles():
    """Test config_id with mixed type particles."""
    particles1 = [1, np.array([2, 3]), "test"]
    weights1 = [0.3, 0.4, 0.3]
    belief1 = WeightedParticleBeliefStateUpdate(particles=particles1, weights=weights1)

    particles2 = [1, np.array([2, 3]), "test"]
    weights2 = [0.3, 0.4, 0.3]
    belief2 = WeightedParticleBeliefStateUpdate(particles=particles2, weights=weights2)

    # Config IDs should be identical for identical mixed type particles
    assert belief1.config_id == belief2.config_id


def test_weighted_particle_belief_state_update_config_id_with_different_order():
    """Test config_id with particles and weights in different order."""
    particles1 = [1, 2, 3]
    weights1 = [0.3, 0.4, 0.3]
    belief1 = WeightedParticleBeliefStateUpdate(particles=particles1, weights=weights1)

    # Same particles and weights but in different order
    particles2 = [3, 1, 2]
    weights2 = [0.3, 0.3, 0.4]  # Weights reordered to match particles
    belief2 = WeightedParticleBeliefStateUpdate(particles=particles2, weights=weights2)

    # Config IDs should be identical since they represent the same belief
    assert belief1.config_id == belief2.config_id


def test_weighted_particle_belief_state_update_config_id_with_different_numpy_order():
    """Test config_id with numpy array particles in different order."""
    particles1 = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    weights1 = [0.3, 0.4, 0.3]
    belief1 = WeightedParticleBeliefStateUpdate(particles=particles1, weights=weights1)

    # Same particles and weights but in different order
    particles2 = [np.array([5, 6]), np.array([1, 2]), np.array([3, 4])]
    weights2 = [0.3, 0.3, 0.4]  # Weights reordered to match particles
    belief2 = WeightedParticleBeliefStateUpdate(particles=particles2, weights=weights2)

    # Config IDs should be identical since they represent the same belief
    assert belief1.config_id == belief2.config_id


def test_weighted_particle_belief_state_update_config_id_with_empty_belief():
    """Test config_id with empty belief."""
    belief1 = WeightedParticleBeliefStateUpdate(particles=[], weights=[])
    belief2 = WeightedParticleBeliefStateUpdate(particles=[], weights=[])

    # Config IDs should be identical for empty beliefs
    assert belief1.config_id == belief2.config_id


def test_weighted_particle_belief_state_update_config_id_with_single_particle():
    """Test config_id with single particle."""
    particles1 = ["single_state"]
    weights1 = [1.0]
    belief1 = WeightedParticleBeliefStateUpdate(particles=particles1, weights=weights1)

    particles2 = ["single_state"]
    weights2 = [1.0]
    belief2 = WeightedParticleBeliefStateUpdate(particles=particles2, weights=weights2)

    # Config IDs should be identical
    assert belief1.config_id == belief2.config_id


def test_weighted_particle_belief_state_update_config_id_with_duplicate_particles():
    """Test config_id with duplicate particles."""
    particles1 = [1, 1, 2, 2, 2, 3]  # Duplicate particles
    weights1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    belief1 = WeightedParticleBeliefStateUpdate(particles=particles1, weights=weights1)

    particles2 = [1, 1, 2, 2, 2, 3]  # Same duplicates
    weights2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    belief2 = WeightedParticleBeliefStateUpdate(particles=particles2, weights=weights2)

    # Config IDs should be identical for identical duplicate particles
    assert belief1.config_id == belief2.config_id


def test_weighted_particle_belief_state_update_config_id_with_extreme_weights():
    """Test config_id with extreme weight values."""
    particles1 = [1, 2, 3]
    weights1 = [1e-10, 1e10, 0.5]  # Very small, very large, and normal weights
    belief1 = WeightedParticleBeliefStateUpdate(particles=particles1, weights=weights1)

    particles2 = [1, 2, 3]
    weights2 = [1e-10, 1e10, 0.5]
    belief2 = WeightedParticleBeliefStateUpdate(particles=particles2, weights=weights2)

    # Config IDs should be identical for identical extreme weights
    assert belief1.config_id == belief2.config_id


def test_weighted_particle_belief_state_update_config_id_uniqueness():
    """Test that config_id is unique for different beliefs."""
    # Create several different beliefs
    belief1 = WeightedParticleBeliefStateUpdate(particles=[1, 2], weights=[0.5, 0.5])
    belief2 = WeightedParticleBeliefStateUpdate(particles=[1, 3], weights=[0.5, 0.5])
    belief3 = WeightedParticleBeliefStateUpdate(particles=[1, 2], weights=[0.3, 0.7])
    belief4 = WeightedParticleBeliefStateUpdate(particles=[], weights=[])
    belief5 = WeightedParticleBeliefStateUpdate(particles=[np.array([1, 2])], weights=[1.0])

    # All config_ids should be unique
    config_ids = [
        belief1.config_id,
        belief2.config_id,
        belief3.config_id,
        belief4.config_id,
        belief5.config_id,
    ]
    assert len(config_ids) == len(set(config_ids)), "All config_ids should be unique"


def test_weighted_particle_belief_state_update_config_id_consistency():
    """Test that config_id is consistent across multiple calls."""
    particles = [1, 2, 3]
    weights = [0.3, 0.4, 0.3]
    belief = WeightedParticleBeliefStateUpdate(particles=particles, weights=weights)

    # Config_id should be the same on multiple calls
    config_id1 = belief.config_id
    config_id2 = belief.config_id
    config_id3 = belief.config_id

    assert config_id1 == config_id2 == config_id3


def test_weighted_particle_belief_state_update_equality():
    """Test equality comparison between WeightedParticleBeliefStateUpdate instances."""
    particles1 = [1, 2, 3]
    weights1 = [0.3, 0.4, 0.3]
    belief1 = WeightedParticleBeliefStateUpdate(particles=particles1, weights=weights1)

    particles2 = [1, 2, 3]
    weights2 = [0.3, 0.4, 0.3]
    belief2 = WeightedParticleBeliefStateUpdate(particles=particles2, weights=weights2)

    # Should be equal
    assert belief1 == belief2
    assert belief2 == belief1


def test_weighted_particle_belief_state_update_inequality():
    """Test inequality comparison between WeightedParticleBeliefStateUpdate instances."""
    particles1 = [1, 2, 3]
    weights1 = [0.3, 0.4, 0.3]
    belief1 = WeightedParticleBeliefStateUpdate(particles=particles1, weights=weights1)

    # Different particles
    particles2 = [1, 2, 4]
    belief2 = WeightedParticleBeliefStateUpdate(particles=particles2, weights=weights1)

    # Different weights
    weights3 = [0.3, 0.5, 0.2]
    belief3 = WeightedParticleBeliefStateUpdate(particles=particles1, weights=weights3)

    assert belief1 != belief2
    assert belief1 != belief3


def test_weighted_particle_belief_state_update_hashable():
    """Test that WeightedParticleBeliefStateUpdate instances are hashable."""
    particles1 = [1, 2, 3]
    weights1 = [0.3, 0.4, 0.3]
    belief1 = WeightedParticleBeliefStateUpdate(particles=particles1, weights=weights1)

    particles2 = [1, 2, 3]
    weights2 = [0.3, 0.4, 0.3]
    belief2 = WeightedParticleBeliefStateUpdate(particles=particles2, weights=weights2)

    # Test that beliefs can be used in a set
    belief_set = {belief1, belief2}
    assert len(belief_set) == 1  # Should only have one unique belief

    # Test that beliefs can be used as dictionary keys
    belief_dict = {belief1: "value1"}
    assert belief_dict[belief2] == "value1"  # Should be able to access using belief2


def test_weighted_particle_belief_state_update_edge_cases():
    """Test edge cases for WeightedParticleBeliefStateUpdate."""

    env = SanityPOMDP()

    # Test with very small weights
    belief = WeightedParticleBeliefStateUpdate(particles=[0], weights=[1e-10])
    belief.inplace_update(action=0, observation=0, pomdp=env, state=1)

    # Should still work
    assert len(belief.particles) == 2
    sample = belief.sample()
    assert sample in [0, 1]

    # Test with very large weights
    belief2 = WeightedParticleBeliefStateUpdate(particles=[0], weights=[1e10])
    belief2.inplace_update(action=0, observation=0, pomdp=env, state=1)

    # Should still work
    assert len(belief2.particles) == 2
    sample2 = belief2.sample()
    assert sample2 in [0, 1]


# Usage Example Tests for WeightedParticleBeliefStateUpdate


def test_weighted_particle_belief_state_update_comprehensive_usage_example():
    """Test the comprehensive WeightedParticleBeliefStateUpdate usage example from the class docstring."""

    # Create environment and empty belief (from docstring)
    env = TigerPOMDP(discount_factor=0.95)
    belief = WeightedParticleBeliefStateUpdate(particles=[], weights=[])

    # Add states incrementally with observations (from docstring)
    belief.inplace_update(action="listen", observation="hear_left", pomdp=env, state="tiger_left")
    belief.inplace_update(action="listen", observation="hear_left", pomdp=env, state="tiger_right")

    # Verify belief state
    assert len(belief.particles) == 2
    assert len(belief.weights) == 2
    assert belief.particles == ["tiger_left", "tiger_right"]

    # Verify weights are reasonable (tiger_left should have higher weight for hear_left observation)
    assert belief.weights[0] > belief.weights[1]

    # Sample weighted by observation probabilities (from docstring)
    sampled_state = belief.sample()  # More likely to be "tiger_left"
    assert sampled_state in ["tiger_left", "tiger_right"]

    # Create new belief with additional particle (from docstring)
    new_belief = belief.update(
        action="listen", observation="hear_right", pomdp=env, state="tiger_right"
    )

    # Verify new belief properties
    assert len(new_belief.particles) == 3
    assert len(new_belief.weights) == 3
    assert new_belief.particles == ["tiger_left", "tiger_right", "tiger_right"]

    # Verify original belief unchanged
    assert len(belief.particles) == 2


def test_weighted_particle_belief_state_update_update_method_example():
    """Test the update method usage example from WeightedParticleBeliefStateUpdate docstring."""

    # Create a mock environment for testing
    environment = TigerPOMDP(discount_factor=0.95)

    # Original belief with 2 particles (from docstring)
    belief = WeightedParticleBeliefStateUpdate(
        particles=["tiger_left", "tiger_right"], weights=[0.7, 0.3]
    )

    # Create new belief with additional particle (from docstring)
    new_belief = belief.update(
        action="listen", observation="hear_left", pomdp=environment, state="tiger_left"
    )

    # Original belief unchanged, new belief has 3 particles (from docstring)
    assert len(belief.particles) == 2
    assert len(new_belief.particles) == 3

    # Verify particle contents
    assert belief.particles == ["tiger_left", "tiger_right"]
    assert new_belief.particles == ["tiger_left", "tiger_right", "tiger_left"]

    # Verify weights
    assert len(belief.weights) == 2
    assert len(new_belief.weights) == 3
    assert belief.weights == [0.7, 0.3]  # Original weights unchanged


def test_weighted_particle_belief_state_update_inplace_update_example():
    """Test the inplace_update method usage example from WeightedParticleBeliefStateUpdate docstring."""

    env = TigerPOMDP(discount_factor=0.95)
    belief = WeightedParticleBeliefStateUpdate([], [])

    # Add particles one by one (from docstring)
    belief.inplace_update("listen", "hear_left", env, "tiger_left")
    belief.inplace_update("listen", "hear_right", env, "tiger_right")
    belief.inplace_update("listen", "hear_left", env, "tiger_left")

    # Belief now contains 3 particles with observation-based weights (from docstring)
    assert len(belief.particles) == 3
    assert belief.weights_sum > 0

    # Verify particle contents and weights
    assert belief.particles == ["tiger_left", "tiger_right", "tiger_left"]
    assert len(belief.weights) == 3
    assert all(w > 0 for w in belief.weights)

    # Verify weights_sum is correct
    expected_sum = sum(belief.weights)
    assert abs(belief.weights_sum - expected_sum) < 1e-10


def test_weighted_particle_belief_state_update_sample_method_example():
    """Test the sample method usage example from WeightedParticleBeliefStateUpdate docstring."""

    # Create belief with different observation likelihoods (from docstring)
    belief = WeightedParticleBeliefStateUpdate(
        particles=["tiger_left", "tiger_right", "tiger_left"],
        weights=[0.85, 0.15, 0.85],  # First and third more likely
    )

    # Sample multiple times (first and third states more probable) (from docstring)
    samples = [belief.sample() for _ in range(100)]

    # Count occurrences (tiger_left should be more frequent) (from docstring)
    left_count = samples.count("tiger_left")
    right_count = samples.count("tiger_right")
    assert left_count > right_count  # Due to higher weights

    # Verify all samples are valid
    assert all(s in ["tiger_left", "tiger_right"] for s in samples)

    # Verify sampling distribution is reasonable (allowing for randomness)
    left_ratio = left_count / 100

    # Allow some tolerance for randomness - tiger_left should be more frequent
    assert left_ratio > 0.6  # Should be much higher due to weights [0.85, 0.15, 0.85]


# Tests for UnweightedParticleBeliefStateUpdate class
def test_unweighted_particle_belief_state_update_initialization_empty():
    """Test UnweightedParticleBeliefStateUpdate initialization with empty list."""
    belief = UnweightedParticleBeliefStateUpdate()

    assert belief.particles == []
    assert belief.weights_sum == 0


def test_unweighted_particle_belief_state_update_initialization_with_data():
    """Test UnweightedParticleBeliefStateUpdate initialization with particles."""
    particles = ["state1", "state2", "state3"]

    belief = UnweightedParticleBeliefStateUpdate(particles=particles)

    assert belief.particles == particles
    assert belief.weights_sum == len(particles)


def test_unweighted_particle_belief_state_update_initialization_weight_sum():
    """Test that weights_sum is calculated correctly during initialization."""
    particles = [1, 2, 3, 4, 5]

    belief = UnweightedParticleBeliefStateUpdate(particles=particles)

    assert belief.weights_sum == 5


def test_unweighted_particle_belief_state_update_inplace_update():
    """Test inplace_update method adds state correctly."""

    env = SanityPOMDP()
    belief = UnweightedParticleBeliefStateUpdate(particles=[0])
    initial_sum = belief.weights_sum

    # Add a new state
    new_state = 1
    action = 0
    observation = 1

    belief.inplace_update(action=action, observation=observation, pomdp=env, state=new_state)

    # Check that the state was added
    assert new_state in belief.particles
    assert len(belief.particles) == 2

    # Check that weights_sum was updated
    assert belief.weights_sum == initial_sum + 1
    assert belief.weights_sum == len(belief.particles)


def test_unweighted_particle_belief_state_update_inplace_update_ignores_observation():
    """Test that inplace_update doesn't use observation probability (unweighted)."""

    env = SanityPOMDP()
    belief = UnweightedParticleBeliefStateUpdate()

    state = 0
    action = 0
    observation = 1  # Different from state, should still be added

    belief.inplace_update(action=action, observation=observation, pomdp=env, state=state)

    # Check that the state was added regardless of observation probability
    assert len(belief.particles) == 1
    assert belief.particles[0] == state
    assert belief.weights_sum == 1


def test_unweighted_particle_belief_state_update_multiple_inplace_updates():
    """Test multiple inplace_updates accumulate correctly."""

    env = SanityPOMDP()
    belief = UnweightedParticleBeliefStateUpdate(particles=[])

    states = [0, 1, 0, 2]
    action = 0
    observation = 0

    for state in states:
        belief.inplace_update(action=action, observation=observation, pomdp=env, state=state)

    assert len(belief.particles) == 4
    assert belief.particles == states
    assert belief.weights_sum == len(belief.particles)


def test_unweighted_particle_belief_state_update_update_returns_new_instance():
    """Test that update method returns a new instance."""

    env = SanityPOMDP()
    original_belief = UnweightedParticleBeliefStateUpdate(particles=[0])

    action = 0
    observation = 0
    state = 1

    updated_belief = original_belief.update(
        action=action, observation=observation, pomdp=env, state=state
    )

    # Should return a new instance
    assert updated_belief is not original_belief
    assert isinstance(updated_belief, UnweightedParticleBeliefStateUpdate)

    # Original belief should remain unchanged
    assert len(original_belief.particles) == 1
    assert original_belief.weights_sum == 1

    # New belief should have the additional state
    assert len(updated_belief.particles) == 2
    assert updated_belief.weights_sum == 2


def test_unweighted_particle_belief_state_update_update_preserves_original_data():
    """Test that update method preserves original particles."""

    env = SanityPOMDP()
    original_particles = [0, 1]
    original_belief = UnweightedParticleBeliefStateUpdate(particles=original_particles)

    action = 0
    observation = 0
    state = 2

    updated_belief = original_belief.update(
        action=action, observation=observation, pomdp=env, state=state
    )

    # Check that original data is preserved in new belief
    assert updated_belief.particles[:2] == original_particles

    # Check that new data is added
    assert updated_belief.particles[2] == state
    assert len(updated_belief.particles) == 3
    assert updated_belief.weights_sum == 3


def test_unweighted_particle_belief_state_update_inplace_vs_update_comparison():
    """Test that inplace_update modifies the belief in-place while update returns a new belief."""

    env = SanityPOMDP()

    # Test 1: inplace_update modifies the original belief
    belief1 = UnweightedParticleBeliefStateUpdate(particles=[0])
    original_particles1 = belief1.particles.copy()
    original_weights_sum1 = belief1.weights_sum

    belief1.inplace_update(action=0, observation=1, pomdp=env, state=1)

    # Verify that the original belief was modified in-place
    assert len(belief1.particles) == 2
    assert belief1.particles[0] == 0  # Original particle preserved
    assert belief1.particles[1] == 1  # New particle added
    assert belief1.weights_sum == original_weights_sum1 + 1  # Weight sum updated

    # Test 2: update returns a new belief without modifying the original
    belief2 = UnweightedParticleBeliefStateUpdate(particles=[0])
    original_particles2 = belief2.particles.copy()
    original_weights_sum2 = belief2.weights_sum

    updated_belief2 = belief2.update(action=0, observation=1, pomdp=env, state=1)

    # Verify that the original belief was NOT modified
    assert len(belief2.particles) == 1  # Original belief unchanged
    assert belief2.particles == original_particles2  # Original particles unchanged
    assert belief2.weights_sum == original_weights_sum2  # Original weight sum unchanged

    # Verify that the returned belief is a new instance with the additional data
    assert updated_belief2 is not belief2  # Different object
    assert len(updated_belief2.particles) == 2  # New belief has additional particle
    assert updated_belief2.particles[0] == 0  # Original particle preserved
    assert updated_belief2.particles[1] == 1  # New particle added
    assert updated_belief2.weights_sum == original_weights_sum2 + 1  # Weight sum updated


def test_unweighted_particle_belief_state_update_sample_basic():
    """Test basic sampling functionality."""
    particles = ["state1", "state2", "state3"]

    belief = UnweightedParticleBeliefStateUpdate(particles=particles)

    # Sample multiple times to test distribution
    samples = [belief.sample() for _ in range(100)]

    # Check that all samples are valid particles
    for sample in samples:
        assert sample in particles

    # Since it's unweighted, all states should appear with roughly equal frequency
    unique_samples = set(samples)
    assert unique_samples == set(particles)


def test_unweighted_particle_belief_state_update_sample_uniform_distribution():
    """Test sampling produces roughly uniform distribution."""
    particles = [1, 2, 3, 4]

    belief = UnweightedParticleBeliefStateUpdate(particles=particles)

    # Sample multiple times
    samples = [belief.sample() for _ in range(200)]

    # Check that all particles appear in samples
    unique_samples = set(samples)
    assert unique_samples == set(particles)

    # Check that distribution is roughly uniform (within reasonable bounds)
    for particle in particles:
        count = samples.count(particle)
        assert 30 <= count <= 70  # Should be roughly 50 ± 20


def test_unweighted_particle_belief_state_update_sample_single_particle():
    """Test sampling with a single particle."""
    particles = ["only_state"]

    belief = UnweightedParticleBeliefStateUpdate(particles=particles)

    # Should always return the only particle
    for _ in range(10):
        sample = belief.sample()
        assert sample == "only_state"


def test_unweighted_particle_belief_state_update_sample_duplicate_particles():
    """Test sampling with duplicate particles."""
    particles = [1, 1, 2, 2, 2, 3]  # Duplicates should increase probability

    belief = UnweightedParticleBeliefStateUpdate(particles=particles)

    # Sample multiple times
    samples = [belief.sample() for _ in range(300)]

    # Count occurrences - particle 2 should be most frequent due to 3 copies
    count_1 = samples.count(1)
    count_2 = samples.count(2)
    count_3 = samples.count(3)

    # Particle 2 should be most frequent (3/6 = 50% of particles)
    assert count_2 > count_1
    assert count_2 > count_3

    # Rough proportion check (allowing for randomness)
    assert 120 <= count_2 <= 180  # Should be roughly 150 ± 30


def test_unweighted_particle_belief_state_update_sample_empty_belief_raises_error():
    """Test that sampling from empty belief raises ValueError."""
    belief = UnweightedParticleBeliefStateUpdate([])

    with pytest.raises(IndexError):  # random.choice raises IndexError on empty sequence
        belief.sample()


def test_unweighted_particle_belief_state_update_with_tiger_pomdp():
    """Test UnweightedParticleBeliefStateUpdate integration with TigerPOMDP."""
    env = TigerPOMDP(discount_factor=0.95)

    # Start with empty belief
    belief = UnweightedParticleBeliefStateUpdate([])

    # Add some states (observations don't affect the weights in unweighted version)
    states = ["tiger_left", "tiger_right", "tiger_left"]
    action = "listen"
    observation = "hear_left"

    for state in states:
        belief.inplace_update(action=action, observation=observation, pomdp=env, state=state)

    # Check that all states were added
    assert len(belief.particles) == 3
    assert belief.particles == states
    assert belief.weights_sum == 3

    # Sample should work
    sample = belief.sample()
    assert sample in ["tiger_left", "tiger_right"]


def test_unweighted_particle_belief_state_update_inheritance():
    """Test that UnweightedParticleBeliefStateUpdate inherits from Belief."""
    belief = UnweightedParticleBeliefStateUpdate()

    assert isinstance(belief, Belief)


def test_unweighted_particle_belief_state_update_config_id():
    """Test that UnweightedParticleBeliefStateUpdate has a config_id property."""
    particles = [1, 2, 3]

    belief = UnweightedParticleBeliefStateUpdate(particles=particles)

    # Should have a config_id
    assert hasattr(belief, "config_id")
    assert isinstance(belief.config_id, str)


def test_unweighted_particle_belief_state_update_config_id_deterministic():
    """Test that identical beliefs have identical config_ids."""
    particles1 = [1, 2, 3]
    belief1 = UnweightedParticleBeliefStateUpdate(particles=particles1)

    particles2 = [1, 2, 3]
    belief2 = UnweightedParticleBeliefStateUpdate(particles=particles2)

    assert belief1.config_id == belief2.config_id


def test_unweighted_particle_belief_state_update_config_id_changes_with_particles():
    """Test that config_id changes when particles change."""
    particles1 = [1, 2, 3]
    belief1 = UnweightedParticleBeliefStateUpdate(particles=particles1)

    particles2 = [1, 2, 4]  # Different particle
    belief2 = UnweightedParticleBeliefStateUpdate(particles=particles2)

    assert belief1.config_id != belief2.config_id


def test_unweighted_particle_belief_state_update_config_id_with_numpy_particles():
    """Test config_id with numpy array particles."""
    particles1 = [np.array([1, 2]), np.array([3, 4])]
    belief1 = UnweightedParticleBeliefStateUpdate(particles=particles1)

    particles2 = [np.array([1, 2]), np.array([3, 4])]
    belief2 = UnweightedParticleBeliefStateUpdate(particles=particles2)

    # Config IDs should be identical for identical numpy array particles
    assert belief1.config_id == belief2.config_id


def test_unweighted_particle_belief_state_update_config_id_with_mixed_particles():
    """Test config_id with mixed type particles."""
    particles1 = [1, np.array([2, 3]), "test"]
    belief1 = UnweightedParticleBeliefStateUpdate(particles=particles1)

    particles2 = [1, np.array([2, 3]), "test"]
    belief2 = UnweightedParticleBeliefStateUpdate(particles=particles2)

    # Config IDs should be identical for identical mixed type particles
    assert belief1.config_id == belief2.config_id


def test_unweighted_particle_belief_state_update_config_id_with_different_order():
    """Test config_id with particles in different order."""
    particles1 = [1, 2, 3]
    belief1 = UnweightedParticleBeliefStateUpdate(particles=particles1)

    # Same particles but in different order
    particles2 = [3, 1, 2]
    belief2 = UnweightedParticleBeliefStateUpdate(particles=particles2)

    # Config IDs should be identical since they represent the same set of particles
    assert belief1.config_id == belief2.config_id


def test_unweighted_particle_belief_state_update_config_id_with_different_numpy_order():
    """Test config_id with numpy array particles in different order."""
    particles1 = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    belief1 = UnweightedParticleBeliefStateUpdate(particles=particles1)

    # Same particles but in different order
    particles2 = [np.array([5, 6]), np.array([1, 2]), np.array([3, 4])]
    belief2 = UnweightedParticleBeliefStateUpdate(particles=particles2)

    # Config IDs should be identical since they represent the same set of particles
    assert belief1.config_id == belief2.config_id


def test_unweighted_particle_belief_state_update_config_id_with_empty_belief():
    """Test config_id with empty belief."""
    belief1 = UnweightedParticleBeliefStateUpdate(particles=[])
    belief2 = UnweightedParticleBeliefStateUpdate(particles=[])

    # Config IDs should be identical for empty beliefs
    assert belief1.config_id == belief2.config_id


def test_unweighted_particle_belief_state_update_config_id_with_single_particle():
    """Test config_id with single particle."""
    particles1 = ["single_state"]
    belief1 = UnweightedParticleBeliefStateUpdate(particles=particles1)

    particles2 = ["single_state"]
    belief2 = UnweightedParticleBeliefStateUpdate(particles=particles2)

    # Config IDs should be identical
    assert belief1.config_id == belief2.config_id


def test_unweighted_particle_belief_state_update_config_id_with_duplicate_particles():
    """Test config_id with duplicate particles."""
    particles1 = [1, 1, 2, 2, 2, 3]  # Duplicate particles
    belief1 = UnweightedParticleBeliefStateUpdate(particles=particles1)

    particles2 = [1, 1, 2, 2, 2, 3]  # Same duplicates
    belief2 = UnweightedParticleBeliefStateUpdate(particles=particles2)

    # Config IDs should be identical for identical duplicate particles
    assert belief1.config_id == belief2.config_id


def test_unweighted_particle_belief_state_update_config_id_different_duplicates():
    """Test config_id with different numbers of duplicate particles."""
    particles1 = [1, 1, 2, 3]  # Two 1s
    belief1 = UnweightedParticleBeliefStateUpdate(particles=particles1)

    particles2 = [1, 2, 3]  # One 1
    belief2 = UnweightedParticleBeliefStateUpdate(particles=particles2)

    # Config IDs should be different since duplicates matter for sampling probability
    assert belief1.config_id != belief2.config_id


def test_unweighted_particle_belief_state_update_config_id_uniqueness():
    """Test that config_id is unique for different beliefs."""
    # Create several different beliefs
    belief1 = UnweightedParticleBeliefStateUpdate(particles=[1, 2])
    belief2 = UnweightedParticleBeliefStateUpdate(particles=[1, 3])
    belief3 = UnweightedParticleBeliefStateUpdate(particles=[])
    belief4 = UnweightedParticleBeliefStateUpdate(particles=[np.array([1, 2])])
    belief5 = UnweightedParticleBeliefStateUpdate(particles=[1, 1, 2])  # Different duplicates

    # All config_ids should be unique
    config_ids = [
        belief1.config_id,
        belief2.config_id,
        belief3.config_id,
        belief4.config_id,
        belief5.config_id,
    ]
    assert len(config_ids) == len(set(config_ids)), "All config_ids should be unique"


def test_unweighted_particle_belief_state_update_config_id_consistency():
    """Test that config_id is consistent across multiple calls."""
    particles = [1, 2, 3]
    belief = UnweightedParticleBeliefStateUpdate(particles=particles)

    # Config_id should be the same on multiple calls
    config_id1 = belief.config_id
    config_id2 = belief.config_id
    config_id3 = belief.config_id

    assert config_id1 == config_id2 == config_id3


def test_unweighted_particle_belief_state_update_equality():
    """Test equality comparison between UnweightedParticleBeliefStateUpdate instances."""
    particles1 = [1, 2, 3]
    belief1 = UnweightedParticleBeliefStateUpdate(particles=particles1)

    particles2 = [1, 2, 3]
    belief2 = UnweightedParticleBeliefStateUpdate(particles=particles2)

    # Should be equal
    assert belief1 == belief2
    assert belief2 == belief1


def test_unweighted_particle_belief_state_update_inequality():
    """Test inequality comparison between UnweightedParticleBeliefStateUpdate instances."""
    particles1 = [1, 2, 3]
    belief1 = UnweightedParticleBeliefStateUpdate(particles=particles1)

    # Different particles
    particles2 = [1, 2, 4]
    belief2 = UnweightedParticleBeliefStateUpdate(particles=particles2)

    # Different number of particles
    particles3 = [1, 2]
    belief3 = UnweightedParticleBeliefStateUpdate(particles=particles3)

    assert belief1 != belief2
    assert belief1 != belief3


def test_unweighted_particle_belief_state_update_hashable():
    """Test that UnweightedParticleBeliefStateUpdate instances are hashable."""
    particles1 = [1, 2, 3]
    belief1 = UnweightedParticleBeliefStateUpdate(particles=particles1)

    particles2 = [1, 2, 3]
    belief2 = UnweightedParticleBeliefStateUpdate(particles=particles2)

    # Test that beliefs can be used in a set
    belief_set = {belief1, belief2}
    assert len(belief_set) == 1  # Should only have one unique belief

    # Test that beliefs can be used as dictionary keys
    belief_dict = {belief1: "value1"}
    assert belief_dict[belief2] == "value1"  # Should be able to access using belief2


def test_unweighted_particle_belief_state_update_edge_cases():
    """Test edge cases for UnweightedParticleBeliefStateUpdate."""

    env = SanityPOMDP()

    # Test with None state
    belief = UnweightedParticleBeliefStateUpdate(particles=[0])
    belief.inplace_update(action=0, observation=0, pomdp=env, state=None)

    # Should still work
    assert len(belief.particles) == 2
    assert belief.particles[1] is None

    # Test sampling with None in particles
    samples = [belief.sample() for _ in range(10)]
    assert all(s in [0, None] for s in samples)


def test_unweighted_particle_belief_state_update_comprehensive_usage_example():
    """Test comprehensive usage example similar to the weighted version."""

    # Create environment and empty belief
    env = TigerPOMDP(discount_factor=0.95)
    belief = UnweightedParticleBeliefStateUpdate(particles=[])

    # Add states incrementally (observations are ignored in unweighted version)
    belief.inplace_update(action="listen", observation="hear_left", pomdp=env, state="tiger_left")
    belief.inplace_update(action="listen", observation="hear_right", pomdp=env, state="tiger_right")

    # Verify belief state
    assert len(belief.particles) == 2
    assert belief.weights_sum == 2
    assert belief.particles == ["tiger_left", "tiger_right"]

    # Sample from uniform distribution (unweighted)
    sampled_state = belief.sample()
    assert sampled_state in ["tiger_left", "tiger_right"]

    # Create new belief with additional particle
    new_belief = belief.update(
        action="listen", observation="hear_left", pomdp=env, state="tiger_left"
    )

    # Verify new belief properties
    assert len(new_belief.particles) == 3
    assert new_belief.weights_sum == 3
    assert new_belief.particles == ["tiger_left", "tiger_right", "tiger_left"]

    # Verify original belief unchanged
    assert len(belief.particles) == 2


def test_unweighted_particle_belief_state_update_differences_from_weighted():
    """Test that UnweightedParticleBeliefStateUpdate behaves differently from WeightedParticleBeliefStateUpdate."""

    env = TigerPOMDP(discount_factor=0.95)

    # Create both types of beliefs
    unweighted_belief = UnweightedParticleBeliefStateUpdate([])
    weighted_belief = WeightedParticleBeliefStateUpdate([], [])

    # Add the same states with different observation matches
    state = "tiger_left"
    action = "listen"
    good_observation = "hear_left"  # Matches tiger_left
    bad_observation = "hear_right"  # Doesn't match tiger_left

    # Add state with good observation
    unweighted_belief.inplace_update(action, good_observation, env, state)
    weighted_belief.inplace_update(action, good_observation, env, state)

    # Add state with bad observation
    unweighted_belief.inplace_update(action, bad_observation, env, state)
    weighted_belief.inplace_update(action, bad_observation, env, state)

    # Unweighted belief should treat both equally (weights_sum = particle count)
    assert unweighted_belief.weights_sum == len(unweighted_belief.particles) == 2

    # Weighted belief should have different weights based on observation probabilities
    assert len(weighted_belief.particles) == 2
    assert (
        weighted_belief.weights[0] != weighted_belief.weights[1]
    )  # Different observation likelihoods

    # Unweighted belief attributes should be simpler
    assert not hasattr(unweighted_belief, "weights")
    assert hasattr(weighted_belief, "weights")


def test_unweighted_particle_belief_state_update_sample_with_extreme_cases():
    """Test sampling behavior with extreme cases."""
    # Test with very large number of particles
    large_particles = list(range(1000))
    belief = UnweightedParticleBeliefStateUpdate(particles=large_particles)

    # Should still work efficiently
    sample = belief.sample()
    assert sample in large_particles

    # Test with mixed data types
    mixed_particles = [1, "string", np.array([1, 2]), [3, 4], {"key": "value"}]
    belief2 = UnweightedParticleBeliefStateUpdate(particles=mixed_particles)

    sample2 = belief2.sample()
    # Sample should be one of the original particles (no type conversion)
    # Need to handle numpy array comparison carefully
    found_sample = False
    for particle in mixed_particles:
        if isinstance(particle, np.ndarray) and isinstance(sample2, np.ndarray):
            if np.array_equal(particle, sample2):
                found_sample = True
                break
        elif particle is sample2:
            found_sample = True
            break
        elif not isinstance(particle, np.ndarray) and not isinstance(sample2, np.ndarray):
            if particle == sample2:
                found_sample = True
                break
    assert found_sample, f"Sample {sample2} not found in particles {mixed_particles}"


# Additional usage example tests for WeightedParticleBeliefStateUpdate
# Following the project standard of testing all usage examples from docstrings


def test_weighted_particle_belief_basic_incremental_construction_usage_example():
    """Test the basic incremental belief construction usage example from WeightedParticleBeliefStateUpdate docstring."""

    # Create environment and empty belief (from docstring)
    env = TigerPOMDP(discount_factor=0.95)
    belief = WeightedParticleBeliefStateUpdate(particles=[], weights=[])

    # Add states incrementally with observations (from docstring)
    belief.inplace_update("listen", "hear_left", env, "tiger_left")
    belief.inplace_update("listen", "hear_left", env, "tiger_right")
    belief.inplace_update("listen", "hear_left", env, "tiger_left")

    # Verify results
    assert len(belief.particles) == 3, f"Expected 3 particles, got {len(belief.particles)}"
    assert len(belief.weights) == 3, f"Expected 3 weights, got {len(belief.weights)}"
    assert belief.weights_sum > 0, f"Expected positive weights_sum, got {belief.weights_sum}"

    # Verify particles
    assert belief.particles == ["tiger_left", "tiger_right", "tiger_left"]

    # Sample weighted by observation probabilities (from docstring)
    sampled_state = belief.sample()  # More likely to be "tiger_left"
    assert sampled_state in [
        "tiger_left",
        "tiger_right",
    ], f"Invalid sampled state: {sampled_state}"


def test_weighted_particle_belief_immutable_updates_usage_example():
    """Test the immutable belief updates for tree search usage example from WeightedParticleBeliefStateUpdate docstring."""

    # Create continuous state environment (from docstring)
    noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
    env = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov)

    # Start with single particle (from docstring)
    initial_state = np.array([0.0, 0.0, 0.1, 0.0])  # [x, x_dot, theta, theta_dot]
    belief = WeightedParticleBeliefStateUpdate([initial_state], [1.0])

    # Create child beliefs for different observations (tree expansion) (from docstring)
    action = 1  # Apply force right
    observations = [
        np.array([0.1, 0.0, 0.08, 0.0]),  # Likely observation
        np.array([0.2, 0.0, 0.12, 0.0]),  # Less likely observation
    ]

    child_beliefs = []
    for obs in observations:
        # Generate potential next state (from docstring)
        next_state = env.state_transition_model(initial_state, action).sample()[0]

        # Create new belief (immutable update) (from docstring)
        child_belief = belief.update(action, obs, env, next_state)
        child_beliefs.append(child_belief)

    # Verify results
    assert len(child_beliefs) == 2, f"Expected 2 child beliefs, got {len(child_beliefs)}"

    for child_belief in child_beliefs:
        assert len(child_belief.particles) == 2, "Each child should have 2 particles"
        assert len(child_belief.weights) == 2, "Each child should have 2 weights"
        assert child_belief.weights[-1] > 0, "New particle should have positive weight"

    # Original belief should remain unchanged
    assert len(belief.particles) == 1, "Original belief should still have 1 particle"
    assert len(belief.weights) == 1, "Original belief should still have 1 weight"


def test_weighted_particle_belief_update_strategies_comparison_usage_example():
    """Test the comparing belief update strategies usage example from WeightedParticleBeliefStateUpdate docstring."""

    env = SanityPOMDP(discount_factor=0.95)

    # Strategy 1: In-place updates (memory efficient) (from docstring)
    belief_inplace = WeightedParticleBeliefStateUpdate([], [])
    states_observations = [
        (0, 0),
        (1, 0),
        (0, 0),
        (1, 1),
        (0, 0),  # (state, observation) pairs
    ]

    for state, obs in states_observations:
        belief_inplace.inplace_update("action", obs, env, state)

    # Strategy 2: Immutable updates (functional style) (from docstring)
    belief_immutable = WeightedParticleBeliefStateUpdate([], [])
    for state, obs in states_observations:
        belief_immutable = belief_immutable.update("action", obs, env, state)

    # Both should have same final state (from docstring)
    assert len(belief_inplace.particles) == len(
        belief_immutable.particles
    ), "Both strategies should yield same number of particles"
    assert (
        len(belief_inplace.particles) == 5
    ), f"Expected 5 particles, got {len(belief_inplace.particles)}"

    # Verify particles match
    assert belief_inplace.particles == belief_immutable.particles, "Particle sequences should match"
    assert belief_inplace.weights == belief_immutable.weights, "Weight sequences should match"


def test_weighted_particle_belief_mcts_integration_usage_example():
    """Test the Monte Carlo Tree Search integration usage example from WeightedParticleBeliefStateUpdate docstring."""

    env = TigerPOMDP(discount_factor=0.95)

    # Root belief node with initial particles (from docstring)
    root_belief = WeightedParticleBeliefStateUpdate(
        particles=["tiger_left", "tiger_right"], weights=[0.5, 0.5]
    )
    root_node = BeliefNode(belief=root_belief)

    # Simulate MCTS expansion (from docstring)
    action = "listen"
    possible_observations = ["hear_left", "hear_right"]

    # Create action node (from docstring)
    action_node = ActionNode(action=action, parent=root_node)

    # For each possible observation, create belief child (from docstring)
    belief_nodes = []
    for observation in possible_observations:
        # Sample particles and create child belief (from docstring)
        child_belief = WeightedParticleBeliefStateUpdate([], [])

        # Add particles based on transition model (from docstring)
        for _ in range(5):  # Multiple particles per observation
            parent_state = root_belief.sample()
            next_state = env.state_transition_model(parent_state, action).sample()[0]
            child_belief.inplace_update(action, observation, env, next_state)

        # Create belief node for tree (from docstring)
        belief_node = BeliefNode(belief=child_belief, observation=observation, parent=action_node)
        belief_nodes.append(belief_node)

    # Verify tree structure
    assert len(belief_nodes) == 2, f"Expected 2 belief nodes, got {len(belief_nodes)}"
    assert len(action_node.children) == 2, "Action node should have 2 children"

    for belief_node in belief_nodes:
        assert len(belief_node.belief.particles) == 5, "Each child belief should have 5 particles"
        assert belief_node.parent == action_node, "Belief node should be child of action node"
        assert belief_node.observation in possible_observations, "Invalid observation"


def test_weighted_particle_belief_weighted_sampling_usage_example():
    """Test the weighted sampling and state estimation usage example from WeightedParticleBeliefStateUpdate docstring."""

    env = TigerPOMDP(discount_factor=0.95)
    belief = WeightedParticleBeliefStateUpdate([], [])

    # Add strongly biased evidence for tiger_left (from docstring)
    evidence_sets = [
        ("tiger_left", "hear_left", 5),  # Strong evidence for left
        ("tiger_right", "hear_left", 2),  # Weak evidence for left from right
        ("tiger_left", "hear_right", 1),  # Weak evidence for right from left
        ("tiger_right", "hear_right", 3),  # Medium evidence for right
    ]

    for state, obs, count in evidence_sets:
        for _ in range(count):
            belief.inplace_update("listen", obs, env, state)

    # Analyze sampling distribution (from docstring)
    samples = [belief.sample() for _ in range(200)]  # Reduced for testing speed
    sample_counts = collections.Counter(samples)

    # Verify we have both states represented
    assert "tiger_left" in sample_counts, "tiger_left should be sampled"
    assert "tiger_right" in sample_counts, "tiger_right should be sampled"

    # Should strongly favor tiger_left due to evidence weighting (from docstring)
    left_proportion = sample_counts["tiger_left"] / len(samples)
    right_proportion = sample_counts["tiger_right"] / len(samples)

    # Given the evidence pattern, tiger_left should be more likely
    assert (
        left_proportion > right_proportion
    ), "tiger_left should be sampled more frequently due to evidence weighting"


def test_weighted_particle_belief_config_id_caching_usage_example():
    """Test the configuration ID and caching usage example from WeightedParticleBeliefStateUpdate docstring."""

    env = SanityPOMDP(discount_factor=0.95)

    # Create two beliefs with same particles in different orders (from docstring)
    belief1 = WeightedParticleBeliefStateUpdate([0, 1, 0], [0.8, 0.2, 0.6])
    belief2 = WeightedParticleBeliefStateUpdate([1, 0, 0], [0.2, 0.8, 0.6])

    # Config IDs should be equal (order-invariant) (from docstring)
    config_id_1 = belief1.config_id
    config_id_2 = belief2.config_id

    assert config_id_1 == config_id_2, f"Config IDs should match: {config_id_1} vs {config_id_2}"

    # Useful for caching in planning algorithms (from docstring)
    belief_cache = {belief1.config_id: "cached_result"}

    # Should find cache hit
    assert belief2.config_id in belief_cache, "Should find cache hit with order-invariant config ID"
    assert belief_cache[belief2.config_id] == "cached_result", "Cache should return correct result"


def test_weighted_particle_belief_custom_particle_types_usage_example():
    """Test the custom particle types usage example from WeightedParticleBeliefStateUpdate docstring."""

    # Works with any particle type - numpy arrays, custom objects, etc. (from docstring)
    noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
    env = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov)

    # Numpy array particles (from docstring)
    particles = [
        np.array([0.0, 0.0, 0.1, 0.0]),
        np.array([0.1, 0.0, 0.08, 0.0]),
        np.array([-0.1, 0.0, 0.12, 0.0]),
    ]
    weights = [0.4, 0.35, 0.25]

    belief = WeightedParticleBeliefStateUpdate(particles, weights)

    # Add more complex state (from docstring)
    complex_state = np.array([0.05, 0.1, 0.09, -0.05])
    action = np.array([1])  # Force right
    observation = np.array([0.06, 0.1, 0.088, -0.05])

    new_belief = belief.update(action, observation, env, complex_state)
    sampled_state = new_belief.sample()

    # Verify results (from docstring)
    assert isinstance(sampled_state, np.ndarray), f"Expected numpy array, got {type(sampled_state)}"
    assert sampled_state.shape == (4,), f"Expected shape (4,), got {sampled_state.shape}"
    assert len(new_belief.particles) == 4, "Expected 4 particles after update"
    assert len(new_belief.weights) == 4, "Expected 4 weights after update"

    # All particles should be numpy arrays
    for particle in new_belief.particles:
        assert isinstance(particle, np.ndarray), "All particles should be numpy arrays"
        assert particle.shape == (4,), "All particles should have shape (4,)"


# UnweightedParticleBeliefStateUpdate Usage Example Tests
# Following the project standard of testing all usage examples from docstrings


def test_unweighted_particle_belief_state_update_basic_uniform_belief_construction_usage_example():
    """Test the basic uniform belief construction usage example from UnweightedParticleBeliefStateUpdate docstring."""

    # Create environment and empty uniform belief
    env = TigerPOMDP(discount_factor=0.95)
    belief = UnweightedParticleBeliefStateUpdate(particles=[])

    # Add states uniformly (all have equal probability)
    belief.inplace_update("listen", "hear_left", env, "tiger_left")
    belief.inplace_update("listen", "hear_right", env, "tiger_right")
    belief.inplace_update("listen", "hear_left", env, "tiger_left")

    assert len(belief.particles) == 3
    assert belief.weights_sum == 3  # Equal to number of particles

    # Sample uniformly from particles - should work
    sampled_state = belief.sample()
    assert sampled_state in ["tiger_left", "tiger_right"]


def test_unweighted_particle_belief_state_update_mcts_with_uniform_beliefs_usage_example():
    """Test the Monte Carlo Tree Search with uniform beliefs usage example from UnweightedParticleBeliefStateUpdate docstring."""

    env = SanityPOMDP(discount_factor=0.95)

    # Root belief with uniform initial distribution
    root_belief = UnweightedParticleBeliefStateUpdate(
        particles=[0, 1, 0, 1, 0]  # More 0s than 1s, but all weighted equally
    )
    root_node = BeliefNode(belief=root_belief)

    # Simulate MCTS node expansion
    action = 0  # Good action
    possible_observations = [0, 1]  # Discrete observations

    # Create action node
    action_node = ActionNode(action=action, parent=root_node)

    # For each observation, accumulate child belief
    for observation in possible_observations:
        child_belief = UnweightedParticleBeliefStateUpdate([])

        # Add particles uniformly based on environment dynamics
        for _ in range(5):  # Multiple simulations
            parent_state = root_belief.sample()
            next_state = env.state_transition_model(parent_state, action).sample()[0]
            child_belief.inplace_update(action, observation, env, next_state)

        # Create belief node
        belief_node = BeliefNode(belief=child_belief, observation=observation, parent=action_node)
        assert len(child_belief.particles) > 0


def test_unweighted_particle_belief_state_update_comparing_weighted_vs_unweighted_usage_example():
    """Test the comparing weighted vs unweighted belief updates usage example from UnweightedParticleBeliefStateUpdate docstring."""

    env = SanityPOMDP(discount_factor=0.95)

    # Same particle sequence for both belief types
    states = [0, 1, 0, 1, 0, 1, 0]  # More 0s (good states) than 1s (bad states)
    observations = [0, 1, 0, 0, 1, 0, 1]
    action = 0

    # Weighted belief considers observation likelihoods
    weighted_belief = WeightedParticleBeliefStateUpdate([], [])
    for state, obs in zip(states, observations):
        weighted_belief.inplace_update(action, obs, env, state)

    # Unweighted belief treats all particles equally
    unweighted_belief = UnweightedParticleBeliefStateUpdate([])
    for state, obs in zip(states, observations):
        unweighted_belief.inplace_update(action, obs, env, state)

    # Compare sampling distributions
    weighted_samples = [weighted_belief.sample() for _ in range(100)]  # Reduced for test speed
    weighted_counts = collections.Counter(weighted_samples)

    unweighted_samples = [unweighted_belief.sample() for _ in range(100)]  # Reduced for test speed
    unweighted_counts = collections.Counter(unweighted_samples)

    # Verify both work and have reasonable distributions
    assert len(weighted_counts) > 0
    assert len(unweighted_counts) > 0
    assert sum(weighted_counts.values()) == 100
    assert sum(unweighted_counts.values()) == 100


def test_unweighted_particle_belief_state_update_discrete_observation_filtering_usage_example():
    """Test the discrete observation filtering usage example from UnweightedParticleBeliefStateUpdate docstring."""

    env = TigerPOMDP(discount_factor=0.95)
    belief = UnweightedParticleBeliefStateUpdate([])

    # Simulate multiple observations (discrete: hear_left or hear_right)
    observation_sequence = [
        ("tiger_left", "hear_left"),  # Consistent evidence
        ("tiger_left", "hear_left"),  # More consistent evidence
        ("tiger_right", "hear_left"),  # Inconsistent evidence
        ("tiger_left", "hear_right"),  # Inconsistent evidence
        ("tiger_left", "hear_left"),  # Back to consistent
        ("tiger_right", "hear_right"),  # Consistent for right
    ]

    for state, obs in observation_sequence:
        belief.inplace_update("listen", obs, env, state)

    assert len(belief.particles) == 6

    # Analyze uniform distribution over accumulated particles
    samples = [belief.sample() for _ in range(100)]  # Reduced for test speed
    sample_counts = collections.Counter(samples)

    # Verify uniform distribution properties
    for state, count in sample_counts.items():
        probability = count / 100
        particle_count = belief.particles.count(state)
        expected_prob = particle_count / len(belief.particles)
        # Allow some variance due to random sampling
        assert (
            abs(probability - expected_prob) < 0.2
        ), f"Probability {probability} too far from expected {expected_prob}"


def test_unweighted_particle_belief_state_update_immutable_belief_trees_usage_example():
    """Test the immutable belief trees for planning usage example from UnweightedParticleBeliefStateUpdate docstring."""

    # Create continuous state environment
    noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
    env = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov)

    # Start with uniform belief over multiple initial states
    initial_states = [
        np.array([0.0, 0.0, 0.1, 0.0]),  # Balanced pole
        np.array([0.1, 0.0, 0.08, 0.0]),  # Slightly right
        np.array([-0.1, 0.0, 0.12, 0.0]),  # Slightly left
    ]

    root_belief = UnweightedParticleBeliefStateUpdate(initial_states)

    # Generate child beliefs for different actions (functional style)
    actions = [0, 1]  # Push left or right
    child_beliefs = {}

    for action in actions:
        child_belief = UnweightedParticleBeliefStateUpdate([])

        # Generate next states uniformly
        for _ in range(3):  # Reduced for test speed
            current_state = root_belief.sample()
            next_state = env.state_transition_model(current_state, action).sample()[0]
            # For simplicity, assume observation equals next state (fully observable case)
            child_belief = child_belief.update(action, next_state, env, next_state)

        child_beliefs[action] = child_belief
        assert len(child_belief.particles) == 3  # Should have 3 particles

    # All child beliefs maintain uniform distribution over their particles
    for action, child_belief in child_beliefs.items():
        sample = child_belief.sample()
        assert isinstance(sample, np.ndarray)
        assert len(sample) == 4  # CartPole state dimension


def test_unweighted_particle_belief_state_update_memory_efficient_accumulation_usage_example():
    """Test the memory-efficient particle accumulation usage example from UnweightedParticleBeliefStateUpdate docstring."""

    env = SanityPOMDP(discount_factor=0.95)

    # Compare memory usage between weighted and unweighted beliefs
    weighted_belief = WeightedParticleBeliefStateUpdate([], [])
    unweighted_belief = UnweightedParticleBeliefStateUpdate([])

    # Add particles (reduced count for test speed)
    states = [0, 1] * 50  # 100 particles instead of 2000
    for state in states:
        weighted_belief.inplace_update("action", 0, env, state)
        unweighted_belief.inplace_update("action", 0, env, state)

    # Both should have same number of particles but unweighted saves memory on weights
    assert len(weighted_belief.particles) == len(unweighted_belief.particles)
    assert len(weighted_belief.weights) == len(weighted_belief.particles)
    # Unweighted belief doesn't store individual weights, just the count
    assert unweighted_belief.weights_sum == len(unweighted_belief.particles)


def test_unweighted_particle_belief_state_update_configuration_caching_usage_example():
    """Test the configuration caching and equality usage example from UnweightedParticleBeliefStateUpdate docstring."""

    env = SanityPOMDP(discount_factor=0.95)

    # Create two beliefs with same particles in different orders
    belief1 = UnweightedParticleBeliefStateUpdate([0, 1, 0, 1, 0])
    belief2 = UnweightedParticleBeliefStateUpdate([1, 0, 1, 0, 0])

    # Config IDs should be equal (order-invariant)
    config_id1 = belief1.config_id
    config_id2 = belief2.config_id
    assert isinstance(config_id1, str)
    assert isinstance(config_id2, str)
    assert config_id1 == config_id2

    # Test belief equality
    assert belief1 == belief2

    # Test hashing for cache usage
    belief_cache = {belief1: "cached_computation"}
    assert belief2 in belief_cache  # Should find it due to equality


def test_unweighted_particle_belief_state_update_large_scale_accumulation_usage_example():
    """Test the large-scale particle accumulation usage example from UnweightedParticleBeliefStateUpdate docstring."""

    env = TigerPOMDP(discount_factor=0.95)
    belief = UnweightedParticleBeliefStateUpdate([])

    # Time large-scale particle addition (reduced scale for testing)
    start_time = time.time()

    # Add particles uniformly (reduced count for test speed)
    states = ["tiger_left", "tiger_right"]
    observations = ["hear_left", "hear_right"]

    n_particles = 100  # Reduced from 10000 for test speed
    for i in range(n_particles):
        state = states[i % 2]  # Alternate between states
        obs = observations[i % 2]  # Alternate between observations
        belief.inplace_update("listen", obs, env, state)

    end_time = time.time()

    assert len(belief.particles) == n_particles
    execution_time = end_time - start_time
    assert execution_time < 10.0  # Should complete in reasonable time

    # Verify uniform distribution
    samples = [belief.sample() for _ in range(100)]  # Reduced for test speed
    sample_counts = collections.Counter(samples)

    # Should be approximately uniform between tiger_left and tiger_right
    for state, count in sample_counts.items():
        probability = count / 100
        assert 0.3 <= probability <= 0.7  # Allow reasonable variance
