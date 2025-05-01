import pytest
import numpy as np
from POMDPPlanners.core.belief import WeightedParticleBelief

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
