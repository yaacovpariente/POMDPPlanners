import pytest
import numpy as np
from POMDPPlanners.core.belief import Belief, WeightedParticleBelief

def test_belief_config_id_deterministic():
    """Test that config_id is deterministic for identical beliefs."""
    class TestBelief(Belief):
        def __init__(self, value):
            self.value = value
            
        def update(self, action, observation, pomdp):
            return self
            
        def sample(self):
            return self.value
    
    # Create two identical beliefs
    belief1 = TestBelief(value=42)
    belief2 = TestBelief(value=42)
    
    # Config IDs should be identical for identical beliefs
    assert belief1.config_id == belief2.config_id

def test_belief_config_id_changes_with_value():
    """Test that config_id changes when belief values change."""
    class TestBelief(Belief):
        def __init__(self, value):
            self.value = value
            
        def update(self, action, observation, pomdp):
            return self
            
        def sample(self):
            return self.value
    
    # Create two beliefs with different values
    belief1 = TestBelief(value=42)
    belief2 = TestBelief(value=43)
    
    # Config IDs should be different
    assert belief1.config_id != belief2.config_id

def test_belief_config_id_with_numpy_values():
    """Test config_id with numpy array values."""
    class TestBelief(Belief):
        def __init__(self, value):
            self.value = value
            
        def update(self, action, observation, pomdp):
            return self
            
        def sample(self):
            return self.value
    
    # Create two beliefs with identical numpy arrays
    belief1 = TestBelief(value=np.array([1, 2, 3]))
    belief2 = TestBelief(value=np.array([1, 2, 3]))
    
    # Config IDs should be identical
    assert belief1.config_id == belief2.config_id

def test_belief_hashable():
    """Test that beliefs are hashable."""
    class TestBelief(Belief):
        def __init__(self, value):
            self.value = value
            
        def update(self, action, observation, pomdp):
            return self
            
        def sample(self):
            return self.value
    
    # Create two identical beliefs
    belief1 = TestBelief(value=42)
    belief2 = TestBelief(value=42)
    
    # Test that beliefs can be used in a set
    belief_set = {belief1, belief2}
    assert len(belief_set) == 1  # Should only have one unique belief
    
    # Test that beliefs can be used as dictionary keys
    belief_dict = {belief1: "value1"}
    assert belief_dict[belief2] == "value1"  # Should be able to access using belief2

def test_belief_equality():
    """Test belief equality."""
    class TestBelief(Belief):
        def __init__(self, value):
            self.value = value
            
        def update(self, action, observation, pomdp):
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
            
        def update(self, action, observation, pomdp):
            return self
            
        def sample(self):
            return self.value
    
    belief = TestBelief(value=42)
    
    # Should return NotImplemented for non-Belief types
    assert belief.__eq__(42) == NotImplemented
    assert belief.__eq__("not a belief") == NotImplemented

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
