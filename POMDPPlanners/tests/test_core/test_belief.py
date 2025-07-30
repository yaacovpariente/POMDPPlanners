import pytest
import numpy as np
from POMDPPlanners.core.belief import Belief, WeightedParticleBelief
from POMDPPlanners.core.config_types import BeliefConfig
from POMDPPlanners.utils.weighted_particle_beliefs import create_belief, WeightedParticleBeliefDiscreteLightDark, WeightedParticleBeliefDiscreteLightDarkFullCoverage, WeightedParticleBeliefContinuousLightDarkFullCoverage, WeightedParticleBeliefSanityPOMDP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

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
    particles = [
        np.array([1, 2]),
        np.array([1, 2]),  # Duplicate
        np.array([3, 4])
    ]
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
        np.array([1, 2])  # Duplicate
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

def test_create_belief_from_config_basic():
    config = BeliefConfig(
        class_name='WeightedParticleBelief',
        params={
            'n_particles': 5,
            'resampling': True,
            'ess_factor': 0.5
        }
    )
    env = TigerPOMDP(discount_factor=0.95)
    belief = create_belief(env, config)
    assert isinstance(belief, WeightedParticleBelief)
    assert len(belief.particles) == 5
    assert np.isclose(np.sum(np.exp(belief.log_weights - np.max(belief.log_weights))), 5)
    assert all(p in env.states for p in belief.particles)
    assert belief.resampling is True
    assert np.isclose(belief.ess_threshold, 2.5)  # ess_threshold = len(particles) * ess_factor = 5 * 0.5

def test_create_belief_particles_and_weights():
    config = BeliefConfig(
        class_name='WeightedParticleBelief',
        params={
            'n_particles': 3,
            'resampling': False,
            'ess_factor': 0.1
        }
    )
    env = TigerPOMDP(discount_factor=0.95)
    belief = create_belief(env, config)
    assert len(belief.particles) == 3
    assert all(p in env.states for p in belief.particles)
    assert np.allclose(np.exp(belief.log_weights - np.max(belief.log_weights)), np.ones(3))
    assert belief.resampling is False
    assert np.isclose(belief.ess_threshold, 0.3)  # ess_threshold = len(particles) * ess_factor = 3 * 0.1

def test_reinvigoration_discrete_light_dark():
    config = BeliefConfig(
        class_name='WeightedParticleBeliefDiscreteLightDark',
        params={
            'n_particles': 5,
            'resampling': True,
            'ess_factor': 0.5,
            'reinvigoration_fraction': 0.2
        }
    )
    env = TigerPOMDP(discount_factor=0.95)
    belief = create_belief(env, config)
    assert isinstance(belief, WeightedParticleBeliefDiscreteLightDark)
    # Call reinvigorate with dummy action and observation
    reinvigorated = belief.reinvigorate(action="up", observation=np.array([0, 0]), pomdp=env, belief=belief)
    assert reinvigorated is belief

def test_reinvigoration_discrete_light_dark_full_coverage():
    config = BeliefConfig(
        class_name='WeightedParticleBeliefDiscreteLightDarkFullCoverage',
        params={
            'n_particles': 5,
            'ess_factor': 0.5,
            'reinvigoration_fraction': 0.05
        }
    )
    env = TigerPOMDP(discount_factor=0.95)
    belief = create_belief(env, config)
    assert isinstance(belief, WeightedParticleBeliefDiscreteLightDarkFullCoverage)
    # Call reinvigorate with dummy action and observation
    reinvigorated = belief.reinvigorate(action="up", observation=np.array([0, 0]), pomdp=env, belief=belief)
    assert reinvigorated is belief

def test_reinvigoration_continuous_light_dark_full_coverage():
    config = BeliefConfig(
        class_name='WeightedParticleBeliefContinuousLightDarkFullCoverage',
        params={
            'n_particles': 5,
            'ess_factor': 0.5,
            'reinvigoration_fraction': 0.05,
            'reinvigoration_cov_matrix': np.eye(2)
        }
    )
    env = TigerPOMDP(discount_factor=0.95)
    belief = create_belief(env, config)
    assert isinstance(belief, WeightedParticleBeliefContinuousLightDarkFullCoverage)
    # Call reinvigorate with dummy action and observation
    reinvigorated = belief.reinvigorate(action="up", observation=np.array([0, 0]), pomdp=env, belief=belief)
    assert reinvigorated is belief

def test_reinvigoration_sanity_pomdp():
    config = BeliefConfig(
        class_name='WeightedParticleBeliefSanityPOMDP',
        params={
            'n_particles': 5,
            'resampling': True,
            'ess_factor': 0.5,
            'reinvigoration_fraction': 0.2
        }
    )
    env = TigerPOMDP(discount_factor=0.95)
    belief = create_belief(env, config)
    assert isinstance(belief, WeightedParticleBeliefSanityPOMDP)
    # Call reinvigorate with dummy action and observation
    reinvigorated = belief.reinvigorate(action="up", observation=np.array([0, 0]), pomdp=env)
    assert isinstance(reinvigorated, WeightedParticleBelief)

# Tests for WeightedParticleBelief.update() method
def test_belief_update_basic():
    """Test basic belief update functionality."""
    from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
    
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
    from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
    
    env = SanityPOMDP()
    particles = [0, 1, 0, 1]
    log_weights = np.array([0.1, 0.1, 0.1, 0.1])  # Non-zero weights
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=True, ess_factor=0.5)
    
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
    from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
    
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
    from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
    
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
    from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
    
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
    from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
    
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
    from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
    
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
    from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
    
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
    from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
    
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
    assert updated_belief1.particles != updated_belief2.particles or \
           not np.array_equal(updated_belief1.log_weights, updated_belief2.log_weights)
