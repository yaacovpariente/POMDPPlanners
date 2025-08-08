import pytest
import numpy as np
from POMDPPlanners.core.belief import Belief, WeightedParticleBelief, WeightedParticleBeliefStateUpdate
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


# Tests for WeightedParticleBeliefStateUpdate class
def test_weighted_particle_belief_state_update_initialization_empty():
    """Test WeightedParticleBeliefStateUpdate initialization with empty lists."""
    belief = WeightedParticleBeliefStateUpdate()
    
    assert belief.particles == []
    assert belief.weights == []
    assert belief.weights_sum == 0


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
    from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
    
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
    from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
    
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
    from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
    
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
    from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
    
    env = SanityPOMDP()
    original_belief = WeightedParticleBeliefStateUpdate(particles=[0], weights=[0.5])
    
    action = 0
    observation = 0
    state = 1
    
    updated_belief = original_belief.update(action=action, observation=observation, pomdp=env, state=state)
    
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
    from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
    
    env = SanityPOMDP()
    original_particles = [0, 1]
    original_weights = [0.3, 0.7]
    original_belief = WeightedParticleBeliefStateUpdate(particles=original_particles, weights=original_weights)
    
    action = 0
    observation = 0
    state = 0
    
    updated_belief = original_belief.update(action=action, observation=observation, pomdp=env, state=state)
    
    # Check that original data is preserved in new belief
    assert updated_belief.particles[:2] == original_particles
    assert updated_belief.weights[:2] == original_weights
    
    # Check that new data is added
    assert updated_belief.particles[2] == state
    assert len(updated_belief.particles) == 3
    assert len(updated_belief.weights) == 3


def test_weighted_particle_belief_state_update_inplace_vs_update_comparison():
    """Test that inplace_update modifies the belief in-place while update returns a new belief."""
    from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
    
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
    assert updated_belief2.weights[1] > 0  # New weight added (should be 1.0 for matching observation)
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
    assert hasattr(belief, 'config_id')
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
    config_ids = [belief1.config_id, belief2.config_id, belief3.config_id, belief4.config_id, belief5.config_id]
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
    from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
    
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
    from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
    
    # Create environment and empty belief (from docstring)
    env = TigerPOMDP(discount_factor=0.95)
    belief = WeightedParticleBeliefStateUpdate(particles=[], weights=[])
    
    # Add states incrementally with observations (from docstring)
    belief.inplace_update(
        action="listen", 
        observation="hear_left", 
        pomdp=env, 
        state="tiger_left"
    )
    belief.inplace_update(
        action="listen", 
        observation="hear_left", 
        pomdp=env, 
        state="tiger_right"
    )
    
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
        action="listen",
        observation="hear_right", 
        pomdp=env,
        state="tiger_right"
    )
    
    # Verify new belief properties
    assert len(new_belief.particles) == 3
    assert len(new_belief.weights) == 3
    assert new_belief.particles == ["tiger_left", "tiger_right", "tiger_right"]
    
    # Verify original belief unchanged
    assert len(belief.particles) == 2


def test_weighted_particle_belief_state_update_update_method_example():
    """Test the update method usage example from WeightedParticleBeliefStateUpdate docstring."""
    from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
    
    # Create a mock environment for testing
    environment = TigerPOMDP(discount_factor=0.95)
    
    # Original belief with 2 particles (from docstring)
    belief = WeightedParticleBeliefStateUpdate(
        particles=["tiger_left", "tiger_right"], 
        weights=[0.7, 0.3]
    )
    
    # Create new belief with additional particle (from docstring)
    new_belief = belief.update(
        action="listen",
        observation="hear_left", 
        pomdp=environment,
        state="tiger_left"
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
    from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
    
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
        weights=[0.85, 0.15, 0.85]  # First and third more likely
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
