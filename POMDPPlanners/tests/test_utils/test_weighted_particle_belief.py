import numpy as np
import pytest
from POMDPPlanners.utils.weighted_particle_beliefs import WeightedParticleBeliefDiscreteLightDark
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

def test_initialization():
    """Test proper initialization of WeightedParticleBeliefDiscreteLightDark"""
    # Create test data
    n_particles = 10
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)
    
    # Initialize belief
    belief = WeightedParticleBeliefDiscreteLightDark(
        particles=particles,
        log_weights=log_weights,
        resampling=False,
        ess_threshold=0.5,
        reinvigoration_fraction=0.2
    )
    
    # Check attributes
    assert len(belief.particles) == n_particles
    assert len(belief.log_weights) == n_particles
    assert belief.reinvigoration_fraction == 0.2
    assert belief.actions == ["up", "down", "right", "left"]
    assert belief.action_to_vector["up"] == pytest.approx(np.array([0, 1]))

def test_reinvigoration():
    """Test reinvigoration functionality"""
    # Create test data
    n_particles = 10
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)
    
    # Initialize belief
    belief = WeightedParticleBeliefDiscreteLightDark(
        particles=particles,
        log_weights=log_weights,
        resampling=False,
        ess_threshold=0.5,
        reinvigoration_fraction=0.2
    )
    
    # Create TigerPOMDP environment
    env = TigerPOMDP(discount_factor=0.95)
    
    # Test reinvigoration
    action = "up"
    observation = np.array([0, 1])
    reinvigorated_belief = belief.reinvigorate(action, observation, env, belief)
    
    # Check that reinvigoration occurred
    assert len(reinvigorated_belief.particles) == n_particles
    assert len(reinvigorated_belief.log_weights) == n_particles
    
    # Check that some particles were reinvigorated
    n_reinvigorate = int(belief.reinvigoration_fraction * n_particles)
    assert n_reinvigorate > 0

def test_invalid_initialization():
    """Test initialization with invalid parameters"""
    # Test with mismatched particles and weights
    with pytest.raises(AssertionError):
        WeightedParticleBeliefDiscreteLightDark(
            particles=[np.array([0, 0]) for _ in range(5)],
            log_weights=np.zeros(10),
            resampling=False
        )
    
    # Test with invalid reinvigoration fraction
    with pytest.raises(AssertionError):
        WeightedParticleBeliefDiscreteLightDark(
            particles=[np.array([0, 0]) for _ in range(10)],
            log_weights=np.zeros(10),
            resampling=False,
            reinvigoration_fraction=1.5  # Should be between 0 and 1
        )

def test_action_to_vector_mapping():
    """Test action to vector mapping"""
    # Create test data with non-zero log weights
    n_particles = 1
    particles = [np.array([0, 0])]
    # Use a non-zero log weight
    log_weights = np.array([-1.0])  # log(1/e) ≈ -1.0
    
    belief = WeightedParticleBeliefDiscreteLightDark(
        particles=particles,
        log_weights=log_weights,
        resampling=False
    )
    
    # Test all actions
    assert belief.action_to_vector["up"] == pytest.approx(np.array([0, 1]))
    assert belief.action_to_vector["down"] == pytest.approx(np.array([0, -1]))
    assert belief.action_to_vector["right"] == pytest.approx(np.array([1, 0]))
    assert belief.action_to_vector["left"] == pytest.approx(np.array([-1, 0]))
    
    # Test invalid action
    with pytest.raises(KeyError):
        _ = belief.action_to_vector["invalid_action"]
