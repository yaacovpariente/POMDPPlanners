import numpy as np
import pytest
from POMDPPlanners.utils.weighted_particle_beliefs import (
    WeightedParticleBeliefDiscreteLightDark,
    WeightedParticleBeliefDiscreteLightDarkFullCoverage,
    WeightedParticleBeliefContinuousLightDarkFullCoverage
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import ContinuousLightDarkPOMDPDiscreteActions

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
    
    # Create ContinuousLightDarkPOMDP environment
    env = ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)
    
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

def test_full_coverage_initialization():
    """Test proper initialization of WeightedParticleBeliefDiscreteLightDarkFullCoverage"""
    # Create test data
    n_particles = 10
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)
    
    # Initialize belief
    belief = WeightedParticleBeliefDiscreteLightDarkFullCoverage(
        particles=particles,
        log_weights=log_weights,
        ess_threshold=0.5,
        reinvigoration_fraction=0.05
    )
    
    # Check attributes
    assert len(belief.particles) == n_particles
    assert len(belief.log_weights) == n_particles
    assert belief.reinvigoration_particles_weights_sum == 0.05
    assert belief.actions == ["up", "down", "right", "left"]
    assert belief.action_to_vector["up"] == pytest.approx(np.array([0, 1]))
    assert not belief.resampling  # Should be False as specified in __init__

def test_full_coverage_reinvigoration():
    """Test reinvigoration functionality of WeightedParticleBeliefDiscreteLightDarkFullCoverage"""
    # Create test data
    n_particles = 10
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)
    
    # Initialize belief
    belief = WeightedParticleBeliefDiscreteLightDarkFullCoverage(
        particles=particles,
        log_weights=log_weights,
        ess_threshold=0.5,
        reinvigoration_fraction=0.05
    )
    
    # Create ContinuousLightDarkPOMDP environment
    env = ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)
    
    # Test reinvigoration
    action = "up"
    observation = np.array([0, 1])
    reinvigorated_belief = belief.reinvigorate(action, observation, env, belief)
    
    # Check that reinvigoration occurred
    assert len(reinvigorated_belief.particles) == n_particles
    assert len(reinvigorated_belief.log_weights) == n_particles
    
    # Check that the last n_states particles are the expected states
    n_states = len(belief.actions) + 1  # +1 for the observation state
    expected_states = [observation + belief.action_to_vector[action] for action in belief.actions]
    expected_states.append(observation)
    
    for i in range(n_states):
        assert np.array_equal(reinvigorated_belief.particles[-(i+1)], expected_states[-(i+1)])

def test_full_coverage_invalid_initialization():
    """Test initialization with invalid parameters for WeightedParticleBeliefDiscreteLightDarkFullCoverage"""
    # Test with mismatched particles and weights
    with pytest.raises(AssertionError):
        WeightedParticleBeliefDiscreteLightDarkFullCoverage(
            particles=[np.array([0, 0]) for _ in range(5)],
            log_weights=np.zeros(10),
            reinvigoration_fraction=0.05
        )
    
    # Test with invalid reinvigoration_particles_weights_sum
    with pytest.raises(AssertionError):
        WeightedParticleBeliefDiscreteLightDarkFullCoverage(
            particles=[np.array([0, 0]) for _ in range(10)],
            log_weights=np.zeros(10),
            reinvigoration_fraction=1.5  # Should be between 0 and 1
        )

def test_full_coverage_resampling():
    """Test that resampling is performed during reinvigoration"""
    # Create test data with degenerate weights
    n_particles = 10
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    # Create degenerate weights (all weight on one particle)
    log_weights = np.array([0.0] + [-100.0] * (n_particles - 1))
    
    belief = WeightedParticleBeliefDiscreteLightDarkFullCoverage(
        particles=particles,
        log_weights=log_weights,
        ess_threshold=0.5,
        reinvigoration_fraction=0.05
    )
    
    # Create ContinuousLightDarkPOMDP environment
    env = ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)
    
    # Test reinvigoration
    action = "up"
    observation = np.array([0, 1])
    reinvigorated_belief = belief.reinvigorate(action, observation, env, belief)
    
    # Check that resampling occurred (weights should be more uniform)
    effective_sample_size = 1 / np.sum(np.square(reinvigorated_belief.normalized_weights))
    assert effective_sample_size > belief.ess_threshold
    
    # Check that the last n_states particles are still the expected states
    n_states = len(belief.actions) + 1
    expected_states = [observation + belief.action_to_vector[action] for action in belief.actions]
    expected_states.append(observation)
    
    for i in range(n_states):
        assert np.array_equal(reinvigorated_belief.particles[-(i+1)], expected_states[-(i+1)])

def test_continuous_full_coverage_initialization():
    """Test proper initialization of WeightedParticleBeliefContinuousLightDarkFullCoverage"""
    # Create test data
    n_particles = 10
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)
    cov_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    
    # Initialize belief
    belief = WeightedParticleBeliefContinuousLightDarkFullCoverage(
        particles=particles,
        log_weights=log_weights,
        ess_threshold=0.5,
        reinvigoration_fraction=0.05,
        reinvigoration_cov_matrix=cov_matrix
    )
    
    # Check attributes
    assert len(belief.particles) == n_particles
    assert len(belief.log_weights) == n_particles
    assert belief.reinvigoration_particles_weights_sum == 0.05
    assert np.array_equal(belief.reinvigoration_cov_matrix, cov_matrix)
    assert belief.actions == ["up", "down", "right", "left"]
    assert belief.action_to_vector["up"] == pytest.approx(np.array([0, 1]))
    assert not belief.resampling  # Should be False as specified in __init__

def test_continuous_full_coverage_reinvigoration():
    """Test reinvigoration functionality of WeightedParticleBeliefContinuousLightDarkFullCoverage"""
    # Create test data
    n_particles = 10
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)
    cov_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    
    # Initialize belief
    belief = WeightedParticleBeliefContinuousLightDarkFullCoverage(
        particles=particles,
        log_weights=log_weights,
        ess_threshold=0.5,
        reinvigoration_fraction=0.05,
        reinvigoration_cov_matrix=cov_matrix
    )
    
    # Create ContinuousLightDarkPOMDP environment
    env = ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)
    
    # Test reinvigoration
    action = "up"
    observation = np.array([0, 1])
    reinvigorated_belief = belief.reinvigorate(action, observation, env, belief)
    
    # Check that reinvigoration occurred
    assert len(reinvigorated_belief.particles) == n_particles
    assert len(reinvigorated_belief.log_weights) == n_particles
    
    # Check that the reinvigorated particles are within bounds
    n_reinvigorate = int(belief.reinvigoration_particles_weights_sum * n_particles)
    reinvigorated_particles = reinvigorated_belief.particles[-n_reinvigorate:]
    
    for particle in reinvigorated_particles:
        assert np.all(particle >= 0)  # Check lower bound
        assert np.all(particle <= env.grid_size)  # Check upper bound
        assert particle.shape == (2,)  # Check shape

def test_continuous_full_coverage_invalid_initialization():
    """Test initialization with invalid parameters for WeightedParticleBeliefContinuousLightDarkFullCoverage"""
    # Test with mismatched particles and weights
    with pytest.raises(AssertionError):
        WeightedParticleBeliefContinuousLightDarkFullCoverage(
            particles=[np.array([0, 0]) for _ in range(5)],
            log_weights=np.zeros(10),
            reinvigoration_fraction=0.05
        )
    
    # Test with invalid reinvigoration_particles_weights_sum
    with pytest.raises(AssertionError):
        WeightedParticleBeliefContinuousLightDarkFullCoverage(
            particles=[np.array([0, 0]) for _ in range(10)],
            log_weights=np.zeros(10),
            reinvigoration_fraction=1.5  # Should be between 0 and 1
        )
    
    # Test with invalid covariance matrix shape
    with pytest.raises(AssertionError):
        WeightedParticleBeliefContinuousLightDarkFullCoverage(
            particles=[np.array([0, 0]) for _ in range(10)],
            log_weights=np.zeros(10),
            reinvigoration_cov_matrix=np.eye(3)  # Should be 2x2
        )

def test_continuous_full_coverage_resampling():
    """Test that resampling is performed during reinvigoration"""
    # Create test data with degenerate weights
    n_particles = 10
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    # Create degenerate weights (all weight on one particle)
    log_weights = np.array([0.0] + [-100.0] * (n_particles - 1))
    cov_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    
    belief = WeightedParticleBeliefContinuousLightDarkFullCoverage(
        particles=particles,
        log_weights=log_weights,
        ess_threshold=0.5,
        reinvigoration_fraction=0.05,
        reinvigoration_cov_matrix=cov_matrix
    )
    
    # Create ContinuousLightDarkPOMDP environment
    env = ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)
    
    # Test reinvigoration
    action = "up"
    observation = np.array([0, 1])
    reinvigorated_belief = belief.reinvigorate(action, observation, env, belief)
    
    # Check that resampling occurred (weights should be more uniform)
    effective_sample_size = 1 / np.sum(np.square(reinvigorated_belief.normalized_weights))
    assert effective_sample_size > belief.ess_threshold
    
    # Check that reinvigorated particles are within bounds and have correct shape
    n_reinvigorate = int(belief.reinvigoration_particles_weights_sum * n_particles)
    reinvigorated_particles = reinvigorated_belief.particles[-n_reinvigorate:]
    
    for particle in reinvigorated_particles:
        assert np.all(particle >= 0)
        assert np.all(particle <= env.grid_size)
        assert particle.shape == (2,)

def test_continuous_full_coverage_gmm_sampling():
    """Test that GMM sampling produces expected distribution of particles"""
    # Create test data
    n_particles = 1000  # Use more particles for better statistical testing
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)
    cov_matrix = np.array([[0.1, 0], [0, 0.1]])  # Small covariance for tight clusters
    
    belief = WeightedParticleBeliefContinuousLightDarkFullCoverage(
        particles=particles,
        log_weights=log_weights,
        ess_threshold=0.5,
        reinvigoration_fraction=0.2,  # Larger fraction for better statistics
        reinvigoration_cov_matrix=cov_matrix
    )
    
    # Create ContinuousLightDarkPOMDP environment
    env = ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)
    
    # Test reinvigoration
    action = "up"
    observation = np.array([5, 5])  # Center observation for better testing
    reinvigorated_belief = belief.reinvigorate(action, observation, env, belief)
    
    # Get reinvigorated particles
    n_reinvigorate = int(belief.reinvigoration_particles_weights_sum * n_particles)
    reinvigorated_particles = np.array(reinvigorated_belief.particles[-n_reinvigorate:])
    
    # Check that particles are clustered around expected centers
    expected_centers = [observation + belief.action_to_vector[action] for action in belief.actions]
    expected_centers.append(observation)
    expected_centers = np.array(expected_centers)
    
    # For each center, check that there are particles nearby
    for center in expected_centers:
        distances = np.linalg.norm(reinvigorated_particles - center, axis=1)
        # Check that at least 10% of particles are within 2 standard deviations
        assert np.mean(distances < 2 * np.sqrt(0.1)) > 0.1
