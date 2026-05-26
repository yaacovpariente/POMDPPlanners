# SPDX-License-Identifier: MIT

"""Tests for weighted particle belief implementations.

This module tests the weighted particle belief implementations, focusing on:
- Basic belief functionality
- Belief updates
- Belief sampling
- Belief types
"""

import random

import numpy as np
import pytest

from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDPDiscreteActions,
)
from POMDPPlanners.tests.test_utils.metric_invariants_utils import (
    verify_belief_invariants,
)
from POMDPPlanners.utils.weighted_particle_beliefs import (
    WeightedParticleBeliefContinuousLightDarkFullCoverage,
    WeightedParticleBeliefDiscreteLightDark,
    WeightedParticleBeliefDiscreteLightDarkFullCoverage,
)

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


def test_initialization():
    """Test proper initialization of WeightedParticleBeliefDiscreteLightDark

    Purpose: Validates proper initialization of WeightedParticleBeliefDiscreteLightDark with particle weights and reinvigoration parameters

    Given: 10 particles at origin [0,0], uniform log weights, resampling=False, ess_factor=0.5, reinvigoration_fraction=0.2
    When: WeightedParticleBeliefDiscreteLightDark is instantiated
    Then: Correct particles/weights lengths, reinvigoration_fraction=0.2, actions=[up,down,right,left], action_to_vector mappings

    Test type: unit
    """
    # Create test data
    n_particles = 10
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)

    # Initialize belief
    belief = WeightedParticleBeliefDiscreteLightDark(
        particles=particles,
        log_weights=log_weights,
        resampling=False,
        ess_factor=0.5,
        reinvigoration_fraction=0.2,
    )

    # Check attributes
    assert len(belief.particles) == n_particles
    assert len(belief.log_weights) == n_particles
    assert belief.reinvigoration_fraction == 0.2
    assert belief.actions == ["up", "down", "right", "left"]
    assert belief.action_to_vector["up"] == pytest.approx(np.array([0, 1]))
    verify_belief_invariants(belief, expected_n_particles=n_particles)


def test_reinvigoration():
    """Test reinvigoration functionality

    Purpose: Validates that reinvigorate method correctly updates particle belief with new particles based on action and observation

    Given: WeightedParticleBeliefDiscreteLightDark with 10 particles, ContinuousLightDarkPOMDP environment, action="up", observation=[0,1]
    When: reinvigorate method is called with these parameters
    Then: Returns new belief with same particle count and at least reinvigoration_fraction*n_particles new particles added

    Test type: unit
    """
    # Create test data
    n_particles = 10
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)

    # Initialize belief
    belief = WeightedParticleBeliefDiscreteLightDark(
        particles=particles,
        log_weights=log_weights,
        resampling=False,
        ess_factor=0.5,
        reinvigoration_fraction=0.2,
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
    verify_belief_invariants(reinvigorated_belief, expected_n_particles=n_particles)


def test_invalid_initialization():
    """Test initialization with invalid parameters

    Purpose: Validates that WeightedParticleBeliefDiscreteLightDark raises appropriate errors for invalid constructor parameters

    Given: Mismatched particles (5) and weights (10) arrays, or invalid reinvigoration_fraction (1.5) outside [0,1] range
    When: WeightedParticleBeliefDiscreteLightDark constructor is called with invalid parameters
    Then: ValueError is raised indicating parameter validation failure

    Test type: unit
    """
    # Test with mismatched particles and weights
    with pytest.raises(ValueError):
        WeightedParticleBeliefDiscreteLightDark(
            particles=[np.array([0, 0]) for _ in range(5)],
            log_weights=np.zeros(10),
            resampling=False,
        )

    # Test with invalid reinvigoration fraction
    with pytest.raises(ValueError):
        WeightedParticleBeliefDiscreteLightDark(
            particles=[np.array([0, 0]) for _ in range(10)],
            log_weights=np.zeros(10),
            resampling=False,
            reinvigoration_fraction=1.5,  # Should be between 0 and 1
        )


def test_action_to_vector_mapping():
    """Test action to vector mapping

    Purpose: Validates that action_to_vector dictionary correctly maps discrete actions to 2D movement vectors

    Given: WeightedParticleBeliefDiscreteLightDark with single particle and non-zero log weight
    When: action_to_vector mappings are accessed for valid and invalid actions
    Then: Correct vectors returned (up=[0,1], down=[0,-1], right=[1,0], left=[-1,0]) and KeyError for invalid actions

    Test type: unit
    """
    # Create test data with non-zero log weights
    n_particles = 1
    particles = [np.array([0, 0])]
    # Use a non-zero log weight
    log_weights = np.array([-1.0])  # log(1/e) ≈ -1.0

    belief = WeightedParticleBeliefDiscreteLightDark(
        particles=particles, log_weights=log_weights, resampling=False
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
    """Test proper initialization of WeightedParticleBeliefDiscreteLightDarkFullCoverage

    Purpose: Validates proper initialization of WeightedParticleBeliefDiscreteLightDarkFullCoverage with full coverage reinvigoration parameters

    Given: 10 particles at origin, uniform log weights, ess_factor=0.5, reinvigoration_fraction=0.05
    When: WeightedParticleBeliefDiscreteLightDarkFullCoverage is instantiated
    Then: Correct attributes set including reinvigoration_particles_weights_sum=0.05, resampling=False, action mappings

    Test type: unit
    """
    # Create test data
    n_particles = 10
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)

    # Initialize belief
    belief = WeightedParticleBeliefDiscreteLightDarkFullCoverage(
        particles=particles,
        log_weights=log_weights,
        ess_factor=0.5,
        reinvigoration_fraction=0.05,
    )

    # Check attributes
    assert len(belief.particles) == n_particles
    assert len(belief.log_weights) == n_particles
    assert belief.reinvigoration_particles_weights_sum == 0.05
    assert belief.actions == ["up", "down", "right", "left"]
    assert belief.action_to_vector["up"] == pytest.approx(np.array([0, 1]))
    assert not belief.resampling  # Should be False as specified in __init__
    verify_belief_invariants(belief, expected_n_particles=n_particles)


def test_full_coverage_reinvigoration():
    """Test reinvigoration functionality of WeightedParticleBeliefDiscreteLightDarkFullCoverage

    Purpose: Validates that full coverage reinvigoration adds systematic state coverage particles based on all possible actions

    Given: WeightedParticleBeliefDiscreteLightDarkFullCoverage with 10 particles, action="up", observation=[0,1]
    When: reinvigorate method executes full coverage strategy
    Then: Last n_states particles match expected states (observation + action_vectors for each action, plus observation itself)

    Test type: unit
    """
    # Create test data
    n_particles = 10
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)

    # Initialize belief
    belief = WeightedParticleBeliefDiscreteLightDarkFullCoverage(
        particles=particles,
        log_weights=log_weights,
        ess_factor=0.5,
        reinvigoration_fraction=0.05,
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
        assert np.array_equal(reinvigorated_belief.particles[-(i + 1)], expected_states[-(i + 1)])
    verify_belief_invariants(reinvigorated_belief, expected_n_particles=n_particles)


def test_full_coverage_invalid_initialization():
    """Test initialization with invalid parameters for WeightedParticleBeliefDiscreteLightDarkFullCoverage

    Purpose: Validates proper initialization of full coverage invalid

    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes

    Test type: unit
    """
    # Test with mismatched particles and weights
    with pytest.raises(ValueError):
        WeightedParticleBeliefDiscreteLightDarkFullCoverage(
            particles=[np.array([0, 0]) for _ in range(5)],
            log_weights=np.zeros(10),
            reinvigoration_fraction=0.05,
        )

    # Test with invalid reinvigoration_particles_weights_sum
    with pytest.raises(ValueError):
        WeightedParticleBeliefDiscreteLightDarkFullCoverage(
            particles=[np.array([0, 0]) for _ in range(10)],
            log_weights=np.zeros(10),
            reinvigoration_fraction=1.5,  # Should be between 0 and 1
        )


def test_full_coverage_resampling():
    """Test that resampling is performed during reinvigoration when ESS is below threshold

    Purpose: Validates that full coverage reinvigoration triggers resampling when Effective Sample Size falls below threshold due to degenerate weights

    Given: WeightedParticleBeliefDiscreteLightDarkFullCoverage with degenerate weights (one particle has all weight), ess_factor=0.5
    When: reinvigorate is called with ESS below threshold
    Then: Resampling occurs improving ESS above threshold while maintaining expected state particles in correct positions

    Test type: unit
    """
    # Create test data with degenerate weights
    n_particles = 10
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    # Create degenerate weights (all weight on one particle)
    log_weights = np.array([0.0] + [-100.0] * (n_particles - 1))

    belief = WeightedParticleBeliefDiscreteLightDarkFullCoverage(
        particles=particles,
        log_weights=log_weights,
        ess_factor=0.5,
        reinvigoration_fraction=0.05,
    )

    # Check that initial ESS is below threshold (degenerate weights)
    initial_ess = 1 / np.sum(np.square(belief.normalized_weights))
    assert initial_ess < belief.ess_threshold

    # Create ContinuousLightDarkPOMDP environment
    env = ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)

    # Test reinvigoration
    action = "up"
    observation = np.array([0, 1])
    reinvigorated_belief = belief.reinvigorate(action, observation, env, belief)

    # Check that resampling occurred (weights should be more uniform)
    effective_sample_size = 1 / np.sum(np.square(reinvigorated_belief.normalized_weights))
    # After resampling, ESS should be improved but may still be below threshold
    assert effective_sample_size >= belief.ess_threshold

    # Check that the last n_states particles are still the expected states
    n_states = len(belief.actions) + 1
    expected_states = [observation + belief.action_to_vector[action] for action in belief.actions]
    expected_states.append(observation)

    for i in range(n_states):
        assert np.array_equal(reinvigorated_belief.particles[-(i + 1)], expected_states[-(i + 1)])


def test_continuous_full_coverage_initialization():
    """Test proper initialization of WeightedParticleBeliefContinuousLightDarkFullCoverage

    Purpose: Validates proper initialization of continuous full coverage

    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes

    Test type: unit
    """
    # Create test data
    n_particles = 10
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)
    cov_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])

    # Initialize belief
    belief = WeightedParticleBeliefContinuousLightDarkFullCoverage(
        particles=particles,
        log_weights=log_weights,
        ess_factor=0.5,
        reinvigoration_fraction=0.05,
        reinvigoration_cov_matrix=cov_matrix,
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
    """Test reinvigoration functionality of WeightedParticleBeliefContinuousLightDarkFullCoverage

    Purpose: Validates that continuous full coverage reinvigoration generates particles within environment bounds using covariance matrix sampling

    Given: WeightedParticleBeliefContinuousLightDarkFullCoverage with 10 particles, covariance matrix, action="up", observation=[0,1]
    When: reinvigorate method samples new particles using continuous distributions
    Then: Reinvigorated particles are within environment bounds [0, grid_size], have correct 2D shape, and particle count preserved

    Test type: unit
    """
    # Create test data
    n_particles = 10
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)
    cov_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])

    # Initialize belief
    belief = WeightedParticleBeliefContinuousLightDarkFullCoverage(
        particles=particles,
        log_weights=log_weights,
        ess_factor=0.5,
        reinvigoration_fraction=0.05,
        reinvigoration_cov_matrix=cov_matrix,
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
    """Test initialization with invalid parameters for WeightedParticleBeliefContinuousLightDarkFullCoverage

    Purpose: Validates proper initialization of continuous full coverage invalid

    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes

    Test type: unit
    """
    # Test with mismatched particles and weights
    with pytest.raises(ValueError):
        WeightedParticleBeliefContinuousLightDarkFullCoverage(
            particles=[np.array([0, 0]) for _ in range(5)],
            log_weights=np.zeros(10),
            reinvigoration_fraction=0.05,
        )

    # Test with invalid reinvigoration_particles_weights_sum
    with pytest.raises(ValueError):
        WeightedParticleBeliefContinuousLightDarkFullCoverage(
            particles=[np.array([0, 0]) for _ in range(10)],
            log_weights=np.zeros(10),
            reinvigoration_fraction=1.5,  # Should be between 0 and 1
        )

    # Test with invalid covariance matrix shape
    with pytest.raises(ValueError):
        WeightedParticleBeliefContinuousLightDarkFullCoverage(
            particles=[np.array([0, 0]) for _ in range(10)],
            log_weights=np.zeros(10),
            reinvigoration_cov_matrix=np.eye(3),  # Should be 2x2
        )


def test_continuous_full_coverage_resampling():
    """Test that resampling is performed during reinvigoration when ESS is below threshold

    Purpose: Validates that continuous full coverage reinvigoration performs resampling when ESS drops below threshold while maintaining particle bounds

    Given: WeightedParticleBeliefContinuousLightDarkFullCoverage with degenerate weights, covariance matrix, ess_factor=0.5
    When: reinvigorate is called with ESS below threshold triggering resampling
    Then: ESS improves above threshold and reinvigorated particles remain within environment bounds [0, grid_size] with correct shapes

    Test type: unit
    """
    # Create test data with degenerate weights
    n_particles = 10
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    # Create degenerate weights (all weight on one particle)
    log_weights = np.array([0.0] + [-100.0] * (n_particles - 1))
    cov_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])

    belief = WeightedParticleBeliefContinuousLightDarkFullCoverage(
        particles=particles,
        log_weights=log_weights,
        ess_factor=0.5,
        reinvigoration_fraction=0.05,
        reinvigoration_cov_matrix=cov_matrix,
    )

    # Check that initial ESS is below threshold (degenerate weights)
    initial_ess = 1 / np.sum(np.square(belief.normalized_weights))
    assert initial_ess < belief.ess_threshold

    # Create ContinuousLightDarkPOMDP environment
    env = ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)

    # Test reinvigoration
    action = "up"
    observation = np.array([0, 1])
    reinvigorated_belief = belief.reinvigorate(action, observation, env, belief)

    # Check that resampling occurred (weights should be more uniform)
    effective_sample_size = 1 / np.sum(np.square(reinvigorated_belief.normalized_weights))
    # After resampling, ESS should be improved but may still be below threshold
    assert effective_sample_size >= belief.ess_threshold

    # Check that reinvigorated particles are within bounds and have correct shape
    n_reinvigorate = int(belief.reinvigoration_particles_weights_sum * n_particles)
    reinvigorated_particles = reinvigorated_belief.particles[-n_reinvigorate:]

    for particle in reinvigorated_particles:
        assert np.all(particle >= 0)
        assert np.all(particle <= env.grid_size)
        assert particle.shape == (2,)


def test_continuous_full_coverage_gmm_sampling():
    """Test that GMM sampling produces expected distribution of particles

    Purpose: Validates that continuous full coverage uses Gaussian Mixture Model sampling to cluster reinvigorated particles around expected action centers

    Given: WeightedParticleBeliefContinuousLightDarkFullCoverage with 1000 particles, reinvigoration_fraction=0.2, tight covariance matrix, center observation [5,5]
    When: reinvigoration with action="up" generates GMM-sampled particles
    Then: Reinvigorated particles cluster around expected centers (observation + action_vectors) with >10% within 2 std deviations of each center

    Test type: unit
    """
    # Create test data
    n_particles = 1000  # Use more particles for better statistical testing
    particles = [np.array([0, 0]) for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)
    cov_matrix = np.array([[0.1, 0], [0, 0.1]])  # Small covariance for tight clusters

    belief = WeightedParticleBeliefContinuousLightDarkFullCoverage(
        particles=particles,
        log_weights=log_weights,
        ess_factor=0.5,
        reinvigoration_fraction=0.2,  # Larger fraction for better statistics
        reinvigoration_cov_matrix=cov_matrix,
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


def test_continuous_full_coverage_particle_type_preservation_after_update():
    """Test that particles maintain numpy array type and shape after belief update.

    Purpose: Validates that WeightedParticleBeliefContinuousLightDarkFullCoverage maintains particle data types and shapes after update

    Given: WeightedParticleBeliefContinuousLightDarkFullCoverage with numpy array particles of shape (2,) and ContinuousLightDarkPOMDP environment
    When: belief.update() is called with valid action, observation, and environment
    Then: All particles in updated belief remain numpy arrays with original shape (2,), preserving data type consistency

    Test type: unit
    """
    # Create test data with numpy array particles
    n_particles = 20
    particles = [np.array([1.0, 2.0], dtype=np.float64) for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)

    # Initialize belief
    belief = WeightedParticleBeliefContinuousLightDarkFullCoverage(
        particles=particles,
        log_weights=log_weights,
        ess_factor=0.5,
        reinvigoration_fraction=0.1,
        reinvigoration_cov_matrix=np.eye(2) * 0.1,
    )

    # Create environment
    env = ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)

    # Verify original particle types and shapes
    for particle in belief.particles:
        assert isinstance(
            particle, np.ndarray
        ), f"Original particle should be numpy array, got {type(particle)}"
        assert particle.shape == (
            2,
        ), f"Original particle should have shape (2,), got {particle.shape}"

    # Perform belief update
    action = "right"
    observation = np.array([3.0, 4.0])
    updated_belief = belief.update(action=action, observation=observation, pomdp=env)

    # Verify all particles maintain numpy array type and correct shape after update
    for i, particle in enumerate(updated_belief.particles):
        assert isinstance(
            particle, np.ndarray
        ), f"Updated particle {i} should be numpy array, got {type(particle)}"
        assert particle.shape == (
            2,
        ), f"Updated particle {i} should have shape (2,), got {particle.shape}"
        assert particle.dtype in [
            np.float64,
            np.float32,
        ], f"Updated particle {i} should have float dtype, got {particle.dtype}"

    # Verify particle count is preserved
    assert (
        len(updated_belief.particles) == n_particles
    ), f"Particle count should be preserved, got {len(updated_belief.particles)} instead of {n_particles}"

    # Verify log_weights shape is preserved
    assert updated_belief.log_weights.shape == (
        n_particles,
    ), f"Log weights shape should be ({n_particles},), got {updated_belief.log_weights.shape}"
    verify_belief_invariants(updated_belief, expected_n_particles=n_particles)
