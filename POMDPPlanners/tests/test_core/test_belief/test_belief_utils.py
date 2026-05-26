# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for belief utility functions.

This module tests the belief utility functions, focusing on:
- is_terminal_belief with various belief types
- is_terminal_particle_belief
"""

import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import (
    Belief,
    UnweightedParticleBeliefStateUpdate,
    WeightedParticleBelief,
    WeightedParticleBeliefStateUpdate,
    is_terminal_belief,
)
from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.laser_tag_pomdp import LaserTagPOMDP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


# Tests for is_terminal_belief function
def test_is_terminal_belief_all_terminal_particles():
    """Test is_terminal_belief returns True when all particles are terminal."""
    # ARRANGE: Create LaserTag environment
    env = LaserTagPOMDP(discount_factor=0.95)

    # Create terminal states
    terminal_state1 = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    terminal_state2 = np.array([2.0, 2.0, 3.0, 3.0, 1.0])
    terminal_state3 = np.array([4.0, 4.0, 5.0, 5.0, 1.0])

    # Create belief with all terminal particles
    particles = [terminal_state1, terminal_state2, terminal_state3]
    log_weights = np.log(np.ones(3) / 3)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    # ACT: Check if belief is terminal
    result = is_terminal_belief(belief, env)

    # ASSERT: Should return True
    assert result is True


def test_is_terminal_belief_no_terminal_particles():
    """Test is_terminal_belief returns False when no particles are terminal."""
    # ARRANGE: Create LaserTag environment
    env = LaserTagPOMDP(discount_factor=0.95)

    # Create non-terminal states
    non_terminal_state1 = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    non_terminal_state2 = np.array([2.0, 2.0, 3.0, 3.0, 0.0])
    non_terminal_state3 = np.array([4.0, 4.0, 5.0, 5.0, 0.0])

    # Create belief with all non-terminal particles
    particles = [non_terminal_state1, non_terminal_state2, non_terminal_state3]
    log_weights = np.log(np.ones(3) / 3)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    # ACT: Check if belief is terminal
    result = is_terminal_belief(belief, env)

    # ASSERT: Should return False
    assert result is False


def test_is_terminal_belief_mixed_particles():
    """Test is_terminal_belief returns False when some particles are terminal."""
    # ARRANGE: Create LaserTag environment
    env = LaserTagPOMDP(discount_factor=0.95)

    # Create mixed states (some terminal, some not)
    terminal_state = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    non_terminal_state1 = np.array([2.0, 2.0, 3.0, 3.0, 0.0])
    non_terminal_state2 = np.array([4.0, 4.0, 5.0, 5.0, 0.0])

    # Create belief with mixed particles
    particles = [terminal_state, non_terminal_state1, non_terminal_state2]
    log_weights = np.log(np.ones(3) / 3)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    # ACT: Check if belief is terminal
    result = is_terminal_belief(belief, env)

    # ASSERT: Should return False (not all particles are terminal)
    assert result is False


def test_is_terminal_belief_single_terminal_particle():
    """Test is_terminal_belief with single terminal particle."""
    # ARRANGE: Create LaserTag environment
    env = LaserTagPOMDP(discount_factor=0.95)

    # Create single terminal state
    terminal_state = np.array([0.0, 0.0, 1.0, 1.0, 1.0])

    # Create belief with single terminal particle
    particles = [terminal_state]
    log_weights = np.array([0.1])  # Small positive log weight
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    # ACT: Check if belief is terminal
    result = is_terminal_belief(belief, env)

    # ASSERT: Should return True
    assert result is True


def test_is_terminal_belief_single_non_terminal_particle():
    """Test is_terminal_belief with single non-terminal particle."""
    # ARRANGE: Create LaserTag environment
    env = LaserTagPOMDP(discount_factor=0.95)

    # Create single non-terminal state
    non_terminal_state = np.array([0.0, 0.0, 1.0, 1.0, 0.0])

    # Create belief with single non-terminal particle
    particles = [non_terminal_state]
    log_weights = np.array([0.1])  # Small positive log weight
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    # ACT: Check if belief is terminal
    result = is_terminal_belief(belief, env)

    # ASSERT: Should return False
    assert result is False


def test_is_terminal_belief_empty_belief():
    """Test is_terminal_belief with empty belief."""
    # ARRANGE: Create LaserTag environment
    env = LaserTagPOMDP(discount_factor=0.95)

    # Create empty belief using WeightedParticleBeliefStateUpdate (which allows empty)
    belief = WeightedParticleBeliefStateUpdate(particles=[], weights=[])

    # ACT: Check if belief is terminal
    result = is_terminal_belief(belief, env)

    # ASSERT: Should return True (all particles in empty set are terminal)
    assert result is True


def test_is_terminal_belief_with_weighted_particle_belief_state_update():
    """Test is_terminal_belief with WeightedParticleBeliefStateUpdate."""
    # ARRANGE: Create LaserTag environment
    env = LaserTagPOMDP(discount_factor=0.95)

    # Create terminal and non-terminal states
    terminal_state = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    non_terminal_state = np.array([2.0, 2.0, 3.0, 3.0, 0.0])

    # Create WeightedParticleBeliefStateUpdate with mixed particles
    particles = [terminal_state, non_terminal_state]
    weights = [0.7, 0.3]
    belief = WeightedParticleBeliefStateUpdate(particles=particles, weights=weights)

    # ACT: Check if belief is terminal
    result = is_terminal_belief(belief, env)

    # ASSERT: Should return False (not all particles are terminal)
    assert result is False


def test_is_terminal_belief_with_unweighted_particle_belief_state_update():
    """Test is_terminal_belief with UnweightedParticleBeliefStateUpdate."""
    # ARRANGE: Create LaserTag environment
    env = LaserTagPOMDP(discount_factor=0.95)

    # Create all terminal states
    terminal_state1 = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    terminal_state2 = np.array([2.0, 2.0, 3.0, 3.0, 1.0])

    # Create UnweightedParticleBeliefStateUpdate with all terminal particles
    particles = [terminal_state1, terminal_state2]
    belief = UnweightedParticleBeliefStateUpdate(particles=particles)

    # ACT: Check if belief is terminal
    result = is_terminal_belief(belief, env)

    # ASSERT: Should return True
    assert result is True


def test_is_terminal_belief_with_different_environments():
    """Test is_terminal_belief with different environment types."""
    # ARRANGE: Create different environments
    laser_tag_env = LaserTagPOMDP(discount_factor=0.95)
    tiger_env = TigerPOMDP(discount_factor=0.95)

    # Create LaserTag terminal state
    laser_tag_terminal = np.array([0.0, 0.0, 1.0, 1.0, 1.0])

    # Create belief with LaserTag state
    particles = [laser_tag_terminal]
    log_weights = np.array([0.1])  # Small positive log weight
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    # ACT & ASSERT: Test with LaserTag environment
    result_laser_tag = is_terminal_belief(belief, laser_tag_env)
    assert result_laser_tag is True

    # ACT & ASSERT: Test with Tiger environment (should handle gracefully)
    # Tiger environment expects different state types, but function should still work
    result_tiger = is_terminal_belief(belief, tiger_env)
    # The result depends on how TigerPOMDP.is_terminal handles LaserTagState
    # This tests the robustness of the function


def test_is_terminal_belief_large_particle_set():
    """Test is_terminal_belief with large number of particles."""
    # ARRANGE: Create LaserTag environment
    env = LaserTagPOMDP(discount_factor=0.95)

    # Create large set of terminal particles
    particles = []
    for i in range(100):
        terminal_state = np.array(
            [float(i % 7), float(i % 11), float((i + 1) % 7), float((i + 1) % 11), 1.0]
        )
        particles.append(terminal_state)

    log_weights = np.log(np.ones(100) / 100)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    # ACT: Check if belief is terminal
    result = is_terminal_belief(belief, env)

    # ASSERT: Should return True
    assert result is True


def test_is_terminal_belief_large_mixed_particle_set():
    """Test is_terminal_belief with large mixed particle set."""
    # ARRANGE: Create LaserTag environment
    env = LaserTagPOMDP(discount_factor=0.95)

    # Create large set of mixed particles (alternating terminal/non-terminal)
    particles = []
    for i in range(100):
        terminal = i % 2 == 0  # Every other particle is terminal
        state = np.array(
            [
                float(i % 7),
                float(i % 11),
                float((i + 1) % 7),
                float((i + 1) % 11),
                1.0 if terminal else 0.0,
            ]
        )
        particles.append(state)

    log_weights = np.log(np.ones(100) / 100)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    # ACT: Check if belief is terminal
    result = is_terminal_belief(belief, env)

    # ASSERT: Should return False (not all particles are terminal)
    assert result is False


def test_is_terminal_belief_edge_case_all_false():
    """Test is_terminal_belief edge case with all False terminal flags."""
    # ARRANGE: Create LaserTag environment
    env = LaserTagPOMDP(discount_factor=0.95)

    # Create states with explicit False terminal flags
    state1 = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    state2 = np.array([2.0, 2.0, 3.0, 3.0, 0.0])
    state3 = np.array([4.0, 4.0, 5.0, 5.0, 0.0])

    particles = [state1, state2, state3]
    log_weights = np.log(np.ones(3) / 3)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    # ACT: Check if belief is terminal
    result = is_terminal_belief(belief, env)

    # ASSERT: Should return False
    assert result is False


def test_is_terminal_belief_edge_case_all_true():
    """Test is_terminal_belief edge case with all True terminal flags."""
    # ARRANGE: Create LaserTag environment
    env = LaserTagPOMDP(discount_factor=0.95)

    # Create states with explicit True terminal flags
    state1 = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    state2 = np.array([2.0, 2.0, 3.0, 3.0, 1.0])
    state3 = np.array([4.0, 4.0, 5.0, 5.0, 1.0])

    particles = [state1, state2, state3]
    log_weights = np.log(np.ones(3) / 3)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    # ACT: Check if belief is terminal
    result = is_terminal_belief(belief, env)

    # ASSERT: Should return True
    assert result is True


def test_is_terminal_belief_comprehensive_scenarios():
    """Test is_terminal_belief with comprehensive scenarios."""
    # ARRANGE: Create LaserTag environment
    env = LaserTagPOMDP(discount_factor=0.95)

    # Scenario 1: All terminal
    all_terminal_particles = [
        np.array([0.0, 0.0, 1.0, 1.0, 1.0]),
        np.array([2.0, 2.0, 3.0, 3.0, 1.0]),
        np.array([4.0, 4.0, 5.0, 5.0, 1.0]),
    ]
    log_weights = np.log(np.ones(3) / 3)
    belief_all_terminal = WeightedParticleBelief(
        particles=all_terminal_particles, log_weights=log_weights
    )

    # Scenario 2: All non-terminal
    all_non_terminal_particles = [
        np.array([0.0, 0.0, 1.0, 1.0, 0.0]),
        np.array([2.0, 2.0, 3.0, 3.0, 0.0]),
        np.array([4.0, 4.0, 5.0, 5.0, 0.0]),
    ]
    belief_all_non_terminal = WeightedParticleBelief(
        particles=all_non_terminal_particles, log_weights=log_weights
    )

    # Scenario 3: Mixed (majority terminal)
    mixed_particles_majority_terminal = [
        np.array([0.0, 0.0, 1.0, 1.0, 1.0]),
        np.array([2.0, 2.0, 3.0, 3.0, 1.0]),
        np.array([4.0, 4.0, 5.0, 5.0, 0.0]),
    ]
    belief_mixed_majority_terminal = WeightedParticleBelief(
        particles=mixed_particles_majority_terminal, log_weights=log_weights
    )

    # Scenario 4: Mixed (majority non-terminal)
    mixed_particles_majority_non_terminal = [
        np.array([0.0, 0.0, 1.0, 1.0, 0.0]),
        np.array([2.0, 2.0, 3.0, 3.0, 0.0]),
        np.array([4.0, 4.0, 5.0, 5.0, 1.0]),
    ]
    belief_mixed_majority_non_terminal = WeightedParticleBelief(
        particles=mixed_particles_majority_non_terminal, log_weights=log_weights
    )

    # ACT & ASSERT: Test all scenarios
    assert is_terminal_belief(belief_all_terminal, env) is True
    assert is_terminal_belief(belief_all_non_terminal, env) is False
    assert is_terminal_belief(belief_mixed_majority_terminal, env) is False
    assert is_terminal_belief(belief_mixed_majority_non_terminal, env) is False


def test_weighted_particle_belief_update_normalized_weights_sum_to_one_cartpole():
    """Test that normalized weights sum to 1 after belief update using CartPole POMDP environment.

    Purpose: Validates that WeightedParticleBelief properly normalizes weights after belief updates
    when using continuous state space environments like CartPole POMDP.

    Given: A WeightedParticleBelief initialized with CartPole states and unequal log weights
    When: A belief update is performed with resampling enabled
    Then: The normalized weights should sum to 1 (or number of particles for proper normalization)

    Test type: unit
    """
    # ARRANGE: Create CartPole POMDP environment
    noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
    env = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov)

    # Create initial states for CartPole (4D continuous states)
    particles = [
        np.array([0.0, 0.0, 0.1, 0.0]),  # [cart_pos, cart_vel, pole_angle, pole_ang_vel]
        np.array([0.1, 0.0, 0.08, 0.0]),
        np.array([-0.1, 0.0, 0.12, 0.0]),
        np.array([0.05, 0.0, 0.09, 0.0]),
    ]

    # Create unequal log weights to test normalization
    log_weights = np.array([1.0, 2.0, 3.0, 4.0])
    belief = WeightedParticleBelief(
        particles=particles, log_weights=log_weights, resampling=True, ess_factor=0.5
    )

    # ACT: Perform belief update
    action = 1  # Apply right force
    observation = np.array([0.06, 0.0, 0.09, 0.0])  # Noisy observation
    updated_belief = belief.update(action=action, observation=observation, pomdp=env)

    # ASSERT: Verify normalized weights sum to 1.0 (proper probability distribution)
    assert np.isclose(
        np.sum(updated_belief.normalized_weights), 1.0
    ), f"Normalized weights should sum to 1.0, got {np.sum(updated_belief.normalized_weights)}"

    # Additional assertions to ensure proper belief update
    assert isinstance(updated_belief, WeightedParticleBelief)
    assert len(updated_belief.particles) == len(belief.particles)
    assert np.all(np.isfinite(updated_belief.log_weights))
    assert not np.array_equal(updated_belief.log_weights, belief.log_weights)

    # Verify normalized_weights is a 1D array with correct shape
    assert (
        updated_belief.normalized_weights.ndim == 1
    ), f"normalized_weights should be 1D, got {updated_belief.normalized_weights.ndim}D"
    assert updated_belief.normalized_weights.shape == (
        len(updated_belief.particles),
    ), f"normalized_weights shape should be ({len(updated_belief.particles)},), got {updated_belief.normalized_weights.shape}"
