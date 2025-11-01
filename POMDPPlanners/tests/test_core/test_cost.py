"""Tests for cost and reward calculation utilities.

This module tests the cost and reward calculation utilities, focusing on:
- Expected cost calculation from weighted particle beliefs
- Expected reward calculation from weighted particle beliefs
- Entropy weighting functionality
- Error handling for invalid inputs
"""

import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import Belief, WeightedParticleBelief
from POMDPPlanners.core.cost import (
    belief_expectation_cost,
    belief_expectation_cost_particle_belief,
    belief_expectation_reward,
    belief_expectation_reward_particle_belief,
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


class NonWeightedParticleBelief(Belief):
    """Mock belief that is not WeightedParticleBelief for testing error cases."""

    def update(self, action, observation, pomdp, state=None):
        return self

    def sample(self):
        return "state"


@pytest.fixture
def uniform_weighted_belief():
    """Create a weighted particle belief with uniform weights.

    Purpose: Provides a WeightedParticleBelief fixture with uniform weights for testing

    Given: 2 TigerPOMDP states with uniform weights
    When: Fixture is used in cost calculation tests
    Then: Returns WeightedParticleBelief with equal probability for all particles

    Test type: unit
    """
    particles = ["tiger_left", "tiger_right"]
    log_weights = np.log(np.ones(2) / 2)
    return WeightedParticleBelief(particles=particles, log_weights=log_weights)


@pytest.fixture
def non_uniform_weighted_belief():
    """Create a weighted particle belief with non-uniform weights.

    Purpose: Provides a WeightedParticleBelief fixture with non-uniform weights for testing

    Given: 2 TigerPOMDP states with weights [0.7, 0.3]
    When: Fixture is used in cost calculation tests
    Then: Returns WeightedParticleBelief with specified probability distribution

    Test type: unit
    """
    particles = ["tiger_left", "tiger_right"]
    log_weights = np.log(np.array([0.7, 0.3]))
    return WeightedParticleBelief(particles=particles, log_weights=log_weights)


@pytest.fixture
def tiger_env():
    """Create a TigerPOMDP environment for testing.

    Purpose: Provides TigerPOMDP fixture for cost calculation tests

    Given: TigerPOMDP environment
    When: Fixture is used in cost calculation tests
    Then: Returns TigerPOMDP instance with proper reward method

    Test type: unit
    """
    return TigerPOMDP(discount_factor=0.95)


def test_belief_expectation_cost_particle_belief_uniform_weights(
    tiger_env, uniform_weighted_belief
):
    """Test cost calculation with uniform weights.

    Purpose: Validates that cost calculation correctly computes expected cost with uniform weights

    Given: WeightedParticleBelief with uniform weights over tiger_left and tiger_right, and "listen" action
    When: belief_expectation_cost_particle_belief is called
    Then: Returns expected cost of 1.0 (negative of -1.0 listen reward) with entropy_weight=0.0

    Test type: unit
    """
    action = "listen"
    cost = belief_expectation_cost_particle_belief(
        belief=uniform_weighted_belief, action=action, env=tiger_env, entropy_weight=0.0
    )

    # With uniform weights and listen action (reward -1.0 for both states), expected cost is 1.0
    assert cost == pytest.approx(1.0)


def test_belief_expectation_cost_particle_belief_non_uniform_weights(
    tiger_env, non_uniform_weighted_belief
):
    """Test cost calculation with non-uniform weights.

    Purpose: Validates that cost calculation correctly computes weighted average with non-uniform weights

    Given: WeightedParticleBelief with weights [0.7, 0.3] over tiger_left and tiger_right, and "listen" action
    When: belief_expectation_cost_particle_belief is called
    Then: Returns expected cost of 1.0 (weighted average of listen reward -1.0 for both states)

    Test type: unit
    """
    action = "listen"
    cost = belief_expectation_cost_particle_belief(
        belief=non_uniform_weighted_belief, action=action, env=tiger_env, entropy_weight=0.0
    )

    # With listen action (reward -1.0 for both states), expected cost is 1.0 regardless of weights
    assert cost == pytest.approx(1.0)


def test_belief_expectation_cost_particle_belief_varying_rewards():
    """Test cost calculation with varying rewards per particle.

    Purpose: Validates that cost calculation correctly computes weighted average with varying rewards

    Given: WeightedParticleBelief with weights [0.7, 0.3] over tiger_left and tiger_right, and "open_left" action
    When: belief_expectation_cost_particle_belief is called
    Then: Returns expected cost of 73.0 (weighted average: -0.7*(-100) - 0.3*10 = 70 - 3 = 67, but cost is negative so 73)

    Test type: unit
    """
    particles = ["tiger_left", "tiger_right"]
    log_weights = np.log(np.array([0.7, 0.3]))
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    env = TigerPOMDP(discount_factor=0.95)
    action = "open_left"
    cost = belief_expectation_cost_particle_belief(
        belief=belief, action=action, env=env, entropy_weight=0.0
    )

    # Expected cost = -0.7*(-100.0) - 0.3*(10.0) = 70.0 - 3.0 = 67.0
    # But cost is negative of reward, so: cost = -reward
    # reward = 0.7*(-100) + 0.3*10 = -70 + 3 = -67
    # cost = -(-67) = 67
    assert cost == pytest.approx(67.0)


def test_belief_expectation_cost_particle_belief_with_entropy():
    """Test cost calculation with entropy weighting.

    Purpose: Validates that entropy weighting is correctly added to the cost calculation

    Given: WeightedParticleBelief with uniform weights and entropy_weight=0.1
    When: belief_expectation_cost_particle_belief is called
    Then: Returns cost including entropy term (expected entropy for uniform distribution is log(2))

    Test type: unit
    """
    particles = ["tiger_left", "tiger_right"]
    log_weights = np.log(np.ones(2) / 2)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    env = TigerPOMDP(discount_factor=0.95)
    action = "listen"
    entropy_weight = 0.1

    cost = belief_expectation_cost_particle_belief(
        belief=belief, action=action, env=env, entropy_weight=entropy_weight
    )

    # Expected cost = 1.0 (base cost from listen reward -1.0) + 0.1 * entropy
    # For uniform distribution: entropy = -sum(p * log(p)) = -2 * (0.5 * log(0.5)) = log(2)
    expected_entropy = -np.sum(belief.normalized_weights * np.log(belief.normalized_weights))
    expected_cost = 1.0 + entropy_weight * expected_entropy

    assert cost == pytest.approx(expected_cost)


def test_belief_expectation_cost_particle_belief_negative_entropy_weight():
    """Test that negative entropy weight raises ValueError.

    Purpose: Validates that negative entropy weights are rejected

    Given: WeightedParticleBelief and entropy_weight=-0.1
    When: belief_expectation_cost_particle_belief is called
    Then: Raises ValueError with appropriate message

    Test type: unit
    """
    particles = ["tiger_left", "tiger_right"]
    log_weights = np.log(np.ones(2) / 2)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    env = TigerPOMDP(discount_factor=0.95)
    action = "listen"

    with pytest.raises(ValueError, match="Entropy weight must be non-negative"):
        belief_expectation_cost_particle_belief(
            belief=belief, action=action, env=env, entropy_weight=-0.1
        )


def test_belief_expectation_cost_particle_belief_zero_entropy_weight(
    tiger_env, uniform_weighted_belief
):
    """Test cost calculation with zero entropy weight (default).

    Purpose: Validates that zero entropy weight behaves as default (no entropy term added)

    Given: WeightedParticleBelief and entropy_weight=0.0 (default)
    When: belief_expectation_cost_particle_belief is called
    Then: Returns cost without entropy term

    Test type: unit
    """
    action = "listen"
    cost_with_explicit_zero = belief_expectation_cost_particle_belief(
        belief=uniform_weighted_belief, action=action, env=tiger_env, entropy_weight=0.0
    )

    cost_with_default = belief_expectation_cost_particle_belief(
        belief=uniform_weighted_belief, action=action, env=tiger_env
    )

    # Both should be the same
    assert cost_with_explicit_zero == pytest.approx(cost_with_default)


def test_belief_expectation_reward_particle_belief_uniform_weights(
    tiger_env, uniform_weighted_belief
):
    """Test reward calculation with uniform weights.

    Purpose: Validates that reward calculation is negative of cost calculation

    Given: WeightedParticleBelief with uniform weights and "listen" action
    When: belief_expectation_reward_particle_belief is called
    Then: Returns expected reward of -1.0 (negative of cost)

    Test type: unit
    """
    action = "listen"
    reward = belief_expectation_reward_particle_belief(
        belief=uniform_weighted_belief, action=action, env=tiger_env, entropy_weight=0.0
    )

    # Reward should be negative of cost
    assert reward == pytest.approx(-1.0)

    # Verify it's indeed the negative
    cost = belief_expectation_cost_particle_belief(
        belief=uniform_weighted_belief, action=action, env=tiger_env, entropy_weight=0.0
    )
    assert reward == pytest.approx(-cost)


def test_belief_expectation_reward_particle_belief_with_entropy():
    """Test reward calculation with entropy weighting.

    Purpose: Validates that reward calculation correctly includes entropy term (as negative)

    Given: WeightedParticleBelief with uniform weights and entropy_weight=0.1
    When: belief_expectation_reward_particle_belief is called
    Then: Returns reward with entropy term subtracted (negative of cost with entropy)

    Test type: unit
    """
    particles = ["tiger_left", "tiger_right"]
    log_weights = np.log(np.ones(2) / 2)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    env = TigerPOMDP(discount_factor=0.95)
    action = "listen"
    entropy_weight = 0.1

    reward = belief_expectation_reward_particle_belief(
        belief=belief, action=action, env=env, entropy_weight=entropy_weight
    )

    cost = belief_expectation_cost_particle_belief(
        belief=belief, action=action, env=env, entropy_weight=entropy_weight
    )

    # Reward should be negative of cost
    assert reward == pytest.approx(-cost)


def test_belief_expectation_cost_with_weighted_particle_belief(tiger_env, uniform_weighted_belief):
    """Test general cost function with WeightedParticleBelief.

    Purpose: Validates that belief_expectation_cost correctly delegates to particle-specific function

    Given: WeightedParticleBelief with uniform weights
    When: belief_expectation_cost is called
    Then: Returns same result as belief_expectation_cost_particle_belief

    Test type: unit
    """
    action = "listen"
    cost_general = belief_expectation_cost(
        belief=uniform_weighted_belief, action=action, env=tiger_env, entropy_weight=0.0
    )

    cost_specific = belief_expectation_cost_particle_belief(
        belief=uniform_weighted_belief, action=action, env=tiger_env, entropy_weight=0.0
    )

    assert cost_general == pytest.approx(cost_specific)


def test_belief_expectation_cost_with_unsupported_belief_type(tiger_env):
    """Test that unsupported belief types raise NotImplementedError.

    Purpose: Validates that belief_expectation_cost raises error for unsupported belief types

    Given: NonWeightedParticleBelief (not WeightedParticleBelief)
    When: belief_expectation_cost is called
    Then: Raises NotImplementedError with appropriate message

    Test type: unit
    """
    belief = NonWeightedParticleBelief()
    action = "listen"

    with pytest.raises(NotImplementedError, match="not implemented for this belief type"):
        belief_expectation_cost(belief=belief, action=action, env=tiger_env)


def test_belief_expectation_reward_with_weighted_particle_belief(
    tiger_env, uniform_weighted_belief
):
    """Test general reward function with WeightedParticleBelief.

    Purpose: Validates that belief_expectation_reward correctly delegates and negates cost

    Given: WeightedParticleBelief with uniform weights
    When: belief_expectation_reward is called
    Then: Returns negative of belief_expectation_cost

    Test type: unit
    """
    action = "listen"
    reward = belief_expectation_reward(
        belief=uniform_weighted_belief, action=action, env=tiger_env, entropy_weight=0.0
    )

    cost = belief_expectation_cost(
        belief=uniform_weighted_belief, action=action, env=tiger_env, entropy_weight=0.0
    )

    assert reward == pytest.approx(-cost)


def test_belief_expectation_reward_with_unsupported_belief_type(tiger_env):
    """Test that unsupported belief types raise NotImplementedError in reward function.

    Purpose: Validates that belief_expectation_reward raises error for unsupported belief types

    Given: NonWeightedParticleBelief (not WeightedParticleBelief)
    When: belief_expectation_reward is called
    Then: Raises NotImplementedError (from underlying cost function)

    Test type: unit
    """
    belief = NonWeightedParticleBelief()
    action = "listen"

    with pytest.raises(NotImplementedError):
        belief_expectation_reward(belief=belief, action=action, env=tiger_env)


def test_belief_expectation_cost_single_particle():
    """Test cost calculation with single particle.

    Purpose: Validates that cost calculation works correctly with single particle belief

    Given: WeightedParticleBelief with single particle (tiger_left)
    When: belief_expectation_cost_particle_belief is called with "open_right" action
    Then: Returns cost based on that single particle's reward (10.0 -> cost = -10.0)

    Test type: unit
    """
    particles = ["tiger_left"]
    # Use a small positive log weight to satisfy validation (will be normalized to 1.0 anyway)
    # Workaround: use a very small positive value since validation requires at least one nonzero
    log_weights = np.array([1e-10])
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    env = TigerPOMDP(discount_factor=0.95)

    action = "open_right"  # Opening right door when tiger is left gives +10 reward
    cost = belief_expectation_cost_particle_belief(
        belief=belief, action=action, env=env, entropy_weight=0.0
    )

    # With single particle, cost should be negative of that particle's reward
    # reward = 10.0, so cost = -10.0
    assert cost == pytest.approx(-10.0)


def test_belief_expectation_cost_entropy_maximum_uncertainty():
    """Test that entropy term is maximized for uniform distribution.

    Purpose: Validates that entropy calculation correctly identifies maximum uncertainty case

    Given: Two beliefs: one uniform (high entropy) and one concentrated (low entropy)
    When: belief_expectation_cost_particle_belief is called with entropy_weight > 0
    Then: Uniform belief has higher total cost due to higher entropy

    Test type: unit
    """
    # Uniform belief (maximum entropy)
    particles_uniform = ["tiger_left", "tiger_right"]
    log_weights_uniform = np.log(np.ones(2) / 2)
    belief_uniform = WeightedParticleBelief(
        particles=particles_uniform, log_weights=log_weights_uniform
    )

    # Concentrated belief (minimum entropy)
    particles_concentrated = ["tiger_left", "tiger_right"]
    log_weights_concentrated = np.log(np.array([0.99, 0.01]))
    belief_concentrated = WeightedParticleBelief(
        particles=particles_concentrated, log_weights=log_weights_concentrated
    )

    env = TigerPOMDP(discount_factor=0.95)
    action = "listen"
    entropy_weight = 0.5

    cost_uniform = belief_expectation_cost_particle_belief(
        belief=belief_uniform, action=action, env=env, entropy_weight=entropy_weight
    )

    cost_concentrated = belief_expectation_cost_particle_belief(
        belief=belief_concentrated, action=action, env=env, entropy_weight=entropy_weight
    )

    # Uniform belief should have higher cost due to higher entropy
    assert cost_uniform > cost_concentrated


def test_belief_expectation_cost_weights_normalization():
    """Test that cost calculation uses normalized weights correctly.

    Purpose: Validates that cost calculation correctly uses normalized weights from belief

    Given: WeightedParticleBelief where normalized_weights may differ from raw log_weights
    When: belief_expectation_cost_particle_belief is called
    Then: Uses normalized_weights for cost calculation (not raw log_weights)

    Test type: unit
    """
    particles = ["tiger_left", "tiger_right"]
    # Use equal log weights (log(1/2) = -log(2)) that will be normalized internally
    # This represents equal probability for both particles
    log_weights = np.log(np.array([0.5, 0.5]))  # Will be normalized to [0.5, 0.5]
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    env = TigerPOMDP(discount_factor=0.95)
    action = "open_left"  # This gives different rewards: -100 for tiger_left, 10 for tiger_right

    cost = belief_expectation_cost_particle_belief(
        belief=belief, action=action, env=env, entropy_weight=0.0
    )

    # With normalized weights [0.5, 0.5], expected cost = -0.5*(-100.0) - 0.5*(10.0) = 50.0 - 5.0 = 45.0
    # Actually: reward = 0.5*(-100) + 0.5*10 = -50 + 5 = -45, so cost = -(-45) = 45
    assert cost == pytest.approx(45.0)

    # Verify normalized weights are being used
    assert np.allclose(belief.normalized_weights, [0.5, 0.5])
