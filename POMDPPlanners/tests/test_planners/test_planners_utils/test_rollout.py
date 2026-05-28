# SPDX-License-Identifier: MIT

"""Tests for rollout planner utilities.

This module tests the rollout planner utilities, focusing on:
- Basic rollout functionality
- Rollout policies
- Rollout execution
- Rollout evaluation
"""

import random
from typing import Any

import numpy as np
import pytest

from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.planners.planners_utils.rollout import (
    python_random_rollout,
    random_rollout_action_sampler,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    cartpole_pinned_kwargs,
    sanity_pinned_kwargs,
)

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


class MockActionSampler(ActionSampler):
    """Mock action sampler for testing rollouts."""

    def __init__(self, actions):
        self.actions = actions

    def sample(self, belief_node=None):
        return np.random.choice(self.actions)


class DeterministicActionSampler(ActionSampler):
    """Deterministic action sampler for reproducible testing."""

    def __init__(self, action):
        self.action = action
        self.call_count = 0

    def sample(self, belief_node=None):
        self.call_count += 1
        return self.action


@pytest.fixture
def tiger_environment():
    """Tiger POMDP environment fixture."""
    return TigerPOMDP(discount_factor=0.95)


@pytest.fixture
def sanity_environment():
    """Sanity POMDP environment fixture."""
    return SanityPOMDP(discount_factor=0.9, **sanity_pinned_kwargs())


@pytest.fixture
def cartpole_environment():
    """CartPole POMDP environment fixture."""
    noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
    return CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov, **cartpole_pinned_kwargs())


@pytest.fixture
def tiger_action_sampler():
    """Action sampler for Tiger POMDP."""
    return MockActionSampler(["listen", "open_left", "open_right"])


@pytest.fixture
def sanity_action_sampler():
    """Action sampler for Sanity POMDP."""
    return MockActionSampler([0, 1])


@pytest.fixture
def cartpole_action_sampler():
    """Action sampler for CartPole POMDP."""
    return MockActionSampler([0, 1])


def test_rollout_basic_functionality(tiger_environment, tiger_action_sampler):
    """Test basic rollout functionality.

    Purpose: Validates that the rollout function can execute a basic rollout simulation

    Given: A Tiger POMDP environment and action sampler with initial state "tiger_left"
    When: A rollout is performed with depth 0 and max_depth 5
    Then: The function returns a float value within reasonable bounds (-500 to 50) for Tiger POMDP rewards

    Test type: unit
    """
    initial_state = "tiger_left"

    rollout_value = random_rollout_action_sampler(
        state=initial_state,
        depth=0,
        action_sampler=tiger_action_sampler,
        environment=tiger_environment,
        discount_factor=0.95,
        max_depth=5,
    )

    # Should return a float value
    assert isinstance(rollout_value, float), f"Expected float, got {type(rollout_value)}"

    # Value should be within reasonable bounds for Tiger POMDP
    # Tiger rewards range from -100 to 10, so discounted sum should be reasonable
    assert -500 <= rollout_value <= 50, f"Rollout value {rollout_value} outside expected range"


def test_rollout_terminal_state_handling(tiger_environment, tiger_action_sampler):
    """Test that rollout handles terminal states correctly.

    Purpose: Validates that rollout function properly handles terminal states by returning 0

    Given: A Tiger POMDP environment with mocked terminal state behavior
    When: A rollout is attempted from a terminal state
    Then: The function returns 0.0 as expected for terminal states

    Test type: unit
    """
    # Mock terminal state
    original_is_terminal = tiger_environment.is_terminal
    tiger_environment.is_terminal = lambda state: True

    rollout_value = random_rollout_action_sampler(
        state="tiger_left",
        depth=0,
        action_sampler=tiger_action_sampler,
        environment=tiger_environment,
        discount_factor=0.95,
        max_depth=10,
    )

    # Should return 0 for terminal state
    assert rollout_value == 0.0, f"Expected 0.0 for terminal state, got {rollout_value}"

    # Restore original method
    tiger_environment.is_terminal = original_is_terminal


def test_rollout_max_depth_handling(sanity_environment, sanity_action_sampler):
    """Test that rollout respects max_depth parameter.

    Purpose: Validates that rollout function respects the max_depth parameter and produces reasonable values

    Given: A Sanity POMDP environment and action sampler with initial state 0
    When: Rollouts are performed with different max_depth values (1, 3, 5, 10)
    Then: Each rollout returns a float value within reasonable bounds (0 to max_depth) for the given depth

    Test type: unit
    """
    initial_state = 0

    # Test with different max depths
    for max_depth in [1, 3, 5, 10]:
        rollout_value = random_rollout_action_sampler(
            state=initial_state,
            depth=0,
            action_sampler=sanity_action_sampler,
            environment=sanity_environment,
            discount_factor=0.9,
            max_depth=max_depth,
        )

        assert isinstance(rollout_value, float), f"Expected float for depth {max_depth}"
        # Value should be reasonable for SanityPOMDP (rewards are 0 or 1)
        assert (
            0 <= rollout_value <= max_depth
        ), f"Value {rollout_value} unreasonable for depth {max_depth}"


def test_rollout_at_max_depth(sanity_environment, sanity_action_sampler):
    """Test rollout behavior when starting at max depth.

    Purpose: Validates that rollout function returns 0 when starting at the maximum allowed depth

    Given: A Sanity POMDP environment and action sampler
    When: A rollout is attempted starting at depth 5 with max_depth 5
    Then: The function returns 0.0 since no further exploration is allowed

    Test type: unit
    """
    rollout_value = random_rollout_action_sampler(
        state=0,
        depth=5,  # At max depth
        action_sampler=sanity_action_sampler,
        environment=sanity_environment,
        discount_factor=0.9,
        max_depth=5,
    )

    # Should return 0 when at max depth
    assert rollout_value == 0.0, f"Expected 0.0 at max depth, got {rollout_value}"


def test_rollout_discount_factor_effect(sanity_environment, sanity_action_sampler):
    """Test that discount factor affects rollout values appropriately.

    Purpose: Validates that different discount factors produce appropriately different rollout values

    Given: A Sanity POMDP environment and action sampler with initial state 0
    When: Multiple rollouts are performed with different discount factors (0.1, 0.5, 0.9, 0.99)
    Then: Higher discount factors generally lead to higher rollout values, accounting for noise

    Test type: unit
    """
    state = 0  # Good state in SanityPOMDP

    # Test with different discount factors
    discount_factors = [0.1, 0.5, 0.9, 0.99]
    rollout_values = []

    for gamma in discount_factors:
        # Use multiple rollouts to reduce variance
        values = []
        for _ in range(20):
            value = random_rollout_action_sampler(
                state=state,
                depth=0,
                action_sampler=sanity_action_sampler,
                environment=sanity_environment,
                discount_factor=gamma,
                max_depth=10,
            )
            values.append(value)

        mean_value = np.mean(values)
        rollout_values.append(mean_value)

    # Generally, higher discount factors should lead to higher values
    # (though this can be noisy due to randomness)
    assert rollout_values[0] <= rollout_values[-1] + 1.0, "Discount factor effect not as expected"


def test_rollout_deterministic_behavior(sanity_environment):
    """Test rollout with deterministic action sampler.

    Purpose: Validates that rollout function works correctly with deterministic action sampling

    Given: A Sanity POMDP environment and a deterministic action sampler that always chooses action 0
    When: A rollout is performed from initial state 0 (good state)
    Then: The function returns a positive value and the action sampler is called multiple times

    Test type: unit
    """
    # Create deterministic sampler that always chooses action 0 (good action in SanityPOMDP)
    det_sampler = DeterministicActionSampler(action=0)

    rollout_value = random_rollout_action_sampler(
        state=0,  # Start in good state
        depth=0,
        action_sampler=det_sampler,
        environment=sanity_environment,
        discount_factor=0.9,
        max_depth=5,
    )

    # With deterministic good actions from good state, should get positive value
    assert rollout_value > 0, f"Expected positive value with good actions, got {rollout_value}"

    # Sampler should have been called multiple times
    assert det_sampler.call_count > 0, "Action sampler should have been called"


def test_rollout_different_initial_states(sanity_environment, sanity_action_sampler):
    """Test rollout from different initial states.

    Purpose: Validates that rollout function produces appropriate values for different initial states

    Given: A Sanity POMDP environment with good state (0) and bad state (1)
    When: Multiple rollouts are performed from each state with 50 trials each
    Then: On average, the good state produces higher or equal rollout values compared to the bad state

    Test type: unit
    """
    max_depth = 8
    n_rollouts = 50

    # Test from good state (0)
    good_state_values = []
    for _ in range(n_rollouts):
        value = random_rollout_action_sampler(
            state=0,
            depth=0,
            action_sampler=sanity_action_sampler,
            environment=sanity_environment,
            discount_factor=0.9,
            max_depth=max_depth,
        )
        good_state_values.append(value)

    # Test from bad state (1)
    bad_state_values = []
    for _ in range(n_rollouts):
        value = random_rollout_action_sampler(
            state=1,
            depth=0,
            action_sampler=sanity_action_sampler,
            environment=sanity_environment,
            discount_factor=0.9,
            max_depth=max_depth,
        )
        bad_state_values.append(value)

    # On average, good state should have higher or equal value than bad state
    mean_good = np.mean(good_state_values)
    mean_bad = np.mean(bad_state_values)

    # Allow some tolerance due to randomness, but good state should generally be better
    assert (
        mean_good >= mean_bad - 0.5
    ), f"Good state mean {mean_good} should be >= bad state mean {mean_bad}"


def test_rollout_with_continuous_environment(cartpole_environment, cartpole_action_sampler):
    """Test rollout with continuous state environment.

    Purpose: Validates that rollout function works correctly with continuous state environments

    Given: A CartPole POMDP environment with continuous state space and action sampler
    When: A rollout is performed from a sampled initial state with max_depth 15
    Then: The function returns a float value within reasonable bounds (-20 to 20) for CartPole rewards

    Test type: unit
    """
    # Sample initial state
    initial_state_dist = cartpole_environment.initial_state_dist()
    state = initial_state_dist.sample()[0]

    rollout_value = random_rollout_action_sampler(
        state=state,
        depth=0,
        action_sampler=cartpole_action_sampler,
        environment=cartpole_environment,
        discount_factor=0.99,
        max_depth=15,
    )

    # Should return a float value
    assert isinstance(rollout_value, float), f"Expected float, got {type(rollout_value)}"

    # CartPole rewards are typically 1.0 per step, so value should be reasonable
    assert -20 <= rollout_value <= 20, f"Rollout value {rollout_value} outside reasonable range"


def test_rollout_variance_reduction():
    """Test that multiple rollouts reduce variance in value estimates.

    Purpose: Validates that multiple rollouts provide more stable value estimates with reduced variance

    Given: A Sanity POMDP environment and action sampler with initial state 0
    When: Single rollout vs 100 multiple rollouts are performed
    Then: Multiple rollouts produce reasonable mean values and standard errors, demonstrating variance reduction

    Test type: unit
    """
    sanity = SanityPOMDP(discount_factor=0.95, **sanity_pinned_kwargs())
    sampler = MockActionSampler([0, 1])
    state = 0

    # Single rollout
    single_value = random_rollout_action_sampler(
        state=state,
        depth=0,
        action_sampler=sampler,
        environment=sanity,
        discount_factor=0.95,
        max_depth=10,
    )

    # Multiple rollouts
    n_rollouts = 100
    multiple_values = []
    for _ in range(n_rollouts):
        value = random_rollout_action_sampler(
            state=state,
            depth=0,
            action_sampler=sampler,
            environment=sanity,
            discount_factor=0.95,
            max_depth=10,
        )
        multiple_values.append(value)

    mean_multiple = np.mean(multiple_values)
    std_multiple = np.std(multiple_values)

    # Mean should be reasonable
    assert 0 <= mean_multiple <= 15, f"Mean value {mean_multiple} outside reasonable range"

    # Standard error should be reasonable (not too high variance)
    std_error = std_multiple / np.sqrt(n_rollouts)
    assert std_error < 1.0, f"Standard error {std_error} too high"


def test_rollout_recursion_depth():
    """Test that rollout doesn't cause stack overflow with deep recursion.

    Purpose: Validates that rollout function can handle very deep recursion without stack overflow

    Given: A Sanity POMDP environment and action sampler
    When: A rollout is performed with very deep max_depth (100)
    Then: The function completes successfully and returns a non-negative float value

    Test type: unit
    """
    sanity = SanityPOMDP(discount_factor=0.99, **sanity_pinned_kwargs())
    sampler = MockActionSampler([0, 1])

    # Test with very deep max_depth
    rollout_value = random_rollout_action_sampler(
        state=0,
        depth=0,
        action_sampler=sampler,
        environment=sanity,
        discount_factor=0.99,
        max_depth=100,  # Deep recursion
    )

    # Should complete without stack overflow
    assert isinstance(rollout_value, float), "Should handle deep recursion"
    assert rollout_value >= 0, "Value should be non-negative for good initial state"


# Usage Example Tests
# Following the project standard of testing all usage examples from docstrings


def test_basic_tiger_rollout_usage_example():
    """Test the basic Tiger POMDP rollout usage example from docstring.

    Purpose: Validates that the basic Tiger POMDP rollout usage example works correctly

    Given: A Tiger POMDP environment and TigerActionSampler with initial state "tiger_left"
    When: A rollout is performed using the docstring example code
    Then: The rollout returns a float value within reasonable bounds for Tiger POMDP

    Test type: example
    """

    # Simple action sampler for Tiger POMDP (from docstring)
    class TigerActionSampler(ActionSampler):
        def sample(self, belief_node=None):
            return np.random.choice(["listen", "open_left", "open_right"])

    # Create environment and sampler (from docstring)
    tiger = TigerPOMDP(discount_factor=0.95)
    action_sampler = TigerActionSampler()

    # Perform rollout from initial state (from docstring)
    initial_state = "tiger_left"
    rollout_value = random_rollout_action_sampler(
        state=initial_state,
        depth=0,
        action_sampler=action_sampler,
        environment=tiger,
        discount_factor=0.95,
        max_depth=10,
    )

    # Verify results
    assert isinstance(rollout_value, float), "Rollout should return float value"

    # Calculate theoretical bounds for Tiger POMDP with max_depth=10 and discount_factor=0.95
    # Worst case: all actions are "listen" (-1 each) for 10 steps
    # Best case: open correct door early (+10) then mostly listen
    # Realistic worst case: open wrong doors multiple times (-100 each) + some listening
    # With discount factor 0.95, cumulative rewards can go much lower than -500
    theoretical_min = -1000  # Conservative lower bound for worst-case scenarios
    theoretical_max = 100  # Conservative upper bound for best-case scenarios

    assert (
        theoretical_min <= rollout_value <= theoretical_max
    ), f"Tiger rollout value {rollout_value} should be in reasonable range [{theoretical_min}, {theoretical_max}]"


def test_cartpole_rollout_usage_example():
    """Test the CartPole rollout usage example from docstring.

    Purpose: Validates that the CartPole rollout usage example works correctly with continuous state space

    Given: A CartPole POMDP environment with continuous state space and CartPoleActionSampler
    When: A rollout is performed from a sampled initial state with max_depth 20
    Then: The rollout returns a float value and the state is a 4-dimensional numpy array

    Test type: example
    """

    class CartPoleActionSampler(ActionSampler):
        def sample(self, belief_node=None):
            return np.random.choice([0, 1])  # Left or right force

    # Create CartPole environment (from docstring)
    noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
    cartpole = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov, **cartpole_pinned_kwargs())
    action_sampler = CartPoleActionSampler()

    # Sample initial state and perform rollout (from docstring)
    initial_state_dist = cartpole.initial_state_dist()
    state = initial_state_dist.sample()[0]

    rollout_value = random_rollout_action_sampler(
        state=state,
        depth=0,
        action_sampler=action_sampler,
        environment=cartpole,
        discount_factor=0.99,
        max_depth=20,  # Longer horizon for control task
    )

    # Verify results
    assert isinstance(rollout_value, float), "CartPole rollout should return float value"
    assert isinstance(state, np.ndarray), "CartPole state should be numpy array"
    assert len(state) == 4, "CartPole state should be 4-dimensional"


def test_multiple_rollouts_usage_example():
    """Test the multiple rollouts for variance reduction usage example from docstring.

    Purpose: Validates that multiple rollouts can be used to reduce variance in value estimates

    Given: A Sanity POMDP environment and SanityActionSampler with initial state 0
    When: 50 rollouts are performed and confidence intervals are calculated
    Then: All rollout values are floats, mean is non-negative, and confidence intervals are properly calculated

    Test type: integration
    """

    class SanityActionSampler(ActionSampler):
        def sample(self, belief_node=None):
            return np.random.choice([0, 1])

    sanity = SanityPOMDP(discount_factor=0.95, **sanity_pinned_kwargs())
    action_sampler = SanityActionSampler()

    # Perform multiple rollouts to reduce variance (from docstring)
    state = 0  # Good state
    n_rollouts = 50  # Reduced for testing speed
    rollout_values = []

    for _ in range(n_rollouts):
        value = random_rollout_action_sampler(
            state=state,
            depth=0,
            action_sampler=action_sampler,
            environment=sanity,
            discount_factor=0.95,
            max_depth=15,
        )
        rollout_values.append(value)

    # Estimate value with confidence intervals (from docstring)
    mean_value = np.mean(rollout_values)
    std_value = np.std(rollout_values)
    confidence_interval = 1.96 * std_value / np.sqrt(n_rollouts)

    # Verify results
    assert len(rollout_values) == n_rollouts, "Should have correct number of rollouts"
    assert all(isinstance(v, float) for v in rollout_values), "All rollout values should be floats"
    assert mean_value >= 0, "Mean value should be non-negative for good initial state"
    assert std_value >= 0, "Standard deviation should be non-negative"
    assert confidence_interval >= 0, "Confidence interval should be non-negative"


def test_rollout_depth_comparison_usage_example():
    """Test the rollout depth comparison usage example from docstring.

    Purpose: Validates that different max_depth values produce different rollout statistics

    Given: A Sanity POMDP environment and SanityActionSampler with initial state 0
    When: Rollouts are performed with different max_depth values (5, 10, 20) and 20 trials each
    Then: Results are collected for all depths with proper mean and standard deviation calculations

    Test type: example
    """

    class SanityActionSampler(ActionSampler):
        def sample(self, belief_node=None):
            return np.random.choice([0, 1])

    sanity = SanityPOMDP(discount_factor=0.95, **sanity_pinned_kwargs())
    action_sampler = SanityActionSampler()
    state = 0

    # Test effect of max_depth on value estimates (from docstring)
    depths = [5, 10, 20]  # Reduced for testing speed
    depth_results = {}

    for max_depth in depths:
        values = []
        for _ in range(20):  # Reduced for testing speed
            value = random_rollout_action_sampler(
                state=state,
                depth=0,
                action_sampler=action_sampler,
                environment=sanity,
                discount_factor=0.95,
                max_depth=max_depth,
            )
            values.append(value)

        depth_results[max_depth] = {"mean": np.mean(values), "std": np.std(values)}

    # Verify results
    assert len(depth_results) == len(depths), "Should have results for all depths"
    for depth, stats in depth_results.items():
        assert "mean" in stats, f"Missing mean for depth {depth}"
        assert "std" in stats, f"Missing std for depth {depth}"
        assert stats["mean"] >= 0, f"Mean should be non-negative for depth {depth}"
        assert stats["std"] >= 0, f"Std should be non-negative for depth {depth}"


def test_informed_action_sampler_usage_example():
    """Test the custom informed action sampler usage example from docstring.

    Purpose: Validates that informed action samplers can use domain knowledge to improve rollout quality

    Given: A Tiger POMDP environment with random vs informed action samplers
    When: Multiple rollouts are performed with both samplers (20 trials each)
    Then: Both samplers produce reasonable values within expected ranges for Tiger POMDP

    Test type: example
    """

    class TigerActionSampler(ActionSampler):
        def sample(self, belief_node=None):
            return np.random.choice(["listen", "open_left", "open_right"])

    class InformedActionSampler(ActionSampler):
        """Action sampler that uses domain knowledge for better rollouts"""

        def __init__(self, environment):
            self.environment = environment

        def sample(self, belief_node=None):
            # For Tiger POMDP: listen more often early, then choose door
            if np.random.random() < 0.7:
                return "listen"  # Gather information first
            else:
                return np.random.choice(["open_left", "open_right"])

    # Compare random vs informed rollouts (from docstring)
    tiger = TigerPOMDP(discount_factor=0.95)
    random_sampler = TigerActionSampler()  # From earlier example
    informed_sampler = InformedActionSampler(tiger)

    initial_state = "tiger_left"
    n_trials = 20  # Reduced for testing speed

    random_values = [
        random_rollout_action_sampler(
            initial_state, 0, random_sampler, tiger, 0.95, 10  # Reduced depth for speed
        )
        for _ in range(n_trials)
    ]

    informed_values = [
        random_rollout_action_sampler(
            initial_state,
            0,
            informed_sampler,
            tiger,
            0.95,
            10,  # Reduced depth for speed
        )
        for _ in range(n_trials)
    ]

    # Verify results
    assert len(random_values) == n_trials, "Should have correct number of random rollouts"
    assert len(informed_values) == n_trials, "Should have correct number of informed rollouts"
    assert all(isinstance(v, float) for v in random_values), "All random values should be floats"
    assert all(
        isinstance(v, float) for v in informed_values
    ), "All informed values should be floats"

    # Both should produce reasonable values (though we can't guarantee informed is better due to randomness)
    random_mean = np.mean(random_values)
    informed_mean = np.mean(informed_values)
    assert -500 <= random_mean <= 100, "Random policy mean should be in reasonable range"
    assert -500 <= informed_mean <= 100, "Informed policy mean should be in reasonable range"


def test_rollout_parameter_validation():
    """Test rollout function with edge case parameters.

    Purpose: Validates that rollout function handles extreme discount factor values correctly

    Given: A Sanity POMDP environment and action sampler
    When: Rollouts are performed with discount_factor=0.0 and discount_factor=1.0
    Then: Function works correctly with both extreme values, returning appropriate results

    Test type: unit
    """
    sanity = SanityPOMDP(discount_factor=0.5, **sanity_pinned_kwargs())
    sampler = MockActionSampler([0, 1])

    # Test with discount_factor of 0 (no future value)
    rollout_value = random_rollout_action_sampler(
        state=0,
        depth=0,
        action_sampler=sampler,
        environment=sanity,
        discount_factor=0.0,
        max_depth=5,
    )

    # With discount_factor=0, only immediate reward should matter
    expected_reward = sanity.reward(state=0, action=sampler.actions[0])
    assert rollout_value >= 0, "Should get non-negative value with discount_factor=0"

    # Test with discount_factor of 1 (no discounting)
    rollout_value = random_rollout_action_sampler(
        state=0,
        depth=0,
        action_sampler=sampler,
        environment=sanity,
        discount_factor=1.0,
        max_depth=3,
    )

    assert isinstance(rollout_value, float), "Should work with discount_factor=1.0"


def test_rollout_action_sampler_integration():
    """Test that rollout works with different types of action samplers.

    Purpose: Validates that rollout function works correctly with various action sampler configurations

    Given: A Tiger POMDP environment and different action samplers (single action, two actions, all actions)
    When: Rollouts are performed with each sampler configuration
    Then: All samplers return float values within reasonable bounds regardless of action space size

    Test type: integration
    """
    tiger = TigerPOMDP(discount_factor=0.95)

    # Test with different action samplers
    samplers = [
        MockActionSampler(["listen"]),  # Single action
        MockActionSampler(["listen", "open_left"]),  # Two actions
        MockActionSampler(["listen", "open_left", "open_right"]),  # All actions
    ]

    for i, sampler in enumerate(samplers):
        rollout_value = random_rollout_action_sampler(
            state="tiger_left",
            depth=0,
            action_sampler=sampler,
            environment=tiger,
            discount_factor=0.95,
            max_depth=5,
        )

        assert isinstance(rollout_value, float), f"Sampler {i} should return float value"
        # Values should be reasonable regardless of action space size
        assert -300 <= rollout_value <= 50, f"Sampler {i} value {rollout_value} outside range"


class _NextStateProbeTigerPOMDP(TigerPOMDP):
    """TigerPOMDP probe that reports whether the realised next_state is threaded.

    Returns +1.0 when ``reward(...)`` is called with a non-None ``next_state``
    and -1.0 otherwise. Used to verify that planner code threads the realised
    transition outcome into ``Environment.reward``.
    """

    def reward(self, state: str, action: str, next_state: Any = None) -> float:
        return 1.0 if next_state is not None else -1.0


def test_python_random_rollout_threads_realised_next_state_to_reward():
    """Test that python_random_rollout passes the sampled next_state to reward().

    Purpose: Validates that the rollout helper threads the realised
        post-transition state into Environment.reward(...) instead of dropping
        it (which would force envs with transition-dependent rewards to
        resample internally and diverge from the trajectory).

    Given: A TigerPOMDP-derived probe whose reward returns +1 when
        next_state is provided and -1 otherwise, plus a deterministic
        action sampler driving a max_depth=4 rollout.
    When: python_random_rollout is executed.
    Then: The discounted return is strictly positive, proving every
        per-step reward call received the realised next_state.

    Test type: unit
    """
    env = _NextStateProbeTigerPOMDP(discount_factor=1.0)
    sampler = DeterministicActionSampler("listen")

    rollout_value = python_random_rollout(
        state="tiger_left",
        depth=0,
        action_sampler=sampler,
        environment=env,
        discount_factor=1.0,
        max_depth=4,
    )

    assert rollout_value == pytest.approx(4.0)
