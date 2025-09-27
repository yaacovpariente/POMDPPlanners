"""Tests for Discrete Light Dark POMDP environment.

This module tests the Discrete Light Dark POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

import random

import numpy as np
import pytest

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import ObservationModel
from POMDPPlanners.core.policy import PolicyInfoVariable, PolicyRunData
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
)


@pytest.fixture
def base_light_dark_environment() -> DiscreteLightDarkPOMDP:
    """Fixture providing a base DiscreteLightDarkPOMDP environment for comparison."""
    return DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        transition_error_prob=0.05,
        observation_error_prob=0.05,
        obstacle_hit_probability=0.2,
        obstacle_reward=-10.0,
        goal_reward=10.0,
        fuel_cost=2.0,
        grid_size=11,
        is_stochastic_reward=True,
    )


class TestDiscreteLightDarkPOMDPEquality:
    """Test suite for DiscreteLightDarkPOMDP equality comparisons."""

    def test_same_discount_factor(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with same discount factor are equal.

        Purpose: Validates that DiscreteLightDarkPOMDP environments with identical parameters are equal and exhibit symmetry

        Given: Two DiscreteLightDarkPOMDP environments with same parameters (discount=0.95, errors=0.05, grid=11x11, etc.)
        When: Equality comparison is performed between identical environment configurations
        Then: Both directions return True (env1 == env2 and env2 == env1) confirming symmetric equality

        Test type: unit
        """
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment == other_env
        assert other_env == base_light_dark_environment  # Test symmetry

    def test_different_discount_factor(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different discount factors are not equal.

        Purpose: Validates that DiscreteLightDarkPOMDP environments with different discount factors are correctly identified as unequal

        Given: Base DiscreteLightDarkPOMDP with discount=0.95 and another with discount=0.8, otherwise identical parameters
        When: Equality comparison is performed between environments with different discount factors
        Then: Both directions return False (env1 != env2 and env2 != env1) confirming inequality detection

        Test type: unit
        """
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.8,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment != other_env
        assert other_env != base_light_dark_environment  # Test symmetry

    def test_different_transition_error(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different transition error probabilities are not equal.

        Purpose: Validates that DiscreteLightDarkPOMDP environments with different transition error probabilities are correctly identified as unequal

        Given: Base DiscreteLightDarkPOMDP with transition_error_prob=0.05 and another with transition_error_prob=0.1
        When: Equality comparison is performed between environments with different transition error rates
        Then: Environments are correctly identified as not equal due to different stochastic transition parameters

        Test type: unit
        """
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.1,  # Different transition error
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment != other_env

    def test_different_observation_error(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different observation error probabilities are not equal.

        Purpose: Validates that DiscreteLightDarkPOMDP environments with different observation error probabilities are correctly identified as unequal

        Given: Base DiscreteLightDarkPOMDP with observation_error_prob=0.05 and another with observation_error_prob=0.1
        When: Equality comparison is performed between environments with different observation noise levels
        Then: Environments are correctly identified as not equal due to different partial observability parameters

        Test type: unit
        """
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.1,  # Different observation error
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment != other_env

    def test_different_beacons(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different beacon positions are not equal.

        Purpose: Validates that DiscreteLightDarkPOMDP environments with different beacon configurations are correctly identified as unequal

        Given: Base DiscreteLightDarkPOMDP with default beacon positions and another with custom beacon array [[1,1,1,6,6,6,11,11,11],[1,6,11,1,6,11,1,6,11]]
        When: Equality comparison is performed between environments with different light sources
        Then: Environments are correctly identified as not equal due to different beacon position configurations

        Test type: unit
        """
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            beacons=[
                (1, 1),
                (1, 6),
                (1, 11),
                (6, 1),
                (6, 6),
                (6, 11),
                (11, 1),
                (11, 6),
                (11, 11),
            ],  # Different beacons
        )
        assert base_light_dark_environment != other_env

    def test_different_obstacles(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different obstacle positions are not equal.

        Purpose: Validates that DiscreteLightDarkPOMDP environments with different obstacle configurations are correctly identified as unequal

        Given: Base DiscreteLightDarkPOMDP with default obstacle positions and another with custom obstacle array [[4,8],[6,6]]
        When: Equality comparison is performed between environments with different obstacle layouts
        Then: Environments are correctly identified as not equal due to different obstacle position configurations

        Test type: unit
        """
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            obstacles=[(4, 8), (6, 6)],  # Different obstacles
        )
        assert base_light_dark_environment != other_env

    def test_different_goal_state(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different goal states are not equal.

        Purpose: Validates that DiscreteLightDarkPOMDP environments with different goal positions are correctly identified as unequal

        Given: Base DiscreteLightDarkPOMDP with default goal_state=[10,5] and another with goal_state=[9,4]
        When: Equality comparison is performed between environments with different target positions
        Then: Environments are correctly identified as not equal due to different goal state configurations

        Test type: unit
        """
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            goal_state=np.array([9, 4]),  # Different goal state
        )
        assert base_light_dark_environment != other_env

    def test_different_start_state(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different start states are not equal.

        Purpose: Validates that DiscreteLightDarkPOMDP environments with different starting positions are correctly identified as unequal

        Given: Base DiscreteLightDarkPOMDP with default start_state=[0,5] and another with start_state=[1,4]
        When: Equality comparison is performed between environments with different initial positions
        Then: Environments are correctly identified as not equal due to different starting state configurations

        Test type: unit
        """
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            start_state=np.array([1, 4]),  # Different start state
        )
        assert base_light_dark_environment != other_env

    def test_different_rewards(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different rewards are not equal.

        Purpose: Validates that DiscreteLightDarkPOMDP environments with different reward structures are correctly identified as unequal

        Given: Base DiscreteLightDarkPOMDP with obstacle_reward=-10.0, goal_reward=10.0, fuel_cost=2.0 vs another with -20.0, 20.0, 3.0
        When: Equality comparison is performed between environments with different reward parameters
        Then: Environments are correctly identified as not equal due to different reward structure configurations

        Test type: unit
        """
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-20.0,  # Different obstacle reward
            goal_reward=20.0,  # Different goal reward
            fuel_cost=3.0,  # Different fuel cost
            grid_size=11,
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment != other_env

    def test_different_grid_size(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different grid sizes are not equal.

        Purpose: Validates that DiscreteLightDarkPOMDP environments with different grid dimensions are correctly identified as unequal

        Given: Base DiscreteLightDarkPOMDP with grid_size=11 (11x11 grid) and another with grid_size=15 (15x15 grid)
        When: Equality comparison is performed between environments with different world dimensions
        Then: Environments are correctly identified as not equal due to different grid size configurations

        Test type: unit
        """
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=15,  # Different grid size
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment != other_env

    def test_different_beacon_radius(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different beacon radii are not equal.

        Purpose: Validates that DiscreteLightDarkPOMDP environments with different beacon influence radii are correctly identified as unequal

        Given: Base DiscreteLightDarkPOMDP with default beacon_radius=1.0 and another with beacon_radius=2.0
        When: Equality comparison is performed between environments with different beacon detection ranges
        Then: Environments are correctly identified as not equal due to different beacon radius configurations

        Test type: unit
        """
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            beacon_radius=2.0,  # Different beacon radius
        )
        assert base_light_dark_environment != other_env

    def test_different_stochastic_reward(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that DiscreteLightDarkPOMDPs with different stochastic reward settings are not equal.

        Purpose: Validates that DiscreteLightDarkPOMDP environments with different reward stochasticity settings are correctly identified as unequal

        Given: Base DiscreteLightDarkPOMDP with is_stochastic_reward=True and another with is_stochastic_reward=False
        When: Equality comparison is performed between environments with different reward determinism settings
        Then: Environments are correctly identified as not equal due to different stochastic reward configurations

        Test type: unit
        """
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=False,  # Different stochastic reward setting
        )
        assert base_light_dark_environment != other_env

    def test_comparison_with_non_environment(
        self, base_light_dark_environment: DiscreteLightDarkPOMDP
    ):
        """Test comparison with non-Environment objects.

        Purpose: Validates that DiscreteLightDarkPOMDP equality returns False when compared with non-Environment objects

        Given: DiscreteLightDarkPOMDP environment and non-Environment objects (string, integer, None)
        When: Equality comparison is performed between environment and non-environment types
        Then: All comparisons return False for incompatible object types

        Test type: unit
        """
        assert base_light_dark_environment != "not an environment"
        assert base_light_dark_environment != 42
        assert base_light_dark_environment != None

    def test_missing_attributes(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test equality when attributes are missing.

        Purpose: Validates that DiscreteLightDarkPOMDP equality returns False when comparing with objects missing critical attributes

        Given: Complete DiscreteLightDarkPOMDP and identical environments with missing attributes (beacons, obstacles)
        When: Equality comparison is performed with environments missing critical configuration attributes
        Then: All comparisons return False when attributes are missing from comparison objects

        Test type: unit
        """
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
        )
        delattr(other_env, "beacons")
        assert base_light_dark_environment != other_env

        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
        )
        delattr(other_env, "obstacles")
        assert base_light_dark_environment != other_env

    def test_deep_copy_equality(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that a deep copy of DiscreteLightDarkPOMDP is equal to original.

        Purpose: Validates that DiscreteLightDarkPOMDP equality works correctly with deep copied objects and exhibits symmetry

        Given: Original DiscreteLightDarkPOMDP environment and its deep copy using copy.deepcopy
        When: Equality comparison is performed between original and deep copied environment
        Then: Both directions return True (original == copy and copy == original) confirming deep copy equality

        Test type: unit
        """
        import copy

        copied_env = copy.deepcopy(base_light_dark_environment)
        assert copied_env == base_light_dark_environment
        assert base_light_dark_environment == copied_env  # Test symmetry


class TestDiscreteLightDarkPOMDPConfigId:
    """Test suite for DiscreteLightDarkPOMDP config_id functionality."""

    def test_config_id_consistency(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that config_id is consistent for identical environments.

        Purpose: Validates that DiscreteLightDarkPOMDP config_id generation produces consistent hashes for identical configurations

        Given: Two DiscreteLightDarkPOMDP environments with identical parameters (discount=0.95, errors=0.05, grid=11x11, etc.)
        When: config_id is generated for both environment instances
        Then: Both environments produce the same config_id hash value

        Test type: configuration
        """
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment.config_id == other_env.config_id

    def test_config_id_different_discount_factor(
        self, base_light_dark_environment: DiscreteLightDarkPOMDP
    ):
        """Test that config_id changes with different discount factor.

        Purpose: Validates that DiscreteLightDarkPOMDP config_id generation produces different hashes for different discount factors

        Given: Base DiscreteLightDarkPOMDP with discount=0.95 and another with discount=0.8, otherwise identical
        When: config_id is generated for both environment instances with different discount factors
        Then: Different discount factors produce different config_id hash values

        Test type: configuration
        """
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.8,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment.config_id != other_env.config_id

    def test_config_id_different_parameters(
        self, base_light_dark_environment: DiscreteLightDarkPOMDP
    ):
        """Test that config_id changes with different parameters.

        Purpose: Validates that DiscreteLightDarkPOMDP config_id generation produces different hashes for different environment parameters

        Given: Base environment and multiple variations with different parameters (transition_error=0.1, observation_error=0.1, beacon positions)
        When: config_id is generated for each parameter variation
        Then: All parameter differences produce different config_id hash values

        Test type: configuration
        """
        # Test different transition error
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.1,  # Different
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment.config_id != other_env.config_id

        # Test different observation error
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.1,  # Different
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
        )
        assert base_light_dark_environment.config_id != other_env.config_id

        # Test different beacons
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            beacons=[
                (1, 1),
                (1, 6),
                (1, 11),
                (6, 1),
                (6, 6),
                (6, 11),
                (11, 1),
                (11, 6),
                (11, 11),
            ],  # Different
        )
        assert base_light_dark_environment.config_id != other_env.config_id

    def test_config_id_format(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that config_id is a valid SHA-256 hash.

        Purpose: Validates that DiscreteLightDarkPOMDP config_id follows proper SHA-256 hash format specification

        Given: DiscreteLightDarkPOMDP environment instance with configuration parameters
        When: config_id property generates hash value from environment configuration
        Then: Returns string with 64 characters, all valid hexadecimal digits (0-9, a-f)

        Test type: configuration
        """
        config_id = base_light_dark_environment.config_id
        assert isinstance(config_id, str)
        assert len(config_id) == 64  # SHA-256 hash length
        assert all(c in "0123456789abcdef" for c in config_id)  # Valid hex characters

    def test_config_id_deterministic(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that config_id is deterministic (same input always produces same output).

        Purpose: Validates that DiscreteLightDarkPOMDP config_id property returns consistent values across multiple accesses

        Given: Single DiscreteLightDarkPOMDP environment instance with fixed configuration parameters
        When: config_id property is accessed multiple times on the same instance
        Then: All accesses return identical config_id hash values (deterministic behavior)

        Test type: configuration
        """
        config_id1 = base_light_dark_environment.config_id
        config_id2 = base_light_dark_environment.config_id
        assert config_id1 == config_id2

    def test_config_id_order_invariance(self, base_light_dark_environment: DiscreteLightDarkPOMDP):
        """Test that config_id is invariant to the order of beacons and obstacles.

        Purpose: Validates config_id behavior for  order invariance

        Given: Belief objects with specific configurations
        When: Config IDs are generated or compared
        Then: Config IDs behave as expected (deterministic, unique, etc.)

        Test type: configuration
        """
        # Create environment with same beacons but in different order
        # Default beacons: [(0, 0), (0, 5), (0, 10), (5, 0), (5, 5), (5, 10), (10, 0), (10, 5), (10, 10)]
        # Reorder by swapping first and last
        beacons_reordered = [
            (10.0, 10.0),
            (0.0, 5.0),
            (0.0, 10.0),
            (5.0, 0.0),
            (5.0, 5.0),
            (5.0, 10.0),
            (10.0, 0.0),
            (10.0, 5.0),
            (0.0, 0.0),
        ]

        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            beacons=beacons_reordered,
        )
        assert base_light_dark_environment.config_id == other_env.config_id

        # Create environment with same obstacles but in different order
        # Default obstacles: [(3, 7), (5, 5)]
        # Reorder by swapping them
        obstacles_reordered = [(5.0, 5.0), (3.0, 7.0)]

        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            obstacles=obstacles_reordered,
        )
        assert base_light_dark_environment.config_id == other_env.config_id

        # Test both beacons and obstacles reordered together
        other_env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            transition_error_prob=0.05,
            observation_error_prob=0.05,
            obstacle_hit_probability=0.2,
            obstacle_reward=-10.0,
            goal_reward=10.0,
            fuel_cost=2.0,
            grid_size=11,
            is_stochastic_reward=True,
            beacons=beacons_reordered,
            obstacles=obstacles_reordered,
        )
        assert base_light_dark_environment.config_id == other_env.config_id


def test_initialization():
    """Test initialization with default parameters

    Purpose: Validates proper initialization of

    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        transition_error_prob=0.05,
        observation_error_prob=0.05,
        obstacle_hit_probability=0.2,
        obstacle_reward=-10.0,
        goal_reward=10.0,
        fuel_cost=2.0,
        grid_size=11,
        is_stochastic_reward=True,
    )

    # Check default parameters
    assert env.transition_error_prob == 0.05
    assert env.observation_error_prob == 0.05
    assert env.obstacle_hit_probability == 0.2
    assert env.obstacle_reward == -10.0
    assert env.goal_reward == 10.0
    assert env.fuel_cost == 2.0
    assert env.grid_size == 11
    assert env.beacon_radius == 1.0

    # Check default state positions
    assert np.array_equal(env.goal_state, np.array([10, 5]))
    assert np.array_equal(env.start_state, np.array([0, 5]))

    # Check default beacons
    expected_beacons = np.array([[0, 0, 0, 5, 5, 5, 10, 10, 10], [0, 5, 10, 0, 5, 10, 0, 5, 10]])
    assert np.array_equal(env.beacons, expected_beacons)

    # Check default obstacles (now 2xN format like beacons)
    expected_obstacles = np.array([[3, 5], [7, 5]])
    assert np.array_equal(env.obstacles, expected_obstacles)


def test_beacons_and_obstacles_array_structure():
    """Test that beacons and obstacles are numpy arrays with correct shapes after initialization.

    Purpose: Validates that DiscreteLightDarkPOMDP properly converts beacons and obstacles
    to numpy arrays with the expected dimensions

    Given: A DiscreteLightDarkPOMDP environment with default parameters
    When: Environment is initialized with default beacons and obstacles
    Then:
        - beacons is a 2xN numpy array where N is the number of beacons
        - obstacles is a 2xN numpy array where N is the number of obstacles (same format as beacons)
        - Both arrays have the correct number of elements

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        transition_error_prob=0.05,
        observation_error_prob=0.05,
        obstacle_hit_probability=0.2,
        obstacle_reward=-10.0,
        goal_reward=10.0,
        fuel_cost=2.0,
        grid_size=11,
        is_stochastic_reward=True,
    )

    # Test beacons structure
    assert isinstance(env.beacons, np.ndarray), "beacons should be a numpy array"
    assert env.beacons.ndim == 2, "beacons should be 2-dimensional"
    assert env.beacons.shape[0] == 2, "beacons should have 2 rows (x and y coordinates)"

    # Default beacons: [(0,0), (0,5), (0,10), (5,0), (5,5), (5,10), (10,0), (10,5), (10,10)]
    # So there should be 9 beacons
    expected_num_beacons = 9
    assert (
        env.beacons.shape[1] == expected_num_beacons
    ), f"beacons should have {expected_num_beacons} columns"

    # Test obstacles structure (now 2xN format like beacons)
    assert isinstance(env.obstacles, np.ndarray), "obstacles should be a numpy array"
    assert env.obstacles.ndim == 2, "obstacles should be 2-dimensional"
    assert env.obstacles.shape[0] == 2, "obstacles should have 2 rows (x and y coordinates)"

    # Default obstacles: [(3,7), (5,5)]
    # So there should be 2 obstacles
    expected_num_obstacles = 2
    assert (
        env.obstacles.shape[0] == expected_num_obstacles
    ), f"obstacles should have {expected_num_obstacles} rows"

    # Verify the coordinate structure
    # Beacons: first row should be x coordinates, second row should be y coordinates
    expected_beacon_x = [0, 0, 0, 5, 5, 5, 10, 10, 10]
    expected_beacon_y = [0, 5, 10, 0, 5, 10, 0, 5, 10]
    assert np.array_equal(
        env.beacons[0, :], expected_beacon_x
    ), "beacons first row should contain x coordinates"
    assert np.array_equal(
        env.beacons[1, :], expected_beacon_y
    ), "beacons second row should contain y coordinates"

    # Obstacles: now 2xN format like beacons (first row=x coords, second row=y coords)
    expected_obstacle_x = [3, 5]  # x coordinates of obstacles (3,7) and (5,5)
    expected_obstacle_y = [7, 5]  # y coordinates of obstacles (3,7) and (5,5)
    assert np.array_equal(
        env.obstacles[0, :], expected_obstacle_x
    ), "obstacles first row should contain x coordinates"
    assert np.array_equal(
        env.obstacles[1, :], expected_obstacle_y
    ), "obstacles second row should contain y coordinates"


def test_custom_beacons_and_obstacles_array_structure():
    """Test that custom beacons and obstacles are properly converted to numpy arrays with correct shapes.

    Purpose: Validates that DiscreteLightDarkPOMDP properly handles custom beacon and obstacle
    configurations and converts them to the expected numpy array format

    Given: A DiscreteLightDarkPOMDP environment with custom beacons and obstacles
    When: Environment is initialized with custom beacon and obstacle lists
    Then:
        - Custom beacons are converted to 2xN numpy array format
        - Custom obstacles are converted to Nx2 numpy array format
        - Arrays contain the correct custom coordinates

    Test type: unit
    """
    custom_beacons = [(1.0, 1.0), (1.0, 6.0), (6.0, 1.0), (6.0, 6.0)]
    custom_obstacles = [(2.0, 3.0), (4.0, 4.0), (7.0, 8.0)]

    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        transition_error_prob=0.05,
        observation_error_prob=0.05,
        obstacle_hit_probability=0.2,
        obstacle_reward=-10.0,
        goal_reward=10.0,
        fuel_cost=2.0,
        grid_size=11,
        is_stochastic_reward=True,
        beacons=custom_beacons,
        obstacles=custom_obstacles,
    )

    # Test custom beacons structure
    assert isinstance(env.beacons, np.ndarray), "custom beacons should be a numpy array"
    assert env.beacons.ndim == 2, "custom beacons should be 2-dimensional"
    assert env.beacons.shape[0] == 2, "custom beacons should have 2 rows (x and y coordinates)"
    assert env.beacons.shape[1] == len(
        custom_beacons
    ), f"custom beacons should have {len(custom_beacons)} columns"

    # Test custom obstacles structure (now 2xN format like beacons)
    assert isinstance(env.obstacles, np.ndarray), "custom obstacles should be a numpy array"
    assert env.obstacles.ndim == 2, "custom obstacles should be 2-dimensional"
    assert env.obstacles.shape[0] == 2, "custom obstacles should have 2 rows (x and y coordinates)"
    assert env.obstacles.shape[1] == len(
        custom_obstacles
    ), f"custom obstacles should have {len(custom_obstacles)} columns"

    # Verify custom coordinate structure
    # Beacons: first row should be x coordinates, second row should be y coordinates
    expected_custom_beacon_x = [1, 1, 6, 6]
    expected_custom_beacon_y = [1, 6, 1, 6]
    assert np.array_equal(
        env.beacons[0, :], expected_custom_beacon_x
    ), "custom beacons first row should contain x coordinates"
    assert np.array_equal(
        env.beacons[1, :], expected_custom_beacon_y
    ), "custom beacons second row should contain y coordinates"

    # Obstacles: now 2xN format like beacons (first row=x coords, second row=y coords)
    expected_custom_obstacle_x = [
        2,
        4,
        7,
    ]  # x coordinates of obstacles (2,3), (4,4), (7,8)
    expected_custom_obstacle_y = [
        3,
        4,
        8,
    ]  # y coordinates of obstacles (2,3), (4,4), (7,8)
    assert np.array_equal(
        env.obstacles[0, :], expected_custom_obstacle_x
    ), "custom obstacles first row should contain x coordinates"
    assert np.array_equal(
        env.obstacles[1, :], expected_custom_obstacle_y
    ), "custom obstacles second row should contain y coordinates"


def test_empty_beacons_and_obstacles():
    """Test that empty beacons and obstacles lists are properly handled.

    Purpose: Validates that DiscreteLightDarkPOMDP properly handles edge cases with empty
    beacon and obstacle configurations

    Given: A DiscreteLightDarkPOMDP environment with empty beacons and obstacles lists
    When: Environment is initialized with empty lists
    Then:
        - Empty beacons list results in 2x0 numpy array
        - Empty obstacles list results in 0x2 numpy array
        - Arrays are properly shaped for empty cases

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        transition_error_prob=0.05,
        observation_error_prob=0.05,
        obstacle_hit_probability=0.2,
        obstacle_reward=-10.0,
        goal_reward=10.0,
        fuel_cost=2.0,
        grid_size=11,
        is_stochastic_reward=True,
        beacons=[],  # Empty beacons
        obstacles=[],  # Empty obstacles
    )

    # Test empty beacons structure
    assert isinstance(env.beacons, np.ndarray), "empty beacons should be a numpy array"
    assert env.beacons.ndim == 2, "empty beacons should be 2-dimensional"
    assert env.beacons.shape[0] == 2, "empty beacons should have 2 rows"
    assert env.beacons.shape[1] == 0, "empty beacons should have 0 columns"

    # Test empty obstacles structure (now 2xN format like beacons)
    assert isinstance(env.obstacles, np.ndarray), "empty obstacles should be a numpy array"
    assert env.obstacles.ndim == 2, "empty obstacles should be 2-dimensional"
    assert env.obstacles.shape[0] == 2, "empty obstacles should have 2 rows"
    assert env.obstacles.shape[1] == 0, "empty obstacles should have 0 columns"

    # Verify arrays are empty but properly shaped
    assert env.beacons.size == 0, "empty beacons array should have size 0"
    assert env.obstacles.size == 0, "empty obstacles array should have size 0"


def test_state_transition_model():
    """Test state transition model

    Purpose: Validates that DiscreteLightDarkPOMDP state transition model correctly handles stochastic movement with error probabilities

    Given: DiscreteLightDarkPOMDP with transition_error_prob=0.1, state=[5,5], action="up"
    When: state_transition_model generates transition distribution with intended and unintended movements
    Then: Returns DiscreteDistribution with 90% probability for "up" and 10%/3 for other directions, correct state values

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(discount_factor=0.95, transition_error_prob=0.1)
    state = np.array([5, 5])

    # Test 'up' action
    dist = env.state_transition_model(state, "up")
    assert isinstance(dist, DiscreteDistribution)

    # Check probabilities
    # The implementation uses the order: up, down, right, left
    # For 'up' action, the first position (index 0) should have high probability
    expected_probs = np.array(
        [0.9, 0.1 / 3, 0.1 / 3, 0.1 / 3]
    )  # 0.9 for correct action, 0.1/3 for others
    assert np.allclose(dist.probs, expected_probs)

    # Check values
    expected_values = [
        state + np.array([0, 1]),  # up
        state + np.array([0, -1]),  # down
        state + np.array([1, 0]),  # right
        state + np.array([-1, 0]),  # left
    ]
    assert all(np.array_equal(v1, v2) for v1, v2 in zip(dist.values, expected_values))


def test_observation_model():
    """Test observation model

    Purpose: Validates that DiscreteLightDarkPOMDP observation model correctly handles partial observability with error probabilities

    Given: DiscreteLightDarkPOMDP with observation_error_prob=0.1, state=[5,5], action="up"
    When: observation_model generates observation distribution with true and noisy observations
    Then: Returns ObservationModel with highest probability (>0.5) for correct state observation

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(discount_factor=0.95, observation_error_prob=0.1)
    state = np.array([5, 5])

    # Test observation model
    dist = env.observation_model(state, "up")
    assert isinstance(dist, ObservationModel)

    # Check that the correct state has highest probability
    # Cast to specific type to access distribution attribute
    from typing import cast
    from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
        DiscreteLDObservationModel,
    )

    typed_dist = cast(DiscreteLDObservationModel, dist)

    correct_state_idx = len(typed_dist.distribution.values) - 1  # Last value is the correct state
    assert (
        typed_dist.distribution.probs[correct_state_idx] > 0.5
    )  # Should be 1 - observation_error_prob


def test_reward_function():
    """Test reward function

    Purpose: Validates that DiscreteLightDarkPOMDP reward function correctly calculates goal rewards, obstacle penalties, and fuel costs

    Given: DiscreteLightDarkPOMDP with obstacle_hit_probability=1.0, various state-action combinations
    When: Reward is calculated for goal-reaching, obstacle-hitting, and normal movement scenarios
    Then: Returns correct rewards: goal_reward-fuel_cost, obstacle penalty with distance cost, normal movement with distance cost

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(discount_factor=0.95, obstacle_hit_probability=1.0)

    # Test goal state reward
    state = np.array([9, 5])
    reward = env.reward(state, "right")  # Moves to [10, 5] which is goal
    assert reward == env.goal_reward - env.fuel_cost

    # Test obstacle hit reward
    state = np.array([4, 5])
    reward = env.reward(state, "right")  # Moves to [5, 5] which is obstacle
    assert (
        reward
        == -env.fuel_cost - np.linalg.norm(np.array([5, 5]) - env.goal_state) + env.obstacle_reward
    )

    # Test normal movement reward
    state = np.array([1, 1])
    reward = env.reward(state, "right")
    assert reward == -env.fuel_cost - np.linalg.norm(np.array([2, 1]) - env.goal_state)


def test_is_terminal():
    """Test terminal state detection

    Purpose: Validates that DiscreteLightDarkPOMDP correctly identifies terminal states including goal, obstacles, and boundary violations

    Given: DiscreteLightDarkPOMDP with goal state, obstacle positions, grid boundaries, and normal states
    When: is_terminal checks various state positions for termination conditions
    Then: Returns True for goal state, obstacle states, out-of-bounds states, False for normal navigation states

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(discount_factor=0.95)

    # Test goal state
    assert env.is_terminal(env.goal_state)

    # Test obstacle state
    assert env.is_terminal(env.obstacles[:, 0])  # First obstacle

    # Test out of grid state
    assert env.is_terminal(np.array([-1, 5]))
    assert env.is_terminal(np.array([12, 5]))  # grid_size + 1

    # Test non-terminal state
    assert not env.is_terminal(np.array([1, 1]))


def test_initial_distributions():
    """Test initial state and observation distributions

    Purpose: Validates that DiscreteLightDarkPOMDP provides correct initial state and observation distributions for episode initialization

    Given: DiscreteLightDarkPOMDP with default start_state=[0,5] and initial observation setup
    When: initial_state_dist and initial_observation_dist generate starting distributions
    Then: State distribution has probability 1.0 for start_state, observation distribution has probability 1.0 for initial value

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(discount_factor=0.95)

    # Test initial state distribution
    state_dist = env.initial_state_dist()
    assert isinstance(state_dist, DiscreteDistribution)
    assert np.array_equal(state_dist.values[0], env.start_state)
    assert state_dist.probs[0] == 1.0

    # Test initial observation distribution
    obs_dist = env.initial_observation_dist()
    assert isinstance(obs_dist, DiscreteDistribution)
    assert obs_dist.values[0] == 1.0
    assert obs_dist.probs[0] == 1.0


def test_get_actions():
    """Test action list retrieval

    Purpose: Validates that DiscreteLightDarkPOMDP provides correct set of available navigation actions

    Given: DiscreteLightDarkPOMDP environment with discrete action space
    When: get_actions retrieves the complete set of available actions
    Then: Returns action set {"up", "down", "right", "left"} for 4-directional grid navigation

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(discount_factor=0.95)
    actions = env.get_actions()
    assert set(actions) == {"up", "down", "right", "left"}


def test_visualize_path(tmp_path):
    """Test path visualization

    Purpose: Validates that DiscreteLightDarkPOMDP visualization creates animated GIF files showing agent path and belief evolution

    Given: DiscreteLightDarkPOMDP, path=[0,5]→[1,5]→[2,5]→[3,5], belief path with distributions, actions=["right","right","right"]
    When: visualize_path creates animation with agent trajectory and belief visualization
    Then: Creates GIF file at specified cache_path location with path animation

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(discount_factor=0.95)
    path = [np.array([0, 5]), np.array([1, 5]), np.array([2, 5]), np.array([3, 5])]

    # Create a simple belief path for testing
    agent_belief_path = [
        DiscreteDistribution(values=[path[i]], probs=np.array([1.0])) for i in range(len(path))
    ]

    # Create actions corresponding to the path
    actions = ["right", "right", "right"]  # 0->1, 1->2, 2->3

    # Test visualization with temporary path
    cache_path = tmp_path / "test_animation.gif"
    env.visualize_path(
        path=path,
        agent_belief_path=agent_belief_path,
        actions=actions,
        cache_path=cache_path,
    )

    # Verify file was created
    assert cache_path.exists()


def test_compute_metrics():
    """Test computation of metrics for different simulation histories

    Purpose: Validates that DiscreteLightDarkPOMDP computes performance metrics from simulation histories including success and failure rates

    Given: Three simulation histories - 2 reaching goal, 1 hitting obstacle, with proper StepData and belief sequences
    When: compute_metrics analyzes the simulation histories for performance statistics
    Then: Returns goal_reaching_rate=2/3 and obstacle_hit_rate=1/3 with confidence bounds

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(discount_factor=0.95)

    # Create test histories
    from POMDPPlanners.core.belief import WeightedParticleBelief
    from POMDPPlanners.core.policy import PolicyInfoVariable, PolicyRunData
    from POMDPPlanners.core.simulation import History, StepData

    # Create a simple belief for testing
    def create_test_belief(state):
        return WeightedParticleBelief(
            particles=[state],
            log_weights=np.array([1.0]),  # Using non-zero log weight
            resampling=False,
        )

    # History 1: Reaches goal in 3 steps
    history1 = History(
        [
            StepData(
                state=np.array([0, 5]),
                action="right",
                next_state=np.array([1, 5]),
                observation=np.array([1, 5]),
                reward=-2.0,
                belief=create_test_belief(np.array([0, 5])),
            ),
            StepData(
                state=np.array([1, 5]),
                action="right",
                next_state=np.array([2, 5]),
                observation=np.array([2, 5]),
                reward=-2.0,
                belief=create_test_belief(np.array([1, 5])),
            ),
            StepData(
                state=np.array([2, 5]),
                action="right",
                next_state=np.array([3, 5]),
                observation=np.array([3, 5]),
                reward=-2.0,
                belief=create_test_belief(np.array([2, 5])),
            ),
            StepData(
                state=np.array([10, 5]),
                action="right",
                next_state=np.array([10, 5]),
                observation=np.array([10, 5]),
                reward=8.0,  # Goal state
                belief=create_test_belief(np.array([10, 5])),
            ),
        ],
        discount_factor=0.95,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=4,
        reach_terminal_state=True,
        policy_run_data=PolicyRunData(info_variables=[]),
    )

    # History 2: Hits obstacle
    history2 = History(
        [
            StepData(
                state=np.array([0, 5]),
                action="right",
                next_state=np.array([1, 5]),
                observation=np.array([1, 5]),
                reward=-2.0,
                belief=create_test_belief(np.array([0, 5])),
            ),
            StepData(
                state=np.array([1, 5]),
                action="right",
                next_state=np.array([2, 5]),
                observation=np.array([2, 5]),
                reward=-2.0,
                belief=create_test_belief(np.array([1, 5])),
            ),
            StepData(
                state=np.array([5, 5]),
                action="right",
                next_state=np.array([5, 5]),
                observation=np.array([5, 5]),
                reward=-12.0,  # Obstacle state at (5, 5)
                belief=create_test_belief(np.array([5, 5])),
            ),
        ],
        discount_factor=0.95,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=3,
        reach_terminal_state=True,
        policy_run_data=PolicyRunData(info_variables=[]),
    )

    # History 3: Reaches goal in 5 steps, avoiding obstacle by going up
    history3 = History(
        [
            StepData(
                state=np.array([0, 5]),
                action="right",
                next_state=np.array([1, 5]),
                observation=np.array([1, 5]),
                reward=-2.0,
                belief=create_test_belief(np.array([0, 5])),
            ),
            StepData(
                state=np.array([1, 5]),
                action="right",
                next_state=np.array([2, 5]),
                observation=np.array([2, 5]),
                reward=-2.0,
                belief=create_test_belief(np.array([1, 5])),
            ),
            StepData(
                state=np.array([2, 5]),
                action="up",
                next_state=np.array([2, 6]),
                observation=np.array([2, 6]),
                reward=-2.0,
                belief=create_test_belief(np.array([2, 5])),
            ),
            StepData(
                state=np.array([2, 6]),
                action="right",
                next_state=np.array([3, 6]),
                observation=np.array([3, 6]),
                reward=-2.0,
                belief=create_test_belief(np.array([2, 6])),
            ),
            StepData(
                state=np.array([3, 6]),
                action="right",
                next_state=np.array([4, 6]),
                observation=np.array([4, 6]),
                reward=-2.0,
                belief=create_test_belief(np.array([3, 6])),
            ),
            StepData(
                state=np.array([10, 5]),
                action="right",
                next_state=np.array([10, 5]),
                observation=np.array([10, 5]),
                reward=8.0,  # Goal state
                belief=create_test_belief(np.array([10, 5])),
            ),
        ],
        discount_factor=0.95,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=6,
        reach_terminal_state=True,
        policy_run_data=PolicyRunData(info_variables=[]),
    )

    # Compute metrics
    metrics = env.compute_metrics([history1, history2, history3])

    # Convert metrics to dictionary for easier access
    metrics_dict = {metric.name: metric for metric in metrics}

    # Test goal reaching rate
    assert "goal_reaching_rate" in metrics_dict
    goal_rate = metrics_dict["goal_reaching_rate"]
    assert goal_rate.value == 2 / 3  # 2 out of 3 histories reach goal
    assert goal_rate.lower_confidence_bound <= goal_rate.value <= goal_rate.upper_confidence_bound

    # Test obstacle hit rate
    assert "obstacle_hit_rate" in metrics_dict
    obstacle_rate = metrics_dict["obstacle_hit_rate"]
    assert obstacle_rate.value == 1 / 3  # 1 out of 3 histories hits obstacle
    assert (
        obstacle_rate.lower_confidence_bound
        <= obstacle_rate.value
        <= obstacle_rate.upper_confidence_bound
    )
