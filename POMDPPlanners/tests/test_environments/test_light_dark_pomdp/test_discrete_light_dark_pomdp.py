"""Tests for Discrete Light Dark POMDP environment.

This module tests the Discrete Light Dark POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

# pylint: disable=too-many-lines

import copy
import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.policy import PolicyRunData
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.tests.test_utils.confidence_interval_utils import (
    verify_metrics_within_confidence_intervals,
)
from POMDPPlanners.tests.test_utils.metric_invariants_utils import (
    verify_history_returns_bounded,
    verify_metric_sanity,
    verify_return_shift_linearity,
)
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
    ObservationModelType,
)

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


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
        assert base_light_dark_environment is not None

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
    """Test state-transition log-probabilities for the discrete env.

    Purpose: Validates that DiscreteLightDarkPOMDP correctly models stochastic
    movement with error probabilities via env.transition_log_probability.

    Given: DiscreteLightDarkPOMDP with transition_error_prob=0.1, state=[5,5],
        action="up", and the four candidate next-states obtained via the
        four cardinal action vectors.
    When: env.transition_log_probability is called for the four candidates.
    Then: The "up" candidate has log(0.9), the other three have log(0.1/3),
        and probabilities sum to 1.

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(discount_factor=0.95, transition_error_prob=0.1)
    state = np.array([5, 5])

    candidates = [
        state + np.array([0, 1]),  # up
        state + np.array([0, -1]),  # down
        state + np.array([1, 0]),  # right
        state + np.array([-1, 0]),  # left
    ]
    candidates_arr = np.stack(candidates, axis=0)
    log_probs = env.transition_log_probability(state, "up", candidates_arr)
    probs = np.exp(log_probs)

    expected_probs = np.array([0.9, 0.1 / 3, 0.1 / 3, 0.1 / 3])
    assert np.allclose(probs, expected_probs)
    assert np.isclose(probs.sum(), 1.0)


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

    Purpose: Validates that DiscreteLightDarkPOMDP correctly identifies terminal states including goal and obstacles

    Given: DiscreteLightDarkPOMDP with goal state, obstacle positions, grid boundaries, and normal states
    When: is_terminal checks various state positions for termination conditions
    Then: Returns True for goal state and obstacle states, False for out-of-bounds states and normal navigation states

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(discount_factor=0.95)

    # Test goal state
    assert env.is_terminal(env.goal_state)

    # Test obstacle state
    assert env.is_terminal(env.obstacles[:, 0])  # First obstacle

    # Test out of grid state (no longer terminal)
    assert not env.is_terminal(np.array([-1, 5]))
    assert not env.is_terminal(np.array([12, 5]))  # grid_size + 1

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
        policy_run_data=[PolicyRunData(info_variables=[])],
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
        policy_run_data=[PolicyRunData(info_variables=[])],
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
        policy_run_data=[PolicyRunData(info_variables=[])],
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


def test_compute_metrics_values_within_confidence_intervals():
    """Test DiscreteLightDarkPOMDP metric values are inside CIs and pass invariants.

    Purpose: Validates that metrics produced by compute_metrics lie inside
        their CI bounds and that all structural invariants hold (rate-in-[0,1],
        counts >= 0, finite CI for n>=2, return-shift linearity).

    Given: A DiscreteLightDarkPOMDP and 3 hand-built histories with varied
        outcomes (goal-reaching, obstacle-hitting, all-safe).
    When: compute_metrics is called and the four invariant helpers are run.
    Then: All checks pass without raising.

    Test type: integration
    """
    env = DiscreteLightDarkPOMDP(discount_factor=0.95)

    def _make_belief(state: np.ndarray) -> WeightedParticleBelief:
        return WeightedParticleBelief(
            particles=[state], log_weights=np.array([1.0]), resampling=False
        )

    # History 0: reach goal cleanly.
    goal_steps = [
        StepData(
            state=np.array([0, 5]),
            action="right",
            next_state=np.array([1, 5]),
            observation=np.array([1, 5]),
            reward=-2.0,
            belief=_make_belief(np.array([0, 5])),
        ),
        StepData(
            state=np.array([1, 5]),
            action="right",
            next_state=np.array([2, 5]),
            observation=np.array([2, 5]),
            reward=-2.0,
            belief=_make_belief(np.array([1, 5])),
        ),
        StepData(
            state=np.array([10, 5]),
            action="right",
            next_state=np.array([10, 5]),
            observation=np.array([10, 5]),
            reward=8.0,
            belief=_make_belief(np.array([10, 5])),
        ),
    ]

    # History 1: hit an obstacle at (5, 5).
    obstacle_steps = [
        StepData(
            state=np.array([0, 5]),
            action="right",
            next_state=np.array([1, 5]),
            observation=np.array([1, 5]),
            reward=-2.0,
            belief=_make_belief(np.array([0, 5])),
        ),
        StepData(
            state=np.array([5, 5]),
            action="right",
            next_state=np.array([5, 5]),
            observation=np.array([5, 5]),
            reward=-12.0,
            belief=_make_belief(np.array([5, 5])),
        ),
    ]

    # History 2: all-safe (no goal, no obstacle).
    safe_steps = [
        StepData(
            state=np.array([0, 5]),
            action="up",
            next_state=np.array([0, 6]),
            observation=np.array([0, 6]),
            reward=-2.0,
            belief=_make_belief(np.array([0, 5])),
        ),
        StepData(
            state=np.array([0, 6]),
            action="right",
            next_state=np.array([1, 6]),
            observation=np.array([1, 6]),
            reward=-2.0,
            belief=_make_belief(np.array([0, 6])),
        ),
    ]

    histories = []
    for steps, reach_terminal in (
        (goal_steps, True),
        (obstacle_steps, True),
        (safe_steps, False),
    ):
        histories.append(
            History(
                history=steps,
                discount_factor=0.95,
                average_state_sampling_time=0.0,
                average_action_time=0.0,
                average_observation_time=0.0,
                average_belief_update_time=0.0,
                average_reward_time=0.0,
                actual_num_steps=len(steps),
                reach_terminal_state=reach_terminal,
                policy_run_data=[PolicyRunData(info_variables=[])],
            )
        )

    metrics = env.compute_metrics(histories)
    verify_metrics_within_confidence_intervals(metrics)
    verify_metric_sanity(metrics, histories, env)
    verify_history_returns_bounded(histories, env)
    verify_return_shift_linearity(histories, env, shift=1.5)


def test_normal_observation_model():
    """Test the env's NORMAL observation sampling always returns ndarray observations.

    Purpose: Validates that under ObservationModelType.NORMAL, the env's
    sample_observation never returns the "None" sentinel.

    Given: A DiscreteLightDarkPOMDP env with ObservationModelType.NORMAL.
    When: env.sample_observation is called for several samples.
    Then: All returned observations are ndarrays.

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        observation_model_type=ObservationModelType.NORMAL,
    )
    assert env.observation_model_type == ObservationModelType.NORMAL

    state = np.array([5, 5])
    action = "up"
    observations = env.sample_observation(state, action, n_samples=10)
    assert isinstance(observations, np.ndarray)
    assert observations.shape == (10, 2)


def test_no_obs_in_dark_observation_model():
    """Test the env's NO_OBS_IN_DARK observation sampling near vs far beacon.

    Purpose: Validates that NO_OBS_IN_DARK returns real observations when
    the next_state is near a beacon and "None" when far.

    Given: A DiscreteLightDarkPOMDP env with ObservationModelType.NO_OBS_IN_DARK.
    When: env.sample_observation is called for a near-beacon and a
        far-from-beacon next_state.
    Then: Near returns ndarray observations; far returns "None" sentinels.

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        observation_model_type=ObservationModelType.NO_OBS_IN_DARK,
    )
    assert env.observation_model_type == ObservationModelType.NO_OBS_IN_DARK
    action = "up"

    # Near beacon (0,0).
    state_near = np.array([0, 0])
    observations_near = env.sample_observation(state_near, action, n_samples=10)
    assert all(isinstance(obs, np.ndarray) for obs in observations_near)

    # Far from any beacon.
    state_far = np.array([3, 3])
    observations_far = env.sample_observation(state_far, action, n_samples=10)
    assert all(obs == "None" for obs in observations_far)


def test_default_observation_model_type():
    """Test that the default observation model type is NORMAL.

    Purpose: Validates backward-compatibility default observation_model_type.

    Given: A DiscreteLightDarkPOMDP env without explicit observation_model_type.
    When: env.observation_model_type is read and env.sample_observation is
        called.
    Then: Default is NORMAL and sampling returns an ndarray observation.

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(discount_factor=0.95)
    assert env.observation_model_type == ObservationModelType.NORMAL

    obs = env.sample_observation(np.array([5, 5]), "up")
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)


def test_observation_model_type_equality():
    """Test that environments with different observation model types are not equal."""
    env1 = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        transition_error_prob=0.05,
        observation_error_prob=0.05,
        observation_model_type=ObservationModelType.NORMAL,
    )
    env2 = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        transition_error_prob=0.05,
        observation_error_prob=0.05,
        observation_model_type=ObservationModelType.NO_OBS_IN_DARK,
    )
    assert env1 != env2, "Environments with different observation model types should not be equal"

    # Test distance-based vs others
    env3 = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        transition_error_prob=0.05,
        observation_error_prob=0.05,
        observation_model_type=ObservationModelType.DISTANCE_BASED,
    )
    assert env1 != env3, "Distance-based should not equal normal"
    assert env2 != env3, "Distance-based should not equal no obs in dark"


def test_observation_model_type_config_id():
    """Test that config_id changes with different observation model types."""
    env1 = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        transition_error_prob=0.05,
        observation_error_prob=0.05,
        observation_model_type=ObservationModelType.NORMAL,
    )
    env2 = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        transition_error_prob=0.05,
        observation_error_prob=0.05,
        observation_model_type=ObservationModelType.NO_OBS_IN_DARK,
    )
    assert (
        env1.config_id != env2.config_id
    ), "Different observation model types should produce different config_ids"

    # Test distance-based produces different config_id
    env3 = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        transition_error_prob=0.05,
        observation_error_prob=0.05,
        observation_model_type=ObservationModelType.DISTANCE_BASED,
    )
    assert (
        env1.config_id != env3.config_id
    ), "Distance-based should produce different config_id from normal"
    assert (
        env2.config_id != env3.config_id
    ), "Distance-based should produce different config_id from no obs in dark"


def test_distance_based_observation_model():
    """Test the env's DISTANCE_BASED sampling and continuous scaling.

    Purpose: Validates that the discrete env's DISTANCE_BASED model returns
    real observations near the beacon (with the correct continuous error
    scaling) and "None" when min_distance > beacon_radius.

    Given: A DiscreteLightDarkPOMDP env with DISTANCE_BASED, observation_error_prob=0.05.
    When: env.sample_observation is called for a near-beacon, far-beacon, and
        at-beacon-radius next_state, and env.observation_log_probability is
        used to read off the correct-state mass for the scaling check.
    Then: At distance 0 from a beacon, the correct-state mass equals
        ``1 - observation_error_prob * 0.0001`` (min_factor); at exactly
        beacon_radius the mass equals ``1 - observation_error_prob`` (full);
        far returns "None" sentinels.

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        observation_error_prob=0.05,
        observation_model_type=ObservationModelType.DISTANCE_BASED,
    )
    assert env.observation_model_type == ObservationModelType.DISTANCE_BASED
    action = "up"

    # Near beacon at (0,0): observations are real arrays.
    state_near = np.array([0, 0])
    observations_near = env.sample_observation(state_near, action, n_samples=10)
    assert all(isinstance(obs, np.ndarray) for obs in observations_near)

    # Continuous scaling at distance 0: factor = min_factor = 0.0001.
    correct_log_prob_near = env.observation_log_probability(state_near, action, [state_near])[0]
    expected_correct_prob_near = 1.0 - 0.05 * 0.0001
    assert np.isclose(np.exp(correct_log_prob_near), expected_correct_prob_near, atol=1e-5)

    # Far from beacons: returns "None".
    state_far = np.array([3, 3])
    observations_far = env.sample_observation(state_far, action, n_samples=10)
    assert all(obs == "None" for obs in observations_far)

    # Exactly at beacon_radius: real observations, factor = 1.0.
    state_at_radius = np.array([1, 0])
    observations_at_radius = env.sample_observation(state_at_radius, action, n_samples=10)
    assert all(isinstance(obs, np.ndarray) for obs in observations_at_radius)

    correct_log_prob_at_radius = env.observation_log_probability(
        state_at_radius, action, [state_at_radius]
    )[0]
    expected_correct_prob_at_radius = 1.0 - 0.05 * 1.0
    assert np.isclose(
        np.exp(correct_log_prob_at_radius), expected_correct_prob_at_radius, atol=1e-10
    )


def test_distance_based_observation_model_continuous_scaling():
    """Test continuous scaling of DISTANCE_BASED as a function of distance.

    Purpose: Validates that the env's DISTANCE_BASED log-probability for
    the correct (no-noise) observation interpolates linearly between
    ``min_factor`` (at distance 0) and ``1.0`` (at distance beacon_radius).

    Given: A DiscreteLightDarkPOMDP env with DISTANCE_BASED,
        observation_error_prob=0.1, beacon_radius=2.0.
    When: env.observation_log_probability is queried for the correct
        observation at distances 0.0, 0.5, 1.0, 1.5, 2.0 from the beacon.
    Then: The implied error factor (1 - p_correct)/0.1 equals 0.0001 at
        distance 0, equals 1.0 at distance 2.0, and is strictly increasing.

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        observation_error_prob=0.1,
        beacon_radius=2.0,
        observation_model_type=ObservationModelType.DISTANCE_BASED,
    )
    base_error_prob = 0.1
    action = "up"
    test_distances = [0.0, 0.5, 1.0, 1.5, 2.0]
    error_factors: list = []
    for d in test_distances:
        state = np.array([d, 0.0])
        log_p = env.observation_log_probability(state, action, [state])[0]
        correct_prob = float(np.exp(log_p))
        error_factors.append((1.0 - correct_prob) / base_error_prob)

    min_factor = 0.0001
    assert np.isclose(error_factors[0], min_factor, atol=1e-5)
    assert np.isclose(error_factors[-1], 1.0, atol=1e-10)
    # Intermediate values are strictly inside [min_factor, 1.0].
    for mid in error_factors[1:-1]:
        assert min_factor < mid < 1.0
    # Monotonic non-decreasing.
    for i in range(len(error_factors) - 1):
        assert error_factors[i] <= error_factors[i + 1]


def test_distance_based_observation_model_probability():
    """Test DISTANCE_BASED log-probability for "None" and real observations.

    Purpose: Validates env.observation_log_probability semantics for the
    DISTANCE_BASED discrete model.

    Given: A DiscreteLightDarkPOMDP env with DISTANCE_BASED and beacon_radius=1.0.
    When: env.observation_log_probability is queried with "None" and a real
        observation for near and far next_state.
    Then: prob(None|far)=1 (log 0); prob(None|near)=0 (-inf); prob(real|near)
        is finite; prob(real|far)=0 (-inf).

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        beacon_radius=1.0,
        observation_model_type=ObservationModelType.DISTANCE_BASED,
    )
    action = "up"

    state_far = np.array([3, 3])
    state_near = np.array([0, 0])

    log_none_far = env.observation_log_probability(state_far, action, ["None"])[0]
    log_none_near = env.observation_log_probability(state_near, action, ["None"])[0]
    assert np.isclose(log_none_far, 0.0)  # log(1)
    assert log_none_near == -np.inf  # log(0)

    obs_value = np.array([0, 0])
    log_real_near = env.observation_log_probability(state_near, action, [obs_value])[0]
    log_real_far = env.observation_log_probability(state_far, action, [obs_value])[0]
    assert np.isfinite(log_real_near)
    assert log_real_far == -np.inf


def test_reward_batch_matches_scalar(base_light_dark_environment):
    """Test that DiscreteLightDarkPOMDP reward_batch matches scalar reward with same seed.

    Purpose: Validates vectorized reward_batch gives same results as scalar reward

    Given: A DiscreteLightDarkPOMDP environment and 100 random integer states on the grid
    When: reward_batch and scalar reward calls are made with identical seeds
    Then: Both produce identical reward arrays and output shape is (N,)

    Test type: unit
    """
    env = base_light_dark_environment
    rng = np.random.RandomState(0)
    states = rng.randint(0, env.grid_size + 1, (100, 2)).astype(float)
    action = "up"

    np.random.seed(99)
    batch_rewards = env.reward_batch(states, action)
    assert batch_rewards.shape == (100,)

    np.random.seed(99)
    expected = np.array([env.reward(states[i], action) for i in range(100)])
    np.testing.assert_allclose(batch_rewards, expected)

    # Test with N=1
    np.random.seed(42)
    single = env.reward_batch(states[:1], action)
    assert single.shape == (1,)
    np.random.seed(42)
    np.testing.assert_allclose(single[0], env.reward(states[0], action))


# ---------------------------------------------------------------------------
# Batch sampling and log-probability API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [1, 5, 100])
def test_discrete_sample_next_state_n_samples_shapes(n_samples):
    """env.sample_next_state honours n_samples and returns the right shapes.

    Purpose: Validates the shape contract of the discrete env's sample_next_state.

    Given: A DiscreteLightDarkPOMDP env and a (state, action) pair.
    When: env.sample_next_state(state, action, n_samples=N) is called for
        N in {1, 5, 100}.
    Then: For n_samples=1 a (2,) ndarray; for n_samples>1 a (N, 2) ndarray.

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(discount_factor=0.95)
    state = np.array([2, 3])
    action = "up"

    direct = env.sample_next_state(state, action, n_samples=n_samples)
    if n_samples == 1:
        assert isinstance(direct, np.ndarray)
        assert direct.shape == (2,)
    else:
        assert isinstance(direct, np.ndarray)
        assert direct.shape == (n_samples, 2)


@pytest.mark.parametrize("n_samples", [1, 5, 100])
def test_discrete_sample_observation_n_samples_shapes_normal(n_samples):
    """env.sample_observation honours n_samples for the NORMAL discrete model.

    Purpose: Validates the shape contract for the discrete env's
    sample_observation under ObservationModelType.NORMAL.

    Given: A DiscreteLightDarkPOMDP env (NORMAL) and a near-beacon next_state.
    When: env.sample_observation(ns, a, n_samples=N) is called for
        N in {1, 5, 100}.
    Then: For n_samples=1 a (2,) ndarray; for n_samples>1 a (N, 2) ndarray.

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        beacons=[(0, 0), (5, 5)],
        beacon_radius=1.5,
        observation_model_type=ObservationModelType.NORMAL,
    )
    next_state = np.array([5, 5])
    action = "up"

    direct = env.sample_observation(next_state, action, n_samples=n_samples)
    if n_samples == 1:
        assert isinstance(direct, np.ndarray)
        assert direct.shape == (2,)
    else:
        assert isinstance(direct, np.ndarray)
        assert direct.shape == (n_samples, 2)


def test_discrete_sample_observation_n_samples_for_non_normal():
    """env.sample_observation honours n_samples for non-NORMAL discrete types.

    Purpose: Validates that NO_OBS_IN_DARK and DISTANCE_BASED return lists of
    length n_samples whose elements are either ndarray observations or "None".

    Given: Two DiscreteLightDarkPOMDP envs, one per non-NORMAL type, and a
        near-beacon next_state.
    When: env.sample_observation(next_state, action, n_samples=4) is called.
    Then: Returned list has length 4 with each element an ndarray of shape
        (2,) or the "None" sentinel.

    Test type: unit
    """
    for obs_type in (
        ObservationModelType.NO_OBS_IN_DARK,
        ObservationModelType.DISTANCE_BASED,
    ):
        env = DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            beacons=[(0, 0), (5, 5)],
            beacon_radius=1.5,
            observation_model_type=obs_type,
        )
        next_state = np.array([5, 5])
        action = "up"
        direct = env.sample_observation(next_state, action, n_samples=4)
        assert len(direct) == 4
        for value in direct:
            assert isinstance(value, np.ndarray) or value == "None"


# ---------------------------------------------------------------------------
# Batch API: sample_next_state_batch and observation_log_probability_per_state
# ---------------------------------------------------------------------------


def test_discrete_sample_next_state_batch_matches_scalar_loop(base_light_dark_environment):
    """sample_next_state_batch under a fixed seed matches per-particle scalar loop.

    Purpose: Validates that DiscreteLightDarkPOMDP.sample_next_state_batch draws
        N next-states in one vectorised call and the results match calling
        sample_next_state once per particle under the same RNG sequence.

    Given: 24 random grid-integer states, action="up", and identical numpy seeds
    When: sample_next_state_batch is called vs. a manual loop of sample_next_state calls
    Then: Output shape is (N, 2) and each row is array-equal to the scalar loop result

    Test type: unit
    """
    env = base_light_dark_environment
    rng = np.random.RandomState(7)
    states = rng.randint(0, env.grid_size + 1, (24, 2)).astype(float)
    action = "up"

    np.random.seed(42)
    batch = env.sample_next_state_batch(states, action)
    assert isinstance(batch, np.ndarray)
    assert batch.shape == (24, 2)

    np.random.seed(42)
    scalar_loop = np.stack([env.sample_next_state(states[i], action) for i in range(24)], axis=0)
    np.testing.assert_array_equal(batch, scalar_loop)


def test_discrete_sample_next_state_batch_action_coverage(base_light_dark_environment):
    """sample_next_state_batch works for every discrete action direction.

    Purpose: Validates sample_next_state_batch for all four actions

    Given: A DiscreteLightDarkPOMDP env and a fixed state, N=16 identical particles
    When: sample_next_state_batch is called for each of the four actions
    Then: Output shape is (16, 2) and all outputs are within valid grid bounds

    Test type: unit
    """
    env = base_light_dark_environment
    states = np.tile(np.array([5.0, 5.0]), (16, 1))
    for action in env.get_actions():
        result = env.sample_next_state_batch(states, action)
        assert isinstance(result, np.ndarray)
        assert result.shape == (16, 2)


def test_discrete_observation_log_probability_per_state_matches_scalar(
    base_light_dark_environment,
):
    """observation_log_probability_per_state matches scalar-loop reference.

    Purpose: Validates that DiscreteLightDarkPOMDP.observation_log_probability_per_state
        returns the same log-probs as calling observation_log_probability once per
        next-state with a single-element observation list.

    Given: 16 random grid-integer next-states, action="up", and a fixed observation
        (the state-itself which is the "no-noise" candidate)
    When: observation_log_probability_per_state is called and compared to scalar loop
    Then: Output shape is (16,) and allclose within atol=1e-12

    Test type: unit
    """
    env = base_light_dark_environment
    rng = np.random.RandomState(99)
    next_states = rng.randint(0, env.grid_size + 1, (16, 2)).astype(float)
    action = "up"
    # Use the first next_state as a fixed observation (deterministic, no RNG needed)
    observation = next_states[0].copy()

    batch = env.observation_log_probability_per_state(next_states, action, observation)
    assert batch.shape == (16,)

    scalar = np.array(
        [
            env.observation_log_probability(next_states[i], action, [observation])[0]
            for i in range(16)
        ]
    )
    np.testing.assert_allclose(batch, scalar, atol=1e-12)


def test_discrete_observation_log_probability_per_state_near_vs_far(
    base_light_dark_environment,
):
    """observation_log_probability_per_state distinguishes near-beacon from far-beacon.

    Purpose: Validates that per-state log-probs correctly switch between the
        near-beacon and far-beacon probability tables based on each next-state's
        distance to the closest beacon.

    Given: A DiscreteLightDarkPOMDP env, two next-states (one near a beacon, one far),
        action="up", and a common observation (the near-beacon state itself)
    When: observation_log_probability_per_state is called with both next-states
    Then: The near-beacon state yields a higher log-prob than the far-beacon state

    Test type: unit
    """
    env = base_light_dark_environment
    near_state = np.array([0.0, 0.0])  # directly on first beacon
    far_state = np.array([6.0, 3.0])  # away from all beacons
    next_states = np.stack([near_state, far_state], axis=0)
    action = "up"
    # observation = near_state itself (the "no noise" candidate for near_state)
    observation = near_state.copy()

    result = env.observation_log_probability_per_state(next_states, action, observation)
    assert result.shape == (2,)
    # The near-beacon state should assign higher (or equal) probability to
    # its own position as the "correct" observation than the far state does.
    # Both can be -inf if observation is off-distribution for far_state, that's fine.
    scalar_near = env.observation_log_probability(near_state, action, [observation])[0]
    scalar_far = env.observation_log_probability(far_state, action, [observation])[0]
    np.testing.assert_allclose(result[0], scalar_near, atol=1e-12)
    np.testing.assert_allclose(result[1], scalar_far, atol=1e-12)


# ===========================================================================
# Feature-driven coverage added by /test-environment skill:
#   - sample/PDF consistency for each ObservationModelType
#   - sample-in-space sanity per ObservationModelType
# ===========================================================================


def _candidate_observations(env: DiscreteLightDarkPOMDP, next_state: np.ndarray) -> np.ndarray:
    """Return the 5 candidate observations for ``next_state`` (4 neighbors + self).

    Mirrors the candidate set used by sample_observation / observation_log_probability
    for the discrete-LD env: ``[next_state + action_vector_i for i in 0..3] + [next_state]``,
    in the same order as ``env.actions`` so candidate j matches probs[j].
    """
    candidates = [
        next_state + env.action_to_vector[a] for a in env.actions  # type: ignore[attr-defined]
    ]
    candidates.append(next_state)
    return np.stack(candidates, axis=0)


def _empirical_frequency_per_candidate(samples: list, candidates: np.ndarray) -> np.ndarray:
    """Empirical frequency of each candidate observation among samples (ignores 'None')."""
    counts = np.zeros(len(candidates), dtype=np.float64)
    for sample in samples:
        if isinstance(sample, str):
            continue
        sample_arr = np.asarray(sample)
        for j, cand in enumerate(candidates):
            if np.array_equal(sample_arr, cand):
                counts[j] += 1.0
                break
    return counts / max(len(samples), 1)


def test_discrete_sample_observation_frequencies_match_log_probability_normal():
    """NORMAL observation sample frequencies match exp(observation_log_probability).

    Purpose: Validates that DiscreteLightDarkPOMDP.sample_observation and
        observation_log_probability describe the same discrete observation
        distribution under the NORMAL model — both at a near-beacon next_state
        and a far-beacon next_state, exercising the near vs far probability
        tables (_obs_probs_near vs _obs_probs_far) on both code paths.

    Given: A default DiscreteLightDarkPOMDP. Anchors: near=[5,5] (on a beacon
        with distance 0 < beacon_radius=1) and far=[3,2] (min beacon distance
        ~2.24 > beacon_radius). Action="up" (unused in observation distribution
        but required by the API).
    When: 5000 observations are sampled at each anchor; analytic log-probs
        evaluated on the 5 candidate observations.
    Then: For each candidate, the empirical frequency is within Wilson-style
        tolerance 3 / sqrt(N) of exp(log_prob).

    Test type: unit
    """
    np.random.seed(101)
    env = DiscreteLightDarkPOMDP(discount_factor=0.95)
    action = "up"
    n_samples = 5_000
    tol = 3.0 / np.sqrt(n_samples)

    for next_state, label in (
        (np.array([5.0, 5.0]), "near"),
        (np.array([3.0, 2.0]), "far"),
    ):
        samples = env.sample_observation(next_state, action, n_samples=n_samples)
        candidates = _candidate_observations(env, next_state)
        empirical = _empirical_frequency_per_candidate(list(samples), candidates)
        analytic = np.exp(env.observation_log_probability(next_state, action, candidates))
        for j, (emp, prob) in enumerate(zip(empirical, analytic)):
            assert abs(emp - prob) < tol, (
                f"{label} candidate {j}: empirical={emp:.4f} vs analytic={prob:.4f} "
                f"(tol={tol:.4f})"
            )


def test_discrete_sample_observation_frequencies_match_log_probability_no_obs_in_dark():
    """NO_OBS_IN_DARK sampling and log-prob agree on both branches.

    Purpose: Validates the mixture contract of NO_OBS_IN_DARK: far from any
        beacon every sample is the "None" sentinel and log_prob("None")=0;
        near a beacon samples are drawn from the near-beacon discrete
        distribution and log_prob("None")=-inf with empirical frequencies
        matching exp(log_prob) for each candidate.

    Given: An env with ObservationModelType.NO_OBS_IN_DARK. Anchors:
        near=[5,5] (beacon distance 0) and far=[3,2] (>beacon_radius). Action
        "up".
    When: 5000 samples drawn at each anchor.
    Then: Far: 100% "None" with analytic log_prob("None")=0. Near: 0% "None"
        with analytic log_prob("None")=-inf, and per-candidate empirical
        frequencies within 3/sqrt(N) of exp(log_prob).

    Test type: unit
    """
    np.random.seed(202)
    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        observation_model_type=ObservationModelType.NO_OBS_IN_DARK,
    )
    action = "up"
    n_samples = 5_000
    tol = 3.0 / np.sqrt(n_samples)

    far = np.array([3.0, 2.0])
    far_samples = env.sample_observation(far, action, n_samples=n_samples)
    assert all(value == "None" for value in far_samples)
    far_none_log_prob = env.observation_log_probability(far, action, ["None"])
    assert far_none_log_prob[0] == pytest.approx(0.0, abs=1e-12)

    near = np.array([5.0, 5.0])
    near_samples = env.sample_observation(near, action, n_samples=n_samples)
    assert all(isinstance(value, np.ndarray) for value in near_samples)
    near_none_log_prob = env.observation_log_probability(near, action, ["None"])
    assert near_none_log_prob[0] == -np.inf

    candidates = _candidate_observations(env, near)
    empirical = _empirical_frequency_per_candidate(list(near_samples), candidates)
    analytic = np.exp(env.observation_log_probability(near, action, candidates))
    for j, (emp, prob) in enumerate(zip(empirical, analytic)):
        assert (
            abs(emp - prob) < tol
        ), f"near candidate {j}: empirical={emp:.4f} vs analytic={prob:.4f} (tol={tol:.4f})"


def test_discrete_sample_observation_frequencies_match_log_probability_distance_based():
    """DISTANCE_BASED sampling and log-prob agree on far ('None') and near branches.

    Purpose: Validates the mixture contract of DISTANCE_BASED: when min beacon
        distance > beacon_radius every sample is "None" and log_prob("None")=0;
        otherwise samples are drawn from a distance-scaled discrete distribution
        and log_prob("None")=-inf, with empirical frequencies matching
        exp(log_prob) for each candidate.

    Given: An env with ObservationModelType.DISTANCE_BASED. Anchors: near=[5,5]
        (min beacon distance 0 <= beacon_radius=1) and far=[3,2]
        (~2.24 > beacon_radius). Action "up".
    When: 5000 samples drawn at each anchor.
    Then: Far: 100% "None" with analytic log_prob("None")=0. Near: 0% "None"
        with analytic log_prob("None")=-inf and per-candidate empirical
        frequencies within 3/sqrt(N) of exp(log_prob).

    Test type: unit
    """
    np.random.seed(303)
    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        observation_model_type=ObservationModelType.DISTANCE_BASED,
    )
    action = "up"
    n_samples = 5_000
    tol = 3.0 / np.sqrt(n_samples)

    far = np.array([3.0, 2.0])
    far_samples = env.sample_observation(far, action, n_samples=n_samples)
    assert all(value == "None" for value in far_samples)
    far_none_log_prob = env.observation_log_probability(far, action, ["None"])
    assert far_none_log_prob[0] == pytest.approx(0.0, abs=1e-12)

    near = np.array([5.0, 5.0])
    near_samples = env.sample_observation(near, action, n_samples=n_samples)
    assert all(isinstance(value, np.ndarray) for value in near_samples)
    near_none_log_prob = env.observation_log_probability(near, action, ["None"])
    assert near_none_log_prob[0] == -np.inf

    candidates = _candidate_observations(env, near)
    empirical = _empirical_frequency_per_candidate(list(near_samples), candidates)
    analytic = np.exp(env.observation_log_probability(near, action, candidates))
    for j, (emp, prob) in enumerate(zip(empirical, analytic)):
        assert (
            abs(emp - prob) < tol
        ), f"near candidate {j}: empirical={emp:.4f} vs analytic={prob:.4f} (tol={tol:.4f})"


@pytest.mark.parametrize("observation_model_type", list(ObservationModelType))
def test_discrete_sample_outputs_lie_in_grid_or_are_none_sentinel(
    observation_model_type: ObservationModelType,
):
    """Sampled next-states are in-grid 2D coords; observations are 'None' or in-grid.

    Purpose: Catches drift in the discrete sampler clipping/sentinel contract.
        Every transition must remain a 2D coord; every observation must be
        either an in-grid 2D coord or the literal "None" sentinel string.

    Given: An env with the parametrized observation_model_type, default
        beacons. Anchor next_state=[5,5] (near a beacon, valid for all three
        models). Action "up".
    When: 500 transitions are drawn from a representative state and 500
        observations from a representative next_state.
    Then: All transition samples are integer-valued 2D coords; all observations
        are either "None" or 2D coords inside [0, grid_size] (the discrete
        sampler does not clip but valid candidates always land in the grid for
        the chosen anchor).

    Test type: unit
    """
    np.random.seed(404)
    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        observation_model_type=observation_model_type,
    )
    state = np.array([5.0, 5.0])
    action = "up"
    n_samples = 500

    transitions = env.sample_next_state(state, action, n_samples=n_samples)
    assert transitions.shape == (n_samples, 2)
    assert np.all(np.isfinite(transitions))

    observations = env.sample_observation(state, action, n_samples=n_samples)
    assert len(observations) == n_samples
    for value in observations:
        if isinstance(value, str):
            assert value == "None"
            continue
        assert isinstance(value, np.ndarray)
        assert value.shape == (2,)
        assert np.all(np.isfinite(value))
        assert np.all(value >= 0.0)
        assert np.all(value <= float(env.grid_size))
