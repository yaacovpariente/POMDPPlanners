"""Tests for action samplers utility functions.

This module tests the action sampler implementations used for continuous action
space sampling in POMDP planners, particularly for PFT-DPW integration.
"""

import json
import pickle
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.core.tree import BeliefNode
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.utils.action_samplers import (
    DiscreteActionSampler,
    UnitCircleActionSampler,
)

np.random.seed(42)
random.seed(42)


class TestUnitCircleActionSampler:
    """Test cases for the UnitCircleActionSampler class."""

    def test_unit_circle_action_sampler_initialization(self):
        """Test initialization of UnitCircleActionSampler.

        Purpose: Validates that UnitCircleActionSampler initializes with correct parameters

        Given: Different max_action_magnitude values
        When: UnitCircleActionSampler is instantiated
        Then: Object is created with correct attributes and default values

        Test type: unit
        """
        # Test default initialization
        sampler_default = UnitCircleActionSampler()
        assert sampler_default.max_action_magnitude == 1.0

        # Test custom initialization
        sampler_custom = UnitCircleActionSampler(max_action_magnitude=2.5)
        assert sampler_custom.max_action_magnitude == 2.5

        # Test zero magnitude
        sampler_zero = UnitCircleActionSampler(max_action_magnitude=0.0)
        assert sampler_zero.max_action_magnitude == 0.0

        # Test very large magnitude
        sampler_large = UnitCircleActionSampler(max_action_magnitude=100.0)
        assert sampler_large.max_action_magnitude == 100.0

    def test_unit_circle_action_sampler_basic_sampling(self):
        """Test basic action sampling functionality.

        Purpose: Validates that action sampling produces 2D vectors within magnitude constraints

        Given: UnitCircleActionSampler with specific max_action_magnitude
        When: Multiple actions are sampled
        Then: All actions are 2D vectors with magnitudes within the specified limit

        Test type: unit
        """
        max_magnitude = 1.5
        sampler = UnitCircleActionSampler(max_action_magnitude=max_magnitude)

        # Sample multiple actions
        num_samples = 100
        actions = [sampler.sample() for _ in range(num_samples)]

        for action in actions:
            # Check that action is a 2D numpy array
            assert isinstance(action, np.ndarray)
            assert action.shape == (2,)
            assert action.dtype == np.float64

            # Check magnitude constraint
            magnitude = np.linalg.norm(action)
            assert magnitude <= max_magnitude + 1e-10  # Small tolerance for floating point

            # Check that we get finite values
            assert np.all(np.isfinite(action))

    def test_unit_circle_action_sampler_magnitude_distribution(self):
        """Test that magnitudes follow correct distribution within the circle.

        Purpose: Validates that action magnitudes are properly distributed within the circle

        Given: UnitCircleActionSampler with unit magnitude
        When: Large number of actions are sampled
        Then: Magnitude distribution follows uniform area distribution (not biased toward center)

        Test type: unit
        """
        sampler = UnitCircleActionSampler(max_action_magnitude=1.0)

        # Sample large number of actions for statistical analysis
        num_samples = 10000
        actions = np.array([sampler.sample() for _ in range(num_samples)])

        # Calculate magnitudes
        magnitudes = np.linalg.norm(actions, axis=1)

        # All magnitudes should be within unit circle
        assert np.all(magnitudes <= 1.0 + 1e-10)

        # For uniform distribution in a circle, the CDF of radius should be r²
        # This means the mean magnitude should be approximately 2/3 for unit circle
        mean_magnitude = np.mean(magnitudes)
        expected_mean = 2.0 / 3.0  # Theoretical mean for uniform distribution in unit circle

        # Allow some statistical variation
        assert abs(mean_magnitude - expected_mean) < 0.05

    def test_unit_circle_action_sampler_angle_distribution(self):
        """Test that angles are uniformly distributed.

        Purpose: Validates that action angles are uniformly distributed around the circle

        Given: UnitCircleActionSampler
        When: Large number of actions are sampled
        Then: Angles are uniformly distributed between 0 and 2π

        Test type: unit
        """
        sampler = UnitCircleActionSampler(max_action_magnitude=1.0)

        # Sample actions
        num_samples = 5000
        actions = np.array([sampler.sample() for _ in range(num_samples)])

        # Calculate angles
        angles = np.arctan2(actions[:, 1], actions[:, 0])
        # Convert to [0, 2π] range
        angles = (angles + 2 * np.pi) % (2 * np.pi)

        # Test uniform distribution by checking distribution across quadrants
        quadrant_1 = np.sum((angles >= 0) & (angles < np.pi / 2))
        quadrant_2 = np.sum((angles >= np.pi / 2) & (angles < np.pi))
        quadrant_3 = np.sum((angles >= np.pi) & (angles < 3 * np.pi / 2))
        quadrant_4 = np.sum((angles >= 3 * np.pi / 2) & (angles < 2 * np.pi))

        # Each quadrant should have roughly equal representation (within statistical variation)
        expected_per_quadrant = num_samples / 4
        tolerance = expected_per_quadrant * 0.15  # 15% tolerance

        assert abs(quadrant_1 - expected_per_quadrant) < tolerance
        assert abs(quadrant_2 - expected_per_quadrant) < tolerance
        assert abs(quadrant_3 - expected_per_quadrant) < tolerance
        assert abs(quadrant_4 - expected_per_quadrant) < tolerance

    def test_unit_circle_action_sampler_different_magnitudes(self):
        """Test sampling with different maximum magnitudes.

        Purpose: Validates that different max_action_magnitude values scale results correctly

        Given: Multiple UnitCircleActionSamplers with different max magnitudes
        When: Actions are sampled from each
        Then: Action magnitudes scale proportionally with max_action_magnitude

        Test type: unit
        """
        magnitudes = [0.5, 1.0, 2.0, 5.0]
        num_samples = 200

        for max_mag in magnitudes:
            sampler = UnitCircleActionSampler(max_action_magnitude=max_mag)
            actions = [sampler.sample() for _ in range(num_samples)]

            # Check all actions respect the magnitude constraint
            for action in actions:
                magnitude = np.linalg.norm(action)
                assert magnitude <= max_mag + 1e-10

            # Check that we actually use the full range (some actions should be near max)
            max_observed = max(float(np.linalg.norm(action)) for action in actions)
            assert max_observed > max_mag * 0.8  # At least 80% of max magnitude observed

    def test_unit_circle_action_sampler_belief_node_parameter(self):
        """Test that belief_node parameter is handled correctly.

        Purpose: Validates that optional belief_node parameter doesn't affect sampling

        Given: UnitCircleActionSampler and real belief node
        When: Sampler is called with and without belief_node parameter
        Then: Both calls produce valid actions of same distribution

        Test type: unit
        """
        sampler = UnitCircleActionSampler(max_action_magnitude=1.0)

        # Create real belief node with WeightedParticleBelief
        particles = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        log_weights = np.array([-0.1, -0.2])
        belief = WeightedParticleBelief(
            particles=particles, log_weights=log_weights, resampling=False
        )
        belief_node = BeliefNode(belief=belief, observation=np.array([0.0, 0.0]))

        # Sample without belief node
        action1 = sampler.sample()

        # Sample with belief node
        action2 = sampler.sample(belief_node=belief_node)

        # Both should be valid actions
        assert isinstance(action1, np.ndarray) and action1.shape == (2,)
        assert isinstance(action2, np.ndarray) and action2.shape == (2,)
        assert np.linalg.norm(action1) <= 1.0 + 1e-10
        assert np.linalg.norm(action2) <= 1.0 + 1e-10

    def test_unit_circle_action_sampler_zero_magnitude(self):
        """Test edge case of zero magnitude sampling.

        Purpose: Validates behavior when max_action_magnitude is zero

        Given: UnitCircleActionSampler with zero max_action_magnitude
        When: Actions are sampled
        Then: All actions should be zero vectors

        Test type: unit
        """
        sampler = UnitCircleActionSampler(max_action_magnitude=0.0)

        # Sample multiple actions
        actions = [sampler.sample() for _ in range(10)]

        for action in actions:
            assert isinstance(action, np.ndarray)
            assert action.shape == (2,)
            assert np.allclose(action, [0.0, 0.0])
            assert np.linalg.norm(action) == 0.0

    def test_unit_circle_action_sampler_repeatability(self):
        """Test action sampling reproducibility with fixed seed.

        Purpose: Validates that sampling is reproducible when random seed is fixed

        Given: Fixed numpy random seed and UnitCircleActionSampler
        When: Actions are sampled multiple times with same seed
        Then: Identical sequences of actions are produced

        Test type: unit
        """
        sampler = UnitCircleActionSampler(max_action_magnitude=1.0)

        # First sequence
        np.random.seed(42)
        actions1 = [sampler.sample() for _ in range(10)]

        # Second sequence with same seed
        np.random.seed(42)
        actions2 = [sampler.sample() for _ in range(10)]

        # Should be identical
        for a1, a2 in zip(actions1, actions2):
            assert np.allclose(a1, a2)

    def test_unit_circle_action_sampler_serialization_pickle(self):
        """Test serialization and deserialization using pickle.

        Purpose: Validates that UnitCircleActionSampler can be serialized and deserialized with pickle

        Given: UnitCircleActionSampler instance with specific parameters
        When: Object is pickled and unpickled
        Then: Deserialized object has same parameters and produces equivalent actions

        Test type: unit
        """
        # Create original sampler
        original_sampler = UnitCircleActionSampler(max_action_magnitude=2.5)

        # Serialize with pickle
        pickled_data = pickle.dumps(original_sampler)

        # Deserialize
        deserialized_sampler = pickle.loads(pickled_data)

        # Check that attributes are preserved
        assert deserialized_sampler.max_action_magnitude == original_sampler.max_action_magnitude
        assert type(deserialized_sampler) == type(original_sampler)

        # Check that both samplers produce valid actions
        np.random.seed(123)
        original_action = original_sampler.sample()

        np.random.seed(123)
        deserialized_action = deserialized_sampler.sample()

        # With same seed, should produce identical results
        assert np.allclose(original_action, deserialized_action)

        # Both actions should respect magnitude constraint
        assert np.linalg.norm(original_action) <= 2.5 + 1e-10
        assert np.linalg.norm(deserialized_action) <= 2.5 + 1e-10

    def test_unit_circle_action_sampler_serialization_file(self):
        """Test serialization to file and loading.

        Purpose: Validates that UnitCircleActionSampler can be saved to and loaded from files

        Given: UnitCircleActionSampler instance and temporary file
        When: Object is saved to file and loaded back
        Then: Loaded object functions identically to original

        Test type: unit
        """
        # Create original sampler
        original_sampler = UnitCircleActionSampler(max_action_magnitude=3.0)

        # Use temporary file for serialization test
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
            temp_path = Path(temp_file.name)

            # Save to file
            with open(temp_path, "wb") as f:
                pickle.dump(original_sampler, f)

            # Load from file
            with open(temp_path, "rb") as f:
                loaded_sampler = pickle.load(f)

            # Clean up
            temp_path.unlink()

        # Verify loaded sampler
        assert loaded_sampler.max_action_magnitude == 3.0
        assert isinstance(loaded_sampler, UnitCircleActionSampler)

        # Test functionality
        action = loaded_sampler.sample()
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        assert np.linalg.norm(action) <= 3.0 + 1e-10

    def test_unit_circle_action_sampler_integration_with_pft_dpw(self):
        """Test integration with PFT-DPW planner.

        Purpose: Validates that UnitCircleActionSampler integrates correctly with PFT-DPW planner

        Given: ContinuousLightDarkPOMDP environment and PFT-DPW with UnitCircleActionSampler
        When: Planner uses the action sampler during planning
        Then: Planner executes successfully and produces valid actions

        Test type: integration
        """
        # Create environment
        env = ContinuousLightDarkPOMDP(
            discount_factor=0.99,
            goal_state=np.array([5, 0]),
            start_state=np.array([0, 0]),
            name="TestContinuousLightDark",
        )

        # Create action sampler
        action_sampler = UnitCircleActionSampler(max_action_magnitude=1.0)

        # Create PFT-DPW planner with action sampler
        planner = PFT_DPW(
            environment=env,
            discount_factor=0.99,
            depth=3,  # Shallow depth for testing
            name="TestPFT_DPW_ActionSampler",
            action_sampler=action_sampler,
            n_simulations=5,  # Few simulations for testing
            exploration_constant=1.0,
            k_a=1.0,
            k_o=1.0,
            alpha_a=0.5,
            alpha_o=0.5,
        )

        # Create initial belief
        initial_belief = get_initial_belief(env, n_particles=10)

        # Test that planner can use the action sampler
        action, run_data = planner.action(initial_belief)

        # Verify action is valid
        # PFT-DPW returns actions as a list
        assert isinstance(action, list)
        assert len(action) == 1  # Single action
        assert isinstance(action[0], np.ndarray)
        assert action[0].shape == (2,)  # 2D action
        assert np.linalg.norm(action[0]) <= 1.0 + 1e-10  # Within magnitude constraint

        # Verify run data exists
        assert run_data is not None

    def test_unit_circle_action_sampler_statistical_properties(self):
        """Test statistical properties of the action sampling distribution.

        Purpose: Validates statistical properties of sampled actions for quality assurance

        Given: UnitCircleActionSampler with unit magnitude
        When: Large number of actions are sampled
        Then: Statistical properties match theoretical expectations for uniform circle sampling

        Test type: unit
        """
        sampler = UnitCircleActionSampler(max_action_magnitude=1.0)

        # Sample large number of actions
        num_samples = 20000
        actions = np.array([sampler.sample() for _ in range(num_samples)])

        # Test mean (should be close to [0, 0] for symmetric distribution)
        mean_action = np.mean(actions, axis=0)
        assert abs(mean_action[0]) < 0.05  # Should be close to 0
        assert abs(mean_action[1]) < 0.05  # Should be close to 0

        # Test that we cover the full circle (check all quadrants)
        positive_x = np.sum(actions[:, 0] > 0.1)
        negative_x = np.sum(actions[:, 0] < -0.1)
        positive_y = np.sum(actions[:, 1] > 0.1)
        negative_y = np.sum(actions[:, 1] < -0.1)

        # Each should have reasonable representation
        min_expected = num_samples * 0.2  # At least 20% in each direction
        assert positive_x > min_expected
        assert negative_x > min_expected
        assert positive_y > min_expected
        assert negative_y > min_expected

    def test_unit_circle_action_sampler_parameter_validation(self):
        """Test parameter validation and edge cases.

        Purpose: Validates that UnitCircleActionSampler handles edge cases and invalid parameters appropriately

        Given: Various parameter values including edge cases
        When: UnitCircleActionSampler is initialized and used
        Then: Appropriate behavior for all parameter ranges

        Test type: unit
        """
        # Test negative magnitude (should work, just interpreted as absolute value)
        sampler_neg = UnitCircleActionSampler(max_action_magnitude=-1.0)
        action = sampler_neg.sample()
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        # The implementation uses the magnitude directly, so negative values may behave unexpectedly
        # but shouldn't crash

        # Test very small positive magnitude
        sampler_small = UnitCircleActionSampler(max_action_magnitude=1e-10)
        action_small = sampler_small.sample()
        assert isinstance(action_small, np.ndarray)
        assert action_small.shape == (2,)
        assert np.linalg.norm(action_small) <= 1e-10 + 1e-15

    def test_unit_circle_action_sampler_performance_characteristics(self):
        """Test performance characteristics of action sampling.

        Purpose: Validates that action sampling performs efficiently for practical use

        Given: UnitCircleActionSampler
        When: Large number of actions are sampled rapidly
        Then: Sampling completes efficiently without memory issues

        Test type: unit
        """
        sampler = UnitCircleActionSampler(max_action_magnitude=1.0)

        # Test that we can sample many actions quickly without issues
        num_samples = 10000

        # This should complete quickly and without memory issues
        actions = [sampler.sample() for _ in range(num_samples)]

        # Verify we got the expected number of valid actions
        assert len(actions) == num_samples

        # Spot check some actions
        for i in [0, num_samples // 2, num_samples - 1]:
            action = actions[i]
            assert isinstance(action, np.ndarray)
            assert action.shape == (2,)
            assert np.linalg.norm(action) <= 1.0 + 1e-10


class TestUsageExamples:
    """Test cases for usage examples from docstrings."""

    def test_docstring_basic_usage_example(self):
        """Test the basic usage example from the docstring.

        Purpose: Validates that the basic usage example from UnitCircleActionSampler docstring works correctly

        Given: Example code from docstring with PFT-DPW integration
        When: Example code is executed with reduced parameters for testing
        Then: Example runs successfully and produces expected results

        Test type: example
        """
        # Create 2D navigation environment (reduced size for testing)
        nav_env = ContinuousLightDarkPOMDP(
            discount_factor=0.99,
            goal_state=np.array([10, 0]),
            start_state=np.array([0, 0]),
        )

        # Create unit circle action sampler
        action_sampler = UnitCircleActionSampler(max_action_magnitude=2.0)

        # Test action sampling
        actions = [action_sampler.sample() for _ in range(5)]
        for i, action in enumerate(actions):
            magnitude = np.linalg.norm(action)
            assert isinstance(action, np.ndarray)
            assert action.shape == (2,)
            assert magnitude <= 2.0 + 1e-10
            # Verify the action has reasonable values
            assert np.all(np.isfinite(action))

        # Use with PFT-DPW planner (reduced parameters for testing)
        planner = PFT_DPW(
            environment=nav_env,
            discount_factor=0.99,
            depth=5,  # Reduced from 10
            name="PFT_DPW_Navigation",
            action_sampler=action_sampler,
            n_simulations=10,  # Reduced from 100
        )

        initial_belief = get_initial_belief(nav_env, n_particles=50)  # Reduced from 200
        action, run_data = planner.action(initial_belief)

        # Verify the selected action
        assert isinstance(action, list)
        assert len(action) == 1
        assert isinstance(action[0], np.ndarray)
        assert action[0].shape == (2,)

    def test_docstring_magnitude_comparison_example(self):
        """Test the magnitude comparison example from the docstring.

        Purpose: Validates that the magnitude comparison example demonstrates different movement strategies

        Given: Conservative and aggressive action samplers with different magnitudes
        When: Actions are sampled from both
        Then: Magnitude distributions show expected differences between strategies

        Test type: example
        """
        # Conservative movement sampler
        conservative_sampler = UnitCircleActionSampler(max_action_magnitude=0.5)

        # Aggressive movement sampler
        aggressive_sampler = UnitCircleActionSampler(max_action_magnitude=2.0)

        # Compare action distributions
        conservative_actions = [conservative_sampler.sample() for _ in range(100)]
        aggressive_actions = [aggressive_sampler.sample() for _ in range(100)]

        conservative_magnitudes = [np.linalg.norm(a) for a in conservative_actions]
        aggressive_magnitudes = [np.linalg.norm(a) for a in aggressive_actions]

        # Validate that conservative actions are indeed smaller
        max_conservative = max(float(mag) for mag in conservative_magnitudes)
        max_aggressive = max(float(mag) for mag in aggressive_magnitudes)
        avg_conservative = np.mean(conservative_magnitudes)
        avg_aggressive = np.mean(aggressive_magnitudes)

        # Conservative should be smaller
        assert max_conservative <= 0.5 + 1e-10
        assert max_aggressive <= 2.0 + 1e-10
        assert avg_conservative < avg_aggressive
        assert max_conservative < max_aggressive

        # Both should have valid ranges
        assert max_conservative > 0.3  # Should use most of the range
        assert max_aggressive > 1.5  # Should use most of the range

    def test_docstring_visualization_example_data_generation(self):
        """Test the data generation part of the visualization example.

        Purpose: Validates that the visualization example generates correct data

        Given: UnitCircleActionSampler with unit magnitude
        When: Actions are sampled for visualization
        Then: Generated data has correct properties for plotting

        Test type: example
        """
        sampler = UnitCircleActionSampler(max_action_magnitude=1.0)
        actions = np.array([sampler.sample() for _ in range(1000)])

        # Verify data structure for plotting
        assert actions.shape == (1000, 2)
        assert np.all(np.isfinite(actions))

        # Verify all points are within unit circle
        magnitudes = np.linalg.norm(actions, axis=1)
        assert np.all(magnitudes <= 1.0 + 1e-10)

        # Verify good distribution (should cover most of the circle)
        x_range = np.max(actions[:, 0]) - np.min(actions[:, 0])
        y_range = np.max(actions[:, 1]) - np.min(actions[:, 1])

        # Should span most of the diameter in both dimensions
        assert x_range > 1.5  # Should span more than 75% of diameter
        assert y_range > 1.5  # Should span more than 75% of diameter


class TestDiscreteActionSampler:
    """Test cases for the DiscreteActionSampler class."""

    def test_discrete_action_sampler_initialization(self):
        """Test initialization of DiscreteActionSampler.

        Purpose: Validates that DiscreteActionSampler initializes with correct parameters

        Given: Different action lists
        When: DiscreteActionSampler is instantiated
        Then: Object is created with correct attributes

        Test type: unit
        """
        # Test with string actions
        string_actions = ["up", "down", "left", "right"]
        sampler_strings = DiscreteActionSampler(actions=string_actions)
        assert sampler_strings.actions == string_actions

        # Test with integer actions
        int_actions = [0, 1, 2, 3, 4]
        sampler_ints = DiscreteActionSampler(actions=int_actions)
        assert sampler_ints.actions == int_actions

        # Test with mixed type actions
        mixed_actions = [0, "up", 1.5, (1, 2)]
        sampler_mixed = DiscreteActionSampler(actions=mixed_actions)
        assert sampler_mixed.actions == mixed_actions

        # Test with single action
        single_action = ["only_action"]
        sampler_single = DiscreteActionSampler(actions=single_action)
        assert sampler_single.actions == single_action

        # Test with empty list (edge case)
        empty_actions = []
        sampler_empty = DiscreteActionSampler(actions=empty_actions)
        assert sampler_empty.actions == []

    def test_discrete_action_sampler_basic_sampling(self):
        """Test basic action sampling functionality.

        Purpose: Validates that action sampling produces actions from the provided list

        Given: DiscreteActionSampler with specific actions
        When: Multiple actions are sampled
        Then: All sampled actions are from the original action list

        Test type: unit
        """
        actions = ["north", "south", "east", "west"]
        sampler = DiscreteActionSampler(actions=actions)

        # Sample multiple actions
        num_samples = 100
        sampled_actions = [sampler.sample() for _ in range(num_samples)]

        for action in sampled_actions:
            # Check that action is from the original list
            assert action in actions

        # Check that we get variety (not all the same action)
        unique_actions = set(sampled_actions)
        assert len(unique_actions) > 1  # Should have some variety

    def test_discrete_action_sampler_uniform_distribution(self):
        """Test that actions are sampled uniformly.

        Purpose: Validates that actions are sampled with roughly uniform distribution

        Given: DiscreteActionSampler with equal-probability actions
        When: Large number of actions are sampled
        Then: Each action appears roughly equally often

        Test type: unit
        """
        actions = ["A", "B", "C", "D"]
        sampler = DiscreteActionSampler(actions=actions)

        # Sample large number of actions for statistical analysis
        num_samples = 10000
        sampled_actions = [sampler.sample() for _ in range(num_samples)]

        # Count occurrences of each action
        action_counts = {}
        for action in sampled_actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        # Each action should appear roughly equally often
        expected_count = num_samples / len(actions)
        tolerance = expected_count * 0.1  # 10% tolerance

        for action in actions:
            count = action_counts.get(action, 0)
            assert abs(count - expected_count) < tolerance

    def test_discrete_action_sampler_belief_node_parameter(self):
        """Test that belief_node parameter is handled correctly.

        Purpose: Validates that optional belief_node parameter doesn't affect sampling

        Given: DiscreteActionSampler and real belief node
        When: Sampler is called with and without belief_node parameter
        Then: Both calls produce valid actions from the action list

        Test type: unit
        """
        actions = [1, 2, 3, 4, 5]
        sampler = DiscreteActionSampler(actions=actions)

        # Create real belief node
        particles = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        log_weights = np.array([-0.1, -0.2])
        belief = WeightedParticleBelief(
            particles=particles, log_weights=log_weights, resampling=False
        )
        belief_node = BeliefNode(belief=belief, observation=np.array([0.0, 0.0]))

        # Sample without belief node
        action1 = sampler.sample()

        # Sample with belief node
        action2 = sampler.sample(belief_node=belief_node)

        # Both should be valid actions from the list
        assert action1 in actions
        assert action2 in actions

    def test_discrete_action_sampler_single_action(self):
        """Test edge case of single action sampling.

        Purpose: Validates behavior when only one action is available

        Given: DiscreteActionSampler with single action
        When: Actions are sampled
        Then: All sampled actions should be the same

        Test type: unit
        """
        single_action = "only_choice"
        sampler = DiscreteActionSampler(actions=[single_action])

        # Sample multiple actions
        actions = [sampler.sample() for _ in range(10)]

        for action in actions:
            assert action == single_action

    def test_discrete_action_sampler_empty_actions(self):
        """Test edge case of empty action list.

        Purpose: Validates behavior when action list is empty

        Given: DiscreteActionSampler with empty action list
        When: Actions are sampled
        Then: Should raise appropriate error

        Test type: unit
        """
        sampler = DiscreteActionSampler(actions=[])

        # Sampling from empty list should raise IndexError
        with pytest.raises(IndexError):
            sampler.sample()

    def test_discrete_action_sampler_repeatability(self):
        """Test action sampling reproducibility with fixed seed.

        Purpose: Validates that sampling is reproducible when random seed is fixed

        Given: Fixed random seed and DiscreteActionSampler
        When: Actions are sampled multiple times with same seed
        Then: Identical sequences of actions are produced

        Test type: unit
        """
        actions = ["red", "green", "blue"]
        sampler = DiscreteActionSampler(actions=actions)

        # First sequence
        random.seed(42)
        actions1 = [sampler.sample() for _ in range(10)]

        # Second sequence with same seed
        random.seed(42)
        actions2 = [sampler.sample() for _ in range(10)]

        # Should be identical
        assert actions1 == actions2

    def test_discrete_action_sampler_serialization_pickle(self):
        """Test serialization and deserialization using pickle.

        Purpose: Validates that DiscreteActionSampler can be serialized and deserialized with pickle

        Given: DiscreteActionSampler instance with specific actions
        When: Object is pickled and unpickled
        Then: Deserialized object has same actions and produces equivalent sampling

        Test type: unit
        """
        # Create original sampler
        original_actions = ["action1", "action2", "action3"]
        original_sampler = DiscreteActionSampler(actions=original_actions)

        # Serialize with pickle
        pickled_data = pickle.dumps(original_sampler)

        # Deserialize
        deserialized_sampler = pickle.loads(pickled_data)

        # Check that attributes are preserved
        assert deserialized_sampler.actions == original_sampler.actions
        assert type(deserialized_sampler) == type(original_sampler)

        # Check that both samplers produce valid actions
        random.seed(123)
        original_action = original_sampler.sample()

        random.seed(123)
        deserialized_action = deserialized_sampler.sample()

        # With same seed, should produce identical results
        assert original_action == deserialized_action

        # Both actions should be from the original list
        assert original_action in original_actions
        assert deserialized_action in original_actions

    def test_discrete_action_sampler_serialization_file(self):
        """Test serialization to file and loading.

        Purpose: Validates that DiscreteActionSampler can be saved to and loaded from files

        Given: DiscreteActionSampler instance and temporary file
        When: Object is saved to file and loaded back
        Then: Loaded object functions identically to original

        Test type: unit
        """
        # Create original sampler
        original_actions = ["move", "stay", "turn_left", "turn_right"]
        original_sampler = DiscreteActionSampler(actions=original_actions)

        # Use temporary file for serialization test
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
            temp_path = Path(temp_file.name)

            # Save to file
            with open(temp_path, "wb") as f:
                pickle.dump(original_sampler, f)

            # Load from file
            with open(temp_path, "rb") as f:
                loaded_sampler = pickle.load(f)

            # Clean up
            temp_path.unlink()

        # Verify loaded sampler
        assert loaded_sampler.actions == original_actions
        assert isinstance(loaded_sampler, DiscreteActionSampler)

        # Test functionality
        action = loaded_sampler.sample()
        assert action in original_actions

    def test_discrete_action_sampler_complex_action_types(self):
        """Test sampling with complex action types.

        Purpose: Validates that DiscreteActionSampler works with various complex action types

        Given: DiscreteActionSampler with complex action objects
        When: Actions are sampled
        Then: Complex actions are sampled correctly

        Test type: unit
        """
        # Test with tuples
        tuple_actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        sampler_tuples = DiscreteActionSampler(actions=tuple_actions)

        sampled_tuple = sampler_tuples.sample()
        assert sampled_tuple in tuple_actions
        assert isinstance(sampled_tuple, tuple)

        # Test with numpy arrays
        array_actions = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0])]
        sampler_arrays = DiscreteActionSampler(actions=array_actions)

        sampled_array = sampler_arrays.sample()
        # Check that the sampled array matches one of the original arrays
        assert any(np.array_equal(sampled_array, arr) for arr in array_actions)
        assert isinstance(sampled_array, np.ndarray)

        # Test with dictionaries
        dict_actions = [{"x": 1, "y": 0}, {"x": 0, "y": 1}, {"x": -1, "y": 0}]
        sampler_dicts = DiscreteActionSampler(actions=dict_actions)

        sampled_dict = sampler_dicts.sample()
        assert sampled_dict in dict_actions
        assert isinstance(sampled_dict, dict)

    def test_discrete_action_sampler_duplicate_actions(self):
        """Test behavior with duplicate actions in the list.

        Purpose: Validates that duplicate actions are handled correctly

        Given: DiscreteActionSampler with duplicate actions
        When: Actions are sampled
        Then: Duplicate actions appear with higher probability

        Test type: unit
        """
        # List with duplicates
        actions_with_duplicates = ["A", "B", "A", "C", "A"]
        sampler = DiscreteActionSampler(actions=actions_with_duplicates)

        # Sample many actions
        num_samples = 10000
        sampled_actions = [sampler.sample() for _ in range(num_samples)]

        # Count occurrences
        action_counts = {}
        for action in sampled_actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        # "A" should appear more often since it's in the list 3 times
        assert action_counts["A"] > action_counts["B"]
        assert action_counts["A"] > action_counts["C"]

        # All sampled actions should be from the original list
        for action in sampled_actions:
            assert action in actions_with_duplicates

    def test_discrete_action_sampler_performance_characteristics(self):
        """Test performance characteristics of action sampling.

        Purpose: Validates that action sampling performs efficiently for practical use

        Given: DiscreteActionSampler with large action space
        When: Large number of actions are sampled rapidly
        Then: Sampling completes efficiently without memory issues

        Test type: unit
        """
        # Create sampler with many actions
        large_action_space = list(range(1000))  # 1000 different actions
        sampler = DiscreteActionSampler(actions=large_action_space)

        # Test that we can sample many actions quickly without issues
        num_samples = 10000

        # This should complete quickly and without memory issues
        actions = [sampler.sample() for _ in range(num_samples)]

        # Verify we got the expected number of valid actions
        assert len(actions) == num_samples

        # All actions should be from the original list
        for action in actions:
            assert action in large_action_space

        # Should have good variety (not all the same)
        unique_actions = set(actions)
        assert len(unique_actions) > 100  # Should sample many different actions

    def test_discrete_action_sampler_reduce_method(self):
        """Test the __reduce__ method for pickle serialization.

        Purpose: Validates that the __reduce__ method works correctly for serialization

        Given: DiscreteActionSampler instance
        When: Object is pickled using __reduce__ method
        Then: Deserialized object is identical to original

        Test type: unit
        """
        actions = ["test1", "test2", "test3"]
        sampler = DiscreteActionSampler(actions=actions)

        # Test __reduce__ method directly
        reduce_result = sampler.__reduce__()

        # Should return tuple with class, constructor arguments, and state
        assert isinstance(reduce_result, tuple)
        assert len(reduce_result) == 3
        assert reduce_result[0] == DiscreteActionSampler
        assert reduce_result[1] == ()  # Empty constructor args
        assert reduce_result[2] == {"actions": actions}  # State dict

        # Test reconstruction
        reconstructed_sampler = reduce_result[0](actions=[])  # Create with empty actions
        reconstructed_sampler.__setstate__(reduce_result[2])
        assert reconstructed_sampler.actions == actions
        assert isinstance(reconstructed_sampler, DiscreteActionSampler)

        # Test that reconstructed sampler works
        action = reconstructed_sampler.sample()
        assert action in actions

    def test_discrete_action_sampler_immutability(self):
        """Test that the action list is properly stored as immutable.

        Purpose: Validates that changes to the original action list don't affect the sampler

        Given: DiscreteActionSampler and original action list
        When: Original action list is modified
        Then: Sampler continues to use the original list

        Test type: unit
        """
        original_actions = ["A", "B", "C"]
        sampler = DiscreteActionSampler(actions=original_actions)

        # Modify the original list
        original_actions.append("D")
        original_actions.remove("A")

        # Sampler should still use the original list (before modification)
        sampled_actions = [sampler.sample() for _ in range(100)]

        # All sampled actions should be from the original list (A, B, C)
        for action in sampled_actions:
            assert action in ["A", "B", "C"]
            assert action not in ["D"]  # Should not sample the added action

    def test_discrete_action_sampler_none_action(self):
        """Test behavior with None as an action.

        Purpose: Validates that None can be used as a valid action

        Given: DiscreteActionSampler with None as one of the actions
        When: Actions are sampled
        Then: None can be sampled as a valid action

        Test type: unit
        """
        actions = ["action1", None, "action3"]
        sampler = DiscreteActionSampler(actions=actions)

        # Sample multiple actions
        sampled_actions = [sampler.sample() for _ in range(100)]

        # Should be able to sample None
        assert None in sampled_actions

        # All sampled actions should be from the original list
        for action in sampled_actions:
            assert action in actions
