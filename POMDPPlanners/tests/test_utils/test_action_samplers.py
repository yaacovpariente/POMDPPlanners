"""Tests for action samplers utility functions.

This module tests the action sampler implementations used for continuous action
space sampling in POMDP planners, particularly for PFT-DPW integration.
"""

import numpy as np
import pytest
import pickle
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler
from POMDPPlanners.core.tree import BeliefNode
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.core.belief import get_initial_belief


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
            assert (
                magnitude <= max_magnitude + 1e-10
            )  # Small tolerance for floating point

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
        expected_mean = (
            2.0 / 3.0
        )  # Theoretical mean for uniform distribution in unit circle

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
            max_observed = max(np.linalg.norm(action) for action in actions)
            assert (
                max_observed > max_mag * 0.8
            )  # At least 80% of max magnitude observed

    def test_unit_circle_action_sampler_belief_node_parameter(self):
        """Test that belief_node parameter is handled correctly.

        Purpose: Validates that optional belief_node parameter doesn't affect sampling

        Given: UnitCircleActionSampler and mock belief node
        When: Sampler is called with and without belief_node parameter
        Then: Both calls produce valid actions of same distribution

        Test type: unit
        """
        sampler = UnitCircleActionSampler(max_action_magnitude=1.0)

        # Mock belief node
        mock_belief_node = Mock(spec=BeliefNode)

        # Sample without belief node
        action1 = sampler.sample()

        # Sample with belief node
        action2 = sampler.sample(belief_node=mock_belief_node)

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
        assert (
            deserialized_sampler.max_action_magnitude
            == original_sampler.max_action_magnitude
        )
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
        max_conservative = max(conservative_magnitudes)
        max_aggressive = max(aggressive_magnitudes)
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
