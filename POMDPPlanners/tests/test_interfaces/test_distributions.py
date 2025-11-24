"""Interface compliance tests for Distribution implementations.

This module validates that all distribution classes correctly implement
the Distribution interface, particularly the probability() method,
by comparing computed probabilities against empirical sampling distributions.

This is part of the interface compliance test suite that ensures all
implementations satisfy their respective contracts.
"""

import numpy as np
import pytest

from POMDPPlanners.core.distributions import DiscreteDistribution, Numpy2DDistribution
from POMDPPlanners.tests.test_utils.test_probability_utils import (
    validate_distribution_probability_matches_empirical,
)


class TestDiscreteDistributionProbability:
    """Test probability method for DiscreteDistribution."""

    def test_discrete_distribution_uniform_probability(self):
        """Test DiscreteDistribution probability method with uniform distribution.

        Purpose: Validates that probability() method matches empirical distribution for uniform probabilities

        Given: DiscreteDistribution with uniform probabilities over string values
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        values = ["a", "b", "c", "d"]
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        dist = DiscreteDistribution(values, probs)

        results = validate_distribution_probability_matches_empirical(
            dist, num_samples=1000, max_js_divergence=0.05
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.05
        assert results["num_unique_values"] == 4  # All four values should appear

    def test_discrete_distribution_skewed_probability(self):
        """Test DiscreteDistribution probability method with skewed distribution.

        Purpose: Validates that probability() method matches empirical distribution for skewed probabilities

        Given: DiscreteDistribution with highly skewed probabilities
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        values = ["a", "b", "c", "d", "e"]
        probs = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
        dist = DiscreteDistribution(values, probs)

        results = validate_distribution_probability_matches_empirical(
            dist, num_samples=1000, max_js_divergence=0.05
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.05

    def test_discrete_distribution_numeric_values_probability(self):
        """Test DiscreteDistribution probability method with numeric values.

        Purpose: Validates that probability() method works correctly with numeric values

        Given: DiscreteDistribution with integer values
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        values = [1, 2, 3, 4, 5]
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        dist = DiscreteDistribution(values, probs)

        results = validate_distribution_probability_matches_empirical(
            dist, num_samples=1000, max_js_divergence=0.05
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.05
        assert results["num_unique_values"] == 5  # All five values should appear

    def test_discrete_distribution_single_value_probability(self):
        """Test DiscreteDistribution probability method with single value (deterministic).

        Purpose: Validates that probability() method works correctly for deterministic distributions

        Given: DiscreteDistribution with single value (probability 1.0)
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance (should be nearly perfect)

        Test type: unit
        """
        values = ["deterministic"]
        probs = np.array([1.0])
        dist = DiscreteDistribution(values, probs)

        results = validate_distribution_probability_matches_empirical(
            dist, num_samples=1000, max_js_divergence=0.01
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.01  # Should be nearly perfect (deterministic)
        assert results["num_unique_values"] == 1  # Only one possible value

    def test_discrete_distribution_large_support_probability(self):
        """Test DiscreteDistribution probability method with large support set.

        Purpose: Validates that probability() method works correctly with many possible values

        Given: DiscreteDistribution with 20 possible values
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        values = list(range(20))
        probs = np.ones(20) / 20.0  # Uniform distribution
        dist = DiscreteDistribution(values, probs)

        results = validate_distribution_probability_matches_empirical(
            dist, num_samples=2000, max_js_divergence=0.05
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.05


class TestNumpy2DDistributionProbability:
    """Test probability method for Numpy2DDistribution."""

    def test_numpy2d_distribution_uniform_probability(self):
        """Test Numpy2DDistribution probability method with uniform distribution.

        Purpose: Validates that probability() method matches empirical distribution for uniform probabilities

        Given: Numpy2DDistribution with uniform probabilities over 2D numpy array values
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        # Create 2D array with shape (2, 4) - 4 possible 2D vectors
        values = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        dist = Numpy2DDistribution(values, probs)

        results = validate_distribution_probability_matches_empirical(
            dist, num_samples=1000, max_js_divergence=0.05
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.05
        assert results["num_unique_values"] == 4  # All four vectors should appear

    def test_numpy2d_distribution_skewed_probability(self):
        """Test Numpy2DDistribution probability method with skewed distribution.

        Purpose: Validates that probability() method matches empirical distribution for skewed probabilities

        Given: Numpy2DDistribution with highly skewed probabilities
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        # Create 2D array with shape (2, 5) - 5 possible 2D vectors
        values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
        probs = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
        dist = Numpy2DDistribution(values, probs)

        results = validate_distribution_probability_matches_empirical(
            dist, num_samples=1000, max_js_divergence=0.05
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.05

    def test_numpy2d_distribution_single_value_probability(self):
        """Test Numpy2DDistribution probability method with single value (deterministic).

        Purpose: Validates that probability() method works correctly for deterministic distributions

        Given: Numpy2DDistribution with single vector (probability 1.0)
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance (should be nearly perfect)

        Test type: unit
        """
        # Create 2D array with shape (2, 1) - single 2D vector
        values = np.array([[1.0], [2.0]])
        probs = np.array([1.0])
        dist = Numpy2DDistribution(values, probs)

        results = validate_distribution_probability_matches_empirical(
            dist, num_samples=1000, max_js_divergence=0.01
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.01  # Should be nearly perfect (deterministic)
        assert results["num_unique_values"] == 1  # Only one possible value

    def test_numpy2d_distribution_large_support_probability(self):
        """Test Numpy2DDistribution probability method with large support set.

        Purpose: Validates that probability() method works correctly with many possible vectors

        Given: Numpy2DDistribution with 20 possible 2D vectors
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        # Create 2D array with shape (2, 20) - 20 possible 2D vectors
        values = np.array([np.arange(20.0), np.arange(20.0) + 100.0])
        probs = np.ones(20) / 20.0  # Uniform distribution
        dist = Numpy2DDistribution(values, probs)

        results = validate_distribution_probability_matches_empirical(
            dist, num_samples=2000, max_js_divergence=0.05
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.05

    def test_numpy2d_distribution_negative_values_probability(self):
        """Test Numpy2DDistribution probability method with negative values.

        Purpose: Validates that probability() method works correctly with negative vector components

        Given: Numpy2DDistribution with vectors containing negative values
        When: Comparing computed probabilities to empirical sampling
        Then: Probabilities match within tolerance and are properly normalized

        Test type: unit
        """
        # Create 2D array with shape (2, 4) containing negative values
        values = np.array([[-1.0, -2.0, 1.0, 2.0], [-5.0, -6.0, 5.0, 6.0]])
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        dist = Numpy2DDistribution(values, probs)

        results = validate_distribution_probability_matches_empirical(
            dist, num_samples=1000, max_js_divergence=0.05
        )

        assert results["probabilities_normalized"]
        assert results["distance"] < 0.05
        assert results["num_unique_values"] == 4  # All four vectors should appear
