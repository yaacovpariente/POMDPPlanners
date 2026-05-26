# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for distribution implementations.

This module tests the distribution implementations, focusing on:
- Basic distribution functionality
- Sampling operations
- Probability calculations
- Distribution types
"""

import random

import numpy as np
import pytest

from POMDPPlanners.core.distributions import DiscreteDistribution, Numpy2DDistribution

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


def test_discrete_distribution_initialization():
    """Test discrete distribution initialization.

    Purpose: Validates proper initialization of discrete distribution

    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes

    Test type: unit
    """
    # Test valid initialization
    values = [1, 2, 3]
    probs = np.array([0.3, 0.3, 0.4])
    dist = DiscreteDistribution(values, probs)
    assert dist.values == values
    assert np.array_equal(dist.probs, probs)

    # Test invalid initialization with mismatched lengths
    with pytest.raises(ValueError):
        DiscreteDistribution([1, 2], np.array([0.5, 0.5, 0.0]))


def test_discrete_distribution_sample():
    """Test discrete distribution sample.

    Purpose: Validates sampling behavior for discrete distribution

    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution

    Test type: unit
    """
    # Test sampling from a simple distribution
    values = [1, 2, 3]
    probs = np.array([0.0, 1.0, 0.0])  # Always returns 2
    dist = DiscreteDistribution(values, probs)

    # Sample multiple times and verify we always get 2
    samples = dist.sample(n_samples=100)
    assert all(s == 2 for s in samples)

    # Test sampling from a uniform distribution
    values = [1, 2, 3]
    probs = np.array([1 / 3, 1 / 3, 1 / 3])
    dist = DiscreteDistribution(values, probs)

    # Sample multiple times and verify we get all values
    samples = dist.sample(n_samples=1000)
    assert all(v in samples for v in values)


def test_discrete_distribution_probability():
    """Test that DiscreteDistribution correctly calculates probabilities for single and multiple values.

    Purpose: Validates that discrete distribution probability method returns correct probability values for known distribution

    Given: DiscreteDistribution with values [1, 2, 3] and probabilities [0.2, 0.5, 0.3]
    When: probability method is called with single values, multiple values, and values not in the distribution
    Then: Correct probabilities are returned (0.5 for value 2, 0.0 for values not in distribution)

    Test type: unit
    """
    # Test probability calculation
    values = [1, 2, 3]
    probs = np.array([0.2, 0.5, 0.3])
    dist = DiscreteDistribution(values, probs)

    # Test single value
    prob_array = dist.probability([2])
    assert np.isclose(prob_array[0], 0.5)

    # Test multiple values
    prob_array = dist.probability([1, 2, 3, 4])
    assert np.isclose(prob_array[0], 0.2)
    assert np.isclose(prob_array[1], 0.5)
    assert np.isclose(prob_array[2], 0.3)
    assert np.isclose(prob_array[3], 0.0)  # Value not in distribution


def test_numpy2d_distribution_initialization():
    """Test numpy2d distribution initialization.

    Purpose: Validates proper initialization of numpy2d distribution

    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes

    Test type: unit
    """
    # Test valid initialization
    values = np.array([[1, 2, 3], [4, 5, 6]])
    probs = np.array([0.3, 0.3, 0.4])
    dist = Numpy2DDistribution(values, probs)
    assert np.array_equal(dist.values, values)
    assert np.array_equal(dist.probs, probs)

    # Test invalid initialization with wrong shape
    with pytest.raises(ValueError):
        Numpy2DDistribution(np.array([[1, 2], [3, 4], [5, 6]]), np.array([0.5, 0.5]))

    # Test invalid initialization with mismatched lengths
    with pytest.raises(ValueError):
        Numpy2DDistribution(np.array([[1, 2], [3, 4]]), np.array([0.5, 0.5, 0.0]))


def test_numpy2d_distribution_sample():
    """Test numpy2d distribution sample.

    Purpose: Validates sampling behavior for numpy2d distribution

    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution

    Test type: unit
    """
    # Test sampling from a simple distribution
    values = np.array([[1, 2, 3], [4, 5, 6]])
    probs = np.array([0.0, 1.0, 0.0])  # Always returns [2, 5]
    dist = Numpy2DDistribution(values, probs)

    # Sample multiple times and verify we always get [2, 5]
    samples = dist.sample(n_samples=100)
    assert all(np.array_equal(s, np.array([2, 5])) for s in samples)

    # Test sampling from a uniform distribution
    values = np.array([[1, 2, 3], [4, 5, 6]])
    probs = np.array([1 / 3, 1 / 3, 1 / 3])
    dist = Numpy2DDistribution(values, probs)

    # Sample multiple times and verify we get all possible values
    samples = dist.sample(n_samples=1000)
    possible_values = [values[:, i] for i in range(values.shape[1])]
    assert all(any(np.array_equal(s, v) for v in possible_values) for s in samples)


def test_numpy2d_distribution_probability():
    """Test that Numpy2DDistribution correctly calculates probabilities for multi-dimensional arrays.

    Purpose: Validates that numpy2d distribution probability method returns correct probability values for 2D array distribution

    Given: Numpy2DDistribution with 2D values [[1,2,3],[4,5,6]] and probabilities [0.2, 0.5, 0.3]
    When: probability method is called with single arrays, multiple arrays, and arrays not in the distribution
    Then: Correct probabilities are returned (0.5 for [2,5], 0.0 for arrays not in distribution)

    Test type: unit
    """
    # Test probability calculation
    values = np.array([[1, 2, 3], [4, 5, 6]])
    probs = np.array([0.2, 0.5, 0.3])
    dist = Numpy2DDistribution(values, probs)

    # Test single value
    prob_array = dist.probability([np.array([2, 5])])
    assert np.isclose(prob_array[0], 0.5)

    # Test multiple values
    prob_array = dist.probability(
        [np.array([1, 4]), np.array([2, 5]), np.array([3, 6]), np.array([0, 0])]
    )
    assert np.isclose(prob_array[0], 0.2)
    assert np.isclose(prob_array[1], 0.5)
    assert np.isclose(prob_array[2], 0.3)
    assert np.isclose(prob_array[3], 0.0)  # Value not in distribution
