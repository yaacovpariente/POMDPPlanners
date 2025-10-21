import random

import numpy as np
import pytest

from POMDPPlanners.utils.statistics_utils import cvar_estimator, cvar_estimator_from_dist

np.random.seed(42)
random.seed(42)


def test_cvar_estimator_negative_values():
    """Test CVaR calculation with negative values.

    Purpose: Validates that cvar_estimator correctly handles negative values and returns expected tail statistics

    Given: Array with negative and positive values [-5.0, -3.0, -1.0, 0.0, 2.0] and alpha=0.8 (80th percentile)
    When: cvar_estimator processes the mixed-sign values array
    Then: Returns mean of highest 20% values (tail average) approximately -0.5

    Test type: unit
    """
    values = np.array([-5.0, -3.0, -1.0, 0.0, 2.0])
    alpha = 0.8
    result = cvar_estimator(values, alpha)
    # For alpha=0.8, we expect the CVaR to be the mean of the highest 20% values
    assert np.isclose(result, -0.5, rtol=1e-5)


def test_cvar_estimator_alpha_boundaries():
    """Test CVaR calculation at alpha boundaries.

    Purpose: Validates that cvar_estimator handles extreme alpha values (0.0 and 1.0) correctly returning min and max values

    Given: Ordered values array [1.0, 2.0, 3.0, 4.0, 5.0] with alpha boundary values 0.0 and 1.0
    When: cvar_estimator is called with alpha=1.0 (maximum) and alpha=0.0 (minimum)
    Then: Returns 5.0 for alpha=1.0 (maximum value) and 1.0 for alpha=0.0 (minimum value)

    Test type: unit
    """
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Test alpha = 1.0 (should return maximum value)
    result_alpha_1 = cvar_estimator(values, 1.0)
    assert np.isclose(result_alpha_1, 5.0)

    # Test alpha = 0.0 (should return minimum value)
    result_alpha_0 = cvar_estimator(values, 0.0)
    assert np.isclose(result_alpha_0, 1.0)


def test_cvar_estimator_identical_values():
    """Test CVaR calculation with identical values.

    Purpose: Validates that cvar_estimator handles constant arrays correctly without variance issues

    Given: Array of identical values [10.0, 10.0, 10.0, 10.0] and alpha=0.9
    When: cvar_estimator processes the constant array
    Then: Returns the constant value 10.0 regardless of alpha percentile

    Test type: unit
    """
    values = np.array([10.0, 10.0, 10.0, 10.0])
    alpha = 0.9
    result = cvar_estimator(values, alpha)
    assert np.isclose(result, 10.0)


def test_cvar_estimator_invalid_input():
    """Test CVaR estimator with invalid inputs.

    Purpose: Validates that cvar_estimator raises appropriate ValueError exceptions for invalid alpha values and empty arrays

    Given: Valid values array [1.0, 2.0, 3.0], invalid alpha values (-0.1, 1.1), and empty array
    When: cvar_estimator is called with out-of-range alpha or empty input
    Then: ValueError is raised for alpha < 0, alpha > 1, and empty array inputs

    Test type: unit
    """
    values = np.array([1.0, 2.0, 3.0])

    # Test invalid alpha values
    with pytest.raises(ValueError):
        cvar_estimator(values, -0.1)
    with pytest.raises(ValueError):
        cvar_estimator(values, 1.1)

    # Test empty array
    with pytest.raises(ValueError):
        cvar_estimator(np.array([]), 0.9)


def test_cvar_estimator_known_distribution():
    """Test CVaR calculation with a known distribution.

    Purpose: Validates that cvar_estimator produces theoretically correct results for uniform distribution U[0,1]

    Given: Uniform distribution samples from U[0,1] with 1000 points and alpha=0.9
    When: cvar_estimator calculates CVaR of the uniform distribution
    Then: Result approximates theoretical CVaR = (1 + (1-alpha))/2 = 0.55 within 1% tolerance

    Test type: unit
    """
    # Create a uniform distribution
    values = np.linspace(0, 1, 1000)
    alpha = 0.9

    # For uniform distribution U[0,1], CVaR of highest values at alpha is (1 + alpha)/2
    theoretical_cvar = (1 + (1 - alpha)) / 2
    result = cvar_estimator(values, alpha)

    assert np.isclose(result, theoretical_cvar, rtol=1e-2)


def test_cvar_estimator_mixed_cases():
    """Test CVaR calculation with mixed cases and different alphas.

    Purpose: Validates that cvar_estimator correctly handles various array patterns and alpha values with precise mathematical expectations

    Given: Multiple test arrays with known distributions and different alpha percentiles (0.2, 1.0, 0.15)
    When: cvar_estimator processes each case with specific alpha values
    Then: Returns mathematically correct tail averages (5.0 for single tail, 7.5 for mixed tail, weighted average for fractional cases)

    Test type: unit
    """
    vec2 = np.array([1.0, 1.0, 1.0, 1.0, 5.0])
    result = cvar_estimator(vec2, alpha=0.2)
    assert np.isclose(result, 5.0)

    vec3 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 10.0])

    result1 = cvar_estimator(vec3, alpha=0.2)
    assert np.isclose(result1, 7.5, atol=1e-4)

    result2 = cvar_estimator(vec3, alpha=1.0)
    assert np.isclose(result2, 10.0, atol=1e-4)

    result3 = cvar_estimator(vec3, alpha=0.15)
    assert np.isclose(result3, 5 / 3 + 10 * 2 / 3, atol=1e-4)


def test_cvar_estimator_single_element():
    """Test CVaR calculation with a single-element array.

    Purpose: Validates that cvar_estimator handles edge case of single-element arrays correctly

    Given: Single-element array [1.0] and alpha=0.5 (median)
    When: cvar_estimator processes the minimal array
    Then: Returns the single value 1.0 as the only possible CVaR result

    Test type: unit
    """
    vec4 = np.array([1.0])
    result = cvar_estimator(vec4, alpha=0.5)
    assert np.isclose(result, 1.0)


def test_cvar_estimator_from_dist():
    """Test CVaR calculation from discrete probability distribution.

    Purpose: Validates that cvar_estimator_from_dist correctly computes CVaR using weighted discrete distributions with various edge cases

    Given: Multiple discrete distributions with values and weights including single value, weighted distributions, duplicates, and edge cases
    When: cvar_estimator_from_dist processes each weighted distribution with alpha=0.2
    Then: Returns correct weighted tail averages (1.0 for single, 2.5 for weighted cases) and raises ValueError for invalid inputs

    Test type: unit
    """
    # Test single value case
    values = np.array([1.0])
    weights = np.array([1.0])
    alpha = 0.2
    cvar = cvar_estimator_from_dist(values, weights, alpha)
    assert np.isclose(cvar, 1.0, atol=1e-4)

    # Test three values with different weights
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.8, 0.1, 0.1])
    cvar = cvar_estimator_from_dist(values, weights, alpha)
    assert np.isclose(cvar, 2.5, atol=1e-4)

    # Test four values with duplicate values
    values = np.array([1.0, 2.0, 3.0, 3.0])
    weights = np.array([0.8, 0.1, 0.05, 0.05])
    cvar = cvar_estimator_from_dist(values, weights, alpha)
    assert np.isclose(cvar, 2.5, atol=1e-4)

    # Test edge case where P(q_alpha) != alpha
    alpha = 0.1
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.8, 0.15, 0.05])
    cvar = cvar_estimator_from_dist(values, weights, alpha)
    assert np.isclose(cvar, 2.5, atol=1e-4)

    # Test two values case
    values = np.array([1.0, 2.0])
    weights = np.array([0.8, 0.2])
    cvar = cvar_estimator_from_dist(values, weights, alpha)
    assert np.isclose(cvar, 2.0, atol=1e-4)

    # Test invalid inputs
    with pytest.raises(ValueError):
        cvar_estimator_from_dist(np.array([1.0]), np.array([0.5]), 0.2)  # weights don't sum to 1

    with pytest.raises(ValueError):
        cvar_estimator_from_dist(np.array([]), np.array([]), 0.2)  # empty arrays

    with pytest.raises(ValueError):
        cvar_estimator_from_dist(np.array([1.0]), np.array([1.0]), 1.1)  # invalid alpha
