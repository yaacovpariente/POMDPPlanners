import random
from typing import List, Any

import numpy as np
import pytest

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.utils.statistics_utils import (
    aggregate_weights_for_duplicate_values,
    cvar_estimator,
    cvar_estimator_from_dist,
    tv_distance,
    tv_distance_grid,
    tv_distance_averaged,
    tv_distance_mixture_sampling,
    tv_distance_monte_carlo,
)

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

    # Test case with identical values and uniform weights (real-world use case)
    values = np.array([7.07106781, 7.07106781, 7.07106781, 7.07106781, 7.07106781])
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    alpha = 0.1
    cvar = cvar_estimator_from_dist(values, weights, alpha)
    assert np.isclose(cvar, 7.07106781, atol=1e-4)

    # Test alpha boundary cases
    # alpha = 1.0 should return weighted average of all values
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.33, 0.33, 0.34])
    cvar = cvar_estimator_from_dist(values, weights, 1.0)
    expected_mean = np.sum(values * weights)
    assert np.isclose(cvar, expected_mean, atol=1e-4)

    # alpha = 0.0 currently returns NaN (division by zero) - test this behavior
    cvar = cvar_estimator_from_dist(values, weights, 0.0)
    assert np.isnan(cvar)

    # Test negative values
    values = np.array([-5.0, -1.0, 3.0])
    weights = np.array([0.33, 0.33, 0.34])
    cvar = cvar_estimator_from_dist(values, weights, 0.2)
    assert isinstance(cvar, (float, np.floating))
    assert np.isclose(cvar, 3.0, atol=1e-4)

    # Test zero weights (but still sum to 1)
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.0, 0.5, 0.5])
    cvar = cvar_estimator_from_dist(values, weights, 0.2)
    assert isinstance(cvar, (float, np.floating))
    assert np.isclose(cvar, 3.0, atol=1e-4)  # Should be average of values with non-zero weights

    # Test very small alpha
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.33, 0.33, 0.34])
    cvar = cvar_estimator_from_dist(values, weights, 0.001)
    assert isinstance(cvar, (float, np.floating))
    assert not np.isnan(cvar)
    assert np.isclose(cvar, 3.0, atol=1e-4)

    # Test alpha very close to 1
    cvar = cvar_estimator_from_dist(values, weights, 0.999999999999)
    assert isinstance(cvar, (float, np.floating))
    assert not np.isnan(cvar)
    assert np.isclose(cvar, np.sum(values * weights), atol=1e-4)

    # Test case where all cumulative probs < (1-alpha) - triggers var_idx = 0 branch
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.1, 0.1, 0.8])
    cvar = cvar_estimator_from_dist(values, weights, 0.95)
    assert isinstance(cvar, (float, np.floating))
    assert not np.isnan(cvar)
    assert np.isclose(cvar, (0.05 * 1 + 0.1 * 2 + 0.8 * 3) / 0.95, atol=1e-4)

    # Test case where cumulative prob exactly equals (1-alpha)
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.5, 0.3, 0.2])
    # For alpha=0.8, cumulative probs are [0.5, 0.8, 1.0], so 0.8 exactly equals (1-0.2)
    cvar = cvar_estimator_from_dist(values, weights, 0.2)
    assert isinstance(cvar, (float, np.floating))
    assert not np.isnan(cvar)
    assert np.isclose(cvar, 3.0, atol=1e-4)

    # Test NaN in values (should propagate NaN)
    values = np.array([1.0, np.nan, 3.0])
    weights = np.array([0.33, 0.33, 0.34])
    cvar = cvar_estimator_from_dist(values, weights, 0.2)
    assert np.isnan(cvar)

    # Test Inf in values (should propagate NaN)
    values = np.array([1.0, np.inf, 3.0])
    weights = np.array([0.33, 0.33, 0.34])
    cvar = cvar_estimator_from_dist(values, weights, 0.2)
    assert np.isnan(cvar) or np.isinf(cvar)

    # Test invalid inputs
    with pytest.raises(ValueError):
        cvar_estimator_from_dist(np.array([1.0]), np.array([0.5]), 0.2)  # weights don't sum to 1

    with pytest.raises(ValueError):
        cvar_estimator_from_dist(np.array([]), np.array([]), 0.2)  # empty arrays

    with pytest.raises(ValueError):
        cvar_estimator_from_dist(np.array([1.0]), np.array([1.0]), 1.1)  # invalid alpha

    # Test array length mismatch (should raise ValueError)
    with pytest.raises(ValueError, match="Values and weights arrays must have the same length"):
        cvar_estimator_from_dist(np.array([1.0, 2.0]), np.array([0.5, 0.3, 0.2]), 0.2)


def test_cvar_estimator_from_dist_with_duplicate_values():
    """Test CVaR calculation with duplicate values and weight aggregation.

    Purpose: Validates that cvar_estimator_from_dist correctly handles duplicate
    values by aggregating their weights, ensuring the CVaR calculation is equivalent
    to the same distribution without duplicates

    Given: Distributions with duplicate values where the same value appears multiple
    times with different weights
    When: cvar_estimator_from_dist processes the distribution
    Then: Returns the same CVaR as the equivalent distribution with aggregated weights
    and unique values

    Test type: unit
    """
    # Test case 1: Multiple duplicates of the same value
    # Distribution: value 2.0 appears 3 times with weights 0.1, 0.2, 0.1
    # This should be equivalent to value 2.0 with weight 0.4
    values_with_duplicates = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
    weights_with_duplicates = np.array([0.3, 0.1, 0.2, 0.1, 0.3])

    # Equivalent distribution without duplicates
    values_unique = np.array([1.0, 2.0, 3.0])
    weights_unique = np.array([0.3, 0.4, 0.3])  # 0.1 + 0.2 + 0.1 = 0.4

    alpha = 0.2
    cvar_with_duplicates = cvar_estimator_from_dist(
        values_with_duplicates, weights_with_duplicates, alpha
    )
    cvar_unique = cvar_estimator_from_dist(values_unique, weights_unique, alpha)

    # Both should produce the same CVaR
    assert np.isclose(cvar_with_duplicates, cvar_unique, atol=1e-4)
    assert isinstance(cvar_with_duplicates, (float, np.floating))

    # Test case 2: Multiple groups of duplicates
    # Distribution: 1.0 appears twice, 3.0 appears three times
    values_multi_dup = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    weights_multi_dup = np.array([0.15, 0.15, 0.2, 0.1, 0.2, 0.2])

    # Equivalent distribution: 1.0 with 0.3, 2.0 with 0.2, 3.0 with 0.5
    values_multi_unique = np.array([1.0, 2.0, 3.0])
    weights_multi_unique = np.array([0.3, 0.2, 0.5])

    cvar_multi_dup = cvar_estimator_from_dist(values_multi_dup, weights_multi_dup, alpha)
    cvar_multi_unique = cvar_estimator_from_dist(values_multi_unique, weights_multi_unique, alpha)

    assert np.isclose(cvar_multi_dup, cvar_multi_unique, atol=1e-4)

    # Test case 3: Duplicates with different alpha values
    # Verify aggregation works correctly across different alpha thresholds
    values_dup = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    weights_dup = np.array([0.1, 0.2, 0.3, 0.2, 0.2])

    values_no_dup = np.array([0.0, 1.0, 2.0, 3.0])
    weights_no_dup = np.array([0.1, 0.5, 0.2, 0.2])  # 0.2 + 0.3 = 0.5

    for test_alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        cvar_dup = cvar_estimator_from_dist(values_dup, weights_dup, test_alpha)
        cvar_no_dup = cvar_estimator_from_dist(values_no_dup, weights_no_dup, test_alpha)
        assert np.isclose(
            cvar_dup, cvar_no_dup, atol=1e-4
        ), f"CVaR mismatch at alpha={test_alpha}: {cvar_dup} vs {cvar_no_dup}"

    # Test case 4: Duplicates at the tail (worst-case values)
    # This tests that duplicate aggregation works correctly for CVaR tail calculations
    values_tail_dup = np.array([1.0, 2.0, 5.0, 5.0, 5.0])
    weights_tail_dup = np.array([0.2, 0.3, 0.15, 0.2, 0.15])

    values_tail_unique = np.array([1.0, 2.0, 5.0])
    weights_tail_unique = np.array([0.2, 0.3, 0.5])  # 0.15 + 0.2 + 0.15 = 0.5

    alpha_tail = 0.3  # Focus on worst 30%
    cvar_tail_dup = cvar_estimator_from_dist(values_tail_dup, weights_tail_dup, alpha_tail)
    cvar_tail_unique = cvar_estimator_from_dist(values_tail_unique, weights_tail_unique, alpha_tail)

    assert np.isclose(cvar_tail_dup, cvar_tail_unique, atol=1e-4)
    # CVaR should be >= 5.0 since worst 30% includes the 5.0 values
    assert cvar_tail_dup >= 5.0 - 1e-4

    # Test case 5: All values are duplicates (edge case)
    # All values are the same, so aggregation should result in single value
    values_all_dup = np.array([10.0, 10.0, 10.0, 10.0])
    weights_all_dup = np.array([0.1, 0.3, 0.4, 0.2])

    cvar_all_dup = cvar_estimator_from_dist(values_all_dup, weights_all_dup, alpha)
    # Should return the single unique value
    assert np.isclose(cvar_all_dup, 10.0, atol=1e-4)

    # Test case 6: Duplicates with negative values
    values_neg_dup = np.array([-2.0, -1.0, -1.0, 0.0, 1.0])
    weights_neg_dup = np.array([0.2, 0.15, 0.25, 0.2, 0.2])

    values_neg_unique = np.array([-2.0, -1.0, 0.0, 1.0])
    weights_neg_unique = np.array([0.2, 0.4, 0.2, 0.2])  # 0.15 + 0.25 = 0.4

    cvar_neg_dup = cvar_estimator_from_dist(values_neg_dup, weights_neg_dup, alpha)
    cvar_neg_unique = cvar_estimator_from_dist(values_neg_unique, weights_neg_unique, alpha)

    assert np.isclose(cvar_neg_dup, cvar_neg_unique, atol=1e-4)


def test_aggregate_weights_for_duplicate_values_basic():
    """Test basic weight aggregation for duplicate values.

    Purpose: Validates that aggregate_weights_for_duplicate_values correctly
    combines weights for duplicate values and returns unique values with
    normalized aggregated weights

    Given: Arrays with duplicate values [1.0, 2.0, 2.0, 3.0] and weights [0.3, 0.2, 0.3, 0.2]
    When: aggregate_weights_for_duplicate_values processes the arrays
    Then: Returns unique values [1.0, 2.0, 3.0] and aggregated weights [0.3, 0.5, 0.2] that sum to 1

    Test type: unit
    """
    values = np.array([1.0, 2.0, 2.0, 3.0])
    weights = np.array([0.3, 0.2, 0.3, 0.2])
    unique_vals, agg_weights = aggregate_weights_for_duplicate_values(values, weights)

    # Check unique values
    expected_unique = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(unique_vals, expected_unique)

    # Check aggregated weights (2.0 appears twice with weights 0.2 and 0.3, so sum is 0.5)
    expected_weights = np.array([0.3, 0.5, 0.2])
    assert np.allclose(agg_weights, expected_weights)

    # Check weights sum to 1
    assert np.isclose(np.sum(agg_weights), 1.0)


def test_aggregate_weights_for_duplicate_values_no_duplicates():
    """Test weight aggregation with no duplicate values.

    Purpose: Validates that aggregate_weights_for_duplicate_values handles
    arrays with no duplicates correctly, returning the same values with normalized weights

    Given: Arrays with unique values [1.0, 2.0, 3.0] and weights [0.3, 0.4, 0.3]
    When: aggregate_weights_for_duplicate_values processes the arrays
    Then: Returns the same unique values and normalized weights that sum to 1

    Test type: unit
    """
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.3, 0.4, 0.3])
    unique_vals, agg_weights = aggregate_weights_for_duplicate_values(values, weights)

    # Should return same values (sorted)
    expected_unique = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(unique_vals, expected_unique)

    # Weights should be normalized (sum to 1)
    assert np.isclose(np.sum(agg_weights), 1.0)
    assert np.allclose(agg_weights, weights / np.sum(weights))


def test_aggregate_weights_for_duplicate_values_all_duplicates():
    """Test weight aggregation with all duplicate values.

    Purpose: Validates that aggregate_weights_for_duplicate_values correctly
    handles arrays where all values are the same, aggregating all weights

    Given: Arrays with all same values [5.0, 5.0, 5.0] and weights [0.2, 0.3, 0.5]
    When: aggregate_weights_for_duplicate_values processes the arrays
    Then: Returns single unique value [5.0] with aggregated weight [1.0]

    Test type: unit
    """
    values = np.array([5.0, 5.0, 5.0])
    weights = np.array([0.2, 0.3, 0.5])
    unique_vals, agg_weights = aggregate_weights_for_duplicate_values(values, weights)

    # Should return single unique value
    assert len(unique_vals) == 1
    assert np.isclose(unique_vals[0], 5.0)

    # Aggregated weight should be 1.0
    assert len(agg_weights) == 1
    assert np.isclose(agg_weights[0], 1.0)


def test_aggregate_weights_for_duplicate_values_multiple_duplicates():
    """Test weight aggregation with multiple groups of duplicates.

    Purpose: Validates that aggregate_weights_for_duplicate_values correctly
    handles multiple groups of duplicate values

    Given: Arrays with multiple duplicate groups [1.0, 1.0, 2.0, 3.0, 3.0, 3.0]
    and weights [0.1, 0.2, 0.1, 0.15, 0.2, 0.25]
    When: aggregate_weights_for_duplicate_values processes the arrays
    Then: Returns unique values [1.0, 2.0, 3.0] with correctly aggregated weights

    Test type: unit
    """
    values = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    weights = np.array([0.1, 0.2, 0.1, 0.15, 0.2, 0.25])
    unique_vals, agg_weights = aggregate_weights_for_duplicate_values(values, weights)

    # Check unique values
    expected_unique = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(unique_vals, expected_unique)

    # Check aggregated weights
    # 1.0: 0.1 + 0.2 = 0.3
    # 2.0: 0.1
    # 3.0: 0.15 + 0.2 + 0.25 = 0.6
    # Total: 1.0, normalized
    expected_weights = np.array([0.3, 0.1, 0.6])
    assert np.allclose(agg_weights, expected_weights)
    assert np.isclose(np.sum(agg_weights), 1.0)


def test_aggregate_weights_for_duplicate_values_floating_point_precision():
    """Test weight aggregation handles floating point precision correctly.

    Purpose: Validates that aggregate_weights_for_duplicate_values correctly
    normalizes weights even when input weights don't sum exactly to 1 due to
    floating point precision

    Given: Arrays with weights that sum to approximately 1.0 due to floating point
    When: aggregate_weights_for_duplicate_values processes the arrays
    Then: Returns normalized weights that sum exactly to 1.0

    Test type: unit
    """
    values = np.array([1.0, 2.0, 2.0])
    # Weights that might not sum exactly to 1.0 due to floating point
    weights = np.array([0.3333333333333333, 0.3333333333333333, 0.3333333333333334])
    unique_vals, agg_weights = aggregate_weights_for_duplicate_values(values, weights)

    # Weights should be normalized to sum exactly to 1.0
    assert np.isclose(np.sum(agg_weights), 1.0, rtol=1e-10)

    # Check that 2.0's weights are aggregated
    assert len(unique_vals) == 2
    assert np.array_equal(unique_vals, np.array([1.0, 2.0]))


def test_aggregate_weights_for_duplicate_values_negative_values():
    """Test weight aggregation with negative values.

    Purpose: Validates that aggregate_weights_for_duplicate_values correctly
    handles negative values

    Given: Arrays with negative duplicate values [-1.0, -1.0, 0.0, 1.0]
    and weights [0.25, 0.25, 0.25, 0.25]
    When: aggregate_weights_for_duplicate_values processes the arrays
    Then: Returns unique values with correctly aggregated weights

    Test type: unit
    """
    values = np.array([-1.0, -1.0, 0.0, 1.0])
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    unique_vals, agg_weights = aggregate_weights_for_duplicate_values(values, weights)

    expected_unique = np.array([-1.0, 0.0, 1.0])
    assert np.array_equal(unique_vals, expected_unique)

    expected_weights = np.array([0.5, 0.25, 0.25])
    assert np.allclose(agg_weights, expected_weights)
    assert np.isclose(np.sum(agg_weights), 1.0)


def test_aggregate_weights_for_duplicate_values_empty_input():
    """Test weight aggregation with empty input arrays.

    Purpose: Validates that aggregate_weights_for_duplicate_values raises
    ValueError for empty input arrays

    Given: Empty arrays
    When: aggregate_weights_for_duplicate_values is called
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match="Input arrays must not be empty"):
        aggregate_weights_for_duplicate_values(np.array([]), np.array([]))

    with pytest.raises(ValueError, match="Input arrays must not be empty"):
        aggregate_weights_for_duplicate_values(np.array([1.0]), np.array([]))

    with pytest.raises(ValueError, match="Input arrays must not be empty"):
        aggregate_weights_for_duplicate_values(np.array([]), np.array([0.5]))


def test_aggregate_weights_for_duplicate_values_length_mismatch():
    """Test weight aggregation with mismatched array lengths.

    Purpose: Validates that aggregate_weights_for_duplicate_values raises
    ValueError when values and weights arrays have different lengths

    Given: Arrays with mismatched lengths
    When: aggregate_weights_for_duplicate_values is called
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match="Values and weights arrays must have the same length"):
        aggregate_weights_for_duplicate_values(np.array([1.0, 2.0]), np.array([0.5, 0.3, 0.2]))

    with pytest.raises(ValueError, match="Values and weights arrays must have the same length"):
        aggregate_weights_for_duplicate_values(np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5]))


def test_aggregate_weights_for_duplicate_values_single_value():
    """Test weight aggregation with single value.

    Purpose: Validates that aggregate_weights_for_duplicate_values correctly
    handles single value arrays

    Given: Single value array [5.0] with weight [1.0]
    When: aggregate_weights_for_duplicate_values processes the arrays
    Then: Returns the same value with normalized weight [1.0]

    Test type: unit
    """
    values = np.array([5.0])
    weights = np.array([1.0])
    unique_vals, agg_weights = aggregate_weights_for_duplicate_values(values, weights)

    assert len(unique_vals) == 1
    assert np.isclose(unique_vals[0], 5.0)
    assert len(agg_weights) == 1
    assert np.isclose(agg_weights[0], 1.0)


# ============================================================================
# TV Distance Tests
# ============================================================================

# Mock distribution class for testing TV distance functions


class MockNormalDistribution(Distribution):
    """Mock normal distribution for testing TV distance functions."""

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def sample(self, n_samples: int = 1) -> List[Any]:
        """Sample from the normal distribution."""
        return np.random.normal(self.mean, self.std, n_samples).tolist()

    def probability(self, values: List[Any]) -> np.ndarray:
        """Compute probability density at given values (PDF for continuous distributions)."""
        values_array = np.array(values)
        return (1.0 / (self.std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((values_array - self.mean) / self.std) ** 2
        )


def test_tv_distance_identical_distributions():
    """Test TV distance between identical distributions.

    Purpose: Validates that TV distance returns approximately 0 for identical distributions

    Given: Two identical Normal(0, 1) distributions
    When: TV distance is computed using grid method
    Then: Returns value close to 0 (within tolerance)

    Test type: unit
    """
    dist1 = MockNormalDistribution(mean=0.0, std=1.0)
    dist2 = MockNormalDistribution(mean=0.0, std=1.0)

    tv = tv_distance_grid(dist1, dist2, x_min=-5.0, x_max=5.0, n_points=1000)
    assert isinstance(tv, float)
    assert 0.0 <= tv <= 1.0
    assert tv < 0.01  # Should be very close to 0 for identical distributions


def test_tv_distance_different_means():
    """Test TV distance between distributions with different means.

    Purpose: Validates that TV distance correctly measures difference when means differ

    Given: Normal(0, 1) and Normal(2, 1) distributions
    When: TV distance is computed using grid method
    Then: Returns value significantly greater than 0

    Test type: unit
    """
    dist1 = MockNormalDistribution(mean=0.0, std=1.0)
    dist2 = MockNormalDistribution(mean=2.0, std=1.0)

    tv = tv_distance_grid(dist1, dist2, x_min=-5.0, x_max=5.0, n_points=1000)
    assert isinstance(tv, float)
    assert 0.0 <= tv <= 1.0
    assert tv > 0.1  # Should be significantly different


def test_tv_distance_different_stds():
    """Test TV distance between distributions with different standard deviations.

    Purpose: Validates that TV distance correctly measures difference when standard deviations differ

    Given: Normal(0, 1) and Normal(0, 2) distributions
    When: TV distance is computed using grid method
    Then: Returns value greater than 0

    Test type: unit
    """
    dist1 = MockNormalDistribution(mean=0.0, std=1.0)
    dist2 = MockNormalDistribution(mean=0.0, std=2.0)

    tv = tv_distance_grid(dist1, dist2, x_min=-5.0, x_max=5.0, n_points=1000)
    assert isinstance(tv, float)
    assert 0.0 <= tv <= 1.0
    assert tv > 0.05  # Should be different


def test_tv_distance_grid_method():
    """Test grid-based TV distance computation.

    Purpose: Validates that tv_distance_grid produces deterministic results with proper bounds

    Given: Two different Normal distributions
    When: tv_distance_grid is called with different grid parameters
    Then: Returns consistent results and handles parameter variations correctly

    Test type: unit
    """
    dist1 = MockNormalDistribution(mean=0.0, std=1.0)
    dist2 = MockNormalDistribution(mean=1.0, std=1.0)

    # Test with default parameters
    tv1 = tv_distance_grid(dist1, dist2, x_min=-5.0, x_max=5.0, n_points=1000)
    assert isinstance(tv1, float)
    assert 0.0 <= tv1 <= 1.0

    # Test with more points (should be more accurate)
    tv2 = tv_distance_grid(dist1, dist2, x_min=-5.0, x_max=5.0, n_points=5000)
    assert isinstance(tv2, float)
    assert 0.0 <= tv2 <= 1.0
    # Results should be similar (within reasonable tolerance)
    assert abs(tv1 - tv2) < 0.1

    # Test with different range
    tv3 = tv_distance_grid(dist1, dist2, x_min=-10.0, x_max=10.0, n_points=1000)
    assert isinstance(tv3, float)
    assert 0.0 <= tv3 <= 1.0


def test_tv_distance_monte_carlo_method():
    """Test Monte Carlo TV distance computation.

    Purpose: Validates that tv_distance_monte_carlo produces reasonable estimates

    Given: Two different Normal distributions
    When: tv_distance_monte_carlo is called with different sample sizes
    Then: Returns values in [0, 1] range and shows consistency with more samples

    Test type: unit
    """
    dist1 = MockNormalDistribution(mean=0.0, std=1.0)
    dist2 = MockNormalDistribution(mean=1.0, std=1.0)

    # Test with small sample size
    tv1 = tv_distance_monte_carlo(dist1, dist2, n_samples=100)
    assert isinstance(tv1, float)
    assert 0.0 <= tv1 <= 1.0

    # Test with larger sample size
    tv2 = tv_distance_monte_carlo(dist1, dist2, n_samples=1000)
    assert isinstance(tv2, float)
    assert 0.0 <= tv2 <= 1.0

    # Both should be in reasonable range (Monte Carlo has variance)
    assert tv1 > 0.0 or tv2 > 0.0  # At least one should detect difference


def test_tv_distance_averaged_method():
    """Test averaged TV distance computation.

    Purpose: Validates that tv_distance_averaged reduces variance by averaging multiple runs

    Given: Two different Normal distributions
    When: tv_distance_averaged is called with different numbers of runs
    Then: Returns values in [0, 1] range

    Test type: unit
    """
    dist1 = MockNormalDistribution(mean=0.0, std=1.0)
    dist2 = MockNormalDistribution(mean=1.0, std=1.0)

    # Test with default parameters
    tv = tv_distance_averaged(dist1, dist2, n_samples=100, n_runs=5)
    assert isinstance(tv, float)
    assert 0.0 <= tv <= 1.0

    # Test with more runs
    tv2 = tv_distance_averaged(dist1, dist2, n_samples=100, n_runs=10)
    assert isinstance(tv2, float)
    assert 0.0 <= tv2 <= 1.0


def test_tv_distance_mixture_sampling_method():
    """Test mixture sampling TV distance computation.

    Purpose: Validates that tv_distance_mixture_sampling produces reasonable estimates

    Given: Two different Normal distributions
    When: tv_distance_mixture_sampling is called
    Then: Returns values in [0, 1] range

    Test type: unit
    """
    dist1 = MockNormalDistribution(mean=0.0, std=1.0)
    dist2 = MockNormalDistribution(mean=1.0, std=1.0)

    tv = tv_distance_mixture_sampling(dist1, dist2, n_samples=1000)
    assert isinstance(tv, float)
    assert 0.0 <= tv <= 1.0


def test_tv_distance_dispatcher_grid():
    """Test TV distance dispatcher with grid method.

    Purpose: Validates that tv_distance correctly dispatches to grid method

    Given: Two Normal distributions
    When: tv_distance is called with method="grid"
    Then: Returns same result as calling tv_distance_grid directly

    Test type: unit
    """
    dist1 = MockNormalDistribution(mean=0.0, std=1.0)
    dist2 = MockNormalDistribution(mean=1.0, std=1.0)

    tv1 = tv_distance(dist1, dist2, method="grid", x_min=-5.0, x_max=5.0, n_points=1000)
    tv2 = tv_distance_grid(dist1, dist2, x_min=-5.0, x_max=5.0, n_points=1000)

    assert isinstance(tv1, float)
    assert isinstance(tv2, float)
    assert np.isclose(tv1, tv2, rtol=1e-5)


def test_tv_distance_dispatcher_monte_carlo():
    """Test TV distance dispatcher with Monte Carlo method.

    Purpose: Validates that tv_distance correctly dispatches to Monte Carlo method

    Given: Two Normal distributions
    When: tv_distance is called with method="monte_carlo"
    Then: Returns result from Monte Carlo method

    Test type: unit
    """
    dist1 = MockNormalDistribution(mean=0.0, std=1.0)
    dist2 = MockNormalDistribution(mean=1.0, std=1.0)

    tv = tv_distance(dist1, dist2, method="monte_carlo", n_samples=1000)
    assert isinstance(tv, float)
    assert 0.0 <= tv <= 1.0


def test_tv_distance_dispatcher_averaged():
    """Test TV distance dispatcher with averaged method.

    Purpose: Validates that tv_distance correctly dispatches to averaged method

    Given: Two Normal distributions
    When: tv_distance is called with method="averaged"
    Then: Returns result from averaged method

    Test type: unit
    """
    dist1 = MockNormalDistribution(mean=0.0, std=1.0)
    dist2 = MockNormalDistribution(mean=1.0, std=1.0)

    tv = tv_distance(dist1, dist2, method="averaged", n_samples=100, n_runs=5)
    assert isinstance(tv, float)
    assert 0.0 <= tv <= 1.0


def test_tv_distance_dispatcher_mixture():
    """Test TV distance dispatcher with mixture method.

    Purpose: Validates that tv_distance correctly dispatches to mixture method

    Given: Two Normal distributions
    When: tv_distance is called with method="mixture"
    Then: Returns result from mixture method

    Test type: unit
    """
    dist1 = MockNormalDistribution(mean=0.0, std=1.0)
    dist2 = MockNormalDistribution(mean=1.0, std=1.0)

    tv = tv_distance(dist1, dist2, method="mixture", n_samples=1000)
    assert isinstance(tv, float)
    assert 0.0 <= tv <= 1.0


def test_tv_distance_dispatcher_invalid_method():
    """Test TV distance dispatcher with invalid method.

    Purpose: Validates that tv_distance raises ValueError for unknown methods

    Given: Two Normal distributions and invalid method name
    When: tv_distance is called with method="invalid"
    Then: ValueError is raised

    Test type: unit
    """
    dist1 = MockNormalDistribution(mean=0.0, std=1.0)
    dist2 = MockNormalDistribution(mean=1.0, std=1.0)

    with pytest.raises(ValueError, match="Unknown method"):
        tv_distance(dist1, dist2, method="invalid")


def test_tv_distance_symmetry():
    """Test TV distance symmetry property.

    Purpose: Validates that TV distance is symmetric: TV(p, q) = TV(q, p)

    Given: Two different Normal distributions
    When: TV distance is computed in both directions
    Then: Results are approximately equal

    Test type: unit
    """
    dist1 = MockNormalDistribution(mean=0.0, std=1.0)
    dist2 = MockNormalDistribution(mean=1.0, std=1.5)

    tv1 = tv_distance_grid(dist1, dist2, x_min=-5.0, x_max=5.0, n_points=1000)
    tv2 = tv_distance_grid(dist2, dist1, x_min=-5.0, x_max=5.0, n_points=1000)

    assert np.isclose(tv1, tv2, rtol=1e-5)


def test_tv_distance_triangle_inequality():
    """Test TV distance triangle inequality.

    Purpose: Validates that TV distance satisfies triangle inequality: TV(p, r) <= TV(p, q) + TV(q, r)

    Given: Three Normal distributions
    When: TV distances are computed between all pairs
    Then: Triangle inequality holds approximately

    Test type: unit
    """
    dist1 = MockNormalDistribution(mean=0.0, std=1.0)
    dist2 = MockNormalDistribution(mean=1.0, std=1.0)
    dist3 = MockNormalDistribution(mean=2.0, std=1.0)

    tv12 = tv_distance_grid(dist1, dist2, x_min=-5.0, x_max=5.0, n_points=1000)
    tv23 = tv_distance_grid(dist2, dist3, x_min=-5.0, x_max=5.0, n_points=1000)
    tv13 = tv_distance_grid(dist1, dist3, x_min=-5.0, x_max=5.0, n_points=1000)

    # Triangle inequality: TV(p, r) <= TV(p, q) + TV(q, r)
    # Allow small numerical tolerance
    assert tv13 <= tv12 + tv23 + 1e-3


def test_tv_distance_bounds():
    """Test TV distance bounds.

    Purpose: Validates that TV distance always returns values in [0, 1]

    Given: Various pairs of distributions
    When: TV distance is computed using different methods
    Then: All results are in [0, 1] range

    Test type: unit
    """
    dist1 = MockNormalDistribution(mean=0.0, std=1.0)
    dist2 = MockNormalDistribution(mean=5.0, std=0.5)

    # Test all methods
    tv_grid = tv_distance_grid(dist1, dist2, x_min=-10.0, x_max=10.0, n_points=1000)
    tv_mc = tv_distance_monte_carlo(dist1, dist2, n_samples=1000)
    tv_avg = tv_distance_averaged(dist1, dist2, n_samples=500, n_runs=5)
    tv_mix = tv_distance_mixture_sampling(dist1, dist2, n_samples=1000)

    assert 0.0 <= tv_grid <= 1.0
    assert 0.0 <= tv_mc <= 1.0
    assert 0.0 <= tv_avg <= 1.0
    assert 0.0 <= tv_mix <= 1.0
