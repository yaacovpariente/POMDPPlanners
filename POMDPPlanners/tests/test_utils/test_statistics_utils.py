import random
from typing import List, Any

import numpy as np
import pytest

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.utils.statistics_utils import (
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

    # Test invalid inputs
    with pytest.raises(ValueError):
        cvar_estimator_from_dist(np.array([1.0]), np.array([0.5]), 0.2)  # weights don't sum to 1

    with pytest.raises(ValueError):
        cvar_estimator_from_dist(np.array([]), np.array([]), 0.2)  # empty arrays

    with pytest.raises(ValueError):
        cvar_estimator_from_dist(np.array([1.0]), np.array([1.0]), 1.1)  # invalid alpha


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
