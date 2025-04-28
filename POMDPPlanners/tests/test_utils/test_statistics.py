import pytest

import numpy as np
from scipy.stats import norm

from POMDPPlanners.utils.statistics import cvar_estimator


def test_cvar_estimator_negative_values():
    """Test CVaR calculation with negative values."""
    values = np.array([-5.0, -3.0, -1.0, 0.0, 2.0])
    alpha = 0.8
    result = cvar_estimator(values, alpha)
    # For alpha=0.8, we expect the CVaR to be the mean of the highest 20% values
    assert np.isclose(result, -0.5, rtol=1e-5)


def test_cvar_estimator_alpha_boundaries():
    """Test CVaR calculation at alpha boundaries."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Test alpha = 1.0 (should return maximum value)
    result_alpha_1 = cvar_estimator(values, 1.0)
    assert np.isclose(result_alpha_1, 5.0)

    # Test alpha = 0.0 (should return minimum value)
    result_alpha_0 = cvar_estimator(values, 0.0)
    assert np.isclose(result_alpha_0, 1.0)


def test_cvar_estimator_identical_values():
    """Test CVaR calculation with identical values."""
    values = np.array([10.0, 10.0, 10.0, 10.0])
    alpha = 0.9
    result = cvar_estimator(values, alpha)
    assert np.isclose(result, 10.0)


def test_cvar_estimator_invalid_input():
    """Test CVaR estimator with invalid inputs."""
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
    """Test CVaR calculation with a known distribution."""
    # Create a uniform distribution
    values = np.linspace(0, 1, 1000)
    alpha = 0.9

    # For uniform distribution U[0,1], CVaR of highest values at alpha is (1 + alpha)/2
    theoretical_cvar = (1 + (1 - alpha)) / 2
    result = cvar_estimator(values, alpha)

    assert np.isclose(result, theoretical_cvar, rtol=1e-2)


def test_cvar_estimator_mixed_cases():
    """Test CVaR calculation with mixed cases and different alphas."""
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
    """Test CVaR calculation with a single-element array."""
    vec4 = np.array([1.0])
    result = cvar_estimator(vec4, alpha=0.5)
    assert np.isclose(result, 1.0)


def test_cvar_estimator_from_dist():
    """Test CVaR calculation from discrete probability distribution."""
    from POMDPPlanners.utils.statistics import cvar_estimator_from_dist
    
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
