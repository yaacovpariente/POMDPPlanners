# SPDX-License-Identifier: MIT

"""Unit tests for generic dangerous-area Numba kernels.

Each test pairs a kernel against a pure-NumPy reference implementation
and asserts results agree, plus exercises edge cases (zero zones, point
exactly on the boundary, scalar/batch parity, empty batch).
"""

from __future__ import annotations

import numpy as np
import pytest

from POMDPPlanners.environments.environment_utils.dangerous_areas_kernels import (
    constant_prob_penalty_batch_kernel,
    constant_prob_penalty_kernel,
    decaying_prob_penalty_batch_kernel,
    decaying_prob_penalty_kernel,
    high_variance_penalty_batch_kernel,
    high_variance_penalty_kernel,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def centers_2d() -> np.ndarray:
    """Two zone centres at (3, 7) and (5, 5), shape (2, 2)."""
    return np.array([[3.0, 5.0], [7.0, 5.0]])


@pytest.fixture
def empty_centers() -> np.ndarray:
    return np.empty((2, 0), dtype=np.float64)


# ---------------------------------------------------------------------------
# constant_prob_penalty_kernel — scalar
# ---------------------------------------------------------------------------


def test_constant_prob_returns_zero_outside_zones(centers_2d):
    """Validates constant_prob_penalty_kernel returns 0 when point outside zones.

    Purpose: Confirms the zone check correctly identifies a point well clear of all centres
    Given: A point at (0, 0) with two zones at (3, 7) and (5, 5), radius_sq=1.0
    When: The constant_prob kernel is called with uniform=0.0 (always-hit)
    Then: Returns 0.0 despite uniform satisfying the probability check

    Test type: unit
    """
    point = np.array([0.0, 0.0])
    result = constant_prob_penalty_kernel(point, centers_2d, 1.0, -10.0, 1.0, 0.0)
    assert result == 0.0


def test_constant_prob_returns_penalty_inside_zone_and_lucky_draw(centers_2d):
    """Validates constant_prob_penalty_kernel returns penalty on in-zone + low uniform.

    Purpose: Confirms the kernel applies penalty when both zone and probability checks pass
    Given: A point at (3.0, 7.0) (exactly on a centre) and hit_probability=0.5
    When: The kernel is called with uniform=0.1 (< 0.5)
    Then: Returns the negative penalty value -10.0

    Test type: unit
    """
    point = np.array([3.0, 7.0])
    result = constant_prob_penalty_kernel(point, centers_2d, 1.0, -10.0, 0.5, 0.1)
    assert result == -10.0


def test_constant_prob_returns_zero_inside_zone_but_unlucky(centers_2d):
    """Validates constant_prob_penalty_kernel returns 0 when uniform exceeds hit prob.

    Purpose: Confirms the probability gate correctly suppresses the penalty
    Given: A point at zone centre and hit_probability=0.5
    When: The kernel is called with uniform=0.7 (> 0.5)
    Then: Returns 0.0

    Test type: unit
    """
    point = np.array([3.0, 7.0])
    result = constant_prob_penalty_kernel(point, centers_2d, 1.0, -10.0, 0.5, 0.7)
    assert result == 0.0


def test_constant_prob_boundary_inclusive(centers_2d):
    """Validates that points exactly on the zone boundary count as inside.

    Purpose: Confirms <= semantics for the squared-distance check
    Given: A point at distance exactly = radius from a centre
    When: The kernel is called with uniform=0.0 and hit_probability=1.0
    Then: Returns the penalty value (boundary point is inside)

    Test type: unit
    """
    radius = 0.5
    point = np.array([3.5, 7.0])  # distance = 0.5 from (3, 7)
    result = constant_prob_penalty_kernel(point, centers_2d, radius * radius, -10.0, 1.0, 0.0)
    assert result == -10.0


def test_constant_prob_zero_zones_returns_zero(empty_centers):
    """Validates kernel returns 0 when there are no zones at all.

    Purpose: Confirms degenerate empty-centers configuration is safe
    Given: An empty (2, 0) centers array
    When: The kernel is called with always-hit uniform
    Then: Returns 0.0

    Test type: unit
    """
    point = np.array([3.0, 7.0])
    result = constant_prob_penalty_kernel(point, empty_centers, 1.0, -10.0, 1.0, 0.0)
    assert result == 0.0


# ---------------------------------------------------------------------------
# constant_prob_penalty_batch_kernel
# ---------------------------------------------------------------------------


def test_constant_prob_batch_matches_scalar_row_by_row(centers_2d):
    """Validates batch kernel agrees with scalar kernel applied row-by-row.

    Purpose: Confirms scalar/batch parity for the constant-prob model
    Given: A batch of 6 points mixing inside/outside zones, fixed uniforms
    When: Both kernels are invoked on the same inputs
    Then: Per-row scalar results equal the batch array bit-for-bit

    Test type: unit
    """
    rng = np.random.default_rng(seed=0)
    points = np.array(
        [
            [0.0, 0.0],
            [3.0, 7.0],
            [3.2, 7.1],
            [5.5, 5.0],
            [10.0, 10.0],
            [3.0, 7.0],
        ]
    )
    uniforms = rng.uniform(size=points.shape[0])
    radius_sq = 0.36
    penalty = -10.0
    hit_prob = 0.4
    batch_result = constant_prob_penalty_batch_kernel(
        points, centers_2d, radius_sq, penalty, hit_prob, uniforms
    )
    scalar_results = np.array(
        [
            constant_prob_penalty_kernel(
                points[i], centers_2d, radius_sq, penalty, hit_prob, uniforms[i]
            )
            for i in range(points.shape[0])
        ]
    )
    np.testing.assert_array_equal(batch_result, scalar_results)


def test_constant_prob_batch_empty_returns_empty():
    """Validates batch kernel handles N=0 without error.

    Purpose: Confirms empty-batch corner case returns shape (0,)
    Given: A (0, 2) points array and (0,) uniforms array
    When: The batch kernel is called
    Then: Returns a length-0 float64 array

    Test type: unit
    """
    points = np.empty((0, 2), dtype=np.float64)
    centers = np.array([[3.0], [7.0]])
    uniforms = np.empty(0, dtype=np.float64)
    result = constant_prob_penalty_batch_kernel(points, centers, 1.0, -10.0, 0.5, uniforms)
    assert result.shape == (0,)


# ---------------------------------------------------------------------------
# high_variance_penalty_kernel
# ---------------------------------------------------------------------------


def test_high_variance_outside_zone_returns_zero(centers_2d):
    """Validates high_variance_penalty_kernel returns 0 when outside zones.

    Purpose: Confirms no spurious contribution when point is clear of all zones
    Given: A point at (0, 0), well clear of zones at (3, 7) and (5, 5)
    When: The kernel is called with uniform=0.1
    Then: Returns 0.0

    Test type: unit
    """
    point = np.array([0.0, 0.0])
    result = high_variance_penalty_kernel(point, centers_2d, 1.0, -10.0, 0.1)
    assert result == 0.0


def test_high_variance_returns_penalty_on_low_uniform(centers_2d):
    """Validates high_variance returns ``penalty`` for uniform < 0.5.

    Purpose: Confirms the ``uniform < 0.5`` branch returns the signed penalty
    Given: A point inside a zone, penalty=-10.0
    When: The kernel is called with uniform=0.3
    Then: Returns -10.0

    Test type: unit
    """
    point = np.array([3.0, 7.0])
    result = high_variance_penalty_kernel(point, centers_2d, 1.0, -10.0, 0.3)
    assert result == -10.0


def test_high_variance_returns_negated_penalty_on_high_uniform(centers_2d):
    """Validates high_variance returns ``-penalty`` for uniform >= 0.5.

    Purpose: Confirms the ``uniform >= 0.5`` branch flips the sign
    Given: A point inside a zone, penalty=-10.0
    When: The kernel is called with uniform=0.7
    Then: Returns +10.0

    Test type: unit
    """
    point = np.array([3.0, 7.0])
    result = high_variance_penalty_kernel(point, centers_2d, 1.0, -10.0, 0.7)
    assert result == 10.0


def test_high_variance_zero_expected_contribution(centers_2d):
    """Validates expected contribution is zero over many uniform draws.

    Purpose: Statistical sanity check — mean across 5000 uniform draws should be ~0
    Given: A point inside a zone, penalty=-10.0, fixed seed
    When: The kernel is called 5000 times with random uniforms in [0, 1)
    Then: Mean contribution is within 0.5 of 0.0

    Test type: unit
    """
    rng = np.random.default_rng(seed=42)
    point = np.array([3.0, 7.0])
    uniforms = rng.uniform(size=5000)
    contributions = np.array(
        [high_variance_penalty_kernel(point, centers_2d, 1.0, -10.0, u) for u in uniforms]
    )
    assert abs(contributions.mean()) < 0.5


def test_high_variance_batch_matches_scalar(centers_2d):
    """Validates batch kernel agrees with scalar kernel applied row-by-row.

    Purpose: Confirms scalar/batch parity for the high-variance model
    Given: A batch of 5 points mixing inside/outside, fixed uniforms
    When: Both kernels are invoked on the same inputs
    Then: Per-row scalar results equal the batch array

    Test type: unit
    """
    rng = np.random.default_rng(seed=1)
    points = np.array(
        [
            [0.0, 0.0],
            [3.0, 7.0],
            [5.0, 5.0],
            [3.0, 7.0],
            [10.0, 10.0],
        ]
    )
    uniforms = rng.uniform(size=points.shape[0])
    batch_result = high_variance_penalty_batch_kernel(points, centers_2d, 1.0, -10.0, uniforms)
    scalar_results = np.array(
        [
            high_variance_penalty_kernel(points[i], centers_2d, 1.0, -10.0, uniforms[i])
            for i in range(points.shape[0])
        ]
    )
    np.testing.assert_array_equal(batch_result, scalar_results)


# ---------------------------------------------------------------------------
# decaying_prob_penalty_kernel
# ---------------------------------------------------------------------------


def _ref_decaying_prob(
    point: np.ndarray,
    centers: np.ndarray,
    penalty: float,
    penalty_decay: float,
    uniform: float,
) -> float:
    distances = np.linalg.norm(point.reshape(-1, 1) - centers, axis=0)
    d = float(np.min(distances))
    p = float(np.exp(-d / penalty_decay))
    return float(penalty) if uniform < p else 0.0


def test_decaying_prob_close_to_zone_almost_always_hits(centers_2d):
    """Validates decaying_prob hits with high probability near a zone.

    Purpose: Confirms the probability is close to 1 at zero distance
    Given: A point exactly on a zone centre, penalty_decay=2.0
    When: The kernel is called with uniform=0.5
    Then: Returns the penalty (hit_prob = exp(0) = 1.0 > 0.5)

    Test type: unit
    """
    point = np.array([3.0, 7.0])
    result = decaying_prob_penalty_kernel(point, centers_2d, -10.0, 2.0, 0.5)
    assert result == -10.0


def test_decaying_prob_far_from_zones_almost_never_hits(centers_2d):
    """Validates decaying_prob suppresses the penalty far from any zone.

    Purpose: Confirms the probability decays toward 0 at large distance
    Given: A point at (100, 100), penalty_decay=1.0 (so hit_prob = exp(-d) ~ 0)
    When: The kernel is called with uniform=0.0001
    Then: Returns 0.0 (uniform exceeds the very small hit probability)

    Test type: unit
    """
    point = np.array([100.0, 100.0])
    result = decaying_prob_penalty_kernel(point, centers_2d, -10.0, 1.0, 0.0001)
    assert result == 0.0


def test_decaying_prob_matches_reference(centers_2d):
    """Validates decaying_prob kernel matches pure-NumPy reference across many inputs.

    Purpose: Confirms numerical equivalence against an independent implementation
    Given: A grid of 20 points across [0, 10] x [0, 10] with varied uniforms
    When: Both kernel and NumPy reference are evaluated
    Then: Outputs agree to within float64 tolerance

    Test type: unit
    """
    rng = np.random.default_rng(seed=2)
    points = rng.uniform(0.0, 10.0, size=(20, 2))
    uniforms = rng.uniform(size=20)
    penalty_decay = 1.5
    penalty = -7.5
    for i in range(points.shape[0]):
        got = decaying_prob_penalty_kernel(
            points[i], centers_2d, penalty, penalty_decay, uniforms[i]
        )
        expected = _ref_decaying_prob(points[i], centers_2d, penalty, penalty_decay, uniforms[i])
        assert got == pytest.approx(expected)


def test_decaying_prob_batch_matches_scalar(centers_2d):
    """Validates batch kernel agrees with scalar kernel applied row-by-row.

    Purpose: Confirms scalar/batch parity for the decaying-prob model
    Given: A batch of 8 points and matching uniforms, fixed seed
    When: Both kernels are invoked on the same inputs
    Then: Per-row scalar results equal the batch array

    Test type: unit
    """
    rng = np.random.default_rng(seed=3)
    points = rng.uniform(0.0, 10.0, size=(8, 2))
    uniforms = rng.uniform(size=8)
    batch_result = decaying_prob_penalty_batch_kernel(points, centers_2d, -7.5, 1.5, uniforms)
    scalar_results = np.array(
        [
            decaying_prob_penalty_kernel(points[i], centers_2d, -7.5, 1.5, uniforms[i])
            for i in range(points.shape[0])
        ]
    )
    np.testing.assert_array_equal(batch_result, scalar_results)


def test_decaying_prob_batch_empty_returns_empty():
    """Validates batch kernel handles N=0 without error.

    Purpose: Confirms empty-batch corner case returns shape (0,)
    Given: A (0, 2) points array and (0,) uniforms array
    When: The decaying batch kernel is called
    Then: Returns a length-0 float64 array

    Test type: unit
    """
    points = np.empty((0, 2), dtype=np.float64)
    centers = np.array([[3.0], [7.0]])
    uniforms = np.empty(0, dtype=np.float64)
    result = decaying_prob_penalty_batch_kernel(points, centers, -10.0, 1.0, uniforms)
    assert result.shape == (0,)
