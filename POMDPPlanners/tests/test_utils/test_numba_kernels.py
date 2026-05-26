# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Numerical-equivalence tests for the shared Numba kernels in ``POMDPPlanners.utils``.

Each test pairs a kernel against a pure-NumPy reference implementation and
asserts results agree within floating-point tolerance over a mix of
representative and edge-case inputs. These kernels are generic primitives
reused across POMDP environments — the tests here exercise them without any
env-specific semantics.
"""

import numpy as np
import pytest

from POMDPPlanners.utils.numba_kernels import (
    any_point_within_radius_kernel,
    min_distance_to_points_kernel,
    mvn_sample_2d_kernel,
)


def _ref_any_point_within_radius(query: np.ndarray, points: np.ndarray, radius: float) -> bool:
    distances = np.linalg.norm(query.reshape(2, 1) - points, axis=0)
    return bool(np.any(distances <= radius))


def _ref_min_distance_to_points(query: np.ndarray, points: np.ndarray) -> float:
    distances = np.linalg.norm(query.reshape(2, 1) - points, axis=0)
    return float(np.min(distances))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_points() -> np.ndarray:
    """3x3 grid of points in 2xN layout (stand-in for beacons/targets/rocks)."""
    xs = [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0]
    ys = [0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0]
    return np.array([xs, ys])


# ---------------------------------------------------------------------------
# any_point_within_radius_kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        np.array([0.0, 5.0]),
        np.array([2.5, 2.5]),
        np.array([5.0, 5.0]),
        np.array([5.4, 5.0]),
        np.array([10.0, 10.0]),
    ],
)
def test_any_point_within_radius_kernel_matches_reference(query, default_points):
    """Validates any_point_within_radius_kernel against NumPy reference.

    Purpose: Ensure boolean proximity checks match for representative queries.

    Given: Queries covering directly-on-point, inside-radius, outside-radius.
    When: Both the kernel and the reference are evaluated.
    Then: They return the same boolean across a range of radii.

    Test type: unit
    """
    for radius in (0.5, 1.0, 2.0):
        got = any_point_within_radius_kernel(query, default_points, radius)
        expected = _ref_any_point_within_radius(query, default_points, radius)
        assert got == expected, f"query={query} radius={radius}"


def test_any_point_within_radius_kernel_empty_points():
    """Validates any_point_within_radius_kernel with an empty (2, 0) array.

    Purpose: Edge case — no candidate points.

    Given: Empty 2x0 points array.
    When: The kernel is called.
    Then: Returns False.

    Test type: unit
    """
    empty = np.empty((2, 0))
    assert not any_point_within_radius_kernel(np.array([1.0, 1.0]), empty, 1.0)


# ---------------------------------------------------------------------------
# min_distance_to_points_kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        np.array([0.0, 5.0]),
        np.array([2.5, 2.5]),
        np.array([5.0, 5.0]),
        np.array([7.5, 2.5]),
    ],
)
def test_min_distance_to_points_kernel_matches_reference(query, default_points):
    """Validates min_distance_to_points_kernel numerical equivalence.

    Purpose: Ensure scalar min-distance output agrees with NumPy reference
        to within floating-point tolerance.

    Given: A query and a 3x3 point grid.
    When: Both min_distance_to_points_kernel and its reference are evaluated.
    Then: Results agree within 1e-12.

    Test type: unit
    """
    got = min_distance_to_points_kernel(query, default_points)
    expected = _ref_min_distance_to_points(query, default_points)
    assert np.isclose(got, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# mvn_sample_2d_kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [1, 5, 64])
def test_mvn_sample_2d_kernel_matches_reference(n_samples):
    """Validates mvn_sample_2d_kernel matches mean + z @ L.T.

    Purpose: Confirm the hand-rolled 2D matmul agrees with NumPy's @ operator
        to machine precision on representative covariance matrices.

    Given: Pre-drawn standard-normal z and a 2x2 Cholesky upper factor.
    When: The kernel and a NumPy reference both compute mean + z @ L.T.
    Then: Outputs are bit-identical within 1e-12 atol.

    Test type: unit
    """
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.3], [0.3, 2.0]])
    chol_L = np.linalg.cholesky(cov)
    chol_L_T = chol_L.T.copy()
    mean = np.array([1.5, -0.25])
    z = rng.standard_normal((n_samples, 2))

    got = mvn_sample_2d_kernel(mean, z, chol_L_T)
    expected = mean + z @ chol_L_T
    assert got.shape == (n_samples, 2)
    assert np.allclose(got, expected, atol=1e-12)
