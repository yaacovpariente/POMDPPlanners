"""Shared Numba-JIT kernels for POMDP environment hot paths.

Holds pure geometric / numerical kernels reused across POMDP environments
(light-dark, push, safety-ant-velocity, continuous laser-tag, ...). Each
kernel is decorated with ``@njit(cache=True)`` so the LLVM-compiled version
is cached to disk between runs.

Conventions
-----------
- **All inputs are plain numeric arrays / scalars.** No Python objects, no
  class instances. Callers hand in ``float64`` arrays (contiguous) and
  ``float``/``int``/``bool`` scalars.
- **RNG stays in Python.** Any stochastic draw (``np.random.standard_normal``,
  ``np.random.rand``) happens in the Python caller and is passed into the
  kernel (e.g. as ``z``). This preserves seed semantics and keeps outputs
  bit-identical under fixed seeds.
- **Point arrays are shape (2, N).** ``x`` coordinates on row 0, ``y`` on
  row 1. Matches the ``_convert_*_to_array`` helpers already used by the
  POMDP environment base classes.

Public kernels
--------------
- :func:`any_point_within_radius_kernel` — ``True`` iff any of ``N`` points
  lies within ``radius`` of the query.
- :func:`min_distance_to_points_kernel` — minimum Euclidean distance from
  the query to any of ``N`` points.
- :func:`mvn_sample_2d_kernel` — 2-D multivariate-normal sample given
  pre-drawn standard normals and a transposed Cholesky factor.
"""

import numpy as np
from numba import njit

# pylint: disable=no-value-for-parameter,not-an-iterable


@njit(cache=True)  # type: ignore[misc]
def any_point_within_radius_kernel(
    query: np.ndarray,
    points: np.ndarray,
    radius: float,
) -> bool:
    n_points = points.shape[1]
    radius_sq = radius * radius
    for i in range(n_points):
        dx = query[0] - points[0, i]
        dy = query[1] - points[1, i]
        if dx * dx + dy * dy <= radius_sq:
            return True
    return False


@njit(cache=True)  # type: ignore[misc]
def min_distance_to_points_kernel(
    query: np.ndarray,
    points: np.ndarray,
) -> float:
    n_points = points.shape[1]
    min_sq = np.inf
    for i in range(n_points):
        dx = query[0] - points[0, i]
        dy = query[1] - points[1, i]
        d_sq = dx * dx + dy * dy
        min_sq = min(min_sq, d_sq)
    return min_sq**0.5


@njit(cache=True)  # type: ignore[misc]
def mvn_sample_2d_kernel(
    mean: np.ndarray,
    z: np.ndarray,
    cholesky_L_T: np.ndarray,
) -> np.ndarray:
    """Sample n points from a 2-D Gaussian with fixed Cholesky factor.

    Given pre-drawn standard-normal ``z`` of shape ``(n, 2)`` and the
    transposed Cholesky factor ``L.T`` of shape ``(2, 2)``, compute
    ``mean + z @ L.T`` and return samples of shape ``(n, 2)``. The
    hand-rolled matmul avoids NumPy's per-call dispatch cost for the tiny
    ``(n, 2) @ (2, 2)`` product.
    """
    n_samples = z.shape[0]
    out = np.empty((n_samples, 2), dtype=np.float64)
    m0 = mean[0]
    m1 = mean[1]
    lt00 = cholesky_L_T[0, 0]
    lt01 = cholesky_L_T[0, 1]
    lt10 = cholesky_L_T[1, 0]
    lt11 = cholesky_L_T[1, 1]
    for i in range(n_samples):
        z0 = z[i, 0]
        z1 = z[i, 1]
        out[i, 0] = m0 + z0 * lt00 + z1 * lt10
        out[i, 1] = m1 + z0 * lt01 + z1 * lt11
    return out
