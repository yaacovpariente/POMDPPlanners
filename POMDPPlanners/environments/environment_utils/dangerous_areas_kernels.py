# SPDX-License-Identifier: MIT

"""Generic Numba kernels for dangerous-area reward contributions.

Each kernel maps a 2-D point (and a batch of points) to the *dangerous-
area* reward contribution under a specific stochastic model. The kernels
are env-agnostic so they can be reused by light-dark, pacman, laser-tag,
and any future grid / continuous environment that needs circular hazard
zones with a reward penalty.

Three reward-model variants are provided, each as a scalar / batch pair:

- **constant_prob**: penalty applied with a fixed probability whenever the
  point lies inside any zone. Matches the light-dark Standard reward
  model and the laser-tag / pacman dangerous-area contract (degenerate
  case ``hit_probability=1.0``).
- **high_variance**: ``±penalty`` with 50/50 split whenever the point
  lies inside any zone. Zero expected contribution, high variance.
  Matches the light-dark ZERO_MEAN_HAZARD_SHOCK reward model.
- **decaying_prob**: penalty applied with probability
  ``exp(-min_dist / penalty_decay)`` based on the *closest* zone centre
  (no radius cutoff). Matches the light-dark Decaying-Hit-Probability
  model.

Conventions (matching :mod:`POMDPPlanners.utils.numba_kernels`):

- All inputs are plain numeric arrays / scalars; no Python objects.
- Points are passed as ``np.ndarray`` of shape ``(2,)`` (scalar kernel)
  or ``(N, 2)`` (batch kernel).
- Zone centres are passed as a contiguous ``float64`` array of shape
  ``(2, D)`` — ``x`` coordinates on row 0, ``y`` on row 1. Matches the
  ``_convert_*_to_array`` helpers already used by light-dark.
- Radii are passed pre-squared (``radius_sq``) so the kernel never sqrts
  on the hot path and callers cache the squaring once at construction.
- RNG stays in Python. The caller draws ``uniform ~ U[0, 1)`` (or
  ``uniforms`` for the batch path) and passes them in. This preserves
  seed semantics and keeps outputs bit-identical under fixed seeds.
- Each kernel returns *only* the dangerous-area contribution. The caller
  composes this with whatever env-specific geometry (fuel, goal, grid
  bounds, ...) it needs.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# pylint: disable=no-value-for-parameter,not-an-iterable


@njit(cache=True)  # type: ignore[misc]
def membership_within_radius_batch_kernel(
    points: np.ndarray,
    centers: np.ndarray,
    radius_sq: float,
) -> np.ndarray:
    """Batched dangerous-area membership test.

    Returns a length-``N`` boolean array; entry ``i`` is ``True`` iff
    ``points[i]`` lies within ``sqrt(radius_sq)`` of *any* centre in
    ``centers``. ``points`` is shape ``(N, 2)``, ``centers`` is shape
    ``(2, D)`` (x on row 0, y on row 1, matching the convention of the
    other kernels in this module).

    Use this when the caller wants the in-zone mask separately from the
    reward contribution (e.g. so it can keep RNG handling in Python and
    preserve bit-identical seeded behaviour with a non-batched reference
    implementation).
    """
    n_points = points.shape[0]
    out = np.zeros(n_points, dtype=np.bool_)
    n_centers = centers.shape[1]
    for s in range(n_points):
        x = points[s, 0]
        y = points[s, 1]
        for i in range(n_centers):
            dx = x - centers[0, i]
            dy = y - centers[1, i]
            if dx * dx + dy * dy <= radius_sq:
                out[s] = True
                break
    return out


@njit(cache=True)  # type: ignore[misc]
def constant_prob_penalty_kernel(
    point: np.ndarray,
    centers: np.ndarray,
    radius_sq: float,
    penalty: float,
    hit_probability: float,
    uniform: float,
) -> float:
    """Constant-probability dangerous-area penalty.

    Returns ``penalty`` iff the point lies inside any of the ``D`` zones
    (squared-distance check against ``radius_sq``) *and*
    ``uniform < hit_probability``. Otherwise returns ``0.0``.
    """
    x = point[0]
    y = point[1]
    n_centers = centers.shape[1]
    for i in range(n_centers):
        dx = x - centers[0, i]
        dy = y - centers[1, i]
        if dx * dx + dy * dy <= radius_sq:
            if uniform < hit_probability:
                return penalty
            return 0.0
    return 0.0


@njit(cache=True)  # type: ignore[misc]
def constant_prob_penalty_batch_kernel(
    points: np.ndarray,
    centers: np.ndarray,
    radius_sq: float,
    penalty: float,
    hit_probability: float,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Batched form of :func:`constant_prob_penalty_kernel`.

    ``points`` is shape ``(N, 2)``, ``uniforms`` is shape ``(N,)``.
    Returns a length-``N`` array of contributions. ``uniforms[i]`` is
    consulted only when ``points[i]`` lies inside some zone, so callers
    can either pre-draw ``N`` uniforms or pre-draw only enough for the
    in-zone subset and pass that subset's rows.
    """
    n_points = points.shape[0]
    out = np.zeros(n_points, dtype=np.float64)
    n_centers = centers.shape[1]
    for s in range(n_points):
        x = points[s, 0]
        y = points[s, 1]
        for i in range(n_centers):
            dx = x - centers[0, i]
            dy = y - centers[1, i]
            if dx * dx + dy * dy <= radius_sq:
                if uniforms[s] < hit_probability:
                    out[s] = penalty
                break
    return out


@njit(cache=True)  # type: ignore[misc]
def high_variance_penalty_kernel(
    point: np.ndarray,
    centers: np.ndarray,
    radius_sq: float,
    penalty: float,
    uniform: float,
) -> float:
    """High-variance dangerous-area contribution.

    Returns ``penalty`` (with ``uniform < 0.5``) or ``-penalty``
    (otherwise) iff the point lies inside any of the ``D`` zones, else
    ``0.0``. Expected contribution is zero; the variance is
    ``penalty**2``. Pass the signed ``penalty`` value the caller wants
    on the ``uniform < 0.5`` branch (e.g. ``obstacle_reward = -10.0``);
    the kernel emits ``+10.0`` on the other branch.
    """
    x = point[0]
    y = point[1]
    n_centers = centers.shape[1]
    for i in range(n_centers):
        dx = x - centers[0, i]
        dy = y - centers[1, i]
        if dx * dx + dy * dy <= radius_sq:
            if uniform < 0.5:
                return penalty
            return -penalty
    return 0.0


@njit(cache=True)  # type: ignore[misc]
def high_variance_penalty_batch_kernel(
    points: np.ndarray,
    centers: np.ndarray,
    radius_sq: float,
    penalty: float,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Batched form of :func:`high_variance_penalty_kernel`."""
    n_points = points.shape[0]
    out = np.zeros(n_points, dtype=np.float64)
    n_centers = centers.shape[1]
    for s in range(n_points):
        x = points[s, 0]
        y = points[s, 1]
        for i in range(n_centers):
            dx = x - centers[0, i]
            dy = y - centers[1, i]
            if dx * dx + dy * dy <= radius_sq:
                if uniforms[s] < 0.5:
                    out[s] = penalty
                else:
                    out[s] = -penalty
                break
    return out


@njit(cache=True)  # type: ignore[misc]
def decaying_prob_penalty_kernel(
    point: np.ndarray,
    centers: np.ndarray,
    penalty: float,
    penalty_decay: float,
    uniform: float,
) -> float:
    """Distance-decaying dangerous-area penalty.

    Computes the minimum Euclidean distance from the point to any zone
    centre, then applies ``penalty`` with probability
    ``exp(-min_dist / penalty_decay)``. No radius cutoff — every point
    feels the (vanishingly small at large distance) penalty risk.
    """
    x = point[0]
    y = point[1]
    n_centers = centers.shape[1]
    min_sq = np.inf
    for i in range(n_centers):
        dx = x - centers[0, i]
        dy = y - centers[1, i]
        d_sq = dx * dx + dy * dy
        min_sq = min(min_sq, d_sq)
    min_dist = min_sq**0.5
    hit_prob = np.exp(-min_dist / penalty_decay)
    if uniform < hit_prob:
        return penalty
    return 0.0


@njit(cache=True)  # type: ignore[misc]
def decaying_prob_penalty_batch_kernel(
    points: np.ndarray,
    centers: np.ndarray,
    penalty: float,
    penalty_decay: float,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Batched form of :func:`decaying_prob_penalty_kernel`."""
    n_points = points.shape[0]
    out = np.zeros(n_points, dtype=np.float64)
    n_centers = centers.shape[1]
    for s in range(n_points):
        x = points[s, 0]
        y = points[s, 1]
        min_sq = np.inf
        for i in range(n_centers):
            dx = x - centers[0, i]
            dy = y - centers[1, i]
            d_sq = dx * dx + dy * dy
            min_sq = min(min_sq, d_sq)
        min_dist = min_sq**0.5
        hit_prob = np.exp(-min_dist / penalty_decay)
        if uniforms[s] < hit_prob:
            out[s] = penalty
    return out


# Reward-variant codes shared with the native (C++) kernels. These mirror the
# ``_REWARD_VARIANT_CODE_BY_TYPE`` mappings used across the environments so the
# hazard hit-probability kernel below dispatches identically to the transition
# and reward paths.
CONSTANT_HAZARD_PENALTY_CODE = 0
ZERO_MEAN_HAZARD_SHOCK_CODE = 1
DISTANCE_DECAYED_HAZARD_PENALTY_CODE = 2


@njit(cache=True)  # type: ignore[misc]
def hazard_hit_probability_kernel(
    point: np.ndarray,
    centers: np.ndarray,
    radius_sq: float,
    hit_probability: float,
    penalty_decay: float,
    reward_variant_code: int,
) -> float:
    """Hazard hit-probability at ``point`` for draw-coupled termination.

    Returns the probability that landing at ``point`` (without having
    reached the goal) triggers an absorbing hazard hit. The draw-coupled
    transition draws a single ``uniform ~ U[0, 1)`` and marks the next
    state terminal iff ``uniform < hazard_hit_probability_kernel(...)``.
    The returned probability is *exactly* the probability with which the
    corresponding reward penalty fires today, so termination and penalty
    stay perfectly coupled:

    - ``CONSTANT_HAZARD_PENALTY_CODE`` (0): returns ``hit_probability`` iff
      the point lies inside any of the ``D`` zones (squared-distance check
      against ``radius_sq``), else ``0.0``. Matches
      :func:`constant_prob_penalty_kernel`.
    - ``DISTANCE_DECAYED_HAZARD_PENALTY_CODE`` (2): returns
      ``exp(-min_dist / penalty_decay)`` for the closest zone centre, with
      no radius cutoff. Matches :func:`decaying_prob_penalty_kernel`.
    - ``ZERO_MEAN_HAZARD_SHOCK_CODE`` (1) or any other code: returns
      ``0.0``. The shock model has no hit probability, so draw-coupled
      termination does not apply; callers must reject that combination
      before reaching this kernel.
    """
    x = point[0]
    y = point[1]
    n_centers = centers.shape[1]
    if reward_variant_code == DISTANCE_DECAYED_HAZARD_PENALTY_CODE:
        min_sq = np.inf
        for i in range(n_centers):
            dx = x - centers[0, i]
            dy = y - centers[1, i]
            d_sq = dx * dx + dy * dy
            min_sq = min(min_sq, d_sq)
        return np.exp(-(min_sq**0.5) / penalty_decay)
    if reward_variant_code == CONSTANT_HAZARD_PENALTY_CODE:
        for i in range(n_centers):
            dx = x - centers[0, i]
            dy = y - centers[1, i]
            if dx * dx + dy * dy <= radius_sq:
                return hit_probability
        return 0.0
    return 0.0


@njit(cache=True)  # type: ignore[misc]
def hazard_hit_probability_batch_kernel(
    points: np.ndarray,
    centers: np.ndarray,
    radius_sq: float,
    hit_probability: float,
    penalty_decay: float,
    reward_variant_code: int,
) -> np.ndarray:
    """Batched form of :func:`hazard_hit_probability_kernel`.

    ``points`` is shape ``(N, 2)``; returns a length-``N`` array of hazard
    hit-probabilities. Vectorized transition / reward paths use this to
    decide draw-coupled termination for a batch of realised next states.
    """
    n_points = points.shape[0]
    out = np.zeros(n_points, dtype=np.float64)
    for s in range(n_points):
        out[s] = hazard_hit_probability_kernel(
            points[s],
            centers,
            radius_sq,
            hit_probability,
            penalty_decay,
            reward_variant_code,
        )
    return out
