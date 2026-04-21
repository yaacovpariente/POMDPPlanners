"""Numba-JIT kernels for the Continuous Light-Dark POMDP hot paths.

This module holds pure numeric kernels that replace the NumPy-overhead-bound
inner loops of the continuous Light-Dark environment. Each kernel is decorated
with ``@njit(cache=True)`` so the LLVM-compiled version is cached to disk
between runs.

Conventions
-----------
- **All inputs are plain numeric arrays / scalars.** No Python objects, no
  class instances. Callers hand in ``float64`` arrays (contiguous) and
  ``float``/``int``/``bool`` scalars.
- **RNG stays in Python.** Any stochastic draw (``np.random.standard_normal``,
  ``np.random.rand``) happens in the Python caller and is passed in as ``z``
  or ``u``. This preserves the seed semantics of the pre-refactor code and
  guarantees bit-identical results under seeded tests.
- **Obstacle / beacon arrays are shape (2, N).** This matches the
  internal representation produced by ``_convert_*_to_array`` in
  ``base_light_dark_pomdp.py``.

Public kernels
--------------
- :func:`is_terminal_kernel` — replaces ``ContinuousLightDarkPOMDP.is_terminal``.
- :func:`near_beacon_kernel` — replaces ``_near_beacon`` proximity check.
- :func:`min_distance_to_beacon_kernel` — distance to the nearest beacon.
- :func:`mvn_sample_2d_kernel` — 2-D multivariate-normal sample given
  pre-drawn standard normals and a Cholesky upper factor.
- :func:`compute_reward_base_kernel` — deterministic part of the Standard /
  DangerousStates reward model plus an ``is_obstacle_hit_region`` flag so the
  Python caller can decide whether to draw a uniform.
- :func:`compute_reward_decaying_hit_prob_kernel` — full reward for the
  Decaying-Hit-Probability model (uniform drawn in Python and passed in).
"""

from typing import Tuple

import numpy as np
from numba import njit

# pylint: disable=no-value-for-parameter,not-an-iterable


@njit(cache=True)  # type: ignore[misc]
def is_terminal_kernel(
    state: np.ndarray,
    goal_state: np.ndarray,
    obstacles: np.ndarray,
    goal_state_radius: float,
    obstacle_radius: float,
    is_obstacle_hit_terminal: bool,
) -> bool:
    dx = state[0] - goal_state[0]
    dy = state[1] - goal_state[1]
    if (dx * dx + dy * dy) ** 0.5 <= goal_state_radius:
        return True

    if not is_obstacle_hit_terminal:
        return False

    n_obs = obstacles.shape[1]
    radius_sq = obstacle_radius * obstacle_radius
    for i in range(n_obs):
        ox = state[0] - obstacles[0, i]
        oy = state[1] - obstacles[1, i]
        if ox * ox + oy * oy <= radius_sq:
            return True
    return False


@njit(cache=True)  # type: ignore[misc]
def near_beacon_kernel(
    next_state: np.ndarray,
    beacons: np.ndarray,
    beacon_radius: float,
) -> bool:
    n_beacons = beacons.shape[1]
    radius_sq = beacon_radius * beacon_radius
    for i in range(n_beacons):
        dx = next_state[0] - beacons[0, i]
        dy = next_state[1] - beacons[1, i]
        if dx * dx + dy * dy <= radius_sq:
            return True
    return False


@njit(cache=True)  # type: ignore[misc]
def min_distance_to_beacon_kernel(
    next_state: np.ndarray,
    beacons: np.ndarray,
) -> float:
    n_beacons = beacons.shape[1]
    min_sq = np.inf
    for i in range(n_beacons):
        dx = next_state[0] - beacons[0, i]
        dy = next_state[1] - beacons[1, i]
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


@njit(cache=True)  # type: ignore[misc]
def compute_reward_base_kernel(
    state: np.ndarray,
    action: np.ndarray,
    goal_state: np.ndarray,
    obstacles: np.ndarray,
    goal_state_radius: float,
    obstacle_radius: float,
    grid_size: float,
    fuel_cost: float,
    goal_reward: float,
    obstacle_reward: float,
) -> Tuple[float, bool]:
    """Return ``(base_reward, is_obstacle_hit_region)``.

    ``base_reward`` already includes fuel, goal-distance, and the
    out-of-grid penalty. The caller (Python) must add the stochastic
    obstacle-hit contribution when ``is_obstacle_hit_region`` is ``True``,
    using its own ``np.random.rand()`` draw so seeded tests stay bit-identical.
    Used by Standard and DangerousStates reward models.
    """
    next_x = state[0] + action[0]
    next_y = state[1] + action[1]
    gx = next_x - goal_state[0]
    gy = next_y - goal_state[1]
    dist_to_goal = (gx * gx + gy * gy) ** 0.5
    reward = -fuel_cost - dist_to_goal
    is_goal = dist_to_goal <= goal_state_radius
    if is_goal:
        return reward + goal_reward, False

    n_obs = obstacles.shape[1]
    obs_radius_sq = obstacle_radius * obstacle_radius
    in_obstacle_range = False
    for i in range(n_obs):
        ox = next_x - obstacles[0, i]
        oy = next_y - obstacles[1, i]
        if ox * ox + oy * oy <= obs_radius_sq:
            in_obstacle_range = True
            break
    if in_obstacle_range:
        return reward, True

    if next_x < 0.0 or next_y < 0.0 or next_x > grid_size or next_y > grid_size:
        return reward + obstacle_reward, False
    return reward, False


@njit(cache=True)  # type: ignore[misc]
def compute_reward_decaying_hit_prob_kernel(
    state: np.ndarray,
    action: np.ndarray,
    goal_state: np.ndarray,
    obstacles: np.ndarray,
    goal_state_radius: float,
    grid_size: float,
    fuel_cost: float,
    goal_reward: float,
    obstacle_reward: float,
    penalty_decay: float,
    uniform: float,
) -> float:
    """Full reward for the Decaying-Hit-Probability model.

    The caller draws a single ``uniform ~ U[0,1)`` (via ``np.random.rand()``)
    and passes it in. Matches the pre-refactor RNG call pattern, which
    unconditionally drew one uniform in ``_obstacle_reward``.
    """
    next_x = state[0] + action[0]
    next_y = state[1] + action[1]
    gx = next_x - goal_state[0]
    gy = next_y - goal_state[1]
    dist_to_goal = (gx * gx + gy * gy) ** 0.5
    reward = -fuel_cost - dist_to_goal

    is_goal = dist_to_goal <= goal_state_radius
    is_out_of_grid = next_x < 0.0 or next_y < 0.0 or next_x > grid_size or next_y > grid_size
    if is_goal:
        reward += goal_reward
    elif is_out_of_grid:
        reward += obstacle_reward

    n_obs = obstacles.shape[1]
    min_obs_sq = np.inf
    for i in range(n_obs):
        ox = next_x - obstacles[0, i]
        oy = next_y - obstacles[1, i]
        d_sq = ox * ox + oy * oy
        min_obs_sq = min(min_obs_sq, d_sq)
    min_obs_dist = min_obs_sq**0.5
    hit_prob = np.exp(-min_obs_dist / penalty_decay)
    if uniform < hit_prob:
        reward += obstacle_reward
    return reward
