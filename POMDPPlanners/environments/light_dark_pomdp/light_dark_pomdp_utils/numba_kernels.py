"""Light-Dark-specific Numba-JIT kernels.

Holds kernels whose signatures or logic hardcode light-dark concepts
(goal+obstacles+out-of-grid shape, standard / high-variance-states / decaying-
hit-probability reward formulas). Generic geometric / numerical primitives
used here also by other envs live in
``POMDPPlanners.utils.numba_kernels`` instead.

Conventions match the shared module: contiguous ``float64`` arrays, scalar
floats/ints/bools, no Python objects, all RNG draws happen in the caller and
are passed in as parameters.

Public kernels
--------------
- :func:`is_terminal_kernel` — replaces ``ContinuousLightDarkPOMDP.is_terminal``.
- :func:`compute_reward_base_kernel` — deterministic part of the Standard /
  HighVarianceStates reward model plus an ``is_obstacle_hit_region`` flag so the
  Python caller can decide whether to draw a uniform.
- :func:`compute_reward_base_batch_kernel` — batched version of
  :func:`compute_reward_base_kernel`. Returns ``(rewards, obstacle_mask)`` so
  the Python caller can apply the stochastic obstacle-hit contribution where
  the mask is ``True``, preserving the seeded RNG call pattern.
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
def compute_reward_base_kernel(
    next_state: np.ndarray,
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
    Used by Standard and HighVarianceStates reward models.

    ``next_state`` is the realised post-transition position. The Python
    caller threads either the draw from
    :meth:`Environment.sample_next_step` or the deterministic
    ``state + action`` fallback when no realised draw is available.
    """
    next_x = next_state[0]
    next_y = next_state[1]
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
def compute_reward_base_batch_kernel(
    next_states: np.ndarray,
    goal_state: np.ndarray,
    obstacles: np.ndarray,
    goal_state_radius: float,
    obstacle_radius: float,
    grid_size: float,
    fuel_cost: float,
    goal_reward: float,
    obstacle_reward: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched form of :func:`compute_reward_base_kernel`.

    ``next_states`` is shape ``(N, 2)`` and contains the realised
    post-transition positions threaded by the Python caller (or the
    deterministic ``states + action`` fallback when no realised batch is
    available). Returns ``(rewards, obstacle_mask)`` both of length
    ``N``. ``rewards`` already includes fuel, goal-distance, goal bonus,
    and the out-of-grid penalty. The Python caller must add the
    stochastic obstacle-hit contribution at indices where
    ``obstacle_mask`` is ``True``, drawing its own RNG so seeded tests
    stay bit-identical to the per-state path.
    """
    n_states = next_states.shape[0]
    n_obs = obstacles.shape[1]
    goal_radius_sq = goal_state_radius * goal_state_radius
    obs_radius_sq = obstacle_radius * obstacle_radius
    gx0 = goal_state[0]
    gy0 = goal_state[1]

    rewards = np.empty(n_states, dtype=np.float64)
    obstacle_mask = np.zeros(n_states, dtype=np.bool_)

    for s in range(n_states):
        next_x = next_states[s, 0]
        next_y = next_states[s, 1]
        gx = next_x - gx0
        gy = next_y - gy0
        sq_dist_to_goal = gx * gx + gy * gy
        reward = -fuel_cost - sq_dist_to_goal**0.5

        if sq_dist_to_goal <= goal_radius_sq:
            rewards[s] = reward + goal_reward
            continue

        in_obstacle_range = False
        for i in range(n_obs):
            ox = next_x - obstacles[0, i]
            oy = next_y - obstacles[1, i]
            if ox * ox + oy * oy <= obs_radius_sq:
                in_obstacle_range = True
                break
        if in_obstacle_range:
            rewards[s] = reward
            obstacle_mask[s] = True
            continue

        if next_x < 0.0 or next_y < 0.0 or next_x > grid_size or next_y > grid_size:
            rewards[s] = reward + obstacle_reward
        else:
            rewards[s] = reward

    return rewards, obstacle_mask


@njit(cache=True)  # type: ignore[misc]
def compute_reward_decaying_hit_prob_kernel(
    next_state: np.ndarray,
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

    ``next_state`` is the realised post-transition position threaded by
    the Python caller. The caller draws a single ``uniform ~ U[0,1)`` (via
    ``np.random.rand()``) and passes it in — this matches the pre-refactor
    RNG call pattern, which unconditionally drew one uniform in
    ``_obstacle_reward``.
    """
    next_x = next_state[0]
    next_y = next_state[1]
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
