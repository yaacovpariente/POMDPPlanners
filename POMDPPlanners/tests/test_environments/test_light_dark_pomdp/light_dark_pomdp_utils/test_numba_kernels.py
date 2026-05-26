# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Numerical-equivalence tests for the light-dark-specific Numba kernels.

Each test pairs a kernel against a pure-NumPy reference implementation
assembled from the pre-refactor class methods, and asserts results agree
within floating-point tolerance. Covers only kernels whose signature /
logic hardcodes light-dark concepts (goal+obstacles+reward shape); tests
for shared generic kernels live under ``POMDPPlanners/tests/test_utils/``.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.numba_kernels import (
    compute_reward_base_kernel,
    compute_reward_decaying_hit_prob_kernel,
    is_terminal_kernel,
)


def _ref_is_terminal(
    state: np.ndarray,
    goal_state: np.ndarray,
    obstacles: np.ndarray,
    goal_state_radius: float,
    obstacle_radius: float,
    is_obstacle_hit_terminal: bool,
) -> bool:
    is_goal = bool(np.linalg.norm(state - goal_state) <= goal_state_radius)
    if is_goal:
        return True
    if not is_obstacle_hit_terminal:
        return False
    distances = np.linalg.norm(state.reshape(-1, 1) - obstacles, axis=0)
    return bool(np.any(distances <= obstacle_radius))


def _ref_compute_reward_standard(
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
    obstacle_hit_probability: float,
    uniform: float,
) -> float:
    next_state = state + action
    is_goal = bool(np.linalg.norm(next_state - goal_state) <= goal_state_radius)
    is_in_obs = bool(
        (np.linalg.norm(next_state.reshape(-1, 1) - obstacles, axis=0) <= obstacle_radius).any()
    )
    is_oog = bool(np.any(next_state < 0) or np.any(next_state > grid_size))
    reward = float(-fuel_cost - np.linalg.norm(next_state - goal_state))
    if is_goal:
        reward += goal_reward
    elif is_in_obs:
        if uniform < obstacle_hit_probability:
            reward += obstacle_reward
    elif is_oog:
        reward += obstacle_reward
    return reward


def _ref_compute_reward_decaying(
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
    next_state = state + action
    is_goal = bool(np.linalg.norm(next_state - goal_state) <= goal_state_radius)
    is_oog = bool(np.any(next_state < 0) or np.any(next_state > grid_size))
    reward = float(-fuel_cost - np.linalg.norm(next_state - goal_state))
    if is_goal:
        reward += goal_reward
    elif is_oog:
        reward += obstacle_reward
    distances = np.linalg.norm(next_state.reshape(-1, 1) - obstacles, axis=0)
    d = float(np.min(distances))
    p = float(np.exp(-d / penalty_decay))
    if uniform < p:
        reward += obstacle_reward
    return reward


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_obstacles() -> np.ndarray:
    """Two obstacles at (3, 7) and (5, 5) in 2xN layout (matches env default)."""
    return np.array([[3.0, 5.0], [7.0, 5.0]])


@pytest.fixture
def default_goal() -> np.ndarray:
    return np.array([10.0, 5.0])


# ---------------------------------------------------------------------------
# is_terminal_kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [
        np.array([0.0, 5.0]),
        np.array([5.0, 5.0]),
        np.array([9.9, 5.1]),
        np.array([3.5, 7.2]),
        np.array([-0.5, 5.0]),
    ],
)
def test_is_terminal_kernel_matches_reference(state, default_goal, default_obstacles):
    """Validates is_terminal_kernel numerical equivalence to the NumPy reference.

    Purpose: Ensure the Numba is_terminal kernel agrees with the pre-refactor
        ContinuousLightDarkPOMDP.is_terminal logic on representative states.

    Given: Various states inside/outside goal, obstacle, and grid regions.
    When: is_terminal_kernel and _ref_is_terminal are both evaluated.
    Then: They return the same boolean.

    Test type: unit
    """
    for is_term in (True, False):
        got = is_terminal_kernel(state, default_goal, default_obstacles, 1.5, 1.5, is_term)
        expected = _ref_is_terminal(state, default_goal, default_obstacles, 1.5, 1.5, is_term)
        assert got == expected


def test_is_terminal_kernel_obstacle_hit_flag_off(default_goal, default_obstacles):
    """Validates that obstacle checks are skipped when is_obstacle_hit_terminal=False.

    Purpose: Ensure that with the obstacle-terminal flag disabled the kernel
        returns False on an obstacle overlap, matching env behavior.

    Given: A state directly on an obstacle coordinate.
    When: is_terminal_kernel is called with is_obstacle_hit_terminal=False.
    Then: The result is False.

    Test type: unit
    """
    # default_obstacles layout: row0=[x0,x1]=[3,5], row1=[y0,y1]=[7,5] → obstacles at (3,7),(5,5)
    on_obstacle = np.array([3.0, 7.0])
    assert not is_terminal_kernel(on_obstacle, default_goal, default_obstacles, 1.5, 1.5, False)
    assert is_terminal_kernel(on_obstacle, default_goal, default_obstacles, 1.5, 1.5, True)


def test_is_terminal_kernel_empty_obstacles():
    """Validates is_terminal_kernel accepts an empty (2, 0) obstacles array.

    Purpose: Edge case — no obstacles configured.

    Given: An empty 2x0 obstacles array.
    When: is_terminal_kernel is called on a non-goal state.
    Then: Returns False (no goal, no obstacles).

    Test type: unit
    """
    empty_obs = np.empty((2, 0))
    state = np.array([5.0, 5.0])
    assert not is_terminal_kernel(state, np.array([10.0, 5.0]), empty_obs, 1.5, 1.5, True)


# ---------------------------------------------------------------------------
# compute_reward_base_kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state,action,hit_prob,uniform,expected_branch",
    [
        # at goal — no obstacle-hit-region flag
        (np.array([9.9, 5.0]), np.array([0.05, 0.0]), 0.2, 0.1, "goal"),
        # in obstacle region — flag True, uniform < hit_prob → obstacle reward
        (np.array([2.0, 5.0]), np.array([1.0, 0.0]), 0.2, 0.1, "obstacle_hit"),
        # in obstacle region — flag True, uniform >= hit_prob → no obstacle reward
        (np.array([2.0, 5.0]), np.array([1.0, 0.0]), 0.2, 0.5, "obstacle_miss"),
        # out of grid — flag False, reward includes OOG penalty
        (np.array([0.5, 0.5]), np.array([-1.0, 0.0]), 0.2, 0.0, "out_of_grid"),
        # plain interior — flag False, reward is just fuel + dist_to_goal
        (np.array([1.0, 5.0]), np.array([1.0, 0.0]), 0.2, 0.0, "interior"),
    ],
)
def test_compute_reward_base_kernel_matches_reference(
    state, action, hit_prob, uniform, expected_branch, default_goal, default_obstacles
):
    """Validates compute_reward_base_kernel across all reward branches.

    Purpose: Verify the deterministic base reward plus the
        is_obstacle_hit_region flag against the NumPy reference, for every
        branch the Standard reward model can take.

    Given: State/action pairs that land in each branch (goal, obstacle-hit,
        obstacle-miss, out-of-grid, interior) and matching uniforms.
    When: The kernel's (reward, flag) is composed in Python with the same
        branching used by ContinuousLightDarkRewardModel._compute_reward,
        then compared to _ref_compute_reward_standard.
    Then: Numerical equivalence to 1e-12.

    Test type: unit
    """
    del expected_branch  # descriptive parameter only
    # NOTE: ``compute_reward_base_kernel`` now consumes the realised
    # ``next_state`` threaded by ``Environment.sample_next_step``. The
    # original test predates that signature and exercised the
    # deterministic ``state + action`` landing point — pass that here.
    next_state = state + action
    reward, should_draw = compute_reward_base_kernel(
        next_state, default_goal, default_obstacles, 1.5, 1.5, 11.0, 2.0, 10.0, -10.0
    )
    if should_draw and uniform < hit_prob:
        reward += -10.0
    expected = _ref_compute_reward_standard(
        state,
        action,
        default_goal,
        default_obstacles,
        1.5,
        1.5,
        11.0,
        2.0,
        10.0,
        -10.0,
        hit_prob,
        uniform,
    )
    assert np.isclose(reward, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# compute_reward_decaying_hit_prob_kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state,action,uniform",
    [
        (np.array([9.9, 5.0]), np.array([0.05, 0.0]), 0.1),  # at goal
        (np.array([2.0, 5.0]), np.array([1.0, 0.0]), 0.01),  # near obstacle, likely hit
        (np.array([1.0, 5.0]), np.array([1.0, 0.0]), 0.99),  # interior, unlikely hit
        (np.array([0.5, 0.5]), np.array([-1.0, 0.0]), 0.5),  # out of grid
    ],
)
def test_compute_reward_decaying_hit_prob_kernel_matches_reference(
    state, action, uniform, default_goal, default_obstacles
):
    """Validates compute_reward_decaying_hit_prob_kernel numerical equivalence.

    Purpose: Verify the decaying-hit-probability reward model kernel agrees
        with the NumPy reference across goal / obstacle-proximity / OOG /
        interior cases.

    Given: Representative state/action pairs and uniforms spanning the
        decision thresholds.
    When: The kernel and reference both compute the full reward.
    Then: Numerical equivalence to 1e-12.

    Test type: unit
    """
    # NOTE: kernel now consumes the realised ``next_state``; pass the
    # deterministic ``state + action`` to preserve the original test
    # intent (no transition noise).
    next_state = state + action
    got = compute_reward_decaying_hit_prob_kernel(
        next_state,
        default_goal,
        default_obstacles,
        1.5,
        11.0,
        2.0,
        10.0,
        -10.0,
        1.0,
        uniform,
    )
    expected = _ref_compute_reward_decaying(
        state,
        action,
        default_goal,
        default_obstacles,
        1.5,
        11.0,
        2.0,
        10.0,
        -10.0,
        1.0,
        uniform,
    )
    assert np.isclose(got, expected, atol=1e-12)
