# SPDX-License-Identifier: MIT

"""C++/Python parity tests for the Continuous Light-Dark reward variants.

These tests target a future ``_native.compute_reward_batch`` kernel that
will mirror the Python reward models bit-for-bit (in expectation). Today
no such kernel exists — reward is inlined inside ``_native.simulate_rollout``
— so every test in this module is expected to FAIL with
``AttributeError: module '_native' has no attribute 'compute_reward_batch'``.

This is a deliberate TDD red-step: once a Stage 2 agent lands
``_native.compute_reward_batch(states, action, next_states, *, reward_variant_code,
penalty_decay, ...)`` the tests must pass without further edits.
"""

# pylint: disable=protected-access  # Tests reach into reward_model for parity checks.

from typing import Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.light_dark_pomdp import _native  # pylint: disable=no-name-in-module
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    RewardModelType,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import continuous_light_dark_pinned_kwargs

_REWARD_VARIANT_CODES = {
    RewardModelType.CONSTANT_HAZARD_PENALTY: 0,
    RewardModelType.ZERO_MEAN_HAZARD_SHOCK: 1,
    RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY: 2,
}

_PENALTY_DECAY_BY_VARIANT = {
    RewardModelType.CONSTANT_HAZARD_PENALTY: 0.0,
    RewardModelType.ZERO_MEAN_HAZARD_SHOCK: 0.0,
    RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY: 1.5,
}


def _build_env(variant: RewardModelType) -> ContinuousLightDarkPOMDP:
    # Non-empty ``obstacles`` is mandatory so the obstacle-penalty branch
    # of every reward model gets exercised; ``penalty_decay`` is only
    # consumed by the DECAYING variant but is harmless elsewhere.
    return ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        **continuous_light_dark_pinned_kwargs(
            obstacles=[(3.0, 3.0), (6.0, 6.0), (8.0, 2.0)],
            reward_model_type=variant,
            penalty_decay=_PENALTY_DECAY_BY_VARIANT[variant],
        ),
    )


def _generate_inputs(
    env: ContinuousLightDarkPOMDP, n_samples: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    # Mix uniform-in-grid draws with explicit in-obstacle draws so the
    # obstacle-penalty path fires for a non-trivial fraction of the rows.
    grid = float(env.grid_size)
    states = rng.uniform(0.0, grid, size=(n_samples, 2)).astype(np.float64)
    obstacles_xy = np.ascontiguousarray(env.obstacles.T, dtype=np.float64)
    n_obstacle_rows = n_samples // 4
    obstacle_idx = rng.randint(0, obstacles_xy.shape[0], size=n_obstacle_rows)
    jitter = rng.uniform(-0.3, 0.3, size=(n_obstacle_rows, 2))
    states[:n_obstacle_rows] = obstacles_xy[obstacle_idx] + jitter
    action = np.array([0.5, 0.5], dtype=np.float64)
    next_states = states + action
    return (
        np.ascontiguousarray(states),
        action,
        np.ascontiguousarray(next_states),
    )


def _call_native_reward_batch(
    env: ContinuousLightDarkPOMDP,
    states: np.ndarray,
    action: np.ndarray,
    next_states: np.ndarray,
    variant: RewardModelType,
) -> np.ndarray:
    # Centralised dispatch so the AttributeError raised today (no
    # ``compute_reward_batch`` in ``_native``) surfaces in one place when
    # the kernel is added downstream.
    obstacles_flat = np.ascontiguousarray(env.obstacles.T.ravel(), dtype=np.float64)
    goal_state = np.ascontiguousarray(env.goal_state, dtype=np.float64)
    kernel = _native.compute_reward_batch  # type: ignore[attr-defined]
    return np.asarray(
        kernel(
            states=states,
            action=action,
            next_states=next_states,
            reward_variant_code=_REWARD_VARIANT_CODES[variant],
            penalty_decay=float(_PENALTY_DECAY_BY_VARIANT[variant]),
            goal_state=goal_state,
            obstacles=obstacles_flat,
            goal_state_radius=float(env.goal_state_radius),
            obstacle_radius=float(env.obstacle_radius),
            grid_size=float(env.grid_size),
            fuel_cost=float(env.fuel_cost),
            goal_reward=float(env.goal_reward),
            obstacle_reward=float(env.obstacle_reward),
            obstacle_hit_probability=float(env.obstacle_hit_probability),
        ),
        dtype=np.float64,
    )


class TestLightDarkRewardCppParity:
    """Sample-mean parity between Python reward models and the C++ kernel."""

    @pytest.mark.parametrize(
        "variant",
        [
            RewardModelType.CONSTANT_HAZARD_PENALTY,
            RewardModelType.ZERO_MEAN_HAZARD_SHOCK,
            RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY,
        ],
    )
    def test_cpp_python_reward_means_match(self, variant: RewardModelType) -> None:
        """Python and C++ reward batches share a sample mean within 3 SE.

        Purpose: Validates that the future ``_native.compute_reward_batch``
            reproduces the Python reward model's expected reward for each
            variant on the same (state, action, next_state) batch.

        Given: A ContinuousLightDarkPOMDP built with the variant under test,
            non-empty obstacles, and N=2000 seeded (state, action, next_state)
            triples — a quarter of which are placed inside obstacle radii so
            the stochastic penalty branch fires.
        When: ``env.reward_model.compute_reward_batch`` and
            ``_native.compute_reward_batch`` are invoked on the same inputs
            with matching ``reward_variant_code`` and ``penalty_decay``.
        Then: ``|py.mean() - cpp.mean()| < 3 * py.std()/sqrt(N) + 1e-9`` —
            the sample-mean gap stays inside a 3-sigma confidence band.

        Test type: integration
        """
        np.random.seed(42)
        env = _build_env(variant)
        n_samples = 2000
        states, action, next_states = _generate_inputs(env, n_samples, seed=42)

        np.random.seed(42)
        py_rewards = np.asarray(
            env.reward_model.compute_reward_batch(states, action, next_states=next_states),
            dtype=np.float64,
        )
        assert py_rewards.shape == (n_samples,)

        cpp_rewards = _call_native_reward_batch(env, states, action, next_states, variant)
        assert cpp_rewards.shape == (n_samples,)

        py_mean = float(py_rewards.mean())
        cpp_mean = float(cpp_rewards.mean())
        py_se = float(py_rewards.std(ddof=1) / np.sqrt(n_samples))
        tolerance = 3.0 * py_se + 1e-9
        gap = abs(py_mean - cpp_mean)
        assert gap < tolerance, (
            f"Variant {variant.value}: |py_mean - cpp_mean| = {gap:.6f} exceeds "
            f"3-sigma tolerance {tolerance:.6f} "
            f"(py_mean={py_mean:.6f}, cpp_mean={cpp_mean:.6f}, py_se={py_se:.6f})"
        )
