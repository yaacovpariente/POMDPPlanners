"""C++/Python reward-parity tests for the Push POMDP reward variants.

Verifies that the future native reward kernels
``_native.push_reward_batch`` (discrete) and
``_native.cont_push_reward_batch`` (continuous) reproduce — in
expectation — the Python reward-model output across all three
:class:`RewardModelType` variants: ``STANDARD``,
``HIGH_VARIANCE_STATES`` and ``DECAYING_HIT_PROBABILITY``.

Today no standalone C++ reward kernel exists for the Push family —
reward is inlined inside ``_native.simulate_rollout_discrete`` /
``_native.cont_simulate_rollout``. Every test in this module is
therefore expected to FAIL today (with ``AttributeError`` because the
``_native`` module has no ``push_reward_batch`` /
``cont_push_reward_batch`` attribute). This is a deliberate TDD
red-step: once a Stage 2 agent lands the standalone kernels accepting
``reward_variant_code`` and ``penalty_decay`` kwargs the tests must go
green without further edits.
"""

# pylint: disable=protected-access  # Tests reach into reward_model for parity checks.

import math
from typing import Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.push_pomdp import _native
from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import (
    ContinuousPushPOMDP,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP
from POMDPPlanners.environments.push_pomdp.push_pomdp_utils.push_reward_models import (
    RewardModelType,
)


_BATCH_SIZE = 2000
_PENALTY_DECAY = 1.5
_DANGEROUS_AREA_PENALTY = -10.0
_DANGEROUS_AREA_RADIUS = 0.8
_DANGEROUS_AREAS_DISCRETE = [(3.0, 3.0), (6.0, 6.0), (8.0, 2.0)]
_DANGEROUS_AREAS_CONTINUOUS = [(3.0, 3.0), (6.0, 6.0), (8.0, 2.0)]

_REWARD_VARIANT_CODES = {
    RewardModelType.STANDARD: 0,
    RewardModelType.HIGH_VARIANCE_STATES: 1,
    RewardModelType.DECAYING_HIT_PROBABILITY: 2,
}

_PENALTY_DECAY_BY_VARIANT = {
    RewardModelType.STANDARD: 0.0,
    RewardModelType.HIGH_VARIANCE_STATES: 0.0,
    RewardModelType.DECAYING_HIT_PROBABILITY: _PENALTY_DECAY,
}


def _sample_mean_parity(py_rewards: np.ndarray, cpp_rewards: np.ndarray) -> Tuple[float, float]:
    """Return ``(|delta|, tolerance)`` for the sample-mean parity assertion."""
    n = py_rewards.shape[0]
    delta = abs(float(py_rewards.mean()) - float(cpp_rewards.mean()))
    tol = 3.0 * (float(py_rewards.std(ddof=0)) / math.sqrt(n)) + 1e-9
    return delta, tol


def _build_discrete_env(variant: RewardModelType) -> PushPOMDP:
    """Build a discrete :class:`PushPOMDP` with the requested reward variant."""
    return PushPOMDP(
        discount_factor=0.95,
        grid_size=10,
        dangerous_areas=_DANGEROUS_AREAS_DISCRETE,
        dangerous_area_radius=_DANGEROUS_AREA_RADIUS,
        dangerous_area_penalty=_DANGEROUS_AREA_PENALTY,
        dangerous_area_hit_probability=1.0,
        reward_model_type=variant,
        penalty_decay=(
            _PENALTY_DECAY_BY_VARIANT[variant]
            if variant == RewardModelType.DECAYING_HIT_PROBABILITY
            else 1.0
        ),
    )


def _build_continuous_env(variant: RewardModelType) -> ContinuousPushPOMDP:
    """Build a continuous :class:`ContinuousPushPOMDP` with the requested variant."""
    return ContinuousPushPOMDP(
        discount_factor=0.95,
        grid_size=10,
        dangerous_areas=_DANGEROUS_AREAS_CONTINUOUS,
        dangerous_area_radius=_DANGEROUS_AREA_RADIUS,
        dangerous_area_penalty=_DANGEROUS_AREA_PENALTY,
        dangerous_area_hit_probability=1.0,
        reward_model_type=variant,
        penalty_decay=(
            _PENALTY_DECAY_BY_VARIANT[variant]
            if variant == RewardModelType.DECAYING_HIT_PROBABILITY
            else 1.0
        ),
    )


def _build_discrete_states(env: PushPOMDP, n_samples: int) -> np.ndarray:
    """Return an (N, 6) state batch mixing dangerous-area and clear robot cells.

    Half of the rows are placed near dangerous-area centres so each variant's
    danger-penalty path fires; the remaining rows are uniformly distributed
    over the grid. Object and target positions are uniform-in-grid so the
    distance / goal-bonus terms vary across rows.
    """
    grid = float(env.grid_size - 1)
    danger_centres = np.asarray(_DANGEROUS_AREAS_DISCRETE, dtype=np.float64)
    half = n_samples // 2
    danger_idx = np.random.randint(0, danger_centres.shape[0], size=half)
    danger_rows = danger_centres[danger_idx] + np.random.uniform(-0.2, 0.2, size=(half, 2))
    clear_rows = np.random.uniform(0.0, grid, size=(n_samples - half, 2))
    robot_xy = np.concatenate([danger_rows, clear_rows], axis=0)
    np.random.shuffle(robot_xy)
    object_xy = np.random.uniform(0.0, grid, size=(n_samples, 2))
    target_xy = np.tile(env.target_pos.astype(np.float64), (n_samples, 1))
    return np.ascontiguousarray(np.concatenate([robot_xy, object_xy, target_xy], axis=1))


def _build_continuous_states(env: ContinuousPushPOMDP, n_samples: int) -> np.ndarray:
    """Return an (N, 6) state batch covering dangerous-area + clear continuous cells."""
    grid = float(env.grid_size - 1)
    danger_centres = np.asarray(_DANGEROUS_AREAS_CONTINUOUS, dtype=np.float64)
    half = n_samples // 2
    danger_idx = np.random.randint(0, danger_centres.shape[0], size=half)
    danger_rows = danger_centres[danger_idx] + np.random.uniform(-0.2, 0.2, size=(half, 2))
    clear_rows = np.random.uniform(0.0, grid, size=(n_samples - half, 2))
    robot_xy = np.concatenate([danger_rows, clear_rows], axis=0)
    np.random.shuffle(robot_xy)
    object_xy = np.random.uniform(0.0, grid, size=(n_samples, 2))
    target_xy = np.tile(env.target_pos.astype(np.float64), (n_samples, 1))
    return np.ascontiguousarray(np.concatenate([robot_xy, object_xy, target_xy], axis=1))


def _call_native_push_reward_batch(
    env: PushPOMDP,
    states: np.ndarray,
    action: str,
    next_states: np.ndarray,
    variant: RewardModelType,
) -> np.ndarray:
    """Invoke the future ``_native.push_reward_batch`` kernel directly.

    Today this raises ``AttributeError`` because ``_native`` has no
    ``push_reward_batch`` attribute — that is the intended failure for
    this red-step test.
    """
    obstacles_arr = np.asarray(env.obstacles, dtype=np.float64).reshape(-1, 2)
    action_idx = int(env.actions.index(action))
    kernel = _native.push_reward_batch  # type: ignore[attr-defined]
    return np.asarray(
        kernel(
            states=np.ascontiguousarray(states, dtype=np.float64),
            action_idx=action_idx,
            next_states=np.ascontiguousarray(next_states, dtype=np.float64),
            obstacles=np.ascontiguousarray(obstacles_arr, dtype=np.float64),
            obstacle_radius=float(env.obstacle_radius),
            obstacle_penalty=float(env.obstacle_penalty),
            obstacle_hit_probability=float(env.obstacle_hit_probability),
            dangerous_areas=env._dangerous_areas_arr,
            dangerous_area_radius=float(env.dangerous_area_radius),
            dangerous_area_penalty=float(env.dangerous_area_penalty),
            dangerous_area_hit_probability=float(env.dangerous_area_hit_probability),
            reward_variant_code=int(_REWARD_VARIANT_CODES[variant]),
            penalty_decay=float(_PENALTY_DECAY_BY_VARIANT[variant]),
        ),
        dtype=np.float64,
    )


def _call_native_cont_push_reward_batch(
    env: ContinuousPushPOMDP,
    states: np.ndarray,
    action: np.ndarray,
    next_states: np.ndarray,
    variant: RewardModelType,
) -> np.ndarray:
    """Invoke the future ``_native.cont_push_reward_batch`` kernel directly.

    Today this raises ``AttributeError`` because ``_native`` has no
    ``cont_push_reward_batch`` attribute.
    """
    kernel = _native.cont_push_reward_batch  # type: ignore[attr-defined]
    return np.asarray(
        kernel(
            states=np.ascontiguousarray(states, dtype=np.float64),
            action=np.ascontiguousarray(action, dtype=np.float64),
            next_states=np.ascontiguousarray(next_states, dtype=np.float64),
            obstacles=np.ascontiguousarray(env.obstacles, dtype=np.float64),
            robot_radius=float(env.robot_radius),
            obstacle_penalty=float(env.obstacle_penalty),
            obstacle_hit_probability=float(env.obstacle_hit_probability),
            dangerous_areas=env._dangerous_areas_arr,
            dangerous_area_radius=float(env.dangerous_area_radius),
            dangerous_area_penalty=float(env.dangerous_area_penalty),
            dangerous_area_hit_probability=float(env.dangerous_area_hit_probability),
            reward_variant_code=int(_REWARD_VARIANT_CODES[variant]),
            penalty_decay=float(_PENALTY_DECAY_BY_VARIANT[variant]),
        ),
        dtype=np.float64,
    )


class TestDiscretePushRewardCppParity:
    """Sample-mean parity between Python and the C++ kernel for :class:`PushPOMDP`."""

    @pytest.mark.parametrize(
        "variant",
        [
            RewardModelType.STANDARD,
            RewardModelType.HIGH_VARIANCE_STATES,
            RewardModelType.DECAYING_HIT_PROBABILITY,
        ],
    )
    def test_cpp_python_reward_means_match(self, variant: RewardModelType) -> None:
        """C++ reward-batch mean matches the Python reward-model mean within 3 SE.

        Purpose: Validates that the future
            ``_native.push_reward_batch`` reproduces the Python
            reward-model's expected reward for each
            :class:`RewardModelType` variant on the same
            ``(state, action, next_state)`` batch.

        Given: A discrete :class:`PushPOMDP` constructed with the chosen
            ``reward_model_type``, non-empty ``dangerous_areas``,
            ``dangerous_area_penalty = -10.0``,
            ``dangerous_area_hit_probability = 1.0`` (so STANDARD is
            deterministic) and (for the Decaying variant)
            ``penalty_decay = 1.5``. A seeded batch of ``N = 2000``
            states is generated mixing dangerous-area and clear cells,
            and ``next_states`` is realised via
            ``env.sample_next_state_batch``.
        When: ``env.reward_model.compute_reward_batch`` and
            ``_native.push_reward_batch`` are evaluated on the same
            batch and action with matching ``reward_variant_code`` and
            ``penalty_decay`` kwargs.
        Then: ``|py.mean() - cpp.mean()| < 3 * py.std()/sqrt(N) + 1e-9``
            — the sample-mean gap stays inside a 3-sigma confidence
            band. Today this fails with ``AttributeError`` because the
            kernel does not yet exist.

        Test type: integration
        """
        np.random.seed(42)
        env = _build_discrete_env(variant)
        states = _build_discrete_states(env, _BATCH_SIZE)
        action = env.get_actions()[0]
        next_states = env.sample_next_state_batch(states, action)

        py_rewards = np.asarray(
            env.reward_model.compute_reward_batch(states, action, next_states),
            dtype=np.float64,
        )
        assert py_rewards.shape == (_BATCH_SIZE,)

        cpp_rewards = _call_native_push_reward_batch(env, states, action, next_states, variant)
        assert cpp_rewards.shape == (_BATCH_SIZE,)

        delta, tol = _sample_mean_parity(py_rewards, cpp_rewards)
        assert delta < tol, (
            f"C++/Python reward mean mismatch for discrete variant {variant.name}: "
            f"|py.mean - cpp.mean| = {delta:.6f} >= tol = {tol:.6f} "
            f"(py.mean={py_rewards.mean():.6f}, cpp.mean={cpp_rewards.mean():.6f}, "
            f"N={py_rewards.shape[0]})"
        )


class TestContinuousPushRewardCppParity:
    """Sample-mean parity between Python and the C++ kernel for :class:`ContinuousPushPOMDP`."""

    @pytest.mark.parametrize(
        "variant",
        [
            RewardModelType.STANDARD,
            RewardModelType.HIGH_VARIANCE_STATES,
            RewardModelType.DECAYING_HIT_PROBABILITY,
        ],
    )
    def test_cpp_python_reward_means_match(self, variant: RewardModelType) -> None:
        """C++ reward-batch mean matches the Python reward-model mean within 3 SE.

        Purpose: Validates that the future
            ``_native.cont_push_reward_batch`` reproduces the Python
            reward-model's expected reward for each
            :class:`RewardModelType` variant on the same
            ``(state, action, next_state)`` batch.

        Given: A :class:`ContinuousPushPOMDP` built with the variant
            under test, non-empty ``dangerous_areas``,
            ``dangerous_area_penalty = -10.0``,
            ``dangerous_area_hit_probability = 1.0`` and (for the
            Decaying variant) ``penalty_decay = 1.5``. A seeded batch
            of ``N = 2000`` states is generated mixing dangerous-area
            and clear cells; ``next_states`` is realised via
            ``env.sample_next_state_batch`` (deterministic action vector
            ``[1.0, 0.0]``).
        When: ``env.reward_model.compute_reward_batch`` and
            ``_native.cont_push_reward_batch`` are evaluated on the same
            batch and action with matching ``reward_variant_code`` and
            ``penalty_decay`` kwargs.
        Then: ``|py.mean() - cpp.mean()| < 3 * py.std()/sqrt(N) + 1e-9``
            — the sample-mean gap stays inside a 3-sigma confidence
            band. Today this fails with ``AttributeError`` because the
            kernel does not yet exist.

        Test type: integration
        """
        np.random.seed(42)
        env = _build_continuous_env(variant)
        states = _build_continuous_states(env, _BATCH_SIZE)
        action = np.array([1.0, 0.0], dtype=np.float64)
        next_states = env.sample_next_state_batch(states, action)

        py_rewards = np.asarray(
            env.reward_model.compute_reward_batch(states, action, next_states),
            dtype=np.float64,
        )
        assert py_rewards.shape == (_BATCH_SIZE,)

        cpp_rewards = _call_native_cont_push_reward_batch(env, states, action, next_states, variant)
        assert cpp_rewards.shape == (_BATCH_SIZE,)

        delta, tol = _sample_mean_parity(py_rewards, cpp_rewards)
        assert delta < tol, (
            f"C++/Python reward mean mismatch for continuous variant {variant.name}: "
            f"|py.mean - cpp.mean| = {delta:.6f} >= tol = {tol:.6f} "
            f"(py.mean={py_rewards.mean():.6f}, cpp.mean={cpp_rewards.mean():.6f}, "
            f"N={py_rewards.shape[0]})"
        )
