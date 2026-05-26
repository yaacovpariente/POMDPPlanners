# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""C++/Python reward-parity tests for the discrete LaserTag reward variants.

Verifies that the native ``_native.lasertag_discrete_reward_batch`` C++
kernel produces results that match the Python reward-model
implementation across all three :class:`RewardModelType` variants —
``CONSTANT_HAZARD_PENALTY``, ``ZERO_MEAN_HAZARD_SHOCK`` and ``DISTANCE_DECAYED_HAZARD_PENALTY``.

This is a TDD red-step test for Stage 2. The current native kernel does
not accept a ``reward_variant_code`` / ``penalty_decay`` keyword pair,
so today the calls below either raise ``TypeError`` (kwargs unknown)
or — once the kernel grows the kwargs but still hard-codes CONSTANT_HAZARD_PENALTY
semantics — produce values that fail the sample-mean parity check on
the stochastic variants. The Stage 2 agent will teach the kernel about
the variant code and decay length; this test will go green then.
"""

import math
from typing import Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.laser_tag_pomdp import _native
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import (
    LaserTagPOMDP,
    RewardModelType,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_utils.laser_tag_reward_models import (
    LaserTagRewardModel,
)


_BATCH_SIZE = 2000
_PENALTY_DECAY = 1.5
_DANGEROUS_AREAS = {(5, 3), (7, 1), (2, 5)}
_DANGEROUS_AREA_PENALTY = 5.0


def _variant_code(rmt: RewardModelType) -> int:
    """Map a :class:`RewardModelType` to the integer code the C++ kernel expects."""
    if rmt == RewardModelType.CONSTANT_HAZARD_PENALTY:
        return 0
    if rmt == RewardModelType.ZERO_MEAN_HAZARD_SHOCK:
        return 1
    if rmt == RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY:
        return 2
    raise ValueError(f"Unknown reward model type: {rmt}")


def _build_env(rmt: RewardModelType) -> LaserTagPOMDP:
    """Build a discrete LaserTag env with non-empty danger zones and the chosen variant."""
    return LaserTagPOMDP(
        discount_factor=0.95,
        dangerous_areas=_DANGEROUS_AREAS,
        dangerous_area_penalty=_DANGEROUS_AREA_PENALTY,
        reward_model_type=rmt,
        penalty_decay=_PENALTY_DECAY,
    )


def _build_states_with_danger_and_clear(env: LaserTagPOMDP, n: int) -> np.ndarray:
    """Return an (N, 5) state batch covering both danger-zone and clear cells.

    Half of the rows are placed at / near dangerous-area centres so the
    variant's danger-penalty path fires; the other half are far from any
    centre so the wall / clear baseline contributes. All rows are
    non-terminal so the reward path is fully exercised.
    """
    rows, cols = env.floor_shape
    danger_centres = np.array(env.dangerous_areas, dtype=np.float64)
    half = n // 2
    danger_rows = danger_centres[np.random.randint(0, len(danger_centres), size=half)]
    clear_rows = np.column_stack(
        [
            np.random.randint(0, rows, size=n - half),
            np.random.randint(0, cols, size=n - half),
        ]
    ).astype(np.float64)
    robot_xy = np.concatenate([danger_rows, clear_rows], axis=0)
    np.random.shuffle(robot_xy)
    opp_xy = np.column_stack(
        [
            np.random.randint(0, rows, size=n),
            np.random.randint(0, cols, size=n),
        ]
    ).astype(np.float64)
    terminal = np.zeros((n, 1), dtype=np.float64)
    return np.ascontiguousarray(np.concatenate([robot_xy, opp_xy, terminal], axis=1))


def _call_cpp_reward_batch(
    env: LaserTagPOMDP,
    states: np.ndarray,
    action: int,
    next_states: np.ndarray,
    variant_code: int,
    penalty_decay: float,
) -> np.ndarray:
    """Invoke the C++ ``lasertag_discrete_reward_batch`` kernel directly.

    Passes the proposed Stage 2 ``reward_variant_code`` / ``penalty_decay``
    / ``next_states`` keyword arguments. Today this raises ``TypeError``
    because the kernel does not yet accept these kwargs — that is the
    intended failure for this red-step test.
    """
    assert isinstance(env.reward_model, LaserTagRewardModel)
    rm = env.reward_model
    # The ``next_states`` / ``reward_variant_code`` / ``penalty_decay`` kwargs
    # are part of the Stage 2 kernel surface and are not yet declared in
    # ``_native.pyi``; pyright is correct that they are absent today, which is
    # exactly the red-step state this test is asserting against. Suppress the
    # report so the test file stays at 0 pyright errors / warnings while still
    # exercising the (currently missing) kernel surface.
    kwargs = {
        "states": np.ascontiguousarray(states, dtype=np.float64),
        "action": int(action),
        "rows": int(env.floor_shape[0]),
        "cols": int(env.floor_shape[1]),
        "walls_flat": rm._reward_walls_flat,  # pylint: disable=protected-access
        "n_walls": rm._reward_n_walls,  # pylint: disable=protected-access
        "dangerous_areas": rm._dangerous_areas_arr,  # pylint: disable=protected-access
        "n_dangerous": int(rm._dangerous_areas_arr.shape[0]),  # pylint: disable=protected-access
        "dangerous_area_radius": float(env.dangerous_area_radius),
        "dangerous_area_penalty": float(env.dangerous_area_penalty),
        "tag_reward": float(env.tag_reward),
        "tag_penalty": float(env.tag_penalty),
        "step_cost": float(env.step_cost),
        "action_directions": rm._action_directions_arr,  # pylint: disable=protected-access
        "next_states": np.ascontiguousarray(next_states, dtype=np.float64),
        "reward_variant_code": int(variant_code),
        "penalty_decay": float(penalty_decay),
    }
    return np.asarray(_native.lasertag_discrete_reward_batch(**kwargs))


def _sample_mean_parity(py_rewards: np.ndarray, cpp_rewards: np.ndarray) -> Tuple[float, float]:
    """Return ``(|delta|, tolerance)`` for the sample-mean parity assertion."""
    n = py_rewards.shape[0]
    delta = abs(float(py_rewards.mean()) - float(cpp_rewards.mean()))
    tol = 3.0 * (float(py_rewards.std(ddof=0)) / math.sqrt(n)) + 1e-9
    return delta, tol


class TestLaserTagDiscreteRewardCppParity:
    """Sample-mean C++ vs Python parity across the 3 discrete reward variants."""

    @pytest.mark.parametrize(
        "rmt, penalty_decay",
        [
            (RewardModelType.CONSTANT_HAZARD_PENALTY, 0.0),
            (RewardModelType.ZERO_MEAN_HAZARD_SHOCK, 0.0),
            (RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY, _PENALTY_DECAY),
        ],
    )
    def test_cpp_matches_python_in_mean(self, rmt: RewardModelType, penalty_decay: float) -> None:
        """C++ reward-batch mean matches the Python reward-model mean within 3 SE.

        Purpose: Validates that the native ``lasertag_discrete_reward_batch``
            kernel reproduces the Python reward-model's expected reward for
            each :class:`RewardModelType` variant, including the stochastic
            HV and Decaying variants. The check is the sample-mean equality
            ``|py.mean() - cpp.mean()| < 3 * py.std()/sqrt(N) + 1e-9``,
            which is tight enough to fail with high probability if the
            kernel hard-codes the wrong variant and loose enough to pass
            when the kernel implements the right one.

        Given: A discrete :class:`LaserTagPOMDP` constructed with the
            chosen ``reward_model_type``, default walls, non-empty
            dangerous areas, ``dangerous_area_penalty = 5.0``, and (for
            the Decaying variant) ``penalty_decay = 1.5``. A batch of
            ``N = 2000`` non-terminal states is generated covering both
            danger-zone cells and clear cells; ``next_states`` is realised
            via ``env.sample_next_state_batch`` so the variant penalty
            path fires.
        When: ``env.reward_model.compute_reward_batch`` and the native
            ``_native.lasertag_discrete_reward_batch`` (with the Stage 2
            ``reward_variant_code`` / ``penalty_decay`` / ``next_states``
            kwargs) are evaluated on the same batch and action.
        Then: The two sample means agree within
            ``3 * py.std() / sqrt(N) + 1e-9``. Today this either raises
            ``TypeError`` (kernel lacks the kwargs) or fails the mean
            check on HV / Decaying (kernel hard-codes CONSTANT_HAZARD_PENALTY); both
            failure modes confirm the kernel is not yet variant-aware.

        Test type: integration
        """
        np.random.seed(42)
        env = _build_env(rmt)
        states = _build_states_with_danger_and_clear(env, _BATCH_SIZE)
        # Use a movement action (1 = South) so the wall / danger penalty
        # path fires; the tag action (4) only changes the base reward.
        action = 1
        next_states = env.sample_next_state_batch(states, action)

        py_rewards = env.reward_model.compute_reward_batch(states, action, next_states)
        cpp_rewards = _call_cpp_reward_batch(
            env,
            states,
            action,
            next_states,
            variant_code=_variant_code(rmt),
            penalty_decay=penalty_decay,
        )

        delta, tol = _sample_mean_parity(py_rewards, cpp_rewards)
        assert delta < tol, (
            f"C++/Python reward mean mismatch for variant {rmt.name}: "
            f"|py.mean - cpp.mean| = {delta:.6f} >= tol = {tol:.6f} "
            f"(py.mean={py_rewards.mean():.6f}, cpp.mean={cpp_rewards.mean():.6f}, "
            f"N={py_rewards.shape[0]})"
        )

    def test_tag_action_path_is_also_covered(self) -> None:
        """Tag-action (action=4) C++ mean matches Python mean for the CONSTANT_HAZARD_PENALTY variant.

        Purpose: Ensures the tag-action branch is exercised on top of the
            movement-action branch covered by the parametrised test, so a
            kernel that handles tag rewards differently from movement
            rewards is caught.

        Given: A CONSTANT_HAZARD_PENALTY discrete LaserTag env with default walls and
            dangerous areas, and a 2000-row state batch covering both
            danger-zone and clear cells.
        When: ``env.reward_model.compute_reward_batch`` and
            ``_native.lasertag_discrete_reward_batch`` are evaluated with
            action ``4`` (Tag) and the realised next-state batch.
        Then: The sample means agree within
            ``3 * py.std() / sqrt(N) + 1e-9``.

        Test type: integration
        """
        np.random.seed(42)
        env = _build_env(RewardModelType.CONSTANT_HAZARD_PENALTY)
        states = _build_states_with_danger_and_clear(env, _BATCH_SIZE)
        action = 4
        next_states = env.sample_next_state_batch(states, action)

        py_rewards = env.reward_model.compute_reward_batch(states, action, next_states)
        cpp_rewards = _call_cpp_reward_batch(
            env,
            states,
            action,
            next_states,
            variant_code=_variant_code(RewardModelType.CONSTANT_HAZARD_PENALTY),
            penalty_decay=0.0,
        )

        delta, tol = _sample_mean_parity(py_rewards, cpp_rewards)
        assert delta < tol, (
            f"C++/Python reward mean mismatch for variant CONSTANT_HAZARD_PENALTY (tag action): "
            f"|py.mean - cpp.mean| = {delta:.6f} >= tol = {tol:.6f} "
            f"(py.mean={py_rewards.mean():.6f}, cpp.mean={cpp_rewards.mean():.6f}, "
            f"N={py_rewards.shape[0]})"
        )
