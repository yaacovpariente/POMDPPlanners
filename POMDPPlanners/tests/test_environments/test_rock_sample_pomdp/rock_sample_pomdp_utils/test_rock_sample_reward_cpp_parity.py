# SPDX-License-Identifier: MIT

"""C++ <-> Python reward parity tests for the RockSample POMDP.

These tests assert that a (yet-to-be-added) standalone C++ reward kernel
``_native.reward_batch`` reproduces the per-variant Python reward model
(``RockSampleRewardModel`` and subclasses) on identical
``(state, action, next_state)`` triples.

Today no standalone C++ reward kernel exists: reward is inlined inside
``_native.simulate_rollout_discrete`` and that path does not include the
dangerous-area term. These tests are therefore expected to FAIL until a
Stage 2 agent adds ``_native.reward_batch``. They form the TDD red step
for that work.

The contract being asserted here is:

* ``_native.reward_batch`` is callable via keyword arguments listing all
  environment scoring parameters, plus ``reward_variant_code`` (int) and
  ``penalty_decay`` (float) selecting one of the three reward variants:

  * ``reward_variant_code=0`` -> CONSTANT_HAZARD_PENALTY
  * ``reward_variant_code=1`` -> ZERO_MEAN_HAZARD_SHOCK
  * ``reward_variant_code=2`` -> DISTANCE_DECAYED_HAZARD_PENALTY

* For each variant, the sample-mean of the C++ rewards over a seeded
  ``N=2000`` batch matches the sample-mean of the Python reward model
  within a 3-sigma confidence band of the Python sample standard error.
"""

# pylint: disable=protected-access  # Tests reach into native module for parity checks.

from typing import List, Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.rock_sample_pomdp import _native
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
    RewardModelType,
    RockSamplePOMDP,
)


_MAP_SIZE: Tuple[int, int] = (7, 7)
_ROCK_POSITIONS: List[Tuple[int, int]] = [(1, 1), (3, 3), (5, 5), (6, 2)]
_INIT_POS: Tuple[int, int] = (0, 0)
_DANGEROUS_AREAS: List[Tuple[int, int]] = [(2, 2), (4, 4)]
_DANGEROUS_AREA_RADIUS: float = 1.0
_DANGEROUS_AREA_PENALTY: float = -5.0
_DANGEROUS_AREA_HIT_PROBABILITY: float = 1.0
_STEP_PENALTY: float = -0.1
_BAD_ROCK_PENALTY: float = -10.0
_GOOD_ROCK_REWARD: float = 10.0
_SENSOR_USE_PENALTY: float = -0.5
_EXIT_REWARD: float = 10.0
_PENALTY_DECAY_DECAYING_VARIANT: float = 1.5
_N: int = 2000
_NUM_ROCKS: int = len(_ROCK_POSITIONS)


def _make_env(reward_model_type: RewardModelType, penalty_decay: float) -> RockSamplePOMDP:
    return RockSamplePOMDP(
        map_size=_MAP_SIZE,
        rock_positions=list(_ROCK_POSITIONS),
        init_pos=_INIT_POS,
        bad_rock_penalty=_BAD_ROCK_PENALTY,
        good_rock_reward=_GOOD_ROCK_REWARD,
        step_penalty=_STEP_PENALTY,
        sensor_use_penalty=_SENSOR_USE_PENALTY,
        exit_reward=_EXIT_REWARD,
        dangerous_areas=list(_DANGEROUS_AREAS),
        dangerous_area_radius=_DANGEROUS_AREA_RADIUS,
        dangerous_area_penalty=_DANGEROUS_AREA_PENALTY,
        dangerous_area_hit_probability=_DANGEROUS_AREA_HIT_PROBABILITY,
        reward_model_type=reward_model_type,
        penalty_decay=penalty_decay,
    )


def _build_random_states(env: RockSamplePOMDP, n: int) -> np.ndarray:
    rows = np.random.randint(0, _MAP_SIZE[0], size=n)
    cols = np.random.randint(0, _MAP_SIZE[1], size=n)
    rocks = np.random.randint(0, 2, size=(n, _NUM_ROCKS)).astype(np.float64)
    state_dim = 2 + _NUM_ROCKS
    states = np.empty((n, state_dim), dtype=np.float64)
    states[:, 0] = rows
    states[:, 1] = cols
    states[:, 2:] = rocks
    assert states.shape[1] == 2 + len(env.rock_positions)
    return states


def _mixed_actions_per_row(env: RockSamplePOMDP, n: int) -> np.ndarray:
    # We want exposure to sample / move / sensor / exit branches across
    # the batch, but ``compute_reward_batch`` is single-action so we
    # bucket the rows by action and parity-test each bucket.
    n_actions = len(env.get_actions())
    return np.random.randint(0, n_actions, size=n).astype(np.int32)


def _native_reward_batch_kwargs(
    states: np.ndarray,
    action: int,
    next_states: np.ndarray,
    reward_variant_code: int,
    penalty_decay: float,
) -> dict:
    dangerous_areas_arr = np.asarray(_DANGEROUS_AREAS, dtype=np.float64)
    rock_positions_arr = np.asarray(_ROCK_POSITIONS, dtype=np.int32)
    return {
        "states": np.ascontiguousarray(states, dtype=np.float64),
        "action": int(action),
        "next_states": np.ascontiguousarray(next_states, dtype=np.float64),
        "map_rows": int(_MAP_SIZE[0]),
        "map_cols": int(_MAP_SIZE[1]),
        "rock_positions": rock_positions_arr,
        "step_penalty": float(_STEP_PENALTY),
        "bad_rock_penalty": float(_BAD_ROCK_PENALTY),
        "good_rock_reward": float(_GOOD_ROCK_REWARD),
        "sensor_use_penalty": float(_SENSOR_USE_PENALTY),
        "exit_reward": float(_EXIT_REWARD),
        "dangerous_areas": dangerous_areas_arr,
        "dangerous_area_radius": float(_DANGEROUS_AREA_RADIUS),
        "dangerous_area_penalty": float(_DANGEROUS_AREA_PENALTY),
        "dangerous_area_hit_probability": float(_DANGEROUS_AREA_HIT_PROBABILITY),
        "reward_variant_code": int(reward_variant_code),
        "penalty_decay": float(penalty_decay),
    }


def _collect_cpp_rewards(
    env: RockSamplePOMDP,
    states: np.ndarray,
    actions: np.ndarray,
    reward_variant_code: int,
    penalty_decay: float,
) -> np.ndarray:
    rewards = np.empty(states.shape[0], dtype=np.float64)
    for action in np.unique(actions):
        mask = actions == action
        bucket_states = states[mask]
        bucket_next = env.sample_next_state_batch(bucket_states, int(action))
        kwargs = _native_reward_batch_kwargs(
            states=bucket_states,
            action=int(action),
            next_states=bucket_next,
            reward_variant_code=reward_variant_code,
            penalty_decay=penalty_decay,
        )
        # ``_native.reward_batch`` does not exist yet (Stage 2 work). The
        # ``getattr`` plus call surfaces a clear AttributeError today and
        # a typed call once the kernel lands.
        kernel = getattr(_native, "reward_batch")
        rewards[mask] = np.asarray(kernel(**kwargs), dtype=np.float64)
    return rewards


def _collect_py_rewards(
    env: RockSamplePOMDP,
    states: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    rewards = np.empty(states.shape[0], dtype=np.float64)
    for action in np.unique(actions):
        mask = actions == action
        bucket_states = states[mask]
        bucket_next = env.sample_next_state_batch(bucket_states, int(action))
        rewards[mask] = np.asarray(
            env.reward_model.compute_reward_batch(
                bucket_states, int(action), next_states=bucket_next
            ),
            dtype=np.float64,
        )
    return rewards


@pytest.mark.parametrize(
    ("variant", "reward_variant_code", "penalty_decay"),
    [
        (RewardModelType.CONSTANT_HAZARD_PENALTY, 0, 0.0),
        (RewardModelType.ZERO_MEAN_HAZARD_SHOCK, 1, 0.0),
        (RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY, 2, _PENALTY_DECAY_DECAYING_VARIANT),
    ],
    ids=["constant_hazard_penalty", "zero_mean_hazard_shock", "distance_decayed_hazard_penalty"],
)
class TestRockSampleRewardCppParity:
    """C++ vs Python reward-mean parity across all RockSample reward variants."""

    def test_cpp_batch_mean_matches_python_within_3_sigma(
        self,
        variant: RewardModelType,
        reward_variant_code: int,
        penalty_decay: float,
    ) -> None:
        """C++ ``reward_batch`` sample-mean matches Python within 3 sigma.

        Purpose: Validates that the (Stage-2) standalone C++ reward kernel
            ``_native.reward_batch`` reproduces the per-variant Python
            reward model in expectation over a seeded batch of
            ``(state, action, next_state)`` triples that exercises sample,
            sensor, move, and exit branches.

        Given: A ``RockSamplePOMDP`` configured with non-empty
            ``dangerous_areas`` and the variant-appropriate
            ``reward_model_type``. ``N=2000`` random states are generated
            under ``np.random.seed(42)``, with one random action per row
            spanning all action codes (sample / move / sensor / exit).
            Next states come from ``env.sample_next_state_batch`` to match
            the realised trajectory the dangerous-area term fires against.
        When: Python rewards come from
            ``env.reward_model.compute_reward_batch``; C++ rewards come
            from ``_native.reward_batch`` called with keyword arguments
            including ``reward_variant_code`` and ``penalty_decay``.
        Then: ``|py.mean() - cpp.mean()| < 3 * (py.std() / sqrt(N)) + 1e-9``.

        Test type: integration
        """
        np.random.seed(42)
        env = _make_env(reward_model_type=variant, penalty_decay=penalty_decay)
        states = _build_random_states(env, _N)
        actions = _mixed_actions_per_row(env, _N)

        py_rewards = _collect_py_rewards(env, states, actions)
        cpp_rewards = _collect_cpp_rewards(
            env=env,
            states=states,
            actions=actions,
            reward_variant_code=reward_variant_code,
            penalty_decay=penalty_decay,
        )

        py_mean = float(py_rewards.mean())
        cpp_mean = float(cpp_rewards.mean())
        py_sem = float(py_rewards.std() / np.sqrt(_N))
        tolerance = 3.0 * py_sem + 1e-9
        assert abs(py_mean - cpp_mean) < tolerance, (
            f"variant={variant.value}: |py.mean - cpp.mean| = "
            f"{abs(py_mean - cpp_mean):.6g} not < tol = {tolerance:.6g} "
            f"(py.mean={py_mean:.6g}, cpp.mean={cpp_mean:.6g}, py.sem={py_sem:.6g})"
        )
