# SPDX-License-Identifier: MIT

"""Cached-kernel parity tests for the RockSample POMDP.

These tests cover the per-action C++ kernel cache that the env now keeps
on hot paths plus the closed-form deterministic-reward shortcut used by
``_reward_batch_vectorized``. Mirrors the kernel-cache patterns landed
on ContinuousLaserTag / Push / ContinuousLightDark.
"""

# pylint: disable=protected-access

import pickle
from typing import List, Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.rock_sample_pomdp import _native
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
    RockSamplePOMDP,
    create_rock_sample_state,
)

_ROCK_POSITIONS: List[Tuple[int, int]] = [(0, 0), (2, 2), (3, 3)]
_MAP_SIZE: Tuple[int, int] = (5, 5)
_INIT_POS: Tuple[int, int] = (0, 0)
_SENSOR_EFFICIENCY: float = 10.0
_NUM_ROCKS: int = len(_ROCK_POSITIONS)
_STATE_DIM: int = 2 + _NUM_ROCKS


def _make_env(dangerous_areas=None) -> RockSamplePOMDP:
    return RockSamplePOMDP(
        map_size=_MAP_SIZE,
        rock_positions=list(_ROCK_POSITIONS),
        init_pos=_INIT_POS,
        sensor_efficiency=_SENSOR_EFFICIENCY,
        dangerous_areas=dangerous_areas,
    )


def _all_actions() -> List[int]:
    return list(range(5 + _NUM_ROCKS))


def _state(row: int, col: int, rocks: Tuple[bool, ...]) -> np.ndarray:
    assert len(rocks) == _NUM_ROCKS
    return create_rock_sample_state((row, col), rocks).astype(np.float64)


@pytest.fixture(name="env")
def _env_fixture() -> RockSamplePOMDP:
    return _make_env()


# ---------------------------------------------------------------------------
# Cache reuse
# ---------------------------------------------------------------------------


def test_trans_kernel_cache_reuses_per_action(env: RockSamplePOMDP) -> None:
    """Purpose: Validates the transition-kernel cache returns the same
    instance across repeat calls for the same action and a different
    instance for a different action.

    Given: A fresh RockSamplePOMDP with empty caches.
    When: ``_get_trans_kernel`` is called twice for action 0 and once for
        action 2.
    Then: The two action-0 kernels are the same Python object; the
        action-2 kernel is a different object.

    Test type: unit
    """
    k_a = env._get_trans_kernel(0)
    k_b = env._get_trans_kernel(0)
    k_c = env._get_trans_kernel(2)
    assert id(k_a) == id(k_b)
    assert id(k_a) != id(k_c)


def test_obs_kernel_cache_reuses_per_action(env: RockSamplePOMDP) -> None:
    """Purpose: Validates the observation-kernel cache returns the same
    instance across repeat calls for the same action.

    Given: A fresh RockSamplePOMDP with empty caches.
    When: ``_get_obs_kernel`` is called twice for action 5 (sense rock 0)
        and once for action 6 (sense rock 1).
    Then: The two action-5 kernels are the same object; the action-6
        kernel is a different object.

    Test type: unit
    """
    k_a = env._get_obs_kernel(5)
    k_b = env._get_obs_kernel(5)
    k_c = env._get_obs_kernel(6)
    assert id(k_a) == id(k_b)
    assert id(k_a) != id(k_c)


# ---------------------------------------------------------------------------
# Numeric parity: cached kernel vs freshly-built kernel
# ---------------------------------------------------------------------------


def _fresh_trans_kernel(env: RockSamplePOMDP, action: int) -> "_native.RockSampleTransitionCpp":
    return _native.RockSampleTransitionCpp(
        state=np.zeros(_STATE_DIM, dtype=np.float64),
        action=int(action),
        map_rows=env.map_size[0],
        map_cols=env.map_size[1],
        num_rocks=len(env.rock_positions),
        rock_positions=env._rock_positions_int32,
        sensor_efficiency=env.sensor_efficiency,
    )


def _fresh_obs_kernel(env: RockSamplePOMDP, action: int) -> "_native.RockSampleObservationCpp":
    return _native.RockSampleObservationCpp(
        next_state=np.zeros(_STATE_DIM, dtype=np.float64),
        action=int(action),
        map_rows=env.map_size[0],
        map_cols=env.map_size[1],
        num_rocks=len(env.rock_positions),
        rock_positions=env._rock_positions_int32,
        sensor_efficiency=env.sensor_efficiency,
    )


def test_sample_next_state_parity_fresh_vs_cached(env: RockSamplePOMDP) -> None:
    """Purpose: Validates ``sample_next_state`` produces identical states
    to a freshly constructed kernel for every action (transitions are
    deterministic).

    Given: A live state and a freshly-built per-action kernel.
    When: We call the cached path and the fresh kernel's ``sample(1)`` for
        each of the 5 + num_rocks actions.
    Then: Both paths produce identical 1-D state arrays.

    Test type: unit
    """
    state = _state(2, 2, (True, True, False))
    for action in _all_actions():
        cached = env.sample_next_state(state=state, action=action)
        fresh_kernel = _fresh_trans_kernel(env, action)
        fresh_kernel.set_state(np.asarray(state, dtype=float))
        fresh = fresh_kernel.sample(1)[0]
        np.testing.assert_array_equal(cached, fresh)


def test_observation_distribution_parity_fresh_vs_cached(env: RockSamplePOMDP) -> None:
    """Purpose: Validates the cached observation kernel produces the same
    distribution over codes as a freshly-built kernel for a sense action.

    Given: A live next state and a sense action (action == 5+rock_idx).
    When: We sample ``N`` observations through the cached kernel and ``N``
        through a freshly-built kernel, after seeding the module-level RNG.
    Then: The empirical frequencies match within a small tolerance.

    Test type: unit
    """
    next_state = _state(0, 1, (True, False, True))
    action = 5  # sense rock 0
    n_samples = 20000

    _native.set_seed(123)
    codes_cached = []
    for _ in range(n_samples):
        codes_cached.append(env.sample_observation(next_state=next_state, action=action))

    _native.set_seed(123)
    fresh = _fresh_obs_kernel(env, action)
    fresh.set_next_state(np.asarray(next_state, dtype=float))
    codes_fresh_int = fresh.sample(n_samples)
    code_to_str = ("none", "good", "bad")
    codes_fresh = [code_to_str[c] for c in codes_fresh_int]

    for label in ("none", "good", "bad"):
        f_cached = sum(c == label for c in codes_cached) / n_samples
        f_fresh = sum(c == label for c in codes_fresh) / n_samples
        assert abs(f_cached - f_fresh) < 5e-3, (label, f_cached, f_fresh)


def test_sample_next_state_batch_parity_fresh_vs_cached(env: RockSamplePOMDP) -> None:
    """Purpose: Validates ``sample_next_state_batch`` matches a freshly-
    built kernel's ``batch_sample`` row-for-row.

    Given: A 2-D batch of live and terminal-sentinel particles.
    When: We dispatch through the cached path and through a fresh kernel.
    Then: Both paths return identical arrays.

    Test type: unit
    """
    states = np.array(
        [
            [0, 0, 1, 0, 1],
            [2, 2, 1, 1, 0],
            [3, 3, 0, 0, 1],
            [-1, -1, 1, 0, 0],  # terminal sentinel
        ],
        dtype=np.float64,
    )
    for action in _all_actions():
        cached = env.sample_next_state_batch(states, action)
        fresh = _fresh_trans_kernel(env, action).batch_sample(states)
        np.testing.assert_array_equal(cached, fresh)


def test_observation_log_probability_per_state_parity(env: RockSamplePOMDP) -> None:
    """Purpose: Validates ``observation_log_probability_per_state`` matches
    a fresh kernel's ``batch_log_likelihood`` element-wise.

    Given: A 2-D batch of next states and a sense action with observation
        "good".
    When: Both the cached and fresh paths are invoked.
    Then: Their log-likelihood arrays are equal.

    Test type: unit
    """
    next_states = np.array(
        [
            [0, 0, 1, 0, 1],
            [2, 2, 1, 1, 0],
            [3, 3, 0, 0, 1],
            [-1, -1, 1, 0, 0],
        ],
        dtype=np.float64,
    )
    action = 5
    cached = env.observation_log_probability_per_state(
        next_states=next_states, action=action, observation="good"
    )
    fresh = _fresh_obs_kernel(env, action).batch_log_likelihood(
        next_particles=next_states, observation=1
    )
    np.testing.assert_array_equal(cached, np.asarray(fresh))


# ---------------------------------------------------------------------------
# Closed-form reward parity
# ---------------------------------------------------------------------------


def _reward_batch_via_native_transition(
    env: RockSamplePOMDP, states: np.ndarray, action: int
) -> np.ndarray:
    """Reference implementation: re-creates the pre-refactor reward path
    that calls ``sample_next_state_batch`` to get post-step robot positions
    for the dangerous-area check.

    Mirrors the scalar ``compute_reward`` / C++ ``reward_batch`` contract:
    only true exits (``robot_col == map_cols - 1`` under action=2) get the
    exit bonus — terminal-sentinel rows do not. Non-exiting East rows
    continue into the dangerous-area branch.
    """
    n = states.shape[0]
    next_states = env.sample_next_state_batch(states, action)
    rewards = np.full(n, env.step_penalty, dtype=np.float64)
    map_cols = env.map_size[1]

    exits_mask: np.ndarray | None = None
    if action == 2:
        exits_mask = states[:, 1].astype(int) == (map_cols - 1)
        rewards[exits_mask] += env.exit_reward

    if action == 0:
        robot_rows = states[:, 0].astype(int)
        robot_cols = states[:, 1].astype(int)
        for i, (rr, rc) in enumerate(env.rock_positions):
            at_rock = (robot_rows == rr) & (robot_cols == rc)
            if not np.any(at_rock):
                continue
            rock_slot = 2 + i
            rock_good = states[:, rock_slot] > 0.5
            rewards[at_rock & rock_good] += env.good_rock_reward
            rewards[at_rock & ~rock_good] += env.bad_rock_penalty

    if action >= 5:
        rewards += env.sensor_use_penalty

    if env.dangerous_areas:
        next_robot_rows = next_states[:, 0].astype(int)
        next_robot_cols = next_states[:, 1].astype(int)
        for j in range(n):
            if exits_mask is not None and exits_mask[j]:
                continue
            if env._is_in_dangerous_area((next_robot_rows[j], next_robot_cols[j])):
                rewards[j] += env.dangerous_area_penalty

    return rewards


def _make_reward_test_states() -> np.ndarray:
    return np.array(
        [
            [0, 0, 1, 0, 1],  # at rock 0
            [2, 2, 1, 1, 0],  # at rock 1
            [3, 3, 0, 0, 1],  # at rock 2
            [1, 1, 1, 0, 0],  # away from rocks
            [0, 4, 1, 0, 0],  # last column (east-exit)
            [4, 0, 0, 0, 0],  # last row
            [-1, -1, 0, 0, 0],  # terminal sentinel
        ],
        dtype=np.float64,
    )


def test_reward_batch_vectorized_parity_no_dangerous_areas() -> None:
    """Purpose: Validates the closed-form reward path matches the legacy
    kernel-driven path exactly when ``dangerous_areas`` is empty.

    Given: Two RockSamplePOMDP instances with the same config but the
        reference implementation re-runs the legacy kernel-driven logic.
    When: ``_reward_batch_vectorized`` is invoked for each action on a
        diverse batch of states.
    Then: The two reward arrays are equal.

    Test type: unit
    """
    env = _make_env()
    states = _make_reward_test_states()
    for action in _all_actions():
        new_path = env._reward_batch_vectorized(states, action)
        ref = _reward_batch_via_native_transition(env, states, action)
        np.testing.assert_array_equal(new_path, ref)


def test_reward_batch_vectorized_parity_with_dangerous_areas() -> None:
    """Purpose: Validates the closed-form reward path matches the legacy
    kernel-driven path exactly when ``dangerous_areas`` is non-empty.

    Given: A RockSamplePOMDP with two dangerous-area centres at (1, 1)
        and (3, 3); radius 1.0; penalty 5.0.
    When: ``_reward_batch_vectorized`` is invoked for each action on a
        diverse batch of states.
    Then: The two reward arrays are equal — the closed-form
        ``_closed_form_next_robot_pos`` produces the same dangerous-area
        membership as the native batch transition.

    Test type: unit
    """
    env = _make_env(dangerous_areas=[(1, 1), (3, 3)])
    states = _make_reward_test_states()
    for action in _all_actions():
        new_path = env._reward_batch_vectorized(states, action)
        ref = _reward_batch_via_native_transition(env, states, action)
        np.testing.assert_array_equal(new_path, ref)


# ---------------------------------------------------------------------------
# Pickle round-trip
# ---------------------------------------------------------------------------


def test_pickle_round_trip_drops_caches_and_preserves_parity() -> None:
    """Purpose: Validates pickling a RockSamplePOMDP drops the kernel
    caches (which hold non-picklable pybind11 objects) and that the
    restored env produces identical sample / reward outputs.

    Given: An env that has populated its trans/obs kernel caches.
    When: We pickle and unpickle the env.
    Then: The restored caches are empty; ``sample_next_state``,
        ``sample_next_state_batch``, and ``_reward_batch_vectorized``
        produce identical outputs to the pre-pickle env.

    Test type: unit
    """
    env = _make_env(dangerous_areas=[(1, 1)])
    state = _state(2, 2, (True, True, False))
    states = _make_reward_test_states()

    # Warm caches.
    for action in _all_actions():
        env.sample_next_state(state=state, action=action)
        env.sample_observation(next_state=state, action=action)
    assert env._trans_kernel_cache  # non-empty
    assert env._obs_kernel_cache  # non-empty

    blob = pickle.dumps(env)
    env2 = pickle.loads(blob)
    assert env2._trans_kernel_cache == {}
    assert env2._obs_kernel_cache == {}

    for action in _all_actions():
        a = env.sample_next_state(state=state, action=action)
        b = env2.sample_next_state(state=state, action=action)
        np.testing.assert_array_equal(a, b)

        a_batch = env.sample_next_state_batch(states, action)
        b_batch = env2.sample_next_state_batch(states, action)
        np.testing.assert_array_equal(a_batch, b_batch)

        a_rew = env._reward_batch_vectorized(states, action)
        b_rew = env2._reward_batch_vectorized(states, action)
        np.testing.assert_array_equal(a_rew, b_rew)
