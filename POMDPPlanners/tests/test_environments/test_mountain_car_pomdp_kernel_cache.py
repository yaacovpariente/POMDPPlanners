# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Cached-kernel parity tests for the MountainCar POMDP.

These tests cover the per-action C++ kernel cache that the env now keeps
on hot paths plus the ``set_state`` / ``set_next_state`` mutators that
let one kernel be reused across calls. Mirrors the kernel-cache patterns
landed for SafetyAnt and (in PR #118) PacMan / RockSample.
"""

# pylint: disable=protected-access

import pickle
from typing import List

import numpy as np
import pytest

from POMDPPlanners.environments.mountain_car_pomdp import _native
from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp import (
    MountainCarPOMDP,
)


_DISCOUNT_FACTOR: float = 0.99
_ACTIONS: List[int] = [-1, 0, 1]


def _make_env() -> MountainCarPOMDP:
    return MountainCarPOMDP(discount_factor=_DISCOUNT_FACTOR)


def _fresh_trans_kernel(env: MountainCarPOMDP, action: int) -> "_native.MountainCarTransitionCpp":
    return _native.MountainCarTransitionCpp(
        state=np.zeros(2, dtype=np.float64),
        action=int(action),
        power=env.power,
        gravity=env.gravity,
        max_speed=env.max_speed,
        min_position=env.min_position,
        max_position=env.max_position,
        covariance=env._state_transition_dist.covariance,
    )


def _fresh_obs_kernel(env: MountainCarPOMDP, action: int) -> "_native.MountainCarObservationCpp":
    return _native.MountainCarObservationCpp(
        next_state=np.zeros(2, dtype=np.float64),
        action=int(action),
        covariance=env._obs_dist.covariance,
    )


@pytest.fixture(name="env")
def _env_fixture() -> MountainCarPOMDP:
    return _make_env()


# ---------------------------------------------------------------------------
# Cache reuse
# ---------------------------------------------------------------------------


def test_trans_kernel_cache_reuses_per_action(env: MountainCarPOMDP) -> None:
    """Validate the transition-kernel cache reuses one kernel per action.

    Purpose: Confirms ``_get_trans_kernel`` returns the same Python object
        on repeat calls for the same action and a different object for a
        different action.

    Given: A fresh MountainCarPOMDP with empty caches.
    When: ``_get_trans_kernel`` is called twice for action -1 and once for
        action +1.
    Then: The two action-(-1) kernels are the same object (id-equal); the
        action-(+1) kernel is a different object.

    Test type: unit
    """
    k_a = env._get_trans_kernel(-1)
    k_b = env._get_trans_kernel(-1)
    k_c = env._get_trans_kernel(1)
    assert id(k_a) == id(k_b)
    assert id(k_a) != id(k_c)


def test_obs_kernel_cache_reuses_per_action(env: MountainCarPOMDP) -> None:
    """Validate the observation-kernel cache reuses one kernel per action.

    Purpose: Confirms ``_get_obs_kernel`` returns the same Python object
        on repeat calls for the same action and a different object for a
        different action.

    Given: A fresh MountainCarPOMDP with empty caches.
    When: ``_get_obs_kernel`` is called twice for action 0 and once for
        action +1.
    Then: The two action-0 kernels are the same object (id-equal); the
        action-(+1) kernel is a different object.

    Test type: unit
    """
    k_a = env._get_obs_kernel(0)
    k_b = env._get_obs_kernel(0)
    k_c = env._get_obs_kernel(1)
    assert id(k_a) == id(k_b)
    assert id(k_a) != id(k_c)


# ---------------------------------------------------------------------------
# Numeric parity: cached kernel vs freshly-built kernel
# ---------------------------------------------------------------------------


def test_sample_next_state_parity_fresh_vs_cached(env: MountainCarPOMDP) -> None:
    """Validate ``sample_next_state`` matches a freshly-built kernel.

    Purpose: Confirms the cached path produces identical samples to a
        fresh kernel under a fixed RNG seed for every discrete action.

    Given: A live state, a discrete action, and a seeded module RNG.
    When: We call the cached ``sample_next_state`` and a fresh kernel's
        ``sample(1)`` after ``set_state`` for each of the three actions.
    Then: Both paths produce identical 1-D state arrays for each action.

    Test type: unit
    """
    state = np.array([-0.5, 0.01], dtype=np.float64)
    for action in _ACTIONS:
        _native.set_seed(123)
        cached = env.sample_next_state(state=state, action=action)
        _native.set_seed(123)
        fresh_kernel = _fresh_trans_kernel(env, action)
        fresh_kernel.set_state(state)
        fresh = fresh_kernel.sample(1)[0]
        np.testing.assert_array_equal(cached, fresh)


def test_transition_log_probability_parity_fresh_vs_cached(env: MountainCarPOMDP) -> None:
    """Validate ``transition_log_probability`` matches a freshly-built kernel.

    Purpose: Confirms the cached path produces identical log-probabilities
        to a fresh kernel for every discrete action across a small batch
        of candidate next-states.

    Given: A live state, three candidate next-states, and each action.
    When: We call cached ``transition_log_probability`` and a fresh
        kernel's ``probability`` (then take ``log`` with the same epsilon).
    Then: The two log-probability arrays are equal within ``atol=1e-12``.

    Test type: unit
    """
    state = np.array([-0.5, 0.01], dtype=np.float64)
    next_states = np.array(
        [
            [-0.49, 0.011],
            [-0.51, 0.0099],
            [-0.5, 0.01],
        ],
        dtype=np.float64,
    )
    for action in _ACTIONS:
        cached = env.transition_log_probability(state=state, action=action, next_states=next_states)
        fresh = _fresh_trans_kernel(env, action)
        fresh.set_state(state)
        ref_probs = np.asarray(fresh.probability(next_states))
        ref = np.log(ref_probs + 1e-300)
        np.testing.assert_allclose(cached, ref, atol=1e-12, rtol=0.0)


def test_observation_log_probability_parity_fresh_vs_cached(env: MountainCarPOMDP) -> None:
    """Validate ``observation_log_probability`` matches a freshly-built kernel.

    Purpose: Confirms the cached path produces identical log-probabilities
        to a fresh kernel for every discrete action across a small batch
        of candidate observations.

    Given: A live next-state, three candidate observations, and each action.
    When: We call cached ``observation_log_probability`` and a fresh
        kernel's ``probability`` (then take ``log`` with the same epsilon).
    Then: The two log-probability arrays are equal within ``atol=1e-12``.

    Test type: unit
    """
    next_state = np.array([-0.4, 0.0], dtype=np.float64)
    observations = np.array(
        [
            [-0.41, 0.0],
            [-0.4, 0.005],
            [-0.39, -0.002],
        ],
        dtype=np.float64,
    )
    for action in _ACTIONS:
        cached = env.observation_log_probability(
            next_state=next_state, action=action, observations=observations
        )
        fresh = _fresh_obs_kernel(env, action)
        fresh.set_next_state(next_state)
        ref_probs = np.asarray(fresh.probability(observations))
        ref = np.log(ref_probs + 1e-300)
        np.testing.assert_allclose(cached, ref, atol=1e-12, rtol=0.0)


def test_sample_next_state_batch_parity_fresh_vs_cached(env: MountainCarPOMDP) -> None:
    """Validate ``sample_next_state_batch`` matches a freshly-built kernel.

    Purpose: Confirms the cached path's ``batch_sample`` produces identical
        outputs to a fresh kernel under a fixed RNG seed.

    Given: A 2-D batch of live states (4 particles) and each action.
    When: We dispatch through the cached path and through a fresh kernel,
        each preceded by ``_native.set_seed(123)``.
    Then: Both paths return identical arrays of shape ``(4, 2)`` for every
        action.

    Test type: unit
    """
    states = np.array(
        [
            [-0.5, 0.0],
            [-0.4, 0.01],
            [-0.6, -0.005],
            [-0.55, 0.02],
        ],
        dtype=np.float64,
    )
    for action in _ACTIONS:
        _native.set_seed(123)
        cached = env.sample_next_state_batch(states, action)
        _native.set_seed(123)
        fresh = np.asarray(_fresh_trans_kernel(env, action).batch_sample(states))
        np.testing.assert_array_equal(cached, fresh)


def test_observation_log_probability_per_state_parity(env: MountainCarPOMDP) -> None:
    """Validate ``observation_log_probability_per_state`` matches a fresh kernel.

    Purpose: Confirms cached ``batch_log_likelihood`` matches a fresh
        kernel element-wise within ``atol=1e-12``.

    Given: A 2-D batch of next-states (4 particles), a single observation,
        and each action.
    When: Both the cached and fresh paths are invoked.
    Then: The two log-likelihood arrays agree within ``atol=1e-12``.

    Test type: unit
    """
    next_states = np.array(
        [
            [-0.5, 0.0],
            [-0.4, 0.01],
            [-0.6, -0.005],
            [-0.55, 0.02],
        ],
        dtype=np.float64,
    )
    observation = np.array([-0.45, 0.005], dtype=np.float64)
    for action in _ACTIONS:
        cached = env.observation_log_probability_per_state(
            next_states=next_states, action=action, observation=observation
        )
        fresh = np.asarray(
            _fresh_obs_kernel(env, action).batch_log_likelihood(
                next_particles=next_states, observation=observation
            )
        )
        np.testing.assert_allclose(cached, fresh, atol=1e-12, rtol=0.0)


# ---------------------------------------------------------------------------
# Pickle round-trip
# ---------------------------------------------------------------------------


def test_pickle_round_trip_drops_caches() -> None:
    """Validate pickling drops the kernel caches on the receiver.

    Purpose: Confirms ``__getstate__`` does not serialise the pybind11
        kernels (which are not picklable) and that ``__setstate__`` rebuilds
        empty caches lazily on the receiver.

    Given: An env that has populated both kernel caches.
    When: We pickle then unpickle the env.
    Then: The restored env's caches are empty dicts.

    Test type: unit
    """
    env = _make_env()
    state = np.array([-0.5, 0.01], dtype=np.float64)
    for action in _ACTIONS:
        env.sample_next_state(state=state, action=action)
        env.sample_observation(next_state=state, action=action)
    assert env._trans_kernel_cache  # non-empty
    assert env._obs_kernel_cache  # non-empty

    blob = pickle.dumps(env)
    env_restored = pickle.loads(blob)
    assert env_restored._trans_kernel_cache == {}
    assert env_restored._obs_kernel_cache == {}


def test_pickle_round_trip_preserves_shapes_and_reward_batch_parity() -> None:
    """Validate post-pickle env produces identical ``reward_batch`` outputs.

    Purpose: Confirms the restored env continues to compute the correct
        ``reward_batch``, ``sample_next_state_batch`` shapes, and
        ``transition_log_probability`` values after pickle round-trip.

    Given: An env with warm caches and a 2-D batch of states.
    When: We pickle / unpickle and call ``reward_batch``,
        ``sample_next_state_batch``, and ``transition_log_probability``
        on both the original and the restored env.
    Then: ``reward_batch`` outputs are equal; the batch-sample arrays have
        the same shape (transitions are stochastic so values differ);
        ``transition_log_probability`` (deterministic given inputs) outputs
        are equal within ``atol=1e-12``.

    Test type: unit
    """
    env = _make_env()
    state = np.array([-0.5, 0.01], dtype=np.float64)
    states = np.array(
        [
            [-0.5, 0.0],
            [-0.4, 0.01],
            [0.6, 0.0],  # at goal: reward should be 0.0
            [-0.55, 0.02],
        ],
        dtype=np.float64,
    )
    next_states = np.array(
        [
            [-0.49, 0.011],
            [-0.51, 0.0099],
            [-0.5, 0.01],
        ],
        dtype=np.float64,
    )

    # Warm caches.
    for action in _ACTIONS:
        env.sample_next_state(state=state, action=action)
        env.sample_observation(next_state=state, action=action)

    blob = pickle.dumps(env)
    env_restored = pickle.loads(blob)

    for action in _ACTIONS:
        rew_a = env.reward_batch(states, action)
        rew_b = env_restored.reward_batch(states, action)
        np.testing.assert_array_equal(rew_a, rew_b)

        a_batch = env.sample_next_state_batch(states, action)
        b_batch = env_restored.sample_next_state_batch(states, action)
        assert a_batch.shape == b_batch.shape == (states.shape[0], 2)

        lp_a = env.transition_log_probability(state=state, action=action, next_states=next_states)
        lp_b = env_restored.transition_log_probability(
            state=state, action=action, next_states=next_states
        )
        np.testing.assert_allclose(lp_a, lp_b, atol=1e-12, rtol=0.0)
