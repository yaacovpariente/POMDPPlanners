# SPDX-License-Identifier: MIT

"""Cached-kernel parity tests for the CartPole POMDP.

These tests cover the per-action C++ kernel cache that the env now keeps
on hot paths. Mirrors the kernel-cache patterns landed on RockSample,
Pacman, ContinuousLaserTag, Push, and ContinuousLightDark.
"""

# pylint: disable=protected-access

import pickle
from typing import List

import numpy as np
import pytest

from POMDPPlanners.environments.cartpole_pomdp import _native
from POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp import CartPolePOMDP

_NOISE_COV = np.diag([0.1, 0.1, 0.1, 0.1])
_STATE_DIM = 4


def _make_env() -> CartPolePOMDP:
    return CartPolePOMDP(discount_factor=0.95, noise_cov=_NOISE_COV)


def _all_actions() -> List[int]:
    return [0, 1]


def _state(x: float, x_dot: float, theta: float, theta_dot: float) -> np.ndarray:
    return np.array([x, x_dot, theta, theta_dot], dtype=np.float64)


@pytest.fixture(name="env")
def _env_fixture() -> CartPolePOMDP:
    return _make_env()


# ---------------------------------------------------------------------------
# Cache reuse
# ---------------------------------------------------------------------------


def test_trans_kernel_cache_reuses_per_action(env: CartPolePOMDP) -> None:
    """Validate transition-kernel cache returns the same instance per action.

    Purpose: Validates the transition-kernel cache returns the same Python
        instance across repeat calls for the same action and a different
        instance for a different action.

    Given: A fresh CartPolePOMDP with empty caches.
    When: ``_get_trans_kernel`` is called twice for action 0 and once for
        action 1.
    Then: The two action-0 kernels are the same Python object; the
        action-1 kernel is a different object.

    Test type: unit
    """
    k_a = env._get_trans_kernel(0)
    k_b = env._get_trans_kernel(0)
    k_c = env._get_trans_kernel(1)
    assert id(k_a) == id(k_b)
    assert id(k_a) != id(k_c)


def test_obs_kernel_cache_reuses_per_action(env: CartPolePOMDP) -> None:
    """Validate observation-kernel cache returns the same instance per action.

    Purpose: Validates the observation-kernel cache returns the same
        Python instance across repeat calls for the same action.

    Given: A fresh CartPolePOMDP with empty caches.
    When: ``_get_obs_kernel`` is called twice for action 0 and once for
        action 1.
    Then: The two action-0 kernels are the same object; the action-1
        kernel is a different object.

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


def _fresh_trans_kernel(env: CartPolePOMDP, action: int) -> "_native.CartPoleTransitionCpp":
    return _native.CartPoleTransitionCpp(
        state=np.zeros(_STATE_DIM, dtype=np.float64),
        action=int(action),
        force_mag=env.force_mag,
        total_mass=env.total_mass,
        polemass_length=env.polemass_length,
        gravity=env.gravity,
        length=env.length,
        kinematics_integrator=env.kinematics_integrator,
        tau=env.tau,
        masspole=env.masspole,
        covariance=env._state_transition_dist.covariance,
    )


def _fresh_obs_kernel(env: CartPolePOMDP, action: int) -> "_native.CartPoleObservationCpp":
    return _native.CartPoleObservationCpp(
        next_state=np.zeros(_STATE_DIM, dtype=np.float64),
        action=int(action),
        covariance=env._obs_dist.covariance,
    )


def test_sample_next_state_parity_fresh_vs_cached(env: CartPolePOMDP) -> None:
    """Validate sample_next_state matches a freshly-built kernel after seeding.

    Purpose: Validates ``sample_next_state`` produces the same first
        sample as a freshly-constructed kernel after seeding the
        module-level RNG (each kernel uses the same default RNG; same
        seed + same deterministic mean + same covariance = same draw).

    Given: A live state and a freshly-built per-action kernel for each
        action.
    When: We seed the module-level RNG before each path and draw one
        sample.
    Then: Both paths produce identical 1-D state arrays.

    Test type: unit
    """
    state = _state(0.01, 0.0, 0.005, 0.0)
    for action in _all_actions():
        _native.set_seed(7)
        cached = env.sample_next_state(state=state, action=action)

        _native.set_seed(7)
        fresh = _fresh_trans_kernel(env, action)
        fresh.set_state(np.asarray(state, dtype=np.float64))
        fresh_sample = fresh.sample(1)[0]

        np.testing.assert_array_equal(cached, fresh_sample)


def test_transition_log_probability_parity_fresh_vs_cached(env: CartPolePOMDP) -> None:
    """Validate transition_log_probability matches a freshly-built kernel.

    Purpose: Validates ``transition_log_probability`` produces numerically
        identical values to a freshly-constructed kernel call.

    Given: A live state, a batch of candidate next states, and a
        freshly-built kernel for each action.
    When: Both paths are invoked.
    Then: The two log-probability arrays match within atol=1e-12.

    Test type: unit
    """
    state = _state(0.01, 0.0, 0.005, 0.0)
    next_states = np.array(
        [
            [0.011, 0.001, 0.0051, 0.0001],
            [0.012, 0.002, 0.0052, 0.0002],
            [-0.5, -0.5, 0.1, -0.1],
        ],
        dtype=np.float64,
    )
    for action in _all_actions():
        cached = env.transition_log_probability(state=state, action=action, next_states=next_states)

        fresh = _fresh_trans_kernel(env, action)
        fresh.set_state(np.asarray(state, dtype=np.float64))
        # The C++ kernel applies a symmetric ``kProbFloor = 1e-300``
        # floor inside ``probability`` so ``np.log(probs)`` returns
        # ``log(1e-300) ~= -690.776`` for impossible events instead
        # of ``-inf`` — matching what the env path returns. No
        # ``errstate`` guard needed because the floored prob is
        # strictly positive.
        fresh_probs = np.asarray(fresh.probability(next_states))
        fresh_logp = np.log(fresh_probs)

        np.testing.assert_allclose(cached, fresh_logp, atol=1e-12, rtol=0.0)


def test_observation_log_probability_parity_fresh_vs_cached(env: CartPolePOMDP) -> None:
    """Validate observation_log_probability matches a freshly-built kernel.

    Purpose: Validates ``observation_log_probability`` produces numerically
        identical values to a freshly-constructed kernel call.

    Given: A live next state, a list with a single observation candidate,
        and a freshly-built kernel for each action.
    When: Both paths are invoked.
    Then: The two log-probability arrays match within atol=1e-12.

    Test type: unit
    """
    next_state = _state(0.01, 0.0, 0.005, 0.0)
    observations = np.array(
        [
            [0.011, 0.001, 0.0051, 0.0001],
        ],
        dtype=np.float64,
    )
    for action in _all_actions():
        cached = env.observation_log_probability(
            next_state=next_state, action=action, observations=observations
        )

        fresh = _fresh_obs_kernel(env, action)
        fresh.set_next_state(np.asarray(next_state, dtype=np.float64))
        # The C++ kernel applies a symmetric ``kProbFloor = 1e-300``
        # floor inside ``probability`` so ``np.log(probs)`` returns
        # ``log(1e-300) ~= -690.776`` for impossible events instead
        # of ``-inf`` — matching what the env path returns. No
        # ``errstate`` guard needed because the floored prob is
        # strictly positive.
        fresh_probs = np.asarray(fresh.probability(observations))
        fresh_logp = np.log(fresh_probs)

        np.testing.assert_allclose(cached, fresh_logp, atol=1e-12, rtol=0.0)


# ---------------------------------------------------------------------------
# Pickle round-trip
# ---------------------------------------------------------------------------


def test_pickle_round_trip_preserves_sample_shapes_and_reward_batch() -> None:
    """Validate pickle round-trip preserves env behavior post-restore.

    Purpose: Validates pickling a CartPolePOMDP drops the kernel caches
        (which hold non-picklable pybind11 objects) and that the restored
        env still produces correct sample shapes and identical
        ``reward_batch`` outputs to the pre-pickle env.

    Given: An env that has populated its trans/obs kernel caches.
    When: We pickle and unpickle the env.
    Then: The restored caches are empty; ``sample_next_state`` returns
        a length-4 array; ``sample_next_state_batch`` returns shape
        ``(N, 4)``; ``reward_batch`` produces identical outputs to the
        pre-pickle env.

    Test type: unit
    """
    env = _make_env()
    state = _state(0.01, 0.0, 0.005, 0.0)
    states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.1, 0.05, 0.0],
            [-0.5, -0.1, -0.05, 0.0],
            [3.0, 0.0, 0.0, 0.0],  # past x_threshold (terminal)
            [0.0, 0.0, 1.0, 0.0],  # past theta_threshold (terminal)
        ],
        dtype=np.float64,
    )

    # Warm caches.
    for action in _all_actions():
        env.sample_next_state(state=state, action=action)
        env.sample_observation(next_state=state, action=action)
    assert env._trans_kernel_cache  # non-empty
    assert env._obs_kernel_cache  # non-empty

    blob = pickle.dumps(env)
    env2 = pickle.loads(blob)
    assert not env2._trans_kernel_cache
    assert not env2._obs_kernel_cache

    for action in _all_actions():
        sample = env2.sample_next_state(state=state, action=action)
        assert np.asarray(sample).shape == (4,)

        batch = env2.sample_next_state_batch(states, action)
        assert batch.shape == (states.shape[0], 4)

        rew_pre = env.reward_batch(states, action)
        rew_post = env2.reward_batch(states, action)
        np.testing.assert_array_equal(rew_pre, rew_post)


def test_getstate_round_trip_caches_are_empty() -> None:
    """Validate __getstate__ drops the kernel caches at serialization.

    Purpose: Validates that ``__getstate__`` returns a state dict with
        empty kernel caches, even when the live env has populated caches.

    Given: An env whose trans/obs kernel caches have been warmed.
    When: We call ``__getstate__``.
    Then: The returned dict's ``_trans_kernel_cache`` and
        ``_obs_kernel_cache`` entries are both empty dicts.

    Test type: unit
    """
    env = _make_env()
    state = _state(0.01, 0.0, 0.005, 0.0)
    for action in _all_actions():
        env.sample_next_state(state=state, action=action)
        env.sample_observation(next_state=state, action=action)
    assert env._trans_kernel_cache
    assert env._obs_kernel_cache

    pickled_state = env.__getstate__()
    assert not pickled_state["_trans_kernel_cache"]
    assert not pickled_state["_obs_kernel_cache"]
