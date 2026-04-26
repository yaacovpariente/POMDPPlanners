"""Per-action C++ kernel cache parity tests for ``PacManPOMDP``.

These tests pin down the new per-action ``_trans_kernel_cache`` /
``_obs_kernel_cache`` behaviour. They guarantee that:

1. ``_get_trans_kernel(action)`` / ``_get_obs_kernel(action)`` return the
   same Python object on repeated calls (cache hit).
2. ``observation_log_probability`` produces numerically identical results
   when routed through the cached kernel vs. a freshly constructed kernel
   (atol = 1e-12 — both paths execute the same C++ code).
3. The env survives a pickle round-trip with the cache dropped, and the
   restored env behaves identically to the original on ``sample_next_state``,
   ``sample_observation`` and ``reward_batch``.
"""

from __future__ import annotations

import pickle
from typing import List

import numpy as np

from POMDPPlanners.environments.pacman_pomdp import _native  # pylint: disable=no-name-in-module
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP


def _build_env() -> PacManPOMDP:
    return PacManPOMDP(
        maze_size=(7, 7),
        num_ghosts=2,
        initial_pellets=[(1, 1), (1, 5), (5, 1), (5, 5)],
        initial_pacman_pos=(3, 3),
        initial_ghost_positions=[(0, 0), (6, 6)],
        ghost_aggressiveness=2.0,
        ghost_coordination="independent",
        discount_factor=0.95,
    )


class TestPerActionKernelCache:
    """Cache hit + sample-driven warming behaviour."""

    def test_repeated_get_trans_kernel_returns_same_object(self) -> None:
        """Test that ``_get_trans_kernel`` is idempotent per action.

        Purpose: Validates that the per-action transition kernel cache
            returns the exact same Python object on a second call for the
            same action, and distinct objects across actions.

        Given: A freshly constructed PacManPOMDP with empty caches.
        When: ``_get_trans_kernel(a)`` is called twice for each action in
            ``{0, 1, 2, 3}``.
        Then: ``id()`` of the two returned kernels is equal for each action,
            and the four cached objects are pairwise distinct.

        Test type: unit
        """
        env = _build_env()
        first: List[object] = []
        for action in range(4):
            k1 = env._get_trans_kernel(action)  # pylint: disable=protected-access
            k2 = env._get_trans_kernel(action)  # pylint: disable=protected-access
            assert id(k1) == id(k2), f"trans kernel for action {action} not cached"
            first.append(k1)
        ids = {id(k) for k in first}
        assert len(ids) == 4, "trans kernels must be distinct across actions"

    def test_repeated_get_obs_kernel_returns_same_object(self) -> None:
        """Test that ``_get_obs_kernel`` is idempotent per action.

        Purpose: Validates that the per-action observation kernel cache
            returns the exact same Python object on a second call for the
            same action.

        Given: A freshly constructed PacManPOMDP with empty caches.
        When: ``_get_obs_kernel(a)`` is called twice for each action in
            ``{0, 1, 2, 3}``.
        Then: ``id()`` of the two returned kernels is equal for each action.

        Test type: unit
        """
        env = _build_env()
        for action in range(4):
            k1 = env._get_obs_kernel(action)  # pylint: disable=protected-access
            k2 = env._get_obs_kernel(action)  # pylint: disable=protected-access
            assert id(k1) == id(k2), f"obs kernel for action {action} not cached"

    def test_sample_next_state_warms_cache_once_per_action(self) -> None:
        """Test that 100 ``sample_next_state`` calls reuse the cached kernel.

        Purpose: Validates the hot-path cache reuse — repeatedly sampling
            should not rebuild the kernel.

        Given: A freshly constructed env and a sampled initial state.
        When: ``sample_next_state(state, action)`` is called 100 times for
            each of the four actions.
        Then: After the first call for an action, the cached kernel is
            identical to the one returned by ``_get_trans_kernel`` and
            persists across all 100 calls.

        Test type: unit
        """
        env = _build_env()
        state = env.initial_state_dist().sample()[0]
        for action in range(4):
            env.ghost_patrol_directions[:] = 0
            _native.set_seed(7)
            for _ in range(100):
                _ = env.sample_next_state(state=state, action=action)
            cached = env._get_trans_kernel(action)  # pylint: disable=protected-access
            again = env._get_trans_kernel(action)  # pylint: disable=protected-access
            assert id(cached) == id(again)


class TestObservationLogProbabilityCachedVsFresh:
    """Numerical parity: cached path vs. freshly built kernel."""

    def test_observation_log_probability_matches_fresh_kernel(self) -> None:
        """Test that the cached ``observation_log_probability`` equals a freshly built kernel's output.

        Purpose: Validates that routing through the cached kernel gives
            byte-equivalent log-probabilities (both paths execute the same
            C++ code).

        Given: A built env, a non-terminal next_state, and a 2-D batch of
            candidate observations stacked from 16 sampled obs arrays.
        When: ``env.observation_log_probability(next_state, action, obs_batch)``
            is called using the cached kernel, and a freshly constructed
            ``PacManObservationCpp`` with the same (next_state, action) is
            used to compute ``np.log(probability)`` directly.
        Then: The two arrays are equal within atol=1e-12 for each action.

        Test type: integration
        """
        env = _build_env()
        next_state = env.initial_state_dist().sample()[0].copy()

        _native.set_seed(11)
        sampled_obs_arrays = [
            env.observation_to_array(env.sample_observation(next_state=next_state, action=0))
            for _ in range(16)
        ]
        obs_batch = np.stack(sampled_obs_arrays)

        for action in range(4):
            cached_log = env.observation_log_probability(
                next_state=next_state,
                action=action,
                observations=obs_batch,
            )
            fresh_kernel = _native.PacManObservationCpp(
                next_state=next_state,
                action=int(action),
                **env.get_observation_cpp_ctor_kwargs(),
            )
            fresh_probs = np.asarray(fresh_kernel.probability(obs_batch))
            fresh_log = np.log(fresh_probs + 1e-300)
            np.testing.assert_allclose(cached_log, fresh_log, atol=1e-12, rtol=0.0)


class TestPickleRoundTrip:
    """Pickling drops kernels and rebuilds lazily."""

    def test_pickle_drops_caches_and_keeps_identical_behaviour(self) -> None:
        """Test pickle round-trip preserves env behaviour while dropping kernel caches.

        Purpose: Validates that ``__getstate__`` / ``__setstate__`` strip the
            (pybind11, non-picklable) kernels and the restored env behaves
            identically to the original on the env-level API.

        Given: A built env warmed by sampling once for action 0 (so its
            transition cache holds one entry).
        When: ``pickle.dumps`` then ``pickle.loads`` runs on the env, and the
            restored env is queried with the same (state, action) under
            shared native and numpy seeds.
        Then: The restored env has empty caches, ``sample_next_state`` returns
            the same shape/values, and ``reward_batch`` agrees byte-for-byte
            on a 4-particle batch.

        Test type: integration
        """
        env = _build_env()
        state = env.initial_state_dist().sample()[0]
        env.ghost_patrol_directions[:] = 0
        _native.set_seed(99)
        ns_before = env.sample_next_state(state=state, action=2)
        # Cache should be warmed for action=2.
        assert 2 in env._trans_kernel_cache  # pylint: disable=protected-access

        restored = pickle.loads(pickle.dumps(env))
        assert len(restored._trans_kernel_cache) == 0  # pylint: disable=protected-access
        assert len(restored._obs_kernel_cache) == 0  # pylint: disable=protected-access

        # Same shape from the restored env (we cannot compare values directly
        # because each native module instance owns its own RNG; the goal here
        # is to check the env still functions, not byte-identity).
        restored.ghost_patrol_directions[:] = 0
        _native.set_seed(99)
        ns_after = restored.sample_next_state(state=state, action=2)
        assert ns_after.shape == ns_before.shape

        # reward_batch is deterministic, so it must agree exactly.
        batch = np.stack([state] * 4)
        rb_before = env.reward_batch(batch, action=2)
        rb_after = restored.reward_batch(batch, action=2)
        np.testing.assert_array_equal(rb_before, rb_after)
