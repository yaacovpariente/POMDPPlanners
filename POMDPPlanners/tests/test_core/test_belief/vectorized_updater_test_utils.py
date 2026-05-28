# SPDX-License-Identifier: MIT

"""Shared test utilities for comparing vectorized vs per-particle belief updates.

This module provides reusable assertion functions that verify vectorized
batch operations (batch_transition, batch_observation_log_likelihood) produce
the same results as the equivalent per-particle loop through the environment's
state_transition_model and observation_model.

Functions:
    assert_batch_transition_matches_loop: Compare batch_transition vs per-particle transitions.
    assert_batch_obs_log_likelihood_matches_loop: Compare batch log-likelihoods vs per-particle.
"""

from typing import Any, Callable, Optional

import numpy as np

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)


def assert_batch_transition_matches_loop(
    updater: VectorizedParticleBeliefUpdater,
    particles: np.ndarray,
    action: Any,
    per_particle_transition_fn: Callable[[np.ndarray, Any], np.ndarray],
    atol: float = 1e-10,
    seed: Optional[int] = None,
    seed_fn: Optional[Callable[[int], None]] = None,
    err_msg: str = "",
) -> None:
    """Assert that batch_transition matches a per-particle transition loop.

    Args:
        updater: Vectorized updater to test.
        particles: Particle array of shape (N, d).
        action: Action to apply.
        per_particle_transition_fn: Callable(particle_1d, action) -> next_state_1d
            that wraps the environment's per-particle transition logic.
        atol: Absolute tolerance for comparison.
        seed: If provided, the RNG is seeded before each path so stochastic
            transitions consume the same random sequence.
        seed_fn: Callable used to seed the RNG. Defaults to ``np.random.seed``.
            Envs whose updater draws noise from a non-numpy RNG (e.g.
            MountainCar's native C++ extension) should pass their own seeder,
            e.g. ``_native.set_seed``.
        err_msg: Optional message appended on failure.
    """
    effective_seed_fn = seed_fn if seed_fn is not None else np.random.seed
    vectorized_result = _run_vectorized_transition(
        updater, particles, action, seed, effective_seed_fn
    )
    per_particle_result = _run_per_particle_transition(
        particles, action, per_particle_transition_fn, seed, effective_seed_fn
    )
    np.testing.assert_allclose(vectorized_result, per_particle_result, atol=atol, err_msg=err_msg)


def assert_batch_obs_log_likelihood_matches_loop(
    updater: VectorizedParticleBeliefUpdater,
    particles: np.ndarray,
    action: Any,
    observation: Any,
    per_particle_ll_fn: Callable[[np.ndarray, Any, Any], float],
    atol: float = 1e-10,
    compare_mode: str = "absolute",
    err_msg: str = "",
) -> None:
    """Assert that batch_observation_log_likelihood matches a per-particle loop.

    Args:
        updater: Vectorized updater to test.
        particles: Particle array of shape (N, d).
        action: Action used for the observation model.
        observation: Observation value.
        per_particle_ll_fn: Callable(particle_1d, action, observation) -> float
            that wraps the environment's per-particle log-likelihood logic.
        atol: Absolute tolerance for comparison.
        compare_mode: ``"absolute"`` for direct comparison, or
            ``"pairwise_diff"`` to compare ``ll - ll[0]`` (for environments
            whose per-particle path omits a normalisation constant).
        err_msg: Optional message appended on failure.

    Raises:
        ValueError: If compare_mode is not ``"absolute"`` or ``"pairwise_diff"``.
    """
    if compare_mode not in ("absolute", "pairwise_diff"):
        raise ValueError(f"Unknown compare_mode: {compare_mode!r}")

    vectorized_ll = updater.batch_observation_log_likelihood(particles, action, observation)
    per_particle_ll = _build_per_particle_log_likelihoods(
        particles, action, observation, per_particle_ll_fn
    )

    if compare_mode == "absolute":
        _assert_log_likelihoods_absolute(vectorized_ll, per_particle_ll, atol, err_msg)
    else:
        _assert_log_likelihoods_pairwise_diff(vectorized_ll, per_particle_ll, atol, err_msg)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _run_vectorized_transition(
    updater: VectorizedParticleBeliefUpdater,
    particles: np.ndarray,
    action: Any,
    seed: Optional[int],
    seed_fn: Callable[[int], None],
) -> np.ndarray:
    if seed is not None:
        seed_fn(seed)
    return updater.batch_transition(particles, action)


def _run_per_particle_transition(
    particles: np.ndarray,
    action: Any,
    per_particle_fn: Callable[[np.ndarray, Any], np.ndarray],
    seed: Optional[int],
    seed_fn: Callable[[int], None],
) -> np.ndarray:
    if seed is not None:
        seed_fn(seed)
    n = len(particles)
    result = np.empty_like(particles)
    for i in range(n):
        result[i] = per_particle_fn(particles[i], action)
    return result


def _build_per_particle_log_likelihoods(
    particles: np.ndarray,
    action: Any,
    observation: Any,
    per_particle_ll_fn: Callable[[np.ndarray, Any, Any], float],
) -> np.ndarray:
    n = len(particles)
    result = np.empty(n)
    for i in range(n):
        result[i] = per_particle_ll_fn(particles[i], action, observation)
    return result


def _assert_log_likelihoods_absolute(
    vectorized_ll: np.ndarray,
    per_particle_ll: np.ndarray,
    atol: float,
    err_msg: str,
) -> None:
    finite_mask = np.isfinite(per_particle_ll)
    np.testing.assert_allclose(
        vectorized_ll[finite_mask],
        per_particle_ll[finite_mask],
        atol=atol,
        err_msg=err_msg,
    )


def _assert_log_likelihoods_pairwise_diff(
    vectorized_ll: np.ndarray,
    per_particle_ll: np.ndarray,
    atol: float,
    err_msg: str,
) -> None:
    vectorized_diff = vectorized_ll - vectorized_ll[0]
    per_particle_diff = per_particle_ll - per_particle_ll[0]
    np.testing.assert_allclose(vectorized_diff, per_particle_diff, atol=atol, err_msg=err_msg)
