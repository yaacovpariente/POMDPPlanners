"""Shared belief-level equivalence checks for particle beliefs.

This module provides environment-agnostic assertion helpers that compare a
standard :class:`~POMDPPlanners.core.belief.particle_beliefs.WeightedParticleBelief`
against a
:class:`~POMDPPlanners.core.belief.vectorized_weighted_particle_belief.VectorizedWeightedParticleBelief`
through their public interface (``update``, ``normalized_weights``, ``sample``).

Callers build the two beliefs with aligned particles and weights; the helpers
run the supplied action/observation through both paths and assert the expected
invariants. Each ``update``-invoking helper seeds ``numpy``'s global RNG
identically on both paths so stochastic transitions and observation draws
consume the same random numbers, mirroring the pattern used in
:mod:`POMDPPlanners.tests.test_core.test_belief.vectorized_updater_test_utils`.

Functions:
    assert_update_particles_match: Next particles agree after one ``update``.
    assert_update_weights_match: Normalized weights agree after one ``update``.
    assert_update_top_k_ranking_agrees: Top-K particle ranking agrees.
    assert_update_equivalence: Combined particle + weight + ranking check.
    assert_chained_update_equivalence: Invariants hold across a sequence of updates.
    assert_normalized_weights_match: Pre-update weight-vector sanity check.
    assert_sample_distributions_match: Empirical ``sample`` histograms agree.
"""

from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.core.environment import Environment


BeliefPair = Tuple[WeightedParticleBelief, VectorizedWeightedParticleBelief]
ActionObservationStep = Tuple[Any, Any]
ParticleToArray = Callable[[Any], np.ndarray]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def assert_update_particles_match(
    base: WeightedParticleBelief,
    vec: VectorizedWeightedParticleBelief,
    action: Any,
    observation: Any,
    pomdp: Environment,
    atol: float = 1e-10,
    seed: Optional[int] = None,
    particle_to_array: Optional[ParticleToArray] = None,
) -> BeliefPair:
    """Run one update on both beliefs and assert next particles agree.

    Args:
        base: Baseline ``WeightedParticleBelief`` with aligned particles.
        vec: Vectorized belief holding the same particles as ``base``.
        action: Action passed to both ``update`` calls.
        observation: Observation passed to both ``update`` calls.
        pomdp: Environment used by the baseline's per-particle transition loop.
        atol: Absolute tolerance for the particle array comparison.
        seed: If provided, ``numpy``'s global RNG is seeded to this value
            before each path so stochastic transitions consume identical
            random sequences.
        particle_to_array: Optional callable converting a baseline particle
            to a 1-D ``np.ndarray``. Needed when ``base.particles`` holds
            non-ndarray objects (e.g. dataclasses) whose layout matches
            ``vec.particles`` rows. Defaults to :func:`np.asarray`.

    Returns:
        The updated ``(base, vec)`` belief pair so callers can chain asserts.
    """
    base_next, vec_next = _run_updates(base, vec, action, observation, pomdp, seed)
    _check_particles(base_next, vec_next, atol=atol, particle_to_array=particle_to_array)
    return base_next, vec_next


def assert_update_weights_match(
    base: WeightedParticleBelief,
    vec: VectorizedWeightedParticleBelief,
    action: Any,
    observation: Any,
    pomdp: Environment,
    atol: float = 1e-6,
    significance_threshold: Optional[float] = None,
    seed: Optional[int] = None,
) -> BeliefPair:
    """Run one update on both beliefs and assert normalized weights agree.

    Args:
        base: Baseline ``WeightedParticleBelief`` with aligned particles.
        vec: Vectorized belief holding the same particles as ``base``.
        action: Action passed to both ``update`` calls.
        observation: Observation passed to both ``update`` calls.
        pomdp: Environment used by the baseline's per-particle transition loop.
        atol: Absolute tolerance for the weight comparison.
        significance_threshold: If given, only compare normalized weights for
            baseline particles with ``normalized_weight > significance_threshold``.
            This absorbs the ``log(eps + p)`` floor in
            ``WeightedParticleBelief._update_weights`` vs. the vectorized
            path's direct log-space accumulation, which drives tiny
            low-weight particles apart without affecting the belief's
            effective distribution.
        seed: If provided, ``numpy``'s global RNG is seeded to this value
            before each path so stochastic transitions consume identical
            random sequences.

    Returns:
        The updated ``(base, vec)`` belief pair.
    """
    base_next, vec_next = _run_updates(base, vec, action, observation, pomdp, seed)
    _check_weights(base_next, vec_next, atol=atol, significance_threshold=significance_threshold)
    return base_next, vec_next


def assert_update_top_k_ranking_agrees(
    base: WeightedParticleBelief,
    vec: VectorizedWeightedParticleBelief,
    action: Any,
    observation: Any,
    pomdp: Environment,
    k: int = 5,
    seed: Optional[int] = None,
) -> BeliefPair:
    """Run one update and assert the top-``k`` particle indices agree.

    Args:
        base: Baseline ``WeightedParticleBelief`` with aligned particles.
        vec: Vectorized belief holding the same particles as ``base``.
        action: Action passed to both ``update`` calls.
        observation: Observation passed to both ``update`` calls.
        pomdp: Environment used by the baseline's per-particle transition loop.
        k: Number of top particles by weight to compare.
        seed: If provided, ``numpy``'s global RNG is seeded to this value
            before each path.

    Returns:
        The updated ``(base, vec)`` belief pair.
    """
    base_next, vec_next = _run_updates(base, vec, action, observation, pomdp, seed)
    _check_top_k_ranking(base_next, vec_next, k=k)
    return base_next, vec_next


def assert_update_equivalence(
    base: WeightedParticleBelief,
    vec: VectorizedWeightedParticleBelief,
    action: Any,
    observation: Any,
    pomdp: Environment,
    atol_particles: float = 1e-10,
    atol_weights: float = 1e-6,
    significance_threshold: Optional[float] = 1e-4,
    top_k: int = 5,
    seed: Optional[int] = None,
    particle_to_array: Optional[ParticleToArray] = None,
) -> BeliefPair:
    """Run one update and assert particles, weights, and top-``k`` all agree.

    Combines :func:`assert_update_particles_match`,
    :func:`assert_update_weights_match`, and
    :func:`assert_update_top_k_ranking_agrees` into a single end-to-end check
    for callers that want the full equivalence invariant in one line.

    Args:
        base: Baseline ``WeightedParticleBelief`` with aligned particles.
        vec: Vectorized belief holding the same particles as ``base``.
        action: Action passed to both ``update`` calls.
        observation: Observation passed to both ``update`` calls.
        pomdp: Environment used by the baseline's per-particle transition loop.
        atol_particles: Absolute tolerance for the next-particle comparison.
        atol_weights: Absolute tolerance for the normalized-weight comparison.
        significance_threshold: Weight-masking threshold passed to the weight
            check; set to ``None`` to disable masking.
        top_k: Number of top particles by weight compared for ranking agreement.
        seed: If provided, ``numpy``'s global RNG is seeded to this value
            before each path.

    Returns:
        The updated ``(base, vec)`` belief pair.
    """
    base_next, vec_next = _run_updates(base, vec, action, observation, pomdp, seed)
    _check_particles(base_next, vec_next, atol=atol_particles, particle_to_array=particle_to_array)
    _check_weights(
        base_next,
        vec_next,
        atol=atol_weights,
        significance_threshold=significance_threshold,
    )
    _check_top_k_ranking(base_next, vec_next, k=top_k)
    return base_next, vec_next


def assert_chained_update_equivalence(
    base: WeightedParticleBelief,
    vec: VectorizedWeightedParticleBelief,
    steps: Sequence[ActionObservationStep],
    pomdp: Environment,
    atol_particles: float = 1e-10,
    atol_weights: float = 1e-5,
    seed: Optional[int] = None,
    particle_to_array: Optional[ParticleToArray] = None,
) -> BeliefPair:
    """Run a sequence of updates and assert final particles and weights agree.

    Args:
        base: Baseline ``WeightedParticleBelief`` with aligned particles.
        vec: Vectorized belief holding the same particles as ``base``.
        steps: Iterable of ``(action, observation)`` pairs applied in order.
        pomdp: Environment used by the baseline's per-particle transition loop.
        atol_particles: Absolute tolerance for the final particle comparison.
        atol_weights: Absolute tolerance for the final normalized-weight comparison.
        seed: If provided, ``numpy``'s global RNG is seeded to this value
            before each path of every step.

    Returns:
        The final ``(base, vec)`` belief pair.
    """
    for action, observation in steps:
        base, vec = _run_updates(base, vec, action, observation, pomdp, seed)
    _check_particles(base, vec, atol=atol_particles, particle_to_array=particle_to_array)
    _check_weights(base, vec, atol=atol_weights, significance_threshold=None)
    return base, vec


def assert_normalized_weights_match(
    base: WeightedParticleBelief,
    vec: VectorizedWeightedParticleBelief,
    atol: float = 1e-10,
) -> None:
    """Assert the two beliefs expose the same probability vector over particles.

    Intended as a pre-update sanity check on aligned beliefs.

    Args:
        base: Baseline ``WeightedParticleBelief``.
        vec: Vectorized belief.
        atol: Absolute tolerance for the weight comparison.
    """
    np.testing.assert_allclose(
        vec.normalized_weights,
        base.normalized_weights,
        atol=atol,
        err_msg="Normalized weights diverge between baseline and vectorized beliefs",
    )


def assert_sample_distributions_match(
    base: WeightedParticleBelief,
    vec: VectorizedWeightedParticleBelief,
    n_samples: int = 20_000,
    tol: float = 0.03,
    atol_weights: float = 0.03,
    seed: Optional[int] = None,
    particle_to_array: Optional[ParticleToArray] = None,
) -> None:
    """Assert ``sample`` output distributions agree across beliefs and match weights.

    Draws ``n_samples`` from each belief, aggregates by particle identity
    (duplicate particle rows are merged into their first occurrence so
    discrete-state environments are handled correctly), and runs three checks:

    1. The two empirical histograms agree in L-infinity within ``tol``.
    2. ``base``'s histogram agrees with its own ``normalized_weights``
       within ``atol_weights`` (verifies ``base.sample()`` is unbiased).
    3. Same for ``vec`` (verifies ``vec.sample()`` is unbiased).

    Check 1 alone would pass even if both ``sample()`` impls were biased
    the same way. Checks 2 and 3 close that hole.

    The helper assumes particles are convertible to fixed-length numeric
    arrays -- the same constraint ``VectorizedWeightedParticleBelief``
    imposes on the particle store itself.

    Args:
        base: Baseline ``WeightedParticleBelief``.
        vec: Vectorized belief sharing the same particles as ``base``.
        n_samples: Number of samples drawn from each belief.
        tol: L-infinity tolerance for the two histograms agreeing.
        atol_weights: L-infinity tolerance for each histogram agreeing with
            its belief's ``normalized_weights``.
        seed: If provided, ``numpy``'s global RNG is seeded to this value
            before each sampling path so the test is deterministic.
    """
    reference = _stack_particles(base.particles, particle_to_array=particle_to_array)
    first_of = _canonical_group_indices(reference)
    base_hist = _sample_histogram(base, reference, n_samples, seed, particle_to_array)
    vec_hist = _sample_histogram(vec, reference, n_samples, seed, particle_to_array=None)
    base_weights_agg = _aggregate_by_groups(base.normalized_weights, first_of)
    vec_weights_agg = _aggregate_by_groups(vec.normalized_weights, first_of)

    _assert_distributions_match(base_hist, vec_hist, tol, "base hist", "vec hist")
    _assert_distributions_match(
        base_hist, base_weights_agg, atol_weights, "base hist", "base normalized_weights"
    )
    _assert_distributions_match(
        vec_hist, vec_weights_agg, atol_weights, "vec hist", "vec normalized_weights"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_updates(
    base: WeightedParticleBelief,
    vec: VectorizedWeightedParticleBelief,
    action: Any,
    observation: Any,
    pomdp: Environment,
    seed: Optional[int],
) -> BeliefPair:
    if seed is not None:
        np.random.seed(seed)
    vec_next = vec.update(action=action, observation=observation, pomdp=pomdp)
    if seed is not None:
        np.random.seed(seed)
    base_next = base.update(action=action, observation=observation, pomdp=pomdp)
    return base_next, vec_next


def _check_particles(
    base: WeightedParticleBelief,
    vec: VectorizedWeightedParticleBelief,
    atol: float,
    particle_to_array: Optional[ParticleToArray] = None,
) -> None:
    base_particles = _stack_particles(base.particles, particle_to_array=particle_to_array)
    np.testing.assert_allclose(
        vec.particles,
        base_particles,
        atol=atol,
        err_msg="Vectorized belief particles diverged from per-particle baseline",
    )


def _check_weights(
    base: WeightedParticleBelief,
    vec: VectorizedWeightedParticleBelief,
    atol: float,
    significance_threshold: Optional[float],
) -> None:
    base_weights = base.normalized_weights
    vec_weights = vec.normalized_weights
    mask = _build_significance_mask(base_weights, significance_threshold)
    np.testing.assert_allclose(
        vec_weights[mask],
        base_weights[mask],
        atol=atol,
        err_msg="Normalized weights diverge between baseline and vectorized beliefs after update",
    )


def _check_top_k_ranking(
    base: WeightedParticleBelief,
    vec: VectorizedWeightedParticleBelief,
    k: int,
) -> None:
    effective_k = min(k, len(base.normalized_weights))
    base_top = np.argsort(base.normalized_weights)[::-1][:effective_k]
    vec_top = np.argsort(vec.normalized_weights)[::-1][:effective_k]
    np.testing.assert_array_equal(
        vec_top,
        base_top,
        err_msg=f"Top-{effective_k} particle ranking disagrees between baseline and vectorized beliefs",
    )


def _build_significance_mask(
    base_weights: np.ndarray,
    significance_threshold: Optional[float],
) -> np.ndarray:
    if significance_threshold is None:
        return np.ones_like(base_weights, dtype=bool)
    mask = base_weights > significance_threshold
    assert np.any(mask), (
        f"No baseline particle exceeds significance_threshold={significance_threshold}; "
        "relax the threshold or pick a different action/observation"
    )
    return mask


def _stack_particles(
    particles: List[Any],
    particle_to_array: Optional[ParticleToArray] = None,
) -> np.ndarray:
    converter = particle_to_array or np.asarray
    return np.stack([converter(p) for p in particles])


def _sample_histogram(
    belief: Any,
    reference_particles: np.ndarray,
    n_samples: int,
    seed: Optional[int],
    particle_to_array: Optional[ParticleToArray] = None,
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    n_particles = reference_particles.shape[0]
    hist = np.zeros(n_particles, dtype=float)
    for _ in range(n_samples):
        idx = _locate_first_match_index(belief.sample(), reference_particles, particle_to_array)
        hist[idx] += 1.0
    hist /= n_samples
    return hist


def _locate_first_match_index(
    sample: Any,
    reference_particles: np.ndarray,
    particle_to_array: Optional[ParticleToArray] = None,
) -> int:
    converter = particle_to_array or np.asarray
    sample_array = converter(sample)
    matches = np.where(np.all(reference_particles == sample_array, axis=1))[0]
    if len(matches) == 0:
        raise AssertionError(
            "Sampled state does not match any reference particle; "
            "beliefs are not aligned on the same particle set"
        )
    return int(matches[0])


def _canonical_group_indices(particles_2d: np.ndarray) -> np.ndarray:
    n = particles_2d.shape[0]
    first_of = np.arange(n)
    for i in range(1, n):
        matches = np.where(np.all(particles_2d[:i] == particles_2d[i], axis=1))[0]
        if len(matches) > 0:
            first_of[i] = matches[0]
    return first_of


def _aggregate_by_groups(values_per_index: np.ndarray, first_of: np.ndarray) -> np.ndarray:
    n = len(first_of)
    agg = np.zeros(n, dtype=float)
    for i in range(n):
        agg[first_of[i]] += float(values_per_index[i])
    return agg


def _assert_distributions_match(
    distribution_a: np.ndarray,
    distribution_b: np.ndarray,
    tol: float,
    label_a: str,
    label_b: str,
) -> None:
    max_diff = float(np.max(np.abs(distribution_a - distribution_b)))
    assert max_diff <= tol, f"{label_a} vs {label_b}: max L-inf diff {max_diff:.4f} > tol={tol}"
