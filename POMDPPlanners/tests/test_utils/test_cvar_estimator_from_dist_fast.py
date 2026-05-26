# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Parity tests for ``cvar_estimator_from_dist_fast``.

The fast variant is the hot-path version used by MCTS backups in
``ICVaR_PFT_DPW`` and ``ICVaR_POMCPOW``. It assumes the caller has
already validated the inputs (weights sum to 1, non-empty arrays,
alpha in [0, 1]) and that values are distinct enough that
duplicate-aggregation is unnecessary. This module pins that the fast
variant agrees numerically with the validating
``cvar_estimator_from_dist`` on every input that satisfies its
preconditions, including the edge cases the slow variant has its own
unit tests for.
"""

import numpy as np
import pytest

from POMDPPlanners.utils.statistics_utils import (
    cvar_estimator_from_dist,
    cvar_estimator_from_dist_fast,
)


# Every input below is a valid (values, weights, alpha) triple per the fast
# variant's contract. Cases mirror the branches the slow variant's own unit
# test exercises plus an unsorted-input case.
_PARITY_CASES = [
    # (id, values, weights, alpha)
    ("single_value", np.array([1.0]), np.array([1.0]), 0.2),
    ("three_skewed", np.array([1.0, 2.0, 3.0]), np.array([0.8, 0.1, 0.1]), 0.2),
    (
        "duplicates_in_tail",
        np.array([1.0, 2.0, 3.0, 3.0]),
        np.array([0.8, 0.1, 0.05, 0.05]),
        0.2,
    ),
    (
        "var_strictly_below_threshold",
        np.array([1.0, 2.0, 3.0]),
        np.array([0.8, 0.15, 0.05]),
        0.1,
    ),
    ("two_values", np.array([1.0, 2.0]), np.array([0.8, 0.2]), 0.1),
    (
        "identical_uniform",
        np.array([7.07106781, 7.07106781, 7.07106781, 7.07106781, 7.07106781]),
        np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        0.1,
    ),
    ("alpha_one", np.array([1.0, 2.0, 3.0]), np.array([0.33, 0.33, 0.34]), 1.0),
    (
        "alpha_zero_propagates_nan",
        np.array([1.0, 2.0, 3.0]),
        np.array([0.33, 0.33, 0.34]),
        0.0,
    ),
    (
        "negative_values",
        np.array([-5.0, -1.0, 3.0]),
        np.array([0.33, 0.33, 0.34]),
        0.2,
    ),
    (
        "zero_weight_at_head",
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, 0.5, 0.5]),
        0.2,
    ),
    (
        "very_small_alpha",
        np.array([1.0, 2.0, 3.0]),
        np.array([0.33, 0.33, 0.34]),
        0.001,
    ),
    (
        "alpha_near_one",
        np.array([1.0, 2.0, 3.0]),
        np.array([0.33, 0.33, 0.34]),
        0.999999999999,
    ),
    (
        "alpha_in_first_bin",
        np.array([1.0, 2.0, 3.0]),
        np.array([0.1, 0.1, 0.8]),
        0.95,
    ),
    (
        "cum_exactly_at_threshold",
        np.array([1.0, 2.0, 3.0]),
        np.array([0.5, 0.3, 0.2]),
        0.2,
    ),
    (
        "nan_value_propagates",
        np.array([1.0, np.nan, 3.0]),
        np.array([0.33, 0.33, 0.34]),
        0.2,
    ),
    (
        "inf_value_propagates",
        np.array([1.0, np.inf, 3.0]),
        np.array([0.33, 0.33, 0.34]),
        0.2,
    ),
    ("unsorted_input", np.array([3.0, 1.0, 2.0]), np.array([0.1, 0.6, 0.3]), 0.4),
    ("many_uniform", np.arange(1.0, 11.0), np.full(10, 0.1), 0.3),
]


def _assert_close_or_nan(slow: float, fast: float) -> None:
    if np.isnan(slow):
        assert np.isnan(fast), f"slow=NaN but fast={fast}"
        return
    if np.isinf(slow):
        assert np.isinf(fast) and np.sign(slow) == np.sign(fast)
        return
    assert np.isclose(slow, fast, atol=1e-9, rtol=1e-9), f"slow={slow} fast={fast}"


@pytest.mark.parametrize(
    "case_id,values,weights,alpha",
    _PARITY_CASES,
    ids=[c[0] for c in _PARITY_CASES],
)
def test_fast_matches_slow_on_curated_cases(
    case_id: str,
    values: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> None:
    """The hot-path CVaR helper agrees numerically with the validating one.

    Purpose: Pin the contract that ``cvar_estimator_from_dist_fast`` is a
    drop-in replacement for ``cvar_estimator_from_dist`` whenever the
    fast variant's preconditions hold (weights sum to 1, non-empty,
    alpha in [0, 1]).

    Given: A (values, weights, alpha) triple that satisfies the fast
    variant's preconditions, including edge cases that exercise the
    same branches the slow variant has unit tests for: duplicate
    values, alpha boundaries, NaN/inf propagation, very small / very
    large alpha, exact-equality threshold.
    When: Both implementations are evaluated on the same input.
    Then: Their results agree (NaN-aware: both NaN, both inf, or close
    to within numerical tolerance).

    Test type: unit
    """
    del case_id  # only used for parametrize ids
    slow = cvar_estimator_from_dist(values, weights, alpha)
    fast = cvar_estimator_from_dist_fast(values, weights, alpha)
    _assert_close_or_nan(slow, fast)


def test_fast_matches_slow_on_random_distributions():
    """Randomized parity check on small distinct-value distributions.

    Purpose: Exercise the fast variant on the same shape of input it
    sees inside MCTS backups (small, distinct v_values, normalized
    visit-count weights) and verify agreement with the validating slow
    variant.

    Given: 200 random (values, weights, alpha) triples with N drawn
    uniformly from [2, 30], distinct values, positive normalized
    weights.
    When: Both implementations process the same triple.
    Then: Results are close to within numerical tolerance.

    Test type: unit
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 31))
        values = rng.standard_normal(n) * 10.0
        if len(np.unique(values)) != n:
            values = np.linspace(-5.0, 5.0, n) + rng.standard_normal(n) * 1e-3
        raw_w = rng.uniform(0.1, 1.0, size=n)
        weights = raw_w / raw_w.sum()
        alpha = float(rng.uniform(0.01, 1.0))
        slow = cvar_estimator_from_dist(values, weights, alpha)
        fast = cvar_estimator_from_dist_fast(values, weights, alpha)
        assert np.isclose(
            slow, fast, atol=1e-9, rtol=1e-9
        ), f"slow={slow} fast={fast} values={values} weights={weights} alpha={alpha}"


def test_single_value_short_circuit():
    """Single-element fast path returns the value itself.

    Purpose: Pin the explicit n==1 short-circuit in the fast variant.

    Given: A one-element distribution.
    When: cvar_estimator_from_dist_fast is called with any alpha.
    Then: Returns the only value as a Python float.

    Test type: unit
    """
    result = cvar_estimator_from_dist_fast(np.array([2.5]), np.array([1.0]), 0.3)
    assert isinstance(result, float)
    assert np.isclose(result, 2.5)
