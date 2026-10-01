# SPDX-License-Identifier: MIT

"""Environment-agnostic two-sample checks for vectorized-vs-scalar parity.

A vectorized kernel and the scalar environment it duplicates almost never
consume the RNG in the same order, so their samples cannot be compared value by
value. What must agree is the *distribution* each one samples from. This module
compares two sample matrices drawn from the same input and assumes nothing
about what the columns mean:

* A column that is constant in both samples is compared exactly. Most state
  components are deterministic given ``(s, a)`` -- a grid position, a flag, a
  counter -- and a deterministic disagreement is a bug a statistical test would
  dilute.
* When the rows take few distinct values overall (a discrete state), the joint
  row distribution is compared as categorical (see below).
* Otherwise every non-constant column is compared on its own: as categorical
  when the column is discrete, with a two-sample Kolmogorov-Smirnov test plus a
  Brown-Forsythe test of spread when it is continuous. The per-column p-values
  are Bonferroni-corrected.

A categorical comparison runs two tests and keeps the smaller p-value
(doubled). The chi-square homogeneity test sees many small shifts at once; a
per-category binomial test sees one category that only one sample produces --
the commonest form of a transition bug (a branch never taken, a cell never
reached) and one a chi-square spread over twenty categories can dilute below
significance.

Every caller seeds its RNGs, so on one platform a given pair of samples -- and
therefore every verdict -- is fixed. Native kernels draw through
``std::*_distribution``, whose streams differ between libc++ and libstdc++, so
the samples (not the laws) differ between macOS and Linux. ``alpha`` is set low
enough that a correct kernel does not sit near the threshold, which is what
keeps the checks from being flaky across platforms or when an unrelated change
shifts the RNG stream.

Functions:
    assert_same_distribution: Assert two ``(n, d)`` sample matrices share a law.
"""

from typing import List, Tuple

import numpy as np
from scipy import stats

# Rows or columns with at most this many distinct values are treated as
# categorical. Continuous samples essentially never repeat, so the cut only has
# to separate "a handful of grid cells" from "a float per draw".
_MAX_CATEGORIES = 64

# ...and only when each category is seen often enough to be a category. With
# few draws a continuous sample also has few distinct values, and treating
# those as categories would make two samples of one continuous law look
# disjoint.
_MIN_DRAWS_PER_CATEGORY = 5

# Chi-square needs an expected count of about five per cell to be valid; rarer
# categories are pooled into one bucket instead of being dropped.
_MIN_EXPECTED_COUNT = 5.0

# Family-wise significance level. With seeded samples this is a fixed verdict,
# and 1e-4 keeps a correct kernel far from the boundary.
DEFAULT_ALPHA = 1e-4


def assert_same_distribution(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    label_a: str = "vectorized",
    label_b: str = "scalar",
    exact_atol: float = 1e-6,
) -> None:
    """Assert two sample matrices are draws from the same distribution.

    Args:
        samples_a: ``(n_a, d)`` samples from the first implementation.
        samples_b: ``(n_b, d)`` samples from the second implementation.
        alpha: Family-wise significance level for the statistical tests.
        label_a: Name of the first implementation, used in failure messages.
        label_b: Name of the second implementation, used in failure messages.
        exact_atol: Tolerance for comparing a column that is constant in both
            samples.

    Raises:
        AssertionError: If the shapes disagree, a deterministic column
            disagrees, or a statistical test rejects equality at ``alpha``.
    """
    a = _as_matrix(samples_a)
    b = _as_matrix(samples_b)
    assert a.shape[1] == b.shape[1], (
        f"{label_a} samples have {a.shape[1]} columns but {label_b} samples have " f"{b.shape[1]}"
    )
    # A kernel that returns NaN or inf must not slip through: the statistical
    # tests below return NaN p-values on such input, and NaN compares as a pass.
    for label, sample in ((label_a, a), (label_b, b)):
        assert np.all(np.isfinite(sample)), (
            f"{label} samples contain non-finite values in columns "
            f"{sorted(set(np.nonzero(~np.isfinite(sample))[1].tolist()))}"
        )
    _assert_constant_columns_agree(a, b, exact_atol, label_a, label_b)
    varying = [
        j for j in range(a.shape[1]) if not (_is_constant(a[:, j]) and _is_constant(b[:, j]))
    ]
    if not varying:
        return

    joint_a = a[:, varying]
    joint_b = b[:, varying]
    if _is_categorical(joint_a, joint_b):
        # Keys index one category table shared by both samples.
        keys = _row_keys(np.vstack([joint_a, joint_b]))
        p_value = _categorical_p_value(keys[: len(joint_a)], keys[len(joint_a) :])
        assert p_value > alpha, (
            f"{label_a} and {label_b} sample different joint distributions over columns "
            f"{varying}: categorical p={p_value:.2e} <= alpha={alpha:g}. "
            f"{_describe_rows(joint_a, joint_b, label_a, label_b)}"
        )
        return

    failures: List[Tuple[int, str, float]] = []
    for column in varying:
        test_name, p_value = _column_p_value(a[:, column], b[:, column])
        if p_value * len(varying) <= alpha:
            failures.append((column, test_name, p_value))
    assert not failures, (
        f"{label_a} and {label_b} sample different marginals (Bonferroni over "
        f"{len(varying)} columns, alpha={alpha:g}): "
        + "; ".join(
            f"column {column} {test_name} p={p_value:.2e} "
            f"(mean {a[:, column].mean():.4g} vs {b[:, column].mean():.4g}, "
            f"std {a[:, column].std():.4g} vs {b[:, column].std():.4g})"
            for column, test_name, p_value in failures
        )
    )


def _as_matrix(samples: np.ndarray) -> np.ndarray:
    matrix = np.asarray(samples, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    return matrix.reshape(matrix.shape[0], -1)


def _is_constant(column: np.ndarray) -> bool:
    return bool(np.all(column == column[0]))


def _assert_constant_columns_agree(
    a: np.ndarray, b: np.ndarray, atol: float, label_a: str, label_b: str
) -> None:
    disagreements = []
    for j in range(a.shape[1]):
        if _is_constant(a[:, j]) and _is_constant(b[:, j]):
            if not np.isclose(a[0, j], b[0, j], atol=atol, rtol=0.0, equal_nan=True):
                disagreements.append(f"column {j}: {label_a}={a[0, j]!r} {label_b}={b[0, j]!r}")
    assert (
        not disagreements
    ), f"Deterministic components disagree between {label_a} and {label_b}: " + "; ".join(
        disagreements
    )


def _row_keys(rows: np.ndarray) -> np.ndarray:
    # Round so float noise in the last ulp does not split one category in two.
    rounded = np.round(rows, decimals=9)
    _, inverse = np.unique(rounded, axis=0, return_inverse=True)
    return np.asarray(inverse).ravel()


def _n_categories(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.unique(np.round(np.vstack([a, b]), decimals=9), axis=0).shape[0])


def _chi_square_p_value(keys_a: np.ndarray, keys_b: np.ndarray) -> float:
    """Chi-square homogeneity p-value for two categorical samples.

    Keys from both samples share one index space. Categories whose expected
    count falls below :data:`_MIN_EXPECTED_COUNT` are pooled into a single
    bucket so the asymptotic test stays valid.
    """
    labels = np.concatenate([keys_a, keys_b])
    n_categories = int(labels.max()) + 1
    counts = np.vstack(
        [
            np.bincount(keys_a, minlength=n_categories),
            np.bincount(keys_b, minlength=n_categories),
        ]
    ).astype(np.float64)
    totals = counts.sum(axis=0)
    share = min(counts[0].sum(), counts[1].sum()) / counts.sum()
    common = totals * share >= _MIN_EXPECTED_COUNT
    pooled = np.hstack([counts[:, common], counts[:, ~common].sum(axis=1, keepdims=True)])
    pooled = pooled[:, pooled.sum(axis=0) > 0]
    if pooled.shape[1] < 2:
        return 1.0
    return float(stats.chi2_contingency(pooled, correction=False)[1])


def _cell_p_value(keys_a: np.ndarray, keys_b: np.ndarray) -> float:
    """Smallest per-category binomial p-value, Bonferroni-corrected.

    Under equal laws each category's draws split between the two samples in
    proportion to the sample sizes; a category drawn 30 times by one sample and
    never by the other has p of about 1e-9.
    """
    n_categories = int(max(keys_a.max(initial=0), keys_b.max(initial=0))) + 1
    counts_a = np.bincount(keys_a, minlength=n_categories)
    counts_b = np.bincount(keys_b, minlength=n_categories)
    share = len(keys_a) / (len(keys_a) + len(keys_b))
    smallest = 1.0
    for count_a, count_b in zip(counts_a, counts_b):
        if count_a + count_b:
            result = stats.binomtest(int(count_a), int(count_a + count_b), share)
            smallest = min(smallest, float(result.pvalue))
    return min(1.0, smallest * n_categories)


def _categorical_p_value(keys_a: np.ndarray, keys_b: np.ndarray) -> float:
    """Combined p-value of the chi-square and per-category tests (Bonferroni over two)."""
    return min(1.0, 2.0 * min(_chi_square_p_value(keys_a, keys_b), _cell_p_value(keys_a, keys_b)))


def _is_categorical(a: np.ndarray, b: np.ndarray) -> bool:
    n_categories = _n_categories(a, b)
    return n_categories <= min(_MAX_CATEGORIES, (len(a) + len(b)) / _MIN_DRAWS_PER_CATEGORY)


def _column_p_value(column_a: np.ndarray, column_b: np.ndarray) -> Tuple[str, float]:
    if _is_categorical(column_a[:, None], column_b[:, None]):
        keys = _row_keys(np.concatenate([column_a, column_b])[:, None])
        return "categorical", _categorical_p_value(keys[: len(column_a)], keys[len(column_a) :])
    # KS is most sensitive near the median and weak on a change of spread
    # alone -- the signature of a mis-scaled noise term. Brown-Forsythe
    # (Levene's test about the median) covers spread; the smaller p-value is
    # kept, doubled.
    ks = float(stats.ks_2samp(column_a, column_b).pvalue)
    spread = float(stats.levene(column_a, column_b, center="median").pvalue)
    # Both columns are finite and non-constant here, so a non-finite p-value
    # can only come from a degenerate spread (one column constant); count it
    # as no evidence for the spread test and keep KS.
    if not np.isfinite(spread):
        spread = 1.0
    if not np.isfinite(ks):
        ks = 0.0
    return "KS+Brown-Forsythe", min(1.0, 2.0 * min(ks, spread))


def _describe_rows(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> str:
    """Return the most frequent rows of each sample, for a readable failure."""

    def top(rows: np.ndarray) -> str:
        unique, counts = np.unique(np.round(rows, 9), axis=0, return_counts=True)
        order = np.argsort(counts)[::-1][:4]
        return ", ".join(
            f"{np.array2string(unique[i], precision=3)}: {counts[i] / len(rows):.3f}" for i in order
        )

    return f"Most frequent rows -- {label_a}: {top(a)} | {label_b}: {top(b)}"
