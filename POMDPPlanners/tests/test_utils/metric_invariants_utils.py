# SPDX-License-Identifier: MIT

"""Metric / history sanity-check helpers shared across env tests.

These helpers complement
:func:`POMDPPlanners.tests.test_utils.confidence_interval_utils.verify_metrics_within_confidence_intervals`,
which only checks that each MetricValue's value lies inside its CI bounds. The
helpers here add structural invariants that should hold for any well-formed
``compute_metrics`` output and for any history compatible with its env's
declared reward range and discount factor.
"""

import math
from typing import List, Optional

import numpy as np

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.simulation import History, MetricValue


_RATE_SUFFIX = "_rate"
_TOTAL_PREFIX = "total_"
_COUNT_PREFIXES = ("total_", "count_", "num_")
# Per-step-bound metrics: a sum-over-episodes of per-step booleans/counts. Each
# such metric is bounded above by the sum of episode lengths (one count per
# step at most). Name hints — collisions / violations / hits / steps — come
# from the env enums in this codebase.
_PER_STEP_BOUND_KEYWORDS = ("collision", "violation", "hit", "_steps")


def verify_metric_sanity(
    metrics: List[MetricValue],
    histories: List[History],
    env: Environment,
) -> None:
    """Run structural invariants over a compute_metrics output.

    Applies four checks driven by metric-name conventions used in this codebase:

    - any metric whose name ends with ``_rate`` has value in [0, 1]
    - any metric whose name starts with ``total_`` / ``count_`` / ``num_`` has
      value >= 0
    - both CI bounds are finite when n_episodes >= 2
    - any "per-step-count over episodes" metric (name starts with ``total_``
      and contains a per-step keyword like ``collision``/``violation``/``hit``,
      and is NOT itself a rate) has value <= sum(episode lengths)

    Args:
        metrics: The MetricValue list returned by ``env.compute_metrics``.
        histories: The histories that were passed to ``compute_metrics``.
        env: The environment instance (kept in the signature so callers thread
            it through; currently unused but reserved for env-aware extensions).

    Raises:
        AssertionError: If any invariant is violated. Message identifies the
            metric, value, and which invariant failed.
    """
    del env  # reserved
    n_episodes = len(histories)
    total_steps = sum(len(h.history) for h in histories)

    for metric in metrics:
        _check_rate_in_unit_interval(metric)
        _check_count_non_negative(metric)
        _check_ci_finite_for_multi_episode(metric, n_episodes)
        _check_per_step_count_bounded(metric, total_steps)


def verify_history_returns_bounded(
    histories: List[History],
    env: Environment,
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> None:
    """Each history's discounted return must lie inside the env-declared bounds.

    For an env with ``reward_range = (r_min, r_max)`` and discount ``gamma`` and
    a history of length H, the discounted return G = sum_{t=0..H-1} gamma^t * r_t
    satisfies:
        - gamma == 1.0:  H * r_min  <= G <= H * r_max
        - gamma <  1.0:  r_min*(1-gamma^H)/(1-gamma) <= G <= r_max*(1-gamma^H)/(1-gamma)

    Skips histories with no rewards available. Skips envs that don't declare a
    reward_range (the helper would have nothing to check against).

    Args:
        histories: The histories to validate.
        env: The environment whose ``reward_range`` and ``discount_factor``
            define the bounds.
        rtol: Relative tolerance forwarded to ``math.isclose`` (currently unused
            but kept for symmetry with absolute-tolerance check).
        atol: Absolute tolerance for floating-point slack at the bounds.

    Raises:
        AssertionError: If a history's discounted return falls outside its
            theoretical bounds.
    """
    del rtol  # reserved
    if env.reward_range is None:
        return

    r_min, r_max = env.reward_range
    gamma = env.discount_factor

    for history_index, history in enumerate(histories):
        rewards = [float(step.reward) for step in history.history if step.reward is not None]
        if not rewards:
            continue

        horizon = len(rewards)
        lower_bound, upper_bound = _discounted_return_bounds(r_min, r_max, gamma, horizon)
        discounted_return = _discounted_return(rewards, gamma)

        assert lower_bound - atol <= discounted_return <= upper_bound + atol, (
            f"history[{history_index}]: discounted return {discounted_return:.6f} "
            f"outside bounds [{lower_bound:.6f}, {upper_bound:.6f}] "
            f"(reward_range=({r_min}, {r_max}), gamma={gamma}, horizon={horizon})"
        )


def verify_return_shift_linearity(
    histories: List[History],
    env: Environment,
    shift: float = 1.0,
    atol: float = 1e-9,
) -> None:
    """Adding constant ``shift`` to every reward shifts the discounted return.

    Property test: G(rewards + c) - G(rewards) == c * sum(gamma^t for t in 0..H-1).
    Catches discount-factor confusion or off-by-one in horizon accounting.

    Args:
        histories: Histories whose rewards drive the property test.
        env: Environment supplying ``discount_factor``.
        shift: Constant added to every reward.
        atol: Absolute tolerance for the equality.

    Raises:
        AssertionError: If the shift in discounted return does not match
            ``shift * sum(gamma^t)`` for any history.
    """
    gamma = env.discount_factor

    for history_index, history in enumerate(histories):
        rewards = [float(step.reward) for step in history.history if step.reward is not None]
        if not rewards:
            continue
        horizon = len(rewards)
        baseline = _discounted_return(rewards, gamma)
        shifted = _discounted_return([r + shift for r in rewards], gamma)
        expected_delta = shift * _geometric_sum(gamma, horizon)
        actual_delta = shifted - baseline
        assert math.isclose(actual_delta, expected_delta, abs_tol=atol), (
            f"history[{history_index}]: shifted return delta {actual_delta:.9f} "
            f"!= expected {expected_delta:.9f} (shift={shift}, gamma={gamma}, "
            f"horizon={horizon})"
        )


def verify_belief_invariants(
    belief: object,
    *,
    expected_n_particles: Optional[int] = None,
    weight_atol: float = 1e-6,
) -> None:
    """Structural invariants for particle beliefs.

    Applies whichever invariants the belief object supports:

    - if it exposes ``log_weights`` or ``weights``: weights are non-negative
      and sum to 1 (within ``weight_atol``); ESS is in [1, N]
    - if it exposes ``particles``: particle count matches ``expected_n_particles``
      when supplied, and is positive otherwise

    Args:
        belief: A particle belief (weighted or unweighted). Anything that
            doesn't expose the relevant attributes is silently skipped per
            attribute, so this helper is safe to call on heterogeneous objects.
        expected_n_particles: If given, asserts ``len(particles) ==
            expected_n_particles``. Useful for confirming particle count is
            stable across a belief update.
        weight_atol: Absolute tolerance for the "weights sum to 1" check.

    Raises:
        AssertionError: If any present invariant is violated. Belief objects
            without the relevant attributes contribute no checks (no failure).
    """
    particles = getattr(belief, "particles", None)
    if particles is not None:
        n_particles = len(particles)
        assert n_particles > 0, "belief has zero particles"
        if expected_n_particles is not None:
            assert (
                n_particles == expected_n_particles
            ), f"particle count {n_particles} != expected {expected_n_particles}"

    weights = _resolve_weights(belief)
    if weights is None:
        return

    assert np.all(
        weights >= -weight_atol
    ), f"belief weights contain negatives: min={float(np.min(weights))}"
    weight_sum = float(np.sum(weights))
    assert math.isclose(
        weight_sum, 1.0, abs_tol=weight_atol
    ), f"belief weights sum to {weight_sum}, expected 1.0 (atol={weight_atol})"

    n_particles = weights.shape[0]
    if n_particles == 0:
        return
    ess = 1.0 / float(np.sum(weights * weights)) if np.any(weights > 0) else 0.0
    assert (
        1.0 - weight_atol <= ess <= n_particles + weight_atol
    ), f"ESS {ess} outside [1, {n_particles}]"


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _check_rate_in_unit_interval(metric: MetricValue) -> None:
    if not metric.name.endswith(_RATE_SUFFIX):
        return
    assert 0.0 <= metric.value <= 1.0, (
        f"{metric.name}: value {metric.value} outside [0, 1] "
        f"(rate metrics must be in the unit interval)"
    )


def _check_count_non_negative(metric: MetricValue) -> None:
    if not metric.name.startswith(_COUNT_PREFIXES) or metric.name.endswith(_RATE_SUFFIX):
        return
    assert metric.value >= 0.0, (
        f"{metric.name}: value {metric.value} < 0 " f"(count / total metrics must be non-negative)"
    )


def _check_ci_finite_for_multi_episode(metric: MetricValue, n_episodes: int) -> None:
    if n_episodes < 2:
        return
    assert math.isfinite(metric.lower_confidence_bound), (
        f"{metric.name}: lower bound is non-finite "
        f"({metric.lower_confidence_bound}) with n_episodes={n_episodes}"
    )
    assert math.isfinite(metric.upper_confidence_bound), (
        f"{metric.name}: upper bound is non-finite "
        f"({metric.upper_confidence_bound}) with n_episodes={n_episodes}"
    )


def _check_per_step_count_bounded(metric: MetricValue, total_steps: int) -> None:
    if not metric.name.startswith(_TOTAL_PREFIX):
        return
    if metric.name.endswith(_RATE_SUFFIX):
        return
    if not any(keyword in metric.name for keyword in _PER_STEP_BOUND_KEYWORDS):
        return
    assert metric.value <= total_steps + 1e-9, (
        f"{metric.name}: value {metric.value} > total_steps {total_steps} "
        f"(per-step-count over episodes cannot exceed sum of episode lengths)"
    )


def _discounted_return(rewards: List[float], gamma: float) -> float:
    if gamma == 1.0:
        return float(sum(rewards))
    total = 0.0
    factor = 1.0
    for reward in rewards:
        total += factor * reward
        factor *= gamma
    return total


def _discounted_return_bounds(r_min: float, r_max: float, gamma: float, horizon: int) -> tuple:
    if gamma >= 1.0:
        return horizon * r_min, horizon * r_max
    geometric = _geometric_sum(gamma, horizon)
    return r_min * geometric, r_max * geometric


def _geometric_sum(gamma: float, horizon: int) -> float:
    if gamma >= 1.0:
        return float(horizon)
    return float((1.0 - gamma**horizon) / (1.0 - gamma))


def _resolve_weights(belief: object) -> Optional[np.ndarray]:
    log_weights = getattr(belief, "log_weights", None)
    if log_weights is not None:
        log_weights_array = np.asarray(log_weights, dtype=float)
        # Stable softmax via shift; avoids overflow when log_weights are large.
        shifted = log_weights_array - float(np.max(log_weights_array))
        exp_weights = np.exp(shifted)
        denom = float(np.sum(exp_weights))
        if denom <= 0.0:
            return None
        return exp_weights / denom

    weights = getattr(belief, "weights", None)
    if weights is None:
        return None
    weights_array = np.asarray(weights, dtype=float)
    return weights_array
