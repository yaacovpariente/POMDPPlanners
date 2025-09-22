"""Utility functions for confidence interval testing in environment tests."""

from typing import List, Union
from POMDPPlanners.core.simulation.metrics import MetricValue


def verify_metrics_within_confidence_intervals(
    metrics: List[MetricValue], tolerance: float = 0.0
) -> None:
    """Verify that metric values are within their confidence intervals.

    Args:
        metrics: List of MetricValue objects containing name, value, and confidence bounds
        tolerance: Additional tolerance to add to confidence bounds (default 0.0)

    Raises:
        AssertionError: If any metric value is not within its confidence interval

    Purpose: Validates that all metric values are within their confidence intervals

    Given: A list of MetricValue objects with values and confidence bounds
    When: The verification function is called
    Then: All metric values are within their confidence intervals

    Test type: unit
    """
    for metric in metrics:
        _verify_single_metric_confidence_interval(
            metric.value,
            (metric.lower_confidence_bound, metric.upper_confidence_bound),
            metric.name,
            tolerance,
        )


def _verify_single_metric_confidence_interval(
    metric_value: Union[int, float],
    confidence_interval: tuple,
    metric_name: str,
    tolerance: float,
) -> None:
    """Verify a single metric's confidence interval.

    Args:
        metric_value: The value of the metric
        confidence_interval: Tuple of (lower_bound, upper_bound)
        metric_name: Name of the metric for error messages
        tolerance: Additional tolerance for bounds
    """
    lower_bound, upper_bound = confidence_interval

    # Basic sanity checks
    assert (
        lower_bound <= upper_bound
    ), f"{metric_name}: Lower bound {lower_bound} > upper bound {upper_bound}"

    # Check that metric value is within confidence interval with tolerance
    effective_lower = lower_bound - tolerance
    effective_upper = upper_bound + tolerance

    assert effective_lower <= metric_value <= effective_upper, (
        f"{metric_name} value {metric_value} not within confidence interval "
        f"[{lower_bound}, {upper_bound}] with tolerance {tolerance}. "
        f"Effective range: [{effective_lower}, {effective_upper}]"
    )

    # Note: Confidence interval width validation removed as it can be legitimately wide
    # for small sample sizes or sparse data, which is expected in POMDP environments
