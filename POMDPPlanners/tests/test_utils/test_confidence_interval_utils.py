"""Tests for confidence interval testing utilities."""

import pytest
from POMDPPlanners.core.simulation.metrics import MetricValue
from POMDPPlanners.tests.test_utils.confidence_interval_utils import (
    verify_metrics_within_confidence_intervals,
)


def test_verify_metrics_valid_intervals():
    """Test confidence interval verification with valid intervals.

    Purpose: Validates that verification passes for metrics within their confidence intervals

    Given: MetricValue objects with values within their confidence bounds
    When: verify_metrics_within_confidence_intervals is called
    Then: No assertions are raised, confirming valid intervals pass

    Test type: unit
    """
    metrics = [
        MetricValue(
            name="test_metric",
            value=5.0,
            lower_confidence_bound=4.0,
            upper_confidence_bound=6.0,
        ),
        MetricValue(
            name="zero_metric",
            value=0.0,
            lower_confidence_bound=0.0,
            upper_confidence_bound=0.0,
        ),
        MetricValue(
            name="narrow_metric",
            value=10.0,
            lower_confidence_bound=10.0,
            upper_confidence_bound=10.0,
        ),
    ]

    # Should not raise any exceptions
    verify_metrics_within_confidence_intervals(metrics)


def test_verify_metrics_outside_intervals():
    """Test confidence interval verification with metrics outside intervals.

    Purpose: Validates that verification fails for metrics outside their confidence intervals

    Given: MetricValue objects with values outside their confidence bounds
    When: verify_metrics_within_confidence_intervals is called
    Then: AssertionError is raised with descriptive message

    Test type: unit
    """
    metrics = [
        MetricValue(
            name="outside_metric",
            value=5.0,
            lower_confidence_bound=6.0,
            upper_confidence_bound=8.0,
        )
    ]

    with pytest.raises(
        AssertionError, match="outside_metric value 5.0 not within confidence interval"
    ):
        verify_metrics_within_confidence_intervals(metrics)


def test_verify_metrics_invalid_bounds():
    """Test confidence interval verification with invalid bound ordering.

    Purpose: Validates that verification fails for metrics with lower > upper bounds

    Given: MetricValue objects with lower bound greater than upper bound
    When: verify_metrics_within_confidence_intervals is called
    Then: AssertionError is raised with descriptive message about bound ordering

    Test type: unit
    """
    metrics = [
        MetricValue(
            name="invalid_bounds",
            value=5.0,
            lower_confidence_bound=8.0,
            upper_confidence_bound=6.0,
        )
    ]

    with pytest.raises(AssertionError, match="invalid_bounds: Lower bound 8.0 > upper bound 6.0"):
        verify_metrics_within_confidence_intervals(metrics)


def test_verify_metrics_with_tolerance():
    """Test confidence interval verification with tolerance parameter.

    Purpose: Validates that tolerance parameter correctly expands effective bounds

    Given: MetricValue with value slightly outside bounds but within tolerance
    When: verify_metrics_within_confidence_intervals is called with tolerance
    Then: No exceptions are raised due to tolerance expansion

    Test type: unit
    """
    metrics = [
        MetricValue(
            name="tolerance_metric",
            value=5.1,
            lower_confidence_bound=4.0,
            upper_confidence_bound=5.0,
        )
    ]

    # Should pass with tolerance of 0.2
    verify_metrics_within_confidence_intervals(metrics, tolerance=0.2)

    # Should fail without tolerance
    with pytest.raises(AssertionError):
        verify_metrics_within_confidence_intervals(metrics, tolerance=0.0)


def test_verify_metrics_with_wide_intervals():
    """Test confidence interval verification accepts wide intervals.

    Purpose: Validates that verification accepts metrics with wide confidence intervals (common in POMDP environments)

    Given: MetricValue objects with confidence intervals much wider than metric values
    When: verify_metrics_within_confidence_intervals is called
    Then: No exceptions are raised, as wide intervals are acceptable for small samples

    Test type: unit
    """
    metrics = [
        MetricValue(
            name="wide_metric",
            value=1.0,
            lower_confidence_bound=0.0,
            upper_confidence_bound=5.0,
        )  # 500% relative width
    ]

    # Should not raise any exceptions - wide intervals are acceptable
    verify_metrics_within_confidence_intervals(metrics)
