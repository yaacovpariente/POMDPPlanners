import numpy as np
from typing import List
from POMDPPlanners.core.simulation import History, StepData, MetricValue
from POMDPPlanners.simulations.simulation_statistics import compute_statistics


def create_test_history(rewards: List[float], discount_factor: float = 0.95) -> History:
    steps = [StepData(None, None, None, None, r) for r in rewards]
    history = History(
        history=steps,
        discount_factor=discount_factor,
        average_state_sampling_time=0.1,
        average_action_time=0.2,
        average_observation_time=0.3,
        average_belief_update_time=0.4,
        average_reward_time=0.5,
        actual_num_steps=len(steps),
        reach_terminal_state=True
    )
    return history


def get_metric_value(metrics: List[MetricValue], name: str) -> MetricValue:
    """Helper function to get a metric value by name from the list."""
    for metric in metrics:
        if metric.name == name:
            return metric
    raise ValueError(f"Metric {name} not found")


def test_compute_statistics_basic():
    # Create test histories with known rewards
    histories = [
        create_test_history([1.0, 2.0, 3.0]),  # Return = 1 + 2*0.95 + 3*0.95^2
        create_test_history([2.0, 3.0, 4.0]),  # Return = 2 + 3*0.95 + 4*0.95^2
        create_test_history([3.0, 4.0, 5.0]),  # Return = 3 + 4*0.95 + 5*0.95^2
    ]

    alpha = 0.1
    confidence_level = 0.95

    stats = compute_statistics(histories, alpha, confidence_level)

    # Test average return
    expected_returns = [
        1.0 + 2.0 * 0.95 + 3.0 * 0.95**2,
        2.0 + 3.0 * 0.95 + 4.0 * 0.95**2,
        3.0 + 4.0 * 0.95 + 5.0 * 0.95**2,
    ]
    expected_avg_return = np.mean(expected_returns)
    avg_return_metric = get_metric_value(stats, "average_return")
    assert abs(avg_return_metric.value - expected_avg_return) < 1e-6

    # Test timing statistics
    assert isinstance(
        get_metric_value(stats, "average_state_sampling_time"), MetricValue
    )
    assert isinstance(get_metric_value(stats, "average_action_time"), MetricValue)
    assert isinstance(get_metric_value(stats, "average_observation_time"), MetricValue)
    assert isinstance(
        get_metric_value(stats, "average_belief_update_time"), MetricValue
    )
    assert isinstance(get_metric_value(stats, "average_reward_time"), MetricValue)

    # Test confidence intervals
    for metric in stats:
        if metric.name not in ["return_value_at_risk", "return_cvar"]:
            assert metric.lower_confidence_bound is not None
            assert metric.upper_confidence_bound is not None

    # Test that all histories have the correct number of steps and terminal state flag
    for history in histories:
        assert history.actual_num_steps == len(history.history)
        assert history.reach_terminal_state is True


def test_compute_statistics_single_history():
    # Test with a single history
    histories = [create_test_history([1.0, 2.0, 3.0])]
    alpha = 0.1
    confidence_level = 0.95

    # For a single history, we expect the return to be the same as the computed return
    expected_return = 1.0 + 2.0 * 0.95 + 3.0 * 0.95**2

    # We need at least two histories to compute confidence intervals
    # So we'll create a duplicate history to satisfy this requirement
    histories = [
        create_test_history([1.0, 2.0, 3.0]),
        create_test_history([1.0, 2.0, 3.0]),
    ]

    stats = compute_statistics(histories, alpha, confidence_level)

    # With identical histories, CVaR and VaR should be the same as the return
    avg_return_metric = get_metric_value(stats, "average_return")
    cvar_metric = get_metric_value(stats, "return_cvar")
    var_metric = get_metric_value(stats, "return_value_at_risk")

    assert abs(avg_return_metric.value - expected_return) < 1e-6
    assert abs(cvar_metric.value - expected_return) < 1e-6
    assert abs(var_metric.value - expected_return) < 1e-6


def test_compute_statistics_negative_rewards():
    # Test with negative rewards
    histories = [
        create_test_history([-1.0, -2.0, -3.0]),
        create_test_history([-2.0, -3.0, -4.0]),
        create_test_history([-3.0, -4.0, -5.0]),
    ]

    alpha = 0.1
    confidence_level = 0.95

    stats = compute_statistics(histories, alpha, confidence_level)

    # Verify that statistics handle negative values correctly
    avg_return_metric = get_metric_value(stats, "average_return")
    cvar_metric = get_metric_value(stats, "return_cvar")
    var_metric = get_metric_value(stats, "return_value_at_risk")

    assert avg_return_metric.value < 0
    assert cvar_metric.value < 0
    assert var_metric.value < 0


def test_compute_statistics_zero_rewards():
    # Test with zero rewards
    histories = [
        create_test_history([0.0, 0.0, 0.0]),
        create_test_history([0.0, 0.0, 0.0]),
        create_test_history([0.0, 0.0, 0.0]),
    ]

    alpha = 0.1
    confidence_level = 0.95

    stats = compute_statistics(histories, alpha, confidence_level)

    # All statistics should be zero
    avg_return_metric = get_metric_value(stats, "average_return")
    cvar_metric = get_metric_value(stats, "return_cvar")
    var_metric = get_metric_value(stats, "return_value_at_risk")

    assert abs(avg_return_metric.value) < 1e-6
    assert abs(cvar_metric.value) < 1e-6
    assert abs(var_metric.value) < 1e-6
