# SPDX-License-Identifier: MIT

"""Metrics comparison plotting utilities.

This module provides functions for comparing and visualizing metrics across different
POMDP policies and environments.
"""

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import MetricValue

matplotlib.use("Agg")  # Use non-interactive backend

# Set up logger
logger = logging.getLogger(__name__)


def _validate_metrics_dict(metrics_dict: Dict[str, Dict[str, List[MetricValue]]]) -> None:
    """Validate the structure of metrics_dict."""
    if not isinstance(metrics_dict, dict):
        raise TypeError("metrics_dict must be a dictionary")
    if len(metrics_dict) == 0:
        raise ValueError("metrics_dict cannot be empty")
    if not all(isinstance(policy_metrics, dict) for policy_metrics in metrics_dict.values()):
        raise TypeError("All policy_metrics must be dictionaries")
    if not all(
        isinstance(metric, MetricValue)
        for policy_metrics in metrics_dict.values()
        for metrics_list in policy_metrics.values()
        for metric in metrics_list
    ):
        raise TypeError("All metric values must be MetricValue objects")


def _collect_all_metric_names(metrics_dict: Dict[str, Dict[str, List[MetricValue]]]) -> set:
    """Collect all unique metric names from the metrics dictionary."""
    all_metric_names = set()
    for policy_metrics_dict in metrics_dict.values():
        for metrics in policy_metrics_dict.values():
            for metric in metrics:
                all_metric_names.add(metric.name)
    return all_metric_names


def _extract_metric_data(
    policy_metrics_dict: Dict[str, List[MetricValue]], metric_name: str
) -> tuple:
    """Extract metric data for a specific metric name from policy metrics."""
    policy_names = []
    metric_values = []
    lower_bounds = []
    upper_bounds = []

    for policy_name, metrics in policy_metrics_dict.items():
        metric_val = next((m for m in metrics if m.name == metric_name), None)
        if metric_val is None:
            continue
        if (
            np.isnan(metric_val.value)
            or np.isnan(metric_val.lower_confidence_bound)
            or np.isnan(metric_val.upper_confidence_bound)
        ):
            continue

        policy_names.append(policy_name)
        metric_values.append(metric_val.value)
        lower_bounds.append(metric_val.lower_confidence_bound)
        upper_bounds.append(metric_val.upper_confidence_bound)

    return policy_names, metric_values, lower_bounds, upper_bounds


def _create_error_bars(
    metric_values: List[float], lower_bounds: List[float], upper_bounds: List[float]
) -> np.ndarray:
    """Create error bars for plotting."""
    yerr = np.vstack(
        [
            np.array(metric_values) - np.array(lower_bounds),  # Lower errors
            np.array(upper_bounds) - np.array(metric_values),  # Upper errors
        ]
    )

    if np.all(yerr == 0):
        yerr = np.full((2, len(metric_values)), 1e-10)

    return yerr


def _create_bar_chart(
    x: np.ndarray,
    metric_values: List[float],
    yerr: np.ndarray,
    width: float = 0.6,
):
    """Create a bar chart with error bars."""
    return plt.bar(
        x,
        metric_values,
        width,
        yerr=yerr,
        capsize=8,
        color="skyblue",
        alpha=0.85,
        edgecolor="black",
        linewidth=3,
    )


def _customize_plot_labels(
    metric_name: str,
    env_name: str,
    policy_names: List[str],
    x: np.ndarray,
):
    """Customize plot labels, title, and ticks."""
    plt.xlabel("Policy", fontsize=42, fontweight="bold")
    plt.ylabel(
        metric_name.replace("_", " ").title(),
        fontsize=42,
        fontweight="bold",
    )
    plt.title(
        f'{metric_name.replace("_", " ").title()} - {env_name}',
        fontsize=47,
        fontweight="bold",
        pad=20,
    )
    plt.xticks(
        x,
        policy_names,
        rotation=45,
        ha="right",
        fontsize=31,
        fontweight="bold",
    )
    plt.yticks(fontsize=31, fontweight="bold")


def _add_value_labels_to_bars(bars, metric_values: List[float], yerr: np.ndarray):
    """Add value labels on top of bars."""
    for i, (bar_artist, value) in enumerate(zip(bars, metric_values)):
        height = bar_artist.get_height()
        plt.text(
            bar_artist.get_x() + bar_artist.get_width() / 2.0,
            height + yerr[1][i] + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=28,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8},
        )


def plot_metrics_comparison(
    statistics: List[List[MetricValue]],
    environments: List[Environment],
    policies: Sequence[Policy],
    cache_dir_path: Path,
) -> None:
    """
    Plot bar plots comparing statistics across environments and policies.

    Args:
        statistics: List of lists of MetricValue objects for each environment-policy combination
        environments: List of environments
        policies: List of policies
        cache_dir_path: Path to save the plots
    """
    if not (len(statistics) > 0 and len(environments) > 0 and len(policies) > 0):
        raise ValueError("Statistics, environments, and policies lists must not be empty")

    # Create plots directory if it doesn't exist
    plots_dir = cache_dir_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Get all unique metric names from the first statistics list
    stat_names = list({metric.name for metric in statistics[0]})

    # Create a figure for each statistic
    for stat_name in stat_names:
        plt.figure(figsize=(12, 6))

        # Prepare data for plotting
        n_pairs = len(statistics)
        x = np.arange(n_pairs)
        width = 0.8

        # Plot bars for each environment-policy pair
        means = []
        lower_bounds = []
        upper_bounds = []
        labels = []

        for i, (env, policy) in enumerate(zip(environments, policies)):
            if i >= len(statistics):
                break

            # Find the metric with the matching name
            metric = next((m for m in statistics[i] if m.name == stat_name), None)
            if metric is None:
                continue

            if (
                np.isnan(metric.value)
                or np.isnan(metric.lower_confidence_bound)
                or np.isnan(metric.upper_confidence_bound)
            ):
                continue

            means.append(metric.value)
            lower_bounds.append(metric.lower_confidence_bound)
            upper_bounds.append(metric.upper_confidence_bound)
            labels.append(f"{env.__class__.__name__}\n{policy.__class__.__name__}")

        if not means:  # Skip if no valid data points
            plt.close()
            continue

        # Plot bars
        # Ensure confidence bounds are properly ordered (lower < upper)
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)
        # Create 2D array for yerr: [lower_errors, upper_errors]
        # This properly handles asymmetric confidence intervals
        yerr = np.vstack(
            [
                np.abs(np.array(means) - lower_bounds),  # Lower errors (absolute value)
                np.abs(upper_bounds - np.array(means)),  # Upper errors (absolute value)
            ]
        )

        if np.all(yerr == 0):
            yerr = np.full((2, len(means)), 1e-10)

        plt.bar(
            x[: len(means)],
            means,
            width,
            yerr=yerr,
            capsize=5,
        )

        # Customize the plot
        plt.xlabel("Environment-Policy Pair")
        plt.ylabel(stat_name.replace("_", " ").title())
        plt.title(f'{stat_name.replace("_", " ").title()} Comparison')
        plt.xticks(x[: len(means)], labels, rotation=45, ha="right")

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save the plot
        plt.savefig(plots_dir / f"{stat_name}_comparison.png")
        plt.close()

        # Log the plot to MLflow
        mlflow.log_artifact(str(plots_dir / f"{stat_name}_comparison.png"))


def plot_policies_comparison_on_environment(
    metrics_dict: Dict[str, Dict[str, List[MetricValue]]],
    cache_dir_path: Path,
) -> None:
    """
    Plot bar plots comparing policies across environments for each metric type, using academic publication style.

    Args:
        metrics_dict: Dictionary mapping environment names to dictionaries of policy metrics
        cache_dir_path: Path to save the plots
    """
    # Validate inputs
    _validate_metrics_dict(metrics_dict)

    # Create plots directory if it doesn't exist
    plots_dir = cache_dir_path
    plots_dir.mkdir(exist_ok=True)

    # Get all unique metric names across all environments and policies
    all_metric_names = _collect_all_metric_names(metrics_dict)

    for env_name, policy_metrics_dict in metrics_dict.items():
        for metric_name in all_metric_names:
            # Extract metric data
            policy_names, metric_values, lower_bounds, upper_bounds = _extract_metric_data(
                policy_metrics_dict, metric_name
            )

            if not policy_names:
                continue

            try:
                # Academic style plot configuration
                plt.figure(figsize=(16, 12))
                x = np.arange(len(policy_names))
                width = 0.6

                # Create error bars
                yerr = _create_error_bars(metric_values, lower_bounds, upper_bounds)

                # Create bar chart
                bars = _create_bar_chart(x, metric_values, yerr, width)

                # Customize plot labels and styling
                _customize_plot_labels(metric_name, env_name, policy_names, x)

                # Add value labels on top of bars
                _add_value_labels_to_bars(bars, metric_values, yerr)

                # Grid and layout
                plt.grid(True, alpha=0.3, axis="y")
                plt.tight_layout()

                # Save the plot
                plot_filename = f"{env_name}_{metric_name}_comparison.png"
                plt.savefig(plots_dir / plot_filename, dpi=300, bbox_inches="tight")
                plt.close()

            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning("Error creating plot for %s - %s: %s", env_name, metric_name, str(e))
                plt.close()  # Make sure to close the figure even if there's an error
                continue
