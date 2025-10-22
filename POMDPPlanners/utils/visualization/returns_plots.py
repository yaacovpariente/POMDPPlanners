"""Returns distribution plotting utilities.

This module provides functions for visualizing discounted returns distributions
across different POMDP policies and environments.
"""

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import History, history_to_discounted_return_value
from POMDPPlanners.utils.visualization.plot_utils import _safe_histplot

matplotlib.use("Agg")  # Use non-interactive backend

# Set up logger
logger = logging.getLogger(__name__)


def plot_reward_comparison(
    histories: List[History],
    environments: List[Environment],
    policies: List[Policy],
    cache_dir_path: Path,
) -> None:
    """
    Plot reward comparison across environments and policies.

    Args:
        histories: List of History objects containing episode data
        environments: List of environments
        policies: List of policies
        cache_dir_path: Path to save the plots
    """
    history_discounted_returns = [
        history_to_discounted_return_value(history) for history in histories
    ]


def plot_discounted_returns_histogram(
    histories: List[History],
    policy: Policy,
    environment: Environment,
    cache_path: Path,
) -> None:
    """
    Create a histogram plot of discounted returns from a list of histories using seaborn.

    Args:
        histories: List of History objects containing episode data
        policy: Policy object used to generate the histories
        environment: Environment object where the histories were generated
        cache_path: Path where the histogram plot will be saved
    """
    # Convert histories to discounted returns
    discounted_returns = [history_to_discounted_return_value(history) for history in histories]

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)

    # Create the figure and axis
    plt.figure(figsize=(10, 6))

    # Safely render histogram (handles degenerate ranges)
    plotted = _safe_histplot(
        discounted_returns,
        max_bins=15,
        color="skyblue",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    if not plotted:
        # Fallback text when no valid data to plot
        plt.text(
            0.5,
            0.5,
            "No valid data to plot",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,  # type: ignore[attr-defined]
        )

    # Customize the plot
    plt.xlabel(f"Discounted Return for {policy.name} in {environment.name}", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Discounted Returns", fontsize=14, pad=20)

    # Add a light grid
    plt.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(cache_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_discounted_returns_histogram_multiple_policies(
    histories: Dict[str, List[History]],
    policies: Sequence[Policy],
    environment: Environment,
    cache_path: Path,
) -> None:
    """
    Create overlapping histogram plots of discounted returns for multiple policies using seaborn.

    Args:
        histories: Dictionary mapping policy names to lists of History objects
        policies: Sequence of Policy objects
        environment: Environment object
        cache_path: Path where the histogram plot will be saved
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)

    # Create the figure and axis
    plt.figure(figsize=(12, 7))

    # Create a color palette
    colors = sns.color_palette("husl", n_colors=len(policies))

    # Plot histogram for each policy
    for policy, color in zip(policies, colors):
        policy_histories = histories[policy.name]
        if not policy_histories:  # Skip if no histories for this policy
            continue

        discounted_returns = [
            history_to_discounted_return_value(history) for history in policy_histories
        ]

        _ = _safe_histplot(
            discounted_returns,
            max_bins=15,
            color=color,
            alpha=0.5,
            edgecolor="black",
            linewidth=0.5,
            label=policy.name,
        )

    # Customize the plot
    plt.xlabel("Discounted Return", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(
        f"Distribution of Discounted Returns for {environment.name}",
        fontsize=14,
        pad=20,
    )

    # Add legend
    plt.legend(title="Policies", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add a light grid
    plt.grid(True, alpha=0.3)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(cache_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_environment_policy_pair_comparison(
    histories: List[History],
    policy: Policy,
    environment: Environment,
    cache_path: Path,
) -> None:
    """
    Create a histogram plot for a single environment-policy pair.

    Args:
        histories: List of History objects containing episode data
        policy: Policy object used to generate the histories
        environment: Environment object where the histories were generated
        cache_path: Path where the histogram plot will be saved
    """
    plot_discounted_returns_histogram(
        histories=histories,
        policy=policy,
        environment=environment,
        cache_path=cache_path,
    )
