from typing import List, Dict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns

matplotlib.use("Agg")  # Use non-interactive backend
import mlflow

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import MetricValue, History, history_to_discounted_return_value

def plot_metrics_comparison(
    statistics: List[List[MetricValue]],
    environments: List[Environment],
    policies: List[Policy],
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
    assert (
        len(statistics) > 0 and len(environments) > 0 and len(policies) > 0
    ), "Statistics, environments, and policies lists must not be empty"

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
        plt.bar(
            x[: len(means)],
            means,
            width,
            yerr=(np.array(upper_bounds) - np.array(lower_bounds)) / 2,
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


def plot_reward_comparison(
    histories: List[History],
    environments: List[Environment],
    policies: List[Policy],
    cache_dir_path: Path,
) -> None:
    history_discounted_returns = [history_to_discounted_return_value(history) for history in histories]
    

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
        cache_path: Path where the histogram plot will be saved
    """
    # Convert histories to discounted returns
    discounted_returns = [history_to_discounted_return_value(history) for history in histories]
    
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    
    # Create the figure and axis
    plt.figure(figsize=(10, 6))
    
    # Create the histogram using seaborn
    sns.histplot(
        data=discounted_returns,
        bins=15,
        edgecolor='black',
        color='skyblue',
        alpha=0.7
    )
    
    # Customize the plot
    plt.xlabel(f'Discounted Return for {policy.name} in {environment.name}', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Discounted Returns', fontsize=14, pad=20)
    
    # Add a light grid
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(cache_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_discounted_returns_histogram_multiple_policies(
    histories: Dict[str, List[History]],
    policies: List[Policy],
    environment: Environment,
    cache_path: Path,
) -> None:
    """
    Create overlapping histogram plots of discounted returns for multiple policies using seaborn.

    Args:
        histories: Dictionary mapping policy names to lists of History objects
        policies: List of Policy objects
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
            
        discounted_returns = [history_to_discounted_return_value(history) for history in policy_histories]
        
        sns.histplot(
            data=discounted_returns,
            bins=15,
            alpha=0.5,
            color=color,
            label=policy.name,
            edgecolor='black',
            linewidth=0.5
        )
    
    # Customize the plot
    plt.xlabel('Discounted Return', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of Discounted Returns for {environment.name}', fontsize=14, pad=20)
    
    # Add legend
    plt.legend(title='Policies', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add a light grid
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(cache_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_environment_policy_pair_comparison(
    histories: List[History],
    policy: Policy,
    environment: Environment,
    cache_path: Path,
) -> None:
    plot_discounted_returns_histogram(histories=histories, policy=policy, environment=environment, cache_path=cache_path)

def plot_policies_comparison_on_environment(
    histories: List[List[History]],
    environments: List[Environment],
    policies: List[Policy],
    cache_path: Path,
) -> None:
    pass
