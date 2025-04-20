from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import mlflow

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import MetricValue

def plot_statistics_comparison(
    statistics: List[List[MetricValue]],
    environments: List[Environment],
    policies: List[Policy],
    cache_dir_path: Path
) -> None:
    """
    Plot bar plots comparing statistics across environments and policies.
    
    Args:
        statistics: List of lists of MetricValue objects for each environment-policy combination
        environments: List of environments
        policies: List of policies
        cache_dir_path: Path to save the plots
    """
    assert len(statistics) > 0 and len(environments) > 0 and len(policies) > 0, \
        "Statistics, environments, and policies lists must not be empty"

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
                
            if np.isnan(metric.value) or np.isnan(metric.lower_confidence_bound) or np.isnan(metric.upper_confidence_bound):
                continue
                
            means.append(metric.value)
            lower_bounds.append(metric.lower_confidence_bound)
            upper_bounds.append(metric.upper_confidence_bound)
            labels.append(f"{env.__class__.__name__}\n{policy.__class__.__name__}")
        
        if not means:  # Skip if no valid data points
            plt.close()
            continue
            
        # Plot bars
        plt.bar(x[:len(means)], means, width, yerr=(np.array(upper_bounds) - np.array(lower_bounds)) / 2,
                capsize=5)
        
        # Customize the plot
        plt.xlabel('Environment-Policy Pair')
        plt.ylabel(stat_name.replace('_', ' ').title())
        plt.title(f'{stat_name.replace("_", " ").title()} Comparison')
        plt.xticks(x[:len(means)], labels, rotation=45, ha='right')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(plots_dir / f"{stat_name}_comparison.png")
        plt.close()
        
        # Log the plot to MLflow
        mlflow.log_artifact(str(plots_dir / f"{stat_name}_comparison.png")) 