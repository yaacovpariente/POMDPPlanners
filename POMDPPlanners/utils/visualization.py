from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import mlflow

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy

def plot_statistics_comparison(
    statistics: List[dict],
    environments: List[Environment],
    policies: List[Policy],
    cache_dir_path: Path
) -> None:
    """
    Plot bar plots comparing statistics across environments and policies.
    
    Args:
        statistics: List of statistics dictionaries for each environment-policy combination
        environments: List of environments
        policies: List of policies
        cache_dir_path: Path to save the plots
    """
    # Create plots directory if it doesn't exist
    plots_dir = cache_dir_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all statistic names from the first statistics dictionary
    stat_names = list(statistics[0].keys())
    
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
            mean, ci = statistics[i][stat_name]
            means.append(mean)
            lower_bounds.append(ci[0])
            upper_bounds.append(ci[1])
            labels.append(f"{env.__class__.__name__}\n{policy.__class__.__name__}")
        
        # Plot bars
        plt.bar(x, means, width, yerr=(np.array(upper_bounds) - np.array(lower_bounds)) / 2,
                capsize=5)
        
        # Customize the plot
        plt.xlabel('Environment-Policy Pair')
        plt.ylabel(stat_name.replace('_', ' ').title())
        plt.title(f'{stat_name.replace("_", " ").title()} Comparison')
        plt.xticks(x, labels, rotation=45, ha='right')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(plots_dir / f"{stat_name}_comparison.png")
        plt.close()
        
        # Log the plot to MLflow
        mlflow.log_artifact(str(plots_dir / f"{stat_name}_comparison.png")) 