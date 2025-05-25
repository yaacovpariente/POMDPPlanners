from typing import List, Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
from joblib import Parallel, delayed
import plotly.graph_objects as go
from plotly.subplots import make_subplots

matplotlib.use("Agg")  # Use non-interactive backend
import mlflow

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import MetricValue, History, history_to_discounted_return_value
from POMDPPlanners.core.tree import BeliefNode, ActionNode
from POMDPPlanners.core.belief import WeightedParticleBelief, Belief
from POMDPPlanners.core.cost import belief_expectation_cost


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
        yerr = (np.array(upper_bounds) - np.array(lower_bounds)) / 2
        if sum(yerr) == 0:
            yerr = 1e-10
            
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

def plot_tree_graphs(root_node: BeliefNode):
    """
    Create two interactive visualizations of the belief tree:
    1. Node visit counts
    2. Node values (v_value for belief nodes, q_value for action nodes)
    
    Args:
        root_node (BeliefNode): Root node of the belief tree
    """
    # Create custom hierarchical layout
    pos = {}
    all_nodes = [node for node in root_node.descendants] + [root_node]
    
    for node in all_nodes:
        depth = node.depth
        siblings = [n for n in all_nodes if n.depth == depth]
        x = (siblings.index(node) - (len(siblings) - 1) / 2) * 0.3
        y = 1 - depth * 0.2
        pos[node] = (x, y)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for node in all_nodes:
        if node.parent:
            x0, y0 = pos[node.parent]
            x1, y1 = pos[node]
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            edge_x.extend([x0, cx, x1, None])
            edge_y.extend([y0, cy, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces for both graphs
    node_x = []
    node_y = []
    node_text = []
    node_values = []
    node_visits = []
    node_sizes = []
    node_labels = []  # New list for node labels
    
    for node in all_nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Create hover text
        if isinstance(node, BeliefNode):
            node_type = "Belief"
            value = node.v_value
            value_type = "v_value"
            if hasattr(node, 'observation') and node.observation is not None:
                if isinstance(node.observation, np.ndarray):
                    node_info = f"Obs: [{node.observation[0]:.1f}, {node.observation[1]:.1f}]"
                else:
                    node_info = f"Obs: {node.observation}"
            else:
                node_info = "Root"
        else:  # ActionNode
            node_type = "Action"
            value = node.q_value
            value_type = "q_value"
            node_info = f"Action: {node.action}"
        
        hover_text = (f"{node_type} Node<br>"
                     f"{node_info}<br>"
                     f"{value_type}: {value:.3f}<br>"
                     f"Visits: {node.visit_count}<br>"
                     f"Depth: {node.depth}")
        if node.parent:
            if isinstance(node.parent, ActionNode):
                hover_text += f"<br>Parent Action: {node.parent.action}"
            else:
                if isinstance(node.parent.observation, np.ndarray):
                    hover_text += f"<br>Parent: [{node.parent.observation[0]:.1f}, {node.parent.observation[1]:.1f}]"
                else:
                    hover_text += f"<br>Parent: {node.parent.observation if node.parent.observation else 'Root'}"
        
        node_text.append(hover_text)
        node_values.append(value)
        node_visits.append(node.visit_count)
        node_sizes.append(20)
        
        # Create node label (just the value)
        if isinstance(node, BeliefNode):
            node_labels.append(f"{value:.2f}")  # v_value with 2 decimal places
        else:
            node_labels.append(f"{value:.2f}")  # q_value with 2 decimal places
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Node Values (v_value/q_value)', 'Visit Counts'),
        horizontal_spacing=0.1
    )
    
    # Add value-based node trace
    node_trace_values = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        textfont=dict(size=10, color='white'),
        marker=dict(
            size=node_sizes,
            color=node_values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Node Value', x=0.45),
            line_width=2
        ),
        name='Values'
    )
    
    # Add visit count-based node trace
    node_trace_visits = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        textfont=dict(size=10, color='white'),
        marker=dict(
            size=node_sizes,
            color=node_visits,
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(title='Visit Count', x=1.0),
            line_width=2
        ),
        name='Visits'
    )
    
    # Add value labels to both plots
    node_trace_values.text = node_labels
    node_trace_visits.text = [f"n={v}" for v in node_visits]  # Show visit counts in right plot
    
    # Add traces to both subplots
    fig.add_trace(edge_trace, row=1, col=1)
    fig.add_trace(edge_trace, row=1, col=2)
    fig.add_trace(node_trace_values, row=1, col=1)
    fig.add_trace(node_trace_visits, row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title_text=f'Tiger POMDP Tree Visualization - Total Nodes: {len(all_nodes)}',
        showlegend=False,
        hovermode='closest',
        width=1800,
        height=800,
        dragmode='pan',
        modebar_add=['zoom', 'pan', 'reset', 'zoomIn', 'zoomOut']
    )
    
    # Update axes
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    
    # Show the plot
    fig.show()
    
    # Print statistics
    print(f"Total number of nodes: {len(all_nodes)}")
    print(f"Tree depth: {max(node.depth for node in all_nodes)}")
    print(f"Number of leaf nodes: {len([node for node in all_nodes if not node.children])}")
    print(f"Value range: [{min(node_values):.3f}, {max(node_values):.3f}]")
    print(f"Visit count range: [{min(node_visits)}, {max(node_visits)}]")

from dataclasses import dataclass

@dataclass
class AgentPath:
    """Data class to store agent path."""
    name: str
    state_sequence: List[Any]
    action_sequence: List[Any]
    n_particles: int


def plot_policy_returns(
    env: Environment,
    agent_paths: List[AgentPath],
    dir_path: Path,
    n_samples: int = 1000,
    n_jobs: int = -1,
) -> None:
    """
    Simulate and plot returns for multiple agent paths.
    
    Args:
        env: POMDP environment
        agent_paths: List of AgentPath objects containing path information
        dir_path: Directory path to save the plot
        n_samples: Number of simulations to run for each path
        n_jobs: Number of parallel jobs to run (-1 for all cores)
        
    Raises:
        ValueError: If any of the input parameters are invalid
        TypeError: If any of the input parameters are of incorrect type
    """
    # Input validation
    if not isinstance(env, Environment):
        raise TypeError("env must be an instance of Environment")
    if not isinstance(agent_paths, list):
        raise TypeError("agent_paths must be a list")
    if not isinstance(dir_path, Path):
        raise TypeError("dir_path must be a Path object")
    if not isinstance(n_samples, int):
        raise TypeError("n_samples must be an integer")
    if not isinstance(n_jobs, int):
        raise TypeError("n_jobs must be an integer")
        
    if not agent_paths:
        raise ValueError("agent_paths cannot be empty")
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than 0")
    if n_jobs < -1:
        raise ValueError("n_jobs must be -1 or greater")
        
    # Validate each agent path
    for i, path in enumerate(agent_paths):
        if not isinstance(path, AgentPath):
            raise TypeError(f"agent_paths[{i}] must be an instance of AgentPath")
        if not isinstance(path.name, str):
            raise TypeError(f"agent_paths[{i}].name must be a string")
        if not isinstance(path.state_sequence, list):
            raise TypeError(f"agent_paths[{i}].state_sequence must be a list")
        if not isinstance(path.action_sequence, list):
            raise TypeError(f"agent_paths[{i}].action_sequence must be a list")
        if not isinstance(path.n_particles, int):
            raise TypeError(f"agent_paths[{i}].n_particles must be an integer")
        if path.n_particles <= 0:
            raise ValueError(f"agent_paths[{i}].n_particles must be greater than 0")
        if len(path.state_sequence) != len(path.action_sequence):
            raise ValueError(f"agent_paths[{i}] has mismatched state and action sequence lengths")
    
    # Create directory if it doesn't exist
    dir_path.mkdir(parents=True, exist_ok=True)
        
    def simulate_sequence(agent_path: AgentPath):
        total_reward = 0
        
        for i in range(len(agent_path.action_sequence)):
            # Create a weighted particle belief centered on the current state
            particles = [agent_path.state_sequence[i]] * agent_path.n_particles  # Two identical particles
            log_weights = np.log(np.array(np.ones(agent_path.n_particles) / agent_path.n_particles))  # One with log(1), one with log(exp(-1))
            belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)
            
            # Use belief_expectation_cost to compute the reward
            total_reward += -belief_expectation_cost(belief=belief, action=agent_path.action_sequence[i], env=env)
        
        return total_reward

    def run_simulation(path_idx):
        return simulate_sequence(agent_paths[path_idx])

    # Run simulations in parallel
    all_returns = []
    for i in range(len(agent_paths)):
        returns = Parallel(n_jobs=n_jobs)(
            delayed(run_simulation)(i) for _ in range(n_samples)
        )
        all_returns.append(returns)

    # Create the plot
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, (returns, agent_path) in enumerate(zip(all_returns, agent_paths)):
        sns.histplot(data=returns, label=agent_path.name, alpha=0.5, color=colors[i % len(colors)])
    
    plt.xlabel('Total Reward')
    plt.ylabel('Count')
    plt.title('Comparison of Returns for Different Agent Paths')
    plt.legend()
    
    # Save the plot
    output_path = dir_path / "policy_returns_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
