"""Tree visualization utilities.

This module provides functions for visualizing POMDP belief trees with interactive
Plotly graphs showing node values and visit counts.
"""

import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from POMDPPlanners.core.tree import ActionNode, BeliefNode

# Set up logger
logger = logging.getLogger(__name__)


def _create_hierarchical_layout(all_nodes: List[Any]) -> Dict[Any, Tuple[float, float]]:
    """Create a hierarchical layout for tree nodes."""
    pos = {}
    for node in all_nodes:
        depth = node.depth
        siblings = [n for n in all_nodes if n.depth == depth]
        x = (siblings.index(node) - (len(siblings) - 1) / 2) * 0.3
        y = 1 - depth * 0.2
        pos[node] = (x, y)
    return pos


def _create_edge_trace(all_nodes: List[Any], pos: Dict[Any, Tuple[float, float]]) -> go.Scatter:
    """Create edge trace for connecting parent and child nodes."""
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

    return go.Scatter(
        x=edge_x,
        y=edge_y,
        line={"width": 1, "color": "#888"},
        hoverinfo="none",
        mode="lines",
    )


def _extract_node_info(node: Any) -> Tuple[str, float, str, str]:
    """Extract information from a node for visualization."""
    if isinstance(node, BeliefNode):
        node_type = "Belief"
        value = node.v_value
        value_type = "v_value"
        if hasattr(node, "observation") and node.observation is not None:
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

    return node_type, value, value_type, node_info


def _create_hover_text(
    node: Any,
    node_type: str,
    node_info: str,
    value_type: str,
    value: float,
) -> str:
    """Create hover text for a node."""
    hover_text = (
        f"{node_type} Node<br>"
        f"{node_info}<br>"
        f"{value_type}: {value:.3f}<br>"
        f"Visits: {node.visit_count}<br>"
        f"Depth: {node.depth}"
    )
    if node.parent:
        if isinstance(node.parent, ActionNode):
            hover_text += f"<br>Parent Action: {node.parent.action}"
        else:
            if isinstance(node.parent.observation, np.ndarray):
                hover_text += f"<br>Parent: [{node.parent.observation[0]:.1f}, {node.parent.observation[1]:.1f}]"
            else:
                hover_text += (
                    f"<br>Parent: {node.parent.observation if node.parent.observation else 'Root'}"
                )

    return hover_text


def _collect_node_data(all_nodes: List[Any], pos: Dict[Any, Tuple[float, float]]) -> Tuple[
    List[float],
    List[float],
    List[str],
    List[float],
    List[int],
    List[float],
    List[str],
]:
    """Collect data from all nodes for visualization."""
    node_x = []
    node_y = []
    node_text = []
    node_values = []
    node_visits = []
    node_sizes = []
    node_labels = []

    for node in all_nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Extract node information
        node_type, value, value_type, node_info = _extract_node_info(node)

        # Create hover text
        hover_text = _create_hover_text(node, node_type, node_info, value_type, value)

        node_text.append(hover_text)
        node_values.append(value)
        node_visits.append(node.visit_count)
        node_sizes.append(20)

        # Create node label (just the value)
        node_labels.append(f"{value:.2f}")

    return node_x, node_y, node_text, node_values, node_visits, node_sizes, node_labels


def _create_node_trace(
    node_x: List[float],
    node_y: List[float],
    node_text: List[str],
    node_labels: List[str],
    node_sizes: List[float],
    node_values: List[float],
    colorscale: str,
    colorbar_title: str,
    colorbar_x: float,
    name: str,
) -> go.Scatter:
    """Create a scatter trace for nodes."""
    return go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_labels,
        textposition="middle center",
        textfont={"size": 10, "color": "white"},
        marker={
            "size": node_sizes,
            "color": node_values,
            "colorscale": colorscale,
            "showscale": True,
            "colorbar": {"title": colorbar_title, "x": colorbar_x},
            "line_width": 2,
        },
        name=name,
        customdata=node_text,
        hovertemplate="%{customdata}<extra></extra>",
    )


def _add_traces_to_figure(
    fig: go.Figure,
    edge_trace: go.Scatter,
    node_trace_values: go.Scatter,
    node_trace_visits: go.Scatter,
) -> None:
    """Add all traces to the figure subplots."""
    fig.add_trace(edge_trace, row=1, col=1)
    fig.add_trace(edge_trace, row=1, col=2)
    fig.add_trace(node_trace_values, row=1, col=1)
    fig.add_trace(node_trace_visits, row=1, col=2)


def _configure_figure_layout(fig: go.Figure, num_nodes: int) -> None:
    """Configure the overall figure layout."""
    fig.update_layout(
        title_text=f"Tiger POMDP Tree Visualization - Total Nodes: {num_nodes}",
        showlegend=False,
        hovermode="closest",
        width=1800,
        height=800,
        dragmode="pan",
        modebar_add=["zoom", "pan", "reset", "zoomIn", "zoomOut"],
    )

    # Update axes
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)


def _print_tree_statistics(
    all_nodes: List[Any], node_values: List[float], node_visits: List[int]
) -> None:
    """Print tree statistics to console."""
    print(f"Total number of nodes: {len(all_nodes)}")
    print(f"Tree depth: {max(node.depth for node in all_nodes)}")
    print(f"Number of leaf nodes: {len([node for node in all_nodes if not node.children])}")
    print(f"Value range: [{min(node_values):.3f}, {max(node_values):.3f}]")
    print(f"Visit count range: [{min(node_visits)}, {max(node_visits)}]")


def plot_tree_graphs(root_node: BeliefNode):
    """
    Create two interactive visualizations of the belief tree:
    1. Node visit counts
    2. Node values (v_value for belief nodes, q_value for action nodes)

    Args:
        root_node: Root node of the belief tree
    """
    # Create custom hierarchical layout
    all_nodes = list(root_node.descendants) + [root_node]
    pos = _create_hierarchical_layout(all_nodes)

    # Create edge trace
    edge_trace = _create_edge_trace(all_nodes, pos)

    # Collect node data
    node_x, node_y, node_text, node_values, node_visits, node_sizes, node_labels = (
        _collect_node_data(all_nodes, pos)
    )

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Node Values (v_value/q_value)", "Visit Counts"),
        horizontal_spacing=0.1,
    )

    # Create node traces
    node_trace_values = _create_node_trace(
        node_x=node_x,
        node_y=node_y,
        node_text=node_text,
        node_labels=node_labels,
        node_sizes=node_sizes,
        node_values=node_values,
        colorscale="Viridis",
        colorbar_title="Node Value",
        colorbar_x=0.45,
        name="Values",
    )

    # Create visit count labels
    visit_labels = [f"n={v}" for v in node_visits]

    node_trace_visits = _create_node_trace(
        node_x=node_x,
        node_y=node_y,
        node_text=node_text,
        node_labels=visit_labels,
        node_sizes=node_sizes,
        node_values=[float(v) for v in node_visits],
        colorscale="RdBu",
        colorbar_title="Visit Count",
        colorbar_x=1.0,
        name="Visits",
    )

    # Add traces to figure
    _add_traces_to_figure(fig, edge_trace, node_trace_values, node_trace_visits)

    # Configure layout
    _configure_figure_layout(fig, len(all_nodes))

    # Show the plot
    fig.show()

    # Print statistics
    _print_tree_statistics(all_nodes, node_values, node_visits)
