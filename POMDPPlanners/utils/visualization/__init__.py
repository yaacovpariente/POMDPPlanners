# SPDX-License-Identifier: MIT

"""Visualization utilities for POMDPPlanners.

This module provides plotting and visualization functions for analyzing
POMDP planning results, including metrics comparison, returns distributions,
tree visualizations, and policy simulations.
"""

# Import from metrics_plots
from POMDPPlanners.utils.visualization.metrics_plots import (
    plot_metrics_comparison,
    plot_policies_comparison_on_environment,
)

# Import from returns_plots
from POMDPPlanners.utils.visualization.returns_plots import (
    plot_discounted_returns_histogram,
    plot_discounted_returns_histogram_multiple_policies,
    plot_environment_policy_pair_comparison,
)

# Import from tree_plots
from POMDPPlanners.utils.visualization.tree_plots import plot_tree_graphs

# Import from policy_simulation_plots
from POMDPPlanners.utils.visualization.policy_simulation_plots import (
    AgentPath,
    plot_policy_returns,
)

# Import from plot_utils (internal utilities, prefixed with underscore)
from POMDPPlanners.utils.visualization.plot_utils import _log_or_print, _safe_histplot

__all__ = [
    # Metrics plotting
    "plot_metrics_comparison",
    "plot_policies_comparison_on_environment",
    # Returns plotting
    "plot_discounted_returns_histogram",
    "plot_discounted_returns_histogram_multiple_policies",
    "plot_environment_policy_pair_comparison",
    # Tree plotting
    "plot_tree_graphs",
    # Policy simulation plotting
    "plot_policy_returns",
    "AgentPath",
    # Utilities (internal, but exported for backward compatibility)
    "_safe_histplot",
    "_log_or_print",
]
