#!/usr/bin/env python3
"""Generate POMDPPlanners Architecture Diagram.

This script creates a comprehensive architecture diagram showing the relationships
and interactions between different components in the POMDPPlanners project.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Set up the figure
fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis("off")

# Define colors for different component types
colors = {
    "core": "#E8F4F8",  # Light blue for core abstractions
    "environment": "#E8F5E9",  # Light green for environments
    "planner": "#FFF3E0",  # Light orange for planners
    "belief": "#F3E5F5",  # Light purple for beliefs
    "simulation": "#FFF9C4",  # Light yellow for simulation
    "utils": "#FFEBEE",  # Light red for utilities
    "config": "#E0F2F1",  # Light teal for config
}

edge_colors = {
    "inherit": "#1976D2",  # Blue for inheritance
    "compose": "#388E3C",  # Green for composition
    "uses": "#F57C00",  # Orange for usage
    "creates": "#7B1FA2",  # Purple for creation
}


def draw_box(ax, x, y, width, height, text, color, fontsize=9, fontweight="normal"):
    """Draw a rounded box with text."""
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.05",
        edgecolor="black",
        facecolor=color,
        linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        zorder=3,
    )


def draw_arrow(ax, x1, y1, x2, y2, color, style="solid", width=1.5, label=""):
    """Draw an arrow between components."""
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->" if style == "solid" else "->",
        color=color,
        linewidth=width,
        linestyle=style,
        mutation_scale=20,
        zorder=1,
    )
    ax.add_patch(arrow)


# Title
ax.text(
    10, 13.5, "POMDPPlanners Architecture", ha="center", va="center", fontsize=20, fontweight="bold"
)

# ===== CORE LAYER (Top) =====
core_y = 11.5
core_height = 1.2

# Environment Core
draw_box(
    ax,
    0.5,
    core_y,
    2.8,
    core_height,
    "Environment (ABC)\n\n• state_transition\n• observation_model\n• reward\n• is_terminal",
    colors["core"],
    fontsize=8,
    fontweight="bold",
)

# Policy Core
draw_box(
    ax,
    3.8,
    core_y,
    2.8,
    core_height,
    "Policy (ABC)\n\n• action(belief)\n• space_info",
    colors["core"],
    fontsize=8,
    fontweight="bold",
)

# Belief Core
draw_box(
    ax,
    7.1,
    core_y,
    2.8,
    core_height,
    "Belief (ABC)\n\n• update(action, obs)\n• sample()",
    colors["core"],
    fontsize=8,
    fontweight="bold",
)

# Tree Core
draw_box(
    ax,
    10.4,
    core_y,
    2.8,
    core_height,
    "Tree Nodes\n\n• BeliefNode\n• ActionNode",
    colors["core"],
    fontsize=8,
    fontweight="bold",
)

# Distribution Core
draw_box(
    ax,
    13.7,
    core_y,
    2.8,
    core_height,
    "Distribution (ABC)\n\n• sample()\n• probability()",
    colors["core"],
    fontsize=8,
    fontweight="bold",
)

# Config Core
draw_box(
    ax,
    17,
    core_y,
    2.5,
    core_height,
    "Config Types\n\n• EnvironmentConfig\n• PolicyConfig\n• BeliefConfig",
    colors["config"],
    fontsize=8,
    fontweight="bold",
)

# ===== IMPLEMENTATION LAYER =====
impl_y = 9.2
impl_height = 1.5

# Concrete Environments
draw_box(
    ax,
    0.5,
    impl_y,
    2.8,
    impl_height,
    "Environments\n\n• TigerPOMDP\n• CartPolePOMDP\n• LightDarkPOMDP\n• PushPOMDP\n• RockSamplePOMDP",
    colors["environment"],
    fontsize=7,
)

# Concrete Planners
draw_box(
    ax,
    3.8,
    impl_y,
    2.8,
    impl_height,
    "Planners\n\n• POMCP\n• POMCP_DPW\n• PFT_DPW\n• SparsePFT\n• SparseSampling",
    colors["planner"],
    fontsize=7,
)

# Concrete Beliefs
draw_box(
    ax,
    7.1,
    impl_y,
    2.8,
    impl_height,
    "Belief Implementations\n\n• UnweightedParticleBelief\n• WeightedParticleBelief\n• WeightedParticleReinvig",
    colors["belief"],
    fontsize=7,
)

# Path Simulation Policy
draw_box(
    ax,
    10.4,
    impl_y,
    2.8,
    impl_height,
    "PathSimulationPolicy\n(ABC)\n\n• _simulate_path()\n• _build_tree()",
    colors["planner"],
    fontsize=7,
)

# Observation/State Models
draw_box(
    ax,
    13.7,
    impl_y,
    2.8,
    impl_height,
    "Models\n\n• ObservationModel\n• StateTransitionModel\n• DiscreteDistribution",
    colors["core"],
    fontsize=7,
)

# ===== SIMULATION LAYER =====
sim_y = 6.5
sim_height = 2

# Simulator
draw_box(
    ax,
    1.5,
    sim_y,
    3.5,
    sim_height,
    "BaseSimulator\n\n• Episode execution\n• History collection\n• Statistics computation\n• Parallel execution\n• MLflow integration",
    colors["simulation"],
    fontsize=7,
)

# Task Management
draw_box(
    ax,
    5.5,
    sim_y,
    3.5,
    sim_height,
    "Task Management\n\n• SimulationTask\n• TaskManager\n• EpisodeSimulationTask\n• HyperParameterTuningTask",
    colors["simulation"],
    fontsize=7,
)

# Simulation APIs
draw_box(
    ax,
    9.5,
    sim_y,
    3.5,
    sim_height,
    "Simulation APIs\n\n• LocalSimulationsAPI\n• DaskSimulationsAPI\n• PBSSimulationsAPI",
    colors["simulation"],
    fontsize=7,
)

# Workflows
draw_box(
    ax,
    13.5,
    sim_y,
    3.5,
    sim_height,
    "Workflows\n\n• planner_evaluation\n• optimization\n• hyperparameter_tuning\n• integrated",
    colors["simulation"],
    fontsize=7,
)

# ===== DATA LAYER =====
data_y = 4.2
data_height = 1.5

# History & StepData
draw_box(
    ax,
    1.5,
    data_y,
    3,
    data_height,
    "Data Structures\n\n• StepData\n• History\n• MetricValue\n• PolicyRunData",
    colors["config"],
    fontsize=7,
)

# Caching
draw_box(
    ax,
    5,
    data_y,
    3,
    data_height,
    "Caching Layer\n\n• DataBaseInterface\n• InMemoryDB\n• FileBasedDB\n• config_id hashing",
    colors["utils"],
    fontsize=7,
)

# Statistics
draw_box(
    ax,
    8.5,
    data_y,
    3,
    data_height,
    "Statistics\n\n• compute_statistics\n• confidence_intervals\n• risk_metrics\n• metrics_to_dataframe",
    colors["utils"],
    fontsize=7,
)

# Visualization
draw_box(
    ax,
    12,
    data_y,
    3,
    data_height,
    "Visualization\n\n• tree_statistics\n• episode_visualization\n• planner_comparison\n• result_plotting",
    colors["utils"],
    fontsize=7,
)

# ===== UTILITIES LAYER =====
utils_y = 2
utils_height = 1.5

# Logger
draw_box(
    ax,
    1.5,
    utils_y,
    2.3,
    utils_height,
    "Logger\n\n• Centralized logging\n• Queue-based\n• Debug modes",
    colors["utils"],
    fontsize=7,
)

# Config Loader
draw_box(
    ax,
    4.2,
    utils_y,
    2.3,
    utils_height,
    "Config Loader\n\n• YAML loading\n• Config validation\n• from_config()",
    colors["utils"],
    fontsize=7,
)

# Distributions Utils
draw_box(
    ax,
    7,
    utils_y,
    2.3,
    utils_height,
    "Distribution Utils\n\n• Action samplers\n• Particle utilities\n• DPW helpers",
    colors["utils"],
    fontsize=7,
)

# Memory & Computing
draw_box(
    ax,
    9.8,
    utils_y,
    2.3,
    utils_height,
    "Computing Utils\n\n• Memory tracking\n• Distributed computing\n• Ray/Dask support",
    colors["utils"],
    fontsize=7,
)

# Other Utils
draw_box(
    ax,
    12.6,
    utils_y,
    2.3,
    utils_height,
    "Other Utils\n\n• config_to_id\n• tree_statistics\n• rollout policies",
    colors["utils"],
    fontsize=7,
)

# ===== DRAW RELATIONSHIPS =====

# Core inheritance relationships
draw_arrow(
    ax, 1.9, impl_y + impl_height, 1.9, core_y, edge_colors["inherit"], width=2
)  # Environment
draw_arrow(ax, 5.2, impl_y + impl_height, 5.2, core_y, edge_colors["inherit"], width=2)  # Policy
draw_arrow(ax, 8.5, impl_y + impl_height, 8.5, core_y, edge_colors["inherit"], width=2)  # Belief
draw_arrow(
    ax, 11.8, impl_y + impl_height, 11.8, core_y, edge_colors["inherit"], width=2
)  # PathSim from Tree
draw_arrow(ax, 15.1, impl_y + impl_height, 15.1, core_y, edge_colors["inherit"], width=2)  # Models

# PathSimulationPolicy inherits from Policy
draw_arrow(
    ax,
    10.4,
    impl_y + impl_height / 2,
    6.6,
    impl_y + impl_height / 2,
    edge_colors["inherit"],
    style="dashed",
    width=1.5,
)

# Planners use PathSimulationPolicy
draw_arrow(
    ax, 5.2, impl_y, 11.8, impl_y + impl_height, edge_colors["inherit"], style="dashed", width=1.5
)

# Policy uses Environment
draw_arrow(
    ax,
    3.8,
    core_y + core_height / 2,
    3.3,
    core_y + core_height / 2,
    edge_colors["compose"],
    width=2,
)

# Policy uses Belief
draw_arrow(
    ax,
    6.6,
    core_y + core_height / 2,
    7.1,
    core_y + core_height / 2,
    edge_colors["compose"],
    width=2,
)

# PathSimulation uses Tree
draw_arrow(ax, 11.8, impl_y + impl_height, 11.8, core_y, edge_colors["uses"], width=2)

# Belief uses Environment models
draw_arrow(
    ax, 8.5, core_y, 8.5, impl_y + impl_height, edge_colors["uses"], style="dashed", width=1.5
)

# Simulator uses everything
draw_arrow(ax, 3, sim_y, 1.9, impl_y + impl_height, edge_colors["uses"], width=1.5)
draw_arrow(ax, 3, sim_y, 5.2, impl_y + impl_height, edge_colors["uses"], width=1.5)
draw_arrow(ax, 3, sim_y, 8.5, impl_y + impl_height, edge_colors["uses"], width=1.5)

# Config creates components
draw_arrow(
    ax, 17, core_y, 16, impl_y + impl_height, edge_colors["creates"], style="dashed", width=1.5
)
draw_arrow(
    ax, 18, core_y, 6.6, impl_y + impl_height, edge_colors["creates"], style="dashed", width=1.5
)

# Simulator creates Data
draw_arrow(ax, 3, sim_y, 3, data_y + data_height, edge_colors["creates"], width=1.5)

# Task Management uses Simulator
draw_arrow(ax, 5.5, sim_y, 5, sim_y, edge_colors["uses"], width=1.5)

# Workflows use APIs
draw_arrow(
    ax, 13.5, sim_y + sim_height / 2, 13, sim_y + sim_height / 2, edge_colors["uses"], width=1.5
)

# APIs use Task Management
draw_arrow(
    ax, 9.5, sim_y + sim_height / 2, 9, sim_y + sim_height / 2, edge_colors["uses"], width=1.5
)

# Statistics uses Data
draw_arrow(
    ax, 8.5, data_y, 4.5, data_y + data_height, edge_colors["uses"], style="dashed", width=1.5
)

# Visualization uses Data & Statistics
draw_arrow(ax, 12, data_y, 11.5, data_y, edge_colors["uses"], style="dashed", width=1.5)

# Simulator uses Utils
draw_arrow(ax, 3, sim_y, 3, utils_y + utils_height, edge_colors["uses"], style="dotted", width=1)

# Config Loader used by all
draw_arrow(ax, 5.5, utils_y + utils_height, 7, sim_y, edge_colors["uses"], style="dotted", width=1)

# ===== LEGEND =====
legend_elements = [
    mpatches.Patch(color=colors["core"], label="Core Abstractions (ABC)"),
    mpatches.Patch(color=colors["environment"], label="Environment Implementations"),
    mpatches.Patch(color=colors["planner"], label="Planner Implementations"),
    mpatches.Patch(color=colors["belief"], label="Belief Implementations"),
    mpatches.Patch(color=colors["simulation"], label="Simulation Framework"),
    mpatches.Patch(color=colors["config"], label="Configuration & Data"),
    mpatches.Patch(color=colors["utils"], label="Utilities"),
    mlines.Line2D([], [], color=edge_colors["inherit"], linewidth=2, label="Inheritance"),
    mlines.Line2D([], [], color=edge_colors["compose"], linewidth=2, label="Composition/Has-a"),
    mlines.Line2D(
        [],
        [],
        color=edge_colors["uses"],
        linewidth=1.5,
        linestyle="dashed",
        label="Uses/Depends on",
    ),
    mlines.Line2D(
        [],
        [],
        color=edge_colors["creates"],
        linewidth=1.5,
        linestyle="dashed",
        label="Creates/Instantiates",
    ),
]

ax.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=4,
    fontsize=8,
    bbox_to_anchor=(0.5, -0.08),
    framealpha=0.9,
)

# Add layer labels
ax.text(
    0.2,
    core_y + core_height / 2,
    "CORE\nABSTRACTIONS",
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    rotation=90,
)
ax.text(
    0.2,
    impl_y + impl_height / 2,
    "IMPLEMENTATIONS",
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    rotation=90,
)
ax.text(
    0.2,
    sim_y + sim_height / 2,
    "SIMULATION\nFRAMEWORK",
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    rotation=90,
)
ax.text(
    0.2,
    data_y + data_height / 2,
    "DATA LAYER",
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    rotation=90,
)
ax.text(
    0.2,
    utils_y + utils_height / 2,
    "UTILITIES",
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    rotation=90,
)

plt.tight_layout()
plt.savefig("pomdp_architecture_diagram.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("pomdp_architecture_diagram.pdf", bbox_inches="tight", facecolor="white")
print("Architecture diagram saved as:")
print("  - pomdp_architecture_diagram.png")
print("  - pomdp_architecture_diagram.pdf")
