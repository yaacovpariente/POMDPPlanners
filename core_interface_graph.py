#!/usr/bin/env python3
"""Generate Core Interface Relations Graph for POMDPPlanners.

This script creates a detailed graph showing how the core interfaces
(Environment, Policy, Belief, Distribution, Tree) interact with each other.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.lines as mlines

# Set up the figure
fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis("off")

# Define colors
colors = {
    "abstract": "#E3F2FD",  # Light blue for abstract classes
    "concrete": "#E8F5E9",  # Light green for concrete classes
    "data": "#FFF3E0",  # Light orange for data structures
    "enum": "#F3E5F5",  # Light purple for enums
    "relation": "#FFEBEE",  # Light red for special relations
}

edge_colors = {
    "inherit": "#1976D2",  # Blue for inheritance
    "compose": "#388E3C",  # Green for composition/has-a
    "uses": "#F57C00",  # Orange for uses/depends on
    "returns": "#7B1FA2",  # Purple for returns
    "implements": "#C62828",  # Red for implements
}


def draw_box(
    ax,
    x,
    y,
    width,
    height,
    text,
    color,
    fontsize=9,
    fontweight="normal",
    linestyle="solid",
    linewidth=2,
):
    """Draw a rounded box with text."""
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.05",
        edgecolor="black",
        facecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
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


def draw_arrow(ax, x1, y1, x2, y2, color, style="solid", width=2, label="", label_offset=(0, 0.1)):
    """Draw an arrow between components with optional label."""
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        color=color,
        linewidth=width,
        linestyle=style,
        mutation_scale=20,
        zorder=1,
    )
    ax.add_patch(arrow)

    if label:
        mid_x = (x1 + x2) / 2 + label_offset[0]
        mid_y = (y1 + y2) / 2 + label_offset[1]
        ax.text(
            mid_x,
            mid_y,
            label,
            ha="center",
            va="center",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            zorder=4,
        )


# Title
ax.text(
    10,
    13.5,
    "POMDPPlanners Core Interface Relations",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
)

# ===== ABSTRACT BASE CLASSES LAYER =====
abstract_y = 11
abstract_height = 1.3

# Distribution (ABC) - Top center, foundational
draw_box(
    ax,
    8.5,
    abstract_y,
    3,
    abstract_height,
    "Distribution (ABC)\n\n• sample()\n• probability()",
    colors["abstract"],
    fontsize=9,
    fontweight="bold",
    linestyle="dashed",
)

# Environment (ABC) - Left
draw_box(
    ax,
    0.5,
    abstract_y,
    3.5,
    abstract_height,
    "Environment (ABC)\n\n• state_transition_model()\n• observation_model()\n• reward()\n• is_terminal()",
    colors["abstract"],
    fontsize=8,
    fontweight="bold",
    linestyle="dashed",
)

# Policy (ABC) - Right
draw_box(
    ax,
    16,
    abstract_y,
    3.5,
    abstract_height,
    "Policy (ABC)\n\n• action(belief)\n• get_space_info()\n• _verify_compatibility()",
    colors["abstract"],
    fontsize=8,
    fontweight="bold",
    linestyle="dashed",
)

# Belief (ABC) - Bottom center of abstract layer
draw_box(
    ax,
    8.5,
    8.5,
    3,
    1.3,
    "Belief (ABC)\n\n• update(action, obs, env)\n• sample()",
    colors["abstract"],
    fontsize=9,
    fontweight="bold",
    linestyle="dashed",
)

# ===== SPECIALIZED DISTRIBUTION CLASSES =====
dist_y = 9
dist_height = 1.0

# StateTransitionModel
draw_box(
    ax,
    4.5,
    dist_y,
    2.2,
    dist_height,
    "StateTransitionModel\n(Distribution)",
    colors["concrete"],
    fontsize=8,
    fontweight="bold",
)

# ObservationModel
draw_box(
    ax,
    7,
    dist_y,
    2.2,
    dist_height,
    "ObservationModel\n(Distribution)",
    colors["concrete"],
    fontsize=8,
    fontweight="bold",
)

# DiscreteDistribution
draw_box(
    ax,
    9.5,
    dist_y,
    2.2,
    dist_height,
    "DiscreteDistribution\n(Distribution)",
    colors["concrete"],
    fontsize=8,
    fontweight="bold",
)

# ===== CONCRETE IMPLEMENTATIONS =====
concrete_y = 7
concrete_height = 1.0

# Concrete Environments
draw_box(
    ax,
    0.5,
    concrete_y,
    1.8,
    concrete_height,
    "TigerPOMDP\nLightDarkPOMDP\nRockSamplePOMDP",
    colors["concrete"],
    fontsize=7,
)

draw_box(
    ax,
    2.5,
    concrete_y,
    1.5,
    concrete_height,
    "DiscreteActions\nEnvironment",
    colors["concrete"],
    fontsize=7,
    fontweight="bold",
)

# Concrete Beliefs
draw_box(
    ax,
    7.5,
    concrete_y,
    2.2,
    concrete_height,
    "WeightedParticleBelief\nUnweightedParticleBelief",
    colors["concrete"],
    fontsize=7,
)

# Concrete Policies
draw_box(
    ax,
    15.5,
    concrete_y,
    2.0,
    concrete_height,
    "POMCP\nPOMCP_DPW\nPOMCPOW",
    colors["concrete"],
    fontsize=7,
)

draw_box(
    ax,
    17.7,
    concrete_y,
    1.8,
    concrete_height,
    "PathSimulation\nPolicy (ABC)",
    colors["abstract"],
    fontsize=7,
    fontweight="bold",
    linestyle="dashed",
)

# ===== DATA STRUCTURES & ENUMS =====
data_y = 5
data_height = 1.0

# SpaceType enum
draw_box(
    ax,
    0.5,
    data_y,
    2.0,
    data_height,
    "SpaceType (Enum)\n\n• DISCRETE\n• CONTINUOUS\n• MIXED",
    colors["enum"],
    fontsize=7,
    fontweight="bold",
)

# SpaceInfo
draw_box(
    ax,
    2.7,
    data_y,
    2.2,
    data_height,
    "SpaceInfo\n\n• action_space\n• observation_space",
    colors["data"],
    fontsize=7,
    fontweight="bold",
)

# PolicySpaceInfo
draw_box(
    ax,
    5.1,
    data_y,
    2.2,
    data_height,
    "PolicySpaceInfo\n\n• action_space\n• observation_space",
    colors["data"],
    fontsize=7,
    fontweight="bold",
)

# PolicyInfoVariable
draw_box(
    ax,
    11,
    data_y,
    2.0,
    data_height,
    "PolicyInfoVariable\n\n• name\n• value",
    colors["data"],
    fontsize=7,
    fontweight="bold",
)

# PolicyRunData
draw_box(
    ax,
    13.2,
    data_y,
    2.0,
    data_height,
    "PolicyRunData\n\n• info_variables",
    colors["data"],
    fontsize=7,
    fontweight="bold",
)

# ===== TREE STRUCTURES =====
tree_y = 2.5
tree_height = 1.8

# BeliefNode
draw_box(
    ax,
    9,
    tree_y,
    2.5,
    tree_height,
    "BeliefNode\n(NodeMixin)\n\n• belief: Belief\n• observation\n• v_value\n• weight\n• update_belief()",
    colors["concrete"],
    fontsize=7,
    fontweight="bold",
)

# ActionNode
draw_box(
    ax,
    11.8,
    tree_y,
    2.5,
    tree_height,
    "ActionNode\n(NodeMixin)\n\n• action\n• q_value\n• sample_child_node()\n• get_belief_node_child()",
    colors["concrete"],
    fontsize=7,
    fontweight="bold",
)

# BaseNode (shown as note)
ax.text(
    10.2,
    1.8,
    "Both inherit from BaseNode (NodeMixin)",
    ha="center",
    va="center",
    fontsize=7,
    style="italic",
    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors["relation"], alpha=0.7),
)

# ===== RELATIONSHIPS =====

# Distribution inheritance relationships
draw_arrow(
    ax,
    5.6,
    dist_y + dist_height,
    9.5,
    abstract_y,
    edge_colors["inherit"],
    width=2,
    label="inherits",
)
draw_arrow(
    ax,
    8.1,
    dist_y + dist_height,
    9.7,
    abstract_y,
    edge_colors["inherit"],
    width=2,
    label="inherits",
)
draw_arrow(
    ax,
    10.6,
    dist_y + dist_height,
    10.3,
    abstract_y,
    edge_colors["inherit"],
    width=2,
    label="inherits",
)

# Environment returns Transition/Observation Models
draw_arrow(
    ax,
    2.5,
    abstract_y,
    5.6,
    dist_y + dist_height,
    edge_colors["returns"],
    width=2,
    style="dashed",
    label="returns",
)
draw_arrow(
    ax,
    3,
    abstract_y - 0.2,
    8.1,
    dist_y + dist_height,
    edge_colors["returns"],
    width=2,
    style="dashed",
    label="returns",
)

# Environment implementations
draw_arrow(
    ax, 1.4, concrete_y + concrete_height, 2.2, abstract_y, edge_colors["inherit"], width=1.5
)
draw_arrow(
    ax, 3.3, concrete_y + concrete_height, 2.5, abstract_y, edge_colors["inherit"], width=1.5
)

# Policy implementations
draw_arrow(
    ax, 16.5, concrete_y + concrete_height, 17.5, abstract_y, edge_colors["inherit"], width=1.5
)
draw_arrow(
    ax, 18.6, concrete_y + concrete_height, 17.8, abstract_y, edge_colors["inherit"], width=1.5
)

# Belief implementations
draw_arrow(ax, 8.6, concrete_y + concrete_height, 9.5, 8.5, edge_colors["inherit"], width=1.5)

# Policy composes Environment
draw_arrow(
    ax,
    16,
    abstract_y + abstract_height / 2,
    4,
    abstract_y + abstract_height / 2,
    edge_colors["compose"],
    width=2.5,
    label="has-a\n(composition)",
    label_offset=(0, 0.3),
)

# Policy action() takes Belief
draw_arrow(
    ax,
    16.5,
    abstract_y,
    11,
    8.5 + 1.3,
    edge_colors["uses"],
    width=2,
    style="dashed",
    label="takes as\nparameter",
)

# Policy action() returns PolicyRunData
draw_arrow(
    ax,
    17.5,
    abstract_y - 0.5,
    14.2,
    data_y + data_height,
    edge_colors["returns"],
    width=2,
    style="dashed",
    label="returns",
)

# PolicyRunData contains PolicyInfoVariable
draw_arrow(
    ax,
    13.2,
    data_y + data_height / 2,
    13,
    data_y + data_height / 2,
    edge_colors["compose"],
    width=1.5,
    label="contains",
)

# Belief update() uses Environment
draw_arrow(
    ax,
    8.5,
    8.5 + 0.5,
    4,
    abstract_y + 0.3,
    edge_colors["uses"],
    width=2,
    style="dashed",
    label="uses for\nupdate()",
)

# SpaceInfo relationships
draw_arrow(
    ax,
    2.7,
    data_y + data_height,
    2.2,
    abstract_y,
    edge_colors["uses"],
    width=1.5,
    style="dotted",
    label="declares",
)
draw_arrow(
    ax,
    5.8,
    data_y + data_height,
    16.5,
    abstract_y,
    edge_colors["uses"],
    width=1.5,
    style="dotted",
    label="declares",
)

# SpaceType used by SpaceInfo
draw_arrow(
    ax,
    2.5,
    data_y + data_height / 2,
    2.7,
    data_y + data_height / 2,
    edge_colors["compose"],
    width=1.5,
    label="uses",
)
draw_arrow(
    ax,
    2.5,
    data_y + data_height / 2 - 0.3,
    5.1,
    data_y + data_height / 2 - 0.3,
    edge_colors["compose"],
    width=1.5,
    style="dotted",
)

# Tree node relationships
draw_arrow(
    ax,
    10.2,
    8.5,
    10.2,
    tree_y + tree_height,
    edge_colors["compose"],
    width=2,
    label="contains\nBelief instance",
    label_offset=(0.5, 0),
)

# BeliefNode - ActionNode alternating structure
draw_arrow(
    ax,
    11.5,
    tree_y + tree_height / 2,
    11.8,
    tree_y + tree_height / 2,
    edge_colors["uses"],
    width=2,
    label="children",
    label_offset=(0, 0.2),
)
draw_arrow(
    ax,
    11.8,
    tree_y + tree_height / 2 - 0.3,
    11.5,
    tree_y + tree_height / 2 - 0.3,
    edge_colors["uses"],
    width=2,
    style="dashed",
    label="parent",
)

# Policies (MCTS) use Tree structures
draw_arrow(
    ax,
    16.5,
    concrete_y,
    14.3,
    tree_y + tree_height,
    edge_colors["uses"],
    width=2,
    style="dashed",
    label="builds tree\nstructure",
)

# ===== LEGEND =====
legend_elements = [
    mpatches.Patch(
        color=colors["abstract"],
        label="Abstract Base Class (ABC)",
        edgecolor="black",
        linewidth=2,
        linestyle="dashed",
    ),
    mpatches.Patch(color=colors["concrete"], label="Concrete Implementation"),
    mpatches.Patch(color=colors["data"], label="Data Structure / Data Class"),
    mpatches.Patch(color=colors["enum"], label="Enumeration"),
    mlines.Line2D([], [], color=edge_colors["inherit"], linewidth=2, label="Inheritance (is-a)"),
    mlines.Line2D([], [], color=edge_colors["compose"], linewidth=2, label="Composition (has-a)"),
    mlines.Line2D(
        [], [], color=edge_colors["uses"], linewidth=2, linestyle="dashed", label="Uses/Depends on"
    ),
    mlines.Line2D(
        [], [], color=edge_colors["returns"], linewidth=2, linestyle="dashed", label="Returns"
    ),
]

ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

# Add annotations
ax.text(
    10,
    12.5,
    "Core Interface Architecture: Shows abstract base classes (dashed borders),\nconcrete implementations, and their relationships",
    ha="center",
    va="center",
    fontsize=9,
    style="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3),
)

# Add key insight boxes
ax.text(
    1,
    0.5,
    "Key Pattern 1: Composition over Inheritance\nPolicy contains Environment (not extends)",
    ha="left",
    va="center",
    fontsize=7,
    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors["relation"], alpha=0.7),
)

ax.text(
    7,
    0.5,
    "Key Pattern 2: Strategy Pattern\nInterchangeable policies with unified interface",
    ha="left",
    va="center",
    fontsize=7,
    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors["relation"], alpha=0.7),
)

ax.text(
    13,
    0.5,
    "Key Pattern 3: Template Method\nPolicy._verify_compatibility() enforced in all subclasses",
    ha="left",
    va="center",
    fontsize=7,
    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors["relation"], alpha=0.7),
)

plt.tight_layout()
plt.savefig("core_interface_relations.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("core_interface_relations.pdf", bbox_inches="tight", facecolor="white")
print("Core interface relations graph saved as:")
print("  - core_interface_relations.png")
print("  - core_interface_relations.pdf")
