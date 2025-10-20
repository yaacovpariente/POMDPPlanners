#!/usr/bin/env python3
"""Generate POMDPPlanners Interaction Flow Diagram.

This script creates a sequence diagram showing the runtime interaction flow
between components during a typical POMDP planning episode.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# Set up the figure
fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis("off")

# Define colors
colors = {
    "simulator": "#E8F4F8",
    "policy": "#FFF3E0",
    "belief": "#F3E5F5",
    "environment": "#E8F5E9",
    "tree": "#FFF9C4",
}


def draw_component(ax, x, y, width, height, text, color):
    """Draw a component box."""
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.1",
        edgecolor="black",
        facecolor=color,
        linewidth=2,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        zorder=3,
    )


def draw_lifeline(ax, x, y_start, y_end):
    """Draw a vertical lifeline."""
    ax.plot([x, x], [y_start, y_end], "k--", linewidth=1, zorder=1)


def draw_message(ax, x1, y, x2, text, offset=0.15):
    """Draw a message arrow with label."""
    arrow = FancyArrowPatch(
        (x1, y), (x2, y), arrowstyle="->", color="black", linewidth=1.5, mutation_scale=15, zorder=2
    )
    ax.add_patch(arrow)
    mid_x = (x1 + x2) / 2
    ax.text(mid_x, y + offset, text, ha="center", va="bottom", fontsize=8, zorder=3)


def draw_return_message(ax, x1, y, x2, text, offset=0.15):
    """Draw a return message arrow with label."""
    arrow = FancyArrowPatch(
        (x1, y),
        (x2, y),
        arrowstyle="->",
        color="blue",
        linewidth=1.5,
        linestyle="dashed",
        mutation_scale=15,
        zorder=2,
    )
    ax.add_patch(arrow)
    mid_x = (x1 + x2) / 2
    ax.text(mid_x, y + offset, text, ha="center", va="bottom", fontsize=8, color="blue", zorder=3)


def draw_self_call(ax, x, y, width, text):
    """Draw a self-call loop."""
    rect = Rectangle(
        (x, y - 0.2),
        width,
        0.4,
        edgecolor="darkgreen",
        facecolor="none",
        linewidth=1.5,
        linestyle="dashed",
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        x + width / 2,
        y + 0.3,
        text,
        ha="center",
        va="bottom",
        fontsize=7,
        color="darkgreen",
        zorder=3,
    )


# Title
ax.text(
    9,
    11.5,
    "POMDPPlanners: Runtime Interaction Flow",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
)
ax.text(
    9,
    11,
    "Episode Execution with MCTS Planning",
    ha="center",
    va="center",
    fontsize=12,
    style="italic",
)

# Define component positions
comp_y = 10
comp_height = 0.8
comp_width = 2.5

# Components
components = [
    (1, "Simulator", colors["simulator"]),
    (4, "Environment", colors["environment"]),
    (7, "Policy\n(POMCP)", colors["policy"]),
    (10, "Belief", colors["belief"]),
    (13, "Tree\n(BeliefNode)", colors["tree"]),
    (16, "Tree\n(ActionNode)", colors["tree"]),
]

component_x = {}
for x, name, color in components:
    draw_component(ax, x - comp_width / 2, comp_y, comp_width, comp_height, name, color)
    component_x[name.split("\n")[0]] = x
    # Draw lifeline
    draw_lifeline(ax, x, comp_y - 0.2, 0.5)

# Starting y position for interactions
y = 9.2

# ===== INITIALIZATION PHASE =====
ax.text(
    0.2, y, "INITIALIZATION", ha="left", va="center", fontsize=10, fontweight="bold", style="italic"
)
y -= 0.5

# Create Environment
draw_message(
    ax, component_x["Simulator"], y, component_x["Environment"], "1. from_config(env_config)"
)
y -= 0.3
draw_return_message(
    ax, component_x["Environment"], y, component_x["Simulator"], "environment instance"
)
y -= 0.5

# Create Policy
draw_message(
    ax, component_x["Simulator"], y, component_x["Policy"], "2. from_config(policy_config, env)"
)
y -= 0.3
draw_return_message(ax, component_x["Policy"], y, component_x["Simulator"], "policy instance")
y -= 0.5

# Initialize Belief
draw_message(
    ax, component_x["Simulator"], y, component_x["Environment"], "3. initial_state_dist().sample()"
)
y -= 0.3
draw_return_message(ax, component_x["Environment"], y, component_x["Simulator"], "initial states")
y -= 0.3
draw_message(
    ax, component_x["Simulator"], y, component_x["Belief"], "4. from_config(belief_config, states)"
)
y -= 0.3
draw_return_message(ax, component_x["Belief"], y, component_x["Simulator"], "initial belief")
y -= 0.6

# ===== EPISODE LOOP =====
ax.text(
    0.2, y, "EPISODE LOOP", ha="left", va="center", fontsize=10, fontweight="bold", style="italic"
)
y -= 0.5

# Get Action
draw_message(ax, component_x["Simulator"], y, component_x["Policy"], "5. action(belief)")
y -= 0.3

# Policy builds tree
draw_message(ax, component_x["Policy"], y, component_x["Tree"], "6. create root BeliefNode(belief)")
y -= 0.3
draw_return_message(ax, component_x["Tree"], y, component_x["Policy"], "root_node")
y -= 0.4

# MCTS Loop
ax.text(
    0.2,
    y,
    "MCTS SIMULATIONS",
    ha="left",
    va="center",
    fontsize=9,
    fontweight="bold",
    style="italic",
    color="darkgreen",
)
y -= 0.3

# Simulate path
draw_self_call(ax, component_x["Policy"] - 0.8, y, 1.6, "for i in range(n_simulations)")
y -= 0.5

# Select action
draw_message(ax, component_x["Policy"], y, component_x["Tree"], "7. select action (UCB1)")
y -= 0.3
draw_return_message(ax, component_x["Tree"], y, component_x["Policy"], "action")
y -= 0.4

# Sample next state
draw_message(ax, component_x["Policy"], y, component_x["Belief"], "8. sample()")
y -= 0.3
draw_return_message(ax, component_x["Belief"], y, component_x["Policy"], "state")
y -= 0.3

draw_message(
    ax,
    component_x["Policy"],
    y,
    component_x["Environment"],
    "9. state_transition_model(state, action).sample()",
)
y -= 0.3
draw_return_message(ax, component_x["Environment"], y, component_x["Policy"], "next_state")
y -= 0.4

# Get observation and reward
draw_message(
    ax,
    component_x["Policy"],
    y,
    component_x["Environment"],
    "10. observation_model(next_state, action).sample()",
)
y -= 0.3
draw_return_message(ax, component_x["Environment"], y, component_x["Policy"], "observation")
y -= 0.3

draw_message(ax, component_x["Policy"], y, component_x["Environment"], "11. reward(state, action)")
y -= 0.3
draw_return_message(ax, component_x["Environment"], y, component_x["Policy"], "reward")
y -= 0.4

# Create/update tree nodes
draw_message(ax, component_x["Policy"], y, component_x["Tree"], "12. create ActionNode(action)")
y -= 0.3
draw_message(
    ax, component_x["Policy"], y, component_x["Tree"], "13. create child BeliefNode(observation)"
)
y -= 0.3
draw_message(ax, component_x["Policy"], y, component_x["Tree"], "14. update Q-values & V-values")
y -= 0.5

# Return best action
ax.text(
    0.2,
    y,
    "RETURN BEST ACTION",
    ha="left",
    va="center",
    fontsize=9,
    fontweight="bold",
    style="italic",
    color="darkgreen",
)
y -= 0.3

draw_message(ax, component_x["Policy"], y, component_x["Tree"], "15. get_optimal_action()")
y -= 0.3
draw_return_message(ax, component_x["Tree"], y, component_x["Policy"], "best_action")
y -= 0.3
draw_return_message(
    ax, component_x["Policy"], y, component_x["Simulator"], "16. action, policy_run_data"
)
y -= 0.5

# Execute action in environment
ax.text(
    0.2,
    y,
    "EXECUTE ACTION",
    ha="left",
    va="center",
    fontsize=9,
    fontweight="bold",
    style="italic",
    color="purple",
)
y -= 0.3

draw_message(
    ax,
    component_x["Simulator"],
    y,
    component_x["Environment"],
    "17. sample_next_step(state, action)",
)
y -= 0.3
draw_return_message(
    ax,
    component_x["Environment"],
    y,
    component_x["Simulator"],
    "18. (next_state, observation, reward)",
)
y -= 0.4

# Update belief
draw_message(
    ax,
    component_x["Simulator"],
    y,
    component_x["Belief"],
    "19. update(action, observation, environment)",
)
y -= 0.3
draw_return_message(ax, component_x["Belief"], y, component_x["Simulator"], "20. updated_belief")
y -= 0.4

# Record step
draw_self_call(ax, component_x["Simulator"] - 0.8, y, 1.6, "21. record StepData")
y -= 0.5

# Check termination
draw_message(
    ax, component_x["Simulator"], y, component_x["Environment"], "22. is_terminal(next_state)"
)
y -= 0.3
draw_return_message(ax, component_x["Environment"], y, component_x["Simulator"], "is_terminal")

# Legend
legend_elements = [
    mpatches.Patch(color=colors["simulator"], label="Simulator"),
    mpatches.Patch(color=colors["environment"], label="Environment"),
    mpatches.Patch(color=colors["policy"], label="Policy (MCTS Planner)"),
    mpatches.Patch(color=colors["belief"], label="Belief"),
    mpatches.Patch(color=colors["tree"], label="Tree Nodes"),
    mpatches.Rectangle(
        (0, 0), 1, 1, edgecolor="black", facecolor="none", linewidth=1.5, label="Method Call"
    ),
    mpatches.Rectangle(
        (0, 0),
        1,
        1,
        edgecolor="blue",
        facecolor="none",
        linewidth=1.5,
        linestyle="dashed",
        label="Return Value",
    ),
    mpatches.Rectangle(
        (0, 0),
        1,
        1,
        edgecolor="darkgreen",
        facecolor="none",
        linewidth=1.5,
        linestyle="dashed",
        label="Loop/Self-call",
    ),
]

ax.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=4,
    fontsize=8,
    bbox_to_anchor=(0.5, -0.05),
    framealpha=0.9,
)

plt.tight_layout()
plt.savefig("pomdp_interaction_flow.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("pomdp_interaction_flow.pdf", bbox_inches="tight", facecolor="white")
print("Interaction flow diagram saved as:")
print("  - pomdp_interaction_flow.png")
print("  - pomdp_interaction_flow.pdf")
