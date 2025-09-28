#!/usr/bin/env python3
"""
Example script demonstrating the visualize_path functionality for ContinuousLightDarkPOMDPDiscreteActions.

This script creates a dummy path and visualizes it using the environment's visualize_path method.
"""

import numpy as np
from pathlib import Path
from typing import List

from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDPDiscreteActions,
    RewardModelType,
)
from POMDPPlanners.core.distributions import DiscreteDistribution


def create_dummy_path() -> tuple[List[np.ndarray], List[str], List[DiscreteDistribution]]:
    """
    Create a dummy path for visualization demonstration.

    Returns:
        tuple: (path, actions, beliefs) where:
            - path: List of 2D numpy arrays representing agent positions
            - actions: List of action strings
            - beliefs: List of DiscreteDistribution objects representing beliefs
    """
    # Create a path that moves from start to goal, avoiding obstacles
    path = [
        np.array([0, 5]),  # Start state
        np.array([1, 5]),  # Move right
        np.array([2, 5]),  # Move right
        np.array([3, 5]),  # Move right (near obstacle)
        np.array([4, 5]),  # Move right (past obstacle)
        np.array([5, 5]),  # Move right
        np.array([6, 5]),  # Move right
        np.array([7, 5]),  # Move right
        np.array([8, 5]),  # Move right
        np.array([9, 5]),  # Move right
        np.array([10, 5]),  # Goal state
    ]

    # Actions corresponding to the path
    actions = ["right"] * (len(path) - 1)

    # Create dummy beliefs (simplified - each belief is centered on the current position)
    beliefs = []
    for i, position in enumerate(path):
        # Create a belief with particles around the current position
        # Add some noise to make it more realistic
        particles = []
        for j in range(5):  # 5 particles per belief
            noise = np.random.normal(0, 0.5, 2)  # Small noise
            particle = position + noise
            particles.append(particle)

        # Create discrete distribution with equal probabilities
        probs = np.ones(len(particles)) / len(particles)
        belief = DiscreteDistribution(values=particles, probs=probs)
        beliefs.append(belief)

    return path, actions, beliefs


def main():
    """Main function to demonstrate visualize_path functionality."""

    # Create the environment
    env = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95,
        state_transition_cov_matrix=np.eye(2) * 0.1,
        observation_cov_matrix=np.eye(2) * 0.1,
        obstacle_hit_probability=0.2,
        obstacle_reward=-10.0,
        goal_reward=10.0,
        fuel_cost=2.0,
        grid_size=11,
        goal_state_radius=1.5,
        beacon_radius=1.0,
        obstacle_radius=1.5,
        beacons=[(0, 0), (0, 5), (0, 10), (5, 0), (5, 5), (5, 10), (10, 0), (10, 5), (10, 10)],
        goal_state=np.array([10, 5]),
        start_state=np.array([0, 5]),
        obstacles=[(3, 7), (5, 5)],
        reward_model_type=RewardModelType.STANDARD,
        penalty_decay=1.0,
    )

    # Create dummy path data
    path, actions, beliefs = create_dummy_path()

    # Create output directory
    output_dir = Path("experiments/results/visualization_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization
    cache_path = output_dir / "agent_path_visualization.gif"

    print(f"Creating visualization at: {cache_path}")
    print(f"Path length: {len(path)}")
    print(f"Actions: {actions}")
    print(f"Environment grid size: {env.grid_size}")
    print(f"Start state: {env.start_state}")
    print(f"Goal state: {env.goal_state}")
    print(f"Obstacles: {env.obstacles}")
    print(f"Beacons: {env.beacons}")

    # Call the visualize_path method
    env.visualize_path(path=path, agent_belief_path=beliefs, actions=actions, cache_path=cache_path)

    print(f"Visualization saved successfully to: {cache_path}")
    print("The GIF shows:")
    print("- Red dot: Agent's current position")
    print("- Red line: Agent's path")
    print("- Red arrow: Current action direction")
    print("- Purple dots: Belief particles")
    print("- Blue dots: Beacons")
    print("- Green dot: Goal state")
    print("- Red dot (start): Start state")
    print("- Black dots: Obstacles")


if __name__ == "__main__":
    main()
