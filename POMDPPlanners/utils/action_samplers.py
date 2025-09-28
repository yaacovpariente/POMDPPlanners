import random
from typing import Any, List, Optional

import numpy as np

from POMDPPlanners.core.tree import BeliefNode
from POMDPPlanners.planners.mcts_planners.pft_dpw import ActionSampler


class UnitCircleActionSampler(ActionSampler):
    """Action sampler for 2D continuous action spaces within a unit circle.

    This sampler generates 2D action vectors uniformly distributed within a circle
    of specified maximum magnitude. It's particularly useful for navigation and
    continuous control problems where actions represent velocities or forces
    constrained to a circular region.

    The sampler uses polar coordinate generation to ensure uniform distribution
    within the circle, avoiding the clustering near the center that would occur
    with naive rectangular sampling.

    Mathematical Foundation:
        - Angle θ ~ Uniform(0, 2π)
        - Radius r ~ √Uniform(0, 1) × max_magnitude
        - Action = [r·cos(θ), r·sin(θ)]

    The square root transformation for radius ensures uniform area distribution
    within the circle rather than biasing toward the center.

    Args:
        max_action_magnitude: Maximum magnitude of action vectors (circle radius)

    Example:
        Basic usage with PFT-DPW for 2D navigation::

            import numpy as np
            from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler
            from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
            from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
                ContinuousLightDarkPOMDP
            )
            from POMDPPlanners.core.belief import get_initial_belief

            # Create 2D navigation environment
            nav_env = ContinuousLightDarkPOMDP(
                discount_factor=0.99,
                goal_state=np.array([10, 0]),
                start_state=np.array([0, 0])
            )

            # Create unit circle action sampler
            action_sampler = UnitCircleActionSampler(max_action_magnitude=2.0)

            # Test action sampling
            actions = [action_sampler.sample() for _ in range(5)]
            for i, action in enumerate(actions):
                magnitude = np.linalg.norm(action)
                print(f"Action {i}: [{action[0]:.3f}, {action[1]:.3f}], magnitude: {magnitude:.3f}")

            # Use with PFT-DPW planner
            planner = PFT_DPW(
                environment=nav_env,
                discount_factor=0.99,
                depth=10,
                name="PFT_DPW_Navigation",
                action_sampler=action_sampler,
                n_simulations=100
            )

            initial_belief = get_initial_belief(nav_env, n_particles=200)
            action, run_data = planner.action(initial_belief)

            print(f"Selected navigation action: [{action[0][0]:.3f}, {action[0][1]:.3f}]")

    Example:
        Comparing different action magnitudes for robot control::

            # Conservative movement sampler
            conservative_sampler = UnitCircleActionSampler(max_action_magnitude=0.5)

            # Aggressive movement sampler
            aggressive_sampler = UnitCircleActionSampler(max_action_magnitude=2.0)

            # Compare action distributions
            conservative_actions = [conservative_sampler.sample() for _ in range(100)]
            aggressive_actions = [aggressive_sampler.sample() for _ in range(100)]

            conservative_magnitudes = [np.linalg.norm(a) for a in conservative_actions]
            aggressive_magnitudes = [np.linalg.norm(a) for a in aggressive_actions]

            print(f"Conservative: max={max(conservative_magnitudes):.3f}, avg={np.mean(conservative_magnitudes):.3f}")
            print(f"Aggressive: max={max(aggressive_magnitudes):.3f}, avg={np.mean(aggressive_magnitudes):.3f}")

    Visualization Example:
        Plotting sampled actions to verify uniform distribution::

            import matplotlib.pyplot as plt

            sampler = UnitCircleActionSampler(max_action_magnitude=1.0)
            actions = np.array([sampler.sample() for _ in range(1000)])

            plt.figure(figsize=(8, 8))
            plt.scatter(actions[:, 0], actions[:, 1], alpha=0.6, s=10)
            plt.xlim(-1.2, 1.2)
            plt.ylim(-1.2, 1.2)
            plt.xlabel('Action X')
            plt.ylabel('Action Y')
            plt.title('Unit Circle Action Sampler Distribution')
            plt.grid(True, alpha=0.3)

            # Add unit circle boundary
            theta = np.linspace(0, 2*np.pi, 100)
            plt.plot(np.cos(theta), np.sin(theta), 'r--', linewidth=2, label='Unit Circle')
            plt.legend()
            plt.axis('equal')
            plt.show()

    Algorithm Integration:
        This sampler integrates seamlessly with PFT-DPW for progressive widening
        in continuous action spaces. The uniform circular distribution provides
        good exploration coverage while respecting action magnitude constraints.

        **Comparison with other sampling strategies:**
        - **vs Gaussian sampling**: Better boundary behavior, no tail effects
        - **vs Rectangle sampling**: More natural for navigation/control tasks
        - **vs Custom domain sampling**: General-purpose, works across domains

    Performance Characteristics:
        - **Time complexity**: O(1) per sample
        - **Memory usage**: Minimal (generates one action at a time)
        - **Quality**: Uniform distribution within circle constraint
        - **Scalability**: Excellent for 2D problems, extend for higher dimensions
    """

    def __init__(self, max_action_magnitude: float = 1.0):
        """
        Initialize the unit circle action sampler.

        Args:
            max_action_magnitude: Maximum magnitude of the action vector (default: 1.0)
        """
        self.max_action_magnitude = max_action_magnitude

    def sample(self, belief_node: Optional[BeliefNode] = None) -> np.ndarray:
        """
        Sample an action from a unit circle.

        Args:
            belief_node: The current belief node (not used in this implementation)

        Returns:
            np.ndarray: A 2D action vector within the unit circle
        """
        # Generate random angle in [0, 2π]
        theta = np.random.uniform(0, 2 * np.pi)

        # Generate random radius in [0, 1] using square root for uniform distribution
        r = np.sqrt(np.random.uniform(0, 1)) * self.max_action_magnitude

        # Convert polar coordinates to Cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return np.array([x, y])


class DiscreteActionSampler(ActionSampler):
    """Simple action sampler for discrete action spaces.

    This class is designed to be fully serializable for use in parallel
    processing environments like joblib.
    """

    def __init__(self, actions: List[Any]):
        """Initialize the sampler with a list of discrete actions.

        Args:
            actions: List of discrete actions to sample from
        """
        self.actions = list(actions)

    def sample(self, belief_node: Optional[Any] = None) -> Any:
        """Sample a random action from the discrete action space.

        Args:
            belief_node: Optional belief node (unused in this implementation)

        Returns:
            Randomly sampled action from the action space
        """
        return random.choice(self.actions)

    def __reduce__(self):
        """Support for pickle serialization via __reduce__."""
        return (self.__class__, (), self.__getstate__())
