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

    def __init__(self, actions: Optional[List[Any]] = None):
        """Initialize the sampler with a list of discrete actions.

        Args:
            actions: List of discrete actions to sample from
        """
        if actions is not None:
            self.actions = list(actions)
        else:
            # For compatibility with pickle deserialization via base class __reduce__
            self.actions = []

    def sample(self, belief_node: Optional[BeliefNode] = None) -> Any:
        """Sample a random action from the discrete action space.

        Args:
            belief_node: Optional belief node (unused in this implementation)

        Returns:
            Randomly sampled action from the action space
        """
        return random.choice(self.actions)

    def __getstate__(self):
        """Get state for pickle serialization."""
        return {"actions": self.actions}

    def __setstate__(self, state):
        """Set state for pickle deserialization."""
        self.actions = state["actions"]

    def __reduce__(self):
        """Support for pickle serialization via __reduce__."""
        cls = self.__class__
        state = self.__getstate__()
        return (cls, (), state)
