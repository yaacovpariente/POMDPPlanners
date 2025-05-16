import numpy as np
from typing import Any
from POMDPPlanners.core.tree import BeliefNode
from POMDPPlanners.planners.mcts_planners.pft_dpw import ActionSampler


class UnitCircleActionSampler(ActionSampler):
    def __init__(self, max_action_magnitude: float = 1.0):
        """
        Initialize the unit circle action sampler.
        
        Args:
            max_action_magnitude: Maximum magnitude of the action vector (default: 1.0)
        """
        self.max_action_magnitude = max_action_magnitude
    
    def sample(self, belief_node: BeliefNode) -> np.ndarray:
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
