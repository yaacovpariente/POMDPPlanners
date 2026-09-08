# SPDX-License-Identifier: MIT

"""Compatibility alias for the shared Maze renderer.

``TMazeVisualizer`` is the pre-move name of ``MazeVisualizer``; both names resolve
to the same class, which renders every environment in the Maze family.
"""

# The renderer imported these state constants at module level before the move, so
# they were reachable from this path too.
from POMDPPlanners.environments.maze_pomdp.maze_pomdp import (  # noqa: F401
    GOAL_LEFT,
    OBSERVATION_LEFT_CUE,
    OBSERVATION_RIGHT_CUE,
    STATE_GOAL,
    STATE_X,
    STATE_Y,
)
from POMDPPlanners.environments.maze_pomdp.maze_visualizer import MazeVisualizer

TMazeVisualizer = MazeVisualizer

__all__ = ["MazeVisualizer", "TMazeVisualizer"]
