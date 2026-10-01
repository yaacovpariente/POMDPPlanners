# SPDX-License-Identifier: MIT

"""How a Maze-family episode is shown.

Two visualizers write episodes as traces for the 3D viewer, one per payload:
``visualizer`` writes ``maze.v1`` for the generated Maze, discrete and
continuous, and ``maze.scene.js`` replays it; ``t_maze_visualizer`` writes
``t_maze.v1`` for the T-Maze, and ``t_maze.scene.js`` replays it.
"""

from POMDPPlanners.environments.maze_pomdp.maze_visualization.t_maze_visualizer import (
    T_MAZE_PAYLOAD_KIND,
    TMazeVisualizer,
)
from POMDPPlanners.environments.maze_pomdp.maze_visualization.maze_visualizer import (
    MAZE_PAYLOAD_KIND,
    MazeVisualizer,
)

__all__ = [
    "MAZE_PAYLOAD_KIND",
    "MazeVisualizer",
    "TMazeVisualizer",
    "T_MAZE_PAYLOAD_KIND",
]
