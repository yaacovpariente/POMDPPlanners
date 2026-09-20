# SPDX-License-Identifier: MIT

"""Everything that turns a Maze-family episode into something you can look at.

The renderer itself is not here. ``MazeVisualizer`` draws every environment in
the family — the generated Maze and the T-Maze alike — so it stays at the
package root rather than being claimed by either one.

What lives here is one trace exporter per environment: the same episodes
written as data, for the browser viewer.
"""

from POMDPPlanners.environments.maze_pomdp.visualizer.t_maze_trace_exporter import (
    T_MAZE_PAYLOAD_KIND,
    build_t_maze_trace,
)

__all__ = [
    "T_MAZE_PAYLOAD_KIND",
    "build_t_maze_trace",
]
