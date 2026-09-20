# SPDX-License-Identifier: MIT

"""Maze presentation: the episode written as data for the browser viewer.

The GIF renderer is *not* in here. ``maze_pomdp.maze_visualizer`` draws the
generated Maze and the T-Maze from one class, and the T-Maze is a separate
environment with its own migration, so moving that file would move a file two
environments own. It stays where both already import it from; only the trace
exporter, which is the Maze's alone, lives here.
"""

from POMDPPlanners.environments.maze_pomdp.visualizer.trace_exporter import (
    MAZE_PAYLOAD_KIND,
    build_maze_trace,
)

__all__ = [
    "MAZE_PAYLOAD_KIND",
    "build_maze_trace",
]
