# SPDX-License-Identifier: MIT

"""Everything that turns a Tiger episode into something you can look at.

Two outputs, from the same recorded episode:

* ``tiger_visualizer`` renders the GIF. Its bytes are pinned by a golden hash,
  so the move into this package is a move and nothing more — the renderer and
  its ``chamber.png`` came across untouched, and the asset kept its directory
  name so even the path the renderer builds is the same string.
* ``trace_exporter`` writes the episode as data, for the browser viewer.

They live together because they answer the same question about the same
episode, and because three visualization files loose at the top of
``environments/`` is how ``tiger_visualization_assets`` ended up needing its own
line in the coverage matrix.
"""

from POMDPPlanners.environments.tiger_pomdp.visualizer.tiger_visualizer import TigerVisualizer
from POMDPPlanners.environments.tiger_pomdp.visualizer.trace_exporter import (
    TIGER_PAYLOAD_KIND,
    build_tiger_trace,
)

__all__ = [
    "TIGER_PAYLOAD_KIND",
    "TigerVisualizer",
    "build_tiger_trace",
]
