# SPDX-License-Identifier: MIT

"""Everything that turns a Battleship episode into something you can look at.

Two outputs, from the same recorded episode:

* ``battleship_visualizer`` renders the GIF. Its bytes are pinned by a golden
  hash, so this package is a move and nothing more — the renderer and its
  colours are unchanged.
* ``trace_exporter`` writes the episode as data, for the browser viewer.

They live together because they answer the same question about the same
episode, and because an environment with more than one presentation file
should keep them in one directory rather than beside its dynamics.
"""

from POMDPPlanners.environments.battleship_pomdp.visualizer.battleship_visualizer import (
    BattleshipVisualizer,
)
from POMDPPlanners.environments.battleship_pomdp.visualizer.trace_exporter import (
    BATTLESHIP_PAYLOAD_KIND,
    build_battleship_trace,
)

__all__ = [
    "BATTLESHIP_PAYLOAD_KIND",
    "BattleshipVisualizer",
    "build_battleship_trace",
]
