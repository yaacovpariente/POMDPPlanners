# SPDX-License-Identifier: MIT

"""Everything that turns a CartPole episode into something you can look at.

Two outputs, from the same recorded episode:

* ``cartpole_visualizer`` renders the GIF. Its bytes are pinned by a golden
  hash, so this package is a move and nothing more — the renderer, its
  geometry and its palette are unchanged.
* ``trace_exporter`` writes the episode as data, for the browser viewer.

They live together because they answer the same question about the same
episode. The sprite files stay in ``cartpole_pomdp/visualization_assets``:
``cartpole_visualizer`` loads them by package resource, and moving them would
change the packaged data paths for no gain.
"""

from POMDPPlanners.environments.cartpole_pomdp.visualizer.cartpole_visualizer import (
    CartPoleVisualizer,
)
from POMDPPlanners.environments.cartpole_pomdp.visualizer.trace_exporter import (
    CARTPOLE_PAYLOAD_KIND,
    build_cartpole_trace,
)

__all__ = [
    "CARTPOLE_PAYLOAD_KIND",
    "CartPoleVisualizer",
    "build_cartpole_trace",
]
