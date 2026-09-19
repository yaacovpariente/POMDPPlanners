# SPDX-License-Identifier: MIT

"""Everything that turns a Light-Dark episode into something you can look at.

Two outputs, from the same recorded episode:

* ``light_dark_visualizer`` renders the GIF. Its bytes are pinned by a golden
  hash, so this package is a move and nothing more — the renderer and its
  sprites are unchanged.
* ``trace_exporter`` writes the episode as data, for the browser viewer.

They live together because they answer the same question about the same
episode, and because an environment with more than one presentation file
should keep them in one directory rather than scattered through its utils.
"""

from POMDPPlanners.environments.light_dark_pomdp.visualizer.light_dark_visualizer import (
    LightDarkPOMDPVisualizer,
)
from POMDPPlanners.environments.light_dark_pomdp.visualizer.trace_exporter import (
    LIGHT_DARK_PAYLOAD_KIND,
    build_light_dark_trace,
)

__all__ = [
    "LIGHT_DARK_PAYLOAD_KIND",
    "LightDarkPOMDPVisualizer",
    "build_light_dark_trace",
]
