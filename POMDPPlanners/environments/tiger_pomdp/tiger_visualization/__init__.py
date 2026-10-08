# SPDX-License-Identifier: MIT

"""How a Tiger episode is shown.

``visualizer`` writes the episode as a trace, and ``tiger.scene.js`` replays
it in the browser.
"""

from POMDPPlanners.environments.tiger_pomdp.tiger_visualization.tiger_visualizer import (
    TIGER_PAYLOAD_KIND,
    TigerVisualizer,
)

__all__ = [
    "TIGER_PAYLOAD_KIND",
    "TigerVisualizer",
]
