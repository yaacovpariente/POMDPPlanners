# SPDX-License-Identifier: MIT

"""How a firefighting episode is shown.

``visualizer`` writes the episode as a trace, and ``firefighting.scene.js``
replays it in the browser.
"""

from POMDPPlanners.environments.firefighting_pomdp.firefighting_visualization.firefighting_visualizer import (
    FIREFIGHTING_PAYLOAD_KIND,
    FirefightingVisualizer,
)

__all__ = [
    "FIREFIGHTING_PAYLOAD_KIND",
    "FirefightingVisualizer",
]
