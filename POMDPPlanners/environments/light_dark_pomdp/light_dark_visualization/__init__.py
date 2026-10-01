# SPDX-License-Identifier: MIT

"""How a Light-Dark episode is shown.

``visualizer`` writes the episode as a trace, and ``light_dark.scene.js``
replays it in the browser.
"""

from POMDPPlanners.environments.light_dark_pomdp.light_dark_visualization.light_dark_visualizer import (
    LIGHT_DARK_PAYLOAD_KIND,
    LightDarkVisualizer,
)

__all__ = [
    "LIGHT_DARK_PAYLOAD_KIND",
    "LightDarkVisualizer",
]
