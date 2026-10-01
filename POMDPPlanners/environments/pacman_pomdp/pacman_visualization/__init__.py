# SPDX-License-Identifier: MIT

"""How a PacMan episode is shown.

``visualizer`` writes the episode as a trace, and ``pacman.scene.js`` replays
it in the browser.
"""

from POMDPPlanners.environments.pacman_pomdp.pacman_visualization.pacman_visualizer import (
    PACMAN_PAYLOAD_KIND,
    PacManVisualizer,
)

__all__ = [
    "PACMAN_PAYLOAD_KIND",
    "PacManVisualizer",
]
