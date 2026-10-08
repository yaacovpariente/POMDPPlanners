# SPDX-License-Identifier: MIT

"""How a Racetrack episode is shown.

``visualizer`` writes the episode as a trace, and ``racetrack.scene.js``
replays it in the browser. ``reference_track`` holds the circuit's exact lane
edges, which the trace carries so the viewer draws the road the car drove.
"""

from POMDPPlanners.environments.racetrack_pomdp.racetrack_visualization.racetrack_visualizer import (
    RACETRACK_PAYLOAD_KIND,
    RacetrackVisualizer,
)

__all__ = [
    "RACETRACK_PAYLOAD_KIND",
    "RacetrackVisualizer",
]
