# SPDX-License-Identifier: MIT

"""How an occupancy-grid mapping episode is shown.

``visualizer`` writes the episode as a trace, and
``occupancy_grid_mapping.scene.js`` replays it in the browser.
"""

from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_visualization.occupancy_grid_mapping_visualizer import (  # noqa: E501
    OCCUPANCY_GRID_MAPPING_PAYLOAD_KIND,
    OccupancyGridMappingVisualizer,
)

__all__ = [
    "OCCUPANCY_GRID_MAPPING_PAYLOAD_KIND",
    "OccupancyGridMappingVisualizer",
]
