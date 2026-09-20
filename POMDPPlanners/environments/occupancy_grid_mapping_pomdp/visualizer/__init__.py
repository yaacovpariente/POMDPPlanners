# SPDX-License-Identifier: MIT

"""Everything that turns an occupancy-grid mapping episode into something to look at.

Three modules, one recorded episode:

* ``occupancy_grid_mapping_visualizer`` renders the GIF. Its bytes are pinned
  by a golden hash, so this package is a move and nothing more -- the renderer
  and its palette are unchanged.
* ``occupancy_grid_mapping_assets`` holds the palette and the sprite work the
  renderer draws with. It moved with the renderer because it is read by nothing
  else.
* ``trace_exporter`` writes the episode as data, for the browser viewer.

They live together because they answer the same question about the same
episode, and because an environment with more than one presentation file should
keep them in one directory rather than loose beside its dynamics.
"""

from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.visualizer.occupancy_grid_mapping_visualizer import (  # noqa: E501
    OccupancyGridMappingVisualizer,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.visualizer.trace_exporter import (
    OCCUPANCY_GRID_MAPPING_PAYLOAD_KIND,
    build_occupancy_grid_mapping_trace,
)

__all__ = [
    "OCCUPANCY_GRID_MAPPING_PAYLOAD_KIND",
    "OccupancyGridMappingVisualizer",
    "build_occupancy_grid_mapping_trace",
]
