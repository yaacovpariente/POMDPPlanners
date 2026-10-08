# SPDX-License-Identifier: MIT

"""How a LaserTag episode is shown.

``visualizer`` writes the episode as a trace, and ``laser_tag.scene.js``
replays it in the browser. Both variants share one payload kind; the
``variant`` field in the world block and the metadata says which one wrote it.
"""

from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_visualization.laser_tag_visualizer import (
    CONTINUOUS_LASER_TAG_VARIANT,
    DISCRETE_LASER_TAG_VARIANT,
    LASER_TAG_PAYLOAD_KIND,
    ContinuousLaserTagVisualizer,
    LaserTagVisualizer,
)

__all__ = [
    "CONTINUOUS_LASER_TAG_VARIANT",
    "DISCRETE_LASER_TAG_VARIANT",
    "LASER_TAG_PAYLOAD_KIND",
    "ContinuousLaserTagVisualizer",
    "LaserTagVisualizer",
]
