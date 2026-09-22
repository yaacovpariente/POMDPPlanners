# SPDX-License-Identifier: MIT

"""Everything that turns a LaserTag episode into something you can look at.

Both variants share one Pillow frame renderer and one sprite sheet, and each
adds the little that is genuinely its own — a tag rule, an action label, a ray
model. They are collected here because they answer the same question about the
same episode, and because an environment with five presentation files should
keep them in one directory rather than loose beside the dynamics.

Three outputs, from the same recorded episode:

* ``laser_tag_visualizer`` renders the discrete GIF and
  ``continuous_laser_tag_visualizer`` the continuous one. Their bytes are
  pinned by golden hashes, so this package is a move and nothing more — the
  renderer, the assets and the sprite sheet are unchanged.
* ``trace_exporter`` writes the episode as data, for the browser viewer.
"""

from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_visualization.laser_tag_renderer import (
    LaserTagFrameRenderer,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_visualization.laser_tag_visualizer import (
    LaserTagVisualizer,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_visualization.continuous_laser_tag_visualizer import (
    ContinuousLaserTagVisualizer,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_visualization.trace_exporter import (
    CONTINUOUS_LASER_TAG_VARIANT,
    DISCRETE_LASER_TAG_VARIANT,
    LASER_TAG_PAYLOAD_KIND,
    build_continuous_laser_tag_trace,
    build_laser_tag_trace,
)

__all__ = [
    "CONTINUOUS_LASER_TAG_VARIANT",
    "DISCRETE_LASER_TAG_VARIANT",
    "LASER_TAG_PAYLOAD_KIND",
    "ContinuousLaserTagVisualizer",
    "LaserTagFrameRenderer",
    "LaserTagVisualizer",
    "build_continuous_laser_tag_trace",
    "build_laser_tag_trace",
]
