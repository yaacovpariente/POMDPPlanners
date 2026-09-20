# SPDX-License-Identifier: MIT

"""Everything that turns a CaptureTheFlag episode into something you can look at.

Three files, two outputs, one recorded episode:

* ``capture_the_flag_assets`` paints the terrain and the actor art, and
  ``capture_the_flag_visualizer`` draws the GIF from it. The GIF's bytes are
  pinned by a golden hash, so moving them into this package is a move and
  nothing more -- the renderer, its seeds and its sprites are unchanged.
* ``trace_exporter`` writes the same episode as data, for the browser viewer.

They live together because they answer the same question about the same
episode, and because an environment with more than one presentation file
should keep them in one directory rather than beside its dynamics.
"""

from POMDPPlanners.environments.capture_the_flag_pomdp.visualizer.capture_the_flag_visualizer import (  # noqa: E501
    CaptureTheFlagVisualizer,
)
from POMDPPlanners.environments.capture_the_flag_pomdp.visualizer.trace_exporter import (
    CAPTURE_THE_FLAG_PAYLOAD_KIND,
    build_capture_the_flag_trace,
)

__all__ = [
    "CAPTURE_THE_FLAG_PAYLOAD_KIND",
    "CaptureTheFlagVisualizer",
    "build_capture_the_flag_trace",
]
