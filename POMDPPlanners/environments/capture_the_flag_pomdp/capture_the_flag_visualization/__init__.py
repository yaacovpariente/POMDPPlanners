# SPDX-License-Identifier: MIT

"""How a CaptureTheFlag episode is shown.

``visualizer`` writes the episode as a trace, and ``capture_the_flag.scene.js``
replays it in the browser.
"""

from POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_visualization.capture_the_flag_visualizer import (  # noqa: E501
    CAPTURE_THE_FLAG_PAYLOAD_KIND,
    CaptureTheFlagVisualizer,
)

__all__ = [
    "CAPTURE_THE_FLAG_PAYLOAD_KIND",
    "CaptureTheFlagVisualizer",
]
