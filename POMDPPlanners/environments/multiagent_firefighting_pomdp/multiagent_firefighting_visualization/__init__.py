# SPDX-License-Identifier: MIT

"""Everything that turns a firefighting episode into something you can look at.

Two outputs, from the same recorded episode:

* ``multiagent_firefighting_visualizer`` renders the three-panel GIF. Its bytes
  are pinned by a golden hash, so moving it in here is a move and nothing more
  -- the renderer, its palette and its fonts are unchanged.
* ``trace_exporter`` writes the episode as data, for the browser viewer.

They live together because they answer the same question about the same
episode, and because an environment with more than one presentation file should
keep them in one directory rather than beside its dynamics.
"""

from POMDPPlanners.environments.multiagent_firefighting_pomdp.multiagent_firefighting_visualization.multiagent_firefighting_visualizer import (  # noqa: E501
    MultiAgentFirefightingVisualizer,
)
from POMDPPlanners.environments.multiagent_firefighting_pomdp.multiagent_firefighting_visualization.trace_exporter import (
    MULTIAGENT_FIREFIGHTING_PAYLOAD_KIND,
    build_multiagent_firefighting_trace,
)

__all__ = [
    "MULTIAGENT_FIREFIGHTING_PAYLOAD_KIND",
    "MultiAgentFirefightingVisualizer",
    "build_multiagent_firefighting_trace",
]
