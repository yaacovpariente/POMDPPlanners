# SPDX-License-Identifier: MIT

"""Everything that turns a Chicheck Invaders episode into something you can look at.

Two outputs, from the same recorded episode:

* ``chicheck_invaders_visualizer`` renders the GIF. Its bytes are pinned by a
  golden hash, so this package is a move and nothing more — the renderer, its
  sprites and its arithmetic are unchanged.
* ``trace_exporter`` writes the episode as data, for the browser viewer.

They live together because they answer the same question about the same
episode, and because an environment with more than one presentation file should
keep them in one directory rather than beside the dynamics.
"""

from POMDPPlanners.environments.chicheck_invaders_pomdp.visualizer.chicheck_invaders_visualizer import (  # noqa: E501  pylint: disable=line-too-long
    ChicheckInvadersVisualizer,
)
from POMDPPlanners.environments.chicheck_invaders_pomdp.visualizer.trace_exporter import (
    CHICHECK_INVADERS_PAYLOAD_KIND,
    build_chicheck_invaders_trace,
)

__all__ = [
    "CHICHECK_INVADERS_PAYLOAD_KIND",
    "ChicheckInvadersVisualizer",
    "build_chicheck_invaders_trace",
]
