# SPDX-License-Identifier: MIT

"""Everything that turns a Safety Ant Velocity episode into something to look at.

Three files, one recorded episode:

* ``safety_ant_velocity_visualizer`` renders the GIF. Its bytes are pinned by a
  golden hash, so this package is a move and nothing more — the renderer is
  unchanged.
* ``safety_ant_assets`` is the art that renderer draws with, and is used by
  nothing else.
* ``trace_exporter`` writes the episode as data, for the browser viewer.

They live together because they answer the same question about the same
episode, and an environment with more than one presentation file should keep
them in one directory rather than loose in the package root.
"""

# The renderer's fully qualified name is two characters past the line limit, so
# the module is imported and the class taken off it rather than the name being
# imported directly. The alternative is a noqa on an import that is otherwise
# ordinary.
from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_visualization import (
    safety_ant_velocity_visualizer as _gif_renderer,
)
from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_visualization.trace_exporter import (
    SAFETY_ANT_VELOCITY_PAYLOAD_KIND,
    build_safety_ant_velocity_trace,
)

SafeAntVelocityVisualizer = _gif_renderer.SafeAntVelocityVisualizer

__all__ = [
    "SAFETY_ANT_VELOCITY_PAYLOAD_KIND",
    "SafeAntVelocityVisualizer",
    "build_safety_ant_velocity_trace",
]
