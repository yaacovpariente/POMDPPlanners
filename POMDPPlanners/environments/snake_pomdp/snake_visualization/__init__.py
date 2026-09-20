# SPDX-License-Identifier: MIT

"""Everything that turns a Snake episode into something you can look at.

Two outputs, from the same recorded episode:

* ``snake_visualizer`` renders the GIF. Its bytes are pinned by a golden hash,
  so its arrival here is a move and nothing more -- the renderer, its palette
  and its layout are unchanged.
* ``trace_exporter`` writes the episode as data, for the browser viewer.

They live together because they answer the same question about the same
episode, and because an environment with more than one presentation file should
keep them in one directory rather than beside its dynamics.
"""

from POMDPPlanners.environments.snake_pomdp.snake_visualization.snake_visualizer import (
    SnakeVisualizer,
)
from POMDPPlanners.environments.snake_pomdp.snake_visualization.trace_exporter import (
    SNAKE_PAYLOAD_KIND,
    build_snake_trace,
)

__all__ = [
    "SNAKE_PAYLOAD_KIND",
    "SnakeVisualizer",
    "build_snake_trace",
]
