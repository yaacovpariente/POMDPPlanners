# SPDX-License-Identifier: MIT

"""How a Snake episode is shown.

``visualizer`` writes the episode as a trace, and ``snake.scene.js`` replays it
in the browser.
"""

from POMDPPlanners.environments.snake_pomdp.snake_visualization.snake_visualizer import (
    SNAKE_PAYLOAD_KIND,
    SnakeVisualizer,
)

__all__ = [
    "SNAKE_PAYLOAD_KIND",
    "SnakeVisualizer",
]
