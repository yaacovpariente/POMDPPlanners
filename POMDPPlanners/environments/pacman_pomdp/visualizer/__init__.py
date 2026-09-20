# SPDX-License-Identifier: MIT

"""Everything that turns a PacMan episode into something you can look at.

Three files, one recorded episode:

* ``pacman_art`` caches the sprites and tiles the maze is drawn from.
* ``pacman_visualizer`` renders the GIF. Its bytes are pinned by a golden
  hash, so this package is a move and nothing more — the renderer, its sprites
  and the ``img/`` directory beside it are unchanged.
* ``trace_exporter`` writes the episode as data, for the browser viewer.

They live together because they answer the same question about the same
episode. ``img/`` moved with them so ``Path(__file__).with_name("img")``, which
both renderer modules use to find the sprite sheets and the fonts, keeps
resolving without a path change.
"""

from POMDPPlanners.environments.pacman_pomdp.visualizer.pacman_visualizer import (
    PacManVisualizer,
)
from POMDPPlanners.environments.pacman_pomdp.visualizer.trace_exporter import (
    PACMAN_PAYLOAD_KIND,
    build_pacman_trace,
)

__all__ = [
    "PACMAN_PAYLOAD_KIND",
    "PacManVisualizer",
    "build_pacman_trace",
]
