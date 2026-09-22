# SPDX-License-Identifier: MIT

"""Everything that turns a RockSample episode into something you can look at.

Two outputs, from the same recorded episode:

* ``rock_sample_visualizer`` renders the GIF. Its bytes are pinned by a golden
  hash, so this package is a move and nothing more — the renderer, its sprites
  and its palette are unchanged.
* ``trace_exporter`` writes the episode as data, for the browser viewer.

``rock_sample_assets`` stays beside the environment rather than moving in here:
it resolves the packaged sprite directory with ``Path(__file__).with_name``, so
moving the module means moving ``visualization_assets`` with it and editing the
package-data declarations in ``pyproject.toml`` and ``MANIFEST.in``. That is a
packaging change, not a visualization one.
"""

from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_visualization.rock_sample_visualizer import (
    RockSampleVisualizer,
)
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_visualization.trace_exporter import (
    ROCK_SAMPLE_PAYLOAD_KIND,
    build_rock_sample_trace,
)

__all__ = [
    "ROCK_SAMPLE_PAYLOAD_KIND",
    "RockSampleVisualizer",
    "build_rock_sample_trace",
]
