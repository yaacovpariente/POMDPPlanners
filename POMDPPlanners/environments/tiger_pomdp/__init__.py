# SPDX-License-Identifier: MIT

"""Tiger POMDP family: the environment, its renderer, and its batched model.

Classes:
    TigerPOMDP: The classic two-door tiger problem.
    TigerVisualizer: Renderer for a Tiger episode.
    TigerVectorizedModel: Batched torch model of the same kernels.

The family used to be three loose modules under ``environments``. It is a
package now so the renderer and the batched model sit beside the environment
they belong to, as every other family already does. The names below are
re-exported because ``POMDPPlanners.environments.tiger_pomdp.TigerPOMDP`` is the
module path recorded in saved configurations, and
:meth:`Environment.from_dict` imports that path to rebuild the environment.
``TigerVectorizedModel`` is not re-exported: it imports torch, which this
package must not pull in on every ``import POMDPPlanners.environments``.
"""

from POMDPPlanners.environments.tiger_pomdp.tiger_pomdp import (
    ACTIONS,
    OBSERVATIONS,
    OPEN_ACTIONS,
    STATES,
    TigerPOMDP,
    TigerPOMDPMetrics,
    TigerStepChannel,
)
from POMDPPlanners.environments.tiger_pomdp.tiger_visualization.tiger_visualizer import TigerVisualizer

__all__ = [
    "ACTIONS",
    "OBSERVATIONS",
    "OPEN_ACTIONS",
    "STATES",
    "TigerPOMDP",
    "TigerPOMDPMetrics",
    "TigerStepChannel",
    "TigerVisualizer",
]
