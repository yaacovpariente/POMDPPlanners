# SPDX-License-Identifier: MIT

"""GPU-vectorized online POMDP planners.

This package hosts planners whose whole search is expressed as batched tensor
operations over the flat-tensor belief tree in
:mod:`POMDPPlanners.core.tree.vectorized_belief_tree`. Unlike the Monte Carlo
Tree Search planners in :mod:`POMDPPlanners.planners.mcts_planners`, these run
tens of thousands of parallel simulations per planning step with no Python
per-simulation loop and no host/device synchronization.

Currently provided:

* :class:`~POMDPPlanners.planners.vectorized_planners.vopp.vopp.VOPPPlanner` --
  the Vectorized Online POMDP Planner (VOPP / PORPP).
"""

from POMDPPlanners.planners.vectorized_planners.vopp import (
    VOPPEpisodeRunner,
    VOPPPlanner,
)

__all__ = ["VOPPPlanner", "VOPPEpisodeRunner"]
