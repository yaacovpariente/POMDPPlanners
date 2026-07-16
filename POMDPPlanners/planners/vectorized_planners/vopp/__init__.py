# SPDX-License-Identifier: MIT

"""Vectorized Online POMDP Planner (VOPP / PORPP).

This package hosts the VOPP planner and its episode runner, whose whole search
is expressed as batched tensor operations over the flat-tensor belief tree in
:mod:`POMDPPlanners.core.tree.vectorized_belief_tree`.

Provides:

* :class:`~POMDPPlanners.planners.vectorized_planners.vopp.vopp.VOPPPlanner` --
  the Vectorized Online POMDP Planner (VOPP / PORPP).
* :class:`~POMDPPlanners.planners.vectorized_planners.vopp.vopp_episode_runner.VOPPEpisodeRunner`
  -- drives full episodes with the VOPP planner.
"""

from POMDPPlanners.planners.vectorized_planners.vopp.vopp import VOPPPlanner
from POMDPPlanners.planners.vectorized_planners.vopp.vopp_episode_runner import (
    VOPPEpisodeRunner,
)

__all__ = ["VOPPPlanner", "VOPPEpisodeRunner"]
