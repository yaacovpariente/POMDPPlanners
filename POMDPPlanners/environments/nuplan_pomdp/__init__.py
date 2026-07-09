# SPDX-License-Identifier: MIT

"""nuPlan POMDP environment: a forward-only world plus a swappable planner model.

This package adapts the `nuPlan <https://www.nuplan.org/>`_ closed-loop planning simulator to the
POMDPPlanners interface, following the same world/model split as the CARLA package:

* :mod:`~POMDPPlanners.environments.nuplan_pomdp.nuplan_pomdp` — the ground-truth *world*
  (:class:`NuPlanPOMDP`), forward-only, emitting a raw ``{ego, agents}`` observation.
* :mod:`~POMDPPlanners.environments.nuplan_pomdp.nuplan_perception` — the swappable per-channel
  observation (encoder) models the planner degrades the raw reading with.
* :mod:`~POMDPPlanners.environments.nuplan_pomdp.nuplan_generative_models` — the planner-side
  generative *model* (``policy.environment``): dynamics + the composed observation model.
* :mod:`~POMDPPlanners.environments.nuplan_pomdp.nuplan_belief` — the particle belief that stamps
  the perceived agent block onto its particles.
"""

from POMDPPlanners.environments.nuplan_pomdp.nuplan_pomdp import (
    NuPlanPOMDP,
    NuPlanPOMDPMetrics,
    assemble_state,
    driving_quality_reward,
    relative_agent_row,
)
from POMDPPlanners.environments.nuplan_pomdp.nuplan_belief import PerceivedAgentsBelief

__all__ = [
    "NuPlanPOMDP",
    "NuPlanPOMDPMetrics",
    "PerceivedAgentsBelief",
    "assemble_state",
    "driving_quality_reward",
    "relative_agent_row",
]
