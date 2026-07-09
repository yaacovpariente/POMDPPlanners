# SPDX-License-Identifier: MIT

"""Standalone, swappable perception stack for the nuPlan planner.

This subpackage turns the nuPlan world's raw ``{ego, agents}`` reading into the perceived
observation the planner reasons about. It is decoupled from both the world and the belief, so a
user can swap in a different perception model without touching either:

* :mod:`~POMDPPlanners.environments.nuplan_pomdp.nuplan_perception.observation_model` — the shared
  per-channel :class:`NuPlanObservationModel` interface (one clean channel -> one perceived
  channel) that the planner's generative models compose into a ``{channel: model}`` map.
* :mod:`~POMDPPlanners.environments.nuplan_pomdp.nuplan_perception.observation_models` — the catalog
  of concrete per-channel models (:class:`EgoObservationModel`,
  :class:`FactoredAgentObservationModel`) registered for user selection by name.

The public names below are re-exported here so callers can import them straight from the
subpackage (e.g. ``from ...nuplan_perception import NuPlanObservationModel``).
"""

from POMDPPlanners.environments.nuplan_pomdp.nuplan_perception.observation_model import (
    NuPlanObservationModel,
)
from POMDPPlanners.environments.nuplan_pomdp.nuplan_perception.observation_models import (
    EgoObservationModel,
    FactoredAgentObservationModel,
    available_observation_models,
    build_observation_model,
    register_observation_model,
)

__all__ = [
    "NuPlanObservationModel",
    "EgoObservationModel",
    "FactoredAgentObservationModel",
    "available_observation_models",
    "build_observation_model",
    "register_observation_model",
]
