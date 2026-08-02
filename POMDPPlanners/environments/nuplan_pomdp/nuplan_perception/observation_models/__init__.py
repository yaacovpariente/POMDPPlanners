# SPDX-License-Identifier: MIT

"""Catalog of per-channel nuPlan observation models, registered for user selection.

Each observation channel has its own module of concrete
:class:`~POMDPPlanners.environments.nuplan_pomdp.nuplan_perception.observation_model.NuPlanObservationModel`
implementations (``ego_models``, ``agent_models``). Importing a channel module runs its
``@register_observation_model`` decorators, so a planner environment can resolve a per-channel
selection like ``{"ego": "gaussian", "agents": "factored"}`` into instances via
:func:`build_observation_model`.

To add a modality: create/extend its ``<channel>_models.py``, register the class, and import it
here so registration runs on import.

Functions:
    register_observation_model: Decorator registering a factory under ``(channel, name)``.
    build_observation_model: Instantiate the model registered under ``(channel, name)``.
    available_observation_models: List the names registered for a channel.

Classes:
    EgoObservationModel: Additive-Gaussian-noise ego proprioception with a matching density.
    FactoredAgentObservationModel: Per-slot detection + occlusion gating + Gaussian pose noise.
"""

from POMDPPlanners.environments.nuplan_pomdp.nuplan_perception.observation_models.registry import (
    available_observation_models,
    build_observation_model,
    register_observation_model,
)
from POMDPPlanners.environments.nuplan_pomdp.nuplan_perception.observation_models.ego_models import (
    EgoObservationModel,
)
from POMDPPlanners.environments.nuplan_pomdp.nuplan_perception.observation_models.agent_models import (
    FactoredAgentObservationModel,
)

__all__ = [
    "register_observation_model",
    "build_observation_model",
    "available_observation_models",
    "EgoObservationModel",
    "FactoredAgentObservationModel",
]
