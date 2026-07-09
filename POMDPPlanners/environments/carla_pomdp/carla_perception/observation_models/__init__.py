# SPDX-License-Identifier: MIT

"""Catalog of per-channel CARLA observation models, registered for user selection.

Each observation channel has its own module of concrete
:class:`~POMDPPlanners.environments.carla_pomdp.carla_perception.observation_model.CarlaObservationModel`
implementations (``gnss_models``, ``agent_models``, and the ``image_models`` / ``lidar_models``
placeholders for future modalities). Importing a channel module runs its
``@register_observation_model`` decorators, so a planner environment can resolve a per-channel
selection like ``{"gnss": "gaussian", "agents": "factored"}`` into instances via
:func:`build_observation_model`.

To add a modality: create/extend its ``<channel>_models.py``, register the class, and import it
here so registration runs on import.

Functions:
    register_observation_model: Decorator registering a factory under ``(channel, name)``.
    build_observation_model: Instantiate the model registered under ``(channel, name)``.
    available_observation_models: List the names registered for a channel.

Classes:
    GnssObservationModel: Additive-Gaussian-noise GNSS reading with a matching density.
    FactoredAgentObservationModel: Per-slot detection + occlusion gating + Gaussian pose noise.
"""

from POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models.registry import (
    available_observation_models,
    build_observation_model,
    register_observation_model,
)
from POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models.gnss_models import (
    GnssObservationModel,
)
from POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models.agent_models import (
    FactoredAgentObservationModel,
)

__all__ = [
    "register_observation_model",
    "build_observation_model",
    "available_observation_models",
    "GnssObservationModel",
    "FactoredAgentObservationModel",
]
