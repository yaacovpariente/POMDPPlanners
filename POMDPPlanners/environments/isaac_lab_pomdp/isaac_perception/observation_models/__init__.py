# SPDX-License-Identifier: MIT

"""Catalog of Isaac observation models, registered for user selection by name.

Concrete
:class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_model.IsaacObservationModel`
implementations are grouped by sensing modality — proprioception, ray casting, and the latent-type
signal a risk study needs. Importing this package runs their ``@register_observation_model``
decorators, so a generative model can resolve a selection like
``{"base_pose": "gaussian", "lidar": "ray_caster"}`` into instances via
:func:`build_observation_model`.

To add a modality: create or extend its ``*_models.py``, register the class, and import it here so
registration runs on import.

Functions:
    register_observation_model: Decorator registering a factory under a name.
    build_observation_model: Instantiate the model registered under a name.
    available_observation_models: List the registered names.

Classes:
    GaussianChannelObservationModel: One state block seen through additive Gaussian noise.
    ExactChannelObservationModel: One state block observed without error.
    RayCasterObservationModel: Planar LiDAR ranges to a disc obstacle set.
    HeightScanObservationModel: Downward height samples on a body-frame grid.
    LatentTypeSignalObservationModel: Per-zone binary signal, informative only inside the zone.
"""

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models.registry import (
    available_observation_models,
    build_observation_model,
    register_observation_model,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models.proprioception_models import (  # noqa: E501  pylint: disable=line-too-long
    ExactChannelObservationModel,
    GaussianChannelObservationModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models.ray_caster_models import (  # noqa: E501  pylint: disable=line-too-long
    HeightScanObservationModel,
    RayCasterObservationModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models.hazard_signal_models import (  # noqa: E501  pylint: disable=line-too-long
    UNINFORMATIVE_ACCURACY,
    LatentTypeSignalObservationModel,
)

__all__ = [
    "available_observation_models",
    "build_observation_model",
    "register_observation_model",
    "ExactChannelObservationModel",
    "GaussianChannelObservationModel",
    "HeightScanObservationModel",
    "RayCasterObservationModel",
    "LatentTypeSignalObservationModel",
    "UNINFORMATIVE_ACCURACY",
]
