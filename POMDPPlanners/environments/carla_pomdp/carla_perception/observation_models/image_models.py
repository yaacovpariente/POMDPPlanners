# SPDX-License-Identifier: MIT

"""Image-channel observation models (catalog placeholder).

Register per-channel models for a future ``image`` observation channel here, with
:func:`~POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models.registry.register_observation_model`
(``@register_observation_model("image", "<name>")``), then import the new class from this
subpackage's ``__init__`` so registration runs on import.

The CARLA world does not yet emit an ``image`` channel, so this catalog is intentionally empty.
When raw camera frames enter the observation schema, add an encoder here (e.g. a learned CNN
feature map). An encoder that only samples (no tractable density) sets ``supports_density =
False``: it is usable by sampling planners but rejected by beliefs that score observations.
"""
