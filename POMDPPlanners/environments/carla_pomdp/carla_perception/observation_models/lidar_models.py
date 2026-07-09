# SPDX-License-Identifier: MIT

"""Lidar-channel observation models (catalog placeholder).

Register per-channel models for a future ``lidar`` observation channel here, with
:func:`~POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models.registry.register_observation_model`
(``@register_observation_model("lidar", "<name>")``), then import the new class from this
subpackage's ``__init__`` so registration runs on import.

The CARLA world does not yet emit a ``lidar`` channel, so this catalog is intentionally empty.
When raw point clouds enter the observation schema, add an encoder here (e.g. a voxel or
range-image feature encoder). An encoder that only samples (no tractable density) sets
``supports_density = False``: it is usable by sampling planners but rejected by beliefs that
score observations.
"""
