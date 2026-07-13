# SPDX-License-Identifier: MIT

"""Standalone, swappable perception + prediction stack for the CARLA planner.

This subpackage turns the CARLA world's raw multi-modal sensors into the perceived, tracked
object list the planner reasons about. It is decoupled from both the world and the belief, so a
user can swap in a different perception model without touching either:

* :mod:`~POMDPPlanners.environments.carla_pomdp.carla_perception.carla_sensors` — pure single-frame
  helpers (lidar corridor clearance, camera looming cue, lidar vehicle clustering, and a
  camera-inferred traffic light).
* :mod:`~POMDPPlanners.environments.carla_pomdp.carla_perception.carla_tracking` — a constant-velocity
  multi-object tracker that adds velocity and coasts briefly occluded vehicles.
* :mod:`~POMDPPlanners.environments.carla_pomdp.carla_perception.carla_pipeline` — the swappable
  :class:`PerceptionModel` / :class:`MotionTracker` interfaces and the composed, immutable
  :class:`CarlaPerceptionPipeline`, a standalone whole-observation sensor-fusion stage (raw
  lidar/camera -> tracked agent block).
* :mod:`~POMDPPlanners.environments.carla_pomdp.carla_perception.observation_model` — the shared
  per-channel :class:`CarlaObservationModel` interface (one clean channel -> one perceived
  channel) that the planner's generative models compose into a ``{channel: model}`` map.
* :mod:`~POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models` — the catalog
  of concrete per-channel models (:class:`GnssObservationModel`,
  :class:`FactoredAgentObservationModel`) registered for user selection by name.

The public names below are re-exported here so callers can import them straight from the
subpackage (e.g. ``from ...carla_perception import CarlaPerceptionPipeline``).
"""

from POMDPPlanners.environments.carla_pomdp.carla_perception.carla_sensors import (
    camera_looming_cue,
    fuse_forward_obstacle,
    lidar_forward_clearance,
    lidar_vehicle_detections,
    traffic_light_from_camera,
    traffic_light_stop_distance,
)
from POMDPPlanners.environments.carla_pomdp.carla_perception.carla_tracking import (
    TRACK_WIDTH,
    update_tracks,
)
from POMDPPlanners.environments.carla_pomdp.carla_perception.observation_model import (
    CarlaObservationModel,
)
from POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models import (
    FactoredAgentObservationModel,
    GnssObservationModel,
    available_observation_models,
    build_observation_model,
    register_observation_model,
)
from POMDPPlanners.environments.carla_pomdp.carla_perception.carla_pipeline import (
    AlphaBetaTracker,
    CarlaPerceptionPipeline,
    Detections,
    LidarCameraPerceptionModel,
    MotionTracker,
    OracleAgentPerceptionModel,
    PerceptionModel,
    PerceptionOutput,
)

__all__ = [
    "camera_looming_cue",
    "fuse_forward_obstacle",
    "lidar_forward_clearance",
    "lidar_vehicle_detections",
    "traffic_light_from_camera",
    "traffic_light_stop_distance",
    "TRACK_WIDTH",
    "update_tracks",
    "CarlaObservationModel",
    "FactoredAgentObservationModel",
    "GnssObservationModel",
    "available_observation_models",
    "build_observation_model",
    "register_observation_model",
    "AlphaBetaTracker",
    "CarlaPerceptionPipeline",
    "Detections",
    "LidarCameraPerceptionModel",
    "MotionTracker",
    "OracleAgentPerceptionModel",
    "PerceptionModel",
    "PerceptionOutput",
]
