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
  :class:`CarlaPerceptionPipeline` that :class:`~POMDPPlanners.environments.carla_pomdp.carla_pomdp.CarlaPOMDP`
  runs inside its observation model.

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
    "AlphaBetaTracker",
    "CarlaPerceptionPipeline",
    "Detections",
    "LidarCameraPerceptionModel",
    "MotionTracker",
    "OracleAgentPerceptionModel",
    "PerceptionModel",
    "PerceptionOutput",
]
