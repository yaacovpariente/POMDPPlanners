# SPDX-License-Identifier: MIT

"""Racetrack POMDP environment module, with a matched fully-observed baseline.

This package adapts HighwayEnv's ``racetrack-v0`` to the POMDPPlanners episode loop as a
**matched pair**: one dynamics path, two observation configurations, selected by
:class:`ObservationMode`. The two assembled simulator configurations are equal except for
the observation block, so a planner's performance gap between them measures partial
observability alone rather than a change of dynamics, reward or map.

It follows the CARLA and nuPlan split — a forward-only world that only steps the true
state, a separate planner-side generative model the search runs inside, and a belief that
does the inference between them.

``highway-env`` is imported lazily and is a development dependency only, so importing this
package, constructing the model, or running the belief never loads the simulator.

Classes:
    RacetrackPOMDP: Forward-only adapter exposing a racetrack session as a world.
    RacetrackModelPOMDP: Abstract planner-side model; concrete models differ only in
        where curvature comes from.
    KnownTrackModel: Planner knows the circuit and looks curvature up by arclength.
    ObservedTrackModel: Planner estimates curvature from the road it can see.
    TrackGeometry: Piecewise-constant curvature of a lap, indexed by arclength.
    RacetrackVectorizedModel: Batched torch counterpart of that model, for VOPP.
    TrackedAgentsBelief: Particle belief that stamps observed detections onto particles.
    SensorObservationModel: The POMDP arm's ego pose, lane camera and detections.
    KinematicsObservationModel: The MDP arm's near-fully-observed kinematics table.
    WorldSensors: What the world measures before its reading leaves the simulator.
    SensorConfig: Every width and limit those sensors are simulated at.
    ObservationMode: Selects the fully-observed or partially-observed arm.
    RacetrackMetric: Episode-level metric names.
    RacetrackStepChannel: Per-step measurement channel names.
"""

from POMDPPlanners.environments.racetrack_pomdp.racetrack_belief import TrackedAgentsBelief
from POMDPPlanners.environments.racetrack_pomdp.racetrack_known_track_model import (
    KnownTrackModel,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp import RacetrackModelPOMDP
from POMDPPlanners.environments.racetrack_pomdp.racetrack_observed_track_model import (
    ObservedTrackModel,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp import (
    RacetrackMetric,
    RacetrackPOMDP,
    RacetrackStepChannel,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_sensor_model import (
    KinematicsObservationModel,
    SensorObservationModel,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_SLOT_WIDTH,
    DEFAULT_ACTION_PRESETS,
    DEFAULT_MAX_TRACKED_AGENTS,
    EGO_STATE_WIDTH,
    ObservationMode,
    build_racetrack_config,
    racetrack_reward,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import (
    TrackGeometry,
    build_track_geometry,
    geometry_from_world,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_vectorized_model import (
    RacetrackVectorizedModel,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_world_sensors import (
    SensorConfig,
    WorldSensors,
)

__all__ = [
    "AGENT_SLOT_WIDTH",
    "DEFAULT_ACTION_PRESETS",
    "DEFAULT_MAX_TRACKED_AGENTS",
    "EGO_STATE_WIDTH",
    "KinematicsObservationModel",
    "KnownTrackModel",
    "ObservationMode",
    "ObservedTrackModel",
    "RacetrackMetric",
    "RacetrackModelPOMDP",
    "RacetrackPOMDP",
    "RacetrackStepChannel",
    "RacetrackVectorizedModel",
    "SensorConfig",
    "SensorObservationModel",
    "TrackGeometry",
    "TrackedAgentsBelief",
    "WorldSensors",
    "build_racetrack_config",
    "build_track_geometry",
    "geometry_from_world",
    "racetrack_reward",
]
