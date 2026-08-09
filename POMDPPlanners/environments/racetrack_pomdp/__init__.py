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
    RacetrackModelPOMDP: Planner-side generative model over the racetrack schema.
    TrackedAgentsBelief: Particle belief that stamps grid-tracked agents onto particles.
    OccupancyVelocityTracker: Turns consecutive occupancy grids into cluster velocities.
    TrackedCluster: One tracked occupancy cluster with its relative velocity.
    ObservationMode: Selects the fully-observed or partially-observed arm.
    RacetrackMetric: Episode-level metric names.
    RacetrackStepChannel: Per-step measurement channel names.
"""

from POMDPPlanners.environments.racetrack_pomdp.racetrack_belief import TrackedAgentsBelief
from POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp import RacetrackModelPOMDP
from POMDPPlanners.environments.racetrack_pomdp.racetrack_occupancy_tracker import (
    OccupancyVelocityTracker,
    TrackedCluster,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp import (
    RacetrackMetric,
    RacetrackPOMDP,
    RacetrackStepChannel,
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

__all__ = [
    "AGENT_SLOT_WIDTH",
    "DEFAULT_ACTION_PRESETS",
    "DEFAULT_MAX_TRACKED_AGENTS",
    "EGO_STATE_WIDTH",
    "ObservationMode",
    "OccupancyVelocityTracker",
    "RacetrackMetric",
    "RacetrackModelPOMDP",
    "RacetrackPOMDP",
    "RacetrackStepChannel",
    "TrackedAgentsBelief",
    "TrackedCluster",
    "build_racetrack_config",
    "racetrack_reward",
]
