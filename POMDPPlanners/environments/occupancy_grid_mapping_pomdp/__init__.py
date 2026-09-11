# SPDX-License-Identifier: MIT

"""Occupancy-grid mapping and exploration POMDP package.

Exports:
    OccupancyGridMappingPOMDP: The environment.
    OccupancyGridAction: Its three action indices.
    OccupancyGridMappingVisualizer: Episode renderer.
    OccupancyGridInitialStateDistribution: The per-episode map prior.
    OccupancyGridMappingBelief: The scalar conditional whole-map filter.
    OccupancyGridMappingVectorizedBelief: Its batched twin.
    OccupancyGridMappingVectorizedUpdater: The batched kernels behind it.
    create_occupancy_grid_mapping_belief: Factory choosing between the two filters.
    RangeNoiseModel: The selectable per-beam range noise laws.
"""

from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_pomdp import (
    COL_INDEX,
    HEADING_INDEX,
    POSE_WIDTH,
    RESOLVED_LOG_ODDS,
    ROW_INDEX,
    STEP_INDEX,
    OccupancyGridAction,
    OccupancyGridMappingMetrics,
    OccupancyGridMappingPOMDP,
    OccupancyGridState,
    OccupancyGridStepChannel,
    create_occupancy_grid_state,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_maps import (
    OccupancyGridInitialStateDistribution,
    sample_occupancy_map,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_sensor import (
    HEADING_LABELS,
    HEADING_STEPS,
    NUM_HEADINGS,
    RangeNoiseModel,
    build_ray_templates,
    cast_scan,
    grid_entropy_bits,
    log_odds_from_probability,
    scan_log_odds_delta,
)

__all__ = [
    "OccupancyGridMappingBelief",
    "OccupancyGridMappingVectorizedBelief",
    "OccupancyGridMappingVectorizedUpdater",
    "create_occupancy_grid_mapping_belief",
    "COL_INDEX",
    "HEADING_INDEX",
    "HEADING_LABELS",
    "HEADING_STEPS",
    "NUM_HEADINGS",
    "POSE_WIDTH",
    "RESOLVED_LOG_ODDS",
    "ROW_INDEX",
    "STEP_INDEX",
    "OccupancyGridAction",
    "OccupancyGridInitialStateDistribution",
    "OccupancyGridMappingMetrics",
    "OccupancyGridMappingPOMDP",
    "OccupancyGridState",
    "OccupancyGridStepChannel",
    "RangeNoiseModel",
    "build_ray_templates",
    "cast_scan",
    "create_occupancy_grid_state",
    "grid_entropy_bits",
    "log_odds_from_probability",
    "sample_occupancy_map",
    "scan_log_odds_delta",
]

from .occupancy_grid_mapping_belief import OccupancyGridMappingBelief
from .occupancy_grid_mapping_beliefs import (
    OccupancyGridMappingVectorizedBelief,
    OccupancyGridMappingVectorizedUpdater,
    create_occupancy_grid_mapping_belief,
)
