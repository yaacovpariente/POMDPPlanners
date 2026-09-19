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
    OccupancyUpdateRule: Abstract map update rule, written on log-odds.
    ProbabilityOccupancyUpdateRule: Abstract map update rule, written on
        occupancy probabilities.
    NearestCellLogOddsUpdateRule: The original rule, and the default.
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
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_update_rules import (
    NearestCellLogOddsUpdateRule,
    OccupancyUpdateRule,
    ProbabilityOccupancyUpdateRule,
    default_update_rule,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_sensor import (
    HEADING_LABELS,
    HEADING_STEPS,
    NUM_HEADINGS,
    RangeNoiseModel,
    build_ray_templates,
    cast_scan,
    batch_observed_scan_evidence_counts,
    grid_entropy_bits,
    log_odds_from_probability,
    observed_scan_evidence_counts,
    observed_scan_log_odds_delta,
    scan_log_odds_delta,
)

__all__ = [
    "COL_INDEX",
    "HEADING_INDEX",
    "HEADING_LABELS",
    "HEADING_STEPS",
    "NUM_HEADINGS",
    "NearestCellLogOddsUpdateRule",
    "OccupancyGridAction",
    "OccupancyGridInitialStateDistribution",
    "OccupancyGridMappingBelief",
    "OccupancyGridMappingMetrics",
    "OccupancyGridMappingPOMDP",
    "OccupancyGridMappingVectorizedBelief",
    "OccupancyGridMappingVectorizedUpdater",
    "OccupancyGridState",
    "OccupancyGridStepChannel",
    "OccupancyUpdateRule",
    "POSE_WIDTH",
    "ProbabilityOccupancyUpdateRule",
    "RESOLVED_LOG_ODDS",
    "ROW_INDEX",
    "RangeNoiseModel",
    "STEP_INDEX",
    "batch_observed_scan_evidence_counts",
    "build_ray_templates",
    "cast_scan",
    "create_occupancy_grid_mapping_belief",
    "create_occupancy_grid_state",
    "default_update_rule",
    "grid_entropy_bits",
    "log_odds_from_probability",
    "observed_scan_evidence_counts",
    "observed_scan_log_odds_delta",
    "sample_occupancy_map",
    "scan_log_odds_delta",
]

from .occupancy_grid_mapping_belief import OccupancyGridMappingBelief
from .occupancy_grid_mapping_beliefs import (
    OccupancyGridMappingVectorizedBelief,
    OccupancyGridMappingVectorizedUpdater,
    create_occupancy_grid_mapping_belief,
)
