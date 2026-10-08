# SPDX-License-Identifier: MIT

"""Firefighting POMDP package.

Exports:
    FirefightingPOMDP: The environment.
    FirefightingVisualizer: Writes episodes as traces for the 3D viewer.
    FirefightingVectorizedBelief: The environment's default belief.
    FirefightingVectorizedModel: Torch generative model for VOPP.
    FirefightingInitialStateDistribution: The reset distribution.
    FireCategory: The five per-cell categories.
    FirefightingAction: The five per-firefighter actions.
    WindDirection, WindStrength: The two halves of the hidden wind.
    create_firefighting_state: Build one state vector in an env's layout.
"""

from POMDPPlanners.environments.firefighting_pomdp.firefighting_pomdp import (
    UNKNOWN_CATEGORY,
    FirefightingState,
    FirefightingMetrics,
    FirefightingPOMDP,
    FirefightingStepChannel,
    create_firefighting_state,
)
from POMDPPlanners.environments.firefighting_pomdp.firefighting_world import (
    DIRECTION_OFFSETS,
    HEAT_DAMAGE,
    MAX_HEAT_DAMAGE_PER_STEP,
    NUM_CATEGORIES,
    NUM_FIREFIGHTER_ACTIONS,
    NUM_WIND_VALUES,
    FIREFIGHTER_FIELD_WIDTH,
    FIREFIGHTER_OFFSET,
    STEP_INDEX,
    FireCategory,
    FirefightingAction,
    FirefightingInitialStateDistribution,
    WindDirection,
    WindStrength,
    default_depot_cell,
    default_obstacle_cells,
    default_firefighter_start_cells,
)

__all__ = [
    "DIRECTION_OFFSETS",
    "FireCategory",
    "FirefightingAction",
    "FirefightingState",
    "HEAT_DAMAGE",
    "MAX_HEAT_DAMAGE_PER_STEP",
    "FirefightingVectorizedBelief",
    "FirefightingVectorizedModel",
    "FirefightingVectorizedUpdater",
    "FirefightingInitialStateDistribution",
    "FirefightingMetrics",
    "FirefightingPOMDP",
    "FirefightingStepChannel",
    "FirefightingVisualizer",
    "create_firefighting_belief",
    "NUM_CATEGORIES",
    "NUM_FIREFIGHTER_ACTIONS",
    "NUM_WIND_VALUES",
    "FIREFIGHTER_FIELD_WIDTH",
    "FIREFIGHTER_OFFSET",
    "STEP_INDEX",
    "UNKNOWN_CATEGORY",
    "WindDirection",
    "WindStrength",
    "create_firefighting_state",
    "default_depot_cell",
    "default_obstacle_cells",
    "default_firefighter_start_cells",
]

from .firefighting_vectorized_belief import (
    FirefightingVectorizedBelief,
    FirefightingVectorizedUpdater,
    create_firefighting_belief,
)
from .firefighting_vectorized_model import FirefightingVectorizedModel
from .firefighting_visualization import FirefightingVisualizer
