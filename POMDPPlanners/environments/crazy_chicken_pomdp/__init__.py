# SPDX-License-Identifier: MIT

"""Crazy Chicken shooter POMDP package.

Exports:
    CrazyChickenPOMDP: The environment.
    CrazyChickenAction: Its four action indices.
    CrazyChickenMetrics: The metric names it reports.
    CrazyChickenStepChannel: The per-step channels behind those metrics.
    ObservationMode: Fully versus partially observable.
    CrazyChickenBelief: Weighted particle filter with flock reinvigoration.
    CrazyChickenInitialStateDistribution: The per-episode flock prior.
    CrazyChickenVisualizer: Episode renderer.
    create_crazy_chicken_belief: Builds the initial belief for an environment.
    create_crazy_chicken_state: Builds one state vector from its parts.
    noiseless_preset: Constructor keywords for the deterministic-sensor preset.
"""

from POMDPPlanners.environments.crazy_chicken_pomdp.crazy_chicken_pomdp import (
    CrazyChickenAction,
    CrazyChickenMetrics,
    CrazyChickenPOMDP,
    CrazyChickenState,
    CrazyChickenStepChannel,
    ObservationMode,
    create_crazy_chicken_state,
    noiseless_preset,
    resolve_observation_mode,
)
from POMDPPlanners.environments.crazy_chicken_pomdp.crazy_chicken_schema import (
    CHICKEN_ALIVE,
    CHICKEN_COLUMN,
    CHICKEN_DIRECTION,
    CHICKEN_MODE,
    CHICKEN_ROW,
    CHICKEN_WIDTH,
    COOLDOWN_INDEX,
    MODE_DIVE,
    MODE_PATROL,
    NO_PROJECTILE,
    OBSERVATION_CHICKEN_WIDTH,
    OBSERVATION_SHIP_WIDTH,
    OBSERVED_CAMERA_OFFSET,
    OBSERVED_CAMERA_REPORTED,
    OBSERVED_RADAR_DROP,
    OBSERVED_RADAR_REPORTED,
    OBSERVED_RADAR_ROWS,
    OBSERVED_SHIP_COLUMN_INDEX,
    SHIP_COLUMN_INDEX,
    SHIP_HIT_INDEX,
    STEP_INDEX,
    CrazyChickenInitialStateDistribution,
    chicken_slots,
    make_state,
    observation_size,
    projectile_rows,
    state_size,
)
from POMDPPlanners.environments.crazy_chicken_pomdp.crazy_chicken_sensors import (
    camera_sees,
    radar_sees,
    rounded_normal_pmf,
    sample_rounded_normal,
)

__all__ = [
    "CHICKEN_ALIVE",
    "CHICKEN_COLUMN",
    "CHICKEN_DIRECTION",
    "CHICKEN_MODE",
    "CHICKEN_ROW",
    "CHICKEN_WIDTH",
    "COOLDOWN_INDEX",
    "CrazyChickenAction",
    "CrazyChickenBelief",
    "CrazyChickenInitialStateDistribution",
    "CrazyChickenMetrics",
    "CrazyChickenPOMDP",
    "CrazyChickenState",
    "CrazyChickenStepChannel",
    "CrazyChickenVisualizer",
    "MODE_DIVE",
    "MODE_PATROL",
    "NO_PROJECTILE",
    "OBSERVATION_CHICKEN_WIDTH",
    "OBSERVATION_SHIP_WIDTH",
    "OBSERVED_CAMERA_OFFSET",
    "OBSERVED_CAMERA_REPORTED",
    "OBSERVED_RADAR_DROP",
    "OBSERVED_RADAR_REPORTED",
    "OBSERVED_RADAR_ROWS",
    "OBSERVED_SHIP_COLUMN_INDEX",
    "ObservationMode",
    "SHIP_COLUMN_INDEX",
    "SHIP_HIT_INDEX",
    "STEP_INDEX",
    "camera_sees",
    "chicken_slots",
    "create_crazy_chicken_belief",
    "create_crazy_chicken_state",
    "make_state",
    "noiseless_preset",
    "observation_size",
    "projectile_rows",
    "radar_sees",
    "resolve_observation_mode",
    "rounded_normal_pmf",
    "sample_rounded_normal",
    "state_size",
]

from .crazy_chicken_belief import CrazyChickenBelief, create_crazy_chicken_belief
from .crazy_chicken_visualizer import CrazyChickenVisualizer
