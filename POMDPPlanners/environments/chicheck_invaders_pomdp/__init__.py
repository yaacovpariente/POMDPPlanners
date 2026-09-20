# SPDX-License-Identifier: MIT

"""Chicheck Invaders shooter POMDP package.

Exports:
    ChicheckInvadersPOMDP: The environment.
    ChicheckInvadersAction: Its four action indices.
    ChicheckInvadersMetrics: The metric names it reports.
    ChicheckInvadersStepChannel: The per-step channels behind those metrics.
    ObservationMode: Fully versus partially observable.
    ChicheckInvadersBelief: Weighted particle filter with flock reinvigoration.
    ChicheckInvadersVectorizedBelief: Its batched twin, and the environment's
        default belief.
    ChicheckInvadersInitialStateDistribution: The per-episode flock prior.
    ChicheckInvadersVisualizer: Episode renderer.
    CHICHECK_INVADERS_PAYLOAD_KIND: Payload kind of this environment's episode trace.
    build_chicheck_invaders_trace: Builds that trace from one episode.
    create_chicheck_invaders_belief: Builds the initial belief for an environment.
    create_chicheck_invaders_belief: Builds the scalar belief for an environment.
    create_chicheck_invaders_vectorized_belief: Builds the batched one.
    create_chicheck_invaders_state: Builds one state vector from its parts.
    noiseless_preset: Constructor keywords for the deterministic-sensor preset.
"""

from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_pomdp import (
    ChicheckInvadersAction,
    ChicheckInvadersMetrics,
    ChicheckInvadersPOMDP,
    ChicheckInvadersState,
    ChicheckInvadersStepChannel,
    ObservationMode,
    create_chicheck_invaders_state,
    noiseless_preset,
    resolve_observation_mode,
)
from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_schema import (
    CHICKEN_ALIVE,
    CHICKEN_COLUMN,
    CHICKEN_DIRECTION,
    CHICKEN_MODE,
    CHICKEN_ROW,
    CHICKEN_WIDTH,
    COOLDOWN_INDEX,
    MODE_DIVE,
    MODE_PATROL,
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
    ChicheckInvadersInitialStateDistribution,
    chicken_slots,
    make_state,
    observation_size,
    state_size,
)
from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_sensors import (
    camera_sees,
    radar_sees,
    rounded_normal_pmf,
    sample_rounded_normal,
)

__all__ = [
    "CHICHECK_INVADERS_PAYLOAD_KIND",
    "CHICKEN_ALIVE",
    "CHICKEN_COLUMN",
    "CHICKEN_DIRECTION",
    "CHICKEN_MODE",
    "CHICKEN_ROW",
    "CHICKEN_WIDTH",
    "COOLDOWN_INDEX",
    "ChicheckInvadersAction",
    "ChicheckInvadersBelief",
    "ChicheckInvadersInitialStateDistribution",
    "ChicheckInvadersMetrics",
    "ChicheckInvadersPOMDP",
    "ChicheckInvadersState",
    "ChicheckInvadersStepChannel",
    "ChicheckInvadersVectorizedBelief",
    "ChicheckInvadersVectorizedUpdater",
    "ChicheckInvadersVisualizer",
    "MODE_DIVE",
    "MODE_PATROL",
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
    "build_chicheck_invaders_trace",
    "camera_sees",
    "chicken_slots",
    "create_chicheck_invaders_belief",
    "create_chicheck_invaders_state",
    "create_chicheck_invaders_vectorized_belief",
    "make_state",
    "noiseless_preset",
    "observation_size",
    "radar_sees",
    "resolve_observation_mode",
    "rounded_normal_pmf",
    "sample_rounded_normal",
    "state_size",
]

from .chicheck_invaders_belief import ChicheckInvadersBelief, create_chicheck_invaders_belief
from .visualizer import (
    CHICHECK_INVADERS_PAYLOAD_KIND,
    ChicheckInvadersVisualizer,
    build_chicheck_invaders_trace,
)
from .chicheck_invaders_vectorized_belief import (
    ChicheckInvadersVectorizedBelief,
    ChicheckInvadersVectorizedUpdater,
    create_chicheck_invaders_vectorized_belief,
)
