# SPDX-License-Identifier: MIT

"""Chicheck Invaders episode trace exporter.

The GIF renderer beside this module draws an episode. This one writes the same
episode as data, so the browser viewer can replay it. Nothing here is
re-derived: every number written comes from the recorded episode or from the
environment's own configuration and its own predicates.

The belief is not serialized here. It is a core abstraction with a closed
family of implementations, so
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, and this exporter is left with what is genuinely
Chicheck Invaders': the grid, the sensor parameters, the recorded state
vectors, the readings, and what the gun did.

States are written as the raw ``float64`` vectors the run actually held, and
the payload carries the field offsets that index them. That is deliberate: a
belief particle *is* a state vector, so the viewer needs a decoder for the
layout regardless, and writing states in a second, decoded shape would give it
two decoders that can drift apart. One layout block serves both.

Nothing records a projectile, because there is none: the gun is hitscan and a
shot resolves inside the step that fired it. What the step leaves behind is
which slot it killed, and that is what ``shots`` carries.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps, to_jsonable
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
    SHIP_WIDTH,
    STEP_INDEX,
    chicken_slots,
)

# Payload version, independent of the envelope's. Bump it when the meaning of a
# payload field changes, so a viewer can refuse a file it would misdraw.
CHICHECK_INVADERS_PAYLOAD_KIND = "chicheck_invaders.v1"


def _layout() -> Dict[str, Any]:
    """The state and observation field offsets, so a viewer can decode a vector.

    Written into every trace rather than hard-coded in the viewer: the layout is
    the environment's, and a viewer that carried its own copy would keep drawing
    confidently after the layout changed under it.

    Returns:
        The two offset tables, plus the two ``mode`` values.
    """
    return {
        "state": {
            "step": int(STEP_INDEX),
            "ship_column": int(SHIP_COLUMN_INDEX),
            "cooldown": int(COOLDOWN_INDEX),
            "ship_hit": int(SHIP_HIT_INDEX),
            "ship_width": int(SHIP_WIDTH),
            "chicken_width": int(CHICKEN_WIDTH),
            "chicken_column": int(CHICKEN_COLUMN),
            "chicken_row": int(CHICKEN_ROW),
            "chicken_direction": int(CHICKEN_DIRECTION),
            "chicken_mode": int(CHICKEN_MODE),
            "chicken_alive": int(CHICKEN_ALIVE),
            "mode_patrol": float(MODE_PATROL),
            "mode_dive": float(MODE_DIVE),
        },
        "observation": {
            "ship_column": int(OBSERVED_SHIP_COLUMN_INDEX),
            "ship_width": int(OBSERVATION_SHIP_WIDTH),
            "chicken_width": int(OBSERVATION_CHICKEN_WIDTH),
            "camera_reported": int(OBSERVED_CAMERA_REPORTED),
            "camera_offset": int(OBSERVED_CAMERA_OFFSET),
            "radar_reported": int(OBSERVED_RADAR_REPORTED),
            "radar_rows": int(OBSERVED_RADAR_ROWS),
            "radar_drop": int(OBSERVED_RADAR_DROP),
        },
    }


def _world(environment: Any) -> Dict[str, Any]:
    """The configuration the episode ran under, copied off the instance.

    Off the instance rather than off the class defaults, because a configured
    run may not be using them, and a viewer that drew the default cone over an
    episode recorded with a wider one would be drawing a sensor nobody had.

    Args:
        environment: The environment the episode ran on.

    Returns:
        The world block of the payload.
    """
    return {
        "num_columns": int(environment.num_columns),
        "num_rows": int(environment.num_rows),
        "num_chickens": int(environment.num_chickens),
        # The two sensor parameters the viewer draws footprints from. They are
        # the same numbers ``camera_sees`` and ``radar_sees`` test against.
        "camera_slope": float(environment.camera_slope),
        "radar_radius": float(environment.radar_radius),
        "ship_start_column": int(environment.ship_start_column),
        "fire_cooldown": int(environment.fire_cooldown),
        "max_steps": int(environment.max_steps),
        "dive_probability": float(environment.dive_probability),
        "initial_dive_probability": float(environment.initial_dive_probability),
        "camera_detection_probability": float(environment.camera_detection_probability),
        "radar_detection_probability": float(environment.radar_detection_probability),
        "camera_offset_noise_std": float(environment.camera_offset_noise_std),
        "radar_range_noise_std": float(environment.radar_range_noise_std),
        "ship_column_noise_std": float(environment.ship_column_noise_std),
        "drop_flag_error_probability": float(environment.drop_flag_error_probability),
        "kill_reward": float(environment.kill_reward),
        "shot_cost": float(environment.shot_cost),
        "step_cost": float(environment.step_cost),
        "ship_hit_penalty": float(environment.ship_hit_penalty),
        "clear_reward": float(environment.clear_reward),
        "observation_mode": str(environment.observation_mode.value),
    }


def _shot(environment: Any, state: np.ndarray, action: Any) -> Dict[str, Any]:
    """What the gun did on one step, asked of the environment rather than guessed.

    ``fires`` and ``shot_target`` are the environment's own; calling them is
    what keeps a drawn beam from disagreeing with the kill the episode actually
    scored. ``shot_target`` is asked of ``state`` rather than of the successor
    the transition aims from, which is the same question: ``FIRE`` never moves
    the ship, so the successor's column and flock are the ones here.

    Args:
        environment: The environment the episode ran on.
        state: The state the step was taken from.
        action: The action taken, or ``None`` on the terminal bookkeeping step.

    Returns:
        ``fired``, the slot it killed (``-1`` for a miss) and that slot's row.
    """
    if action is None or not environment.fires(state, action):
        return {"fired": False, "target_slot": -1, "target_row": -1}
    slot = int(environment.shot_target(state))
    if slot < 0:
        return {"fired": True, "target_slot": -1, "target_row": -1}
    flock = chicken_slots(np.asarray(state, dtype=np.float64), environment.num_chickens)
    return {"fired": True, "target_slot": slot, "target_row": int(flock[slot, CHICKEN_ROW])}


def build_chicheck_invaders_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one Chicheck Invaders episode.

    Args:
        environment: The Chicheck Invaders environment the episode was run on.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind
        ``chicheck_invaders.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    if not history:
        raise ValueError("Cannot export a trace for an empty history")

    states: List[List[float]] = []
    next_states: List[Optional[List[float]]] = []
    observations: List[Any] = []
    beliefs: List[Dict[str, Any]] = []
    shots: List[Dict[str, Any]] = []

    for step in history:
        state = np.asarray(step.state, dtype=float).reshape(-1)
        states.append([float(value) for value in state])
        if step.next_state is None:
            next_states.append(None)
        else:
            successor = np.asarray(step.next_state, dtype=float).reshape(-1)
            next_states.append([float(value) for value in successor])
        observations.append(to_jsonable(step.observation))
        beliefs.append(belief_to_payload(step.belief))
        shots.append(_shot(environment, state, step.action))

    payload: Dict[str, Any] = {
        "world": _world(environment),
        "layout": _layout(),
        "states": states,
        "next_states": next_states,
        "observations": observations,
        "beliefs": beliefs,
        "shots": shots,
    }

    return EpisodeTrace(
        environment=str(environment.name),
        payload_kind=CHICHECK_INVADERS_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload=payload,
        policy=policy_name,
        reach_terminal_state=bool(environment.is_terminal(history[-1].state)),
        metadata={"environment_class": type(environment).__name__},
    )
