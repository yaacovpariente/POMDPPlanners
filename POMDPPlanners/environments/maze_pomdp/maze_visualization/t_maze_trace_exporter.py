# SPDX-License-Identifier: MIT

"""T-Maze episode trace exporter.

``MazeVisualizer`` draws a T-Maze episode as a GIF. This module writes the same
episode as data, so the browser viewer can replay it. Nothing here is
re-derived: every number written comes from the recorded episode or from the
environment instance the episode actually ran on.

The one field this environment cannot leave out is the **cue phase**. T-Maze is
a memory task: the cue fires once, one cell above the start, and every
observation after that is ``"empty"``. A trace that carried only positions and
a belief would replay as an agent walking down a corridor for no reason —
the phase is what says when the single reading happened and that it is gone.
It is already in the state, at :data:`STATE_CUE_PHASE`; it is also written out
as a per-step label so a reader does not have to know that ``1.0`` means
"emitting".

The belief is not serialized here. It is a core abstraction with a closed
family of implementations, so
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, and this exporter is left with what is genuinely
T-Maze's: the corridor's geometry, the agent's states, the observations and the
cue's delivery phase.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps, to_jsonable
from POMDPPlanners.environments.maze_pomdp.t_maze_pomdp import (
    CUE_CONSUMED,
    CUE_EMITTING,
    CUE_UNSEEN,
    GOAL_LEFT,
    OBSERVATION_LEFT_CUE,
    OBSERVATION_RIGHT_CUE,
    STATE_CUE_PHASE,
    STATE_GOAL,
    STATE_WIDTH,
    STATE_X,
    STATE_Y,
)

# Payload version, independent of the envelope's. Bump it when the meaning of a
# payload field changes, so a viewer can refuse a file it would misdraw.
T_MAZE_PAYLOAD_KIND = "t_maze.v1"

# The phase names a reader switches on. They are the drawing contract; the
# float encodings stay in the environment, where they are the model.
CUE_PHASE_NAMES: Dict[float, str] = {
    CUE_UNSEEN: "unseen",
    CUE_EMITTING: "emitting",
    CUE_CONSUMED: "consumed",
}

# Which arm a cue reading names. The observation alphabet is the environment's;
# this is only the side each label points at.
CUE_READING_SIDES: Dict[str, str] = {
    OBSERVATION_LEFT_CUE: "left",
    OBSERVATION_RIGHT_CUE: "right",
}


def _cue_phase_name(state: np.ndarray) -> str:
    """Name the cue's delivery phase in ``state``.

    An unrecognised value is reported as ``"unknown"`` rather than silently
    rounded to a neighbour: a phase a reader cannot name is a bug worth seeing,
    and drawing it as "consumed" would hide it behind a plausible picture.
    """
    return CUE_PHASE_NAMES.get(float(state[STATE_CUE_PHASE]), "unknown")


def _side_name(goal_side: float) -> str:
    """``"left"`` or ``"right"`` for a goal-side encoding."""
    return "left" if float(goal_side) == GOAL_LEFT else "right"


def build_t_maze_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one T-Maze episode.

    Args:
        environment: The T-Maze the episode was run on. Its geometry and reward
            constants are copied into the payload's ``world`` block so a viewer
            can build the corridor without importing Python.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind ``t_maze.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    if not history:
        raise ValueError("Cannot export a trace for an empty history")

    states: List[List[float]] = []
    next_states: List[Optional[List[float]]] = []
    observations: List[Any] = []
    cue_phases: List[str] = []
    beliefs: List[Dict[str, Any]] = []

    for step in history:
        state = np.asarray(step.state, dtype=float).reshape(-1)
        states.append([float(v) for v in state[:STATE_WIDTH]])
        cue_phases.append(_cue_phase_name(state))
        if step.next_state is None:
            next_states.append(None)
        else:
            nxt = np.asarray(step.next_state, dtype=float).reshape(-1)
            next_states.append([float(v) for v in nxt[:STATE_WIDTH]])
        observations.append(to_jsonable(step.observation))
        beliefs.append(belief_to_payload(step.belief))

    # The one reading of the episode, taken from the observations that were
    # actually received. It is read off the recorded episode rather than from
    # the true goal side, because a noisy cue is allowed to lie and a viewer
    # that drew the truth here would hide exactly that case.
    cue_reading: Optional[str] = None
    for observation in observations:
        side = CUE_READING_SIDES.get(observation) if isinstance(observation, str) else None
        if side is not None:
            cue_reading = side
            break

    payload: Dict[str, Any] = {
        # Everything a viewer needs to build the corridor, taken from the
        # environment instance the episode ran on — not from the class
        # defaults, which a configured run may not be using.
        "world": {
            "stem_length": int(environment.stem_length),
            "arm_length": int(environment.arm_length),
            "cells": [[int(x), int(y)] for x, y in sorted(environment.valid_cells)],
            "start_cell": [int(v) for v in environment.start_cell],
            "cue_cell": [int(v) for v in environment.cue_cell],
            "junction": [int(v) for v in environment.junction],
            "left_endpoint": [int(v) for v in environment.left_endpoint],
            "right_endpoint": [int(v) for v in environment.right_endpoint],
            "cue_accuracy": float(environment.cue_accuracy),
            "goal_reward": float(environment.goal_reward),
            "wrong_goal_penalty": float(environment.wrong_goal_penalty),
            "step_penalty": float(environment.step_penalty),
            # Where each quantity sits in a state vector, so a reader can take
            # the goal side out of a belief particle without hard-coding an
            # index this module would be free to change.
            "state_slots": {
                "x": int(STATE_X),
                "y": int(STATE_Y),
                "goal_side": int(STATE_GOAL),
                "cue_phase": int(STATE_CUE_PHASE),
            },
            "goal_left_value": float(GOAL_LEFT),
        },
        "states": states,
        "next_states": next_states,
        "observations": observations,
        # The memory task's own channel: which phase the cue was in at each
        # recorded state, named rather than left as the model's float.
        "cue_phases": cue_phases,
        "cue_reading": cue_reading,
        # Observer furniture. The agent never sees this; the viewer marks the
        # paying endpoint with it so a human can tell a correct turn from a
        # lucky one.
        "goal_side": _side_name(states[0][STATE_GOAL]),
        "beliefs": beliefs,
    }

    return EpisodeTrace(
        environment=str(environment.name),
        payload_kind=T_MAZE_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload=payload,
        policy=policy_name,
        reach_terminal_state=bool(environment.is_terminal(history[-1].state)),
        metadata={"environment_class": type(environment).__name__},
    )


__all__ = [
    "CUE_PHASE_NAMES",
    "CUE_READING_SIDES",
    "T_MAZE_PAYLOAD_KIND",
    "build_t_maze_trace",
]
