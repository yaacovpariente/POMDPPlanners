# SPDX-License-Identifier: MIT

"""Maze episode trace exporter, for the discrete and continuous variants alike.

``maze_visualizer`` draws an episode as a GIF. This module writes the same
episode as data, so the browser viewer can replay it. Nothing here is
re-derived: every number comes from the recorded episode or from the
environment instance the episode ran on.

The belief is not serialized here.
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, so this exporter is left with what is genuinely the
Maze's: the generated map, the agent's positions, the cue's phase and the
observations.

What the viewer has to do with that belief is particular to this task, though.
The hidden variable is one bit — which of the two corner goals pays — so the
belief's *positions* say nothing: every particle sits where the agent is,
because movement is deterministic and fully observed. The information is in how
the particle mass splits between ``goal_side == GOAL_LEFT`` and
``GOAL_RIGHT``. The payload therefore records the state layout
(:data:`~POMDPPlanners.environments.maze_pomdp.maze_pomdp.STATE_GOAL` and the
two side encodings) so the scene can do that split itself, rather than the
exporter collapsing the belief and shipping two numbers that could no longer be
checked against the run.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps, to_jsonable
from POMDPPlanners.environments.maze_pomdp.maze_pomdp import (
    CUE_CONSUMED,
    CUE_EMITTING,
    CUE_UNSEEN,
    GOAL_LEFT,
    GOAL_RIGHT,
    STATE_CUE_PHASE,
    STATE_GOAL,
    STATE_X,
    STATE_Y,
)

# Payload version, independent of the envelope's. Bump it when the meaning of a
# payload field changes, so a viewer can refuse a file it would misdraw.
MAZE_PAYLOAD_KIND = "maze.v1"


def _cell(cell: Any) -> List[int]:
    """One grid cell as a JSON pair."""
    return [int(cell[0]), int(cell[1])]


def _position(state: Any) -> List[float]:
    """The ``(x, y)`` of a maze state, as floats in grid coordinates."""
    array = np.asarray(state, dtype=np.float64).reshape(-1)
    return [float(array[STATE_X]), float(array[STATE_Y])]


def _walkable_cells(environment: Any) -> List[List[int]]:
    """Every walkable cell, sorted.

    The set the geometry holds has no order, and an unordered payload would
    make two traces of the same map differ byte for byte. Sorting costs
    nothing here and makes the file comparable.
    """
    return [_cell(cell) for cell in sorted(environment.walkable_cells)]


def _world_block(environment: Any) -> Dict[str, Any]:
    """Everything a viewer needs to build the maze, from this instance.

    Read off the environment the episode actually ran on rather than from the
    class defaults, because a configured run may be using neither the default
    size nor the default seed, and a viewer that drew the defaults would draw
    a different maze from the one the agent walked.
    """
    return {
        "width": int(environment.maze_width),
        "height": int(environment.maze_height),
        "maze_seed": int(environment.maze_seed),
        "loop_fraction": float(environment.loop_fraction),
        "walkable": _walkable_cells(environment),
        "start_cell": _cell(environment.start_cell),
        "cue_cell": _cell(environment.cue_cell),
        "left_goal_cell": _cell(environment.left_goal_cell),
        "right_goal_cell": _cell(environment.right_goal_cell),
        "cue_accuracy": float(environment.cue_accuracy),
        "goal_reward": float(environment.goal_reward),
        "wrong_goal_penalty": float(environment.wrong_goal_penalty),
        "step_penalty": float(environment.step_penalty),
        # True where a position is a cell and the step is a cell hop, false
        # where positions are real. The scene rules cell guides on that same
        # rule the GIF renderer uses, so the two agree about what a step is.
        "draws_cell_guides": bool(getattr(environment, "draws_cell_guides", True)),
        # Present only on the continuous variant; ``null`` says "cell moves".
        "max_step_size": (
            float(environment.max_step_size) if hasattr(environment, "max_step_size") else None
        ),
        # The state layout, so the scene can split a serialized belief by goal
        # side without importing Python or hard-coding an index.
        "state_goal_index": int(STATE_GOAL),
        "state_cue_phase_index": int(STATE_CUE_PHASE),
        "goal_left": float(GOAL_LEFT),
        "goal_right": float(GOAL_RIGHT),
        "cue_unseen": float(CUE_UNSEEN),
        "cue_emitting": float(CUE_EMITTING),
        "cue_consumed": float(CUE_CONSUMED),
    }


def _episode_arrays(
    history: List[StepData],
) -> Tuple[List[List[float]], List[Optional[List[float]]], List[float], List[Any], List[Any]]:
    """Split the recorded episode into the per-step lists the payload carries."""
    states: List[List[float]] = []
    next_states: List[Optional[List[float]]] = []
    cue_phases: List[float] = []
    observations: List[Any] = []
    beliefs: List[Any] = []

    for step in history:
        state_array = np.asarray(step.state, dtype=np.float64).reshape(-1)
        states.append(_position(state_array))
        cue_phases.append(float(state_array[STATE_CUE_PHASE]))
        next_states.append(None if step.next_state is None else _position(step.next_state))
        observations.append(to_jsonable(step.observation))
        beliefs.append(belief_to_payload(step.belief))

    return states, next_states, cue_phases, observations, beliefs


def build_maze_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one Maze episode, discrete or continuous.

    Args:
        environment: The Maze environment the episode ran on. Its generated
            geometry and reward constants are copied into the payload's
            ``world`` block so a viewer can build the scene without importing
            Python.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind ``maze.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    if not history:
        raise ValueError("Cannot export a trace for an empty history")

    states, next_states, cue_phases, observations, beliefs = _episode_arrays(history)

    # The hidden side is constant through an episode and is observer
    # information, exactly like the GIF's "True goal" caption: it is what makes
    # a replay legible, and it is never handed to the belief widget.
    true_goal_side = float(np.asarray(history[0].state, dtype=np.float64).reshape(-1)[STATE_GOAL])

    payload: Dict[str, Any] = {
        "world": _world_block(environment),
        "states": states,
        "next_states": next_states,
        "cue_phases": cue_phases,
        "observations": observations,
        "beliefs": beliefs,
        "true_goal_side": true_goal_side,
    }

    return EpisodeTrace(
        environment=str(environment.name),
        payload_kind=MAZE_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload=payload,
        policy=policy_name,
        reach_terminal_state=bool(environment.is_terminal(history[-1].state)),
        metadata={"environment_class": type(environment).__name__},
    )
