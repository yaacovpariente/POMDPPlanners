# SPDX-License-Identifier: MIT

"""CaptureTheFlag episode trace exporter.

The GIF renderer beside this module draws an episode. This one writes the same
episode as data, so the browser viewer can replay it. Nothing here is
re-derived: every number written comes from the recorded episode or from the
environment's own configuration.

The belief is not serialized here. It is a core abstraction with a closed
family of implementations, so
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, and this exporter is left with what is genuinely
CaptureTheFlag's: the field, both teams' cells, and the bookkeeping a viewer
needs to say who carries what.

One field here exists only because of how this environment's belief is shaped.
A CaptureTheFlag particle is a whole flat state vector, so a reader that got
the particles alone would have a list of numbers and no way to find the red
players or the flag inside them. ``state_layout`` is the environment's own
:class:`StateLayout` offsets, written out once, so the viewer reads the same
indices the environment does instead of re-deriving index arithmetic that
would silently drift the first time a team size changes.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps, to_jsonable
from POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_pomdp_utils import (
    decode_joint_action,
)

# Payload version, independent of the envelope's. Bump it when the meaning of
# a payload field changes, so a viewer can refuse a file it would misdraw.
CAPTURE_THE_FLAG_PAYLOAD_KIND = "capture_the_flag.v1"


def _cells(environment: Any, state: Any, blue: bool) -> List[List[int]]:
    """Read one team's cells out of a state vector."""
    reader = environment.layout.blue_cells if blue else environment.layout.red_cells
    return [[int(cell[0]), int(cell[1])] for cell in reader(np.asarray(state, dtype=float))]


def _state_layout(environment: Any) -> Dict[str, int]:
    """Write the state vector's index offsets, for a reader of raw particles."""
    layout = environment.layout
    return {
        "blue_pos": int(layout.blue_pos),
        "red_pos": int(layout.red_pos),
        "flag_cell": int(layout.flag_cell),
        "carrier_red_flag": int(layout.carrier_red_flag),
        "carrier_blue_flag": int(layout.carrier_blue_flag),
        "freeze_blue": int(layout.freeze_blue),
        "freeze_red": int(layout.freeze_red),
        "cooldown_blue": int(layout.cooldown_blue),
        "cooldown_red": int(layout.cooldown_red),
        "score_blue": int(layout.score_blue),
        "score_red": int(layout.score_red),
        "size": int(layout.size),
    }


def _world(environment: Any) -> Dict[str, Any]:
    """Write the field a viewer has to build, from the configured instance.

    Taken from the environment the episode actually ran on, never from the
    class defaults: a run configured onto a smaller field would otherwise be
    replayed on the default one, with the units walking through trees that are
    not there.
    """
    return {
        "grid_size": [int(environment.grid_size[0]), int(environment.grid_size[1])],
        "midline": int(environment.midline),
        "trees": [[int(x), int(y)] for x, y in environment.trees],
        "n_blue": int(environment.n_blue),
        "n_red": int(environment.n_red),
        "red_roles": [role.value for role in environment.red_roles],
        "blue_base": [int(v) for v in environment.blue_base],
        "red_base": [int(v) for v in environment.red_base],
        "blue_flag_cell": [int(v) for v in environment.blue_flag_cell],
        "red_flag_candidates": [[int(x), int(y)] for x, y in environment.red_flag_candidates],
        "red_alert_radius": int(environment.red_alert_radius),
        "freeze_steps": int(environment.freeze_steps),
        "detector_half_distance_move": float(environment.detector_half_distance_move),
        "detector_half_distance_scan": float(environment.detector_half_distance_scan),
        "score_to_win": int(environment.score_to_win),
        "capture_reward": float(environment.capture_reward),
        "concede_penalty": float(environment.concede_penalty),
        "tagged_penalty": float(environment.tagged_penalty),
        "tag_reward": float(environment.tag_reward),
        "pickup_reward": float(environment.pickup_reward),
        "move_cost": float(environment.move_cost),
        "scan_cost": float(environment.scan_cost),
    }


def build_capture_the_flag_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one CaptureTheFlag episode.

    Args:
        environment: The CaptureTheFlag environment the episode was run on. Its
            field and reward constants are copied into the payload's ``world``
            block so a viewer can build the scene without importing Python.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind
        ``capture_the_flag.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    if not history:
        raise ValueError("Cannot export a trace for an empty history")

    layout = environment.layout
    blue_cells: List[List[List[int]]] = []
    red_cells: List[List[List[int]]] = []
    red_flag_cells: List[List[int]] = []
    carrier_red_flag: List[int] = []
    carrier_blue_flag: List[int] = []
    freeze_blue: List[List[int]] = []
    freeze_red: List[List[int]] = []
    scores: List[List[int]] = []
    player_actions: List[Optional[List[int]]] = []
    observations: List[Any] = []
    beliefs: List[Dict[str, Any]] = []

    for step in history:
        state = np.asarray(step.state, dtype=float)
        blue_cells.append(_cells(environment, state, blue=True))
        red_cells.append(_cells(environment, state, blue=False))
        red_flag_cells.append([int(v) for v in environment.red_flag_cell(state)])
        carrier_red_flag.append(int(state[layout.carrier_red_flag]))
        carrier_blue_flag.append(int(state[layout.carrier_blue_flag]))
        freeze_blue.append([int(state[layout.freeze_blue + i]) for i in range(environment.n_blue)])
        freeze_red.append([int(state[layout.freeze_red + j]) for j in range(environment.n_red)])
        scores.append([int(state[layout.score_blue]), int(state[layout.score_red])])
        # The envelope already carries the joint action id. The per-player
        # split is written too, because the viewer labels each player's move
        # and a scan is the only action with a visible effect of its own --
        # decoding a base-6 odometer in JavaScript is a second place for the
        # team size to be assumed rather than read.
        if step.action is None:
            player_actions.append(None)
        else:
            player_actions.append(
                [int(a) for a in decode_joint_action(int(step.action), environment.n_blue)]
            )
        observations.append(to_jsonable(step.observation))
        beliefs.append(belief_to_payload(step.belief))

    payload: Dict[str, Any] = {
        "world": _world(environment),
        "state_layout": _state_layout(environment),
        "blue_cells": blue_cells,
        "red_cells": red_cells,
        "red_flag_cell": red_flag_cells,
        "carrier_red_flag": carrier_red_flag,
        "carrier_blue_flag": carrier_blue_flag,
        "freeze_blue": freeze_blue,
        "freeze_red": freeze_red,
        "scores": scores,
        "player_actions": player_actions,
        "observations": observations,
        "beliefs": beliefs,
    }

    return EpisodeTrace(
        environment=str(environment.name),
        payload_kind=CAPTURE_THE_FLAG_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload=payload,
        policy=policy_name,
        reach_terminal_state=bool(environment.is_terminal(history[-1].state)),
        metadata={"environment_class": type(environment).__name__},
    )
