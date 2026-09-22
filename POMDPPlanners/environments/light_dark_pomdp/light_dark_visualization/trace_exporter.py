# SPDX-License-Identifier: MIT

"""Light-Dark episode trace exporter.

The GIF renderer beside this module draws an episode. This one writes the same
episode as data, so the browser viewer can replay it. Nothing here is
re-derived: every number written comes from the recorded episode or from the
environment's own configuration.

The belief is not serialized here. It is a core abstraction with a closed
family of implementations, so
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, and this exporter is left with what is genuinely
Light-Dark's: the world's geometry, the rover's states and the observations.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps, to_jsonable

# Payload version, independent of the envelope's. Bump it when the meaning of
# a payload field changes, so a viewer can refuse a file it would misdraw.
LIGHT_DARK_PAYLOAD_KIND = "light_dark.v1"


def build_light_dark_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one Light-Dark episode.

    Args:
        environment: The Light-Dark environment the episode was run on. Its
            geometry and reward constants are copied into the payload's
            ``world`` block so a viewer can build the scene without importing
            Python.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind
        ``light_dark.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    if not history:
        raise ValueError("Cannot export a trace for an empty history")

    states: List[List[float]] = []
    next_states: List[Optional[List[float]]] = []
    observations: List[Any] = []
    beliefs: List[Dict[str, Any]] = []

    for step in history:
        position = np.asarray(step.state, dtype=float).reshape(-1)
        states.append([float(position[0]), float(position[1])])
        if step.next_state is None:
            next_states.append(None)
        else:
            nxt = np.asarray(step.next_state, dtype=float).reshape(-1)
            next_states.append([float(nxt[0]), float(nxt[1])])
        observations.append(to_jsonable(step.observation))
        beliefs.append(belief_to_payload(step.belief))

    beacons = np.asarray(environment.beacons, dtype=float)
    obstacles = np.asarray(environment.obstacles, dtype=float)

    payload: Dict[str, Any] = {
        # Everything a viewer needs to build the world, taken from the
        # environment instance the episode actually ran on — not from the
        # class defaults, which a configured run may not be using.
        "world": {
            "grid_size": int(environment.grid_size),
            "beacons": beacons.T.tolist() if beacons.size else [],
            "obstacles": obstacles.T.tolist() if obstacles.size else [],
            "obstacle_radius": float(environment.obstacle_radius),
            "beacon_radius": float(environment.beacon_radius),
            "goal_state": [float(v) for v in np.asarray(environment.goal_state).reshape(-1)[:2]],
            "start_state": [float(v) for v in np.asarray(environment.start_state).reshape(-1)[:2]],
            "goal_reward": float(environment.goal_reward),
            "obstacle_reward": float(environment.obstacle_reward),
            "obstacle_hit_probability": float(environment.obstacle_hit_probability),
            "fuel_cost": float(environment.fuel_cost),
        },
        "states": states,
        "next_states": next_states,
        "observations": observations,
        "beliefs": beliefs,
    }

    return EpisodeTrace(
        environment=str(environment.name),
        payload_kind=LIGHT_DARK_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload=payload,
        policy=policy_name,
        reach_terminal_state=bool(environment.is_terminal(history[-1].state)),
        metadata={"environment_class": type(environment).__name__},
    )
