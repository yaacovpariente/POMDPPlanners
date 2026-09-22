# SPDX-License-Identifier: MIT

"""RockSample episode trace exporter.

The GIF renderer beside this module draws an episode. This one writes the same
episode as data, so the browser viewer can replay it. Nothing here is
re-derived: every number written comes from the recorded episode or from the
environment instance the episode ran on.

The belief is not serialized here. It is a core abstraction with a closed
family of implementations, so
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, and this exporter is left with what is genuinely
RockSample's: the grid, the rocks, the hazard, the rover's cells, and which
rock each check was aimed at.

That last field is the reason this payload is more than a position list. A
RockSample belief is per-rock — the probability each rock is good — and it
moves only when a ``check_rock_i`` action returns a reading. The viewer draws
that update as an event, so it needs to know which rock was checked and what
came back. Both are already in the episode: the action names the rock, the
observation is the reading. They are pulled out here, once, using the
environment's own action layout, rather than left for a reader to reconstruct
from an integer whose meaning depends on how many rocks the run was configured
with.

The per-rock posterior itself is *not* written. It is a marginal of the belief
core already serialized, and a viewer computes it by summing the weights of the
particles in which that rock is good. Writing it here as well would create a
second number that could disagree with the belief it claims to summarise.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps, to_jsonable

# Payload version, independent of the envelope's. Bump it when the meaning of
# a payload field changes, so a viewer can refuse a file it would misdraw.
ROCK_SAMPLE_PAYLOAD_KIND = "rock_sample.v1"

# Actions 0..4 are sample and the four moves; every later action checks one
# rock. Written into the payload rather than hard-coded in the viewer, because
# the index of the first check action is a property of this environment.
FIRST_CHECK_ACTION = 5


def _check_of(action: Any, observation: Any, num_rocks: int) -> Optional[Dict[str, Any]]:
    """Describe the sensor check a step made, if it made one.

    Args:
        action: The action recorded on the step. ``None`` on the terminal
            bookkeeping step, which took no action.
        observation: The observation recorded on the step: ``"good"`` or
            ``"bad"`` after a check, ``"none"`` otherwise.
        num_rocks: How many rocks the environment has, so an action beyond the
            check block is reported as no check rather than as rock 97.

    Returns:
        ``{"rock": index, "observation": reading}`` for a check step, and
        ``None`` for every other step.
    """
    if action is None:
        return None
    try:
        index = int(action) - FIRST_CHECK_ACTION
    except (TypeError, ValueError):
        return None
    if index < 0 or index >= num_rocks:
        return None
    return {"rock": index, "observation": to_jsonable(observation)}


def build_rock_sample_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one RockSample episode.

    Args:
        environment: The RockSample environment the episode was run on. Its
            grid, rocks, hazard and reward constants are copied into the
            payload's ``world`` block so a viewer can build the scene without
            importing Python.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind
        ``rock_sample.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    if not history:
        raise ValueError("Cannot export a trace for an empty history")

    num_rocks = len(environment.rock_positions)

    states: List[List[int]] = []
    rock_truth: List[List[bool]] = []
    observations: List[Any] = []
    checks: List[Optional[Dict[str, Any]]] = []
    beliefs: List[Dict[str, Any]] = []

    for step in history:
        state = np.asarray(step.state, dtype=float).reshape(-1)
        # The exit is recorded as the (-1, -1) sentinel, and it is written
        # through unchanged: it is where the episode says the rover went. The
        # viewer decides how to draw a rover that has left the grid; turning
        # the sentinel into a cell here would invent a position the run never
        # had.
        states.append([int(state[0]), int(state[1])])
        # The true rock qualities live in the state. The rover never observes
        # them, and the viewer draws them apart from the belief for exactly
        # that reason — the two are supposed to be able to disagree.
        rock_truth.append([bool(value > 0.5) for value in state[2 : 2 + num_rocks]])
        observations.append(to_jsonable(step.observation))
        checks.append(_check_of(step.action, step.observation, num_rocks))
        beliefs.append(belief_to_payload(step.belief))

    payload: Dict[str, Any] = {
        # Everything a viewer needs to build the world, taken from the
        # environment instance the episode actually ran on — not from the
        # class defaults, which a configured run may not be using.
        "world": {
            "map_size": [int(environment.map_size[0]), int(environment.map_size[1])],
            "rock_positions": [[int(r), int(c)] for r, c in environment.rock_positions],
            "init_pos": [int(environment.init_pos[0]), int(environment.init_pos[1])],
            "sensor_efficiency": float(environment.sensor_efficiency),
            # Which accuracy law the run used. Version 2 is Smith & Simmons'
            # ``(1 + 2 ** (-d / efficiency)) / 2``; version 1 was a decaying
            # exponential that fell below a coin flip. A viewer that prints
            # P(correct) has to know which one it is quoting.
            "sensor_contract_version": int(getattr(environment, "sensor_contract_version", 1)),
            "good_rock_reward": float(environment.good_rock_reward),
            "bad_rock_penalty": float(environment.bad_rock_penalty),
            "exit_reward": float(environment.exit_reward),
            "step_penalty": float(environment.step_penalty),
            "sensor_use_penalty": float(environment.sensor_use_penalty),
            "dangerous_areas": [[int(r), int(c)] for r, c in environment.dangerous_areas],
            "dangerous_area_radius": float(environment.dangerous_area_radius),
            "dangerous_area_penalty": float(environment.dangerous_area_penalty),
            "dangerous_area_hit_probability": float(environment.dangerous_area_hit_probability),
            "action_names": [str(name) for name in environment.action_names],
            "first_check_action": FIRST_CHECK_ACTION,
        },
        "states": states,
        "rock_truth": rock_truth,
        "observations": observations,
        "checks": checks,
        "beliefs": beliefs,
    }

    return EpisodeTrace(
        environment=str(environment.name),
        payload_kind=ROCK_SAMPLE_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload=payload,
        policy=policy_name,
        reach_terminal_state=bool(environment.is_terminal(history[-1].state)),
        metadata={"environment_class": type(environment).__name__},
    )
