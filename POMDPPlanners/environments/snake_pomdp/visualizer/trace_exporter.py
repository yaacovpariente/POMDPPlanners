# SPDX-License-Identifier: MIT

"""Snake episode trace exporter.

The GIF renderer beside this module draws an episode. This one writes the same
episode as data, so the browser viewer can replay it. Nothing here is
re-derived from anything but the recorded episode and the environment the
episode ran on.

The belief is not serialized here. It is a core abstraction with a closed
family of implementations, so
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, and this exporter is left with what is genuinely
Snake's: the board, the body, the apple, and the readings.

Two alignments in here are easy to get wrong and are the reason the GIF
renderer and this file agree frame for frame:

* a :class:`~POMDPPlanners.core.simulation.StepData` records the state a step
  was taken *from* and the belief the agent held *before* acting, so body,
  apple and belief at index ``i`` all describe the same moment: the one the
  agent chose in;
* its ``observation``, by contrast, is what that step's transition *produced*.
  The reading the agent actually chose on is the previous step's, so the
  sighting and the scent are shifted by one and step 0 has none. Writing the
  step's own observation beside its board would credit the agent with a reading
  it had not received yet.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps, to_jsonable
from POMDPPlanners.environments.snake_pomdp.snake_pomdp import (
    BODY_OFFSET,
    FOOD_COL_INDEX,
    FOOD_ROW_INDEX,
    LENGTH_INDEX,
    STATUS_INDEX,
    STEPS_SINCE_FOOD_INDEX,
    SnakeAction,
    SnakeTermination,
)

# Payload version, independent of the envelope's. Bump it when the meaning of
# a payload field changes, so a viewer can refuse a file it would misdraw.
SNAKE_PAYLOAD_KIND = "snake.v1"


def _reading(environment: Any, observation: Any) -> Tuple[Optional[List[int]], Optional[int]]:
    """Decode one reading into a sighting and a scent.

    Args:
        environment: The Snake environment.
        observation: A reading this environment produced, or ``None``.

    Returns:
        ``(sighted_cell, scent_quadrant)``. Both are ``None`` for the terminal
        reading, which carries neither, and for a missing one.
    """
    if observation is None:
        return None, None
    flat = tuple(int(value) for value in observation)
    if not flat or flat[0] != 1:
        # The terminal reading. It names no body, sighting or scent by design.
        return None, None
    _, seen, scent = environment.decode_observation(flat)
    return (None if seen is None else [int(seen[0]), int(seen[1])]), int(scent)


def _turn_outcomes(environment: Any, state: Any) -> Optional[List[int]]:
    """Resolve what each of the three turns would do from ``state``.

    This is knowledge the agent has, not a peek at the hidden apple: the body
    is observed exactly and the walls are fixed, so which turns are fatal is
    something a planner can work out and a reader should be able to see. The
    apple only enters through eating, which never kills.

    Args:
        environment: The Snake environment.
        state: The state the step is taken from.

    Returns:
        One :class:`SnakeTermination` value per action, in action order, or
        ``None`` for a terminal state, from which no turn resolves.
    """
    if environment.is_terminal(state):
        return None
    return [int(environment.transition_outcome(state, int(action))[3]) for action in SnakeAction]


def build_snake_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one Snake episode.

    Args:
        environment: The Snake environment the episode was run on. Its board
            and its sensor settings are copied into the payload's ``world``
            block so a viewer can build the scene without importing Python, and
            they come from the instance the episode actually ran on rather than
            from the class defaults a configured run may not be using.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind ``snake.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    if not history:
        raise ValueError("Cannot export a trace for an empty history")

    bodies: List[List[List[int]]] = []
    foods: List[Optional[List[int]]] = []
    lengths: List[int] = []
    hunger: List[int] = []
    terminations: List[int] = []
    turns: List[Optional[List[int]]] = []
    sightings: List[Optional[List[int]]] = []
    scents: List[Optional[int]] = []
    observations: List[Any] = []
    beliefs: List[Dict[str, Any]] = []

    for index, step in enumerate(history):
        state = np.asarray(step.state, dtype=np.float64)
        bodies.append([[int(row), int(col)] for row, col in environment.body(state)])
        food = environment.food(state)
        foods.append(None if food is None else [int(food[0]), int(food[1])])
        lengths.append(int(environment.snake_length(state)))
        hunger.append(int(environment.steps_since_food(state)))
        terminations.append(int(environment.termination(state)))
        turns.append(_turn_outcomes(environment, state))

        # The reading the agent held when it chose here: the previous step's.
        previous = history[index - 1].observation if index > 0 else None
        seen, scent = _reading(environment, previous)
        sightings.append(seen)
        scents.append(scent)

        observations.append(to_jsonable(step.observation))
        beliefs.append(belief_to_payload(step.belief))

    payload: Dict[str, Any] = {
        "world": {
            "grid_size": int(environment.grid_size),
            "target_length": int(environment.target_length),
            "window_radius": int(environment.window_radius),
            "detection_probability": float(environment.detection_probability),
            "scent_accuracy": float(environment.scent_accuracy),
            "starvation_limit": int(environment.starvation_limit),
        },
        # Where each field sits inside a state vector. Written rather than
        # assumed, because the viewer reads the apple cell out of the belief's
        # own particles and a layout it hard-coded would go stale in silence
        # the day a slot moves.
        "state_layout": {
            "status": int(STATUS_INDEX),
            "length": int(LENGTH_INDEX),
            "steps_since_food": int(STEPS_SINCE_FOOD_INDEX),
            "food_row": int(FOOD_ROW_INDEX),
            "food_col": int(FOOD_COL_INDEX),
            "body_offset": int(BODY_OFFSET),
        },
        "terminations_by_value": {
            str(int(reason)): reason.name.lower() for reason in SnakeTermination
        },
        "bodies": bodies,
        "foods": foods,
        "lengths": lengths,
        "steps_since_food": hunger,
        "terminations": terminations,
        "turn_outcomes": turns,
        "sightings": sightings,
        "scents": scents,
        "observations": observations,
        "beliefs": beliefs,
    }

    return EpisodeTrace(
        environment=str(environment.name),
        payload_kind=SNAKE_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload=payload,
        policy=policy_name,
        reach_terminal_state=bool(environment.is_terminal(history[-1].state)),
        metadata={"environment_class": type(environment).__name__},
    )
