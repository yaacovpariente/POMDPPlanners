# SPDX-License-Identifier: MIT

"""Tiger episode trace exporter.

The GIF renderer beside this module draws an episode. This one writes the same
episode as data, so the browser viewer can replay it. Nothing here is
re-derived: every number written comes from the recorded episode or is read
back out of the environment's own models.

The belief is not serialized here. It is a core abstraction with a closed
family of implementations, so
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, and this exporter is left with what is genuinely
Tiger's: which side the tiger was on, what was heard, and the three constants a
viewer needs to label the room.

Those constants are read out of the environment rather than copied from this
package's module-level literals. ``listen_accuracy`` is
``exp(observation_log_probability(...))`` and the rewards are ``reward(...)``
calls, so a Tiger subclass that listens at 0.7 or pays a different penalty
exports its own numbers instead of the defaults a reader would then believe.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps, to_jsonable

# Payload version, independent of the envelope's. Bump it when the meaning of
# a payload field changes, so a viewer can refuse a file it would misdraw.
TIGER_PAYLOAD_KIND = "tiger.v1"


def _listen_accuracy(environment: Any) -> float:
    """Probability that a listen names the side the tiger is actually on."""
    log_probability = environment.observation_log_probability("tiger_left", "listen", ["hear_left"])
    return float(np.exp(np.asarray(log_probability, dtype=float).reshape(-1)[0]))


def _redraw_probabilities(environment: Any) -> List[float]:
    """Where the tiger ends up after a door is opened, as the model says.

    This is the field the whole environment turns on: opening a door is not
    terminal here, and the transition re-draws the tiger's side, so the belief
    the agent spent listens to earn is thrown away. A viewer that assumed the
    episode ends at the first open would draw the rest of it wrong.
    """
    log_probabilities = environment.transition_log_probability(
        "tiger_left", "open_left", list(environment.states)
    )
    return [float(value) for value in np.exp(np.asarray(log_probabilities, dtype=float))]


def build_tiger_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one Tiger episode.

    Args:
        environment: The Tiger environment the episode was run on. Its models
            are queried for the payload's ``world`` block, so a viewer can label
            the room without importing Python.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind ``tiger.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    if not history:
        raise ValueError("Cannot export a trace for an empty history")

    states: List[Any] = []
    next_states: List[Any] = []
    observations: List[Any] = []
    beliefs: List[Dict[str, Any]] = []

    for step in history:
        states.append(to_jsonable(step.state))
        next_states.append(to_jsonable(step.next_state))
        observations.append(to_jsonable(step.observation))
        beliefs.append(belief_to_payload(step.belief))

    payload: Dict[str, Any] = {
        # Everything a viewer needs to label the world, read back out of the
        # environment the episode actually ran on — not from the class
        # defaults, which a configured run may not be using.
        "world": {
            "states": [str(state) for state in environment.states],
            "actions": [str(action) for action in environment.get_actions()],
            "observations": [str(observation) for observation in environment.observations],
            "listen_accuracy": _listen_accuracy(environment),
            "listen_reward": float(environment.reward("tiger_left", "listen")),
            "correct_door_reward": float(environment.reward("tiger_left", "open_right")),
            "wrong_door_reward": float(environment.reward("tiger_left", "open_left")),
            # False for every Tiger: the episode runs on past an open, which is
            # why the belief visibly resets rather than the page stopping.
            "open_is_terminal": bool(environment.is_terminal("tiger_left")),
            "open_redraw_probabilities": _redraw_probabilities(environment),
        },
        "states": states,
        "next_states": next_states,
        "observations": observations,
        "beliefs": beliefs,
    }

    return EpisodeTrace(
        environment=str(environment.name),
        payload_kind=TIGER_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload=payload,
        policy=policy_name,
        reach_terminal_state=bool(environment.is_terminal(history[-1].state)),
        metadata={"environment_class": type(environment).__name__},
    )
