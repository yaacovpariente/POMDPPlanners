# SPDX-License-Identifier: MIT

"""Mountain Car episode trace exporter.

The GIF renderer beside this module draws an episode. This one writes the same
episode as data, so the browser viewer can replay it. Nothing here is
re-derived: the viewer replays the states this episode actually visited, and
never re-integrates the car. A viewer that re-ran the physics would be showing
its own rollout, which is exactly the thing a recorded episode exists to rule
out.

The belief is not serialized here either. It is a core abstraction with a
closed family of implementations, so
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, and this exporter is left with what is genuinely
Mountain Car's: the valley, the two-dimensional states and the noisy readings.

The hill's shape is written into the payload rather than left for the viewer to
assume. It is the one piece of this world that is not a constructor argument —
:class:`~POMDPPlanners.environments.mountain_car_pomdp.mountain_car_visualizer.MountainCarVisualizer`
carries it as a literal expression — and a viewer that guessed it would draw the
car floating above, or buried under, the slope it actually drove.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps, to_jsonable

# Payload version, independent of the envelope's. Bump it when the meaning of a
# payload field changes, so a viewer can refuse a file it would misdraw.
MOUNTAIN_CAR_PAYLOAD_KIND = "mountain_car.v1"

# Number of state components: [position, velocity]. Named because the viewer
# indexes the payload by position and a silently shorter row would put the car
# somewhere the episode never went.
STATE_DIMENSION = 2

# The valley, as ``height = amplitude * sin(frequency * position) + offset``.
# These are the GIF renderer's own numbers, restated here so the payload can
# carry them; a test pins them against that renderer, because two drawings of
# one episode disagreeing about where the ground is would be worse than either.
HILL_AMPLITUDE = 0.45
HILL_FREQUENCY = 3.0
HILL_OFFSET = 0.55


def hill_height(position: float) -> float:
    """Height of the valley floor under ``position``, in the GIF's own units.

    Args:
        position: A car position along the valley.

    Returns:
        The height of the ground there.
    """
    return HILL_AMPLITUDE * np.sin(HILL_FREQUENCY * position) + HILL_OFFSET


def _state_row(value: Any) -> List[float]:
    """Coerce one recorded state or observation to a two-float row.

    Args:
        value: A recorded state or observation.

    Returns:
        The position and velocity as plain floats.

    Raises:
        ValueError: If the value does not hold exactly two components. The
            viewer reads these by index, so a wrong-length row would place the
            car at a position nothing in the episode ever held.
    """
    row = np.asarray(value, dtype=float).reshape(-1)
    if row.size != STATE_DIMENSION:
        raise ValueError(
            f"Mountain Car trace rows need {STATE_DIMENSION} components, got {row.size}"
        )
    return [float(component) for component in row]


def build_mountain_car_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one Mountain Car episode.

    Args:
        environment: The Mountain Car environment the episode was run on. Its
            limits, engine power and both noise covariances are copied into the
            payload's ``world`` block so a viewer can build the valley without
            importing Python — and so it reads the instance's values, not the
            class defaults a configured run may have moved.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind
        ``mountain_car.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    if not history:
        raise ValueError("Cannot export a trace for an empty history")

    states: List[List[float]] = []
    next_states: List[Optional[List[float]]] = []
    observations: List[Optional[List[float]]] = []
    beliefs: List[Dict[str, Any]] = []

    for step in history:
        states.append(_state_row(step.state))
        next_states.append(None if step.next_state is None else _state_row(step.next_state))
        # The observation is the whole state plus noise, so it is written in
        # the same shape — but it is genuinely absent on the terminal
        # bookkeeping step, and None there says so rather than repeating the
        # previous reading.
        observations.append(None if step.observation is None else _state_row(step.observation))
        beliefs.append(belief_to_payload(step.belief))

    payload: Dict[str, Any] = {
        # The valley, taken from the environment instance the episode ran on.
        "world": {
            "min_position": float(environment.min_position),
            "max_position": float(environment.max_position),
            "max_speed": float(environment.max_speed),
            "goal_position": float(environment.goal_position),
            # The engine, and the slope that beats it. Their ratio is the whole
            # problem: a viewer can say why the car cannot simply drive up.
            "power": float(environment.power),
            "gravity": float(environment.gravity),
            "actions": [int(action) for action in environment.actions],
            "hill": {
                "amplitude": HILL_AMPLITUDE,
                "frequency": HILL_FREQUENCY,
                "offset": HILL_OFFSET,
            },
            # Both covariances, so a reader can state how noisy the episode's
            # observations were rather than assume the defaults.
            "observation_cov": to_jsonable(np.asarray(environment.cov_matrix, dtype=float)),
            "state_transition_cov": to_jsonable(
                np.asarray(environment.state_transition_cov, dtype=float)
            ),
        },
        "states": states,
        "next_states": next_states,
        "observations": observations,
        "beliefs": beliefs,
    }

    return EpisodeTrace(
        environment=str(environment.name),
        payload_kind=MOUNTAIN_CAR_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload=payload,
        policy=policy_name,
        reach_terminal_state=bool(environment.is_terminal(history[-1].state)),
        metadata={"environment_class": type(environment).__name__},
    )
