# SPDX-License-Identifier: MIT

"""CartPole episode trace exporter.

The GIF renderer beside this module draws an episode. This one writes the same
episode as data, so the browser viewer can replay it. Nothing here is
re-derived: the viewer replays the states this episode actually visited, and
never re-integrates the plant. A viewer that re-ran the physics would be
showing its own rollout, which is exactly the thing a recorded episode exists
to rule out.

The belief is not serialized here either. It is a core abstraction with a
closed family of implementations, so
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, and this exporter is left with what is genuinely
CartPole's: the rig's constants, the four-dimensional states and the noisy
observations.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps, to_jsonable

# Payload version, independent of the envelope's. Bump it when the meaning of a
# payload field changes, so a viewer can refuse a file it would misdraw.
CARTPOLE_PAYLOAD_KIND = "cartpole.v1"

# Number of state components: [cart_position, cart_velocity, pole_angle,
# pole_angular_velocity]. Named because the viewer indexes the payload by
# position and a silently shorter row would draw a pole at the wrong angle.
STATE_DIMENSION = 4


def _state_row(value: Any) -> List[float]:
    """Coerce one recorded state or observation to a four-float row.

    Args:
        value: A recorded state or observation.

    Returns:
        The four components as plain floats.

    Raises:
        ValueError: If the value does not hold exactly four components. The
            viewer reads these by index, so a wrong-length row would be drawn
            as a pole at an angle nothing in the episode ever held.
    """
    row = np.asarray(value, dtype=float).reshape(-1)
    if row.size != STATE_DIMENSION:
        raise ValueError(f"CartPole trace rows need {STATE_DIMENSION} components, got {row.size}")
    return [float(component) for component in row]


def build_cartpole_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one CartPole episode.

    Args:
        environment: The CartPole environment the episode was run on. Its
            physical constants and both termination thresholds are copied into
            the payload's ``world`` block so a viewer can build the rig without
            importing Python — and so it reads the instance's values, not the
            class defaults a configured run may have moved.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind ``cartpole.v1``.

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
        # The rig, taken from the environment instance the episode ran on.
        "world": {
            "gravity": float(environment.gravity),
            "masscart": float(environment.masscart),
            "masspole": float(environment.masspole),
            # Half the pole's length, as CartPolePOMDP defines it. The drawn
            # pole is twice this, which is the trap the field name sets.
            "length": float(environment.length),
            "force_mag": float(environment.force_mag),
            "tau": float(environment.tau),
            "kinematics_integrator": str(environment.kinematics_integrator),
            "x_threshold": float(environment.x_threshold),
            "theta_threshold_radians": float(environment.theta_threshold_radians),
            # Both covariances, so a reader can state how noisy the episode's
            # observations were rather than assume the test value.
            "noise_cov": to_jsonable(np.asarray(environment.noise_cov, dtype=float)),
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
        payload_kind=CARTPOLE_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload=payload,
        policy=policy_name,
        reach_terminal_state=bool(environment.is_terminal(history[-1].state)),
        metadata={"environment_class": type(environment).__name__},
    )
