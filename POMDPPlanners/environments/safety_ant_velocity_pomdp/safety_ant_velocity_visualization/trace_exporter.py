# SPDX-License-Identifier: MIT

"""Safety Ant Velocity episode trace exporter.

The GIF renderer beside this module draws an episode. This one writes the same
episode as data, so the browser viewer can replay it. Nothing here is
re-derived from a re-simulation: every number written comes from the recorded
episode or from the environment's own configuration.

The belief is not serialized here. It is a core abstraction with a closed
family of implementations, so
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, and this exporter is left with what is genuinely
this environment's: the safety constants, the four-dimensional states and the
observations.

One field is computed rather than copied. The environment samples the force
*direction* inside its C++ kernel and never records it, so a viewer that wants
to draw the force that was actually applied cannot read it anywhere. It is
recovered here by inverting the recorded transition, which is the same
inversion :meth:`SafeAntVelocityPOMDP._transition_probability` already relies
on:

    f = (v' - v) * mass / dt + damping * v

That is algebra on two recorded states, not a re-run of the physics — if the
states are the episode's, so is the force.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps, to_jsonable

# Payload version, independent of the envelope's. Bump it when the meaning of a
# payload field changes, so a viewer can refuse a file it would misdraw.
SAFETY_ANT_VELOCITY_PAYLOAD_KIND = "safety_ant_velocity.v1"

# The environment's terminal margin: ``is_terminal`` ends the episode above
# 1.5x the safe threshold. Written into the payload rather than hardcoded in
# the viewer, so the two cannot drift apart.
CRITICAL_MARGIN = 1.5


def _state_vector(value: Any) -> List[float]:
    """Coerce one recorded state or observation to ``[x, y, vx, vy]``."""
    array = np.asarray(value, dtype=float).reshape(-1)
    return [float(component) for component in array[:4]]


def _applied_force(
    state: List[float], next_state: Optional[List[float]], mass: float, dt: float, damping: float
) -> Optional[List[float]]:
    """Recover the force vector that carried ``state`` to ``next_state``.

    The environment integrates ``v' = v + ((f - damping * v) / mass) * dt``, so
    the force is determined by the pair of states. Returning it lets a viewer
    draw the direction that was drawn, instead of inventing one.

    Args:
        state: The recorded state before the step.
        next_state: The recorded state after it, or ``None`` on the terminal
            bookkeeping step, which has no transition to invert.
        mass: The environment's ``mass``.
        dt: The environment's integration step.
        damping: The environment's ``damping``.

    Returns:
        ``[fx, fy]``, or ``None`` when there is no transition.
    """
    if next_state is None:
        return None
    velocity = np.asarray(state[2:4], dtype=float)
    next_velocity = np.asarray(next_state[2:4], dtype=float)
    force = (next_velocity - velocity) * mass / dt + damping * velocity
    return [float(force[0]), float(force[1])]


def build_safety_ant_velocity_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one Safety Ant Velocity episode.

    Args:
        environment: The environment the episode was run on. Its safety
            constants are copied into the payload's ``world`` block so a viewer
            can build the scene without importing Python — and from the
            instance the episode actually ran on, not from the class defaults,
            which a configured run may not be using.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind
        ``safety_ant_velocity.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    if not history:
        raise ValueError("Cannot export a trace for an empty history")

    # Imported inside the function: the environment module imports this
    # package's GIF renderer, so a module-scope import back into it would be a
    # cycle. The force scales are a module constant there, not an instance
    # attribute, which is why they cannot simply be read off ``environment``.
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_pomdp import (
        DEFAULT_FORCE_SCALES,
    )

    mass = float(environment.mass)
    dt = float(environment.dt)
    damping = float(environment.damping)

    states: List[List[float]] = []
    next_states: List[Optional[List[float]]] = []
    observations: List[Any] = []
    applied_forces: List[Optional[List[float]]] = []
    beliefs: List[Dict[str, Any]] = []

    for step in history:
        state = _state_vector(step.state)
        next_state = None if step.next_state is None else _state_vector(step.next_state)
        states.append(state)
        next_states.append(next_state)
        # The observation is four-dimensional like the state, but it is written
        # through ``to_jsonable`` rather than coerced: an environment variant
        # that observed something else should reach the viewer as what it is.
        observations.append(to_jsonable(step.observation))
        applied_forces.append(_applied_force(state, next_state, mass, dt, damping))
        beliefs.append(belief_to_payload(step.belief))

    safe_threshold = float(environment.safe_velocity_threshold)

    payload: Dict[str, Any] = {
        "world": {
            "safe_velocity_threshold": safe_threshold,
            # The speed ``is_terminal`` ends the episode above. Written out
            # rather than left as a margin the viewer would have to apply.
            "critical_velocity_threshold": safe_threshold * CRITICAL_MARGIN,
            "critical_margin": CRITICAL_MARGIN,
            "max_force": float(environment.max_force),
            # One scale per discrete action, in action order, so a viewer can
            # name the force an action commands without a table of its own.
            "force_scales": [float(scale) for scale in DEFAULT_FORCE_SCALES],
            "actions": [to_jsonable(action) for action in environment.get_actions()],
            "dt": dt,
            "mass": mass,
            "damping": damping,
            "position_noise": float(environment.position_noise),
            "velocity_noise": float(environment.velocity_noise),
            "safety_violation_penalty": float(environment.safety_violation_penalty),
            "movement_reward_scale": float(environment.movement_reward_scale),
        },
        "states": states,
        "next_states": next_states,
        "observations": observations,
        "applied_forces": applied_forces,
        "beliefs": beliefs,
    }

    return EpisodeTrace(
        environment=str(environment.name),
        payload_kind=SAFETY_ANT_VELOCITY_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload=payload,
        policy=policy_name,
        reach_terminal_state=bool(environment.is_terminal(history[-1].state)),
        metadata={"environment_class": type(environment).__name__},
    )
