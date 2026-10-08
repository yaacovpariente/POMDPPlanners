# SPDX-License-Identifier: MIT

"""Safety Ant Velocity episode visualizer.

Writes an episode as a trace, which ``safety_ant_velocity.scene.js`` beside
this module replays in the browser. Nothing here is re-derived from a re-simulation: every number written comes from the recorded
episode or from the environment's own configuration.

The belief is not serialized here. It is a core abstraction with a closed
family of implementations, so
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, and this visualizer is left with what is genuinely
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
from POMDPPlanners.core.simulation.episode_visualizers import TraceVisualizer
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import to_jsonable

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


class SafeAntVelocityVisualizer(TraceVisualizer):
    """Writes Safety Ant Velocity episodes as ``safety_ant_velocity.v1`` traces."""

    payload_kind = SAFETY_ANT_VELOCITY_PAYLOAD_KIND

    def build_payload(self, history: List[StepData]) -> Dict[str, Any]:
        """Build the Safety Ant Velocity half of the trace.

        The environment's safety constants are copied into the payload's
        ``world`` block so a viewer can build the scene without importing
        Python — and from the instance the episode actually ran on, not from
        the class defaults, which a configured run may not be using.

        Args:
            history: The episode's ``StepData`` records, in order.

        Returns:
            The ``safety_ant_velocity.v1`` payload.
        """
        # Typed as Any: each field below is specific to this environment class.
        environment: Any = self.environment

        # Imported here so this module does not load the environment module at
        # import time, mirroring the environment's lazy import of this one. The
        # force scales are a module constant there, not an instance
        # attribute, which is why they cannot simply be read off
        # ``environment``.
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

        return payload
