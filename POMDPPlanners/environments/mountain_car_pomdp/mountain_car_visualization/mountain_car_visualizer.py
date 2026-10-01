# SPDX-License-Identifier: MIT

"""Mountain Car episode visualizer.

Writes an episode as a trace, which ``mountain_car.scene.js`` beside this
module replays in the browser. Nothing here is re-derived: the viewer replays
the states this episode actually visited, and never re-integrates the car. A
viewer that re-ran the physics would be showing its own rollout, which is
exactly the thing a recorded episode exists to rule out.

The belief is not serialized here either. It is a core abstraction with a
closed family of implementations, so
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, and this visualizer is left with what is genuinely
Mountain Car's: the valley, the two-dimensional states and the noisy readings.

The hill's shape is written into the payload rather than left for the viewer to
assume. It is the one piece of this world that is not a constructor argument,
and a viewer that guessed it would draw the car floating above, or buried
under, the slope it actually drove.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.episode_visualizers import TraceVisualizer
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import to_jsonable

# Payload version, independent of the envelope's. Bump it when the meaning of a
# payload field changes, so a viewer can refuse a file it would misdraw.
MOUNTAIN_CAR_PAYLOAD_KIND = "mountain_car.v1"

# Number of state components: [position, velocity]. Named because the viewer
# indexes the payload by position and a silently shorter row would put the car
# somewhere the episode never went.
STATE_DIMENSION = 2

# The valley, as ``height = amplitude * sin(frequency * position) + offset``.
# They are not constructor arguments, so they are written into the payload for
# the viewer to read rather than to assume.
HILL_AMPLITUDE = 0.45
HILL_FREQUENCY = 3.0
HILL_OFFSET = 0.55


def hill_height(position: float) -> float:
    """Height of the valley floor under ``position``, in the drawing's units.

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


class MountainCarVisualizer(TraceVisualizer):
    """Writes Mountain Car episodes as ``mountain_car.v1`` traces."""

    payload_kind = MOUNTAIN_CAR_PAYLOAD_KIND

    def build_payload(self, history: List[StepData]) -> Dict[str, Any]:
        """Build the Mountain Car half of the trace.

        The environment's limits, engine power and both noise covariances are
        copied into the payload's ``world`` block so a viewer can build the
        valley without importing Python — and so it reads the instance's
        values, not the class defaults a configured run may have moved.

        Args:
            history: The episode's ``StepData`` records, in order.

        Returns:
            The ``mountain_car.v1`` payload.
        """
        # Typed as Any: each field below is specific to this environment class.
        environment: Any = self.environment

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

        return payload
