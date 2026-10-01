# SPDX-License-Identifier: MIT

"""Light-Dark episode visualizer.

Writes an episode as a trace, which ``light_dark.scene.js`` beside this module
replays in the browser. Nothing here is re-derived: every number written comes from the recorded episode or from the
environment's own configuration.

The belief is not serialized here. It is a core abstraction with a closed
family of implementations, so
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, and this visualizer is left with what is genuinely
Light-Dark's: the world's geometry, the rover's states and the observations.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.episode_visualizers import TraceVisualizer
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import to_jsonable

# Payload version, independent of the envelope's. Bump it when the meaning of
# a payload field changes, so a viewer can refuse a file it would misdraw.
LIGHT_DARK_PAYLOAD_KIND = "light_dark.v1"


class LightDarkVisualizer(TraceVisualizer):
    """Writes Light-Dark episodes as ``light_dark.v1`` traces."""

    payload_kind = LIGHT_DARK_PAYLOAD_KIND

    def build_payload(self, history: List[StepData]) -> Dict[str, Any]:
        """Build the Light-Dark half of the trace.

        The environment's geometry and reward constants are copied into the
        payload's ``world`` block so a viewer can build the scene without
        importing Python.

        Args:
            history: The episode's ``StepData`` records, in order.

        Returns:
            The ``light_dark.v1`` payload.
        """
        # Typed as Any: each field below is specific to this environment class.
        environment: Any = self.environment

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
                "goal_state": [
                    float(v) for v in np.asarray(environment.goal_state).reshape(-1)[:2]
                ],
                "start_state": [
                    float(v) for v in np.asarray(environment.start_state).reshape(-1)[:2]
                ],
                "goal_reward": float(environment.goal_reward),
                "obstacle_reward": float(environment.obstacle_reward),
                "obstacle_hit_probability": float(environment.obstacle_hit_probability),
                "fuel_cost": float(environment.fuel_cost),
            },
            "ended_in_danger_zone": self._ended_in_danger_zone(history),
            "states": states,
            "next_states": next_states,
            "observations": observations,
            "beliefs": beliefs,
        }

        return payload

    def _ended_in_danger_zone(self, history: List[StepData]) -> bool:
        """Whether the episode ended because the rover was hit in a hazard.

        With the hazard-terminal flag on, the transition appends a terminal slot
        that only a hazard hit sets; reaching the goal leaves it at 0. With the
        flag off, the discrete world still ends on an obstacle cell, so standing
        exactly on one at the end, away from the goal, is a hazard ending too.

        Args:
            history: The episode's ``StepData`` records, in order.

        Returns:
            ``True`` only for an episode a hazard ended.
        """
        if not self.reached_terminal_state(history):
            return False
        environment: Any = self.environment
        final = np.asarray(history[-1].state, dtype=float).reshape(-1)
        if environment.is_obstacle_hit_terminal:
            return final.shape[0] > 2 and float(final[2]) > 0.5
        obstacles = np.asarray(environment.obstacles, dtype=float)
        if not obstacles.size:
            return False
        on_obstacle = bool(np.any(np.all(obstacles.T == final[:2], axis=1)))
        at_goal = bool(
            np.all(np.asarray(environment.goal_state, dtype=float).reshape(-1)[:2] == final[:2])
        )
        return on_obstacle and not at_goal
