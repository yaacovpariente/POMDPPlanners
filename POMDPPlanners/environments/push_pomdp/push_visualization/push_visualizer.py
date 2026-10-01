# SPDX-License-Identifier: MIT

"""Push episode visualizer, for both variants.

Writes an episode as a trace, which ``push.scene.js`` beside this module
replays in the browser. Nothing here is re-derived: every number written comes from the recorded episode or from the
environment instance the episode ran on.

One payload kind covers the discrete and the continuous Push worlds. They are
the same task with the same six-number state and the same reward, and they
differ in exactly two things a viewer has to draw differently — the shape of an
obstacle (a circle of ``obstacle_radius`` against an axis-aligned square) and
the shape of an action (one of four labels against a 2-D vector). Both are
named in the payload, so the viewer branches on what the world says it is
rather than on two payload kinds that would share every other field.

The belief is not serialized here. It is a core abstraction with a closed
family of implementations, so
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, and this visualizer is left with what is genuinely
Push's: the world's geometry, the robot and object positions, and the
observations.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.episode_visualizers import TraceVisualizer
from POMDPPlanners.core.simulation.traces import to_jsonable

# Payload version, independent of the envelope's. Bump it when the meaning of a
# payload field changes, so a viewer can refuse a file it would misdraw.
PUSH_PAYLOAD_KIND = "push.v1"

# Both variants end the episode when the object is within 0.5 of the target,
# and both spend +100 at that moment. Neither number is an attribute of the
# environment — ``is_terminal`` and the reward models hardcode them — so they
# are written from here and are the one pair of constants in this file.
GOAL_TOLERANCE = 0.5
GOAL_REWARD = 100.0


def _positions(state: Any) -> List[float]:
    """The robot and object positions of one state, as four floats.

    Args:
        state: A Push state, six numbers, or seven when the continuous variant
            carries its hazard-terminal slot.

    Returns:
        ``[robot_x, robot_y, object_x, object_y]``.
    """
    values = np.asarray(state, dtype=float).reshape(-1)
    return [float(values[0]), float(values[1]), float(values[2]), float(values[3])]


def _target(state: Any) -> List[float]:
    values = np.asarray(state, dtype=float).reshape(-1)
    return [float(values[4]), float(values[5])]


def _continuous_world(environment: Any) -> Dict[str, Any]:
    """The geometry a continuous Push viewer needs.

    Obstacles are held as an ``(N, 4)`` array of ``(cx, cy, half_x, half_y)``
    corners; they are written in that form, so a viewer draws the box the
    environment actually collides against rather than one derived from the
    constructor's ``(cx, cy, half_size)`` tuples.
    """
    obstacles = np.asarray(environment.obstacles, dtype=float).reshape(-1, 4)
    return {
        "variant": "continuous",
        "obstacle_shape": "square",
        "obstacles": obstacles.tolist(),
        "obstacle_radius": None,
        "robot_radius": float(environment.robot_radius),
        "max_push": float(environment.max_push),
        # A hazard can end the episode in this variant, which changes what the
        # last recorded state means.
        "obstacle_hit_terminal": bool(environment.is_obstacle_hit_terminal),
        "dangerous_area_hit_terminal": bool(environment.is_dangerous_area_hit_terminal),
    }


def _discrete_world(environment: Any) -> Dict[str, Any]:
    """The geometry a discrete Push viewer needs."""
    obstacles = [[float(x), float(y)] for x, y in environment.obstacles]
    return {
        "variant": "discrete",
        "obstacle_shape": "circle",
        "obstacles": obstacles,
        "obstacle_radius": float(environment.obstacle_radius),
        "robot_radius": None,
        "max_push": None,
        "obstacle_hit_terminal": False,
        "dangerous_area_hit_terminal": False,
    }


def _world(environment: Any, target: List[float]) -> Dict[str, Any]:
    """Build the world block from the environment the episode ran on.

    The variant is decided by the attributes the environment has rather than by
    its class, so this module never has to import either Push class.
    """
    variant = (
        _continuous_world(environment)
        if hasattr(environment, "robot_radius")
        else _discrete_world(environment)
    )
    # The four action labels, with the vector each one means, when the
    # environment has them. The crate travels along the action, so a viewer
    # that drew "up" as its own idea of up could disagree with the transition.
    action_vectors = {
        label: [float(v) for v in np.asarray(vector, dtype=float).reshape(-1)]
        for label, vector in getattr(environment, "action_to_vector", {}).items()
    }
    world: Dict[str, Any] = {
        "grid_size": int(environment.grid_size),
        "target": target,
        "goal_tolerance": GOAL_TOLERANCE,
        "goal_reward": GOAL_REWARD,
        "push_threshold": float(environment.push_threshold),
        "friction_coefficient": float(environment.friction_coefficient),
        "observation_noise": float(environment.observation_noise),
        "obstacle_penalty": float(environment.obstacle_penalty),
        "dangerous_areas": [[float(x), float(y)] for x, y in environment.dangerous_areas],
        "dangerous_area_radius": float(environment.dangerous_area_radius),
        "dangerous_area_penalty": float(environment.dangerous_area_penalty),
        "action_vectors": action_vectors,
    }
    world.update(variant)
    return world


class PushVisualizer(TraceVisualizer):
    """Writes Push episodes, discrete or continuous, as ``push.v1`` traces.

    The environment's geometry and reward constants are copied into the
    payload's ``world`` block so a viewer can build the scene without
    importing Python.
    """

    payload_kind = PUSH_PAYLOAD_KIND

    def build_payload(self, history: List[StepData]) -> Dict[str, Any]:
        """Build the Push half of the trace.

        Args:
            history: The episode's ``StepData`` records, in order.

        Returns:
            The ``push.v1`` payload.
        """
        # Typed as Any: the attributes read below belong to this environment
        # class, not to the base Environment the visualizer is typed against.
        environment: Any = self.environment

        states: List[List[float]] = []
        next_states: List[Optional[List[float]]] = []
        observations: List[Any] = []
        beliefs: List[Dict[str, Any]] = []

        for step in history:
            states.append(_positions(step.state))
            next_states.append(None if step.next_state is None else _positions(step.next_state))
            observations.append(to_jsonable(step.observation))
            beliefs.append(belief_to_payload(step.belief))

        # The target is part of every state rather than a separate field, and it
        # never moves, so it is read off the first recorded state rather than
        # from ``environment.target_pos``, which a fixed initial state may not
        # agree with.
        return {
            "world": _world(environment, _target(history[0].state)),
            "ended_in_danger_zone": self._ended_in_danger_zone(history),
            "states": states,
            "next_states": next_states,
            "observations": observations,
            "beliefs": beliefs,
        }

    def _ended_in_danger_zone(self, history: List[StepData]) -> bool:
        """Whether the episode ended because the rover was hit in a hazard.

        Only the continuous world can end on a hazard. Its terminal slot is
        shared by a dangerous-area hit and an obstacle hit, so the slot alone
        does not say which; the rover finishing inside a dangerous area does.
        Reaching the target ends the episode without setting the slot.

        Args:
            history: The episode's ``StepData`` records, in order.

        Returns:
            ``True`` only for an episode a hazard ended.
        """
        environment: Any = self.environment
        if not getattr(environment, "is_dangerous_area_hit_terminal", False):
            return False
        final = np.asarray(history[-1].state, dtype=float).reshape(-1)
        if final.shape[0] <= 6 or float(final[6]) <= 0.5:
            return False
        # pylint: disable-next=protected-access
        return bool(environment._is_robot_in_dangerous_area(final[:2]))
