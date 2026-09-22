# SPDX-License-Identifier: MIT

"""Push episode trace exporter, for both variants.

The Pillow renderers beside this module draw an episode. This one writes the
same episode as data, so the browser viewer can replay it. Nothing here is
re-derived: every number written comes from the recorded episode or from the
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
it for every environment, and this exporter is left with what is genuinely
Push's: the world's geometry, the robot and object positions, and the
observations.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps, to_jsonable

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
    its class, because importing either Push class here would close a cycle:
    both of them import this package to render their GIF.
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


def build_push_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one Push episode, discrete or continuous.

    Args:
        environment: The Push environment the episode was run on. Its geometry
            and reward constants are copied into the payload's ``world`` block
            so a viewer can build the scene without importing Python.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind ``push.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    if not history:
        raise ValueError("Cannot export a trace for an empty history")

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
    # never moves, so it is read off the first recorded state rather than from
    # ``environment.target_pos``, which a fixed initial state may not agree with.
    payload: Dict[str, Any] = {
        "world": _world(environment, _target(history[0].state)),
        "states": states,
        "next_states": next_states,
        "observations": observations,
        "beliefs": beliefs,
    }

    return EpisodeTrace(
        environment=str(environment.name),
        payload_kind=PUSH_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload=payload,
        policy=policy_name,
        reach_terminal_state=bool(environment.is_terminal(history[-1].state)),
        metadata={"environment_class": type(environment).__name__},
    )
