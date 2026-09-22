# SPDX-License-Identifier: MIT

"""LaserTag episode trace exporter.

The GIF renderers beside this module draw an episode. This one writes the same
episode as data, so the browser viewer can replay it. Nothing here is
re-derived: every number written comes from the recorded episode or from the
environment's own configuration.

The eight laser ranges are the whole observation in this environment, so they
are the one thing the payload cannot leave to the reader. A viewer that recast
its own rays would be a second, drifting copy of the observation model — and
the two copies have already disagreed once, because the discrete GIF's ray
walk ignores the opponent while the observation model stops at it. So this
exporter calls the environment's own range function and writes the eight
numbers out, and the viewer draws each beam ending exactly where that number
puts it.

Two range arrays are written per step, and the difference between them is the
environment:

* ``laser_ranges`` — the true ranges at the drawn state, from the model with
  no noise added.
* ``observed_ranges`` — the noisy reading the agent actually received at that
  same state. An observation is emitted for a step's ``next_state``, so the
  reading taken *at* state ``i`` is ``history[i - 1].observation``; that is the
  pairing written here, and the first state has no recorded reading and is
  written as ``null`` rather than paired with a reading from elsewhere.

The belief is not serialized here. It is a core abstraction with a closed
family of implementations, so
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, and this exporter is left with what is genuinely
LaserTag's: the arena, the two bodies and the ranges.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps, to_jsonable

# Payload version, independent of the envelope's. Bump it when the meaning of
# a payload field changes, so a viewer can refuse a file it would misdraw.
LASER_TAG_PAYLOAD_KIND = "laser_tag.v1"

DISCRETE_LASER_TAG_VARIANT = "discrete"
CONTINUOUS_LASER_TAG_VARIANT = "continuous"

# The eight beams, in the order both variants report them.
BEAM_LABELS: Tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

# The sentinel a terminal state emits instead of a measurement.
_TERMINAL_OBSERVATION = -1.0


def _positions(state: Any) -> Tuple[List[float], List[float], bool]:
    """Split a LaserTag state into robot, opponent and the terminal flag."""
    values = np.asarray(state, dtype=float).reshape(-1)
    return (
        [float(values[0]), float(values[1])],
        [float(values[2]), float(values[3])],
        bool(values[4]),
    )


def _observation_row(observation: Any) -> Optional[List[float]]:
    """A recorded observation as eight floats, or ``None`` if it is not one.

    A terminal step carries ``None``, and a terminal state emits the all
    ``-1`` sentinel rather than a measurement. Neither is a range, so neither
    is written as one.
    """
    if observation is None:
        return None
    values = np.asarray(observation, dtype=float).reshape(-1)
    if values.size != len(BEAM_LABELS):
        return None
    if bool(np.all(values == _TERMINAL_OBSERVATION)):
        return None
    return [float(v) for v in values]


def _paired_observations(history: Sequence[StepData]) -> List[Optional[List[float]]]:
    """Line each step's recorded reading up with the state it was taken at.

    ``sample_observation`` is called on a step's ``next_state``, so the reading
    that belongs beside state ``i`` was recorded on step ``i - 1``. The pairing
    is checked rather than assumed: if a recorded ``next_state`` does not match
    the following step's ``state`` — a stitched or filtered history — the
    reading is dropped instead of being attached to a state it was not taken
    from.
    """
    rows: List[Optional[List[float]]] = [None] * len(history)
    for index in range(1, len(history)):
        previous = history[index - 1]
        if previous.next_state is None:
            continue
        if not np.array_equal(
            np.asarray(previous.next_state, dtype=float).reshape(-1),
            np.asarray(history[index].state, dtype=float).reshape(-1),
        ):
            continue
        rows[index] = _observation_row(previous.observation)
    return rows


def _discrete_laser_directions() -> Sequence[Tuple[int, int]]:
    """The discrete environment's own ``(drow, dcol)`` table.

    Imported inside the function because the environment module imports this
    package at module scope: at module scope here the two would form an import
    cycle.
    """
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import _LASER_DIRECTIONS

    return _LASER_DIRECTIONS


def _discrete_scan(
    environment: Any, robot: List[float], opponent: List[float]
) -> Tuple[List[float], List[bool]]:
    """Discrete ranges, from the environment's own cell walk.

    ``hit_opponent`` is derived rather than re-implemented: the same walk is
    run a second time with the opponent placed off the grid, and a beam whose
    range grows when the opponent is removed is a beam the opponent stopped.
    """
    # pylint: disable-next=protected-access
    walk = environment._laser_distance_inline
    robot_cell = (int(robot[0]), int(robot[1]))
    opponent_cell = (int(opponent[0]), int(opponent[1]))
    # A cell no grid contains, so the second walk sees walls and edges only.
    nowhere = (-1, -1)

    ranges: List[float] = []
    hit: List[bool] = []
    for direction in _discrete_laser_directions():
        with_opponent = float(walk(robot_cell, direction, opponent_cell))
        without_opponent = float(walk(robot_cell, direction, nowhere))
        ranges.append(with_opponent)
        hit.append(with_opponent < without_opponent)
    return ranges, hit


def _continuous_scan(
    environment: Any, robot: List[float], opponent: List[float]
) -> Tuple[List[float], List[bool]]:
    """Continuous ranges, from the environment's own ray cast.

    Same trick as the discrete scan for ``hit_opponent``: cast once with the
    opponent where it is and once with it moved far outside the arena, and
    compare.
    """
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_geometry import (
        compute_laser_measurements,
    )

    robot_pos = np.asarray(robot, dtype=float)
    walls = np.asarray(environment.walls, dtype=float).reshape(-1, 4)
    grid = np.asarray(environment.grid_size, dtype=float).reshape(-1)
    far = grid * 1.0e6 + 1.0e6

    with_opponent = compute_laser_measurements(
        robot_pos, np.asarray(opponent, dtype=float), environment.opponent_radius, walls, grid
    )
    without_opponent = compute_laser_measurements(
        robot_pos, far, environment.opponent_radius, walls, grid
    )
    return (
        [float(v) for v in with_opponent],
        [bool(a < b) for a, b in zip(with_opponent, without_opponent)],
    )


def _shared_payload(environment: Any, variant: str) -> Dict[str, Any]:
    """The world block both variants fill in the same way."""
    return {
        "variant": variant,
        "beam_labels": list(BEAM_LABELS),
        "hazards": [[float(r), float(c)] for r, c in environment.dangerous_areas],
        "hazard_radius": float(environment.dangerous_area_radius),
        "hazard_penalty": float(environment.dangerous_area_penalty),
        "hazard_is_terminal": bool(environment.is_dangerous_area_hit_terminal),
        "tag_reward": float(environment.tag_reward),
        "tag_penalty": float(environment.tag_penalty),
        "step_cost": float(environment.step_cost),
        "measurement_noise": float(environment.measurement_noise),
    }


def _build_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    variant: str,
    world: Dict[str, Any],
    scan: Any,
    policy_name: Optional[str],
) -> EpisodeTrace:
    """Assemble one LaserTag episode's trace, whichever variant it came from."""
    if not history:
        raise ValueError("Cannot export a trace for an empty history")

    robots: List[List[float]] = []
    opponents: List[List[float]] = []
    terminals: List[bool] = []
    laser_ranges: List[List[float]] = []
    hit_opponent: List[List[bool]] = []
    beliefs: List[Dict[str, Any]] = []

    for step in history:
        robot, opponent, terminal = _positions(step.state)
        robots.append(robot)
        opponents.append(opponent)
        terminals.append(terminal)
        # A terminal state emits the sentinel, not a measurement, so it gets no
        # beams rather than beams of length zero, which would read as a robot
        # boxed in on all eight sides.
        if terminal:
            laser_ranges.append([])
            hit_opponent.append([])
        else:
            ranges, hit = scan(environment, robot, opponent)
            laser_ranges.append(ranges)
            hit_opponent.append(hit)
        beliefs.append(belief_to_payload(step.belief))

    payload: Dict[str, Any] = {
        "world": world,
        "robots": robots,
        "opponents": opponents,
        "terminals": terminals,
        "laser_ranges": laser_ranges,
        "observed_ranges": _paired_observations(history),
        "hit_opponent": hit_opponent,
        "beliefs": beliefs,
        "actions": [to_jsonable(step.action) for step in history],
    }

    return EpisodeTrace(
        environment=str(environment.name),
        payload_kind=LASER_TAG_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload=payload,
        policy=policy_name,
        reach_terminal_state=bool(environment.is_terminal(history[-1].state)),
        metadata={"environment_class": type(environment).__name__, "variant": variant},
    )


def build_laser_tag_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one discrete LaserTag episode.

    Args:
        environment: The :class:`LaserTagPOMDP` the episode was run on. Its
            grid, walls, hazards and reward constants are copied into the
            payload's ``world`` block so a viewer can build the arena without
            importing Python.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind
        ``laser_tag.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    world = _shared_payload(environment, DISCRETE_LASER_TAG_VARIANT)
    world.update(
        {
            # (rows, cols), as the environment stores it. A discrete position
            # is a cell index, so the arena spans 0..rows-1 by 0..cols-1 and
            # the drawn plate is one cell wider on each axis than that span.
            "floor_shape": [int(environment.floor_shape[0]), int(environment.floor_shape[1])],
            "walls": [[int(row), int(col)] for row, col in sorted(environment.walls)],
            # (drow, dcol) cell steps, the environment's own table. Written out
            # so the viewer never has to hold a second copy of it.
            "laser_directions": [[int(dr), int(dc)] for dr, dc in _discrete_laser_directions()],
            "transition_error_prob": float(environment.transition_error_prob),
            "opponent_policy": str(environment.opponent_policy.value),
        }
    )
    return _build_trace(
        environment=environment,
        history=history,
        episode_index=episode_index,
        variant=DISCRETE_LASER_TAG_VARIANT,
        world=world,
        scan=_discrete_scan,
        policy_name=policy_name,
    )


def build_continuous_laser_tag_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one continuous LaserTag episode.

    Args:
        environment: The :class:`ContinuousLaserTagPOMDP` the episode was run
            on, or its discrete-action wrapper.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind
        ``laser_tag.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_geometry import (
        LASER_DIRECTIONS,
    )

    grid = np.asarray(environment.grid_size, dtype=float).reshape(-1)
    world = _shared_payload(environment, CONTINUOUS_LASER_TAG_VARIANT)
    world.update(
        {
            "grid_size": [float(grid[0]), float(grid[1])],
            # (cx, cy, hx, hy) AABBs, as the environment stores them.
            "walls": [
                [float(v) for v in wall]
                for wall in np.asarray(environment.walls, dtype=float).reshape(-1, 4)
            ],
            # Unit (dx, dy) vectors, the environment's own table. It is not the
            # discrete table in another frame and must not be treated as one.
            "laser_directions": [[float(dx), float(dy)] for dx, dy in LASER_DIRECTIONS],
            "robot_radius": float(environment.robot_radius),
            "opponent_radius": float(environment.opponent_radius),
            "tag_radius": float(environment.tag_radius),
            "hazard_hit_probability": float(environment.dangerous_area_hit_probability),
            "opponent_policy": str(environment.opponent_policy.value),
        }
    )
    return _build_trace(
        environment=environment,
        history=history,
        episode_index=episode_index,
        variant=CONTINUOUS_LASER_TAG_VARIANT,
        world=world,
        scan=_continuous_scan,
        policy_name=policy_name,
    )
