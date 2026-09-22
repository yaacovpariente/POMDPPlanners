# SPDX-License-Identifier: MIT

"""Racetrack episode trace exporter.

The GIF renderer beside this module draws an episode. This one writes the same
episode as data, so the browser viewer can replay it. Nothing here is
re-derived and nothing is asked of the live simulator: the episode's recorded
states are the whole source, which is the only way a forward-only world can be
drawn at all. :class:`RacetrackPOMDP` advances one true state one tick per
interaction and cannot be re-run from an injected state, so a viewer that
re-simulated anything would be showing a different episode.

Two things make this exporter different from the other environments' and both
follow from what this world is for.

**The track is shipped with the episode.** The circuit is highway-env's, and
this package already carries its exact lane parameters for the GIF; the payload
carries the same sampled edges so the browser draws the road the car actually
drove rather than the planner's re-integrated curvature map, which is a
different curve and is allowed to be wrong. It is shipped only for the scenario
those parameters describe, on the same test the GIF renderer applies.

**The reading is exported next to the truth.** This environment exists to make
partial observability measurable: the MDP and POMDP arms share one dynamics
path and differ only in what the sensor reports. So the payload carries both
the other vehicles' true positions and, on the POMDP arm, the detections the
ego actually received — which is what lets a viewer show the car the planner
could not see, rather than implying it saw everything.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps, to_jsonable
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_PRESENT,
    AGENT_SLOT_WIDTH,
    AGENT_REL_VX,
    AGENT_REL_VY,
    AGENT_REL_X,
    AGENT_REL_Y,
    DEFAULT_ENV_ID,
    DETECTION_PRESENT,
    DETECTION_REL_VX,
    DETECTION_REL_VY,
    DETECTION_REL_X,
    DETECTION_REL_Y,
    EGO_ANG,
    EGO_ARCLENGTH_M,
    EGO_HEADING,
    EGO_LAT,
    EGO_SPEED,
    EGO_STATE_WIDTH,
    EGO_X,
    EGO_Y,
    ObservationMode,
    state_agent_rows,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_visualizer_track import (
    reference_track_lanes,
)

# Payload version, independent of the envelope's. Bump it when the meaning of a
# payload field changes, so a viewer can refuse a file it would misdraw.
RACETRACK_PAYLOAD_KIND = "racetrack.v1"

# How finely the track's edges are written, in metres. The reference geometry
# is sampled at 0.25 m for the GIF, which is far more than a 3D scene needs and
# would multiply the trace's size for nothing: the tightest arc on this circuit
# has a 15 m radius, where a 1 m chord departs from the true edge by under a
# centimetre — a hundredth of the road's width.
TRACK_SAMPLE_SPACING_M = 1.0

# Vehicle footprint, in metres. These are highway-env 1.12.1's ``Vehicle``
# class constants, the same source this package's reference lane geometry came
# from. They are written into the payload rather than assumed by the viewer
# because a car drawn at the wrong size misreports every gap in the episode.
# The width agrees with the occlusion model's own note beside
# ``DEFAULT_BLOCKER_HALF_WIDTH_M``: a blocker is a 1 m-radius disc, "so a 2 m-
# wide car".
VEHICLE_LENGTH_M = 5.0
VEHICLE_WIDTH_M = 2.0

# The step channels that mean the episode ended rather than ran out of clock.
# A time limit is a truncation: the car was still driving when the recorder
# stopped, and calling that a terminal state would misreport the outcome on
# every page that counts them.
TERMINAL_CHANNELS = ("crashed", "off_road")


def _kept_indices(outer: np.ndarray, spacing_m: float) -> List[int]:
    """Which samples of a lane to keep, thinning to roughly ``spacing_m``.

    One index list is chosen for the whole lane, from its *outer* edge, and
    both edges are then thinned by it. Thinning each edge on its own arclength
    would keep a different number of points on the inside of a bend than on the
    outside, and the two edges would no longer pair up — which is exactly what
    a surface built between them needs them to do.

    Args:
        outer: The longer of the lane's two edges, as an ``(N, 2)`` array.
        spacing_m: Target spacing between kept points, in metres.

    Returns:
        Indices into the edge arrays, always including the first and the last
        so a thinned lane still starts and ends where the real one does and
        adjacent lanes still meet at their seams.
    """
    if outer.shape[0] < 3:
        return list(range(outer.shape[0]))
    steps = np.linalg.norm(np.diff(outer, axis=0), axis=1)
    travelled = np.concatenate(([0.0], np.cumsum(steps)))
    kept = [0]
    last = 0.0
    for index in range(1, outer.shape[0] - 1):
        if travelled[index] - last >= spacing_m:
            kept.append(index)
            last = travelled[index]
    kept.append(outer.shape[0] - 1)
    return kept


def _points(edge: np.ndarray, kept: Sequence[int]) -> List[List[float]]:
    """Take the kept samples of one edge as ``[x, y]`` pairs."""
    return [[float(edge[i, 0]), float(edge[i, 1])] for i in kept]


def track_payload(spacing_m: float = TRACK_SAMPLE_SPACING_M) -> List[Dict[str, Any]]:
    """The circuit's lanes, thinned for transport.

    Args:
        spacing_m: Target spacing between kept edge points, in metres.

    Returns:
        One entry per lane: its two edges as ``left`` and ``right``, sampled at
        the same points so the surface between them can be built pair by pair,
        and ``line_types``, the marking highway-env gives each edge.
    """
    lanes = []
    for lane in reference_track_lanes():
        edges = [np.asarray(line["points"], dtype=float) for line in lane["lines"]]
        longer = edges[0] if edges[0].shape[0] >= edges[1].shape[0] else edges[1]
        kept = _kept_indices(longer, spacing_m)
        lanes.append(
            {
                "left": _points(edges[0], kept),
                "right": _points(edges[1], kept),
                "line_types": [int(line["type"]) for line in lane["lines"]],
            }
        )
    return lanes


def _ego_row(state: np.ndarray) -> Dict[str, float]:
    """The ego's own slots, named rather than left as indices into a vector."""
    return {
        "x": float(state[EGO_X]),
        "y": float(state[EGO_Y]),
        "heading": float(state[EGO_HEADING]),
        "speed": float(state[EGO_SPEED]),
        "lane_offset": float(state[EGO_LAT]),
        "lane_heading": float(state[EGO_ANG]),
        "arclength": float(state[EGO_ARCLENGTH_M]),
    }


def _ego_velocity(state: np.ndarray) -> np.ndarray:
    """The ego's own velocity in world metres per second.

    highway-env's kinematic vehicle moves along its heading, so its velocity is
    its speed times its heading direction. Both sit in the ego's own state
    slots, which is what lets a recorded episode recover it without the
    simulator.

    Args:
        state: One recorded state vector.

    Returns:
        A ``(2,)`` array of world-frame velocity.
    """
    heading = float(state[EGO_HEADING])
    speed = float(state[EGO_SPEED])
    return np.array([speed * np.cos(heading), speed * np.sin(heading)], dtype=float)


def _world_velocities(state: np.ndarray, rows: np.ndarray, vx: int, vy: int) -> np.ndarray:
    """Body-frame *relative* velocities, returned as world-frame absolute ones.

    Both the state's agent slots and a detection's rows hold
    ``other.velocity - ego.velocity`` rotated into the ego body frame; see
    ``racetrack_world_sensors.relative_vehicles``. Rotating that back into
    world axes is only half the job — the ego's own velocity has to be added
    back, or what comes out is a closing rate wearing a world-frame coat.

    This is worth spelling out because getting it wrong does not look wrong. A
    relative velocity has a plausible magnitude and a plausible direction, so
    nothing in the data looks broken; but a viewer taking a vehicle's heading
    from it draws a car that spins on the spot whenever the two run at similar
    speeds — the difference of two nearly equal vectors is almost all noise —
    and draws it facing backwards whenever the ego is the faster of the two.

    Args:
        state: The ego's own state vector for this step.
        rows: An ``(N, W)`` block of body-frame rows.
        vx: Column holding the forward velocity component.
        vy: Column holding the lateral velocity component.

    Returns:
        An ``(N, 2)`` array of world-frame velocities in metres per second.
    """
    angle = float(state[EGO_HEADING])
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=float
    )
    return rows[:, [vx, vy]] @ rotation.T + _ego_velocity(state)


def _to_world(state: np.ndarray, rows: np.ndarray, x_index: int, y_index: int) -> np.ndarray:
    """Rotate body-frame offsets into world metres with this record's heading.

    The state holds the other vehicles relative to the ego, and a detection is
    reported the same way. A viewer would have to redo this rotation for every
    row of every step, with the ego heading it happened to have parsed; doing
    it once here means the two drawings of one episode — the GIF and the
    browser — place a car in the same place by construction.

    Args:
        state: The ego's own state vector for this step.
        rows: An ``(N, W)`` block of body-frame rows.
        x_index: Column holding the forward offset.
        y_index: Column holding the lateral offset.

    Returns:
        An ``(N, 2)`` array of world-frame metres.
    """
    angle = float(state[EGO_HEADING])
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=float
    )
    return rows[:, [x_index, y_index]] @ rotation.T + state[[EGO_X, EGO_Y]]


def _agents(state: np.ndarray, max_tracked_agents: int) -> List[Dict[str, Any]]:
    """The other vehicles this step, in world metres.

    Args:
        state: One recorded state vector.
        max_tracked_agents: Number of fixed agent slots in the state.

    Returns:
        One entry per *occupied* slot, with ``x``/``y`` in world metres and
        ``vx``/``vy`` in world metres per second — absolute, not relative to
        the ego. An empty slot produces no entry at all, because the state says
        there is no vehicle there; writing a row of zeros would park a car on
        the start line for the whole episode.
    """
    rows = np.asarray(state_agent_rows(state, max_tracked_agents), dtype=float)
    positions = _to_world(np.asarray(state, dtype=float), rows, AGENT_REL_X, AGENT_REL_Y)
    velocities = _world_velocities(np.asarray(state, dtype=float), rows, AGENT_REL_VX, AGENT_REL_VY)

    agents = []
    for index in range(rows.shape[0]):
        if rows[index, AGENT_PRESENT] <= 0:
            continue
        agents.append(
            {
                "slot": int(index),
                "x": float(positions[index, 0]),
                "y": float(positions[index, 1]),
                "vx": float(velocities[index, 0]),
                "vy": float(velocities[index, 1]),
            }
        )
    return agents


def _detections(state: np.ndarray, observation: Any) -> Optional[List[Dict[str, Any]]]:
    """What the ego's sensor reported this step, in world metres.

    Args:
        state: The ego's own state vector, which supplies the heading the
            body-frame reading is rotated by.
        observation: The recorded observation.

    Returns:
        One entry per occupied detection slot, or ``None`` when this step's
        observation is not a POMDP reading — the MDP arm reports a table of
        absolute rows and has no detections to draw, and the terminal
        bookkeeping step has no observation at all. ``None`` says "this step
        has no reading", which a viewer must not confuse with "the sensor saw
        nothing", and an empty list says exactly that second thing.
    """
    rows = getattr(observation, "detections", None)
    if rows is None:
        return None
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[0] == 0:
        return []
    positions = _to_world(np.asarray(state, dtype=float), rows, DETECTION_REL_X, DETECTION_REL_Y)
    velocities = _world_velocities(
        np.asarray(state, dtype=float), rows, DETECTION_REL_VX, DETECTION_REL_VY
    )

    detections = []
    for index in range(rows.shape[0]):
        if rows[index, DETECTION_PRESENT] <= 0:
            continue
        detections.append(
            {
                "x": float(positions[index, 0]),
                "y": float(positions[index, 1]),
                "vx": float(velocities[index, 0]),
                "vy": float(velocities[index, 1]),
            }
        )
    return detections


def _ended_in_terminal_state(history: Sequence[StepData]) -> bool:
    """Whether the episode ended in a terminal state, read from its own record.

    :meth:`RacetrackPOMDP.is_terminal` refuses any state but the live one, and
    by the time a trace is written the session has moved on or been closed, so
    the outcome is taken from the channels the episode already recorded.

    Args:
        history: The episode's step records, in order.

    Returns:
        ``True`` when any step reported a crash or an off-road excursion.
    """
    for step in history:
        info = getattr(step, "info", None) or {}
        if any(float(info.get(channel, 0.0)) > 0 for channel in TERMINAL_CHANNELS):
            return True
    return False


def build_racetrack_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one Racetrack episode.

    Args:
        environment: The Racetrack world the episode was run on. Its sensor
            range, observation mode, agent slots and action vocabulary are
            copied into the payload's ``world`` block so a viewer can say what
            the arm was — and so it reads the instance's values, not the
            defaults a configured run may have moved.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind ``racetrack.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    if not history:
        raise ValueError("Cannot export a trace for an empty history")

    slots = int(environment.max_tracked_agents)
    ego: List[Dict[str, float]] = []
    agents: List[List[Dict[str, Any]]] = []
    detections: List[Optional[List[Dict[str, Any]]]] = []
    beliefs: List[Dict[str, Any]] = []

    for step in history:
        state = np.asarray(step.state, dtype=float)
        ego.append(_ego_row(state))
        agents.append(_agents(state, slots))
        detections.append(_detections(state, step.observation))
        beliefs.append(belief_to_payload(step.belief))

    mode = environment.observation_mode
    payload: Dict[str, Any] = {
        "world": {
            # The circuit, at the exact lane parameters the GIF is drawn from —
            # and only when the episode was actually run on that circuit.
            # ``racetrack-v0``'s layout is fixed, so another scenario must not
            # inherit its road: a viewer drawing this track under a different
            # one would show a car cornering through scenery that was not
            # there. ``null`` means "no map", which the viewer reports rather
            # than fills in. The GIF renderer withholds it on the same test.
            "lanes": track_payload() if environment.env_id == DEFAULT_ENV_ID else None,
            "vehicle_length_m": VEHICLE_LENGTH_M,
            "vehicle_width_m": VEHICLE_WIDTH_M,
            # Which arm this episode was run under, and the dial that separates
            # the two. A viewer that did not say so would draw the MDP arm and
            # the POMDP arm identically.
            "observation_mode": mode.value if isinstance(mode, ObservationMode) else str(mode),
            "max_detection_range_m": float(environment.max_detection_range_m),
            "max_tracked_agents": slots,
            # Decisions per second. A viewer replaying this episode has no
            # other way to know how much time a step took: the envelope counts
            # steps, and a step here is a fifth of a second, while a step in a
            # grid world is "one decision" and has no duration at all. Without
            # it the car is replayed at whatever rate suits a board game, which
            # for this world is roughly a third of the speed it actually drove.
            "policy_frequency_hz": float(environment.simulator_config["policy_frequency"]),
            # The state's own layout. The belief's particles are whole state
            # vectors, so a viewer that wants to draw what the filter believes
            # about the *traffic* — which is this environment's hidden state —
            # has to index them, and guessing the widths would put a car in a
            # field.
            "ego_state_width": int(EGO_STATE_WIDTH),
            "agent_slot_width": int(AGENT_SLOT_WIDTH),
            # The action vocabulary, so a step's integer action can be shown as
            # the (acceleration, steering) command it actually was.
            "action_presets": [
                [float(acceleration), float(steering)]
                for acceleration, steering in environment.action_presets
            ],
        },
        "ego": ego,
        "agents": agents,
        "detections": detections,
        "beliefs": beliefs,
    }

    return EpisodeTrace(
        environment=str(environment.name),
        payload_kind=RACETRACK_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload=payload,
        policy=policy_name,
        reach_terminal_state=_ended_in_terminal_state(history),
        metadata={"environment_class": type(environment).__name__},
    )
