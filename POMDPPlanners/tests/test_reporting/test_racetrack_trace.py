# SPDX-License-Identifier: MIT

"""Tests for the Racetrack episode trace exporter.

None of these need the simulator. That is deliberate and it is also what the
exporter is for: a forward-only world cannot be re-run from a recorded state,
so the trace has to be buildable from the episode's own record alone, and a
test that needed a live session would be testing something the exporter never
does.

Two properties get most of the attention here, because both are ways the file
could look right and mean something false:

* the other vehicles and the detections must stay apart. This environment
  exists to measure what a planner loses when it cannot see a car, so a payload
  that merged the world's traffic with the sensor's returns would erase the
  measurement;
* the map must not travel further than the scenario it describes. ``racetrack-
  v0``'s layout is fixed, and a trace from another scenario carrying this
  circuit would put a car through scenery that was never there.
"""

from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.racetrack_pomdp import (
    RACETRACK_PAYLOAD_KIND,
    RacetrackPOMDP,
    build_racetrack_trace,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_SLOT_WIDTH,
    EGO_ANG,
    EGO_ARCLENGTH_M,
    EGO_HEADING,
    EGO_LAT,
    EGO_SPEED,
    EGO_STATE_WIDTH,
    EGO_X,
    EGO_Y,
    ObservationMode,
    RacetrackObservation,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_trace_exporter import (
    VEHICLE_LENGTH_M,
    VEHICLE_WIDTH_M,
    track_payload,
)

SLOTS = 2
STATE_WIDTH = EGO_STATE_WIDTH + SLOTS * AGENT_SLOT_WIDTH


@pytest.fixture(name="env")
def env_fixture():
    """A Racetrack world with two agent slots, built without the simulator."""
    return RacetrackPOMDP(discount_factor=0.95, max_tracked_agents=SLOTS)


def _state(x, y, heading, speed=8.0, agents=()):
    """One state vector: the ego's seven slots, then one block per agent.

    Args:
        x: Ego world x, in metres.
        y: Ego world y, in metres.
        heading: Ego heading, in radians.
        speed: Ego speed, in metres per second.
        agents: ``(slot, rel_x, rel_y, rel_vx, rel_vy)`` tuples to occupy.
    """
    state = np.zeros(STATE_WIDTH, dtype=float)
    state[EGO_X], state[EGO_Y] = x, y
    state[EGO_HEADING], state[EGO_SPEED] = heading, speed
    state[EGO_LAT], state[EGO_ANG], state[EGO_ARCLENGTH_M] = 0.4, 0.05, 12.0
    for slot, rel_x, rel_y, rel_vx, rel_vy in agents:
        base = EGO_STATE_WIDTH + slot * AGENT_SLOT_WIDTH
        state[base : base + AGENT_SLOT_WIDTH] = [1.0, rel_x, rel_y, rel_vx, rel_vy]
    return state


def _observation(detections):
    """A POMDP-arm reading whose detections are the given body-frame rows."""
    rows = np.zeros((SLOTS, 5), dtype=np.float32)
    for index, (rel_x, rel_y, rel_vx, rel_vy) in enumerate(detections):
        rows[index] = [1.0, rel_x, rel_y, rel_vx, rel_vy]
    return RacetrackObservation(
        ego_pose=np.zeros(4, dtype=np.float32),
        ego_speed=np.array([8.0], dtype=np.float32),
        lane_pose=np.zeros(2, dtype=np.float32),
        curvature_ahead=np.zeros(3, dtype=np.float32),
        detections=rows,
    )


def _belief(states):
    particles = [np.asarray(state, dtype=float) for state in states]
    weights = np.ones(len(particles)) / len(particles)
    return WeightedParticleBelief(particles=particles, log_weights=np.log(weights))


def _step(state, action, observation, reward, info=None):
    return StepData(
        state=state,
        action=action,
        next_state=None,
        observation=observation,
        reward=reward,
        belief=_belief([state, state + 0.1]),
        info=dict(info or {}),
    )


def _episode():
    """A short episode: two cars ahead, only the nearer one reported.

    The far car is inside the state and outside the reading, which is this
    environment's whole subject and the case the payload has to keep separable.
    """
    near, far = (0, 20.0, 1.5, 1.0, 0.0), (1, 60.0, -1.5, 1.0, 0.0)
    return [
        _step(
            _state(0.0, 0.0, 0.0, agents=(near, far)),
            3,
            _observation([(20.4, 1.2, 1.0, 0.0)]),
            0.8,
            {"speed_mps": 8.0},
        ),
        _step(
            _state(8.0, 0.0, 0.0, agents=(near, far)),
            4,
            _observation([(19.6, 1.7, 1.0, 0.0)]),
            0.8,
            {"speed_mps": 8.0},
        ),
        _step(_state(16.0, 0.0, 0.0, agents=(near,)), None, None, None, {"crashed": 1.0}),
    ]


def test_trace_round_trips_through_json(env, tmp_path: Path):
    """A trace written and read back is the same trace.

    Purpose: The file is the interface between Python and the browser viewer,
    so a field that does not survive the round trip is a field the viewer
    silently never sees.

    Given: A Racetrack episode.
    When: Its trace is written to disk and read back.
    Then: Envelope and payload match the original, field for field.

    Test type: unit
    """
    trace = build_racetrack_trace(env, _episode(), episode_index=4, policy_name="PFT_DPW")
    restored = EpisodeTrace.read(trace.write(tmp_path / "trace_4.json"))

    assert restored.payload_kind == RACETRACK_PAYLOAD_KIND
    assert restored.episode_index == 4
    assert restored.policy == "PFT_DPW"
    assert restored.payload == trace.payload
    assert [step.to_dict() for step in restored.steps] == [step.to_dict() for step in trace.steps]


def test_world_names_the_arm_and_the_dial(env):
    """The observation mode and the sensor range are written down.

    Purpose: The MDP and POMDP arms share every other number in this payload.
    If a trace did not say which arm produced it, the site would draw a
    fully-observed run and a partially-observed one identically, and the
    comparison the environment exists for would be invisible.

    Test type: unit
    """
    world = build_racetrack_trace(env, _episode(), 0).payload["world"]

    assert world["observation_mode"] == ObservationMode.POMDP.value
    assert world["max_detection_range_m"] == pytest.approx(env.max_detection_range_m)
    assert world["max_tracked_agents"] == SLOTS
    assert world["ego_state_width"] == EGO_STATE_WIDTH
    assert world["agent_slot_width"] == AGENT_SLOT_WIDTH
    assert world["action_presets"] == [list(preset) for preset in env.action_presets]


def test_agents_are_placed_in_world_metres(env):
    """A vehicle's body-frame slot becomes a world position.

    Purpose: The state holds the other vehicles relative to the ego. A viewer
    that drew those numbers directly would put every car near the origin, and
    one that rotated them with the wrong heading would put them in the
    infield. The rotation happens once, here, so the GIF and the browser agree
    by construction.

    Given: An ego at the origin facing +x, with a car 20 m ahead and 1.5 m left.
    When: The trace is built.
    Then: That car is at (20, 1.5) in world metres.

    Test type: unit
    """
    payload = build_racetrack_trace(env, _episode(), 0).payload

    first = payload["agents"][0]
    assert len(first) == 2
    assert (first[0]["x"], first[0]["y"]) == pytest.approx((20.0, 1.5))
    assert (first[1]["x"], first[1]["y"]) == pytest.approx((60.0, -1.5))


def test_a_rotated_ego_rotates_its_traffic(env):
    """The rotation uses the ego's own heading, not the world axes.

    Purpose: This is the failure that looks plausible: with the ego facing
    along +y, a car "20 m ahead" is at (0, 20), and an exporter that skipped
    the rotation would place it at (20, 0) — on the road, in front of nothing,
    and wrong by ninety degrees.

    Test type: unit
    """
    turned = [_step(_state(5.0, 5.0, np.pi / 2, agents=((0, 20.0, 0.0, 1.0, 0.0),)), 0, None, 0.1)]

    agents = build_racetrack_trace(env, turned, 0).payload["agents"][0]

    assert (agents[0]["x"], agents[0]["y"]) == pytest.approx((5.0, 25.0))


def test_agent_velocity_is_absolute_not_relative_to_the_ego(env):
    """A vehicle's exported velocity is its own, not its closing rate.

    Purpose: The state stores ``other.velocity - ego.velocity`` in the ego body
    frame. Rotating that into world axes and stopping there is the failure that
    does not look like one — a relative velocity has a plausible magnitude and
    direction, so the data looks fine, but a viewer reading a heading off it
    draws a car that spins on the spot whenever the two vehicles run at similar
    speeds and faces backwards whenever the ego is the faster.

    Given: An ego heading along +x at 8 m/s, and a car ahead whose recorded
        relative velocity is +1 m/s forward — so it is truly doing 9 m/s.
    When: The trace is built.
    Then: The exported velocity is (9, 0), not the relative (1, 0).

    Test type: unit
    """
    ahead = (0, 20.0, 0.0, 1.0, 0.0)
    episode = [_step(_state(0.0, 0.0, 0.0, speed=8.0, agents=(ahead,)), 0, None, 0.1)]

    agents = build_racetrack_trace(env, episode, 0).payload["agents"][0]

    assert (agents[0]["vx"], agents[0]["vy"]) == pytest.approx((9.0, 0.0))


def test_a_vehicle_matching_the_ego_is_not_left_with_a_zero_velocity(env):
    """Two cars at the same speed give a heading, not a spin.

    Purpose: This is the case the bug actually showed up in. When an opponent
    matches the ego's speed its *relative* velocity is zero, so a viewer taking
    the heading from a relative velocity gets ``atan2(0, 0)`` plus noise and
    spins the car on the spot for the whole lap. The absolute velocity is the
    ego's, which points where the car is really going.

    Test type: unit
    """
    matched = (0, 25.0, 0.0, 0.0, 0.0)
    episode = [_step(_state(0.0, 0.0, np.pi / 2, speed=8.0, agents=(matched,)), 0, None, 0.1)]

    agents = build_racetrack_trace(env, episode, 0).payload["agents"][0]

    assert (agents[0]["vx"], agents[0]["vy"]) == pytest.approx((0.0, 8.0), abs=1e-9)
    assert np.hypot(agents[0]["vx"], agents[0]["vy"]) == pytest.approx(8.0)


def test_detection_velocity_is_absolute_too(env):
    """The sensor's reported velocity gets the same correction as the truth.

    Purpose: Detections carry the same body-frame relative velocity as the
    state's slots do. Correcting one and not the other would draw the true
    vehicle and its own detection pointing in different directions, which reads
    as a sensor error the episode never had.

    Test type: unit
    """
    payload = build_racetrack_trace(env, _episode(), 0).payload

    # The first step's ego runs along +x at 8 m/s; the detection's relative
    # velocity is +1 m/s forward, so the reported vehicle is doing 9 m/s.
    assert (payload["detections"][0][0]["vx"], payload["detections"][0][0]["vy"]) == pytest.approx(
        (9.0, 0.0)
    )


def test_an_empty_agent_slot_produces_no_vehicle(env):
    """A slot the state says is empty is absent from the payload.

    Purpose: An unoccupied slot is a row of zeros. Writing it out would park a
    car at the ego's own position for the whole episode, which reads as a
    collision that never happened.

    Test type: unit
    """
    payload = build_racetrack_trace(env, _episode(), 0).payload

    assert len(payload["agents"][0]) == 2
    assert len(payload["agents"][-1]) == 1


def test_detections_stay_separate_from_the_truth(env):
    """The sensor's returns are exported beside the world's vehicles, not merged.

    Purpose: The whole measurement this environment supports is the difference
    between what was there and what was seen. Two cars are in the state and one
    is in the reading; a payload that reported two detections — or two vehicles
    because there were two detections — would erase that difference, and the
    viewer would show a planner that saw everything.

    Test type: unit
    """
    payload = build_racetrack_trace(env, _episode(), 0).payload

    assert len(payload["agents"][0]) == 2
    assert len(payload["detections"][0]) == 1
    # The reading is noisy, so it is near the true car and not equal to it.
    detected = payload["detections"][0][0]
    assert detected["x"] == pytest.approx(20.4)
    assert detected["x"] != pytest.approx(payload["agents"][0][0]["x"])


def test_a_step_with_no_reading_is_not_a_step_that_saw_nothing(env):
    """The terminal step's absent observation is ``None``, not an empty list.

    Purpose: These two mean opposite things — "this step has no reading at all"
    against "the sensor looked and reported nothing" — and a viewer that showed
    the second where the first happened would claim the road was clear at the
    moment of the crash.

    Test type: unit
    """
    episode = _episode()
    empty_reading = _step(
        _state(2.0, 0.0, 0.0, agents=((0, 30.0, 0.0, 1.0, 0.0),)), 1, _observation([]), 0.5
    )

    payload = build_racetrack_trace(env, [empty_reading] + episode, 0).payload

    assert payload["detections"][0] == []
    assert payload["detections"][-1] is None


def test_vehicle_footprint_matches_the_simulator(env):
    """The exported car size is highway-env's, not a remembered number.

    Purpose: ``VEHICLE_LENGTH_M`` and ``VEHICLE_WIDTH_M`` are written down in
    the exporter because the payload has to carry them and the state does not.
    Written down means they can drift from the simulator that actually places
    the cars, and a car drawn at the wrong size misreports every gap in the
    episode — including the one just before a crash.

    Test type: unit
    """
    highway_env = pytest.importorskip("highway_env")
    from highway_env.vehicle.kinematics import Vehicle  # noqa: PLC0415

    assert VEHICLE_LENGTH_M == pytest.approx(Vehicle.LENGTH)
    assert VEHICLE_WIDTH_M == pytest.approx(Vehicle.WIDTH)

    world = build_racetrack_trace(env, _episode(), 0).payload["world"]
    assert world["vehicle_length_m"] == pytest.approx(Vehicle.LENGTH)
    assert world["vehicle_width_m"] == pytest.approx(Vehicle.WIDTH)


def test_the_track_travels_only_with_its_own_scenario(env):
    """A different scenario gets no map rather than this circuit's.

    Purpose: ``racetrack-v0``'s lane parameters describe one layout. Shipping
    them with an episode run elsewhere would draw a car cornering through
    scenery that was not there, and the viewer has no way to tell. The GIF
    renderer withholds the map on the same test.

    Test type: unit
    """
    elsewhere = RacetrackPOMDP(
        discount_factor=0.95, max_tracked_agents=SLOTS, env_id="racetrack-v1"
    )

    assert build_racetrack_trace(env, _episode(), 0).payload["world"]["lanes"]
    assert build_racetrack_trace(elsewhere, _episode(), 0).payload["world"]["lanes"] is None


def test_the_track_edges_stay_paired_and_keep_their_ends(env):
    """Thinning a lane keeps both edges in step and keeps the seams.

    Purpose: The viewer builds the road surface pair by pair between a lane's
    two edges, so edges of different lengths have nothing to build between. And
    a dropped endpoint would open a gap at every seam where one lane meets the
    next.

    Test type: unit
    """
    lanes = track_payload()
    reference = track_payload(spacing_m=0.0)

    assert lanes
    for thinned, dense in zip(lanes, reference):
        assert len(thinned["left"]) == len(thinned["right"])
        assert len(thinned["left"]) >= 2
        assert thinned["left"][0] == pytest.approx(dense["left"][0])
        assert thinned["left"][-1] == pytest.approx(dense["left"][-1])
        assert thinned["right"][-1] == pytest.approx(dense["right"][-1])


def test_the_outcome_is_read_from_the_recorded_channels(env):
    """A crash makes the episode terminal; running out of clock does not.

    Purpose: ``is_terminal`` refuses any state but the live one, so the outcome
    has to come from what the episode recorded. A time limit is a truncation —
    the car was still driving — and filing it as a terminal state would move a
    number the site reports for the whole run.

    Test type: unit
    """
    timed_out = _episode()
    timed_out[-1] = _step(timed_out[-1].state, None, None, None, {"time_limit": 1.0})

    assert build_racetrack_trace(env, _episode(), 0).reach_terminal_state is True
    assert build_racetrack_trace(env, timed_out, 0).reach_terminal_state is False


def test_the_exporter_never_touches_the_live_world(env):
    """Building a trace asks the session nothing.

    Purpose: This world is forward-only. A query during export would either
    raise or, worse, advance the simulator past the episode being written.

    Test type: unit
    """
    env.is_terminal = lambda state: pytest.fail("is_terminal was queried during export")

    trace = build_racetrack_trace(env, _episode(), 0)

    assert trace.reach_terminal_state is True


def test_an_empty_history_is_refused(env):
    """There is no trace for an episode with no steps.

    Purpose: An empty payload would reach the viewer as a circuit with no car
    on it and no explanation, rather than as the recording failure it is.

    Test type: unit
    """
    with pytest.raises(ValueError, match="empty history"):
        build_racetrack_trace(env, [], 0)


def test_the_environment_hook_produces_the_same_trace(env):
    """``build_episode_trace`` is the exporter, not a second implementation.

    Purpose: The episode loop calls the hook and the tests above call the
    function. If those diverge, everything above is testing code nothing runs.

    Test type: unit
    """
    episode = _episode()

    hooked = env.build_episode_trace(episode, episode_index=7, policy_name="POMCPOW")
    direct = build_racetrack_trace(env, episode, episode_index=7, policy_name="POMCPOW")

    assert hooked.to_dict() == direct.to_dict()
