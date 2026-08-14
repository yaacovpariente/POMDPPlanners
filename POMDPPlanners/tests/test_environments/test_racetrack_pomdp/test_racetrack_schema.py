# SPDX-License-Identifier: MIT

"""Unit tests for the shared racetrack schema: config, reward, sensors and state helpers.

Nothing here touches ``highway_env``. The schema module is deliberately backend-free so
the planner's model and the belief can be built where the simulator is not installed, and
these tests hold that line by importing only NumPy alongside the module under test.

The centre of the suite is still the *matched pair*, now asserted against the **dynamics**
configuration rather than against the observation key: the POMDP arm's reading is no longer
something highway-env produces, so "the two configs differ in the observation key alone" no
longer states the guarantee that matters. The rest covers the sensor geometry the
redesigned observation is built from -- the range gate and occlusion rule in
:func:`detection_visibility`, which is the arm's *only* hidden state -- and the flat layout
the torch model indexes, which now leads with the observed ego pose.

:func:`radial_velocities` is tested here too, but no longer as an observation channel: a
detection carries both components of relative velocity, so the closing rate is a derived
geometric quantity a gap-acceptance rule may want and nothing on the sensing path projects
it out. Its tests are tests of the projection.
"""

# pylint: disable=too-many-lines  # One test module per source module, as the test layout requires.

import copy
import json
import math
from typing import Any, Dict

import numpy as np
import pytest

from POMDPPlanners.environments.racetrack_pomdp.racetrack_detection import (
    validate_detection_rates,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    ACCELERATION_PRESETS,
    AGENT_SLOT_WIDTH,
    DEFAULT_ACTION_PRESETS,
    DEFAULT_ACTION_REWARD,
    DEFAULT_COLLISION_REWARD,
    DEFAULT_DETECTION_POSITION_STD_M,
    DEFAULT_DETECTION_VELOCITY_STD,
    DEFAULT_EGO_ARCLENGTH_STD_M,
    DEFAULT_EGO_HEADING_STD_RAD,
    DEFAULT_EGO_POSITION_STD_M,
    DEFAULT_MAX_DETECTION_RANGE_M,
    DEFAULT_MAX_TRACKED_AGENTS,
    DEFAULT_PRESENCE_FALSE_ALARM_PROB,
    DEFAULT_PRESENCE_MISS_PROB,
    DETECTION_PRESENT,
    DETECTION_REL_VX,
    DETECTION_REL_VY,
    DETECTION_REL_X,
    DETECTION_REL_Y,
    DETECTION_SLOT_WIDTH,
    EGO_POSE_ARCLENGTH,
    EGO_POSE_HEADING,
    EGO_POSE_X,
    EGO_POSE_Y,
    EGO_STATE_WIDTH,
    OBSERVED_EGO_POSE_WIDTH,
    OBSERVED_EGO_SPEED_WIDTH,
    OBSERVED_LANE_POSE_WIDTH,
    POMDP_OBS_CURVATURE_INDEX,
    POMDP_OBS_EGO_POSE_INDEX,
    POMDP_OBS_EGO_SPEED_INDEX,
    POMDP_OBS_LANE_POSE_INDEX,
    STEERING_PRESETS,
    ObservationMode,
    RacetrackObservation,
    build_racetrack_config,
    detection_visibility,
    ego_speed_from_kinematics_row,
    observation_config,
    pomdp_observation_width,
    racetrack_reward,
    radial_velocities,
    rotate,
    state_agent_rows,
    wrap_to_pi,
)

_OBSERVATION_KEY = "observation"


def _dynamics_fingerprint(config: Dict[str, Any]) -> str:
    """A canonical string of everything in a configuration except the observation.

    Serialised with sorted keys so the comparison is on content rather than on insertion
    order, and so a failure prints the two configurations side by side instead of a
    dictionary diff the reader has to line up by eye.
    """
    stripped = copy.deepcopy(config)
    del stripped[_OBSERVATION_KEY]
    return json.dumps(stripped, sort_keys=True)


def _example_observation() -> RacetrackObservation:
    """A well-formed reading: three curvature lookaheads, four detection slots, all empty.

    Built once because two tests need the same five channels at five different shapes, and
    the shapes are the point of both -- an inline copy that drifted in one of them would
    weaken the other silently.
    """
    return RacetrackObservation(
        ego_pose=np.array([12.0, -3.5, 0.4, 87.0], dtype=np.float32),
        ego_speed=np.array([8.0], dtype=np.float32),
        lane_pose=np.array([0.2, -0.05], dtype=np.float32),
        curvature_ahead=np.array([0.01, 0.02, 0.03], dtype=np.float32),
        detections=np.zeros((4, DETECTION_SLOT_WIDTH), dtype=np.float32),
    )


# ── The matched pair ────────────────────────────────────────────────────


def test_the_two_arms_share_a_byte_identical_dynamics_configuration() -> None:
    """Everything except the observation emission is identical between the arms.

    Purpose: Validates the matched-pair guarantee the whole environment exists for. If any
        dynamics key differed, a planner's performance gap between the arms would confound
        partial observability with a change in the world itself.

        The guarantee is re-expressed here, not weakened. It used to read "the two configs
        differ on exactly the observation key", which was complete only while highway-env
        produced both readings. It no longer does: the POMDP arm's lane pose, curvature and
        detections are measured world-side and never appear in the config at all. So the
        phrasing that still covers the redesign is the stronger half of the old one -- a
        byte-identical dynamics configuration -- plus the observation emission being the one
        thing allowed to differ.

    Given: The MDP and POMDP configurations built with identical arguments
    When: Each is serialised with its "observation" block removed and sorted keys
    Then: The two serialisations are equal string for string, the key sets match, and the
        observation blocks differ

    Test type: unit
    """
    mdp = build_racetrack_config(ObservationMode.MDP)
    pomdp = build_racetrack_config(ObservationMode.POMDP)

    assert set(mdp) == set(pomdp)
    assert _dynamics_fingerprint(mdp) == _dynamics_fingerprint(pomdp)
    assert mdp[_OBSERVATION_KEY] != pomdp[_OBSERVATION_KEY]


def test_matched_pair_survives_non_default_arguments() -> None:
    """Changing the tunables moves both arms together, never just one.

    Purpose: Validates that the matched pair is a property of the code path rather than
        of the default arguments alone.

    Given: A non-default set of step rates, weights and vehicle counts
    When: Both arms are built from exactly those arguments
    Then: Their dynamics configurations are still byte-identical, and the non-default
        values really did land

    Test type: unit
    """
    kwargs: Dict[str, Any] = {
        "max_tracked_agents": 2,
        "other_vehicles": 3,
        "duration": 40,
        "policy_frequency": 2,
        "simulation_frequency": 10,
        "collision_reward": -2.5,
        "lane_centering_cost": 6.0,
        "lane_centering_reward": 0.5,
        "action_reward": -0.1,
        "speed_limit": 22.0,
        "terminate_off_road": False,
    }

    mdp = build_racetrack_config(ObservationMode.MDP, **kwargs)
    pomdp = build_racetrack_config(ObservationMode.POMDP, **kwargs)

    assert _dynamics_fingerprint(mdp) == _dynamics_fingerprint(pomdp)
    assert mdp["policy_frequency"] == 2
    assert mdp["terminate_off_road"] is False


# ── Observation blocks ──────────────────────────────────────────────────


def test_pomdp_observation_block_asks_the_simulator_only_for_the_ego_kinematics() -> None:
    """The POMDP arm requests one thing from highway-env: the ego's own velocity row.

    Purpose: Pins what the simulator is asked for in the partially observed arm. Everything
        else the arm emits -- lane pose, curvature ahead, detections -- is measured by the
        world adapter off the road network and the vehicle list, because highway-env has no
        observation type reporting occlusion, a range gate or radial velocity. A block that
        quietly grew extra features would hand the planner state the redesign withholds.

    Given: The POMDP configuration
    When: Its "observation" block is inspected
    Then: It is an absolute, unnormalised, sorted Kinematics block of [vx, vy, cos_h, sin_h]

    Test type: unit
    """
    block = build_racetrack_config(ObservationMode.POMDP)[_OBSERVATION_KEY]

    assert block["type"] == "Kinematics"
    assert block["features"] == ["vx", "vy", "cos_h", "sin_h"]
    # Two, not one: at vehicles_count=1 highway-env 1.12.1 returns every nearby vehicle
    # rather than the ego alone. See the note beside the constant.
    assert block["vehicles_count"] == 2
    assert block["absolute"] is True
    assert block["normalize"] is False
    assert block["order"] == "sorted"
    assert block["see_behind"] is True


def test_mdp_observation_block_reports_absolute_unnormalised_kinematics() -> None:
    """The MDP arm reports one absolute kinematics row per tracked vehicle plus the ego.

    Purpose: Validates the near-MDP baseline's observation block, including the explicit
        absolute/unnormalised settings that override highway-env's relative defaults.

    Given: The MDP configuration with a chosen max_tracked_agents
    When: Its "observation" block is inspected
    Then: vehicles_count is max_tracked_agents + 1, and absolute, normalize and see_behind
        are set explicitly

    Test type: unit
    """
    block = build_racetrack_config(ObservationMode.MDP, max_tracked_agents=3)[_OBSERVATION_KEY]

    assert block["type"] == "Kinematics"
    assert block["features"] == ["presence", "x", "y", "vx", "vy"]
    assert block["vehicles_count"] == 3 + 1
    assert block["absolute"] is True
    assert block["normalize"] is False
    assert block["see_behind"] is True


def test_mdp_observation_order_is_sorted_and_never_shuffled() -> None:
    """The MDP rows are sorted, because shuffling would draw from the env's RNG.

    Purpose: Validates the "order" setting for the reason it was chosen. highway-env's
        "shuffled" ordering permutes the observation rows using the environment's own
        random generator. That draw would advance the RNG in the MDP arm and not in the
        POMDP arm, so the two arms would consume different randomness and their shared
        dynamics -- the matched pair -- would break silently, with nothing differing except
        the observation the test above already allows to differ.

    Given: The MDP configuration
    When: The "order" field of its observation block is read
    Then: It is "sorted", and specifically not "shuffled"

    Test type: unit
    """
    block = build_racetrack_config(ObservationMode.MDP)[_OBSERVATION_KEY]

    assert block["order"] == "sorted"
    assert block["order"] != "shuffled"


def test_observation_config_selects_the_arm_directly() -> None:
    """observation_config returns the same block build_racetrack_config attaches.

    Purpose: Validates the standalone observation-block helper against the assembled
        configuration, so callers that need only the block cannot drift from it.

    Given: Both observation modes and a fixed agent count
    When: observation_config is called directly
    Then: Each block equals the one in the corresponding full configuration

    Test type: unit
    """
    for mode in (ObservationMode.MDP, ObservationMode.POMDP):
        full = build_racetrack_config(mode, max_tracked_agents=2)
        assert observation_config(mode, 2) == full[_OBSERVATION_KEY]


# ── Dynamics keys ───────────────────────────────────────────────────────


def test_action_block_enables_longitudinal_control() -> None:
    """Acceleration is controllable, unlike the racetrack default.

    Purpose: Validates the deliberate deviation from the shipped racetrack action config.
        Under the default, ContinuousAction is lateral-only and acceleration is pinned at
        zero, so the ego could never brake for an opponent -- which removes most of what
        partial observability is supposed to cost.

    Given: Either arm's configuration
    When: The "action" block is inspected
    Then: Both longitudinal and lateral control are enabled

    Test type: unit
    """
    for mode in (ObservationMode.MDP, ObservationMode.POMDP):
        action = build_racetrack_config(mode)["action"]
        assert action["type"] == "ContinuousAction"
        assert action["longitudinal"] is True
        assert action["lateral"] is True


def test_action_presets_are_the_full_acceleration_by_steering_grid() -> None:
    """The preset table is every acceleration crossed with every steering angle, in order.

    Purpose: Validates the shape of the action vocabulary the world, the scalar model and
        the vectorized model all index into. They share it by construction, so an index
        that moves moves everywhere at once and nothing complains

    Given: The shipped acceleration and steering preset tuples
    When: DEFAULT_ACTION_PRESETS is inspected
    Then: It is the full product, acceleration-major, with no duplicates

    Test type: unit
    """
    expected = [
        (acceleration, steering)
        for acceleration in ACCELERATION_PRESETS
        for steering in STEERING_PRESETS
    ]
    assert list(DEFAULT_ACTION_PRESETS) == expected
    assert len(DEFAULT_ACTION_PRESETS) == len(ACCELERATION_PRESETS) * len(STEERING_PRESETS)
    assert len(set(DEFAULT_ACTION_PRESETS)) == len(DEFAULT_ACTION_PRESETS)


def test_the_shipped_detection_rates_are_zero_because_detection_is_deterministic() -> None:
    """Both detection rates default to zero, and that is a claim about this world.

    Purpose: The rates were 0.05 and 0.02, carried over from an earlier arm, and they
        described a radar this world does not have: the range gate and the occlusion rule run
        on true positions, and nothing is randomly dropped or invented. A nonzero default is
        unfittable here and it softens an exclusion the data actually justifies, so the value
        is worth pinning rather than leaving to whoever edits the module next

    Given: The shipped schema defaults
    When: The two detection rates are read
    Then: Both are exactly zero, and both are still accepted by the validator that would
        reject an unusable pair

    Test type: configuration
    """
    assert DEFAULT_PRESENCE_MISS_PROB == 0.0
    assert DEFAULT_PRESENCE_FALSE_ALARM_PROB == 0.0
    assert (
        validate_detection_rates(DEFAULT_PRESENCE_MISS_PROB, DEFAULT_PRESENCE_FALSE_ALARM_PROB)
        is None
    )


def test_steering_presets_are_fine_near_zero_and_include_the_useful_angle() -> None:
    """Steering is sampled finely near centre, not bang-bang between the lock stops.

    Purpose: Pins the reason the table is nine steering angles rather than three. Full lock
        is pi/4, which on this track is a spin: sweeping constant steering through the first
        bend, -1.0 survives 5 steps while -0.05 survives 29. A set of {-1, 0, +1} does not
        contain the manoeuvre the track needs, so no amount of planning can select it

    Given: The shipped steering presets
    When: They are inspected
    Then: They are symmetric about an exact zero, ascending, and include -0.05

    Test type: unit
    """
    assert 0.0 in STEERING_PRESETS
    assert -0.05 in STEERING_PRESETS
    assert list(STEERING_PRESETS) == sorted(STEERING_PRESETS)
    assert list(STEERING_PRESETS) == [-angle for angle in reversed(STEERING_PRESETS)]


def test_overrides_are_applied_to_both_arms() -> None:
    """An override lands identically in the MDP and POMDP configurations.

    Purpose: Validates that overrides are a shared dynamics knob, not a per-arm one.

    Given: An override dictionary setting a scenario key
    When: Both arms are built with it
    Then: Both carry the overridden value and their dynamics configurations still match

    Test type: unit
    """
    overrides = {"screen_width": 400, "vehicles_density": 2.0}

    mdp = build_racetrack_config(ObservationMode.MDP, overrides=overrides)
    pomdp = build_racetrack_config(ObservationMode.POMDP, overrides=overrides)

    assert mdp["screen_width"] == 400
    assert pomdp["vehicles_density"] == 2.0
    assert _dynamics_fingerprint(mdp) == _dynamics_fingerprint(pomdp)


def test_override_of_the_observation_key_is_rejected() -> None:
    """Overriding the observation is refused rather than silently honoured.

    Purpose: Validates the guard protecting the matched pair from the one override that
        could break it.

    Given: An override dictionary containing an "observation" entry
    When: A configuration is built with it
    Then: ValueError is raised naming the matched pair

    Test type: unit
    """
    with pytest.raises(ValueError, match="matched MDP/POMDP pair"):
        build_racetrack_config(
            ObservationMode.POMDP, overrides={"observation": {"type": "TimeToCollision"}}
        )


def test_non_integral_substep_ratio_is_rejected() -> None:
    """A simulation rate that is not a multiple of the policy rate raises.

    Purpose: Validates the step-rate guard. A fractional substep count would make the
        planner's model integrate a different number of physics steps than the world.

    Given: simulation_frequency 10 with policy_frequency 4
    When: A configuration is built
    Then: ValueError is raised mentioning the integer multiple requirement

    Test type: unit
    """
    with pytest.raises(ValueError, match="integer multiple"):
        build_racetrack_config(ObservationMode.POMDP, simulation_frequency=10, policy_frequency=4)


@pytest.mark.parametrize(
    "simulation_frequency,policy_frequency",
    [(0, 5), (15, 0), (-15, 5), (15, -5)],
)
def test_non_positive_step_rates_are_rejected(
    simulation_frequency: int, policy_frequency: int
) -> None:
    """Zero or negative step rates raise instead of producing a degenerate config.

    Purpose: Validates the positivity half of the step-rate guard.

    Given: A step-rate pair with a zero or negative entry
    When: A configuration is built
    Then: ValueError is raised mentioning positivity

    Test type: unit
    """
    with pytest.raises(ValueError, match="must be positive"):
        build_racetrack_config(
            ObservationMode.POMDP,
            simulation_frequency=simulation_frequency,
            policy_frequency=policy_frequency,
        )


def test_integral_substep_ratio_is_accepted() -> None:
    """A rate pair that divides evenly is accepted unchanged.

    Purpose: Validates the positive case of the step-rate guard, so the rejection tests
        above are not passing for an unrelated reason.

    Given: simulation_frequency 20 with policy_frequency 5
    When: A configuration is built
    Then: Both rates appear verbatim in the configuration

    Test type: unit
    """
    config = build_racetrack_config(
        ObservationMode.POMDP, simulation_frequency=20, policy_frequency=5
    )

    assert config["simulation_frequency"] == 20
    assert config["policy_frequency"] == 5


# ── Reward ──────────────────────────────────────────────────────────────


def test_reward_is_maximal_at_the_lane_centre_with_no_control_effort() -> None:
    """A centred, uncommanded, uncrashed step scores the top of the normalised range.

    Purpose: Validates the reward's maximum under the default weights.

    Given: Zero lateral offset, a zero command, no crash, on the road
    When: The reward is computed
    Then: It is exactly 1.0

    Test type: unit
    """
    assert racetrack_reward(0.0, (0.0, 0.0), False, True) == pytest.approx(1.0)


def test_reward_decreases_with_distance_from_the_lane_centre() -> None:
    """Drifting off the centreline costs reward, symmetrically in either direction.

    Purpose: Validates the lane-centering term's shape.

    Given: A sequence of increasing absolute lateral offsets
    When: Each is scored with the same zero command
    Then: The rewards decrease strictly, and +/- the same offset score alike

    Test type: unit
    """
    rewards = [
        racetrack_reward(lateral, (0.0, 0.0), False, True) for lateral in (0.0, 0.5, 1.0, 2.0)
    ]

    assert all(later < earlier for earlier, later in zip(rewards, rewards[1:]))
    assert racetrack_reward(-1.5, (0.0, 0.0), False, True) == pytest.approx(
        racetrack_reward(1.5, (0.0, 0.0), False, True)
    )


def test_control_effort_is_penalised() -> None:
    """A large command scores below the same transition with no command.

    Purpose: Validates the action-effort term.

    Given: An identical centred, uncrashed transition with and without a command
    When: Both are scored
    Then: The commanded one scores lower by the effort weight times the command norm,
        divided by the normalisation span

    Test type: unit
    """
    idle = racetrack_reward(0.0, (0.0, 0.0), False, True)
    commanded = racetrack_reward(0.0, (1.0, 0.0), False, True)

    span = 1.0 - DEFAULT_COLLISION_REWARD
    assert commanded == pytest.approx(idle + DEFAULT_ACTION_REWARD / span)
    assert commanded < idle


def test_crashing_lowers_the_reward() -> None:
    """The collision weight subtracts from an otherwise identical transition.

    Purpose: Validates the collision term.

    Given: The same centred, uncommanded transition with and without a crash
    When: Both are scored
    Then: The crashed one scores strictly lower, by the collision weight over the span

    Test type: unit
    """
    clean = racetrack_reward(0.0, (0.0, 0.0), False, True)
    crashed = racetrack_reward(0.0, (0.0, 0.0), True, True)

    span = 1.0 - DEFAULT_COLLISION_REWARD
    assert crashed == pytest.approx(clean + DEFAULT_COLLISION_REWARD / span)
    assert crashed < clean


def test_leaving_the_road_zeroes_the_reward_exactly() -> None:
    """Off-road transitions score exactly zero regardless of the other terms.

    Purpose: Validates the multiplicative on-road gate, which is a hard zero upstream
        rather than a penalty.

    Given: Several otherwise well-scoring and badly-scoring transitions, all off-road
    When: Each is scored
    Then: Every result is exactly 0.0

    Test type: unit
    """
    assert racetrack_reward(0.0, (0.0, 0.0), False, False) == 0.0
    assert racetrack_reward(2.0, (1.0, 1.0), True, False) == 0.0


def test_normalisation_uses_the_literal_upper_endpoint_and_does_not_clip() -> None:
    """An unusual lane_centering_reward pushes the result above 1 rather than clipping.

    Purpose: Pins two upstream quirks reproduced on purpose: the normalisation maps from
        [collision_reward, 1] using the literal 1 rather than lane_centering_reward, and
        it does not clip. Asserting the quirk is the point -- the reward exists to match
        highway-env's own, so "fixing" it here would silently desynchronise the two.

    Given: A lane_centering_reward of 3.0 on a centred, uncommanded, uncrashed step
    When: The reward is computed
    Then: It exceeds 1.0, at the exact unclipped value (3 - (-1)) / (1 - (-1)) = 2.0

    Test type: unit
    """
    reward = racetrack_reward(0.0, (0.0, 0.0), False, True, lane_centering_reward=3.0)

    assert reward > 1.0
    assert reward == pytest.approx(2.0)


def test_reward_honours_custom_weights() -> None:
    """Every weight argument reaches the formula.

    Purpose: Validates the keyword weights against a hand-computed value, so a wiring
        mistake in any one of them is caught.

    Given: A transition with a non-zero offset, a command, a crash, and custom weights
    When: The reward is computed
    Then: It equals the closed-form value computed from those same weights

    Test type: unit
    """
    lateral, action = 0.5, (1.0, -1.0)
    weights = {
        "collision_reward": -2.0,
        "lane_centering_cost": 2.0,
        "lane_centering_reward": 0.5,
        "action_reward": -0.25,
    }

    centering = 0.5 / (1.0 + 2.0 * 0.25)
    raw = centering + (-0.25) * math.sqrt(2.0) + (-2.0)
    expected = (raw - (-2.0)) / (1.0 - (-2.0))

    assert racetrack_reward(lateral, action, True, True, **weights) == pytest.approx(expected)


# ── State helpers ───────────────────────────────────────────────────────


def test_state_agent_rows_reshapes_the_agent_block() -> None:
    """The trailing agent slots are viewed as one row per slot.

    Purpose: Validates the state-layout helper the world, model and belief all share.

    Given: A state vector holding the ego block followed by two filled agent slots
    When: state_agent_rows is called for two tracked agents
    Then: The result is (2, 5) and each row holds that slot's values in order

    Test type: unit
    """
    ego = np.arange(EGO_STATE_WIDTH, dtype=float)
    slots = np.array([1.0, 3.0, -1.0, 0.5, 0.0, 1.0, 8.0, 2.0, -0.5, 0.25])
    state = np.concatenate([ego, slots])

    rows = state_agent_rows(state, 2)

    assert rows.shape == (2, AGENT_SLOT_WIDTH)
    assert np.array_equal(rows[0], np.array([1.0, 3.0, -1.0, 0.5, 0.0]))
    assert np.array_equal(rows[1], np.array([1.0, 8.0, 2.0, -0.5, 0.25]))


def test_state_agent_rows_rejects_a_mismatched_width() -> None:
    """A state whose width contradicts max_tracked_agents raises rather than reshaping.

    Purpose: Validates the width guard. A silent reshape of a wider or narrower vector
        would scramble the agent slots without any visible failure.

    Given: A state sized for four agent slots
    When: state_agent_rows is asked to read it as two
    Then: ValueError is raised naming both widths

    Test type: unit
    """
    state = np.zeros(EGO_STATE_WIDTH + DEFAULT_MAX_TRACKED_AGENTS * AGENT_SLOT_WIDTH)

    with pytest.raises(ValueError, match="does not match max_tracked_agents"):
        state_agent_rows(state, 2)


def test_state_agent_rows_supports_a_batch_of_states() -> None:
    """A stack of states reshapes to a stack of agent-row blocks.

    Purpose: Validates the leading-axis handling, which the belief relies on to reshape a
        whole particle set at once.

    Given: Three stacked states with three agent slots each
    When: state_agent_rows is called on the stack
    Then: The result is (3, 3, 5) and matches the per-state result row for row

    Test type: unit
    """
    states = np.arange(3 * (EGO_STATE_WIDTH + 3 * AGENT_SLOT_WIDTH), dtype=float).reshape(3, -1)

    rows = state_agent_rows(states, 3)

    assert rows.shape == (3, 3, AGENT_SLOT_WIDTH)
    assert np.array_equal(rows[1], state_agent_rows(states[1], 3))


def test_rotate_by_quarter_turn_maps_x_axis_onto_y_axis() -> None:
    """A +pi/2 rotation is counter-clockwise: (1, 0) becomes (0, 1).

    Purpose: Validates the rotation's sign convention, which fixes whether agent slots are
        read in the ego body frame or its mirror image.

    Given: The unit x vector
    When: It is rotated by +pi/2
    Then: The result is (0, 1)

    Test type: unit
    """
    rotated = rotate(np.array([1.0, 0.0]), np.pi / 2)

    assert rotated == pytest.approx(np.array([0.0, 1.0]))


def test_rotate_round_trips_under_the_inverse_angle() -> None:
    """Rotating by an angle and back returns the original vectors.

    Purpose: Validates that rotate is an exact inverse of itself at -angle, which is what
        makes the world-frame to body-frame conversion lossless.

    Given: A block of 2-D row vectors and an arbitrary angle
    When: They are rotated by the angle and then by its negation
    Then: The result matches the input, and the shape is preserved throughout

    Test type: unit
    """
    vectors = np.array([[1.0, 0.0], [0.0, -2.0], [3.0, 4.0]])
    angle = 0.7

    round_tripped = rotate(rotate(vectors, angle), -angle)

    assert round_tripped.shape == vectors.shape
    assert round_tripped == pytest.approx(vectors)


def test_rotate_preserves_vector_length() -> None:
    """Rotation is a rigid motion: norms are unchanged.

    Purpose: Validates that the rotation matrix is orthonormal, so relative positions and
        velocities keep their magnitudes when moved into the ego frame.

    Given: A block of 2-D row vectors
    When: They are rotated by an arbitrary angle
    Then: Each row's norm is unchanged

    Test type: unit
    """
    vectors = np.array([[1.0, 0.0], [0.0, -2.0], [3.0, 4.0]])

    rotated = rotate(vectors, -1.3)

    assert np.linalg.norm(rotated, axis=1) == pytest.approx(np.linalg.norm(vectors, axis=1))


@pytest.mark.parametrize(
    "angle,expected",
    [
        (3.0 * np.pi, -np.pi),
        (np.pi, -np.pi),
        (-np.pi, -np.pi),
        (2.0 * np.pi, 0.0),
        (0.0, 0.0),
        (0.5, 0.5),
        (-0.5, -0.5),
        (3.0 * np.pi / 2.0, -np.pi / 2.0),
    ],
)
def test_wrap_to_pi_maps_angles_into_the_half_open_interval(angle: float, expected: float) -> None:
    """Angles are wrapped into [-pi, pi), with the upper endpoint folding to -pi.

    Purpose: Validates the wrapping convention on its exact boundary cases. The interval
        is half-open, so both +pi and 3*pi land on -pi rather than +pi.

    Given: An angle inside or outside the interval, including both endpoints
    When: wrap_to_pi is applied
    Then: The result is the equivalent angle in [-pi, pi), and in-range angles are
        returned unchanged

    Test type: unit
    """
    assert wrap_to_pi(angle) == pytest.approx(expected)


def test_wrap_to_pi_returns_a_plain_float() -> None:
    """The wrapped angle is a Python float, not a NumPy scalar.

    Purpose: Validates the return type, which matters because wrapped angles are written
        into state vectors and step_info dictionaries that are compared by type elsewhere.

    Given: A NumPy float input
    When: wrap_to_pi is applied
    Then: The result is a plain float

    Test type: unit
    """
    wrapped = wrap_to_pi(float(np.float64(4.0)))

    assert isinstance(wrapped, float)
    assert not isinstance(wrapped, np.floating)


# ── The ego speedometer ─────────────────────────────────────────────────


@pytest.mark.parametrize("speed", [10.0, 0.0, -4.0, -30.0, 7.25])
@pytest.mark.parametrize("heading", [0.0, 0.7, -2.4, math.pi])
def test_ego_speed_is_recovered_with_its_sign_at_any_heading(speed: float, heading: float) -> None:
    """Projecting the velocity onto the heading returns the signed speed that produced it.

    Purpose: Validates the reduction the world applies to the simulator's kinematics row.
        The state's EGO_SPEED slot is signed and the racetrack ego really does reverse --
        braking flat out takes it to -30 m/s -- so a reduction that lost the sign would
        report a car accelerating backwards as one accelerating forwards.

    Given: A velocity built as speed * (cos heading, sin heading), for signed speeds and
        headings spanning the circle
    When: ego_speed_from_kinematics_row reduces the row
    Then: The original signed speed comes back

    Test type: unit
    """
    row = [
        speed * math.cos(heading),
        speed * math.sin(heading),
        math.cos(heading),
        math.sin(heading),
    ]

    assert ego_speed_from_kinematics_row(row) == pytest.approx(speed, abs=1e-12)


def test_ego_speed_keeps_the_sign_a_norm_would_discard() -> None:
    """A reversing ego reads negative, where hypot(vx, vy) would read positive.

    Purpose: Pins the one behaviour that separates this reduction from the obvious
        alternative. The norm agrees with it everywhere the ego drives forwards, so a test
        that only drives forwards would pass against the wrong implementation.

    Given: An ego reversing at 4 m/s along the +x heading
    When: The row is reduced
    Then: The result is -4.0, while the Euclidean norm of the same velocity is +4.0

    Test type: unit
    """
    row = [-4.0, 0.0, 1.0, 0.0]

    assert ego_speed_from_kinematics_row(row) == pytest.approx(-4.0)
    assert float(np.hypot(row[0], row[1])) == pytest.approx(4.0)


def test_ego_speed_rejects_a_row_of_the_wrong_width() -> None:
    """A row that is not the four configured features is refused rather than reduced.

    Purpose: The reduction indexes four features by position. Handed a two-feature
        [vx, vy] row it would otherwise unpack into a ValueError from NumPy with no clue
        about which observation block was misconfigured.

    Given: A two-element row
    When: It is reduced
    Then: ValueError names the expected feature list

    Test type: unit
    """
    with pytest.raises(ValueError, match="vx"):
        ego_speed_from_kinematics_row([1.0, 2.0])


def test_the_pomdp_arm_asks_for_more_than_one_vehicle_on_purpose() -> None:
    """The ego kinematics block requests two vehicles, working around an upstream bug.

    Purpose: highway-env 1.12.1 treats vehicles_count=1 as "no limit" -- close_objects_to
        guards its truncation with `if count:` and the slice [-vehicles_count + 1:] becomes
        [0:] -- so the obvious value returns every nearby vehicle's velocity. This test
        exists so that someone tidying the 2 back down to 1 has to read why.

    Given: The POMDP observation block
    When: Its vehicles_count is inspected
    Then: It is 2, not 1

    Test type: configuration
    """
    block = observation_config(ObservationMode.POMDP, DEFAULT_MAX_TRACKED_AGENTS)

    assert block["vehicles_count"] == 2


# ── The radar: range gate and occlusion ─────────────────────────────────


def test_detection_visibility_gates_on_range_with_the_limit_itself_inside() -> None:
    """A vehicle at exactly the stated range is reported; half a metre beyond it is not.

    Purpose: Validates the range gate on its own boundary. The world and the planner's
        model must agree on which side of it a vehicle falls, or the model predicts a
        detection the world never reported -- and at the shipped speed limit an opponent
        crosses the boundary every few seconds, so this is the common case.

    Given: A single vehicle straight ahead at 39.5 m, 40.0 m and 40.5 m, with a 40 m gate
    When: detection_visibility is asked about each
    Then: The 39.5 m and 40.0 m rows are visible and the 40.5 m row is not

    Test type: unit
    """
    present = np.array([True])

    inside = detection_visibility(np.array([[39.5, 0.0]]), present, 40.0)
    at_limit = detection_visibility(np.array([[40.0, 0.0]]), present, 40.0)
    beyond = detection_visibility(np.array([[40.5, 0.0]]), present, 40.0)

    assert list(inside) == [True]
    assert list(at_limit) == [True]
    assert list(beyond) == [False]


def test_detection_visibility_masks_a_vehicle_behind_a_closer_one() -> None:
    """A vehicle directly behind a nearer one is absent from the reading entirely.

    Purpose: Validates the occlusion rule, which is one of the four things the POMDP arm
        hides on purpose. The blocker at 10 m subtends a half-angle of arcsin(1/10) = 0.100
        rad, and the vehicle at 25 m sits at a bearing of exactly 0, so it falls inside that
        cone and is not reported.

    Given: Two vehicles straight ahead at 10 m and 25 m, both well inside a 40 m gate
    When: detection_visibility is asked about them
    Then: The nearer is visible, the further is masked, and the masking survives dropping
        the gate entirely

    Test type: unit
    """
    positions = np.array([[10.0, 0.0], [25.0, 0.0]])
    present = np.array([True, True])

    assert list(detection_visibility(positions, present, 40.0)) == [True, False]
    assert list(detection_visibility(positions, present, float("inf"))) == [True, False]
    assert math.asin(1.0 / 10.0) == pytest.approx(0.10017, abs=1e-5)


def test_detection_visibility_reports_a_vehicle_beside_the_blocker() -> None:
    """Stepping out of the blocker's cone is enough to be seen again.

    Purpose: Validates that occlusion is the finite angular half-width of a disc and not a
        blanket "anything further away is hidden". The two cases straddle the cone edge by a
        metre of lateral offset, so a wrong half-width -- or none -- fails one of them.

    Given: A blocker 10 m ahead, whose cone is arcsin(1/10) = 0.1002 rad wide, and a second
        vehicle at 25 m offset either 3 m sideways (bearing 0.1194 rad, outside the cone) or
        2 m sideways (bearing 0.0798 rad, inside it)
    When: detection_visibility is asked about each pair
    Then: The 3 m offset vehicle is reported and the 2 m offset one is masked

    Test type: unit
    """
    present = np.array([True, True])
    beside = np.array([[10.0, 0.0], [25.0, 3.0]])
    behind = np.array([[10.0, 0.0], [25.0, 2.0]])

    assert list(detection_visibility(beside, present, 40.0)) == [True, True]
    assert list(detection_visibility(behind, present, 40.0)) == [True, False]
    assert math.atan2(3.0, 25.0) > math.asin(1.0 / 10.0) > math.atan2(2.0, 25.0)


def test_detection_visibility_counts_a_blocker_the_range_gate_itself_drops() -> None:
    """Occlusion is geometry: a blocker occludes whether or not it is reported.

    Purpose: Pins that blockers are collected before the range gate is applied. The two
        rules cannot contradict each other -- anything a blocker hides is further away than
        the blocker, so it is outside any gate the blocker is outside of -- but the ordering
        still has to hold for the widened gate below, where the blocker is reported and the
        vehicle behind it is still not.

    Given: A vehicle at 25 m and one directly behind it at 30 m
    When: The pair is gated at 20 m and then at 40 m
    Then: At 20 m neither is reported; at 40 m the blocker is reported and the vehicle
        behind it is still masked

    Test type: unit
    """
    positions = np.array([[25.0, 0.0], [30.0, 0.0]])
    present = np.array([True, True])

    assert list(detection_visibility(positions, present, 20.0)) == [False, False]
    assert list(detection_visibility(positions, present, 40.0)) == [True, False]


def test_detection_visibility_ignores_absent_rows_in_both_roles() -> None:
    """An empty slot is never reported and never occludes.

    Purpose: Validates the `present` mask on both sides of the rule. The planner's model
        calls this on a particle's agent slots, most of which are empty in light traffic;
        an empty slot that still cast a shadow would have the model predicting a miss the
        world never had a vehicle to produce.

    Given: An empty slot 5 m straight ahead and a real vehicle 10 m straight ahead behind it
    When: detection_visibility is asked about the pair
    Then: The empty slot is not reported and the real vehicle behind it is

    Test type: unit
    """
    positions = np.array([[5.0, 0.0], [10.0, 0.0]])

    visible = detection_visibility(positions, np.array([False, True]), 40.0)

    assert list(visible) == [False, True]
    assert list(detection_visibility(positions, np.array([True, True]), 40.0)) == [True, False]


def test_the_range_dial_is_the_only_thing_between_the_two_arms() -> None:
    """Widening the gate reports more cars and never fewer; at a huge R it reports them all.

    Purpose: Checks the ObservationMode.POMDP docstring's central claim where it is
        machine-checkable. That docstring says the two arms are a continuum in one number --
        as ``max_detection_range_m -> inf`` the POMDP reading becomes the state to within
        the sensor widths, and as it shrinks the traffic drops out of the reading first.
        The cheap and checkable end of that claim is this function: nothing but range and
        occlusion may remove a vehicle, so a gate large enough to hold every vehicle must
        report every unoccluded one. A gate that quietly dropped, say, vehicles behind the
        ego would break the continuum while every test using a 40 m gate still passed,
        because at 40 m those cars are usually out of range anyway.

        The other half -- that the reported numbers converge on the state -- is a statement
        about the world adapter's noise widths and is measured where the adapter is.

    Given: Five vehicles at ranges from 10 m to 1 km, on bearings spread far wider than any
        blocker's angular half-width so none occludes another
    When: They are gated at the shipped default range and then at 1e9 m
    Then: The default gate reports only the two inside it, the huge gate reports all five,
        the default gate's reported set is a subset of the huge gate's, and the shipped
        default is itself a positive finite number rather than an infinity

    Test type: unit
    """
    positions = np.array(
        [[10.0, 0.0], [24.0, 25.0], [-40.0, -45.0], [-150.0, 60.0], [800.0, -600.0]]
    )
    present = np.ones(len(positions), dtype=bool)

    shipped = detection_visibility(positions, present, DEFAULT_MAX_DETECTION_RANGE_M)
    unbounded = detection_visibility(positions, present, 1e9)

    assert 0.0 < DEFAULT_MAX_DETECTION_RANGE_M < float("inf")
    assert list(shipped) == [True, True, False, False, False]
    assert list(unbounded) == [True] * len(positions)
    assert np.all(unbounded | ~shipped)


# ── The closing-rate projection ─────────────────────────────────────────
#
# These are tests of :func:`radial_velocities` as a *projection*, not as an observation
# channel. Detections carry both components of relative velocity, so nothing on the sensing
# path calls this function any more; it survives because the closing rate is the quantity a
# time-to-collision or a gap-acceptance rule is written in, and deriving it from a detection
# row is a one-liner nobody should write twice.


def test_radial_velocity_returns_the_full_speed_for_pure_closing_motion() -> None:
    """Motion straight along the line of sight projects at its full magnitude.

    Purpose: Validates the projection's magnitude and its sign convention, which is
        negative while the range shrinks. A sign flip here would have every caller deriving
        a time-to-collision read an approaching car as a departing one.

    Given: A vehicle 10 m ahead closing at 2 m/s, one at (3, 4) -- range 5 -- closing at
        5 m/s straight down its own line of sight, and one 10 m ahead pulling away at 3 m/s
    When: radial_velocities projects each
    Then: The closing pair read -2.0 and -5.0, and the departing one reads +3.0

    Test type: unit
    """
    positions = np.array([[10.0, 0.0], [3.0, 4.0], [10.0, 0.0]])
    velocities = np.array([[-2.0, 0.0], [-3.0, -4.0], [3.0, 0.0]])

    closing = radial_velocities(positions, velocities)

    assert closing == pytest.approx(np.array([-2.0, -5.0, 3.0]))


def test_radial_velocity_is_zero_for_pure_crossing_motion() -> None:
    """The projection discards the tangential half, which is what makes it a projection.

    Purpose: Pins that this function returns a closing rate and not a speed. A car crossing
        the ego's path at 5 m/s directly abeam has a range that is momentarily unchanging,
        so its closing rate is exactly zero however fast it is moving -- and a caller
        writing a time-to-collision needs that, because the crossing component contributes
        nothing to the range closing.

        This is no longer a statement about hidden state. The observation reports both
        components of a visible vehicle's relative velocity, so the 5 m/s crossing below is
        in the reading; it is absent from *this number* alone. The invariant covering the
        observation is
        :func:`test_detection_slot_carries_both_relative_velocity_components`.

    Given: A vehicle 10 m ahead moving 5 m/s to the left, one 6 m abeam moving 4 m/s
        backwards, and one 10 m ahead closing at 3 m/s while also crossing at 5 m/s
    When: radial_velocities projects each
    Then: The two pure crossings read exactly 0.0, and the mixed one reads -3.0 with its
        5 m/s crossing component projected away

    Test type: unit
    """
    positions = np.array([[10.0, 0.0], [0.0, 6.0], [10.0, 0.0]])
    velocities = np.array([[0.0, 5.0], [-4.0, 0.0], [-3.0, 5.0]])

    closing = radial_velocities(positions, velocities)

    assert closing == pytest.approx(np.array([0.0, 0.0, -3.0]))


def test_radial_velocity_is_zero_for_a_row_sitting_on_the_ego() -> None:
    """A detection at the origin returns 0.0 rather than a division by zero.

    Purpose: Validates the degenerate case. The line of sight is undefined at zero range,
        and an empty slot in a particle's agent block is exactly a row of zeros, so a
        caller projecting a whole agent block hits this row routinely rather than
        pathologically -- and one NaN propagates through everything derived from it.

    Given: A row at the origin carrying a 5 m/s velocity, alongside an ordinary row
    When: radial_velocities projects them
    Then: The origin row reads exactly 0.0, the ordinary row is unaffected, and nothing is
        NaN

    Test type: unit
    """
    positions = np.array([[0.0, 0.0], [8.0, 0.0]])
    velocities = np.array([[3.0, -4.0], [-1.5, 0.0]])

    closing = radial_velocities(positions, velocities)

    assert closing == pytest.approx(np.array([0.0, -1.5]))
    assert np.all(np.isfinite(closing))


# ── The flattened observation layout ────────────────────────────────────


def test_pomdp_observation_width_counts_every_channel_once() -> None:
    """The flat width is 4 + 1 + 2 + L + 5K, the documented layout summed.

    Purpose: Validates the width the torch model's [N, do] tensors are allocated at against
        the layout the scalar model unflattens. The two models are compared entry for entry
        in a parity test, so a width that drifted would misalign every channel after the
        one that moved rather than failing loudly.

        Asserted against the width *constants* rather than against the literal 30 the
        shipped counts produce. A literal passes whenever the arithmetic happens to land on
        it -- widening the ego pose while narrowing a detection slot, say -- and it also
        fails for a deliberate, correctly propagated change, which trains the reader to
        update the number rather than to check the layout.

    Given: The shipped four detection slots and three curvature lookaheads, then the same
        with one more of each
    When: pomdp_observation_width is evaluated
    Then: It is the four channel widths summed, an empty reading is exactly the fixed
        channels, and the marginal cost of a detection slot is one whole slot width while
        the marginal cost of a lookahead is one entry

    Test type: unit
    """
    fixed_channels = OBSERVED_EGO_POSE_WIDTH + OBSERVED_EGO_SPEED_WIDTH + OBSERVED_LANE_POSE_WIDTH

    assert pomdp_observation_width(4, 3) == fixed_channels + 3 + 4 * DETECTION_SLOT_WIDTH
    assert pomdp_observation_width(0, 0) == fixed_channels
    assert pomdp_observation_width(5, 3) - pomdp_observation_width(4, 3) == DETECTION_SLOT_WIDTH
    assert pomdp_observation_width(4, 4) - pomdp_observation_width(4, 3) == 1


def test_pomdp_obs_offsets_tile_the_flat_vector_without_gaps_or_overlaps() -> None:
    """Every channel's offset is the previous one's offset plus the previous one's width.

    Purpose: Validates the offsets against the widths they are derived from, rather than
        against the numbers they currently evaluate to. The failure this guards against is
        an offset moved without the width following it -- the ego pose gaining a fifth
        entry while POMDP_OBS_EGO_SPEED_INDEX stays at 4, say. That leaves every constant
        individually defensible and the layout silently overlapping, and it does not change
        the total width, so a test asserting 30 sails past it. Concatenating the channels
        back and requiring the original vector catches it: an overlap repeats an entry and
        a gap drops one.

    Given: A flat reading numbered 0..width-1 for two detection slots and three curvature
        lookaheads
    When: It is sliced channel by channel at the declared offsets and the pieces are
        concatenated back
    Then: The pose starts at entry zero, each channel begins exactly where the previous one
        ended, the detections fill the rest to the last entry, and the reassembled vector
        is the original

    Test type: unit
    """
    lookahead_count, max_detections = 3, 2
    width = pomdp_observation_width(max_detections, lookahead_count)
    flat = np.arange(width, dtype=float)
    detections_index = POMDP_OBS_CURVATURE_INDEX + lookahead_count
    channels = [
        (POMDP_OBS_EGO_POSE_INDEX, OBSERVED_EGO_POSE_WIDTH),
        (POMDP_OBS_EGO_SPEED_INDEX, OBSERVED_EGO_SPEED_WIDTH),
        (POMDP_OBS_LANE_POSE_INDEX, OBSERVED_LANE_POSE_WIDTH),
        (POMDP_OBS_CURVATURE_INDEX, lookahead_count),
        (detections_index, max_detections * DETECTION_SLOT_WIDTH),
    ]

    assert POMDP_OBS_EGO_POSE_INDEX == 0
    for (start, channel_width), (next_start, _) in zip(channels, channels[1:]):
        assert start + channel_width == next_start
    assert channels[-1][0] + channels[-1][1] == width

    reassembled = np.concatenate([flat[start : start + size] for start, size in channels])
    assert list(reassembled) == list(flat)


def test_pomdp_obs_detection_block_reshapes_to_five_wide_rows_ending_the_vector() -> None:
    """The tail of the flat reading is K rows of five, the last entry being rel_vy.

    Purpose: Validates the one arithmetic step a caller has to do for itself. The detections
        have no offset constant of their own -- they start after a variable number of
        curvature samples -- so "curvature index plus lookahead count" is the layout rule,
        and it is pinned here rather than re-derived in each model.

    Given: A flat reading numbered 0..width-1 for two detection slots and three lookaheads
    When: The tail is reshaped at the detection slot width
    Then: The block is (2, 5), its first row is the five entries following the curvature
        samples, and the last row's rel_vy column is the final entry of the whole vector

    Test type: unit
    """
    lookahead_count, max_detections = 3, 2
    width = pomdp_observation_width(max_detections, lookahead_count)
    flat = np.arange(width, dtype=float)
    detections_index = POMDP_OBS_CURVATURE_INDEX + lookahead_count

    detections = flat[detections_index:].reshape(max_detections, DETECTION_SLOT_WIDTH)

    assert detections.shape == (2, DETECTION_SLOT_WIDTH)
    assert list(detections[0]) == list(
        range(detections_index, detections_index + DETECTION_SLOT_WIDTH)
    )
    assert detections[-1, DETECTION_REL_VY] == width - 1
    assert detections[-1, DETECTION_PRESENT] == width - DETECTION_SLOT_WIDTH


def test_racetrack_observation_names_one_field_per_sensor() -> None:
    """The reading is five named channels with the shapes the world promises.

    Purpose: Validates the container the POMDP arm emits. The channels are one field per
        *sensor* rather than per number, so anything identifying a channel by its size --
        a lone lateral offset mistaken for a second speedometer -- is caught here, and the
        detection column order is pinned alongside because the belief indexes those columns
        by name.

        The field *order* is asserted as a tuple and not merely as a set, because the flat
        layout the torch model indexes is this order concatenated: ego_pose leads, and a
        reading that named the same five channels in a different order would flatten into a
        vector every offset constant then reads wrongly.

    Given: A reading built for three curvature lookaheads and four detection slots
    When: Its fields and shapes are inspected
    Then: The fields are ego_pose, ego_speed, lane_pose, curvature_ahead and detections
        with shapes (4,), (1,), (2,), (3,) and (4, 5), and the detection columns are
        [detected, rel_x, rel_y, rel_vx, rel_vy]

    Test type: unit
    """
    observation = _example_observation()

    assert observation._fields == (
        "ego_pose",
        "ego_speed",
        "lane_pose",
        "curvature_ahead",
        "detections",
    )
    assert observation.ego_pose.shape == (OBSERVED_EGO_POSE_WIDTH,)
    assert observation.ego_speed.shape == (OBSERVED_EGO_SPEED_WIDTH,)
    assert observation.lane_pose.shape == (OBSERVED_LANE_POSE_WIDTH,)
    assert observation.curvature_ahead.shape == (3,)
    assert observation.detections.shape == (4, DETECTION_SLOT_WIDTH)
    assert (
        DETECTION_PRESENT,
        DETECTION_REL_X,
        DETECTION_REL_Y,
        DETECTION_REL_VX,
        DETECTION_REL_VY,
    ) == (0, 1, 2, 3, 4)


def test_racetrack_observation_refuses_to_become_one_array() -> None:
    """np.asarray on the reading raises rather than producing something wrong.

    Purpose: Pins the reason the reading is a named tuple of arrays and not a stacked one.
        Its channels have nothing in common -- a pose, metres per second, a metre-and-radian
        pair, reciprocal metres, and a (K, 5) block -- so anything that flattened them into
        one array would either be ragged or would have to invent a padding convention. The
        error is the useful behaviour: a caller who writes np.asarray(observation) is told
        so at the call site instead of carrying an object array of five buffers into
        arithmetic that quietly produces nonsense.

    Given: A well-formed reading with channels of five different shapes
    When: It is handed to np.asarray
    Then: ValueError is raised naming the inhomogeneous shape, and the reading's own
        channels are still usable afterwards

    Test type: unit
    """
    observation = _example_observation()

    with pytest.raises(ValueError, match="inhomogeneous"):
        np.asarray(observation)

    assert observation.detections.shape == (4, DETECTION_SLOT_WIDTH)


def test_detection_slot_carries_both_relative_velocity_components() -> None:
    """A detection reports rel_vx and rel_vy in two distinct columns, not one closing rate.

    Purpose: Replaces the invariant the redesign removed. A detection used to carry a single
        Doppler closing rate, so a car crossing directly abeam was reported as stationary
        and its crossing rate had to be inferred. It now carries the vehicle's whole
        relative velocity, and the layout is where that is either true or not: a slot width
        of four with one velocity column would restore the old sensor while every caller
        still compiled.

    Given: A detection slot holding a vehicle crossing at 5 m/s with no closing rate at all
    When: Its columns are read by name
    Then: The two velocity columns are distinct, adjacent and the last two of the slot, and
        the crossing rate survives in rel_vy where a closing-rate-only slot would have
        reported zero

    Test type: unit
    """
    slot = np.zeros(DETECTION_SLOT_WIDTH)
    slot[DETECTION_PRESENT] = 1.0
    slot[DETECTION_REL_X] = 10.0
    slot[DETECTION_REL_VY] = 5.0

    assert DETECTION_REL_VX != DETECTION_REL_VY
    assert DETECTION_REL_VY == DETECTION_REL_VX + 1
    assert DETECTION_REL_VY == DETECTION_SLOT_WIDTH - 1
    assert slot[DETECTION_REL_VY] == pytest.approx(5.0)

    old_channel = radial_velocities(
        slot[None, DETECTION_REL_X : DETECTION_REL_Y + 1], np.array([[0.0, 5.0]])
    )
    assert old_channel == pytest.approx(np.array([0.0]))


def test_observed_ego_pose_is_four_contiguous_channels_measured_near_exactly() -> None:
    """The pose channel is [x, y, heading, arclength], four wide, at small non-zero widths.

    Purpose: Validates the channel the redesign added, on the two properties the design
        argument rests on. Four contiguous columns, because the pose is flattened into the
        head of the observation vector and a gap or a repeat there shifts every later
        channel. And small but strictly positive widths, because the pose is meant to be
        near-exact -- a production stack localises to decimetres -- while a zero width would
        make the channel a delta in the likelihood and annihilate the first particle whose
        dead reckoning missed by a hair.

    Given: The ego-pose column constants and their shipped noise widths
    When: They are inspected against the pose width and against the detection position width
    Then: The columns are 0..3 with no gaps, the width is 4, every noise width is positive
        and finite, the two distance widths are tighter than the radar's own position width,
        and the heading width is under a degree

    Test type: unit
    """
    columns = (EGO_POSE_X, EGO_POSE_Y, EGO_POSE_HEADING, EGO_POSE_ARCLENGTH)

    assert columns == (0, 1, 2, 3)
    assert len(set(columns)) == OBSERVED_EGO_POSE_WIDTH
    assert max(columns) == OBSERVED_EGO_POSE_WIDTH - 1

    widths = (
        DEFAULT_EGO_POSITION_STD_M,
        DEFAULT_EGO_HEADING_STD_RAD,
        DEFAULT_EGO_ARCLENGTH_STD_M,
    )
    assert all(0.0 < width < float("inf") for width in widths)
    assert DEFAULT_EGO_POSITION_STD_M < DEFAULT_DETECTION_POSITION_STD_M
    assert DEFAULT_EGO_ARCLENGTH_STD_M < DEFAULT_DETECTION_POSITION_STD_M
    assert DEFAULT_EGO_HEADING_STD_RAD < math.radians(1.0)


def test_detection_velocity_width_is_one_number_covering_both_components() -> None:
    """The renamed velocity width kept its value, so measurements stay comparable.

    Purpose: Pins the rename rather than the number. DEFAULT_DETECTION_RADIAL_VELOCITY_STD
        became DEFAULT_DETECTION_VELOCITY_STD when the detection stopped being a Doppler
        closing rate and started carrying both components, and the value was deliberately
        carried over unchanged: it is the sensor's velocity accuracy, and the axis it
        happened to be quoted along is not what made it small. Holding it fixed is what
        keeps every likelihood measured before the redesign comparable with one measured
        after. A single width also means the two components are scored alike, so there is
        one number here and not two.

    Given: The shipped detection velocity width
    When: It is inspected
    Then: It is positive, finite, unchanged at 0.3 m/s, and there is no separate per-axis
        width to disagree with it

    Test type: unit
    """
    assert 0.0 < DEFAULT_DETECTION_VELOCITY_STD < float("inf")
    assert DEFAULT_DETECTION_VELOCITY_STD == pytest.approx(0.3)
