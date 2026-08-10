# SPDX-License-Identifier: MIT

"""Unit tests for the shared racetrack schema: config, reward and state helpers.

Nothing here touches ``highway_env``. The schema module is deliberately backend-free so
the planner's model and the belief can be built where the simulator is not installed, and
these tests hold that line by importing only NumPy alongside the module under test.

The centre of the suite is the *matched pair*: the MDP and POMDP configurations must
differ in the ``"observation"`` key and in nothing else, because that is the only reason a
performance gap between the two arms can be attributed to partial observability.
"""

import copy
import math
from typing import Any, Dict

import numpy as np
import pytest

from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    ACCELERATION_PRESETS,
    AGENT_SLOT_WIDTH,
    DEFAULT_ACTION_PRESETS,
    DEFAULT_ACTION_REWARD,
    DEFAULT_COLLISION_REWARD,
    DEFAULT_MAX_TRACKED_AGENTS,
    EGO_STATE_WIDTH,
    GRID_HALF_EXTENT_M,
    GRID_STEP_M,
    STEERING_PRESETS,
    ObservationMode,
    build_racetrack_config,
    observation_config,
    racetrack_reward,
    rotate,
    state_agent_rows,
    wrap_to_pi,
)

_OBSERVATION_KEY = "observation"


def _dynamics_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    """A configuration with the one key the two arms are allowed to differ in removed."""
    stripped = copy.deepcopy(config)
    del stripped[_OBSERVATION_KEY]
    return stripped


# ── The matched pair ────────────────────────────────────────────────────


def test_mdp_and_pomdp_configs_are_equal_apart_from_the_observation() -> None:
    """The two arms share every dynamics key and differ only in the observation.

    Purpose: Validates the matched-pair guarantee the whole environment exists for --
        if any key besides "observation" differed, a planner's performance gap between
        the arms would confound partial observability with a dynamics change.

    Given: The MDP and POMDP configurations built with identical arguments
    When: The "observation" key is deleted from both
    Then: The remaining dictionaries are equal, and the full key sets match

    Test type: unit
    """
    mdp = build_racetrack_config(ObservationMode.MDP)
    pomdp = build_racetrack_config(ObservationMode.POMDP)

    assert set(mdp) == set(pomdp)
    assert _dynamics_keys(mdp) == _dynamics_keys(pomdp)
    assert mdp[_OBSERVATION_KEY] != pomdp[_OBSERVATION_KEY]


def test_matched_pair_survives_non_default_arguments() -> None:
    """Changing the tunables moves both arms together, never just one.

    Purpose: Validates that the matched pair is a property of the code path rather than
        of the default arguments alone.

    Given: A non-default set of step rates, weights and vehicle counts
    When: Both arms are built from exactly those arguments
    Then: They still agree on every key except "observation"

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

    assert _dynamics_keys(mdp) == _dynamics_keys(pomdp)
    assert mdp["policy_frequency"] == 2
    assert mdp["terminate_off_road"] is False


# ── Observation blocks ──────────────────────────────────────────────────


def test_pomdp_observation_block_matches_the_shipped_occupancy_grid() -> None:
    """The POMDP arm asks for the racetrack's own presence/on-road occupancy grid.

    Purpose: Validates the partially observed arm's observation block against the values
        verified for highway-env 1.12.1.

    Given: The POMDP configuration
    When: Its "observation" block is inspected
    Then: It is a +/-18 m, 3 m-step OccupancyGrid of presence and on_road, not an image,
        aligned to the vehicle axes

    Test type: unit
    """
    block = build_racetrack_config(ObservationMode.POMDP)[_OBSERVATION_KEY]

    assert block["type"] == "OccupancyGrid"
    assert block["features"] == ["presence", "on_road"]
    assert block["grid_size"] == [
        [-GRID_HALF_EXTENT_M, GRID_HALF_EXTENT_M],
        [-GRID_HALF_EXTENT_M, GRID_HALF_EXTENT_M],
    ]
    assert block["grid_step"] == [GRID_STEP_M, GRID_STEP_M]
    assert block["as_image"] is False
    assert block["align_to_vehicle_axes"] is True


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
        dynamics -- the matched pair -- would break silently, with no key differing except
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
    Then: Both carry the overridden value and still agree away from "observation"

    Test type: unit
    """
    overrides = {"screen_width": 400, "vehicles_density": 2.0}

    mdp = build_racetrack_config(ObservationMode.MDP, overrides=overrides)
    pomdp = build_racetrack_config(ObservationMode.POMDP, overrides=overrides)

    assert mdp["screen_width"] == 400
    assert pomdp["vehicles_density"] == 2.0
    assert _dynamics_keys(mdp) == _dynamics_keys(pomdp)


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
