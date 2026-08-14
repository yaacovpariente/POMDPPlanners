# SPDX-License-Identifier: MIT

"""Unit tests for the analytic manipulator model: kinematics, joint lag and reach reward.

The kinematics is anchored to a published pose rather than to itself, because a DH table with one
sign flipped still produces smooth, plausible arm trajectories — it just reaches to the wrong
place, which reads downstream as "the planner never succeeded".

The action-separation tests are the regression guard for the failure this model exists to fix: a
ridge-fitted linear map over a 7-DoF arm scored every action alike, so the planner picked one index
and held it for the whole episode.
"""

from typing import Any

import numpy as np
import pytest

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
    IsaacChannelSchema,
    JointLagTransition,
    ManipulatorIsaacModel,
    ModifiedDHChain,
    ReachRewardModel,
    calibrate_tracking_gain,
    franka_panda_chain,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
    GaussianChannelObservationModel,
)

JOINTS = 7
SCHEMA = IsaacChannelSchema(
    (("joint_pos", JOINTS), ("joint_vel", JOINTS), ("command", 7), ("last_action", JOINTS))
)

#: The Panda's documented "home" configuration.
HOME_POSE = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])

#: Published ``panda_link8`` / ``panda_hand`` position at :data:`HOME_POSE`, in metres. The widely
#: quoted TCP value ``(0.307, 0, 0.487)`` is this frame less the 0.1034 m gripper offset.
HOME_FLANGE_POSITION = np.array([0.3069, 0.0, 0.5903])

#: IsaacLab's default Franka joint pose, the offset its ``joint_pos_rel`` observation subtracts.
ISAAC_DEFAULT_POSE = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])


def _exact_transition(**overrides: Any) -> JointLagTransition:
    settings: dict = {
        "position_width": JOINTS,
        "step_dt": 0.1,
        "tracking_gain": 0.5,
        "action_scale": 1.0,
        "position_noise_std": 1e-9,
        "velocity_noise_std": 1e-9,
        "action_noise_std": 1e-9,
    }
    settings.update(overrides)
    return JointLagTransition(**settings)


def _model(**overrides: Any) -> ManipulatorIsaacModel:
    settings: dict = {
        "state_schema": SCHEMA,
        "action_presets": [np.zeros(JOINTS), np.full(JOINTS, 0.4), np.full(JOINTS, -0.4)],
        "discount_factor": 0.99,
        "step_dt": 0.1,
        "tracking_gain": 0.5,
        "chain": franka_panda_chain(),
        "default_joint_positions": ISAAC_DEFAULT_POSE,
        "position_noise_std": 1e-9,
        "velocity_noise_std": 1e-9,
        "action_noise_std": 1e-9,
        "observation_models": {
            "joint_pos": GaussianChannelObservationModel(channel="joint_pos", noise_std=0.01)
        },
    }
    settings.update(overrides)
    return ManipulatorIsaacModel(**settings)


def _state(joint_pos: np.ndarray, command: np.ndarray) -> np.ndarray:
    return SCHEMA.pack(
        {
            "joint_pos": joint_pos,
            "joint_vel": np.zeros(JOINTS),
            "command": command,
            "last_action": np.zeros(JOINTS),
        }
    )


# ── Forward kinematics ──────────────────────────────────────────────────


def test_franka_home_pose_matches_the_published_flange_position() -> None:
    """A DH table is only worth having if it agrees with the real robot's documented geometry.

    Purpose: Anchors the Panda DH parameters to an externally published pose

    Given: The Panda's documented home joint configuration
    When: Forward kinematics places the hand frame
    Then: The position matches the published flange position to a tenth of a millimetre

    Test type: unit
    """
    position = franka_panda_chain().end_effector_position(HOME_POSE)
    assert position == pytest.approx(HOME_FLANGE_POSITION, abs=1e-4)


def test_forward_kinematics_returns_a_rigid_transform() -> None:
    """A chain that quietly scales or shears would place the hand plausibly but wrongly.

    Purpose: Validates that the composed DH transform is a rotation plus a translation

    Given: An arbitrary joint configuration
    When: The full end-effector transform is computed
    Then: Its rotation block is orthonormal with unit determinant and its last row is (0,0,0,1)

    Test type: unit
    """
    transform = franka_panda_chain().end_effector_transform(np.linspace(-1.0, 1.0, JOINTS))
    rotation = transform[:3, :3]
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-9)
    assert float(np.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-9)
    assert transform[3] == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_batched_kinematics_agrees_with_the_single_pose_path() -> None:
    """Belief particles arrive in batches; a batched path that disagreed would corrupt the reward.

    Purpose: Validates that the batched and unbatched FK paths return the same positions

    Given: Three different joint configurations
    When: They are evaluated one at a time and as a batch
    Then: The two sets of positions agree, and the batch has one row per configuration

    Test type: unit
    """
    chain = franka_panda_chain()
    poses = np.stack([HOME_POSE, np.zeros(JOINTS), ISAAC_DEFAULT_POSE])
    batched = chain.end_effector_position(poses)
    assert batched.shape == (3, 3)
    for index, pose in enumerate(poses):
        assert batched[index] == pytest.approx(chain.end_effector_position(pose))


def test_a_wrong_number_of_joint_angles_is_rejected() -> None:
    """Silently broadcasting a short angle vector would place the hand somewhere arbitrary.

    Purpose: Validates the width check on the FK input

    Given: A seven-joint chain and a five-angle vector
    When: Forward kinematics is requested
    Then: ValueError names both widths

    Test type: unit
    """
    with pytest.raises(ValueError, match="expected 7 joint angles"):
        franka_panda_chain().end_effector_position(np.zeros(5))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"link_offsets": np.zeros(3)}, "equal length"),
        ({"tool_transform": np.eye(3)}, "4x4"),
    ],
)
def test_an_inconsistent_chain_is_rejected(overrides: dict, message: str) -> None:
    """A mismatched DH table would index past its own parameters or compose the wrong transform.

    Purpose: Validates the construction-time guards on ModifiedDHChain

    Given: DH arrays of unequal length, or a tool transform that is not 4x4
    When: The chain is constructed
    Then: ValueError is raised

    Test type: unit
    """
    settings: dict = {
        "link_lengths": np.zeros(JOINTS),
        "link_offsets": np.zeros(JOINTS),
        "link_twists": np.zeros(JOINTS),
        "tool_transform": np.eye(4),
    }
    settings.update(overrides)
    with pytest.raises(ValueError, match=message):
        ModifiedDHChain(**settings)


# ── Joint-lag transition ────────────────────────────────────────────────


def test_joints_close_a_fixed_fraction_of_their_target_error() -> None:
    """A position controller lags; a model that teleported to the target would over-promise.

    Purpose: Validates the first-order lag against a hand-computed step

    Given: A joint at 0.2 rad commanded to 1.0 rad with a gain of 0.5
    When: One control step is taken
    Then: The joint sits at 0.6 rad, half way to its target

    Test type: unit
    """
    transition = _exact_transition(position_width=1)
    state = np.array([0.2, 0.0, 0.0])
    assert float(transition.sample_next_state(state, [1.0])[0]) == pytest.approx(0.6, abs=1e-6)


def test_the_reported_velocity_is_the_step_the_joint_actually_took() -> None:
    """The velocity channel is observed, so a velocity inconsistent with the position misweights.

    Purpose: Validates that the velocity block is the lag's own difference quotient

    Given: A transition with a 0.1 s control step and a joint that moves 0.5 rad
    When: One control step is taken
    Then: The velocity block reads 5 rad/s

    Test type: unit
    """
    transition = _exact_transition(position_width=1)
    successor = transition.sample_next_state(np.zeros(3), [1.0])
    assert float(successor[1]) == pytest.approx(5.0, abs=1e-5)


def test_the_applied_action_is_recorded_in_the_state() -> None:
    """The task's own observation carries the last action, so the model has to reproduce it.

    Purpose: Validates the recorded-action block of the transition

    Given: A transition and a distinctive command
    When: One control step is taken
    Then: The trailing block equals the command that was applied

    Test type: unit
    """
    command = np.linspace(-0.5, 0.5, JOINTS)
    successor = _exact_transition().sample_next_state(np.zeros(3 * JOINTS), command)
    assert successor[2 * JOINTS :] == pytest.approx(command, abs=1e-6)


def test_two_different_commands_move_the_joints_to_different_places() -> None:
    """This is the failure the analytic model exists to fix: a flat model hides the actions.

    Purpose: Validates that the transition separates distinct actions

    Given: One starting configuration and two different commands
    When: A step is taken under each
    Then: The resulting joint positions differ by an amount a planner can see

    Test type: unit
    """
    transition = _exact_transition()
    start = np.zeros(3 * JOINTS)
    left = transition.sample_next_state(start, np.full(JOINTS, 0.5))
    right = transition.sample_next_state(start, np.full(JOINTS, -0.5))
    assert float(np.linalg.norm(left[:JOINTS] - right[:JOINTS])) > 0.1


def test_joints_the_action_does_not_command_hold_their_position() -> None:
    """The Franka reach task observes nine joints and commands seven; the gripper is not driven.

    Purpose: Validates that unactuated joints are predicted to stay where they are

    Given: A nine-joint block whose last two joints no action entry commands
    When: A full-scale command is applied
    Then: The seven commanded joints move and the two uncommanded ones do not

    Test type: unit
    """
    transition = _exact_transition(position_width=9, actuated_indices=range(JOINTS))
    state = np.zeros(2 * 9 + JOINTS)
    state[7:9] = 0.04  # the gripper fingers sit open
    successor = transition.sample_next_state(state, np.ones(JOINTS))
    assert np.all(np.abs(successor[:JOINTS]) > 0.4)
    assert successor[7:9] == pytest.approx([0.04, 0.04], abs=1e-6)
    assert successor[9 + 7 : 9 + 9] == pytest.approx([0.0, 0.0], abs=1e-6)


@pytest.mark.parametrize(
    "indices, message",
    [([0, 0], "same joint twice"), ([0, 9], "must index a 9-wide"), ([], "at least one joint")],
)
def test_an_invalid_actuated_index_map_is_rejected(indices: list, message: str) -> None:
    """A repeated or out-of-range index would drive the wrong joint, silently, for a whole study.

    Purpose: Validates the index-map guards on the joint-lag transition

    Given: An index map that repeats a joint, points past the block, or is empty
    When: The transition is constructed
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match=message):
        _exact_transition(position_width=9, actuated_indices=indices)


def test_a_batch_of_successors_is_dispersed_around_the_lag_prediction() -> None:
    """Progressive widening needs distinct successors, not one repeated draw.

    Purpose: Validates batched sampling of the joint-lag transition

    Given: A transition with appreciable position noise
    When: Two thousand successors are drawn in one call
    Then: They differ from each other and average to the noise-free prediction

    Test type: unit
    """
    np.random.seed(0)
    transition = JointLagTransition(
        position_width=1, step_dt=0.1, tracking_gain=0.5, position_noise_std=0.02
    )
    batch = transition.sample_next_state(np.zeros(3), [1.0], n_samples=2000)
    assert batch.shape == (2000, 3)
    assert batch[:, 0].mean() == pytest.approx(0.5, abs=0.01)
    assert not np.allclose(batch[0], batch[1])


def test_density_peaks_at_the_predicted_configuration() -> None:
    """A transition density that did not peak on its own prediction would misweight particles.

    Purpose: Validates the joint-lag transition's log-density

    Given: A noise-carrying transition and the configuration it predicts
    When: The prediction and a displaced configuration are scored
    Then: The prediction scores higher

    Test type: unit
    """
    transition = JointLagTransition(position_width=JOINTS, step_dt=0.1, tracking_gain=0.5)
    command = np.full(JOINTS, 0.5)
    predicted = _exact_transition().sample_next_state(np.zeros(3 * JOINTS), command)
    displaced = predicted.copy()
    displaced[:JOINTS] += 0.3
    scores = transition.log_probability(
        np.zeros(3 * JOINTS), command, np.stack([predicted, displaced])
    )
    assert scores[0] > scores[1]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"position_width": 0}, "position_width"),
        ({"step_dt": 0.0}, "step_dt"),
        ({"tracking_gain": 0.0}, "tracking_gain"),
        ({"tracking_gain": 1.5}, "tracking_gain"),
        ({"position_noise_std": 0.0}, "strictly positive"),
    ],
)
def test_invalid_transition_configuration_is_rejected(overrides: dict, message: str) -> None:
    """A zero step, a zero gain or a zero noise each makes the model degenerate rather than tight.

    Purpose: Validates the construction-time guards on the joint-lag transition

    Given: An out-of-range joint count, step, gain or noise std
    When: The transition is constructed
    Then: ValueError is raised naming the offending parameter

    Test type: unit
    """
    with pytest.raises(ValueError, match=message):
        _exact_transition(**overrides)


# ── Gain calibration ────────────────────────────────────────────────────


def test_the_lag_gain_is_recovered_from_a_rollout_that_used_it() -> None:
    """The gain has to be measured, not assumed, or the model describes a different robot.

    Purpose: Validates that calibrate_tracking_gain inverts the lag it was generated from

    Given: A synthetic rollout produced by a lag of exactly 0.35
    When: The gain is calibrated from the recorded positions and commands
    Then: 0.35 is recovered

    Test type: unit
    """
    rng = np.random.default_rng(0)
    positions = rng.uniform(-1.0, 1.0, size=(200, JOINTS))
    actions = rng.uniform(-1.0, 1.0, size=(200, JOINTS))
    following = positions + 0.35 * (2.0 * actions - positions)
    gain = calibrate_tracking_gain(positions, following, actions, action_scale=2.0)
    assert gain == pytest.approx(0.35, abs=1e-9)


def test_calibration_refuses_a_rollout_that_never_commands_a_move() -> None:
    """With no commanded error there is nothing to divide by, and a silent zero would be worse.

    Purpose: Validates the identifiability guard in calibrate_tracking_gain

    Given: A rollout whose commands always match the current positions exactly
    When: The gain is calibrated
    Then: ValueError explains that the gain is unidentifiable

    Test type: unit
    """
    positions = np.zeros((5, JOINTS))
    with pytest.raises(ValueError, match="unidentifiable"):
        calibrate_tracking_gain(positions, positions, positions, action_scale=1.0)


def test_calibration_rejects_mismatched_rollout_arrays() -> None:
    """Three arrays of different lengths mean the rollout was assembled wrongly.

    Purpose: Validates the shape guard in calibrate_tracking_gain

    Given: Position and action arrays of different lengths
    When: The gain is calibrated
    Then: ValueError reports the shapes

    Test type: unit
    """
    with pytest.raises(ValueError, match="must share a shape"):
        calibrate_tracking_gain(
            np.zeros((4, JOINTS)), np.zeros((4, JOINTS)), np.zeros((3, JOINTS)), action_scale=1.0
        )


def test_a_gain_measured_above_one_is_clipped_to_one() -> None:
    """A first-order lag cannot overshoot within a step; an overshooting fit is measurement noise.

    Purpose: Validates the range clipping in calibrate_tracking_gain

    Given: A rollout whose joints overshoot their commanded targets
    When: The gain is calibrated
    Then: The reported gain is exactly 1.0

    Test type: unit
    """
    positions = np.zeros((10, JOINTS))
    actions = np.ones((10, JOINTS))
    overshooting = np.full((10, JOINTS), 1.8)
    assert calibrate_tracking_gain(positions, overshooting, actions, action_scale=1.0) == 1.0


# ── Reach reward ────────────────────────────────────────────────────────


def _reward_model(**overrides: Any) -> ReachRewardModel:
    settings: dict = {
        "state_schema": SCHEMA,
        "chain": franka_panda_chain(),
        "joint_position_channel": "joint_pos",
        "command_channel": "command",
        "default_joint_positions": np.zeros(JOINTS),
    }
    settings.update(overrides)
    return ReachRewardModel(**settings)


def test_the_reward_falls_off_monotonically_with_distance_to_the_goal() -> None:
    """The planner climbs this surface; a non-monotone one would reward moving away.

    Purpose: Validates that the analytic reach reward decreases as the hand moves from the goal

    Given: One arm configuration and three goals at increasing distance from its hand
    When: Each state is scored
    Then: The rewards decrease in the same order

    Test type: unit
    """
    reward_model = _reward_model()
    hand = franka_panda_chain().end_effector_position(HOME_POSE)
    goals = [hand, hand + np.array([0.1, 0.0, 0.0]), hand + np.array([0.4, 0.0, 0.0])]
    scores = [
        reward_model.reward(_state(HOME_POSE, np.concatenate([goal, np.zeros(4)])), None, None)
        for goal in goals
    ]
    assert scores[0] > scores[1] > scores[2]


def test_the_reported_distance_is_the_hand_to_goal_distance() -> None:
    """The success predicate measures this same distance, so the model must agree with it.

    Purpose: Validates ReachRewardModel.end_effector_distance against a hand-computed offset

    Given: An arm at the home pose and a goal 0.25 m along x from its hand
    When: The distance is queried
    Then: It reads 0.25 m

    Test type: unit
    """
    hand = franka_panda_chain().end_effector_position(HOME_POSE)
    goal = np.concatenate([hand + np.array([0.25, 0.0, 0.0]), np.zeros(4)])
    distance = _reward_model().end_effector_distance(_state(HOME_POSE, goal))
    assert distance == pytest.approx(0.25, abs=1e-6)


def test_the_default_pose_offset_is_added_before_the_kinematics() -> None:
    """IsaacLab observes joint positions relative to a default; FK needs absolute angles.

    Purpose: Validates that default_joint_positions is applied inside the reward

    Given: A zero relative configuration and the Isaac default pose as the offset
    When: The hand position is computed
    Then: It matches evaluating the chain at the default pose directly

    Test type: unit
    """
    reward_model = _reward_model(default_joint_positions=ISAAC_DEFAULT_POSE)
    expected = franka_panda_chain().end_effector_position(ISAAC_DEFAULT_POSE)
    goal = np.concatenate([expected, np.zeros(4)])
    assert reward_model.end_effector_distance(_state(np.zeros(JOINTS), goal)) == pytest.approx(
        0.0, abs=1e-9
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"state_schema": IsaacChannelSchema((("joint_pos", 3), ("command", 7)))}, "joints"),
        ({"command_position_width": 9}, "too narrow"),
        ({"default_joint_positions": np.zeros(3)}, "default_joint_positions"),
    ],
)
def test_a_misconfigured_reach_reward_is_rejected(overrides: dict, message: str) -> None:
    """Naming the wrong block does not raise by itself; it silently scores the wrong quantity.

    Purpose: Validates the construction-time guards on ReachRewardModel

    Given: A channel of the wrong width, a command too narrow, or a wrong-length default pose
    When: The reward model is constructed
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match=message):
        _reward_model(**overrides)


# ── The assembled model ─────────────────────────────────────────────────


def test_the_model_drives_the_arm_and_carries_the_command() -> None:
    """A commanded pose is task data; a transition that moved it would chase its own goal.

    Purpose: Validates that ManipulatorIsaacModel wires the driven and carried channels

    Given: A schema with three arm channels and a command channel
    When: A successor is sampled under a non-zero command
    Then: The arm channels moved and the command block is untouched

    Test type: unit
    """
    np.random.seed(0)
    model = _model()
    command = np.array([0.4, 0.1, 0.5, 1.0, 0.0, 0.0, 0.0])
    state = _state(np.zeros(JOINTS), command)
    successor = model.sample_next_state(state, model.get_actions()[1])
    assert np.any(np.abs(SCHEMA.block(successor, "joint_pos")) > 0.1)
    assert SCHEMA.block(successor, "command") == pytest.approx(command)


def test_the_model_scores_two_actions_differently() -> None:
    """The measured failure was one action index repeated for a whole episode; this is its guard.

    Purpose: Validates that the assembled model separates its action presets by reward

    Given: A state whose goal is reachable and three distinct action presets
    When: Each preset is applied and the resulting state scored
    Then: The three rewards are all different

    Test type: unit
    """
    np.random.seed(0)
    model = _model()
    state = _state(np.zeros(JOINTS), np.array([0.4, 0.1, 0.5, 1.0, 0.0, 0.0, 0.0]))
    rewards = [
        model.reward(state, action, model.sample_next_state(state, action))
        for action in model.get_actions()
    ]
    assert len(set(round(value, 6) for value in rewards)) == len(rewards)


def test_a_channel_that_is_not_as_wide_as_the_arm_is_rejected() -> None:
    """A narrower channel would silently plan for a different robot than the one being driven.

    Purpose: Validates the construction-time width check on the driven channels

    Given: A schema whose velocity channel is three wide for a seven-joint chain
    When: The model is constructed
    Then: ValueError names the channel and both widths

    Test type: unit
    """
    schema = IsaacChannelSchema(
        (("joint_pos", JOINTS), ("joint_vel", 3), ("command", 7), ("last_action", JOINTS))
    )
    with pytest.raises(ValueError, match="joint_vel"):
        _model(state_schema=schema)


def test_the_franka_reach_layout_wires_end_to_end() -> None:
    """This is the real task's shape; getting the index maps wrong reaches to the wrong place.

    Purpose: Validates the model on the Franka reach observation layout of 9 + 9 + 7 + 7

    Given: A schema with nine observed joints, a seven-wide command block and a seven-wide action
    When: The arm is stepped from its default pose toward a goal placed at its own hand
    Then: The state stays 32 wide, the gripper joints do not move, and the distance reads zero

    Test type: unit
    """
    np.random.seed(0)
    schema = IsaacChannelSchema(
        (("joint_pos", 9), ("joint_vel", 9), ("command", 7), ("last_action", JOINTS))
    )
    chain = franka_panda_chain()
    hand = chain.end_effector_position(ISAAC_DEFAULT_POSE)
    model = _model(
        state_schema=schema,
        arm_joint_indices=range(JOINTS),
        actuated_indices=range(JOINTS),
        observation_models=None,
    )
    state = schema.pack(
        {
            "joint_pos": np.concatenate([np.zeros(JOINTS), [0.04, 0.04]]),
            "joint_vel": np.zeros(9),
            "command": np.concatenate([hand, [1.0, 0.0, 0.0, 0.0]]),
            "last_action": np.zeros(JOINTS),
        }
    )
    assert state.shape == (32,)
    reach_reward = model.reach_reward
    assert reach_reward is not None
    assert reach_reward.end_effector_distance(state) == pytest.approx(0.0, abs=1e-9)
    successor = model.sample_next_state(state, model.get_actions()[1])
    assert schema.block(successor, "joint_pos")[7:9] == pytest.approx([0.04, 0.04], abs=1e-6)


def test_a_supplied_reward_model_replaces_the_analytic_one() -> None:
    """The class is a wiring convenience, not a lock-in; another task needs another objective.

    Purpose: Validates that an explicit reward model is used and reach_reward reports its absence

    Given: A model constructed with a reward model that is not a ReachRewardModel
    When: A transition is scored
    Then: The supplied model's value is returned and reach_reward is None

    Test type: unit
    """

    class _ConstantReward:
        def reward(self, state: Any, action: Any, next_state: Any) -> float:
            del state, action, next_state
            return 4.0

    model = _model(reward_model=_ConstantReward())
    state = _state(np.zeros(JOINTS), np.zeros(7))
    assert model.reward(state, model.get_actions()[0], state) == pytest.approx(4.0)
    assert model.reach_reward is None
