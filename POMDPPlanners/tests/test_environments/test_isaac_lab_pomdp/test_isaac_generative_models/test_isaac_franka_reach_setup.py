# SPDX-License-Identifier: MIT

"""Unit tests for calibrating the analytic manipulator model against a reach task.

The task is stubbed rather than launched. What these tests guard is not physics but *wiring*: an
index map read in the wrong order, a default pose taken for the whole articulation instead of the
arm, or a schema that silently disagrees with the observation width. Each of those produces a
model that runs perfectly and reaches to the wrong place, which no simulator run would flag.
"""

from typing import Any, Dict, List, Sequence

import numpy as np
import pytest

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
    FrankaReachLayout,
    build_franka_reach_model,
    calibrate_lag_noise,
    franka_reach_layout,
)

ARM_JOINTS = [f"panda_joint{index}" for index in range(1, 8)]
GRIPPER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
DEFAULT_ARM_POSE = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]
ACTION_SCALE = 0.5
STEP_DT = 1.0 / 30.0


class _StubRobot:
    """The one articulation attribute set this calibration reads."""

    def __init__(self, joint_names: Sequence[str], defaults: Sequence[float]) -> None:
        self.joint_names = list(joint_names)
        self.data = type("_Data", (), {"default_joint_pos": np.asarray([list(defaults)])})()

    def find_joints(self, pattern: str) -> Any:
        del pattern  # the stub always resolves the arm-joint regex
        ids = [self.joint_names.index(name) for name in ARM_JOINTS]
        return sorted(ids), ARM_JOINTS


def _stub_env(joint_names: Sequence[str], defaults: Sequence[float]) -> Any:
    """Build the smallest object graph ``franka_reach_layout`` walks."""
    robot = _StubRobot(joint_names, defaults)
    action_cfg = type("_ActionCfg", (), {"joint_names": "panda_joint.*", "scale": ACTION_SCALE})()
    cfg = type("_Cfg", (), {"actions": type("_Actions", (), {"arm_action": action_cfg})()})()
    scene: Dict[str, Any] = {"robot": robot}
    unwrapped = type(
        "_Unwrapped", (), {"scene": scene, "cfg": cfg, "step_dt": STEP_DT, "physics_dt": STEP_DT}
    )()
    return type("_Env", (), {"unwrapped": unwrapped})()


def _default_layout() -> FrankaReachLayout:
    return franka_reach_layout(
        _stub_env(ARM_JOINTS + GRIPPER_JOINTS, DEFAULT_ARM_POSE + [0.04, 0.04])
    )


def test_the_layout_reads_the_arm_out_of_a_wider_articulation() -> None:
    """The observation covers nine joints and the action seven; conflating them reaches wrongly.

    Purpose: Validates that the layout separates the observed joints from the commanded ones

    Given: An articulation of seven arm joints followed by two gripper joints
    When: The layout is read
    Then: Nine joints are reported, seven are commanded, and the default pose is the arm's

    Test type: unit
    """
    layout = _default_layout()
    assert layout.joint_width == 9
    assert layout.arm_indices == tuple(range(7))
    assert layout.actuated_indices == tuple(range(7))
    assert layout.default_arm_pose == pytest.approx(DEFAULT_ARM_POSE)
    assert layout.step_dt == pytest.approx(STEP_DT)
    assert layout.action_scale == pytest.approx(ACTION_SCALE)


def test_the_arm_is_found_by_name_not_by_position() -> None:
    """A differently ordered articulation is the failure this index map exists to survive.

    Purpose: Validates that arm indices follow the chain's joint order, not the articulation's

    Given: An articulation that lists the gripper joints first
    When: The layout is read
    Then: The arm indices point past the gripper and the default pose is still the arm's

    Test type: unit
    """
    names = GRIPPER_JOINTS + ARM_JOINTS
    layout = franka_reach_layout(_stub_env(names, [0.04, 0.04] + DEFAULT_ARM_POSE))
    assert layout.arm_indices == tuple(range(2, 9))
    assert layout.default_arm_pose == pytest.approx(DEFAULT_ARM_POSE)


def test_an_articulation_without_the_chain_joints_is_rejected() -> None:
    """Calibrating against the wrong robot would produce a confident, wholly wrong model.

    Purpose: Validates the joint-name guard in franka_reach_layout

    Given: An articulation carrying only gripper joints
    When: The layout is read
    Then: RuntimeError names the missing joints

    Test type: unit
    """
    with pytest.raises(RuntimeError, match="missing chain joints"):
        franka_reach_layout(_stub_env(GRIPPER_JOINTS, [0.04, 0.04]))


def test_the_schema_matches_the_policy_observation_group() -> None:
    """The reach observation is 9 + 9 + 7 + 7; a schema off by one misreads every block after it.

    Purpose: Validates the schema the layout implies

    Given: The Franka reach layout
    When: Its schema is built
    Then: It is 32 wide with the four named blocks in observation order

    Test type: unit
    """
    schema = _default_layout().state_schema()
    assert schema.total_dim == 32
    assert schema.names == ("joint_pos", "joint_vel", "command", "last_action")
    assert schema.width("last_action") == 7


def test_the_measured_noise_is_zero_on_a_rollout_the_lag_explains_exactly() -> None:
    """Noise measured on a perfectly explained rollout must collapse to the floor, not to a guess.

    Purpose: Validates calibrate_lag_noise against a synthetic rollout with no residual

    Given: A rollout generated by exactly the lag being measured
    When: The residual noise is measured
    Then: The position std sits at the floor and the velocity std is that floor over the step

    Test type: unit
    """
    layout = _default_layout()
    rng = np.random.default_rng(0)
    positions = rng.uniform(-0.5, 0.5, size=(50, 9))
    actions = rng.uniform(-1.0, 1.0, size=(50, 7))
    following = positions.copy()
    arm = list(layout.actuated_indices)
    following[:, arm] = positions[:, arm] + 0.3 * (
        actions * layout.action_scale - positions[:, arm]
    )
    position_std, velocity_std = calibrate_lag_noise(positions, following, actions, layout, 0.3)
    assert position_std == pytest.approx(1e-4)
    assert velocity_std == pytest.approx(1e-4 / STEP_DT)


def test_the_measured_noise_grows_with_the_residual() -> None:
    """The number has to be a measurement; a constant would make the belief a fiction.

    Purpose: Validates that calibrate_lag_noise tracks the size of the unexplained residual

    Given: Two rollouts differing only in the magnitude of the noise added to their successors
    When: The residual noise is measured on each
    Then: The noisier rollout yields the larger std, close to the noise it was given

    Test type: unit
    """
    layout = _default_layout()
    rng = np.random.default_rng(1)
    positions = rng.uniform(-0.5, 0.5, size=(4000, 9))
    actions = rng.uniform(-1.0, 1.0, size=(4000, 7))
    arm = list(layout.actuated_indices)
    exact = positions.copy()
    exact[:, arm] = positions[:, arm] + 0.3 * (actions * layout.action_scale - positions[:, arm])
    quiet = calibrate_lag_noise(
        positions, exact + rng.normal(0.0, 0.01, exact.shape), actions, layout, 0.3
    )[0]
    loud = calibrate_lag_noise(
        positions, exact + rng.normal(0.0, 0.05, exact.shape), actions, layout, 0.3
    )[0]
    assert quiet == pytest.approx(0.01, abs=2e-3)
    assert loud == pytest.approx(0.05, abs=5e-3)
    assert loud > quiet


def _rollout(gain: float, count: int = 400) -> List[np.ndarray]:
    """A warm-up rollout generated by a known lag, in the task's 32-wide observation space."""
    rng = np.random.default_rng(2)
    states = np.zeros((count, 32))
    states[:, :9] = rng.uniform(-0.4, 0.4, size=(count, 9))
    states[:, 18:25] = rng.uniform(-0.3, 0.5, size=(count, 7))
    actions = rng.uniform(-1.0, 1.0, size=(count, 7))
    next_states = states.copy()
    next_states[:, :7] = states[:, :7] + gain * (actions * ACTION_SCALE - states[:, :7])
    return [states, actions, next_states]


def test_the_assembled_model_recovers_the_gain_that_generated_its_rollout() -> None:
    """The gain is the one fitted quantity; if calibration does not find it nothing else matters.

    Purpose: Validates build_franka_reach_model end to end against a synthetic rollout

    Given: A 32-wide warm-up rollout generated by a lag of exactly 0.42
    When: The model is built from it
    Then: Its transition carries that gain, the task's step and scale, and the arm index maps

    Test type: unit
    """
    states, actions, next_states = _rollout(0.42)
    model = build_franka_reach_model(
        _stub_env(ARM_JOINTS + GRIPPER_JOINTS, DEFAULT_ARM_POSE + [0.04, 0.04]),
        (states, actions, next_states),
        [np.zeros(7), np.full(7, 0.3)],
        discount_factor=0.99,
    )
    lag = model.joint_transition
    assert lag.tracking_gain == pytest.approx(0.42, abs=1e-6)
    assert lag.step_dt == pytest.approx(STEP_DT)
    assert lag.action_scale == pytest.approx(ACTION_SCALE)
    assert lag.position_width == 9
    assert lag.action_dim == 7
    assert model.reach_reward is not None
    assert model.reach_reward.default_joint_positions == pytest.approx(DEFAULT_ARM_POSE)


def test_a_rollout_of_the_wrong_width_is_rejected() -> None:
    """A changed observation group silently reinterprets every block; it must stop the run.

    Purpose: Validates the width guard in build_franka_reach_model

    Given: A rollout 30 wide where the task's configuration implies 32
    When: The model is built
    Then: RuntimeError reports both widths

    Test type: unit
    """
    states, actions, next_states = _rollout(0.4)
    with pytest.raises(RuntimeError, match="observation group has changed"):
        build_franka_reach_model(
            _stub_env(ARM_JOINTS + GRIPPER_JOINTS, DEFAULT_ARM_POSE + [0.04, 0.04]),
            (states[:, :30], actions, next_states[:, :30]),
            [np.zeros(7)],
            discount_factor=0.99,
        )
