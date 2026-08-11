# SPDX-License-Identifier: MIT

"""Calibrate an analytic manipulator model against a live ``Isaac-Reach-Franka-v0`` task.

:class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_manipulator_model.ManipulatorIsaacModel`
is deliberately generic: it takes a control step, a lag gain, an action scale, a joint index map
and a default pose, and asks no questions about where they came from. This module is the answer
for one concrete task — it reads those numbers off the running task and measures the two that are
not written down anywhere.

The split matters for the honesty claim. Everything read here is **configuration**: the
articulation's joint names, its default joint pose, the action term's scale and the control-step
duration are fixed before the episode starts and would be equally available from a datasheet and a
tape measure on a real arm. The two *measured* numbers — the lag gain and its residual noise —
come from a warm-up rollout, which is exactly how you would characterise a real robot. Neither is
a reading of the live state at planning time, which is the thing a planner must never have.

Layout
------
The reach task's ``policy`` observation group concatenates ``joint_pos_rel``, ``joint_vel_rel``,
the ``ee_pose`` command and the last action. The two joint blocks cover the *whole* articulation
— nine joints on a Panda with its gripper — while the action commands only the seven arm joints,
so the joint blocks and the action block have different widths and the model needs an index map
between them. Discovering that map by name rather than assuming ``0..6`` is what stops a
differently ordered articulation from silently reaching to the wrong place.

Functions:
    franka_reach_layout: Read the task's fixed joint, timing and action-scale configuration.
    calibrate_lag_noise: Measure the process noise the calibrated lag leaves unexplained.
    build_franka_reach_model: Assemble the calibrated analytic model for the reach task.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_manipulator_model import (
    ManipulatorIsaacModel,
    calibrate_tracking_gain,
    franka_panda_chain,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_model_pomdp import (
    IsaacChannelSchema,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_helpers import first_row

#: Franka arm joints in the order the kinematic chain expects them.
FRANKA_ARM_JOINTS: Tuple[str, ...] = tuple(f"panda_joint{index}" for index in range(1, 8))

#: Width of the ``ee_pose`` command block: a position plus a ``(w, x, y, z)`` quaternion.
POSE_COMMAND_WIDTH = 7

#: Floor on a measured noise std, so a suspiciously clean rollout cannot produce a degenerate
#: density in which every particle but one scores ``-inf``.
MINIMUM_NOISE_STD = 1e-4


@dataclass(frozen=True)
class FrankaReachLayout:
    """The reach task's fixed configuration, as the analytic model needs it.

    Attributes:
        joint_names: Every joint of the articulation, in its own order.
        arm_indices: Which joints feed the kinematic chain, in the chain's order.
        actuated_indices: Which joint each action entry commands.
        default_arm_pose: The chain-ordered default joint pose, in radians.
        step_dt: Control-step duration, in seconds.
        action_scale: Scale mapping an action into a joint-position target.
    """

    joint_names: Tuple[str, ...]
    arm_indices: Tuple[int, ...]
    actuated_indices: Tuple[int, ...]
    default_arm_pose: np.ndarray
    step_dt: float
    action_scale: float

    @property
    def joint_width(self) -> int:
        """Number of joints the observation reports."""
        return len(self.joint_names)

    def state_schema(self) -> IsaacChannelSchema:
        """The schema the task's ``policy`` observation group implies."""
        return IsaacChannelSchema(
            (
                ("joint_pos", self.joint_width),
                ("joint_vel", self.joint_width),
                ("command", POSE_COMMAND_WIDTH),
                ("last_action", len(self.actuated_indices)),
            )
        )


def franka_reach_layout(
    env: Any, arm_joints: Sequence[str] = FRANKA_ARM_JOINTS
) -> FrankaReachLayout:
    """Read the reach task's fixed joint, timing and action-scale configuration.

    Args:
        env: The live IsaacLab task environment (``IsaacLabPOMDP.task_env``).
        arm_joints: Joint names in the kinematic chain's order.

    Returns:
        The layout the analytic model needs.

    Raises:
        RuntimeError: If the articulation is missing any of the chain's joints.
    """
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_pomdp import step_duration

    robot = env.unwrapped.scene["robot"]
    joint_names: List[str] = list(robot.joint_names)
    missing = [name for name in arm_joints if name not in joint_names]
    if missing:
        raise RuntimeError(
            f"the articulation is missing chain joints {missing}; it carries {joint_names}"
        )
    arm_indices = tuple(joint_names.index(name) for name in arm_joints)
    action_cfg = env.unwrapped.cfg.actions.arm_action
    defaults = first_row(robot.data.default_joint_pos)
    return FrankaReachLayout(
        joint_names=tuple(joint_names),
        arm_indices=arm_indices,
        actuated_indices=tuple(
            int(index) for index in robot.find_joints(action_cfg.joint_names)[0]
        ),
        default_arm_pose=defaults[list(arm_indices)],
        step_dt=float(step_duration(env)),
        action_scale=float(action_cfg.scale),
    )


def calibrate_lag_noise(
    joint_positions: ArrayLike,
    next_joint_positions: ArrayLike,
    actions: ArrayLike,
    layout: FrankaReachLayout,
    tracking_gain: float,
) -> Tuple[float, float]:
    """Measure the process noise the calibrated lag leaves unexplained, in rad and rad/s.

    A guessed process noise leaves the belief overconfident or diffuse for no stated reason. The
    lag's own residual on the warm-up rollout is the honest estimate, and it absorbs the
    observation corruption the task applies to its joint readings along with the dynamics the lag
    does not model.

    Args:
        joint_positions: ``(N, P)`` joint-position blocks from the rollout.
        next_joint_positions: ``(N, P)`` the blocks one control step later.
        actions: ``(N, J)`` the commands applied.
        layout: The task layout, for the action scale and control step.
        tracking_gain: The gain whose residual is being measured.

    Returns:
        The position and velocity noise stds.
    """
    current = np.atleast_2d(np.asarray(joint_positions, dtype=float))
    following = np.atleast_2d(np.asarray(next_joint_positions, dtype=float))
    commands = np.atleast_2d(np.asarray(actions, dtype=float))
    actuated = list(layout.actuated_indices)
    driven = current[:, actuated]
    predicted = current.copy()
    predicted[:, actuated] = driven + tracking_gain * (commands * layout.action_scale - driven)
    position_std = max(float(np.std(following - predicted)), MINIMUM_NOISE_STD)
    return position_std, max(position_std / layout.step_dt, MINIMUM_NOISE_STD)


def build_franka_reach_model(
    env: Any,
    rollout: Tuple[ArrayLike, ArrayLike, ArrayLike],
    action_presets: Sequence[ArrayLike],
    discount_factor: float,
    layout: Optional[FrankaReachLayout] = None,
) -> ManipulatorIsaacModel:
    """Assemble the calibrated analytic model for ``Isaac-Reach-Franka-v0``.

    Args:
        env: The live IsaacLab task environment, read for its fixed configuration.
        rollout: ``(states, actions, next_states)`` from a warm-up of arbitrary actions. Only the
            joint blocks and the commands are used; the rewards are not needed, because the
            objective is analytic rather than regressed.
        action_presets: The finite joint-target commands the planner chooses among.
        discount_factor: POMDP discount factor, shared with the world.
        layout: A layout already read from ``env``; read afresh when omitted.

    Returns:
        The scalar analytic model, ready to be wrapped for a vectorized planner.

    Raises:
        RuntimeError: If the task's observation width disagrees with the schema its configuration
            implies, which means the observation group has changed and the index maps are stale.
    """
    resolved = layout if layout is not None else franka_reach_layout(env)
    states = np.atleast_2d(np.asarray(rollout[0], dtype=float))
    actions = np.atleast_2d(np.asarray(rollout[1], dtype=float))
    next_states = np.atleast_2d(np.asarray(rollout[2], dtype=float))
    schema = resolved.state_schema()
    if schema.total_dim != states.shape[1]:
        raise RuntimeError(
            f"the task's observation is {states.shape[1]} wide but its configuration implies "
            f"{schema.channels} totalling {schema.total_dim}; the observation group has changed"
        )

    joints = schema.slice_of("joint_pos")
    actuated = list(resolved.actuated_indices)
    gain = calibrate_tracking_gain(
        states[:, joints][:, actuated],
        next_states[:, joints][:, actuated],
        actions,
        action_scale=resolved.action_scale,
    )
    position_std, velocity_std = calibrate_lag_noise(
        states[:, joints], next_states[:, joints], actions, resolved, gain
    )
    return ManipulatorIsaacModel(
        state_schema=schema,
        action_presets=action_presets,
        discount_factor=discount_factor,
        step_dt=resolved.step_dt,
        tracking_gain=gain,
        chain=franka_panda_chain(),
        default_joint_positions=resolved.default_arm_pose,
        arm_joint_indices=resolved.arm_indices,
        actuated_indices=actuated,
        action_scale=resolved.action_scale,
        position_noise_std=position_std,
        velocity_noise_std=velocity_std,
        name="franka_reach_analytic",
    )
