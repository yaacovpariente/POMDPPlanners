# SPDX-License-Identifier: MIT

"""Calibrate a goal-relative navigation model against a live ``Isaac-Navigation-Flat-Anymal-C-v0``.

:class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_navigation_model.NavigationIsaacModel`
is generic: it takes a control step, two tracking scales and three noise stds and asks no questions
about where they came from. This module is the answer for one concrete task.

The split matters for the honesty claim. The control step is **configuration** — it is fixed before
the episode starts and would be equally available from a datasheet. The four *measured* numbers —
the linear and angular tracking scales and the residual noise they leave — come from a warm-up
rollout, which is how you would characterise a real robot.

Nothing here reads a privileged quantity, not even during calibration. The task's policy
observation contains no base position and no yaw rate, so both scales are recovered from the
``pose_command`` block itself, inverting the model's own update:

* the heading entry moves by ``-dyaw``, so the achieved yaw rate is ``-wrap(h' - h) / dt``;
* the position entry then gives the achieved displacement as ``goal - R(dyaw) @ goal'``.

Both are functions of two consecutive observations and the command issued between them, which is
exactly what an operator driving a real robot already has.

Episode boundaries are excluded rather than fitted. IsaacLab auto-resets on timeout and the pose
command resamples with it, so the transition spanning a reset shows a metre-scale jump in a goal
that can physically move only centimetres. Those rows are dropped by a bound derived from the
commands actually issued, not by a tuned threshold.

Functions:
    navigation_state_schema: The schema the task's ``policy`` observation group implies.
    calibrate_command_tracking: Measure the achieved fraction of the linear and angular commands.
    calibrate_navigation_noise: Measure the process noise the calibrated tracking leaves.
    build_anymal_navigation_model: Assemble the calibrated model for the navigation task.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_model_pomdp import (
    IsaacChannelSchema,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_navigation_model import (
    BASE_VELOCITY_WIDTH,
    POSE_COMMAND_WIDTH,
    GoalRelativeTransition,
    NavigationIsaacModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_unicycle_model import (
    wrap_angle,
)

#: Width of the ``projected_gravity`` block the task reports between velocity and command.
GRAVITY_WIDTH = 3

#: Floor on a measured noise std, so a suspiciously clean rollout cannot produce a degenerate
#: density in which every particle but one scores ``-inf``.
MINIMUM_NOISE_STD = 1e-4

#: How far past the largest commanded displacement a transition may go before it is read as an
#: episode reset rather than a step. Two is generous for a tracker that cannot exceed its command
#: by much, and leaves a wide margin against a goal resampled metres away.
RESET_DETECTION_SLACK = 2.0

#: Bounds on a measured tracking scale. Zero would say the base ignores its command; well above one
#: would say it overshoots every command, which no velocity tracker does and which in practice
#: means the rollout was too short or too clean to identify.
TRACKING_SCALE_BOUNDS = (1e-3, 2.0)


@dataclass(frozen=True)
class AnymalNavigationLayout:
    """The navigation task's fixed configuration, as the goal-relative model needs it.

    Attributes:
        step_dt: Control-step duration, in seconds.
    """

    step_dt: float

    def state_schema(self) -> IsaacChannelSchema:
        """The schema the task's ``policy`` observation group implies."""
        return navigation_state_schema()


def navigation_state_schema() -> IsaacChannelSchema:
    """The schema of the navigation task's ``policy`` observation group.

    The group concatenates ``base_lin_vel``, ``projected_gravity`` and the ``pose_command``, in
    that order, and carries no base position at all — which is the whole reason the model is
    goal-relative.

    Returns:
        The three-channel, ten-wide schema.
    """
    return IsaacChannelSchema(
        (
            ("base_lin_vel", BASE_VELOCITY_WIDTH),
            ("projected_gravity", GRAVITY_WIDTH),
            ("pose_command", POSE_COMMAND_WIDTH),
        )
    )


def anymal_navigation_layout(env: Any) -> AnymalNavigationLayout:
    """Read the navigation task's fixed timing configuration.

    Args:
        env: The live IsaacLab task environment (``IsaacLabPOMDP.task_env``).

    Returns:
        The layout the goal-relative model needs.
    """
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_pomdp import step_duration

    return AnymalNavigationLayout(step_dt=float(step_duration(env)))


def _measured_yaw_delta(goals: np.ndarray, next_goals: np.ndarray) -> np.ndarray:
    """Achieved yaw change per step, read off the heading entry of the base-frame command."""
    return -wrap_angle(next_goals[:, 3] - goals[:, 3])


def _measured_displacement(
    goals: np.ndarray, next_goals: np.ndarray, yaw_delta: np.ndarray
) -> np.ndarray:
    """Achieved planar displacement per step, in the frame the step started in.

    Inverts ``goal' = R(-dyaw) @ (goal - d)`` for ``d``, using the *measured* yaw change so the
    two calibrations do not lean on each other.
    """
    cos_d, sin_d = np.cos(yaw_delta), np.sin(yaw_delta)
    rotated_x = cos_d * next_goals[:, 0] - sin_d * next_goals[:, 1]
    rotated_y = sin_d * next_goals[:, 0] + cos_d * next_goals[:, 1]
    return goals[:, :2] - np.stack([rotated_x, rotated_y], axis=-1)


def _stepwise_rows(
    displacement: np.ndarray, yaw_delta: np.ndarray, commands: np.ndarray, step_dt: float
) -> np.ndarray:
    """Mask of transitions small enough to be one control step rather than an episode reset.

    Selecting on the measured displacement censors the very quantity the fit then regresses, which
    would bias the estimate if the bound cut anywhere near the data. It does not: the bound is
    twice the *commanded* displacement while a base that tracks a fraction of its command moves
    well under it, so on this task the retained steps sit roughly an order of magnitude inside the
    bound and the resamples sit metres outside. The bound separates two clusters rather than
    trimming one.
    """
    linear_bound = (
        RESET_DETECTION_SLACK
        * float(np.max(np.linalg.norm(commands[:, :2], axis=-1), initial=0.0))
        * step_dt
    )
    angular_bound = RESET_DETECTION_SLACK * float(np.max(np.abs(commands[:, 2]), initial=0.0))
    angular_bound *= step_dt
    return (np.linalg.norm(displacement, axis=-1) <= linear_bound) & (
        np.abs(yaw_delta) <= angular_bound
    )


def _least_squares_scale(achieved: np.ndarray, commanded: np.ndarray, label: str) -> float:
    """The single scalar best explaining ``achieved`` as a multiple of ``commanded``."""
    denominator = float(np.sum(commanded * commanded))
    if denominator <= 0.0:
        raise ValueError(
            f"the rollout never issues a non-zero {label} command, so its tracking scale is "
            "unidentifiable; drive the base with non-trivial actions before calibrating"
        )
    scale = float(np.sum(achieved * commanded) / denominator)
    return float(np.clip(scale, *TRACKING_SCALE_BOUNDS))


def calibrate_command_tracking(
    states: ArrayLike,
    actions: ArrayLike,
    next_states: ArrayLike,
    step_dt: float,
    schema: Optional[IsaacChannelSchema] = None,
) -> Tuple[float, float]:
    """Measure the fraction of the commanded velocity the low-level policy achieves.

    Assuming perfect tracking is the most likely way a goal-relative model quietly fails: the base
    lags its command, the model believes it does not, and the predicted goal drifts a little
    further from the observed one every step. Two scales are measured rather than one because a
    legged base does not track a turn and a translation alike.

    Args:
        states: ``(N, 10)`` policy observations from a warm-up rollout.
        actions: ``(N, 3)`` velocity commands applied at each step.
        next_states: ``(N, 10)`` the observations one control step later.
        step_dt: Control-step duration, in seconds.
        schema: The state schema; the task's own when omitted.

    Returns:
        The ``(linear_scale, angular_scale)`` pair, each clipped into
        :data:`TRACKING_SCALE_BOUNDS`.

    Raises:
        ValueError: If the rollout leaves no usable transition, or never commands a move.
    """
    resolved = schema if schema is not None else navigation_state_schema()
    goals = np.atleast_2d(np.asarray(states, dtype=float))[:, resolved.slice_of("pose_command")]
    next_goals = np.atleast_2d(np.asarray(next_states, dtype=float))[
        :, resolved.slice_of("pose_command")
    ]
    commands = np.atleast_2d(np.asarray(actions, dtype=float))
    yaw_delta = _measured_yaw_delta(goals, next_goals)
    displacement = _measured_displacement(goals, next_goals, yaw_delta)
    keep = _stepwise_rows(displacement, yaw_delta, commands, step_dt)
    if not np.any(keep):
        raise ValueError(
            "every transition in the rollout looks like an episode reset rather than a control "
            "step; the pose command jumps further than the issued velocity commands could move "
            "the base, so the tracking scales cannot be identified"
        )
    linear = _least_squares_scale(
        displacement[keep].reshape(-1),
        (commands[keep][:, :2] * step_dt).reshape(-1),
        "linear",
    )
    angular = _least_squares_scale(yaw_delta[keep], commands[keep][:, 2] * step_dt, "angular")
    return linear, angular


def calibrate_navigation_noise(
    states: ArrayLike,
    actions: ArrayLike,
    next_states: ArrayLike,
    step_dt: float,
    scales: Tuple[float, float],
    schema: Optional[IsaacChannelSchema] = None,
) -> Tuple[float, float, float]:
    """Measure the process noise the calibrated tracking leaves unexplained.

    A guessed process noise leaves the belief overconfident or diffuse for no stated reason. The
    calibrated model's own residual on the warm-up rollout is the honest estimate, and it absorbs
    everything the composition of two rigid motions does not capture — a leg slipping, the base
    pitching, the low-level policy taking a step to respond to a reversal.

    Args:
        states: ``(N, 10)`` policy observations from a warm-up rollout.
        actions: ``(N, 3)`` velocity commands applied at each step.
        next_states: ``(N, 10)`` the observations one control step later.
        step_dt: Control-step duration, in seconds.
        scales: The ``(linear_scale, angular_scale)`` pair whose residual is being measured.
        schema: The state schema; the task's own when omitted.

    Returns:
        The ``(velocity_std, position_std, heading_std)`` triple, in m/s, metres and radians.

    Raises:
        ValueError: If the rollout leaves no usable transition.
    """
    resolved = schema if schema is not None else navigation_state_schema()
    model = _calibrated_transition(step_dt, scales)
    current = np.atleast_2d(np.asarray(states, dtype=float))
    following = np.atleast_2d(np.asarray(next_states, dtype=float))
    commands = np.atleast_2d(np.asarray(actions, dtype=float))
    driven = resolved.indices_of(("base_lin_vel", "pose_command"))
    goals = current[:, resolved.slice_of("pose_command")]
    next_goals = following[:, resolved.slice_of("pose_command")]
    yaw_delta = _measured_yaw_delta(goals, next_goals)
    keep = _stepwise_rows(
        _measured_displacement(goals, next_goals, yaw_delta), yaw_delta, commands, step_dt
    )
    if not np.any(keep):
        raise ValueError("the rollout leaves no control-step transition to measure noise on")
    predicted = model.predict_next(current[keep][:, driven], commands[keep])
    residual = following[keep][:, driven] - predicted
    residual[:, -1] = wrap_angle(residual[:, -1])
    velocity_std = float(np.std(residual[:, :BASE_VELOCITY_WIDTH]))
    position_std = float(np.std(residual[:, BASE_VELOCITY_WIDTH:-1]))
    heading_std = float(np.std(residual[:, -1]))
    return (
        max(velocity_std, MINIMUM_NOISE_STD),
        max(position_std, MINIMUM_NOISE_STD),
        max(heading_std, MINIMUM_NOISE_STD),
    )


def _calibrated_transition(step_dt: float, scales: Tuple[float, float]) -> GoalRelativeTransition:
    """A goal-relative transition at the measured scales, used only for its mean prediction."""
    return GoalRelativeTransition(
        step_dt=step_dt,
        linear_scale=scales[0],
        angular_scale=scales[1],
        velocity_noise_std=MINIMUM_NOISE_STD,
        position_noise_std=MINIMUM_NOISE_STD,
        heading_noise_std=MINIMUM_NOISE_STD,
    )


def build_anymal_navigation_model(
    env: Any,
    rollout: Tuple[ArrayLike, ArrayLike, ArrayLike],
    action_presets: Sequence[ArrayLike],
    discount_factor: float,
    layout: Optional[AnymalNavigationLayout] = None,
) -> NavigationIsaacModel:
    """Assemble the calibrated goal-relative model for ``Isaac-Navigation-Flat-Anymal-C-v0``.

    Args:
        env: The live IsaacLab task environment, read for its control-step duration.
        rollout: ``(states, actions, next_states)`` from a warm-up of arbitrary actions. The
            rewards are not needed, because the objective is the task's own rather than regressed.
        action_presets: The finite velocity commands the planner chooses among.
        discount_factor: POMDP discount factor, shared with the world.
        layout: A layout already read from ``env``; read afresh when omitted.

    Returns:
        The scalar goal-relative model, ready to be wrapped for a vectorized planner.

    Raises:
        RuntimeError: If the task's observation width disagrees with the schema its observation
            group implies, which means the group has changed and the channel offsets are stale.
    """
    resolved = layout if layout is not None else anymal_navigation_layout(env)
    states = np.atleast_2d(np.asarray(rollout[0], dtype=float))
    actions = np.atleast_2d(np.asarray(rollout[1], dtype=float))
    next_states = np.atleast_2d(np.asarray(rollout[2], dtype=float))
    schema = resolved.state_schema()
    if schema.total_dim != states.shape[1]:
        raise RuntimeError(
            f"the task's observation is {states.shape[1]} wide but its observation group implies "
            f"{schema.channels} totalling {schema.total_dim}; the group has changed"
        )

    scales = calibrate_command_tracking(states, actions, next_states, resolved.step_dt, schema)
    velocity_std, position_std, heading_std = calibrate_navigation_noise(
        states, actions, next_states, resolved.step_dt, scales, schema
    )
    return NavigationIsaacModel(
        state_schema=schema,
        action_presets=action_presets,
        discount_factor=discount_factor,
        step_dt=resolved.step_dt,
        linear_scale=scales[0],
        angular_scale=scales[1],
        velocity_noise_std=velocity_std,
        position_noise_std=position_std,
        heading_noise_std=heading_std,
        name="anymal_navigation_goal_relative",
    )
