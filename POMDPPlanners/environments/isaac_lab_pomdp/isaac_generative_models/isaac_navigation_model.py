# SPDX-License-Identifier: MIT

"""Analytic goal-relative navigation model for IsaacLab pose-command tasks.

``Isaac-Navigation-Flat-Anymal-C-v0`` commands a base velocity ``(v_x, v_y, omega_z)`` that rides
a pre-trained locomotion policy, so its motion is the planar unicycle
:mod:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_unicycle_model`
already writes down. What it does *not* give the planner is a place to put that motion: the task's
policy observation is ``base_lin_vel(3) + projected_gravity(3) + pose_command(4)``, and there is no
base position anywhere in it. The robot does not know where it is on the floor, and a model that
tracks ``(x, y, yaw)`` is therefore modelling a quantity nothing ever observes.

The goal is base-relative, so propagate the goal
------------------------------------------------
``pose_command`` is the goal *expressed in the base frame* — the world-frame goal minus the base
position, rotated into the base's yaw — plus the heading error. That quantity does have a
closed-form update, and it is the only one the planner needs::

    goal_b'     = R(-dyaw) @ (goal_b - d)
    heading_b'  = wrap(heading_b - dyaw)

where ``(d, dyaw)`` is the base's own displacement over the step. Advancing moves the goal toward
you; turning left rotates it right. The world-frame goal cancels out of both lines, which is what
makes this computable with no position at all.

``(d, dyaw)`` is exactly :meth:`~...isaac_unicycle_model.UnicycleTransition.body_frame_delta`, so
the dynamics here are the unicycle's, composed as an SE(2) inverse rather than re-derived. Only the
meaning of the state changed.

Five honest approximations, stated so a study can check them rather than discover them:

* **Tracking is imperfect.** The low-level policy achieves some fraction of the commanded velocity.
  ``linear_scale`` and ``angular_scale`` are separate because a legged base tracks a turn and a
  translation differently; both are measurable from a rollout of the *observation alone* (see
  :mod:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_anymal_navigation_setup`).
  Assuming 1.0 is the most likely way this model quietly fails.
* **Translation and rotation are composed, not integrated jointly.** The displacement is taken in
  the frame the step *starts* in, matching the unicycle's own Euler step. Over a 0.2 s control step
  at 1 m/s and 1 rad/s the chord-versus-arc error is under a centimetre.
* **The base stays level.** ``projected_gravity`` is carried through the transition unchanged and
  the goal's z entry with it, which is right on flat terrain and wrong on a slope.
* **Termination is the world's.** A fall is not modelled; it enters as the process noise the
  tracking is wrapped in.
* **The heading density ignores the wrap.** Residuals are wrapped into ``(-pi, pi]`` before they
  are scored, which is right for an angle, but the result is then a wrapped Gaussian truncated to
  one revolution rather than the full wrapped normal — the same approximation
  :mod:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_unicycle_model`
  makes. With a measured per-step heading noise around 0.04 rad the omitted mass is far below
  floating-point resolution; it would matter only at a noise approaching a radian.

Classes:
    GoalRelativeTransition: Integrates a base-frame goal forward under a velocity command.
    NavigationRewardModel: The navigation task's own pose-tracking objective.
    NavigationIsaacModel: Factored model wired with goal-relative dynamics and that objective.
"""

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_factored_model import (
    FactoredIsaacModelPOMDP,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_model_pomdp import (
    IsaacChannelSchema,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_unicycle_model import (
    UnicycleTransition,
    wrap_angle,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp import (
    RewardModel,
    TransitionModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_model import (
    IsaacObservationModel,
)

#: Width of the ``pose_command`` block: a base-frame position ``(x, y, z)`` plus a heading error.
POSE_COMMAND_WIDTH = 4

#: Width of the ``base_lin_vel`` block: body-frame linear velocity ``(v_x, v_y, v_z)``.
BASE_VELOCITY_WIDTH = 3

#: Width of the action: a body-frame velocity command ``(v_x, v_y, omega_z)``.
VELOCITY_COMMAND_WIDTH = 3


class GoalRelativeTransition(TransitionModel):
    """Integrates a base-frame goal and the tracked base velocity under a velocity command.

    The driven state block is ``[base_lin_vel (3), pose_command (4)]``, in that order; the action
    is ``(v_x, v_y, omega_z)`` in the body frame. One control step moves the base by the scaled
    command and moves the goal the opposite way, in the base's own frame.

    Attributes:
        dim: Width of the driven block, ``BASE_VELOCITY_WIDTH + POSE_COMMAND_WIDTH``.
        step_dt: Control-step duration in seconds.
        linear_scale: Fraction of the commanded linear velocity the low-level policy achieves.
        angular_scale: Fraction of the commanded yaw rate the low-level policy achieves.

    Example:
        >>> import numpy as np
        >>> transition = GoalRelativeTransition(step_dt=0.2, position_noise_std=1e-9,
        ...                                     heading_noise_std=1e-9, velocity_noise_std=1e-9)
        >>> state = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])  # goal 2 m straight ahead
        >>> ahead = transition.sample_next_state(state, np.array([1.0, 0.0, 0.0]))
        >>> round(float(ahead[3]), 3)  # driving forward closes the gap by v * dt
        1.8
        >>> turned = transition.sample_next_state(state, np.array([0.0, 0.0, 1.0]))
        >>> round(float(turned[4]), 3)  # turning left swings the goal to the robot's right
        -0.397
    """

    def __init__(
        self,
        step_dt: float,
        linear_scale: float = 1.0,
        angular_scale: float = 1.0,
        velocity_noise_std: float = 0.1,
        position_noise_std: float = 0.05,
        heading_noise_std: float = 0.05,
    ) -> None:
        """Initialize the goal-relative transition.

        Args:
            step_dt: Control-step duration in seconds. Read it from the live task
                (``env.unwrapped.step_dt``) rather than guessing — every predicted displacement is
                linear in it.
            linear_scale: Fraction of the commanded linear velocity actually achieved. Measure it
                from a rollout; see
                :func:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_anymal_navigation_setup.calibrate_command_tracking`.
            angular_scale: Fraction of the commanded yaw rate actually achieved.
            velocity_noise_std: Std of the noise on the tracked ``base_lin_vel`` block, in m/s.
            position_noise_std: Std of the noise on the base-frame goal position, in metres.
            heading_noise_std: Std of the noise on the heading error, in radians.

        Raises:
            ValueError: If ``step_dt`` is not positive, a tracking scale is not positive, or any
                noise std is not strictly positive.
        """
        if step_dt <= 0.0:
            raise ValueError(f"step_dt must be positive, got {step_dt}")
        if linear_scale <= 0.0 or angular_scale <= 0.0:
            raise ValueError(
                "tracking scales must be positive; a non-positive scale says the base moves "
                f"against its command, got linear={linear_scale}, angular={angular_scale}"
            )
        stds = np.array([velocity_noise_std, position_noise_std, heading_noise_std], dtype=float)
        if np.any(stds <= 0.0):
            raise ValueError("process noise standard deviations must be strictly positive")
        self.step_dt = float(step_dt)
        self.linear_scale = float(linear_scale)
        self.angular_scale = float(angular_scale)
        self.dim = BASE_VELOCITY_WIDTH + POSE_COMMAND_WIDTH
        # The inner unicycle is used for its mean displacement only, so its own process noise is
        # never drawn; this transition owns the noise, split per block.
        self._unicycle = UnicycleTransition(step_dt=step_dt)
        self._command_scale = np.array(
            [self.linear_scale, self.linear_scale, self.angular_scale], dtype=float
        )
        self._std = np.concatenate(
            [
                np.full(BASE_VELOCITY_WIDTH, stds[0]),
                np.full(POSE_COMMAND_WIDTH - 1, stds[1]),
                np.full(1, stds[2]),
            ]
        )

    @property
    def process_noise_std(self) -> np.ndarray:
        """Per-entry process-noise std over the driven block, ``(dim,)``."""
        return self._std.copy()

    def body_step(self, action: Any) -> np.ndarray:
        """The base's own ``(dx, dy, dyaw)`` over one step under ``action``, tracking included.

        The goal update is this step applied backwards, so anything that needs to reproduce the
        dynamics — a batched torch kernel, a plot of the reachable set — should read it here rather
        than re-multiply the scales itself.

        Args:
            action: The ``(v_x, v_y, omega_z)`` velocity command.

        Returns:
            The scaled ``(dx, dy, dyaw)`` displacement in the frame the step starts in.
        """
        command = np.asarray(action, dtype=float).reshape(-1)[:VELOCITY_COMMAND_WIDTH]
        return self._unicycle.body_frame_delta(command * self._command_scale)

    def predict_next(self, states: Any, actions: Any) -> np.ndarray:
        """Noise-free prediction of the driven block for a batch of states and commands.

        This is the model's mean, exposed because calibration needs to compare it against what the
        robot actually did — measuring the residual of a *sampled* prediction would fold the
        assumed noise into the measured noise.

        Args:
            states: A ``(dim,)`` driven block or an ``(N, dim)`` batch of them.
            actions: The matching ``(3,)`` velocity command or ``(N, 3)`` batch.

        Returns:
            An ``(N, dim)`` batch of predicted driven blocks.
        """
        block = np.atleast_2d(np.asarray(states, dtype=float))[:, : self.dim]
        commands = np.atleast_2d(np.asarray(actions, dtype=float))[:, :VELOCITY_COMMAND_WIDTH]
        goal = block[:, BASE_VELOCITY_WIDTH:]
        deltas = np.asarray([self.body_step(row) for row in commands])
        cos_d, sin_d = np.cos(deltas[:, 2]), np.sin(deltas[:, 2])
        relative = goal[:, :2] - deltas[:, :2]
        return np.stack(
            [
                commands[:, 0] * self.linear_scale,
                commands[:, 1] * self.linear_scale,
                np.zeros(commands.shape[0]),
                cos_d * relative[:, 0] + sin_d * relative[:, 1],
                -sin_d * relative[:, 0] + cos_d * relative[:, 1],
                goal[:, 2],
                wrap_angle(goal[:, 3] - deltas[:, 2]),
            ],
            axis=-1,
        )

    def _mean(self, state: Any, action: Any) -> np.ndarray:
        return self.predict_next(np.asarray(state, dtype=float).reshape(-1), action)[0]

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        mean = self._mean(state, action)
        samples = mean[np.newaxis, :] + np.random.normal(0.0, self._std, size=(n_samples, self.dim))
        samples[:, -1] = wrap_angle(samples[:, -1])
        return samples[0] if n_samples == 1 else samples

    def log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        mean = self._mean(state, action)
        candidates = np.atleast_2d(np.asarray(next_states, dtype=float))[:, : self.dim]
        residual = candidates - mean[np.newaxis, :]
        # The heading entry is an angle: an error of 2*pi is no error at all, and scoring the raw
        # difference would call a wrapped candidate impossible.
        residual[:, -1] = wrap_angle(residual[:, -1])
        normalizer = float(np.sum(np.log(self._std)) + 0.5 * self.dim * np.log(2.0 * np.pi))
        return -0.5 * np.sum((residual / self._std[np.newaxis, :]) ** 2, axis=-1) - normalizer


class NavigationRewardModel(RewardModel):
    """The navigation task's own pose-tracking objective, read off the base-frame command.

    The reward is the task's two ``tanh`` position-tracking terms plus its heading penalty, all
    computed from the ``pose_command`` block alone. Two length scales is not decoration: the coarse
    term (``std = 2 m``) is the only one with useful gradient while the goal is still metres away,
    which is what lets a search of a few steps steer at all, and the fine term (``std = 0.2 m``) is
    what makes it stop on the goal rather than near it.

    The task's ``-400 * is_terminated`` fall penalty is omitted, because this model's terminal test
    is constantly false — the world owns termination — so the term would contribute a constant zero
    and only misstate the reward's range.

    Attributes:
        command_channel: The block holding ``(x, y, z, heading)`` in the base frame.
        coarse_weight: Weight on the wide ``tanh`` tracking term.
        coarse_std: Length scale of the wide term, in metres.
        fine_weight: Weight on the narrow ``tanh`` tracking term.
        fine_std: Length scale of the narrow term, in metres.
        heading_weight: Weight on the absolute heading error (positive penalises error).

    Example:
        >>> import numpy as np
        >>> schema = IsaacChannelSchema((("base_lin_vel", 3), ("pose_command", 4)))
        >>> reward_model = NavigationRewardModel(state_schema=schema)
        >>> near = schema.pack({"base_lin_vel": np.zeros(3), "pose_command": [0.1, 0.0, 0.0, 0.0]})
        >>> far = schema.pack({"base_lin_vel": np.zeros(3), "pose_command": [2.0, 0.0, 0.0, 0.0]})
        >>> bool(reward_model.reward(near, None, near) > reward_model.reward(far, None, far))
        True
    """

    def __init__(
        self,
        state_schema: IsaacChannelSchema,
        command_channel: str = "pose_command",
        coarse_weight: float = 0.5,
        coarse_std: float = 2.0,
        fine_weight: float = 0.5,
        fine_std: float = 0.2,
        heading_weight: float = 0.2,
    ) -> None:
        """Initialize the analytic navigation reward.

        Args:
            state_schema: Named blocks of the flat state vector.
            command_channel: Block holding the base-frame pose command; it must be
                :data:`POSE_COMMAND_WIDTH` wide.
            coarse_weight: Weight on the wide ``tanh`` position term.
            coarse_std: Length scale of the wide term, in metres.
            fine_weight: Weight on the narrow ``tanh`` position term.
            fine_std: Length scale of the narrow term, in metres.
            heading_weight: Weight on ``|heading_error|``; positive penalises error.

        Raises:
            ValueError: If ``command_channel`` is not :data:`POSE_COMMAND_WIDTH` wide, or a length
                scale is not positive.
        """
        if state_schema.width(command_channel) != POSE_COMMAND_WIDTH:
            raise ValueError(
                f"command_channel {command_channel!r} must be {POSE_COMMAND_WIDTH} wide "
                f"(x, y, z, heading), but the schema declares "
                f"{state_schema.width(command_channel)}"
            )
        if coarse_std <= 0.0 or fine_std <= 0.0:
            raise ValueError(
                f"tanh length scales must be positive, got coarse={coarse_std}, fine={fine_std}"
            )
        self.state_schema = state_schema
        self.command_channel = command_channel
        self.coarse_weight = float(coarse_weight)
        self.coarse_std = float(coarse_std)
        self.fine_weight = float(fine_weight)
        self.fine_std = float(fine_std)
        self.heading_weight = float(heading_weight)

    def goal_distance(self, state: Any) -> float:
        """Distance from the base to its commanded position, in metres.

        The norm is over all three position entries, matching the task's own
        ``position_command_error_tanh``. The z entry is the gap between the commanded and current
        base height and is near zero on flat ground, so this is a planar distance in practice.
        """
        goal = self.state_schema.block(state, self.command_channel).reshape(-1)
        return float(np.linalg.norm(goal[:3]))

    def planar_goal_distance(self, state: Any) -> float:
        """Distance from the base to its commanded position in the ground plane, in metres.

        This is the quantity the episode's success predicate thresholds, kept separate from
        :meth:`goal_distance` so the objective and the score cannot silently drift apart.
        """
        goal = self.state_schema.block(state, self.command_channel).reshape(-1)
        return float(np.linalg.norm(goal[:2]))

    def reward(self, state: Any, action: Any, next_state: Any) -> float:
        del action  # the objective scores where the base ended up, not how it got there
        resulting = state if next_state is None else next_state
        distance = self.goal_distance(resulting)
        heading = self.state_schema.block(resulting, self.command_channel).reshape(-1)[3]
        return (
            self.coarse_weight * (1.0 - float(np.tanh(distance / self.coarse_std)))
            + self.fine_weight * (1.0 - float(np.tanh(distance / self.fine_std)))
            - self.heading_weight * float(np.abs(wrap_angle(heading)))
        )


class NavigationIsaacModel(FactoredIsaacModelPOMDP):
    """Factored Isaac model whose state is goal-relative and whose reward is the task's own.

    A convenience wiring of :class:`GoalRelativeTransition` and :class:`NavigationRewardModel` into
    :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_factored_model.FactoredIsaacModelPOMDP`.
    The driven channels are ``(base_velocity, command)`` in that order; every other block —
    ``projected_gravity`` on the ANYmal task — is carried through, which is right for a quantity
    that is constant on flat terrain.

    Attributes:
        goal_transition: The :class:`GoalRelativeTransition` driving the base and the goal.
        navigation_reward: The :class:`NavigationRewardModel` scoring it, or ``None`` when a
            different reward model was supplied.
        base_velocity_channel: The block holding the body-frame base linear velocity.
        command_channel: The block holding the base-frame pose command.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
        ...     GaussianChannelObservationModel)
        >>>
        >>> schema = IsaacChannelSchema(
        ...     (("base_lin_vel", 3), ("projected_gravity", 3), ("pose_command", 4)))
        >>> model = NavigationIsaacModel(
        ...     state_schema=schema,
        ...     action_presets=[np.zeros(3), np.array([1.0, 0.0, 0.0])],
        ...     discount_factor=0.99,
        ...     step_dt=0.2,
        ...     observation_models={
        ...         "pose_command": GaussianChannelObservationModel(channel="pose_command")},
        ... )
        >>> state = schema.pack({"base_lin_vel": np.zeros(3),
        ...                      "projected_gravity": [0.0, 0.0, -1.0],
        ...                      "pose_command": [2.0, 0.0, 0.0, 0.0]})
        >>> moved = model.sample_next_state(state, model.get_actions()[1])
        >>> bool(schema.block(moved, "pose_command")[0] < 2.0)  # driving forward closes the gap
        True
        >>> schema.block(moved, "projected_gravity").tolist()  # level ground is not dynamics
        [0.0, 0.0, -1.0]
    """

    def __init__(
        self,
        state_schema: IsaacChannelSchema,
        action_presets: Sequence[ArrayLike],
        discount_factor: float,
        step_dt: float,
        linear_scale: float = 1.0,
        angular_scale: float = 1.0,
        base_velocity_channel: str = "base_lin_vel",
        command_channel: str = "pose_command",
        velocity_noise_std: float = 0.1,
        position_noise_std: float = 0.05,
        heading_noise_std: float = 0.05,
        reward_model: Optional[RewardModel] = None,
        observation_models: Optional[Mapping[str, IsaacObservationModel]] = None,
        raw_observation_schema: Optional[IsaacChannelSchema] = None,
        reward_range: Optional[Tuple[float, float]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the goal-relative navigation model.

        Args:
            state_schema: Named blocks of the flat state vector.
            action_presets: Finite list of ``(v_x, v_y, omega_z)`` velocity commands to plan over.
            discount_factor: POMDP discount factor (shared with the world).
            step_dt: Control-step duration in seconds, from the live task.
            linear_scale: Fraction of the commanded linear velocity actually achieved.
            angular_scale: Fraction of the commanded yaw rate actually achieved.
            base_velocity_channel: Block holding the body-frame base linear velocity.
            command_channel: Block holding the base-frame pose command.
            velocity_noise_std: Std of the base-velocity process noise, in m/s.
            position_noise_std: Std of the goal-position process noise, in metres.
            heading_noise_std: Std of the heading-error process noise, in radians.
            reward_model: Objective to optimize. ``None`` (the default) builds a
                :class:`NavigationRewardModel`, which is the point of this class — pass one
                explicitly only to score a different task.
            observation_models: ``{channel: IsaacObservationModel}``.
            raw_observation_schema: Named blocks of the world's flat raw observation.
            reward_range: Optional ``(min, max)`` reward bounds.
            name: Model name, also used to label planner output.

        Raises:
            ValueError: If either driven channel is not the width its contents require.
        """
        if state_schema.width(base_velocity_channel) != BASE_VELOCITY_WIDTH:
            raise ValueError(
                f"base_velocity_channel {base_velocity_channel!r} must be {BASE_VELOCITY_WIDTH} "
                f"wide (v_x, v_y, v_z), but the schema declares "
                f"{state_schema.width(base_velocity_channel)}"
            )
        if state_schema.width(command_channel) != POSE_COMMAND_WIDTH:
            raise ValueError(
                f"command_channel {command_channel!r} must be {POSE_COMMAND_WIDTH} wide "
                f"(x, y, z, heading), but the schema declares "
                f"{state_schema.width(command_channel)}"
            )
        self.base_velocity_channel = base_velocity_channel
        self.command_channel = command_channel
        self.goal_transition = GoalRelativeTransition(
            step_dt=step_dt,
            linear_scale=linear_scale,
            angular_scale=angular_scale,
            velocity_noise_std=velocity_noise_std,
            position_noise_std=position_noise_std,
            heading_noise_std=heading_noise_std,
        )
        objective = reward_model or NavigationRewardModel(
            state_schema=state_schema, command_channel=command_channel
        )
        self.navigation_reward = objective if isinstance(objective, NavigationRewardModel) else None
        super().__init__(
            state_schema=state_schema,
            action_presets=action_presets,
            discount_factor=discount_factor,
            transition=self.goal_transition,
            reward_model=objective,
            transition_channels=(base_velocity_channel, command_channel),
            observation_models=observation_models,
            raw_observation_schema=raw_observation_schema,
            reward_range=reward_range,
            name=name,
        )
