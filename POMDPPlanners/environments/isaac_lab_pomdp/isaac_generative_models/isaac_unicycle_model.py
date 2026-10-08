# SPDX-License-Identifier: MIT

"""Analytic unicycle dynamics for IsaacLab velocity-command tasks.

A task like ``Isaac-Navigation-Flat-Anymal-C-v0`` does not hand the planner joint torques. Its
action is a 3-D base velocity command ``(v_x, v_y, omega_z)`` riding a pre-trained low-level
locomotion policy, and that policy's job is precisely to make the base track the command. The
high-level dynamics are therefore close to a planar unicycle, and writing them down analytically is
a far more faithful model than ridge-fitting a linear map to whole-body state — the same move
CARLA's kinematic-bicycle model makes against a learned CARLA transition.

Two honest approximations, both stated so a study can check them rather than discover them:

* **Tracking is imperfect.** ``command_scale`` is the fraction of the commanded velocity the
  low-level policy actually achieves; measure it from a rollout rather than assuming 1.0.
* **The density ignores yaw wrap-around.** The residual is wrapped into ``(-pi, pi]`` before it is
  scored, which is right for a heading, but the resulting density is a wrapped Gaussian truncated
  to one revolution. With a per-step heading noise well under a radian — the regime a velocity
  command is in — the omitted mass is negligible.

Classes:
    UnicycleTransition: Planar unicycle integration of a body-frame velocity command.
    UnicycleIsaacModel: Factored model wired with a unicycle transition on a pose channel.
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
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp import (
    NoiseStd,
    RewardModel,
    TransitionModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_model import (
    IsaacObservationModel,
)

#: Width of the pose block a unicycle transition drives: ``(x, y, yaw)``.
POSE_WIDTH = 3


def wrap_angle(angle: ArrayLike) -> np.ndarray:
    """Wrap angles into ``(-pi, pi]``."""
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


class UnicycleTransition(TransitionModel):
    """Planar unicycle integration of a body-frame velocity command, with Gaussian process noise.

    The driven state block is exactly ``(x, y, yaw)``; the action is ``(v_x, v_y, omega_z)`` in the
    body frame. One control step of duration ``step_dt`` moves the pose by the rotated linear
    velocity and the yaw rate, both scaled by ``command_scale``.

    Attributes:
        dim: Width of the driven block, always :data:`POSE_WIDTH`.
        step_dt: Control-step duration in seconds.
        command_scale: Fraction of the commanded velocity the low-level policy achieves.

    Example:
        >>> import numpy as np
        >>> transition = UnicycleTransition(step_dt=0.5, process_noise_std=1e-6)
        >>> nxt = transition.sample_next_state([0.0, 0.0, 0.0], [2.0, 0.0, 0.0])
        >>> bool(abs(nxt[0] - 1.0) < 1e-3)
        True
    """

    dim = POSE_WIDTH

    def __init__(
        self,
        step_dt: float,
        process_noise_std: NoiseStd = 0.05,
        command_scale: float = 1.0,
    ) -> None:
        """Initialize the unicycle transition.

        Args:
            step_dt: Control-step duration in seconds. Read it from the live task
                (``env.unwrapped.step_dt``) rather than guessing — the integration is linear in it,
                so a wrong value scales every predicted displacement.
            process_noise_std: Scalar or per-channel std of the noise on ``(x, y, yaw)``.
            command_scale: Fraction of the commanded velocity the low-level policy achieves.

        Raises:
            ValueError: If ``step_dt`` is not positive or any noise std is not positive.
        """
        if step_dt <= 0.0:
            raise ValueError(f"step_dt must be positive, got {step_dt}")
        self.step_dt = float(step_dt)
        self.command_scale = float(command_scale)
        self._std = np.broadcast_to(
            np.asarray(process_noise_std, dtype=float), (POSE_WIDTH,)
        ).astype(float)
        if np.any(self._std <= 0.0):
            raise ValueError("process noise standard deviations must be strictly positive")

    def _mean(self, state: Any, action: Any) -> np.ndarray:
        pose = np.asarray(state, dtype=float).reshape(-1)[:POSE_WIDTH]
        command = np.asarray(action, dtype=float).reshape(-1)[:POSE_WIDTH] * self.command_scale
        yaw = pose[2]
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        return np.array(
            [
                pose[0] + (command[0] * cos_yaw - command[1] * sin_yaw) * self.step_dt,
                pose[1] + (command[0] * sin_yaw + command[1] * cos_yaw) * self.step_dt,
                float(wrap_angle(yaw + command[2] * self.step_dt)),
            ]
        )

    def body_frame_delta(self, action: Any) -> np.ndarray:
        """The base's own ``(dx, dy, dyaw)`` over one step, in the frame it starts the step in.

        Integrating from the origin with zero yaw is what makes this the *relative* motion: the
        rotation by the current yaw that :meth:`sample_next_state` applies drops out. A model whose
        state is expressed relative to the base — a goal in the base frame, say — needs exactly
        this quantity and nothing else about where the base is on the floor, so exposing it keeps
        one integration serving both.

        Args:
            action: The ``(v_x, v_y, omega_z)`` body-frame velocity command.

        Returns:
            The ``(dx, dy, dyaw)`` displacement, already scaled by ``command_scale`` and
            ``step_dt``.
        """
        return self._mean(np.zeros(POSE_WIDTH), action)

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        mean = self._mean(state, action)
        samples = mean[np.newaxis, :] + np.random.normal(
            0.0, self._std, size=(n_samples, POSE_WIDTH)
        )
        samples[:, 2] = wrap_angle(samples[:, 2])
        return samples[0] if n_samples == 1 else samples

    def log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        mean = self._mean(state, action)
        candidates = np.atleast_2d(np.asarray(next_states, dtype=float))[:, :POSE_WIDTH]
        residual = candidates - mean[np.newaxis, :]
        residual[:, 2] = wrap_angle(residual[:, 2])
        normalizer = float(np.sum(np.log(self._std)) + 0.5 * POSE_WIDTH * np.log(2.0 * np.pi))
        return -0.5 * np.sum((residual / self._std[np.newaxis, :]) ** 2, axis=-1) - normalizer


class UnicycleIsaacModel(FactoredIsaacModelPOMDP):
    """Factored Isaac model whose dynamics are a unicycle on one named pose channel.

    A convenience wiring of :class:`UnicycleTransition` into
    :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_factored_model.FactoredIsaacModelPOMDP`.
    Every state block other than ``pose_channel`` is carried through the transition unchanged, so
    a latent block rides along without being resampled.

    Attributes:
        pose_channel: The state block the unicycle drives; it must be 3 wide.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
        ...     GaussianChannelObservationModel)
        >>> from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
        ...     IsaacChannelSchema)
        >>>
        >>> schema = IsaacChannelSchema((("base_pose", 3), ("goal", 2)))
        >>> model = UnicycleIsaacModel(
        ...     state_schema=schema,
        ...     action_presets=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])],
        ...     discount_factor=0.99,
        ...     step_dt=0.5,
        ...     observation_models={
        ...         "base_pose": GaussianChannelObservationModel(channel="base_pose")},
        ... )
        >>> state = schema.pack({"base_pose": [0.0, 0.0, 0.0], "goal": [3.0, 0.0]})
        >>> next_state = model.sample_next_state(state, model.get_actions()[0])
        >>> bool(next_state[0] > 0.0)  # a forward command moves the base forward
        True
        >>> schema.block(next_state, "goal").tolist()  # the goal is not dynamics
        [3.0, 0.0]
    """

    def __init__(
        self,
        state_schema: IsaacChannelSchema,
        action_presets: Sequence[ArrayLike],
        discount_factor: float,
        step_dt: float,
        pose_channel: str = "base_pose",
        process_noise_std: NoiseStd = 0.05,
        command_scale: float = 1.0,
        reward_model: Optional[RewardModel] = None,
        observation_models: Optional[Mapping[str, IsaacObservationModel]] = None,
        raw_observation_schema: Optional[IsaacChannelSchema] = None,
        reward_range: Optional[Tuple[float, float]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the unicycle-driven Isaac model.

        Args:
            state_schema: Named blocks of the flat state vector.
            action_presets: Finite list of ``(v_x, v_y, omega_z)`` velocity commands to plan over.
            discount_factor: POMDP discount factor (shared with the world).
            step_dt: Control-step duration in seconds, from the live task.
            pose_channel: The state block holding ``(x, y, yaw)``. Defaults to ``"base_pose"``.
            process_noise_std: Scalar or per-channel std of the pose process noise.
            command_scale: Fraction of the commanded velocity the low-level policy achieves.
            reward_model: The objective the planner optimizes.
            observation_models: ``{channel: IsaacObservationModel}``.
            raw_observation_schema: Named blocks of the world's flat raw observation.
            reward_range: Optional ``(min, max)`` reward bounds.
            name: Model name, also used to label planner output.

        Raises:
            ValueError: If ``pose_channel`` is not 3 wide in the schema.
        """
        if state_schema.width(pose_channel) != POSE_WIDTH:
            raise ValueError(
                f"pose_channel {pose_channel!r} must be {POSE_WIDTH} wide (x, y, yaw) for a "
                f"unicycle transition, but the schema declares {state_schema.width(pose_channel)}"
            )
        self.pose_channel = pose_channel
        super().__init__(
            state_schema=state_schema,
            action_presets=action_presets,
            discount_factor=discount_factor,
            transition=UnicycleTransition(
                step_dt=step_dt,
                process_noise_std=process_noise_std,
                command_scale=command_scale,
            ),
            reward_model=reward_model,
            transition_channels=(pose_channel,),
            observation_models=observation_models,
            raw_observation_schema=raw_observation_schema,
            reward_range=reward_range,
            name=name,
        )
