# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for the analytic manipulator model.

This module provides :class:`ManipulatorVectorizedModel`, a fully batched implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for
:class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_manipulator_model.ManipulatorIsaacModel`.
It re-expresses that model's joint-lag transition, its analytic forward kinematics and reach
reward, its always-false terminal test and an additive-Gaussian proprioceptive observation as
torch tensor kernels, so VOPP can run its whole forward search on one device without a host sync.

Every constant — the lag gain, the action scale, the control step, the DH table, the default joint
pose, the reward weights and the channel offsets — is read from a live ``ManipulatorIsaacModel``,
so the scalar model stays the single source of truth for configuration and only the numeric
kernels are duplicated here. The accompanying parity test pins the two together.

What this buys over the fitted linear model
-------------------------------------------
The fitted ``s' = A s + B a + b`` surrogate makes the reward a linear functional of the state, so
its ranking of a 7-DoF arm's actions is nearly flat and the planner's argmax never moves. Here the
reward passes through a trigonometric chain: two joint commands place the hand in two different
places, and the distances to the goal genuinely differ. That difference is what a search can
climb.

Observation model
-----------------
Proprioception on a real arm is accurate but not exact, and a strictly positive noise std is what
keeps the particle filter from collapsing onto one particle. The observation here is therefore the
whole state vector plus isotropic Gaussian noise — the same shape as the one-space Isaac model's,
kept because the world reports the full policy-observation group and nothing in it is hidden.

Classes:
    ManipulatorVectorizedModel: Batched torch kernels for the analytic manipulator model.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_manipulator_model import (
    ManipulatorIsaacModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_helpers import spatial_hash_primes


class ManipulatorVectorizedModel:
    """Fully vectorized torch generative model for the analytic manipulator model.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype used for state / observation / reward tensors.
        num_actions: Number of preset actions the planner chooses among.
        state_dim: Width of the flat state vector.

    Example:
        >>> import numpy as np
        >>> import torch
        >>> from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
        ...     IsaacChannelSchema, ManipulatorIsaacModel, franka_panda_chain)
        >>>
        >>> schema = IsaacChannelSchema(
        ...     (("joint_pos", 7), ("joint_vel", 7), ("command", 7), ("last_action", 7)))
        >>> scalar = ManipulatorIsaacModel(
        ...     state_schema=schema,
        ...     action_presets=[np.zeros(7), np.full(7, 0.4), np.full(7, -0.4)],
        ...     discount_factor=0.99, step_dt=0.1, tracking_gain=0.5,
        ...     chain=franka_panda_chain(), default_joint_positions=np.zeros(7))
        >>> model = ManipulatorVectorizedModel(scalar, device=torch.device("cpu"))
        >>> states = torch.zeros(4, 28)
        >>> actions = torch.tensor([0, 1, 2, 1])
        >>> next_states = model.sample_next_states(states, actions)
        >>> tuple(next_states.shape), model.num_actions
        ((4, 28), 3)
        >>> rewards = model.rewards(states, actions, next_states)
        >>> bool(rewards[0] != rewards[1])  # different commands score differently
        True
    """

    def __init__(
        self,
        model: ManipulatorIsaacModel,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        observation_noise_std: float = 0.1,
        observation_resolution: float = 0.5,
    ) -> None:
        """Build the vectorized kernels from a configured scalar manipulator model.

        Args:
            model: The scalar model to mirror. Its transition, kinematics, reward weights and
                channel layout are all read from it.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.
            observation_noise_std: Std of the additive proprioceptive observation noise.
            observation_resolution: Grid spacing used to quantize observations into tree keys.

        Raises:
            ValueError: If the model carries no :class:`ReachRewardModel`, an action preset is not
                as wide as the arm has joints, or a numeric argument is not positive.
        """
        if model.reach_reward is None:
            raise ValueError(
                "ManipulatorVectorizedModel mirrors the analytic reach objective, but this model "
                "was built with a different reward model; there is nothing to vectorize"
            )
        if observation_noise_std <= 0.0:
            raise ValueError(f"observation_noise_std must be positive, got {observation_noise_std}")
        if observation_resolution <= 0.0:
            raise ValueError(
                f"observation_resolution must be positive, got {observation_resolution}"
            )
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self.state_dim = int(model.state_schema.total_dim)
        self._obs_std = float(observation_noise_std)
        self._obs_resolution = float(observation_resolution)
        self._obs_lognorm = float(
            -self.state_dim * math.log(self._obs_std)
            - 0.5 * self.state_dim * math.log(2.0 * math.pi)
        )
        self._build_action_table(model)
        self._build_layout(model)
        self._build_transition(model)
        self._build_kinematics(model)
        self._build_reward(model)
        self._hash_primes = torch.as_tensor(
            spatial_hash_primes(self.state_dim), dtype=torch.int64, device=self.device
        )

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    def _build_action_table(self, model: ManipulatorIsaacModel) -> None:
        presets = np.asarray([np.asarray(action, dtype=float) for action in model.get_actions()])
        if presets.ndim != 2 or presets.shape[0] == 0:
            raise ValueError("action presets must form a non-empty [num_actions, J] table")
        commanded = model.joint_transition.action_dim
        if presets.shape[1] != commanded:
            raise ValueError(
                f"action presets are {presets.shape[1]} wide but the arm has "
                f"{commanded} commanded joints"
            )
        self._action_table = self._tensor(presets)
        self.num_actions = int(presets.shape[0])

    def _build_layout(self, model: ManipulatorIsaacModel) -> None:
        schema = model.state_schema
        self._joint_slice = _bounds(schema.slice_of(model.joint_position_channel))
        self._velocity_slice = _bounds(schema.slice_of(model.joint_velocity_channel))
        self._action_slice = _bounds(schema.slice_of(model.last_action_channel))
        command = _bounds(schema.slice_of(model.command_channel))
        width = model.reach_reward.command_position_width if model.reach_reward else 3
        self._command_slice = (command[0], command[0] + int(width))

    def _build_transition(self, model: ManipulatorIsaacModel) -> None:
        transition = model.joint_transition
        self._gain = transition.tracking_gain
        self._action_scale = transition.action_scale
        self._step_dt = transition.step_dt
        width = transition.position_width
        stds = transition.process_noise_std
        self._position_std = self._tensor(stds[:width])
        self._velocity_std = self._tensor(stds[width : 2 * width])
        self._recorded_action_std = self._tensor(stds[2 * width :])
        self._actuated = torch.as_tensor(
            transition.actuated_indices, dtype=torch.int64, device=self.device
        )

    def _build_kinematics(self, model: ManipulatorIsaacModel) -> None:
        chain = model.chain
        self._num_joints = chain.num_joints
        self._link_lengths = self._tensor(chain.link_lengths)
        self._link_offsets = self._tensor(chain.link_offsets)
        self._cos_twist = self._tensor(np.cos(np.asarray(chain.link_twists, dtype=float)))
        self._sin_twist = self._tensor(np.sin(np.asarray(chain.link_twists, dtype=float)))
        self._tool_transform = self._tensor(chain.tool_transform)

    def _build_reward(self, model: ManipulatorIsaacModel) -> None:
        reward = model.reach_reward
        if reward is None:  # pragma: no cover - guarded in __init__
            raise ValueError("no reach reward to vectorize")
        self._default_joints = self._tensor(reward.default_joint_positions)
        self._arm_indices = torch.as_tensor(
            reward.arm_joint_indices, dtype=torch.int64, device=self.device
        )
        self._distance_weight = reward.distance_weight
        self._shaping_weight = reward.shaping_weight
        self._shaping_std = reward.shaping_std

    def _tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)

    @property
    def action_vectors(self) -> Tensor:
        """The ``[num_actions, J]`` table of continuous joint-target commands."""
        return self._action_table

    # ------------------------------------------------------------------ #
    # Kinematics
    # ------------------------------------------------------------------ #

    def end_effector_positions(self, joint_angles: Tensor) -> Tensor:
        """Batched hand position in the robot base frame for ``[N, J]`` absolute joint angles."""
        cos_t, sin_t = torch.cos(joint_angles), torch.sin(joint_angles)
        batch = joint_angles.shape[0]
        links = torch.zeros(
            batch, self._num_joints, 4, 4, dtype=self.dtype, device=joint_angles.device
        )
        links[..., 0, 0] = cos_t
        links[..., 0, 1] = -sin_t
        links[..., 0, 3] = self._link_lengths
        links[..., 1, 0] = sin_t * self._cos_twist
        links[..., 1, 1] = cos_t * self._cos_twist
        links[..., 1, 2] = -self._sin_twist
        links[..., 1, 3] = -self._link_offsets * self._sin_twist
        links[..., 2, 0] = sin_t * self._sin_twist
        links[..., 2, 1] = cos_t * self._sin_twist
        links[..., 2, 2] = self._cos_twist
        links[..., 2, 3] = self._link_offsets * self._cos_twist
        links[..., 3, 3] = 1.0
        pose = links[:, 0]
        for index in range(1, self._num_joints):
            pose = pose @ links[:, index]
        return (pose @ self._tool_transform)[:, :3, 3]

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        commands = self._action_table[actions]
        joint_start, joint_stop = self._joint_slice
        positions = states[:, joint_start:joint_stop]
        driven = positions[:, self._actuated]
        next_positions = positions.clone()
        next_positions[:, self._actuated] = driven + self._gain * (
            commands * self._action_scale - driven
        )
        next_velocities = (next_positions - positions) / self._step_dt
        next_states = states.clone()
        next_states[:, joint_start:joint_stop] = (
            next_positions + self._position_std * torch.randn_like(next_positions)
        )
        velocity_start, velocity_stop = self._velocity_slice
        next_states[:, velocity_start:velocity_stop] = (
            next_velocities + self._velocity_std * torch.randn_like(next_velocities)
        )
        action_start, action_stop = self._action_slice
        next_states[:, action_start:action_stop] = (
            commands + self._recorded_action_std * torch.randn_like(commands)
        )
        return next_states

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions  # proprioceptive noise is additive and action-independent
        return next_states + self._obs_std * torch.randn_like(next_states)

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        del states, actions  # the objective scores where the hand ended up
        distance = self.goal_distances(next_states)
        shaped = 1.0 - torch.tanh(distance / self._shaping_std)
        return self._distance_weight * distance + self._shaping_weight * shaped

    def goal_distances(self, states: Tensor) -> Tensor:
        """Distance from the modelled hand to each row's commanded position, in metres."""
        joint_start, joint_stop = self._joint_slice
        angles = states[:, joint_start:joint_stop][:, self._arm_indices] + self._default_joints
        command_start, command_stop = self._command_slice
        goal = states[:, command_start:command_stop]
        return torch.linalg.norm(self.end_effector_positions(angles) - goal, dim=-1)

    def terminal_mask(self, states: Tensor) -> Tensor:
        # The world owns termination; a model that guessed at it would prune states the episode is
        # still able to visit.
        return torch.zeros(states.shape[0], dtype=torch.bool, device=states.device)

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions  # additive-Gaussian likelihood does not depend on the action
        residual = observations - next_states
        return self._obs_lognorm - 0.5 * (residual * residual).sum(dim=-1) / (self._obs_std**2)

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        quantized = torch.floor(observations / self._obs_resolution).to(torch.int64)
        return (quantized * self._hash_primes).sum(dim=1)


def _bounds(span: slice) -> Tuple[int, int]:
    """Turn a schema slice into an explicit ``(start, stop)`` pair."""
    return int(span.start), int(span.stop)
