# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for the goal-relative navigation model.

This module provides :class:`NavigationVectorizedModel`, a fully batched implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for
:class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_navigation_model.NavigationIsaacModel`.
It re-expresses that model's goal-relative transition, its two-scale ``tanh`` pose-tracking reward,
its always-false terminal test and an additive-Gaussian observation as torch tensor kernels, so
VOPP can run its whole forward search on one device without a host sync.

Every constant — the control step, the two tracking scales, the per-block noise, the reward weights
and length scales and the channel offsets — is read from a live ``NavigationIsaacModel``, so the
scalar model stays the single source of truth for configuration and only the numeric kernels are
duplicated here. The accompanying parity test pins the two together.

What this buys over the fitted linear model
-------------------------------------------
A fitted ``s' = A s + B a + b`` surrogate has to learn that turning rotates the goal, from a few
hundred samples of a state that never says where the robot is. It cannot, so its reward ranking of
the velocity commands is nearly flat and the planner's argmax never moves. Here the goal is rotated
and translated exactly, so a command that drives at the goal and one that drives away from it lead
to measurably different distances. That difference is what a search can climb.

Observation model
-----------------
The whole state vector plus isotropic Gaussian noise, matching the manipulator model's. The task's
own observation carries no configured noise, so the std here is the sensor error the belief is
asked to assume; a strictly positive value is what keeps the particle filter from collapsing onto
one particle.

Classes:
    NavigationVectorizedModel: Batched torch kernels for the goal-relative navigation model.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_navigation_model import (
    BASE_VELOCITY_WIDTH,
    VELOCITY_COMMAND_WIDTH,
    NavigationIsaacModel,
)


class NavigationVectorizedModel:
    """Fully vectorized torch generative model for the goal-relative navigation model.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype used for state / observation / reward tensors.
        num_actions: Number of preset actions the planner chooses among.
        state_dim: Width of the flat state vector.

    Example:
        >>> import numpy as np
        >>> import torch
        >>> from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
        ...     IsaacChannelSchema, NavigationIsaacModel)
        >>>
        >>> schema = IsaacChannelSchema(
        ...     (("base_lin_vel", 3), ("projected_gravity", 3), ("pose_command", 4)))
        >>> scalar = NavigationIsaacModel(
        ...     state_schema=schema,
        ...     action_presets=[np.zeros(3), np.array([1.0, 0.0, 0.0]),
        ...                     np.array([0.0, 0.0, 1.0])],
        ...     discount_factor=0.99, step_dt=0.2)
        >>> model = NavigationVectorizedModel(scalar, device=torch.device("cpu"))
        >>> states = torch.zeros(4, 10)
        >>> states[:, 6] = 2.0  # the goal is 2 m straight ahead of every particle
        >>> actions = torch.tensor([0, 1, 2, 1])
        >>> next_states = model.sample_next_states(states, actions)
        >>> tuple(next_states.shape), model.num_actions
        ((4, 10), 3)
        >>> rewards = model.rewards(states, actions, next_states)
        >>> bool(rewards[1] != rewards[2])  # driving at the goal beats turning away from it
        True
    """

    def __init__(
        self,
        model: NavigationIsaacModel,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        observation_noise_std: float = 0.1,
        observation_resolution: float = 0.5,
    ) -> None:
        """Build the vectorized kernels from a configured scalar navigation model.

        Args:
            model: The scalar model to mirror. Its transition, reward weights and channel layout
                are all read from it.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.
            observation_noise_std: Std of the additive observation noise.
            observation_resolution: Grid spacing used to quantize observations into tree keys.

        Raises:
            ValueError: If the model carries no :class:`NavigationRewardModel`, an action preset is
                not a three-wide velocity command, or a numeric argument is not positive.
        """
        if model.navigation_reward is None:
            raise ValueError(
                "NavigationVectorizedModel mirrors the analytic pose-tracking objective, but this "
                "model was built with a different reward model; there is nothing to vectorize"
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
        self._build_reward(model)
        self._hash_primes = torch.as_tensor(
            _first_primes(self.state_dim), dtype=torch.int64, device=self.device
        )

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    def _build_action_table(self, model: NavigationIsaacModel) -> None:
        presets = np.asarray([np.asarray(action, dtype=float) for action in model.get_actions()])
        if presets.ndim != 2 or presets.shape[0] == 0:
            raise ValueError("action presets must form a non-empty [num_actions, 3] table")
        if presets.shape[1] != VELOCITY_COMMAND_WIDTH:
            raise ValueError(
                f"action presets are {presets.shape[1]} wide but a base velocity command is "
                f"{VELOCITY_COMMAND_WIDTH} wide (v_x, v_y, omega_z)"
            )
        self._action_table = self._tensor(presets)
        self.num_actions = int(presets.shape[0])

    def _build_layout(self, model: NavigationIsaacModel) -> None:
        schema = model.state_schema
        self._velocity_slice = _bounds(schema.slice_of(model.base_velocity_channel))
        self._command_slice = _bounds(schema.slice_of(model.command_channel))
        # The heading error is the last entry of the command block, and it is the one entry of the
        # state that is an angle rather than a length.
        self._heading_index = self._command_slice[1] - 1

    def _build_transition(self, model: NavigationIsaacModel) -> None:
        transition = model.goal_transition
        self._linear_scale = transition.linear_scale
        stds = transition.process_noise_std
        self._velocity_std = self._tensor(stds[:BASE_VELOCITY_WIDTH])
        self._goal_std = self._tensor(stds[BASE_VELOCITY_WIDTH:])
        # Precompute the per-action body-frame step: it depends on the command alone, so it is a
        # small constant table rather than something to recompute for every particle.
        steps = np.asarray([transition.body_step(preset) for preset in model.get_actions()])
        self._step_translation = self._tensor(steps[:, :2])
        self._step_rotation = self._tensor(steps[:, 2])

    def _build_reward(self, model: NavigationIsaacModel) -> None:
        reward = model.navigation_reward
        if reward is None:  # pragma: no cover - guarded in __init__
            raise ValueError("no navigation reward to vectorize")
        self._coarse_weight = reward.coarse_weight
        self._coarse_std = reward.coarse_std
        self._fine_weight = reward.fine_weight
        self._fine_std = reward.fine_std
        self._heading_weight = reward.heading_weight

    def _tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)

    @property
    def action_vectors(self) -> Tensor:
        """The ``[num_actions, 3]`` table of continuous ``(v_x, v_y, omega_z)`` commands."""
        return self._action_table

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        commands = self._action_table[actions]
        translation = self._step_translation[actions]
        rotation = self._step_rotation[actions]
        command_start, command_stop = self._command_slice
        goal = states[:, command_start:command_stop]
        cos_d, sin_d = torch.cos(rotation), torch.sin(rotation)
        relative = goal[:, :2] - translation
        moved = torch.stack(
            [
                cos_d * relative[:, 0] + sin_d * relative[:, 1],
                -sin_d * relative[:, 0] + cos_d * relative[:, 1],
                goal[:, 2],
                _wrap_angle(goal[:, 3] - rotation),
            ],
            dim=-1,
        )
        next_states = states.clone()
        next_goal = moved + self._goal_std * torch.randn_like(moved)
        next_goal[:, 3] = _wrap_angle(next_goal[:, 3])
        next_states[:, command_start:command_stop] = next_goal
        velocity_start, velocity_stop = self._velocity_slice
        tracked = torch.zeros_like(states[:, velocity_start:velocity_stop])
        tracked[:, :2] = commands[:, :2] * self._linear_scale
        next_states[:, velocity_start:velocity_stop] = (
            tracked + self._velocity_std * torch.randn_like(tracked)
        )
        return next_states

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions  # observation noise is additive and action-independent
        observations = next_states + self._obs_std * torch.randn_like(next_states)
        observations[:, self._heading_index] = _wrap_angle(observations[:, self._heading_index])
        return observations

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        del states, actions  # the objective scores where the base ended up
        command_start, command_stop = self._command_slice
        goal = next_states[:, command_start:command_stop]
        distance = torch.linalg.norm(goal[:, :3], dim=-1)
        return (
            self._coarse_weight * (1.0 - torch.tanh(distance / self._coarse_std))
            + self._fine_weight * (1.0 - torch.tanh(distance / self._fine_std))
            - self._heading_weight * torch.abs(_wrap_angle(goal[:, 3]))
        )

    def goal_distances(self, states: Tensor) -> Tensor:
        """Distance from each row's base to its commanded position, in metres."""
        command_start, _ = self._command_slice
        return torch.linalg.norm(states[:, command_start : command_start + 3], dim=-1)

    def planar_goal_distances(self, states: Tensor) -> Tensor:
        """Ground-plane distance from each row's base to its commanded position, in metres.

        This is the quantity the episode's success predicate thresholds, kept separate from
        :meth:`goal_distances` so the objective and the score cannot silently drift apart.
        """
        command_start, _ = self._command_slice
        return torch.linalg.norm(states[:, command_start : command_start + 2], dim=-1)

    def terminal_mask(self, states: Tensor) -> Tensor:
        # The world owns termination; a model that guessed at it would prune states the episode is
        # still able to visit.
        return torch.zeros(states.shape[0], dtype=torch.bool, device=states.device)

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions  # additive-Gaussian likelihood does not depend on the action
        residual = observations - next_states
        # The heading entry is an angle, and the task's own heading command is uniform over a full
        # revolution, so the wrap boundary is not a corner case here — it is where a large share of
        # the episodes start. Scoring the raw difference would put a particle at +pi and an
        # observation at -pi two revolutions apart and resample away exactly the particles that
        # were right.
        residual[:, self._heading_index] = _wrap_angle(residual[:, self._heading_index])
        return self._obs_lognorm - 0.5 * (residual * residual).sum(dim=-1) / (self._obs_std**2)

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        quantized = torch.floor(observations / self._obs_resolution).to(torch.int64)
        return (quantized * self._hash_primes).sum(dim=1)


def _wrap_angle(angle: Tensor) -> Tensor:
    """Wrap angles into ``(-pi, pi]``, matching the scalar model's convention."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _bounds(span: slice) -> Tuple[int, int]:
    """Turn a schema slice into an explicit ``(start, stop)`` pair."""
    return int(span.start), int(span.stop)


def _first_primes(count: int) -> np.ndarray:
    """Return ``count`` large primes used as spatial-hash weights."""
    primes = []
    candidate = 73856093
    while len(primes) < count:
        if all(candidate % divisor != 0 for divisor in range(2, int(candidate**0.5) + 1)):
            primes.append(candidate)
        candidate += 2
    return np.asarray(primes, dtype=np.int64)
