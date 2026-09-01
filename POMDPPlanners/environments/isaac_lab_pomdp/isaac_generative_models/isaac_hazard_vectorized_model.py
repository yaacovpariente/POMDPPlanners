# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized twin of the Franka hazard reach model.

:class:`~...isaac_manipulator_vectorized_model.ManipulatorVectorizedModel` already
expresses the joint lag, the modified-DH kinematics and the reach objective as
batched tensor kernels. Three things the hazard task adds are missing there, and
each is the reason a vectorized planner could not be used on this environment at
all:

**The contact penalty.** Reaching the goal is not the whole objective; sweeping
through the obstacle while it is present costs ``C * (v / v_max)^2``, with ``v``
the hand's own displacement rate over the step. Without it the planner optimises
a task nobody is studying.

**The latent presence bit, and the signal that reveals it.** The bit is drawn once
per episode and carried, so a state's own copy must survive the transition
untouched -- the batch path clones the state, which is what preserves it. The
observation carries one extra slot: a bit that reports the true presence with
``signal_accuracy`` while the hand is inside ``signal_radius``, and is a coin
flip outside it. That slot is the only thing that moves the belief over the
latent, so its likelihood has to be scored as a Bernoulli rather than folded into
the Gaussian on the continuous channels -- a Gaussian over a 0/1 slot would let
the filter update on a quantity that has no scale.

**Termination.** Contact sets a sticky flag when the model is configured that
way, and the planner has to see it or it will search past the end of episodes.

Classes:
    HazardReachVectorizedModel: Batched torch kernels for the Franka hazard reach model.
"""

import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_hazard_models import (
    EPISODE_DONE_CHANNEL,
    OBSTACLE_PRESENCE_CHANNEL,
    HazardReachIsaacModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_manipulator_vectorized_model import (  # noqa: E501
    ManipulatorVectorizedModel,
)


class HazardReachVectorizedModel(ManipulatorVectorizedModel):
    """Fully vectorized torch generative model for :class:`HazardReachIsaacModel`.

    Attributes:
        obstacle_center: Obstacle centre in the robot base frame, shape ``(3,)``.
        obstacle_radius: Contact radius, in metres.
        signal_radius: Hand range within which the presence signal is informative.
        signal_accuracy: In-range probability the signal reports the true presence.
        observation_dim: Width of an observation: the state plus the signal slot.

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_hazard_models import (
        ...     HazardReachIsaacModel)
        >>> scalar = HazardReachIsaacModel(p_present=1.0)
        >>> model = HazardReachVectorizedModel(scalar, device=torch.device("cpu"))
        >>> states = torch.zeros(4, model.state_dim)
        >>> actions = torch.tensor([0, 1, 2, 1])
        >>> next_states = model.sample_next_states(states, actions)
        >>> tuple(next_states.shape)
        (4, 34)
        >>> observations = model.sample_observations(next_states, actions)
        >>> observations.shape[1] == model.observation_dim  # proprioception + signal bit
        True
    """

    def __init__(
        self,
        model: HazardReachIsaacModel,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        observation_noise_std: float = 0.1,
        observation_resolution: float = 0.5,
    ) -> None:
        """Build the hazard kernels from a configured scalar hazard reach model.

        Args:
            model: The scalar model to mirror. Every constant -- obstacle geometry,
                penalty, saturation speed, signal range and accuracy -- is read
                from it, so the scalar model stays the single source of truth.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.
            observation_noise_std: Std of the additive proprioceptive noise.
            observation_resolution: Grid spacing quantizing observations into tree keys.

        Raises:
            ValueError: If the model carries no hazard reward.
        """
        super().__init__(
            model,
            device=device,
            dtype=dtype,
            observation_noise_std=observation_noise_std,
            observation_resolution=observation_resolution,
        )
        reward = model.hazard_reward
        if reward is None:
            raise ValueError("HazardReachVectorizedModel needs a HazardReachRewardModel")
        self.obstacle_center = self._tensor(reward.obstacle_center)
        self.obstacle_radius = float(reward.obstacle_radius)
        self._collision_penalty = float(reward.collision_penalty)
        self._speed_max = float(reward.ee_speed_max)
        self.signal_radius = float(model.signal_radius)
        self.signal_accuracy = float(model.signal_accuracy)
        self._is_contact_terminal = bool(model.is_contact_terminal)
        schema = model.state_schema
        self._presence_index = int(schema.slice_of(OBSTACLE_PRESENCE_CHANNEL).start)
        self._done_index = int(schema.slice_of(EPISODE_DONE_CHANNEL).start)
        # Proprioception only. The base class observes the whole state vector,
        # which here would hand the planner the latent presence bit and the
        # terminal flag directly -- the belief would never need the signal, and
        # a planner that can read the latent is not solving this task.
        self._proprioceptive_width = self._velocity_slice[1] - self._joint_slice[0]
        self.observation_dim = self._proprioceptive_width + 1
        # The Gaussian term covers the observed continuous channels only; the
        # signal slot is Bernoulli and is scored separately.
        self._obs_lognorm = float(
            -self._proprioceptive_width * math.log(self._obs_std)
            - 0.5 * self._proprioceptive_width * math.log(2.0 * math.pi)
        )

    # ------------------------------------------------------------------ #
    # Contact geometry
    # ------------------------------------------------------------------ #

    def hand_positions(self, states: Tensor) -> Tensor:
        """Batched hand position for ``[N, ds]`` states, shape ``[N, 3]``."""
        joint_start, joint_stop = self._joint_slice
        angles = states[:, joint_start:joint_stop][:, self._arm_indices] + self._default_joints
        return self.end_effector_positions(angles)

    def contact_mask(self, states: Tensor) -> Tensor:
        """Rows whose hand is inside the obstacle while the obstacle is present."""
        distance = torch.linalg.norm(
            self.hand_positions(states) - self.obstacle_center, dim=-1
        )
        present = states[:, self._presence_index] > 0.5
        return present & (distance <= self.obstacle_radius)

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        """Step the arm and, when configured, latch the terminal flag on contact.

        The base clones the state, so the presence bit, the command and the flag
        ride along unchanged -- which is what keeps the belief's variance over the
        latent from being written away by the dynamics.
        """
        next_states = super().sample_next_states(states, actions)
        if self._is_contact_terminal:
            latched = torch.maximum(
                next_states[:, self._done_index],
                self.contact_mask(next_states).to(next_states.dtype),
            )
            next_states[:, self._done_index] = latched
        return next_states

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        """Reach objective, minus the speed-scaled cost of touching a present obstacle."""
        base = super().rewards(states, actions, next_states)
        speed = (
            torch.linalg.norm(
                self.hand_positions(next_states) - self.hand_positions(states), dim=-1
            )
            / self._step_dt
        )
        severity = torch.clamp(speed / self._speed_max, min=0.0, max=1.0) ** 2
        penalty = -self._collision_penalty * severity
        return base + torch.where(
            self.contact_mask(next_states), penalty, torch.zeros_like(penalty)
        )

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        """Noisy proprioception, plus one bit reporting the obstacle's presence.

        The bit is truthful with ``signal_accuracy`` while the hand is within
        ``signal_radius`` of the obstacle centre, and a coin flip beyond it. That
        gate is the task: the presence has to be probed by approaching.
        """
        del actions  # proprioceptive noise is additive and action-independent
        observed = next_states[:, self._joint_slice[0] : self._velocity_slice[1]]
        continuous = observed + self._obs_std * torch.randn_like(observed)
        signal = self._signal_bits(next_states)
        return torch.cat([continuous, signal.unsqueeze(1)], dim=1)

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        """Gaussian on the continuous channels, Bernoulli on the signal slot."""
        del actions
        continuous = observations[:, : self._proprioceptive_width]
        residual = continuous - next_states[:, self._joint_slice[0] : self._velocity_slice[1]]
        gaussian = self._obs_lognorm - 0.5 * (residual * residual).sum(dim=-1) / (
            self._obs_std**2
        )
        reported = observations[:, self._proprioceptive_width]
        probability = self._signal_true_probability(next_states)
        bernoulli = torch.where(reported > 0.5, probability, 1.0 - probability)
        return gaussian + torch.log(torch.clamp(bernoulli, min=1e-12))

    def terminal_mask(self, states: Tensor) -> Tensor:
        """Rows whose episode has ended, read from the sticky flag."""
        if not self._is_contact_terminal:
            return torch.zeros(states.shape[0], dtype=torch.bool, device=states.device)
        return states[:, self._done_index] > 0.5

    def observation_keys(self, observations: Tensor) -> Tensor:
        """Quantize the continuous channels, and keep the signal bit exact.

        Quantizing the bit with the same grid as a joint angle would merge "saw a
        hazard" with "saw none" whenever the grid is coarser than one, which is
        every useful resolution.
        """
        continuous = observations[:, : self._proprioceptive_width]
        quantized = torch.floor(continuous / self._obs_resolution).to(torch.int64)
        keys = (quantized * self._hash_primes[: self._proprioceptive_width]).sum(dim=1)
        return keys * 2 + (observations[:, self._proprioceptive_width] > 0.5).to(torch.int64)

    # ------------------------------------------------------------------ #
    # Signal helpers
    # ------------------------------------------------------------------ #

    def _signal_true_probability(self, states: Tensor) -> Tensor:
        """Probability the signal slot reads 1, per row."""
        distance = torch.linalg.norm(
            self.hand_positions(states) - self.obstacle_center, dim=-1
        )
        inside = distance <= self.signal_radius
        present = states[:, self._presence_index] > 0.5
        truthful = torch.where(
            present,
            torch.full_like(distance, self.signal_accuracy),
            torch.full_like(distance, 1.0 - self.signal_accuracy),
        )
        return torch.where(inside, truthful, torch.full_like(distance, 0.5))

    def _signal_bits(self, states: Tensor) -> Tensor:
        """Draw the signal slot for each row."""
        probability = self._signal_true_probability(states)
        return (torch.rand_like(probability) < probability).to(self.dtype)
