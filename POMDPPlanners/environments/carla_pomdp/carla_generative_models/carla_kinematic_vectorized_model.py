# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for the kinematic CARLA model.

This module provides :class:`CarlaKinematicVectorizedModel`, a fully batched,
GPU-friendly implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for :class:`~POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_kinematic_model_pomdp.KinematicCarlaModelPOMDP`.

It re-expresses the scalar model's kinematic-bicycle transition, obstacle-aware
driving-quality reward, predicted-collision terminal check, and factored
perception (GNSS Gaussian noise plus per-slot agent detection with range and
occlusion gating and additive pose noise) as torch tensor kernels, so a
vectorized planner (VOPP) can run tens of thousands of parallel simulations on
the GPU without a host/device sync. Every constant (control presets, kinematic
coefficients, reward weights, perception parameters) is read from a live
:class:`KinematicCarlaModelPOMDP` instance, so the environment stays the single
source of truth for configuration; only the numeric kernels are duplicated in
torch. The accompanying parity test pins these kernels to the scalar model.

State layout is ``[ego(7)] + K*[present, rel_x, rel_y, rel_yaw, rel_speed]`` with
``K = max_tracked_agents`` (default ``ds = 7 + 5*5 = 32``); the observation drops
the ego block down to the GNSS position and keeps the agent slots
(``do = 2 + K*5 = 27``). Actions are integer indices into the discrete
``(throttle, steer, brake)`` control presets.
"""

import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_kinematic_model_pomdp import (
    KinematicCarlaModelPOMDP,
)
from POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models.agent_models import (
    FactoredAgentObservationModel,
)
from POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models.gnss_models import (
    GnssObservationModel,
)
from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
    AGENT_SLOT_WIDTH,
    EGO_STATE_WIDTH,
    REWARD_FAST_PENALTY,
    REWARD_LAT_WEIGHT,
    REWARD_OUT_PENALTY,
    REWARD_SPEED_WEIGHT,
    REWARD_STEER_WEIGHT,
    REWARD_STEP_COST,
)

# Log-prob floor for impossible / near-zero events, mirroring ``_LOG_EPS`` in
# ``carla_perception.observation_models.agent_models``.
_LOG_EPS = -50.0


def _first_n_primes(count: int) -> np.ndarray:
    """Return the first ``count`` primes as an int64 array (spatial-hash weights)."""
    primes = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % p for p in primes):
            primes.append(candidate)
        candidate += 1
    return np.array(primes, dtype=np.int64)


class CarlaKinematicVectorizedModel:
    """Fully vectorized torch generative model for the kinematic CARLA model.

    The model batches the transition, observation, reward, terminal, and
    observation-likelihood kernels over a leading particle dimension and keeps
    every tensor on a single device. Actions are integer indices into the fixed
    ``(throttle, steer, brake)`` control-preset table read from the scalar model.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype used for state / observation / reward tensors.
        num_actions: Number of discrete control presets.
        state_dim: Width of the state vectors (``7 + K*5``).
        observation_dim: Width of the observation vectors (``2 + K*5``).

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_kinematic_model_pomdp import (
        ...     KinematicCarlaModelPOMDP,
        ... )
        >>> from POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_kinematic_vectorized_model import (
        ...     CarlaKinematicVectorizedModel,
        ... )
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> env = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05)
        >>> model = CarlaKinematicVectorizedModel(env, device=torch.device("cpu"))
        >>> states = torch.zeros(3, model.state_dim)
        >>> actions = torch.zeros(3, dtype=torch.int64)  # cruise straight
        >>> next_states = model.sample_next_states(states, actions)
        >>> rewards = model.rewards(states, actions, next_states)
        >>> tuple(next_states.shape), tuple(rewards.shape)
        ((3, 32), (3,))
    """

    def __init__(
        self,
        env: KinematicCarlaModelPOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        observation_resolution: float = 0.5,
    ) -> None:
        """Build the model from a live kinematic CARLA model instance.

        Args:
            env: The scalar model whose parameters and presets are mirrored.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.
            observation_resolution: Grid spacing used to quantize continuous
                observations into integer tree keys.

        Raises:
            ValueError: If ``observation_resolution`` is not positive.
            NotImplementedError: If ``env`` does not use the factored agent and
                gaussian GNSS observation models.
        """
        if observation_resolution <= 0.0:
            raise ValueError("observation_resolution must be positive")
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self._obs_resolution = float(observation_resolution)
        self._num_agents = int(env.max_tracked_agents)
        self.state_dim = EGO_STATE_WIDTH + self._num_agents * AGENT_SLOT_WIDTH
        self.observation_dim = 2 + self._num_agents * AGENT_SLOT_WIDTH
        self._action_table = self._to_tensor(np.asarray(env.action_presets, dtype=np.float64))
        self.num_actions = int(self._action_table.shape[0])
        self._read_dynamics_params(env)
        self._read_perception_params(env)
        self._obs_hash = torch.as_tensor(
            _first_n_primes(self.observation_dim), dtype=torch.int64, device=self.device
        )

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    def _read_dynamics_params(self, env: KinematicCarlaModelPOMDP) -> None:
        self._dt = float(env.dt)
        self._wheelbase = float(env.wheelbase)
        self._max_steer_angle = float(env.max_steer_angle)
        self._accel = float(env.accel)
        self._brake_decel = float(env.brake_decel)
        self._drag = float(env.drag)
        self._collision_gap = float(env.collision_gap)
        self._collision_halfwidth = float(env.collision_halfwidth)
        self._safe_distance = float(env.safe_distance)
        self._stop_gap = float(env.stop_gap)
        self._desired_speed = float(env.desired_speed)
        self._out_lane_thresh = float(env.out_lane_thresh)
        self._collision_penalty = float(env.collision_penalty)

    def _read_perception_params(self, env: KinematicCarlaModelPOMDP) -> None:
        models = env.observation_models or {}
        agents = models.get("agents")
        gnss = models.get("gnss")
        if not isinstance(agents, FactoredAgentObservationModel):
            raise NotImplementedError(
                "vectorized model requires the factored agent observation model"
            )
        if not isinstance(gnss, GnssObservationModel):
            raise NotImplementedError(
                "vectorized model requires the gaussian gnss observation model"
            )
        perception_range = agents.perception_range
        self._perception_range = math.inf if perception_range is None else float(perception_range)
        self._occlusion_radius = float(agents.occlusion_radius)
        self._pose_std = float(agents.pose_std)
        self._detect_prob = float(agents.detect_prob)
        self._gnss_std = float(gnss.gnss_std)

    def _to_tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)

    def _agent_rows(self, states: Tensor) -> Tensor:
        return states[:, EGO_STATE_WIDTH:].reshape(
            states.shape[0], self._num_agents, AGENT_SLOT_WIDTH
        )

    # ------------------------------------------------------------------ #
    # Transition
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        control = self._action_table[actions]
        ego_next = self._propagate_ego(states[:, :EGO_STATE_WIDTH], control)
        agents_next = self._advance_agents(self._agent_rows(states), ego_next[:, 3], ego_next[:, 4])
        return torch.cat([ego_next, agents_next.reshape(states.shape[0], -1)], dim=1)

    def _propagate_ego(self, ego: Tensor, control: Tensor) -> Tensor:
        throttle, steer, brake = control[:, 0], control[:, 1], control[:, 2]
        yaw = torch.deg2rad(ego[:, 2])
        speed = torch.hypot(ego[:, 3], ego[:, 4])
        accel = throttle * self._accel - brake * self._brake_decel - self._drag * speed
        speed_next = torch.clamp_min(speed + accel * self._dt, 0.0)
        yaw_rate = (speed_next / self._wheelbase) * torch.tan(steer * self._max_steer_angle)
        yaw_next = yaw + yaw_rate * self._dt
        vx_next = speed_next * torch.cos(yaw_next)
        vy_next = speed_next * torch.sin(yaw_next)
        heading_err_next = ego[:, 6] + yaw_rate * self._dt
        lateral_next = ego[:, 5] + speed_next * torch.sin(heading_err_next) * self._dt
        return torch.stack(
            [
                ego[:, 0] + vx_next * self._dt,
                ego[:, 1] + vy_next * self._dt,
                torch.rad2deg(yaw_next),
                vx_next,
                vy_next,
                lateral_next,
                heading_err_next,
            ],
            dim=1,
        )

    def _advance_agents(self, rows: Tensor, vx_next: Tensor, vy_next: Tensor) -> Tensor:
        ego_speed = torch.hypot(vx_next, vy_next)
        present = rows[..., 0] == 1.0
        closing = (rows[..., 4] - ego_speed[:, None]) * self._dt
        rel_x_next = rows[..., 1] + torch.where(present, closing, torch.zeros_like(closing))
        agents = rows.clone()
        agents[..., 1] = rel_x_next
        return agents

    # ------------------------------------------------------------------ #
    # Terminal / reward
    # ------------------------------------------------------------------ #

    def terminal_mask(self, states: Tensor) -> Tensor:
        rows = self._agent_rows(states)
        present = rows[..., 0] == 1.0
        rel_x, rel_y = rows[..., 1], rows[..., 2]
        hit = present & (rel_x >= 0.0) & (rel_x < self._collision_gap)
        hit = hit & (rel_y.abs() < self._collision_halfwidth)
        return hit.any(dim=1)

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        del states  # Reward scores the realised next state only.
        steer = self._action_table[actions, 1]
        ego_yaw = torch.deg2rad(next_states[:, 2])
        lane_yaw = ego_yaw - next_states[:, 6]
        lspeed = next_states[:, 3] * torch.cos(lane_yaw) + next_states[:, 4] * torch.sin(lane_yaw)
        desired = self._obstacle_aware_desired_speed(next_states)
        r_fast = torch.where(lspeed > desired, -1.0, 0.0)
        r_out = torch.where(next_states[:, 5].abs() > self._out_lane_thresh, -1.0, 0.0)
        r_coll = torch.where(self.terminal_mask(next_states), -1.0, 0.0)
        return (
            self._collision_penalty * r_coll
            + REWARD_SPEED_WEIGHT * lspeed
            + REWARD_FAST_PENALTY * r_fast
            + REWARD_OUT_PENALTY * r_out
            - REWARD_STEER_WEIGHT * steer**2
            - REWARD_LAT_WEIGHT * steer.abs() * lspeed**2
            - REWARD_STEP_COST
        )

    def _obstacle_aware_desired_speed(self, next_states: Tensor) -> Tensor:
        full = torch.full(
            (next_states.shape[0],), self._desired_speed, dtype=self.dtype, device=self.device
        )
        if self._stop_gap == 0.0:
            return full
        rows = self._agent_rows(next_states)
        rel_x, rel_y = rows[..., 1], rows[..., 2]
        lead = (rows[..., 0] == 1.0) & (rel_x > 0.0) & (rel_y.abs() < self._collision_halfwidth)
        gaps = torch.where(lead, rel_x, torch.full_like(rel_x, math.inf))
        gap = gaps.min(dim=1).values
        ramp = self._desired_speed * (gap - self._stop_gap) / (self._safe_distance - self._stop_gap)
        ramped = torch.where(gap <= self._stop_gap, torch.zeros_like(gap), ramp)
        return torch.where(gap >= self._safe_distance, full, ramped)

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions  # Perception does not depend on the action.
        gnss = (
            next_states[:, :2]
            + torch.randn(next_states.shape[0], 2, dtype=self.dtype, device=self.device)
            * self._gnss_std
        )
        rows = self._agent_rows(next_states)
        visible = self._visible_mask(rows)
        noise = (
            torch.randn(
                rows.shape[0],
                self._num_agents,
                AGENT_SLOT_WIDTH - 1,
                dtype=self.dtype,
                device=self.device,
            )
            * self._pose_std
        )
        perceived = torch.zeros_like(rows)
        perceived[..., 0] = visible.to(self.dtype)
        perceived[..., 1:] = torch.where(
            visible[..., None], rows[..., 1:] + noise, torch.zeros_like(noise)
        )
        return torch.cat([gnss, perceived.reshape(rows.shape[0], -1)], dim=1)

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions  # Perception does not depend on the action.
        gnss_diff = observations[:, :2] - next_states[:, :2]
        gnss_lp = -0.5 * (gnss_diff / self._gnss_std).pow(2).sum(dim=1) - 2.0 * math.log(
            self._gnss_std
        )
        return gnss_lp + self._agent_log_probs(next_states, observations)

    def _agent_log_probs(self, next_states: Tensor, observations: Tensor) -> Tensor:
        rows = self._agent_rows(next_states)
        obs_rows = observations[:, 2:].reshape(rows.shape[0], self._num_agents, AGENT_SLOT_WIDTH)
        vis_present = (rows[..., 0] == 1.0) & self._visible_mask(rows)
        obs_present = obs_rows[..., 0] == 1.0
        pose_diff = obs_rows[..., 1:] - rows[..., 1:]
        gauss = (
            math.log(self._detect_prob)
            - 0.5 * (pose_diff / self._pose_std).pow(2).sum(dim=-1)
            - (AGENT_SLOT_WIDTH - 1) * math.log(self._pose_std)
        )
        miss = math.log(max(1.0 - self._detect_prob, math.exp(_LOG_EPS)))
        detected = torch.where(obs_present, gauss, torch.full_like(gauss, miss))
        gated = torch.where(obs_present, torch.full_like(gauss, _LOG_EPS), torch.zeros_like(gauss))
        slot_lp = torch.where(vis_present, detected, gated)
        return slot_lp.sum(dim=1)

    def _visible_mask(self, rows: Tensor) -> Tensor:
        present = rows[..., 0] == 1.0
        distance = torch.hypot(rows[..., 1], rows[..., 2])
        in_range = distance <= self._perception_range
        return present & in_range & ~self._occluded_mask(rows, present)

    def _occluded_mask(self, rows: Tensor, present: Tensor) -> Tensor:
        target_x, target_y = rows[..., 1:2], rows[..., 2:3]  # [N, K, 1]
        blocker_x, blocker_y = rows[..., 1].unsqueeze(1), rows[..., 2].unsqueeze(1)  # [N, 1, K]
        seg_len_sq = target_x * target_x + target_y * target_y  # [N, K, 1]
        dot = blocker_x * target_x + blocker_y * target_y  # [N, K, K]
        param = dot / torch.where(seg_len_sq > 0.0, seg_len_sq, torch.ones_like(seg_len_sq))
        perp_x = param * target_x - blocker_x
        perp_y = param * target_y - blocker_y
        perp = torch.hypot(perp_x, perp_y)
        not_self = ~torch.eye(self._num_agents, dtype=torch.bool, device=self.device)
        occ = (
            present.unsqueeze(1)
            & not_self.unsqueeze(0)
            & (seg_len_sq > 0.0)
            & (param > 0.0)
            & (param < 1.0)
            & (perp < self._occlusion_radius)
        )
        return occ.any(dim=2)

    # ------------------------------------------------------------------ #
    # Tree keys
    # ------------------------------------------------------------------ #

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        quantized = torch.floor(observations / self._obs_resolution).to(torch.int64)
        return (quantized * self._obs_hash).sum(dim=1)
