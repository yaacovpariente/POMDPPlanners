# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for the racetrack POMDP.

This module provides :class:`RacetrackVectorizedModel`, a fully batched implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for :class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp.RacetrackModelPOMDP`.
It re-expresses that model's kinematic-bicycle transition, Frenet integration, agent-slot
drift, highway-env reward, collision terminal check and both observation arms as torch
tensor kernels, so a vectorized planner (VOPP) can run tens of thousands of parallel
simulations without a per-row Python loop or a host/device sync.

Every dynamics, reward and perception parameter is read off a live ``RacetrackModelPOMDP``
instance, so the scalar model stays the single source of truth for configuration; only the
numeric kernels are duplicated here, and the parity test pins the two together.

**Tensor layouts.** The state is the schema's own vector, ``EGO_STATE_WIDTH + K *
AGENT_SLOT_WIDTH`` wide, and its column indices come from
:mod:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_schema` rather than from
literals here. The observation is the scalar model's encoded observation *flattened*, since
the protocol trades in ``[N, do]`` tensors rather than dictionaries:

* POMDP mode: the ``(2, 12, 12)`` occupancy grid in C order, so ``do = 288`` and
  ``obs.reshape(2, 12, 12)`` recovers the array the scalar model's ``"occupancy"`` key
  holds. The presence layer occupies the first 144 entries.
* MDP mode: ``[x, y, vx, vy]`` followed by ``K`` agent slots, so ``do = 4 + 5 * K``, which
  is the scalar model's ``"ego"`` and ``"agents"`` arrays concatenated.

Note:
    The scalar model's deliberate approximations carry over unchanged, because reproducing
    it is the point: agent slots drift at constant velocity while the world drives them
    with IDM, collisions use a centre-distance circle rather than an oriented rectangle,
    and the lane curvature is held fixed over the horizon.

Classes:
    RacetrackVectorizedModel: Batched torch counterpart of ``RacetrackModelPOMDP``.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp import RacetrackModelPOMDP
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_PRESENT,
    AGENT_REL_VX,
    AGENT_REL_X,
    AGENT_REL_Y,
    AGENT_SLOT_WIDTH,
    EGO_ANG,
    EGO_CURVATURE,
    EGO_HEADING,
    EGO_LAT,
    EGO_SPEED,
    EGO_STATE_WIDTH,
    EGO_X,
    EGO_Y,
    GRID_CELLS,
    GRID_HALF_EXTENT_M,
    GRID_STEP_M,
    MAX_ACCELERATION_MPS2,
    MAX_STEERING_RAD,
    ON_ROAD_LAYER,
    PRESENCE_LAYER,
    ObservationMode,
)

# Mirrors the scalar model: process noise touches the first six ego entries only, stopping
# short of curvature, which is a property of the lane rather than a state the vehicle
# diffuses through.
_EGO_NOISE_WIDTH = EGO_CURVATURE

# The ego sits in the middle cell of the occupancy grid, at (6, 6) for the shipped 12x12.
_GRID_CENTRE = GRID_CELLS // 2
_GRID_CELL_COUNT = GRID_CELLS * GRID_CELLS
_GRID_CENTRE_INDEX = _GRID_CENTRE * GRID_CELLS + _GRID_CENTRE

# The shipped occupancy observation carries two feature layers: presence and on_road.
_OCCUPANCY_LAYERS = 2

# The MDP arm's ego row is [x, y, vx, vy].
_EGO_OBS_WIDTH = 4

# Floor for the Frenet denominator ``1 - curvature * lat``, which vanishes at the centre of
# curvature and would otherwise divide by zero on a tight arc.
_MIN_FRENET_DENOMINATOR = 1e-3

# Keeps a flip probability of exactly 0 or 1 from turning a single mismatched cell into a
# -inf log-likelihood, which would annihilate an otherwise good particle.
_FLIP_PROB_EPS = 1e-12

# Seeds the observation-hash weights. Fixed so belief-tree keys are reproducible across
# processes; the particular value carries no meaning.
_HASH_SEED = 20260809


def _wrap_to_pi(angles: Tensor) -> Tensor:
    """Wrap angles in radians to ``[-pi, pi)``, elementwise."""
    return (angles + math.pi) % (2.0 * math.pi) - math.pi


def _rotate(vectors: Tensor, angles: Tensor) -> Tensor:
    """Rotate ``[N, K, 2]`` vectors counter-clockwise by a per-row angle ``[N]``."""
    cos_a = torch.cos(angles)[:, None]
    sin_a = torch.sin(angles)[:, None]
    x_component, y_component = vectors[..., 0], vectors[..., 1]
    return torch.stack(
        [
            cos_a * x_component - sin_a * y_component,
            sin_a * x_component + cos_a * y_component,
        ],
        dim=-1,
    )


def _hash_weights(count: int) -> np.ndarray:
    """Fixed pseudo-random odd int64 weights, one per observation entry.

    Not the first ``count`` primes, which is the obvious choice and the wrong one here: the
    POMDP arm's observation is a 288-entry *binary* vector, and a prime-weighted sum of a
    handful of small primes lands in a range of a few thousand, so two genuinely different
    occupancy grids collide onto one belief-tree node routinely. Weights spread over 48 bits
    push the collision rate down to the point where it stops mattering, and a fixed seed
    keeps the keys reproducible across processes.
    """
    generator = np.random.default_rng(_HASH_SEED)
    return generator.integers(1, 2**48, size=count, dtype=np.int64) | 1


class RacetrackVectorizedModel:
    """Fully vectorized torch generative model for the racetrack POMDP.

    Batches the transition, observation, reward, terminal and observation-likelihood
    kernels over a leading particle dimension and keeps every tensor on one device. Actions
    are integer indices into the scalar model's ``(acceleration, steering)`` preset table,
    which is the same vocabulary the world uses.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype used for state, observation and reward tensors.
        observation_mode: Which arm of the matched pair this model observes in.
        num_actions: Number of discrete control presets.
        state_dim: Width of the state vectors (``EGO_STATE_WIDTH + K * AGENT_SLOT_WIDTH``).
        observation_dim: Width of the flattened observation vectors.

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp import (
        ...     RacetrackModelPOMDP,
        ... )
        >>> from POMDPPlanners.environments.racetrack_pomdp.racetrack_vectorized_model import (
        ...     RacetrackVectorizedModel,
        ... )
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> env = RacetrackModelPOMDP(discount_factor=0.95)
        >>> model = RacetrackVectorizedModel(env, device=torch.device("cpu"))
        >>> states = torch.zeros(3, model.state_dim)
        >>> states[:, 3] = 10.0  # speed, m/s
        >>> actions = torch.full((3,), 4, dtype=torch.int64)  # coast, straight ahead
        >>> next_states = model.sample_next_states(states, actions)
        >>> rewards = model.rewards(states, actions, next_states)
        >>> tuple(next_states.shape), tuple(rewards.shape)
        ((3, 27), (3,))
    """

    def __init__(
        self,
        env: RacetrackModelPOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        observation_resolution: float = 1.0,
    ) -> None:
        """Build the model from a live scalar racetrack model.

        Args:
            env: The scalar model whose parameters and presets are mirrored. Every
                dynamics, reward and perception constant is read from it.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.
            observation_resolution: Grid spacing used to quantize observations into integer
                tree keys. Defaults to 1.0, which leaves the POMDP arm's already-binary
                occupancy cells untouched and bins the MDP arm's metres.

        Raises:
            ValueError: If ``observation_resolution`` is not positive.
        """
        if observation_resolution <= 0.0:
            raise ValueError(
                f"observation_resolution must be positive, got {observation_resolution}."
            )
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self.observation_mode = env.observation_mode
        self._obs_resolution = float(observation_resolution)
        self._num_agents = int(env.max_tracked_agents)
        self.state_dim = EGO_STATE_WIDTH + self._num_agents * AGENT_SLOT_WIDTH
        self.observation_dim = self._observation_width()
        self._read_action_table(env)
        self._read_dynamics_params(env)
        self._read_reward_params(env)
        self._read_perception_params(env)
        self._obs_hash = torch.as_tensor(
            _hash_weights(self.observation_dim), dtype=torch.int64, device=self.device
        )

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    def _observation_width(self) -> int:
        if self.observation_mode is ObservationMode.POMDP:
            return _OCCUPANCY_LAYERS * _GRID_CELL_COUNT
        return _EGO_OBS_WIDTH + self._num_agents * AGENT_SLOT_WIDTH

    def _read_action_table(self, env: RacetrackModelPOMDP) -> None:
        """Precompute the per-action slip, acceleration and control effort in float64.

        Deriving these in numpy and casting once keeps them bit-identical to the scalar
        model's own ``arctan(0.5 * tan(delta))``, rather than re-deriving them in whatever
        dtype the caller asked for.
        """
        presets = np.asarray(env.action_presets, dtype=np.float64)
        self.num_actions = int(presets.shape[0])
        self._slip = self._to_tensor(np.arctan(0.5 * np.tan(presets[:, 1] * MAX_STEERING_RAD)))
        self._acceleration = self._to_tensor(presets[:, 0] * MAX_ACCELERATION_MPS2)
        self._effort = self._to_tensor(np.linalg.norm(presets, axis=1))

    def _read_dynamics_params(self, env: RacetrackModelPOMDP) -> None:
        self._substeps = int(env.substeps)
        self._step = float(env.dt) / self._substeps
        self._wheelbase = float(env.vehicle_length) / 2.0
        self._lane_half_width = float(env.lane_half_width)
        self._collision_distance = float(env.collision_distance)
        self._process_noise_std = float(env.process_noise_std)

    def _read_reward_params(self, env: RacetrackModelPOMDP) -> None:
        self._collision_reward = float(env.collision_reward)
        self._lane_centering_cost = float(env.lane_centering_cost)
        self._lane_centering_reward = float(env.lane_centering_reward)
        self._action_reward = float(env.action_reward)

    def _read_perception_params(self, env: RacetrackModelPOMDP) -> None:
        self._cell_flip_prob = float(
            np.clip(env.cell_flip_prob, _FLIP_PROB_EPS, 1.0 - _FLIP_PROB_EPS)
        )
        self._ego_pose_std = float(env.ego_pose_std)
        self._agent_pose_std = float(env.agent_pose_std)
        self._agent_velocity_std = float(env.agent_velocity_std)

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
        """Propagate one decision forward per row, with the scalar model's process noise.

        Args:
            states: ``[N, state_dim]`` current states.
            actions: ``[N]`` integer indices into the control presets.

        Returns:
            ``[N, state_dim]`` sampled next states.
        """
        return self._perturb(self.propagate(states, actions))

    def propagate(self, states: Tensor, actions: Tensor) -> Tensor:
        """Return the noise-free propagation of each row, the transition's mean.

        Exposed because it is the deterministic kernel a parity test can compare exactly
        and a planner may want when it is predicting rather than sampling.

        Args:
            states: ``[N, state_dim]`` current states.
            actions: ``[N]`` integer indices into the control presets.

        Returns:
            ``[N, state_dim]`` propagated states, with no process noise added.
        """
        slip = self._slip[actions]
        acceleration = self._acceleration[actions]
        ego = states[:, :EGO_STATE_WIDTH].clone()
        agents = self._agent_rows(states).clone()
        for _ in range(self._substeps):
            ego, agents = self._integrate_substep(ego, agents, slip, acceleration)
        return torch.cat([ego, agents.reshape(states.shape[0], -1)], dim=1)

    def _integrate_substep(
        self, ego: Tensor, agents: Tensor, slip: Tensor, acceleration: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """One explicit-Euler sub-interval, mirroring the scalar model's update order."""
        speed, heading = ego[:, EGO_SPEED], ego[:, EGO_HEADING]
        lateral, angle = ego[:, EGO_LAT], ego[:, EGO_ANG]
        curvature = ego[:, EGO_CURVATURE]

        yaw_rate = speed * torch.sin(slip) / self._wheelbase
        # The Frenet rates use the velocity direction ``ang + slip``, not the heading, so
        # the lane offset tracks the world-frame update the two lines above already make.
        drift = angle + slip
        denominator = torch.clamp_min(1.0 - curvature * lateral, _MIN_FRENET_DENOMINATOR)
        along_rate = speed * torch.cos(drift) / denominator

        # Cloned rather than empty: curvature is carried over untouched, and a widened ego
        # block would inherit its old value rather than uninitialised memory.
        updated = ego.clone()
        updated[:, EGO_X] = ego[:, EGO_X] + speed * torch.cos(heading + slip) * self._step
        updated[:, EGO_Y] = ego[:, EGO_Y] + speed * torch.sin(heading + slip) * self._step
        updated[:, EGO_HEADING] = _wrap_to_pi(heading + yaw_rate * self._step)
        updated[:, EGO_LAT] = lateral + speed * torch.sin(drift) * self._step
        updated[:, EGO_ANG] = _wrap_to_pi(angle + (yaw_rate - curvature * along_rate) * self._step)
        updated[:, EGO_SPEED] = speed + acceleration * self._step
        return updated, self._drift_agents(agents, -yaw_rate * self._step)

    def _drift_agents(self, agents: Tensor, rotation: Tensor) -> Tensor:
        """Carry constant-velocity agent slots through one sub-interval of ego motion."""
        positions = agents[..., AGENT_REL_X : AGENT_REL_X + 2]
        velocities = agents[..., AGENT_REL_VX : AGENT_REL_VX + 2]
        drifted = positions + velocities * self._step
        moved = agents.clone()
        # Absent slots are all-zero and stay all-zero: rotating the origin is the origin.
        moved[..., AGENT_REL_X : AGENT_REL_X + 2] = _rotate(drifted, rotation)
        moved[..., AGENT_REL_VX : AGENT_REL_VX + 2] = _rotate(velocities, rotation)
        return moved

    def _perturb(self, states: Tensor) -> Tensor:
        if self._process_noise_std <= 0.0:
            return states
        noise = torch.randn(states.shape[0], _EGO_NOISE_WIDTH, dtype=self.dtype, device=self.device)
        perturbed = states.clone()
        perturbed[:, :_EGO_NOISE_WIDTH] += noise * self._process_noise_std
        return perturbed

    # ------------------------------------------------------------------ #
    # Reward and termination
    # ------------------------------------------------------------------ #

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        """Score each transition with highway-env's racetrack reward.

        Args:
            states: ``[N, state_dim]`` states the steps were taken from; unused, because the
                reward reads the realised successor only.
            actions: ``[N]`` integer indices into the control presets.
            next_states: ``[N, state_dim]`` realised successors.

        Returns:
            ``[N]`` immediate rewards, zero wherever the ego has left the lane.
        """
        del states  # The reward scores the resulting state, as the world does.
        lateral = next_states[:, EGO_LAT]
        centering = self._lane_centering_reward / (1.0 + self._lane_centering_cost * lateral**2)
        crashed = self._collision_mask(next_states).to(self.dtype)
        raw = (
            centering
            + self._action_reward * self._effort[actions]
            + self._collision_reward * crashed
        )
        # The upstream normalisation maps from [collision_reward, 1] using the literal 1,
        # not lane_centering_reward, and does not clip. Both quirks are reproduced.
        scaled = (raw - self._collision_reward) / (1.0 - self._collision_reward)
        on_road = (lateral.abs() <= self._lane_half_width).to(self.dtype)
        return scaled * on_road

    def terminal_mask(self, states: Tensor) -> Tensor:
        """Flag rows where the ego has left the lane or is inside a tracked agent.

        Args:
            states: ``[N, state_dim]`` states to test.

        Returns:
            ``[N]`` boolean tensor, ``True`` where the state is terminal.
        """
        off_lane = states[:, EGO_LAT].abs() > self._lane_half_width
        return off_lane | self._collision_mask(states)

    def _collision_mask(self, states: Tensor) -> Tensor:
        rows = self._agent_rows(states)
        present = rows[..., AGENT_PRESENT] > 0.5
        # sqrt of the summed squares, not torch.hypot: this has to agree with numpy's
        # ``linalg.norm`` to the last bit, or a slot sitting exactly on the collision
        # radius flips the terminal flag between the two implementations.
        ranges = torch.sqrt(rows[..., AGENT_REL_X] ** 2 + rows[..., AGENT_REL_Y] ** 2)
        return (present & (ranges <= self._collision_distance)).any(dim=1)

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        """Draw one observation per row in this model's observation mode.

        Args:
            next_states: ``[N, state_dim]`` post-transition states.
            actions: ``[N]`` action indices; unused, the observation depends on the state.

        Returns:
            ``[N, observation_dim]`` sampled observations, flattened as described in the
            module docstring.
        """
        del actions  # The observation depends on the state alone.
        if self.observation_mode is ObservationMode.POMDP:
            return self._sample_occupancy(next_states)
        return self._sample_kinematics(next_states)

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        """Score each observation against its row's state.

        In POMDP mode this is an independent Bernoulli over the 144 presence cells; the
        on-road layer is excluded because it is identical across every particle and a term
        that cannot discriminate has no business in a weight. In MDP mode it is a diagonal
        Gaussian over the ego row and the *present* agent slots, with presence read from the
        state rather than from the observation.

        Args:
            next_states: ``[N, state_dim]`` post-transition states.
            actions: ``[N]`` action indices; unused, the observation depends on the state.
            observations: ``[N, observation_dim]`` observations to score.

        Returns:
            ``[N]`` log-likelihoods.
        """
        del actions  # The observation depends on the state alone.
        if self.observation_mode is ObservationMode.POMDP:
            return self._occupancy_log_probs(next_states, observations)
        return self._kinematics_log_probs(next_states, observations)

    def _sample_occupancy(self, next_states: Tensor) -> Tensor:
        grid = self._render_presence_grid(next_states)
        flips = torch.rand(grid.shape, dtype=self.dtype, device=self.device) < self._cell_flip_prob
        occupancy = torch.zeros(
            next_states.shape[0],
            _OCCUPANCY_LAYERS,
            _GRID_CELL_COUNT,
            dtype=self.dtype,
            device=self.device,
        )
        occupancy[:, PRESENCE_LAYER] = torch.logical_xor(grid, flips).to(self.dtype)
        occupancy[:, ON_ROAD_LAYER] = 1.0
        return occupancy.reshape(next_states.shape[0], -1)

    def _occupancy_log_probs(self, next_states: Tensor, observations: Tensor) -> Tensor:
        grid = self._render_presence_grid(next_states)
        start = PRESENCE_LAYER * _GRID_CELL_COUNT
        observed = observations[:, start : start + _GRID_CELL_COUNT] > 0.5
        agreements = (observed == grid).sum(dim=1).to(self.dtype)
        disagreements = float(_GRID_CELL_COUNT) - agreements
        return agreements * math.log1p(-self._cell_flip_prob) + disagreements * math.log(
            self._cell_flip_prob
        )

    def _render_presence_grid(self, states: Tensor) -> Tensor:
        """Rasterise the presence layer as a flat ``[N, 144]`` boolean grid.

        Axis 0 of the grid is along-track and axis 1 across-track, matching highway-env
        1.12.1, and the ego is always written into the centre cell. A vehicle marks exactly
        one cell, not a footprint.
        """
        rows = self._agent_rows(states)
        along = torch.floor((rows[..., AGENT_REL_X] + GRID_HALF_EXTENT_M) / GRID_STEP_M)
        across = torch.floor((rows[..., AGENT_REL_Y] + GRID_HALF_EXTENT_M) / GRID_STEP_M)
        inside = (
            (rows[..., AGENT_PRESENT] > 0.5)
            & (along >= 0.0)
            & (along < GRID_CELLS)
            & (across >= 0.0)
            & (across < GRID_CELLS)
        )
        cell = (along * GRID_CELLS + across).to(torch.int64)
        # Out-of-window slots are redirected onto the ego's own centre cell, which is
        # already set: writing there is a no-op, which is what "do not mark" means here.
        index = torch.where(inside, cell, torch.full_like(cell, _GRID_CENTRE_INDEX))
        flat = torch.zeros(states.shape[0], _GRID_CELL_COUNT, dtype=self.dtype, device=self.device)
        flat[:, _GRID_CENTRE_INDEX] = 1.0
        flat.scatter_(1, index, 1.0)
        return flat > 0.5

    def _clean_kinematics(self, states: Tensor) -> Tuple[Tensor, Tensor]:
        """The noise-free MDP reading: the ego row and the state's own agent slots."""
        speed, heading = states[:, EGO_SPEED], states[:, EGO_HEADING]
        ego = torch.stack(
            [
                states[:, EGO_X],
                states[:, EGO_Y],
                speed * torch.cos(heading),
                speed * torch.sin(heading),
            ],
            dim=1,
        )
        return ego, self._agent_rows(states)

    def _sample_kinematics(self, next_states: Tensor) -> Tensor:
        clean_ego, agents = self._clean_kinematics(next_states)
        batch = next_states.shape[0]
        ego = (
            clean_ego
            + torch.randn(batch, _EGO_OBS_WIDTH, dtype=self.dtype, device=self.device)
            * self._ego_pose_std
        )
        present = (agents[..., AGENT_PRESENT] > 0.5)[..., None]
        shape = (batch, self._num_agents, 2)
        pose_noise = torch.randn(shape, dtype=self.dtype, device=self.device)
        velocity_noise = torch.randn(shape, dtype=self.dtype, device=self.device)
        observed = agents.clone()
        observed[..., AGENT_REL_X : AGENT_REL_X + 2] += torch.where(
            present, pose_noise * self._agent_pose_std, torch.zeros_like(pose_noise)
        )
        observed[..., AGENT_REL_VX : AGENT_REL_VX + 2] += torch.where(
            present, velocity_noise * self._agent_velocity_std, torch.zeros_like(velocity_noise)
        )
        return torch.cat([ego, observed.reshape(batch, -1)], dim=1)

    def _kinematics_log_probs(self, next_states: Tensor, observations: Tensor) -> Tensor:
        clean_ego, agents = self._clean_kinematics(next_states)
        observed = observations[:, _EGO_OBS_WIDTH:].reshape(
            next_states.shape[0], self._num_agents, AGENT_SLOT_WIDTH
        )
        present = agents[..., AGENT_PRESENT] > 0.5
        count = present.sum(dim=1).to(self.dtype)
        positions = slice(AGENT_REL_X, AGENT_REL_X + 2)
        velocities = slice(AGENT_REL_VX, AGENT_REL_VX + 2)
        return (
            _gaussian_log_prob(
                ((observations[:, :_EGO_OBS_WIDTH] - clean_ego) ** 2).sum(dim=1),
                torch.full_like(count, float(_EGO_OBS_WIDTH)),
                self._ego_pose_std,
            )
            + _gaussian_log_prob(
                self._masked_square_error(observed, agents, positions, present),
                2.0 * count,
                self._agent_pose_std,
            )
            + _gaussian_log_prob(
                self._masked_square_error(observed, agents, velocities, present),
                2.0 * count,
                self._agent_velocity_std,
            )
        )

    @staticmethod
    def _masked_square_error(
        observed: Tensor, agents: Tensor, columns: slice, present: Tensor
    ) -> Tensor:
        """Squared error over a slot's columns, summed over the *present* slots only."""
        deviation = observed[..., columns] - agents[..., columns]
        per_slot = (deviation**2).sum(dim=-1)
        return torch.where(present, per_slot, torch.zeros_like(per_slot)).sum(dim=1)

    # ------------------------------------------------------------------ #
    # Tree keys
    # ------------------------------------------------------------------ #

    def action_keys(self, actions: Tensor) -> Tensor:
        """Return the belief-tree key of each action index.

        Args:
            actions: ``[N]`` integer action indices.

        Returns:
            ``[N]`` int64 keys; the index is already a dense key.
        """
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        """Hash observations into integer belief-tree keys.

        Args:
            observations: ``[N, observation_dim]`` observations.

        Returns:
            ``[N]`` int64 keys: a weighted sum of the quantized entries, wrapping mod 2**64
            on the rare row large enough to overflow. In POMDP mode the entries are already
            binary, so the quantization is the identity and only the hashing is lossy.
        """
        quantized = torch.floor(observations / self._obs_resolution).to(torch.int64)
        return (quantized * self._obs_hash).sum(dim=1)


def _gaussian_log_prob(square_error: Tensor, count: Tensor, std: float) -> Tensor:
    """Diagonal-Gaussian log-density from a summed squared error and a per-row term count."""
    variance = std**2
    return -0.5 * square_error / variance - 0.5 * count * math.log(2.0 * math.pi * variance)


__all__ = ["RacetrackVectorizedModel"]
