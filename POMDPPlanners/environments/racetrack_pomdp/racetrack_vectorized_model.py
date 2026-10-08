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

**One torch model, whatever the planner knows about the road.** The scalar side is abstract
in exactly one place — where the road bends under each particle — and its subclasses answer
it from a track map or from the camera. That is a *parameter* here rather than a second
model: this class takes a curvature source, resolves it off the scalar model it is built
from, and is otherwise one implementation. Two torch models differing in one lookup would
drift. The sources live in
:mod:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_vectorized_road`.

**Tensor layouts.** The state is the schema's own vector, ``EGO_STATE_WIDTH + K *
AGENT_SLOT_WIDTH`` wide, and its column indices come from
:mod:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_schema` rather than from
literals here. The observation is the scalar model's encoded observation *flattened*, since
the protocol trades in ``[N, do]`` tensors rather than dictionaries:

* POMDP mode: the ego's ``[x, y, heading, arclength]`` pose, its own speed, the lane
  camera's ``[lateral, angle]`` pair, its ``L`` curvature samples, then ``K`` detection rows
  of ``[detected, rel_x, rel_y, rel_vx, rel_vy]`` in C order, so ``do = 7 + L + 5 * K`` — 30
  at the shipped ``L = 3`` and ``K = 4``, against 291 for the occupancy grid this replaced.
  The offsets come from
  :mod:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_schema` rather than from
  literals here, so the flat layout has one definition.
* MDP mode: ``[x, y, vx, vy]`` followed by ``K`` agent slots, so ``do = 4 + 5 * K``, which
  is the scalar model's ``"ego"`` and ``"agents"`` arrays concatenated.

Note:
    The scalar model's deliberate approximations carry over unchanged, because reproducing
    it is the point: agent slots drift at constant velocity while the world drives them with
    IDM, collisions use a centre-distance circle rather than an oriented rectangle, and
    detections are associated to slots by range rank.

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
    DETECTION_PRESENT,
    DETECTION_REL_VX,
    DETECTION_REL_X,
    DETECTION_SLOT_WIDTH,
    EGO_ANG,
    EGO_ARCLENGTH_M,
    EGO_HEADING,
    EGO_LAT,
    EGO_POSE_ARCLENGTH,
    EGO_POSE_HEADING,
    EGO_POSE_X,
    EGO_SPEED,
    EGO_STATE_WIDTH,
    EGO_X,
    EGO_Y,
    LANE_POSE_ANG,
    LANE_POSE_LAT,
    MAX_ACCELERATION_MPS2,
    MAX_STEERING_RAD,
    OBSERVED_EGO_POSE_WIDTH,
    POMDP_OBS_CURVATURE_INDEX,
    POMDP_OBS_EGO_POSE_INDEX,
    POMDP_OBS_EGO_SPEED_INDEX,
    POMDP_OBS_LANE_POSE_INDEX,
    ObservationMode,
    pomdp_observation_width,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_vectorized_road import (
    CurvatureSource,
    ObservedCurvature,
    TrackMapCurvature,
    curvature_ahead_of,
    resolve_curvature_source,
)

# Mirrors the scalar model: process noise touches the first six ego entries only, stopping
# short of arclength. Arclength is integrated from the along-track rate, so jittering it
# would teleport a particle to a different part of the circuit rather than blur its pose.
_EGO_NOISE_WIDTH = EGO_ARCLENGTH_M

# The MDP arm's ego row is [x, y, vx, vy].
_EGO_OBS_WIDTH = 4

# Keeps a probability of exactly 0 or 1 from turning a single disagreement into a -inf
# log-likelihood, which at the shipped detection rates of 0 is what every disagreement is. A
# numerical guard rather than a sensor property: an all-zero weight vector is a crash, not an
# inference. The torch model's own clamp; the parity test is what holds it to the same number
# as the scalar side's.
CELL_PROB_EPS = 1e-12

# Floor for the Frenet denominator ``1 - curvature * lat``, which vanishes at the centre of
# curvature and would otherwise divide by zero on a tight arc.
_MIN_FRENET_DENOMINATOR = 1e-3

# Mirrors the scalar side's floor on every likelihood width, kept here rather than imported
# for the same reason `CELL_PROB_EPS` is: these are the torch model's own clamps, and the
# parity test is what holds the two sides to the same number.
_STD_EPS = 1e-9

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

    Not the first ``count`` primes, which is the obvious choice and the wrong one here: much
    of the POMDP arm's observation is binary detection flags and small quantized metres, and
    a prime-weighted sum of a handful of small primes lands in a range of a few thousand, so
    two genuinely different readings collide onto one belief-tree node routinely. Weights
    spread over 48 bits push the collision rate down to the point where it stops mattering,
    and a fixed seed keeps the keys reproducible across processes.
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
        curvature_source: Where each substep's curvature comes from — a track map, a live
            per-step estimate, or whatever the caller supplied. It also answers what the
            camera's curvature channel should read from each particle.

    Example:
        >>> import numpy as np
        >>> import torch
        >>> from POMDPPlanners.environments.racetrack_pomdp.racetrack_known_track_model import (
        ...     KnownTrackModel,
        ... )
        >>> from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import (
        ...     TrackGeometry,
        ... )
        >>> from POMDPPlanners.environments.racetrack_pomdp.racetrack_vectorized_model import (
        ...     RacetrackVectorizedModel,
        ... )
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> geometry = TrackGeometry(  # a 10 m straight into a 10 m left-hand arc
        ...     segment_starts=np.array([0.0, 10.0]),
        ...     segment_curvatures=np.array([0.0, 0.05]),
        ...     total_length_m=20.0,
        ... )
        >>> env = KnownTrackModel(discount_factor=0.95, track_geometry=geometry)
        >>> model = RacetrackVectorizedModel(env, device=torch.device("cpu"))
        >>> states = torch.zeros(3, model.state_dim)
        >>> states[:, 3] = 10.0  # speed, m/s
        >>> actions = torch.full((3,), 13, dtype=torch.int64)  # coast, straight ahead
        >>> next_states = model.sample_next_states(states, actions)
        >>> rewards = model.rewards(states, actions, next_states)
        >>> tuple(next_states.shape), tuple(rewards.shape)
        ((3, 27), (3,))
        >>> bool(next_states[0, 6] > 0.0)  # the arclength advanced along the track
        True
    """

    def __init__(
        self,
        env: RacetrackModelPOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        observation_resolution: float = 1.0,
        curvature_source: Optional[CurvatureSource] = None,
    ) -> None:
        """Build the model from a live scalar racetrack model.

        Args:
            env: The scalar model whose parameters and presets are mirrored. Every
                dynamics, reward and perception constant is read from it.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.
            observation_resolution: Grid spacing used to quantize observations into integer
                tree keys. Defaults to 1.0, which bins metres and metres per second; the
                POMDP arm's detection flags are already binary, so the quantization leaves
                them alone.
            curvature_source: Torch counterpart of the scalar model's ``_curvature_for``
                and ``curvature_ahead``, mapping an ego block to a per-row curvature.
                Defaults to None, which reads a track map or a live per-step estimate off
                ``env``.

        Raises:
            ValueError: If ``observation_resolution`` is not positive, or if
                ``curvature_source`` is omitted and ``env`` exposes neither a track map nor
                a per-step curvature estimate.
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
        self._lookahead_count = len(env.curvature_lookahead_m)
        self._detections_index = POMDP_OBS_CURVATURE_INDEX + self._lookahead_count
        self.state_dim = EGO_STATE_WIDTH + self._num_agents * AGENT_SLOT_WIDTH
        self.observation_dim = self._observation_width()
        self.curvature_source: CurvatureSource = (
            curvature_source
            if curvature_source is not None
            else resolve_curvature_source(env, self.device, dtype)
        )
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
            return pomdp_observation_width(self._num_agents, self._lookahead_count)
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
        # The detection rates: whether a report happens at all, as distinct from how wrong it
        # is. Validated on the scalar side, so both are already inside [0, 1). Zero at the
        # shipped defaults, because the world's detection decision is deterministic; the
        # clamp in `_cell_probability_log_probs` is what keeps a contradiction finite there.
        self._presence_miss_prob = float(env.presence_miss_prob)
        self._presence_false_alarm_prob = float(env.presence_false_alarm_prob)
        # The clutter model: what a false alarm reports, so part of the lossy-radar
        # configuration. Without it the two branches of the detection model are a density and
        # a bare probability, which is not a comparison.
        self._clutter_position_scale = float(env.clutter_position_scale_m)
        self._clutter_velocity_scale = float(env.clutter_velocity_scale)
        # The sensor geometry, mirrored so the torch model predicts *whether* a slot should
        # have been reported by the same rule the world applied.
        self._max_detection_range = float(env.max_detection_range_m)
        self._blocker_half_width = float(env.blocker_half_width_m)
        # Two values per width, matching the scalar model: it *draws* with the width as
        # configured and only floors it when scoring, so a zero-width model samples the truth
        # exactly rather than the truth plus a nanometre of torch noise.
        self._ego_position_std = float(env.ego_position_std_m)
        self._ego_heading_std = float(env.ego_heading_std_rad)
        self._ego_arclength_std = float(env.ego_arclength_std_m)
        self._ego_speed_std = float(env.ego_speed_std)
        self._lane_lateral_std = float(env.lane_lateral_std_m)
        self._lane_heading_std = float(env.lane_heading_std_rad)
        self._curvature_std = float(env.curvature_std_1pm)
        self._detection_position_std = float(env.detection_position_std_m)
        self._detection_velocity_std = float(env.detection_velocity_std)
        self._score_std = {
            name: max(value, _STD_EPS)
            for name, value in (
                ("ego_position", self._ego_position_std),
                ("ego_heading", self._ego_heading_std),
                ("ego_arclength", self._ego_arclength_std),
                ("ego_speed", self._ego_speed_std),
                ("lane_lateral", self._lane_lateral_std),
                ("lane_heading", self._lane_heading_std),
                ("curvature", self._curvature_std),
                ("detection_position", self._detection_position_std),
                ("detection_velocity", self._detection_velocity_std),
            )
        }
        # The per-entry widths the ego-pose channel is drawn at, in the channel's own order.
        self._ego_pose_draw_std = self._to_tensor(
            np.array(
                [
                    self._ego_position_std,
                    self._ego_position_std,
                    self._ego_heading_std,
                    self._ego_arclength_std,
                ]
            )
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
        # Read at the *start* of the substep, as the scalar model does, so a particle that
        # crosses into a corner mid-decision picks the new curvature up on the next
        # sub-interval rather than a whole decision late.
        curvature = self.curvature_source(ego)

        yaw_rate = speed * torch.sin(slip) / self._wheelbase
        # The Frenet rates use the velocity direction ``ang + slip``, not the heading, so
        # the lane offset tracks the world-frame update the two lines above already make.
        drift = angle + slip
        denominator = torch.clamp_min(1.0 - curvature * lateral, _MIN_FRENET_DENOMINATOR)
        along_rate = speed * torch.cos(drift) / denominator

        # Cloned rather than empty: a widened ego block would then carry its old values
        # over rather than uninitialised memory.
        updated = ego.clone()
        updated[:, EGO_X] = ego[:, EGO_X] + speed * torch.cos(heading + slip) * self._step
        updated[:, EGO_Y] = ego[:, EGO_Y] + speed * torch.sin(heading + slip) * self._step
        updated[:, EGO_HEADING] = _wrap_to_pi(heading + yaw_rate * self._step)
        updated[:, EGO_LAT] = lateral + speed * torch.sin(drift) * self._step
        updated[:, EGO_ANG] = _wrap_to_pi(angle + (yaw_rate - curvature * along_rate) * self._step)
        updated[:, EGO_SPEED] = speed + acceleration * self._step
        # Advanced by the along-track rate, not by ``speed * step``: the two differ once the
        # ego is yawed relative to the lane or offset from its centreline on an arc, and it
        # is the along-track one that indexes the curvature profile correctly.
        updated[:, EGO_ARCLENGTH_M] = ego[:, EGO_ARCLENGTH_M] + along_rate * self._step
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
            return self._sample_sensors(next_states)
        return self._sample_kinematics(next_states)

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        """Score each observation against its row's state.

        In POMDP mode this is a Gaussian in the speedometer residual, two in the lane
        camera's, one per curvature-ahead sample against whatever the curvature source
        predicts, and a Bernoulli over the detection ranks with a Gaussian in each matched
        detection's position and closing rate. In MDP mode it is a diagonal Gaussian over the
        ego row, the same detection model over the agent slots' presence flags, and a
        diagonal Gaussian over the slots *both* the state and the observation fill. Both arms
        therefore score whether a vehicle is there and not only where, and both do it at a
        rate rather than a hard zero.

        Args:
            next_states: ``[N, state_dim]`` post-transition states.
            actions: ``[N]`` action indices; unused, the observation depends on the state.
            observations: ``[N, observation_dim]`` observations to score.

        Returns:
            ``[N]`` log-likelihoods.
        """
        del actions  # The observation depends on the state alone.
        if self.observation_mode is ObservationMode.POMDP:
            return self._sensor_log_probs(next_states, observations)
        return self._kinematics_log_probs(next_states, observations)

    def _sample_sensors(self, next_states: Tensor) -> Tensor:
        batch = next_states.shape[0]
        noise = torch.randn(batch, 3, dtype=self.dtype, device=self.device)
        speed = next_states[:, EGO_SPEED] + self._ego_speed_std * noise[:, 0]
        lateral = next_states[:, EGO_LAT] + self._lane_lateral_std * noise[:, 1]
        angle = next_states[:, EGO_ANG] + self._lane_heading_std * noise[:, 2]
        curvature = self._curvature_ahead(next_states) + self._curvature_std * torch.randn(
            batch, self._lookahead_count, dtype=self.dtype, device=self.device
        )
        return torch.cat(
            [
                self._sample_ego_pose(next_states),
                speed[:, None],
                torch.stack([lateral, _wrap_to_pi(angle)], dim=1),
                curvature,
                self._sample_detections(next_states).reshape(batch, -1),
            ],
            dim=1,
        )

    def _ego_pose_of(self, states: Tensor) -> Tensor:
        """The four state slots the ego-pose channel reports, in the channel's own order."""
        return torch.stack(
            [
                states[:, EGO_X],
                states[:, EGO_Y],
                states[:, EGO_HEADING],
                states[:, EGO_ARCLENGTH_M],
            ],
            dim=1,
        )

    def _sample_ego_pose(self, states: Tensor) -> Tensor:
        noise = torch.randn(
            states.shape[0], OBSERVED_EGO_POSE_WIDTH, dtype=self.dtype, device=self.device
        )
        measured = self._ego_pose_of(states) + noise * self._ego_pose_draw_std
        # Wrapped after the noise, as the world does, so a car pointing just short of pi does
        # not read as pointing just past -pi.
        measured[:, EGO_POSE_HEADING] = _wrap_to_pi(measured[:, EGO_POSE_HEADING])
        return measured

    def _ego_pose_log_probs(self, next_states: Tensor, observations: Tensor) -> Tensor:
        """Gaussians over the four ego-pose entries, the heading one wrapped.

        Three terms rather than one because the widths differ per entry and the heading
        residual has to be taken modulo 2*pi: without that, a particle on the far side of the
        branch cut is charged 6.28 rad of error for a hundredth of a radian of disagreement.
        """
        pose = observations[
            :, POMDP_OBS_EGO_POSE_INDEX : POMDP_OBS_EGO_POSE_INDEX + OBSERVED_EGO_POSE_WIDTH
        ]
        residual = pose - self._ego_pose_of(next_states)
        ones = torch.ones_like(residual[:, EGO_POSE_HEADING])
        position = residual[:, EGO_POSE_X : EGO_POSE_X + 2]
        return (
            _gaussian_log_prob(
                (position**2).sum(dim=1), 2.0 * ones, self._score_std["ego_position"]
            )
            + _gaussian_log_prob(
                _wrap_to_pi(residual[:, EGO_POSE_HEADING]) ** 2,
                ones,
                self._score_std["ego_heading"],
            )
            + _gaussian_log_prob(
                residual[:, EGO_POSE_ARCLENGTH] ** 2, ones, self._score_std["ego_arclength"]
            )
        )

    def _curvature_ahead(self, states: Tensor) -> Tensor:
        return curvature_ahead_of(
            self.curvature_source, states[:, :EGO_STATE_WIDTH], self._lookahead_count
        )

    def _sensor_log_probs(self, next_states: Tensor, observations: Tensor) -> Tensor:
        speed = observations[:, POMDP_OBS_EGO_SPEED_INDEX] - next_states[:, EGO_SPEED]
        lateral = (
            observations[:, POMDP_OBS_LANE_POSE_INDEX + LANE_POSE_LAT] - next_states[:, EGO_LAT]
        )
        angle = _wrap_to_pi(
            observations[:, POMDP_OBS_LANE_POSE_INDEX + LANE_POSE_ANG] - next_states[:, EGO_ANG]
        )
        curvature = observations[
            :, POMDP_OBS_CURVATURE_INDEX : self._detections_index
        ] - self._curvature_ahead(next_states)
        ones = torch.ones_like(speed)
        return (
            self._ego_pose_log_probs(next_states, observations)
            + _gaussian_log_prob(speed**2, ones, self._score_std["ego_speed"])
            + _gaussian_log_prob(lateral**2, ones, self._score_std["lane_lateral"])
            + _gaussian_log_prob(angle**2, ones, self._score_std["lane_heading"])
            + _gaussian_log_prob(
                (curvature**2).sum(dim=1),
                torch.full_like(speed, float(self._lookahead_count)),
                self._score_std["curvature"],
            )
            + self._detection_log_probs(next_states, observations)
        )

    # ------------------------------------------------------------------ #
    # Detections
    # ------------------------------------------------------------------ #

    def _detected_probabilities(self, occupancy: Tensor) -> Tensor:
        # The torch mirror of `racetrack_detection.detected_probabilities`, shared by the
        # POMDP arm's detection ranks and the MDP arm's slot flags so the two cannot drift.
        return (
            self._presence_false_alarm_prob
            + (1.0 - self._presence_miss_prob - self._presence_false_alarm_prob) * occupancy
        )

    def _predicted_detections(self, states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """What should be reported per row: positions, relative velocities, and how many.

        The torch mirror of the scalar model's ``predicted_detections``. Every slot is
        evaluated and the invisible ones are sorted to the back rather than dropped, because
        reading a count off the device to slice with would be a host sync per row, and this
        module exists to keep those out of the hot loop.

        Returns:
            ``([N, K, 2]`` positions, ``[N, K, 2]`` relative velocities, ``[N]`` visible
            counts``)``, the first two ordered nearest-first with the invisible slots
            trailing.
        """
        rows = self._agent_rows(states)
        present = rows[..., AGENT_PRESENT] > 0.5
        positions = rows[..., AGENT_REL_X : AGENT_REL_X + 2]
        velocities = rows[..., AGENT_REL_VX : AGENT_REL_VX + 2]
        ranges = torch.sqrt(positions[..., 0] ** 2 + positions[..., 1] ** 2)
        bearings = torch.atan2(positions[..., 1], positions[..., 0])
        # arcsin of a clipped ratio: a blocker closer than its own half-width would leave the
        # domain, and the clip turns that into "everything behind it is hidden".
        half_width = torch.asin(
            (self._blocker_half_width / ranges.clamp_min(1e-9)).clamp(-1.0, 1.0)
        )
        separation = _wrap_to_pi(bearings[:, :, None] - bearings[:, None, :]).abs()
        blocked = (
            present[:, None, :]
            & (ranges[:, None, :] < ranges[:, :, None])
            & (separation < half_width[:, None, :])
        ).any(dim=2)
        visible = present & (ranges <= self._max_detection_range) & ~blocked

        order = torch.argsort(
            torch.where(visible, ranges, torch.full_like(ranges, float("inf"))), dim=1
        )
        pairs = order[:, :, None].expand(-1, -1, 2)
        return (
            torch.gather(positions, 1, pairs),
            torch.gather(velocities, 1, pairs),
            visible.sum(dim=1),
        )

    def _detection_log_probs(self, next_states: Tensor, observations: Tensor) -> Tensor:
        """Bernoulli over the detection ranks, plus a Gaussian per matched detection.

        Association is by range rank, exactly as on the scalar side: the ``i``-th visible
        slot is scored against the ``i``-th detection. A detection no slot reaches costs its
        false-alarm rate *and* the clutter density of what was reported, which is what keeps
        the two branches comparable.
        """
        positions, moving, visible_count = self._predicted_detections(next_states)
        observed = observations[:, self._detections_index :].reshape(
            observations.shape[0], self._num_agents, DETECTION_SLOT_WIDTH
        )
        ranks = torch.arange(self._num_agents, device=self.device)[None, :]
        predicted = ranks < visible_count[:, None]
        reported = ranks < (observed[..., DETECTION_PRESENT] > 0.5).sum(dim=1)[:, None]
        matched = predicted & reported

        offset = observed[..., DETECTION_REL_X : DETECTION_REL_X + 2] - positions
        velocity = observed[..., DETECTION_REL_VX : DETECTION_REL_VX + 2] - moving
        pairs = matched.sum(dim=1).to(self.dtype)
        zero = torch.zeros_like(pairs)[:, None].expand(-1, self._num_agents)
        return (
            _cell_probability_log_probs(
                self._detected_probabilities(predicted.to(self.dtype)), reported
            )
            + _gaussian_log_prob(
                torch.where(matched, (offset**2).sum(dim=-1), zero).sum(dim=1),
                2.0 * pairs,
                self._score_std["detection_position"],
            )
            + _gaussian_log_prob(
                torch.where(matched, (velocity**2).sum(dim=-1), zero).sum(dim=1),
                2.0 * pairs,
                self._score_std["detection_velocity"],
            )
            + _cauchy_log_probs(
                observed[..., DETECTION_REL_X : DETECTION_REL_X + 2],
                reported & ~predicted,
                self._clutter_position_scale,
            )
            + _cauchy_log_probs(
                observed[..., DETECTION_REL_VX : DETECTION_REL_VX + 2],
                reported & ~predicted,
                self._clutter_velocity_scale,
            )
        )

    def _sample_detections(self, next_states: Tensor) -> Tensor:
        """The sampler's half of the detection model, term for term with the density."""
        positions, moving, visible_count = self._predicted_detections(next_states)
        batch = next_states.shape[0]
        ranks = torch.arange(self._num_agents, device=self.device)[None, :]
        predicted = ranks < visible_count[:, None]
        draws = torch.rand(batch, self._num_agents, dtype=self.dtype, device=self.device)
        kept = predicted & (draws >= self._presence_miss_prob)
        invented = ~predicted & (draws < self._presence_false_alarm_prob)
        flag = kept | invented

        reported = [
            self._report_block(truth, invented, flag, noise_std, clutter_scale)
            for truth, noise_std, clutter_scale in (
                (positions, self._detection_position_std, self._clutter_position_scale),
                (moving, self._detection_velocity_std, self._clutter_velocity_scale),
            )
        ]
        return self._pack_detections(reported[0], reported[1], flag)

    def _report_block(
        self,
        truth: Tensor,
        invented: Tensor,
        flag: Tensor,
        noise_std: float,
        clutter_scale: float,
    ) -> Tensor:
        # One two-column block of a detection row: the truth plus sensor noise where a real
        # vehicle was kept, a Cauchy phantom where one was invented, zero where neither.
        # Branch-free rather than indexed, so no boolean is read off the device.
        shape = truth.shape
        measured = truth + noise_std * torch.randn(shape, dtype=self.dtype, device=self.device)
        clutter = clutter_scale * torch.tan(
            math.pi * (torch.rand(shape, dtype=self.dtype, device=self.device) - 0.5)
        )
        return torch.where(invented[..., None], clutter, measured) * flag[..., None]

    def _pack_detections(self, positions: Tensor, moving: Tensor, flag: Tensor) -> Tensor:
        """Order the drawn reports by measured range and pack them into a prefix of slots.

        The world emits its detections nearest-first with the empty slots trailing, and the
        density's rank association depends on that, so a sampled reading has to have the same
        shape. Sorting on a range of ``inf`` for the empty slots is what packs them.
        """
        ranges = torch.sqrt(positions[..., 0] ** 2 + positions[..., 1] ** 2)
        order = torch.argsort(
            torch.where(flag, ranges, torch.full_like(ranges, float("inf"))), dim=1
        )
        return torch.stack(
            [
                torch.gather(flag.to(self.dtype), 1, order),
                torch.gather(positions[..., 0], 1, order),
                torch.gather(positions[..., 1], 1, order),
                torch.gather(moving[..., 0], 1, order),
                torch.gather(moving[..., 1], 1, order),
            ],
            dim=-1,
        )

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
        return torch.cat([ego, self._detect_agent_slots(agents).reshape(batch, -1)], dim=1)

    def _detect_agent_slots(self, agents: Tensor) -> Tensor:
        # The torch mirror of the scalar `_detect_agent_slots`: a filled slot is dropped at
        # `presence_miss_prob`, an empty one filled with a Cauchy phantom at
        # `presence_false_alarm_prob`. Branch-free rather than indexed, so no boolean is read
        # off the device -- this runs per particle, per step.
        shape = agents.shape[:-1]
        draws = torch.rand(shape, dtype=self.dtype, device=self.device)
        filled = agents[..., AGENT_PRESENT] > 0.5
        kept = filled & (draws >= self._presence_miss_prob)
        invented = ~filled & (draws < self._presence_false_alarm_prob)
        real, phantom = kept[..., None], invented[..., None]
        observed = agents * kept.to(self.dtype)[..., None]
        observed[..., AGENT_PRESENT] = (kept | invented).to(self.dtype)
        for columns, noise_std, clutter_scale in (
            (
                slice(AGENT_REL_X, AGENT_REL_X + 2),
                self._agent_pose_std,
                self._clutter_position_scale,
            ),
            (
                slice(AGENT_REL_VX, AGENT_REL_VX + 2),
                self._agent_velocity_std,
                self._clutter_velocity_scale,
            ),
        ):
            block = (*shape, 2)
            measured = (
                observed[..., columns]
                + torch.randn(block, dtype=self.dtype, device=self.device) * noise_std
            )
            clutter = clutter_scale * torch.tan(
                math.pi * (torch.rand(block, dtype=self.dtype, device=self.device) - 0.5)
            )
            observed[..., columns] = torch.where(
                phantom, clutter, torch.where(real, measured, torch.zeros_like(measured))
            )
        return observed

    def _kinematics_log_probs(self, next_states: Tensor, observations: Tensor) -> Tensor:
        clean_ego, agents = self._clean_kinematics(next_states)
        observed = observations[:, _EGO_OBS_WIDTH:].reshape(
            next_states.shape[0], self._num_agents, AGENT_SLOT_WIDTH
        )
        present = agents[..., AGENT_PRESENT] > 0.5
        reported = observed[..., AGENT_PRESENT] > 0.5
        both = present & reported
        count = both.sum(dim=1).to(self.dtype)
        positions = slice(AGENT_REL_X, AGENT_REL_X + 2)
        velocities = slice(AGENT_REL_VX, AGENT_REL_VX + 2)
        return (
            _gaussian_log_prob(
                ((observations[:, :_EGO_OBS_WIDTH] - clean_ego) ** 2).sum(dim=1),
                torch.full_like(count, float(_EGO_OBS_WIDTH)),
                self._ego_pose_std,
            )
            + self._slot_detection_log_probs(present, reported)
            + _gaussian_log_prob(
                self._masked_square_error(observed, agents, positions, both),
                2.0 * count,
                self._agent_pose_std,
            )
            + _gaussian_log_prob(
                self._masked_square_error(observed, agents, velocities, both),
                2.0 * count,
                self._agent_velocity_std,
            )
            + self._clutter_log_probs(observed, reported & ~present)
        )

    def _clutter_log_probs(self, observed: Tensor, phantom: Tensor) -> Tensor:
        # Density of what a false alarm reported, over the slots only the observation fills.
        # The torch mirror of the scalar `_clutter_log_prob`, and it is not optional: without
        # it the miss branch is a density and the false-alarm branch a bare probability, and
        # the likelihood comes out inverted.
        return _cauchy_log_probs(
            observed[..., AGENT_REL_X : AGENT_REL_X + 2], phantom, self._clutter_position_scale
        ) + _cauchy_log_probs(
            observed[..., AGENT_REL_VX : AGENT_REL_VX + 2], phantom, self._clutter_velocity_scale
        )

    def _slot_detection_log_probs(self, present: Tensor, reported: Tensor) -> Tensor:
        # The MDP arm's counterpart to the presence layer's detection model, and the torch
        # mirror of the scalar `_slot_detection_log_prob`. Without it presence is read from
        # the state and never from the observation, so a particle with empty slots pays
        # nothing for an observation full of traffic. The kinematics of a slot only one side
        # fills are not scored on top: one of the two numbers is a placeholder zero.
        return _cell_probability_log_probs(
            self._detected_probabilities(present.to(self.dtype)), reported
        )

    @staticmethod
    def _masked_square_error(
        observed: Tensor, agents: Tensor, columns: slice, keep: Tensor
    ) -> Tensor:
        """Squared error over a slot's columns, summed over the kept slots only."""
        deviation = observed[..., columns] - agents[..., columns]
        per_slot = (deviation**2).sum(dim=-1)
        return torch.where(keep, per_slot, torch.zeros_like(per_slot)).sum(dim=1)

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
            on the rare row large enough to overflow. In POMDP mode the detection flags are
            already binary, so the quantization is the identity over them; every other entry
            is continuous and is floored to the resolution, which buckets readings a fraction
            of a unit apart onto one key. Curvature is in 1/m and runs to 0.05 on this
            circuit, so at the default resolution of 1.0 it quantizes to a single bucket —
            raise the resolution, or accept that the channel does not separate tree nodes.
        """
        quantized = torch.floor(observations / self._obs_resolution).to(torch.int64)
        return (quantized * self._obs_hash).sum(dim=1)


def _gaussian_log_prob(square_error: Tensor, count: Tensor, std: float) -> Tensor:
    """Diagonal-Gaussian log-density from a summed squared error and a per-row term count."""
    variance = std**2
    return -0.5 * square_error / variance - 0.5 * count * math.log(2.0 * math.pi * variance)


def _cauchy_log_probs(values: Tensor, rows: Tensor, scale: float) -> Tensor:
    """Zero-median Cauchy log-density per batch row, summed over the selected slots."""
    per_entry = -math.log(math.pi * scale) - torch.log1p((values / scale) ** 2)
    return torch.where(rows[..., None], per_entry, torch.zeros_like(per_entry)).sum(dim=(-2, -1))


def _cell_probability_log_probs(probabilities: Tensor, observed: Tensor) -> Tensor:
    """Bernoulli log-likelihood per row of an observed boolean grid under cell probabilities."""
    clipped = probabilities.clamp(CELL_PROB_EPS, 1.0 - CELL_PROB_EPS)
    return torch.where(observed, torch.log(clipped), torch.log1p(-clipped)).sum(dim=1)


__all__ = [
    "CurvatureSource",
    "ObservedCurvature",
    "RacetrackVectorizedModel",
    "TrackMapCurvature",
]
