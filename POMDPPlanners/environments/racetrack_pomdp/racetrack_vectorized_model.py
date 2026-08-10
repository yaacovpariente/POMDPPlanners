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

**One torch model, two curvature sources.** The scalar side is abstract in exactly one
place — :meth:`RacetrackModelPOMDP._curvature_for`, which answers where the road bends under
each particle — and its two subclasses answer it from a track map and from the observation
respectively. That difference is a single lookup, so it is a *parameter* here rather than a
second module: :class:`RacetrackVectorizedModel` takes a curvature source, resolves the
right one off the scalar model it is built from, and is otherwise one implementation. Two
torch files differing in one line would drift.

The map lookup is written in torch — a ``searchsorted`` over the segment starts — rather
than by calling
:meth:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry.TrackGeometry.curvature_at`
and converting. That method is NumPy, so reusing it would mean a device round-trip *per
substep*, which on CUDA is a synchronisation in the middle of the hot loop and defeats the
point of batching. The profile is three short arrays of constants, so mirroring the lookup
costs four lines and no accuracy: both sides take ``searchsorted(..., right) - 1`` on the
same floored modulo.

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
    with IDM, and collisions use a centre-distance circle rather than an oriented rectangle.

The on-road layer is parameterised the same way and dispatched on the same signal, because
it is the same question asked twice: a model that reads the road out of its observations is
exactly the model that has to predict the road back in order to score them, while a model
holding a map gains nothing by rendering a layer every particle would agree on.

Classes:
    RacetrackVectorizedModel: Batched torch counterpart of ``RacetrackModelPOMDP``.
"""

import math
import numbers
from typing import Callable, Optional, Protocol, Tuple

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp import RacetrackModelPOMDP

# Imported from the mapless model rather than copied, deliberately. The torch rasteriser has
# to sample the predicted centreline at exactly the points the scalar one does, or the two
# layers disagree on the cells at either end of the sweep -- a difference the parity test
# would catch, but only after someone had already shipped two definitions of the same road.
from POMDPPlanners.environments.racetrack_pomdp.racetrack_observed_track_model import (
    LANE_SAMPLE_REACH_M,
    LANE_SAMPLE_STEP_M,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_PRESENT,
    AGENT_REL_VX,
    AGENT_REL_X,
    AGENT_REL_Y,
    AGENT_SLOT_WIDTH,
    EGO_ANG,
    EGO_ARCLENGTH_M,
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
from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import TrackGeometry

# Given the ego block at the start of a substep, ``[N, EGO_STATE_WIDTH]``, return the signed
# curvature in 1/m under each row, ``[N]``. The torch counterpart of the scalar model's
# ``_curvature_for``, and the one thing that differs between a planner with a map and one
# without.
CurvatureSource = Callable[[Tensor], Tensor]

# The public attribute a mapless scalar model exposes its per-step estimate on. Read by name
# rather than by class, so a model does not have to be one of the two shipped ones.
_CURVATURE_ESTIMATE_ATTR = "curvature_estimate"


class RoadLayer(Protocol):
    """What the model predicts the observation's on-road layer looks like, and its weight.

    The second thing the scalar side leaves to its subclass. A model with a map does not
    bother predicting the road; a model that reads the road out of its observations has to,
    because that is what it scores them against.
    """

    def render(self, states: Tensor) -> Tensor:
        """Predicted on-road layer per row, ``[N, 144]``, in the model's floating dtype."""
        ...  # pylint: disable=unnecessary-ellipsis

    def log_probs(self, states: Tensor, observations: Tensor) -> Tensor:
        """Contribution of the on-road layer to each row's log-likelihood, ``[N]``."""
        ...  # pylint: disable=unnecessary-ellipsis


# Mirrors the scalar model: process noise touches the first six ego entries only, stopping
# short of arclength. Arclength is integrated from the along-track rate, so jittering it
# would teleport a particle to a different part of the circuit rather than blur its pose.
_EGO_NOISE_WIDTH = EGO_ARCLENGTH_M

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


class TrackMapCurvature:
    """Torch mirror of ``TrackGeometry.curvature_at``: a table lookup by arclength.

    Holds the profile as device tensors so the whole substep loop stays on one device. The
    NumPy original is not called here on purpose — see the module docstring — but the two
    agree by construction: the same floored modulo, the same ``searchsorted(..., right) - 1``
    and the same clamp, so a rollout on the map cannot diverge between the two models.

    Attributes:
        total_length_m: Length of one lap in metres, the modulus the arclength wraps on.
    """

    def __init__(self, geometry: TrackGeometry, device: torch.device, dtype: torch.dtype) -> None:
        """Move a curvature profile onto a device.

        Args:
            geometry: The lap's piecewise-constant curvature profile.
            device: Device the lookup's tensors live on.
            dtype: Floating dtype of the returned curvature.
        """
        # The starts are held in the *model's* dtype, matching the arclength they are
        # compared against, and that is deliberate. In a float32 model the arclength itself
        # is already rounded -- a boundary like 372.2208 m is off by up to 1.1e-5 m before
        # the lookup sees it -- so holding the starts in float64 cannot recover the
        # intended segment, and measurably makes it worse: on the shipped circuit's nine
        # boundaries, float32 starts agree with NumPy on all nine and float64 starts on
        # five. Matching dtypes is what makes a boundary compare equal.
        self._starts = torch.as_tensor(
            np.asarray(geometry.segment_starts, dtype=np.float64), dtype=dtype, device=device
        )
        self._curvatures = torch.as_tensor(
            np.asarray(geometry.segment_curvatures, dtype=np.float64), dtype=dtype, device=device
        )
        self.total_length_m = float(geometry.total_length_m)

    def __call__(self, ego: Tensor) -> Tensor:
        distance = torch.remainder(ego[:, EGO_ARCLENGTH_M], self.total_length_m)
        index = torch.searchsorted(self._starts, distance.contiguous(), right=True) - 1
        return self._curvatures[index.clamp_(0, self._curvatures.shape[0] - 1)]


class ObservedCurvature:
    """A single estimated curvature, refreshed each real step and shared by every row.

    The counterpart of :class:`TrackMapCurvature` for a planner with no map, which estimates
    one curvature per step from what it can see and has nothing better to assume further
    ahead. The value is read through a callable rather than copied in, because the estimate
    is replaced on every real step while this model is built once: caching it would freeze
    the planner on the corner it happened to start in.
    """

    def __init__(self, read_curvature: Callable[[], float]) -> None:
        """Wrap a live per-step curvature estimate.

        Args:
            read_curvature: Zero-argument callable returning the current estimate in 1/m.
        """
        self._read_curvature = read_curvature

    def __call__(self, ego: Tensor) -> Tensor:
        return torch.full_like(ego[:, EGO_ARCLENGTH_M], float(self._read_curvature()))


class AllRoadLayer:
    """The base model's on-road layer: drivable everywhere, and worth nothing in a weight.

    All-ones is the honest answer for a model with no picture of the road, and because it
    does not depend on the state it is identical across every particle — so its likelihood
    term shifts all the log-weights alike and vanishes at normalisation. Returning zero is
    not a shortcut; it is the same number, without 144 cells of arithmetic per particle.
    """

    def __init__(self, dtype: torch.dtype, device: torch.device) -> None:
        """Record the tensor kind the rendered layers should come back as."""
        self._dtype = dtype
        self._device = device

    def render(self, states: Tensor) -> Tensor:
        return torch.ones(states.shape[0], _GRID_CELL_COUNT, dtype=self._dtype, device=self._device)

    def log_probs(self, states: Tensor, observations: Tensor) -> Tensor:
        del observations
        return torch.zeros(states.shape[0], dtype=self._dtype, device=self._device)


class LaneCorridorLayer:
    """The mapless model's on-road layer: its own lane centreline, drawn from one estimate.

    One lane, not the whole road — the model has no idea how many lanes the circuit has, so
    it draws the only piece of road it can locate, the centreline it measures its own offset
    from. Unlike :class:`AllRoadLayer` this *does* depend on the state, through the ego's
    lateral offset and lane-relative angle, so it separates particles and its likelihood has
    to be scored rather than dropped.
    """

    def __init__(
        self,
        read_curvature: Callable[[], float],
        cell_flip_prob: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Build the corridor rasteriser.

        Args:
            read_curvature: Zero-argument callable returning the current estimate in 1/m.
            cell_flip_prob: Per-cell probability that an on-road bit is read wrong.
            dtype: Floating dtype of the rendered layers.
            device: Device the rendered layers live on.
        """
        self._read_curvature = read_curvature
        self._flip_prob = cell_flip_prob
        self._dtype = dtype
        self._device = device
        # Built in NumPy with the scalar model's own arange arguments, then moved once. The
        # sample points have to land on exactly the same cells as the scalar rasteriser's,
        # and re-deriving the sequence in torch risks a different final element.
        self._arclength = torch.as_tensor(
            np.arange(
                -LANE_SAMPLE_REACH_M,
                LANE_SAMPLE_REACH_M + LANE_SAMPLE_STEP_M,
                LANE_SAMPLE_STEP_M,
            ),
            dtype=dtype,
            device=device,
        )

    def render(self, states: Tensor) -> Tensor:
        lateral = states[:, EGO_LAT][:, None]
        angle = states[:, EGO_ANG][:, None]
        curvature = float(self._read_curvature())
        # Centreline in the lane frame relative to the ego, then rotated into the body frame
        # by the ego's lane-relative angle. A centreline the ego sits ``lateral`` metres from
        # appears at ``-lateral`` across-track.
        lane_across = 0.5 * curvature * self._arclength**2 - lateral
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        body_along = cos_a * self._arclength + sin_a * lane_across
        body_across = -sin_a * self._arclength + cos_a * lane_across
        return self._scatter_cells(body_along, body_across)

    def log_probs(self, states: Tensor, observations: Tensor) -> Tensor:
        start = ON_ROAD_LAYER * _GRID_CELL_COUNT
        observed = observations[:, start : start + _GRID_CELL_COUNT] > 0.5
        return _bernoulli_cell_log_probs(
            self.render(states) > 0.5, observed, self._flip_prob, self._dtype
        )

    def _scatter_cells(self, body_along: Tensor, body_across: Tensor) -> Tensor:
        """Mark the cell each sample point falls in, dropping the ones off the window."""
        rows = torch.floor((body_along + GRID_HALF_EXTENT_M) / GRID_STEP_M)
        columns = torch.floor((body_across + GRID_HALF_EXTENT_M) / GRID_STEP_M)
        inside = (rows >= 0.0) & (rows < GRID_CELLS) & (columns >= 0.0) & (columns < GRID_CELLS)
        cell = (rows * GRID_CELLS + columns).to(torch.int64)
        # One column wider than the grid, used as a bin for the sample points that fall
        # outside it. Unlike the presence grid there is no always-set cell to redirect them
        # onto, so the extra column is sliced off at the end instead.
        sink = torch.full_like(cell, _GRID_CELL_COUNT)
        flat = torch.zeros(
            body_along.shape[0], _GRID_CELL_COUNT + 1, dtype=self._dtype, device=self._device
        )
        flat.scatter_(1, torch.where(inside, cell, sink), 1.0)
        return flat[:, :_GRID_CELL_COUNT]


def _bernoulli_cell_log_probs(
    predicted: Tensor, observed: Tensor, flip_prob: float, dtype: torch.dtype
) -> Tensor:
    """Independent per-cell flip model over two boolean ``[N, cells]`` grids."""
    agreements = (observed == predicted).sum(dim=1).to(dtype)
    disagreements = float(predicted.shape[1]) - agreements
    return agreements * math.log1p(-flip_prob) + disagreements * math.log(flip_prob)


def _resolve_curvature_source(
    env: RacetrackModelPOMDP, device: torch.device, dtype: torch.dtype
) -> CurvatureSource:
    """Pick the torch curvature lookup matching the scalar model's own ``_curvature_for``.

    Dispatches on what the model exposes rather than on its class, so a third subclass with
    a track map or a per-step estimate works without editing this module. Anything else is
    rejected rather than defaulted to zero curvature: a silent straight-line model would
    still run, still look plausible, and drive through every corner.
    """
    geometry = getattr(env, "track_geometry", None)
    if isinstance(geometry, TrackGeometry):
        return TrackMapCurvature(geometry, device, dtype)
    if _has_curvature_estimate(env):
        return ObservedCurvature(lambda: float(getattr(env, _CURVATURE_ESTIMATE_ATTR)))
    raise ValueError(
        f"Cannot infer a curvature source from {type(env).__name__}: it exposes neither a "
        f"'track_geometry' nor a '{_CURVATURE_ESTIMATE_ATTR}'. Pass curvature_source= "
        f"explicitly. Defaulting to zero curvature would give a model that runs, looks "
        f"plausible, and drives straight through every corner."
    )


def _resolve_road_layer(
    env: RacetrackModelPOMDP, device: torch.device, dtype: torch.dtype
) -> RoadLayer:
    """Pick the on-road layer matching the scalar model's ``_render_on_road_layer``.

    Dispatched on the same signal as the curvature, and for the same reason: a model that
    reads the road out of its observations is exactly the model that has to predict the road
    back to score them, while a model holding a map gains nothing by rendering a layer every
    particle would agree on.
    """
    if _has_curvature_estimate(env):
        return LaneCorridorLayer(
            lambda: float(getattr(env, _CURVATURE_ESTIMATE_ATTR)),
            float(np.clip(env.cell_flip_prob, _FLIP_PROB_EPS, 1.0 - _FLIP_PROB_EPS)),
            dtype,
            device,
        )
    return AllRoadLayer(dtype, device)


def _has_curvature_estimate(env: RacetrackModelPOMDP) -> bool:
    # numbers.Real rather than float, so an estimate held as a NumPy scalar -- the natural
    # thing to fall out of a fit over the on-road layer -- is recognised too.
    return isinstance(getattr(env, _CURVATURE_ESTIMATE_ATTR, None), numbers.Real)


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
        curvature_source: Where each substep's curvature comes from — a track map, a live
            per-step estimate, or whatever the caller supplied.
        road_layer: What the model predicts the observation's on-road layer looks like, and
            what that prediction is worth in a weight.

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
        road_layer: Optional[RoadLayer] = None,
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
            curvature_source: Torch counterpart of the scalar model's ``_curvature_for``,
                mapping an ego block to a per-row curvature. Defaults to None, which reads
                a track map or a live per-step estimate off ``env``.
            road_layer: Torch counterpart of the scalar model's ``_render_on_road_layer``
                and ``_on_road_log_prob``. Defaults to None, which picks the corridor
                render for a mapless model and the all-ones layer for every other.

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
        self.state_dim = EGO_STATE_WIDTH + self._num_agents * AGENT_SLOT_WIDTH
        self.observation_dim = self._observation_width()
        self.curvature_source: CurvatureSource = (
            curvature_source
            if curvature_source is not None
            else _resolve_curvature_source(env, self.device, dtype)
        )
        self.road_layer: RoadLayer = (
            road_layer if road_layer is not None else _resolve_road_layer(env, self.device, dtype)
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
        # The on-road layer is drawn without flip noise, as the scalar model draws it.
        occupancy[:, ON_ROAD_LAYER] = self.road_layer.render(next_states)
        return occupancy.reshape(next_states.shape[0], -1)

    def _occupancy_log_probs(self, next_states: Tensor, observations: Tensor) -> Tensor:
        grid = self._render_presence_grid(next_states)
        start = PRESENCE_LAYER * _GRID_CELL_COUNT
        observed = observations[:, start : start + _GRID_CELL_COUNT] > 0.5
        presence = _bernoulli_cell_log_probs(grid, observed, self._cell_flip_prob, self.dtype)
        return presence + self.road_layer.log_probs(next_states, observations)

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


__all__ = [
    "AllRoadLayer",
    "CurvatureSource",
    "LaneCorridorLayer",
    "ObservedCurvature",
    "RacetrackVectorizedModel",
    "RoadLayer",
    "TrackMapCurvature",
]
