# SPDX-License-Identifier: MIT

"""Planner-side generative model for the racetrack POMDP.

:class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp.RacetrackPOMDP` is a
forward-only *world*: it advances one true state per interaction and has no densities. A
planner instead carries this model on ``policy.environment``. It samples transitions from
an arbitrary state, scores observations against a state, and supplies the same reward the
world scores — :func:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_schema.racetrack_reward`
is called by both, so the planner cannot be optimising a different objective.

The transition reproduces highway-env's kinematic bicycle **exactly** for the ego: the same
``arctan(tan(delta)/2)`` slip angle, the same ``LENGTH / 2`` wheelbase term, the same update
order, integrated over ``substeps`` sub-intervals of ``dt / substeps`` so the model takes
the same number of physics steps per decision as the simulator. The Frenet pair
``(lat, ang)`` is integrated alongside it against the curvature of the road under the ego.

**Where the road bends is the subclass's job.** This class is abstract for exactly one
reason: it does not know the track. Every substep asks :meth:`RacetrackModelPOMDP._curvature_for`
what the curvature is under each particle, and the two shipped subclasses answer it from
two different places —
:class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_known_track_model.KnownTrackModel`
looks the circuit up by arclength, and
:class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_observed_track_model.ObservedTrackModel`
reads it out of the observation's on-road layer. Curvature is deliberately *not* a state
slot: it is a property of the road, so freezing it in the state would encode a prediction
rather than a fact, and a rollout reusing one frozen value drives straight through every
corner.

**The model error is deliberate and lives in the other vehicles.** Agent slots are
propagated as constant-velocity drift in the ego body frame, while the world drives them
with IDM. A planner that treats its own model as truth will therefore mispredict exactly
what partial observability is supposed to punish.

This module imports **nothing** from ``highway_env`` — it is pure NumPy, so the model can be
constructed, pickled and tested on a machine without the simulator.

Note:
    Headings here are wrapped to ``[-pi, pi)``; the world lets highway-env's heading
    accumulate past a full turn instead. The two agree geometrically and differ by a
    multiple of ``2 * pi`` once the ego has driven far enough round the loop, so compare a
    model heading with a world heading modulo ``2 * pi`` rather than by subtraction.

Classes:
    RacetrackModelPOMDP: Abstract generative model paired with the forward-only world.
"""

from abc import abstractmethod
from collections.abc import Hashable
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import DiscreteActionsEnvironment, SpaceInfo, SpaceType
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_PRESENT,
    AGENT_REL_VX,
    AGENT_REL_X,
    AGENT_REL_Y,
    AGENT_SLOT_WIDTH,
    DEFAULT_ACTION_PRESETS,
    DEFAULT_ACTION_REWARD,
    DEFAULT_COLLISION_REWARD,
    DEFAULT_LANE_CENTERING_COST,
    DEFAULT_LANE_CENTERING_REWARD,
    DEFAULT_MAX_TRACKED_AGENTS,
    DEFAULT_SPEED_LIMIT,
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
    racetrack_reward,
    rotate,
    state_agent_rows,
)

# Process noise is applied to the first six ego entries only. Arclength is the seventh and
# is excluded on purpose: it is not an independent coordinate but the running integral of
# the ego's own along-track motion, already noisy through the speed it is integrated from.
# Jittering it on top would teleport a particle down the track and, worse, re-index the
# curvature profile a known-track model reads with it.
_EGO_NOISE_WIDTH = EGO_ARCLENGTH_M

# The ego sits in the middle cell of the occupancy grid, at (6, 6) for the shipped 12x12.
_GRID_CENTRE = GRID_CELLS // 2

# Floor for the Frenet denominator ``1 - curvature * lat``, which vanishes at the centre of
# curvature and would otherwise divide by zero on a tight arc.
_MIN_FRENET_DENOMINATOR = 1e-3

# Keeps a flip probability of exactly 0 or 1 from turning a single mismatched cell into a
# -inf log-likelihood, which would annihilate an otherwise good particle.
_FLIP_PROB_EPS = 1e-12

OCCUPANCY_KEY = "occupancy"
_EGO_KEY = "ego"
_AGENTS_KEY = "agents"


def _wrap_to_pi_array(angles: np.ndarray) -> np.ndarray:
    """Array form of the schema's scalar ``wrap_to_pi``, for the vectorised hot path."""
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def _rotate_per_row(vectors: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rotate ``(N, K, 2)`` vectors counter-clockwise by a per-row angle.

    The schema's ``rotate`` takes one scalar angle for the whole array; here the rotation is
    the ego's own yaw increment, which differs per particle because their speeds differ.
    """
    cos_a = np.cos(angles)[:, None]
    sin_a = np.sin(angles)[:, None]
    x_component = vectors[..., 0]
    y_component = vectors[..., 1]
    return np.stack(
        [
            cos_a * x_component - sin_a * y_component,
            sin_a * x_component + cos_a * y_component,
        ],
        axis=-1,
    )


def _gaussian_log_prob(deviation: np.ndarray, std: float) -> float:
    """Log-density of a zero-mean isotropic Gaussian at ``deviation``, summed over entries."""
    variance = float(std) ** 2
    flat = np.asarray(deviation, dtype=float).reshape(-1)
    return float(
        -0.5 * np.sum(flat**2) / variance - 0.5 * flat.size * np.log(2.0 * np.pi * variance)
    )


class RacetrackModelPOMDP(DiscreteActionsEnvironment):
    """Abstract generative racetrack model: the planner's beliefs about the world.

    Reproduces the world's ego dynamics exactly and its other vehicles only crudely (see the
    module docstring). The observation follows whichever arm of the matched pair the world is
    running, selected by ``observation_mode``; :meth:`encode_observation` is the single seam
    where the world's raw reading enters, and every other observation method works in the
    encoded space.

    Subclasses supply :meth:`_curvature_for`, the one thing this class does not know: where
    the road bends. They may also override :meth:`_render_on_road_layer` and
    :meth:`_on_road_log_prob` if their curvature source lets them predict the observation's
    on-road layer; the defaults here decline to, and say why.

    Attributes:
        observation_mode: Which arm of the matched pair this model scores.
        dt: Seconds per decision; must equal ``1 / policy_frequency`` in the world.
        substeps: Physics sub-intervals per decision; must equal the world's
            ``simulation_frequency / policy_frequency``.
        action_presets: Normalised ``(acceleration, steering)`` commands; an action is an
            index into this sequence.
        max_tracked_agents: Number of fixed agent slots in the state.
        collision_distance: Range (m) at or below which a present agent slot is a collision.
        lane_half_width: Lateral offset (m) beyond which the ego has left the lane.

    Note:
        This is an abstract base class and cannot be instantiated directly. Use
        :class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_known_track_model.KnownTrackModel`
        or
        :class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_observed_track_model.ObservedTrackModel`.
    """

    def __init__(
        self,
        discount_factor: float,
        observation_mode: ObservationMode = ObservationMode.POMDP,
        dt: float = 0.2,
        substeps: int = 3,
        action_presets: Optional[Sequence[Tuple[float, float]]] = None,
        max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
        vehicle_length: float = 5.0,
        lane_half_width: float = 2.5,
        collision_distance: float = 4.0,
        collision_reward: float = DEFAULT_COLLISION_REWARD,
        lane_centering_cost: float = DEFAULT_LANE_CENTERING_COST,
        lane_centering_reward: float = DEFAULT_LANE_CENTERING_REWARD,
        action_reward: float = DEFAULT_ACTION_REWARD,
        cell_flip_prob: float = 0.05,
        ego_pose_std: float = 0.5,
        agent_pose_std: float = 1.0,
        agent_velocity_std: float = 2.0,
        process_noise_std: float = 0.05,
        name: Optional[str] = None,
        reward_range: Optional[Tuple[float, float]] = None,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ) -> None:
        """Initialize the racetrack generative model.

        Args:
            discount_factor: Discount factor for future rewards (0 < d <= 1).
            observation_mode: Which arm of the matched pair to model. Defaults to POMDP.
            dt: Seconds per decision. Defaults to 0.2, matching a policy frequency of 5.
            substeps: Physics sub-intervals per decision. Defaults to 3, matching a
                simulation frequency of 15.
            action_presets: Normalised ``(acceleration, steering)`` commands. Defaults to
                the shared 3x3 throttle-by-steer grid.
            max_tracked_agents: Fixed agent slots in the state. Defaults to 4.
            vehicle_length: Ego length (m); highway-env uses half of it as the wheelbase.
                Defaults to 5.0.
            lane_half_width: Lateral offset (m) beyond which the ego is off the lane.
                Defaults to 2.5.
            collision_distance: Range (m) to a present agent slot counted as a collision.
                Defaults to 4.0.
            collision_reward: Weight applied to a collision. Defaults to -1.0.
            lane_centering_cost: Sharpness of the lane-centering falloff. Defaults to 4.0.
            lane_centering_reward: Weight on the lane-centering term. Defaults to 1.0.
            action_reward: Weight on the control-effort penalty. Defaults to -0.3.
            cell_flip_prob: Per-cell probability an occupancy presence bit is wrong.
                Defaults to 0.05.
            ego_pose_std: Observation noise (m and m/s) on the MDP ego row. Defaults to 0.5.
            agent_pose_std: Observation noise (m) on an MDP agent's relative position.
                Defaults to 1.0.
            agent_velocity_std: Observation noise (m/s) on an MDP agent's relative velocity.
                Defaults to 2.0.
            process_noise_std: Transition noise on the ego block, which keeps a particle
                filter from collapsing onto one trajectory. Defaults to 0.05.
            name: Environment identifier. Defaults to the class name plus the mode.
            reward_range: Optional ``(min, max)`` reward bounds.
            output_dir: Optional directory for logging output.
            debug: Enable debug logging.
            use_queue_logger: Whether to use queue-based logging.

        Raises:
            ValueError: If ``max_tracked_agents`` or ``substeps`` is below 1, or ``dt`` is
                not positive.
        """
        if max_tracked_agents < 1:
            raise ValueError(f"max_tracked_agents must be at least 1, got {max_tracked_agents}.")
        if substeps < 1:
            raise ValueError(f"substeps must be at least 1, got {substeps}.")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")

        self.observation_mode = observation_mode
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.action_presets: Tuple[Tuple[float, float], ...] = tuple(
            (float(acceleration), float(steering))
            for acceleration, steering in (
                action_presets if action_presets is not None else DEFAULT_ACTION_PRESETS
            )
        )
        self.max_tracked_agents = max_tracked_agents
        self.vehicle_length = float(vehicle_length)
        self.lane_half_width = float(lane_half_width)
        self.collision_distance = float(collision_distance)
        self.collision_reward = collision_reward
        self.lane_centering_cost = lane_centering_cost
        self.lane_centering_reward = lane_centering_reward
        self.action_reward = action_reward
        self.cell_flip_prob = float(cell_flip_prob)
        self.ego_pose_std = float(ego_pose_std)
        self.agent_pose_std = float(agent_pose_std)
        self.agent_velocity_std = float(agent_velocity_std)
        self.process_noise_std = float(process_noise_std)

        super().__init__(
            discount_factor=discount_factor,
            name=name if name is not None else f"{type(self).__name__}-{observation_mode.value}",
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE,
                observation_space=SpaceType.CONTINUOUS,
            ),
            reward_range=reward_range,
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

    @property
    def state_width(self) -> int:
        """Width of the state vector, ego block plus the fixed agent slots."""
        return EGO_STATE_WIDTH + self.max_tracked_agents * AGENT_SLOT_WIDTH

    def get_actions(self) -> List[int]:
        """Indices into :attr:`action_presets`, the shared world/model action vocabulary."""
        return list(range(len(self.action_presets)))

    def hash_action(self, action: Any) -> Hashable:
        return int(action)

    # ── Transition (highway-env's kinematic bicycle, reproduced) ─────────
    @abstractmethod
    def _curvature_for(self, ego: np.ndarray) -> np.ndarray:
        """Signed curvature in 1/m for each row of the ego block, shape ``(B,)``.

        Called once per integration substep with the ego block ``(B, EGO_STATE_WIDTH)`` as
        it stands *at the start of that substep*, so a subclass indexing by arclength sees
        the arclength the ego has actually reached rather than the one it started the
        decision at.

        Args:
            ego: The ego block of the particle batch, shape ``(B, EGO_STATE_WIDTH)``.

        Returns:
            Signed curvature in 1/m per row, positive in the same sense as
            :class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry.TrackGeometry`.

        Note:
            Subclasses must implement this. It is the only thing separating a model that
            knows the circuit from one that has to read the road out of its observations.
        """

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        """Propagate ``state`` one decision forward under ``action``, with process noise.

        Args:
            state: A state vector of width :attr:`state_width`.
            action: An index into :attr:`action_presets`.
            n_samples: Number of independent draws. Defaults to 1.

        Returns:
            Shape ``(state_width,)`` when ``n_samples == 1``, else
            ``(n_samples, state_width)``.
        """
        mean = self._propagate(state, action)
        if n_samples == 1:
            return self._perturb(mean.reshape(1, -1))[0]
        return self._perturb(np.tile(mean, (n_samples, 1)))

    def sample_next_state_batch(self, states: Any, action: Any) -> np.ndarray:
        """Propagate N particles one decision forward under one shared action.

        Vectorised over the particle axis with no Python loop over rows: this is the
        particle filter's hot path.

        Args:
            states: Array-like of shape ``(N, state_width)``.
            action: An index into :attr:`action_presets`.

        Returns:
            ``np.ndarray`` of shape ``(N, state_width)``.
        """
        propagated = self._propagate_batch(np.asarray(states, dtype=float), action)
        return self._perturb(propagated)

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        """Log-density of each candidate successor under the noisy propagation.

        The density covers the first six ego entries only. Arclength and every agent slot
        are *deterministic* functions of ``(state, action)`` — no noise is added to them — so
        including them would contribute a constant at best and an infinite mismatch penalty
        at worst.

        Strictly the transition is a delta on those deterministic coordinates, so a
        candidate that disagrees there has probability zero rather than the finite value
        returned here. It is left finite deliberately: every candidate a filter scores is a
        draw from this same sampler, so the deterministic block always agrees by
        construction, and a ``-inf`` guarded by a bitwise float comparison would annihilate
        an entire particle set over a rounding difference — a far worse failure than the
        impropriety it fixes.

        Args:
            state: The state the transition starts from.
            action: An index into :attr:`action_presets`.
            next_states: One state vector, or an array-like of ``N`` of them.

        Returns:
            ``np.ndarray`` of shape ``(N,)`` of log-densities.

        Raises:
            ValueError: If ``process_noise_std`` is zero. The transition is then a point
                mass with no density, and scoring it would divide by zero and hand the
                caller a silent NaN weight.
        """
        if self.process_noise_std <= 0.0:
            raise ValueError(
                "transition_log_probability needs process_noise_std > 0: a noise-free "
                "transition is a point mass with no density. Build the model with process "
                "noise for belief updates, and keep the zero-noise setting for tests and "
                "open-loop prediction."
            )
        mean = self._propagate(state, action)[:_EGO_NOISE_WIDTH]
        candidates = np.atleast_2d(np.asarray(next_states, dtype=float))[:, :_EGO_NOISE_WIDTH]
        variance = self.process_noise_std**2
        normaliser = 0.5 * _EGO_NOISE_WIDTH * np.log(2.0 * np.pi * variance)
        deviation = candidates - mean
        return -0.5 * np.sum(deviation**2, axis=1) / variance - normaliser

    def _propagate(self, state: Any, action: Any) -> np.ndarray:
        return self._propagate_batch(np.asarray(state, dtype=float).reshape(1, -1), action)[0]

    def _propagate_batch(self, states: np.ndarray, action: Any) -> np.ndarray:
        acceleration_norm, steering_norm = self.action_presets[int(action)]
        slip = float(np.arctan(0.5 * np.tan(steering_norm * MAX_STEERING_RAD)))
        acceleration = acceleration_norm * MAX_ACCELERATION_MPS2
        step = self.dt / self.substeps

        ego = np.array(states[:, :EGO_STATE_WIDTH], dtype=float)
        agents = np.array(states[:, EGO_STATE_WIDTH:], dtype=float).reshape(
            (len(states), self.max_tracked_agents, AGENT_SLOT_WIDTH)
        )
        for _ in range(self.substeps):
            self._integrate_substep(ego, agents, slip, acceleration, step)
        return np.concatenate([ego, agents.reshape((len(states), -1))], axis=1)

    def _integrate_substep(
        self,
        ego: np.ndarray,
        agents: np.ndarray,
        slip: float,
        acceleration: float,
        step: float,
    ) -> None:
        """One explicit-Euler sub-interval, mutating ``ego`` and ``agents`` in place."""
        # Every derivative is evaluated at the start of the sub-interval, including the
        # speed — that is what highway-env does, and it is what makes ``ang`` track
        # ``heading`` exactly on a straight lane.
        speed = ego[:, EGO_SPEED].copy()
        heading = ego[:, EGO_HEADING].copy()
        lateral = ego[:, EGO_LAT].copy()
        angle = ego[:, EGO_ANG].copy()
        curvature = self._curvature_for(ego)

        yaw_rate = speed * np.sin(slip) / (self.vehicle_length / 2.0)
        # The Frenet rates use the *velocity* direction, ``ang + slip``, not the heading:
        # the bicycle's centre of mass moves at the slip angle off its own nose, exactly as
        # the world-frame update above already does. Dropping the slip here costs ~2 m of
        # lateral error over ten steps of full-lock steering; including it makes the model's
        # lane offset match the simulator's measured one to float precision, which the
        # live-simulator test pins down.
        drift = angle + slip
        denominator = np.maximum(1.0 - curvature * lateral, _MIN_FRENET_DENOMINATOR)
        along_rate = speed * np.cos(drift) / denominator

        ego[:, EGO_X] += speed * np.cos(heading + slip) * step
        ego[:, EGO_Y] += speed * np.sin(heading + slip) * step
        ego[:, EGO_HEADING] = _wrap_to_pi_array(heading + yaw_rate * step)
        ego[:, EGO_LAT] = lateral + speed * np.sin(drift) * step
        ego[:, EGO_ANG] = _wrap_to_pi_array(angle + (yaw_rate - curvature * along_rate) * step)
        ego[:, EGO_SPEED] = speed + acceleration * step
        # Advanced by the *along-track* rate, not by ``speed * step``: the two differ once
        # the ego is yawed relative to the lane or offset from its centreline on an arc,
        # and it is the along-track one that indexes the curvature profile correctly.
        ego[:, EGO_ARCLENGTH_M] = ego[:, EGO_ARCLENGTH_M] + along_rate * step

        self._drift_agents(agents, -yaw_rate * step, step)

    @staticmethod
    def _drift_agents(agents: np.ndarray, rotation: np.ndarray, step: float) -> None:
        """Carry constant-velocity agents through one sub-interval of ego body-frame motion."""
        positions = agents[:, :, AGENT_REL_X : AGENT_REL_X + 2]
        velocities = agents[:, :, AGENT_REL_VX : AGENT_REL_VX + 2]
        drifted = positions + velocities * step
        # Absent slots are all-zero and stay all-zero: rotating the origin is the origin.
        agents[:, :, AGENT_REL_X : AGENT_REL_X + 2] = _rotate_per_row(drifted, rotation)
        agents[:, :, AGENT_REL_VX : AGENT_REL_VX + 2] = _rotate_per_row(velocities, rotation)

    def _perturb(self, states: np.ndarray) -> np.ndarray:
        if self.process_noise_std <= 0.0:
            return states
        states[:, :_EGO_NOISE_WIDTH] += np.random.normal(
            scale=self.process_noise_std, size=(len(states), _EGO_NOISE_WIDTH)
        )
        return states

    # ── Reward and termination ───────────────────────────────────────────
    @property
    def reward_requires_next_state(self) -> bool:
        """True: the reward reads the successor's lane offset and collision state."""
        return True

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        """Score a transition with the same function the world scores it with.

        Scored on the *resulting* state's lateral offset, as the world does; when
        ``next_state`` is None the deterministic propagation stands in for it.

        Args:
            state: The state the step was taken from.
            action: An index into :attr:`action_presets`.
            next_state: The realised successor, or None to propagate one.

        Returns:
            The scalar reward, zero once the ego has left the lane.
        """
        resulting = np.asarray(
            next_state if next_state is not None else self._propagate(state, action), dtype=float
        )
        lateral = float(resulting[EGO_LAT])
        return racetrack_reward(
            lateral,
            self.action_presets[int(action)],
            self._has_collision(resulting),
            abs(lateral) <= self.lane_half_width,
            collision_reward=self.collision_reward,
            lane_centering_cost=self.lane_centering_cost,
            lane_centering_reward=self.lane_centering_reward,
            action_reward=self.action_reward,
        )

    def is_terminal(self, state: Any) -> bool:
        """Whether the ego has left the lane or is inside a tracked agent.

        Args:
            state: A state vector of width :attr:`state_width`.

        Returns:
            True if the lateral offset exceeds :attr:`lane_half_width`, or a present agent
            slot is within :attr:`collision_distance` of the ego.
        """
        array = np.asarray(state, dtype=float)
        if abs(float(array[EGO_LAT])) > self.lane_half_width:
            return True
        return self._has_collision(array)

    def _has_collision(self, state: np.ndarray) -> bool:
        # A centre-distance circle, where the world uses highway-env's oriented
        # rectangle intersection. The two disagree on near misses whose outcome depends
        # on relative heading and lateral offset: a 4 m centre gap is a collision here
        # and often is not there, and two vehicles nose-to-tail in adjacent lanes are the
        # reverse. That gap matters twice over, because this predicate feeds both
        # `is_terminal` and the `crashed` term of `racetrack_reward` -- so the world and
        # the model sharing one reward *function* does not by itself mean the planner is
        # optimising the world's reward. Closing it means reproducing the footprint
        # intersection with the ego's own heading, which needs an agent heading the state
        # does not currently carry.
        rows = state_agent_rows(state, self.max_tracked_agents)
        present = rows[rows[:, AGENT_PRESENT] > 0.5]
        if present.size == 0:
            return False
        ranges = np.linalg.norm(present[:, AGENT_REL_X : AGENT_REL_X + 2], axis=1)
        return bool(np.any(ranges <= self.collision_distance))

    # ── Observation ──────────────────────────────────────────────────────
    def encode_observation(self, observation: Any) -> Dict[str, np.ndarray]:
        """Encode the world's raw reading into the space the belief and planner use.

        This is the single raw-observation seam. In POMDP mode the raw ``(2, 12, 12)``
        occupancy grid is wrapped unchanged; in MDP mode the raw ``(K + 1, 5)`` table of
        absolute ``[presence, x, y, vx, vy]`` rows is split into the ego row and the other
        vehicles, and the others are moved into the ego body frame — the frame the state's
        agent slots already live in, so the model scores like against like.

        Args:
            observation: The raw observation emitted by the world, in this model's mode.

        Returns:
            POMDP mode: ``{"occupancy": (2, 12, 12) float32}``. MDP mode:
            ``{"ego": (4,) [x, y, vx, vy], "agents": (K, 5) [present, rel_x, rel_y, rel_vx,
            rel_vy]}``, with absent rows left at zero.
        """
        if self.observation_mode is ObservationMode.POMDP:
            return {OCCUPANCY_KEY: np.asarray(observation, dtype=np.float32)}
        return self._encode_kinematics(np.asarray(observation, dtype=float))

    def _encode_kinematics(self, rows: np.ndarray) -> Dict[str, np.ndarray]:
        ego = np.array(rows[0, 1:5], dtype=float)
        heading = float(np.arctan2(ego[3], ego[2]))
        others = rows[1 : self.max_tracked_agents + 1]
        present = others[:, AGENT_PRESENT] > 0.5

        block = np.zeros((len(others), AGENT_SLOT_WIDTH), dtype=float)
        block[present, AGENT_PRESENT] = 1.0
        block[present, AGENT_REL_X : AGENT_REL_X + 2] = rotate(
            others[present, 1:3] - ego[:2], -heading
        )
        block[present, AGENT_REL_VX : AGENT_REL_VX + 2] = rotate(
            others[present, 3:5] - ego[2:4], -heading
        )
        agents = np.zeros((self.max_tracked_agents, AGENT_SLOT_WIDTH), dtype=float)
        agents[: len(others)] = block
        return {_EGO_KEY: ego, _AGENTS_KEY: agents}

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        """Draw observations of ``next_state`` in this model's observation mode.

        Args:
            next_state: The state being observed.
            action: Unused; the observation depends on the state alone.
            n_samples: Number of independent draws. Defaults to 1.

        Returns:
            One observation dictionary when ``n_samples == 1``, else a list of them.
        """
        del action
        draws = [self._draw_observation(next_state) for _ in range(max(n_samples, 1))]
        return draws[0] if n_samples == 1 else draws

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        """Log-density of each observation given ``next_state``.

        In POMDP mode this is an independent Bernoulli over the 144 presence cells of the
        grid :meth:`sample_observation` would have rasterised, plus whatever
        :meth:`_on_road_log_prob` contributes for the second layer — nothing at all, unless
        the subclass can predict the road.

        In MDP mode it is a diagonal Gaussian over the ego row and the *present* agent slots;
        absent slots contribute nothing, because a slot that holds no vehicle carries no
        measurement.

        Note:
            Presence is read from the state, never from the observation, so the MDP
            likelihood cannot discriminate on *whether* a vehicle is there — only on where
            it is. A particle whose slots are empty is scored on its ego row alone and pays
            nothing for an observation full of traffic. Fixing that needs a detection model
            (a miss and false-alarm rate) rather than a hard zero: presence flags disagree
            routinely as vehicles enter and leave the tracking window, and a ``-inf`` there
            would collapse the filter every time one did.

        Args:
            next_state: The state being observed.
            action: Unused; the observation depends on the state alone.
            observations: One observation dictionary, or a sequence of ``N`` of them.

        Returns:
            ``np.ndarray`` of shape ``(N,)`` of log-densities.
        """
        del action
        candidates = [observations] if isinstance(observations, dict) else list(observations)
        if self.observation_mode is ObservationMode.POMDP:
            grid = self._render_presence_grid(next_state)
            return np.array(
                [
                    self._grid_log_prob(grid, obs) + self._on_road_log_prob(next_state, obs)
                    for obs in candidates
                ],
                dtype=float,
            )
        clean = self._clean_kinematics(next_state)
        return np.array([self._kinematics_log_prob(clean, obs) for obs in candidates], dtype=float)

    def _draw_observation(self, state: Any) -> Dict[str, np.ndarray]:
        if self.observation_mode is ObservationMode.POMDP:
            return self._draw_occupancy(state)
        return self._draw_kinematics(state)

    def _draw_occupancy(self, state: Any) -> Dict[str, np.ndarray]:
        grid = self._render_presence_grid(state)
        flips = np.random.random(grid.shape) < self.cell_flip_prob
        occupancy = np.zeros((2, GRID_CELLS, GRID_CELLS), dtype=np.float32)
        occupancy[PRESENCE_LAYER] = np.logical_xor(grid, flips)
        occupancy[ON_ROAD_LAYER] = self._render_on_road_layer(state)
        return {OCCUPANCY_KEY: occupancy}

    def _render_on_road_layer(self, state: Any) -> np.ndarray:
        """The on-road layer this model predicts for ``state``; all-ones by default.

        All-ones is the honest answer for a model with no picture of the road: it says
        "drivable everywhere I can see", which is what the shipped racetrack shows on a
        straight anyway. A subclass that carries a road model should override this so that
        the observations it *samples* are the same shape as the ones it *reads* — otherwise
        it feeds itself an all-clear corridor on the approach to every corner.
        """
        del state
        return np.ones((GRID_CELLS, GRID_CELLS), dtype=np.float32)

    def _on_road_log_prob(self, state: Any, observation: Any) -> float:
        """Contribution of the on-road layer to the likelihood; zero by default.

        Zero, and not a Bernoulli over the layer, because the default
        :meth:`_render_on_road_layer` does not depend on ``state``: a term identical across
        every particle shifts all the log-weights alike and vanishes at normalisation, so it
        buys nothing but 144 cells of arithmetic per particle per step.
        """
        del state, observation
        return 0.0

    def _draw_kinematics(self, state: Any) -> Dict[str, np.ndarray]:
        clean = self._clean_kinematics(state)
        ego = clean[_EGO_KEY] + np.random.normal(scale=self.ego_pose_std, size=4)
        agents = clean[_AGENTS_KEY]
        present = agents[:, AGENT_PRESENT] > 0.5
        count = int(np.count_nonzero(present))
        agents[present, AGENT_REL_X : AGENT_REL_X + 2] += np.random.normal(
            scale=self.agent_pose_std, size=(count, 2)
        )
        agents[present, AGENT_REL_VX : AGENT_REL_VX + 2] += np.random.normal(
            scale=self.agent_velocity_std, size=(count, 2)
        )
        return {_EGO_KEY: ego, _AGENTS_KEY: agents}

    def _render_presence_grid(self, state: Any) -> np.ndarray:
        """Rasterise the presence layer the world's occupancy grid would show.

        One rasteriser serves both the sampler and the density, so the two agree by
        construction rather than by two edits staying in step. Axis 0 is along-track and
        axis 1 across-track, matching highway-env 1.12.1, and the ego is written into the
        centre cell exactly as the simulator writes the observer into its own grid. A
        vehicle marks one cell, not a footprint.
        """
        grid = np.zeros((GRID_CELLS, GRID_CELLS), dtype=bool)
        grid[_GRID_CENTRE, _GRID_CENTRE] = True

        rows = state_agent_rows(np.asarray(state, dtype=float), self.max_tracked_agents)
        along = np.floor((rows[:, AGENT_REL_X] + GRID_HALF_EXTENT_M) / GRID_STEP_M).astype(int)
        across = np.floor((rows[:, AGENT_REL_Y] + GRID_HALF_EXTENT_M) / GRID_STEP_M).astype(int)
        inside = (
            (rows[:, AGENT_PRESENT] > 0.5)
            & (along >= 0)
            & (along < GRID_CELLS)
            & (across >= 0)
            & (across < GRID_CELLS)
        )
        grid[along[inside], across[inside]] = True
        return grid

    def _grid_log_prob(self, grid: np.ndarray, observation: Any) -> float:
        observed = np.asarray(observation[OCCUPANCY_KEY], dtype=float)[PRESENCE_LAYER] > 0.5
        return self._bernoulli_cell_log_prob(grid, observed)

    def _bernoulli_cell_log_prob(self, predicted: np.ndarray, observed: np.ndarray) -> float:
        """Independent per-cell flip model over two boolean grids of the same shape."""
        flip_prob = float(np.clip(self.cell_flip_prob, _FLIP_PROB_EPS, 1.0 - _FLIP_PROB_EPS))
        agreements = int(np.count_nonzero(observed == predicted))
        disagreements = observed.size - agreements
        return agreements * float(np.log1p(-flip_prob)) + disagreements * float(np.log(flip_prob))

    def _clean_kinematics(self, state: Any) -> Dict[str, np.ndarray]:
        """The noise-free MDP reading of a state; the mean both the sampler and density use."""
        array = np.asarray(state, dtype=float)
        speed, heading = float(array[EGO_SPEED]), float(array[EGO_HEADING])
        ego = np.array(
            [array[EGO_X], array[EGO_Y], speed * np.cos(heading), speed * np.sin(heading)],
            dtype=float,
        )
        agents = np.array(state_agent_rows(array, self.max_tracked_agents), dtype=float)
        return {_EGO_KEY: ego, _AGENTS_KEY: agents}

    def _kinematics_log_prob(self, clean: Dict[str, np.ndarray], observation: Any) -> float:
        agents = clean[_AGENTS_KEY]
        present = agents[:, AGENT_PRESENT] > 0.5
        observed_agents = np.asarray(observation[_AGENTS_KEY], dtype=float)
        positions = slice(AGENT_REL_X, AGENT_REL_X + 2)
        velocities = slice(AGENT_REL_VX, AGENT_REL_VX + 2)
        return (
            _gaussian_log_prob(
                np.asarray(observation[_EGO_KEY], dtype=float) - clean[_EGO_KEY],
                self.ego_pose_std,
            )
            + _gaussian_log_prob(
                observed_agents[present, positions] - agents[present, positions],
                self.agent_pose_std,
            )
            + _gaussian_log_prob(
                observed_agents[present, velocities] - agents[present, velocities],
                self.agent_velocity_std,
            )
        )

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        if set(observation1) != set(observation2):
            return False
        return all(
            np.array_equal(np.asarray(observation1[key]), np.asarray(observation2[key]))
            for key in observation1
        )

    def hash_observation(self, observation: Any) -> Hashable:
        return tuple((key, np.asarray(observation[key]).tobytes()) for key in sorted(observation))

    # ── Initial distributions (particle seeds) ───────────────────────────
    def initial_state_dist(self) -> Distribution:
        """Seed particles on the lane centreline at the speed limit.

        In a two-environment episode the true start comes from the world, so this exists to
        give the belief a prior rather than to describe the track: a zero ego block at
        :data:`DEFAULT_SPEED_LIMIT` with no agents, spread by one step's worth of process
        noise. A zero-noise model therefore seeds a deterministic point mass.
        """
        parent = self

        class InitialState(Distribution):
            def sample(self, n_samples: int = 1) -> List[np.ndarray]:
                # pylint: disable=protected-access
                return [parent._seed_state() for _ in range(n_samples)]

        return InitialState()

    def initial_observation_dist(self) -> Distribution:
        """Observe a state drawn from :meth:`initial_state_dist`."""
        parent = self

        class InitialObservation(Distribution):
            def sample(self, n_samples: int = 1) -> List[Any]:
                # pylint: disable=protected-access
                return [
                    parent.sample_observation(parent._seed_state(), action=None)
                    for _ in range(n_samples)
                ]

        return InitialObservation()

    def _seed_state(self) -> np.ndarray:
        state = np.zeros(self.state_width, dtype=float)
        state[EGO_SPEED] = DEFAULT_SPEED_LIMIT
        return self._perturb(state.reshape(1, -1))[0]


__all__ = ["RacetrackModelPOMDP"]
