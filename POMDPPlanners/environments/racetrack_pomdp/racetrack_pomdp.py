# SPDX-License-Identifier: MIT

"""Racetrack POMDP world environment, with a matched fully-observed baseline.

This module adapts HighwayEnv's ``racetrack-v0`` to the POMDPPlanners
:class:`~POMDPPlanners.core.environment.Environment` interface so it can serve as the
**ground-truth world** in an
:class:`~POMDPPlanners.simulations.episodes.EpisodeRunner`.

Its reason to exist is the *matched pair*. To attribute a planner's performance drop to
partial observability, the two runs being compared must differ in what the agent sees and
in nothing else. Every other driving environment here is partially observed by
construction, so that comparison was impossible. Selecting
:class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_schema.ObservationMode`
changes only what the world measures; the track, the opponent, the dynamics, the step rates
and the reward come off one code path, and a test asserts the two arms' simulator
configurations are byte-identical once the observation block is removed. See
:mod:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_schema`.

Like the CARLA and nuPlan worlds, this one is **forward-only**: it advances a single true
state one tick per interaction and cannot be re-run from an injected state, so it cannot
serve as a planner's generative model. :meth:`RacetrackPOMDP.transition_log_probability`
and :meth:`RacetrackPOMDP.observation_log_probability` therefore raise
:class:`NotImplementedError` — in the intended world/model split they are never called.
The planner keeps its own model on ``policy.environment``; see
:class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp.RacetrackModelPOMDP`.

The **state** is the same in both modes and is documented in ``racetrack_schema``. The
**observation** is what changes:

- ``ObservationMode.MDP``: a ``(max_tracked_agents + 1, 5)`` float32 table of
  ``[presence, x, y, vx, vy]``, absolute and unnormalised, ego first. Only the other
  vehicles' driver policy stays hidden, so this is a *near*-MDP baseline, not a true MDP.
- ``ObservationMode.POMDP``: a :class:`~POMDPPlanners.environments.racetrack_pomdp.
  racetrack_schema.RacetrackObservation` — the ego's own pose and speed, a noisy lane-camera
  reading of lateral offset and lane-relative heading, the lane's curvature at fixed
  distances ahead, and a list of unlabeled detections.

**The rule the POMDP arm is built on: the reading is the whole state, minus the vehicles the
sensor cannot see.** That is what decides each channel:

* **Ego pose.** ``x``, ``y``, heading and arclength along the lap, at the near-exact widths
  GPS/IMU and a wheel odometer deliver. Withholding these made the arm a localisation
  problem as well as a tracking one, and only the second is what a range gate controls.
* **Speedometer.** Emitted exactly, which a real one is to well under a percent.
* **Lane camera.** Lateral offset and lane-relative heading, with noise, because
  highway-env's ``lane_offset`` is exact and no camera is.
* **Curvature ahead.** The same camera's other product: the curvature of the lane at each of
  ``curvature_lookahead_m`` metres along it. This world reads it off the true track geometry
  — the lane graph walked from the ego's own lane, which is also what its arclength slot is
  numbered against — and corrupts it at ``curvature_std_1pm``.
* **Detections.** For every vehicle within ``max_detection_range_m`` that is not behind a
  closer one, a noisy ``[rel_x, rel_y, rel_vx, rel_vy]`` in the ego body frame — the whole
  kinematic row, not a projection of it. A vehicle failing either test produces **no row at
  all**, and that absence is the arm's hidden state. Detections carry no identity and are
  ordered by measured range, so a planner cannot follow one across a step except through its
  filter.

``max_detection_range_m`` is therefore the dial, and the two arms are its two ends: at
``R -> inf`` this reading is the state to within the sensor widths, and as R shrinks the
traffic drops out of it first. What stays hidden at any R is the other drivers' intent,
whatever sits behind an occluder, and which return was which vehicle.

The noise is drawn from NumPy's global generator, so ``np.random.seed`` reproduces a run.
The constructor's ``seed=`` covers the simulator's own randomness and not this, which is a
known limit rather than a design choice.

Classes:
    RacetrackPOMDP: Forward-only adapter exposing a racetrack session as a world.
    RacetrackStepChannel: Per-step measurement channel names.
    RacetrackMetric: Episode-level metric names.

Example:
    >>> import numpy as np
    >>> np.random.seed(42)  # For reproducible results
    >>>
    >>> # Initialize environment
    >>> env = RacetrackPOMDP(discount_factor=0.95, seed=0)
    >>>
    >>> # Get initial state
    >>> initial_state = env.initial_state_dist().sample()[0]
    >>>
    >>> # Check terminal condition before stepping the live world forward
    >>> env.is_terminal(initial_state)
    False
    >>>
    >>> # Sample complete step (action is an index into the control presets)
    >>> action = 4  # coast, straight ahead
    >>> next_state, observation, reward = env.sample_next_step(initial_state, action)
    >>> observation.ego_pose.shape
    (4,)
    >>> observation.ego_speed.shape
    (1,)
    >>> observation.lane_pose.shape
    (2,)
    >>> observation.curvature_ahead.shape
    (3,)
    >>> observation.detections.shape
    (4, 5)
"""

from collections.abc import Hashable
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import Environment, SpaceInfo, SpaceType
from POMDPPlanners.core.simulation.step_info_metrics import EpisodeReduction, StepInfoMetric
from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import (
    TrackGeometry,
    build_track_geometry,
    lane_curvature,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_world_sensors import (
    SensorConfig,
    WorldSensors,
    relative_vehicles,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_PRESENT,
    AGENT_REL_X,
    AGENT_SLOT_WIDTH,
    DEFAULT_ACTION_PRESETS,
    DEFAULT_ACTION_REWARD,
    DEFAULT_BLOCKER_HALF_WIDTH_M,
    DEFAULT_COLLISION_REWARD,
    DEFAULT_CURVATURE_LOOKAHEAD_M,
    DEFAULT_CURVATURE_STD_1PM,
    DEFAULT_DETECTION_POSITION_STD_M,
    DEFAULT_DETECTION_VELOCITY_STD,
    DEFAULT_DURATION,
    DEFAULT_EGO_ARCLENGTH_STD_M,
    DEFAULT_EGO_HEADING_STD_RAD,
    DEFAULT_EGO_POSITION_STD_M,
    DEFAULT_ENV_ID,
    DEFAULT_LANE_CENTERING_COST,
    DEFAULT_LANE_CENTERING_REWARD,
    DEFAULT_LANE_HEADING_STD_RAD,
    DEFAULT_LANE_LATERAL_STD_M,
    DEFAULT_MAX_DETECTION_RANGE_M,
    DEFAULT_MAX_TRACKED_AGENTS,
    DEFAULT_NEAR_MISS_DISTANCE_M,
    DEFAULT_OTHER_VEHICLES,
    DEFAULT_POLICY_FREQUENCY,
    DEFAULT_SIMULATION_FREQUENCY,
    DEFAULT_SPEED_LIMIT,
    EGO_LAT,
    EGO_SPEED,
    EGO_STATE_WIDTH,
    MAX_ACCELERATION_MPS2,
    ObservationMode,
    RacetrackObservation,
    build_racetrack_config,
    ego_speed_from_kinematics_row,
    racetrack_reward,
    state_agent_rows,
    wrap_to_pi,
)

_ROLE_NEXT_STATE = "next_state"
_ROLE_REWARD = "reward"


def _observation_arrays(observation: Any) -> Tuple[np.ndarray, ...]:
    """The arrays making up one reading: four in POMDP mode, one in MDP mode."""
    if isinstance(observation, tuple):
        return tuple(np.asarray(part) for part in observation)
    return (np.asarray(observation),)


def _copy_observation(observation: Any) -> Any:
    if isinstance(observation, RacetrackObservation):
        return RacetrackObservation(*(np.array(part, copy=True) for part in observation))
    return np.array(observation, copy=True)


class RacetrackStepChannel(Enum):
    """Per-step measurement channels written to ``StepData.info``."""

    CRASHED = "crashed"
    OFF_ROAD = "off_road"
    TIME_LIMIT = "time_limit"
    ABS_LANE_OFFSET_M = "abs_lane_offset_m"
    SPEED_MPS = "speed_mps"
    COLLISION_SPEED_MPS = "collision_speed_mps"
    NEAR_MISS = "near_miss"


class RacetrackMetric(Enum):
    """Episode-level metric names reported by ``compute_metrics``."""

    COLLISION_RATE = "collision_rate"
    OFF_ROAD_RATE = "off_road_rate"
    TIME_LIMIT_RATE = "time_limit_rate"
    MEAN_ABS_LANE_OFFSET_M = "mean_abs_lane_offset_m"
    MEAN_SPEED_MPS = "mean_speed_mps"
    COLLISION_SPEED_MPS = "collision_speed_mps"
    NEAR_MISS_RATE = "near_miss_rate"


class _RacetrackSession:
    """Live highway-env session: the only object in this package touching the backend."""

    def __init__(
        self,
        env_id: str,
        config: Dict[str, Any],
        max_tracked_agents: int,
        observation_mode: ObservationMode,
        sensor: SensorConfig,
        render_mode: Optional[str] = None,
    ) -> None:
        # Imported here, not at module scope, so the package can be imported, and the
        # planner model and belief used, without highway-env installed. Matches how the
        # CARLA world defers `import carla`.
        import gymnasium  # pylint: disable=import-outside-toplevel
        import highway_env  # pylint: disable=import-outside-toplevel,import-error

        # highway_env is imported for its env-registration side effect; keeping the
        # reference makes that explicit rather than looking like an unused import.
        self._highway_env = highway_env
        # Typed as Any deliberately: highway-env ships no stubs, and gymnasium's Env does
        # not declare the `.vehicle` / `.road` attributes this adapter reads off the
        # unwrapped racetrack env.
        self._env: Any = gymnasium.make(env_id, config=config, render_mode=render_mode)
        self._max_tracked_agents = max_tracked_agents
        self._observation_mode = observation_mode
        self._sensors = WorldSensors(sensor, max_tracked_agents)
        self._lane_offsets: Dict[Any, float] = {}
        self._geometry: Optional[TrackGeometry] = None

    def render_frame(self) -> Optional[np.ndarray]:
        """Return the current bird's-eye frame, or None if not rendering."""
        frame = self._env.render()
        return None if frame is None else np.asarray(frame)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Any]:
        observation = self._env.reset(seed=seed)[0]
        return self._read_state(), self._encode(observation)

    def step(self, acceleration: float, steering: float) -> Dict[str, Any]:
        command = np.array([acceleration, steering], dtype=np.float32)
        step_result = self._env.step(command)
        observation, truncated = step_result[0], step_result[3]
        vehicle = self._ego()
        return {
            "state": self._read_state(),
            "observation": self._encode(observation),
            "crashed": bool(vehicle.crashed),
            "off_road": not bool(vehicle.on_road),
            "truncated": bool(truncated),
        }

    def _encode(self, observation: Any) -> Any:
        """Turn the simulator's raw reading into what this world promises to emit.

        In MDP mode that is the kinematics table unchanged. In POMDP mode only the ego's own
        speed comes from the simulator's reading, and everything else in that block is
        dropped here before the observation leaves the world; see the note beside
        ``EGO_KINEMATICS_VEHICLES_COUNT`` for why the block arrives wider than the ego row.

        Every other channel is measured off the ego vehicle, the road network and the vehicle
        list by
        :class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_world_sensors.WorldSensors`,
        because highway-env has no observation type reporting arclength around a lap,
        lane-relative pose, curvature ahead, a range gate or occlusion.
        """
        if self._observation_mode is not ObservationMode.POMDP:
            return np.asarray(observation)
        vehicle = self._ego()
        return self._sensors.read(
            ego_speed=ego_speed_from_kinematics_row(np.asarray(observation, dtype=float)[0]),
            ego=vehicle,
            arclength=self._arclength_of(vehicle),
            geometry=self._track_geometry(),
            others=self._other_vehicles(vehicle),
        )

    def _other_vehicles(self, ego: Any) -> List[Any]:
        return [vehicle for vehicle in self._env.unwrapped.road.vehicles if vehicle is not ego]

    def _track_geometry(self) -> TrackGeometry:
        if self._geometry is None:
            self._arclength_of(self._ego())
        assert self._geometry is not None
        return self._geometry

    def _ego(self) -> Any:
        return self._env.unwrapped.vehicle

    def _read_state(self) -> np.ndarray:
        vehicle = self._ego()
        offset = np.asarray(vehicle.lane_offset, dtype=float)
        ego = np.array(
            [
                float(vehicle.position[0]),
                float(vehicle.position[1]),
                float(vehicle.heading),
                float(vehicle.speed),
                float(offset[1]),
                float(offset[2]),
                self._arclength_of(vehicle),
            ],
            dtype=float,
        )
        return np.concatenate([ego, self._read_agent_slots(vehicle)])

    def _arclength_of(self, vehicle: Any) -> float:
        """Distance travelled along the track centreline, in metres.

        ``lane_offset`` is measured within the current lane, so it restarts at every
        segment boundary. Adding the lap offset of that lane makes it monotonic around the
        lap, which is what the planner's model indexes its curvature profile with.
        """
        lane_index = vehicle.lane_index
        if lane_index not in self._lane_offsets:
            # The curvature profile is rebuilt from the same lane in the same breath. Both
            # number distance from that lane's start, so a re-base that moved one and not the
            # other would have the camera reporting the corner at the wrong arclength.
            self._lane_offsets = self._build_lane_offsets(lane_index)
            self._geometry = build_track_geometry(self._env.unwrapped.road.network, lane_index)
        local = float(np.asarray(vehicle.lane_offset, dtype=float)[0])
        return self._lane_offsets.get(lane_index, 0.0) + local

    def _build_lane_offsets(self, lane_index: Any) -> Dict[Any, float]:
        """Cumulative lap offset of every lane reachable by following ``next_lane``."""
        network = self._env.unwrapped.road.network
        offsets: Dict[Any, float] = {}
        total = 0.0
        current = lane_index
        for _ in range(64):
            if current in offsets:
                break
            lane = network.get_lane(current)
            offsets[current] = total
            total += float(lane.length)
            current = network.next_lane(current, position=lane.position(lane.length, 0))
        return offsets

    def _read_agent_slots(self, ego: Any) -> np.ndarray:
        # The true state, so no range gate, no occlusion and no noise: what the sensor can
        # and cannot see is a property of the observation, not of the world's state.
        rows = np.zeros((self._max_tracked_agents, AGENT_SLOT_WIDTH), dtype=float)
        positions, velocities = relative_vehicles(ego, self._other_vehicles(ego))
        if len(positions) == 0:
            return rows.reshape(-1)
        ranked = np.argsort(np.linalg.norm(positions, axis=1))[: self._max_tracked_agents]
        for slot, index in enumerate(ranked):
            rows[slot] = [
                1.0,
                positions[index, 0],
                positions[index, 1],
                velocities[index, 0],
                velocities[index, 1],
            ]
        return rows.reshape(-1)


class RacetrackPOMDP(Environment):
    """Forward-only racetrack world with selectable full or partial observation.

    Advances one true state per interaction against a live HighwayEnv session. It is a
    world, not a model: it has no transition or observation density and cannot be
    resampled from an arbitrary state. Pair it with
    :class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp.RacetrackModelPOMDP`
    on ``policy.environment``.

    Attributes:
        observation_mode: Which arm of the matched pair this instance runs.
        action_presets: Normalised ``(acceleration, steering)`` commands; an action is an
            index into this sequence.
        max_tracked_agents: Number of fixed agent slots in the state.
        near_miss_distance_m: Range at or below which a step counts as a near miss.
        ego_position_std_m: Localisation noise on the reported ``x`` and ``y``, in metres.
        ego_heading_std_rad: Localisation noise on the reported heading, in radians.
        ego_arclength_std_m: Odometry noise on the reported arclength, in metres.
        lane_lateral_std_m: Lane camera's lateral-offset noise, in metres.
        lane_heading_std_rad: Lane camera's heading noise, in radians.
        curvature_lookahead_m: Distances along the lane the camera reports curvature at.
        curvature_std_1pm: Lane camera's curvature noise, in 1/m.
        max_detection_range_m: Range beyond which no vehicle is reported. The dial.
        detection_position_std_m: Per-axis position noise on a detection, in metres.
        detection_velocity_std: Per-axis relative-velocity noise on a detection, in m/s.
        blocker_half_width_m: Half-width of an occluding vehicle, in metres.

    Example:
        Running one interaction against the live simulator::

            world = RacetrackPOMDP(discount_factor=0.95, seed=0)
            state = world.initial_state_dist().sample()[0]
            next_state, observation, reward = world.sample_next_step(state, action=4)
    """

    def __init__(
        self,
        discount_factor: float,
        observation_mode: ObservationMode = ObservationMode.POMDP,
        env_id: str = DEFAULT_ENV_ID,
        action_presets: Optional[Sequence[Tuple[float, float]]] = None,
        max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
        other_vehicles: int = DEFAULT_OTHER_VEHICLES,
        duration: int = DEFAULT_DURATION,
        policy_frequency: int = DEFAULT_POLICY_FREQUENCY,
        simulation_frequency: int = DEFAULT_SIMULATION_FREQUENCY,
        collision_reward: float = DEFAULT_COLLISION_REWARD,
        lane_centering_cost: float = DEFAULT_LANE_CENTERING_COST,
        lane_centering_reward: float = DEFAULT_LANE_CENTERING_REWARD,
        action_reward: float = DEFAULT_ACTION_REWARD,
        speed_limit: float = DEFAULT_SPEED_LIMIT,
        near_miss_distance_m: float = DEFAULT_NEAR_MISS_DISTANCE_M,
        ego_position_std_m: float = DEFAULT_EGO_POSITION_STD_M,
        ego_heading_std_rad: float = DEFAULT_EGO_HEADING_STD_RAD,
        ego_arclength_std_m: float = DEFAULT_EGO_ARCLENGTH_STD_M,
        lane_lateral_std_m: float = DEFAULT_LANE_LATERAL_STD_M,
        lane_heading_std_rad: float = DEFAULT_LANE_HEADING_STD_RAD,
        curvature_lookahead_m: Sequence[float] = DEFAULT_CURVATURE_LOOKAHEAD_M,
        curvature_std_1pm: float = DEFAULT_CURVATURE_STD_1PM,
        max_detection_range_m: float = DEFAULT_MAX_DETECTION_RANGE_M,
        detection_position_std_m: float = DEFAULT_DETECTION_POSITION_STD_M,
        detection_velocity_std: float = DEFAULT_DETECTION_VELOCITY_STD,
        blocker_half_width_m: float = DEFAULT_BLOCKER_HALF_WIDTH_M,
        terminate_off_road: bool = True,
        config_overrides: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
        reward_range: Optional[Tuple[float, float]] = None,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ) -> None:
        """Initialize the racetrack world.

        Args:
            discount_factor: Discount factor for future rewards.
            observation_mode: Which arm of the matched pair to run. Defaults to POMDP.
            env_id: HighwayEnv scenario id. Defaults to ``"racetrack-v0"``.
            action_presets: Normalised ``(acceleration, steering)`` commands. Defaults to
                a 3x3 throttle-by-steer grid.
            max_tracked_agents: Fixed agent slots in the state. Defaults to 4.
            other_vehicles: Extra opponents beyond the one always spawned. Defaults to 1.
            duration: Episode length limit in simulator time. Defaults to 300.
            policy_frequency: Decisions per second. Defaults to 5.
            simulation_frequency: Physics steps per second. Defaults to 15.
            collision_reward: Weight applied to a collision. Defaults to -1.0.
            lane_centering_cost: Sharpness of the lane-centering falloff. Defaults to 4.0.
            lane_centering_reward: Weight on the lane-centering term. Defaults to 1.0.
            action_reward: Weight on the control-effort penalty. Defaults to -0.3.
            speed_limit: Track speed limit in m/s. Defaults to 10.0.
            near_miss_distance_m: Near-miss range in metres. Defaults to 5.0.
            ego_position_std_m: Localisation noise on the reported ``x`` and ``y``, in
                metres. Defaults to 0.1, the decimetre band a production GPS/IMU stack
                delivers. Near-exact on purpose: this arm withholds *vehicles*, not the
                ego's own pose, and a wide width here would add a localisation problem on
                top of the tracking one the range gate poses.
            ego_heading_std_rad: Localisation noise on the reported heading, in radians.
                Defaults to 0.01, which is 0.57 degrees.
            ego_arclength_std_m: Odometry noise on the reported distance round the lap, in
                metres. Defaults to 0.1. It reads off the same lane walk the arclength state
                slot is numbered against, so the two cannot disagree about where a corner is.
            lane_lateral_std_m: Lane camera's lateral-offset noise, in metres. Defaults to
                0.05, the conservative end of the centimetre-scale accuracy a production
                mono-camera lane detector is specified at over the few metres this window
                covers. Unlike the speedometer this is real sensor noise: highway-env's
                ``lane_offset`` is exact, and emitting it unchanged would hand the planner
                a lane-relative pose no camera delivers. Set to 0.0 only in tests.
            lane_heading_std_rad: Lane camera's heading noise, in radians. Defaults to
                0.01, which is 0.57 degrees — the sub-degree band the same detectors quote.
            curvature_lookahead_m: Distances along the lane, in metres, that the camera
                reports curvature at. Defaults to ``(10.0, 20.0, 30.0)``. The planner's
                model must be built with the same distances; it scores one Gaussian per
                entry, so a mismatch compares curvature at one distance against another.
            curvature_std_1pm: Lane camera's curvature noise, in 1/m. Defaults to 0.002,
                derived from the same detector's decimetre lateral accuracy carried out to
                the nearest lookahead; see the note beside its constant.
            max_detection_range_m: Range in metres beyond which no vehicle is reported.
                Defaults to 40.0. **This is the arm's dial**: everything else in the state
                is observed, so raising it towards infinity walks the POMDP arm continuously
                back to the MDP baseline and lowering it hides more of the traffic.
            detection_position_std_m: Per-axis position noise on a detection, in metres.
                Defaults to 0.5.
            detection_velocity_std: Per-axis relative-velocity noise on a detection, in m/s.
                Defaults to 0.3, tighter than the position width because a velocity comes
                off a frequency shift rather than a time of flight. Applied to both
                components alike: a visible vehicle's whole velocity is reported.
            blocker_half_width_m: Half-width in metres of a vehicle treated as an occluder.
                Defaults to 1.0, a 2 m-wide car. Occlusion is deterministic: a vehicle is
                masked when a closer one lies within the half-angle its body subtends.
            terminate_off_road: Whether leaving the road ends the episode. Defaults to True.
            config_overrides: Extra HighwayEnv config keys, applied to both arms alike.
            seed: Seed for the first reset, for reproducibility. Defaults to None.
            name: Environment name. Defaults to ``"RacetrackPOMDP-<mode>"``.
            reward_range: Optional ``(min, max)`` reward bounds.
            output_dir: Optional directory for logging output.
            debug: Enable debug logging.
            use_queue_logger: Whether to use queue-based logging.

        Raises:
            ValueError: If ``max_tracked_agents`` is below 1, or the step rates do not
                divide evenly.
        """
        if max_tracked_agents < 1:
            raise ValueError(f"max_tracked_agents must be at least 1, got {max_tracked_agents}.")

        self.observation_mode = observation_mode
        self.env_id = env_id
        self.action_presets: Tuple[Tuple[float, float], ...] = tuple(
            action_presets if action_presets is not None else DEFAULT_ACTION_PRESETS
        )
        self.max_tracked_agents = max_tracked_agents
        self.near_miss_distance_m = near_miss_distance_m
        self.ego_position_std_m = float(ego_position_std_m)
        self.ego_heading_std_rad = float(ego_heading_std_rad)
        self.ego_arclength_std_m = float(ego_arclength_std_m)
        self.lane_lateral_std_m = float(lane_lateral_std_m)
        self.lane_heading_std_rad = float(lane_heading_std_rad)
        self.curvature_lookahead_m: Tuple[float, ...] = tuple(
            float(distance) for distance in curvature_lookahead_m
        )
        self.curvature_std_1pm = float(curvature_std_1pm)
        self.max_detection_range_m = float(max_detection_range_m)
        self.detection_position_std_m = float(detection_position_std_m)
        self.detection_velocity_std = float(detection_velocity_std)
        self.blocker_half_width_m = float(blocker_half_width_m)
        # Kept as an attribute as well as a config key: highway-env consults it in
        # `_is_terminated`, and this adapter decides termination itself rather than
        # reading the simulator's flag, so the two must be driven by the same value.
        self.terminate_off_road = terminate_off_road
        self.render_mode = render_mode
        self.seed = seed
        self.collision_reward = collision_reward
        self.lane_centering_cost = lane_centering_cost
        self.lane_centering_reward = lane_centering_reward
        self.action_reward = action_reward
        # Raises on a non-integral substep ratio, which would silently desynchronise the
        # planner's model from the world.
        self.simulator_config = build_racetrack_config(
            observation_mode,
            max_tracked_agents=max_tracked_agents,
            other_vehicles=other_vehicles,
            duration=duration,
            policy_frequency=policy_frequency,
            simulation_frequency=simulation_frequency,
            collision_reward=collision_reward,
            lane_centering_cost=lane_centering_cost,
            lane_centering_reward=lane_centering_reward,
            action_reward=action_reward,
            speed_limit=speed_limit,
            terminate_off_road=terminate_off_road,
            overrides=config_overrides,
        )

        self._session: Optional[_RacetrackSession] = None
        self._live_state: Optional[np.ndarray] = None
        self._latest_obs: Optional[Any] = None
        self._terminated: bool = False
        self._seeded: bool = False
        self._pending: Optional[Dict[str, Any]] = None
        self._served_roles: Set[str] = set()

        super().__init__(
            discount_factor=discount_factor,
            name=name if name is not None else f"RacetrackPOMDP-{observation_mode.value}",
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

    # ── Serialization: drop the non-picklable live handle ───────────────
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_session"] = None
        state["_live_state"] = None
        state["_latest_obs"] = None
        state["_terminated"] = False
        state["_seeded"] = False
        state["_pending"] = None
        state["_served_roles"] = set()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        vars(self).update(state)
        self._session = None
        self._live_state = None
        self._latest_obs = None
        self._terminated = False
        self._seeded = False
        self._pending = None
        self._served_roles = set()

    # ── Live simulator management ───────────────────────────────────────
    def _get_session(self) -> _RacetrackSession:
        if self._session is None:
            self._session = _RacetrackSession(
                env_id=self.env_id,
                config=self.simulator_config,
                max_tracked_agents=self.max_tracked_agents,
                observation_mode=self.observation_mode,
                sensor=SensorConfig(
                    ego_position_std_m=self.ego_position_std_m,
                    ego_heading_std_rad=self.ego_heading_std_rad,
                    ego_arclength_std_m=self.ego_arclength_std_m,
                    lane_lateral_std_m=self.lane_lateral_std_m,
                    lane_heading_std_rad=self.lane_heading_std_rad,
                    curvature_lookahead_m=self.curvature_lookahead_m,
                    curvature_std_1pm=self.curvature_std_1pm,
                    max_detection_range_m=self.max_detection_range_m,
                    detection_position_std_m=self.detection_position_std_m,
                    detection_velocity_std=self.detection_velocity_std,
                    blocker_half_width_m=self.blocker_half_width_m,
                ),
                render_mode=self.render_mode,
            )
        return self._session

    def render_frame(self) -> Optional[np.ndarray]:
        """Return the current bird's-eye RGB frame for visualisation.

        Returns:
            An ``(H, W, 3)`` uint8 array, or ``None`` when the environment was built
            without a ``render_mode`` or before the first reset.
        """
        if self.render_mode is None or self._session is None:
            return None
        return self._session.render_frame()

    def _reset(self) -> np.ndarray:
        session = self._get_session()
        if self.seed is not None and not self._seeded:
            state, observation = session.reset(seed=self.seed)
            self._seeded = True
        else:
            state, observation = session.reset()
        state = np.asarray(state)
        self._live_state = state
        self._latest_obs = observation
        self._terminated = False
        self._pending = None
        self._served_roles = set()
        return state

    def _to_control(self, action: Any) -> Tuple[float, float]:
        return self.action_presets[int(action)]

    def _states_equal(self, state_a: Any, state_b: Any) -> bool:
        if state_a is None or state_b is None:
            return False
        return np.array_equal(np.asarray(state_a), np.asarray(state_b))

    def _min_agent_range(self, state: np.ndarray) -> float:
        rows = state_agent_rows(state, self.max_tracked_agents)
        present = rows[rows[:, AGENT_PRESENT] > 0.5]
        if present.size == 0:
            return float("inf")
        offsets = present[:, AGENT_REL_X : AGENT_REL_X + 2]
        return float(np.min(np.linalg.norm(offsets, axis=1)))

    def _measure(self, outcome: Dict[str, Any], next_state: np.ndarray) -> Dict[str, float]:
        """Derive the per-step measurement channels from one completed tick."""
        crashed = bool(outcome["crashed"])
        speed = float(next_state[EGO_SPEED])
        return {
            RacetrackStepChannel.CRASHED.value: float(crashed),
            RacetrackStepChannel.OFF_ROAD.value: float(outcome["off_road"]),
            RacetrackStepChannel.TIME_LIMIT.value: float(outcome["truncated"]),
            RacetrackStepChannel.ABS_LANE_OFFSET_M.value: abs(float(next_state[EGO_LAT])),
            RacetrackStepChannel.SPEED_MPS.value: speed,
            # Impact severity, not just its occurrence. This is the speed the ego was
            # doing when it hit something: highway-env only applies its crash braking
            # (acceleration = -speed) on the *following* action, so the state recorded
            # on the crashing step still carries the pre-impact speed. Non-crash steps
            # report 0.0 so a MAX reduction picks the impact out of the episode.
            RacetrackStepChannel.COLLISION_SPEED_MPS.value: speed if crashed else 0.0,
            RacetrackStepChannel.NEAR_MISS.value: float(
                self._min_agent_range(next_state) <= self.near_miss_distance_m
            ),
        }

    def _ensure_stepped(self, state: Any, action: Any, role: str) -> Dict[str, Any]:
        """Advance the world one tick for ``(state, action)`` exactly once, and cache it.

        The episode loop asks for the reward and the next state through two separate
        calls; the first advances the world and caches the outcome, the second is served
        from that cache. The cache is keyed on ``(state, action)`` *and* on the requesting
        role, because a repeated role means a genuinely new step even when the ego did not
        move — which happens here, since a crashed vehicle is braked to rest.
        """
        pending = self._pending
        if (
            pending is not None
            and role not in self._served_roles
            and self._states_equal(pending["state"], state)
            and self.hash_action(pending["action"]) == self.hash_action(action)
        ):
            self._served_roles.add(role)
            return pending

        if self._live_state is None or not self._states_equal(state, self._live_state):
            raise RuntimeError(
                "RacetrackPOMDP is a forward-only world environment; it cannot resample "
                "from an arbitrary state. Give the planner a separate model environment "
                "(policy.environment) and only step the world forward from its live state."
            )

        acceleration, steering = self._to_control(action)
        outcome = self._get_session().step(acceleration, steering)
        next_state = np.asarray(outcome["state"])
        # Mirrors highway-env's own `_is_terminated`, which gates the off-road ending on
        # `terminate_off_road`, plus its separate truncation. Hard-coding the off-road
        # ending here would make the constructor flag a lie: the simulator would keep the
        # episode alive while this adapter reported it over.
        ends_episode = outcome["crashed"] or (self.terminate_off_road and outcome["off_road"])
        done = bool(ends_episode or outcome["truncated"])
        pending = {
            "state": np.asarray(state).copy(),
            "action": action,
            "next_state": next_state,
            "observation": outcome["observation"],
            "reward": racetrack_reward(
                float(next_state[EGO_LAT]),
                (acceleration, steering),
                outcome["crashed"],
                not outcome["off_road"],
                collision_reward=self.collision_reward,
                lane_centering_cost=self.lane_centering_cost,
                lane_centering_reward=self.lane_centering_reward,
                action_reward=self.action_reward,
            ),
            "terminated": done,
            "info": self._measure(outcome, next_state),
        }
        self._pending = pending
        self._served_roles = {role}
        self._live_state = next_state
        self._latest_obs = outcome["observation"]
        self._terminated = done
        return pending

    # ── Environment interface ───────────────────────────────────────────
    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        if n_samples != 1:
            raise ValueError("RacetrackPOMDP is forward-only and only supports n_samples=1")
        return self._ensure_stepped(state, action, _ROLE_NEXT_STATE)["next_state"]

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        del action
        if n_samples != 1:
            raise ValueError("RacetrackPOMDP is forward-only and only supports n_samples=1")
        pending = self._pending
        if pending is not None and self._states_equal(pending["next_state"], next_state):
            return pending["observation"]
        raise RuntimeError(
            "RacetrackPOMDP.sample_observation was queried for a next state other than "
            "the live one; a forward-only world only knows the reading it just produced."
        )

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        del next_state
        return float(self._ensure_stepped(state, action, _ROLE_REWARD)["reward"])

    def is_terminal(self, state: Any) -> bool:
        if self._live_state is not None and not self._states_equal(state, self._live_state):
            raise RuntimeError(
                "RacetrackPOMDP.is_terminal was queried for a state other than the live "
                "world state; this forward-only world only knows whether its current "
                "state is terminal."
            )
        return self._terminated

    def step_info(self, state: Any, action: Any, next_state: Any) -> Dict[str, float]:
        """Report the measurements of the step just taken.

        Values are served from the cache the single simulator tick already filled — a
        forward-only world cannot re-measure a transition — so this consumes no randomness
        and has no side effects, which matters because it runs inside the episode loop.
        All three arguments are used, and only to confirm the request refers to that tick.
        Matching on the successor alone would be too weak: a crashed vehicle is braked to
        rest, so ``next_state == state`` is an ordinary reading here, and a later request
        carrying that same array as its *predecessor* would then be served the earlier
        step's measurements.

        Args:
            state: The state the step was taken from.
            action: The action taken.
            next_state: The realised successor state, or ``None`` on the terminal
                bookkeeping call.

        Returns:
            The measurement channels for that transition, or an empty mapping when the
            request does not refer to the cached tick.
        """
        pending = self._pending
        if pending is None or next_state is None:
            return {}
        if not self._states_equal(pending["next_state"], next_state):
            return {}
        if not self._states_equal(pending["state"], state):
            return {}
        if self.hash_action(pending["action"]) != self.hash_action(action):
            return {}
        return dict(pending["info"])

    def get_metric_specs(self) -> List[StepInfoMetric]:
        """Declare the episode metrics derived from the per-step channels.

        Failure modes are kept apart rather than collapsed into one terminated flag: the
        point of the MDP-versus-POMDP comparison is *why* a planner failed, and merging
        "hit a car" with "left the road" destroys exactly that.

        Returns:
            One spec per metric, in the order ``compute_metrics`` reports them.
        """
        return [
            StepInfoMetric(
                name=RacetrackMetric.COLLISION_RATE.value,
                channel=RacetrackStepChannel.CRASHED.value,
                per_episode=EpisodeReduction.ANY,
            ),
            StepInfoMetric(
                name=RacetrackMetric.OFF_ROAD_RATE.value,
                channel=RacetrackStepChannel.OFF_ROAD.value,
                per_episode=EpisodeReduction.ANY,
            ),
            StepInfoMetric(
                name=RacetrackMetric.TIME_LIMIT_RATE.value,
                channel=RacetrackStepChannel.TIME_LIMIT.value,
                per_episode=EpisodeReduction.ANY,
            ),
            StepInfoMetric(
                name=RacetrackMetric.MEAN_ABS_LANE_OFFSET_M.value,
                channel=RacetrackStepChannel.ABS_LANE_OFFSET_M.value,
                per_episode=EpisodeReduction.MEAN,
            ),
            StepInfoMetric(
                name=RacetrackMetric.MEAN_SPEED_MPS.value,
                channel=RacetrackStepChannel.SPEED_MPS.value,
                per_episode=EpisodeReduction.MEAN,
            ),
            # MAX, not MEAN: an episode has at most one crash, so the maximum is the
            # impact speed itself, while a mean over mostly-zero steps would report a
            # number no collision ever happened at. Episodes that never crash
            # contribute 0.0, so read this alongside collision_rate.
            StepInfoMetric(
                name=RacetrackMetric.COLLISION_SPEED_MPS.value,
                channel=RacetrackStepChannel.COLLISION_SPEED_MPS.value,
                per_episode=EpisodeReduction.MAX,
            ),
            StepInfoMetric(
                name=RacetrackMetric.NEAR_MISS_RATE.value,
                channel=RacetrackStepChannel.NEAR_MISS.value,
                per_episode=EpisodeReduction.ANY,
            ),
        ]

    def initial_state_dist(self) -> Distribution:
        parent = self

        class InitialState(Distribution):
            def sample(self, n_samples: int = 1) -> List[np.ndarray]:
                # pylint: disable=protected-access
                return [parent._reset() for _ in range(n_samples)]

        return InitialState()

    def initial_observation_dist(self) -> Distribution:
        parent = self

        class InitialObservation(Distribution):
            def sample(self, n_samples: int = 1) -> List[Any]:
                # pylint: disable=protected-access
                observation = parent._latest_obs
                if observation is None:
                    parent._reset()
                    observation = parent._latest_obs
                assert observation is not None
                return [_copy_observation(observation) for _ in range(n_samples)]

        return InitialObservation()

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        left = _observation_arrays(observation1)
        right = _observation_arrays(observation2)
        return len(left) == len(right) and all(
            np.array_equal(one, other) for one, other in zip(left, right)
        )

    def hash_observation(self, observation: Any) -> Hashable:
        return tuple(array.tobytes() for array in _observation_arrays(observation))

    def hash_action(self, action: Any) -> Hashable:
        if isinstance(action, np.ndarray):
            return action.tobytes()
        return action

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        del state, action, next_states
        raise NotImplementedError(
            "RacetrackPOMDP is a forward-only world environment with no transition "
            "density. Belief updates must run on the planner's model environment."
        )

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        del next_state, action, observations
        raise NotImplementedError(
            "RacetrackPOMDP is a forward-only world environment with no observation "
            "density. Belief updates must run on the planner's model environment."
        )


__all__ = [
    "MAX_ACCELERATION_MPS2",
    "RacetrackMetric",
    "RacetrackPOMDP",
    "RacetrackStepChannel",
    "lane_curvature",
    "wrap_to_pi",
]
