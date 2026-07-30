# SPDX-License-Identifier: MIT

# pylint: disable=too-many-lines  # Multi-agent traffic + perception grew the module.

"""CARLA POMDP world environment.

This module adapts the `CARLA <https://carla.org/>`_ autonomous-driving
simulator to the POMDPPlanners :class:`~POMDPPlanners.core.environment.Environment`
interface so it can serve as the **ground-truth world** in an
:class:`~POMDPPlanners.simulations.episodes.EpisodeRunner`.

CARLA is *forward-only*: it is a live Unreal server driven over a Python client
that exposes ``reset``/``tick`` on a single true state and cannot be queried for
a transition/observation density nor re-run from an arbitrary injected state. It
therefore cannot act as a planner's generative model. In the two-environment
episode design the planner keeps its own generative model (``policy.environment``)
and this wrapper only advances the single true state forward, one step per real
interaction. Consequently :meth:`CarlaPOMDP.transition_log_probability` and
:meth:`CarlaPOMDP.observation_log_probability` intentionally raise
:class:`NotImplementedError` — in the intended world/model split they are never
called.

Unlike :class:`~POMDPPlanners.environments.gym_pomdp.gym_pomdp.GymPOMDP` (which is
fully observed: observation equals state), CARLA is genuinely partially observed.

The world is populated each reset with surrounding autopilot traffic and walking
pedestrians (via CARLA's Traffic Manager), so it poses a genuine multi-agent
perception problem rather than an empty course.

The **state** is the ego vehicle's ground-truth kinematics and lane pose,
``[x, y, yaw, vx, vy, lat, heading_err]``, **followed by fixed slots for the
``max_tracked_agents`` nearest other vehicles** (ground truth). Each agent slot is
``[present, rel_x, rel_y, rel_yaw, rel_speed]`` expressed in the ego frame
(``rel_x`` forward, ``rel_y`` left, ``rel_yaw`` in **radians**, ``rel_speed`` in
m/s); ``present`` is ``1`` for a filled slot and ``0`` for padding. The ego part
is read straight from the simulator, where:

- ``x``, ``y``: ego position in the CARLA map (world) frame, in metres, read from
  the actor transform's location.
- ``yaw``: ego heading about the world Z axis, in **degrees** (CARLA convention),
  read from the actor transform's rotation.
- ``vx``, ``vy``: ego linear-velocity components in the world frame, in metres per
  second, read from the actor's velocity vector.
- ``lat``: signed lateral offset from the centre of the nearest driving lane, in
  metres (positive to the lane's left), from the CARLA map's lane geometry.
- ``heading_err``: ego heading minus the lane direction, wrapped to
  ``[-pi, pi]``, in **radians**.

The state ends with **one traffic-light slot**,
``[present, rel_x, rel_y, state_code, time_to_change]`` (ego frame; ``state_code`` is a
``TRAFFIC_LIGHT_*`` code, ``time_to_change`` in seconds), carrying the light governing the
ego as ground truth (``present == 0`` when none affects it). It is always in the state — used
by the red-light-violation metrics — and is independent of whether the *observation* exposes
the light (``include_traffic_light``); a planner can thus be scored for running reds even when
it is given no light information.

(The vertical axis ``z`` and roll/pitch are intentionally omitted; the ego is
modelled on the ground plane.)

The lane-relative terms (``lat``, ``heading_err``) drive a gym-carla-style
driving-quality reward: it rewards longitudinal progress along the lane while
penalising overspeed, drifting out of lane, and harsh / high-speed steering, plus
a per-step time cost and a terminal collision penalty. See
:data:`REWARD_SPEED_WEIGHT` and the sibling weights for the fixed coefficients.

The **observation** is a multi-modal dict of native CARLA sensor payloads:

- ``"gnss"``: ``[lat, lon, alt]`` (always present) — latitude and longitude in
  **degrees** and altitude in metres, from a ``sensor.other.gnss`` reading.
- ``"agents"`` (always present): the ``max_tracked_agents`` agent slots of the
  state flattened, reported **raw** at their true ego-frame poses. The world applies
  no perception, so this is the ground-truth channel; range-gating, occlusion and
  sensor noise are the planner model's observation model, not the world's.
- ``"camera"`` (present iff ``include_camera``): a front-facing RGB image,
  ``(H, W, 3)`` ``uint8``, from a ``sensor.camera.rgb``.
- ``"lidar"`` (present iff ``include_lidar``): a point cloud, ``(N, 4)`` ``float32``
  with rows ``[x, y, z, intensity]`` in the LiDAR **sensor** frame (metres;
  ``intensity`` normalised to ``[0, 1]``), from a ``sensor.lidar.ray_cast``. ``N``
  varies per tick.
- ``"traffic_light"`` (present iff ``include_traffic_light``): ``[should_stop, distance_m]``
  — ``should_stop`` is ``1.0`` when the ego is affected by a **red or yellow** light (else
  ``0.0``) and ``distance_m`` is the forward distance to that light's stop line. This is a
  privileged **ground-truth** read (no noise), letting a planner treat a red light as a
  virtual obstacle to stop for; disable it to withhold the signal entirely.

Any measurement noise is CARLA's own, configured through the sensor blueprint
attributes — the wrapper adds none.

Classes:
    CarlaPOMDP: Forward-only adapter exposing a CARLA session as a world Environment.
"""

import os
import sys
from collections.abc import Hashable
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import Environment, SpaceInfo, SpaceType
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.environments.carla_pomdp.carla_server_pool import acquire_pool_lease
from POMDPPlanners.utils.statistics_utils import confidence_interval

# Default discrete control presets as ``(throttle, steer, brake)`` triples.
DEFAULT_ACTION_PRESETS: Tuple[Tuple[float, float, float], ...] = (
    (0.5, 0.0, 0.0),  # cruise straight
    (0.3, -0.5, 0.0),  # steer left
    (0.3, 0.5, 0.0),  # steer right
    (0.0, 0.0, 1.0),  # brake
)

# Per-step getter roles that may trigger the single CARLA tick. Each role may be
# served from the cached tick at most once, so a repeated role signals a new step.
_ROLE_NEXT_STATE = "next_state"
_ROLE_REWARD = "reward"

# Fixed gym-carla-style reward term weights. The collision weight is the tunable
# ``collision_penalty`` constructor argument; these shape the driving-quality
# terms and are held fixed so the reward is a single well-defined objective.
REWARD_SPEED_WEIGHT = 1.0  # reward per m/s of along-lane (longitudinal) progress
REWARD_FAST_PENALTY = 10.0  # penalty when longitudinal speed exceeds desired_speed
REWARD_OUT_PENALTY = 1.0  # penalty when |lat| exceeds out_lane_thresh
REWARD_STEER_WEIGHT = 5.0  # penalty on squared steering (harsh-steer smoothness)
REWARD_LAT_WEIGHT = 0.2  # penalty on |steer| * longitudinal_speed**2 (fast turns)
REWARD_STEP_COST = 0.1  # constant per-step time cost


def _wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians to the ``[-pi, pi]`` interval."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def driving_quality_reward(
    next_state: np.ndarray,
    steer: float,
    collided: bool,
    desired_speed: float,
    out_lane_thresh: float,
    collision_penalty: float,
    success: bool = False,
    success_reward: float = 0.0,
) -> float:
    """Score a transition with a gym-carla-style driving-quality reward.

    Rewards along-route progress and penalises overspeed, drifting off the route, harsh /
    high-speed steering, each elapsed step, and a terminal collision; reaching the
    destination earns a terminal success bonus. Shared by the :class:`CarlaPOMDP` world
    and the planner-side factored model so the two score a transition identically by
    construction.

    Args:
        next_state: Resulting ego state ``[x, y, yaw(deg), vx, vy, lat, heading_err]``.
        steer: Steering command applied on the transition (from the action preset).
        collided: Whether the transition ended in a terminal collision.
        desired_speed: Target longitudinal speed (m/s); exceeding it is penalised.
        out_lane_thresh: Lateral offset (m) beyond which the ego is treated as off route.
        collision_penalty: Penalty scale applied on a terminal collision.
        success: Whether the transition reached the destination. Defaults to False.
        success_reward: Bonus applied on a successful arrival. Defaults to 0.0.

    Returns:
        The scalar reward for the transition.
    """
    ego_yaw = float(np.radians(next_state[2]))
    vel_x, vel_y = float(next_state[3]), float(next_state[4])
    lateral, heading_err = float(next_state[5]), float(next_state[6])
    lane_yaw = ego_yaw - heading_err
    lspeed_lon = vel_x * np.cos(lane_yaw) + vel_y * np.sin(lane_yaw)

    r_fast = -1.0 if lspeed_lon > desired_speed else 0.0
    r_out = -1.0 if abs(lateral) > out_lane_thresh else 0.0
    r_collision = -1.0 if collided else 0.0
    r_success = 1.0 if success else 0.0
    r_steer = -(steer**2)
    r_lat = -abs(steer) * lspeed_lon**2

    return float(
        collision_penalty * r_collision
        + success_reward * r_success
        + REWARD_SPEED_WEIGHT * lspeed_lon
        + REWARD_FAST_PENALTY * r_fast
        + REWARD_OUT_PENALTY * r_out
        + REWARD_STEER_WEIGHT * r_steer
        + REWARD_LAT_WEIGHT * r_lat
        - REWARD_STEP_COST
    )


# Surrounding-traffic / perception defaults.
DEFAULT_NUM_VEHICLES = 30  # other autopilot vehicles spawned into the world
DEFAULT_NUM_WALKERS = 10  # pedestrians spawned into the world
DEFAULT_MAX_TRACKED_AGENTS = 5  # nearest other vehicles carried in state/observation
DEFAULT_PERCEPTION_RANGE = 50.0  # metres beyond which another agent is undetectable
DEFAULT_OCCLUSION_RADIUS = 1.5  # metres; a vehicle nearer than this to the ego->target
# sight line is treated as blocking (geometric occlusion among vehicles)
DEFAULT_TRAFFIC_MANAGER_PORT = 8000  # CARLA Traffic Manager RPC port

# State/observation layout for other agents. Each tracked agent occupies a fixed
# slot ``[present, rel_x, rel_y, rel_yaw, rel_speed]`` expressed in the ego frame
# (``rel_x`` forward, ``rel_y`` left, ``rel_yaw`` in radians, ``rel_speed`` in m/s).
EGO_STATE_WIDTH = 7  # [x, y, yaw, vx, vy, lat, heading_err]
AGENT_SLOT_WIDTH = 5  # [present, rel_x, rel_y, rel_yaw, rel_speed]

# The ground-truth state ends with a single traffic-light slot
# ``[present, rel_x, rel_y, state_code, time_to_change]`` (ego frame; ``state_code`` is one
# of the ``TRAFFIC_LIGHT_*`` codes below, ``time_to_change`` in seconds). It carries the light
# governing the ego (CARLA affiliates at most one) as ground truth for the evaluation metrics,
# independent of whether the *observation* exposes the light. ``present == 0`` when no light
# affects the ego. The full state width is
# ``EGO_STATE_WIDTH + K*AGENT_SLOT_WIDTH + LIGHT_SLOT_WIDTH + GOAL_SLOT_WIDTH``.
LIGHT_SLOT_WIDTH = 5  # [present, rel_x, rel_y, state_code, time_to_change]

# The ground-truth state additionally ends with a goal slot ``[goal_x, goal_y,
# route_progress_frac]`` (world coordinates; progress is the fraction of the planned
# route's arc length already covered). Like the light slot it is world-side ground
# truth for the evaluation metrics; the planner-model state does not carry it.
GOAL_SLOT_WIDTH = 3  # [goal_x, goal_y, route_progress_frac]

# Route/goal defaults. The route is traced once per reset by CARLA's
# GlobalRoutePlanner at this waypoint sampling resolution (metres).
_ROUTE_RESOLUTION = 2.0
# Forward window (in waypoints, ~20 m at the 2 m resolution) searched when advancing
# the nearest-route-waypoint index. Bounding the search keeps a route that revisits
# the same area (loops, repeated junctions) from snapping the index far ahead; the
# window is far wider than the ego can travel in one tick.
_ROUTE_SEARCH_WINDOW = 10
DEFAULT_GOAL_RADIUS = 5.0  # metres to destination that counts as arrival
DEFAULT_MIN_ROUTE_LENGTH = 100.0  # minimum route length (m) for sampled destinations
DEFAULT_SUCCESS_REWARD = 100.0  # terminal bonus on reaching the destination

# state_code values carried in the traffic-light state slot.
TRAFFIC_LIGHT_GREEN = 0.0
TRAFFIC_LIGHT_RED = 1.0
TRAFFIC_LIGHT_YELLOW = 2.0
TRAFFIC_LIGHT_OFF = 3.0  # light present but not operating (dark / flashing)
TRAFFIC_LIGHT_UNKNOWN = 4.0
# A stop-line crossing is only counted when the ego was moving at least this fast (m/s), so a
# car stopped at the line (whose affiliation later drops) is not mistaken for a pass-through.
_LIGHT_CROSS_MOVING_SPEED = 0.5

# Centre-to-centre ego->vehicle distance (m) at/under which a step counts toward a near-miss.
# A near-miss event is a contiguous run below this threshold that did NOT end in a collision,
# catching close calls the physics collision sensor misses (Roach's BEV-overlap idea).
_NEAR_MISS_DISTANCE = 2.5

# Indices into the ego state block, used by the evaluation metrics.
_EGO_POSITION_SLICE = slice(0, 2)  # (x, y) world position in metres
_EGO_VELOCITY_SLICE = slice(3, 5)  # (vx, vy) world velocity in m/s


class CarlaPOMDPMetrics(Enum):
    """Metric names for the CARLA POMDP environment."""

    COLLISION_RATE = "collision_rate"
    SUCCESS_RATE = "success_rate"
    ROUTE_COMPLETION = "route_completion"
    AVERAGE_PROGRESS = "average_progress"
    AVERAGE_SPEED = "average_speed"
    RED_LIGHT_VIOLATION_RATE = "red_light_violation_rate"
    RED_LIGHT_VIOLATION_COUNT = "red_light_violation_count"
    TRAFFIC_LIGHT_MALFUNCTION_COUNT = "traffic_light_malfunction_count"
    NEAR_MISS_COUNT = "near_miss_count"
    MIN_VEHICLE_DISTANCE = "min_vehicle_distance"


def _relative_agent_row(
    ego_x: float,
    ego_y: float,
    ego_yaw_rad: float,
    other_x: float,
    other_y: float,
    other_yaw_rad: float,
    other_speed: float,
) -> np.ndarray:
    """Express another agent's pose/speed in the ego frame as a present slot row.

    Returns ``[1.0, rel_x, rel_y, rel_yaw, rel_speed]`` with ``rel_x`` pointing
    along the ego heading, ``rel_y`` to its left, and ``rel_yaw`` wrapped to
    ``[-pi, pi]``.
    """
    delta_x = other_x - ego_x
    delta_y = other_y - ego_y
    cos_yaw = np.cos(ego_yaw_rad)
    sin_yaw = np.sin(ego_yaw_rad)
    rel_x = float(cos_yaw * delta_x + sin_yaw * delta_y)
    rel_y = float(-sin_yaw * delta_x + cos_yaw * delta_y)
    rel_yaw = float(_wrap_to_pi(other_yaw_rad - ego_yaw_rad))
    return np.array([1.0, rel_x, rel_y, rel_yaw, float(other_speed)])


def _segment_occludes(
    ego_x: float,
    ego_y: float,
    target_x: float,
    target_y: float,
    blocker_x: float,
    blocker_y: float,
    radius: float,
) -> bool:
    """Whether a blocker vehicle lies on the ego->target sight line within ``radius``.

    Projects the blocker onto the ego->target segment; it occludes when the
    projection falls strictly between the endpoints and its perpendicular distance
    to the line is below ``radius``.
    """
    seg_x = target_x - ego_x
    seg_y = target_y - ego_y
    seg_len_sq = seg_x * seg_x + seg_y * seg_y
    if seg_len_sq == 0.0:
        return False
    param = ((blocker_x - ego_x) * seg_x + (blocker_y - ego_y) * seg_y) / seg_len_sq
    if param <= 0.0 or param >= 1.0:
        return False
    perp_x = ego_x + param * seg_x - blocker_x
    perp_y = ego_y + param * seg_y - blocker_y
    return float(np.hypot(perp_x, perp_y)) < radius


# Default chase RGB camera resolution / field of view (blueprint attributes).
DEFAULT_CAMERA_CONFIG: Dict[str, str] = {
    "image_size_x": "640",
    "image_size_y": "480",
    "fov": "90",
}

# Default front-facing observation RGB camera blueprint attributes.
DEFAULT_OBSERVATION_CAMERA_CONFIG: Dict[str, str] = {
    "image_size_x": "128",
    "image_size_y": "128",
    "fov": "90",
}

# Default roof-mounted LiDAR blueprint attributes.
DEFAULT_LIDAR_CONFIG: Dict[str, str] = {
    "channels": "32",
    "range": "50.0",
    "points_per_second": "100000",
    "rotation_frequency": "20.0",
}


class _CarlaSession:
    """Live CARLA session: server handle, ego vehicle, and attached sensors.

    This is the only object that talks to the ``carla`` package. It exposes a
    small forward-only interface (:meth:`reset` / :meth:`step`) that mirrors a
    Gymnasium env so :class:`CarlaPOMDP` can drive it and tests can substitute a
    scripted fake with the same two methods.
    """

    def __init__(
        self,
        host: str,
        port: int,
        town: str,
        fixed_delta_seconds: float,
        sensor_config: Dict[str, Any],
        vehicle_filter: str,
        timeout: float,
        record_camera: bool = False,
        camera_config: Optional[Dict[str, Any]] = None,
        include_camera: bool = True,
        include_lidar: bool = True,
        include_traffic_light: bool = True,
        observation_camera_config: Optional[Dict[str, Any]] = None,
        lidar_config: Optional[Dict[str, Any]] = None,
        num_vehicles: int = DEFAULT_NUM_VEHICLES,
        num_walkers: int = DEFAULT_NUM_WALKERS,
        max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
        traffic_manager_port: int = DEFAULT_TRAFFIC_MANAGER_PORT,
        randomize_spawn: bool = True,
        observation_extractor: Optional[Callable[[Dict[str, np.ndarray]], Any]] = None,
        destination: Optional[Tuple[float, float]] = None,
        goal_radius: float = DEFAULT_GOAL_RADIUS,
        min_route_length: float = DEFAULT_MIN_ROUTE_LENGTH,
    ) -> None:
        import carla  # pylint: disable=import-outside-toplevel,import-error

        self._carla = carla
        self._destination = destination
        self._goal_radius = goal_radius
        self._min_route_length = min_route_length
        self._observation_extractor = observation_extractor
        self._sensor_config = sensor_config
        self._vehicle_filter = vehicle_filter
        self._num_vehicles = num_vehicles
        self._num_walkers = num_walkers
        self._max_tracked_agents = max_tracked_agents
        self._traffic_manager_port = traffic_manager_port
        self._randomize_spawn = randomize_spawn
        self._record_camera = record_camera
        self._camera_config = camera_config if camera_config is not None else {}
        self._include_camera = include_camera
        self._include_lidar = include_lidar
        self._include_traffic_light = include_traffic_light
        self._obs_camera_config = (
            observation_camera_config if observation_camera_config is not None else {}
        )
        self._lidar_config = lidar_config if lidar_config is not None else {}
        self._camera_width = int(self._obs_camera_config.get("image_size_x", 128))
        self._camera_height = int(self._obs_camera_config.get("image_size_y", 128))
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        self._client = client
        self._world = client.load_world(town)
        self._map = self._world.get_map()
        self._apply_synchronous_settings(fixed_delta_seconds)

        self._vehicle: Optional[Any] = None
        self._gnss_sensor: Optional[Any] = None
        self._collision_sensor: Optional[Any] = None
        self._camera_sensor: Optional[Any] = None
        self._camera_queue: Optional["Queue[Any]"] = None
        self._frames: List[np.ndarray] = []
        self._obs_camera_sensor: Optional[Any] = None
        self._lidar_sensor: Optional[Any] = None
        self._latest_gnss: Optional[np.ndarray] = None
        self._latest_camera: Optional[np.ndarray] = None
        self._latest_lidar: Optional[np.ndarray] = None
        self._collided: bool = False
        self._route_planner: Optional[Any] = None
        self._route_index: int = 0
        # Route polyline live state; filled by _plan_route on each reset.
        self._route_xy = self._route_yaw = self._route_cumlen = self._goal_xy = None
        self._traffic_vehicles: List[Any] = []
        self._walkers: List[Any] = []
        self._walker_controllers: List[Any] = []
        self._rng = np.random.default_rng()

    @property
    def frames(self) -> List[np.ndarray]:
        """RGB chase-camera frames captured so far, one ``(H, W, 3)`` array per tick."""
        return self._frames

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Respawn the ego, surrounding traffic, and sensors, tick once, read start."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._teardown_actors()
        self._frames = []
        self._randomize_weather()
        self._spawn_actors()
        self._plan_route()
        self._spawn_traffic()
        self._collided = False
        self._world.tick()
        self._capture_frame()
        return self._read_state(), self._read_observation()

    def _randomize_weather(self) -> None:
        presets = [
            attr
            for attr in dir(self._carla.WeatherParameters)
            if not attr.startswith("_")
            and isinstance(
                getattr(self._carla.WeatherParameters, attr), self._carla.WeatherParameters
            )
        ]
        if not presets:
            return
        choice = presets[int(self._rng.integers(len(presets)))]
        self._world.set_weather(getattr(self._carla.WeatherParameters, choice))

    def step(
        self, throttle: float, steer: float, brake: float
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], bool, bool]:
        """Apply a control, advance one fixed tick, and read the outcome.

        Returns:
            ``(state, observation, collided, reached_goal)`` — the two terminal
            causes are reported separately so the reward can penalise a collision
            and reward an arrival.
        """
        control = self._carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self._vehicle.apply_control(control)
        self._world.tick()
        self._capture_frame()
        return self._read_state(), self._read_observation(), self._collided, self._reached_goal()

    def _apply_synchronous_settings(self, fixed_delta_seconds: float) -> None:
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta_seconds
        self._world.apply_settings(settings)

    def _spawn_actors(self) -> None:
        blueprint_library = self._world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(self._vehicle_filter)[0]
        spawn_points = self._map.get_spawn_points()
        index = int(self._rng.integers(len(spawn_points))) if self._randomize_spawn else 0
        self._vehicle = self._world.spawn_actor(vehicle_bp, spawn_points[index])
        self._gnss_sensor = self._attach_gnss(blueprint_library)
        self._collision_sensor = self._attach_collision(blueprint_library)
        if self._record_camera:
            self._camera_sensor = self._attach_camera(blueprint_library)
        if self._include_camera:
            self._obs_camera_sensor = self._attach_observation_camera(blueprint_library)
        if self._include_lidar:
            self._lidar_sensor = self._attach_lidar(blueprint_library)

    # ── Route planning (destination/goal support) ────────────────────────
    def _get_route_planner(self) -> Any:
        # GlobalRoutePlanner ships with the CARLA install (PythonAPI/carla/agents),
        # not with the pip wheel, so it is imported from $CARLA_ROOT lazily.
        if self._route_planner is None:
            carla_root = os.environ.get("CARLA_ROOT")
            if not carla_root:
                raise RuntimeError(
                    "Route planning needs the CARLA_ROOT environment variable pointing at "
                    "the CARLA installation; its PythonAPI/carla directory provides "
                    "agents.navigation.global_route_planner."
                )
            api_path = str(Path(carla_root) / "PythonAPI" / "carla")
            if api_path not in sys.path:
                sys.path.insert(0, api_path)
            from agents.navigation.global_route_planner import (  # pylint: disable=import-outside-toplevel,import-error
                GlobalRoutePlanner,
            )

            self._route_planner = GlobalRoutePlanner(self._map, _ROUTE_RESOLUTION)
        return self._route_planner

    def _plan_route(self) -> None:
        # Trace the episode route from the ego spawn to the configured destination,
        # or to a sampled spawn point at least `_min_route_length` metres of route away.
        start = self._vehicle.get_transform().location
        if self._destination is None:
            self._store_route(self._sample_destination_route(start))
            return
        goal = self._carla.Location(
            x=float(self._destination[0]), y=float(self._destination[1]), z=start.z
        )
        route = self._get_route_planner().trace_route(start, goal)
        if len(route) >= 2:
            self._store_route(route)
            return
        # Degenerate trace: the spawn already sits at/beside the destination (possible
        # under randomize_spawn). Use the straight start->goal segment so reset still
        # yields a valid episode, one that terminates in immediate success.
        xy = np.array([[start.x, start.y], [goal.x, goal.y]])
        yaw = np.full(2, float(np.arctan2(goal.y - start.y, goal.x - start.x)))
        self._store_polyline(xy, yaw)

    def _sample_destination_route(self, start: Any) -> List[Any]:
        spawn_points = self._map.get_spawn_points()
        planner = self._get_route_planner()
        for index in self._rng.permutation(len(spawn_points)):
            candidate = spawn_points[int(index)].location
            route = planner.trace_route(start, candidate)
            if self._route_length(route) >= self._min_route_length:
                return route
        raise RuntimeError(
            f"No spawn point yields a route of at least {self._min_route_length} m "
            "from the ego spawn; lower min_route_length or pick another town."
        )

    @staticmethod
    def _route_length(route: Sequence[Any]) -> float:
        locations = [waypoint.transform.location for waypoint, _option in route]
        return float(
            sum(
                np.hypot(second.x - first.x, second.y - first.y)
                for first, second in zip(locations, locations[1:])
            )
        )

    def _store_route(self, route: Sequence[Any]) -> None:
        xy = np.array(
            [
                [waypoint.transform.location.x, waypoint.transform.location.y]
                for waypoint, _option in route
            ]
        )
        yaw = np.array([np.radians(waypoint.transform.rotation.yaw) for waypoint, _option in route])
        self._store_polyline(xy, yaw)

    def _store_polyline(self, xy: np.ndarray, yaw: np.ndarray) -> None:
        # pylint: disable=attribute-defined-outside-init
        self._route_xy = xy
        self._route_yaw = yaw
        segment_lengths = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        self._route_cumlen = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        self._route_index = 0
        self._goal_xy = xy[-1]

    def _reached_goal(self) -> bool:
        if self._goal_xy is None:
            return False
        location = self._vehicle.get_location()
        distance = float(np.hypot(location.x - self._goal_xy[0], location.y - self._goal_xy[1]))
        return distance <= self._goal_radius

    def _spawn_traffic(self) -> None:
        """Populate the world with autopilot vehicles and walking pedestrians."""
        self._spawn_traffic_vehicles()
        self._spawn_walkers()

    def _spawn_traffic_vehicles(self) -> None:
        blueprint_library = self._world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter("vehicle.*")
        traffic_manager = self._client.get_trafficmanager(self._traffic_manager_port)
        traffic_manager.set_synchronous_mode(True)
        spawn_points = self._map.get_spawn_points()
        ego_location = self._vehicle.get_location()
        candidates = [
            point for point in spawn_points if point.location.distance(ego_location) > 1.0
        ]
        # Fill the nearest spawn points first so traffic clusters around the ego
        # (a random permutation scatters it across the whole map, which reads as an
        # empty road on large towns). Blueprint choice stays rng-driven for variety.
        candidates.sort(key=lambda point: point.location.distance(ego_location))
        for point in candidates[: self._num_vehicles]:
            blueprint = vehicle_bps[int(self._rng.integers(len(vehicle_bps)))]
            vehicle: Any = self._world.try_spawn_actor(blueprint, point)
            if vehicle is None:
                continue
            vehicle.set_autopilot(True, traffic_manager.get_port())
            self._traffic_vehicles.append(vehicle)

    def _spawn_walkers(self) -> None:
        blueprint_library = self._world.get_blueprint_library()
        walker_bps = blueprint_library.filter("walker.pedestrian.*")
        controller_bp = blueprint_library.find("controller.ai.walker")
        for _ in range(self._num_walkers):
            location = self._world.get_random_location_from_navigation()
            if location is None:
                continue
            blueprint = walker_bps[int(self._rng.integers(len(walker_bps)))]
            transform = self._carla.Transform(location)
            walker: Any = self._world.try_spawn_actor(blueprint, transform)
            if walker is None:
                continue
            controller: Any = self._world.spawn_actor(
                controller_bp, self._carla.Transform(), attach_to=walker
            )
            controller.start()
            controller.go_to_location(self._world.get_random_location_from_navigation())
            self._walkers.append(walker)
            self._walker_controllers.append(controller)

    def _attach_gnss(self, blueprint_library: Any) -> Any:
        gnss_bp = blueprint_library.find("sensor.other.gnss")
        for attribute, value in self._sensor_config.items():
            gnss_bp.set_attribute(attribute, str(value))
        sensor: Any = self._world.spawn_actor(
            gnss_bp, self._carla.Transform(), attach_to=self._vehicle
        )
        sensor.listen(self._on_gnss)
        return sensor

    def _attach_collision(self, blueprint_library: Any) -> Any:
        collision_bp = blueprint_library.find("sensor.other.collision")
        sensor: Any = self._world.spawn_actor(
            collision_bp, self._carla.Transform(), attach_to=self._vehicle
        )
        sensor.listen(self._on_collision)
        return sensor

    def _attach_camera(self, blueprint_library: Any) -> Any:
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        for attribute, value in self._camera_config.items():
            camera_bp.set_attribute(attribute, str(value))
        # Chase view: behind and above the ego, tilted slightly down.
        transform = self._carla.Transform(
            self._carla.Location(x=-6.0, z=3.0),
            self._carla.Rotation(pitch=-15.0),
        )
        sensor: Any = self._world.spawn_actor(camera_bp, transform, attach_to=self._vehicle)
        self._camera_queue = Queue()
        sensor.listen(self._camera_queue.put)
        return sensor

    def _attach_observation_camera(self, blueprint_library: Any) -> Any:
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        for attribute, value in self._obs_camera_config.items():
            camera_bp.set_attribute(attribute, str(value))
        # Front-facing view: ahead of and above the ego, no rotation.
        transform = self._carla.Transform(self._carla.Location(x=1.5, z=2.4))
        sensor: Any = self._world.spawn_actor(camera_bp, transform, attach_to=self._vehicle)
        sensor.listen(self._on_camera)
        return sensor

    def _attach_lidar(self, blueprint_library: Any) -> Any:
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        for attribute, value in self._lidar_config.items():
            lidar_bp.set_attribute(attribute, str(value))
        # Roof-mounted, centered on the ego.
        transform = self._carla.Transform(self._carla.Location(z=2.4))
        sensor: Any = self._world.spawn_actor(lidar_bp, transform, attach_to=self._vehicle)
        sensor.listen(self._on_lidar)
        return sensor

    def _capture_frame(self) -> None:
        """Pull the RGB frame CARLA rendered for the tick just executed."""
        if not self._record_camera or self._camera_queue is None:
            return
        image = self._camera_queue.get()
        buffer = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = np.reshape(buffer, (image.height, image.width, 4))
        # CARLA stores BGRA; take B,G,R reversed to RGB and drop alpha.
        self._frames.append(np.ascontiguousarray(bgra[:, :, 2::-1]))

    def _on_gnss(self, measurement: Any) -> None:
        self._latest_gnss = np.array(
            [measurement.latitude, measurement.longitude, measurement.altitude]
        )

    def _on_camera(self, image: Any) -> None:
        buffer = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = np.reshape(buffer, (image.height, image.width, 4))
        # CARLA stores BGRA; take B,G,R reversed to RGB and drop alpha.
        self._latest_camera = np.ascontiguousarray(bgra[:, :, 2::-1])

    def _on_lidar(self, data: Any) -> None:
        self._latest_lidar = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)

    def _on_collision(self, event: Any) -> None:
        del event
        self._collided = True

    def _read_state(self) -> np.ndarray:
        transform = self._vehicle.get_transform()
        velocity = self._vehicle.get_velocity()
        lateral, heading_err = self._lane_geometry(transform.location, transform.rotation.yaw)
        ego = np.array(
            [
                transform.location.x,
                transform.location.y,
                transform.rotation.yaw,
                velocity.x,
                velocity.y,
                lateral,
                heading_err,
            ]
        )
        agent_rows = self._agent_rows()
        return np.concatenate(
            [
                ego,
                agent_rows.reshape(-1),
                self._read_state_traffic_light(),
                self._read_state_goal(),
            ]
        )

    def _read_state_goal(self) -> np.ndarray:
        """Ground-truth goal slot ``[goal_x, goal_y, route_progress_frac]``.

        Records the episode destination and the fraction of the planned route's arc
        length already covered, for the evaluation metrics. All zeros when the session
        has no route (e.g. unit-test fixtures).
        """
        if self._goal_xy is None or self._route_cumlen is None:
            return np.zeros(GOAL_SLOT_WIDTH)
        total = float(self._route_cumlen[-1])
        progress = float(self._route_cumlen[self._route_index]) / total if total > 0 else 1.0
        return np.array([self._goal_xy[0], self._goal_xy[1], progress])

    def _read_state_traffic_light(self) -> np.ndarray:
        """Ground-truth light slot ``[present, rel_x, rel_y, state_code, time_to_change]``.

        Carries the light governing the ego (``present == 0`` when none affects it). This is
        the state's ground-truth record used by the evaluation metrics, distinct from the
        simplified ``traffic_light`` observation the planner may consume.
        """
        light = self._vehicle.get_traffic_light()
        if light is None:
            return np.zeros(LIGHT_SLOT_WIDTH)
        rel_x, rel_y = self._stop_line_rel_position(light)
        code = self._traffic_light_code(light.get_state())
        return np.array([1.0, rel_x, rel_y, code, self._time_to_change(light)])

    def _traffic_light_code(self, state: Any) -> float:
        return {
            self._carla.TrafficLightState.Green: TRAFFIC_LIGHT_GREEN,
            self._carla.TrafficLightState.Red: TRAFFIC_LIGHT_RED,
            self._carla.TrafficLightState.Yellow: TRAFFIC_LIGHT_YELLOW,
            self._carla.TrafficLightState.Off: TRAFFIC_LIGHT_OFF,
        }.get(state, TRAFFIC_LIGHT_UNKNOWN)

    def _stop_line_rel_position(self, light: Any) -> Tuple[float, float]:
        ego_transform = self._vehicle.get_transform()
        ego_x, ego_y = ego_transform.location.x, ego_transform.location.y
        ego_yaw = np.radians(ego_transform.rotation.yaw)
        waypoints = light.get_stop_waypoints()
        if waypoints:
            nearest = min(
                waypoints, key=lambda wp: wp.transform.location.distance(ego_transform.location)
            )
            target = nearest.transform.location
        else:
            target = light.get_location()
        row = _relative_agent_row(ego_x, ego_y, ego_yaw, target.x, target.y, 0.0, 0.0)
        return float(row[1]), float(row[2])

    def _time_to_change(self, light: Any) -> float:
        duration = {
            self._carla.TrafficLightState.Red: light.get_red_time,
            self._carla.TrafficLightState.Yellow: light.get_yellow_time,
            self._carla.TrafficLightState.Green: light.get_green_time,
        }.get(light.get_state())
        if duration is None:
            return 0.0
        return float(max(0.0, duration() - light.get_elapsed_time()))

    def _nearest_agents(self) -> List[Any]:
        """The ``max_tracked_agents`` other vehicles nearest the ego, by true distance."""
        ego_location = self._vehicle.get_location()
        others = [
            actor
            for actor in self._world.get_actors().filter("vehicle.*")
            if actor.id != self._vehicle.id
        ]
        others.sort(key=lambda actor: actor.get_location().distance(ego_location))
        return others[: self._max_tracked_agents]

    def _agent_rows(self) -> np.ndarray:
        """Fixed ``(K, AGENT_SLOT_WIDTH)`` matrix of the nearest-agent slots in ego frame.

        Every nearest agent is reported at its true ego-frame pose; the world applies no
        perception. Range-gating and occlusion are the planner model's observation model,
        so the world's ``agents`` block is the raw ground-truth channel.
        """
        rows = np.zeros((self._max_tracked_agents, AGENT_SLOT_WIDTH))
        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_x, ego_y = ego_location.x, ego_location.y
        ego_yaw = np.radians(ego_transform.rotation.yaw)
        nearest = self._nearest_agents()
        for slot, actor in enumerate(nearest):
            other_transform = actor.get_transform()
            other_velocity = actor.get_velocity()
            rows[slot] = _relative_agent_row(
                ego_x,
                ego_y,
                ego_yaw,
                other_transform.location.x,
                other_transform.location.y,
                np.radians(other_transform.rotation.yaw),
                float(np.hypot(other_velocity.x, other_velocity.y)),
            )
        return rows

    def _lane_geometry(self, location: Any, yaw_deg: float) -> Tuple[float, float]:
        """Signed lateral offset (m) and heading error (rad) w.r.t. the route.

        Projects the ego onto the nearest waypoint of the planned route, then returns
        how far it sits to the side of the route line and how far its heading deviates
        from the route direction (wrapped to ``[-pi, pi]``). When no route exists (a
        session built without one, e.g. in unit tests) the reference falls back to the
        centre of the nearest driving lane via the CARLA map.
        """
        if self._route_xy is not None:
            return self._route_geometry(location, yaw_deg)
        waypoint = self._map.get_waypoint(
            location, project_to_road=True, lane_type=self._carla.LaneType.Driving
        )
        lane_transform = waypoint.transform
        lane_yaw = np.radians(lane_transform.rotation.yaw)
        delta_x = location.x - lane_transform.location.x
        delta_y = location.y - lane_transform.location.y
        lateral = float(-np.sin(lane_yaw) * delta_x + np.cos(lane_yaw) * delta_y)
        heading_err = float(_wrap_to_pi(np.radians(yaw_deg) - lane_yaw))
        return lateral, heading_err

    def _route_geometry(self, location: Any, yaw_deg: float) -> Tuple[float, float]:
        # The nearest-waypoint index only ever advances, and only within a bounded
        # forward window, so on self-crossing routes the projection can neither snap
        # back to an earlier segment nor jump ahead to a later pass through the area.
        assert self._route_xy is not None and self._route_yaw is not None
        window = self._route_xy[self._route_index : self._route_index + _ROUTE_SEARCH_WINDOW]
        distances = np.hypot(window[:, 0] - location.x, window[:, 1] - location.y)
        self._route_index += int(np.argmin(distances))
        reference_xy = self._route_xy[self._route_index]
        reference_yaw = float(self._route_yaw[self._route_index])
        delta_x = location.x - reference_xy[0]
        delta_y = location.y - reference_xy[1]
        lateral = float(-np.sin(reference_yaw) * delta_x + np.cos(reference_yaw) * delta_y)
        heading_err = float(_wrap_to_pi(np.radians(yaw_deg) - reference_yaw))
        return lateral, heading_err

    def _read_observation(self) -> Dict[str, np.ndarray]:
        observation: Dict[str, np.ndarray] = {
            "gnss": self._read_gnss(),
            "agents": self._agent_rows().reshape(-1),
        }
        if self._include_traffic_light:
            observation["traffic_light"] = self._read_traffic_light()
        if self._include_camera:
            observation["camera"] = self._read_camera()
        if self._include_lidar:
            observation["lidar"] = self._read_lidar()
        if self._observation_extractor is not None:
            return self._observation_extractor(observation)
        return observation

    def _read_traffic_light(self) -> np.ndarray:
        """Stop signal for a red/yellow light ahead: ``[should_stop, distance_m]``.

        ``should_stop`` is 1.0 when the ego is affected by a red or yellow light (else
        0.0); ``distance_m`` is the forward distance to that light's stop line. This lets
        a planner treat a red light as a virtual obstacle to brake for.
        """
        state = self._vehicle.get_traffic_light_state()
        stop_states = (self._carla.TrafficLightState.Red, self._carla.TrafficLightState.Yellow)
        if state not in stop_states:
            return np.array([0.0, 0.0])
        return np.array([1.0, self._stop_line_distance(self._vehicle.get_traffic_light())])

    def _stop_line_distance(self, light: Any) -> float:
        if light is None:
            return 0.0
        ego = self._vehicle.get_location()
        try:
            waypoints = light.get_stop_waypoints()
        except (AttributeError, RuntimeError):
            waypoints = []
        if waypoints:
            return float(min(wp.transform.location.distance(ego) for wp in waypoints))
        return float(light.get_location().distance(ego))

    def _read_gnss(self) -> np.ndarray:
        if self._latest_gnss is None:
            return np.zeros(3)
        return self._latest_gnss

    def _read_camera(self) -> np.ndarray:
        if self._latest_camera is None:
            return np.zeros((self._camera_height, self._camera_width, 3), dtype=np.uint8)
        return self._latest_camera

    def _read_lidar(self) -> np.ndarray:
        if self._latest_lidar is None:
            return np.zeros((0, 4), dtype=np.float32)
        return self._latest_lidar

    def _teardown_actors(self) -> None:
        for controller in self._walker_controllers:
            controller.stop()
        for actor in (
            *self._walker_controllers,
            *self._walkers,
            *self._traffic_vehicles,
            self._camera_sensor,
            self._obs_camera_sensor,
            self._lidar_sensor,
            self._collision_sensor,
            self._gnss_sensor,
            self._vehicle,
        ):
            if actor is not None:
                actor.destroy()
        self._traffic_vehicles = []
        self._walkers = []
        self._walker_controllers = []
        self._vehicle = None
        self._gnss_sensor = None
        self._collision_sensor = None
        self._camera_sensor = None
        self._camera_queue = None
        self._obs_camera_sensor = None
        self._lidar_sensor = None
        self._latest_gnss = None
        self._latest_camera = None
        self._latest_lidar = None


class CarlaPOMDP(Environment):
    """Forward-only adapter exposing a CARLA session as a world POMDP.

    The wrapper drives a CARLA server as the ground-truth world of an episode. It
    ticks the simulator exactly once per real interaction and serves the resulting
    next state, observation and reward from a small cache, because the
    POMDPPlanners episode loop requests those three quantities through separate
    method calls while CARLA produces them atomically. The state is the ego
    vehicle's ground-truth kinematics; the observation is a native CARLA sensor
    payload (GNSS by default), so the world is genuinely partially observed. See
    the module docstring for the exact state and observation variables, units,
    and frames.

    Note:
        This is a *world* environment, not a generative model. It cannot sample a
        transition from an arbitrary state, so belief particle propagation and
        density queries are unsupported and raise ``NotImplementedError`` /
        ``RuntimeError``. Pair it with a generative model environment on the
        planner (``policy.environment``).

    Attributes:
        host: CARLA server host.
        port: CARLA server RPC port.
        town: CARLA map name loaded on reset.
        sensor_config: GNSS blueprint attributes (e.g. noise stddev) forwarded to
            the sensor; measurement noise, if any, is CARLA's own.
        action_presets: Discrete ``(throttle, steer, brake)`` control triples.
        seed: Optional seed applied to the first ``reset`` for reproducibility.

    Example:
        The environment is used as the forward-only world of an
        :class:`~POMDPPlanners.simulations.episodes.EpisodeRunner`, paired with a
        separate generative model on the planner. It requires a running CARLA
        server, so this snippet is illustrative rather than executed::

            env = CarlaPOMDP(discount_factor=0.95, town="Town03")
            state = env.initial_state_dist().sample()[0]
            next_state, observation, reward = env.sample_next_step(state, 0)
            # state is [ego(7), nearest-agent slots...]; observation is a
            # gnss/agents/camera/lidar dict hiding out-of-range / occluded agents.
    """

    def __init__(
        self,
        discount_factor: float,
        host: str = "localhost",
        port: int = 2000,
        town: str = "Town03",
        sensor_config: Optional[Dict[str, Any]] = None,
        action_presets: Optional[Sequence[Tuple[float, float, float]]] = None,
        record_camera: bool = False,
        camera_config: Optional[Dict[str, Any]] = None,
        include_camera: bool = True,
        include_lidar: bool = True,
        include_traffic_light: bool = True,
        observation_camera_config: Optional[Dict[str, Any]] = None,
        lidar_config: Optional[Dict[str, Any]] = None,
        fixed_delta_seconds: float = 0.05,
        collision_penalty: float = 100.0,
        desired_speed: float = 8.0,
        out_lane_thresh: float = 2.0,
        destination: Optional[Tuple[float, float]] = None,
        goal_radius: float = DEFAULT_GOAL_RADIUS,
        min_route_length: float = DEFAULT_MIN_ROUTE_LENGTH,
        success_reward: float = DEFAULT_SUCCESS_REWARD,
        num_vehicles: int = DEFAULT_NUM_VEHICLES,
        num_walkers: int = DEFAULT_NUM_WALKERS,
        max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
        traffic_manager_port: int = DEFAULT_TRAFFIC_MANAGER_PORT,
        server_pool_dir: Optional[Union[str, Path]] = None,
        randomize_spawn: bool = True,
        observation_extractor: Optional[Callable[[Dict[str, np.ndarray]], Any]] = None,
        vehicle_filter: str = "vehicle.tesla.model3",
        timeout: float = 10.0,
        seed: Optional[int] = None,
        name: Optional[str] = None,
        reward_range: Optional[Tuple[float, float]] = None,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the CARLA world environment.

        Args:
            discount_factor: Discount factor for future rewards (0 < d <= 1).
            host: CARLA server host. Defaults to ``"localhost"``.
            port: CARLA server RPC port. Defaults to 2000.
            town: CARLA map loaded on reset. Defaults to ``"Town03"``.
            sensor_config: GNSS blueprint attributes (e.g. ``noise_lat_stddev``)
                forwarded to the sensor. Defaults to none (clean readings).
            action_presets: Discrete ``(throttle, steer, brake)`` triples. Defaults
                to :data:`DEFAULT_ACTION_PRESETS`.
            record_camera: If True, attach a chase RGB camera to the ego and buffer
                one rendered frame per tick for :meth:`save_camera_video`. Adds GPU
                render cost, so it defaults to False.
            camera_config: RGB camera blueprint attributes (e.g. ``image_size_x``)
                overriding :data:`DEFAULT_CAMERA_CONFIG`. Defaults to none.
            include_camera: If True, attach a front-facing RGB observation camera
                and include its frame under the ``"camera"`` observation key.
                Defaults to True.
            include_lidar: If True, attach a roof-mounted LiDAR and include its
                point cloud under the ``"lidar"`` observation key. Defaults to True.
            include_traffic_light: If True, include the ground-truth ``"traffic_light"``
                ``[should_stop, distance_m]`` signal in the observation. Set False to
                withhold the privileged light oracle entirely. Defaults to True.
            observation_camera_config: Front camera blueprint attributes overriding
                :data:`DEFAULT_OBSERVATION_CAMERA_CONFIG`. Defaults to none.
            lidar_config: LiDAR blueprint attributes overriding
                :data:`DEFAULT_LIDAR_CONFIG`. Defaults to none.
            fixed_delta_seconds: Synchronous-mode tick length. Defaults to 0.05.
            collision_penalty: Reward penalty applied on a terminal collision.
                Defaults to 100.0.
            desired_speed: Target longitudinal speed in m/s; exceeding it incurs the
                overspeed penalty. Defaults to 8.0.
            out_lane_thresh: Lateral offset in metres beyond which the ego is treated
                as out of lane and penalised. Defaults to 2.0.
            destination: Optional goal ``(x, y)`` in world coordinates. A route from
                the ego spawn to the destination is traced each reset and the ego
                state's ``lat``/``heading_err`` are measured against it. The effective
                goal is the traced route's final waypoint — the destination projected
                onto the drivable road network — so an off-road coordinate resolves to
                the nearest reachable road point. When None, a destination is sampled
                each reset from the map's spawn points with a route length of at least
                ``min_route_length`` (reproducible via ``seed``). Defaults to None.
            goal_radius: Distance to the destination (m) at which the episode
                terminates in success. Defaults to :data:`DEFAULT_GOAL_RADIUS`.
            min_route_length: Minimum route length (m) required of sampled
                destinations. Defaults to :data:`DEFAULT_MIN_ROUTE_LENGTH`.
            success_reward: Terminal reward bonus on reaching the destination.
                Defaults to :data:`DEFAULT_SUCCESS_REWARD`.
            num_vehicles: Number of autopilot traffic vehicles spawned into the
                world each reset. Defaults to :data:`DEFAULT_NUM_VEHICLES`.
            num_walkers: Number of walking pedestrians spawned into the world each
                reset. Defaults to :data:`DEFAULT_NUM_WALKERS`.
            max_tracked_agents: Number of nearest other vehicles carried as fixed
                slots in the state and observation. Defaults to
                :data:`DEFAULT_MAX_TRACKED_AGENTS`.
            traffic_manager_port: CARLA Traffic Manager RPC port used to drive the
                traffic vehicles. Defaults to :data:`DEFAULT_TRAFFIC_MANAGER_PORT`.
            server_pool_dir: Optional directory of a
                :class:`~POMDPPlanners.environments.carla_pomdp.carla_server_pool.CarlaServerPool`.
                When set, ``host``/``port``/``traffic_manager_port`` are overridden
                by a per-process server lease acquired lazily on first session
                build, so parallel workers each connect to their own pool server.
                The pool must outlive the environment. Defaults to None.
            randomize_spawn: If True, sample a random ego spawn point (and weather)
                each reset instead of a fixed one. Defaults to True.
            observation_extractor: Optional callable applied to the full
                ``{gnss, agents, camera, lidar}`` observation dict each step,
                returning the observation actually emitted (e.g. a subset of the
                keys or a flattened vector). Must be a picklable (module-level)
                callable, since the environment is pickled for distributed runs.
                Defaults to None, which emits the full dict unchanged.
            vehicle_filter: Blueprint filter for the ego vehicle. Defaults to
                ``"vehicle.tesla.model3"``.
            timeout: CARLA client timeout in seconds. Defaults to 10.0.
            seed: Optional seed applied to the first ``reset``. Defaults to None.
            name: Environment identifier. Defaults to ``"CarlaPOMDP-<town>"``.
            reward_range: Optional ``(min, max)`` reward bounds. Defaults to None.
            output_dir: Optional directory for logging output. Defaults to None.
            debug: Enable debug logging. Defaults to False.
            use_queue_logger: Whether to use queue-based logging. Defaults to False.
        """
        self.host = host
        self.port = port
        self.town = town
        self.sensor_config: Dict[str, Any] = dict(sensor_config) if sensor_config else {}
        presets = action_presets if action_presets is not None else DEFAULT_ACTION_PRESETS
        self.action_presets: List[Tuple[float, float, float]] = [
            (float(throttle), float(steer), float(brake)) for throttle, steer, brake in presets
        ]
        self.record_camera = record_camera
        self.camera_config: Dict[str, Any] = dict(DEFAULT_CAMERA_CONFIG)
        if camera_config:
            self.camera_config.update(camera_config)
        self.include_camera = include_camera
        self.include_lidar = include_lidar
        self.include_traffic_light = include_traffic_light
        self.observation_camera_config: Dict[str, Any] = dict(DEFAULT_OBSERVATION_CAMERA_CONFIG)
        if observation_camera_config:
            self.observation_camera_config.update(observation_camera_config)
        self.lidar_config: Dict[str, Any] = dict(DEFAULT_LIDAR_CONFIG)
        if lidar_config:
            self.lidar_config.update(lidar_config)
        self.fixed_delta_seconds = fixed_delta_seconds
        self.collision_penalty = collision_penalty
        self.desired_speed = desired_speed
        self.out_lane_thresh = out_lane_thresh
        self.destination = (
            (float(destination[0]), float(destination[1])) if destination is not None else None
        )
        self.goal_radius = goal_radius
        self.min_route_length = min_route_length
        self.success_reward = success_reward
        self.num_vehicles = num_vehicles
        self.num_walkers = num_walkers
        self.max_tracked_agents = max_tracked_agents
        self.traffic_manager_port = traffic_manager_port
        self.server_pool_dir = str(server_pool_dir) if server_pool_dir is not None else None
        self.randomize_spawn = randomize_spawn
        self.observation_extractor = observation_extractor
        self.vehicle_filter = vehicle_filter
        self.timeout = timeout
        self.seed = seed

        # Live-session state: rebuilt lazily and never serialized.
        self._session: Optional[Any] = None
        self._live_state: Optional[np.ndarray] = None
        self._latest_obs: Optional[Dict[str, np.ndarray]] = None
        self._terminated: bool = False
        self._seeded: bool = False
        self._pending: Optional[Dict[str, Any]] = None
        self._served_roles: Set[str] = set()

        # Action space is discrete (preset index); observation is a continuous
        # sensor vector. Both are fixed by design, so no live server is needed at
        # construction time (unlike GymPOMDP, which must probe gym spaces).
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.CONTINUOUS,
        )

        super().__init__(
            discount_factor=discount_factor,
            name=name if name is not None else f"CarlaPOMDP-{town}",
            space_info=space_info,
            reward_range=reward_range,
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

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
    def _resolve_connection(self) -> Tuple[str, int, int]:
        # Without a pool the static endpoints apply; with one, this process's
        # lease (acquired once, cached for the process lifetime) overrides them.
        if self.server_pool_dir is None:
            return self.host, self.port, self.traffic_manager_port
        lease = acquire_pool_lease(self.server_pool_dir)
        return lease.host, lease.rpc_port, lease.traffic_manager_port

    def _get_session(self) -> Any:
        if self._session is None:
            host, port, traffic_manager_port = self._resolve_connection()
            self._session = _CarlaSession(
                host=host,
                port=port,
                town=self.town,
                fixed_delta_seconds=self.fixed_delta_seconds,
                sensor_config=self.sensor_config,
                vehicle_filter=self.vehicle_filter,
                timeout=self.timeout,
                record_camera=self.record_camera,
                camera_config=self.camera_config,
                include_camera=self.include_camera,
                include_lidar=self.include_lidar,
                include_traffic_light=self.include_traffic_light,
                observation_camera_config=self.observation_camera_config,
                lidar_config=self.lidar_config,
                num_vehicles=self.num_vehicles,
                num_walkers=self.num_walkers,
                max_tracked_agents=self.max_tracked_agents,
                traffic_manager_port=traffic_manager_port,
                randomize_spawn=self.randomize_spawn,
                observation_extractor=self.observation_extractor,
                destination=self.destination,
                goal_radius=self.goal_radius,
                min_route_length=self.min_route_length,
            )
        return self._session

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

    def _to_control(self, action: Any) -> Tuple[float, float, float]:
        return self.action_presets[int(action)]

    def _states_equal(self, state_a: Any, state_b: Any) -> bool:
        return np.array_equal(np.asarray(state_a), np.asarray(state_b))

    def _compute_reward(
        self, next_state: np.ndarray, action: Any, collided: bool, success: bool
    ) -> float:
        """Score a transition with a gym-carla-style driving-quality reward.

        Rewards along-route progress and penalises overspeed, drifting off route,
        harsh / high-speed steering, each elapsed step, and a terminal collision;
        reaching the destination earns the ``success_reward`` bonus.
        """
        _, steer, _ = self._to_control(action)
        return driving_quality_reward(
            next_state,
            steer,
            collided,
            self.desired_speed,
            self.out_lane_thresh,
            self.collision_penalty,
            success=success,
            success_reward=self.success_reward,
        )

    def _ensure_stepped(self, state: Any, action: Any, role: str) -> Dict[str, Any]:
        """Advance the world one tick for ``(state, action)`` (once) and cache it.

        The reward and next-state getters of a single episode step share one tick:
        the first of them advances the world and caches the outcome, the second is
        served from that cache. The cache is keyed on ``(state, action)`` *and* on
        the requesting ``role`` — each role is served from a given tick at most
        once, so a repeated role means a new step and forces another tick even when
        the ego was momentarily stationary and ``next_state`` equals ``state``.
        Raises when asked to step from a state other than the live one.
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
                "CarlaPOMDP is a forward-only world environment; it cannot resample "
                "from an arbitrary state. Give the planner a separate model "
                "environment (policy.environment) and only step the world forward "
                "from its live state."
            )

        session = self._get_session()
        throttle, steer, brake = self._to_control(action)
        next_state, observation, collided, reached_goal = session.step(throttle, steer, brake)
        next_state = np.asarray(next_state)
        collided = bool(collided)
        reached_goal = bool(reached_goal)
        done = collided or reached_goal
        pending = {
            "state": np.asarray(state).copy(),
            "action": action,
            "next_state": next_state,
            "observation": observation,
            "reward": self._compute_reward(next_state, action, collided, reached_goal),
            "terminated": done,
            "collided": collided,
            "reached_goal": reached_goal,
        }
        self._pending = pending
        self._served_roles = {role}
        self._live_state = next_state
        self._latest_obs = observation
        self._terminated = done
        return pending

    # ── Environment interface ───────────────────────────────────────────
    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        if n_samples != 1:
            raise ValueError("CarlaPOMDP is forward-only and only supports n_samples=1")
        return self._ensure_stepped(state, action, _ROLE_NEXT_STATE)["next_state"]

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        del action
        if n_samples != 1:
            raise ValueError("CarlaPOMDP is forward-only and only supports n_samples=1")
        if self._pending is not None and self._states_equal(
            self._pending["next_state"], next_state
        ):
            return self._pending["observation"]
        raise RuntimeError(
            "CarlaPOMDP.sample_observation was queried for a next state other than "
            "the live one; a forward-only world only knows the sensor reading it "
            "just produced by stepping."
        )

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        del next_state
        return self._ensure_stepped(state, action, _ROLE_REWARD)["reward"]

    def is_terminal(self, state: Any) -> bool:
        if self._live_state is not None and not self._states_equal(state, self._live_state):
            raise RuntimeError(
                "CarlaPOMDP.is_terminal was queried for a state other than the live "
                "world state; this forward-only world only knows whether its "
                "current state is terminal."
            )
        return self._terminated

    def get_metric_names(self) -> List[str]:
        """Names of the CARLA-specific evaluation metrics.

        Returns:
            The metric name strings produced by :meth:`compute_metrics`:
            ``collision_rate``, ``success_rate``, ``route_completion``,
            ``average_progress``, ``average_speed``, ``red_light_violation_rate``,
            ``red_light_violation_count``, ``traffic_light_malfunction_count``,
            ``near_miss_count`` and ``min_vehicle_distance``.
        """
        return [metric.value for metric in CarlaPOMDPMetrics]

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        """Compute CARLA driving-quality metrics from episode histories.

        Args:
            histories: Episode histories to summarise.

        Returns:
            A list of :class:`MetricValue` with 95% confidence bounds across
            episodes:

            - ``collision_rate``: fraction of episodes that ended in a terminal state
              without reaching the destination (i.e. in a collision).
            - ``success_rate``: fraction of episodes whose final state is within
              ``goal_radius`` of the episode destination.
            - ``route_completion``: mean over episodes of the fraction of the planned
              route's arc length covered by the end of the episode.
            - ``average_progress``: mean per-episode ground distance travelled by
              the ego, in metres.
            - ``average_speed``: mean ego speed over the driven trajectory, in m/s.
            - ``red_light_violation_rate``: fraction of *functioning*-light stop-line
              crossings taken while the light was red (averaged over episodes that crossed
              at least one working light).
            - ``red_light_violation_count``: mean number of red-light crossings per episode.
            - ``traffic_light_malfunction_count``: mean number of crossings per episode where
              the light was off / unknown — recorded separately and never counted as a
              violation, since the light was not operating.
            - ``near_miss_count``: mean number of near-miss events per episode (a run within
              ``_NEAR_MISS_DISTANCE`` of another vehicle that did not become a collision).
            - ``min_vehicle_distance``: mean over episodes of the closest the ego came to any
              vehicle, in metres (a safety-margin metric; episodes that saw no vehicle are
              excluded).
        """
        if not histories:
            return []
        outcomes = [self._episode_goal_outcome(h) for h in histories]
        successes = [success for success, _completion in outcomes]
        completions = [completion for _success, completion in outcomes]
        collisions = [
            1.0 if history.reach_terminal_state and success == 0.0 else 0.0
            for history, (success, _completion) in zip(histories, outcomes)
        ]
        path_lengths = [self._episode_path_length(h) for h in histories if h.history]
        mean_speeds = [self._episode_mean_speed(h) for h in histories if h.history]
        events = [self._episode_traffic_light_events(h) for h in histories if h.history]
        red_counts = [float(red) for red, _functioning, _malfunction in events]
        red_rates = [red / functioning for red, functioning, _m in events if functioning > 0]
        malfunctions = [float(malfunction) for _red, _functioning, malfunction in events]
        near = [self._episode_near_misses(h) for h in histories if h.history]
        near_counts = [float(count) for count, _min_distance in near]
        min_distances = [dist for _count, dist in near if np.isfinite(dist)]
        return [
            self._metric_from_samples(CarlaPOMDPMetrics.COLLISION_RATE.value, collisions),
            self._metric_from_samples(CarlaPOMDPMetrics.SUCCESS_RATE.value, successes),
            self._metric_from_samples(CarlaPOMDPMetrics.ROUTE_COMPLETION.value, completions),
            self._metric_from_samples(CarlaPOMDPMetrics.AVERAGE_PROGRESS.value, path_lengths),
            self._metric_from_samples(CarlaPOMDPMetrics.AVERAGE_SPEED.value, mean_speeds),
            self._metric_from_samples(CarlaPOMDPMetrics.RED_LIGHT_VIOLATION_RATE.value, red_rates),
            self._metric_from_samples(
                CarlaPOMDPMetrics.RED_LIGHT_VIOLATION_COUNT.value, red_counts
            ),
            self._metric_from_samples(
                CarlaPOMDPMetrics.TRAFFIC_LIGHT_MALFUNCTION_COUNT.value, malfunctions
            ),
            self._metric_from_samples(CarlaPOMDPMetrics.NEAR_MISS_COUNT.value, near_counts),
            self._metric_from_samples(CarlaPOMDPMetrics.MIN_VEHICLE_DISTANCE.value, min_distances),
        ]

    def _episode_near_misses(self, history: History) -> Tuple[int, float]:
        """Return ``(near_miss_events, min_ego_vehicle_distance)`` for one episode.

        A near-miss event is a contiguous run of steps whose closest other vehicle is within
        ``_NEAR_MISS_DISTANCE`` (centre-to-centre, ego frame). ``min_ego_vehicle_distance`` is
        the closest the ego came to any vehicle over the episode (``inf`` if it never saw one).
        States lacking agent slots (non-CARLA fixtures) contribute nothing.
        """
        agents_end = EGO_STATE_WIDTH + self.max_tracked_agents * AGENT_SLOT_WIDTH
        distances: List[float] = []
        for step in history.history:
            state = np.asarray(step.state, dtype=float)
            if len(state) < agents_end:
                continue
            rows = state[EGO_STATE_WIDTH:agents_end].reshape(
                self.max_tracked_agents, AGENT_SLOT_WIDTH
            )
            present = rows[rows[:, 0] == 1.0]
            distances.append(
                float(np.min(np.hypot(present[:, 1], present[:, 2]))) if len(present) else np.inf
            )
        events = was_near = 0
        for distance in distances:
            near = distance < _NEAR_MISS_DISTANCE
            events += 1 if (near and not was_near) else 0
            was_near = near
        finite = [distance for distance in distances if np.isfinite(distance)]
        return events, (min(finite) if finite else float("inf"))

    def _episode_traffic_light_events(self, history: History) -> Tuple[int, int, int]:
        """Count ``(red_runs, functioning_crossings, malfunction_crossings)`` for one episode.

        A stop-line crossing is an affiliation drop (light ``present`` 1 -> 0) while the ego is
        moving. It is a *red run* when the light was red at the crossing, a *functioning*
        crossing when the light was operating (red / yellow / green), and a *malfunction*
        crossing when the light was off / unknown (recorded separately, never a violation).
        States lacking a light slot (e.g. non-CARLA test fixtures) yield no events.
        """
        offset = EGO_STATE_WIDTH + self.max_tracked_agents * AGENT_SLOT_WIDTH
        states = [np.asarray(step.state, dtype=float) for step in history.history]
        states.append(np.asarray(history.history[-1].next_state, dtype=float))
        red_runs = functioning = malfunction = 0
        for prev, curr in zip(states, states[1:]):
            if len(prev) < offset + LIGHT_SLOT_WIDTH:
                continue
            next_present = curr[offset] if len(curr) >= offset + LIGHT_SLOT_WIDTH else 0.0
            moving = float(np.hypot(prev[3], prev[4])) > _LIGHT_CROSS_MOVING_SPEED
            if not (prev[offset] == 1.0 and next_present == 0.0 and moving):
                continue
            code = prev[offset + 3]
            if code in (TRAFFIC_LIGHT_OFF, TRAFFIC_LIGHT_UNKNOWN):
                malfunction += 1
            else:
                functioning += 1
                red_runs += 1 if code == TRAFFIC_LIGHT_RED else 0
        return red_runs, functioning, malfunction

    def _episode_goal_outcome(self, history: History) -> Tuple[float, float]:
        """Return ``(success, route_completion)`` for one episode.

        Reads the goal slot of the final ``next_state``: success is 1.0 when the ego
        ended within ``goal_radius`` of the episode destination, and route completion
        is the fraction of the planned route's arc length covered. States lacking a
        goal slot (non-CARLA fixtures) yield ``(0.0, 0.0)``.
        """
        goal_offset = (
            EGO_STATE_WIDTH + self.max_tracked_agents * AGENT_SLOT_WIDTH + LIGHT_SLOT_WIDTH
        )
        if not history.history:
            return 0.0, 0.0
        final = np.asarray(history.history[-1].next_state, dtype=float)
        if len(final) < goal_offset + GOAL_SLOT_WIDTH:
            return 0.0, 0.0
        goal_x, goal_y, completion = final[goal_offset : goal_offset + GOAL_SLOT_WIDTH]
        distance = float(np.hypot(final[0] - goal_x, final[1] - goal_y))
        success = 1.0 if distance <= self.goal_radius else 0.0
        return success, float(completion)

    def _episode_path_length(self, history: History) -> float:
        total = 0.0
        for step in history.history:
            start = np.asarray(step.state)[_EGO_POSITION_SLICE]
            end = np.asarray(step.next_state)[_EGO_POSITION_SLICE]
            total += float(np.linalg.norm(end - start))
        return total

    def _episode_mean_speed(self, history: History) -> float:
        speeds = [
            float(np.linalg.norm(np.asarray(step.next_state)[_EGO_VELOCITY_SLICE]))
            for step in history.history
        ]
        return float(np.mean(speeds)) if speeds else 0.0

    def _metric_from_samples(self, name: str, samples: List[float]) -> MetricValue:
        if not samples:
            return MetricValue(
                name=name, value=0.0, lower_confidence_bound=0.0, upper_confidence_bound=0.0
            )
        lower, upper = confidence_interval(data=samples, confidence=0.95)
        return MetricValue(
            name=name,
            value=float(np.mean(samples)),
            lower_confidence_bound=float(lower),
            upper_confidence_bound=float(upper),
        )

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
                return [dict(observation) for _ in range(n_samples)]

        return InitialObservation()

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        is_dict1 = isinstance(observation1, dict)
        is_dict2 = isinstance(observation2, dict)
        if is_dict1 and is_dict2:
            if observation1.keys() != observation2.keys():
                return False
            return all(np.array_equal(observation1[key], observation2[key]) for key in observation1)
        if is_dict1 or is_dict2:
            return False
        return np.array_equal(np.asarray(observation1), np.asarray(observation2))

    def hash_observation(self, observation: Any) -> Hashable:
        if isinstance(observation, dict):
            return tuple(
                (key, np.asarray(observation[key]).tobytes()) for key in sorted(observation)
            )
        array = np.asarray(observation)
        return array.tobytes()

    def hash_action(self, action: Any) -> Hashable:
        if isinstance(action, np.ndarray):
            return action.tobytes()
        return action

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        del state, action, next_states
        raise NotImplementedError(
            "CarlaPOMDP is a forward-only world environment with no transition "
            "density. Belief updates must run on the planner's model environment."
        )

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        del next_state, action, observations
        raise NotImplementedError(
            "CarlaPOMDP is a forward-only world environment with no observation "
            "density. Belief updates must run on the planner's model environment."
        )

    def cache_visualization(
        self, history: List[StepData], output_dir: Path, episode_index: int
    ) -> None:
        """Save the episode as CARLA's own chase-camera MP4 footage.

        The episode ``history`` is unused: the video is the native camera
        rendering buffered live while the world was stepped, not a plot
        reconstructed from the step data. The environment must have been
        constructed with ``record_camera=True``.

        Args:
            history: Episode step data (unused; kept for the hook signature).
            output_dir: Directory into which the ``.mp4`` video is written.
            episode_index: Zero-based episode index, used to name the file.
        """
        del history
        self.save_camera_video(output_dir / f"agent_path_{episode_index}.mp4")

    def save_camera_video(self, cache_path: Path, fps: int = 20) -> None:
        """Write CARLA's own chase-camera footage to an MP4 video.

        This is the native CARLA rendering (an RGB camera following the ego), not a
        reconstructed plot. Frames are captured live while the world is stepped, so
        the environment must have been constructed with ``record_camera=True`` and
        driven for at least one tick before calling this.

        Args:
            cache_path: File path ending in ``.mp4`` where the video is saved.
            fps: Playback frame rate. Defaults to 20.

        Raises:
            RuntimeError: If camera recording is disabled or no frames were captured.
        """
        # pylint: disable=import-outside-toplevel
        from POMDPPlanners.environments.carla_pomdp.carla_video import write_frames_to_mp4

        if not self.record_camera:
            raise RuntimeError(
                "Camera recording is disabled; construct CarlaPOMDP with "
                "record_camera=True to save native CARLA footage."
            )
        if self._session is None or not self._session.frames:
            raise RuntimeError(
                "No camera frames were captured; reset and step the world before "
                "calling save_camera_video."
            )
        write_frames_to_mp4(self._session.frames, cache_path, fps=fps)
