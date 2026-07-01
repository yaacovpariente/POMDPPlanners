# SPDX-License-Identifier: MIT

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
The **state** is the ego vehicle's ground-truth kinematics ``[x, y, yaw, vx, vy]``
read straight from the simulator, while the **observation** is a native CARLA
sensor payload (GNSS ``[lat, lon, alt]`` by default). Any measurement noise is
CARLA's own, configured through the sensor blueprint attributes in
``sensor_config`` — the wrapper adds none.

Classes:
    CarlaPOMDP: Forward-only adapter exposing a CARLA session as a world Environment.
"""

from collections.abc import Hashable
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import Environment, SpaceInfo, SpaceType
from POMDPPlanners.core.simulation import StepData

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

# Default chase RGB camera resolution / field of view (blueprint attributes).
DEFAULT_CAMERA_CONFIG: Dict[str, str] = {
    "image_size_x": "640",
    "image_size_y": "480",
    "fov": "90",
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
    ) -> None:
        import carla  # pylint: disable=import-outside-toplevel,import-error

        self._carla = carla
        self._sensor_config = sensor_config
        self._vehicle_filter = vehicle_filter
        self._record_camera = record_camera
        self._camera_config = camera_config if camera_config is not None else {}
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        self._client = client
        self._world = client.load_world(town)
        self._apply_synchronous_settings(fixed_delta_seconds)

        self._vehicle: Optional[Any] = None
        self._gnss_sensor: Optional[Any] = None
        self._collision_sensor: Optional[Any] = None
        self._camera_sensor: Optional[Any] = None
        self._camera_queue: Optional["Queue[Any]"] = None
        self._frames: List[np.ndarray] = []
        self._latest_gnss: Optional[np.ndarray] = None
        self._collided: bool = False

    @property
    def frames(self) -> List[np.ndarray]:
        """RGB chase-camera frames captured so far, one ``(H, W, 3)`` array per tick."""
        return self._frames

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Respawn the ego vehicle and sensors, tick once, and read the start."""
        if seed is not None:
            self._world.get_settings()  # touch settings so a seed hook can attach
        self._teardown_actors()
        self._frames = []
        self._spawn_actors()
        self._collided = False
        self._world.tick()
        self._capture_frame()
        return self._read_state(), self._read_observation()

    def step(
        self, throttle: float, steer: float, brake: float
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Apply a control, advance one fixed tick, and read the outcome."""
        control = self._carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self._vehicle.apply_control(control)
        self._world.tick()
        self._capture_frame()
        return self._read_state(), self._read_observation(), self._collided

    def _apply_synchronous_settings(self, fixed_delta_seconds: float) -> None:
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta_seconds
        self._world.apply_settings(settings)

    def _spawn_actors(self) -> None:
        blueprint_library = self._world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(self._vehicle_filter)[0]
        spawn_point = self._world.get_map().get_spawn_points()[0]
        self._vehicle = self._world.spawn_actor(vehicle_bp, spawn_point)
        self._gnss_sensor = self._attach_gnss(blueprint_library)
        self._collision_sensor = self._attach_collision(blueprint_library)
        if self._record_camera:
            self._camera_sensor = self._attach_camera(blueprint_library)

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

    def _on_collision(self, event: Any) -> None:
        del event
        self._collided = True

    def _read_state(self) -> np.ndarray:
        transform = self._vehicle.get_transform()
        velocity = self._vehicle.get_velocity()
        return np.array(
            [
                transform.location.x,
                transform.location.y,
                transform.rotation.yaw,
                velocity.x,
                velocity.y,
            ]
        )

    def _read_observation(self) -> np.ndarray:
        if self._latest_gnss is None:
            return np.zeros(3)
        return self._latest_gnss

    def _teardown_actors(self) -> None:
        for actor in (
            self._camera_sensor,
            self._collision_sensor,
            self._gnss_sensor,
            self._vehicle,
        ):
            if actor is not None:
                actor.destroy()
        self._vehicle = None
        self._gnss_sensor = None
        self._collision_sensor = None
        self._camera_sensor = None
        self._camera_queue = None
        self._latest_gnss = None


class CarlaPOMDP(Environment):
    """Forward-only adapter exposing a CARLA session as a world POMDP.

    The wrapper drives a CARLA server as the ground-truth world of an episode. It
    ticks the simulator exactly once per real interaction and serves the resulting
    next state, observation and reward from a small cache, because the
    POMDPPlanners episode loop requests those three quantities through separate
    method calls while CARLA produces them atomically. The state is the ego
    vehicle's ground-truth kinematics; the observation is a native CARLA sensor
    payload (GNSS by default), so the world is genuinely partially observed.

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
            # state is [x, y, yaw, vx, vy]; observation is a GNSS reading.
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
        fixed_delta_seconds: float = 0.05,
        collision_penalty: float = 100.0,
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
            fixed_delta_seconds: Synchronous-mode tick length. Defaults to 0.05.
            collision_penalty: Reward penalty applied on a terminal collision.
                Defaults to 100.0.
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
        self.fixed_delta_seconds = fixed_delta_seconds
        self.collision_penalty = collision_penalty
        self.vehicle_filter = vehicle_filter
        self.timeout = timeout
        self.seed = seed

        # Live-session state: rebuilt lazily and never serialized.
        self._session: Optional[Any] = None
        self._live_state: Optional[np.ndarray] = None
        self._latest_obs: Optional[np.ndarray] = None
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
    def _get_session(self) -> Any:
        if self._session is None:
            self._session = _CarlaSession(
                host=self.host,
                port=self.port,
                town=self.town,
                fixed_delta_seconds=self.fixed_delta_seconds,
                sensor_config=self.sensor_config,
                vehicle_filter=self.vehicle_filter,
                timeout=self.timeout,
                record_camera=self.record_camera,
                camera_config=self.camera_config,
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
        self._latest_obs = np.asarray(observation)
        self._terminated = False
        self._pending = None
        self._served_roles = set()
        return state

    def _to_control(self, action: Any) -> Tuple[float, float, float]:
        return self.action_presets[int(action)]

    def _states_equal(self, state_a: Any, state_b: Any) -> bool:
        return np.array_equal(np.asarray(state_a), np.asarray(state_b))

    def _compute_reward(self, next_state: np.ndarray, terminated: bool) -> float:
        speed = float(np.hypot(next_state[3], next_state[4]))
        penalty = self.collision_penalty if terminated else 0.0
        return speed - penalty

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
        next_state, observation, terminated = session.step(throttle, steer, brake)
        next_state = np.asarray(next_state)
        observation = np.asarray(observation)
        done = bool(terminated)
        pending = {
            "state": np.asarray(state).copy(),
            "action": action,
            "next_state": next_state,
            "observation": observation,
            "reward": self._compute_reward(next_state, done),
            "terminated": done,
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

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
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
            def sample(self, n_samples: int = 1) -> List[np.ndarray]:
                # pylint: disable=protected-access
                observation = parent._latest_obs
                if observation is None:
                    parent._reset()
                    observation = parent._latest_obs
                return [np.asarray(observation) for _ in range(n_samples)]

        return InitialObservation()

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return np.array_equal(np.asarray(observation1), np.asarray(observation2))

    def hash_observation(self, observation: Any) -> Hashable:
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
