# SPDX-License-Identifier: MIT

"""Standalone, swappable perception + prediction pipeline for the CARLA planner.

In the intended architecture the belief is a plain particle filter over the ego and *does not*
run perception itself. This module holds that perception, decoupled from both the world and the
belief, so a user can swap in a different perception model without touching either. It turns the
world's raw multi-modal observation (``lidar``/``camera``/``traffic_light``/``agents``) into the
two things a planner-side belief needs each step:

* the ego-frame **agent block** — the ``max_tracked_agents`` nearest vehicles as
  ``[present, rel_x, rel_y, rel_yaw, rel_speed]`` slots — perceived and tracked (with velocity)
  from the sensors, and
* a single fused **forward-obstacle distance** (lidar corridor + camera looming cue, optionally
  a red/amber traffic light), or ``None`` when the way ahead is clear.

Two swappable interfaces compose the pipeline:

* :class:`PerceptionModel` — the single-frame stage (raw sensors -> :class:`Detections`).
* :class:`MotionTracker` — the temporal stage that estimates agent velocity over time.

The defaults (:class:`LidarCameraPerceptionModel` + :class:`AlphaBetaTracker`) reconstruct a real
autonomous-driving perception stack: vehicles are clustered from the lidar cloud, their velocity
is estimated by a constant-velocity tracker whose coasting carries a briefly occluded vehicle
through a dropout, and the traffic light is *inferred from the camera image* rather than read from
a ground-truth channel. :class:`OracleAgentPerceptionModel` is provided for studies/tests that want
exact agent positions instead of inferred ones.

The pipeline is immutable and owns the tracker state: :meth:`CarlaPerceptionPipeline.process`
returns a :class:`PerceptionOutput` carrying a *successor* pipeline with the advanced tracks, so a
belief can thread perception forward without holding any perception state of its own.

Classes:
    PerceptionModel: Abstract single-frame perception interface.
    MotionTracker: Abstract temporal (velocity-estimating) tracking interface.
    LidarCameraPerceptionModel: Default lidar+camera perception with camera traffic lights.
    OracleAgentPerceptionModel: Ground-truth-agent perception for studies/tests.
    AlphaBetaTracker: Default constant-velocity multi-object tracker.
    CarlaPerceptionPipeline: Composed, immutable perception + prediction stage.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from POMDPPlanners.environments.carla_pomdp.carla_perception.carla_sensors import (
    DEFAULT_CORRIDOR_HALFWIDTH,
    camera_looming_cue,
    fuse_forward_obstacle,
    lidar_forward_clearance,
    lidar_vehicle_detections,
    traffic_light_from_camera,
    traffic_light_stop_distance,
)
from POMDPPlanners.environments.carla_pomdp.carla_perception.carla_tracking import (
    TRACK_WIDTH,
    update_tracks,
)
from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
    AGENT_SLOT_WIDTH,
    DEFAULT_MAX_TRACKED_AGENTS,
)

DEFAULT_OBSTACLE_DETECTION_RANGE = 30.0  # only obstacles nearer than this (m) are reported
DEFAULT_DT = 0.05  # tracker time step (s) for constant-velocity prediction


@dataclass(frozen=True)
class Detections:
    """Single-frame perception output.

    Attributes:
        vehicle_positions: ``(M, 3)`` ego-frame vehicle detections ``[rel_x, rel_y, confidence]``.
        forward_clearance: Fused lidar+camera forward-obstacle distance (m); large when clear.
        traffic_light: ``[should_stop, distance_m]`` stop signal for a red/amber light.
    """

    vehicle_positions: np.ndarray
    forward_clearance: float
    traffic_light: np.ndarray


@dataclass(frozen=True)
class PerceptionOutput:
    """Per-step pipeline output consumed by the belief.

    Attributes:
        agent_rows: ``(K, AGENT_SLOT_WIDTH)`` ego-frame agent slots
            ``[present, rel_x, rel_y, rel_yaw, rel_speed]``.
        obstacle_distance: Nearest fused forward-obstacle distance (m), or ``None`` when clear.
        pipeline: Successor pipeline carrying the advanced tracker state.
    """

    agent_rows: np.ndarray
    obstacle_distance: Optional[float]
    pipeline: "CarlaPerceptionPipeline"


class PerceptionModel(ABC):
    """Abstract single-frame perception interface: raw sensor dict -> :class:`Detections`.

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    @abstractmethod
    def detect(self, observation: Mapping[str, Any]) -> Detections:
        """Perceive vehicles, a forward obstacle, and a traffic light from one observation.

        Args:
            observation: The world's raw observation dict (``lidar``/``camera``/``agents``/
                ``traffic_light`` keys, any subset present).

        Returns:
            The single-frame :class:`Detections` for this observation.
        """


class MotionTracker(ABC):
    """Abstract temporal prediction interface: prior tracks + detections -> tracks with velocity.

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    @abstractmethod
    def update(
        self, tracks: Optional[np.ndarray], vehicle_positions: np.ndarray, dt: float
    ) -> np.ndarray:
        """Advance the tracker one step and return the ``(N, 5)`` track set with velocity.

        Args:
            tracks: Prior ``(N, 5)`` tracks ``[rel_x, rel_y, vx, vy, confidence]``, or ``None``.
            vehicle_positions: ``(M, 3)`` detections ``[rel_x, rel_y, confidence]``.
            dt: Time step (s) since the previous update.

        Returns:
            The updated ``(N, 5)`` track set.
        """


class LidarCameraPerceptionModel(PerceptionModel):
    """Default CARLA perception: lidar vehicle clustering + camera obstacle/traffic-light cues.

    Vehicles are clustered from the lidar cloud, the forward obstacle fuses the lidar corridor
    clearance with the camera looming cue, and the traffic light is inferred from the camera
    image (no ground-truth channel is consulted) unless ``traffic_light_source='channel'``.

    Attributes:
        lidar_corridor_halfwidth: Half-width (m) of the forward corridor scanned for an obstacle.
        traffic_light_source: ``'camera'`` infers the light from the RGB image; ``'channel'``
            reads it from the observation's ``traffic_light`` key.

    Example:
        >>> import numpy as np
        >>> model = LidarCameraPerceptionModel()
        >>> obs = {"lidar": np.zeros((0, 4)), "camera": np.zeros((8, 8, 3), dtype=np.uint8)}
        >>> model.detect(obs).vehicle_positions.shape
        (0, 3)
    """

    def __init__(
        self,
        lidar_corridor_halfwidth: float = DEFAULT_CORRIDOR_HALFWIDTH,
        traffic_light_source: str = "camera",
    ) -> None:
        """Initialize the lidar+camera perception model.

        Args:
            lidar_corridor_halfwidth: Half-width (m) of the forward obstacle corridor.
            traffic_light_source: ``'camera'`` to infer the light from the image, or
                ``'channel'`` to read the observation's ``traffic_light`` key.
        """
        self.lidar_corridor_halfwidth = lidar_corridor_halfwidth
        self.traffic_light_source = traffic_light_source

    def detect(self, observation: Mapping[str, Any]) -> Detections:
        obs = observation if isinstance(observation, Mapping) else {}
        lidar = obs.get("lidar")
        clearance = fuse_forward_obstacle(
            lidar_forward_clearance(lidar, self.lidar_corridor_halfwidth),
            camera_looming_cue(obs.get("camera")),
        )
        return Detections(
            lidar_vehicle_detections(lidar), clearance, self._perceive_traffic_light(obs)
        )

    def _perceive_traffic_light(self, obs: Mapping[str, Any]) -> np.ndarray:
        if self.traffic_light_source == "channel":
            signal = obs.get("traffic_light")
            return np.asarray(signal, dtype=float) if signal is not None else np.array([0.0, 0.0])
        return traffic_light_from_camera(obs.get("camera"))


class OracleAgentPerceptionModel(PerceptionModel):
    """Ground-truth-agent perception for studies/tests: vehicles from the ``agents`` channel.

    Reads the observation's ground-truth ``agents`` rows as vehicle detections (position with
    confidence ``1.0``), still fusing lidar/camera for the forward obstacle and reading the light
    from the ``traffic_light`` channel. Use when a study wants exact agent positions rather than
    inferred ones; the tracker then re-estimates velocity from the position stream.

    Attributes:
        max_tracked_agents: Number of fixed agent slots in the ground-truth ``agents`` channel.
        lidar_corridor_halfwidth: Half-width (m) of the forward obstacle corridor.

    Example:
        >>> import numpy as np
        >>> model = OracleAgentPerceptionModel(max_tracked_agents=1)
        >>> obs = {"agents": np.array([1.0, 8.0, 0.0, 0.0, 5.0])}
        >>> model.detect(obs).vehicle_positions.shape
        (1, 3)
    """

    def __init__(
        self,
        max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
        lidar_corridor_halfwidth: float = DEFAULT_CORRIDOR_HALFWIDTH,
    ) -> None:
        """Initialize the oracle-agent perception model.

        Args:
            max_tracked_agents: Number of fixed agent slots in the ``agents`` channel.
            lidar_corridor_halfwidth: Half-width (m) of the forward obstacle corridor.
        """
        self.max_tracked_agents = max_tracked_agents
        self.lidar_corridor_halfwidth = lidar_corridor_halfwidth

    def detect(self, observation: Mapping[str, Any]) -> Detections:
        obs = observation if isinstance(observation, Mapping) else {}
        clearance = fuse_forward_obstacle(
            lidar_forward_clearance(obs.get("lidar"), self.lidar_corridor_halfwidth),
            camera_looming_cue(obs.get("camera")),
        )
        signal = obs.get("traffic_light")
        traffic_light = (
            np.asarray(signal, dtype=float) if signal is not None else np.array([0.0, 0.0])
        )
        return Detections(self._oracle_positions(obs.get("agents")), clearance, traffic_light)

    def _oracle_positions(self, agents: Any) -> np.ndarray:
        if agents is None:
            return np.zeros((0, 3))
        rows = np.asarray(agents, dtype=float).reshape(self.max_tracked_agents, AGENT_SLOT_WIDTH)
        present = rows[rows[:, 0] == 1.0]
        if present.size == 0:
            return np.zeros((0, 3))
        return np.column_stack([present[:, 1], present[:, 2], np.ones(len(present))])


class AlphaBetaTracker(MotionTracker):
    """Default constant-velocity multi-object tracker (alpha-beta) over vehicle detections.

    A thin :class:`MotionTracker` wrapper around
    :func:`~POMDPPlanners.environments.carla_pomdp.carla_perception.carla_tracking.update_tracks`; its coasting of
    undetected tracks is what carries a briefly occluded vehicle through a sensor dropout.

    Example:
        >>> import numpy as np
        >>> tracker = AlphaBetaTracker()
        >>> tracks = tracker.update(None, np.array([[8.0, 0.0, 1.0]]), dt=0.05)
        >>> tracks.shape
        (1, 5)
    """

    def update(
        self, tracks: Optional[np.ndarray], vehicle_positions: np.ndarray, dt: float
    ) -> np.ndarray:
        return update_tracks(tracks, vehicle_positions, dt)


class CarlaPerceptionPipeline:
    """Standalone perception + prediction stage: raw observation -> agent slots + obstacle.

    Composes a :class:`PerceptionModel` (single-frame) and a :class:`MotionTracker` (temporal),
    owns the tracker state, and produces the ego-frame agent block a belief stamps onto its
    particles plus a fused forward-obstacle distance. Immutable: :meth:`process` returns a
    :class:`PerceptionOutput` carrying a successor pipeline with the advanced tracks.

    Attributes:
        max_tracked_agents: Number of agent slots produced in the agent block.
        perception: The single-frame :class:`PerceptionModel`.
        tracker: The temporal :class:`MotionTracker`.
        sensor_fusion: Whether the fused lidar/camera forward obstacle is reported.
        stop_for_traffic_lights: Whether a red/amber light is reported as a forward obstacle.
        obstacle_detection_range: Only obstacles nearer than this (m) are reported.
        dt: Tracker time step (s).

    Example:
        >>> import numpy as np
        >>> pipeline = CarlaPerceptionPipeline(max_tracked_agents=1)
        >>> obs = {"lidar": np.zeros((0, 4)), "camera": np.zeros((8, 8, 3), dtype=np.uint8)}
        >>> output = pipeline.process(obs)
        >>> output.agent_rows.shape
        (1, 5)
    """

    def __init__(
        self,
        max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
        perception: Optional[PerceptionModel] = None,
        tracker: Optional[MotionTracker] = None,
        sensor_fusion: bool = True,
        stop_for_traffic_lights: bool = True,
        obstacle_detection_range: float = DEFAULT_OBSTACLE_DETECTION_RANGE,
        dt: float = DEFAULT_DT,
        lidar_corridor_halfwidth: float = DEFAULT_CORRIDOR_HALFWIDTH,
        tracks: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the perception + prediction pipeline.

        Args:
            max_tracked_agents: Number of agent slots produced in the agent block.
            perception: The single-frame perception model. Defaults to
                :class:`LidarCameraPerceptionModel`.
            tracker: The temporal tracker. Defaults to :class:`AlphaBetaTracker`.
            sensor_fusion: When True, report the fused lidar/camera forward obstacle.
            stop_for_traffic_lights: When True, report a red/amber light as a forward obstacle.
            obstacle_detection_range: Only obstacles nearer than this (m) are reported.
            dt: Tracker time step (s) for the constant-velocity prediction.
            lidar_corridor_halfwidth: Half-width (m) of the default model's obstacle corridor.
            tracks: Carried ``(N, 5)`` tracker state (``None`` starts empty).
        """
        self.max_tracked_agents = max_tracked_agents
        self.perception = (
            perception
            if perception is not None
            else LidarCameraPerceptionModel(lidar_corridor_halfwidth)
        )
        self.tracker = tracker if tracker is not None else AlphaBetaTracker()
        self.sensor_fusion = sensor_fusion
        self.stop_for_traffic_lights = stop_for_traffic_lights
        self.obstacle_detection_range = obstacle_detection_range
        self.dt = dt
        self.lidar_corridor_halfwidth = lidar_corridor_halfwidth
        self.tracks = (
            np.zeros((0, TRACK_WIDTH)) if tracks is None else np.asarray(tracks, dtype=float)
        )

    def process(self, observation: Mapping[str, Any]) -> PerceptionOutput:
        """Perceive and track one observation into an agent block and forward obstacle.

        Args:
            observation: The world's raw observation dict.

        Returns:
            A :class:`PerceptionOutput` with the agent block, the fused obstacle distance (or
            ``None``), and the successor pipeline carrying the advanced tracks.
        """
        detections = self.perception.detect(observation)
        tracks = self.tracker.update(self.tracks, detections.vehicle_positions, self.dt)
        agent_rows = self._pack_agent_rows(tracks)
        obstacle = self._fused_obstacle_distance(detections)
        if obstacle is not None:
            self._inject_obstacle(agent_rows, obstacle)
        return PerceptionOutput(
            agent_rows=agent_rows,
            obstacle_distance=obstacle,
            pipeline=self._successor(tracks),
        )

    def _pack_agent_rows(self, tracks: np.ndarray) -> np.ndarray:
        rows = np.zeros((self.max_tracked_agents, AGENT_SLOT_WIDTH))
        if tracks.size == 0:
            return rows
        nearest = tracks[np.argsort(np.hypot(tracks[:, 0], tracks[:, 1]))]
        for slot, track in enumerate(nearest[: self.max_tracked_agents]):
            rel_x, rel_y, vel_x, vel_y = track[0], track[1], track[2], track[3]
            rows[slot] = [
                1.0,
                rel_x,
                rel_y,
                float(np.arctan2(vel_y, vel_x)),
                float(np.hypot(vel_x, vel_y)),
            ]
        return rows

    def _fused_obstacle_distance(self, detections: Detections) -> Optional[float]:
        distance = float("inf")
        if self.sensor_fusion:
            distance = min(distance, detections.forward_clearance)
        if self.stop_for_traffic_lights:
            distance = min(distance, traffic_light_stop_distance(detections.traffic_light))
        return distance if distance < self.obstacle_detection_range else None

    def _inject_obstacle(self, rows: np.ndarray, obstacle_x: float) -> None:
        """Fold the fused forward obstacle into the agent block as a stationary hazard slot.

        Skips the injection when a tracked vehicle already covers the hazard; otherwise fills an
        empty slot, or displaces the farthest tracked vehicle when the hazard is nearer.
        """
        halfwidth = self.lidar_corridor_halfwidth
        for slot in range(self.max_tracked_agents):
            if (
                rows[slot, 0] == 1.0
                and rows[slot, 1] > 0.0
                and abs(rows[slot, 2]) < halfwidth
                and rows[slot, 1] <= obstacle_x + 2.0
            ):
                return  # a tracked vehicle already represents this obstacle
        phantom = np.array([1.0, obstacle_x, 0.0, 0.0, 0.0])  # stationary hazard, straight ahead
        for slot in range(self.max_tracked_agents):
            if rows[slot, 0] != 1.0:
                rows[slot] = phantom
                return
        ranges = np.array(
            [
                self._row_range(rows[slot]) if rows[slot, 0] == 1.0 else np.inf
                for slot in range(self.max_tracked_agents)
            ]
        )
        farthest = int(np.argmax(ranges))
        if obstacle_x < ranges[farthest]:
            rows[farthest] = phantom  # displace a farther track for the nearer hazard

    @staticmethod
    def _row_range(row: np.ndarray) -> float:
        return float(np.hypot(row[1], row[2]))

    def _successor(self, tracks: np.ndarray) -> "CarlaPerceptionPipeline":
        return CarlaPerceptionPipeline(
            max_tracked_agents=self.max_tracked_agents,
            perception=self.perception,
            tracker=self.tracker,
            sensor_fusion=self.sensor_fusion,
            stop_for_traffic_lights=self.stop_for_traffic_lights,
            obstacle_detection_range=self.obstacle_detection_range,
            dt=self.dt,
            lidar_corridor_halfwidth=self.lidar_corridor_halfwidth,
            tracks=tracks,
        )
