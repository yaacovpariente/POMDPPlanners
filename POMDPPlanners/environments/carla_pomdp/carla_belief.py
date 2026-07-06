# SPDX-License-Identifier: MIT

"""Particle belief that re-injects freshly observed traffic into the CARLA state.

The CARLA planner-side models carry other vehicles as fixed ego-frame agent slots in the
state. A plain particle filter can only *reweight and propagate* the agent slots its
particles were seeded with — it can never acquire an agent that appears mid-episode, and a
slot seeded empty stays empty forever. Over a drive the ego passes cars and new ones come
into range, so the belief's picture of nearby traffic goes stale exactly where the planner
needs it (the vehicle it is about to hit).

:class:`CarlaAgentReinvigoration` closes that gap. It is a
:class:`~POMDPPlanners.core.belief.particle_beliefs.WeightedParticleBeliefReinvigoration`
whose reinvigoration step reconciles each particle's *propagated* agent block (motion model
already applied by the transition) with the *current observation's* detections by proximity
track association:

* **Matched** — an observed detection near a propagated track (within ``association_radius``)
  corrects that track to the measured pose plus per-particle Gaussian jitter.
* **New** — an observed detection with no nearby propagated track is added as a fresh track.
* **Coasted** — a propagated track with no matching detection is *kept* (dead-reckoned by the
  motion model) as long as it is still within ``persistence_range``, so a briefly occluded
  or out-of-range agent survives instead of blinking out; beyond that range it is evicted to
  avoid accumulating ghosts that can never be re-confirmed.

Association runs per particle (propagated poses differ across particles) and is stateless
across steps — it needs no per-track age counter, so it is unaffected by the resampling that
reorders particles inside the filter update. The ego block is left to the ordinary
particle-filter update, which estimates it well. The returned belief is itself a
:class:`CarlaAgentReinvigoration`, so the reconciliation repeats on every step of the episode.

Note:
    The CARLA schema carries no persistent agent identity — the observation's slot ``k`` is
    the ``k``-th nearest agent *at read time*, re-sorted each read. Association is therefore
    geometric (nearest propagated track within a gate), not identity-based, which is why the
    ``association_radius`` gate and the ``persistence_range`` eviction exist.

Classes:
    CarlaAgentReinvigoration: Particle belief that re-injects observed CARLA traffic.
"""

from typing import Any, List

import numpy as np

from POMDPPlanners.core.belief.particle_beliefs import (
    WeightedParticleBelief,
    WeightedParticleBeliefReinvigoration,
)
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.environments.carla_pomdp.carla_perception import (
    DEFAULT_CORRIDOR_HALFWIDTH,
    camera_looming_cue,
    fuse_forward_obstacle,
    lidar_forward_clearance,
    traffic_light_stop_distance,
)
from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
    AGENT_SLOT_WIDTH,
    DEFAULT_MAX_TRACKED_AGENTS,
    DEFAULT_PERCEPTION_RANGE,
    EGO_STATE_WIDTH,
)


class CarlaAgentReinvigoration(WeightedParticleBeliefReinvigoration):
    """Weighted particle belief that reconciles agent slots with each observation.

    After the standard particle-filter weight update and resample, the reinvigoration step
    associates each particle's propagated agent tracks with the current observation's
    detections by proximity: matched tracks snap to the measured pose (plus jitter), new
    detections are added, and unmatched-but-in-range tracks coast on their motion model. The
    ego block is untouched. This lets a planner search against the traffic actually in view
    while still carrying briefly occluded vehicles through the dropout.

    Attributes:
        max_tracked_agents: Number of fixed agent slots carried in each particle.
        agent_pose_jitter: Std of Gaussian noise added to each measured agent's
            ``[rel_x, rel_y, rel_yaw, rel_speed]`` pose, for particle diversity.
        association_radius: Max ego-frame distance (m) at which an observed detection is
            associated with a propagated track rather than treated as a new agent.
        persistence_range: Ego-frame range (m) within which an unmatched (undetected) track
            is coasted on its motion model; beyond it the track is evicted.
        sensor_fusion: Whether lidar/camera obstacles are injected as extra slots.
        stop_for_traffic_lights: Whether a red/yellow light is injected as a virtual
            stop-obstacle at its stop line so the ego brakes for it.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
        ...     AGENT_SLOT_WIDTH, EGO_STATE_WIDTH)
        >>> width = EGO_STATE_WIDTH + 1 * AGENT_SLOT_WIDTH
        >>> particles = [np.zeros(width) for _ in range(4)]
        >>> belief = CarlaAgentReinvigoration(
        ...     particles=particles,
        ...     log_weights=np.log(np.ones(4) / 4),
        ...     max_tracked_agents=1,
        ... )
        >>> observation = {
        ...     "gnss": np.zeros(2),
        ...     "agents": np.array([1.0, 8.0, 0.0, 0.0, 5.0]),  # one agent 8 m ahead
        ... }
        >>> base = WeightedParticleBelief(particles=particles, log_weights=belief.log_weights)
        >>> refreshed = belief.reinvigorate("noop", observation, None, base)
        >>> bool(np.asarray(refreshed.particles)[0, EGO_STATE_WIDTH] == 1.0)  # slot now present
        True
    """

    def __init__(
        self,
        particles: Any,
        log_weights: np.ndarray,
        max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
        agent_pose_jitter: float = 0.3,
        association_radius: float = 5.0,
        persistence_range: float = DEFAULT_PERCEPTION_RANGE,
        sensor_fusion: bool = False,
        lidar_corridor_halfwidth: float = DEFAULT_CORRIDOR_HALFWIDTH,
        obstacle_detection_range: float = 30.0,
        stop_for_traffic_lights: bool = False,
        resampling: bool = True,
        ess_factor: float = 0.5,
    ):
        """Initialize the traffic-reconciling CARLA belief.

        Args:
            particles: State particles ``[ego(7) | agent slots(K*5)]``.
            log_weights: Log-weights for the particles.
            max_tracked_agents: Number of fixed agent slots carried per particle.
            agent_pose_jitter: Std of Gaussian noise on each measured agent pose.
            association_radius: Max ego-frame distance (m) to associate a detection with a
                propagated track instead of adding a new one.
            persistence_range: Ego-frame range (m) within which an undetected track coasts;
                beyond it the track is evicted.
            sensor_fusion: When True, fuse the observation's ``lidar``/``camera`` into an
                extra forward-obstacle slot so the planner brakes for hazards the tracked
                vehicle slots miss (cut-ins, non-vehicle geometry).
            lidar_corridor_halfwidth: Half-width (m) of the forward corridor scanned for a
                lidar obstacle when ``sensor_fusion`` is enabled.
            obstacle_detection_range: Only fuse a sensed obstacle nearer than this (m); a
                farther clearance is treated as no hazard.
            stop_for_traffic_lights: When True, inject a red/yellow light (from the
                ``traffic_light`` observation) as a virtual stop-obstacle at its stop line,
                so the same obstacle-aware braking makes the ego halt for it.
            resampling: Enable automatic resampling when ESS drops. Defaults to True.
            ess_factor: Effective-sample-size threshold factor. Defaults to 0.5.
        """
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            resampling=resampling,
            ess_factor=ess_factor,
        )
        self.max_tracked_agents = max_tracked_agents
        self.agent_pose_jitter = agent_pose_jitter
        self.association_radius = association_radius
        self.persistence_range = persistence_range
        self.sensor_fusion = sensor_fusion
        self.lidar_corridor_halfwidth = lidar_corridor_halfwidth
        self.obstacle_detection_range = obstacle_detection_range
        self.stop_for_traffic_lights = stop_for_traffic_lights

    def reinvigorate(  # type: ignore[override]
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        belief: "WeightedParticleBelief",
    ) -> "CarlaAgentReinvigoration":
        """Reconcile each particle's agent tracks with the observed traffic and sensors."""
        del action, pomdp
        particles = np.array(belief.particles, dtype=float, copy=True)
        observed_rows = np.asarray(observation["agents"], dtype=float).reshape(
            self.max_tracked_agents, AGENT_SLOT_WIDTH
        )
        obstacle_x = None
        if self.sensor_fusion or self.stop_for_traffic_lights:
            obstacle_x = self._sensed_obstacle_distance(observation)
        for index in range(len(particles)):
            propagated_rows = particles[index, EGO_STATE_WIDTH:].reshape(
                self.max_tracked_agents, AGENT_SLOT_WIDTH
            )
            reconciled = self._associate_tracks(propagated_rows, observed_rows)
            if obstacle_x is not None:
                self._inject_sensed_obstacle(reconciled, obstacle_x)
            particles[index, EGO_STATE_WIDTH:] = reconciled.reshape(-1)
        return CarlaAgentReinvigoration(
            particles=particles,
            log_weights=np.array(belief.log_weights, dtype=float, copy=True),
            max_tracked_agents=self.max_tracked_agents,
            agent_pose_jitter=self.agent_pose_jitter,
            association_radius=self.association_radius,
            persistence_range=self.persistence_range,
            sensor_fusion=self.sensor_fusion,
            lidar_corridor_halfwidth=self.lidar_corridor_halfwidth,
            obstacle_detection_range=self.obstacle_detection_range,
            stop_for_traffic_lights=self.stop_for_traffic_lights,
            resampling=belief.resampling,
            ess_factor=belief.ess_factor,
        )

    def _sensed_obstacle_distance(self, observation: Any):
        """Nearest forward obstacle from lidar/camera and a red light, or None if clear."""
        if not isinstance(observation, dict):
            return None
        distance = float("inf")
        if self.sensor_fusion:
            clearance = lidar_forward_clearance(
                observation.get("lidar"), self.lidar_corridor_halfwidth
            )
            distance = min(
                distance,
                fuse_forward_obstacle(clearance, camera_looming_cue(observation.get("camera"))),
            )
        if self.stop_for_traffic_lights:
            distance = min(distance, traffic_light_stop_distance(observation.get("traffic_light")))
        return distance if distance < self.obstacle_detection_range else None

    def _inject_sensed_obstacle(self, rows: np.ndarray, obstacle_x: float) -> None:
        """Add a forward-obstacle slot unless a tracked agent already covers that hazard."""
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
                self._track_range(rows[slot]) if rows[slot, 0] == 1.0 else np.inf
                for slot in range(self.max_tracked_agents)
            ]
        )
        farthest = int(np.argmax(ranges))
        if obstacle_x < ranges[farthest]:
            rows[farthest] = phantom  # displace a farther track for the nearer hazard

    def _associate_tracks(
        self, propagated_rows: np.ndarray, observed_rows: np.ndarray
    ) -> np.ndarray:
        """Reconcile one particle's propagated tracks with the observed detections."""
        tracks: List[np.ndarray] = []
        matched_slots: set = set()
        for obs_slot in range(self.max_tracked_agents):
            if observed_rows[obs_slot, 0] != 1.0:
                continue
            match = self._nearest_unmatched_track(
                observed_rows[obs_slot], propagated_rows, matched_slots
            )
            if match is not None:
                matched_slots.add(match)
            tracks.append(self._measured_row(observed_rows[obs_slot]))
        for prop_slot in range(self.max_tracked_agents):
            if propagated_rows[prop_slot, 0] != 1.0 or prop_slot in matched_slots:
                continue
            if self._track_range(propagated_rows[prop_slot]) <= self.persistence_range:
                tracks.append(propagated_rows[prop_slot].copy())  # coast the occluded track
        return self._pack_nearest_tracks(tracks)

    def _nearest_unmatched_track(
        self, detection: np.ndarray, propagated_rows: np.ndarray, matched_slots: set
    ):
        best_slot = None
        best_dist = self.association_radius
        for slot in range(self.max_tracked_agents):
            if propagated_rows[slot, 0] != 1.0 or slot in matched_slots:
                continue
            dist = float(
                np.hypot(
                    detection[1] - propagated_rows[slot, 1],
                    detection[2] - propagated_rows[slot, 2],
                )
            )
            if dist <= best_dist:
                best_dist = dist
                best_slot = slot
        return best_slot

    def _measured_row(self, detection: np.ndarray) -> np.ndarray:
        row = np.zeros(AGENT_SLOT_WIDTH)
        row[0] = 1.0
        jitter = np.random.normal(0.0, self.agent_pose_jitter, size=AGENT_SLOT_WIDTH - 1)
        row[1:] = detection[1:] + jitter
        return row

    def _pack_nearest_tracks(self, tracks: List[np.ndarray]) -> np.ndarray:
        packed = np.zeros((self.max_tracked_agents, AGENT_SLOT_WIDTH))
        if not tracks:
            return packed
        tracks_arr = np.stack(tracks)
        if len(tracks_arr) > self.max_tracked_agents:
            ranges = np.array([self._track_range(track) for track in tracks_arr])
            keep = np.argsort(ranges)[: self.max_tracked_agents]
            tracks_arr = tracks_arr[keep]
        packed[: len(tracks_arr)] = tracks_arr
        return packed

    @staticmethod
    def _track_range(row: np.ndarray) -> float:
        return float(np.hypot(row[1], row[2]))
