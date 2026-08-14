# SPDX-License-Identifier: MIT

"""What each arm of the matched pair measures, and what a reading is worth in a weight.

Two observation models, one per arm, kept out of
:class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp.RacetrackModelPOMDP`
because they answer a different question from the rest of it. That class is about how the
car moves; these are about what the car can see, and the two change for unrelated reasons.

:class:`SensorObservationModel` is the POMDP arm and the reason this package exists in its
current shape. Its rule is that **the reading is the whole state except the vehicles the
sensor cannot see**. So the ego's pose, speed and lane-relative pose are all in it, along
with the road's curvature ahead, and the other vehicles are in it in full — position *and*
both components of relative velocity — whenever they are inside the range gate and not
behind a closer one. What is withheld is a vehicle that fails either test, which produces no
row at all. The likelihood is a product of closed-form terms, Gaussians and a Bernoulli,
rather than a rasterisation.

:class:`KinematicsObservationModel` is the MDP baseline: absolute position and velocity for
the ego and the nearest few vehicles, with only the other drivers' policy withheld.

Both charge a particle for *whether* a vehicle is there and not only where, using the
composition in :mod:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_detection`. At the
shipped rates of zero that charge is total: a particle whose visibility prediction contradicts
the reading is ruled out, because the world's detection decision is deterministic and there is
no miss rate to explain the disagreement away with. The cost is floored at 27.6 nats rather
than ``-inf``, which is arithmetic hygiene and not a lossy sensor.

Classes:
    CurvatureAhead: Protocol for whatever predicts the road ahead of a particle.
    SensorObservationModel: The POMDP arm — ego pose, speed, lane camera, curvature, radar.
    KinematicsObservationModel: The MDP arm — absolute kinematics for K + 1 vehicles.
"""

from typing import Any, Dict, Protocol

import numpy as np

from POMDPPlanners.environments.racetrack_pomdp.racetrack_detection import (
    bernoulli_log_prob,
    cauchy_draw,
    cauchy_log_prob,
    detected_probabilities,
    gaussian_log_prob,
    pack_detections,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_PRESENT,
    AGENT_REL_VX,
    AGENT_REL_X,
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
    OBSERVED_EGO_POSE_WIDTH,
    detection_visibility,
    rotate,
    state_agent_rows,
    wrap_to_pi,
)

EGO_POSE_KEY = "ego_pose"
EGO_SPEED_KEY = "ego_speed"
LANE_POSE_KEY = "lane_pose"
CURVATURE_AHEAD_KEY = "curvature_ahead"
DETECTIONS_KEY = "detections"
EGO_KEY = "ego"
AGENTS_KEY = "agents"

SENSOR_KEYS = (
    EGO_POSE_KEY,
    EGO_SPEED_KEY,
    LANE_POSE_KEY,
    CURVATURE_AHEAD_KEY,
    DETECTIONS_KEY,
)

# Ego pose, speedometer, lane camera, curvature ahead, detections: one POMDP reading.
_POMDP_OBSERVATION_PARTS = len(SENSOR_KEYS)

# The state slots the ego-pose channel reports directly, in the channel's own order. Heading
# is among them but is scored separately, because it wraps and a plain difference does not.
_EGO_POSE_SLOTS = (EGO_X, EGO_Y, EGO_HEADING, EGO_ARCLENGTH_M)

# One detection report before the presence flag is prepended: position and full velocity.
_DETECTION_REPORT_WIDTH = DETECTION_SLOT_WIDTH - 1

# Floor on every likelihood width. A zero-width Gaussian is a delta, and the first particle
# whose dead-reckoned speed misses the reading by a hair would be annihilated. It also guards
# the zero-noise world a test configures: matching that world's sigma exactly would make the
# likelihood a delta on a reading that is now exact.
_STD_EPS = 1e-9

# The MDP arm's ego row is [x, y, vx, vy], read out of columns 1:5 of the raw table.
_EGO_OBS_WIDTH = 4


class CurvatureAhead(Protocol):
    """Whatever can say where the road bends in front of a particle.

    Implemented by the planner-side model, which is the only thing that knows — from a track
    map, or from the curvature channel it just read. Held as a protocol so the sensor model
    does not import the model that owns it.
    """

    def curvature_ahead(self, ego: np.ndarray) -> np.ndarray:
        """Signed curvature in 1/m at each lookahead distance, ``[B, L]`` for ``[B, W]`` ego."""
        ...  # pylint: disable=unnecessary-ellipsis


class ObservationArm(Protocol):
    """One arm of the matched pair: how a reading is encoded, drawn and scored.

    The planner-side model holds exactly one of these and delegates to it, so switching arms
    changes an object rather than branching in three methods.
    """

    def encode(self, observation: Any) -> Dict[str, np.ndarray]:
        """Turn the world's raw reading into the keys the planner works in."""
        ...  # pylint: disable=unnecessary-ellipsis

    def draw(self, state: Any) -> Dict[str, np.ndarray]:
        """Sample one reading of ``state``."""
        ...  # pylint: disable=unnecessary-ellipsis

    def log_prob(self, state: Any, observation: Any) -> float:
        """Score one encoded reading against one state."""
        ...  # pylint: disable=unnecessary-ellipsis


class SensorObservationModel:
    """The POMDP arm: the state, minus what the sensor cannot see, with a closed-form density.

    Attributes:
        max_detections: Number of detection slots a reading carries.
        curvature_lookahead_count: Number of samples the curvature channel carries. Only the
            count is held here — the distances belong to the road model that predicts them,
            and keeping a second copy is how the two come to disagree.
    """

    # The widths are one per sensor channel and they travel together, so grouping them into
    # a config object would only add a layer to read through.
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        road: CurvatureAhead,
        max_detections: int,
        *,
        ego_position_std_m: float,
        ego_heading_std_rad: float,
        ego_arclength_std_m: float,
        ego_speed_std: float,
        lane_lateral_std_m: float,
        lane_heading_std_rad: float,
        curvature_std_1pm: float,
        curvature_lookahead_count: int,
        max_detection_range_m: float,
        detection_position_std_m: float,
        detection_velocity_std: float,
        blocker_half_width_m: float,
        presence_miss_prob: float,
        presence_false_alarm_prob: float,
        clutter_position_scale_m: float,
        clutter_velocity_scale: float,
    ) -> None:
        """Build the POMDP arm's observation model.

        Args:
            road: The object that predicts curvature ahead of a particle.
            max_detections: Number of detection slots a reading carries.
            ego_position_std_m: Width (m) of the ego-pose channel's ``x`` and ``y``.
            ego_heading_std_rad: Width (rad) of the ego-pose channel's heading.
            ego_arclength_std_m: Width (m) of the ego-pose channel's arclength.
            ego_speed_std: Width (m/s) of the speedometer likelihood.
            lane_lateral_std_m: Width (m) of the lane camera's lateral likelihood.
            lane_heading_std_rad: Width (rad) of the lane camera's heading likelihood.
            curvature_std_1pm: Width (1/m) of the curvature channel's likelihood.
            curvature_lookahead_count: Number of curvature samples a reading carries.
            max_detection_range_m: Range (m) beyond which a vehicle is not reported.
            detection_position_std_m: Width (m) of a detection's position likelihood.
            detection_velocity_std: Width (m/s) of a detection's relative-velocity
                likelihood, applied to both components alike.
            blocker_half_width_m: Half-width (m) of a vehicle when predicting occlusion.
            presence_miss_prob: Rate at which a visible vehicle is not reported. Zero by
                default, because this world's detection is deterministic; above zero models
                a lossy radar rather than this world.
            presence_false_alarm_prob: Rate at which a detection has nothing behind it. Zero
                by default, for the same reason.
            clutter_position_scale_m: Cauchy scale (m) of where a false alarm reports.
            clutter_velocity_scale: Cauchy scale (m/s) of what a false alarm reports.
        """
        self._road = road
        self.max_detections = int(max_detections)
        self.curvature_lookahead_count = int(curvature_lookahead_count)
        self._max_range = float(max_detection_range_m)
        self._blocker_half_width = float(blocker_half_width_m)
        self._miss = float(presence_miss_prob)
        self._false_alarm = float(presence_false_alarm_prob)
        self._clutter_position_scale = float(clutter_position_scale_m)
        self._clutter_velocity_scale = float(clutter_velocity_scale)
        # Two copies of every width. A *draw* uses it as configured, so a zero-noise model
        # samples the exact truth rather than the truth plus a nanometre; a *score* uses the
        # floored one, because a zero-width Gaussian is a delta that annihilates the first
        # particle to miss by a hair.
        self._draw_std = {
            "ego_position": float(ego_position_std_m),
            "ego_heading": float(ego_heading_std_rad),
            "ego_arclength": float(ego_arclength_std_m),
            "ego_speed": float(ego_speed_std),
            "lane_lateral": float(lane_lateral_std_m),
            "lane_heading": float(lane_heading_std_rad),
            "curvature": float(curvature_std_1pm),
            "detection_position": float(detection_position_std_m),
            "detection_velocity": float(detection_velocity_std),
        }
        self._score_std = {name: max(width, _STD_EPS) for name, width in self._draw_std.items()}

    def encode(self, observation: Any) -> Dict[str, np.ndarray]:
        """Split the world's five-part reading into the five keys the planner works in.

        Args:
            observation: The world's ``(ego_pose, ego_speed, lane_pose, curvature_ahead,
                detections)`` tuple, or an already-encoded dictionary.

        Returns:
            ``{"ego_pose": (4,), "ego_speed": (1,), "lane_pose": (2,),
            "curvature_ahead": (L,), "detections": (K, 5)}``, all float32.

        Raises:
            ValueError: If the reading is neither a dictionary nor the five-part tuple.
        """
        if isinstance(observation, dict):
            # Already encoded -- a planner re-encoding its own draw, which the episode loop
            # does not do but a caller reasonably might.
            return {key: np.asarray(observation[key], dtype=np.float32) for key in SENSOR_KEYS}
        if not isinstance(observation, tuple) or len(observation) != _POMDP_OBSERVATION_PARTS:
            raise ValueError(
                "ObservationMode.POMDP expects the world's five-part (ego_pose, ego_speed, "
                f"lane_pose, curvature_ahead, detections) reading, got "
                f"{type(observation).__name__}. Accepting a shorter tuple here would leave a "
                "whole sensor silently unobserved."
            )
        pose, speed, lane_pose, curvature_ahead, detections = observation
        return {
            EGO_POSE_KEY: np.asarray(pose, dtype=np.float32).reshape(-1),
            EGO_SPEED_KEY: np.asarray(speed, dtype=np.float32).reshape(-1),
            LANE_POSE_KEY: np.asarray(lane_pose, dtype=np.float32).reshape(-1),
            CURVATURE_AHEAD_KEY: np.asarray(curvature_ahead, dtype=np.float32).reshape(-1),
            DETECTIONS_KEY: np.asarray(detections, dtype=np.float32).reshape(
                -1, DETECTION_SLOT_WIDTH
            ),
        }

    def draw(self, state: Any) -> Dict[str, np.ndarray]:
        """Sample one reading of ``state``, term for term with :meth:`log_prob`.

        Args:
            state: The state being observed.

        Returns:
            One encoded observation dictionary.
        """
        array = np.asarray(state, dtype=float)
        widths = self._draw_std
        lateral = float(array[EGO_LAT]) + np.random.normal(scale=widths["lane_lateral"])
        angle = float(array[EGO_ANG]) + np.random.normal(scale=widths["lane_heading"])
        clean = self.curvature_ahead(array)
        return {
            EGO_POSE_KEY: self._draw_ego_pose(array),
            EGO_SPEED_KEY: np.array(
                [float(array[EGO_SPEED]) + np.random.normal(scale=widths["ego_speed"])],
                dtype=np.float32,
            ),
            LANE_POSE_KEY: np.array([lateral, wrap_to_pi(float(angle))], dtype=np.float32),
            CURVATURE_AHEAD_KEY: (
                clean + np.random.normal(scale=widths["curvature"], size=clean.shape)
            ).astype(np.float32),
            DETECTIONS_KEY: self._draw_detections(array),
        }

    def log_prob(self, state: Any, observation: Any) -> float:
        """Score one reading against one state.

        A product of closed-form terms, one per sensor: four Gaussians over the ego-pose
        channel, one in the speedometer residual, two in the lane camera's, one per curvature
        sample, and a Bernoulli over the detection ranks with a Gaussian in each matched
        detection's position and full relative velocity.

        Nothing here can return ``-inf``. A particle predicting a detection the reading does
        not show pays ``log(presence_miss_prob)``; a detection the particle cannot explain
        costs ``log(presence_false_alarm_prob)`` plus the clutter density of what was
        reported. At the shipped rates of zero both are floored at
        :data:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_detection.PROBABILITY_EPS`,
        so either contradiction costs about 27.6 nats — a hypothesis this reading rules out,
        priced so that the filter's normalisation still has something to divide by.

        Args:
            state: The state being observed.
            observation: One encoded observation dictionary.

        Returns:
            The log-density.
        """
        array = np.asarray(state, dtype=float)
        observed_speed = float(np.asarray(observation[EGO_SPEED_KEY], dtype=float).reshape(-1)[0])
        pose = np.asarray(observation[LANE_POSE_KEY], dtype=float).reshape(-1)
        angle = wrap_to_pi(float(pose[LANE_POSE_ANG]) - float(array[EGO_ANG]))
        return (
            self._ego_pose_log_prob(array, observation)
            + gaussian_log_prob(
                np.array([observed_speed - float(array[EGO_SPEED])]), self._score_std["ego_speed"]
            )
            + gaussian_log_prob(
                np.array([float(pose[LANE_POSE_LAT]) - float(array[EGO_LAT])]),
                self._score_std["lane_lateral"],
            )
            + gaussian_log_prob(np.array([angle]), self._score_std["lane_heading"])
            + self._curvature_log_prob(array, observation)
            + self._detections_log_prob(array, observation)
        )

    def _draw_ego_pose(self, array: np.ndarray) -> np.ndarray:
        # GPS/IMU and an odometer: the four state slots the ego knows about itself, at the
        # near-exact widths that hardware delivers. Heading is wrapped after the noise, so a
        # car pointing just short of pi does not read as pointing just past -pi.
        widths = self._draw_std
        truth = array[list(_EGO_POSE_SLOTS)]
        scales = np.array(
            [
                widths["ego_position"],
                widths["ego_position"],
                widths["ego_heading"],
                widths["ego_arclength"],
            ]
        )
        measured = truth + np.random.normal(scale=1.0, size=OBSERVED_EGO_POSE_WIDTH) * scales
        measured[EGO_POSE_HEADING] = wrap_to_pi(float(measured[EGO_POSE_HEADING]))
        return measured.astype(np.float32)

    def _ego_pose_log_prob(self, array: np.ndarray, observation: Any) -> float:
        """Gaussians over the four ego-pose entries, the heading one wrapped.

        This is the term that pins a particle's *arclength*, and it is why the POMDP arm no
        longer has to infer where round the lap it is. The heading residual is wrapped
        because the channel is: without it a car at ``+3.14`` scored against a particle at
        ``-3.14`` reads as 6.28 rad of error rather than 0.003.
        """
        observed = np.asarray(observation[EGO_POSE_KEY], dtype=float).reshape(-1)
        if observed.size != OBSERVED_EGO_POSE_WIDTH:
            raise ValueError(
                f"The ego-pose channel carries {observed.size} values, expected "
                f"{OBSERVED_EGO_POSE_WIDTH} ([x, y, heading, arclength])."
            )
        truth = array[list(_EGO_POSE_SLOTS)]
        heading = wrap_to_pi(float(observed[EGO_POSE_HEADING] - truth[EGO_POSE_HEADING]))
        return (
            gaussian_log_prob(
                observed[EGO_POSE_X : EGO_POSE_X + 2] - truth[EGO_POSE_X : EGO_POSE_X + 2],
                self._score_std["ego_position"],
            )
            + gaussian_log_prob(np.array([heading]), self._score_std["ego_heading"])
            + gaussian_log_prob(
                np.array([observed[EGO_POSE_ARCLENGTH] - truth[EGO_POSE_ARCLENGTH]]),
                self._score_std["ego_arclength"],
            )
        )

    def curvature_ahead(self, array: np.ndarray) -> np.ndarray:
        """Curvature the road predicts at each lookahead for one state, shape ``(L,)``."""
        return np.asarray(
            self._road.curvature_ahead(array[None, :EGO_STATE_WIDTH]), dtype=float
        ).reshape(-1)

    def predicted_detections(self, array: np.ndarray) -> np.ndarray:
        """The detections a state should produce: ``(V, 4)`` of ``[rel_x, rel_y, vx, vy]``.

        Runs the world's own sensor geometry over the particle's agent slots — the same range
        gate and the same occlusion rule, from
        :func:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_schema.detection_visibility`
        — so *whether* a slot should have been reported is decided by one definition rather
        than two. That is what lets a particle placing an opponent behind a closer one keep
        its weight when the reading does not show it, and it is the whole mechanism the range
        dial acts through: a slot outside ``max_detection_range_m`` produces no row here, so
        the reading not showing one costs the particle nothing.

        Args:
            array: One state vector.

        Returns:
            Rows ordered by predicted range, nearest first, which is the order the
            association pairs against.
        """
        rows = state_agent_rows(array, self.max_detections)
        positions = rows[:, AGENT_REL_X : AGENT_REL_X + 2]
        visible = detection_visibility(
            positions,
            rows[:, AGENT_PRESENT] > 0.5,
            self._max_range,
            self._blocker_half_width,
        )
        seen = positions[visible]
        velocities = rows[visible][:, AGENT_REL_VX : AGENT_REL_VX + 2]
        order = np.argsort(np.linalg.norm(seen, axis=1))
        return np.column_stack([seen[order], velocities[order]])

    def _curvature_log_prob(self, array: np.ndarray, observation: Any) -> float:
        """Gaussian per lookahead sample against what the road model predicts.

        For a mapped model this is the term that scores *arclength*: a particle 20 m down the
        track from the truth reads the wrong curvature at every distance at once, and no
        other channel in the reading says where along the lap the car is. For a mapless model
        the prediction came out of this very reading, so the residual is the same for every
        particle and the term drops out at normalisation, which is the honest outcome.
        """
        observed = np.asarray(observation[CURVATURE_AHEAD_KEY], dtype=float).reshape(-1)
        predicted = self.curvature_ahead(array)
        if observed.size != self.curvature_lookahead_count or predicted.size != observed.size:
            raise ValueError(
                f"The observation reports curvature at {observed.size} distances, this model "
                f"was built for {self.curvature_lookahead_count} and its road predicts "
                f"{predicted.size}. All three must agree: the residuals are taken pairwise, "
                f"so a mismatch compares one lookahead against another."
            )
        return gaussian_log_prob(observed - predicted, self._score_std["curvature"])

    def _detections_log_prob(self, array: np.ndarray, observation: Any) -> float:
        """Bernoulli over the detection ranks, and a Gaussian per matched detection.

        **Association is by range rank.** The particle's visible slots are ordered by range
        and the ``i``-th is scored against the ``i``-th detection. Detections carry no
        identity, so some rule is needed, and rank is the cheapest defensible one. It is a
        known limit in dense traffic: two opponents at nearly equal range can swap order
        between the particle and the reading, and the residuals are then taken against the
        wrong pair. A proper joint-probabilistic association would fix it at ``K!`` cost.
        """
        predicted = self.predicted_detections(array)
        observed = np.asarray(observation[DETECTIONS_KEY], dtype=float).reshape(
            -1, DETECTION_SLOT_WIDTH
        )
        reported = observed[observed[:, DETECTION_PRESENT] > 0.5]
        ranks = np.arange(self.max_detections)
        total = bernoulli_log_prob(
            detected_probabilities(
                (ranks < len(predicted)).astype(float), self._miss, self._false_alarm
            ),
            ranks < len(reported),
        )
        matched = min(len(predicted), len(reported))
        if matched:
            total += gaussian_log_prob(
                reported[:matched, DETECTION_REL_X : DETECTION_REL_X + 2] - predicted[:matched, :2],
                self._score_std["detection_position"],
            )
            # Both components, where this used to score the closing rate alone. A particle
            # whose slot crosses the ego's path the wrong way now pays for it.
            total += gaussian_log_prob(
                reported[:matched, DETECTION_REL_VX : DETECTION_REL_VX + 2]
                - predicted[:matched, 2:4],
                self._score_std["detection_velocity"],
            )
        return total + self._clutter_log_prob(reported[matched:])

    def _clutter_log_prob(self, phantoms: np.ndarray) -> float:
        # Density of what a false alarm reported, over the detections no slot explains.
        # Without it the two branches of the detection model are a probability density on one
        # side and a bare probability on the other, and a particle that explains nothing beats
        # one that explains everything.
        if phantoms.size == 0:
            return 0.0
        return cauchy_log_prob(
            phantoms[:, DETECTION_REL_X : DETECTION_REL_X + 2], self._clutter_position_scale
        ) + cauchy_log_prob(
            phantoms[:, DETECTION_REL_VX : DETECTION_REL_VX + 2], self._clutter_velocity_scale
        )

    def _draw_detections(self, array: np.ndarray) -> np.ndarray:
        # The sampler's half of the detection model, term for term with
        # `_detections_log_prob`: a visible slot is dropped at `presence_miss_prob`, and a
        # rank no slot reaches is filled with a Cauchy phantom at
        # `presence_false_alarm_prob`. Sampling and scoring the same model is the point --
        # this package has already shipped a layer that was rendered and then not scored. At
        # the shipped rates of zero both masks are empty and the draw is exactly the
        # predicted-visible set, measured at the sensor widths.
        widths = self._draw_std
        predicted = self.predicted_detections(array)
        ranks = np.arange(self.max_detections)
        draws = np.random.random(self.max_detections)
        real = ranks < len(predicted)
        survived = real & (draws >= self._miss)
        invented = ~real & (draws < self._false_alarm)
        kept, phantoms = int(np.count_nonzero(survived)), int(np.count_nonzero(invented))

        measured = predicted[survived[: len(predicted)]] + np.column_stack(
            [
                np.random.normal(scale=widths["detection_position"], size=(kept, 2)),
                np.random.normal(scale=widths["detection_velocity"], size=(kept, 2)),
            ]
        )
        clutter = np.column_stack(
            [
                cauchy_draw((phantoms, 2), self._clutter_position_scale),
                cauchy_draw((phantoms, 2), self._clutter_velocity_scale),
            ]
        )
        packed = pack_detections(
            np.concatenate([measured, clutter]).reshape(-1, _DETECTION_REPORT_WIDTH),
            self.max_detections,
        )
        return packed.astype(np.float32)


class KinematicsObservationModel:
    """The MDP baseline: absolute position and velocity for the ego and the nearest few.

    Only the other drivers' policy stays hidden, so this is a *near*-MDP rather than a true
    one. Left unchanged by the POMDP arm's sensor redesign on purpose — it is the control the
    other arm is measured against, so re-pricing it would move the yardstick.

    Attributes:
        max_tracked_agents: Number of fixed agent slots the reading carries.
    """

    # Same reason as the sensor model above: one width per channel, travelling together.
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        max_tracked_agents: int,
        *,
        ego_pose_std: float,
        agent_pose_std: float,
        agent_velocity_std: float,
        presence_miss_prob: float,
        presence_false_alarm_prob: float,
        clutter_position_scale_m: float,
        clutter_velocity_scale: float,
    ) -> None:
        """Build the MDP arm's observation model.

        Args:
            max_tracked_agents: Number of fixed agent slots the reading carries.
            ego_pose_std: Observation noise (m and m/s) on the ego row.
            agent_pose_std: Observation noise (m) on an agent's relative position.
            agent_velocity_std: Observation noise (m/s) on an agent's relative velocity.
            presence_miss_prob: Rate at which a filled slot is not reported. Zero by default,
                because this world's detection is deterministic; above zero models a lossy
                radar rather than this world.
            presence_false_alarm_prob: Rate at which an empty slot is reported. Zero by
                default, for the same reason.
            clutter_position_scale_m: Cauchy scale (m) of where a false alarm reports.
            clutter_velocity_scale: Cauchy scale (m/s) of what a false alarm reports.
        """
        self.max_tracked_agents = int(max_tracked_agents)
        self._ego_pose_std = float(ego_pose_std)
        self._agent_pose_std = float(agent_pose_std)
        self._agent_velocity_std = float(agent_velocity_std)
        self._miss = float(presence_miss_prob)
        self._false_alarm = float(presence_false_alarm_prob)
        self._clutter_position_scale = float(clutter_position_scale_m)
        self._clutter_velocity_scale = float(clutter_velocity_scale)

    def encode(self, observation: Any) -> Dict[str, np.ndarray]:
        """Split the raw kinematics table into the ego row and body-frame agent slots.

        Args:
            observation: The raw ``(K + 1, 5)`` table of absolute
                ``[presence, x, y, vx, vy]`` rows, ego first.

        Returns:
            ``{"ego": (4,) [x, y, vx, vy], "agents": (K, 5) [present, rel_x, rel_y, rel_vx,
            rel_vy]}``, with absent rows left at zero. The agent rows are moved into the ego
            body frame, which is the frame the state's own slots live in, so the model scores
            like against like.
        """
        rows = np.asarray(observation, dtype=float)
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
        return {EGO_KEY: ego, AGENTS_KEY: agents}

    def clean(self, state: Any) -> Dict[str, np.ndarray]:
        """The noise-free reading of a state; the mean both the sampler and density use.

        Args:
            state: One state vector.

        Returns:
            The same two keys :meth:`encode` produces, with no noise applied.
        """
        array = np.asarray(state, dtype=float)
        speed, heading = float(array[EGO_SPEED]), float(array[EGO_HEADING])
        ego = np.array(
            [array[EGO_X], array[EGO_Y], speed * np.cos(heading), speed * np.sin(heading)],
            dtype=float,
        )
        agents = np.array(state_agent_rows(array, self.max_tracked_agents), dtype=float)
        return {EGO_KEY: ego, AGENTS_KEY: agents}

    def draw(self, state: Any) -> Dict[str, np.ndarray]:
        """Sample one reading of ``state``.

        Both halves of the detection model are applied, so the draw and :meth:`log_prob` are
        one model. At the shipped rates of zero neither fires: every filled slot is reported
        and no empty one is, which is what the deterministic world does.

        Args:
            state: The state being observed.

        Returns:
            One encoded observation dictionary.
        """
        clean = self.clean(state)
        return {
            EGO_KEY: clean[EGO_KEY]
            + np.random.normal(scale=self._ego_pose_std, size=_EGO_OBS_WIDTH),
            AGENTS_KEY: self._detect_agent_slots(clean[AGENTS_KEY]),
        }

    def log_prob(self, state: Any, observation: Any) -> float:
        """Score one reading against one state.

        A diagonal Gaussian over the ego row, a Bernoulli over the agent slots' presence
        flags, a diagonal Gaussian over the slots *both* sides fill, and a clutter density
        over the slots only the observation fills. A slot only the state fills is charged its
        miss rate and no more: there is no residual to take when one of the two numbers being
        differenced is a placeholder zero.

        Args:
            state: The state being observed.
            observation: One encoded observation dictionary.

        Returns:
            The log-density.
        """
        clean = self.clean(state)
        agents = clean[AGENTS_KEY]
        present = agents[:, AGENT_PRESENT] > 0.5
        observed_agents = np.asarray(observation[AGENTS_KEY], dtype=float)
        reported = observed_agents[:, AGENT_PRESENT] > 0.5
        both = present & reported
        positions = slice(AGENT_REL_X, AGENT_REL_X + 2)
        velocities = slice(AGENT_REL_VX, AGENT_REL_VX + 2)
        return (
            gaussian_log_prob(
                np.asarray(observation[EGO_KEY], dtype=float) - clean[EGO_KEY], self._ego_pose_std
            )
            + self.slot_detection_log_prob(present, reported)
            + gaussian_log_prob(
                observed_agents[both, positions] - agents[both, positions], self._agent_pose_std
            )
            + gaussian_log_prob(
                observed_agents[both, velocities] - agents[both, velocities],
                self._agent_velocity_std,
            )
            + self.clutter_log_prob(observed_agents[reported & ~present])
        )

    def slot_detection_log_prob(self, present: np.ndarray, reported: np.ndarray) -> float:
        """Bernoulli over the agent rows' presence flags.

        Without it presence is read from the state and never from the observation, so a
        particle whose slots are empty is scored on its ego row alone and pays nothing for a
        reading full of traffic.

        Args:
            present: Mask of the slots the state fills.
            reported: Mask of the slots the observation fills.

        Returns:
            The log-likelihood of the reported flags.
        """
        return bernoulli_log_prob(
            detected_probabilities(present.astype(float), self._miss, self._false_alarm), reported
        )

    def clutter_log_prob(self, phantoms: np.ndarray) -> float:
        """Density of what a false alarm reported, over the slots only the observation fills.

        Part of the lossy-radar configuration, and at ``presence_false_alarm_prob = 0`` it
        only reaches a detection the particle is already excluded by. Configure a rate and it
        stops being optional: a matched slot scores a 4-D Gaussian whose peak at
        ``agent_pose_std = 1`` and ``agent_velocity_std = 2`` is ``exp(-5.06) = 0.0063``,
        below a false-alarm rate of 0.02, so a bare rate would beat a perfect match every
        time and the model would explain every vehicle it can see as spurious. Measured
        before this term existed, at those rates: a state holding the observed vehicle scored
        1.20 nats *worse* than one holding nothing.

        Args:
            phantoms: The observed rows no state slot accounts for.

        Returns:
            The summed clutter log-density, or zero when there are none.
        """
        if phantoms.size == 0:
            return 0.0
        return cauchy_log_prob(
            phantoms[:, AGENT_REL_X : AGENT_REL_X + 2], self._clutter_position_scale
        ) + cauchy_log_prob(
            phantoms[:, AGENT_REL_VX : AGENT_REL_VX + 2], self._clutter_velocity_scale
        )

    def _detect_agent_slots(self, agents: np.ndarray) -> np.ndarray:
        # A phantom is Cauchy and nothing else. The Gaussian observation noise below belongs
        # to a real vehicle being measured, and adding it on top would make a phantom's
        # sampled density a convolution the scorer does not compute.
        positions = slice(AGENT_REL_X, AGENT_REL_X + 2)
        velocities = slice(AGENT_REL_VX, AGENT_REL_VX + 2)
        filled = agents[:, AGENT_PRESENT] > 0.5
        draws = np.random.random(agents.shape[0])
        kept = filled & (draws >= self._miss)
        invented = ~filled & (draws < self._false_alarm)
        agents[~kept & ~invented] = 0.0

        real = (int(np.count_nonzero(kept)), 2)
        agents[kept, positions] += np.random.normal(scale=self._agent_pose_std, size=real)
        agents[kept, velocities] += np.random.normal(scale=self._agent_velocity_std, size=real)

        phantom = (int(np.count_nonzero(invented)), 2)
        agents[invented, AGENT_PRESENT] = 1.0
        agents[invented, positions] = cauchy_draw(phantom, self._clutter_position_scale)
        agents[invented, velocities] = cauchy_draw(phantom, self._clutter_velocity_scale)
        return agents


__all__ = [
    "AGENTS_KEY",
    "CURVATURE_AHEAD_KEY",
    "CurvatureAhead",
    "DETECTIONS_KEY",
    "EGO_KEY",
    "EGO_POSE_KEY",
    "EGO_SPEED_KEY",
    "KinematicsObservationModel",
    "LANE_POSE_KEY",
    "ObservationArm",
    "SENSOR_KEYS",
    "SensorObservationModel",
]
