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
reads it out of the observation's curvature channel. Curvature is deliberately *not* a state
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
from POMDPPlanners.environments.racetrack_pomdp.racetrack_detection import (
    validate_detection_rates,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_sensor_model import (
    CURVATURE_AHEAD_KEY,
    DETECTIONS_KEY,
    EGO_POSE_KEY,
    EGO_SPEED_KEY,
    LANE_POSE_KEY,
    KinematicsObservationModel,
    ObservationArm,
    SensorObservationModel,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_PRESENT,
    AGENT_REL_VX,
    AGENT_REL_X,
    AGENT_SLOT_WIDTH,
    DEFAULT_ACTION_PRESETS,
    DEFAULT_ACTION_REWARD,
    DEFAULT_BLOCKER_HALF_WIDTH_M,
    DEFAULT_CLUTTER_POSITION_SCALE_M,
    DEFAULT_CLUTTER_VELOCITY_SCALE,
    DEFAULT_COLLISION_REWARD,
    DEFAULT_CURVATURE_LOOKAHEAD_M,
    DEFAULT_CURVATURE_STD_1PM,
    DEFAULT_DETECTION_POSITION_STD_M,
    DEFAULT_DETECTION_VELOCITY_STD,
    DEFAULT_EGO_ARCLENGTH_STD_M,
    DEFAULT_EGO_HEADING_STD_RAD,
    DEFAULT_EGO_POSITION_STD_M,
    DEFAULT_LANE_CENTERING_COST,
    DEFAULT_LANE_CENTERING_REWARD,
    DEFAULT_LANE_HEADING_STD_RAD,
    DEFAULT_LANE_LATERAL_STD_M,
    DEFAULT_MAX_DETECTION_RANGE_M,
    DEFAULT_MAX_TRACKED_AGENTS,
    DEFAULT_PRESENCE_FALSE_ALARM_PROB,
    DEFAULT_PRESENCE_MISS_PROB,
    DEFAULT_SPEED_LIMIT,
    EGO_ANG,
    EGO_ARCLENGTH_M,
    EGO_HEADING,
    EGO_LAT,
    EGO_SPEED,
    EGO_STATE_WIDTH,
    EGO_X,
    EGO_Y,
    MAX_ACCELERATION_MPS2,
    MAX_STEERING_RAD,
    ObservationMode,
    racetrack_reward,
    state_agent_rows,
)

# Process noise is applied to the first six ego entries only. Arclength is the seventh and
# is excluded on purpose: it is not an independent coordinate but the running integral of
# the ego's own along-track motion, already noisy through the speed it is integrated from.
# Jittering it on top would teleport a particle down the track and, worse, re-index the
# curvature profile a known-track model reads with it.
_EGO_NOISE_WIDTH = EGO_ARCLENGTH_M

# Floor for the Frenet denominator ``1 - curvature * lat``, which vanishes at the centre of
# curvature and would otherwise divide by zero on a tight arc.
_MIN_FRENET_DENOMINATOR = 1e-3


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


class RacetrackModelPOMDP(DiscreteActionsEnvironment):
    """Abstract generative racetrack model: the planner's beliefs about the world.

    Reproduces the world's ego dynamics exactly and its other vehicles only crudely (see the
    module docstring). The observation follows whichever arm of the matched pair the world is
    running, selected by ``observation_mode``; the reading is encoded, drawn and scored by
    the :class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_sensor_model.ObservationArm`
    on :attr:`observation_model`, and :meth:`encode_observation` is the single seam where the
    world's raw reading enters.

    Subclasses supply :meth:`_curvature_for`, the one thing this class does not know: where
    the road bends. A subclass whose curvature source reaches further than the ego's own
    position should also override :meth:`curvature_ahead`, which is what the observation's
    curvature channel is scored against; the default here holds one value across the channel
    and says why.

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
        curvature_lookahead_m: Distances the observed curvature channel reports at.

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
        curvature_lookahead_m: Sequence[float] = DEFAULT_CURVATURE_LOOKAHEAD_M,
        curvature_std_1pm: float = DEFAULT_CURVATURE_STD_1PM,
        max_detection_range_m: float = DEFAULT_MAX_DETECTION_RANGE_M,
        detection_position_std_m: float = DEFAULT_DETECTION_POSITION_STD_M,
        detection_velocity_std: float = DEFAULT_DETECTION_VELOCITY_STD,
        blocker_half_width_m: float = DEFAULT_BLOCKER_HALF_WIDTH_M,
        presence_miss_prob: float = DEFAULT_PRESENCE_MISS_PROB,
        presence_false_alarm_prob: float = DEFAULT_PRESENCE_FALSE_ALARM_PROB,
        clutter_position_scale_m: float = DEFAULT_CLUTTER_POSITION_SCALE_M,
        clutter_velocity_scale: float = DEFAULT_CLUTTER_VELOCITY_SCALE,
        ego_position_std_m: float = DEFAULT_EGO_POSITION_STD_M,
        ego_heading_std_rad: float = DEFAULT_EGO_HEADING_STD_RAD,
        ego_arclength_std_m: float = DEFAULT_EGO_ARCLENGTH_STD_M,
        ego_speed_std: float = 0.1,
        lane_lateral_std_m: float = DEFAULT_LANE_LATERAL_STD_M,
        lane_heading_std_rad: float = DEFAULT_LANE_HEADING_STD_RAD,
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
            curvature_lookahead_m: Distances (m) along the lane the observed curvature
                channel reports at. Defaults to ``(10.0, 20.0, 30.0)`` and **must match the
                world's**: the likelihood scores one Gaussian per entry, so a mismatch
                compares the curvature at one distance against the curvature at another.
            curvature_std_1pm: Width (1/m) of the curvature channel's likelihood. Defaults
                to 0.002, the world's own camera noise.
            max_detection_range_m: Range (m) beyond which this model predicts a tracked
                opponent is *not* reported. Defaults to 40.0, matching the world. This is the
                arm's dial, and it acts through the likelihood as much as through the world:
                a slot the model believes is out of range costs a particle nothing when the
                reading does not show it, and a slot inside range does.
            detection_position_std_m: Width (m) of a detection's position likelihood.
                Defaults to 0.5, the world's own noise.
            detection_velocity_std: Width (m/s) of a detection's relative-velocity
                likelihood, applied to both components. Defaults to 0.3, the world's own.
            blocker_half_width_m: Half-width (m) of a vehicle when predicting occlusion.
                Defaults to 1.0. Together with ``max_detection_range_m`` this is what lets
                the model predict *whether* a slot should have produced a detection, which
                is why a particle placing an opponent behind another one is not punished for
                the observation not showing it.
            presence_miss_prob: Rate at which a vehicle this model tracks fails to appear in
                the observation. Defaults to 0.0.
            presence_false_alarm_prob: Rate at which the observation reports a vehicle this
                model tracks nothing behind. Defaults to 0.0. Together with
                ``presence_miss_prob`` this is the **detection model**, and it is a different
                kind of number from the widths above: those say how wrong a reported quantity
                is, these say whether the report happens at all. It is what lets the
                likelihood score *whether* a vehicle is there rather than only where —
                without it a particle with empty slots is scored on its ego row alone and
                pays nothing for a reading full of traffic. Both default to zero because this
                world's detection decision is deterministic: the range gate and the occlusion
                rule run on true positions, and the radar drops nothing it can see and
                invents nothing. A particle contradicting the reading's visibility is
                therefore excluded, at a 27.6-nat floor rather than ``-inf`` so the filter's
                normalisation survives it. Nonzero values model a lossy radar, which is a
                legitimate configuration and not this world.
            clutter_position_scale_m: Cauchy scale (m) of where a false alarm reports a
                phantom. Defaults to 18.0.
            clutter_velocity_scale: Cauchy scale (m/s) of a phantom's reported velocity.
                Defaults to 10.0, the speed limit. Both belong to the lossy-radar
                configuration: at a false-alarm rate of zero no phantom is ever drawn, and
                with a rate configured they are what keeps a bare probability comparable with
                a matched detection's *density* — leaving the clutter term out inverts the
                likelihood outright, see ``DEFAULT_CLUTTER_POSITION_SCALE_M``.
            ego_position_std_m: Width (m) of the ego-pose channel's ``x`` and ``y``
                likelihood. Defaults to 0.1, the world's own localisation noise. Like the
                lane camera's widths and unlike ``ego_speed_std``, this **is** a sensor
                model and must match the world's or the filter is confidently wrong.
            ego_heading_std_rad: Width (rad) of the ego-pose channel's heading likelihood,
                taken modulo 2*pi so a particle either side of the branch cut is not charged
                6.28 rad of error. Defaults to 0.01.
            ego_arclength_std_m: Width (m) of the ego-pose channel's arclength likelihood.
                Defaults to 0.1. This is the term that pins a particle's position around the
                lap, which the curvature channel alone used to have to do.
            ego_speed_std: Width (m/s) of the POMDP speedometer likelihood. Defaults to 0.1.
                **Not a sensor model**: the world emits ego speed exactly, and a real
                speedometer is accurate to well under a percent. It stands for this model's
                own ego-dynamics error, and a zero-width likelihood would be a delta that
                collapses the filter on the first mismatch. Left unfitted deliberately —
                fitting it to the world would fit it to zero.
            lane_lateral_std_m: Width (m) of the lane camera's lateral likelihood. Defaults
                to 0.05. Unlike ``ego_speed_std`` this one **is** a sensor model: the world
                genuinely corrupts its lane reading at this width, and the two must match or
                the filter is confidently wrong. Change it only alongside the world's.
            lane_heading_std_rad: Width (rad) of the lane camera's heading likelihood.
                Defaults to 0.01, matching the world's default for the same reason.
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
            ValueError: If ``max_tracked_agents`` or ``substeps`` is below 1, ``dt`` is not
                positive, ``curvature_lookahead_m`` is empty, or the two detection rates are
                not each in ``[0, 1)`` and summing below 1.
        """
        if max_tracked_agents < 1:
            raise ValueError(f"max_tracked_agents must be at least 1, got {max_tracked_agents}.")
        if substeps < 1:
            raise ValueError(f"substeps must be at least 1, got {substeps}.")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        if len(tuple(curvature_lookahead_m)) == 0:
            raise ValueError(
                "curvature_lookahead_m must name at least one distance; an empty channel "
                "would silently drop the only reading that says where the road bends."
            )
        validate_detection_rates(presence_miss_prob, presence_false_alarm_prob)
        for scale_name, scale in (
            ("clutter_position_scale_m", clutter_position_scale_m),
            ("clutter_velocity_scale", clutter_velocity_scale),
        ):
            if scale <= 0.0:
                raise ValueError(
                    f"{scale_name} must be positive, got {scale}. It is the scale of a "
                    f"Cauchy over where a false alarm reports a phantom."
                )

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
        self.curvature_lookahead_m: Tuple[float, ...] = tuple(
            float(distance) for distance in curvature_lookahead_m
        )
        self.curvature_std_1pm = float(curvature_std_1pm)
        self.max_detection_range_m = float(max_detection_range_m)
        self.detection_position_std_m = float(detection_position_std_m)
        self.detection_velocity_std = float(detection_velocity_std)
        self.blocker_half_width_m = float(blocker_half_width_m)
        self.presence_miss_prob = float(presence_miss_prob)
        self.presence_false_alarm_prob = float(presence_false_alarm_prob)
        self.clutter_position_scale_m = float(clutter_position_scale_m)
        self.clutter_velocity_scale = float(clutter_velocity_scale)
        self.ego_position_std_m = float(ego_position_std_m)
        self.ego_heading_std_rad = float(ego_heading_std_rad)
        self.ego_arclength_std_m = float(ego_arclength_std_m)
        self.ego_speed_std = float(ego_speed_std)
        self.lane_lateral_std_m = float(lane_lateral_std_m)
        self.lane_heading_std_rad = float(lane_heading_std_rad)
        self.ego_pose_std = float(ego_pose_std)
        self.agent_pose_std = float(agent_pose_std)
        self.agent_velocity_std = float(agent_velocity_std)
        self.process_noise_std = float(process_noise_std)
        self._observation_model = self._build_observation_model()

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
    def observation_model(self) -> ObservationArm:
        """The arm this model encodes, draws and scores readings with.

        A property rather than an ordinary attribute, and not for style: ``config_id``
        walks ``__dict__`` and recurses into anything holding one, so a public attribute
        pointing at a collaborator that points back here recurses until the stack ends.
        Nothing is lost by hiding it from the identity — every width the collaborator was
        built from is already a public attribute of this model.
        """
        return self._observation_model

    def _build_observation_model(self) -> ObservationArm:
        """The arm this model observes in, built once from the widths above."""
        if self.observation_mode is ObservationMode.POMDP:
            return SensorObservationModel(
                self,
                self.max_tracked_agents,
                ego_position_std_m=self.ego_position_std_m,
                ego_heading_std_rad=self.ego_heading_std_rad,
                ego_arclength_std_m=self.ego_arclength_std_m,
                ego_speed_std=self.ego_speed_std,
                lane_lateral_std_m=self.lane_lateral_std_m,
                lane_heading_std_rad=self.lane_heading_std_rad,
                curvature_std_1pm=self.curvature_std_1pm,
                curvature_lookahead_count=len(self.curvature_lookahead_m),
                max_detection_range_m=self.max_detection_range_m,
                detection_position_std_m=self.detection_position_std_m,
                detection_velocity_std=self.detection_velocity_std,
                blocker_half_width_m=self.blocker_half_width_m,
                presence_miss_prob=self.presence_miss_prob,
                presence_false_alarm_prob=self.presence_false_alarm_prob,
                clutter_position_scale_m=self.clutter_position_scale_m,
                clutter_velocity_scale=self.clutter_velocity_scale,
            )
        return KinematicsObservationModel(
            self.max_tracked_agents,
            ego_pose_std=self.ego_pose_std,
            agent_pose_std=self.agent_pose_std,
            agent_velocity_std=self.agent_velocity_std,
            presence_miss_prob=self.presence_miss_prob,
            presence_false_alarm_prob=self.presence_false_alarm_prob,
            clutter_position_scale_m=self.clutter_position_scale_m,
            clutter_velocity_scale=self.clutter_velocity_scale,
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
    def curvature_ahead(self, ego: np.ndarray) -> np.ndarray:
        """Curvature at each lookahead distance per ego row, shape ``(B, L)``.

        The default holds the curvature under the ego across the whole channel, which is the
        honest answer for a model whose only source *is* that channel: a mapless planner has
        nothing to say about 30 m ahead that it did not read off the observation it is
        scoring, so the residual comes out identical for every particle and the term drops
        out at normalisation. A model holding a track map should override this — the
        curvature 30 m along the lane from a particle's own arclength is exactly the kind of
        prediction that separates one particle from another.

        Args:
            ego: The ego block of the particle batch, shape ``(B, EGO_STATE_WIDTH)``.

        Returns:
            Signed curvature in 1/m, one column per entry of :attr:`curvature_lookahead_m`.
        """
        return np.repeat(
            self._curvature_for(ego).reshape(-1, 1), len(self.curvature_lookahead_m), axis=1
        )

    def encode_observation(self, observation: Any) -> Dict[str, np.ndarray]:
        """Encode the world's raw reading into the space the belief and planner use.

        This is the single raw-observation seam, and it delegates to whichever arm's
        observation model this instance was built with; see
        :mod:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_sensor_model`.

        Args:
            observation: The raw observation emitted by the world, in this model's mode.

        Returns:
            POMDP mode: ``{"ego_pose": (4,), "ego_speed": (1,), "lane_pose": (2,),
            "curvature_ahead": (L,), "detections": (K, 5)}``, all float32.
            MDP mode: ``{"ego": (4,), "agents": (K, 5)}``, the agent rows in the ego body
            frame with absent ones left at zero.

        Raises:
            ValueError: If a POMDP reading is not the five-part sensor tuple.
        """
        return self._observation_model.encode(observation)

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
        draws = [self._observation_model.draw(next_state) for _ in range(max(n_samples, 1))]
        return draws[0] if n_samples == 1 else draws

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        """Log-density of each observation given ``next_state``.

        In POMDP mode this is a product of closed-form terms, one per sensor: four Gaussians
        over the ego-pose channel, one in the speedometer residual, two in the lane camera's,
        one per curvature-ahead sample against whatever :meth:`curvature_ahead` predicts, and
        a Bernoulli over the detection ranks with a Gaussian in each matched detection's
        position and full relative velocity. Detections are associated to slots by range
        rank, which is a known limit in dense traffic. In MDP mode it is the
        near-fully-observed kinematics density.

        Both arms discriminate on *whether* a vehicle is there and not only where, and both
        do it at a rate rather than a hard zero — an opponent leaves sensor range or slips
        behind a closer car every few steps, and a ``-inf`` would collapse the filter each
        time one did. That term is where ``max_detection_range_m`` bites: a reading with no
        row for a car a particle places inside the gate is charged, and one for a car it
        places outside is not.

        Args:
            next_state: The state being observed.
            action: Unused; the observation depends on the state alone.
            observations: One observation dictionary, or a sequence of ``N`` of them.

        Returns:
            ``np.ndarray`` of shape ``(N,)`` of log-densities.
        """
        del action
        candidates = [observations] if isinstance(observations, dict) else list(observations)
        return np.array(
            [self._observation_model.log_prob(next_state, obs) for obs in candidates], dtype=float
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


__all__ = [
    "CURVATURE_AHEAD_KEY",
    "DETECTIONS_KEY",
    "EGO_POSE_KEY",
    "EGO_SPEED_KEY",
    "LANE_POSE_KEY",
    "RacetrackModelPOMDP",
]
