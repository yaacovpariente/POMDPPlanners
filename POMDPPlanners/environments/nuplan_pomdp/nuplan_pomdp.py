# SPDX-License-Identifier: MIT

# pylint: disable=too-many-lines  # Multi-agent traffic + world backend grew the module.

"""nuPlan POMDP world environment.

This module adapts the `nuPlan <https://www.nuplan.org/>`_ closed-loop planning
simulator to the POMDPPlanners :class:`~POMDPPlanners.core.environment.Environment`
interface so it can serve as the **ground-truth world** in an
:class:`~POMDPPlanners.simulations.episodes.EpisodeRunner`.

nuPlan is *forward-only*: a :class:`Simulation` advances a single true state one
iteration per call, propagating the ego under a planned trajectory while reactive
background agents (IDM) respond. It cannot be queried for a transition/observation
density nor re-run from an arbitrary injected state, so it cannot act as a planner's
generative model. In the two-environment episode design the planner keeps its own
generative model (``policy.environment``) and this wrapper only advances the single
true state forward, one step per real interaction. Consequently
:meth:`NuPlanPOMDP.transition_log_probability` and
:meth:`NuPlanPOMDP.observation_log_probability` intentionally raise
:class:`NotImplementedError` — in the intended world/model split they are never called.

Unlike a fully-observed gym wrapper (observation equals state), nuPlan is genuinely
partially observed: the ego reads its own proprioception plus a *tracked-object* list
(``DetectionsTracks``) of the nearest agents, not the world's full ground truth.

The **state** is the ego vehicle's ground-truth kinematics and lane pose,
``[x, y, yaw, vx, vy, lat, heading_err]``, **followed by fixed slots for the
``max_tracked_agents`` nearest other agents** (ground truth). Each agent slot is
``[present, rel_x, rel_y, rel_yaw, rel_speed]`` expressed in the ego frame
(``rel_x`` forward, ``rel_y`` left, ``rel_yaw`` in **radians**, ``rel_speed`` in
m/s); ``present`` is ``1`` for a filled slot and ``0`` for padding. The ego part is
read straight from the simulator, where:

- ``x``, ``y``: ego rear-axle position in the map frame, in metres.
- ``yaw``: ego heading about the map Z axis, in **radians** (nuPlan convention).
- ``vx``, ``vy``: ego linear-velocity components in the map frame, in metres per second.
- ``lat``: signed lateral offset from the centre of the ego's route baseline, in metres
  (positive to the baseline's left).
- ``heading_err``: ego heading minus the route-baseline direction, wrapped to
  ``[-pi, pi]``, in **radians**.

The state ends with **one traffic-light slot**,
``[present, rel_x, rel_y, state_code, time_to_change]`` (ego frame; ``state_code`` is a
``TRAFFIC_LIGHT_*`` code, ``time_to_change`` in seconds), carrying the light governing the
ego lane as ground truth (``present == 0`` when none affects it). It is always in the state
and is independent of whether the *observation* exposes the light.

(The vertical axis ``z`` and roll/pitch are intentionally omitted; the ego is modelled on
the ground plane.)

The lane-relative terms (``lat``, ``heading_err``) drive a gym-carla-style driving-quality
reward: it rewards longitudinal progress along the route while penalising overspeed,
drifting off the baseline, and harsh / high-speed steering, plus a per-step time cost and a
terminal collision penalty. See :data:`REWARD_SPEED_WEIGHT` and the sibling weights.

The **observation** is a multi-modal dict of native nuPlan payloads:

- ``"ego"`` (always present): the ego's proprioceptive kinematics
  ``[x, y, yaw, vx, vy, lat, heading_err]`` — nuPlan gives the ego near-perfect
  self-localisation, so this is the ego measurement channel.
- ``"agents"`` (always present): the ``max_tracked_agents`` agent slots of the state
  flattened, reported **raw** at their true ego-frame poses. The world applies no
  perception, so this is the ground-truth channel; range-gating, occlusion and sensor
  noise are the planner model's observation model, not the world's.

Any measurement noise is the planner model's; the wrapper adds none.

Classes:
    NuPlanPOMDP: Forward-only adapter exposing a nuPlan session as a world Environment.
"""

from collections.abc import Hashable
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import Environment, SpaceInfo, SpaceType
from POMDPPlanners.core.simulation import History, MetricValue
from POMDPPlanners.core.simulation.step_info_metrics import require_non_empty_histories
from POMDPPlanners.utils.statistics_utils import confidence_interval

# Default discrete control presets as ``(acceleration, steering_angle)`` pairs, in
# nuPlan's native control space: longitudinal acceleration in m/s^2 and front-wheel
# steering angle in radians.
DEFAULT_ACTION_PRESETS: Tuple[Tuple[float, float], ...] = (
    (1.5, 0.0),  # accelerate straight
    (0.0, -0.3),  # coast, steer left
    (0.0, 0.3),  # coast, steer right
    (-3.0, 0.0),  # brake
)

# Per-step getter roles that may trigger the single nuPlan iteration. Each role may be
# served from the cached iteration at most once, so a repeated role signals a new step.
_ROLE_NEXT_STATE = "next_state"
_ROLE_REWARD = "reward"

# Fixed gym-carla-style reward term weights. The collision weight is the tunable
# ``collision_penalty`` constructor argument; these shape the driving-quality terms and
# are held fixed so the reward is a single well-defined objective.
REWARD_SPEED_WEIGHT = 1.0  # reward per m/s of along-route (longitudinal) progress
REWARD_FAST_PENALTY = 10.0  # penalty when longitudinal speed exceeds desired_speed
REWARD_OUT_PENALTY = 1.0  # penalty when |lat| exceeds out_lane_thresh
REWARD_STEER_WEIGHT = 5.0  # penalty on squared steering (harsh-steer smoothness)
REWARD_LAT_WEIGHT = 0.2  # penalty on |steer| * longitudinal_speed**2 (fast turns)
REWARD_STEP_COST = 0.1  # constant per-step time cost


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians to the ``[-pi, pi]`` interval.

    Every angle in the nuPlan state/observation layout (ego ``yaw`` and ``heading_err``, each
    agent slot's ``rel_yaw``) carries this invariant, so world, model and belief all route their
    angle arithmetic through here.

    Args:
        angle: Angle in radians.

    Returns:
        The equivalent angle in ``[-pi, pi]``.
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def driving_quality_reward(
    next_state: np.ndarray,
    steering_angle: float,
    terminated: bool,
    desired_speed: float,
    out_lane_thresh: float,
    collision_penalty: float,
) -> float:
    """Score a transition with a gym-carla-style driving-quality reward.

    Rewards along-route progress and penalises overspeed, drifting off the route
    baseline, harsh / high-speed steering, each elapsed step, and a terminal collision.
    Shared by the :class:`NuPlanPOMDP` world and the planner-side factored model so the
    two score a transition identically by construction.

    Args:
        next_state: Resulting ego state ``[x, y, yaw(rad), vx, vy, lat, heading_err]``.
        steering_angle: Steering command applied on the transition (from the action preset).
        terminated: Whether the transition ended in a terminal collision.
        desired_speed: Target longitudinal speed (m/s); exceeding it is penalised.
        out_lane_thresh: Lateral offset (m) beyond which the ego is treated as off-route.
        collision_penalty: Penalty scale applied on a terminal collision.

    Returns:
        The scalar reward for the transition.
    """
    ego_yaw = float(next_state[2])  # radians (nuPlan-native)
    vel_x, vel_y = float(next_state[3]), float(next_state[4])
    lateral, heading_err = float(next_state[5]), float(next_state[6])
    lane_yaw = ego_yaw - heading_err
    lspeed_lon = vel_x * np.cos(lane_yaw) + vel_y * np.sin(lane_yaw)

    r_fast = -1.0 if lspeed_lon > desired_speed else 0.0
    r_out = -1.0 if abs(lateral) > out_lane_thresh else 0.0
    r_collision = -1.0 if terminated else 0.0
    r_steer = -(steering_angle**2)
    r_lat = -abs(steering_angle) * lspeed_lon**2

    return float(
        collision_penalty * r_collision
        + REWARD_SPEED_WEIGHT * lspeed_lon
        + REWARD_FAST_PENALTY * r_fast
        + REWARD_OUT_PENALTY * r_out
        + REWARD_STEER_WEIGHT * r_steer
        + REWARD_LAT_WEIGHT * r_lat
        - REWARD_STEP_COST
    )


# Surrounding-traffic / perception defaults.
DEFAULT_MAX_TRACKED_AGENTS = 5  # nearest other agents carried in state/observation
DEFAULT_PERCEPTION_RANGE = 50.0  # metres beyond which another agent is undetectable
DEFAULT_OCCLUSION_RADIUS = 1.5  # metres; a vehicle nearer than this to the ego->target
# sight line is treated as blocking (geometric occlusion among vehicles)
DEFAULT_SIMULATION_HORIZON = 8.0  # seconds of trajectory rolled out per propagate

# State/observation layout for other agents. Each tracked agent occupies a fixed slot
# ``[present, rel_x, rel_y, rel_yaw, rel_speed]`` expressed in the ego frame
# (``rel_x`` forward, ``rel_y`` left, ``rel_yaw`` in radians, ``rel_speed`` in m/s).
EGO_STATE_WIDTH = 7  # [x, y, yaw, vx, vy, lat, heading_err]
AGENT_SLOT_WIDTH = 5  # [present, rel_x, rel_y, rel_yaw, rel_speed]

# The ground-truth state ends with a single traffic-light slot
# ``[present, rel_x, rel_y, state_code, time_to_change]`` (ego frame; ``state_code`` is one
# of the ``TRAFFIC_LIGHT_*`` codes below, ``time_to_change`` in seconds). ``present == 0``
# when no light affects the ego lane.
LIGHT_SLOT_WIDTH = 5  # [present, rel_x, rel_y, state_code, time_to_change]

# state_code values carried in the traffic-light state slot.
TRAFFIC_LIGHT_GREEN = 0.0
TRAFFIC_LIGHT_RED = 1.0
TRAFFIC_LIGHT_YELLOW = 2.0
TRAFFIC_LIGHT_OFF = 3.0  # light present but not operating (dark / flashing)
TRAFFIC_LIGHT_UNKNOWN = 4.0

# Centre-to-centre ego->agent distance (m) at/under which a step counts toward a near-miss.
_NEAR_MISS_DISTANCE = 2.5

# Indices into the ego state block, used by the evaluation metrics.
_EGO_POSITION_SLICE = slice(0, 2)  # (x, y) map position in metres
_EGO_VELOCITY_SLICE = slice(3, 5)  # (vx, vy) map velocity in m/s


class NuPlanPOMDPMetrics(Enum):
    """Metric names for the nuPlan POMDP environment."""

    COLLISION_RATE = "collision_rate"
    AVERAGE_PROGRESS = "average_progress"
    AVERAGE_SPEED = "average_speed"
    NEAR_MISS_COUNT = "near_miss_count"
    MIN_VEHICLE_DISTANCE = "min_vehicle_distance"


def _driven_steps(history: History) -> List[Any]:
    """The steps of ``history`` that actually advanced the world.

    An episode that ends in a collision is closed by the runner with a terminal
    ``StepData`` carrying ``next_state=None``; metrics reading a transition must skip it.
    """
    return [step for step in history.history if step.next_state is not None]


def relative_agent_row(
    ego_x: float,
    ego_y: float,
    ego_yaw_rad: float,
    other_x: float,
    other_y: float,
    other_yaw_rad: float,
    other_speed: float,
) -> np.ndarray:
    """Express another agent's pose/speed in the ego frame as a present slot row.

    Returns ``[1.0, rel_x, rel_y, rel_yaw, rel_speed]`` with ``rel_x`` pointing along the
    ego heading, ``rel_y`` to its left, and ``rel_yaw`` wrapped to ``[-pi, pi]``.

    Args:
        ego_x: Ego x position in the map frame (m).
        ego_y: Ego y position in the map frame (m).
        ego_yaw_rad: Ego heading (rad).
        other_x: Other agent x position in the map frame (m).
        other_y: Other agent y position in the map frame (m).
        other_yaw_rad: Other agent heading (rad).
        other_speed: Other agent speed (m/s).

    Returns:
        The ego-frame present slot row for the agent.
    """
    delta_x = other_x - ego_x
    delta_y = other_y - ego_y
    cos_yaw = np.cos(ego_yaw_rad)
    sin_yaw = np.sin(ego_yaw_rad)
    rel_x = float(cos_yaw * delta_x + sin_yaw * delta_y)
    rel_y = float(-sin_yaw * delta_x + cos_yaw * delta_y)
    rel_yaw = float(wrap_to_pi(other_yaw_rad - ego_yaw_rad))
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
    """Whether a blocker agent lies on the ego->target sight line within ``radius``.

    Projects the blocker onto the ego->target segment; it occludes when the projection
    falls strictly between the endpoints and its perpendicular distance to the line is
    below ``radius``.
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


def assemble_state(
    ego_row: Union[Sequence[float], np.ndarray],
    agent_rows: Union[Sequence[Sequence[float]], Sequence[np.ndarray]],
    max_tracked_agents: int,
    light_row: Optional[Union[Sequence[float], np.ndarray]] = None,
) -> np.ndarray:
    """Concatenate an ego row, padded agent slots, and a light slot into a state vector.

    Pure numeric assembly of the nuPlan state layout, factored out of the live session so
    the state geometry can be exercised without a nuPlan installation. Agent rows are
    written into the nearest fixed slots (already ego-frame); missing slots are padded with
    zeros (``present == 0``).

    Args:
        ego_row: The ``EGO_STATE_WIDTH`` ego block ``[x, y, yaw, vx, vy, lat, heading_err]``.
        agent_rows: Zero or more ego-frame agent rows ``[present, rel_x, rel_y, rel_yaw,
            rel_speed]``; only the first ``max_tracked_agents`` are kept.
        max_tracked_agents: Number of fixed agent slots to emit.
        light_row: Optional ``LIGHT_SLOT_WIDTH`` traffic-light slot; a zero (absent) slot is
            emitted when ``None``.

    Returns:
        The full state vector of width
        ``EGO_STATE_WIDTH + max_tracked_agents * AGENT_SLOT_WIDTH + LIGHT_SLOT_WIDTH``.
    """
    ego = np.asarray(ego_row, dtype=float).reshape(EGO_STATE_WIDTH)
    slots = np.zeros((max_tracked_agents, AGENT_SLOT_WIDTH), dtype=float)
    for index, row in enumerate(agent_rows):
        if index >= max_tracked_agents:
            break
        slots[index] = np.asarray(row, dtype=float).reshape(AGENT_SLOT_WIDTH)
    light = (
        np.zeros(LIGHT_SLOT_WIDTH, dtype=float)
        if light_row is None
        else np.asarray(light_row, dtype=float).reshape(LIGHT_SLOT_WIDTH)
    )
    return np.concatenate([ego, slots.reshape(-1), light])


class _NuPlanSession:
    """Live nuPlan session: a closed-loop :class:`Simulation` over one scenario.

    This is the only object that talks to the ``nuplan`` package. It exposes a small
    forward-only interface (:meth:`reset` / :meth:`step`) that mirrors a Gymnasium env so
    :class:`NuPlanPOMDP` can drive it and tests can substitute a scripted fake with the same
    two methods.

    The session drives nuPlan's closed-loop stack: a :class:`Simulation` with reactive
    ``IDMAgents`` background traffic and a two-stage tracking controller. Each :meth:`step`
    rolls a constant-control kinematic-bicycle trajectory over the simulation horizon,
    propagates the simulation one iteration along it, and reads back the ego state plus the
    tracked-object observation. It therefore requires the nuPlan devkit **and** a scenario
    (from the nuPlan dataset), supplied by the ``scenario_loader`` callable.
    """

    def __init__(
        self,
        scenario_loader: Callable[[], Any],
        max_tracked_agents: int,
        simulation_horizon: float,
        fixed_delta_seconds: float,
        reactive_agents: bool,
        collision_distance: float,
    ) -> None:
        # nuPlan is a heavy optional dependency; import lazily so the module stays
        # importable (and testable via a fake session) without the devkit installed.
        # pylint: disable=import-outside-toplevel,import-error
        import nuplan  # noqa: F401  (presence check; submodules imported on reset)

        del nuplan
        self._scenario_loader = scenario_loader
        self._max_tracked_agents = max_tracked_agents
        self._simulation_horizon = simulation_horizon
        self._fixed_delta_seconds = fixed_delta_seconds
        self._reactive_agents = reactive_agents
        self._collision_distance = collision_distance
        self._simulation: Optional[Any] = None
        self._scenario: Optional[Any] = None
        self._motion_model: Optional[Any] = None
        self._collided = False

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Build a fresh simulation from a scenario and read the start state/observation."""
        del seed  # scenario selection is the loader's responsibility
        self._scenario = self._scenario_loader()
        self._simulation = self._build_simulation(self._scenario)
        self._simulation.initialize()
        self._collided = False
        return self._read_state(), self._read_observation()

    def step(
        self, acceleration: float, steering_angle: float
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], bool]:
        """Propagate one iteration under a constant-control trajectory; read the outcome."""
        if self._simulation is None:
            raise RuntimeError("_NuPlanSession.step called before reset")
        trajectory = self._constant_control_trajectory(acceleration, steering_angle)
        self._simulation.propagate(trajectory)
        state = self._read_state()
        observation = self._read_observation()
        self._collided = self._detect_collision(state)
        return state, observation, self._collided

    def _build_simulation(self, scenario: Any) -> Any:
        """Assemble a closed-loop nuPlan :class:`Simulation` for ``scenario``."""
        # pylint: disable=import-outside-toplevel,import-error
        from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import (
            KinematicBicycleModel,
        )
        from nuplan.planning.simulation.controller.perfect_tracking import (
            PerfectTrackingController,
        )
        from nuplan.planning.simulation.observation.idm_agents import IDMAgents
        from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
        from nuplan.planning.simulation.simulation import Simulation
        from nuplan.planning.simulation.simulation_setup import SimulationSetup
        from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (  # noqa: E501
            StepSimulationTimeController,
        )

        self._motion_model = KinematicBicycleModel(
            scenario.get_ego_state_at_iteration(0).car_footprint.vehicle
        )
        observation = (
            IDMAgents(scenario=scenario) if self._reactive_agents else TracksObservation(scenario)
        )
        setup = SimulationSetup(
            time_controller=StepSimulationTimeController(scenario),
            observations=observation,
            ego_controller=PerfectTrackingController(scenario),
            scenario=scenario,
        )
        return Simulation(simulation_setup=setup)

    def _constant_control_trajectory(self, acceleration: float, steering_angle: float) -> Any:
        """Roll the kinematic-bicycle model forward under a constant control command.

        The action preset is ``(longitudinal acceleration, steering *angle*)``, while nuPlan's
        motion model consumes a rear-axle acceleration vector and a tire steering *rate*. The
        acceleration is commanded directly; the steering angle is converted into the rate that
        reaches it within one integration step and then held (rate 0), so the rollout tracks the
        constant angle the reward scores.
        """
        # pylint: disable=import-outside-toplevel,import-error
        from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
        from nuplan.common.actor_state.state_representation import StateVector2D
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
            InterpolatedTrajectory,
        )

        planner_input = self._simulation.get_planner_input()
        current = planner_input.history.current_state[0]
        states = [current]
        steps = max(1, int(round(self._simulation_horizon / self._fixed_delta_seconds)))
        for _ in range(steps):
            steering_rate = (
                steering_angle - float(states[-1].tire_steering_angle)
            ) / self._fixed_delta_seconds
            command = DynamicCarState.build_from_rear_axle(
                rear_axle_to_center_dist=states[-1].car_footprint.rear_axle_to_center_dist,
                rear_axle_velocity_2d=states[-1].dynamic_car_state.rear_axle_velocity_2d,
                rear_axle_acceleration_2d=StateVector2D(acceleration, 0.0),
                tire_steering_rate=steering_rate,
            )
            states.append(
                self._motion_model.propagate_state(states[-1], command, self._fixed_delta_seconds)
            )
        return InterpolatedTrajectory(states)

    def _read_state(self) -> np.ndarray:
        """Read the ego + nearest-agent + light ground truth into the state layout."""
        planner_input = self._simulation.get_planner_input()
        ego_state, detections = planner_input.history.current_state
        ego_row = self._ego_row(ego_state)
        agent_rows = self._nearest_agent_rows(ego_state, detections)
        return assemble_state(ego_row, agent_rows, self._max_tracked_agents)

    def _read_observation(self) -> Dict[str, np.ndarray]:
        """The raw, fully-detected ``{ego, agents}`` observation from the live state."""
        state = self._read_state()
        agents_end = EGO_STATE_WIDTH + self._max_tracked_agents * AGENT_SLOT_WIDTH
        return {
            "ego": state[:EGO_STATE_WIDTH].copy(),
            "agents": state[EGO_STATE_WIDTH:agents_end].copy(),
        }

    def _ego_row(self, ego_state: Any) -> np.ndarray:
        center = ego_state.center
        velocity = ego_state.dynamic_car_state.center_velocity_2d
        return np.array(
            [
                float(center.x),
                float(center.y),
                float(center.heading),
                float(velocity.x),
                float(velocity.y),
                0.0,  # lateral offset from route baseline (filled by a route-aware build)
                0.0,  # heading error vs route baseline
            ]
        )

    def _nearest_agent_rows(self, ego_state: Any, detections: Any) -> List[np.ndarray]:
        ego_x = float(ego_state.center.x)
        ego_y = float(ego_state.center.y)
        ego_yaw = float(ego_state.center.heading)
        rows: List[Tuple[float, np.ndarray]] = []
        for obj in getattr(detections, "tracked_objects", []):
            other_x = float(obj.center.x)
            other_y = float(obj.center.y)
            speed = float(getattr(obj.velocity, "magnitude", lambda: 0.0)())
            row = relative_agent_row(
                ego_x, ego_y, ego_yaw, other_x, other_y, float(obj.center.heading), speed
            )
            rows.append((float(np.hypot(row[1], row[2])), row))
        rows.sort(key=lambda item: item[0])
        return [row for _distance, row in rows[: self._max_tracked_agents]]

    def _detect_collision(self, state: np.ndarray) -> bool:
        agents_end = EGO_STATE_WIDTH + self._max_tracked_agents * AGENT_SLOT_WIDTH
        rows = state[EGO_STATE_WIDTH:agents_end].reshape(self._max_tracked_agents, AGENT_SLOT_WIDTH)
        present = rows[rows[:, 0] == 1.0]
        if present.shape[0] == 0:
            return False
        return bool(np.min(np.hypot(present[:, 1], present[:, 2])) < self._collision_distance)


class NuPlanPOMDP(Environment):
    """Forward-only adapter exposing a nuPlan closed-loop session as a world POMDP.

    The wrapper drives a nuPlan :class:`Simulation` as the ground-truth world of an episode.
    It advances the simulator exactly one iteration per real interaction and serves the
    resulting next state, observation and reward from a small cache, because the
    POMDPPlanners episode loop requests those three quantities through separate method calls
    while nuPlan produces them atomically. The state is the ego vehicle's ground-truth
    kinematics; the observation is the ego proprioception plus a tracked-object list, so the
    world is genuinely partially observed. See the module docstring for the exact state and
    observation variables, units, and frames.

    Note:
        This is a *world* environment, not a generative model. It cannot sample a transition
        from an arbitrary state, so belief particle propagation and density queries are
        unsupported and raise ``NotImplementedError`` / ``RuntimeError``. Pair it with a
        generative model environment on the planner (``policy.environment``).

    Attributes:
        action_presets: Discrete ``(acceleration, steering_angle)`` control pairs.
        max_tracked_agents: Number of nearest agents carried in state/observation.
        seed: Optional seed applied to the first ``reset`` for reproducibility.

    Example:
        The environment is used as the forward-only world of an
        :class:`~POMDPPlanners.simulations.episodes.EpisodeRunner`, paired with a separate
        generative model on the planner. It requires the nuPlan devkit and a scenario
        loader, so this snippet is illustrative rather than executed::

            env = NuPlanPOMDP(discount_factor=0.95, scenario_loader=load_scenario)
            state = env.initial_state_dist().sample()[0]
            next_state, observation, reward = env.sample_next_step(state, 0)
    """

    def __init__(
        self,
        discount_factor: float,
        scenario_loader: Optional[Callable[[], Any]] = None,
        action_presets: Optional[Sequence[Tuple[float, float]]] = None,
        max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
        simulation_horizon: float = DEFAULT_SIMULATION_HORIZON,
        fixed_delta_seconds: float = 0.1,
        reactive_agents: bool = True,
        collision_distance: float = 2.0,
        collision_penalty: float = 100.0,
        desired_speed: float = 8.0,
        out_lane_thresh: float = 2.0,
        observation_extractor: Optional[Callable[[Dict[str, np.ndarray]], Any]] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
        reward_range: Optional[Tuple[float, float]] = None,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the nuPlan world environment.

        Args:
            discount_factor: Discount factor for future rewards (0 < d <= 1).
            scenario_loader: Zero-argument callable returning a nuPlan ``AbstractScenario``
                to simulate; called once per ``reset``. Required to run a live world; may be
                ``None`` when a fake session is injected for testing.
            action_presets: Discrete ``(acceleration, steering_angle)`` pairs. Defaults to
                :data:`DEFAULT_ACTION_PRESETS`.
            max_tracked_agents: Number of nearest other agents carried as fixed slots in the
                state and observation. Defaults to :data:`DEFAULT_MAX_TRACKED_AGENTS`.
            simulation_horizon: Seconds of constant-control trajectory rolled out and handed
                to the tracker each step. Defaults to :data:`DEFAULT_SIMULATION_HORIZON`.
            fixed_delta_seconds: Simulation iteration length in seconds. Defaults to 0.1.
            reactive_agents: If True, background agents react (IDM); otherwise they replay
                the log. Defaults to True.
            collision_distance: Centre-to-centre ego->agent distance (m) at/under which the
                step is treated as a terminal collision. Defaults to 2.0.
            collision_penalty: Reward penalty applied on a terminal collision. Defaults to
                100.0.
            desired_speed: Target longitudinal speed in m/s; exceeding it incurs the overspeed
                penalty. Defaults to 8.0.
            out_lane_thresh: Lateral offset in metres beyond which the ego is treated as
                off-route and penalised. Defaults to 2.0.
            observation_extractor: Optional callable applied to the full ``{ego, agents}``
                observation dict each step, returning the observation actually emitted. Must
                be a picklable (module-level) callable. Defaults to None (full dict).
            seed: Optional seed applied to the first ``reset``. Defaults to None.
            name: Environment identifier. Defaults to ``"NuPlanPOMDP"``.
            reward_range: Optional ``(min, max)`` reward bounds. Defaults to None.
            output_dir: Optional directory for logging output. Defaults to None.
            debug: Enable debug logging. Defaults to False.
            use_queue_logger: Whether to use queue-based logging. Defaults to False.
        """
        self.scenario_loader = scenario_loader
        presets = action_presets if action_presets is not None else DEFAULT_ACTION_PRESETS
        self.action_presets: List[Tuple[float, float]] = [
            (float(acceleration), float(steering_angle)) for acceleration, steering_angle in presets
        ]
        self.max_tracked_agents = max_tracked_agents
        self.simulation_horizon = simulation_horizon
        self.fixed_delta_seconds = fixed_delta_seconds
        self.reactive_agents = reactive_agents
        self.collision_distance = collision_distance
        self.collision_penalty = collision_penalty
        self.desired_speed = desired_speed
        self.out_lane_thresh = out_lane_thresh
        self.observation_extractor = observation_extractor
        self.seed = seed

        # Live-session state: rebuilt lazily and never serialized.
        self._session: Optional[Any] = None
        self._live_state: Optional[np.ndarray] = None
        self._latest_obs: Optional[Dict[str, np.ndarray]] = None
        self._terminated: bool = False
        self._seeded: bool = False
        self._pending: Optional[Dict[str, Any]] = None
        self._served_roles: Set[str] = set()

        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.CONTINUOUS,
        )
        super().__init__(
            discount_factor=discount_factor,
            name=name if name is not None else "NuPlanPOMDP",
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
            if self.scenario_loader is None:
                raise RuntimeError(
                    "NuPlanPOMDP has no scenario_loader; supply one returning a nuPlan "
                    "AbstractScenario, or inject a fake session for testing."
                )
            self._session = _NuPlanSession(
                scenario_loader=self.scenario_loader,
                max_tracked_agents=self.max_tracked_agents,
                simulation_horizon=self.simulation_horizon,
                fixed_delta_seconds=self.fixed_delta_seconds,
                reactive_agents=self.reactive_agents,
                collision_distance=self.collision_distance,
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
        observation = self._emit_observation(observation)
        self._live_state = state
        self._latest_obs = observation
        self._terminated = False
        self._pending = None
        self._served_roles = set()
        return state

    def _emit_observation(self, observation: Dict[str, np.ndarray]) -> Any:
        if self.observation_extractor is None:
            return observation
        return self.observation_extractor(observation)

    def _to_control(self, action: Any) -> Tuple[float, float]:
        return self.action_presets[int(action)]

    def _states_equal(self, state_a: Any, state_b: Any) -> bool:
        return np.array_equal(np.asarray(state_a), np.asarray(state_b))

    def _compute_reward(self, next_state: np.ndarray, action: Any, terminated: bool) -> float:
        _, steering_angle = self._to_control(action)
        return driving_quality_reward(
            next_state,
            steering_angle,
            terminated,
            self.desired_speed,
            self.out_lane_thresh,
            self.collision_penalty,
        )

    def _ensure_stepped(self, state: Any, action: Any, role: str) -> Dict[str, Any]:
        """Advance the world one iteration for ``(state, action)`` (once) and cache it.

        The reward and next-state getters of a single episode step share one iteration:
        the first advances the world and caches the outcome, the second is served from that
        cache. The cache is keyed on ``(state, action)`` *and* on the requesting ``role`` —
        each role is served from a given iteration at most once, so a repeated role means a
        new step and forces another iteration. Raises when asked to step from a state other
        than the live one.
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
                "NuPlanPOMDP is a forward-only world environment; it cannot resample from "
                "an arbitrary state. Give the planner a separate model environment "
                "(policy.environment) and only step the world forward from its live state."
            )

        session = self._get_session()
        acceleration, steering_angle = self._to_control(action)
        next_state, observation, terminated = session.step(acceleration, steering_angle)
        next_state = np.asarray(next_state)
        observation = self._emit_observation(observation)
        done = bool(terminated)
        pending = {
            "state": np.asarray(state).copy(),
            "action": action,
            "next_state": next_state,
            "observation": observation,
            "reward": self._compute_reward(next_state, action, done),
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
            raise ValueError("NuPlanPOMDP is forward-only and only supports n_samples=1")
        return self._ensure_stepped(state, action, _ROLE_NEXT_STATE)["next_state"]

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        del action
        if n_samples != 1:
            raise ValueError("NuPlanPOMDP is forward-only and only supports n_samples=1")
        if self._pending is not None and self._states_equal(
            self._pending["next_state"], next_state
        ):
            return self._pending["observation"]
        raise RuntimeError(
            "NuPlanPOMDP.sample_observation was queried for a next state other than the "
            "live one; a forward-only world only knows the reading it just produced by "
            "stepping."
        )

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        del next_state
        return self._ensure_stepped(state, action, _ROLE_REWARD)["reward"]

    def is_terminal(self, state: Any) -> bool:
        if self._live_state is not None and not self._states_equal(state, self._live_state):
            raise RuntimeError(
                "NuPlanPOMDP.is_terminal was queried for a state other than the live world "
                "state; this forward-only world only knows whether its current state is "
                "terminal."
            )
        return self._terminated

    def get_metric_names(self) -> List[str]:
        """Names of the nuPlan-specific evaluation metrics.

        Returns:
            The metric name strings produced by :meth:`compute_metrics`: ``collision_rate``,
            ``average_progress``, ``average_speed``, ``near_miss_count`` and
            ``min_vehicle_distance``.
        """
        return [metric.value for metric in NuPlanPOMDPMetrics]

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        """Compute nuPlan driving-quality metrics from episode histories.

        Args:
            histories: Episode histories to summarise.

        Returns:
            A list of :class:`MetricValue` with 95% confidence bounds across episodes:

            - ``collision_rate``: fraction of episodes that ended in a collision.
            - ``average_progress``: mean per-episode ground distance travelled (m).
            - ``average_speed``: mean ego speed over the driven trajectory (m/s).
            - ``near_miss_count``: mean number of near-miss events per episode.
            - ``min_vehicle_distance``: mean over episodes of the closest the ego came to any
              agent (m); episodes that saw no agent are excluded.
        """
        require_non_empty_histories(histories, type(self).__name__)
        collisions = [1.0 if history.reach_terminal_state else 0.0 for history in histories]
        path_lengths = [self._episode_path_length(h) for h in histories if h.history]
        mean_speeds = [self._episode_mean_speed(h) for h in histories if h.history]
        near = [self._episode_near_misses(h) for h in histories if h.history]
        near_counts = [float(count) for count, _min_distance in near]
        min_distances = [dist for _count, dist in near if np.isfinite(dist)]
        return [
            self._metric_from_samples(NuPlanPOMDPMetrics.COLLISION_RATE.value, collisions),
            self._metric_from_samples(NuPlanPOMDPMetrics.AVERAGE_PROGRESS.value, path_lengths),
            self._metric_from_samples(NuPlanPOMDPMetrics.AVERAGE_SPEED.value, mean_speeds),
            self._metric_from_samples(NuPlanPOMDPMetrics.NEAR_MISS_COUNT.value, near_counts),
            self._metric_from_samples(NuPlanPOMDPMetrics.MIN_VEHICLE_DISTANCE.value, min_distances),
        ]

    def _episode_near_misses(self, history: History) -> Tuple[int, float]:
        """Return ``(near_miss_events, min_ego_agent_distance)`` for one episode.

        A near-miss event is a contiguous run of steps whose closest other agent is within
        ``_NEAR_MISS_DISTANCE`` (centre-to-centre, ego frame). ``min_ego_agent_distance`` is
        the closest the ego came to any agent over the episode (``inf`` if it never saw one).
        States lacking agent slots contribute nothing.
        """
        agents_end = EGO_STATE_WIDTH + self.max_tracked_agents * AGENT_SLOT_WIDTH
        distances: List[float] = []
        for state in self._episode_states(history):
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

    @staticmethod
    def _episode_states(history: History) -> List[np.ndarray]:
        """Every state the episode actually visited, including the one it ended in.

        Acting states alone miss the final transition's endpoint — the very state a
        collision or closest approach happens in — so the last driven ``next_state`` is
        appended when the runner did not already close the episode with it.
        """
        states = [np.asarray(step.state, dtype=float) for step in history.history]
        driven = _driven_steps(history)
        if driven and driven[-1] is history.history[-1]:
            states.append(np.asarray(driven[-1].next_state, dtype=float))
        return states

    def _episode_path_length(self, history: History) -> float:
        total = 0.0
        for step in _driven_steps(history):
            start = np.asarray(step.state)[_EGO_POSITION_SLICE]
            end = np.asarray(step.next_state)[_EGO_POSITION_SLICE]
            total += float(np.linalg.norm(end - start))
        return total

    def _episode_mean_speed(self, history: History) -> float:
        speeds = [
            float(np.linalg.norm(np.asarray(step.next_state)[_EGO_VELOCITY_SLICE]))
            for step in _driven_steps(history)
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
                # The emitted observation is the ``{ego, agents}`` dict unless an
                # observation_extractor reshaped it (e.g. a flattened vector); copy a dict
                # defensively and pass any other emitted type through unchanged.
                if isinstance(observation, dict):
                    return [dict(observation) for _ in range(n_samples)]
                return [observation for _ in range(n_samples)]

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
        return np.asarray(observation).tobytes()

    def hash_action(self, action: Any) -> Hashable:
        if isinstance(action, np.ndarray):
            return action.tobytes()
        return action

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        del state, action, next_states
        raise NotImplementedError(
            "NuPlanPOMDP is a forward-only world environment with no transition density. "
            "Belief updates must run on the planner's model environment."
        )

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        del next_state, action, observations
        raise NotImplementedError(
            "NuPlanPOMDP is a forward-only world environment with no observation density. "
            "Belief updates must run on the planner's model environment."
        )
