# SPDX-License-Identifier: MIT

"""Shared schema for the racetrack POMDP: state layout, config, and reward.

This module is the single source of truth shared by the three racetrack pieces — the
forward-only world, the planner-side generative model, and the belief. It deliberately
imports **nothing** from ``highway_env``, so the model and the belief stay pure NumPy and
can be constructed, tested and pickled on a machine where the simulator is not installed.
Only :mod:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp` touches the
backend.

**The matched pair.** :func:`build_racetrack_config` assembles the whole highway-env
configuration once and attaches exactly one of two ``"observation"`` blocks. Every
dynamics key — the action type, the step rates, the reward weights, the vehicle counts —
comes off the same code path before the observation is chosen, so the MDP baseline and
the POMDP differ in what the agent *sees* and in nothing else. That property is what lets
a planner's performance gap be attributed to partial observability, so it is asserted by
a test rather than trusted.

**Longitudinal control is enabled**, unlike the racetrack defaults. Under the shipped
configuration ``ContinuousAction`` is lateral-only: acceleration is pinned at zero and the
``target_speeds`` key is inert (it belongs to ``DiscreteMetaAction``). The ego could then
never brake for an opponent, which removes most of what partial observability costs. The
flag is a dynamics key applied identically to both arms, so the matched pair is preserved.

**State layout** (identical in the world and in the model, on purpose — a wider world
state is what makes CARLA's agent-slot reshape unsafe against a world vector)::

    [x, y, heading, speed, lat, ang, s] + max_tracked_agents * [present, rel_x, rel_y, rel_vx, rel_vy]

The ego block is world-frame position in metres, heading in radians, scalar speed in m/s,
then the Frenet terms: signed lateral offset from the lane centreline in metres, the angle
between the heading and the lane direction in radians, and the distance travelled along the
track centreline in metres.

The last slot is deliberately the car's **arclength**, not the road's curvature. Curvature is
a property of the road, so freezing it in the state encodes a prediction about the future
rather than a fact about the present, and a rollout that reuses it drives straight through
every corner. Where the road bends is the transition model's business: see
:mod:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry`.

Agent slots are in the ego body frame — ``rel_x`` forward, ``rel_y`` left — and hold
**relative** velocity, because that is exactly what differencing two ego-aligned
occupancy grids can measure.

Classes:
    ObservationMode: Which observation the world emits; dynamics are unaffected.
"""

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ── State layout ────────────────────────────────────────────────────────
EGO_STATE_WIDTH = 7
AGENT_SLOT_WIDTH = 5
DEFAULT_MAX_TRACKED_AGENTS = 4

EGO_X = 0
EGO_Y = 1
EGO_HEADING = 2
EGO_SPEED = 3
EGO_LAT = 4
EGO_ANG = 5
EGO_ARCLENGTH_M = 6

AGENT_PRESENT = 0
AGENT_REL_X = 1
AGENT_REL_Y = 2
AGENT_REL_VX = 3
AGENT_REL_VY = 4

# ── Occupancy grid geometry ─────────────────────────────────────────────
# Verified against highway-env 1.12.1: the observation is (len(features), 12, 12)
# float32, axis 0 is the along-track index and axis 1 the across-track index, and
# the ego is always written into the centre cell.
GRID_HALF_EXTENT_M = 18.0
GRID_STEP_M = 3.0
GRID_CELLS = 12
PRESENCE_LAYER = 0
ON_ROAD_LAYER = 1

# ── Action presets ──────────────────────────────────────────────────────
# highway-env maps a normalised action in [-1, 1] onto these ranges.
MAX_ACCELERATION_MPS2 = 5.0
MAX_STEERING_RAD = np.pi / 4

# Every acceleration crossed with every steering angle, both normalised to [-1, 1].
# The planner selects an index into this tuple, so the world and the model share
# one action vocabulary by construction rather than by convention.
# Steering is sampled finely near zero and coarsely at the extremes. Full lock is
# pi/4 = 45 degrees, which on this track is a spin rather than a correction: sweeping
# constant steering through the first bend, -1.0 survives 5 steps while -0.05 survives
# 29. A bang-bang set of {-1, 0, +1} simply does not contain the manoeuvre the track
# needs, so no amount of planning can select it.
STEERING_PRESETS: Tuple[float, ...] = (-1.0, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 1.0)
ACCELERATION_PRESETS: Tuple[float, ...] = (1.0, 0.0, -1.0)

DEFAULT_ACTION_PRESETS: Tuple[Tuple[float, float], ...] = tuple(
    (acceleration, steering)
    for acceleration in ACCELERATION_PRESETS
    for steering in STEERING_PRESETS
)

# ── Simulator defaults (racetrack-v0, verified against highway-env 1.12.1) ──
DEFAULT_ENV_ID = "racetrack-v0"
DEFAULT_DURATION = 300
DEFAULT_POLICY_FREQUENCY = 5
DEFAULT_SIMULATION_FREQUENCY = 15
DEFAULT_OTHER_VEHICLES = 1
DEFAULT_SPEED_LIMIT = 10.0
DEFAULT_COLLISION_REWARD = -1.0
DEFAULT_LANE_CENTERING_COST = 4.0
DEFAULT_LANE_CENTERING_REWARD = 1.0
DEFAULT_ACTION_REWARD = -0.3
DEFAULT_NEAR_MISS_DISTANCE_M = 5.0

_CONFIG_OBSERVATION_KEY = "observation"


class ObservationMode(Enum):
    """Which observation the racetrack world emits.

    The two modes share one dynamics path and differ only in the observation, so a
    planner's performance gap between them measures partial observability alone.

    Attributes:
        MDP: Absolute position and velocity for the ego and nearby vehicles. Only the
            other vehicles' driver policy stays hidden, so this is a near-MDP baseline
            rather than a true MDP.
        POMDP: A local occupancy grid of presence and on-road flags. Every velocity,
            every vehicle identity, and everything beyond the grid window is withheld.
    """

    MDP = "mdp"
    POMDP = "pomdp"


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians to ``[-pi, pi)``.

    Args:
        angle: Angle in radians.

    Returns:
        The equivalent angle in ``[-pi, pi)``.
    """
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def state_agent_rows(state: np.ndarray, max_tracked_agents: int) -> np.ndarray:
    """View a state vector's agent slots as a ``(max_tracked_agents, 5)`` block.

    Args:
        state: A state vector of width ``EGO_STATE_WIDTH + max_tracked_agents * 5``.
        max_tracked_agents: Number of fixed agent slots in the state.

    Returns:
        The agent slots reshaped to one row per slot. This is a reshape of a slice, so
        it may share memory with ``state``; copy it before mutating.

    Raises:
        ValueError: If the state width does not match ``max_tracked_agents``.
    """
    array = np.asarray(state, dtype=float)
    expected = EGO_STATE_WIDTH + max_tracked_agents * AGENT_SLOT_WIDTH
    if array.shape[-1] != expected:
        raise ValueError(
            f"State width {array.shape[-1]} does not match max_tracked_agents="
            f"{max_tracked_agents} (expected {expected})."
        )
    return array[..., EGO_STATE_WIDTH:].reshape(
        *array.shape[:-1], max_tracked_agents, AGENT_SLOT_WIDTH
    )


def rotate(vectors: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 2-D row vectors counter-clockwise by ``angle`` radians.

    Args:
        vectors: Array whose last axis has length 2.
        angle: Rotation angle in radians.

    Returns:
        The rotated vectors, same shape as the input.
    """
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)
    return np.asarray(vectors, dtype=float) @ matrix.T


def racetrack_reward(
    lateral: float,
    action: Tuple[float, float],
    crashed: bool,
    on_road: bool,
    *,
    collision_reward: float = DEFAULT_COLLISION_REWARD,
    lane_centering_cost: float = DEFAULT_LANE_CENTERING_COST,
    lane_centering_reward: float = DEFAULT_LANE_CENTERING_REWARD,
    action_reward: float = DEFAULT_ACTION_REWARD,
) -> float:
    """Score a racetrack transition, reproducing highway-env's own reward.

    Both the world and the planner's model call this one function, so the planner can
    never be optimising a different objective than the one being scored. It reproduces
    ``RacetrackEnv._reward`` in closed form; a test pins the two together against a live
    simulator, which is what makes that a fact rather than an intention.

    Args:
        lateral: Signed lateral offset from the lane centreline, in metres.
        action: The normalised ``(acceleration, steering)`` command actually applied.
        crashed: Whether the ego collided on this step.
        on_road: Whether the ego is still on the road.
        collision_reward: Weight applied to a collision. Defaults to -1.0.
        lane_centering_cost: Sharpness of the lane-centering falloff. Defaults to 4.0.
        lane_centering_reward: Weight on the lane-centering term. Defaults to 1.0.
        action_reward: Weight on the control-effort penalty. Defaults to -0.3.

    Returns:
        The scalar reward, zero whenever the ego is off the road.

    Note:
        Two details of the upstream formula are reproduced rather than tidied: the
        normalisation maps from ``[collision_reward, 1]`` using the literal ``1`` and not
        ``lane_centering_reward``, and it does not clip, so unusual weights can push the
        result outside ``[0, 1]``.
    """
    centering = lane_centering_reward / (1.0 + lane_centering_cost * float(lateral) ** 2)
    effort = float(np.linalg.norm(np.asarray(action, dtype=float)))
    raw = centering + action_reward * effort + collision_reward * float(crashed)
    scaled = (raw - collision_reward) / (1.0 - collision_reward)
    return float(scaled * float(on_road))


def _pomdp_observation_config() -> Dict[str, Any]:
    """The shipped racetrack occupancy-grid block: presence and on-road, 12x12 at 3 m."""
    half = GRID_HALF_EXTENT_M
    step = GRID_STEP_M
    return {
        "type": "OccupancyGrid",
        "features": ["presence", "on_road"],
        "grid_size": [[-half, half], [-half, half]],
        "grid_step": [step, step],
        "as_image": False,
        "align_to_vehicle_axes": True,
    }


def _mdp_observation_config(max_tracked_agents: int) -> Dict[str, Any]:
    """Absolute kinematics for the ego and the nearest ``max_tracked_agents`` vehicles.

    ``absolute`` and ``normalize`` are set explicitly because highway-env defaults to
    relative, normalised rows, which is not the "absolute x, y, vx, vy" baseline this
    comparison needs. ``order`` is pinned to ``"sorted"`` for a second reason beyond
    determinism: ``"shuffled"`` draws from the environment's random generator, which
    would make the two arms consume different randomness and break the shared-dynamics
    guarantee.
    """
    return {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy"],
        "vehicles_count": max_tracked_agents + 1,
        "absolute": True,
        "normalize": False,
        "order": "sorted",
        "see_behind": True,
    }


def observation_config(mode: ObservationMode, max_tracked_agents: int) -> Dict[str, Any]:
    """Build the highway-env observation block for one arm of the matched pair.

    Args:
        mode: Which arm to build.
        max_tracked_agents: Number of other vehicles the MDP arm reports.

    Returns:
        The ``"observation"`` sub-dictionary for a highway-env config.
    """
    if mode is ObservationMode.POMDP:
        return _pomdp_observation_config()
    return _mdp_observation_config(max_tracked_agents)


def build_racetrack_config(
    mode: ObservationMode,
    *,
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
    terminate_off_road: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the highway-env configuration for one arm of the matched pair.

    Every key except ``"observation"`` is built from the same arguments regardless of
    ``mode``, which is what makes the two arms a controlled comparison. ``overrides`` is
    applied last and is deliberately shared by both arms too — an override that differed
    between them would silently break the guarantee.

    Args:
        mode: Which observation the environment should emit.
        max_tracked_agents: Fixed number of agent slots in the state and the MDP
            observation. Defaults to 4.
        other_vehicles: Extra vehicles beyond the one the racetrack always spawns.
            Defaults to 1.
        duration: Episode length limit in simulator time. Defaults to 300.
        policy_frequency: Decisions per second. Defaults to 5.
        simulation_frequency: Physics steps per second. Defaults to 15.
        collision_reward: Weight applied to a collision. Defaults to -1.0.
        lane_centering_cost: Sharpness of the lane-centering falloff. Defaults to 4.0.
        lane_centering_reward: Weight on the lane-centering term. Defaults to 1.0.
        action_reward: Weight on the control-effort penalty. Defaults to -0.3.
        speed_limit: Track speed limit in m/s. Defaults to 10.0.
        terminate_off_road: Whether leaving the road ends the episode. Defaults to True.
        overrides: Extra highway-env keys merged in last. Defaults to None.

    Returns:
        A configuration dictionary ready for ``gymnasium.make(..., config=...)``.

    Raises:
        ValueError: If ``overrides`` tries to set the observation, which would defeat the
            matched pair, or if the substep ratio is not a positive integer.
    """
    _require_integral_substeps(simulation_frequency, policy_frequency)
    config: Dict[str, Any] = {
        # Longitudinal control is enabled on purpose; see the module docstring.
        "action": {"type": "ContinuousAction", "longitudinal": True, "lateral": True},
        "simulation_frequency": simulation_frequency,
        "policy_frequency": policy_frequency,
        "duration": duration,
        "collision_reward": collision_reward,
        "lane_centering_cost": lane_centering_cost,
        "lane_centering_reward": lane_centering_reward,
        "action_reward": action_reward,
        "controlled_vehicles": 1,
        "other_vehicles": other_vehicles,
        "speed_limit": speed_limit,
        "terminate_off_road": terminate_off_road,
    }
    if overrides:
        if _CONFIG_OBSERVATION_KEY in overrides:
            raise ValueError(
                "Overriding 'observation' would break the matched MDP/POMDP pair, whose "
                "whole purpose is that the two arms differ in that key alone. Select an "
                "arm with ObservationMode instead."
            )
        config.update(overrides)
    config[_CONFIG_OBSERVATION_KEY] = observation_config(mode, max_tracked_agents)
    return config


def _require_integral_substeps(simulation_frequency: int, policy_frequency: int) -> None:
    """Reject a step-rate pair the planner's model could not reproduce exactly."""
    if policy_frequency <= 0 or simulation_frequency <= 0:
        raise ValueError(
            f"simulation_frequency and policy_frequency must be positive, got "
            f"{simulation_frequency} and {policy_frequency}."
        )
    if simulation_frequency % policy_frequency != 0:
        raise ValueError(
            f"simulation_frequency ({simulation_frequency}) must be an integer multiple "
            f"of policy_frequency ({policy_frequency}); otherwise the planner's model "
            f"integrates a different number of substeps than the world and the two "
            f"diverge silently."
        )
