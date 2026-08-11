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
swaps only the observation block of the simulator config; the track, the opponent, the
dynamics, the step rates and the reward are shared. See
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
- ``ObservationMode.POMDP``: a ``(2, 12, 12)`` float32 occupancy grid of presence and
  on-road flags over a +/-18 m window at 3 m resolution, aligned to the ego's own axes.
  Every velocity, every vehicle identity, and everything outside the window is withheld.

Note:
    The ego is written into the occupancy grid, always at the centre cell. Anything
    reading the grid must account for that; the belief's tracker drops it explicitly.

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
    >>> observation.shape
    (2, 12, 12)
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
    lane_curvature,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_PRESENT,
    AGENT_REL_X,
    AGENT_SLOT_WIDTH,
    DEFAULT_ACTION_PRESETS,
    DEFAULT_ACTION_REWARD,
    DEFAULT_COLLISION_REWARD,
    DEFAULT_DURATION,
    DEFAULT_ENV_ID,
    DEFAULT_LANE_CENTERING_COST,
    DEFAULT_LANE_CENTERING_REWARD,
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
    build_racetrack_config,
    racetrack_reward,
    rotate,
    state_agent_rows,
    wrap_to_pi,
)

_ROLE_NEXT_STATE = "next_state"
_ROLE_REWARD = "reward"


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
        self._lane_offsets: Dict[Any, float] = {}

    def render_frame(self) -> Optional[np.ndarray]:
        """Return the current bird's-eye frame, or None if not rendering."""
        frame = self._env.render()
        return None if frame is None else np.asarray(frame)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        observation = self._env.reset(seed=seed)[0]
        return self._read_state(), np.asarray(observation)

    def step(self, acceleration: float, steering: float) -> Dict[str, Any]:
        command = np.array([acceleration, steering], dtype=np.float32)
        step_result = self._env.step(command)
        observation, truncated = step_result[0], step_result[3]
        vehicle = self._ego()
        return {
            "state": self._read_state(),
            "observation": np.asarray(observation),
            "crashed": bool(vehicle.crashed),
            "off_road": not bool(vehicle.on_road),
            "truncated": bool(truncated),
        }

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
            self._lane_offsets = self._build_lane_offsets(lane_index)
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
        rows = np.zeros((self._max_tracked_agents, AGENT_SLOT_WIDTH), dtype=float)
        others = [v for v in self._env.unwrapped.road.vehicles if v is not ego]
        ranked = sorted(others, key=lambda v: float(np.linalg.norm(v.position - ego.position)))
        heading = float(ego.heading)
        for slot, other in enumerate(ranked[: self._max_tracked_agents]):
            position = rotate(np.asarray(other.position, dtype=float) - ego.position, -heading)
            velocity = rotate(np.asarray(other.velocity, dtype=float) - ego.velocity, -heading)
            rows[slot] = [1.0, position[0], position[1], velocity[0], velocity[1]]
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
        self._latest_obs: Optional[np.ndarray] = None
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
                return [np.array(observation, copy=True) for _ in range(n_samples)]

        return InitialObservation()

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return np.array_equal(np.asarray(observation1), np.asarray(observation2))

    def hash_observation(self, observation: Any) -> Hashable:
        return np.asarray(observation).tobytes()

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
