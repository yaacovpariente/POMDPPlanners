# SPDX-License-Identifier: MIT

"""Reference concrete CARLA generative model with a fixed factored observation model.

:class:`FactoredCarlaModelPOMDP` implements the
:class:`~POMDPPlanners.environments.carla_pomdp.carla_model_pomdp.CarlaModelPOMDP`
interface with a fixed, factored observation model (per-slot agent detection with range +
occlusion gating and additive Gaussian pose noise, plus a Gaussian GNSS reading) and the
same gym-carla driving-quality reward the world uses. The transition dynamics are a
documented identity placeholder for a specific study to replace with real (e.g. learned)
motion.

State/observation layout, geometry helpers, detection constants, and the shared reward are
imported from :mod:`~POMDPPlanners.environments.carla_pomdp.carla_pomdp` so world and model
agree by construction.

Classes:
    FactoredCarlaModelPOMDP: Concrete CARLA model with a factored observation model.
"""

from typing import Any, Optional, Sequence, Tuple

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.environments.carla_pomdp.carla_model_pomdp import CarlaModelPOMDP
from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
    AGENT_SLOT_WIDTH,
    DEFAULT_MAX_TRACKED_AGENTS,
    DEFAULT_OCCLUSION_RADIUS,
    DEFAULT_PERCEPTION_RANGE,
    _segment_occludes,
    driving_quality_reward,
)

_LOG_EPS = -50.0  # log-prob floor for impossible / near-zero events


class FactoredCarlaModelPOMDP(CarlaModelPOMDP):
    """Concrete CARLA generative model with a fixed factored observation model.

    The observation model is factored per agent slot (detection gated by perception range
    and geometric occlusion, additive Gaussian pose noise) plus a Gaussian GNSS reading;
    the reward is the shared gym-carla driving-quality reward. The transition is a
    documented identity placeholder to be replaced with real dynamics per study.

    Attributes:
        perception_range: Metres beyond which an agent is undetectable.
        occlusion_radius: Sight-line blocking radius among agents.
        pose_std: Std of Gaussian noise on a detected agent's ``[rel_x, rel_y, rel_yaw,
            rel_speed]`` measurement.
        gnss_std: Std of Gaussian noise on the ``gnss`` reading.
        detect_prob: Probability of detecting a visible (in-range, un-occluded) agent;
            ``1 - detect_prob`` is the miss rate.
        desired_speed: Target longitudinal speed (m/s) used by the reward.
        out_lane_thresh: Lateral offset (m) beyond which the reward penalises out-of-lane.
        collision_penalty: Penalty scale applied on a terminal collision in the reward.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
        ...     AGENT_SLOT_WIDTH, EGO_STATE_WIDTH)
        >>> env = FactoredCarlaModelPOMDP(discount_factor=0.95)
        >>>
        >>> width = EGO_STATE_WIDTH + env.max_tracked_agents * AGENT_SLOT_WIDTH
        >>> state = np.zeros(width)
        >>> action = env.get_actions()[0]
        >>>
        >>> next_state, observation, reward = env.sample_next_step(state, action)
        >>> sorted(observation)
        ['agents', 'gnss']
        >>> env.is_terminal(state)
        False
    """

    def __init__(
        self,
        discount_factor: float,
        action_presets: Optional[Sequence[Tuple[float, float, float]]] = None,
        max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
        perception_range: float = DEFAULT_PERCEPTION_RANGE,
        occlusion_radius: float = DEFAULT_OCCLUSION_RADIUS,
        pose_std: float = 0.5,
        gnss_std: float = 1e-5,
        detect_prob: float = 0.95,
        desired_speed: float = 8.0,
        out_lane_thresh: float = 2.0,
        collision_penalty: float = 100.0,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the factored CARLA generative model.

        Args:
            discount_factor: Discount factor for future rewards (0 < d <= 1).
            action_presets: Discrete ``(throttle, steer, brake)`` triples. Defaults to
                the world's default presets.
            max_tracked_agents: Number of fixed agent slots in state/observation.
            perception_range: Metres beyond which an agent is undetectable.
            occlusion_radius: Sight-line blocking radius among agents.
            pose_std: Std of Gaussian noise on a detected agent's pose measurement.
            gnss_std: Std of Gaussian noise on the ``gnss`` reading.
            detect_prob: Probability of detecting a visible agent.
            desired_speed: Target longitudinal speed (m/s) for the reward.
            out_lane_thresh: Lateral offset (m) beyond which the reward penalises.
            collision_penalty: Penalty scale for a terminal collision in the reward.
            name: Environment identifier. Defaults to the class name.
        """
        self.perception_range = perception_range
        self.occlusion_radius = occlusion_radius
        self.pose_std = pose_std
        self.gnss_std = gnss_std
        self.detect_prob = detect_prob
        self.desired_speed = desired_speed
        self.out_lane_thresh = out_lane_thresh
        self.collision_penalty = collision_penalty
        super().__init__(
            discount_factor=discount_factor,
            action_presets=action_presets,
            max_tracked_agents=max_tracked_agents,
            name=name,
        )

    # ── Transition (identity placeholder — replace per study) ────────────
    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        # Placeholder identity dynamics: a study replaces this with ego-kinematics
        # propagation under the control preset plus an agent-slot motion model.
        del action
        state = np.asarray(state, dtype=float)
        if n_samples == 1:
            return state
        return np.stack([state for _ in range(n_samples)])

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        # Placeholder: implement the density matching sample_next_state's motion model.
        del state, action, next_states
        raise NotImplementedError("FactoredCarlaModelPOMDP transition density not yet implemented.")

    # ── Observation model (fixed, factored) ─────────────────────────────
    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        del action
        obs = self._render_observation(np.asarray(next_state, dtype=float), noisy=True)
        return [obs for _ in range(n_samples)] if n_samples != 1 else obs

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        del action
        state = np.asarray(next_state, dtype=float)
        obs_list = observations if isinstance(observations, list) else [observations]
        return np.array([self._log_prob_single(state, obs) for obs in obs_list])

    # ── Reward (shared with the world by construction) ───────────────────
    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        resulting = np.asarray(next_state if next_state is not None else state, dtype=float)
        _, steer, _ = self.action_presets[int(action)]
        # The model cannot observe collisions, so the terminal collision term is
        # necessarily approximated (is_terminal is always False for the model).
        return driving_quality_reward(
            resulting,
            steer,
            self.is_terminal(resulting),
            self.desired_speed,
            self.out_lane_thresh,
            self.collision_penalty,
        )

    # ── Terminal / initial hooks ─────────────────────────────────────────
    def is_terminal(self, state: Any) -> bool:
        del state
        return False

    def initial_state_dist(self) -> Distribution:
        raise NotImplementedError("Seed the belief from the world's initial observation.")

    def initial_observation_dist(self) -> Distribution:
        raise NotImplementedError("Seed the belief from the world's initial observation.")

    # ── Factored observation helpers ─────────────────────────────────────
    def _visible(self, rows: np.ndarray, slot: int) -> bool:
        """Whether true agent ``slot`` is in range and not occluded (ego-frame)."""
        rel_x, rel_y = rows[slot, 1], rows[slot, 2]
        if float(np.hypot(rel_x, rel_y)) > self.perception_range:
            return False
        for other in range(self.max_tracked_agents):
            if other == slot or rows[other, 0] != 1.0:
                continue
            if _segment_occludes(
                0.0, 0.0, rel_x, rel_y, rows[other, 1], rows[other, 2], self.occlusion_radius
            ):
                return False
        return True

    def _render_observation(self, state: np.ndarray, noisy: bool) -> dict:
        rows = self._state_agent_rows(state).copy()
        for slot in range(self.max_tracked_agents):
            if rows[slot, 0] != 1.0 or not self._visible(rows, slot):
                rows[slot] = 0.0
            elif noisy:
                rows[slot, 1:] += np.random.normal(0.0, self.pose_std, size=AGENT_SLOT_WIDTH - 1)
        gnss = state[:2].copy()
        if noisy:
            gnss = gnss + np.random.normal(0.0, self.gnss_std, size=2)
        return {"gnss": gnss, "agents": rows.reshape(-1)}

    def _log_prob_single(self, state: np.ndarray, obs: dict) -> float:
        rows = self._state_agent_rows(state)
        obs_rows = np.asarray(obs["agents"], dtype=float).reshape(
            self.max_tracked_agents, AGENT_SLOT_WIDTH
        )
        total = self._gnss_log_prob(state, obs)
        for slot in range(self.max_tracked_agents):
            total += self._slot_log_prob(rows, obs_rows, slot)
        return float(total)

    def _gnss_log_prob(self, state: np.ndarray, obs: dict) -> float:
        gnss = np.asarray(obs["gnss"], dtype=float)[:2]
        diff = gnss - state[:2]
        return float(-0.5 * np.sum((diff / self.gnss_std) ** 2) - 2 * np.log(self.gnss_std))

    def _slot_log_prob(self, rows: np.ndarray, obs_rows: np.ndarray, slot: int) -> float:
        true_present = rows[slot, 0] == 1.0
        obs_present = obs_rows[slot, 0] == 1.0
        if not true_present:
            return 0.0 if not obs_present else _LOG_EPS  # empty state slot -> empty obs slot
        if not self._visible(rows, slot):
            return 0.0 if not obs_present else _LOG_EPS  # invisible -> must be missed
        if not obs_present:
            return float(np.log(max(1.0 - self.detect_prob, np.exp(_LOG_EPS))))  # a miss
        diff = obs_rows[slot, 1:] - rows[slot, 1:]
        gauss = -0.5 * np.sum((diff / self.pose_std) ** 2) - (AGENT_SLOT_WIDTH - 1) * np.log(
            self.pose_std
        )
        return float(np.log(self.detect_prob) + gauss)
