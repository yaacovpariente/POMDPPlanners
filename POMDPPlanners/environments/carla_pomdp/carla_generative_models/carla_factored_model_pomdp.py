# SPDX-License-Identifier: MIT

"""Reference concrete CARLA generative model pairing dynamics with factored perception.

:class:`FactoredCarlaModelPOMDP` implements the
:class:`~POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_model_pomdp.CarlaModelPOMDP`
interface by composing per-channel observation models — a
:class:`~POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models.agent_models.FactoredAgentObservationModel`
on the ``agents`` channel (per-slot detection with range + occlusion gating and additive Gaussian
pose noise) and a
:class:`~POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models.gnss_models.GnssObservationModel`
on the ``gnss`` channel — the observation methods themselves are inherited from the base and
driven by that map, together with the same gym-carla driving-quality reward the world uses. The
transition dynamics are a documented identity placeholder for a specific study to replace with
real (e.g. learned) motion.

State/observation layout and the shared reward are imported from
:mod:`~POMDPPlanners.environments.carla_pomdp.carla_pomdp` so world and model agree by
construction.

Classes:
    FactoredCarlaModelPOMDP: Concrete CARLA model with a factored-perception observation model.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_model_pomdp import (
    CarlaModelPOMDP,
)
from POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models import (
    build_observation_model,
)
from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
    DEFAULT_MAX_TRACKED_AGENTS,
    DEFAULT_OCCLUSION_RADIUS,
    DEFAULT_PERCEPTION_RANGE,
    driving_quality_reward,
)


class FactoredCarlaModelPOMDP(CarlaModelPOMDP):
    """Concrete CARLA generative model pairing placeholder dynamics with factored perception.

    The observation is composed per channel (held as ``self.observation_models``): a
    :class:`~POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models.agent_models.FactoredAgentObservationModel`
    on ``agents`` (detection gated by perception range and geometric occlusion, additive Gaussian
    pose noise) and a
    :class:`~POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models.gnss_models.GnssObservationModel`
    on ``gnss``. The reward is the shared gym-carla driving-quality reward. The transition is a
    documented identity placeholder to be replaced with real dynamics per study.

    Attributes:
        observation_models: The per-channel ``{channel: CarlaObservationModel}`` map carrying the
            observation parameters (``perception_range``, ``occlusion_radius``, ``pose_std``,
            ``gnss_std``, ``detect_prob``).
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
        perception_range: Optional[float] = DEFAULT_PERCEPTION_RANGE,
        occlusion_radius: float = DEFAULT_OCCLUSION_RADIUS,
        pose_std: float = 0.5,
        gnss_std: float = 1e-5,
        detect_prob: float = 0.95,
        desired_speed: float = 8.0,
        out_lane_thresh: float = 2.0,
        collision_penalty: float = 100.0,
        observation: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the factored CARLA generative model.

        Args:
            discount_factor: Discount factor for future rewards (0 < d <= 1).
            action_presets: Discrete ``(throttle, steer, brake)`` triples. Defaults to
                the world's default presets.
            max_tracked_agents: Number of fixed agent slots in state/observation.
            perception_range: Metres beyond which an agent is undetectable, or ``None`` to
                disable the range gate (agents stay visible at any distance).
            occlusion_radius: Sight-line blocking radius among agents.
            pose_std: Std of Gaussian noise on a detected agent's pose measurement.
            gnss_std: Std of Gaussian noise on the ``gnss`` reading.
            detect_prob: Probability of detecting a visible agent.
            desired_speed: Target longitudinal speed (m/s) for the reward.
            out_lane_thresh: Lateral offset (m) beyond which the reward penalises.
            collision_penalty: Penalty scale for a terminal collision in the reward.
            observation: Per-channel model selection ``{channel: registered_name}`` resolved via
                the observation-model registry. Defaults to
                ``{"gnss": "gaussian", "agents": "factored"}``.
            name: Environment identifier. Defaults to the class name.
        """
        self.desired_speed = desired_speed
        self.out_lane_thresh = out_lane_thresh
        self.collision_penalty = collision_penalty
        selection = (
            observation if observation is not None else {"gnss": "gaussian", "agents": "factored"}
        )
        channel_kwargs: Dict[str, Dict[str, Any]] = {
            "gnss": {"gnss_std": gnss_std},
            "agents": {
                "max_tracked_agents": max_tracked_agents,
                "perception_range": perception_range,
                "occlusion_radius": occlusion_radius,
                "pose_std": pose_std,
                "detect_prob": detect_prob,
            },
        }
        observation_models = {
            channel: build_observation_model(channel, model_name, **channel_kwargs.get(channel, {}))
            for channel, model_name in selection.items()
        }
        super().__init__(
            discount_factor=discount_factor,
            action_presets=action_presets,
            max_tracked_agents=max_tracked_agents,
            observation_models=observation_models,
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
