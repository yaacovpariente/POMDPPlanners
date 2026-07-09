# SPDX-License-Identifier: MIT

"""Agent-channel observation models.

Catalog of per-channel models for the ``agents`` observation channel (the fixed agent-slot
block). Add new agent-perception models here and register them with
:func:`register_observation_model` so they can be selected by name.

Classes:
    FactoredAgentObservationModel: Per-slot detection + occlusion gating + Gaussian pose noise.
"""

from typing import Any, Optional

import numpy as np

from POMDPPlanners.environments.nuplan_pomdp.nuplan_pomdp import (
    AGENT_SLOT_WIDTH,
    DEFAULT_MAX_TRACKED_AGENTS,
    DEFAULT_OCCLUSION_RADIUS,
    DEFAULT_PERCEPTION_RANGE,
    _segment_occludes,
)
from POMDPPlanners.environments.nuplan_pomdp.nuplan_perception.observation_model import (
    NuPlanObservationModel,
)
from POMDPPlanners.environments.nuplan_pomdp.nuplan_perception.observation_models.registry import (
    register_observation_model,
)

_LOG_EPS = -50.0  # log-prob floor for impossible / near-zero events


@register_observation_model("agents", "factored")
class FactoredAgentObservationModel(NuPlanObservationModel):
    """Reference agent perception: per-slot detection gating plus additive Gaussian pose noise.

    Each agent slot is detected only when in perception range and not geometrically occluded by
    another agent on the ego->target sight line; a detected agent's pose is corrupted with
    additive Gaussian noise. Provides both a sampler and a matching density, so it can back a
    scoring generative model.

    Attributes:
        max_tracked_agents: Number of fixed agent slots in the ``agents`` block.
        perception_range: Metres beyond which an agent is undetectable (``None`` disables the
            range gate).
        occlusion_radius: Sight-line blocking radius among agents.
        pose_std: Std of Gaussian noise on a detected agent's pose measurement.
        detect_prob: Probability of detecting a visible agent; ``1 - detect_prob`` is the miss
            rate scored by the density.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> model = FactoredAgentObservationModel(max_tracked_agents=1, perception_range=50.0)
        >>> agents = np.array([1.0, 10.0, 0.0, 0.0, 0.0])
        >>> float(model.perceive(agents)[0])  # near agent detected
        1.0
    """

    channel = "agents"
    supports_density = True

    def __init__(
        self,
        max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
        perception_range: Optional[float] = DEFAULT_PERCEPTION_RANGE,
        occlusion_radius: float = DEFAULT_OCCLUSION_RADIUS,
        pose_std: float = 0.5,
        detect_prob: float = 0.95,
    ) -> None:
        """Initialize the factored agent perception.

        Args:
            max_tracked_agents: Number of fixed agent slots in the ``agents`` block.
            perception_range: Metres beyond which an agent is undetectable, or ``None``.
            occlusion_radius: Sight-line blocking radius among agents.
            pose_std: Std of Gaussian noise on a detected agent's pose measurement.
            detect_prob: Probability of detecting a visible agent.
        """
        self.max_tracked_agents = max_tracked_agents
        self.perception_range = perception_range
        self.occlusion_radius = occlusion_radius
        self.pose_std = pose_std
        self.detect_prob = detect_prob

    def perceive(self, clean_channel: Any) -> np.ndarray:
        return self.render(clean_channel, noisy=True)

    def render(self, clean_channel: Any, noisy: bool) -> np.ndarray:
        """Gate the clean agent block per slot, optionally adding Gaussian pose noise.

        Args:
            clean_channel: The noise-free flat ``agents`` block.
            noisy: When True, add pose Gaussian noise (the sampler path); when False, return the
                gated but noise-free block.

        Returns:
            The perceived flat ``agents`` block.
        """
        rows = self._agent_rows(clean_channel).copy()
        for slot in range(self.max_tracked_agents):
            if rows[slot, 0] != 1.0 or not self._visible(rows, slot):
                rows[slot] = 0.0
            elif noisy:
                rows[slot, 1:] += np.random.normal(0.0, self.pose_std, size=AGENT_SLOT_WIDTH - 1)
        return rows.reshape(-1)

    def log_probability(self, clean_channel: Any, channel_observation: Any) -> float:
        rows = self._agent_rows(clean_channel)
        obs_rows = np.asarray(channel_observation, dtype=float).reshape(
            self.max_tracked_agents, AGENT_SLOT_WIDTH
        )
        total = 0.0
        for slot in range(self.max_tracked_agents):
            total += self._slot_log_prob(rows, obs_rows, slot)
        return float(total)

    def _agent_rows(self, channel_value: Any) -> np.ndarray:
        return np.asarray(channel_value, dtype=float).reshape(
            self.max_tracked_agents, AGENT_SLOT_WIDTH
        )

    def _visible(self, rows: np.ndarray, slot: int) -> bool:
        """Whether true agent ``slot`` is in range and not occluded (ego-frame)."""
        rel_x, rel_y = rows[slot, 1], rows[slot, 2]
        if self.perception_range is not None and float(np.hypot(rel_x, rel_y)) > (
            self.perception_range
        ):
            return False
        for other in range(self.max_tracked_agents):
            if other == slot or rows[other, 0] != 1.0:
                continue
            if _segment_occludes(
                0.0, 0.0, rel_x, rel_y, rows[other, 1], rows[other, 2], self.occlusion_radius
            ):
                return False
        return True

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
