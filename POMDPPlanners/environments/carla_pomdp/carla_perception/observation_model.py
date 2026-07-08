# SPDX-License-Identifier: MIT

"""Swappable observation model: a clean observation in, a perceived observation out.

Every CARLA variation — the forward-only world and each planner-side generative model —
shares one observation *definition* (the ``{gnss, agents}`` schema keyed off the CARLA
state layout). What differs between them is the **perception**: the transform applied when
a clean, fully-detected observation is fed through the observation model to yield the
degraded reading a planner actually sees. This module holds that transform as a single
swappable interface so the same object can drive the world and the models alike.

Two capabilities, with different reach:

* :meth:`CarlaObservationModel.perceive` — sample a perceived observation from a clean one.
  Required; used by the world (to emit a reading) and by a model (``sample_observation``).
* :meth:`CarlaObservationModel.log_probability` — the observation density. Optional; a
  sample-only perception (e.g. a learned/sensor pipeline) may leave it unimplemented and is
  still usable as the world's perception, but is rejected by a generative model that must
  score observations for a belief update.

Classes:
    CarlaObservationModel: Abstract clean-observation -> perceived-observation interface.
    FactoredObservationModel: Per-slot detection + Gaussian-noise reference perception.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
    AGENT_SLOT_WIDTH,
    DEFAULT_MAX_TRACKED_AGENTS,
    DEFAULT_OCCLUSION_RADIUS,
    DEFAULT_PERCEPTION_RANGE,
    _segment_occludes,
)

_LOG_EPS = -50.0  # log-prob floor for impossible / near-zero events


class CarlaObservationModel(ABC):
    """Abstract observation model: clean observation -> perceived observation.

    A concrete perception maps a clean, fully-detected observation (built from a state by
    the world or a model) to the degraded reading a planner sees. Implementations set
    :attr:`supports_density` to ``True`` when they also provide :meth:`log_probability`.

    Attributes:
        supports_density: Whether :meth:`log_probability` is implemented. Sample-only
            perceptions leave this ``False`` and are usable only where sampling is needed.

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    supports_density: bool = False

    @abstractmethod
    def perceive(self, clean_observation: Mapping[str, Any]) -> dict:
        """Sample a perceived observation from a clean, fully-detected one.

        Args:
            clean_observation: The noise-free observation dict (``{gnss, agents}`` in the
                shared CARLA schema) built from a state.

        Returns:
            The perceived observation dict of the same schema.
        """

    def log_probability(
        self, clean_observation: Mapping[str, Any], observation: Mapping[str, Any]
    ) -> float:
        """Log-density of ``observation`` given the clean observation.

        Args:
            clean_observation: The noise-free observation dict built from a state.
            observation: The observation whose likelihood is scored.

        Returns:
            The observation log-probability.

        Raises:
            NotImplementedError: If this is a sample-only perception without a density.
        """
        del clean_observation, observation
        raise NotImplementedError(
            f"{type(self).__name__} is a sample-only perception with no observation "
            "density; it cannot back a generative model that scores observations."
        )


class FactoredObservationModel(CarlaObservationModel):
    """Reference perception: per-slot detection gating plus additive Gaussian noise.

    Each agent slot is detected only when in perception range and not geometrically
    occluded by another agent on the ego->target sight line; a detected agent's pose is
    corrupted with additive Gaussian noise, as is the ``gnss`` reading. Provides both a
    sampler and a matching density, so it can back a scoring generative model.

    Attributes:
        max_tracked_agents: Number of fixed agent slots in the observation.
        perception_range: Metres beyond which an agent is undetectable (``None`` disables
            the range gate).
        occlusion_radius: Sight-line blocking radius among agents.
        pose_std: Std of Gaussian noise on a detected agent's pose measurement.
        gnss_std: Std of Gaussian noise on the ``gnss`` reading.
        detect_prob: Probability of detecting a visible agent; ``1 - detect_prob`` is the
            miss rate scored by the density.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> model = FactoredObservationModel(max_tracked_agents=1, perception_range=50.0)
        >>> clean = {"gnss": np.zeros(2), "agents": np.array([1.0, 10.0, 0.0, 0.0, 0.0])}
        >>> float(model.perceive(clean)["agents"][0])  # near agent detected
        1.0
    """

    supports_density = True

    def __init__(
        self,
        max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
        perception_range: Optional[float] = DEFAULT_PERCEPTION_RANGE,
        occlusion_radius: float = DEFAULT_OCCLUSION_RADIUS,
        pose_std: float = 0.5,
        gnss_std: float = 1e-5,
        detect_prob: float = 0.95,
    ) -> None:
        """Initialize the factored perception.

        Args:
            max_tracked_agents: Number of fixed agent slots in the observation.
            perception_range: Metres beyond which an agent is undetectable, or ``None``.
            occlusion_radius: Sight-line blocking radius among agents.
            pose_std: Std of Gaussian noise on a detected agent's pose measurement.
            gnss_std: Std of Gaussian noise on the ``gnss`` reading.
            detect_prob: Probability of detecting a visible agent.
        """
        self.max_tracked_agents = max_tracked_agents
        self.perception_range = perception_range
        self.occlusion_radius = occlusion_radius
        self.pose_std = pose_std
        self.gnss_std = gnss_std
        self.detect_prob = detect_prob

    def perceive(self, clean_observation: Mapping[str, Any]) -> dict:
        return self.render(clean_observation, noisy=True)

    def render(self, clean_observation: Mapping[str, Any], noisy: bool) -> dict:
        """Gate the clean observation per slot, optionally adding Gaussian noise.

        Args:
            clean_observation: The noise-free ``{gnss, agents}`` observation.
            noisy: When True, add pose and GNSS Gaussian noise (the sampler path); when
                False, return the gated but noise-free observation.

        Returns:
            The perceived observation dict.
        """
        rows = self._agent_rows(clean_observation).copy()
        for slot in range(self.max_tracked_agents):
            if rows[slot, 0] != 1.0 or not self._visible(rows, slot):
                rows[slot] = 0.0
            elif noisy:
                rows[slot, 1:] += np.random.normal(0.0, self.pose_std, size=AGENT_SLOT_WIDTH - 1)
        gnss = np.asarray(clean_observation["gnss"], dtype=float)[:2].copy()
        if noisy:
            gnss = gnss + np.random.normal(0.0, self.gnss_std, size=2)
        return {"gnss": gnss, "agents": rows.reshape(-1)}

    def log_probability(
        self, clean_observation: Mapping[str, Any], observation: Mapping[str, Any]
    ) -> float:
        rows = self._agent_rows(clean_observation)
        obs_rows = np.asarray(observation["agents"], dtype=float).reshape(
            self.max_tracked_agents, AGENT_SLOT_WIDTH
        )
        total = self._gnss_log_prob(clean_observation, observation)
        for slot in range(self.max_tracked_agents):
            total += self._slot_log_prob(rows, obs_rows, slot)
        return float(total)

    def _agent_rows(self, observation: Mapping[str, Any]) -> np.ndarray:
        return np.asarray(observation["agents"], dtype=float).reshape(
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

    def _gnss_log_prob(
        self, clean_observation: Mapping[str, Any], observation: Mapping[str, Any]
    ) -> float:
        truth = np.asarray(clean_observation["gnss"], dtype=float)[:2]
        gnss = np.asarray(observation["gnss"], dtype=float)[:2]
        diff = gnss - truth
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
