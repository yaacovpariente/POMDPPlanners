# SPDX-License-Identifier: MIT

"""Abstract planner-side generative-model interface for the CARLA world.

:class:`~POMDPPlanners.environments.carla_pomdp.carla_pomdp.CarlaPOMDP` is a
forward-only *world* (no densities, no state injection). A planner instead carries a
generative *model* as ``policy.environment`` — one that can sample transitions from an
arbitrary state, score an observation against a state, and supply a reward. This module
defines the **interface** that model must satisfy: :class:`CarlaModelPOMDP` owns only the
CARLA state/observation *schema* (agent-slot layout, discrete action set, observation-dict
hashing/equality) and leaves every dynamic quantity — transition, observation, reward,
terminal — abstract for a study- or task-specific subclass (e.g. a learned model) to fill
in.

A runnable reference implementation with a fixed factored observation model lives in
:mod:`~POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_factored_model_pomdp`.

The schema (state/observation layout, action presets) is imported from
:mod:`~POMDPPlanners.environments.carla_pomdp.carla_pomdp` so world and model agree by
construction.

Classes:
    CarlaModelPOMDP: Abstract generative-model interface over the CARLA schema.
"""

from abc import abstractmethod
from collections.abc import Hashable
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from POMDPPlanners.core.environment import DiscreteActionsEnvironment, SpaceInfo, SpaceType

from POMDPPlanners.environments.carla_pomdp.carla_perception.observation_model import (
    CarlaObservationModel,
)
from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
    AGENT_SLOT_WIDTH,
    DEFAULT_ACTION_PRESETS,
    DEFAULT_MAX_TRACKED_AGENTS,
    EGO_STATE_WIDTH,
)


class CarlaModelPOMDP(DiscreteActionsEnvironment):
    """Abstract generative-model interface paired with the forward-only CARLA world.

    Concrete subclasses supply the dynamics — :meth:`sample_next_state`,
    :meth:`transition_log_probability`, :meth:`reward`, :meth:`is_terminal`, and the
    initial-distribution hooks — while this base owns the fixed CARLA schema shared by
    every model (the ego + agent-slot state layout, the discrete action set, and the
    observation-dict hashing/equality) *and* the observation model. The observation is
    produced by running a clean, fully-detected observation (built from the state) through
    a swappable
    :class:`~POMDPPlanners.environments.carla_pomdp.carla_perception.observation_model.CarlaObservationModel`
    perception. Two models differ in their observation only by the perception they hold; a
    model that scores observations for a belief update needs a perception with a density
    (``supports_density``). A subclass whose observation model is not a clean-observation
    transform (e.g. a learned latent decoder) may instead override :meth:`sample_observation`
    and :meth:`observation_log_probability` directly.

    Attributes:
        action_presets: Discrete ``(throttle, steer, brake)`` control triples; the
            discrete action set is the indices into this list.
        max_tracked_agents: Number of fixed agent slots in the state/observation.
        perception: The swappable observation model applied to the clean observation, or
            ``None`` for a subclass that overrides the observation methods directly.

    Note:
        This is an abstract base class and cannot be instantiated directly. See
        :class:`~POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_factored_model_pomdp.FactoredCarlaModelPOMDP`
        for a concrete reference implementation.
    """

    def __init__(
        self,
        discount_factor: float,
        action_presets: Optional[Sequence[Tuple[float, float, float]]] = None,
        max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
        perception: Optional[CarlaObservationModel] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the CARLA generative-model interface.

        Args:
            discount_factor: Discount factor for future rewards (0 < d <= 1).
            action_presets: Discrete ``(throttle, steer, brake)`` triples. Defaults to
                :data:`~POMDPPlanners.environments.carla_pomdp.carla_pomdp.DEFAULT_ACTION_PRESETS`.
            max_tracked_agents: Number of fixed agent slots carried in the state and
                observation. Defaults to
                :data:`~POMDPPlanners.environments.carla_pomdp.carla_pomdp.DEFAULT_MAX_TRACKED_AGENTS`.
            perception: The swappable observation model applied to the clean observation.
                ``None`` (the default) is for subclasses that override the observation
                methods directly.
            name: Environment identifier. Defaults to the class name.
        """
        presets = action_presets if action_presets is not None else DEFAULT_ACTION_PRESETS
        self.action_presets: List[Tuple[float, float, float]] = [
            (float(throttle), float(steer), float(brake)) for throttle, steer, brake in presets
        ]
        self.max_tracked_agents = max_tracked_agents
        self.perception = perception
        super().__init__(
            discount_factor=discount_factor,
            name=name if name is not None else type(self).__name__,
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE,
                observation_space=SpaceType.CONTINUOUS,
            ),
        )

    # ── Schema (concrete, shared by every model) ────────────────────────
    def get_actions(self) -> List[int]:
        """Discrete action set: indices into ``action_presets``."""
        return list(range(len(self.action_presets)))

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return np.array_equal(
            np.asarray(observation1["agents"]), np.asarray(observation2["agents"])
        ) and np.array_equal(np.asarray(observation1["gnss"]), np.asarray(observation2["gnss"]))

    def hash_observation(self, observation: Any) -> Hashable:
        return tuple((key, np.asarray(observation[key]).tobytes()) for key in sorted(observation))

    def hash_action(self, action: Any) -> Hashable:
        return action

    def _state_agent_rows(self, state: np.ndarray) -> np.ndarray:
        return state[EGO_STATE_WIDTH:].reshape(self.max_tracked_agents, AGENT_SLOT_WIDTH)

    def _clean_observation(self, state: Any) -> Dict[str, np.ndarray]:
        """Build the clean, fully-detected observation from a state (shared schema).

        This is the noise-free ``{gnss, agents}`` observation the perception degrades; both
        the sampler and the density route the state through it so the two agree by
        construction.
        """
        arr = np.asarray(state, dtype=float)
        return {
            "gnss": arr[:2].copy(),
            "agents": self._state_agent_rows(arr).reshape(-1).copy(),
        }

    # ── Observation model (perception applied to the clean observation) ──
    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        """Sample ``n_samples`` observations by perceiving ``next_state``'s clean reading."""
        del action
        perception = self._require_perception()
        obs = perception.perceive(self._clean_observation(next_state))
        return [obs for _ in range(n_samples)] if n_samples != 1 else obs

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        """Log-density of ``observations`` under the perception given ``next_state``."""
        del action
        perception = self._require_density()
        clean = self._clean_observation(next_state)
        obs_list = observations if isinstance(observations, list) else [observations]
        return np.array([perception.log_probability(clean, obs) for obs in obs_list])

    def _require_perception(self) -> CarlaObservationModel:
        if self.perception is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no perception; supply a CarlaObservationModel "
                "or override sample_observation."
            )
        return self.perception

    def _require_density(self) -> CarlaObservationModel:
        perception = self._require_perception()
        if not perception.supports_density:
            raise NotImplementedError(
                f"{type(perception).__name__} is a sample-only perception with no "
                "observation density; belief updates need a perception that scores "
                "observations, or override observation_log_probability."
            )
        return perception

    # ── Dynamics (abstract — the interface a model must implement) ───────
    @abstractmethod
    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        """Sample ``n_samples`` next states for ``(state, action)``."""

    @abstractmethod
    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        """Log-density of ``next_states`` under the transition model for ``(state, action)``."""
