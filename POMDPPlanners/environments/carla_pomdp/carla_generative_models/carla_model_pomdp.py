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
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from POMDPPlanners.core.environment import DiscreteActionsEnvironment, SpaceInfo, SpaceType

from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
    AGENT_SLOT_WIDTH,
    DEFAULT_ACTION_PRESETS,
    DEFAULT_MAX_TRACKED_AGENTS,
    EGO_STATE_WIDTH,
)


class CarlaModelPOMDP(DiscreteActionsEnvironment):
    """Abstract generative-model interface paired with the forward-only CARLA world.

    Concrete subclasses supply the dynamics — :meth:`sample_next_state`,
    :meth:`transition_log_probability`, :meth:`sample_observation`,
    :meth:`observation_log_probability`, :meth:`reward`, :meth:`is_terminal`, and the
    initial-distribution hooks — while this base owns the fixed CARLA schema shared by
    every model: the ego + agent-slot state layout, the discrete action set, and the
    observation-dict hashing/equality.

    Attributes:
        action_presets: Discrete ``(throttle, steer, brake)`` control triples; the
            discrete action set is the indices into this list.
        max_tracked_agents: Number of fixed agent slots in the state/observation.

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
            name: Environment identifier. Defaults to the class name.
        """
        presets = action_presets if action_presets is not None else DEFAULT_ACTION_PRESETS
        self.action_presets: List[Tuple[float, float, float]] = [
            (float(throttle), float(steer), float(brake)) for throttle, steer, brake in presets
        ]
        self.max_tracked_agents = max_tracked_agents
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

    # ── Dynamics (abstract — the interface a model must implement) ───────
    @abstractmethod
    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        """Sample ``n_samples`` next states for ``(state, action)``."""

    @abstractmethod
    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        """Log-density of ``next_states`` under the transition model for ``(state, action)``."""

    @abstractmethod
    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        """Sample ``n_samples`` observations for a resulting ``next_state``."""

    @abstractmethod
    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        """Log-density of ``observations`` under the observation model given ``next_state``."""
