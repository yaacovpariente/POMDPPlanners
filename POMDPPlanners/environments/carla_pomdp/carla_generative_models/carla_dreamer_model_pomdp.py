# SPDX-License-Identifier: MIT

"""Dreamer-backed concrete CARLA generative model.

:class:`DreamerCarlaModelPOMDP` implements the
:class:`~POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_model_pomdp.CarlaModelPOMDP`
interface by delegating every dynamic quantity to a trained Dreamer world model (an
RSSM). The POMDP *state* carried through the planner is the Dreamer **latent** (the
packed deterministic + stochastic recurrent state); the interface methods map onto the
world model's own components:

* :meth:`~DreamerCarlaModelPOMDP.sample_next_state` -> RSSM imagination step (advance the
  recurrent state under the action's control triple and sample the stochastic prior).
* :meth:`~DreamerCarlaModelPOMDP.sample_observation` -> decoder over the ``{gnss, agents}``
  observation heads.
* :meth:`~DreamerCarlaModelPOMDP.observation_log_probability` -> decoder log-density (used
  to reweight particles in the belief update).
* :meth:`~DreamerCarlaModelPOMDP.reward` -> learned reward head.
* :meth:`~DreamerCarlaModelPOMDP.is_terminal` -> continue/termination head, thresholded.

The trained network is injected as a :class:`DreamerWorldModel` — a small framework-
agnostic protocol — so this module carries no JAX/TF dependency and is testable with a
lightweight fake. Any concrete Dreamer implementation (e.g. a DreamerV3 RSSM) that exposes
those batched operations plugs in unchanged.

The discrete action set and the observation-dict hashing/equality are inherited from
:class:`~POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_model_pomdp.CarlaModelPOMDP` so the
world and the model agree on the schema by construction.

Classes:
    DreamerWorldModel: Protocol a trained Dreamer RSSM must satisfy.
    DreamerCarlaModelPOMDP: Concrete CARLA model backed by a Dreamer world model.
"""

from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_model_pomdp import (
    CarlaModelPOMDP,
)
from POMDPPlanners.environments.carla_pomdp.carla_pomdp import DEFAULT_MAX_TRACKED_AGENTS


class DreamerWorldModel(Protocol):
    """Batched operations a trained Dreamer RSSM must expose to back the CARLA model.

    Every latent is a 1-D float vector of length :attr:`latent_dim` (the packed
    deterministic + stochastic recurrent state). All methods are batched: they take a
    ``(batch, latent_dim)`` array of latents and return per-row results, so a single
    network call serves a whole particle set.

    Attributes:
        latent_dim: Width of a packed latent vector.
    """

    # Protocol stubs: the ``...`` bodies satisfy the type checker's return-path check.
    # pylint: disable=unnecessary-ellipsis
    latent_dim: int

    def encode(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        """Encode a real observation into a latent via the posterior (belief seed)."""
        ...

    def imagine(self, latents: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """Advance ``(batch, latent_dim)`` latents under ``(batch, 3)`` control triples."""
        ...

    def decode(self, latents: np.ndarray) -> Dict[str, np.ndarray]:
        """Decode ``(batch, latent_dim)`` latents to ``{gnss, agents}`` observation heads."""
        ...

    def decode_log_prob(
        self, latents: np.ndarray, observation: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """Log-density of one observation under each of ``(batch, latent_dim)`` latents."""
        ...

    def reward(self, latents: np.ndarray) -> np.ndarray:
        """Predicted reward for each of ``(batch, latent_dim)`` latents."""
        ...

    def continue_prob(self, latents: np.ndarray) -> np.ndarray:
        """Probability the episode continues for each of ``(batch, latent_dim)`` latents."""
        ...


class DreamerCarlaModelPOMDP(CarlaModelPOMDP):
    """Concrete CARLA generative model whose dynamics are a trained Dreamer world model.

    The planner-side *state* is the Dreamer latent; transitions, observations, reward, and
    termination are served by the injected :class:`DreamerWorldModel`. The belief is seeded
    by encoding the world's initial observation with the posterior.

    Attributes:
        world_model: The trained Dreamer RSSM backing every dynamic quantity.
        continue_threshold: Termination fires when the continue head's probability drops
            below this value.

    Note:
        Reward comes from the world model's **learned reward head**, not the analytic
        :func:`~POMDPPlanners.environments.carla_pomdp.carla_pomdp.driving_quality_reward`;
        a Dreamer model predicts reward directly from its latent.

    Example:
        >>> import numpy as np
        >>>
        >>> class _IdentityWorldModel:
        ...     latent_dim = 4
        ...     def encode(self, observation):
        ...         return np.zeros(self.latent_dim)
        ...     def imagine(self, latents, controls):
        ...         return np.asarray(latents, dtype=float)
        ...     def decode(self, latents):
        ...         batch = np.asarray(latents).shape[0]
        ...         return {"gnss": np.zeros((batch, 3)), "agents": np.zeros((batch, 25))}
        ...     def decode_log_prob(self, latents, observation):
        ...         return np.zeros(np.asarray(latents).shape[0])
        ...     def reward(self, latents):
        ...         return np.zeros(np.asarray(latents).shape[0])
        ...     def continue_prob(self, latents):
        ...         return np.ones(np.asarray(latents).shape[0])
        >>>
        >>> obs = {"gnss": np.zeros(3), "agents": np.zeros(25)}
        >>> env = DreamerCarlaModelPOMDP(
        ...     _IdentityWorldModel(), discount_factor=0.95, initial_observation=obs)
        >>>
        >>> state = env.initial_state_dist().sample()[0]
        >>> action = env.get_actions()[0]
        >>> next_state, observation, reward = env.sample_next_step(state, action)
        >>> sorted(observation)
        ['agents', 'gnss']
        >>> env.is_terminal(state)
        False
    """

    def __init__(
        self,
        world_model: DreamerWorldModel,
        discount_factor: float,
        action_presets: Optional[Sequence[Tuple[float, float, float]]] = None,
        max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
        continue_threshold: float = 0.5,
        initial_observation: Optional[Mapping[str, np.ndarray]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the Dreamer-backed CARLA generative model.

        Args:
            world_model: Trained Dreamer RSSM satisfying :class:`DreamerWorldModel`.
            discount_factor: Discount factor for future rewards (0 < d <= 1).
            action_presets: Discrete ``(throttle, steer, brake)`` triples. Defaults to the
                world's default presets.
            max_tracked_agents: Number of fixed agent slots in the observation schema.
            continue_threshold: Continue-head probability below which a state is terminal.
            initial_observation: The world's first real observation, encoded via the
                posterior to seed the belief. If omitted, the initial-distribution hooks
                raise, mirroring the factored reference model.
            name: Environment identifier. Defaults to the class name.
        """
        self.world_model = world_model
        self.continue_threshold = continue_threshold
        self._initial_observation: Optional[Dict[str, np.ndarray]] = (
            dict(initial_observation) if initial_observation is not None else None
        )
        super().__init__(
            discount_factor=discount_factor,
            action_presets=action_presets,
            max_tracked_agents=max_tracked_agents,
            name=name,
        )

    # ── Transition (RSSM imagination) ────────────────────────────────────
    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        controls = np.repeat(self._control_for(action), n_samples, axis=0)
        latents = np.repeat(self._as_batch(state), n_samples, axis=0)
        successors = np.asarray(self.world_model.imagine(latents, controls), dtype=float)
        return successors[0] if n_samples == 1 else successors

    def sample_next_state_batch(self, states: Any, action: Any) -> np.ndarray:
        latents = self._as_batch(states)
        controls = np.repeat(self._control_for(action), latents.shape[0], axis=0)
        return np.asarray(self.world_model.imagine(latents, controls), dtype=float)

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        del state, action, next_states
        raise NotImplementedError(
            "Dreamer's deterministic recurrent transition has no tractable joint density; "
            "the MCTS planners only require the sampling path (sample_next_state)."
        )

    # ── Observation model (decoder) ──────────────────────────────────────
    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        del action
        decoded = self.world_model.decode(self._as_batch(next_state))
        obs = {
            "gnss": np.asarray(decoded["gnss"], dtype=float)[0],
            "agents": np.asarray(decoded["agents"], dtype=float)[0],
        }
        if n_samples == 1:
            return obs
        return [
            {"gnss": obs["gnss"].copy(), "agents": obs["agents"].copy()} for _ in range(n_samples)
        ]

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        del action
        latents = self._as_batch(next_state)
        obs_list = observations if isinstance(observations, list) else [observations]
        return np.array(
            [float(self.world_model.decode_log_prob(latents, obs)[0]) for obs in obs_list]
        )

    def observation_log_probability_per_state(
        self, next_states: Any, action: Any, observation: Any
    ) -> np.ndarray:
        del action
        return np.asarray(
            self.world_model.decode_log_prob(self._as_batch(next_states), observation),
            dtype=float,
        )

    # ── Reward (learned reward head) ─────────────────────────────────────
    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        del action
        resulting = state if next_state is None else next_state
        return float(self.world_model.reward(self._as_batch(resulting))[0])

    # ── Terminal / initial hooks ─────────────────────────────────────────
    def is_terminal(self, state: Any) -> bool:
        prob = float(self.world_model.continue_prob(self._as_batch(state))[0])
        return prob < self.continue_threshold

    def initial_state_dist(self) -> Distribution:
        latent = np.asarray(
            self.world_model.encode(self._require_initial_observation()), dtype=float
        )

        class InitialState(Distribution):
            def sample(self, n_samples: int = 1) -> List[np.ndarray]:
                return [latent.copy() for _ in range(n_samples)]

        return InitialState()

    def initial_observation_dist(self) -> Distribution:
        observation = self._require_initial_observation()

        class InitialObservation(Distribution):
            def sample(self, n_samples: int = 1) -> List[Dict[str, np.ndarray]]:
                return [
                    {key: np.asarray(value).copy() for key, value in observation.items()}
                    for _ in range(n_samples)
                ]

        return InitialObservation()

    # ── Helpers ──────────────────────────────────────────────────────────
    def _control_for(self, action: Any) -> np.ndarray:
        return np.asarray(self.action_presets[int(action)], dtype=float).reshape(1, 3)

    def _as_batch(self, latents: Any) -> np.ndarray:
        array = np.asarray(latents, dtype=float)
        return array.reshape(1, -1) if array.ndim == 1 else array

    def _require_initial_observation(self) -> Dict[str, np.ndarray]:
        if self._initial_observation is None:
            raise NotImplementedError(
                "Seed the belief from the world's initial observation; pass it as "
                "initial_observation to encode it via the posterior."
            )
        return self._initial_observation
