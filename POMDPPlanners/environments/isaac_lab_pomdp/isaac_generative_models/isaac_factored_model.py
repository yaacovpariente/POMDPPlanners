# SPDX-License-Identifier: MIT

"""Concrete Isaac generative model composing a transition, a reward and per-channel perception.

:class:`FactoredIsaacModelPOMDP` is the runnable model the one-space
:class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp.IsaacLabModelPOMDP`
becomes once state and observation are allowed to differ. It holds three swappable pieces:

* a :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp.TransitionModel`
  for the dynamics,
* a :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp.RewardModel` for
  the objective,
* and the ``{channel: IsaacObservationModel}`` map it inherits from the base.

Partially-driven state
----------------------
``transition_channels`` names the state blocks the transition actually moves; every other block is
copied through unchanged. That is what a persistent latent variable needs — a hazard type is a
property of the world fixed at episode start, not a random variable of the transition, and
resampling it each step would destroy exactly the belief dispersion a risk-sensitive planner
grades. It also means a transition fitted on the robot block alone can be dropped into an
augmented schema without rewriting it.

Regression to the one-space model
---------------------------------
A single ``("state", dim)`` channel, one
:class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models.proprioception_models.GaussianChannelObservationModel`
over it, and no ``transition_channels`` reproduces ``observation = state + N(0, Sigma)`` exactly.
The only difference is that the observation arrives as ``{"state": vector}`` rather than as a bare
vector.

Classes:
    FactoredIsaacModelPOMDP: Transition + reward + per-channel perception over a named schema.
"""

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_model_pomdp import (
    IsaacChannelSchema,
    IsaacModelPOMDP,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp import (
    RewardModel,
    TransitionModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_model import (
    IsaacObservationModel,
)


class FactoredIsaacModelPOMDP(IsaacModelPOMDP):
    """Isaac generative model pairing a swappable transition and reward with factored perception.

    Attributes:
        state_schema: Named blocks of the flat state vector.
        transition_channels: The state blocks the transition drives, or ``None`` when it drives
            the whole vector. Blocks outside it are carried through unchanged.
        action_presets: The finite set of action vectors the planner chooses among.
        observation_models: The ``{channel: IsaacObservationModel}`` map.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> from POMDPPlanners.environments.isaac_lab_pomdp import GaussianRandomWalkTransition
        >>> from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
        ...     GaussianChannelObservationModel)
        >>> from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
        ...     IsaacChannelSchema)
        >>>
        >>> schema = IsaacChannelSchema((("base_pose", 3), ("hazard_type", 2)))
        >>> model = FactoredIsaacModelPOMDP(
        ...     state_schema=schema,
        ...     action_presets=[np.zeros(3), np.ones(3)],
        ...     discount_factor=0.99,
        ...     transition=GaussianRandomWalkTransition(dim=3, process_noise_std=0.05),
        ...     transition_channels=("base_pose",),
        ...     observation_models={
        ...         "base_pose": GaussianChannelObservationModel(channel="base_pose")},
        ... )
        >>>
        >>> state = schema.pack({"base_pose": [0.0, 0.0, 0.0], "hazard_type": [0.0, 1.0]})
        >>> next_state, observation, reward = model.sample_next_step(state, model.get_actions()[0])
        >>> sorted(observation)
        ['base_pose']
        >>> schema.block(next_state, "hazard_type").tolist()  # latent block carried through
        [0.0, 1.0]
        >>> model.is_terminal(state)
        False
    """

    def __init__(
        self,
        state_schema: IsaacChannelSchema,
        action_presets: Sequence[ArrayLike],
        discount_factor: float,
        transition: TransitionModel,
        reward_model: Optional[RewardModel] = None,
        transition_channels: Optional[Sequence[str]] = None,
        observation_models: Optional[Mapping[str, IsaacObservationModel]] = None,
        raw_observation_schema: Optional[IsaacChannelSchema] = None,
        reward_range: Optional[Tuple[float, float]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the factored Isaac generative model.

        Args:
            state_schema: Named blocks of the flat state vector.
            action_presets: Finite list of continuous action vectors to plan over.
            discount_factor: POMDP discount factor (shared with the world).
            transition: The dynamics the planner searches under. It sees the driven block, so its
                width is that block's width, not necessarily the whole state's.
            reward_model: The objective the planner optimizes. ``None`` (the default) yields a
                flat zero reward and therefore undirected planning — supply one to make the
                planner solve the task.
            transition_channels: State blocks the transition drives. ``None`` (the default) means
                it drives the whole vector.
            observation_models: ``{channel: IsaacObservationModel}``.
            raw_observation_schema: Named blocks of the world's flat raw observation.
            reward_range: Optional ``(min, max)`` reward bounds.
            name: Model name, also used to label planner output.

        Raises:
            ValueError: If ``transition_channels`` names a block the schema does not declare.
        """
        super().__init__(
            state_schema=state_schema,
            action_presets=action_presets,
            discount_factor=discount_factor,
            observation_models=observation_models,
            raw_observation_schema=raw_observation_schema,
            reward_range=reward_range,
            name=name,
        )
        self._transition = transition
        self._reward_model = reward_model
        self.transition_channels = (
            tuple(transition_channels) if transition_channels is not None else None
        )
        self._driven_indices = self._resolve_driven_indices()

    def _resolve_driven_indices(self) -> Optional[np.ndarray]:
        if self.transition_channels is None:
            return None
        unknown = sorted(set(self.transition_channels) - set(self.state_schema.names))
        if unknown:
            raise ValueError(
                f"transition_channels names blocks the schema does not declare: {unknown}; "
                f"schema has {list(self.state_schema.names)}"
            )
        return self.state_schema.indices_of(self.transition_channels)

    def _driven(self, vectors: Any) -> np.ndarray:
        """Slice the transition-driven block out of one or many state vectors."""
        array = np.asarray(vectors, dtype=float)
        return array if self._driven_indices is None else array[..., self._driven_indices]

    def _recombine(self, state: Any, driven_next: np.ndarray, n_samples: int) -> np.ndarray:
        """Write a driven block back into copies of ``state``, keeping carried blocks fixed."""
        base = np.asarray(state, dtype=float).reshape(-1)
        if self._driven_indices is None:
            return driven_next
        if n_samples == 1:
            combined = base.copy()
            combined[self._driven_indices] = np.asarray(driven_next, dtype=float).reshape(-1)
            return combined
        combined = np.tile(base, (n_samples, 1))
        combined[:, self._driven_indices] = np.asarray(driven_next, dtype=float).reshape(
            n_samples, -1
        )
        return combined

    # ── Dynamics ────────────────────────────────────────────────────────
    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        driven_next = self._transition.sample_next_state(
            self._driven(state), action, n_samples=n_samples
        )
        return self._recombine(state, driven_next, n_samples)

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        # Blocks outside the driven set are deterministic, so they contribute no density term.
        return self._transition.log_probability(
            self._driven(state), action, self._driven(next_states)
        )

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        if self._reward_model is None:
            return 0.0  # no reward model -> undirected planning (see __init__ docstring)
        resulting = state if next_state is None else next_state
        return float(self._reward_model.reward(state, action, resulting))

    def is_terminal(self, state: Any) -> bool:
        # The world owns termination; a model that guessed at it would prune the search tree on
        # states the episode is still perfectly able to visit.
        del state
        return False
