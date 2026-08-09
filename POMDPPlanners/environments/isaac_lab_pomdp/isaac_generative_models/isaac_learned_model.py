# SPDX-License-Identifier: MIT

"""Fitted linear-Gaussian dynamics and linear reward, wired into the factored schema.

This is the system-identification path: when a task's high-level dynamics have no clean analytic
form (unlike the velocity-command tasks
:mod:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_unicycle_model`
covers), fit a first-order model from warm-up rollouts and plan against that. It is the same
:class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp.LinearGaussianTransition`
and :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp.LinearRewardModel`
the one-space model used, now over a *named* subset of the state rather than the whole vector.

That distinction is the point. A fit is only valid on the channels the rollouts actually measured;
an augmented schema that adds a latent block must not feed that block to a model fitted without it.
:class:`BlockRewardModel` and ``transition_channels`` together keep the fitted pieces looking at
exactly the block they were trained on.

Classes:
    BlockRewardModel: Restrict a reward model to named state blocks.
    LearnedIsaacModel: Factored model driven by a fitted linear-Gaussian transition and reward.
"""

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_factored_model import (
    FactoredIsaacModelPOMDP,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_model_pomdp import (
    IsaacChannelSchema,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp import (
    LinearGaussianTransition,
    LinearRewardModel,
    RewardModel,
    TransitionModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_model import (
    IsaacObservationModel,
)


class BlockRewardModel(RewardModel):
    """Restrict a reward model to named state blocks before delegating to it.

    A reward fitted on warm-up rollouts saw only the channels those rollouts recorded. Handing it a
    wider augmented state would silently multiply its coefficients against blocks it has never
    seen, producing a reward that looks plausible and is wrong. This adapter slices first.

    Attributes:
        channels: The state blocks passed through to the wrapped model, in order.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
        ...     IsaacChannelSchema)
        >>> class SumReward:
        ...     def reward(self, s, a, n): return float(np.sum(n))
        >>> schema = IsaacChannelSchema((("base_pose", 2), ("hazard_type", 2)))
        >>> model = BlockRewardModel(SumReward(), schema, ("base_pose",))
        >>> state = schema.pack({"base_pose": [1.0, 2.0], "hazard_type": [0.0, 1.0]})
        >>> model.reward(state, None, state)  # the hazard block is not summed
        3.0
    """

    def __init__(
        self,
        reward_model: RewardModel,
        state_schema: IsaacChannelSchema,
        channels: Sequence[str],
    ) -> None:
        """Initialize the block-restricted reward model.

        Args:
            reward_model: The model to delegate to, fitted on the named blocks.
            state_schema: Named blocks of the flat state vector.
            channels: The blocks to slice out and pass through, in order.

        Raises:
            ValueError: If ``channels`` names a block the schema does not declare.
        """
        unknown = sorted(set(channels) - set(state_schema.names))
        if unknown:
            raise ValueError(
                f"BlockRewardModel channels not declared by the schema: {unknown}; "
                f"schema has {list(state_schema.names)}"
            )
        self._reward_model = reward_model
        self._indices = state_schema.indices_of(list(channels))
        self.channels = tuple(channels)

    def reward(self, state: Any, action: Any, next_state: Any) -> float:
        return float(
            self._reward_model.reward(
                np.asarray(state, dtype=float)[..., self._indices],
                action,
                np.asarray(next_state, dtype=float)[..., self._indices],
            )
        )


class LearnedIsaacModel(FactoredIsaacModelPOMDP):
    """Factored Isaac model driven by a fitted linear-Gaussian transition and linear reward.

    Both fitted pieces are confined to ``dynamics_channels``: the transition drives that block and
    carries the rest, and the reward is wrapped in a :class:`BlockRewardModel` over the same block.
    An augmented schema can therefore add channels the rollouts never measured without refitting.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
        ...     GaussianChannelObservationModel)
        >>> from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
        ...     IsaacChannelSchema)
        >>>
        >>> rng = np.random.default_rng(0)
        >>> states = rng.normal(size=(200, 4))
        >>> actions = rng.normal(size=(200, 3))
        >>> next_states = states + 0.05 * rng.normal(size=(200, 4))
        >>> rewards = -np.linalg.norm(states, axis=1)
        >>>
        >>> schema = IsaacChannelSchema((("robot", 4), ("hazard_type", 2)))
        >>> model = LearnedIsaacModel.fit(
        ...     state_schema=schema,
        ...     dynamics_channels=("robot",),
        ...     action_presets=[np.zeros(3), np.ones(3)],
        ...     discount_factor=0.99,
        ...     states=states, actions=actions, next_states=next_states, rewards=rewards,
        ...     observation_models={"robot": GaussianChannelObservationModel(channel="robot")},
        ... )
        >>> state = schema.pack({"robot": np.zeros(4), "hazard_type": [0.0, 1.0]})
        >>> schema.block(model.sample_next_state(state, np.ones(3)), "hazard_type").tolist()
        [0.0, 1.0]
    """

    def __init__(
        self,
        state_schema: IsaacChannelSchema,
        action_presets: Sequence[ArrayLike],
        discount_factor: float,
        transition: TransitionModel,
        dynamics_channels: Sequence[str],
        task_reward: Optional[RewardModel] = None,
        observation_models: Optional[Mapping[str, IsaacObservationModel]] = None,
        raw_observation_schema: Optional[IsaacChannelSchema] = None,
        reward_range: Optional[Tuple[float, float]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the learned Isaac model from already-fitted pieces.

        Args:
            state_schema: Named blocks of the flat state vector.
            action_presets: Finite list of continuous action vectors to plan over.
            discount_factor: POMDP discount factor (shared with the world).
            transition: The fitted dynamics over ``dynamics_channels``.
            dynamics_channels: The blocks the fitted pieces were trained on.
            task_reward: The fitted reward over the same blocks, or ``None`` for undirected
                planning.
            observation_models: ``{channel: IsaacObservationModel}``.
            raw_observation_schema: Named blocks of the world's flat raw observation.
            reward_range: Optional ``(min, max)`` reward bounds.
            name: Model name, also used to label planner output.
        """
        self.dynamics_channels = tuple(dynamics_channels)
        super().__init__(
            state_schema=state_schema,
            action_presets=action_presets,
            discount_factor=discount_factor,
            transition=transition,
            reward_model=(
                BlockRewardModel(task_reward, state_schema, self.dynamics_channels)
                if task_reward is not None
                else None
            ),
            transition_channels=self.dynamics_channels,
            observation_models=observation_models,
            raw_observation_schema=raw_observation_schema,
            reward_range=reward_range,
            name=name,
        )

    @classmethod
    def fit(
        cls,
        state_schema: IsaacChannelSchema,
        dynamics_channels: Sequence[str],
        action_presets: Sequence[ArrayLike],
        discount_factor: float,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        rewards: Optional[np.ndarray] = None,
        observation_models: Optional[Mapping[str, IsaacObservationModel]] = None,
        raw_observation_schema: Optional[IsaacChannelSchema] = None,
        reward_range: Optional[Tuple[float, float]] = None,
        name: Optional[str] = None,
    ) -> "LearnedIsaacModel":
        """Fit the transition (and reward, when rewards are given) from warm-up rollouts.

        Args:
            state_schema: Named blocks of the flat state vector.
            dynamics_channels: The blocks the rollouts measured; the fit is confined to them.
            action_presets: Finite list of continuous action vectors to plan over.
            discount_factor: POMDP discount factor (shared with the world).
            states: Source states, shape ``(N, d)`` where ``d`` is the dynamics-block width.
            actions: Applied actions, shape ``(N, action_dim)``.
            next_states: Resulting states, shape ``(N, d)``.
            rewards: Observed rewards, shape ``(N,)``. ``None`` leaves the model without an
                objective, which means undirected planning.
            observation_models: ``{channel: IsaacObservationModel}``.
            raw_observation_schema: Named blocks of the world's flat raw observation.
            reward_range: Optional ``(min, max)`` reward bounds.
            name: Model name, also used to label planner output.

        Returns:
            The fitted :class:`LearnedIsaacModel`.
        """
        return cls(
            state_schema=state_schema,
            action_presets=action_presets,
            discount_factor=discount_factor,
            transition=LinearGaussianTransition.fit(states, actions, next_states),
            dynamics_channels=dynamics_channels,
            task_reward=(
                LinearRewardModel.fit(states, actions, next_states, rewards)
                if rewards is not None
                else None
            ),
            observation_models=observation_models,
            raw_observation_schema=raw_observation_schema,
            reward_range=reward_range,
            name=name,
        )
