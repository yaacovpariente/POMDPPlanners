# SPDX-License-Identifier: MIT

"""Torch vectorized generative model for the Discrete Light-Dark POMDP.

VOPP plans through a batched torch model, so it cannot run on
:class:`~POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp.DiscreteLightDarkPOMDP`
without one. :class:`DiscreteLightDarkVectorizedModel` writes the
environment's transition, ``NORMAL`` observation, reward and terminal rules as
torch operations over a batch of grid states. Every probability table and
coordinate is read from a live environment, so the environment stays the one
place the configuration is set; only the arithmetic is duplicated, and the
registry parity tests compare the two.

Two configurations raise :class:`NotImplementedError` at construction:

* ``is_obstacle_hit_terminal=True``. The state then gains a third, absorbing
  terminal slot drawn with the obstacle hit, and this model keeps 2-D states.
* ``observation_model_type`` other than ``NORMAL``. ``NO_OBS_IN_DARK`` and
  ``DISTANCE_BASED`` emit the string ``"None"`` far from beacons, and a
  ``[N, 2]`` observation tensor has no row for it.
"""

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
    ObservationModelType,
)

# Primes that mix the two integer coordinates of an observation into one key.
_HASH_PRIME_X = 73856093
_HASH_PRIME_Y = 19349663


class DiscreteLightDarkVectorizedModel:
    """Batched torch model of the Discrete Light-Dark POMDP kernels.

    States and observations are ``[N, 2]`` grid coordinates stored as floats.
    Action index ``i`` is ``env.get_actions()[i]`` (``up``, ``down``,
    ``right``, ``left``).

    * Transition: action ``i`` moves by its own vector with probability
      ``1 - transition_error_prob`` and by each other action's vector with
      ``transition_error_prob / 3``, as in the environment's
      ``_transition_probs``.
    * Observation: the next state shifted by one action vector, or unshifted.
      The probabilities come from the environment's near-beacon table when the
      next state is closer than ``beacon_radius`` to a beacon, and from its
      far table otherwise.
    * Reward: ``-fuel_cost - ||s' - goal||``, plus ``goal_reward`` on the goal,
      else ``obstacle_reward`` with probability ``obstacle_hit_probability`` on
      an obstacle cell, else ``obstacle_reward`` outside ``[0, grid_size]``.
    * Terminal: the state equals the goal or an obstacle cell.

    Attributes:
        device: Device of every tensor argument and return value.
        dtype: Floating dtype of state, observation and reward tensors.
        num_actions: Number of actions (4).

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
        ...     DiscreteLightDarkPOMDP,
        ... )
        >>> from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_vectorized_model import (
        ...     DiscreteLightDarkVectorizedModel,
        ... )
        >>> env = DiscreteLightDarkPOMDP(discount_factor=0.95)
        >>> model = DiscreteLightDarkVectorizedModel(env, device=torch.device("cpu"))
        >>> states = torch.tensor([[0.0, 5.0], [1.0, 5.0]])
        >>> actions = torch.tensor([2, 2])  # "right"
        >>> next_states = model.sample_next_states(states, actions)
        >>> observations = model.sample_observations(next_states, actions)
        >>> tuple(model.rewards(states, actions, next_states).shape)
        (2,)
        >>> model.terminal_mask(torch.tensor([[10.0, 5.0], [3.0, 7.0], [0.0, 5.0]])).tolist()
        [True, True, False]
    """

    def __init__(
        self,
        env: DiscreteLightDarkPOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Read the kernels' tables and geometry from ``env``.

        Args:
            env: The environment whose configuration the model copies.
            device: Target device; defaults to CPU.
            dtype: Floating dtype of state, observation and reward tensors.

        Raises:
            NotImplementedError: If ``env.is_obstacle_hit_terminal`` is True or
                ``env.observation_model_type`` is not ``NORMAL``.
        """
        self._require_supported_config(env)
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        actions = env.get_actions()
        self.num_actions = len(actions)
        vectors = np.stack([np.asarray(env.action_to_vector[a], dtype=np.float64) for a in actions])
        self._action_vectors = self._to_tensor(vectors)
        # Observation index j < num_actions shifts by action j's vector; the
        # last index is the unshifted next state, matching the env's ordering.
        self._obs_offsets = self._to_tensor(np.vstack([vectors, np.zeros((1, 2))]))
        # The env's own probability tables, so the two cannot disagree on them.
        # pylint: disable=protected-access
        transition_probs = env._transition_probs
        obs_probs = np.stack([env._obs_probs_near, env._obs_probs_far]).astype(np.float64)
        # pylint: enable=protected-access
        self._transition_cum = self._to_tensor(
            np.stack([np.cumsum(transition_probs[a]) for a in actions])
        )
        self._obs_cum = self._to_tensor(np.cumsum(obs_probs, axis=1))
        with np.errstate(divide="ignore"):
            self._obs_log_probs = self._to_tensor(np.log(obs_probs))
        self._beacons = self._to_tensor(np.asarray(env.beacons, dtype=np.float64).T)
        self._beacon_radius_sq = float(env.beacon_radius) ** 2
        self._obstacles = self._to_tensor(np.asarray(env.obstacles, dtype=np.float64).T)
        self._goal = self._to_tensor(np.asarray(env.goal_state, dtype=np.float64))
        self._grid_size = float(env.grid_size)
        self._fuel_cost = float(env.fuel_cost)
        self._goal_reward = float(env.goal_reward)
        self._obstacle_reward = float(env.obstacle_reward)
        self._hit_probability = float(env.obstacle_hit_probability)

    @staticmethod
    def _require_supported_config(env: DiscreteLightDarkPOMDP) -> None:
        if env.is_obstacle_hit_terminal:
            raise NotImplementedError(
                "vectorized model requires is_obstacle_hit_terminal=False "
                "(the hazard-terminal absorbing state slot is not modeled)"
            )
        if env.observation_model_type is not ObservationModelType.NORMAL:
            raise NotImplementedError(
                "vectorized model supports only the NORMAL observation model "
                "(the 'None' observation has no tensor encoding)"
            )

    def _to_tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)

    @property
    def action_vectors(self) -> Tensor:
        """``[num_actions, 2]`` grid displacement of each action index."""
        return self._action_vectors

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        draws = torch.rand(states.shape[0], 1, dtype=self.dtype, device=self.device)
        # Inverse-CDF draw: the env's np.searchsorted(cum, u) counts cum < u.
        index = (self._transition_cum[actions] < draws).sum(dim=1)
        index = index.clamp(max=self.num_actions - 1)
        return states + self._action_vectors[index]

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions  # The observation depends only on the next state.
        cum = self._obs_cum[self._far_index(next_states)]
        draws = torch.rand(next_states.shape[0], 1, dtype=self.dtype, device=self.device)
        index = (cum < draws).sum(dim=1).clamp(max=self.num_actions)
        return next_states + self._obs_offsets[index]

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        del states, actions  # The reward scores the realised next state only.
        reward = -self._fuel_cost - torch.linalg.vector_norm(next_states - self._goal, dim=1)
        is_goal = (next_states == self._goal).all(dim=1)
        on_obstacle = self._on_obstacle(next_states) & ~is_goal
        out_of_grid = (
            ((next_states < 0.0) | (next_states > self._grid_size)).any(dim=1)
            & ~is_goal
            & ~on_obstacle
        )
        draws = torch.rand(next_states.shape[0], dtype=self.dtype, device=self.device)
        hit = on_obstacle & (draws < self._hit_probability)
        reward = reward + self._goal_reward * is_goal.to(self.dtype)
        reward = reward + self._obstacle_reward * (out_of_grid | hit).to(self.dtype)
        return reward

    def terminal_mask(self, states: Tensor) -> Tensor:
        positions = states[:, :2]
        return (positions == self._goal).all(dim=1) | self._on_obstacle(positions)

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions  # The likelihood depends only on the next state.
        candidates = next_states[:, None, :] + self._obs_offsets[None, :, :]
        matches = (observations[:, None, :] == candidates).all(dim=2)
        log_probs = self._obs_log_probs[self._far_index(next_states)]
        # The offsets are distinct, so at most one candidate matches a row.
        matched = torch.where(matches, log_probs, torch.full_like(log_probs, -torch.inf))
        return matched.max(dim=1).values

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        cells = torch.round(observations).to(torch.int64)
        return cells[:, 0] * _HASH_PRIME_X + cells[:, 1] * _HASH_PRIME_Y

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #

    def _far_index(self, points: Tensor) -> Tensor:
        # 0 selects the near-beacon table, 1 the far one. The env tests
        # distance < beacon_radius, so a point exactly on the radius is far.
        diff = points[:, None, :] - self._beacons[None, :, :]
        min_sq = (diff * diff).sum(dim=-1).min(dim=1).values
        return (min_sq >= self._beacon_radius_sq).to(torch.long)

    def _on_obstacle(self, points: Tensor) -> Tensor:
        if self._obstacles.shape[0] == 0:
            return torch.zeros(points.shape[0], dtype=torch.bool, device=self.device)
        return (points[:, None, :] == self._obstacles[None, :, :]).all(dim=2).any(dim=1)
