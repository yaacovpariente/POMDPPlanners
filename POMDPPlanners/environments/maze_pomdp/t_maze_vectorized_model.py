# SPDX-License-Identifier: MIT

"""Torch vectorized generative model for :class:`TMazePOMDP`, for VOPP.

VOPP plans over a batch of particles on one device and cannot call the scalar
environment once per row. :class:`TMazeVectorizedModel` gives it the
transition, observation, reward and terminal kernels as torch gathers.

The T-Maze has the generated maze's state layout ``[x, y, goal_side,
cue_phase]``, its cue phases, its observation alphabet and its reward rule, on
the cells of a T instead of a carved grid. So the model reuses
:func:`~POMDPPlanners.environments.maze_pomdp.maze_vectorized_model.build_cue_maze_tables`:
the constructor enumerates every ``(cell, goal_side, cue_phase)`` state and
asks the scalar ``TMazePOMDP`` for each state's successor under each action,
its terminal flag and its observation likelihoods. Only the reward rule is
written out again in torch, because it reads the realised ``next_states``; the
parity test pins it to ``TMazePOMDP.reward``.

Encoding
    * State row: ``[x, y, goal_side, cue_phase]``, the scalar state array;
      ``x`` is negative in the left arm.
    * Action: index into ``env.get_actions()`` (``up``, ``down``, ``left``,
      ``right``).
    * Observation row: ``[k]`` with ``k`` the index into
      :data:`~POMDPPlanners.environments.maze_pomdp.t_maze_pomdp.OBSERVATIONS`
      (0 ``left_cue``, 1 ``right_cue``, 2 ``empty``).

Rows passed to the kernels must be states the environment can reach: a
position off the T is clamped onto the cell grid instead of raising, because a
check would force a host/device sync on every call.

Supported configurations: every ``TMazePOMDP``.
"""

from typing import Optional

import torch
from torch import Tensor

from POMDPPlanners.environments.maze_pomdp.maze_vectorized_model import (
    build_cue_maze_tables,
    cue_maze_rewards,
    cue_observation_log_probs,
    state_indices,
)
from POMDPPlanners.environments.maze_pomdp.t_maze_pomdp import OBSERVATIONS, TMazePOMDP


class TMazeVectorizedModel:
    """Torch generative model of :class:`TMazePOMDP`.

    Attributes:
        device: Device of every tensor argument and return value.
        dtype: Floating dtype of states, observations, rewards and log-probs.
        num_actions: Number of actions (4).
        num_observations: Number of observation codes (3).

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.maze_pomdp import TMazePOMDP
        >>> from POMDPPlanners.environments.maze_pomdp.t_maze_vectorized_model import (
        ...     TMazeVectorizedModel,
        ... )
        >>> env = TMazePOMDP(discount_factor=0.95)
        >>> model = TMazeVectorizedModel(env, device=torch.device("cpu"))
        >>> start = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        >>> model.sample_next_states(start, torch.tensor([0])).tolist()
        [[0.0, 1.0, 1.0, 1.0]]
    """

    def __init__(
        self,
        env: TMazePOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Build the tables from ``env``.

        Args:
            env: The T-Maze to mirror.
            device: Target device; defaults to CPU.
            dtype: Floating dtype of states, rewards and log-probabilities.
        """
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        actions = env.get_actions()
        self.num_actions = len(actions)
        self.num_observations = len(OBSERVATIONS)
        self._rewards = (env.goal_reward, env.wrong_goal_penalty, env.step_penalty)
        self._tables = build_cue_maze_tables(
            env,
            env.valid_cells,
            (env.left_endpoint, env.right_endpoint),
            actions,
            self.device,
            dtype,
        )

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        """``[N, 4]`` successors; the T-Maze's transitions are deterministic."""
        successor = self._tables.next_state_index[
            actions.to(torch.int64), state_indices(self._tables, states)
        ]
        return self._tables.state_rows[successor].to(states.dtype)

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        """``[N, 1]`` observation codes drawn from ``Z(. | s')``."""
        del actions
        log_probs = self._tables.observation_log_probs[state_indices(self._tables, next_states)]
        codes = torch.multinomial(torch.exp(log_probs), num_samples=1)
        return codes.to(self.dtype)

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        """``[N]`` rewards; see :func:`cue_maze_rewards`."""
        del actions
        return cue_maze_rewards(self._tables, states, next_states, self._rewards)

    def terminal_mask(self, states: Tensor) -> Tensor:
        """``[N]`` bool, ``True`` on an arm endpoint."""
        return self._tables.terminal[state_indices(self._tables, states)]

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        """``[N]`` ``log Z(o | s')``."""
        del actions
        return cue_observation_log_probs(self._tables, next_states, observations)

    def action_keys(self, actions: Tensor) -> Tensor:
        """Action keys are the action indices."""
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        """Observation keys are the observation codes."""
        return torch.round(observations.reshape(observations.shape[0], -1)[:, 0]).to(torch.int64)


__all__ = ["TMazeVectorizedModel"]
