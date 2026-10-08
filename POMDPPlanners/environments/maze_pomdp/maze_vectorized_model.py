# SPDX-License-Identifier: MIT

"""Torch vectorized generative model for :class:`DiscreteMazePOMDP`, for VOPP.

VOPP plans over a batch of particles on one device and cannot call the scalar
environment once per row. :class:`DiscreteMazeVectorizedModel` gives it the
transition, observation, reward and terminal kernels as torch gathers.

The maze's state space is finite and small: a walkable cell, a goal side
(2 values) and a cue phase (3 values), so ``6 * len(walkable_cells)`` states.
The constructor enumerates every one of them and asks the scalar environment
for its successor under each action (``sample_next_state``), whether it is
terminal (``is_terminal``) and its observation likelihoods
(``observation_log_probability``). The kernels then look these tables up. The
scalar environment therefore decides every transition, terminal flag and
observation probability; only the reward rule is written out again in torch,
because it reads the realised ``next_states`` the planner passes in, and the
parity test pins it to ``BaseMazePOMDP.reward``.

:func:`build_cue_maze_tables` does the enumeration. The T-Maze model in
:mod:`~POMDPPlanners.environments.maze_pomdp.t_maze_vectorized_model` uses it
too, because the T-Maze has the same state layout, cue phases and reward rule
on a different set of cells.

Encoding
    * State row: ``[x, y, goal_side, cue_phase]``, the scalar state array.
    * Action: index into ``env.get_actions()`` (``up``, ``down``, ``left``,
      ``right``).
    * Observation row: ``[k]`` with ``k`` the index into
      :data:`~POMDPPlanners.environments.maze_pomdp.maze_pomdp.OBSERVATIONS`
      (0 ``left_cue``, 1 ``right_cue``, 2 ``empty``). This is the code the
      maze's vectorized belief uses, so the two read the same tensors.

Rows passed to the kernels must be states the environment can reach: a
position that is not a walkable cell is clamped onto the cell grid instead of
raising, because a check would force a host/device sync on every call.

Supported configurations: every ``DiscreteMazePOMDP``. ``ContinuousMazePOMDP``
has no model: its actions are real 2-vectors, and VOPP needs a finite action
set.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.maze_pomdp.maze_pomdp import (
    CUE_CONSUMED,
    CUE_EMITTING,
    CUE_UNSEEN,
    GOAL_LEFT,
    GOAL_RIGHT,
    OBSERVATIONS,
    STATE_CUE_PHASE,
    STATE_GOAL,
    STATE_X,
    STATE_Y,
    DiscreteMazePOMDP,
)

Cell = Tuple[int, int]

_GOAL_SIDES = (GOAL_LEFT, GOAL_RIGHT)
_CUE_PHASES = (CUE_UNSEEN, CUE_EMITTING, CUE_CONSUMED)
_STATES_PER_CELL = len(_GOAL_SIDES) * len(_CUE_PHASES)
# Goal code of a cell that is not a goal.
_NOT_A_GOAL = -1


@dataclass(frozen=True)
class CueMazeTables:
    """Lookup tables for a cue maze, built from the scalar environment.

    A state's index is ``(cell * 2 + goal_side) * 3 + cue_phase``, where
    ``cell`` is the position's index in the sorted cell list.

    Attributes:
        cell_grid: ``[width, height]`` long tensor; the cell index at grid
            position ``(x - x_offset, y - y_offset)``, ``-1`` where no cell is.
        x_offset: Smallest ``x`` of any cell.
        y_offset: Smallest ``y`` of any cell.
        state_rows: ``[S, 4]`` state row of each state index.
        next_state_index: ``[A, S]`` index of the successor of each state.
        terminal: ``[S]`` bool, ``env.is_terminal`` of each state.
        goal_code: ``[C]`` long; 0 for the left goal cell, 1 for the right
            one, ``-1`` for every other cell.
        observation_log_probs: ``[S, O]`` ``log Z(o | s')`` per state and
            observation code.
    """

    cell_grid: Tensor
    x_offset: int
    y_offset: int
    state_rows: Tensor
    next_state_index: Tensor
    terminal: Tensor
    goal_code: Tensor
    observation_log_probs: Tensor


def build_cue_maze_tables(
    env: object,
    cells: Iterable[Cell],
    goal_cells: Tuple[Cell, Cell],
    actions: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> CueMazeTables:
    """Enumerate every state of a cue maze and record what the scalar env does.

    Args:
        env: A maze environment with ``sample_next_state``, ``is_terminal`` and
            ``observation_log_probability``, and the ``[x, y, goal_side,
            cue_phase]`` state layout.
        cells: Every cell a state can occupy.
        goal_cells: ``(left_goal_cell, right_goal_cell)``.
        actions: The environment's actions, in index order.
        device: Device of the returned tensors.
        dtype: Floating dtype of the returned float tensors.

    Returns:
        The tables the kernels index.
    """
    ordered = sorted((int(x), int(y)) for x, y in cells)
    position_of = {cell: index for index, cell in enumerate(ordered)}
    xs = [cell[0] for cell in ordered]
    ys = [cell[1] for cell in ordered]
    x_offset, y_offset = min(xs), min(ys)
    grid = np.full((max(xs) - x_offset + 1, max(ys) - y_offset + 1), -1, dtype=np.int64)
    for index, (x, y) in enumerate(ordered):
        grid[x - x_offset, y - y_offset] = index

    rows = np.array(
        [
            [float(x), float(y), goal, phase]
            for x, y in ordered
            for goal in _GOAL_SIDES
            for phase in _CUE_PHASES
        ],
        dtype=np.float64,
    )

    def state_index(row: np.ndarray) -> int:
        cell = position_of[(int(round(row[STATE_X])), int(round(row[STATE_Y])))]
        goal = _GOAL_SIDES.index(float(row[STATE_GOAL]))
        phase = _CUE_PHASES.index(float(row[STATE_CUE_PHASE]))
        return (cell * len(_GOAL_SIDES) + goal) * len(_CUE_PHASES) + phase

    successors = np.array(
        [
            [state_index(env.sample_next_state(row, action)) for row in rows]  # type: ignore[attr-defined]
            for action in actions
        ],
        dtype=np.int64,
    )
    terminal = np.array([bool(env.is_terminal(row)) for row in rows])  # type: ignore[attr-defined]
    log_probs = np.stack(
        [
            np.asarray(
                env.observation_log_probability(row, None, list(OBSERVATIONS)),  # type: ignore[attr-defined]
                dtype=np.float64,
            )
            for row in rows
        ]
    )
    goal_code = np.full(len(ordered), _NOT_A_GOAL, dtype=np.int64)
    for side, cell in enumerate(goal_cells):
        goal_code[position_of[(int(cell[0]), int(cell[1]))]] = side

    return CueMazeTables(
        cell_grid=torch.as_tensor(grid, device=device),
        x_offset=x_offset,
        y_offset=y_offset,
        state_rows=torch.as_tensor(rows, dtype=dtype, device=device),
        next_state_index=torch.as_tensor(successors, device=device),
        terminal=torch.as_tensor(terminal, device=device),
        goal_code=torch.as_tensor(goal_code, device=device),
        observation_log_probs=torch.as_tensor(log_probs, dtype=dtype, device=device),
    )


def cell_indices(tables: CueMazeTables, states: Tensor) -> Tensor:
    """``[N]`` cell index of each state row's position, clamped onto the grid."""
    x = torch.round(states[:, STATE_X]).to(torch.int64) - tables.x_offset
    y = torch.round(states[:, STATE_Y]).to(torch.int64) - tables.y_offset
    x = x.clamp(0, tables.cell_grid.shape[0] - 1)
    y = y.clamp(0, tables.cell_grid.shape[1] - 1)
    return tables.cell_grid[x, y].clamp(min=0)


def state_indices(tables: CueMazeTables, states: Tensor) -> Tensor:
    """``[N]`` state index of each state row."""
    goal = torch.round(states[:, STATE_GOAL]).to(torch.int64).clamp(0, len(_GOAL_SIDES) - 1)
    phase = torch.round(states[:, STATE_CUE_PHASE]).to(torch.int64).clamp(0, len(_CUE_PHASES) - 1)
    cell = cell_indices(tables, states)
    return (cell * len(_GOAL_SIDES) + goal) * len(_CUE_PHASES) + phase


def cue_maze_rewards(
    tables: CueMazeTables,
    states: Tensor,
    next_states: Tensor,
    rewards: Tuple[float, float, float],
) -> Tensor:
    """``[N]`` rewards under the shared maze reward rule.

    0 from a terminal state; otherwise ``goal_reward`` when ``next_states``
    lies on the goal cell ``states`` names, ``-wrong_goal_penalty`` on the
    other goal cell, and ``-step_penalty`` anywhere else.

    Args:
        tables: The maze's tables.
        states: ``[N, 4]`` current states.
        next_states: ``[N, 4]`` realised next states.
        rewards: ``(goal_reward, wrong_goal_penalty, step_penalty)``.

    Returns:
        ``[N]`` rewards in the dtype of ``tables.state_rows``.
    """
    goal_reward, wrong_goal_penalty, step_penalty = rewards
    dtype = tables.state_rows.dtype
    reached = tables.goal_code[cell_indices(tables, next_states)]
    side = torch.round(states[:, STATE_GOAL]).to(torch.int64)
    zeros = torch.zeros(reached.shape, dtype=dtype, device=reached.device)
    on_goal = torch.where(reached == side, zeros + goal_reward, zeros - wrong_goal_penalty)
    live = torch.where(reached == _NOT_A_GOAL, zeros - step_penalty, on_goal)
    current_terminal = tables.terminal[state_indices(tables, states)]
    return torch.where(current_terminal, zeros, live)


def cue_observation_log_probs(
    tables: CueMazeTables, next_states: Tensor, observations: Tensor
) -> Tensor:
    """``[N]`` ``log Z(o | s')``; ``-inf`` for a code outside the alphabet."""
    codes = torch.round(observations.reshape(observations.shape[0], -1)[:, 0]).to(torch.int64)
    known = (codes >= 0) & (codes < len(OBSERVATIONS))
    values = tables.observation_log_probs[
        state_indices(tables, next_states), codes.clamp(0, len(OBSERVATIONS) - 1)
    ]
    return torch.where(known, values, torch.full_like(values, -torch.inf))


class DiscreteMazeVectorizedModel:
    """Torch generative model of :class:`DiscreteMazePOMDP`.

    Attributes:
        device: Device of every tensor argument and return value.
        dtype: Floating dtype of states, observations, rewards and log-probs.
        num_actions: Number of actions (4).
        num_observations: Number of observation codes (3).

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.maze_pomdp import DiscreteMazePOMDP
        >>> from POMDPPlanners.environments.maze_pomdp.maze_vectorized_model import (
        ...     DiscreteMazeVectorizedModel,
        ... )
        >>> env = DiscreteMazePOMDP(discount_factor=0.95)
        >>> model = DiscreteMazeVectorizedModel(env, device=torch.device("cpu"))
        >>> start = torch.tensor([[3.0, 1.0, 0.0, 0.0]])
        >>> model.sample_next_states(start, torch.tensor([0])).tolist()
        [[3.0, 2.0, 0.0, 1.0]]
    """

    def __init__(
        self,
        env: DiscreteMazePOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Build the tables from ``env``.

        Args:
            env: The discrete maze to mirror.
            device: Target device; defaults to CPU.
            dtype: Floating dtype of states, rewards and log-probabilities.

        Raises:
            NotImplementedError: If ``env`` is not a ``DiscreteMazePOMDP``.
                ``ContinuousMazePOMDP`` takes real 2-vector actions, and VOPP
                needs a finite action set.
        """
        if not isinstance(env, DiscreteMazePOMDP):
            raise NotImplementedError(
                "DiscreteMazeVectorizedModel needs a DiscreteMazePOMDP; "
                f"{type(env).__name__} has no finite action set for VOPP"
            )
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        actions = env.get_actions()
        self.num_actions = len(actions)
        self.num_observations = len(OBSERVATIONS)
        self._rewards = (env.goal_reward, env.wrong_goal_penalty, env.step_penalty)
        self._tables = build_cue_maze_tables(
            env,
            env.walkable_cells,
            (env.left_goal_cell, env.right_goal_cell),
            actions,
            self.device,
            dtype,
        )

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        """``[N, 4]`` successors; the maze's transitions are deterministic."""
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
        """``[N]`` bool, ``True`` on a goal cell."""
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


__all__ = [
    "CueMazeTables",
    "DiscreteMazeVectorizedModel",
    "build_cue_maze_tables",
    "cell_indices",
    "cue_maze_rewards",
    "cue_observation_log_probs",
    "state_indices",
]
