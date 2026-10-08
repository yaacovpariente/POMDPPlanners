# SPDX-License-Identifier: MIT

"""Torch vectorized generative model for the Snake POMDP.

VOPP searches on a batch of particles at once, so it needs Snake's transition,
observation, reward and terminal kernels as tensor operations.
:class:`SnakeVectorizedModel` re-expresses the kernels of
:class:`~POMDPPlanners.environments.snake_pomdp.snake_pomdp.SnakePOMDP` in
torch and reads every constant (grid size, target length, window radius,
sensor probabilities, starvation limit) from a live environment, so the
environment stays the single source of truth for configuration.

The snake's body has a variable length, but the scalar state vector already
has a fixed width: ``[status, length, steps_since_food, food_row, food_col,
r0, c0, ..., r_{T-1}, c_{T-1}]`` with ``T = target_length`` body slots, and
``-1`` in every slot past ``length``. A live snake is shorter than ``T``
(reaching ``T`` ends the episode), so ``T`` slots always suffice. The model
uses that layout unchanged, so a scalar state and a model row are the same
numbers.

An observation row has width ``4 + 2 * T``: ``[tag, scent, seen_row, seen_col,
r0, c0, ...]`` with ``-1`` in every body slot past the snake's length. It is the
scalar reading padded to a fixed width. The scalar terminal reading ``(0,)``
becomes ``[0, -1, ..., -1]``. :func:`snake_observation_to_row` and
:func:`snake_row_to_observation` convert between the two.

Every ``SnakePOMDP`` configuration is supported. The constructor has no enum
or switch parameter that changes the dynamics, so nothing is declined.
"""

import math
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.snake_pomdp.snake_pomdp import (
    BODY_OFFSET,
    DIRECTIONS,
    EMPTY,
    FOOD_COL_INDEX,
    FOOD_ROW_INDEX,
    LENGTH_INDEX,
    OBSERVATION_LIVE,
    OBSERVATION_TERMINAL,
    STATUS_INDEX,
    STEPS_SINCE_FOOD_INDEX,
    SnakeAction,
    SnakePOMDP,
    SnakeQuadrant,
    SnakeTermination,
)

#: Columns of an observation row before the body block.
_OBS_TAG = 0
_OBS_SCENT = 1
_OBS_SEEN_ROW = 2
_OBS_SEEN_COL = 3
_OBS_BODY_OFFSET = 4

#: Vertical sign (-1 north, +1 south) and horizontal sign (+1 east, -1 west) of
#: each quadrant, indexed by :class:`SnakeQuadrant`.
_QUADRANT_SIGNS: Tuple[Tuple[int, int], ...] = tuple(
    (
        -1 if quadrant in (SnakeQuadrant.NORTH_EAST, SnakeQuadrant.NORTH_WEST) else 1,
        1 if quadrant in (SnakeQuadrant.NORTH_EAST, SnakeQuadrant.SOUTH_EAST) else -1,
    )
    for quadrant in SnakeQuadrant
)


def snake_observation_to_row(env: SnakePOMDP, observation: Any) -> np.ndarray:
    """Pad a scalar Snake reading to the model's fixed-width observation row.

    Args:
        env: The environment, which fixes the row width.
        observation: A reading from ``env.sample_observation``.

    Returns:
        A ``float64`` row of width ``4 + 2 * env.target_length``.
    """
    flat = [int(value) for value in observation]
    row = np.full(_OBS_BODY_OFFSET + 2 * env.target_length, EMPTY, dtype=np.float64)
    row[: len(flat)] = flat
    return row


def snake_row_to_observation(env: SnakePOMDP, row: Any) -> Tuple[int, ...]:
    """Strip a model observation row back to the scalar Snake reading.

    Live body cells are inside the grid, so a ``-1`` body slot can only be
    padding.

    Args:
        env: The environment.
        row: A row from :meth:`SnakeVectorizedModel.sample_observations`.

    Returns:
        The reading as ``env.sample_observation`` would return it.
    """
    del env
    values = [int(round(float(value))) for value in np.asarray(row, dtype=np.float64)]
    if values[_OBS_TAG] != OBSERVATION_LIVE:
        return (OBSERVATION_TERMINAL,)
    end = _OBS_BODY_OFFSET
    while end + 1 < len(values) and values[end] >= 0 and values[end + 1] >= 0:
        end += 2
    return tuple(values[:end])


class SnakeVectorizedModel:  # pylint: disable=too-many-instance-attributes
    """Batched torch generative model for :class:`SnakePOMDP`.

    Actions are the :class:`SnakeAction` values used as indices, so
    ``num_actions == 3``. States and observations use the layouts in the module
    docstring.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype of state, observation and reward tensors.
        num_actions: Number of actions, 3.
        state_size: Width of a state row, ``5 + 2 * target_length``.
        observation_size: Width of an observation row, ``4 + 2 * target_length``.

    Example:
        >>> import numpy as np
        >>> import torch
        >>> from POMDPPlanners.environments.snake_pomdp import SnakePOMDP
        >>> from POMDPPlanners.environments.snake_pomdp.snake_vectorized_model import (
        ...     SnakeVectorizedModel,
        ... )
        >>> np.random.seed(0)
        >>> env = SnakePOMDP()
        >>> model = SnakeVectorizedModel(env, device=torch.device("cpu"))
        >>> states = torch.as_tensor(np.stack(env.initial_state_dist().sample(4)))
        >>> actions = torch.tensor([0, 1, 2, 1])
        >>> next_states = model.sample_next_states(states, actions)
        >>> tuple(next_states.shape), tuple(model.sample_observations(next_states, actions).shape)
        ((4, 25), (4, 24))
    """

    def __init__(
        self,
        env: SnakePOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Read the configuration of ``env``.

        Args:
            env: The environment whose kernels are mirrored.
            device: Target device. Defaults to CPU.
            dtype: Floating dtype for state, observation and reward tensors.
                Every stored value is a small integer, so ``float32`` holds
                them exactly.
        """
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self.num_actions = len(SnakeAction)
        self._grid = int(env.grid_size)
        self._target = int(env.target_length)
        self._radius = int(env.window_radius)
        self._detection = float(env.detection_probability)
        self._accuracy = float(env.scent_accuracy)
        self._starvation = int(env.starvation_limit)
        self.state_size = int(env.state_size)
        self.observation_size = _OBS_BODY_OFFSET + 2 * self._target

        self._directions = torch.tensor(DIRECTIONS, dtype=torch.int64, device=self.device)
        # Index into DIRECTIONS of a (row, col) step, looked up at
        # (d_row + 1) * 3 + (d_col + 1). Entries that are not axis steps map
        # to 0; only a terminal row's body can produce one, and terminal rows
        # are held unchanged.
        heading_index = torch.zeros(9, dtype=torch.int64, device=self.device)
        for index, (d_row, d_col) in enumerate(DIRECTIONS):
            heading_index[(d_row + 1) * 3 + (d_col + 1)] = index
        self._heading_index = heading_index
        turn_of_action = {SnakeAction.TURN_LEFT: -1, SnakeAction.TURN_RIGHT: 1}
        self._turn = torch.tensor(
            [turn_of_action.get(action, 0) for action in SnakeAction],
            dtype=torch.int64,
            device=self.device,
        )
        self._quadrant_signs = torch.tensor(_QUADRANT_SIGNS, dtype=torch.int64, device=self.device)
        self._slots = torch.arange(self._target, device=self.device)

    # ------------------------------------------------------------------ #
    # Protocol kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        """Move each snake and respawn its food uniformly on a free cell when eaten.

        Args:
            states: ``[N, state_size]`` states.
            actions: ``[N]`` action indices.

        Returns:
            ``[N, state_size]`` next states. Terminal rows are returned
            unchanged, as the scalar environment holds a terminal state.
        """
        body, new_len, eat, counter, status, food = self._outcome(states, actions)
        count = states.shape[0]

        # Food: a wall or self hit keeps the old food, a win clears it, and an
        # eat that leaves the episode running draws a free cell of the new body.
        respawn = eat & (status == int(SnakeTermination.RUNNING))
        new_food = food.clone()
        new_food[status == int(SnakeTermination.WIN)] = int(EMPTY)
        drawn = self._draw_free_cells(body, new_len)
        new_food = torch.where(respawn[:, None], drawn, new_food)

        nxt = torch.empty_like(states)
        nxt[:, STATUS_INDEX] = status.to(states.dtype)
        nxt[:, LENGTH_INDEX] = new_len.to(states.dtype)
        nxt[:, STEPS_SINCE_FOOD_INDEX] = counter.to(states.dtype)
        nxt[:, FOOD_ROW_INDEX] = new_food[:, 0].to(states.dtype)
        nxt[:, FOOD_COL_INDEX] = new_food[:, 1].to(states.dtype)
        nxt[:, BODY_OFFSET:] = body.reshape(count, -1).to(states.dtype)
        terminal = self.terminal_mask(states)
        return torch.where(terminal[:, None], states, nxt)

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        """Report the body, a sighting with no false positives, and a noisy scent.

        Args:
            next_states: ``[N, state_size]`` post-transition states.
            actions: ``[N]`` action indices; unused, as in the scalar sensor.

        Returns:
            ``[N, observation_size]`` readings. Terminal rows get
            ``[0, -1, ..., -1]``.
        """
        del actions
        count = next_states.shape[0]
        head, food, inside, scent_probs = self._sensor(next_states)
        terminal = self.terminal_mask(next_states)

        detected = inside & (
            torch.rand(count, dtype=self.dtype, device=self.device) < self._detection
        )
        # Terminal rows can have degenerate scent laws; give them a uniform one
        # so the draw is defined. Their reading is overwritten below.
        safe_probs = torch.where(terminal[:, None], torch.full_like(scent_probs, 0.25), scent_probs)
        scent = torch.multinomial(safe_probs, num_samples=1).squeeze(1)
        del head

        obs = torch.full(
            (count, self.observation_size), EMPTY, dtype=next_states.dtype, device=self.device
        )
        obs[:, _OBS_TAG] = float(OBSERVATION_LIVE)
        obs[:, _OBS_SCENT] = scent.to(next_states.dtype)
        seen = torch.where(detected[:, None], food, torch.full_like(food, int(EMPTY)))
        obs[:, _OBS_SEEN_ROW] = seen[:, 0].to(next_states.dtype)
        obs[:, _OBS_SEEN_COL] = seen[:, 1].to(next_states.dtype)
        obs[:, _OBS_BODY_OFFSET:] = next_states[:, BODY_OFFSET:]

        terminal_obs = torch.full_like(obs, EMPTY)
        terminal_obs[:, _OBS_TAG] = float(OBSERVATION_TERMINAL)
        return torch.where(terminal[:, None], terminal_obs, obs)

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        """Score each step: ``+1`` for eating, ``-1`` for dying, ``0`` otherwise.

        The reward depends on ``(state, action)`` only, as in the scalar
        environment, so ``next_states`` is unused.

        Args:
            states: ``[N, state_size]`` states.
            actions: ``[N]`` action indices.
            next_states: ``[N, state_size]``; unused.

        Returns:
            ``[N]`` rewards. A terminal source state earns ``0``.
        """
        del next_states
        _, _, eat, _, status, _ = self._outcome(states, actions)
        died = (
            (status == int(SnakeTermination.WALL))
            | (status == int(SnakeTermination.SELF))
            | (status == int(SnakeTermination.STARVATION))
        )
        reward = torch.where(
            eat,
            torch.ones(states.shape[0], dtype=self.dtype, device=self.device),
            torch.where(
                died,
                torch.full((states.shape[0],), -1.0, dtype=self.dtype, device=self.device),
                torch.zeros(states.shape[0], dtype=self.dtype, device=self.device),
            ),
        )
        return torch.where(self.terminal_mask(states), torch.zeros_like(reward), reward)

    def terminal_mask(self, states: Tensor) -> Tensor:
        """Flag rows whose status is not ``RUNNING``."""
        return states[:, STATUS_INDEX].round().to(torch.int64) != int(SnakeTermination.RUNNING)

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        """Score each reading as ``SnakePOMDP.observation_log_probability`` does.

        A terminal state emits only the terminal reading (log-probability 0).
        A live state's reading must carry its exact body and a scent in
        ``0..3``; the sighting adds ``log p`` for the food cell seen inside the
        window, ``log(1 - p)`` for no sighting while the food is inside, and 0
        for no sighting while it is outside, where ``p`` is the detection
        probability. Any other reading scores ``-inf``.

        Args:
            next_states: ``[N, state_size]`` post-transition states.
            actions: ``[N]`` action indices; unused.
            observations: ``[N, observation_size]`` readings.

        Returns:
            ``[N]`` log-likelihoods.
        """
        del actions
        count = next_states.shape[0]
        _, food, inside, scent_probs = self._sensor(next_states)
        terminal = self.terminal_mask(next_states)
        obs = observations.round().to(torch.int64)
        minus_inf = torch.full((count,), -math.inf, dtype=self.dtype, device=self.device)
        zero = torch.zeros(count, dtype=self.dtype, device=self.device)

        tag = obs[:, _OBS_TAG]
        terminal_score = torch.where(tag == OBSERVATION_TERMINAL, zero, minus_inf)

        body_match = (
            obs[:, _OBS_BODY_OFFSET:] == next_states[:, BODY_OFFSET:].round().to(torch.int64)
        ).all(dim=1)
        scent = obs[:, _OBS_SCENT]
        scent_valid = (scent >= 0) & (scent < len(SnakeQuadrant))
        log_scent_all = torch.log(scent_probs.to(self.dtype))
        log_scent = log_scent_all.gather(1, scent.clamp(0, len(SnakeQuadrant) - 1)[:, None])[:, 0]

        seen = obs[:, _OBS_SEEN_ROW : _OBS_SEEN_COL + 1]
        nothing_seen = seen[:, 0] < 0
        saw_food = inside & (seen == food).all(dim=1)
        log_miss = math.log1p(-self._detection) if self._detection < 1.0 else -math.inf
        log_hit = math.log(self._detection) if self._detection > 0.0 else -math.inf
        sighting = torch.where(
            nothing_seen,
            torch.where(inside, torch.full_like(zero, log_miss), zero),
            torch.where(saw_food, torch.full_like(zero, log_hit), minus_inf),
        )
        possible = (tag == OBSERVATION_LIVE) & body_match & scent_valid
        live_score = torch.where(possible, sighting + log_scent, minus_inf)
        return torch.where(terminal, terminal_score, live_score)

    def action_keys(self, actions: Tensor) -> Tensor:
        """Use the action index as its key."""
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        """Map each reading to an integer key that equal readings share.

        A live body is a chain of adjacent cells, so the key packs the head
        cell, the length and one of four directions per later segment in
        mixed radix, with the tag, scent and sighted cell. That packing is one
        to one while it fits in 63 bits (``target_length`` up to 22 on a
        12x12 grid); past that the int64 arithmetic wraps and two readings can
        share a key. The terminal reading has key 0; live readings start at 1.

        Args:
            observations: ``[N, observation_size]`` readings.

        Returns:
            ``[N]`` int64 keys.
        """
        obs = observations.round().to(torch.int64)
        cells = self._grid * self._grid
        scent = obs[:, _OBS_SCENT].clamp(0, len(SnakeQuadrant) - 1)
        seen_cell = torch.where(
            obs[:, _OBS_SEEN_ROW] < 0,
            torch.zeros_like(scent),
            1 + obs[:, _OBS_SEEN_ROW] * self._grid + obs[:, _OBS_SEEN_COL],
        )
        body = obs[:, _OBS_BODY_OFFSET:].reshape(obs.shape[0], self._target, 2)
        length = (body[:, :, 0] >= 0).sum(dim=1)
        head_cell = (body[:, 0, 0] * self._grid + body[:, 0, 1]).clamp(0, cells - 1)

        key = scent
        key = key * (cells + 1) + seen_cell
        key = key * cells + head_cell
        key = key * (self._target + 1) + length
        steps = (body[:, :-1] - body[:, 1:]).clamp(-1, 1)
        step_codes = self._heading_index[(steps[:, :, 0] + 1) * 3 + (steps[:, :, 1] + 1)]
        in_body = self._slots[1:][None, :] < length[:, None]
        step_codes = torch.where(in_body, step_codes, torch.zeros_like(step_codes))
        for segment in range(self._target - 1):
            key = key * len(DIRECTIONS) + step_codes[:, segment]
        live = obs[:, _OBS_TAG] == OBSERVATION_LIVE
        return torch.where(live, key + 1, torch.zeros_like(key))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _outcome(
        self, states: Tensor, actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Resolve the deterministic half of a step, as ``transition_outcome`` does.

        Returns:
            ``(body, new_len, eat, counter, status, food)``: the ``[N, T, 2]``
            int64 body after the step with ``-1`` past ``new_len``, the new
            length, whether the step ate, the new steps-since-food counter, the
            new termination status and the ``[N, 2]`` food before the step.
            Only live rows are meaningful.
        """
        count = states.shape[0]
        ints = states.round().to(torch.int64)
        old_body = ints[:, BODY_OFFSET:].reshape(count, self._target, 2)
        length = ints[:, LENGTH_INDEX]
        food = ints[:, FOOD_ROW_INDEX : FOOD_COL_INDEX + 1]
        head = old_body[:, 0]

        step = (old_body[:, 0] - old_body[:, 1]).clamp(-1, 1)
        heading = self._heading_index[(step[:, 0] + 1) * 3 + (step[:, 1] + 1)]
        new_heading = (heading + self._turn[actions.to(torch.int64)]) % len(DIRECTIONS)
        target = head + self._directions[new_heading]
        eat = (target == food).all(dim=1)

        # Shift the body one slot back behind the new head. Eating keeps the old
        # tail, so the length grows by one; otherwise the tail slot is cleared.
        new_len = length + eat.to(torch.int64)
        body = torch.cat([target[:, None, :], old_body[:, :-1]], dim=1)
        keep = self._slots[None, :] < new_len[:, None]
        body = torch.where(keep[:, :, None], body, torch.full_like(body, int(EMPTY)))

        counter = torch.where(eat, torch.zeros_like(length), ints[:, STEPS_SINCE_FOOD_INDEX] + 1)
        off_grid = (target < 0).any(dim=1) | (target >= self._grid).any(dim=1)
        rest = keep & (self._slots[None, :] >= 1)
        self_hit = ((body == target[:, None, :]).all(dim=2) & rest).any(dim=1)

        status = torch.full_like(length, int(SnakeTermination.RUNNING))
        status = torch.where(counter >= self._starvation, int(SnakeTermination.STARVATION), status)
        status = torch.where(new_len >= self._target, int(SnakeTermination.WIN), status)
        status = torch.where(self_hit, int(SnakeTermination.SELF), status)
        status = torch.where(off_grid, int(SnakeTermination.WALL), status)
        return body, new_len, eat, counter, status, food

    def _draw_free_cells(self, body: Tensor, new_len: Tensor) -> Tensor:
        """Draw one cell per row uniformly from the cells ``body`` leaves free.

        Rows with no free cell get cell 0; only a body that fills the grid has
        none, and such a body is a win, which does not respawn.

        Returns:
            ``[N, 2]`` int64 ``(row, col)`` cells.
        """
        count = body.shape[0]
        cells = self._grid * self._grid
        in_grid = (
            (self._slots[None, :] < new_len[:, None])
            & (body >= 0).all(dim=2)
            & (body < self._grid).all(dim=2)
        )
        flat = (body[:, :, 0] * self._grid + body[:, :, 1]).clamp(0, cells - 1)
        occupied = torch.zeros(count, cells, dtype=torch.int64, device=self.device)
        occupied.scatter_add_(1, flat, in_grid.to(torch.int64))
        free = occupied == 0
        free_count = free.sum(dim=1)
        pick = torch.floor(
            torch.rand(count, dtype=torch.float64, device=self.device)
            * free_count.to(torch.float64)
        ).to(torch.int64)
        pick = torch.minimum(pick, (free_count - 1).clamp(min=0))
        # Index of the (pick + 1)-th free cell.
        cumulative = free.to(torch.int64).cumsum(dim=1)
        cell = (cumulative <= pick[:, None]).sum(dim=1).clamp(max=cells - 1)
        return torch.stack([cell // self._grid, cell % self._grid], dim=1)

    def _sensor(self, next_states: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Head, food, window test and scent law of each row.

        Returns:
            ``(head, food, inside, scent_probs)``: ``[N, 2]`` int64 head and
            food cells, ``[N]`` whether the food is inside the vision window,
            and ``[N, 4]`` quadrant probabilities indexed by
            :class:`SnakeQuadrant`. Food on the head, which no live state has,
            treats all four quadrants as correct.
        """
        ints = next_states.round().to(torch.int64)
        head = ints[:, BODY_OFFSET : BODY_OFFSET + 2]
        food = ints[:, FOOD_ROW_INDEX : FOOD_COL_INDEX + 1]
        offset = food - head
        # The food is always on the grid, so the window test is the Chebyshev
        # distance alone; the scalar window drops off-grid cells, which the
        # food is never on.
        inside = offset.abs().max(dim=1).values <= self._radius

        sign = offset.sign()
        signs = self._quadrant_signs[None, :, :]
        compatible = ((sign[:, None, 0] == 0) | (sign[:, None, 0] == signs[:, :, 0])) & (
            (sign[:, None, 1] == 0) | (sign[:, None, 1] == signs[:, :, 1])
        )
        n_right = compatible.sum(dim=1, keepdim=True).to(self.dtype)
        n_wrong = (len(SnakeQuadrant) - n_right).clamp(min=1.0)
        right = self._accuracy / n_right
        wrong = (1.0 - self._accuracy) / n_wrong
        scent_probs = torch.where(compatible, right, wrong)
        return head, food, inside, scent_probs
