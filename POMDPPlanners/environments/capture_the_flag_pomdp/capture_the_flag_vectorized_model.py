# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for the CaptureTheFlag POMDP.

This module provides :class:`CaptureTheFlagVectorizedModel`, a batched
implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for :class:`~POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_pomdp.CaptureTheFlagPOMDP`,
so VOPP can plan on it.

Every rule parameter is read from a live environment instance; only the
kernels are re-expressed in torch. Each row carries its own joint action, so
the per-player actions are decoded per row. The kernels loop over *players*
(the team sizes are small and fixed) and run every step of those loops as one
tensor operation across the batch. The loops follow the environment's order,
because the order is the game: blue moves, then red picks a target from where
blue landed and moves, then pick-up, tagging (red first, then blue, each tagger
tagging at most one opponent), scoring against the pre-scoring carrier ids,
and the counters, decremented before any stage sets a fresh one.

Every configuration the environment accepts is modeled; nothing is declined.

The draws are made in a different order from the scalar environment's, so the
two agree in distribution, not draw for draw.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_pomdp import (
    CaptureTheFlagPOMDP,
)
from POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_pomdp_utils import (
    ACTION_DELTAS,
    ACTION_SCAN,
    N_PLAYER_ACTIONS,
    PERPENDICULAR,
    RedRole,
)

# The sentinel every terminal state emits, mirrored from the environment.
_TERMINAL_OBSERVATION_VALUE = -1.0

# The five cells a red player may step to, itself first, in the environment's
# own neighbour order.
_RED_STEPS = ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0))

# Number of per-player move actions (north, east, south, west); ids 4 and 5
# (stay, scan) do not move.
_N_MOVE_ACTIONS = 4


class CaptureTheFlagVectorizedModel:  # pylint: disable=too-many-instance-attributes
    """Batched torch generative model for the CaptureTheFlag POMDP.

    States are the environment's flat state vectors (see
    :class:`~POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_pomdp_utils.StateLayout`);
    actions are joint action ids in ``[0, 6 ** n_blue)``; observations are the
    environment's flat observation tuples as rows, with the all ``-1`` sentinel
    for terminal states.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype of state, observation and reward tensors.
        num_actions: Number of joint actions, ``6 ** n_blue``.
        observation_size: Length of an observation row.

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.capture_the_flag_pomdp import CaptureTheFlagPOMDP
        >>> from POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_vectorized_model import (  # noqa: E501
        ...     CaptureTheFlagVectorizedModel,
        ... )
        >>> env = CaptureTheFlagPOMDP(discount_factor=0.95)
        >>> model = CaptureTheFlagVectorizedModel(env, device=torch.device("cpu"))
        >>> state = torch.as_tensor(env.initial_state_dist().sample()[0], dtype=model.dtype)
        >>> states = state.repeat(3, 1)
        >>> actions = torch.tensor([1, 7, 35])
        >>> next_states = model.sample_next_states(states, actions)
        >>> observations = model.sample_observations(next_states, actions)
        >>> rewards = model.rewards(states, actions, next_states)
        >>> tuple(next_states.shape), tuple(observations.shape), tuple(rewards.shape)
        ((3, 21), (3, 16), (3,))
    """

    def __init__(
        self,
        env: CaptureTheFlagPOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Build the model from a live environment instance.

        Args:
            env: The environment whose rules are mirrored.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for state, observation and reward tensors.
        """
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self._build_rules(env)
        self._build_geometry(env)
        self._build_rewards(env)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    def _build_rules(self, env: CaptureTheFlagPOMDP) -> None:
        self._layout = env.layout
        self._n_blue = int(env.n_blue)
        self._n_red = int(env.n_red)
        self.num_actions = int(len(env.get_actions()))
        self.observation_size = int(env.observation_size)
        self._width, self._height = int(env.grid_size[0]), int(env.grid_size[1])
        self._midline = int(env.midline)
        self._red_is_attacker = [role is RedRole.ATTACK for role in env.red_roles]
        self._red_alert_radius = int(env.red_alert_radius)
        self._red_pursuit_probability = float(env.red_pursuit_probability)
        self._slip_probability = float(env.slip_probability)
        self._freeze_steps = int(env.freeze_steps)
        self._tagger_cooldown_steps = int(env.tagger_cooldown_steps)
        self._score_to_win = int(env.score_to_win)
        self._range_error_probability = float(env.range_error_probability)
        self._max_range = int(env.max_range)
        self._half_distance_move = float(env.detector_half_distance_move)
        self._half_distance_scan = float(env.detector_half_distance_scan)
        # Base of the observation-key polynomial: one more than the largest
        # shifted component value, so distinct observations get distinct keys
        # until the polynomial overflows int64 and wraps into a hash.
        self._key_base = (
            max(
                self._width,
                self._height,
                self._max_range + 1,
                self._freeze_steps + 1,
                self._score_to_win + 1,
                self._n_blue + 1,
            )
            + 2
        )

    def _build_geometry(self, env: CaptureTheFlagPOMDP) -> None:
        free = np.zeros((self._width, self._height), dtype=bool)
        for cell in env.free_cells():
            free[int(cell[0]), int(cell[1])] = True
        self._free = torch.as_tensor(free, device=self.device)
        self._blue_base = self._long(env.blue_base)
        self._red_base = self._long(env.red_base)
        self._blue_flag_cell = self._long(env.blue_flag_cell)
        self._flag_candidates = self._long(env.red_flag_candidates)
        # One guard post per flag candidate: a pure function of the candidate.
        self._guard_posts = self._long([env.guard_post(cell) for cell in env.red_flag_candidates])

        deltas = np.array([ACTION_DELTAS[a] for a in range(N_PLAYER_ACTIONS)], dtype=np.int64)
        self._action_deltas = self._long(deltas)
        # Perpendicular slip directions per player action; a non-move maps to
        # itself and is never displaced anyway.
        left = [PERPENDICULAR.get(a, (a, a))[0] for a in range(N_PLAYER_ACTIONS)]
        right = [PERPENDICULAR.get(a, (a, a))[1] for a in range(N_PLAYER_ACTIONS)]
        self._slip_left = self._long(left)
        self._slip_right = self._long(right)
        self._red_steps = self._long(_RED_STEPS)
        self._player_radix = self._long([N_PLAYER_ACTIONS**i for i in range(self._n_blue)])

    def _build_rewards(self, env: CaptureTheFlagPOMDP) -> None:
        self._capture_reward = float(env.capture_reward)
        self._concede_penalty = float(env.concede_penalty)
        self._tagged_penalty = float(env.tagged_penalty)
        self._tag_reward = float(env.tag_reward)
        self._pickup_reward = float(env.pickup_reward)
        self._move_cost = float(env.move_cost)
        self._scan_cost = float(env.scan_cost)

    def _long(self, values: object) -> Tensor:
        return torch.as_tensor(np.asarray(values, dtype=np.int64), device=self.device)

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        """Sample one successor per row; terminal rows come back unchanged.

        Args:
            states: ``[N, ds]`` current states.
            actions: ``[N]`` joint action ids.

        Returns:
            ``[N, ds]`` successor states.
        """
        ints = states.round().to(torch.int64)
        player_actions = self._player_actions(actions)
        blue = self._draw_blue_cells(ints, player_actions)
        red = self._draw_red_cells(ints, blue)
        successors = self._apply_deterministic_stages(ints, blue, red).to(states.dtype)
        terminal = self._terminal_rows(ints)
        return torch.where(terminal[:, None], states, successors)

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        """Sample one observation per row; terminal rows emit the sentinel.

        Args:
            next_states: ``[N, ds]`` successor states.
            actions: ``[N]`` joint action ids, which set each player's
                detector range.

        Returns:
            ``[N, do]`` observations.
        """
        ints = next_states.round().to(torch.int64)
        count = ints.shape[0]
        ranges, flag_distances = self._true_distances(ints)

        # Range readings: off by one either way with probability e / 2 each,
        # then clipped to [0, max_range]. Clipping after the draw gives the
        # same law as the environment's table, which folds the clipped mass
        # onto the end point.
        error = self._range_error_probability
        draws = torch.rand(ranges.shape, dtype=torch.float64, device=self.device)
        offset = torch.where(
            draws < error / 2.0,
            torch.full_like(ranges, -1),
            torch.where(
                draws < error / 2.0 + (1.0 - error),
                torch.zeros_like(ranges),
                torch.ones_like(ranges),
            ),
        )
        readings = torch.clamp(ranges + offset, 0, self._max_range)

        detection = self._detection_probability(flag_distances, self._player_actions(actions))
        bits = torch.rand(detection.shape, dtype=torch.float64, device=self.device) < detection

        observations = torch.cat(
            [
                self._observed_prefix(ints).to(self.dtype),
                readings.reshape(count, -1).to(self.dtype),
                bits.to(self.dtype),
                self._observed_suffix(ints).to(self.dtype),
            ],
            dim=1,
        )
        sentinel = torch.full_like(observations, _TERMINAL_OBSERVATION_VALUE)
        return torch.where(self._terminal_rows(ints)[:, None], sentinel, observations)

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        """Return the reward of each transition; zero from a terminal state.

        Args:
            states: ``[N, ds]`` states the step was taken from.
            actions: ``[N]`` joint action ids.
            next_states: ``[N, ds]`` realised successors.

        Returns:
            ``[N]`` rewards.
        """
        layout = self._layout
        before = states.round().to(torch.int64)
        after = next_states.round().to(torch.int64)
        score_blue = (after[:, layout.score_blue] - before[:, layout.score_blue]).to(self.dtype)
        score_red = (after[:, layout.score_red] - before[:, layout.score_red]).to(self.dtype)
        picked_up = (before[:, layout.carrier_red_flag] == 0) & (
            after[:, layout.carrier_red_flag] != 0
        )
        suffered = self._fresh_tags(before, after, layout.freeze_blue, self._n_blue)
        inflicted = self._fresh_tags(before, after, layout.freeze_red, self._n_red)
        scans = (self._player_actions(actions) == ACTION_SCAN).sum(dim=1).to(self.dtype)
        action_cost = scans * self._scan_cost + (self._n_blue - scans) * self._move_cost
        reward = (
            self._capture_reward * score_blue
            - self._concede_penalty * score_red
            - self._tagged_penalty * suffered.to(self.dtype)
            + self._tag_reward * inflicted.to(self.dtype)
            + self._pickup_reward * picked_up.to(self.dtype)
            - action_cost
        )
        return torch.where(self._terminal_rows(before), torch.zeros_like(reward), reward)

    def terminal_mask(self, states: Tensor) -> Tensor:
        """Return whether either side has reached the winning score, per row.

        Args:
            states: ``[N, ds]`` states.

        Returns:
            ``[N]`` boolean terminal flags.
        """
        return self._terminal_rows(states.round().to(torch.int64))

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        """Return ``log Z(o | s', a)`` per row.

        The exactly-observed components act as a delta factor: a row that
        disagrees with them scores ``-inf``. A fractional range reading or a
        detector bit other than 0 or 1 is impossible, not approximate.

        Args:
            next_states: ``[N, ds]`` successor states.
            actions: ``[N]`` joint action ids.
            observations: ``[N, do]`` observations to score.

        Returns:
            ``[N]`` log-likelihoods.
        """
        ints = next_states.round().to(torch.int64)
        values = observations.to(torch.float64)
        count = ints.shape[0]
        n_prefix = 2 * self._n_blue
        n_pairs = self._n_blue * self._n_red
        prefix = values[:, :n_prefix]
        readings = values[:, n_prefix : n_prefix + n_pairs]
        bits = values[:, n_prefix + n_pairs : n_prefix + n_pairs + self._n_blue]
        suffix = values[:, n_prefix + n_pairs + self._n_blue :]

        exact = (prefix == self._observed_prefix(ints).to(torch.float64)).all(dim=1)
        exact &= (suffix == self._observed_suffix(ints).to(torch.float64)).all(dim=1)

        ranges, flag_distances = self._true_distances(ints)
        ranges = ranges.reshape(count, -1)
        error = self._range_error_probability
        range_probability = torch.zeros(readings.shape, dtype=torch.float64, device=self.device)
        for offset, weight in ((0, 1.0 - error), (-1, error / 2.0), (1, error / 2.0)):
            clipped = torch.clamp(ranges + offset, 0, self._max_range).to(torch.float64)
            range_probability = range_probability + (clipped == readings).to(torch.float64) * weight
        # A fractional reading never equals an integer clipped distance, so it
        # already scores zero above.

        detection = self._detection_probability(flag_distances, self._player_actions(actions))
        valid_bits = ((bits == 0.0) | (bits == 1.0)).all(dim=1)
        bit_probability = torch.where(bits == 1.0, detection, 1.0 - detection)

        # log(0) is -inf in torch, with no warning to silence.
        log_probability = torch.log(range_probability).sum(dim=1) + torch.log(bit_probability).sum(
            dim=1
        )
        neg_inf = torch.full_like(log_probability, float("-inf"))
        live = torch.where(exact & valid_bits, log_probability, neg_inf)

        is_sentinel = (values == _TERMINAL_OBSERVATION_VALUE).all(dim=1)
        terminal = torch.where(is_sentinel, torch.zeros_like(live), neg_inf)
        return torch.where(self._terminal_rows(ints), terminal, live).to(self.dtype)

    def action_keys(self, actions: Tensor) -> Tensor:
        """Return the joint action ids as int64 tree keys."""
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        """Map observations to int64 tree keys.

        Every component is an integer at least ``-1``, so the key is the
        observation read as digits of base :attr:`_key_base` after shifting by
        two. The key is exact while the polynomial fits in int64; beyond that
        the multiplication wraps and the key becomes a hash.

        Args:
            observations: ``[N, do]`` observations.

        Returns:
            ``[N]`` int64 keys.
        """
        digits = observations.round().to(torch.int64) + 2
        key = torch.zeros(observations.shape[0], dtype=torch.int64, device=observations.device)
        for column in range(digits.shape[1]):
            key = key * self._key_base + digits[:, column]
        return key

    # ------------------------------------------------------------------ #
    # State views
    # ------------------------------------------------------------------ #

    def _player_actions(self, actions: Tensor) -> Tensor:
        """``[N, n_blue]`` per-player action ids from joint action ids."""
        joint = actions.to(torch.int64).reshape(-1, 1)
        return torch.remainder(
            torch.div(joint, self._player_radix, rounding_mode="floor"), N_PLAYER_ACTIONS
        )

    def _blue_cells(self, ints: Tensor) -> Tensor:
        base = self._layout.blue_pos
        return ints[:, base : base + 2 * self._n_blue].reshape(-1, self._n_blue, 2)

    def _red_cells(self, ints: Tensor) -> Tensor:
        base = self._layout.red_pos
        return ints[:, base : base + 2 * self._n_red].reshape(-1, self._n_red, 2)

    def _flag_index(self, ints: Tensor) -> Tensor:
        return ints[:, self._layout.flag_cell]

    def _terminal_rows(self, ints: Tensor) -> Tensor:
        layout = self._layout
        return (ints[:, layout.score_blue] >= self._score_to_win) | (
            ints[:, layout.score_red] >= self._score_to_win
        )

    def _block(self, ints: Tensor, start: int, width: int) -> Tensor:
        return ints[:, start : start + width]

    def _is_free(self, cells: Tensor) -> Tensor:
        """Whether each ``[..., 2]`` cell is inside the field and not a tree."""
        x, y = cells[..., 0], cells[..., 1]
        inside = (x >= 0) & (x < self._width) & (y >= 0) & (y < self._height)
        lookup = self._free[x.clamp(0, self._width - 1), y.clamp(0, self._height - 1)]
        return inside & lookup

    # ------------------------------------------------------------------ #
    # Movement
    # ------------------------------------------------------------------ #

    def _draw_blue_cells(self, ints: Tensor, player_actions: Tensor) -> Tensor:
        """Where each blue player lands, drawn per row.

        A move goes as intended with probability ``1 - slip`` and slips to each
        perpendicular direction with probability ``slip / 2``; a move into a
        tree or off the field leaves the player where it stood. Stay, scan and
        a frozen player do not move.
        """
        cells = self._blue_cells(ints).clone()
        frozen = self._block(ints, self._layout.freeze_blue, self._n_blue) > 0
        slip = self._slip_probability
        for i in range(self._n_blue):
            intended = player_actions[:, i]
            draws = torch.rand(intended.shape, dtype=torch.float64, device=self.device)
            chosen = torch.where(
                draws < 1.0 - slip,
                intended,
                torch.where(
                    draws < 1.0 - slip / 2.0, self._slip_left[intended], self._slip_right[intended]
                ),
            )
            target = cells[:, i, :] + self._action_deltas[chosen]
            stays = ~self._is_free(target) | frozen[:, i] | (intended >= _N_MOVE_ACTIONS)
            cells[:, i, :] = torch.where(stays[:, None], cells[:, i, :], target)
        return cells

    def _draw_red_cells(self, ints: Tensor, blue: Tensor) -> Tensor:
        """Where each red player lands, drawn per row.

        A red player takes one of its free neighbours or stays: with
        probability ``red_pursuit_probability`` one that closes on its target,
        otherwise uniformly among all of them.
        """
        cells = self._red_cells(ints).clone()
        frozen = self._block(ints, self._layout.freeze_red, self._n_red) > 0
        carrier_blue = ints[:, self._layout.carrier_blue_flag]
        pursue = self._red_pursuit_probability
        for j in range(self._n_red):
            target = self._red_target(j, blue, cells[:, j, :], ints, carrier_blue)
            options = cells[:, j, None, :] + self._red_steps[None, :, :]
            legal = self._is_free(options)
            # The cell itself is always an option: the environment lists it
            # before testing any neighbour.
            legal[:, 0] = True
            distances = (options - target[:, None, :]).abs().sum(dim=-1)
            masked = torch.where(legal, distances, torch.full_like(distances, 1 << 30))
            best = masked.min(dim=1, keepdim=True).values
            closing = legal & (distances == best)
            legal_f = legal.to(torch.float64)
            closing_f = closing.to(torch.float64)
            weights = legal_f * (1.0 - pursue) / legal_f.sum(dim=1, keepdim=True)
            weights = weights + closing_f * pursue / closing_f.sum(dim=1, keepdim=True)
            picked = torch.multinomial(weights, 1).squeeze(1)
            drawn = options[torch.arange(options.shape[0], device=self.device), picked]
            cells[:, j, :] = torch.where(frozen[:, j, None], cells[:, j, :], drawn)
        return cells

    def _red_target(
        self, red_index: int, blue: Tensor, red_cell: Tensor, ints: Tensor, carrier_blue: Tensor
    ) -> Tensor:
        """The cell one red player heads for this step, per row.

        Carrying beats the role: whoever holds the blue flag runs it home.
        An attacker runs for the blue flag. A defender chases the nearest blue
        intruder in the red half once it is within the alert radius, ties
        broken by ``(distance, x, y)`` as the environment's ``min`` does, and
        otherwise stands on its guard post.
        """
        count = blue.shape[0]
        if self._red_is_attacker[red_index]:
            role_target = self._blue_flag_cell.expand(count, 2)
        else:
            intruder = blue[:, :, 0] > self._midline
            distance = (blue - red_cell[:, None, :]).abs().sum(dim=-1)
            key = (distance * self._width + blue[:, :, 0]) * self._height + blue[:, :, 1]
            key = torch.where(intruder, key, torch.full_like(key, torch.iinfo(torch.int64).max))
            nearest = key.argmin(dim=1)
            rows = torch.arange(count, device=self.device)
            chosen = blue[rows, nearest]
            chase = intruder.any(dim=1) & (distance[rows, nearest] <= self._red_alert_radius)
            posts = self._guard_posts[self._flag_index(ints)]
            role_target = torch.where(chase[:, None], chosen, posts)
        carrying = (carrier_blue == red_index + 1)[:, None]
        return torch.where(carrying, self._red_base.expand(count, 2), role_target)

    # ------------------------------------------------------------------ #
    # Deterministic stages
    # ------------------------------------------------------------------ #

    def _apply_deterministic_stages(  # pylint: disable=too-many-locals
        self, ints: Tensor, blue: Tensor, red: Tensor
    ) -> Tensor:
        """Resolve pick-up, tagging, scoring and the counters for every row.

        Args:
            ints: ``[N, ds]`` int64 states the step was taken from.
            blue: ``[N, n_blue, 2]`` blue cells after the blue move.
            red: ``[N, n_red, 2]`` red cells after the red move.

        Returns:
            ``[N, ds]`` int64 successor states.
        """
        layout = self._layout
        nb, nr = self._n_blue, self._n_red
        count = ints.shape[0]
        old_freeze_blue = self._block(ints, layout.freeze_blue, nb)
        old_freeze_red = self._block(ints, layout.freeze_red, nr)
        old_cooldown_blue = self._block(ints, layout.cooldown_blue, nb)
        old_cooldown_red = self._block(ints, layout.cooldown_red, nr)
        counters = {
            "freeze_blue": (old_freeze_blue - 1).clamp(min=0),
            "freeze_red": (old_freeze_red - 1).clamp(min=0),
            "cooldown_blue": (old_cooldown_blue - 1).clamp(min=0),
            "cooldown_red": (old_cooldown_red - 1).clamp(min=0),
        }
        was_frozen_blue = old_freeze_blue > 0
        was_frozen_red = old_freeze_red > 0

        flag_cell = self._flag_candidates[self._flag_index(ints)]
        carrier_red = self._pick_up(
            ints[:, layout.carrier_red_flag], blue, flag_cell, was_frozen_blue
        )
        carrier_blue = self._pick_up(
            ints[:, layout.carrier_blue_flag],
            red,
            self._blue_flag_cell.expand(count, 2),
            was_frozen_red,
        )

        blue, red, carrier_red, carrier_blue = self._tagging(
            blue,
            red,
            (old_cooldown_blue, old_cooldown_red),
            (was_frozen_blue, was_frozen_red),
            counters,
            (carrier_red, carrier_blue),
        )

        blue_scores = (
            (carrier_red != 0)
            & _at_cell(blue, carrier_red - 1, self._blue_base)
            & (carrier_blue == 0)
        )
        red_scores = (
            (carrier_blue != 0)
            & _at_cell(red, carrier_blue - 1, self._red_base)
            & (carrier_red == 0)
        )
        carrier_red = torch.where(blue_scores, torch.zeros_like(carrier_red), carrier_red)
        carrier_blue = torch.where(red_scores, torch.zeros_like(carrier_blue), carrier_blue)

        return torch.cat(
            [
                blue.reshape(count, -1),
                red.reshape(count, -1),
                ints[:, layout.flag_cell, None],
                carrier_red[:, None],
                carrier_blue[:, None],
                counters["freeze_blue"],
                counters["freeze_red"],
                counters["cooldown_blue"],
                counters["cooldown_red"],
                (ints[:, layout.score_blue] + blue_scores.to(torch.int64))[:, None],
                (ints[:, layout.score_red] + red_scores.to(torch.int64))[:, None],
            ],
            dim=1,
        )

    @staticmethod
    def _pick_up(carrier: Tensor, cells: Tensor, flag_cell: Tensor, was_frozen: Tensor) -> Tensor:
        """Give the flag to the lowest-indexed unfrozen player standing on it."""
        updated = carrier.clone()
        for index in range(cells.shape[1]):
            on_flag = (cells[:, index, :] == flag_cell).all(dim=1) & ~was_frozen[:, index]
            updated = torch.where(
                (updated == 0) & on_flag, torch.full_like(updated, index + 1), updated
            )
        return updated

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
    def _tagging(
        self,
        blue: Tensor,
        red: Tensor,
        old_cooldowns: Tuple[Tensor, Tensor],
        was_frozen: Tuple[Tensor, Tensor],
        counters: dict,
        carriers: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Send tagged players home, in the environment's own order.

        Red tags first, then blue; each tagger tags at most one opponent. A
        player tagged earlier in the stage has already been sent home and
        cannot be tagged again, nor tag from its base.
        """
        blue, red = blue.clone(), red.clone()
        carrier_red, carrier_blue = carriers
        old_cooldown_blue, old_cooldown_red = old_cooldowns
        was_frozen_blue, was_frozen_red = was_frozen
        count = blue.shape[0]
        no_one = torch.zeros(count, dtype=torch.bool, device=self.device)
        tagged_blue = [no_one.clone() for _ in range(self._n_blue)]
        tagged_red = [no_one.clone() for _ in range(self._n_red)]

        for j in range(self._n_red):
            free_tagger = ~was_frozen_red[:, j] & (old_cooldown_red[:, j] <= 0)
            done = no_one.clone()
            for i in range(self._n_blue):
                tags = (
                    free_tagger
                    & ~done
                    & (blue[:, i, :] == red[:, j, :]).all(dim=1)
                    & (blue[:, i, 0] > self._midline)
                    & ~was_frozen_blue[:, i]
                    & ~tagged_blue[i]
                )
                blue[:, i, :] = torch.where(tags[:, None], self._blue_base, blue[:, i, :])
                counters["freeze_blue"][:, i] = torch.where(
                    tags, self._freeze_steps, counters["freeze_blue"][:, i]
                )
                counters["cooldown_red"][:, j] = torch.where(
                    tags, self._tagger_cooldown_steps, counters["cooldown_red"][:, j]
                )
                carrier_red = torch.where(
                    tags & (carrier_red == i + 1), torch.zeros_like(carrier_red), carrier_red
                )
                tagged_blue[i] = tagged_blue[i] | tags
                done = done | tags

        for i in range(self._n_blue):
            free_tagger = ~was_frozen_blue[:, i] & ~tagged_blue[i] & (old_cooldown_blue[:, i] <= 0)
            done = no_one.clone()
            for j in range(self._n_red):
                tags = (
                    free_tagger
                    & ~done
                    & (red[:, j, :] == blue[:, i, :]).all(dim=1)
                    & (red[:, j, 0] < self._midline)
                    & ~was_frozen_red[:, j]
                    & ~tagged_red[j]
                )
                red[:, j, :] = torch.where(tags[:, None], self._red_base, red[:, j, :])
                counters["freeze_red"][:, j] = torch.where(
                    tags, self._freeze_steps, counters["freeze_red"][:, j]
                )
                counters["cooldown_blue"][:, i] = torch.where(
                    tags, self._tagger_cooldown_steps, counters["cooldown_blue"][:, i]
                )
                carrier_blue = torch.where(
                    tags & (carrier_blue == j + 1), torch.zeros_like(carrier_blue), carrier_blue
                )
                tagged_red[j] = tagged_red[j] | tags
                done = done | tags

        return blue, red, carrier_red, carrier_blue

    # ------------------------------------------------------------------ #
    # Observation and reward helpers
    # ------------------------------------------------------------------ #

    def _true_distances(self, ints: Tensor) -> Tuple[Tensor, Tensor]:
        """``[N, n_blue, n_red]`` range distances and ``[N, n_blue]`` flag distances."""
        blue = self._blue_cells(ints)
        red = self._red_cells(ints)
        ranges = (blue[:, :, None, :] - red[:, None, :, :]).abs().sum(dim=-1)
        flag_cell = self._flag_candidates[self._flag_index(ints)]
        flag_distances = (blue - flag_cell[:, None, :]).abs().sum(dim=-1)
        return ranges, flag_distances

    def _detection_probability(self, flag_distances: Tensor, player_actions: Tensor) -> Tensor:
        """Probability each player's flag detector fires, ``[N, n_blue]`` float64."""
        half = torch.where(
            player_actions == ACTION_SCAN, self._half_distance_scan, self._half_distance_move
        ).to(torch.float64)
        return 0.5 * (1.0 + torch.pow(2.0, -flag_distances.to(torch.float64) / half))

    def _observed_prefix(self, ints: Tensor) -> Tensor:
        """Blue positions, observed exactly."""
        return ints[:, self._layout.blue_pos : self._layout.blue_pos + 2 * self._n_blue]

    def _observed_suffix(self, ints: Tensor) -> Tensor:
        """Carrier ids (blue's exactly, red's as a bit), blue freezes and both scores."""
        layout = self._layout
        return torch.cat(
            [
                ints[:, layout.carrier_red_flag, None],
                (ints[:, layout.carrier_blue_flag, None] != 0).to(torch.int64),
                self._block(ints, layout.freeze_blue, self._n_blue),
                ints[:, layout.score_blue, None],
                ints[:, layout.score_red, None],
            ],
            dim=1,
        )

    def _fresh_tags(self, before: Tensor, after: Tensor, start: int, width: int) -> Tensor:
        """How many players went from unfrozen to a full freeze, per row."""
        was_free = self._block(before, start, width) == 0
        fresh = self._block(after, start, width) == self._freeze_steps
        return (was_free & fresh).sum(dim=1)


def _at_cell(cells: Tensor, index: Tensor, cell: Tensor) -> Tensor:
    """Whether the ``index``-th player of each row stands on ``cell``.

    An index of ``-1`` (nobody carrying) is excluded by the caller's own mask;
    it is clamped here so the gather stays in range.
    """
    safe = index.clamp(0, cells.shape[1] - 1)
    chosen = cells[torch.arange(cells.shape[0], device=cells.device), safe]
    return (chosen == cell).all(dim=1)
