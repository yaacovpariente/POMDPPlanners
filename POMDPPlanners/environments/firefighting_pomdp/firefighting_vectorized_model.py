# SPDX-License-Identifier: MIT

"""Torch vectorized generative model for the firefighting POMDP, so VOPP can plan on it.

:class:`FirefightingVectorizedModel` implements
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for :class:`~POMDPPlanners.environments.firefighting_pomdp.firefighting_pomdp.FirefightingPOMDP`.
Each kernel takes a batch of rows and runs as torch tensor operations with no
per-row Python loop; the only Python loops run over the ``N`` firefighters and
the four compass directions, which are fixed by the configuration.

Every constant (grid size, obstacles, depot, probabilities, costs, step budget)
is read from the environment instance passed to the constructor, so the
environment stays the one place the configuration is set. The kernels follow
``FirefightingPOMDP._transition``, ``_draw_observation``,
``observation_log_probability``, ``reward`` and ``is_terminal`` step by step,
and the conformance suite in
``tests/test_environments/test_vectorized_model_conformance.py`` compares the
two.

The state row is the environment's own state vector,
``[step, (row, col, tank, health) x N, wind_direction, wind_strength, cells]``,
and the observation row is the environment's own observation vector,
``[(row, col, tank, health) x N, cells]`` with ``-1`` for an unseen cell. The
action is the joint action index, whose base-5 digits are the per-firefighter
actions.

Every configuration of the environment is supported, so the constructor
declines nothing.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.firefighting_pomdp.firefighting_pomdp import (
    UNKNOWN_CATEGORY,
    FirefightingPOMDP,
)
from POMDPPlanners.environments.firefighting_pomdp.firefighting_world import (
    DIRECTION_OFFSETS,
    FIREFIGHTER_FIELD_WIDTH,
    FIREFIGHTER_OFFSET,
    HEAT_DAMAGE,
    NUM_CATEGORIES,
    STEP_INDEX,
    FireCategory,
    FirefightingAction,
    WindStrength,
)

# Log-likelihood returned for an observation the state cannot produce. A finite
# floor rather than -inf, as in the other models, so a particle weight
# normalisation over rows that are all impossible gives a uniform weight
# instead of NaN. log(1e-300).
_LOG_PROB_FLOOR = -690.7755278982137

# Multiplier of the polynomial observation hash. Odd, so multiplication is a
# bijection modulo 2**64 and two observations that differ in one entry always
# get different keys.
_OBS_HASH_MULTIPLIER = 1_000_003

_UNBURNT = int(FireCategory.UNBURNT)
_SMOLDERING = int(FireCategory.SMOLDERING)
_BURNING = int(FireCategory.BURNING)
_BURNT = int(FireCategory.BURNT)
_WET = int(FireCategory.WET)
_SUPPRESS = int(FirefightingAction.SUPPRESS)


def _wrapping_powers(multiplier: int, count: int) -> np.ndarray:
    """Return ``multiplier ** (count - 1 - j)`` modulo ``2**64`` as signed int64.

    Computed with Python integers so the wrap is exact, then reinterpreted as
    the int64 a torch multiply would produce.
    """
    powers = []
    value = 1
    for _ in range(count):
        powers.append(value)
        value = (value * multiplier) % (1 << 64)
    powers.reverse()
    signed = [p - (1 << 64) if p >= (1 << 63) else p for p in powers]
    return np.asarray(signed, dtype=np.int64)


# pylint: disable-next=too-many-instance-attributes
class FirefightingVectorizedModel:
    """Batched torch generative model for :class:`FirefightingPOMDP`.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype of state, observation and reward tensors.
        num_actions: Number of joint actions, ``5 ** num_firefighters``.

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.firefighting_pomdp import FirefightingPOMDP
        >>> from POMDPPlanners.environments.firefighting_pomdp.firefighting_vectorized_model import (
        ...     FirefightingVectorizedModel,
        ... )
        >>> _ = torch.manual_seed(0)
        >>> env = FirefightingPOMDP(discount_factor=0.95)
        >>> model = FirefightingVectorizedModel(env, device=torch.device("cpu"))
        >>> state = torch.as_tensor(env.initial_state_dist().sample()[0]).unsqueeze(0)
        >>> states = state.repeat(4, 1).to(model.dtype)
        >>> actions = torch.tensor([0, 4, 24, 12])
        >>> next_states = model.sample_next_states(states, actions)
        >>> observations = model.sample_observations(next_states, actions)
        >>> rewards = model.rewards(states, actions, next_states)
        >>> tuple(next_states.shape), tuple(observations.shape), tuple(rewards.shape)
        ((4, 111), (4, 108), (4,))
    """

    def __init__(
        self,
        env: FirefightingPOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Build the model from an environment instance.

        Args:
            env: The environment whose configuration and kernels are mirrored.
            device: Target device. Defaults to CPU.
            dtype: Floating dtype for state, observation and reward tensors.
        """
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self.num_actions = int(env.num_actions)
        self._num_rows = int(env.num_rows)
        self._num_cols = int(env.num_cols)
        self._num_cells = int(env.num_cells)
        self._num_firefighters = int(env.num_firefighters)
        self._wind_direction_index = int(env.wind_direction_index)
        self._wind_strength_index = int(env.wind_strength_index)
        self._fire_offset = int(env.fire_offset)
        self._build_tables(env)
        self._build_parameters(env)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    def _build_tables(self, env: FirefightingPOMDP) -> None:
        long = {"dtype": torch.int64, "device": self.device}
        # Row a holds the per-firefighter actions of joint action a, from the
        # environment's own decoding table so the two cannot disagree.
        joint = np.stack([env.decode_action(a) for a in range(self.num_actions)])
        self._action_table = torch.as_tensor(joint, **long)
        self._offsets = torch.as_tensor(np.asarray(DIRECTION_OFFSETS), **long)
        self._obstacle = torch.as_tensor(env.obstacle_mask, dtype=torch.bool, device=self.device)
        self._depot_row, self._depot_col = int(env.depot_cell[0]), int(env.depot_cell[1])
        self._heat_damage = torch.as_tensor(np.asarray(HEAT_DAMAGE), **long)
        self._suppression = torch.tensor(
            [
                env.suppression_probability_unburnt,
                env.suppression_probability_smoldering,
                env.suppression_probability_burning,
                0.0,
                0.0,
            ],
            dtype=self.dtype,
            device=self.device,
        )
        self._rows = torch.arange(self._num_rows, **long)
        self._cols = torch.arange(self._num_cols, **long)
        width = FIREFIGHTER_FIELD_WIDTH * self._num_firefighters + self._num_cells
        self._obs_hash_powers = torch.as_tensor(
            _wrapping_powers(_OBS_HASH_MULTIPLIER, width), **long
        )

    def _build_parameters(self, env: FirefightingPOMDP) -> None:
        self._max_tank = int(env.max_tank)
        self._sensing_radius = int(env.sensing_radius)
        self._observation_error = float(env.observation_error_probability)
        self._slip = float(env.slip_probability)
        self._spread = float(env.spread_probability)
        self._gain_low = float(env.wind_gain_low)
        self._gain_high = float(env.wind_gain_high)
        self._crosswind_rate = self._spread * (1.0 - float(env.crosswind_attenuation))
        self._growth = float(env.growth_probability)
        self._burnout = float(env.burnout_probability)
        self._max_steps = int(env.max_steps)
        self._success_reward = float(env.success_reward)
        self._step_cost = float(env.step_cost)
        self._smoldering_cost = float(env.smoldering_cell_cost)
        self._burning_cost = float(env.burning_cell_cost)
        self._burnt_cost = float(env.burnt_cell_cost)
        self._damage_cost = float(env.damage_cost)
        self._water_cost = float(env.water_cost)
        self._disabled_terminal = bool(env.is_all_firefighters_disabled_terminal)

    # ------------------------------------------------------------------ #
    # State accessors
    # ------------------------------------------------------------------ #

    def _firefighters(self, states: Tensor) -> Tensor:
        """``[B, N, 4]`` int64 row, column, tank, health."""
        block = states[
            :,
            FIREFIGHTER_OFFSET : FIREFIGHTER_OFFSET
            + FIREFIGHTER_FIELD_WIDTH * self._num_firefighters,
        ]
        return (
            block.round()
            .to(torch.int64)
            .reshape(-1, self._num_firefighters, FIREFIGHTER_FIELD_WIDTH)
        )

    def _fire(self, states: Tensor) -> Tensor:
        """``[B, R, C]`` int64 category codes."""
        cells = states[:, self._fire_offset :]
        return cells.round().to(torch.int64).reshape(-1, self._num_rows, self._num_cols)

    def _sprayers(self, firefighters: Tensor, per_firefighter: Tensor) -> Tensor:
        """``[B, N]`` live firefighters that chose SUPPRESS with a non-empty tank."""
        return (
            (firefighters[:, :, 3] > 0)
            & (per_firefighter == _SUPPRESS)
            & (firefighters[:, :, 2] > 0)
        )

    @staticmethod
    def _alight(fire: Tensor) -> Tensor:
        return (fire == _SMOLDERING) | (fire == _BURNING)

    def _rand(self, shape: Tuple[int, ...]) -> Tensor:
        return torch.rand(shape, dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        """Draw one successor per row, in the order ``FirefightingPOMDP._transition`` uses.

        Motion, then suppression and the tank, then spread, then growth and
        burnout, then heat damage, then the step counter.
        """
        batch = states.shape[0]
        per_firefighter = self._action_table[actions.to(torch.int64)]
        firefighters = self._firefighters(states)
        fire = self._fire(states)

        positions = self._move(firefighters, per_firefighter, fire)
        sprayers = self._sprayers(firefighters, per_firefighter)
        fire = self._suppress(fire, positions, sprayers)
        tanks = firefighters[:, :, 2] - sprayers.to(torch.int64)
        at_depot = (positions[:, :, 0] == self._depot_row) & (positions[:, :, 1] == self._depot_col)
        tanks = torch.where(at_depot, torch.full_like(tanks, self._max_tank), tanks)

        after_suppression = fire
        fire = self._spread_fire(fire, states)
        fire = self._grow_and_burn_out(fire, after_suppression)

        flat_fire = fire.reshape(batch, self._num_cells)
        cell_index = positions[:, :, 0] * self._num_cols + positions[:, :, 1]
        damage = self._heat_damage[flat_fire.gather(1, cell_index)]
        healths = torch.clamp(firefighters[:, :, 3] - damage, min=0)

        block = torch.stack([positions[:, :, 0], positions[:, :, 1], tanks, healths], dim=2)
        nxt = states.clone()
        nxt[:, STEP_INDEX] = states[:, STEP_INDEX] + 1.0
        nxt[
            :,
            FIREFIGHTER_OFFSET : FIREFIGHTER_OFFSET
            + FIREFIGHTER_FIELD_WIDTH * self._num_firefighters,
        ] = block.reshape(batch, -1).to(self.dtype)
        nxt[:, self._fire_offset :] = flat_fire.to(self.dtype)
        return nxt

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        """Draw one reading per row: exact firefighter fields, noisy visible cells."""
        del actions
        batch = next_states.shape[0]
        visible = self._visible(self._firefighters(next_states)).reshape(batch, self._num_cells)
        truth = self._fire(next_states).reshape(batch, self._num_cells)
        wrong = self._rand((batch, self._num_cells)) < self._observation_error
        offsets = torch.randint(
            1, NUM_CATEGORIES, (batch, self._num_cells), dtype=torch.int64, device=self.device
        )
        reported = torch.where(wrong, (truth + offsets) % NUM_CATEGORIES, truth).to(self.dtype)
        reported = torch.where(visible, reported, torch.full_like(reported, UNKNOWN_CATEGORY))
        return torch.cat([self._firefighter_fields(next_states), reported], dim=1)

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        """Score each row as ``FirefightingPOMDP.reward`` does."""
        fire = self._fire(states)
        next_fire = self._fire(next_states)
        firefighters = self._firefighters(states)
        next_firefighters = self._firefighters(next_states)
        per_firefighter = self._action_table[actions.to(torch.int64)]

        smoldering = (next_fire == _SMOLDERING).sum(dim=(1, 2)).to(self.dtype)
        burning = (next_fire == _BURNING).sum(dim=(1, 2)).to(self.dtype)
        newly_burnt = ((next_fire == _BURNT) & (fire != _BURNT)).sum(dim=(1, 2)).to(self.dtype)
        health_lost = (firefighters[:, :, 3] - next_firefighters[:, :, 3]).sum(dim=1).to(self.dtype)
        sprays = self._sprayers(firefighters, per_firefighter).sum(dim=1).to(self.dtype)

        reward = torch.full_like(smoldering, -self._step_cost)
        reward = reward - self._smoldering_cost * smoldering
        reward = reward - self._burning_cost * burning
        reward = reward - self._burnt_cost * newly_burnt
        reward = reward - self._damage_cost * health_lost
        reward = reward - self._water_cost * sprays
        fire_out = (smoldering == 0) & (burning == 0)
        return reward + fire_out.to(self.dtype) * self._success_reward

    def terminal_mask(self, states: Tensor) -> Tensor:
        """Fire out, or every firefighter down (when that is terminal), or the step budget spent."""
        fire_out = ~self._alight(self._fire(states)).flatten(1).any(dim=1)
        terminal = fire_out | (states[:, STEP_INDEX].round() >= self._max_steps)
        if self._disabled_terminal:
            all_down = ~(self._firefighters(states)[:, :, 3] > 0).any(dim=1)
            terminal = terminal | all_down
        return terminal

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        """Score each reading as ``FirefightingPOMDP.observation_log_probability`` does.

        A reading the state cannot produce scores ``_LOG_PROB_FLOOR`` where the
        environment returns ``-inf``.
        """
        del actions
        batch = next_states.shape[0]
        fields = FIREFIGHTER_FIELD_WIDTH * self._num_firefighters
        expected = self._firefighter_fields(next_states)
        fields_match = (observations[:, :fields] == expected).all(dim=1)

        visible = self._visible(self._firefighters(next_states)).reshape(batch, self._num_cells)
        truth = self._fire(next_states).reshape(batch, self._num_cells).to(self.dtype)
        reported = observations[:, fields:]
        unseen_ok = (reported == UNKNOWN_CATEGORY) | visible
        category_ok = (reported >= 0) & (reported < NUM_CATEGORIES) & (reported == reported.round())
        seen_ok = category_ok | ~visible
        possible = fields_match & unseen_ok.all(dim=1) & seen_ok.all(dim=1)

        matches = (visible & (reported == truth)).sum(dim=1).to(self.dtype)
        mismatches = visible.sum(dim=1).to(self.dtype) - matches
        # Each term is added only when its count is non-zero: at an error
        # probability of 0 or 1 one of the two logs is -inf, and 0 * -inf is NaN.
        log_correct, log_wrong = self._confusion_logs()
        zero = torch.zeros_like(matches)
        score = torch.where(matches > 0, matches * log_correct, zero)
        score = score + torch.where(mismatches > 0, mismatches * log_wrong, zero)
        score = torch.clamp(score, min=_LOG_PROB_FLOOR)
        return torch.where(possible, score, torch.full_like(score, _LOG_PROB_FLOOR))

    def action_keys(self, actions: Tensor) -> Tensor:
        """The joint action index is the tree key."""
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        """Hash each reading to one int64 with a polynomial hash modulo ``2**64``.

        A reading has ``4N + R*C`` entries, too many for an exact mixed-radix
        key in 63 bits, so two different readings can share a key. Every entry
        is shifted by one so the unseen marker ``-1`` maps to 0.
        """
        coords = observations.round().to(torch.int64) + 1
        return (coords * self._obs_hash_powers).sum(dim=1)

    # ------------------------------------------------------------------ #
    # Transition internals
    # ------------------------------------------------------------------ #

    def _move(self, firefighters: Tensor, per_firefighter: Tensor, fire: Tensor) -> Tensor:
        """``[B, N, 2]`` post-motion cells.

        A firefighter moves when it is live, chose a move, the target is on the
        grid, not an obstacle and not burnt in the pre-step map, and the slip
        draw does not fire.
        """
        batch = firefighters.shape[0]
        flat_fire = fire.reshape(batch, self._num_cells)
        positions = firefighters[:, :, :2].clone()
        for firefighter in range(self._num_firefighters):
            action = per_firefighter[:, firefighter]
            offset = self._offsets[torch.clamp(action, max=len(DIRECTION_OFFSETS) - 1)]
            row = firefighters[:, firefighter, 0] + offset[:, 0]
            col = firefighters[:, firefighter, 1] + offset[:, 1]
            on_grid = (row >= 0) & (row < self._num_rows) & (col >= 0) & (col < self._num_cols)
            safe_row = torch.clamp(row, 0, self._num_rows - 1)
            safe_col = torch.clamp(col, 0, self._num_cols - 1)
            target_fire = flat_fire.gather(
                1, (safe_row * self._num_cols + safe_col).unsqueeze(1)
            ).squeeze(1)
            admissible = on_grid & ~self._obstacle[safe_row, safe_col] & (target_fire != _BURNT)
            moves = (
                (firefighters[:, firefighter, 3] > 0)
                & (action != _SUPPRESS)
                & admissible
                & (self._rand((batch,)) >= self._slip)
            )
            positions[:, firefighter, 0] = torch.where(moves, row, positions[:, firefighter, 0])
            positions[:, firefighter, 1] = torch.where(moves, col, positions[:, firefighter, 1])
        return positions

    def _suppress(self, fire: Tensor, positions: Tensor, sprayers: Tensor) -> Tensor:
        """Soak covered cells: each spray covers its own cell and four neighbours, counts add."""
        batch = fire.shape[0]
        counts = torch.zeros(batch, self._num_cells, dtype=torch.int64, device=self.device)
        stencil = [(0, 0), *DIRECTION_OFFSETS]
        for firefighter in range(self._num_firefighters):
            spraying = sprayers[:, firefighter]
            for offset_row, offset_col in stencil:
                row = positions[:, firefighter, 0] + offset_row
                col = positions[:, firefighter, 1] + offset_col
                valid = (
                    spraying
                    & (row >= 0)
                    & (row < self._num_rows)
                    & (col >= 0)
                    & (col < self._num_cols)
                )
                index = torch.clamp(row, 0, self._num_rows - 1) * self._num_cols + torch.clamp(
                    col, 0, self._num_cols - 1
                )
                counts.scatter_add_(1, index.unsqueeze(1), valid.to(torch.int64).unsqueeze(1))
        flat_fire = fire.reshape(batch, self._num_cells)
        per_spray = self._suppression[flat_fire]
        soak = 1.0 - torch.pow(1.0 - per_spray, counts.to(self.dtype))
        soaked = (counts > 0) & (self._rand((batch, self._num_cells)) < soak)
        flat_fire = torch.where(soaked, torch.full_like(flat_fire, _WET), flat_fire)
        return flat_fire.reshape(batch, self._num_rows, self._num_cols)

    def _spread_fire(self, fire: Tensor, states: Tensor) -> Tensor:
        """Ignite unburnt, non-obstacle cells from their alight four-neighbours.

        The neighbour directly upwind contributes the boosted rate
        ``min(1, spread * gain)``; the other three contribute the crosswind
        rate. The survival product runs over directions 0..3 in the order
        ``FirefightingPOMDP._ignition_probability`` uses.
        """
        batch = fire.shape[0]
        direction = states[:, self._wind_direction_index].round().to(torch.int64)
        strength = states[:, self._wind_strength_index].round().to(torch.int64)
        gain = torch.where(
            strength == int(WindStrength.HIGH),
            torch.full((batch,), self._gain_high, dtype=self.dtype, device=self.device),
            torch.full((batch,), self._gain_low, dtype=self.dtype, device=self.device),
        )
        downwind_rate = torch.clamp(self._spread * gain, max=1.0)

        alight = self._alight(fire)
        padded = torch.nn.functional.pad(alight, (1, 1, 1, 1))
        survive = torch.ones(
            batch, self._num_rows, self._num_cols, dtype=self.dtype, device=self.device
        )
        for code, (offset_row, offset_col) in enumerate(DIRECTION_OFFSETS):
            # shifted[r, c] = alight[r - offset_row, c - offset_col], zero off the grid.
            shifted = padded[
                :,
                1 - offset_row : 1 - offset_row + self._num_rows,
                1 - offset_col : 1 - offset_col + self._num_cols,
            ]
            rate = torch.where(
                direction == code,
                downwind_rate,
                torch.full_like(downwind_rate, self._crosswind_rate),
            )[:, None, None]
            survive = torch.where(shifted, survive * (1.0 - rate), survive)
        ignition = 1.0 - survive
        ignitable = (fire == _UNBURNT) & ~self._obstacle
        ignites = ignitable & (self._rand((batch, self._num_rows, self._num_cols)) < ignition)
        return torch.where(ignites, torch.full_like(fire, _SMOLDERING), fire)

    def _grow_and_burn_out(self, fire: Tensor, after_suppression: Tensor) -> Tensor:
        """Grow and burn out the cells that were alight before the spread."""
        shape = tuple(fire.shape)
        grows = (after_suppression == _SMOLDERING) & (self._rand(shape) < self._growth)
        fire = torch.where(grows, torch.full_like(fire, _BURNING), fire)
        burns_out = (after_suppression == _BURNING) & (self._rand(shape) < self._burnout)
        return torch.where(burns_out, torch.full_like(fire, _BURNT), fire)

    # ------------------------------------------------------------------ #
    # Observation internals
    # ------------------------------------------------------------------ #

    def _visible(self, firefighters: Tensor) -> Tensor:
        """``[B, R, C]`` cells within Chebyshev ``sensing_radius`` of some live firefighter."""
        batch = firefighters.shape[0]
        visible = torch.zeros(
            batch, self._num_rows, self._num_cols, dtype=torch.bool, device=self.device
        )
        for firefighter in range(self._num_firefighters):
            row = firefighters[:, firefighter, 0][:, None]
            col = firefighters[:, firefighter, 1][:, None]
            near_row = (self._rows[None, :] - row).abs() <= self._sensing_radius
            near_col = (self._cols[None, :] - col).abs() <= self._sensing_radius
            live = (firefighters[:, firefighter, 3] > 0)[:, None, None]
            visible = visible | (live & near_row[:, :, None] & near_col[:, None, :])
        return visible

    def _firefighter_fields(self, states: Tensor) -> Tensor:
        return states[
            :,
            FIREFIGHTER_OFFSET : FIREFIGHTER_OFFSET
            + FIREFIGHTER_FIELD_WIDTH * self._num_firefighters,
        ]

    def _confusion_logs(self) -> Tuple[float, float]:
        """``log(1 - error)`` and ``log(error / 4)``, ``-inf`` where the argument is 0."""
        error = self._observation_error
        log_correct = float(np.log1p(-error)) if error < 1.0 else float("-inf")
        log_wrong = float(np.log(error / (NUM_CATEGORIES - 1))) if error > 0.0 else float("-inf")
        return log_correct, log_wrong


__all__ = ["FirefightingVectorizedModel"]
