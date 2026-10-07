# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for the Chicheck Invaders POMDP.

This module provides :class:`ChicheckInvadersVectorizedModel`, a batched
implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for
:class:`~POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_pomdp.ChicheckInvadersPOMDP`,
so VOPP can plan on it.

The kernels follow the scalar environment step for step, with particles on the
first tensor axis and chicken slots on the second:

* the transition moves the ship, resolves the hitscan shot, flips one dive coin
  per slot, moves the flock, and settles every live chicken that reached row 0;
* the observation is the camera and radar reading in the environment's layout
  (or the state itself in ``ObservationMode.FULL``);
* the likelihood multiplies in one camera and one radar factor for every
  chicken slot, reported or not, and floors at the environment's
  ``IMPOSSIBLE_LOG_PROBABILITY``;
* the reward and the terminal test read the same state fields the scalar
  ``reward`` and ``is_terminal`` read.

Like the scalar environment, the transition does not hold a terminal state
fixed: stepping one advances the step counter and moves the flock, and the
reward of that step is scored by the same formula. The model copies that
behaviour rather than freezing finished rows, because the parity tests compare
the two on terminal states too.

Both observation modes are supported, and the grid, flock, sensor, reward and
step-budget arguments are read from the live environment, so no configuration
is declined.

Precision: the conformance tests build the model in ``float64``. In the default
``float32`` a rounded-Gaussian mass below about ``1e-45`` (about 14 noise
standard deviations off) underflows to zero and scores the impossible floor,
where the scalar environment still returns a finite log-probability below
``-103``. Use ``float64`` when far-tail likelihoods matter.
"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_pomdp import (
    IMPOSSIBLE_LOG_PROBABILITY,
    ChicheckInvadersAction,
    ChicheckInvadersPOMDP,
    ObservationMode,
)
from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_schema import (
    CHICKEN_ALIVE,
    CHICKEN_COLUMN,
    CHICKEN_DIRECTION,
    CHICKEN_MODE,
    CHICKEN_ROW,
    CHICKEN_WIDTH,
    COOLDOWN_INDEX,
    MODE_DIVE,
    MODE_PATROL,
    OBSERVATION_CHICKEN_WIDTH,
    OBSERVATION_SHIP_WIDTH,
    OBSERVED_CAMERA_OFFSET,
    OBSERVED_CAMERA_REPORTED,
    OBSERVED_RADAR_DROP,
    OBSERVED_RADAR_REPORTED,
    OBSERVED_RADAR_ROWS,
    OBSERVED_SHIP_COLUMN_INDEX,
    SHIP_COLUMN_INDEX,
    SHIP_HIT_INDEX,
    SHIP_WIDTH,
    STEP_INDEX,
)

# Odd multiplier for the rolling polynomial hash that turns an integer-valued
# observation vector into one int64 tree key. int64 multiplication wraps, so
# the key is defined for any vector length.
_OBS_HASH_MULTIPLIER = 2654435761
# Added to every field before hashing so a field of 0 still changes the key.
_OBS_HASH_OFFSET = 1009
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _standard_normal_cdf(values: Tensor) -> Tensor:
    """Standard normal CDF, accurate in the lower tail.

    Written through ``erfc`` rather than ``torch.special.ndtr``: in float64
    torch's ``ndtr`` returns exactly 0 below about -8.5, where SciPy's -- which
    the scalar environment uses -- still returns ``1e-19`` and smaller. That
    turned a far but finite reading into the impossible floor.
    """
    return 0.5 * torch.special.erfc(-values * _INV_SQRT2)


class ChicheckInvadersVectorizedModel:  # pylint: disable=too-many-instance-attributes
    """Batched torch generative model for the Chicheck Invaders POMDP.

    States and observations are the environment's own fixed-length vectors, so
    a row converts to the scalar environment's value with no reshaping.
    Actions are the four ``ChicheckInvadersAction`` indices used directly.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype of state, observation and reward tensors.
        num_actions: Number of discrete actions (4).
        state_dim: Length of a state row.
        observation_dim: Length of an observation row.

    Example:
        >>> import numpy as np
        >>> import torch
        >>> from POMDPPlanners.environments.chicheck_invaders_pomdp import (
        ...     ChicheckInvadersPOMDP,
        ...     ChicheckInvadersVectorizedModel,
        ... )
        >>> _ = torch.manual_seed(0)
        >>> np.random.seed(0)
        >>> env = ChicheckInvadersPOMDP()
        >>> model = ChicheckInvadersVectorizedModel(env, device=torch.device("cpu"))
        >>> state = torch.as_tensor(
        ...     np.asarray(env.initial_state_dist().sample()[0]), dtype=model.dtype
        ... ).unsqueeze(0)
        >>> actions = torch.tensor([3])  # FIRE
        >>> next_states = model.sample_next_states(state, actions)
        >>> observations = model.sample_observations(next_states, actions)
        >>> rewards = model.rewards(state, actions, next_states)
        >>> tuple(next_states.shape), tuple(observations.shape), tuple(rewards.shape)
        ((1, 24), (1, 21), (1,))
    """

    def __init__(
        self,
        env: ChicheckInvadersPOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Build the model from a live environment instance.

        Args:
            env: The environment whose parameters are mirrored.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.
        """
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self.num_actions = len(ChicheckInvadersAction)
        self._full = env.observation_mode is ObservationMode.FULL
        self._num_chickens = int(env.num_chickens)
        self._num_columns = int(env.num_columns)
        self._num_rows = int(env.num_rows)
        self._fire_cooldown = float(env.fire_cooldown)
        self._dive_probability = float(env.dive_probability)
        self._max_steps = int(env.max_steps)
        self.state_dim = int(env.state_size)
        self.observation_dim = int(env.observation_size)
        self._sensor = {
            "camera_p": float(env.camera_detection_probability),
            "radar_p": float(env.radar_detection_probability),
            "ship_std": float(env.ship_column_noise_std),
            "camera_std": float(env.camera_offset_noise_std),
            "radar_std": float(env.radar_range_noise_std),
            "flag_error": float(env.drop_flag_error_probability),
            "slope": float(env.camera_slope),
            "radius_sq": float(env.radar_radius) ** 2,
        }
        self._reward = {
            "kill": float(env.kill_reward),
            "shot": float(env.shot_cost),
            "step": float(env.step_cost),
            "hit": float(env.ship_hit_penalty),
            "clear": float(env.clear_reward),
        }

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        """Advance every row one step, in the scalar environment's order."""
        successor = states.clone()
        successor[:, STEP_INDEX] = states[:, STEP_INDEX] + 1.0
        fired = self._fires(states, actions)
        ship = self._moved_ship_column(states, actions)
        successor[:, SHIP_COLUMN_INDEX] = ship
        cooldown = torch.clamp(states[:, COOLDOWN_INDEX] - 1.0, min=0.0)
        successor[:, COOLDOWN_INDEX] = torch.where(
            fired, torch.full_like(cooldown, self._fire_cooldown), cooldown
        )

        flock = self._flock(successor).clone()
        self._resolve_shot(flock, fired, ship)
        self._move_flock(flock)
        hit = self._resolve_arrivals(flock, ship)
        successor[:, SHIP_HIT_INDEX] = torch.where(
            hit, torch.ones_like(ship), successor[:, SHIP_HIT_INDEX]
        )
        successor[:, SHIP_WIDTH:] = flock.reshape(states.shape[0], -1)
        return successor

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        """Draw one camera-and-radar reading per row (the state itself in FULL mode)."""
        del actions  # The reading depends on the successor only.
        if self._full:
            return next_states.clone()
        count = next_states.shape[0]
        offsets, rows, camera, radar = self._sensor_geometry(next_states)
        observation = torch.zeros(count, self.observation_dim, dtype=self.dtype, device=self.device)
        observation[:, OBSERVED_SHIP_COLUMN_INDEX] = self._rounded_normal_sample(
            next_states[:, SHIP_COLUMN_INDEX], self._sensor["ship_std"]
        )
        slots = observation[:, OBSERVATION_SHIP_WIDTH:].view(
            count, self._num_chickens, OBSERVATION_CHICKEN_WIDTH
        )
        zeros = torch.zeros_like(offsets)

        camera_reported = camera & (self._uniform(offsets) < self._sensor["camera_p"])
        slots[..., OBSERVED_CAMERA_REPORTED] = camera_reported.to(self.dtype)
        camera_reading = self._rounded_normal_sample(offsets, self._sensor["camera_std"])
        slots[..., OBSERVED_CAMERA_OFFSET] = torch.where(camera_reported, camera_reading, zeros)

        radar_reported = radar & (self._uniform(offsets) < self._sensor["radar_p"])
        slots[..., OBSERVED_RADAR_REPORTED] = radar_reported.to(self.dtype)
        radar_reading = self._rounded_normal_sample(rows, self._sensor["radar_std"])
        slots[..., OBSERVED_RADAR_ROWS] = torch.where(radar_reported, radar_reading, zeros)
        true_drop = self._true_drop(next_states)
        flipped = self._uniform(offsets) < self._sensor["flag_error"]
        drop = torch.where(flipped, -1.0 - true_drop, true_drop)
        slots[..., OBSERVED_RADAR_DROP] = torch.where(radar_reported, drop, zeros)
        return observation

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        """Score each step as the scalar ``reward`` does when given the successor."""
        fired = self._fires(states, actions).to(self.dtype)
        before = self._live_count(states)
        after = self._live_count(next_states)
        reward = -self._reward["step"] - self._reward["shot"] * fired
        reward = reward + self._reward["kill"] * (before - after)
        newly_hit = (next_states[:, SHIP_HIT_INDEX] > 0.0) & ~(states[:, SHIP_HIT_INDEX] > 0.0)
        reward = reward - self._reward["hit"] * newly_hit.to(self.dtype)
        cleared = (after == 0.0) & (before > 0.0)
        return reward + self._reward["clear"] * cleared.to(self.dtype)

    def terminal_mask(self, states: Tensor) -> Tensor:
        """Flock cleared, ship hit, or step budget used up."""
        hit = states[:, SHIP_HIT_INDEX] > 0.0
        timed_out = torch.round(states[:, STEP_INDEX]) >= float(self._max_steps)
        return hit | timed_out | (self._live_count(states) == 0.0)

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        """Log ``Z(o | s')`` per row, floored at ``IMPOSSIBLE_LOG_PROBABILITY``."""
        del actions  # The reading depends on the successor only.
        if self._full:
            same = (next_states == observations).all(dim=1)
            return torch.where(
                same,
                torch.zeros(same.shape, dtype=self.dtype, device=self.device),
                torch.full(
                    same.shape, IMPOSSIBLE_LOG_PROBABILITY, dtype=self.dtype, device=self.device
                ),
            )
        count = next_states.shape[0]
        offsets, rows, camera, radar = self._sensor_geometry(next_states)
        slots = observations[:, OBSERVATION_SHIP_WIDTH:].reshape(
            count, self._num_chickens, OBSERVATION_CHICKEN_WIDTH
        )
        total = self._log(
            self._rounded_normal_pmf(
                observations[:, OBSERVED_SHIP_COLUMN_INDEX],
                next_states[:, SHIP_COLUMN_INDEX],
                self._sensor["ship_std"],
            )
        )
        camera_factor = self._detection_log_factor(
            reported=slots[..., OBSERVED_CAMERA_REPORTED] > 0.0,
            in_reach=camera,
            probability=self._sensor["camera_p"],
            reading_log=self._log(
                self._rounded_normal_pmf(
                    slots[..., OBSERVED_CAMERA_OFFSET], offsets, self._sensor["camera_std"]
                )
            ),
        )
        flag = torch.where(
            slots[..., OBSERVED_RADAR_DROP] == self._true_drop(next_states),
            torch.full_like(offsets, 1.0 - self._sensor["flag_error"]),
            torch.full_like(offsets, self._sensor["flag_error"]),
        )
        radar_reading_log = self._log(
            self._rounded_normal_pmf(
                slots[..., OBSERVED_RADAR_ROWS], rows, self._sensor["radar_std"]
            )
        ) + self._log(flag)
        radar_factor = self._detection_log_factor(
            reported=slots[..., OBSERVED_RADAR_REPORTED] > 0.0,
            in_reach=radar,
            probability=self._sensor["radar_p"],
            reading_log=radar_reading_log,
        )
        total = total + camera_factor.sum(dim=1) + radar_factor.sum(dim=1)
        # The scalar path stops at the floor as soon as any factor reaches it;
        # every factor is <= 0, so clamping the sum gives the same number.
        return torch.clamp(total, min=IMPOSSIBLE_LOG_PROBABILITY)

    def action_keys(self, actions: Tensor) -> Tensor:
        """Action indices are already the tree keys."""
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        """Hash each integer-valued observation row into one int64 key."""
        fields = torch.round(observations).to(torch.int64) + _OBS_HASH_OFFSET
        key = torch.zeros(observations.shape[0], dtype=torch.int64, device=observations.device)
        for column in range(fields.shape[1]):
            key = key * _OBS_HASH_MULTIPLIER + fields[:, column]
        return key

    # ------------------------------------------------------------------ #
    # Transition internals
    # ------------------------------------------------------------------ #

    def _flock(self, states: Tensor) -> Tensor:
        """``[N, num_chickens, 5]`` view of the chicken slots."""
        return states[:, SHIP_WIDTH:].reshape(states.shape[0], self._num_chickens, CHICKEN_WIDTH)

    def _fires(self, states: Tensor, actions: Tensor) -> Tensor:
        """Whether each row's action discharges the gun (FIRE with the cooldown over)."""
        return (actions == int(ChicheckInvadersAction.FIRE)) & (states[:, COOLDOWN_INDEX] <= 0.0)

    def _moved_ship_column(self, states: Tensor, actions: Tensor) -> Tensor:
        """The ship's column after LEFT / RIGHT, clamped at the walls."""
        delta = (actions == int(ChicheckInvadersAction.RIGHT)).to(self.dtype) - (
            actions == int(ChicheckInvadersAction.LEFT)
        ).to(self.dtype)
        column = torch.round(states[:, SHIP_COLUMN_INDEX]) + delta
        return torch.clamp(column, 0.0, float(self._num_columns - 1))

    def _resolve_shot(self, flock: Tensor, fired: Tensor, ship: Tensor) -> None:
        """Kill the lowest live chicken in the ship's column (rows 1 and up) where fired.

        Ties go to the lower slot, as in the scalar ``shot_target``: ``argmin``
        returns the first minimum.
        """
        candidate = (
            (flock[..., CHICKEN_ALIVE] > 0.0)
            & (torch.trunc(flock[..., CHICKEN_COLUMN]) == ship.unsqueeze(1))
            & (flock[..., CHICKEN_ROW] >= 1.0)
        )
        masked_rows = torch.where(
            candidate,
            flock[..., CHICKEN_ROW],
            torch.full_like(flock[..., CHICKEN_ROW], float("inf")),
        )
        target = masked_rows.argmin(dim=1)
        kill = fired & candidate.any(dim=1)
        slot_index = torch.arange(self._num_chickens, device=self.device).unsqueeze(0)
        killed = kill.unsqueeze(1) & (slot_index == target.unsqueeze(1))
        flock[..., CHICKEN_ALIVE] = torch.where(
            killed, torch.zeros_like(flock[..., CHICKEN_ALIVE]), flock[..., CHICKEN_ALIVE]
        )

    def _move_flock(self, flock: Tensor) -> None:
        """Flip one dive coin per slot, drop the divers, step and bounce the patrollers."""
        alive = flock[..., CHICKEN_ALIVE] > 0.0
        coins = self._uniform(flock[..., CHICKEN_MODE]) < self._dive_probability
        switch = alive & (flock[..., CHICKEN_MODE] == MODE_PATROL) & coins
        mode = torch.where(
            switch, torch.full_like(flock[..., CHICKEN_MODE], MODE_DIVE), flock[..., CHICKEN_MODE]
        )
        flock[..., CHICKEN_MODE] = mode
        diving = alive & (mode == MODE_DIVE)
        patrolling = alive & ~diving
        flock[..., CHICKEN_ROW] = flock[..., CHICKEN_ROW] - diving.to(self.dtype)
        columns = flock[..., CHICKEN_COLUMN]
        directions = flock[..., CHICKEN_DIRECTION]
        stepped = columns + directions
        bounced = (stepped < 0.0) | (stepped > float(self._num_columns - 1))
        new_directions = torch.where(bounced, -directions, directions)
        flock[..., CHICKEN_DIRECTION] = torch.where(patrolling, new_directions, directions)
        flock[..., CHICKEN_COLUMN] = torch.where(patrolling, columns + new_directions, columns)

    def _resolve_arrivals(self, flock: Tensor, ship: Tensor) -> Tensor:
        """Settle live chickens on row 0 or below; return which rows hit the ship.

        One in the ship's column stays on row 0 and destroys the ship. One
        anywhere else pulls up to the top row and goes back to patrolling.
        """
        arrived = (flock[..., CHICKEN_ALIVE] > 0.0) & (flock[..., CHICKEN_ROW] <= 0.0)
        on_ship = arrived & (torch.trunc(flock[..., CHICKEN_COLUMN]) == ship.unsqueeze(1))
        pulled_up = arrived & ~on_ship
        rows = flock[..., CHICKEN_ROW]
        rows = torch.where(on_ship, torch.zeros_like(rows), rows)
        rows = torch.where(pulled_up, torch.full_like(rows, float(self._num_rows - 1)), rows)
        flock[..., CHICKEN_ROW] = rows
        flock[..., CHICKEN_MODE] = torch.where(
            pulled_up, torch.full_like(rows, MODE_PATROL), flock[..., CHICKEN_MODE]
        )
        return on_ship.any(dim=1)

    def _live_count(self, states: Tensor) -> Tensor:
        """Number of live chicken slots per row, as ``dtype``."""
        return (self._flock(states)[..., CHICKEN_ALIVE] > 0.0).to(self.dtype).sum(dim=1)

    # ------------------------------------------------------------------ #
    # Sensor internals
    # ------------------------------------------------------------------ #

    def _sensor_geometry(self, states: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Column offsets, row distances, and the camera / radar reach masks."""
        flock = self._flock(states)
        offsets = flock[..., CHICKEN_COLUMN] - states[:, SHIP_COLUMN_INDEX].unsqueeze(1)
        rows = flock[..., CHICKEN_ROW]
        alive = flock[..., CHICKEN_ALIVE] > 0.0
        camera = alive & (offsets.abs() <= self._sensor["slope"] * rows)
        radar = alive & (offsets * offsets + rows * rows <= self._sensor["radius_sq"])
        return offsets, rows, camera, radar

    def _true_drop(self, states: Tensor) -> Tensor:
        """The radar's noiseless drop flag: -1 for a diving chicken, 0 otherwise."""
        diving = self._flock(states)[..., CHICKEN_MODE] == MODE_DIVE
        return -diving.to(self.dtype)

    def _detection_log_factor(
        self, reported: Tensor, in_reach: Tensor, probability: float, reading_log: Tensor
    ) -> Tensor:
        """One sensor's per-chicken log factor, as in the scalar ``_*_log_factor``."""
        floor = torch.full_like(reading_log, IMPOSSIBLE_LOG_PROBABILITY)
        outside = torch.where(reported, floor, torch.zeros_like(reading_log))
        missed = torch.full_like(reading_log, self._log_scalar(1.0 - probability))
        seen = self._log_scalar(probability) + reading_log
        inside = torch.where(reported, seen, missed)
        return torch.where(in_reach, inside, outside)

    def _rounded_normal_pmf(self, readings: Tensor, mean: Tensor, std: float) -> Tensor:
        """Mass of ``round(mean + std * xi)`` at ``readings``, with the scalar tail handling."""
        if std <= 0.0:
            return (readings == mean).to(self.dtype)
        upper = (readings + 0.5 - mean) / std
        lower = (readings - 0.5 - mean) / std
        mass = torch.where(
            lower > 0.0,
            _standard_normal_cdf(-lower) - _standard_normal_cdf(-upper),
            _standard_normal_cdf(upper) - _standard_normal_cdf(lower),
        )
        return torch.clamp(mass, min=0.0)

    def _rounded_normal_sample(self, mean: Tensor, std: float) -> Tensor:
        """One integer reading per element; ``std == 0`` returns ``round(mean)``."""
        if std <= 0.0:
            return torch.round(mean)
        noise = torch.randn(mean.shape, dtype=self.dtype, device=self.device)
        return torch.round(mean + std * noise)

    def _uniform(self, template: Tensor) -> Tensor:
        """Uniform ``[0, 1)`` draws shaped like ``template``."""
        return torch.rand(template.shape, dtype=self.dtype, device=self.device)

    @staticmethod
    def _log(values: Tensor) -> Tensor:
        """Natural log, with zero mapped to the impossible floor."""
        positive = values > 0.0
        safe = torch.where(positive, values, torch.ones_like(values))
        return torch.where(
            positive, torch.log(safe), torch.full_like(values, IMPOSSIBLE_LOG_PROBABILITY)
        )

    @staticmethod
    def _log_scalar(value: float) -> float:
        """Natural log of a constant, with zero mapped to the impossible floor."""
        return math.log(value) if value > 0.0 else IMPOSSIBLE_LOG_PROBABILITY
