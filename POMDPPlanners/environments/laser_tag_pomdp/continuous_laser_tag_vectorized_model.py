# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for Continuous LaserTag.

This module provides :class:`ContinuousLaserTagVectorizedModel`, a batched
implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for
:class:`~POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp.ContinuousLaserTagPOMDPDiscreteActions`.

It re-expresses the environment's native C++ kernels in torch:

* the transition -- a Gaussian robot step and a Gaussian opponent step, each
  pushed out of the wall AABBs (one wall at a time, in wall order) and clamped
  to the arena, the opponent moving under any of the three opponent policies,
  and the tag and hazard-terminal rules that set the terminal flag;
* the observation -- eight laser ranges to the nearest wall, arena boundary or
  opponent disc, plus Gaussian noise clipped at zero, and the all ``-1``
  reading from a terminal state;
* the reward the scalar ``reward(state, action, next_state)`` returns, with the
  dangerous-area penalty scored on the realised next robot position.

Every constant is read from a live environment instance, so the environment
stays the single source of truth for configuration. The registry conformance
test compares each kernel with the scalar environment.

The discrete-action variant is required: VOPP plans over a fixed, finite action
set, and its five actions are the model's action indices ``0..4`` in
``env.get_actions()`` order. Every configuration of that variant is modeled --
all three opponent policies, any ``dangerous_area_hit_probability``, and
``is_dangerous_area_hit_terminal`` -- because the hazard-terminal flag reuses
the state's existing terminal slot, so it needs no extra state dimension.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
    ContinuousLaserTagPOMDPDiscreteActions,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_utils import (
    OpponentPolicy,
)

# Beam directions, in the order of the native kernel's kLaserDirections (and
# continuous_laser_tag_geometry.LASER_DIRECTIONS).
_SQRT2_INV = 0.70710678118654752440
_LASER_DIRECTIONS = np.array(
    [
        [0.0, 1.0],
        [_SQRT2_INV, _SQRT2_INV],
        [1.0, 0.0],
        [_SQRT2_INV, -_SQRT2_INV],
        [0.0, -1.0],
        [-_SQRT2_INV, -_SQRT2_INV],
        [-1.0, 0.0],
        [-_SQRT2_INV, _SQRT2_INV],
    ],
    dtype=np.float64,
)
# Numeric constants of the native kernel (kRayMax, kParallelEps, kHitEps).
_RAY_MAX = 1e4
_PARALLEL_EPS = 1e-12
_HIT_EPS = 1e-9
# Tolerance of the native terminal-sentinel test, |obs + 1| <= 1e-8.
_SENTINEL_TOLERANCE = 1e-8
# Distinct primes for hashing quantized 8-D observations into integer keys.
_HASH_PRIMES = np.array(
    [73856093, 19349663, 83492791, 39916801, 51539607, 15485863, 32452843, 49979687],
    dtype=np.int64,
)


class ContinuousLaserTagVectorizedModel:
    """Batched torch generative model for the discrete-action Continuous LaserTag POMDP.

    States are ``[robot_x, robot_y, opponent_x, opponent_y, terminal]`` rows;
    actions are the integer indices of ``env.get_actions()`` (``up``,
    ``down``, ``right``, ``left``, ``tag``); observations are the eight laser
    ranges, or eight ``-1`` entries from a terminal state.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype used for state / observation / reward tensors.
        num_actions: Number of discrete actions (five).

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
        ...     ContinuousLaserTagPOMDPDiscreteActions,
        ... )
        >>> from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_vectorized_model import (
        ...     ContinuousLaserTagVectorizedModel,
        ... )
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> env = ContinuousLaserTagPOMDPDiscreteActions(discount_factor=0.95)
        >>> model = ContinuousLaserTagVectorizedModel(env, device=torch.device("cpu"))
        >>> states = torch.tensor([[1.0, 1.0, 8.0, 5.0, 0.0], [2.0, 3.0, 2.2, 3.0, 0.0]])
        >>> actions = torch.tensor([2, 4])  # move right, tag
        >>> next_states = model.sample_next_states(states, actions)
        >>> observations = model.sample_observations(next_states, actions)
        >>> rewards = model.rewards(states, actions, next_states)
        >>> tuple(next_states.shape), tuple(observations.shape), tuple(rewards.shape)
        ((2, 5), (2, 8), (2,))
        >>> next_states[1, 4].item()  # the tag succeeded, so the row is terminal
        1.0
    """

    def __init__(
        self,
        env: ContinuousLaserTagPOMDPDiscreteActions,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        observation_resolution: float = 0.1,
    ) -> None:
        """Build the model from a live environment instance.

        Args:
            env: The discrete-action environment whose parameters are mirrored.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.
            observation_resolution: Bin width used to quantize each laser range
                into an integer tree key.

        Raises:
            NotImplementedError: If ``env`` is not the discrete-action variant;
                the continuous-action environment has no finite action set.
            ValueError: If ``observation_resolution`` is not positive.
        """
        if not isinstance(env, ContinuousLaserTagPOMDPDiscreteActions):
            raise NotImplementedError(
                "vectorized model needs ContinuousLaserTagPOMDPDiscreteActions: VOPP plans "
                "over a finite action set"
            )
        if observation_resolution <= 0.0:
            raise ValueError("observation_resolution must be positive")
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self._obs_resolution = float(observation_resolution)
        vectors = np.stack([env.action_to_vector[name] for name in env.get_actions()])
        self._action_moves = self._to_tensor(vectors[:, :2])
        self._action_is_tag = torch.as_tensor(vectors[:, 2] > 0.5, device=self.device)
        self.num_actions = int(vectors.shape[0])
        self._build_geometry(env)
        self._build_dynamics(env)
        self._build_reward(env)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    def _build_geometry(self, env: ContinuousLaserTagPOMDPDiscreteActions) -> None:
        walls = np.asarray(env.walls, dtype=np.float64).reshape(-1, 4)
        self._walls = self._to_tensor(walls)
        self._wall_rows = [tuple(float(v) for v in row) for row in walls]
        self._wall_min = self._to_tensor(
            np.stack([walls[:, 0] - walls[:, 2], walls[:, 1] - walls[:, 3]], axis=1)
        )
        self._wall_max = self._to_tensor(
            np.stack([walls[:, 0] + walls[:, 2], walls[:, 1] + walls[:, 3]], axis=1)
        )
        grid = np.asarray(env.grid_size, dtype=np.float64).reshape(-1)
        self._grid_w = float(grid[0])
        self._grid_h = float(grid[1])
        self._directions = self._to_tensor(_LASER_DIRECTIONS)
        self._robot_radius = float(env.robot_radius)
        self._opponent_radius = float(env.opponent_radius)
        self._hash_primes = torch.as_tensor(_HASH_PRIMES, device=self.device)

    def _build_dynamics(self, env: ContinuousLaserTagPOMDPDiscreteActions) -> None:
        robot_cov = np.asarray(env.robot_transition_cov_matrix, dtype=np.float64)
        opponent_cov = np.asarray(env.opponent_transition_cov_matrix, dtype=np.float64)
        self._robot_chol_t = self._to_tensor(np.linalg.cholesky(robot_cov).T)
        self._opponent_chol_t = self._to_tensor(np.linalg.cholesky(opponent_cov).T)
        self._evasion_speed = float(env.evasion_speed)
        self._policy = env.opponent_policy
        self._tag_radius = float(env.tag_radius)
        sigma = float(env.measurement_noise)
        self._sigma = sigma
        self._inv_2var = 0.5 / (sigma * sigma)
        self._log_norm = 8.0 * (-0.5 * math.log(2.0 * math.pi * sigma * sigma))

    def _build_reward(self, env: ContinuousLaserTagPOMDPDiscreteActions) -> None:
        self._danger_centres = self._to_tensor(
            np.asarray(env.dangerous_areas, dtype=np.float64).reshape(-1, 2)
        )
        self._danger_radius_sq = float(env.dangerous_area_radius) ** 2
        self._danger_penalty = float(env.dangerous_area_penalty)
        self._danger_hit_prob = float(env.dangerous_area_hit_probability)
        self._hazard_terminal = bool(env.is_dangerous_area_hit_terminal)
        self._tag_reward = float(env.tag_reward)
        self._tag_penalty = float(env.tag_penalty)
        self._step_cost = float(env.step_cost)

    def _to_tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        indices = actions.to(torch.int64)
        robot, opponent = states[:, 0:2], states[:, 2:4]
        is_tag = self._action_is_tag[indices]
        # A tag leaves the robot in place; every other action is a noisy step.
        stepped = self._sample_move(
            robot + self._action_moves[indices], self._robot_radius, self._robot_chol_t
        )
        robot_next = torch.where(is_tag[:, None], robot, stepped)
        # PURSUE reacts to the robot's post-move position, the evading policies
        # to its pre-move position.
        reference = robot_next if self._policy is OpponentPolicy.PURSUE else robot
        opponent_next = self._sample_move(
            self._opponent_mean(reference, opponent), self._opponent_radius, self._opponent_chol_t
        )
        tagged = is_tag & (self._distance(robot, opponent) <= self._tag_radius)
        live_next = torch.cat([robot_next, opponent_next, torch.zeros_like(states[:, 4:5])], dim=1)
        # A successful tag freezes both positions and sets the terminal flag.
        tagged_next = torch.cat([robot, opponent, torch.ones_like(states[:, 4:5])], dim=1)
        next_states = torch.where(tagged[:, None], tagged_next, live_next)
        if self._hazard_terminal:
            hazard = (
                ~tagged & self._in_danger(robot_next) & self._draw(self._danger_hit_prob, states)
            )
            next_states[:, 4] = torch.where(hazard, 1.0, next_states[:, 4])
        # A terminal row is absorbing.
        return torch.where((states[:, 4] != 0.0)[:, None], states, next_states)

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions  # The laser reading does not depend on the action.
        mean = self._laser_ranges(next_states[:, 0:2], next_states[:, 2:4])
        noise = torch.randn(mean.shape, dtype=self.dtype, device=self.device)
        readings = torch.clamp(mean + self._sigma * noise, min=0.0)
        terminal = (next_states[:, 4] != 0.0)[:, None]
        return torch.where(terminal, torch.full_like(readings, -1.0), readings)

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        indices = actions.to(torch.int64)
        is_tag = self._action_is_tag[indices]
        gap = states[:, 0:2] - states[:, 2:4]
        in_reach = (gap * gap).sum(dim=1) <= self._tag_radius**2
        tag_term = torch.where(
            in_reach,
            torch.full_like(states[:, 4], self._tag_reward),
            torch.full_like(states[:, 4], -self._tag_penalty),
        )
        base = torch.where(is_tag, tag_term, torch.zeros_like(tag_term)) - self._step_cost
        live = states[:, 4] == 0.0
        next_live = next_states[:, 4] == 0.0
        matches = self._danger_matches(next_states[:, 0:2])
        if self._hazard_terminal:
            # The penalty is paid on the step that ends in a dangerous area,
            # unless that step was the successful tag.
            charged = live & ~next_live & ~(is_tag & in_reach)
        else:
            charged = live & next_live & self._draw(self._danger_hit_prob, states)
        penalty = self._danger_penalty * matches * charged.to(self.dtype)
        return torch.where(live, base - penalty, torch.zeros_like(base))

    def terminal_mask(self, states: Tensor) -> Tensor:
        return states[:, 4] != 0.0

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions  # The laser likelihood does not depend on the action.
        mean = self._laser_ranges(next_states[:, 0:2], next_states[:, 2:4])
        diff = observations - mean
        gaussian = self._log_norm - (diff * diff).sum(dim=1) * self._inv_2var
        is_sentinel = ((observations + 1.0).abs() <= _SENTINEL_TOLERANCE).all(dim=1)
        terminal = next_states[:, 4] != 0.0
        neg_inf = torch.full_like(gaussian, float("-inf"))
        # A terminal state emits only the sentinel; a live one never does.
        terminal_logp = torch.where(is_sentinel, torch.zeros_like(gaussian), neg_inf)
        live_logp = torch.where(is_sentinel, neg_inf, gaussian)
        return torch.where(terminal, terminal_logp, live_logp)

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        quantized = torch.floor(observations / self._obs_resolution).to(torch.int64)
        return (quantized * self._hash_primes).sum(dim=1)

    # ------------------------------------------------------------------ #
    # Motion helpers
    # ------------------------------------------------------------------ #

    def _sample_move(self, mean: Tensor, radius: float, chol_t: Tensor) -> Tensor:
        noise = torch.randn(mean.shape[0], 2, dtype=self.dtype, device=self.device)
        position = self._resolve_walls(mean + noise @ chol_t, radius)
        x = torch.clamp(position[:, 0], radius, self._grid_w - radius)
        y = torch.clamp(position[:, 1], radius, self._grid_h - radius)
        return torch.stack([x, y], dim=1)

    def _resolve_walls(self, position: Tensor, radius: float) -> Tensor:
        # The native kernel resolves the walls one after another, each against
        # the position the previous one left, so this loop is over walls only.
        x, y = position[:, 0], position[:, 1]
        r_sq = radius * radius
        for cx, cy, hx, hy in self._wall_rows:
            dx = x - torch.clamp(x, cx - hx, cx + hx)
            dy = y - torch.clamp(y, cy - hy, cy + hy)
            dist_sq = dx * dx + dy * dy
            overlap = dist_sq < r_sq
            dist = torch.where(dist_sq > 1e-24, torch.sqrt(dist_sq), torch.zeros_like(dist_sq))
            inside = overlap & (dist < 1e-12)
            outside = overlap & ~inside
            safe = torch.where(outside, dist, torch.ones_like(dist))
            push_x = x + dx / safe * (radius - dist)
            push_y = y + dy / safe * (radius - dist)
            inside_x, inside_y = self._push_out_of_box(x, y, (cx, cy, hx, hy), radius)
            x = torch.where(outside, push_x, torch.where(inside, inside_x, x))
            y = torch.where(outside, push_y, torch.where(inside, inside_y, y))
        return torch.stack([x, y], dim=1)

    @staticmethod
    def _push_out_of_box(
        x: Tensor, y: Tensor, wall: Tuple[float, float, float, float], radius: float
    ) -> Tuple[Tensor, Tensor]:
        # Centre inside the box: move along the axis of least penetration;
        # ties go to the first of left, right, down, up, as in the native code.
        cx, cy, hx, hy = wall
        pen_left = x - (cx - hx)
        pen_right = (cx + hx) - x
        pen_down = y - (cy - hy)
        pen_up = (cy + hy) - y
        min_pen = torch.minimum(torch.minimum(pen_left, pen_right), torch.minimum(pen_down, pen_up))
        left = min_pen == pen_left
        right = ~left & (min_pen == pen_right)
        down = ~left & ~right & (min_pen == pen_down)
        up = ~left & ~right & ~down
        new_x = torch.where(left, cx - hx - radius, torch.where(right, cx + hx + radius, x))
        new_y = torch.where(down, cy - hy - radius, torch.where(up, cy + hy + radius, y))
        return new_x, new_y

    def _opponent_mean(self, reference: Tensor, opponent: Tensor) -> Tensor:
        sign = -1.0 if self._policy is OpponentPolicy.PURSUE else 1.0
        diff = sign * (opponent - reference)
        dist = torch.hypot(diff[:, 0], diff[:, 1])
        # Coincident robot and opponent: the native kernel flees along a random
        # unit direction.
        random_dir = torch.randn(diff.shape, dtype=self.dtype, device=self.device)
        random_norm = torch.clamp(torch.hypot(random_dir[:, 0], random_dir[:, 1]), min=1e-9)
        coincident = (dist < 1e-9)[:, None]
        safe = torch.where(dist < 1e-9, torch.ones_like(dist), dist)
        direction = torch.where(coincident, random_dir / random_norm[:, None], diff / safe[:, None])
        mean = opponent + self._evasion_speed * direction
        if self._policy is OpponentPolicy.EVADE_WHEN_SPOTTED:
            # Out of sight, the opponent holds its position (plus noise).
            hidden = ~self._opponent_visible(reference, opponent)
            mean = torch.where(hidden[:, None], opponent, mean)
        return mean

    # ------------------------------------------------------------------ #
    # Laser helpers
    # ------------------------------------------------------------------ #

    def _laser_ranges(self, robot: Tensor, opponent: Tensor) -> Tensor:
        occluder = torch.minimum(self._wall_ranges(robot), self._boundary_ranges(robot))
        return torch.minimum(occluder, self._opponent_ranges(robot, opponent))

    def _opponent_visible(self, robot: Tensor, opponent: Tensor) -> Tensor:
        occluder = torch.minimum(self._wall_ranges(robot), self._boundary_ranges(robot))
        return (self._opponent_ranges(robot, opponent) < occluder).any(dim=1)

    def _wall_ranges(self, robot: Tensor) -> Tensor:
        """``[N, 8]`` distance along each beam to the nearest wall, or ``_RAY_MAX``."""
        n = robot.shape[0]
        if self._walls.shape[0] == 0:
            return torch.full((n, 8), _RAY_MAX, dtype=self.dtype, device=self.device)
        origin = robot[:, None, None, :]  # [N, 1, 1, 2]
        direction = self._directions[None, :, None, :]  # [1, 8, 1, 2]
        low = self._wall_min[None, None, :, :]  # [1, 1, M, 2]
        high = self._wall_max[None, None, :, :]
        parallel = direction.abs() <= _PARALLEL_EPS
        inv = 1.0 / torch.where(parallel, torch.ones_like(direction), direction)
        t1 = (low - origin) * inv
        t2 = (high - origin) * inv
        inside = (origin >= low) & (origin <= high)
        inf = torch.full_like(t1, float("inf"))
        enter = torch.where(parallel, torch.where(inside, -inf, inf), torch.minimum(t1, t2))
        leave = torch.where(parallel, torch.where(inside, inf, -inf), torch.maximum(t1, t2))
        t_min = enter.max(dim=-1).values  # [N, 8, M]
        t_max = leave.min(dim=-1).values
        hit = torch.where(t_min > 0.0, t_min, t_max)
        valid = (t_max > torch.clamp(t_min, min=0.0)) & (hit > _HIT_EPS)
        distance = torch.where(valid, hit, torch.full_like(hit, _RAY_MAX))
        return distance.min(dim=-1).values

    def _boundary_ranges(self, robot: Tensor) -> Tensor:
        """``[N, 8]`` distance along each beam to the arena boundary, or ``_RAY_MAX``."""
        origin = robot[:, None, :]  # [N, 1, 2]
        direction = self._directions[None, :, :]  # [1, 8, 2]
        parallel = direction.abs() <= _PARALLEL_EPS
        safe = torch.where(parallel, torch.ones_like(direction), direction)
        size = torch.tensor([self._grid_w, self._grid_h], dtype=self.dtype, device=self.device)
        near = -origin / safe
        far = (size - origin) / safe
        candidates = torch.cat([near, far], dim=-1)  # [N, 8, 4]
        usable = torch.cat([~parallel, ~parallel], dim=-1) & (candidates > _HIT_EPS)
        candidates = torch.where(usable, candidates, torch.full_like(candidates, _RAY_MAX))
        return torch.clamp(candidates.min(dim=-1).values, max=_RAY_MAX)

    def _opponent_ranges(self, robot: Tensor, opponent: Tensor) -> Tensor:
        """``[N, 8]`` distance along each beam to the opponent disc, or ``inf``."""
        offset = (robot - opponent)[:, None, :]  # [N, 1, 2]
        b = (offset * self._directions[None, :, :]).sum(dim=-1)  # [N, 8]
        c = (offset * offset).sum(dim=-1) - self._opponent_radius**2  # [N, 1]
        disc = b * b - c
        root = torch.sqrt(torch.clamp(disc, min=0.0))
        t1 = -b - root
        t2 = -b + root
        inf = torch.full_like(t1, float("inf"))
        hit = torch.where(t1 > _HIT_EPS, t1, torch.where(t2 > _HIT_EPS, t2, inf))
        return torch.where(disc < 0.0, inf, hit)

    # ------------------------------------------------------------------ #
    # Reward helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _distance(a: Tensor, b: Tensor) -> Tensor:
        diff = a - b
        return torch.hypot(diff[:, 0], diff[:, 1])

    def _danger_matches(self, robot: Tensor) -> Tensor:
        """Number of dangerous areas that contain each robot position, as floats."""
        if self._danger_centres.shape[0] == 0:
            return torch.zeros(robot.shape[0], dtype=self.dtype, device=self.device)
        diff = robot[:, None, :] - self._danger_centres[None, :, :]
        inside = (diff * diff).sum(dim=-1) <= self._danger_radius_sq
        return inside.sum(dim=1).to(self.dtype)

    def _in_danger(self, robot: Tensor) -> Tensor:
        return self._danger_matches(robot) > 0.0

    def _draw(self, probability: float, like: Tensor) -> Tensor:
        """One Bernoulli(``probability``) per row."""
        if probability >= 1.0:
            return torch.ones(like.shape[0], dtype=torch.bool, device=self.device)
        return torch.rand(like.shape[0], dtype=self.dtype, device=self.device) < probability
