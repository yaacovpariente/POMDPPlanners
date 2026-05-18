"""Reward models for the discrete LaserTag POMDP.

Mirrors the abstract-base / concrete-subclass layout used by
:mod:`POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models`,
so that further LaserTag reward variants can be added without growing
the env class.

The reward model owns all the parameters and pre-built buffers that
the reward computation needs (walls, dangerous areas, action
directions, native-kernel cache). The environment retains its own
parameter copies for the transition / observation paths and delegates
``reward()`` / ``reward_batch()`` to the model.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from POMDPPlanners.environments.environment_utils.dangerous_areas_kernels import (
    decaying_prob_penalty_batch_kernel,
    decaying_prob_penalty_kernel,
)


class BaseLaserTagRewardModel(ABC):
    """Abstract reward model for LaserTag POMDP variants."""

    @abstractmethod
    def compute_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: Optional[np.ndarray] = None,
    ) -> float:
        """Return the scalar reward for ``(state, action, next_state)``."""

    @abstractmethod
    def compute_reward_batch(
        self,
        states: np.ndarray,
        action: int,
        next_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the per-row reward for a batch of states under a single action."""


class LaserTagRewardModel(BaseLaserTagRewardModel):
    """Standard LaserTag reward model.

    Reward structure:
        * Tag action (``4``):
            ``+tag_reward`` if robot and opponent occupy the same cell,
            otherwise ``-tag_penalty``.
        * Movement actions (``0..3``): ``-step_cost``.
        * Penalty: ``-dangerous_area_penalty`` is subtracted whenever the
          *realised* post-action robot position is inside a wall or
          within ``dangerous_area_radius`` of a dangerous-area centre.

    Terminal states return ``0.0`` reward.
    """

    def __init__(
        self,
        floor_shape: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        dangerous_areas: List[Tuple[int, int]],
        dangerous_area_radius: float,
        dangerous_area_penalty: float,
        tag_reward: float,
        tag_penalty: float,
        step_cost: float,
        action_directions: Dict[int, Tuple[int, int]],
    ):
        self.floor_shape = floor_shape
        self.walls = walls
        self.dangerous_areas = dangerous_areas
        self.dangerous_area_radius = dangerous_area_radius
        self.dangerous_area_penalty = dangerous_area_penalty
        self.tag_reward = tag_reward
        self.tag_penalty = tag_penalty
        self.step_cost = step_cost
        self.action_directions = action_directions

        # Pre-built C-contiguous int64 (4, 2) array of (dr, dc) for actions 0..3,
        # consumed by the native ``lasertag_discrete_reward_batch`` kernel so the
        # hot path doesn't repack a Python dict on every call.
        self._action_directions_arr: np.ndarray = np.ascontiguousarray(
            np.array(
                [self.action_directions[a] for a in (0, 1, 2, 3)],
                dtype=np.int64,
            )
        )
        # Pre-built (D, 2) C-contiguous float64 array of dangerous-area centres
        # (or empty (0, 2) when none are configured) so the native kernel can
        # consume it directly without per-call reallocation.
        if self.dangerous_areas:
            self._dangerous_areas_arr: np.ndarray = np.ascontiguousarray(
                np.asarray(self.dangerous_areas, dtype=np.float64).reshape(-1, 2)
            )
        else:
            self._dangerous_areas_arr = np.empty((0, 2), dtype=np.float64)
        # Flattened int64 walls buffer (length 2 * n_walls) for the native
        # kernel; pairs are (row, col). Sorted for deterministic ordering.
        walls_list = sorted(self.walls)
        self._reward_walls_flat: np.ndarray = np.array(
            [coord for pair in walls_list for coord in pair],
            dtype=np.int64,
        )
        self._reward_n_walls: int = len(walls_list)
        # Pre-built (2, D) C-contiguous float64 view of dangerous-area centres
        # for the generic environment_utils.dangerous_areas_kernels (which
        # expect ``(2, D)`` with axis 0 holding the two coordinate components).
        if self.dangerous_areas:
            self._dangerous_centers_xy: np.ndarray = np.ascontiguousarray(
                self._dangerous_areas_arr.T
            )
        else:
            self._dangerous_centers_xy = np.empty((2, 0), dtype=np.float64)
        self._dangerous_radius_sq: float = float(dangerous_area_radius) * float(
            dangerous_area_radius
        )

    def __getstate__(self) -> Dict[str, Any]:
        # The native reward-batch cache holds a pybind11 function reference that
        # is not picklable. Drop it at serialization time; the lazy accessor
        # rebuilds it on demand after unpickling.
        state = self.__dict__.copy()
        state.pop("_cached_native_reward_batch", None)
        state.pop("_wall_grid", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        vars(self).update(state)

    def _is_in_dangerous_area(self, position: Tuple[int, int]) -> bool:
        """Check if a grid position is within any dangerous area."""
        if not self.dangerous_areas:
            return False

        pos_row, pos_col = position
        # Square-distance comparison avoids np.sqrt per area (~400 ns each).
        radius_sq = self.dangerous_area_radius * self.dangerous_area_radius
        for danger_row, danger_col in self.dangerous_areas:
            dr = pos_row - danger_row
            dc = pos_col - danger_col
            if dr * dr + dc * dc <= radius_sq:
                return True

        return False

    def compute_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: Optional[np.ndarray] = None,
    ) -> float:
        if bool(state[4]):
            return 0.0

        if next_state is None:
            # The env-side ``reward`` resamples a transition before delegating
            # so the penalty is always scored against a realised draw. The
            # ``None`` branch here is the legacy intended-position fallback
            # used by callers that bypass the env wrapper (e.g. internal
            # tests that exercise the model directly).
            dr, dc = self.action_directions[action]
            realised_pos = (int(state[0]) + dr, int(state[1]) + dc)
        else:
            realised_pos = (int(next_state[0]), int(next_state[1]))

        robot_pos = (int(state[0]), int(state[1]))
        opponent_pos = (int(state[2]), int(state[3]))

        if action == 4:  # Tag action
            base_reward = self.tag_reward if robot_pos == opponent_pos else -self.tag_penalty
        else:
            base_reward = -self.step_cost

        base_reward += self._compute_area_penalty_scalar(realised_pos)

        return base_reward

    def _compute_area_penalty_scalar(self, position: Tuple[int, int]) -> float:
        """Combined wall + dangerous-area contribution (single ``-penalty`` on OR).

        Standard semantics: one ``-dangerous_area_penalty`` is subtracted iff
        ``position`` is on a wall *or* inside any dangerous-area zone. Subclasses
        override to apply wall and danger contributions independently (e.g. so
        the danger contribution can be stochastic or distance-decaying while
        wall hits remain deterministic).
        """
        if position in self.walls or self._is_in_dangerous_area(position):
            return -self.dangerous_area_penalty
        return 0.0

    def compute_reward_batch(
        self,
        states: np.ndarray,
        action: int,
        next_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Vectorised reward for a batch of states under a single action.

        When ``next_states`` is supplied the danger-area / wall penalty is
        evaluated against the realised positions in ``next_states[:, :2]``.
        When it is ``None`` the legacy intended-position branch is used
        (``state[:, :2] + action_direction``), which is also what the native
        C++ kernel encodes. The env-side wrapper resamples a realised
        ``next_states`` whenever walls / dangerous areas exist, so the
        ``None`` branch here is only entered when the penalty term would
        not fire anyway (or by callers that bypass the env wrapper).
        """
        states_arr = np.asarray(states, dtype=np.float64)
        if states_arr.ndim == 1:
            states_arr = states_arr.reshape(1, -1)
        states_arr = np.ascontiguousarray(states_arr)

        next_states_arr: Optional[np.ndarray] = None
        if next_states is not None:
            ns_arr = np.asarray(next_states, dtype=np.float64)
            if ns_arr.ndim == 1:
                ns_arr = ns_arr.reshape(1, -1)
            next_states_arr = np.ascontiguousarray(ns_arr)

        if self._can_use_native_reward_batch():
            native_fn = self._get_native_reward_batch()
            if native_fn is not None:
                return np.asarray(
                    native_fn(
                        states=states_arr,
                        action=int(action),
                        rows=int(self.floor_shape[0]),
                        cols=int(self.floor_shape[1]),
                        walls_flat=self._reward_walls_flat,
                        n_walls=self._reward_n_walls,
                        dangerous_areas=self._dangerous_areas_arr,
                        n_dangerous=int(self._dangerous_areas_arr.shape[0]),
                        dangerous_area_radius=float(self.dangerous_area_radius),
                        dangerous_area_penalty=float(self.dangerous_area_penalty),
                        tag_reward=float(self.tag_reward),
                        tag_penalty=float(self.tag_penalty),
                        step_cost=float(self.step_cost),
                        action_directions=self._action_directions_arr,
                        next_states=(
                            next_states_arr
                            if next_states_arr is not None
                            else np.empty((0, 0), dtype=np.float64)
                        ),
                        reward_variant_code=self._reward_variant_code(),
                        penalty_decay=float(self._penalty_decay_for_native()),
                    )
                )
        return self._compute_reward_batch_python(states_arr, action, next_states_arr)

    def _can_use_native_reward_batch(self) -> bool:
        # The variant-aware native kernel handles all three reward-model
        # variants (CONSTANT_HAZARD_PENALTY, ZERO_MEAN_HAZARD_SHOCK, DISTANCE_DECAYED_HAZARD_PENALTY)
        # and both the intended-position fallback and the realised-position
        # path, so it is always preferred when available.
        return True

    def _reward_variant_code(self) -> int:
        # Base class is the CONSTANT_HAZARD_PENALTY variant. Subclasses override.
        return 0

    def _penalty_decay_for_native(self) -> float:
        # CONSTANT_HAZARD_PENALTY / ZERO_MEAN_HAZARD_SHOCK ignore penalty_decay; pass a sentinel
        # positive value so the C++ kernel's "must be > 0 for DECAYING" guard
        # does not need to special-case the unused-parameter path.
        return 1.0

    def _get_native_reward_batch(self) -> Optional[Any]:
        cached = getattr(self, "_cached_native_reward_batch", None)
        if cached is not None:
            return cached if cached is not False else None
        try:
            from POMDPPlanners.environments.laser_tag_pomdp import (  # pylint: disable=import-outside-toplevel
                _native,
            )
        except ImportError:
            # pylint: disable=attribute-defined-outside-init
            self._cached_native_reward_batch = False
            return None
        fn = getattr(_native, "lasertag_discrete_reward_batch", None)
        # pylint: disable=attribute-defined-outside-init
        self._cached_native_reward_batch = fn if fn is not None else False
        return fn

    def _compute_reward_batch_python(
        self,
        states: np.ndarray,
        action: int,
        next_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = states.shape[0]
        terminal_mask = states[:, 4].astype(bool)

        robot_r = states[:, 0].astype(np.int64)
        robot_c = states[:, 1].astype(np.int64)
        opp_r = states[:, 2].astype(np.int64)
        opp_c = states[:, 3].astype(np.int64)

        rewards = self._compute_base_rewards(action, robot_r, robot_c, opp_r, opp_c, n)

        # Realised post-action robot positions for the penalty check. When
        # ``next_states`` is None, fall back to the legacy intended-position
        # branch so callers that explicitly invoke the python helper without
        # threading a draw still observe deterministic behaviour identical
        # to the pre-refactor implementation.
        if next_states is None:
            int_r, int_c = self._compute_intended_positions(action, robot_r, robot_c)
        else:
            int_r = next_states[:, 0].astype(np.int64)
            int_c = next_states[:, 1].astype(np.int64)

        self._apply_area_penalty_batch_at(rewards, int_r, int_c)

        rewards[terminal_mask] = 0.0
        return rewards

    def _compute_base_rewards(
        self,
        action: int,
        robot_r: np.ndarray,
        robot_c: np.ndarray,
        opp_r: np.ndarray,
        opp_c: np.ndarray,
        n: int,
    ) -> np.ndarray:
        if action == 4:
            at_same_pos = (robot_r == opp_r) & (robot_c == opp_c)
            rewards = np.where(at_same_pos, float(self.tag_reward), float(-self.tag_penalty))
        else:
            rewards = np.full(n, float(-self.step_cost), dtype=np.float64)
        return rewards

    def _compute_intended_positions(
        self, action: int, robot_r: np.ndarray, robot_c: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if action in (0, 1, 2, 3):
            dr, dc = self.action_directions[action]
            return robot_r + dr, robot_c + dc
        return robot_r.copy(), robot_c.copy()

    def _apply_area_penalty_batch_at(
        self,
        rewards: np.ndarray,
        positions_r: np.ndarray,
        positions_c: np.ndarray,
    ) -> None:
        wall_mask = self._compute_wall_hit_mask(positions_r, positions_c)
        danger_mask = self._compute_danger_mask(positions_r, positions_c)

        penalty_mask = wall_mask | danger_mask
        rewards[penalty_mask] -= self.dangerous_area_penalty

    def _compute_wall_hit_mask(self, int_r: np.ndarray, int_c: np.ndarray) -> np.ndarray:
        rows, cols = self.floor_shape
        wall_grid = self._get_wall_grid()

        # Out-of-bounds positions are NOT in self.walls, so they return False here.
        # We clip to grid bounds for safe indexing, then zero-out OOB entries via the mask.
        clipped_r = np.clip(int_r, 0, rows - 1)
        clipped_c = np.clip(int_c, 0, cols - 1)
        in_bounds = (int_r >= 0) & (int_r < rows) & (int_c >= 0) & (int_c < cols)
        return in_bounds & wall_grid[clipped_r, clipped_c]

    def _compute_danger_mask(self, int_r: np.ndarray, int_c: np.ndarray) -> np.ndarray:
        if not self.dangerous_areas:
            return np.zeros(len(int_r), dtype=bool)
        centers = np.array(self.dangerous_areas, dtype=np.float64)  # shape (D, 2)
        dr = int_r[:, np.newaxis] - centers[:, 0]
        dc = int_c[:, np.newaxis] - centers[:, 1]
        dist = np.sqrt(dr**2 + dc**2)
        return np.asarray((dist <= self.dangerous_area_radius).any(axis=1), dtype=bool)

    def _get_wall_grid(self) -> np.ndarray:
        cached = getattr(self, "_wall_grid", None)
        if cached is not None:
            return cached
        grid = np.zeros(self.floor_shape, dtype=bool)
        for row, col in self.walls:
            grid[row, col] = True
        # pylint: disable=attribute-defined-outside-init
        self._wall_grid = grid
        return grid


class LaserTagZeroMeanHazardShockRewardModel(LaserTagRewardModel):
    """Dangerous-area penalty has zero mean and high variance.

    Wall hits remain deterministically penalised by ``-dangerous_area_penalty``.
    Dangerous-area hits emit ``+dangerous_area_penalty`` or
    ``-dangerous_area_penalty`` with equal probability, mirroring
    :class:`~POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models.ContinuousLDZeroMeanHazardShockRewardModel`.
    Walls and danger zones are scored independently — a position that
    is both a wall and inside a danger zone receives both contributions.
    """

    def _reward_variant_code(self) -> int:
        return 1

    def _compute_area_penalty_scalar(self, position: Tuple[int, int]) -> float:
        contrib = 0.0
        if position in self.walls:
            contrib -= self.dangerous_area_penalty
        if self._is_in_dangerous_area(position):
            # ±dangerous_area_penalty with 50/50 split (expected 0, variance penalty^2).
            if np.random.rand() < 0.5:
                contrib -= self.dangerous_area_penalty
            else:
                contrib += self.dangerous_area_penalty
        return contrib

    def _apply_area_penalty_batch_at(
        self,
        rewards: np.ndarray,
        positions_r: np.ndarray,
        positions_c: np.ndarray,
    ) -> None:
        wall_mask = self._compute_wall_hit_mask(positions_r, positions_c)
        rewards[wall_mask] -= self.dangerous_area_penalty

        danger_mask = self._compute_danger_mask(positions_r, positions_c)
        n_in = int(np.count_nonzero(danger_mask))
        if n_in > 0:
            # ``signs`` matches the per-row independent ±penalty draw. Single
            # batched ``np.random.rand`` keeps seed semantics aligned with the
            # light-dark HV batch path.
            signs = np.where(np.random.rand(n_in) < 0.5, 1.0, -1.0)
            rewards[danger_mask] += self.dangerous_area_penalty * signs


class LaserTagDistanceDecayedHazardPenaltyRewardModel(LaserTagRewardModel):
    """Dangerous-area penalty fires with distance-decaying probability.

    Wall hits remain deterministically penalised by ``-dangerous_area_penalty``.
    The dangerous-area contribution is ``-dangerous_area_penalty`` with
    probability ``exp(-min_dist / penalty_decay)`` where ``min_dist`` is
    the Euclidean distance from the realised position to the nearest
    dangerous-area centre. No radius cutoff is applied, so even faraway
    positions feel a (vanishingly small) penalty risk. Mirrors
    :class:`~POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models.ContinuousLightDarkDistanceDecayedHazardPenaltyRewardModel`.
    """

    def __init__(
        self,
        floor_shape: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        dangerous_areas: List[Tuple[int, int]],
        dangerous_area_radius: float,
        dangerous_area_penalty: float,
        tag_reward: float,
        tag_penalty: float,
        step_cost: float,
        action_directions: Dict[int, Tuple[int, int]],
        penalty_decay: float,
    ):
        super().__init__(
            floor_shape=floor_shape,
            walls=walls,
            dangerous_areas=dangerous_areas,
            dangerous_area_radius=dangerous_area_radius,
            dangerous_area_penalty=dangerous_area_penalty,
            tag_reward=tag_reward,
            tag_penalty=tag_penalty,
            step_cost=step_cost,
            action_directions=action_directions,
        )
        if penalty_decay <= 0.0:
            raise ValueError("penalty_decay must be strictly positive")
        self.penalty_decay = penalty_decay

    def _reward_variant_code(self) -> int:
        return 2

    def _penalty_decay_for_native(self) -> float:
        return float(self.penalty_decay)

    def _compute_area_penalty_scalar(self, position: Tuple[int, int]) -> float:
        contrib = 0.0
        if position in self.walls:
            contrib -= self.dangerous_area_penalty
        if self._dangerous_centers_xy.shape[1] > 0:
            point = np.array([float(position[0]), float(position[1])])
            contrib += decaying_prob_penalty_kernel(
                point,
                self._dangerous_centers_xy,
                -self.dangerous_area_penalty,
                self.penalty_decay,
                float(np.random.rand()),
            )
        return contrib

    def _apply_area_penalty_batch_at(
        self,
        rewards: np.ndarray,
        positions_r: np.ndarray,
        positions_c: np.ndarray,
    ) -> None:
        wall_mask = self._compute_wall_hit_mask(positions_r, positions_c)
        rewards[wall_mask] -= self.dangerous_area_penalty

        if self._dangerous_centers_xy.shape[1] == 0:
            return
        # Decay penalty is applied to *every* row (no radius cutoff). One
        # uniform draw per row keeps seed semantics aligned with the
        # light-dark Decaying batch path.
        points = np.ascontiguousarray(
            np.column_stack([positions_r.astype(np.float64), positions_c.astype(np.float64)])
        )
        uniforms = np.random.rand(points.shape[0])
        rewards += decaying_prob_penalty_batch_kernel(
            points,
            self._dangerous_centers_xy,
            -self.dangerous_area_penalty,
            self.penalty_decay,
            uniforms,
        )
