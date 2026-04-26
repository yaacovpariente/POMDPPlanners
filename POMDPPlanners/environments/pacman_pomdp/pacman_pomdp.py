# pylint: disable=too-many-lines
"""Module for PacMan POMDP environment.

This module provides the PacMan POMDP environment implementation inspired by the
classic arcade game. The environment features a grid world where PacMan must
collect pellets while avoiding ghosts, with partial observability of ghost positions.

The environment involves PacMan navigating a maze with walls, collecting pellets,
and avoiding ghosts that move according to stochastic policies. PacMan receives
noisy observations about nearby ghost positions. The state is a flat float64
ndarray in the canonical layout
``[pac_row, pac_col, g0_row, g0_col, ..., pellet_mask[0..P-1], score, terminal]``;
build states via :meth:`PacManPOMDP.make_state` and read fields back with
``get_pacman_pos`` / ``get_ghost_positions`` / ``get_pellets`` / ``get_score`` /
``get_terminal``.

Classes:
    PacManPOMDP: The main POMDP environment implementation
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.environments.pacman_pomdp import _native  # pylint: disable=no-name-in-module
from POMDPPlanners.utils.statistics_utils import confidence_interval

_GHOST_COORDINATION_CODES = {"independent": 0, "coordinated": 1, "mixed": 2}
_GHOST_STRATEGY_CODES = {"aggressive": 0, "patrol": 1, "ambush": 2}


class PacManPOMDPMetrics(Enum):
    """Metric names for PacMan POMDP environment."""

    WIN_RATE = "win_rate"
    AVG_PELLETS_COLLECTED = "avg_pellets_collected"
    AVG_EPISODE_LENGTH = "avg_episode_length"
    AVG_PACMAN_CLOSEST_GHOST_DISTANCE = "avg_pacman_closest_ghost_distance"
    AVG_COLLISION_ENCOUNTERS = "avg_collision_encounters"


class PacManPOMDP(DiscreteActionsEnvironment):  # pylint: disable=too-many-public-methods
    """PacMan POMDP environment inspired by the classic arcade game.

    This environment implements a simplified PacMan game where PacMan must collect
    pellets while avoiding a single ghost. The ghost position is only partially
    observable through noisy sensor readings.

    Attributes:
        maze_size: Grid dimensions as (rows, cols)
        walls: Set of wall positions as (row, col) tuples
        initial_pellets: List of initial pellet positions
        pellet_reward: Reward for collecting a pellet
        ghost_collision_penalty: Penalty for collision with ghost
        step_penalty: Cost per action
        win_reward: Reward for collecting all pellets
        ghost_aggressiveness: Temperature parameter for ghost movement policy
        observation_noise_factor: Multiplier for observation noise based on distance
        max_observation_noise: Maximum noise standard deviation

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Initialize environment
        >>> env = PacManPOMDP(maze_size=(7, 7))
        >>>
        >>> # Get initial state and actions
        >>> initial_state = env.initial_state_dist().sample()[0]
        >>> actions = env.get_actions()
        >>>
        >>> # Sample complete step using convenience method
        >>> action = actions[0]
        >>> next_state, observation, reward = env.sample_next_step(initial_state, action)
        >>>
        >>> # Check terminal condition
        >>> env.is_terminal(initial_state)
        False
    """

    def __init__(
        self,
        maze_size: Tuple[int, int] = (7, 7),
        walls: Optional[Set[Tuple[int, int]]] = None,
        initial_pellets: Optional[List[Tuple[int, int]]] = None,
        initial_pacman_pos: Tuple[int, int] = (0, 0),
        num_ghosts: int = 1,
        initial_ghost_positions: Optional[List[Tuple[int, int]]] = None,
        initial_ghost_pos: Optional[Tuple[int, int]] = None,  # Backward compatibility
        pellet_reward: float = 10.0,
        ghost_collision_penalty: float = -100.0,
        step_penalty: float = -1.0,
        win_reward: float = 100.0,
        ghost_aggressiveness: float = 2.0,
        ghost_coordination: str = "independent",
        ghost_strategies: Optional[List[str]] = None,
        observation_noise_factor: float = 0.3,
        max_observation_noise: float = 1.5,
        discount_factor: float = 0.95,
        name: str = "PacManPOMDP",
        output_dir: Optional[Path] = None,
        debug: bool = False,
    ):
        """Initialize PacMan POMDP.

        Args:
            maze_size: Grid dimensions (rows, cols). Defaults to (7, 7).
            walls: Set of wall positions. Defaults to predefined wall set.
            initial_pellets: Initial pellet positions. Defaults to corner positions.
            initial_pacman_pos: PacMan starting position. Defaults to (0, 0).
            num_ghosts: Number of ghosts in the game. Defaults to 1.
            initial_ghost_positions: Ghost starting positions. Defaults to None (auto-generated).
            initial_ghost_pos: Single ghost position (backward compatibility). Defaults to None.
            pellet_reward: Reward for collecting pellets. Defaults to 10.0.
            ghost_collision_penalty: Penalty for ghost collision. Defaults to -100.0.
            step_penalty: Cost per action. Defaults to -1.0.
            win_reward: Reward for collecting all pellets. Defaults to 100.0.
            ghost_aggressiveness: Ghost pursuit intensity. Defaults to 2.0.
            ghost_coordination: Ghost coordination strategy ("independent", "coordinated", "mixed"). Defaults to "independent".
            ghost_strategies: Individual ghost strategies (list of "aggressive", "patrol", "ambush"). Defaults to None.
            observation_noise_factor: Observation noise factor. Defaults to 0.3.
            max_observation_noise: Maximum observation noise. Defaults to 1.5.
            discount_factor: Discount factor. Defaults to 0.95.
            name: Environment name. Defaults to "PacManPOMDP".
            output_dir: Output directory for logging. Defaults to None.
            debug: Enable debug logging. Defaults to False.
        """
        # Calculate reward range based on parameters
        min_reward = step_penalty + ghost_collision_penalty
        max_reward = step_penalty + pellet_reward + win_reward

        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )

        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=(min_reward, max_reward),
            output_dir=output_dir,
            debug=debug,
        )

        if walls is None:
            walls = {(2, 2), (2, 3), (3, 2), (4, 4), (3, 5)}
        self.maze_size = maze_size
        self.walls = walls
        self.initial_pacman_pos = initial_pacman_pos
        self.num_ghosts = num_ghosts
        self.pellet_reward = pellet_reward
        self.ghost_collision_penalty = ghost_collision_penalty
        self.step_penalty = step_penalty
        self.win_reward = win_reward
        self.ghost_aggressiveness = ghost_aggressiveness
        self.ghost_coordination = ghost_coordination
        self.ghost_strategies = ghost_strategies or ["aggressive"] * num_ghosts
        self.observation_noise_factor = observation_noise_factor
        self.max_observation_noise = max_observation_noise

        # Handle ghost positions (backward compatibility)
        if initial_ghost_positions is not None:
            if len(initial_ghost_positions) != num_ghosts:
                raise ValueError(
                    f"initial_ghost_positions length ({len(initial_ghost_positions)}) must match num_ghosts ({num_ghosts})"
                )
            self.initial_ghost_positions = initial_ghost_positions
        elif initial_ghost_pos is not None:
            # Backward compatibility: single ghost position provided
            if num_ghosts != 1:
                raise ValueError("initial_ghost_pos can only be used when num_ghosts=1")
            self.initial_ghost_positions = [initial_ghost_pos]
        else:
            # Auto-generate ghost positions
            self.initial_ghost_positions = self._generate_ghost_positions(num_ghosts)

        # Set default pellets if none provided
        if initial_pellets is None:
            # Place pellets away from corners to avoid conflicts with initial positions
            self.initial_pellets: List[Tuple[int, int]] = [
                (1, 1),
                (1, maze_size[1] - 2),
                (maze_size[0] - 2, 1),
                (maze_size[0] - 2, maze_size[1] - 2),
            ]
        else:
            self.initial_pellets = initial_pellets

        # Validate parameters
        self._validate_parameters()

        # Define actions: 0=North, 1=East, 2=South, 3=West
        self.action_names = ["north", "east", "south", "west"]

        # Action to direction vector mapping for visualization
        self.action_to_vector = {
            0: (0, -1),  # north - up (negative row)
            1: (1, 0),  # east - right (positive col)
            2: (0, 1),  # south - down (positive row)
            3: (-1, 0),  # west - left (negative col)
        }

        # Patrol-direction state, mutated in place by the C++ transition kernel.
        self.ghost_patrol_directions: np.ndarray = np.zeros(self.num_ghosts, dtype=np.int32)

        # Precompute array state layout for vectorized belief support
        self._all_pellet_positions = tuple(self.initial_pellets)
        self._num_initial_pellets = len(self._all_pellet_positions)
        self._pellet_to_index = {pos: i for i, pos in enumerate(self._all_pellet_positions)}

        self._idx_pac_row = 0
        self._idx_pac_col = 1
        self._idx_ghosts_start = 2
        self._idx_ghosts_end = 2 + 2 * self.num_ghosts
        self._idx_pellets_start = self._idx_ghosts_end
        self._idx_pellets_end = self._idx_pellets_start + self._num_initial_pellets
        self._idx_score = self._idx_pellets_end
        self._idx_terminal = self._idx_pellets_end + 1
        self._state_dim = self._idx_terminal + 1
        self._cached_neighbor_table: Optional[np.ndarray] = None
        self._cached_neighbor_validity: Optional[np.ndarray] = None
        self._cached_transition_cpp_ctor_kwargs: Optional[Dict[str, Any]] = None

    @property
    def initial_ghost_pos(self) -> Tuple[int, int]:
        """Backward compatibility: returns first ghost position."""
        return self.initial_ghost_positions[0] if self.initial_ghost_positions else (0, 0)

    def observation_to_array(self, obs: Tuple[Tuple[int, int], ...]) -> np.ndarray:
        """Convert a PacMan observation tuple to a flat numpy array.

        Args:
            obs: Observation as tuple of ghost (row, col) positions.

        Returns:
            1-D array of shape ``(2 * num_ghosts,)``.
        """
        return np.array([coord for gpos in obs for coord in gpos], dtype=np.float64)

    def array_to_observation(self, arr: np.ndarray) -> Tuple[Tuple[int, int], ...]:
        """Convert a flat numpy array back to a PacMan observation tuple.

        Args:
            arr: 1-D array of shape ``(2 * num_ghosts,)``.

        Returns:
            Observation as tuple of (row, col) tuples.
        """
        flat = arr.ravel()
        return tuple((int(flat[2 * g]), int(flat[2 * g + 1])) for g in range(self.num_ghosts))

    # ------------------------------------------------------------------
    # Array-state factory and readers
    # ------------------------------------------------------------------

    def make_state(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        *,
        pacman_pos: Tuple[int, int],
        ghost_positions: Tuple[Tuple[int, int], ...],
        pellets: Optional[Tuple[Tuple[int, int], ...]] = None,
        score: float = 0.0,
        terminal: bool = False,
    ) -> np.ndarray:
        """Build a PacMan state array in the canonical layout.

        The array layout is
        ``[pac_row, pac_col, g0_row, g0_col, ..., pellet_mask[0..P-1], score, terminal]``.

        Args:
            pacman_pos: PacMan grid position ``(row, col)``.
            ghost_positions: Per-ghost positions as a tuple of length ``num_ghosts``.
            pellets: Active pellet positions. ``None`` means every initial pellet
                is active (useful for constructing initial states).
            score: Current game score.
            terminal: Whether the state is terminal.

        Returns:
            1-D ``float64`` array of shape ``(self._state_dim,)``.

        Raises:
            ValueError: If any argument has the wrong type or length, or if a
                pellet position was not registered at env construction.
        """
        self._validate_make_state_args(pacman_pos, ghost_positions, pellets)
        arr = np.zeros(self._state_dim, dtype=np.float64)
        arr[self._idx_pac_row] = pacman_pos[0]
        arr[self._idx_pac_col] = pacman_pos[1]
        for g, gpos in enumerate(ghost_positions):
            arr[self._idx_ghosts_start + 2 * g] = gpos[0]
            arr[self._idx_ghosts_start + 2 * g + 1] = gpos[1]
        active_positions = self._all_pellet_positions if pellets is None else pellets
        for pos in active_positions:
            idx = self._pellet_to_index.get(pos)
            if idx is not None:
                arr[self._idx_pellets_start + idx] = 1.0
        arr[self._idx_score] = float(score)
        arr[self._idx_terminal] = 1.0 if terminal else 0.0
        return arr

    def _validate_make_state_args(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: Tuple[Tuple[int, int], ...],
        pellets: Optional[Tuple[Tuple[int, int], ...]],
    ) -> None:
        if not isinstance(pacman_pos, tuple) or len(pacman_pos) != 2:
            raise ValueError("pacman_pos must be a tuple of two integers")
        if not isinstance(ghost_positions, tuple):
            raise ValueError("ghost_positions must be a tuple of position tuples")
        if len(ghost_positions) != self.num_ghosts:
            raise ValueError(
                f"ghost_positions length ({len(ghost_positions)}) must equal "
                f"num_ghosts ({self.num_ghosts})"
            )
        for i, gpos in enumerate(ghost_positions):
            if not isinstance(gpos, tuple) or len(gpos) != 2:
                raise ValueError(f"ghost_positions[{i}] must be a tuple of two integers")
        if pellets is None:
            return
        if not isinstance(pellets, tuple):
            raise ValueError("pellets must be a tuple of position tuples")
        for i, pos in enumerate(pellets):
            if not isinstance(pos, tuple) or len(pos) != 2:
                raise ValueError(f"pellets[{i}] must be a tuple of two integers")

    def get_pacman_pos(self, state: np.ndarray) -> Tuple[int, int]:
        """Return PacMan's ``(row, col)`` position from a state array."""
        return int(state[self._idx_pac_row]), int(state[self._idx_pac_col])

    def get_ghost_positions(self, state: np.ndarray) -> Tuple[Tuple[int, int], ...]:
        """Return ghost positions as a tuple of ``(row, col)`` pairs."""
        return tuple(
            (
                int(state[self._idx_ghosts_start + 2 * g]),
                int(state[self._idx_ghosts_start + 2 * g + 1]),
            )
            for g in range(self.num_ghosts)
        )

    def get_pellets(self, state: np.ndarray) -> Tuple[Tuple[int, int], ...]:
        """Return the tuple of active pellet positions."""
        mask = state[self._idx_pellets_start : self._idx_pellets_end]
        return self._pellet_mask_to_positions(mask)

    def _pellet_mask_to_positions(self, mask: np.ndarray) -> Tuple[Tuple[int, int], ...]:
        return tuple(pos for pos, idx in self._pellet_to_index.items() if mask[idx] > 0.5)

    def get_score(self, state: np.ndarray) -> float:
        """Return the state's score as a Python float."""
        return float(state[self._idx_score])

    def get_terminal(self, state: np.ndarray) -> bool:
        """Return whether the state is terminal."""
        return bool(state[self._idx_terminal] > 0.5)

    def _require_state_array(self, state: Any) -> np.ndarray:
        if not isinstance(state, np.ndarray) or state.shape != (self._state_dim,):
            raise TypeError(
                f"expected np.ndarray of shape ({self._state_dim},); "
                f"use PacManPOMDP.make_state(...) to build a state"
            )
        return state

    def _generate_ghost_positions(self, num_ghosts: int) -> List[Tuple[int, int]]:
        """Generate ghost starting positions automatically."""
        rows, cols = self.maze_size

        # Try to place ghosts in corners first, then other positions
        corner_positions = [
            (rows - 1, cols - 1),  # Bottom-right
            (0, cols - 1),  # Top-right
            (rows - 1, 0),  # Bottom-left
            (rows // 2, cols // 2),  # Center
        ]

        # Filter out walls and PacMan position
        available_positions = []
        for pos in corner_positions:
            if pos not in self.walls and pos != self.initial_pacman_pos:
                available_positions.append(pos)

        # If we need more positions, add valid positions from the maze
        if len(available_positions) < num_ghosts:
            for r in range(rows):
                for c in range(cols):
                    pos = (r, c)
                    if (
                        pos not in self.walls
                        and pos != self.initial_pacman_pos
                        and pos not in available_positions
                    ):
                        available_positions.append(pos)
                        if len(available_positions) >= num_ghosts:
                            break
                if len(available_positions) >= num_ghosts:
                    break

        if len(available_positions) < num_ghosts:
            raise ValueError(
                f"Cannot place {num_ghosts} ghosts in maze with {len(available_positions)} available positions"
            )

        return available_positions[:num_ghosts]

    def _validate_parameters(self):  # pylint: disable=too-many-branches
        """Validate environment parameters."""
        if self.maze_size[0] <= 0 or self.maze_size[1] <= 0:
            raise ValueError("Maze size must be positive")

        if self.num_ghosts <= 0:
            raise ValueError("Number of ghosts must be positive")

        # Check PacMan position is within bounds
        if not (
            0 <= self.initial_pacman_pos[0] < self.maze_size[0]
            and 0 <= self.initial_pacman_pos[1] < self.maze_size[1]
        ):
            raise ValueError(
                f"PacMan position {self.initial_pacman_pos} is outside maze bounds {self.maze_size}"
            )

        # Check all ghost positions are within bounds
        for i, ghost_pos in enumerate(self.initial_ghost_positions):
            if not (
                0 <= ghost_pos[0] < self.maze_size[0] and 0 <= ghost_pos[1] < self.maze_size[1]
            ):
                raise ValueError(
                    f"Ghost {i} position {ghost_pos} is outside maze bounds {self.maze_size}"
                )

        # Check pellets are within bounds and not walls
        for pellet_pos in self.initial_pellets:
            if not (
                0 <= pellet_pos[0] < self.maze_size[0] and 0 <= pellet_pos[1] < self.maze_size[1]
            ):
                raise ValueError(
                    f"Pellet position {pellet_pos} is outside maze bounds {self.maze_size}"
                )
            if pellet_pos in self.walls:
                raise ValueError(f"Pellet position {pellet_pos} is inside a wall")

        # Check initial positions are not walls
        if self.initial_pacman_pos in self.walls:
            raise ValueError(f"Initial PacMan position {self.initial_pacman_pos} is inside a wall")

        for i, ghost_pos in enumerate(self.initial_ghost_positions):
            if ghost_pos in self.walls:
                raise ValueError(f"Initial ghost {i} position {ghost_pos} is inside a wall")

        # Validate ghost coordination strategy
        valid_coordination = ["independent", "coordinated", "mixed"]
        if self.ghost_coordination not in valid_coordination:
            raise ValueError(f"ghost_coordination must be one of {valid_coordination}")

        # Validate ghost strategies
        valid_strategies = ["aggressive", "patrol", "ambush"]
        for i, strategy in enumerate(self.ghost_strategies):
            if strategy not in valid_strategies:
                raise ValueError(f"ghost_strategies[{i}] must be one of {valid_strategies}")

    def get_actions(self) -> List[int]:
        """Get all available actions."""
        return list(range(len(self.action_names)))

    # ── Env-API sampling/log-prob methods ───────────────────────────
    # These construct the C++ native kernel directly from per-env cached
    # ctor kwargs and run sampling / probability evaluation in C++. The
    # earlier per-call PacMan{Transition,Observation} Python wrappers
    # were deleted in PR-D-Pacman.

    def sample_next_state(self, state: np.ndarray, action: int, n_samples: int = 1) -> Any:
        kernel = _native.PacManTransitionCpp(
            state=self._require_state_array(state),
            action=int(action),
            **self.get_transition_cpp_ctor_kwargs(),
            patrol_dir_state=self.ghost_patrol_directions,
        )
        samples = kernel.sample(n_samples)
        if n_samples == 1:
            return samples[0]
        return samples

    def sample_observation(self, next_state: np.ndarray, action: int, n_samples: int = 1) -> Any:
        kernel = _native.PacManObservationCpp(
            next_state=self._require_state_array(next_state),
            action=int(action),
            **self.get_observation_cpp_ctor_kwargs(),
        )
        arrays = kernel.sample(n_samples)
        if n_samples == 1:
            return self.array_to_observation(arrays[0])
        return [self.array_to_observation(arr) for arr in arrays]

    def transition_log_probability(
        self, state: np.ndarray, action: int, next_states: Any
    ) -> np.ndarray:
        kernel = _native.PacManTransitionCpp(
            state=self._require_state_array(state),
            action=int(action),
            **self.get_transition_cpp_ctor_kwargs(),
            patrol_dir_state=self.ghost_patrol_directions,
        )
        # Accept either a sequence of 1-D state arrays or a 2-D ndarray.
        if isinstance(next_states, np.ndarray) and next_states.ndim == 2:
            stacked = next_states
        else:
            stacked = np.stack([np.asarray(s, dtype=np.float64) for s in next_states])
        probs = np.asarray(kernel.probability(stacked))
        return np.log(probs + 1e-300)

    def observation_log_probability(
        self, next_state: np.ndarray, action: int, observations: Any
    ) -> np.ndarray:
        kernel = _native.PacManObservationCpp(
            next_state=self._require_state_array(next_state),
            action=int(action),
            **self.get_observation_cpp_ctor_kwargs(),
        )
        # Accept either a 2-D ndarray of shape (N, 2*num_ghosts) or a sequence
        # of public tuple-of-(row, col) observations.
        if isinstance(observations, np.ndarray) and observations.ndim == 2:
            stacked = observations
            probs = np.asarray(kernel.probability(stacked))
            return np.log(probs + 1e-300)
        n = len(observations)
        probs = np.zeros(n, dtype=np.float64)
        usable_rows: List[np.ndarray] = []
        usable_indices: List[int] = []
        for i, obs in enumerate(observations):
            if len(obs) != self.num_ghosts:
                continue  # wrong ghost count -> probability 0 -> -inf
            usable_rows.append(self.observation_to_array(obs))
            usable_indices.append(i)
        if usable_rows:
            stacked = np.stack(usable_rows)
            sub_probs = np.asarray(kernel.probability(stacked))
            for idx, p in zip(usable_indices, sub_probs):
                probs[idx] = p
        return np.log(probs + 1e-300)

    def sample_next_state_batch(self, states: Any, action: int) -> np.ndarray:
        if isinstance(states, np.ndarray) and states.ndim == 2:
            states_array = np.ascontiguousarray(states, dtype=np.float64)
        else:
            states_array = np.ascontiguousarray(
                np.stack([self._require_state_array(s) for s in states]),
                dtype=np.float64,
            )
        kernel = _native.PacManTransitionCpp(
            state=states_array[0],
            action=int(action),
            **self.get_transition_cpp_ctor_kwargs(),
            patrol_dir_state=self.ghost_patrol_directions,
        )
        return np.asarray(kernel.batch_sample(states_array), dtype=np.float64)

    def observation_log_probability_per_state(
        self, next_states: Any, action: int, observation: Any
    ) -> np.ndarray:
        if isinstance(next_states, np.ndarray) and next_states.ndim == 2:
            next_states_array = np.ascontiguousarray(next_states, dtype=np.float64)
        else:
            next_states_array = np.ascontiguousarray(
                np.stack([self._require_state_array(s) for s in next_states]),
                dtype=np.float64,
            )
        if isinstance(observation, np.ndarray):
            observation_array = np.ascontiguousarray(observation, dtype=np.float64).ravel()
        else:
            observation_array = np.ascontiguousarray(
                self.observation_to_array(observation), dtype=np.float64
            )
        kernel = _native.PacManObservationCpp(
            next_state=next_states_array[0],
            action=int(action),
            **self.get_observation_cpp_ctor_kwargs(),
        )
        return np.asarray(
            kernel.batch_log_likelihood(
                next_particles=next_states_array,
                observation=observation_array,
            ),
            dtype=np.float64,
        )

    def reward(self, state: np.ndarray, action: int) -> float:
        """Calculate immediate reward."""
        state = self._require_state_array(state)
        if state[self._idx_terminal] > 0.5:
            return 0.0

        total_reward = self.step_penalty
        next_state = self.sample_next_state(state, action)

        next_pac_row = int(next_state[self._idx_pac_row])
        next_pac_col = int(next_state[self._idx_pac_col])
        for g in range(self.num_ghosts):
            g_row = int(next_state[self._idx_ghosts_start + 2 * g])
            g_col = int(next_state[self._idx_ghosts_start + 2 * g + 1])
            if next_pac_row == g_row and next_pac_col == g_col:
                total_reward += self.ghost_collision_penalty
                break

        if next_state[self._idx_score] > state[self._idx_score]:
            total_reward += self.pellet_reward

        if next_state[self._idx_terminal] > 0.5:
            pellet_mask = next_state[self._idx_pellets_start : self._idx_pellets_end]
            if not np.any(pellet_mask > 0.5):
                total_reward += self.win_reward

        return total_reward

    def reward_batch(  # type: ignore[override]
        self, states: Union[np.ndarray, Sequence[Any]], action: int
    ) -> np.ndarray:
        """Calculate rewards for a batch of states.

        Accepts a 2-D numpy array of shape ``(N, state_dim)`` on the fast
        vectorized path, or a sequence of 1-D state arrays on the fallback
        per-particle path.

        Computes deterministic reward components only: step penalty, pellet
        collection, and win bonus. Ghost collision penalty is excluded because
        it depends on stochastic ghost movement.

        Args:
            states: Array of shape ``(N, state_dim)`` or sequence of 1-D
                state arrays.
            action: Discrete action index (0-3).

        Returns:
            1-D array of reward values with shape ``(N,)``.
        """
        states_arr = np.asarray(states)
        if states_arr.dtype.kind == "f":
            if states_arr.ndim == 1:
                states_arr = states_arr.reshape(1, -1)
            return self._compute_reward_batch(states_arr, action)
        return np.array([self.reward(states[i], action) for i in range(len(states))])

    def _compute_reward_batch(self, states_arr: np.ndarray, action: int) -> np.ndarray:
        terminal = states_arr[:, self._idx_terminal] > 0.5
        rewards = np.where(terminal, 0.0, self.step_penalty)

        new_pac_rows, new_pac_cols = self._batch_move_pacman(states_arr, action)
        pellet_collected, all_collected = self._batch_check_pellets(
            states_arr, new_pac_rows, new_pac_cols
        )
        rewards += np.where(~terminal & pellet_collected, self.pellet_reward, 0.0)
        rewards += np.where(~terminal & all_collected, self.win_reward, 0.0)
        return rewards

    def _batch_move_pacman(
        self, states_arr: np.ndarray, action: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        pac_rows = states_arr[:, self._idx_pac_row].astype(np.int32)
        pac_cols = states_arr[:, self._idx_pac_col].astype(np.int32)
        table = self._get_neighbor_table()
        new_positions = table[pac_rows, pac_cols, action]
        return new_positions[:, 0], new_positions[:, 1]

    def _batch_check_pellets(
        self,
        states_arr: np.ndarray,
        pac_rows: np.ndarray,
        pac_cols: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = states_arr.shape[0]
        pellet_mask = states_arr[:, self._idx_pellets_start : self._idx_pellets_end]
        pellet_pos = np.array(self._all_pellet_positions, dtype=np.int32)

        if len(pellet_pos) == 0:
            return np.zeros(n, dtype=bool), np.ones(n, dtype=bool)

        # Check if new pacman position matches any active pellet
        row_match = pac_rows[:, None] == pellet_pos[None, :, 0]
        col_match = pac_cols[:, None] == pellet_pos[None, :, 1]
        pos_match = row_match & col_match
        active_match = pos_match & (pellet_mask > 0.5)
        collected = active_match.any(axis=1)

        remaining_after = pellet_mask.sum(axis=1) - collected.astype(np.float64)
        all_collected = collected & (remaining_after < 0.5)
        return collected, all_collected

    def _get_neighbor_table(self) -> np.ndarray:
        if self._cached_neighbor_table is None:
            self._populate_neighbor_caches()
        assert self._cached_neighbor_table is not None
        return self._cached_neighbor_table

    def _get_neighbor_validity(self) -> np.ndarray:
        if self._cached_neighbor_validity is None:
            self._populate_neighbor_caches()
        assert self._cached_neighbor_validity is not None
        return self._cached_neighbor_validity

    def _populate_neighbor_caches(self) -> None:
        # Deferred import — `pacman_pomdp_beliefs` package imports back from
        # this module, so a top-level import would cycle.
        from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_beliefs.pacman_grid_utils import (  # pylint: disable=import-outside-toplevel
            precompute_neighbor_table,
            precompute_neighbor_validity,
            precompute_valid_cell_mask,
        )

        valid_mask = precompute_valid_cell_mask(self.maze_size, self.walls)
        self._cached_neighbor_table = precompute_neighbor_table(self.maze_size, valid_mask)
        self._cached_neighbor_validity = precompute_neighbor_validity(self.maze_size, valid_mask)

    def get_observation_cpp_ctor_kwargs(self) -> Dict[str, Any]:
        """Return the kwargs dict passed to PacManObservationCpp."""
        return {
            "num_ghosts": int(self.num_ghosts),
            "maze_rows": int(self.maze_size[0]),
            "maze_cols": int(self.maze_size[1]),
            "observation_noise_factor": float(self.observation_noise_factor),
            "max_observation_noise": float(self.max_observation_noise),
            "idx_pac_row": self._idx_pac_row,
            "idx_pac_col": self._idx_pac_col,
            "idx_ghosts_start": self._idx_ghosts_start,
            "idx_terminal": self._idx_terminal,
        }

    def get_transition_cpp_ctor_kwargs(self) -> Dict[str, Any]:
        """Return the cached per-env kwargs dict passed to PacManTransitionCpp."""
        if self._cached_transition_cpp_ctor_kwargs is None:
            self._cached_transition_cpp_ctor_kwargs = self._build_transition_cpp_ctor_kwargs()
        return self._cached_transition_cpp_ctor_kwargs

    def _build_transition_cpp_ctor_kwargs(self) -> Dict[str, Any]:
        neighbor_table = np.ascontiguousarray(self._get_neighbor_table(), dtype=np.int32)
        neighbor_validity = np.ascontiguousarray(
            self._get_neighbor_validity().astype(np.uint8), dtype=np.uint8
        )
        pellet_positions = np.ascontiguousarray(
            np.asarray(self._all_pellet_positions, dtype=np.int32).reshape(-1, 2),
            dtype=np.int32,
        )
        ghost_strategy_codes = np.array(
            [_GHOST_STRATEGY_CODES[s] for s in self.ghost_strategies],
            dtype=np.int32,
        )
        return {
            "maze_rows": int(self.maze_size[0]),
            "maze_cols": int(self.maze_size[1]),
            "neighbor_table": neighbor_table,
            "neighbor_validity": neighbor_validity,
            "pellet_positions": pellet_positions,
            "ghost_aggressiveness": float(self.ghost_aggressiveness),
            "ghost_coordination_code": _GHOST_COORDINATION_CODES[self.ghost_coordination],
            "ghost_strategy_codes": ghost_strategy_codes,
            "num_ghosts": int(self.num_ghosts),
            "num_pellets": int(self._num_initial_pellets),
            "pellet_reward": float(self.pellet_reward),
            "idx_pac_row": self._idx_pac_row,
            "idx_pac_col": self._idx_pac_col,
            "idx_ghosts_start": self._idx_ghosts_start,
            "idx_pellets_start": self._idx_pellets_start,
            "idx_pellets_end": self._idx_pellets_end,
            "idx_score": self._idx_score,
            "idx_terminal": self._idx_terminal,
        }

    def is_terminal(self, state: np.ndarray) -> bool:
        """Check if state is terminal."""
        return self.get_terminal(self._require_state_array(state))

    def initial_state_dist(self) -> DiscreteDistribution:
        """Get initial state distribution."""
        initial_state = self.make_state(
            pacman_pos=self.initial_pacman_pos,
            ghost_positions=tuple(self.initial_ghost_positions),
            pellets=tuple(self.initial_pellets),
            score=0.0,
            terminal=False,
        )
        return DiscreteDistribution(values=[initial_state], probs=np.array([1.0]))

    def initial_observation_dist(self) -> DiscreteDistribution:
        """Get initial observation distribution."""
        # Initial observation is the true ghost position with some noise
        initial_obs = self.sample_observation(
            next_state=self.initial_state_dist().sample()[0], action=0  # Dummy action
        )

        return DiscreteDistribution(values=[initial_obs], probs=np.array([1.0]))

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        """Check if two observations are equal."""
        return observation1 == observation2

    def _check_episode_win_status(self, final_state: np.ndarray) -> int:
        pellet_mask = final_state[self._idx_pellets_start : self._idx_pellets_end]
        won = self.get_terminal(final_state) and not bool(np.any(pellet_mask > 0.5))
        return 1 if won else 0

    def _count_pellets_collected(self, final_state: np.ndarray) -> int:
        pellet_mask = final_state[self._idx_pellets_start : self._idx_pellets_end]
        remaining_pellets = int(np.sum(pellet_mask > 0.5))
        return len(self.initial_pellets) - remaining_pellets

    def _calculate_manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions."""
        return float(abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))

    def _get_closest_ghost_distance(
        self, pacman_pos: Tuple[int, int], ghost_positions: Sequence[Tuple[int, int]]
    ) -> Optional[float]:
        """Get the distance to the closest ghost from PacMan's position."""
        if not ghost_positions:
            return None

        ghost_distances = [
            self._calculate_manhattan_distance(pacman_pos, ghost_pos)
            for ghost_pos in ghost_positions
        ]
        return min(ghost_distances)

    def _is_collision(
        self, pacman_pos: Tuple[int, int], ghost_positions: Sequence[Tuple[int, int]]
    ) -> bool:
        """Check if PacMan is at the same position as any ghost."""
        return pacman_pos in ghost_positions

    def _collect_step_distances_and_collisions(self, history: History) -> Tuple[List[float], int]:
        """Collect distances to closest ghost and collision count from episode history."""
        episode_distances = []
        episode_collisions = 0

        for step_data in history.history:
            state = step_data.state
            if not isinstance(state, np.ndarray) or state.shape != (self._state_dim,):
                continue
            pacman_pos = self.get_pacman_pos(state)
            ghost_positions = self.get_ghost_positions(state)

            closest_distance = self._get_closest_ghost_distance(pacman_pos, ghost_positions)
            if closest_distance is not None:
                episode_distances.append(closest_distance)

            if self._is_collision(pacman_pos, ghost_positions):
                episode_collisions += 1

        return episode_distances, episode_collisions

    def _process_episode_metrics(self, history: History) -> Dict[str, Any]:
        """Process metrics for a single episode."""
        metrics_data = {
            "episode_length": len(history.history),
            "won": 0,
            "pellets_collected": 0,
            "avg_distance": None,
            "collisions": 0,
        }

        if not history.history:
            return metrics_data

        final_state = history.history[-1].state

        if not isinstance(final_state, np.ndarray) or final_state.shape != (self._state_dim,):
            return metrics_data

        # Calculate win status and pellets collected
        metrics_data["won"] = self._check_episode_win_status(final_state)
        metrics_data["pellets_collected"] = self._count_pellets_collected(final_state)

        # Collect distances and collisions from episode steps
        episode_distances, episode_collisions = self._collect_step_distances_and_collisions(history)

        if episode_distances:
            metrics_data["avg_distance"] = float(np.mean(episode_distances))

        metrics_data["collisions"] = episode_collisions
        return metrics_data

    def _create_metric_value(self, name: str, values: List[float]) -> Optional[MetricValue]:
        """Create a MetricValue with confidence intervals."""
        if not values:
            return None

        avg_value = float(np.mean(values))
        ci_low, ci_high = confidence_interval(values)
        return MetricValue(
            name=name,
            value=avg_value,
            lower_confidence_bound=ci_low,
            upper_confidence_bound=ci_high,
        )

    def _collect_distances_per_ghost_from_episode(self, history: History) -> List[List[float]]:
        """Collect distances to each ghost for all steps in an episode.

        Returns a list where index i contains all distances to ghost i throughout the episode.
        """
        ghost_distances_per_episode: List[List[float]] = [[] for _ in range(self.num_ghosts)]

        for step_data in history.history:
            state = step_data.state
            if not isinstance(state, np.ndarray) or state.shape != (self._state_dim,):
                continue
            pacman_pos = self.get_pacman_pos(state)
            for ghost_id, ghost_pos in enumerate(self.get_ghost_positions(state)):
                if ghost_id < len(ghost_distances_per_episode):
                    distance = self._calculate_manhattan_distance(pacman_pos, ghost_pos)
                    ghost_distances_per_episode[ghost_id].append(distance)

        return ghost_distances_per_episode

    def _calculate_average_distances_per_ghost(
        self, ghost_distances_per_episode: List[List[float]]
    ) -> List[float]:
        """Calculate average distance to each ghost for an episode."""
        return [
            float(np.mean(dist_list)) if dist_list else 0.0
            for dist_list in ghost_distances_per_episode
        ]

    def _aggregate_episode_distances_by_ghost(self, histories: List[History]) -> List[List[float]]:
        """Aggregate average distances per ghost across all episodes.

        Returns a list where index i contains average distances to ghost i from each episode.
        """
        per_ghost_avg_distances: List[List[float]] = []

        for history in histories:
            if not history.history:
                continue

            ghost_distances = self._collect_distances_per_ghost_from_episode(history)
            episode_ghost_avgs = self._calculate_average_distances_per_ghost(ghost_distances)
            per_ghost_avg_distances.append(episode_ghost_avgs)

        return per_ghost_avg_distances

    def _create_metrics_for_each_ghost(
        self, per_ghost_avg_distances: List[List[float]]
    ) -> List[MetricValue]:
        """Create MetricValue objects for each ghost's distance statistics."""
        metrics = []

        for ghost_id in range(self.num_ghosts):
            ghost_distance_values = [
                episode_avgs[ghost_id]
                for episode_avgs in per_ghost_avg_distances
                if ghost_id < len(episode_avgs)
            ]
            metric = self._create_metric_value(
                f"avg_pacman_ghost_{ghost_id}_distance", ghost_distance_values
            )
            if metric:
                metrics.append(metric)

        return metrics

    def _compute_per_ghost_distance_metrics(self, histories: List[History]) -> List[MetricValue]:
        """Compute per-ghost distance metrics for multi-ghost scenarios."""
        per_ghost_avg_distances = self._aggregate_episode_distances_by_ghost(histories)

        if not per_ghost_avg_distances:
            return []

        return self._create_metrics_for_each_ghost(per_ghost_avg_distances)

    def get_metric_names(self) -> List[str]:
        """Get names of PacMan POMDP specific metrics.

        Returns:
            List containing metric names including standard metrics (win_rate,
            avg_pellets_collected, avg_episode_length, avg_pacman_closest_ghost_distance,
            avg_collision_encounters) and dynamically generated per-ghost distance metrics
            for multi-ghost scenarios (avg_pacman_ghost_0_distance, avg_pacman_ghost_1_distance, etc.)
        """
        # Start with standard metrics
        metric_names = [metric.value for metric in PacManPOMDPMetrics]

        # Add dynamic per-ghost metrics for multi-ghost scenarios
        if self.num_ghosts > 1:
            for ghost_id in range(self.num_ghosts):
                metric_names.append(f"avg_pacman_ghost_{ghost_id}_distance")

        return metric_names

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        """Compute environment-specific metrics."""
        if not histories:
            return []

        # Collect metrics from all episodes
        wins = []
        pellets_collected = []
        episode_lengths = []
        pacman_ghost_distances = []
        collision_encounters = []

        for history in histories:
            episode_data = self._process_episode_metrics(history)
            episode_lengths.append(episode_data["episode_length"])
            wins.append(episode_data["won"])
            pellets_collected.append(episode_data["pellets_collected"])
            collision_encounters.append(episode_data["collisions"])

            if episode_data["avg_distance"] is not None:
                pacman_ghost_distances.append(episode_data["avg_distance"])

        # Create standard metrics using helper
        metrics = []
        metric_definitions = [
            (PacManPOMDPMetrics.WIN_RATE.value, wins),
            (PacManPOMDPMetrics.AVG_PELLETS_COLLECTED.value, pellets_collected),
            (PacManPOMDPMetrics.AVG_EPISODE_LENGTH.value, episode_lengths),
            (PacManPOMDPMetrics.AVG_PACMAN_CLOSEST_GHOST_DISTANCE.value, pacman_ghost_distances),
            (PacManPOMDPMetrics.AVG_COLLISION_ENCOUNTERS.value, collision_encounters),
        ]

        for name, values in metric_definitions:
            metric = self._create_metric_value(name, values)
            if metric:
                metrics.append(metric)

        # Multi-ghost specific metrics
        if self.num_ghosts > 1:
            per_ghost_metrics = self._compute_per_ghost_distance_metrics(histories)
            metrics.extend(per_ghost_metrics)

        return metrics

    def visualize_path(self, path: List[np.ndarray], actions: List[int], cache_path: Path):
        """Visualize PacMan path through the maze using sprite-based rendering.

        Args:
            path: List of state arrays representing the path through the maze.
            actions: List of actions taken at each step.
            cache_path: Path where the GIF should be saved.
        """
        from POMDPPlanners.environments.pacman_pomdp.pacman_visualizer import (
            PacManVisualizer,
        )  # pylint: disable=import-outside-toplevel

        visualizer = PacManVisualizer(self)
        visualizer.visualize_path(path, actions, cache_path)

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Cache visualization of episode history.

        Args:
            history: List of StepData objects representing the episode
            cache_path: Path where the GIF should be saved
        """
        from POMDPPlanners.environments.pacman_pomdp.pacman_visualizer import (
            PacManVisualizer,
        )  # pylint: disable=import-outside-toplevel

        visualizer = PacManVisualizer(self)
        visualizer.cache_visualization(history, cache_path)


def create_simple_maze_pacman(
    maze_size: int = 7,
    num_walls: int = 5,
    num_ghosts: int = 1,
    seed: Optional[int] = None,
) -> PacManPOMDP:
    """Create a simple PacMan instance with random walls and multiple ghosts.

    Args:
        maze_size: Size of square maze. Defaults to 7.
        num_walls: Number of walls to place randomly. Defaults to 5.
        num_ghosts: Number of ghosts in the game. Defaults to 1.
        seed: Random seed. Defaults to None.

    Returns:
        Randomly configured PacMan POMDP with multi-ghost support
    """
    if seed is not None:
        np.random.seed(seed)

    # Define pellet positions first
    pellets = [
        (1, 1),
        (1, maze_size - 2),
        (maze_size - 2, 1),
        (maze_size - 2, maze_size - 2),
    ]

    # Create walls randomly, avoiding corners, center, and pellet positions
    avoid_positions = {
        (0, 0),
        (0, maze_size - 1),
        (maze_size - 1, 0),
        (maze_size - 1, maze_size - 1),
        (maze_size // 2, maze_size // 2),
    }
    avoid_positions.update(set(pellets))  # Also avoid pellet positions

    all_positions = [(r, c) for r in range(1, maze_size - 1) for c in range(1, maze_size - 1)]
    available_positions = [pos for pos in all_positions if pos not in avoid_positions]

    num_walls = min(num_walls, len(available_positions))

    if num_walls > 0:
        wall_indices = np.random.choice(len(available_positions), size=num_walls, replace=False)
        walls = {available_positions[i] for i in wall_indices}
    else:
        walls = set()

    return PacManPOMDP(
        maze_size=(maze_size, maze_size),
        walls=walls,
        initial_pellets=pellets,
        initial_pacman_pos=(0, 0),
        num_ghosts=num_ghosts,
        # Let the environment auto-generate ghost positions
    )
