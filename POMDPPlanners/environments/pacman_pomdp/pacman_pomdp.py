"""Module for PacMan POMDP environment.

This module provides the PacMan POMDP environment implementation inspired by the
classic arcade game. The environment features a grid world where PacMan must
collect pellets while avoiding ghosts, with partial observability of ghost positions.

The environment involves PacMan navigating a maze with walls, collecting pellets,
and avoiding ghosts that move according to stochastic policies. PacMan receives
noisy observations about nearby ghost positions.

Classes:
    PacManState: Represents the state of the environment
    PacManPOMDP: The main POMDP environment implementation
"""

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    ObservationModel,
    SpaceInfo,
    SpaceType,
    StateTransitionModel,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.utils.statistics import confidence_interval


@dataclass(frozen=True)
class PacManState:
    """State representation for PacMan POMDP.

    Attributes:
        pacman_pos: PacMan position as (row, col) tuple
        ghost_positions: Tuple of ghost positions as (row, col) tuples
        pellets: Tuple of remaining pellet positions as (row, col) tuples
        score: Current game score
        terminal: Whether the game has ended
    """

    pacman_pos: Tuple[int, int]
    ghost_positions: Tuple[Tuple[int, int], ...]  # Multiple ghost positions
    pellets: Tuple[Tuple[int, int], ...]  # Tuple for immutability
    score: Union[int, float] = 0
    terminal: bool = False

    @property
    def ghost_pos(self) -> Tuple[int, int]:
        """Backward compatibility: returns first ghost position."""
        return self.ghost_positions[0] if self.ghost_positions else (0, 0)

    @property
    def num_ghosts(self) -> int:
        """Number of ghosts in the game."""
        return len(self.ghost_positions)

    def __post_init__(self):
        """Validate state components."""
        if not isinstance(self.pacman_pos, tuple) or len(self.pacman_pos) != 2:
            raise ValueError("pacman_pos must be a tuple of two integers")
        if not isinstance(self.ghost_positions, tuple):
            raise ValueError("ghost_positions must be a tuple of position tuples")
        for i, ghost_pos in enumerate(self.ghost_positions):
            if not isinstance(ghost_pos, tuple) or len(ghost_pos) != 2:
                raise ValueError(f"ghost_positions[{i}] must be a tuple of two integers")
        if not isinstance(self.pellets, tuple):
            raise ValueError("pellets must be a tuple of position tuples")


class PacManStateTransitionModel(StateTransitionModel):
    """State transition model for PacMan POMDP."""

    def __init__(self, state: PacManState, action: int, pomdp: "PacManPOMDP"):
        """Initialize transition model.

        Args:
            state: Current state
            action: Action to execute
            pomdp: Reference to the POMDP environment
        """
        self.state = state
        self.action = action
        self.pomdp = pomdp

    def sample(self, n_samples: int = 1) -> List[PacManState]:
        """Sample next states."""
        next_state = self._compute_next_state()
        return [next_state] * n_samples

    def _compute_next_state(self) -> PacManState:
        """Compute the next state."""
        if self.state.terminal:
            return self.state

        # Move PacMan based on action
        pacman_pos = self._move_pacman()

        # Move all ghosts stochastically
        ghost_positions = self._move_ghosts()

        # Check for collision with any ghost
        if pacman_pos in ghost_positions:
            return PacManState(
                pacman_pos=pacman_pos,
                ghost_positions=ghost_positions,
                pellets=self.state.pellets,
                score=self.state.score,
                terminal=True,
            )

        # Check for pellet collection
        pellets = list(self.state.pellets)
        score = self.state.score

        if pacman_pos in pellets:
            pellets.remove(pacman_pos)
            score = score + self.pomdp.pellet_reward

        # Check for winning condition (all pellets collected)
        terminal = len(pellets) == 0

        return PacManState(
            pacman_pos=pacman_pos,
            ghost_positions=ghost_positions,
            pellets=tuple(pellets),
            score=score,
            terminal=terminal,
        )

    def _move_pacman(self) -> Tuple[int, int]:
        """Move PacMan based on action."""
        row, col = self.state.pacman_pos

        # Actions: 0=North, 1=East, 2=South, 3=West
        if self.action == 0:  # North
            new_pos = (row - 1, col)
        elif self.action == 1:  # East
            new_pos = (row, col + 1)
        elif self.action == 2:  # South
            new_pos = (row + 1, col)
        elif self.action == 3:  # West
            new_pos = (row, col - 1)
        else:
            new_pos = (row, col)  # Invalid action, stay in place

        # Check if new position is valid (not a wall and within bounds)
        if self._is_valid_position(new_pos):
            return new_pos
        else:
            return self.state.pacman_pos  # Stay in current position

    def _move_ghosts(self) -> Tuple[Tuple[int, int], ...]:
        """Move all ghosts with their respective policies."""
        new_positions = []

        for i, ghost_pos in enumerate(self.state.ghost_positions):
            if self.pomdp.ghost_coordination == "independent":
                # Each ghost acts independently
                new_pos = self._move_single_ghost(ghost_pos, i)
            elif self.pomdp.ghost_coordination == "coordinated":
                # Ghosts coordinate to surround PacMan
                new_pos = self._move_coordinated_ghost(ghost_pos, i, self.state.ghost_positions)
            else:  # "mixed"
                # Alternate between coordinated and independent behavior
                if i % 2 == 0:
                    new_pos = self._move_coordinated_ghost(ghost_pos, i, self.state.ghost_positions)
                else:
                    new_pos = self._move_single_ghost(ghost_pos, i)

            new_positions.append(new_pos)

        return tuple(new_positions)

    def _move_single_ghost(self, ghost_pos: Tuple[int, int], ghost_id: int) -> Tuple[int, int]:
        """Move individual ghost with stochastic policy."""
        possible_moves = self._get_valid_ghost_moves(ghost_pos)

        if not possible_moves:
            return ghost_pos  # Can't move, stay in place

        pacman_pos = self.state.pacman_pos

        # Apply ghost-specific strategy if defined
        if hasattr(self.pomdp, "ghost_strategies") and ghost_id < len(self.pomdp.ghost_strategies):
            strategy = self.pomdp.ghost_strategies[ghost_id]
            if strategy == "patrol":
                return self._move_patrol_ghost(ghost_pos, possible_moves, ghost_id)
            elif strategy == "ambush":
                return self._move_ambush_ghost(ghost_pos, possible_moves, pacman_pos)

        # Default aggressive behavior - move toward PacMan
        move_scores = []
        for move_pos in possible_moves:
            distance = abs(move_pos[0] - pacman_pos[0]) + abs(move_pos[1] - pacman_pos[1])
            move_scores.append(-distance)  # Negative distance (closer is better)

        # Convert to probabilities (softmax with temperature)
        temperature = self.pomdp.ghost_aggressiveness
        exp_scores = np.exp(np.array(move_scores) / temperature)
        probabilities = exp_scores / np.sum(exp_scores)

        # Sample next ghost position
        chosen_idx = np.random.choice(len(possible_moves), p=probabilities)
        return possible_moves[chosen_idx]

    def _move_coordinated_ghost(
        self,
        ghost_pos: Tuple[int, int],
        ghost_id: int,
        all_ghost_positions: Tuple[Tuple[int, int], ...],
    ) -> Tuple[int, int]:
        """Move ghost with coordination strategy."""
        possible_moves = self._get_valid_ghost_moves(ghost_pos)
        if not possible_moves:
            return ghost_pos

        pacman_pos = self.state.pacman_pos

        # Lead ghost (id=0) chases directly, others try to cut off escape routes
        if ghost_id == 0:
            # Direct pursuit
            target = pacman_pos
        else:
            # Try to block PacMan's escape routes
            target = self._predict_pacman_escape_route(pacman_pos, all_ghost_positions, ghost_id)

        return self._move_toward_target(ghost_pos, target, possible_moves)

    def _move_patrol_ghost(
        self,
        ghost_pos: Tuple[int, int],
        possible_moves: List[Tuple[int, int]],
        ghost_id: int,
    ) -> Tuple[int, int]:
        """Move ghost in patrol pattern."""
        # Simple patrol: move in predictable rectangular pattern
        if not hasattr(self.pomdp, "_ghost_patrol_directions"):
            self.pomdp._ghost_patrol_directions = {}

        if ghost_id not in self.pomdp._ghost_patrol_directions:
            self.pomdp._ghost_patrol_directions[ghost_id] = 0  # Start moving north

        # Try to continue in current direction
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        current_dir = self.pomdp._ghost_patrol_directions[ghost_id]
        dr, dc = directions[current_dir]

        preferred_pos = (ghost_pos[0] + dr, ghost_pos[1] + dc)

        if preferred_pos in possible_moves:
            return preferred_pos
        else:
            # Change direction if blocked
            self.pomdp._ghost_patrol_directions[ghost_id] = (current_dir + 1) % 4
            # Try new direction or just pick randomly
            return np.random.choice(possible_moves) if possible_moves else ghost_pos

    def _move_ambush_ghost(
        self,
        ghost_pos: Tuple[int, int],
        possible_moves: List[Tuple[int, int]],
        pacman_pos: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Move ghost to ambush position ahead of PacMan."""
        # Try to move to a position that intercepts PacMan's likely path
        # Predict where PacMan might go and position accordingly

        # Simple heuristic: move to position that's 2-3 steps ahead of PacMan in the direction they're likely to go
        best_pos = ghost_pos
        best_score = float("inf")

        for move_pos in possible_moves:
            # Score based on being ahead of PacMan rather than directly chasing
            distance_to_pacman = abs(move_pos[0] - pacman_pos[0]) + abs(move_pos[1] - pacman_pos[1])
            # Prefer positions that are 2-4 tiles away (good ambush distance)
            if 2 <= distance_to_pacman <= 4:
                score = distance_to_pacman
            else:
                score = distance_to_pacman + 10  # Penalty for being too close or too far

            if score < best_score:
                best_score = score
                best_pos = move_pos

        return best_pos

    def _predict_pacman_escape_route(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: Tuple[Tuple[int, int], ...],
        current_ghost_id: int,
    ) -> Tuple[int, int]:
        """Predict where PacMan is likely to escape and target that area."""
        # Find the direction that maximizes distance from all ghosts
        possible_pacman_moves = self._get_valid_moves_for_position(pacman_pos)

        best_escape_pos = pacman_pos
        max_min_distance = -1

        for pacman_move in possible_pacman_moves:
            # Calculate minimum distance to any ghost from this position
            min_distance_to_ghosts = min(
                abs(pacman_move[0] - ghost_pos[0]) + abs(pacman_move[1] - ghost_pos[1])
                for i, ghost_pos in enumerate(ghost_positions)
                if i != current_ghost_id
            )

            if min_distance_to_ghosts > max_min_distance:
                max_min_distance = min_distance_to_ghosts
                best_escape_pos = pacman_move

        return best_escape_pos

    def _move_toward_target(
        self,
        current_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        possible_moves: List[Tuple[int, int]],
    ) -> Tuple[int, int]:
        """Move toward target position from possible moves."""
        if not possible_moves:
            return current_pos

        best_move = current_pos
        min_distance = float("inf")

        for move_pos in possible_moves:
            distance = abs(move_pos[0] - target_pos[0]) + abs(move_pos[1] - target_pos[1])
            if distance < min_distance:
                min_distance = distance
                best_move = move_pos

        return best_move

    def _get_valid_ghost_moves(self, ghost_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid positions for ghost to move to."""
        row, col = ghost_pos
        possible_moves = [
            (row - 1, col),  # North
            (row, col + 1),  # East
            (row + 1, col),  # South
            (row, col - 1),  # West
            (row, col),  # Stay in place
        ]

        return [pos for pos in possible_moves if self._is_valid_position(pos)]

    def _get_valid_moves_for_position(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid moves for any position (used for prediction)."""
        row, col = pos
        possible_moves = [
            (row - 1, col),  # North
            (row, col + 1),  # East
            (row + 1, col),  # South
            (row, col - 1),  # West
        ]

        return [move_pos for move_pos in possible_moves if self._is_valid_position(move_pos)]

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds and not a wall)."""
        row, col = pos
        return (
            0 <= row < self.pomdp.maze_size[0]
            and 0 <= col < self.pomdp.maze_size[1]
            and pos not in self.pomdp.walls
        )


class PacManObservationModel(ObservationModel):
    """Observation model for PacMan POMDP."""

    def __init__(self, next_state: PacManState, action: int, pomdp: "PacManPOMDP"):
        """Initialize observation model.

        Args:
            next_state: Next state after transition
            action: Action that was executed
            pomdp: Reference to the POMDP environment
        """
        self.next_state = next_state
        self.action = action
        self.pomdp = pomdp

    def sample(self, n_samples: int = 1) -> List[Tuple[Tuple[int, int], ...]]:
        """Sample observations of all ghost positions with noise."""
        if self.next_state.terminal:
            # Terminal observation: return (-1, -1) for each ghost
            terminal_obs = tuple([(-1, -1)] * len(self.next_state.ghost_positions))
            return [terminal_obs] * n_samples

        pacman_pos = self.next_state.pacman_pos

        observations = []
        for _ in range(n_samples):
            ghost_obs = []
            for ghost_pos in self.next_state.ghost_positions:
                # Add noise based on distance from PacMan to each ghost
                distance = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
                noise_std = min(
                    distance * self.pomdp.observation_noise_factor,
                    self.pomdp.max_observation_noise,
                )

                # Add Gaussian noise to ghost position
                noise_row = np.random.normal(0, noise_std)
                noise_col = np.random.normal(0, noise_std)

                observed_row = int(np.round(ghost_pos[0] + noise_row))
                observed_col = int(np.round(ghost_pos[1] + noise_col))

                # Clamp to valid bounds
                observed_row = max(0, min(self.pomdp.maze_size[0] - 1, observed_row))
                observed_col = max(0, min(self.pomdp.maze_size[1] - 1, observed_col))

                ghost_obs.append((observed_row, observed_col))

            observations.append(tuple(ghost_obs))

        return observations

    def sample_closest_ghosts(
        self, max_ghosts: int = 2, n_samples: int = 1
    ) -> List[Tuple[Tuple[int, int], ...]]:
        """Sample observations of only the closest ghosts."""
        if self.next_state.terminal:
            terminal_obs = tuple([(-1, -1)] * min(max_ghosts, len(self.next_state.ghost_positions)))
            return [terminal_obs] * n_samples

        pacman_pos = self.next_state.pacman_pos

        # Calculate distances and sort ghosts by proximity
        ghost_distances = [
            (
                ghost_pos,
                abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1]),
            )
            for ghost_pos in self.next_state.ghost_positions
        ]
        closest_ghosts = sorted(ghost_distances, key=lambda x: x[1])[:max_ghosts]

        observations = []
        for _ in range(n_samples):
            ghost_obs = []
            for ghost_pos, distance in closest_ghosts:
                noise_std = min(
                    distance * self.pomdp.observation_noise_factor,
                    self.pomdp.max_observation_noise,
                )

                noise_row = np.random.normal(0, noise_std)
                noise_col = np.random.normal(0, noise_std)

                observed_row = int(np.round(ghost_pos[0] + noise_row))
                observed_col = int(np.round(ghost_pos[1] + noise_col))

                observed_row = max(0, min(self.pomdp.maze_size[0] - 1, observed_row))
                observed_col = max(0, min(self.pomdp.maze_size[1] - 1, observed_col))

                ghost_obs.append((observed_row, observed_col))

            observations.append(tuple(ghost_obs))

        return observations

    def probability(self, values: List[Tuple[Tuple[int, int], ...]]) -> np.ndarray:
        """Calculate observation probabilities for multi-ghost observations."""
        if self.next_state.terminal:
            terminal_obs = tuple([(-1, -1)] * len(self.next_state.ghost_positions))
            return np.array([1.0 if obs == terminal_obs else 0.0 for obs in values])

        pacman_pos = self.next_state.pacman_pos
        true_ghost_positions = self.next_state.ghost_positions

        probs = []
        for obs_tuple in values:
            if len(obs_tuple) != len(true_ghost_positions):
                # Incorrect number of ghost observations
                prob = 0.0
            elif all(obs == (-1, -1) for obs in obs_tuple):
                # All terminal observations in non-terminal state
                prob = 0.0
            else:
                # Calculate probability as product of individual ghost observation probabilities
                total_prob = 1.0
                for i, (obs_pos, true_ghost_pos) in enumerate(zip(obs_tuple, true_ghost_positions)):
                    # Distance-based noise for this ghost
                    distance = abs(true_ghost_pos[0] - pacman_pos[0]) + abs(
                        true_ghost_pos[1] - pacman_pos[1]
                    )
                    noise_std = min(
                        distance * self.pomdp.observation_noise_factor,
                        self.pomdp.max_observation_noise,
                    )

                    if noise_std == 0:
                        noise_std = 1e-6  # Avoid division by zero

                    if obs_pos == (-1, -1):
                        # Terminal observation for individual ghost in non-terminal state
                        ghost_prob = 0.0
                    else:
                        # Gaussian probability for this ghost
                        row_diff = obs_pos[0] - true_ghost_pos[0]
                        col_diff = obs_pos[1] - true_ghost_pos[1]
                        distance_sq = row_diff**2 + col_diff**2
                        ghost_prob = np.exp(-distance_sq / (2 * noise_std**2))

                    total_prob *= ghost_prob

                prob = total_prob

            probs.append(prob)

        probs = np.array(probs)
        total: float = float(np.sum(probs))
        if total > 0:
            probs = probs / total

        return probs


class PacManPOMDP(DiscreteActionsEnvironment):
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
        >>> # Create 7x7 maze with walls and pellets
        >>> walls = {(1, 1), (1, 2), (3, 3)}
        >>> pellets = [(0, 2), (2, 0), (4, 4)]
        >>> pomdp = PacManPOMDP(
        ...     maze_size=(7, 7),
        ...     walls=walls,
        ...     initial_pellets=pellets
        ... )
        >>>
        >>> # Sample initial state
        >>> initial_state = pomdp.initial_state_dist().sample()[0]
        >>> isinstance(initial_state, PacManState)
        True
        >>>
        >>> # Execute action
        >>> next_state, obs, reward = pomdp.sample_next_step(initial_state, 1)  # doctest: +SKIP
    """

    def __init__(
        self,
        maze_size: Tuple[int, int] = (7, 7),
        walls: Set[Tuple[int, int]] = {(2, 2), (2, 3), (3, 2), (4, 4), (3, 5)},
        initial_pellets: List[Tuple[int, int]] = [(1, 1), (1, 5), (5, 1), (5, 5)],
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

        self.maze_size = maze_size
        self.walls = walls if walls is not None else set()
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

        # Initialize ghost patrol directions for patrol strategy
        self._ghost_patrol_directions: Dict[int, int] = {}

    @property
    def initial_ghost_pos(self) -> Tuple[int, int]:
        """Backward compatibility: returns first ghost position."""
        return self.initial_ghost_positions[0] if self.initial_ghost_positions else (0, 0)

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

    def _validate_parameters(self):
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

    def state_transition_model(self, state: PacManState, action: int) -> PacManStateTransitionModel:
        """Get state transition model."""
        return PacManStateTransitionModel(state, action, self)

    def observation_model(self, next_state: PacManState, action: int) -> PacManObservationModel:
        """Get observation model."""
        return PacManObservationModel(next_state, action, self)

    def reward(self, state: PacManState, action: int) -> float:
        """Calculate immediate reward."""
        if state.terminal:
            return 0.0  # No reward for terminal states

        # Base step penalty
        total_reward = self.step_penalty

        # Simulate next state to check for events
        next_state = self.state_transition_model(state, action).sample()[0]

        # Ghost collision penalty
        if next_state.pacman_pos in next_state.ghost_positions:
            total_reward += self.ghost_collision_penalty

        # Pellet collection reward
        if next_state.score > state.score:
            total_reward += self.pellet_reward

        # Win condition bonus
        if next_state.terminal and len(next_state.pellets) == 0:
            total_reward += self.win_reward

        return total_reward

    def is_terminal(self, state: PacManState) -> bool:
        """Check if state is terminal."""
        return state.terminal

    def initial_state_dist(self) -> DiscreteDistribution:
        """Get initial state distribution."""
        # Single deterministic initial state
        initial_state = PacManState(
            pacman_pos=self.initial_pacman_pos,
            ghost_positions=tuple(self.initial_ghost_positions),
            pellets=tuple(self.initial_pellets),
            score=0,
            terminal=False,
        )

        return DiscreteDistribution(values=[initial_state], probs=np.array([1.0]))

    def initial_observation_dist(self) -> DiscreteDistribution:
        """Get initial observation distribution."""
        # Initial observation is the true ghost position with some noise
        initial_obs = self.observation_model(
            next_state=self.initial_state_dist().sample()[0], action=0  # Dummy action
        ).sample()[0]

        return DiscreteDistribution(values=[initial_obs], probs=np.array([1.0]))

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        """Check if two observations are equal."""
        return observation1 == observation2

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        """Compute environment-specific metrics."""
        if not histories:
            return []

        metrics = []

        # Calculate win rate (all pellets collected)
        wins = []
        pellets_collected = []
        episode_lengths = []
        pacman_ghost_distances = []
        collision_encounters = []

        for history in histories:
            episode_lengths.append(len(history.history))
            final_state = history.history[-1].state if history.history else None

            if isinstance(final_state, PacManState):
                # Check if won (no pellets remaining)
                won = final_state.terminal and len(final_state.pellets) == 0
                wins.append(1 if won else 0)

                # Count pellets collected
                initial_pellets = len(self.initial_pellets)
                remaining_pellets = len(final_state.pellets)
                collected = initial_pellets - remaining_pellets
                pellets_collected.append(collected)

                # Calculate average distance between PacMan and closest ghost throughout episode
                # and count collision encounters
                episode_distances = []
                episode_collisions = 0
                per_ghost_distances: List[List[float]] = [
                    [] for _ in range(len(self.initial_ghost_positions))
                ]

                for step_data in history.history:
                    if isinstance(step_data.state, PacManState):
                        pacman_pos = step_data.state.pacman_pos

                        # Calculate distances to all ghosts
                        ghost_distances = []
                        for i, ghost_pos in enumerate(step_data.state.ghost_positions):
                            distance = abs(pacman_pos[0] - ghost_pos[0]) + abs(
                                pacman_pos[1] - ghost_pos[1]
                            )
                            ghost_distances.append(distance)

                            # Track per-ghost distances
                            if i < len(per_ghost_distances):
                                per_ghost_distances[i].append(distance)

                        # Use minimum distance (closest ghost)
                        min_distance = min(ghost_distances) if ghost_distances else 0
                        episode_distances.append(min_distance)

                        # Count collision encounters (when PacMan and any ghost are at same position)
                        if pacman_pos in step_data.state.ghost_positions:
                            episode_collisions += 1

                if episode_distances:
                    avg_distance = float(np.mean(episode_distances))
                    pacman_ghost_distances.append(avg_distance)

                collision_encounters.append(episode_collisions)
            else:
                wins.append(0)
                pellets_collected.append(0)
                collision_encounters.append(0)

        if wins:
            win_rate = float(np.mean(wins))
            ci_low, ci_high = confidence_interval(wins)
            metrics.append(
                MetricValue(
                    name="win_rate",
                    value=win_rate,
                    lower_confidence_bound=ci_low,
                    upper_confidence_bound=ci_high,
                )
            )

        if pellets_collected:
            avg_pellets = float(np.mean(pellets_collected))
            ci_low, ci_high = confidence_interval(pellets_collected)
            metrics.append(
                MetricValue(
                    name="avg_pellets_collected",
                    value=avg_pellets,
                    lower_confidence_bound=ci_low,
                    upper_confidence_bound=ci_high,
                )
            )

        if episode_lengths:
            avg_length = float(np.mean(episode_lengths))
            ci_low, ci_high = confidence_interval(episode_lengths)
            metrics.append(
                MetricValue(
                    name="avg_episode_length",
                    value=avg_length,
                    lower_confidence_bound=ci_low,
                    upper_confidence_bound=ci_high,
                )
            )

        # Average distance between PacMan and closest ghost metric
        if pacman_ghost_distances:
            avg_distance = float(np.mean(pacman_ghost_distances))
            ci_low, ci_high = confidence_interval(pacman_ghost_distances)
            metrics.append(
                MetricValue(
                    name="avg_pacman_closest_ghost_distance",
                    value=avg_distance,
                    lower_confidence_bound=ci_low,
                    upper_confidence_bound=ci_high,
                )
            )

        # Collision encounters metric
        if collision_encounters:
            avg_collisions = float(np.mean(collision_encounters))
            ci_low, ci_high = confidence_interval(collision_encounters)
            metrics.append(
                MetricValue(
                    name="avg_collision_encounters",
                    value=avg_collisions,
                    lower_confidence_bound=ci_low,
                    upper_confidence_bound=ci_high,
                )
            )

        # Multi-ghost specific metrics
        if self.num_ghosts > 1:
            # Add per-ghost distance metrics
            per_ghost_avg_distances: List[List[float]] = []
            for history in histories:
                if history.history:
                    ghost_distances_per_episode: List[List[float]] = [
                        [] for _ in range(self.num_ghosts)
                    ]

                    for step_data in history.history:
                        if isinstance(step_data.state, PacManState):
                            pacman_pos = step_data.state.pacman_pos
                            for i, ghost_pos in enumerate(step_data.state.ghost_positions):
                                if i < len(ghost_distances_per_episode):
                                    ghost_distance: float = float(
                                        abs(pacman_pos[0] - ghost_pos[0])
                                        + abs(pacman_pos[1] - ghost_pos[1])
                                    )
                                    ghost_distances_per_episode[i].append(ghost_distance)

                    # Calculate average distance per ghost for this episode
                    episode_ghost_avgs: List[float] = []
                    for ghost_dist_list in ghost_distances_per_episode:
                        if ghost_dist_list:
                            episode_ghost_avgs.append(float(np.mean(ghost_dist_list)))
                        else:
                            episode_ghost_avgs.append(0.0)

                    per_ghost_avg_distances.append(episode_ghost_avgs)

            # Create metrics for each ghost
            if per_ghost_avg_distances:
                for ghost_id in range(self.num_ghosts):
                    ghost_distance_values: List[float] = [
                        episode_avgs[ghost_id]
                        for episode_avgs in per_ghost_avg_distances
                        if ghost_id < len(episode_avgs)
                    ]
                    if ghost_distance_values:
                        avg_distance = float(np.mean(ghost_distance_values))
                        ci_low, ci_high = confidence_interval(ghost_distance_values)
                        metrics.append(
                            MetricValue(
                                name=f"avg_pacman_ghost_{ghost_id}_distance",
                                value=avg_distance,
                                lower_confidence_bound=ci_low,
                                upper_confidence_bound=ci_high,
                            )
                        )

        return metrics

    def visualize_path(self, path: List[PacManState], actions: List[int], cache_path: Path):
        """Visualize PacMan path through the maze using sprite-based rendering."""
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")

        # Constants for sprite rendering
        TILE_SIZE = 32

        # Get sprite directory (relative to this module)
        module_dir = Path(__file__).parent
        sprite_dir = module_dir / "img"

        def _colorize_sprite(image, color):
            """Apply color tint to sprite image."""
            # Create a colored overlay
            overlay = Image.new("RGBA", image.size, color)

            # Blend the overlay with the original image
            # This creates a tinted version while preserving the original alpha
            result = Image.blend(image.convert("RGBA"), overlay, 0.3)

            # Preserve original alpha channel
            result.putalpha(image.split()[-1])
            return result

        def load_sprites():
            """Load sprite images with fallback to simple colored shapes."""
            sprites = {}

            # Try to load PacMan sprite - prioritize the new pacman_head.jpg
            pacman_head_path = sprite_dir / "pacman_head.jpg"
            pacman_png_path = sprite_dir / "pocman.png"

            if pacman_head_path.exists():
                sprites["pacman"] = (
                    Image.open(pacman_head_path).convert("RGBA").resize((TILE_SIZE, TILE_SIZE))
                )
            elif pacman_png_path.exists():
                sprites["pacman"] = (
                    Image.open(pacman_png_path).convert("RGBA").resize((TILE_SIZE, TILE_SIZE))
                )
            else:
                # Create simple yellow circle for PacMan
                img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.ellipse([4, 4, TILE_SIZE - 4, TILE_SIZE - 4], fill=(255, 255, 0, 255))
                sprites["pacman"] = img

            # Load multiple ghost sprites with different colors
            ghost_colors = [
                (255, 0, 0, 255),  # Red ghost
                (0, 255, 0, 255),  # Green ghost
                (0, 0, 255, 255),  # Blue ghost
                (255, 0, 255, 255),  # Magenta ghost
                (255, 165, 0, 255),  # Orange ghost
                (0, 255, 255, 255),  # Cyan ghost
                (255, 255, 0, 255),  # Yellow ghost
                (128, 0, 128, 255),  # Purple ghost
            ]

            ghost_path = sprite_dir / "ghosts.png"
            if ghost_path.exists():
                base_ghost = Image.open(ghost_path).convert("RGBA")
                for i, color in enumerate(ghost_colors):
                    # Create colored version of ghost sprite
                    colored_ghost = _colorize_sprite(base_ghost, color)
                    sprites[f"ghost_{i}"] = colored_ghost.resize((TILE_SIZE, TILE_SIZE))
            else:
                # Create colored rectangles for each ghost
                for i, color in enumerate(ghost_colors):
                    img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([4, 4, TILE_SIZE - 4, TILE_SIZE - 4], fill=color)
                    sprites[f"ghost_{i}"] = img

            return sprites

        def render_frame(state: PacManState, step_num: int, action_name: str = ""):
            """Render a single frame as PIL Image."""
            rows, cols = self.maze_size
            canvas = Image.new(
                "RGBA", (cols * TILE_SIZE, rows * TILE_SIZE + 60)
            )  # Extra space for text
            draw = ImageDraw.Draw(canvas)

            # Draw background and maze
            for r in range(rows):
                for c in range(cols):
                    x, y = c * TILE_SIZE, r * TILE_SIZE

                    # Check if this position is a wall
                    if (r, c) in self.walls:
                        # Wall = dark blue block
                        draw.rectangle([x, y, x + TILE_SIZE, y + TILE_SIZE], fill=(20, 20, 80, 255))
                    else:
                        # Floor = black
                        draw.rectangle([x, y, x + TILE_SIZE, y + TILE_SIZE], fill=(0, 0, 0, 255))

                    # Draw pellets
                    if (r, c) in state.pellets:
                        # Small white dot for pellet
                        cx, cy = x + TILE_SIZE // 2, y + TILE_SIZE // 2
                        rdot = 4
                        draw.ellipse(
                            [cx - rdot, cy - rdot, cx + rdot, cy + rdot],
                            fill=(255, 255, 255, 255),
                        )

            # Draw sprites
            sprites = load_sprites()

            # Draw all ghosts with different colors
            for i, ghost_pos in enumerate(state.ghost_positions):
                gr, gc = ghost_pos
                if 0 <= gr < rows and 0 <= gc < cols:
                    ghost_x, ghost_y = gc * TILE_SIZE, gr * TILE_SIZE
                    ghost_sprite_key = f"ghost_{i % 8}"  # Cycle through 8 colors
                    if ghost_sprite_key in sprites:
                        canvas.paste(
                            sprites[ghost_sprite_key],
                            (ghost_x, ghost_y),
                            sprites[ghost_sprite_key],
                        )

            # Draw PacMan (last so it can overlap with ghost if collision)
            pr, pc = state.pacman_pos
            if 0 <= pr < rows and 0 <= pc < cols:
                pacman_x, pacman_y = pc * TILE_SIZE, pr * TILE_SIZE
                if state.pacman_pos in state.ghost_positions:
                    # Collision with any ghost - draw enhanced explosion effect
                    draw.ellipse(
                        [
                            pacman_x,
                            pacman_y,
                            pacman_x + TILE_SIZE,
                            pacman_y + TILE_SIZE,
                        ],
                        fill=(255, 0, 0, 200),
                    )
                    # Add multiple explosion symbols for multi-ghost collision effect
                    num_colliding_ghosts = sum(
                        1 for ghost_pos in state.ghost_positions if ghost_pos == state.pacman_pos
                    )
                    explosion_text = "💥" * min(num_colliding_ghosts, 3)  # Limit to 3 symbols
                    draw.text(
                        (pacman_x + 2, pacman_y + 2),
                        explosion_text,
                        fill=(255, 255, 255),
                    )
                else:
                    canvas.paste(sprites["pacman"], (pacman_x, pacman_y), sprites["pacman"])

            # Add text overlay at bottom
            text_y = rows * TILE_SIZE + 5
            draw.text((5, text_y), f"Step {step_num}: {action_name}", fill=(255, 255, 255))
            draw.text(
                (5, text_y + 15),
                f"Score: {state.score}, Pellets: {len(state.pellets)}",
                fill=(255, 255, 255),
            )

            # Add terminal state message
            if state.terminal:
                if len(state.pellets) == 0:
                    draw.text((5, text_y + 30), "🎉 YOU WIN! 🎉", fill=(0, 255, 0))
                else:
                    draw.text((5, text_y + 30), "👻 GAME OVER! 👻", fill=(255, 0, 0))

            return canvas

        # Generate all frames
        frames = []
        for i, state in enumerate(path):
            if i < len(actions):
                action_name = self.action_names[actions[i]]
            else:
                action_name = "Terminal"

            frame = render_frame(state, i + 1, action_name)
            frames.append(frame)

        # Save as animated GIF
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if frames:
            # Save animated GIF
            frames[0].save(
                cache_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000,  # 1 second per frame
                loop=0,
            )
            print(f"Sprite-based visualization saved as GIF: {cache_path}")
        else:
            print("No frames generated for visualization")

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Cache visualization of episode history."""
        if not isinstance(history, List):
            raise TypeError("history must be a List object")
        if not history:
            raise ValueError("Cannot visualize empty history")
        for step in history:
            if not isinstance(step, StepData):
                raise TypeError("history must be a List of StepData objects")
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")

        # Extract path and actions
        path = [step.state for step in history]
        actions = [step.action for step in history[:-1]]  # Last step has no action

        self.visualize_path(path, actions, cache_path)


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

    if len(available_positions) < num_walls:
        num_walls = len(available_positions)

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
