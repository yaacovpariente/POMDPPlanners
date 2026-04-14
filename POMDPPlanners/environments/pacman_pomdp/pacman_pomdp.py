# pylint: disable=too-many-lines
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

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    ObservationModel,
    SpaceInfo,
    SpaceType,
    StateTransitionModel,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.utils.statistics_utils import confidence_interval


class PacManPOMDPMetrics(Enum):
    """Metric names for PacMan POMDP environment."""

    WIN_RATE = "win_rate"
    AVG_PELLETS_COLLECTED = "avg_pellets_collected"
    AVG_EPISODE_LENGTH = "avg_episode_length"
    AVG_PACMAN_CLOSEST_GHOST_DISTANCE = "avg_pacman_closest_ghost_distance"
    AVG_COLLISION_ENCOUNTERS = "avg_collision_encounters"


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
        super().__init__(state=state, action=action)
        self.pomdp = pomdp

    def sample(self, n_samples: int = 1) -> List[PacManState]:
        """Sample next states."""
        next_state = self._compute_next_state()
        return [next_state] * n_samples

    def probability(self, values: List[PacManState]) -> np.ndarray:
        """Calculate transition probabilities to next states.

        Args:
            values: List of potential next states

        Returns:
            Array of probabilities for each state in values
        """
        if self.state.terminal:
            # Terminal states only transition to themselves with probability 1
            return np.array([1.0 if s == self.state else 0.0 for s in values])

        # Determine PacMan's next position (deterministic)
        pacman_next_pos = self._move_pacman()

        probs = []
        for next_state in values:
            # Check if this is a valid next state
            if next_state.pacman_pos != pacman_next_pos:
                # PacMan position doesn't match - impossible transition
                probs.append(0.0)
                continue

            # Calculate probability of ghost movements
            ghost_prob = self._calculate_ghost_transition_probability(next_state.ghost_positions)

            # Verify pellet and score consistency
            if not self._is_valid_pellet_configuration(next_state):
                probs.append(0.0)
                continue

            # Verify terminal status consistency
            if not self._is_valid_terminal_status(next_state, pacman_next_pos):
                probs.append(0.0)
                continue

            probs.append(ghost_prob)

        # Normalize probabilities
        probs = np.array(probs)
        total = np.sum(probs)
        if total > 0:
            probs = probs / total

        return probs

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
            if strategy == "ambush":
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

    def _move_patrol_ghost(  # pylint: disable=protected-access
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

    def _calculate_ghost_transition_probability(
        self, target_ghost_positions: Tuple[Tuple[int, int], ...]
    ) -> float:
        """Calculate probability of ghosts moving to target positions."""
        if len(target_ghost_positions) != len(self.state.ghost_positions):
            return 0.0

        total_prob = 1.0

        for i, (current_ghost_pos, target_ghost_pos) in enumerate(
            zip(self.state.ghost_positions, target_ghost_positions)
        ):
            ghost_prob = self._single_ghost_move_probability(current_ghost_pos, target_ghost_pos, i)
            total_prob *= ghost_prob

            # If any ghost has zero probability, entire transition is impossible
            if total_prob == 0.0:
                return 0.0

        return total_prob

    def _single_ghost_move_probability(
        self, current_pos: Tuple[int, int], target_pos: Tuple[int, int], ghost_id: int
    ) -> float:
        """Calculate probability of a single ghost moving to target position."""
        possible_moves = self._get_valid_ghost_moves(current_pos)

        if target_pos not in possible_moves:
            return 0.0

        # Get strategy for this ghost
        if self.pomdp.ghost_coordination == "independent":
            return self._independent_ghost_move_probability(
                current_pos, target_pos, possible_moves, ghost_id
            )
        if self.pomdp.ghost_coordination == "coordinated":
            return self._coordinated_ghost_move_probability(
                current_pos, target_pos, possible_moves, ghost_id
            )
        # "mixed"
        if ghost_id % 2 == 0:
            return self._coordinated_ghost_move_probability(
                current_pos, target_pos, possible_moves, ghost_id
            )
        return self._independent_ghost_move_probability(
            current_pos, target_pos, possible_moves, ghost_id
        )

    def _independent_ghost_move_probability(  # pylint: disable=unused-argument
        self,
        current_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        possible_moves: List[Tuple[int, int]],
        ghost_id: int,
    ) -> float:
        """Calculate probability for independent ghost movement."""
        # Check for special strategies
        if hasattr(self.pomdp, "ghost_strategies") and ghost_id < len(self.pomdp.ghost_strategies):
            strategy = self.pomdp.ghost_strategies[ghost_id]
            if strategy in ["patrol", "ambush"]:
                # For patrol and ambush, we use uniform probability as exact policy is complex
                return 1.0 / len(possible_moves)

        # Default aggressive behavior - softmax based on distance to PacMan
        pacman_pos = self.state.pacman_pos
        move_scores = []

        for move_pos in possible_moves:
            distance = abs(move_pos[0] - pacman_pos[0]) + abs(move_pos[1] - pacman_pos[1])
            move_scores.append(-distance)  # Negative distance (closer is better)

        # Softmax with temperature
        temperature = self.pomdp.ghost_aggressiveness
        exp_scores = np.exp(np.array(move_scores) / temperature)
        probabilities = exp_scores / np.sum(exp_scores)

        # Find index of target position
        target_idx = possible_moves.index(target_pos)
        return float(probabilities[target_idx])

    def _coordinated_ghost_move_probability(  # pylint: disable=unused-argument
        self,
        current_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        possible_moves: List[Tuple[int, int]],
        ghost_id: int,
    ) -> float:
        """Calculate probability for coordinated ghost movement."""
        # For coordinated movement, use uniform probability as exact policy is complex
        # The coordination involves predicting PacMan's escape routes which is deterministic
        # but complex to compute probability for
        return 1.0 / len(possible_moves)

    def _is_valid_pellet_configuration(self, next_state: PacManState) -> bool:
        """Check if pellet configuration is consistent with transition."""
        pacman_next_pos = self._move_pacman()

        # Check if pellets were collected correctly
        if pacman_next_pos in self.state.pellets:
            # PacMan moved to a pellet position - pellet should be removed
            expected_pellets = [p for p in self.state.pellets if p != pacman_next_pos]
            expected_score = self.state.score + self.pomdp.pellet_reward

            return (
                set(next_state.pellets) == set(expected_pellets)
                and next_state.score == expected_score
            )
        # No pellet collected - pellets and score should remain the same
        return next_state.pellets == self.state.pellets and next_state.score == self.state.score

    def _is_valid_terminal_status(
        self, next_state: PacManState, pacman_next_pos: Tuple[int, int]
    ) -> bool:
        """Check if terminal status is consistent with the transition."""
        # Terminal if collision or all pellets collected
        collision = pacman_next_pos in next_state.ghost_positions
        all_pellets_collected = len(next_state.pellets) == 0

        expected_terminal = collision or all_pellets_collected

        return next_state.terminal == expected_terminal


class PacManObservationModel(ObservationModel):
    """Observation model for PacMan POMDP."""

    def __init__(self, next_state: PacManState, action: int, pomdp: "PacManPOMDP"):
        """Initialize observation model.

        Args:
            next_state: Next state after transition
            action: Action that was executed
            pomdp: Reference to the POMDP environment
        """
        super().__init__(next_state=next_state, action=action)
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
                for _, (obs_pos, true_ghost_pos) in enumerate(zip(obs_tuple, true_ghost_positions)):
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
                        # 2D Gaussian PDF: (1/(2*pi*sigma^2)) * exp(-d^2/(2*sigma^2))
                        row_diff = obs_pos[0] - true_ghost_pos[0]
                        col_diff = obs_pos[1] - true_ghost_pos[1]
                        distance_sq = row_diff**2 + col_diff**2
                        variance = noise_std**2
                        normalization = 1.0 / (2.0 * np.pi * variance)
                        ghost_prob = normalization * np.exp(-distance_sq / (2 * variance))

                    total_prob *= ghost_prob

                prob = total_prob

            probs.append(prob)

        return np.array(probs)


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

        # Initialize ghost patrol directions for patrol strategy
        self._ghost_patrol_directions: Dict[int, int] = {}

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

    @property
    def initial_ghost_pos(self) -> Tuple[int, int]:
        """Backward compatibility: returns first ghost position."""
        return self.initial_ghost_positions[0] if self.initial_ghost_positions else (0, 0)

    # ------------------------------------------------------------------
    # Array state conversion (for vectorized belief support)
    # ------------------------------------------------------------------

    def state_to_array(self, state: PacManState) -> np.ndarray:
        """Convert a PacManState to a fixed-size numpy array.

        The array layout is:
        ``[pac_row, pac_col, g0_row, g0_col, ..., pellet_mask[0..P-1], score, terminal]``

        Args:
            state: A PacManState instance.

        Returns:
            1-D float array of shape ``(self._state_dim,)``.
        """
        arr = np.zeros(self._state_dim, dtype=np.float64)
        arr[self._idx_pac_row] = state.pacman_pos[0]
        arr[self._idx_pac_col] = state.pacman_pos[1]
        for g, gpos in enumerate(state.ghost_positions):
            arr[self._idx_ghosts_start + 2 * g] = gpos[0]
            arr[self._idx_ghosts_start + 2 * g + 1] = gpos[1]
        pellet_set = set(state.pellets)
        for pos, idx in self._pellet_to_index.items():
            if pos in pellet_set:
                arr[self._idx_pellets_start + idx] = 1.0
        arr[self._idx_score] = state.score
        arr[self._idx_terminal] = 1.0 if state.terminal else 0.0
        return arr

    def array_to_state(self, arr: np.ndarray) -> PacManState:
        """Convert a numpy array back to a PacManState.

        Args:
            arr: 1-D array of shape ``(self._state_dim,)`` produced by
                :meth:`state_to_array`.

        Returns:
            Reconstructed PacManState.
        """
        pacman_pos = (int(arr[self._idx_pac_row]), int(arr[self._idx_pac_col]))
        ghost_positions = tuple(
            (
                int(arr[self._idx_ghosts_start + 2 * g]),
                int(arr[self._idx_ghosts_start + 2 * g + 1]),
            )
            for g in range(self.num_ghosts)
        )
        pellets = tuple(
            pos
            for pos, idx in self._pellet_to_index.items()
            if arr[self._idx_pellets_start + idx] > 0.5
        )
        score = float(arr[self._idx_score])
        terminal = arr[self._idx_terminal] > 0.5
        return PacManState(
            pacman_pos=pacman_pos,
            ghost_positions=ghost_positions,
            pellets=pellets,
            score=score,
            terminal=terminal,
        )

    def states_to_array(self, states: List[PacManState]) -> np.ndarray:
        """Batch-convert a list of PacManState to a 2-D numpy array.

        Args:
            states: List of PacManState instances.

        Returns:
            Array of shape ``(len(states), self._state_dim)``.
        """
        return np.array([self.state_to_array(s) for s in states])

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

    def _ensure_pacman_state(self, state: Any) -> PacManState:
        if isinstance(state, np.ndarray):
            return self.array_to_state(state)
        return state

    def state_transition_model(self, state: Any, action: int) -> PacManStateTransitionModel:
        """Get state transition model."""
        return PacManStateTransitionModel(self._ensure_pacman_state(state), action, self)

    def observation_model(self, next_state: Any, action: int) -> PacManObservationModel:
        """Get observation model."""
        return PacManObservationModel(self._ensure_pacman_state(next_state), action, self)

    def reward(self, state: Any, action: int) -> float:
        """Calculate immediate reward."""
        state = self._ensure_pacman_state(state)
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

    def reward_batch(  # type: ignore[override]
        self, states: Union[np.ndarray, Sequence[Any]], action: int
    ) -> np.ndarray:
        """Calculate rewards for a batch of states.

        Accepts either a 2-D numpy array of shape ``(N, state_dim)``
        (vectorized path) or a sequence of PacManState objects (falls back
        to the loop-based default).

        Computes deterministic reward components only: step penalty, pellet
        collection, and win bonus. Ghost collision penalty is excluded because
        it depends on stochastic ghost movement.

        Args:
            states: Array of shape ``(N, state_dim)`` or sequence of states.
            action: Discrete action index (0-3).

        Returns:
            1-D array of reward values with shape ``(N,)``.
        """
        states_arr = np.asarray(states)
        if states_arr.dtype.kind == "f":
            if states_arr.ndim == 1:
                states_arr = states_arr.reshape(1, -1)
            return self._compute_reward_batch(states_arr, action)
        # Fallback for PacManState sequences (non-vectorized beliefs)
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
            from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_beliefs.pacman_grid_utils import (  # pylint: disable=import-outside-toplevel
                precompute_neighbor_table,
                precompute_valid_cell_mask,
            )

            valid_mask = precompute_valid_cell_mask(self.maze_size, self.walls)
            self._cached_neighbor_table = precompute_neighbor_table(self.maze_size, valid_mask)
        return self._cached_neighbor_table

    def is_terminal(self, state: Any) -> bool:
        """Check if state is terminal."""
        state = self._ensure_pacman_state(state)
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

    def _check_episode_win_status(self, final_state: PacManState) -> int:
        """Check if the episode was won (all pellets collected and terminal)."""
        won = final_state.terminal and len(final_state.pellets) == 0
        return 1 if won else 0

    def _count_pellets_collected(self, final_state: PacManState) -> int:
        """Count the number of pellets collected during the episode."""
        initial_pellets = len(self.initial_pellets)
        remaining_pellets = len(final_state.pellets)
        return initial_pellets - remaining_pellets

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
            if isinstance(step_data.state, PacManState):
                pacman_pos = step_data.state.pacman_pos
                ghost_positions = step_data.state.ghost_positions

                # Track distance to closest ghost
                closest_distance = self._get_closest_ghost_distance(pacman_pos, ghost_positions)
                if closest_distance is not None:
                    episode_distances.append(closest_distance)

                # Count collisions
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

        if not isinstance(final_state, PacManState):
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
            if isinstance(step_data.state, PacManState):
                pacman_pos = step_data.state.pacman_pos
                for ghost_id, ghost_pos in enumerate(step_data.state.ghost_positions):
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

    def visualize_path(self, path: List[PacManState], actions: List[int], cache_path: Path):
        """Visualize PacMan path through the maze using sprite-based rendering.

        Args:
            path: List of states representing the path through the maze
            actions: List of actions taken at each step
            cache_path: Path where the GIF should be saved
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
