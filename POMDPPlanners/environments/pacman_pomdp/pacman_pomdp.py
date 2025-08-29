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

from typing import List, Any, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os
from PIL import Image, ImageDraw

from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.simulation import History, StepData, MetricValue
from POMDPPlanners.utils.statistics import confidence_interval


@dataclass(frozen=True)
class PacManState:
    """State representation for PacMan POMDP.

    Attributes:
        pacman_pos: PacMan position as (row, col) tuple
        ghost_pos: Ghost position as (row, col) tuple
        pellets: Tuple of remaining pellet positions as (row, col) tuples
        score: Current game score
        terminal: Whether the game has ended
    """

    pacman_pos: Tuple[int, int]
    ghost_pos: Tuple[int, int]
    pellets: Tuple[Tuple[int, int], ...]  # Tuple for immutability
    score: int = 0
    terminal: bool = False

    def __post_init__(self):
        """Validate state components."""
        if not isinstance(self.pacman_pos, tuple) or len(self.pacman_pos) != 2:
            raise ValueError("pacman_pos must be a tuple of two integers")
        if not isinstance(self.ghost_pos, tuple) or len(self.ghost_pos) != 2:
            raise ValueError("ghost_pos must be a tuple of two integers")
        if not isinstance(self.pellets, tuple):
            raise ValueError("pellets must be a tuple of position tuples")


class PacManStateTransitionModel(Distribution):
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

        # Move ghost stochastically
        ghost_pos = self._move_ghost()

        # Check for collision
        if pacman_pos == ghost_pos:
            return PacManState(
                pacman_pos=pacman_pos,
                ghost_pos=ghost_pos,
                pellets=self.state.pellets,
                score=self.state.score,
                terminal=True,
            )

        # Check for pellet collection
        pellets = list(self.state.pellets)
        score = self.state.score

        if pacman_pos in pellets:
            pellets.remove(pacman_pos)
            score += self.pomdp.pellet_reward

        # Check for winning condition (all pellets collected)
        terminal = len(pellets) == 0

        return PacManState(
            pacman_pos=pacman_pos,
            ghost_pos=ghost_pos,
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

    def _move_ghost(self) -> Tuple[int, int]:
        """Move ghost with stochastic policy."""
        # Ghost moves toward PacMan with probability, otherwise randomly
        possible_moves = self._get_valid_ghost_moves()

        if not possible_moves:
            return self.state.ghost_pos  # Can't move, stay in place

        pacman_pos = self.state.pacman_pos
        ghost_pos = self.state.ghost_pos

        # Calculate distances to PacMan for each possible move
        move_scores = []
        for move_pos in possible_moves:
            distance = abs(move_pos[0] - pacman_pos[0]) + abs(
                move_pos[1] - pacman_pos[1]
            )
            move_scores.append(-distance)  # Negative distance (closer is better)

        # Convert to probabilities (softmax with temperature)
        temperature = self.pomdp.ghost_aggressiveness
        exp_scores = np.exp(np.array(move_scores) / temperature)
        probabilities = exp_scores / np.sum(exp_scores)

        # Sample next ghost position
        chosen_idx = np.random.choice(len(possible_moves), p=probabilities)
        return possible_moves[chosen_idx]

    def _get_valid_ghost_moves(self) -> List[Tuple[int, int]]:
        """Get valid positions for ghost to move to."""
        row, col = self.state.ghost_pos
        possible_moves = [
            (row - 1, col),  # North
            (row, col + 1),  # East
            (row + 1, col),  # South
            (row, col - 1),  # West
            (row, col),  # Stay in place
        ]

        return [pos for pos in possible_moves if self._is_valid_position(pos)]

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds and not a wall)."""
        row, col = pos
        return (
            0 <= row < self.pomdp.maze_size[0]
            and 0 <= col < self.pomdp.maze_size[1]
            and pos not in self.pomdp.walls
        )


class PacManObservationModel(Distribution):
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

    def sample(self, n_samples: int = 1) -> List[Tuple[int, int]]:
        """Sample observations of ghost position with noise."""
        if self.next_state.terminal:
            return [(-1, -1)] * n_samples  # Terminal observation

        true_ghost_pos = self.next_state.ghost_pos
        pacman_pos = self.next_state.pacman_pos

        observations = []
        for _ in range(n_samples):
            # Add noise based on distance
            distance = abs(true_ghost_pos[0] - pacman_pos[0]) + abs(
                true_ghost_pos[1] - pacman_pos[1]
            )
            noise_std = min(
                distance * self.pomdp.observation_noise_factor,
                self.pomdp.max_observation_noise,
            )

            # Add Gaussian noise to ghost position
            noise_row = np.random.normal(0, noise_std)
            noise_col = np.random.normal(0, noise_std)

            observed_row = int(np.round(true_ghost_pos[0] + noise_row))
            observed_col = int(np.round(true_ghost_pos[1] + noise_col))

            # Clamp to valid bounds
            observed_row = max(0, min(self.pomdp.maze_size[0] - 1, observed_row))
            observed_col = max(0, min(self.pomdp.maze_size[1] - 1, observed_col))

            observations.append((observed_row, observed_col))

        return observations

    def probability(self, values: List[Tuple[int, int]]) -> np.ndarray:
        """Calculate observation probabilities."""
        if self.next_state.terminal:
            return np.array([1.0 if obs == (-1, -1) else 0.0 for obs in values])

        true_ghost_pos = self.next_state.ghost_pos
        pacman_pos = self.next_state.pacman_pos
        distance = abs(true_ghost_pos[0] - pacman_pos[0]) + abs(
            true_ghost_pos[1] - pacman_pos[1]
        )
        noise_std = min(
            distance * self.pomdp.observation_noise_factor,
            self.pomdp.max_observation_noise,
        )

        if noise_std == 0:
            noise_std = 1e-6  # Avoid division by zero

        probs = []
        for obs in values:
            if obs == (-1, -1):
                prob = 0.0
            else:
                # Gaussian probability
                row_diff = obs[0] - true_ghost_pos[0]
                col_diff = obs[1] - true_ghost_pos[1]
                distance_sq = row_diff**2 + col_diff**2
                prob = np.exp(-distance_sq / (2 * noise_std**2))
            probs.append(prob)

        probs = np.array(probs)
        total = np.sum(probs)
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
        Basic usage::

            # Create 7x7 maze with walls and pellets
            walls = {(1, 1), (1, 2), (3, 3)}
            pellets = [(0, 2), (2, 0), (4, 4)]
            pomdp = PacManPOMDP(
                maze_size=(7, 7),
                walls=walls,
                initial_pellets=pellets
            )

            # Sample initial state
            initial_state = pomdp.initial_state_dist().sample()[0]

            # Execute action
            next_state, obs, reward = pomdp.sample_next_step(initial_state, 1)
    """

    def __init__(
        self,
        maze_size: Tuple[int, int] = (7, 7),
        walls: Set[Tuple[int, int]] = None,
        initial_pellets: List[Tuple[int, int]] = None,
        initial_pacman_pos: Tuple[int, int] = (0, 0),
        initial_ghost_pos: Tuple[int, int] = (6, 6),
        pellet_reward: float = 10.0,
        ghost_collision_penalty: float = -100.0,
        step_penalty: float = -1.0,
        win_reward: float = 100.0,
        ghost_aggressiveness: float = 2.0,
        observation_noise_factor: float = 0.5,
        max_observation_noise: float = 2.0,
        discount_factor: float = 0.95,
        name: str = "PacManPOMDP",
        output_dir: Optional[Path] = None,
        debug: bool = False,
    ):
        """Initialize PacMan POMDP.

        Args:
            maze_size: Grid dimensions (rows, cols). Defaults to (7, 7).
            walls: Set of wall positions. Defaults to empty set.
            initial_pellets: Initial pellet positions. Defaults to corner positions.
            initial_pacman_pos: PacMan starting position. Defaults to (0, 0).
            initial_ghost_pos: Ghost starting position. Defaults to (6, 6).
            pellet_reward: Reward for collecting pellets. Defaults to 10.0.
            ghost_collision_penalty: Penalty for ghost collision. Defaults to -100.0.
            step_penalty: Cost per action. Defaults to -1.0.
            win_reward: Reward for collecting all pellets. Defaults to 100.0.
            ghost_aggressiveness: Ghost pursuit intensity. Defaults to 2.0.
            observation_noise_factor: Observation noise factor. Defaults to 0.5.
            max_observation_noise: Maximum observation noise. Defaults to 2.0.
            discount_factor: Discount factor. Defaults to 0.95.
            name: Environment name. Defaults to "PacManPOMDP".
            output_dir: Output directory for logging. Defaults to None.
            debug: Enable debug logging. Defaults to False.
        """
        # Calculate reward range based on parameters
        min_reward = step_penalty + ghost_collision_penalty
        max_reward = step_penalty + pellet_reward + win_reward
        
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE, 
            observation_space=SpaceType.DISCRETE
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
        self.initial_ghost_pos = initial_ghost_pos
        self.pellet_reward = pellet_reward
        self.ghost_collision_penalty = ghost_collision_penalty
        self.step_penalty = step_penalty
        self.win_reward = win_reward
        self.ghost_aggressiveness = ghost_aggressiveness
        self.observation_noise_factor = observation_noise_factor
        self.max_observation_noise = max_observation_noise

        # Set default pellets if none provided
        if initial_pellets is None:
            # Place pellets away from corners to avoid conflicts with initial positions
            self.initial_pellets = [
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

    def _validate_parameters(self):
        """Validate environment parameters."""
        if self.maze_size[0] <= 0 or self.maze_size[1] <= 0:
            raise ValueError("Maze size must be positive")

        # Check positions are within bounds
        for pos in [self.initial_pacman_pos, self.initial_ghost_pos]:
            if not (
                0 <= pos[0] < self.maze_size[0] and 0 <= pos[1] < self.maze_size[1]
            ):
                raise ValueError(
                    f"Position {pos} is outside maze bounds {self.maze_size}"
                )

        # Check pellets are within bounds and not walls
        for pellet_pos in self.initial_pellets:
            if not (
                0 <= pellet_pos[0] < self.maze_size[0]
                and 0 <= pellet_pos[1] < self.maze_size[1]
            ):
                raise ValueError(
                    f"Pellet position {pellet_pos} is outside maze bounds {self.maze_size}"
                )
            if pellet_pos in self.walls:
                raise ValueError(f"Pellet position {pellet_pos} is inside a wall")

        # Check initial positions are not walls
        if self.initial_pacman_pos in self.walls:
            raise ValueError(
                f"Initial PacMan position {self.initial_pacman_pos} is inside a wall"
            )
        if self.initial_ghost_pos in self.walls:
            raise ValueError(
                f"Initial ghost position {self.initial_ghost_pos} is inside a wall"
            )

    def get_actions(self) -> List[int]:
        """Get all available actions."""
        return list(range(len(self.action_names)))

    def state_transition_model(
        self, state: PacManState, action: int
    ) -> PacManStateTransitionModel:
        """Get state transition model."""
        return PacManStateTransitionModel(state, action, self)

    def observation_model(
        self, next_state: PacManState, action: int
    ) -> PacManObservationModel:
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
        if next_state.pacman_pos == next_state.ghost_pos:
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
            ghost_pos=self.initial_ghost_pos,
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

                # Calculate average distance between PacMan and ghost throughout episode
                # and count collision encounters
                episode_distances = []
                episode_collisions = 0
                for step_data in history.history:
                    if isinstance(step_data.state, PacManState):
                        pacman_pos = step_data.state.pacman_pos
                        ghost_pos = step_data.state.ghost_pos

                        # Manhattan distance
                        distance = abs(pacman_pos[0] - ghost_pos[0]) + abs(
                            pacman_pos[1] - ghost_pos[1]
                        )
                        episode_distances.append(distance)

                        # Count collision encounters (when PacMan and ghost are at same position)
                        if pacman_pos == ghost_pos:
                            episode_collisions += 1

                if episode_distances:
                    avg_distance = np.mean(episode_distances)
                    pacman_ghost_distances.append(avg_distance)

                collision_encounters.append(episode_collisions)
            else:
                wins.append(0)
                pellets_collected.append(0)
                collision_encounters.append(0)

        if wins:
            win_rate = np.mean(wins)
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
            avg_pellets = np.mean(pellets_collected)
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
            avg_length = np.mean(episode_lengths)
            ci_low, ci_high = confidence_interval(episode_lengths)
            metrics.append(
                MetricValue(
                    name="avg_episode_length",
                    value=avg_length,
                    lower_confidence_bound=ci_low,
                    upper_confidence_bound=ci_high,
                )
            )

        # Average distance between PacMan and ghost metric
        if pacman_ghost_distances:
            avg_distance = np.mean(pacman_ghost_distances)
            ci_low, ci_high = confidence_interval(pacman_ghost_distances)
            metrics.append(
                MetricValue(
                    name="avg_pacman_ghost_distance",
                    value=avg_distance,
                    lower_confidence_bound=ci_low,
                    upper_confidence_bound=ci_high,
                )
            )

        # Collision encounters metric
        if collision_encounters:
            avg_collisions = np.mean(collision_encounters)
            ci_low, ci_high = confidence_interval(collision_encounters)
            metrics.append(
                MetricValue(
                    name="avg_collision_encounters",
                    value=avg_collisions,
                    lower_confidence_bound=ci_low,
                    upper_confidence_bound=ci_high,
                )
            )
            
            # Total collision encounters metric
            total_collisions = sum(collision_encounters)
            total_ci_low, total_ci_high = confidence_interval(collision_encounters)
            metrics.append(
                MetricValue(
                    name="total_collision_encounters",
                    value=total_collisions,
                    lower_confidence_bound=total_ci_low * len(collision_encounters),
                    upper_confidence_bound=total_ci_high * len(collision_encounters),
                )
            )

        return metrics

    def visualize_path(
        self, path: List[PacManState], actions: List[int], cache_path: Path
    ):
        """Visualize PacMan path through the maze using sprite-based rendering."""
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")

        # Constants for sprite rendering
        TILE_SIZE = 32

        # Get sprite directory (relative to this module)
        module_dir = Path(__file__).parent
        sprite_dir = module_dir / "img"

        def load_sprites():
            """Load sprite images with fallback to simple colored circles."""
            sprites = {}

            # Try to load PacMan sprite - prioritize the new pacman_head.jpg
            pacman_head_path = sprite_dir / "pacman_head.jpg"
            pacman_png_path = sprite_dir / "pocman.png"

            if pacman_head_path.exists():
                sprites["pacman"] = (
                    Image.open(pacman_head_path)
                    .convert("RGBA")
                    .resize((TILE_SIZE, TILE_SIZE))
                )
            elif pacman_png_path.exists():
                sprites["pacman"] = (
                    Image.open(pacman_png_path)
                    .convert("RGBA")
                    .resize((TILE_SIZE, TILE_SIZE))
                )
            else:
                # Create simple yellow circle for PacMan
                img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.ellipse(
                    [4, 4, TILE_SIZE - 4, TILE_SIZE - 4], fill=(255, 255, 0, 255)
                )
                sprites["pacman"] = img

            # Try to load Ghost sprite
            ghost_path = sprite_dir / "ghosts.png"
            if ghost_path.exists():
                sprites["ghost"] = (
                    Image.open(ghost_path)
                    .convert("RGBA")
                    .resize((TILE_SIZE, TILE_SIZE))
                )
            else:
                # Create simple red square for Ghost
                img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.rectangle(
                    [4, 4, TILE_SIZE - 4, TILE_SIZE - 4], fill=(255, 0, 0, 255)
                )
                sprites["ghost"] = img

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
                        draw.rectangle(
                            [x, y, x + TILE_SIZE, y + TILE_SIZE], fill=(20, 20, 80, 255)
                        )
                    else:
                        # Floor = black
                        draw.rectangle(
                            [x, y, x + TILE_SIZE, y + TILE_SIZE], fill=(0, 0, 0, 255)
                        )

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

            # Draw Ghost
            gr, gc = state.ghost_pos
            if 0 <= gr < rows and 0 <= gc < cols:
                ghost_x, ghost_y = gc * TILE_SIZE, gr * TILE_SIZE
                canvas.paste(sprites["ghost"], (ghost_x, ghost_y), sprites["ghost"])

            # Draw PacMan (last so it can overlap with ghost if collision)
            pr, pc = state.pacman_pos
            if 0 <= pr < rows and 0 <= pc < cols:
                pacman_x, pacman_y = pc * TILE_SIZE, pr * TILE_SIZE
                if state.pacman_pos == state.ghost_pos:
                    # Collision - draw explosion effect
                    draw.ellipse(
                        [
                            pacman_x,
                            pacman_y,
                            pacman_x + TILE_SIZE,
                            pacman_y + TILE_SIZE,
                        ],
                        fill=(255, 0, 0, 200),
                    )
                    draw.text((pacman_x + 8, pacman_y + 8), "💥", fill=(255, 255, 255))
                else:
                    canvas.paste(
                        sprites["pacman"], (pacman_x, pacman_y), sprites["pacman"]
                    )

            # Add text overlay at bottom
            text_y = rows * TILE_SIZE + 5
            draw.text(
                (5, text_y), f"Step {step_num}: {action_name}", fill=(255, 255, 255)
            )
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

    def cache_visualization(self, history: History, cache_path: Path) -> None:
        """Cache visualization of episode history."""
        if not history.history:
            raise ValueError("Cannot visualize empty history")

        # Extract path and actions
        path = [step.state for step in history.history]
        actions = [
            step.action for step in history.history[:-1]
        ]  # Last step has no action

        self.visualize_path(path, actions, cache_path)


def create_simple_maze_pacman(
    maze_size: int = 7, num_walls: int = 5, seed: Optional[int] = None
) -> PacManPOMDP:
    """Create a simple PacMan instance with random walls.

    Args:
        maze_size: Size of square maze. Defaults to 7.
        num_walls: Number of walls to place randomly. Defaults to 5.
        seed: Random seed. Defaults to None.

    Returns:
        Randomly configured PacMan POMDP
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

    all_positions = [
        (r, c) for r in range(1, maze_size - 1) for c in range(1, maze_size - 1)
    ]
    available_positions = [pos for pos in all_positions if pos not in avoid_positions]

    if len(available_positions) < num_walls:
        num_walls = len(available_positions)

    if num_walls > 0:
        wall_indices = np.random.choice(
            len(available_positions), size=num_walls, replace=False
        )
        walls = {available_positions[i] for i in wall_indices}
    else:
        walls = set()

    return PacManPOMDP(
        maze_size=(maze_size, maze_size),
        walls=walls,
        initial_pellets=pellets,
        initial_pacman_pos=(0, 0),
        initial_ghost_pos=(maze_size - 1, maze_size - 1),
    )
