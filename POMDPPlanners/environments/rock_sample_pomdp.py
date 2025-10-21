"""Module for RockSample POMDP environment.

This module provides the RockSample POMDP environment implementation based on the
classic rock sampling problem, aligned with the Julia RockSample.jl implementation.

The environment involves a robot navigating a grid world with rocks that are either
good or bad. The robot must use a noisy sensor to determine rock quality and decide
whether to sample them, balancing exploration and exploitation.

Classes:
    RockSampleState: Represents the state of the environment
    RockSamplePOMDP: The main POMDP environment implementation
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
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


@dataclass(frozen=True)
class RockSampleState:
    """State representation for RockSample POMDP.

    Attributes:
        robot_pos: Robot position as (row, col) tuple
        rocks: Boolean array indicating rock quality (True=good, False=bad)
    """

    robot_pos: Tuple[int, int]
    rocks: Tuple[bool, ...]  # Tuple for immutability

    def __post_init__(self):
        """Validate state components."""
        if not isinstance(self.robot_pos, tuple) or len(self.robot_pos) != 2:
            raise ValueError("robot_pos must be a tuple of two integers")
        if not isinstance(self.rocks, tuple):
            raise ValueError("rocks must be a tuple of booleans")


class RockSampleStateTransitionModel(StateTransitionModel):
    """State transition model for RockSample POMDP."""

    def __init__(self, state: RockSampleState, action: int, pomdp: "RockSamplePOMDP"):
        """Initialize transition model.

        Args:
            state: Current state
            action: Action to execute
            pomdp: Reference to the POMDP environment
        """
        super().__init__(state=state, action=action)
        self.pomdp = pomdp

    def sample(self, n_samples: int = 1) -> List[RockSampleState]:
        """Sample next states (deterministic transitions)."""
        next_state = self._compute_next_state()
        return [next_state] * n_samples

    def _compute_next_state(self) -> RockSampleState:
        """Compute the deterministic next state."""
        robot_row, robot_col = self.state.robot_pos
        rocks = list(self.state.rocks)

        # Handle terminal state
        if robot_col >= self.pomdp.map_size[1]:
            return RockSampleState((-1, -1), tuple(rocks))

        # Movement actions
        if self.action == 1:  # North
            new_pos = (max(0, robot_row - 1), robot_col)
        elif self.action == 2:  # East
            new_pos = (
                robot_row,
                robot_col + 1,
            )  # Allow moving beyond boundary for exit
        elif self.action == 3:  # South
            new_pos = (min(self.pomdp.map_size[0] - 1, robot_row + 1), robot_col)
        elif self.action == 4:  # West
            new_pos = (robot_row, max(0, robot_col - 1))
        elif self.action == 0:  # Sample
            new_pos = (robot_row, robot_col)
            # Check if robot is at a rock position and sample it
            for i, rock_pos in enumerate(self.pomdp.rock_positions):
                if (robot_row, robot_col) == rock_pos:
                    rocks[i] = False  # Rock becomes bad after sampling
                    break
        else:  # Check actions (5 and above)
            new_pos = (robot_row, robot_col)  # Stay in place for checking

        # Handle exit condition - robot must move beyond right boundary to exit
        if new_pos[1] >= self.pomdp.map_size[1]:
            return RockSampleState((-1, -1), tuple(rocks))

        return RockSampleState(new_pos, tuple(rocks))


class RockSampleObservationModel(ObservationModel):
    """Observation model for RockSample POMDP."""

    def __init__(self, next_state: RockSampleState, action: int, pomdp: "RockSamplePOMDP"):
        """Initialize observation model.

        Args:
            next_state: Next state after transition
            action: Action that was executed
            pomdp: Reference to the POMDP environment
        """
        super().__init__(next_state=next_state, action=action)
        self.pomdp = pomdp

    def sample(self, n_samples: int = 1) -> List[str]:
        """Sample observations."""
        if self.action <= 4:  # Movement or sample actions
            return ["none"] * n_samples

        # Check actions (5 and above)
        rock_idx = self.action - 5
        if rock_idx >= len(self.pomdp.rock_positions):
            return ["none"] * n_samples

        # Calculate observation probabilities based on distance and rock quality
        robot_pos = self.next_state.robot_pos
        rock_pos = self.pomdp.rock_positions[rock_idx]
        rock_quality = self.next_state.rocks[rock_idx]

        # Calculate Euclidean distance
        distance = math.sqrt((robot_pos[0] - rock_pos[0]) ** 2 + (robot_pos[1] - rock_pos[1]) ** 2)

        # Sensor efficiency decreases exponentially with distance
        efficiency = math.exp(-distance / self.pomdp.sensor_efficiency)

        observations = []
        for _ in range(n_samples):
            if np.random.random() < efficiency:
                # Correct observation with high probability
                obs = "good" if rock_quality else "bad"
            else:
                # Incorrect observation
                obs = "bad" if rock_quality else "good"
            observations.append(obs)

        return observations

    def probability(self, values: List[str]) -> np.ndarray:
        """Calculate observation probabilities."""
        if self.action <= 4:  # Movement or sample actions
            probs = np.array([1.0 if obs == "none" else 0.0 for obs in values])
            return probs

        # Check actions
        rock_idx = self.action - 5
        if rock_idx >= len(self.pomdp.rock_positions):
            probs = np.array([1.0 if obs == "none" else 0.0 for obs in values])
            return probs

        robot_pos = self.next_state.robot_pos
        rock_pos = self.pomdp.rock_positions[rock_idx]
        rock_quality = self.next_state.rocks[rock_idx]

        distance = math.sqrt((robot_pos[0] - rock_pos[0]) ** 2 + (robot_pos[1] - rock_pos[1]) ** 2)

        efficiency = math.exp(-distance / self.pomdp.sensor_efficiency)

        probs = []
        for obs in values:
            if obs == "none":
                prob = 0.0
            elif obs == "good":
                prob = efficiency if rock_quality else (1.0 - efficiency)
            elif obs == "bad":
                prob = (1.0 - efficiency) if rock_quality else efficiency
            else:
                prob = 0.0
            probs.append(prob)

        return np.array(probs)


class RockSamplePOMDP(DiscreteActionsEnvironment):
    """RockSample POMDP environment aligned with Julia RockSample.jl.

    This environment implements the classic rock sampling problem where a robot
    must navigate a grid, use sensors to evaluate rocks, and decide which ones
    to sample while balancing exploration costs and sampling rewards.

    Attributes:
        map_size: Grid dimensions as (rows, cols)
        rock_positions: List of rock positions as (row, col) tuples
        init_pos: Initial robot position
        sensor_efficiency: Sensor noise parameter (higher = less noise)
        bad_rock_penalty: Penalty for sampling a bad rock
        good_rock_reward: Reward for sampling a good rock
        step_penalty: Cost for each action
        sensor_use_penalty: Additional cost for using sensor
        exit_reward: Reward for reaching the exit

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Initialize environment
        >>> env = RockSamplePOMDP(map_size=(5, 5), rock_positions=[(0, 0), (2, 2), (3, 3)])
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
        map_size: Tuple[int, int] = (5, 5),
        rock_positions: Optional[List[Tuple[int, int]]] = None,
        init_pos: Tuple[int, int] = (0, 0),
        sensor_efficiency: float = 10.0,
        bad_rock_penalty: float = -10.0,
        good_rock_reward: float = 10.0,
        step_penalty: float = 0.0,
        sensor_use_penalty: float = 0.0,
        exit_reward: float = 10.0,
        dangerous_areas: Optional[List[Tuple[int, int]]] = None,
        dangerous_area_radius: float = 1.0,
        dangerous_area_penalty: float = 5.0,
        discount_factor: float = 0.95,
        name: str = "RockSample",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize RockSample POMDP.

        Args:
            map_size: Grid dimensions (rows, cols). Defaults to (5, 5).
            rock_positions: Rock locations. Defaults to [(0,0), (2,2), (3,3)].
            init_pos: Initial robot position. Defaults to (0, 0).
            sensor_efficiency: Sensor parameter. Defaults to 20.0.
            bad_rock_penalty: Bad rock penalty. Defaults to -10.0.
            good_rock_reward: Good rock reward. Defaults to 10.0.
            step_penalty: Action cost. Defaults to 0.0.
            sensor_use_penalty: Sensor cost. Defaults to 0.0.
            exit_reward: Exit reward. Defaults to 10.0.
            dangerous_areas: List of dangerous area center positions as (row, col) tuples. Defaults to None.
            dangerous_area_radius: Radius around dangerous area centers. Defaults to 1.0.
            dangerous_area_penalty: Penalty magnitude applied randomly when in dangerous areas. Defaults to 5.0.
            discount_factor: Discount factor. Defaults to 0.95.
            name: Environment name. Defaults to "RockSample".
            output_dir: Output directory for logging. Defaults to None.
            debug: Enable debug logging. Defaults to False.
        """
        # Calculate reward range based on parameters
        min_reward = step_penalty + bad_rock_penalty + sensor_use_penalty
        max_reward = step_penalty + exit_reward

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
            use_queue_logger=use_queue_logger,
        )

        self.map_size = map_size
        self.rock_positions = (
            rock_positions if rock_positions is not None else [(0, 0), (2, 2), (3, 3)]
        )
        self.init_pos = init_pos
        self.sensor_efficiency = sensor_efficiency
        self.bad_rock_penalty = bad_rock_penalty
        self.good_rock_reward = good_rock_reward
        self.step_penalty = step_penalty
        self.sensor_use_penalty = sensor_use_penalty
        self.exit_reward = exit_reward
        self.dangerous_areas: List[Tuple[int, int]] = (
            dangerous_areas if dangerous_areas is not None else []
        )
        self.dangerous_area_radius = dangerous_area_radius
        self.dangerous_area_penalty = dangerous_area_penalty

        # Validate parameters
        self._validate_parameters()

        # Define actions: 0=sample, 1=north, 2=east, 3=south, 4=west, 5+=check_rock_i
        self.action_names = ["sample", "north", "east", "south", "west"]
        self.action_names.extend([f"check_rock_{i}" for i in range(len(self.rock_positions))])

        # Action to direction vector mapping for visualization
        self.action_to_vector = {
            0: (0, 0),  # sample - no movement
            1: (0, -1),  # north - up (negative row)
            2: (1, 0),  # east - right (positive col)
            3: (0, 1),  # south - down (positive row)
            4: (-1, 0),  # west - left (negative col)
        }
        # Check actions don't involve movement
        for i in range(5, len(self.action_names)):
            self.action_to_vector[i] = (0, 0)

    def _validate_parameters(self):
        """Validate environment parameters."""
        if self.map_size[0] <= 0 or self.map_size[1] <= 0:
            raise ValueError("Map size must be positive")
        for pos in self.rock_positions:
            if not (0 <= pos[0] < self.map_size[0] and 0 <= pos[1] < self.map_size[1]):
                raise ValueError(f"Rock position {pos} is outside map bounds {self.map_size}")
        if not (
            0 <= self.init_pos[0] < self.map_size[0] and 0 <= self.init_pos[1] < self.map_size[1]
        ):
            raise ValueError(
                f"Initial position {self.init_pos} is outside map bounds {self.map_size}"
            )

    def _is_in_dangerous_area(self, position: Tuple[int, int]) -> bool:
        """Check if a position is within any dangerous area.

        Args:
            position: Position to check as (row, col) tuple

        Returns:
            True if position is within radius of any dangerous area center
        """
        if not self.dangerous_areas:
            return False

        pos_row, pos_col = position

        for danger_row, danger_col in self.dangerous_areas:
            # Calculate Euclidean distance
            distance = math.sqrt((pos_row - danger_row) ** 2 + (pos_col - danger_col) ** 2)
            if distance <= self.dangerous_area_radius:
                return True

        return False

    def get_actions(self) -> List[int]:
        """Get all available actions."""
        return list(range(len(self.action_names)))

    def state_transition_model(
        self, state: RockSampleState, action: int
    ) -> RockSampleStateTransitionModel:
        """Get state transition model."""
        return RockSampleStateTransitionModel(state, action, self)

    def observation_model(
        self, next_state: RockSampleState, action: int
    ) -> RockSampleObservationModel:
        """Get observation model."""
        return RockSampleObservationModel(next_state, action, self)

    def reward(self, state: RockSampleState, action: int) -> float:
        """Calculate immediate reward."""
        total_reward = self.step_penalty

        # Check if robot exits the grid
        robot_row, robot_col = state.robot_pos
        if action == 2 and robot_col == self.map_size[1] - 1:  # East at right edge
            total_reward += self.exit_reward
            return total_reward

        # Sample action rewards
        if action == 0:  # Sample
            for i, rock_pos in enumerate(self.rock_positions):
                if (robot_row, robot_col) == rock_pos:
                    if state.rocks[i]:  # Good rock
                        total_reward += self.good_rock_reward
                    else:  # Bad rock
                        total_reward += self.bad_rock_penalty
                    break

        # Sensor use penalty
        if action >= 5:  # Check actions
            total_reward += self.sensor_use_penalty

        # Add dangerous area penalty/bonus with 50% probability
        if self._is_in_dangerous_area(state.robot_pos):
            # Random penalty or bonus with equal probability
            danger_modifier = (
                self.dangerous_area_penalty
                if np.random.random() < 0.5
                else -self.dangerous_area_penalty
            )
            total_reward += danger_modifier

        return total_reward

    def is_terminal(self, state: RockSampleState) -> bool:
        """Check if state is terminal."""
        return state.robot_pos == (-1, -1)

    def initial_state_dist(self) -> DiscreteDistribution:
        """Get initial state distribution."""
        # All rocks start as good with equal probability
        num_rocks = len(self.rock_positions)
        possible_rock_states = []

        # Generate all possible rock configurations
        for i in range(2**num_rocks):
            rock_config = tuple(bool(i & (1 << j)) for j in range(num_rocks))
            initial_state = RockSampleState(self.init_pos, rock_config)
            possible_rock_states.append(initial_state)

        # Equal probability for all configurations
        probs = np.ones(len(possible_rock_states)) / len(possible_rock_states)

        return DiscreteDistribution(values=possible_rock_states, probs=probs)

    def initial_observation_dist(self) -> DiscreteDistribution:
        """Get initial observation distribution."""
        return DiscreteDistribution(values=["none"], probs=np.array([1.0]))

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        """Check if two observations are equal."""
        return observation1 == observation2

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        """Compute environment-specific metrics."""
        if not histories:
            return []

        metrics = []

        # Calculate average number of rocks sampled
        rocks_sampled = []
        for history in histories:
            sampled_count = 0
            for step in history.history:
                if hasattr(step, "action") and step.action == 0:  # Sample action
                    sampled_count += 1
            rocks_sampled.append(sampled_count)

        if rocks_sampled:
            mean_rocks = float(np.mean(rocks_sampled))
            ci_low, ci_high = confidence_interval(rocks_sampled)
            metrics.append(
                MetricValue(
                    name="avg_rocks_sampled",
                    value=mean_rocks,
                    lower_confidence_bound=ci_low,
                    upper_confidence_bound=ci_high,
                )
            )

        # Calculate exit success rate
        exits = [
            1 if any(self.is_terminal(step.state) for step in history.history) else 0
            for history in histories
        ]

        if exits:
            exit_rate = float(np.mean(exits))
            ci_low, ci_high = confidence_interval(exits)
            metrics.append(
                MetricValue(
                    name="exit_success_rate",
                    value=exit_rate,
                    lower_confidence_bound=ci_low,
                    upper_confidence_bound=ci_high,
                )
            )

        # Calculate dangerous area metrics
        dangerous_area_steps = []
        for history in histories:
            steps_in_danger = 0
            for step in history.history:
                if self._is_in_dangerous_area(step.state.robot_pos):
                    steps_in_danger += 1
            dangerous_area_steps.append(steps_in_danger)

        if dangerous_area_steps:
            avg_dangerous_steps = float(np.mean(dangerous_area_steps))
            ci_low, ci_high = confidence_interval(dangerous_area_steps)

            metrics.append(
                MetricValue(
                    name="average_dangerous_area_steps",
                    value=avg_dangerous_steps,
                    lower_confidence_bound=ci_low,
                    upper_confidence_bound=ci_high,
                )
            )

        return metrics

    def visualize_path(self, path: List[RockSampleState], actions: List[int], cache_path: Path):
        """Visualize robot path through the environment."""
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")

        fig: Figure
        ax: Axes
        fig_temp, ax_temp = plt.subplots(figsize=(10, 8))
        fig = cast(Figure, fig_temp)
        ax = cast(Axes, ax_temp)
        ax.set_xlim(-0.5, self.map_size[1] - 0.5)
        ax.set_ylim(self.map_size[0] - 0.5, -0.5)  # Flip y-axis for standard grid display
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title("RockSample POMDP Episode Visualization")

        # Initialize empty scatter plots for rocks (will be updated dynamically)
        rock_scatters = []
        for i, rock_pos in enumerate(self.rock_positions):
            scatter = ax.scatter([], [], s=200, marker="s", alpha=0.7, label=f"Rock {i}")
            rock_scatters.append(scatter)

        # Plot dangerous areas as red circles
        danger_patches = []
        for i, danger_center in enumerate(self.dangerous_areas):
            row, col = danger_center
            circle = plt.Circle(  # type: ignore[attr-defined]
                (col, row),
                float(self.dangerous_area_radius),  # type: ignore[arg-type]
                facecolor="red",
                edgecolor="none",
                alpha=0.3,
                label="Dangerous Areas" if i == 0 else "",
            )  # Only label first area
            ax.add_patch(circle)
            if i == 0:  # Keep reference for legend
                danger_patches.append(circle)

        # Plot exit zone
        exit_x = self.map_size[1] - 0.5
        ax.axvline(x=float(exit_x), color="gold", linewidth=3, alpha=0.7, label="Exit")  # type: ignore[arg-type]

        # Initialize robot position
        robot_scatter = ax.scatter([], [], s=150, c="blue", marker="o", zorder=5, label="Robot")
        path_line = cast(Line2D, ax.plot([], [], "b-", alpha=0.5, linewidth=2, label="Path")[0])

        # Initialize action arrow
        arrow = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": "red", "lw": 2},
            zorder=6,
        )

        # Action text
        action_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            verticalalignment="top",
        )

        # Sample result text (success/failure indicator)
        sample_text = ax.text(
            0.02,
            0.02,
            "",
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="bottom",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "gold",
                "edgecolor": "red",
                "linewidth": 3,
                "alpha": 0.9,
            },
            color="red",
            visible=False,
        )

        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        def animate(frame):
            if frame >= len(path):
                return tuple(
                    [robot_scatter, path_line, arrow, action_text, sample_text] + rock_scatters
                )

            state = path[frame]
            robot_pos = state.robot_pos

            # Update rock colors based on current state
            for i, rock_pos in enumerate(self.rock_positions):
                if i < len(state.rocks):
                    color = "green" if state.rocks[i] else "red"
                    rock_scatters[i].set_offsets([[rock_pos[1], rock_pos[0]]])
                    rock_scatters[i].set_color(color)

            # Update robot position (handle terminal state)
            if robot_pos == (-1, -1):
                # Robot has exited - don't show it
                robot_scatter.set_offsets(np.empty((0, 2)))
                # Hide arrow and sample text for terminal state
                arrow.set_visible(False)
                sample_text.set_visible(False)
            else:
                robot_scatter.set_offsets([[robot_pos[1], robot_pos[0]]])

                # Update action arrow
                if frame < len(actions):
                    action = actions[frame]
                    dx, dy = self.action_to_vector.get(action, (0, 0))

                    # Only show arrow for movement actions (not sample or check)
                    if dx != 0 or dy != 0:
                        # Position arrow from robot position
                        x, y = robot_pos[1], robot_pos[0]  # Convert to plot coordinates
                        # Scale arrow for better visibility
                        arrow_scale = 0.4
                        arrow.set_position((x, y))
                        arrow.xy = (x + dx * arrow_scale, y + dy * arrow_scale)
                        arrow.set_visible(True)
                    else:
                        # Hide arrow for non-movement actions
                        arrow.set_visible(False)
                else:
                    # Hide arrow when no action
                    arrow.set_visible(False)

            # Update path
            valid_positions = [
                pos for pos in [p.robot_pos for p in path[: frame + 1]] if pos != (-1, -1)
            ]
            if valid_positions:
                path_x = [pos[1] for pos in valid_positions]
                path_y = [pos[0] for pos in valid_positions]
                path_line.set_data(path_x, path_y)

            # Update action text and sample result
            if frame < len(actions):
                action = actions[frame]
                action_name = self.action_names[action]
                action_text.set_text(f"Step: {frame+1}/{len(path)}\nAction: {action_name}")

                # Check for sample action and determine success/failure
                if action == 0:  # Sample action
                    robot_row, robot_col = robot_pos
                    sample_success = False

                    # Check if robot is on a rock position
                    for i, rock_pos in enumerate(self.rock_positions):
                        if (robot_row, robot_col) == rock_pos:
                            # Check if the rock is good (True)
                            if state.rocks[i]:
                                sample_success = True
                            break

                    if sample_success:
                        sample_text.set_text("★ VALUABLE! ★")
                        sample_text.set_bbox(
                            {
                                "boxstyle": "round,pad=0.5",
                                "facecolor": "lightgreen",
                                "edgecolor": "green",
                                "linewidth": 3,
                                "alpha": 0.9,
                            }
                        )
                        sample_text.set_color("darkgreen")
                        sample_text.set_visible(True)
                    else:
                        sample_text.set_text("✗ WORTHLESS! ✗")
                        sample_text.set_bbox(
                            {
                                "boxstyle": "round,pad=0.5",
                                "facecolor": "lightcoral",
                                "edgecolor": "red",
                                "linewidth": 3,
                                "alpha": 0.9,
                            }
                        )
                        sample_text.set_color("darkred")
                        sample_text.set_visible(True)
                else:
                    # Hide sample result for non-sample actions
                    sample_text.set_visible(False)
            else:
                action_text.set_text(f"Step: {frame+1}/{len(path)}\nAction: Terminal")
                sample_text.set_visible(False)

            return tuple(
                [robot_scatter, path_line, arrow, action_text, sample_text] + rock_scatters
            )

        ani = animation.FuncAnimation(
            fig, animate, frames=len(path), interval=1000, blit=False, repeat=False
        )

        plt.tight_layout()

        # Save animation
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            ani.save(cache_path, writer="pillow", fps=1)

        plt.close()

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


def create_random_rock_sample(
    map_size: int = 7, num_rocks: int = 8, seed: Optional[int] = None
) -> RockSamplePOMDP:
    """Create a random RockSample instance.

    Args:
        map_size: Size of square grid. Defaults to 7.
        num_rocks: Number of rocks to place. Defaults to 8.
        seed: Random seed. Defaults to None.

    Returns:
        Randomly configured RockSample POMDP
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random rock positions
    all_positions = [(r, c) for r in range(map_size) for c in range(map_size)]
    rock_positions = list(
        np.random.choice(len(all_positions), size=min(num_rocks, len(all_positions)), replace=False)
    )
    rock_positions = [all_positions[i] for i in rock_positions]

    return RockSamplePOMDP(
        map_size=(map_size, map_size), rock_positions=rock_positions, init_pos=(0, 0)
    )
