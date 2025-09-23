"""Safety Ant Velocity POMDP Environment Implementation.

This module implements a safety-critical velocity control task where an agent
must navigate while avoiding unsafe velocities. The challenge is balancing
exploration and movement rewards with safety constraints under partial observability.

The Safety Ant Velocity POMDP features:
- Continuous 4D state space: [position_x, position_y, velocity_x, velocity_y]
- Discrete action space: [0 (no force), 1 (small), 2 (medium), 3 (large force)]
- Physics-based dynamics with force application and damping
- Noisy observations of both position and velocity
- Safety constraints on maximum velocity magnitude
- Safety-focused metrics tracking violation rates

Key aspects:
- Rewards encourage movement but heavily penalize safety violations
- Episode terminates if velocity becomes critically high
- Force direction is randomized to create uncertainty
- Safety metrics track violation rates over episodes

Classes:
    SafeAntVelocityStateTransition: Physics simulation with force control
    SafeAntVelocityObservation: Noisy position and velocity observations
    SafeAntVelocityPOMDP: Main safety-critical velocity control environment
"""

from typing import List, Any, Tuple, Optional
import numpy as np
from pathlib import Path

from POMDPPlanners.core.environment import (
    ObservationModel,
    StateTransitionModel,
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.simulation import History, StepData, MetricValue
from POMDPPlanners.utils.statistics import confidence_interval
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class SafeAntVelocityStateTransition(StateTransitionModel):
    """Physics-based state transition model for Safety Ant Velocity POMDP.

    This model simulates simplified physics with force application, damping, and
    random force directions. The agent can choose different force magnitudes
    but cannot control the direction, creating uncertainty in the outcomes.

    Physics equations:
    - acceleration = (force - damping * velocity) / mass
    - velocity += acceleration * dt
    - position += velocity * dt

    Attributes:
        state: Current state [position_x, position_y, velocity_x, velocity_y]
        action: Force magnitude index (0=no force, 1=small, 2=medium, 3=large)
        dt: Time step for physics integration
        mass: Mass of the agent (affects acceleration)
        damping: Damping coefficient (opposes velocity)
        max_force: Maximum force magnitude
        force_scales: Force scaling factors for each action [0.0, 0.33, 0.67, 1.0]
        position: Current position [x, y]
        velocity: Current velocity [vx, vy]

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> # Define current state [pos_x, pos_y, vel_x, vel_y]
        >>> state = np.array([0.5, -0.2, 1.0, 0.5])
        >>> action = 2  # Apply medium force
        >>>
        >>> # Create transition model
        >>> transition = SafeAntVelocityStateTransition(
        ...     state=state,
        ...     action=action,
        ...     dt=0.1,
        ...     mass=1.0,
        ...     damping=0.1,
        ...     max_force=1.0
        ... )
        >>>
        >>> # Simulate physics step with random force direction
        >>> next_state = transition.sample()[0]  # doctest: +SKIP
        >>> # Returns new [pos_x, pos_y, vel_x, vel_y] after physics
        >>> new_pos = next_state[:2]  # doctest: +SKIP
        >>> new_vel = next_state[2:4]  # doctest: +SKIP
    """

    def __init__(
        self,
        state: np.ndarray,
        action: int,
        dt: float = 0.1,
        mass: float = 1.0,
        damping: float = 0.1,
        max_force: float = 1.0,
    ):
        super().__init__(state, action)
        self.dt = dt
        self.mass = mass
        self.damping = damping
        self.max_force = max_force

        # Convert action index to force magnitude
        self.force_scales = [0.0, 0.33, 0.67, 1.0]  # Different force magnitudes

        # State components
        self.position = state[:2]
        self.velocity = state[2:4]

    def sample(self, n_samples: int = 1) -> List[Any]:
        next_states = []
        for _ in range(n_samples):
            # Get force magnitude from action
            force_magnitude = self.force_scales[self.action] * self.max_force

            # Random force direction
            force_direction = np.random.uniform(-np.pi, np.pi)
            force = force_magnitude * np.array([np.cos(force_direction), np.sin(force_direction)])

            # Physics simulation (simplified)
            acceleration = (force - self.damping * self.velocity) / self.mass
            new_velocity = self.velocity + acceleration * self.dt
            new_position = self.position + new_velocity * self.dt

            # Combine into new state
            next_state = np.concatenate([new_position, new_velocity])
            next_states.append(next_state)

        return next_states


class SafeAntVelocityObservation(ObservationModel):
    """Noisy observation model for Safety Ant Velocity POMDP.

    This model adds Gaussian noise to both position and velocity measurements,
    creating partial observability that makes velocity estimation challenging.
    Higher noise in velocity measurements reflects the difficulty of measuring
    velocity precisely in practice.

    Attributes:
        next_state: True state after action execution
        action: Action that was taken (not used in observation generation)
        position_noise: Standard deviation of Gaussian noise for position
        velocity_noise: Standard deviation of Gaussian noise for velocity
        position: True position [x, y]
        velocity: True velocity [vx, vy]

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> # True state after physics simulation
        >>> true_state = np.array([0.6, -0.1, 1.2, 0.8])  # [x, y, vx, vy]
        >>> action = 2
        >>>
        >>> # Create observation model
        >>> obs_model = SafeAntVelocityObservation(
        ...     next_state=true_state,
        ...     action=action,
        ...     position_noise=0.1,
        ...     velocity_noise=0.2
        ... )
        >>>
        >>> # Sample noisy observation
        >>> observation = obs_model.sample()[0]  # doctest: +SKIP
        >>> # Returns [noisy_x, noisy_y, noisy_vx, noisy_vy]
        >>> # Position noise: ±0.1, velocity noise: ±0.2
        >>>
        >>> # Calculate observation probability
        >>> prob = obs_model.probability([observation])  # doctest: +SKIP
    """

    def __init__(
        self,
        next_state: np.ndarray,
        action: int,
        position_noise: float = 0.1,
        velocity_noise: float = 0.2,
    ):
        super().__init__(next_state=next_state, action=action)
        self.position_noise = position_noise
        self.velocity_noise = velocity_noise

        # State components
        self.position = next_state[:2]
        self.velocity = next_state[2:4]

    def sample(self, n_samples: int = 1) -> List[Any]:
        observations = []
        for _ in range(n_samples):
            # Add noise to position and velocity observations
            noisy_position = self.position + np.random.normal(0, self.position_noise, size=2)
            noisy_velocity = self.velocity + np.random.normal(0, self.velocity_noise, size=2)

            # Combine observations
            observation = np.concatenate([noisy_position, noisy_velocity])
            observations.append(observation)
        return observations

    def probability(self, observations: List[Any]) -> np.ndarray:
        # Calculate probabilities based on Gaussian noise model for list of observations
        probabilities = []
        for observation in observations:
            # Ensure observation is numpy array with correct shape
            if not isinstance(observation, np.ndarray) or observation.size == 0:
                raise ValueError(
                    f"Expected non-empty numpy array observation, got {type(observation)} with shape {getattr(observation, 'shape', 'unknown')}"
                )

            if observation.shape != (4,):
                raise ValueError(f"Expected observation shape (4,), got {observation.shape}")

            position_diff = observation[:2] - self.position
            velocity_diff = observation[2:4] - self.velocity

            position_log_prob = -0.5 * np.sum(position_diff**2) / (self.position_noise**2)
            velocity_log_prob = -0.5 * np.sum(velocity_diff**2) / (self.velocity_noise**2)

            prob = np.exp(position_log_prob + velocity_log_prob)
            probabilities.append(prob)

        return np.array(probabilities)


class SafeAntVelocityPOMDP(DiscreteActionsEnvironment):
    """Safety-critical velocity control task formulated as a POMDP.

    This environment presents a safety-critical control problem where an agent
    must navigate while keeping velocity below a safety threshold. The challenge
    comes from balancing exploration rewards with safety constraints under noisy
    velocity observations.

    Problem Structure:
    - State: [position_x, position_y, velocity_x, velocity_y] (continuous)
    - Actions: [0=no force, 1=small, 2=medium, 3=large force] (discrete)
    - Observations: Noisy position and velocity measurements (continuous)
    - Rewards: Movement reward - safety violation penalty (if unsafe)
    - Safety constraint: velocity magnitude ≤ safe_velocity_threshold
    - Termination: Velocity exceeds 1.5x safety threshold

    Safety Features:
    - Tracks safety and critical violation rates
    - Heavy penalties for constraint violations
    - Configurable safety thresholds and penalties
    - Physics simulation with uncertainty in force direction

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> # Create safety-critical environment
        >>> safe_env = SafeAntVelocityPOMDP(
        ...     discount_factor=0.99,
        ...     safe_velocity_threshold=2.0,
        ...     safety_violation_penalty=-100.0,
        ...     movement_reward_scale=1.0
        ... )
        >>>
        >>> # Get initial state
        >>> initial_state_dist = safe_env.initial_state_dist()
        >>> state = initial_state_dist.sample()[0]  # [x, y, vx, vy]  # doctest: +SKIP
        >>>
        >>> # Choose force magnitude action
        >>> actions = safe_env.get_actions()  # [0, 1, 2, 3]
        >>> len(actions) == 4
        True
        >>> action = 1  # Apply small force
        >>> # reward = safe_env.reward(state, action)
        >>>
        >>> # Check safety constraint
        >>> test_state = np.array([0.0, 0.0, 1.0, 0.5])  # Test state
        >>> velocity = test_state[2:4]
        >>> speed = np.linalg.norm(velocity)
        >>> is_safe = speed <= safe_env.safe_velocity_threshold
        >>> bool(is_safe)
        True
    """

    def __init__(
        self,
        discount_factor: float,
        safe_velocity_threshold: float = 2.0,
        max_force: float = 1.0,
        dt: float = 0.1,
        mass: float = 1.0,
        damping: float = 0.1,
        position_noise: float = 0.1,
        velocity_noise: float = 0.2,
        safety_violation_penalty: float = -100.0,
        movement_reward_scale: float = 1.0,
        name: str = "SafeVelocityPOMDP",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        self.safe_velocity_threshold = safe_velocity_threshold
        self.max_force = max_force
        self.dt = dt
        self.mass = mass
        self.damping = damping
        self.position_noise = position_noise
        self.velocity_noise = velocity_noise
        self.safety_violation_penalty = safety_violation_penalty
        self.movement_reward_scale = movement_reward_scale

        # Define actions (different force magnitudes)
        self.actions = [0, 1, 2, 3]  # 0: no force, 1: small, 2: medium, 3: large

        # Calculate reward range based on parameters
        # Minimum: no movement reward + safety penalty
        min_reward = 0.0 + safety_violation_penalty
        # Maximum: maximum safe speed (termination threshold) * movement scale
        max_reward = safe_velocity_threshold * 1.5 * movement_reward_scale

        # Create space info with appropriate bounds
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,  # Action space is discrete force magnitudes
            observation_space=SpaceType.CONTINUOUS,  # Observation space is positions and velocities with noise
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

    def state_transition_model(self, state: np.ndarray, action: int) -> StateTransitionModel:
        return SafeAntVelocityStateTransition(
            state=state,
            action=action,
            dt=self.dt,
            mass=self.mass,
            damping=self.damping,
            max_force=self.max_force,
        )

    def observation_model(self, next_state: np.ndarray, action: int) -> ObservationModel:
        return SafeAntVelocityObservation(
            next_state=next_state,
            action=action,
            position_noise=self.position_noise,
            velocity_noise=self.velocity_noise,
        )

    def reward(self, state: np.ndarray, action: int) -> float:
        # Extract velocity components
        velocity = state[2:4]
        speed = np.linalg.norm(velocity)

        # Base reward is proportional to movement (encourage exploration)
        reward = speed * self.movement_reward_scale

        # Penalty for exceeding safe velocity
        if speed > self.safe_velocity_threshold:
            reward += self.safety_violation_penalty

        return reward

    def is_terminal(self, state: np.ndarray) -> bool:
        # Episode ends if velocity exceeds safe threshold by too much
        velocity = state[2:4]
        speed = np.linalg.norm(velocity)
        return speed > self.safe_velocity_threshold * 1.5  # 50% margin

    def initial_state_dist(self) -> Distribution:
        class InitialState(Distribution):
            def sample(self, n_samples: int = 1) -> List[Any]:
                initial_states = []
                for _ in range(n_samples):
                    # Start with zero velocity and random position
                    position = np.random.uniform(-1, 1, size=2)
                    velocity = np.zeros(2)
                    initial_state = np.concatenate([position, velocity])
                    initial_states.append(initial_state)
                return initial_states

        return InitialState()

    def initial_observation_dist(self) -> Distribution:
        return self.initial_state_dist()

    def get_actions(self) -> List[int]:
        return self.actions

    def is_equal_observation(self, observation1: np.ndarray, observation2: np.ndarray) -> bool:
        return np.array_equal(observation1, observation2)

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Cache animated visualization of the safety ant velocity episode.

        Creates an animated GIF showing the ant's movement trajectory with velocity vectors,
        safety zones, force applications, and safety constraint violations.

        Args:
            history: Episode history containing states, actions, and rewards
            cache_path: Path where to save the visualization (must end with .gif)

        Raises:
            ValueError: If history is empty or cache_path doesn't end with .gif
            TypeError: If cache_path is not a Path object
        """
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

        # Extract episode data
        states = [step.state for step in history]
        actions = [step.action for step in history[:-1]]  # Last step has no action
        rewards = [step.reward for step in history]

        # Calculate position bounds for visualization
        all_positions = np.array([state[:2] for state in states])
        x_min, x_max = all_positions[:, 0].min() - 1, all_positions[:, 0].max() + 1
        y_min, y_max = all_positions[:, 1].min() - 1, all_positions[:, 1].max() + 1

        # Set up the plot with dual panels
        fig, (ax_main, ax_speed) = plt.subplots(1, 2, figsize=(16, 8))

        # Main trajectory plot
        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(y_min, y_max)
        ax_main.set_aspect("equal")
        ax_main.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax_main.set_xlabel("X Position", fontsize=12)
        ax_main.set_ylabel("Y Position", fontsize=12)
        ax_main.set_title(
            "Safety Ant Velocity POMDP: Trajectory & Safety Zones",
            fontsize=14,
            fontweight="bold",
        )

        # Speed plot
        ax_speed.set_xlim(0, len(states))
        max_speed = max(np.linalg.norm(state[2:4]) for state in states) if states else 0
        ax_speed.set_ylim(0, max(max_speed * 1.1, self.safe_velocity_threshold * 2))
        ax_speed.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax_speed.set_xlabel("Time Step", fontsize=12)
        ax_speed.set_ylabel("Speed (Velocity Magnitude)", fontsize=12)
        ax_speed.set_title("Speed Over Time", fontsize=14, fontweight="bold")

        # Safety threshold lines on speed plot
        ax_speed.axhline(
            y=self.safe_velocity_threshold,
            color="orange",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"Safety Threshold ({self.safe_velocity_threshold:.1f})",
        )
        ax_speed.axhline(
            y=self.safe_velocity_threshold * 1.5,
            color="red",
            linestyle="-",
            linewidth=2,
            alpha=0.8,
            label=f"Critical Threshold ({self.safe_velocity_threshold * 1.5:.1f})",
        )

        # Initialize visual elements for main plot
        ant_scatter = ax_main.scatter(
            [],
            [],
            s=200,
            c="blue",
            marker="o",
            edgecolor="darkblue",
            linewidth=2,
            zorder=5,
            label="Ant",
        )
        (path_line,) = ax_main.plot([], [], "b-", alpha=0.6, linewidth=2, label="Trajectory")

        # Velocity vector arrow
        velocity_arrow = ax_main.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="green", lw=3, alpha=0.8),
            zorder=6,
            visible=False,
        )

        # Force vector arrow (different color)
        force_arrow = ax_main.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="purple", lw=3, alpha=0.8),
            zorder=6,
            visible=False,
        )

        # Safety zone circles (concentric circles around current position)
        safety_circle = plt.Circle(
            (0, 0),
            0,
            fill=False,
            edgecolor="orange",
            linewidth=2,
            alpha=0.6,
            linestyle="--",
            visible=False,
        )
        critical_circle = plt.Circle(
            (0, 0),
            0,
            fill=False,
            edgecolor="red",
            linewidth=2,
            alpha=0.6,
            linestyle="-",
            visible=False,
        )
        ax_main.add_patch(safety_circle)
        ax_main.add_patch(critical_circle)

        # Speed plot elements
        (speed_line,) = ax_speed.plot([], [], "b-", linewidth=2, label="Speed")
        speed_points = ax_speed.scatter(
            [],
            [],
            s=50,
            c=[],
            cmap="RdYlGn_r",
            vmin=0,
            vmax=self.safe_velocity_threshold * 1.5,
            edgecolor="black",
            linewidth=1,
            zorder=5,
        )

        # Text displays for main plot
        step_text = ax_main.text(
            0.02,
            0.98,
            "",
            transform=ax_main.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
            verticalalignment="top",
            horizontalalignment="left",
        )

        velocity_text = ax_main.text(
            0.02,
            0.88,
            "",
            transform=ax_main.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            verticalalignment="top",
            horizontalalignment="left",
        )

        reward_text = ax_main.text(
            0.02,
            0.78,
            "",
            transform=ax_main.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
            verticalalignment="top",
            horizontalalignment="left",
        )

        # Safety status banner
        safety_text = ax_main.text(
            0.5,
            0.95,
            "",
            transform=ax_main.transAxes,
            fontsize=14,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="green", alpha=0.9),
            horizontalalignment="center",
            verticalalignment="top",
            visible=False,
            zorder=10,
        )

        # Add legends
        ax_main.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax_speed.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        def animate(frame):
            if frame >= len(states):
                return (
                    ant_scatter,
                    path_line,
                    velocity_arrow,
                    force_arrow,
                    speed_line,
                    speed_points,
                    step_text,
                    velocity_text,
                    reward_text,
                    safety_text,
                )

            state = states[frame]
            position = state[:2]
            velocity = state[2:4]
            speed = np.linalg.norm(velocity)

            # Update ant position
            ant_scatter.set_offsets([position])

            # Update trajectory path
            path_positions = [s[:2] for s in states[: frame + 1]]
            if len(path_positions) > 1:
                path_x = [pos[0] for pos in path_positions]
                path_y = [pos[1] for pos in path_positions]
                path_line.set_data(path_x, path_y)

            # Update velocity vector
            if np.linalg.norm(velocity) > 0.01:  # Only show if significant velocity
                velocity_arrow.set_position(position)
                velocity_arrow.xy = position + velocity * 0.5  # Scale for visibility
                velocity_arrow.set_visible(True)
            else:
                velocity_arrow.set_visible(False)

            # Update force vector (if action is applied)
            if frame < len(actions):
                action = actions[frame]
                force_scales = [0.0, 0.33, 0.67, 1.0]
                force_magnitude = force_scales[action] * self.max_force
                if force_magnitude > 0:
                    # Show force direction (simplified representation since actual direction is random)
                    # Use a representative direction that shows the relative force magnitude
                    force_direction = np.array([1.0, 0.5])  # Simplified representation
                    force_direction = force_direction / np.linalg.norm(force_direction)
                    force_arrow.set_position(position)
                    force_arrow.xy = position + force_direction * force_magnitude * 0.8
                    force_arrow.set_visible(True)
                else:
                    force_arrow.set_visible(False)
            else:
                force_arrow.set_visible(False)

            # Update safety circles (showing velocity-based danger zones)
            # Circle radius represents how close the ant is to safety limits
            safety_radius = (
                speed / self.safe_velocity_threshold * 1.0
                if self.safe_velocity_threshold > 0
                else 0.1
            )
            critical_radius = (
                speed / (self.safe_velocity_threshold * 1.5) * 1.5
                if self.safe_velocity_threshold > 0
                else 0.1
            )

            safety_circle.center = position
            safety_circle.set_radius(max(0.1, safety_radius))
            safety_circle.set_visible(True)

            critical_circle.center = position
            critical_circle.set_radius(max(0.1, critical_radius))
            critical_circle.set_visible(speed > self.safe_velocity_threshold)

            # Update speed plot
            speeds = [np.linalg.norm(s[2:4]) for s in states[: frame + 1]]
            time_steps = list(range(len(speeds)))
            speed_line.set_data(time_steps, speeds)

            # Color-code speed points based on safety
            colors = [
                (
                    "green"
                    if s <= self.safe_velocity_threshold
                    else "orange"
                    if s <= self.safe_velocity_threshold * 1.5
                    else "red"
                )
                for s in speeds
            ]
            if speeds:
                speed_points.set_offsets(list(zip(time_steps, speeds)))
                speed_points.set_color(colors)

            # Update text displays
            action_name = f"Force Level {actions[frame]}" if frame < len(actions) else "Terminal"
            step_text.set_text(f"Step: {frame+1}/{len(states)}\nAction: {action_name}")

            velocity_text.set_text(
                f"Velocity: [{velocity[0]:.2f}, {velocity[1]:.2f}]\n" + f"Speed: {speed:.2f}"
            )

            current_reward = rewards[frame] if frame < len(rewards) else 0.0
            total_reward = sum(rewards[: frame + 1])
            reward_text.set_text(
                f"Step Reward: {current_reward:.1f}\nTotal Reward: {total_reward:.1f}"
            )

            # Update safety status
            if speed > self.safe_velocity_threshold * 1.5:
                safety_text.set_text("⚠ CRITICAL VIOLATION ⚠\nTerminal State!")
                safety_text.get_bbox_patch().set_facecolor("red")
                safety_text.set_visible(True)
            elif speed > self.safe_velocity_threshold:
                safety_text.set_text("⚠ SAFETY VIOLATION ⚠")
                safety_text.get_bbox_patch().set_facecolor("orange")
                safety_text.set_visible(True)
            else:
                safety_text.set_text("✓ SAFE OPERATION ✓")
                safety_text.get_bbox_patch().set_facecolor("green")
                safety_text.set_visible(True)

            return (
                ant_scatter,
                path_line,
                velocity_arrow,
                force_arrow,
                speed_line,
                speed_points,
                step_text,
                velocity_text,
                reward_text,
                safety_text,
            )

        # Create animation
        ani = animation.FuncAnimation(
            fig, animate, frames=len(states), interval=1200, blit=False, repeat=True
        )

        plt.tight_layout()

        # Save animation
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ani.save(cache_path, writer="pillow", fps=0.8)
        plt.close(fig)  # Free memory

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        # Initialize metrics
        safety_violations = []
        critical_violations = []

        # Process each history
        for history in histories:
            history_safety_violations = 0
            history_critical_violations = 0
            total_steps = len(history.history)

            for step in history.history:
                # Get velocity from state
                velocity = step.state[2:4]  # [vx, vy]
                speed = np.linalg.norm(velocity)

                # Count safety violations
                if speed > self.safe_velocity_threshold:
                    history_safety_violations += 1

                # Count critical violations (termination condition)
                if speed > self.safe_velocity_threshold * 1.5:
                    history_critical_violations += 1

            # Convert to rates for this history
            if total_steps > 0:
                safety_violations.append(history_safety_violations)
                critical_violations.append(history_critical_violations)

        # Calculate average rates and confidence intervals
        total_steps = sum(len(history.history) for history in histories)
        avg_safety_violations = sum(safety_violations) / total_steps if total_steps > 0 else 0
        avg_critical_violations = sum(critical_violations) / total_steps if total_steps > 0 else 0

        # Calculate confidence intervals using bootstrap
        safety_rates = [
            v / len(history.history) for v, history in zip(safety_violations, histories)
        ]
        critical_rates = [
            v / len(history.history) for v, history in zip(critical_violations, histories)
        ]

        safety_violations_ci = confidence_interval(data=safety_rates, confidence=0.95)
        critical_violations_ci = confidence_interval(data=critical_rates, confidence=0.95)

        # Calculate confidence intervals for total violations
        total_safety_violations_ci = confidence_interval(data=safety_violations, confidence=0.95)
        total_critical_violations_ci = confidence_interval(
            data=critical_violations, confidence=0.95
        )

        return [
            MetricValue(
                name="safety_violation_rate",
                value=avg_safety_violations,
                lower_confidence_bound=safety_violations_ci[0],
                upper_confidence_bound=safety_violations_ci[1],
            ),
            MetricValue(
                name="critical_violation_rate",
                value=avg_critical_violations,
                lower_confidence_bound=critical_violations_ci[0],
                upper_confidence_bound=critical_violations_ci[1],
            ),
            MetricValue(
                name="total_safety_violations",
                value=sum(safety_violations),
                lower_confidence_bound=total_safety_violations_ci[0],
                upper_confidence_bound=total_safety_violations_ci[1],
            ),
            MetricValue(
                name="total_critical_violations",
                value=sum(critical_violations),
                lower_confidence_bound=total_critical_violations_ci[0],
                upper_confidence_bound=total_critical_violations_ci[1],
            ),
        ]

    def sample_next_step(
        self, state: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        next_state = self.state_transition_model(state=state, action=action).sample()[0]
        next_observation = self.observation_model(next_state=next_state, action=action).sample()[0]
        reward = self.reward(state=next_state, action=action)

        return next_state, next_observation, reward
