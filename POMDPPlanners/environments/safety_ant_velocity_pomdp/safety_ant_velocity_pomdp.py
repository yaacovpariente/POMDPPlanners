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

from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    ObservationModel,
    SpaceInfo,
    SpaceType,
    StateTransitionModel,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_visualizer import (
    SafeAntVelocityVisualizer,
)
from POMDPPlanners.utils.statistics_utils import confidence_interval


class SafeAntVelocityPOMDPMetrics(Enum):
    """Metric names for Safety Ant Velocity POMDP environment."""

    SAFETY_VIOLATION_RATE = "safety_violation_rate"
    CRITICAL_VIOLATION_RATE = "critical_violation_rate"
    TOTAL_SAFETY_VIOLATIONS = "total_safety_violations"
    TOTAL_CRITICAL_VIOLATIONS = "total_critical_violations"


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

    def probability(self, values: List[Any]) -> np.ndarray:
        """Calculate transition probabilities for given next states.

        Since the force direction is uniformly random over [-π, π], the probability
        distribution is continuous and depends on the distance from expected dynamics.
        We approximate this using a mixture of Gaussians representing the random
        force direction uncertainty.

        Args:
            values: List of potential next states

        Returns:
            Array of (unnormalized) probabilities for each state
        """
        # Get force magnitude from action
        force_magnitude = self.force_scales[self.action] * self.max_force

        if force_magnitude == 0:
            # No force applied - deterministic transition
            # Only damping affects the dynamics
            acceleration = -self.damping * self.velocity / self.mass
            expected_velocity = self.velocity + acceleration * self.dt
            expected_position = self.position + expected_velocity * self.dt
            expected_state = np.concatenate([expected_position, expected_velocity])

            # For zero force, transition is deterministic
            probs = np.array(
                [
                    1.0 if np.allclose(state, expected_state, rtol=1e-5, atol=1e-8) else 0.0
                    for state in values
                ]
            )
        else:
            # Force is applied with random direction uniformly sampled from [-π, π]
            # The resulting next states form a continuous distribution on a ring
            # All states consistent with the force magnitude have equal probability density

            # Tolerance for checking consistency
            position_tolerance = 0.01
            force_tolerance = 0.05

            probs = []
            for next_state in values:
                # Extract components from the next state
                next_position = next_state[:2]
                next_velocity = next_state[2:4]

                # Check if position is consistent with velocity
                # next_position = position + next_velocity * dt
                expected_position = self.position + next_velocity * self.dt
                position_error = np.linalg.norm(next_position - expected_position)

                # Check if velocity is consistent with some force direction
                # next_velocity = velocity + (force - damping * velocity) / mass * dt
                # Solving for force: force = (next_velocity - velocity) * mass / dt + damping * velocity
                required_force = (
                    next_velocity - self.velocity
                ) * self.mass / self.dt + self.damping * self.velocity
                required_force_magnitude = np.linalg.norm(required_force)
                force_magnitude_error = abs(required_force_magnitude - force_magnitude)

                # Assign uniform probability if state is consistent with physics
                if position_error < position_tolerance and force_magnitude_error < force_tolerance:
                    prob = 1.0  # Uniform over consistent states
                else:
                    prob = 0.0  # Zero for inconsistent states

                probs.append(prob)

            probs = np.array(probs)

        # Normalize probabilities
        total = np.sum(probs)
        if total > 0:
            probs = probs / total

        return probs


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

    def probability(self, values: List[Any]) -> np.ndarray:
        # Calculate probabilities based on Gaussian noise model for list of observations
        probabilities = []
        for observation in values:
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
        >>>
        >>> # Initialize environment
        >>> env = SafeAntVelocityPOMDP(discount_factor=0.99)
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

        return float(reward)

    def reward_batch(self, states: Union[np.ndarray, Sequence[Any]], action: int) -> np.ndarray:
        states_arr = np.asarray(states)
        speeds = np.linalg.norm(states_arr[:, 2:4], axis=1)
        rewards = speeds * self.movement_reward_scale
        rewards[speeds > self.safe_velocity_threshold] += self.safety_violation_penalty
        return rewards

    def is_terminal(self, state: np.ndarray) -> bool:
        # Episode ends if velocity exceeds safe threshold by too much
        velocity = state[2:4]
        speed = np.linalg.norm(velocity)
        return bool(speed > self.safe_velocity_threshold * 1.5)  # 50% margin

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
        visualizer = SafeAntVelocityVisualizer(self)
        visualizer.create_animation(history, cache_path)

    def get_metric_names(self) -> List[str]:
        """Get names of Safety Ant Velocity POMDP specific metrics.

        Returns:
            List containing metric names: safety_violation_rate, critical_violation_rate,
            total_safety_violations, and total_critical_violations
        """
        return [metric.value for metric in SafeAntVelocityPOMDPMetrics]

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
                name=SafeAntVelocityPOMDPMetrics.SAFETY_VIOLATION_RATE.value,
                value=avg_safety_violations,
                lower_confidence_bound=safety_violations_ci[0],
                upper_confidence_bound=safety_violations_ci[1],
            ),
            MetricValue(
                name=SafeAntVelocityPOMDPMetrics.CRITICAL_VIOLATION_RATE.value,
                value=avg_critical_violations,
                lower_confidence_bound=critical_violations_ci[0],
                upper_confidence_bound=critical_violations_ci[1],
            ),
            MetricValue(
                name=SafeAntVelocityPOMDPMetrics.TOTAL_SAFETY_VIOLATIONS.value,
                value=sum(safety_violations),
                lower_confidence_bound=total_safety_violations_ci[0],
                upper_confidence_bound=total_safety_violations_ci[1],
            ),
            MetricValue(
                name=SafeAntVelocityPOMDPMetrics.TOTAL_CRITICAL_VIOLATIONS.value,
                value=sum(critical_violations),
                lower_confidence_bound=total_critical_violations_ci[0],
                upper_confidence_bound=total_critical_violations_ci[1],
            ),
        ]

    def sample_next_step(
        self, state: np.ndarray, action: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        next_state = self.state_transition_model(state=state, action=action).sample()[0]
        next_observation = self.observation_model(next_state=next_state, action=action).sample()[0]
        reward = self.reward(state=next_state, action=action)

        return next_state, next_observation, reward
