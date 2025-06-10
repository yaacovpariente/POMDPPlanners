from typing import List, Any, Tuple, Optional
import numpy as np
from pathlib import Path

from POMDPPlanners.core.environment import (
    ObservationModel,
    StateTransitionModel,
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType
)
from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.simulation import History, StepData, MetricValue
from POMDPPlanners.utils.statistics import confidence_interval


class SafeAntVelocityStateTransition(StateTransitionModel):
    def __init__(
        self,
        state: np.ndarray,
        action: int,
        dt: float = 0.1,
        mass: float = 1.0,
        damping: float = 0.1,
        max_force: float = 1.0,
    ):
        """
        State: [position_x, position_y, velocity_x, velocity_y]
        Action: Force magnitude index (0: no force, 1: small force, 2: medium force, 3: large force)
        """
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

    def sample(self) -> np.ndarray:
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
        return next_state


class SafeAntVelocityObservation(ObservationModel):
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

    def sample(self) -> np.ndarray:
        # Add noise to position and velocity observations
        noisy_position = self.position + np.random.normal(0, self.position_noise, size=2)
        noisy_velocity = self.velocity + np.random.normal(0, self.velocity_noise, size=2)
        
        # Combine observations
        observation = np.concatenate([noisy_position, noisy_velocity])
        return observation

    def probability(self, observation: np.ndarray) -> float:
        # Calculate probability based on Gaussian noise model
        position_diff = observation[:2] - self.position
        velocity_diff = observation[2:4] - self.velocity
        
        position_log_prob = -0.5 * np.sum(position_diff**2) / (self.position_noise**2)
        velocity_log_prob = -0.5 * np.sum(velocity_diff**2) / (self.velocity_noise**2)
        
        return np.exp(position_log_prob + velocity_log_prob)


class SafeAntVelocityPOMDP(DiscreteActionsEnvironment):
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

        # Create space info with appropriate bounds
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,  # Action space is discrete force magnitudes
            observation_space=SpaceType.CONTINUOUS  # Observation space is positions and velocities with noise
        )
        super().__init__(discount_factor=discount_factor, name=name, space_info=space_info, output_dir=output_dir, debug=debug)

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
            def sample(self) -> np.ndarray:
                # Start with zero velocity and random position
                position = np.random.uniform(-1, 1, size=2)
                velocity = np.zeros(2)
                return np.concatenate([position, velocity])
        
        return InitialState()

    def initial_observation_dist(self) -> Distribution:
        return self.initial_state_dist()

    def get_actions(self) -> List[int]:
        return self.actions

    def is_equal_observation(self, observation1: np.ndarray, observation2: np.ndarray) -> bool:
        return np.array_equal(observation1, observation2)

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        # TODO: Implement visualization
        pass

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
        safety_rates = [v / len(history.history) for v, history in zip(safety_violations, histories)]
        critical_rates = [v / len(history.history) for v, history in zip(critical_violations, histories)]
        
        safety_violations_ci = confidence_interval(data=safety_rates, confidence=0.95)
        critical_violations_ci = confidence_interval(data=critical_rates, confidence=0.95)
        
        return [
            MetricValue(
                name="safety_violation_rate",
                value=avg_safety_violations,
                lower_confidence_bound=safety_violations_ci[0],
                upper_confidence_bound=safety_violations_ci[1]
            ),
            MetricValue(
                name="critical_violation_rate",
                value=avg_critical_violations,
                lower_confidence_bound=critical_violations_ci[0],
                upper_confidence_bound=critical_violations_ci[1]
            )
        ]

    def sample_next_step(self, state: Any, action: Any) -> Tuple[Any, Any, float]:
        next_state = self.state_transition_model(state=state, action=action).sample()
        next_observation = self.observation_model(
            next_state=next_state, action=action
        ).sample()
        reward = self.reward(state=next_state, action=action)

        return next_state, next_observation, reward 