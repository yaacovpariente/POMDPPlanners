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
from POMDPPlanners.environments.safety_ant_velocity_pomdp import _native
from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_visualizer import (
    SafeAntVelocityVisualizer,
)
from POMDPPlanners.utils.statistics_utils import confidence_interval

DEFAULT_FORCE_SCALES: np.ndarray = np.array([0.0, 0.33, 0.67, 1.0])


class SafeAntVelocityPOMDPMetrics(Enum):
    """Metric names for Safety Ant Velocity POMDP environment."""

    SAFETY_VIOLATION_RATE = "safety_violation_rate"
    CRITICAL_VIOLATION_RATE = "critical_violation_rate"
    TOTAL_SAFETY_VIOLATIONS = "total_safety_violations"
    TOTAL_CRITICAL_VIOLATIONS = "total_critical_violations"


class SafeAntVelocityStateTransition(_native.SafeAntVelocityTransitionCpp):
    """Physics-based state transition model for Safety Ant Velocity POMDP.

    This model simulates simplified physics with force application, damping, and
    random force directions. The agent can choose different force magnitudes
    but cannot control the direction, creating uncertainty in the outcomes.

    Physics equations:
    - acceleration = (force - damping * velocity) / mass
    - velocity += acceleration * dt
    - position += velocity * dt

    The ``sample()`` and ``batch_sample()`` methods execute entirely in C++
    via the ``_native`` extension. ``probability()`` remains a Python
    override because the force-direction distribution is uniform on a ring
    (not Gaussian) and is evaluated via a tolerance-based consistency check
    rather than a closed-form density.

    Attributes:
        state: Current state [position_x, position_y, velocity_x, velocity_y]
        action: Force magnitude index (0=no force, 1=small, 2=medium, 3=large)
        dt: Time step for physics integration
        mass: Mass of the agent (affects acceleration)
        damping: Damping coefficient (opposes velocity)
        max_force: Maximum force magnitude
        force_scales: Force scaling factors for each action

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.safety_ant_velocity_pomdp import _native
        >>> _native.set_seed(42)
        >>>
        >>> state = np.array([0.5, -0.2, 1.0, 0.5])
        >>> transition = SafeAntVelocityStateTransition(
        ...     state=state,
        ...     action=2,
        ...     dt=0.1,
        ...     mass=1.0,
        ...     damping=0.1,
        ...     max_force=1.0,
        ... )
        >>> next_state = transition.sample()[0]
        >>> next_state.shape
        (4,)
    """

    def __init__(
        self,
        state: np.ndarray,
        action: int,
        dt: float = 0.1,
        mass: float = 1.0,
        damping: float = 0.1,
        max_force: float = 1.0,
        force_scales: Optional[np.ndarray] = None,
    ):
        effective_force_scales = (
            DEFAULT_FORCE_SCALES if force_scales is None else np.asarray(force_scales, dtype=float)
        )
        super().__init__(
            state=state,
            action=action,
            dt=dt,
            mass=mass,
            damping=damping,
            max_force=max_force,
            force_scales=effective_force_scales,
        )
        self.position = np.asarray(state, dtype=float)[:2]
        self.velocity = np.asarray(state, dtype=float)[2:4]

    def probability(self, values: List[Any]) -> np.ndarray:
        """Tolerance-based transition probability over a uniformly-random ring.

        Since the force direction is uniformly random over [-π, π], the next
        states lie on a continuous ring in state space. This method returns
        a uniform mass over the ring-consistent subset of ``values`` (and
        degenerates to a point mass when ``force_magnitude == 0``).

        Args:
            values: List of potential next states.

        Returns:
            Normalized probabilities (summing to 1 if any state is consistent).
        """
        # pylint: disable=unsubscriptable-object,invalid-unary-operand-type
        # force_scales / damping / mass / dt / max_force / action are inherited
        # from the C++ parent class; pylint does not trace the .pyi stub.
        force_magnitude = float(self.force_scales[self.action]) * self.max_force

        if force_magnitude == 0:
            acceleration = -self.damping * self.velocity / self.mass
            expected_velocity = self.velocity + acceleration * self.dt
            expected_position = self.position + expected_velocity * self.dt
            expected_state = np.concatenate([expected_position, expected_velocity])
            probs = np.array(
                [
                    1.0 if np.allclose(state, expected_state, rtol=1e-5, atol=1e-8) else 0.0
                    for state in values
                ]
            )
        else:
            position_tolerance = 0.01
            force_tolerance = 0.05

            probs_list: List[float] = []
            for next_state in values:
                next_position = next_state[:2]
                next_velocity = next_state[2:4]

                expected_position = self.position + next_velocity * self.dt
                position_error = np.linalg.norm(next_position - expected_position)

                required_force = (
                    next_velocity - self.velocity
                ) * self.mass / self.dt + self.damping * self.velocity
                required_force_magnitude = np.linalg.norm(required_force)
                force_magnitude_error = abs(required_force_magnitude - force_magnitude)

                if position_error < position_tolerance and force_magnitude_error < force_tolerance:
                    probs_list.append(1.0)
                else:
                    probs_list.append(0.0)

            probs = np.array(probs_list)

        total = np.sum(probs)
        if total > 0:
            probs = probs / total

        return probs


StateTransitionModel.register(SafeAntVelocityStateTransition)


def _build_safe_ant_obs_covariance(position_noise: float, velocity_noise: float) -> np.ndarray:
    return np.diag(
        [
            position_noise**2,
            position_noise**2,
            velocity_noise**2,
            velocity_noise**2,
        ]
    )


class SafeAntVelocityObservation(_native.SafeAntVelocityObservationCpp):
    """Noisy observation model for Safety Ant Velocity POMDP.

    This model adds Gaussian noise to both position and velocity measurements,
    creating partial observability that makes velocity estimation challenging.
    Higher noise in velocity measurements reflects the difficulty of measuring
    velocity precisely in practice.

    The ``sample()``, ``probability()``, and ``batch_log_likelihood()``
    methods execute entirely in C++ via the ``_native`` extension
    (``ObservationModelCpp<4>`` specialized with a diagonal covariance).

    Attributes:
        next_state: True state after action execution
        action: Action that was taken (not used in observation generation)
        position_noise: Standard deviation of Gaussian noise for position
        velocity_noise: Standard deviation of Gaussian noise for velocity

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.safety_ant_velocity_pomdp import _native
        >>> _native.set_seed(42)
        >>>
        >>> true_state = np.array([0.6, -0.1, 1.2, 0.8])
        >>> obs_model = SafeAntVelocityObservation(
        ...     next_state=true_state,
        ...     action=2,
        ...     position_noise=0.1,
        ...     velocity_noise=0.2,
        ... )
        >>> observation = obs_model.sample()[0]
        >>> observation.shape
        (4,)
    """

    def __init__(
        self,
        next_state: np.ndarray,
        action: int,
        position_noise: float = 0.1,
        velocity_noise: float = 0.2,
    ):
        covariance = _build_safe_ant_obs_covariance(position_noise, velocity_noise)
        super().__init__(next_state=next_state, action=action, covariance=covariance)
        self.position_noise = position_noise
        self.velocity_noise = velocity_noise
        self.position = np.asarray(next_state, dtype=float)[:2]
        self.velocity = np.asarray(next_state, dtype=float)[2:4]


ObservationModel.register(SafeAntVelocityObservation)


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

        # Cached observation covariance — built once from env params; identical
        # to what the per-call observation wrapper builds via
        # ``_build_safe_ant_obs_covariance``. Reused on the hot-path native-
        # kernel observation override to skip the per-call allocation.
        self._observation_covariance = _build_safe_ant_obs_covariance(
            position_noise, velocity_noise
        )

    def state_transition_model(self, state: np.ndarray, action: int) -> StateTransitionModel:
        return SafeAntVelocityStateTransition(  # pyright: ignore[reportReturnType]
            state=state,
            action=action,
            dt=self.dt,
            mass=self.mass,
            damping=self.damping,
            max_force=self.max_force,
        )

    def observation_model(self, next_state: np.ndarray, action: int) -> ObservationModel:
        return SafeAntVelocityObservation(  # pyright: ignore[reportReturnType]
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

    # ── Hot-path sampling overrides ─────────────────────────────────
    # The default base-class implementations build a Python wrapper
    # subclass per call (``SafeAntVelocityStateTransition`` /
    # ``SafeAntVelocityObservation``) that forwards to the native C++
    # kernel. The overrides below construct the native kernel directly
    # and skip the wrapper allocation, while preserving the identical
    # kernel-construction sequence and arguments.

    def sample_next_state(self, state: np.ndarray, action: int, n_samples: int = 1) -> Any:
        kernel = _native.SafeAntVelocityTransitionCpp(
            state=state,
            action=action,
            dt=self.dt,
            mass=self.mass,
            damping=self.damping,
            max_force=self.max_force,
            force_scales=DEFAULT_FORCE_SCALES,
        )
        samples = kernel.sample(n_samples)
        if n_samples == 1:
            return samples[0]
        return samples

    def sample_observation(self, next_state: np.ndarray, action: int, n_samples: int = 1) -> Any:
        kernel = _native.SafeAntVelocityObservationCpp(
            next_state=next_state,
            action=action,
            covariance=self._observation_covariance,
        )
        samples = kernel.sample(n_samples)
        if n_samples == 1:
            return samples[0]
        return samples

    def transition_log_probability(
        self, state: np.ndarray, action: int, next_states: Any
    ) -> np.ndarray:
        # Re-use the Python ``probability()`` defined on the wrapper subclass:
        # the force-direction distribution is uniform on a ring (no closed-form
        # density), so the wrapper's tolerance-based consistency check is the
        # canonical implementation. Wrapper construction is a thin shim over
        # ``SafeAntVelocityTransitionCpp`` and matches the env's settings.
        wrapper = SafeAntVelocityStateTransition(
            state=state,
            action=action,
            dt=self.dt,
            mass=self.mass,
            damping=self.damping,
            max_force=self.max_force,
        )
        probs = np.asarray(wrapper.probability(list(next_states)))
        return np.log(probs + 1e-300)

    def observation_log_probability(
        self, next_state: np.ndarray, action: int, observations: Any
    ) -> np.ndarray:
        kernel = _native.SafeAntVelocityObservationCpp(
            next_state=next_state,
            action=action,
            covariance=self._observation_covariance,
        )
        probs = np.asarray(kernel.probability(observations))
        return np.log(probs + 1e-300)

    def sample_next_step(
        self, state: np.ndarray, action: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        next_state = self.sample_next_state(state=state, action=action)
        next_observation = self.sample_observation(next_state=next_state, action=action)
        reward = self.reward(state=next_state, action=action)

        return next_state, next_observation, reward
