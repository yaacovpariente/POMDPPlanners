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
    SafeAntVelocityPOMDP: Main safety-critical velocity control environment
"""

from enum import Enum
from math import hypot
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
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


def _build_safe_ant_obs_covariance(position_noise: float, velocity_noise: float) -> np.ndarray:
    return np.diag(
        [
            position_noise**2,
            position_noise**2,
            velocity_noise**2,
            velocity_noise**2,
        ]
    )


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

        # Cached force scales as a contiguous float64 array for the native
        # simulate_rollout kernel.
        self._force_scales_f64: np.ndarray = np.ascontiguousarray(
            DEFAULT_FORCE_SCALES, dtype=np.float64
        )

        # Per-action native-kernel caches. The transition kernel only depends on
        # ``(action, dt, mass, damping, max_force, force_scales)``; the
        # observation kernel only depends on ``(action, covariance)``. The
        # state / next_state argument is a per-call mutable. We construct one
        # kernel per discrete action on first use and reuse it across
        # subsequent calls via ``set_state`` / ``set_next_state`` -- this
        # collapses the 5x per-call ``numpy.asarray`` + Cholesky + GIL
        # round-trip overhead that showed up in cProfile.
        self._trans_kernel_cache: Dict[int, Any] = {}
        self._obs_kernel_cache: Dict[int, Any] = {}

    def __getstate__(self) -> dict:
        # pybind11 kernels are not picklable; rebuild lazily on the receiver.
        state = self.__dict__.copy()
        state["_trans_kernel_cache"] = {}
        state["_obs_kernel_cache"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        vars(self).update(state)
        self._trans_kernel_cache = {}
        self._obs_kernel_cache = {}

    def _get_trans_kernel(self, action: int) -> Any:
        kernel = self._trans_kernel_cache.get(int(action))
        if kernel is None:
            kernel = _native.SafeAntVelocityTransitionCpp(
                state=np.zeros(4, dtype=np.float64),
                action=int(action),
                dt=self.dt,
                mass=self.mass,
                damping=self.damping,
                max_force=self.max_force,
                force_scales=DEFAULT_FORCE_SCALES,
            )
            self._trans_kernel_cache[int(action)] = kernel
        return kernel

    def _get_obs_kernel(self, action: int) -> Any:
        kernel = self._obs_kernel_cache.get(int(action))
        if kernel is None:
            kernel = _native.SafeAntVelocityObservationCpp(
                next_state=np.zeros(4, dtype=np.float64),
                action=int(action),
                covariance=self._observation_covariance,
            )
            self._obs_kernel_cache[int(action)] = kernel
        return kernel

    def reward(self, state: np.ndarray, action: int) -> float:
        # ``hypot`` outperforms ``numpy.linalg.norm`` for fixed 2-D inputs:
        # cProfile attributed ~25% of POMCPOW wall time to the norm calls
        # in this method plus ``is_terminal``.
        speed = hypot(float(state[2]), float(state[3]))
        reward = speed * self.movement_reward_scale
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
        # Episode ends if velocity exceeds safe threshold by too much.
        # ``hypot`` is significantly faster than ``numpy.linalg.norm`` on
        # 2-element input — see ``reward`` for the same optimisation.
        speed = hypot(float(state[2]), float(state[3]))
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
                # Get velocity from state — ``hypot`` is used for the same
                # reason as in ``reward`` / ``is_terminal``.
                speed = hypot(float(step.state[2]), float(step.state[3]))

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

    # ── Env-API sampling and density methods ─────────────────────────
    # These methods construct the native C++ kernel directly and
    # provide the canonical sampling/density entry points used
    # throughout the simulator and belief-update paths.

    def sample_next_state(self, state: np.ndarray, action: int, n_samples: int = 1) -> Any:
        kernel = self._get_trans_kernel(action)
        kernel.set_state(state)
        samples = kernel.sample(n_samples)
        if n_samples == 1:
            return samples[0]
        return samples

    def sample_observation(self, next_state: np.ndarray, action: int, n_samples: int = 1) -> Any:
        kernel = self._get_obs_kernel(action)
        kernel.set_next_state(next_state)
        samples = kernel.sample(n_samples)
        if n_samples == 1:
            return samples[0]
        return samples

    def transition_log_probability(
        self, state: np.ndarray, action: int, next_states: Any
    ) -> np.ndarray:
        # The force-direction distribution is uniform on a ring (no closed-form
        # density). We use a tolerance-based consistency check: each candidate
        # next state either matches the deterministic damped-integration step
        # (zero-force regime) or lies on the action's force-magnitude ring
        # within position/force tolerances (non-zero-force regime).
        probs = np.asarray(self._transition_probability(state, action, list(next_states)))
        return np.log(probs + 1e-300)

    def _transition_probability(
        self, state: np.ndarray, action: int, values: List[Any]
    ) -> np.ndarray:
        position = np.asarray(state, dtype=float)[:2]
        velocity = np.asarray(state, dtype=float)[2:4]
        force_magnitude = float(DEFAULT_FORCE_SCALES[action]) * self.max_force

        if force_magnitude == 0:
            acceleration = -self.damping * velocity / self.mass
            expected_velocity = velocity + acceleration * self.dt
            expected_position = position + expected_velocity * self.dt
            expected_state = np.concatenate([expected_position, expected_velocity])
            probs = np.array(
                [
                    1.0 if np.allclose(s, expected_state, rtol=1e-5, atol=1e-8) else 0.0
                    for s in values
                ]
            )
        else:
            position_tolerance = 0.01
            force_tolerance = 0.05

            probs_list: List[float] = []
            for next_state in values:
                next_position = next_state[:2]
                next_velocity = next_state[2:4]

                expected_position = position + next_velocity * self.dt
                position_error = np.linalg.norm(next_position - expected_position)

                required_force = (
                    next_velocity - velocity
                ) * self.mass / self.dt + self.damping * velocity
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

    def observation_log_probability(
        self, next_state: np.ndarray, action: int, observations: Any
    ) -> np.ndarray:
        kernel = self._get_obs_kernel(action)
        kernel.set_next_state(next_state)
        probs = np.asarray(kernel.probability(observations))
        return np.log(probs + 1e-300)

    def sample_next_state_batch(self, states: Any, action: int) -> np.ndarray:
        states_array = np.ascontiguousarray(np.asarray(states, dtype=np.float64))
        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)
        # batch_sample reads the per-row state from ``states_array`` and never
        # touches the kernel's stored state, so no set_state is needed here.
        kernel = self._get_trans_kernel(action)
        return np.asarray(kernel.batch_sample(states_array), dtype=np.float64)

    def observation_log_probability_per_state(
        self, next_states: Any, action: int, observation: Any
    ) -> np.ndarray:
        next_states_array = np.ascontiguousarray(np.asarray(next_states, dtype=np.float64))
        if next_states_array.ndim == 1:
            next_states_array = next_states_array.reshape(1, -1)
        observation_array = np.ascontiguousarray(np.asarray(observation, dtype=np.float64))
        # batch_log_likelihood reads per-row from ``next_states_array``; no
        # set_next_state needed.
        kernel = self._get_obs_kernel(action)
        return np.asarray(
            kernel.batch_log_likelihood(
                next_particles=next_states_array,
                observation=observation_array,
            ),
            dtype=np.float64,
        )

    def simulate_random_rollout(  # pylint: disable=unused-argument
        self,
        state: Any,
        action_sampler: Any,
        max_depth: int,
        discount_factor: float,
        depth: int = 0,
    ) -> float:
        """Random rollout via native C++ physics and reward kernel.

        Pre-draws action indices and runs the full rollout in a single C++
        call, avoiding per-step Python dispatch.

        Args:
            state: Current 4-D state ``[px, py, vx, vy]``.
            action_sampler: Accepted for interface compatibility with the
                base ``simulate_random_rollout`` signature; the native
                rollout draws action indices via ``np.random.randint``
                directly and never invokes the sampler.
            max_depth: Maximum rollout depth.
            discount_factor: Per-step discount factor.
            depth: Depth already consumed by the search tree. Defaults to 0.

        Returns:
            Discounted sum of immediate rewards along the sampled trajectory.
        """
        steps_left = max_depth - depth
        if steps_left <= 0:
            return 0.0

        state_arr = np.ascontiguousarray(np.asarray(state, dtype=np.float64).ravel())
        action_indices = np.random.randint(0, len(self.actions), size=steps_left, dtype=np.int32)

        return _native.simulate_rollout(
            initial_state=state_arr,
            action_indices=action_indices,
            force_scales=self._force_scales_f64,
            max_depth=max_depth,
            start_depth=depth,
            discount_factor=discount_factor,
            dt=float(self.dt),
            mass=float(self.mass),
            damping=float(self.damping),
            max_force=float(self.max_force),
            safe_velocity_threshold=float(self.safe_velocity_threshold),
            safety_violation_penalty=float(self.safety_violation_penalty),
            movement_reward_scale=float(self.movement_reward_scale),
        )

    def sample_next_step(
        self, state: np.ndarray, action: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        next_state = self.sample_next_state(state=state, action=action)
        next_observation = self.sample_observation(next_state=next_state, action=action)
        reward = self.reward(state=next_state, action=action)

        return next_state, next_observation, reward
