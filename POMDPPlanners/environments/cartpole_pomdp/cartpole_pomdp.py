"""CartPole POMDP Environment Implementation.

This module implements a CartPole balancing task as a POMDP, where an agent
must balance a pole on a cart using discrete left/right force actions,
with noisy observations of the cart-pole state.

The CartPole POMDP features:
- Continuous 4D state space: [cart_position, cart_velocity, pole_angle, pole_velocity]
- Discrete binary action space: [left_force, right_force]
- Noisy continuous observations of the state
- Physics-based dynamics simulation
- Episode termination when pole falls beyond threshold or cart moves too far

Classes:
    CartPolePOMDP: Main CartPole environment with POMDP formulation
"""

import math
from enum import Enum
from pathlib import Path
from collections.abc import Hashable
from typing import Any, Dict, List, Optional, Sequence, Union

_INTEGRATOR_CODES: dict[str, int] = {"euler": 0, "semi-implicit euler": 1}

import numpy as np
from numpy.typing import NDArray

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.simulation import History, MetricValue
from POMDPPlanners.environments.cartpole_pomdp import _native
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal
from POMDPPlanners.utils.statistics_utils import confidence_interval


class CartPolePOMDPMetrics(Enum):
    """Metric names for CartPole POMDP environment."""

    GOAL_REACHING_RATE = "goal_reaching_rate"


class CartPoleInitialStateDistribution(Distribution):
    """Initial state distribution for CartPole POMDP.

    This distribution generates random initial states for the cart-pole system
    by sampling uniformly from a small range around the equilibrium position.
    All state variables (position, velocity, angle, angular velocity) are
    initialized close to zero with small random perturbations.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> # Create initial state distribution
        >>> initial_dist = CartPoleInitialStateDistribution()

        >>> # Sample initial state
        >>> initial_state = initial_dist.sample()[0]
        >>> len(initial_state) == 4
        True
        >>> all(-0.05 <= x <= 0.05 for x in initial_state)  # Values in valid range
        True

        >>> # Sample multiple initial states
        >>> states = initial_dist.sample(n_samples=3)
        >>> len(states) == 3
        True
        >>> all(len(state) == 4 for state in states)
        True

        >>> # Each state has 4 components: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
        >>> position, velocity, angle, angular_velocity = initial_state
        >>> isinstance(position, (int, float, np.floating))
        True
    """

    def __init__(self):
        pass

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        # Generate all samples at once using vectorized uniform distribution
        samples_array = np.random.uniform(low=-0.05, high=0.05, size=(n_samples, 4))
        return list(samples_array)


class CartPolePOMDP(DiscreteActionsEnvironment):
    """CartPole balancing task formulated as a POMDP.

    This environment simulates the classic cart-pole balancing problem where an agent
    must apply left or right forces to keep a pole balanced on a moving cart.
    The challenge comes from noisy observations of the cart-pole state.

    Problem Structure:
    - State: [cart_position, cart_velocity, pole_angle, pole_velocity] (continuous)
    - Actions: [left_force, right_force] (discrete)
    - Observations: Noisy state measurements (continuous)
    - Rewards: +1.0 per time step alive, 0.0 when terminated
    - Termination: Pole falls beyond angle threshold or cart moves too far

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Initialize environment
        >>> noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
        >>> env = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov)
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

    DEFAULT_STATE_TRANSITION_COV = np.diag([1e-4, 1e-4, 2.5e-5, 1e-4])

    def __init__(
        self,
        discount_factor: float,
        noise_cov: NDArray[np.floating[Any]],
        state_transition_cov: Optional[NDArray[np.floating[Any]]] = None,
        name: str = "CartPolePOMDP",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        # Set all configuration parameters first
        self.noise_cov = noise_cov
        self._obs_dist = CovarianceParameterizedMultivariateNormal(noise_cov)
        self.state_transition_cov = (
            state_transition_cov
            if state_transition_cov is not None
            else self.DEFAULT_STATE_TRANSITION_COV
        )
        self._state_transition_dist = CovarianceParameterizedMultivariateNormal(
            self.state_transition_cov
        )
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        self._kinematics_integrator_int: int = _INTEGRATOR_CODES[self.kinematics_integrator]
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        # Create space info
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,  # Binary action space
            observation_space=SpaceType.CONTINUOUS,  # Continuous state space
        )

        # Call parent's __init__ last, which will generate the config_id
        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=(0.0, 1.0),
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        # Per-action C++ kernel caches: actions are ``int`` (0 / 1) so a
        # plain ``Dict[int, Any]`` suffices. Lazily built by
        # ``_get_trans_kernel`` / ``_get_obs_kernel`` and reset on
        # unpickle. Mirrors the RockSample / Pacman pattern.
        self._trans_kernel_cache: Dict[int, Any] = {}
        self._obs_kernel_cache: Dict[int, Any] = {}

    # ── Native-backed env-API implementations ────────────────────────
    # Each method fetches a cached per-action C++ kernel, mutates its
    # stored state via ``set_state`` / ``set_next_state`` (when needed),
    # and dispatches to the same native sample / probability /
    # batch_sample / batch_log_likelihood entry points as before. The
    # kernel itself caches frozen physics params, action int, and
    # covariance so we no longer rebuild those per call.

    def _get_trans_kernel(self, action: int) -> Any:
        kernel = self._trans_kernel_cache.get(action)
        if kernel is None:
            placeholder = np.zeros(4, dtype=np.float64)
            kernel = _native.CartPoleTransitionCpp(
                state=placeholder,
                action=int(action),
                force_mag=self.force_mag,
                total_mass=self.total_mass,
                polemass_length=self.polemass_length,
                gravity=self.gravity,
                length=self.length,
                kinematics_integrator=self.kinematics_integrator,
                tau=self.tau,
                masspole=self.masspole,
                covariance=self._state_transition_dist.covariance,
            )
            self._trans_kernel_cache[action] = kernel
        return kernel

    def _get_obs_kernel(self, action: int) -> Any:
        kernel = self._obs_kernel_cache.get(action)
        if kernel is None:
            placeholder = np.zeros(4, dtype=np.float64)
            kernel = _native.CartPoleObservationCpp(
                next_state=placeholder,
                action=int(action),
                covariance=self._obs_dist.covariance,
            )
            self._obs_kernel_cache[action] = kernel
        return kernel

    def sample_next_state(
        self, state: np.ndarray, action: int, n_samples: int = 1
    ) -> NDArray[np.float64]:
        kernel = self._get_trans_kernel(int(action))
        kernel.set_state(np.asarray(state, dtype=np.float64))
        samples = kernel.sample(n_samples)
        if n_samples == 1:
            return samples[0]
        return np.asarray(samples)

    def sample_observation(
        self, next_state: np.ndarray, action: int, n_samples: int = 1
    ) -> NDArray[np.float64]:
        kernel = self._get_obs_kernel(int(action))
        kernel.set_next_state(np.asarray(next_state, dtype=np.float64))
        samples = kernel.sample(n_samples)
        if n_samples == 1:
            return samples[0]
        return np.asarray(samples)

    def transition_log_probability(
        self,
        state: np.ndarray,
        action: int,
        next_states: Union[Sequence[Any], np.ndarray],
    ) -> np.ndarray:
        kernel = self._get_trans_kernel(int(action))
        kernel.set_state(np.asarray(state, dtype=np.float64))
        probs = np.asarray(kernel.probability(next_states))
        return np.log(probs + 1e-300)

    def observation_log_probability(
        self,
        next_state: np.ndarray,
        action: int,
        observations: Union[Sequence[Any], np.ndarray],
    ) -> np.ndarray:
        kernel = self._get_obs_kernel(int(action))
        kernel.set_next_state(np.asarray(next_state, dtype=np.float64))
        probs = np.asarray(kernel.probability(observations))
        return np.log(probs + 1e-300)

    def sample_next_state_batch(self, states: Any, action: int) -> np.ndarray:
        states_array = np.ascontiguousarray(np.asarray(states, dtype=np.float64))
        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)
        # ``batch_sample`` reads each row's state from the input array;
        # the kernel's stored ``state_`` is not consulted, so we skip
        # ``set_state`` on this hot path.
        kernel = self._get_trans_kernel(int(action))
        return np.asarray(kernel.batch_sample(states_array), dtype=np.float64)

    def observation_log_probability_per_state(
        self, next_states: Any, action: int, observation: Any
    ) -> np.ndarray:
        next_states_array = np.ascontiguousarray(np.asarray(next_states, dtype=np.float64))
        if next_states_array.ndim == 1:
            next_states_array = next_states_array.reshape(1, -1)
        observation_array = np.ascontiguousarray(np.asarray(observation, dtype=np.float64))
        # ``batch_log_likelihood`` reads each row's next-state from the
        # input array; ``next_state_`` on the kernel is unused, so we
        # skip ``set_next_state`` on this hot path.
        kernel = self._get_obs_kernel(int(action))
        return np.asarray(
            kernel.batch_log_likelihood(
                next_particles=next_states_array,
                observation=observation_array,
            ),
            dtype=np.float64,
        )

    def __getstate__(self) -> Dict[str, Any]:
        # Per-action C++ kernel caches hold pybind11 objects that aren't
        # picklable. Drop them at serialization time; ``__setstate__``
        # rebuilds empty caches so the env works after unpickling.
        state = self.__dict__.copy()
        state["_trans_kernel_cache"] = {}
        state["_obs_kernel_cache"] = {}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        vars(self).update(state)
        self._trans_kernel_cache = {}
        self._obs_kernel_cache = {}

    def reward(self, state: np.ndarray, action: int) -> float:
        terminated = self.is_terminal(state)

        if not terminated:
            reward = 1.0
        else:
            reward = 0.0

        return reward

    def reward_batch(self, states: Union[np.ndarray, Sequence[Any]], action: int) -> np.ndarray:
        states_arr = np.asarray(states)
        x = states_arr[:, 0]
        theta = states_arr[:, 2]
        terminated = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )
        return np.where(terminated, 0.0, 1.0)

    def is_terminal(self, state: np.ndarray) -> bool:
        x, theta = state[0], state[2]

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        return terminated

    def simulate_random_rollout(  # pylint: disable=unused-argument
        self,
        state: Any,
        action_sampler: Any,
        max_depth: int,
        discount_factor: float,
        depth: int = 0,
    ) -> float:
        """Random rollout via native C++.

        Args:
            state: Current 4-D cart-pole state ``[x, x_dot, theta, theta_dot]``.
            action_sampler: Object with a ``sample()`` method (used only on the
                Python fallback path).
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
        action_indices = np.random.randint(0, 2, size=steps_left, dtype=np.int32)

        return _native.simulate_rollout(
            initial_state=state_arr,
            action_indices=action_indices,
            max_depth=max_depth,
            start_depth=depth,
            discount_factor=discount_factor,
            force_mag=self.force_mag,
            total_mass=self.total_mass,
            polemass_length=self.polemass_length,
            gravity=self.gravity,
            length=self.length,
            kinematics_integrator=self._kinematics_integrator_int,
            tau=self.tau,
            masspole=self.masspole,
            x_threshold=self.x_threshold,
            theta_threshold=self.theta_threshold_radians,
            covariance=self._state_transition_dist.covariance,
        )

    def initial_state_dist(self) -> Distribution:
        return CartPoleInitialStateDistribution()

    def initial_observation_dist(self) -> Distribution:
        return CartPoleInitialStateDistribution()

    def get_actions(self) -> List[int]:
        return [0, 1]

    def is_equal_observation(self, observation1: np.ndarray, observation2: np.ndarray) -> bool:
        return np.array_equal(observation1, observation2)

    def hash_action(self, action: Any) -> Hashable:
        # Discrete int actions (0, 1); already hashable.
        return action

    def get_metric_names(self) -> List[str]:
        """Get names of CartPole POMDP specific metrics.

        Returns:
            List containing metric names: goal_reaching_rate
        """
        return [metric.value for metric in CartPolePOMDPMetrics]

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        """Compute CartPole POMDP specific metrics from simulation histories.

        Args:
            histories: List of simulation histories

        Returns:
            List of MetricValue objects containing the computed metrics
        """
        goal_reached = []
        for history in histories:
            goal_reached_in_history = True  # Goal is reached if episode didn't crash
            for step in history.history:
                if self.is_terminal(step.state):
                    goal_reached_in_history = False
                    break
            goal_reached.append(1 if goal_reached_in_history else 0)

        avg_goal_reached = float(np.mean(goal_reached))
        goal_reached_ci = confidence_interval(data=goal_reached, confidence=0.95)

        return [
            MetricValue(
                name=CartPolePOMDPMetrics.GOAL_REACHING_RATE.value,
                value=avg_goal_reached,
                lower_confidence_bound=goal_reached_ci[0],
                upper_confidence_bound=goal_reached_ci[1],
            ),
        ]
