"""Mountain Car POMDP Environment Implementation.

This module implements the classic Mountain Car problem as a POMDP, where an
agent must drive an underpowered car up a steep mountain by building momentum
through oscillating motion, with noisy observations of the car's state.

The Mountain Car POMDP features:
- Continuous 2D state space: [position, velocity]
- Discrete action space: [-1 (reverse), 0 (neutral), 1 (forward)]
- Noisy continuous observations of position and velocity
- Physics-based dynamics with gravity and momentum
- Sparse reward: 0 for reaching goal, -1 per time step otherwise

The key challenge is that the car's engine is too weak to drive directly up
the mountain, so the agent must learn to build momentum by first moving away
from the goal.

Classes:
    MountainCarPOMDP: Main Mountain Car environment with POMDP formulation
"""

from enum import Enum
from pathlib import Path
from collections.abc import Hashable
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np
from numpy.typing import NDArray

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.environments.mountain_car_pomdp import _native
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal
from POMDPPlanners.utils.statistics_utils import confidence_interval

matplotlib.use("Agg")  # Use non-interactive backend


class MountainCarPOMDPMetrics(Enum):
    """Metric names for Mountain Car POMDP environment."""

    GOAL_REACHING_RATE = "goal_reaching_rate"


class MountainCarPOMDP(DiscreteActionsEnvironment):
    """Mountain Car problem formulated as a POMDP.

    This environment simulates an underpowered car trying to reach the top of
    a steep mountain. The car must build momentum by oscillating back and forth
    to gain enough energy to reach the goal, with noisy observations of its state.

    Problem Structure:
    - State: [position, velocity] (continuous, position ∈ [-1.2, 0.6], velocity ∈ [-0.07, 0.07])
    - Actions: [-1 (reverse), 0 (neutral), 1 (forward)] (discrete)
    - Observations: Noisy state measurements (continuous)
    - Rewards: 0 for reaching goal (position ≥ 0.5), -1 per time step otherwise
    - Goal: Drive car to position ≥ 0.5 (top of mountain)

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Initialize environment
        >>> env = MountainCarPOMDP(discount_factor=0.99)
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

    DEFAULT_STATE_TRANSITION_COV = np.diag([2.5e-5, 1e-6])

    def __init__(
        self,
        discount_factor: float,
        state_transition_cov: Optional[NDArray[np.floating[Any]]] = None,
        name: str = "MountainCarPOMDP",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.power = 0.001
        self.gravity = 0.0025

        # Define actions: -1 (left), 0 (no acceleration), 1 (right)
        self.actions = [-1, 0, 1]
        self._actions_int32: np.ndarray = np.ascontiguousarray(
            np.array(self.actions, dtype=np.int32)
        )

        # Define observation noise parameters
        self.position_noise = 0.1
        self.velocity_noise = 0.01

        # Observation noise matrix
        self.cov_matrix = np.array([[self.position_noise**2, 0], [0, self.velocity_noise**2]])

        # Pre-compute Cholesky decomposition for efficient sampling and PDF
        self._obs_dist = CovarianceParameterizedMultivariateNormal(self.cov_matrix)

        # State transition noise
        self.state_transition_cov = (
            state_transition_cov
            if state_transition_cov is not None
            else self.DEFAULT_STATE_TRANSITION_COV
        )
        self._state_transition_dist = CovarianceParameterizedMultivariateNormal(
            self.state_transition_cov
        )

        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,  # Action space is [-1, 0, 1]
            observation_space=SpaceType.CONTINUOUS,  # Observation space is position and velocity
        )
        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=(-1.0, 0.0),
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        # Per-action C++ kernel caches: actions are ``int`` so a plain
        # ``Dict[int, Any]`` suffices. Lazily built by ``_get_trans_kernel``
        # / ``_get_obs_kernel`` and reset on unpickle. The kernel keeps the
        # frozen env params (action, power, gravity, max_speed,
        # min_position, max_position, covariance) and is mutated only via
        # ``set_state`` / ``set_next_state`` on hot paths.
        self._trans_kernel_cache: Dict[int, Any] = {}
        self._obs_kernel_cache: Dict[int, Any] = {}

    def __getstate__(self) -> Dict[str, Any]:
        # pybind11 kernels are not picklable; rebuild lazily on the receiver.
        state = self.__dict__.copy()
        state["_trans_kernel_cache"] = {}
        state["_obs_kernel_cache"] = {}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        vars(self).update(state)
        self._trans_kernel_cache = {}
        self._obs_kernel_cache = {}

    def _get_trans_kernel(self, action: int) -> Any:
        kernel = self._trans_kernel_cache.get(int(action))
        if kernel is None:
            kernel = _native.MountainCarTransitionCpp(
                state=np.zeros(2, dtype=np.float64),
                action=int(action),
                power=self.power,
                gravity=self.gravity,
                max_speed=self.max_speed,
                min_position=self.min_position,
                max_position=self.max_position,
                covariance=self._state_transition_dist.covariance,
            )
            self._trans_kernel_cache[int(action)] = kernel
        return kernel

    def _get_obs_kernel(self, action: int) -> Any:
        kernel = self._obs_kernel_cache.get(int(action))
        if kernel is None:
            kernel = _native.MountainCarObservationCpp(
                next_state=np.zeros(2, dtype=np.float64),
                action=int(action),
                covariance=self._obs_dist.covariance,
            )
            self._obs_kernel_cache[int(action)] = kernel
        return kernel

    # ── Hot-path sampling overrides ─────────────────────────────────
    # Each method fetches a cached per-action C++ kernel, mutates its
    # stored state via ``set_state`` / ``set_next_state`` (when needed),
    # and dispatches to the same native sample / probability /
    # batch_sample / batch_log_likelihood entry points as before.

    def sample_next_state(
        self,
        state: Union[Tuple[float, float], NDArray[np.float64]],
        action: int,
        n_samples: int = 1,
    ) -> NDArray[np.float64]:
        kernel = self._get_trans_kernel(int(action))
        kernel.set_state(state)
        samples = kernel.sample(n_samples)
        if n_samples == 1:
            return samples[0]
        return np.asarray(samples)

    def sample_observation(
        self,
        next_state: Union[Tuple[float, float], NDArray[np.float64]],
        action: int,
        n_samples: int = 1,
    ) -> NDArray[np.float64]:
        kernel = self._get_obs_kernel(int(action))
        kernel.set_next_state(next_state)
        samples = kernel.sample(n_samples)
        if n_samples == 1:
            return samples[0]
        return np.asarray(samples)

    def transition_log_probability(
        self,
        state: Union[Tuple[float, float], NDArray[np.float64]],
        action: int,
        next_states: Union[Sequence[Any], NDArray[np.float64]],
    ) -> np.ndarray:
        kernel = self._get_trans_kernel(int(action))
        kernel.set_state(state)
        probs = np.asarray(kernel.probability(next_states))
        return np.log(probs + 1e-300)

    def observation_log_probability(
        self,
        next_state: Union[Tuple[float, float], NDArray[np.float64]],
        action: int,
        observations: Union[Sequence[Any], NDArray[np.float64]],
    ) -> np.ndarray:
        kernel = self._get_obs_kernel(int(action))
        kernel.set_next_state(next_state)
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

    def reward(self, state: Tuple[float, float], action: int) -> float:
        position, _ = state

        # Reward for reaching the goal
        if position >= self.goal_position:
            return 0.0

        # Small negative reward for each step to encourage reaching the goal quickly
        return -1.0

    def reward_batch(self, states: Union[np.ndarray, Sequence[Any]], action: int) -> np.ndarray:
        states_arr = np.asarray(states)
        positions = states_arr[:, 0]
        return np.where(positions >= self.goal_position, 0.0, -1.0)

    def is_terminal(self, state: Tuple[float, float]) -> bool:
        position, _ = state
        return bool(position >= self.goal_position)

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
            state: Current 2-D car state ``[position, velocity]``.
            action_sampler: Object with a ``sample()`` method (kept for API
                parity with the base ``Environment`` contract; unused on the
                native rollout path which draws indices directly via NumPy).
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
        n_actions = len(self.actions)
        action_indices = np.random.randint(0, n_actions, size=steps_left, dtype=np.int32)

        return _native.simulate_rollout(
            initial_state=state_arr,
            actions=self._actions_int32,
            action_indices=action_indices,
            max_depth=max_depth,
            start_depth=depth,
            discount_factor=discount_factor,
            power=self.power,
            gravity=self.gravity,
            max_speed=self.max_speed,
            min_position=self.min_position,
            max_position=self.max_position,
            goal_position=self.goal_position,
            covariance=self._state_transition_dist.covariance,
        )

    def initial_state_dist(self) -> Distribution:
        class InitialState(Distribution):
            def sample(self, n_samples: int = 1) -> List[np.ndarray]:
                samples = []
                for _ in range(n_samples):
                    position = np.random.uniform(-0.6, -0.4)
                    velocity = 0.0
                    samples.append(np.array([position, velocity]))
                return samples

        return InitialState()

    def initial_observation_dist(self) -> Distribution:
        class InitialObservation(Distribution):
            def sample(self, n_samples: int = 1) -> List[np.ndarray]:
                return [np.array([0.0, 0.0])] * n_samples

        return InitialObservation()

    def get_actions(self) -> List[Any]:
        return self.actions

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        # Create a figure and axis
        pass

    def is_equal_observation(
        self, observation1: Tuple[float, float], observation2: Tuple[float, float]
    ) -> bool:
        return np.array_equal(observation1, observation2)

    def hash_action(self, action: Any) -> Hashable:
        # Discrete int actions (-1, 0, 1); already hashable.
        return action

    def get_metric_names(self) -> List[str]:
        """Get names of Mountain Car POMDP specific metrics.

        Returns:
            List containing metric names: goal_reaching_rate
        """
        return [metric.value for metric in MountainCarPOMDPMetrics]

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        """Compute Mountain Car POMDP specific metrics from simulation histories.

        Args:
            histories: List of simulation histories

        Returns:
            List of MetricValue objects containing the computed metrics
        """
        goal_reached = []
        for history in histories:
            goal_reached_in_history = False
            for step in history.history:
                position, _ = step.state
                if position >= self.goal_position:
                    goal_reached_in_history = True
                    break
            goal_reached.append(1 if goal_reached_in_history else 0)

        avg_goal_reached = float(np.mean(goal_reached))
        goal_reached_ci = confidence_interval(data=goal_reached, confidence=0.95)

        return [
            MetricValue(
                name=MountainCarPOMDPMetrics.GOAL_REACHING_RATE.value,
                value=avg_goal_reached,
                lower_confidence_bound=goal_reached_ci[0],
                upper_confidence_bound=goal_reached_ci[1],
            ),
        ]
