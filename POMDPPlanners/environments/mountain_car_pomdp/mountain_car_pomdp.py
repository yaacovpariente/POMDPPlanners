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
    MountainCarTransition: Physics-based state transition model
    MountainCarObservation: Gaussian noise observation model
    MountainCarPOMDP: Main Mountain Car environment with POMDP formulation
"""

from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np
from numpy.typing import NDArray

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    ObservationModel,
    SpaceInfo,
    SpaceType,
    StateTransitionModel,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.environments.mountain_car_pomdp import _native
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal
from POMDPPlanners.utils.statistics_utils import confidence_interval

matplotlib.use("Agg")  # Use non-interactive backend


class MountainCarPOMDPMetrics(Enum):
    """Metric names for Mountain Car POMDP environment."""

    GOAL_REACHING_RATE = "goal_reaching_rate"


class MountainCarTransition(_native.MountainCarTransitionCpp):
    """Physics-based state transition model for Mountain Car POMDP.

    This model implements the physics of a car on a sinusoidal hill surface
    with additive Gaussian process noise. The car's velocity is affected by
    both the applied action (engine force) and gravitational force that depends
    on the slope of the hill.

    The physics equations are:
    - velocity += action * power + cos(3 * position) * (-gravity)
    - position += velocity

    After computing the deterministic next state, Normal noise is sampled
    from the provided distribution and added. The result is then clipped
    to respect position and velocity bounds.

    The ``sample()`` and ``probability()`` methods execute entirely in C++
    via the ``_native`` extension; this Python subclass only wraps the
    constructor so existing call sites that pass a
    :class:`CovarianceParameterizedMultivariateNormal` keep working.

    Attributes:
        state: Current state (position, velocity) tuple
        action: Engine action (-1, 0, or 1)
        power: Engine power scaling factor
        gravity: Gravitational force constant
        max_speed: Maximum velocity magnitude
        min_position: Minimum position boundary
        max_position: Maximum position boundary

    Example:
        Using the Mountain Car transition model::

            >>> import numpy as np
            >>> np.random.seed(42)  # For reproducible results
            >>> from POMDPPlanners.environments.mountain_car_pomdp import _native
            >>> _native.set_seed(42)
            >>> from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal
            >>>
            >>> # Define car state: position=-0.5 (in valley), velocity=0.0
            >>> state = (-0.5, 0.0)
            >>> action = 1  # Accelerate right/forward
            >>>
            >>> # Create transition model with noise
            >>> state_transition_cov = np.diag([2.5e-5, 1e-6])
            >>> state_transition_dist = CovarianceParameterizedMultivariateNormal(state_transition_cov)
            >>> transition = MountainCarTransition(
            ...     state=state,
            ...     action=action,
            ...     power=0.001,
            ...     gravity=0.0025,
            ...     max_speed=0.07,
            ...     min_position=-1.2,
            ...     max_position=0.6,
            ...     state_transition_dist=state_transition_dist
            ... )
            >>>
            >>> # Simulate physics step
            >>> next_state = transition.sample()[0]
            >>> # Returns new [position, velocity] with physics and noise applied
            >>> len(next_state) == 2
            True
    """

    def __init__(
        self,
        state: Tuple[float, float],
        action: int,
        power: float,
        gravity: float,
        max_speed: float,
        min_position: float,
        max_position: float,
        state_transition_dist: CovarianceParameterizedMultivariateNormal,
    ):
        super().__init__(
            state=state,
            action=action,
            power=power,
            gravity=gravity,
            max_speed=max_speed,
            min_position=min_position,
            max_position=max_position,
            covariance=state_transition_dist.covariance,
        )
        self._state_transition_dist = state_transition_dist


StateTransitionModel.register(MountainCarTransition)


class MountainCarObservation(_native.MountainCarObservationCpp):
    """Noisy observation model for Mountain Car POMDP.

    This model adds Gaussian noise to the true car state (position, velocity)
    to create partial observability. The agent receives noisy measurements
    of both position and velocity, making state estimation challenging.

    The ``sample()`` and ``probability()`` methods execute entirely in C++
    via the ``_native`` extension.

    Attributes:
        next_state: True state after action execution
        action: Action that was taken (not used in observation generation)
        mean: Expected observation (equals true state)

    Example:
        Using the Mountain Car observation model::

            >>> import numpy as np
            >>> np.random.seed(42)  # For reproducible results
            >>> from POMDPPlanners.environments.mountain_car_pomdp import _native
            >>> _native.set_seed(42)
            >>>
            >>> # Define true state after physics step
            >>> true_state = (-0.45, 0.02)  # [position, velocity]
            >>> action = 1
            >>>
            >>> # Define observation noise covariance and create distribution
            >>> from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal
            >>> cov_matrix = np.array([[0.1**2, 0], [0, 0.01**2]])  # Position and velocity noise
            >>> obs_dist = CovarianceParameterizedMultivariateNormal(cov_matrix)
            >>>
            >>> # Create observation model
            >>> obs_model = MountainCarObservation(
            ...     next_state=true_state,
            ...     action=action,
            ...     obs_dist=obs_dist
            ... )
            >>>
            >>> # Sample noisy observation
            >>> observation = obs_model.sample()[0]
            >>> # Returns noisy [position, velocity] close to true_state
            >>> len(observation) == 2
            True
            >>>
            >>> # Calculate observation probability
            >>> prob = obs_model.probability([observation])
            >>> prob.shape == (1,)
            True
    """

    def __init__(
        self,
        next_state: Tuple[float, float],
        action: int,
        obs_dist: CovarianceParameterizedMultivariateNormal,
    ):
        super().__init__(next_state=next_state, action=action, covariance=obs_dist.covariance)
        self._obs_dist = obs_dist


ObservationModel.register(MountainCarObservation)


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

    def state_transition_model(
        self, state: Tuple[float, float], action: int
    ) -> StateTransitionModel:
        return MountainCarTransition(  # pyright: ignore[reportReturnType]
            state=state,
            action=action,
            power=self.power,
            gravity=self.gravity,
            max_speed=self.max_speed,
            min_position=self.min_position,
            max_position=self.max_position,
            state_transition_dist=self._state_transition_dist,
        )

    def observation_model(self, next_state: Tuple[float, float], action: int) -> ObservationModel:
        return MountainCarObservation(  # pyright: ignore[reportReturnType]
            next_state=next_state, action=action, obs_dist=self._obs_dist
        )

    # ── Hot-path sampling overrides ─────────────────────────────────
    # The Python wrappers above only forward arguments to the native
    # C++ constructor and keep a reference to the dist object. Skip
    # the wrapper subclass and build the native kernel directly so
    # ``sample()`` produces a byte-identical RNG draw to the legacy
    # path.

    def sample_next_state(
        self,
        state: Union[Tuple[float, float], NDArray[np.float64]],
        action: int,
        n_samples: int = 1,
    ) -> NDArray[np.float64]:
        kernel = _native.MountainCarTransitionCpp(
            state=state,
            action=action,
            power=self.power,
            gravity=self.gravity,
            max_speed=self.max_speed,
            min_position=self.min_position,
            max_position=self.max_position,
            covariance=self._state_transition_dist.covariance,
        )
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
        kernel = _native.MountainCarObservationCpp(
            next_state=next_state,
            action=action,
            covariance=self._obs_dist.covariance,
        )
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
        kernel = _native.MountainCarTransitionCpp(
            state=state,
            action=action,
            power=self.power,
            gravity=self.gravity,
            max_speed=self.max_speed,
            min_position=self.min_position,
            max_position=self.max_position,
            covariance=self._state_transition_dist.covariance,
        )
        probs = np.asarray(kernel.probability(next_states))
        return np.log(probs + 1e-300)

    def observation_log_probability(
        self,
        next_state: Union[Tuple[float, float], NDArray[np.float64]],
        action: int,
        observations: Union[Sequence[Any], NDArray[np.float64]],
    ) -> np.ndarray:
        kernel = _native.MountainCarObservationCpp(
            next_state=next_state,
            action=action,
            covariance=self._obs_dist.covariance,
        )
        probs = np.asarray(kernel.probability(observations))
        return np.log(probs + 1e-300)

    def sample_next_state_batch(self, states: Any, action: int) -> np.ndarray:
        states_array = np.ascontiguousarray(np.asarray(states, dtype=np.float64))
        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)
        kernel = _native.MountainCarTransitionCpp(
            state=states_array[0],
            action=action,
            power=self.power,
            gravity=self.gravity,
            max_speed=self.max_speed,
            min_position=self.min_position,
            max_position=self.max_position,
            covariance=self._state_transition_dist.covariance,
        )
        return np.asarray(kernel.batch_sample(states_array), dtype=np.float64)

    def observation_log_probability_per_state(
        self, next_states: Any, action: int, observation: Any
    ) -> np.ndarray:
        next_states_array = np.ascontiguousarray(np.asarray(next_states, dtype=np.float64))
        if next_states_array.ndim == 1:
            next_states_array = next_states_array.reshape(1, -1)
        observation_array = np.ascontiguousarray(np.asarray(observation, dtype=np.float64))
        kernel = _native.MountainCarObservationCpp(
            next_state=next_states_array[0],
            action=action,
            covariance=self._obs_dist.covariance,
        )
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
