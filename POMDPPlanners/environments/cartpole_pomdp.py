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
    CartPoleStateTransition: Physics-based state transition model
    CartPoleObservation: Gaussian noise observation model
    CartPolePOMDP: Main CartPole environment with POMDP formulation
"""

import math
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray

import scipy.stats as stats

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    ObservationModel,
    SpaceInfo,
    SpaceType,
    StateTransitionModel,
)


class CartPoleStateTransition(StateTransitionModel):
    """Physics-based state transition model for CartPole POMDP.

    This model implements the classical cart-pole dynamics with deterministic
    physics simulation. The cart experiences forces that affect both cart
    acceleration and pole angular acceleration through coupled equations of motion.

    Attributes:
        state: Current state [cart_position, cart_velocity, pole_angle, pole_velocity]
        action: Force direction (0 for left, 1 for right)
        force_mag: Magnitude of applied force
        total_mass: Combined mass of cart and pole
        polemass_length: Pole mass times pole length (moment calculation)
        gravity: Gravitational acceleration constant
        length: Half the pole's length
        kinematics_integrator: Integration method ("euler" or "semi-implicit euler")
        tau: Time step for integration
        masspole: Mass of the pole

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> # Define initial state [position, velocity, angle, angular_velocity]
        >>> state = np.array([0.0, 0.0, 0.1, 0.0])
        >>> action = 1  # Apply right force

        >>> # Create transition model with physics parameters
        >>> transition = CartPoleStateTransition(
        ...     state=state,
        ...     action=action,
        ...     force_mag=10.0,
        ...     total_mass=1.1,
        ...     polemass_length=0.05,
        ...     gravity=9.8,
        ...     length=0.5,
        ...     kinematics_integrator="euler",
        ...     tau=0.02,
        ...     masspole=0.1
        ... )

        >>> # Simulate physics step
        >>> next_state = transition.sample()[0]
        >>> len(next_state) == 4  # [pos, vel, angle, ang_vel]
        True
        >>> isinstance(next_state, np.ndarray)
        True
    """

    def __init__(
        self,
        state: np.ndarray,
        action: int,
        force_mag: float,
        total_mass: float,
        polemass_length: float,
        gravity: float,
        length: float,
        kinematics_integrator: str,
        tau: float,
        masspole: float,
    ):
        super().__init__(state, action)

        self.force_mag = force_mag
        self.total_mass = total_mass
        self.polemass_length = polemass_length
        self.gravity = gravity
        self.length = length
        self.kinematics_integrator = kinematics_integrator
        self.tau = tau
        self.masspole = masspole

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if self.action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        next_state = np.array([x, x_dot, theta, theta_dot])
        return [next_state] * n_samples

    def probability(self, values: List[np.ndarray]) -> np.ndarray:
        # Deterministic transition - probability is 1.0 for the exact next state, 0.0 otherwise
        result = np.zeros(len(values))
        expected_next_state = self.sample()[0]  # Get the deterministic next state
        for i, value in enumerate(values):
            if np.array_equal(value, expected_next_state):
                result[i] = 1.0
        return result


class CartPoleObservation(ObservationModel):
    """Noisy observation model for CartPole POMDP.

    This model adds Gaussian noise to the true state to create partial observability.
    The agent receives a noisy version of the full state vector, making it challenging
    to determine the exact cart-pole configuration.

    Attributes:
        next_state: True state after action execution
        action: Action that was taken (not used in observation generation)
        noise_cov: Covariance matrix for observation noise

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> # Define true state after action
        >>> true_state = np.array([0.1, 0.05, 0.02, -0.1])
        >>> action = 1

        >>> # Define observation noise covariance
        >>> noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])

        >>> # Create observation model
        >>> obs_model = CartPoleObservation(
        ...     next_state=true_state,
        ...     action=action,
        ...     noise_cov=noise_cov
        ... )

        >>> # Sample noisy observation
        >>> observation = obs_model.sample()[0]
        >>> len(observation) == 4  # Same dimensionality as state
        True
        >>> isinstance(observation, np.ndarray)
        True

        >>> # Calculate probability of specific observation
        >>> prob = obs_model.probability([observation])
        >>> len(prob) == 1
        True
    """

    def __init__(self, next_state: np.ndarray, action: int, noise_cov: NDArray[np.floating[Any]]):
        super().__init__(next_state=next_state, action=action)
        self.noise_cov = noise_cov

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        samples = []
        for _ in range(n_samples):
            sample = np.random.multivariate_normal(self.next_state, self.noise_cov)
            samples.append(sample)
        return samples

    def probability(self, values: List[np.ndarray]) -> np.ndarray:
        # Handle empty list case
        if len(values) == 0:
            return np.array([])

        # Convert list of arrays to 2D numpy array for vectorized computation
        values_array = np.array(values)
        mvn = stats.multivariate_normal(mean=self.next_state, cov=self.noise_cov)  # type: ignore
        pdf_values = mvn.pdf(values_array)

        # Ensure result is always a 1D array (pdf returns scalar for single input)
        return np.atleast_1d(pdf_values)


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
        super().__init__()

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        # Generate all samples at once using vectorized uniform distribution
        samples_array = np.random.uniform(low=-0.05, high=0.05, size=(n_samples, 4))
        return [sample for sample in samples_array]


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

    def __init__(
        self,
        discount_factor: float,
        noise_cov: NDArray[np.floating[Any]],
        name: str = "CartPolePOMDP",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        # Set all configuration parameters first
        self.noise_cov = noise_cov
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
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

    def state_transition_model(self, state: np.ndarray, action: int) -> StateTransitionModel:
        return CartPoleStateTransition(
            state=state,
            action=action,
            force_mag=self.force_mag,
            total_mass=self.total_mass,
            polemass_length=self.polemass_length,
            gravity=self.gravity,
            length=self.length,
            kinematics_integrator=self.kinematics_integrator,
            tau=self.tau,
            masspole=self.masspole,
        )

    def observation_model(self, next_state: np.ndarray, action: int) -> ObservationModel:
        return CartPoleObservation(next_state=next_state, action=action, noise_cov=self.noise_cov)

    def reward(self, state: np.ndarray, action: int) -> float:
        x, x_dot, theta, theta_dot = state

        terminated = self.is_terminal(state)

        if not terminated:
            reward = 1.0
        else:
            reward = 0.0

        return reward

    def is_terminal(self, state: np.ndarray) -> bool:
        x, x_dot, theta, theta_dot = state

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        return terminated

    def initial_state_dist(self) -> Distribution:
        return CartPoleInitialStateDistribution()

    def initial_observation_dist(self) -> Distribution:
        return CartPoleInitialStateDistribution()

    def get_actions(self) -> List[int]:
        return [0, 1]

    def is_equal_observation(self, observation1: np.ndarray, observation2: np.ndarray) -> bool:
        return np.array_equal(observation1, observation2)
