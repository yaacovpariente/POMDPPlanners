from typing import List
import math

import numpy as np
import scipy.stats as stats
from POMDPPlanners.core.environment import (
    ObservationModel,
    StateTransitionModel,
    DiscreteActionsEnvironment,
)
from POMDPPlanners.core.distributions import Distribution


class CartPoleStateTransition(StateTransitionModel):
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

    def sample(self) -> np.ndarray:
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if self.action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
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
        return next_state


class CartPoleObservation(ObservationModel):
    def __init__(self, next_state: np.ndarray, action: int, noise_cov: np.ndarray):
        super().__init__(next_state=next_state, action=action)

        self.noise_cov = noise_cov

    def sample(self) -> np.ndarray:
        return np.random.multivariate_normal(self.next_state, self.noise_cov)

    def probability(self, observation: np.ndarray) -> float:
        return stats.multivariate_normal(self.next_state, self.noise_cov).pdf(
            observation
        )


class CartPoleInitialStateDistribution(Distribution):
    def __init__(self):
        super().__init__()

    def sample(self) -> np.ndarray:
        return np.random.uniform(low=-0.05, high=0.05, size=(4,))


class CartPolePOMDP(DiscreteActionsEnvironment):
    def __init__(self, discount_factor: float, noise_cov: np.ndarray):
        super().__init__(discount_factor)

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

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

    def state_transition_model(
        self, state: np.ndarray, action: int
    ) -> StateTransitionModel:
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

    def observation_model(self, state: np.ndarray, action: int) -> ObservationModel:
        return CartPoleObservation(
            next_state=state, action=action, noise_cov=self.noise_cov
        )

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
