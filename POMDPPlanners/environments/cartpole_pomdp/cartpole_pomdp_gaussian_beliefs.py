# SPDX-License-Identifier: MIT

"""Factory for pre-configured Gaussian beliefs for the CartPole POMDP.

This module provides a single factory function that creates a
:class:`~POMDPPlanners.core.belief.gaussian_belief.GaussianBelief` instance
pre-configured for the
:class:`~POMDPPlanners.environments.cartpole_pomdp.CartPolePOMDP`
environment, with an enum-based selector for the updater type (EKF or UKF).

The CartPole POMDP has nonlinear dynamics (coupled cart-pole physics) with
a linear-Gaussian observation model (identity plus additive noise).
Because the dynamics are nonlinear, a standard linear Kalman filter is not
applicable; only EKF (which requires analytical Jacobians) and UKF
(Jacobian-free sigma-point propagation) are supported.

Classes:
    GaussianBeliefUpdaterType: Enum selecting the Gaussian updater variant.

Functions:
    create_cartpole_gaussian_belief: Factory producing a configured GaussianBelief.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np

from POMDPPlanners.core.belief.gaussian_belief import GaussianBelief
from POMDPPlanners.core.belief.gaussian_belief_updaters import (
    ExtendedKalmanFilterUpdater,
    GaussianBeliefUpdater,
    UnscentedKalmanFilterUpdater,
)

if TYPE_CHECKING:
    from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP


class GaussianBeliefUpdaterType(Enum):
    """Selector for the Gaussian belief updater variant.

    Attributes:
        EKF: Extended Kalman filter (linearised via analytical Jacobians).
        UKF: Unscented Kalman filter (sigma-point propagation).
    """

    EKF = "ekf"
    UKF = "ukf"


def create_cartpole_gaussian_belief(
    env: "CartPolePOMDP",
    updater_type: GaussianBeliefUpdaterType,
    initial_covariance: Optional[np.ndarray] = None,
    process_noise_scale: float = 1e-4,
) -> GaussianBelief:
    """Create a GaussianBelief configured for a CartPolePOMDP.

    The CartPole POMDP has nonlinear dynamics::

        x_{t+1} = f(x_t, u_t)      (deterministic cart-pole physics)
        z_t     = x_{t+1} + v,      v ~ N(0, R)

    where R is ``env.noise_cov``.  A small process noise Q is added for
    numerical stability of the Kalman covariance updates.

    Args:
        env: CartPolePOMDP instance.
        updater_type: Which Gaussian updater to use (EKF or UKF).
        initial_covariance: Initial belief covariance of shape (4, 4).
            Defaults to ``np.eye(4) * (0.1**2 / 12)`` (variance of
            Uniform(-0.05, 0.05)).
        process_noise_scale: Diagonal scaling for the process noise
            covariance Q.  Defaults to 1e-4.

    Returns:
        A :class:`GaussianBelief` with the selected updater.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
        >>> noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
        >>> env = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov)
        >>> belief = create_cartpole_gaussian_belief(
        ...     env=env,
        ...     updater_type=GaussianBeliefUpdaterType.EKF,
        ... )
        >>> belief.mean.shape
        (4,)
    """
    if initial_covariance is None:
        initial_covariance = np.eye(4) * (0.1**2 / 12.0)
    initial_covariance = np.asarray(initial_covariance, dtype=float)

    Q = np.eye(4) * process_noise_scale
    R = env.noise_cov.copy()
    updater = _build_updater(updater_type, env, Q, R)

    return GaussianBelief(
        mean=np.zeros(4),
        covariance=initial_covariance,
        updater=updater,
    )


def _build_updater(
    updater_type: GaussianBeliefUpdaterType,
    env: "CartPolePOMDP",
    Q: np.ndarray,
    R: np.ndarray,
) -> GaussianBeliefUpdater:
    transition_fn = _make_transition_fn(env)
    observation_fn = _cartpole_observation_fn

    if updater_type is GaussianBeliefUpdaterType.EKF:
        return ExtendedKalmanFilterUpdater(
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            transition_jacobian=_make_transition_jacobian(env),
            observation_jacobian=_cartpole_observation_jacobian,
            Q=Q,
            R=R,
        )

    if updater_type is GaussianBeliefUpdaterType.UKF:
        return UnscentedKalmanFilterUpdater(
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            Q=Q,
            R=R,
        )

    raise ValueError(f"Unknown updater type: {updater_type}")


def _make_transition_fn(env: "CartPolePOMDP"):
    force_mag = env.force_mag
    gravity = env.gravity
    masspole = env.masspole
    total_mass = env.total_mass
    length = env.length
    polemass_length = env.polemass_length
    tau = env.tau
    integrator = env.kinematics_integrator

    def _transition_fn(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        action = float(np.asarray(u).ravel()[0])
        force = force_mag if action >= 0.5 else -force_mag

        cart_pos, cart_vel, theta, theta_dot = x
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        temp = (force + polemass_length * theta_dot**2 * sin_theta) / total_mass
        theta_acc = (gravity * sin_theta - cos_theta * temp) / (
            length * (4.0 / 3.0 - masspole * cos_theta**2 / total_mass)
        )
        x_acc = temp - polemass_length * theta_acc * cos_theta / total_mass

        if integrator == "euler":
            new_cart_pos = cart_pos + tau * cart_vel
            new_cart_vel = cart_vel + tau * x_acc
            new_theta = theta + tau * theta_dot
            new_theta_dot = theta_dot + tau * theta_acc
        else:
            new_cart_vel = cart_vel + tau * x_acc
            new_cart_pos = cart_pos + tau * new_cart_vel
            new_theta_dot = theta_dot + tau * theta_acc
            new_theta = theta + tau * new_theta_dot

        return np.array([new_cart_pos, new_cart_vel, new_theta, new_theta_dot])

    return _transition_fn


def _cartpole_observation_fn(x: np.ndarray) -> np.ndarray:
    return x.copy()


def _make_transition_jacobian(env: "CartPolePOMDP"):
    force_mag = env.force_mag
    gravity = env.gravity
    masspole = env.masspole
    total_mass = env.total_mass
    length = env.length
    polemass_length = env.polemass_length
    tau = env.tau
    integrator = env.kinematics_integrator

    def _transition_jacobian(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        action = float(np.asarray(u).ravel()[0])
        force = force_mag if action >= 0.5 else -force_mag

        theta = x[2]
        theta_dot = x[3]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        denom_inner = 4.0 / 3.0 - masspole * cos_t**2 / total_mass
        denom = length * denom_inner

        temp = (force + polemass_length * theta_dot**2 * sin_t) / total_mass

        numer = gravity * sin_t - cos_t * temp

        # d(temp)/d(theta) and d(temp)/d(theta_dot)
        dtemp_dtheta = (polemass_length * theta_dot**2 * cos_t) / total_mass
        dtemp_dtheta_dot = (2.0 * polemass_length * theta_dot * sin_t) / total_mass

        # d(numer)/d(theta) = g*cos - (-sin*temp + cos*dtemp/dtheta)
        dnumer_dtheta = gravity * cos_t + sin_t * temp - cos_t * dtemp_dtheta
        # d(numer)/d(theta_dot) = -cos * dtemp/dtheta_dot
        dnumer_dtheta_dot = -cos_t * dtemp_dtheta_dot

        # d(denom)/d(theta) via chain rule on cos^2
        ddenom_dtheta = length * (2.0 * masspole * cos_t * sin_t / total_mass)

        # theta_acc = numer / denom  ->  quotient rule
        dthetaacc_dtheta = (dnumer_dtheta * denom - numer * ddenom_dtheta) / (denom**2)
        dthetaacc_dtheta_dot = dnumer_dtheta_dot / denom

        # x_acc = temp - (polemass_length * theta_acc * cos_t) / total_mass
        theta_acc = numer / denom
        pml_tm = polemass_length / total_mass
        dxacc_dtheta = dtemp_dtheta - pml_tm * (dthetaacc_dtheta * cos_t + theta_acc * (-sin_t))
        dxacc_dtheta_dot = dtemp_dtheta_dot - pml_tm * (dthetaacc_dtheta_dot * cos_t)

        J = np.eye(4)
        if integrator == "euler":
            J[0, 1] = tau
            J[1, 2] = tau * dxacc_dtheta
            J[1, 3] = tau * dxacc_dtheta_dot
            J[2, 3] = tau
            J[3, 2] = tau * dthetaacc_dtheta
            J[3, 3] = 1.0 + tau * dthetaacc_dtheta_dot
        else:
            J[1, 2] = tau * dxacc_dtheta
            J[1, 3] = tau * dxacc_dtheta_dot
            J[0, 1] = tau
            J[0, 2] = tau * J[1, 2]
            J[0, 3] = tau * J[1, 3]
            J[3, 2] = tau * dthetaacc_dtheta
            J[3, 3] = 1.0 + tau * dthetaacc_dtheta_dot
            J[2, 2] = 1.0 + tau * J[3, 2]
            J[2, 3] = tau * J[3, 3]

        return J

    return _transition_jacobian


def _cartpole_observation_jacobian(x: np.ndarray) -> np.ndarray:
    return np.eye(len(x))
