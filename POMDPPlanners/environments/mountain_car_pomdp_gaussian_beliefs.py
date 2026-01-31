"""Factory for pre-configured Gaussian beliefs for the Mountain Car POMDP.

This module provides a single factory function that creates a
:class:`~POMDPPlanners.core.belief.gaussian_belief.GaussianBelief` instance
pre-configured for the
:class:`~POMDPPlanners.environments.mountain_car_pomdp.MountainCarPOMDP`
environment, with an enum-based selector for the updater type (EKF or UKF).

The Mountain Car POMDP has nonlinear dynamics (velocity depends on
``cos(3 * position)``) with a linear-Gaussian observation model (identity
plus additive noise).  Because the dynamics are nonlinear, a standard
linear Kalman filter is not applicable; only EKF (which requires analytical
Jacobians) and UKF (Jacobian-free sigma-point propagation) are supported.

Classes:
    GaussianBeliefUpdaterType: Enum selecting the Gaussian updater variant.

Functions:
    create_mountain_car_gaussian_belief: Factory producing a configured GaussianBelief.
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
    from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP


class GaussianBeliefUpdaterType(Enum):
    """Selector for the Gaussian belief updater variant.

    Attributes:
        EKF: Extended Kalman filter (linearised via analytical Jacobians).
        UKF: Unscented Kalman filter (sigma-point propagation).
    """

    EKF = "ekf"
    UKF = "ukf"


def create_mountain_car_gaussian_belief(
    env: "MountainCarPOMDP",
    updater_type: GaussianBeliefUpdaterType,
    initial_covariance: Optional[np.ndarray] = None,
    process_noise_scale: float = 1e-4,
) -> GaussianBelief:
    """Create a GaussianBelief configured for a MountainCarPOMDP.

    The Mountain Car POMDP has nonlinear dynamics::

        v_{t+1} = clip(v_t + action * power + cos(3 * p_t) * (-gravity))
        p_{t+1} = clip(p_t + v_{t+1})
        z_t     = [p_{t+1}, v_{t+1}] + w,   w ~ N(0, R)

    where R is ``env.cov_matrix``.  A small process noise Q is added for
    numerical stability of the Kalman covariance updates.

    Args:
        env: MountainCarPOMDP instance.
        updater_type: Which Gaussian updater to use (EKF or UKF).
        initial_covariance: Initial belief covariance of shape (2, 2).
            Defaults to ``np.diag([0.2**2 / 12, 1e-4])`` (variance of
            Uniform(-0.6, -0.4) for position, small for velocity).
        process_noise_scale: Diagonal scaling for the process noise
            covariance Q.  Defaults to 1e-4.

    Returns:
        A :class:`GaussianBelief` with the selected updater.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
        >>> env = MountainCarPOMDP(discount_factor=0.99)
        >>> belief = create_mountain_car_gaussian_belief(
        ...     env=env,
        ...     updater_type=GaussianBeliefUpdaterType.EKF,
        ... )
        >>> belief.mean.shape
        (2,)
    """
    if initial_covariance is None:
        initial_covariance = np.diag([0.2**2 / 12.0, 1e-4])
    initial_covariance = np.asarray(initial_covariance, dtype=float)

    Q = np.eye(2) * process_noise_scale
    R = env.cov_matrix.copy()
    updater = _build_updater(updater_type, env, Q, R)

    return GaussianBelief(
        mean=np.array([-0.5, 0.0]),
        covariance=initial_covariance,
        updater=updater,
    )


def _build_updater(
    updater_type: GaussianBeliefUpdaterType,
    env: "MountainCarPOMDP",
    Q: np.ndarray,
    R: np.ndarray,
) -> GaussianBeliefUpdater:
    transition_fn = _make_transition_fn(env)
    observation_fn = _mountain_car_observation_fn

    if updater_type is GaussianBeliefUpdaterType.EKF:
        return ExtendedKalmanFilterUpdater(
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            transition_jacobian=_make_transition_jacobian(env),
            observation_jacobian=_mountain_car_observation_jacobian,
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


def _make_transition_fn(env: "MountainCarPOMDP"):
    power = env.power
    gravity = env.gravity
    max_speed = env.max_speed
    min_position = env.min_position
    max_position = env.max_position

    def _transition_fn(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        action = float(np.asarray(u).ravel()[0])
        position, velocity = x

        velocity = velocity + action * power + math.cos(3.0 * position) * (-gravity)
        velocity = np.clip(velocity, -max_speed, max_speed)
        position = position + velocity
        position = np.clip(position, min_position, max_position)

        if position == min_position and velocity < 0:
            velocity = 0.0

        return np.array([position, velocity])

    return _transition_fn


def _mountain_car_observation_fn(x: np.ndarray) -> np.ndarray:
    return x.copy()


def _make_transition_jacobian(env: "MountainCarPOMDP"):
    power = env.power
    gravity = env.gravity
    max_speed = env.max_speed
    min_position = env.min_position
    max_position = env.max_position

    def _transition_jacobian(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        action = float(np.asarray(u).ravel()[0])
        position, velocity = x

        new_velocity = velocity + action * power + math.cos(3.0 * position) * (-gravity)
        clipped_velocity = np.clip(new_velocity, -max_speed, max_speed)
        new_position = position + clipped_velocity
        clipped_position = np.clip(new_position, min_position, max_position)

        velocity_not_clipped = -max_speed < new_velocity < max_speed
        position_not_clipped = min_position < new_position < max_position

        dv_dp = 3.0 * gravity * math.sin(3.0 * position) if velocity_not_clipped else 0.0
        dv_dv = 1.0 if velocity_not_clipped else 0.0

        if position_not_clipped:
            dp_dp = 1.0 + dv_dp
            dp_dv = dv_dv
        else:
            dp_dp = 0.0
            dp_dv = 0.0

        if clipped_position == min_position and clipped_velocity < 0:
            dv_dp = 0.0
            dv_dv = 0.0

        return np.array([[dp_dp, dp_dv], [dv_dp, dv_dv]])

    return _transition_jacobian


def _mountain_car_observation_jacobian(x: np.ndarray) -> np.ndarray:
    return np.eye(len(x))
