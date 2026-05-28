# SPDX-License-Identifier: MIT

"""Factory for pre-configured Gaussian beliefs for the Continuous Light-Dark POMDP.

This module provides a single factory function that creates a
:class:`~POMDPPlanners.core.belief.gaussian_belief.GaussianBelief` instance
pre-configured for the
:class:`~POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp.ContinuousLightDarkPOMDP`
environment, with an enum-based selector for the updater type
(Linear Kalman, EKF, or UKF).

Classes:
    GaussianBeliefUpdaterType: Enum selecting the Gaussian updater variant.

Functions:
    create_continuous_light_dark_gaussian_belief: Factory producing a configured GaussianBelief.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from POMDPPlanners.core.belief.gaussian_belief import GaussianBelief
from POMDPPlanners.core.belief.gaussian_belief_updaters import (
    ExtendedKalmanFilterUpdater,
    GaussianBeliefUpdater,
    LinearKalmanFilterUpdater,
    UnscentedKalmanFilterUpdater,
)

if TYPE_CHECKING:
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        ContinuousLightDarkPOMDP,
    )


class GaussianBeliefUpdaterType(Enum):
    """Selector for the Gaussian belief updater variant.

    Attributes:
        LINEAR_KALMAN: Standard Kalman filter (exact for the linear dynamics).
        EKF: Extended Kalman filter (linearised; equivalent to LKF here).
        UKF: Unscented Kalman filter (sigma-point propagation).
    """

    LINEAR_KALMAN = "linear_kalman"
    EKF = "ekf"
    UKF = "ukf"


def create_continuous_light_dark_gaussian_belief(
    env: "ContinuousLightDarkPOMDP",
    updater_type: GaussianBeliefUpdaterType,
    initial_covariance: np.ndarray,
    use_near_beacon_noise: bool = False,
) -> GaussianBelief:
    """Create a GaussianBelief configured for a ContinuousLightDarkPOMDP.

    The Continuous Light-Dark POMDP has linear-Gaussian dynamics::

        x_{t+1} = x_t + u_t + w,  w ~ N(0, Q)
        z_t     = x_{t+1} + v,    v ~ N(0, R)

    so A = I, B = I, H = I with Q and R taken from the environment.

    Args:
        env: ContinuousLightDarkPOMDP instance.
        updater_type: Which Gaussian updater to use.
        initial_covariance: Initial belief covariance of shape (2, 2).
        use_near_beacon_noise: If ``True`` use the near-beacon observation
            covariance (``env.observation_cov_matrix * 0.5``); otherwise use
            the full covariance. Defaults to False.

    Returns:
        A :class:`GaussianBelief` with the selected updater.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        ...     ContinuousLightDarkPOMDP,
        ... )
        >>> env = ContinuousLightDarkPOMDP(discount_factor=0.95)
        >>> belief = create_continuous_light_dark_gaussian_belief(
        ...     env=env,
        ...     updater_type=GaussianBeliefUpdaterType.LINEAR_KALMAN,
        ...     initial_covariance=np.eye(2) * 5.0,
        ... )
        >>> belief.mean.shape
        (2,)
    """
    initial_covariance = np.asarray(initial_covariance, dtype=float)
    Q = env.state_transition_cov_matrix.copy()
    R = _select_observation_covariance(env, use_near_beacon_noise)
    updater = _build_updater(updater_type, Q, R)

    return GaussianBelief(
        mean=env.start_state.astype(float),
        covariance=initial_covariance,
        updater=updater,
    )


def _select_observation_covariance(
    env: "ContinuousLightDarkPOMDP", use_near_beacon_noise: bool
) -> np.ndarray:
    if use_near_beacon_noise:
        return env.observation_cov_matrix * 0.5
    return env.observation_cov_matrix.copy()


def _build_updater(
    updater_type: GaussianBeliefUpdaterType,
    Q: np.ndarray,
    R: np.ndarray,
) -> GaussianBeliefUpdater:
    d = Q.shape[0]
    I = np.eye(d)

    if updater_type is GaussianBeliefUpdaterType.LINEAR_KALMAN:
        return LinearKalmanFilterUpdater(A=I, B=I, H=I, Q=Q, R=R)

    if updater_type is GaussianBeliefUpdaterType.EKF:
        return ExtendedKalmanFilterUpdater(
            transition_fn=lambda x, u: x + u,
            observation_fn=lambda x: x,
            transition_jacobian=lambda x, u: I,
            observation_jacobian=lambda x: I,
            Q=Q,
            R=R,
        )

    if updater_type is GaussianBeliefUpdaterType.UKF:
        return UnscentedKalmanFilterUpdater(
            transition_fn=lambda x, u: x + u,
            observation_fn=lambda x: x,
            Q=Q,
            R=R,
        )

    raise ValueError(f"Unknown updater type: {updater_type}")
