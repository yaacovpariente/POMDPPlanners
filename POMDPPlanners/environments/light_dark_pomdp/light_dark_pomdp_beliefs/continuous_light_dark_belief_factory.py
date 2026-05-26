# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Belief factory for the Continuous Light-Dark POMDP.

Functions:
    create_continuous_light_dark_belief: Factory producing a configured belief
        for ContinuousLightDarkPOMDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_beliefs.continuous_light_dark_gaussian_beliefs import (
    GaussianBeliefUpdaterType,
    create_continuous_light_dark_gaussian_belief,
)
from POMDPPlanners.core.belief.belief_utils import get_initial_belief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ObservationModelType,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_beliefs.continuous_light_dark_vectorized_updater import (
    ContinuousLightDarkDistanceBasedVectorizedUpdater,
    ContinuousLightDarkNoObsInDarkVectorizedUpdater,
    ContinuousLightDarkVectorizedUpdater,
)
from POMDPPlanners.utils.belief_factory import BeliefType

if TYPE_CHECKING:
    from POMDPPlanners.core.belief.base_belief import Belief
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        ContinuousLightDarkPOMDP,
    )

_SUPPORTED_TYPES = {
    BeliefType.PARTICLE,
    BeliefType.VECTORIZED_PARTICLE,
    BeliefType.GAUSSIAN,
}
_DEFAULT_TYPE = BeliefType.VECTORIZED_PARTICLE


def create_continuous_light_dark_belief(
    env: "ContinuousLightDarkPOMDP",
    belief_type: BeliefType = _DEFAULT_TYPE,
    n_particles: int = 200,
    **kwargs: Any,
) -> "Belief":
    """Create a ready-to-use belief for the Continuous Light-Dark POMDP.

    For ``BeliefType.GAUSSIAN``, the following keyword arguments are
    forwarded to
    :func:`create_continuous_light_dark_gaussian_belief`:

    - ``updater_type`` (:class:`GaussianBeliefUpdaterType`): defaults to
      ``GaussianBeliefUpdaterType.UKF``.
    - ``initial_covariance`` (``np.ndarray``): defaults to ``np.eye(2) * 5.0``.
    - ``use_near_beacon_noise`` (``bool``): defaults to ``False``.

    Args:
        env: ContinuousLightDarkPOMDP environment instance.
        belief_type: Desired belief representation.
            Defaults to ``BeliefType.VECTORIZED_PARTICLE``.
        n_particles: Number of particles (ignored for GAUSSIAN).
            Defaults to 200.
        **kwargs: Extra arguments forwarded to the Gaussian factory.

    Returns:
        A configured :class:`Belief` object.

    Raises:
        ValueError: If *belief_type* is not supported.
    """
    if belief_type not in _SUPPORTED_TYPES:
        raise ValueError(
            f"ContinuousLightDarkPOMDP does not support {belief_type}. "
            f"Supported: {_SUPPORTED_TYPES}"
        )
    if belief_type == BeliefType.PARTICLE:
        return _create_particle_belief(env, n_particles)
    if belief_type == BeliefType.GAUSSIAN:
        return _create_gaussian_belief(env, **kwargs)
    return _create_vectorized_belief(env, n_particles)


def _create_particle_belief(env: "ContinuousLightDarkPOMDP", n_particles: int) -> "Belief":
    return get_initial_belief(env, n_particles)


def _create_gaussian_belief(env: "ContinuousLightDarkPOMDP", **kwargs: Any) -> "Belief":
    updater_type = kwargs.pop("updater_type", GaussianBeliefUpdaterType.UKF)
    initial_covariance = kwargs.pop("initial_covariance", np.eye(2) * 5.0)
    use_near_beacon_noise = kwargs.pop("use_near_beacon_noise", False)
    return create_continuous_light_dark_gaussian_belief(
        env=env,
        updater_type=updater_type,
        initial_covariance=initial_covariance,
        use_near_beacon_noise=use_near_beacon_noise,
    )


_CONTINUOUS_UPDATER_MAP = {
    ObservationModelType.NORMAL_NOISE: ContinuousLightDarkVectorizedUpdater,
    ObservationModelType.NORMAL_NOISE_NO_OBS_IN_DARK: ContinuousLightDarkNoObsInDarkVectorizedUpdater,
    ObservationModelType.DISTANCE_BASED: ContinuousLightDarkDistanceBasedVectorizedUpdater,
}


def _create_vectorized_belief(env: "ContinuousLightDarkPOMDP", n_particles: int) -> "Belief":
    updater_cls = _CONTINUOUS_UPDATER_MAP[env.observation_model_type]
    updater = updater_cls.from_environment(env)
    particles = np.array(env.initial_state_dist().sample(n_samples=n_particles))
    log_weights = np.log(np.ones(n_particles) / n_particles)
    return VectorizedWeightedParticleBelief(particles, log_weights, updater)
