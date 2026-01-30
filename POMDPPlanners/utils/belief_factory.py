"""Top-level factory for creating environment-specific belief states.

This module provides a unified entry-point for constructing ready-to-use
:class:`~POMDPPlanners.core.belief.base_belief.Belief` objects for any POMDP
environment in the library.  It dispatches to per-environment factories when
a custom belief implementation exists (e.g., vectorized particle filters or
Gaussian beliefs), and falls back to a generic
:class:`~POMDPPlanners.core.belief.particle_beliefs.WeightedParticleBelief`
otherwise.

Classes:
    BeliefType: Enum of supported belief representations.

Functions:
    create_environment_belief: Top-level factory returning a configured Belief.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from POMDPPlanners.core.belief.belief_utils import get_initial_belief

if TYPE_CHECKING:
    from POMDPPlanners.core.belief.base_belief import Belief
    from POMDPPlanners.core.environment import Environment


class BeliefType(Enum):
    """Supported belief representations.

    Attributes:
        PARTICLE: Standard weighted particle belief.
        VECTORIZED_PARTICLE: Vectorized weighted particle belief with
            batched NumPy updates.
        GAUSSIAN: Single Gaussian (mean + covariance) belief.
        GAUSSIAN_MIXTURE: Gaussian mixture model belief.
    """

    PARTICLE = "particle"
    VECTORIZED_PARTICLE = "vectorized_particle"
    GAUSSIAN = "gaussian"
    GAUSSIAN_MIXTURE = "gaussian_mixture"


# ---------------------------------------------------------------------------
# Environment class name -> (per-env factory function path, default type)
# Uses lazy imports to avoid pulling in heavy environment modules at import
# time.
# ---------------------------------------------------------------------------

_ENV_FACTORY_REGISTRY: dict[str, tuple[str, str, BeliefType]] = {
    # (module_path, function_name, default_belief_type)
    "CartPolePOMDP": (
        "POMDPPlanners.environments.cartpole_pomdp_beliefs",
        "create_cartpole_belief",
        BeliefType.VECTORIZED_PARTICLE,
    ),
    "MountainCarPOMDP": (
        "POMDPPlanners.environments.mountain_car_pomdp_beliefs",
        "create_mountain_car_belief",
        BeliefType.VECTORIZED_PARTICLE,
    ),
    "LaserTagPOMDP": (
        "POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_beliefs",
        "create_laser_tag_belief",
        BeliefType.VECTORIZED_PARTICLE,
    ),
    "PushPOMDP": (
        "POMDPPlanners.environments.push_pomdp.push_pomdp_beliefs",
        "create_push_belief",
        BeliefType.VECTORIZED_PARTICLE,
    ),
    "SafeAntVelocityPOMDP": (
        "POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_pomdp_beliefs",
        "create_safety_ant_velocity_belief",
        BeliefType.VECTORIZED_PARTICLE,
    ),
    "ContinuousLightDarkPOMDP": (
        "POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_beliefs",
        "create_continuous_light_dark_belief",
        BeliefType.VECTORIZED_PARTICLE,
    ),
    "ContinuousLightDarkPOMDPDiscreteActions": (
        "POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_beliefs",
        "create_continuous_light_dark_belief",
        BeliefType.VECTORIZED_PARTICLE,
    ),
}


def create_environment_belief(
    env: "Environment",
    belief_type: BeliefType | None = None,
    n_particles: int = 200,
    **kwargs: Any,
) -> "Belief":
    """Create a ready-to-use belief for the given environment.

    When *belief_type* is ``None`` the environment's default belief type is
    used (typically ``VECTORIZED_PARTICLE`` for environments that have a
    custom updater, or ``PARTICLE`` otherwise).

    For environments without a registered per-environment factory the
    function falls back to a generic
    :func:`~POMDPPlanners.core.belief.belief_utils.get_initial_belief`
    producing a :class:`WeightedParticleBelief`.

    Args:
        env: POMDP environment instance.
        belief_type: Desired belief representation.  ``None`` selects the
            environment default.
        n_particles: Number of particles (used by PARTICLE and
            VECTORIZED_PARTICLE types).  Defaults to 200.
        **kwargs: Forwarded to per-environment factories (e.g.
            ``updater_type`` for Gaussian light-dark beliefs).

    Returns:
        A configured :class:`Belief` object.

    Raises:
        ValueError: If *belief_type* is not supported by the environment.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> env = TigerPOMDP(discount_factor=0.95)
        >>> belief = create_environment_belief(env, n_particles=100)
        >>> belief.sample() in ["tiger-left", "tiger-right"]
        True
    """
    env_class_name = type(env).__name__

    registry_entry = _ENV_FACTORY_REGISTRY.get(env_class_name)
    if registry_entry is None:
        return _fallback_belief(env, belief_type, n_particles)

    module_path, func_name, default_type = registry_entry
    resolved_type = belief_type if belief_type is not None else default_type

    factory_fn = _lazy_import(module_path, func_name)
    return factory_fn(env, belief_type=resolved_type, n_particles=n_particles, **kwargs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fallback_belief(
    env: "Environment",
    belief_type: BeliefType | None,
    n_particles: int,
) -> "Belief":
    if belief_type is not None and belief_type != BeliefType.PARTICLE:
        raise ValueError(
            f"{type(env).__name__} does not have a custom belief factory. "
            f"Only BeliefType.PARTICLE is supported.  Got: {belief_type}"
        )
    return get_initial_belief(env, n_particles)


def _lazy_import(module_path: str, attr_name: str):
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, attr_name)
