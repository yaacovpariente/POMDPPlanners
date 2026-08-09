# SPDX-License-Identifier: MIT

"""Registry of Isaac observation models keyed by name.

Concrete models register themselves with the :func:`register_observation_model` decorator so a
user can select, per observation channel, which model the planner's generative environment holds
— e.g. ``{"base_pose": "gaussian", "lidar": "ray_caster"}``. The environment resolves the
selection into instances via :func:`build_observation_model`.

Keyed by name alone, unlike the CARLA registry's ``(channel, name)`` key. CARLA's channels are
fixed by its world schema, so a per-channel namespace is natural there. Isaac channel names are
chosen per task (``base_pose`` on a mobile base, ``ee_pose`` on a manipulator, ``hazard_signal``
in this study), so keying by channel would force the same Gaussian model to be re-registered under
every task's spelling. The channel a model produces is a construction argument instead.

Functions:
    register_observation_model: Decorator registering a factory under ``name``.
    build_observation_model: Instantiate the model registered under ``name``.
    available_observation_models: List the registered names.
"""

from typing import Any, Callable, Dict, List, TypeVar

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_model import (
    IsaacObservationModel,
)

ObservationModelFactory = Callable[..., IsaacObservationModel]
_FactoryT = TypeVar("_FactoryT", bound=ObservationModelFactory)

_REGISTRY: Dict[str, ObservationModelFactory] = {}


def register_observation_model(name: str) -> Callable[[_FactoryT], _FactoryT]:
    """Register an observation-model factory under ``name`` for user selection.

    Args:
        name: The catalog name the user selects the model by.

    Returns:
        A decorator that registers the factory (a class or callable returning an
        :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_model.IsaacObservationModel`)
        and returns it unchanged (its type is preserved).
    """

    def decorator(factory: _FactoryT) -> _FactoryT:
        _REGISTRY[name] = factory
        return factory

    return decorator


def build_observation_model(name: str, **kwargs: Any) -> IsaacObservationModel:
    """Instantiate the observation model registered under ``name``.

    Args:
        name: The registered catalog name.
        **kwargs: Forwarded to the registered factory. Every model takes at least the
            ``channel`` it produces.

    Returns:
        The instantiated observation model.

    Raises:
        KeyError: If no model is registered under ``name``.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"No Isaac observation model '{name}' registered; available: {sorted(_REGISTRY)}."
        )
    return _REGISTRY[name](**kwargs)


def available_observation_models() -> List[str]:
    """Return the registered catalog names, sorted."""
    return sorted(_REGISTRY)
