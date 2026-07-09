# SPDX-License-Identifier: MIT

"""Registry of per-channel CARLA observation models keyed by ``(channel, name)``.

Concrete per-channel models register themselves with the :func:`register_observation_model`
decorator so a user can select, per observation channel, which model the planner's generative
environment holds — e.g. ``{"gnss": "gaussian", "agents": "factored"}``. The environment
resolves the selection into instances via :func:`build_observation_model`.

Functions:
    register_observation_model: Decorator registering a factory under ``(channel, name)``.
    build_observation_model: Instantiate the model registered under ``(channel, name)``.
    available_observation_models: List the names registered for a channel.
"""

from typing import Any, Callable, Dict, List, TypeVar

from POMDPPlanners.environments.carla_pomdp.carla_perception.observation_model import (
    CarlaObservationModel,
)

ObservationModelFactory = Callable[..., CarlaObservationModel]
_FactoryT = TypeVar("_FactoryT", bound=ObservationModelFactory)

_REGISTRY: Dict[str, Dict[str, ObservationModelFactory]] = {}


def register_observation_model(channel: str, name: str) -> Callable[[_FactoryT], _FactoryT]:
    """Register an observation-model factory under ``(channel, name)`` for user selection.

    Args:
        channel: The observation-dict key the model handles (e.g. ``"gnss"``, ``"agents"``).
        name: The catalog name the user selects the model by within that channel.

    Returns:
        A decorator that registers the factory (a class or callable returning a
        :class:`~POMDPPlanners.environments.carla_pomdp.carla_perception.observation_model.CarlaObservationModel`)
        and returns it unchanged (its type is preserved).
    """

    def decorator(factory: _FactoryT) -> _FactoryT:
        _REGISTRY.setdefault(channel, {})[name] = factory
        return factory

    return decorator


def build_observation_model(channel: str, name: str, **kwargs: Any) -> CarlaObservationModel:
    """Instantiate the observation model registered under ``(channel, name)``.

    Args:
        channel: The observation-dict key to resolve the model for.
        name: The registered catalog name within that channel.
        **kwargs: Forwarded to the registered factory.

    Returns:
        The instantiated per-channel observation model.

    Raises:
        KeyError: If no model is registered under ``(channel, name)``.
    """
    channel_models = _REGISTRY.get(channel, {})
    if name not in channel_models:
        raise KeyError(
            f"No observation model '{name}' registered for channel '{channel}'; "
            f"available: {sorted(channel_models)}."
        )
    return channel_models[name](**kwargs)


def available_observation_models(channel: str) -> List[str]:
    """Return the catalog names registered for ``channel``, sorted."""
    return sorted(_REGISTRY.get(channel, {}))
