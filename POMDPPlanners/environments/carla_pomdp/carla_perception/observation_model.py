# SPDX-License-Identifier: MIT

"""Per-channel observation model: one clean observation channel in, one perceived channel out.

The forward-only world emits a raw, ground-truth observation; the planner-side generative model
degrades it into the reading a planner actually sees. That degradation is *factored by
observation channel* — each channel (``gnss``, ``agents``, and, in future, ``image`` / ``lidar``)
is handled by its own :class:`CarlaObservationModel`, and the generative model composes a
``{channel: CarlaObservationModel}`` map. This module holds the single-channel interface;
concrete per-channel models live in the
:mod:`~POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models` catalog.

Two capabilities, with different reach:

* :meth:`CarlaObservationModel.perceive` — sample this channel's perceived value from its clean
  one. Required; used by a model to generate a tree observation (``sample_observation``) and to
  encode the world's raw channel (``encode_observation``).
* :meth:`CarlaObservationModel.log_probability` — this channel's observation density. Optional;
  a sample-only channel (e.g. a learned encoder) may leave it unimplemented and is still usable
  to generate observations, but is rejected by a generative model that must score observations
  for a belief update.

Classes:
    CarlaObservationModel: Abstract single-channel clean -> perceived observation interface.
"""

from abc import ABC, abstractmethod
from typing import Any


class CarlaObservationModel(ABC):
    """Abstract single-channel observation model: clean channel -> perceived channel.

    A concrete perception maps one observation channel's clean, fully-detected value (built from
    a state by a model, or taken from the world's raw reading) to the degraded value a planner
    sees. Implementations declare which channel they handle via :attr:`channel` and set
    :attr:`supports_density` to ``True`` when they also provide :meth:`log_probability`.

    Attributes:
        channel: The observation-dict key this model handles (e.g. ``"gnss"`` or ``"agents"``).
        supports_density: Whether :meth:`log_probability` is implemented. Sample-only channels
            leave this ``False`` and are usable only where sampling is needed.

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    channel: str = ""
    supports_density: bool = False

    @abstractmethod
    def perceive(self, clean_channel: Any) -> Any:
        """Sample this channel's perceived value from its clean, fully-detected value.

        Args:
            clean_channel: The noise-free value of this channel, built from a state or taken
                from the world's raw reading.

        Returns:
            The perceived value of the same channel.
        """

    def log_probability(self, clean_channel: Any, channel_observation: Any) -> float:
        """Log-density of ``channel_observation`` given the clean channel value.

        Args:
            clean_channel: The noise-free value of this channel built from a state.
            channel_observation: The channel value whose likelihood is scored.

        Returns:
            The channel's observation log-probability.

        Raises:
            NotImplementedError: If this is a sample-only channel without a density.
        """
        del clean_channel, channel_observation
        raise NotImplementedError(
            f"{type(self).__name__} is a sample-only observation model with no density; "
            "it cannot back a generative model that scores observations."
        )
