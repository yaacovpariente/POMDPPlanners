# SPDX-License-Identifier: MIT

"""GNSS-channel observation models.

Catalog of per-channel models for the ``gnss`` observation channel. Add new GNSS models here
and register them with :func:`register_observation_model` so they can be selected by name.

Classes:
    GnssObservationModel: Additive-Gaussian-noise GNSS reading with a matching density.
"""

from typing import Any

import numpy as np

from POMDPPlanners.environments.carla_pomdp.carla_perception.observation_model import (
    CarlaObservationModel,
)
from POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models.registry import (
    register_observation_model,
)


@register_observation_model("gnss", "gaussian")
class GnssObservationModel(CarlaObservationModel):
    """GNSS channel corrupted by additive Gaussian noise, with a matching density.

    Attributes:
        gnss_std: Std of the zero-mean Gaussian noise added to the 2-D ``gnss`` reading.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> model = GnssObservationModel(gnss_std=1e-5)
        >>> perceived = model.perceive(np.zeros(2))
        >>> perceived.shape
        (2,)
    """

    channel = "gnss"
    supports_density = True

    def __init__(self, gnss_std: float = 1e-5) -> None:
        """Initialize the GNSS observation model.

        Args:
            gnss_std: Std of Gaussian noise on the ``gnss`` reading.
        """
        self.gnss_std = gnss_std

    def perceive(self, clean_channel: Any) -> np.ndarray:
        gnss = np.asarray(clean_channel, dtype=float)[:2].copy()
        return gnss + np.random.normal(0.0, self.gnss_std, size=2)

    def log_probability(self, clean_channel: Any, channel_observation: Any) -> float:
        truth = np.asarray(clean_channel, dtype=float)[:2]
        gnss = np.asarray(channel_observation, dtype=float)[:2]
        diff = gnss - truth
        return float(-0.5 * np.sum((diff / self.gnss_std) ** 2) - 2 * np.log(self.gnss_std))
