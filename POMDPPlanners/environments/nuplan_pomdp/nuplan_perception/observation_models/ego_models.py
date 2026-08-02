# SPDX-License-Identifier: MIT

"""Ego-channel observation models.

Catalog of per-channel models for the ``ego`` observation channel (the ego proprioception
block ``[x, y, yaw, vx, vy, lat, heading_err]``). Add new ego models here and register them with
:func:`register_observation_model` so they can be selected by name.

Classes:
    EgoObservationModel: Additive-Gaussian-noise ego proprioception with a matching density.
"""

from typing import Any

import numpy as np

from POMDPPlanners.environments.nuplan_pomdp.nuplan_pomdp import EGO_STATE_WIDTH
from POMDPPlanners.environments.nuplan_pomdp.nuplan_perception.observation_model import (
    NuPlanObservationModel,
)
from POMDPPlanners.environments.nuplan_pomdp.nuplan_perception.observation_models.registry import (
    register_observation_model,
)


@register_observation_model("ego", "gaussian")
class EgoObservationModel(NuPlanObservationModel):
    """Ego proprioception corrupted by additive Gaussian noise, with a matching density.

    nuPlan gives the ego near-perfect self-localisation, so this channel is modelled as the
    true ego block plus small zero-mean Gaussian measurement noise. Provides both a sampler and
    a matching density, so it can back a scoring generative model.

    Attributes:
        ego_std: Std of the zero-mean Gaussian noise added to the ego proprioception vector.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> model = EgoObservationModel(ego_std=0.01)
        >>> perceived = model.perceive(np.zeros(7))
        >>> perceived.shape
        (7,)
    """

    channel = "ego"
    supports_density = True

    def __init__(self, ego_std: float = 0.01) -> None:
        """Initialize the ego observation model.

        Args:
            ego_std: Std of Gaussian noise on the ego proprioception reading.
        """
        self.ego_std = ego_std

    def perceive(self, clean_channel: Any) -> np.ndarray:
        ego = np.asarray(clean_channel, dtype=float)[:EGO_STATE_WIDTH].copy()
        return ego + np.random.normal(0.0, self.ego_std, size=EGO_STATE_WIDTH)

    def log_probability(self, clean_channel: Any, channel_observation: Any) -> float:
        truth = np.asarray(clean_channel, dtype=float)[:EGO_STATE_WIDTH]
        ego = np.asarray(channel_observation, dtype=float)[:EGO_STATE_WIDTH]
        diff = ego - truth
        return float(
            -0.5 * np.sum((diff / self.ego_std) ** 2)
            - EGO_STATE_WIDTH * np.log(self.ego_std)
            - 0.5 * EGO_STATE_WIDTH * np.log(2.0 * np.pi)
        )
