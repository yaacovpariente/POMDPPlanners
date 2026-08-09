# SPDX-License-Identifier: MIT

"""Proprioceptive observation models: root pose, joint state, and other direct state readings.

These are the channels an Isaac task measures on the robot itself rather than through an
exteroceptive sensor. Both models here copy one state block to one observation channel; they
differ only in whether the copy is noisy.

:class:`GaussianChannelObservationModel` is also the compatibility path: a single channel covering
the whole state vector reproduces the ``observation = state + N(0, Sigma)`` behaviour of the
one-space
:class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp.IsaacLabModelPOMDP`
exactly, so a factored model can be configured to regress to it.

Classes:
    GaussianChannelObservationModel: One state block seen through additive Gaussian noise.
    ExactChannelObservationModel: One state block observed without error.
"""

from typing import Any, Dict, Mapping, Optional

import numpy as np

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp import (
    NoiseStd,
    _diagonal_covariance,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_model import (
    IsaacObservationModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models.registry import (
    register_observation_model,
)
from POMDPPlanners.utils.multivariate_normal import (
    CovarianceParameterizedMultivariateNormal,
)


@register_observation_model("gaussian")
class GaussianChannelObservationModel(IsaacObservationModel):
    """One state block seen through additive Gaussian noise, with the matching density.

    The block width is not a construction argument: it is read from the state block on first use
    and the factorized normal is cached against it, so the same model definition serves any task
    whose named block happens to be wider or narrower.

    Attributes:
        channel: The observation-dict key this model produces.
        state_channels: The single state block read, as a one-tuple.
        noise_std: Scalar or per-channel standard deviation of the additive noise.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> model = GaussianChannelObservationModel(channel="base_pose", noise_std=0.1)
        >>> perceived = model.perceive({"base_pose": np.zeros(3)})
        >>> perceived.shape
        (3,)
        >>> float(model.log_probability({"base_pose": np.zeros(3)}, np.zeros(3))) > -1e9
        True
    """

    supports_density = True

    def __init__(
        self,
        channel: str,
        state_channel: Optional[str] = None,
        noise_std: NoiseStd = 0.1,
    ) -> None:
        """Initialize the Gaussian channel model.

        Args:
            channel: The observation-dict key this model produces.
            state_channel: The state block read. Defaults to ``channel``.
            noise_std: Scalar (isotropic) or per-channel std of the additive noise.
        """
        self.channel = channel
        self.state_channel = state_channel if state_channel is not None else channel
        self.state_channels = (self.state_channel,)
        self.noise_std = noise_std
        self._normals: Dict[int, CovarianceParameterizedMultivariateNormal] = {}

    def _normal(self, dim: int) -> CovarianceParameterizedMultivariateNormal:
        if dim not in self._normals:
            self._normals[dim] = CovarianceParameterizedMultivariateNormal(
                _diagonal_covariance(dim, self.noise_std)
            )
        return self._normals[dim]

    def perceive(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        mean = np.asarray(clean_state[self.state_channel], dtype=float).reshape(-1)
        return self._normal(mean.size).sample(mean, n_samples=1)[0]

    def log_probability(
        self, clean_state: Mapping[str, np.ndarray], channel_observation: Any
    ) -> float:
        mean = np.asarray(clean_state[self.state_channel], dtype=float).reshape(-1)
        reading = np.asarray(channel_observation, dtype=float).reshape(-1)
        return float(np.ravel(self._normal(mean.size).log_pdf(reading, mean))[0])


@register_observation_model("exact")
class ExactChannelObservationModel(IsaacObservationModel):
    """One state block observed without error, e.g. a commanded goal pose.

    Some channels are *known* rather than sensed — the pose command a navigation task issues is
    part of the task definition, not a measurement. Modelling those as noisy would make the belief
    spend particles disagreeing about something it cannot be wrong about.

    The density is the degenerate one implied by that: ``0.0`` when the reading matches the state
    block and ``-inf`` when it does not. That is a real likelihood for a deterministic channel, and
    it is what makes a particle carrying the wrong known value get weighted out rather than
    lingering.

    Attributes:
        channel: The observation-dict key this model produces.
        state_channels: The single state block read, as a one-tuple.
        tolerance: Absolute tolerance below which a reading counts as matching.

    Example:
        >>> import numpy as np
        >>> model = ExactChannelObservationModel(channel="pose_command")
        >>> model.perceive({"pose_command": np.array([1.0, 2.0])}).tolist()
        [1.0, 2.0]
        >>> model.log_probability({"pose_command": np.array([1.0, 2.0])}, np.array([1.0, 2.0]))
        0.0
    """

    supports_density = True

    def __init__(
        self,
        channel: str,
        state_channel: Optional[str] = None,
        tolerance: float = 1e-8,
    ) -> None:
        """Initialize the exact channel model.

        Args:
            channel: The observation-dict key this model produces.
            state_channel: The state block read. Defaults to ``channel``.
            tolerance: Absolute tolerance below which a reading counts as matching.
        """
        self.channel = channel
        self.state_channel = state_channel if state_channel is not None else channel
        self.state_channels = (self.state_channel,)
        self.tolerance = float(tolerance)

    def perceive(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.asarray(clean_state[self.state_channel], dtype=float).reshape(-1).copy()

    def log_probability(
        self, clean_state: Mapping[str, np.ndarray], channel_observation: Any
    ) -> float:
        truth = np.asarray(clean_state[self.state_channel], dtype=float).reshape(-1)
        reading = np.asarray(channel_observation, dtype=float).reshape(-1)
        if truth.shape != reading.shape:
            return float("-inf")
        return 0.0 if np.allclose(truth, reading, atol=self.tolerance) else float("-inf")
