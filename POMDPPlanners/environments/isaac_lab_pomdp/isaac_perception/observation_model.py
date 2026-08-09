# SPDX-License-Identifier: MIT

"""Per-channel observation model: named state blocks in, one perceived channel out.

The IsaacLab world reads privileged state from the physics engine and the observation from a
sensor, so the two need not share a width or a meaning. A planner-side generative model
therefore composes its observation *by channel*: each channel (``base_pose``, ``lidar``,
``hazard_signal``, ...) is produced by its own :class:`IsaacObservationModel`, and the
generative model holds a ``{channel: IsaacObservationModel}`` map.

Where this differs from the CARLA equivalent
--------------------------------------------
CARLA's per-channel model is a *clean channel -> perceived channel* transform, because the CARLA
world emits a clean, fully-detected reading of the same channel. Isaac is not like that:

* A channel is derived from **named state blocks**, not from a same-named clean channel. A LiDAR
  reading is a function of the robot pose and the scene; a latent-type signal is a function of the
  hidden type *and* the robot position. So :meth:`IsaacObservationModel.perceive` receives the
  whole ``{state_channel: block}`` mapping and reads the blocks it declares in
  :attr:`state_channels`.
* :meth:`IsaacObservationModel.encode` defaults to the **identity**, not to ``perceive``. The
  IsaacLab world's reading is a real sensor buffer that already carries the physics' own noise, so
  re-perceiving it would corrupt it twice. A channel the world happens to emit clean (a commanded
  goal pose, say) overrides ``encode`` to call ``perceive``.

Two capabilities, with different reach:

* :meth:`IsaacObservationModel.perceive` — sample this channel from the clean state. Required;
  used to generate a tree observation (``sample_observation``).
* :meth:`IsaacObservationModel.log_probability` — this channel's observation density. Optional; a
  sample-only channel (a learned encoder, say) may leave it unimplemented and is still usable to
  generate observations, but is rejected by a generative model that must score observations for a
  belief update.

Classes:
    IsaacObservationModel: Abstract state-blocks -> one-perceived-channel interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Mapping, Tuple

import numpy as np


class IsaacObservationModel(ABC):
    """Abstract single-channel observation model: named state blocks -> one perceived channel.

    A concrete perception maps the clean state blocks it declares in :attr:`state_channels` to the
    degraded value a planner sees on the one observation channel it declares in :attr:`channel`.
    Implementations set :attr:`supports_density` to ``True`` when they also provide
    :meth:`log_probability`.

    Attributes:
        channel: The observation-dict key this model produces (e.g. ``"lidar"``).
        state_channels: The state-schema block names this model reads. Declared so a generative
            model can check its schema supplies them before a run rather than failing mid-search.
        supports_density: Whether :meth:`log_probability` is implemented. Sample-only channels
            leave this ``False`` and are usable only where sampling is needed.

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    channel: str = ""
    state_channels: Tuple[str, ...] = ()
    supports_density: bool = False

    @abstractmethod
    def perceive(self, clean_state: Mapping[str, np.ndarray]) -> Any:
        """Sample this channel's perceived value from the clean state blocks.

        Args:
            clean_state: The state's named blocks, ``{state_channel: block}``. A model reads only
                the blocks it declares in :attr:`state_channels`.

        Returns:
            The perceived value of this channel.
        """

    def log_probability(
        self, clean_state: Mapping[str, np.ndarray], channel_observation: Any
    ) -> float:
        """Log-density of ``channel_observation`` given the clean state blocks.

        Args:
            clean_state: The state's named blocks, ``{state_channel: block}``.
            channel_observation: The channel value whose likelihood is scored.

        Returns:
            This channel's observation log-probability.

        Raises:
            NotImplementedError: If this is a sample-only channel with no density.
        """
        del clean_state, channel_observation
        raise NotImplementedError(
            f"{type(self).__name__} is a sample-only observation model with no density; "
            "it cannot back a generative model that scores observations."
        )

    def encode(self, raw_channel: Any) -> Any:
        """Map the world's raw reading of this channel into the perceived space.

        The identity by default: an IsaacLab sensor buffer is already a real, physics-noised
        reading, and passing it through :meth:`perceive` would add a second layer of noise the
        belief would then have to explain. Override this on a channel the world emits *clean* —
        there, ``return self.perceive({...})`` is the right encoding.

        Args:
            raw_channel: This channel's value as emitted by the world.

        Returns:
            The channel value the belief and planner consume.
        """
        return raw_channel
