# SPDX-License-Identifier: MIT

"""Observation model for a persistent latent hazard type revealed only on entry.

Some studies need a state variable the belief has to *learn* rather than read off: a hazard whose
severity is a hidden property of the zone, fixed for the episode, and knowable only once the agent
is already inside. A risk-sensitive planner grades the spread of value across belief children, so
it can only see a severity that survives as belief dispersion. A severity resolved inside a single
reward call is averaged away before it reaches the risk measure.

This model is the observation half of that construction: a per-zone binary signal whose accuracy
is high inside the zone it describes and uninformative everywhere else. The type block itself
never appears in the observation — a channel that echoed the state would leave nothing to learn.

The zone geometry (centres, radii) is injected. It is the same disc description the ray-cast
models use, so a study can attach a physical obstacle to a zone and have the sensor see it.

Classes:
    LatentTypeSignalObservationModel: Per-zone binary signal, informative only inside the zone.
"""

from typing import Any, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_model import (
    IsaacObservationModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models.registry import (
    register_observation_model,
)

#: Accuracy of a binary signal that carries no information. A Bayes update against it leaves the
#: type posterior exactly where it was, which is what defers the reveal until after the agent has
#: committed to entering.
UNINFORMATIVE_ACCURACY = 0.5


@register_observation_model("latent_type_signal")
class LatentTypeSignalObservationModel(IsaacObservationModel):
    """Per-zone binary signal about a latent type, informative only inside that zone.

    Each zone contributes one 0/1 channel entry. While the agent is inside zone *j*, entry *j*
    reports that zone's true type with probability ``accuracy_inside``; every other entry, and all
    entries while the agent is outside every zone, is a coin flip. Summed over zones, the
    likelihood is therefore flat outside and separating inside — the reveal rule the construction
    depends on.

    Attributes:
        channel: The observation-dict key this model produces.
        state_channels: The type block and the position block, in that order.
        zone_centers: Zone centres, shape ``(Z, 2)``.
        zone_radii: Zone radii, shape ``(Z,)``.
        accuracy_inside: Probability the in-zone signal reports the true type.

    Example:
        >>> import numpy as np
        >>> model = LatentTypeSignalObservationModel(
        ...     channel="hazard_signal", type_channel="hazard_type",
        ...     position_channel="base_pose", zone_centers=[(0.0, 0.0), (5.0, 0.0)],
        ...     zone_radii=[1.0, 1.0], accuracy_inside=0.9, rng=np.random.default_rng(0))
        >>> inside = {"hazard_type": np.array([1.0, 0.0]), "base_pose": np.zeros(3)}
        >>> model.accuracy_at(inside).tolist()
        [0.9, 0.5]
        >>> far = {"hazard_type": np.array([1.0, 0.0]), "base_pose": np.array([2.5, 0.0, 0.0])}
        >>> model.accuracy_at(far).tolist()
        [0.5, 0.5]
    """

    supports_density = True

    def __init__(
        self,
        channel: str,
        type_channel: str,
        position_channel: str,
        zone_centers: ArrayLike,
        zone_radii: ArrayLike,
        accuracy_inside: float = 0.9,
        position_indices: Sequence[int] = (0, 1),
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Initialize the latent-type signal model.

        Args:
            channel: The observation-dict key this model produces.
            type_channel: The state block holding the per-zone latent types (entries in {0, 1}).
            position_channel: The state block holding the agent's task-space position.
            zone_centers: Zone centres, shape ``(Z, 2)``.
            zone_radii: Zone radii, shape ``(Z,)``.
            accuracy_inside: Probability the in-zone signal reports the true type. Must exceed
                :data:`UNINFORMATIVE_ACCURACY`, or entering a zone teaches the belief nothing and
                the construction collapses to the instantaneous-severity case it exists to avoid.
            position_indices: Positions of ``(x, y)`` within the position block.
            rng: Source of randomness for :meth:`perceive`. Defaults to a fresh default generator;
                call :meth:`seed` to make a run reproducible.

        Raises:
            ValueError: If the zone arrays disagree in count, or ``accuracy_inside`` is not in
                ``(0.5, 1]``.
        """
        if not UNINFORMATIVE_ACCURACY < accuracy_inside <= 1.0:
            raise ValueError(
                "accuracy_inside must be in (0.5, 1] for the in-zone signal to be informative, "
                f"got {accuracy_inside}"
            )
        self.channel = channel
        self.type_channel = type_channel
        self.position_channel = position_channel
        self.state_channels = (type_channel, position_channel)
        self.zone_centers = np.asarray(zone_centers, dtype=float).reshape(-1, 2)
        self.zone_radii = np.asarray(zone_radii, dtype=float).reshape(-1)
        if self.zone_centers.shape[0] != self.zone_radii.shape[0]:
            raise ValueError(
                f"zone_centers has {self.zone_centers.shape[0]} entries but zone_radii has "
                f"{self.zone_radii.shape[0]}"
            )
        self.accuracy_inside = float(accuracy_inside)
        self.position_indices = tuple(int(index) for index in position_indices)
        self._rng = rng if rng is not None else np.random.default_rng()

    @property
    def num_zones(self) -> int:
        """Number of zones, i.e. the width of the type block and of this channel."""
        return self.zone_centers.shape[0]

    def seed(self, seed: Optional[int]) -> None:
        """Reseed the signal sampler so a planning run is reproducible."""
        self._rng = np.random.default_rng(seed)

    def occupancy(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        """Per-zone occupancy indicator at the position in ``clean_state``, shape ``(Z,)``."""
        block = np.asarray(clean_state[self.position_channel], dtype=float).reshape(-1)
        position = block[list(self.position_indices)]
        distance = np.linalg.norm(self.zone_centers - position[np.newaxis, :], axis=-1)
        return (distance <= self.zone_radii).astype(float)

    def accuracy_at(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        """Per-zone signal accuracy at the position in ``clean_state``, shape ``(Z,)``."""
        inside = self.occupancy(clean_state)
        return UNINFORMATIVE_ACCURACY + inside * (self.accuracy_inside - UNINFORMATIVE_ACCURACY)

    def _types(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.asarray(clean_state[self.type_channel], dtype=float).reshape(-1)

    def perceive(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        types = self._types(clean_state)
        truthful = self._rng.random(types.size) < self.accuracy_at(clean_state)
        return np.where(truthful, types, 1.0 - types)

    def log_probability(
        self, clean_state: Mapping[str, np.ndarray], channel_observation: Any
    ) -> float:
        types = self._types(clean_state)
        signals = np.asarray(channel_observation, dtype=float).reshape(-1)
        if signals.shape != types.shape:
            return float("-inf")
        accuracy = self.accuracy_at(clean_state)
        agrees = np.isclose(types, signals)
        return float(np.log(np.where(agrees, accuracy, 1.0 - accuracy)).sum())

    def posterior_after_signal(
        self,
        prior: ArrayLike,
        clean_state: Mapping[str, np.ndarray],
        signals: ArrayLike,
    ) -> np.ndarray:
        """Bayes-update the per-zone high-type probability on one signal.

        Provided for diagnostics and for the gate check a study runs before committing compute; a
        particle filter goes through :meth:`log_probability` instead.

        Args:
            prior: Prior probability of the high type per zone, shape ``(Z,)``.
            clean_state: The state's named blocks, read for the agent's position.
            signals: The observed 0/1 signals, shape ``(Z,)``.

        Returns:
            Posterior probability of the high type per zone, shape ``(Z,)``.
        """
        accuracy = self.accuracy_at(clean_state)
        signal = np.asarray(signals, dtype=float).reshape(-1)
        prior_high = np.asarray(prior, dtype=float).reshape(-1)
        likelihood_high = np.where(signal > 0.5, accuracy, 1.0 - accuracy)
        likelihood_low = np.where(signal > 0.5, 1.0 - accuracy, accuracy)
        joint_high = prior_high * likelihood_high
        evidence = joint_high + (1.0 - prior_high) * likelihood_low
        return np.divide(joint_high, evidence, out=prior_high.copy(), where=evidence > 0.0)
