# SPDX-License-Identifier: MIT

"""Abstract planner-side generative-model interface over a factored IsaacLab schema.

:class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_pomdp.IsaacLabPOMDP` is a
forward-only *world*: it steps a live task and reports what happened, but it cannot resample from
an arbitrary state or score an observation density. A planner therefore carries a generative
*model* as ``policy.environment``. This module defines the interface that model satisfies.

What it changes against the one-space model
-------------------------------------------
:class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp.IsaacLabModelPOMDP`
keeps state and observation in the same space, with ``observation = state + N(0, Sigma)``. That is
generic across tasks, but it forces every observable quantity to be a state channel of the same
width — so a LiDAR ring has nowhere to go, and a quantity that must stay *hidden* has to be
smuggled in as a same-width trailing block that the observation happens to reinterpret.

Here the two spaces are separate:

* **State** is a flat vector carved into named blocks by an :class:`IsaacChannelSchema`. Flat,
  because a particle belief and a fitted linear model both want an array.
* **Observation** is a ``{channel: value}`` mapping, produced by a
  ``{channel: IsaacObservationModel}`` map. Channels are free to have any width, to read several
  state blocks, or to read none that appear in the observation at all.

A state block with no observation channel derived from it is simply hidden — which is how a
latent hazard type stays latent, with no packing trick.

Classes:
    IsaacChannelSchema: Named contiguous blocks over a flat vector.
    IsaacModelPOMDP: Abstract generative-model interface over the factored Isaac schema.
"""

from abc import abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

from POMDPPlanners.core.environment import DiscreteActionsEnvironment, SpaceInfo, SpaceType
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_model import (
    IsaacObservationModel,
)


@dataclass(frozen=True)
class IsaacChannelSchema:
    """Named contiguous blocks over a flat vector.

    Used for the state vector, and optionally for a flat raw observation the world emits, so that
    every consumer names a block instead of hard-coding an offset. Offsets are the failure mode
    this exists to remove: naming the wrong slice does not raise, it silently plants a zone
    somewhere the robot never goes, and reads downstream as "no separation".

    Args:
        channels: ``(name, width)`` pairs in packing order.

    Example:
        >>> import numpy as np
        >>> schema = IsaacChannelSchema((("base_pose", 3), ("hazard_type", 2)))
        >>> schema.total_dim
        5
        >>> state = schema.pack({"base_pose": [1.0, 2.0, 0.0], "hazard_type": [0.0, 1.0]})
        >>> state.tolist()
        [1.0, 2.0, 0.0, 0.0, 1.0]
        >>> schema.block(state, "hazard_type").tolist()
        [0.0, 1.0]
    """

    channels: Tuple[Tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("IsaacChannelSchema needs at least one channel")
        names = [name for name, _ in self.channels]
        if len(set(names)) != len(names):
            raise ValueError(f"channel names must be unique, got {names}")
        for name, width in self.channels:
            if width <= 0:
                raise ValueError(f"channel {name!r} must have positive width, got {width}")

    @property
    def names(self) -> Tuple[str, ...]:
        """Channel names in packing order."""
        return tuple(name for name, _ in self.channels)

    @property
    def total_dim(self) -> int:
        """Width of the full flat vector."""
        return sum(width for _, width in self.channels)

    def width(self, name: str) -> int:
        """Width of one channel."""
        return self.channels[self.names.index(self._known(name))][1]

    def slice_of(self, name: str) -> slice:
        """Slice covering one channel within the flat vector."""
        start = 0
        for channel, width in self.channels:
            if channel == self._known(name):
                return slice(start, start + width)
            start += width
        raise KeyError(name)  # unreachable: _known already validated

    def indices_of(self, names: Sequence[str]) -> np.ndarray:
        """Flat-vector indices covering several channels, in the order given."""
        parts = [np.arange(*self.slice_of(name).indices(self.total_dim)) for name in names]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=int)

    def block(self, vector: ArrayLike, name: str) -> np.ndarray:
        """Slice one channel out of a flat vector, preserving any batch dimension."""
        return np.asarray(vector, dtype=float)[..., self.slice_of(name)]

    def split(self, vector: ArrayLike) -> Dict[str, np.ndarray]:
        """Split a flat vector into its ``{name: block}`` mapping."""
        array = np.asarray(vector, dtype=float)
        return {name: array[..., self.slice_of(name)] for name in self.names}

    def pack(self, blocks: Mapping[str, ArrayLike]) -> np.ndarray:
        """Concatenate a ``{name: block}`` mapping into a flat vector, in packing order.

        Args:
            blocks: One entry per channel. Batched entries (shape ``(N, width)``) may be mixed
                with unbatched ones; an unbatched block is broadcast across the batch, which is
                what lets a per-episode latent block ride along with a batch of sampled states.

        Returns:
            The flat vector, shape ``(total_dim,)`` or ``(N, total_dim)``.

        Raises:
            KeyError: If a channel is missing from ``blocks``.
            ValueError: If a block's width disagrees with the schema, or two batched blocks
                disagree in length.
        """
        arrays = [self._checked_block(name, blocks) for name in self.names]
        batch = {array.shape[0] for array in arrays if array.ndim == 2}
        if len(batch) > 1:
            raise ValueError(f"batched blocks disagree in length: {sorted(batch)}")
        if not batch:
            return np.concatenate(arrays, axis=-1)
        size = batch.pop()
        widened = [
            array if array.ndim == 2 else np.broadcast_to(array, (size, array.shape[0]))
            for array in arrays
        ]
        return np.concatenate(widened, axis=-1)

    def _checked_block(self, name: str, blocks: Mapping[str, ArrayLike]) -> np.ndarray:
        if name not in blocks:
            raise KeyError(f"channel {name!r} missing from the blocks to pack")
        array = np.asarray(blocks[name], dtype=float)
        if array.ndim > 2:
            raise ValueError(f"channel {name!r} block must be 1-D or 2-D, got {array.ndim}-D")
        if array.shape[-1] != self.width(name):
            raise ValueError(
                f"channel {name!r} is {self.width(name)} wide in the schema but the block is "
                f"{array.shape[-1]}"
            )
        return array

    def _known(self, name: str) -> str:
        if name not in self.names:
            raise KeyError(f"unknown channel {name!r}; schema has {list(self.names)}")
        return name


class IsaacModelPOMDP(DiscreteActionsEnvironment):
    """Abstract generative-model interface paired with the forward-only IsaacLab world.

    This base owns the schema shared by every Isaac model — the named state blocks, the finite
    action set, observation-dict hashing and equality — *and* the observation, which it composes
    from a ``{channel: IsaacObservationModel}`` map: :meth:`sample_observation` perceives each
    channel from the state's clean blocks, :meth:`observation_log_probability` sums the per-channel
    densities, and :meth:`encode_observation` maps the world's raw reading into the same perceived
    space. Two models differ in their observation only by the channel models they hold. A model
    that scores observations for a belief update needs every channel to provide a density
    (``supports_density``).

    Concrete subclasses supply the dynamics: :meth:`sample_next_state`,
    :meth:`transition_log_probability`, :meth:`reward` and :meth:`is_terminal`.

    Attributes:
        state_schema: Named blocks of the flat state vector.
        action_presets: The finite set of action vectors the planner chooses among.
        observation_models: The ``{channel: IsaacObservationModel}`` map, or ``None`` for a
            subclass that overrides the observation methods directly.
        raw_observation_schema: Named blocks of the world's *flat* raw observation, used by
            :meth:`encode_observation` to split it into channels. ``None`` when the world already
            emits a channel mapping.

    Note:
        This is an abstract base class and cannot be instantiated directly. See
        :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_factored_model.FactoredIsaacModelPOMDP`
        for a concrete reference implementation.
    """

    def __init__(
        self,
        state_schema: IsaacChannelSchema,
        action_presets: Sequence[ArrayLike],
        discount_factor: float,
        observation_models: Optional[Mapping[str, IsaacObservationModel]] = None,
        raw_observation_schema: Optional[IsaacChannelSchema] = None,
        reward_range: Optional[Tuple[float, float]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the factored Isaac generative-model interface.

        Args:
            state_schema: Named blocks of the flat state vector.
            action_presets: Finite list of continuous action vectors to plan over.
            discount_factor: POMDP discount factor (shared with the world).
            observation_models: ``{channel: IsaacObservationModel}``. ``None`` (the default) is
                for subclasses that override the observation methods directly.
            raw_observation_schema: Named blocks of the world's flat raw observation. Supply it
                when the world emits a vector rather than a channel mapping.
            reward_range: Optional ``(min, max)`` reward bounds. Worth setting for a constrained
                planner, which reads it to cap the Lagrange multiplier.
            name: Model name, also used to label planner output.

        Raises:
            ValueError: If an observation model reads a state channel the schema does not declare.
        """
        super().__init__(
            discount_factor=discount_factor,
            name=name if name is not None else type(self).__name__,
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE,
                observation_space=SpaceType.CONTINUOUS,
            ),
            reward_range=reward_range,
        )
        self.state_schema = state_schema
        self.action_presets = [np.asarray(a, dtype=float).reshape(-1) for a in action_presets]
        self.observation_models: Optional[Dict[str, IsaacObservationModel]] = (
            dict(observation_models) if observation_models is not None else None
        )
        self.raw_observation_schema = raw_observation_schema
        self._validate_observation_models()

    def _validate_observation_models(self) -> None:
        """Fail at construction when a channel reads a block the schema has no name for.

        Left to run time this surfaces as a ``KeyError`` from inside the search tree, thousands of
        simulations into an episode, with no indication of which channel was misconfigured.
        """
        if self.observation_models is None:
            return
        known = set(self.state_schema.names)
        missing = {
            channel: sorted(set(model.state_channels) - known)
            for channel, model in self.observation_models.items()
            if set(model.state_channels) - known
        }
        if missing:
            raise ValueError(
                f"observation channels read state blocks the schema does not declare: {missing}; "
                f"schema has {sorted(known)}"
            )

    # ── Schema (concrete, shared by every model) ────────────────────────
    def get_actions(self) -> List[np.ndarray]:
        """Return the finite set of action vectors the planner chooses among."""
        return [preset.copy() for preset in self.action_presets]

    def clean_state(self, state: Any) -> Dict[str, np.ndarray]:
        """Split a state into the named blocks the observation channels read.

        Args:
            state: A flat state vector.

        Returns:
            The ``{channel: block}`` mapping handed to each observation model.
        """
        return self.state_schema.split(state)

    def seed(self, seed: Optional[int]) -> None:
        """Reseed every observation channel that carries its own sampler.

        Args:
            seed: Seed forwarded to each channel's ``seed`` method. Channels that draw from the
                global numpy generator have no such method and are skipped.
        """
        for model in (self.observation_models or {}).values():
            channel_seed = getattr(model, "seed", None)
            if callable(channel_seed):
                channel_seed(seed)

    # ── Observation (per-channel perception composed over the clean state) ──
    def encode_observation(self, observation: Any) -> Any:
        """Map the world's raw observation into the belief/planner observation space.

        This is the single raw-observation seam; every other observation method works in the
        encoded space. A raw mapping is taken channel by channel; a raw flat vector is split by
        ``raw_observation_schema`` first. Each channel's own
        :meth:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_model.IsaacObservationModel.encode`
        decides what happens next — the identity for a real sensor reading, which is what an Isaac
        world emits.

        Args:
            observation: The raw observation emitted by the world, either a ``{channel: value}``
                mapping or a flat vector.

        Returns:
            The perceived observation the belief and planner consume.

        Raises:
            RuntimeError: If the raw observation is a flat vector and no
                ``raw_observation_schema`` was supplied to split it.
        """
        if self.observation_models is None:
            return super().encode_observation(observation)
        raw = self._raw_channels(observation)
        return {
            channel: model.encode(raw[channel])
            for channel, model in self.observation_models.items()
        }

    def _raw_channels(self, observation: Any) -> Mapping[str, Any]:
        if isinstance(observation, Mapping):
            return observation
        if self.raw_observation_schema is None:
            raise RuntimeError(
                f"{type(self).__name__} received a flat raw observation but has no "
                "raw_observation_schema to split it into channels; supply one, or have the "
                "world's observation_extractor emit a {channel: value} mapping."
            )
        return self.raw_observation_schema.split(observation)

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        """Sample ``n_samples`` observations by perceiving ``next_state``'s clean blocks.

        Args:
            next_state: The state being observed.
            action: Unused; the observation depends on the resulting state alone.
            n_samples: Number of independent observations to draw.

        Returns:
            One ``{channel: value}`` mapping when ``n_samples == 1``, else a list of them. The
            draws are independent, not one draw repeated.
        """
        del action
        models = self._require_observation_models()
        clean = self.clean_state(next_state)
        draws = [
            {channel: model.perceive(clean) for channel, model in models.items()}
            for _ in range(n_samples)
        ]
        return draws[0] if n_samples == 1 else draws

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        """Log-density of ``observations`` under the per-channel perception given ``next_state``.

        Args:
            next_state: The state being observed.
            action: Unused; the observation depends on the resulting state alone.
            observations: One observation mapping or a list of them.

        Returns:
            A ``(n,)`` array of log-densities, one per observation.
        """
        del action
        models = self._require_density_models()
        clean = self.clean_state(next_state)
        listed = observations if isinstance(observations, list) else [observations]
        return np.array(
            [
                float(
                    sum(
                        model.log_probability(clean, observation[channel])
                        for channel, model in models.items()
                    )
                )
                for observation in listed
            ]
        )

    def _require_observation_models(self) -> Dict[str, IsaacObservationModel]:
        if self.observation_models is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no observation models; supply a "
                "{channel: IsaacObservationModel} map or override sample_observation."
            )
        return self.observation_models

    def _require_density_models(self) -> Dict[str, IsaacObservationModel]:
        models = self._require_observation_models()
        sample_only = [channel for channel, model in models.items() if not model.supports_density]
        if sample_only:
            raise NotImplementedError(
                f"Observation channels {sample_only} are sample-only (no density); belief "
                "updates need every channel to score observations, or override "
                "observation_log_probability."
            )
        return models

    # ── Hashing / equality over the observation mapping ─────────────────
    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        if set(observation1) != set(observation2):
            return False
        return all(
            np.array_equal(np.asarray(observation1[key]), np.asarray(observation2[key]))
            for key in observation1
        )

    def hash_observation(self, observation: Any) -> Hashable:
        return tuple(
            (key, np.asarray(observation[key], dtype=float).tobytes())
            for key in sorted(observation)
        )

    def hash_action(self, action: Any) -> Hashable:
        return np.asarray(action).tobytes()

    # ── Initial distributions (the world seeds the belief) ──────────────
    def initial_state_dist(self) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} has no initial-state prior; seed the belief from the "
            "world's initial observation."
        )

    def initial_observation_dist(self) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} has no initial-observation prior; seed the belief from "
            "the world's initial observation."
        )

    # ── Dynamics (abstract — the interface a model must implement) ──────
    @abstractmethod
    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        """Sample ``n_samples`` next states for ``(state, action)``."""

    @abstractmethod
    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        """Log-density of ``next_states`` under the transition for ``(state, action)``."""
