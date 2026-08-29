# SPDX-License-Identifier: MIT

"""Module for POMDP environment abstractions.

This module provides the foundational classes and interfaces for defining
POMDP environments, including abstract base classes for state transitions,
observation models, and reward functions.

Classes:
    Environment: Abstract base class for POMDP environments
    DiscreteActionsEnvironment: Specialized for discrete action spaces
    ObservationModel: Abstract observation model interface
    StateTransitionModel: Abstract state transition interface
    EnvironmentGenerator: Factory pattern for environment creation
    SpaceType: Enumeration for action/observation space types
    SpaceInfo: Data class containing space type information
"""

# pylint: disable=too-many-lines  # foundational module; split tracked separately

import importlib
import inspect
import logging
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.serialization import (
    deserialize_value as deserialize_value_base,
    register_deserializer,
    register_serializer,
    serialize_value as serialize_value_base,
)
from POMDPPlanners.utils.config_to_id import config_to_id
from POMDPPlanners.utils.logger import get_logger

if TYPE_CHECKING:
    from POMDPPlanners.core.simulation import History, MetricValue, StepData
    from POMDPPlanners.core.simulation.step_info_metrics import StepInfoMetric


# Cap on how deep __eq__ descends into sub-objects that define no equality of
# their own, so a cyclic back-reference cannot recurse forever.
_MAX_EQ_RECURSION_DEPTH = 6


def _serialize_space_info(space_info: Any) -> dict:
    """Serialize SpaceInfo to plain dict without type markers.

    Maintains backward compatibility with existing saved environments.
    Format: {"action_space": "discrete", "observation_space": "continuous"}

    Args:
        space_info: SpaceInfo instance to serialize

    Returns:
        Plain dict with action_space and observation_space string values
    """
    return {
        "action_space": space_info.action_space.value,
        "observation_space": space_info.observation_space.value,
    }


def _deserialize_space_info(data: dict) -> Any:
    """Deserialize SpaceInfo from plain dict format.

    Handles dicts with action_space and observation_space keys without
    requiring __type__ markers for backward compatibility.

    Args:
        data: Dict with action_space and observation_space keys

    Returns:
        SpaceInfo instance

    Raises:
        ValueError: If data cannot be deserialized to SpaceInfo
    """
    if isinstance(data, dict) and "action_space" in data and "observation_space" in data:
        # Import SpaceType here to avoid circular dependency
        return SpaceInfo(
            action_space=SpaceType(data["action_space"]),
            observation_space=SpaceType(data["observation_space"]),
        )
    raise ValueError(f"Cannot deserialize SpaceInfo from {data}")


def _is_config_callable(value: Any) -> bool:
    """Whether a callable attribute is configuration rather than a memoized shortcut.

    A class or plain function named in a config is a real choice, and its
    qualified name identifies it stably. Anything else callable — a bound
    method, a builtin, a pybind11 method — is a fast path an environment
    memoizes onto itself on first use, and must not reach either
    :meth:`Environment.config_id` or :meth:`Environment.__eq__`: it is not
    part of what the environment *is*, and its ``repr`` embeds a memory
    address.

    Args:
        value: The attribute value to classify.

    Returns:
        ``True`` if the callable is part of the configuration.
    """
    return isinstance(value, type) or inspect.isfunction(value)


def _config_attributes(obj: Any) -> Dict[str, Any]:
    """Return the attributes of ``obj`` that make up its configuration identity.

    Shared by :meth:`Environment.config_id` and :meth:`Environment.__eq__` when
    they descend into a sub-object, so the two answers to "is this the same
    environment" are computed over the same surface and cannot drift apart.

    Args:
        obj: The object whose ``__dict__`` is being inspected.

    Returns:
        The configuration-bearing attributes, keyed by name.
    """
    return {
        key: value
        for key, value in vars(obj).items()
        if not isinstance(value, logging.Logger)
        and not (callable(value) and not _is_config_callable(value))
    }


class SpaceType(Enum):
    """Enumeration for categorizing action and observation spaces.

    This enum is used to classify the mathematical structure of action
    and observation spaces in POMDP environments.

    Attributes:
        DISCRETE: Finite, countable spaces (e.g., {0, 1, 2, ...})
        CONTINUOUS: Real-valued continuous spaces (e.g., R^n)
        MIXED: Combination of discrete and continuous elements
    """

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MIXED = "mixed"


@dataclass
class SpaceInfo:
    """Data class containing space type information for an environment.

    This class encapsulates the space type classifications for both
    actions and observations in a POMDP environment.

    Attributes:
        action_space: The type of action space (discrete, continuous, or mixed)
        observation_space: The type of observation space (discrete, continuous, or mixed)

    Example:
        Creating space info for different environment types:

        >>> # Discrete actions, continuous observations
        >>> space_info = SpaceInfo(
        ...     action_space=SpaceType.DISCRETE,
        ...     observation_space=SpaceType.CONTINUOUS
        ... )
    """

    action_space: SpaceType
    observation_space: SpaceType


# Register SpaceInfo serialization handlers at module load time
# This enables centralized serialization system to handle SpaceInfo automatically
register_serializer(SpaceInfo, _serialize_space_info)
register_deserializer(SpaceInfo, _deserialize_space_info)


class Environment(ABC):  # pylint: disable=too-many-public-methods
    """Abstract base class for POMDP environments.

    This is the core abstract class that all POMDP environments must inherit from.
    It defines the essential interface for POMDP environments including state
    transitions, observations, rewards, and terminal conditions.

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses must implement all abstract methods.

    Attributes:
        discount_factor: Discount factor for future rewards
        name: Environment identifier string
        space_info: Information about action and observation space types
        reward_range: Optional tuple containing (min_reward, max_reward)
        output_dir: Optional directory for logging output
        debug: Flag to enable debug logging
    """

    def __init__(
        self,
        discount_factor: float,
        name: str,
        space_info: SpaceInfo,
        reward_range: Optional[Tuple[float, float]] = None,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the POMDP environment.

        Args:
            discount_factor: Discount factor for future rewards (0 < discount_factor <= 1)
            name: Unique identifier for the environment
            space_info: Information about action and observation space types
            reward_range: Optional tuple containing (min_reward, max_reward) for the environment.
                Defaults to None. If provided, will be validated.
            output_dir: Optional directory for logging output. Defaults to None.
            debug: Enable debug logging. Defaults to False.
            use_queue_logger: Whether to use queue-based logging. Defaults to True.
        """
        self.discount_factor = discount_factor
        self.name = name
        self.space_info = space_info
        self.reward_range = self._validate_reward_range(reward_range)
        self.output_dir = output_dir
        self.debug = debug
        self.use_queue_logger = use_queue_logger

        self.logger.info(
            "Initializing %s environment with discount factor %s", self.name, self.discount_factor
        )
        self.logger.debug(
            "Space info: action_space=%s, observation_space=%s",
            self.space_info.action_space,
            self.space_info.observation_space,
        )
        if self.reward_range is not None:
            self.logger.debug("Reward range: %s", self.reward_range)

    def _validate_reward_range(
        self, reward_range: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """Validate reward_range if provided.

        Args:
            reward_range: Optional tuple containing (min_reward, max_reward)

        Returns:
            Validated reward_range tuple or None if input was None

        Raises:
            ValueError: If reward_range structure or values are invalid
            TypeError: If reward_range values are not numeric
        """
        if reward_range is None:
            return None

        # Validate structure
        if not isinstance(reward_range, tuple) or len(reward_range) != 2:
            raise ValueError("reward_range must be a tuple of exactly two float values")

        min_reward, max_reward = reward_range

        # Check that both values are numeric (float or int)
        if not isinstance(min_reward, (int, float)) or not isinstance(max_reward, (int, float)):
            raise TypeError("reward_range values must be numeric (int or float)")

        # Convert to float to ensure consistency
        min_reward, max_reward = float(min_reward), float(max_reward)

        # Check for NaN values
        if np.isnan(min_reward) or np.isnan(max_reward):
            raise ValueError("reward_range values cannot be NaN")

        # Check that min_reward <= max_reward (allowing inf values)
        if min_reward > max_reward:
            raise ValueError(
                f"reward_range minimum ({min_reward}) must be less than or equal to maximum ({max_reward})"
            )

        return (min_reward, max_reward)

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this environment.

        The logger is implemented as a property to maintain pickle compatibility,
        as logger objects cannot be pickled directly.

        Returns:
            Configured logger instance with hierarchical naming
        """
        return get_logger(
            name=f"environment.{self.name}",
            output_dir=self.output_dir,
            debug=self.debug,
            use_queue=self.use_queue_logger,
        )

    def __eq__(self, other):
        if not isinstance(other, Environment):
            return False
        if self.__class__ != other.__class__:
            return False

        def _compare_values(v1, v2, depth=0):  # pylint: disable=too-many-return-statements
            """Helper function to compare values, handling numpy arrays specially."""
            if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
                if not (isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray)):
                    return False
                return np.array_equal(v1, v2)
            if isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
                if len(v1) != len(v2):
                    return False
                return all(_compare_values(x1, x2, depth) for x1, x2 in zip(v1, v2))
            if isinstance(v1, dict) and isinstance(v2, dict):
                if v1.keys() != v2.keys():
                    return False
                return all(_compare_values(v1[k], v2[k], depth) for k in v1)
            # Enum members define no __eq__ of their own, so without this they
            # would fall into the structural branch below and be compared by
            # walking __objclass__ — the enum class, whose _member_map_ leads
            # back to every member and to the class again. Correct but
            # needlessly expensive, and config_id compares an Enum by its
            # value long before it reaches its __dict__. Identity is the right
            # comparison for a member of one enum class.
            if isinstance(v1, Enum) or isinstance(v2, Enum):
                return v1 == v2
            # A class or plain function named in a config is compared the way
            # config_id serializes it: by qualified name. Structural comparison
            # would call every distinct function equal, since ``vars(fn)`` is
            # empty for essentially all of them — and "equal but hashing
            # differently" is the one direction that actually breaks dict and
            # set deduplication.
            if _is_config_callable(v1) or _is_config_callable(v2):
                if not (_is_config_callable(v1) and _is_config_callable(v2)):
                    return False
                return (v1.__module__, v1.__qualname__) == (v2.__module__, v2.__qualname__)
            # A sub-object built deterministically from this env's own config
            # (a reward model, an observation model) carries no identity of its
            # own. Left to ``v1 == v2`` it would compare by *identity*, because
            # those classes define no __eq__ — so two envs built from identical
            # kwargs would never be equal, and neither would an env and its own
            # from_dict rebuild. config_id already compares such sub-objects
            # structurally; do the same here, over the same attribute surface,
            # so equality and the config hash agree. The depth cap stops a
            # cyclic back-reference from recursing forever.
            if (
                depth < _MAX_EQ_RECURSION_DEPTH
                and type(v1) is type(v2)
                and hasattr(v1, "__dict__")
                and type(v1).__eq__ is object.__eq__
            ):
                return _compare_values(
                    _config_attributes(v1), _config_attributes(v2), depth + 1
                )
            return v1 == v2

        # Compare all public attributes (excluding callables and private)
        for key, value in self.__dict__.items():
            if key.startswith("_") or callable(value):
                continue
            if not hasattr(other, key):
                return False
            other_value = getattr(other, key)
            if not _compare_values(value, other_value):
                return False

        # Check for any attributes in other that aren't in self
        for key in other.__dict__:
            if key.startswith("_") or callable(getattr(other, key)):
                continue
            if not hasattr(self, key):
                return False

        return True

    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on environment configuration.

        Note:
            Uses custom serialization logic (not centralized serialize_value) to ensure:
            - Deterministic dict key ordering for consistent hashing
            - Compact format without __type__ markers
            - Recursive handling of nested objects
            Changing this serialization format would invalidate all cached results.
        """

        def serialize_value(value):  # pylint: disable=too-many-return-statements
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            if isinstance(value, dict):
                serialized_items = {}
                for k, v in sorted(value.items()):
                    if isinstance(v, logging.Logger):
                        continue
                    if callable(v):
                        # See _is_config_callable: a class or plain function is
                        # config and gets a stable qualified name; a memoized
                        # bound / native method is not, and would otherwise
                        # reach ``str(value)`` and embed a memory address.
                        if _is_config_callable(v):
                            serialized_items[str(k)] = f"{v.__module__}.{v.__qualname__}"
                        continue
                    serialized_items[str(k)] = serialize_value(v)
                return serialized_items
            if isinstance(value, SpaceInfo):
                return {
                    "action_space": serialize_value(value.action_space),
                    "observation_space": serialize_value(value.observation_space),
                }
            if isinstance(value, Enum):
                return value.value
            if hasattr(value, "__dict__"):
                # Skip logger objects
                if isinstance(value, logging.Logger):
                    return None
                return serialize_value(value.__dict__)
            return str(value)

        config_dict = {}
        for key, value in self.__dict__.items():
            # Skip logger and private attributes
            if key.startswith("_") or callable(value) or isinstance(value, logging.Logger):
                continue
            serialized_value = serialize_value(value)
            if serialized_value is not None:  # Skip None values (like logger)
                config_dict[key] = serialized_value
        config_dict = dict(sorted(config_dict.items()))
        return config_to_id(config_dict)

    def __hash__(self) -> int:
        return hash(self.config_id)

    @property
    def reward_requires_next_state(self) -> bool:
        """Whether :meth:`reward` must be given the realised ``next_state``.

        Simulation and rollout drivers consult this hook to decide the order
        in which they sample the transition and compute the reward:

        - ``False`` (default): the reward is a pure function of
          ``(state, action)``. Drivers compute ``reward(state, action)``
          *before* sampling the transition, preserving the historical
          RNG-draw interleaving so seeded trajectories stay bit-identical.
        - ``True``: the reward depends on the realised post-transition
          state (e.g. draw-coupled hazard termination). Drivers sample the
          transition first, then call ``reward(state, action, next_state)``
          with the realised next state so both consume the same draw.

        Subclasses whose reward becomes next-state dependent (e.g. when a
        draw-coupled ``is_*_hit_terminal`` flag is enabled) must override
        this to return ``True`` only in that configuration; otherwise the
        default keeps flag-off behaviour bit-identical to today.

        Returns:
            ``True`` if :meth:`reward` needs the realised next state,
            ``False`` otherwise.
        """
        return False

    # pylint: disable-next=unused-argument
    def step_info(self, state: Any, action: Any, next_state: Any) -> Dict[str, float]:
        """Report auxiliary measurements for the transition just taken.

        Simulation drivers call this once per recorded step and store the result
        on :attr:`~POMDPPlanners.core.simulation.history.StepData.info`, so the
        values travel back with the episode ``History``.

        This exists because a quantity may only be *measurable* at step time —
        an impact impulse read from a physics engine, a termination reason from a
        task manager — while
        :meth:`compute_metrics` runs afterwards, in the parent process, on a
        different instance of this environment. Accumulating such a value on
        ``self`` during an episode is therefore silently lost under every
        multiprocess task manager; routing it through the returned mapping is
        what makes it survive.

        Environments whose measurements are a pure function of the transition
        should implement this as such. Environments wrapping a live simulator may
        instead serve values cached during their own step, in which case the
        arguments are typically used only to assert the request matches the
        transition actually taken.

        This is also called once more per *terminated* episode, for the terminal
        bookkeeping step, with ``action`` and ``next_state`` both ``None`` — that
        step records the final state, which no transition ever produces a
        successor for. Implementations must therefore tolerate ``None`` and
        report a neutral value for any channel that describes the transition
        rather than the state. Predicates written as
        ``float(action == <something>)`` already do this, since ``None`` matches
        nothing. Channels measured from ``state`` alone should still be reported,
        because a metric that counts every visited state needs the final one.

        Args:
            state: The state the step was taken from, or the final state on the
                terminal bookkeeping step.
            action: The action taken, or ``None`` on the terminal step.
            next_state: The realised successor state, or ``None`` on the terminal
                step.

        Returns:
            A flat mapping of channel name to scalar, e.g.
            ``{"success": 1.0, "impact": 12.4}``. Values must be plain picklable
            scalars. The default implementation reports nothing.

        Note:
            Name channels after the quantity and unit actually measured (e.g.
            ``"contact_impulse_ns"``), not after a category. There is no shared
            cross-environment channel vocabulary on purpose: a name like
            ``"impact"`` could mean a force, an impulse, an energy or a count,
            and a shared name would imply a comparability that does not hold.

        Warning:
            Implementations must be side-effect-free and must not consume
            randomness. This runs inside the episode loop, between the transition
            and the belief update, so a single ``np.random`` draw here shifts the
            stream for every subsequent transition and observation — silently
            changing seeded trajectories throughout the run, not just the
            metrics. Where a value would have to be resampled to be reported, do
            not report it.
        """
        return {}

    @abstractmethod
    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        """Calculate the immediate reward for a state-action(-next_state) tuple.

        ``next_state`` is the realised post-transition state when known
        (e.g. threaded by :meth:`sample_next_step`), allowing rewards that
        depend on stochastic transition outcomes to use the same draw as
        the trajectory instead of resampling. Subclasses whose reward is
        a pure function of ``(state, action)`` may ignore it; subclasses
        whose reward depends on the realised next state (collision
        penalties, win bonuses) should consume it when provided and fall
        back to drawing/computing one when ``None``.

        Args:
            state: Current state.
            action: Action executed from ``state``.
            next_state: Realised next state, or ``None`` if the caller
                did not pre-sample one. Defaults to ``None``.

        Returns:
            Immediate reward value.

        Note:
            Subclasses must implement this method to define reward structure.
        """

    def reward_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: Any,
        next_states: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    ) -> np.ndarray:
        """Calculate rewards for a batch of states given a single action.

        Provides a loop-based default that subclasses can override with
        vectorized numpy implementations for better performance.

        Args:
            states: Sequence of states of length ``N``.
            action: Action executed from each state.
            next_states: Optional realised next states (length ``N``)
                threaded through to :meth:`reward`. Defaults to ``None``.

        Returns:
            1-D array of reward values with shape ``(N,)``.
        """
        if next_states is None:
            return np.array([self.reward(states[i], action) for i in range(len(states))])
        return np.array(
            [self.reward(states[i], action, next_states[i]) for i in range(len(states))]
        )

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        """Check if a state is terminal.

        Args:
            state: State to check for terminal condition

        Returns:
            True if the state is terminal, False otherwise

        Note:
            Subclasses must implement this method to define terminal conditions.
        """

    @abstractmethod
    def initial_state_dist(self) -> Distribution:
        """Get the initial state distribution.

        Returns:
            Distribution over initial states

        Note:
            Subclasses must implement this method to define the starting distribution.
        """

    @abstractmethod
    def initial_observation_dist(self) -> Distribution:
        """Get the initial observation distribution.

        Returns:
            Distribution over initial observations

        Note:
            Subclasses must implement this method to define initial observations.
        """

    @abstractmethod
    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        """Check if two observations are equal.

        Args:
            observation1: First observation to compare
            observation2: Second observation to compare

        Returns:
            True if observations are considered equal, False otherwise

        Note:
            Subclasses must implement this method to define observation equality.
            This is particularly important for discrete observation spaces.
        """

    def hash_observation(self, observation: Any) -> Hashable:
        """Return a hashable key consistent with :meth:`is_equal_observation`.

        Used by tree-search planners to index belief children by observation
        in O(1). The returned key MUST satisfy the contract::

            is_equal_observation(a, b) implies hash_observation(a) == hash_observation(b)

        Args:
            observation: Observation to hash.

        Returns:
            A hashable key derived from ``observation`` (default: the
            observation itself when it is already hashable).

        Raises:
            NotImplementedError: If the observation is not hashable and the
                subclass has not provided an override. Subclasses with
                non-hashable observations (e.g. ``np.ndarray``) MUST override.
        """
        try:
            hash(observation)
        except TypeError as exc:
            raise NotImplementedError(
                f"{type(self).__name__} must override hash_observation "
                "for non-hashable observations"
            ) from exc
        return observation

    @abstractmethod
    def hash_action(self, action: Any) -> Hashable:
        """Return a hashable key consistent with action equality.

        Used by tree-search planners to index action children of a belief
        node in O(1). The returned key MUST satisfy::

            action_a == action_b   (per env's notion of equality)
            ==> hash_action(action_a) == hash_action(action_b)

        Subclasses with non-hashable actions (e.g. ``np.ndarray``) must
        override to return a hashable surrogate (``tobytes()`` is the
        standard choice for ndarray actions, which mirrors the
        ``np.array_equal`` semantics used by the linear-scan fallback).

        Args:
            action: Action to hash.

        Returns:
            A hashable key derived from ``action``.
        """

    @abstractmethod
    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> Any:
        """Sample one or more next states for ``(state, action)``.

        Hot-path entry point used by MCTS planners and particle filters.
        Subclasses must implement.

        Returns:
            When ``n_samples == 1``: a single next state of the env's
            native type. When ``n_samples > 1``: an array-like of length
            ``n_samples`` (numeric envs return ``np.ndarray`` of shape
            ``(n_samples, *dim)``; structured envs return ``List[T]``).
        """

    @abstractmethod
    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        """Sample one or more observations for ``(next_state, action)``.

        Hot-path entry point used by MCTS planners and particle filters.
        Subclasses must implement.

        Returns:
            When ``n_samples == 1``: a single observation. When
            ``n_samples > 1``: an array-like of length ``n_samples``.
        """

    @abstractmethod
    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        """Log-probability of each candidate next state under ``(state, action)``.

        Returns ``np.ndarray`` of shape ``(N,)`` where N is the number of
        candidate next states. Subclasses must implement.
        """

    @abstractmethod
    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        """Log-probability of each candidate observation under ``(next_state, action)``.

        Returns ``np.ndarray`` of shape ``(N,)`` where N is the number of
        candidate observations. Subclasses must implement.
        """

    def observation_log_probability_single(
        self, next_state: Any, action: Any, observation: Any
    ) -> float:
        """Scalar log-likelihood for one ``(next_state, observation)`` pair.

        Per-state fast-path used by incremental belief updates
        (e.g. POMCPOW's :meth:`WeightedParticleBeliefStateUpdate.inplace_update`)
        to skip the per-call numpy setup overhead of the batched
        :meth:`observation_log_probability` path on a singleton input.

        The default falls back to the batched method with a one-element
        observations list. Envs with cheap scalar likelihoods (e.g. the
        2-D Gaussian on Push or the cached-inverse-cov path on
        ContinuousLightDark) should override to skip array allocation.
        """
        arr = self.observation_log_probability(
            next_state=next_state, action=action, observations=[observation]
        )
        return float(arr[0])

    def sample_next_state_batch(self, states: Any, action: Any) -> Any:
        """Sample one next state per input state, all under the same action.

        Used by particle filters: given N current particles and one action,
        draw N next states (one per particle) in a single vectorized call.

        The default implementation falls back to a per-state Python loop
        delegating to :meth:`sample_next_state`. Native-backed envs (those
        whose state-transition kernel exposes ``batch_sample(states_array)``)
        should override to avoid the loop.

        Args:
            states: A sequence (length N) or ndarray of shape ``(N, *dim)``
                of input particles.
            action: A single action to apply to every particle.

        Returns:
            For numeric envs: ``np.ndarray`` of shape ``(N, *dim)``.
            For structured envs (Tiger strings, Pacman tuples): a list of
            length N.
        """
        return [self.sample_next_state(state=s, action=action) for s in states]

    def observation_log_probability_per_state(
        self, next_states: Any, action: Any, observation: Any
    ) -> np.ndarray:
        """Log-probability of one observation under each candidate next-state.

        Used by particle filters: given N candidate next-states and ONE
        observation, return N log-likelihoods.

        The default implementation falls back to a per-state Python loop
        delegating to :meth:`observation_log_probability`. Native-backed envs
        (those whose observation kernel exposes
        ``batch_log_likelihood(next_states_array, observation_array)``) should
        override to avoid the loop.

        Args:
            next_states: A sequence (length N) or ndarray of shape ``(N, *dim)``
                of candidate next-states.
            action: The action that was executed.
            observation: A single observation.

        Returns:
            ndarray of shape ``(N,)`` with log-probabilities or log-PDFs.
        """
        return np.asarray(
            [
                self.observation_log_probability(
                    next_state=ns, action=action, observations=[observation]
                )[0]
                for ns in next_states
            ]
        )

    def sample_next_step(self, state: Any, action: Any) -> Tuple[Any, Any, float]:
        """Sample a complete state transition step.

        This convenience method combines state transition, observation generation,
        and reward calculation in a single operation.

        Args:
            state: Current state
            action: Action to execute

        Returns:
            Tuple containing:
                - next_state: Sampled next state
                - next_observation: Sampled observation
                - reward: Immediate reward
        """
        next_state = self.sample_next_state(state=state, action=action)
        next_observation = self.sample_observation(next_state=next_state, action=action)
        # Thread the realised next_state into reward() so subclasses with
        # transition-dependent reward terms (collision penalties, win bonuses)
        # score against the same draw as the trajectory rather than resampling.
        # pylint: disable-next=assignment-from-no-return
        reward = self.reward(state=state, action=action, next_state=next_state)

        return next_state, next_observation, reward

    def encode_observation(self, observation: Any) -> Any:
        """Encode a raw observation into the space the belief and planner use.

        The base implementation is the identity: for environments whose raw and
        working observations coincide (the classic single-environment case), the
        observation is returned unchanged. A planner-side model whose working
        observation is an encoding of a richer raw observation (e.g. an image
        encoder the user supplies) overrides this to map the world's raw
        observation into that encoded space.

        This is the *only* method that consumes a raw observation; every other
        observation method (:meth:`sample_observation`,
        :meth:`observation_log_probability`, :meth:`hash_observation`,
        :meth:`is_equal_observation`) operates in the encoded space.

        Args:
            observation: The raw observation emitted by the world.

        Returns:
            The observation in the encoded space the belief and planner use.
        """
        return observation

    def cache_visualization(
        self, history: "List[StepData]", output_dir: Path, episode_index: int
    ) -> None:
        """Cache visualization data for an episode history.

        This method can be overridden by subclasses to provide environment-specific
        visualization caching capabilities. The environment owns the output file
        name and extension: callers provide only the destination directory and the
        episode index, and each environment writes whatever artifact(s) it chooses.

        Args:
            history: List of step data from an episode
            output_dir: Directory into which the visualization file(s) are written
            episode_index: Zero-based index of the episode, used to name the file
        """

    def get_metric_specs(self) -> "List[StepInfoMetric]":
        """Declare metrics derived from this environment's per-step channels.

        Environments that report measurements through :meth:`step_info` can
        declare how those channels become metrics here, and get the aggregation,
        averaging and confidence intervals from the default :meth:`compute_metrics`
        instead of hand-rolling them.

        Only declare a channel this environment actually emits on every step. A
        declared-but-unreported channel yields a metric that is silently dropped,
        which breaks the invariant that declared names match produced names.

        Returns:
            The metric specifications to compute. Empty by default, so
            environments that do not use the per-step channel are unaffected.
        """
        return []

    def get_metric_names(self) -> List[str]:
        """Get names of environment-specific metrics.

        This method returns the names of custom metrics that this environment
        computes in the compute_metrics() method. It enables users to discover
        what metrics are available for hyperparameter optimization.

        Returns:
            List of metric names that this environment produces. Derived from
            :meth:`get_metric_specs`, so declaring a spec is enough; environments
            with a bespoke ``compute_metrics`` override this directly instead.

        Note:
            Subclasses that override compute_metrics() should also override this method
            to return the names of metrics they produce. Use an Enum to ensure consistency
            between the names returned here and the names used in compute_metrics().
        """
        return [spec.name for spec in self.get_metric_specs()]

    def compute_metrics(self, histories: "List[History]") -> "List[MetricValue]":
        """Compute environment-specific metrics from episode histories.

        The default implementation aggregates the per-step channels reported by
        :meth:`step_info` according to :meth:`get_metric_specs`. Subclasses with
        metrics that are not expressible as a per-step channel (for example ones
        needing cross-step reasoning) override this instead.

        Args:
            histories: List of episode histories to analyze

        Returns:
            List of computed metrics with confidence intervals

        Raises:
            ValueError: If ``histories`` is empty — a metric over no episodes is
                an average over nothing, so any value reported for it is
                invented. Also if this environment declares metric specs and any
                episode recorded transition steps while carrying none of the
                declared channels — a history produced before the per-step
                channel existed, or by a runner that does not call
                :meth:`step_info`. Scoring those would report every metric as
                zero rather than as unmeasured.
        """
        # Imported here rather than at module scope: core.simulation pulls in
        # modules that reference Environment, so a top-level import would close
        # a cycle.
        # pylint: disable-next=import-outside-toplevel
        from POMDPPlanners.core.simulation.step_info_metrics import (
            aggregate_step_info_metrics,
            extract_episode_step_infos,
            require_measured_episodes,
            require_non_empty_histories,
        )

        require_non_empty_histories(histories, type(self).__name__)
        specs = self.get_metric_specs()
        if not specs:
            return []
        require_measured_episodes(histories, specs, type(self).__name__)
        return aggregate_step_info_metrics(extract_episode_step_infos(histories), specs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize environment to dictionary format.

        Extracts environment class information and constructor parameters
        to enable JSON serialization and reconstruction.

        Returns:
            Dictionary with structure:
                - class: Full class path (module.ClassName)
                - module: Module name
                - params: Constructor parameters
                - config_id: Deterministic configuration identifier

        Example:
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> env = TigerPOMDP(discount_factor=0.95)
            >>> env_dict = env.to_dict()
            >>> 'class' in env_dict and 'params' in env_dict
            True

        Note:
            Uses centralized serialization system with registered SpaceInfo handler.
        """
        # Get environment class information
        env_class = self.__class__
        env_module = env_class.__module__
        env_class_name = env_class.__name__

        # Extract constructor parameters
        sig = inspect.signature(env_class.__init__)
        params = {}

        for param_name, _ in sig.parameters.items():
            if param_name == "self":
                continue
            if hasattr(self, param_name):
                value = getattr(self, param_name)
                # Use centralized serialization (SpaceInfo handled by registered handler)
                serialized_value = serialize_value_base(value)
                if serialized_value is not None:  # Skip None values (like logger)
                    params[param_name] = serialized_value

        return {
            "class": f"{env_module}.{env_class_name}",
            "module": env_module,
            "params": params,
            "config_id": self.config_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Environment":
        """Reconstruct environment from dictionary.

        Dynamically imports the environment class and instantiates it
        with the saved parameters.

        Args:
            data: Dictionary containing environment serialization data
                with keys: class, module, params, config_id

        Returns:
            Reconstructed environment instance

        Raises:
            ImportError: If environment class cannot be imported
            ValueError: If required data fields are missing
            TypeError: If parameters are invalid for environment constructor

        Example:
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> env = TigerPOMDP(discount_factor=0.95)
            >>> env_dict = env.to_dict()
            >>> reconstructed_env = Environment.from_dict(env_dict)
            >>> reconstructed_env.discount_factor
            0.95
        """

        def deserialize_value(
            value, target_type, param_name=""
        ):  # pylint: disable=too-many-branches
            """Deserialize value with environment-specific handling.

            Handles environment-specific patterns before delegating to centralized system:
            - List[Tuple[...]] / Set[Tuple[...]] for obstacles, rock positions
            - Matrix parameters (covariance matrices) with parameter name detection

            Note:
                SpaceInfo is handled automatically by registered handler in centralized system.
            """
            # Unwrap Optional[T] types first
            unwrapped_type = target_type
            if hasattr(target_type, "__origin__") and target_type.__origin__ is Union:
                # Get non-None type from Optional
                # pylint: disable=unidiomatic-typecheck
                args = [arg for arg in target_type.__args__ if arg is not type(None)]
                if args:
                    unwrapped_type = args[0]

            # Environment-specific pattern: List[Tuple[...]] and Set[Tuple[...]]
            # Used by PushPOMDP (obstacles) and RockSamplePOMDP (rock_positions)
            # Handles multiple serialized formats for compatibility
            if hasattr(unwrapped_type, "__origin__"):
                if unwrapped_type.__origin__ in (list, set):
                    # Check if the element type is a tuple
                    args = getattr(unwrapped_type, "__args__", ())
                    if args and hasattr(args[0], "__origin__") and args[0].__origin__ is tuple:
                        # This is List[Tuple[...]] or Set[Tuple[...]]
                        if isinstance(value, list) and value:
                            # Format 1: Tuple markers like {'__type__': 'tuple', 'values': [x, y]}
                            if isinstance(value[0], dict) and value[0].get("__type__") == "tuple":
                                return [deserialize_value_base(elem, None) for elem in value]

                        # First deserialize the value (might be ndarray marker or plain list)
                        deserialized = deserialize_value_base(value, None)

                        # Format 2: NumPy array shape (2, N) → [(x1,y1), (x2,y2), ...]
                        if isinstance(deserialized, np.ndarray):
                            if deserialized.ndim == 2 and deserialized.shape[0] == 2:
                                return list(zip(deserialized[0], deserialized[1]))
                        # Format 3: 2D list [[x1,x2,...], [y1,y2,...]] → [(x1,y1), ...]
                        elif isinstance(deserialized, list) and deserialized:
                            if len(deserialized) == 2 and isinstance(deserialized[0], list):
                                return list(zip(deserialized[0], deserialized[1]))

            # Environment-specific pattern: Matrix parameter name detection
            # Ensures covariance matrices are always numpy arrays
            matrix_param_names = [
                "noise_cov",
                "_cov",
                "cov_matrix",
                "state_transition_cov_matrix",
                "observation_cov_matrix",
            ]
            if any(name in param_name.lower() for name in matrix_param_names):
                result = deserialize_value_base(value, target_type)
                if not isinstance(result, np.ndarray):
                    result = np.array(result)
                return result

            # Handle numpy array type annotations. Tested against the
            # Optional-unwrapped type: on Python 3.10 get_type_hints still
            # auto-wraps ``x: np.ndarray = None`` into Optional[np.ndarray],
            # which would otherwise slip past this branch and come back as a
            # plain list.
            if unwrapped_type == np.ndarray or (
                hasattr(unwrapped_type, "__name__") and "ndarray" in unwrapped_type.__name__
            ):
                result = deserialize_value_base(value, unwrapped_type)
                if not isinstance(result, np.ndarray):
                    result = np.array(result)
                return result

            # Delegate to centralized deserialization for all other types
            return deserialize_value_base(value, target_type)

        # Validate required fields
        if "class" not in data or "module" not in data or "params" not in data:
            raise ValueError("Environment data missing required fields: class, module, or params")

        # Import environment class dynamically
        module_name = data["module"]
        class_name = data["class"].split(".")[-1]

        try:
            module = importlib.import_module(module_name)
            env_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import environment class {data['class']}: {str(e)}"
            ) from e

        # Deserialize parameters with type hints. Annotations are resolved
        # rather than read off the signature: a module using postponed
        # evaluation (``from __future__ import annotations``) hands back the
        # annotation as a *string*, which every type-directed branch below
        # silently fails to match — so e.g. a List[Tuple[...]] parameter would
        # come back as raw {"__type__": "tuple"} envelopes the constructor
        # rejects. Unresolvable forward references fall back to the raw
        # annotation, which is no worse than before.
        sig = inspect.signature(env_class.__init__)
        try:
            resolved_hints = typing.get_type_hints(env_class.__init__)
        except Exception:  # pylint: disable=broad-except
            resolved_hints = {}
        params = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name in data["params"]:
                value = data["params"][param_name]
                annotation = resolved_hints.get(param_name, param.annotation)
                # Try to deserialize with type annotation if available
                if annotation != inspect.Parameter.empty:
                    value = deserialize_value(value, annotation, param_name)
                else:
                    value = deserialize_value(value, type(value), param_name)
                params[param_name] = value

        # Reconstruct environment
        try:
            return env_class(**params)
        except TypeError as e:
            raise TypeError(
                f"Failed to construct {class_name} with params {params}: {str(e)}"
            ) from e


class DiscreteActionsEnvironment(Environment):
    """Abstract base class for POMDP environments with discrete action spaces.

    This class extends the base Environment class with additional functionality
    specific to environments that have finite, enumerable action sets.

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses must implement all abstract methods from Environment plus
        the get_actions() method.
    """

    def __init__(
        self,
        discount_factor: float,
        name: str,
        space_info: SpaceInfo,
        reward_range: Optional[Tuple[float, float]] = None,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the discrete actions environment.

        Args:
            discount_factor: Discount factor for future rewards (0 < discount_factor <= 1)
            name: Unique identifier for the environment
            space_info: Information about action and observation space types
            reward_range: Optional tuple containing (min_reward, max_reward) for the environment.
                Defaults to None. If provided, will be validated.
            output_dir: Optional directory for logging output. Defaults to None.
            debug: Enable debug logging. Defaults to False.
        """
        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=reward_range,
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )
        self.logger.debug("Initialized DiscreteActionsEnvironment")

    @abstractmethod
    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        pass

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        pass

    @abstractmethod
    def initial_state_dist(self) -> Distribution:
        pass

    @abstractmethod
    def initial_observation_dist(self) -> Distribution:
        pass

    @abstractmethod
    def get_actions(self) -> List[Any]:
        """Get all possible actions in the discrete action space.

        Returns:
            List containing all valid actions that can be executed

        Note:
            Subclasses must implement this method to enumerate all possible actions.
            This is used by planning algorithms that need to iterate over actions.
        """

    @abstractmethod
    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        pass


class EnvironmentGenerator(ABC):
    """Abstract base class for environment generators.

    This class implements the factory pattern for creating environment instances.
    It's useful for generating environments with randomized parameters or
    for creating multiple environment variants.

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses must implement the generate_environment() method.

    Attributes:
        name: Identifier for the generator
    """

    def __init__(self, name: str):
        """Initialize the environment generator.

        Args:
            name: Unique identifier for this generator
        """
        self.name = name

    @abstractmethod
    def generate_environment(self) -> Environment:
        """Generate a new environment instance.

        Returns:
            Newly created environment instance

        Note:
            Subclasses must implement this method to define environment creation logic.
            This may involve randomization, parameter sampling, or deterministic generation.
        """
