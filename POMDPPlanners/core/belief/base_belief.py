"""Abstract base class for POMDP belief state representations.

This module provides the foundational Belief abstract base class that defines
the interface for all belief state representations in POMDP environments.

Classes:
    Belief: Abstract base class for belief representations
"""

import inspect
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.utils.config_to_id import config_to_id


class Belief(ABC):
    """Abstract base class for POMDP belief state representations.

    This class defines the interface for belief states in POMDP environments.
    Belief states represent probability distributions over the state space,
    capturing the agent's uncertainty about the current state.

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses must implement the update() and sample() methods.
    """

    @classmethod
    def from_config(cls, config):
        """Create a belief instance from configuration.

        Factory method that dynamically creates belief instances based on
        configuration objects specifying the class name and parameters.

        Args:
            config: Configuration object with class_name and params attributes

        Returns:
            New belief instance of the specified type

        Raises:
            ValueError: If the specified belief class is not found
        """

        # Get all subclasses of Belief recursively
        def get_all_subclasses(c):
            subclasses = c.__subclasses__()
            for subclass in subclasses:
                subclasses.extend(get_all_subclasses(subclass))
            return subclasses

        all_subclasses = get_all_subclasses(cls)
        for subclass in all_subclasses:
            if subclass.__name__ == config.class_name:
                # Skip abstract classes - they cannot be instantiated
                if inspect.isabstract(subclass):
                    raise ValueError(
                        f"Belief class '{config.class_name}' is abstract and cannot be instantiated"
                    )
                return subclass(**config.params)  # pylint: disable=abstract-class-instantiated
        raise ValueError(f"Belief class '{config.class_name}' not found")

    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on belief configuration."""

        def serialize_value(value):
            """Serialize values in a deterministic way."""
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            if isinstance(value, dict):
                return {str(k): serialize_value(v) for k, v in sorted(value.items())}
            if hasattr(value, "__dict__"):
                return serialize_value(value.__dict__)
            return str(value)

        config_dict = {}
        for key, value in self.__dict__.items():
            if key.startswith("_") or callable(value):
                continue
            config_dict[key] = serialize_value(value)
        config_dict = dict(sorted(config_dict.items()))
        return config_to_id(config_dict)

    def __hash__(self) -> int:
        """Make the belief hashable by using its config_id."""
        return hash(self.config_id)

    def __eq__(self, other: object) -> bool:
        """Define equality based on config_id."""
        if not isinstance(other, Belief):
            return NotImplemented
        return self.config_id == other.config_id

    def inplace_update(
        self, action: Any, observation: Any, pomdp: Environment, state: Optional[Any] = None
    ) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        state: Optional[Any] = None,
    ) -> "Belief":
        """Update belief given an action-observation pair.

        Performs Bayesian belief update using the environment's transition
        and observation models.

        Args:
            action: Action that was executed
            observation: Observation that was received
            pomdp: Environment providing transition and observation models

        Returns:
            Updated belief state reflecting the new information

        Note:
            Subclasses must implement this method according to their
            specific belief representation and update strategy.
        """

    @abstractmethod
    def sample(self) -> Any:
        """Sample a state from the current belief distribution.

        Returns:
            A state sampled according to the belief's probability distribution

        Note:
            Subclasses must implement this method to enable state sampling
            for planning and simulation purposes.
        """
        raise NotImplementedError("Subclasses must implement this method")
