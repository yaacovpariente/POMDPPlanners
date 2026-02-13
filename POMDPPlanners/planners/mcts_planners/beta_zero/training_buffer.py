"""Circular replay buffer for BetaZero training examples.

This module provides a fixed-capacity buffer that stores training tuples
(φ(b), π_qw, g_t) collected during BetaZero policy iteration.

Classes:
    TrainingExample: A single training datum.
    TrainingBuffer: Fixed-capacity circular buffer with batch sampling.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TrainingExample:
    """Single training datum for BetaZero network training.

    Attributes:
        belief_features: Belief feature vector φ(b), shape ``(belief_dim,)``.
        policy_target: Q-weighted policy target π_qw, shape ``(n_actions,)`` (discrete).
        value_target: Discounted return g_t.
    """

    belief_features: np.ndarray
    policy_target: np.ndarray
    value_target: float


class TrainingBuffer:
    """Fixed-capacity circular replay buffer for BetaZero training.

    When the buffer is full, new examples overwrite the oldest ones.

    Args:
        capacity: Maximum number of stored examples.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.planners.mcts_planners.beta_zero.training_buffer import (
        ...     TrainingBuffer, TrainingExample,
        ... )
        >>> buf = TrainingBuffer(capacity=100)
        >>> buf.add(TrainingExample(np.zeros(4), np.array([0.5, 0.5]), 1.0))
        >>> len(buf)
        1
        >>> batch = buf.sample_batch(1)
        >>> batch[0].shape
        (1, 4)
    """

    def __init__(self, capacity: int = 100_000):
        self._capacity = capacity
        self._buffer: list = []
        self._position = 0

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        return self._capacity

    def add(self, example: TrainingExample) -> None:
        """Append an example, overwriting the oldest if at capacity."""
        if len(self._buffer) < self._capacity:
            self._buffer.append(example)
        else:
            self._buffer[self._position] = example
        self._position = (self._position + 1) % self._capacity

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random mini-batch from the buffer.

        Args:
            batch_size: Number of examples to sample (with replacement if
                ``batch_size > len(buffer)``).

        Returns:
            Tuple of (belief_features, policy_targets, value_targets) arrays.
            - belief_features: shape ``(batch_size, belief_dim)``
            - policy_targets: shape ``(batch_size, policy_dim)``
            - value_targets: shape ``(batch_size,)``
        """
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=True)
        beliefs = np.array([self._buffer[i].belief_features for i in indices])
        policies = np.array([self._buffer[i].policy_target for i in indices])
        values = np.array([self._buffer[i].value_target for i in indices])
        return beliefs, policies, values

    def clear(self) -> None:
        """Remove all stored examples."""
        self._buffer.clear()
        self._position = 0
