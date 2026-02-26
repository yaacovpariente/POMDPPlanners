"""Iteration-slot replay buffer for BetaZero training examples.

This module provides a buffer that stores training tuples (φ(b), π_qw, g_t)
collected during BetaZero policy iteration.  The buffer is partitioned into
*iteration slots*: each call to :meth:`TrainingBuffer.begin_iteration` commits
the current slot to history and opens a fresh one.  At most ``n_buffer``
iteration slots are retained; older slots are evicted automatically, which
mirrors the ``CircularBuffer`` design used in the reference Julia implementation
(BetaZero.jl, ``n_buffer`` parameter).

With the default ``n_buffer=1`` only the current iteration's data is ever in
the buffer, keeping training fully on-policy.  Set ``n_buffer > 1`` to retain
a rolling window of recent iterations.

Reference:
    Moss, R. J., Corso, A., Caers, J., & Kochenderfer, M. J. (2024). BetaZero:
    Belief-State Planning for Long-Horizon POMDPs using Learned Approximations.
    Reinforcement Learning Conference (RLC).

Classes:
    TrainingExample: A single training datum.
    TrainingBuffer: Iteration-slot buffer with uniform batch sampling.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

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
    """Iteration-slot replay buffer for BetaZero training.

    Examples are grouped into *iteration slots*.  Calling
    :meth:`begin_iteration` commits the current slot to a fixed-length history
    deque (capacity ``n_buffer - 1`` past slots) and opens a fresh slot for the
    new iteration.  Training samples uniformly from all examples across all
    retained slots plus the current slot.

    With ``n_buffer=1`` (the default) the history deque has capacity 0, so
    only the current iteration's data is ever visible to training — matching
    the on-policy behaviour of the Julia reference implementation.

    Args:
        n_buffer: Number of iteration slots to retain (including the current
            slot).  Must be >= 1.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.planners.mcts_planners.beta_zero.training_buffer import (
        ...     TrainingBuffer, TrainingExample,
        ... )
        >>> buf = TrainingBuffer(n_buffer=1)
        >>> buf.begin_iteration()
        >>> buf.add(TrainingExample(np.zeros(4), np.array([0.5, 0.5]), 1.0))
        >>> len(buf)
        1
        >>> batch = buf.sample_batch(1)
        >>> batch[0].shape
        (1, 4)
    """

    def __init__(self, n_buffer: int = 1):
        if n_buffer < 1:
            raise ValueError(f"n_buffer must be >= 1, got {n_buffer}")
        self._n_buffer = n_buffer
        # History holds at most n_buffer-1 committed past iterations.
        self._historical: Deque[List[TrainingExample]] = deque(maxlen=n_buffer - 1)
        self._current: List[TrainingExample] = []

    def __len__(self) -> int:
        return sum(len(slot) for slot in self._historical) + len(self._current)

    @property
    def n_buffer(self) -> int:
        """Number of iteration slots retained (including the current slot)."""
        return self._n_buffer

    def begin_iteration(self) -> None:
        """Commit the current slot and open a fresh one for the new iteration.

        If the current slot contains examples it is pushed into the history
        deque (oldest slot evicted automatically when the deque is full), then
        the current slot is reset to empty.
        """
        if self._current:
            self._historical.append(self._current)
        self._current = []

    def add(self, example: TrainingExample) -> None:
        """Append an example to the current iteration slot."""
        self._current.append(example)

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random mini-batch uniformly from all retained examples.

        Args:
            batch_size: Number of examples to sample (with replacement if
                ``batch_size > len(buffer)``).

        Returns:
            Tuple of (belief_features, policy_targets, value_targets) arrays.
            - belief_features: shape ``(batch_size, belief_dim)``
            - policy_targets: shape ``(batch_size, policy_dim)``
            - value_targets: shape ``(batch_size,)``
        """
        all_examples: List[TrainingExample] = [
            ex for slot in self._historical for ex in slot
        ] + self._current
        indices = np.random.choice(len(all_examples), size=batch_size, replace=True)
        beliefs = np.array([all_examples[i].belief_features for i in indices])
        policies = np.array([all_examples[i].policy_target for i in indices])
        values = np.array([all_examples[i].value_target for i in indices])
        return beliefs, policies, values

    def get_all_examples(self) -> List[TrainingExample]:
        """Return all examples across all buffer slots (historical + current)."""
        all_examples: List[TrainingExample] = []
        for slot in self._historical:
            all_examples.extend(slot)
        all_examples.extend(self._current)
        return all_examples

    def clear(self) -> None:
        """Remove all stored examples from all slots."""
        self._historical.clear()
        self._current.clear()
