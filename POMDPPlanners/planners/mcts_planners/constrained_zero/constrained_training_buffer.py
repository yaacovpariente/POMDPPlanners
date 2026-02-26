"""Iteration-slot replay buffer for ConstrainedZero training examples.

This module extends the BetaZero training buffer with an additional failure
target, used for training the 3-head ConstrainedZero network.

Reference:
    Moss, R. J., Jamgochian, A., Fischer, J., Corso, A., & Kochenderfer, M. J. (2024).
    ConstrainedZero: Chance-Constrained POMDP Planning Using Learned Probabilistic Failure
    Surrogates and Adaptive Safety Constraints. Proceedings of the Thirty-Third International
    Joint Conference on Artificial Intelligence (IJCAI), 6752-6760.
    https://www.ijcai.org/proceedings/2024/746

Classes:
    ConstrainedTrainingExample: Training datum with failure target.
    ConstrainedTrainingBuffer: Buffer returning 4-tuple batches.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from POMDPPlanners.planners.mcts_planners.beta_zero.training_buffer import (
    TrainingBuffer,
)


@dataclass
class ConstrainedTrainingExample:
    """Single training datum for ConstrainedZero network training.

    Attributes:
        belief_features: Belief feature vector phi(b), shape ``(belief_dim,)``.
        policy_target: Q-weighted policy target pi_qw, shape ``(n_actions,)`` (discrete).
        value_target: Discounted return g_t.
        failure_target: Binary episode-level failure indicator (1.0 if failure occurred).
    """

    belief_features: np.ndarray
    policy_target: np.ndarray
    value_target: float
    failure_target: float


class ConstrainedTrainingBuffer(TrainingBuffer):
    """Iteration-slot replay buffer for ConstrainedZero training.

    Extends ``TrainingBuffer`` to store and sample ``ConstrainedTrainingExample``
    instances, returning a 4-tuple from ``sample_batch()`` that includes failure
    targets.

    Args:
        n_buffer: Number of policy-iteration slots to retain.  With the default
            ``n_buffer=1`` only the current iteration's data is used for
            training (on-policy).

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_training_buffer import (
        ...     ConstrainedTrainingBuffer, ConstrainedTrainingExample,
        ... )
        >>> buf = ConstrainedTrainingBuffer(n_buffer=1)
        >>> buf.begin_iteration()
        >>> buf.add(ConstrainedTrainingExample(np.zeros(4), np.array([0.5, 0.5]), 1.0, 0.0))
        >>> len(buf)
        1
        >>> batch = buf.sample_batch(1)
        >>> len(batch)
        4
        >>> batch[0].shape
        (1, 4)
    """

    def add(self, example: ConstrainedTrainingExample) -> None:  # type: ignore[override]
        """Append a constrained example to the current iteration slot."""
        self._current.append(example)  # type: ignore[arg-type]

    def sample_batch(  # type: ignore[override]
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random mini-batch including failure targets.

        Args:
            batch_size: Number of examples to sample (with replacement if
                ``batch_size > len(buffer)``).

        Returns:
            Tuple of (belief_features, policy_targets, value_targets, failure_targets).
            - belief_features: shape ``(batch_size, belief_dim)``
            - policy_targets: shape ``(batch_size, policy_dim)``
            - value_targets: shape ``(batch_size,)``
            - failure_targets: shape ``(batch_size,)``
        """
        all_examples: List[ConstrainedTrainingExample] = [
            ex for slot in self._historical for ex in slot  # type: ignore[misc]
        ] + self._current  # type: ignore[assignment]
        indices = np.random.choice(len(all_examples), size=batch_size, replace=True)
        beliefs = np.array([all_examples[i].belief_features for i in indices])
        policies = np.array([all_examples[i].policy_target for i in indices])
        values = np.array([all_examples[i].value_target for i in indices])
        failures = np.array([all_examples[i].failure_target for i in indices])
        return beliefs, policies, values, failures
