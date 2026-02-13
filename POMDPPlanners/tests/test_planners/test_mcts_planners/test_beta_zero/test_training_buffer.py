"""Tests for the BetaZero training buffer module.

This module tests the TrainingBuffer circular replay buffer and the
TrainingExample dataclass used for BetaZero network training.
"""

import numpy as np
import pytest

from POMDPPlanners.planners.mcts_planners.beta_zero.training_buffer import (
    TrainingBuffer,
    TrainingExample,
)


def _make_example(belief_dim=4, policy_dim=3, value=1.0):
    return TrainingExample(
        belief_features=np.random.rand(belief_dim),
        policy_target=np.random.rand(policy_dim),
        value_target=value,
    )


class TestTrainingBuffer:
    """Tests for TrainingBuffer circular replay buffer."""

    def test_add_and_length(self):
        """Test that adding examples increments the buffer length correctly.

        Purpose: Validates that the buffer length reflects the number of added examples.

        Given: An empty TrainingBuffer with default capacity.
        When: Three TrainingExample instances are added to the buffer.
        Then: The length of the buffer equals 3.

        Test type: unit
        """
        buf = TrainingBuffer()
        for _ in range(3):
            buf.add(_make_example())

        assert len(buf) == 3

    def test_sample_batch_shapes(self):
        """Test that sampled batch arrays have the correct shapes.

        Purpose: Validates that sample_batch returns arrays with dimensions
        matching the belief_dim, policy_dim, and batch_size.

        Given: A TrainingBuffer containing 5 examples with belief_dim=4 and policy_dim=3.
        When: A batch of size 2 is sampled from the buffer.
        Then: The returned belief_features has shape (2, 4), policy_targets has
              shape (2, 3), and value_targets has shape (2,).

        Test type: unit
        """
        buf = TrainingBuffer()
        for _ in range(5):
            buf.add(_make_example(belief_dim=4, policy_dim=3))

        beliefs, policies, values = buf.sample_batch(batch_size=2)

        assert beliefs.shape == (2, 4)
        assert policies.shape == (2, 3)
        assert values.shape == (2,)

    def test_circular_overflow_maintains_capacity(self):
        """Test that the buffer does not exceed its capacity when overfilled.

        Purpose: Validates the circular overwrite behavior when more examples
        are added than the buffer capacity allows.

        Given: A TrainingBuffer with capacity=5.
        When: 7 examples are added, exceeding the capacity by 2.
        Then: The buffer length remains 5, equal to the capacity.

        Test type: unit
        """
        buf = TrainingBuffer(capacity=5)
        for _ in range(7):
            buf.add(_make_example())

        assert len(buf) == 5

    def test_clear_resets_buffer(self):
        """Test that clear empties the buffer completely.

        Purpose: Validates that calling clear removes all stored examples and
        resets the buffer length to zero.

        Given: A TrainingBuffer containing 4 examples.
        When: The clear method is called.
        Then: The buffer length equals 0.

        Test type: unit
        """
        buf = TrainingBuffer()
        for _ in range(4):
            buf.add(_make_example())
        assert len(buf) == 4

        buf.clear()

        assert len(buf) == 0

    def test_sample_with_replacement(self):
        """Test that sampling with replacement allows batch_size > buffer length.

        Purpose: Validates that sample_batch works when the requested batch size
        exceeds the number of stored examples by sampling with replacement.

        Given: A TrainingBuffer containing only 2 examples with belief_dim=4
               and policy_dim=3.
        When: A batch of size 10 is sampled from the buffer.
        Then: The returned arrays have shapes (10, 4), (10, 3), and (10,)
              respectively, confirming replacement-based sampling succeeded.

        Test type: unit
        """
        buf = TrainingBuffer()
        for _ in range(2):
            buf.add(_make_example(belief_dim=4, policy_dim=3))

        beliefs, policies, values = buf.sample_batch(batch_size=10)

        assert beliefs.shape == (10, 4)
        assert policies.shape == (10, 3)
        assert values.shape == (10,)

    def test_capacity_property(self):
        """Test that the capacity property returns the configured capacity.

        Purpose: Validates that the capacity property correctly exposes the
        maximum buffer size set during initialization.

        Given: A TrainingBuffer initialized with capacity=50.
        When: The capacity property is accessed.
        Then: The returned value equals 50.

        Test type: unit
        """
        buf = TrainingBuffer(capacity=50)

        assert buf.capacity == 50
