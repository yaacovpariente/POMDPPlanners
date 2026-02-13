"""Tests for the ConstrainedTrainingBuffer module.

This module tests the ConstrainedTrainingBuffer and ConstrainedTrainingExample,
which extend the BetaZero training buffer with failure targets.
"""

import random

import numpy as np

from POMDPPlanners.planners.mcts_planners.beta_zero.training_buffer import (
    TrainingBuffer,
)
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_training_buffer import (
    ConstrainedTrainingBuffer,
    ConstrainedTrainingExample,
)

np.random.seed(42)
random.seed(42)


def _make_example(belief_dim=4, policy_dim=3, value=1.0, failure=0.0):
    return ConstrainedTrainingExample(
        belief_features=np.random.rand(belief_dim).astype(np.float32),
        policy_target=np.random.rand(policy_dim).astype(np.float32),
        value_target=value,
        failure_target=failure,
    )


class TestConstrainedTrainingExample:
    """Tests for the ConstrainedTrainingExample dataclass."""

    def test_has_failure_target_field(self):
        """Test ConstrainedTrainingExample has failure_target.

        Purpose: Validates the dataclass includes the failure field.

        Given: A ConstrainedTrainingExample with failure_target=0.5.
        When: Accessing the failure_target attribute.
        Then: The value is 0.5.

        Test type: unit
        """
        ex = ConstrainedTrainingExample(
            belief_features=np.zeros(4),
            policy_target=np.array([0.5, 0.5]),
            value_target=1.0,
            failure_target=0.5,
        )
        assert ex.failure_target == 0.5

    def test_inherits_standard_fields(self):
        """Test ConstrainedTrainingExample has standard BetaZero fields.

        Purpose: Validates belief_features, policy_target, and value_target exist.

        Given: A ConstrainedTrainingExample.
        When: Accessing standard fields.
        Then: All fields are accessible with correct values.

        Test type: unit
        """
        ex = _make_example(value=2.5, failure=1.0)
        assert ex.belief_features.shape == (4,)
        assert ex.policy_target.shape == (3,)
        assert ex.value_target == 2.5
        assert ex.failure_target == 1.0


class TestConstrainedTrainingBuffer:
    """Tests for the ConstrainedTrainingBuffer class."""

    def test_inherits_from_training_buffer(self):
        """Test ConstrainedTrainingBuffer inherits from TrainingBuffer.

        Purpose: Validates the subclass relationship.

        Given: A ConstrainedTrainingBuffer instance.
        When: Checking isinstance.
        Then: It is an instance of TrainingBuffer.

        Test type: unit
        """
        buf = ConstrainedTrainingBuffer(capacity=10)
        assert isinstance(buf, TrainingBuffer)

    def test_add_and_length(self):
        """Test adding examples and checking buffer length.

        Purpose: Validates basic add/length functionality.

        Given: An empty ConstrainedTrainingBuffer.
        When: Adding 3 examples.
        Then: Length is 3.

        Test type: unit
        """
        buf = ConstrainedTrainingBuffer(capacity=100)
        for _ in range(3):
            buf.add(_make_example())
        assert len(buf) == 3

    def test_sample_batch_returns_four_arrays(self):
        """Test sample_batch returns 4-tuple with failure targets.

        Purpose: Validates the 4-tuple output from sample_batch.

        Given: A buffer with 10 examples (belief_dim=4, policy_dim=3).
        When: sample_batch(5) is called.
        Then: Returns (beliefs, policies, values, failures) with correct shapes.

        Test type: unit
        """
        buf = ConstrainedTrainingBuffer(capacity=100)
        for _ in range(10):
            buf.add(_make_example())

        batch = buf.sample_batch(5)
        assert len(batch) == 4
        beliefs, policies, values, failures = batch
        assert beliefs.shape == (5, 4)
        assert policies.shape == (5, 3)
        assert values.shape == (5,)
        assert failures.shape == (5,)

    def test_failure_targets_propagate(self):
        """Test failure targets are correctly stored and retrieved.

        Purpose: Validates failure targets survive the buffer roundtrip.

        Given: A buffer with examples having known failure targets (all 1.0).
        When: sample_batch is called.
        Then: All failure targets in the batch are 1.0.

        Test type: unit
        """
        buf = ConstrainedTrainingBuffer(capacity=100)
        for _ in range(5):
            buf.add(_make_example(failure=1.0))

        _, _, _, failures = buf.sample_batch(5)
        np.testing.assert_array_equal(failures, np.ones(5))

    def test_mixed_failure_targets(self):
        """Test buffer handles mixed failure targets correctly.

        Purpose: Validates buffer stores both 0.0 and 1.0 failure targets.

        Given: A buffer with alternating failure targets.
        When: All examples are sampled.
        Then: The failure targets contain both 0.0 and 1.0.

        Test type: unit
        """
        buf = ConstrainedTrainingBuffer(capacity=100)
        for i in range(10):
            buf.add(_make_example(failure=float(i % 2)))

        # Sample many times to ensure we see both values
        all_failures = set()
        for _ in range(50):
            _, _, _, failures = buf.sample_batch(10)
            all_failures.update(failures.tolist())
        assert 0.0 in all_failures
        assert 1.0 in all_failures

    def test_capacity_circular_overflow(self):
        """Test circular buffer overwrites oldest entries.

        Purpose: Validates capacity enforcement.

        Given: A buffer with capacity=5.
        When: 8 examples are added.
        Then: Buffer length remains 5.

        Test type: unit
        """
        buf = ConstrainedTrainingBuffer(capacity=5)
        for i in range(8):
            buf.add(_make_example(value=float(i)))
        assert len(buf) == 5

    def test_clear(self):
        """Test clear empties the buffer.

        Purpose: Validates clear() works for the constrained buffer.

        Given: A buffer with 5 examples.
        When: clear() is called.
        Then: Buffer length is 0.

        Test type: unit
        """
        buf = ConstrainedTrainingBuffer(capacity=100)
        for _ in range(5):
            buf.add(_make_example())
        buf.clear()
        assert len(buf) == 0
