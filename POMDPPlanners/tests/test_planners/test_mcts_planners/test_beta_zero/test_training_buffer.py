"""Tests for the BetaZero training buffer module.

This module tests the TrainingBuffer iteration-slot replay buffer and the
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
    """Tests for TrainingBuffer iteration-slot replay buffer."""

    def test_add_and_length(self):
        """Test that adding examples within an iteration slot increments length.

        Purpose: Validates that the buffer length reflects examples added to
        the current iteration slot.

        Given: A TrainingBuffer with n_buffer=1 after begin_iteration().
        When: Three TrainingExample instances are added to the buffer.
        Then: The length of the buffer equals 3.

        Test type: unit
        """
        buf = TrainingBuffer(n_buffer=1)
        buf.begin_iteration()
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
        buf = TrainingBuffer(n_buffer=1)
        buf.begin_iteration()
        for _ in range(5):
            buf.add(_make_example(belief_dim=4, policy_dim=3))

        beliefs, policies, values = buf.sample_batch(batch_size=2)

        assert beliefs.shape == (2, 4)
        assert policies.shape == (2, 3)
        assert values.shape == (2,)

    def test_begin_iteration_discards_old_data_with_n_buffer_1(self):
        """Test that begin_iteration evicts previous data when n_buffer=1.

        Purpose: Validates the on-policy behaviour: with n_buffer=1, calling
        begin_iteration() discards all data from the previous iteration so
        that training only sees the current iteration's examples.

        Given: A TrainingBuffer(n_buffer=1) with 5 examples from iteration 0.
        When: begin_iteration() is called and 2 new examples are added.
        Then: The buffer length equals 2 (only the new iteration's examples).

        Test type: unit
        """
        buf = TrainingBuffer(n_buffer=1)
        buf.begin_iteration()
        for _ in range(5):
            buf.add(_make_example())

        buf.begin_iteration()
        for _ in range(2):
            buf.add(_make_example())

        assert len(buf) == 2

    def test_n_buffer_2_retains_previous_iteration(self):
        """Test that n_buffer=2 retains one past iteration alongside the current one.

        Purpose: Validates the rolling-window behaviour: with n_buffer=2, data
        from the previous iteration is retained and included in sampling.

        Given: A TrainingBuffer(n_buffer=2) with 5 examples from iteration 0.
        When: begin_iteration() is called and 3 examples are added for iteration 1.
        Then: The buffer length equals 8 (5 from iter 0 + 3 from iter 1).

        Test type: unit
        """
        buf = TrainingBuffer(n_buffer=2)
        buf.begin_iteration()
        for _ in range(5):
            buf.add(_make_example())

        buf.begin_iteration()
        for _ in range(3):
            buf.add(_make_example())

        assert len(buf) == 8

    def test_n_buffer_2_evicts_oldest_after_third_iteration(self):
        """Test that n_buffer=2 evicts the oldest slot when a third iteration starts.

        Purpose: Validates that the rolling window is bounded: with n_buffer=2,
        only the 2 most recent iterations are ever retained.

        Given: A TrainingBuffer(n_buffer=2) with examples across 3 iterations
        (sizes 5, 3, 4 respectively).
        When: begin_iteration() is called before each iteration.
        Then: After the third iteration's data is added, the buffer contains
        only 7 examples (3 from iter 1 + 4 from iter 2), not 12.

        Test type: unit
        """
        buf = TrainingBuffer(n_buffer=2)
        buf.begin_iteration()
        for _ in range(5):
            buf.add(_make_example())

        buf.begin_iteration()
        for _ in range(3):
            buf.add(_make_example())

        buf.begin_iteration()
        for _ in range(4):
            buf.add(_make_example())

        assert len(buf) == 7

    def test_clear_resets_buffer(self):
        """Test that clear empties the buffer completely.

        Purpose: Validates that calling clear removes all stored examples and
        resets the buffer length to zero.

        Given: A TrainingBuffer containing 4 examples across two iterations.
        When: The clear method is called.
        Then: The buffer length equals 0.

        Test type: unit
        """
        buf = TrainingBuffer(n_buffer=2)
        buf.begin_iteration()
        for _ in range(2):
            buf.add(_make_example())
        buf.begin_iteration()
        for _ in range(2):
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
        buf = TrainingBuffer(n_buffer=1)
        buf.begin_iteration()
        for _ in range(2):
            buf.add(_make_example(belief_dim=4, policy_dim=3))

        beliefs, policies, values = buf.sample_batch(batch_size=10)

        assert beliefs.shape == (10, 4)
        assert policies.shape == (10, 3)
        assert values.shape == (10,)

    def test_n_buffer_property(self):
        """Test that the n_buffer property returns the configured value.

        Purpose: Validates that the n_buffer property correctly exposes the
        number of iteration slots set during initialization.

        Given: A TrainingBuffer initialized with n_buffer=3.
        When: The n_buffer property is accessed.
        Then: The returned value equals 3.

        Test type: unit
        """
        buf = TrainingBuffer(n_buffer=3)

        assert buf.n_buffer == 3

    def test_invalid_n_buffer_raises(self):
        """Test that n_buffer < 1 raises ValueError.

        Purpose: Validates that the buffer rejects invalid configuration.

        Given: An attempt to create a TrainingBuffer with n_buffer=0.
        When: The constructor is called.
        Then: A ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError):
            TrainingBuffer(n_buffer=0)

    def test_empty_iteration_not_committed_to_history(self):
        """Test that calling begin_iteration on an empty slot does not pollute history.

        Purpose: Validates that an iteration with no data added does not push
        an empty slot into the history deque, keeping n_buffer slots clean.

        Given: A TrainingBuffer(n_buffer=2) where begin_iteration is called
        twice before any data is added, then 3 examples are added.
        When: The buffer length is checked.
        Then: The buffer length equals 3 (empty slots are not retained).

        Test type: unit
        """
        buf = TrainingBuffer(n_buffer=2)
        buf.begin_iteration()
        buf.begin_iteration()  # empty iteration — should not be pushed to history
        for _ in range(3):
            buf.add(_make_example())

        assert len(buf) == 3
