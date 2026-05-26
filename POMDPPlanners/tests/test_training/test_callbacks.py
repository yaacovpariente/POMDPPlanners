# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for training callback implementations.

This module tests the TrainerCallback hierarchy: EarlyStopping,
ModelCheckpoint, OptunaPruning, and TensorBoardCallback.
"""

# pylint: disable=protected-access

from unittest.mock import MagicMock, patch

import pytest

from POMDPPlanners.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    OptunaPruning,
    TensorBoardCallback,
)


class TestEarlyStopping:
    """Tests for the EarlyStopping callback."""

    def test_triggers_after_patience(self):
        """Test that EarlyStopping returns True after patience is exhausted.

        Purpose: Validates that EarlyStopping fires after ``patience``
        consecutive iterations without improvement.

        Given: EarlyStopping with patience=2 monitoring "total_loss" (min).
        When: Three iterations report the same (non-improving) metric value.
        Then: The first two return None, the third returns True.

        Test type: unit
        """
        cb = EarlyStopping(monitor="total_loss", patience=2, mode="min")
        trainer = MagicMock()

        # First call sets best
        result = cb.on_iteration_end(trainer, 0, {"total_loss": [1.0]})
        assert result is None

        # No improvement
        result = cb.on_iteration_end(trainer, 1, {"total_loss": [1.0]})
        assert result is None

        # Still no improvement — patience=2 exhausted
        result = cb.on_iteration_end(trainer, 2, {"total_loss": [1.0]})
        assert result is True

    def test_resets_on_improvement(self):
        """Test that counter resets when the monitored metric improves.

        Purpose: Validates that improvement resets the wait counter.

        Given: EarlyStopping with patience=2 monitoring "total_loss" (min).
        When: Metrics improve after one non-improving iteration.
        Then: The wait counter resets and no early stop is triggered.

        Test type: unit
        """
        cb = EarlyStopping(monitor="total_loss", patience=2, mode="min")
        trainer = MagicMock()

        cb.on_iteration_end(trainer, 0, {"total_loss": [1.0]})
        cb.on_iteration_end(trainer, 1, {"total_loss": [1.5]})  # worse
        assert cb._wait == 1

        cb.on_iteration_end(trainer, 2, {"total_loss": [0.8]})  # better
        assert cb._wait == 0
        assert cb._best == 0.8

    def test_max_mode(self):
        """Test EarlyStopping in maximize mode.

        Purpose: Validates that mode='max' stops when metric stops increasing.

        Given: EarlyStopping with patience=1 monitoring "accuracy" (max).
        When: Accuracy decreases after one iteration.
        Then: Returns True on the next non-improving iteration.

        Test type: unit
        """
        cb = EarlyStopping(monitor="accuracy", patience=1, mode="max")
        trainer = MagicMock()

        cb.on_iteration_end(trainer, 0, {"accuracy": [0.9]})
        result = cb.on_iteration_end(trainer, 1, {"accuracy": [0.8]})
        assert result is True

    def test_missing_metric_is_noop(self):
        """Test that missing metric key is silently ignored.

        Purpose: Validates graceful handling of missing metrics.

        Given: EarlyStopping monitoring "total_loss".
        When: Metrics dict does not contain "total_loss".
        Then: Returns None (no stop).

        Test type: unit
        """
        cb = EarlyStopping(monitor="total_loss", patience=1)
        trainer = MagicMock()
        result = cb.on_iteration_end(trainer, 0, {"other_loss": [1.0]})
        assert result is None


class TestModelCheckpoint:
    """Tests for the ModelCheckpoint callback."""

    def test_saves_on_improvement(self, tmp_path):
        """Test checkpoint saves when monitored metric improves.

        Purpose: Validates that ModelCheckpoint calls policy.save() on
        improvement.

        Given: ModelCheckpoint monitoring "total_loss" (min) with a mock
               trainer whose policy has a save() method.
        When: Two iterations — first improves (sets baseline), second improves
              again (lower loss).
        Then: policy.save() is called twice.

        Test type: unit
        """
        cb = ModelCheckpoint(filepath=str(tmp_path / "ckpt"), monitor="total_loss")
        trainer = MagicMock()

        cb.on_iteration_end(trainer, 0, {"total_loss": [1.0]})
        trainer.policy.save.assert_called_once()

        cb.on_iteration_end(trainer, 1, {"total_loss": [0.5]})
        assert trainer.policy.save.call_count == 2

    def test_does_not_save_without_improvement(self, tmp_path):
        """Test checkpoint does not save when metric does not improve.

        Purpose: Validates that save is skipped when metric worsens.

        Given: ModelCheckpoint monitoring "total_loss" (min).
        When: Second iteration has higher loss.
        Then: policy.save() is called only once (for the first iteration).

        Test type: unit
        """
        cb = ModelCheckpoint(filepath=str(tmp_path / "ckpt"), monitor="total_loss")
        trainer = MagicMock()

        cb.on_iteration_end(trainer, 0, {"total_loss": [1.0]})
        cb.on_iteration_end(trainer, 1, {"total_loss": [2.0]})
        assert trainer.policy.save.call_count == 1

    def test_save_every_flag(self, tmp_path):
        """Test save_every=True saves regardless of improvement.

        Purpose: Validates unconditional saving behaviour.

        Given: ModelCheckpoint with save_every=True.
        When: Two iterations regardless of metric direction.
        Then: policy.save() is called twice.

        Test type: unit
        """
        cb = ModelCheckpoint(filepath=str(tmp_path / "ckpt"), monitor="total_loss", save_every=True)
        trainer = MagicMock()

        cb.on_iteration_end(trainer, 0, {"total_loss": [1.0]})
        cb.on_iteration_end(trainer, 1, {"total_loss": [2.0]})
        assert trainer.policy.save.call_count == 2


class TestOptunaPruning:
    """Tests for the OptunaPruning callback."""

    def test_reports_and_prunes(self):
        """Test OptunaPruning reports metric and raises TrialPruned.

        Purpose: Validates Optuna integration — metric reporting and pruning.

        Given: A mock Optuna trial that requests pruning.
        When: on_iteration_end is called with a metric value.
        Then: trial.report() is called with the value, and TrialPruned is raised.

        Test type: unit
        """
        import optuna  # pylint: disable=import-outside-toplevel

        trial = MagicMock()
        trial.should_prune.return_value = True

        cb = OptunaPruning(trial=trial, monitor="total_loss")
        trainer = MagicMock()

        with pytest.raises(optuna.TrialPruned):
            cb.on_iteration_end(trainer, 0, {"total_loss": [0.5]})

        trial.report.assert_called_once_with(0.5, step=0)

    def test_no_prune_when_trial_says_no(self):
        """Test OptunaPruning does not prune when trial.should_prune() is False.

        Purpose: Validates normal (non-pruning) operation.

        Given: A mock Optuna trial that does not request pruning.
        When: on_iteration_end is called.
        Then: trial.report() is called but no exception is raised.

        Test type: unit
        """
        trial = MagicMock()
        trial.should_prune.return_value = False

        cb = OptunaPruning(trial=trial, monitor="total_loss")
        trainer = MagicMock()

        assert cb.on_iteration_end(trainer, 0, {"total_loss": [0.5]}) is None
        trial.report.assert_called_once()


class TestTensorBoardCallback:
    """Tests for the TensorBoardCallback class."""

    def test_writer_created_on_train_begin(self):
        """Test that SummaryWriter is created when training begins.

        Purpose: Validates that on_train_begin creates a SummaryWriter with
        the configured log_dir, comment, and flush_secs parameters.

        Given: A TensorBoardCallback with explicit log_dir and comment.
        When: on_train_begin is called.
        Then: SummaryWriter is instantiated with the correct arguments and
              _global_step is reset to 0.

        Test type: unit
        """
        with patch("torch.utils.tensorboard.SummaryWriter") as mock_sw_class:
            mock_sw_class.return_value = MagicMock()
            cb = TensorBoardCallback(log_dir="/tmp/runs", comment="test", flush_secs=30)
            trainer = MagicMock()

            cb._global_step = 99  # set to non-zero to verify reset
            cb.on_train_begin(trainer)

            mock_sw_class.assert_called_once_with(
                log_dir="/tmp/runs", comment="test", flush_secs=30
            )
            assert cb._global_step == 0

    def test_buffer_size_logged_on_collection_end(self):
        """Test that buffer size is logged when episode collection finishes.

        Purpose: Validates that on_collection_end calls add_scalar with
        'train/buffer_size' and the current buffer size.

        Given: A TensorBoardCallback after on_train_begin; a trainer whose
               policy.buffer_size() returns 500.
        When: on_collection_end is called.
        Then: writer.add_scalar is called with ('train/buffer_size', 500, 0).

        Test type: unit
        """
        with patch("torch.utils.tensorboard.SummaryWriter") as mock_sw_class:
            mock_writer = MagicMock()
            mock_sw_class.return_value = mock_writer

            cb = TensorBoardCallback()
            trainer = MagicMock()
            trainer.policy.buffer_size.return_value = 500

            cb.on_train_begin(trainer)
            cb.on_collection_end(trainer, 0)

            mock_writer.add_scalar.assert_called_once_with("train/buffer_size", 500, 0)

    def test_loss_metrics_logged_on_iteration_end(self):
        """Test that all metric keys are logged at the end of each iteration.

        Purpose: Validates that on_iteration_end calls add_scalar for each key
        in the metrics dict, using the last value in each list.

        Given: A TensorBoardCallback after on_train_begin; metrics dict with
               'total_loss' and 'value_loss' containing multiple values.
        When: on_iteration_end is called.
        Then: add_scalar is called with 'train/total_loss' and the last value,
              and 'train/value_loss' and its last value.

        Test type: unit
        """
        with patch("torch.utils.tensorboard.SummaryWriter") as mock_sw_class:
            mock_writer = MagicMock()
            mock_sw_class.return_value = mock_writer

            cb = TensorBoardCallback()
            trainer = MagicMock()
            metrics = {"total_loss": [1.0, 0.9], "value_loss": [0.5, 0.4]}

            cb.on_train_begin(trainer)
            cb.on_iteration_end(trainer, 0, metrics)

            logged = {call.args[0]: call.args[1] for call in mock_writer.add_scalar.call_args_list}
            assert "train/total_loss" in logged
            assert "train/value_loss" in logged
            assert logged["train/total_loss"] == pytest.approx(0.9)
            assert logged["train/value_loss"] == pytest.approx(0.4)

    def test_global_step_increments_each_iteration(self):
        """Test that the global step counter advances after each iteration.

        Purpose: Validates that _global_step increments by 1 on each call to
        on_iteration_end, providing a monotonically increasing x-axis for logs.

        Given: A TensorBoardCallback after on_train_begin.
        When: on_iteration_end is called twice.
        Then: _global_step is 1 after the first call, 2 after the second.

        Test type: unit
        """
        with patch("torch.utils.tensorboard.SummaryWriter") as mock_sw_class:
            mock_sw_class.return_value = MagicMock()
            cb = TensorBoardCallback()
            trainer = MagicMock()

            cb.on_train_begin(trainer)
            assert cb._global_step == 0

            cb.on_iteration_end(trainer, 0, {"total_loss": [1.0]})
            assert cb._global_step == 1

            cb.on_iteration_end(trainer, 1, {"total_loss": [0.9]})
            assert cb._global_step == 2

    def test_writer_closed_on_train_end(self):
        """Test that the writer is flushed and closed when training ends.

        Purpose: Validates that on_train_end calls flush() then close() on the
        SummaryWriter to ensure all events are persisted.

        Given: A TensorBoardCallback after on_train_begin.
        When: on_train_end is called.
        Then: writer.flush() and writer.close() are each called once.

        Test type: unit
        """
        with patch("torch.utils.tensorboard.SummaryWriter") as mock_sw_class:
            mock_writer = MagicMock()
            mock_sw_class.return_value = mock_writer

            cb = TensorBoardCallback()
            trainer = MagicMock()

            cb.on_train_begin(trainer)
            cb.on_train_end(trainer, {})

            mock_writer.flush.assert_called_once()
            mock_writer.close.assert_called_once()

    def test_weight_histograms_logged_when_enabled(self):
        """Test that weight histograms are logged when log_histograms=True.

        Purpose: Validates that when log_histograms is True and the policy
        exposes a non-None network via get_network(), add_histogram is called
        for each named parameter.

        Given: A TensorBoardCallback with log_histograms=True; a mock network
               with one named parameter 'layer.weight'.
        When: on_iteration_end is called.
        Then: writer.add_histogram is called with 'weights/layer.weight'.

        Test type: unit
        """
        with patch("torch.utils.tensorboard.SummaryWriter") as mock_sw_class:
            mock_writer = MagicMock()
            mock_sw_class.return_value = mock_writer

            mock_param = MagicMock()
            mock_param.detach.return_value = mock_param
            mock_network = MagicMock()
            mock_network.named_parameters.return_value = [("layer.weight", mock_param)]

            mock_policy = MagicMock()
            mock_policy.get_network.return_value = mock_network
            trainer = MagicMock()
            trainer.policy = mock_policy

            cb = TensorBoardCallback(log_histograms=True)
            cb.on_train_begin(trainer)
            cb.on_iteration_end(trainer, 0, {"total_loss": [1.0]})

            mock_writer.add_histogram.assert_called_once_with("weights/layer.weight", mock_param, 0)

    def test_histograms_skipped_when_get_network_returns_none(self):
        """Test that no histograms are logged when get_network() returns None.

        Purpose: Validates that the callback gracefully handles policies that
        do not expose a network (get_network() returns None).

        Given: A TensorBoardCallback with log_histograms=True; a mock policy
               whose get_network() returns None.
        When: on_iteration_end is called.
        Then: writer.add_histogram is never called.

        Test type: unit
        """
        with patch("torch.utils.tensorboard.SummaryWriter") as mock_sw_class:
            mock_writer = MagicMock()
            mock_sw_class.return_value = mock_writer

            mock_policy = MagicMock()
            mock_policy.get_network.return_value = None
            trainer = MagicMock()
            trainer.policy = mock_policy

            cb = TensorBoardCallback(log_histograms=True)
            cb.on_train_begin(trainer)
            cb.on_iteration_end(trainer, 0, {"total_loss": [1.0]})

            mock_writer.add_histogram.assert_not_called()
