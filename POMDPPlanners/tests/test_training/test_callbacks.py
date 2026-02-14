"""Tests for training callback implementations.

This module tests the TrainerCallback hierarchy: EarlyStopping,
ModelCheckpoint, and OptunaPruning.
"""

# pylint: disable=protected-access

from unittest.mock import MagicMock

import pytest

from POMDPPlanners.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    OptunaPruning,
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

        result = cb.on_iteration_end(trainer, 0, {"total_loss": [0.5]})
        assert result is None
        trial.report.assert_called_once()
