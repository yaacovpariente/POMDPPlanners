"""Trainer callbacks for policy training loops.

This module provides a callback interface and concrete implementations for
monitoring and controlling the :class:`~POMDPPlanners.training.PolicyTrainer`
training loop.

Classes:
    TrainerCallback: Abstract base with default no-op hooks.
    EarlyStopping: Stops training after ``patience`` iterations without improvement.
    ModelCheckpoint: Saves the policy on metric improvement (or every iteration).
    OptunaPruning: Reports metrics to an Optuna trial and prunes when appropriate.
    TensorBoardCallback: Logs training metrics and weight histograms to TensorBoard.
"""

from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from POMDPPlanners.training.policy_trainer import PolicyTrainer


class TrainerCallback(ABC):
    """Abstract base class for training-loop callbacks.

    All hooks have default no-op implementations so that subclasses only
    need to override the methods they care about.

    Note:
        This is an abstract base class.  Instantiate one of the concrete
        subclasses (:class:`EarlyStopping`, :class:`ModelCheckpoint`, or
        :class:`OptunaPruning`) or write your own.
    """

    def on_train_begin(self, trainer: "PolicyTrainer") -> None:
        """Called once before the first training iteration."""

    def on_train_end(self, trainer: "PolicyTrainer", all_metrics: Dict[str, List[float]]) -> None:
        """Called once after the last training iteration."""

    def on_iteration_begin(self, trainer: "PolicyTrainer", iteration: int) -> None:
        """Called at the start of every iteration."""

    def on_iteration_end(  # pylint: disable=unused-argument
        self,
        trainer: "PolicyTrainer",
        iteration: int,
        metrics: Dict[str, List[float]],
    ) -> Optional[bool]:
        """Called at the end of every iteration.

        Args:
            trainer: The running trainer instance.
            iteration: Zero-based iteration index.
            metrics: Loss metrics returned by the current ``train_step``.

        Returns:
            ``True`` to request early stopping; ``None`` or ``False`` to
            continue.
        """
        return None

    def on_collection_begin(self, trainer: "PolicyTrainer", iteration: int) -> None:
        """Called before episode collection starts."""

    def on_collection_end(self, trainer: "PolicyTrainer", iteration: int) -> None:
        """Called after episode collection finishes."""


class EarlyStopping(TrainerCallback):
    """Stop training when a monitored metric stops improving.

    Attributes:
        monitor: Metric key to watch (e.g. ``"total_loss"``).
        patience: Number of iterations with no improvement before stopping.
        mode: ``"min"`` to stop when metric stops decreasing, ``"max"`` for
            increasing.
    """

    def __init__(self, monitor: str, patience: int = 5, mode: str = "min"):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self._best: Optional[float] = None
        self._wait: int = 0

    def _is_improvement(self, current: float) -> bool:
        if self._best is None:
            return True
        if self.mode == "min":
            return current < self._best
        return current > self._best

    def on_iteration_end(
        self,
        trainer: "PolicyTrainer",
        iteration: int,
        metrics: Dict[str, List[float]],
    ) -> Optional[bool]:
        values = metrics.get(self.monitor)
        if not values:
            return None

        current = values[-1]
        if self._is_improvement(current):
            self._best = current
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                return True
        return None


class ModelCheckpoint(TrainerCallback):
    """Save the policy whenever a monitored metric improves.

    Attributes:
        filepath: Directory where checkpoints are saved.
        monitor: Metric key to watch.
        mode: ``"min"`` or ``"max"``.
        save_every: When ``True``, save after every iteration regardless
            of improvement.
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = "total_loss",
        mode: str = "min",
        save_every: bool = False,
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_every = save_every
        self._best: Optional[float] = None

    def _is_improvement(self, current: float) -> bool:
        if self._best is None:
            return True
        if self.mode == "min":
            return current < self._best
        return current > self._best

    def on_iteration_end(
        self,
        trainer: "PolicyTrainer",
        iteration: int,
        metrics: Dict[str, List[float]],
    ) -> Optional[bool]:
        values = metrics.get(self.monitor)
        if not values and not self.save_every:
            return None

        should_save = self.save_every
        if values:
            current = values[-1]
            if self._is_improvement(current):
                self._best = current
                should_save = True

        if should_save:
            trainer.policy.save(filepath=self.filepath)  # type: ignore[attr-defined]

        return None


class OptunaPruning(TrainerCallback):
    """Report metrics to an Optuna trial and prune when appropriate.

    The ``optuna`` package is imported lazily so that it is not a hard
    dependency of the training module.

    Attributes:
        trial: An active Optuna ``Trial`` object.
        monitor: Metric key to report.
    """

    def __init__(self, trial: Any, monitor: str = "total_loss"):
        self.trial = trial
        self.monitor = monitor

    def on_iteration_end(
        self,
        trainer: "PolicyTrainer",
        iteration: int,
        metrics: Dict[str, List[float]],
    ) -> Optional[bool]:
        import optuna  # pylint: disable=import-outside-toplevel

        values = metrics.get(self.monitor)
        if not values:
            return None

        self.trial.report(values[-1], step=iteration)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return None


class TensorBoardCallback(TrainerCallback):
    """Log training metrics to TensorBoard.

    The ``torch.utils.tensorboard`` package is imported lazily so it is not
    a hard startup dependency of the training module.

    Attributes:
        log_dir: Directory for TensorBoard event files.
        comment: Suffix appended to the auto-generated run directory name.
        flush_secs: How often the writer flushes to disk (seconds).
        log_histograms: When ``True`` and the policy exposes
            ``get_network()``, logs per-parameter weight histograms each
            iteration.

    Example:
        >>> from unittest.mock import MagicMock, patch
        >>> with patch("torch.utils.tensorboard.SummaryWriter"):
        ...     cb = TensorBoardCallback(log_dir="/tmp/tb_test")
        ...     trainer = MagicMock()
        ...     cb.on_train_begin(trainer)
        ...     cb.on_train_end(trainer, {})
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        comment: str = "",
        flush_secs: int = 120,
        log_histograms: bool = False,
    ):
        """Initialize TensorBoardCallback.

        Args:
            log_dir: Directory for TensorBoard event files.
                Defaults to a timestamped subdirectory of ``runs/``.
            comment: Suffix appended to the auto-generated run directory.
            flush_secs: How often (in seconds) the writer flushes to disk.
            log_histograms: When ``True`` and the policy exposes
                ``get_network()``, log weight histograms each iteration.
        """
        self.log_dir = log_dir
        self.comment = comment
        self.flush_secs = flush_secs
        self.log_histograms = log_histograms
        self._writer: Optional[Any] = None
        self._global_step: int = 0

    def on_train_begin(self, trainer: "PolicyTrainer") -> None:
        from torch.utils.tensorboard import SummaryWriter  # pylint: disable=import-outside-toplevel

        self._writer = SummaryWriter(
            log_dir=self.log_dir,
            comment=self.comment,
            flush_secs=self.flush_secs,
        )
        self._global_step = 0

    def on_collection_end(  # pylint: disable=unused-argument
        self, trainer: "PolicyTrainer", iteration: int
    ) -> None:
        if self._writer is None:
            return
        self._writer.add_scalar(
            "train/buffer_size",
            trainer.policy.buffer_size(),  # type: ignore[attr-defined]
            self._global_step,
        )

    def on_iteration_end(  # pylint: disable=unused-argument
        self,
        trainer: "PolicyTrainer",
        iteration: int,
        metrics: Dict[str, List[float]],
    ) -> Optional[bool]:
        if self._writer is None:
            return None
        for key, values in metrics.items():
            if values:
                self._writer.add_scalar(f"train/{key}", values[-1], self._global_step)
        if self.log_histograms:
            self._log_weight_histograms(trainer)
        self._global_step += 1
        return None

    def on_train_end(self, trainer: "PolicyTrainer", all_metrics: Dict[str, List[float]]) -> None:
        if self._writer is None:
            return
        self._writer.flush()
        self._writer.close()

    def _log_weight_histograms(self, trainer: "PolicyTrainer") -> None:
        if self._writer is None:
            return
        network = trainer.policy.get_network()  # type: ignore[attr-defined]
        if network is None:
            return
        for name, param in network.named_parameters():
            self._writer.add_histogram(f"weights/{name}", param.detach(), self._global_step)
