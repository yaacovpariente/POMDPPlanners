"""Trainer callbacks for policy training loops.

This module provides a callback interface and concrete implementations for
monitoring and controlling the :class:`~POMDPPlanners.training.PolicyTrainer`
training loop.

Classes:
    TrainerCallback: Abstract base with default no-op hooks.
    EarlyStopping: Stops training after ``patience`` iterations without improvement.
    ModelCheckpoint: Saves the policy on metric improvement (or every iteration).
    OptunaPruning: Reports metrics to an Optuna trial and prunes when appropriate.
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
