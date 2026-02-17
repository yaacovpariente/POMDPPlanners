"""Training utilities for POMDP policies.

This package provides a trainer and callbacks for offline policy-iteration
training of policies that implement the
:class:`~POMDPPlanners.core.policy.TrainablePolicy` mixin.

Classes:
    PolicyTrainer: Concrete collect-then-train loop orchestrator.
    TrainerCallback: Abstract base for training callbacks.
    EarlyStopping: Stop training when a metric stops improving.
    ModelCheckpoint: Save the policy on metric improvement.
    OptunaPruning: Report metrics to Optuna and prune unpromising trials.
    TensorBoardCallback: Log training metrics and weight histograms to TensorBoard.
"""

from POMDPPlanners.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    OptunaPruning,
    TensorBoardCallback,
    TrainerCallback,
)
from POMDPPlanners.training.policy_trainer import PolicyTrainer

__all__ = [
    "PolicyTrainer",
    "TrainerCallback",
    "EarlyStopping",
    "ModelCheckpoint",
    "OptunaPruning",
    "TensorBoardCallback",
]
