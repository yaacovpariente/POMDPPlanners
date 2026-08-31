# SPDX-License-Identifier: MIT

"""Fitting procedures that turn a dataset of transitions into a transition model.

Fitting is the only thing that differs between model classes, so it is the only
thing this package asks a caller to swap. The loop, the data, the diagnostics
and the plumbing back into the planner are identical whichever learner is used,
which is what makes a run comparing two model classes a comparison of the models
and not of everything around them.

Two ship here. The linear-Gaussian learner wraps the ridge fit the package
already had; it is the floor, and its job is to make "the ensemble helped" a
statement with a baseline behind it. The ensemble learner is the candidate.

Both are trained by likelihood, not squared error. That is not a preference: the
bound relating model quality to control performance is stated in the distance
between predicted and true next-state distributions, and likelihood is the part
of it that can be computed from samples.

Classes:
    TransitionModelLearner: Interface for a fitting procedure.
    LinearGaussianLearner: Ridge fit of a linear-Gaussian transition.
    ProbabilisticEnsembleLearner: Gaussian-likelihood fit of an ensemble.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from POMDPPlanners.core.environment import TransitionModel
from POMDPPlanners.training.model_learning.ensemble_transition import (
    ProbabilisticEnsembleTransition,
    build_members,
    normalization_stats,
)
from POMDPPlanners.training.model_learning.transition_dataset import (
    TransitionBatch,
    TransitionDataset,
)


class TransitionModelLearner(ABC):
    """Interface for a procedure that fits a transition model from transitions.

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, used to label a round's checkpoint and its metrics."""

    @abstractmethod
    def fit(self, dataset: TransitionDataset) -> TransitionModel:
        """Fit a transition model on the dataset's training split.

        Args:
            dataset: The transitions collected so far.

        Returns:
            The fitted model.

        Raises:
            ValueError: If the training split holds too few transitions to fit.
        """

    @abstractmethod
    def training_metrics(self) -> Dict[str, List[float]]:
        """Per-epoch metrics from the most recent fit, keyed by metric name.

        The shape matches what the package's callbacks already consume, so early
        stopping and checkpointing work against a model fit unchanged.
        """


class LinearGaussianLearner(TransitionModelLearner):
    """Ridge fit of ``next = A @ state + B @ action + b + N(0, Sigma)``.

    The floor baseline. It has a closed-form solution, so it cannot fail to
    converge and cannot be blamed on a learning rate -- which is what makes it
    useful as the thing an ensemble has to beat.

    Args:
        regularization: Ridge penalty on the normal-equations diagonal.
        min_variance: Floor on each residual variance, keeping the covariance
            positive definite.
    """

    def __init__(self, regularization: float = 1e-4, min_variance: float = 1e-6) -> None:
        self._regularization = regularization
        self._min_variance = min_variance
        self._metrics: Dict[str, List[float]] = {}

    @property
    def name(self) -> str:
        """Short identifier for this learner."""
        return "linear"

    def fit(self, dataset: TransitionDataset) -> TransitionModel:
        """Fit the linear-Gaussian transition on the training split.

        Args:
            dataset: The transitions collected so far.

        Returns:
            A fitted
            :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp.LinearGaussianTransition`.

        Raises:
            ValueError: If fewer than two training transitions exist.
        """
        # Deferred: the concrete linear model lives in the environments layer,
        # and importing it at module scope would make every use of this package
        # pay for the Isaac import chain.
        # pylint: disable-next=import-outside-toplevel
        from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp import (
            LinearGaussianTransition,
        )

        batch = dataset.training_batch()
        if len(batch) < 2:
            raise ValueError(f"fitting a linear transition needs at least 2 rows, got {len(batch)}")
        model = LinearGaussianTransition.fit(
            batch.states,
            batch.actions,
            batch.next_states,
            regularization=self._regularization,
            min_variance=self._min_variance,
        )
        self._metrics = {"train_nll": [_mean_negative_log_likelihood(model, batch)]}
        return model

    def training_metrics(self) -> Dict[str, List[float]]:
        """Mean training negative log-likelihood of the most recent fit."""
        return dict(self._metrics)


class ProbabilisticEnsembleLearner(TransitionModelLearner):
    """Fit an ensemble of Gaussian networks on the state change, by likelihood.

    Args:
        num_members: Number of independently initialised members.
        hidden_sizes: Hidden layer widths, shared by every member.
        epochs: Passes over the training split.
        batch_size: Rows per gradient step.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        seed: Seed for the member initialisations, the batch order and sampling.
        device: Device to train on. ``None`` trains on CPU, which is the right
            default: these are small networks and the batches are small, so the
            transfer cost outweighs the kernel.

    Raises:
        ValueError: If ``num_members`` is not positive.
    """

    def __init__(
        self,
        num_members: int = 5,
        hidden_sizes: Sequence[int] = (200, 200),
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        if num_members <= 0:
            raise ValueError(f"num_members must be positive, got {num_members}")
        self._num_members = num_members
        self._hidden_sizes = tuple(hidden_sizes)
        self._epochs = epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._seed = seed
        self._device = device
        self._metrics: Dict[str, List[float]] = {}

    @property
    def name(self) -> str:
        """Short identifier for this learner."""
        return "ensemble"

    def fit(self, dataset: TransitionDataset) -> TransitionModel:
        """Fit every member on the training split and return the ensemble.

        Each member sees a bootstrap resample of the training rows, so members
        differ in data as well as in initialisation -- two independent sources of
        the disagreement the ensemble reads as uncertainty.

        Args:
            dataset: The transitions collected so far.

        Returns:
            A fitted :class:`ProbabilisticEnsembleTransition`.

        Raises:
            ValueError: If fewer than two training transitions exist.
        """
        batch = dataset.training_batch()
        if len(batch) < 2:
            raise ValueError(f"fitting an ensemble needs at least 2 rows, got {len(batch)}")

        state_mean, state_scale = normalization_stats(batch.states)
        action_mean, action_scale = normalization_stats(batch.actions)
        delta_mean, delta_scale = normalization_stats(batch.deltas)

        inputs = np.concatenate(
            [
                (batch.states - state_mean) / state_scale,
                (batch.actions - action_mean) / action_scale,
            ],
            axis=1,
        )
        targets = (batch.deltas - delta_mean) / delta_scale

        members = build_members(
            input_dim=inputs.shape[1],
            output_dim=targets.shape[1],
            num_members=self._num_members,
            hidden_sizes=self._hidden_sizes,
            seed=self._seed,
            device=self._device,
        )
        holdout = dataset.holdout_batch()
        holdout_inputs, holdout_targets = _normalized(
            holdout, state_mean, state_scale, action_mean, action_scale, delta_mean, delta_scale
        )

        rng = np.random.default_rng(self._seed)
        train_curves: List[List[float]] = []
        holdout_curves: List[List[float]] = []
        for index, member in enumerate(members):
            resample = rng.integers(len(batch), size=len(batch))
            train_curve, holdout_curve = _train_member(
                member,
                inputs[resample],
                targets[resample],
                epochs=self._epochs,
                batch_size=self._batch_size,
                learning_rate=self._learning_rate,
                weight_decay=self._weight_decay,
                seed=self._seed + index,
                device=self._device,
                holdout_inputs=holdout_inputs,
                holdout_targets=holdout_targets,
            )
            train_curves.append(train_curve)
            holdout_curves.append(holdout_curve)
        # The mean over members describes the ensemble; the per-member curves are
        # kept beside it because a single member that diverged is invisible in the
        # mean and is exactly what makes an ensemble's spread meaningless. The
        # holdout curve is what says whether more epochs are still buying
        # anything -- a training curve alone cannot tell fitting from memorizing.
        self._metrics = {"train_nll": list(np.mean(np.asarray(train_curves), axis=0))}
        if holdout_inputs is not None:
            self._metrics["holdout_nll"] = list(np.mean(np.asarray(holdout_curves), axis=0))
        for index, curve in enumerate(train_curves):
            self._metrics[f"train_nll_member_{index}"] = list(curve)
        return ProbabilisticEnsembleTransition(
            members=members,
            state_mean=state_mean,
            state_scale=state_scale,
            action_mean=action_mean,
            action_scale=action_scale,
            delta_mean=delta_mean,
            delta_scale=delta_scale,
            seed=self._seed,
        )

    def training_metrics(self) -> Dict[str, List[float]]:
        """Per-epoch curves of the most recent fit.

        Returns:
            ``train_nll`` and, when the dataset held rows back, ``holdout_nll``
            -- both means across members -- plus one ``train_nll_member_i`` per
            member.
        """
        return dict(self._metrics)


def _normalized(
    batch: Any,
    state_mean: np.ndarray,
    state_scale: np.ndarray,
    action_mean: np.ndarray,
    action_scale: np.ndarray,
    delta_mean: np.ndarray,
    delta_scale: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Normalized inputs and targets of a batch, using the training split's statistics.

    The statistics are the *training* split's on purpose: normalizing the holdout
    by its own would make its loss incomparable to the training loss it is
    plotted against.

    Returns:
        ``(inputs, targets)``, or ``(None, None)`` for an empty batch.
    """
    if len(batch) == 0:
        return None, None
    inputs = np.concatenate(
        [
            (batch.states - state_mean) / state_scale,
            (batch.actions - action_mean) / action_scale,
        ],
        axis=1,
    )
    return inputs, (batch.deltas - delta_mean) / delta_scale


def _gaussian_nll(member: Any, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood, up to the constant term."""
    mean, log_variance = member(inputs)
    residual = targets - mean
    # The constant term shifts every model equally and only makes the reported
    # number harder to compare against a hand-computed one.
    return torch.mean(torch.sum(residual**2 * torch.exp(-log_variance) + log_variance, dim=-1))


def _train_member(
    member: Any,
    inputs: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: Optional[torch.device],
    holdout_inputs: Optional[np.ndarray] = None,
    holdout_targets: Optional[np.ndarray] = None,
) -> Tuple[List[float], List[float]]:
    """Train one member by Gaussian negative log-likelihood.

    Returns:
        ``(train_losses, holdout_losses)``, one entry per epoch. The second is
        empty when no holdout rows were given.
    """
    optimizer = torch.optim.Adam(
        member.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    input_tensor = torch.as_tensor(inputs, dtype=torch.float32)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32)
    if device is not None:
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

    holdout_input_tensor = None
    holdout_target_tensor = None
    if holdout_inputs is not None and holdout_targets is not None:
        holdout_input_tensor = torch.as_tensor(holdout_inputs, dtype=torch.float32)
        holdout_target_tensor = torch.as_tensor(holdout_targets, dtype=torch.float32)
        if device is not None:
            holdout_input_tensor = holdout_input_tensor.to(device)
            holdout_target_tensor = holdout_target_tensor.to(device)

    rng = np.random.default_rng(seed)
    num_rows = input_tensor.shape[0]
    steps = max(num_rows // batch_size, 1)
    member.train()
    losses: List[float] = []
    holdout_losses: List[float] = []
    for _ in range(epochs):
        order = rng.permutation(num_rows)
        epoch_loss = 0.0
        for step in range(steps):
            rows = order[step * batch_size : (step + 1) * batch_size]
            if rows.size == 0:
                continue
            index = torch.as_tensor(rows, dtype=torch.long, device=input_tensor.device)
            loss = _gaussian_nll(member, input_tensor[index], target_tensor[index])
            loss = loss + member.bound_penalty()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
        losses.append(epoch_loss / steps)
        if holdout_input_tensor is not None:
            member.eval()
            with torch.no_grad():
                holdout_losses.append(
                    float(_gaussian_nll(member, holdout_input_tensor, holdout_target_tensor))
                )
            member.train()
    member.eval()
    return losses, holdout_losses


def _mean_negative_log_likelihood(model: TransitionModel, batch: TransitionBatch) -> float:
    """Mean negative log-density the model assigns to the observed successors."""
    if len(batch) == 0:
        return float("nan")
    total = 0.0
    for state, action, next_state in zip(batch.states, batch.actions, batch.next_states):
        total += float(model.log_probability(state, action, next_state[None, :])[0])
    return -total / len(batch)
