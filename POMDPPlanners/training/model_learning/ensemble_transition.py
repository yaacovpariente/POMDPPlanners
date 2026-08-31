# SPDX-License-Identifier: MIT

"""A transition fitted as an ensemble of Gaussian networks over the state change.

Why a *distribution* and not a point prediction. The guarantee that makes
iterative model learning work runs through the distance between the predicted
and the true next-state distributions, so a model trained on squared error sits
outside it entirely. The practical consequence is sharper than the theoretical
one: a deterministic transition collapses a particle belief to a point, and a
planner grading a tail measure then has no tail to grade. The model must be able
to say how unsure it is, and be right about it.

Why an ensemble and not one network. Two kinds of uncertainty matter and they
behave differently. Noise the system really has (a slipping foot) is aleatoric,
and a single network's variance head captures it. Ignorance about a region no
rollout visited is epistemic, and a single network is confidently wrong there --
which is precisely the region an iterative loop drives into. Disagreement
between independently initialised members is the cheap estimate of it. Sampling
therefore draws a *member* first and then draws from that member, so
disagreement widens the belief instead of being averaged into a mean nobody
predicted.

Why the variance head is bounded. Left free, the fastest way to reduce Gaussian
negative log-likelihood early in training is to declare enormous variance on the
hard dimensions and stop modelling them. Soft bounds, learned but penalized,
stop that without hard-clipping a variance that genuinely should be large.

Classes:
    GaussianMLP: One ensemble member -- predicts mean and variance of the state change.
    ProbabilisticEnsembleTransition: Ensemble of members, usable as a TransitionModel.
"""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from POMDPPlanners.core.environment import TransitionModel
from POMDPPlanners.core.environment.array_backend import as_backend, as_rows, is_tensor

#: Weight on the soft-bound penalty. Small: it is a nudge keeping the bounds
#: honest, not a term the fit should trade accuracy against.
_BOUND_PENALTY = 0.01

#: Floor on a normalization scale, so a channel that never moves in the training
#: data does not divide the whole design matrix by zero.
_MIN_SCALE = 1e-6


class GaussianMLP(nn.Module):
    """One ensemble member: ``(state, action) -> mean and variance of the state change``.

    Args:
        input_dim: Width of the concatenated ``(state, action)`` input.
        output_dim: Width of the state block being predicted.
        hidden_sizes: Widths of the hidden layers.

    Attributes:
        max_log_variance: Learned upper soft bound on the log-variance.
        min_log_variance: Learned lower soft bound on the log-variance.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int] = (200, 200),
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        width = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(width, hidden))
            layers.append(nn.SiLU())
            width = hidden
        layers.append(nn.Linear(width, 2 * output_dim))
        self._net = nn.Sequential(*layers)
        self._output_dim = output_dim
        self.max_log_variance = nn.Parameter(torch.full((output_dim,), 0.5))
        self.min_log_variance = nn.Parameter(torch.full((output_dim,), -10.0))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the mean and log-variance of the normalized state change.

        Args:
            inputs: ``(N, input_dim)`` normalized ``(state, action)`` rows.

        Returns:
            ``(mean, log_variance)``, each ``(N, output_dim)``. The log-variance
            is squeezed between the two learned bounds with softplus, so it
            approaches them smoothly instead of being clipped flat.
        """
        raw = self._net(inputs)
        mean, raw_log_variance = raw[..., : self._output_dim], raw[..., self._output_dim :]
        log_variance = self.max_log_variance - nn.functional.softplus(
            self.max_log_variance - raw_log_variance
        )
        log_variance = self.min_log_variance + nn.functional.softplus(
            log_variance - self.min_log_variance
        )
        return mean, log_variance

    def bound_penalty(self) -> torch.Tensor:
        """Penalty pulling the soft bounds inward, so they track the data."""
        return _BOUND_PENALTY * (self.max_log_variance.sum() - self.min_log_variance.sum())


class ProbabilisticEnsembleTransition(TransitionModel):
    """Ensemble of Gaussian networks, planned with as an ordinary transition.

    Sampling draws one member uniformly per sample and then draws from that
    member's Gaussian, so the model's own disagreement is what widens a belief.
    The density is the matching uniform mixture over members -- the honest
    log-density of what sampling actually does, which is what a particle filter
    needs in order to weight correctly.

    Args:
        members: Trained :class:`GaussianMLP` members.
        state_mean: ``(dim,)`` mean of the training states, for input normalization.
        state_scale: ``(dim,)`` scale of the training states.
        action_mean: ``(action_dim,)`` mean of the training actions.
        action_scale: ``(action_dim,)`` scale of the training actions.
        delta_mean: ``(dim,)`` mean of the training state changes.
        delta_scale: ``(dim,)`` scale of the training state changes.
        seed: Seed for the member choice and the noise draw.

    Raises:
        ValueError: If no members are supplied.
    """

    def __init__(
        self,
        members: Sequence[GaussianMLP],
        state_mean: np.ndarray,
        state_scale: np.ndarray,
        action_mean: np.ndarray,
        action_scale: np.ndarray,
        delta_mean: np.ndarray,
        delta_scale: np.ndarray,
        seed: int = 0,
    ) -> None:
        if not members:
            raise ValueError("a probabilistic ensemble needs at least one member")
        self._members = list(members)
        for member in self._members:
            member.eval()
        self._state_mean = np.asarray(state_mean, dtype=float)
        self._state_scale = np.asarray(state_scale, dtype=float)
        self._action_mean = np.asarray(action_mean, dtype=float)
        self._action_scale = np.asarray(action_scale, dtype=float)
        self._delta_mean = np.asarray(delta_mean, dtype=float)
        self._delta_scale = np.asarray(delta_scale, dtype=float)
        self._rng = np.random.default_rng(seed)

    @property
    def dim(self) -> int:
        """Width of the state block this transition predicts."""
        return int(self._delta_mean.size)

    @property
    def num_members(self) -> int:
        """Number of ensemble members."""
        return len(self._members)

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> Any:
        """Sample next states, each from a uniformly drawn member.

        Args:
            state: A ``(dim,)`` state block, or a ``(N, dim)`` batch -- one row
                per particle, each with its own action.
            action: The action applied at ``state``, or one per row.
            n_samples: Draws per state.

        Returns:
            For a single state: ``(dim,)`` when ``n_samples == 1``, else
            ``(n_samples, dim)``. For a batch: ``(N, dim)`` when
            ``n_samples == 1``, else ``(N, n_samples, dim)``. The result follows
            the state's backend.

        Note:
            A member is drawn **per row and per sample**, never once for the whole
            batch. Drawing one member per call would replace the ensemble's
            disagreement with a single member's opinion, and that disagreement is
            the epistemic spread a risk-sensitive planner is there to price.
        """
        rows, batched = as_rows(state, self.dim)
        means, variances = self._predict_rows(rows, action)  # (M, N, dim)
        num_rows = int(rows.shape[0])
        choice = self._rng.integers(len(self._members), size=(num_rows, n_samples))
        noise = self._rng.standard_normal((num_rows, n_samples, self.dim))
        row_index = np.arange(num_rows)[:, None]
        picked_mean = means[choice, row_index]  # (N, n_samples, dim)
        picked_var = variances[choice, row_index]
        deltas = picked_mean + noise * np.sqrt(picked_var)
        base = rows.detach().cpu().numpy() if is_tensor(rows) else rows
        next_states = base[:, None, :] + deltas
        result = _shape_ensemble_samples(next_states, batched, n_samples)
        return as_backend(result, state) if is_tensor(state) else result

    def log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        """Log-density of each next state under the uniform mixture over members.

        Args:
            state: The current state block, a length-``dim`` vector.
            action: The action applied at ``state``.
            next_states: A ``(dim,)`` next state or a ``(n, dim)`` batch.

        Returns:
            A ``(n,)`` array of log-densities.
        """
        rows, batched = as_rows(state, self.dim)
        base = rows.detach().cpu().numpy() if is_tensor(rows) else np.asarray(rows, dtype=float)
        candidates = np.atleast_2d(
            np.asarray(
                next_states.detach().cpu().numpy() if is_tensor(next_states) else next_states,
                dtype=float,
            )
        )
        means, variances = self._predict_rows(rows, action)  # (M, N, dim)
        if batched:
            # Row-wise pairing: candidate i is scored under state i.
            deltas = candidates - base
            residual = deltas[None, :, :] - means
            var = variances
        else:
            # One state, many candidates.
            deltas = candidates - base[0][None, :]
            residual = deltas[None, :, :] - means[:, 0][:, None, :]
            var = variances[:, 0][:, None, :]
        per_member = -0.5 * (
            np.sum(residual**2 / var, axis=-1)
            + np.sum(np.log(var), axis=-1)
            + self.dim * np.log(2.0 * np.pi)
        )
        result = _log_mean_exp(per_member, axis=0)
        return as_backend(result, state) if is_tensor(state) else result

    def _predict_rows(self, rows: Any, action: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Per-member mean and variance of the state change, one column per row.

        Args:
            rows: ``(N, dim)`` states.
            action: The action, or one per row.

        Returns:
            ``(means, variances)``, each ``(num_members, N, dim)`` in raw state
            units.

        Note:
            Every member sees the whole batch in one forward pass. The scalar path
            used to build a one-row tensor per call, which for a learned model is
            the awkward case -- batching is the natural shape here, not an
            optimization bolted on afterwards.
        """
        states = rows.detach().cpu().numpy() if is_tensor(rows) else np.asarray(rows, dtype=float)
        actions, _ = as_rows(
            np.asarray(
                action.detach().cpu().numpy() if is_tensor(action) else action, dtype=float
            ),
            self._action_mean.size,
        )
        actions = np.broadcast_to(actions, (states.shape[0], actions.shape[1]))
        normalized = np.concatenate(
            [
                (states - self._state_mean) / self._state_scale,
                (actions - self._action_mean) / self._action_scale,
            ],
            axis=1,
        )
        inputs = torch.as_tensor(normalized, dtype=torch.float32)
        num_rows = states.shape[0]
        means = np.empty((len(self._members), num_rows, self.dim), dtype=float)
        variances = np.empty_like(means)
        with torch.no_grad():
            for index, member in enumerate(self._members):
                mean, log_variance = member(inputs)
                means[index] = mean.numpy()
                variances[index] = np.exp(log_variance.numpy())
        # Undo the target normalization: a scaled Gaussian scales its variance by
        # the square of the same factor.
        return (
            means * self._delta_scale + self._delta_mean,
            variances * self._delta_scale**2,
        )


def _shape_ensemble_samples(samples: np.ndarray, batched: bool, n_samples: int) -> np.ndarray:
    """Give a ``(N, n_samples, dim)`` draw back in the shape the caller asked for."""
    if not batched:
        return samples[0, 0] if n_samples == 1 else samples[0]
    return samples[:, 0] if n_samples == 1 else samples


def normalization_stats(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and scale of ``values``, with a floor on the scale.

    Args:
        values: ``(N, dim)`` array.

    Returns:
        ``(mean, scale)``, each ``(dim,)``. A channel that never moves gets the
        floor rather than a zero, so normalizing it is a no-op instead of a
        divide-by-zero.
    """
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), _MIN_SCALE)
    return mean, scale


def _log_mean_exp(values: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable ``log(mean(exp(values)))`` along ``axis``."""
    peak = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - peak)
    return np.squeeze(peak, axis=axis) + np.log(shifted.mean(axis=axis))


def build_members(
    input_dim: int,
    output_dim: int,
    num_members: int,
    hidden_sizes: Sequence[int],
    seed: int,
    device: Optional[torch.device] = None,
) -> List[GaussianMLP]:
    """Construct ``num_members`` independently initialised members.

    The initialisations must differ, because that difference *is* the epistemic
    uncertainty estimate; identical members would agree everywhere and report
    confidence the ensemble has not earned.

    Args:
        input_dim: Width of the concatenated ``(state, action)`` input.
        output_dim: Width of the predicted state block.
        num_members: Number of members.
        hidden_sizes: Hidden layer widths, shared by every member.
        seed: Base seed; member ``i`` is seeded with ``seed + i``.
        device: Device to place the members on.

    Returns:
        The constructed members.
    """
    members: List[GaussianMLP] = []
    for index in range(num_members):
        torch.manual_seed(seed + index)
        member = GaussianMLP(input_dim, output_dim, hidden_sizes)
        if device is not None:
            member.to(device)
        members.append(member)
    return members
