"""Training utilities for BetaZero network.

This module provides the loss function and training loop used during
BetaZero policy iteration to update the dual-head network.

Functions:
    compute_beta_zero_loss: Combined value + policy + L2 loss (Eq. 7 in paper).
    train_network: Run multiple epochs of training on a replay buffer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from POMDPPlanners.planners.mcts_planners.beta_zero.training_buffer import (
    TrainingBuffer,
)

if TYPE_CHECKING:
    from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_network import (
        BetaZeroNetwork,
    )

_GRAD_NORM_KEYS = [
    "grad_norm/global",
    "grad_norm/trunk",
    "grad_norm/policy_head",
    "grad_norm/value_head",
]


def compute_beta_zero_loss(
    network: BetaZeroNetwork,
    belief_features: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the BetaZero combined loss.

    L = MSE(v, g_t) + CrossEntropy(p, π_t)    (Eq. 7)

    For discrete action spaces the policy loss is cross-entropy between the
    network's log-softmax output and the target distribution. For continuous
    action spaces a Gaussian negative-log-likelihood is used.

    Args:
        network: The BetaZero network.
        belief_features: Batch of belief feature vectors, shape ``(B, belief_dim)``.
        policy_targets: Batch of policy targets, shape ``(B, policy_dim)``.
        value_targets: Batch of scalar value targets, shape ``(B,)``.

    Returns:
        Tuple of (total_loss, component_dict) where component_dict contains
        ``"value_loss"`` and ``"policy_loss"`` as Python floats.
    """
    log_policy, value_pred = network(belief_features)
    value_pred = value_pred.squeeze(-1)

    value_loss = nn.functional.mse_loss(value_pred, value_targets)

    if network.action_space_type == "discrete":
        policy_loss = _discrete_policy_loss(log_policy, policy_targets)
    else:
        policy_loss = _continuous_policy_loss(log_policy, policy_targets)

    total_loss = value_loss + policy_loss

    loss_dict = {
        "value_loss": value_loss.item(),
        "policy_loss": policy_loss.item(),
    }
    return total_loss, loss_dict


def _discrete_policy_loss(log_policy: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -torch.mean(torch.sum(targets * log_policy, dim=-1))


def _continuous_policy_loss(raw_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    half = raw_output.shape[-1] // 2
    mean = raw_output[..., :half]
    log_std = raw_output[..., half:]
    std = torch.exp(log_std) + 1e-6
    target_mean = targets[..., :half]
    nll = 0.5 * (((target_mean - mean) / std) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return nll.mean()


def _compute_gradient_norms(network: BetaZeroNetwork) -> Dict[str, float]:
    module_sq: Dict[str, float] = {"trunk": 0.0, "policy_head": 0.0, "value_head": 0.0}
    global_sq = 0.0
    for name, param in network.named_parameters():
        if param.grad is None:
            continue
        sq = param.grad.detach().norm(2).item() ** 2
        global_sq += sq
        for mod in module_sq:
            if name.startswith(mod):
                module_sq[mod] += sq
                break
    return {
        "grad_norm/global": global_sq**0.5,
        "grad_norm/trunk": module_sq["trunk"] ** 0.5,
        "grad_norm/policy_head": module_sq["policy_head"] ** 0.5,
        "grad_norm/value_head": module_sq["value_head"] ** 0.5,
    }


def _compute_weight_norms(network: BetaZeroNetwork) -> Dict[str, float]:
    total_sq = 0.0
    for param in network.parameters():
        total_sq += param.detach().norm(2).item() ** 2
    return {"weight_norm/global": total_sq**0.5}


def train_network(
    network: BetaZeroNetwork,
    buffer: TrainingBuffer,
    n_epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    track_gradients: bool = False,
) -> Dict[str, List[float]]:
    """Train the network for multiple epochs on buffered data.

    Args:
        network: Network to train (modified in-place).
        buffer: Replay buffer with training examples.
        n_epochs: Number of full passes over the buffer.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        weight_decay: L2 regularisation coefficient (λ in Eq. 7).

    Returns:
        Dictionary with per-epoch loss lists: ``"total_loss"``,
        ``"value_loss"``, ``"policy_loss"``. When ``track_gradients`` is
        ``True``, also includes ``"grad_norm/global"``,
        ``"grad_norm/trunk"``, ``"grad_norm/policy_head"``,
        ``"grad_norm/value_head"``, and ``"weight_norm/global"``.
    """
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    metrics: Dict[str, List[float]] = {
        "total_loss": [],
        "value_loss": [],
        "policy_loss": [],
    }
    if track_gradients:
        for key in _GRAD_NORM_KEYS + ["weight_norm/global"]:
            metrics[key] = []

    n_batches = max(len(buffer) // batch_size, 1)

    for _ in range(n_epochs):
        epoch_totals = {"total": 0.0, "value": 0.0, "policy": 0.0}
        epoch_grad_totals: Dict[str, float] = (
            {k: 0.0 for k in _GRAD_NORM_KEYS} if track_gradients else {}
        )

        for _ in range(n_batches):
            beliefs_np, policies_np, values_np = buffer.sample_batch(batch_size)
            beliefs_t = torch.as_tensor(beliefs_np, dtype=torch.float32)
            policies_t = torch.as_tensor(policies_np, dtype=torch.float32)
            values_t = torch.as_tensor(values_np, dtype=torch.float32)

            loss, components = compute_beta_zero_loss(network, beliefs_t, policies_t, values_t)

            optimizer.zero_grad()
            loss.backward()

            if track_gradients:
                for key, val in _compute_gradient_norms(network).items():
                    epoch_grad_totals[key] = epoch_grad_totals.get(key, 0.0) + val

            optimizer.step()

            epoch_totals["total"] += loss.item()
            epoch_totals["value"] += components["value_loss"]
            epoch_totals["policy"] += components["policy_loss"]

        metrics["total_loss"].append(epoch_totals["total"] / n_batches)
        metrics["value_loss"].append(epoch_totals["value"] / n_batches)
        metrics["policy_loss"].append(epoch_totals["policy"] / n_batches)

        if track_gradients:
            for key in _GRAD_NORM_KEYS:
                metrics[key].append(epoch_grad_totals[key] / n_batches)
            for key, val in _compute_weight_norms(network).items():
                metrics[key].append(val)

    return metrics
