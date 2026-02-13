"""Training utilities for ConstrainedZero network.

This module provides the loss function and training loop for the 3-head
ConstrainedZero network. It extends the BetaZero training with an additional
binary cross-entropy loss for the failure head.

Functions:
    compute_constrained_zero_loss: Combined value + policy + failure loss.
    train_constrained_network: Multi-epoch training on a replay buffer.
"""

from typing import Dict, List, Tuple

import torch
from torch import nn

from POMDPPlanners.planners.mcts_planners.beta_zero.training import (
    _continuous_policy_loss,
    _discrete_policy_loss,
)
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_training_buffer import (
    ConstrainedTrainingBuffer,
)
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_zero_network import (
    ConstrainedZeroNetwork,
)


def compute_constrained_zero_loss(
    network: ConstrainedZeroNetwork,
    belief_features: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    failure_targets: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the ConstrainedZero combined loss.

    L = MSE(v, g_t) + CrossEntropy(p, pi_t) + BCE(failure_logit, f_t)

    Args:
        network: The ConstrainedZero 3-head network.
        belief_features: Batch of belief feature vectors, shape ``(B, belief_dim)``.
        policy_targets: Batch of policy targets, shape ``(B, policy_dim)``.
        value_targets: Batch of scalar value targets, shape ``(B,)``.
        failure_targets: Batch of binary failure targets, shape ``(B,)``.

    Returns:
        Tuple of (total_loss, component_dict) where component_dict contains
        ``"value_loss"``, ``"policy_loss"``, and ``"failure_loss"`` as Python floats.
    """
    log_policy, value_pred, failure_logit = network(belief_features)
    value_pred = value_pred.squeeze(-1)
    failure_logit = failure_logit.squeeze(-1)

    value_loss = nn.functional.mse_loss(value_pred, value_targets)

    if network.action_space_type == "discrete":
        policy_loss = _discrete_policy_loss(log_policy, policy_targets)
    else:
        policy_loss = _continuous_policy_loss(log_policy, policy_targets)

    failure_loss = nn.functional.binary_cross_entropy_with_logits(failure_logit, failure_targets)

    total_loss = value_loss + policy_loss + failure_loss

    loss_dict = {
        "value_loss": value_loss.item(),
        "policy_loss": policy_loss.item(),
        "failure_loss": failure_loss.item(),
    }
    return total_loss, loss_dict


def train_constrained_network(
    network: ConstrainedZeroNetwork,
    buffer: ConstrainedTrainingBuffer,
    n_epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Dict[str, List[float]]:
    """Train the 3-head network for multiple epochs on buffered data.

    Args:
        network: Network to train (modified in-place).
        buffer: Replay buffer with constrained training examples.
        n_epochs: Number of full passes over the buffer.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        weight_decay: L2 regularisation coefficient.

    Returns:
        Dictionary with per-epoch loss lists: ``"total_loss"``,
        ``"value_loss"``, ``"policy_loss"``, ``"failure_loss"``.
    """
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    metrics: Dict[str, List[float]] = {
        "total_loss": [],
        "value_loss": [],
        "policy_loss": [],
        "failure_loss": [],
    }

    n_batches = max(len(buffer) // batch_size, 1)

    for _ in range(n_epochs):
        epoch_totals = {"total": 0.0, "value": 0.0, "policy": 0.0, "failure": 0.0}

        for _ in range(n_batches):
            beliefs_np, policies_np, values_np, failures_np = buffer.sample_batch(batch_size)
            beliefs_t = torch.as_tensor(beliefs_np, dtype=torch.float32)
            policies_t = torch.as_tensor(policies_np, dtype=torch.float32)
            values_t = torch.as_tensor(values_np, dtype=torch.float32)
            failures_t = torch.as_tensor(failures_np, dtype=torch.float32)

            loss, components = compute_constrained_zero_loss(
                network, beliefs_t, policies_t, values_t, failures_t
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_totals["total"] += loss.item()
            epoch_totals["value"] += components["value_loss"]
            epoch_totals["policy"] += components["policy_loss"]
            epoch_totals["failure"] += components["failure_loss"]

        metrics["total_loss"].append(epoch_totals["total"] / n_batches)
        metrics["value_loss"].append(epoch_totals["value"] / n_batches)
        metrics["policy_loss"].append(epoch_totals["policy"] / n_batches)
        metrics["failure_loss"].append(epoch_totals["failure"] / n_batches)

    return metrics
