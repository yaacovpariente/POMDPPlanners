# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Dual-head neural network for BetaZero policy and value prediction.

This module provides the PyTorch neural network used by BetaZero. The network
has a shared trunk with two output heads: a policy head that produces action
probabilities (discrete) or Gaussian parameters (continuous), and a value head
that estimates the state value V(φ(b)).

Reference:
    Moss, R. J., Corso, A., Caers, J., & Kochenderfer, M. J. (2024). BetaZero:
    Belief-State Planning for Long-Horizon POMDPs using Learned Approximations.
    Reinforcement Learning Conference (RLC).

Classes:
    AbstractBetaZeroNetwork: Abstract base class for BetaZero policy and value networks.
    BetaZeroNetwork: Shared-trunk network with policy and value heads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from POMDPPlanners.planners.mcts_planners.beta_zero.training_buffer import TrainingBuffer
from POMDPPlanners.planners.mcts_planners.beta_zero.training import train_network


class AbstractBetaZeroNetwork(ABC):
    """Abstract base class for BetaZero policy and value networks.

    Defines the inference and training interface required by the BetaZero
    planner. Concrete subclasses provide the underlying model architecture.

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    @property
    @abstractmethod
    def action_space_type(self) -> str:
        """Action space type: ``"discrete"`` or ``"continuous"``."""

    @property
    def n_actions(self) -> Optional[int]:
        """Number of discrete actions. ``None`` for continuous spaces."""
        return None

    @property
    def action_dim(self) -> Optional[int]:
        """Dimensionality of continuous actions. ``None`` for discrete spaces."""
        return None

    @abstractmethod
    def predict(self, belief_features: np.ndarray) -> Tuple[np.ndarray, float]:
        """Single-sample inference returning (policy, value)."""

    @abstractmethod
    def predict_batch(self, belief_features_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batched inference returning (policies, values)."""

    @abstractmethod
    def fit(
        self,
        buffer: TrainingBuffer,
        n_epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        track_gradients: bool,
    ) -> Dict[str, List[float]]:
        """Train on replay buffer and return per-epoch loss metrics."""

    @abstractmethod
    def save_weights(self, filepath: Path) -> None:
        """Persist network weights to disk."""

    @abstractmethod
    def load_weights(self, filepath: Path) -> None:
        """Load network weights from disk."""


class BetaZeroNetwork(AbstractBetaZeroNetwork, nn.Module):
    """Dual-head neural network for BetaZero.

    Architecture:
      - **Shared trunk**: ``Linear(belief_dim, h) → ReLU → Linear(h, h) → ReLU``
      - **Policy head (discrete)**: ``Linear(h, h//2) → ReLU → Linear(h//2, n_actions) → LogSoftmax``
      - **Policy head (continuous)**: ``Linear(h, h//2) → ReLU → Linear(h//2, 2*action_dim)`` (mean, log_std)
      - **Value head**: ``Linear(h, h//2) → ReLU → Linear(h//2, 1)``

    Args:
        belief_dim: Dimensionality of the belief feature vector φ(b).
        action_space_type: ``"discrete"`` or ``"continuous"``.
        n_actions: Number of discrete actions (required when ``action_space_type="discrete"``).
        action_dim: Dimensionality of continuous actions (required when ``action_space_type="continuous"``).
        hidden_sizes: Tuple of hidden layer widths for the shared trunk.

    Raises:
        ValueError: If required parameters for the chosen action space type are missing.

    Example:
        >>> import torch, numpy as np
        >>> from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_network import BetaZeroNetwork
        >>> net = BetaZeroNetwork(belief_dim=4, action_space_type="discrete", n_actions=3)
        >>> policy, value = net.predict(np.zeros(4, dtype=np.float32))
        >>> policy.shape
        (3,)
        >>> isinstance(value, float)
        True
    """

    def __init__(
        self,
        belief_dim: int,
        action_space_type: str,
        n_actions: Optional[int] = None,
        action_dim: Optional[int] = None,
        hidden_sizes: Sequence[int] = (128, 128),
    ):
        super().__init__()
        self._validate_params(action_space_type, n_actions, action_dim)

        self.belief_dim = belief_dim
        self._action_space_type = action_space_type
        self._n_actions = n_actions
        self._action_dim = action_dim
        self.hidden_sizes = tuple(hidden_sizes)

        self._build_trunk(belief_dim, hidden_sizes)
        self._build_policy_head(hidden_sizes[-1])
        self._build_value_head(hidden_sizes[-1])

    # ── Properties ────────────────────────────────────────────────────

    @property
    def action_space_type(self) -> str:
        return self._action_space_type

    @property
    def n_actions(self) -> Optional[int]:
        return self._n_actions

    @property
    def action_dim(self) -> Optional[int]:
        return self._action_dim

    # ── Construction helpers ──────────────────────────────────────────

    @staticmethod
    def _validate_params(
        action_space_type: str,
        n_actions: Optional[int],
        action_dim: Optional[int],
    ) -> None:
        if action_space_type not in ("discrete", "continuous"):
            raise ValueError(
                f"action_space_type must be 'discrete' or 'continuous', got '{action_space_type}'"
            )
        if action_space_type == "discrete" and n_actions is None:
            raise ValueError("n_actions is required for discrete action spaces")
        if action_space_type == "continuous" and action_dim is None:
            raise ValueError("action_dim is required for continuous action spaces")

    def _build_trunk(self, input_dim: int, hidden_sizes: Sequence[int]) -> None:
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.trunk = nn.Sequential(*layers)

    def _build_policy_head(self, trunk_out: int) -> None:
        mid = max(trunk_out // 2, 1)
        if self._action_space_type == "discrete":
            assert self._n_actions is not None  # validated in _validate_params
            self.policy_head = nn.Sequential(
                nn.Linear(trunk_out, mid),
                nn.ReLU(),
                nn.Linear(mid, self._n_actions),
                nn.LogSoftmax(dim=-1),
            )
        else:
            assert self._action_dim is not None  # validated in _validate_params
            self.policy_head = nn.Sequential(
                nn.Linear(trunk_out, mid),
                nn.ReLU(),
                nn.Linear(mid, 2 * self._action_dim),
            )

    def _build_value_head(self, trunk_out: int) -> None:
        mid = max(trunk_out // 2, 1)
        self.value_head = nn.Sequential(
            nn.Linear(trunk_out, mid),
            nn.ReLU(),
            nn.Linear(mid, 1),
        )

    # ── Forward / predict ─────────────────────────────────────────────

    def forward(self, belief_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning raw policy and value outputs.

        Args:
            belief_features: Tensor of shape ``(batch, belief_dim)`` or ``(belief_dim,)``.

        Returns:
            Tuple of (policy_output, value) tensors.
            - Discrete policy: log-probabilities of shape ``(batch, n_actions)``.
            - Continuous policy: ``[mean, log_std]`` of shape ``(batch, 2*action_dim)``.
            - Value: shape ``(batch, 1)``.
        """
        h = self.trunk(belief_features)
        return self.policy_head(h), self.value_head(h)

    def predict(self, belief_features: np.ndarray) -> Tuple[np.ndarray, float]:
        """Single-sample inference returning numpy arrays.

        Runs in ``torch.no_grad()`` mode. For discrete action spaces the output
        policy is exponentiated to give probabilities. Supports both CPU and
        CUDA: input is moved to the model device and outputs are moved to CPU
        for numpy conversion.

        Args:
            belief_features: 1-D array of shape ``(belief_dim,)``.

        Returns:
            Tuple of (policy, value).
            - Discrete: policy is a probability vector summing to 1.
            - Continuous: policy is ``[mean, log_std]``.
            - value is a Python float.
        """
        device = next(self.parameters()).device
        x = torch.as_tensor(belief_features, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            log_policy, value = self.forward(x)
        log_policy = log_policy.squeeze(0)
        if log_policy.is_cuda:
            log_policy = log_policy.cpu()
        policy_np = log_policy.numpy()
        if self._action_space_type == "discrete":
            policy_np = np.exp(policy_np)
        return policy_np, float(value.item())

    def predict_batch(self, belief_features_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batched inference returning numpy arrays.

        Runs in ``torch.no_grad()`` mode. Processes multiple belief feature
        vectors in a single forward pass for efficiency. Supports both CPU and
        CUDA; outputs are moved to CPU for numpy conversion when on CUDA.

        Args:
            belief_features_batch: 2-D array of shape ``(N, belief_dim)``.

        Returns:
            Tuple of (policies, values).
            - Discrete: policies is ``(N, n_actions)`` probability matrix.
            - Continuous: policies is ``(N, 2*action_dim)`` with ``[mean, log_std]``.
            - values is ``(N,)`` array of floats.
        """
        device = next(self.parameters()).device
        x = torch.as_tensor(belief_features_batch, dtype=torch.float32, device=device)
        with torch.no_grad():
            log_policy, value = self.forward(x)
        if log_policy.is_cuda:
            log_policy = log_policy.cpu()
        value = value.squeeze(-1)
        if value.is_cuda:
            value = value.cpu()
        policy_np = log_policy.numpy()
        if self._action_space_type == "discrete":
            policy_np = np.exp(policy_np)
        return policy_np, value.numpy()

    # ── Training ──────────────────────────────────────────────────────

    def fit(
        self,
        buffer: TrainingBuffer,
        n_epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        track_gradients: bool,
    ) -> Dict[str, List[float]]:
        """Train the network on a replay buffer.

        Args:
            buffer: Replay buffer with training examples.
            n_epochs: Number of full passes over the buffer.
            batch_size: Mini-batch size.
            learning_rate: Adam learning rate.
            weight_decay: L2 regularisation coefficient.
            track_gradients: When ``True``, gradient and weight norms are
                included in the returned metrics.

        Returns:
            Dictionary with per-epoch loss lists.
        """
        return train_network(
            network=self,
            buffer=buffer,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            track_gradients=track_gradients,
        )

    # ── Serialisation ─────────────────────────────────────────────────

    def save_weights(self, filepath: Path) -> None:
        """Save network weights to disk.

        Args:
            filepath: Destination ``.pt`` file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load_weights(self, filepath: Path) -> None:
        """Load network weights from disk.

        Args:
            filepath: Source ``.pt`` file.
        """
        state = torch.load(filepath, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
