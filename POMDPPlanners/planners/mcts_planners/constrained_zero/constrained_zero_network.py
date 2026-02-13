"""Three-head neural network for ConstrainedZero.

This module extends the BetaZero dual-head network with an additional failure
probability head. The failure head outputs a raw logit; sigmoid is applied
during prediction to produce a probability in [0, 1].

Classes:
    ConstrainedZeroNetwork: Shared-trunk network with policy, value, and failure heads.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_network import (
    BetaZeroNetwork,
)


class ConstrainedZeroNetwork(BetaZeroNetwork):
    """Three-head neural network for ConstrainedZero.

    Architecture:
      - **Shared trunk**: inherited from ``BetaZeroNetwork``
      - **Policy head**: inherited from ``BetaZeroNetwork``
      - **Value head**: inherited from ``BetaZeroNetwork``
      - **Failure head**: ``Linear(h, h//2) -> ReLU -> Linear(h//2, 1)``

    The failure head outputs a raw logit. During ``predict()``, sigmoid is
    applied to produce a failure probability in [0, 1].

    Args:
        belief_dim: Dimensionality of the belief feature vector phi(b).
        action_space_type: ``"discrete"`` or ``"continuous"``.
        n_actions: Number of discrete actions (required when ``action_space_type="discrete"``).
        action_dim: Dimensionality of continuous actions (required when ``action_space_type="continuous"``).
        hidden_sizes: Tuple of hidden layer widths for the shared trunk.

    Example:
        >>> import torch, numpy as np
        >>> from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_zero_network import ConstrainedZeroNetwork
        >>> net = ConstrainedZeroNetwork(belief_dim=4, action_space_type="discrete", n_actions=3)
        >>> policy, value, failure_prob = net.predict(np.zeros(4, dtype=np.float32))
        >>> policy.shape
        (3,)
        >>> isinstance(value, float)
        True
        >>> 0.0 <= failure_prob <= 1.0
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
        super().__init__(
            belief_dim=belief_dim,
            action_space_type=action_space_type,
            n_actions=n_actions,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        )
        self._build_failure_head(hidden_sizes[-1])

    def _build_failure_head(self, trunk_out: int) -> None:
        mid = max(trunk_out // 2, 1)
        self.failure_head = nn.Sequential(
            nn.Linear(trunk_out, mid),
            nn.ReLU(),
            nn.Linear(mid, 1),
        )

    def forward(  # type: ignore[override]
        self, belief_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning raw policy, value, and failure logit.

        Args:
            belief_features: Tensor of shape ``(batch, belief_dim)`` or ``(belief_dim,)``.

        Returns:
            Tuple of (policy_output, value, failure_logit) tensors.
        """
        h = self.trunk(belief_features)
        return self.policy_head(h), self.value_head(h), self.failure_head(h)

    def predict(  # type: ignore[override]
        self, belief_features: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Single-sample inference returning numpy policy, value, and failure probability.

        Args:
            belief_features: 1-D array of shape ``(belief_dim,)``.

        Returns:
            Tuple of (policy, value, failure_prob).
            - Discrete: policy is a probability vector summing to 1.
            - Continuous: policy is ``[mean, log_std]``.
            - value is a Python float.
            - failure_prob is a Python float in [0, 1].
        """
        x = torch.as_tensor(belief_features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            log_policy, value, failure_logit = self.forward(x)
        policy_np = log_policy.squeeze(0).numpy()
        if self.action_space_type == "discrete":
            policy_np = np.exp(policy_np)
        failure_prob = float(torch.sigmoid(failure_logit).item())
        return policy_np, float(value.item()), failure_prob
