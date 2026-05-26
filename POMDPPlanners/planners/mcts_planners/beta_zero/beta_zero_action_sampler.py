# SPDX-License-Identifier: MIT

"""Network-guided action sampler for BetaZero progressive widening.

This module provides an ``ActionSampler`` subclass that draws new candidate
actions from the policy network's output distribution, enabling the progressive
widening mechanism to propose actions guided by learned priors rather than
uniform random sampling.

Reference:
    Moss, R. J., Corso, A., Caers, J., & Kochenderfer, M. J. (2024). BetaZero:
    Belief-State Planning for Long-Horizon POMDPs using Learned Approximations.
    Reinforcement Learning Conference (RLC).

Classes:
    BetaZeroActionSampler: Samples actions from the BetaZero policy network.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from POMDPPlanners.core.tree import BeliefNode
from POMDPPlanners.planners.mcts_planners.beta_zero.belief_representation import (
    BeliefRepresentation,
)
from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_network import (
    AbstractBetaZeroNetwork,
)
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler


class BetaZeroActionSampler(ActionSampler):
    """Action sampler that draws from the BetaZero policy network.

    For **discrete** action spaces, samples categorically from the network's
    softmax distribution. For **continuous** action spaces, samples from the
    predicted Gaussian and adds exploration noise.

    When no belief node is available (e.g. during random rollout) or the
    network has not been set, the sampler delegates to a ``fallback_sampler``.

    Args:
        fallback_sampler: Sampler used when the network is not available.
            Optional for pickle deserialization compatibility; should always
            be provided during normal construction.
        actions: List of discrete actions (required for discrete spaces).
        noise_scale: Standard deviation of exploration noise added to
            continuous action samples.
    """

    def __init__(
        self,
        fallback_sampler: Optional[ActionSampler] = None,
        actions: Optional[List[Any]] = None,
        noise_scale: float = 0.1,
    ):
        # For compatibility with pickle deserialization via __reduce__
        self.fallback_sampler = fallback_sampler
        self.actions = actions
        self.noise_scale = noise_scale
        # Set externally by BetaZero planner once network + representation exist
        self._network: Optional[AbstractBetaZeroNetwork] = None
        self._belief_representation = None
        self._normalize_fn: Callable[[np.ndarray], np.ndarray] = lambda x: x

    def set_network_and_representation(
        self,
        network: AbstractBetaZeroNetwork,
        belief_representation: BeliefRepresentation,
        normalize_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        """Attach the network, belief representation, and optional normalizer.

        Args:
            network: ``AbstractBetaZeroNetwork`` instance.
            belief_representation: ``BeliefRepresentation`` instance.
            normalize_fn: Optional callable that normalizes raw belief features
                before passing them to the network. When ``None``, features are
                passed through unchanged. Pass ``planner._get_normalized_features``
                to keep the sampler in sync with BetaZero's normalization stats.
        """
        self._network = network
        self._belief_representation = belief_representation
        self._normalize_fn = normalize_fn if normalize_fn is not None else (lambda x: x)

    def sample(self, belief_node: Optional[BeliefNode] = None) -> Any:
        """Sample a new action for progressive widening.

        Args:
            belief_node: Optional current belief node for informed sampling.

        Returns:
            A sampled action.
        """
        if not self._can_use_network(belief_node):
            return self.fallback_sampler.sample(belief_node)

        features = self._normalize_fn(self._belief_representation(belief_node.belief))
        policy, _ = self._network.predict(features)

        if self.actions is not None:
            return self._sample_discrete(policy)
        return self._sample_continuous(policy)

    def _can_use_network(self, belief_node: Optional[BeliefNode]) -> bool:
        return (
            self._network is not None
            and self._belief_representation is not None
            and belief_node is not None
        )

    def _sample_discrete(self, probs: np.ndarray) -> Any:
        assert self.actions is not None
        probs = np.maximum(probs, 0.0)
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            probs = np.ones(len(self.actions)) / len(self.actions)
        idx = np.random.choice(len(self.actions), p=probs)
        return self.actions[idx]

    def _sample_continuous(self, policy_output: np.ndarray) -> np.ndarray:
        half = len(policy_output) // 2
        mean = policy_output[:half]
        log_std = policy_output[half:]
        std = np.exp(log_std) + self.noise_scale
        return np.random.normal(mean, std).astype(np.float32)

    # ── Serialisation ─────────────────────────────────────────────────

    def __getstate__(self) -> Dict[str, Any]:
        """Get state for pickle serialization."""
        state = self.__dict__.copy()
        state["_network"] = None
        state["_belief_representation"] = None
        state["_normalize_fn"] = None  # lambdas/bound methods are not picklable
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set state for pickle deserialization."""
        for key, value in state.items():
            setattr(self, key, value)
        if self._normalize_fn is None:
            self._normalize_fn = lambda x: x

    def __reduce__(self):
        """Support for pickle serialization via __reduce__."""
        cls = self.__class__
        state = self.__getstate__()
        return (cls, (), state)
