# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Policy factory module for creating POMDP policies."""

from typing import Any, Dict, Type

from POMDPPlanners.planners.planners_utils.path_simulations_policy import (
    PathSimulationPolicy,
    PathSimulationPolicyCostSetting,
)
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.planners.mcts_planners.pomcp_dpw import POMCP_DPW
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero import BetaZero
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_zero import (
    ConstrainedZero,
)
from POMDPPlanners.planners.mcts_planners.icvar_pft_dpw import ICVaR_PFT_DPW
from POMDPPlanners.planners.mcts_planners.icvar_pomcpow import ICVaR_POMCPOW
from POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner import (
    DiscreteActionSequencesPlanner,
)
from POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling import (
    SparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.planners.sparse_sampling_planners.icvar_sparse_sampling import (
    ICVaRSparseSampling,
)

__all__ = [
    "POMCP",
    "SparseSamplingDiscreteActionsPlanner",
    "SparsePFT",
    "PathSimulationPolicy",
    "POMCPOW",
    "PFT_DPW",
    "POMCP_DPW",
    "DiscreteActionSequencesPlanner",
    "BetaZero",
    "ConstrainedZero",
    "ICVaR_PFT_DPW",
    "ICVaR_POMCPOW",
    "PathSimulationPolicyCostSetting",
    "ICVaRSparseSampling",
]

# Registry of available policies
POLICY_REGISTRY: Dict[str, Type] = {
    "POMCP": POMCP,
    "SparseSamplingDiscreteActionsPlanner": SparseSamplingDiscreteActionsPlanner,
    "SparsePFT": SparsePFT,
    "PathSimulationPolicy": PathSimulationPolicy,
    "POMCPOW": POMCPOW,
    "PFT_DPW": PFT_DPW,
    "POMCP_DPW": POMCP_DPW,
    "DiscreteActionSequencesPlanner": DiscreteActionSequencesPlanner,
    "BetaZero": BetaZero,
    "ConstrainedZero": ConstrainedZero,
    "ICVaR_PFT_DPW": ICVaR_PFT_DPW,
    "ICVaR_POMCPOW": ICVaR_POMCPOW,
    "ICVaRSparseSampling": ICVaRSparseSampling,
}


def get_policy(policy_type: str, **kwargs) -> Any:
    """
    Factory function to create policy instances.

    Args:
        policy_type: Type of policy to create
        **kwargs: Additional arguments to pass to the policy constructor

    Returns:
        An instance of the requested policy

    Raises:
        ValueError: If the policy type is not supported
    """
    if policy_type not in POLICY_REGISTRY:
        raise ValueError(
            f"Unsupported policy type: {policy_type}. "
            f"Available types: {list(POLICY_REGISTRY.keys())}"
        )

    return POLICY_REGISTRY[policy_type](**kwargs)
