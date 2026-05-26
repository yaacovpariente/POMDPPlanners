# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""BetaZero: Neural MCTS for POMDPs.

This package implements the BetaZero algorithm (Moss et al., 2024), which adapts
AlphaZero to POMDPs by planning in belief space with learned neural network priors.

Reference:
    Moss, R. J., Corso, A., Caers, J., & Kochenderfer, M. J. (2024). BetaZero:
    Belief-State Planning for Long-Horizon POMDPs using Learned Approximations.
    Reinforcement Learning Conference (RLC).

Classes:
    BetaZero: Main planner combining MCTS with neural network value/policy estimates
    AbstractBetaZeroNetwork: Abstract base class for BetaZero policy and value networks
    BetaZeroNetwork: Dual-head neural network for policy and value prediction
    BetaZeroActionSampler: Network-guided action sampling for progressive widening
    BeliefRepresentation: Abstract belief-to-feature mapping
    ParticleMeanStdRepresentation: Default belief representation using particle statistics
    TrainingBuffer: Circular replay buffer for training examples
    TrainingExample: Single training datum (belief features, policy target, value target)
"""

from POMDPPlanners.planners.mcts_planners.beta_zero.belief_representation import (
    BeliefRepresentation,
    ParticleMeanStdRepresentation,
)
from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_network import (
    AbstractBetaZeroNetwork,
    BetaZeroNetwork,
)
from POMDPPlanners.planners.mcts_planners.beta_zero.training_buffer import (
    TrainingBuffer,
    TrainingExample,
)
from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_action_sampler import (
    BetaZeroActionSampler,
)
from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero import BetaZero

__all__ = [
    "BetaZero",
    "AbstractBetaZeroNetwork",
    "BetaZeroNetwork",
    "BetaZeroActionSampler",
    "BeliefRepresentation",
    "ParticleMeanStdRepresentation",
    "TrainingBuffer",
    "TrainingExample",
]
