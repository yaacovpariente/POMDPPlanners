# SPDX-License-Identifier: MIT

"""ConstrainedZero: Neural MCTS for Chance-Constrained POMDPs.

This package implements the ConstrainedZero algorithm (Moss et al., IJCAI 2024),
which extends BetaZero to solve CC-POMDPs by adding a failure probability head,
safety-constrained PUCT, and adaptive failure threshold calibration.

References:
    Moss, R. J., Jamgochian, A., Fischer, J., Corso, A., & Kochenderfer, M. J. (2024).
    ConstrainedZero: Chance-Constrained POMDP Planning Using Learned Probabilistic Failure
    Surrogates and Adaptive Safety Constraints. Proceedings of the Thirty-Third International
    Joint Conference on Artificial Intelligence (IJCAI), 6752-6760. arXiv:2405.00644.
    https://arxiv.org/abs/2405.00644

Classes:
    ConstrainedZero: Main planner extending BetaZero for CC-POMDPs
    ConstrainedZeroNetwork: Three-head network with policy, value, and failure heads
    ConstrainedTrainingBuffer: Replay buffer with failure targets
    ConstrainedTrainingExample: Training datum with failure target
"""

from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_training_buffer import (
    ConstrainedTrainingBuffer,
    ConstrainedTrainingExample,
)
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_zero_network import (
    ConstrainedZeroNetwork,
)
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_zero import (
    ConstrainedZero,
)

__all__ = [
    "ConstrainedZero",
    "ConstrainedZeroNetwork",
    "ConstrainedTrainingBuffer",
    "ConstrainedTrainingExample",
]
