# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""PacMan POMDP belief support with vectorized particle filter.

This package provides vectorized belief updater, grid utilities,
and a belief factory for the PacMan POMDP environment.

Classes:
    PacManVectorizedUpdater: Batched updater for PacMan POMDP.

Functions:
    create_pacman_belief: Factory producing a configured belief for PacManPOMDP.
"""

from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_beliefs.pacman_belief_factory import (
    create_pacman_belief,
)
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_beliefs.pacman_vectorized_updater import (
    PacManVectorizedUpdater,
)

__all__ = [
    "PacManVectorizedUpdater",
    "create_pacman_belief",
]
