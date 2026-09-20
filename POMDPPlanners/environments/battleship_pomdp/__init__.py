# SPDX-License-Identifier: MIT

"""Battleship POMDP environment package.

Exports:
    BattleshipPOMDP: The environment.
    BattleshipBelief: The exact belief over legal fleet layouts.
    BattleshipVectorizedWeightedParticleBelief: Its batched twin, and the
        environment's default belief.
    BattleshipVisualizer: Episode renderer.
    FleetLayoutTable: The enumerated legal layouts.
"""

from POMDPPlanners.environments.battleship_pomdp.battleship_layouts import (
    BattleshipInitialStateDistribution,
    FleetLayoutTable,
    get_layout_table,
)
from POMDPPlanners.environments.battleship_pomdp.battleship_pomdp import (
    HIT,
    MISS,
    BattleshipPOMDP,
    BattleshipPOMDPMetrics,
    BattleshipState,
    BattleshipStepChannel,
    create_battleship_state,
)
from POMDPPlanners.environments.battleship_pomdp.battleship_belief import BattleshipBelief
from POMDPPlanners.environments.battleship_pomdp.battleship_vectorized_belief import (
    BattleshipVectorizedUpdater,
    BattleshipVectorizedWeightedParticleBelief,
    create_battleship_belief,
)
from POMDPPlanners.environments.battleship_pomdp.visualizer import (
    BattleshipVisualizer,
)

__all__ = [
    "HIT",
    "MISS",
    "BattleshipBelief",
    "BattleshipVectorizedUpdater",
    "BattleshipVectorizedWeightedParticleBelief",
    "BattleshipInitialStateDistribution",
    "BattleshipPOMDP",
    "BattleshipPOMDPMetrics",
    "BattleshipState",
    "BattleshipStepChannel",
    "BattleshipVisualizer",
    "FleetLayoutTable",
    "create_battleship_belief",
    "create_battleship_state",
    "get_layout_table",
]
