# SPDX-License-Identifier: MIT

"""Battleship POMDP environment package.

Exports:
    BattleshipPOMDP: The environment.
    BattleshipBelief: The exact belief over legal fleet layouts.
    BattleshipVisualizer: Episode renderer.
    FleetLayoutTable: The enumerated legal layouts.
    build_battleship_trace: Writes an episode as data for the browser viewer.
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
from POMDPPlanners.environments.battleship_pomdp.visualizer import (
    BATTLESHIP_PAYLOAD_KIND,
    BattleshipVisualizer,
    build_battleship_trace,
)

__all__ = [
    "BATTLESHIP_PAYLOAD_KIND",
    "HIT",
    "MISS",
    "BattleshipBelief",
    "BattleshipInitialStateDistribution",
    "BattleshipPOMDP",
    "BattleshipPOMDPMetrics",
    "BattleshipState",
    "BattleshipStepChannel",
    "BattleshipVisualizer",
    "FleetLayoutTable",
    "build_battleship_trace",
    "create_battleship_state",
    "get_layout_table",
]
