# SPDX-License-Identifier: MIT

"""Shared utilities for the LaserTag POMDP environments.

Classes:
    OpponentPolicy: Selectable opponent transition behaviour (evade, pursue, or
        evade-when-spotted).
"""

from enum import Enum


class OpponentPolicy(Enum):
    """Opponent transition behaviour selectable on the LaserTag environments.

    Three policies are offered:

    - ``EVADE`` (default): the opponent flees the robot, placing its directional
      probability mass on the cell that *increases* distance, and reacts to the
      robot's current (pre-move) position. This matches JuliaPOMDP/LaserTag.jl.
    - ``PURSUE``: the opponent chases the robot, placing its directional mass on
      the cell that *decreases* distance, and reacts to the robot's post-move
      position. This restores the behaviour used before the evader alignment fix.
    - ``EVADE_WHEN_SPOTTED``: a partially-observed reactive opponent. When the
      robot has a clear line of sight to the opponent (the opponent lies on one
      of the robot's unoccluded laser rays, evaluated from the robot's pre-move
      position), it behaves exactly like ``EVADE``. Otherwise the unspotted
      behaviour is environment-specific: the discrete grid env moves randomly
      (uniformly over the moves, with the usual stay/wall handling), while the
      continuous env holds its position (only the Gaussian opponent noise jitters
      it). The opponent is memoryless — visibility is recomputed each step from
      the current state.

    ``EVADE`` and ``PURSUE`` couple both the directional choice and the
    reference-position choice, so they are mutually exclusive opposites.
    """

    EVADE = "evade"
    PURSUE = "pursue"
    EVADE_WHEN_SPOTTED = "evade_when_spotted"

    @property
    def native_code(self) -> int:
        """Integer code passed to the C++ kernels.

        ``EVADE`` = 0, ``PURSUE`` = 1, ``EVADE_WHEN_SPOTTED`` = 2.
        """
        return {
            OpponentPolicy.EVADE: 0,
            OpponentPolicy.PURSUE: 1,
            OpponentPolicy.EVADE_WHEN_SPOTTED: 2,
        }[self]
