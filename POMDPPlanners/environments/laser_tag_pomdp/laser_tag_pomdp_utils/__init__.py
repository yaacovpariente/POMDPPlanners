# SPDX-License-Identifier: MIT

"""Shared utilities for the LaserTag POMDP environments.

Classes:
    OpponentPolicy: Selectable opponent transition behaviour (evade vs pursue).
"""

from enum import Enum


class OpponentPolicy(Enum):
    """Opponent transition behaviour selectable on the LaserTag environments.

    Two policies are offered:

    - ``EVADE`` (default): the opponent flees the robot, placing its directional
      probability mass on the cell that *increases* distance, and reacts to the
      robot's current (pre-move) position. This matches JuliaPOMDP/LaserTag.jl.
    - ``PURSUE``: the opponent chases the robot, placing its directional mass on
      the cell that *decreases* distance, and reacts to the robot's post-move
      position. This restores the behaviour used before the evader alignment fix.

    Both directional choices and reference-position choices are coupled per
    policy, so ``PURSUE`` and ``EVADE`` are mutually exclusive opposites.
    """

    EVADE = "evade"
    PURSUE = "pursue"

    @property
    def native_code(self) -> int:
        """Integer code passed to the C++ kernels (``EVADE`` = 0, ``PURSUE`` = 1)."""
        return 0 if self is OpponentPolicy.EVADE else 1
