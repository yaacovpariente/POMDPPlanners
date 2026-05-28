# SPDX-License-Identifier: MIT

"""LaserTag POMDP Environment Package.

This package implements the LaserTag pursuit-evasion POMDP environment
in both discrete-grid and continuous-space variants.

Note:
    LaserTagState is now represented as numpy arrays with shape (5,).
    See laser_tag_pomdp.py for state vector structure documentation.
"""

from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import LaserTagPOMDP
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
    ContinuousLaserTagPOMDP,
    ContinuousLaserTagPOMDPDiscreteActions,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_utils import (
    OpponentPolicy,
)

__all__ = [
    "LaserTagPOMDP",
    "ContinuousLaserTagPOMDP",
    "ContinuousLaserTagPOMDPDiscreteActions",
    "OpponentPolicy",
]
