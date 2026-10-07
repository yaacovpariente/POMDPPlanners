# SPDX-License-Identifier: MIT

"""LaserTag POMDP Environment Package.

This package implements the LaserTag pursuit-evasion POMDP environment
in both discrete-grid and continuous-space variants.

Exports:
    LaserTagPOMDP, ContinuousLaserTagPOMDP,
    ContinuousLaserTagPOMDPDiscreteActions: The environments.
    ContinuousLaserTagVectorizedModel: Batched torch model of the
        discrete-action continuous environment, for VOPP.
    OpponentPolicy: The opponent's transition behaviour.
    LaserTagVisualizer, ContinuousLaserTagVisualizer: Write episodes as traces
        for the 3D viewer.

Note:
    LaserTagState is now represented as numpy arrays with shape (5,).
    See laser_tag_pomdp.py for state vector structure documentation.
"""

from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import LaserTagPOMDP
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
    ContinuousLaserTagPOMDP,
    ContinuousLaserTagPOMDPDiscreteActions,
)
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_vectorized_model import (
    ContinuousLaserTagVectorizedModel,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_utils import (
    OpponentPolicy,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_visualization import (
    ContinuousLaserTagVisualizer,
    LaserTagVisualizer,
)

__all__ = [
    "LaserTagPOMDP",
    "ContinuousLaserTagPOMDP",
    "ContinuousLaserTagPOMDPDiscreteActions",
    "ContinuousLaserTagVectorizedModel",
    "OpponentPolicy",
    "LaserTagVisualizer",
    "ContinuousLaserTagVisualizer",
]
