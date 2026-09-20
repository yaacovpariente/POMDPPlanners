# SPDX-License-Identifier: MIT

"""Sanity POMDP family: the two-state debugging baseline and its batched model.

Classes:
    SanityPOMDP: Two-state environment used to check a planner runs at all.
    SanityInitialStateDist: Its initial state distribution.
    SanityInitialObservationDist: Its initial observation distribution.
    SanityVectorizedModel: Batched torch model of the same kernels.

The names below are re-exported because
``POMDPPlanners.environments.sanity_pomdp.SanityPOMDP`` is the module path
recorded in saved configurations, and :meth:`Environment.from_dict` imports that
path to rebuild the environment. ``SanityVectorizedModel`` is not re-exported:
it imports torch, which this package must not pull in on every
``import POMDPPlanners.environments``.
"""

from POMDPPlanners.environments.sanity_pomdp.sanity_pomdp import (
    SanityInitialObservationDist,
    SanityInitialStateDist,
    SanityPOMDP,
)

__all__ = [
    "SanityInitialObservationDist",
    "SanityInitialStateDist",
    "SanityPOMDP",
]
