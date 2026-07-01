# SPDX-License-Identifier: MIT

"""CARLA POMDP world environment module.

This module provides a forward-only adapter exposing the CARLA autonomous-driving
simulator as a ground-truth world for the POMDPPlanners episode loop.

Classes:
    CarlaPOMDP: Forward-only adapter exposing a CARLA session as a world Environment.
"""

from POMDPPlanners.environments.carla_pomdp.carla_pomdp import CarlaPOMDP

__all__ = [
    "CarlaPOMDP",
]
