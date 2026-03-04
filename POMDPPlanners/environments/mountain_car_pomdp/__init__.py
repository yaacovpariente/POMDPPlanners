"""Mountain Car POMDP Environment Module.

This module provides the Mountain Car POMDP environment implementation and
related components for hill-climbing tasks with noisy observations.

Classes:
    MountainCarPOMDP: Main Mountain Car environment with POMDP formulation
    MountainCarTransition: Physics-based state transition model
    MountainCarObservation: Gaussian noise observation model
    MountainCarPOMDPMetrics: Metric names for Mountain Car POMDP environment
"""

from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp import (
    MountainCarObservation,
    MountainCarPOMDP,
    MountainCarPOMDPMetrics,
    MountainCarTransition,
)

__all__ = [
    "MountainCarPOMDP",
    "MountainCarObservation",
    "MountainCarTransition",
    "MountainCarPOMDPMetrics",
]
