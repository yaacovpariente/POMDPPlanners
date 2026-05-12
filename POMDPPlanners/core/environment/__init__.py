"""Public surface of the ``core.environment`` package.

Re-exports the same names that used to live in ``core/environment.py`` so
that ``from POMDPPlanners.core.environment import Environment, SpaceInfo, ...``
continues to work after the package refactor. ``ConstrainedEnvironment`` is
added here as the new sibling module's public class.
"""

from POMDPPlanners.core.environment.environment import (
    DiscreteActionsEnvironment,
    Environment,
    EnvironmentGenerator,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.environment.constrained_environment import ConstrainedEnvironment


__all__ = [
    "ConstrainedEnvironment",
    "DiscreteActionsEnvironment",
    "Environment",
    "EnvironmentGenerator",
    "SpaceInfo",
    "SpaceType",
]
