# SPDX-License-Identifier: MIT

"""Belief factory for the occupancy-grid mapping POMDP.

Both belief types this factory returns are the environment's own conditional
whole-map filters. The generic bootstrap filter from ``get_initial_belief``
is deliberately not offered: it draws a fresh scan per particle and compares
it to the observed one, which has zero support for a continuous scan.

Functions:
    create_occupancy_grid_mapping_belief: Factory producing a configured belief.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_belief import (
    OccupancyGridMappingBelief,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_beliefs.occupancy_grid_mapping_vectorized_belief import (
    OccupancyGridMappingVectorizedBelief,
)
from POMDPPlanners.utils.belief_factory import BeliefType

if TYPE_CHECKING:
    from POMDPPlanners.core.belief.base_belief import Belief
    from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_pomdp import (
        OccupancyGridMappingPOMDP,
    )

_SUPPORTED_TYPES = {BeliefType.PARTICLE, BeliefType.VECTORIZED_PARTICLE}
_DEFAULT_TYPE = BeliefType.VECTORIZED_PARTICLE


def create_occupancy_grid_mapping_belief(
    env: "OccupancyGridMappingPOMDP",
    belief_type: BeliefType = _DEFAULT_TYPE,
    n_particles: int = 30,
) -> "Belief":
    """Create a ready-to-use whole-map filter for the occupancy-grid mapping POMDP.

    Args:
        env: The environment instance.
        belief_type: ``PARTICLE`` for the scalar filter, ``VECTORIZED_PARTICLE``
            for the batched one. Defaults to ``VECTORIZED_PARTICLE``.
        n_particles: Number of map hypotheses. Defaults to 30, the QA choice.

    Returns:
        A configured filter over fresh prior maps.

    Raises:
        ValueError: If ``belief_type`` is not one of the two particle types.
    """
    if belief_type not in _SUPPORTED_TYPES:
        raise ValueError(
            f"OccupancyGridMappingPOMDP does not support {belief_type}. "
            f"Supported: {_SUPPORTED_TYPES}"
        )
    if belief_type == BeliefType.PARTICLE:
        return OccupancyGridMappingBelief.initial(env, n_particles=n_particles)
    return OccupancyGridMappingVectorizedBelief.initial(env, n_particles=n_particles)
