# SPDX-License-Identifier: MIT

# Import all classes to maintain backward compatibility
from POMDPPlanners.core.belief.base_belief import Belief
from POMDPPlanners.core.belief.particle_beliefs import (
    UnweightedParticleBelief,
    WeightedParticleBelief,
    WeightedParticleBeliefReinvigoration,
    WeightedParticleBeliefStateUpdate,
    UnweightedParticleBeliefStateUpdate,
    get_unique_support,
)
from POMDPPlanners.core.belief.gaussian_belief import GaussianBelief
from POMDPPlanners.core.belief.gaussian_belief_updaters import (
    GaussianBeliefUpdater,
    LinearKalmanFilterUpdater,
    ExtendedKalmanFilterUpdater,
    UnscentedKalmanFilterUpdater,
)
from POMDPPlanners.core.belief.gaussian_mixture_belief import (
    GaussianMixtureBelief,
    GaussianMixtureBeliefUpdater,
)
from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.core.belief.belief_utils import (
    sample_next_belief,
    get_initial_belief,
    is_terminal_particle_belief,
    is_terminal_belief,
)

__all__ = [
    "Belief",
    "UnweightedParticleBelief",
    "WeightedParticleBelief",
    "WeightedParticleBeliefReinvigoration",
    "WeightedParticleBeliefStateUpdate",
    "UnweightedParticleBeliefStateUpdate",
    "GaussianBelief",
    "GaussianBeliefUpdater",
    "LinearKalmanFilterUpdater",
    "ExtendedKalmanFilterUpdater",
    "UnscentedKalmanFilterUpdater",
    "GaussianMixtureBelief",
    "GaussianMixtureBeliefUpdater",
    "VectorizedParticleBeliefUpdater",
    "VectorizedWeightedParticleBelief",
    "get_unique_support",
    "sample_next_belief",
    "get_initial_belief",
    "is_terminal_particle_belief",
    "is_terminal_belief",
]
