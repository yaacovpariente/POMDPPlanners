# SPDX-License-Identifier: MIT

"""IsaacLab POMDP wrapper environment module.

This module provides a forward-only adapter exposing a registered IsaacLab task
as a ground-truth world for the POMDPPlanners episode loop, together with the
planner-side generative models that search inside it.

Two generative-model stacks live here:

* The **one-space** model
  (:mod:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp`),
  which keeps state and observation in the same space with
  ``observation = state + N(0, Sigma)``. Generic across every task, but it cannot
  express a real sensor reading or a state variable that must stay hidden.
* The **factored** stack
  (:mod:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models` plus
  :mod:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception`), where the
  state is a flat vector of named blocks and the observation is a
  ``{channel: value}`` mapping built by a swappable per-channel perception stack.
  Prefer it for anything reading a sensor.

Classes:
    IsaacLabPOMDP: Forward-only adapter exposing an IsaacLab task as a world.
    IsaacLabPOMDPVisualizer: RGB-frame-to-``.mp4`` video writer for episodes.
    IsaacLabModelPOMDP: One-space planner-side generative model.
    GaussianObservationModel: Additive-Normal observation model (obs = state + noise).
    TransitionModel: Interface for a state-transition model.
    GaussianRandomWalkTransition: Action-ignoring Gaussian random-walk transition.
    LinearGaussianTransition: Fit-from-data linear-Gaussian action-conditioned transition.
    RewardModel: Interface for a reward model.
    LinearRewardModel: Fit-from-data linear reward model POMCPOW optimizes.
    IsaacChannelSchema: Named contiguous blocks over a flat vector.
    IsaacModelPOMDP: Abstract generative-model interface over the factored schema.
    FactoredIsaacModelPOMDP: Transition + reward + per-channel perception.
    UnicycleIsaacModel: Factored model with analytic unicycle dynamics.
    LearnedIsaacModel: Factored model with a fitted linear-Gaussian transition.
    IsaacObservationModel: Abstract per-channel observation model.
"""

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp import (
    GaussianObservationModel,
    GaussianRandomWalkTransition,
    IsaacLabModelPOMDP,
    IsaacLabSimulatorTransition,
    LinearGaussianTransition,
    LinearRewardModel,
    RewardModel,
    TransitionModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_pomdp import IsaacLabPOMDP
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_visualizer import (
    IsaacLabPOMDPVisualizer,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
    FactoredIsaacModelPOMDP,
    IsaacChannelSchema,
    IsaacModelPOMDP,
    LearnedIsaacModel,
    UnicycleIsaacModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
    IsaacObservationModel,
)

__all__ = [
    "IsaacLabPOMDP",
    "IsaacLabPOMDPVisualizer",
    "IsaacLabModelPOMDP",
    "GaussianObservationModel",
    "TransitionModel",
    "GaussianRandomWalkTransition",
    "LinearGaussianTransition",
    "IsaacLabSimulatorTransition",
    "RewardModel",
    "LinearRewardModel",
    "IsaacChannelSchema",
    "IsaacModelPOMDP",
    "FactoredIsaacModelPOMDP",
    "UnicycleIsaacModel",
    "LearnedIsaacModel",
    "IsaacObservationModel",
]
