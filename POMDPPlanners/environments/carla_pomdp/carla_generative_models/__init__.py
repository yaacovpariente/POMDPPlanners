# SPDX-License-Identifier: MIT

"""Planner-side generative models paired with the forward-only CARLA world.

While :mod:`~POMDPPlanners.environments.carla_pomdp.carla_pomdp` is the ground-truth
*world* (forward-only, no densities), a planner carries a generative *model* as
``policy.environment``. This subpackage holds that model: the abstract interface and its
concrete implementations, all sharing the CARLA state/observation schema defined by the
world.

Classes:
    CarlaModelPOMDP: Abstract generative-model interface over the CARLA schema.
    FactoredCarlaModelPOMDP: Concrete CARLA model with a factored observation model.
    KinematicCarlaModelPOMDP: Factored model with a kinematic bicycle transition.
    DreamerCarlaModelPOMDP: Concrete CARLA model backed by a Dreamer world model.
    DreamerWorldModel: Protocol a trained Dreamer RSSM must satisfy.
    CarDreamerWorldModel: DreamerV3-backed ``DreamerWorldModel`` from a CarDreamer checkpoint.
"""

from POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_model_pomdp import (
    CarlaModelPOMDP,
)
from POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_factored_model_pomdp import (
    FactoredCarlaModelPOMDP,
)
from POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_kinematic_model_pomdp import (
    KinematicCarlaModelPOMDP,
)
from POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_dreamer_model_pomdp import (
    DreamerCarlaModelPOMDP,
    DreamerWorldModel,
)
from POMDPPlanners.environments.carla_pomdp.carla_generative_models.cardreamer_world_model import (
    CarDreamerWorldModel,
    build_cardreamer_model,
)

__all__ = [
    "CarlaModelPOMDP",
    "FactoredCarlaModelPOMDP",
    "KinematicCarlaModelPOMDP",
    "DreamerCarlaModelPOMDP",
    "DreamerWorldModel",
    "CarDreamerWorldModel",
    "build_cardreamer_model",
]
