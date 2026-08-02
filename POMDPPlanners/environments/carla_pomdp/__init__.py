# SPDX-License-Identifier: MIT

"""CARLA POMDP environment module.

This module provides a forward-only adapter exposing the CARLA autonomous-driving
simulator as a ground-truth world for the POMDPPlanners episode loop, plus the
planner-side generative-model interface paired with it and a concrete reference model.

Classes:
    CarlaPOMDP: Forward-only adapter exposing a CARLA session as a world Environment.
    CarlaModelPOMDP: Abstract generative-model interface over the CARLA schema.
    FactoredCarlaModelPOMDP: Concrete CARLA model with a factored observation model.
    DreamerCarlaModelPOMDP: Concrete CARLA model backed by a Dreamer world model.
    PerceivedAgentsBelief: Particle belief that stamps the perception pipeline's agent block.
    CarlaPerceptionPipeline: Standalone, swappable perception + prediction stage.
    CarlaServerPool: Context manager owning N headless CARLA servers for parallel episodes.
    CarlaServerLease: Connection endpoints of one leased pool server.
"""

from POMDPPlanners.environments.carla_pomdp.carla_pomdp import CarlaPOMDP
from POMDPPlanners.environments.carla_pomdp.carla_server_pool import (
    CarlaServerLease,
    CarlaServerPool,
    acquire_pool_lease,
)
from POMDPPlanners.environments.carla_pomdp.carla_belief import PerceivedAgentsBelief
from POMDPPlanners.environments.carla_pomdp.carla_perception import (
    CarlaPerceptionPipeline,
    LidarCameraPerceptionModel,
    MotionTracker,
    OracleAgentPerceptionModel,
    PerceptionModel,
)
from POMDPPlanners.environments.carla_pomdp.carla_generative_models import (
    CarlaModelPOMDP,
    DreamerCarlaModelPOMDP,
    FactoredCarlaModelPOMDP,
)

__all__ = [
    "CarlaPOMDP",
    "CarlaModelPOMDP",
    "FactoredCarlaModelPOMDP",
    "DreamerCarlaModelPOMDP",
    "PerceivedAgentsBelief",
    "CarlaPerceptionPipeline",
    "PerceptionModel",
    "MotionTracker",
    "LidarCameraPerceptionModel",
    "OracleAgentPerceptionModel",
    "CarlaServerPool",
    "CarlaServerLease",
    "acquire_pool_lease",
]
