# SPDX-License-Identifier: MIT

"""Planner-side generative models paired with the forward-only IsaacLab world.

While :mod:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_pomdp` is the ground-truth
*world* (forward-only, no densities), a planner carries a generative *model* as
``policy.environment``. This subpackage holds that model in its factored form: state is a flat
vector carved into named blocks, and observation is a ``{channel: value}`` mapping produced by a
swappable per-channel perception stack. The two spaces are independent, so a real sensor reading
has somewhere to go and a latent variable can stay latent without a packing trick.

The older one-space
:class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp.IsaacLabModelPOMDP`
(``observation = state + N(0, Sigma)``) remains available and is unchanged; the transition and
reward interfaces still live beside it and are reused here.

Classes:
    IsaacChannelSchema: Named contiguous blocks over a flat vector.
    IsaacModelPOMDP: Abstract generative-model interface over the factored Isaac schema.
    FactoredIsaacModelPOMDP: Transition + reward + per-channel perception over a named schema.
    UnicycleTransition: Planar unicycle integration of a body-frame velocity command.
    UnicycleIsaacModel: Factored model wired with a unicycle transition on a pose channel.
    BlockRewardModel: Restrict a reward model to named state blocks.
    LearnedIsaacModel: Factored model driven by a fitted linear-Gaussian transition and reward.
"""

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_model_pomdp import (
    IsaacChannelSchema,
    IsaacModelPOMDP,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_factored_model import (
    FactoredIsaacModelPOMDP,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_unicycle_model import (
    POSE_WIDTH,
    UnicycleIsaacModel,
    UnicycleTransition,
    wrap_angle,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_learned_model import (
    BlockRewardModel,
    LearnedIsaacModel,
)

__all__ = [
    "IsaacChannelSchema",
    "IsaacModelPOMDP",
    "FactoredIsaacModelPOMDP",
    "POSE_WIDTH",
    "UnicycleIsaacModel",
    "UnicycleTransition",
    "wrap_angle",
    "BlockRewardModel",
    "LearnedIsaacModel",
]
