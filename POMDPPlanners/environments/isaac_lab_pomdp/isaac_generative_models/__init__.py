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
    ModifiedDHChain: Offline forward kinematics of a serial chain in modified DH parameters.
    JointLagTransition: First-order lag of joint positions toward a commanded joint target.
    ReachRewardModel: The reach task's own objective, computed analytically through the chain.
    ManipulatorIsaacModel: Factored model wired with a joint lag and an analytic reach reward.
    FrankaReachLayout: The reach task's fixed joint, timing and action-scale configuration.
    GoalRelativeTransition: Integrates a base-frame goal forward under a velocity command.
    NavigationRewardModel: The navigation task's own pose-tracking objective.
    NavigationIsaacModel: Factored model wired with goal-relative dynamics and that objective.
    AnymalNavigationLayout: The navigation task's fixed timing configuration.
    BlockRewardModel: Restrict a reward model to named state blocks.
    LearnedIsaacModel: Factored model driven by a fitted linear-Gaussian transition and reward.

Functions:
    calibrate_tracking_gain: Least-squares estimate of the controller lag gain from a rollout.
    franka_panda_chain: The Franka Emika Panda chain from ``panda_link0`` to ``panda_hand``.
    franka_reach_layout: Read the reach task's fixed configuration off a live task.
    calibrate_lag_noise: Measure the process noise the calibrated lag leaves unexplained.
    build_franka_reach_model: Assemble the calibrated analytic model for the reach task.
    navigation_state_schema: The schema the navigation task's ``policy`` group implies.
    anymal_navigation_layout: Read the navigation task's control step off a live task.
    calibrate_command_tracking: Measure the achieved fraction of the velocity command.
    calibrate_navigation_noise: Measure the process noise the calibrated tracking leaves.
    build_anymal_navigation_model: Assemble the calibrated model for the navigation task.
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
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_manipulator_model import (
    JointLagTransition,
    ManipulatorIsaacModel,
    ModifiedDHChain,
    ReachRewardModel,
    calibrate_tracking_gain,
    franka_panda_chain,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_franka_reach_setup import (
    FrankaReachLayout,
    build_franka_reach_model,
    calibrate_lag_noise,
    franka_reach_layout,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_navigation_model import (
    BASE_VELOCITY_WIDTH,
    POSE_COMMAND_WIDTH,
    VELOCITY_COMMAND_WIDTH,
    GoalRelativeTransition,
    NavigationIsaacModel,
    NavigationRewardModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_anymal_navigation_setup import (
    AnymalNavigationLayout,
    anymal_navigation_layout,
    build_anymal_navigation_model,
    calibrate_command_tracking,
    calibrate_navigation_noise,
    navigation_state_schema,
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
    "JointLagTransition",
    "ManipulatorIsaacModel",
    "ModifiedDHChain",
    "ReachRewardModel",
    "calibrate_tracking_gain",
    "franka_panda_chain",
    "FrankaReachLayout",
    "build_franka_reach_model",
    "calibrate_lag_noise",
    "franka_reach_layout",
    "BASE_VELOCITY_WIDTH",
    "POSE_COMMAND_WIDTH",
    "VELOCITY_COMMAND_WIDTH",
    "GoalRelativeTransition",
    "NavigationIsaacModel",
    "NavigationRewardModel",
    "AnymalNavigationLayout",
    "anymal_navigation_layout",
    "build_anymal_navigation_model",
    "calibrate_command_tracking",
    "calibrate_navigation_noise",
    "navigation_state_schema",
    "BlockRewardModel",
    "LearnedIsaacModel",
]
