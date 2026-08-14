# SPDX-License-Identifier: MIT

"""Standalone, swappable perception stack for the IsaacLab planner.

An IsaacLab task gives a genuine ``observation = h(state)`` split — privileged state from the
physics engine, observation from a simulated sensor — which the one-space
:class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp.IsaacLabModelPOMDP`
cannot express: it treats the observation as the state plus Gaussian noise, so a LiDAR reading has
nowhere to go. This subpackage supplies the missing half, and is decoupled from both the world and
the belief so a user can swap a sensor model without touching either:

* :mod:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.isaac_env_cfg_injectors` —
  builders that reshape a parsed task config before the env is built: attach the sensors the task
  switched off, widen a contact history to the control step, and add the reset terms it lacks.
* :mod:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.isaac_sensors` — world-side
  single-frame readers over a live scene (root pose, joint state, ray-cast ranges, height scan,
  the task's own policy observation), shaped to drop into ``IsaacLabPOMDP``'s ``state_extractor``
  and ``observation_extractor`` hooks.
* :mod:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_model` — the
  planner-side per-channel :class:`IsaacObservationModel` interface: named clean state blocks in,
  one perceived observation channel out. A generative model composes these into a
  ``{channel: model}`` map, which is what frees state and observation from sharing a width.
* :mod:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models` — the
  catalog of concrete per-channel models, registered for selection by name.

The public names below are re-exported so callers can import them straight from the subpackage.
"""

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.isaac_env_cfg_injectors import (
    compose_env_cfg_modifiers,
    disable_terminations_injector,
    make_contact_history_injector,
    make_height_scanner_injector,
    make_joint_reset_injector,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.isaac_sensors import (
    command_pose_base,
    command_pose_world,
    concat_extractors,
    constant_extractor,
    contact_body_indices,
    contact_impulse,
    height_scan,
    joint_state,
    make_peak_contact_force_extractor,
    peak_contact_force,
    policy_observation,
    ray_caster_ranges,
    root_planar_pose,
    root_state,
    yaw_from_quaternion,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_model import (
    IsaacObservationModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models import (
    UNINFORMATIVE_ACCURACY,
    ExactChannelObservationModel,
    GaussianChannelObservationModel,
    GoalRelativePoseObservationModel,
    HeightScanObservationModel,
    LatentTypeSignalObservationModel,
    RayCasterObservationModel,
    available_observation_models,
    build_observation_model,
    grid_scan_pattern,
    register_observation_model,
    wrap_to_pi,
)

__all__ = [
    "IsaacObservationModel",
    "ExactChannelObservationModel",
    "GaussianChannelObservationModel",
    "GoalRelativePoseObservationModel",
    "HeightScanObservationModel",
    "LatentTypeSignalObservationModel",
    "RayCasterObservationModel",
    "UNINFORMATIVE_ACCURACY",
    "available_observation_models",
    "build_observation_model",
    "register_observation_model",
    "wrap_to_pi",
    "grid_scan_pattern",
    "command_pose_base",
    "command_pose_world",
    "compose_env_cfg_modifiers",
    "concat_extractors",
    "constant_extractor",
    "contact_body_indices",
    "contact_impulse",
    "disable_terminations_injector",
    "height_scan",
    "make_contact_history_injector",
    "make_height_scanner_injector",
    "make_joint_reset_injector",
    "make_peak_contact_force_extractor",
    "peak_contact_force",
    "joint_state",
    "policy_observation",
    "ray_caster_ranges",
    "root_planar_pose",
    "root_state",
    "yaw_from_quaternion",
]
