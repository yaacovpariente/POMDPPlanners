# SPDX-License-Identifier: MIT

"""The planner-side models :mod:`isaac_vopp_metrics_example` can hand VOPP.

Each builder takes the *same* warm-up rollout and the same action presets and returns a vectorized
generative model, so switching between them with ``--model-kind`` compares models rather than
budgets. They live beside the runner rather than inside the package because the choice of which
model suits which task is a property of this study, not of the library.

Functions:
    build_vectorized_model: Ridge-fit a linear-Gaussian surrogate over the whole observation.
    build_manipulator_model: The analytic joint-lag arm model, calibrated on the rollout.
    build_navigation_model: The goal-relative base model, calibrated on the rollout.
"""

from typing import Any, Tuple

import numpy as np

#: POMDP discount factor, shared by the world and every planner-side model.
GAMMA = 0.99

#: Std of the additive observation noise every planner-side model assumes.
OBSERVATION_NOISE_STD = 0.1

#: Grid spacing used to quantize observations into planner tree keys.
OBSERVATION_RESOLUTION = 5.0


def build_vectorized_model(
    samples: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    action_presets: np.ndarray,
    device: Any,
) -> Any:
    """Fit the linear dynamics/reward and wrap them in the vectorized model."""
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.isaac_lab_pomdp import (
        GaussianObservationModel,
        LinearGaussianTransition,
        LinearRewardModel,
    )

    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_vectorized_model import (
        IsaacLabVectorizedModel,
    )

    states, actions, next_states, rewards = samples
    transition = LinearGaussianTransition.fit(states, actions, next_states)
    reward_model = LinearRewardModel.fit(states, actions, next_states, rewards)
    observation_model = GaussianObservationModel(
        observation_dim=states.shape[1], noise_std=OBSERVATION_NOISE_STD
    )
    return IsaacLabVectorizedModel(
        transition=transition,
        observation_model=observation_model,
        reward_model=reward_model,
        action_presets=action_presets,
        device=device,
        observation_resolution=OBSERVATION_RESOLUTION,
    )


def build_manipulator_model(
    world: Any,
    samples: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    action_presets: np.ndarray,
    device: Any,
) -> Any:
    """Build the analytic manipulator model, calibrated on the same warm-up the linear fit uses.

    The two models therefore see identical data and differ only in structure, which is what makes
    the before/after comparison a comparison of models rather than of budgets.

    Args:
        world: The live IsaacLab world, read for its fixed task configuration.
        samples: The warm-up rollout as ``(states, actions, next_states, rewards)``.
        action_presets: The ``[num_actions, action_dim]`` table VOPP indexes into.
        device: Target torch device.

    Returns:
        A vectorized generative model VOPP can search.
    """
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_franka_reach_setup import (  # noqa: E501
        build_franka_reach_model,
    )

    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_manipulator_vectorized_model import (  # noqa: E501
        ManipulatorVectorizedModel,
    )

    states, actions, next_states, _ = samples
    scalar = build_franka_reach_model(
        world.task_env, (states, actions, next_states), list(action_presets), GAMMA
    )
    lag = scalar.joint_transition
    print(
        f"calibrated joint lag: gain={lag.tracking_gain:.4f}, step_dt={lag.step_dt:.5f}s, "
        f"action_scale={lag.action_scale}, "
        f"position_noise={lag.process_noise_std[0]:.4f} rad, "
        f"commanded joints={lag.actuated_indices.tolist()} of {lag.position_width}"
    )
    return ManipulatorVectorizedModel(
        scalar,
        device=device,
        observation_noise_std=OBSERVATION_NOISE_STD,
        observation_resolution=OBSERVATION_RESOLUTION,
    )


def build_navigation_model(
    world: Any,
    samples: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    action_presets: np.ndarray,
    device: Any,
) -> Any:
    """Build the goal-relative navigation model, calibrated on the same warm-up the linear fit uses.

    The two models therefore see identical data and differ only in structure, which is what makes
    the before/after comparison a comparison of models rather than of budgets.

    Args:
        world: The live IsaacLab world, read for its control-step duration.
        samples: The warm-up rollout as ``(states, actions, next_states, rewards)``.
        action_presets: The ``[num_actions, 3]`` table VOPP indexes into.
        device: Target torch device.

    Returns:
        A vectorized generative model VOPP can search.
    """
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_anymal_navigation_setup import (  # noqa: E501
        build_anymal_navigation_model,
    )

    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_navigation_vectorized_model import (  # noqa: E501
        NavigationVectorizedModel,
    )

    states, actions, next_states, _ = samples
    scalar = build_anymal_navigation_model(
        world.task_env, (states, actions, next_states), list(action_presets), GAMMA
    )
    tracking = scalar.goal_transition
    print(
        f"calibrated command tracking: linear={tracking.linear_scale:.4f}, "
        f"angular={tracking.angular_scale:.4f}, step_dt={tracking.step_dt:.5f}s, "
        f"velocity_noise={tracking.process_noise_std[0]:.4f} m/s, "
        f"position_noise={tracking.process_noise_std[3]:.4f} m, "
        f"heading_noise={tracking.process_noise_std[-1]:.4f} rad"
    )
    return NavigationVectorizedModel(
        scalar,
        device=device,
        observation_noise_std=OBSERVATION_NOISE_STD,
        observation_resolution=OBSERVATION_RESOLUTION,
    )
