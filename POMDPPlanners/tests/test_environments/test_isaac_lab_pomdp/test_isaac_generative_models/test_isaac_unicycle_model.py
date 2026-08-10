# SPDX-License-Identifier: MIT

"""Unit tests for the analytic unicycle transition and the model built on it.

The integration is checked against hand-computed displacements, because a unicycle with a wrong
frame convention still produces smooth, plausible trajectories — it just drives sideways, which
reads downstream as "the planner did not reach the goal".
"""

from typing import Any

import numpy as np
import pytest

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
    IsaacChannelSchema,
    UnicycleIsaacModel,
    UnicycleTransition,
    wrap_angle,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
    GaussianChannelObservationModel,
)

SCHEMA = IsaacChannelSchema((("base_pose", 3), ("goal", 2)))
FORWARD = np.array([1.0, 0.0, 0.0])
STRAFE = np.array([0.0, 1.0, 0.0])
TURN = np.array([0.0, 0.0, 1.0])


def _exact_transition(**overrides: Any) -> UnicycleTransition:
    settings: dict = {"step_dt": 0.5, "process_noise_std": 1e-9}
    settings.update(overrides)
    return UnicycleTransition(**settings)


def test_forward_command_moves_along_the_heading() -> None:
    """A body-frame command has to be rotated into the world, not applied to world axes.

    Purpose: Validates the linear integration against a hand-computed displacement

    Given: A robot at the origin yawed a quarter-turn left, commanded 1 m/s forward for 0.5 s
    When: One step is taken
    Then: It has moved +0.5 m in y, not in x

    Test type: unit
    """
    transition = _exact_transition()
    nxt = transition.sample_next_state([0.0, 0.0, np.pi / 2.0], FORWARD)
    assert nxt[:2] == pytest.approx([0.0, 0.5], abs=1e-5)


def test_lateral_command_moves_across_the_heading() -> None:
    """A legged base can strafe, so the y command must be a separate body axis.

    Purpose: Validates the lateral term of the integration

    Given: A robot at the origin facing +x, commanded 1 m/s sideways for 0.5 s
    When: One step is taken
    Then: It has moved +0.5 m in y

    Test type: unit
    """
    nxt = _exact_transition().sample_next_state([0.0, 0.0, 0.0], STRAFE)
    assert nxt[:2] == pytest.approx([0.0, 0.5], abs=1e-5)


def test_yaw_rate_turns_the_base_and_stays_wrapped() -> None:
    """An unwrapped heading grows without bound and breaks every downstream cosine.

    Purpose: Validates the yaw integration and its wrapping

    Given: A robot near +pi commanded to keep turning
    When: Several steps are taken
    Then: The heading advances and stays inside (-pi, pi]

    Test type: unit
    """
    transition = _exact_transition(step_dt=1.0)
    pose = np.array([0.0, 0.0, 3.0])
    for _ in range(5):
        pose = transition.sample_next_state(pose, TURN)
        assert -np.pi < pose[2] <= np.pi


def test_command_scale_models_imperfect_tracking() -> None:
    """The low-level policy does not achieve the command exactly, and the model should say so.

    Purpose: Validates that command_scale scales the achieved displacement

    Given: Two transitions differing only in command_scale
    When: The same forward command is applied
    Then: The scaled one travels the scaled distance

    Test type: unit
    """
    full = _exact_transition().sample_next_state([0.0, 0.0, 0.0], FORWARD)
    scaled = _exact_transition(command_scale=0.6).sample_next_state([0.0, 0.0, 0.0], FORWARD)
    assert scaled[0] == pytest.approx(0.6 * full[0], abs=1e-5)


def test_a_batch_of_successors_is_dispersed_around_the_same_mean() -> None:
    """Progressive widening needs distinct successors, not one repeated draw.

    Purpose: Validates batched sampling of the unicycle transition

    Given: A transition with appreciable process noise
    When: Sixteen successors are drawn in one call
    Then: They differ from each other and average to the noise-free prediction

    Test type: unit
    """
    np.random.seed(0)
    transition = UnicycleTransition(step_dt=0.5, process_noise_std=0.05)
    batch = transition.sample_next_state([0.0, 0.0, 0.0], FORWARD, n_samples=2000)
    assert batch.shape == (2000, 3)
    assert batch[:, 0].mean() == pytest.approx(0.5, abs=0.01)
    assert not np.allclose(batch[0], batch[1])


def test_density_peaks_at_the_predicted_pose() -> None:
    """A transition density that did not peak on its own prediction would misweight particles.

    Purpose: Validates the unicycle transition's log-density

    Given: A noise-carrying transition and the pose it predicts
    When: The prediction and a displaced pose are scored
    Then: The prediction scores higher

    Test type: unit
    """
    transition = UnicycleTransition(step_dt=0.5, process_noise_std=0.05)
    predicted = _exact_transition().sample_next_state([0.0, 0.0, 0.0], FORWARD)
    displaced = predicted + np.array([0.3, 0.0, 0.0])
    scores = transition.log_probability([0.0, 0.0, 0.0], FORWARD, np.stack([predicted, displaced]))
    assert scores[0] > scores[1]


def test_density_treats_a_heading_and_its_wrapped_twin_alike() -> None:
    """Yaw is an angle; a residual of 2*pi is a residual of zero.

    Purpose: Validates that the yaw residual is wrapped before it is scored

    Given: A predicted pose and the same pose with its yaw offset by exactly 2*pi
    When: Both are scored
    Then: The scores agree

    Test type: unit
    """
    transition = UnicycleTransition(step_dt=0.5, process_noise_std=0.05)
    predicted = _exact_transition().sample_next_state([0.0, 0.0, 0.0], TURN)
    twin = predicted + np.array([0.0, 0.0, 2.0 * np.pi])
    scores = transition.log_probability([0.0, 0.0, 0.0], TURN, np.stack([predicted, twin]))
    assert scores[0] == pytest.approx(scores[1])


@pytest.mark.parametrize(
    "overrides, message",
    [({"step_dt": 0.0}, "step_dt"), ({"process_noise_std": 0.0}, "strictly positive")],
)
def test_invalid_transition_configuration_is_rejected(overrides: dict, message: str) -> None:
    """A zero timestep or zero noise makes the density degenerate rather than tight.

    Purpose: Validates the construction-time guards on the unicycle transition

    Given: An invalid step duration or noise std
    When: The transition is constructed
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match=message):
        _exact_transition(**overrides)


def test_wrap_angle_maps_into_the_half_open_interval() -> None:
    """Every angle consumer downstream assumes this interval.

    Purpose: Validates the angle-wrapping helper at its boundaries

    Given: Angles above, below and exactly at the wrap points
    When: They are wrapped
    Then: All land in (-pi, pi]

    Test type: unit
    """
    wrapped = wrap_angle([3.0 * np.pi, -3.0 * np.pi, np.pi, 0.0])
    assert np.all(wrapped > -np.pi - 1e-12)
    assert np.all(wrapped <= np.pi + 1e-12)
    assert wrapped[3] == pytest.approx(0.0)


def test_the_model_drives_only_the_pose_channel() -> None:
    """A goal is task data, not dynamics; the unicycle must not move it.

    Purpose: Validates that UnicycleIsaacModel wires the pose channel as the driven block

    Given: A schema with a pose channel and a goal channel
    When: A successor is sampled
    Then: The pose has advanced and the goal is untouched

    Test type: unit
    """
    np.random.seed(0)
    model = UnicycleIsaacModel(
        state_schema=SCHEMA,
        action_presets=[FORWARD, TURN],
        discount_factor=0.99,
        step_dt=0.5,
        process_noise_std=1e-9,
        observation_models={
            "base_pose": GaussianChannelObservationModel(channel="base_pose", noise_std=0.05)
        },
    )
    state = SCHEMA.pack({"base_pose": [0.0, 0.0, 0.0], "goal": [3.0, 0.0]})
    successor = model.sample_next_state(state, FORWARD)
    assert SCHEMA.block(successor, "base_pose")[0] == pytest.approx(0.5, abs=1e-4)
    assert SCHEMA.block(successor, "goal") == pytest.approx([3.0, 0.0])


def test_a_pose_channel_of_the_wrong_width_is_rejected() -> None:
    """A unicycle integrates exactly (x, y, yaw); a wider block means a different model.

    Purpose: Validates the construction-time width check on the pose channel

    Given: A schema whose pose channel is 4 wide
    When: The unicycle model is constructed
    Then: ValueError is raised naming the channel and both widths

    Test type: unit
    """
    schema = IsaacChannelSchema((("base_pose", 4),))
    with pytest.raises(ValueError, match="base_pose"):
        UnicycleIsaacModel(
            state_schema=schema,
            action_presets=[FORWARD],
            discount_factor=0.99,
            step_dt=0.5,
        )
