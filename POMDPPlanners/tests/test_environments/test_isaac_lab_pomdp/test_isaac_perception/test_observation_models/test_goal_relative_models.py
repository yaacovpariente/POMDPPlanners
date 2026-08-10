# SPDX-License-Identifier: MIT

"""Unit tests for the goal-relative observation model.

The model is the localisation channel: the goal's world pose lives in the state and the robot sees
only its base-frame offset, so a bug here shows up downstream as "the belief cannot localise"
rather than as anything pointing at this file. The tests therefore check the frame convention
against hand-computed geometry, and the wrap handling against the discontinuity that breaks it.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
    GoalRelativePoseObservationModel,
    wrap_to_pi,
)


def _state(pose, goal):
    return {"base_pose": np.asarray(pose, dtype=float), "goal": np.asarray(goal, dtype=float)}


@pytest.mark.parametrize(
    "yaw, expected",
    [
        (0.0, (2.0, 0.0)),
        (0.5 * np.pi, (0.0, -2.0)),
        (np.pi, (-2.0, 0.0)),
        (-0.5 * np.pi, (0.0, 2.0)),
    ],
)
def test_clean_offset_rotates_the_goal_into_the_base_frame(yaw, expected) -> None:
    """A goal 2 m east is ahead, behind or to the side depending only on the heading.

    Purpose: Validates the world-to-base rotation of the goal offset

    Given: The robot at the origin with a given yaw and the goal 2 m along world +x
    When: The clean base-frame offset is computed
    Then: It matches the hand-computed rotation of (2, 0) by -yaw

    Test type: unit
    """
    model = GoalRelativePoseObservationModel()
    offset = model.clean_offset(_state((0.0, 0.0, yaw), (2.0, 0.0, 0.0)))
    assert offset[:2] == pytest.approx(np.asarray(expected), abs=1e-9)


def test_clean_offset_reports_the_heading_error_wrapped() -> None:
    """An unwrapped heading error would put a 2*pi gap between neighbouring headings.

    Purpose: Validates that the heading entry is wrapped into (-pi, pi]

    Given: A robot heading just below -pi and a goal heading just above +pi
    When: The clean offset is computed
    Then: The heading entry is the small difference, not the 2*pi one

    Test type: unit
    """
    model = GoalRelativePoseObservationModel()
    offset = model.clean_offset(_state((0.0, 0.0, -np.pi + 0.05), (0.0, 0.0, np.pi - 0.05)))
    assert abs(float(offset[2])) == pytest.approx(0.1, abs=1e-9)


def test_log_probability_scores_a_wrapped_residual_as_a_near_match() -> None:
    """Scoring an unwrapped residual would weight out a particle that is in fact correct.

    Purpose: Validates that the density wraps the heading residual before scoring

    Given: A reading whose heading entry differs from the truth by exactly 2*pi
    When: Its log-probability is scored
    Then: It equals the log-probability of the unshifted reading

    Test type: unit
    """
    model = GoalRelativePoseObservationModel(heading_std=0.1)
    state = _state((0.0, 0.0, 0.0), (1.0, 1.0, 0.3))
    truth = model.clean_offset(state)
    shifted = truth.copy()
    shifted[2] += 2.0 * np.pi
    assert model.log_probability(state, shifted) == pytest.approx(
        model.log_probability(state, truth), abs=1e-9
    )


def test_log_probability_rejects_a_reading_of_the_wrong_width() -> None:
    """A width mismatch means the channel was mis-wired, not that the state is unlikely.

    Purpose: Validates that a differently shaped reading scores as impossible

    Given: A two-wide reading against a three-wide channel
    When: Its log-probability is scored
    Then: The result is -inf

    Test type: unit
    """
    model = GoalRelativePoseObservationModel()
    state = _state((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert model.log_probability(state, np.zeros(2)) == float("-inf")


def test_perceive_recovers_the_truth_on_average() -> None:
    """Biased noise would pull the belief off the true pose by a fixed amount every step.

    Purpose: Validates that the perceived channel is the clean offset plus zero-mean noise

    Given: A fixed pose and goal, sampled many times
    When: The perceived offsets are averaged
    Then: The mean matches the clean offset to within the standard error

    Test type: unit
    """
    np.random.seed(0)
    model = GoalRelativePoseObservationModel(position_std=0.2, heading_std=0.1)
    state = _state((0.5, -0.5, 0.3), (2.0, 1.0, -0.4))
    samples = np.stack([model.perceive(state) for _ in range(4000)])
    assert samples.mean(axis=0) == pytest.approx(model.clean_offset(state), abs=0.02)


def test_a_goal_block_without_a_heading_is_accepted() -> None:
    """A study whose goal is a position only should not have to pad it with a fake heading.

    Purpose: Validates the position-only goal block path

    Given: A model built with goal_indices=None and a two-wide goal block
    When: The clean offset is computed
    Then: The position entries are correct and the heading is measured against zero

    Test type: unit
    """
    model = GoalRelativePoseObservationModel(goal_indices=None)
    offset = model.clean_offset({"base_pose": np.zeros(3), "goal": np.array([3.0, 0.0])})
    assert offset == pytest.approx(np.array([3.0, 0.0, 0.0]), abs=1e-9)


@pytest.mark.parametrize(
    "std_name, position_std, heading_std",
    [("position_std", 0.0, 0.05), ("heading_std", 0.1, 0.0)],
)
def test_a_non_positive_noise_scale_is_rejected(
    std_name: str, position_std: float, heading_std: float
) -> None:
    """A zero std makes every particle score -inf and the belief collapse without warning.

    Purpose: Validates construction-time rejection of non-positive noise scales

    Given: A model built with one noise scale set to zero
    When: It is constructed
    Then: ValueError is raised naming that scale

    Test type: unit
    """
    with pytest.raises(ValueError, match=std_name):
        GoalRelativePoseObservationModel(position_std=position_std, heading_std=heading_std)


def test_state_channels_declare_both_blocks_the_model_reads() -> None:
    """A generative model checks its schema against this before a run, not during one.

    Purpose: Validates the declared state-channel dependency

    Given: A model built over named pose and goal blocks
    When: Its declared state channels are read
    Then: Both block names appear, pose first

    Test type: unit
    """
    model = GoalRelativePoseObservationModel(pose_channel="pose", goal_channel="target")
    assert model.state_channels == ("pose", "target")


@pytest.mark.parametrize("angle", [0.0, 3.0, -3.0, 7.0, -7.0, np.pi])
def test_wrap_to_pi_lands_in_the_half_open_interval(angle: float) -> None:
    """Wrapping is used on every heading residual, so an off-by-one interval is systemic.

    Purpose: Validates that wrapping maps any angle into (-pi, pi]

    Given: Angles inside and well outside one turn
    When: They are wrapped
    Then: Each result is in (-pi, pi] and differs from the input by a multiple of 2*pi

    Test type: unit
    """
    wrapped = float(wrap_to_pi(angle))
    assert -np.pi - 1e-9 < wrapped <= np.pi + 1e-9
    assert (angle - wrapped) / (2.0 * np.pi) == pytest.approx(
        round((angle - wrapped) / (2.0 * np.pi)), abs=1e-9
    )
