# SPDX-License-Identifier: MIT

"""Unit tests for the proprioceptive per-channel observation models.

The Gaussian model carries the compatibility burden: a single channel over the whole state must
reproduce the one-space model's ``observation = state + N(0, Sigma)`` exactly, or rebasing an
existing study onto the factored stack silently changes its noise.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.isaac_lab_pomdp import GaussianObservationModel
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
    ExactChannelObservationModel,
    GaussianChannelObservationModel,
)


def test_gaussian_channel_reproduces_the_one_space_density_exactly() -> None:
    """The factored Gaussian channel must be the same distribution as the one-space model.

    Purpose: Validates the regression path from the one-space model to a factored channel

    Given: A one-space GaussianObservationModel and a GaussianChannelObservationModel over the
        whole state, both with the same noise std
    When: The same observation is scored under both
    Then: The log-densities agree

    Test type: unit
    """
    state = np.array([0.5, -1.5, 2.0, 0.0])
    reading = np.array([0.6, -1.4, 2.2, 0.1])

    one_space = GaussianObservationModel(observation_dim=4, noise_std=0.2)
    factored = GaussianChannelObservationModel(channel="state", noise_std=0.2)

    expected = float(np.ravel(one_space.log_probability(state, reading))[0])
    assert factored.log_probability({"state": state}, reading) == pytest.approx(expected)


def test_gaussian_channel_samples_around_the_state_block() -> None:
    """A perceived channel is the block plus zero-mean noise, not an arbitrary draw.

    Purpose: Validates the mean and spread of the Gaussian channel's samples

    Given: A Gaussian channel with std 0.1 over a fixed 3-wide block
    When: Many observations are drawn
    Then: Their mean is the block and their std is the configured one

    Test type: unit
    """
    np.random.seed(0)
    block = np.array([1.0, -2.0, 0.5])
    model = GaussianChannelObservationModel(channel="base_pose", noise_std=0.1)
    draws = np.stack([model.perceive({"base_pose": block}) for _ in range(4000)])
    assert draws.mean(axis=0) == pytest.approx(block, abs=0.02)
    assert draws.std(axis=0) == pytest.approx([0.1, 0.1, 0.1], abs=0.02)


def test_gaussian_channel_reads_a_differently_named_state_block() -> None:
    """The observation channel and the state block it comes from need not share a name.

    Purpose: Validates the state_channel indirection

    Given: A channel "measured_pose" configured to read the state block "true_pose"
    When: It perceives a clean state carrying both
    Then: The draw is centred on "true_pose" and the declared state_channels say so

    Test type: unit
    """
    np.random.seed(1)
    model = GaussianChannelObservationModel(
        channel="measured_pose", state_channel="true_pose", noise_std=1e-6
    )
    assert model.state_channels == ("true_pose",)
    clean = {"true_pose": np.array([3.0, 4.0]), "measured_pose": np.array([-9.0, -9.0])}
    assert model.perceive(clean) == pytest.approx([3.0, 4.0], abs=1e-4)


def test_gaussian_channel_handles_blocks_of_different_widths() -> None:
    """One model definition serves several tasks, whose blocks differ in width.

    Purpose: Validates the lazily built, width-keyed normal

    Given: One Gaussian channel model
    When: It perceives a 2-wide block and then a 5-wide block
    Then: Both draws come back at the right width

    Test type: unit
    """
    np.random.seed(2)
    model = GaussianChannelObservationModel(channel="joints", noise_std=0.05)
    assert model.perceive({"joints": np.zeros(2)}).shape == (2,)
    assert model.perceive({"joints": np.zeros(5)}).shape == (5,)


def test_exact_channel_returns_the_block_and_scores_a_mismatch_at_minus_infinity() -> None:
    """A known channel must weight out any particle that disagrees with it.

    Purpose: Validates the degenerate density of a noise-free channel

    Given: An exact channel over a commanded goal pose
    When: A matching and a mismatching reading are scored
    Then: The match scores 0.0 and the mismatch scores -inf

    Test type: unit
    """
    model = ExactChannelObservationModel(channel="pose_command")
    clean = {"pose_command": np.array([2.0, -1.0])}
    assert model.perceive(clean) == pytest.approx([2.0, -1.0])
    assert model.log_probability(clean, np.array([2.0, -1.0])) == 0.0
    assert model.log_probability(clean, np.array([2.0, -1.1])) == float("-inf")


def test_exact_channel_scores_a_wrong_width_at_minus_infinity() -> None:
    """A reading of the wrong width is impossible, not merely unlikely.

    Purpose: Validates the shape guard on the exact channel's density

    Given: An exact channel over a 2-wide block
    When: A 3-wide reading is scored
    Then: The score is -inf rather than a broadcast comparison

    Test type: unit
    """
    model = ExactChannelObservationModel(channel="pose_command")
    clean = {"pose_command": np.array([2.0, -1.0])}
    assert model.log_probability(clean, np.array([2.0, -1.0, 0.0])) == float("-inf")


def test_perceived_channel_does_not_alias_the_state_block() -> None:
    """A returned view would let a caller mutate the state through its observation.

    Purpose: Validates that the exact channel copies rather than aliases

    Given: An exact channel and a clean state block
    When: The perceived value is mutated
    Then: The state block is unchanged

    Test type: unit
    """
    block = np.array([1.0, 2.0])
    perceived = ExactChannelObservationModel(channel="pose_command").perceive(
        {"pose_command": block}
    )
    perceived[0] = 99.0
    assert block[0] == pytest.approx(1.0)
