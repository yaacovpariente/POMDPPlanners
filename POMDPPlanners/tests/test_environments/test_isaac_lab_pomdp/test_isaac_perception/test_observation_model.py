# SPDX-License-Identifier: MIT

"""Unit tests for the per-channel IsaacObservationModel interface.

Covers the two behaviours the ABC itself owns: the default sample-only density (which must refuse
rather than silently return something) and the default ``encode``, whose identity semantics are
the deliberate difference from the CARLA equivalent.
"""

from typing import Mapping

import numpy as np
import pytest

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import IsaacObservationModel


class _SampleOnlyModel(IsaacObservationModel):
    """A channel that can be sampled but not scored, like a learned encoder."""

    channel = "latent"
    state_channels = ("robot",)

    def perceive(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.asarray(clean_state["robot"], dtype=float) * 2.0


def test_default_log_probability_refuses_for_a_sample_only_channel() -> None:
    """A channel with no density must raise, naming itself.

    Purpose: Validates that a sample-only observation channel cannot be silently used to score

    Given: A concrete model that implements perceive but not log_probability
    When: log_probability is called on it
    Then: NotImplementedError is raised and the message names the concrete class

    Test type: unit
    """
    model = _SampleOnlyModel()
    with pytest.raises(NotImplementedError, match="_SampleOnlyModel"):
        model.log_probability({"robot": np.zeros(2)}, np.zeros(2))


def test_supports_density_defaults_to_false() -> None:
    """The density flag must default off so a missing density is opt-in to discover.

    Purpose: Validates the conservative default of the supports_density flag

    Given: A concrete model that does not set supports_density
    When: The flag is read
    Then: It is False, so a generative model rejects the channel for belief updates

    Test type: unit
    """
    assert _SampleOnlyModel().supports_density is False


def test_encode_defaults_to_the_identity_not_to_perceive() -> None:
    """An Isaac world emits a real sensor reading; re-perceiving it would noise it twice.

    Purpose: Validates that encoding a raw world reading leaves it untouched by default

    Given: A model whose perceive doubles its input
    When: encode is called on a raw channel reading
    Then: The reading is returned unchanged rather than doubled

    Test type: unit
    """
    model = _SampleOnlyModel()
    raw = np.array([1.0, 2.0])
    assert np.array_equal(model.encode(raw), raw)
    assert np.array_equal(model.perceive({"robot": raw}), 2.0 * raw)
