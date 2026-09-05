# SPDX-License-Identifier: MIT

"""Conformance tests for planning with a transition that is not the environment's own.

Every model-learning study builds one of these: the task, with its dynamics
replaced. The failures these tests describe are silent at runtime and expensive
afterwards -- a study that ran, produced numbers, and measured the wrong thing.

The environment under test is a factored model with a carried latent, which is
the shape every real use has here.
"""

from typing import Any

import numpy as np
import pytest

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
    FactoredIsaacModelPOMDP,
    IsaacChannelSchema,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
    GaussianChannelObservationModel,
)
from POMDPPlanners.tests.test_utils.substituted_transition import (
    RecordingTransition,
    assert_carried_channels_preserved,
    assert_observation_matches_reference,
    assert_transition_is_used_everywhere,
)

SCHEMA = IsaacChannelSchema((("robot", 3), ("hazard_type", 2)))
PRESETS = [np.array([1.0, 0.0]), np.array([-1.0, 0.0])]
STATE = SCHEMA.pack({"robot": np.array([0.4, -0.2, 0.1]), "hazard_type": np.array([1.0, 0.0])})
CARRIED = list(SCHEMA.indices_of(["hazard_type"]))
# The task exposes the robot block and hides the latent -- which is the whole
# point of the width check below.
OBSERVATION_MODELS = {"robot": GaussianChannelObservationModel(channel="robot", noise_std=0.05)}


def _model(transition: Any) -> FactoredIsaacModelPOMDP:
    """The task, built to plan with ``transition`` over the robot block."""
    return FactoredIsaacModelPOMDP(
        state_schema=SCHEMA,
        action_presets=PRESETS,
        discount_factor=0.99,
        transition=transition,
        transition_channels=("robot",),
        observation_models=OBSERVATION_MODELS,
    )


class _BatchPathIgnoresSubstitute(FactoredIsaacModelPOMDP):
    """A model whose batch path is written against dynamics of its own.

    This is the mistake, made deliberately: the single-step path uses the
    supplied transition and the batch path does not, so a planner that
    propagates particles in batches searches something else entirely.
    """

    def sample_next_state_batch(self, states: Any, action: Any) -> np.ndarray:
        del action
        return np.atleast_2d(np.asarray(states, dtype=float))


class TestEveryDynamicsPathUsesTheSubstitute:
    """A path that answers without asking is a path still on the old dynamics."""

    def test_a_correctly_wired_model_consults_it_on_every_path(self) -> None:
        """Purpose: Validates the check passes on a model wired the way it should be

        Given: A factored model whose paths all route through the stored transition
        When: Every dynamics path is exercised with a recording transition
        Then: Each one consulted it
        """
        assert_transition_is_used_everywhere(
            build_model=_model, state=STATE, action=PRESETS[0], dim=3
        )

    def test_a_batch_path_of_its_own_is_caught(self) -> None:
        """Purpose: Validates the check catches the trap it exists for

        Given: A model whose batch path ignores the supplied transition
        When: The paths are exercised
        Then: The check fails, naming the batch path, instead of the study
            reporting numbers from the wrong dynamics
        """
        with pytest.raises(AssertionError, match="sample_next_state_batch"):
            assert_transition_is_used_everywhere(
                build_model=lambda transition: _BatchPathIgnoresSubstitute(
                    state_schema=SCHEMA,
                    action_presets=PRESETS,
                    discount_factor=0.99,
                    transition=transition,
                    transition_channels=("robot",),
                ),
                state=STATE,
                action=PRESETS[0],
                dim=3,
            )


class TestCarriedChannels:
    """The latent has to come out of a step exactly as it went in."""

    def test_a_carried_latent_survives_the_substitute(self) -> None:
        """Purpose: Validates that a substitute confined to its block leaves the latent alone

        Given: A model planning with a substitute over the robot block only
        When: Successors are drawn
        Then: The hazard type is unchanged in every one
        """
        assert_carried_channels_preserved(
            _model(RecordingTransition(dim=3)), STATE, PRESETS[0], CARRIED
        )

    def test_a_substitute_that_drives_the_whole_state_is_caught(self) -> None:
        """Purpose: Validates the check catches a fit that was given the latent

        Given: A model whose transition drives every channel, latent included
        When: Successors are drawn
        Then: The check fails rather than the belief quietly losing its spread
            over the latent
        """
        whole_state = FactoredIsaacModelPOMDP(
            state_schema=SCHEMA,
            action_presets=PRESETS,
            discount_factor=0.99,
            transition=RecordingTransition(dim=5),
            transition_channels=None,
        )
        with pytest.raises(AssertionError, match="carried channel moved"):
            assert_carried_channels_preserved(whole_state, STATE, PRESETS[0], CARRIED)


class TestObservationWidth:
    """An observation wider than the task's hands the planner what it should infer."""

    def test_an_observation_of_the_same_width_passes(self) -> None:
        """Purpose: Validates the check passes when the twin observes what the task exposes

        Given: Two models built the same way
        When: Their observations are compared
        Then: The check passes
        """
        assert_observation_matches_reference(
            _model(RecordingTransition(dim=3)),
            _model(RecordingTransition(dim=3)),
            STATE,
            PRESETS[0],
        )

    def test_an_observation_carrying_extra_channels_is_caught(self) -> None:
        """Purpose: Validates the check catches a twin that observes the latent

        Given: A model whose observation appends channels the reference does not have
        When: The two are compared
        Then: The check fails, naming the widths
        """

        class _WiderObservation(FactoredIsaacModelPOMDP):
            """Reports the latent as an observation channel -- the leak, on purpose."""

            def sample_observation(
                self, next_state: Any, action: Any = None, n_samples: int = 1
            ) -> Any:
                observation = dict(super().sample_observation(next_state, action, n_samples))
                observation["hazard_type"] = SCHEMA.block(next_state, "hazard_type")
                return observation

        wider = _WiderObservation(
            state_schema=SCHEMA,
            action_presets=PRESETS,
            discount_factor=0.99,
            transition=RecordingTransition(dim=3),
            transition_channels=("robot",),
            observation_models=OBSERVATION_MODELS,
        )
        with pytest.raises(AssertionError, match="the planner should have to infer"):
            assert_observation_matches_reference(
                wider, _model(RecordingTransition(dim=3)), STATE, PRESETS[0]
            )
