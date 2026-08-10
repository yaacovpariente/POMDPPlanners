# SPDX-License-Identifier: MIT

"""Unit tests for the fitted-dynamics Isaac model and its block-restricted reward.

The property under test is confinement: a model fitted on the channels the warm-up rollouts
measured must never be handed a wider augmented state. Violating it does not raise — it multiplies
fitted coefficients against blocks the fit never saw and returns a plausible wrong number.
"""

from typing import Any

import numpy as np
import pytest

from POMDPPlanners.environments.isaac_lab_pomdp import LinearRewardModel, RewardModel
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
    BlockRewardModel,
    IsaacChannelSchema,
    LearnedIsaacModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
    GaussianChannelObservationModel,
)

SCHEMA = IsaacChannelSchema((("robot", 4), ("hazard_type", 2)))
ACTIONS = [np.zeros(3), np.ones(3), -np.ones(3)]


class _WidthReportingReward(RewardModel):
    """A reward that returns the width of the state it was handed."""

    def reward(self, state: Any, action: Any, next_state: Any) -> float:
        del action, next_state
        return float(np.asarray(state, dtype=float).shape[-1])


def _rollouts(seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(200, 4))
    actions = rng.normal(size=(200, 3))
    next_states = states + 0.05 * rng.normal(size=(200, 4))
    rewards = -np.linalg.norm(states, axis=1)
    return states, actions, next_states, rewards


def _fitted(**overrides: Any) -> LearnedIsaacModel:
    states, actions, next_states, rewards = _rollouts()
    settings: dict = {
        "state_schema": SCHEMA,
        "dynamics_channels": ("robot",),
        "action_presets": ACTIONS,
        "discount_factor": 0.99,
        "states": states,
        "actions": actions,
        "next_states": next_states,
        "rewards": rewards,
        "observation_models": {
            "robot": GaussianChannelObservationModel(channel="robot", noise_std=0.1)
        },
    }
    settings.update(overrides)
    return LearnedIsaacModel.fit(**settings)


def _state(robot: Any = (0.0, 0.0, 0.0, 0.0), types: Any = (0.0, 1.0)) -> np.ndarray:
    return SCHEMA.pack({"robot": np.asarray(robot, dtype=float), "hazard_type": types})


def test_block_reward_hands_the_wrapped_model_only_its_own_channels() -> None:
    """A fit is valid on the block it was trained on and nowhere else.

    Purpose: Validates that BlockRewardModel slices before delegating

    Given: A reward that reports the width of the state it receives, wrapped over a 4-wide block
        inside a 6-wide schema
    When: A full augmented state is scored
    Then: The wrapped model reports 4, not 6

    Test type: unit
    """
    model = BlockRewardModel(_WidthReportingReward(), SCHEMA, ("robot",))
    assert model.reward(_state(), None, _state()) == pytest.approx(4.0)


def test_block_reward_slices_in_the_order_requested() -> None:
    """The fit's coefficient order is fixed, so the slice order has to match it.

    Purpose: Validates that BlockRewardModel honours the channel order given

    Given: A reward summing its next_state, restricted to the type block then the robot block
    When: A state with a distinguishable layout is scored
    Then: The delegate sees the channels in the requested order

    Test type: unit
    """

    class _FirstEntry(RewardModel):
        def reward(self, state: Any, action: Any, next_state: Any) -> float:
            del state, action
            return float(np.asarray(next_state, dtype=float)[0])

    model = BlockRewardModel(_FirstEntry(), SCHEMA, ("hazard_type", "robot"))
    state = _state(robot=(9.0, 0.0, 0.0, 0.0), types=(7.0, 0.0))
    assert model.reward(state, None, state) == pytest.approx(7.0)


def test_block_reward_rejects_an_undeclared_channel() -> None:
    """A misnamed block would slice the wrong coefficients into the fit.

    Purpose: Validates the construction-time guard on BlockRewardModel

    Given: A channel the schema does not declare
    When: The adapter is constructed
    Then: ValueError is raised naming it

    Test type: unit
    """
    with pytest.raises(ValueError, match="wheels"):
        BlockRewardModel(_WidthReportingReward(), SCHEMA, ("wheels",))


def test_fit_confines_both_the_transition_and_the_reward_to_the_measured_block() -> None:
    """An augmented schema must not require refitting the pieces trained without it.

    Purpose: Validates that a model fitted on 4-wide rollouts runs inside a 6-wide schema

    Given: Rollouts recorded on the robot block only
    When: A successor is sampled and a transition is scored in the augmented schema
    Then: Both succeed and the latent block is carried through unchanged

    Test type: unit
    """
    model = _fitted()
    state = _state(types=(0.0, 1.0))
    successor = model.sample_next_state(state, np.ones(3))
    assert successor.shape == (SCHEMA.total_dim,)
    assert SCHEMA.block(successor, "hazard_type") == pytest.approx([0.0, 1.0])
    assert model.reward(state, np.ones(3), successor) == pytest.approx(
        float(
            LinearRewardModel.fit(*_rollouts()).reward(
                SCHEMA.block(state, "robot"), np.ones(3), SCHEMA.block(successor, "robot")
            )
        )
    )


def test_fit_without_rewards_leaves_the_model_undirected() -> None:
    """Warm-up rollouts sometimes record no reward, and that must be an explicit state.

    Purpose: Validates the rewards=None path of the fit constructor

    Given: A fit with no rewards supplied
    When: A transition is scored
    Then: The reward is 0.0

    Test type: unit
    """
    assert _fitted(rewards=None).reward(_state(), np.ones(3), _state()) == 0.0


def test_the_fitted_transition_is_action_conditioned() -> None:
    """A transition that ignored the action would make the planner's lookahead cosmetic.

    Purpose: Validates that different actions produce different predicted successors

    Given: A fitted model and two different action presets
    When: The same state is advanced under each with the noise averaged out
    Then: The predicted successors differ

    Test type: unit
    """
    np.random.seed(0)
    model = _fitted()
    state = _state(robot=(0.5, -0.5, 0.25, 0.0))
    first = model.sample_next_state(state, ACTIONS[1], n_samples=2000).mean(axis=0)
    second = model.sample_next_state(state, ACTIONS[2], n_samples=2000).mean(axis=0)
    assert not np.allclose(SCHEMA.block(first, "robot"), SCHEMA.block(second, "robot"), atol=1e-3)


def test_the_fitted_model_observes_only_the_channels_it_was_configured_with() -> None:
    """A learned model still has to keep the latent block out of the observation.

    Purpose: Validates the observation mapping of the fitted model

    Given: A fitted model with a single robot observation channel
    When: An observation is sampled
    Then: It carries only that channel

    Test type: unit
    """
    np.random.seed(0)
    observation = _fitted().sample_observation(_state(), None)
    assert sorted(observation) == ["robot"]
