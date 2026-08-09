# SPDX-License-Identifier: MIT

"""Unit tests for the concrete factored Isaac generative model.

Two properties carry the weight here. First, that a single Gaussian channel over the whole state
reproduces the one-space model exactly, so rebasing an existing study does not silently change its
noise. Second, that ``transition_channels`` really does hold the undriven blocks fixed — a latent
type that drifted would destroy the belief dispersion a risk-sensitive planner grades.
"""

from typing import Any

import numpy as np
import pytest

from POMDPPlanners.environments.isaac_lab_pomdp import (
    GaussianRandomWalkTransition,
    IsaacLabModelPOMDP,
    LinearGaussianTransition,
    RewardModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
    FactoredIsaacModelPOMDP,
    IsaacChannelSchema,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
    GaussianChannelObservationModel,
)

SCHEMA = IsaacChannelSchema((("robot", 4), ("hazard_type", 2)))
ACTIONS = [np.zeros(3), np.ones(3), -np.ones(3)]


class _PositionReward(RewardModel):
    """Reward equal to minus the distance from the origin, over whatever block it is given."""

    def reward(self, state: Any, action: Any, next_state: Any) -> float:
        del state, action
        return -float(np.linalg.norm(np.asarray(next_state, dtype=float)))


def _fitted_transition(seed: int = 0) -> LinearGaussianTransition:
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(200, 4))
    actions = rng.normal(size=(200, 3))
    next_states = states + 0.05 * rng.normal(size=(200, 4))
    return LinearGaussianTransition.fit(states, actions, next_states)


def _model(**overrides: Any) -> FactoredIsaacModelPOMDP:
    settings: dict = {
        "state_schema": SCHEMA,
        "action_presets": ACTIONS,
        "discount_factor": 0.99,
        "transition": _fitted_transition(),
        "transition_channels": ("robot",),
        "observation_models": {
            "robot": GaussianChannelObservationModel(channel="robot", noise_std=0.1)
        },
    }
    settings.update(overrides)
    return FactoredIsaacModelPOMDP(**settings)


def _state(robot: Any = (0.0, 0.0, 0.0, 0.0), types: Any = (0.0, 1.0)) -> np.ndarray:
    return SCHEMA.pack({"robot": np.asarray(robot, dtype=float), "hazard_type": types})


# ── Regression to the one-space model ────────────────────────────────────


def test_one_channel_over_the_whole_state_reproduces_the_one_space_model() -> None:
    """The factored stack must be able to regress exactly to what it replaces.

    Purpose: Validates that a single Gaussian channel over the whole state matches
        IsaacLabModelPOMDP's observation and transition densities

    Given: A one-space model and a factored model, both with a random-walk transition of the same
        noise and a Gaussian observation of the same noise
    When: The same transition and the same observation are scored under both
    Then: The densities agree

    Test type: unit
    """
    schema = IsaacChannelSchema((("state", 4),))
    one_space = IsaacLabModelPOMDP(
        observation_dim=4,
        action_presets=ACTIONS,
        discount_factor=0.99,
        observation_noise_std=0.2,
        process_noise_std=0.05,
    )
    factored = FactoredIsaacModelPOMDP(
        state_schema=schema,
        action_presets=ACTIONS,
        discount_factor=0.99,
        transition=GaussianRandomWalkTransition(dim=4, process_noise_std=0.05),
        observation_models={
            "state": GaussianChannelObservationModel(channel="state", noise_std=0.2)
        },
    )

    state = np.array([0.5, -1.0, 2.0, 0.0])
    next_state = np.array([0.55, -0.95, 2.05, 0.02])
    reading = np.array([0.6, -0.9, 2.1, 0.05])

    assert float(factored.transition_log_probability(state, ACTIONS[0], next_state)[0]) == (
        pytest.approx(float(one_space.transition_log_probability(state, ACTIONS[0], next_state)[0]))
    )
    assert float(
        factored.observation_log_probability(next_state, None, {"state": reading})[0]
    ) == pytest.approx(
        float(np.ravel(one_space.observation_log_probability(next_state, None, reading))[0])
    )


# ── Partially-driven state ───────────────────────────────────────────────


def test_an_undriven_block_is_carried_through_the_transition_unchanged() -> None:
    """Resampling the latent type each step would destroy the dispersion CVaR needs.

    Purpose: Validates that transition_channels holds the undriven blocks fixed

    Given: A model whose transition drives only the robot block
    When: One and then eight successors are sampled
    Then: Every successor carries the original hazard block, while the robot block moves

    Test type: unit
    """
    model = _model()
    state = _state(types=(0.0, 1.0))
    single = model.sample_next_state(state, np.ones(3))
    batch = model.sample_next_state(state, np.ones(3), n_samples=8)

    assert SCHEMA.block(single, "hazard_type") == pytest.approx([0.0, 1.0])
    assert batch.shape == (8, SCHEMA.total_dim)
    assert np.all(SCHEMA.block(batch, "hazard_type") == np.array([0.0, 1.0]))
    assert not np.allclose(SCHEMA.block(batch, "robot")[0], SCHEMA.block(batch, "robot")[1])


def test_the_transition_only_ever_sees_the_driven_block() -> None:
    """A transition fitted on the robot block must not be handed an augmented vector.

    Purpose: Validates the width the transition is queried at

    Given: A transition fitted on 4-wide robot states inside a 6-wide schema
    When: A successor is sampled
    Then: It succeeds and the driven block is 4 wide, so no wider vector was passed in

    Test type: unit
    """
    model = _model()
    successor = model.sample_next_state(_state(), np.ones(3))
    assert successor.shape == (SCHEMA.total_dim,)
    assert SCHEMA.block(successor, "robot").shape == (4,)


def test_transition_density_scores_only_the_driven_block() -> None:
    """The carried block is deterministic, so it must contribute no density term.

    Purpose: Validates that the transition density ignores the undriven channels

    Given: Two candidate successors identical on the robot block but differing on the type block
    When: Both are scored
    Then: The scores agree

    Test type: unit
    """
    model = _model()
    state = _state()
    robot_next = np.array([0.1, 0.1, 0.1, 0.1])
    first = SCHEMA.pack({"robot": robot_next, "hazard_type": [0.0, 1.0]})
    second = SCHEMA.pack({"robot": robot_next, "hazard_type": [1.0, 0.0]})
    scores = model.transition_log_probability(state, np.ones(3), np.stack([first, second]))
    assert scores[0] == pytest.approx(scores[1])


def test_without_transition_channels_the_transition_drives_the_whole_state() -> None:
    """The default has to stay the simple case, or every existing wiring changes meaning.

    Purpose: Validates the None default of transition_channels

    Given: A model with no transition_channels and a random walk over the full width
    When: A successor is sampled
    Then: Every channel has moved

    Test type: unit
    """
    model = _model(
        transition=GaussianRandomWalkTransition(dim=SCHEMA.total_dim, process_noise_std=0.1),
        transition_channels=None,
    )
    successor = model.sample_next_state(_state(), np.ones(3))
    assert not np.allclose(SCHEMA.block(successor, "hazard_type"), [0.0, 1.0])


def test_transition_channels_naming_an_unknown_block_is_rejected() -> None:
    """A misspelled channel would silently drive nothing and freeze the robot.

    Purpose: Validates the construction-time guard on transition_channels

    Given: A transition channel the schema does not declare
    When: The model is constructed
    Then: ValueError is raised naming it

    Test type: unit
    """
    with pytest.raises(ValueError, match="wheels"):
        _model(transition_channels=("wheels",))


def test_sampling_does_not_mutate_the_state_it_was_given() -> None:
    """The belief reuses particle arrays, so an in-place write corrupts the parent.

    Purpose: Validates that sample_next_state leaves its argument alone

    Given: A state vector handed to the sampler
    When: A successor is drawn
    Then: The original vector is unchanged

    Test type: unit
    """
    model = _model()
    state = _state(robot=(1.0, 2.0, 3.0, 4.0))
    before = state.copy()
    model.sample_next_state(state, np.ones(3))
    assert np.array_equal(state, before)


# ── Reward and terminal ──────────────────────────────────────────────────


def test_reward_delegates_to_the_reward_model_on_the_resulting_state() -> None:
    """The objective is a property of where the step landed, not where it started.

    Purpose: Validates that the reward is evaluated at next_state

    Given: A reward equal to minus the distance from the origin
    When: A step from the origin to a distant state is scored
    Then: The reward reflects the distant state

    Test type: unit
    """
    model = _model(reward_model=_PositionReward())
    origin = _state(types=(0.0, 0.0))
    far = _state(robot=(3.0, 4.0, 0.0, 0.0), types=(0.0, 0.0))
    assert model.reward(origin, ACTIONS[0], far) == pytest.approx(-5.0)


def test_reward_falls_back_to_the_state_when_no_successor_is_given() -> None:
    """Some call sites score a state alone; scoring nothing would look like a zero reward.

    Purpose: Validates the next_state=None path of reward

    Given: A reward model and a single state
    When: reward is called without a successor
    Then: The state itself is scored

    Test type: unit
    """
    model = _model(reward_model=_PositionReward())
    state = _state(robot=(3.0, 4.0, 0.0, 0.0), types=(0.0, 0.0))
    assert model.reward(state, ACTIONS[0]) == pytest.approx(-5.0)


def test_no_reward_model_yields_a_flat_zero() -> None:
    """Undirected planning is a documented state, so it must be an explicit zero.

    Purpose: Validates the default reward when no model is supplied

    Given: A model constructed without a reward model
    When: A transition is scored
    Then: The reward is 0.0

    Test type: unit
    """
    assert _model().reward(_state(), ACTIONS[0], _state(robot=(9.0, 9.0, 9.0, 9.0))) == 0.0


def test_the_model_never_declares_a_state_terminal() -> None:
    """Termination belongs to the world; a model guessing it would prune reachable states.

    Purpose: Validates that is_terminal is always False on the model side

    Given: A factored model
    When: Any state is tested
    Then: It is not terminal

    Test type: unit
    """
    model = _model()
    assert model.is_terminal(_state()) is False
    assert model.is_terminal(_state(robot=(1e6, 1e6, 1e6, 1e6))) is False


def test_reward_range_reaches_the_environment_for_a_constrained_planner() -> None:
    """A constrained planner caps its Lagrange multiplier with this; without it the cap is +inf.

    Purpose: Validates that reward_range is forwarded to the Environment base

    Given: A model constructed with explicit reward bounds
    When: The bounds are read back
    Then: They match

    Test type: unit
    """
    assert _model(reward_range=(-30.0, 30.0)).reward_range == (-30.0, 30.0)
