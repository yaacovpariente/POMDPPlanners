# SPDX-License-Identifier: MIT

"""Parity and behaviour tests for the vectorized goal-relative navigation model.

The torch kernels duplicate numeric code that also exists in numpy, so most of these tests pin the
two to each other. A drift between them would not raise anywhere -- the planner would simply search
a different model from the one the scalar tests describe, and the scalar tests would keep passing.

The action-separation test is the regression guard for the measured failure this model exists to
fix: under the fitted linear surrogate every action scored alike and VOPP emitted one action index
for a whole episode.
"""

from typing import Any

import numpy as np
import pytest
import torch

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
    NavigationIsaacModel,
    navigation_state_schema,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_navigation_vectorized_model import (  # noqa: E501  pylint: disable=line-too-long
    NavigationVectorizedModel,
)

SCHEMA = navigation_state_schema()
STATE_DIM = SCHEMA.total_dim
STEP_DT = 0.2
PRESETS = [
    np.zeros(3),
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.6, -0.4, 0.5]),
]
CPU = torch.device("cpu")
CLEAN_NOISE = {
    "velocity_noise_std": 1e-9,
    "position_noise_std": 1e-9,
    "heading_noise_std": 1e-9,
}


def _scalar_model(**overrides: Any) -> NavigationIsaacModel:
    settings: dict = {
        "state_schema": SCHEMA,
        "action_presets": PRESETS,
        "discount_factor": 0.99,
        "step_dt": STEP_DT,
        "linear_scale": 0.8,
        "angular_scale": 0.6,
        **CLEAN_NOISE,
    }
    settings.update(overrides)
    return NavigationIsaacModel(**settings)


def _vectorized(**overrides: Any) -> NavigationVectorizedModel:
    return NavigationVectorizedModel(_scalar_model(**overrides), device=CPU, dtype=torch.float64)


def _random_states(count: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    states = np.zeros((count, STATE_DIM))
    states[:, SCHEMA.slice_of("base_lin_vel")] = rng.uniform(-1.0, 1.0, size=(count, 3))
    states[:, SCHEMA.slice_of("projected_gravity")] = rng.uniform(-0.2, 0.2, size=(count, 3))
    goal = SCHEMA.slice_of("pose_command")
    states[:, goal.start : goal.start + 2] = rng.uniform(-3.0, 3.0, size=(count, 2))
    states[:, goal.start + 2] = rng.uniform(-0.05, 0.05, size=count)
    states[:, goal.start + 3] = rng.uniform(-np.pi, np.pi, size=count)
    return torch.as_tensor(states, dtype=torch.float64, device=CPU)


# ── Parity with the scalar model ────────────────────────────────────────


def test_the_torch_transition_matches_the_numpy_prediction() -> None:
    """Two implementations of one integration drift silently; only a parity test notices.

    Purpose: Pins the batched torch transition to the scalar model's own prediction

    Given: Sixteen random states and one action index each, with process noise near zero
    When: Both implementations take one step
    Then: Every entry of the predicted next states agrees

    Test type: unit
    """
    scalar, model = _scalar_model(), _vectorized()
    states = _random_states(16, seed=1)
    actions = torch.arange(16) % len(PRESETS)
    predicted = model.sample_next_states(states, actions)
    for row in range(16):
        expected = scalar.sample_next_state(
            states[row].numpy(), np.asarray(PRESETS[int(actions[row])])
        )
        np.testing.assert_allclose(predicted[row].numpy(), expected, atol=1e-6)


def test_the_torch_reward_matches_the_numpy_objective() -> None:
    """The reward is what the search climbs; a mismatch would optimize an undocumented objective.

    Purpose: Pins the batched reward to the scalar NavigationRewardModel

    Given: Sixteen random next states
    When: Both implementations score them
    Then: Every reward agrees

    Test type: unit
    """
    scalar, model = _scalar_model(), _vectorized()
    reward_model = scalar.navigation_reward
    assert reward_model is not None
    states = _random_states(16, seed=2)
    actions = torch.zeros(16, dtype=torch.long)
    rewards = model.rewards(states, actions, states)
    expected = [reward_model.reward(row.numpy(), None, row.numpy()) for row in states]
    np.testing.assert_allclose(rewards.numpy(), expected, atol=1e-9)


def test_the_torch_goal_distances_match_the_numpy_accessors() -> None:
    """The planar distance is what the episode is scored on, the 3-D one what the reward reads.

    Purpose: Pins both distance accessors to the scalar reward model's

    Given: Sixteen random states
    When: Both implementations measure the goal distance, in three dimensions and in the plane
    Then: Both agree entry by entry

    Test type: unit
    """
    scalar, model = _scalar_model(), _vectorized()
    reward_model = scalar.navigation_reward
    assert reward_model is not None
    states = _random_states(16, seed=3)
    np.testing.assert_allclose(
        model.goal_distances(states).numpy(),
        [reward_model.goal_distance(row.numpy()) for row in states],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        model.planar_goal_distances(states).numpy(),
        [reward_model.planar_goal_distance(row.numpy()) for row in states],
        atol=1e-9,
    )


# ── Behaviour ───────────────────────────────────────────────────────────


def test_distinct_actions_produce_distinct_goals_and_rewards() -> None:
    """This is the regression guard: the fitted linear model scored every action alike.

    Purpose: Validates that the preset action set is separated by the vectorized kernels

    Given: One shared state replicated once per preset action
    When: Each action is applied for a step and the result scored
    Then: Every predicted goal and every reward is distinct

    Test type: unit
    """
    model = _vectorized()
    state = SCHEMA.pack(
        {
            "base_lin_vel": np.zeros(3),
            "projected_gravity": [0.0, 0.0, -1.0],
            "pose_command": [2.3, 1.4, 0.0, 0.5],
        }
    )
    states = torch.as_tensor(np.tile(state, (len(PRESETS), 1)), dtype=torch.float64)
    actions = torch.arange(len(PRESETS))
    goals = model.sample_next_states(states, actions)[:, SCHEMA.slice_of("pose_command")][:, :2]
    separations = torch.cdist(goals, goals)
    assert float(separations[~torch.eye(len(PRESETS), dtype=torch.bool)].min()) > 1e-6
    rewards = model.rewards(states, actions, model.sample_next_states(states, actions))
    assert len(set(round(float(value), 9) for value in rewards)) == len(PRESETS)


def test_undriven_channels_survive_a_batched_step_unchanged() -> None:
    """Level ground is configuration, not dynamics; the vectorized path must agree with that.

    Purpose: Validates that projected_gravity is carried through the torch transition

    Given: Sixteen random states with distinct gravity blocks
    When: The batched transition takes a step
    Then: Every gravity block is bit-identical to its input

    Test type: unit
    """
    model = _vectorized()
    states = _random_states(16, seed=4)
    stepped = model.sample_next_states(states, torch.arange(16) % len(PRESETS))
    gravity = SCHEMA.slice_of("projected_gravity")
    assert torch.equal(stepped[:, gravity], states[:, gravity])


def test_sampled_headings_stay_inside_one_revolution() -> None:
    """A heading outside (-pi, pi] would break the wrapped comparisons everything downstream makes.

    Purpose: Validates that the torch transition re-wraps the heading after adding noise

    Given: States near the wrap boundary and a model with wide heading noise
    When: A batched step is taken
    Then: Every sampled heading lies in (-pi, pi]

    Test type: unit
    """
    torch.manual_seed(0)
    model = _vectorized(heading_noise_std=0.8)
    states = _random_states(512, seed=5)
    states[:, SCHEMA.slice_of("pose_command")][:, 3] = 3.0
    headings = model.sample_next_states(states, torch.zeros(512, dtype=torch.long))[:, 9]
    assert bool((headings > -np.pi).all()) and bool((headings <= np.pi).all())


def test_the_model_never_declares_a_state_terminal() -> None:
    """A model that guessed at termination would prune states the episode is able to visit.

    Purpose: Validates that termination is left to the world

    Given: Sixteen random states, including some already at the goal
    When: The terminal mask is read
    Then: Every entry is False

    Test type: unit
    """
    model = _vectorized()
    assert not bool(model.terminal_mask(_random_states(16, seed=6)).any())


def test_the_observation_likelihood_peaks_on_the_true_state() -> None:
    """A likelihood that did not peak on the truth would resample the belief away from it.

    Purpose: Validates the additive-Gaussian observation density

    Given: A batch of states and observations displaced from them by growing amounts
    When: The log-likelihood is scored
    Then: It falls monotonically as the displacement grows

    Test type: unit
    """
    model = _vectorized()
    states = _random_states(4, seed=7)
    actions = torch.zeros(4, dtype=torch.long)
    scores = [
        float(model.observation_log_probs(states, actions, states + shift)[0])
        for shift in (0.0, 0.05, 0.2, 1.0)
    ]
    assert all(later < earlier for earlier, later in zip(scores, scores[1:]))


def test_the_observation_likelihood_treats_the_heading_as_an_angle() -> None:
    """The task's heading command is uniform over a full turn, so the wrap boundary is common.

    Purpose: Regression guard -- an unwrapped heading residual scores a particle at +pi against an
        observation at -pi as two revolutions wrong and resamples away the particles that are right

    Given: A particle whose heading sits just below pi and an observation just above -pi, a
        hundredth of a radian away around the circle
    When: Both that observation and one displaced by the same hundredth on the same side are scored
    Then: The two log-likelihoods agree

    Test type: unit
    """
    model = _vectorized()
    heading_index = SCHEMA.slice_of("pose_command").stop - 1
    particle = _random_states(1, seed=9)
    particle[0, heading_index] = np.pi - 0.005
    across_the_wrap = particle.clone()
    across_the_wrap[0, heading_index] = -np.pi + 0.005
    same_side = particle.clone()
    same_side[0, heading_index] = np.pi - 0.015
    actions = torch.zeros(1, dtype=torch.long)
    assert float(model.observation_log_probs(particle, actions, across_the_wrap)[0]) == (
        pytest.approx(float(model.observation_log_probs(particle, actions, same_side)[0]), rel=1e-9)
    )


def test_sampled_observation_headings_stay_inside_one_revolution() -> None:
    """The world only ever reports a wrapped heading; a model that samples otherwise is not it.

    Purpose: Validates that the sampled observation re-wraps the heading after adding noise

    Given: States near the wrap boundary and a wide observation noise
    When: Observations are sampled
    Then: Every sampled heading lies in (-pi, pi]

    Test type: unit
    """
    torch.manual_seed(1)
    model = NavigationVectorizedModel(
        _scalar_model(), device=CPU, dtype=torch.float64, observation_noise_std=0.8
    )
    states = _random_states(512, seed=10)
    heading_index = SCHEMA.slice_of("pose_command").stop - 1
    states[:, heading_index] = 3.0
    headings = model.sample_observations(states, torch.zeros(512, dtype=torch.long))[
        :, heading_index
    ]
    assert bool((headings > -np.pi).all()) and bool((headings <= np.pi).all())


def test_observation_keys_separate_observations_that_are_grid_cells_apart() -> None:
    """Colliding keys would merge unrelated branches of the search tree without any error.

    Purpose: Validates that the observation hash distinguishes distinct quantized observations

    Given: Sixteen random observations spread wider than the quantization grid
    When: Their keys are computed
    Then: The keys are all distinct

    Test type: unit
    """
    model = _vectorized()
    keys = model.observation_keys(_random_states(16, seed=8) * 10.0)
    assert len(set(int(key) for key in keys)) == 16


# ── Configuration guards ────────────────────────────────────────────────


def test_the_vectorized_model_refuses_an_objective_it_cannot_mirror() -> None:
    """Silently vectorizing the wrong reward would make the planner optimize an unstated objective.

    Purpose: Validates the guard against a model built with a non-navigation reward

    Given: A scalar model carrying a flat custom reward model
    When: The vectorized mirror is built from it
    Then: ValueError is raised

    Test type: unit
    """

    class _Flat:
        def reward(self, state: Any, action: Any, next_state: Any) -> float:
            del state, action, next_state
            return 0.0

    with pytest.raises(ValueError, match="nothing to vectorize"):
        NavigationVectorizedModel(_scalar_model(reward_model=_Flat()), device=CPU)


@pytest.mark.parametrize(
    "overrides", [{"observation_noise_std": 0.0}, {"observation_resolution": -1.0}]
)
def test_the_vectorized_model_rejects_non_positive_observation_settings(
    overrides: dict,
) -> None:
    """A zero observation std makes every particle weight -inf, far from where it was configured.

    Purpose: Validates constructor validation of the observation arguments

    Given: A non-positive observation noise std or resolution
    When: The vectorized model is constructed
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match="must be positive"):
        NavigationVectorizedModel(_scalar_model(), device=CPU, **overrides)


def test_the_vectorized_model_rejects_presets_that_are_not_velocity_commands() -> None:
    """A mis-shaped preset table would index the wrong entries and turn where it meant to drive.

    Purpose: Validates the width check on the action preset table

    Given: A scalar model whose presets are two wide rather than three
    When: The vectorized mirror is built from it
    Then: ValueError is raised naming the expected width

    Test type: unit
    """
    with pytest.raises(ValueError, match="velocity command"):
        NavigationVectorizedModel(
            _scalar_model(action_presets=[np.zeros(2), np.ones(2)]), device=CPU
        )
