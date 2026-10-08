# SPDX-License-Identifier: MIT

"""Conformance suite pinning every model's batched path to its scalar one.

A model that answers both a scalar planner and a vectorized one has two
implementations of the same arithmetic, and nothing forces them to agree. They
drift the way this kind of thing always drifts: someone fixes a sign or a scale
on the path their test exercises, the other path keeps the old behaviour, and
the disagreement surfaces months later as a planner that performs differently on
GPU than on CPU -- which reads as a hardware problem and is not one.

So the rule is one test file, parameterized over every model that supports both,
asserting three things per model:

* a batch of identical rows gives the same distribution as the scalar call;
* a batch of *different* rows gives, row by row, what a Python loop gives -- this
  is the property `n_samples` cannot express and the reason the batch axis was
  added at all;
* a torch input gives a tensor whose values match the numpy answer.

Adding a model to the parameter lists below is how a new model joins the
contract. Leaving it out is how it silently escapes.
"""

from typing import Any, Tuple

import numpy as np
import pytest
import torch

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp import (
    GaussianRandomWalkTransition,
    LinearGaussianTransition,
    LinearRewardModel,
)
from POMDPPlanners.training.model_learning import (
    ProbabilisticEnsembleLearner,
    TransitionDataset,
)

DIM = 3
ACTION_DIM = 2


def _linear_transition() -> LinearGaussianTransition:
    return LinearGaussianTransition(
        weight_state=np.array([[0.9, 0.05, 0.0], [0.0, 0.92, 0.05], [0.0, 0.0, 0.95]]),
        weight_action=np.array([[0.4, 0.0], [0.1, 0.3], [0.0, 0.2]]),
        bias=np.array([0.01, 0.0, -0.01]),
        covariance=np.diag([0.01, 0.02, 0.015]),
    )


def _random_walk() -> GaussianRandomWalkTransition:
    return GaussianRandomWalkTransition(dim=DIM, process_noise_std=0.05)


def _reward() -> LinearRewardModel:
    return LinearRewardModel(
        weight_state=np.array([1.0, -0.5, 0.25]),
        weight_action=np.array([2.0, 0.5]),
        weight_next_state=np.array([0.0, 1.0, -1.0]),
        bias=0.5,
    )


def _rows(count: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(count, DIM)), rng.normal(size=(count, ACTION_DIM))


def _fitted_ensemble() -> Any:
    """A small ensemble fitted on a linear system, cached across the parameterization."""
    if _fitted_ensemble.cached is None:  # type: ignore[attr-defined]
        rng = np.random.default_rng(11)
        states = rng.normal(size=(600, DIM))
        actions = rng.normal(size=(600, ACTION_DIM))
        next_states = states * 0.9 + actions @ np.ones((ACTION_DIM, DIM)) * 0.1
        next_states = next_states + rng.normal(scale=0.03, size=(600, DIM))
        dataset = TransitionDataset(holdout_fraction=0.0, seed=0)
        dataset.add_episode(states, actions, next_states, source="exploration")
        _fitted_ensemble.cached = ProbabilisticEnsembleLearner(  # type: ignore[attr-defined]
            num_members=3, hidden_sizes=(32, 32), epochs=15, seed=0
        ).fit(dataset)
    return _fitted_ensemble.cached  # type: ignore[attr-defined]


_fitted_ensemble.cached = None  # type: ignore[attr-defined]


TRANSITIONS = [
    pytest.param(_linear_transition, id="linear_gaussian"),
    pytest.param(_random_walk, id="random_walk"),
    pytest.param(_fitted_ensemble, id="probabilistic_ensemble"),
]


@pytest.mark.parametrize("build", TRANSITIONS)
def test_batched_sampling_matches_the_scalar_distribution(build: Any) -> None:
    """Purpose: Validates that batching does not change what the transition samples

    Given: One state and action, drawn 4000 times scalar-wise and 4000 times as a
        batch of identical rows
    When: The two sample means and spreads are compared
    Then: They agree within Monte-Carlo error
    """
    model = build()
    state = np.array([0.4, -0.2, 0.1])
    action = np.array([0.5, -0.3])

    scalar = np.atleast_2d(model.sample_next_state(state, action, 4000))
    batch = model.sample_next_state(np.tile(state, (4000, 1)), np.tile(action, (4000, 1)))

    assert np.allclose(scalar.mean(axis=0), batch.mean(axis=0), atol=0.01)
    assert np.allclose(scalar.std(axis=0), batch.std(axis=0), rtol=0.1)


@pytest.mark.parametrize("build", TRANSITIONS)
def test_batched_log_density_equals_a_loop_over_rows(build: Any) -> None:
    """Purpose: Validates the row-wise pairing the batch axis exists to provide

    Given: Six *different* states, each with its own action and its own candidate
        successor
    When: The batched log-density is compared to a Python loop over the rows
    Then: They agree exactly -- this is the case n_samples cannot express, since
        it has one state by construction
    """
    model = build()
    states, actions = _rows(6, seed=1)
    candidates = states + 0.05

    batched = model.log_probability(states, actions, candidates)
    looped = np.array(
        [
            float(model.log_probability(state, action, candidate[None, :])[0])
            for state, action, candidate in zip(states, actions, candidates)
        ]
    )

    assert np.allclose(batched, looped, atol=1e-9)


@pytest.mark.parametrize("build", TRANSITIONS)
def test_a_tensor_state_returns_a_tensor_with_the_numpy_values(build: Any) -> None:
    """Purpose: Validates the backend-following rule the vectorized planner needs

    Given: The same six states as numpy and as a float32 tensor
    When: The log-density is taken both ways
    Then: A tensor comes back, on the input's device, with the numpy values --
        so a GPU rollout never has to cross to the host
    """
    model = build()
    states, actions = _rows(6, seed=2)
    candidates = states + 0.05

    numpy_result = model.log_probability(states, actions, candidates)
    tensor_result = model.log_probability(
        torch.as_tensor(states, dtype=torch.float32),
        torch.as_tensor(actions, dtype=torch.float32),
        torch.as_tensor(candidates, dtype=torch.float32),
    )

    assert isinstance(tensor_result, torch.Tensor)
    assert np.allclose(tensor_result.numpy(), numpy_result, atol=1e-3)


@pytest.mark.parametrize("build", TRANSITIONS)
def test_a_single_state_keeps_the_shapes_it_always_returned(build: Any) -> None:
    """Purpose: Validates that widening did not move the pre-existing call shapes

    Given: A single state, with n_samples of 1 and of 5
    When: Successors are sampled
    Then: The shapes are (dim,) and (n_samples, dim) exactly as before batching
    """
    model = build()
    state = np.array([0.4, -0.2, 0.1])
    action = np.array([0.5, -0.3])

    assert model.sample_next_state(state, action).shape == (DIM,)
    assert model.sample_next_state(state, action, 5).shape == (5, DIM)


@pytest.mark.parametrize("build", TRANSITIONS)
def test_a_batch_with_many_draws_nests_the_sample_axis(build: Any) -> None:
    """Purpose: Validates that the two batch axes compose instead of colliding

    Given: Four states and three draws each
    When: Successors are sampled
    Then: The result is (4, 3, dim) -- particles outermost, draws inside
    """
    model = build()
    states, actions = _rows(4, seed=3)

    assert model.sample_next_state(states, actions, 3).shape == (4, 3, DIM)


def test_batched_reward_equals_a_loop_over_rows() -> None:
    """Purpose: Validates the reward's batch path against its scalar one

    Given: Six transitions
    When: The batched reward is compared to a loop
    Then: They agree exactly, and a single transition still returns a float
    """
    model = _reward()
    states, actions = _rows(6, seed=4)
    next_states = states + 0.1

    batched = model.reward(states, actions, next_states)
    looped = np.array(
        [
            model.reward(state, action, next_state)
            for state, action, next_state in zip(states, actions, next_states)
        ]
    )

    assert np.allclose(batched, looped, atol=1e-12)
    assert isinstance(model.reward(states[0], actions[0], next_states[0]), float)


def test_reward_follows_the_state_backend() -> None:
    """Purpose: Validates that a vectorized rollout can score rewards on device

    Given: A batch of transitions as tensors
    When: The reward is taken
    Then: A tensor comes back with the numpy values, so the rollout never syncs
    """
    model = _reward()
    states, actions = _rows(6, seed=5)
    next_states = states + 0.1

    result = model.reward(
        torch.as_tensor(states, dtype=torch.float32),
        torch.as_tensor(actions, dtype=torch.float32),
        torch.as_tensor(next_states, dtype=torch.float32),
    )

    assert isinstance(result, torch.Tensor)
    assert np.allclose(result.numpy(), model.reward(states, actions, next_states), atol=1e-4)


def test_a_wrong_width_raises_instead_of_flattening() -> None:
    """Purpose: Validates that the old silent-flatten failure is now an error

    Given: A state whose trailing dimension is 4 for a 3-dimensional model
    When: A successor is sampled
    Then: A ValueError names the expected width. Before batching, reshape(-1)
        accepted this and returned confident nonsense
    """
    model = _linear_transition()

    with pytest.raises(ValueError, match="trailing dimension 3"):
        model.sample_next_state(np.zeros((5, 4)), np.zeros((5, ACTION_DIM)))


def test_the_ensemble_draws_a_member_per_row_not_per_batch() -> None:
    """Purpose: Validates that batching preserves the ensemble's epistemic spread

    Given: A fitted ensemble and one state repeated across 4000 rows
    When: One successor is drawn per row
    Then: The spread across rows matches the spread of 4000 scalar draws. Drawing
        a single member for the whole batch would collapse it to one member's
        opinion, which is the spread a risk-sensitive planner is there to price
    """
    model = _fitted_ensemble()
    state = np.array([0.3, -0.4, 0.2])
    action = np.array([0.5, -0.1])

    scalar = np.atleast_2d(model.sample_next_state(state, action, 4000))
    batch = model.sample_next_state(np.tile(state, (4000, 1)), np.tile(action, (4000, 1)))

    assert np.allclose(scalar.std(axis=0), batch.std(axis=0), rtol=0.15)
