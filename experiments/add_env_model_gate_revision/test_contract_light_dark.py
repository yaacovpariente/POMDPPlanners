# SPDX-License-Identifier: MIT

"""The batch-parity assertions of ``tests/test_core/test_batched_model_parity.py``,
run over the three LightDark candidates at their own width, plus the
substituted-transition conformance helpers on the wired model.

Run with ``PYTHONPATH=. .venv/bin/pytest experiments/add_env_model_gate_revision/test_contract_light_dark.py``.
"""

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pytest
import torch

from experiments.add_env_model_gate_revision.substituted_transition import (
    assert_carried_channels_preserved,
    assert_observation_matches_reference,
    assert_transition_is_used_everywhere,
)

from experiments.add_env_model_gate_revision import world as W
from experiments.add_env_model_gate_revision.light_dark_model import SubstitutedLightDarkModel
from experiments.add_env_model_gate_revision.models import (
    GaussianMLPObservation,
    GaussianMLPTransition,
    bad_transition,
    truth_transition,
)

DIM = 2
ACTION_DIM = 2
LEARNED_DIR = Path("results/add-env-model-gate-revision/learned_model")
WORLD = W.make_world()


def _truth() -> Any:
    return truth_transition(WORLD.state_transition_cov_matrix)


def _bad() -> Any:
    return bad_transition(WORLD.state_transition_cov_matrix)


def _learned() -> Any:
    path = LEARNED_DIR / "transition.npz"
    if not path.exists():
        pytest.skip("learned transition not fitted yet")
    return GaussianMLPTransition.load(path)


def _rows(count: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 10.0, size=(count, DIM)), rng.normal(size=(count, ACTION_DIM))


TRANSITIONS = [
    pytest.param(_truth, id="truth"),
    pytest.param(_bad, id="bad_std_x2"),
    pytest.param(_learned, id="learned_mlp"),
]
STATE = np.array([4.0, 5.0])
ACTION = np.array([0.5, -0.3])


@pytest.mark.parametrize("build", TRANSITIONS)
def test_batched_sampling_matches_the_scalar_distribution(build: Any) -> None:
    model = build()
    scalar = np.atleast_2d(model.sample_next_state(STATE, ACTION, 4000))
    batch = model.sample_next_state(np.tile(STATE, (4000, 1)), np.tile(ACTION, (4000, 1)))
    assert np.allclose(scalar.mean(axis=0), batch.mean(axis=0), atol=0.02)
    assert np.allclose(scalar.std(axis=0), batch.std(axis=0), rtol=0.1)


@pytest.mark.parametrize("build", TRANSITIONS)
def test_batched_log_density_equals_a_loop_over_rows(build: Any) -> None:
    model = build()
    states, actions = _rows(6, seed=1)
    candidates = states + 0.05
    batched = model.log_probability(states, actions, candidates)
    looped = np.array(
        [float(model.log_probability(s, a, c[None, :])[0]) for s, a, c in zip(states, actions, candidates)]
    )
    assert np.allclose(batched, looped, atol=1e-9)


@pytest.mark.parametrize("build", TRANSITIONS)
def test_a_tensor_state_returns_a_tensor_with_the_numpy_values(build: Any) -> None:
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
    model = build()
    assert model.sample_next_state(STATE, ACTION).shape == (DIM,)
    assert model.sample_next_state(STATE, ACTION, 5).shape == (5, DIM)


@pytest.mark.parametrize("build", TRANSITIONS)
def test_a_batch_with_many_draws_nests_the_sample_axis(build: Any) -> None:
    model = build()
    states, actions = _rows(4, seed=3)
    assert model.sample_next_state(states, actions, 3).shape == (4, 3, DIM)


def _wired(transition: Any) -> SubstitutedLightDarkModel:
    return SubstitutedLightDarkModel(
        transition=transition, observation=W.true_observation(WORLD), model_label="probe",
        discount_factor=W.DISCOUNT, **W.world_kwargs(),
    )


FULL_STATE = np.array([0.0, 5.0])
#: The world has no carried channel: the whole state is the driven block. The
#: helper is run with an empty index set so its applicability is on record.
CARRIED_INDICES: list = []


def test_every_dynamics_path_consults_the_substitute() -> None:
    assert_transition_is_used_everywhere(build_model=_wired, state=FULL_STATE, action=ACTION, dim=DIM)


@pytest.mark.parametrize("build", TRANSITIONS)
def test_no_channel_is_carried_so_the_helper_is_degenerate(build: Any) -> None:
    model = _wired(build())
    assert_carried_channels_preserved(model, FULL_STATE, ACTION, carried_indices=CARRIED_INDICES)
    assert model.sample_next_state(FULL_STATE, ACTION).shape == (DIM,)


@pytest.mark.parametrize("build", TRANSITIONS)
def test_the_observation_is_no_wider_than_the_task(build: Any) -> None:
    assert_observation_matches_reference(_wired(build()), WORLD, FULL_STATE, action=ACTION)


def test_learned_observation_is_no_wider_than_the_task() -> None:
    path = LEARNED_DIR / "observation.npz"
    if not path.exists():
        pytest.skip("learned observation not fitted yet")
    model = SubstitutedLightDarkModel(
        transition=_truth(), observation=GaussianMLPObservation.load(path), model_label="probe",
        discount_factor=W.DISCOUNT, **W.world_kwargs(),
    )
    assert_observation_matches_reference(model, WORLD, FULL_STATE, action=ACTION)
