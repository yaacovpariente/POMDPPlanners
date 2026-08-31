# SPDX-License-Identifier: MIT

"""Unit tests for the accumulating transition dataset.

Two properties matter and both fail silently when broken. Aggregation: the
iterative loop's guarantee rests on refitting over every round, so a dataset
that quietly drops the oldest round turns the loop into a sequence of unrelated
fits. Confinement: a fit that reaches a carried latent channel flattens the
belief dispersion a risk-sensitive planner grades, and returns a plausible
number while doing it.
"""

from typing import Any

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation.history import History, StepData
from POMDPPlanners.training.model_learning import TransitionDataset, block_indices


def _rows(count: int, width: int, offset: float = 0.0) -> tuple:
    rng = np.random.default_rng(int(offset) + 7)
    states = rng.normal(size=(count, width)) + offset
    actions = rng.normal(size=(count, 2))
    next_states = states + 0.1 * rng.normal(size=(count, width))
    return states, actions, next_states


def _step(state: Any, action: Any, next_state: Any) -> StepData:
    return StepData(
        state=state,
        action=action,
        next_state=next_state,
        observation=None,
        reward=0.0,
        belief=WeightedParticleBelief([np.zeros(3), np.ones(3)], np.array([-0.5, -1.0])),
    )


def test_rounds_aggregate_rather_than_replace() -> None:
    """Purpose: Validates that a later round adds to the dataset instead of replacing it

    Given: A dataset that has taken one round of 50 transitions
    When: A second round of 50 is added
    Then: The dataset holds 100 rows and both rounds' sources are counted
    """
    dataset = TransitionDataset(holdout_fraction=0.0, seed=0)
    dataset.add_episode(*_rows(50, 3, offset=0.0), source="exploration")
    assert len(dataset) == 50

    dataset.add_episode(*_rows(50, 3, offset=10.0), source="planner")

    assert len(dataset) == 100
    assert dataset.counts_by_source() == {"exploration": 50, "planner": 50}
    # The first round's rows are still reachable, not merely counted.
    assert float(dataset.training_batch().states.min()) < 5.0


def test_fit_never_sees_channels_outside_the_named_block() -> None:
    """Purpose: Validates that a carried latent channel is sliced out before any fit

    Given: A 6-wide state whose last two columns are a latent type, and a dataset
        restricted to the 4-wide robot block
    When: The training batch is taken
    Then: It is 4 columns wide, and the latent values are absent
    """
    states = np.hstack([np.ones((20, 4)), np.full((20, 2), 99.0)])
    next_states = states + 0.01
    actions = np.zeros((20, 2))
    dataset = TransitionDataset(state_indices=np.arange(4), holdout_fraction=0.0)

    dataset.add_episode(states, actions, next_states, source="exploration")

    batch = dataset.training_batch()
    assert batch.states.shape == (20, 4)
    assert not np.any(np.isclose(batch.states, 99.0))
    assert not np.any(np.isclose(batch.next_states, 99.0))


def test_block_indices_reads_channels_from_a_schema() -> None:
    """Purpose: Validates that named channels resolve to the right flat indices

    Given: A schema of a 4-wide robot block followed by a 2-wide hazard type
    When: The robot channel's indices are requested
    Then: They are the first four positions
    """

    class _Schema:
        def indices_of(self, names: Any) -> np.ndarray:
            del names
            return np.arange(4)

    assert block_indices(_Schema(), ("robot",)).tolist() == [0, 1, 2, 3]


def test_holdout_is_split_by_episode_not_by_row() -> None:
    """Purpose: Validates that a held-out episode contributes no training rows

    Given: Twenty episodes added with a 0.5 holdout fraction
    When: The two splits are taken
    Then: They partition the rows exactly, and each split's size is a multiple of
        the episode length, which a row-wise split would not guarantee
    """
    dataset = TransitionDataset(holdout_fraction=0.5, seed=3)
    for index in range(20):
        dataset.add_episode(*_rows(10, 3, offset=float(index)), source="exploration")

    train = dataset.training_batch()
    holdout = dataset.holdout_batch()

    assert len(train) + len(holdout) == 200
    assert len(train) % 10 == 0
    assert len(holdout) % 10 == 0
    assert len(holdout) > 0


def test_terminal_bookkeeping_step_contributes_no_transition() -> None:
    """Purpose: Validates that the actionless terminal step is dropped

    Given: A history of two real steps followed by the terminal bookkeeping step
    When: The history is added
    Then: Two transitions are stored, not three
    """
    steps = [
        _step(np.zeros(3), np.zeros(2), np.ones(3)),
        _step(np.ones(3), np.zeros(2), 2 * np.ones(3)),
        _step(2 * np.ones(3), None, None),
    ]
    history = History(
        history=steps,
        discount_factor=0.99,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=len(steps),
        reach_terminal_state=True,
        policy_run_data=[],
    )
    dataset = TransitionDataset(holdout_fraction=0.0)

    added = dataset.add_history(history, source="planner")

    assert added == 2
    assert len(dataset) == 2


def test_mismatched_lengths_are_rejected() -> None:
    """Purpose: Validates that a truncated action array is caught rather than zipped short

    Given: 10 states, 9 actions and 10 next states
    When: They are added
    Then: A ValueError names the disagreeing lengths
    """
    dataset = TransitionDataset()
    with pytest.raises(ValueError, match="equal length"):
        dataset.add_episode(np.zeros((10, 3)), np.zeros((9, 2)), np.zeros((10, 3)), source="x")


@pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
def test_impossible_holdout_fractions_are_rejected(fraction: float) -> None:
    """Purpose: Validates that a holdout fraction leaving no training data is refused

    Given: A holdout fraction outside [0, 1)
    When: A dataset is constructed
    Then: A ValueError is raised
    """
    with pytest.raises(ValueError, match="holdout_fraction"):
        TransitionDataset(holdout_fraction=fraction)
