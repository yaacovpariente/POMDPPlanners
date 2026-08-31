# SPDX-License-Identifier: MIT

"""Unit tests for the round loop and the exploration collector.

Three properties carry the loop's guarantee, and each fails quietly. The halves
must stay balanced, or the exploration rows shrink to a fraction the fit stops
paying for. The dataset must aggregate, or the loop becomes unrelated fits that
can oscillate. Every round's model must be kept, because the guarantee is on the
best of the sequence and not on the last one.

The collector has its own two: an action held across several steps, because a
command redrawn every step measures a transient rather than the system, and no
transition recorded across an episode boundary, because the simulator resets
inside its own step and that successor belongs to a different episode.
"""

from typing import Any, List, Tuple

import numpy as np
import pytest

from POMDPPlanners.training.model_learning import (
    DAggerModelTrainer,
    LinearGaussianLearner,
    TransitionDataset,
    collect_random_preset_episode,
)

PRESETS = np.array([[1.0], [-1.0], [0.0]])


class _LinearWorld:
    """A tiny deterministic-plus-noise world with a controllable terminal rule."""

    def __init__(self, terminal_after: int = 10**6, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._terminal_after = terminal_after
        self._steps = 0

    def restart(self) -> np.ndarray:
        """Reset the step counter and return the start state."""
        self._steps = 0
        return np.zeros(2)

    def initial_state_dist(self) -> Any:
        world = self

        class _Dist:
            def sample(self, num_samples: int = 1) -> np.ndarray:
                del num_samples
                return world.restart()

        return _Dist()

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        self._steps += 1
        state_vector = np.asarray(state, dtype=float).ravel()
        action_vector = np.asarray(action, dtype=float).ravel()
        mean = 0.9 * state_vector + np.array([action_vector[0], 0.5 * action_vector[0]])
        draws = mean + self._rng.normal(scale=0.02, size=(n_samples, 2))
        return draws[0] if n_samples == 1 else draws

    def is_terminal(self, state: Any) -> bool:
        del state
        return self._steps >= self._terminal_after


def _fake_planner_rollouts(model: Any, round_index: int, num_episodes: int) -> List[Tuple]:
    """Stand-in for running a planner: episodes marked so their source is checkable."""
    del model, round_index
    rng = np.random.default_rng(0)
    episodes = []
    for _ in range(num_episodes):
        states = rng.normal(size=(8, 2)) + 100.0
        actions = PRESETS[rng.integers(len(PRESETS), size=8)]
        episodes.append((states, actions, states + 0.1))
    return episodes


def test_each_round_takes_half_its_episodes_from_exploration() -> None:
    """Purpose: Validates the fixed exploration/planner balance the loop depends on

    Given: A round of 10 episodes with a planner rollout function supplied
    When: Two rounds run
    Then: Round 2 contributes 5 episodes from each source, so the exploration data
        does not shrink as a fraction of the buffer. Round 1 has no model to plan
        with, so its whole budget goes to exploration rather than being halved.
    """
    trainer = DAggerModelTrainer(
        world=_LinearWorld(),
        learner=LinearGaussianLearner(),
        dataset=TransitionDataset(holdout_fraction=0.0),
        action_presets=PRESETS,
        planner_rollout_fn=_fake_planner_rollouts,
        initial_model=None,
        num_rounds=2,
        episodes_per_round=10,
        steps_per_episode=8,
    )

    rounds = trainer.run()

    # Round 1 has no model to plan with, so all 10 episodes explore; round 2 adds
    # five of each, which is the balance the loop must hold from then on.
    assert rounds[0].source_counts == {"exploration": 80}
    assert rounds[1].source_counts["planner"] == 40
    assert rounds[1].source_counts["exploration"] == 120


def test_the_dataset_grows_across_rounds_instead_of_being_replaced() -> None:
    """Purpose: Validates that round n refits on rounds 1..n, not on round n alone

    Given: Three rounds of 4 episodes each
    When: The loop runs
    Then: Each round's fit saw strictly more transitions than the round before
    """
    trainer = DAggerModelTrainer(
        world=_LinearWorld(),
        learner=LinearGaussianLearner(),
        dataset=TransitionDataset(holdout_fraction=0.0),
        action_presets=PRESETS,
        planner_rollout_fn=_fake_planner_rollouts,
        num_rounds=3,
        episodes_per_round=4,
        steps_per_episode=8,
    )

    sizes = [result.dataset_size for result in trainer.run()]

    assert sizes == sorted(sizes)
    assert len(set(sizes)) == 3


def test_every_round_model_is_kept() -> None:
    """Purpose: Validates that the loop returns the sequence, not just the last model

    Given: Three rounds
    When: The loop runs
    Then: Three distinct models come back, because the guarantee is on the best of
        the sequence rather than on the final one
    """
    trainer = DAggerModelTrainer(
        world=_LinearWorld(),
        learner=LinearGaussianLearner(),
        dataset=TransitionDataset(holdout_fraction=0.0),
        action_presets=PRESETS,
        planner_rollout_fn=None,
        num_rounds=3,
        episodes_per_round=4,
        steps_per_episode=8,
    )

    rounds = trainer.run()

    assert [result.round_index for result in rounds] == [1, 2, 3]
    assert len({id(result.model) for result in rounds}) == 3


def test_diagnostics_are_reported_per_round() -> None:
    """Purpose: Validates that each round carries the numbers used to judge its model

    Given: A loop with a holdout split
    When: Two rounds run
    Then: Each round reports a finite held-out likelihood and a drift ratio
    """
    trainer = DAggerModelTrainer(
        world=_LinearWorld(),
        learner=LinearGaussianLearner(),
        dataset=TransitionDataset(holdout_fraction=0.5, seed=2),
        action_presets=PRESETS,
        planner_rollout_fn=None,
        num_rounds=2,
        episodes_per_round=8,
        steps_per_episode=8,
        horizon=5,
    )

    rounds = trainer.run()

    for result in rounds:
        assert np.isfinite(result.diagnostics["held_out_log_likelihood"])
        assert np.isfinite(result.diagnostics["horizon_drift_ratio"])


def test_exploration_holds_each_action_for_several_steps() -> None:
    """Purpose: Validates that a drawn command is held, not redrawn every step

    Given: A 12-step exploration rollout with a hold of 4
    When: The recorded actions are inspected
    Then: They form runs of 4 identical commands, so the rollout excites the
        system at the timescale it responds on
    """
    _, actions, _ = collect_random_preset_episode(
        world=_LinearWorld(),
        action_presets=PRESETS,
        num_steps=12,
        rng=np.random.default_rng(0),
        hold_steps=4,
    )

    for start in range(0, 12, 4):
        block = actions[start : start + 4]
        assert np.all(block == block[0])


def test_no_transition_is_recorded_across_an_episode_boundary() -> None:
    """Purpose: Validates that the row spanning a reset is dropped, not fitted

    Given: A world that terminates after 3 steps, and a 10-step rollout
    When: The rollout is collected
    Then: Two transitions come back -- the third ends the episode and its
        successor is a fresh episode, which is not a transition of the system
    """
    states, _, _ = collect_random_preset_episode(
        world=_LinearWorld(terminal_after=3),
        action_presets=PRESETS,
        num_steps=10,
        rng=np.random.default_rng(0),
        hold_steps=2,
    )

    assert len(states) == 2


def test_a_non_positive_hold_is_rejected() -> None:
    """Purpose: Validates that a zero hold is refused rather than looping forever

    Given: hold_steps of 0
    When: A rollout is collected
    Then: A ValueError is raised
    """
    with pytest.raises(ValueError, match="hold_steps"):
        collect_random_preset_episode(
            world=_LinearWorld(),
            action_presets=PRESETS,
            num_steps=4,
            rng=np.random.default_rng(0),
            hold_steps=0,
        )


@pytest.mark.parametrize("field,value", [("num_rounds", 0), ("episodes_per_round", 0)])
def test_a_loop_that_would_collect_nothing_is_rejected(field: str, value: int) -> None:
    """Purpose: Validates that a degenerate loop configuration raises at construction

    Given: Zero rounds, or zero episodes per round
    When: The trainer is constructed
    Then: A ValueError names the offending field
    """
    kwargs: dict = {
        "world": _LinearWorld(),
        "learner": LinearGaussianLearner(),
        "dataset": TransitionDataset(),
        "action_presets": PRESETS,
        field: value,
    }
    with pytest.raises(ValueError, match=field):
        DAggerModelTrainer(**kwargs)
