# SPDX-License-Identifier: MIT

"""Unit tests for the control-performance curve and its comparison harness.

The curve is the only thing that answers whether the fitting loop is working, so
the ways it can be silently wrong all matter. Its x-axis has to be the data
spent, or two methods are compared at different budgets. Its spread has to be
across repetitions, or it answers a question nobody asked. Its reported round
has to be the best of the sequence, because that is what the guarantee covers
and the sequence is not monotone.

The harness has one job beyond looping: refusing configurations that still
produce a plot. A baseline built with planner rollouts is a second iterative run
under another name, and unpaired seeds leave the methods' difference tangled
with the draw of start states.
"""

from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import pytest

from POMDPPlanners.training.model_learning import (
    ControlPoint,
    DAggerModelTrainer,
    LearningCurve,
    LinearGaussianLearner,
    TransitionDataset,
    aggregate_curves,
    best_point,
    curves_by_method,
    evaluate_control,
    load_learning_curves,
    run_learning_curves,
    save_learning_curves,
)

PRESETS = np.array([[1.0], [-1.0], [0.0]])


class _LinearWorld:
    """A tiny world the loop can be driven against without a simulator."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def initial_state_dist(self) -> Any:
        class _Dist:
            def sample(self, num_samples: int = 1) -> np.ndarray:
                del num_samples
                return np.zeros(2)

        return _Dist()

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        state_vector = np.asarray(state, dtype=float).ravel()
        action_vector = np.asarray(action, dtype=float).ravel()
        mean = 0.9 * state_vector + np.array([action_vector[0], 0.5 * action_vector[0]])
        draws = mean + self._rng.normal(scale=0.02, size=(n_samples, 2))
        return draws[0] if n_samples == 1 else draws

    def is_terminal(self, state: Any) -> bool:
        del state
        return False


def _planner_rollouts(model: Any, round_index: int, num_episodes: int) -> List[Any]:
    """Stand-in for running a planner against the fitted model."""
    del model, round_index
    rng = np.random.default_rng(0)
    episodes = []
    for _ in range(num_episodes):
        states = rng.normal(size=(6, 2))
        actions = PRESETS[rng.integers(len(PRESETS), size=6)]
        episodes.append((states, actions, states + 0.1))
    return episodes


def _returns_by_round(model: Any, round_index: int, num_episodes: int) -> List[float]:
    """A stand-in evaluator whose return improves with the round, deterministically."""
    del model
    return [float(round_index) + 0.1 * index for index in range(num_episodes)]


def _trainer(
    method: str,
    seed: int,
    evaluation_fn: Any = _returns_by_round,
    planner_rollout_fn: Any = _planner_rollouts,
) -> DAggerModelTrainer:
    """Build a small trainer, with the planner half switched off for ``batch``."""
    return DAggerModelTrainer(
        world=_LinearWorld(seed=seed),
        learner=LinearGaussianLearner(),
        dataset=TransitionDataset(seed=seed),
        action_presets=PRESETS,
        planner_rollout_fn=None if method == "batch" else planner_rollout_fn,
        evaluation_fn=evaluation_fn,
        evaluation_episodes=3,
        num_rounds=2,
        episodes_per_round=6,
        steps_per_episode=6,
        hold_steps=2,
        horizon=3,
        seed=seed,
    )


def test_a_point_carries_the_data_it_cost_not_just_the_round() -> None:
    """The x-axis is transitions; comparing methods by round compares budgets."""
    point = evaluate_control(
        model=object(),
        evaluation_fn=_returns_by_round,
        round_index=2,
        cumulative_transitions=140,
        num_episodes=3,
    )
    assert point.round_index == 2
    assert point.cumulative_transitions == 140
    assert point.mean_return == pytest.approx(2.1)


def test_evaluate_control_rejects_a_non_positive_episode_count() -> None:
    with pytest.raises(ValueError, match="num_episodes must be positive"):
        evaluate_control(
            model=object(),
            evaluation_fn=_returns_by_round,
            round_index=1,
            cumulative_transitions=0,
            num_episodes=0,
        )


def test_each_round_records_what_the_planner_scored_in_the_true_world() -> None:
    """The loop's own numbers describe the model; this one describes the decision."""
    trainer = _trainer("dagger", seed=0)
    rounds = trainer.run()

    assert [result.control.round_index for result in rounds] == [1, 2]
    assert all(result.control.cumulative_transitions == result.dataset_size for result in rounds)
    assert rounds[1].control.mean_return > rounds[0].control.mean_return


def test_no_evaluation_function_leaves_the_curve_empty_rather_than_nan() -> None:
    """An unevaluated run must not produce a plottable but meaningless curve."""
    trainer = _trainer("dagger", seed=0, evaluation_fn=None)
    rounds = trainer.run()

    assert all(result.control is None for result in rounds)
    assert trainer.learning_curve("dagger").points == ()


def test_the_reported_round_is_the_best_of_the_sequence_not_the_last() -> None:
    """The guarantee covers the best model in the sequence, and it is not monotone."""
    curve = LearningCurve(
        method="dagger",
        seed=0,
        points=(
            ControlPoint(1, 100, (1.0, 1.0)),
            ControlPoint(2, 200, (5.0, 5.0)),
            ControlPoint(3, 300, (2.0, 2.0)),
        ),
    )
    chosen = best_point(curve)
    assert chosen is not None
    assert chosen.round_index == 2


def test_best_point_is_none_when_nothing_was_evaluated() -> None:
    assert best_point(LearningCurve(method="dagger", seed=0, points=())) is None


def test_the_band_is_the_spread_across_seeds_not_across_episodes() -> None:
    """Episode spread says how noisy one run was; seed spread says the methods differed."""
    # Each seed is internally noisy but the two seed means are 1.0 and 3.0, so a
    # band computed across episodes would be wide and one across seeds is 1.0.
    curves = [
        LearningCurve("dagger", 0, (ControlPoint(1, 100, (-9.0, 11.0)),)),
        LearningCurve("dagger", 1, (ControlPoint(1, 100, (-7.0, 13.0)),)),
    ]
    aggregated = aggregate_curves(curves)

    assert aggregated.num_seeds == 2
    assert aggregated.mean_returns[0] == pytest.approx(2.0)
    assert aggregated.standard_errors[0] == pytest.approx(1.0)


def test_aggregation_truncates_to_the_shortest_run() -> None:
    """A run that stopped early must not silently extend the others' average."""
    curves = [
        LearningCurve("dagger", 0, (ControlPoint(1, 100, (1.0,)), ControlPoint(2, 200, (2.0,)))),
        LearningCurve("dagger", 1, (ControlPoint(1, 100, (3.0,)),)),
    ]
    assert len(aggregate_curves(curves).mean_returns) == 1


def test_aggregation_refuses_to_mix_methods() -> None:
    curves = [
        LearningCurve("dagger", 0, (ControlPoint(1, 100, (1.0,)),)),
        LearningCurve("batch", 0, (ControlPoint(1, 100, (2.0,)),)),
    ]
    with pytest.raises(ValueError, match="expects one method"):
        aggregate_curves(curves)


def test_aggregation_refuses_an_empty_set() -> None:
    with pytest.raises(ValueError, match="at least one curve"):
        aggregate_curves([])


def test_both_methods_run_on_every_seed() -> None:
    """The paper gives all approaches the same seeds; unpaired draws hide the effect."""
    seeds = [0, 1]
    curves = run_learning_curves(trainer_factory=_trainer, seeds=seeds)

    grouped = curves_by_method(curves)
    assert sorted(grouped) == ["batch", "dagger"]
    for method_curves in grouped.values():
        assert [curve.seed for curve in method_curves] == seeds
        assert all(len(curve.points) == 2 for curve in method_curves)


def test_a_baseline_built_with_planner_rollouts_is_rejected() -> None:
    """It would be a second iterative run under another name, and the gap would vanish."""

    def factory(method: str, seed: int) -> DAggerModelTrainer:
        del method
        return _trainer("dagger", seed)

    with pytest.raises(ValueError, match="must be built with planner_rollout_fn=None"):
        run_learning_curves(trainer_factory=factory, seeds=[0], methods=["batch"])


def test_a_trainer_built_on_the_wrong_seed_is_rejected() -> None:
    """Unpaired seeds leave the method difference tangled with the start-state draw."""

    def factory(method: str, seed: int) -> DAggerModelTrainer:
        del seed
        return _trainer(method, seed=99)

    with pytest.raises(ValueError, match="the comparison must be paired"):
        run_learning_curves(trainer_factory=factory, seeds=[0], methods=["dagger"])


def test_a_trainer_with_no_evaluator_is_rejected() -> None:
    """It would run the full cost of the loop and contribute nothing to the plot."""

    def factory(method: str, seed: int) -> DAggerModelTrainer:
        return _trainer(method, seed, evaluation_fn=None)

    with pytest.raises(ValueError, match="has no evaluation_fn"):
        run_learning_curves(trainer_factory=factory, seeds=[0], methods=["dagger"])


def test_run_learning_curves_needs_seeds_and_methods() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        run_learning_curves(trainer_factory=_trainer, seeds=[])
    with pytest.raises(ValueError, match="at least one method"):
        run_learning_curves(trainer_factory=_trainer, seeds=[0], methods=[])


def test_curves_survive_a_round_trip_through_json(tmp_path: Path) -> None:
    """The plot must be regenerable without paying for the episodes again."""
    curves = [
        LearningCurve("dagger", 0, (ControlPoint(1, 100, (1.0, 2.0)),)),
        LearningCurve("batch", 0, (ControlPoint(1, 100, (0.5,)),)),
    ]
    path = save_learning_curves(curves, tmp_path / "curves.json")

    assert load_learning_curves(path) == curves


def test_the_trainer_labels_its_curve_with_its_own_seed() -> None:
    """Aggregation pairs methods by seed, so a curve that forgets it cannot be paired."""
    trainer = _trainer("batch", seed=7)
    trainer.run()
    assert trainer.learning_curve("batch").seed == 7


def test_evaluation_episodes_must_be_positive_when_an_evaluator_is_given() -> None:
    with pytest.raises(ValueError, match="evaluation_episodes must be positive"):
        DAggerModelTrainer(
            world=_LinearWorld(),
            learner=LinearGaussianLearner(),
            dataset=TransitionDataset(),
            action_presets=PRESETS,
            evaluation_fn=_returns_by_round,
            evaluation_episodes=0,
        )


def test_the_plot_writes_a_file_and_skips_an_empty_comparison(tmp_path: Path) -> None:
    """Guards the one import that crosses into the visualization layer."""
    from POMDPPlanners.utils.visualization.model_learning_plots import plot_learning_curves

    curves = [
        LearningCurve("dagger", seed, (ControlPoint(1, 100, (1.0,)), ControlPoint(2, 200, (2.0,))))
        for seed in (0, 1)
    ]
    written = plot_learning_curves(curves, tmp_path / "curve.png", baseline_return=0.5)
    assert written is not None and written.exists()

    empty: Sequence[LearningCurve] = [LearningCurve("dagger", 0, ())]
    assert plot_learning_curves(empty, tmp_path / "empty.png") is None
