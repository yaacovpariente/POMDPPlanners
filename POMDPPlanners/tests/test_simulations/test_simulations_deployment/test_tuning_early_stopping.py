# SPDX-License-Identifier: MIT

import optuna
import pytest

from POMDPPlanners.simulations.simulations_deployment.tuning_early_stopping import (
    EarlyStoppingCallback,
    EarlyStoppingConfig,
    build_early_stopping_callback,
    hypervolume,
    non_dominated,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class TestHypervolume:
    def test_single_point_is_its_own_box(self):
        assert hypervolume([(0.5, 0.4)]) == pytest.approx(0.2)

    def test_dominated_points_add_nothing(self):
        assert hypervolume([(0.5, 0.4), (0.2, 0.1)]) == pytest.approx(0.2)

    def test_union_of_boxes_is_not_double_counted(self):
        # Two incomparable points overlap on the [0, 0.3] x [0, 0.3] square.
        assert hypervolume([(1.0, 0.3), (0.3, 1.0)]) == pytest.approx(0.3 + 0.3 - 0.09)

    def test_three_objectives(self):
        assert hypervolume([(1.0, 1.0, 1.0)]) == pytest.approx(1.0)
        assert hypervolume([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]) == pytest.approx(0.0)

    def test_one_objective_is_the_best_value(self):
        assert hypervolume([(0.2,), (0.7,), (0.5,)]) == pytest.approx(0.7)

    def test_empty_front(self):
        assert hypervolume([]) == 0.0

    def test_non_dominated_drops_dominated_points(self):
        front = non_dominated([(1.0, 0.0), (0.0, 1.0), (0.4, 0.4), (0.5, 0.5)])
        assert (0.4, 0.4) not in front
        assert (0.5, 0.5) in front


class TestEarlyStoppingConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"patience": 0},
            {"min_trials": -1},
            {"min_relative_improvement": -0.1},
        ],
    )
    def test_rejects_invalid_settings(self, kwargs):
        with pytest.raises(ValueError):
            EarlyStoppingConfig(**kwargs)

    def test_to_dict_is_config_id_friendly(self):
        assert EarlyStoppingConfig(patience=3, min_trials=2).to_dict() == {
            "patience": 3,
            "min_trials": 2,
            "min_relative_improvement": 1e-3,
        }


class TestEarlyStoppingCallback:
    def test_stops_well_before_the_trial_budget(self):
        config = EarlyStoppingConfig(patience=5, min_trials=5, min_relative_improvement=1e-3)
        study = optuna.create_study(directions=["maximize", "minimize"])
        callback = EarlyStoppingCallback(config=config, directions=study.directions)
        study.optimize(lambda t: (1.0, 0.5), n_trials=200, callbacks=[callback])

        assert callback.stopped_at_trial is not None
        assert len(study.trials) < 200

    def test_does_not_stop_before_min_trials(self):
        config = EarlyStoppingConfig(patience=1, min_trials=20)
        study = optuna.create_study(directions=["maximize", "minimize"])
        callback = EarlyStoppingCallback(config=config, directions=study.directions)
        study.optimize(lambda t: (1.0, 0.5), n_trials=200, callbacks=[callback])

        assert len(study.trials) >= 20

    def test_keeps_running_while_the_front_improves(self):
        config = EarlyStoppingConfig(patience=5, min_trials=5)
        study = optuna.create_study(directions=["maximize", "minimize"])
        callback = EarlyStoppingCallback(config=config, directions=study.directions)

        counter = {"n": 0}

        def improving(trial):
            counter["n"] += 1
            return float(counter["n"]), -float(counter["n"])

        study.optimize(improving, n_trials=30, callbacks=[callback])

        assert callback.stopped_at_trial is None
        assert len(study.trials) == 30

    def test_history_is_monotone_and_recorded(self):
        config = EarlyStoppingConfig(patience=4, min_trials=6)
        study = optuna.create_study(directions=["maximize", "minimize"])
        callback = EarlyStoppingCallback(config=config, directions=study.directions)
        study.optimize(
            lambda t: (t.suggest_float("x", 0.0, 1.0), t.suggest_float("y", 0.0, 1.0)),
            n_trials=60,
            callbacks=[callback],
        )

        assert callback.history
        qualities = [quality for _, quality in callback.history]
        assert all(later >= earlier - 1e-12 for earlier, later in zip(qualities, qualities[1:]))

    def test_minimized_objectives_are_flipped(self):
        study = optuna.create_study(directions=["minimize"])
        callback = EarlyStoppingCallback(
            config=EarlyStoppingConfig(patience=2, min_trials=2), directions=study.directions
        )
        # Freeze the bounds on the observed range, then check the ordering:
        # lower is better, so 0.0 must score above 1.0.
        assert callback.front_quality([[0.0], [1.0]]) == pytest.approx(1.0)
        assert callback.front_quality([[0.0]]) == pytest.approx(1.0)
        assert callback.front_quality([[1.0]]) == pytest.approx(0.0)

    def test_build_returns_none_when_disabled(self):
        assert build_early_stopping_callback(None, []) is None
        assert isinstance(
            build_early_stopping_callback(EarlyStoppingConfig(), []), EarlyStoppingCallback
        )
