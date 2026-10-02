# SPDX-License-Identifier: MIT

"""Tests for the tuning study's diagnostic charts.

Built from an in-memory study, because the charts read nothing but the
records: what is tested is that each chart draws what its records say.
"""

import re

from POMDPPlanners.reporting import tuning, tuning_charts


def _trial(number, ret, rate, depth, rollout, pareto=False, state="COMPLETE"):
    return tuning.Trial(
        number=number,
        state=state,
        params={"depth": depth, "rollout": rollout},
        objective_values={"average_return": ret, "collision_rate": rate},
        metric_statistics={
            "average_return": [ret, ret - 1.0, ret + 1.0],
            "collision_rate": [rate, rate, rate],
            "average_steps": [10.0 + number, 9.0 + number, 11.0 + number],
        },
        duration_seconds=2.0 + number,
        is_pareto=pareto,
    )


def _study(**overrides):
    fields = dict(
        planner="PFT_DPW",
        environment="LightDark",
        environment_name="LightDark",
        policy_name="PFT_DPW",
        objectives=[("average_return", "maximize"), ("collision_rate", "minimize")],
        parameters=[
            tuning.SearchParameter(name="depth", best=8, low=2, high=10),
            tuning.SearchParameter(name="rollout", best="greedy", choices=["random", "greedy"]),
        ],
        best_trial_number=2,
        best_trial_scores={},
        pareto_trial_numbers=[1, 2],
        stopped_at_trial=4,
        front_quality_history=[(2, 0.5), (3, 0.9), (4, 0.9)],
        trials=[
            _trial(0, 1.0, 0.5, 3, "random"),
            _trial(1, 4.0, 0.1, 6, "greedy", pareto=True),
            _trial(2, 9.0, 0.3, 8, "greedy", pareto=True),
            _trial(3, 2.0, 0.6, 4, "random"),
            tuning.Trial(4, "FAIL", {"depth": 9}, {}, {}, 1.0, False),
        ],
    )
    fields.update(overrides)
    return tuning.TuningStudy(**fields)


def _tips(svg):
    return re.findall(r"<title>([^<]*)</title>", svg)


def test_objective_history_draws_each_completed_trial_and_the_best_so_far():
    """One chart per objective; the best-so-far curve follows each direction.

    Given: Four completed trials and one failed, return maximized and
        collision rate minimized.
    When: The objective history is drawn.
    Then: Two charts, four marks each, the chosen trial marked as such, and
        the curves step to 9.0 for return and down to 0.1 for collisions.
    """
    charts = tuning_charts.objective_history(_study())

    assert len(charts) == 2
    returns, collisions = charts
    assert len(_tips(returns)) == 4
    assert 'class="pt pt-best"' in returns
    assert "early stop" in returns
    assert "Trial #2 · average_return 9" in returns
    # The step curve has a vertex for each trial plus one per step.
    assert returns.count("<polyline") == 1
    assert "best so far" in collisions
    assert "(minimized)" in collisions


def test_hover_tips_name_the_trial_and_its_parameters():
    """Every mark says which trial it is, which a PNG cannot."""
    svg = tuning_charts.objective_history(_study())[0]

    assert "Trial #1 · average_return 4 · depth=6, rollout=greedy" in _tips(svg)


def test_ranked_intervals_put_the_best_first_with_whiskers():
    """Trials ranked by value carry their interval as a whisker."""
    svg = tuning_charts.objective_confidence_intervals(_study())[0]
    tips = _tips(svg)

    assert tips[-1].startswith("Trial #2 · rank 1 · average_return 9 [8, 10]")
    assert svg.count('class="chart-ci"') == 4


def test_pareto_front_needs_exactly_two_objectives():
    """The front is drawn for two objectives and skipped otherwise."""
    two = tuning_charts.pareto_front(_study())
    one = tuning_charts.pareto_front(_study(objectives=[("average_return", "maximize")]))

    assert "Pareto front" in two and "chart-front" in two
    assert len(_tips(two)) == 4
    assert one == ""


def test_front_quality_draws_the_logged_curve_and_the_stop():
    """The early-stopping curve comes from the logged history, with the stop marked."""
    svg = tuning_charts.front_quality(_study())

    assert len(_tips(svg)) == 3
    assert "stopped" in svg
    assert tuning_charts.front_quality(_study(front_quality_history=[])) == ""


def test_parameters_are_charted_over_trials_and_against_each_objective():
    """A numeric parameter spans its search range; a categorical one names its choices.

    Given: depth searched over 2-10, rollout over two choices.
    When: Parameter history and slices are drawn.
    Then: Two history charts (failed trials included, since they were
        sampled too), four slices, the depth axis reaching both ends of its
        range, and the rollout axis labelled with its choices.
    """
    history = tuning_charts.parameter_history(_study())
    slices = tuning_charts.parameter_slices(_study())

    assert len(history) == 2 and len(slices) == 4
    assert len(_tips(history[0])) == 5
    assert ">random<" in history[1] and ">greedy<" in history[1]
    depth_slice = slices[0]
    # The sampled depths run 3-8, but the axis spans the 2-10 range searched
    # (with its 8% margin), so the gap to each edge shows.
    assert ">1.36<" in depth_slice and ">10.6<" in depth_slice
    assert ">random<" in slices[1]


def test_secondary_metrics_are_the_recorded_ones_not_optimized():
    """Only metrics outside the objectives get a secondary chart."""
    charts = tuning_charts.secondary_metrics(_study())

    assert len(charts) == 1
    assert "average_steps" in charts[0]
    assert charts[0].count('class="chart-ci"') == 4


def test_trial_durations_include_failed_trials():
    """A failed trial still spent its time, so it is on the duration chart."""
    svg = tuning_charts.trial_durations(_study())

    assert len(_tips(svg)) == 5
    assert "Fail" in svg


def test_labels_from_user_code_are_escaped():
    """Parameter values come from user code and reach the page escaped."""
    study = _study(
        trials=[_trial(0, 1.0, 0.1, 3, "<script>")],
        parameters=[tuning.SearchParameter(name="rollout")],
    )
    svg = "".join(tuning_charts.parameter_history(study))

    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg


def test_no_trials_draws_nothing():
    """A study without records yields no charts rather than empty frames."""
    study = _study(trials=[])

    assert tuning_charts.objective_history(study) == []
    assert tuning_charts.trial_durations(study) == ""


def test_tiny_values_are_scaled_and_ticks_never_repeat():
    """Axis labels stay short and distinct on tiny or narrow ranges.

    Purpose: A belief-update time of 6e-05 printed in full overran the margin,
    and a range of 1.025 to 1.035 printed "1.03" twice.

    Given: Tick values of order 1e-5, and ticks on a narrow range.
    When: Their labels are made.
    Then: The tiny ones are divided by the power of ten the axis title names,
        and the narrow ones get enough digits to differ.
    """
    # pylint: disable-next=protected-access
    tiny, suffix = tuning_charts._tick_labels([2e-5, 4e-5, 6e-5])
    # pylint: disable-next=protected-access
    narrow, plain = tuning_charts._tick_labels([1.025, 1.0275, 1.03, 1.0325, 1.035])

    assert tiny == ["2", "4", "6"] and suffix == " (×1e-5)"
    assert len(set(narrow)) == 5 and plain == ""


def test_the_early_stop_marker_sits_on_the_trial_that_completed_the_count():
    """A failed trial before the stop moves the marker to the right trial number.

    Given: Trials #0 and #2 completed and #1 failed, and early stopping fired
        after 2 completed trials.
    When: The marker's position is computed.
    Then: It is trial #2, not #1.
    """
    trials = [
        _trial(0, 1.0, 0.5, 3, "random"),
        tuning.Trial(1, "FAIL", {"depth": 4}, {}, {}, 1.0, False),
        _trial(2, 4.0, 0.1, 6, "greedy"),
    ]
    # pylint: disable-next=protected-access
    assert tuning_charts._stop_trial_number(_study(trials=trials, stopped_at_trial=2)) == 2.0
    # pylint: disable-next=protected-access
    assert tuning_charts._stop_trial_number(_study(trials=trials, stopped_at_trial=None)) is None
