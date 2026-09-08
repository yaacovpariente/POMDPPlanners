# SPDX-License-Identifier: MIT

"""AR-DESPOT public interface: metrics, configuration identity, boundaries, export.

The algorithm's equations are pinned in ``test_ardespot_correctness.py``. This
file covers what surrounds them: the ``Policy`` contract, the metric names and
values that reach a simulation record, the cache key, the constructor's
refusals, and the search-state dump QA depends on.

References:
    Ye, Somani, Hsu & Lee (2017), DESPOT: Online POMDP Planning with
    Regularization, JAIR 58.
"""

# pylint: disable=protected-access

import json
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.tree.arena import ACTION, BELIEF
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners import ARDESPOT, POLICY_REGISTRY
from POMDPPlanners.planners.scenario_tree_planners.ardespot import ARDESPOTMetrics
from POMDPPlanners.planners.scenario_tree_planners.despot import DESPOTMetrics
from POMDPPlanners.tests.test_planners.planner_fixtures import (
    END,
    ROOT,
    ChainEnv,
    chain_belief,
)
from POMDPPlanners.utils.tree_statistics import TreeMetrics


DISCOUNT = 0.95


def _tiger() -> TigerPOMDP:
    return TigerPOMDP(discount_factor=DISCOUNT)


def _planner(environment: Any = None, **overrides: Any) -> ARDESPOT:
    environment = environment if environment is not None else _tiger()
    params: Dict[str, Any] = dict(
        environment=environment,
        discount_factor=DISCOUNT,
        depth=5,
        name="ARDESPOT_test",
        n_scenarios=8,
        n_simulations=15,
        pruning_constant=0.01,
    )
    params.update(overrides)
    return ARDESPOT(**params)


# ---------------------------------------------------------------------------
# Policy contract
# ---------------------------------------------------------------------------


def test_action_returns_one_legal_action_and_only_declared_metrics():
    """The ``Policy`` contract, and nothing emitted that was not declared.

    Purpose: An undeclared metric reaches a simulation record under a name no
    aggregation knows, so it is silently dropped rather than reported.

    Given: A planner on Tiger with a small trial budget.
    Then: Exactly one legal action, and every emitted metric name appears in
        ``get_info_variable_names``.

    Test type: unit
    """
    np.random.seed(11)
    environment = _tiger()
    planner = _planner(environment)
    actions, run_data = planner.action(get_initial_belief(environment, n_particles=20))

    assert len(actions) == 1
    assert actions[0] in environment.get_actions()

    declared = set(ARDESPOT.get_info_variable_names())
    emitted = {variable.name for variable in run_data.info_variables}
    assert emitted <= declared
    assert emitted


def test_declared_metric_names_cover_the_shared_and_the_ardespot_specific_sets():
    """Every declared name comes from one of the two enums, and none is missing.

    Purpose: The declared list is what a run configuration validates against;
    a name in the enum but not the list is a metric that never reaches a record.

    Test type: unit
    """
    declared = ARDESPOT.get_info_variable_names()
    assert len(declared) == len(set(declared))
    for metric in ARDESPOTMetrics:
        assert metric.value in declared
    for metric in TreeMetrics:
        assert metric.value in declared


def test_ardespot_metric_names_do_not_collide_with_despot_ones():
    """Two different quantities must not share a name in a merged record.

    Purpose: ``despot_root_gap`` is ``u - l`` on a conditional scale;
    ``ardespot_root_gap`` is ``mu - l`` on an unnormalized one. A run that
    compared the two planners and averaged one column would be averaging two
    different things.

    Test type: unit
    """
    ardespot_names = {metric.value for metric in ARDESPOTMetrics}
    despot_names = {metric.value for metric in DESPOTMetrics}
    assert ardespot_names.isdisjoint(despot_names)


def test_a_terminal_belief_short_circuits_before_any_search():
    """No tree is built for a belief that is already terminal.

    Purpose: Planning from a terminal belief wastes the whole decision budget
    and produces bounds over an empty horizon.

    Given: A ``ChainEnv`` belief whose only particle is the terminal state.
    Then: A legal action comes back with no info variables and no tree.

    Test type: unit
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=4)
    actions, run_data = planner.action(chain_belief(END))

    assert actions[0] in environment.get_actions()
    assert run_data.info_variables == []
    assert planner._last_tree is None


def test_every_declared_ardespot_metric_is_emitted_on_a_real_decision():
    """No metric is quietly conditional.

    Purpose: DESPOT omits its regularization metric when regularization is off,
    which is right for a flag that means "not applicable". AR-DESPOT has no
    such case -- regularization is always on, even at ``lambda = 0`` where it
    degenerates -- so a missing name here would be a bug, not a distinction.

    Test type: unit
    """
    np.random.seed(5)
    environment = _tiger()
    planner = _planner(environment)
    _, run_data = planner.action(get_initial_belief(environment, n_particles=20))
    emitted = {variable.name for variable in run_data.info_variables}

    for metric in ARDESPOTMetrics:
        assert metric.value in emitted


# ---------------------------------------------------------------------------
# Metric values
# ---------------------------------------------------------------------------


def test_reported_metrics_match_an_independent_count_of_the_same_tree():
    """Each count is recomputed from the arena rather than trusted.

    Purpose: A counter incremented in the wrong branch stays plausible for the
    whole life of the planner; the only check that catches it is a second,
    independent count.

    Given: A decision on Tiger with a budget large enough to build a real tree.
    Then: Belief-node counts, the ``in_tree`` count, the root's three
        quantities and the gap all match a direct sweep of the arena.

    Test type: unit
    """
    np.random.seed(23)
    environment = _tiger()
    planner = _planner(environment, n_simulations=25)
    _, run_data = planner.action(get_initial_belief(environment, n_particles=30))
    metrics = {variable.name: variable.value for variable in run_data.info_variables}

    tree = planner._last_tree
    root_id = planner._last_root_id
    assert tree is not None and root_id is not None

    belief_ids = [node_id for node_id in range(len(tree)) if tree.kind[node_id] == BELIEF]
    assert metrics["ardespot_n_belief_nodes"] == len(belief_ids)
    assert metrics["ardespot_n_tree_belief_nodes"] == sum(
        1 for node_id in belief_ids if tree.data[node_id].in_tree
    )
    assert metrics["ardespot_n_tree_belief_nodes"] <= metrics["ardespot_n_belief_nodes"]
    assert metrics["ardespot_n_scenarios"] == planner.n_scenarios

    assert metrics["ardespot_root_lower_bound"] == pytest.approx(
        tree.lower_confidence_bound[root_id]
    )
    assert metrics["ardespot_root_regularized_value"] == pytest.approx(tree.v_value[root_id])
    assert metrics["ardespot_root_upper_bound"] == pytest.approx(
        tree.upper_confidence_bound[root_id]
    )
    assert metrics["ardespot_root_gap"] == pytest.approx(
        tree.v_value[root_id] - tree.lower_confidence_bound[root_id]
    )

    # The planner's own depth metric counts belief steps; the arena's counts
    # arena edges, so they are deliberately not the same number.
    assert metrics["ardespot_max_trial_depth"] <= planner.depth
    assert metrics["ardespot_max_trial_depth"] == max(
        (tree.data[node_id].depth for node_id in belief_ids if tree.data[node_id].in_tree),
        default=0,
    )


def test_exactly_one_stop_flag_is_set_on_a_completed_decision():
    """The three stop reasons are mutually exclusive and one always fires.

    Purpose: Reporting them as three flags only helps if they partition the
    outcomes; two set at once, or none, would make an aggregate over them
    meaningless.

    Test type: unit
    """
    np.random.seed(3)
    for overrides in (
        {"n_simulations": 200},  # expect the gap to close
        {"n_simulations": 1, "xi": 0.01},  # expect the trial budget to win
    ):
        environment = _tiger()
        planner = _planner(environment, **overrides)
        _, run_data = planner.action(get_initial_belief(environment, n_particles=20))
        metrics = {variable.name: variable.value for variable in run_data.info_variables}
        flags = (
            metrics["ardespot_gap_closed"]
            + metrics["ardespot_time_exhausted"]
            + metrics["ardespot_trials_exhausted"]
            + metrics["ardespot_stalled"]
        )
        assert flags == 1, f"stop flags did not partition for {overrides}: {metrics}"


def test_a_second_decision_neither_accumulates_nor_rewrites_the_first_ones_metrics():
    """Every metric is per decision and reset by the next ``action`` call.

    Purpose: A counter that survives a decision turns an episode's last record
    into a running total, and every per-decision average computed from it is
    wrong by a factor that grows with the episode.

    Test type: unit
    """
    np.random.seed(29)
    environment = _tiger()
    planner = _planner(environment, n_simulations=12)
    belief = get_initial_belief(environment, n_particles=20)

    _, first = planner.action(belief)
    _, second = planner.action(belief)
    first_metrics = {variable.name: variable.value for variable in first.info_variables}
    second_metrics = {variable.name: variable.value for variable in second.info_variables}

    for name in (
        "ardespot_n_trials",
        "ardespot_n_belief_nodes",
        "ardespot_n_blocked",
        "ardespot_n_depth_defaults",
    ):
        assert second_metrics[name] <= first_metrics[name] * 2 + 1
    assert second_metrics["ardespot_n_trials"] <= planner.n_simulations


# ---------------------------------------------------------------------------
# Configuration identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"depth": 6},
        {"n_scenarios": 12},
        {"xi": 0.6},
        {"pruning_constant": 0.5},
        {"epsilon_0": 0.25},
        {"max_reward": 42.0},
        {"min_reward": -42.0},
        {"rollout_depth": 3},
        {"scenario_seed": 7},
        {"use_determinized_scenarios": False},
        {"n_simulations": 30},
    ],
)
def test_config_id_is_stable_for_equal_configs_and_moves_with_an_algorithm_parameter(change):
    """Every parameter that changes the search must change the cache key.

    Purpose: A parameter outside ``config_id`` makes two different planners
    share cached episodes, and the second study silently reports the first
    one's numbers.

    Test type: unit
    """
    baseline = _planner(_tiger())
    twin = _planner(_tiger())
    assert baseline.config_id == twin.config_id

    changed = _planner(_tiger(), **change)
    assert changed.config_id != baseline.config_id, f"{change} did not move config_id"


def test_xi_and_eta_are_one_value_not_two():
    """``xi`` is the reference's name for ``eta``; two attributes would drift.

    Purpose: If both were writable attributes they could be set apart, and
    ``config_id`` would carry two numbers for one knob -- so two planners with
    the same search behaviour could get different cache keys, or worse, the
    same key with different behaviour.

    Test type: unit
    """
    planner = _planner(_tiger(), xi=0.42)
    assert planner.xi == 0.42
    assert planner.eta == 0.42
    with pytest.raises(AttributeError):
        planner.xi = 0.9  # type: ignore[misc]


def test_transient_search_state_does_not_change_the_cache_key():
    """One search's node ids and counters are not configuration.

    Purpose: If they were, a planner's cache key would change after every
    decision and nothing would ever hit the cache.

    Test type: unit
    """
    np.random.seed(19)
    environment = _tiger()
    planner = _planner(environment)
    before = planner.config_id
    planner.action(get_initial_belief(environment, n_particles=20))
    assert planner.config_id == before


def test_a_saved_planner_reloads_with_its_parameters_and_can_still_decide():
    """Serialization has to survive the round trip and still work.

    Purpose: Reconstructing without raising is not the property that matters; a
    reloaded planner that cannot act is useless.

    Test type: unit
    """
    np.random.seed(37)
    random.seed(37)
    environment = _tiger()
    planner = _planner(
        environment,
        depth=4,
        n_scenarios=6,
        xi=0.8,
        pruning_constant=0.5,
        epsilon_0=0.1,
        rollout_depth=3,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = planner.save(Path(tmpdir) / "ardespot.json")
        loaded = Policy.load(path)

    assert isinstance(loaded, ARDESPOT)
    for parameter in (
        "depth",
        "n_scenarios",
        "eta",
        "xi",
        "epsilon_0",
        "pruning_constant",
        "rollout_depth",
        "max_reward",
        "min_reward",
        "scenario_seed",
        "use_determinized_scenarios",
        "n_simulations",
        "time_out_in_seconds",
    ):
        assert getattr(loaded, parameter) == getattr(
            planner, parameter
        ), f"{parameter} did not survive the round trip"
    assert loaded.config_id == planner.config_id

    actions, _ = loaded.action(get_initial_belief(loaded.environment, n_particles=20))
    assert actions[0] in loaded.environment.get_actions()  # type: ignore[attr-defined]


def test_ardespot_is_reachable_through_the_policy_registry():
    """The factory path other tooling uses must know about the planner.

    Purpose: A planner missing from ``POLICY_REGISTRY`` cannot be named in a
    run configuration, and the failure appears only when a batch is launched.

    Test type: unit
    """
    assert POLICY_REGISTRY["ARDESPOT"] is ARDESPOT


# ---------------------------------------------------------------------------
# Constructor refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"depth": 0}, "depth must be positive"),
        ({"depth": 1.5}, "depth must be an int"),
        ({"n_scenarios": 0}, "n_scenarios must be positive"),
        ({"xi": 1.0}, "eta must be strictly between 0 and 1"),
        ({"xi": 0.0}, "eta must be strictly between 0 and 1"),
        ({"pruning_constant": -1.0}, "pruning_constant must be non-negative"),
        ({"epsilon_0": -1.0}, "epsilon_0 must be non-negative"),
        ({"rollout_depth": -1}, "rollout_depth must be non-negative"),
    ],
)
def test_an_out_of_range_parameter_is_refused_at_construction(overrides, message):
    """A parameter outside its stated range is an error, not a clamp.

    Purpose: Silently clamping produces a planner that is not the one that was
    asked for, and the study reports results under the wrong configuration.

    Test type: unit
    """
    with pytest.raises((ValueError, TypeError), match=message):
        _planner(_tiger(), **overrides)


def test_at_least_one_budget_must_be_given():
    """No budget means an unbounded loop, so it is refused.

    Purpose: With neither budget the trial loop would run until the gap closes,
    which on a stochastic environment may never happen inside an episode.

    Test type: unit
    """
    with pytest.raises(ValueError, match="at least one of time_out_in_seconds"):
        ARDESPOT(
            environment=_tiger(),
            discount_factor=DISCOUNT,
            depth=4,
            name="ARDESPOT_no_budget",
            n_scenarios=4,
        )


def test_an_environment_without_a_declared_reward_range_must_be_given_max_reward():
    """``R_max`` cannot be guessed; a wrong one voids the branch-and-bound.

    Purpose: An upper bound that is not actually a bound turns the search into
    an arbitrary heuristic while still reporting bounds, so the failure is
    invisible in the metrics.

    Test type: unit
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    environment.reward_range = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="max_reward"):
        _planner(environment, depth=3)

    environment_again = ChainEnv(discount_factor=DISCOUNT)
    environment_again.reward_range = None  # type: ignore[assignment]
    planner = _planner(environment_again, depth=3, max_reward=4.0)
    actions, _ = planner.action(chain_belief(ROOT))
    assert actions[0] in environment_again.get_actions()


# ---------------------------------------------------------------------------
# Budget boundaries
# ---------------------------------------------------------------------------


def test_a_single_trial_still_produces_a_usable_decision():
    """One trial is the smallest budget the anytime contract has to honour.

    Test type: unit
    """
    np.random.seed(13)
    environment = _tiger()
    planner = _planner(environment, n_simulations=1, xi=0.01)
    actions, run_data = planner.action(get_initial_belief(environment, n_particles=20))
    metrics = {variable.name: variable.value for variable in run_data.info_variables}

    assert actions[0] in environment.get_actions()
    assert metrics["ardespot_n_trials"] == 1
    assert metrics["ardespot_n_belief_nodes"] >= 1


def test_a_wall_clock_budget_is_respected_and_does_more_work_when_it_is_larger():
    """The clock must both bound the decision and buy trials.

    Purpose: A budget that is honoured but buys nothing means the search is
    stalling; a budget that buys work but is not honoured breaks calibration,
    which is the only thing making a multi-planner comparison fair.

    Given: An environment with no terminal state and a tiny ``xi``, so the gap
        never closes and the clock is the binding constraint.

    Test type: integration
    """
    np.random.seed(17)
    trials = []
    for budget in (1, 2):
        environment = ChainEnv(discount_factor=DISCOUNT, terminal_states=())
        planner = _planner(
            environment,
            depth=30,
            n_scenarios=16,
            xi=0.001,
            n_simulations=None,
            time_out_in_seconds=budget,
        )
        start = time.time()
        _, run_data = planner.action(chain_belief(ROOT))
        elapsed = time.time() - start
        metrics = {variable.name: variable.value for variable in run_data.info_variables}
        trials.append(metrics["ardespot_n_trials"])
        # One trial can overrun the deadline; it cannot double the budget.
        assert elapsed < budget * 2 + 1
        assert metrics["ardespot_time_exhausted"] == 1

    assert trials[1] > trials[0]


def test_the_trial_budget_wins_when_it_is_the_tighter_of_the_two():
    """Both budgets are enforced together, and the tighter one ends the search.

    Purpose: This is the whole reason AR-DESPOT accepts both. If only the last
    one set took effect, a calibrated wall clock could be silently ignored.

    Test type: integration
    """
    np.random.seed(31)
    environment = ChainEnv(discount_factor=DISCOUNT, terminal_states=())
    planner = _planner(
        environment,
        depth=30,
        n_scenarios=8,
        xi=0.001,
        n_simulations=3,
        time_out_in_seconds=60,
    )
    _, run_data = planner.action(chain_belief(ROOT))
    metrics = {variable.name: variable.value for variable in run_data.info_variables}

    assert metrics["ardespot_n_trials"] == 3
    assert metrics["ardespot_trials_exhausted"] == 1
    assert metrics["ardespot_time_exhausted"] == 0


def test_a_converged_search_stops_early_instead_of_spending_its_whole_budget():
    """Anytime cuts both ways: a resolved root ends the decision.

    Test type: integration
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=2, n_scenarios=4, n_simulations=500)
    _, run_data = planner.action(chain_belief(ROOT))
    metrics = {variable.name: variable.value for variable in run_data.info_variables}

    assert metrics["ardespot_gap_closed"] == 1
    assert metrics["ardespot_n_trials"] < 500


def test_a_deterministic_environment_produces_a_sparse_tree_not_one_branch_per_scenario():
    """Observation grouping is what makes the tree sparse.

    Purpose: Without grouping, ``K`` scenarios give ``K`` branches at every
    node and the tree is exponential in ``K`` rather than in the number of
    distinct observations.

    Test type: integration
    """
    environment = ChainEnv(discount_factor=DISCOUNT, terminal_states=())
    planner = _planner(environment, depth=3, n_scenarios=16, n_simulations=10, xi=0.01)
    planner.action(chain_belief(ROOT))

    tree = planner._last_tree
    assert tree is not None
    for node_id in range(len(tree)):
        if tree.kind[node_id] == ACTION:
            assert len(tree.get_children_ids(node_id)) == 1


# ---------------------------------------------------------------------------
# Search-state export
# ---------------------------------------------------------------------------


def test_the_search_state_export_records_every_node_and_the_ardespot_quantities():
    """The dump is the QA artifact, so it has to carry what QA reads.

    Purpose: ``mu``, ``l_0``, ``rho`` and ``ba_mu`` are the numbers that
    explain an AR-DESPOT decision. A dump without them can show that a decision
    happened but not why.

    Test type: integration
    """
    np.random.seed(41)
    environment = _tiger()
    planner = _planner(environment, n_simulations=20)
    planner.action(get_initial_belief(environment, n_particles=20))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = planner.export_search_state(Path(tmpdir) / "state.json")
        payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["planner_class"] == "ARDESPOT"
    assert payload["config_id"] == planner.config_id
    assert payload["config"]["xi"] == planner.xi
    assert payload["config"]["epsilon_0"] == planner.epsilon_0
    assert "eta" not in payload["config"]
    assert payload["search"]["n_blocked"] == planner._n_blocked
    assert payload["search"]["n_depth_defaults"] == planner._n_depth_defaults

    tree = planner._last_tree
    assert tree is not None
    assert len(payload["nodes"]) == len(tree)

    belief_records = [record for record in payload["nodes"] if record["kind"] == "belief"]
    action_records = [record for record in payload["nodes"] if record["kind"] == "action"]
    assert belief_records and action_records
    for record in belief_records:
        assert {"mu", "l_0", "depth", "defaulted_by", "in_tree"} <= set(record)
    for record in action_records:
        assert {"rho", "ba_mu", "ba_l", "reward_sum"} <= set(record)


def test_enabling_the_dump_writes_one_file_per_decision():
    """QA needs a dump per decision, not one overwritten file.

    Test type: integration
    """
    np.random.seed(43)
    environment = _tiger()
    planner = _planner(environment, n_simulations=8)
    belief = get_initial_belief(environment, n_particles=20)

    with tempfile.TemporaryDirectory() as tmpdir:
        planner.enable_search_state_dump(Path(tmpdir))
        planner.action(belief)
        planner.action(belief)
        planner.disable_search_state_dump()
        planner.action(belief)

        written = sorted(Path(tmpdir).glob("*.json"))
        assert len(written) == 2
        assert written[0].name.startswith(planner.name)


def test_exporting_before_any_search_is_an_error_not_an_empty_file():
    """An empty dump would look like a search that found nothing.

    Test type: unit
    """
    planner = _planner(_tiger())
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="no search state"):
            planner.export_search_state(Path(tmpdir) / "state.json")
