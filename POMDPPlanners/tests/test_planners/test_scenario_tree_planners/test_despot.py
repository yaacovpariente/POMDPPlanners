# SPDX-License-Identifier: MIT

"""DESPOT public interface: metrics, configuration identity, boundaries, export.

The algorithm's equations are pinned in ``test_despot_correctness.py``. This
file covers what surrounds them: the ``Policy`` contract, the metric names and
values that reach a simulation record, the cache key, the constructor's refusals,
and the search-state dump QA depends on.

References:
    Somani, Ye, Hsu & Lee (2013), NeurIPS 26. Ye, Somani, Hsu & Lee (2017),
    JAIR 58.
"""

# pylint: disable=protected-access

import json
import math
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
from POMDPPlanners.planners import POLICY_REGISTRY, DESPOT
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


def _planner(environment: Any = None, **overrides: Any) -> DESPOT:
    environment = environment if environment is not None else _tiger()
    params: Dict[str, Any] = dict(
        environment=environment,
        discount_factor=DISCOUNT,
        depth=5,
        name="DESPOT_test",
        n_scenarios=8,
        n_simulations=15,
    )
    params.update(overrides)
    return DESPOT(**params)


# ---------------------------------------------------------------------------
# Policy contract
# ---------------------------------------------------------------------------


def test_action_returns_one_legal_action_and_only_declared_metrics():
    """The ``Policy`` contract, and that every metric name was declared.

    Purpose: A metric emitted but not declared cannot be selected for tuning,
    and one declared but never emitted reads as a missing column downstream.

    Given: DESPOT on Tiger with a fixed simulation count.
    When: One decision is taken.
    Then: Exactly one legal action comes back, every emitted metric name is in
        ``get_info_variable_names``, every value is a finite number, and no name
        is emitted twice.

    Test type: unit
    """
    np.random.seed(17)
    random.seed(17)
    environment = _tiger()
    planner = _planner(environment)

    actions, run_data = planner.action(get_initial_belief(environment, n_particles=30))

    assert len(actions) == 1
    assert actions[0] in environment.get_actions()

    declared = set(DESPOT.get_info_variable_names())
    names = [variable.name for variable in run_data.info_variables]
    assert len(names) == len(set(names)), f"duplicate metric names emitted: {names}"
    assert set(names) <= declared, f"undeclared metrics emitted: {set(names) - declared}"
    for variable in run_data.info_variables:
        assert isinstance(variable.value, (int, float))
        assert math.isfinite(variable.value), f"{variable.name} is not finite"


def test_declared_metric_names_cover_both_the_shared_and_the_despot_specific_sets():
    """``get_info_variable_names`` is the union of the two enums.

    Purpose: The declaration is what a tuning study reads before any simulation
    runs, so it has to be complete before the planner is ever called.

    Test type: unit
    """
    declared = set(DESPOT.get_info_variable_names())
    assert {metric.value for metric in TreeMetrics} <= declared
    assert {metric.value for metric in DESPOTMetrics} <= declared


def test_a_terminal_belief_short_circuits_before_any_search():
    """No tree is built when there is nothing left to decide.

    Purpose: Matches the shared base's contract, and the empty metric list is
    what tells a record reader "this decision had no search", rather than a row
    of measured zeros claiming a search that found nothing.

    Given: ``ChainEnv`` and a belief concentrated on the terminal ``end`` state.
    When: A decision is taken.
    Then: A legal action comes back with an empty metric list, and no search
        state was recorded.

    Test type: unit
    """
    environment = ChainEnv(discount_factor=0.5)
    planner = _planner(environment, discount_factor=0.5, depth=3)

    actions, run_data = planner.action(chain_belief(END))

    assert actions[0] in environment.get_actions()
    assert run_data.info_variables == []
    assert planner._last_tree is None


def test_the_regularization_metric_is_absent_rather_than_zero_when_it_does_not_apply():
    """A missing metric means "not applicable"; a zero would be a measurement.

    Purpose: ``despot_regularized_fell_back`` only means something when
    regularization is on. Reporting ``0`` with ``pruning_constant = 0`` would
    claim the regularized pass ran and did not fall back.

    Given: Two planners differing only in ``pruning_constant``.
    When: Each takes a decision.
    Then: The metric appears for the regularized one and not for the other.

    Test type: unit
    """
    np.random.seed(19)
    random.seed(19)
    environment = _tiger()
    belief = get_initial_belief(environment, n_particles=30)

    _, plain = _planner(environment, pruning_constant=0.0).action(belief)
    _, regularized = _planner(environment, pruning_constant=1.0).action(belief)

    plain_names = {variable.name for variable in plain.info_variables}
    regularized_names = {variable.name for variable in regularized.info_variables}
    assert DESPOTMetrics.REGULARIZED_FELL_BACK.value not in plain_names
    assert DESPOTMetrics.REGULARIZED_FELL_BACK.value in regularized_names


# ---------------------------------------------------------------------------
# Metrics, recomputed independently from the same tree
# ---------------------------------------------------------------------------


def test_reported_metrics_match_an_independent_count_of_the_same_tree():
    """Every DESPOT metric is recomputed here from the arena, not from the helper.

    Purpose: A metric that is wrong in the same way as the code that produced it
    is invisible; the expected values below are counted straight off the columns.

    Given: DESPOT on Tiger, 20 trials, 8 scenarios, depth 5.
    When: One decision is taken and the retained tree inspected.
    Then: Trial count equals the root's visit count; the belief-node count,
        the in-tree count and the gap all match an independent sweep; the
        deepest trial depth does not exceed ``depth``; and the shared
        ``root_visit_count`` and ``n_actions_from_root`` agree with the arena.

    Test type: unit
    """
    np.random.seed(23)
    random.seed(23)
    environment = _tiger()
    planner = _planner(environment, n_simulations=20)

    _, run_data = planner.action(get_initial_belief(environment, n_particles=30))
    metrics = {variable.name: variable.value for variable in run_data.info_variables}
    tree = planner._last_tree
    root_id = planner._last_root_id

    belief_nodes = [nid for nid in range(len(tree)) if tree.kind[nid] == BELIEF]
    in_tree = [nid for nid in belief_nodes if tree.data[nid].in_tree]

    assert metrics[DESPOTMetrics.N_BELIEF_NODES.value] == len(belief_nodes)
    assert metrics[DESPOTMetrics.N_TREE_BELIEF_NODES.value] == len(in_tree)
    assert len(in_tree) < len(
        belief_nodes
    ), "expansion should leave a fringe, otherwise the two counts cannot be told apart"
    assert metrics[DESPOTMetrics.N_TRIALS.value] == tree.visit_count[root_id]
    assert metrics[DESPOTMetrics.N_SCENARIOS.value] == planner.n_scenarios

    lower = tree.lower_confidence_bound[root_id]
    upper = tree.upper_confidence_bound[root_id]
    assert metrics[DESPOTMetrics.ROOT_LOWER_BOUND.value] == pytest.approx(lower)
    assert metrics[DESPOTMetrics.ROOT_UPPER_BOUND.value] == pytest.approx(upper)
    assert metrics[DESPOTMetrics.ROOT_GAP.value] == pytest.approx(upper - lower)

    deepest = max(tree.data[nid].depth for nid in in_tree)
    assert metrics[DESPOTMetrics.MAX_TRIAL_DEPTH.value] == deepest
    assert deepest <= planner.depth

    root_actions = [cid for cid in tree.children_ids[root_id] if tree.kind[cid] == ACTION]
    assert metrics[TreeMetrics.N_ACTIONS_FROM_ROOT.value] == len(root_actions)
    assert metrics[TreeMetrics.ROOT_VISIT_COUNT.value] == tree.visit_count[root_id]
    assert metrics[TreeMetrics.IS_LEAF.value] == 0

    visits = [tree.visit_count[cid] for cid in root_actions]
    assert metrics[TreeMetrics.MIN_ACTIONS_VISIT_COUNT.value] == min(visits)
    assert metrics[TreeMetrics.MAX_ACTIONS_VISIT_COUNT.value] == max(visits)
    total = sum(visits)
    expected_entropy = -sum(
        (count / total) * math.log2(count / total) for count in visits if count > 0
    )
    assert metrics[TreeMetrics.ACTIONS_VISIT_COUNT_ENTROPY.value] == pytest.approx(expected_entropy)


def test_a_second_decision_neither_accumulates_nor_rewrites_the_first_ones_metrics():
    """Metrics are per decision; nothing carries over.

    Purpose: DESPOT's counters live on the planner instance rather than on the
    tree, which is exactly the shape that leaks across calls if it is not reset.

    Given: Two consecutive decisions from the same belief at a fixed trial count.
    When: Both run.
    Then: The first call's ``PolicyRunData`` is unchanged, and the second
        reports no more trials than its budget rather than double.

    Test type: unit
    """
    np.random.seed(29)
    random.seed(29)
    environment = _tiger()
    planner = _planner(environment, n_simulations=12)
    belief = get_initial_belief(environment, n_particles=30)

    _, first = planner.action(belief)
    snapshot = [(variable.name, variable.value) for variable in first.info_variables]
    _, second = planner.action(belief)

    assert [
        (variable.name, variable.value) for variable in first.info_variables
    ] == snapshot, "the second decision mutated the first decision's run data"
    second_metrics = {variable.name: variable.value for variable in second.info_variables}
    assert second_metrics[DESPOTMetrics.N_TRIALS.value] <= 12, (
        f"the second decision reports {second_metrics[DESPOTMetrics.N_TRIALS.value]} trials "
        f"against a budget of 12, so the counter carried over"
    )


# ---------------------------------------------------------------------------
# Configuration identity
# ---------------------------------------------------------------------------


CONFIG_CHANGES = [
    {"depth": 6},
    {"n_scenarios": 16},
    {"eta": 0.5},
    {"pruning_constant": 2.0},
    {"max_reward": 25.0},
    {"rollout_depth": 2},
    {"scenario_seed": 99},
    {"use_determinized_scenarios": False},
    {"n_simulations": 30},
]


@pytest.mark.parametrize("change", CONFIG_CHANGES, ids=lambda c: next(iter(c)))
def test_config_id_is_stable_for_equal_configs_and_moves_with_an_algorithm_parameter(change):
    """``config_id`` is the simulation cache key.

    Purpose: If it ignored a parameter, a tuning study would serve one setting's
    cached episodes for another's -- silent data corruption rather than a crash.

    Given: Two identically configured planners and a third differing in exactly
        one algorithm parameter.
    When: Their ``config_id`` values are compared.
    Then: The first two match and the third does not.

    Test type: unit
    """
    first = _planner()
    same = _planner()
    different = _planner(**change)

    assert (
        first.config_id == same.config_id
    ), "two identically configured planners produced different cache keys"
    (parameter,) = change
    assert (
        first.config_id != different.config_id
    ), f"changing {parameter} to {change[parameter]!r} left the cache key unchanged"


def test_transient_search_state_does_not_change_the_cache_key():
    """One search's node IDs must not become part of the planner's identity.

    Purpose: A cache key that moved after every decision would make every rerun
    recompute, and would differ between a resumed batch and a fresh one.

    Given: A planner before and after taking a decision, and a second planner
        with a search-state dump enabled.
    When: The cache keys are compared.
    Then: All three are equal.

    Test type: unit
    """
    np.random.seed(31)
    random.seed(31)
    environment = _tiger()
    planner = _planner(environment)
    before = planner.config_id
    planner.action(get_initial_belief(environment, n_particles=20))
    assert planner.config_id == before, "planning changed the planner's cache key"

    with tempfile.TemporaryDirectory() as tmpdir:
        dumping = _planner(environment)
        dumping.enable_search_state_dump(Path(tmpdir))
        assert dumping.config_id == before, (
            "enabling a QA dump changed the cache key, so a dump run could not reuse "
            "a performance run's cached episodes"
        )


def test_a_saved_planner_reloads_with_its_parameters_and_can_still_decide():
    """Serialization has to survive the round trip and still work.

    Purpose: Reconstructing without raising is not the property that matters; a
    reloaded planner that cannot act is useless.

    Given: A configured planner saved to a temporary file.
    When: It is loaded back.
    Then: Every algorithm parameter matches, the cache key matches, and the
        reloaded planner returns a legal action.

    Test type: unit
    """
    np.random.seed(37)
    random.seed(37)
    environment = _tiger()
    planner = _planner(
        environment, depth=4, n_scenarios=6, eta=0.8, pruning_constant=0.5, rollout_depth=3
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = planner.save(Path(tmpdir) / "despot.json")
        loaded = Policy.load(path)

    assert isinstance(loaded, DESPOT)
    for parameter in (
        "depth",
        "n_scenarios",
        "eta",
        "pruning_constant",
        "rollout_depth",
        "max_reward",
        "min_reward",
        "scenario_seed",
        "use_determinized_scenarios",
        "n_simulations",
    ):
        assert getattr(loaded, parameter) == getattr(
            planner, parameter
        ), f"{parameter} did not survive the round trip"
    assert loaded.config_id == planner.config_id

    actions, _ = loaded.action(get_initial_belief(loaded.environment, n_particles=20))
    assert actions[0] in loaded.environment.get_actions()  # type: ignore[attr-defined]


def test_despot_is_reachable_through_the_policy_registry():
    """The factory path other tooling uses must know about the planner.

    Purpose: A planner missing from ``POLICY_REGISTRY`` cannot be named in a run
    configuration, and the failure appears only when a batch is launched.

    Test type: unit
    """
    assert POLICY_REGISTRY["DESPOT"] is DESPOT


# ---------------------------------------------------------------------------
# Constructor refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"depth": 0}, "depth must be positive"),
        ({"depth": 2.5}, "depth must be an int"),
        ({"n_scenarios": 0}, "n_scenarios must be positive"),
        ({"eta": 1.0}, "eta must be strictly between 0 and 1"),
        ({"eta": 0.0}, "eta must be strictly between 0 and 1"),
        ({"pruning_constant": -1.0}, "pruning_constant must be non-negative"),
        ({"rollout_depth": -1}, "rollout_depth must be non-negative"),
        ({"max_reward": float("inf")}, "max_reward must be finite"),
    ],
)
def test_an_out_of_range_parameter_is_refused_at_construction(overrides, message):
    """Bad configuration fails at build time, not four hours into a batch.

    ``eta = 1`` is in this list for a reason that is not obvious: it makes
    ``(1 - eta) * gap`` zero before the first trial, so the planner would return
    from the initial bounds without searching at all -- a silent no-op rather
    than an error.

    Test type: unit
    """
    with pytest.raises((ValueError, TypeError), match=message):
        _planner(**overrides)


@pytest.mark.parametrize(
    "budget",
    [
        {"n_simulations": None, "time_out_in_seconds": None},
        {"n_simulations": 10, "time_out_in_seconds": 1},
    ],
)
def test_exactly_one_budget_must_be_given(budget):
    """Neither budget, or both, is a configuration error.

    Purpose: Two budgets would leave which one applies to the calling order, and
    a comparison across planners would stop being a comparison at equal compute.

    Test type: unit
    """
    with pytest.raises(ValueError, match="Only one of"):
        _planner(**budget)


def test_an_environment_without_a_declared_reward_range_must_be_given_max_reward():
    """DESPOT refuses to invent its own upper bound.

    Purpose: An upper bound that is not one turns branch-and-bound into an
    arbitrary heuristic and voids the paper's guarantee, so guessing is worse
    than failing.

    Given: ``ChainEnv`` with its ``reward_range`` removed.
    When: A planner is constructed without ``max_reward``.
    Then: A ``ValueError`` naming ``max_reward`` and the environment is raised,
        and supplying ``max_reward`` explicitly works.

    Test type: unit
    """
    environment = ChainEnv(discount_factor=0.5)
    environment.reward_range = None

    with pytest.raises(ValueError, match="max_reward"):
        _planner(environment, discount_factor=0.5, depth=3)

    planner = _planner(environment, discount_factor=0.5, depth=3, max_reward=4.0)
    assert planner.max_reward == 4.0


# ---------------------------------------------------------------------------
# Budgets and boundaries
# ---------------------------------------------------------------------------


def test_a_single_trial_still_produces_a_usable_decision():
    """The minimum budget must not be a special case.

    Purpose: A calibration sweep will hit one trial per decision on an expensive
    environment; that has to return an action rather than fall through to the
    random fallback.

    Given: DESPOT on Tiger with one trial.
    When: A decision is taken.
    Then: A legal action comes back, the root has action children, and the
        trial count is one.

    Test type: unit
    """
    np.random.seed(41)
    random.seed(41)
    environment = _tiger()
    planner = _planner(environment, n_simulations=1)

    actions, run_data = planner.action(get_initial_belief(environment, n_particles=20))
    metrics = {variable.name: variable.value for variable in run_data.info_variables}

    assert actions[0] in environment.get_actions()
    assert metrics[DESPOTMetrics.N_TRIALS.value] == 1
    assert metrics[TreeMetrics.IS_LEAF.value] == 0


def test_a_wall_clock_budget_is_respected_and_does_more_work_when_it_is_larger():
    """The time budget bounds the decision and buys trials.

    Purpose: Calibration sets a per-decision time budget, so it must actually
    bound the call, and a larger budget must buy search rather than being
    ignored.

    Given: Tiger with a deep horizon so the gap does not close early, at two
        budgets.
    When: One decision is taken at each.
    Then: Each call finishes within its budget plus one trial's slack, and the
        longer budget runs at least as many trials.

    Test type: unit
    """
    np.random.seed(43)
    random.seed(43)
    environment = _tiger()
    belief = get_initial_belief(environment, n_particles=40)

    trials = {}
    for budget in (1, 2):
        planner = _planner(
            environment,
            depth=30,
            n_scenarios=32,
            n_simulations=None,
            time_out_in_seconds=budget,
        )
        start = time.time()
        _, run_data = planner.action(belief)
        elapsed = time.time() - start
        metrics = {variable.name: variable.value for variable in run_data.info_variables}
        trials[budget] = metrics[DESPOTMetrics.N_TRIALS.value]
        assert elapsed < budget + 2.0, (
            f"a {budget}s decision took {elapsed:.2f}s; the loop checks its budget only "
            f"between trials, so the slack allows one trial to overrun"
        )

    assert trials[2] >= trials[1], (
        f"twice the budget ran {trials[2]} trials against {trials[1]}; the budget is "
        f"not buying search"
    )


def test_a_converged_search_stops_early_instead_of_spending_its_whole_budget():
    """The ``eta`` stopping rule is what makes DESPOT anytime.

    Purpose: A planner that always burns its budget is not using the bound, and
    the ``despot_gap_closed`` flag would be meaningless.

    Given: The deterministic ``ChainEnv``, where the interval closes exactly,
        with a budget of 50 trials.
    When: A decision is taken.
    Then: Fewer than 50 trials ran, the gap is zero and the flag is set.

    Test type: unit
    """
    environment = ChainEnv(discount_factor=0.5)
    planner = _planner(environment, discount_factor=0.5, depth=3, n_scenarios=4, n_simulations=50)

    _, run_data = planner.action(chain_belief(ROOT))
    metrics = {variable.name: variable.value for variable in run_data.info_variables}

    assert metrics[DESPOTMetrics.GAP_CLOSED.value] == 1
    assert metrics[DESPOTMetrics.N_TRIALS.value] < 50
    assert metrics[DESPOTMetrics.ROOT_GAP.value] == pytest.approx(0.0, abs=1e-12)


def test_a_deterministic_environment_produces_a_sparse_tree_not_one_branch_per_scenario():
    """Observation grouping is what "sparse" means here.

    Purpose: If scenarios were not merged by observation, ``K`` scenarios would
    give ``K`` branches per action and the tree would be exponential in ``K``
    rather than in the number of distinct observations.

    Given: ``ChainEnv`` with 16 scenarios, whose transition is deterministic so
        every scenario sees the same observation.
    When: A decision is taken.
    Then: Every action node has exactly one belief child.

    Test type: unit
    """
    environment = ChainEnv(discount_factor=0.5)
    planner = _planner(environment, discount_factor=0.5, depth=3, n_scenarios=16, n_simulations=10)
    planner.action(chain_belief(ROOT))
    tree = planner._last_tree

    action_nodes = [nid for nid in range(len(tree)) if tree.kind[nid] == ACTION]
    assert action_nodes
    for node_id in action_nodes:
        assert len(tree.children_ids[node_id]) == 1, (
            f"action node {node_id} has {len(tree.children_ids[node_id])} observation "
            f"branches on a deterministic environment; scenarios are not being merged"
        )


# ---------------------------------------------------------------------------
# Search-state export
# ---------------------------------------------------------------------------


def test_the_search_state_export_records_every_node_and_the_configuration():
    """The QA artifact must be complete enough to reason from.

    Purpose: ``planner-qa`` reasons about a decision from this file alone, so a
    dump missing bounds, weights or scenario membership cannot answer why a
    branch was expanded.

    Given: A decision on Tiger.
    When: The search state is exported.
    Then: The file holds one record per arena node with the fields the review
        needs, the configuration that produced it, and the search summary.

    Test type: unit
    """
    np.random.seed(47)
    random.seed(47)
    environment = _tiger()
    planner = _planner(environment, n_simulations=10)
    planner.action(get_initial_belief(environment, n_particles=20))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = planner.export_search_state(Path(tmpdir) / "dump.json")
        payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["planner_class"] == "DESPOT"
    assert payload["config_id"] == planner.config_id
    assert payload["config"]["n_scenarios"] == planner.n_scenarios
    assert payload["search"]["n_trials"] == planner._n_trials
    assert len(payload["nodes"]) == len(planner._last_tree)

    beliefs = [record for record in payload["nodes"] if record["kind"] == "belief"]
    actions = [record for record in payload["nodes"] if record["kind"] == "action"]
    assert beliefs and actions
    for record in beliefs:
        for field in (
            "depth",
            "weight",
            "lower_bound",
            "upper_bound",
            "default_value",
            "expanded",
            "in_tree",
            "scenario_ids",
            "particles",
        ):
            assert field in record, f"belief record {record['id']} is missing {field}"
        assert len(record["scenario_ids"]) == len(record["particles"])
    for record in actions:
        for field in ("action", "immediate_reward", "q_lower", "lower_bound", "upper_bound"):
            assert field in record, f"action record {record['id']} is missing {field}"


def test_enabling_the_dump_writes_one_file_per_decision():
    """Opt-in dumping, one artifact per decision, numbered in order.

    Purpose: QA needs to point at a specific decision in a specific episode; one
    overwritten file cannot do that.

    Test type: unit
    """
    np.random.seed(53)
    random.seed(53)
    environment = _tiger()
    planner = _planner(environment, n_simulations=5)
    belief = get_initial_belief(environment, n_particles=20)

    with tempfile.TemporaryDirectory() as tmpdir:
        planner.enable_search_state_dump(Path(tmpdir))
        planner.action(belief)
        planner.action(belief)
        planner.disable_search_state_dump()
        planner.action(belief)
        written = sorted(p.name for p in Path(tmpdir).glob("*.json"))

    assert written == [
        "DESPOT_test_decision_0000.json",
        "DESPOT_test_decision_0001.json",
    ], f"got {written}"


def test_exporting_before_any_search_is_an_error_not_an_empty_file():
    """An empty dump would read as "the search found nothing".

    Test type: unit
    """
    planner = _planner()
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="no search state"):
            planner.export_search_state(Path(tmpdir) / "dump.json")
