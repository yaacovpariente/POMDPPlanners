# SPDX-License-Identifier: MIT

"""AdaOPS public contract and search-equation tests."""

# pylint: disable=protected-access

import copy
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.core.tree.arena import ACTION, BELIEF
from POMDPPlanners.planners import POLICY_REGISTRY, AdaOPS
from POMDPPlanners.planners.scenario_tree_planners.adaops import AdaOPSMetrics
from POMDPPlanners.tests.test_planners.planner_fixtures import ROOT, ChainEnv, chain_belief
from POMDPPlanners.tests.test_planners.test_scenario_tree_planners.test_despot_correctness import (
    CoinEnv,
)
from POMDPPlanners.tests.test_simulations.test_planner_metrics_persistence import (
    _history_with_metrics,
)
from POMDPPlanners.core.simulation.history import History
from POMDPPlanners.utils.tree_statistics import TreeMetrics


def state_bin(state):
    """Importable bin function used by the JSON round-trip test."""
    return state


def _planner(environment=None, **overrides):
    environment = environment or ChainEnv(discount_factor=0.5)
    params = dict(
        environment=environment,
        discount_factor=environment.discount_factor,
        depth=3,
        name="AdaOPS_test",
        min_particles=2,
        max_particles=8,
        n_simulations=4,
        random_seed=7,
    )
    params.update(overrides)
    return AdaOPS(**params)


def test_registration_metrics_and_alternating_topology_are_exercised():
    planner = _planner()
    actions, run_data = planner.action(chain_belief(ROOT))
    tree = planner._last_tree
    assert POLICY_REGISTRY["AdaOPS"] is AdaOPS
    assert actions[0] in planner.environment.get_actions()
    assert tree is not None and len(tree) > 3
    assert any(
        tree.kind[node] == BELIEF and tree.parent_id[node] is not None for node in range(len(tree))
    )
    assert any(tree.kind[node] == ACTION for node in range(len(tree)))
    for node in range(len(tree)):
        parent = tree.parent_id[node]
        if parent is not None:
            assert tree.kind[parent] != tree.kind[node], f"node {node} does not alternate kind"
            assert node in tree.children_ids[parent]
    names = [item.name for item in run_data.info_variables]
    assert len(names) == len(set(names))
    assert set(names) == set(AdaOPS.get_info_variable_names())
    assert {metric.value for metric in TreeMetrics} <= set(names)
    assert {metric.value for metric in AdaOPSMetrics} <= set(names)
    assert all(
        isinstance(item.value, (int, float)) and math.isfinite(item.value)
        for item in run_data.info_variables
    )


def test_bounds_backup_discount_and_final_choice_use_lower_bound():
    planner = _planner(n_simulations=1)
    tree, root = planner._learn_tree(chain_belief(ROOT))
    action_ids = tree.children_ids[root]
    assert len(action_ids) == 2
    for action_id in action_ids:
        # Chain reward is 2 now and 4 next; gamma=.5, so Q=4.
        assert tree.lower_confidence_bound[action_id] == pytest.approx(4.0, abs=1e-12)
        assert tree.upper_confidence_bound[action_id] >= tree.lower_confidence_bound[action_id]
    tree.lower_confidence_bound[action_ids[0]] = 1.0
    tree.upper_confidence_bound[action_ids[0]] = 100.0
    tree.lower_confidence_bound[action_ids[1]] = 2.0
    tree.upper_confidence_bound[action_ids[1]] = 3.0
    assert planner._action_of_best_lower_bound(tree, root) == tree.action[action_ids[1]]


def test_root_kld_sampling_obeys_particle_caps_even_for_uniform_input():
    planner = _planner(min_particles=2, max_particles=5, n_simulations=0)
    belief = chain_belief(ROOT)
    belief.particles = [ROOT] * 20
    tree, root = planner._learn_tree(belief)
    assert len(tree.get_belief(root).particles) == 5
    assert tree.data[root].weights.tolist() == pytest.approx([0.2] * 5)


def test_excess_uncertainty_weights_probability_and_discount():
    planner = _planner(n_simulations=1, xi=0.5)
    tree, root = planner._learn_tree(chain_belief(ROOT))
    action_id = tree.children_ids[root][0]
    child = tree.children_ids[action_id][0]
    tree.lower_confidence_bound[root] = 0.0
    tree.upper_confidence_bound[root] = 4.0
    tree.lower_confidence_bound[child] = 0.0
    tree.upper_confidence_bound[child] = 10.0
    selected, value = planner._best_excess_uncertainty_child(tree, action_id)
    expected = (tree.weight[child] / tree.weight[root]) * (10.0 - 0.5 * 4.0 / 0.5)
    assert selected == child
    assert value == pytest.approx(expected, abs=1e-12)


def test_packing_merges_at_delta_and_preserves_action_probability_mass():
    env = CoinEnv(discount_factor=0.5)
    planner = _planner(env, delta=2.0, min_particles=4, max_particles=4, n_simulations=1)
    belief = chain_belief(("start", 0))
    # Use four particles so both generated observations are overwhelmingly likely.
    belief.particles = [("start", 0)] * 4
    tree, root = planner._learn_tree(belief)
    exercised = 0
    for action_id in tree.children_ids[root]:
        children = tree.children_ids[action_id]
        if planner._merged_branches:
            exercised += 1
            assert len(children) == 1
            assert tree.weight[children[0]] == pytest.approx(tree.weight[root], abs=1e-12)
            assert tree.data[children[0]].observation_probability == pytest.approx(1.0, abs=1e-12)
    assert exercised > 0, "seed did not exercise a packed multi-observation action"


def test_configuration_identity_reset_pickle_and_snapshot_immutability(tmp_path: Path):
    planner = _planner()
    same = _planner()
    changed = _planner(delta=0.2)
    assert planner.config_id == same.config_id
    assert planner.config_id != changed.config_id
    with pytest.raises(ValueError, match="state_binner_id"):
        _planner(state_binner=lambda state: state)

    planner.action(chain_belief(ROOT))
    first = planner.get_last_search_state()
    frozen = copy.deepcopy(first)
    planner.action(chain_belief(ROOT))
    assert first == frozen
    first["nodes"][0]["weights"][0] = -1.0
    assert planner.get_last_search_state()["nodes"][0]["weights"][0] >= 0.0

    restored = pickle.loads(pickle.dumps(planner))
    assert restored.config_id == planner.config_id
    assert restored.action(chain_belief(ROOT))[0][0] in restored.environment.get_actions()

    config_path = tmp_path / "planner.json"
    planner.save(config_path)
    loaded = AdaOPS.load(config_path)
    assert loaded.config_id == planner.config_id
    assert loaded.action(chain_belief(ROOT))[0][0] in loaded.environment.get_actions()

    binned = _planner(
        state_binner=state_bin,
        state_binner_id=(
            "POMDPPlanners.tests.test_planners.test_scenario_tree_planners." "test_adaops.state_bin"
        ),
    )
    binned_path = tmp_path / "binned.json"
    binned.save(binned_path)
    loaded_binned = AdaOPS.load(binned_path)
    assert loaded_binned.config_id == binned.config_id
    assert loaded_binned._state_binner(ROOT) == ROOT

    path = tmp_path / "search.json"
    planner.export_search_state(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["selected_action"]
    assert all("lower_bound" in node and "upper_bound" in node for node in payload["nodes"])
    assert any("particles" in node and "weights" in node for node in payload["nodes"])


def test_budget_boundaries_and_metrics_reset_between_calls():
    zero = _planner(n_simulations=0)
    actions, run_data = zero.action(chain_belief(ROOT))
    assert actions[0] in zero.environment.get_actions()
    metrics = {item.name: item.value for item in run_data.info_variables}
    assert metrics[AdaOPSMetrics.N_TRIALS.value] == 0
    assert metrics[AdaOPSMetrics.STOPPED_BY_TRIALS.value] == 1

    planner = _planner(n_simulations=2)
    _, first = planner.action(chain_belief(ROOT))
    _, second = planner.action(chain_belief(ROOT))
    first_metrics = {item.name: item.value for item in first.info_variables}
    second_metrics = {item.name: item.value for item in second.info_variables}
    assert first_metrics[AdaOPSMetrics.N_TRIALS.value] <= 2
    assert second_metrics[AdaOPSMetrics.N_TRIALS.value] <= 2
    assert (
        second_metrics[AdaOPSMetrics.N_TRIALS.value]
        != first_metrics[AdaOPSMetrics.N_TRIALS.value] + 2
    )


def test_metrics_survive_the_simulation_history_record_shape():
    planner = _planner(n_simulations=2)
    _, run_data = planner.action(chain_belief(ROOT))
    restored = History.from_dict(_history_with_metrics(policy_run_data=[run_data]).to_dict())
    original = {item.name: item.value for item in run_data.info_variables}
    persisted = {item.name: item.value for item in restored.policy_run_data[0].info_variables}
    assert persisted == original
    assert set(persisted) == set(AdaOPS.get_info_variable_names())
