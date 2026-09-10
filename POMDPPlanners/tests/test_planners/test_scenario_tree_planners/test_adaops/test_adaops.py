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
from typing import cast

from POMDPPlanners.core.tree.arena import ACTION, BELIEF
from POMDPPlanners.planners import POLICY_REGISTRY, AdaOPS
from POMDPPlanners.planners.scenario_tree_planners.adaops import AdaOPSMetrics
from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.tests.test_planners.planner_fixtures import (
    END,
    NEXT,
    ROOT,
    ChainEnv,
    chain_belief,
)
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
    assert actions[0] in cast(ChainEnv, planner.environment).get_actions()
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
    assert len(cast(WeightedParticleBelief, tree.get_belief(root)).particles) == 5
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
    assert (
        loaded.action(chain_belief(ROOT))[0][0] in cast(ChainEnv, loaded.environment).get_actions()
    )

    binned = _planner(
        state_binner=state_bin,
        state_binner_id=(
            "POMDPPlanners.tests.test_planners.test_scenario_tree_planners.test_adaops."
            "test_adaops.state_bin"
        ),
    )
    binned_path = tmp_path / "binned.json"
    binned.save(binned_path)
    loaded_binned = AdaOPS.load(binned_path)
    assert loaded_binned.config_id == binned.config_id
    assert cast(AdaOPS, loaded_binned)._state_binner(ROOT) == ROOT

    path = tmp_path / "search.json"
    planner.export_search_state(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["selected_action"]
    assert all("lower_bound" in node and "upper_bound" in node for node in payload["nodes"])
    assert any("particles" in node and "weights" in node for node in payload["nodes"])


def test_budget_boundaries_and_metrics_reset_between_calls():
    zero = _planner(n_simulations=0)
    actions, run_data = zero.action(chain_belief(ROOT))
    assert actions[0] in cast(ChainEnv, zero.environment).get_actions()
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


# ---------------------------------------------------------------------------
# Regression tests for the 2026-09-08 review findings
# ---------------------------------------------------------------------------


class _SamplerOnlyBelief:
    """A belief that can only be sampled from -- no ``particles`` list.

    This is the branch ``_root_particles`` falls back to, and the one no
    existing fixture reached, which is how the global-RNG leak stayed hidden.
    """

    particles = None

    def __init__(self, state=ROOT):
        self._state = state
        self.samples = 0

    def sample(self, n_samples: int = 1):
        del n_samples
        self.samples += 1
        # The particle carries the uniform it was drawn with, so a test can see
        # which random stream the draw actually ran on.
        return (self._state, round(float(np.random.random()), 12))


def test_particle_less_belief_does_not_leak_into_the_global_rng():
    """Sampling a belief with no particle list must restore global state."""
    np.random.seed(1234)
    import random as _random

    _random.seed(1234)
    expected_np = np.random.get_state()[1].copy()
    expected_pos = np.random.get_state()[2]
    expected_py = _random.getstate()
    untouched_next_draw = float(np.random.random())
    np.random.seed(1234)
    _random.seed(1234)

    planner = _planner()
    planner._call_index = 0
    planner._particle_rng = np.random.default_rng(planner.random_seed)
    planner._scenario_rng = planner._particle_rng
    belief = _SamplerOnlyBelief()
    particles, weights = planner._root_particles(belief)

    assert belief.samples == planner.max_particles
    assert len(particles) == len(weights) >= planner.min_particles
    # Compare the whole Mersenne state, position included: a restore that put
    # back the key but not the position would pass a prefix-only check.
    assert np.random.get_state()[1].tolist() == expected_np.tolist()
    assert np.random.get_state()[2] == expected_pos
    assert _random.getstate() == expected_py, "python random stream was perturbed"
    # The decisive check: the caller's next draw is the one it would have got
    # had the planner never run.
    np.random.seed(1234)
    _random.seed(1234)
    assert float(np.random.random()) == pytest.approx(untouched_next_draw)


def test_particle_less_root_depends_on_the_planner_seed_not_the_caller_stream():
    """The root draw must come from the planner's own generator.

    The two calls below differ only in the caller's global numpy seed. If the
    draw still ran on the unseeded global stream the particles would track that
    caller seed instead of ``random_seed``.
    """

    def draw(planner_seed, caller_seed):
        planner = _planner(random_seed=planner_seed)
        planner._call_index = 0
        planner._particle_rng = np.random.default_rng(planner_seed)
        planner._scenario_rng = planner._particle_rng
        np.random.seed(caller_seed)
        return planner._root_particles(_SamplerOnlyBelief())[0]

    assert draw(11, 999) == draw(11, 4242), "root particles tracked the caller's RNG"
    assert draw(11, 999) != draw(31, 999), "root particles ignored the planner seed"


class _TerminalStringObsEnv(ChainEnv):
    """A ChainEnv whose real observation for ``next`` is the text ``<terminal>``.

    ``END`` is terminal, so one particle takes AdaOPS's terminal branch while
    the other emits an observation whose *string* value used to be the terminal
    bucket's key. The two must stay separate branches.
    """

    TEXT = "<terminal>"

    def sample_observation(self, next_state, action, n_samples: int = 1):
        del action
        observation = self.TEXT if next_state == END else next_state
        return observation if n_samples == 1 else [observation] * n_samples

    def observation_log_probability(self, next_state, action, observations):
        del action
        expected = self.TEXT if next_state == END else next_state
        return np.array(
            [0.0 if obs == expected else -50.0 for obs in observations], dtype=np.float64
        )


def test_terminal_bucket_does_not_swallow_a_matching_string_observation():
    """A real ``"<terminal>"`` observation must not merge into the terminal
    bucket (review finding 6)."""
    from POMDPPlanners.planners.scenario_tree_planners.despot import TERMINAL_OBSERVATION

    environment = _TerminalStringObsEnv(discount_factor=0.5, terminal_states=(END,))
    planner = _planner(environment=environment, n_simulations=1)
    # One terminal particle and one live particle whose observation is the text.
    belief = WeightedParticleBelief(particles=[END, NEXT], log_weights=np.array([-1.0, -1.0]))
    tree, root = planner._learn_tree(belief)

    action_ids = [node for node in tree.children_ids[root]]
    assert action_ids
    for action_id in action_ids:
        keys = [key for (parent, key) in tree.obs_child_lookup if parent == action_id]
        assert TERMINAL_OBSERVATION in keys, "terminal scenarios lost their own branch"
        assert (
            _TerminalStringObsEnv.TEXT in keys
        ), "the real string observation was merged into the terminal bucket"
        assert len(set(keys)) == 2
