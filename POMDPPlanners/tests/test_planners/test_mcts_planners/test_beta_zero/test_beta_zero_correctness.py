# SPDX-License-Identifier: MIT

"""BetaZero correctness tests: real terminal handling and exact leaf backups.

``test_beta_zero.py`` has good PUCT and policy-target coverage. Two of its
claims are not backed by their fixtures: the "terminal belief" test admits in a
comment that TigerPOMDP is never terminal and only checks the action is legal,
and the leaf-value test asserts only ``isinstance(value, float)``, which a
random rollout would satisfy just as well as a network.

These tests supply the missing evidence. The immediate reward is pinned by
Tiger's own reward table — ``listen`` costs exactly -1 in every state, so
``belief_expectation_reward`` is -1 for any belief and the backup arithmetic is
hand-checkable without recomputing a belief expectation.

Reference:
    Moss, R. J., Corso, A., Caers, J., & Kochenderfer, M. J. (2024). BetaZero:
    Belief-State Planning for Long-Horizon POMDPs using Learned Approximations.
    Reinforcement Learning Journal.
"""

# pylint: disable=protected-access

import random
from typing import Any

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero import BetaZero
from POMDPPlanners.tests.test_planners.tree_assertions import (
    action_ids,
    assert_values_within_bounds,
    walk_arena_tree,
)
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler


np.random.seed(42)
random.seed(42)

DISCOUNT = 0.5
TOL = 1e-9

# Tiger's reward table: listen costs -1 in every state, open_* is +10 or -100.
LISTEN_REWARD = -1.0
TIGER_REWARD_MIN = -100.0
TIGER_REWARD_MAX = 10.0


class TerminalTigerPOMDP(TigerPOMDP):
    """Tiger with a genuinely terminal state, which the shipped one lacks.

    ``TigerPOMDP.is_terminal`` returns ``False`` unconditionally — its comment
    says termination is handled inside the transition model — so no belief
    built on it can exercise a planner's terminal branch. This subclass makes
    ``tiger_left`` terminal and changes nothing else, which is the smallest
    edit that lets the branch be reached at all.
    """

    def is_terminal(self, state: str) -> bool:
        return state == "tiger_left"


class _StubNetwork:
    """A network with a fixed value head and a recording call counter.

    ``predict`` returns a uniform policy and a constant value, so a leaf backup
    becomes an equation in known numbers. ``calls`` is what proves the terminal
    branch never consults it.
    """

    action_space_type = "discrete"
    n_actions = 3
    action_dim = 1

    def __init__(self, value: float = 6.0) -> None:
        self.value = value
        self.calls: list = []

    def predict(self, features: Any):
        self.calls.append(np.array(features, copy=True))
        return np.full(self.n_actions, 1.0 / self.n_actions), self.value


def _planner(env, *, depth=2, n_simulations=4, network=None, k_o=1.0, alpha_o=0.0, name="bz"):
    planner = BetaZero(
        environment=env,
        discount_factor=DISCOUNT,
        depth=depth,
        name=name,
        action_sampler=DiscreteActionSampler(env.get_actions()),
        n_simulations=n_simulations,
        state_dim=1,
        k_o=k_o,
        alpha_o=alpha_o,
        normalize_inputs=False,
        normalize_values=False,
    )
    if network is not None:
        # The stub deliberately does not subclass the production network: the
        # point is to control its outputs, and only ``predict`` is used here.
        planner.network = network  # type: ignore[assignment]
    return planner


# ---------------------------------------------------------------------------
# Terminal input
# ---------------------------------------------------------------------------


def test_a_genuinely_terminal_belief_runs_no_search_and_never_calls_the_network():
    """A wholly terminal belief short-circuits before the tree or the network.

    Purpose: Replaces the existing "terminal" test, whose fixture is not
        terminal and which only asserts the returned action is legal — a check
        a full search passes just as well as a short-circuit.

    Given: ``TerminalTigerPOMDP``, where ``tiger_left`` is terminal, a belief
        every particle of which is ``tiger_left``, and a stub network that
        records each call.
    When: ``action()`` is called.
    Then: One legal action comes back, ``info_variables`` is empty (the tree
        metrics are only produced by a real search), and the network was never
        consulted — so no tree was built and no leaf was evaluated.

    Test type: unit
    """
    env = TerminalTigerPOMDP(discount_factor=DISCOUNT)
    network = _StubNetwork()
    planner = _planner(env, network=network, name="bz_terminal")
    belief = WeightedParticleBelief(
        particles=["tiger_left", "tiger_left"], log_weights=np.array([-1.0, -1.0])
    )

    actions, run_data = planner.action(belief)

    assert len(actions) == 1 and actions[0] in env.get_actions()
    assert run_data.info_variables == [], (
        f"a terminal belief produced tree metrics {run_data.info_variables}; no search should "
        "have run"
    )
    assert network.calls == [], (
        f"the network was consulted {len(network.calls)} times on a terminal belief; the "
        "terminal branch must return before any search or leaf evaluation"
    )


def test_a_non_terminal_belief_does_run_a_search_and_does_call_the_network():
    """The counterpart, so the terminal check above is not vacuously true.

    Purpose: A short-circuit test proves nothing unless the same planner and
        the same network demonstrably do the opposite on a live belief.

    Given: The same planner and stub network on a non-terminal belief.
    When: ``action()`` is called.
    Then: Tree metrics come back and the network was consulted at least once.

    Test type: unit
    """
    env = TerminalTigerPOMDP(discount_factor=DISCOUNT)
    network = _StubNetwork()
    planner = _planner(env, network=network, name="bz_live")
    belief = WeightedParticleBelief(
        particles=["tiger_right", "tiger_right"], log_weights=np.array([-1.0, -1.0])
    )

    _, run_data = planner.action(belief)

    assert run_data.info_variables, "a live belief must produce tree metrics"
    assert network.calls, "a live belief must reach at least one network leaf evaluation"


# ---------------------------------------------------------------------------
# Exact network-leaf backup
# ---------------------------------------------------------------------------


def test_network_leaf_backup_is_immediate_reward_plus_discounted_network_value():
    """A leaf expansion gives ``Q = r + gamma * V_net``, exactly.

    Purpose: Replaces a test that only asserted the leaf value was a float. The
        numbers here separate the correct backup from a missing discount
        (``-1 + 6 = 5``), a missing immediate reward (``3``) and a rollout used
        in place of the network (anything not equal to 2).

    Given: A stub network whose value head always returns 6.0, normalisation
        off, the ``listen`` action whose immediate reward is Tiger's constant
        -1, and discount 0.5.
    When: ``_simulate_return`` expands a fresh belief child.
    Then: The return is ``-1 + 0.5 * 6 = 2.0``, the action's Q is 2.0 after its
        first sample, and the cached immediate reward on the action node is -1.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    network = _StubNetwork(value=6.0)
    planner = _planner(env, network=network, name="bz_leaf")
    tree = Tree()
    root_id = tree.add_belief_node(get_initial_belief(pomdp=env, n_particles=8))
    action_id = tree.add_action_node(action="listen", parent_id=root_id)

    total = planner._simulate_return(tree=tree, belief_id=root_id, action_id=action_id, depth=0)

    expected = LISTEN_REWARD + DISCOUNT * network.value
    assert expected == 2.0
    assert total == pytest.approx(
        expected, abs=TOL
    ), f"leaf return {total} != {LISTEN_REWARD} + {DISCOUNT} * {network.value}"
    assert tree.get_immediate_reward(action_id) == pytest.approx(LISTEN_REWARD, abs=TOL)
    assert network.calls, "the leaf value must come from the network, not a rollout"

    planner._update_node_statistics(tree=tree, belief_id=root_id, action_id=action_id, total=total)
    assert tree.q_value[action_id] == pytest.approx(expected, abs=TOL)
    assert tree.v_value[root_id] == pytest.approx(expected, abs=TOL)


def test_the_leaf_value_tracks_the_network_output():
    """Changing the network's value changes the backup by exactly gamma times it.

    Purpose: Proves the network output really is the leaf estimate rather than
        a coincidence of one fixture's numbers.

    Given: The same setup with the network value changed from 6.0 to -4.0.
    When: ``_simulate_return`` runs.
    Then: The return is ``-1 + 0.5 * (-4) = -3``.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = _planner(env, network=_StubNetwork(value=-4.0), name="bz_leaf_neg")
    tree = Tree()
    root_id = tree.add_belief_node(get_initial_belief(pomdp=env, n_particles=8))
    action_id = tree.add_action_node(action="listen", parent_id=root_id)

    total = planner._simulate_return(tree=tree, belief_id=root_id, action_id=action_id, depth=0)

    assert total == pytest.approx(LISTEN_REWARD + DISCOUNT * -4.0, abs=TOL)


def test_denormalisation_is_applied_to_the_network_value_when_enabled():
    """With value normalisation on, the leaf value is denormalised before use.

    Purpose: A missing denormalisation is invisible to a type check and shifts
        every backup by the recorded mean.

    Given: A planner with ``normalize_values=True`` whose recorded value mean
        and standard deviation are set to known numbers, and a network whose
        raw output is 2.0.
    When: ``_network_leaf_value`` runs.
    Then: The result is ``2.0 * (std + 1e-8) + mean``. The epsilon is the
        implementation's own guard against a zero standard deviation; it is
        written into the expectation rather than hidden in a loose tolerance,
        so a change to it shows up as a test failure and a decision.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = BetaZero(
        environment=env,
        discount_factor=DISCOUNT,
        depth=2,
        name="bz_denorm",
        action_sampler=DiscreteActionSampler(env.get_actions()),
        n_simulations=2,
        state_dim=1,
        normalize_inputs=False,
        normalize_values=True,
    )
    planner.network = _StubNetwork(value=2.0)  # type: ignore[assignment]
    planner._value_mean = 5.0
    planner._value_std = 3.0

    value = planner._network_leaf_value(get_initial_belief(pomdp=env, n_particles=4))

    expected = 2.0 * (3.0 + 1e-8) + 5.0
    assert value == pytest.approx(
        expected, abs=1e-12
    ), f"denormalised leaf value {value} != 2.0 * (std + 1e-8) + mean = {expected}"


# ---------------------------------------------------------------------------
# Whole search: structure and bounded-leaf value ranges
# ---------------------------------------------------------------------------


def test_full_search_structure_and_bounded_leaf_value_ranges():
    """Structure holds and every value sits inside a bound the *stub* makes finite.

    Purpose: A production BetaZero network has no proved output bound, so a
        range check on it would be meaningless. With a stub whose value head is
        the constant 6.0, the leaf estimate lies in ``[6, 6]`` and the whole
        discounted return becomes boundable — which is the documented way to
        keep this check honest rather than skipping it.

    Given: Tiger (rewards in [-100, 10]), discount 0.5, depth 2, eight
        simulations, a constant-6.0 stub network, fixed seeds.
    When: ``_learn_tree`` builds the tree.
    Then: The walk reaches every allocated node with correct links and CDFs and
        saw at least one visited action; and every visited Q and every V lies
        inside the interval a three-term discounted sum of rewards in
        [-100, 10] with a leaf in [6, 6] allows.

    Test type: unit
    """
    np.random.seed(2024)
    random.seed(2024)
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = _planner(
        env,
        depth=2,
        n_simulations=8,
        network=_StubNetwork(value=6.0),
        k_o=2.0,
        alpha_o=0.5,
        name="bz_structure",
    )

    tree, root_id = planner._learn_tree(belief=get_initial_belief(pomdp=env, n_particles=10))

    counters = walk_arena_tree(tree, root_id)
    assert counters.visited_action_nodes >= 1
    assert counters.action_nodes >= 1

    def horizon_of(node_id: int, edge_depth: int):
        del node_id
        return max(planner.depth + 1 - edge_depth // 2, 0)

    checked = assert_values_within_bounds(
        tree,
        root_id,
        horizon_of=horizon_of,
        reward_min=TIGER_REWARD_MIN,
        reward_max=TIGER_REWARD_MAX,
        discount=DISCOUNT,
        leaf_min=6.0,
        leaf_max=6.0,
    )
    assert checked >= 2, f"only {checked} values were range-checked"


def test_root_actions_carry_legal_actions_and_the_declared_metrics_come_back():
    """Public contract: one legal action and exactly the declared metric names.

    Test type: unit
    """
    np.random.seed(8)
    random.seed(8)
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = _planner(env, n_simulations=6, network=_StubNetwork(), name="bz_contract")

    actions, run_data = planner.action(get_initial_belief(pomdp=env, n_particles=8))

    assert len(actions) == 1 and actions[0] in env.get_actions()
    assert [v.name for v in run_data.info_variables] == BetaZero.get_info_variable_names()


def test_configuration_identity_distinguishes_algorithm_parameters():
    """Equal configurations share a config ID; a changed depth does not.

    Purpose: The config ID is the simulation cache key.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    first = _planner(env, depth=2, n_simulations=4, name="bz_cfg")
    same = _planner(env, depth=2, n_simulations=4, name="bz_cfg")
    deeper = _planner(env, depth=5, n_simulations=4, name="bz_cfg")

    assert first.config_id == same.config_id
    assert first.config_id != deeper.config_id


def test_repeated_calls_keep_the_first_snapshot_and_do_not_accumulate_visits():
    """A second ``action()`` builds a fresh tree and leaves the first data alone.

    Test type: unit
    """
    np.random.seed(21)
    random.seed(21)
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = _planner(env, n_simulations=4, network=_StubNetwork(), name="bz_reset")
    belief = get_initial_belief(pomdp=env, n_particles=8)

    _, first = planner.action(belief)
    snapshot = [(v.name, v.value) for v in first.info_variables]
    _, second = planner.action(belief)

    assert [(v.name, v.value) for v in first.info_variables] == snapshot
    root_visits = dict((v.name, v.value) for v in second.info_variables)["root_visit_count"]
    assert (
        root_visits == 4
    ), f"second call reports {root_visits} root visits; each call must start a fresh tree"
