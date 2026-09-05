# SPDX-License-Identifier: MIT

"""ConstrainedZero correctness: the public safe action, and real recursion.

``test_constrained_zero.py`` covers SPUCT selection, the safety mask and the
compound-failure helper well. What it does not cover is what the planner
actually hands back. Search selection and final selection are separate
contracts, and until the fix these tests accompany, ConstrainedZero searched
under a safety mask and then answered with the base class's reward-only
``argmax`` over Q — so a search that carefully avoided an unsafe action could
still return it.

The compound-failure test here also runs through ``_simulate_return_constrained``
rather than calling the arithmetic helper directly, so the recursion's use of
it is what is checked, not the formula in isolation.

Reference:
    Moss, R. J., Jamgochian, A., Fischer, J., Corso, A., & Kochenderfer, M. J.
    (2024). ConstrainedZero: Chance-Constrained POMDP Planning Using Learned
    Probabilistic Failure Surrogates and Adaptive Safety Constraints. IJCAI 33,
    6752-6760.
"""

# pylint: disable=protected-access

import random
from typing import Any

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_zero import (
    ConstrainedZero,
)
from POMDPPlanners.tests.test_planners.tree_assertions import (
    assert_values_within_bounds,
    walk_arena_tree,
)
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler


np.random.seed(42)
random.seed(42)

DISCOUNT = 0.5
TOL = 1e-9
LISTEN_REWARD = -1.0
TIGER_REWARD_MIN = -100.0
TIGER_REWARD_MAX = 10.0


def _no_failure(_state: Any) -> bool:
    return False


def _tiger_left_fails(state: Any) -> bool:
    return state == "tiger_left"


class _StubNetwork:
    """Three-head stub: uniform policy, constant value, constant failure."""

    action_space_type = "discrete"
    n_actions = 3
    action_dim = 1

    def __init__(self, value: float = 0.0, failure: float = 0.0) -> None:
        self.value = value
        self.failure = failure
        self.calls: list = []

    def predict(self, features: Any):
        self.calls.append(np.array(features, copy=True))
        return np.full(self.n_actions, 1.0 / self.n_actions), self.value, self.failure


def _planner(
    env,
    *,
    failure_fn=_no_failure,
    delta_0=0.1,
    delta_compounding=1.0,
    depth=2,
    n_simulations=4,
    network=None,
    name="cz",
):
    planner = ConstrainedZero(
        environment=env,
        discount_factor=DISCOUNT,
        depth=depth,
        name=name,
        action_sampler=DiscreteActionSampler(env.get_actions()),
        failure_fn=failure_fn,
        delta_0=delta_0,
        delta_compounding=delta_compounding,
        n_simulations=n_simulations,
        state_dim=1,
        normalize_inputs=False,
        normalize_values=False,
    )
    if network is not None:
        # The stub deliberately does not subclass the production network: the
        # point is to control its outputs, and only ``predict`` is used here.
        planner.network = network  # type: ignore[assignment]
    return planner


# ---------------------------------------------------------------------------
# The public final action
# ---------------------------------------------------------------------------


def test_the_returned_action_is_the_safe_one_even_when_an_unsafe_action_has_a_higher_q():
    """The action handed back respects the same safety threshold the search used.

    Purpose: This is the contract the audit flagged as untested and which the
        base class did not honour. It fails against a reward-only final
        selection, which is exactly the behaviour this test exists to forbid.

    Given: A root with two action children. ``open_left`` has the higher value
        (Q = 10) but an estimated failure probability of 0.9, far above the
        threshold; ``listen`` has the lower value (Q = 1) and a failure
        probability of 0.0, comfortably under it. Delta_0 is 0.1.
    When: ``_select_final_action`` chooses the action to return.
    Then: ``listen`` comes back. A reward-only rule returns ``open_left``.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = _planner(env, delta_0=0.1, name="cz_final_safe")
    tree = Tree()
    root_id = tree.add_belief_node(get_initial_belief(pomdp=env, n_particles=4))
    unsafe_id = tree.add_action_node(action="open_left", parent_id=root_id)
    safe_id = tree.add_action_node(action="listen", parent_id=root_id)
    tree.visit_count[unsafe_id] = 5
    tree.q_value[unsafe_id] = 10.0
    tree.visit_count[safe_id] = 5
    tree.q_value[safe_id] = 1.0
    planner._failure_dict[unsafe_id] = 0.9
    planner._failure_dict[safe_id] = 0.0

    chosen = planner._select_final_action(tree=tree, root_id=root_id)

    assert chosen == "listen", (
        f"the planner returned {chosen!r}; open_left has the higher Q but an estimated failure "
        "probability of 0.9 against a threshold of 0.1, so a safety-aware final selection must "
        "return listen"
    )
    assert (
        tree.best_action_by_reward(root_id) == "open_left"
    ), "the fixture must make the reward-only rule disagree, or the test proves nothing"


class _ScriptedConstrainedZero(ConstrainedZero):
    """ConstrainedZero whose per-simulation return and failure are scripted.

    Overriding ``_simulate_return_constrained`` — not the selection rule — is
    what makes this a black-box test of the public answer. The search runs for
    real: it widens actions, backs values up, averages the failure estimates
    and adapts delta exactly as it always does. Only the numbers each
    simulation reports are fixed, so the tree ends with ``listen`` safe and
    unattractive and every other action unsafe and attractive.
    """

    SAFE_ACTION = "listen"
    SAFE_RETURN = -10.0
    UNSAFE_RETURN = 10.0

    def _simulate_return_constrained(self, tree, belief_id, action_id, depth):
        del belief_id, depth
        if tree.get_action(action_id) == self.SAFE_ACTION:
            return self.SAFE_RETURN, 0.0
        return self.UNSAFE_RETURN, 1.0


def test_the_full_public_action_call_returns_the_safe_action():
    """Through ``action()`` alone, the planner answers with the safe action.

    Purpose: This is the black-box statement of the contract, and it is the
        test that fails against a reward-only final selection rather than
        merely reporting a missing method. Nothing here mentions the selection
        rule; only ``action()`` is called and only the returned action is
        checked.

    Given: A ConstrainedZero whose simulations report a return of -10 and a
        failure probability of 0 for ``listen``, and +10 with a failure
        probability of 1 for every other action, against a threshold
        ``delta_0 = 0.1``. The search therefore ends with ``listen`` holding
        the *worst* value and the only acceptable failure estimate.
    When: ``action()`` is called.
    Then: ``listen`` comes back.

    A reward-only final selection returns ``open_left`` or ``open_right``,
    whose Q is +10 — which is exactly the behaviour the safety-aware final
    selection exists to prevent.

    Test type: unit
    """
    np.random.seed(19)
    random.seed(19)
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = _ScriptedConstrainedZero(
        environment=env,
        discount_factor=DISCOUNT,
        depth=2,
        name="cz_public_blackbox",
        action_sampler=DiscreteActionSampler(env.get_actions()),
        failure_fn=_no_failure,
        delta_0=0.1,
        n_simulations=12,
        state_dim=1,
        normalize_inputs=False,
        normalize_values=False,
    )
    planner.network = _StubNetwork()  # type: ignore[assignment]

    actions, run_data = planner.action(get_initial_belief(pomdp=env, n_particles=8))

    assert run_data.info_variables, "the search did not run, so nothing was selected from"
    assert actions[0] == _ScriptedConstrainedZero.SAFE_ACTION, (
        f"action() returned {actions[0]!r}; every alternative was estimated to fail with "
        "probability 1 against a threshold of 0.1 while carrying the higher value, so only a "
        "reward-only final selection picks one of them"
    )


def test_the_scripted_fixture_really_does_make_reward_and_safety_disagree():
    """The fixture above is only meaningful if the two rules disagree on it.

    Purpose: Without this, the safe-action test could pass because ``listen``
        happened to have the highest Q anyway.

    Given: The same scripted planner and search.
    When: The tree is built directly.
    Then: The reward-only rule picks a *different* action, and every action it
        might pick has a failure estimate above the threshold.

    Test type: unit
    """
    np.random.seed(19)
    random.seed(19)
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = _ScriptedConstrainedZero(
        environment=env,
        discount_factor=DISCOUNT,
        depth=2,
        name="cz_public_blackbox_control",
        action_sampler=DiscreteActionSampler(env.get_actions()),
        failure_fn=_no_failure,
        delta_0=0.1,
        n_simulations=12,
        state_dim=1,
        normalize_inputs=False,
        normalize_values=False,
    )
    planner.network = _StubNetwork()  # type: ignore[assignment]

    tree, root_id = planner._learn_tree(belief=get_initial_belief(pomdp=env, n_particles=8))

    reward_only = tree.best_action_by_reward(root_id)
    assert reward_only != _ScriptedConstrainedZero.SAFE_ACTION, (
        f"the reward-only rule already picks {reward_only!r}, so the safety test above would "
        "pass even with no safety-aware selection at all"
    )
    unsafe_ids = [
        cid
        for cid in tree.get_children_ids(root_id)
        if tree.get_action(cid) != _ScriptedConstrainedZero.SAFE_ACTION
    ]
    assert unsafe_ids, "the search expanded no unsafe action"
    for cid in unsafe_ids:
        assert planner._failure_dict.get(cid, 0.0) > planner._get_delta_prime(root_id), (
            f"action {tree.get_action(cid)!r} has failure estimate "
            f"{planner._failure_dict.get(cid, 0.0)} at or below the threshold "
            f"{planner._get_delta_prime(root_id)}, so it is not actually unsafe"
        )


def test_when_every_action_is_unsafe_the_best_value_action_is_returned():
    """The documented all-unsafe fallback: the mask is dropped, not the answer.

    Purpose: The safety mask helper falls back to unconstrained selection when
        nothing qualifies; the final selection reuses that same helper rather
        than inventing a different rule. Without this the planner would have no
        action to return at all.

    Given: Two actions, both with failure probability 0.9 against a threshold
        of 0.1, with Q values 1 and 10.
    When: ``_select_final_action`` runs.
    Then: The Q = 10 action is returned.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = _planner(env, delta_0=0.1, name="cz_all_unsafe")
    tree = Tree()
    root_id = tree.add_belief_node(get_initial_belief(pomdp=env, n_particles=4))
    low_id = tree.add_action_node(action="listen", parent_id=root_id)
    high_id = tree.add_action_node(action="open_left", parent_id=root_id)
    tree.q_value[low_id] = 1.0
    tree.q_value[high_id] = 10.0
    planner._failure_dict[low_id] = 0.9
    planner._failure_dict[high_id] = 0.9

    assert planner._select_final_action(tree=tree, root_id=root_id) == "open_left"


def test_a_negative_q_safe_action_still_beats_an_excluded_unsafe_one():
    """Masking excludes an action; it does not rescale its value to zero.

    Purpose: A mask applied by multiplication would turn an unsafe action's
        Q = -100 into 0 and let it win the maximum over a safe action at
        Q = -5. This fixture makes that mistake visible.

    Given: A safe action at Q = -5 and an unsafe action at Q = -100.
    When: ``_select_final_action`` runs.
    Then: The safe action is returned.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = _planner(env, delta_0=0.1, name="cz_negative")
    tree = Tree()
    root_id = tree.add_belief_node(get_initial_belief(pomdp=env, n_particles=4))
    safe_id = tree.add_action_node(action="listen", parent_id=root_id)
    unsafe_id = tree.add_action_node(action="open_left", parent_id=root_id)
    tree.q_value[safe_id] = -5.0
    tree.q_value[unsafe_id] = -100.0
    planner._failure_dict[safe_id] = 0.0
    planner._failure_dict[unsafe_id] = 0.9

    assert planner._select_final_action(tree=tree, root_id=root_id) == "listen"


def test_the_base_class_default_final_selection_is_unchanged():
    """Adding the hook did not change what every other arena planner returns.

    Purpose: The hook was added to the shared base. Its default must be exactly
        the previous behaviour, or every reward-maximising planner silently
        changes.

    Given: An arena tree with a clear best-Q action, and a plain POMCP.
    When: The base class's ``_select_final_action`` runs.
    Then: It returns the same action as ``tree.best_action_by_reward``.

    Test type: unit
    """
    from POMDPPlanners.planners.mcts_planners.pomcp import POMCP

    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = POMCP(
        environment=env,
        discount_factor=DISCOUNT,
        depth=2,
        exploration_constant=1.0,
        name="pomcp_default_final",
        n_simulations=2,
    )
    tree = Tree()
    root_id = tree.add_belief_node(get_initial_belief(pomdp=env, n_particles=4))
    low_id = tree.add_action_node(action="listen", parent_id=root_id)
    high_id = tree.add_action_node(action="open_left", parent_id=root_id)
    tree.q_value[low_id] = 1.0
    tree.q_value[high_id] = 10.0

    assert planner._select_final_action(tree=tree, root_id=root_id) == "open_left"
    assert planner._select_final_action(tree=tree, root_id=root_id) == tree.best_action_by_reward(
        root_id
    )


# ---------------------------------------------------------------------------
# Compound failure, through the real recursion
# ---------------------------------------------------------------------------


def test_compound_failure_arithmetic_runs_through_the_actual_recursion():
    """``p = p_imm + c * (1 - p_imm) * p_next`` is applied by the recursion.

    Purpose: The existing suite tests ``_compound_failure`` directly. This
        drives the same arithmetic through ``_simulate_return_constrained``,
        which is where a wrong argument order or a dropped ``(1 - p_imm)``
        factor would actually bite.

    Given: A belief every particle of which fails, wrapped so the immediate
        estimate is 0.2 rather than 1.0, a network whose failure head returns
        0.3, and ``delta_compounding = 1``.
    When: ``_simulate_return_constrained`` expands a leaf.
    Then: The failure estimate is ``0.2 + 1 * (1 - 0.2) * 0.3 = 0.44``.
        Adding them gives 0.5, multiplying gives 0.06, and taking the maximum
        gives 0.3, so the fixture separates all four.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    network = _StubNetwork(value=0.0, failure=0.3)
    planner = _planner(env, network=network, delta_compounding=1.0, name="cz_compound")
    planner._estimate_belief_failure_prob = lambda belief: 0.2  # type: ignore[method-assign]
    tree = Tree()
    root_id = tree.add_belief_node(get_initial_belief(pomdp=env, n_particles=4))
    action_id = tree.add_action_node(action="listen", parent_id=root_id)

    total, failure = planner._simulate_return_constrained(
        tree=tree, belief_id=root_id, action_id=action_id, depth=0
    )

    assert failure == pytest.approx(
        0.44, abs=TOL
    ), f"compound failure {failure} != 0.2 + (1 - 0.2) * 0.3 = 0.44"
    assert total == pytest.approx(
        LISTEN_REWARD + DISCOUNT * 0.0, abs=TOL
    ), "the reward channel must be unaffected by the failure channel"


def test_delta_compounding_zero_keeps_only_the_immediate_failure():
    """With ``delta_compounding = 0`` the future term drops out entirely.

    Purpose: Pins the coefficient's role, so the parameter cannot become a
        no-op unnoticed.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = _planner(
        env, network=_StubNetwork(failure=0.3), delta_compounding=0.0, name="cz_compound0"
    )
    planner._estimate_belief_failure_prob = lambda belief: 0.2  # type: ignore[method-assign]
    tree = Tree()
    root_id = tree.add_belief_node(get_initial_belief(pomdp=env, n_particles=4))
    action_id = tree.add_action_node(action="listen", parent_id=root_id)

    _, failure = planner._simulate_return_constrained(
        tree=tree, belief_id=root_id, action_id=action_id, depth=0
    )

    assert failure == pytest.approx(0.2, abs=TOL)


def test_action_failure_estimate_is_a_running_mean_over_its_samples():
    """The per-action failure estimate averages its samples incrementally.

    Given: An action node updated with failure samples 0.0 then 1.0.
    When: ``_update_action_failure`` runs after each visit increment.
    Then: The estimate is 0.0 then 0.5.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = _planner(env, name="cz_failure_mean")
    tree = Tree()
    root_id = tree.add_belief_node(get_initial_belief(pomdp=env, n_particles=4))
    action_id = tree.add_action_node(action="listen", parent_id=root_id)

    tree.visit_count[action_id] = 1
    planner._update_action_failure(tree=tree, action_id=action_id, failure=0.0)
    assert planner._failure_dict[action_id] == pytest.approx(0.0, abs=TOL)

    tree.visit_count[action_id] = 2
    planner._update_action_failure(tree=tree, action_id=action_id, failure=1.0)
    assert planner._failure_dict[action_id] == pytest.approx(0.5, abs=TOL)


def test_failure_and_delta_state_are_cleared_between_calls():
    """The per-search failure and delta dictionaries are keyed by node ID.

    Purpose: Both dictionaries are indexed by integers that mean nothing
        outside one tree. Carrying them over would apply one search's failure
        estimates to another search's unrelated nodes.

    Given: A planner whose dictionaries have been filled with stale entries.
    When: ``_learn_tree`` starts a new search.
    Then: Neither dictionary contains the stale key afterwards.

    Test type: unit
    """
    np.random.seed(77)
    random.seed(77)
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = _planner(env, n_simulations=4, network=_StubNetwork(), name="cz_reset")
    planner._failure_dict[9999] = 0.7
    planner._delta_dict[9999] = 0.7

    planner._learn_tree(belief=get_initial_belief(pomdp=env, n_particles=8))

    assert 9999 not in planner._failure_dict
    assert 9999 not in planner._delta_dict


# ---------------------------------------------------------------------------
# Structure, bounds, contract
# ---------------------------------------------------------------------------


def test_full_search_structure_and_bounded_value_ranges():
    """Structure holds and every value is inside a stub-bounded interval.

    Given: Tiger (rewards in [-100, 10]), discount 0.5, depth 2, eight
        simulations, a stub network whose value head is the constant 0.0.
    When: ``_learn_tree`` builds the tree.
    Then: The walk reaches every allocated node and saw a visited action; every
        visited Q and every V is inside the derived interval; and every
        recorded failure estimate is a probability in [0, 1].

    Test type: unit
    """
    np.random.seed(606)
    random.seed(606)
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = _planner(
        env,
        depth=2,
        n_simulations=8,
        failure_fn=_tiger_left_fails,
        network=_StubNetwork(value=0.0, failure=0.25),
        name="cz_structure",
    )

    tree, root_id = planner._learn_tree(belief=get_initial_belief(pomdp=env, n_particles=10))

    counters = walk_arena_tree(tree, root_id)
    assert counters.visited_action_nodes >= 1

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
        leaf_min=0.0,
        leaf_max=0.0,
    )
    assert checked >= 2

    assert planner._failure_dict, "no failure estimate was recorded, so the check below is vacuous"
    for action_id, probability in planner._failure_dict.items():
        assert (
            0.0 <= probability <= 1.0
        ), f"action {action_id} has failure estimate {probability}, which is not a probability"


def test_action_returns_one_legal_action_and_the_declared_metrics():
    """Public contract: one legal action plus exactly the declared metric names.

    Test type: unit
    """
    np.random.seed(12)
    random.seed(12)
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = _planner(env, n_simulations=6, network=_StubNetwork(), name="cz_contract")

    actions, run_data = planner.action(get_initial_belief(pomdp=env, n_particles=8))

    assert len(actions) == 1 and actions[0] in env.get_actions()
    assert [v.name for v in run_data.info_variables] == ConstrainedZero.get_info_variable_names()


def test_configuration_identity_distinguishes_the_safety_threshold():
    """Two planners differing only in delta_0 must not share a cache key.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    first = _planner(env, delta_0=0.1, name="cz_cfg")
    same = _planner(env, delta_0=0.1, name="cz_cfg")
    stricter = _planner(env, delta_0=0.01, name="cz_cfg")

    assert first.config_id == same.config_id
    assert first.config_id != stricter.config_id, (
        "delta_0 is the chance constraint; two planners with different thresholds are different "
        "planners and must not share cached results"
    )
