# SPDX-License-Identifier: MIT

"""CPOMCPOW correctness tests.

CPOMCPOW ships with no tests at all. It is a concrete planner, so it owes the
same evidence as its siblings: controlled backups on both the reward and every
cost channel, visit accounting, observation reuse, the widening boundary, the
terminal and depth cut-offs, Lagrangian selection over the whole cost vector,
and the public contract.

The cost fixture uses ``[1, 3]`` leaving ``root`` and ``[2, 0]`` leaving
``next``. Those were chosen so the discounted total ``[1,3] + 0.5*[2,0] =
[2, 3]`` has two different entries whose *order* also differs from the
immediate cost's — a planner that swapped the channels, or dropped the future
term, produces a visibly different vector rather than a coincidentally equal
one.

Reference:
    Jamgochian, A., Corso, A., & Kochenderfer, M. J. (2023). Online Planning for
    Constrained POMDPs with Continuous Spaces through Dual Ascent. ICAPS 33,
    198-202.
"""

import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import (
    WeightedParticleBelief,
    WeightedParticleBeliefStateUpdate,
)
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.planners.mcts_planners.constrained_pomcpow import CPOMCPOW
from POMDPPlanners.tests.test_planners.planner_fixtures import (
    CHAIN_REWARDS,
    END,
    NEXT,
    ROOT,
    ChainEnv,
    ConstrainedChainEnv,
    FixedActionSampler,
    SingleActionSampler,
    chain_belief,
)
from POMDPPlanners.tests.test_planners.tree_assertions import (
    action_ids,
    assert_subtree_unchanged,
    assert_values_within_bounds,
    belief_ids,
    running_mean,
    snapshot_subtree,
    walk_arena_tree,
)


DISCOUNT = 0.5
TOL = 1e-12

# Hand-computed from the fixture's cost table, never read back from the planner.
IMMEDIATE_COST = np.array([1.0, 3.0])
FUTURE_COST = np.array([2.0, 0.0])
EXPECTED_COST_RETURN = IMMEDIATE_COST + DISCOUNT * FUTURE_COST  # -> [2.0, 3.0]
EXPECTED_REWARD_RETURN = CHAIN_REWARDS[ROOT] + DISCOUNT * CHAIN_REWARDS[NEXT]  # -> 4.0


def _planner(
    env,
    *,
    depth=1,
    n_simulations=1,
    k_o=1.0,
    alpha_o=0.0,
    k_a=1.0,
    alpha_a=0.0,
    sampler=None,
    cost_budget=None,
    lambda_init=0.0,
    lambda_step=0.1,
    return_minimal_cost=False,
    exploration_constant=0.0,
):
    return CPOMCPOW(
        environment=env,
        discount_factor=env.discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        name="CPOMCPOW_correctness",
        action_sampler=sampler or SingleActionSampler("a"),
        cost_budget=np.array([1.0, 1.0]) if cost_budget is None else cost_budget,
        lambda_init=lambda_init,
        lambda_step=lambda_step,
        return_minimal_cost=return_minimal_cost,
        n_simulations=n_simulations,
    )


# ---------------------------------------------------------------------------
# Construction contract
# ---------------------------------------------------------------------------


def test_plain_environment_is_rejected():
    """An unconstrained environment cannot supply a constraint cost.

    Given: A plain ``ChainEnv``.
    When: A CPOMCPOW is constructed on it.
    Then: ``TypeError`` names the requirement.

    Test type: unit
    """
    with pytest.raises(TypeError, match="ConstrainedEnvironment"):
        _planner(ChainEnv(discount_factor=DISCOUNT))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"cost_budget": np.array([])}, "at least one constraint dimension"),
        ({"cost_budget": np.array([-1.0, 1.0])}, "non-negative"),
        ({"cost_budget": np.array([np.nan, 1.0])}, "finite"),
        ({"lambda_init": np.array([-1.0, 0.0])}, "non-negative"),
        ({"lambda_step": 0.0}, "positive"),
    ],
)
def test_invalid_constraint_parameters_are_rejected(kwargs, message):
    """Each constraint-parameter boundary raises with a message that says why.

    Purpose: A NaN budget or a negative multiplier silently corrupts every
        later dual-ascent step, so these are checked at construction.

    Test type: unit
    """
    with pytest.raises(ValueError, match=message):
        _planner(ConstrainedChainEnv(discount_factor=DISCOUNT), **kwargs)


# ---------------------------------------------------------------------------
# Exact reward and per-channel cost backups
# ---------------------------------------------------------------------------


def test_two_step_reward_and_every_cost_channel_are_backed_up_exactly():
    """Q = 4 and Q_C = [2, 3] after one controlled two-step simulation.

    Purpose: The single most informative CPOMCPOW test — it separates the
        correct backup from a dropped future cost (``[1, 3]``), a channel swap
        (``[3, 2]``), a missing discount (``[3, 3]``) and a doubled immediate
        cost (``[2, 6]``), none of which equals ``[2, 3]``.

    Given: The constrained chain: reward 2 and cost ``[1, 3]`` leaving ``root``,
        reward 4 and cost ``[2, 0]`` leaving ``next``, ``end`` terminal, discount
        0.5, planner depth 1 so the leaf expansion rolls out exactly one step.
    When: One simulation runs from the root.
    Then: The returned reward is ``2 + 0.5*4 = 4``, the returned cost vector is
        ``[1,3] + 0.5*[2,0] = [2, 3]``, and the action node's stored Q and Q_C
        equal those same values after their first sample.

    Test type: unit
    """
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=1)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))

    total_v, total_c = planner._simulate_state_path_with_cost(
        tree=tree, state=ROOT, belief_id=root_id, depth=0
    )

    assert total_v == pytest.approx(
        EXPECTED_REWARD_RETURN, abs=TOL
    ), f"reward return {total_v} != {EXPECTED_REWARD_RETURN}"
    np.testing.assert_allclose(
        total_c,
        EXPECTED_COST_RETURN,
        atol=TOL,
        err_msg=(
            f"cost return {total_c} != {EXPECTED_COST_RETURN} = [1,3] + 0.5*[2,0]; a dropped "
            "future term gives [1,3] and a channel swap gives [3,2]"
        ),
    )
    (action_id,) = action_ids(tree, root_id)
    assert tree.q_value[action_id] == pytest.approx(EXPECTED_REWARD_RETURN, abs=TOL)
    np.testing.assert_allclose(planner._cost_q(action_id), EXPECTED_COST_RETURN, atol=TOL)
    assert tree.visit_count[action_id] == 1
    assert tree.visit_count[root_id] == 1


def test_cost_q_is_a_per_channel_running_mean():
    """Q_C after two samples is their per-channel mean, not a sum.

    Purpose: The cost channel uses the same incremental mean as the reward
        channel; a missed division shows up here and nowhere else.

    Given: An action node backed up with cost vectors ``[1, 3]`` then ``[3, 1]``.
    When: ``_backup_action_node`` runs twice.
    Then: Q_C is ``[1, 3]`` after the first and ``[2, 2]`` after the second —
        the hand-written running mean per channel — and the visit count is 2.

    Test type: unit
    """
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)

    planner._backup_action_node(
        tree=tree,
        belief_id=root_id,
        action_id=action_id,
        total_v=2.0,
        total_c=np.array([1.0, 3.0]),
    )
    np.testing.assert_allclose(planner._cost_q(action_id), [1.0, 3.0], atol=TOL)

    planner._backup_action_node(
        tree=tree,
        belief_id=root_id,
        action_id=action_id,
        total_v=6.0,
        total_c=np.array([3.0, 1.0]),
    )

    expected = np.array(
        [
            running_mean(1.0, 1, 3.0),
            running_mean(3.0, 1, 1.0),
        ]
    )
    np.testing.assert_allclose(expected, [2.0, 2.0], atol=TOL)
    np.testing.assert_allclose(planner._cost_q(action_id), expected, atol=TOL)
    assert tree.q_value[action_id] == pytest.approx(running_mean(2.0, 1, 6.0), abs=TOL)
    assert tree.visit_count[action_id] == 2


def test_belief_value_is_the_all_children_maximum_of_the_reward_channel():
    """V(b) follows the reward Q only; the cost channel does not enter it.

    Given: Two action children, one with reward Q = 1 and a huge cost, one with
        reward Q = 3 and no cost.
    When: A backup runs.
    Then: V equals 3, the larger reward Q, unaffected by the cost vectors.

    Test type: unit
    """
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    cheap_id = tree.add_action_node(action="a", parent_id=root_id)
    dear_id = tree.add_action_node(action="b", parent_id=root_id)
    tree.update_action_q_with_return(dear_id, 3.0)
    planner._set_cost_q(dear_id, np.array([99.0, 99.0]))

    planner._backup_action_node(
        tree=tree,
        belief_id=root_id,
        action_id=cheap_id,
        total_v=1.0,
        total_c=np.zeros(2),
    )

    assert tree.v_value[root_id] == pytest.approx(3.0, abs=TOL), (
        f"V(root) = {tree.v_value[root_id]}; the belief value is the maximum reward Q over all "
        "action children and must ignore Q_C"
    )


def test_terminal_state_returns_zero_reward_and_a_zero_cost_vector():
    """Termination contributes nothing on either channel.

    Given: A belief node reached with the terminal state.
    When: The constrained simulation runs.
    Then: The reward return is 0.0, the cost vector is all zeros with the right
        length, the node gained one visit and nothing was expanded.

    Test type: unit
    """
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=3)
    tree = Tree()
    node_id = tree.add_belief_node(chain_belief(END))

    total_v, total_c = planner._simulate_state_path_with_cost(
        tree=tree, state=END, belief_id=node_id, depth=0
    )

    assert total_v == 0.0
    np.testing.assert_allclose(total_c, np.zeros(planner.n_constraints), atol=TOL)
    assert total_c.shape == (planner.n_constraints,)
    assert tree.visit_count[node_id] == 1
    assert tree.children_ids[node_id] == []


def test_depth_cutoff_returns_zeros_and_leaves_the_node_alone():
    """Past the depth limit both channels return zero and nothing is counted.

    Test type: unit
    """
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=1)
    tree = Tree()
    node_id = tree.add_belief_node(chain_belief(ROOT))

    total_v, total_c = planner._simulate_state_path_with_cost(
        tree=tree, state=ROOT, belief_id=node_id, depth=2
    )

    assert total_v == 0.0
    np.testing.assert_allclose(total_c, np.zeros(2), atol=TOL)
    assert tree.visit_count[node_id] == 0
    assert tree.children_ids[node_id] == []


def test_a_cost_vector_of_the_wrong_shape_is_rejected_at_the_boundary():
    """A mismatched constraint dimensionality fails loudly, not silently.

    Purpose: A shape mismatch would otherwise broadcast into Q_C and corrupt
        dual ascent with no error at all.

    Given: An environment whose ``constraint_cost`` returns one channel, against
        a two-channel cost budget.
    When: ``_read_constraint_cost`` runs.
    Then: ``ValueError`` names the expected shape.

    Test type: unit
    """
    one_channel_env = ConstrainedChainEnv(
        discount_factor=DISCOUNT, costs={ROOT: [1.0], NEXT: [1.0], END: [0.0]}
    )
    planner = _planner(one_channel_env, cost_budget=np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match=r"expected \(2,\)"):
        planner._read_constraint_cost(state=ROOT, action="a", next_state=NEXT)


def test_a_non_finite_cost_is_rejected_at_the_boundary():
    """A NaN constraint cost fails rather than poisoning Q_C and lambda.

    Test type: unit
    """
    env = ConstrainedChainEnv(
        discount_factor=DISCOUNT,
        costs={ROOT: [np.nan, 0.0], NEXT: [0.0, 0.0], END: [0.0, 0.0]},
    )
    planner = _planner(env)
    with pytest.raises(ValueError, match="non-finite"):
        planner._read_constraint_cost(state=ROOT, action="a", next_state=NEXT)


# ---------------------------------------------------------------------------
# Constrained selection over the whole cost vector
# ---------------------------------------------------------------------------


def test_lagrangian_selection_reads_every_cost_channel():
    """The greedy pick uses ``Q - lambda . Q_C`` across all channels.

    Purpose: An implementation that only scored the first channel would pick
        the wrong action here, and the reward-only rule would too.

    Given: Two actions with equal reward Q = 5. Action "a" costs ``[0, 4]`` and
        action "b" costs ``[4, 0]``. Lambda is ``[1, 0]``, so only the *first*
        channel is penalised: score("a") = 5 - 0 = 5, score("b") = 5 - 4 = 1.
    When: The greedy Lagrangian action is chosen.
    Then: "a" wins. Swapping lambda to ``[0, 1]`` flips the winner to "b",
        which proves the second channel is read too.

    Test type: unit
    """
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    a_id = tree.add_action_node(action="a", parent_id=root_id)
    b_id = tree.add_action_node(action="b", parent_id=root_id)
    for node_id in (a_id, b_id):
        tree.visit_count[node_id] = 1
        tree.q_value[node_id] = 5.0
    planner._set_cost_q(a_id, np.array([0.0, 4.0]))
    planner._set_cost_q(b_id, np.array([4.0, 0.0]))

    planner._lambda = np.array([1.0, 0.0])
    assert planner._lagrangian_best_action_id(tree=tree, belief_id=root_id) == a_id

    planner._lambda = np.array([0.0, 1.0])
    assert planner._lagrangian_best_action_id(tree=tree, belief_id=root_id) == b_id, (
        "flipping lambda to the second channel must flip the winner; a selection rule that "
        "only reads channel 0 keeps choosing 'a'"
    )


def test_dual_ascent_moves_lambda_toward_the_constraint_violation():
    """Lambda rises where the chosen action exceeds its budget and stays at 0 below it.

    Purpose: Pins the sign and the step size of the multiplier update, and the
        non-negative projection.

    Given: A root whose best action has Q_C = ``[3, 0]`` against a budget of
        ``[1, 1]``, lambda starting at zero and a step of 0.1.
    When: One dual-ascent step runs.
    Then: Lambda is ``[0.1*(3-1), max(0, 0.1*(0-1))] = [0.2, 0.0]`` — it grows
        on the violated channel and is clipped at zero on the slack one.

    Test type: unit
    """
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, cost_budget=np.array([1.0, 1.0]), lambda_step=0.1)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    tree.visit_count[action_id] = 1
    tree.q_value[action_id] = 1.0
    planner._set_cost_q(action_id, np.array([3.0, 0.0]))

    planner._dual_ascent_step(tree=tree, root_id=root_id)

    np.testing.assert_allclose(planner._lambda, [0.2, 0.0], atol=1e-9)


def test_minimal_cost_propagation_returns_a_real_siblings_vector():
    """The propagated cost is one sibling's Q_C verbatim, not an elementwise minimum.

    Purpose: The paper's minimal-cost trick substitutes the cost of the best
        sibling. An elementwise minimum would synthesise ``[0, 0]`` here, a
        vector no action achieves.

    Given: Two visited siblings with Q_C ``[0, 5]`` and ``[5, 0]``, lambda
        ``[1, 0]`` so the Lagrangian score prefers the first.
    When: ``_minimal_cost_propagation`` runs.
    Then: The result is exactly ``[0, 5]``.

    Test type: unit
    """
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, return_minimal_cost=True)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    a_id = tree.add_action_node(action="a", parent_id=root_id)
    b_id = tree.add_action_node(action="b", parent_id=root_id)
    tree.visit_count[a_id] = 1
    tree.visit_count[b_id] = 1
    planner._set_cost_q(a_id, np.array([0.0, 5.0]))
    planner._set_cost_q(b_id, np.array([5.0, 0.0]))
    planner._lambda = np.array([1.0, 0.0])

    result = planner._minimal_cost_propagation(
        tree=tree, belief_id=root_id, fallback=np.array([99.0, 99.0])
    )

    np.testing.assert_allclose(result, [0.0, 5.0], atol=TOL)


def test_minimal_cost_propagation_falls_back_when_no_sibling_is_visited():
    """With no visited sibling the propagated cost is the caller's own.

    Test type: unit
    """
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, return_minimal_cost=True)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    tree.add_action_node(action="a", parent_id=root_id)
    fallback = np.array([7.0, 8.0])

    result = planner._minimal_cost_propagation(tree=tree, belief_id=root_id, fallback=fallback)

    np.testing.assert_allclose(result, fallback, atol=TOL)


# ---------------------------------------------------------------------------
# Observation widening and isolation
# ---------------------------------------------------------------------------


def test_reused_observation_child_bumps_weight_and_cdf():
    """Reusing an observation child increments its weight and patches the CDF.

    Test type: unit
    """
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, k_o=5.0, alpha_o=1.0)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    tree.visit_count[action_id] = 4

    first_id = planner._observation_widening(tree, action_id, NEXT)
    second_id = planner._observation_widening(tree, action_id, NEXT)

    assert second_id == first_id
    assert len(belief_ids(tree, action_id)) == 1
    assert tree.weight[first_id] == pytest.approx(2.0, abs=TOL)
    assert tree.children_cdf[action_id][-1] == pytest.approx(2.0, abs=TOL)


def test_widening_boundary_stops_adding_observation_children():
    """CPOMCPOW inherits POMCPOW's ``<=`` widening condition unchanged.

    Purpose: The constrained variant reuses the observation-widening path, so
        the same boundary must hold. With ``k_o=1, alpha_o=0`` the bound is the
        constant 1 and the condition is checked before the child is added, so
        two children may be created and the third distinct observation must
        sample instead.

    Test type: unit
    """
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, k_o=1.0, alpha_o=0.0)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    tree.visit_count[action_id] = 3
    first_id = planner._observation_widening(tree, action_id, NEXT)
    second_id = planner._observation_widening(tree, action_id, END)
    assert len(belief_ids(tree, action_id)) == 2

    third_id = planner._observation_widening(tree, action_id, ROOT)

    assert (
        len(belief_ids(tree, action_id)) == 2
    ), f"the bound of 1 was exceeded: {len(belief_ids(tree, action_id))} children"
    assert third_id in (first_id, second_id)


def test_a_simulation_leaves_the_untouched_action_branch_alone():
    """A simulation down one action changes nothing on its sibling.

    Test type: unit
    """
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=1, k_a=5.0, alpha_a=1.0, sampler=SingleActionSampler("a"))
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    sibling_id = tree.add_action_node(action="b", parent_id=root_id)
    tree.add_belief_node(
        belief=WeightedParticleBeliefStateUpdate(),
        observation=NEXT,
        parent_id=sibling_id,
        weight=1.0,
        obs_key=NEXT,
    )
    tree.visit_count[sibling_id] = 2
    tree.q_value[sibling_id] = 9.0
    planner._set_cost_q(sibling_id, np.array([1.0, 2.0]))
    before = snapshot_subtree(tree, sibling_id)
    cost_before = planner._cost_q(sibling_id).copy()

    planner._simulate_state_path_with_cost(tree=tree, state=ROOT, belief_id=root_id, depth=0)

    assert_subtree_unchanged(tree, before, label="untouched CPOMCPOW action branch")
    np.testing.assert_allclose(planner._cost_q(sibling_id), cost_before, atol=TOL)


# ---------------------------------------------------------------------------
# Whole search, ranges, public contract, reset
# ---------------------------------------------------------------------------


def test_full_search_structure_and_both_channel_ranges():
    """Reachability, links, CDFs, and both reward and cost values inside bounds.

    Purpose: The cost channel needs its own derived interval; a reward bound
        says nothing about it.

    Given: The constrained chain with rewards in [0, 4] and each cost channel in
        [0, 3], discount 0.5, depth 2, ten simulations, fixed seeds.
    When: ``action()`` builds a tree.
    Then: The walk reaches every allocated node and saw an expanded non-root
        belief; every reward Q and V is inside the reward interval; and every
        stored Q_C channel is inside ``[0, 3 * (1 + 0.5 + 0.25)]``, the largest
        a three-term discounted sum of costs in [0, 3] can be.

    Test type: unit
    """
    np.random.seed(31337)
    random.seed(31337)
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _planner(
        env,
        depth=2,
        n_simulations=10,
        k_a=2.0,
        alpha_a=0.5,
        k_o=2.0,
        alpha_o=0.5,
        sampler=FixedActionSampler(["a", "b"]),
        exploration_constant=1.0,
        return_minimal_cost=True,
    )

    planner._reset_per_action_state()
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))

    counters = walk_arena_tree(tree, root_id)
    assert counters.expanded_non_root_belief_nodes >= 1
    assert counters.visited_action_nodes >= 1

    def horizon_of(node_id: int, edge_depth: int):
        del node_id
        return max(planner.depth + 1 - edge_depth // 2, 0)

    checked = assert_values_within_bounds(
        tree, root_id, horizon_of=horizon_of, reward_min=0.0, reward_max=4.0, discount=DISCOUNT
    )
    assert checked >= 3

    max_cost_channel = 3.0 * sum(DISCOUNT**t for t in range(planner.depth + 1))
    for action_id, cost_q in planner._action_cost_q.items():
        assert np.all(np.isfinite(cost_q)), f"action {action_id} has non-finite Q_C {cost_q}"
        assert np.all(cost_q >= -TOL), f"action {action_id} has a negative Q_C {cost_q}"
        assert np.all(cost_q <= max_cost_channel + TOL), (
            f"action {action_id} has Q_C {cost_q} above the largest possible discounted cost "
            f"{max_cost_channel} for per-step costs in [0, 3]"
        )


def test_action_returns_one_legal_action_and_the_declared_metrics():
    """Public contract: one action and exactly the declared metric names.

    Test type: unit
    """
    np.random.seed(101)
    random.seed(101)
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=2, n_simulations=6, sampler=FixedActionSampler(["a", "b"]))

    actions, run_data = planner.action(chain_belief(ROOT))

    assert len(actions) == 1 and actions[0] in env.get_actions()
    assert [v.name for v in run_data.info_variables] == CPOMCPOW.get_info_variable_names()


def test_terminal_belief_returns_a_legal_action_with_no_metrics():
    """A wholly terminal belief short-circuits before any search.

    Test type: unit
    """
    np.random.seed(2)
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=2, n_simulations=6)
    belief = WeightedParticleBelief(particles=[END, END], log_weights=np.array([-1.0, -1.0]))
    actions, run_data = planner.action(belief)

    assert len(actions) == 1 and actions[0] in env.get_actions()
    assert run_data.info_variables == []


def test_second_call_resets_lambda_and_the_cost_q_table():
    """Per-search state is rebuilt on every ``action()`` call.

    Purpose: Lambda and the action-ID-keyed Q_C dict are indexed by node IDs
        that only mean anything inside one tree; carrying them over would apply
        one tree's costs to another tree's nodes.

    Given: A planner whose lambda has been driven away from its initial value
        by a first call.
    When: A second ``action()`` runs.
    Then: The Q_C table's keys all belong to the new tree, and lambda starts
        each call from ``lambda_init`` — checked by asserting the recorded
        value at the start of the second call.

    Test type: unit
    """
    np.random.seed(55)
    random.seed(55)
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _planner(
        env,
        depth=2,
        n_simulations=6,
        cost_budget=np.array([0.0, 0.0]),
        lambda_step=0.5,
        sampler=FixedActionSampler(["a", "b"]),
    )
    _, first = planner.action(chain_belief(ROOT))
    first_snapshot = [(v.name, v.value) for v in first.info_variables]
    assert np.any(
        planner._lambda > 0.0
    ), "the fixture must drive lambda off its initial value, or the reset check is vacuous"

    observed = {}
    original_reset = planner._reset_per_action_state

    def spy():
        original_reset()
        observed["lambda"] = planner._lambda.copy()
        observed["cost_q"] = dict(planner._action_cost_q)

    planner._reset_per_action_state = spy  # type: ignore[method-assign]
    planner.action(chain_belief(ROOT))

    np.testing.assert_allclose(observed["lambda"], planner.lambda_init, atol=TOL)
    assert observed["cost_q"] == {}, "the cost-Q table must be empty at the start of a new search"
    assert [
        (v.name, v.value) for v in first.info_variables
    ] == first_snapshot, "the first call's metrics changed when the second call ran"


def test_configuration_identity_distinguishes_algorithm_parameters():
    """Equal configurations share a config ID and a changed one does not.

    Purpose: The config ID is the simulation cache key; two planners that
        differ in an algorithm parameter must not share cached results.

    Test type: unit
    """
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    first = _planner(env, depth=2, n_simulations=6)
    same = _planner(env, depth=2, n_simulations=6)
    different_budget = _planner(env, depth=2, n_simulations=6, cost_budget=np.array([2.0, 2.0]))
    different_depth = _planner(env, depth=3, n_simulations=6)

    assert first.config_id == same.config_id
    assert (
        first.config_id != different_budget.config_id
    ), "a different cost budget is a different planner and must not share a cache key"
    assert first.config_id != different_depth.config_id
