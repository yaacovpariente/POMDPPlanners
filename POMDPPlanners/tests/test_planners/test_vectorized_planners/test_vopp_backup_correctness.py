# SPDX-License-Identifier: MIT

"""VOPP planner-level backup arithmetic, masks and repeated-index updates.

The core vector-tree suite covers allocation, parents, duplicate indexes,
clearing, validation and state dictionaries; the existing VOPP suite covers
control flow, and its expansion test asserts only that some count is positive.
What is missing is the backup itself: the exact ``Q``, the averaging over
repeated leaf indexes, and the masking that keeps padded and terminal slots out.

Every expectation is a scalar calculation of the same rule, written out in the
test. Batch slots are given deliberately distinct values, and node indexes are
deliberately repeated, because a lost ``index_add_`` collision is invisible when
every index is unique.

Reference:
    the PORPP preference-backup rule the planner implements (Algorithm 3).
"""

# pylint: disable=protected-access

import math

import pytest
import torch

from POMDPPlanners.planners.vectorized_planners import VOPPPlanner
from POMDPPlanners.tests.test_planners.test_vectorized_planners.test_vopp import (
    MockGenerativeModel,
)


DISCOUNT = 0.5
# float32 tensors: ~1e-7 relative error, and these sums have at most four terms
# of order 10, so 1e-5 is loose enough for the dtype and far tighter than any
# real arithmetic error.
TOL = 1e-5


def _planner(
    reward_table,
    *,
    max_depth=1,
    num_particles=8,
    iterations=1,
    temperature=1.0,
    always_terminal=False,
    discount_factor=DISCOUNT,
):
    device = torch.device("cpu")
    model = MockGenerativeModel(
        torch.tensor(reward_table), device=device, always_terminal=always_terminal
    )
    return VOPPPlanner(
        model,
        num_actions=model.num_actions,
        num_particles=num_particles,
        max_depth=max_depth,
        num_planning_iterations=iterations,
        discount_factor=discount_factor,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Leaf averaging with repeated indexes
# ---------------------------------------------------------------------------


def test_repeated_leaf_indexes_average_rather_than_accumulate():
    """Several particles landing on one belief node give the mean of their values.

    Purpose: ``_init_leaf_values`` uses ``index_add_`` twice — once for a count
        and once for a sum — and then divides. If either accumulation were lost
        the result would be a sum or the last writer's value, not a mean. The
        fixture repeats one index three times and gives a second index one
        entry, so a sum (12 versus 4) and a last-write (2 versus 4) both differ
        from the correct answer.

    Given: Leaf belief indexes ``[0, 0, 0, 1]`` with values ``[3, 7, 2, 5]``.
    When: ``_init_leaf_values`` runs over two belief nodes.
    Then: Node 0 holds ``(3 + 7 + 2) / 3 = 4`` and node 1 holds 5; the returned
        visit counts are 3 and 1.

    Test type: unit
    """
    planner = _planner([1.0, 0.0])
    # Grow the tree to two belief nodes so index 1 is live.
    tree = planner.tree
    root = tree.root_index
    action_nodes, _ = tree.get_or_create_actions(
        torch.tensor([root], dtype=tree.index_dtype), torch.tensor([0], dtype=tree.index_dtype)
    )
    tree.get_or_create_beliefs(action_nodes, torch.tensor([0], dtype=tree.index_dtype))
    assert tree.num_belief_nodes == 2

    leaf_beliefs = torch.tensor([0, 0, 0, 1], dtype=tree.index_dtype)
    leaf_values = torch.tensor([3.0, 7.0, 2.0, 5.0], dtype=planner._value_dtype)

    visits = planner._init_leaf_values(leaf_beliefs, leaf_values, tree.num_belief_nodes)

    value = tree.belief_field("value")
    assert value[0].item() == pytest.approx(4.0, abs=TOL), (
        f"belief 0 holds {value[0].item()}; three particles with values 3, 7 and 2 must average "
        "to 4 — a sum would give 12 and a last write would give 2"
    )
    assert value[1].item() == pytest.approx(5.0, abs=TOL)
    assert visits[0].item() == pytest.approx(3.0, abs=TOL)
    assert visits[1].item() == pytest.approx(1.0, abs=TOL)


def test_belief_nodes_with_no_leaf_particles_keep_their_value():
    """A belief node no particle reached is left alone, not zeroed.

    Purpose: The division is masked to nodes with a positive visit count; an
        unmasked divide would produce a NaN and poison the whole backup.

    Given: Two belief nodes, only one of which any particle reached, and a
        recognisable prior value on the other.
    When: ``_init_leaf_values`` runs.
    Then: The unreached node's value is unchanged and finite.

    Test type: unit
    """
    planner = _planner([1.0, 0.0])
    tree = planner.tree
    root = tree.root_index
    action_nodes, _ = tree.get_or_create_actions(
        torch.tensor([root], dtype=tree.index_dtype), torch.tensor([0], dtype=tree.index_dtype)
    )
    tree.get_or_create_beliefs(action_nodes, torch.tensor([0], dtype=tree.index_dtype))
    tree.belief_field("value")[1] = -12.0

    planner._init_leaf_values(
        torch.tensor([0], dtype=tree.index_dtype),
        torch.tensor([9.0], dtype=planner._value_dtype),
        tree.num_belief_nodes,
    )

    unreached = tree.belief_field("value")[1].item()
    assert math.isfinite(unreached), "an unreached belief node was divided by a zero visit count"
    assert unreached == pytest.approx(-12.0, abs=TOL)


def test_no_leaves_at_all_leaves_every_value_untouched():
    """An empty leaf batch short-circuits without writing anything.

    Test type: unit
    """
    planner = _planner([1.0, 0.0])
    tree = planner.tree
    tree.belief_field("value")[0] = 3.5

    visits = planner._init_leaf_values(
        torch.empty(0, dtype=tree.index_dtype),
        torch.empty(0, dtype=planner._value_dtype),
        tree.num_belief_nodes,
    )

    assert tree.belief_field("value")[0].item() == pytest.approx(3.5, abs=TOL)
    assert bool((visits == 0).all())


# ---------------------------------------------------------------------------
# Exact action Q
# ---------------------------------------------------------------------------


def test_action_q_is_the_mean_reward_plus_the_discounted_weighted_future():
    """``Q = mean immediate reward + gamma * (sum visits*value) / N(ha)``.

    Purpose: The single most informative VOPP test. The numbers separate the
        correct backup from an unweighted future mean, a missing division by
        the action's visit count and a missing discount.

    Given: One action node at depth 0 with visit count 4 and reward sum 12, so
        its mean reward is 3. It has two belief children at depth 1 whose
        values are 2 and 6 with leaf visit counts 3 and 1, so the weighted
        future sum is ``3*2 + 1*6 = 12`` and the weighted future is
        ``12 / 4 = 3``. Discount 0.5.
    When: ``_action_q_values`` runs for depth 0.
    Then: Q is ``3 + 0.5 * 3 = 4.5``.

    Test type: unit
    """
    planner = _planner([1.0, 0.0], discount_factor=DISCOUNT)
    tree = planner.tree
    root = tree.root_index
    action_nodes, _ = tree.get_or_create_actions(
        torch.tensor([root], dtype=tree.index_dtype), torch.tensor([0], dtype=tree.index_dtype)
    )
    action = int(action_nodes[0].item())
    children, _ = tree.get_or_create_beliefs(
        torch.tensor([action, action], dtype=tree.index_dtype),
        torch.tensor([0, 1], dtype=tree.index_dtype),
    )
    tree.action_visit_count[action] = 4
    tree.action_reward_sum[action] = 12.0
    value = tree.belief_field("value")
    value[children[0]] = 2.0
    value[children[1]] = 6.0

    belief_visits = torch.zeros(tree.num_belief_nodes, dtype=planner._value_dtype)
    belief_visits[children[0]] = 3.0
    belief_visits[children[1]] = 1.0
    stats = planner._action_statistics()

    q_values = planner._action_q_values(0, belief_visits, stats)

    assert stats["mean_reward"][action].item() == pytest.approx(3.0, abs=TOL)
    assert q_values[action].item() == pytest.approx(4.5, abs=TOL), (
        f"Q = {q_values[action].item()}, expected 3 + 0.5 * (3*2 + 1*6)/4 = 4.5; an unweighted "
        "child mean would give 5.0 and a missing discount 6.0"
    )


def test_two_actions_sharing_a_depth_do_not_leak_into_each_other():
    """``index_add_`` over parents keeps each action's future term separate.

    Purpose: The future term is accumulated by scattering child values onto
        their parent action indexes. A missing or wrong parent index would mix
        two actions' children together.

    Given: Two action nodes at the same depth, each with one child; the first
        child is worth 10 and the second 0, both with unit visits, and both
        actions have visit count 1 and reward sum 0.
    When: ``_action_q_values`` runs.
    Then: The first action's Q is ``0 + 0.5 * 10 = 5`` and the second's is 0.

    Test type: unit
    """
    planner = _planner([0.0, 0.0], discount_factor=DISCOUNT)
    tree = planner.tree
    root = tree.root_index
    action_nodes, _ = tree.get_or_create_actions(
        torch.tensor([root, root], dtype=tree.index_dtype),
        torch.tensor([0, 1], dtype=tree.index_dtype),
    )
    first, second = int(action_nodes[0].item()), int(action_nodes[1].item())
    children, _ = tree.get_or_create_beliefs(
        torch.tensor([first, second], dtype=tree.index_dtype),
        torch.tensor([0, 0], dtype=tree.index_dtype),
    )
    tree.action_visit_count[[first, second]] = 1
    tree.action_reward_sum[[first, second]] = 0.0
    value = tree.belief_field("value")
    value[children[0]] = 10.0
    value[children[1]] = 0.0
    belief_visits = torch.zeros(tree.num_belief_nodes, dtype=planner._value_dtype)
    belief_visits[children] = 1.0

    q_values = planner._action_q_values(0, belief_visits, planner._action_statistics())

    assert q_values[first].item() == pytest.approx(5.0, abs=TOL)
    assert q_values[second].item() == pytest.approx(0.0, abs=TOL), (
        f"the second action's Q is {q_values[second].item()}; its only child is worth 0, so a "
        "non-zero value means the first action's child leaked across"
    )


def test_an_action_with_no_children_contributes_no_future_term():
    """A childless action's Q is its mean reward alone.

    Purpose: The division uses ``clamp_min(1.0)`` on the visit count, which
        must not turn an absent future into a spurious one.

    Test type: unit
    """
    planner = _planner([0.0, 0.0], discount_factor=DISCOUNT)
    tree = planner.tree
    action_nodes, _ = tree.get_or_create_actions(
        torch.tensor([tree.root_index], dtype=tree.index_dtype),
        torch.tensor([0], dtype=tree.index_dtype),
    )
    action = int(action_nodes[0].item())
    tree.action_visit_count[action] = 2
    tree.action_reward_sum[action] = 7.0

    belief_visits = torch.zeros(tree.num_belief_nodes, dtype=planner._value_dtype)
    q_values = planner._action_q_values(0, belief_visits, planner._action_statistics())

    assert q_values[action].item() == pytest.approx(3.5, abs=TOL)


# ---------------------------------------------------------------------------
# Repeated action indexes in the statistics update
# ---------------------------------------------------------------------------


def test_repeated_action_indexes_accumulate_visits_and_rewards():
    """Two particles taking the same action in one batch both count.

    Purpose: ``update_action_statistics`` scatters into shared slots. A plain
        assignment instead of an accumulation would record one of the two and
        drop the other, halving every visit count in a real search.

    Given: A batch of four particles whose action nodes are ``[a, a, a, b]``
        with rewards ``[1, 2, 3, 10]``.
    When: ``update_action_statistics`` runs once.
    Then: Action ``a`` has 3 visits and a reward sum of 6; action ``b`` has 1
        visit and a reward sum of 10.

    Test type: unit
    """
    planner = _planner([0.0, 0.0])
    tree = planner.tree
    action_nodes, _ = tree.get_or_create_actions(
        torch.tensor([tree.root_index, tree.root_index], dtype=tree.index_dtype),
        torch.tensor([0, 1], dtype=tree.index_dtype),
    )
    first, second = int(action_nodes[0].item()), int(action_nodes[1].item())

    tree.update_action_statistics(
        torch.tensor([first, first, first, second], dtype=tree.index_dtype),
        torch.tensor([1.0, 2.0, 3.0, 10.0], dtype=tree.value_dtype),
    )

    assert int(tree.action_visit_count[first].item()) == 3
    assert tree.action_reward_sum[first].item() == pytest.approx(6.0, abs=TOL)
    assert int(tree.action_visit_count[second].item()) == 1
    assert tree.action_reward_sum[second].item() == pytest.approx(10.0, abs=TOL)


# ---------------------------------------------------------------------------
# Masks: terminal, padding, inactive storage
# ---------------------------------------------------------------------------


def test_a_fully_terminal_batch_creates_no_successor_belief():
    """Terminal transitions are masked out of the expansion.

    Purpose: Replaces an assertion that only checks a count is positive. Here
        the exact belief-node count is pinned, and the actions' recorded
        rewards prove the search did run rather than exiting early for some
        other reason.

    Given: A model whose every transition is terminal.
    When: ``plan`` runs.
    Then: The tree has exactly one belief node — the root — while the root's
        action nodes did record visits, so the search ran and only the
        successor creation was suppressed.

    Test type: unit
    """
    torch.manual_seed(3)
    planner = _planner([1.0, 0.0], max_depth=4, iterations=3, always_terminal=True)

    planner.plan(torch.zeros(32, 1))

    tree = planner.tree
    assert tree.num_belief_nodes == 1, (
        f"a fully terminal model produced {tree.num_belief_nodes} belief nodes; every successor "
        "must be masked out"
    )
    assert tree.num_action_nodes >= 1
    assert int(tree.action_visit_count[: tree.num_action_nodes].sum().item()) >= 1


def test_padded_capacity_is_never_read_by_the_backup():
    """Poisoned slots past the live region do not change the planned action.

    Purpose: This is the failure mode unique to the flat-tensor backend. Every
        column is preallocated to capacity; a slice that read past
        ``num_action_nodes`` or ``num_belief_nodes`` would fold garbage into
        the backup.

    Given: Two identical seeded plans on the same planner, the second run after
        the padded regions of the visit, reward, depth and value columns have
        been filled with large poison values.
    When: Both plans run.
    Then: They return the same action and the same root preferences.

    Test type: unit
    """
    torch.manual_seed(11)
    planner = _planner([0.0, 1.0], max_depth=2, iterations=3, num_particles=16)
    particles = torch.zeros(16, 1)

    torch.manual_seed(5)
    clean_action = planner.plan(particles)
    clean_preferences = planner.tree.belief_field("preferences")[planner.tree.root_index].clone()

    tree = planner.tree
    tree.action_visit_count[tree.num_action_nodes :] = 10_000
    tree.action_reward_sum[tree.num_action_nodes :] = 10_000.0
    tree.action_depth[tree.num_action_nodes :] = 0
    tree.belief_depth[tree.num_belief_nodes :] = 0

    torch.manual_seed(5)
    poisoned_action = planner.plan(particles)
    poisoned_preferences = planner.tree.belief_field("preferences")[planner.tree.root_index].clone()

    assert poisoned_action == clean_action, (
        f"poisoning the padded slots changed the planned action from {clean_action} to "
        f"{poisoned_action}; the backup must only read the live region"
    )
    assert torch.allclose(
        poisoned_preferences, clean_preferences, atol=TOL
    ), "poisoning the padded slots changed the root preferences"


def test_plan_clears_the_tree_so_two_calls_do_not_accumulate():
    """Each ``plan`` call starts from a cleared tree.

    Purpose: Preferences are additive, so a tree carried across calls would
        keep sharpening toward whatever the first call preferred.

    Given: Two identical seeded calls.
    When: Both run.
    Then: They produce the same node counts and the same root preferences, so
        the second call did not build on the first.

    Test type: unit
    """
    planner = _planner([0.0, 1.0], max_depth=2, iterations=3, num_particles=16)
    particles = torch.zeros(16, 1)

    torch.manual_seed(1)
    planner.plan(particles)
    first_counts = (planner.tree.num_belief_nodes, planner.tree.num_action_nodes)
    first_preferences = planner.tree.belief_field("preferences")[planner.tree.root_index].clone()

    torch.manual_seed(1)
    planner.plan(particles)

    assert (planner.tree.num_belief_nodes, planner.tree.num_action_nodes) == first_counts
    assert torch.allclose(
        planner.tree.belief_field("preferences")[planner.tree.root_index],
        first_preferences,
        atol=TOL,
    ), "the second call's preferences differ, so the tree was not cleared"


# ---------------------------------------------------------------------------
# The preference update itself
# ---------------------------------------------------------------------------


def test_preference_update_adds_q_and_recentres_by_the_log_sum_exp():
    """Preferences are recentred then incremented by Q, and V is the log-sum-exp.

    Purpose: Pins the update rule with a hand-computable case. With
        temperature 1 and both preferences at zero, the recentring subtracts
        ``log(1 + 1) = log 2`` from each, then Q is added to the entry of the
        action that was updated.

    Given: A root with two action children, both preferences at 0, temperature
        1, and Q values 4 for action key 0 and 0 for key 1.
    When: ``_apply_preference_update`` runs.
    Then: The preferences become ``[-log2 + 4, -log2]`` and the root's value is
        the log-sum-exp of those, ``log(exp(4 - log2) + exp(-log2))``.

    Test type: unit
    """
    planner = _planner([0.0, 0.0], temperature=1.0)
    tree = planner.tree
    root = tree.root_index
    action_nodes, _ = tree.get_or_create_actions(
        torch.tensor([root, root], dtype=tree.index_dtype),
        torch.tensor([0, 1], dtype=tree.index_dtype),
    )
    tree.action_visit_count[action_nodes] = 1
    stats = planner._action_statistics()
    q_values = torch.zeros(stats["count"], dtype=planner._value_dtype)
    q_values[int(action_nodes[0].item())] = 4.0

    planner._apply_preference_update(
        beliefs_at_depth=torch.tensor([root], dtype=tree.index_dtype),
        actions_at_depth=action_nodes,
        q_values=q_values,
        stats=stats,
    )

    preferences = tree.belief_field("preferences")[root]
    shift = math.log(2.0)
    assert preferences[0].item() == pytest.approx(4.0 - shift, abs=1e-4)
    assert preferences[1].item() == pytest.approx(-shift, abs=1e-4)
    expected_value = math.log(math.exp(4.0 - shift) + math.exp(-shift))
    assert tree.belief_field("value")[root].item() == pytest.approx(expected_value, abs=1e-4)


def test_the_dominant_reward_action_wins_the_root_preference():
    """A whole plan on a model where one action dominates picks that action.

    Purpose: End-to-end confirmation that the backup arithmetic above steers
        the returned decision, not only the intermediate tensors.

    Given: A model where action 1 earns 1 and every other action earns 0.
    When: ``plan`` runs with enough iterations to separate them.
    Then: Action 1 is returned and holds the largest root preference.

    Test type: unit
    """
    torch.manual_seed(0)
    planner = _planner([0.0, 1.0, 0.0], max_depth=3, iterations=8, num_particles=64)

    chosen = planner.plan(torch.zeros(64, 1))

    preferences = planner.tree.belief_field("preferences")[planner.tree.root_index]
    assert chosen == 1, f"planner chose action {chosen}; action 1 is the only rewarding one"
    assert int(torch.argmax(preferences).item()) == 1


def test_tree_metrics_come_from_the_live_region_only():
    """The reported metrics are the ones the live tree supports.

    Purpose: VOPP reports the shared ``TreeMetrics`` set so it can be compared
        with the MCTS planners; the arithmetic of that helper is pinned in
        ``test_tree_metrics_backends.py`` and this checks the planner's own
        hand-off.

    Test type: unit
    """
    torch.manual_seed(2)
    planner = _planner([0.0, 1.0], max_depth=2, iterations=4, num_particles=32)
    planner.plan(torch.zeros(32, 1))

    metrics = {m.name: m.value for m in planner.tree_metrics()}

    tree = planner.tree
    root_actions = [
        index
        for index in range(tree.num_action_nodes)
        if int(tree.action_parent_belief[index].item()) == tree.root_index
    ]
    visits = [int(tree.action_visit_count[index].item()) for index in root_actions]
    assert metrics["n_actions_from_root"] == len(root_actions)
    assert metrics["root_visit_count"] == sum(visits)
    assert metrics["min_actions_visit_count"] == min(visits)
    assert metrics["max_actions_visit_count"] == max(visits)
