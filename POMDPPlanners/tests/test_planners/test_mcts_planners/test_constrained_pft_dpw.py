# pylint: disable=protected-access  # Tests reach into private state to verify backups
import random
from typing import Any

import numpy as np
import pytest

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.environment import ConstrainedEnvironment
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.planners.mcts_planners.constrained_pft_dpw import CPFT_DPW
from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler


np.random.seed(42)
random.seed(42)


class _ZeroCostEnv(ContinuousLightDarkPOMDP, ConstrainedEnvironment):
    """ContinuousLightDarkPOMDP + ConstrainedEnvironment with zero constraint cost.

    Used by tests that exercise the planner without needing the cost track
    to drive any behaviour (smoke tests, λ-reset tests, ...).
    """

    def constraint_cost(self, state: Any, action: Any, next_state: Any) -> np.ndarray:
        del state, action, next_state
        return np.array([0.0])


class _OneCostEnv(ContinuousLightDarkPOMDP, ConstrainedEnvironment):
    """ContinuousLightDarkPOMDP + ConstrainedEnvironment with constant cost = 1.

    Used by tests that need the cost track to accumulate so QC > 0 backups
    can be observed.
    """

    def constraint_cost(self, state: Any, action: Any, next_state: Any) -> np.ndarray:
        del state, action, next_state
        return np.array([1.0])


@pytest.fixture
def environment():
    return _ZeroCostEnv(discount_factor=0.95)


@pytest.fixture
def action_sampler():
    return UnitCircleActionSampler(max_action_magnitude=1.0)


@pytest.fixture
def planner(environment, action_sampler):
    return CPFT_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=5,
        name="test_constrained_pft_dpw",
        action_sampler=action_sampler,
        cost_budget=0.5,
        lambda_init=0.0,
        lambda_step=0.1,
        return_minimal_cost=True,
        k_a=1.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.0,
        n_simulations=50,
        min_visit_count_per_action=1,
    )


@pytest.fixture
def initial_belief(environment):
    return get_initial_belief(pomdp=environment, n_particles=20, resampling=True)


def test_initialization(planner):
    """Verify CPFT_DPW constructor parameters land on the instance.

    Purpose: Planner state after construction with valid args.

    Given: A CPFT_DPW built with the fixture parameters.
    When: Public attributes are inspected.
    Then: Constraint params are stored as ndarrays of shape ``(1,)`` (since
        cost_budget is a scalar); inherited PFT params are set.

    Test type: unit
    """
    np.testing.assert_array_equal(planner.cost_budget, np.array([0.5]))
    np.testing.assert_array_equal(planner.lambda_init, np.array([0.0]))
    assert planner.lambda_step == 0.1
    assert planner.return_minimal_cost is True
    assert planner.n_constraints == 1
    np.testing.assert_array_equal(planner._lambda, np.array([0.0]))
    assert planner.depth == 5
    assert planner.k_a == 1.0
    assert planner.alpha_a == 0.5


def test_rejects_unconstrained_environment(action_sampler):
    """Constructor raises ``TypeError`` if env is not a ``ConstrainedEnvironment``.

    Purpose: Forces the caller to use the new env-side interface.

    Given: A plain ``ContinuousLightDarkPOMDP`` (not constrained).
    When: ``CPFT_DPW(environment=env, ...)`` is constructed.
    Then: ``TypeError`` is raised mentioning ``ConstrainedEnvironment``.

    Test type: unit
    """
    plain_env = ContinuousLightDarkPOMDP(discount_factor=0.95)
    with pytest.raises(TypeError, match="ConstrainedEnvironment"):
        CPFT_DPW(
            environment=plain_env,
            discount_factor=0.95,
            depth=2,
            name="bad",
            action_sampler=action_sampler,
            cost_budget=0.5,
            n_simulations=5,
        )


def test_invalid_cost_budget_raises(environment, action_sampler):
    """Constructor rejects negative ``cost_budget``.

    Purpose: Guards against silently planning with a meaningless budget.

    Given: A CPFT_DPW constructor call with ``cost_budget=-0.1``.
    When: ``__init__`` runs.
    Then: ``ValueError`` is raised, message mentions ``cost_budget``.

    Test type: unit
    """
    with pytest.raises(ValueError, match="cost_budget"):
        CPFT_DPW(
            environment=environment,
            discount_factor=0.95,
            depth=2,
            name="bad",
            action_sampler=action_sampler,
            cost_budget=-0.1,
            n_simulations=5,
        )


def test_invalid_lambda_init_raises(environment, action_sampler):
    """Constructor rejects negative ``lambda_init``.

    Purpose: Guards against an inadmissible dual variable on construction.

    Given: A CPFT_DPW constructor call with ``lambda_init=-1.0``.
    When: ``__init__`` runs.
    Then: ``ValueError`` is raised, message mentions ``lambda_init``.

    Test type: unit
    """
    with pytest.raises(ValueError, match="lambda_init"):
        CPFT_DPW(
            environment=environment,
            discount_factor=0.95,
            depth=2,
            name="bad",
            action_sampler=action_sampler,
            cost_budget=0.5,
            lambda_init=-1.0,
            n_simulations=5,
        )


def test_invalid_lambda_step_raises(environment, action_sampler):
    """Constructor rejects non-positive ``lambda_step``.

    Purpose: A zero / negative dual step would freeze or invert ascent.

    Given: A CPFT_DPW constructor call with ``lambda_step=0.0``.
    When: ``__init__`` runs.
    Then: ``ValueError`` is raised, message mentions ``lambda_step``.

    Test type: unit
    """
    with pytest.raises(ValueError, match="lambda_step"):
        CPFT_DPW(
            environment=environment,
            discount_factor=0.95,
            depth=2,
            name="bad",
            action_sampler=action_sampler,
            cost_budget=0.5,
            lambda_step=0.0,
            n_simulations=5,
        )


def test_action_returns_valid_tuple(planner, initial_belief):
    """``action(belief)`` returns ``([action], PolicyRunData)``.

    Purpose: Smoke-test the public planner interface.

    Given: The fixture planner and a 20-particle initial belief.
    When: ``planner.action(initial_belief)`` is called.
    Then: A length-1 action list and a ``PolicyRunData`` with tree metrics
        are returned.

    Test type: integration
    """
    actions, run_data = planner.action(initial_belief)
    assert isinstance(actions, list) and len(actions) == 1
    assert isinstance(actions[0], np.ndarray)
    assert run_data.info_variables  # tree metrics populated


def test_lambda_resets_per_action_call(planner, initial_belief):
    """``_lambda`` is reset to ``lambda_init`` on every ``action()`` call.

    Purpose: Online formulation uses a fresh dual variable per decision.

    Given: A planner whose ``_lambda`` has been pushed up by a previous run.
    When: ``action()`` is called again.
    Then: ``_lambda`` starts the new run from ``lambda_init`` and ends
        bounded below by zero (elementwise).

    Test type: unit
    """
    planner._lambda = np.array([10.0])
    planner.action(initial_belief)
    assert (planner._lambda >= 0.0).all()

    planner._lambda = np.array([10.0])
    seen_lambda_at_start = {}

    original = planner._dual_ascent_step

    def _track(tree, root_id):
        seen_lambda_at_start.setdefault("first", planner._lambda.copy())
        original(tree=tree, root_id=root_id)

    planner._dual_ascent_step = _track  # type: ignore[method-assign]
    try:
        planner.action(initial_belief)
    finally:
        planner._dual_ascent_step = original  # type: ignore[method-assign]
    np.testing.assert_array_equal(seen_lambda_at_start["first"], planner.lambda_init)


def test_dual_ascent_increases_lambda_when_qc_exceeds_budget(planner, initial_belief):
    """Ascent step pushes ``lambda`` up when cost-Q exceeds budget.

    Purpose: Verifies the sign and magnitude of the per-dimension update.

    Given: A tree with a single action child whose cost-Q exceeds the
        budget.
    When: ``_dual_ascent_step`` is invoked once.
    Then: ``_lambda`` is increased by exactly ``lambda_step * (qc - budget)``.

    Test type: unit
    """
    tree = Tree()
    root_id = tree.add_belief_node(initial_belief)
    action_id = tree.add_action_node(action=np.array([0.5, 0.0]), parent_id=root_id)
    tree.increment_visit_count(action_id)
    planner._set_cost_q(action_id, np.array([1.0]))
    planner._lambda = np.array([0.0])
    planner._dual_ascent_step(tree=tree, root_id=root_id)
    # qc=1.0, budget=0.5, step=0.1 → λ ← 0 + 0.1 * (1.0 - 0.5) = 0.05
    np.testing.assert_allclose(planner._lambda, np.array([0.05]))


def test_dual_ascent_clamps_lambda_at_zero(planner, initial_belief):
    """Ascent step keeps ``lambda >= 0`` element-wise (dual feasibility).

    Purpose: Lagrange multiplier on an inequality constraint cannot be negative.

    Given: A tree whose Lagrangian-best action has cost-Q far below budget
        and a small positive ``_lambda``.
    When: ``_dual_ascent_step`` is invoked.
    Then: ``_lambda`` is clamped to ``0.0`` rather than going negative.

    Test type: unit
    """
    tree = Tree()
    root_id = tree.add_belief_node(initial_belief)
    action_id = tree.add_action_node(action=np.array([0.1, 0.0]), parent_id=root_id)
    tree.increment_visit_count(action_id)
    planner._set_cost_q(action_id, np.array([0.0]))
    planner._lambda = np.array([0.01])
    planner._dual_ascent_step(tree=tree, root_id=root_id)
    # qc=0, budget=0.5 → λ ← max(0, 0.01 + 0.1*(0 - 0.5)) = max(0, -0.04) = 0
    np.testing.assert_array_equal(planner._lambda, np.array([0.0]))


def test_lagrangian_best_action_switches_with_lambda(planner, initial_belief):
    """Root pick flips from max-Q to min-QC as λ grows.

    Purpose: Verifies the Lagrangian root-pick interpolates between
        unconstrained-optimal and cost-minimal as the multiplier scales.

    Given: A root with two action children. Action A has higher reward and
        higher cost; action B has lower reward and zero cost.
    When: ``_lagrangian_best_action_id`` is queried with λ=0 vs a large λ.
    Then: λ=0 picks A; large λ picks B.

    Test type: unit
    """
    tree = Tree()
    root_id = tree.add_belief_node(initial_belief)
    a_high = tree.add_action_node(action=np.array([1.0, 0.0]), parent_id=root_id)
    a_safe = tree.add_action_node(action=np.array([-1.0, 0.0]), parent_id=root_id)
    tree.q_value[a_high] = 5.0
    tree.q_value[a_safe] = 1.0
    planner._set_cost_q(a_high, np.array([2.0]))
    planner._set_cost_q(a_safe, np.array([0.0]))

    planner._lambda = np.array([0.0])
    assert planner._lagrangian_best_action_id(tree=tree, belief_id=root_id) == a_high

    planner._lambda = np.array([5.0])
    assert planner._lagrangian_best_action_id(tree=tree, belief_id=root_id) == a_safe


def test_cost_q_backup_runs_running_mean(planner, initial_belief):
    """Cost-Q on an action node is the running mean of per-simulation returns.

    Purpose: Backup matches the standard MCTS incremental-mean rule for QC.

    Given: A two-children tree (one action, one belief child), simulated
        twice with explicit total-cost arrays via the private update helper.
    When: ``_update_node_statistics_with_cost`` is called twice with
        ``total_c=[1.0]`` then ``total_c=[3.0]``.
    Then: Action cost-Q is the unweighted mean ``[2.0]``.

    Test type: unit
    """
    tree = Tree()
    root_id = tree.add_belief_node(initial_belief)
    action_id = tree.add_action_node(action=np.array([0.0, 0.0]), parent_id=root_id)
    planner._update_node_statistics_with_cost(
        tree=tree, belief_id=root_id, action_id=action_id, total_v=0.0, total_c=np.array([1.0])
    )
    planner._update_node_statistics_with_cost(
        tree=tree, belief_id=root_id, action_id=action_id, total_v=0.0, total_c=np.array([3.0])
    )
    np.testing.assert_allclose(planner._cost_q(action_id), np.array([2.0]))


def test_terminal_belief_returns_action_without_search(environment, action_sampler):
    """Terminal beliefs short-circuit ``action()`` and skip tree search.

    Purpose: Avoid wasted search and dual-ascent updates on terminal beliefs.

    Given: A planner whose ``_is_terminal_belief`` returns ``True``.
    When: ``action()`` is called.
    Then: A one-element action list is returned and ``run_data.info_variables``
        is empty (search was skipped).

    Test type: unit
    """
    cpft = CPFT_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        name="t",
        action_sampler=action_sampler,
        cost_budget=0.5,
        n_simulations=5,
    )
    belief = get_initial_belief(pomdp=environment, n_particles=10, resampling=True)
    cpft._is_terminal_belief = lambda belief: True  # type: ignore[method-assign]
    actions, run_data = cpft.action(belief)
    assert len(actions) == 1
    assert run_data.info_variables == []


def test_full_search_populates_cost_q_after_action(action_sampler):
    """End-to-end: ``action()`` populates ``_action_cost_q``.

    Purpose: Confirms the cost track is exercised through the real search.

    Given: A planner backed by ``_OneCostEnv`` (constraint cost = 1.0 per
        transition).
    When: ``action()`` runs with 40 simulations.
    Then: ``_action_cost_q`` is non-empty; every recorded cost-Q[0] is
        non-negative and below the ``1/(1-γ)`` geometric ceiling.

    Test type: integration
    """
    env = _OneCostEnv(discount_factor=0.95)
    cpft = CPFT_DPW(
        environment=env,
        discount_factor=0.95,
        depth=4,
        name="full",
        action_sampler=action_sampler,
        cost_budget=10.0,
        lambda_step=0.05,
        n_simulations=40,
        k_a=1.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
    )
    belief = get_initial_belief(pomdp=env, n_particles=15, resampling=True)
    cpft.action(belief)
    assert cpft._action_cost_q
    upper_bound = 1.0 / (1.0 - 0.95) + 1e-6
    for qc in cpft._action_cost_q.values():
        assert qc.shape == (1,)
        assert qc[0] >= 0.0
        assert qc[0] <= upper_bound


def test_return_minimal_cost_toggle(action_sampler):
    """Both ``return_minimal_cost`` settings produce valid actions end-to-end.

    Purpose: Verifies the toggle gates the Jamgochian minimal-cost
        propagation trick without breaking the search.

    Given: Two planners identical except for ``return_minimal_cost``.
    When: Each runs ``action()`` with the same seed.
    Then: Both produce a valid action.

    Test type: integration
    """
    env = _OneCostEnv(discount_factor=0.95)
    belief = get_initial_belief(pomdp=env, n_particles=10, resampling=True)

    np.random.seed(7)
    random.seed(7)
    clamped = CPFT_DPW(
        environment=env,
        discount_factor=0.95,
        depth=3,
        name="clamped",
        action_sampler=action_sampler,
        cost_budget=10.0,
        return_minimal_cost=True,
        n_simulations=30,
    )
    actions_c, _ = clamped.action(belief)
    assert isinstance(actions_c[0], np.ndarray)

    np.random.seed(7)
    random.seed(7)
    unclamped = CPFT_DPW(
        environment=env,
        discount_factor=0.95,
        depth=3,
        name="unclamped",
        action_sampler=action_sampler,
        cost_budget=10.0,
        return_minimal_cost=False,
        n_simulations=30,
    )
    actions_u, _ = unclamped.action(belief)
    assert isinstance(actions_u[0], np.ndarray)


def test_constraint_cost_shape_mismatch_raises(action_sampler):
    """Mismatched constraint-dimensionality between env and ``cost_budget`` is caught.

    Purpose: Surfaces a configuration error early instead of silently
        propagating shape mismatches into the cost track.

    Given: An env that returns a length-2 constraint cost and a planner
        constructed with a scalar (length-1) ``cost_budget``.
    When: ``action()`` runs (the first new-belief-child expansion calls
        ``constraint_cost``).
    Then: ``ValueError`` is raised mentioning the shape mismatch.

    Test type: unit
    """

    class _TwoDimCostEnv(ContinuousLightDarkPOMDP, ConstrainedEnvironment):
        def constraint_cost(self, state: Any, action: Any, next_state: Any) -> np.ndarray:
            del state, action, next_state
            return np.array([0.0, 0.0])

    env = _TwoDimCostEnv(discount_factor=0.95)
    cpft = CPFT_DPW(
        environment=env,
        discount_factor=0.95,
        depth=2,
        name="mismatch",
        action_sampler=action_sampler,
        cost_budget=0.5,  # length 1, env returns length 2
        n_simulations=5,
    )
    belief = get_initial_belief(pomdp=env, n_particles=5, resampling=True)
    with pytest.raises(ValueError, match="constraint_cost returned shape"):
        cpft.action(belief)


def test_leaf_rollout_accumulates_multi_step_cost(action_sampler):
    """Cost-aware rollout reflects multi-step hazards, not just immediate cost.

    Purpose: Regression test for the prior "c_child = zeros" leaf treatment
        that underestimated QC for hazards beyond depth 1.

    Given: A planner backed by ``_OneCostEnv`` so every rollout transition
        contributes ``1`` to the cost. With ``depth=5`` the rollout horizon
        is 6 steps; the discounted cost is
        ``(1 − γ^6) / (1 − γ) ≈ 5.10`` for γ=0.95.
    When: ``action()`` runs enough simulations to score the root actions.
    Then: At least one action's cost-Q exceeds the single-step cost of
        ``1.0`` — proving the rollout's downstream cost is being backed up.

    Test type: integration
    """
    env = _OneCostEnv(discount_factor=0.95)
    cpft = CPFT_DPW(
        environment=env,
        discount_factor=0.95,
        depth=5,
        name="leaf_cost_rollout",
        action_sampler=action_sampler,
        cost_budget=10.0,
        lambda_step=0.01,  # small step so dual ascent doesn't dominate
        return_minimal_cost=False,  # don't let the clamp suppress QC
        n_simulations=80,
    )
    belief = get_initial_belief(pomdp=env, n_particles=10, resampling=True)
    cpft.action(belief)
    assert cpft._action_cost_q
    max_qc = max(qc[0] for qc in cpft._action_cost_q.values())
    # Single-step cost would be exactly 1.0; multi-step accumulation must
    # push at least one action's QC clearly above that.
    assert max_qc > 1.5, f"max action QC = {max_qc}; rollout cost not accumulating"


def test_minimal_cost_clamp_picks_lagrangian_min_sibling(planner, initial_belief):
    """The minimal-cost clamp returns one full sibling QC, not elementwise min.

    Purpose: Regression for the prior elementwise-``min`` behaviour that
        could synthesise a cost vector no real action achieved (e.g.
        siblings ``[1, 10]`` and ``[10, 1]`` previously clamped to
        ``[1, 1]``).

    Given: A tree with two action children at root whose QC vectors are
        ``[1.0]`` and ``[10.0]``, both visited, plus a third unvisited
        child seeded with a low (but unobserved) QC.
    When: ``_simulate_path_with_cost`` returns from the root.
    Then: With ``return_minimal_cost=True`` the propagated cost equals
        the Lagrangian-min sibling's QC verbatim (``[1.0]``), and the
        unvisited child is ignored.

    Test type: unit
    """
    # Build a root tree by hand with controlled siblings.
    tree = Tree()
    root_id = tree.add_belief_node(initial_belief)
    cheap_id = tree.add_action_node(action=np.array([0.0, 0.0]), parent_id=root_id)
    expensive_id = tree.add_action_node(action=np.array([1.0, 0.0]), parent_id=root_id)
    unvisited_id = tree.add_action_node(action=np.array([-1.0, 0.0]), parent_id=root_id)
    tree.increment_visit_count(cheap_id)
    tree.increment_visit_count(expensive_id)
    planner._set_cost_q(cheap_id, np.array([1.0]))
    planner._set_cost_q(expensive_id, np.array([10.0]))
    planner._set_cost_q(unvisited_id, np.array([0.0]))  # seeded but unvisited
    planner._lambda = np.array([1.0])  # gives the cheap sibling the lower Lagrangian score

    # Drive one simulation that ends up backing through the root. Easiest
    # path: monkeypatch _simulate_return_with_cost to return a known
    # (total_v, total_c) and let the clamp run on top.
    def _fake_return(*, tree, belief_id, action_id, depth):
        del tree, belief_id, action_id, depth
        # Pretend the chosen action had a high open-loop cost; the clamp
        # should replace it with the Lagrangian-min sibling's QC.
        return 0.0, np.array([5.0])

    original_return = planner._simulate_return_with_cost
    original_widening = planner._lagrangian_action_progressive_widening
    planner._simulate_return_with_cost = _fake_return  # type: ignore[method-assign]
    planner._lagrangian_action_progressive_widening = (  # type: ignore[method-assign]
        lambda tree, belief_id: expensive_id
    )
    try:
        _, propagated_c = planner._simulate_path_with_cost(tree=tree, belief_id=root_id, depth=0)
    finally:
        planner._simulate_return_with_cost = original_return  # type: ignore[method-assign]
        planner._lagrangian_action_progressive_widening = (  # type: ignore[method-assign]
            original_widening
        )

    # Lagrangian-min sibling at belief is the cheap one (QC=[1.0] vs
    # expensive=[10.0]). Unvisited is excluded. Propagated cost equals
    # the cheap sibling's QC verbatim.
    np.testing.assert_array_equal(propagated_c, np.array([1.0]))


def test_minimal_cost_clamp_preserves_real_vectors(action_sampler):
    """Vector clamp picks one sibling's qc as-is — never an unrealisable mix.

    Purpose: Regression for the elementwise-``np.minimum`` bug that could
        return ``[0, 0]`` when one sibling had ``[0, 10]`` and another had
        ``[10, 0]``.

    Given: A 2-D constraint env, a planner whose root has two visited
        action children with orthogonal cost vectors, and a fake
        ``_simulate_return_with_cost`` to feed a known ``total_c``.
    When: ``_simulate_path_with_cost`` runs at the root.
    Then: The propagated cost equals one of the two sibling QC vectors
        verbatim (whichever has the lower Lagrangian score), NOT the
        elementwise minimum.

    Test type: unit
    """

    class _TwoDimCostEnv(ContinuousLightDarkPOMDP, ConstrainedEnvironment):
        def constraint_cost(self, state: Any, action: Any, next_state: Any) -> np.ndarray:
            del state, action, next_state
            return np.array([0.0, 0.0])

    env = _TwoDimCostEnv(discount_factor=0.95)
    cpft = CPFT_DPW(
        environment=env,
        discount_factor=0.95,
        depth=2,
        name="vector_clamp",
        action_sampler=action_sampler,
        cost_budget=np.array([1.0, 1.0]),
        n_simulations=5,
    )
    belief = get_initial_belief(pomdp=env, n_particles=5, resampling=True)

    tree = Tree()
    root_id = tree.add_belief_node(belief)
    a_zero_ten = tree.add_action_node(action=np.array([0.0, 0.0]), parent_id=root_id)
    a_ten_zero = tree.add_action_node(action=np.array([1.0, 0.0]), parent_id=root_id)
    tree.increment_visit_count(a_zero_ten)
    tree.increment_visit_count(a_ten_zero)
    cpft._set_cost_q(a_zero_ten, np.array([0.0, 10.0]))
    cpft._set_cost_q(a_ten_zero, np.array([10.0, 0.0]))
    cpft._lambda = np.array([1.0, 1.0])  # equal weights → tied; tie broken by argmin order

    def _fake_return(*, tree, belief_id, action_id, depth):
        del tree, belief_id, action_id, depth
        return 0.0, np.array([5.0, 5.0])

    cpft._simulate_return_with_cost = _fake_return  # type: ignore[method-assign]
    cpft._lagrangian_action_progressive_widening = (  # type: ignore[method-assign]
        lambda tree, belief_id: a_zero_ten
    )
    _, propagated_c = cpft._simulate_path_with_cost(tree=tree, belief_id=root_id, depth=0)

    # Result must equal ONE of the sibling QC vectors verbatim, never the
    # elementwise min ([0, 0]).
    assert propagated_c.tolist() in (
        [0.0, 10.0],
        [10.0, 0.0],
    ), f"clamp produced {propagated_c}, expected a real sibling QC"


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "inf", "neg_inf"],
)
def test_non_finite_cost_budget_rejected(environment, action_sampler, bad_value):
    """``cost_budget`` containing NaN / ±inf is rejected at construction.

    Purpose: Regression for the bug where ``(arr < 0).any()`` silently
        admitted NaN (since ``NaN < 0`` is ``False``) and let λ go to NaN
        downstream.

    Given: A CPFT_DPW constructor call with ``cost_budget`` set to one of
        ``nan``, ``+inf``, ``-inf``.
    When: ``__init__`` runs.
    Then: ``ValueError`` is raised mentioning either "finite" or
        "non-negative" (``-inf`` may trip either guard depending on
        ordering).

    Test type: unit
    """
    with pytest.raises(ValueError, match="cost_budget"):
        CPFT_DPW(
            environment=environment,
            discount_factor=0.95,
            depth=2,
            name="bad",
            action_sampler=action_sampler,
            cost_budget=bad_value,
            n_simulations=5,
        )


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "inf", "neg_inf"],
)
def test_non_finite_lambda_init_rejected(environment, action_sampler, bad_value):
    """``lambda_init`` containing NaN / ±inf is rejected at construction.

    Purpose: Same regression coverage as cost_budget but for the
        dual variable's initial value.

    Given: A CPFT_DPW constructor call with ``lambda_init`` set to one of
        ``nan``, ``+inf``, ``-inf``.
    When: ``__init__`` runs.
    Then: ``ValueError`` is raised mentioning ``lambda_init``.

    Test type: unit
    """
    with pytest.raises(ValueError, match="lambda_init"):
        CPFT_DPW(
            environment=environment,
            discount_factor=0.95,
            depth=2,
            name="bad",
            action_sampler=action_sampler,
            cost_budget=0.5,
            lambda_init=bad_value,
            n_simulations=5,
        )


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "inf", "neg_inf"],
)
def test_non_finite_lambda_step_rejected(environment, action_sampler, bad_value):
    """``lambda_step`` set to NaN / ±inf is rejected at construction.

    Purpose: A non-finite step size would drive λ to NaN / inf after the
        first ascent update.

    Given: A CPFT_DPW constructor call with ``lambda_step`` set to one of
        ``nan``, ``+inf``, ``-inf``.
    When: ``__init__`` runs.
    Then: ``ValueError`` is raised mentioning ``lambda_step``.

    Test type: unit
    """
    with pytest.raises(ValueError, match="lambda_step"):
        CPFT_DPW(
            environment=environment,
            discount_factor=0.95,
            depth=2,
            name="bad",
            action_sampler=action_sampler,
            cost_budget=0.5,
            lambda_step=bad_value,
            n_simulations=5,
        )


def test_empty_cost_budget_rejected(environment, action_sampler):
    """An empty ``cost_budget`` is rejected at construction.

    Purpose: Regression for a quiet-no-op bug where ``cost_budget=[]``
        produced ``n_constraints=0`` and the planner silently ran
        unconstrained while still claiming to enforce a constraint.

    Given: A CPFT_DPW constructor call with ``cost_budget=[]``.
    When: ``__init__`` runs.
    Then: ``ValueError`` is raised mentioning ``constraint dimension``.

    Test type: unit
    """
    with pytest.raises(ValueError, match="constraint dimension"):
        CPFT_DPW(
            environment=environment,
            discount_factor=0.95,
            depth=2,
            name="bad",
            action_sampler=action_sampler,
            cost_budget=np.array([]),
            n_simulations=5,
        )


def test_lambda_init_accepts_zero_d_ndarray(environment, action_sampler):
    """``lambda_init=np.array(0.0)`` (0-d ndarray) is accepted.

    Purpose: Regression for an asymmetry where Python ``float`` and 1-D
        arrays were accepted but 0-D arrays (a common numpy idiom) were
        rejected with a misleading shape error.

    Given: A CPFT_DPW constructor with ``lambda_init=np.array(0.0)``.
    When: ``__init__`` runs.
    Then: The planner constructs and stores ``lambda_init`` as a 1-D array
        of shape ``(1,)`` matching the scalar ``cost_budget``.

    Test type: unit
    """
    cpft = CPFT_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=2,
        name="zero_d_lambda",
        action_sampler=action_sampler,
        cost_budget=0.5,
        lambda_init=np.array(0.0),
        n_simulations=5,
    )
    np.testing.assert_array_equal(cpft.lambda_init, np.array([0.0]))


def test_lambda_init_scalar_broadcasts_to_vector_budget(environment, action_sampler):
    """A length-1 ``lambda_init`` broadcasts to a multi-dim ``cost_budget``.

    Purpose: Convenience guarantee — users supplying multi-dim budgets
        shouldn't need to pre-broadcast a uniform initial λ themselves.

    Given: A CPFT_DPW constructor with a length-3 ``cost_budget`` and
        ``lambda_init=0.0`` (or ``np.array([0.0])``).
    When: ``__init__`` runs.
    Then: ``lambda_init`` is stored as ``[0.0, 0.0, 0.0]``.

    Test type: unit
    """
    cpft = CPFT_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=2,
        name="broadcast",
        action_sampler=action_sampler,
        cost_budget=np.array([1.0, 2.0, 3.0]),
        lambda_init=np.array([0.0]),
        n_simulations=5,
    )
    np.testing.assert_array_equal(cpft.lambda_init, np.zeros(3))


def test_non_finite_constraint_cost_from_env_rejected(action_sampler):
    """A NaN / ±inf returned from ``env.constraint_cost`` raises at search time.

    Purpose: Defensive boundary check. Without it, a buggy env that
        produced ``NaN`` for a single transition would silently poison QC,
        ``_lambda``, and Lagrangian scores for the rest of the search.

    Given: A ``ConstrainedEnvironment`` mock that returns ``np.array([nan])``
        from ``constraint_cost``.
    When: ``CPFT_DPW.action(belief)`` runs (first new-belief expansion
        invokes ``constraint_cost``).
    Then: ``ValueError`` is raised mentioning "non-finite".

    Test type: unit
    """

    class _NaNCostEnv(ContinuousLightDarkPOMDP, ConstrainedEnvironment):
        def constraint_cost(self, state: Any, action: Any, next_state: Any) -> np.ndarray:
            del state, action, next_state
            return np.array([float("nan")])

    env = _NaNCostEnv(discount_factor=0.95)
    cpft = CPFT_DPW(
        environment=env,
        discount_factor=0.95,
        depth=2,
        name="nan_env",
        action_sampler=action_sampler,
        cost_budget=0.5,
        n_simulations=5,
    )
    belief = get_initial_belief(pomdp=env, n_particles=5, resampling=True)
    with pytest.raises(ValueError, match="non-finite"):
        cpft.action(belief)


def test_lambda_max_computed_from_reward_range_and_budget(environment, action_sampler):
    """``_lambda_max`` equals ``(R_max − R_min) / ((1−γ) · budget)`` per dim.

    Purpose: Regression for the λ stability cap that prevents dual ascent
        from running away on infeasible problems.

    Given: A planner whose env has a known ``reward_range``.
    When: The planner is constructed.
    Then: ``_lambda_max`` equals the closed-form formula element-wise.

    Test type: unit
    """
    cpft = CPFT_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=2,
        name="cap",
        action_sampler=action_sampler,
        cost_budget=np.array([0.5, 2.0]),
        n_simulations=5,
    )
    r_min, r_max = environment.reward_range
    expected = (r_max - r_min) / ((1.0 - 0.95) * np.array([0.5, 2.0]))
    np.testing.assert_allclose(cpft._lambda_max, expected)


def test_dual_ascent_clips_lambda_at_lambda_max(environment, action_sampler):
    """``λ`` is upper-clipped at ``_lambda_max`` even when ascent would exceed it.

    Purpose: Regression for the unbounded-λ growth observed on infeasible
        problems before the cap was added.

    Given: A planner with ``_lambda_max`` overridden to a small value and
        ``_lambda`` already at the cap; a tree showing a violated constraint
        (QC well above budget).
    When: ``_dual_ascent_step`` runs.
    Then: ``_lambda`` is clipped at ``_lambda_max``, not pushed beyond it.

    Test type: unit
    """
    cpft = CPFT_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=2,
        name="cap_fires",
        action_sampler=action_sampler,
        cost_budget=0.5,
        lambda_step=1.0,
        n_simulations=5,
    )
    cpft._lambda_max = np.array([10.0])
    cpft._lambda = np.array([9.5])

    tree = Tree()
    root_id = tree.add_belief_node(
        get_initial_belief(pomdp=environment, n_particles=5, resampling=True)
    )
    action_id = tree.add_action_node(action=np.array([0.0, 0.0]), parent_id=root_id)
    tree.increment_visit_count(action_id)
    cpft._set_cost_q(action_id, np.array([100.0]))  # huge violation
    cpft._dual_ascent_step(tree=tree, root_id=root_id)
    # Without the cap: λ ← 9.5 + 1.0 * (100 − 0.5) = 109. With cap: 10.
    np.testing.assert_array_equal(cpft._lambda, np.array([10.0]))


def test_lambda_max_is_infinity_when_env_has_no_reward_range(action_sampler):
    """Envs without a ``reward_range`` leave ``λ`` unbounded above.

    Purpose: Document the fallback path — older envs without a declared
        reward range should still construct and behave like the pre-cap
        planner (lower-clipped at 0 only).

    Given: A ConstrainedEnvironment mock whose ``reward_range`` is None.
    When: A planner is built on it.
    Then: ``_lambda_max`` is all ``+inf`` and dual ascent applies only the
        non-negative floor.

    Test type: unit
    """

    class _NoRangeEnv(ContinuousLightDarkPOMDP, ConstrainedEnvironment):
        def __init__(self):
            super().__init__(discount_factor=0.95)
            self.reward_range = None  # type: ignore[assignment]

        def constraint_cost(self, state: Any, action: Any, next_state: Any) -> np.ndarray:
            del state, action, next_state
            return np.array([0.0])

    env = _NoRangeEnv()
    cpft = CPFT_DPW(
        environment=env,
        discount_factor=0.95,
        depth=2,
        name="no_range",
        action_sampler=action_sampler,
        cost_budget=0.5,
        n_simulations=5,
    )
    assert np.isinf(cpft._lambda_max).all()
