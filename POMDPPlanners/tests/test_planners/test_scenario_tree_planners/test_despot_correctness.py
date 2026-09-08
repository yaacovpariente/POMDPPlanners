# SPDX-License-Identifier: MIT

"""DESPOT correctness: bounds arithmetic, determinization, descent, selection.

``test_despot.py`` checks that the planner returns a legal action and a
well-formed tree. None of that separates DESPOT from a planner that discounts
the wrong term, normalizes by the wrong weight, descends on the lower bound, or
answers with the optimistic action. These tests pin the equations of Somani et
al. (2013) / Ye et al. (2017) on fixtures small enough to work out by hand, and
on synthetic trees where the correct and the plausible-but-wrong rule give
different answers.

Equation numbers refer to the module docstring of
``POMDPPlanners.planners.scenario_tree_planners.despot``.

References:
    Somani, Ye, Hsu & Lee (2013), DESPOT: Online POMDP Planning with
    Regularization, NeurIPS 26. Ye, Somani, Hsu & Lee (2017), JAIR 58.
"""

# pylint: disable=protected-access

import math
import random
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock

import numpy as np
import pytest

from POMDPPlanners.core.belief import UnweightedParticleBeliefStateUpdate
from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.tree.arena import ACTION, BELIEF, Tree
from POMDPPlanners.planners.scenario_tree_planners.despot import (
    DESPOT,
    TERMINAL_OBSERVATION,
    _BeliefNodeData,
)
from POMDPPlanners.tests.test_planners.planner_fixtures import (
    CHAIN_REWARDS,
    END,
    NEXT,
    ROOT,
    ChainEnv,
    chain_belief,
)


DISCOUNT = 0.5

#: Rewards are integers and the discount is 0.5, so every expected value below
#: is an exactly representable binary fraction. Anything above this tolerance
#: is a real arithmetic error, not float accumulation.
TOL = 1e-12

#: ``ChainEnv`` declares ``reward_range=(0.0, 4.0)``, so this is the ``R_max``
#: DESPOT resolves for it. Written out here rather than read off the planner so
#: the expected upper bounds below do not come from the code under test.
CHAIN_MAX_REWARD = 4.0


def _chain_planner(
    environment: Any,
    depth: int,
    n_scenarios: int = 4,
    n_simulations: int = 10,
    **overrides: Any,
) -> DESPOT:
    params: Dict[str, Any] = dict(
        environment=environment,
        discount_factor=environment.discount_factor,
        depth=depth,
        name="DESPOT_correctness",
        n_scenarios=n_scenarios,
        n_simulations=n_simulations,
    )
    params.update(overrides)
    return DESPOT(**params)


def _geometric(discount: float, n_terms: int) -> float:
    """``sum_{t<n} g^t``, written out so no test reads it off the planner."""
    return float(sum(discount**t for t in range(n_terms)))


# ---------------------------------------------------------------------------
# A stochastic fixture, for the checks a deterministic chain cannot make
# ---------------------------------------------------------------------------


class CoinEnv(DiscreteActionsEnvironment):
    """A one-step coin flip per transition, with every draw recorded.

    ``ChainEnv`` is deterministic, which is what makes its arithmetic
    hand-checkable -- and exactly why it cannot test determinization: with no
    randomness there is nothing to pin. Here the successor and observation are
    ``numpy.random.random() < 0.5``, so two branches that share a scenario and a
    depth must produce *identical* draws if the scenario streams are working,
    and will differ almost surely if they are not.

    ``draws`` records ``(state, action, uniform)`` for every transition, so a
    test can compare the numbers the environment actually saw rather than
    inferring them from the tree.
    """

    def __init__(self, discount_factor: float = 0.5) -> None:
        super().__init__(
            discount_factor=discount_factor,
            name="CoinEnv",
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE,
                observation_space=SpaceType.DISCRETE,
            ),
            reward_range=(0.0, 1.0),
        )
        self.draws: List[Tuple[Any, Any, float]] = []

    def get_actions(self) -> List[Any]:
        return ["a", "b"]

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> Any:
        uniform = float(np.random.random())
        self.draws.append((state, action, uniform))
        successor = ("heads" if uniform < 0.5 else "tails", int(str(state).count("/")) + 1)
        return successor if n_samples == 1 else [successor] * n_samples

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        del action
        return next_state[0] if n_samples == 1 else [next_state[0]] * n_samples

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        del state, action
        return 1.0 if next_state is not None and next_state[0] == "heads" else 0.0

    def is_terminal(self, state: Any) -> bool:
        del state
        return False

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        del state, action
        return np.full(len(next_states), math.log(0.5))

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        del action
        return np.array(
            [0.0 if obs == next_state[0] else -50.0 for obs in observations], dtype=np.float64
        )

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return observation1 == observation2

    def hash_observation(self, observation: Any) -> Any:
        return observation

    def hash_action(self, action: Any) -> Any:
        return action

    def initial_state_dist(self) -> Distribution:
        dist = Mock(spec=Distribution)
        dist.sample = Mock(return_value=[("start", 0)])
        return dist

    def initial_observation_dist(self) -> Distribution:
        dist = Mock(spec=Distribution)
        dist.sample = Mock(return_value=["start"])
        return dist


class UnhashableObservationEnv(ChainEnv):
    """``hash_observation`` returns a list, which cannot key an observation branch."""

    def hash_observation(self, observation: Any) -> Any:
        return [observation]


# ---------------------------------------------------------------------------
# Equation 6: leaf bounds
# ---------------------------------------------------------------------------


def test_leaf_upper_bound_is_max_reward_sustained_over_the_remaining_horizon():
    """``u_0(b) = R_max * sum_{t < D-d} gamma^t``.

    Purpose: Pins equation (6)'s upper bound, including that it is truncated at
    the search horizon rather than run to infinity.

    Given: ``ChainEnv`` (``R_max = 4``), discount ``0.5``, search depth 3, a
        fresh root at depth 0.
    When: The root node is allocated.
    Then: Its upper bound is ``4 * (1 + 0.5 + 0.25) = 7``.

    An infinite-horizon bound would give ``4 / 0.5 = 8``, and a bound that
    forgot to discount would give ``12``, so this fixture separates the three.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, n_simulations=0)
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))

    expected = CHAIN_MAX_REWARD * _geometric(DISCOUNT, 3)
    assert tree.upper_confidence_bound[root_id] == pytest.approx(expected, abs=TOL), (
        f"root upper bound {tree.upper_confidence_bound[root_id]} != R_max * sum_(t<3) 0.5^t "
        f"= {expected}; an infinite-horizon bound would be "
        f"{CHAIN_MAX_REWARD / (1 - DISCOUNT)}"
    )


def test_leaf_lower_bound_is_the_default_policys_own_discounted_return():
    """``l_0(b)`` is the return the default policy actually collects.

    Purpose: Pins equation (6)'s lower bound and its discount placement.

    Given: ``ChainEnv`` from ``root``: reward 2 leaving ``root``, 4 leaving
        ``next``, and ``end`` terminal with reward 0. Discount ``0.5``, search
        depth 3. Every action has the same effect, so the default policy's
        choice cannot change the number.
    When: The root node is allocated.
    Then: ``l_0(root) = 2 + 0.5 * 4 = 4``.

    Counting the terminal step would give ``4``, dropping the discount ``6``.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, n_simulations=0)
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))

    expected = CHAIN_REWARDS[ROOT] + DISCOUNT * CHAIN_REWARDS[NEXT]
    assert tree.lower_confidence_bound[root_id] == pytest.approx(expected, abs=TOL), (
        f"root lower bound {tree.lower_confidence_bound[root_id]} != "
        f"{CHAIN_REWARDS[ROOT]} + {DISCOUNT} * {CHAIN_REWARDS[NEXT]} = {expected}"
    )
    assert tree.data[root_id].default_value == pytest.approx(
        expected, abs=TOL
    ), "default_value must keep l_0 for the regularized pass even after l(b) rises"


def test_a_node_whose_scenarios_have_all_terminated_has_a_zero_width_interval():
    """A terminated node is worth exactly zero, so nothing can be gained there.

    Purpose: Establishes the stopping condition equation (5) relies on -- a
    zero-width interval makes excess uncertainty negative, so no trial descends
    into a terminated branch and no reward is charged twice.

    Given: ``ChainEnv`` with a belief concentrated on the terminal ``end``
        state, search depth 3.
    When: The node is allocated.
    Then: Both bounds are ``0.0`` and the node is flagged terminal.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, n_simulations=0)
    tree = Tree()
    node_id = planner._add_belief_node(
        tree=tree,
        states=[END, END],
        scenario_ids=[0, 1],
        depth=1,
        parent_id=None,
        observation=None,
        obs_key=None,
    )

    assert tree.data[node_id].terminal is True
    assert tree.lower_confidence_bound[node_id] == 0.0
    assert tree.upper_confidence_bound[node_id] == 0.0


def test_a_node_at_the_search_horizon_has_a_zero_width_interval():
    """The search optimises a ``depth``-step return, so depth ``D`` is worth zero.

    Purpose: Pins the finite-horizon semantics both bounds share, which is what
    makes ``l <= V_D <= u`` an equality-tight bracket rather than a claim about
    the infinite-horizon value.

    Given: ``ChainEnv``, search depth 2, a non-terminal ``root`` state placed at
        depth 2.
    When: The node is allocated.
    Then: Both bounds are ``0.0`` even though the state is not terminal.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=2, n_simulations=0)
    tree = Tree()
    node_id = planner._add_belief_node(
        tree=tree,
        states=[ROOT],
        scenario_ids=[0],
        depth=2,
        parent_id=None,
        observation=None,
        obs_key=None,
    )

    assert tree.data[node_id].terminal is False, "the state itself is not terminal"
    assert tree.lower_confidence_bound[node_id] == 0.0
    assert tree.upper_confidence_bound[node_id] == 0.0


def test_a_shortened_rollout_charges_the_uncovered_horizon_at_min_reward():
    """Truncating the default policy must not be mistaken for zero future reward.

    Purpose: A rollout that stops early and assumes zero afterwards is an
    over-estimate whenever rewards can be negative, and an over-estimating
    "lower bound" silently voids branch-and-bound.

    Given: ``ChainEnv`` with a reward range widened to ``(-6, 4)``, discount
        ``0.5``, search depth 3, and ``rollout_depth=1`` so only the first step
        is simulated.
    When: The root is allocated.
    Then: ``l_0(root) = 2 + 0.5 * (-6) * (1 + 0.5) = -2.5`` -- the one simulated
        reward plus the worst case for the two steps not simulated.

    Assuming zero for the tail would give ``2``.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    env.reward_range = (-6.0, 4.0)
    planner = _chain_planner(env, depth=3, rollout_depth=1, n_simulations=0)
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))

    expected = CHAIN_REWARDS[ROOT] + DISCOUNT * (-6.0) * _geometric(DISCOUNT, 2)
    assert tree.lower_confidence_bound[root_id] == pytest.approx(expected, abs=TOL), (
        f"shortened rollout gave {tree.lower_confidence_bound[root_id]}, expected "
        f"{expected}; assuming zero for the uncovered horizon would give "
        f"{CHAIN_REWARDS[ROOT]}"
    )


# ---------------------------------------------------------------------------
# Equations 1-3: expansion, first-step reward, action intervals
# ---------------------------------------------------------------------------


def test_first_step_reward_is_the_scenario_weighted_mean_not_a_sum():
    """``rho(b,a) = sum_phi w_phi r / w_b``.

    Purpose: Separates the weighted *mean* from a weighted *sum*. With every
    scenario at the same state the two differ by exactly the scenario count,
    so a missing normalization is a factor-of-K error, not a rounding one.

    Given: ``ChainEnv``, 4 scenarios all at ``root`` (reward 2), search depth 3.
    When: The root is expanded.
    Then: Every action child's immediate reward is ``2.0``, not ``8.0``.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, n_scenarios=4, n_simulations=0)
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))
    planner._expand(tree=tree, belief_id=root_id)

    action_children = tree.get_children_ids(root_id)
    assert len(action_children) == len(env.get_actions())
    for action_id in action_children:
        assert tree.get_immediate_reward(action_id) == pytest.approx(
            CHAIN_REWARDS[ROOT], abs=TOL
        ), (
            f"action node {action_id} (action={tree.get_action(action_id)!r}) has "
            f"rho={tree.get_immediate_reward(action_id)}, expected {CHAIN_REWARDS[ROOT]}; "
            f"an unnormalized sum over 4 scenarios would give "
            f"{4 * CHAIN_REWARDS[ROOT]}"
        )


def test_action_interval_discounts_the_children_exactly_once():
    """``Q_l(b,a) = rho(b,a) + gamma L(b,a)`` and the same for the upper bound.

    Purpose: Pins equation (3)'s discount placement, the one arithmetic slip
    that leaves every value plausible-looking but wrong.

    Given: ``ChainEnv``, discount ``0.5``, search depth 3, 4 scenarios at
        ``root``. The transition is deterministic, so each action has exactly
        one observation child, holding all 4 scenarios at ``next``.
    When: The root is expanded.
    Then: The child has ``l_0 = 4`` (reward 4 then terminal) and
        ``u_0 = 4 * (1 + 0.5) = 6``, so every action has
        ``Q_l = 2 + 0.5 * 4 = 4`` and ``Q_u = 2 + 0.5 * 6 = 5``.

    Undiscounted children would give ``6`` and ``8``; double-discounted, ``3``
    and ``3.5``.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, n_scenarios=4, n_simulations=0)
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))
    planner._expand(tree=tree, belief_id=root_id)

    expected_child_lower = CHAIN_REWARDS[NEXT]
    expected_child_upper = CHAIN_MAX_REWARD * _geometric(DISCOUNT, 2)
    expected_q_lower = CHAIN_REWARDS[ROOT] + DISCOUNT * expected_child_lower
    expected_q_upper = CHAIN_REWARDS[ROOT] + DISCOUNT * expected_child_upper

    checked = 0
    for action_id in tree.get_children_ids(root_id):
        belief_children = tree.get_children_ids(action_id)
        assert len(belief_children) == 1, (
            f"ChainEnv is deterministic, so action {tree.get_action(action_id)!r} must have "
            f"exactly one observation child, got {len(belief_children)}"
        )
        child_id = belief_children[0]
        assert tree.lower_confidence_bound[child_id] == pytest.approx(expected_child_lower, abs=TOL)
        assert tree.upper_confidence_bound[child_id] == pytest.approx(expected_child_upper, abs=TOL)
        assert tree.lower_confidence_bound[action_id] == pytest.approx(expected_q_lower, abs=TOL), (
            f"Q_l at action node {action_id} is {tree.lower_confidence_bound[action_id]}, "
            f"expected {CHAIN_REWARDS[ROOT]} + {DISCOUNT} * {expected_child_lower} = "
            f"{expected_q_lower}"
        )
        assert tree.upper_confidence_bound[action_id] == pytest.approx(expected_q_upper, abs=TOL)
        assert tree.q_value[action_id] == pytest.approx(expected_q_lower, abs=TOL), (
            "q_value must carry the lower bound, because that is what the final "
            "decision is made on"
        )
        checked += 1
    assert checked == len(env.get_actions())


def test_terminated_scenarios_get_their_own_zero_valued_branch():
    """Terminated scenarios keep their weight but earn nothing further.

    Purpose: Establishes that a terminated scenario is neither re-simulated
    (which would charge its terminal reward twice) nor dropped (which would
    renormalize the remaining branches and inflate the node's value).

    Given: ``ChainEnv`` with two scenarios, one at ``next`` and one already at
        the terminal ``end``, search depth 3.
    When: The node is expanded.
    Then: Each action has two observation children -- one for the surviving
        scenario and one keyed by the terminal sentinel with both bounds zero --
        their weights sum to the parent's, and ``rho`` is ``4 * 0.5 = 2``: the
        surviving scenario's reward halved by the dead one's weight, not the
        full ``4``.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, n_scenarios=2, n_simulations=0)
    tree = Tree()
    node_id = planner._add_belief_node(
        tree=tree,
        states=[NEXT, END],
        scenario_ids=[0, 1],
        depth=0,
        parent_id=None,
        observation=None,
        obs_key=None,
    )
    planner._root_id = node_id
    planner._expand(tree=tree, belief_id=node_id)

    for action_id in tree.get_children_ids(node_id):
        children = tree.get_children_ids(action_id)
        assert len(children) == 2, (
            f"action {tree.get_action(action_id)!r} should split into a live branch and "
            f"the terminal branch, got {len(children)}"
        )
        terminal_children = [
            cid for cid in children if tree.get_observation(cid) is TERMINAL_OBSERVATION
        ]
        assert len(terminal_children) == 1
        terminal_id = terminal_children[0]
        assert tree.lower_confidence_bound[terminal_id] == 0.0
        assert tree.upper_confidence_bound[terminal_id] == 0.0

        total_weight = sum(tree.weight[cid] for cid in children)
        assert total_weight == pytest.approx(tree.weight[node_id], abs=TOL), (
            f"child weights sum to {total_weight}, parent weight is {tree.weight[node_id]}; "
            f"a dropped scenario would make them disagree"
        )
        assert tree.get_immediate_reward(action_id) == pytest.approx(
            CHAIN_REWARDS[NEXT] * 0.5, abs=TOL
        ), (
            f"rho={tree.get_immediate_reward(action_id)}; the terminated scenario must "
            f"contribute reward 0 at its full weight, giving {CHAIN_REWARDS[NEXT] * 0.5}, "
            f"not {CHAIN_REWARDS[NEXT]}"
        )


# ---------------------------------------------------------------------------
# Determinization
# ---------------------------------------------------------------------------


def test_a_scenario_draws_the_same_random_number_in_every_action_branch():
    """The defining property: the tree is determinized over ``K`` fixed draws.

    Purpose: Without this, DESPOT is sparse sampling with extra bookkeeping --
    the paper's bound is stated over ``K`` fixed scenarios, and two branches
    that disagree about a scenario's future are not searching the same tree.

    Given: ``CoinEnv``, whose every transition draws one uniform and records it,
        with two actions and 8 scenarios. The root expansion pushes all 8
        scenarios through action ``a`` and then all 8 through action ``b``, at
        the same depth.
    When: The root is expanded.
    Then: The 8 uniforms drawn under ``a`` equal the 8 drawn under ``b``,
        in order; and not all 8 are equal to each other, so the check is about
        determinization rather than about a degenerate generator.

    Test type: unit
    """
    env = CoinEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=2, n_scenarios=8, n_simulations=0)
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))
    env.draws.clear()
    planner._expand(tree=tree, belief_id=root_id)

    # Expansion transitions are the only ones that leave the root state; the
    # leaf lower bounds roll out from the freshly created children, whose states
    # are tuples. Filtering on the source state separates the two rather than
    # relying on call order.
    uniforms_by_action: Dict[Any, List[float]] = {}
    for state, action, uniform in env.draws:
        if state == ROOT:
            uniforms_by_action.setdefault(action, []).append(uniform)

    assert set(uniforms_by_action) == {"a", "b"}
    assert len(uniforms_by_action["a"]) == 8
    assert uniforms_by_action["a"] == uniforms_by_action["b"], (
        "the same scenario at the same depth drew different numbers under two "
        f"actions: {uniforms_by_action['a']} vs {uniforms_by_action['b']}"
    )
    assert len(set(uniforms_by_action["a"])) > 1, (
        "all 8 scenarios drew the same number, so this fixture cannot tell "
        "determinization from a stuck generator"
    )


def test_turning_determinization_off_breaks_the_shared_draw():
    """The flag does what it says, so an ablation is a real ablation.

    Purpose: Guards against the flag silently being a no-op, which would make
    ``use_determinized_scenarios=False`` claim an honest fallback while still
    determinizing (or the reverse).

    Given: The same ``CoinEnv`` root expansion with
        ``use_determinized_scenarios=False`` and 16 scenarios.
    When: The root is expanded.
    Then: The two actions' draw sequences differ. With 16 independent uniforms
        per action the chance of coincidental equality is zero.

    Test type: unit
    """
    env = CoinEnv(discount_factor=DISCOUNT)
    np.random.seed(11)
    random.seed(11)
    planner = _chain_planner(
        env, depth=2, n_scenarios=16, n_simulations=0, use_determinized_scenarios=False
    )
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))
    env.draws.clear()
    planner._expand(tree=tree, belief_id=root_id)

    uniforms_by_action: Dict[Any, List[float]] = {}
    for state, action, uniform in env.draws:
        if state == ROOT:
            uniforms_by_action.setdefault(action, []).append(uniform)
    assert len(uniforms_by_action["a"]) == 16
    assert (
        uniforms_by_action["a"] != uniforms_by_action["b"]
    ), "with determinization off the two actions must draw independently"


def test_planning_leaves_the_callers_random_streams_where_it_found_them():
    """Seeding the global generators must not reshape the episode's randomness.

    Purpose: The planner pins transitions by seeding the process-global
    generators. If it did not put them back, every draw the episode runner made
    after a decision would be a deterministic function of the planner's last
    transition -- the simulation's randomness would come from the planner.

    Given: ``ChainEnv`` and a planner with determinization on.
    When: One decision is taken.
    Then: Both global generator states are byte-identical to before the call.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, n_simulations=5)

    np.random.seed(3)
    random.seed(3)
    numpy_before = np.random.get_state()
    python_before = random.getstate()

    planner.action(chain_belief(ROOT))

    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert python_before == random.getstate()


def test_the_default_policy_is_one_action_sequence_per_node_not_per_scenario():
    """A default policy that reads the scenario is clairvoyant, not a policy.

    Purpose: Regression. An earlier version drew the rollout action from each
    scenario's own seed, so different scenarios at the same node took different
    actions. The resulting value is achievable only by something that already
    knows the hidden state, so it is not a lower bound on anything the planner
    could execute -- and it showed up as ``l > u`` crossings during backup.

    Given: ``CoinEnv``, 12 scenarios, search depth 4.
    When: The default policy's action list is requested for a node.
    Then: It has one entry per remaining step, and requesting it again for the
        same node gives the same list, while a node keyed on a different
        scenario gives a different one (so the sequence is not simply constant).

    Test type: unit
    """
    env = CoinEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=4, n_scenarios=12, n_simulations=0)
    planner._learn_tree(belief=chain_belief(ROOT))

    first = planner._default_policy_actions(reference_scenario_id=0, depth=1)
    again = planner._default_policy_actions(reference_scenario_id=0, depth=1)
    assert len(first) == 3, f"expected one action per remaining step (4 - 1), got {len(first)}"
    assert first == again, "the default policy must be a function of the node, not a fresh draw"

    others = [
        planner._default_policy_actions(reference_scenario_id=sid, depth=1) for sid in range(12)
    ]
    assert any(other != first for other in others), (
        "every node produced the identical sequence, so this check cannot tell a "
        "per-node policy from a hard-coded one"
    )


def test_a_full_search_on_a_stochastic_environment_never_crosses_its_bounds():
    """``l(b) <= u(b)`` at every node, which branch-and-bound depends on.

    Purpose: The end-to-end regression for the clairvoyant-default-policy bug.
    A crossing means the interval no longer brackets the value, so pruning a
    branch on it is unsound.

    Given: ``CoinEnv``, 16 scenarios, depth 4, 60 trials.
    When: A full search runs.
    Then: Every belief and action node satisfies ``l <= u``, at least 20 nodes
        are checked so the sweep is not vacuous, and the planner reports zero
        bound clamps.

    Test type: unit
    """
    env = CoinEnv(discount_factor=DISCOUNT)
    np.random.seed(5)
    random.seed(5)
    planner = _chain_planner(env, depth=4, n_scenarios=16, n_simulations=60)
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))

    checked = 0
    for node_id in range(len(tree)):
        lower = tree.lower_confidence_bound[node_id]
        upper = tree.upper_confidence_bound[node_id]
        assert lower <= upper + TOL, (
            f"node {node_id} (kind={'ACTION' if tree.kind[node_id] == ACTION else 'BELIEF'}, "
            f"parent={tree.parent_id[node_id]}) has l={lower} > u={upper}"
        )
        checked += 1
    assert checked >= 20, f"only {checked} nodes built; the sweep proves little"
    assert planner._bound_clamps == 0, (
        f"{planner._bound_clamps} real bound crossings were repaired; the bounds "
        f"disagree by more than float noise"
    )
    assert root_id == 0


# ---------------------------------------------------------------------------
# Equation 4: backup
# ---------------------------------------------------------------------------


def _two_action_tree(
    action_bounds: List[Tuple[float, float]],
    root_lower: float = -100.0,
) -> Tuple[Tree, int, List[int]]:
    """A root with one child belief per action, whose bounds are set by hand.

    The child bounds are chosen by the caller, so the correct backup and a
    plausible wrong one give different answers. ``rho`` is zero on every action
    and the discount is 1 inside this helper's arithmetic only in the sense that
    the caller does the multiplication; the planner's own discount still
    applies, so expectations are written as ``gamma * bound``.
    """
    tree = Tree()
    root_id = tree.add_belief_node(UnweightedParticleBeliefStateUpdate(particles=[ROOT]))
    tree.lower_confidence_bound[root_id] = root_lower
    tree.upper_confidence_bound[root_id] = 100.0
    tree.data[root_id] = _BeliefNodeData(
        depth=0, default_value=root_lower, expanded=True, scenario_ids=[0]
    )

    action_ids: List[int] = []
    for index, (lower, upper) in enumerate(action_bounds):
        action_id = tree.add_action_node(action=f"a{index}", parent_id=root_id)
        tree.set_immediate_reward(action_id, 0.0)
        child_id = tree.add_belief_node(
            UnweightedParticleBeliefStateUpdate(particles=[NEXT]),
            observation=f"o{index}",
            weight=tree.weight[root_id],
            parent_id=action_id,
        )
        tree.lower_confidence_bound[child_id] = lower
        tree.upper_confidence_bound[child_id] = upper
        tree.data[child_id] = _BeliefNodeData(
            depth=1, default_value=lower, expanded=False, scenario_ids=[0]
        )
        action_ids.append(action_id)
    return tree, root_id, action_ids


def test_backup_raises_the_lower_bound_using_the_upper_bounds_maximiser():
    """``l(b) <- max(l(b), Q_l(b, argmax_a Q_u))``, not ``max_a Q_l``.

    Purpose: The two rules coincide whenever one action dominates on both
    bounds, so a test needs an action set where they disagree -- otherwise it
    passes for a planner that took the wrong maximum.

    Given: A root with two actions and zero first-step reward. Action 0's child
        has interval ``[2, 20]``; action 1's has ``[8, 10]``. So
        ``argmax_a Q_u`` is action 0 (upper ``0.5 * 20 = 10``) while
        ``argmax_a Q_l`` is action 1 (lower ``0.5 * 8 = 4``).
    When: The root is backed up.
    Then: ``l(root) = 0.5 * 2 = 1`` -- action 0's lower bound -- and
        ``u(root) = 10``. Taking ``max_a Q_l`` would give ``l(root) = 4``.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, n_simulations=0)
    tree, root_id, action_ids = _two_action_tree([(2.0, 20.0), (8.0, 10.0)])
    planner._root_id = root_id

    planner._backup(tree=tree, belief_id=root_id)

    assert tree.lower_confidence_bound[root_id] == pytest.approx(DISCOUNT * 2.0, abs=TOL), (
        f"l(root)={tree.lower_confidence_bound[root_id]}; equation (4) takes the lower "
        f"bound of the *upper*-bound maximiser (action 0, {DISCOUNT * 2.0}), not the "
        f"best lower bound ({DISCOUNT * 8.0})"
    )
    assert tree.upper_confidence_bound[root_id] == pytest.approx(DISCOUNT * 20.0, abs=TOL)
    assert tree.data[root_id].best_upper_action_id == action_ids[0]


def test_backup_never_lowers_a_lower_bound_it_has_already_earned():
    """``l(b)`` is a max against its own previous value.

    Purpose: The lower bound records a policy that was shown to be achievable;
    a later trial exploring elsewhere must not retract it. A plain assignment
    would, and the search would then never converge from below.

    Given: The same two-action root, but with ``l(root)`` already at ``7`` --
        higher than anything the children can currently justify.
    When: The root is backed up.
    Then: ``l(root)`` stays ``7``.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, n_simulations=0)
    tree, root_id, _ = _two_action_tree([(2.0, 20.0), (8.0, 10.0)], root_lower=7.0)
    planner._root_id = root_id

    planner._backup(tree=tree, belief_id=root_id)

    assert tree.lower_confidence_bound[root_id] == pytest.approx(7.0, abs=TOL), (
        f"l(root) fell to {tree.lower_confidence_bound[root_id]} from 7.0; the lower "
        f"bound must only rise"
    )


def test_upper_bound_is_recomputed_over_every_action_not_just_the_incumbent():
    """``u(b) <- max_a Q_u(b,a)``, recomputed from scratch each backup.

    Purpose: The incumbent's upper bound falls as its subtree is explored, and
    can drop below another action's. Updating only the incumbent leaves ``u(b)``
    too *low*, which is the dangerous direction: it can prune the optimum.

    Given: A root already backed up with action 0 as the upper maximiser
        (child ``[2, 20]``). Action 0's child is then tightened to ``[2, 3]``,
        as exploring it would do.
    When: The root is backed up again.
    Then: ``u(root)`` becomes ``0.5 * 10 = 5`` from action 1, and the recorded
        maximiser switches to action 1. Keeping the incumbent would give ``1.5``.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, n_simulations=0)
    tree, root_id, action_ids = _two_action_tree([(2.0, 20.0), (8.0, 10.0)])
    planner._root_id = root_id
    planner._backup(tree=tree, belief_id=root_id)
    assert tree.data[root_id].best_upper_action_id == action_ids[0]

    tightened_child = tree.get_children_ids(action_ids[0])[0]
    tree.upper_confidence_bound[tightened_child] = 3.0
    planner._backup(tree=tree, belief_id=root_id)

    assert tree.upper_confidence_bound[root_id] == pytest.approx(DISCOUNT * 10.0, abs=TOL), (
        f"u(root)={tree.upper_confidence_bound[root_id]}; after action 0 was tightened "
        f"the maximum moves to action 1 at {DISCOUNT * 10.0}"
    )
    assert tree.data[root_id].best_upper_action_id == action_ids[1]


def test_action_bounds_normalize_by_the_parent_weight_not_by_a_child_sum():
    """``L(b,a) = sum_o (w_o / w_b) l(b_o)`` uses the parent's weight.

    Purpose: The two normalizers are equal only when every scenario lands in
    some child. Renormalizing over the children that happen to exist is how a
    dropped scenario silently inflates a node's value, so the test uses a node
    whose weight exceeds its children's stated total to make the two differ.

    Given: A belief node of weight ``1.0`` with one action whose single child
        has weight ``0.25`` and bounds ``[8, 12]``, and zero first-step reward.
    When: The action's bounds are computed.
    Then: ``Q_l = 0.5 * (0.25 * 8) / 1.0 = 1.0``. Normalizing by the child sum
        would give ``0.5 * 8 = 4.0``.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, n_simulations=0)

    tree = Tree()
    root_id = tree.add_belief_node(UnweightedParticleBeliefStateUpdate(particles=[ROOT]))
    tree.data[root_id] = _BeliefNodeData(depth=0, default_value=0.0, scenario_ids=[0])
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    tree.set_immediate_reward(action_id, 0.0)
    child_id = tree.add_belief_node(
        UnweightedParticleBeliefStateUpdate(particles=[NEXT]),
        observation="o",
        weight=0.25,
        parent_id=action_id,
    )
    tree.lower_confidence_bound[child_id] = 8.0
    tree.upper_confidence_bound[child_id] = 12.0
    tree.data[child_id] = _BeliefNodeData(depth=1, default_value=8.0, scenario_ids=[0])

    planner._update_action_bounds(tree=tree, action_id=action_id)

    assert tree.lower_confidence_bound[action_id] == pytest.approx(
        DISCOUNT * 0.25 * 8.0, abs=TOL
    ), (
        f"Q_l={tree.lower_confidence_bound[action_id]}; normalizing by the child weight "
        f"sum instead of the parent's would give {DISCOUNT * 8.0}"
    )
    assert tree.upper_confidence_bound[action_id] == pytest.approx(DISCOUNT * 0.25 * 12.0, abs=TOL)


# ---------------------------------------------------------------------------
# Equation 5: which branch a trial follows
# ---------------------------------------------------------------------------


def test_weighted_excess_uncertainty_picks_the_branch_by_weight_times_slack():
    """``WEU(b_o) = (w_o / w_b) [(u - l) - eta (u_root - l_root) gamma^-d]``.

    Purpose: Three plausible rules -- widest interval, heaviest branch, and the
    weighted excess -- are separated here, so a planner that descends on the raw
    gap or on weight alone fails.

    Given: A root with gap ``10`` and ``eta = 0.5``, one action with two
        children at depth 1: child A with weight ``0.2`` and interval width
        ``20``, child B with weight ``0.8`` and width ``12``. The target slack
        at depth 1 is ``0.5 * 10 * 0.5^-1 = 10``.
    When: The descent branch is chosen.
    Then: A wins. Its weighted excess is ``0.2 * (20 - 10) = 2.0`` against B's
        ``0.8 * (12 - 10) = 1.6`` -- even though B carries four times the
        weight, and even though a rule that ignored the target and compared
        ``w * width`` would pick B (``9.6`` against ``4.0``).

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, eta=0.5, n_simulations=0)

    tree = Tree()
    root_id = tree.add_belief_node(UnweightedParticleBeliefStateUpdate(particles=[ROOT]))
    tree.lower_confidence_bound[root_id] = 0.0
    tree.upper_confidence_bound[root_id] = 10.0
    tree.data[root_id] = _BeliefNodeData(depth=0, default_value=0.0, scenario_ids=[0])
    planner._root_id = root_id

    action_id = tree.add_action_node(action="a", parent_id=root_id)
    tree.set_immediate_reward(action_id, 0.0)

    specs = [("A", 0.2, 0.0, 20.0), ("B", 0.8, 0.0, 12.0)]
    child_ids = {}
    for label, weight, lower, upper in specs:
        child_id = tree.add_belief_node(
            UnweightedParticleBeliefStateUpdate(particles=[NEXT]),
            observation=label,
            weight=weight,
            parent_id=action_id,
        )
        tree.lower_confidence_bound[child_id] = lower
        tree.upper_confidence_bound[child_id] = upper
        tree.data[child_id] = _BeliefNodeData(depth=1, default_value=lower, scenario_ids=[0])
        child_ids[label] = child_id

    target = 0.5 * 10.0 * DISCOUNT ** (-1)
    expected_a = 0.2 * (20.0 - target)
    expected_b = 0.8 * (12.0 - target)
    assert expected_a > expected_b, "fixture must make the correct answer the non-obvious one"

    chosen_id, chosen_value = planner._best_excess_uncertainty_child(tree=tree, action_id=action_id)

    assert chosen_id == child_ids["A"], (
        f"descended into {tree.get_observation(chosen_id)!r}; weighted excess is "
        f"A={expected_a}, B={expected_b}, so A wins even though B carries four times "
        f"the weight and the raw gaps are 20 and 12"
    )
    assert chosen_value == pytest.approx(expected_a, abs=TOL)


def test_a_branch_inside_the_target_precision_stops_the_trial():
    """``WEU <= 0`` means the branch is already as resolved as the root needs.

    Purpose: Without this the trial descends to the horizon on every pass and
    the "anytime" part of the algorithm does nothing.

    Given: The same root gap of ``10`` with ``eta = 0.5``, and a single child at
        depth 1 whose interval width is ``1`` -- far inside the depth-1 target
        of ``10``.
    When: The descent branch is chosen and a trial is run from the root.
    Then: The weighted excess is negative, and the trial expands the root but
        does not go past it.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, eta=0.5, n_simulations=0)

    tree = Tree()
    root_id = tree.add_belief_node(UnweightedParticleBeliefStateUpdate(particles=[ROOT]))
    tree.lower_confidence_bound[root_id] = 0.0
    tree.upper_confidence_bound[root_id] = 10.0
    tree.data[root_id] = _BeliefNodeData(
        depth=0, default_value=0.0, expanded=True, scenario_ids=[0]
    )
    planner._root_id = root_id
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    tree.set_immediate_reward(action_id, 0.0)
    child_id = tree.add_belief_node(
        UnweightedParticleBeliefStateUpdate(particles=[NEXT]),
        observation="o",
        weight=tree.weight[root_id],
        parent_id=action_id,
    )
    tree.lower_confidence_bound[child_id] = 0.0
    tree.upper_confidence_bound[child_id] = 1.0
    tree.data[child_id] = _BeliefNodeData(depth=1, default_value=0.0, scenario_ids=[0])

    _, value = planner._best_excess_uncertainty_child(tree=tree, action_id=action_id)
    assert value < 0.0, f"weighted excess {value} should be negative for a resolved branch"

    tree.data[root_id].best_upper_action_id = action_id
    planner._simulate_path(tree=tree, belief_id=root_id, depth=0)

    assert (
        tree.data[child_id].in_tree is False
    ), "the trial descended into a branch whose excess uncertainty was already negative"
    assert tree.data[root_id].in_tree is True


# ---------------------------------------------------------------------------
# Equation 7: search selection versus final selection
# ---------------------------------------------------------------------------


def test_the_returned_action_is_the_lower_bound_maximiser_not_the_optimistic_one():
    """Search descends on ``Q_u``; the answer is taken on ``Q_l``.

    Purpose: These are different rules and DESPOT uses both. Returning the
    optimistic action would hand back whichever branch the search understood
    least -- precisely the one whose upper bound has not yet been refuted.

    Given: The two-action root where action 0 wins on the upper bound and
        action 1 wins on the lower bound, backed up once.
    When: The final action is selected.
    Then: Action 1 is returned, while the recorded search maximiser is action 0.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, n_simulations=0)
    tree, root_id, action_ids = _two_action_tree([(2.0, 20.0), (8.0, 10.0)])
    planner._root_id = root_id
    planner._backup(tree=tree, belief_id=root_id)

    chosen = planner._select_final_action(tree=tree, root_id=root_id)

    assert chosen == tree.get_action(action_ids[1]), (
        f"returned {chosen!r}; the lower-bound maximiser is "
        f"{tree.get_action(action_ids[1])!r} while the search maximiser is "
        f"{tree.get_action(action_ids[0])!r}"
    )
    assert tree.data[root_id].best_upper_action_id == action_ids[0]


def test_the_returned_action_ignores_visit_counts():
    """Visits are a diagnostic here, not a decision rule.

    Purpose: Every MCTS planner in this repository can be read as "return the
    most-visited or best-Q action". DESPOT's answer is neither of those by
    construction, so a fixture where the most-visited action is the wrong one
    keeps a future refactor from quietly substituting the familiar rule.

    Given: The two-action root, with action 0 (the lower-bound loser) given 50
        visits and action 1 given 1.
    When: The final action is selected.
    Then: Action 1 is still returned.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, n_simulations=0)
    tree, root_id, action_ids = _two_action_tree([(2.0, 20.0), (8.0, 10.0)])
    planner._root_id = root_id
    planner._backup(tree=tree, belief_id=root_id)
    tree.visit_count[action_ids[0]] = 50
    tree.visit_count[action_ids[1]] = 1

    assert planner._select_final_action(tree=tree, root_id=root_id) == tree.get_action(
        action_ids[1]
    )


def test_chain_search_converges_to_the_hand_computed_chain_return():
    """End to end: the root's interval closes on ``2 + 0.5 * 4 = 4``.

    Purpose: A whole search on a fixture whose optimal value is known by hand,
    checking the composition of every equation rather than each in isolation.

    Given: ``ChainEnv`` from ``root``, discount ``0.5``, search depth 3, 4
        scenarios, 20 trials. Both actions have identical effects, so the
        optimal 3-step return is ``2 + 0.5 * 4 + 0.25 * 0 = 4``.
    When: The search runs to convergence.
    Then: ``l(root) = u(root) = 4``, the loop stopped on the gap rather than the
        budget, and every root action's ``Q_l`` is ``4``.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, n_scenarios=4, n_simulations=20)
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))

    expected = CHAIN_REWARDS[ROOT] + DISCOUNT * CHAIN_REWARDS[NEXT]
    assert tree.lower_confidence_bound[root_id] == pytest.approx(expected, abs=TOL)
    assert tree.upper_confidence_bound[root_id] == pytest.approx(expected, abs=TOL), (
        f"the interval did not close: [{tree.lower_confidence_bound[root_id]}, "
        f"{tree.upper_confidence_bound[root_id]}], expected both at {expected}"
    )
    assert planner._gap_closed is True, "the loop should end on the gap, not on the budget"
    assert planner._n_trials < 20, "convergence should not need the whole budget here"
    for action_id in tree.get_children_ids(root_id):
        assert tree.q_value[action_id] == pytest.approx(expected, abs=TOL)


# ---------------------------------------------------------------------------
# Equation 8: the regularized utility
# ---------------------------------------------------------------------------


def _regularization_tree() -> Tuple[Tree, int, int]:
    """Root, one action with first-step reward 6, one child of default value 4.

    All weights are 1 and the depth is 0 at the root, so the ``gamma^d w``
    scaling is the identity at the root and ``gamma`` at the child. That keeps
    the hand arithmetic below to two terms.
    """
    tree = Tree()
    root_id = tree.add_belief_node(UnweightedParticleBeliefStateUpdate(particles=[ROOT]))
    tree.data[root_id] = _BeliefNodeData(
        depth=0, default_value=1.0, expanded=True, scenario_ids=[0]
    )
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    tree.set_immediate_reward(action_id, 6.0)
    child_id = tree.add_belief_node(
        UnweightedParticleBeliefStateUpdate(particles=[NEXT]),
        observation="o",
        weight=1.0,
        parent_id=action_id,
    )
    tree.data[child_id] = _BeliefNodeData(depth=1, default_value=4.0, scenario_ids=[0])
    return tree, root_id, action_id


def test_regularized_utility_charges_one_lambda_per_policy_node():
    """``nu(b) = max{ g^d w l_0 - lam, max_a [g^d w rho - lam + sum_o nu(b_o)] }``.

    Purpose: Pins the arithmetic of equation (8), including that the child's
    contribution enters already scaled by ``gamma^depth`` so it can be added
    without renormalizing, and that a leaf pays its own ``lambda``.

    Given: A root of weight 1 and default value 1, one action with first-step
        reward 6, and one child at depth 1 of weight 1 and default value 4,
        with ``lambda = 0.5`` and ``gamma = 0.5``.
    When: The regularized value is computed.
    Then: The child is a leaf, so ``nu(child) = 0.5 * 4 - 0.5 = 1.5``. The
        root's action option is ``6 - 0.5 + 1.5 = 7`` and its default option is
        ``1 - 0.5 = 0.5``, so ``nu(root) = 7`` and the action is chosen.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, pruning_constant=0.5, n_simulations=0)
    tree, root_id, action_id = _regularization_tree()

    value, best_action_id = planner._regularized_value(tree=tree, belief_id=root_id)

    expected_child = DISCOUNT * 4.0 - 0.5
    expected_root = 6.0 - 0.5 + expected_child
    assert value == pytest.approx(expected_root, abs=TOL), (
        f"nu(root)={value}, expected {expected_root} = rho(6) - lambda(0.5) + "
        f"nu(child)({expected_child})"
    )
    assert best_action_id == action_id


def test_a_large_regularization_constant_prefers_the_default_policy():
    """The tree-size penalty can beat every action subtree, and that is an answer.

    Purpose: This is the mechanism the NIPS 2013 paper adds -- a big enough
    ``lambda`` says the sampled tree is not worth trusting. A planner that could
    never reach that branch has not implemented the regularization.

    Given: The same tree with ``lambda = 100``. The action option becomes
        ``6 - 100 + (2 - 100) = -192``, the default option ``1 - 100 = -99``.
    When: The regularized value is computed and an action is selected.
    Then: ``nu(root) = -99``, no action is returned, and the planner records
        that it fell back.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, pruning_constant=100.0, n_simulations=0)
    tree, root_id, _ = _regularization_tree()
    tree.lower_confidence_bound[tree.get_children_ids(root_id)[0]] = 3.0

    value, best_action_id = planner._regularized_value(tree=tree, belief_id=root_id)
    assert value == pytest.approx(1.0 - 100.0, abs=TOL)
    assert best_action_id is None

    chosen = planner._select_final_action(tree=tree, root_id=root_id)
    assert planner._regularized_fell_back is True, (
        "the substitution of the lower-bound action for the paper's default-policy "
        "action must be recorded, not silent"
    )
    assert chosen == "a", "the fallback is the lower-bound action"


def test_zero_regularization_uses_the_plain_lower_bound_rule():
    """``lambda = 0`` must reproduce the unregularized reference exactly.

    Purpose: The default configuration is the one every comparison will run, so
    it needs to be the reference algorithm and not a regularized variant with
    the penalty set small.

    Given: The two-action root where the lower- and upper-bound maximisers
        differ, with ``pruning_constant=0``.
    When: The final action is selected.
    Then: The lower-bound maximiser is returned and no regularized pass ran.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, pruning_constant=0.0, n_simulations=0)
    tree, root_id, action_ids = _two_action_tree([(2.0, 20.0), (8.0, 10.0)])
    planner._root_id = root_id
    planner._backup(tree=tree, belief_id=root_id)

    assert planner._select_final_action(tree=tree, root_id=root_id) == tree.get_action(
        action_ids[1]
    )
    assert planner._regularized_fell_back is False


# ---------------------------------------------------------------------------
# Trial bookkeeping and isolation
# ---------------------------------------------------------------------------


def test_a_trial_marks_only_the_nodes_it_walked_as_being_in_the_tree():
    """``|D|`` counts nodes a trial entered, not nodes that merely exist.

    Purpose: Expansion creates a fringe of belief nodes that no trial has
    evaluated. Counting them as tree members would misreport the sparse tree
    size the paper's bound is stated over.

    Given: ``ChainEnv``, depth 3, one trial from a fresh root.
    When: The trial runs.
    Then: Some belief nodes exist that are not ``in_tree``, and every
        ``in_tree`` node is expanded.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, n_scenarios=4, n_simulations=1)
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))

    belief_nodes = [nid for nid in range(len(tree)) if tree.kind[nid] == BELIEF]
    in_tree = [nid for nid in belief_nodes if tree.data[nid].in_tree]
    fringe = [nid for nid in belief_nodes if not tree.data[nid].in_tree]

    assert root_id in in_tree
    assert fringe, "expansion must leave a fringe, otherwise this check is vacuous"
    for node_id in in_tree:
        assert tree.data[
            node_id
        ].expanded, f"belief node {node_id} is marked in_tree but was never expanded"


def test_a_trial_down_one_action_leaves_the_other_actions_subtree_untouched():
    """Backups are local to the path the trial walked.

    Purpose: A backup that reached across branches would apply one branch's
    evidence to another's estimate, and the error would be invisible in any
    aggregate check.

    Given: A ``CoinEnv`` root expanded once so both actions have children, with
        the search maximiser pinned to action 0.
    When: One trial runs.
    Then: Action 1's own bounds and every field of its subtree are unchanged.

    Test type: unit
    """
    env = CoinEnv(discount_factor=DISCOUNT)
    np.random.seed(2)
    random.seed(2)
    planner = _chain_planner(env, depth=3, n_scenarios=8, n_simulations=0)
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))
    planner._expand(tree=tree, belief_id=root_id)

    action_children = tree.get_children_ids(root_id)
    tree.data[root_id].best_upper_action_id = action_children[0]
    untouched_id = action_children[1]

    def snapshot(node_id: int) -> List[Tuple[Any, ...]]:
        records = []
        pending = [node_id]
        while pending:
            current = pending.pop()
            records.append(
                (
                    current,
                    tree.visit_count[current],
                    tree.weight[current],
                    tree.lower_confidence_bound[current],
                    tree.upper_confidence_bound[current],
                    tree.q_value[current],
                    tuple(tree.children_ids[current]),
                )
            )
            pending.extend(tree.children_ids[current])
        return sorted(records)

    before = snapshot(untouched_id)
    planner._simulate_path(tree=tree, belief_id=root_id, depth=0)
    after = snapshot(untouched_id)

    assert before == after, (
        f"the trial changed the sibling subtree under action "
        f"{tree.get_action(untouched_id)!r}:\n  before {before}\n  after  {after}"
    )


def test_root_visits_equal_the_number_of_trials_that_entered_it():
    """Visit accounting: one visit per trial that got past the root's cutoffs.

    Purpose: Visit counts are the only quantity the shared arena metrics read,
    so their meaning for this planner has to be stated and checked rather than
    inherited from POMCP, where a visit means a completed simulation.

    Given: ``CoinEnv`` -- never terminal, so no trial is cut short at the root
        -- depth 3, 8 scenarios, 12 trials.
    When: The search runs.
    Then: The root's visit count equals the number of trials, and every action
        node's visits sum to no more than its parent belief's.

    Test type: unit
    """
    env = CoinEnv(discount_factor=DISCOUNT)
    np.random.seed(4)
    random.seed(4)
    planner = _chain_planner(env, depth=3, n_scenarios=8, n_simulations=12)
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))

    assert tree.visit_count[root_id] == planner._n_trials

    checked = 0
    for node_id in range(len(tree)):
        if tree.kind[node_id] != BELIEF:
            continue
        child_visits = sum(tree.visit_count[cid] for cid in tree.children_ids[node_id])
        assert child_visits <= tree.visit_count[node_id], (
            f"belief node {node_id} has {tree.visit_count[node_id]} visits but its action "
            f"children total {child_visits}"
        )
        checked += 1
    assert checked >= 3


def test_alternating_kinds_and_valid_links_across_the_whole_arena():
    """Structural sweep: no cycles, no orphans, kinds alternate.

    Purpose: The generic arena contract, checked on this planner's own
    construction order, so a mis-parented node shows up here rather than as a
    puzzling number in QA.

    Given: ``CoinEnv``, depth 3, 8 scenarios, 15 trials.
    When: The search runs.
    Then: Every logical node is reachable exactly once from the root, parents
        and children agree, kinds alternate, belief nodes carry a belief and a
        ``_BeliefNodeData``, and action nodes carry a legal action.

    Test type: unit
    """
    env = CoinEnv(discount_factor=DISCOUNT)
    np.random.seed(6)
    random.seed(6)
    planner = _chain_planner(env, depth=3, n_scenarios=8, n_simulations=15)
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))

    assert tree.kind[root_id] == BELIEF
    assert tree.parent_id[root_id] is None

    legal_actions = env.get_actions()
    depths: Dict[int, int] = {}
    pending = [(root_id, 0)]
    while pending:
        node_id, depth = pending.pop()
        assert node_id not in depths, f"node {node_id} reached twice: cycle or duplicate edge"
        depths[node_id] = depth
        kind = tree.kind[node_id]
        assert kind in (BELIEF, ACTION)
        if kind == BELIEF:
            assert isinstance(tree.data[node_id], _BeliefNodeData)
            assert tree.get_belief(node_id) is not None
            assert len(tree.sample[node_id]) == len(tree.get_belief(node_id).particles)
            assert tree.weight[node_id] == pytest.approx(
                len(tree.sample[node_id]) / planner.n_scenarios, abs=TOL
            )
        else:
            assert tree.get_action(node_id) in legal_actions
        children = tree.children_ids[node_id]
        assert len(children) == len(set(children))
        for child_id in children:
            assert tree.parent_id[child_id] == node_id
            assert tree.kind[child_id] == (ACTION if kind == BELIEF else BELIEF)
            pending.append((child_id, depth + 1))

    assert set(depths) == set(range(len(tree))), "unreachable logical nodes in the arena"
    assert max(depths.values()) <= 2 * planner.depth, (
        f"arena depth {max(depths.values())} exceeds 2 * search depth "
        f"{2 * planner.depth}: one planning step is one belief edge plus one action edge"
    )


def test_an_unhashable_observation_key_is_refused_with_a_pointed_message():
    """Observation grouping is what makes the tree sparse; it needs a key.

    Purpose: Silently failing to merge would leave one branch per scenario --
    a tree that still runs, still returns an action, and is no longer DESPOT.

    Given: A ``ChainEnv`` subclass whose ``hash_observation`` returns a list.
    When: The root is expanded.
    Then: A ``TypeError`` naming ``hash_observation`` is raised.

    Test type: unit
    """
    env = UnhashableObservationEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=2, n_simulations=0)
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))

    with pytest.raises(TypeError, match="hash_observation"):
        planner._expand(tree=tree, belief_id=root_id)


def test_scenario_states_are_drawn_by_systematic_resampling_of_the_belief():
    """Every particle whose weight exceeds ``1/K`` is represented at least once.

    Purpose: Independent draws can miss a well-supported state entirely, and a
    decision blind to a plausible state is worse than a noisy one. Systematic
    resampling is what the reference uses, and it is checkable exactly.

    Given: A belief with two particles at weights ``0.75`` and ``0.25``, and
        ``K = 8``.
    When: Scenario states are drawn.
    Then: There are exactly 8 of them, split 6 and 2 -- the deterministic
        outcome of systematic resampling at those weights.

    Test type: unit
    """
    from POMDPPlanners.tests.test_planners.planner_fixtures import (  # local: fixture-only
        two_state_belief,
    )

    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=2, n_scenarios=8, n_simulations=0)
    states = planner._sample_scenario_states(belief=two_state_belief(ROOT, NEXT, 0.75))

    assert len(states) == 8
    assert states.count(ROOT) == 6, f"expected 6 of 8 at weight 0.75, got {states.count(ROOT)}"
    assert states.count(NEXT) == 2


def test_the_systematic_offset_moves_between_decisions():
    """Consecutive decisions do not reuse one frozen scenario set.

    Purpose: :meth:`action` restores the caller's generators on the way out, so
    drawing the scenarios from them would make every decision on an unchanged
    belief see the identical ``K`` states -- the planner would stop being a
    randomized approximation. The private generator exists for exactly this, and
    a test with ``K`` not dividing the weights shows the offset actually moving.

    Given: A belief split ``0.5 / 0.5`` and ``K = 3``, where systematic
        resampling gives ``[root, root, next]`` or ``[root, next, next]``
        depending on the offset.
    When: Scenario states are drawn twenty times.
    Then: Both compositions appear.

    (With ``K`` a multiple of the weights the composition is fixed by design --
    that is what systematic resampling buys -- so this fixture avoids that case
    on purpose.)

    Test type: unit
    """
    from POMDPPlanners.tests.test_planners.planner_fixtures import (  # local: fixture-only
        two_state_belief,
    )

    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=2, n_scenarios=3, n_simulations=0)
    belief = two_state_belief(ROOT, NEXT, 0.5)

    compositions = {tuple(planner._sample_scenario_states(belief=belief)) for _ in range(20)}
    assert compositions == {(ROOT, ROOT, NEXT), (ROOT, NEXT, NEXT)}, (
        f"got {compositions}; both systematic-resampling outcomes should appear "
        f"across twenty draws"
    )


def test_each_decision_gets_a_fresh_scenario_stream_table():
    """The determinized random numbers change from one decision to the next.

    Purpose: Fixing ``K`` scenarios *within* a decision is the algorithm; fixing
    them *across* decisions would make the planner replay the same imagined
    futures every step, so an unlucky table would poison a whole episode.

    Given: A planner asked to plan twice from the same belief.
    When: Both decisions run.
    Then: The two decisions' seed tables differ.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=2, n_scenarios=4, n_simulations=1)

    planner.action(chain_belief(ROOT))
    first = [planner._streams.seed_for(scenario_id=i, depth=0) for i in range(4)]
    planner.action(chain_belief(ROOT))
    second = [planner._streams.seed_for(scenario_id=i, depth=0) for i in range(4)]

    assert first != second, "both decisions used the identical scenario stream table"


def test_bounds_bracket_the_achievable_return_at_every_node():
    """Every interval lies inside the horizon's independently derived range.

    Purpose: A value can be arithmetically consistent with its children and
    still be impossible for the environment. This checks the bounds against
    ``ChainEnv``'s declared reward range and the remaining horizon, derived
    here rather than from anything the tree reports.

    Given: ``ChainEnv`` (rewards in ``[0, 4]``), discount ``0.5``, depth 3, 4
        scenarios, 20 trials.
    When: The search runs.
    Then: Every belief node's interval sits inside
        ``[0, 4 * sum_(t < 3-d) 0.5^t]``, and at least 4 nodes are checked.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, n_scenarios=4, n_simulations=20)
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))
    del root_id

    checked = 0
    for node_id in range(len(tree)):
        if tree.kind[node_id] != BELIEF:
            continue
        remaining = planner.depth - tree.data[node_id].depth
        high = CHAIN_MAX_REWARD * _geometric(DISCOUNT, max(remaining, 0))
        for label, value in (
            ("lower", tree.lower_confidence_bound[node_id]),
            ("upper", tree.upper_confidence_bound[node_id]),
        ):
            assert -TOL <= value <= high + TOL, (
                f"belief node {node_id} (depth={tree.data[node_id].depth}, "
                f"remaining={remaining}) has {label} bound {value} outside [0, {high}]"
            )
        checked += 1
    assert checked >= 4, f"only {checked} belief nodes checked"


def test_search_and_final_selection_agree_when_one_action_dominates():
    """A sanity case: with a dominant action both rules pick it.

    Purpose: The tests above deliberately separate the two rules. This one
    confirms they are not accidentally *always* different -- a planner that
    inverted the final rule would pass the separating tests' negation but fail
    here.

    Given: A root whose action 0 dominates on both bounds (``[9, 20]`` against
        ``[1, 2]``).
    When: The root is backed up and the final action selected.
    Then: Action 0 is both the search maximiser and the returned action.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _chain_planner(env, depth=3, n_simulations=0)
    tree, root_id, action_ids = _two_action_tree([(9.0, 20.0), (1.0, 2.0)])
    planner._root_id = root_id
    planner._backup(tree=tree, belief_id=root_id)

    assert tree.data[root_id].best_upper_action_id == action_ids[0]
    assert planner._select_final_action(tree=tree, root_id=root_id) == tree.get_action(
        action_ids[0]
    )


def test_discount_one_keeps_the_undiscounted_sum_over_a_finite_horizon():
    """``gamma = 1`` must not divide by zero and must not drop future terms.

    Purpose: The geometric-sum shortcut ``(1 - g^n) / (1 - g)`` is undefined at
    ``g = 1``; the planner writes it as a finite sum for that reason, and the
    boundary deserves its own case.

    Given: ``ChainEnv`` with discount ``1.0``, depth 3, from ``root``.
    When: The root's bounds are computed.
    Then: ``l_0(root) = 2 + 4 = 6`` and ``u_0(root) = 4 * 3 = 12``.

    Test type: unit
    """
    env = ChainEnv(discount_factor=1.0)
    planner = _chain_planner(env, depth=3, n_simulations=0)
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))

    assert tree.lower_confidence_bound[root_id] == pytest.approx(
        CHAIN_REWARDS[ROOT] + CHAIN_REWARDS[NEXT], abs=TOL
    )
    assert tree.upper_confidence_bound[root_id] == pytest.approx(CHAIN_MAX_REWARD * 3, abs=TOL)
