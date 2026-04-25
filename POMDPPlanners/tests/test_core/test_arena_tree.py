"""Tests for the column-store Tree in ``core.tree.arena``.

The arena tree has no node objects: a node is an integer ID into the
``Tree``'s columns, and every operation is a method of ``Tree``. These
tests validate that contract.
"""

import random
from unittest.mock import Mock

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import (
    Environment,
    ObservationModel,
    SpaceInfo,
    SpaceType,
    StateTransitionModel,
)
from POMDPPlanners.core.tree.arena import ACTION, BELIEF, Tree

np.random.seed(42)
random.seed(42)


class _MockEnvironment(Environment):
    def __init__(self) -> None:
        space_info = SpaceInfo(SpaceType.DISCRETE, SpaceType.DISCRETE)
        super().__init__(discount_factor=0.95, name="MockEnvironment", space_info=space_info)

    def state_transition(self, state, action):
        del action
        return state

    def observation_model(self, next_state, action):
        del next_state, action
        mock_model = Mock(spec=ObservationModel)
        mock_model.probability.return_value = 1.0
        return mock_model

    def is_equal_observation(self, observation1, observation2):
        return observation1 == observation2

    def is_terminal(self, state):
        del state
        return False

    def reward(self, state, action):
        del state, action
        return 0.0

    def initial_state_dist(self):
        return DiscreteDistribution(values=[0], probs=np.array([1.0]))

    def initial_observation_dist(self):
        return DiscreteDistribution(values=["obs"], probs=np.array([1.0]))

    def state_transition_model(self, state, action):
        del state, action
        mock_model = Mock(spec=StateTransitionModel)
        mock_model.probability.return_value = 1.0
        return mock_model


@pytest.fixture(name="belief")
def fixture_belief():
    return WeightedParticleBelief(particles=[1, 2], log_weights=np.log(np.array([0.6, 0.4])))


@pytest.fixture(name="env")
def fixture_env():
    return _MockEnvironment()


def test_add_belief_node_creates_root_and_populates_columns(belief):
    """Allocating a root belief node initialises every column at the new ID.

    Purpose: Validates the schema declared in __init__ matches the schema
    populated by _allocate — every column has a value at index 0 after
    one allocation.

    Given: A fresh Tree.
    When: add_belief_node is called with a belief and no parent.
    Then: ID is 0, kind is BELIEF, parent is None, children empty,
    metrics defaulted, payload populated.

    Test type: unit
    """
    tree = Tree()
    nid = tree.add_belief_node(belief)
    assert nid == 0
    assert len(tree) == 1
    assert tree.kind[nid] == BELIEF
    assert tree.parent_id[nid] is None
    assert tree.children_ids[nid] == []
    assert tree.visit_count[nid] == 0
    assert tree.v_value[nid] == 0.0
    assert tree.lower_confidence_bound[nid] == 0.0
    assert tree.upper_confidence_bound[nid] == 0.0
    assert tree.immediate_cost[nid] is None
    assert tree.immediate_reward[nid] is None
    assert tree.weight[nid] == 1.0
    assert tree.belief[nid] is belief
    assert tree.observation[nid] is None
    assert tree.action[nid] is None
    assert tree.sample[nid] == []


def test_add_action_node_links_to_parent_and_populates_columns(belief):
    """Action node allocation appends to parent.children_ids.

    Purpose: Validates the parent linkage edge of _allocate.

    Given: A Tree with one belief root.
    When: add_action_node is called with parent_id=root.
    Then: kind is ACTION, parent_id linked, action populated, parent's
    children_ids contains the new ID.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    assert tree.kind[action] == ACTION
    assert tree.parent_id[action] == root
    assert tree.children_ids[root] == [action]
    assert tree.action[action] == "a"
    assert tree.q_value[action] == 0.0


def test_add_belief_node_rejects_non_belief():
    """Runtime type guard fires when belief is not a Belief instance.

    Purpose: Same defensive guard as the legacy BeliefNode.

    Given: A fresh Tree.
    When: add_belief_node is called with a non-Belief object.
    Then: TypeError.

    Test type: unit
    """
    tree = Tree()
    with pytest.raises(TypeError):
        tree.add_belief_node("not a belief")  # type: ignore[arg-type]


def test_depth_walks_parent_chain(belief):
    """depth(node_id) returns the number of edges to the root.

    Purpose: Validates the parent-chain walk used for depth-limited search.

    Given: A 3-deep chain root→action→belief.
    When: depth() is called on each node.
    Then: 0, 1, 2.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    leaf = tree.add_belief_node(belief, parent_id=action)
    assert tree.depth(root) == 0
    assert tree.depth(action) == 1
    assert tree.depth(leaf) == 2


def test_get_belief_child_finds_or_returns_none(belief, env):
    """get_belief_child returns the ID matching the observation, or None.

    Purpose: Same lookup contract as the legacy ActionNode.get_belief_node_child.

    Given: An action node with three belief children with distinct observations.
    When: get_belief_child is called for present and absent observations.
    Then: Correct ID for matches, None otherwise.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    b1 = tree.add_belief_node(belief, observation="o1", parent_id=action)
    b2 = tree.add_belief_node(belief, observation="o2", parent_id=action)
    assert tree.get_belief_child(action, "o1", env) == b1
    assert tree.get_belief_child(action, "o2", env) == b2
    assert tree.get_belief_child(action, "absent", env) is None


def test_get_belief_child_returns_first_for_duplicate_observations(belief, env):
    """When two belief children share an observation, get_belief_child returns the first.

    Purpose: Match first-match semantics of the legacy implementation.

    Given: An action node with two belief children sharing an observation.
    When: get_belief_child looks up that observation.
    Then: The earlier-allocated ID is returned.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    first = tree.add_belief_node(belief, observation="same", parent_id=action)
    tree.add_belief_node(belief, observation="same", parent_id=action)
    assert tree.get_belief_child(action, "same", env) == first


def test_get_action_child_finds_or_returns_none(belief):
    """get_action_child looks up an action child by hashable label.

    Purpose: Same lookup contract as the legacy BeliefNode.get_child for hashable labels.

    Given: A belief node with action children labelled by string and int.
    When: get_action_child is called.
    Then: Correct ID for matches, None for absent.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    a_str = tree.add_action_node("up", parent_id=root)
    a_int = tree.add_action_node(7, parent_id=root)
    assert tree.get_action_child(root, "up") == a_str
    assert tree.get_action_child(root, 7) == a_int
    assert tree.get_action_child(root, "absent") is None


def test_get_action_child_compares_numpy_arrays_by_content(belief):
    """Numpy-array action labels compared by content, not identity.

    Purpose: Validates the np.array_equal branch in get_action_child.

    Given: A belief node with two numpy-array action children.
    When: get_action_child is called with arrays equal-by-content.
    Then: Correct ID for matches.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    a1 = tree.add_action_node(np.array([1.0, 2.0]), parent_id=root)
    a2 = tree.add_action_node(np.array([3.0, 4.0]), parent_id=root)
    assert tree.get_action_child(root, np.array([1.0, 2.0])) == a1
    assert tree.get_action_child(root, np.array([3.0, 4.0])) == a2
    assert tree.get_action_child(root, np.array([9.0, 9.0])) is None


def test_sample_belief_child_returns_one_of_children(belief):
    """sample_belief_child returns a child ID weighted by .weight.

    Purpose: Validates the weighted-sampling primitive.

    Given: An action node with three belief children of equal weight.
    When: sample_belief_child is called many times.
    Then: Every result is a child ID; at least two distinct IDs are seen.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    children = [
        tree.add_belief_node(belief, observation=f"o{i}", parent_id=action) for i in range(3)
    ]
    seen = {tree.sample_belief_child(action) for _ in range(100)}
    assert seen.issubset(set(children))
    assert len(seen) >= 2


def test_sample_belief_child_no_children_raises(belief):
    """sample_belief_child on a childless action raises.

    Purpose: Same edge case as the legacy ActionNode.sample_child_node.

    Given: An action node with no children.
    When: sample_belief_child is called.
    Then: ValueError.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    with pytest.raises(ValueError):
        tree.sample_belief_child(action)


def test_update_belief_replaces_belief_at_id(belief):
    """update_belief stores belief.update(...)'s return at the same ID.

    Purpose: Validates the in-tree belief replacement.

    Given: A belief node whose belief.update returns a known sentinel.
    When: update_belief is called.
    Then: tree.belief[id] is the sentinel.

    Test type: unit
    """
    sentinel = WeightedParticleBelief(particles=[99, 100], log_weights=np.log(np.array([0.7, 0.3])))
    mock_belief = Mock(spec=WeightedParticleBelief)
    mock_belief.update.return_value = sentinel

    tree = Tree()
    nid = tree.add_belief_node(belief)
    tree.belief[nid] = mock_belief
    tree.update_belief(nid, action="a", observation="o", pomdp=Mock(spec=Environment))
    assert tree.belief[nid] is sentinel


def test_set_immediate_cost_mirrors_to_immediate_reward(belief):
    """set_immediate_cost(v) sets immediate_reward to -v unless v is None.

    Purpose: Validates the coupled-setter contract from the legacy node.

    Given: A node with default cost/reward (both None).
    When: set_immediate_cost is called with various values.
    Then: immediate_reward mirrors as -value; None leaves the partner alone.

    Test type: unit
    """
    tree = Tree()
    nid = tree.add_belief_node(belief)
    tree.set_immediate_cost(nid, 5.0)
    assert tree.immediate_cost[nid] == 5.0
    assert tree.immediate_reward[nid] == -5.0
    tree.set_immediate_cost(nid, -3.5)
    assert tree.immediate_cost[nid] == -3.5
    assert tree.immediate_reward[nid] == 3.5
    tree.immediate_reward[nid] = 99.0  # restore a sentinel
    tree.set_immediate_cost(nid, None)
    assert tree.immediate_cost[nid] is None
    assert tree.immediate_reward[nid] == 99.0


def test_increment_weight_updates_weight_and_parent_cdf(belief):
    """``Tree.increment_weight`` bumps a child's weight and patches the parent's CDF.

    Purpose: Validates the incremental CDF maintenance primitive used by POMCPOW
    on every observation re-visit (cheaper than full ``recompute_children_cdf``).

    Given: A belief root with two action children of default weight 1.0 each;
    parent CDF is [1.0, 2.0].
    When: ``increment_weight(c1, delta=2.0)`` is called on the second child.
    Then: weight[c1] becomes 3.0; only CDF entries from c1's position onward
    are bumped by 2.0 — CDF becomes [1.0, 4.0].

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    c0 = tree.add_action_node("a0", parent_id=root)
    c1 = tree.add_action_node("a1", parent_id=root)
    assert list(tree.children_cdf[root]) == [1.0, 2.0]

    tree.increment_weight(c1, delta=2.0)
    assert tree.weight[c1] == 3.0
    assert tree.weight[c0] == 1.0
    assert list(tree.children_cdf[root]) == [1.0, 4.0]


def test_increment_weight_root_only_updates_weight(belief):
    """``increment_weight`` on the root (no parent) updates only the weight.

    Purpose: Documents the no-op-on-CDF behaviour for the root node.

    Given: A root with no parent.
    When: ``increment_weight(root, delta=0.5)``.
    Then: weight[root] is bumped; no error raised; no CDF to patch.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    initial = tree.weight[root]
    tree.increment_weight(root, delta=0.5)
    assert tree.weight[root] == initial + 0.5


def test_increment_weight_negative_delta_decreases(belief):
    """Negative delta decreases the child's weight and patches CDF correctly.

    Purpose: Validates that the increment is a signed delta, not a positive bump.

    Given: A belief root with one action child of weight 1.0.
    When: ``increment_weight(c, delta=-0.4)``.
    Then: weight[c] == 0.6; CDF reflects 0.6.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    c = tree.add_action_node("a", parent_id=root)
    tree.increment_weight(c, delta=-0.4)
    assert tree.weight[c] == pytest.approx(0.6)
    assert tree.children_cdf[root][0] == pytest.approx(0.6)


def test_increment_weight_then_sample_distribution_correct(belief):
    """After ``increment_weight``, ``sample_belief_child`` honours the new weights.

    Purpose: Validates that the incremental CDF patch keeps the sampler in sync
    with the actual weights — no need to call ``recompute_children_cdf`` after.

    Given: An action node with two belief children of weight 1.0; we then bump
    the second child's weight to 4.0 (so light:heavy = 1:4 in expectation).
    When: 5000 samples are drawn from the parent.
    Then: The 4.0-weight child is sampled ~4× as often as the 1.0-weight child
    (within statistical noise).

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    light = tree.add_belief_node(belief, observation="light", weight=1.0, parent_id=action)
    heavy = tree.add_belief_node(belief, observation="heavy", weight=1.0, parent_id=action)
    tree.increment_weight(heavy, delta=3.0)
    counts = {light: 0, heavy: 0}
    for _ in range(5000):
        counts[tree.sample_belief_child(action)] += 1
    ratio = counts[heavy] / counts[light]
    assert 3.4 < ratio < 4.6  # expected 4.0; allow noise


def test_get_belief_child_indexed_finds_or_returns_none(belief):
    """O(1) indexed lookup of belief children by hashable observation.

    Purpose: Validates the new ``obs_child_lookup`` index path is populated
    on add and queried on get.

    Given: An action node with belief children whose observations are strings.
    When: ``get_belief_child_indexed`` is called for present and absent observations.
    Then: Correct ID for matches, None otherwise.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    b1 = tree.add_belief_node(belief, observation="o1", parent_id=action)
    b2 = tree.add_belief_node(belief, observation="o2", parent_id=action)
    assert tree.get_belief_child_indexed(action, "o1") == b1
    assert tree.get_belief_child_indexed(action, "o2") == b2
    assert tree.get_belief_child_indexed(action, "absent") is None


def test_get_belief_child_indexed_returns_none_for_unhashable_observation(belief):
    """Indexed lookup gracefully returns None for unhashable observations.

    Purpose: When ``observation`` is e.g. ``np.ndarray``, the ``(parent, obs)``
    key cannot be hashed; the indexed method must not raise.

    Given: An action node with one belief child whose observation is an ndarray.
    When: ``get_belief_child_indexed`` is called with an ndarray.
    Then: Returns None (the linear-scan ``get_belief_child`` should be used
    for ndarray observations).

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    tree.add_belief_node(belief, observation=np.array([1.0, 2.0]), parent_id=action)
    assert tree.get_belief_child_indexed(action, np.array([1.0, 2.0])) is None


def test_get_action_child_indexed_finds_or_returns_none(belief):
    """O(1) indexed lookup of action children by hashable action label.

    Purpose: Validates the new ``action_child_lookup`` index.

    Given: A belief node with action children labelled by string and int.
    When: ``get_action_child_indexed`` is called for present and absent labels.
    Then: Correct ID for matches, None otherwise.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    a_str = tree.add_action_node("up", parent_id=root)
    a_int = tree.add_action_node(7, parent_id=root)
    assert tree.get_action_child_indexed(root, "up") == a_str
    assert tree.get_action_child_indexed(root, 7) == a_int
    assert tree.get_action_child_indexed(root, "absent") is None


def test_get_action_child_indexed_returns_none_for_unhashable_action(belief):
    """Indexed lookup gracefully returns None for ndarray action labels.

    Purpose: Same edge case as observation indexing — ndarray actions can't
    be dict keys.

    Given: A belief node with an action child whose label is an ndarray.
    When: ``get_action_child_indexed`` is called with an ndarray.
    Then: Returns None (use linear-scan ``get_action_child`` instead).

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    tree.add_action_node(np.array([1.0, 2.0]), parent_id=root)
    assert tree.get_action_child_indexed(root, np.array([1.0, 2.0])) is None


def test_children_cdf_is_maintained_incrementally(belief):
    """Adding a child extends the parent's CDF by the new child's weight.

    Purpose: Validates the invariant that powers O(log K) sampling.

    Given: A fresh action node.
    When: Three belief children are added with weights 0.5, 1.5, 2.0.
    Then: The parent's CDF is [0.5, 2.0, 4.0].

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    tree.add_belief_node(belief, observation="o1", weight=0.5, parent_id=action)
    tree.add_belief_node(belief, observation="o2", weight=1.5, parent_id=action)
    tree.add_belief_node(belief, observation="o3", weight=2.0, parent_id=action)
    assert tree.children_cdf[action] == [0.5, 2.0, 4.0]


def test_sample_belief_child_distribution_matches_weights(belief):
    """``sample_belief_child`` samples in proportion to weights via the CDF.

    Purpose: Validates the CDF-based sampler honours the weight ratios.

    Given: An action node with two belief children of weight 1.0 and 3.0.
    When: 4000 samples are drawn.
    Then: The 3.0-weight child is sampled roughly 3× as often (within
    statistical noise).

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    light = tree.add_belief_node(belief, observation="light", weight=1.0, parent_id=action)
    heavy = tree.add_belief_node(belief, observation="heavy", weight=3.0, parent_id=action)
    counts = {light: 0, heavy: 0}
    for _ in range(4000):
        counts[tree.sample_belief_child(action)] += 1
    ratio = counts[heavy] / counts[light]
    assert 2.5 < ratio < 3.5  # expected 3.0; allow ±0.5 noise


def test_recompute_children_cdf_picks_up_post_add_weight_changes(belief):
    """``recompute_children_cdf`` rebuilds the CDF from current ``weight`` values.

    Purpose: Validates the escape hatch for callers that mutate weights after add.

    Given: An action with belief children whose weights are mutated post-add.
    When: ``recompute_children_cdf`` is called on the parent.
    Then: ``children_cdf[parent]`` reflects the new weights.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    c1 = tree.add_belief_node(belief, observation="o1", weight=1.0, parent_id=action)
    c2 = tree.add_belief_node(belief, observation="o2", weight=1.0, parent_id=action)
    tree.weight[c1] = 2.0
    tree.weight[c2] = 5.0
    tree.recompute_children_cdf(action)
    assert tree.children_cdf[action] == [2.0, 7.0]


def test_set_immediate_reward_mirrors_to_immediate_cost(belief):
    """set_immediate_reward(v) sets immediate_cost to -v unless v is None.

    Purpose: Symmetric counterpart of the cost setter contract.

    Given: A node with default cost/reward.
    When: set_immediate_reward is called with various values.
    Then: immediate_cost mirrors as -value; None leaves the partner alone.

    Test type: unit
    """
    tree = Tree()
    nid = tree.add_action_node("a", parent_id=tree.add_belief_node(belief))
    tree.set_immediate_reward(nid, 7.5)
    assert tree.immediate_reward[nid] == 7.5
    assert tree.immediate_cost[nid] == -7.5
    tree.set_immediate_reward(nid, -2.0)
    assert tree.immediate_reward[nid] == -2.0
    assert tree.immediate_cost[nid] == 2.0
    tree.immediate_cost[nid] = 99.0
    tree.set_immediate_reward(nid, None)
    assert tree.immediate_reward[nid] is None
    assert tree.immediate_cost[nid] == 99.0
