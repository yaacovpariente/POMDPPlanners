# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for the column-store Tree in ``core.tree.arena``.

The arena tree has no node objects: a node is an integer ID into the
``Tree``'s columns, and every operation is a method of ``Tree``. These
tests validate that contract.
"""

# pylint: disable=too-many-lines

import random
from unittest.mock import Mock

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import (
    Environment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.tree.arena import ACTION, BELIEF, Tree

np.random.seed(42)
random.seed(42)


class _MockEnvironment(Environment):
    def __init__(self) -> None:
        space_info = SpaceInfo(SpaceType.DISCRETE, SpaceType.DISCRETE)
        super().__init__(discount_factor=0.95, name="MockEnvironment", space_info=space_info)

    def sample_next_state(self, state, action, n_samples: int = 1):
        del action
        if n_samples == 1:
            return state
        return [state] * n_samples

    def sample_observation(self, next_state, action, n_samples: int = 1):
        del next_state, action
        if n_samples == 1:
            return "obs"
        return ["obs"] * n_samples

    def transition_log_probability(self, state, action, next_states) -> np.ndarray:
        del state, action
        return np.zeros(len(next_states))

    def observation_log_probability(self, next_state, action, observations) -> np.ndarray:
        del next_state, action
        return np.zeros(len(observations))

    def is_equal_observation(self, observation1, observation2):
        return observation1 == observation2

    def hash_action(self, action):
        return action

    def is_terminal(self, state):
        del state
        return False

    def reward(self, state, action, next_state=None):
        del state, action, next_state
        return 0.0

    def initial_state_dist(self):
        return DiscreteDistribution(values=[0], probs=np.array([1.0]))

    def initial_observation_dist(self):
        return DiscreteDistribution(values=["obs"], probs=np.array([1.0]))


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


def test_best_action_by_reward_returns_action_with_highest_q_value(belief):
    """best_action_by_reward picks the action label of the max-q child.

    Purpose: Validates the reward-setting argmax used to extract the
    final action recommendation from a search tree.

    Given: A belief root with three action children whose q_values are
    1.0, 5.0, 3.0.
    When: best_action_by_reward is called on the root.
    Then: The action label of the q=5.0 child is returned.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    a_low = tree.add_action_node("low", parent_id=root)
    a_high = tree.add_action_node("high", parent_id=root)
    a_mid = tree.add_action_node("mid", parent_id=root)
    tree.q_value[a_low] = 1.0
    tree.q_value[a_high] = 5.0
    tree.q_value[a_mid] = 3.0
    assert tree.best_action_by_reward(root) == "high"


def test_best_action_by_reward_raises_when_belief_has_no_action_children(belief):
    """best_action_by_reward on a childless belief raises ValueError.

    Purpose: Validates the empty-children edge case — there is no
    well-defined argmax when there are no candidates.

    Given: A belief root with no action children.
    When: best_action_by_reward is called.
    Then: ValueError mentioning "no action children".

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    with pytest.raises(ValueError, match="no action children"):
        tree.best_action_by_reward(root)


def test_best_action_by_cost_returns_action_with_lowest_q_value(belief):
    """best_action_by_cost picks the action label of the min-q child.

    Purpose: Validates the cost-setting argmin — symmetric counterpart
    to best_action_by_reward.

    Given: A belief root with three action children whose q_values are
    4.0, -2.0, 1.0.
    When: best_action_by_cost is called on the root.
    Then: The action label of the q=-2.0 child is returned.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    a_high = tree.add_action_node("high", parent_id=root)
    a_low = tree.add_action_node("low", parent_id=root)
    a_mid = tree.add_action_node("mid", parent_id=root)
    tree.q_value[a_high] = 4.0
    tree.q_value[a_low] = -2.0
    tree.q_value[a_mid] = 1.0
    assert tree.best_action_by_cost(root) == "low"


def test_best_action_by_cost_raises_when_belief_has_no_action_children(belief):
    """best_action_by_cost on a childless belief raises ValueError.

    Purpose: Symmetric edge case to best_action_by_reward.

    Given: A belief root with no action children.
    When: best_action_by_cost is called.
    Then: ValueError mentioning "no action children".

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    with pytest.raises(ValueError, match="no action children"):
        tree.best_action_by_cost(root)


def test_sample_belief_child_zero_total_weight_raises(belief):
    """sample_belief_child raises when all child weights are zero.

    Purpose: Validates the second guard in sample_belief_child — children
    exist but their CDF total is non-positive, so weighted sampling is
    undefined. (The "no children" branch is covered separately.)

    Given: An action node with two belief children each added with weight=0.0.
    When: sample_belief_child is called.
    Then: ValueError mentioning "non-positive".

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    tree.add_belief_node(belief, observation="o1", weight=0.0, parent_id=action)
    tree.add_belief_node(belief, observation="o2", weight=0.0, parent_id=action)
    with pytest.raises(ValueError, match="non-positive"):
        tree.sample_belief_child(action)


def test_add_belief_node_obs_key_overrides_observation_in_index(belief):
    """Explicit obs_key shadows the raw observation in obs_child_lookup.

    Purpose: Validates the obs_key branch of _register_obs_child — when
    the caller supplies a hashable surrogate, the index is keyed by that
    surrogate alone; the raw observation is NOT also registered.

    Given: An action node and a belief child added with both a hashable
    observation="raw" and obs_key="K".
    When: get_belief_child_indexed is called with obs_key="K" and again
    with the raw observation.
    Then: obs_key="K" returns the child ID; the raw observation returns
    None (only the surrogate was registered).

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    cid = tree.add_belief_node(belief, observation="raw", parent_id=action, obs_key="K")
    assert tree.get_belief_child_indexed(action, obs_key="K") == cid
    assert tree.get_belief_child_indexed(action, observation="raw") is None


def test_add_action_node_action_key_overrides_action_in_index(belief):
    """Explicit action_key shadows the raw action in action_child_lookup.

    Purpose: Symmetric to obs_key override — when the caller supplies a
    hashable surrogate, only the surrogate is registered.

    Given: A belief root and an action child added with both a hashable
    action="raw" and action_key="K".
    When: get_action_child_indexed is called with action_key="K" and again
    with the raw action.
    Then: action_key="K" returns the child ID; the raw action returns None.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    aid = tree.add_action_node("raw", parent_id=root, action_key="K")
    assert tree.get_action_child_indexed(root, action_key="K") == aid
    assert tree.get_action_child_indexed(root, action="raw") is None


def test_get_belief_child_indexed_with_obs_key_for_unhashable_observation(belief):
    """obs_key enables indexed lookup when the raw observation is unhashable.

    Purpose: Validates the typical caller pattern — supply a hashable
    surrogate (here an int) when the actual observation is an ndarray.

    Given: An action node with a belief child whose observation is an
    ndarray and whose obs_key is the integer 42.
    When: get_belief_child_indexed is called with obs_key=42 and obs_key=99.
    Then: 42 returns the child ID; 99 returns None.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    cid = tree.add_belief_node(
        belief, observation=np.array([1.0, 2.0]), parent_id=action, obs_key=42
    )
    assert tree.get_belief_child_indexed(action, obs_key=42) == cid
    assert tree.get_belief_child_indexed(action, obs_key=99) is None


def test_get_action_child_indexed_with_action_key_for_unhashable_action(belief):
    """action_key enables indexed lookup when the raw action is unhashable.

    Purpose: Symmetric to the obs_key get-side test — supply a hashable
    surrogate when the actual action is an ndarray.

    Given: A belief root with an action child whose action is an ndarray
    and whose action_key is the integer 42.
    When: get_action_child_indexed is called with action_key=42 and 99.
    Then: 42 returns the child ID; 99 returns None.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    aid = tree.add_action_node(np.array([1.0, 2.0]), parent_id=root, action_key=42)
    assert tree.get_action_child_indexed(root, action_key=42) == aid
    assert tree.get_action_child_indexed(root, action_key=99) is None


def test_reserve_does_not_change_logical_size(belief):
    """reserve(capacity) leaves len(tree) unchanged.

    Purpose: Validates that pre-allocation is invisible to the logical
    node count — only the underlying column buffers are padded.

    Given: A fresh Tree.
    When: reserve(100) is called, then one belief node is added.
    Then: len(tree) is 0 immediately after reserve, and 1 after the add.

    Test type: unit
    """
    tree = Tree()
    tree.reserve(100)
    assert len(tree) == 0
    tree.add_belief_node(belief)
    assert len(tree) == 1


def test_reserve_pads_every_column_to_capacity():
    """reserve(capacity) physically grows every per-node column.

    Purpose: Validates that the schema list inside reserve is complete —
    if a column is added to __init__/_allocate but not to reserve's list,
    that column would be undersized after reserve and the next overwrite
    would IndexError. This test catches that.

    Given: A fresh Tree.
    When: reserve(50) is called.
    Then: Every per-node column has length 50.

    Test type: unit
    """
    tree = Tree()
    tree.reserve(50)
    columns = [
        tree.kind,
        tree.parent_id,
        tree.children_ids,
        tree.visit_count,
        tree.q_value,
        tree.v_value,
        tree.lower_confidence_bound,
        tree.upper_confidence_bound,
        tree.immediate_cost,
        tree.immediate_reward,
        tree.weight,
        tree.action,
        tree.observation,
        tree.belief,
        tree.data,
        tree.sample,
        tree.children_cdf,
        tree.position_in_parent,
    ]
    for column in columns:
        assert len(column) == 50


def test_reserve_behavioral_equivalence_with_unreserved_tree(belief):
    """A reserved tree and an unreserved tree built with the same calls
    have identical column contents.

    Purpose: Validates that the "reserved-slot overwrite" branch in
    _allocate produces the same defaults and structure as the "append"
    branch — i.e. reserve is a perf-only optimisation, not a behavioural
    change.

    Given: Two Tree instances built with identical add sequences (root +
    two actions + two beliefs under the first action), with non-default
    weights and observations. One calls reserve(10) before the adds.
    When: All five nodes have been added on each tree.
    Then: For every node ID, every column has the same value in both trees.

    Test type: unit
    """

    def build(reserve_capacity):
        tree = Tree()
        if reserve_capacity is not None:
            tree.reserve(reserve_capacity)
        root = tree.add_belief_node(belief, weight=2.0)
        a0 = tree.add_action_node("a0", parent_id=root)
        tree.add_action_node("a1", parent_id=root)
        tree.add_belief_node(belief, observation="o0", weight=1.5, parent_id=a0)
        tree.add_belief_node(belief, observation="o1", weight=0.5, parent_id=a0)
        return tree

    reserved = build(reserve_capacity=10)
    unreserved = build(reserve_capacity=None)

    assert len(reserved) == len(unreserved) == 5
    for nid in range(len(reserved)):
        assert reserved.kind[nid] == unreserved.kind[nid]
        assert reserved.parent_id[nid] == unreserved.parent_id[nid]
        assert reserved.children_ids[nid] == unreserved.children_ids[nid]
        assert reserved.weight[nid] == unreserved.weight[nid]
        assert reserved.observation[nid] == unreserved.observation[nid]
        assert reserved.action[nid] == unreserved.action[nid]
        assert reserved.children_cdf[nid] == unreserved.children_cdf[nid]
        assert reserved.position_in_parent[nid] == unreserved.position_in_parent[nid]
        assert reserved.q_value[nid] == unreserved.q_value[nid]
        assert reserved.visit_count[nid] == unreserved.visit_count[nid]
        assert reserved.v_value[nid] == unreserved.v_value[nid]
        assert reserved.immediate_cost[nid] == unreserved.immediate_cost[nid]
        assert reserved.immediate_reward[nid] == unreserved.immediate_reward[nid]


def test_reserve_then_allocate_past_capacity_falls_back_to_append(belief):
    """Allocations beyond reserved capacity transparently fall back to append.

    Purpose: Validates the second branch of _allocate — once the cursor
    crosses len(self.kind), columns grow normally via append.

    Given: A tree with reserve(2).
    When: Five belief nodes are added.
    Then: IDs are 0..4, len(tree) is 5, and column length is 5 (not 2).

    Test type: unit
    """
    tree = Tree()
    tree.reserve(2)
    ids = [tree.add_belief_node(belief) for _ in range(5)]
    assert ids == [0, 1, 2, 3, 4]
    assert len(tree) == 5
    assert len(tree.kind) == 5


def test_reserve_idempotent_when_growing_preserves_existing_nodes(belief):
    """Re-reserving to a larger capacity preserves already-allocated nodes.

    Purpose: Validates the "pad up to capacity, leaving existing nodes
    untouched" semantics — calling reserve more than once must not
    corrupt prior data.

    Given: A tree with two nodes already added (root belief and one action).
    When: reserve(20) is called.
    Then: len(tree) is unchanged at 2; column length is 20; node 0 and
    node 1 retain their kind, weight, action, and parent values.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief, weight=2.5)
    tree.add_action_node("a", parent_id=root)
    assert len(tree.kind) == 2

    tree.reserve(20)

    assert len(tree) == 2
    assert len(tree.kind) == 20
    assert tree.kind[0] == BELIEF
    assert tree.kind[1] == ACTION
    assert tree.weight[0] == 2.5
    assert tree.action[1] == "a"
    assert tree.parent_id[1] == 0


def test_reserve_is_noop_when_capacity_below_current_size(belief):
    """reserve(capacity) with capacity < _size never shrinks the columns.

    Purpose: Validates the target_len = max(_size, capacity) clause —
    reserve grows-only, never truncates.

    Given: A tree with 3 nodes already added.
    When: reserve(1) is called (capacity below current size).
    Then: len(tree) stays at 3; column length stays at 3; existing entries
    are preserved.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    tree.add_belief_node(belief, parent_id=action)
    assert len(tree.kind) == 3

    tree.reserve(1)

    assert len(tree) == 3
    assert len(tree.kind) == 3
    assert tree.kind[0] == BELIEF
    assert tree.kind[1] == ACTION
    assert tree.kind[2] == BELIEF


def test_print_emits_root_label_and_metrics_format(belief, capsys):
    """print on a single root emits a BeliefNode[0] line with metrics.

    Purpose: Validates the rendering format for a node — label, ID,
    observation payload, and visits/q/v formatted with three decimals.

    Given: A Tree with one belief root.
    When: print(0) is called and stdout is captured.
    Then: Output contains "BeliefNode[0]", "obs=None", "visits=0",
    "q=0.000", "v=0.000".

    Test type: unit
    """
    tree = Tree()
    tree.add_belief_node(belief)
    tree.print(0)
    out = capsys.readouterr().out
    assert "BeliefNode[0]" in out
    assert "obs=None" in out
    assert "visits=0" in out
    assert "q=0.000" in out
    assert "v=0.000" in out


def test_print_uses_branch_markers_for_first_and_last_children(belief, capsys):
    """A parent with two children renders both branch glyphs.

    Purpose: Validates the last-vs-not-last branch in _render — the only
    conditional in the renderer.

    Given: A belief root with two action children "a0" and "a1".
    When: print(0) is called.
    Then: Output contains "├── " (first child), "└── " (last child),
    "ActionNode[1]", "ActionNode[2]", "action=a0", "action=a1".

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    tree.add_action_node("a0", parent_id=root)
    tree.add_action_node("a1", parent_id=root)
    tree.print(0)
    out = capsys.readouterr().out
    assert "├── " in out
    assert "└── " in out
    assert "ActionNode[1]" in out
    assert "ActionNode[2]" in out
    assert "action=a0" in out
    assert "action=a1" in out


def test_print_subtree_omits_lines_outside_subtree(belief, capsys):
    """print(node_id) renders only the subtree rooted at node_id.

    Purpose: Validates that _render's recursion stays local — sibling
    and ancestor nodes are not printed.

    Given: A 3-node tree: BeliefNode[0] -> ActionNode[1] -> BeliefNode[2].
    When: print(1) is called (the action node, not the root).
    Then: ActionNode[1] and BeliefNode[2] appear in the output;
    BeliefNode[0] does not.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    tree.add_belief_node(belief, observation="o", parent_id=action)
    tree.print(action)
    out = capsys.readouterr().out
    assert "ActionNode[1]" in out
    assert "BeliefNode[2]" in out
    assert "BeliefNode[0]" not in out


# --- typed accessors ---


def test_get_visit_count_reads_per_node_visit(belief):
    """``get_visit_count`` returns the same int the column store holds.

    Purpose: Validates the shared accessor reads from ``visit_count``
    correctly for both belief and action nodes.

    Given: A tree with a root belief and one action child, visit counts
    bumped to known values via direct list write.
    When: ``get_visit_count`` is called for each id.
    Then: The returned int matches the underlying column entry.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    tree.visit_count[root] = 7
    tree.visit_count[action] = 3

    assert tree.get_visit_count(root) == 7
    assert tree.get_visit_count(action) == 3


def test_get_parent_id_returns_none_for_root_and_id_for_child(belief):
    """``get_parent_id`` returns ``None`` for the root and the parent id otherwise.

    Purpose: Validates the shared parent accessor.

    Given: A root belief with one action child.
    When: ``get_parent_id`` is queried for each.
    Then: Root returns ``None``; child returns the root id.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)

    assert tree.get_parent_id(root) is None
    assert tree.get_parent_id(action) == root


def test_get_children_ids_returns_appended_children_in_order(belief):
    """``get_children_ids`` returns the same list the column store holds, in order.

    Purpose: Validates the children-id accessor preserves insertion order.

    Given: A root belief with two action children added in order ``"a"``, ``"b"``.
    When: ``get_children_ids`` is called on the root.
    Then: The returned list is ``[a_id, b_id]``.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    a_id = tree.add_action_node("a", parent_id=root)
    b_id = tree.add_action_node("b", parent_id=root)

    assert tree.get_children_ids(root) == [a_id, b_id]


def test_action_only_accessors_round_trip_action_q_and_immediate_reward(belief):
    """Action-side accessors read the action payload, q-value, and immediate-reward.

    Purpose: Validates ``get_action`` / ``get_q_value`` /
    ``get_immediate_reward`` / ``get_immediate_cost`` read the same
    values their column entries hold (after explicit writes for
    setup).

    Given: A root with one action child; q-value, immediate cost and
    reward populated via the existing setters.
    When: Each typed accessor is invoked.
    Then: Each returns the populated value.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("listen", parent_id=root)
    tree.q_value[action] = 1.25
    tree.set_immediate_reward(action, 4.0)

    assert tree.get_action(action) == "listen"
    assert tree.get_q_value(action) == pytest.approx(1.25)
    assert tree.get_immediate_reward(action) == pytest.approx(4.0)
    assert tree.get_immediate_cost(action) == pytest.approx(-4.0)


def test_belief_only_accessors_round_trip_belief_v_observation_weight(belief):
    """Belief-side accessors read the belief, v-value, observation, and weight.

    Purpose: Validates ``get_belief`` / ``get_v_value`` /
    ``get_observation`` / ``get_weight``.

    Given: A tree with a child belief node attached under an action,
    weight=2.0, observation="o", v-value=3.5 populated by direct
    column write.
    When: Each typed accessor is invoked on the child.
    Then: Each returns the populated value.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)
    child = tree.add_belief_node(belief, observation="o", weight=2.0, parent_id=action)
    tree.v_value[child] = 3.5

    assert tree.get_belief(child) is belief
    assert tree.get_v_value(child) == pytest.approx(3.5)
    assert tree.get_observation(child) == "o"
    assert tree.get_weight(child) == pytest.approx(2.0)


# --- compound mutations ---


def test_update_action_q_with_return_increments_visits_and_running_average(belief):
    """``update_action_q_with_return`` matches the standard MCTS Q-update.

    Purpose: Validates the bundled "increment visits + incrementally
    average return" mutation produces the same q-value sequence as
    a manual incremental mean.

    Given: Action child of a root belief, fresh (q=0, visits=0).
    When: Returns 4.0, then 2.0 are backed up via
    ``update_action_q_with_return``.
    Then: After the first call q=4.0 and visits=1; after the second
    q=3.0 and visits=2 (running mean of 4 and 2).

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)

    tree.update_action_q_with_return(action, 4.0)
    assert tree.get_visit_count(action) == 1
    assert tree.get_q_value(action) == pytest.approx(4.0)

    tree.update_action_q_with_return(action, 2.0)
    assert tree.get_visit_count(action) == 2
    assert tree.get_q_value(action) == pytest.approx(3.0)


def test_increment_visit_count_works_for_belief_and_action(belief):
    """``increment_visit_count`` bumps either node kind by exactly one.

    Purpose: Validates the shared visit-count incrementer.

    Given: A tree with a root belief and one action child.
    When: ``increment_visit_count`` is called twice on each.
    Then: Each visit count is exactly two; siblings are unaffected.

    Test type: unit
    """
    tree = Tree()
    root = tree.add_belief_node(belief)
    action = tree.add_action_node("a", parent_id=root)

    tree.increment_visit_count(root)
    tree.increment_visit_count(root)
    tree.increment_visit_count(action)
    tree.increment_visit_count(action)

    assert tree.get_visit_count(root) == 2
    assert tree.get_visit_count(action) == 2
