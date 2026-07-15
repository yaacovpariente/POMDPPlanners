# SPDX-License-Identifier: MIT

"""Tests for :class:`VectorizedBeliefTree`.

The tree stores a POMDP belief tree as flat tensors: a node is an integer
index into every column, belief and action nodes alternate, and every
structural or statistical operation is batched. These tests validate that
contract on CPU (always) and CUDA (when available).
"""

# pylint: disable=too-many-lines

import pytest
import torch

from POMDPPlanners.core.tree.vectorized_belief_tree import (
    VectorizedBeliefTree,
)

CPU = torch.device("cpu")


def _tensor(values, device=CPU, dtype=torch.int64):
    return torch.tensor(values, device=device, dtype=dtype)


@pytest.fixture(name="tree")
def _tree_fixture():
    """Return a fresh CPU tree for each test."""
    return VectorizedBeliefTree(device=CPU)


class TestInitialization:
    """Root creation and initial counters."""

    def test_root_is_created_at_index_zero(self, tree):
        """A new tree contains exactly the root belief at index 0.

        Purpose: Validates that construction creates the root correctly.

        Given: A freshly constructed tree
        When: Its counters and root columns are inspected
        Then: The root index is 0, there is one belief and no actions, and the
            root has no parent and zero depth

        Test type: unit
        """
        assert tree.root_index == 0
        assert tree.num_belief_nodes == 1
        assert tree.num_action_nodes == 0
        assert int(tree.belief_parent_action[0]) == -1
        assert int(tree.belief_parent_observation[0]) == -1
        assert int(tree.belief_depth[0]) == 0
        assert not bool(tree.belief_terminal[0])

    def test_invalid_capacity_raises(self):
        """Non-positive capacities are rejected at construction.

        Purpose: Validates capacity argument checking.

        Given: A zero belief capacity
        When: A tree is constructed
        Then: A ValueError is raised

        Test type: unit
        """
        with pytest.raises(ValueError):
            VectorizedBeliefTree(device=CPU, belief_capacity=0)

    def test_invalid_growth_factor_raises(self):
        """A growth factor of 1 or less is rejected.

        Purpose: Validates growth-factor argument checking.

        Given: A growth factor of 1.0
        When: A tree is constructed
        Then: A ValueError is raised

        Test type: unit
        """
        with pytest.raises(ValueError):
            VectorizedBeliefTree(device=CPU, growth_factor=1.0)


class TestActionInsertion:
    """Action-node lookup and creation semantics."""

    def test_duplicate_pairs_map_to_single_action(self, tree):
        """Repeated (belief, action) pairs in one batch create one node each.

        Purpose: Validates in-batch deduplication of action creation.

        Given: A batch where pair (0, 2) appears twice alongside distinct pairs
        When: get_or_create_actions is called
        Then: The duplicate rows return the same index and only unique pairs
            create nodes

        Test type: unit
        """
        parents = _tensor([0, 0, 0, 0])
        actions = _tensor([2, 2, 4, 1])

        indices, created = tree.get_or_create_actions(parents, actions)

        assert int(indices[0]) == int(indices[1])
        assert tree.num_action_nodes == 3
        assert created.tolist() == [True, True, True, True]

    def test_existing_action_lookup_creates_nothing(self, tree):
        """Re-inserting an existing pair returns it without creating a node.

        Purpose: Validates against-tree deduplication.

        Given: A tree already holding pair (0, 2)
        When: get_or_create_actions is called again with (0, 2)
        Then: The same index is returned, created is False, and the count is
            unchanged

        Test type: unit
        """
        first, _ = tree.get_or_create_actions(_tensor([0]), _tensor([2]))

        second, created = tree.get_or_create_actions(_tensor([0]), _tensor([2]))

        assert int(first[0]) == int(second[0])
        assert created.tolist() == [False]
        assert tree.num_action_nodes == 1

    def test_find_actions_reports_hits_and_misses(self, tree):
        """find_actions returns existing indices and -1 for absent pairs.

        Purpose: Validates non-mutating action lookup.

        Given: A tree holding pair (0, 2)
        When: find_actions queries (0, 2) and (0, 9)
        Then: The present pair resolves to its index and the absent pair to -1
            without creating nodes

        Test type: unit
        """
        created_index, _ = tree.get_or_create_actions(_tensor([0]), _tensor([2]))

        indices, found = tree.find_actions(_tensor([0, 0]), _tensor([2, 9]))

        assert int(indices[0]) == int(created_index[0])
        assert int(indices[1]) == -1
        assert found.tolist() == [True, False]
        assert tree.num_action_nodes == 1

    def test_mixed_existing_and_new_batch(self, tree):
        """A batch mixing existing, repeated-new, and distinct-new pairs resolves.

        Purpose: Validates the combined dedup + lookup + create path.

        Given: A tree holding pair (0, 1), and a batch with (0, 1), (0, 5)
            twice, and (0, 7)
        When: get_or_create_actions is called
        Then: The existing pair keeps its index, repeats share an index, and
            created flags reflect only newly created rows

        Test type: unit
        """
        existing, _ = tree.get_or_create_actions(_tensor([0]), _tensor([1]))

        parents = _tensor([0, 0, 0, 0])
        actions = _tensor([1, 5, 5, 7])
        indices, created = tree.get_or_create_actions(parents, actions)

        assert int(indices[0]) == int(existing[0])
        assert int(indices[1]) == int(indices[2])
        assert created.tolist() == [False, True, True, True]
        assert tree.num_action_nodes == 3


class TestBeliefInsertion:
    """Successor-belief lookup and creation semantics."""

    def _make_actions(self, tree):
        indices, _ = tree.get_or_create_actions(_tensor([0, 0, 0]), _tensor([0, 1, 2]))
        return indices

    def test_duplicate_action_observation_pairs_collapse(self, tree):
        """Repeated (action, observation) pairs create a single belief.

        Purpose: Validates successor-belief uniqueness.

        Given: One action node and a batch where observation 2 repeats under it
        When: get_or_create_beliefs is called
        Then: Both rows return the same belief index and only unique pairs add
            nodes

        Test type: unit
        """
        actions = self._make_actions(tree)
        parent_actions = _tensor([int(actions[0]), int(actions[0]), int(actions[1])])
        observations = _tensor([2, 2, 7])

        beliefs, created = tree.get_or_create_beliefs(parent_actions, observations)

        assert int(beliefs[0]) == int(beliefs[1])
        assert tree.num_belief_nodes == 1 + 2
        assert created.tolist() == [True, True, True]

    def test_new_belief_depth_is_parent_action_depth_plus_one(self, tree):
        """A created belief's depth is one greater than its parent action.

        Purpose: Validates depth propagation on belief creation.

        Given: A root-level action node at depth 0
        When: A successor belief is created under it
        Then: The successor belief has depth 1

        Test type: unit
        """
        actions = self._make_actions(tree)

        beliefs, _ = tree.get_or_create_beliefs(_tensor([int(actions[0])]), _tensor([5]))

        assert int(tree.belief_depth[int(beliefs[0])]) == 1

    def test_terminal_flag_applies_and_ors_on_repeat(self, tree):
        """Terminal flags are set on creation and OR-combined thereafter.

        Purpose: Validates the documented logical-OR terminal semantics.

        Given: A belief created as non-terminal
        When: The same belief is re-inserted with terminal True
        Then: The existing belief is promoted to terminal

        Test type: unit
        """
        actions = self._make_actions(tree)
        beliefs, _ = tree.get_or_create_beliefs(
            _tensor([int(actions[0])]),
            _tensor([5]),
            terminal_mask=torch.tensor([False], device=CPU),
        )
        assert not bool(tree.belief_terminal[int(beliefs[0])])

        tree.get_or_create_beliefs(
            _tensor([int(actions[0])]),
            _tensor([5]),
            terminal_mask=torch.tensor([True], device=CPU),
        )

        assert bool(tree.belief_terminal[int(beliefs[0])])

    def test_find_beliefs_reports_hits_and_misses(self, tree):
        """find_beliefs resolves present pairs and marks absent ones -1.

        Purpose: Validates non-mutating belief lookup.

        Given: A belief created under action a for observation 5
        When: find_beliefs queries (a, 5) and (a, 6)
        Then: The present pair resolves and the absent pair returns -1

        Test type: unit
        """
        actions = self._make_actions(tree)
        created, _ = tree.get_or_create_beliefs(_tensor([int(actions[0])]), _tensor([5]))

        parent = int(actions[0])
        indices, found = tree.find_beliefs(_tensor([parent, parent]), _tensor([5, 6]))

        assert int(indices[0]) == int(created[0])
        assert int(indices[1]) == -1
        assert found.tolist() == [True, False]


class TestStatistics:
    """Scatter-based accumulation of action statistics."""

    def test_duplicate_indices_accumulate_visits_and_rewards(self, tree):
        """Duplicate action indices sum visits and rewards.

        Purpose: Validates scatter accumulation over repeated indices.

        Given: Three action nodes and an update batch referencing node 1 twice
        When: update_action_statistics is applied
        Then: Node 1's visits and reward sum reflect both contributions

        Test type: unit
        """
        actions, _ = tree.get_or_create_actions(_tensor([0, 0, 0]), _tensor([0, 1, 2]))
        update_indices = _tensor([int(actions[1]), int(actions[1]), int(actions[0])])
        rewards = torch.tensor([2.0, 3.0, 1.0], device=CPU)

        tree.update_action_statistics(update_indices, rewards)

        node_one = int(actions[1])
        assert int(tree.action_visit_count[node_one]) == 2
        assert float(tree.action_reward_sum[node_one]) == pytest.approx(5.0)
        assert int(tree.action_visit_count[int(actions[0])]) == 1

    def test_increment_visits_defaults_to_one(self, tree):
        """increment_action_visits adds one per row when counts is omitted.

        Purpose: Validates the default visit increment.

        Given: A single action node
        When: increment_action_visits is called without counts twice
        Then: The visit count is 2

        Test type: unit
        """
        actions, _ = tree.get_or_create_actions(_tensor([0]), _tensor([0]))

        tree.increment_action_visits(actions)
        tree.increment_action_visits(actions)

        assert int(tree.action_visit_count[int(actions[0])]) == 2


class TestTraversal:
    """Depth queries, parent links, and CSR child queries."""

    def _two_levels(self, tree):
        actions, _ = tree.get_or_create_actions(_tensor([0, 0]), _tensor([0, 1]))
        beliefs, _ = tree.get_or_create_beliefs(
            _tensor([int(actions[0]), int(actions[0]), int(actions[1])]),
            _tensor([5, 6, 7]),
        )
        return actions, beliefs

    def test_depth_queries_return_expected_nodes(self, tree):
        """Nodes-at-depth returns the correct belief and action indices.

        Purpose: Validates depth-indexed traversal.

        Given: A two-level tree
        When: Belief and action nodes are queried by depth
        Then: The root is the only depth-0 belief and successors are depth 1

        Test type: unit
        """
        self._two_levels(tree)

        root_level = tree.belief_nodes_at_depth(0)
        next_level = tree.belief_nodes_at_depth(1)
        action_level = tree.action_nodes_at_depth(0)

        assert root_level.tolist() == [0]
        assert next_level.shape[0] == 3
        assert action_level.shape[0] == 2

    def test_parent_links_round_trip(self, tree):
        """Parent accessors return the linking action/belief indices.

        Purpose: Validates parent-link accessors.

        Given: A two-level tree
        When: parent_actions and parent_beliefs are queried
        Then: A successor belief's parent action's parent belief is the root

        Test type: unit
        """
        _, beliefs = self._two_levels(tree)

        parent_action = tree.parent_actions(_tensor([int(beliefs[0])]))
        grandparent_belief = tree.parent_beliefs(parent_action)

        assert int(grandparent_belief[0]) == tree.root_index

    def test_action_children_csr_layout(self, tree):
        """action_children returns children of each belief in CSR form.

        Purpose: Validates CSR child grouping for beliefs.

        Given: A root with two action children
        When: action_children queries the root
        Then: The offsets delimit exactly the two action nodes

        Test type: unit
        """
        actions, _ = self._two_levels(tree)

        flat, offsets = tree.action_children(_tensor([tree.root_index]))

        assert offsets.tolist() == [0, 2]
        assert sorted(flat.tolist()) == sorted(actions.tolist())

    def test_belief_children_csr_layout(self, tree):
        """belief_children returns successor beliefs of each action in CSR form.

        Purpose: Validates CSR child grouping for actions.

        Given: An action node with two successor beliefs
        When: belief_children queries that action
        Then: The offsets delimit exactly its two belief children

        Test type: unit
        """
        actions, beliefs = self._two_levels(tree)

        flat, offsets = tree.belief_children(_tensor([int(actions[0])]))

        assert offsets.tolist() == [0, 2]
        assert sorted(flat.tolist()) == sorted([int(beliefs[0]), int(beliefs[1])])

    def test_reduce_belief_children_sums_per_parent(self, tree):
        """reduce_belief_children reduces child values grouped by parent action.

        Purpose: Validates the generic segmented reduction over children.

        Given: Two successor beliefs under one action with values 2 and 3
        When: reduce_belief_children sums them
        Then: The parent action's entry equals 5 and unrelated actions are 0

        Test type: unit
        """
        actions, beliefs = self._two_levels(tree)
        parent_of_children = _tensor([int(actions[0]), int(actions[0])])
        values = torch.tensor([2.0, 3.0], device=CPU)

        reduced = tree.reduce_belief_children(parent_of_children, values, reduction="sum")

        assert float(reduced[int(actions[0])]) == pytest.approx(5.0)
        assert float(reduced[int(actions[1])]) == pytest.approx(0.0)
        assert beliefs.shape[0] == 3


class TestRegisteredFields:
    """Registration, defaults, growth, reset, and access of extra fields."""

    def test_register_and_write_belief_field(self, tree):
        """A registered belief field initialises to its default and is writable.

        Purpose: Validates field registration, defaults, and access.

        Given: A belief field 'value' registered with default 0
        When: The active view is read and then written
        Then: The default is observed and the write persists

        Test type: unit
        """
        tree.register_belief_field("value", default=0.0)

        view = tree.belief_field("value")
        assert view.tolist() == [0.0]
        view[0] = 7.5
        assert float(tree.belief_field("value")[0]) == pytest.approx(7.5)

    def test_vector_valued_field_shape(self, tree):
        """A field with a trailing shape allocates a per-node vector.

        Purpose: Validates non-scalar registered fields.

        Given: A belief field 'preferences' of shape (3,)
        When: Its active view is inspected
        Then: The view has one row of width 3

        Test type: unit
        """
        tree.register_belief_field("preferences", shape=(3,), default=0.0)

        view = tree.belief_field("preferences")

        assert tuple(view.shape) == (1, 3)

    def test_duplicate_field_name_raises(self, tree):
        """Registering a duplicate or reserved field name raises.

        Purpose: Validates field-name collision checks.

        Given: A tree with 'value' already registered
        When: 'value' is registered again and 'depth' (reserved) is registered
        Then: Both raise ValueError

        Test type: unit
        """
        tree.register_belief_field("value")

        with pytest.raises(ValueError):
            tree.register_belief_field("value")
        with pytest.raises(ValueError):
            tree.register_belief_field("depth")

    def test_registered_field_grows_with_capacity(self):
        """A registered field is preserved and extended across capacity growth.

        Purpose: Validates that field tensors grow in lockstep with nodes.

        Given: A tiny-capacity tree with a registered action field
        When: Enough action nodes are inserted to force growth
        Then: Every active node has the field default and no error occurs

        Test type: unit
        """
        tree = VectorizedBeliefTree(device=CPU, action_capacity=2)
        tree.register_action_field("q_value", default=0.0)

        parents = _tensor([0, 0, 0, 0, 0])
        actions = _tensor([0, 1, 2, 3, 4])
        tree.get_or_create_actions(parents, actions)

        assert tree.action_capacity >= 5
        assert tree.action_field("q_value").tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


class TestCapacityAndReset:
    """Geometric growth, clear, and statistic reset."""

    def test_capacity_growth_preserves_nodes(self):
        """Forcing multiple expansions preserves existing nodes.

        Purpose: Validates geometric capacity growth correctness.

        Given: A tree with belief and action capacity 1
        When: Several actions and beliefs are inserted, forcing growth
        Then: Counts are correct, capacities grew, and validate passes

        Test type: unit
        """
        tree = VectorizedBeliefTree(device=CPU, belief_capacity=1, action_capacity=1)

        actions, _ = tree.get_or_create_actions(_tensor([0, 0, 0]), _tensor([0, 1, 2]))
        tree.get_or_create_beliefs(
            _tensor([int(actions[0]), int(actions[1]), int(actions[2])]),
            _tensor([1, 2, 3]),
        )

        assert tree.num_action_nodes == 3
        assert tree.num_belief_nodes == 4
        assert tree.action_capacity >= 3
        assert tree.belief_capacity >= 4
        tree.validate()

    def test_clear_keeps_only_root(self, tree):
        """clear removes all nodes except the root and keeps capacity.

        Purpose: Validates the clear operation.

        Given: A populated tree
        When: clear is called
        Then: Only the root remains and validate passes

        Test type: unit
        """
        actions, _ = tree.get_or_create_actions(_tensor([0, 0]), _tensor([0, 1]))
        tree.get_or_create_beliefs(_tensor([int(actions[0])]), _tensor([5]))

        tree.clear()

        assert tree.num_belief_nodes == 1
        assert tree.num_action_nodes == 0
        tree.validate()

    def test_reset_statistics_preserves_topology(self, tree):
        """reset_statistics zeroes stats but keeps nodes and links.

        Purpose: Validates statistic reset without topology loss.

        Given: A tree with accumulated visit counts and rewards
        When: reset_statistics is called
        Then: Stats are zero, node counts are unchanged, and validate passes

        Test type: unit
        """
        actions, _ = tree.get_or_create_actions(_tensor([0, 0]), _tensor([0, 1]))
        tree.update_action_statistics(actions, torch.tensor([1.0, 2.0], device=CPU))

        tree.reset_statistics()

        assert int(tree.action_visit_count[: tree.num_action_nodes].sum()) == 0
        assert float(tree.action_reward_sum[: tree.num_action_nodes].sum()) == pytest.approx(0.0)
        assert tree.num_action_nodes == 2
        tree.validate()


class TestSerialization:
    """State-dict round trips and validation."""

    def test_state_dict_round_trip_restores_tree(self, tree):
        """A tree survives a state_dict / load_state_dict round trip.

        Purpose: Validates serialization of topology, stats, and fields.

        Given: A populated tree with a registered action field
        When: Its state is saved and loaded into a new tree
        Then: Counts, statistics, and field values match and validate passes

        Test type: unit
        """
        tree.register_action_field("q_value", default=0.0)
        actions, _ = tree.get_or_create_actions(_tensor([0, 0]), _tensor([0, 1]))
        tree.update_action_statistics(actions, torch.tensor([1.0, 2.0], device=CPU))
        tree.action_field("q_value")[0] = 9.0
        state = tree.state_dict()

        restored = VectorizedBeliefTree(device=CPU)
        restored.load_state_dict(state)

        assert restored.num_belief_nodes == tree.num_belief_nodes
        assert restored.num_action_nodes == tree.num_action_nodes
        assert float(restored.action_field("q_value")[0]) == pytest.approx(9.0)
        assert restored.action_reward_sum[: restored.num_action_nodes].tolist() == [1.0, 2.0]
        restored.validate()

    def test_validate_detects_valid_tree(self, tree):
        """validate passes on a well-formed multi-level tree.

        Purpose: Validates the invariant checker on a correct tree.

        Given: A two-level tree built through the public API
        When: validate is called
        Then: No assertion is raised

        Test type: unit
        """
        actions, _ = tree.get_or_create_actions(_tensor([0, 0]), _tensor([0, 1]))
        tree.get_or_create_beliefs(_tensor([int(actions[0])]), _tensor([5]))

        tree.validate()


class TestInvalidInputs:
    """Input validation for device, dtype, and shape."""

    def test_non_integer_keys_rejected(self, tree):
        """Float action keys are rejected.

        Purpose: Validates integer-dtype enforcement on keys.

        Given: A float action-key tensor
        When: get_or_create_actions is called
        Then: A TypeError is raised

        Test type: unit
        """
        with pytest.raises(TypeError):
            tree.get_or_create_actions(_tensor([0]), torch.tensor([1.0], device=CPU))

    def test_mismatched_batch_sizes_rejected(self, tree):
        """Parent and key tensors of different lengths are rejected.

        Purpose: Validates batch-size matching.

        Given: A parent batch of length 2 and a key batch of length 1
        When: get_or_create_actions is called
        Then: A ValueError is raised

        Test type: unit
        """
        with pytest.raises(ValueError):
            tree.get_or_create_actions(_tensor([0, 0]), _tensor([1]))

    def test_wrong_dimensionality_rejected(self, tree):
        """A 2-D index tensor is rejected.

        Purpose: Validates the one-dimensional input requirement.

        Given: A 2-D parent tensor
        When: get_or_create_actions is called
        Then: A ValueError is raised

        Test type: unit
        """
        with pytest.raises(ValueError):
            tree.get_or_create_actions(_tensor([[0]]), _tensor([[1]]))

    def test_wrong_device_rejected(self, tree):
        """A CUDA tensor is rejected by a CPU tree.

        Purpose: Validates device consistency without silent movement.

        Given: A CPU tree and a CUDA parent tensor
        When: get_or_create_actions is called
        Then: A ValueError is raised

        Test type: unit
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        with pytest.raises(ValueError):
            tree.get_or_create_actions(
                torch.tensor([0], device="cuda"), torch.tensor([1], device="cuda")
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCuda:
    """The tree operates end-to-end on CUDA."""

    def test_build_and_validate_on_cuda(self):
        """A tree builds, accumulates stats, and validates on CUDA.

        Purpose: Validates that all persistent tensors stay on the CUDA device.

        Given: A tree constructed on cuda
        When: Actions and beliefs are inserted and statistics accumulated
        Then: All columns are on cuda and validate passes

        Test type: integration
        """
        device = torch.device("cuda")
        tree = VectorizedBeliefTree(device=device)
        actions, _ = tree.get_or_create_actions(
            torch.tensor([0, 0, 0], device=device), torch.tensor([0, 1, 1], device=device)
        )
        tree.update_action_statistics(actions, torch.tensor([1.0, 2.0, 3.0], device=device))
        tree.get_or_create_beliefs(actions, torch.tensor([4, 5, 5], device=device))

        assert tree.action_reward_sum.device.type == "cuda"
        assert int(tree.action_visit_count[int(actions[1])]) == 2
        tree.validate()

    def test_move_between_devices(self):
        """to() moves every built-in and registered tensor.

        Purpose: Validates device movement of a populated tree.

        Given: A populated CPU tree with a registered field
        When: to('cuda') then to('cpu') is called
        Then: Columns and fields end on CPU and validate passes

        Test type: integration
        """
        tree = VectorizedBeliefTree(device=CPU)
        tree.register_action_field("q_value", default=0.0)
        tree.get_or_create_actions(_tensor([0, 0]), _tensor([0, 1]))

        tree.to(torch.device("cuda")).to(CPU)

        assert tree.action_parent_belief.device.type == "cpu"
        assert tree.action_field("q_value").device.type == "cpu"
        tree.validate()
