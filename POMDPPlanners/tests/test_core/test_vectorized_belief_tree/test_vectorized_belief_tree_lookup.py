# SPDX-License-Identifier: MIT

"""Tests for the pure vectorized lookup and grouping helpers.

These cover the batched primitives underpinning the belief tree —
composite-pair matching, pair deduplication, CSR child grouping, boolean
scatter-OR, and segmented reductions — in isolation from the tree class.
"""

import torch

from POMDPPlanners.core.tree.vectorized_belief_tree.vectorized_belief_tree_lookup import (
    csr_children,
    match_pairs,
    scatter_or,
    segment_reduce,
    unique_pairs,
)


def test_match_pairs_finds_existing_and_marks_missing():
    """Composite-pair matching returns existing indices and -1 for misses.

    Purpose: Validates that match_pairs resolves query pairs against existing
        pairs while preserving batch order.

    Given: Two existing (first, second) pairs and a query batch mixing a hit,
        a miss, and a repeated hit
    When: match_pairs is called
    Then: Matching queries return the existing index and misses return -1, in
        input order

    Test type: unit
    """
    existing_first = torch.tensor([0, 3])
    existing_second = torch.tensor([2, 1])
    query_first = torch.tensor([0, 5, 3, 0])
    query_second = torch.tensor([2, 9, 1, 2])

    matched = match_pairs(existing_first, existing_second, query_first, query_second)

    assert matched.tolist() == [0, -1, 1, 0]


def test_match_pairs_with_no_existing_returns_all_missing():
    """match_pairs against an empty existing set returns all -1.

    Purpose: Validates the empty-tree lookup path.

    Given: No existing pairs and a non-empty query batch
    When: match_pairs is called
    Then: Every query resolves to -1

    Test type: unit
    """
    empty = torch.empty(0, dtype=torch.int64)
    query_first = torch.tensor([1, 2])
    query_second = torch.tensor([3, 4])

    matched = match_pairs(empty, empty.clone(), query_first, query_second)

    assert matched.tolist() == [-1, -1]


def test_unique_pairs_collapses_duplicates_with_inverse():
    """unique_pairs deduplicates and returns a reconstructing inverse.

    Purpose: Validates deduplication of a pair batch and the inverse mapping.

    Given: A batch of pairs with two identical entries
    When: unique_pairs is called
    Then: Distinct pairs appear once and the inverse reproduces the input

    Test type: unit
    """
    first = torch.tensor([0, 0, 1, 0])
    second = torch.tensor([2, 2, 5, 9])

    unique_first, unique_second, inverse = unique_pairs(first, second)

    reconstructed_first = unique_first[inverse]
    reconstructed_second = unique_second[inverse]
    assert unique_first.shape[0] == 3
    assert reconstructed_first.tolist() == first.tolist()
    assert reconstructed_second.tolist() == second.tolist()


def test_csr_children_groups_by_parent():
    """csr_children returns a CSR grouping of nodes by parent.

    Purpose: Validates segmented child grouping with offsets.

    Given: A parent column assigning four nodes to parents 0, 1, 0, 2
    When: csr_children queries parents [0, 1, 2, 3]
    Then: Offsets delimit each parent's children and an absent parent is empty

    Test type: unit
    """
    parent_column = torch.tensor([0, 1, 0, 2])
    query_parents = torch.tensor([0, 1, 2, 3])

    flat, offsets = csr_children(parent_column, query_parents)

    children_of_zero = sorted(flat[offsets[0] : offsets[1]].tolist())
    children_of_three = flat[offsets[3] : offsets[4]].tolist()
    assert children_of_zero == [0, 2]
    assert flat[offsets[1] : offsets[2]].tolist() == [1]
    assert children_of_three == []


def test_scatter_or_reduces_booleans_by_index():
    """scatter_or ORs boolean values into their target positions.

    Purpose: Validates duplicate-safe boolean scatter used for terminal flags.

    Given: Values routed to indices where one target sees both True and False
    When: scatter_or reduces them into a size-4 tensor
    Then: A target is True iff any routed value is True; others are False

    Test type: unit
    """
    indices = torch.tensor([0, 0, 2])
    values = torch.tensor([False, True, False])

    result = scatter_or(indices, values, size=4)

    assert result.tolist() == [True, False, False, False]


def test_segment_reduce_supports_sum_mean_max_min():
    """segment_reduce computes each supported reduction with empty defaults.

    Purpose: Validates the generic segmented reduction helper.

    Given: Values grouped into segments with one empty segment
    When: segment_reduce is applied with each supported reduction
    Then: Non-empty segments reduce correctly and empty segments are 0

    Test type: unit
    """
    segment_ids = torch.tensor([0, 0, 2])
    values = torch.tensor([1.0, 3.0, 5.0])

    total = segment_reduce(segment_ids, values, 4, "sum")
    mean = segment_reduce(segment_ids, values, 4, "mean")
    maximum = segment_reduce(segment_ids, values, 4, "max")
    minimum = segment_reduce(segment_ids, values, 4, "min")

    assert total.tolist() == [4.0, 0.0, 5.0, 0.0]
    assert mean.tolist() == [2.0, 0.0, 5.0, 0.0]
    assert maximum.tolist() == [3.0, 0.0, 5.0, 0.0]
    assert minimum.tolist() == [1.0, 0.0, 5.0, 0.0]


def test_segment_reduce_rejects_unknown_reduction():
    """segment_reduce raises on an unsupported reduction name.

    Purpose: Validates input validation of the reduction argument.

    Given: A valid segment/value batch
    When: segment_reduce is called with an unknown reduction name
    Then: A ValueError is raised

    Test type: unit
    """
    segment_ids = torch.tensor([0, 1])
    values = torch.tensor([1.0, 2.0])

    try:
        segment_reduce(segment_ids, values, 2, "median")
        raised = False
    except ValueError:
        raised = True

    assert raised
