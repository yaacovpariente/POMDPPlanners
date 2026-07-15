# SPDX-License-Identifier: MIT

"""Vectorized composite-key lookup and segmented-grouping helpers.

These are pure tensor functions used by
:class:`~POMDPPlanners.core.tree.vectorized_belief_tree.vectorized_belief_tree.VectorizedBeliefTree`.
None of them mutate persistent tree state; each operates on the flat column
tensors passed in and returns index or value tensors. Every function is
batched — there are no Python loops over individual rows.

The two structural keys of a belief tree are composite integer pairs:

* ``(parent_belief_index, action_key)`` identifying an action node;
* ``(parent_action_index, observation_key)`` identifying a belief node.

Matching them is done with :func:`match_pairs`, which packs a pair-match into
a single :func:`torch.unique` call rather than an unsafe integer packing that
could overflow ``int64`` when either component is large.

Complexity notes use ``E`` for the number of existing rows, ``Q`` for the
query batch size, ``A`` for the number of active nodes, and ``M`` for the
number of queried parents.
"""

from typing import Tuple

import torch
from torch import Tensor


def match_pairs(
    existing_first: Tensor,
    existing_second: Tensor,
    query_first: Tensor,
    query_second: Tensor,
) -> Tensor:
    """Match each query pair against the existing pairs.

    Args:
        existing_first: ``[E]`` first component of the existing pairs.
        existing_second: ``[E]`` second component of the existing pairs.
        query_first: ``[Q]`` first component of the query pairs.
        query_second: ``[Q]`` second component of the query pairs.

    Returns:
        ``[Q]`` tensor whose ``i``-th entry is the index into the existing
        pairs that matches query ``i``, or ``-1`` if no existing pair matches.
        Order follows the query batch.

    In belief-tree terms each pair is a composite node key: for action-node
    lookup the components are ``(parent_belief_index, action_key)``; for
    belief-node lookup they are ``(parent_action_index, observation_key)``. The
    ``existing_*`` tensors are the corresponding column tensors of the nodes
    already in the tree and the ``query_*`` tensors are the keys being searched;
    a returned index is the matching child node, ``-1`` meaning that edge does
    not exist yet.

    Complexity:
        ``O((E + Q) log (E + Q))`` from a single lexicographic ``torch.unique``.

    Example:
        Three existing action nodes keyed by ``(parent_belief, action_key)``,
        looked up by three query pairs (matching on the pair, not either
        component alone)::

            >>> import torch
            >>> existing_first = torch.tensor([0, 0, 1])   # parent belief index
            >>> existing_second = torch.tensor([2, 4, 2])  # action key
            >>> query_first = torch.tensor([0, 1, 0])
            >>> query_second = torch.tensor([4, 2, 9])
            >>> match_pairs(existing_first, existing_second, query_first, query_second)
            tensor([ 1,  2, -1])

        Query ``(0, 4)`` matches existing row ``1``, ``(1, 2)`` matches row
        ``2``, and ``(0, 9)`` has no match so it returns ``-1``.
    """
    device = query_first.device
    num_existing = existing_first.shape[0]
    num_query = query_first.shape[0]
    if num_query == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    all_first = torch.cat([existing_first, query_first])
    all_second = torch.cat([existing_second, query_second])
    pairs = torch.stack([all_first, all_second], dim=1)
    unique_pairs_tensor, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    num_groups = unique_pairs_tensor.shape[0]
    group_to_existing = torch.full((num_groups,), -1, dtype=torch.int64, device=device)
    if num_existing > 0:
        existing_groups = inverse[:num_existing]
        group_to_existing[existing_groups] = torch.arange(
            num_existing, dtype=torch.int64, device=device
        )
    query_groups = inverse[num_existing:]
    return group_to_existing[query_groups]


def unique_pairs(first: Tensor, second: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Deduplicate a batch of integer pairs.

    Args:
        first: ``[Q]`` first component of the pairs.
        second: ``[Q]`` second component of the pairs.

    Returns:
        A tuple ``(unique_first, unique_second, inverse)`` where the unique
        arrays hold each distinct pair once and ``inverse`` maps every input
        row to its unique-pair index (``inverse`` has length ``Q``).

    Complexity:
        ``O(Q log Q)``.
    """
    if first.shape[0] == 0:
        empty = torch.empty(0, dtype=first.dtype, device=first.device)
        return empty, empty.clone(), empty.clone()
    pairs = torch.stack([first, second], dim=1)
    unique_tensor, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    return unique_tensor[:, 0].contiguous(), unique_tensor[:, 1].contiguous(), inverse


def csr_children(parent_column: Tensor, query_parents: Tensor) -> Tuple[Tensor, Tensor]:
    """Group node indices by parent into a CSR (segmented) representation.

    Args:
        parent_column: ``[A]`` parent id of every active node of one kind.
            Position ``a`` holds the parent id of node ``a``.
        query_parents: ``[M]`` parent ids whose children are requested.

    Returns:
        A tuple ``(flat_children, offsets)`` where ``offsets`` has length
        ``M + 1`` and the children of ``query_parents[m]`` are
        ``flat_children[offsets[m]:offsets[m + 1]]``.

    Complexity:
        ``O(A log A + M log A)``.
    """
    device = parent_column.device
    num_query = query_parents.shape[0]
    order = torch.argsort(parent_column)
    sorted_parents = parent_column[order]
    low = torch.searchsorted(sorted_parents, query_parents, side="left")
    high = torch.searchsorted(sorted_parents, query_parents, side="right")
    counts = high - low
    offsets = torch.zeros(num_query + 1, dtype=torch.int64, device=device)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    total = int(offsets[-1].item())
    if total == 0:
        return torch.empty(0, dtype=torch.int64, device=device), offsets
    segment = torch.repeat_interleave(torch.arange(num_query, device=device), counts)
    within = torch.arange(total, device=device) - offsets[segment]
    source = low[segment] + within
    return order[source], offsets


def scatter_or(indices: Tensor, values: Tensor, size: int) -> Tensor:
    """Reduce boolean ``values`` by logical OR, grouped by ``indices``.

    Args:
        indices: ``[N]`` target index of every value (may contain duplicates).
        values: ``[N]`` boolean values to OR into their target positions.
        size: Length of the output tensor.

    Returns:
        ``[size]`` boolean tensor; position ``p`` is ``True`` iff any value
        routed to ``p`` is ``True``. Untargeted positions are ``False``.
    """
    contrib = torch.zeros(size, dtype=torch.uint8, device=values.device)
    contrib.scatter_reduce_(0, indices, values.to(torch.uint8), reduce="amax", include_self=True)
    return contrib.to(torch.bool)


def segment_reduce(
    segment_ids: Tensor,
    values: Tensor,
    num_segments: int,
    reduction: str,
) -> Tensor:
    """Reduce ``values`` grouped by ``segment_ids`` into a dense tensor.

    Args:
        segment_ids: ``[N]`` segment index of every value.
        values: ``[N]`` values to reduce.
        num_segments: Length of the output tensor.
        reduction: One of ``"sum"``, ``"mean"``, ``"max"``, ``"min"``.

    Returns:
        ``[num_segments]`` reduced values. Empty segments are ``0`` for every
        reduction (documented default so the result is always well defined).

    Raises:
        ValueError: If ``reduction`` is not a supported name.
    """
    out = torch.zeros(num_segments, dtype=values.dtype, device=values.device)
    if reduction == "sum":
        return out.index_add_(0, segment_ids, values)
    if reduction == "mean":
        return _segment_mean(segment_ids, values, num_segments, out)
    if reduction in ("max", "min"):
        reduce_op = "amax" if reduction == "max" else "amin"
        out.scatter_reduce_(0, segment_ids, values, reduce=reduce_op, include_self=False)
        return out
    raise ValueError(f"unsupported reduction '{reduction}'")


def _segment_mean(
    segment_ids: Tensor,
    values: Tensor,
    num_segments: int,
    out: Tensor,
) -> Tensor:
    totals = out.index_add_(0, segment_ids, values)
    counts = torch.zeros(num_segments, dtype=values.dtype, device=values.device)
    counts.index_add_(0, segment_ids, torch.ones_like(values))
    safe_counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    return totals / safe_counts
