# SPDX-License-Identifier: MIT

"""Vectorized belief tree for GPU-batched online POMDP planning.

This package provides :class:`VectorizedBeliefTree`, a reusable belief-tree
data structure stored entirely in flat PyTorch tensors. It mirrors the tensor
tree layout used by GPU-vectorized planners (VOPP / PORPP) but contains no
planning-algorithm logic — only the batched structural and statistical
primitives (keyed child insertion, composite-key lookup, scatter
aggregation, group-by-parent, depth-wise traversal) that such algorithms are
built from.

See
:mod:`POMDPPlanners.core.tree.vectorized_belief_tree.vectorized_belief_tree`
for the design.
"""

from POMDPPlanners.core.tree.vectorized_belief_tree.vectorized_belief_store import (
    BeliefStore,
)
from POMDPPlanners.core.tree.vectorized_belief_tree.vectorized_belief_tree_fields import (
    FieldRegistry,
    FieldSpec,
)
from POMDPPlanners.core.tree.vectorized_belief_tree.vectorized_belief_tree import (
    VectorizedBeliefTree,
)

__all__ = [
    "VectorizedBeliefTree",
    "BeliefStore",
    "FieldSpec",
    "FieldRegistry",
]
