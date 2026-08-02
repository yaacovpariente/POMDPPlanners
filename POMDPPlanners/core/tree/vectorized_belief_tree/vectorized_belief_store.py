# SPDX-License-Identifier: MIT

"""Protocol separating belief *content* from belief-tree *topology*.

:class:`~POMDPPlanners.core.tree.vectorized_belief_tree.vectorized_belief_tree.VectorizedBeliefTree`
stores only the topology, integer keys, and statistics of a POMDP belief
tree. The mathematical belief attached to each belief node (a particle set,
a categorical vector, a Gaussian, an action-observation history, a latent
vector, ...) lives outside the tree in an object implementing
:class:`BeliefStore`.

The contract is that a belief-tree node index is the identifier the external
store uses to look up the belief for that node::

    VectorizedBeliefTree stores tree topology and statistics.
    BeliefStore          stores or reconstructs the mathematical beliefs.

The tree never calls a :class:`BeliefStore`; the protocol exists so planners
can keep a parallel store keyed by the same integer node indices the tree
hands out.
"""

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class BeliefStore(Protocol):
    """Interface for an external store of per-node belief representations.

    Implementations map belief-tree node indices to concrete belief objects.
    They are entirely optional: the tree is fully functional without one.
    """

    def create_root(self, belief: object) -> int:
        """Register the root belief and return its node index (expected ``0``).

        Args:
            belief: The concrete root belief object to store.

        Returns:
            The integer node index the belief was stored under.
        """
        ...  # pylint: disable=unnecessary-ellipsis

    def create_successors(
        self,
        parent_belief_indices: Tensor,
        action_keys: Tensor,
        observation_keys: Tensor,
    ) -> Tensor:
        """Create and store successor beliefs for a batch of transitions.

        Args:
            parent_belief_indices: ``[batch]`` parent belief node indices.
            action_keys: ``[batch]`` integer action keys taken.
            observation_keys: ``[batch]`` integer observation keys received.

        Returns:
            ``[batch]`` successor belief node indices, aligned with the inputs.
        """
        ...  # pylint: disable=unnecessary-ellipsis
