from typing import Union

from anytree import RenderTree

from POMDPPlanners.core.tree.anytree_based.action_node import ActionNode
from POMDPPlanners.core.tree.anytree_based.belief_node import BeliefNode


def print_tree(tree: Union[BeliefNode, ActionNode]):
    for pre, _, node in RenderTree(tree):
        if isinstance(node, BeliefNode):
            name = node.spec
        elif isinstance(node, ActionNode):
            name = node.spec
        else:
            name = str(node)

        print(f"{pre}{name}")
