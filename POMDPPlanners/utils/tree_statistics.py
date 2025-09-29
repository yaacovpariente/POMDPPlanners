from typing import List

import numpy as np
from scipy.stats import entropy

from POMDPPlanners.core.policy import PolicyInfoVariable
from POMDPPlanners.core.tree import ActionNode, BeliefNode


def get_v_values_sample(action_node: ActionNode) -> np.ndarray:
    if not action_node.is_leaf:
        v_values_sample = np.array([child.v_value for child in action_node.children], dtype=float)
        children_visit_counts = np.array(
            [child.visit_count for child in action_node.children], dtype=int
        )
        v_values_sample = np.repeat(v_values_sample, children_visit_counts)
    else:
        v_values_sample = np.array([], dtype=float)

    return v_values_sample


def compute_tree_metrics(tree: BeliefNode) -> List[PolicyInfoVariable]:
    """Compute comprehensive statistics for MCTS tree analysis and debugging.

    Extracts key metrics from MCTS search trees to understand algorithm behavior,
    convergence properties, and search quality. These metrics are essential for
    algorithm debugging, parameter tuning, and performance analysis.

    The function analyzes the root belief node's action children to compute
    visitation statistics and exploration patterns that indicate search quality.

    Args:
        tree: Root belief node of the MCTS search tree

    Returns:
        List of PolicyInfoVariable containing computed tree metrics:
        - min_actions_visit_count: Minimum visits to any action
        - max_actions_visit_count: Maximum visits to any action
        - actions_visit_count_entropy: Shannon entropy of action visit distribution
        - n_actions_from_root: Number of actions from the root node
        - root_visit_count: Number of visits to the root node
    Raises:
        TypeError: If tree is not a BeliefNode instance

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> from POMDPPlanners.utils.tree_statistics import compute_tree_metrics
        >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>>
        >>> # Create POMCP planner and run planning
        >>> env = TigerPOMDP(discount_factor=0.95)
        >>> planner = POMCP(
        ...     environment=env,
        ...     discount_factor=0.95,
        ...     depth=20,
        ...     exploration_constant=1.0,
        ...     name="POMCP_Analysis",
        ...     n_simulations=100
        ... )
        >>>
        >>> initial_belief = get_initial_belief(env, n_particles=200)
        >>> action, run_data = planner.action(initial_belief)
        >>>
        >>> # Extract tree metrics from run data
        >>> metrics = run_data.info_variables
        >>> len(metrics) > 0
        True
        >>> import numbers
        >>> isinstance(metrics[0].value, numbers.Number)
        True

    Metric Interpretation:
        **min_actions_visit_count**:
        - Higher values indicate more balanced exploration
        - Very low values suggest some actions are barely explored
        - Zero indicates leaf node (no actions expanded)

        **max_actions_visit_count**:
        - Shows most visited action's exploration intensity
        - Compare with total simulations to assess concentration
        - Higher values indicate convergence to preferred action

        **actions_visit_count_entropy**:
        - Measures uniformity of action exploration
        - Higher entropy = more uniform exploration
        - Lower entropy = more concentrated search
        - log₂(|A|) is maximum possible entropy for |A| actions

        **Visit Ratio (max/min)**:
        - 1.0 = perfectly uniform exploration
        - Higher values = more concentrated/converged search
        - Very high ratios may indicate premature convergence

    Algorithm Debugging Applications:
        **Parameter Tuning**:
        - Low entropy → increase exploration parameter
        - High min visits but low entropy → decrease simulations
        - Extreme visit ratios → adjust exploration/exploitation balance

        **Convergence Analysis**:
        - Entropy decreasing over time indicates convergence
        - Stable visit ratios suggest algorithm has converged
        - Oscillating metrics may indicate unstable parameters

        **Comparative Analysis**:
        - Compare entropy across algorithms to assess exploration quality
        - Use visit patterns to understand different search strategies
        - Identify algorithms with better exploration-exploitation tradeoffs
    """
    if not isinstance(tree, BeliefNode):
        raise TypeError("tree must be a BeliefNode instance")

    if tree.is_leaf:
        return [
            PolicyInfoVariable(
                name="min_actions_visit_count",
                value=0,
            ),
            PolicyInfoVariable(
                name="max_actions_visit_count",
                value=0,
            ),
            PolicyInfoVariable(
                name="actions_visit_count_entropy",
                value=0,
            ),
            PolicyInfoVariable(
                name="is_leaf",
                value=1,
            ),
        ]

    visit_counts = np.array([node.visit_count for node in tree.children])
    n_actions_from_root = len(visit_counts)
    root_visit_count = tree.visit_count
    # Calculate entropy of visit counts
    total_visits: int = int(np.sum(visit_counts))
    if total_visits > 0:
        probabilities = visit_counts / total_visits
        entropy_value = entropy(probabilities, base=2)
    else:
        entropy_value = 0.0

    return [
        PolicyInfoVariable(
            name="min_actions_visit_count",
            value=np.min(visit_counts),
        ),
        PolicyInfoVariable(
            name="max_actions_visit_count",
            value=np.max(visit_counts),
        ),
        PolicyInfoVariable(
            name="actions_visit_count_entropy",
            value=entropy_value,  # type: ignore
        ),
        PolicyInfoVariable(
            name="n_actions_from_root",
            value=n_actions_from_root,
        ),
        PolicyInfoVariable(
            name="root_visit_count",
            value=root_visit_count,
        ),
        PolicyInfoVariable(
            name="tree_max_depth",
            value=tree.height - 1,
        ),
        PolicyInfoVariable(
            name="is_leaf",
            value=0,
        ),
    ]
