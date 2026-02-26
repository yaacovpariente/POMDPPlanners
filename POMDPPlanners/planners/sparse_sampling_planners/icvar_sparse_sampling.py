"""ICVaR Sparse Sampling POMDP Planning Algorithm Implementation.

This module implements a risk-sensitive variant of the sparse sampling algorithm
for POMDP planning. Instead of using the expected value (mean) for Bellman backups,
it uses the Conditional Value at Risk (CVaR) to focus on the worst-alpha fraction
of outcomes.

Reference:
    Pariente, Y., & Indelman, V. (2026). Online Risk-Averse Planning in POMDPs Using
    Iterated CVaR Value Function. arXiv preprint arXiv:2601.20554.
    https://arxiv.org/abs/2601.20554

Classes:
    ICVaRSparseSampling: Risk-sensitive sparse sampling with CVaR-based value updates
"""

from typing import List

import numpy as np

from POMDPPlanners.core.cost import belief_expectation_cost
from POMDPPlanners.core.environment import DiscreteActionsEnvironment, SpaceType
from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.core.tree import ActionNode
from POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling import (
    SparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.utils.statistics_utils import cvar_estimator_from_dist


class ICVaRSparseSampling(SparseSamplingDiscreteActionsPlanner):
    """Risk-sensitive sparse sampling planner using CVaR for value backups.

    This planner extends the standard sparse sampling algorithm by replacing the
    expected value (mean) in Q-value computation with the Conditional Value at Risk
    (CVaR). CVaR focuses on the worst-alpha fraction of outcomes, making the planner
    risk-sensitive.

    The standard Q-value update uses:
        Q = immediate_cost + gamma * mean(child_v_values)

    The ICVaR variant replaces this with:
        Q = immediate_cost + gamma * CVaR_alpha(child_v_values)

    Attributes:
        alpha: CVaR confidence level (0 < alpha <= 1). Lower alpha means more
            risk-sensitive (focuses on worse outcomes). alpha=1.0 recovers
            the standard expected value.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Create environment and risk-sensitive planner
        >>> tiger = TigerPOMDP(discount_factor=0.95)
        >>> planner = ICVaRSparseSampling(
        ...     environment=tiger,
        ...     branching_factor=2,
        ...     depth=2,
        ...     alpha=0.3,
        ...     name="ICVaRPlanner"
        ... )
        >>>
        >>> # Basic planner interface usage
        >>> planner.name
        'ICVaRPlanner'
        >>> planner.alpha
        0.3
        >>>
        >>> # Action selection from belief
        >>> initial_belief = get_initial_belief(tiger, n_particles=10)
        >>> actions, run_data = planner.action(initial_belief)
        >>>
        >>> # Planner space information
        >>> space_info = ICVaRSparseSampling.get_space_info()
        >>> space_info.action_space.name
        'DISCRETE'
    """

    def __init__(
        self,
        environment: DiscreteActionsEnvironment,
        branching_factor: int,
        depth: int,
        alpha: float,
        name: str = "ICVaRSparseSampling",
    ):
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")

        super().__init__(
            environment=environment,
            branching_factor=branching_factor,
            depth=depth,
            name=name,
        )

        self.alpha = alpha

    def _update_non_leaf_action_node_q_value(self, node: ActionNode):
        if node.immediate_cost is None:
            node.immediate_cost = belief_expectation_cost(
                belief=node.parent.belief, action=node.action, env=self.environment  # type: ignore
            )

        children_v_values = np.array([child.v_value for child in node.children])
        uniform_weights = np.ones(len(children_v_values)) / len(children_v_values)

        node.q_value = node.immediate_cost + self.environment.discount_factor * float(
            cvar_estimator_from_dist(
                values=children_v_values,
                weights=uniform_weights,
                alpha=self.alpha,
            )
        )

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(action_space=SpaceType.DISCRETE, observation_space=SpaceType.MIXED)

    @classmethod
    def get_info_variable_names(cls) -> List[str]:
        """Get names of policy info variables.

        Returns:
            Empty list as this planner produces no info variables.
        """
        return []
