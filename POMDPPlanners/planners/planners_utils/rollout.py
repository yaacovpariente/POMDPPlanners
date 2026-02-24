from typing import Any

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler


def random_rollout_action_sampler(
    state: Any,
    depth: int,
    action_sampler: ActionSampler,
    environment: Environment,
    discount_factor: float,
    max_depth: int = 10,
) -> float:
    """Perform random rollout to estimate value from leaf node.

    Rollout policy samples random actions using the action_sampler until
    reaching maximum depth or terminal state. This provides value estimates
    for leaf nodes in the search tree during Monte Carlo Tree Search.

    The rollout uses a random policy (via action_sampler) to quickly estimate
    the value of a state without expensive planning. This is a key component
    of MCTS algorithms where accurate value estimation is traded off against
    computational efficiency.

    Args:
        state: Current state to rollout from
        depth: Current depth in rollout (starts at 0)
        action_sampler: Action sampler for selecting rollout actions
        environment: POMDP environment to simulate in
        discount_factor: Discount factor for future rewards (0 < γ ≤ 1)
        max_depth: Maximum rollout depth to prevent infinite loops

    Returns:
        Total discounted return from rollout simulation

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> from POMDPPlanners.planners.planners_utils.rollout import random_rollout_action_sampler
        >>> from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>>
        >>> # Simple action sampler for Tiger POMDP
        >>> class TigerActionSampler(ActionSampler):
        ...     def sample(self, belief_node=None):
        ...         return np.random.choice(["listen", "open_left", "open_right"])
        >>>
        >>> # Create environment and sampler
        >>> tiger = TigerPOMDP(discount_factor=0.95)
        >>> action_sampler = TigerActionSampler()
        >>>
        >>> # Perform rollout from initial state
        >>> initial_state = "tiger_left"
        >>> rollout_value = random_rollout_action_sampler(
        ...     state=initial_state,
        ...     depth=0,
        ...     action_sampler=action_sampler,
        ...     environment=tiger,
        ...     discount_factor=0.95,
        ...     max_depth=10
        ... )  # doctest: +SKIP
    """
    if depth >= max_depth or environment.is_terminal(state=state):
        return 0.0

    action = action_sampler.sample()
    next_state = environment.state_transition_model(state=state, action=action).sample()[0]
    reward = environment.reward(state=state, action=action)

    return reward + discount_factor * random_rollout_action_sampler(
        state=next_state,
        depth=depth + 1,
        action_sampler=action_sampler,
        environment=environment,
        discount_factor=discount_factor,
        max_depth=max_depth,
    )
