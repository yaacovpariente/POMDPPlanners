# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

from typing import Any, Tuple

import numpy as np

from POMDPPlanners.core.environment import ConstrainedEnvironment, Environment
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler


def python_random_rollout(
    state: Any,
    depth: int,
    action_sampler: ActionSampler,
    environment: Environment,
    discount_factor: float,
    max_depth: int = 10,
) -> float:
    """Recursive Python rollout that bypasses the env-level native dispatch.

    Used as the fallback path inside env-specific ``simulate_random_rollout``
    overrides (e.g. when the native kernel cannot handle a particular reward
    model or parameter configuration).
    """
    if depth >= max_depth or environment.is_terminal(state=state):
        return 0.0
    action = action_sampler.sample()
    next_state = environment.sample_next_state(state=state, action=action)
    reward = environment.reward(state=state, action=action, next_state=next_state)
    return reward + discount_factor * python_random_rollout(
        state=next_state,
        depth=depth + 1,
        action_sampler=action_sampler,
        environment=environment,
        discount_factor=discount_factor,
        max_depth=max_depth,
    )


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
    native_rollout = getattr(environment, "simulate_random_rollout", None)
    if native_rollout is not None:
        return native_rollout(
            state=state,
            action_sampler=action_sampler,
            max_depth=max_depth,
            discount_factor=discount_factor,
            depth=depth,
        )

    return python_random_rollout(
        state=state,
        depth=depth,
        action_sampler=action_sampler,
        environment=environment,
        discount_factor=discount_factor,
        max_depth=max_depth,
    )


def cost_aware_random_rollout(
    state: Any,
    depth: int,
    action_sampler: ActionSampler,
    environment: ConstrainedEnvironment,
    discount_factor: float,
    max_depth: int,
    n_constraints: int,
) -> Tuple[float, np.ndarray]:
    """Random rollout that returns ``(discounted_reward, discounted_cost_vector)``.

    Used by constrained planners (CPOMCPOW, CPFT_DPW) at leaf expansion so
    constraint cost is accumulated across the rollout trajectory rather than
    approximated as zero. The cost vector is read from
    :meth:`ConstrainedEnvironment.constraint_cost` at every transition.

    Iterative (no recursion) so deep rollouts don't hit Python's stack limit.
    Does NOT dispatch to native rollout kernels — those compute reward only
    and have no constraint-cost channel.

    Args:
        state: Starting state.
        depth: Current depth (rollout runs from ``depth`` to ``max_depth``).
        action_sampler: Sampler for rollout actions.
        environment: Constrained POMDP env. Must expose ``constraint_cost``.
        discount_factor: Per-step discount.
        max_depth: Exclusive upper bound on rollout depth.
        n_constraints: Length of the constraint-cost vector. Used to size
            the zero return and validate per-step ``constraint_cost`` shape.

    Returns:
        ``(discounted_reward, discounted_cost_vector)``. The cost vector has
        shape ``(n_constraints,)``.

    Raises:
        ValueError: If ``constraint_cost`` returns a vector whose length
            does not match ``n_constraints``.
    """
    total_r = 0.0
    total_c = np.zeros(n_constraints, dtype=np.float64)
    discount = 1.0
    current_state = state
    current_depth = depth
    while current_depth < max_depth:
        if environment.is_terminal(state=current_state):
            break
        action = action_sampler.sample()
        next_state, _, reward = environment.sample_next_step(state=current_state, action=action)
        cost = np.asarray(
            environment.constraint_cost(current_state, action, next_state),
            dtype=np.float64,
        )
        if cost.shape != (n_constraints,):
            raise ValueError(
                f"constraint_cost returned shape {cost.shape}, expected ({n_constraints},)"
            )
        if not np.isfinite(cost).all():
            raise ValueError(f"constraint_cost returned non-finite values during rollout: {cost}")
        total_r += discount * float(reward)
        total_c += discount * cost
        discount *= discount_factor
        current_state = next_state
        current_depth += 1
    return total_r, total_c
