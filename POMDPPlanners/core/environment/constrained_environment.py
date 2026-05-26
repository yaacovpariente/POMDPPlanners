# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Abstract base class for constrained POMDP environments.

Defines :class:`ConstrainedEnvironment`, a subtype of
:class:`POMDPPlanners.core.environment.Environment` that adds a
**vector-valued constraint cost channel** alongside the standard reward
channel. Constrained planners (CPOMCPOW, C-PFT-DPW, future C-POMCP) read
this method to compute Lagrangian Q-values and dual-ascent updates.

Why a separate name from ``reward``: the word "cost" is overloaded in this
repo. :mod:`POMDPPlanners.core.cost` defines ``belief_expectation_cost`` as
the sign-flipped reward used by ICVaR-style planners. The constraint cost
in a constrained POMDP is a *different* quantity — an inequality-constraint
metric that is independent of reward. Naming the new method
``constraint_cost`` keeps the two unambiguous.

Why vector-valued: matches the standard constrained-POMDP definition
(``costs(s, a, s') -> R^K``) so multi-constraint problems are first-class
without an interface change. Scalar constraints return a length-1 array.

Standard constrained-POMDP definition: the CPOMDP tuple augments the
POMDP with a vector-valued cost function ``C(s, a, s') -> R^K`` and a
cost-budget vector; an optimal policy maximises expected discounted
reward subject to a budget on each dimension of the expected discounted
cost. This ABC exposes the cost-vector channel on the env; the budget
lives on the constrained planner.

Classes:
    ConstrainedEnvironment: ABC extending :class:`Environment` with one
        abstract method ``constraint_cost`` and one default-implemented
        batched helper ``constraint_cost_batch``.
"""

from abc import abstractmethod
from typing import Any, Sequence, Union

import numpy as np

from POMDPPlanners.core.environment.environment import Environment


class ConstrainedEnvironment(Environment):
    """Environment with a vector-valued constraint-cost channel.

    Subclasses implement :meth:`constraint_cost` to expose the per-transition
    cost vector that constrained planners (CPOMCPOW, C-PFT-DPW) read to
    compute Lagrangian Q-values and dual-ascent updates. Everything else
    on :class:`Environment` — reward, transitions, observations, terminal
    handling — is inherited unchanged.

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    @abstractmethod
    def constraint_cost(self, state: Any, action: Any, next_state: Any) -> np.ndarray:
        """Compute the per-transition constraint cost vector.

        Args:
            state: Current state.
            action: Action executed from ``state``.
            next_state: Realised post-transition state.

        Returns:
            1-D array of shape ``(K,)`` where ``K`` is the number of
            constraint dimensions. For scalar constraints return a length-1
            array; for chance-constrained planning with a binary failure
            indicator return ``np.array([1.0])`` on unsafe transitions and
            ``np.array([0.0])`` otherwise.

        Note:
            Subclasses must implement this method.
        """

    def constraint_cost_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: Any,
        next_states: Union[np.ndarray, Sequence[Any]],
    ) -> np.ndarray:
        """Batched constraint cost for ``N`` transitions sharing one action.

        Provides a loop-based default that subclasses with native /
        vectorized kernels can override.

        Args:
            states: Sequence of ``N`` current states.
            action: Action executed from each state.
            next_states: Sequence of ``N`` realised next states.

        Returns:
            2-D array of shape ``(N, K)`` where ``K`` is the number of
            constraint dimensions.
        """
        return np.stack(
            [self.constraint_cost(states[i], action, next_states[i]) for i in range(len(states))]
        )
