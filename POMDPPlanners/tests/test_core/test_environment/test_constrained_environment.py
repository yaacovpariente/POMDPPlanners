# SPDX-License-Identifier: MIT

"""Tests for the ``ConstrainedEnvironment`` ABC.

Validates that the abstract contract is enforced (cannot instantiate without
``constraint_cost``), the scalar/vector return shapes are honoured, and the
default batched helper composes correctly with single-call ``constraint_cost``.
"""

import random
from typing import Any, List, Sequence, Union

import numpy as np
import pytest

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import (
    ConstrainedEnvironment,
    SpaceInfo,
    SpaceType,
)


np.random.seed(42)
random.seed(42)


class _MockDistribution(Distribution):
    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        return [np.array([0.0])] * n_samples

    def probability(self, values: List[np.ndarray]) -> np.ndarray:
        return np.ones(len(values))


class _ScalarConstrainedMock(ConstrainedEnvironment):
    """Minimal concrete subclass with a scalar (K=1) constraint cost.

    Constraint cost is the indicator ``state[0] > 0``: useful for testing
    binary failure-indicator semantics.
    """

    def __init__(self, discount_factor: float = 0.95):
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )
        super().__init__(
            discount_factor=discount_factor,
            name="ScalarConstrainedMock",
            space_info=space_info,
        )

    def sample_next_state(self, state, action, n_samples: int = 1):
        if n_samples == 1:
            return np.array([0.0])
        return [np.array([0.0])] * n_samples

    def sample_observation(self, next_state, action, n_samples: int = 1):
        if n_samples == 1:
            return np.array([0.0])
        return [np.array([0.0])] * n_samples

    def transition_log_probability(self, state, action, next_states) -> np.ndarray:
        return np.zeros(len(next_states))

    def observation_log_probability(self, next_state, action, observations) -> np.ndarray:
        return np.zeros(len(observations))

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        return 0.0

    def is_terminal(self, state: Any) -> bool:
        return False

    def initial_state_dist(self) -> Distribution:
        return _MockDistribution()

    def initial_observation_dist(self) -> Distribution:
        return _MockDistribution()

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return np.array_equal(observation1, observation2)

    def hash_action(self, action: Any):
        return action

    def constraint_cost(self, state: Any, action: Any, next_state: Any) -> np.ndarray:
        return np.array([1.0 if state[0] > 0.0 else 0.0])


class _VectorConstrainedMock(_ScalarConstrainedMock):
    """Concrete subclass with K=3 constraint dimensions.

    Returns ``[state[0], abs(action), next_state[0]]`` so each dimension is
    distinguishable and shape checks are meaningful.
    """

    def constraint_cost(self, state: Any, action: Any, next_state: Any) -> np.ndarray:
        return np.array([float(state[0]), abs(float(action)), float(next_state[0])])


class _OverriddenBatchMock(_ScalarConstrainedMock):
    """Subclass that overrides ``constraint_cost_batch`` with a marker value.

    Used to confirm the override path is taken instead of the default loop.
    """

    def constraint_cost_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: Any,
        next_states: Union[np.ndarray, Sequence[Any]],
    ) -> np.ndarray:
        return np.full((len(states), 1), -99.0)


def test_cannot_instantiate_abstract_constrained_environment():
    """Direct instantiation of the ABC must fail.

    Purpose: Verify the ABC enforces its abstract contract.

    Given: The ``ConstrainedEnvironment`` ABC.
    When: An attempt is made to instantiate it directly.
    Then: A ``TypeError`` is raised mentioning the abstract method
        ``constraint_cost``.

    Test type: unit
    """
    with pytest.raises(TypeError, match="constraint_cost"):
        ConstrainedEnvironment(  # type: ignore[abstract]
            discount_factor=0.95,
            name="bad",
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE,
                observation_space=SpaceType.DISCRETE,
            ),
        )


def test_concrete_scalar_subclass_returns_length_one_vector():
    """Scalar-constraint subclass returns a length-1 array, not a Python float.

    Purpose: Verify the vector-only return convention.

    Given: A concrete subclass whose constraint is a scalar failure
        indicator.
    When: ``constraint_cost(state, action, next_state)`` is called.
    Then: The return is a 1-D ``np.ndarray`` of shape ``(1,)`` whose value
        is the indicator.

    Test type: unit
    """
    env = _ScalarConstrainedMock()
    result = env.constraint_cost(np.array([1.0]), 0, np.array([0.0]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert result[0] == 1.0
    safe = env.constraint_cost(np.array([-1.0]), 0, np.array([0.0]))
    assert safe[0] == 0.0


def test_concrete_vector_subclass_returns_correct_shape():
    """Multi-constraint subclass returns a vector with the expected length.

    Purpose: Verify the interface is vector-by-default (no scalar special case).

    Given: A concrete subclass declaring 3 constraint dimensions.
    When: ``constraint_cost`` is called with a state, action, next-state.
    Then: A 1-D ``np.ndarray`` of shape ``(3,)`` with the expected entries
        is returned.

    Test type: unit
    """
    env = _VectorConstrainedMock()
    result = env.constraint_cost(np.array([2.0]), -3.0, np.array([5.0]))
    assert result.shape == (3,)
    np.testing.assert_array_equal(result, np.array([2.0, 3.0, 5.0]))


def test_subclass_missing_constraint_cost_cannot_be_instantiated():
    """A subclass that does not implement ``constraint_cost`` must remain abstract.

    Purpose: Verify the ABC machinery propagates through subclasses.

    Given: A subclass that overrides every other abstract method except
        ``constraint_cost``.
    When: The subclass is instantiated.
    Then: ``TypeError`` is raised mentioning ``constraint_cost``.

    Test type: unit
    """

    class _IncompleteSubclass(_ScalarConstrainedMock):
        constraint_cost = ConstrainedEnvironment.constraint_cost  # re-mark as abstract

    with pytest.raises(TypeError, match="constraint_cost"):
        _IncompleteSubclass()  # type: ignore[abstract]


def test_constraint_cost_batch_default_loops_over_constraint_cost():
    """Default batched helper matches per-sample ``constraint_cost`` calls.

    Purpose: Verify the loop-based default produces shape ``(N, K)`` and
        matches the single-call values element-wise.

    Given: A vector-constraint subclass and 4 sample transitions.
    When: ``constraint_cost_batch`` is called.
    Then: The result has shape ``(4, 3)`` and each row equals the single-call
        ``constraint_cost`` for that transition.

    Test type: unit
    """
    env = _VectorConstrainedMock()
    states = [np.array([float(i)]) for i in range(4)]
    next_states = [np.array([float(i + 10)]) for i in range(4)]
    action = 2.0
    batch = env.constraint_cost_batch(states, action, next_states)
    assert batch.shape == (4, 3)
    for i in range(4):
        np.testing.assert_array_equal(
            batch[i], env.constraint_cost(states[i], action, next_states[i])
        )


def test_constraint_cost_batch_override_is_used():
    """Subclass override of ``constraint_cost_batch`` is preferred over the default.

    Purpose: Confirm the override path (e.g. for native / vectorized
        kernels) is reachable via the public interface.

    Given: A subclass whose ``constraint_cost_batch`` returns a constant
        marker (``-99.0``) regardless of input.
    When: The batched helper is called.
    Then: The marker value is returned (override path taken), not the
        per-element loop result.

    Test type: unit
    """
    env = _OverriddenBatchMock()
    states = [np.array([1.0]), np.array([2.0])]
    next_states = [np.array([0.0]), np.array([0.0])]
    batch = env.constraint_cost_batch(states, 0, next_states)
    assert batch.shape == (2, 1)
    assert (batch == -99.0).all()
