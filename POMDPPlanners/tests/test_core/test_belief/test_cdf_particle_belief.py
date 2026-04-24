"""Tests for the CDF-backed particle belief in core.belief.cdf_particle_belief."""

import random as _stdlib_random
from collections import Counter
from unittest.mock import Mock

import numpy as np
import pytest

from POMDPPlanners.core.belief import CDFParticleBelief
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import (
    Environment,
    ObservationModel,
    SpaceInfo,
    SpaceType,
    StateTransitionModel,
)

np.random.seed(42)
_stdlib_random.seed(42)


class _MockEnv(Environment):
    """Tiny env with a deterministic observation model that returns weight 0.5."""

    def __init__(self) -> None:
        super().__init__(
            discount_factor=0.95,
            name="MockEnv",
            space_info=SpaceInfo(SpaceType.DISCRETE, SpaceType.DISCRETE),
        )

    def state_transition(self, state, action):
        del action
        return state

    def observation_model(self, next_state, action):
        del next_state, action
        mock = Mock(spec=ObservationModel)
        mock.probability.return_value = [0.5]
        return mock

    def is_equal_observation(self, observation1, observation2):
        return observation1 == observation2

    def is_terminal(self, state):
        del state
        return False

    def reward(self, state, action):
        del state, action
        return 0.0

    def initial_state_dist(self):
        return DiscreteDistribution(values=[0], probs=np.array([1.0]))

    def initial_observation_dist(self):
        return DiscreteDistribution(values=["o"], probs=np.array([1.0]))

    def state_transition_model(self, state, action):
        del state, action
        return Mock(spec=StateTransitionModel)


def test_empty_belief_construction():
    """Default-constructed belief has no particles and an empty CDF.

    Purpose: Validate the empty-init invariants.

    Test type: unit
    """
    b = CDFParticleBelief()
    assert not b.particles
    assert not b.cdf


def test_constructor_initialises_cdf_from_weights():
    """Constructor with non-empty weights builds the running-sum CDF.

    Test type: unit
    """
    b = CDFParticleBelief(particles=[1, 2, 3], weights=[0.5, 1.5, 2.0])
    assert b.particles == [1, 2, 3]
    assert b.cdf == [0.5, 2.0, 4.0]


def test_constructor_rejects_mismatched_lengths():
    """particles and weights must have the same length.

    Test type: unit
    """
    with pytest.raises(ValueError):
        CDFParticleBelief(particles=[1, 2], weights=[1.0])


def test_push_weighted_extends_both_lists():
    """push_weighted appends the particle and extends the CDF by the weight.

    Test type: unit
    """
    b = CDFParticleBelief()
    b.push_weighted("a", 0.7)
    b.push_weighted("b", 1.3)
    assert b.particles == ["a", "b"]
    assert b.cdf == [0.7, 2.0]


def test_inplace_update_uses_observation_model_weight():
    """inplace_update appends the state with weight = P(obs | state, action).

    Test type: unit
    """
    env = _MockEnv()
    b = CDFParticleBelief()
    b.inplace_update(action="listen", observation="hear", pomdp=env, state=42)
    assert b.particles == [42]
    # _MockEnv returns probability 0.5
    assert b.cdf == [0.5]


def test_inplace_update_rejects_none_state():
    """state=None raises ValueError.

    Test type: unit
    """
    env = _MockEnv()
    b = CDFParticleBelief()
    with pytest.raises(ValueError):
        b.inplace_update(action="listen", observation="hear", pomdp=env, state=None)


def test_sample_distribution_matches_weights():
    """sample() draws particles in proportion to their weights.

    Test type: unit
    """
    b = CDFParticleBelief(
        particles=["light", "heavy"],
        weights=[1.0, 3.0],
    )
    counts = Counter(b.sample() for _ in range(4000))
    ratio = counts["heavy"] / counts["light"]
    assert 2.5 < ratio < 3.5  # expected 3.0 ± noise


def test_sample_empty_raises():
    """Sampling an empty belief raises ValueError.

    Test type: unit
    """
    b = CDFParticleBelief()
    with pytest.raises(ValueError):
        b.sample()


def test_sample_zero_total_weight_raises():
    """Sampling when every weight is zero raises.

    Test type: unit
    """
    b = CDFParticleBelief(particles=[1, 2], weights=[0.0, 0.0])
    with pytest.raises(ValueError):
        b.sample()


def test_update_returns_independent_belief():
    """update() returns a new CDFParticleBelief without mutating the original.

    Test type: unit
    """
    env = _MockEnv()
    original = CDFParticleBelief(particles=[1, 2], weights=[0.5, 0.5])
    new = original.update(action="listen", observation="hear", pomdp=env, state=99)
    assert original.particles == [1, 2]
    assert original.cdf == [0.5, 1.0]
    assert new.particles == [1, 2, 99]
    # _MockEnv returns probability 0.5; new cdf is [0.5, 1.0, 1.5]
    assert new.cdf == [0.5, 1.0, 1.5]
    assert new is not original
