# SPDX-License-Identifier: MIT

"""Contract and integration tests for ``Environment.hash_observation``.

Validates the contract pair with ``is_equal_observation`` for the three envs
that drive POMCPOW's observation widening hot path
(ContinuousLightDark, LaserTag, Push), and that POMCPOW's observation
indexing actually populates the tree's ``obs_child_lookup`` for those envs.
"""

# pylint: disable=protected-access  # tests inspect internal lookup dict

import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
    ContinuousLaserTagPOMDPDiscreteActions,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import LaserTagPOMDP
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDPDiscreteActions,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler


np.random.seed(42)
random.seed(42)


class _DiscreteActionSampler(ActionSampler):
    """Trivial action sampler over a fixed list of discrete actions."""

    def __init__(self, actions):
        self._actions = list(actions)

    def sample(self, belief_node=None):
        return self._actions[int(np.random.randint(0, len(self._actions)))]

    def get_space(self):
        return self._actions


def _make_continuous_light_dark():
    return ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)


def _make_continuous_laser_tag():
    return ContinuousLaserTagPOMDPDiscreteActions(discount_factor=0.95)


def _make_laser_tag():
    return LaserTagPOMDP(discount_factor=0.95)


def _make_push():
    return PushPOMDP(discount_factor=0.95)


# ---------------------------------------------------------------------------
# Contract tests: is_equal_observation(a, b) implies hash_observation(a) ==
# hash_observation(b) for envs whose observations are non-hashable (ndarray).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_factory,observation",
    [
        (_make_continuous_light_dark, np.array([1.5, -2.0])),
        (_make_continuous_laser_tag, np.array([0.4, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7])),
        (_make_push, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
    ],
)
def test_hash_observation_self_equal(env_factory, observation):
    """hash_observation is reflexive: an observation hashes equal to itself.

    Purpose: Validates the trivial reflexive contract for ndarray observations

    Given: An environment with non-hashable ndarray observations and a sample observation
    When: hash_observation is called twice on the same observation
    Then: The two returned keys are equal and is_equal_observation also returns True

    Test type: unit
    """
    env = env_factory()
    key_a = env.hash_observation(observation)
    key_b = env.hash_observation(observation.copy())
    assert key_a == key_b
    assert env.is_equal_observation(observation, observation.copy())


@pytest.mark.parametrize(
    "env_factory,obs_a,obs_b",
    [
        (_make_continuous_light_dark, np.array([1.5, -2.0]), np.array([1.5, -2.0])),
        (
            _make_continuous_laser_tag,
            np.array([0.4, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]),
            np.array([0.4, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]),
        ),
        (
            _make_push,
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        ),
    ],
)
def test_hash_observation_equal_pair(env_factory, obs_a, obs_b):
    """Equal observations under is_equal_observation share a hash key.

    Purpose: Validates the forward direction of the hash/equality contract

    Given: An environment and two element-wise equal but distinct ndarray observations
    When: hash_observation is computed for each and is_equal_observation is invoked
    Then: hash_observation returns identical keys and is_equal_observation returns True

    Test type: unit
    """
    env = env_factory()
    assert env.is_equal_observation(obs_a, obs_b)
    assert env.hash_observation(obs_a) == env.hash_observation(obs_b)


@pytest.mark.parametrize(
    "env_factory,obs_a,obs_b",
    [
        (_make_continuous_light_dark, np.array([1.5, -2.0]), np.array([1.5, 2.0])),
        (
            _make_continuous_laser_tag,
            np.array([0.4, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]),
            np.array([0.5, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]),
        ),
        (
            _make_push,
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0]),
        ),
    ],
)
def test_hash_observation_unequal_pair(env_factory, obs_a, obs_b):
    """Unequal observations produce distinct hash keys.

    Purpose: Validates that the indexed lookup distinguishes different observations

    Given: An environment and two distinct ndarray observations
    When: hash_observation is computed for each
    Then: The keys differ and is_equal_observation also returns False

    Test type: unit
    """
    env = env_factory()
    assert not env.is_equal_observation(obs_a, obs_b)
    assert env.hash_observation(obs_a) != env.hash_observation(obs_b)


def test_laser_tag_default_hash_uses_tuple_observations():
    """LaserTagPOMDP keeps the default hash because its observations are tuples.

    Purpose: Validates that hashable-observation envs work with the default override

    Given: LaserTagPOMDP whose sample_observation returns tuples (already hashable)
    When: hash_observation is called on a representative tuple observation
    Then: The returned key equals the tuple and the env did NOT override the default

    Test type: unit
    """
    env = _make_laser_tag()
    sample_tuple = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    assert env.hash_observation(sample_tuple) == sample_tuple
    # Confirm the LaserTagPOMDP class itself does not define hash_observation;
    # i.e. it inherits the default from Environment.
    assert "hash_observation" not in LaserTagPOMDP.__dict__


# ---------------------------------------------------------------------------
# POMCPOW integration: indexed lookup populates obs_child_lookup so the
# linear scan is short-circuited for the three benchmarked envs.
# ---------------------------------------------------------------------------


def _planner_for(env, action_sampler):
    return POMCPOW(
        environment=env,
        discount_factor=env.discount_factor,
        depth=10,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        n_simulations=20,
        action_sampler=action_sampler,
        name="TestPOMCPOWHashObs",
    )


@pytest.mark.parametrize(
    "env_factory",
    [_make_continuous_light_dark, _make_push],
)
def test_pomcpow_observation_widening_populates_obs_child_lookup(env_factory, monkeypatch):
    """POMCPOW's _observation_widening routes through the indexed lookup.

    Purpose: Confirms hash_observation flows through POMCPOW into
        tree.obs_child_lookup, so the linear-scan fallback is short-circuited
        for envs with ndarray observations.

    Given: An env with ndarray observations (LightDark or Push) that overrides
        hash_observation, and a POMCPOW planner running ``_observation_widening``
        on a small hand-built tree.
    When: ``_observation_widening`` is invoked twice with the same observation.
    Then: After the first call ``obs_child_lookup`` has been populated; the
        second call hits the indexed-lookup fast path (no linear scan via
        ``get_belief_child``) and bumps the existing child's weight rather
        than allocating a new one.

    Test type: integration
    """
    np.random.seed(0)
    random.seed(0)
    env = env_factory()
    actions = env.get_actions()
    sampler = _DiscreteActionSampler(actions)
    planner = _planner_for(env, sampler)

    from POMDPPlanners.core.tree.arena import Tree  # pylint: disable=import-outside-toplevel

    # Hand-build a minimal tree: root belief -> one action node.
    belief = get_initial_belief(pomdp=env, n_particles=8, resampling=True)
    tree = Tree()
    root_id = tree.add_belief_node(belief=belief)
    tree.visit_count[root_id] += 1
    action_id = tree.add_action_node(action=actions[0], parent_id=root_id)
    tree.visit_count[action_id] += 1

    state = belief.sample()
    next_state = env.sample_next_state(state=state, action=actions[0])
    observation = env.sample_observation(next_state=next_state, action=actions[0])

    # First call: should add a new belief child and register it in the lookup.
    first_id = planner._observation_widening(  # pyright: ignore[reportPrivateUsage]
        tree=tree, action_id=action_id, observation=observation
    )
    assert tree.obs_child_lookup, (
        "_observation_widening must register the new ndarray-keyed child via "
        "hash_observation in obs_child_lookup"
    )
    weight_after_first = tree.weight[first_id]

    # Patch get_belief_child to fail loudly; the second call MUST hit the
    # indexed fast path and never fall through to the linear scan.
    def _boom(*_args, **_kwargs):
        raise AssertionError("get_belief_child should not be called when indexed lookup hits")

    monkeypatch.setattr(tree, "get_belief_child", _boom)

    second_id = planner._observation_widening(  # pyright: ignore[reportPrivateUsage]
        tree=tree, action_id=action_id, observation=observation
    )
    assert second_id == first_id, "indexed lookup should resolve to the existing child"
    assert tree.weight[second_id] == pytest.approx(weight_after_first + 1.0)
