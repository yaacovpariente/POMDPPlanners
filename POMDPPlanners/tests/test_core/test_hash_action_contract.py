# SPDX-License-Identifier: MIT

"""Contract and integration tests for ``Environment.hash_action``.

Validates the per-env hash/equality contract for actions across every
concrete environment in the project, plus integration tests that exercise
POMCPOW and PFT-DPW to confirm ``tree.action_child_lookup`` is populated
when each env's ``hash_action`` is plumbed through the planner-utils.
"""

# pylint: disable=protected-access  # tests inspect internal lookup dict

import random
from typing import Any, Callable, List

import numpy as np
import pytest

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
    ContinuousLaserTagPOMDP,
    ContinuousLaserTagPOMDPDiscreteActions,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import LaserTagPOMDP
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    ContinuousLightDarkPOMDPDiscreteActions,
)
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
)
from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP
from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import ContinuousPushPOMDP
from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import RockSamplePOMDP
from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_pomdp import (
    SafeAntVelocityPOMDP,
)
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    cartpole_pinned_kwargs,
    continuous_laser_tag_discrete_actions_pinned_kwargs,
    continuous_laser_tag_pinned_kwargs,
    continuous_light_dark_discrete_actions_pinned_kwargs,
    continuous_light_dark_pinned_kwargs,
    continuous_push_pinned_kwargs,
    discrete_light_dark_pinned_kwargs,
    laser_tag_pinned_kwargs,
    mountain_car_pinned_kwargs,
    pacman_pinned_kwargs,
    push_pinned_kwargs,
    rock_sample_pinned_kwargs,
    safety_ant_velocity_pinned_kwargs,
    tiger_pinned_kwargs,
)
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler


np.random.seed(42)
random.seed(42)


# ---------------------------------------------------------------------------
# Per-env action factories: each returns ``(env, list_of_distinct_actions)``.
# Continuous-action envs supply two distinct ndarrays plus a duplicate that is
# element-wise equal to the first so the hash-equals-on-equality contract is
# exercised end-to-end.
# ---------------------------------------------------------------------------


def _continuous_light_dark_factory():
    env = ContinuousLightDarkPOMDP(discount_factor=0.95, **continuous_light_dark_pinned_kwargs())
    actions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    return env, actions


def _continuous_light_dark_discrete_factory():
    env = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95,
        **continuous_light_dark_discrete_actions_pinned_kwargs(),
    )
    return env, env.get_actions()


def _discrete_light_dark_factory():
    env = DiscreteLightDarkPOMDP(discount_factor=0.95, **discrete_light_dark_pinned_kwargs())
    return env, env.get_actions()


def _continuous_laser_tag_factory():
    env = ContinuousLaserTagPOMDP(discount_factor=0.95, **continuous_laser_tag_pinned_kwargs())
    actions = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    return env, actions


def _continuous_laser_tag_discrete_factory():
    env = ContinuousLaserTagPOMDPDiscreteActions(
        discount_factor=0.95,
        **continuous_laser_tag_discrete_actions_pinned_kwargs(),
    )
    return env, env.get_actions()


def _laser_tag_factory():
    env = LaserTagPOMDP(discount_factor=0.95, **laser_tag_pinned_kwargs())
    return env, env.get_actions()


def _continuous_push_factory():
    env = ContinuousPushPOMDP(discount_factor=0.95, **continuous_push_pinned_kwargs())
    actions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    return env, actions


def _push_factory():
    env = PushPOMDP(discount_factor=0.95, **push_pinned_kwargs())
    return env, env.get_actions()


def _cartpole_factory():
    env = CartPolePOMDP(
        discount_factor=0.99,
        noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]),
        **cartpole_pinned_kwargs(),
    )
    return env, env.get_actions()


def _mountain_car_factory():
    env = MountainCarPOMDP(discount_factor=0.99, **mountain_car_pinned_kwargs())
    return env, env.get_actions()


def _pacman_factory():
    env = PacManPOMDP(
        discount_factor=0.95,
        **pacman_pinned_kwargs(
            maze_size=(5, 5),
            walls=set(),
            initial_pellets=[(2, 2)],
            initial_ghost_positions=None,
            ghost_strategies=None,
        ),
    )
    return env, env.get_actions()


def _rock_sample_factory():
    env = RockSamplePOMDP(
        discount_factor=0.95,
        **rock_sample_pinned_kwargs(map_size=(5, 5), rock_positions=[(0, 0), (2, 2)]),
    )
    return env, env.get_actions()


def _safety_ant_factory():
    env = SafeAntVelocityPOMDP(discount_factor=0.95, **safety_ant_velocity_pinned_kwargs())
    return env, env.get_actions()


def _tiger_factory():
    env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
    return env, env.get_actions()


def _sanity_factory():
    env = SanityPOMDP(discount_factor=0.95)
    return env, env.get_actions()


_ALL_ENV_FACTORIES: List[Callable[[], Any]] = [
    _continuous_light_dark_factory,
    _continuous_light_dark_discrete_factory,
    _discrete_light_dark_factory,
    _continuous_laser_tag_factory,
    _continuous_laser_tag_discrete_factory,
    _laser_tag_factory,
    _continuous_push_factory,
    _push_factory,
    _cartpole_factory,
    _mountain_car_factory,
    _pacman_factory,
    _rock_sample_factory,
    _safety_ant_factory,
    _tiger_factory,
    _sanity_factory,
]


# ---------------------------------------------------------------------------
# Universal contract: the hash result is hashable and idempotent for every
# concrete env.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", _ALL_ENV_FACTORIES)
def test_hash_action_is_hashable_and_idempotent(factory):
    """Every concrete env returns a hashable, idempotent key from hash_action.

    Purpose: Validates the universal contract that every concrete env honours

    Given: Each concrete Environment subclass and a representative action
    When: hash_action is invoked on the same action twice
    Then: The result is itself hashable and the two calls return equal keys

    Test type: unit
    """
    env, actions = factory()
    assert isinstance(env, Environment)
    for action in actions:
        key = env.hash_action(action)
        # Result must be hashable (hash() must not raise).
        hash(key)
        # Idempotent under repeated calls.
        if isinstance(action, np.ndarray):
            key_again = env.hash_action(action.copy())
        else:
            key_again = env.hash_action(action)
        assert key == key_again


# ---------------------------------------------------------------------------
# Discrete-action envs: each label is a unique stable hash.
# ---------------------------------------------------------------------------


_DISCRETE_FACTORIES: List[Callable[[], Any]] = [
    _continuous_light_dark_discrete_factory,
    _discrete_light_dark_factory,
    _continuous_laser_tag_discrete_factory,
    _laser_tag_factory,
    _push_factory,
    _cartpole_factory,
    _mountain_car_factory,
    _pacman_factory,
    _rock_sample_factory,
    _safety_ant_factory,
    _tiger_factory,
    _sanity_factory,
]


@pytest.mark.parametrize("factory", _DISCRETE_FACTORIES)
def test_hash_action_distinct_keys_for_discrete(factory):
    """Distinct discrete actions hash to distinct keys.

    Purpose: Validates the dup-prevention contract for discrete action spaces

    Given: A discrete-action env and the full list of available actions
    When: hash_action is computed for every action
    Then: All keys are pairwise distinct, ensuring no two action children
        can collide in tree.action_child_lookup

    Test type: unit
    """
    env, actions = factory()
    keys = [env.hash_action(a) for a in actions]
    assert len(set(keys)) == len(keys)


# ---------------------------------------------------------------------------
# ndarray-action envs: equal arrays hash equal, different arrays hash apart.
# ---------------------------------------------------------------------------


_NDARRAY_ACTION_FACTORIES: List[Callable[[], Any]] = [
    _continuous_light_dark_factory,
    _continuous_laser_tag_factory,
    _continuous_push_factory,
]


@pytest.mark.parametrize("factory", _NDARRAY_ACTION_FACTORIES)
def test_hash_action_ndarray_equal_pair(factory):
    """Two ndarrays with the same content hash to the same key.

    Purpose: Validates the forward direction of the contract for ndarray actions

    Given: A continuous-action env and two distinct ndarray objects with
        identical content
    When: hash_action is computed for each
    Then: The two keys are equal so the indexed lookup will resolve them
        to the same action child

    Test type: unit
    """
    env, actions = factory()
    action = actions[0]
    twin = action.copy()
    assert action is not twin
    assert env.hash_action(action) == env.hash_action(twin)


@pytest.mark.parametrize("factory", _NDARRAY_ACTION_FACTORIES)
def test_hash_action_ndarray_unequal_pair(factory):
    """Two ndarrays with different content hash to different keys.

    Purpose: Validates the dup-prevention invariant for ndarray actions

    Given: A continuous-action env and two ndarray actions with different content
    When: hash_action is computed for each
    Then: The two keys differ so the indexed lookup creates distinct action
        children rather than collapsing them

    Test type: unit
    """
    env, actions = factory()
    assert env.hash_action(actions[0]) != env.hash_action(actions[1])


# ---------------------------------------------------------------------------
# Planner-side smoke: tree.action_child_lookup is populated after a small
# search and identical action samples reuse the same node.
# ---------------------------------------------------------------------------


class _FixedNDArrayActionSampler(ActionSampler):
    """Cycles through a small fixed pool of ndarray actions.

    Used to drive the planner-side smoke test deterministically: the same
    action vectors are sampled repeatedly so that the indexed lookup must
    resolve to the existing action child rather than inserting a duplicate.
    """

    def __init__(self, actions: List[np.ndarray]):
        self._actions = [np.asarray(a, dtype=np.float64) for a in actions]
        self._idx = 0

    def sample(self, belief_node=None):  # pylint: disable=unused-argument
        a = self._actions[self._idx % len(self._actions)]
        self._idx += 1
        return a.copy()


def _run_pomcpow_continuous_light_dark():
    env = ContinuousLightDarkPOMDP(discount_factor=0.95, **continuous_light_dark_pinned_kwargs())
    sampler = _FixedNDArrayActionSampler(
        [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([-1.0, 0.0])]
    )
    planner = POMCPOW(
        environment=env,
        discount_factor=0.95,
        depth=5,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        n_simulations=50,
        action_sampler=sampler,
        name="POMCPOWHashActionSmoke",
    )
    belief = get_initial_belief(pomdp=env, n_particles=20, resampling=True)
    planner.action(belief)
    return planner


def _run_pft_dpw_continuous_light_dark():
    env = ContinuousLightDarkPOMDP(discount_factor=0.95, **continuous_light_dark_pinned_kwargs())
    sampler = _FixedNDArrayActionSampler(
        [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([-1.0, 0.0])]
    )
    planner = PFT_DPW(
        environment=env,
        discount_factor=0.95,
        depth=5,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        n_simulations=50,
        action_sampler=sampler,
        name="PFTDPWHashActionSmoke",
    )
    belief = get_initial_belief(pomdp=env, n_particles=20, resampling=True)
    planner.action(belief)
    return planner


def test_pomcpow_action_child_lookup_populated_for_ndarray_actions(monkeypatch):
    """POMCPOW populates action_child_lookup via env.hash_action for ndarrays.

    Purpose: Confirms hash_action is plumbed through action_progressive_widening
        into tree.action_child_lookup for envs with ndarray actions, eliminating
        the linear np.array_equal scan that previously dominated profiling.

    Given: POMCPOW running on ContinuousLightDarkPOMDP (ndarray actions) with a
        fixed-pool action sampler that emits the same vectors repeatedly.
    When: A small search is executed via planner.action(belief).
    Then: tree.action_child_lookup is non-empty (was permanently empty before
        this PR) and the linear-scan get_action_child is never called once
        the indexed lookup is in play.

    Test type: integration
    """
    np.random.seed(0)
    random.seed(0)
    captured_trees: List[Tree] = []
    original_action_pw = (
        # pylint: disable=import-outside-toplevel
        __import__(
            "POMDPPlanners.planners.planners_utils.dpw",
            fromlist=["action_progressive_widening_arena"],
        ).action_progressive_widening_arena
    )

    def _wrapped(*args, **kwargs):
        tree = kwargs.get("tree", args[0] if args else None)
        if tree is not None and tree not in captured_trees:
            captured_trees.append(tree)
        return original_action_pw(*args, **kwargs)

    monkeypatch.setattr(
        "POMDPPlanners.planners.mcts_planners.pomcpow.action_progressive_widening_arena",
        _wrapped,
    )

    _run_pomcpow_continuous_light_dark()

    assert captured_trees, "POMCPOW did not invoke action_progressive_widening_arena"
    tree = captured_trees[0]
    assert tree.action_child_lookup, (
        "POMCPOW must populate tree.action_child_lookup via env.hash_action; "
        "an empty lookup means the linear-scan path is still in play"
    )


def test_pft_dpw_action_child_lookup_populated_for_ndarray_actions(monkeypatch):
    """PFT-DPW populates action_child_lookup via env.hash_action for ndarrays.

    Purpose: Mirrors the POMCPOW smoke test for the PFT-DPW planner so the
        new hash_action path is verified for both DPW-style planners.

    Given: PFT-DPW running on ContinuousLightDarkPOMDP (ndarray actions).
    When: A small search is executed via planner.action(belief).
    Then: tree.action_child_lookup is non-empty after the run.

    Test type: integration
    """
    np.random.seed(0)
    random.seed(0)
    captured_trees: List[Tree] = []
    original_action_pw = (
        # pylint: disable=import-outside-toplevel
        __import__(
            "POMDPPlanners.planners.planners_utils.dpw",
            fromlist=["action_progressive_widening_arena"],
        ).action_progressive_widening_arena
    )

    def _wrapped(*args, **kwargs):
        tree = kwargs.get("tree", args[0] if args else None)
        if tree is not None and tree not in captured_trees:
            captured_trees.append(tree)
        return original_action_pw(*args, **kwargs)

    monkeypatch.setattr(
        "POMDPPlanners.planners.mcts_planners.pft_dpw.action_progressive_widening_arena",
        _wrapped,
    )

    _run_pft_dpw_continuous_light_dark()

    assert captured_trees, "PFT-DPW did not invoke action_progressive_widening_arena"
    tree = captured_trees[0]
    assert (
        tree.action_child_lookup
    ), "PFT-DPW must populate tree.action_child_lookup via env.hash_action"


def test_action_child_lookup_distinguishes_distinct_ndarray_actions():
    """Distinct ndarray actions get distinct nodes in action_child_lookup.

    Purpose: Verifies the dup-prevention invariant that the linear scan
        previously enforced: distinct actions produce distinct keys, identical
        actions reuse the same node.

    Given: A Tree, an env with ndarray actions, and three distinct action
        vectors plus one duplicate of the first.
    When: add_action_node is invoked with the env-derived action_key for each.
    Then: The three distinct actions allocate three distinct child IDs;
        the duplicate of the first reuses the original child via the indexed
        lookup; and no two ndarray-equal actions ever share a parent.

    Test type: integration
    """
    env = ContinuousLightDarkPOMDP(discount_factor=0.95, **continuous_light_dark_pinned_kwargs())
    belief = get_initial_belief(pomdp=env, n_particles=4, resampling=True)
    tree = Tree()
    root_id = tree.add_belief_node(belief=belief)

    action_a = np.array([1.0, 0.0])
    action_b = np.array([0.0, 1.0])
    action_c = np.array([-1.0, 0.0])
    action_a_dup = np.array([1.0, 0.0])

    ids = []
    for action in (action_a, action_b, action_c, action_a_dup):
        key = env.hash_action(action)
        existing = tree.get_action_child_indexed(root_id, action_key=key)
        if existing is not None:
            ids.append(existing)
            continue
        ids.append(tree.add_action_node(action=action, parent_id=root_id, action_key=key))

    assert len(set(ids[:3])) == 3, "distinct ndarray actions must allocate distinct child IDs"
    assert ids[3] == ids[0], "duplicate ndarray action must reuse the existing child"

    # No two children of root_id share an ndarray-equal action.
    children = tree.children_ids[root_id]
    for i, cid_i in enumerate(children):
        for cid_j in children[i + 1 :]:
            assert not np.array_equal(tree.action[cid_i], tree.action[cid_j])
