# SPDX-License-Identifier: MIT

"""Cross-environment conformance tests for the Environment API.

Covers two contracts that planners and beliefs rely on but that, prior
to this file, were tested for at most a handful of environments:

* ``hash_action(a)`` — must return a hashable key, agree on equal actions,
  and be pairwise distinct across the discrete action set when the env
  is a :class:`DiscreteActionsEnvironment`.
* ``hash_observation(o)`` — must return a hashable key and agree on two
  observations that ``is_equal_observation`` considers equal.

The conformance tests are parametrized over every concrete environment
class so that a new env wired into :data:`ENV_BUILDERS` is automatically
checked.

Environments whose ``hash_observation`` override is missing (so the base
class's default raises on ndarray observations) are marked ``xfail``
with ``strict=True`` so the contract gap is documented and the suite
turns green automatically the moment the override lands.
"""

from copy import deepcopy
from typing import Any, Callable, List, Tuple

import numpy as np
import pytest

from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    Environment,
    SpaceType,
)
from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
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
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.environments.pacman_pomdp import PacManPOMDP
from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import (
    ContinuousPushPOMDP,
    ContinuousPushPOMDPDiscreteActions,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import RockSamplePOMDP
from POMDPPlanners.environments.safety_ant_velocity_pomdp import SafeAntVelocityPOMDP
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    cartpole_pinned_kwargs,
    continuous_laser_tag_discrete_actions_pinned_kwargs,
    continuous_laser_tag_pinned_kwargs,
    continuous_light_dark_discrete_actions_pinned_kwargs,
    continuous_light_dark_pinned_kwargs,
    continuous_push_discrete_actions_pinned_kwargs,
    continuous_push_pinned_kwargs,
    discrete_light_dark_pinned_kwargs,
    laser_tag_pinned_kwargs,
    mountain_car_pinned_kwargs,
    pacman_pinned_kwargs,
    push_pinned_kwargs,
    rock_sample_pinned_kwargs,
    safety_ant_velocity_pinned_kwargs,
    sanity_pinned_kwargs,
    tiger_pinned_kwargs,
)


EnvBuilder = Callable[[], Environment]


def _build_tiger() -> TigerPOMDP:
    return TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())


def _build_sanity() -> SanityPOMDP:
    return SanityPOMDP(discount_factor=0.95, **sanity_pinned_kwargs())


def _build_cartpole() -> CartPolePOMDP:
    return CartPolePOMDP(
        discount_factor=0.95, noise_cov=np.eye(4) * 0.1, **cartpole_pinned_kwargs()
    )


def _build_mountain_car() -> MountainCarPOMDP:
    return MountainCarPOMDP(discount_factor=0.95, **mountain_car_pinned_kwargs())


def _build_push() -> PushPOMDP:
    return PushPOMDP(discount_factor=0.95, **push_pinned_kwargs())


def _build_continuous_push() -> ContinuousPushPOMDP:
    return ContinuousPushPOMDP(discount_factor=0.95, **continuous_push_pinned_kwargs())


def _build_continuous_push_discrete() -> ContinuousPushPOMDPDiscreteActions:
    return ContinuousPushPOMDPDiscreteActions(
        discount_factor=0.95, **continuous_push_discrete_actions_pinned_kwargs()
    )


def _build_rock_sample() -> RockSamplePOMDP:
    return RockSamplePOMDP(discount_factor=0.95, **rock_sample_pinned_kwargs())


def _build_discrete_light_dark() -> DiscreteLightDarkPOMDP:
    return DiscreteLightDarkPOMDP(discount_factor=0.95, **discrete_light_dark_pinned_kwargs())


def _build_continuous_light_dark() -> ContinuousLightDarkPOMDP:
    return ContinuousLightDarkPOMDP(discount_factor=0.95, **continuous_light_dark_pinned_kwargs())


def _build_continuous_light_dark_discrete() -> ContinuousLightDarkPOMDPDiscreteActions:
    return ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95, **continuous_light_dark_discrete_actions_pinned_kwargs()
    )


def _build_pacman() -> PacManPOMDP:
    return PacManPOMDP(discount_factor=0.95, **pacman_pinned_kwargs())


def _build_laser_tag() -> LaserTagPOMDP:
    return LaserTagPOMDP(discount_factor=0.95, **laser_tag_pinned_kwargs())


def _build_continuous_laser_tag() -> ContinuousLaserTagPOMDP:
    return ContinuousLaserTagPOMDP(discount_factor=0.95, **continuous_laser_tag_pinned_kwargs())


def _build_continuous_laser_tag_discrete() -> ContinuousLaserTagPOMDPDiscreteActions:
    return ContinuousLaserTagPOMDPDiscreteActions(
        discount_factor=0.95, **continuous_laser_tag_discrete_actions_pinned_kwargs()
    )


def _build_safety_ant() -> SafeAntVelocityPOMDP:
    return SafeAntVelocityPOMDP(discount_factor=0.95, **safety_ant_velocity_pinned_kwargs())


# Registry of (env_id, builder). New envs added here are automatically
# covered by every parametrized test below.
ENV_BUILDERS: List[Tuple[str, EnvBuilder]] = [
    ("TigerPOMDP", _build_tiger),
    ("SanityPOMDP", _build_sanity),
    ("CartPolePOMDP", _build_cartpole),
    ("MountainCarPOMDP", _build_mountain_car),
    ("PushPOMDP", _build_push),
    ("ContinuousPushPOMDP", _build_continuous_push),
    ("ContinuousPushPOMDPDiscreteActions", _build_continuous_push_discrete),
    ("RockSamplePOMDP", _build_rock_sample),
    ("DiscreteLightDarkPOMDP", _build_discrete_light_dark),
    ("ContinuousLightDarkPOMDP", _build_continuous_light_dark),
    ("ContinuousLightDarkPOMDPDiscreteActions", _build_continuous_light_dark_discrete),
    ("PacManPOMDP", _build_pacman),
    ("LaserTagPOMDP", _build_laser_tag),
    ("ContinuousLaserTagPOMDP", _build_continuous_laser_tag),
    ("ContinuousLaserTagPOMDPDiscreteActions", _build_continuous_laser_tag_discrete),
    ("SafeAntVelocityPOMDP", _build_safety_ant),
]


# Envs whose ``hash_observation`` override is missing — observations are
# unhashable ndarrays and the base class's default ``hash(observation)``
# raises ``NotImplementedError``. Marked ``xfail(strict=True)`` so the
# moment a real override lands, the unexpected pass forces this list to
# be trimmed.
#
# Note on ContinuousPushPOMDP: it inherits from ``Environment`` directly,
# *not* from ``PushPOMDP``, so the override that PushPOMDP carries does
# not apply here. ``ContinuousPushPOMDPDiscreteActions`` inherits from
# ``ContinuousPushPOMDP`` and is broken transitively.
HASH_OBSERVATION_BROKEN_ENVS = frozenset(
    {
        "CartPolePOMDP",
        "MountainCarPOMDP",
        "SafeAntVelocityPOMDP",
        "ContinuousPushPOMDP",
        "ContinuousPushPOMDPDiscreteActions",
    }
)


def _all_env_params() -> List[pytest.param]:  # type: ignore[valid-type]
    """Build the full env-builder param list with no marks applied."""
    return [pytest.param(builder, id=env_id) for env_id, builder in ENV_BUILDERS]


def _hash_observation_env_params() -> List[pytest.param]:  # type: ignore[valid-type]
    """Env-builder param list for hash_observation tests, with xfail on broken envs."""
    params: List[pytest.param] = []  # type: ignore[valid-type]
    for env_id, builder in ENV_BUILDERS:
        if env_id in HASH_OBSERVATION_BROKEN_ENVS:
            mark = pytest.mark.xfail(
                strict=True,
                reason=(
                    f"{env_id} does not override hash_observation; the base class "
                    "default raises NotImplementedError on ndarray observations. "
                    "Remove this xfail when the override lands."
                ),
            )
            params.append(pytest.param(builder, id=env_id, marks=mark))
        else:
            params.append(pytest.param(builder, id=env_id))
    return params


def _discrete_action_env_params() -> List[pytest.param]:  # type: ignore[valid-type]
    """Param list filtered to envs that expose ``get_actions``."""
    discrete: List[pytest.param] = []  # type: ignore[valid-type]
    for env_id, builder in ENV_BUILDERS:
        env = builder()
        if isinstance(env, DiscreteActionsEnvironment):
            discrete.append(pytest.param(builder, id=env_id))
    return discrete


def _sample_action(env: Environment) -> Any:
    """Return one valid action for ``env``.

    For discrete-action envs we use the first enumerated action; for
    continuous-action envs we hand-pick a 2-D unit vector, which is the
    action shape the three continuous envs in the registry all accept.
    """
    if isinstance(env, DiscreteActionsEnvironment):
        return env.get_actions()[0]
    if env.space_info.action_space is SpaceType.CONTINUOUS:
        return np.array([1.0, 0.0])
    raise NotImplementedError(
        f"_sample_action does not know how to build an action for {type(env).__name__}"
    )


def _equal_copy(value: Any) -> Any:
    """Return a value equal to ``value`` but not the same object.

    Ensures hash/equality tests exercise actual value semantics rather
    than object identity. Uses ``np.array(..., copy=True)`` for ndarrays
    (the natural distinct-buffer copy) and ``deepcopy`` otherwise.
    """
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return deepcopy(value)


# ---------------------------------------------------------------------------
# hash_action conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_builder", _all_env_params())
def test_hash_action_returns_hashable_key(env_builder: EnvBuilder) -> None:
    """``hash_action`` returns a value usable as a dict / set key.

    Purpose: Verifies the most basic part of the hash_action contract
        across every environment. Tree-search planners index action
        children by ``hash_action(a)`` and rely on the result being
        hashable.

    Given: A freshly built environment from the registry and one valid
        action sampled from it.
    When: ``env.hash_action(action)`` is called and the result is
        passed to the built-in ``hash``.
    Then: ``hash(...)`` does not raise.

    Test type: integration
    """
    env = env_builder()
    action = _sample_action(env)
    key = env.hash_action(action)
    hash(key)  # raises TypeError if not hashable; that is a real bug


@pytest.mark.parametrize("env_builder", _all_env_params())
def test_hash_action_consistent_with_action_equality(env_builder: EnvBuilder) -> None:
    """Equal actions hash to equal keys.

    Purpose: Enforces the ``a == b ==> hash_action(a) == hash_action(b)``
        half of the hash contract. Without this, planner action lookup
        silently misses children and re-expands actions that already
        exist.

    Given: An environment and two equal-but-distinct copies of one
        valid action (``np.array(copy=True)`` for ndarrays, ``deepcopy``
        otherwise).
    When: ``hash_action`` is invoked on both copies.
    Then: The two returned keys compare equal.

    Test type: integration
    """
    env = env_builder()
    action = _sample_action(env)
    action_copy = _equal_copy(action)
    assert env.hash_action(action) == env.hash_action(action_copy)


@pytest.mark.parametrize("env_builder", _discrete_action_env_params())
def test_hash_action_distinct_across_discrete_action_set(env_builder: EnvBuilder) -> None:
    """Every action in ``get_actions()`` hashes to a distinct key.

    Purpose: Discrete-action planners (POMCP, PFT, sparse-PFT) keep one
        child per action and look them up by ``hash_action``. If two
        distinct actions collide, one child silently overwrites the
        other and the planner explores the wrong subtree.

    Given: A discrete-action environment.
    When: ``hash_action`` is applied to every action in
        ``env.get_actions()``.
    Then: The number of distinct hash keys equals the number of actions.

    Test type: integration
    """
    env = env_builder()
    assert isinstance(env, DiscreteActionsEnvironment)
    actions = env.get_actions()
    hashes = {env.hash_action(a) for a in actions}
    assert len(hashes) == len(actions), (
        f"{type(env).__name__}.hash_action collided across the discrete action set: "
        f"{len(actions)} actions but only {len(hashes)} distinct keys"
    )


# ---------------------------------------------------------------------------
# hash_observation conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_builder", _hash_observation_env_params())
def test_hash_observation_returns_hashable_key(env_builder: EnvBuilder) -> None:
    """``hash_observation`` returns a value usable as a dict / set key.

    Purpose: Belief-update structures (e.g. POMCPOW's per-action
        observation map) index belief children by
        ``hash_observation(o)`` and rely on the result being hashable.
        For envs whose observations are themselves unhashable (ndarray)
        the env MUST override ``hash_observation`` to return a surrogate
        such as ``ndarray.tobytes()``.

    Given: A freshly built environment, one initial state, one valid
        action, and one observation drawn from
        ``sample_observation(state, action)``.
    When: ``env.hash_observation(obs)`` is called and the result is
        passed to ``hash``.
    Then: ``hash(...)`` does not raise.

    Test type: integration
    """
    env = env_builder()
    np.random.seed(0)
    state = env.initial_state_dist().sample()[0]
    action = _sample_action(env)
    observation = env.sample_observation(next_state=state, action=action)
    key = env.hash_observation(observation)
    hash(key)


@pytest.mark.parametrize("env_builder", _hash_observation_env_params())
def test_hash_observation_consistent_with_equality(env_builder: EnvBuilder) -> None:
    """Equal observations hash to equal keys.

    Purpose: Enforces the
        ``is_equal_observation(a, b) ==> hash_observation(a) == hash_observation(b)``
        half of the hash contract. The base ``Environment`` docstring
        calls this contract out explicitly because tree planners rely on
        it for O(1) child lookup.

    Given: One sampled observation and an equal-but-distinct copy
        (distinct ndarray buffer for ndarray-valued observations,
        ``deepcopy`` otherwise).
    When: ``hash_observation`` is invoked on both copies.
    Then: ``is_equal_observation`` confirms the two copies are equal,
        and the two hash keys compare equal.

    Test type: integration
    """
    env = env_builder()
    np.random.seed(0)
    state = env.initial_state_dist().sample()[0]
    action = _sample_action(env)
    observation = env.sample_observation(next_state=state, action=action)
    observation_copy = _equal_copy(observation)
    assert env.is_equal_observation(observation, observation_copy), (
        f"{type(env).__name__}.is_equal_observation rejected an exact copy of an "
        "observation it had just produced — equality and copy semantics disagree"
    )
    assert env.hash_observation(observation) == env.hash_observation(observation_copy)
