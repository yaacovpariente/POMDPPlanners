# SPDX-License-Identifier: MIT

"""Cross-environment conformance tests for the Environment API.

Covers the contracts that planners and beliefs rely on but that, prior
to this file, were tested for at most a handful of environments:

* ``hash_action(a)`` — must return a hashable key, agree on equal actions,
  and be pairwise distinct across the discrete action set when the env
  is a :class:`DiscreteActionsEnvironment`.
* ``hash_observation(o)`` — must return a hashable key and agree on two
  observations that ``is_equal_observation`` considers equal.
* batch/single agreement — ``reward_batch`` must agree with a loop over
  ``reward``, ``sample_next_state_batch`` must produce the same state
  type/shape as ``sample_next_state``, and
  ``observation_log_probability_single`` must agree with the batched
  ``observation_log_probability``. Planners mix the two paths freely
  (particle filters take the batch path, tree expansion the single
  path), so a divergence silently changes the model mid-search.
* ``reward_requires_next_state`` — when it is ``True`` the reward really
  must depend on the realised ``next_state``, and the no-``next_state``
  call must still return a usable number rather than a silent bogus one.
* serialization — ``to_dict`` / ``from_dict`` must round-trip and
  ``config_id`` must survive the trip and be stable across two identical
  constructions. ``test_environment_serialization.py`` hand-writes one
  block per env, so a new env gets zero coverage until someone
  remembers to add one; the parametrized test here closes that gap.
* ``step_info`` — must consume no randomness (a single draw there shifts
  the RNG stream for every later transition and observation), and must
  tolerate the terminal bookkeeping step's ``action=None,
  next_state=None``.
* declared metric channels — every channel in ``get_metric_specs()`` must
  be a key ``step_info`` actually emits, and every name in
  ``get_metric_names()`` must actually be produced by
  ``compute_metrics``, or the metric is silently dropped by consumers
  that look it up by name.
* declared reward range — every reward observed on a rollout must fall
  inside ``reward_range``, which the base class validates on
  construction and which downstream CVaR / confidence-interval code
  treats as a hard bound.
* seed determinism — the same seed applied to a freshly built env must
  reproduce the same trajectory.

The conformance tests are parametrized over every concrete environment
class so that a new env wired into :data:`ENV_BUILDERS` is automatically
checked.

Environments that violate one of these contracts today are marked
``xfail`` with ``strict=True`` so the gap is documented and the suite
turns green automatically the moment the fix lands.
"""

import importlib
import random
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, List, Tuple

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    Environment,
    SpaceType,
)
from POMDPPlanners.core.simulation.history import History, StepData
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


# Envs whose ``sample_next_state_batch`` disagrees with the per-state
# ``sample_next_state`` on the *type* of the states it produces. Marked
# ``xfail(strict=True)`` so the gap is documented until the fix lands.
SAMPLE_NEXT_STATE_BATCH_BROKEN_ENVS: frozenset = frozenset()


# No env is known to break the ``to_dict`` / ``from_dict`` round trip.
SERIALIZATION_ROUND_TRIP_BROKEN_ENVS: frozenset = frozenset()


# No env is known to change its ``config_id`` once it has been used.
CONFIG_ID_UNSTABLE_AFTER_USE_ENVS: frozenset = frozenset()


# Envs whose declared ``reward_range`` is defined over a subset of the states a
# rollout can reach. Not a bug and not a TODO -- a recorded decision.
#
# Both ContinuousLightDark variants declare the bound over *in-grid* states,
# which is the space they model: the minimum is the reward at the far corner of
# the grid, and nothing inside the grid scores lower. Out-of-grid states are
# reachable -- leaving the grid is penalised but not terminal, and _sample_mvn
# deliberately does not clip samples, because clipping the sampler while the
# observation PDF stays unclipped would break importance weights near the edges
# -- and a reward sampled from one falls below that minimum. Those states are
# outside the region the bound is defined over, so the two do not contradict.
#
# The rollout below reaches such a state, so it reports a violation. That is
# recorded here rather than silenced, and ``strict=True`` is deliberate: if
# someone clips the sampler or widens the bound, this passes unexpectedly and
# the decision comes back into view instead of being re-made silently.
#
# Both variants are listed. The base one violates on only 12 of 200 seeds
# against the discrete variant's 44, so with the originally pinned seeds it
# passed -- by luck, not by construction. Seed 7 is pinned below precisely so
# both reach an out-of-grid state deterministically; a test that passes by luck
# is worse than one marked xfail.
REWARD_RANGE_IN_GRID_ONLY_ENVS = frozenset(
    {
        "ContinuousLightDarkPOMDP",
        "ContinuousLightDarkPOMDPDiscreteActions",
    }
)


# No env is known to lose seed determinism.
#
# ``ContinuousPushPOMDP`` used to, and the cause is worth remembering: its
# per-action C++ kernel caches were keyed by ``id(action)`` while holding no
# reference to the action array. CPython hands the id of a freed object to the
# next object allocated there, so a transient action array could hit the entry
# left by an earlier one and be simulated with the wrong action's kernel
# (measured: 10 kernels built over a 200-step rollout, 190 steps served the
# wrong one). Those caches are now populated only for env-owned action arrays
# pinned via ``_pin_action_vectors``.
SEED_DETERMINISM_BROKEN_ENVS: frozenset = frozenset()


# ``_native`` extension modules that own their own C++ RNG. ``np.random.seed``
# alone does NOT reach them, so any determinism/agreement test that crosses a
# native kernel must seed these too (the convention the native-equivalence
# tests already use).
_NATIVE_SEED_MODULES: Tuple[str, ...] = (
    "POMDPPlanners.core._native",
    "POMDPPlanners.environments.cartpole_pomdp._native",
    "POMDPPlanners.environments.laser_tag_pomdp._native",
    "POMDPPlanners.environments.light_dark_pomdp._native",
    "POMDPPlanners.environments.mountain_car_pomdp._native",
    "POMDPPlanners.environments.pacman_pomdp._native",
    "POMDPPlanners.environments.push_pomdp._native",
    "POMDPPlanners.environments.rock_sample_pomdp._native",
    "POMDPPlanners.environments.safety_ant_velocity_pomdp._native",
)


def _native_seed_functions() -> List[Callable[[int], None]]:
    """Collect the ``set_seed`` entry point of every built native module."""
    seeders: List[Callable[[int], None]] = []
    for module_path in _NATIVE_SEED_MODULES:
        try:
            module = importlib.import_module(module_path)
        except ImportError:  # extension not built in this checkout
            continue
        for attr in ("set_seed", "set_default_seed"):
            seed_fn = getattr(module, attr, None)
            if seed_fn is not None:
                seeders.append(seed_fn)
                break
    return seeders


_NATIVE_SEEDERS = _native_seed_functions()


def _seed_all(seed: int) -> None:
    """Seed every RNG a step can draw from, so a rollout is reproducible.

    Three separate streams are in play: the stdlib ``random`` module
    (DiscreteLightDark's observation model draws from it), ``np.random``,
    and one C++ RNG per native extension module.
    """
    random.seed(seed)
    np.random.seed(seed)
    for seed_fn in _NATIVE_SEEDERS:
        seed_fn(seed)


def _all_env_params() -> List[pytest.param]:  # type: ignore[valid-type]
    """Build the full env-builder param list with no marks applied."""
    return [pytest.param(builder, id=env_id) for env_id, builder in ENV_BUILDERS]


def _params_with_xfail(  # type: ignore[valid-type]
    broken: frozenset, contract: str
) -> List[pytest.param]:
    """Env-builder param list with ``xfail(strict=True)`` on ``broken`` envs.

    Args:
        broken: Env ids known to violate ``contract`` today.
        contract: Human-readable description of the violated contract,
            used verbatim in the xfail reason.

    Returns:
        One ``pytest.param`` per registry entry, marked where applicable.
    """
    params: List[pytest.param] = []  # type: ignore[valid-type]
    for env_id, builder in ENV_BUILDERS:
        if env_id in broken:
            mark = pytest.mark.xfail(
                strict=True,
                reason=(
                    f"{env_id} violates the contract: {contract}. "
                    "Remove this xfail when the fix lands."
                ),
            )
            params.append(pytest.param(builder, id=env_id, marks=mark))
        else:
            params.append(pytest.param(builder, id=env_id))
    return params


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


def _random_action(env: Environment, rng: np.random.Generator) -> Any:
    """Draw one valid action for ``env`` from ``rng``.

    Discrete envs draw uniformly from ``get_actions()``; continuous envs
    get a random 2-D unit vector, the action shape every continuous env
    in the registry accepts.
    """
    if isinstance(env, DiscreteActionsEnvironment):
        actions = env.get_actions()
        return actions[int(rng.integers(len(actions)))]
    vector = rng.normal(size=2)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else np.array([1.0, 0.0])


def _trajectory_key(value: Any) -> Any:
    """Return a comparable, hashable surrogate for a state or observation."""
    if isinstance(value, np.ndarray):
        return value.tobytes()
    return repr(value)


# Rollouts are capped so the whole file stays in the low seconds. Long enough
# to leave the initial state and hit collision / goal branches, short enough
# that 16 envs times a handful of seeds costs nothing.
_ROLLOUT_STEPS = 15
# Seed 7 is here on purpose: it is the lowest seed on which *both*
# ContinuousLightDark variants wander far enough outside the grid to score
# below their in-grid bound, which is what makes their xfail deterministic
# rather than luck. See REWARD_RANGE_IN_GRID_ONLY_ENVS.
_REWARD_RANGE_SEEDS = (0, 1, 2, 3, 4, 7)


def _rollout(
    env: Environment, seed: int, max_steps: int = _ROLLOUT_STEPS
) -> List[Tuple[Any, Any, float]]:
    """Roll ``env`` forward under random actions from a fully seeded RNG.

    Both the numpy RNG and every native C++ RNG are seeded from ``seed``,
    and the action stream is drawn from its own ``default_rng(seed)`` so
    the action sequence does not depend on how much randomness the env
    itself consumes.

    Returns:
        One ``(next_state_key, observation_key, reward)`` triple per step,
        truncated at the first terminal state.
    """
    _seed_all(seed)
    action_rng = np.random.default_rng(seed)
    state = env.initial_state_dist().sample()[0]
    trace: List[Tuple[Any, Any, float]] = []
    for _ in range(max_steps):
        if env.is_terminal(state):
            break
        action = _random_action(env, action_rng)
        next_state, observation, reward = env.sample_next_step(state, action)
        trace.append((_trajectory_key(next_state), _trajectory_key(observation), float(reward)))
        state = next_state
    return trace


def _uniform_belief(env: Environment) -> WeightedParticleBelief:
    """Return a four-particle uniform belief drawn from the initial state dist."""
    particles = env.initial_state_dist().sample(4)
    return WeightedParticleBelief(particles, np.log(np.full(4, 0.25)))


def _rollout_history(env: Environment, seed: int, max_steps: int = 8) -> History:
    """Build a :class:`History` from a short random rollout of ``env``.

    ``compute_metrics`` consumes histories, so exercising the declared
    metric channels needs a real (if tiny) episode. The beliefs attached
    to each step are uniform particle beliefs drawn from the initial
    state distribution — metric code reads them at most for summary
    statistics, never for correctness of the belief itself.
    """
    _seed_all(seed)
    action_rng = np.random.default_rng(seed)
    state = env.initial_state_dist().sample()[0]
    steps: List[StepData] = []
    terminated = False
    for _ in range(max_steps):
        if env.is_terminal(state):
            terminated = True
            break
        action = _random_action(env, action_rng)
        next_state, observation, reward = env.sample_next_step(state, action)
        steps.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=float(reward),
                belief=_uniform_belief(env),
                info=env.step_info(state, action, next_state) or None,
            )
        )
        state = next_state
    if terminated:
        # Mirror EpisodeRunner._add_terminal_step: the final state is only ever
        # recorded here, and metrics that scan every visited state need it.
        steps.append(
            StepData(
                state=state,
                action=None,
                next_state=None,
                observation=None,
                reward=None,
                belief=_uniform_belief(env),
                info=env.step_info(state, None, None) or None,
            )
        )
    return History(
        history=steps,
        discount_factor=env.discount_factor,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=len(steps),
        reach_terminal_state=terminated,
        policy_run_data=[],
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


# ---------------------------------------------------------------------------
# batch / single agreement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_builder", _all_env_params())
def test_reward_batch_agrees_with_looped_reward(env_builder: EnvBuilder) -> None:
    """``reward_batch`` matches a plain loop over ``reward``.

    Purpose: ``reward_batch`` exists so vectorized envs can skip the
        Python loop, and several envs override it with a numpy
        implementation. Any divergence means a particle filter and a
        tree-search rollout score the same transition differently.

    Given: Four states drawn from the initial state distribution and one
        valid action, with numpy and every native RNG re-seeded
        identically before each of the two paths (some reward models
        draw — probabilistic hazard hits, for instance — so an unseeded
        comparison would compare draws rather than code paths).
    When: ``reward_batch(states, action)`` and
        ``[reward(s, action) for s in states]`` are both evaluated.
    Then: The two reward vectors agree elementwise.

    Test type: integration
    """
    env = env_builder()
    _seed_all(0)
    states = env.initial_state_dist().sample(4)
    action = _sample_action(env)

    _seed_all(1)
    batched = np.asarray(env.reward_batch(states, action), dtype=float)
    _seed_all(1)
    looped = np.array([float(env.reward(state, action)) for state in states])

    assert np.allclose(batched, looped), (
        f"{type(env).__name__}.reward_batch disagrees with looped reward: "
        f"{batched} vs {looped}"
    )


@pytest.mark.parametrize(
    "env_builder",
    _params_with_xfail(
        SAMPLE_NEXT_STATE_BATCH_BROKEN_ENVS,
        "sample_next_state_batch produces states of a different shape or "
        "dtype than sample_next_state",
    ),
)
def test_sample_next_state_batch_matches_single_sample(env_builder: EnvBuilder) -> None:
    """``sample_next_state_batch`` produces the same state type as ``sample_next_state``.

    Purpose: Particle filters take the batch path while tree expansion
        takes the single path, and the two feed the same belief. If the
        batch path returns a different shape or dtype the belief silently
        holds two incompatible kinds of particle.

    Given: Four states from the initial state distribution and one valid
        action, with all RNGs re-seeded identically before each path.
    When: ``sample_next_state_batch(states, action)`` and a loop over
        ``sample_next_state(state, action)`` are both evaluated.
    Then: The two results have the same length, and the first element of
        each has the same shape and dtype. Values are not compared — the
        two paths legitimately consume randomness differently.

    Test type: integration
    """
    env = env_builder()
    _seed_all(0)
    states = env.initial_state_dist().sample(4)
    action = _sample_action(env)

    _seed_all(1)
    batched = env.sample_next_state_batch(states, action)
    _seed_all(1)
    singles = [env.sample_next_state(state=state, action=action) for state in states]

    assert len(batched) == len(singles), (
        f"{type(env).__name__}.sample_next_state_batch returned {len(batched)} states "
        f"for {len(singles)} input particles"
    )
    batched_first = np.asarray(batched[0])
    single_first = np.asarray(singles[0])
    assert batched_first.shape == single_first.shape, (
        f"{type(env).__name__}: batch state shape {batched_first.shape} != "
        f"single state shape {single_first.shape}"
    )
    assert batched_first.dtype == single_first.dtype, (
        f"{type(env).__name__}: batch state dtype {batched_first.dtype} != "
        f"single state dtype {single_first.dtype}"
    )


@pytest.mark.parametrize("env_builder", _all_env_params())
def test_observation_log_probability_single_agrees_with_batched(
    env_builder: EnvBuilder,
) -> None:
    """The scalar likelihood fast-path matches the batched likelihood.

    Purpose: ``observation_log_probability_single`` is an optimization —
        several envs override it to skip numpy allocation on a singleton
        input. Incremental belief updates use the fast path and batch
        reweighting uses the batched one, so a divergence changes the
        observation model depending on which planner is running.

    Given: One sampled ``(next_state, action, observation)`` triple.
    When: ``observation_log_probability_single`` and
        ``observation_log_probability(..., observations=[observation])``
        are both evaluated on it.
    Then: The scalar equals the single element of the batched result.

    Test type: integration
    """
    env = env_builder()
    _seed_all(0)
    state = env.initial_state_dist().sample()[0]
    action = _sample_action(env)
    next_state = env.sample_next_state(state=state, action=action)
    observation = env.sample_observation(next_state=next_state, action=action)

    single = float(
        env.observation_log_probability_single(
            next_state=next_state, action=action, observation=observation
        )
    )
    batched = float(
        env.observation_log_probability(
            next_state=next_state, action=action, observations=[observation]
        )[0]
    )
    assert np.isclose(single, batched), (
        f"{type(env).__name__}.observation_log_probability_single returned {single} "
        f"but the batched path returned {batched}"
    )


# ---------------------------------------------------------------------------
# reward_requires_next_state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_builder", _all_env_params())
def test_reward_requires_next_state_is_honored(env_builder: EnvBuilder) -> None:
    """A ``True`` ``reward_requires_next_state`` really means the reward uses it.

    Purpose: Simulation drivers reorder their RNG draws based on this
        flag — ``True`` makes them sample the transition first so reward
        and trajectory share one draw. A flag that is ``True`` while the
        reward ignores ``next_state`` buys the reordering for nothing;
        worse, a ``reward(state, action)`` call with no ``next_state``
        must not silently return a wrong number, because that is exactly
        the call a driver that missed the flag would make.

    Given: An env whose ``reward_requires_next_state`` is ``True``, one
        state, one action, and eight independently drawn next states.
    When: ``reward`` is scored against each next state with the RNG
        pinned, and once with ``next_state=None``.
    Then: At least two of the eight rewards differ (the reward genuinely
        consumes ``next_state``), and the ``None`` call still returns a
        finite value inside the declared reward range rather than a
        silent NaN or out-of-range number.

    Test type: integration
    """
    env = env_builder()
    if not env.reward_requires_next_state:
        pytest.skip(f"{type(env).__name__}.reward_requires_next_state is False")

    _seed_all(0)
    state = env.initial_state_dist().sample()[0]
    action = _sample_action(env)

    rewards = set()
    for draw_seed in range(8):
        _seed_all(draw_seed)
        next_state = env.sample_next_state(state=state, action=action)
        _seed_all(100)
        rewards.add(round(float(env.reward(state=state, action=action, next_state=next_state)), 9))
    assert len(rewards) > 1, (
        f"{type(env).__name__} declares reward_requires_next_state=True but scored "
        f"eight different next states identically ({rewards}) — the flag is vacuous"
    )

    fallback = float(env.reward(state=state, action=action))
    assert np.isfinite(fallback), (
        f"{type(env).__name__}.reward(state, action) returned {fallback} when called "
        "without next_state; the docstring requires a drawn or computed fallback"
    )
    if env.reward_range is not None:
        low, high = env.reward_range
        assert low - 1e-9 <= fallback <= high + 1e-9, (
            f"{type(env).__name__}.reward(state, action) returned {fallback} without a "
            f"next_state, outside its declared reward_range {env.reward_range}"
        )


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_builder",
    _params_with_xfail(
        SERIALIZATION_ROUND_TRIP_BROKEN_ENVS,
        "to_dict/from_dict does not round-trip, or the rebuilt env's config_id "
        "differs from the original's",
    ),
)
def test_serialization_round_trip_preserves_config_id(env_builder: EnvBuilder) -> None:
    """``from_dict(to_dict(env))`` rebuilds an env with the same ``config_id``.

    Purpose: ``test_environment_serialization.py`` hand-writes one block
        per environment, so a newly added env has zero serialization
        coverage until someone remembers to add another block. This
        parametrized round-trip closes that gap for every env in the
        registry. ``config_id`` is the identity experiment caches and
        result tables key on, so a round-trip that changes it makes a
        reloaded env look like a different environment.

    Given: A freshly built environment.
    When: It is serialized with ``to_dict`` and rebuilt with
        ``from_dict``.
    Then: The rebuild succeeds, produces an instance of the same class,
        and carries the original's ``config_id`` and discount factor.

    Test type: integration
    """
    env = env_builder()
    data = env.to_dict()
    rebuilt = type(env).from_dict(data)

    assert isinstance(rebuilt, type(env))
    assert rebuilt.discount_factor == env.discount_factor
    assert rebuilt.config_id == env.config_id, (
        f"{type(env).__name__} config_id changed across a to_dict/from_dict round trip: "
        f"{env.config_id} -> {rebuilt.config_id}"
    )


@pytest.mark.parametrize("env_builder", _all_env_params())
def test_serialization_round_trip_preserves_equality(env_builder: EnvBuilder) -> None:
    """``from_dict(to_dict(env))`` compares equal to the env it came from.

    Purpose: ``__eq__`` and ``config_id`` are two answers to the same
        question — is this the same environment — and they have to agree.
        They did not: ``__eq__`` fell back to ``v1 == v2`` for a sub-object
        like the reward model, whose class defines no ``__eq__``, so it
        compared by *identity*. An env and its own rebuild were never
        equal, however faithfully the rebuild was constructed.

    Given: A freshly built environment.
    When: It is serialized with ``to_dict`` and rebuilt with ``from_dict``.
    Then: The rebuild compares equal to the original.

    Test type: integration
    """
    env = env_builder()
    rebuilt = type(env).from_dict(env.to_dict())
    assert rebuilt == env, (
        f"{type(env).__name__} does not compare equal to its own to_dict/from_dict "
        "rebuild, even though the two carry the same configuration"
    )


@pytest.mark.parametrize("env_builder", _all_env_params())
def test_equality_agrees_with_hash_and_config_id(env_builder: EnvBuilder) -> None:
    """Two identically built envs are equal, and hash alike.

    Purpose: ``__hash__`` is ``hash(config_id)``, so equality is already
        *defined* to be about configuration. If ``__eq__`` disagrees, two
        envs with one configuration land in the same hash bucket and then
        refuse to match, and every dict or set keyed on an environment
        stops deduplicating.

    Given: Two environments built from the same builder with identical
        pinned kwargs.
    When: They are compared, hashed, and their ``config_id`` read.
    Then: They are equal, their hashes match, and their ``config_id``
        values match — all three agreeing.

    Test type: integration
    """
    first = env_builder()
    second = env_builder()
    assert first == second, (
        f"{type(first).__name__} instances built from identical config are not equal; "
        "__eq__ is comparing something that is not configuration"
    )
    assert hash(first) == hash(second)
    assert first.config_id == second.config_id


def _perturbable_config_sub_attribute(env: Environment) -> Any:
    """Find a scalar inside a config sub-object of ``env`` that can be nudged.

    The structural branch of ``Environment.__eq__`` exists for values like a
    reward or observation model: built from the env's own config, defining no
    equality of their own, and otherwise compared by object identity. This
    locates a scalar *inside* one so a test can perturb it and check the
    difference is still noticed.

    Returns:
        A ``(sub_object, attribute_name)`` pair, or ``None`` when the env
        holds no such sub-object carrying a scalar.
    """
    for key, value in vars(env).items():
        if key.startswith("_") or callable(value):
            continue
        # Enum members are process-wide singletons compared by identity, not
        # structurally — perturbing one would mutate it for both envs and for
        # every other test in the session.
        if isinstance(value, (Enum, np.ndarray, Environment)):
            continue
        if not hasattr(value, "__dict__"):
            continue
        if type(value).__eq__ is not object.__eq__:
            continue
        for attribute, inner in vars(value).items():
            if isinstance(inner, bool) or not isinstance(inner, (int, float)):
                continue
            return value, attribute
    return None


@pytest.mark.parametrize("env_builder", _all_env_params())
def test_inequality_survives_a_changed_discount_factor(env_builder: EnvBuilder) -> None:
    """Envs differing in a top-level value compare unequal.

    Purpose: The base half of the equality contract — equality must still
        separate environments that differ in their plainest scalar
        configuration.

    Given: Two environments from the same builder differing only in
        ``discount_factor``.
    When: They are compared.
    Then: They are not equal, and their ``config_id`` values differ.

    Test type: integration
    """
    first = env_builder()
    second = env_builder()
    second.discount_factor = first.discount_factor / 2.0

    assert first != second, (
        f"{type(first).__name__} compares equal to an env with a different "
        "discount factor"
    )
    assert first.config_id != second.config_id


@pytest.mark.parametrize("env_builder", _all_env_params())
def test_inequality_survives_a_changed_config_sub_object(env_builder: EnvBuilder) -> None:
    """A difference inside a config sub-object still makes envs unequal.

    Purpose: Guards the structural comparison specifically. ``__eq__``
        descends into sub-objects that define no equality of their own
        (the reward model) instead of comparing them by identity, and
        that must not turn into "any two reward models are the same".
        A perturbation of a top-level scalar would not exercise this
        branch at all, so this reaches inside one.

    Given: Two environments from the same builder, with one scalar
        attribute inside a config sub-object perturbed on the second.
    When: They are compared.
    Then: They are not equal, and their ``config_id`` values differ —
        equality and the config hash agreeing on the difference.

    Test type: integration
    """
    first = env_builder()
    second = env_builder()
    found = _perturbable_config_sub_attribute(second)
    if found is None:
        pytest.skip(f"{type(first).__name__} holds no comparable config sub-object")
    sub_object, perturbed = found
    setattr(sub_object, perturbed, float(getattr(sub_object, perturbed)) + 123.5)

    assert first != second, (
        f"{type(first).__name__} compares equal to an env whose "
        f"{type(sub_object).__name__}.{perturbed} differs; the structural "
        "comparison is too permissive"
    )
    assert first.config_id != second.config_id


@pytest.mark.parametrize("env_builder", _all_env_params())
def test_config_id_is_stable_across_identical_constructions(env_builder: EnvBuilder) -> None:
    """Two identically configured envs share one ``config_id``.

    Purpose: ``config_id`` is a content hash of the constructor config.
        If it picks up anything instance-specific — an object id, a
        timestamp, an RNG state — every run gets a fresh cache key and
        the experiment cache silently never hits.

    Given: Two environments built from the same builder with identical
        pinned kwargs.
    When: ``config_id`` is read from each.
    Then: The two ids are equal, and each is stable when read twice.

    Test type: integration
    """
    first = env_builder()
    second = env_builder()
    assert first.config_id == second.config_id, (
        f"{type(first).__name__}.config_id is not a pure function of the config: "
        f"{first.config_id} != {second.config_id}"
    )
    assert first.config_id == first.config_id


@pytest.mark.parametrize(
    "env_builder",
    _params_with_xfail(
        CONFIG_ID_UNSTABLE_AFTER_USE_ENVS,
        "config_id changes once the env has been used, because a lazily "
        "populated cache attribute is hashed into it",
    ),
)
def test_config_id_survives_using_the_environment(env_builder: EnvBuilder) -> None:
    """Using an env does not change its ``config_id``.

    Purpose: ``config_id`` hashes the instance ``__dict__``, so any
        attribute the env fills in lazily — a cached vectorized updater,
        a memoized kernel — becomes part of its identity. An env then
        hashes one way before a particle filter touches it and another
        way after, which splits one experiment's cache entries in two
        and makes a mid-run ``config_id`` unusable as a key.

    Given: A freshly built environment and its ``config_id``.
    When: The env is exercised over the batch and single paths and then
        rolled forward a few steps.
    Then: ``config_id`` is unchanged.

    Test type: integration
    """
    env = env_builder()
    before = env.config_id

    _seed_all(0)
    states = env.initial_state_dist().sample(4)
    action = _sample_action(env)
    env.reward_batch(states, action)
    env.sample_next_state_batch(states, action)
    next_state = env.sample_next_state(state=states[0], action=action)
    observation = env.sample_observation(next_state=next_state, action=action)
    env.observation_log_probability_per_state(
        next_states=states, action=action, observation=observation
    )
    _rollout(env, seed=0, max_steps=5)

    assert env.config_id == before, (
        f"{type(env).__name__}.config_id changed from {before} to {env.config_id} after "
        "the env was used; a lazily populated attribute is leaking into the config hash"
    )


# ---------------------------------------------------------------------------
# step_info
# ---------------------------------------------------------------------------


def _one_transition(env: Environment, seed: int = 0) -> Tuple[Any, Any, Any]:
    """Return one ``(state, action, next_state)`` triple from a seeded draw."""
    _seed_all(seed)
    state = env.initial_state_dist().sample()[0]
    action = _sample_action(env)
    next_state = env.sample_next_state(state=state, action=action)
    return state, action, next_state


def _numpy_state_key() -> Tuple[Any, ...]:
    """Return a comparable snapshot of the global numpy RNG state."""
    state = np.random.get_state()
    return (state[0], state[1].tobytes(), state[2], state[3], state[4])


@pytest.mark.parametrize("env_builder", _all_env_params())
def test_step_info_consumes_no_randomness(env_builder: EnvBuilder) -> None:
    """``step_info`` draws from no RNG stream.

    Purpose: The base ``step_info`` docstring carries an explicit warning
        about this. The hook runs inside the episode loop, between the
        transition and the belief update, so a single draw there shifts
        the stream for every subsequent transition and observation —
        silently changing seeded trajectories throughout the run, not
        just the metrics. The damage is invisible: the metric looks
        fine, and the trajectory is wrong somewhere else entirely.

    Given: One seeded ``(state, action, next_state)`` transition, and
        snapshots of both Python-level RNG streams.
    When: ``step_info`` is called on that transition.
    Then: The stdlib ``random`` and ``np.random`` states are byte-for-byte
        unchanged, and a probe transition drawn afterwards matches the
        one drawn without the ``step_info`` call — the probe is what
        catches a draw from a native C++ RNG, whose state no Python API
        exposes.

    Test type: integration
    """
    env = env_builder()
    state, action, next_state = _one_transition(env)

    _seed_all(7)
    expected_probe = _trajectory_key(env.sample_next_state(state=state, action=action))

    _seed_all(7)
    numpy_before = _numpy_state_key()
    random_before = random.getstate()
    env.step_info(state, action, next_state)
    assert _numpy_state_key() == numpy_before, (
        f"{type(env).__name__}.step_info advanced the np.random stream; every "
        "subsequent transition and observation in a seeded run is now shifted"
    )
    assert random.getstate() == random_before, (
        f"{type(env).__name__}.step_info advanced the stdlib random stream; every "
        "subsequent transition and observation in a seeded run is now shifted"
    )
    actual_probe = _trajectory_key(env.sample_next_state(state=state, action=action))
    assert actual_probe == expected_probe, (
        f"{type(env).__name__}.step_info consumed randomness from a native RNG: the "
        "next transition drawn after it differs from the same transition drawn without it"
    )


@pytest.mark.parametrize("env_builder", _all_env_params())
def test_step_info_tolerates_terminal_bookkeeping_step(env_builder: EnvBuilder) -> None:
    """``step_info`` accepts the terminal step's ``action=None, next_state=None``.

    Purpose: ``EpisodeRunner._add_terminal_step`` calls ``step_info``
        once per terminated episode with both transition arguments
        ``None``, because the final state is only ever recorded there and
        metrics that scan every visited state need it. An implementation
        that assumes a transition raises, and the whole episode is lost
        at the point where its result is being recorded.

    Given: A state reached by a short seeded rollout.
    When: ``step_info(state, None, None)`` is called.
    Then: It does not raise, and returns a mapping of channel name to a
        plain finite scalar — the values must be picklable scalars to
        survive the trip back from a worker process.

    Test type: integration
    """
    env = env_builder()
    _seed_all(0)
    state = env.initial_state_dist().sample()[0]

    info = env.step_info(state, None, None)

    assert isinstance(info, dict), (
        f"{type(env).__name__}.step_info returned {type(info).__name__} on the terminal "
        "bookkeeping step; the contract is a flat name -> scalar mapping"
    )
    for channel, value in info.items():
        assert isinstance(channel, str), (
            f"{type(env).__name__}.step_info used a non-string channel name {channel!r}"
        )
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"{type(env).__name__}.step_info reported {channel!r} as "
            f"{type(value).__name__}; values must be plain picklable scalars"
        )
        assert np.isfinite(float(value)), (
            f"{type(env).__name__}.step_info reported {channel!r} as {value} on the "
            "terminal bookkeeping step; transition-describing channels must report a "
            "neutral value there, not NaN"
        )


@pytest.mark.parametrize("env_builder", _all_env_params())
def test_declared_metric_spec_channels_are_emitted(env_builder: EnvBuilder) -> None:
    """Every channel in ``get_metric_specs()`` is a key ``step_info`` emits.

    Purpose: ``get_metric_specs`` states the rule directly: only declare
        a channel the environment actually emits on every step, because a
        declared-but-unreported channel yields a metric that is silently
        dropped. Nothing enforces it — the aggregation just omits the
        metric, so the name survives in ``get_metric_names`` while no
        value is ever produced for it.

        ``test_step_info_migration.py`` checks the same invariant, but
        over its own list of migrated env slugs. This one rides on
        :data:`ENV_BUILDERS`, so an env added to the registry is covered
        without also being remembered into that list.

    Given: An env that declares at least one metric spec, and one seeded
        transition.
    When: ``step_info`` is called on that transition.
    Then: Every declared channel appears among the returned keys.

    Test type: integration
    """
    env = env_builder()
    declared = {spec.channel for spec in env.get_metric_specs()}
    if not declared:
        pytest.skip(f"{type(env).__name__} declares no step_info metric specs")

    state, action, next_state = _one_transition(env)
    emitted = set(env.step_info(state, action, next_state))
    missing = sorted(declared - emitted)
    assert not missing, (
        f"{type(env).__name__} declares metric spec channels that step_info never "
        f"emits, so those metrics are silently dropped: {missing}"
    )


# ---------------------------------------------------------------------------
# declared metric channels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_builder", _all_env_params())
def test_declared_metric_names_are_emitted(env_builder: EnvBuilder) -> None:
    """Every name in ``get_metric_names()`` is produced by ``compute_metrics``.

    Purpose: ``get_metric_names`` is the discovery surface hyperparameter
        optimization reads to decide what it can optimize. A name
        declared there but never emitted by ``compute_metrics`` is a
        metric that is silently dropped: the consumer looks it up, finds
        nothing, and reports no value rather than an error.

    Given: An env that declares at least one metric name, and a short
        seeded random rollout packaged as a single ``History``.
    When: ``compute_metrics([history])`` is evaluated.
    Then: Every declared name appears among the emitted metric names.

    Test type: integration
    """
    env = env_builder()
    declared = set(env.get_metric_names())
    if not declared:
        pytest.skip(f"{type(env).__name__} declares no environment-specific metrics")

    history = _rollout_history(env, seed=3)
    emitted = {metric.name for metric in env.compute_metrics([history])}
    missing = sorted(declared - emitted)
    assert not missing, (
        f"{type(env).__name__} declares metric names that compute_metrics never "
        f"emits, so they are silently dropped: {missing}"
    )


# ---------------------------------------------------------------------------
# declared reward range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_builder",
    _params_with_xfail(
        REWARD_RANGE_IN_GRID_ONLY_ENVS,
        "its declared reward_range is defined over in-grid states and this rollout "
        "reaches an out-of-grid state, which is a recorded decision rather than a "
        "wrong bound",
    ),
)
def test_declared_reward_range_bounds_observed_rewards(env_builder: EnvBuilder) -> None:
    """Every reward seen on a rollout falls inside the declared ``reward_range``.

    Purpose: ``_validate_reward_range`` only checks the declared tuple is
        well-formed; nothing checks it against reality. Downstream CVaR
        and confidence-interval code treats the range as a hard bound and
        rejects or mis-normalizes returns that escape it, so a range that
        is too narrow surfaces far from the env that declared it.

    Given: An env that declares a reward range, and five seeded random
        rollouts of at most fifteen steps each.
    When: Every step reward is compared against the declared bounds.
    Then: All of them fall inside, within a floating-point tolerance.

    Test type: integration
    """
    env = env_builder()
    reward_range = env.reward_range
    if reward_range is None:
        pytest.skip(f"{type(env).__name__} declares no reward_range")
    # Restates the guard above for the type checker: pytest.skip is not
    # annotated NoReturn in the pinned stubs, so the narrowing does not carry.
    assert reward_range is not None
    low, high = reward_range

    for seed in _REWARD_RANGE_SEEDS:
        for _, _, reward in _rollout(env, seed=seed):
            assert low - 1e-9 <= reward <= high + 1e-9, (
                f"{type(env).__name__} emitted reward {reward} on seed {seed}, outside "
                f"its declared reward_range {env.reward_range}"
            )


# ---------------------------------------------------------------------------
# seed determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_builder", _all_env_params())
def test_same_seed_reproduces_trajectory(env_builder: EnvBuilder) -> None:
    """One seed and one env config give one trajectory.

    Purpose: Every reported result depends on a seeded rollout being
        reproducible. Seeding ``np.random`` alone is not enough — the
        native C++ kernels own separate RNGs, which is why
        :func:`_seed_all` seeds those too. A regression here turns every
        seeded benchmark number into noise.

    Given: Two freshly built environments from the same builder, each
        rolled forward under the same seed with the same action stream.
        Fresh instances matter: several envs carry mutable per-instance
        caches, so re-rolling one instance is not the same experiment.
    When: The two trajectories of ``(next_state, observation, reward)``
        are compared.
    Then: They are identical step for step.

    Test type: integration
    """
    first = _rollout(env_builder(), seed=1234)
    second = _rollout(env_builder(), seed=1234)
    assert first, "rollout produced no steps; the determinism check would be vacuous"
    assert first == second, (
        f"{type(env_builder()).__name__} produced two different trajectories from the "
        "same seed"
    )
