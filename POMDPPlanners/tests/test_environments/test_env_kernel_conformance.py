# SPDX-License-Identifier: MIT

"""An environment's kernels agree with one another.

``test_env_api_conformance.py`` checks each batch path against its
single-state twin. This file checks the single-state kernels against each
other -- the pairs that describe one model from two sides and are written
separately:

* a sampler and its density: ``sample_next_state`` with
  ``transition_log_probability``, and ``sample_observation`` with
  ``observation_log_probability``. A particle filter draws with one and weights
  with the other; if the density calls a drawn value impossible, every particle
  that took that branch is thrown away;
* one draw and ``n_samples`` draws from the same sampler;
* ``observation_log_probability_per_state`` and a loop over
  ``observation_log_probability``;
* ``sample_next_step`` and the three kernels it combines;
* the initial-state distribution and the transition: both produce states, and
  a belief stacks them into one array.

It also checks that a kernel leaves the state and action it was given
untouched, which every caller that reuses a particle assumes.

Every test is parametrized over ``ENV_BUILDERS`` and uses the ``Environment``
interface only, so each runs on every configuration variant as well.

``constraint_cost_batch`` against ``constraint_cost`` is not here: no
registered environment is a ``ConstrainedEnvironment``. The base class's
default is covered by ``test_core/test_environment/test_constrained_environment.py``.
"""

from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.planners.planners_utils.rollout import random_rollout_action_sampler
from POMDPPlanners.tests.test_environments._env_config_variants import base_id_of
from POMDPPlanners.tests.test_environments._sample_distribution_checks import (
    assert_same_distribution,
)
from POMDPPlanners.tests.test_environments.test_env_api_conformance import (
    ENV_BUILDERS,
    REWARD_RANGE_IN_GRID_ONLY_ENVS,
    EnvBuilder,
    _random_action,
    _seed_all,
    _states_as_rows,
    _trajectory_key,
)

# States a short random rollout visits, per environment.
_ROLLOUT_STEPS = 6

# Transitions checked per environment in the sampler-against-density checks.
_TRANSITIONS_CHECKED = 12

# Draws in the checks that compare two samples.
_DRAWS = 400

# A log-density at or below this is "impossible". Implementations floor an
# impossible value differently (-inf, log(1e-300) = -690.8, -1e18); a value a
# sampler has just drawn must be above all of them.
_IMPOSSIBLE_LOG_DENSITY = -500.0

# Each list below records a real gap found by this suite, as ``env_id:
# reason``. Strict xfail turns the test red the moment it is fixed.

TRANSITION_DENSITY_DISAGREEMENTS: Dict[str, str] = {}
OBSERVATION_DENSITY_DISAGREEMENTS: Dict[str, str] = {}
MULTI_SAMPLE_NEXT_STATE_DISAGREEMENTS: Dict[str, str] = {}
MULTI_SAMPLE_OBSERVATION_DISAGREEMENTS: Dict[str, str] = {}
NEXT_STEP_DISAGREEMENTS: Dict[str, str] = {}
INITIAL_STATE_FORM_DISAGREEMENTS: Dict[str, str] = {}
MUTATED_ARGUMENT_ENVS: Dict[str, str] = {}
PER_STATE_LIKELIHOOD_DISAGREEMENTS: Dict[str, str] = {}


def _params(disagreements: Dict[str, str]) -> List[Any]:
    """One param per registered env, ``xfail(strict=True)`` where a gap is recorded."""
    params = []
    for env_id, builder in ENV_BUILDERS:
        marks = []
        if env_id in disagreements:
            marks.append(pytest.mark.xfail(strict=True, reason=disagreements[env_id]))
        params.append(pytest.param(env_id, builder, id=env_id, marks=marks))
    return params


def _transitions(env: Environment) -> List[Tuple[Any, Any, Any]]:
    """``(state, action, next_state)`` triples from short seeded random rollouts."""
    triples: List[Tuple[Any, Any, Any]] = []
    for seed in (0, 1, 2):
        _seed_all(seed)
        action_rng = np.random.default_rng(seed)
        state = env.initial_state_dist().sample()[0]
        for _ in range(_ROLLOUT_STEPS):
            if env.is_terminal(state):
                break
            action = _random_action(env, action_rng)
            next_state = env.sample_next_state(state=deepcopy(state), action=action)
            triples.append((state, action, next_state))
            state = next_state
    assert triples, f"{type(env).__name__}: every rollout started in a terminal state"
    return triples[:_TRANSITIONS_CHECKED]


# ---------------------------------------------------------------------------
# sampler against density
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id,env_builder", _params(TRANSITION_DENSITY_DISAGREEMENTS))
def test_transition_log_probability_supports_sampled_next_states(
    env_id: str, env_builder: EnvBuilder
) -> None:
    """``transition_log_probability`` gives a drawn next state a possible density.

    Purpose: The sampler and the density are two descriptions of one
        transition model. Planners that reweight particles through the
        transition density discard whatever it calls impossible, so a density
        that rejects a state its own sampler produced -- a slip branch it
        forgot, a clip it applies and the sampler does not -- removes real
        trajectories from the belief. No general test called this method.

    Given: ``(state, action, next_state)`` triples from seeded random
        rollouts, the next state drawn by ``sample_next_state``.
    When: ``transition_log_probability`` scores each drawn next state.
    Then: It returns one finite value per next state, above
        ``_IMPOSSIBLE_LOG_DENSITY``. An environment whose density is not
        implemented skips, with the method's own message.

    Test type: integration
    """
    env = env_builder()
    for state, action, next_state in _transitions(env):
        try:
            values = env.transition_log_probability(deepcopy(state), action, [deepcopy(next_state)])
        except NotImplementedError as error:
            pytest.skip(f"{env_id}.transition_log_probability is not implemented: {error}")
        values = np.asarray(values, dtype=np.float64).ravel()
        assert values.shape == (1,), (
            f"{env_id}.transition_log_probability returned shape {values.shape} for one "
            "next state"
        )
        assert np.isfinite(values[0]) and values[0] > _IMPOSSIBLE_LOG_DENSITY, (
            f"{env_id}.transition_log_probability gives {values[0]} to a next state that "
            f"sample_next_state drew: state={state!r} action={action!r} "
            f"next_state={next_state!r}"
        )


@pytest.mark.parametrize("env_id,env_builder", _params(OBSERVATION_DENSITY_DISAGREEMENTS))
def test_observation_log_probability_supports_sampled_observations(
    env_id: str, env_builder: EnvBuilder
) -> None:
    """``observation_log_probability`` gives a drawn observation a possible likelihood.

    Purpose: A belief update weights each particle by the likelihood of the
        observation the environment emitted. If that likelihood is impossible
        at the very state the observation was drawn from, the true state's
        particle gets weight zero and the belief moves off the truth on that
        step.

    Given: Next states from seeded random rollouts, each with an observation
        drawn from it by ``sample_observation``.
    When: ``observation_log_probability`` scores each observation at the
        state it was drawn from.
    Then: The value is finite and above ``_IMPOSSIBLE_LOG_DENSITY``.

    Test type: integration
    """
    env = env_builder()
    for index, (_, action, next_state) in enumerate(_transitions(env)):
        _seed_all(100 + index)
        observation = env.sample_observation(next_state=deepcopy(next_state), action=action)
        value = float(
            np.asarray(
                env.observation_log_probability(
                    next_state=deepcopy(next_state), action=action, observations=[observation]
                )
            ).ravel()[0]
        )
        assert np.isfinite(value) and value > _IMPOSSIBLE_LOG_DENSITY, (
            f"{env_id}.observation_log_probability gives {value} to an observation that "
            f"sample_observation drew from the same state: next_state={next_state!r} "
            f"action={action!r} observation={observation!r}"
        )


@pytest.mark.parametrize("env_id,env_builder", _params(PER_STATE_LIKELIHOOD_DISAGREEMENTS))
def test_observation_log_probability_per_state_agrees_with_looped_likelihood(
    env_id: str, env_builder: EnvBuilder
) -> None:
    """The per-state likelihood hook matches a loop over ``observation_log_probability``.

    Purpose: ``WeightedParticleBelief`` reweights through
        ``observation_log_probability_per_state``, and several environments
        override it with a native or vectorized kernel. That is a second
        implementation of the observation model on the scalar belief's hot
        path; where it differs from ``observation_log_probability``, the
        belief and the search weigh the same reading differently.

    Given: Next states from seeded random rollouts, one action, and an
        observation drawn from one of them.
    When: ``observation_log_probability_per_state`` scores the observation
        against all of them, and ``observation_log_probability`` scores it
        against each in turn.
    Then: The two agree to 1e-6 wherever either is above
        ``_IMPOSSIBLE_LOG_DENSITY``.

    Test type: integration
    """
    env = env_builder()
    triples = _transitions(env)
    next_states = [next_state for _, _, next_state in triples]
    stacked: Any = next_states
    if isinstance(next_states[0], np.ndarray):
        stacked = np.stack(next_states)

    for index in (0, len(triples) // 2):
        _, action, observed_from = triples[index]
        _seed_all(200 + index)
        observation = env.sample_observation(next_state=deepcopy(observed_from), action=action)
        per_state = np.asarray(
            env.observation_log_probability_per_state(
                next_states=deepcopy(stacked), action=action, observation=observation
            ),
            dtype=np.float64,
        )
        looped = np.array(
            [
                float(
                    np.asarray(
                        env.observation_log_probability(
                            next_state=deepcopy(s), action=action, observations=[observation]
                        )
                    ).ravel()[0]
                )
                for s in next_states
            ]
        )
        assert per_state.shape == looped.shape, (
            f"{env_id}.observation_log_probability_per_state returned shape {per_state.shape} "
            f"for {len(next_states)} states"
        )
        assert not np.any(np.isnan(per_state)), f"{env_id}: per-state likelihood has NaN"
        relevant = (per_state > _IMPOSSIBLE_LOG_DENSITY) | (looped > _IMPOSSIBLE_LOG_DENSITY)
        np.testing.assert_allclose(
            per_state[relevant],
            looped[relevant],
            rtol=1e-6,
            atol=1e-6,
            err_msg=(
                f"{env_id} action={action!r} observation={observation!r}: "
                "observation_log_probability_per_state disagrees with the looped likelihood"
            ),
        )


# ---------------------------------------------------------------------------
# one draw against many
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id,env_builder", _params(MULTI_SAMPLE_NEXT_STATE_DISAGREEMENTS))
def test_sample_next_state_with_n_samples_matches_repeated_single_draws(
    env_id: str, env_builder: EnvBuilder
) -> None:
    """``sample_next_state(n_samples=n)`` returns ``n`` draws from the single-draw law.

    Purpose: ``n_samples`` is its own code path in most environments -- a
        vectorized draw, or a loop that reuses state between iterations.
        Sparse-sampling planners expand a node with it. If it returns another
        count, a draw of another shape, or draws from another distribution,
        those planners search a different model from the ones that draw one
        state at a time.

    Given: Two reachable ``(state, action)`` pairs.
    When: ``sample_next_state`` is called once with ``n_samples=_DRAWS`` and
        ``_DRAWS`` times with the default, from independent seeds.
    Then: The first call returns ``_DRAWS`` states, each of the single draw's
        shape, and the two samples have the same distribution.

    Test type: integration
    """
    env = env_builder()
    triples = _transitions(env)
    for state, action, _ in (triples[0], triples[len(triples) // 2]):
        _seed_all(1)
        many = list(env.sample_next_state(state=deepcopy(state), action=action, n_samples=_DRAWS))
        _seed_all(2)
        singles = [
            env.sample_next_state(state=deepcopy(state), action=action) for _ in range(_DRAWS)
        ]
        context = f"{env_id} state={state!r} action={action!r}"
        assert len(many) == _DRAWS, f"{context}: n_samples={_DRAWS} returned {len(many)} states"
        assert np.asarray(many[0]).shape == np.asarray(singles[0]).shape, (
            f"{context}: a state drawn with n_samples has shape {np.asarray(many[0]).shape}, "
            f"a single draw has shape {np.asarray(singles[0]).shape}"
        )
        many_rows, single_rows = _states_as_rows(many, singles)
        try:
            assert_same_distribution(many_rows, single_rows, label_a="n_samples", label_b="single")
        except AssertionError as error:
            raise AssertionError(f"{context}: {error}") from None


@pytest.mark.parametrize("env_id,env_builder", _params(MULTI_SAMPLE_OBSERVATION_DISAGREEMENTS))
def test_sample_observation_with_n_samples_matches_repeated_single_draws(
    env_id: str, env_builder: EnvBuilder
) -> None:
    """``sample_observation(n_samples=n)`` returns ``n`` draws from the single-draw law.

    Purpose: The same second code path as in ``sample_next_state``, on the
        observation side. Observation-widening planners draw several readings
        per expanded state with it.

    Given: Two reachable ``(next_state, action)`` pairs.
    When: ``sample_observation`` is called once with ``n_samples=_DRAWS`` and
        ``_DRAWS`` times with the default, from independent seeds.
    Then: The first call returns ``_DRAWS`` observations and the two samples
        have the same distribution.

    Test type: integration
    """
    env = env_builder()
    triples = _transitions(env)
    for _, action, next_state in (triples[0], triples[len(triples) // 2]):
        _seed_all(1)
        many = list(
            env.sample_observation(next_state=deepcopy(next_state), action=action, n_samples=_DRAWS)
        )
        _seed_all(2)
        singles = [
            env.sample_observation(next_state=deepcopy(next_state), action=action)
            for _ in range(_DRAWS)
        ]
        context = f"{env_id} next_state={next_state!r} action={action!r}"
        assert (
            len(many) == _DRAWS
        ), f"{context}: n_samples={_DRAWS} returned {len(many)} observations"
        many_rows, single_rows = _states_as_rows(many, singles)
        try:
            assert_same_distribution(many_rows, single_rows, label_a="n_samples", label_b="single")
        except AssertionError as error:
            raise AssertionError(f"{context}: {error}") from None


# ---------------------------------------------------------------------------
# sample_next_step and the kernels it combines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id,env_builder", _params(NEXT_STEP_DISAGREEMENTS))
def test_sample_next_step_scores_the_next_state_it_returns(
    env_id: str, env_builder: EnvBuilder
) -> None:
    """``sample_next_step`` returns the reward of the transition it returns.

    Purpose: ``sample_next_step`` is what an episode and a rollout advance
        with. Its three results have to describe one transition: a reward
        scored against another draw than the returned next state -- a second
        transition sampled inside ``reward``, a hazard rolled twice -- gives
        a trajectory whose return does not belong to its states.

    Given: Two reachable ``(state, action)`` pairs.
    When: ``sample_next_step`` is called ``_DRAWS`` times, and ``reward`` is
        then called on each returned ``(state, action, next_state)``.
    Then: Where ``reward`` is a function of the transition -- recomputing it
        twice from two seeds gives the same numbers -- each returned reward
        equals the reward of its own returned next state, draw by draw. Where
        ``reward`` draws, the returned and the recomputed rewards have the
        same distribution.

    Test type: integration
    """
    env = env_builder()
    triples = _transitions(env)
    for state, action, _ in (triples[0], triples[len(triples) // 2]):
        _seed_all(1)
        steps = [env.sample_next_step(deepcopy(state), action) for _ in range(_DRAWS)]
        returned = np.array([float(reward) for _, _, reward in steps])

        def recompute(seed: int) -> np.ndarray:
            _seed_all(seed)
            return np.array(
                [
                    float(
                        env.reward(
                            deepcopy(state),  # pylint: disable=cell-var-from-loop
                            action,  # pylint: disable=cell-var-from-loop
                            deepcopy(next_state),
                        )
                    )
                    for next_state, _, _ in steps  # pylint: disable=cell-var-from-loop
                ]
            )

        recomputed = recompute(2)
        if np.array_equal(recomputed, recompute(3)):
            # Paired: a reward scored against a second, hidden draw has the
            # same law as the right one and only differs draw by draw.
            mismatched = np.flatnonzero(~np.isclose(returned, recomputed))
            assert mismatched.size == 0, (
                f"{env_id} state={state!r} action={action!r}: sample_next_step returned a "
                f"reward that is not the reward of its own next state on {mismatched.size} "
                f"of {_DRAWS} draws, first {returned[mismatched[0]]} against "
                f"{recomputed[mismatched[0]]}"
            )
            continue
        try:
            assert_same_distribution(
                returned[:, None],
                recomputed[:, None],
                label_a="sample_next_step",
                label_b="reward on its next state",
            )
        except AssertionError as error:
            raise AssertionError(f"{env_id} state={state!r} action={action!r}: {error}") from None


# ---------------------------------------------------------------------------
# the form of a state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id,env_builder", _params(INITIAL_STATE_FORM_DISAGREEMENTS))
def test_transitioned_states_fit_the_form_of_an_initial_state(
    env_id: str, env_builder: EnvBuilder
) -> None:
    """A transitioned state has an initial state's type and shape and survives its dtype.

    Purpose: A belief is built from initial states and then filled with
        transitioned ones. A vectorized belief takes its array's dtype from
        the first: if the initial state is an integer array and a later state
        holds fractions, each later state written into the array is truncated.
        A scalar belief holds a mix of two kinds of state, which compare and
        hash differently.

    Given: An initial state and the next states of seeded random rollouts.
    When: Each next state is compared with the initial state's type and
        shape, and cast to the initial state's dtype.
    Then: Type and shape are the same, and the cast changes no value. An
        integer initial state with integer-valued float successors passes: it
        loses nothing.

    Test type: integration
    """
    env = env_builder()
    _seed_all(0)
    initial = env.initial_state_dist().sample()[0]
    initial_array = np.asarray(initial)
    for _, _, following in _transitions(env):
        following_array = np.asarray(following)
        assert type(following) is type(initial), (
            f"{env_id}: initial_state_dist returns a {type(initial).__name__} and "
            f"sample_next_state a {type(following).__name__}"
        )
        assert following_array.shape == initial_array.shape, (
            f"{env_id}: an initial state has shape {initial_array.shape} and a transitioned "
            f"state has shape {following_array.shape}"
        )
        if initial_array.dtype.kind not in "iuf" or following_array.dtype.kind not in "iuf":
            continue
        stored = following_array.astype(initial_array.dtype)
        assert np.array_equal(stored, following_array), (
            f"{env_id}: an initial state has dtype {initial_array.dtype}; the transitioned "
            f"state {following!r} becomes {stored!r} when stored with it"
        )


# ---------------------------------------------------------------------------
# kernels leave their arguments alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id,env_builder", _params(MUTATED_ARGUMENT_ENVS))
def test_kernels_do_not_mutate_their_arguments(env_id: str, env_builder: EnvBuilder) -> None:
    """No kernel changes the state, next state or action it is given.

    Purpose: Callers reuse what they pass. A tree node keeps its state and
        expands it again; a particle filter passes the same array to the
        transition and then to the reward; an action object is shared by every
        node that took it. A kernel that writes into its argument corrupts all
        of those, and the damage shows up far from the call that caused it.

    Given: A reachable ``(state, action, next_state)`` triple, with copies
        taken before any call.
    When: Each kernel is called on it: ``sample_next_state``,
        ``sample_observation``, ``reward`` with and without the next state,
        ``observation_log_probability`` and ``is_terminal``.
    Then: After each call the state, the action and the next state are
        unchanged.

    Test type: integration
    """
    env = env_builder()
    state, action, next_state = _transitions(env)[0]
    _seed_all(0)
    observation = env.sample_observation(next_state=deepcopy(next_state), action=action)
    before = tuple(_trajectory_key(value) for value in (state, action, next_state))

    calls = {
        "sample_next_state": lambda: env.sample_next_state(state=state, action=action),
        "sample_observation": lambda: env.sample_observation(next_state=next_state, action=action),
        "reward(state, action)": lambda: env.reward(state, action),
        "reward(state, action, next_state)": lambda: env.reward(state, action, next_state),
        "observation_log_probability": lambda: env.observation_log_probability(
            next_state=next_state, action=action, observations=[observation]
        ),
        "is_terminal": lambda: env.is_terminal(state),
    }
    for name, call in calls.items():
        _seed_all(1)
        call()
        after = tuple(_trajectory_key(value) for value in (state, action, next_state))
        changed = [
            label
            for label, old, new in zip(("state", "action", "next_state"), before, after)
            if old != new
        ]
        assert not changed, f"{env_id}.{name} changed its {' and '.join(changed)} argument"


# ---------------------------------------------------------------------------
# random rollout
# ---------------------------------------------------------------------------

# Depth of the leaf-value rollouts below, and the seeds they start from.
ROLLOUT_MAX_DEPTH = 10
_ROLLOUT_SEEDS = (0, 1, 2, 3)


class _UniformActionSampler(ActionSampler):
    """Draws actions the way the harness rollouts do, from a seeded generator."""

    def __init__(self, env: Environment, seed: int = 0):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def sample(self, belief_node: Any = None) -> Any:
        return _random_action(self.env, self.rng)


def leaf_rollout(env: Environment, state: Any, depth: int = 0, seed: int = 0) -> float:
    """The leaf-value rollout a planner runs from ``state``.

    Goes through ``random_rollout_action_sampler``, the entry point the MCTS
    planners call, so an env with a native ``simulate_random_rollout`` is
    checked on that and every other env on the Python fallback.
    """
    _seed_all(seed)
    return random_rollout_action_sampler(
        state=deepcopy(state),
        depth=depth,
        action_sampler=_UniformActionSampler(env, seed),
        environment=env,
        discount_factor=env.discount_factor,
        max_depth=ROLLOUT_MAX_DEPTH,
    )


@pytest.mark.parametrize("env_id,env_builder", _params({}))
def test_random_rollout_returns_zero_at_max_depth(env_id: str, env_builder: EnvBuilder) -> None:
    """A rollout that starts at the depth limit returns exactly 0.

    Purpose: A planner calls the rollout at whatever depth the tree has
        reached. At the limit there are no steps left, so any non-zero value
        is a reward the search never earned. Native rollouts compute the
        steps left themselves; each env test file checked this for its own
        override.

    Given: A non-terminal state reached by a short rollout.
    When: The leaf rollout runs with ``depth == max_depth``, and with
        ``depth > max_depth``.
    Then: Both return 0.0.

    Test type: unit
    """
    env = env_builder()
    state = _transitions(env)[0][0]
    assert leaf_rollout(env, state, depth=ROLLOUT_MAX_DEPTH) == 0.0
    assert leaf_rollout(env, state, depth=ROLLOUT_MAX_DEPTH + 1) == 0.0


@pytest.mark.parametrize("env_id,env_builder", _params({}))
def test_random_rollout_returns_a_finite_bounded_return(
    env_id: str, env_builder: EnvBuilder
) -> None:
    """A rollout's return is finite and inside what the reward range allows.

    Purpose: The rollout value is added straight into a node's Q estimate.
        A NaN poisons every ancestor, and a value beyond
        ``max|reward| * sum(gamma^k)`` means the rollout scored rewards
        the environment says it cannot pay -- a native kernel whose reward
        has drifted from the Python one.

    Given: Non-terminal states reached by short rollouts.
    When: The leaf rollout runs from each, from depth 0.
    Then: Every return is finite, and when the env declares a reward range,
        its absolute value is at most ``max(|r_min|, |r_max|)`` times the
        discounted horizon. The bound is skipped for the light-dark envs
        whose range is declared over in-grid states only (see
        ``REWARD_RANGE_IN_GRID_ONLY_ENVS``).

    Test type: integration
    """
    env = env_builder()
    states = [triple[0] for triple in _transitions(env)][: len(_ROLLOUT_SEEDS)]
    bounded = env.reward_range is not None and base_id_of(env_id) not in (
        REWARD_RANGE_IN_GRID_ONLY_ENVS
    )
    if bounded:
        horizon = sum(env.discount_factor**k for k in range(ROLLOUT_MAX_DEPTH))
        limit = max(abs(env.reward_range[0]), abs(env.reward_range[1])) * horizon
    for state, seed in zip(states, _ROLLOUT_SEEDS):
        value = leaf_rollout(env, state, seed=seed)
        assert np.isfinite(value), f"{env_id}: rollout from {state!r} returned {value}"
        if bounded:
            assert abs(value) <= limit + 1e-9, (
                f"{env_id}: rollout from {state!r} returned {value}, beyond the "
                f"{limit} that reward_range {env.reward_range} allows over "
                f"{ROLLOUT_MAX_DEPTH} steps"
            )
