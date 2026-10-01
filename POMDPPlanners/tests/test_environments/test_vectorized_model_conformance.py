# SPDX-License-Identifier: MIT

"""Vectorized generative models agree with the scalar environment they duplicate.

A :class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
re-expresses an environment's transition, observation, reward and terminal
kernels in torch so VOPP can search on a batch at once. It is a hand-written
copy of the scalar :class:`~POMDPPlanners.core.environment.Environment`, and a
planner that searches on a copy that has drifted optimizes a different problem
from the one it is evaluated on -- with nothing at runtime to show it.

This file checks every model in
:data:`~POMDPPlanners.tests.test_environments._vectorized_registry.MODEL_SPECS`
the same way, through the protocol methods only:

* ``observation_log_probs`` against ``observation_log_probability``, and
  ``terminal_mask`` against ``is_terminal`` -- exact, since both are
  deterministic functions;
* ``rewards`` against ``reward`` -- by distribution over repeated calls, which
  is exact for a deterministic reward (every draw is the same value, and a
  constant column is compared exactly) and still meaningful for the
  environments whose reward draws a hazard roll;
* ``sample_next_states`` and ``sample_observations`` against
  ``sample_next_state`` and ``sample_observation`` -- by distribution.

The only per-environment knowledge is the registry's conversion between the
model's tensors and the scalar environment's values. Models whose simulator
dependency is missing are skipped.
"""

from typing import Any, List, Tuple

import numpy as np
import pytest
import torch

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.tests.test_environments._sample_distribution_checks import (
    assert_same_distribution,
)
from POMDPPlanners.tests.test_environments._vectorized_registry import (
    MODEL_CLASS_EXCLUSIONS,
    MODEL_SPECS,
    NEXT_STATE_DISTRIBUTION_DISAGREEMENTS,
    OBSERVATION_DISTRIBUTION_DISAGREEMENTS,
    OBSERVATION_LOG_PROB_DISAGREEMENTS,
    REWARD_DISAGREEMENTS,
    TERMINAL_MASK_DISAGREEMENTS,
    ModelSpec,
    discover_vectorized_classes,
    missing_module,
)
from POMDPPlanners.tests.test_environments.test_env_api_conformance import (
    _seed_all as _seed_scalar,
)

_ROLLOUT_SEEDS = (0, 1, 2)
_ROLLOUT_STEPS = 6

# Action indices checked per state; see _pairs.
_MAX_ACTIONS_CHECKED = 8

# Draws per (state, action) in the distribution checks.
_DRAWS = 2000

# See test_vectorized_belief_conformance._NEGLIGIBLE_LOG_LIKELIHOOD.
_NEGLIGIBLE_LOG_LIKELIHOOD = -500.0


def _seed_all(seed: int) -> None:
    """Seed the scalar env's RNG streams and torch's."""
    _seed_scalar(seed)
    torch.manual_seed(seed)


def _params(disagreements: dict) -> List[Any]:
    params = []
    for spec in MODEL_SPECS:
        marks = []
        if spec.model_id in disagreements:
            reason, raises = disagreements[spec.model_id]
            marks.append(pytest.mark.xfail(strict=True, reason=reason, raises=raises))
        params.append(pytest.param(spec, id=spec.model_id, marks=marks))
    return params


# ---------------------------------------------------------------------------
# Helpers -- protocol and registry conversions only
# ---------------------------------------------------------------------------


def _build(spec: ModelSpec) -> Tuple[Environment, Any]:
    missing = missing_module(spec.required_modules)
    if missing is not None:
        pytest.skip(f"{spec.model_id} needs the optional module {missing!r}")
    try:
        return spec.build()
    except ImportError as error:
        pytest.skip(f"{spec.model_id} needs an optional dependency: {error}")


def _initial_state(spec: ModelSpec, env: Environment, seed: int) -> Any:
    if spec.initial_state is not None:
        return spec.initial_state(env, np.random.default_rng(seed))
    return env.initial_state_dist().sample()[0]


def _rollout_states(spec: ModelSpec, env: Environment, model: Any) -> Tuple[List[Any], List[Any]]:
    """Reachable states, split into non-terminal ones and every one visited.

    The terminal-mask check needs the terminal states a rollout reaches; the
    kernel checks need states a step can start from.
    """
    live: List[Any] = []
    visited: List[Any] = []
    if spec.probe_states is not None:
        for probe in spec.probe_states(env):
            visited.append(probe)
            if not env.is_terminal(probe):
                live.append(probe)
    n_actions = spec.num_actions(env, model)
    for seed in _ROLLOUT_SEEDS:
        _seed_all(seed)
        action_rng = np.random.default_rng(seed)
        state = _initial_state(spec, env, seed)
        for _ in range(_ROLLOUT_STEPS):
            visited.append(state)
            if env.is_terminal(state):
                break
            live.append(state)
            action = spec.action_of_index(env, model, int(action_rng.integers(n_actions)))
            state = env.sample_next_state(state=state, action=action)
        else:
            visited.append(state)
    assert live, f"{spec.model_id}: every rollout started in a terminal state"
    return live, visited


def _pairs(
    spec: ModelSpec, env: Environment, model: Any, states: List[Any], n_states: int
) -> List[Tuple[Any, int]]:
    """(state, action index) pairs: ``n_states`` states, each with every checked action.

    Every action index is checked when there are at most ``_MAX_ACTIONS_CHECKED``;
    a larger set gets a fixed random subset of that size.
    """
    rng = np.random.default_rng(1234)
    picks = np.linspace(0, len(states) - 1, num=n_states).round().astype(int)
    n_actions = spec.num_actions(env, model)
    indices = list(range(n_actions))
    if n_actions > _MAX_ACTIONS_CHECKED:
        indices = sorted(int(i) for i in rng.choice(n_actions, _MAX_ACTIONS_CHECKED, replace=False))
    return [(states[i], index) for i in picks for index in indices]


def _tensor(rows: Any, model: Any) -> torch.Tensor:
    return torch.as_tensor(
        np.asarray(rows, dtype=np.float64), dtype=torch.float64, device=model.device
    )


def _actions(index: int, count: int, model: Any) -> torch.Tensor:
    return torch.full((count,), index, dtype=torch.int64, device=model.device)


def _numpy(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy().astype(np.float64)


def _copy(state: Any) -> Any:
    return np.array(state, copy=True) if isinstance(state, np.ndarray) else state


# ---------------------------------------------------------------------------
# Exact kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", _params(OBSERVATION_LOG_PROB_DISAGREEMENTS))
def test_observation_log_probs_match_scalar_env(spec: ModelSpec) -> None:
    """The model scores an observation exactly as the scalar environment does.

    Purpose: VOPP weights its particles with ``observation_log_probs``. A
        likelihood that differs from the environment's observation model makes
        the planner's belief track a different sensor than the one the episode
        runs.

    Given: Reachable states and actions, candidate next states from the scalar
        env, and observations from both the scalar env and the model's own
        sampler (converted to the other side's encoding by the registry).
    When: ``observation_log_probs`` scores every (candidate, observation) row
        and the scalar ``observation_log_probability`` scores the same pairs.
    Then: The two agree to 1e-6 wherever either is above
        ``_NEGLIGIBLE_LOG_LIKELIHOOD``.

    Test type: integration
    """
    if spec.observation_not_comparable is not None:
        pytest.skip(f"{spec.model_id}: {spec.observation_not_comparable}")
    env, model = _build(spec)
    live, _ = _rollout_states(spec, env, model)

    for state, index in _pairs(spec, env, model, live, n_states=3):
        action = spec.action_of_index(env, model, index)
        _seed_all(7)
        candidates = [env.sample_next_state(state=_copy(s), action=action) for s in live[:16]]
        true_next = env.sample_next_state(state=_copy(state), action=action)
        candidates.insert(0, true_next)
        rows = np.stack([spec.row_of_state(env, s) for s in candidates])

        scalar_observation = env.sample_observation(next_state=true_next, action=action)
        model_observation_row = _numpy(
            model.sample_observations(_tensor(rows[:1], model), _actions(index, 1, model))
        )[0]
        for observation_row in (
            spec.row_of_observation(env, scalar_observation),
            model_observation_row,
        ):
            observation = spec.observation_of_row(env, observation_row)
            vectorized = _numpy(
                model.observation_log_probs(
                    _tensor(rows, model),
                    _actions(index, len(rows), model),
                    _tensor(np.repeat(observation_row[None, :], len(rows), axis=0), model),
                )
            )
            scalar = np.array(
                [
                    float(
                        env.observation_log_probability(
                            next_state=s, action=action, observations=[observation]
                        )[0]
                    )
                    for s in candidates
                ]
            )
            context = f"{spec.model_id} action={action!r} observation={observation!r}"
            assert not np.any(np.isnan(vectorized)), f"{context}: NaN log-probability"
            relevant = (vectorized > _NEGLIGIBLE_LOG_LIKELIHOOD) | (
                scalar > _NEGLIGIBLE_LOG_LIKELIHOOD
            )
            assert np.any(relevant), f"{context}: impossible under every candidate"
            np.testing.assert_allclose(
                vectorized[relevant],
                scalar[relevant],
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"{context}: observation_log_probs disagrees with the scalar env",
            )


@pytest.mark.parametrize("spec", _params(TERMINAL_MASK_DISAGREEMENTS))
def test_terminal_mask_matches_scalar_env(spec: ModelSpec) -> None:
    """``terminal_mask`` flags exactly the states ``is_terminal`` does.

    Purpose: A state the model thinks is live keeps accruing value in the
        search tree after the episode would have ended; one it thinks is
        terminal stops the search short. Both change the action chosen.

    Given: Every state visited by several random rollouts, including the
        terminal state a rollout ends in.
    When: ``terminal_mask`` and ``is_terminal`` classify them.
    Then: The classifications are identical.

    Test type: integration
    """
    env, model = _build(spec)
    _, visited = _rollout_states(spec, env, model)
    rows = np.stack([spec.row_of_state(env, s) for s in visited])
    vectorized = model.terminal_mask(_tensor(rows, model)).detach().cpu().numpy().astype(bool)
    scalar = np.array([bool(env.is_terminal(s)) for s in visited])
    np.testing.assert_array_equal(
        vectorized, scalar, err_msg=f"{spec.model_id}: terminal_mask disagrees with is_terminal"
    )


@pytest.mark.parametrize("spec", _params(REWARD_DISAGREEMENTS))
def test_rewards_match_scalar_env(spec: ModelSpec) -> None:
    """``rewards`` scores a transition as ``reward`` does.

    Purpose: The reward is what the planner maximizes. A model whose reward
        differs -- a term scored on the wrong state, a penalty missing --
        optimizes a different objective from the one the run is judged on.

    Given: Reachable (state, action) pairs, each with a next state the scalar
        env sampled, the triple repeated ``_DRAWS`` times.
    When: ``rewards`` scores the repeated block and ``reward(state, action,
        next_state)`` is called once per row.
    Then: The two reward samples have the same distribution. For a
        deterministic reward that means the same value exactly; for one with a
        hazard roll it means the same law.

    Test type: integration
    """
    env, model = _build(spec)
    live, _ = _rollout_states(spec, env, model)

    for state, index in _pairs(spec, env, model, live, n_states=3):
        action = spec.action_of_index(env, model, index)
        _seed_all(17)
        next_state = env.sample_next_state(state=_copy(state), action=action)
        states = np.repeat(spec.row_of_state(env, state)[None, :], _DRAWS, axis=0)
        next_states = np.repeat(spec.row_of_state(env, next_state)[None, :], _DRAWS, axis=0)
        _seed_all(18)
        vectorized = _numpy(
            model.rewards(
                _tensor(states, model), _actions(index, _DRAWS, model), _tensor(next_states, model)
            )
        )
        _seed_all(19)
        scalar = np.array(
            [float(env.reward(_copy(state), action, _copy(next_state))) for _ in range(_DRAWS)]
        )
        try:
            assert_same_distribution(vectorized[:, None], scalar[:, None])
        except AssertionError as error:
            raise AssertionError(
                f"{spec.model_id} action={action!r} state={state!r} next_state={next_state!r}: "
                f"{error}"
            ) from None


# ---------------------------------------------------------------------------
# Stochastic kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", _params(NEXT_STATE_DISTRIBUTION_DISAGREEMENTS))
def test_sample_next_states_distribution_matches_scalar_env(spec: ModelSpec) -> None:
    """``sample_next_states`` samples the scalar environment's transition law.

    Purpose: The search tree is grown from these samples. A missing noise
        term or a wrong clip plans against dynamics the episode does not have.

    Given: Several reachable states, each with one action, each state
        repeated ``_DRAWS`` times.
    When: The block goes through ``sample_next_states`` once and through
        ``sample_next_state`` once per row, from independent seeds.
    Then: The two samples have the same distribution: deterministic
        components agree exactly, the rest pass a chi-square or
        Kolmogorov-Smirnov test at a fixed family-wise level.

    Test type: integration
    """
    env, model = _build(spec)
    live, _ = _rollout_states(spec, env, model)

    for state, index in _pairs(spec, env, model, live, n_states=2):
        action = spec.action_of_index(env, model, index)
        block = np.repeat(spec.row_of_state(env, state)[None, :], _DRAWS, axis=0)
        _seed_all(21)
        vectorized = _numpy(
            model.sample_next_states(_tensor(block, model), _actions(index, _DRAWS, model))
        )
        _seed_all(22)
        scalar = np.stack(
            [
                spec.row_of_state(env, env.sample_next_state(state=_copy(state), action=action))
                for _ in range(_DRAWS)
            ]
        )
        try:
            assert_same_distribution(vectorized, scalar)
        except AssertionError as error:
            raise AssertionError(f"{spec.model_id} action={action!r}: {error}") from None


@pytest.mark.parametrize("spec", _params(OBSERVATION_DISTRIBUTION_DISAGREEMENTS))
def test_sample_observations_distribution_matches_scalar_env(spec: ModelSpec) -> None:
    """``sample_observations`` samples the scalar environment's observation law.

    Purpose: The model's observations decide which branches the search
        expands. If they are drawn from a different sensor than the episode's,
        the planner prepares for readings it will not get.

    Given: Several reachable next states, each with one action, each repeated
        ``_DRAWS`` times.
    When: The block goes through ``sample_observations`` once and through
        ``sample_observation`` once per row, from independent seeds.
    Then: The two observation samples, in the model's flat encoding, have the
        same distribution.

    Test type: integration
    """
    if spec.observation_not_comparable is not None:
        pytest.skip(f"{spec.model_id}: {spec.observation_not_comparable}")
    env, model = _build(spec)
    live, _ = _rollout_states(spec, env, model)

    for state, index in _pairs(spec, env, model, live, n_states=2):
        action = spec.action_of_index(env, model, index)
        _seed_all(31)
        next_state = env.sample_next_state(state=_copy(state), action=action)
        block = np.repeat(spec.row_of_state(env, next_state)[None, :], _DRAWS, axis=0)
        _seed_all(32)
        vectorized = _numpy(
            model.sample_observations(_tensor(block, model), _actions(index, _DRAWS, model))
        )
        _seed_all(33)
        scalar = np.stack(
            [
                spec.row_of_observation(
                    env, env.sample_observation(next_state=_copy(next_state), action=action)
                )
                for _ in range(_DRAWS)
            ]
        )
        try:
            assert_same_distribution(vectorized, scalar)
        except AssertionError as error:
            raise AssertionError(f"{spec.model_id} action={action!r}: {error}") from None


# ---------------------------------------------------------------------------
# Registry coverage
# ---------------------------------------------------------------------------


def test_every_vectorized_model_is_registered_or_excluded() -> None:
    """No vectorized generative model escapes this suite silently.

    Purpose: A new model gets no parity coverage until someone registers it,
        and nothing fails when they forget. The source tree is scanned for
        every class that implements the protocol's kernels, so a new one fails
        here until it is registered or excluded with a stated reason.

    Given: Every class under ``POMDPPlanners/environments`` defining all of
        the protocol's sampling, reward, terminal and likelihood methods.
    When: Each is looked up in ``MODEL_SPECS`` and ``MODEL_CLASS_EXCLUSIONS``.
    Then: Each is in exactly one, and no exclusion names a missing class.

    Test type: unit
    """
    found = discover_vectorized_classes().models
    registered = {spec.model_class for spec in MODEL_SPECS}
    unregistered = sorted(found - registered - set(MODEL_CLASS_EXCLUSIONS))
    assert not unregistered, (
        f"Vectorized models with no conformance coverage: {unregistered}. Add a ModelSpec "
        "in _vectorized_registry.py, or add the class to MODEL_CLASS_EXCLUSIONS with the "
        "reason it needs none."
    )
    stale = sorted(set(MODEL_CLASS_EXCLUSIONS) - found)
    assert not stale, f"MODEL_CLASS_EXCLUSIONS names classes that no longer exist: {stale}"
    both = sorted(set(MODEL_CLASS_EXCLUSIONS) & registered)
    assert not both, f"Classes both registered and excluded: {both}"
    missing = sorted(registered - found)
    assert not missing, f"MODEL_SPECS names classes the source scan does not find: {missing}"


@pytest.mark.parametrize("spec", [pytest.param(s, id=s.model_id) for s in MODEL_SPECS])
def test_model_spec_builds_the_registered_class(spec: ModelSpec) -> None:
    """Each registry entry builds the model class it claims to cover.

    Purpose: The coverage test trusts ``model_class``; an entry that builds a
        different class would make that coverage claim false.

    Given: The registry entry.
    When: Its builder runs.
    Then: The model is an instance of the named class.

    Test type: unit
    """
    _, model = _build(spec)
    assert type(model).__name__ == spec.model_class
