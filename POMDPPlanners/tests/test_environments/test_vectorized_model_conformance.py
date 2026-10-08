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

Rollouts stop at the first terminal state, and a short one seldom reaches it,
so those checks start from live states only. A search does not: VOPP steps and
scores whole batches, finished rows included, and relies on ``terminal_mask``
to discount them. The last group of checks repeats the kernel comparisons on
terminal states and on the step that enters one, and the ``terminal_mask``
check is required to see both a terminal and a live state. The terminal states
come from ``_terminal_states``.

The only per-environment knowledge is the registry's conversion between the
model's tensors and the scalar environment's values. Models whose simulator
dependency is missing are skipped.
"""

from typing import Any, List, Tuple

import numpy as np
import pytest
import torch

from POMDPPlanners.core.environment import DiscreteActionsEnvironment, Environment
from POMDPPlanners.core.environment.vectorized_generative_model import VectorizedGenerativeModel
from POMDPPlanners.tests.test_environments._sample_distribution_checks import (
    assert_same_distribution,
)
from POMDPPlanners.tests.test_environments._env_config_variants import base_id_of
from POMDPPlanners.tests.test_environments._terminal_states import (
    ENVS_WITHOUT_TERMINAL_STATES,
    TerminalTransition,
    find_terminal_transitions,
    probe_terminal_states,
    probe_terminal_transitions,
)
from POMDPPlanners.tests.test_environments._vectorized_registry import (
    MODEL_CLASS_EXCLUSIONS,
    MODEL_SPECS,
    MODELS_WITHOUT_TERMINAL_STATES,
    MODEL_VARIANT_DECLINES,
    NEXT_STATE_DISTRIBUTION_DISAGREEMENTS,
    OBSERVATION_DISTRIBUTION_DISAGREEMENTS,
    OBSERVATION_LOG_PROB_DISAGREEMENTS,
    REWARD_DISAGREEMENTS,
    TERMINAL_ENTRY_REWARD_DISAGREEMENTS,
    TERMINAL_MASK_DISAGREEMENTS,
    TERMINAL_NEXT_STATE_DISAGREEMENTS,
    TERMINAL_OBSERVATION_DISAGREEMENTS,
    TERMINAL_REWARD_DISAGREEMENTS,
    ModelSpec,
    discover_vectorized_classes,
    missing_module,
    model_variants,
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

# Actions and draws for each terminal state the registry and the rollouts
# supply. Fewer draws than the live-state checks: most environments hold a
# terminal state still, and a constant column is compared exactly.
_TERMINAL_ACTIONS_CHECKED = 4
_TERMINAL_DRAWS = 500


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


def _pinned_model_id(spec: ModelSpec) -> str:
    """The id of the model on its environment's pinned configuration."""
    return spec.model_id.split("[", 1)[0]


def _action_index(spec: ModelSpec, env: Environment, model: Any, action: Any) -> int:
    """The model's index of a scalar-environment action."""
    for index in range(spec.num_actions(env, model)):
        if np.array_equal(np.asarray(spec.action_of_index(env, model, index)), np.asarray(action)):
            return index
    raise AssertionError(f"{spec.model_id}: {action!r} is not one of the model's actions")


def _terminal_transitions(
    spec: ModelSpec, env: Environment, model: Any
) -> List[TerminalTransition]:
    """Steps into a terminal state of the scalar env; the action is its index.

    The registry environment's entry probe first, then the steps on which
    random rollouts ended.
    """
    n_actions = spec.num_actions(env, model)
    probed: List[TerminalTransition] = []
    if spec.env_id is not None:
        probed = [
            TerminalTransition(
                entry.state, _action_index(spec, env, model, entry.action), entry.terminal
            )
            for entry in probe_terminal_transitions(spec.env_id, env)
        ]
    return probed + find_terminal_transitions(
        env,
        initial_state=lambda seed: _initial_state(spec, env, seed),
        random_action=lambda rng: int(rng.integers(n_actions)),
        seed_all=_seed_all,
        env_action=lambda index: spec.action_of_index(env, model, index),
    )


def _terminal_states_found(spec: ModelSpec, env: Environment, model: Any) -> List[Any]:
    """Terminal states of the scalar env: the registry's probes, then rollouts."""
    probes = list(spec.probe_states(env)) if spec.probe_states is not None else []
    if spec.env_id is not None:
        probes += probe_terminal_states(spec.env_id, env)
    return [state for state in probes if env.is_terminal(state)] + [
        transition.terminal for transition in _terminal_transitions(spec, env, model)
    ]


def _terminal_states_of(spec: ModelSpec, env: Environment, model: Any) -> List[Any]:
    """The terminal states the kernel checks use; skips a model declared to have none."""
    states = _terminal_states_found(spec, env, model)
    if not states:
        # test_terminal_mask_matches_scalar_env fails for any model that
        # supplies no terminal state without being declared.
        pytest.skip(
            f"{spec.model_id} has no terminal state: "
            f"{MODELS_WITHOUT_TERMINAL_STATES.get(_pinned_model_id(spec), 'none found')}"
        )
    return states


def _terminal_action_indices(spec: ModelSpec, env: Environment, model: Any) -> List[int]:
    n_actions = spec.num_actions(env, model)
    if n_actions <= _TERMINAL_ACTIONS_CHECKED:
        return list(range(n_actions))
    rng = np.random.default_rng(1234)
    return sorted(int(i) for i in rng.choice(n_actions, _TERMINAL_ACTIONS_CHECKED, replace=False))


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

    Given: Every state visited by several random rollouts, and the terminal
        states found by longer rollouts and by the environment's probe.
    When: ``terminal_mask`` and ``is_terminal`` classify them.
    Then: The classifications are identical, and they include at least one
        live state and -- unless the model is declared to have no terminal
        state -- at least one terminal state.

    Test type: integration
    """
    env, model = _build(spec)
    _, visited = _rollout_states(spec, env, model)
    visited = visited + _terminal_states_found(spec, env, model)
    rows = np.stack([spec.row_of_state(env, s) for s in visited])
    vectorized = model.terminal_mask(_tensor(rows, model)).detach().cpu().numpy().astype(bool)
    scalar = np.array([bool(env.is_terminal(s)) for s in visited])
    np.testing.assert_array_equal(
        vectorized, scalar, err_msg=f"{spec.model_id}: terminal_mask disagrees with is_terminal"
    )
    # A mask that is false everywhere agrees with is_terminal on a set of live
    # states. The check only means something once it has seen both kinds.
    assert not scalar.all(), f"{spec.model_id}: every state checked was terminal"
    declared_without = _pinned_model_id(spec) in MODELS_WITHOUT_TERMINAL_STATES
    assert scalar.any() != declared_without, (
        f"{spec.model_id}: the check saw {'a' if scalar.any() else 'no'} terminal state and "
        f"the model is {'' if declared_without else 'not '}declared in "
        "MODELS_WITHOUT_TERMINAL_STATES. Give its environment a terminal probe, or fix "
        "the declaration."
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
# Terminal states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", _params(TERMINAL_NEXT_STATE_DISAGREEMENTS))
def test_sample_next_states_matches_scalar_env_on_terminal_states(spec: ModelSpec) -> None:
    """``sample_next_states`` does to a finished row what the scalar env does.

    Purpose: VOPP steps a whole batch, finished rows included. Whatever the
        scalar environment does to a terminal state -- hold it, move it, clear
        its flag -- the model must do the same, or a row the search believes
        finished comes back to life in the model, or the reverse. A terminal
        rule applied to the whole batch rather than row by row gets a mixed
        batch wrong, so the batch here mixes terminal states with a live one.

    Given: Terminal states of the scalar env and one reachable live state,
        interleaved into one block with each state repeated
        ``_TERMINAL_DRAWS`` times.
    When: The block goes through ``sample_next_states`` once and through the
        scalar ``sample_next_state`` row by row, from independent seeds.
    Then: For every source state, terminal or live, the two samples have the
        same distribution.

    Test type: integration
    """
    env, model = _build(spec)
    live, _ = _rollout_states(spec, env, model)
    sources = _terminal_states_of(spec, env, model) + live[:1]
    rows = np.stack([spec.row_of_state(env, state) for state in sources])
    block = np.tile(rows, (_TERMINAL_DRAWS, 1))
    repeated = [sources[i % len(sources)] for i in range(len(block))]

    for index in _terminal_action_indices(spec, env, model):
        action = spec.action_of_index(env, model, index)
        _seed_all(21)
        vectorized = _numpy(
            model.sample_next_states(_tensor(block, model), _actions(index, len(block), model))
        )
        _seed_all(22)
        scalar = np.stack(
            [
                spec.row_of_state(env, env.sample_next_state(state=_copy(state), action=action))
                for state in repeated
            ]
        )
        for position, source in enumerate(sources):
            try:
                assert_same_distribution(
                    vectorized[position :: len(sources)], scalar[position :: len(sources)]
                )
            except AssertionError as error:
                kind = "terminal" if env.is_terminal(source) else "live"
                raise AssertionError(
                    f"{spec.model_id} action={action!r} {kind} state={source!r}: {error}"
                ) from None


@pytest.mark.parametrize("spec", _params(TERMINAL_OBSERVATION_DISAGREEMENTS))
def test_observation_kernels_match_scalar_env_on_terminal_states(spec: ModelSpec) -> None:
    """The model observes and scores a finished row as the scalar env does.

    Purpose: The step that ends an episode still emits an observation, and the
        search weights its particles with it. A model that emits another
        reading from a terminal state, or scores a reading against one
        differently, branches and reweights on a sensor the episode does not
        have -- on exactly the step that decides the outcome.

    Given: Terminal states of the scalar env as next states, each with several
        actions.
    When: ``sample_observations`` and the scalar ``sample_observation`` each
        draw ``_TERMINAL_DRAWS`` readings from a terminal state, and
        ``observation_log_probs`` and ``observation_log_probability`` score
        one scalar reading against the terminal states and a live one.
    Then: The two observation samples have the same distribution, and the
        log-probabilities agree to 1e-6 wherever either is above
        ``_NEGLIGIBLE_LOG_LIKELIHOOD``.

    Test type: integration
    """
    if spec.observation_not_comparable is not None:
        pytest.skip(f"{spec.model_id}: {spec.observation_not_comparable}")
    env, model = _build(spec)
    live, _ = _rollout_states(spec, env, model)
    terminal = _terminal_states_of(spec, env, model)
    candidates = terminal + live[:4]
    candidate_rows = np.stack([spec.row_of_state(env, state) for state in candidates])

    for index in _terminal_action_indices(spec, env, model):
        action = spec.action_of_index(env, model, index)
        for next_state in terminal:
            context = f"{spec.model_id} action={action!r} terminal next_state={next_state!r}"
            block = np.repeat(spec.row_of_state(env, next_state)[None, :], _TERMINAL_DRAWS, axis=0)
            _seed_all(32)
            vectorized_observations = _numpy(
                model.sample_observations(
                    _tensor(block, model), _actions(index, _TERMINAL_DRAWS, model)
                )
            )
            _seed_all(33)
            scalar_observations = [
                env.sample_observation(next_state=_copy(next_state), action=action)
                for _ in range(_TERMINAL_DRAWS)
            ]
            try:
                assert_same_distribution(
                    vectorized_observations,
                    np.stack([spec.row_of_observation(env, o) for o in scalar_observations]),
                )
            except AssertionError as error:
                raise AssertionError(f"{context}: {error}") from None

            observation = scalar_observations[0]
            observation_row = spec.row_of_observation(env, observation)
            vectorized = _numpy(
                model.observation_log_probs(
                    _tensor(candidate_rows, model),
                    _actions(index, len(candidates), model),
                    _tensor(np.repeat(observation_row[None, :], len(candidates), axis=0), model),
                )
            )
            scalar = np.array(
                [
                    float(
                        env.observation_log_probability(
                            next_state=_copy(s), action=action, observations=[observation]
                        )[0]
                    )
                    for s in candidates
                ]
            )
            assert not np.any(np.isnan(vectorized)), f"{context}: NaN log-probability"
            relevant = (vectorized > _NEGLIGIBLE_LOG_LIKELIHOOD) | (
                scalar > _NEGLIGIBLE_LOG_LIKELIHOOD
            )
            np.testing.assert_allclose(
                vectorized[relevant],
                scalar[relevant],
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"{context}: observation_log_probs disagrees with the scalar env",
            )


def _assert_rewards_match(
    spec: ModelSpec, env: Environment, model: Any, state: Any, index: int, next_state: Any
) -> None:
    """Assert ``rewards`` and ``reward`` score one transition alike, by distribution."""
    action = spec.action_of_index(env, model, index)
    states = np.repeat(spec.row_of_state(env, state)[None, :], _TERMINAL_DRAWS, axis=0)
    next_states = np.repeat(spec.row_of_state(env, next_state)[None, :], _TERMINAL_DRAWS, axis=0)
    _seed_all(18)
    vectorized = _numpy(
        model.rewards(
            _tensor(states, model),
            _actions(index, _TERMINAL_DRAWS, model),
            _tensor(next_states, model),
        )
    )
    _seed_all(19)
    scalar = np.array(
        [float(env.reward(_copy(state), action, _copy(next_state))) for _ in range(_TERMINAL_DRAWS)]
    )
    try:
        assert_same_distribution(vectorized[:, None], scalar[:, None])
    except AssertionError as error:
        raise AssertionError(
            f"{spec.model_id} action={action!r} state={state!r} next_state={next_state!r}: "
            f"{error}"
        ) from None


@pytest.mark.parametrize("spec", _params(TERMINAL_REWARD_DISAGREEMENTS))
def test_rewards_match_scalar_env_on_terminal_states(spec: ModelSpec) -> None:
    """``rewards`` scores a step taken from a finished row as ``reward`` does.

    Purpose: A search values a batch that holds finished rows, and what a
        finished row earns is part of the value. A model that keeps charging
        it the step cost, or pays it the goal reward again, where the scalar
        environment pays nothing -- or the reverse -- ranks actions by a
        return the episode never delivers.

    Given: Terminal states of the scalar env, each with several actions and a
        next state the scalar env sampled from it.
    When: ``rewards`` scores the transition repeated ``_TERMINAL_DRAWS`` times
        and ``reward`` scores it once per row.
    Then: The two reward samples have the same distribution.

    Test type: integration
    """
    env, model = _build(spec)
    for state in _terminal_states_of(spec, env, model):
        for index in _terminal_action_indices(spec, env, model):
            _seed_all(17)
            next_state = env.sample_next_state(
                state=_copy(state), action=spec.action_of_index(env, model, index)
            )
            _assert_rewards_match(spec, env, model, state, index, next_state)


@pytest.mark.parametrize("spec", _params(TERMINAL_ENTRY_REWARD_DISAGREEMENTS))
def test_rewards_match_scalar_env_on_the_step_into_a_terminal_state(spec: ModelSpec) -> None:
    """``rewards`` scores the step that ends an episode as ``reward`` does.

    Purpose: The terminal reward -- the goal bonus, the collision penalty --
        is usually the largest of the episode and is paid on exactly this
        step. It is what the planner is choosing between. The live-state
        reward check seldom reaches it: its rollouts are six steps long.

    Given: The steps on which random rollouts of the scalar env ended, and the
        steps its entry probe builds, each as its state, its action index and
        the terminal state it produced.
    When: ``rewards`` scores each step repeated ``_TERMINAL_DRAWS`` times and
        ``reward`` scores it once per row.
    Then: The two reward samples have the same distribution.

    Test type: integration
    """
    env, model = _build(spec)
    transitions = _terminal_transitions(spec, env, model)
    if not transitions:
        pytest.skip(
            f"{spec.model_id}: no random rollout ends, so there is no observed step into a "
            "terminal state"
        )
    for transition in transitions:
        _assert_rewards_match(
            spec, env, model, transition.state, transition.action, transition.terminal
        )


def test_models_without_terminal_states_agree_with_the_environment_list() -> None:
    """A model is declared to have no terminal state exactly when its environment is.

    Purpose: Two lists say the same thing about the registry environments --
        one for the scalar suite, one for this one. If they part, the same
        environment is checked on terminal states in one suite and skipped in
        the other, and the skip reads as coverage.

    Given: Every registry-built model, ``MODELS_WITHOUT_TERMINAL_STATES`` and
        ``ENVS_WITHOUT_TERMINAL_STATES``.
    When: Each model's declaration is compared with its environment's.
    Then: They agree, and no model declaration names a model that is not
        registered.

    Test type: unit
    """
    for spec in MODEL_SPECS:
        if spec.env_id is None:
            continue
        model_declared = _pinned_model_id(spec) in MODELS_WITHOUT_TERMINAL_STATES
        env_declared = base_id_of(spec.env_id) in ENVS_WITHOUT_TERMINAL_STATES
        assert model_declared == env_declared, (
            f"{spec.model_id} and its environment {spec.env_id} disagree on whether "
            "there is a terminal state"
        )
    stale = sorted(set(MODELS_WITHOUT_TERMINAL_STATES) - {spec.model_id for spec in MODEL_SPECS})
    assert not stale, f"MODELS_WITHOUT_TERMINAL_STATES names unregistered models: {stale}"


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


@pytest.mark.parametrize("spec", [pytest.param(s, id=s.model_id) for s in MODEL_SPECS])
def test_model_kernels_follow_the_protocol_shapes(spec: ModelSpec) -> None:
    """Every kernel returns the shape and dtype ``VectorizedGenerativeModel`` documents.

    Purpose: The checks above compare values after converting them to
        numpy, which accepts a ``[N, 1]`` reward, a float terminal mask, or a
        float observation key without complaint. The planner does not: it
        indexes tree tables with the keys and masks with the terminal flag.
        Each model test file checked its own model's shapes by hand.

    Given: A registry model and a batch of reachable states.
    When: Every protocol method is called once on the batch.
    Then: The model satisfies the protocol; its action count matches the
        environment's action set; next states keep the state shape;
        observations are ``[N, do]``; rewards and observation log-probs are
        float ``[N]`` without NaN; the terminal mask is bool ``[N]``; action
        keys are distinct integers, one per action; observation keys are
        integer ``[N]``, the same on a second call, and equal for equal rows
        wherever they sit in a batch.

    Test type: unit
    """
    env, model = _build(spec)
    assert isinstance(model, VectorizedGenerativeModel)
    n_actions = spec.num_actions(env, model)
    if isinstance(env, DiscreteActionsEnvironment):
        assert n_actions == len(env.get_actions())

    live, _ = _rollout_states(spec, env, model)
    states = _tensor([spec.row_of_state(env, state) for state in live], model)
    count = states.shape[0]
    actions = torch.as_tensor(np.arange(count) % n_actions, dtype=torch.int64, device=model.device)

    _seed_all(0)
    next_states = model.sample_next_states(states, actions)
    assert tuple(next_states.shape) == tuple(states.shape)
    observations = model.sample_observations(next_states, actions)
    assert observations.ndim == 2 and observations.shape[0] == count

    rewards = model.rewards(states, actions, next_states)
    assert tuple(rewards.shape) == (count,) and rewards.is_floating_point()
    assert not torch.any(torch.isnan(rewards))

    mask = model.terminal_mask(states)
    assert tuple(mask.shape) == (count,) and mask.dtype == torch.bool

    log_probs = model.observation_log_probs(next_states, actions, observations)
    assert tuple(log_probs.shape) == (count,) and log_probs.is_floating_point()
    assert not torch.any(torch.isnan(log_probs))

    action_keys = model.action_keys(torch.arange(n_actions, device=model.device))
    assert tuple(action_keys.shape) == (n_actions,)
    assert not action_keys.is_floating_point() and action_keys.dtype != torch.bool
    assert len(set(action_keys.tolist())) == n_actions, "two actions share a key"

    keys = model.observation_keys(observations)
    assert tuple(keys.shape) == (count,)
    assert not keys.is_floating_point() and keys.dtype != torch.bool
    assert torch.equal(keys, model.observation_keys(observations.clone()))
    # Equal rows must share a key. Distinct rows may share one too: a model
    # with continuous observations buckets them on purpose (Push quantizes the
    # object position), so that half is not a contract.
    doubled = torch.cat([observations, observations.clone()])
    doubled_keys = model.observation_keys(doubled)
    assert torch.equal(doubled_keys[:count], doubled_keys[count:])


def test_model_variant_declines_are_current() -> None:
    """Every declined configuration variant is still declined, and for a stated reason.

    Purpose: A torch model implements part of its environment's configuration
        space. The registry sweeps every variant and keeps the ones the model
        declines out of the suite, so a decline is the one way a variant gets
        no kernel check at all. Each must therefore be a recorded decision: an
        entry that names no real variant covers nothing, and an entry whose
        variant now builds is hiding a model that should be under test.

    Given: Every swept variant of every registry-built model, and
        ``MODEL_VARIANT_DECLINES``.
    When: Each declined variant's builder runs.
    Then: It raises ``NotImplementedError``, and every entry names a variant
        the sweep finds.

    Test type: unit
    """
    variants = {
        variant.model_id: variant
        for spec in MODEL_SPECS
        if "[" not in spec.model_id
        for variant in model_variants(spec)
    }
    stale = sorted(set(MODEL_VARIANT_DECLINES) - set(variants))
    assert not stale, f"MODEL_VARIANT_DECLINES names variants the sweep no longer finds: {stale}"
    for model_id in MODEL_VARIANT_DECLINES:
        with pytest.raises(NotImplementedError):
            variants[model_id].build()
