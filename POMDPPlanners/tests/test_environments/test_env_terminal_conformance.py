# SPDX-License-Identifier: MIT

"""Every environment treats terminal states the same way on every path.

``is_terminal`` decides where an episode, a rollout and a search branch stop.
The batch paths -- ``sample_next_state_batch``, ``reward_batch`` -- are second
implementations of the single-state kernels, and a belief hands them terminal
states all the time: once some particles have finished, a planner that tests
one sampled particle for termination keeps stepping and scoring the rest.

The API conformance suite compares the batch and single paths only on states a
rollout can still step from, so a batch path that freezes a finished particle
where the single path moves it, or charges it a step cost where the single
path charges nothing, went unseen. This file compares them on terminal states,
on batches that mix terminal and live states, and on the step that enters a
terminal state.

It states no rule about what a terminal state *should* do. Environments
differ -- some absorb, some keep moving, some pay a reward after the end --
and each is free to. What must hold is that an environment's own paths agree.

Terminal states come from
:mod:`~POMDPPlanners.tests.test_environments._terminal_states`: found by
random rollouts through the interface where a rollout ends, built by a
registered probe where it does not. Every test is parametrized over
``ENV_BUILDERS``, so each runs on every configuration variant as well.
"""

import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.tests.test_environments._env_config_variants import base_id_of
from POMDPPlanners.tests.test_environments._terminal_states import (
    ENVS_WITHOUT_TERMINAL_STATES,
    TERMINAL_STATE_PROBES,
    TERMINAL_ENTRY_PROBES,
    TerminalTransition,
    copy_state,
    probe_terminal_states,
    probe_terminal_transitions,
    terminal_states,
    terminal_transitions,
)
from POMDPPlanners.tests.test_environments.test_env_api_conformance import (
    ENV_BUILDERS,
    EnvBuilder,
    _assert_next_state_batch_matches_single,
    _assert_reward_batch_matches_reward,
    _numpy_state_key,
    _random_action,
    _seed_all,
    _trajectory_key,
)

# Live states checked per environment, beside every terminal state
# ``terminal_states`` supplies (its probe states and up to two rollout ones).
_LIVE_STATES = 2

# Actions checked from each batch.
_ACTIONS_CHECKED = 2

# Initial states drawn in the not-terminal check.
_INITIAL_DRAWS = 32

# A log-density at or below this is "impossible"; see
# test_env_kernel_conformance._IMPOSSIBLE_LOG_DENSITY.
_IMPOSSIBLE_LOG_DENSITY = -500.0

# Each list below records a real disagreement found by this suite, as
# ``env_id: reason``. Strict xfail turns the test red the moment it is fixed,
# so an entry cannot outlive its bug.

# Envs whose batch transition disagrees with the single-state one on a
# terminal state.
TERMINAL_NEXT_STATE_BATCH_DISAGREEMENTS: Dict[str, str] = {}

# Envs whose batch reward disagrees with the single-state one on a terminal
# state.
TERMINAL_REWARD_BATCH_DISAGREEMENTS: Dict[str, str] = {}

# Envs whose transition density calls the sampler's step from a terminal state
# impossible.
#
# DiscreteLightDarkPOMDP with the hazard-terminal slot: ``sample_next_state``
# holds a hazard-terminated state (slot set) still. ``transition_log_probability``
# drops the slot, scores the position as a live move, and gives the unchanged
# state -inf. The density does not model the slot at all.
TERMINAL_TRANSITION_DENSITY_DISAGREEMENTS: Dict[str, str] = {
    "DiscreteLightDarkPOMDP[is_obstacle_hit_terminal=True]": (
        "transition_log_probability ignores the hazard-terminal slot and gives -inf to "
        "the unchanged state sample_next_state returns from a hazard-terminated state"
    ),
}

# Envs whose batch reward disagrees with the single-state one on the step
# that enters a terminal state.
TERMINAL_ENTRY_REWARD_DISAGREEMENTS: Dict[str, str] = {}

# Envs whose initial-state distribution can produce a terminal state.
TERMINAL_INITIAL_STATE_ENVS: Dict[str, str] = {}


def _params(disagreements: Dict[str, str]) -> List[Any]:
    """One param per registered env, ``xfail(strict=True)`` where one is recorded."""
    params = []
    for env_id, builder in ENV_BUILDERS:
        marks = []
        if env_id in disagreements:
            marks.append(pytest.mark.xfail(strict=True, reason=disagreements[env_id]))
        params.append(pytest.param(env_id, builder, id=env_id, marks=marks))
    return params


def _live_states(env: Environment) -> List[Any]:
    """Non-terminal states: one initial state and one a short rollout reaches."""
    _seed_all(0)
    action_rng = np.random.default_rng(0)
    state = env.initial_state_dist().sample()[0]
    states = [state]
    for _ in range(4):
        following = env.sample_next_state(
            state=copy_state(state), action=_random_action(env, action_rng)
        )
        if env.is_terminal(following):
            break
        state = following
    if _trajectory_key(state) != _trajectory_key(states[0]):
        states.append(state)
    return states[:_LIVE_STATES]


def _terminal_and_live(env_id: str, env: Environment) -> Tuple[List[Any], List[Any]]:
    """Terminal and live states of ``env``; skips an env declared to have no terminal state."""
    terminal = terminal_states(env_id, env)
    if not terminal:
        # test_environment_supplies_a_terminal_state fails for any env that
        # lands here without being declared, so this skip is never silent.
        pytest.skip(
            f"{env_id} has no terminal state: "
            f"{ENVS_WITHOUT_TERMINAL_STATES.get(base_id_of(env_id), 'none found')}"
        )
    return terminal, _live_states(env)


def _actions(env: Environment) -> List[Any]:
    rng = np.random.default_rng(5)
    return [_random_action(env, rng) for _ in range(_ACTIONS_CHECKED)]


# ---------------------------------------------------------------------------
# is_terminal itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id,env_builder", _params({}))
def test_is_terminal_is_a_pure_predicate(env_id: str, env_builder: EnvBuilder) -> None:
    """``is_terminal`` answers with a bool, the same one twice, and touches nothing.

    Purpose: Every planner calls ``is_terminal`` inside its search loop, between
        the draws that make a seeded run reproducible. A predicate that draws
        a random number shifts every later transition; one that mutates its
        argument corrupts the particle it was asked about; one that answers
        with an array or ``None`` is truthy or falsy by accident.

    Given: Terminal and live states of the environment, and snapshots of the
        Python and numpy RNG streams.
    When: ``is_terminal`` is called twice on each state.
    Then: Both answers are the same ``bool``, the state is unchanged, both RNG
        streams are unchanged, and a transition drawn afterwards matches the
        one drawn without the calls -- which is what catches a draw from a
        native RNG.

    Test type: integration
    """
    env = env_builder()
    states = terminal_states(env_id, env) + _live_states(env)
    action = _actions(env)[0]
    live = _live_states(env)[0]

    _seed_all(7)
    expected_probe = _trajectory_key(env.sample_next_state(state=copy_state(live), action=action))
    _seed_all(7)
    numpy_before = _numpy_state_key()
    random_before = random.getstate()
    for state in states:
        key_before = _trajectory_key(state)
        first = env.is_terminal(state)
        second = env.is_terminal(state)
        assert isinstance(
            first, (bool, np.bool_)
        ), f"{env_id}.is_terminal returned {type(first).__name__}, not a bool"
        assert first == second, f"{env_id}.is_terminal gave two answers for {state!r}"
        assert (
            _trajectory_key(state) == key_before
        ), f"{env_id}.is_terminal changed the state it was asked about"
    assert _numpy_state_key() == numpy_before, f"{env_id}.is_terminal advanced np.random"
    assert random.getstate() == random_before, f"{env_id}.is_terminal advanced stdlib random"
    actual_probe = _trajectory_key(env.sample_next_state(state=copy_state(live), action=action))
    assert actual_probe == expected_probe, (
        f"{env_id}.is_terminal consumed randomness from a native RNG: the next transition "
        "drawn after it differs from the same transition drawn without it"
    )


@pytest.mark.parametrize("env_id,env_builder", _params(TERMINAL_INITIAL_STATE_ENVS))
def test_initial_states_are_not_terminal(env_id: str, env_builder: EnvBuilder) -> None:
    """An episode never starts in a terminal state.

    Purpose: The episode runner, every rollout and every conformance check
        take the first step without asking. An episode that starts finished is
        recorded with zero steps and a return of zero, and it counts as a
        completed episode in every statistic.

    Given: ``_INITIAL_DRAWS`` states from ``initial_state_dist``.
    When: ``is_terminal`` is asked about each.
    Then: None is terminal.

    Test type: integration
    """
    env = env_builder()
    _seed_all(0)
    terminal = [s for s in env.initial_state_dist().sample(_INITIAL_DRAWS) if env.is_terminal(s)]
    assert not terminal, (
        f"{env_id}.initial_state_dist produced {len(terminal)} terminal states out of "
        f"{_INITIAL_DRAWS}, for example {terminal[0]!r}"
    )


@pytest.mark.parametrize("env_id,env_builder", _params({}))
def test_environment_supplies_a_terminal_state(env_id: str, env_builder: EnvBuilder) -> None:
    """Each environment yields a terminal state, or is declared to have none.

    Purpose: Every check below, and every terminal check in the vectorized
        suites, runs on the terminal states found here. An environment that
        quietly supplies none passes all of them by running none of them. So
        "no terminal state" has to be a statement someone made about the
        environment, not the absence of a result.

    Given: The environment, its registered probes if it has any, and the
        terminal transitions found by random rollouts.
    When: The found states are compared with the declarations.
    Then: An environment declared to have no terminal states has none -- no
        rollout ends and it has no probe. Any other environment has at least
        one, every state its probe builds is terminal, and every step its
        entry probe builds goes from a live state to a terminal one.

    Test type: integration
    """
    env = env_builder()
    base_id = base_id_of(env_id)
    found = terminal_states(env_id, env)
    probed = probe_terminal_states(env_id, env)

    if base_id in ENVS_WITHOUT_TERMINAL_STATES:
        assert (
            base_id not in TERMINAL_STATE_PROBES and base_id not in TERMINAL_ENTRY_PROBES
        ), f"{base_id} is declared to have no terminal states and has a terminal probe"
        assert not found, (
            f"{env_id} is declared to have no terminal states "
            f"({ENVS_WITHOUT_TERMINAL_STATES[base_id]}) but a rollout reached {found[0]!r}; "
            "remove it from ENVS_WITHOUT_TERMINAL_STATES"
        )
        return

    not_terminal = [state for state in probed if not env.is_terminal(state)]
    assert not not_terminal, (
        f"{env_id}: the registered probe built a state is_terminal does not call terminal: "
        f"{not_terminal[0]!r}"
    )
    if base_id in TERMINAL_ENTRY_PROBES:
        entries = probe_terminal_transitions(env_id, env)
        assert entries, f"{env_id}: the registered entry probe built no step"
        for entry in entries:
            assert not env.is_terminal(entry.state) and env.is_terminal(entry.terminal), (
                f"{env_id}: the registered entry probe built a step that does not go from a "
                f"live state to a terminal one: {entry!r}"
            )
    assert found, (
        f"{env_id}: no random rollout reached a terminal state and no probe is registered. "
        "Add a probe to TERMINAL_STATE_PROBES in _terminal_states.py, or declare the "
        "environment in ENVS_WITHOUT_TERMINAL_STATES with the reason it never terminates."
    )


def test_terminal_state_lists_name_registered_environments() -> None:
    """No probe or declaration outlives the environment it was written for.

    Purpose: Both lists are keyed by a pinned configuration's id. An entry
        under an id that is no longer registered applies to nothing, while
        still reading as if that environment were covered.

    Given: ``TERMINAL_STATE_PROBES``, ``TERMINAL_ENTRY_PROBES`` and
        ``ENVS_WITHOUT_TERMINAL_STATES``.
    When: Each key is looked up in ``ENV_BUILDERS``.
    Then: Every key is a registered pinned configuration.

    Test type: unit
    """
    registered = {env_id for env_id, _ in ENV_BUILDERS}
    stale = sorted(
        (
            set(TERMINAL_STATE_PROBES)
            | set(TERMINAL_ENTRY_PROBES)
            | set(ENVS_WITHOUT_TERMINAL_STATES)
        )
        - registered
    )
    assert not stale, f"terminal-state lists name ids that are not in ENV_BUILDERS: {stale}"


# ---------------------------------------------------------------------------
# batch / single agreement on terminal states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id,env_builder", _params(TERMINAL_NEXT_STATE_BATCH_DISAGREEMENTS))
def test_sample_next_state_batch_matches_single_sample_on_terminal_states(
    env_id: str, env_builder: EnvBuilder
) -> None:
    """The batch transition does to a finished particle what the single one does.

    Purpose: A belief keeps its finished particles and steps them with the
        rest. Whatever the environment does to a terminal state -- hold it,
        move it, clear its flag -- the batch path and the single path must do
        the same, or the belief and the search disagree about the finished
        part of the belief. A batch path that applies its terminal rule to the
        whole array instead of row by row gets a mixed batch wrong, which a
        batch of terminal states alone would not show.

    Given: Terminal and live states of the environment, interleaved into one
        batch with each state repeated, and two random actions.
    When: ``sample_next_state_batch`` steps the batch once and
        ``sample_next_state`` steps it row by row, from independent seeds.
    Then: For every source state, terminal or live, the two samples have the
        same distribution.

    Test type: integration
    """
    env = env_builder()
    terminal, live = _terminal_and_live(env_id, env)
    for action in _actions(env):
        _assert_next_state_batch_matches_single(env, terminal + live, action)


@pytest.mark.parametrize("env_id,env_builder", _params(TERMINAL_TRANSITION_DENSITY_DISAGREEMENTS))
def test_transition_log_probability_supports_the_step_from_a_terminal_state(
    env_id: str, env_builder: EnvBuilder
) -> None:
    """The transition density allows what the sampler does to a terminal state.

    Purpose: The sampler and the density describe one transition model. On a
        terminal state they are easy to write apart: the sampler holds the
        state still while the density keeps computing a live move, and so
        calls the sampler's own output impossible. A filter that weights
        finished particles through the density then discards all of them.

    Given: Terminal states of the environment and two random actions.
    When: ``sample_next_state`` steps each terminal state, and
        ``transition_log_probability`` scores the state it produced.
    Then: The log-density is finite and above the floor implementations use
        for an impossible value. An environment whose density is not
        implemented skips, with the method's own message.

    Test type: integration
    """
    env = env_builder()
    terminal, _ = _terminal_and_live(env_id, env)
    for action in _actions(env):
        for index, state in enumerate(terminal):
            _seed_all(index)
            next_state = env.sample_next_state(state=copy_state(state), action=action)
            try:
                values = env.transition_log_probability(
                    copy_state(state), action, [copy_state(next_state)]
                )
            except NotImplementedError as error:
                pytest.skip(f"{env_id}.transition_log_probability is not implemented: {error}")
            value = float(np.asarray(values, dtype=np.float64).ravel()[0])
            assert np.isfinite(value) and value > _IMPOSSIBLE_LOG_DENSITY, (
                f"{env_id}.transition_log_probability gives {value} to the next state "
                f"sample_next_state drew from a terminal state: state={state!r} "
                f"action={action!r} next_state={next_state!r}"
            )


@pytest.mark.parametrize("env_id,env_builder", _params(TERMINAL_REWARD_BATCH_DISAGREEMENTS))
def test_reward_batch_agrees_with_reward_on_terminal_states(
    env_id: str, env_builder: EnvBuilder
) -> None:
    """The batch reward scores a finished particle as the single reward does.

    Purpose: The belief-level cost scores every particle of a belief with
        ``reward_batch``, finished ones included. If the batch path charges a
        finished particle a step cost the single path does not -- or pays it
        the goal reward a second time -- the value of a belief depends on
        which path scored it.

    Given: Terminal and live states of the environment, interleaved into one
        batch with each state repeated, and two random actions.
    When: ``reward_batch`` scores the batch once and ``reward`` scores it row
        by row.
    Then: For every source state, terminal or live, the two reward samples
        have the same distribution.

    Test type: integration
    """
    env = env_builder()
    terminal, live = _terminal_and_live(env_id, env)
    for action in _actions(env):
        _assert_reward_batch_matches_reward(env, terminal + live, action)


@pytest.mark.parametrize("env_id,env_builder", _params(TERMINAL_ENTRY_REWARD_DISAGREEMENTS))
def test_reward_batch_agrees_with_reward_on_the_step_into_a_terminal_state(
    env_id: str, env_builder: EnvBuilder
) -> None:
    """The batch reward scores the step that ends an episode as the single reward does.

    Purpose: The terminal reward -- the goal bonus, the collision penalty --
        is usually the largest reward of the episode and is paid on exactly
        this step. It is also where the two paths most often part: one reads
        the terminal flag of the next state, the other recomputes the event
        from positions.

    Given: The steps on which random rollouts ended, and the steps the
        environment's entry probe builds, each as its state, its action and
        the terminal state it produced.
    When: ``reward_batch`` and ``reward`` score each step with that next state.
    Then: The two reward samples have the same distribution.

    Test type: integration
    """
    env = env_builder()
    transitions: List[TerminalTransition] = terminal_transitions(env_id, env)
    if not transitions:
        if base_id_of(env_id) in ENVS_WITHOUT_TERMINAL_STATES:
            pytest.skip(f"{env_id} has no terminal state")
        pytest.skip(
            f"{env_id}: no random rollout ends, so there is no observed step into a "
            "terminal state; its terminal states come from a probe"
        )
    for transition in transitions:
        _assert_reward_batch_matches_reward(
            env, [transition.state], transition.action, next_sources=[transition.terminal]
        )
