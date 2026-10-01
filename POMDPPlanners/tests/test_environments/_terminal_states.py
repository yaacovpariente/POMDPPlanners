# SPDX-License-Identifier: MIT

"""Terminal states of every registered environment, for the conformance suites.

The conformance suites compare an environment's copies -- the batch path, the
vectorized belief, the torch model -- on states reached by short random
rollouts, and every rollout stops at the first terminal state. So no copy was
ever compared *on* a terminal state, or on the step that enters one. Those are
exactly the states a belief holds once some of its particles have finished:
planners that test one sampled particle for termination keep stepping the
rest.

This module supplies terminal states through the interface only:

* :func:`find_terminal_transitions` rolls the scalar environment forward under
  random actions, for longer than the kernel checks do, and keeps each
  ``(state, action, terminal next state)`` it ends on. Where that works the
  suites need to know nothing about what "terminal" means.
* :data:`TERMINAL_STATE_PROBES` builds a terminal state for the environments a
  random walk does not finish -- a goal it does not find, a tag it does not
  land. This is the one place that knows what terminal means for them, and it
  is data: :func:`terminal_states` keeps a probe state only if ``is_terminal``
  agrees.
* :data:`TERMINAL_ENTRY_PROBES` builds a step into a terminal state where the
  way an episode ends matters and a random rollout ends that way only
  sometimes.
* :data:`ENVS_WITHOUT_TERMINAL_STATES` names the environments whose
  ``is_terminal`` is never true, so that "found no terminal state" is a stated
  fact about them and a failure for anything else.

Rollouts differ between platforms: native kernels draw through
``std::*_distribution``, whose streams differ between libc++ and libstdc++. A
recorded disagreement must fail on both, so wherever one depends on the kind of
terminal state reached, a probe supplies that kind.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.tests.test_environments._env_config_variants import base_id_of
from POMDPPlanners.tests.test_environments.test_env_api_conformance import (
    _random_action,
    _seed_all,
)

# Rollouts that look for a terminal state. Longer and more numerous than the
# rollouts that supply source states, because many episodes take dozens of
# random steps to end, and whether a given seed ends differs between
# platforms: native kernels draw through std::*_distribution, whose streams
# differ between libc++ and libstdc++.
TERMINAL_SEARCH_SEEDS: Sequence[int] = tuple(range(20))
TERMINAL_SEARCH_STEPS = 300

# Rollout transitions kept per environment. Two, so the checks see more than
# one way of ending (a goal, a hazard hit) where a rollout finds more than one.
MAX_TERMINAL_TRANSITIONS = 2


@dataclass(frozen=True)
class TerminalTransition:
    """One step that ended an episode.

    Attributes:
        state: The non-terminal state the step started from.
        action: The action taken.
        terminal: The terminal state the step produced.
    """

    state: Any
    action: Any
    terminal: Any


def copy_state(state: Any) -> Any:
    """A copy the environment can mutate without touching the original."""
    return np.array(state, copy=True) if isinstance(state, np.ndarray) else state


def find_terminal_transitions(
    env: Environment,
    initial_state: Callable[[int], Any],
    random_action: Callable[[np.random.Generator], Any],
    seed_all: Callable[[int], None] = _seed_all,
    env_action: Callable[[Any], Any] = lambda action: action,
) -> List[TerminalTransition]:
    """Roll ``env`` forward under random actions and keep the steps that end an episode.

    Args:
        env: The scalar environment.
        initial_state: ``seed -> state``, called after the RNGs are seeded.
        random_action: Draws one valid action from the given generator.
        seed_all: Seeds every RNG a step can draw from.
        env_action: Converts what ``random_action`` drew into the scalar
            environment's action, for a caller that draws an action index.
            The transition keeps what was drawn.

    Returns:
        Up to ``MAX_TERMINAL_TRANSITIONS`` transitions, one per rollout that
        terminated, in seed order. Empty if no rollout terminated.
    """
    transitions: List[TerminalTransition] = []
    for seed in TERMINAL_SEARCH_SEEDS:
        seed_all(seed)
        action_rng = np.random.default_rng(seed)
        state = initial_state(seed)
        if env.is_terminal(state):
            continue
        for _ in range(TERMINAL_SEARCH_STEPS):
            action = random_action(action_rng)
            next_state = env.sample_next_state(state=copy_state(state), action=env_action(action))
            if env.is_terminal(next_state):
                transitions.append(TerminalTransition(state, action, next_state))
                break
            state = next_state
        if len(transitions) >= MAX_TERMINAL_TRANSITIONS:
            break
    return transitions


# ---------------------------------------------------------------------------
# Probes: a terminal state built from an initial one
# ---------------------------------------------------------------------------


def _initial(env: Environment) -> np.ndarray:
    # The env's own dtype is kept, so a probe state is a state the env's
    # kernels are written for.
    _seed_all(0)
    return np.array(env.initial_state_dist().sample()[0], copy=True)


def _tagged(env: Environment) -> List[Any]:
    # (robot x, robot y, opponent x, opponent y, terminal): the episode ends
    # when the terminal flag is set.
    state = _initial(env)
    state[4] = 1.0
    return [state]


def _object_on_target(env: Environment) -> List[Any]:
    # (robot x, robot y, object x, object y, target x, target y, ...): the
    # episode ends when the object reaches the target.
    state = _initial(env)
    state[2:4] = state[4:6]
    return [state]


def _car_at_goal(env: Environment) -> List[Any]:
    state = _initial(env)
    state[0] = env.goal_position  # type: ignore[attr-defined]
    return [state]


def _ant_too_fast(env: Environment) -> List[Any]:
    # (x, y, vx, vy, ...): the episode ends at 1.5 times the safe speed.
    state = _initial(env)
    state[2] = 2.0 * env.safe_velocity_threshold  # type: ignore[attr-defined]
    state[3] = 0.0
    return [state]


def _agent_on_goal_cell(env: Environment) -> List[Any]:
    # Light-dark ends on reaching the goal as well as on a hazard hit; a random
    # walk finds the hazard and rarely the goal. With the hazard-terminal slot
    # the state has a third entry, and a second state has it set: a
    # hazard-terminated state, so that kind is checked on every platform
    # rather than only where a rollout happens to end on one.
    at_goal = _initial(env)
    at_goal[:2] = np.asarray(env.goal_state)  # type: ignore[attr-defined]
    states = [at_goal]
    if at_goal.shape[0] == 3:
        hazard_terminated = _initial(env)
        hazard_terminated[2] = 1
        states.append(hazard_terminated)
    return states


def _agent_in_goal_region(env: Environment) -> List[Any]:
    # The continuous goal is a disc. Two states: its centre, and a point inside
    # it with fractional coordinates, 0.9 of the radius from the centre. The
    # second is what a goal reached by a noisy step looks like, and it is
    # terminal only as long as its coordinates are kept as floats.
    goal = np.asarray(env.goal_state, dtype=np.float64)  # type: ignore[attr-defined]
    offset = 0.9 * float(env.goal_state_radius)  # type: ignore[attr-defined]
    states = []
    for position in (goal, goal - offset * np.array([np.cos(0.6), np.sin(0.6)])):
        state = _initial(env).astype(np.float64)
        state[:2] = position
        states.append(state)
    return states


def _robot_exited(env: Environment) -> List[Any]:
    # (row, column, rocks...): the exit is the (-1, -1) sentinel position.
    state = _initial(env)
    state[:2] = -1
    return [state]


TerminalProbe = Callable[[Environment], List[Any]]

# Keyed by the pinned configuration's id; every variant of it uses the same
# probe. An environment needs an entry only if random rollouts do not end it
# reliably.
TERMINAL_STATE_PROBES: Dict[str, TerminalProbe] = {
    "ContinuousLaserTagPOMDP": _tagged,
    "ContinuousLaserTagPOMDPDiscreteActions": _tagged,
    "ContinuousLightDarkPOMDP": _agent_in_goal_region,
    "ContinuousLightDarkPOMDPDiscreteActions": _agent_in_goal_region,
    "ContinuousPushPOMDP": _object_on_target,
    "ContinuousPushPOMDPDiscreteActions": _object_on_target,
    "DiscreteLightDarkPOMDP": _agent_on_goal_cell,
    "LaserTagPOMDP": _tagged,
    "MountainCarPOMDP": _car_at_goal,
    "PushPOMDP": _object_on_target,
    "RockSamplePOMDP": _robot_exited,
    "SafeAntVelocityPOMDP": _ant_too_fast,
}


def _pacman_ghost_swap(env: Environment) -> List[TerminalTransition]:
    # (PacMan row, column, ghost row, column, ...). The ghost is put one cell
    # from PacMan, on two free cells of the pinned maze, and PacMan steps until
    # the two pass through each other: each ends the step in the other's cell.
    # That ends the episode like a collision in one cell does, and a random
    # rollout ends either way.
    state = _initial(env)
    state[0:2] = (1, 4)
    state[2:4] = (1, 5)
    for action in env.get_actions():  # type: ignore[attr-defined]
        for seed in range(64):
            _seed_all(seed)
            next_state = np.asarray(env.sample_next_state(state=copy_state(state), action=action))
            swapped = np.array_equal(next_state[0:2], state[2:4]) and np.array_equal(
                next_state[2:4], state[0:2]
            )
            if swapped and env.is_terminal(next_state):
                return [TerminalTransition(state, action, next_state)]
    return []


TerminalEntryProbe = Callable[[Environment], List[TerminalTransition]]

# Steps into a terminal state, for the environments where a recorded
# disagreement depends on how the episode ends. Keyed like the probes above.
TERMINAL_ENTRY_PROBES: Dict[str, TerminalEntryProbe] = {
    "PacManPOMDP": _pacman_ghost_swap,
}

# Environments whose ``is_terminal`` is never true, and why. Keyed by the
# pinned configuration's id.
ENVS_WITHOUT_TERMINAL_STATES: Dict[str, str] = {
    "SanityPOMDP": "has no terminal states; an episode runs to the step limit",
    "TigerPOMDP": (
        "is_terminal always returns False: opening a door resets the tiger in the "
        "transition model instead of ending the episode"
    ),
}


def probe_terminal_states(env_id: str, env: Environment) -> List[Any]:
    """The registered probe's states for ``env_id``, or an empty list."""
    probe = TERMINAL_STATE_PROBES.get(base_id_of(env_id))
    return [] if probe is None else probe(env)


def probe_terminal_transitions(env_id: str, env: Environment) -> List[TerminalTransition]:
    """The registered entry probe's transitions for ``env_id``, or an empty list."""
    probe = TERMINAL_ENTRY_PROBES.get(base_id_of(env_id))
    return [] if probe is None else probe(env)


def rollout_terminal_transitions(env: Environment) -> List[TerminalTransition]:
    """Terminal transitions of an ``ENV_BUILDERS`` environment, from rollouts."""
    return find_terminal_transitions(
        env,
        initial_state=lambda seed: env.initial_state_dist().sample()[0],
        random_action=lambda rng: _random_action(env, rng),
    )


def terminal_transitions(env_id: str, env: Environment) -> List[TerminalTransition]:
    """Every step into a terminal state the suites use: entry probes, then rollouts."""
    return probe_terminal_transitions(env_id, env) + rollout_terminal_transitions(env)


def terminal_states(env_id: str, env: Environment) -> List[Any]:
    """Every terminal state the suites use for ``env_id``: probes, then rollouts.

    A probe state is kept only if ``is_terminal`` calls it terminal; the
    coverage test checks that every probe state is.
    """
    probed = [state for state in probe_terminal_states(env_id, env) if env.is_terminal(state)]
    return probed + [transition.terminal for transition in terminal_transitions(env_id, env)]
