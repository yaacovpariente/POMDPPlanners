# SPDX-License-Identifier: MIT

"""Collecting the exploration half of a round, without setting the world's state.

The algorithm this package implements asks for transitions drawn from an
exploration distribution over ``(state, action)`` pairs -- place the system in a
state, apply an action, record the successor. A live IsaacLab world cannot do
that: it has no state setter, and asking it for the successor of any state other
than the one it is currently in raises. Its only entry point is a reset followed
by forward stepping.

That is the "reset model" case the algorithm's authors treat explicitly, and it
is the case where iterating helps *most* -- a one-shot fit's guarantee carries a
train/test mismatch factor that the iterative version does not. Their sanctioned
substitute is a base policy with randomized actions, which is what this module
collects.

Two rules, both learned the expensive way:

**The action is held for several steps.** A rollout has to excite the system at
the timescale it responds on. Redrawing every control step measures a permanent
transient: on the ANYmal navigation task the fraction of a commanded velocity
the base actually achieves reads 0.25 when the command changes every 0.2 s
against 0.71 when it is held ten steps. A model fitted on redrawn commands
describes a robot that is never allowed to follow one. Holding costs nothing in
coverage, because the action set is a small preset table.

**Boundary rows are dropped, not fitted.** The simulator auto-resets inside its
step, so the successor of a terminal transition is a fresh episode metres away.
Keeping those rows teaches a model that some action teleports.

Functions:
    collect_random_preset_episode: One held-random-action rollout, boundaries dropped.
"""

from typing import Any, List, Optional, Tuple

import numpy as np

#: Control steps a drawn action is held for. Ten is the ANYmal figure above; it
#: is roughly the settling time of a velocity command, which is the quantity
#: that matters rather than the number itself.
DEFAULT_HOLD_STEPS = 10


def collect_random_preset_episode(
    world: Any,
    action_presets: Any,
    num_steps: int,
    rng: np.random.Generator,
    hold_steps: int = DEFAULT_HOLD_STEPS,
    initial_state: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out held random presets and return the usable transitions.

    Args:
        world: The true world. Must expose ``sample_next_state(state, action)``,
            ``is_terminal(state)`` and, when ``initial_state`` is omitted,
            ``initial_state_dist()``.
        action_presets: The table of action vectors to draw from.
        num_steps: Steps to attempt. Fewer transitions are returned when the
            episode ends early, because boundary rows are dropped.
        rng: Source of the action draws, so a round can be reproduced.
        hold_steps: Control steps each drawn action is held for.
        initial_state: State to start from. ``None`` draws one from the world's
            initial-state distribution.

    Returns:
        ``(states, actions, next_states)``, each ``(N, .)`` with ``N <= num_steps``.

    Raises:
        ValueError: If ``hold_steps`` is not positive.
    """
    if hold_steps <= 0:
        raise ValueError(f"hold_steps must be positive, got {hold_steps}")
    presets = np.asarray(action_presets, dtype=float)

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    next_states: List[np.ndarray] = []

    state = (
        np.asarray(world.initial_state_dist().sample(), dtype=float).ravel()
        if initial_state is None
        else np.asarray(initial_state, dtype=float).ravel()
    )
    action = presets[0]
    held_for = 0
    for _ in range(num_steps):
        if held_for == 0:
            action = presets[int(rng.integers(len(presets)))]
        next_state = np.asarray(world.sample_next_state(state, action), dtype=float).ravel()
        ended = bool(world.is_terminal(next_state))
        if not ended:
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
        state = next_state
        held_for = (held_for + 1) % hold_steps
        if ended:
            break

    if not states:
        width = int(np.asarray(state).size)
        return (
            np.zeros((0, width)),
            np.zeros((0, presets.shape[-1])),
            np.zeros((0, width)),
        )
    return (
        np.asarray(states, dtype=float),
        np.asarray(actions, dtype=float),
        np.asarray(next_states, dtype=float),
    )
