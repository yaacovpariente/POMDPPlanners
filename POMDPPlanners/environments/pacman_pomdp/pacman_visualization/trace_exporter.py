# SPDX-License-Identifier: MIT

"""PacMan episode trace exporter.

The GIF renderer beside this module draws an episode. This one writes the same
episode as data, so the browser viewer can replay it. Nothing here is
re-derived: every number written comes from the recorded episode or from the
environment's own configuration.

The belief is not serialized here. It is a core abstraction with a closed
family of implementations, so
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, and this exporter is left with what is genuinely
PacMan's: the maze, PacMan's and the ghosts' cells, the pellets still on the
board, and the noisy ghost readings.

PacMan's belief is over *states*, and a PacMan state is a flat array, so a
particle arrives in the payload as that array and means nothing to a reader
that does not know the layout. The ``state_layout`` block is therefore part of
the payload: it is how a viewer turns a particle back into ghost cells without
importing Python, and it is read from the environment instance rather than
assumed, because the layout moves with ``num_ghosts`` and the pellet count.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps, to_jsonable

# Payload version, independent of the envelope's. Bump it when the meaning of
# a payload field changes, so a viewer can refuse a file it would misdraw.
PACMAN_PAYLOAD_KIND = "pacman.v1"


def _cells(positions: Any) -> List[List[int]]:
    """Write a collection of grid cells as ``[[row, col], ...]``."""
    return [[int(row), int(col)] for row, col in positions]


def _observation(observation: Any) -> Optional[List[List[int]]]:
    """Write one PacMan observation: a noisy cell reading per ghost.

    The terminal bookkeeping step carries no observation, which is not the
    same as a reading of ``(0, 0)``, so it stays ``None``.
    """
    if observation is None:
        return None
    return _cells(observation)


def _world(environment: Any) -> Dict[str, Any]:
    """The maze and its rules, taken from the instance the episode ran on.

    Class defaults are deliberately not consulted: a configured run may use a
    different maze, a different ghost count or a hazard zone, and a viewer
    built from the defaults would draw a world the episode never happened in.
    """
    rows, cols = environment.maze_size
    return {
        "maze_size": [int(rows), int(cols)],
        # Sorted so two traces of the same maze produce the same bytes; the
        # environment holds walls in a set, whose iteration order is not.
        "walls": _cells(sorted(environment.walls)),
        "initial_pellets": _cells(environment.initial_pellets),
        "initial_pacman_pos": _cells([environment.initial_pacman_pos])[0],
        "initial_ghost_positions": _cells(environment.initial_ghost_positions),
        "num_ghosts": int(environment.num_ghosts),
        "action_names": [str(name) for name in environment.action_names],
        "pellet_reward": float(environment.pellet_reward),
        "ghost_collision_penalty": float(environment.ghost_collision_penalty),
        "step_penalty": float(environment.step_penalty),
        "win_reward": float(environment.win_reward),
        "ghost_aggressiveness": float(environment.ghost_aggressiveness),
        "ghost_coordination": str(environment.ghost_coordination),
        "ghost_strategies": [str(s) for s in environment.ghost_strategies],
        "observation_noise_factor": float(environment.observation_noise_factor),
        "max_observation_noise": float(environment.max_observation_noise),
        "dangerous_areas": _cells(sorted(environment.dangerous_areas)),
        "dangerous_area_radius": float(environment.dangerous_area_radius),
        "dangerous_area_penalty": float(environment.dangerous_area_penalty),
        "is_dangerous_area_hit_terminal": bool(environment.is_dangerous_area_hit_terminal),
    }


def _read_state(
    environment: Any, state: Any
) -> Tuple[List[int], List[List[int]], List[List[int]], float, bool]:
    """Pull one recorded state apart using the environment's own readers."""
    array = np.asarray(state, dtype=float).reshape(-1)
    pacman = list(environment.get_pacman_pos(array))
    ghosts = _cells(environment.get_ghost_positions(array))
    pellets = _cells(environment.get_pellets(array))
    return (
        [int(pacman[0]), int(pacman[1])],
        ghosts,
        pellets,
        float(environment.get_score(array)),
        bool(environment.get_terminal(array)),
    )


def build_pacman_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one PacMan episode.

    Args:
        environment: The PacMan environment the episode was run on. Its maze,
            rules and state layout are copied into the payload so a viewer can
            build the scene without importing Python.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind ``pacman.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    if not history:
        raise ValueError("Cannot export a trace for an empty history")

    pacman_positions: List[List[int]] = []
    ghost_positions: List[List[List[int]]] = []
    pellets: List[List[List[int]]] = []
    scores: List[float] = []
    terminals: List[bool] = []
    observations: List[Any] = []
    beliefs: List[Dict[str, Any]] = []

    for step in history:
        pac, ghosts, remaining, score, terminal = _read_state(environment, step.state)
        pacman_positions.append(pac)
        ghost_positions.append(ghosts)
        pellets.append(remaining)
        scores.append(score)
        terminals.append(terminal)
        observations.append(to_jsonable(_observation(step.observation)))
        beliefs.append(belief_to_payload(step.belief))

    return EpisodeTrace(
        environment=str(environment.name),
        payload_kind=PACMAN_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload={
            "world": _world(environment),
            "state_layout": environment.state_layout(),
            "pacman_positions": pacman_positions,
            "ghost_positions": ghost_positions,
            "pellets": pellets,
            "scores": scores,
            "terminals": terminals,
            "observations": observations,
            "beliefs": beliefs,
        },
        policy=policy_name,
        reach_terminal_state=bool(environment.is_terminal(history[-1].state)),
        metadata={"environment_class": type(environment).__name__},
    )
