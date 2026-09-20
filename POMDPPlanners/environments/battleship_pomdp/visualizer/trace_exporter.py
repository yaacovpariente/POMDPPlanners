# SPDX-License-Identifier: MIT

"""Battleship episode trace exporter.

The GIF renderer beside this module draws an episode. This one writes the same
episode as data, so the browser viewer can replay it. Nothing here is
re-derived: every number written comes from the recorded episode or from the
environment's own configuration.

The belief itself is not serialized here.
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment, and this exporter is left with what is genuinely
Battleship's: the board's geometry, the hidden occupancy, what had been probed
when each decision was made, and the one-bit readings that came back.

Two Battleship-specific projections are written alongside core's belief
payload, and both are exact quantities the belief can hand over rather than
summaries invented here:

* ``occupancy_marginals`` — :meth:`BattleshipBelief.occupancy_marginal`, the
  per-cell posterior the GIF already draws. It is computed from the whole
  consistent-layout set, so it is the exact marginal; the particle cloud core
  writes is a sample of that same posterior and a viewer projecting *it* would
  show a noisy estimate, including cells that look deduced-certain by the luck
  of a hundred draws. The distinction matters because the viewer draws a
  column only where doubt remains.
* ``support_sizes`` — how many legal layouts are still consistent. A particle
  count cannot say this: the belief carries a fixed number of particles
  whatever the support has collapsed to.

Both are written as ``None`` for a belief that cannot supply them — a generic
particle filter run on this environment, say — so the viewer falls back to
projecting core's particles and says which it is showing.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps, to_jsonable

# Payload version, independent of the envelope's. Bump it when the meaning of a
# payload field changes, so a viewer can refuse a file it would misdraw.
BATTLESHIP_PAYLOAD_KIND = "battleship.v1"

#: What ``marginal_source`` says when the belief supplied the exact posterior.
MARGINAL_SOURCE_EXACT = "exact_layout_posterior"
#: What it says when no step's belief could, and the viewer must project
#: core's particle payload instead.
MARGINAL_SOURCE_PARTICLES = "belief_particles"


def _exact_marginal(belief: Any, environment: Any) -> Optional[List[float]]:
    """Return the belief's exact per-cell posterior, or ``None``.

    Args:
        belief: The belief recorded on one step.
        environment: The Battleship environment the episode ran on.

    Returns:
        One probability per cell, or ``None`` when this belief has no exact
        marginal to give — every belief but
        :class:`~POMDPPlanners.environments.battleship_pomdp.battleship_belief.BattleshipBelief`.
    """
    marginal = getattr(belief, "occupancy_marginal", None)
    if marginal is None:
        return None
    return [float(value) for value in np.asarray(marginal(environment), dtype=float).reshape(-1)]


def _support_size(belief: Any, environment: Any) -> Optional[int]:
    """Return how many legal layouts the belief still admits, or ``None``."""
    indices = getattr(belief, "consistent_indices", None)
    if indices is None:
        return None
    return int(np.asarray(indices(environment)).size)


def build_battleship_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one Battleship episode.

    Args:
        environment: The Battleship environment the episode was run on. Its
            geometry and reward constants are copied into the payload's
            ``world`` block so a viewer can build the scene without importing
            Python.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind
        ``battleship.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    if not history:
        raise ValueError("Cannot export a trace for an empty history")

    num_cells = int(environment.num_cells)

    probed: List[List[int]] = []
    observations: List[Any] = []
    beliefs: List[Dict[str, Any]] = []
    marginals: List[Optional[List[float]]] = []
    supports: List[Optional[int]] = []

    for step in history:
        state = np.asarray(step.state, dtype=float).reshape(-1)
        probed.append([int(value > 0.5) for value in state[num_cells:]])
        observations.append(to_jsonable(step.observation))
        beliefs.append(belief_to_payload(step.belief))
        marginals.append(_exact_marginal(step.belief, environment))
        supports.append(_support_size(step.belief, environment))

    # The occupancy half never changes — a probe writes only the probe half —
    # so it is written once. It is read from the last recorded state rather
    # than the first for no reason but symmetry with ``reach_terminal_state``
    # below; every step carries the same board.
    final_state = np.asarray(history[-1].state, dtype=float).reshape(-1)
    occupancy = [int(value > 0.5) for value in final_state[:num_cells]]

    payload: Dict[str, Any] = {
        # Everything a viewer needs to build the world, taken from the
        # environment instance the episode actually ran on — not from the class
        # defaults, which a configured run may not be using.
        "world": {
            "board_size": int(environment.board_size),
            "ship_lengths": [int(length) for length in environment.ship_lengths],
            "allow_adjacent_ships": bool(environment.allow_adjacent_ships),
            "num_cells": num_cells,
            "num_ship_cells": int(environment.num_ship_cells),
            "hit_reward": float(environment.hit_reward),
            "miss_penalty": float(environment.miss_penalty),
            "num_layouts": int(environment.layouts.num_layouts),
        },
        # The hidden board. It is the truth layer and nothing else: the agent
        # is told one bit per probe and is never told which ship it hit, so a
        # viewer must keep this out of anything it presents as knowledge.
        #
        # The environment stores occupancy with no ship identities at all —
        # ``FleetLayoutTable._enumerate`` writes a flat bitmap — so splitting
        # these cells into hulls is the viewer's choice and is not in general
        # unique. Nothing here pretends otherwise.
        "occupancy": occupancy,
        "probed": probed,
        "observations": observations,
        "beliefs": beliefs,
        "occupancy_marginals": marginals,
        "support_sizes": supports,
        "marginal_source": (
            MARGINAL_SOURCE_EXACT
            if any(entry is not None for entry in marginals)
            else MARGINAL_SOURCE_PARTICLES
        ),
    }

    return EpisodeTrace(
        environment=str(environment.name),
        payload_kind=BATTLESHIP_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload=payload,
        policy=policy_name,
        reach_terminal_state=bool(environment.is_terminal(history[-1].state)),
        metadata={"environment_class": type(environment).__name__},
    )
