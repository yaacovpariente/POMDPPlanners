# SPDX-License-Identifier: MIT

"""Multi-agent firefighting episode trace exporter.

The GIF renderer beside this module draws an episode. This one writes the same
episode as data, so the browser viewer can replay it. Nothing here is
re-derived: every number written is read off the recorded episode or off the
environment instance the episode actually ran on.

Two decisions are worth stating, because a reader of the viewer will want to
know whether what they are looking at is the run or a reconstruction.

* **The belief is not serialized here.** It is a core abstraction with a closed
  family of implementations, so
  :func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload`
  writes it for every environment. A firefighting particle *is* a whole state
  vector -- robots, wind and the hundred cells -- so the viewer's wind rose is
  a projection the viewer computes from those particles, at the indices this
  payload names in ``world.state_layout``. It is the run's own belief, read a
  different way, not a second belief invented for drawing.
* **The fire map is written as the discrete categories it is.** A cell is
  unburnt, smoldering, burning, burnt or wet, and suppression flips it in one
  step. The viewer tweens the *rendering* of that flip across the step so a
  change reads as a change; nothing in this file smooths anything, and the
  step-by-step counts a viewer reports come from these integers.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps, to_jsonable
from POMDPPlanners.environments.multiagent_firefighting_pomdp.multiagent_firefighting_world import (
    DIRECTION_OFFSETS,
    HEAT_DAMAGE,
    ROBOT_FIELD_WIDTH,
    ROBOT_OFFSET,
    STEP_INDEX,
    FireCategory,
    FirefightingAction,
    WindDirection,
    WindStrength,
)

# Payload version, independent of the envelope's. Bump it when the meaning of a
# payload field changes, so a viewer can refuse a file it would misdraw.
MULTIAGENT_FIREFIGHTING_PAYLOAD_KIND = "multiagent_firefighting.v1"

#: Category names in code order, so a viewer labels a cell without a table of
#: its own that could drift from :class:`FireCategory`.
CATEGORY_NAMES = [category.name.lower() for category in FireCategory]

#: Per-robot action names in code order, for the same reason.
ACTION_NAMES = [action.name.lower() for action in FirefightingAction]

#: The eight hidden wind values, indexed ``direction * len(WindStrength) +
#: strength`` -- the order the GIF's histogram panel uses, so the two
#: presentations of the same belief agree.
WIND_LABELS = [
    f"{direction.name[0]} {strength.name.lower()}"
    for direction in WindDirection
    for strength in WindStrength
]


def _world_block(environment: Any) -> Dict[str, Any]:
    """Everything a viewer needs to build the world, from the instance.

    Read off the environment the episode ran on rather than off the class
    defaults: a configured run may be on a different grid, with a different
    depot or a different spread rate, and a viewer drawing the defaults would
    draw a world nobody solved.

    Args:
        environment: The firefighting environment the episode ran on.

    Returns:
        The payload's ``world`` block.
    """
    return {
        "num_rows": int(environment.num_rows),
        "num_cols": int(environment.num_cols),
        "num_robots": int(environment.num_robots),
        "obstacle_cells": [[int(row), int(col)] for row, col in environment.obstacle_cells],
        "depot_cell": [int(environment.depot_cell[0]), int(environment.depot_cell[1])],
        "robot_start_cells": [[int(row), int(col)] for row, col in environment.robot_start_cells],
        "sensing_radius": int(environment.sensing_radius),
        "max_tank": int(environment.max_tank),
        "max_health": int(environment.max_health),
        "max_steps": int(environment.max_steps),
        "num_initial_fires": int(environment.num_initial_fires),
        # The names and offsets the codes above mean. Written out so a viewer
        # never hard-codes an enum that lives in Python.
        "category_names": list(CATEGORY_NAMES),
        "action_names": list(ACTION_NAMES),
        "wind_labels": list(WIND_LABELS),
        "direction_offsets": [[int(row), int(col)] for row, col in DIRECTION_OFFSETS],
        "heat_damage": [int(value) for value in HEAT_DAMAGE],
        "num_wind_strengths": len(WindStrength),
        "unknown_category": -1.0,
        # Where the fields sit inside one state vector. A belief particle is a
        # whole state, and core writes particles verbatim, so this is what lets
        # the viewer read the wind out of the run's own belief.
        "state_layout": {
            "step_index": int(STEP_INDEX),
            "robot_offset": int(ROBOT_OFFSET),
            "robot_field_width": int(ROBOT_FIELD_WIDTH),
            "wind_direction_index": int(environment.wind_direction_index),
            "wind_strength_index": int(environment.wind_strength_index),
            "fire_offset": int(environment.fire_offset),
            "state_size": int(environment.state_size),
        },
        # The dynamics and the reward, so the page can state the world it is
        # showing instead of asserting numbers from a design document.
        "observation_error_probability": float(environment.observation_error_probability),
        "slip_probability": float(environment.slip_probability),
        "spread_probability": float(environment.spread_probability),
        "wind_gain_low": float(environment.wind_gain_low),
        "wind_gain_high": float(environment.wind_gain_high),
        "crosswind_attenuation": float(environment.crosswind_attenuation),
        "growth_probability": float(environment.growth_probability),
        "burnout_probability": float(environment.burnout_probability),
        "suppression_probability_unburnt": float(environment.suppression_probability_unburnt),
        "suppression_probability_smoldering": float(
            environment.suppression_probability_smoldering
        ),
        "suppression_probability_burning": float(environment.suppression_probability_burning),
        "success_reward": float(environment.success_reward),
        "step_cost": float(environment.step_cost),
        "smoldering_cell_cost": float(environment.smoldering_cell_cost),
        "burning_cell_cost": float(environment.burning_cell_cost),
        "burnt_cell_cost": float(environment.burnt_cell_cost),
        "damage_cost": float(environment.damage_cost),
        "water_cost": float(environment.water_cost),
    }


def build_multiagent_firefighting_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one multi-agent firefighting episode.

    Args:
        environment: The firefighting environment the episode ran on.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind
        ``multiagent_firefighting.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    if not history:
        raise ValueError("Cannot export a trace for an empty history")

    fires: List[List[int]] = []
    robots: List[List[List[int]]] = []
    winds: List[List[int]] = []
    step_counts: List[int] = []
    robot_actions: List[Optional[List[int]]] = []
    observations: List[Any] = []
    beliefs: List[Dict[str, Any]] = []

    for step in history:
        state = np.asarray(step.state, dtype=np.float64)
        # Flat row-major, which is how the state stores it and how a viewer
        # indexes a grid; reshaping is the viewer's business.
        fires.append([int(value) for value in environment.fire_map(state).ravel()])
        robots.append([[int(field) for field in row] for row in environment.robots(state)])
        winds.append([int(value) for value in environment.wind(state)])
        step_counts.append(int(environment.step_count(state)))
        # The joint action is one integer; its per-robot digits are what a
        # viewer draws. Decoded here rather than in the viewer, so the base-5
        # convention lives in exactly one language.
        robot_actions.append(
            None
            if step.action is None
            else [int(value) for value in environment.decode_action(step.action)]
        )
        observations.append(to_jsonable(step.observation))
        beliefs.append(belief_to_payload(step.belief))

    payload: Dict[str, Any] = {
        "world": _world_block(environment),
        "fires": fires,
        "robots": robots,
        "winds": winds,
        "step_counts": step_counts,
        "robot_actions": robot_actions,
        "observations": observations,
        "beliefs": beliefs,
    }

    return EpisodeTrace(
        environment=str(environment.name),
        payload_kind=MULTIAGENT_FIREFIGHTING_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload=payload,
        policy=policy_name,
        reach_terminal_state=bool(environment.is_terminal(history[-1].state)),
        metadata={"environment_class": type(environment).__name__},
    )
