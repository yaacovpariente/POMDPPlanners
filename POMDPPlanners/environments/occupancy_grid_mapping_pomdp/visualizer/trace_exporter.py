# SPDX-License-Identifier: MIT

"""Occupancy-grid mapping episode trace exporter.

The GIF renderer beside this module draws an episode. This one writes the same
episode as data, so the browser viewer can replay it. Nothing here is
re-derived: every number written is read off the recorded states with the
environment's own accessors, or copied from its configuration.

This environment keeps two map-shaped quantities apart, and so does the
payload, because confusing them is the one mistake a reader of this world can
make:

* ``log_odds`` is the robot's own map -- the inverse-sensor estimate carried
  inside the state, built from the scans the robot actually received. It is
  *known* to the robot, not uncertain, and it is what the reward is paid on and
  what terminates the episode. It is what the viewer draws as fog, blocks and
  swept floor.
* ``true_map`` is the hidden occupancy the sensor model reads and the robot
  never sees. It is written so the viewer can offer a ground-truth layer and
  mark where the robot's map is wrong; the viewer labels it as such.

The belief is not serialized here. It is a core abstraction with a closed
family of implementations, so
:func:`~POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes
it for every environment. One particle of this environment's belief is a whole
world -- a pose, a hidden map, a log-odds map and a scan, several hundred
numbers -- rather than a two-vector, so the cap on how many are written is
lowered here from core's default. Twenty-four whole maps is the cloud the
environment's own QA runs with, and it is a file a browser can load; four
hundred would be tens of megabytes of trace for a picture nothing draws.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace, envelope_steps
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_pomdp import (
    RESOLVED_LOG_ODDS,
)

# Payload version, independent of the envelope's. Bump it when the meaning of a
# payload field changes, so a viewer can refuse a file it would misdraw.
OCCUPANCY_GRID_MAPPING_PAYLOAD_KIND = "occupancy_grid_mapping.v1"

#: How many whole-map particles of one belief are written. See the module
#: docstring: a particle here is a world, not a point.
MAX_TRACE_PARTICLES = 24


def build_occupancy_grid_mapping_trace(
    environment: Any,
    history: List[StepData],
    episode_index: int,
    policy_name: Optional[str] = None,
) -> EpisodeTrace:
    """Build the trace for one occupancy-grid mapping episode.

    Args:
        environment: The environment the episode was run on. Its grid, sensor
            and reward constants are copied into the payload's ``world`` block
            so a viewer can build the scene without importing Python.
        history: The episode's ``StepData`` records, in order.
        episode_index: Zero-based index of the episode within its run.
        policy_name: Name of the policy that produced the episode.

    Returns:
        The episode's :class:`EpisodeTrace`, with payload kind
        ``occupancy_grid_mapping.v1``.

    Raises:
        ValueError: If ``history`` is empty; there is no episode to write.
    """
    if not history:
        raise ValueError("Cannot export a trace for an empty history")

    poses: List[List[int]] = []
    log_odds: List[List[float]] = []
    ranges: List[List[float]] = []
    scanned: List[bool] = []
    entropy_bits: List[float] = []
    beliefs: List[Dict[str, Any]] = []

    for step in history:
        state = np.asarray(step.state, dtype=np.float64)
        row, col, heading = environment.pose(state)
        poses.append([int(row), int(col), int(heading)])
        log_odds.append([float(v) for v in environment.log_odds(state).reshape(-1)])
        ranges.append([float(v) for v in state[environment.scan_offset :]])
        # The initial state carries zero range placeholders rather than a
        # reading, and the renderer beside this module draws no beams for it.
        # The step counter is what distinguishes the two, because a real scan
        # of all zeros is not a thing this sensor can produce.
        scanned.append(int(round(float(state[0]))) > 0)
        entropy_bits.append(float(environment.entropy_bits(state)))
        beliefs.append(belief_to_payload(step.belief, max_particles=MAX_TRACE_PARTICLES))

    # The hidden map is fixed for the episode -- the transition never writes to
    # that block -- so it is written once rather than per step.
    true_map = environment.true_map(np.asarray(history[0].state, dtype=np.float64))

    payload: Dict[str, Any] = {
        # Taken from the environment instance the episode actually ran on, not
        # from the class defaults, which a configured run may not be using.
        "world": {
            "num_rows": int(environment.num_rows),
            "num_cols": int(environment.num_cols),
            "num_beams": int(environment.num_beams),
            "field_of_view_degrees": float(environment.field_of_view_degrees),
            "max_range_cells": float(environment.max_range_cells),
            "range_noise_std_cells": float(environment.range_noise_std_cells),
            "log_odds_clamp": float(environment.log_odds_clamp),
            "hit_probability": float(environment.hit_probability),
            "miss_probability": float(environment.miss_probability),
            "max_steps": int(environment.max_steps),
            "step_cost": float(environment.step_cost),
            "initial_entropy_bits": float(environment.initial_entropy_bits),
            "entropy_threshold_bits": float(environment.entropy_threshold_bits),
            "resolved_log_odds": float(RESOLVED_LOG_ODDS),
            "start": [
                int(environment.start_row),
                int(environment.start_col),
                int(environment.start_heading),
            ],
            # Hidden from the robot. The viewer draws it only in its
            # ground-truth layers, and says so there.
            "true_map": [float(v) for v in np.asarray(true_map, dtype=float).reshape(-1)],
            # Where each block sits inside a state vector, so a reader can take
            # a belief particle apart without reproducing this module's
            # arithmetic. Belief particles are whole state vectors.
            "state_layout": {
                "state_size": int(environment.state_size),
                "map_offset": int(environment.map_offset),
                "log_odds_offset": int(environment.log_odds_offset),
                "scan_offset": int(environment.scan_offset),
                "num_cells": int(environment.num_cells),
            },
        },
        "poses": poses,
        "log_odds": log_odds,
        "ranges": ranges,
        "scanned": scanned,
        "entropy_bits": entropy_bits,
        "beliefs": beliefs,
    }

    return EpisodeTrace(
        environment=str(environment.name),
        payload_kind=OCCUPANCY_GRID_MAPPING_PAYLOAD_KIND,
        episode_index=int(episode_index),
        discount_factor=float(environment.discount_factor),
        steps=envelope_steps(history),
        payload=payload,
        policy=policy_name,
        reach_terminal_state=bool(environment.is_terminal(history[-1].state)),
        metadata={"environment_class": type(environment).__name__},
    )
