# SPDX-License-Identifier: MIT

"""Tests for the firefighting episode trace.

The payload must round-trip and say what the episode actually was -- the true
fire maps, the recorded poses, the hidden wind and the run's own belief.
"""

from pathlib import Path
from typing import List

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import (
    MAX_PAYLOAD_PARTICLES,
    belief_to_payload,
)
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.firefighting_pomdp import (
    FireCategory,
    FirefightingAction,
    FirefightingPOMDP,
    FirefightingVisualizer,
    WindDirection,
    WindStrength,
    create_firefighting_state,
)
from POMDPPlanners.environments.firefighting_pomdp.firefighting_visualization.firefighting_visualizer import (
    FIREFIGHTING_PAYLOAD_KIND,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    firefighting_pinned_kwargs,
)


@pytest.fixture(name="env")
def env_fixture() -> FirefightingPOMDP:
    """A firefighting environment with the suite's pinned configuration."""
    return FirefightingPOMDP(discount_factor=0.95, **firefighting_pinned_kwargs())


def _belief(env: FirefightingPOMDP, winds, fire) -> WeightedParticleBelief:
    """A particle belief over the wind alone, carrying the given fire map."""
    particles = [
        create_firefighting_state(
            env,
            firefighters=[
                (2, 2, env.max_tank, env.max_health),
                (2, 3, env.max_tank, env.max_health),
            ],
            wind=wind,
            fire=fire,
        )
        for wind in winds
    ]
    return WeightedParticleBelief(
        particles=particles,
        log_weights=np.log(np.full(len(particles), 1.0 / len(particles))),
    )


def _episode(env: FirefightingPOMDP, length: int = 3) -> List[StepData]:
    """A short, fully stated episode: no sampling, so the assertions are exact."""
    fire = np.full((env.num_rows, env.num_cols), float(FireCategory.UNBURNT))
    fire[4, 4] = float(FireCategory.BURNING)
    winds = [
        (int(WindDirection.EAST), int(WindStrength.HIGH)),
        (int(WindDirection.NORTH), int(WindStrength.LOW)),
    ]

    history: List[StepData] = []
    for step in range(length):
        is_last = step == length - 1
        state = create_firefighting_state(
            env,
            firefighters=[
                (2, 2 + step, env.max_tank - step, env.max_health),
                (2, 3 + step, env.max_tank, env.max_health - step),
            ],
            wind=(int(WindDirection.EAST), int(WindStrength.HIGH)),
            fire=fire,
            step=step,
        )
        action = (
            None if is_last else int(FirefightingAction.SUPPRESS) + 5 * int(FirefightingAction.EAST)
        )
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=None if is_last else state,
                observation=None if is_last else env.sample_observation(state, action),
                reward=None if is_last else -1.5,
                belief=_belief(env, winds, fire),
                info=None if is_last else {"alight_cells": 1.0},
            )
        )
    return history


def test_trace_round_trips_through_json(env, tmp_path: Path):
    """The trace a firefighting episode writes reads back as the same trace.

    Purpose: The viewer is served this file and nothing else. A field that does
        not survive the write is a field the viewer silently draws wrong.

    Given: A stated three-step episode.
    When: The trace is written and read back.
    Then: The envelope and every payload block come back unchanged.

    Test type: unit
    """
    history = _episode(env)
    trace = FirefightingVisualizer(env).build_trace(history, episode_index=2, policy_name="PFT")

    written = tmp_path / "trace_2.json"
    trace.write(written)
    restored = EpisodeTrace.read(written)

    assert restored.payload_kind == FIREFIGHTING_PAYLOAD_KIND
    assert restored.episode_index == 2
    assert restored.policy == "PFT"
    assert restored.num_steps == len(history)
    assert restored.payload == trace.payload


def test_trace_carries_the_recorded_state_not_a_reconstruction(env):
    """The fire maps, the poses and the wind are the episode's own.

    Purpose: Everything the viewer draws about the world comes from these three
        blocks. Re-deriving any of them from the environment's defaults would
        produce a convincing picture of an episode nobody ran.

    Given: An episode whose firefighters move and spend tank and health, over a fire
        map with one burning cell.
    When: The trace is built.
    Then: Each step's fire map, firefighter block and wind equal the recorded state's,
        and the joint action is written as its per-firefighter digits.

    Test type: unit
    """
    history = _episode(env)
    payload = FirefightingVisualizer(env).build_trace(history, episode_index=0).payload

    for index, step in enumerate(history):
        assert payload["fires"][index] == [int(v) for v in env.fire_map(step.state).ravel()]
        assert payload["firefighters"][index] == [
            [int(f) for f in row] for row in env.firefighters(step.state)
        ]
        assert payload["winds"][index] == list(env.wind(step.state))
        assert payload["step_counts"][index] == env.step_count(step.state)

    assert payload["firefighter_actions"][0] == [
        int(FirefightingAction.SUPPRESS),
        int(FirefightingAction.EAST),
    ]
    # The terminal bookkeeping step carries a state but no decision.
    assert payload["firefighter_actions"][-1] is None


def test_trace_delegates_belief_serialization_to_core(env):
    """The exporter writes no belief format of its own.

    Purpose: Belief is a core abstraction with a closed family of
        implementations. Serializing it per environment would mean fifteen
        copies of the same dispatch and fifteen chances to miss a class.

    Given: An episode whose steps carry weighted particle beliefs.
    When: The trace is built.
    Then: Each step's belief payload equals ``belief_to_payload`` of that step's
        belief, field for field.

    Test type: unit
    """
    history = _episode(env)
    trace = FirefightingVisualizer(env).build_trace(history, episode_index=0)

    for step, written in zip(history, trace.payload["beliefs"]):
        assert written == belief_to_payload(step.belief)


def test_trace_names_where_the_wind_sits_in_a_particle(env):
    """The layout block is what lets a viewer read the wind out of the belief.

    Purpose: A firefighting particle is a whole state vector, so the viewer's
        wind rose is a projection it takes at these indices. If they drifted
        from the environment's own, the rose would plot a tank or a row number
        and still look like a posterior.

    Given: The configured environment.
    When: The trace is built.
    Then: The layout indices are the environment's, and reading a particle at
        them recovers the wind that particle was built with.

    Test type: unit
    """
    history = _episode(env)
    trace = FirefightingVisualizer(env).build_trace(history, episode_index=0)
    layout = trace.payload["world"]["state_layout"]

    assert layout["wind_direction_index"] == env.wind_direction_index
    assert layout["wind_strength_index"] == env.wind_strength_index
    assert layout["fire_offset"] == env.fire_offset
    assert layout["state_size"] == env.state_size

    particles = trace.payload["beliefs"][0]["particles"]
    winds = {
        (int(p[layout["wind_direction_index"]]), int(p[layout["wind_strength_index"]]))
        for p in particles
    }
    assert winds == {
        (int(WindDirection.EAST), int(WindStrength.HIGH)),
        (int(WindDirection.NORTH), int(WindStrength.LOW)),
    }
    assert len(trace.payload["world"]["wind_labels"]) == len(WindDirection) * len(WindStrength)


def test_trace_takes_the_world_from_the_instance(env):
    """The world block comes from the configured environment, not class defaults.

    Purpose: A run may be configured away from the defaults, and a viewer that
        drew the defaults would draw a different world from the one the planner
        solved.

    Test type: unit
    """
    payload = (
        FirefightingVisualizer(env).build_trace(_episode(env), episode_index=0).payload["world"]
    )

    assert payload["num_rows"] == env.num_rows
    assert payload["num_cols"] == env.num_cols
    assert payload["num_firefighters"] == env.num_firefighters
    assert payload["depot_cell"] == [env.depot_cell[0], env.depot_cell[1]]
    assert payload["obstacle_cells"] == [[r, c] for r, c in env.obstacle_cells]
    assert payload["sensing_radius"] == env.sensing_radius
    assert payload["spread_probability"] == pytest.approx(env.spread_probability)
    assert payload["burnt_cell_cost"] == pytest.approx(env.burnt_cell_cost)


def test_trace_inherits_the_subsampling_cap(env):
    """A belief larger than core's cap is trimmed in the exporter's output too.

    Purpose: One firefighting particle is a whole world, so an uncapped cloud
        would make a trace file enormous and a browser scene unusable.

    Test type: unit
    """
    fire = np.full((env.num_rows, env.num_cols), float(FireCategory.UNBURNT))
    fire[4, 4] = float(FireCategory.BURNING)
    count = MAX_PAYLOAD_PARTICLES + 20
    winds = [(index % 4, index % 2) for index in range(count)]
    history = [
        StepData(
            state=create_firefighting_state(
                env,
                firefighters=[
                    (2, 2, env.max_tank, env.max_health),
                    (2, 3, env.max_tank, env.max_health),
                ],
                wind=(int(WindDirection.EAST), int(WindStrength.HIGH)),
                fire=fire,
            ),
            action=0,
            next_state=None,
            observation=None,
            reward=-1.0,
            belief=_belief(env, winds, fire),
        )
    ]
    belief = FirefightingVisualizer(env).build_trace(history, episode_index=0).payload["beliefs"][0]

    assert belief["num_particles"] == count
    assert belief["num_written"] == MAX_PAYLOAD_PARTICLES


def test_environment_writes_a_trace_file(env, tmp_path: Path):
    """The environment's visualizer writes a readable file under the episode's index."""
    written = env.episode_visualizer().write(
        history=_episode(env), output_dir=tmp_path, episode_index=3, policy_name="POMCPOW"
    )
    assert written == tmp_path / "trace_3.json"
    assert EpisodeTrace.read(written).payload_kind == FIREFIGHTING_PAYLOAD_KIND


def test_trace_rejects_an_empty_history(env):
    """There is no trace for an episode with no steps."""
    with pytest.raises(ValueError, match="empty history"):
        FirefightingVisualizer(env).build_trace([], episode_index=0)
