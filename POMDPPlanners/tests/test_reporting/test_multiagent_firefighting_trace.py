# SPDX-License-Identifier: MIT

"""Tests for the multi-agent firefighting trace exporter and its viewer package.

Two things are checked here, and they are the two ways this migration could
quietly break something:

* the payload round-trips and says what the episode actually was -- the true
  fire maps, the recorded poses, the hidden wind and the run's own belief;
* moving the renderer into ``visualizer/`` moved it and nothing else, so the
  GIF whose bytes are pinned by a golden hash still renders from the same
  history to the same bytes.
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
from POMDPPlanners.environments.multiagent_firefighting_pomdp import (
    FireCategory,
    FirefightingAction,
    MultiAgentFirefightingPOMDP,
    MultiAgentFirefightingVisualizer,
    WindDirection,
    WindStrength,
    create_firefighting_state,
)
from POMDPPlanners.environments.multiagent_firefighting_pomdp.visualizer.trace_exporter import (
    MULTIAGENT_FIREFIGHTING_PAYLOAD_KIND,
    build_multiagent_firefighting_trace,
)
from POMDPPlanners.tests.test_environments.test_environment_visualizations_golden_files import (
    create_deterministic_multiagent_firefighting_episode,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    multiagent_firefighting_pinned_kwargs,
)


@pytest.fixture(name="env")
def env_fixture() -> MultiAgentFirefightingPOMDP:
    """A firefighting environment with the suite's pinned configuration."""
    return MultiAgentFirefightingPOMDP(
        discount_factor=0.95, **multiagent_firefighting_pinned_kwargs()
    )


def _belief(env: MultiAgentFirefightingPOMDP, winds, fire) -> WeightedParticleBelief:
    """A particle belief over the wind alone, carrying the given fire map."""
    particles = [
        create_firefighting_state(
            env,
            robots=[(2, 2, env.max_tank, env.max_health), (2, 3, env.max_tank, env.max_health)],
            wind=wind,
            fire=fire,
        )
        for wind in winds
    ]
    return WeightedParticleBelief(
        particles=particles,
        log_weights=np.log(np.full(len(particles), 1.0 / len(particles))),
    )


def _episode(env: MultiAgentFirefightingPOMDP, length: int = 3) -> List[StepData]:
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
            robots=[
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
    trace = build_multiagent_firefighting_trace(env, history, episode_index=2, policy_name="PFT")

    written = tmp_path / "trace_2.json"
    trace.write(written)
    restored = EpisodeTrace.read(written)

    assert restored.payload_kind == MULTIAGENT_FIREFIGHTING_PAYLOAD_KIND
    assert restored.episode_index == 2
    assert restored.policy == "PFT"
    assert restored.num_steps == len(history)
    assert restored.payload == trace.payload


def test_trace_carries_the_recorded_state_not_a_reconstruction(env):
    """The fire maps, the poses and the wind are the episode's own.

    Purpose: Everything the viewer draws about the world comes from these three
        blocks. Re-deriving any of them from the environment's defaults would
        produce a convincing picture of an episode nobody ran.

    Given: An episode whose robots move and spend tank and health, over a fire
        map with one burning cell.
    When: The trace is built.
    Then: Each step's fire map, robot block and wind equal the recorded state's,
        and the joint action is written as its per-robot digits.

    Test type: unit
    """
    history = _episode(env)
    payload = build_multiagent_firefighting_trace(env, history, 0).payload

    for index, step in enumerate(history):
        assert payload["fires"][index] == [int(v) for v in env.fire_map(step.state).ravel()]
        assert payload["robots"][index] == [[int(f) for f in row] for row in env.robots(step.state)]
        assert payload["winds"][index] == list(env.wind(step.state))
        assert payload["step_counts"][index] == env.step_count(step.state)

    assert payload["robot_actions"][0] == [
        int(FirefightingAction.SUPPRESS),
        int(FirefightingAction.EAST),
    ]
    # The terminal bookkeeping step carries a state but no decision.
    assert payload["robot_actions"][-1] is None


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
    trace = build_multiagent_firefighting_trace(env, history, 0)

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
    trace = build_multiagent_firefighting_trace(env, history, 0)
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
    payload = build_multiagent_firefighting_trace(env, _episode(env), 0).payload["world"]

    assert payload["num_rows"] == env.num_rows
    assert payload["num_cols"] == env.num_cols
    assert payload["num_robots"] == env.num_robots
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
                robots=[(2, 2, env.max_tank, env.max_health), (2, 3, env.max_tank, env.max_health)],
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
    belief = build_multiagent_firefighting_trace(env, history, 0).payload["beliefs"][0]

    assert belief["num_particles"] == count
    assert belief["num_written"] == MAX_PAYLOAD_PARTICLES


def test_environment_writes_a_trace_file(env, tmp_path: Path):
    """cache_trace writes a readable file under the episode's index."""
    written = env.cache_trace(
        history=_episode(env), output_dir=tmp_path, episode_index=3, policy_name="POMCPOW"
    )
    assert written == tmp_path / "trace_3.json"
    assert EpisodeTrace.read(written).payload_kind == MULTIAGENT_FIREFIGHTING_PAYLOAD_KIND


def test_trace_rejects_an_empty_history(env):
    """There is no trace for an episode with no steps."""
    with pytest.raises(ValueError, match="empty history"):
        build_multiagent_firefighting_trace(env, [], 0)


def test_moving_the_renderer_into_the_visualizer_package_did_not_move_the_gif(tmp_path: Path):
    """The same episode still renders to the same bytes after the package move.

    Purpose: The GIF's bytes are pinned by a golden hash that only runs inside
        the project's Docker image, so a change to the renderer's output made
        on a laptop would not be caught until CI. This renders the golden
        fixture's own episode twice through the moved module and requires the
        two to be byte-identical, which is the property the golden hash
        depends on and the one a move could break.

    Given: The deterministic episode the golden firefighting GIF is built from.
    When: It is rendered twice through the relocated visualizer.
    Then: The two files are byte-identical and non-empty.

    Test type: integration
    """
    history = create_deterministic_multiagent_firefighting_episode(seed=5)
    env = MultiAgentFirefightingPOMDP(
        discount_factor=0.95, **multiagent_firefighting_pinned_kwargs()
    )
    first, second = tmp_path / "a.gif", tmp_path / "b.gif"
    MultiAgentFirefightingVisualizer(env).create_visualization(history, first)
    MultiAgentFirefightingVisualizer(env).create_visualization(history, second)

    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size > 0


def test_the_environment_still_writes_its_gif_from_the_moved_package(tmp_path: Path):
    """``cache_visualization`` finds the renderer at its new import path.

    Purpose: The environment imports the renderer lazily by module path, so a
        move breaks it at render time rather than at import time -- in a
        simulation worker, hours into a run.

    Test type: integration
    """
    env = MultiAgentFirefightingPOMDP(
        discount_factor=0.95, **multiagent_firefighting_pinned_kwargs()
    )
    env.cache_visualization(
        history=create_deterministic_multiagent_firefighting_episode(seed=5),
        output_dir=tmp_path,
        episode_index=4,
    )
    assert (tmp_path / "multiagent_firefighting_4.gif").stat().st_size > 0
