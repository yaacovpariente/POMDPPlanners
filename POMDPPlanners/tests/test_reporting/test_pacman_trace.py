# SPDX-License-Identifier: MIT

"""Tests for the PacMan trace exporter and the GIF its move must not change."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import (
    MAX_PAYLOAD_PARTICLES,
    belief_to_payload,
)
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP
from POMDPPlanners.environments.pacman_pomdp.pacman_visualization import PacManVisualizer
from POMDPPlanners.environments.pacman_pomdp.pacman_visualization.trace_exporter import (
    PACMAN_PAYLOAD_KIND,
    build_pacman_trace,
)
from POMDPPlanners.tests.test_environments.test_environment_visualizations_golden_files import (
    create_deterministic_pacman_episode,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import pacman_pinned_kwargs

GOLDEN_GIF = (
    Path(__file__).resolve().parents[1]
    / "test_environments"
    / "golden_visualizations"
    / "pacman_visualization.gif"
)


def _env(**overrides) -> PacManPOMDP:
    return PacManPOMDP(
        discount_factor=0.95,
        **pacman_pinned_kwargs(
            maze_size=(7, 7),
            num_ghosts=2,
            initial_ghost_positions=None,
            ghost_strategies=None,
            **overrides,
        ),
    )


def _belief(env: PacManPOMDP, ghost_sets):
    """A weighted particle belief whose particles are real PacMan states."""
    particles = [
        env.make_state(pacman_pos=(0, 0), ghost_positions=tuple(ghosts)) for ghosts in ghost_sets
    ]
    return WeightedParticleBelief(
        particles=particles,
        log_weights=np.log(np.ones(len(particles)) / len(particles)),
    )


def _episode(env: PacManPOMDP, length: int = 4):
    """A short hand-built episode, so exporter tests do not depend on a planner."""
    history = []
    for step in range(length):
        is_last = step == length - 1
        state = env.make_state(
            pacman_pos=(0, min(step, env.maze_size[1] - 1)),
            ghost_positions=((6, 6), (0, 6)),
            pellets=tuple(env.initial_pellets),
            terminal=is_last,
        )
        history.append(
            StepData(
                state=state,
                action=None if is_last else 1,
                next_state=None if is_last else state,
                observation=None if is_last else ((6, 5), (1, 6)),
                reward=None if is_last else -1.0,
                belief=_belief(env, [[(6, 6), (0, 6)], [(5, 6), (1, 6)]]),
            )
        )
    return history


@pytest.fixture(name="pacman_env")
def pacman_env_fixture() -> PacManPOMDP:
    """A PacMan environment with the suite's pinned two-ghost configuration."""
    return _env()


def test_pacman_trace_round_trips_through_json(pacman_env, tmp_path: Path):
    """A written trace reads back with the same payload it was built from.

    Purpose: The trace file is the only thing between a finished episode and
    the browser viewer. A field that does not survive ``write``/``read`` is a
    field the viewer will never see, and nothing else in the pipeline would
    notice.

    Given: A PacMan episode with states, observations and particle beliefs.
    When: Its trace is written to disk and read back.
    Then: The envelope and every payload block match what was built.
    """
    trace = build_pacman_trace(
        pacman_env, _episode(pacman_env, 4), episode_index=3, policy_name="POMCPOW"
    )
    restored = EpisodeTrace.read(trace.write(tmp_path / "trace_3.json"))

    assert restored.payload_kind == PACMAN_PAYLOAD_KIND
    assert restored.episode_index == 3
    assert restored.policy == "POMCPOW"
    assert restored.discount_factor == pytest.approx(pacman_env.discount_factor)
    assert restored.payload == trace.payload
    assert [step.to_dict() for step in restored.steps] == [step.to_dict() for step in trace.steps]


def test_pacman_trace_records_every_recorded_cell(pacman_env):
    """Each step's cells come from that step's recorded state.

    Purpose: PacMan's positions are the spine of the replay. Re-deriving them
    from the actions would produce a plausible route that is not the one the
    episode took whenever a move was blocked by a wall.

    Given: An episode whose PacMan walks east along row zero.
    When: The trace is built.
    Then: The payload's cells equal the environment's own readers on each
        recorded state.
    """
    history = _episode(pacman_env, 4)
    payload = build_pacman_trace(pacman_env, history, episode_index=0).payload

    assert len(payload["pacman_positions"]) == len(history)
    for step, pac, ghosts, pellets in zip(
        history, payload["pacman_positions"], payload["ghost_positions"], payload["pellets"]
    ):
        assert pac == list(pacman_env.get_pacman_pos(step.state))
        assert ghosts == [list(g) for g in pacman_env.get_ghost_positions(step.state)]
        assert sorted(pellets) == sorted(list(p) for p in pacman_env.get_pellets(step.state))


def test_pacman_trace_publishes_the_state_layout_it_writes_particles_in(pacman_env):
    """The layout in the payload really is the layout of a PacMan state.

    Purpose: A belief particle reaches the viewer as a bare array, and the
    layout block is the only thing telling it which two numbers are a ghost's
    cell. If the block drifts from ``make_state``, the viewer draws a belief
    over the wrong cells and still looks convincing.

    Given: A state built at known cells by the environment itself.
    When: The trace's layout block is used to slice that state.
    Then: The slices recover the cells the state was built with.
    """
    layout = build_pacman_trace(pacman_env, _episode(pacman_env, 2), 0).payload["state_layout"]
    state = pacman_env.make_state(pacman_pos=(2, 1), ghost_positions=((4, 5), (6, 0)))

    assert len(state) == layout["dim"]
    assert (state[layout["pacman_row"]], state[layout["pacman_col"]]) == (2, 1)
    start = layout["ghosts_start"]
    assert [(state[start + 2 * g], state[start + 2 * g + 1]) for g in range(2)] == [
        (4, 5),
        (6, 0),
    ]
    assert pacman_env.get_terminal(state) is bool(state[layout["terminal"]] > 0.5)


def test_pacman_trace_delegates_belief_serialization_to_core(pacman_env):
    """The exporter writes no belief format of its own.

    Purpose: Belief is a core abstraction with a closed family of
    implementations. Serializing it per environment would mean fifteen copies
    of the same dispatch and fifteen chances to miss a class. Comparing against
    core's own output is what stops the two drifting.

    Given: An episode whose steps carry weighted particle beliefs.
    When: The trace is built.
    Then: Each step's belief payload equals ``belief_to_payload`` of that
        step's belief, field for field.
    """
    history = _episode(pacman_env, 3)
    trace = build_pacman_trace(pacman_env, history, episode_index=0)

    for step, written in zip(history, trace.payload["beliefs"]):
        assert written == belief_to_payload(step.belief)


def test_pacman_trace_inherits_the_subsampling_cap(pacman_env):
    """A belief larger than core's cap is trimmed in the exporter's output too.

    Purpose: A vectorized PacMan belief carries far more particles than a
    browser scene can draw, and each particle is a whole state array, so the
    trace file would be enormous. This checks the exporter inherits the cap
    rather than writing the whole cloud.
    """
    count = MAX_PAYLOAD_PARTICLES * 2
    ghost_sets = [[(row % 7, col % 7), (0, 6)] for row, col in enumerate(range(count))]
    history = [
        StepData(
            state=pacman_env.make_state(pacman_pos=(0, 0), ghost_positions=((6, 6), (0, 6))),
            action=1,
            next_state=None,
            observation=((6, 6), (0, 6)),
            reward=-1.0,
            belief=_belief(pacman_env, ghost_sets),
        )
    ]
    belief = build_pacman_trace(pacman_env, history, episode_index=0).payload["beliefs"][0]

    assert belief["num_particles"] == count
    assert belief["num_written"] == MAX_PAYLOAD_PARTICLES
    assert len(belief["particles"]) == MAX_PAYLOAD_PARTICLES


def test_pacman_trace_takes_the_world_from_the_instance():
    """The world block comes from the configured environment, not class defaults.

    Purpose: A run may be configured away from the defaults — a different
    maze, more ghosts, a hazard zone — and a viewer built from the defaults
    would draw a world the episode never happened in.

    Given: An environment with a hazard zone and a wall set of its own.
    When: The trace is built.
    Then: The payload's world reports that instance's values.
    """
    env = _env(walls={(1, 2)}, dangerous_areas={(3, 3)}, dangerous_area_radius=1.5)
    world = build_pacman_trace(env, _episode(env, 2), 0).payload["world"]

    assert world["walls"] == [[1, 2]]
    assert world["dangerous_areas"] == [[3, 3]]
    assert world["dangerous_area_radius"] == pytest.approx(1.5)
    assert world["num_ghosts"] == env.num_ghosts
    assert world["maze_size"] == list(env.maze_size)
    assert world["action_names"] == env.action_names


def test_pacman_environment_writes_a_trace_file(pacman_env, tmp_path: Path):
    """cache_trace writes a readable file under the episode's index."""
    written = pacman_env.cache_trace(
        history=_episode(pacman_env, 3), output_dir=tmp_path, episode_index=2, policy_name="POMCP"
    )
    assert written == tmp_path / "trace_2.json"

    restored = EpisodeTrace.read(written)
    assert restored.payload_kind == PACMAN_PAYLOAD_KIND
    assert restored.policy == "POMCP"
    assert restored.episode_index == 2


def test_pacman_trace_rejects_an_empty_history(pacman_env):
    """There is no trace for an episode with no steps."""
    with pytest.raises(ValueError, match="empty history"):
        build_pacman_trace(pacman_env, [], episode_index=0)


def test_moving_the_renderer_left_the_gif_byte_identical(tmp_path: Path):
    """The renderer's move into ``visualizer/`` changed no rendered byte.

    Purpose: The PacMan GIF's bytes are pinned by a golden file, and this
    migration is supposed to be a move plus an added exporter. Rendering the
    pinned deterministic episode and comparing to the committed golden is the
    only check that the move did not disturb a sprite path, a palette or a
    frame.

    Given: The pinned deterministic PacMan episode and environment.
    When: The visualizer, now imported from the ``visualizer`` package,
        renders it.
    Then: The bytes equal the committed golden GIF's.
    """
    history = create_deterministic_pacman_episode(seed=42)
    rendered = tmp_path / "agent_path_0.gif"
    PacManVisualizer(_env()).cache_visualization(history, rendered)

    assert (
        hashlib.sha256(rendered.read_bytes()).hexdigest()
        == hashlib.sha256(GOLDEN_GIF.read_bytes()).hexdigest()
    )
