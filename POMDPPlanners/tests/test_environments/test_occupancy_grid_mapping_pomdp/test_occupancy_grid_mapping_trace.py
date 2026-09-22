# SPDX-License-Identifier: MIT

"""The occupancy-grid mapping episode trace, and the GIF beside it.

Two things are checked here, and they are two halves of the same migration.

The trace is new, so it is checked for the properties a viewer depends on: it
round-trips through JSON unchanged, its belief comes from core rather than from
this environment, and the robot's map and the hidden map are written from the
blocks they actually live in. The last one is worth a test of its own because
both are ``num_cells`` floats sitting next to each other in the same state
vector: swapping them would produce a trace that looks entirely plausible and
shows a viewer the answer instead of the robot's guess at it.

The GIF is old, and the renderer only moved into a package. Its bytes are
pinned by the golden-file suite, which runs the byte comparison inside the
project's Docker image. What that suite does not check, and what the move could
break, is the hook in between: the environment's ``cache_visualization`` has to
still reach the moved renderer, and reach the same one the golden suite renders
with. So the check here is that the two paths agree byte for byte.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_visualization import (
    OCCUPANCY_GRID_MAPPING_PAYLOAD_KIND,
    OccupancyGridMappingVisualizer,
    build_occupancy_grid_mapping_trace,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_visualization.trace_exporter import (
    MAX_TRACE_PARTICLES,
)
from POMDPPlanners.tests.test_environments.test_environment_visualizations_golden_files import (
    build_occupancy_grid_mapping_env,
    create_deterministic_occupancy_grid_mapping_episode,
)


@pytest.fixture(name="episode")
def episode_fixture():
    """The pinned deterministic episode the golden GIF is rendered from."""
    return create_deterministic_occupancy_grid_mapping_episode(seed=3)


@pytest.fixture(name="env")
def env_fixture():
    """The environment that episode was run on."""
    return build_occupancy_grid_mapping_env()


def test_trace_round_trips_through_json(env, episode, tmp_path: Path):
    """Test that a written trace reads back as the same trace.

    Purpose: The viewer reads a file, not an object, so anything in the payload
        that ``json`` cannot carry -- a numpy scalar, an ndarray -- would be
        lost or stringified between the two

    Given: A trace built from a real occupancy-grid episode
    When: It is written to disk and read back
    Then: The payload and the envelope are unchanged

    Test type: unit
    """
    trace = build_occupancy_grid_mapping_trace(env, episode, episode_index=2, policy_name="PFT_DPW")
    path = trace.write(tmp_path / "trace_2.json")
    restored = EpisodeTrace.read(path)

    assert restored.payload_kind == OCCUPANCY_GRID_MAPPING_PAYLOAD_KIND
    assert restored.episode_index == 2
    assert restored.policy == "PFT_DPW"
    assert restored.num_steps == len(episode)
    assert restored.payload == trace.payload
    # Nothing survives to the browser that is not plain JSON data.
    json.loads(path.read_text(encoding="utf-8"))


def test_trace_writes_the_world_from_the_instance(env, episode):
    """Test that the world block is copied from the environment that ran.

    Purpose: A configured run may use none of the class defaults, and a viewer
        that rebuilt the room from defaults would draw a different room

    Given: An environment with its own grid and sensor settings
    When: A trace is built from it
    Then: The world block matches the instance, not the defaults

    Test type: unit
    """
    world = build_occupancy_grid_mapping_trace(env, episode, 0).payload["world"]

    assert world["num_rows"] == env.num_rows
    assert world["num_cols"] == env.num_cols
    assert world["num_beams"] == env.num_beams
    assert world["max_range_cells"] == pytest.approx(env.max_range_cells)
    assert world["log_odds_clamp"] == pytest.approx(env.log_odds_clamp)
    assert world["entropy_threshold_bits"] == pytest.approx(env.entropy_threshold_bits)
    assert world["start"] == [env.start_row, env.start_col, env.start_heading]
    assert world["state_layout"]["state_size"] == env.state_size
    assert world["state_layout"]["map_offset"] == env.map_offset


def test_trace_keeps_the_robots_map_and_the_hidden_map_apart(env, episode):
    """Test that each map block is written from where it actually lives.

    Purpose: Both maps are ``num_cells`` floats in one state vector, so reading
        the wrong offset would show a viewer the answer -- a perfect map from
        step zero -- rather than what the robot had established

    Given: A recorded episode whose robot has mapped only part of the room
    When: A trace is built
    Then: ``log_odds`` is the state's inverse-sensor map at every step, and
        ``true_map`` is the hidden occupancy, which is not the same thing

    Test type: unit
    """
    payload = build_occupancy_grid_mapping_trace(env, episode, 0).payload

    for index, step in enumerate(episode):
        expected = env.log_odds(np.asarray(step.state, dtype=np.float64)).reshape(-1)
        assert payload["log_odds"][index] == pytest.approx(list(expected))
        assert payload["entropy_bits"][index] == pytest.approx(env.entropy_bits(step.state))

    expected_truth = env.true_map(np.asarray(episode[0].state, dtype=np.float64)).reshape(-1)
    assert payload["world"]["true_map"] == pytest.approx(list(expected_truth))
    # The first state is all-unknown, so a swapped pair of offsets would be
    # obvious here: the robot's map starts at zero everywhere and the hidden
    # one does not.
    assert set(payload["world"]["true_map"]) == {0.0, 1.0}
    assert payload["log_odds"][0] == pytest.approx([0.0] * env.num_cells)


def test_trace_marks_the_pre_scan_state_as_unscanned(env, episode):
    """Test that the initial state is not reported as carrying a reading.

    Purpose: The initial state holds zero range placeholders rather than a
        scan, and a viewer that drew them would put twenty-four beams of length
        zero under a robot that has not measured anything yet

    Given: An episode starting from the initial state
    When: A trace is built
    Then: Only the first step is marked unscanned, and every step carries one
        range per beam

    Test type: unit
    """
    payload = build_occupancy_grid_mapping_trace(env, episode, 0).payload

    assert payload["scanned"][0] is False
    assert all(payload["scanned"][1:])
    assert {len(ranges) for ranges in payload["ranges"]} == {env.num_beams}


def test_trace_delegates_belief_serialization_to_core(env, episode):
    """Test that the belief payload is core's, not this environment's.

    Purpose: Every environment must write a belief the same way, so a reader
        draws particles once rather than once per environment

    Given: An episode whose steps carry a real whole-map particle belief
    When: A trace is built
    Then: Each belief payload equals what ``belief_to_payload`` produces

    Test type: unit
    """
    payload = build_occupancy_grid_mapping_trace(env, episode, 0).payload

    for index, step in enumerate(episode):
        expected = belief_to_payload(step.belief, max_particles=MAX_TRACE_PARTICLES)
        assert payload["beliefs"][index] == expected
        assert payload["beliefs"][index]["kind"] == "particles"


def test_trace_caps_how_many_whole_maps_are_written(env, episode):
    """Test that the written cloud is bounded, and says how much it dropped.

    Purpose: One particle here is a whole world of several hundred numbers, so
        core's default cap would make a trace tens of megabytes

    Given: An episode whose belief carries more particles than the cap
    When: A trace is built
    Then: At most ``MAX_TRACE_PARTICLES`` are written, and the payload still
        reports the cloud's true size

    Test type: unit
    """
    step = episode[1]
    crowded = type(step.belief)(
        particles=np.repeat(np.asarray(step.belief.particles), 3, axis=0),
        log_weights=np.repeat(np.asarray(step.belief.log_weights), 3),
    )
    history = [step._replace(belief=crowded)]

    belief = build_occupancy_grid_mapping_trace(env, history, 0).payload["beliefs"][0]

    assert belief["num_particles"] == len(crowded.particles)
    assert belief["num_written"] == min(MAX_TRACE_PARTICLES, len(crowded.particles))
    assert len(belief["particles"]) == belief["num_written"]


def test_environment_writes_a_trace_file(env, episode, tmp_path: Path):
    """Test that the environment's own hook writes the trace.

    Purpose: The simulation layer calls ``cache_trace``, not the exporter, so a
        trace that only builds when called by hand would never be written by a
        real run

    Given: An environment and an episode
    When: ``cache_trace`` is called
    Then: A ``trace_<index>.json`` is written and reads back with this
        environment's payload kind

    Test type: integration
    """
    written = env.cache_trace(episode, tmp_path, episode_index=4, policy_name="PFT_DPW")

    assert written == tmp_path / "trace_4.json"
    assert EpisodeTrace.read(written).payload_kind == OCCUPANCY_GRID_MAPPING_PAYLOAD_KIND


def test_trace_rejects_an_empty_history(env):
    """Test that an empty episode is refused rather than written.

    Purpose: An empty trace would be a file a viewer loads and then draws
        nothing from, with no way to tell it apart from a broken one

    Given: No recorded steps
    When: A trace is requested
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match="empty history"):
        build_occupancy_grid_mapping_trace(env, [], 0)


def test_cache_visualization_renders_the_moved_renderer(env, episode, tmp_path: Path):
    """Test that the environment's GIF hook and the golden suite agree.

    Purpose: The renderer moved into a ``visualizer`` package, and the risk a
        move creates is that the environment's hook reaches a different module
        -- or no longer reaches one -- than the suite whose hash pins the bytes

    Given: The pinned deterministic episode
    When: The GIF is written by ``cache_visualization`` and, separately, by the
        visualizer class the golden suite uses
    Then: The two files are byte for byte identical

    Test type: integration
    """
    hook_dir = tmp_path / "hook"
    hook_dir.mkdir()
    env.cache_visualization(episode, hook_dir, 0)
    through_hook = hook_dir / "occupancy_grid_mapping_0.gif"

    direct = tmp_path / "direct.gif"
    OccupancyGridMappingVisualizer(env).create_visualization(episode, direct)

    assert through_hook.exists()
    assert hashlib.sha256(through_hook.read_bytes()).hexdigest() == (
        hashlib.sha256(direct.read_bytes()).hexdigest()
    )
