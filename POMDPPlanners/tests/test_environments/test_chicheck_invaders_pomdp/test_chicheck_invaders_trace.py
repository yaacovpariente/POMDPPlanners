# SPDX-License-Identifier: MIT

"""Tests for the Chicheck Invaders episode trace.

Two things are checked here, and they are the two things the migration could
break.

* The payload round-trips and means what it says: a viewer decoding a state
  vector with the layout the trace carries gets the flock the episode had, and
  the shot record agrees with the environment's own ``fires`` and
  ``shot_target`` rather than with a second rule written in the exporter.
* The GIF is untouched. Its bytes are pinned by a golden hash, so moving the
  renderer into a package and adding an exporter beside it must leave the
  rendered episode identical, including when the trace is written from the same
  history first.
"""

from pathlib import Path

import pytest

from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.chicheck_invaders_pomdp import (
    CHICHECK_INVADERS_PAYLOAD_KIND,
    ChicheckInvadersVisualizer,
    build_chicheck_invaders_trace,
)
from POMDPPlanners.tests.test_environments.test_environment_visualizations_golden_files import (
    build_chicheck_invaders_env,
    create_deterministic_chicheck_invaders_episode,
)


@pytest.fixture(name="episode")
def episode_fixture():
    """A real, deterministic Chicheck Invaders episode.

    The same fixture the golden GIF is rendered from, so a trace test and a
    picture test are talking about one episode rather than two.
    """
    return create_deterministic_chicheck_invaders_episode(seed=11)


@pytest.fixture(name="env")
def env_fixture():
    """The environment that episode ran on."""
    return build_chicheck_invaders_env()


def _decode_chicken(vector, layout, slot):
    """Decode one chicken slot out of a raw state vector, the way a viewer does.

    Written here rather than imported from the schema on purpose: the point of
    the layout block is that a reader who has only the trace can decode it, and
    a test that used the Python constants would pass even if the block were
    wrong.
    """
    base = layout["ship_width"] + slot * layout["chicken_width"]
    return {
        "column": vector[base + layout["chicken_column"]],
        "row": vector[base + layout["chicken_row"]],
        "direction": vector[base + layout["chicken_direction"]],
        "diving": vector[base + layout["chicken_mode"]] == layout["mode_dive"],
        "alive": vector[base + layout["chicken_alive"]] > 0,
    }


def test_trace_round_trips_through_json(env, episode, tmp_path: Path):
    """The written file reads back as the same trace.

    Purpose: The viewer reads JSON, not Python objects. A payload that only
    survives in memory is a payload nobody can draw.

    Given: A Chicheck Invaders episode.
    When: Its trace is written and read back.
    Then: The envelope and the whole payload compare equal.
    """
    trace = build_chicheck_invaders_trace(env, episode, episode_index=2, policy_name="PFT_DPW")
    path = trace.write(tmp_path / "trace_2.json")

    reloaded = EpisodeTrace.read(path)

    assert reloaded.payload_kind == CHICHECK_INVADERS_PAYLOAD_KIND
    assert reloaded.episode_index == 2
    assert reloaded.policy == "PFT_DPW"
    assert reloaded.discount_factor == pytest.approx(env.discount_factor)
    assert len(reloaded.steps) == len(episode)
    assert reloaded.payload == trace.to_dict()["payload"]


def test_trace_layout_decodes_the_recorded_states(env, episode):
    """The layout block is enough to read a state vector back.

    Purpose: States are written as the raw vectors the run held, so the layout
    the trace carries is the only thing standing between a viewer and a flock
    drawn in the wrong cells. A belief particle is a state vector too, so this
    decoder is the one the belief uses as well.

    Given: A Chicheck Invaders episode.
    When: Each recorded state is decoded with the payload's own layout.
    Then: The ship's column and every chicken slot match the episode's states.
    """
    trace = build_chicheck_invaders_trace(env, episode, episode_index=0)
    layout = trace.payload["layout"]["state"]

    assert len(trace.payload["states"]) == len(episode)
    for step, vector in zip(episode, trace.payload["states"]):
        assert vector[layout["ship_column"]] == pytest.approx(env.ship_column(step.state))
        flock = env.chickens(step.state)
        for slot in range(env.num_chickens):
            decoded = _decode_chicken(vector, layout, slot)
            assert decoded["column"] == pytest.approx(flock[slot, 0])
            assert decoded["row"] == pytest.approx(flock[slot, 1])
            assert decoded["alive"] == bool(flock[slot, 4] > 0)


def test_trace_shots_come_from_the_environment(env, episode):
    """What the gun did is asked of the environment, not re-derived.

    Purpose: The beam is the whole hitscan event — there is no projectile to
    check it against — so a shot record that disagreed with the kill the episode
    scored would draw a lie nothing else would catch.

    Given: A Chicheck Invaders episode including steps that fire.
    When: The trace is built.
    Then: Each step's shot record equals the environment's own ``fires`` and
        ``shot_target`` for that step, and at least one step fired. The terminal
        bookkeeping step carries no action and therefore never fires.
    """
    trace = build_chicheck_invaders_trace(env, episode, episode_index=0)
    shots = trace.payload["shots"]

    assert len(shots) == len(episode)
    assert any(shot["fired"] for shot in shots), "fixture must exercise the gun"
    for step, shot in zip(episode, shots):
        if step.action is None:
            assert shot["fired"] is False
            continue
        assert shot["fired"] == env.fires(step.state, step.action)
        if not shot["fired"]:
            assert shot["target_slot"] == -1
            continue
        assert shot["target_slot"] == env.shot_target(step.state)


def test_trace_delegates_belief_serialization_to_core(env, episode):
    """The exporter writes no belief format of its own.

    Purpose: Belief is a core abstraction with a closed family of
    implementations. Serializing it per environment would mean a copy of the
    same dispatch in every environment and a chance to miss a class in each.

    Given: An episode whose steps carry real particle beliefs.
    When: The trace is built.
    Then: Each step's belief payload equals ``belief_to_payload`` of that step's
        belief, field for field.
    """
    trace = build_chicheck_invaders_trace(env, episode, episode_index=0)

    for step, written in zip(episode, trace.payload["beliefs"]):
        assert written == belief_to_payload(step.belief)


def test_trace_takes_the_sensor_parameters_from_the_instance(episode):
    """The cone and the ring drawn are the ones this run was configured with.

    Purpose: The viewer builds both footprints straight from these two numbers.
    Reading them off the class defaults would draw a sensor nobody had the
    moment a run configured a wider cone.

    Given: An environment with a non-default camera slope and radar radius.
    When: The trace is built.
    Then: The payload's world block carries that run's values.
    """
    env = build_chicheck_invaders_env()
    env.camera_slope = 0.5
    env.radar_radius = 3.25

    world = build_chicheck_invaders_trace(env, episode, episode_index=0).payload["world"]

    assert world["camera_slope"] == pytest.approx(0.5)
    assert world["radar_radius"] == pytest.approx(3.25)


def test_trace_rejects_an_empty_history(env):
    """An empty history is an error, not an empty trace.

    Purpose: A zero-step trace would render as a blank viewer that looks like a
    working page, which is worse than a failure.

    Given: No steps.
    When: A trace is asked for.
    Then: ValueError.
    """
    with pytest.raises(ValueError):
        build_chicheck_invaders_trace(env, [], 0)


def test_environment_writes_a_trace_file(env, episode, tmp_path: Path):
    """The environment writes the trace through the core hook.

    Purpose: The reporting site finds traces by a fixed file name across
    environments it has never heard of, so the environment must go through
    ``cache_trace`` rather than name its own file.

    Given: A Chicheck Invaders episode.
    When: cache_trace is called.
    Then: ``trace_3.json`` appears and reads back with this payload kind.
    """
    written = env.cache_trace(history=episode, output_dir=tmp_path, episode_index=3)

    assert written == tmp_path / "trace_3.json"
    assert EpisodeTrace.read(written).payload_kind == CHICHECK_INVADERS_PAYLOAD_KIND


def test_writing_a_trace_does_not_change_the_gif(env, episode, tmp_path: Path):
    """The GIF is byte-identical whether or not a trace was written first.

    Purpose: The GIF's bytes are pinned by a golden hash. This migration moved
    the renderer and put an exporter beside it, and both read the same history
    and the same belief objects; a trace that consumed or mutated either would
    move the golden hash, which the migration may not do.

    Given: One deterministic episode.
    When: The GIF is rendered alone, and again after the trace is exported from
        the same history.
    Then: The two files are byte-identical.
    """
    before = tmp_path / "before.gif"
    ChicheckInvadersVisualizer(env).create_visualization(episode, before)

    env.cache_trace(history=episode, output_dir=tmp_path, episode_index=0)

    after = tmp_path / "after.gif"
    ChicheckInvadersVisualizer(env).create_visualization(episode, after)

    assert after.read_bytes() == before.read_bytes()
