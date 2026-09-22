# SPDX-License-Identifier: MIT

"""Tests for artifact-kind classification and player dispatch.

The site chooses a player from an artifact's kind and never from the
environment that produced it. That is what makes a video-only environment
(CARLA, Isaac Lab, nuPlan) a first-class episode without a code change, so it
is tested directly rather than only through a page.
"""

import json
from pathlib import Path

import pytest

from POMDPPlanners.core.simulation.traces import ArtifactKind, EpisodeTrace
from POMDPPlanners.reporting.artifacts import (
    PLAYER_FOR_KIND,
    classify,
    episode_index_from_name,
    group_by_episode,
    media_type_for,
    preferred,
)

FIXTURE_VIDEO = Path(__file__).parent / "fixtures" / "agent_path_0.mp4"


def _classified(path: Path, relative_path: str):
    """Classify a file that the test knows is an artifact."""
    artifact = classify(path, relative_path)
    assert artifact is not None, f"{relative_path} was not classified as an artifact"
    return artifact


@pytest.mark.parametrize(
    "name, expected",
    [
        ("agent_path_3.gif", 3),
        ("trace_12.json", 12),
        ("battleship_board_0.gif", 0),
        ("agent_path_0.mp4", 0),
        ("discounted_returns_histogram.png", None),
        ("policy_comparison_histogram.png", None),
    ],
)
def test_episode_index_is_read_from_the_trailing_number(name, expected):
    """The index comes from the trailing number, not a fixed prefix.

    Purpose: Environments name their per-episode files differently, so a fixed
    prefix would silently miss half of them.

    Given: File names from several environments and two per-policy plots.
    When: The index is read.
    Then: Per-episode files yield their index and plots yield None.
    """
    assert episode_index_from_name(name) == expected


@pytest.mark.parametrize(
    "filename, expected_kind, expected_player",
    [
        ("agent_path_0.mp4", ArtifactKind.VIDEO, "video"),
        ("episode.webm", ArtifactKind.VIDEO, "video"),
        ("agent_path_0.gif", ArtifactKind.GIF, "image"),
        ("discounted_returns_histogram.png", ArtifactKind.PLOT, "image"),
    ],
)
def test_classification_picks_the_player_from_the_kind(
    tmp_path: Path, filename, expected_kind, expected_player
):
    """Every kind maps to exactly one player, chosen without knowing the source.

    Purpose: This is the abstraction the whole site rests on. CARLA and Isaac
    Lab write an MP4 and no trace; they must get a <video> player from the
    extension alone.

    Given: Files of each artifact kind.
    When: They are classified.
    Then: The kind and its player are as declared, with no environment lookup.
    """
    path = tmp_path / filename
    path.write_bytes(b"not really media, but the kind is the extension")

    artifact = classify(path, filename)
    assert artifact is not None
    assert artifact.kind is expected_kind
    assert artifact.player == expected_player
    assert PLAYER_FOR_KIND[expected_kind] == expected_player


def test_a_real_video_file_is_classified_as_a_video():
    """The shipped fixture MP4 classifies as a video and serves as video/mp4.

    Purpose: A hand-named empty file would prove only that the extension map
    works. This is a real, playable H.264 file, which is what the video path
    will actually be handed.

    Given: The fixture MP4.
    When: It is classified and its media type is resolved.
    Then: It is a video, played with <video>, served as video/mp4.
    """
    assert FIXTURE_VIDEO.is_file() and FIXTURE_VIDEO.stat().st_size > 0

    artifact = classify(FIXTURE_VIDEO, FIXTURE_VIDEO.name)
    assert artifact is not None
    assert artifact.kind is ArtifactKind.VIDEO
    assert artifact.player == "video"
    assert artifact.episode_index == 0
    assert media_type_for(FIXTURE_VIDEO) == "video/mp4"


def test_a_trace_json_is_recognised_by_its_envelope(tmp_path: Path):
    """A JSON file is a trace only when it carries the envelope's fields.

    Purpose: A run directory also holds MLflow's own JSON tables. Treating
    those as traces would hand the viewer a file it cannot draw.

    Given: A real trace and an MLflow-shaped table, both .json.
    When: They are classified.
    Then: Only the trace is an artifact, and it reports its payload kind.
    """
    trace_path = EpisodeTrace(
        environment="Env",
        payload_kind="light_dark.v1",
        episode_index=4,
        discount_factor=0.95,
    ).write(tmp_path / "trace_4.json")
    table_path = tmp_path / "comparison_results.json"
    table_path.write_text(json.dumps({"columns": ["a"], "data": [[1]]}), encoding="utf-8")

    trace_artifact = classify(trace_path, trace_path.name)
    assert trace_artifact is not None
    assert trace_artifact.kind is ArtifactKind.TRACE
    assert trace_artifact.player == "trace-viewer"
    assert trace_artifact.payload_kind == "light_dark.v1"
    assert trace_artifact.episode_index == 4

    assert classify(table_path, table_path.name) is None


def test_unknown_extensions_are_not_artifacts(tmp_path: Path):
    """A file of no recognised kind is ignored rather than guessed at."""
    path = tmp_path / "profiling_results.txt"
    path.write_text("timings", encoding="utf-8")
    assert classify(path, path.name) is None


def test_preferred_ranks_trace_over_video_over_gif(tmp_path: Path):
    """An episode with several artifacts plays the most useful one.

    Purpose: Light-Dark writes both a trace and a GIF. The interactive viewer
    is the better page, and a video beats a GIF because it seeks.

    Given: An episode with a trace, a video and a GIF.
    When: The preferred artifact is chosen, then again with the trace removed.
    Then: Trace wins, then video, then GIF.
    """
    trace = _classified(
        EpisodeTrace(
            environment="Env", payload_kind="light_dark.v1", episode_index=0, discount_factor=1.0
        ).write(tmp_path / "trace_0.json"),
        "trace_0.json",
    )
    video = _classified(FIXTURE_VIDEO, "agent_path_0.mp4")
    gif_path = tmp_path / "agent_path_0.gif"
    gif_path.write_bytes(b"GIF89a")
    gif = _classified(gif_path, "agent_path_0.gif")

    assert preferred([gif, video, trace]) is trace
    assert preferred([gif, video]) is video
    assert preferred([gif]) is gif
    assert preferred([]) is None


def test_group_by_episode_leaves_out_per_policy_plots(tmp_path: Path):
    """Plots with no episode index belong to the policy page, not an episode."""
    gif = tmp_path / "agent_path_1.gif"
    gif.write_bytes(b"GIF89a")
    plot = tmp_path / "discounted_returns_histogram.png"
    plot.write_bytes(b"\x89PNG")

    grouped = group_by_episode([_classified(gif, gif.name), _classified(plot, plot.name)])
    assert list(grouped) == [1]
    assert len(grouped[1]) == 1
