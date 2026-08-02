# SPDX-License-Identifier: MIT

"""Unit tests for the CARLA chase-camera MP4 encoder.

These tests exercise :func:`write_frames_to_mp4`, which streams buffered RGB
frames to an ``ffmpeg`` subprocess. They require ``ffmpeg`` on PATH (skipped
otherwise) but no CARLA server or the ``carla`` package.
"""

import shutil
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import pytest

from POMDPPlanners.environments.carla_pomdp.carla_video import write_frames_to_mp4

_FFMPEG_MISSING = shutil.which("ffmpeg") is None
_SKIP_REASON = "ffmpeg not available on PATH"


def _make_frames(count: int, height: int = 48, width: int = 64) -> List[np.ndarray]:
    """Build a list of distinct ``(H, W, 3)`` uint8 RGB frames."""
    return [
        np.full((height, width, 3), fill_value=(index * 7) % 256, dtype=np.uint8)
        for index in range(count)
    ]


@pytest.mark.skipif(_FFMPEG_MISSING, reason=_SKIP_REASON)
def test_write_frames_produces_playable_mp4(tmp_path: Path) -> None:
    """Encoding a frame buffer writes a non-empty, ffprobe-readable MP4.

    Purpose: Validates the raw-pipe ffmpeg encoder produces a real H.264 MP4.

    Given: A buffer of ten 64x48 RGB frames and a target .mp4 path.
    When: write_frames_to_mp4 streams the frames to ffmpeg.
    Then: The file exists, is non-empty, and ffprobe reports an h264 video stream.

    Test type: integration
    """
    output = tmp_path / "clip.mp4"
    write_frames_to_mp4(_make_frames(10), output, fps=20)

    assert output.exists()
    assert output.stat().st_size > 0
    probe = subprocess.run(
        [
            "ffprobe",
            "-loglevel",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=nk=1:nw=1",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "h264" in probe.stdout


@pytest.mark.skipif(_FFMPEG_MISSING, reason=_SKIP_REASON)
def test_write_frames_creates_missing_parent_directory(tmp_path: Path) -> None:
    """The encoder creates missing parent directories for the output path.

    Purpose: Validates that a nested output path is created on demand.

    Given: A .mp4 path inside a not-yet-existing subdirectory.
    When: write_frames_to_mp4 is called with a single-frame buffer.
    Then: The parent directory is created and the MP4 is written.

    Test type: unit
    """
    output = tmp_path / "nested" / "deeper" / "clip.mp4"
    write_frames_to_mp4(_make_frames(1), output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_write_frames_rejects_non_path_cache(tmp_path: Path) -> None:
    """A string cache path raises TypeError before any encoding.

    Purpose: Validates the Path type guard on the output argument.

    Given: A cache path passed as a plain string.
    When: write_frames_to_mp4 is called.
    Then: TypeError is raised.

    Test type: unit
    """
    with pytest.raises(TypeError):
        write_frames_to_mp4(_make_frames(1), str(tmp_path / "clip.mp4"))  # type: ignore[arg-type]


def test_write_frames_rejects_empty_frame_list(tmp_path: Path) -> None:
    """An empty frame list raises ValueError.

    Purpose: Validates the non-empty-frames precondition.

    Given: An empty list of frames.
    When: write_frames_to_mp4 is called.
    Then: ValueError is raised.

    Test type: unit
    """
    with pytest.raises(ValueError, match="empty frame list"):
        write_frames_to_mp4([], tmp_path / "clip.mp4")


def test_write_frames_rejects_non_mp4_extension(tmp_path: Path) -> None:
    """A cache path without a .mp4 suffix raises ValueError.

    Purpose: Validates the .mp4 extension precondition.

    Given: A cache path ending in .avi.
    When: write_frames_to_mp4 is called.
    Then: ValueError is raised.

    Test type: unit
    """
    with pytest.raises(ValueError, match=".mp4"):
        write_frames_to_mp4(_make_frames(1), tmp_path / "clip.avi")
