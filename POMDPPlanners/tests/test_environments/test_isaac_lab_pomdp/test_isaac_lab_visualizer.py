# SPDX-License-Identifier: MIT

"""Unit tests for the IsaacLabPOMDP episode video visualizer.

These tests exercise the RGB-frame-to-``.mp4`` writer end-to-end: the writer
pipes the frames straight into the system ``ffmpeg`` binary, so the success cases
encode a real (tiny) video and assert a non-empty ``.mp4`` is produced. They need
no Isaac Sim, since the visualizer only consumes a list of already-captured RGB
frames. They mirror the CarlaPOMDP visualizer test patterns (parameter validation
+ successful encode).
"""

from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.environments.isaac_lab_pomdp import IsaacLabPOMDPVisualizer


def _frame(value: int, channels: int = 3) -> np.ndarray:
    """Build a small constant ``(H, W, channels)`` uint8 RGB(A) frame."""
    return np.full((4, 6, channels), value, dtype=np.uint8)


def test_frames_to_video_rejects_non_list() -> None:
    """frames_to_video rejects a non-list frames argument.

    Purpose: Validates the frames container-type guard.

    Given: An IsaacLabPOMDPVisualizer
    When: frames_to_video is called with a numpy array instead of a list
    Then: TypeError is raised

    Test type: unit
    """
    visualizer = IsaacLabPOMDPVisualizer()
    with pytest.raises(TypeError, match="frames must be a List object"):
        visualizer.frames_to_video(_frame(0), Path("out.mp4"))  # type: ignore[arg-type]


def test_frames_to_video_empty_frames_raises() -> None:
    """frames_to_video rejects an empty frame list.

    Purpose: Validates the empty-frames guard.

    Given: An IsaacLabPOMDPVisualizer and an empty frame list
    When: frames_to_video is called
    Then: ValueError is raised

    Test type: unit
    """
    visualizer = IsaacLabPOMDPVisualizer()
    with pytest.raises(ValueError, match="Cannot visualize empty frames"):
        visualizer.frames_to_video([], Path("out.mp4"))


def test_frames_to_video_parameter_validation() -> None:
    """frames_to_video rejects a non-Path or non-.mp4 cache path.

    Purpose: Validates cache-path type and suffix checks.

    Given: An IsaacLabPOMDPVisualizer and a one-frame sequence
    When: frames_to_video is called with a string path, then a .gif path
    Then: TypeError then ValueError are raised

    Test type: unit
    """
    visualizer = IsaacLabPOMDPVisualizer()
    frames = [_frame(0)]

    with pytest.raises(TypeError, match="cache_path must be a Path object"):
        visualizer.frames_to_video(frames, "out.mp4")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cache_path must end with .mp4"):
        visualizer.frames_to_video(frames, Path("out.gif"))


def test_frames_to_video_rejects_malformed_frame() -> None:
    """frames_to_video rejects a frame that is not an (H, W, 3/4) array.

    Purpose: Validates the per-frame shape guard.

    Given: An IsaacLabPOMDPVisualizer and a 2-D (grayscale) frame
    When: frames_to_video is called
    Then: ValueError is raised

    Test type: unit
    """
    visualizer = IsaacLabPOMDPVisualizer()
    with pytest.raises(ValueError, match="each frame must be"):
        visualizer.frames_to_video([np.zeros((4, 6), dtype=np.uint8)], Path("out.mp4"))


def test_frames_to_video_success(tmp_path: Path) -> None:
    """A valid RGB frame sequence encodes a non-empty .mp4 video.

    Purpose: Validates the end-to-end ffmpeg encode path.

    Given: An IsaacLabPOMDPVisualizer and a three-frame RGB sequence
    When: frames_to_video is called with a .mp4 path
    Then: the .mp4 file is created and is non-empty

    Test type: unit
    """
    visualizer = IsaacLabPOMDPVisualizer()
    frames = [_frame(0), _frame(120), _frame(255)]
    video_path = tmp_path / "episode.mp4"

    visualizer.frames_to_video(frames, video_path)

    assert video_path.exists()
    assert video_path.stat().st_size > 0


def test_frames_to_video_drops_alpha_channel(tmp_path: Path) -> None:
    """An RGBA frame sequence is accepted (alpha dropped) and encodes a video.

    Purpose: Validates that 4-channel RGBA frames are normalized to RGB.

    Given: An IsaacLabPOMDPVisualizer and a two-frame RGBA sequence
    When: frames_to_video is called with a .mp4 path
    Then: the .mp4 file is created and is non-empty

    Test type: unit
    """
    visualizer = IsaacLabPOMDPVisualizer()
    frames = [_frame(0, channels=4), _frame(200, channels=4)]
    video_path = tmp_path / "episode.mp4"

    visualizer.frames_to_video(frames, video_path)

    assert video_path.exists()
    assert video_path.stat().st_size > 0
