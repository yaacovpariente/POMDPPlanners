# SPDX-License-Identifier: MIT

"""Encode CARLA RGB camera frames as an MP4 video.

:class:`~POMDPPlanners.environments.carla_pomdp.carla_pomdp.CarlaPOMDP` can attach
a chase RGB camera to the ego vehicle and buffer one rendered frame per simulator
tick. This module turns that buffer of ``(H, W, 3)`` uint8 frames into an H.264
MP4 by piping the raw RGB bytes straight to an ``ffmpeg`` subprocess. Streaming
the pixels to ffmpeg avoids matplotlib's per-frame figure re-render, so the saved
footage is CARLA's own rendering and encoding it is roughly 2-3x faster than the
previous matplotlib writer.

Functions:
    write_frames_to_mp4: Encode a list of RGB frames as an MP4 video.
"""

import shutil
import subprocess
from pathlib import Path
from typing import List

import numpy as np


def write_frames_to_mp4(frames: List[np.ndarray], cache_path: Path, fps: int = 20) -> None:
    """Encode buffered CARLA chase-camera frames as an MP4 video.

    The frames are streamed as raw ``rgb24`` bytes to an ``ffmpeg`` subprocess,
    which encodes them to an H.264 MP4. This is CARLA's own rendering, not a
    reconstructed plot.

    Args:
        frames: Non-empty list of ``(H, W, 3)`` uint8 RGB frames, one per tick.
        cache_path: File path ending in ``.mp4`` where the video is saved.
        fps: Playback frame rate. Defaults to 20.

    Raises:
        TypeError: If ``cache_path`` is not a Path object.
        ValueError: If ``frames`` is empty or ``cache_path`` does not end in ``.mp4``.
        RuntimeError: If ``ffmpeg`` is not on PATH or the encode fails.
    """
    _validate_inputs(frames, cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    command = _build_ffmpeg_command(width, height, fps, cache_path)
    _stream_frames_to_ffmpeg(command, frames)


def _validate_inputs(frames: List[np.ndarray], cache_path: Path) -> None:
    if not isinstance(cache_path, Path):
        raise TypeError("cache_path must be a Path object")
    if not frames:
        raise ValueError("Cannot write an empty frame list to video")
    if not str(cache_path).endswith(".mp4"):
        raise ValueError("cache_path must end with .mp4")


def _build_ffmpeg_command(width: int, height: int, fps: int, cache_path: Path) -> List[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg executable not found on PATH; cannot encode video")
    return [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(cache_path),
    ]


def _stream_frames_to_ffmpeg(command: List[str], frames: List[np.ndarray]) -> None:
    with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        stdin = process.stdin
        if stdin is None:  # pragma: no cover - Popen with PIPE always provides stdin
            raise RuntimeError("Failed to open ffmpeg stdin pipe")
        try:
            for frame in frames:
                stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        finally:
            stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    if return_code != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to encode video (exit {return_code}): {message}")
