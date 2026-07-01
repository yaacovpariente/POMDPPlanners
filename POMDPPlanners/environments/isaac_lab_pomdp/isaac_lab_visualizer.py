# SPDX-License-Identifier: MIT

"""IsaacLab POMDP episode video visualizer.

Renders an episode as an ``.mp4`` video of the simulator viewport: each frame is
the RGB image the IsaacLab simulator produced for that step (captured via
:meth:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_pomdp.IsaacLabPOMDP.render`),
so the video shows *what is seen in the simulator* rather than an abstract plot
of the state or observation.

Video (rather than GIF) output is used because IsaacLab frames are full-colour,
high-resolution renders where an animated GIF would be large and heavily
quantized; ``.mp4`` keeps the file small and the colours faithful. Encoding pipes
the raw RGB frames straight into the system ``ffmpeg`` binary (located via
matplotlib's configured ``animation.ffmpeg_path``), which adds no extra Python
dependency and avoids the per-frame matplotlib figure round-trip that dominates a
``FuncAnimation``-based encode.

Classes:
    IsaacLabPOMDPVisualizer: RGB-frame-to-``.mp4`` video writer for IsaacLabPOMDP.
"""

import subprocess
from pathlib import Path
from typing import Any, List

import numpy as np
from matplotlib.animation import FFMpegWriter


class IsaacLabPOMDPVisualizer:
    """RGB-frame-to-``.mp4`` video writer for an IsaacLabPOMDP episode.

    The visualizer needs only the list of RGB frames captured while an episode was
    rolled forward; it stores the environment for parity with the other
    environment visualizers but does not require any geometry from it.

    Example:
        Rendering is driven from the RGB frames captured during an episode::

            visualizer = IsaacLabPOMDPVisualizer(world)
            visualizer.frames_to_video(frames, Path("isaac_episode.mp4"))
    """

    def __init__(self, environment: Any = None) -> None:
        """Initialize the visualizer.

        Args:
            environment: The IsaacLabPOMDP instance the frames were produced by.
                Optional; retained for parity with other visualizers.
        """
        self.environment = environment

    def frames_to_video(self, frames: List[np.ndarray], cache_path: Path, fps: int = 10) -> None:
        """Write a sequence of RGB frames to an ``.mp4`` video.

        Args:
            frames: List of ``(H, W, 3)`` (or ``(H, W, 4)``) ``uint8`` RGB(A) frames
                captured from the simulator, one per step.
            cache_path: File path ending in ``.mp4`` where the video is saved.
            fps: Playback frame rate of the encoded video. Defaults to 10.

        Raises:
            TypeError: If ``frames`` is not a list or ``cache_path`` is not a Path.
            ValueError: If ``frames`` is empty or ``cache_path`` does not end with
                ``.mp4``.
        """
        if not isinstance(frames, List):
            raise TypeError("frames must be a List object")
        if not frames:
            raise ValueError("Cannot visualize empty frames")
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".mp4"):
            raise ValueError("cache_path must end with .mp4")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_video(self._normalize_frames(frames), cache_path, fps)

    @staticmethod
    def _normalize_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
        """Coerce frames to ``uint8`` RGB and drop any alpha channel."""
        normalized = []
        for frame in frames:
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] not in (3, 4):
                raise ValueError(
                    "each frame must be an (H, W, 3) or (H, W, 4) RGB(A) array; "
                    f"got shape {array.shape}"
                )
            normalized.append(array[:, :, :3].astype(np.uint8))
        return normalized

    def _write_video(self, frames: List[np.ndarray], cache_path: Path, fps: int) -> None:
        """Encode normalized RGB frames to ``cache_path`` by piping to ``ffmpeg``."""
        height, width = frames[0].shape[:2]
        command = [
            FFMpegWriter.bin_path(),
            "-y",
            "-loglevel",
            "error",
            "-nostats",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            # libx264/yuv420p require even dimensions; round down to the nearest even.
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(cache_path),
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            if process.stdin is None:  # pragma: no cover - PIPE always sets stdin
                raise RuntimeError("failed to open ffmpeg stdin pipe")
            for frame in frames:
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
            process.stdin.close()
            # stderr is kept tiny by '-loglevel error -nostats', so reading it in
            # full before waiting cannot deadlock on a full pipe buffer.
            stderr = b"" if process.stderr is None else process.stderr.read()
            process.wait()
        if process.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to encode video (exit {process.returncode}): "
                f"{stderr.decode(errors='replace').strip()}"
            )
