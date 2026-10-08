# SPDX-License-Identifier: MIT

"""How a CARLA episode is shown.

CARLA renders itself, so the episode's visualization is its own chase-camera
footage, buffered live while the world was stepped. The recorded step data is
not used: the footage cannot be rebuilt from it.

Classes:
    CarlaVisualizer: Writes the episode's chase-camera footage as an ``.mp4``.
"""

from pathlib import Path
from typing import Any

from POMDPPlanners.core.simulation.episode_visualizers import VideoVisualizer


class CarlaVisualizer(VideoVisualizer):
    """Writes a CARLA episode's chase-camera footage.

    The environment must have been constructed with ``record_camera=True``.
    """

    def write_video(self, path: Path) -> bool:
        """Write the buffered chase-camera frames to ``path``.

        Args:
            path: Destination ``.mp4``.

        Returns:
            ``True``; the video is always written when this returns.

        Raises:
            RuntimeError: If camera recording is disabled or no frames were
                captured, as raised by :meth:`CarlaPOMDP.save_camera_video`.
        """
        # The environment is typed as the base class; only CarlaPOMDP returns
        # this visualizer.
        environment: Any = self.environment
        environment.save_camera_video(path)
        return True
