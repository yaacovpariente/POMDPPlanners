# SPDX-License-Identifier: MIT
"""Continuous LaserTag GIFs with a cached Pillow background per episode."""

import numpy as np

from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_geometry import (
    LASER_DIRECTIONS,
    compute_laser_measurements,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_renderer import (
    CANVAS_SIZE,
    LASER_COLOR,
    OPPONENT_BELIEF_COLOR,
    ROBOT_BELIEF_COLOR,
    LaserTagFrameRenderer,
)


class ContinuousLaserTagVisualizer(LaserTagFrameRenderer):
    """Draw recorded continuous states with the existing coordinates and GIF timing."""

    _STRING_ACTION_DIRS = {
        "up": np.array([0.0, 1.0]),
        "down": np.array([0.0, -1.0]),
        "right": np.array([1.0, 0.0]),
        "left": np.array([-1.0, 0.0]),
        "tag": np.array([0.0, 0.0]),
    }

    def _action_info(self, action):
        if isinstance(action, str):
            return (
                f"Action: {action}",
                action == "tag",
                self._STRING_ACTION_DIRS.get(action, np.zeros(2)),
            )
        values = np.asarray(action, dtype=float).ravel()
        text = (
            f"Action: [{values[0]:.1f}, {values[1]:.1f}, {values[2]:.1f}]"
            if values.size > 2
            else ""
        )
        return (
            text,
            values.size > 2 and values[2] > 0.5,
            values[:2] if values.size >= 2 else np.zeros(2),
        )

    def _laser_segments(self, rp, op):
        measurements = compute_laser_measurements(
            rp, op, self.opponent_radius, self._wall_rectangles, self.grid_size
        )
        return [
            (rp, rp + direction * distance)
            for direction, distance in zip(LASER_DIRECTIONS, measurements)
        ]

    def _is_successful_tag(self, rp, op) -> bool:
        return bool(np.linalg.norm(rp - op) <= self.robot_radius + self.opponent_radius + 0.5)
