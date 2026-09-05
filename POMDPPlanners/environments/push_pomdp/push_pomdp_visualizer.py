# SPDX-License-Identifier: MIT

"""Cached Pillow visualization for the discrete Push POMDP."""

from typing import TYPE_CHECKING, Any, Tuple

import numpy as np
from PIL import ImageDraw

from POMDPPlanners.environments.push_pomdp.push_visualization_utils import (
    OBSTACLE,
    OBSTACLE_EDGE,
    PushRendererBase,
)

if TYPE_CHECKING:
    from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP


class PushPOMDPVisualizer(PushRendererBase):
    """Render discrete Push episodes with one cached static scene."""

    _ACTION_DIRS = {
        "up": np.array([0.0, 1.0]),
        "down": np.array([0.0, -1.0]),
        "right": np.array([1.0, 0.0]),
        "left": np.array([-1.0, 0.0]),
    }

    def __init__(self, env: "PushPOMDP"):
        super().__init__(env)
        self.obstacle_radius = env.obstacle_radius

    def _variant_scene_key(self) -> Tuple[Any, ...]:
        return (float(self.obstacle_radius),)

    def _draw_static_obstacles(self, draw: ImageDraw.ImageDraw) -> None:
        radius = float(self.obstacle_radius) * self._px_per_unit
        for center in self.obstacles:
            x, y = self._to_px(center)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=OBSTACLE,
                outline=OBSTACLE_EDGE,
                width=3,
            )

    def _action_vector(self, action: Any) -> np.ndarray:
        return self._ACTION_DIRS.get(action, np.array([0.0, 0.0]))

    def _collisions(self, robot_pos: np.ndarray, object_pos: np.ndarray) -> Tuple[bool, bool]:
        return (
            self.env._is_colliding_with_obstacle(robot_pos),  # pylint: disable=protected-access
            self.env._is_colliding_with_obstacle(object_pos),  # pylint: disable=protected-access
        )
