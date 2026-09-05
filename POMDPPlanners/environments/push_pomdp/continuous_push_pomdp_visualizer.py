# SPDX-License-Identifier: MIT

"""Cached Pillow visualization for the continuous Push POMDP."""

from typing import TYPE_CHECKING, Any, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from POMDPPlanners.environments.push_pomdp.push_visualization_assets import paste_obstacle

from POMDPPlanners.environments.push_pomdp.push_visualization_utils import (
    ROBOT_RADIUS,
    PushRendererBase,
)

if TYPE_CHECKING:
    from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import (
        ContinuousPushPOMDP,
    )


class ContinuousPushPOMDPVisualizer(PushRendererBase):
    """Render continuous Push episodes with one cached static scene."""

    title = "Continuous Push POMDP Episode Visualization"

    def __init__(self, env: "ContinuousPushPOMDP"):
        super().__init__(env)
        self.robot_radius = env.robot_radius

    def _variant_scene_key(self) -> Tuple[Any, ...]:
        return (float(self.robot_radius),)

    def _draw_static_obstacles(self, canvas: Image.Image) -> None:
        for cx, cy, hx, hy in np.asarray(self.obstacles, dtype=float):
            left, top = self._to_px((cx - hx, cy + hy))
            right, bottom = self._to_px((cx + hx, cy - hy))
            paste_obstacle(canvas, (left, top, right, bottom), circle=False)

    def _legend_entries(self) -> List[Tuple[str, str]]:
        entries = super()._legend_entries()
        entries.insert(3, ("Robot Radius", "radius"))
        return entries

    def _draw_variant_robot(self, draw: ImageDraw.ImageDraw, robot_pos: np.ndarray) -> None:
        x, y = self._to_px(robot_pos)
        radius = float(self.robot_radius) * self._px_per_unit
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=ROBOT_RADIUS,
            outline=(120, 120, 220),
            width=2,
        )

    def _action_vector(self, action: Any) -> np.ndarray:
        if isinstance(action, str):
            return {
                "up": np.array([0.0, 1.0]),
                "down": np.array([0.0, -1.0]),
                "right": np.array([1.0, 0.0]),
                "left": np.array([-1.0, 0.0]),
            }.get(action, np.array([0.0, 0.0]))
        return np.asarray(action, dtype=float)

    def _format_action(self, action: Any) -> str:
        if isinstance(action, str):
            return action
        vector = np.asarray(action, dtype=float)
        return f"({vector[0]:.2f}, {vector[1]:.2f})"

    def _distance_arrow(self) -> str:
        return "->"

    def _collisions(self, robot_pos: np.ndarray, object_pos: np.ndarray) -> Tuple[bool, bool]:
        return (
            self.env._is_circle_colliding_with_obstacle(  # pylint: disable=protected-access
                robot_pos, self.robot_radius
            ),
            self.env._is_point_colliding_with_obstacle(  # pylint: disable=protected-access
                object_pos
            ),
        )
