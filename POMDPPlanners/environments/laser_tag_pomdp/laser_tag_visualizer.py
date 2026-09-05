# SPDX-License-Identifier: MIT
"""Cached Pillow renderer for the discrete LaserTag environment."""

from pathlib import Path
from typing import List, Set, Tuple

import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_renderer import (
    LaserTagFrameRenderer,
)


class LaserTagVisualizer(LaserTagFrameRenderer):
    """Draw discrete LaserTag episodes without episode-time Matplotlib work."""

    _ACTION_DIRS = {
        0: np.array([-1.0, 0.0]),
        1: np.array([1.0, 0.0]),
        2: np.array([0.0, 1.0]),
        3: np.array([0.0, -1.0]),
        4: np.array([0.0, 0.0]),
    }
    _ACTION_NAMES = {0: "North", 1: "South", 2: "East", 3: "West", 4: "Tag"}

    def __init__(
        self,
        floor_shape: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        dangerous_areas: List[Tuple[int, int]],
        dangerous_area_radius: float,
    ):
        self.floor_shape = floor_shape
        self._wall_cells = set(walls)
        wall_rectangles = np.asarray(
            [(row, col, 0.4, 0.4) for row, col in sorted(walls)], dtype=float
        ).reshape(-1, 4)
        super().__init__(
            grid_size=np.asarray(floor_shape, dtype=float) - 1,
            walls=wall_rectangles,
            robot_radius=0.0,
            opponent_radius=0.0,
            dangerous_areas=[(float(row), float(col)) for row, col in dangerous_areas],
            dangerous_area_radius=dangerous_area_radius,
        )
        self._title = "LaserTag POMDP Episode Visualization"
        self.walls = walls
        self._x_label = "Row"
        self._y_label = "Column"

    def _world_to_pixel(self, point) -> Tuple[float, float]:
        return (
            self._left + (float(point[0]) + 0.5) * self._scale,
            self._top + (float(point[1]) + 0.5) * self._scale,
        )

    def create_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Preserve the public discrete visualization contract."""
        super().create_visualization(history, cache_path)

    def _extract_history(self, history: List[StepData]) -> Tuple:
        robot_path, opponent_path, actions, beliefs = super()._extract_history(history)
        return (
            [np.asarray(point, dtype=int) for point in robot_path],
            [np.asarray(point, dtype=int) for point in opponent_path],
            actions,
            beliefs,
        )

    def _action_info(self, action):
        if action not in self._ACTION_DIRS:
            return "", False, np.zeros(2)
        return f"Action: {self._ACTION_NAMES[action]}", action == 4, self._ACTION_DIRS[action]

    def _laser_segments(self, rp, op):
        del op
        segments = []
        for direction in (
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
        ):
            dr, dc = direction
            distance = 0
            while True:
                row = int(rp[0]) + dr * (distance + 1)
                col = int(rp[1]) + dc * (distance + 1)
                if (
                    row < 0
                    or row >= self.floor_shape[0]
                    or col < 0
                    or col >= self.floor_shape[1]
                    or (row, col) in self._wall_cells
                ):
                    break
                distance += 1
            end = np.asarray(rp, dtype=float) + np.asarray(direction, dtype=float) * distance
            segments.append((rp, end))
        return segments

    def _is_successful_tag(self, rp, op) -> bool:
        return bool(np.array_equal(rp, op))

    def _belief_radius(self, distribution, index) -> float:
        """Keep marker area proportional to probability, as in the original scatter."""
        probability = float(distribution.probs[index])
        # The original scatter used area p * 100 points squared at 100 dpi.
        return float(np.sqrt(max(0.0, probability) * 100) * 100 / 72 / 2)
