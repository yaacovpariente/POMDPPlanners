# SPDX-License-Identifier: MIT

"""The goal seen from the robot: a bearing-and-range reading of a commanded pose.

A navigation task issues its goal as a *base-frame* offset — IsaacLab's ``UniformPose2dCommand``
keeps the target in world coordinates and hands the policy ``(dx, dy, dz, dheading)`` rotated into
the robot's frame. That base-frame vector is what a real robot has: a beacon bearing, a map match,
a relative waypoint from a global planner. The world target itself is privileged.

That split is what makes this channel carry localisation. Given the goal's world pose in the state,
the base-frame reading pins down ``(x, y, yaw)``; given only the reading, it says nothing about
where the robot is. So a belief that also has to infer the goal, or that starts dispersed in pose,
gets a real filtering problem rather than a restatement of its own state — and the same channel is
the one the robot genuinely observes.

Classes:
    GoalRelativePoseObservationModel: Goal offset and heading error in the base frame, with noise.
"""

from typing import Any, Mapping, Optional, Tuple

import numpy as np

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_model import (
    IsaacObservationModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models.registry import (
    register_observation_model,
)


def wrap_to_pi(angle: Any) -> np.ndarray:
    """Wrap angles into ``(-pi, pi]``, elementwise."""
    return np.arctan2(
        np.sin(np.asarray(angle, dtype=float)), np.cos(np.asarray(angle, dtype=float))
    )


@register_observation_model("goal_relative")
class GoalRelativePoseObservationModel(IsaacObservationModel):
    """The commanded goal expressed in the robot's base frame, with Gaussian noise.

    Reads two state blocks — the robot's planar pose and the goal's world pose — and returns
    ``(dx, dy, dheading)``: the goal offset rotated into the robot's frame, and the difference
    between the goal heading and the robot's own.

    The heading entry is wrapped into ``(-pi, pi]`` both when perceived and when scored. Scoring an
    unwrapped residual is the failure this guards: a particle a hair on the far side of the wrap
    boundary from the reading differs by ``2*pi``, gets a weight of essentially zero, and the belief
    collapses onto whichever side of the discontinuity it happened to start.

    Attributes:
        channel: The observation-dict key this model produces.
        state_channels: The pose block and the goal block, in that order.
        position_std: Std of the noise on each of ``dx`` and ``dy``, in metres.
        heading_std: Std of the noise on ``dheading``, in radians.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> model = GoalRelativePoseObservationModel(
        ...     channel="goal_relative", position_std=1e-6, heading_std=1e-6)
        >>> state = {"base_pose": np.array([1.0, 0.0, np.pi / 2]),
        ...          "goal": np.array([1.0, 2.0, np.pi / 2])}
        >>> reading = model.perceive(state)  # the goal is 2 m straight ahead
        >>> np.round(reading, 3).tolist()
        [2.0, 0.0, 0.0]
        >>> float(model.log_probability(state, reading)) > 0.0  # a near-degenerate density
        True
    """

    supports_density = True

    def __init__(
        self,
        channel: str = "goal_relative",
        pose_channel: str = "base_pose",
        goal_channel: str = "goal",
        pose_indices: Tuple[int, int, int] = (0, 1, 2),
        goal_indices: Optional[Tuple[int, int, int]] = (0, 1, 2),
        position_std: float = 0.1,
        heading_std: float = 0.05,
    ) -> None:
        """Initialize the goal-relative model.

        Args:
            channel: The observation-dict key this model produces.
            pose_channel: The state block holding the robot's planar pose.
            goal_channel: The state block holding the goal's world pose.
            pose_indices: Positions of ``(x, y, yaw)`` within the pose block.
            goal_indices: Positions of ``(x, y, heading)`` within the goal block. Pass ``None``
                for a goal block that carries position only; the heading entry is then reported
                relative to a goal heading of zero.
            position_std: Std of the noise on ``dx`` and ``dy``, in metres.
            heading_std: Std of the noise on ``dheading``, in radians.

        Raises:
            ValueError: If ``position_std`` or ``heading_std`` is not positive.
        """
        if position_std <= 0.0:
            raise ValueError(f"position_std must be positive, got {position_std}")
        if heading_std <= 0.0:
            raise ValueError(f"heading_std must be positive, got {heading_std}")

        self.channel = channel
        self.pose_channel = pose_channel
        self.goal_channel = goal_channel
        self.state_channels = (pose_channel, goal_channel)
        self.pose_indices = tuple(int(index) for index in pose_indices)
        self.goal_indices = None if goal_indices is None else tuple(int(i) for i in goal_indices)
        self.position_std = float(position_std)
        self.heading_std = float(heading_std)

    def clean_offset(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        """Noise-free ``(dx, dy, dheading)`` in the robot's base frame.

        Args:
            clean_state: The state's named blocks; the pose and goal blocks are read.

        Returns:
            Shape ``(3,)``, with the heading entry wrapped into ``(-pi, pi]``.
        """
        pose = np.asarray(clean_state[self.pose_channel], dtype=float).reshape(-1)
        goal = np.asarray(clean_state[self.goal_channel], dtype=float).reshape(-1)
        x, y, yaw = (float(pose[index]) for index in self.pose_indices)
        if self.goal_indices is None:
            goal_x, goal_y, goal_heading = float(goal[0]), float(goal[1]), 0.0
        else:
            goal_x, goal_y, goal_heading = (float(goal[index]) for index in self.goal_indices)

        offset_x, offset_y = goal_x - x, goal_y - y
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        return np.array(
            [
                offset_x * cos_yaw + offset_y * sin_yaw,
                -offset_x * sin_yaw + offset_y * cos_yaw,
                float(wrap_to_pi(goal_heading - yaw)),
            ]
        )

    def _scales(self) -> np.ndarray:
        return np.array([self.position_std, self.position_std, self.heading_std])

    def perceive(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        noisy = self.clean_offset(clean_state) + np.random.normal(0.0, self._scales())
        noisy[2] = float(wrap_to_pi(noisy[2]))
        return noisy

    def log_probability(
        self, clean_state: Mapping[str, np.ndarray], channel_observation: Any
    ) -> float:
        truth = self.clean_offset(clean_state)
        reading = np.asarray(channel_observation, dtype=float).reshape(-1)
        if reading.shape != truth.shape:
            return float("-inf")
        scales = self._scales()
        difference = reading - truth
        difference[2] = float(wrap_to_pi(difference[2]))
        residual = difference / scales
        return float(
            -0.5 * np.sum(residual**2) - np.sum(np.log(scales)) - 1.5 * np.log(2.0 * np.pi)
        )
