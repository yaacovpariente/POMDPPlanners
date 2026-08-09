# SPDX-License-Identifier: MIT

"""Unit tests for the world-side single-frame Isaac scene readers.

Driven against a scripted fake scene rather than a live simulator: these functions are pure
readers, so a fake with IsaacLab's buffer names and shapes exercises everything that can go wrong
in them (frame conventions, infinite ray hits, batch indexing).
"""

from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest
import torch

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
    concat_extractors,
    constant_extractor,
    height_scan,
    joint_state,
    policy_observation,
    ray_caster_ranges,
    root_planar_pose,
    root_state,
    yaw_from_quaternion,
)


class _FakeEntity:
    """A scene entry exposing a ``.data`` buffer namespace, as IsaacLab's do."""

    def __init__(self, data: Any) -> None:
        self.data = data


class _FakeScene:
    """A scene mapping keys to entities."""

    def __init__(self, entries: Dict[str, Any]) -> None:
        self._entries = entries

    def __getitem__(self, key: str) -> Any:
        return self._entries[key]


def _quaternion_for_yaw(yaw: float) -> torch.Tensor:
    """IsaacLab's ``(w, x, y, z)`` quaternion for a pure yaw rotation."""
    return torch.tensor([[np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]], dtype=torch.float64)


def _robot_env(yaw: float = 0.0) -> Any:
    articulation = _FakeEntity(
        SimpleNamespace(
            root_pos_w=torch.tensor([[1.0, 2.0, 0.5]], dtype=torch.float64),
            root_quat_w=_quaternion_for_yaw(yaw),
            root_lin_vel_w=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64),
            root_ang_vel_w=torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float64),
            joint_pos=torch.tensor([[0.7, 0.8]], dtype=torch.float64),
            joint_vel=torch.tensor([[0.9, 1.0]], dtype=torch.float64),
        )
    )
    return SimpleNamespace(unwrapped=SimpleNamespace(scene=_FakeScene({"robot": articulation})))


def _sensor_env(hits: torch.Tensor, origin: torch.Tensor) -> Any:
    sensor = _FakeEntity(SimpleNamespace(ray_hits_w=hits, pos_w=origin))
    return SimpleNamespace(unwrapped=SimpleNamespace(scene=_FakeScene({"lidar": sensor})))


@pytest.mark.parametrize("yaw", [0.0, 0.5, -1.25, 3.0])
def test_yaw_from_quaternion_inverts_a_pure_yaw_rotation(yaw: float) -> None:
    """A wrong quaternion convention silently rotates every body-frame ray.

    Purpose: Validates the (w, x, y, z) yaw extraction against known rotations

    Given: A quaternion built from a known yaw in IsaacLab's ordering
    When: The yaw is extracted
    Then: It matches the original angle

    Test type: unit
    """
    quaternion = _quaternion_for_yaw(yaw)[0].numpy()
    assert yaw_from_quaternion(quaternion) == pytest.approx(yaw)


def test_root_planar_pose_reads_x_y_and_yaw() -> None:
    """The planar pose is what the zone geometry and the unicycle model both consume.

    Purpose: Validates the planar-pose reader against a scripted articulation

    Given: A robot at (1, 2) yawed by 0.5 rad
    When: The planar pose is read
    Then: It is (1, 2, 0.5)

    Test type: unit
    """
    assert root_planar_pose(_robot_env(yaw=0.5)) == pytest.approx([1.0, 2.0, 0.5])


def test_root_state_concatenates_pose_and_both_velocities() -> None:
    """The packing order is a contract the state schema is written against.

    Purpose: Validates the root-state reader's width and ordering

    Given: A scripted articulation with position, quaternion and both velocities
    When: The root state is read
    Then: It is the 13-wide concatenation in pose, quaternion, linear, angular order

    Test type: unit
    """
    state = root_state(_robot_env())
    assert state.shape == (13,)
    assert state[:3] == pytest.approx([1.0, 2.0, 0.5])
    assert state[7:10] == pytest.approx([0.1, 0.2, 0.3])
    assert state[10:] == pytest.approx([0.4, 0.5, 0.6])


def test_joint_state_concatenates_positions_then_velocities() -> None:
    """A reader that swapped the halves would be undetectable downstream.

    Purpose: Validates the joint-state reader's ordering

    Given: A scripted articulation with two joints
    When: The joint state is read
    Then: Positions come first, velocities second

    Test type: unit
    """
    assert joint_state(_robot_env()) == pytest.approx([0.7, 0.8, 0.9, 1.0])


def test_ray_caster_ranges_measure_from_the_sensor_origin() -> None:
    """A range measured from the world origin instead of the sensor is silently wrong.

    Purpose: Validates that ranges are distances from the sensor's own position

    Given: A sensor at (1, 0, 0) with hits at 2 m and 3 m along +x
    When: The ranges are read
    Then: They are 1.0 and 2.0, not 2.0 and 3.0

    Test type: unit
    """
    hits = torch.tensor([[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], dtype=torch.float64)
    origin = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    assert ray_caster_ranges(_sensor_env(hits, origin)) == pytest.approx([1.0, 2.0])


def test_ray_caster_ranges_clip_a_ray_that_hit_nothing() -> None:
    """IsaacLab writes ``inf`` for a miss, and an ``inf`` weight kills a belief particle.

    Purpose: Validates that non-finite ray hits are replaced by the max range

    Given: A sensor whose second ray hit nothing
    When: The ranges are read with a 10 m maximum
    Then: The miss reads 10.0 and every entry is finite

    Test type: unit
    """
    hits = torch.tensor(
        [[[1.0, 0.0, 0.0], [float("inf"), float("inf"), float("inf")]]], dtype=torch.float64
    )
    origin = torch.zeros((1, 3), dtype=torch.float64)
    ranges = ray_caster_ranges(_sensor_env(hits, origin), max_range=10.0)
    assert np.all(np.isfinite(ranges))
    assert ranges == pytest.approx([1.0, 10.0])


def test_height_scan_reports_hit_height_relative_to_the_sensor() -> None:
    """A height scan is only comparable across poses if it is sensor-relative.

    Purpose: Validates the height-scan reader's reference frame

    Given: A sensor 1 m up with hits at z = 0.0 and z = 0.4
    When: The scan is read
    Then: It reports -1.0 and -0.6

    Test type: unit
    """
    hits = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.4]]], dtype=torch.float64)
    origin = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    assert height_scan(_sensor_env(hits, origin), sensor="lidar") == pytest.approx([-1.0, -0.6])


def test_a_non_ray_caster_sensor_raises_rather_than_returning_nothing() -> None:
    """Pointing a sensor reader at the wrong scene key must fail loudly.

    Purpose: Validates the error path when the sensor exposes no ray buffer

    Given: A scene entry with no ray_hits_w buffer
    When: The ranges are read
    Then: RuntimeError is raised naming the key

    Test type: unit
    """
    sensor = _FakeEntity(SimpleNamespace(pos_w=torch.zeros((1, 3))))
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=_FakeScene({"lidar": sensor})))
    with pytest.raises(RuntimeError, match="lidar"):
        ray_caster_ranges(env)


def test_policy_observation_reads_the_named_group() -> None:
    """The task's own policy observation is the natural robot block for a schema.

    Purpose: Validates the observation-manager reader

    Given: A task whose observation manager returns a 3-wide policy group
    When: The group is read
    Then: The flat vector comes back

    Test type: unit
    """
    manager = SimpleNamespace(
        compute_group=lambda group: torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(observation_manager=manager))
    assert policy_observation(env) == pytest.approx([1.0, 2.0, 3.0])


def test_policy_observation_raises_when_the_task_has_no_manager() -> None:
    """Guessing a substitute would hand the planner a silently different state.

    Purpose: Validates the error path when no observation manager exists

    Given: A task exposing no observation manager
    When: The policy group is read
    Then: RuntimeError is raised

    Test type: unit
    """
    env = SimpleNamespace(unwrapped=SimpleNamespace())
    with pytest.raises(RuntimeError, match="observation manager"):
        policy_observation(env)


def test_concat_extractors_packs_in_the_order_given() -> None:
    """The composed order is what the state schema's channel order has to match.

    Purpose: Validates the extractor composition helper

    Given: A planar-pose reader followed by a constant hazard block
    When: They are composed and run
    Then: The result is the pose followed by the constant

    Test type: unit
    """
    composed = concat_extractors(root_planar_pose, constant_extractor([0.0, 1.0]))
    assert composed(_robot_env(yaw=0.5)) == pytest.approx([1.0, 2.0, 0.5, 0.0, 1.0])


def test_constant_extractor_hands_back_a_fresh_copy_each_call() -> None:
    """A shared array would let one episode's mutation leak into the next.

    Purpose: Validates that the constant extractor does not alias its stored block

    Given: A constant extractor over a two-entry block
    When: One returned block is mutated and the extractor is called again
    Then: The second call is unaffected

    Test type: unit
    """
    extractor = constant_extractor([0.0, 1.0])
    first = extractor(None)
    first[0] = 99.0
    assert extractor(None) == pytest.approx([0.0, 1.0])
