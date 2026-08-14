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
    command_pose_base,
    command_pose_world,
    concat_extractors,
    constant_extractor,
    contact_body_indices,
    contact_impulse,
    height_scan,
    make_peak_contact_force_extractor,
    peak_contact_force,
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


def test_height_scan_defaults_to_world_frame_heights() -> None:
    """The default must match what HeightScanObservationModel predicts, which is absolute.

    Purpose: Validates the height-scan reader's default reference frame

    Given: A sensor 1 m up with hits at z = 0.0 and z = 0.4
    When: The scan is read with the default convention
    Then: It reports the hits' world heights, 0.0 and 0.4, not their offsets from the sensor

    Test type: unit
    """
    hits = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.4]]], dtype=torch.float64)
    origin = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    assert height_scan(_sensor_env(hits, origin), sensor="lidar") == pytest.approx([0.0, 0.4])


def test_height_scan_can_report_sensor_relative_heights() -> None:
    """IsaacLab's own observation terms use the sensor-relative form, so it stays available.

    Purpose: Validates the opt-in sensor-relative convention

    Given: The same sensor 1 m up
    When: The scan is read with relative_to_sensor set
    Then: It reports -1.0 and -0.6

    Test type: unit
    """
    hits = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.4]]], dtype=torch.float64)
    origin = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    scan = height_scan(_sensor_env(hits, origin), sensor="lidar", relative_to_sensor=True)
    assert scan == pytest.approx([-1.0, -0.6])


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


class _FakeContactSensor(_FakeEntity):
    """A contact sensor exposing both a body-name list and a force history."""

    def __init__(self, body_names, data) -> None:
        super().__init__(data)
        self.body_names = body_names


def _contact_env(history: torch.Tensor, step_dt: float = 0.2, physics_dt: float = 0.05) -> Any:
    """A contact sensor over one env, whose history is ``(num_envs, samples, bodies, 3)``."""
    sensor = _FakeContactSensor(
        ["base", "LF_FOOT", "RF_FOOT"],
        SimpleNamespace(net_forces_w_history=history.unsqueeze(0)),
    )
    return SimpleNamespace(
        unwrapped=SimpleNamespace(
            scene=_FakeScene({"contact_forces": sensor}),
            step_dt=step_dt,
            physics_dt=physics_dt,
        )
    )


def _command_env(base: torch.Tensor, world_pos: torch.Tensor, world_heading: torch.Tensor) -> Any:
    term = SimpleNamespace(command=base, pos_command_w=world_pos, heading_command_w=world_heading)
    manager = SimpleNamespace(get_term=lambda name: term)
    return SimpleNamespace(unwrapped=SimpleNamespace(command_manager=manager))


def test_command_pose_world_reads_the_privileged_target() -> None:
    """The world-frame target is what makes the base-frame reading a function of the pose.

    Purpose: Validates that the command's world target is read as (x, y, heading)

    Given: A command term holding a world position and heading
    When: The world pose is read
    Then: It is the planar position followed by the heading

    Test type: unit
    """
    env = _command_env(
        torch.zeros((1, 4), dtype=torch.float64),
        torch.tensor([[1.5, -2.5, 0.6]], dtype=torch.float64),
        torch.tensor([0.25], dtype=torch.float64),
    )
    assert command_pose_world(env) == pytest.approx(np.array([1.5, -2.5, 0.25]))


def test_command_pose_base_reads_what_the_robot_sees() -> None:
    """The base-frame command is the observation; the world target must not leak into it.

    Purpose: Validates that the command term's own value is returned flat

    Given: A command term holding a four-wide base-frame command
    When: The base pose is read
    Then: It is that vector for the single env

    Test type: unit
    """
    env = _command_env(
        torch.tensor([[2.0, 0.5, 0.0, -0.3]], dtype=torch.float64),
        torch.zeros((1, 3), dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
    )
    assert command_pose_base(env) == pytest.approx(np.array([2.0, 0.5, 0.0, -0.3]))


def test_command_readers_reject_a_task_with_no_command_manager() -> None:
    """Guessing a goal would silently invent the quantity the study measures against.

    Purpose: Validates the error when the task exposes no command manager

    Given: An env whose unwrapped object has no command manager
    When: A command reader is called
    Then: RuntimeError names the command and asks for an explicit extractor

    Test type: unit
    """
    env = SimpleNamespace(unwrapped=SimpleNamespace())
    with pytest.raises(RuntimeError, match="pose_command"):
        command_pose_base(env)


def test_contact_body_indices_matches_names_in_full() -> None:
    """A substring match would fold the base into a foot-only measurement.

    Purpose: Validates full-match regex body selection

    Given: A sensor with bodies base, LF_FOOT and RF_FOOT
    When: The feet are selected by a suffix regex
    Then: Only the two feet are returned, in sensor order

    Test type: unit
    """
    env = _contact_env(torch.zeros((4, 3, 3), dtype=torch.float64))
    assert contact_body_indices(env, "contact_forces", ".*FOOT") == [1, 2]
    assert contact_body_indices(env, "contact_forces", "base") == [0]


def test_contact_body_indices_rejects_a_pattern_matching_nothing() -> None:
    """An empty body set reduces to a measurement over no bodies, silently.

    Purpose: Validates the error when no body name matches

    Given: A sensor with no body called "gripper"
    When: That pattern is selected
    Then: RuntimeError lists the names that are available

    Test type: unit
    """
    env = _contact_env(torch.zeros((4, 3, 3), dtype=torch.float64))
    with pytest.raises(RuntimeError, match="LF_FOOT"):
        contact_body_indices(env, "contact_forces", "gripper")


def test_peak_contact_force_finds_the_transient_the_mean_hides() -> None:
    """A spike lasting one substep is the whole difference between a hard and a soft hit.

    Purpose: Validates that the peak is the maximum over samples and bodies

    Given: A base whose force spikes to 900 N for one of four samples and is 100 N otherwise
    When: The peak force over the base is read
    Then: It is the spike, not the 300 N mean the impulse reports

    Test type: unit
    """
    history = torch.zeros((4, 3, 3), dtype=torch.float64)
    history[:, 0, 0] = torch.tensor([900.0, 100.0, 100.0, 100.0], dtype=torch.float64)
    env = _contact_env(history)
    assert peak_contact_force(env, "contact_forces", "base") == pytest.approx(900.0)
    assert contact_impulse(env, "contact_forces", "base") == pytest.approx(300.0 * 0.2)


def test_contact_readers_use_only_this_step_of_the_history() -> None:
    """Folding in older substeps makes the reading depend on how the sensor was configured.

    Purpose: Validates that the history is sliced to the step's substeps

    Given: A history of 8 samples where a control step spans 4, with a spike in the stale half
    When: The peak force is read
    Then: The stale spike is excluded

    Test type: unit
    """
    history = torch.zeros((8, 3, 3), dtype=torch.float64)
    history[:, 0, 0] = torch.tensor(
        [10.0, 10.0, 10.0, 10.0, 5000.0, 0.0, 0.0, 0.0], dtype=torch.float64
    )
    env = _contact_env(history, step_dt=0.2, physics_dt=0.05)
    assert peak_contact_force(env, "contact_forces", "base") == pytest.approx(10.0)


def test_a_sensor_without_history_falls_back_to_the_end_of_step_reading() -> None:
    """A task whose sensor keeps no history must still report something, not raise.

    Purpose: Validates the no-history path

    Given: A sensor exposing only net_forces_w
    When: The peak force is read
    Then: It is the magnitude of that single reading

    Test type: unit
    """
    sensor = _FakeContactSensor(
        ["base"], SimpleNamespace(net_forces_w=torch.tensor([[[3.0, 4.0, 0.0]]]))
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            scene=_FakeScene({"contact_forces": sensor}), step_dt=0.2, physics_dt=0.05
        )
    )
    assert peak_contact_force(env, "contact_forces", "base") == pytest.approx(5.0)


def test_peak_contact_force_extractor_binds_the_sensor_and_bodies() -> None:
    """The extractor is handed to the world as a bare env -> float, so the binding is the API.

    Purpose: Validates the impact_extractor factory

    Given: An extractor bound to the feet of a contact sensor
    When: It is called on a scene whose feet peak at 700 N and whose base peaks higher
    Then: It reports the feet's peak

    Test type: unit
    """
    history = torch.zeros((4, 3, 3), dtype=torch.float64)
    history[:, 0, 0] = 5000.0
    history[:, 1, 0] = torch.tensor([700.0, 10.0, 10.0, 10.0], dtype=torch.float64)
    extractor = make_peak_contact_force_extractor("contact_forces", ".*FOOT")
    assert extractor(_contact_env(history)) == pytest.approx(700.0)
