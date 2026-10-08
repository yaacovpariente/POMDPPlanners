# SPDX-License-Identifier: MIT

"""Single-frame readers over a live IsaacLab scene.

These are the world-side half of the perception stack: plain ``env -> np.ndarray`` functions
suitable for :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_pomdp.IsaacLabPOMDP`'s
``state_extractor`` and ``observation_extractor`` hooks. They read one frame from the scene and
return a flat vector; nothing here holds state or touches the planner.

Keeping them separate from the per-channel observation models is what makes the
``observation = h(state)`` split real. The world reads a genuine sensor buffer through
:func:`ray_caster_ranges`; the planner's generative model *predicts* that buffer through a
:class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models.ray_caster_models.RayCasterObservationModel`.
The two are independent implementations of the same quantity, which is exactly the property that
lets a localisation error show up as one.

Functions:
    root_planar_pose: The articulation's ``(x, y, yaw)`` in the world frame.
    root_state: Root pose and velocity as one flat vector.
    joint_state: Joint positions and velocities as one flat vector.
    ray_caster_ranges: Distance from a ``RayCaster``'s origin to each of its ray hits.
    height_scan: Height of each ray hit relative to the sensor origin.
    policy_observation: The task's own policy observation group.
    command_pose_world: A pose command's target in world coordinates.
    command_pose_base: A pose command's target in the robot's base frame.
    contact_body_indices: Contact-sensor body slots whose names match a regex.
    peak_contact_force: Largest instantaneous contact force over a control step.
    contact_impulse: Step-averaged contact impulse over a control step.
    make_peak_contact_force_extractor: Bind :func:`peak_contact_force` to a body set.
    concat_extractors: Compose several extractors into one flat vector.
"""

import re
from typing import Any, Callable, List, Sequence

import numpy as np

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_pomdp import (
    _to_numpy,
    contact_force_samples,
    step_duration,
)

Extractor = Callable[[Any], np.ndarray]


def yaw_from_quaternion(quaternion: Any) -> float:
    """Yaw angle (radians) of a ``(w, x, y, z)`` quaternion, IsaacLab's convention."""
    w, x, y, z = np.asarray(quaternion, dtype=float).reshape(-1)[:4]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def root_planar_pose(env: Any, asset: str = "robot") -> np.ndarray:
    """Read the articulation's planar pose ``(x, y, yaw)`` from the physics buffers.

    Args:
        env: The live IsaacLab env.
        asset: Scene key of the articulation. Defaults to ``"robot"``.

    Returns:
        Shape ``(3,)`` of ``(x, y, yaw)`` in the world frame.
    """
    data = env.unwrapped.scene[asset].data
    position = _to_numpy(data.root_pos_w)[0].reshape(-1)
    yaw = yaw_from_quaternion(_to_numpy(data.root_quat_w)[0])
    return np.array([position[0], position[1], yaw])


def root_state(env: Any, asset: str = "robot") -> np.ndarray:
    """Read root position, orientation and both velocities as one flat vector.

    Args:
        env: The live IsaacLab env.
        asset: Scene key of the articulation. Defaults to ``"robot"``.

    Returns:
        The concatenation of ``root_pos_w``, ``root_quat_w``, ``root_lin_vel_w`` and
        ``root_ang_vel_w`` for the single env.
    """
    data = env.unwrapped.scene[asset].data
    parts = ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w")
    return np.concatenate([_to_numpy(getattr(data, part))[0].reshape(-1) for part in parts])


def joint_state(env: Any, asset: str = "robot") -> np.ndarray:
    """Read joint positions and velocities as one flat vector.

    Args:
        env: The live IsaacLab env.
        asset: Scene key of the articulation. Defaults to ``"robot"``.

    Returns:
        The concatenation of ``joint_pos`` and ``joint_vel`` for the single env.
    """
    data = env.unwrapped.scene[asset].data
    return np.concatenate(
        [_to_numpy(getattr(data, part))[0].reshape(-1) for part in ("joint_pos", "joint_vel")]
    )


def _ray_hits(env: Any, sensor: str) -> np.ndarray:
    data = env.unwrapped.scene[sensor].data
    hits = getattr(data, "ray_hits_w", None)
    if hits is None:
        raise RuntimeError(
            f"Scene sensor '{sensor}' exposes no ray_hits_w buffer; it is not a RayCaster."
        )
    return _to_numpy(hits)[0].reshape(-1, 3)


def _sensor_origin(env: Any, sensor: str) -> np.ndarray:
    data = env.unwrapped.scene[sensor].data
    return _to_numpy(data.pos_w)[0].reshape(-1)


def ray_caster_ranges(env: Any, sensor: str = "lidar", max_range: float = 10.0) -> np.ndarray:
    """Distance from a ``RayCaster``'s origin to each of its ray hits.

    IsaacLab reports ray hits as world-frame points and writes ``inf`` where a ray hit nothing.
    Those are clipped to ``max_range`` so the reading stays finite: an ``inf`` entry propagates
    through a belief weight and silently kills the particle.

    Args:
        env: The live IsaacLab env.
        sensor: Scene key of the ``RayCaster``. Defaults to ``"lidar"``.
        max_range: Range substituted for a ray that hit nothing.

    Returns:
        Shape ``(R,)`` of ranges in ``[0, max_range]``.
    """
    hits = _ray_hits(env, sensor)
    origin = _sensor_origin(env, sensor)
    distances = np.linalg.norm(hits - origin[np.newaxis, :], axis=-1)
    return np.clip(np.nan_to_num(distances, nan=max_range, posinf=max_range), 0.0, max_range)


def height_scan(
    env: Any, sensor: str = "height_scanner", relative_to_sensor: bool = False
) -> np.ndarray:
    """World-frame height of each ray hit, or its height relative to the sensor.

    The default is the **world-frame** height, because that is the quantity
    :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models.ray_caster_models.HeightScanObservationModel`
    predicts: its ``floor_height`` and obstacle heights are absolute. Pairing the two on
    different conventions is silent and expensive — over a zero-height floor the world would
    report a large negative number where the model expects zero, and every particle weight would
    be wrong in the same direction, which reads as "the belief cannot localise" rather than as a
    unit mismatch.

    Args:
        env: The live IsaacLab env.
        sensor: Scene key of the height-scanner ``RayCaster``.
        relative_to_sensor: Return ``hit_z - sensor_z`` instead. This is the convention
            IsaacLab's own height-scan observation terms use; pick it only when whatever consumes
            the reading uses it too.

    Returns:
        Shape ``(M,)`` of heights, one per ray.
    """
    hits = _ray_hits(env, sensor)
    heights = hits[:, 2]
    if relative_to_sensor:
        heights = heights - _sensor_origin(env, sensor)[2]
    return np.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)


def policy_observation(env: Any, group: str = "policy") -> np.ndarray:
    """Read the task's own observation group, the vector its trained policy consumes.

    Args:
        env: The live IsaacLab env.
        group: Observation-manager group name. Defaults to ``"policy"``.

    Returns:
        The group's flat observation vector for the single env.

    Raises:
        RuntimeError: If the task exposes no observation manager, or no such group.
    """
    manager = getattr(env.unwrapped, "observation_manager", None)
    compute = getattr(manager, "compute_group", None)
    if compute is None:
        raise RuntimeError(
            f"Task exposes no observation manager, so group '{group}' cannot be read; "
            "pass an explicit extractor."
        )
    return _to_numpy(compute(group))[0].reshape(-1)


def _command_term(env: Any, command_name: str) -> Any:
    manager = getattr(env.unwrapped, "command_manager", None)
    get_term = getattr(manager, "get_term", None)
    if get_term is None:
        raise RuntimeError(
            f"Task exposes no command manager, so command '{command_name}' cannot be read; "
            "pass an explicit extractor."
        )
    return get_term(command_name)


def command_pose_world(env: Any, command_name: str = "pose_command") -> np.ndarray:
    """Read a 2-D pose command's target as ``(x, y, heading)`` in world coordinates.

    This is **privileged**: the command term keeps the target in world coordinates for its own
    bookkeeping, and a robot only ever sees the base-frame version through
    :func:`command_pose_base`. It belongs in the state, where it makes the base-frame reading a
    genuine function of the robot's pose — which is what turns that reading into a localisation
    signal instead of a restatement of the state.

    Args:
        env: The live IsaacLab env.
        command_name: Name of the command term. Defaults to ``"pose_command"``.

    Returns:
        Shape ``(3,)`` of ``(x, y, heading)`` in the world frame.

    Raises:
        RuntimeError: If the task exposes no command manager, or the named term keeps no
            world-frame target.
    """
    term = _command_term(env, command_name)
    position = getattr(term, "pos_command_w", None)
    heading = getattr(term, "heading_command_w", None)
    if position is None or heading is None:
        raise RuntimeError(
            f"Command term '{command_name}' exposes no world-frame target "
            "(pos_command_w / heading_command_w); pass an explicit extractor."
        )
    world = _to_numpy(position)[0].reshape(-1)
    return np.array([world[0], world[1], float(_to_numpy(heading).reshape(-1)[0])])


def command_pose_base(env: Any, command_name: str = "pose_command") -> np.ndarray:
    """Read a command term's value in the robot's base frame — what the robot actually sees.

    Args:
        env: The live IsaacLab env.
        command_name: Name of the command term. Defaults to ``"pose_command"``.

    Returns:
        The command vector for the single env, flat. For a 2-D pose command that is
        ``(dx, dy, dz, dheading)`` with the offsets expressed in the base frame.

    Raises:
        RuntimeError: If the task exposes no command manager or no such term.
    """
    return _to_numpy(_command_term(env, command_name).command)[0].reshape(-1)


def contact_body_indices(
    env: Any, sensor: str = "contact_forces", pattern: str = ".*"
) -> List[int]:
    """Body slots of a contact sensor whose names fully match a regex.

    Args:
        env: The live IsaacLab env.
        sensor: Scene key of the ``ContactSensor``.
        pattern: Regex matched against each body name in full, IsaacLab's own convention for
            body selection. Defaults to every body.

    Returns:
        The matching indices into the sensor's body axis, in sensor order.

    Raises:
        RuntimeError: If no body name matches, which would otherwise reduce silently to a
            measurement over an empty body set.
    """
    names = list(env.unwrapped.scene[sensor].body_names)
    compiled = re.compile(pattern)
    indices = [index for index, name in enumerate(names) if compiled.fullmatch(name)]
    if not indices:
        raise RuntimeError(
            f"No body of scene sensor '{sensor}' matches '{pattern}'; available: {names}"
        )
    return indices


def _contact_magnitudes(env: Any, sensor: str, pattern: str) -> np.ndarray:
    samples = contact_force_samples(env, sensor)
    magnitudes = np.linalg.norm(samples.reshape(samples.shape[0], -1, 3), axis=-1)
    return magnitudes[:, contact_body_indices(env, sensor, pattern)]


def peak_contact_force(env: Any, sensor: str = "contact_forces", pattern: str = ".*") -> float:
    """Largest instantaneous contact force (N) on the selected bodies this control step.

    Prefer this to :func:`contact_impulse` when the point is *how hard* an obstacle was, not how
    long it was leaned on. A step-averaged impulse cannot tell a hard obstacle from a soft one:
    pushing steadily against an immovable post and shoving a light one along integrate to about the
    same number, and on a legged robot the post-impact lean dominates the average outright. The
    difference lives in the transient at first contact, which only the peak sees — and it is only
    visible at all when the sensor's ``history_length`` covers the step's substeps.

    Args:
        env: The live IsaacLab env.
        sensor: Scene key of the ``ContactSensor``.
        pattern: Regex selecting the bodies to measure over.

    Returns:
        The peak force magnitude in newtons.
    """
    return float(_contact_magnitudes(env, sensor, pattern).max())


def contact_impulse(env: Any, sensor: str = "contact_forces", pattern: str = ".*") -> float:
    """Step-averaged contact impulse (N*s) on the worst of the selected bodies.

    Args:
        env: The live IsaacLab env.
        sensor: Scene key of the ``ContactSensor``.
        pattern: Regex selecting the bodies to measure over.

    Returns:
        The largest per-body mean force over the step, times the control-step duration.
    """
    return float(_contact_magnitudes(env, sensor, pattern).mean(axis=0).max()) * step_duration(env)


def make_peak_contact_force_extractor(
    sensor: str = "contact_forces", pattern: str = ".*"
) -> Callable[[Any], float]:
    """Bind :func:`peak_contact_force` to a sensor and body set.

    Shaped for ``IsaacLabPOMDP``'s ``impact_extractor`` hook, which replaces the world's default
    step-averaged impulse with the peak.

    Args:
        sensor: Scene key of the ``ContactSensor``.
        pattern: Regex selecting the bodies to measure over.

    Returns:
        An ``env -> float`` reader of the peak contact force in newtons.
    """

    def _extract(env: Any) -> float:
        return peak_contact_force(env, sensor, pattern)

    return _extract


def concat_extractors(*extractors: Extractor) -> Extractor:
    """Compose extractors into one that concatenates their outputs.

    Args:
        *extractors: The ``env -> np.ndarray`` readers to run, in packing order.

    Returns:
        An ``env -> np.ndarray`` extractor producing the concatenated vector.
    """

    def _extract(env: Any) -> np.ndarray:
        return np.concatenate(
            [np.asarray(extractor(env), dtype=float).reshape(-1) for extractor in extractors]
        )

    return _extract


def constant_extractor(values: Sequence[float]) -> Extractor:
    """An extractor returning a fixed block, for a channel the world supplies out of band.

    A latent hazard type is drawn once per episode by the study, not measured by the simulator, so
    the block that carries it into the state vector is a constant for the episode's duration.

    Args:
        values: The block to emit.

    Returns:
        An ``env -> np.ndarray`` extractor ignoring its argument.
    """
    block = np.asarray(values, dtype=float).reshape(-1).copy()

    def _extract(env: Any) -> np.ndarray:
        del env
        return block.copy()

    return _extract
