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
    concat_extractors: Compose several extractors into one flat vector.
"""

from typing import Any, Callable, Sequence

import numpy as np

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_pomdp import _to_numpy

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
