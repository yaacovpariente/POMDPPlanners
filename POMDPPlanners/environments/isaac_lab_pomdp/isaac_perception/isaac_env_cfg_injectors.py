# SPDX-License-Identifier: MIT

"""Reshape a task config before the env is built.

An IsaacLab task's config is fixed by its registration, and what a study needs is usually switched
off there: the flat ANYmal variants delete the height scanner their rough-terrain parent defines,
most manipulation tasks ship no contact sensor, and the reset events cover only the terms the task
author happened to need. These builders return ``env_cfg_modifier`` callables for
:class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_pomdp.IsaacLabPOMDP`, which applies
them to the parsed config before the env is built — the one point where any of it can still change.

Three catches that are easy to hit and expensive to diagnose:

* **A stock ``RayCaster`` casts against exactly one mesh** and raises on a second. So a height
  scanner configured the way the locomotion tasks configure it — ``mesh_prim_paths=["/World/ground"]``
  — sees the floor and *nothing else*, silently: obstacles the study spawned are simply not there,
  and the scan reads as flat ground. :func:`make_height_scanner_injector` uses
  ``MultiMeshRayCaster`` whenever more than one path is asked for, which also tracks a moving
  obstacle's transform rather than freezing it at spawn.
* **A contact sensor's ``history_length`` is independent of the control step.** The locomotion
  tasks keep 3 samples while one control step can span 40 physics substeps, so an impact transient
  falls between samples and only the sustained push afterwards is recorded — which reads a hard
  obstacle and a soft one as the same number.
* **A reset event only resets what it names.** The ANYmal navigation task resets the robot's root
  pose and nothing else, which is invisible while a process runs one episode and wrong as soon as
  it runs two: the second episode starts with the *first* episode's joint angles under a
  teleported base, and the robot falls over at step 2. Measured over 8 episodes, that produced a
  9 kN base contact in 3 of them before the robot had gone anywhere near a hazard.

Functions:
    make_height_scanner_injector: Attach a downward grid ray-caster over named meshes.
    make_contact_history_injector: Widen a contact sensor's history to the task's decimation.
    make_joint_reset_injector: Put an articulation's joints back to default on every reset.
    disable_terminations_injector: Switch named termination terms off.
    compose_env_cfg_modifiers: Run several modifiers in order as one.
"""

from typing import Any, Callable, Optional, Sequence, Tuple

EnvCfgModifier = Callable[[Any], None]


def compose_env_cfg_modifiers(*modifiers: Optional[EnvCfgModifier]) -> EnvCfgModifier:
    """Run several ``env_cfg_modifier`` callables in order, as one.

    Args:
        *modifiers: The modifiers to run, in order. ``None`` entries are skipped so a caller can
            pass an optional hook through without branching.

    Returns:
        A single ``cfg -> None`` modifier applying each in turn.
    """
    selected = [modifier for modifier in modifiers if modifier is not None]

    def _inject(cfg: Any) -> None:
        for modifier in selected:
            modifier(cfg)

    return _inject


def disable_terminations_injector(*terms: str) -> EnvCfgModifier:
    """Switch named termination terms off.

    IsaacLab auto-resets inside ``env.step`` when a term fires, which zeroes the scene's sensor
    buffers before the caller can read them. A study that *measures* a collision rather than
    treating it as an episode-ending failure has to turn the corresponding term off, or the
    measurement is destroyed by the reset it triggers.

    Args:
        *terms: Attribute names on ``cfg.terminations`` to set to ``None``. A term the task does
            not define is skipped.

    Returns:
        A ``cfg -> None`` modifier.
    """

    def _inject(cfg: Any) -> None:
        for term in terms:
            if getattr(cfg.terminations, term, None) is not None:
                setattr(cfg.terminations, term, None)

    return _inject


def make_contact_history_injector(
    sensor_key: str = "contact_forces",
    history_length: Optional[int] = None,
) -> EnvCfgModifier:
    """Widen a contact sensor's force history so it covers a whole control step.

    Args:
        sensor_key: Scene key of the ``ContactSensor``. A task without one is left alone.
        history_length: Explicit number of samples to keep. Defaults to the task's ``decimation``,
            i.e. one sample per physics substep of a control step.

    Returns:
        A ``cfg -> None`` modifier.
    """

    def _inject(cfg: Any) -> None:
        sensor = getattr(cfg.scene, sensor_key, None)
        if sensor is None:
            return
        length = (
            history_length if history_length is not None else int(getattr(cfg, "decimation", 0))
        )
        if length > 0:
            sensor.history_length = length

    return _inject


def make_joint_reset_injector(
    asset: str = "robot", term_name: str = "reset_default_joints"
) -> EnvCfgModifier:
    """Add a reset event putting an articulation's joints back to their default pose.

    Needed whenever one process runs more than one episode on a task whose reset events cover only
    the root pose. Without it the robot inherits the previous episode's joint angles under a
    teleported base — a legged robot then spends the first steps falling over, and every contact
    measurement in that window is a fall rather than whatever the study meant to measure.

    Args:
        asset: Scene key of the articulation.
        term_name: Attribute name the event term is registered under. Terms run in definition
            order and this one is appended, so it lands after the task's own reset terms.

    Returns:
        A ``cfg -> None`` modifier.
    """

    def _inject(cfg: Any) -> None:
        # pylint: disable-next=import-outside-toplevel,import-error
        from isaaclab.envs import mdp

        # pylint: disable-next=import-outside-toplevel,import-error
        from isaaclab.managers import EventTermCfg, SceneEntityCfg

        setattr(
            cfg.events,
            term_name,
            EventTermCfg(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (0.0, 0.0),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg(asset),
                },
            ),
        )

    return _inject


def _raycast_targets(static_paths: Sequence[str], tracked_paths: Sequence[str]) -> list:
    """Wrap each path in a target cfg, tracking only the ones that can move."""
    # pylint: disable-next=import-outside-toplevel,import-error
    from isaaclab.sensors import MultiMeshRayCasterCfg

    return [
        MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=path, track_mesh_transforms=tracked)
        for paths, tracked in ((static_paths, False), (tracked_paths, True))
        for path in paths
    ]


def make_height_scanner_injector(
    sensor_key: str = "height_scanner",
    prim_path: str = "{ENV_REGEX_NS}/Robot/base",
    mesh_prim_paths: Sequence[str] = ("/World/ground",),
    tracked_mesh_prim_paths: Sequence[str] = (),
    resolution: float = 0.1,
    size: Tuple[float, float] = (1.6, 1.0),
    offset_z: float = 20.0,
    ray_alignment: str = "yaw",
    update_period: float = 0.0,
    debug_vis: bool = False,
) -> EnvCfgModifier:
    """Attach a downward grid ray-caster reading the ground under and around a body.

    The defaults reproduce the scanner the IsaacLab locomotion tasks define — a 0.1 m grid over
    1.6 x 1.0 m, cast downward from 20 m up, following the body's yaw only — except that they cast
    against every named mesh rather than just the floor.

    Args:
        sensor_key: Scene key to register the sensor under.
        prim_path: Prim the sensor rides on, in IsaacLab's scene-namespace form.
        mesh_prim_paths: Prim paths or regexes to cast against whose pose never changes. More
            than one target in total requires ``MultiMeshRayCaster``, which this selects
            automatically.
        tracked_mesh_prim_paths: Prim paths or regexes whose transform is re-read every step, for
            obstacles that can be pushed. Keep the ground out of this list: a terrain plane is not
            an xformable prim and tracking it raises at sensor init.
        resolution: Grid spacing in metres.
        size: Grid extent ``(length, width)`` in metres.
        offset_z: Height above the body the rays start from. It must clear the tallest obstacle,
            since a ray starting *inside* a mesh does not hit it.
        ray_alignment: ``"base"``, ``"yaw"`` or ``"world"``. ``"yaw"`` keeps the grid level while
            following the body's heading, which is what makes the scan a function of the planar
            pose alone.
        update_period: Seconds between sensor updates. ``0.0`` updates every step.
        debug_vis: Draw the ray hits in the viewer.

    Returns:
        A ``cfg -> None`` modifier attaching the sensor.

    Raises:
        ValueError: If no mesh path is given at all, which would build a sensor that can hit
            nothing.
    """
    static_paths, tracked_paths = list(mesh_prim_paths), list(tracked_mesh_prim_paths)
    if not static_paths and not tracked_paths:
        raise ValueError("at least one mesh prim path is needed to cast against")

    def _inject(cfg: Any) -> None:
        # pylint: disable-next=import-outside-toplevel,import-error
        from isaaclab.sensors import MultiMeshRayCasterCfg, RayCasterCfg, patterns

        common = {
            "prim_path": prim_path,
            "offset": RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, offset_z)),
            "ray_alignment": ray_alignment,
            "pattern_cfg": patterns.GridPatternCfg(resolution=resolution, size=list(size)),
            "update_period": update_period,
            "debug_vis": debug_vis,
        }
        if len(static_paths) == 1 and not tracked_paths:
            sensor = RayCasterCfg(mesh_prim_paths=static_paths, **common)
        else:
            sensor = MultiMeshRayCasterCfg(
                mesh_prim_paths=_raycast_targets(static_paths, tracked_paths), **common
            )
        setattr(cfg.scene, sensor_key, sensor)

    return _inject
