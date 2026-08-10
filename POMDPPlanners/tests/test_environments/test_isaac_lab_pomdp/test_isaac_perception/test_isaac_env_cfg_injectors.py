# SPDX-License-Identifier: MIT

"""Unit tests for the task-config injectors.

The injectors mutate a parsed IsaacLab config, so they are driven against a stand-in config with
the attribute names IsaacLab uses. The builders that need ``isaaclab`` itself only import it inside
the returned callable, which is what lets everything up to the call be tested without a simulator;
those calls are exercised against an injected fake module instead.
"""

import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List

import pytest

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
    compose_env_cfg_modifiers,
    disable_terminations_injector,
    make_contact_history_injector,
    make_height_scanner_injector,
    make_joint_reset_injector,
)


def _config(decimation: int = 40, history_length: int = 3) -> Any:
    """A stand-in for a parsed task config, with the attributes the injectors touch."""
    return SimpleNamespace(
        decimation=decimation,
        scene=SimpleNamespace(contact_forces=SimpleNamespace(history_length=history_length)),
        terminations=SimpleNamespace(time_out=object(), base_contact=object()),
        events=SimpleNamespace(reset_base=object()),
    )


class _FakeCfg:
    """A configclass stand-in recording whatever keyword arguments it was built with."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeTargetCfg(_FakeCfg):
    pass


class _FakeMultiMeshCfg(_FakeCfg):
    RaycastTargetCfg = _FakeTargetCfg


class _FakeRayCasterCfg(_FakeCfg):
    OffsetCfg = SimpleNamespace


@pytest.fixture(name="fake_isaaclab")
def fixture_fake_isaaclab() -> Iterator[Dict[str, Any]]:
    """Install a minimal fake ``isaaclab`` for the duration of one test."""
    built: Dict[str, Any] = {}

    def _event_term(**kwargs: Any) -> Any:
        built["event"] = kwargs
        return kwargs

    # ``import isaaclab.envs.mdp`` walks the attribute chain from the package down, so the parent
    # namespaces have to carry their children as attributes and not only sit in sys.modules.
    mdp = SimpleNamespace(reset_joints_by_offset="reset_joints_by_offset")
    envs = SimpleNamespace(mdp=mdp)
    sensors = SimpleNamespace(
        MultiMeshRayCasterCfg=_FakeMultiMeshCfg,
        RayCasterCfg=_FakeRayCasterCfg,
        patterns=SimpleNamespace(GridPatternCfg=_FakeCfg),
    )
    managers = SimpleNamespace(EventTermCfg=_event_term, SceneEntityCfg=str)
    modules: Dict[str, Any] = {
        "isaaclab": SimpleNamespace(envs=envs, sensors=sensors, managers=managers),
        "isaaclab.sensors": sensors,
        "isaaclab.envs": envs,
        "isaaclab.envs.mdp": mdp,
        "isaaclab.managers": managers,
    }
    saved: Dict[str, Any] = {name: sys.modules.get(name) for name in modules}
    for name, module in modules.items():
        sys.modules[name] = module
    try:
        yield built
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_compose_runs_modifiers_in_order_and_skips_none() -> None:
    """Order matters: a later modifier may overwrite what an earlier one set.

    Purpose: Validates composition order and the None passthrough

    Given: Three modifiers, the middle one None, each appending its name
    When: The composed modifier is applied
    Then: The two real ones ran in the order given

    Test type: unit
    """
    calls: List[str] = []
    composed = compose_env_cfg_modifiers(
        lambda cfg: calls.append("first"), None, lambda cfg: calls.append("second")
    )
    composed(_config())
    assert calls == ["first", "second"]


def test_disable_terminations_clears_only_the_named_terms() -> None:
    """Clearing the wrong term would silently change when episodes end.

    Purpose: Validates selective termination-term removal

    Given: A config with time_out and base_contact terms
    When: Only base_contact is disabled
    Then: base_contact is None and time_out is untouched

    Test type: unit
    """
    cfg = _config()
    time_out = cfg.terminations.time_out
    disable_terminations_injector("base_contact")(cfg)
    assert cfg.terminations.base_contact is None
    assert cfg.terminations.time_out is time_out


def test_disable_terminations_ignores_a_term_the_task_does_not_define() -> None:
    """One injector serves several tasks, which do not all define the same terms.

    Purpose: Validates that an absent term is skipped rather than raising

    Given: A config with no such term
    When: That term is disabled
    Then: Nothing is raised and no attribute is created

    Test type: unit
    """
    cfg = _config()
    disable_terminations_injector("no_such_term")(cfg)
    assert not hasattr(cfg.terminations, "no_such_term")


def test_contact_history_defaults_to_the_task_decimation() -> None:
    """A history shorter than the control step drops the impact transient entirely.

    Purpose: Validates that the history is widened to one sample per physics substep

    Given: A config with decimation 40 and a sensor keeping 3 samples
    When: The history injector is applied with no explicit length
    Then: The sensor keeps 40 samples

    Test type: unit
    """
    cfg = _config(decimation=40, history_length=3)
    make_contact_history_injector()(cfg)
    assert cfg.scene.contact_forces.history_length == 40


def test_contact_history_honours_an_explicit_length() -> None:
    """A task whose decimation is not the sampling rate needs the length set directly.

    Purpose: Validates the explicit history-length override

    Given: A config with decimation 40
    When: The injector is applied with history_length 8
    Then: The sensor keeps 8 samples

    Test type: unit
    """
    cfg = _config(decimation=40)
    make_contact_history_injector(history_length=8)(cfg)
    assert cfg.scene.contact_forces.history_length == 8


def test_contact_history_leaves_a_task_without_the_sensor_alone() -> None:
    """Most manipulation tasks ship no contact sensor; that is not an error.

    Purpose: Validates that a missing sensor is skipped

    Given: A config whose scene has no contact sensor
    When: The history injector is applied
    Then: Nothing is raised and no sensor is created

    Test type: unit
    """
    cfg = _config()
    del cfg.scene.contact_forces
    make_contact_history_injector()(cfg)
    assert not hasattr(cfg.scene, "contact_forces")


def test_joint_reset_adds_a_reset_event_for_the_named_asset(fake_isaaclab) -> None:
    """Without it a second episode starts with the first one's joint angles and falls over.

    Purpose: Validates that a default-joint reset term is registered

    Given: A config whose events reset only the root pose
    When: The joint-reset injector is applied
    Then: A reset-mode term for the named asset is added, with zero randomisation

    Test type: unit
    """
    cfg = _config()
    make_joint_reset_injector(asset="robot")(cfg)
    term = fake_isaaclab["event"]
    assert hasattr(cfg.events, "reset_default_joints")
    assert term["mode"] == "reset"
    assert term["params"]["position_range"] == (0.0, 0.0)
    assert term["params"]["asset_cfg"] == "robot"


def test_height_scanner_uses_the_plain_ray_caster_for_a_single_static_mesh(
    fake_isaaclab,
) -> None:
    """The stock sensor is cheaper, and one static mesh is all it can do.

    Purpose: Validates the single-mesh path

    Given: One static mesh path and no tracked paths
    When: The scanner injector is applied
    Then: A plain RayCasterCfg carrying that path is registered

    Test type: unit
    """
    del fake_isaaclab
    cfg = _config()
    make_height_scanner_injector(mesh_prim_paths=["/World/ground"])(cfg)
    sensor = cfg.scene.height_scanner
    assert isinstance(sensor, _FakeRayCasterCfg)
    assert sensor.kwargs["mesh_prim_paths"] == ["/World/ground"]


def test_height_scanner_tracks_only_the_paths_that_can_move(fake_isaaclab) -> None:
    """Tracking the ground raises at init; not tracking an obstacle freezes it at spawn.

    Purpose: Validates that transform tracking is set per target

    Given: A static ground path and one movable obstacle path
    When: The scanner injector is applied
    Then: A multi-mesh sensor is registered whose ground target is untracked and whose
        obstacle target is tracked

    Test type: unit
    """
    del fake_isaaclab
    cfg = _config()
    make_height_scanner_injector(
        mesh_prim_paths=["/World/ground"], tracked_mesh_prim_paths=["/World/envs/env_.*/Post"]
    )(cfg)
    targets = cfg.scene.height_scanner.kwargs["mesh_prim_paths"]
    assert [target.kwargs["prim_expr"] for target in targets] == [
        "/World/ground",
        "/World/envs/env_.*/Post",
    ]
    assert [target.kwargs["track_mesh_transforms"] for target in targets] == [False, True]


def test_height_scanner_rejects_an_empty_target_list() -> None:
    """A scanner with nothing to cast against reads as flat ground everywhere.

    Purpose: Validates construction-time rejection of an empty mesh set

    Given: No static and no tracked mesh paths
    When: The scanner injector is built
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match="at least one mesh prim path"):
        make_height_scanner_injector(mesh_prim_paths=[], tracked_mesh_prim_paths=[])
