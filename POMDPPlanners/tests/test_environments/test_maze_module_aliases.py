# SPDX-License-Identifier: MIT

"""The pre-rename ``t_maze_pomdp`` import paths must keep working.

The Maze family moved from ``POMDPPlanners.environments.t_maze_pomdp`` to
``POMDPPlanners.environments.maze_pomdp``. Nothing in the library reads the old
paths any more, so without these tests the shims would rot unnoticed — and they
are not decoration: :meth:`Environment.from_dict` imports the module string
recorded in a saved configuration, so a config written before the move names a
``t_maze_pomdp`` module and would fail to load if that module disappeared.
"""

import importlib

import pytest

from POMDPPlanners.core.environment.environment import Environment
from POMDPPlanners.environments.maze_pomdp import (
    ContinuousMazePOMDP,
    DiscreteMazePOMDP,
    MazeVisualizer,
    TMazePOMDP,
)

LEGACY_PACKAGE = "POMDPPlanners.environments.t_maze_pomdp"


@pytest.mark.parametrize(
    "legacy_module, name, expected",
    [
        (LEGACY_PACKAGE, "TMazePOMDP", TMazePOMDP),
        (LEGACY_PACKAGE, "DiscreteMazePOMDP", DiscreteMazePOMDP),
        (LEGACY_PACKAGE, "ContinuousMazePOMDP", ContinuousMazePOMDP),
        (LEGACY_PACKAGE, "MazeVisualizer", MazeVisualizer),
        (f"{LEGACY_PACKAGE}.maze_pomdp", "DiscreteMazePOMDP", DiscreteMazePOMDP),
        (f"{LEGACY_PACKAGE}.maze_pomdp", "ContinuousMazePOMDP", ContinuousMazePOMDP),
        (f"{LEGACY_PACKAGE}.t_maze_pomdp", "TMazePOMDP", TMazePOMDP),
        (f"{LEGACY_PACKAGE}.maze_visualizer", "MazeVisualizer", MazeVisualizer),
        (f"{LEGACY_PACKAGE}.t_maze_visualizer", "MazeVisualizer", MazeVisualizer),
        # TMazeVisualizer was the renderer's name before the move. It is the same
        # class, not a T-only subclass.
        (f"{LEGACY_PACKAGE}.t_maze_visualizer", "TMazeVisualizer", MazeVisualizer),
    ],
)
def test_legacy_path_resolves_to_the_same_object(legacy_module, name, expected):
    """An old import path returns the moved object itself, not a copy of it."""
    assert getattr(importlib.import_module(legacy_module), name) is expected


def test_legacy_maze_geometry_module_still_imports():
    """``maze_geometry`` moved too, and callers imported ``Cell`` from it directly."""
    from POMDPPlanners.environments.maze_pomdp.maze_geometry import Cell, MazeGeometry

    legacy = importlib.import_module(f"{LEGACY_PACKAGE}.maze_geometry")
    assert legacy.MazeGeometry is MazeGeometry
    assert legacy.Cell is Cell


@pytest.mark.parametrize(
    "env, legacy_module",
    [
        (DiscreteMazePOMDP(), f"{LEGACY_PACKAGE}.maze_pomdp"),
        (ContinuousMazePOMDP(), f"{LEGACY_PACKAGE}.maze_pomdp"),
        (TMazePOMDP(), f"{LEGACY_PACKAGE}.t_maze_pomdp"),
    ],
)
def test_config_saved_before_the_rename_still_loads(env, legacy_module):
    """A saved config names the old module; ``from_dict`` must still import it."""
    data = env.to_dict()
    class_name = type(env).__name__
    data["module"] = legacy_module
    data["class"] = f"{legacy_module}.{class_name}"

    restored = Environment.from_dict(data)

    assert type(restored) is type(env)
    # The rename must not move any cached result: config_id is built from the
    # instance's attributes, never from its module path.
    assert restored.config_id == env.config_id


@pytest.mark.parametrize("env_class", [DiscreteMazePOMDP, ContinuousMazePOMDP, TMazePOMDP])
def test_new_configs_record_the_maze_module_path(env_class):
    """Freshly saved configs point at the new package, so the shims stop growing."""
    assert env_class().to_dict()["module"].startswith("POMDPPlanners.environments.maze_pomdp")


@pytest.mark.parametrize(
    "submodule",
    ["maze_geometry", "maze_pomdp", "maze_visualizer", "t_maze_pomdp", "t_maze_visualizer"],
)
def test_legacy_submodules_are_reachable_by_attribute(submodule):
    """``import t_maze_pomdp`` then ``t_maze_pomdp.maze_pomdp`` worked before the move.

    A package only binds a submodule as an attribute once something imports it, and
    the shim package imports from ``maze_pomdp`` rather than from its own submodules,
    so this would silently stop working without an explicit import.
    """
    package = importlib.import_module(LEGACY_PACKAGE)
    assert hasattr(package, submodule)


def test_legacy_maze_pomdp_still_exposes_the_geometry_names():
    """The pre-move module imported ``Cell`` and ``MazeGeometry``, so both leaked into it."""
    legacy = importlib.import_module(f"{LEGACY_PACKAGE}.maze_pomdp")
    from POMDPPlanners.environments.maze_pomdp.maze_geometry import Cell, MazeGeometry

    assert legacy.MazeGeometry is MazeGeometry
    assert legacy.Cell is Cell
