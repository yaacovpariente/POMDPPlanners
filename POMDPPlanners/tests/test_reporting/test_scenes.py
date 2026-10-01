# SPDX-License-Identifier: MIT

"""Every trace visualizer has exactly one scene script, and every scene has a visualizer.

The Python payload kind, the scene's file name and the key the script
registers itself under are three spellings of one name, and nothing at run
time checks they agree: a mismatch shows up as "No viewer for payload kind"
on someone's results page. So they are checked here, for every environment.
"""

import importlib
import re
from typing import Dict, List, Type

import pytest

from POMDPPlanners.core.simulation.episode_visualizers import TraceVisualizer, scene_name
from POMDPPlanners.reporting.scenes import (
    ENVIRONMENTS_ROOT,
    scene_script_file,
    scene_script_url,
    scene_scripts,
)

_REGISTRATION = re.compile(r"""V\.scenes\[\s*["']([^"']+)["']\s*\]\s*=""")


def _all_trace_visualizers() -> List[Type[TraceVisualizer]]:
    """Import every module in an environment's visualization package and collect the visualizers."""
    for path in sorted(ENVIRONMENTS_ROOT.glob("*/*_visualization/*.py")):
        relative = path.relative_to(ENVIRONMENTS_ROOT.parent.parent).with_suffix("")
        importlib.import_module(".".join(relative.parts))

    found: List[Type[TraceVisualizer]] = []
    pending = list(TraceVisualizer.__subclasses__())
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        if getattr(cls, "payload_kind", None) and cls.__module__.startswith(
            "POMDPPlanners.environments."
        ):
            found.append(cls)
    return found


VISUALIZERS = _all_trace_visualizers()


def test_every_environment_with_a_scene_is_found():
    """The collection above is not silently empty."""
    assert len(VISUALIZERS) >= 18


@pytest.mark.parametrize("visualizer", VISUALIZERS, ids=lambda cls: cls.__name__)
def test_visualizer_has_its_scene_script(visualizer: Type[TraceVisualizer]):
    """The scene that draws the visualizer's traces sits next to it and registers its kind.

    Given: A trace visualizer from an environment package.
    When: Its scene script is looked up beside it and through the site's lookup.
    Then: Both find the same file, and the file registers the visualizer's
        exact payload kind.
    """
    script = visualizer.scene_script()
    assert script.is_file(), f"{visualizer.__name__} has no scene script at {script}"
    assert scene_script_file(scene_name(visualizer.payload_kind)) == script
    registered = _REGISTRATION.findall(script.read_text(encoding="utf-8"))
    assert registered == [visualizer.payload_kind]


def test_every_scene_script_has_a_visualizer():
    """A scene no visualizer writes traces for is dead code the site would still serve."""
    drawn: Dict[str, str] = {scene_name(cls.payload_kind): cls.__name__ for cls in VISUALIZERS}
    assert sorted(scene_scripts()) == sorted(drawn)


def test_scene_url_is_flat_and_versionless():
    """The site and the docs serve every scene under one directory, by scene name."""
    assert scene_script_url("light_dark.v1") == "/static/scenes/light_dark.js"
    assert scene_script_url("t_maze.v1", static_root="..") == "../scenes/t_maze.js"
