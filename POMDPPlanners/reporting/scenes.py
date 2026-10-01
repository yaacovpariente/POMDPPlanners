# SPDX-License-Identifier: MIT

"""Where the three.js scene scripts live.

Each environment keeps the scene that draws its traces next to its trace
visualizer, as ``<scene name>.scene.js`` — ``battleship.scene.js`` draws
``battleship.v1``. The site and the Sphinx docs both serve every scene under
one flat URL, ``<static root>/scenes/<scene name>.js``, and this module maps
that name back to the file inside the environment's folder.

The map is built by finding the files rather than by importing the
environments, because importing them pulls in every simulator dependency and
the site must start without any of them.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from POMDPPlanners.core.simulation.episode_visualizers import SCENE_SUFFIX, scene_name

ENVIRONMENTS_ROOT = Path(__file__).resolve().parent.parent / "environments"


@lru_cache(maxsize=1)
def scene_scripts() -> Dict[str, Path]:
    """Every scene script under ``environments/``, keyed by scene name.

    Returns:
        Scene name to absolute path.

    Raises:
        RuntimeError: If two environments ship a scene with the same name,
            since the site could not tell which one a trace means.
    """
    scripts: Dict[str, Path] = {}
    for path in sorted(ENVIRONMENTS_ROOT.rglob(f"*{SCENE_SUFFIX}")):
        name = path.name[: -len(SCENE_SUFFIX)]
        if name in scripts:
            raise RuntimeError(f"Two scene scripts are named {name!r}: {scripts[name]} and {path}")
        scripts[name] = path
    return scripts


def scene_script_file(name: str) -> Optional[Path]:
    """The scene script named ``name``, or ``None`` when there is none.

    Args:
        name: A scene name, such as ``"battleship"``.

    Returns:
        The script's path.
    """
    return scene_scripts().get(name)


def scene_script_url(payload_kind: str, static_root: str = "/static") -> str:
    """URL of the scene script that draws ``payload_kind``.

    Args:
        payload_kind: A trace payload kind, such as ``"light_dark.v1"``.
        static_root: Where the viewer is served from: ``/static`` on the
            results site, a relative path in the Sphinx docs.

    Returns:
        ``<static_root>/scenes/<scene name>.js``.
    """
    return f"{static_root}/scenes/{scene_name(payload_kind)}.js"
