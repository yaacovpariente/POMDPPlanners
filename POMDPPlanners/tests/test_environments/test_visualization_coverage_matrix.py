# SPDX-License-Identifier: MIT

"""Every environment family must stay visualizable, documented and illustrated.

An environment page is illustrated in one of two ways. A family whose traces
have a 3D scene module (``POMDPPlanners/reporting/static/viewer/scenes``)
embeds a live replay with ``.. episode-viewer:: traces/<world>.json``, and may
no longer show a recorded GIF in its place. A family with no scene module
(CARLA, Isaac Lab) keeps a still image. The golden GIFs stay in the test tree
either way; they pin the package renderer, not the docs.

Three things have to line up before a reader can see what an environment does:
the package has to expose a visualization hook, the docs have to describe the
family, and the image the docs point a reader at has to exist. Each of the three
has been added by hand, per environment, which is exactly the kind of work that
gets skipped when the next environment lands. Nothing else in the suite notices:
a family with no ``cache_visualization`` still passes the API conformance tests,
and a docs page that was never written still builds.

The matrix below is therefore written out in full rather than derived. Adding an
environment means adding a row and being told what is missing; exempting one
means writing the exemption down with its reason, where a reviewer sees it.

The three checks are paired both ways, because the first version of this file
checked each of them in isolation and a page could drift from the row that was
supposed to cover it. A declared image must exist *and* be embedded by the page
that claims it; a page may not embed an image no row declares. The second half
is what keeps the matrix honest as families grow: a continuous variant added to
an existing page arrives with its own GIF, and the page now fails until a row
takes responsibility for it.

An image also has to decode and, when it is a GIF, carry more than one frame.
Existence and a non-zero size were the whole check once, which let a truncated
file pass while the page rendered a broken image.

What it still does not check, so nobody reads more into a green run than is
there: pages no family row names (``index.rst``, ``custom.rst``) are outside the
undeclared-image check, and an image is matched against its whole page rather
than against the family's section, so moving the Racetrack GIF into the CARLA
section of ``realistic.rst`` would go unnoticed.

Checks are source-level: modules are parsed with :mod:`ast`, docs are read as
text, and images are opened only to decode their headers. Nothing here imports
an environment, so the families that need CARLA, Isaac Sim or ``highway-env``
are checked on a machine that has none of them, and nothing here needs a built
``docs/_build`` tree.
"""

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest
from PIL import Image

from POMDPPlanners.reporting.pages import scene_script_path

# tests/test_environments/<this file> -> tests/ -> POMDPPlanners/ -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENTS_DIR = REPO_ROOT / "POMDPPlanners" / "environments"
DOCS_ROOT = REPO_ROOT / "docs"
DOCS_ENVIRONMENTS_DIR = DOCS_ROOT / "environments"

# The reStructuredText characters used to underline a section heading in docs/.
_HEADING_UNDERLINE_CHARACTERS = set("=-~^\"'`#*+_:.")

# Frames a recorded episode must have before it counts as an animation. A GIF
# written from a single frame is what a stalled or misconfigured recorder
# produces, and it looks like a still on the page.
MIN_ANIMATION_FRAMES = 2

# ``.. image:: path``, ``.. figure:: path`` and the substitution form
# ``.. |name| image:: path``. Only these render an image; a ``:download:`` link
# names a file without showing it, so it is deliberately not matched. The target
# is optional here because reST also allows it on the following line.
_IMAGE_DIRECTIVE = re.compile(r"^(\s*)\.\.\s+(?:\|[^|]+\|\s+)?(?:image|figure)::\s*(\S*)\s*$")

# ``.. episode-viewer:: traces/<world>.json`` -- the docs extension in
# docs/_ext/episode_viewer.py that embeds a live 3D replay of that trace.
_VIEWER_DIRECTIVE = re.compile(r"^\s*\.\.\s+episode-viewer::\s*(\S+)\s*$")

# Module-level ``SOMETHING_PAYLOAD_KIND = "kind.v1"`` in a family's trace
# exporter: the payload kinds the family writes.
_PAYLOAD_KIND_NAME = re.compile(r"^[A-Z0-9_]*PAYLOAD_KIND$")


@dataclass(frozen=True)
class EnvironmentFamily:
    """One environment family and the three things it has to keep.

    Attributes:
        package: Directory or module name under ``POMDPPlanners/environments``.
            This is the key discovery matches against, so a new directory with
            no row here fails ``test_matrix_lists_every_environment_family``.
        label: Human name used in failure messages.
        hooks: ``(module path, class name)`` pairs, relative to
            ``POMDPPlanners/environments``. Every pair must define
            ``cache_visualization`` -- the method the simulation layer calls to
            write an episode's visualization. A family with several concrete
            worlds lists each class that owns its own rendering.
        docs_page: File name under ``docs/environments``.
        docs_section: Section heading inside that page, or ``None`` when the
            whole page belongs to this family.
        images: Repository-relative paths to every still image that illustrates
            the family. Only a family with no 3D scene module may declare one.
        viewers: Repository-relative paths to the trace JSON each
            ``.. episode-viewer::`` on the page replays. A family with several
            concrete worlds lists one per world, because a single declared
            illustration left the others -- the continuous Push, LaserTag and
            Maze worlds among them -- embedded by the docs and checked by
            nothing.
    """

    package: str
    label: str
    hooks: Tuple[Tuple[str, str], ...]
    docs_page: str
    docs_section: Optional[str]
    images: Tuple[str, ...] = ()
    viewers: Tuple[str, ...] = ()


# Families with no visualization at all, and why that is a decision rather than
# a gap. Both are checked below: an exempt family that grows a hook should lose
# its exemption, so the test refuses an exemption that is no longer true.
EXEMPT_FAMILIES: Dict[str, str] = {
    "sanity_pomdp": (
        "SanityPOMDP is a two-state debugging baseline with nothing spatial to "
        "draw; its docs entry exists to say so."
    ),
    "nuplan_pomdp": (
        "NuPlanPOMDP has no rendering path in this package -- playback belongs "
        "to nuboard in nuplan-devkit, which is not a dependency here."
    ),
}

# Families that have a 3D scene module but cannot have a docs trace yet, and
# why. Each keeps its still image until the block lifts. The test below refuses
# an entry once its trace is committed, so the exception cannot outlive its
# reason.
VIEWER_BLOCKED: Dict[str, str] = {
    "racetrack_pomdp": (
        "No real episode can be run for a trace yet: Policy.config_id recurses "
        "forever on every Racetrack planner model (model._observation_model._road "
        "points back at the model), so LocalSimulationsAPI fails while building "
        "the cache key. Regenerate with scripts/generate_docs_traces.py --only "
        "racetrack once that is fixed."
    ),
}

# Entries under POMDPPlanners/environments that are not environment families.
NON_FAMILY_ENTRIES: Dict[str, str] = {
    "__init__.py": "The package's own registry module.",
    "__pycache__": "Bytecode cache.",
    "environment_utils": "Shared helpers, not a world.",
    "t_maze_pomdp": "Import shim kept for configurations saved before the Maze rename.",
}

FAMILIES: Tuple[EnvironmentFamily, ...] = (
    EnvironmentFamily(
        package="battleship_pomdp",
        label="Battleship",
        hooks=(("battleship_pomdp/battleship_pomdp.py", "BattleshipPOMDP"),),
        docs_page="battleship.rst",
        docs_section=None,
        viewers=("docs/environments/traces/battleship.json",),
    ),
    EnvironmentFamily(
        package="capture_the_flag_pomdp",
        label="CaptureTheFlag",
        hooks=(("capture_the_flag_pomdp/capture_the_flag_pomdp.py", "CaptureTheFlagPOMDP"),),
        docs_page="capture_the_flag.rst",
        docs_section=None,
        viewers=("docs/environments/traces/capture_the_flag.json",),
    ),
    EnvironmentFamily(
        package="carla_pomdp",
        label="CARLA",
        hooks=(("carla_pomdp/carla_pomdp.py", "CarlaPOMDP"),),
        docs_page="realistic.rst",
        docs_section="CARLA",
        images=("docs/images/carla_chase_camera.png",),
    ),
    EnvironmentFamily(
        package="cartpole_pomdp",
        label="CartPole",
        hooks=(("cartpole_pomdp/cartpole_pomdp.py", "CartPolePOMDP"),),
        docs_page="cartpole.rst",
        docs_section=None,
        viewers=("docs/environments/traces/cartpole.json",),
    ),
    EnvironmentFamily(
        package="chicheck_invaders_pomdp",
        label="ChicheckInvaders",
        hooks=(("chicheck_invaders_pomdp/chicheck_invaders_pomdp.py", "ChicheckInvadersPOMDP"),),
        docs_page="chicheck_invaders.rst",
        docs_section=None,
        viewers=("docs/environments/traces/chicheck_invaders.json",),
    ),
    EnvironmentFamily(
        package="isaac_lab_pomdp",
        label="IsaacLab",
        hooks=(("isaac_lab_pomdp/isaac_lab_pomdp.py", "IsaacLabPOMDP"),),
        docs_page="realistic.rst",
        docs_section="Isaac Lab",
        images=("docs/images/isaac_lab_franka_reach.png",),
    ),
    EnvironmentFamily(
        package="laser_tag_pomdp",
        label="LaserTag",
        hooks=(
            ("laser_tag_pomdp/laser_tag_pomdp.py", "LaserTagPOMDP"),
            ("laser_tag_pomdp/continuous_laser_tag_pomdp.py", "ContinuousLaserTagPOMDP"),
        ),
        docs_page="laser_tag.rst",
        docs_section=None,
        viewers=(
            "docs/environments/traces/laser_tag.json",
            "docs/environments/traces/continuous_laser_tag.json",
        ),
    ),
    EnvironmentFamily(
        package="light_dark_pomdp",
        label="Light-Dark",
        # The discrete and continuous worlds share one renderer on their base.
        hooks=(
            (
                "light_dark_pomdp/light_dark_pomdp_utils/base_light_dark_pomdp.py",
                "BaseLightDarkPOMDP",
            ),
        ),
        docs_page="light_dark.rst",
        docs_section=None,
        viewers=("docs/environments/traces/light_dark.json",),
    ),
    EnvironmentFamily(
        package="maze_pomdp",
        label="Maze",
        hooks=(
            ("maze_pomdp/maze_pomdp.py", "BaseMazePOMDP"),
            ("maze_pomdp/t_maze_pomdp.py", "TMazePOMDP"),
        ),
        docs_page="maze.rst",
        docs_section=None,
        viewers=(
            "docs/environments/traces/discrete_maze.json",
            "docs/environments/traces/continuous_maze.json",
            "docs/environments/traces/t_maze.json",
        ),
    ),
    EnvironmentFamily(
        package="mountain_car_pomdp",
        label="MountainCar",
        hooks=(("mountain_car_pomdp/mountain_car_pomdp.py", "MountainCarPOMDP"),),
        docs_page="mountain_car.rst",
        docs_section=None,
        viewers=("docs/environments/traces/mountain_car.json",),
    ),
    EnvironmentFamily(
        package="multiagent_firefighting_pomdp",
        label="MultiAgentFirefighting",
        hooks=(
            (
                "multiagent_firefighting_pomdp/multiagent_firefighting_pomdp.py",
                "MultiAgentFirefightingPOMDP",
            ),
        ),
        docs_page="multiagent_firefighting.rst",
        docs_section=None,
        viewers=("docs/environments/traces/multiagent_firefighting.json",),
    ),
    EnvironmentFamily(
        package="occupancy_grid_mapping_pomdp",
        label="OccupancyGridMapping",
        hooks=(
            (
                "occupancy_grid_mapping_pomdp/occupancy_grid_mapping_pomdp.py",
                "OccupancyGridMappingPOMDP",
            ),
        ),
        docs_page="occupancy_grid_mapping.rst",
        docs_section=None,
        viewers=("docs/environments/traces/occupancy_grid_mapping.json",),
    ),
    EnvironmentFamily(
        package="pacman_pomdp",
        label="PacMan",
        hooks=(("pacman_pomdp/pacman_pomdp.py", "PacManPOMDP"),),
        docs_page="pacman.rst",
        docs_section=None,
        viewers=("docs/environments/traces/pacman.json",),
    ),
    EnvironmentFamily(
        package="push_pomdp",
        label="Push",
        hooks=(
            ("push_pomdp/push_pomdp.py", "PushPOMDP"),
            ("push_pomdp/continuous_push_pomdp.py", "ContinuousPushPOMDP"),
        ),
        docs_page="push.rst",
        docs_section=None,
        viewers=(
            "docs/environments/traces/push.json",
            "docs/environments/traces/continuous_push.json",
        ),
    ),
    EnvironmentFamily(
        package="racetrack_pomdp",
        label="Racetrack",
        hooks=(("racetrack_pomdp/racetrack_pomdp.py", "RacetrackPOMDP"),),
        docs_page="realistic.rst",
        docs_section="Racetrack",
        images=("docs/images/racetrack_recorded_episode.gif",),
    ),
    EnvironmentFamily(
        package="rock_sample_pomdp",
        label="RockSample",
        hooks=(("rock_sample_pomdp/rock_sample_pomdp.py", "RockSamplePOMDP"),),
        docs_page="rock_sample.rst",
        docs_section=None,
        viewers=("docs/environments/traces/rock_sample.json",),
    ),
    EnvironmentFamily(
        package="safety_ant_velocity_pomdp",
        label="SafeAntVelocity",
        hooks=(
            (
                "safety_ant_velocity_pomdp/safety_ant_velocity_pomdp.py",
                "SafeAntVelocityPOMDP",
            ),
        ),
        docs_page="safety_ant_velocity.rst",
        docs_section=None,
        viewers=("docs/environments/traces/safety_ant_velocity.json",),
    ),
    EnvironmentFamily(
        package="snake_pomdp",
        label="Snake",
        hooks=(("snake_pomdp/snake_pomdp.py", "SnakePOMDP"),),
        docs_page="snake.rst",
        docs_section=None,
        viewers=("docs/environments/traces/snake.json",),
    ),
    EnvironmentFamily(
        package="tiger_pomdp",
        label="Tiger",
        hooks=(("tiger_pomdp/tiger_pomdp.py", "TigerPOMDP"),),
        docs_page="tiger.rst",
        docs_section=None,
        viewers=("docs/environments/traces/tiger.json",),
    ),
)

FAMILIES_BY_PACKAGE: Dict[str, EnvironmentFamily] = {family.package: family for family in FAMILIES}


def _class_defines_cache_visualization(source_path: Path, class_name: str) -> Optional[bool]:
    """Report whether ``class_name`` defines ``cache_visualization`` in this file.

    Args:
        source_path: Python module to parse.
        class_name: Top-level class expected in it.

    Returns:
        ``True`` or ``False`` when the class is found, ``None`` when the module
        has no such top-level class.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return any(
                isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                and member.name == "cache_visualization"
                for member in node.body
            )
    return None


def find_missing_hooks(family: EnvironmentFamily) -> List[str]:
    """List the family's declared hooks that are not there."""
    problems: List[str] = []
    for module_name, class_name in family.hooks:
        source_path = ENVIRONMENTS_DIR / module_name
        location = f"POMDPPlanners/environments/{module_name}"
        if not source_path.is_file():
            problems.append(
                f"{family.label}: package visualization hook missing -- "
                f"{location} does not exist, so {class_name}.cache_visualization "
                f"cannot be checked."
            )
            continue
        defines = _class_defines_cache_visualization(source_path, class_name)
        if defines is None:
            problems.append(
                f"{family.label}: package visualization hook missing -- "
                f"{location} defines no class {class_name}."
            )
        elif not defines:
            problems.append(
                f"{family.label}: package visualization hook missing -- "
                f"{class_name} in {location} defines no cache_visualization method."
            )
    return problems


def _directive_targets(text: str) -> List[str]:
    """Every image directive target in ``text``, including next-line targets.

    reST lets the argument sit on the line after the directive, indented under
    it. Reading only the directive line would silently miss such an image, which
    is the failure mode this whole file exists to prevent.
    """
    lines = text.splitlines()
    targets: List[str] = []
    for index, line in enumerate(lines):
        match = _IMAGE_DIRECTIVE.match(line)
        if not match:
            continue
        indent, target = match.group(1), match.group(2)
        if target:
            targets.append(target)
            continue
        for following in lines[index + 1 :]:
            if not following.strip():
                break
            stripped = following.strip()
            # Option lines (``:width: 480px``) come before the argument only if
            # the argument was on the directive line, which it was not here.
            if stripped.startswith(":") or len(following) - len(following.lstrip()) <= len(indent):
                break
            targets.append(stripped.split()[0])
            break
    return targets


def referenced_images(page_path: Path) -> Set[str]:
    """Repository-relative paths of every image a docs page embeds.

    A target is normally relative to the page. Sphinx also accepts a target
    beginning with ``/``, which means the documentation source root rather than
    the filesystem root; resolving that against the page's directory would
    produce an absolute path outside the repository and drop the image without
    saying so, so it is resolved against ``docs/`` instead. A target that still
    lands outside the repository is dropped, since no matrix row can name it.
    """
    if not page_path.is_file():
        return set()
    images: Set[str] = set()
    for target in _directive_targets(page_path.read_text(encoding="utf-8")):
        if target.startswith("/"):
            # Read through REPO_ROOT rather than DOCS_ROOT so that a test which
            # relocates the repository root relocates the source root with it.
            resolved = (REPO_ROOT / "docs" / target.lstrip("/")).resolve()
        else:
            resolved = (page_path.parent / target).resolve()
        try:
            images.add(resolved.relative_to(REPO_ROOT).as_posix())
        except ValueError:
            continue
    return images


def _has_section_heading(text: str, heading: str) -> bool:
    """Report whether reStructuredText ``text`` has ``heading`` as a section title."""
    lines = text.splitlines()
    for index, line in enumerate(lines[:-1]):
        if line.strip() != heading:
            continue
        underline = lines[index + 1].strip()
        if (
            len(underline) >= len(heading)
            and len(set(underline)) == 1
            and underline[0] in _HEADING_UNDERLINE_CHARACTERS
        ):
            return True
    return False


def find_missing_docs(family: EnvironmentFamily) -> List[str]:
    """List what the family's documentation is missing."""
    page_path = DOCS_ENVIRONMENTS_DIR / family.docs_page
    location = f"docs/environments/{family.docs_page}"
    if not page_path.is_file():
        return [f"{family.label}: documentation missing -- {location} does not exist."]
    text = page_path.read_text(encoding="utf-8")
    if family.docs_section is None:
        if not text.strip():
            return [f"{family.label}: documentation missing -- {location} is empty."]
        return []
    if not _has_section_heading(text, family.docs_section):
        return [
            f"{family.label}: documentation missing -- {location} has no "
            f'"{family.docs_section}" section.'
        ]
    return []


def find_missing_image(family: EnvironmentFamily) -> List[str]:
    """List the family's image problems, for every image the family declares.

    A zero-byte check was the whole test here once, which let a truncated or
    corrupt file count as coverage: the page renders a broken image and the
    suite stays green. Each declared image now has to decode, and an animation
    has to carry more than the one frame a stalled recorder writes.
    """
    problems: List[str] = []
    for image in family.images:
        image_path = REPO_ROOT / image
        if not image_path.is_file():
            problems.append(
                f"{family.label}: visualization image missing -- {image} does not exist."
            )
            continue
        if image_path.stat().st_size == 0:
            problems.append(f"{family.label}: visualization image missing -- {image} is empty.")
            continue
        try:
            with Image.open(image_path) as handle:
                handle.verify()
            with Image.open(image_path) as handle:
                frames = getattr(handle, "n_frames", 1)
                # verify() checks the header, not the pixel data, so a file
                # truncated part way through still passes it. Seeking to the
                # last frame and loading it forces every frame to be decoded,
                # which is what actually catches a half-written GIF.
                handle.seek(frames - 1)
                handle.load()
        except (OSError, ValueError, EOFError) as error:
            problems.append(
                f"{family.label}: visualization image unreadable -- {image} does not "
                f"decode as an image ({error})."
            )
            continue
        if image_path.suffix.lower() == ".gif" and frames < MIN_ANIMATION_FRAMES:
            problems.append(
                f"{family.label}: visualization image is not an animation -- {image} has "
                f"{frames} frame(s), fewer than the {MIN_ANIMATION_FRAMES} an episode needs."
            )
    return problems


def find_unreferenced_images(family: EnvironmentFamily) -> List[str]:
    """List declared images the family's docs page does not actually embed.

    Declaring an image and embedding one were separate facts before this: the
    matrix checked a path on disk while the page was free to reference a
    different file, or none at all. Stripping the ``.. image::`` directive out
    of a page used to leave the whole matrix green.
    """
    page_path = DOCS_ENVIRONMENTS_DIR / family.docs_page
    if not page_path.is_file():
        # find_missing_docs already reports the absent page; do not repeat it.
        return []
    referenced = referenced_images(page_path)
    return [
        f"{family.label}: documentation does not show its visualization -- "
        f"docs/environments/{family.docs_page} embeds no image directive for {image}."
        for image in family.images
        if image not in referenced
    ]


def find_undeclared_images(page_name: str) -> List[str]:
    """List images a docs page embeds that no family on that page declares.

    Without this the matrix only ever grows by hand: a second world added to an
    existing page brings its own GIF, no row mentions it, and deleting that GIF
    breaks the docs build while every test still passes. That is exactly how the
    continuous Push, LaserTag and Maze GIFs went uncovered.
    """
    page_path = DOCS_ENVIRONMENTS_DIR / page_name
    declared = {
        image for family in FAMILIES if family.docs_page == page_name for image in family.images
    }
    return [
        f"docs/environments/{page_name} embeds {image}, which no matrix row declares. "
        f"Add it to the images of the family that page documents."
        for image in sorted(referenced_images(page_path) - declared)
    ]


def _resolve_page_target(page_path: Path, target: str) -> Optional[str]:
    """Repository-relative path of a directive target, or ``None`` outside the repo."""
    if target.startswith("/"):
        resolved = (REPO_ROOT / "docs" / target.lstrip("/")).resolve()
    else:
        resolved = (page_path.parent / target).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return None


def referenced_viewers(page_path: Path) -> Set[str]:
    """Repository-relative paths of every trace a page replays in an episode viewer."""
    if not page_path.is_file():
        return set()
    traces: Set[str] = set()
    for line in page_path.read_text(encoding="utf-8").splitlines():
        match = _VIEWER_DIRECTIVE.match(line)
        if match:
            resolved = _resolve_page_target(page_path, match.group(1))
            if resolved is not None:
                traces.add(resolved)
    return traces


def scene_module(payload_kind: str) -> Path:
    """The scene module that draws ``payload_kind``, by the results site's own rule."""
    relative = scene_script_path(payload_kind, static_root="").lstrip("/")
    return REPO_ROOT / "POMDPPlanners" / "reporting" / "static" / relative


def family_payload_kinds(family: EnvironmentFamily) -> Set[str]:
    """Payload kinds the family's trace exporters declare, read from source.

    A kind is a module-level string assigned to a name ending in
    ``PAYLOAD_KIND``, which is how every exporter names its own.
    """
    root = ENVIRONMENTS_DIR / family.package
    sources = sorted(root.rglob("*.py")) if root.is_dir() else []
    kinds: Set[str] = set()
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                continue
            if not isinstance(node.value.value, str):
                continue
            if any(
                isinstance(t, ast.Name) and _PAYLOAD_KIND_NAME.match(t.id) for t in node.targets
            ):
                kinds.add(node.value.value)
    return kinds


def family_scene_kinds(family: EnvironmentFamily) -> Set[str]:
    """The family's payload kinds that a 3D scene module can draw."""
    return {kind for kind in family_payload_kinds(family) if scene_module(kind).is_file()}


def _trace_payload_kind(trace: str) -> Tuple[Optional[str], Optional[str]]:
    """``(payload kind, problem)`` for a declared trace file."""
    path = REPO_ROOT / trace
    if not path.is_file():
        return None, f"{trace} does not exist"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        return None, f"{trace} is not JSON ({error})"
    kind = data.get("payload_kind") if isinstance(data, dict) else None
    if not isinstance(kind, str) or not kind:
        return None, f"{trace} has no payload_kind, so it is not an episode trace"
    steps = data.get("steps")
    if not isinstance(steps, list) or len(steps) < MIN_ANIMATION_FRAMES:
        return kind, (
            f"{trace} records {len(steps) if isinstance(steps, list) else 0} step(s), "
            f"fewer than the {MIN_ANIMATION_FRAMES} a replay needs"
        )
    return kind, None


def find_broken_viewers(family: EnvironmentFamily) -> List[str]:
    """List declared viewer traces that would not replay on the page.

    The docs build also refuses these, but a docs build is not part of the test
    suite; this catches the same faults from the source tree alone.
    """
    problems: List[str] = []
    for trace in family.viewers:
        kind, problem = _trace_payload_kind(trace)
        if problem:
            problems.append(f"{family.label}: episode viewer trace broken -- {problem}.")
            continue
        assert kind is not None
        if not scene_module(kind).is_file():
            problems.append(
                f"{family.label}: episode viewer trace broken -- {trace} has payload kind "
                f"{kind}, which no scene module draws ({scene_module(kind).name} is missing)."
            )
    return problems


def find_missing_viewers(family: EnvironmentFamily) -> List[str]:
    """List the family's 3D scenes that its docs page does not replay.

    A scene module exists so an episode of that world can be watched; a page
    that still shows a GIF, or nothing, when the scene exists is the gap this
    check closes.
    """
    if family.package in VIEWER_BLOCKED:
        return []
    declared = {_trace_payload_kind(trace)[0] for trace in family.viewers}
    return [
        f"{family.label}: has a 3D scene for {kind} ({scene_module(kind).name}) but "
        f"docs/environments/{family.docs_page} declares no episode viewer that replays it."
        for kind in sorted(family_scene_kinds(family) - declared)
    ]


def find_legacy_images(family: EnvironmentFamily) -> List[str]:
    """List still images declared by a family that has a 3D scene.

    Where a scene exists the page shows the live replay instead. A GIF left
    beside it would show a second, older picture of the same world.
    """
    if family.package in VIEWER_BLOCKED:
        return []
    if not family.images or not family_scene_kinds(family):
        return []
    return [
        f"{family.label}: has a 3D scene, so docs/environments/{family.docs_page} should "
        f"replay an episode instead of showing the legacy image {image}."
        for image in family.images
    ]


def find_unembedded_viewers(family: EnvironmentFamily) -> List[str]:
    """List declared viewer traces the family's page does not embed."""
    page_path = DOCS_ENVIRONMENTS_DIR / family.docs_page
    if not page_path.is_file():
        return []
    referenced = referenced_viewers(page_path)
    return [
        f"{family.label}: documentation does not show its episode viewer -- "
        f"docs/environments/{family.docs_page} embeds no episode-viewer for {trace}."
        for trace in family.viewers
        if trace not in referenced
    ]


def find_undeclared_viewers(page_name: str) -> List[str]:
    """List viewer traces a docs page embeds that no family on that page declares."""
    page_path = DOCS_ENVIRONMENTS_DIR / page_name
    declared = {
        trace for family in FAMILIES if family.docs_page == page_name for trace in family.viewers
    }
    return [
        f"docs/environments/{page_name} replays {trace}, which no matrix row declares. "
        f"Add it to the viewers of the family that page documents."
        for trace in sorted(referenced_viewers(page_path) - declared)
    ]


NON_EXEMPT_FAMILIES = [family for family in FAMILIES if family.package not in EXEMPT_FAMILIES]


def test_matrix_lists_every_environment_family():
    """A new environment directory has to be added to the matrix or written off.

    Without this the matrix silently stops covering the package: the next
    environment lands, none of the per-family tests know about it, and the whole
    file keeps passing.
    """
    on_disk = {
        entry.name
        for entry in ENVIRONMENTS_DIR.iterdir()
        if entry.is_dir() or entry.suffix == ".py"
    }
    # A family key is a directory name: every family is a package, so a loose
    # module under environments is either a non-family entry or an oversight.
    accounted = set(FAMILIES_BY_PACKAGE) | set(EXEMPT_FAMILIES) | set(NON_FAMILY_ENTRIES)
    unaccounted = sorted(on_disk - accounted)
    assert not unaccounted, (
        "These entries under POMDPPlanners/environments are in neither FAMILIES, "
        "EXEMPT_FAMILIES nor NON_FAMILY_ENTRIES: "
        + ", ".join(unaccounted)
        + ". Add a matrix row for a new environment family, or record why the "
        "entry is not one."
    )
    missing = sorted(
        package for package in FAMILIES_BY_PACKAGE if not (ENVIRONMENTS_DIR / package).is_dir()
    )
    assert (
        not missing
    ), "These matrix rows name nothing under POMDPPlanners/environments: " + ", ".join(missing)


def test_only_sanity_and_nuplan_are_exempt():
    """The exemption list is short on purpose, so changing it is visible."""
    assert set(EXEMPT_FAMILIES) == {"sanity_pomdp", "nuplan_pomdp"}
    for package, reason in EXEMPT_FAMILIES.items():
        assert reason.strip(), f"{package} is exempt with no reason given."
        assert (
            package not in FAMILIES_BY_PACKAGE
        ), f"{package} is both exempt and a matrix row; it can only be one."


@pytest.mark.parametrize("package", sorted(EXEMPT_FAMILIES))
def test_exempt_family_still_has_no_visualization_hook(package):
    """An exemption that stopped being true has to be removed, not left to rot."""
    sources = sorted((ENVIRONMENTS_DIR / package).rglob("*.py"))
    with_hook = [
        f"POMDPPlanners/environments/{source.relative_to(ENVIRONMENTS_DIR)}"
        for source in sources
        if "def cache_visualization" in source.read_text(encoding="utf-8")
    ]
    assert not with_hook, (
        f"{package} is exempt from the visualization coverage matrix, but "
        + ", ".join(with_hook)
        + " now defines cache_visualization. Drop the exemption and add a matrix row."
    )


@pytest.mark.parametrize(
    "family", NON_EXEMPT_FAMILIES, ids=[family.label for family in NON_EXEMPT_FAMILIES]
)
def test_family_has_package_visualization_hook(family):
    """Every non-exempt family renders its own episodes."""
    problems = find_missing_hooks(family)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "family", NON_EXEMPT_FAMILIES, ids=[family.label for family in NON_EXEMPT_FAMILIES]
)
def test_family_has_documentation(family):
    """Every non-exempt family has a docs page, or a section of a shared one."""
    problems = find_missing_docs(family)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "family", NON_EXEMPT_FAMILIES, ids=[family.label for family in NON_EXEMPT_FAMILIES]
)
def test_family_has_existing_image(family):
    """Every declared image is a real file that decodes, and animates if it is a GIF."""
    problems = find_missing_image(family)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "family", NON_EXEMPT_FAMILIES, ids=[family.label for family in NON_EXEMPT_FAMILIES]
)
def test_family_is_illustrated(family):
    """Every non-exempt family shows something: a live replay, or a still image."""
    assert family.images or family.viewers, (
        f"{family.label} declares neither an episode viewer nor an image, so its "
        f"docs page shows no picture of the world."
    )


@pytest.mark.parametrize(
    "family", NON_EXEMPT_FAMILIES, ids=[family.label for family in NON_EXEMPT_FAMILIES]
)
def test_family_with_a_scene_replays_it_on_its_page(family):
    """A family whose traces have a 3D scene replays one, and shows no legacy image."""
    problems = (
        find_missing_viewers(family)
        + find_legacy_images(family)
        + find_broken_viewers(family)
        + find_unembedded_viewers(family)
    )
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "family", NON_EXEMPT_FAMILIES, ids=[family.label for family in NON_EXEMPT_FAMILIES]
)
def test_family_documentation_embeds_its_images(family):
    """The page a family claims has to actually show the images the row declares."""
    problems = find_unreferenced_images(family)
    assert not problems, "\n".join(problems)


DOCUMENTED_PAGES = sorted({family.docs_page for family in FAMILIES})


@pytest.mark.parametrize("page_name", DOCUMENTED_PAGES)
def test_documentation_declares_every_image_it_embeds(page_name):
    """A page cannot embed an image or a replay that no matrix row is responsible for."""
    problems = find_undeclared_images(page_name) + find_undeclared_viewers(page_name)
    assert not problems, "\n".join(problems)


def test_only_the_simulator_worlds_keep_a_still_image():
    """CARLA and Isaac Lab have no scene module; everything else replays in 3D.

    Racetrack is the one blocked family (see ``VIEWER_BLOCKED``). Written out
    so that dropping a scene module, or adding a GIF back, shows up as a
    change to this list rather than passing quietly.
    """
    assert sorted(f.package for f in NON_EXEMPT_FAMILIES if f.images) == [
        "carla_pomdp",
        "isaac_lab_pomdp",
        "racetrack_pomdp",
    ]


@pytest.mark.parametrize("package", sorted(VIEWER_BLOCKED))
def test_blocked_viewer_is_still_blocked(package):
    """A blocked family must really have a scene, and lose the entry once it has a trace."""
    family = FAMILIES_BY_PACKAGE[package]
    assert VIEWER_BLOCKED[package].strip(), f"{package} is blocked with no reason given."
    assert family_scene_kinds(family), (
        f"{package} is listed in VIEWER_BLOCKED but has no scene module; " "it needs no exception."
    )
    committed = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (DOCS_ENVIRONMENTS_DIR / "traces").glob("*.json")
        if _trace_payload_kind(path.relative_to(REPO_ROOT).as_posix())[0]
        in family_scene_kinds(family)
    )
    assert not family.viewers and not committed, (
        f"{package} now has a docs trace ({', '.join(committed) or 'declared viewer'}); "
        "remove it from VIEWER_BLOCKED, declare the viewer and drop its legacy image."
    )


def short_table_rows(page: Path) -> List[str]:
    """Rows of a ``list-table`` that carry fewer cells than the table declares.

    reStructuredText does not pad a short row, it drops the whole table and
    emits a build warning nobody reads, so the catalog silently loses an entry.
    That is what happened to the ``CaptureTheFlagPOMDP`` row in ``index.rst``,
    which arrived with three cells in a seven-column table.

    Only tables that declare ``:widths:`` are checked, because that is the only
    place the intended column count is written down rather than inferred from
    the rows -- and inferring it from the rows is exactly what a short row
    would corrupt.

    Args:
        page: Path to the reStructuredText file.

    Returns:
        One message per short row, naming the file, line and cell counts.
    """
    lines = page.read_text(encoding="utf-8").split("\n")
    problems: List[str] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip().startswith(".. list-table::"):
            index += 1
            continue
        table_indent = len(lines[index]) - len(lines[index].lstrip())
        widths = None
        index += 1
        while index < len(lines) and (
            not lines[index].strip()
            or lines[index].strip() == ""
            or lines[index].strip().startswith(":")
        ):
            option = lines[index].strip()
            if option.startswith(":widths:"):
                widths = len(option.split(":", 2)[2].split())
            index += 1

        rows: List[Tuple[int, int]] = []
        cells: Optional[int] = None
        start = 0
        while index < len(lines):
            line, text = lines[index], lines[index].strip()
            outdented = len(line) - len(line.lstrip()) <= table_indent
            if text and outdented and not text.startswith(("*", "-")):
                break
            if text.startswith("* -") or text == "*":
                if cells is not None:
                    rows.append((start, cells))
                cells, start = 1, index + 1
            elif cells is not None and (text == "-" or text.startswith("- ")):
                cells += 1
            index += 1
        if cells is not None:
            rows.append((start, cells))

        if widths is not None:
            for line_number, count in rows:
                if count != widths:
                    problems.append(
                        f"{page.relative_to(REPO_ROOT)}:{line_number}: table row has "
                        f"{count} cells, but the table declares {widths} columns."
                    )
    return problems


@pytest.mark.parametrize(
    "page_name", sorted(path.name for path in DOCS_ENVIRONMENTS_DIR.glob("*.rst"))
)
def test_no_documentation_table_drops_a_row(page_name):
    """A short row makes reStructuredText discard the whole table, quietly."""
    problems = short_table_rows(DOCS_ENVIRONMENTS_DIR / page_name)
    assert not problems, "\n".join(problems)


def _family_with(**overrides) -> EnvironmentFamily:
    """Copy the Tiger row with fields replaced, for the failure-message tests."""
    base = FAMILIES_BY_PACKAGE["tiger_pomdp"]
    fields = {
        "package": base.package,
        "label": base.label,
        "hooks": base.hooks,
        "docs_page": base.docs_page,
        "docs_section": base.docs_section,
        "images": base.images,
        "viewers": base.viewers,
    }
    fields.update(overrides)
    return EnvironmentFamily(**fields)


def test_missing_hook_is_reported_with_the_family_name():
    """A family whose hook went away is named, with the file and method that is gone."""
    family = _family_with(
        label="Widget", hooks=(("tiger_pomdp/tiger_pomdp.py", "TigerVisualizer"),)
    )
    problems = find_missing_hooks(family)
    assert len(problems) == 1
    assert problems[0] == (
        "Widget: package visualization hook missing -- "
        "POMDPPlanners/environments/tiger_pomdp/tiger_pomdp.py "
        "defines no class TigerVisualizer."
    )


def test_missing_hook_module_is_reported_with_the_family_name():
    """A hook whose module was moved or deleted names the path it looked for."""
    family = _family_with(label="Widget", hooks=(("widget_pomdp/widget_pomdp.py", "WidgetPOMDP"),))
    problems = find_missing_hooks(family)
    assert len(problems) == 1
    assert "Widget: package visualization hook missing" in problems[0]
    assert "POMDPPlanners/environments/widget_pomdp/widget_pomdp.py does not exist" in problems[0]


def test_missing_docs_page_is_reported_with_the_family_name():
    """A family with no docs page is named, with the page that was expected."""
    family = _family_with(label="Widget", docs_page="widget.rst")
    assert find_missing_docs(family) == [
        "Widget: documentation missing -- docs/environments/widget.rst does not exist."
    ]


def test_missing_docs_section_is_reported_with_the_family_name():
    """A family sharing a page is named together with the heading that is absent."""
    family = _family_with(label="Widget", docs_page="tiger.rst", docs_section="WidgetPOMDP")
    assert find_missing_docs(family) == [
        "Widget: documentation missing -- docs/environments/tiger.rst has no "
        '"WidgetPOMDP" section.'
    ]


def test_missing_image_is_reported_with_the_family_name():
    """A family whose image was never committed is named, with the path."""
    family = _family_with(label="Widget", images=("docs/images/widget_visualization.gif",))
    assert find_missing_image(family) == [
        "Widget: visualization image missing -- docs/images/widget_visualization.gif "
        "does not exist."
    ]


def test_every_declared_image_is_checked_not_just_the_first(tmp_path, monkeypatch):
    """A second world's image cannot ride along unchecked behind a good first one."""
    good = tmp_path / "docs" / "images" / "good.gif"
    good.parent.mkdir(parents=True)
    frame = Image.new("RGB", (8, 8), "white")
    frame.save(good, save_all=True, append_images=[Image.new("RGB", (8, 8), "black")])
    monkeypatch.setattr(
        "POMDPPlanners.tests.test_environments.test_visualization_coverage_matrix.REPO_ROOT",
        tmp_path,
    )
    family = _family_with(
        label="Widget", images=("docs/images/good.gif", "docs/images/widget_visualization.gif")
    )
    assert find_missing_image(family) == [
        "Widget: visualization image missing -- docs/images/widget_visualization.gif "
        "does not exist."
    ]


def test_corrupt_image_is_reported_as_unreadable(tmp_path, monkeypatch):
    """A truncated GIF is not an image; the page would render a broken link."""
    corrupt = tmp_path / "docs" / "images" / "widget.gif"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"GIF89a not really a gif")
    monkeypatch.setattr(
        "POMDPPlanners.tests.test_environments.test_visualization_coverage_matrix.REPO_ROOT",
        tmp_path,
    )
    family = _family_with(label="Widget", images=("docs/images/widget.gif",))
    problems = find_missing_image(family)
    assert len(problems) == 1
    assert "Widget: visualization image unreadable" in problems[0]
    assert "docs/images/widget.gif does not decode as an image" in problems[0]


def test_half_written_gif_is_reported_as_unreadable(tmp_path, monkeypatch):
    """A GIF cut off mid-file keeps a valid header, so only decoding finds it.

    This is the realistic corruption: an interrupted export, or a partial copy.
    Checking the header alone passed it, and the page rendered half an image.
    """
    source = tmp_path / "docs" / "images" / "whole.gif"
    source.parent.mkdir(parents=True)
    frames = [Image.new("RGB", (64, 64), shade) for shade in ("white", "grey", "black")]
    frames[0].save(source, save_all=True, append_images=frames[1:])
    truncated = tmp_path / "docs" / "images" / "widget.gif"
    truncated.write_bytes(source.read_bytes()[: int(source.stat().st_size * 0.6)])
    monkeypatch.setattr(
        "POMDPPlanners.tests.test_environments.test_visualization_coverage_matrix.REPO_ROOT",
        tmp_path,
    )
    family = _family_with(label="Widget", images=("docs/images/widget.gif",))
    problems = find_missing_image(family)
    assert len(problems) == 1
    assert "Widget: visualization image unreadable" in problems[0]


def test_single_frame_gif_is_reported_as_not_an_animation(tmp_path, monkeypatch):
    """One frame is what a stalled recorder writes, and it reads as a still."""
    still = tmp_path / "docs" / "images" / "widget.gif"
    still.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(still)
    monkeypatch.setattr(
        "POMDPPlanners.tests.test_environments.test_visualization_coverage_matrix.REPO_ROOT",
        tmp_path,
    )
    family = _family_with(label="Widget", images=("docs/images/widget.gif",))
    assert find_missing_image(family) == [
        "Widget: visualization image is not an animation -- docs/images/widget.gif has "
        f"1 frame(s), fewer than the {MIN_ANIMATION_FRAMES} an episode needs."
    ]


def test_still_png_is_accepted_because_a_screenshot_is_not_an_animation(tmp_path, monkeypatch):
    """CARLA and Isaac Lab illustrate with a screenshot, so PNGs skip the frame rule."""
    shot = tmp_path / "docs" / "images" / "widget.png"
    shot.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(shot)
    monkeypatch.setattr(
        "POMDPPlanners.tests.test_environments.test_visualization_coverage_matrix.REPO_ROOT",
        tmp_path,
    )
    family = _family_with(label="Widget", images=("docs/images/widget.png",))
    assert find_missing_image(family) == []


def test_unreferenced_image_is_reported_with_the_page_that_should_show_it():
    """An image nothing embeds is not coverage, however real the file is."""
    family = _family_with(label="Widget", images=("docs/images/mountaincar_recorded_history.gif",))
    assert find_unreferenced_images(family) == [
        "Widget: documentation does not show its visualization -- "
        "docs/environments/tiger.rst embeds no image directive for "
        "docs/images/mountaincar_recorded_history.gif."
    ]


def test_referenced_images_resolve_relative_to_the_page(tmp_path, monkeypatch):
    """Directive targets are page-relative; the matrix rows are repo-relative.

    The ``:download:`` link is in the fixture on purpose: it names a file without
    showing it, so counting it as an embedded image would be wrong.
    """
    page = tmp_path / "docs" / "environments" / "widget.rst"
    page.parent.mkdir(parents=True)
    page.write_text(
        "Widget\n======\n\n"
        ".. image:: ../images/widget.gif\n"
        "   :alt: A widget.\n\n"
        ".. figure:: ../../POMDPPlanners/assets/widget.png\n\n"
        ":download:`notes <../images/notes.txt>`\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "POMDPPlanners.tests.test_environments.test_visualization_coverage_matrix.REPO_ROOT",
        tmp_path,
    )
    assert referenced_images(page) == {
        "docs/images/widget.gif",
        "POMDPPlanners/assets/widget.png",
    }


def test_referenced_images_reads_the_other_directive_forms(tmp_path, monkeypatch):
    """Substitution, next-line and source-root-absolute targets all count.

    Each is valid reST that renders an image. Missing one would either hide an
    undeclared image or, for the ``/`` form, report a declared image as absent
    from a page that plainly shows it.
    """
    page = tmp_path / "docs" / "environments" / "widget.rst"
    page.parent.mkdir(parents=True)
    page.write_text(
        "Widget\n======\n\n"
        ".. |badge| image:: ../images/badge.png\n\n"
        ".. image::\n"
        "   ../images/next_line.gif\n\n"
        ".. figure:: /images/source_root.gif\n"
        "   :width: 480px\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "POMDPPlanners.tests.test_environments.test_visualization_coverage_matrix.REPO_ROOT",
        tmp_path,
    )
    assert referenced_images(page) == {
        "docs/images/badge.png",
        "docs/images/next_line.gif",
        "docs/images/source_root.gif",
    }


def test_option_lines_are_not_mistaken_for_a_next_line_target(tmp_path, monkeypatch):
    """A directive with its target inline must not also swallow its options."""
    page = tmp_path / "docs" / "environments" / "widget.rst"
    page.parent.mkdir(parents=True)
    page.write_text(
        "Widget\n======\n\n"
        ".. image:: ../images/widget.gif\n"
        "   :alt: A widget.\n"
        "   :width: 100%\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "POMDPPlanners.tests.test_environments.test_visualization_coverage_matrix.REPO_ROOT",
        tmp_path,
    )
    assert referenced_images(page) == {"docs/images/widget.gif"}


def test_a_short_table_row_is_reported_with_its_line(tmp_path, monkeypatch):
    """The check has to fail on the shape that actually shipped, not just run.

    Purpose: This is the defect the ``CaptureTheFlagPOMDP`` row had -- a row
        with three cells in a seven-column table. Asserting only that the real
        pages pass would leave a check that could never fail.
    """
    page = tmp_path / "docs" / "environments" / "widget.rst"
    page.parent.mkdir(parents=True)
    page.write_text(
        "Widget\n======\n\n"
        ".. list-table::\n"
        "   :header-rows: 1\n"
        "   :widths: 20 20 20\n"
        "\n"
        "   * - Environment\n"
        "     - Purpose\n"
        "     - Guide\n"
        "   * - ``WidgetPOMDP``\n"
        "     - :doc:`widget`\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "POMDPPlanners.tests.test_environments.test_visualization_coverage_matrix.REPO_ROOT",
        tmp_path,
    )
    problems = short_table_rows(page)
    assert len(problems) == 1
    assert "2 cells" in problems[0] and "3 columns" in problems[0]


def test_an_empty_cell_written_as_a_bare_dash_still_counts(tmp_path, monkeypatch):
    """An empty cell is a cell, so counting only ``- `` would flag good tables.

    Purpose: Several argument tables leave the description of a required
        argument blank, written as a bare ``-``. A check that missed those
        would report seven healthy pages and get switched off.
    """
    page = tmp_path / "docs" / "environments" / "widget.rst"
    page.parent.mkdir(parents=True)
    page.write_text(
        "Widget\n======\n\n"
        ".. list-table::\n"
        "   :widths: 20 20 20\n"
        "\n"
        "   * - ``discount_factor``\n"
        "     - *required*\n"
        "     -\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "POMDPPlanners.tests.test_environments.test_visualization_coverage_matrix.REPO_ROOT",
        tmp_path,
    )
    assert not short_table_rows(page)


def test_empty_image_counts_as_missing(tmp_path, monkeypatch):
    """A zero-byte placeholder is not an image; it would render as a broken link."""
    placeholder = tmp_path / "docs" / "images" / "widget.gif"
    placeholder.parent.mkdir(parents=True)
    placeholder.touch()
    monkeypatch.setattr(
        "POMDPPlanners.tests.test_environments.test_visualization_coverage_matrix.REPO_ROOT",
        tmp_path,
    )
    family = _family_with(label="Widget", images=("docs/images/widget.gif",))
    assert find_missing_image(family) == [
        "Widget: visualization image missing -- docs/images/widget.gif is empty."
    ]


def _write_trace(root: Path, relative: str, kind: str, steps: int = 3) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"payload_kind": kind, "steps": [{} for _ in range(steps)]}),
        encoding="utf-8",
    )


def _fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository root holding a tiger package, a tiger scene and a docs page."""
    package = tmp_path / "POMDPPlanners" / "environments" / "tiger_pomdp"
    package.mkdir(parents=True)
    (package / "trace_exporter.py").write_text(
        'TIGER_PAYLOAD_KIND = "tiger.v1"\n', encoding="utf-8"
    )
    scenes = tmp_path / "POMDPPlanners" / "reporting" / "static" / "viewer" / "scenes"
    scenes.mkdir(parents=True)
    (scenes / "tiger.js").write_text("// scene\n", encoding="utf-8")
    docs = tmp_path / "docs" / "environments"
    docs.mkdir(parents=True)
    module = "POMDPPlanners.tests.test_environments.test_visualization_coverage_matrix"
    monkeypatch.setattr(f"{module}.REPO_ROOT", tmp_path)
    monkeypatch.setattr(f"{module}.ENVIRONMENTS_DIR", tmp_path / "POMDPPlanners" / "environments")
    monkeypatch.setattr(f"{module}.DOCS_ENVIRONMENTS_DIR", docs)
    return tmp_path


def test_a_family_with_a_scene_and_no_viewer_is_reported(tmp_path, monkeypatch):
    """The scene exists, the page shows only the old GIF: that is the gap to report."""
    _fake_repo(tmp_path, monkeypatch)
    family = _family_with(viewers=(), images=("docs/images/tiger.gif",))
    assert find_missing_viewers(family) == [
        "Tiger: has a 3D scene for tiger.v1 (tiger.js) but docs/environments/tiger.rst "
        "declares no episode viewer that replays it."
    ]
    assert find_legacy_images(family) == [
        "Tiger: has a 3D scene, so docs/environments/tiger.rst should replay an episode "
        "instead of showing the legacy image docs/images/tiger.gif."
    ]


def test_a_family_without_a_scene_may_keep_its_image(tmp_path, monkeypatch):
    """CARLA's case: no scene module, so a still image is the right illustration."""
    _fake_repo(tmp_path, monkeypatch)
    (
        tmp_path / "POMDPPlanners" / "reporting" / "static" / "viewer" / "scenes" / "tiger.js"
    ).unlink()
    family = _family_with(viewers=(), images=("docs/images/tiger.png",))
    assert find_missing_viewers(family) == []
    assert find_legacy_images(family) == []


def test_a_declared_viewer_must_be_embedded_and_replayable(tmp_path, monkeypatch):
    """A trace on disk is not coverage until the page embeds it and a scene draws it."""
    root = _fake_repo(tmp_path, monkeypatch)
    _write_trace(root, "docs/environments/traces/tiger.json", "tiger.v1")
    _write_trace(root, "docs/environments/traces/ghost.json", "ghost.v1")
    (root / "docs" / "environments" / "tiger.rst").write_text(
        "Tiger\n=====\n\n.. episode-viewer:: traces/ghost.json\n", encoding="utf-8"
    )
    family = _family_with(
        viewers=("docs/environments/traces/tiger.json", "docs/environments/traces/ghost.json"),
        images=(),
    )
    assert find_unembedded_viewers(family) == [
        "Tiger: documentation does not show its episode viewer -- docs/environments/tiger.rst "
        "embeds no episode-viewer for docs/environments/traces/tiger.json."
    ]
    assert find_broken_viewers(family) == [
        "Tiger: episode viewer trace broken -- docs/environments/traces/ghost.json has payload "
        "kind ghost.v1, which no scene module draws (ghost.js is missing)."
    ]


def test_a_trace_too_short_to_replay_is_broken(tmp_path, monkeypatch):
    """One recorded step is a still frame, the viewer's version of a one-frame GIF."""
    root = _fake_repo(tmp_path, monkeypatch)
    _write_trace(root, "docs/environments/traces/tiger.json", "tiger.v1", steps=1)
    family = _family_with(viewers=("docs/environments/traces/tiger.json",), images=())
    assert find_broken_viewers(family) == [
        "Tiger: episode viewer trace broken -- docs/environments/traces/tiger.json records "
        f"1 step(s), fewer than the {MIN_ANIMATION_FRAMES} a replay needs."
    ]


def test_referenced_viewers_resolve_like_images(tmp_path, monkeypatch):
    """Page-relative and source-root targets both count, as they do for images."""
    root = _fake_repo(tmp_path, monkeypatch)
    page = root / "docs" / "environments" / "widget.rst"
    page.write_text(
        "Widget\n======\n\n"
        ".. episode-viewer:: traces/a.json\n\n"
        "   A caption.\n\n"
        ".. episode-viewer:: /environments/traces/b.json\n",
        encoding="utf-8",
    )
    assert referenced_viewers(page) == {
        "docs/environments/traces/a.json",
        "docs/environments/traces/b.json",
    }
