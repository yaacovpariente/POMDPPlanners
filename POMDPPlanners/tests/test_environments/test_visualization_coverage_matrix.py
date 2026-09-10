# SPDX-License-Identifier: MIT

"""Every environment family must stay visualizable, documented and illustrated.

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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest
from PIL import Image

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
        images: Repository-relative paths to every image that illustrates the
            family. A family with several concrete worlds lists one per world,
            because a single declared image left the others -- the continuous
            Push, LaserTag and Maze GIFs among them -- embedded by the docs and
            checked by nothing.
    """

    package: str
    label: str
    hooks: Tuple[Tuple[str, str], ...]
    docs_page: str
    docs_section: Optional[str]
    images: Tuple[str, ...]


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

# Entries under POMDPPlanners/environments that are not environment families.
NON_FAMILY_ENTRIES: Dict[str, str] = {
    "__init__.py": "The package's own registry module.",
    "__pycache__": "Bytecode cache.",
    "environment_utils": "Shared helpers, not a world.",
    "sanity_pomdp_vectorized_model.py": "Batched model for the sanity family.",
    "t_maze_pomdp": "Import shim kept for configurations saved before the Maze rename.",
    "tiger_pomdp_vectorized_model.py": "Batched model for the tiger family.",
    "tiger_visualization_assets": "Image assets used by the tiger renderer.",
    "tiger_visualizer.py": "The tiger family's renderer, which lives beside its module.",
}

FAMILIES: Tuple[EnvironmentFamily, ...] = (
    EnvironmentFamily(
        package="tiger_pomdp",
        label="Tiger",
        hooks=(("tiger_pomdp.py", "TigerPOMDP"),),
        docs_page="tiger.rst",
        docs_section=None,
        images=(
            "POMDPPlanners/tests/test_environments/golden_visualizations/tiger_visualization.gif",
        ),
    ),
    EnvironmentFamily(
        package="rock_sample_pomdp",
        label="RockSample",
        hooks=(("rock_sample_pomdp/rock_sample_pomdp.py", "RockSamplePOMDP"),),
        docs_page="rock_sample.rst",
        docs_section=None,
        images=(
            "POMDPPlanners/tests/test_environments/golden_visualizations/"
            "rock_sample_visualization.gif",
        ),
    ),
    EnvironmentFamily(
        package="battleship_pomdp",
        label="Battleship",
        hooks=(("battleship_pomdp/battleship_pomdp.py", "BattleshipPOMDP"),),
        docs_page="battleship.rst",
        docs_section=None,
        # The page embeds the approved review artifact, not the golden fixture.
        # That is deliberate: docs/artifacts/battleship_redesign/README.md pins
        # this file's SHA-256 and records the golden as a separate asset.
        images=("docs/artifacts/battleship_redesign/review.gif",),
    ),
    EnvironmentFamily(
        package="pacman_pomdp",
        label="PacMan",
        hooks=(("pacman_pomdp/pacman_pomdp.py", "PacManPOMDP"),),
        docs_page="pacman.rst",
        docs_section=None,
        images=(
            "POMDPPlanners/tests/test_environments/golden_visualizations/"
            "pacman_visualization.gif",
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
        images=(
            "POMDPPlanners/tests/test_environments/golden_visualizations/"
            "light_dark_visualization.gif",
        ),
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
        images=(
            "docs/images/discrete_maze_visualization.gif",
            "docs/images/continuous_maze_visualization.gif",
            "docs/images/t_maze_visualization.gif",
        ),
    ),
    EnvironmentFamily(
        package="cartpole_pomdp",
        label="CartPole",
        hooks=(("cartpole_pomdp/cartpole_pomdp.py", "CartPolePOMDP"),),
        docs_page="cartpole.rst",
        docs_section=None,
        # The docs copy, byte-identical to the golden GIF the renderer test
        # pins. The matrix checks the file the page actually embeds, so this
        # row follows the page rather than the test fixture.
        images=("docs/images/cartpole_visualization.gif",),
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
        images=(
            "POMDPPlanners/tests/test_environments/golden_visualizations/push_visualization.gif",
            "POMDPPlanners/tests/test_environments/golden_visualizations/"
            "continuous_push_visualization.gif",
        ),
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
        images=(
            "POMDPPlanners/tests/test_environments/golden_visualizations/"
            "laser_tag_visualization.gif",
            "POMDPPlanners/tests/test_environments/golden_visualizations/"
            "continuous_laser_tag_visualization.gif",
        ),
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
        images=(
            "POMDPPlanners/tests/test_environments/golden_visualizations/"
            "safety_ant_velocity_visualization.gif",
        ),
    ),
    EnvironmentFamily(
        package="mountain_car_pomdp",
        label="MountainCar",
        hooks=(("mountain_car_pomdp/mountain_car_pomdp.py", "MountainCarPOMDP"),),
        # Shares a page with SanityPOMDP, so the section heading is the check.
        docs_page="simple.rst",
        docs_section="MountainCarPOMDP",
        images=("docs/images/mountaincar_recorded_history.gif",),
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
        package="carla_pomdp",
        label="CARLA",
        hooks=(("carla_pomdp/carla_pomdp.py", "CarlaPOMDP"),),
        docs_page="realistic.rst",
        docs_section="CARLA",
        images=("docs/images/carla_chase_camera.png",),
    ),
    EnvironmentFamily(
        package="isaac_lab_pomdp",
        label="IsaacLab",
        hooks=(("isaac_lab_pomdp/isaac_lab_pomdp.py", "IsaacLabPOMDP"),),
        docs_page="realistic.rst",
        docs_section="Isaac Lab",
        images=("docs/images/isaac_lab_franka_reach.png",),
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
        image
        for family in FAMILIES
        if family.docs_page == page_name
        for image in family.images
    }
    return [
        f"docs/environments/{page_name} embeds {image}, which no matrix row declares. "
        f"Add it to the images of the family that page documents."
        for image in sorted(referenced_images(page_path) - declared)
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
    keys = set(FAMILIES_BY_PACKAGE) | set(EXEMPT_FAMILIES)
    # A family key is a directory name, except for the single-module families
    # (tiger, sanity) whose entry on disk carries a ``.py`` suffix.
    accounted = keys | {f"{key}.py" for key in keys} | set(NON_FAMILY_ENTRIES)
    unaccounted = sorted(on_disk - accounted)
    assert not unaccounted, (
        "These entries under POMDPPlanners/environments are in neither FAMILIES, "
        "EXEMPT_FAMILIES nor NON_FAMILY_ENTRIES: "
        + ", ".join(unaccounted)
        + ". Add a matrix row for a new environment family, or record why the "
        "entry is not one."
    )
    missing = sorted(
        package
        for package in FAMILIES_BY_PACKAGE
        if not (ENVIRONMENTS_DIR / package).exists()
        and not (ENVIRONMENTS_DIR / f"{package}.py").exists()
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
    directory = ENVIRONMENTS_DIR / package
    sources = (
        sorted(directory.rglob("*.py"))
        if directory.is_dir()
        else [ENVIRONMENTS_DIR / f"{package}.py"]
    )
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
def test_family_documentation_embeds_its_images(family):
    """The page a family claims has to actually show the images the row declares."""
    problems = find_unreferenced_images(family)
    assert not problems, "\n".join(problems)


DOCUMENTED_PAGES = sorted({family.docs_page for family in FAMILIES})


@pytest.mark.parametrize("page_name", DOCUMENTED_PAGES)
def test_documentation_declares_every_image_it_embeds(page_name):
    """A page cannot embed an image that no matrix row is responsible for."""
    problems = find_undeclared_images(page_name)
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
    }
    fields.update(overrides)
    return EnvironmentFamily(**fields)


def test_missing_hook_is_reported_with_the_family_name():
    """A family whose hook went away is named, with the file and method that is gone."""
    family = _family_with(label="Widget", hooks=(("tiger_pomdp.py", "TigerVisualizer"),))
    problems = find_missing_hooks(family)
    assert len(problems) == 1
    assert problems[0] == (
        "Widget: package visualization hook missing -- "
        "POMDPPlanners/environments/tiger_pomdp.py defines no class TigerVisualizer."
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
    family = _family_with(label="Widget", docs_page="simple.rst", docs_section="WidgetPOMDP")
    assert find_missing_docs(family) == [
        "Widget: documentation missing -- docs/environments/simple.rst has no "
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
    family = _family_with(
        label="Widget", images=("docs/images/mountaincar_recorded_history.gif",)
    )
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
