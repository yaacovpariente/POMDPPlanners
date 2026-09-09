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

Checks are source-level: modules are parsed with :mod:`ast` and docs are read as
text. Nothing here imports an environment, so the families that need CARLA,
Isaac Sim or ``highway-env`` are checked on a machine that has none of them, and
nothing here needs a built ``docs/_build`` tree.
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

# tests/test_environments/<this file> -> tests/ -> POMDPPlanners/ -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENTS_DIR = REPO_ROOT / "POMDPPlanners" / "environments"
DOCS_ENVIRONMENTS_DIR = REPO_ROOT / "docs" / "environments"

# The reStructuredText characters used to underline a section heading in docs/.
_HEADING_UNDERLINE_CHARACTERS = set("=-~^\"'`#*+_:.")


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
        image: Repository-relative path to the image that illustrates the
            family.
    """

    package: str
    label: str
    hooks: Tuple[Tuple[str, str], ...]
    docs_page: str
    docs_section: Optional[str]
    image: str


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
        image=(
            "POMDPPlanners/tests/test_environments/golden_visualizations/" "tiger_visualization.gif"
        ),
    ),
    EnvironmentFamily(
        package="rock_sample_pomdp",
        label="RockSample",
        hooks=(("rock_sample_pomdp/rock_sample_pomdp.py", "RockSamplePOMDP"),),
        docs_page="rock_sample.rst",
        docs_section=None,
        image=(
            "POMDPPlanners/tests/test_environments/golden_visualizations/"
            "rock_sample_visualization.gif"
        ),
    ),
    EnvironmentFamily(
        package="battleship_pomdp",
        label="Battleship",
        hooks=(("battleship_pomdp/battleship_pomdp.py", "BattleshipPOMDP"),),
        docs_page="battleship.rst",
        docs_section=None,
        image="docs/artifacts/battleship_redesign/review.gif",
    ),
    EnvironmentFamily(
        package="pacman_pomdp",
        label="PacMan",
        hooks=(("pacman_pomdp/pacman_pomdp.py", "PacManPOMDP"),),
        docs_page="pacman.rst",
        docs_section=None,
        image=(
            "POMDPPlanners/tests/test_environments/golden_visualizations/"
            "pacman_visualization.gif"
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
        image=(
            "POMDPPlanners/tests/test_environments/golden_visualizations/"
            "light_dark_visualization.gif"
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
        image="docs/images/discrete_maze_visualization.gif",
    ),
    EnvironmentFamily(
        package="cartpole_pomdp",
        label="CartPole",
        hooks=(("cartpole_pomdp/cartpole_pomdp.py", "CartPolePOMDP"),),
        docs_page="cartpole.rst",
        docs_section=None,
        image=(
            "POMDPPlanners/tests/test_environments/golden_visualizations/"
            "cartpole_visualization.gif"
        ),
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
        image=(
            "POMDPPlanners/tests/test_environments/golden_visualizations/" "push_visualization.gif"
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
        image=(
            "POMDPPlanners/tests/test_environments/golden_visualizations/"
            "laser_tag_visualization.gif"
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
        image=(
            "POMDPPlanners/tests/test_environments/golden_visualizations/"
            "safety_ant_velocity_visualization.gif"
        ),
    ),
    EnvironmentFamily(
        package="mountain_car_pomdp",
        label="MountainCar",
        hooks=(("mountain_car_pomdp/mountain_car_pomdp.py", "MountainCarPOMDP"),),
        # Shares a page with SanityPOMDP, so the section heading is the check.
        docs_page="simple.rst",
        docs_section="MountainCarPOMDP",
        image="docs/images/mountaincar_recorded_history.gif",
    ),
    EnvironmentFamily(
        package="racetrack_pomdp",
        label="Racetrack",
        hooks=(("racetrack_pomdp/racetrack_pomdp.py", "RacetrackPOMDP"),),
        docs_page="realistic.rst",
        docs_section="Racetrack",
        image="docs/images/racetrack_recorded_episode.gif",
    ),
    EnvironmentFamily(
        package="carla_pomdp",
        label="CARLA",
        hooks=(("carla_pomdp/carla_pomdp.py", "CarlaPOMDP"),),
        docs_page="realistic.rst",
        docs_section="CARLA",
        image="docs/images/carla_chase_camera.png",
    ),
    EnvironmentFamily(
        package="isaac_lab_pomdp",
        label="IsaacLab",
        hooks=(("isaac_lab_pomdp/isaac_lab_pomdp.py", "IsaacLabPOMDP"),),
        docs_page="realistic.rst",
        docs_section="Isaac Lab",
        image="docs/images/isaac_lab_franka_reach.png",
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
    """List the family's image problems: absent file, or a placeholder of zero bytes."""
    image_path = REPO_ROOT / family.image
    if not image_path.is_file():
        return [
            f"{family.label}: visualization image missing -- {family.image} " f"does not exist."
        ]
    if image_path.stat().st_size == 0:
        return [f"{family.label}: visualization image missing -- {family.image} is empty."]
    return []


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
    """Every non-exempt family's declared image is a real, non-empty file."""
    problems = find_missing_image(family)
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
        "image": base.image,
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
    family = _family_with(label="Widget", image="docs/images/widget_visualization.gif")
    assert find_missing_image(family) == [
        "Widget: visualization image missing -- docs/images/widget_visualization.gif "
        "does not exist."
    ]


def test_empty_image_counts_as_missing(tmp_path, monkeypatch):
    """A zero-byte placeholder is not an image; it would render as a broken link."""
    placeholder = tmp_path / "docs" / "images" / "widget.gif"
    placeholder.parent.mkdir(parents=True)
    placeholder.touch()
    monkeypatch.setattr(
        "POMDPPlanners.tests.test_environments.test_visualization_coverage_matrix.REPO_ROOT",
        tmp_path,
    )
    family = _family_with(label="Widget", image="docs/images/widget.gif")
    assert find_missing_image(family) == [
        "Widget: visualization image missing -- docs/images/widget.gif is empty."
    ]
