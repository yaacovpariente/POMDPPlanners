"""Static sdist content checks.

These run without compiling anything (~1s). They guard against the class of
packaging bug that produced a broken 0.3.0 on PyPI: ``MANIFEST.in`` failed
to bundle the C++ headers referenced by ``setup.py``'s ``include_dirs``,
so every ``pip install`` from PyPI hit a missing-header compile error.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tarfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "POMDPPlanners"
NATIVE_SUFFIXES = {".cpp", ".hpp", ".h", ".pyi"}


@pytest.fixture(scope="module")
def built_sdist(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Build the project sdist once per module run, return its path.

    A stale ``POMDPPlanners.egg-info/SOURCES.txt`` from a previous build
    will silently override ``MANIFEST.in``, so this fixture deletes it
    first to mirror the state of a fresh CI checkout.
    """
    egg_info = REPO_ROOT / "POMDPPlanners.egg-info"
    if egg_info.exists():
        shutil.rmtree(egg_info)
    out_dir = tmp_path_factory.mktemp("sdist")
    subprocess.check_call(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(out_dir)],
        cwd=REPO_ROOT,
    )
    matches = list(out_dir.glob("*.tar.gz"))
    assert len(matches) == 1, f"expected one sdist, got {matches}"
    return matches[0]


def _sdist_members(sdist: pathlib.Path) -> set[str]:
    with tarfile.open(sdist) as tar:
        members: set[str] = set()
        for member in tar.getmembers():
            if "/" not in member.name:
                continue
            members.add(member.name.split("/", 1)[1])
        return members


def test_sdist_bundles_all_native_sources_and_stubs(built_sdist: pathlib.Path) -> None:
    """sdist must bundle every C++ source/header/stub the build needs.

    Purpose: Validates that ``MANIFEST.in`` keeps every native source, header,
        and ``.pyi`` stub on disk inside the sdist tarball.

    Given: The project sdist freshly built from the working tree.
    When: The tarball is inspected for entries matching expected native files.
    Then: Every ``.cpp``, ``.hpp``, ``.h``, and ``.pyi`` under
        ``POMDPPlanners/`` on disk appears in the sdist.

    Test type: integration
    """
    on_disk = {
        str(path.relative_to(REPO_ROOT))
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file() and path.suffix in NATIVE_SUFFIXES
    }
    in_sdist = _sdist_members(built_sdist)
    missing = sorted(on_disk - in_sdist)
    assert not missing, (
        "Native sources/headers/stubs present on disk but missing from sdist. "
        f"Update MANIFEST.in. Missing: {missing}"
    )


def test_sdist_includes_packaging_metadata(built_sdist: pathlib.Path) -> None:
    """sdist must include the metadata files PyPI / pip rely on.

    Purpose: Validates that core project metadata files are bundled in the sdist.

    Given: The project sdist freshly built from the working tree.
    When: The tarball is inspected for top-level metadata entries.
    Then: ``README.md``, ``LICENSE.md``, ``CHANGELOG.md``, ``pyproject.toml``,
        and ``setup.py`` are all present.

    Test type: integration
    """
    in_sdist = _sdist_members(built_sdist)
    candidates = (
        "README.md",
        "LICENSE.md",
        "CHANGELOG.md",
        "pyproject.toml",
        "setup.py",
        "MANIFEST.in",
    )
    required = {name for name in candidates if (REPO_ROOT / name).exists()}
    missing = sorted(required - in_sdist)
    assert not missing, f"sdist missing required metadata files: {missing}"


def test_version_matches_pyproject() -> None:
    """``POMDPPlanners.__version__`` must match ``pyproject.toml`` version.

    Purpose: Validates a single source of truth for the package version so a
        release cannot ship while ``__init__.py`` lags ``pyproject.toml``.

    Given: The current package source tree.
    When: ``__version__`` from ``POMDPPlanners`` and the ``version =`` line
        in ``pyproject.toml`` are read.
    Then: The two strings are equal.

    Test type: unit
    """
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pyproject_version: str | None = None
    for line in pyproject_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("version") and "=" in stripped:
            pyproject_version = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            break
    assert pyproject_version is not None, "no version= line in pyproject.toml"

    import POMDPPlanners as pkg  # pylint: disable=import-outside-toplevel

    assert pkg.__version__ == pyproject_version, (
        f"version mismatch: POMDPPlanners.__version__={pkg.__version__!r}, "
        f"pyproject.toml version={pyproject_version!r}"
    )
