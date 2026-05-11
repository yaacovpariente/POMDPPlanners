"""End-to-end sdist install check.

Slow: this builds the sdist, creates a fresh venv, installs the tarball
(compiling pybind11 extensions), and imports the package. Marked ``slow``
so it stays out of the default ``pytest`` run; intended for release CI.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import venv

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


@pytest.mark.slow
def test_sdist_installs_and_imports_in_clean_venv(tmp_path: pathlib.Path) -> None:
    """Sdist must install cleanly and expose its native extension.

    Purpose: Validates that the published artifact is actually installable end
        to end: the sdist compiles, the wheel imports, and the pybind11
        ``_native`` extension loads. Catches MANIFEST.in gaps that would
        otherwise only surface on PyPI.

    Given: The project working tree.
    When: ``python -m build --sdist`` runs, then the resulting tarball is
        installed into a brand-new venv with ``pip install``.
    Then: ``import POMDPPlanners`` and ``from POMDPPlanners.core import
        _native`` both succeed from a working directory outside the repo.

    Test type: integration
    """
    probe = subprocess.run(
        [sys.executable, "-m", "build", "--version"],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "build"])
    subprocess.check_call(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(tmp_path)],
        cwd=REPO_ROOT,
    )
    sdists = list(tmp_path.glob("*.tar.gz"))
    assert len(sdists) == 1, f"expected one sdist, got {sdists}"
    sdist = sdists[0]

    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=True)
    venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists():
        venv_python = venv_dir / "Scripts" / "python.exe"
    assert venv_python.exists(), f"venv python not found under {venv_dir}"

    subprocess.check_call([str(venv_python), "-m", "pip", "install", str(sdist)])

    # `cwd` must NOT be the repo root: Python prepends `sys.path[0]` from the
    # script's directory, and from the repo root that would import the source
    # tree instead of the installed package, masking a broken sdist.
    subprocess.check_call(
        [
            str(venv_python),
            "-c",
            "import POMDPPlanners; from POMDPPlanners.core import _native; "
            "print('ok', POMDPPlanners.__version__, _native.__file__)",
        ],
        cwd=tmp_path,
    )
