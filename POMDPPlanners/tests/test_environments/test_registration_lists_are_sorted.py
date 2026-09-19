# SPDX-License-Identifier: MIT

"""Every hand-maintained environment registry stays in alphabetical order.

Adding an environment means adding a line to a handful of central lists:
``ENVIRONMENT_REGISTRY`` and ``__all__`` in the environments package, the
``ENV_BUILDERS`` table the API conformance suite iterates, the ``FAMILIES``
matrix that guards visualization coverage, and one pinned-kwargs function per
environment. None of those lists had an order, so every author appended at the
same place, and two environment branches developed in parallel inserted at the
same line. Git cannot order two insertions at one position, so it stops --
every such merge so far has been resolved by keeping both sides, a conflict
that never carried a real decision.

Alphabetical order spreads the insertions out. Two environments now collide
only when their names are adjacent in the alphabet, which turns a conflict on
every pair of concurrent environment branches into a rare one.

This does not make the lists redundant. They are still written by hand, and a
missing entry still fails the suite that reads the list. Order is the only
thing enforced here.

The keys are compared with plain :func:`sorted`, so ordering is by code point:
``TMazePOMDP`` sorts before ``TigerPOMDP`` because ``M`` precedes ``i``. A
case-insensitive order would read better and would be one more rule to
remember when the failure message says where to move the line.
"""

import ast
from pathlib import Path
from typing import List

from POMDPPlanners.environments import ENVIRONMENT_REGISTRY, __all__ as ENVIRONMENT_EXPORTS
from POMDPPlanners.tests.test_environments.test_env_api_conformance import ENV_BUILDERS
from POMDPPlanners.tests.test_environments.test_visualization_coverage_matrix import FAMILIES

PINNED_KWARGS_MODULE = Path(__file__).resolve().parents[1] / "test_utils" / "env_pinned_kwargs.py"


def _misplaced(names: List[str]) -> str:
    """Name the first entry that breaks alphabetical order, for the failure message.

    Args:
        names: The list's keys, in the order the source file declares them.

    Returns:
        A sentence naming the offending entry and where it belongs, or an empty
        string when the list is already sorted.
    """
    ordered = sorted(names)
    for declared, expected in zip(names, ordered):
        if declared != expected:
            return f"{declared!r} is declared where {expected!r} belongs"
    return ""


def test_environment_registry_is_alphabetical():
    """Test that ENVIRONMENT_REGISTRY is declared in alphabetical order.

    Purpose: Keeps two environment branches from inserting at the same line

    Given: The environments package's public registry
    When: Its keys are read in declaration order
    Then: They are already sorted

    Test type: unit
    """
    names = list(ENVIRONMENT_REGISTRY)
    assert names == sorted(names), (
        "ENVIRONMENT_REGISTRY must be alphabetical so concurrent environment "
        f"branches insert at different lines: {_misplaced(names)}"
    )


def test_environment_exports_are_alphabetical():
    """Test that the environments package's ``__all__`` is alphabetical.

    Purpose: Same insertion-point argument as the registry

    Given: ``POMDPPlanners.environments.__all__``
    When: Its entries are read in declaration order
    Then: They are already sorted

    Test type: unit
    """
    names = list(ENVIRONMENT_EXPORTS)
    assert names == sorted(names), (
        f"__all__ in POMDPPlanners/environments/__init__.py must be alphabetical: "
        f"{_misplaced(names)}"
    )


def test_env_builders_are_alphabetical():
    """Test that the API conformance table is alphabetical by environment label.

    Purpose: Same insertion-point argument, for the list every conformance test
        is parametrized over

    Given: ``ENV_BUILDERS``
    When: Its labels are read in declaration order
    Then: They are already sorted

    Test type: unit
    """
    names = [label for label, _ in ENV_BUILDERS]
    assert names == sorted(
        names
    ), f"ENV_BUILDERS must be alphabetical by label: {_misplaced(names)}"


def test_visualization_families_are_alphabetical():
    """Test that the visualization coverage matrix is alphabetical by package.

    Purpose: Same insertion-point argument, for the matrix that has conflicted
        on the last two environment branches

    Given: ``FAMILIES``
    When: The package names are read in declaration order
    Then: They are already sorted

    Test type: unit
    """
    names = [family.package for family in FAMILIES]
    assert names == sorted(names), f"FAMILIES must be alphabetical by package: {_misplaced(names)}"


def test_pinned_kwargs_functions_are_alphabetical():
    """Test that the pinned-kwargs helpers are defined in alphabetical order.

    Purpose: A new environment appends its helper at the end of this module,
        which is the one place where two branches collide by construction

    Given: ``env_pinned_kwargs.py`` parsed as source, so nothing is imported
    When: Its top-level function names are read in definition order
    Then: They are already sorted

    Test type: unit
    """
    module = ast.parse(PINNED_KWARGS_MODULE.read_text())
    names = [node.name for node in module.body if isinstance(node, ast.FunctionDef)]
    assert names == sorted(names), (
        f"{PINNED_KWARGS_MODULE.name} must define its helpers in alphabetical "
        f"order: {_misplaced(names)}"
    )
