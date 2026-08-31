# SPDX-License-Identifier: MIT

"""Guards the dependency direction: core must not depend on the layers above it.

Core defines the contracts -- Environment, Policy, Belief, the tuning configs --
that simulations, planners and environments implement. An import pointing the
other way makes those layers impossible to change without touching core, and
the failure shows up as a circular import in an unrelated file weeks later.

Module-level imports are banned outright, and so are ``if TYPE_CHECKING``
imports -- they cost nothing at runtime but they still state that core knows
about a layer above it, which is the thing being prevented. Deferred imports
inside a function are the existing escape hatch for genuine cycles, so they are
allowlisted one by one: adding a new one is a deliberate act recorded here.
"""

import ast
from pathlib import Path
from typing import List, Set, Tuple

CORE_ROOT = Path(__file__).resolve().parents[2] / "core"

# Layers that sit above core and must never be imported from it.
FORBIDDEN_PREFIXES = (
    "POMDPPlanners.simulations",
    "POMDPPlanners.planners",
    "POMDPPlanners.environments",
    "POMDPPlanners.configs",
    "POMDPPlanners.training",
)

# Deferred imports that already existed, each breaking a real cycle:
# validation needs metric names and the trainer signature, both of which import
# core themselves. Format: (module path relative to core, imported module).
ALLOWED_DEFERRED_IMPORTS: Set[Tuple[str, str]] = {
    ("simulation/hyperparameter_tuning.py", "POMDPPlanners.simulations.simulation_statistics"),
    ("simulation/hyperparameter_tuning.py", "POMDPPlanners.training.policy_trainer"),
}


def _imported_module(node: ast.AST) -> str:
    if isinstance(node, ast.ImportFrom):
        return node.module or ""
    return node.names[0].name


def _module_level_import_ids(tree: ast.Module) -> Set[int]:
    """Ids of imports that run at import time, or declare a type-only dependency.

    A ``TYPE_CHECKING`` block is syntactically nested but is not a deferred
    import: nothing later un-nests it into a function. It is a statement about
    what core is allowed to know, so it is judged by the same rule.
    """
    ids = {id(node) for node in tree.body}
    for node in tree.body:
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test):
            ids.update(id(child) for child in ast.walk(node) if child is not node)
    return ids


def _collect_cross_layer_imports() -> Tuple[List[str], Set[Tuple[str, str]]]:
    """Return (module-level violations, deferred imports found)."""
    module_level: List[str] = []
    deferred: Set[Tuple[str, str]] = set()

    for path in sorted(CORE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        top_level = _module_level_import_ids(tree)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            module = _imported_module(node)
            if not module.startswith(FORBIDDEN_PREFIXES):
                continue

            relative = path.relative_to(CORE_ROOT).as_posix()
            if id(node) in top_level:
                module_level.append(f"{relative}:{node.lineno} imports {module}")
            else:
                deferred.add((relative, module))

    return module_level, deferred


def test_core_has_no_module_level_imports_from_upper_layers():
    """Test that no core module imports simulations, planners or environments at import time.

    Purpose: Keeps the dependency direction one-way so upper layers stay replaceable

    Given: Every Python module under POMDPPlanners/core
    When: Their module-level and TYPE_CHECKING imports are collected
    Then: None of them import a layer that sits above core

    Test type: unit
    """
    module_level, _ = _collect_cross_layer_imports()

    assert not module_level, (
        "core must not import from the layers above it:\n  "
        + "\n  ".join(module_level)
        + "\n\nMove the shared piece down into core, or defer the import into the "
        "function that needs it and add it to ALLOWED_DEFERRED_IMPORTS."
    )


def test_no_new_deferred_cross_layer_imports_in_core():
    """Test that deferred cross-layer imports in core stay limited to the known set.

    Purpose: A function-level import still couples the layers, so new ones need a decision

    Given: Every Python module under POMDPPlanners/core
    When: Their function-level imports of upper layers are collected
    Then: The set matches ALLOWED_DEFERRED_IMPORTS exactly

    Test type: unit
    """
    _, deferred = _collect_cross_layer_imports()

    added = deferred - ALLOWED_DEFERRED_IMPORTS
    removed = ALLOWED_DEFERRED_IMPORTS - deferred

    assert not added, (
        "new deferred import from core into an upper layer:\n  "
        + "\n  ".join(f"{path} imports {module}" for path, module in sorted(added))
        + "\n\nPrefer moving the shared piece into core. If the cycle is real, "
        "add it to ALLOWED_DEFERRED_IMPORTS with a comment saying why."
    )
    assert not removed, (
        "a cycle was broken -- remove it from ALLOWED_DEFERRED_IMPORTS so the "
        "list keeps meaning something:\n  "
        + "\n  ".join(f"{path} imports {module}" for path, module in sorted(removed))
    )
