# SPDX-License-Identifier: MIT

"""Guards the metric naming rules so names cannot drift between environments again.

One concept had become four names (``success_rate``, ``goal_reaching_rate``,
``win_rate``, ``exit_success_rate``), so a cross-environment comparison or a
tuning config had to know each environment's spelling. The shared names now live
in :class:`~POMDPPlanners.core.simulation.metrics.CommonMetricName`. These tests
fail when an environment spells a common metric its own way again.

The environment enums are read from source with :mod:`ast` rather than imported,
so the check covers CARLA, IsaacLab and nuPlan without their simulators and runs
the same way on every machine. Names built at runtime (PacMan's per-ghost
distances) are covered by a second check over the hermetic environment registry.
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from POMDPPlanners.core.simulation.metrics import CommonMetricName
from POMDPPlanners.tests.test_utils.golden_metric_snapshot import build_registry

_ENVIRONMENTS_DIR = Path(__file__).resolve().parents[1] / "environments"

# Every metrics enum the repository had when this test was written, less the
# one removed with OccupancyGridMappingPOMDP. A count below this means the
# source scan broke, not that the names became clean.
_MIN_METRIC_ENUMS = 23

# An environment-specific name that ends like a common one but is not one: laser
# tag reports tagging the opponent separately from reaching its goal.
_ALLOWED_NAMES = frozenset({"tag_success_rate"})

_FORBIDDEN_PREFIXES = ("avg_", "total_", "mean_")
_FORBIDDEN_SUFFIXES = ("success_rate", "goal_reaching_rate", "win_rate", "_count")
_FORBIDDEN_EXACT = frozenset({"ended_by_goal", "ended_by_failure", "ended_by_timeout"})


def naming_violation(name: str) -> Optional[str]:
    """Return why ``name`` breaks the metric naming rules, or ``None`` if it is fine.

    Args:
        name: A metric name as an environment reports it.

    Returns:
        A short reason, or ``None`` when the name is allowed.
    """
    if name in _ALLOWED_NAMES:
        return None
    for prefix in _FORBIDDEN_PREFIXES:
        if name.startswith(prefix):
            return f"starts with {prefix!r}; use 'average_' for a per-episode average"
    for suffix in _FORBIDDEN_SUFFIXES:
        if name.endswith(suffix):
            return f"ends with {suffix!r}; use a CommonMetricName or the 'average_' prefix"
    if name in _FORBIDDEN_EXACT:
        return "is an episode-end channel name; the metric is the '_rate' form"
    return None


def _is_metrics_enum(node: ast.ClassDef) -> bool:
    is_enum = any(
        (isinstance(base, ast.Name) and base.id == "Enum")
        or (isinstance(base, ast.Attribute) and base.attr == "Enum")
        for base in node.bases
    )
    return is_enum and (node.name.endswith("Metric") or node.name.endswith("Metrics"))


def _common_member(value: ast.expr) -> Optional[str]:
    """Return ``X`` when ``value`` is ``CommonMetricName.X.value``."""
    if not (isinstance(value, ast.Attribute) and value.attr == "value"):
        return None
    member = value.value
    if (
        isinstance(member, ast.Attribute)
        and isinstance(member.value, ast.Name)
        and member.value.id == "CommonMetricName"
    ):
        return member.attr
    return None


def _scan_metric_enums() -> Dict[str, List[Tuple[str, ast.expr]]]:
    """Map ``path::EnumName`` to its ``(member, value expression)`` pairs."""
    enums: Dict[str, List[Tuple[str, ast.expr]]] = {}
    for path in sorted(_ENVIRONMENTS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ClassDef) and _is_metrics_enum(node)):
                continue
            members = [
                (statement.targets[0].id, statement.value)
                for statement in node.body
                if isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ]
            key = f"{path.relative_to(_ENVIRONMENTS_DIR)}::{node.name}"
            enums[key] = members
    return enums


_COMMON_VALUES = {member.value: member.name for member in CommonMetricName}


class TestMetricNamingRules:
    """The naming rules applied to every environment's declared metric names."""

    def test_source_scan_finds_every_metrics_enum(self) -> None:
        """Test that the source scan actually sees the environments' metric enums.

        Purpose: Validates the scan is not vacuous, so the checks below cannot
            pass by finding nothing

        Given: The environments package source
        When: Metric enums are collected with ast
        Then: At least as many enums are found as the repository had when the
            rule was introduced, each with at least one member

        Test type: unit
        """
        enums = _scan_metric_enums()

        assert len(enums) >= _MIN_METRIC_ENUMS, sorted(enums)
        assert all(members for members in enums.values())

    def test_common_names_come_from_the_shared_enum(self) -> None:
        """Test that no environment writes a common metric name as a literal.

        Purpose: Validates the shared names have one source, so a typo or a
            local respelling cannot reintroduce a second name for one concept

        Given: Every member of every environment metrics enum
        When: A member's value is a string literal
        Then: That literal is not one of the CommonMetricName values, and any
            CommonMetricName reference names a real member

        Test type: unit
        """
        literal_copies = []
        unknown_references = []
        for enum_key, members in _scan_metric_enums().items():
            for member, value in members:
                common = _common_member(value)
                if common is not None:
                    if common not in CommonMetricName.__members__:
                        unknown_references.append(f"{enum_key}.{member} -> {common}")
                elif isinstance(value, ast.Constant) and value.value in _COMMON_VALUES:
                    literal_copies.append(
                        f"{enum_key}.{member} = {value.value!r}; use "
                        f"CommonMetricName.{_COMMON_VALUES[value.value]}.value"
                    )

        assert not literal_copies, "\n".join(literal_copies)
        assert not unknown_references, "\n".join(unknown_references)

    def test_declared_metric_names_follow_the_rules(self) -> None:
        """Test that every declared environment metric name follows the naming rules.

        Purpose: Validates no environment declares a common metric under a
            different spelling (avg_, total_, success_rate, win_rate, ...)

        Given: Every string-valued member of every environment metrics enum
        When: Each name is checked against the naming rules
        Then: No name violates them

        Test type: unit
        """
        violations = []
        for enum_key, members in _scan_metric_enums().items():
            for member, value in members:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    reason = naming_violation(value.value)
                    if reason is not None:
                        violations.append(f"{enum_key}.{member} = {value.value!r}: {reason}")

        assert not violations, "\n".join(violations)

    def test_metric_enum_values_are_distinct(self) -> None:
        """Test that no environment metrics enum gives two members one name.

        Purpose: Validates no metric silently disappears. Two members with one
            value make the second an Enum alias, which iteration skips, so it
            drops out of get_metric_names(). Pointing members at shared
            CommonMetricName values makes such a collision easy to write.

        Given: Every environment metrics enum, values resolved from source
        When: The resolved values of each enum are compared
        Then: Each enum's values are distinct

        Test type: unit
        """
        duplicates = []
        for enum_key, members in _scan_metric_enums().items():
            seen: Dict[str, str] = {}
            for member, value in members:
                common = _common_member(value)
                if common is not None and common in CommonMetricName.__members__:
                    resolved = CommonMetricName[common].value
                elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                    resolved = value.value
                else:
                    continue
                if resolved in seen:
                    duplicates.append(f"{enum_key}: {seen[resolved]} and {member} = {resolved!r}")
                seen[resolved] = member

        assert not duplicates, "\n".join(duplicates)

    @pytest.mark.parametrize("slug", sorted(build_registry()))
    def test_runtime_metric_names_follow_the_rules(self, slug: str) -> None:
        """Test that names an environment reports at runtime follow the rules.

        Purpose: Validates names built at runtime, which the source scan cannot
            see (PacMan's per-ghost distance metrics are f-strings)

        Given: A hermetic environment from the golden-snapshot registry
        When: get_metric_names() is called
        Then: No reported name violates the naming rules

        Test type: unit
        """
        names = build_registry()[slug]().get_metric_names()

        violations = [
            f"{name}: {naming_violation(name)}" for name in names if naming_violation(name)
        ]

        assert not violations, f"{slug}: " + "; ".join(violations)

    def test_common_names_follow_the_rules(self) -> None:
        """Test that the shared names themselves follow the naming rules.

        Purpose: Validates the shared vocabulary is a valid example to copy

        Given: Every CommonMetricName member
        When: Its value is checked against the naming rules
        Then: None violates them and all values are distinct

        Test type: unit
        """
        # __members__ includes aliases; iterating the enum would skip them and
        # make the distinctness check unable to fail.
        values = [member.value for member in CommonMetricName.__members__.values()]

        assert len(values) == len(set(values))
        assert all(naming_violation(value) is None for value in values)

    @pytest.mark.parametrize(
        "name",
        [
            "avg_episode_length",
            "total_safety_violations",
            "mean_speed_mps",
            "success_rate",
            "goal_reaching_rate",
            "win_rate",
            "exit_success_rate",
            "near_miss_count",
            "ended_by_goal",
        ],
    )
    def test_rule_rejects_old_spellings(self, name: str) -> None:
        """Test that the rule check rejects each spelling that drifted before.

        Purpose: Validates the detector itself, so a weakened rule fails here
            instead of silently passing every environment

        Given: A metric name spelled the way an environment once spelled it
        When: naming_violation() checks it
        Then: It reports a violation

        Test type: unit
        """
        assert naming_violation(name) is not None

    def test_rule_allows_laser_tag_tag_success_rate(self) -> None:
        """Test that laser tag's tag_success_rate is allowed.

        Purpose: Validates the one deliberate exception: tagging the opponent is
            a different event from completing the task

        Given: The name tag_success_rate
        When: naming_violation() checks it
        Then: It reports no violation

        Test type: unit
        """
        assert naming_violation("tag_success_rate") is None
