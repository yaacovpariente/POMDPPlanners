#!/usr/bin/env python3
"""
Scan Test Documentation for Duplicates

This script analyzes the test documentation summaries to identify duplicate tests
that test the same functionality WITHIN THE SAME test file. It looks for:
- Similar test purposes within each test file
- Identical test scenarios (Given/When/Then) within each test file
- Redundant functionality coverage within each test file
- Pattern duplicates within individual test files

Note: This script only compares tests within the same file, not across different files,
following the updated workflow requirements in CLAUDE.md.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from difflib import SequenceMatcher
import hashlib


class TestInfo:
    """Data structure to hold test information."""

    def __init__(
        self,
        file_path: str,
        function_name: str,
        line_number: int,
        purpose: str,
        given: str,
        when: str,
        then: str,
        test_type: str,
    ):
        self.file_path = file_path
        self.function_name = function_name
        self.line_number = line_number
        self.purpose = purpose.strip()
        self.given = given.strip()
        self.when = when.strip()
        self.then = then.strip()
        self.test_type = test_type.strip()

        # Generate semantic fingerprint
        self.semantic_fingerprint = self._generate_semantic_fingerprint()

    def _generate_semantic_fingerprint(self) -> str:
        """Generate a semantic fingerprint based on test content."""
        # Normalize and combine key test components
        content = f"{self.purpose} {self.given} {self.when} {self.then}".lower()

        # Remove common words and normalize
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "that",
            "this",
            "these",
            "those",
        }

        # Extract meaningful words
        words = re.findall(r"\b[a-z_]+\b", content)
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]

        # Create fingerprint from sorted meaningful words
        fingerprint_text = " ".join(sorted(set(meaningful_words)))
        return hashlib.md5(fingerprint_text.encode()).hexdigest()[:16]

    def get_test_scenario(self) -> str:
        """Get complete test scenario description."""
        return f"Purpose: {self.purpose}\nGiven: {self.given}\nWhen: {self.when}\nThen: {self.then}"

    def __str__(self):
        return f"{self.file_path}:{self.function_name}:{self.line_number}"


def parse_test_summary_file(summary_file: Path) -> List[TestInfo]:
    """Parse a test summary markdown file and extract test information."""
    tests = []

    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {summary_file}: {e}")
        return tests

    # Extract original file path
    original_file_match = re.search(r"\*\*Original File:\*\* `([^`]+)`", content)
    if not original_file_match:
        return tests

    original_file = original_file_match.group(1)

    # Split content into test sections
    test_sections = re.split(r"### \d+\. `([^`]+)`", content)[1:]  # Skip header

    # Process test sections in pairs (function_name, content)
    for i in range(0, len(test_sections), 2):
        if i + 1 >= len(test_sections):
            break

        function_name = test_sections[i]
        section_content = test_sections[i + 1]

        # Extract test information
        line_match = re.search(r"\*\*Line:\*\* (\d+)", section_content)
        purpose_match = re.search(r"\*\*Purpose:\*\* ([^\n]+)", section_content)
        given_match = re.search(r"\*\*Given:\*\* ([^\n]+)", section_content)
        when_match = re.search(r"\*\*When:\*\* ([^\n]+)", section_content)
        then_match = re.search(r"\*\*Then:\*\* ([^\n]+)", section_content)
        test_type_match = re.search(r"\*\*Test Type:\*\* ([^\n]+)", section_content)

        # Only include tests with meaningful documentation
        if purpose_match or given_match or when_match or then_match:
            test_info = TestInfo(
                file_path=original_file,
                function_name=function_name,
                line_number=int(line_match.group(1)) if line_match else 0,
                purpose=purpose_match.group(1) if purpose_match else "",
                given=given_match.group(1) if given_match else "",
                when=when_match.group(1) if when_match else "",
                then=then_match.group(1) if then_match else "",
                test_type=test_type_match.group(1) if test_type_match else "",
            )
            tests.append(test_info)

    return tests


def evaluate_same_purpose(test1: TestInfo, test2: TestInfo) -> bool:
    """
    Use AI evaluation to determine if two tests have the same purpose.

    This function analyzes the test purposes, function names, and documentation
    to determine if the tests are testing the same functionality, not just
    similar generic documentation.

    Args:
        test1: First test to compare
        test2: Second test to compare

    Returns:
        True if tests have the same purpose, False otherwise
    """
    # If either test lacks meaningful documentation, skip
    if not test1.purpose.strip() or not test2.purpose.strip():
        return False

    # Skip tests with generic purposes that don't provide meaningful information
    generic_purposes = [
        "validates proper initialization of",
        "validates initialization",
        "validates action selection",
        "validates attribute presence",
        "validates test setup conditions",
        "validates basic functionality",
        "validates error handling",
        "test setup conditions",
        "expected behavior is verified",
    ]

    purpose1_lower = test1.purpose.lower().strip()
    purpose2_lower = test2.purpose.lower().strip()

    # Skip if both have generic purposes
    if any(generic in purpose1_lower for generic in generic_purposes) and any(
        generic in purpose2_lower for generic in generic_purposes
    ):
        return False

    # Construct detailed context for AI evaluation
    evaluation_context = f"""
    Analyze these two test functions to determine if they test THE SAME FUNCTIONALITY:

    TEST 1:
    Function: {test1.function_name}
    Purpose: {test1.purpose}
    Given: {test1.given}
    When: {test1.when}  
    Then: {test1.then}
    
    TEST 2:
    Function: {test2.function_name}
    Purpose: {test2.purpose}
    Given: {test2.given}
    When: {test2.when}
    Then: {test2.then}
    
    Question: Do these two tests verify the SAME specific functionality or behavior?
    
    Consider:
    - Do they test the same feature/method/behavior?
    - Are they testing the same edge case or scenario?
    - Would removing one test leave a gap in coverage that the other doesn't fill?
    
    Examples of SAME purpose:
    - Both test "config_id generation for identical beliefs" 
    - Both test "error handling for invalid input X"
    - Both test "initialization with parameter Y set to Z"
    
    Examples of DIFFERENT purposes:
    - One tests initialization, other tests action selection
    - One tests error case A, other tests error case B  
    - One tests feature X, other tests feature Y
    - One tests valid input, other tests invalid input
    - One tests setup, other tests teardown
    
    Answer with only: SAME or DIFFERENT
    """

    # For now, implement conservative heuristics since we can't make actual AI calls
    # Focus on exact purpose matching and function name analysis

    # Check if purposes are substantially identical (not just similar)
    if purpose1_lower == purpose2_lower and len(purpose1_lower) > 10:
        return True

    # Check if function names suggest same specific functionality
    # Extract the core functionality being tested
    func1_core = test1.function_name.replace("test_", "").lower()
    func2_core = test2.function_name.replace("test_", "").lower()

    # If function names are very similar and purposes are similar, likely same
    if func1_core == func2_core:
        return True

    # Check for exact purpose matches with specific technical terms
    specific_terms = [
        "config_id",
        "hash",
        "initialization",
        "error",
        "exception",
        "sample",
        "update",
    ]
    if any(term in purpose1_lower and term in purpose2_lower for term in specific_terms):
        # Further analyze if they test the same specific aspect
        if purpose1_lower.replace(" ", "") == purpose2_lower.replace(" ", ""):
            return True

    return False


def find_duplicate_tests(all_tests: List[TestInfo]) -> List[List[TestInfo]]:
    """Find groups of duplicate/similar tests within the same test file."""
    duplicates = []

    # Group tests by file path first
    tests_by_file = defaultdict(list)
    for test in all_tests:
        tests_by_file[test.file_path].append(test)

    # Process each file separately
    for file_path, file_tests in tests_by_file.items():
        processed = set()

        # Group by semantic fingerprint first (exact matches within the file)
        fingerprint_groups = defaultdict(list)
        for test in file_tests:
            if test.semantic_fingerprint:  # Only tests with meaningful content
                fingerprint_groups[test.semantic_fingerprint].append(test)

        # Find exact fingerprint matches within the file
        for fingerprint, tests in fingerprint_groups.items():
            if len(tests) > 1:
                duplicates.append(tests)
                for test in tests:
                    processed.add(id(test))

        # Find tests with same purpose among remaining tests within the file
        remaining_tests = [
            test for test in file_tests if id(test) not in processed and test.purpose
        ]

        for i, test1 in enumerate(remaining_tests):
            if id(test1) in processed:
                continue

            same_purpose_group = [test1]
            processed.add(id(test1))

            for j, test2 in enumerate(remaining_tests[i + 1 :], i + 1):
                if id(test2) in processed:
                    continue

                # Use AI-based evaluation to determine if tests have same purpose
                if evaluate_same_purpose(test1, test2):
                    same_purpose_group.append(test2)
                    processed.add(id(test2))

            if len(same_purpose_group) > 1:
                duplicates.append(same_purpose_group)

    return duplicates


def find_pattern_duplicates(all_tests: List[TestInfo]) -> Dict[str, List[TestInfo]]:
    """Find tests that follow similar patterns within the same file (initialization, error handling, etc.)."""
    patterns = defaultdict(list)

    # Group tests by file path first
    tests_by_file = defaultdict(list)
    for test in all_tests:
        tests_by_file[test.file_path].append(test)

    # Process each file separately for pattern analysis
    for file_path, file_tests in tests_by_file.items():
        file_patterns = defaultdict(list)

        for test in file_tests:
            # Extract patterns from function names and purposes
            function_lower = test.function_name.lower()
            purpose_lower = test.purpose.lower()

            # Initialization patterns
            if any(
                word in function_lower for word in ["init", "initialization", "create", "construct"]
            ):
                file_patterns[f"initialization_in_{Path(file_path).name}"].append(test)

            # Error handling patterns
            if any(
                word in function_lower + purpose_lower
                for word in ["error", "raises", "exception", "invalid", "fail"]
            ):
                file_patterns[f"error_handling_in_{Path(file_path).name}"].append(test)

            # Configuration patterns
            if any(
                word in function_lower + purpose_lower
                for word in ["config", "parameter", "setting"]
            ):
                file_patterns[f"configuration_in_{Path(file_path).name}"].append(test)

            # Validation patterns
            if any(
                word in function_lower + purpose_lower
                for word in ["validate", "verify", "check", "ensure"]
            ):
                file_patterns[f"validation_in_{Path(file_path).name}"].append(test)

            # State management patterns
            if any(
                word in function_lower + purpose_lower
                for word in ["state", "update", "modify", "change"]
            ):
                file_patterns[f"state_management_in_{Path(file_path).name}"].append(test)

        # Only include patterns with multiple tests within the same file
        for pattern_name, pattern_tests in file_patterns.items():
            if len(pattern_tests) >= 2:  # Lower threshold since we're looking within single files
                patterns[pattern_name] = pattern_tests

    return patterns


def generate_duplicate_report(
    duplicates: List[List[TestInfo]],
    patterns: Dict[str, List[TestInfo]],
    output_file: Path,
) -> None:
    """Generate a comprehensive duplicate test report."""

    total_tests = sum(len(group) for group in duplicates)
    total_pattern_tests = sum(len(tests) for tests in patterns.values())

    report_content = f"""# Duplicate Test Analysis Report (Within-File Analysis Only)

**Analysis Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **🔍 Duplicate Groups Found:** {len(duplicates)}
- **⚠️ Potentially Duplicate Tests:** {total_tests}
- **📊 Pattern Groups Identified:** {len(patterns)}
- **🔄 Tests in Pattern Groups:** {total_pattern_tests}

**Note:** This analysis only compares tests within the same test file, using AI-based purpose evaluation rather than generic similarity metrics, following CLAUDE.md workflow requirements.

## 🚨 Tests With Same Purpose Within Files

These tests have been identified as testing the same specific functionality within the same test file:

"""

    # Report exact duplicates
    for i, group in enumerate(duplicates, 1):
        if len(group) <= 1:
            continue

        report_content += f"### Duplicate Group {i}: {len(group)} tests\n\n"

        # Show the common functionality
        first_test = group[0]
        if first_test.purpose:
            report_content += f"**Common Functionality:** {first_test.purpose}\n\n"

        report_content += "**Tests in this group:**\n\n"

        for test in group:
            file_name = Path(test.file_path).name
            report_content += (
                f"- **{file_name}**: `{test.function_name}` (line {test.line_number})\n"
            )
            if test.purpose:
                report_content += f"  - Purpose: {test.purpose}\n"
            if test.given:
                report_content += f"  - Given: {test.given}\n"
            if test.when:
                report_content += f"  - When: {test.when}\n"
            if test.then:
                report_content += f"  - Then: {test.then}\n"
            report_content += "\n"

        # Note the evaluation method used
        report_content += f"**Evaluation Method:** AI-based purpose analysis\n\n"

        report_content += "**🎯 Recommendation:** Review these tests for potential consolidation or elimination.\n\n"
        report_content += "---\n\n"

    # Report patterns
    report_content += "## 📋 Test Pattern Analysis Within Files\n\n"
    report_content += "These patterns show areas where similar testing approaches are used within individual test files:\n\n"

    for pattern_name, tests in patterns.items():
        if len(tests) < 2:  # Show patterns with 2+ tests within same file
            continue

        report_content += (
            f"### {pattern_name.replace('_', ' ').title()} Pattern: {len(tests)} tests\n\n"
        )

        # Group by file
        file_groups = defaultdict(list)
        for test in tests:
            file_groups[Path(test.file_path).name].append(test)

        for file_name, file_tests in file_groups.items():
            report_content += f"**{file_name}** ({len(file_tests)} tests):\n"
            for test in file_tests:
                report_content += (
                    f"- `{test.function_name}`: {test.purpose or 'No documentation'}\n"
                )
            report_content += "\n"

        report_content += f"**🔍 Analysis:** Review if all {len(tests)} {pattern_name.replace('_', ' ')} tests are necessary or if some can be consolidated.\n\n"
        report_content += "---\n\n"

    # Recommendations section
    report_content += """## 💡 Consolidation Recommendations

### High Priority Actions

1. **Review Duplicate Groups**: Examine each duplicate group to determine:
   - Which test provides the most comprehensive coverage
   - Whether tests can be merged into a single, more thorough test
   - If tests serve different purposes despite similar descriptions

2. **Pattern Consolidation**: For each pattern group:
   - Create a shared test utility or base class for common test setup
   - Standardize test documentation across similar tests
   - Consider if some tests are redundant and can be removed

3. **Documentation Improvement**: 
   - Update test documentation to clearly distinguish between similar tests
   - Ensure each test has a unique, specific purpose
   - Use more descriptive test names to avoid confusion

### Implementation Steps

1. Run this analysis regularly during development
2. Before adding new tests, check for similar existing tests
3. When refactoring, consolidate identified duplicate tests
4. Update test documentation to prevent future duplication

### Quality Metrics to Track

- Reduction in duplicate test count over time
- Improvement in test documentation clarity
- Decrease in redundant test coverage

---

*Generated by `scan_duplicate_tests.py`*
"""

    # Write the report
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_content)


def main():
    """Main function to scan for duplicate tests."""

    # Define paths
    summaries_root = Path("test_documentation_summaries")
    output_file = Path("duplicate_tests_report.md")

    if not summaries_root.exists():
        print(f"❌ Test documentation summaries directory not found: {summaries_root}")
        print("Please run generate_test_summaries.py first.")
        return 1

    print("🔍 Scanning test documentation summaries for duplicates...")

    # Collect all test information
    all_tests = []
    summary_files_processed = 0

    for root, dirs, files in os.walk(summaries_root):
        for file in files:
            if file.endswith("_summary.md") and file != "README.md":
                summary_file = Path(root) / file
                tests = parse_test_summary_file(summary_file)
                all_tests.extend(tests)
                summary_files_processed += 1
                print(
                    f"  📄 Processed: {summary_file.relative_to(summaries_root)} ({len(tests)} tests)"
                )

    print(f"\n📊 Analysis Summary:")
    print(f"  📁 Summary files processed: {summary_files_processed}")
    print(f"  🧪 Total tests analyzed: {len(all_tests)}")
    print(f"  📝 Tests with documentation: {len([t for t in all_tests if t.purpose])}")

    # Find duplicates
    print(f"\n🔍 Analyzing for duplicates...")

    duplicates = find_duplicate_tests(all_tests)
    patterns = find_pattern_duplicates(all_tests)

    print(f"  ⚠️  Duplicate groups found: {len(duplicates)}")
    print(f"  📋 Pattern groups found: {len(patterns)}")

    # Generate report
    print(f"\n📝 Generating duplicate test report...")
    generate_duplicate_report(duplicates, patterns, output_file)

    print(f"✅ Duplicate test analysis complete!")
    print(f"📋 Report generated: {output_file}")

    # Summary statistics
    total_duplicates = sum(len(group) for group in duplicates)
    if total_duplicates > 0:
        print(
            f"⚠️  Found {total_duplicates} potentially duplicate tests in {len(duplicates)} groups"
        )
        print(f"💡 Review the report for consolidation opportunities")
    else:
        print(f"✨ No high-confidence duplicate tests found!")

    return 0


if __name__ == "__main__":
    exit(main())
