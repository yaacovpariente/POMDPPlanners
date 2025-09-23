#!/usr/bin/env python3
"""Test script for Jupyter notebook examples.

This script validates that all notebook examples in docs/examples/
can be executed without errors. It uses nbval for notebook testing.

Usage:
    python test_notebooks.py
    pytest --nbval docs/examples/
"""

import subprocess
import sys
from pathlib import Path


def test_notebook_execution(notebook_path: Path) -> bool:
    """Test if a notebook can be executed without errors.

    Args:
        notebook_path: Path to the Jupyter notebook file

    Returns:
        True if notebook executes successfully, False otherwise
    """
    try:
        # Use nbval to test the notebook
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--nbval", str(notebook_path), "-v"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            print(f"✅ {notebook_path.name} - PASSED")
            return True
        else:
            print(f"❌ {notebook_path.name} - FAILED")
            print(f"   Error output: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ {notebook_path.name} - ERROR: {e}")
        return False


def main():
    """Main function to test all notebook examples."""
    print("Testing Jupyter notebook examples...")

    # Find all notebooks in docs/examples/
    examples_dir = Path("docs/examples")
    if not examples_dir.exists():
        print(f"❌ Examples directory not found: {examples_dir}")
        sys.exit(1)

    notebook_files = list(examples_dir.glob("*.ipynb"))

    if not notebook_files:
        print(f"❌ No notebook files found in {examples_dir}")
        sys.exit(1)

    print(f"Found {len(notebook_files)} notebook(s) to test:")
    for nb in notebook_files:
        print(f"  - {nb.name}")

    print("\nRunning tests...")

    # Test each notebook
    passed = 0
    failed = 0

    for notebook in sorted(notebook_files):
        success = test_notebook_execution(notebook)
        if success:
            passed += 1
        else:
            failed += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"SUMMARY: {passed} passed, {failed} failed")

    if failed > 0:
        print("\nFailed notebooks need attention before they can be used in documentation.")
        sys.exit(1)
    else:
        print("\nAll notebook examples are working correctly! ✅")


if __name__ == "__main__":
    main()
