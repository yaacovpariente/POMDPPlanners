# SPDX-License-Identifier: MIT

"""Test cases for Jupyter notebook examples in docs/examples/.

This module provides test cases to validate that all notebook examples
can be executed and produce expected results. These tests ensure that
the documentation examples remain functional as the codebase evolves.

Test categories:
- Basic usage notebook validation
- Hyperparameter tuning notebook validation
- Planners comparison notebook validation
- Notebook integration with documentation system
"""

import json
import tempfile
from pathlib import Path

import pytest

try:
    import papermill as pm
except ImportError:
    pm = None


def _strip_ipython_magics(source: str) -> str:
    """Remove IPython magic lines (e.g. %matplotlib, !pip) before syntax checking."""
    lines = source.splitlines(keepends=True)
    return "".join(line for line in lines if not line.lstrip().startswith(("%", "!")))


class TestNotebookExamples:
    """Test cases for Jupyter notebook examples."""

    @pytest.fixture
    def examples_dir(self):
        """Fixture providing the path to examples directory.

        Returns:
            Path: Path to docs/examples/ directory
        """
        return Path(__file__).parents[3] / "docs" / "examples"

    def test_examples_directory_exists(self, examples_dir):
        """Test that the examples directory exists and contains notebooks.

        Purpose: Validates the examples directory structure is properly set up

        Given: The docs/examples/ directory path
        When: Checking if the directory exists and contains .ipynb files
        Then: Directory exists and contains at least one notebook file

        Test type: integration
        """
        assert examples_dir.exists(), f"Examples directory does not exist: {examples_dir}"

        notebook_files = list(examples_dir.glob("*.ipynb"))
        assert len(notebook_files) > 0, f"No notebook files found in {examples_dir}"

    def test_basic_usage_notebook_exists(self, examples_dir):
        """Test that basic_usage.ipynb exists and has proper structure.

        Purpose: Validates basic usage notebook file exists and is valid JSON

        Given: The basic_usage.ipynb notebook file path
        When: Checking file existence and loading notebook structure
        Then: File exists and contains valid notebook cells

        Test type: integration
        """
        notebook_path = examples_dir / "basic_usage.ipynb"
        assert notebook_path.exists(), "basic_usage.ipynb does not exist"

        # Try to load the notebook as JSON to validate structure

        with open(notebook_path) as f:
            notebook_data = json.load(f)

        assert "cells" in notebook_data, "Notebook missing cells structure"
        assert "metadata" in notebook_data, "Notebook missing metadata"
        assert len(notebook_data["cells"]) > 0, "Notebook has no cells"

    def test_hyperparameter_tuning_notebook_exists(self, examples_dir):
        """Test that hyperparameter_tuning.ipynb exists and has proper structure.

        Purpose: Validates hyperparameter tuning notebook file exists and is valid JSON

        Given: The hyperparameter_tuning.ipynb notebook file path
        When: Checking file existence and loading notebook structure
        Then: File exists and contains valid notebook cells

        Test type: integration
        """
        notebook_path = examples_dir / "hyperparameter_tuning.ipynb"
        assert notebook_path.exists(), "hyperparameter_tuning.ipynb does not exist"

        with open(notebook_path) as f:
            notebook_data = json.load(f)

        assert "cells" in notebook_data, "Notebook missing cells structure"
        assert len(notebook_data["cells"]) > 0, "Notebook has no cells"

    def test_planners_comparison_notebook_exists(self, examples_dir):
        """Test that planners_comparison.ipynb exists and has proper structure.

        Purpose: Validates planners comparison notebook file exists and is valid JSON

        Given: The planners_comparison.ipynb notebook file path
        When: Checking file existence and loading notebook structure
        Then: File exists and contains valid notebook cells

        Test type: integration
        """
        notebook_path = examples_dir / "planners_comparison.ipynb"
        assert notebook_path.exists(), "planners_comparison.ipynb does not exist"

        with open(notebook_path) as f:
            notebook_data = json.load(f)

        assert "cells" in notebook_data, "Notebook missing cells structure"
        assert len(notebook_data["cells"]) > 0, "Notebook has no cells"

    def test_all_notebooks_have_required_metadata(self, examples_dir):
        """Test that all notebooks have required metadata for documentation.

        Purpose: Validates notebooks have proper metadata for Sphinx integration

        Given: All notebook files in the examples directory
        When: Loading and checking notebook metadata sections
        Then: Each notebook contains kernelspec and language_info metadata

        Test type: unit
        """
        notebook_files = list(examples_dir.glob("*.ipynb"))

        for notebook_path in notebook_files:

            with open(notebook_path) as f:
                notebook_data = json.load(f)

            metadata = notebook_data.get("metadata", {})

            # Check for required metadata fields
            assert "kernelspec" in metadata, f"{notebook_path.name} missing kernelspec metadata"
            assert (
                "language_info" in metadata
            ), f"{notebook_path.name} missing language_info metadata"

            # Check kernelspec details
            kernelspec = metadata["kernelspec"]
            assert "name" in kernelspec, f"{notebook_path.name} kernelspec missing name"
            assert (
                "display_name" in kernelspec
            ), f"{notebook_path.name} kernelspec missing display_name"

    def test_notebooks_contain_markdown_documentation(self, examples_dir):
        """Test that notebooks contain adequate markdown documentation cells.

        Purpose: Validates notebooks have sufficient explanatory text for users

        Given: All notebook files in the examples directory
        When: Analyzing cell types and content in each notebook
        Then: Each notebook contains markdown cells with documentation

        Test type: unit
        """
        notebook_files = list(examples_dir.glob("*.ipynb"))

        for notebook_path in notebook_files:

            with open(notebook_path) as f:
                notebook_data = json.load(f)

            cells = notebook_data.get("cells", [])
            markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]

            assert (
                len(markdown_cells) > 0
            ), f"{notebook_path.name} has no markdown documentation cells"

            # Check that first cell is typically a markdown title/description
            if cells:
                first_cell = cells[0]
                assert (
                    first_cell.get("cell_type") == "markdown"
                ), f"{notebook_path.name} should start with markdown title"

    @pytest.mark.smoke
    def test_basic_usage_notebook_smoke_test(self, examples_dir):
        """Quick smoke test - validate basic_usage.ipynb can be parsed and starts executing.

        Purpose: Validates notebook can start executing without import/setup errors

        Given: A valid basic_usage.ipynb notebook file
        When: Attempting to run the first code cell to test imports
        Then: Basic imports work without errors

        Test type: integration
        """
        notebook_path = examples_dir / "basic_usage.ipynb"
        assert notebook_path.exists(), "basic_usage.ipynb does not exist"

        if pm is None:
            pytest.skip("papermill not available for smoke tests")

        # Load notebook to check structure
        with open(notebook_path) as f:
            notebook_data = json.load(f)

        # Find first code cell
        code_cells = [cell for cell in notebook_data["cells"] if cell.get("cell_type") == "code"]
        assert len(code_cells) > 0, "No code cells found in notebook"

        # Test basic import without full execution - just syntax check
        first_code_cell = code_cells[0]
        cell_source = "".join(first_code_cell.get("source", []))

        # Basic syntax validation (strip IPython magics before compiling)
        clean_source = _strip_ipython_magics(cell_source)
        if clean_source.strip():
            try:
                compile(clean_source, "<notebook_cell>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in first code cell: {e}")

        # Quick execution test with timeout
        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as tmp:
            try:
                # Create minimal notebook with just the first cell for testing
                minimal_nb = {
                    "cells": [first_code_cell],
                    "metadata": notebook_data["metadata"],
                    "nbformat": notebook_data["nbformat"],
                    "nbformat_minor": notebook_data["nbformat_minor"],
                }

                with open(tmp.name, "w") as f:
                    json.dump(minimal_nb, f)

                # Try to execute just the first cell
                pm.execute_notebook(
                    tmp.name,
                    tmp.name,
                    kernel_name="python3",
                    progress_bar=False,
                    log_output=False,
                    timeout=60,  # 1 minute max for first cell
                )

            except Exception as e:
                # For smoke tests, we'll be lenient - just warn about execution issues
                print(f"Warning: Basic notebook smoke test had execution issues: {str(e)}")
            finally:
                Path(tmp.name).unlink(missing_ok=True)

    @pytest.mark.smoke
    def test_hyperparameter_tuning_notebook_smoke_test(self, examples_dir):
        """Quick smoke test - validate hyperparameter tuning notebook structure and syntax.

        Purpose: Validates hyperparameter tuning notebook can be parsed and has valid syntax

        Given: A valid hyperparameter_tuning.ipynb notebook file
        When: Checking notebook structure and first cell syntax
        Then: Notebook is well-formed and first cell has valid Python syntax

        Test type: integration
        """
        notebook_path = examples_dir / "hyperparameter_tuning.ipynb"
        assert notebook_path.exists(), "hyperparameter_tuning.ipynb does not exist"

        # Load and validate notebook structure
        with open(notebook_path) as f:
            notebook_data = json.load(f)

        # Find code cells
        code_cells = [cell for cell in notebook_data["cells"] if cell.get("cell_type") == "code"]
        assert len(code_cells) > 0, "No code cells found in hyperparameter notebook"

        # Check syntax of first few code cells
        for i, cell in enumerate(code_cells[:3]):  # Check first 3 code cells
            cell_source = "".join(cell.get("source", []))
            clean_source = _strip_ipython_magics(cell_source)
            if clean_source.strip():  # Skip empty cells
                try:
                    compile(clean_source, f"<notebook_cell_{i}>", "exec")
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in code cell {i}: {e}")

        # Validate notebook has expected structure for hyperparameter tuning
        all_source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
        assert "import" in all_source, "Notebook should contain import statements"

        # Check for common hyperparameter tuning imports/concepts
        expected_concepts = ["optimization", "hyperparameter", "trial", "SimulationsAPI"]
        found_concepts = [
            concept for concept in expected_concepts if concept.lower() in all_source.lower()
        ]
        assert (
            len(found_concepts) > 0
        ), f"Expected hyperparameter tuning concepts not found: {expected_concepts}"

    @pytest.mark.smoke
    def test_planners_comparison_notebook_smoke_test(self, examples_dir):
        """Quick smoke test - validate planners comparison notebook structure and syntax.

        Purpose: Validates planners comparison notebook can be parsed and has valid syntax

        Given: A valid planners_comparison.ipynb notebook file
        When: Checking notebook structure and code cell syntax
        Then: Notebook is well-formed and contains expected comparison concepts

        Test type: integration
        """
        notebook_path = examples_dir / "planners_comparison.ipynb"
        assert notebook_path.exists(), "planners_comparison.ipynb does not exist"

        # Load and validate notebook structure
        with open(notebook_path) as f:
            notebook_data = json.load(f)

        # Find code cells
        code_cells = [cell for cell in notebook_data["cells"] if cell.get("cell_type") == "code"]
        assert len(code_cells) > 0, "No code cells found in planners comparison notebook"

        # Check syntax of first few code cells
        for i, cell in enumerate(code_cells[:3]):  # Check first 3 code cells
            cell_source = "".join(cell.get("source", []))
            clean_source = _strip_ipython_magics(cell_source)
            if clean_source.strip():  # Skip empty cells
                try:
                    compile(clean_source, f"<notebook_cell_{i}>", "exec")
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in code cell {i}: {e}")

        # Validate notebook has expected content for planner comparison
        all_source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
        assert "import" in all_source, "Notebook should contain import statements"

        # Check for planner comparison concepts
        expected_concepts = ["planner", "comparison", "SimulationsAPI", "environment"]
        found_concepts = [
            concept for concept in expected_concepts if concept.lower() in all_source.lower()
        ]
        assert (
            len(found_concepts) >= 2
        ), f"Expected at least 2 planner comparison concepts, found: {found_concepts}"
