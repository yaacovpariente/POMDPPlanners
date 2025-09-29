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

import pytest
import subprocess
import sys
from pathlib import Path


# Check if nbval pytest plugin is available
def _check_nbval_available():
    """Check if nbval pytest plugin is properly installed and available."""
    try:
        import nbval

        # Test if the --nbval option is actually available by running pytest --help
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--help"], capture_output=True, text=True, timeout=10
        )
        return "--nbval" in result.stdout
    except (ImportError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


NBVAL_AVAILABLE = _check_nbval_available()


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
        import json

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

        import json

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

        import json

        with open(notebook_path) as f:
            notebook_data = json.load(f)

        assert "cells" in notebook_data, "Notebook missing cells structure"
        assert len(notebook_data["cells"]) > 0, "Notebook has no cells"

    @pytest.mark.slow
    @pytest.mark.skipif(not NBVAL_AVAILABLE, reason="nbval not available")
    def test_basic_usage_notebook_execution(self, examples_dir):
        """Test that basic_usage.ipynb can be executed without errors.

        Purpose: Validates that basic usage notebook executes successfully

        Given: A valid basic_usage.ipynb notebook file
        When: Running the notebook with nbval validation
        Then: Notebook executes without errors and produces expected outputs

        Test type: integration
        """
        notebook_path = examples_dir / "basic_usage.ipynb"

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--nbval", str(notebook_path), "-v"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Notebook execution failed: {result.stderr}"

    @pytest.mark.slow
    @pytest.mark.skipif(not NBVAL_AVAILABLE, reason="nbval not available")
    def test_hyperparameter_tuning_notebook_execution(self, examples_dir):
        """Test that hyperparameter_tuning.ipynb can be executed without errors.

        Purpose: Validates that hyperparameter tuning notebook executes successfully

        Given: A valid hyperparameter_tuning.ipynb notebook file
        When: Running the notebook with nbval validation
        Then: Notebook executes without errors and produces optimization results

        Test type: integration
        """
        notebook_path = examples_dir / "hyperparameter_tuning.ipynb"

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--nbval", str(notebook_path), "-v"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Notebook execution failed: {result.stderr}"

    @pytest.mark.slow
    @pytest.mark.skipif(not NBVAL_AVAILABLE, reason="nbval not available")
    def test_planners_comparison_notebook_execution(self, examples_dir):
        """Test that planners_comparison.ipynb can be executed without errors.

        Purpose: Validates that planners comparison notebook executes successfully

        Given: A valid planners_comparison.ipynb notebook file
        When: Running the notebook with nbval validation
        Then: Notebook executes without errors and produces comparison results

        Test type: integration
        """
        notebook_path = examples_dir / "planners_comparison.ipynb"

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--nbval", str(notebook_path), "-v"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Notebook execution failed: {result.stderr}"

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
            import json

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
            import json

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
