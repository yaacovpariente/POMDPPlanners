"""Tests for ExperimentVisualizer abstract base class."""

import pytest

from POMDPPlanners.core.simulation import ExperimentVisualizer


def test_experiment_visualizer_is_abstract():
    """ExperimentVisualizer cannot be instantiated without overriding render.

    Purpose: Validates that ExperimentVisualizer enforces its abstract contract.

    Given: The ExperimentVisualizer abstract base class.
    When: Direct instantiation is attempted.
    Then: A TypeError is raised because ``render`` is abstract.

    Test type: unit
    """
    with pytest.raises(TypeError):
        ExperimentVisualizer()  # type: ignore[abstract]


def test_experiment_visualizer_requires_render_method():
    """A concrete subclass must implement render to be instantiable.

    Purpose: Validates that subclasses must implement the abstract ``render`` method.

    Given: A subclass that omits the ``render`` method.
    When: Instantiation is attempted.
    Then: A TypeError is raised. A subclass that does implement ``render``
        instantiates successfully.

    Test type: unit
    """

    class IncompleteVisualizer(ExperimentVisualizer):  # pylint: disable=abstract-method
        pass

    with pytest.raises(TypeError):
        IncompleteVisualizer()  # type: ignore[abstract]

    class CompleteVisualizer(ExperimentVisualizer):
        def render(
            self, env_name, environment, policy_results, policies, output_dir, cache_visualizations
        ):
            return output_dir

    assert isinstance(CompleteVisualizer(), ExperimentVisualizer)
