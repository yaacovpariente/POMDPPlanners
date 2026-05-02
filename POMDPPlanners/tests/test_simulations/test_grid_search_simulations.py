"""Tests for GridSearchOptimizer and the underlying GridSearchTuningTask.

Mirrors the structure and fixture conventions of
``test_hyper_parameter_tuning_simulations.py`` (Tiger env, real belief,
small num_episodes/num_steps) so the two optimizer implementations can be
exercised side-by-side without divergence.
"""

# pylint: disable=protected-access  # Tests inspect private metadata helpers.

import shutil
import tempfile
from pathlib import Path

import pytest

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.simulation import (
    CategoricalHyperParameter,
    NumericalGridSpec,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParamPlannerConfig,
    HyperParameterOptimizationDirection,
    HyperParameterRunParams,
    NumericalHyperParameter,
    OptimizedPolicyResult,
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling import (
    SparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.simulations.grid_search_simulations import GridSearchOptimizer
from POMDPPlanners.simulations.simulations_deployment.tasks.grid_search_tuning_task import (
    GridSearchTuningTask,
)

pytestmark = [pytest.mark.slow]


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for caching test artifacts."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def tiger_env():
    """Tiger POMDP with the standard 0.95 discount factor."""
    return TigerPOMDP(discount_factor=0.95)


@pytest.fixture
def tiger_belief(tiger_env):
    """Initial belief over Tiger's two states with 10 particles."""
    return get_initial_belief(tiger_env, n_particles=10)


@pytest.fixture
def sparse_sampling_grid_config(tiger_env, tiger_belief):
    """A grid-compatible config: SparseSampling on Tiger with two grid axes.

    Two axes, two values each → 4 combinations. Small ``num_episodes`` and
    ``num_steps`` keep tests fast while still hitting every code path.
    """
    planner_config = HyperParamPlannerConfig(
        policy_cls=SparseSamplingDiscreteActionsPlanner,
        hyper_parameters=[
            CategoricalHyperParameter(choices=[1, 2], name="branching_factor"),
            CategoricalHyperParameter(choices=[1, 2], name="depth"),
        ],
        constant_parameters={},
    )
    return HyperParameterRunParams(
        environment=tiger_env,
        belief=tiger_belief,
        hyper_param_planner_config=planner_config,
        num_episodes=2,
        num_steps=3,
        n_trials=1,  # Ignored by grid search; keeping non-zero satisfies validation.
        parameters_to_optimize=[("average_return", HyperParameterOptimizationDirection.MAXIMIZE)],
    )


class TestGridSearchTuningTaskRun:
    """Run-time behavior of the inner GridSearchTuningTask."""

    def test_task_returns_optimized_policy_result_with_winning_combo(
        self, tiger_env, tiger_belief, temp_cache_dir
    ):
        """A direct task.run() returns an OptimizedPolicyResult for the winning combo.

        Purpose: Validates the task's end-to-end happy path

        Given: A 2-axis × 2-values grid (4 combos) on Tiger with 2 episodes
        When: task.run() is called
        Then: An OptimizedPolicyResult is returned whose chosen_hyper_parameters
            keys match the grid axes and whose policy was instantiated from one
            of the grid combinations

        Test type: integration
        """
        task = GridSearchTuningTask(
            environment=tiger_env,
            belief=tiger_belief,
            policy_cls=SparseSamplingDiscreteActionsPlanner,
            hyper_parameters=[
                CategoricalHyperParameter(choices=[1, 2], name="branching_factor"),
                CategoricalHyperParameter(choices=[1, 2], name="depth"),
            ],
            constant_parameters={},
            num_episodes=2,
            num_steps=3,
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
            cache_dir=temp_cache_dir,
            n_jobs=1,
            base_seed=42,
        )

        result = task.run()

        assert result is not None, "task.run() should not fail on a small Tiger grid"
        assert isinstance(result, OptimizedPolicyResult)
        assert set(result.chosen_hyper_parameters.keys()) == {"branching_factor", "depth"}
        assert result.chosen_hyper_parameters["branching_factor"] in {1, 2}
        assert result.chosen_hyper_parameters["depth"] in {1, 2}
        assert "average_return" in result.optimized_metric_values

    def test_metadata_reports_n_combos_and_winning_index(
        self, tiger_env, tiger_belief, temp_cache_dir
    ):
        """get_optimization_metadata reflects 4 combos and a valid winner index.

        Purpose: Validates the metadata shape used by base optimizer's MLflow logging

        Given: A 2x2 grid evaluated end-to-end
        When: get_optimization_metadata is called after run()
        Then: n_trials=4, best_trial_number is in [0, 3], and
            best_trial_metrics contains the average_return key

        Test type: integration
        """
        task = GridSearchTuningTask(
            environment=tiger_env,
            belief=tiger_belief,
            policy_cls=SparseSamplingDiscreteActionsPlanner,
            hyper_parameters=[
                CategoricalHyperParameter(choices=[1, 2], name="branching_factor"),
                CategoricalHyperParameter(choices=[1, 2], name="depth"),
            ],
            constant_parameters={},
            num_episodes=2,
            num_steps=3,
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
            cache_dir=temp_cache_dir,
            n_jobs=1,
            base_seed=42,
        )
        task.run()

        metadata = task.get_optimization_metadata()
        assert metadata is not None
        assert metadata["n_trials"] == 4
        assert metadata["best_trial_number"] in {0, 1, 2, 3}
        assert "average_return" in metadata["best_trial_metrics"]
        assert metadata["config_id"] == task.get_config_id()


class TestGridSearchOptimizerHappyPath:
    """End-to-end optimizer behavior with a real environment."""

    def test_optimize_returns_one_result_per_config(
        self, sparse_sampling_grid_config, temp_cache_dir
    ):
        """A single config yields a single OptimizedPolicyResult.

        Purpose: Validates the optimize() contract aligns with HyperParameterOptimizer

        Given: A single HyperParameterRunParams with a 4-combo grid
        When: GridSearchOptimizer.optimize is called
        Then: Returns a list of length 1 with an OptimizedPolicyResult whose
            chosen_hyper_parameters cover both grid axes

        Test type: integration
        """
        optimizer = GridSearchOptimizer(
            cache_dir_path=temp_cache_dir, experiment_name="grid_test", n_jobs=1
        )
        try:
            results = optimizer.optimize([sparse_sampling_grid_config])
        finally:
            optimizer.cleanup()

        assert len(results) == 1
        assert isinstance(results[0], OptimizedPolicyResult)
        assert set(results[0].chosen_hyper_parameters.keys()) == {"branching_factor", "depth"}

    def test_optimize_handles_numerical_grid_spec(self, tiger_env, tiger_belief, temp_cache_dir):
        """A NumericalGridSpec axis is iterated over its expanded values.

        Purpose: Validates that the new NumericalGridSpec type integrates end-to-end

        Given: A grid using NumericalGridSpec(low=1, high=2, n_points=2) for branching_factor
        When: optimize() is called
        Then: The winning combo's branching_factor is one of the two expanded values

        Test type: integration
        """
        planner_config = HyperParamPlannerConfig(
            policy_cls=SparseSamplingDiscreteActionsPlanner,
            hyper_parameters=[
                NumericalGridSpec(1, 2, 2, "branching_factor", "linear"),
                CategoricalHyperParameter(choices=[1], name="depth"),
            ],
            constant_parameters={},
        )
        config = HyperParameterRunParams(
            environment=tiger_env,
            belief=tiger_belief,
            hyper_param_planner_config=planner_config,
            num_episodes=2,
            num_steps=3,
            n_trials=1,
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
        )

        optimizer = GridSearchOptimizer(
            cache_dir_path=temp_cache_dir, experiment_name="grid_numeric_test", n_jobs=1
        )
        try:
            results = optimizer.optimize([config])
        finally:
            optimizer.cleanup()

        assert len(results) == 1
        chosen = results[0].chosen_hyper_parameters["branching_factor"]
        assert chosen == pytest.approx(1.0) or chosen == pytest.approx(2.0)


class TestGridSearchOptimizerValidation:
    """Per-optimizer hyperparameter type validation."""

    def test_rejects_bare_numerical_hyperparameter(self, tiger_env, tiger_belief, temp_cache_dir):
        """A NumericalHyperParameter (continuous range) is rejected at task creation.

        Purpose: Validates the per-optimizer type guard added to
            GridSearchOptimizer._validate_grid_hyperparameters

        Given: A config whose hyperparameters include a bare
            NumericalHyperParameter (Optuna-style continuous range)
        When: optimize() is called
        Then: A TypeError is raised whose message names NumericalGridSpec

        Test type: unit
        """
        planner_config = HyperParamPlannerConfig(
            policy_cls=SparseSamplingDiscreteActionsPlanner,
            hyper_parameters=[
                NumericalHyperParameter(1, 3, "branching_factor"),
            ],
            constant_parameters={"depth": 1},
        )
        config = HyperParameterRunParams(
            environment=tiger_env,
            belief=tiger_belief,
            hyper_param_planner_config=planner_config,
            num_episodes=2,
            num_steps=3,
            n_trials=1,
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
        )

        optimizer = GridSearchOptimizer(
            cache_dir_path=temp_cache_dir, experiment_name="grid_reject_test", n_jobs=1
        )
        try:
            with pytest.raises((TypeError, RuntimeError), match="NumericalGridSpec"):
                optimizer.optimize([config])
        finally:
            optimizer.cleanup()


class TestGridSearchMatchedPairsDeterminism:
    """Matched-pairs seeding makes runs reproducible."""

    def test_two_consecutive_runs_pick_same_winner(
        self, sparse_sampling_grid_config, temp_cache_dir
    ):
        """Repeated optimize() with the same config picks the same combination.

        Purpose: Validates that base_seed=42 + matched-pairs seeding gives
            deterministic winners on identical inputs

        Given: The same HyperParameterRunParams optimized twice with separate optimizers
        When: optimize() is called twice
        Then: Both runs return the same chosen_hyper_parameters

        Test type: integration
        """
        optimizer_a = GridSearchOptimizer(
            cache_dir_path=temp_cache_dir / "run_a",
            experiment_name="grid_det_a",
            n_jobs=1,
        )
        try:
            results_a = optimizer_a.optimize([sparse_sampling_grid_config])
        finally:
            optimizer_a.cleanup()

        optimizer_b = GridSearchOptimizer(
            cache_dir_path=temp_cache_dir / "run_b",
            experiment_name="grid_det_b",
            n_jobs=1,
        )
        try:
            results_b = optimizer_b.optimize([sparse_sampling_grid_config])
        finally:
            optimizer_b.cleanup()

        assert (
            results_a[0].chosen_hyper_parameters == results_b[0].chosen_hyper_parameters
        ), "Matched-pairs seeding should give the same winner across runs"
