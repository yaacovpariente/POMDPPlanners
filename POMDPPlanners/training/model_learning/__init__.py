# SPDX-License-Identifier: MIT

"""Fitting a planner's transition model from rollouts, iteratively.

Every planner needs a generative model to search, and a forward-only world -- a
live simulator -- cannot serve as one. Where the model has been written by hand
and calibrated, its error against the real simulator is unmeasured; where it is
fitted from a fixed batch of rollouts, it is accurate only where those rollouts
went, which is not where the planner goes. This package addresses the second
problem, and gives you the measurement for the first.

The loop is the one from Ross & Bagnell (ICML 2012): plan with the current model,
record what the planner actually did in the true world, aggregate it with
everything collected before, refit, repeat. Its guarantee survives the true
system being outside the model class, which is the realistic case.

See :mod:`~POMDPPlanners.training.model_learning.dagger_trainer` for the three
details that decide whether the loop works, and
:mod:`~POMDPPlanners.training.model_learning.diagnostics` for why held-out
prediction error alone will not tell you whether a model is worth planning with.

None of those diagnostics is the measure the algorithm is judged by. That is the
return of the planner searching each round's model in the true world, against
the data that round cost, with the non-iterative baseline drawn on the same
axes -- see
:mod:`~POMDPPlanners.training.model_learning.control_evaluation` and
:mod:`~POMDPPlanners.training.model_learning.curve_comparison`.

Classes:
    TransitionDataset: Transitions accumulated across rounds, sliced to named channels.
    TransitionBatch: States, actions and next states as parallel arrays.
    TransitionModelLearner: Interface for a fitting procedure.
    LinearGaussianLearner: Ridge fit of a linear-Gaussian transition.
    ProbabilisticEnsembleLearner: Gaussian-likelihood fit of an ensemble.
    ProbabilisticEnsembleTransition: The ensemble, usable as a transition model.
    GaussianMLP: One ensemble member.
    DAggerModelTrainer: The round loop.
    RoundResult: What one round produced.
    ControlPoint: One round's return, with the data it cost.
    LearningCurve: One method's points for one seed.
    AggregatedCurve: A method's curve averaged across seeds.

Functions:
    block_indices: Flat indices of named channels in a schema.
    collect_random_preset_episode: One held-random-action exploration rollout.
    evaluate_model: The three diagnostics for a fitted model.
    evaluate_control: Score one round's model by running the planner with it.
    aggregate_curves: Average one method's curves across seeds.
    best_point: The round a curve should be reported at.
    run_learning_curves: Run every method at every seed and collect the curves.
    curves_by_method: Group curves by method name.
    save_learning_curves: Write curves to JSON.
    load_learning_curves: Read curves back.
"""

from POMDPPlanners.training.model_learning.control_evaluation import (
    AggregatedCurve,
    ControlPoint,
    LearningCurve,
    aggregate_curves,
    best_point,
    evaluate_control,
    load_learning_curves,
    save_learning_curves,
)
from POMDPPlanners.training.model_learning.curve_comparison import (
    DEFAULT_METHODS,
    curves_by_method,
    run_learning_curves,
)
from POMDPPlanners.training.model_learning.dagger_trainer import (
    DAggerModelTrainer,
    RoundResult,
)
from POMDPPlanners.training.model_learning.diagnostics import (
    evaluate_model,
    held_out_log_likelihood,
    horizon_drift_ratio,
    preset_ranking_agreement,
)
from POMDPPlanners.training.model_learning.ensemble_transition import (
    GaussianMLP,
    ProbabilisticEnsembleTransition,
)
from POMDPPlanners.training.model_learning.exploration import (
    collect_random_preset_episode,
)
from POMDPPlanners.training.model_learning.learners import (
    LinearGaussianLearner,
    ProbabilisticEnsembleLearner,
    TransitionModelLearner,
)
from POMDPPlanners.training.model_learning.reporting import (
    curve_table_markdown,
    metrics_table_csv,
    metrics_table_markdown,
    run_metrics,
    round_table_csv,
    round_table_markdown,
    write_reports,
)
from POMDPPlanners.training.model_learning.tracking import (
    MLflowModelLearningTracker,
    ModelLearningTracker,
    curve_summaries,
    load_round_models,
    log_study_comparison,
)
from POMDPPlanners.training.model_learning.transition_dataset import (
    TransitionBatch,
    TransitionDataset,
    block_indices,
)

__all__ = [
    "DEFAULT_METHODS",
    "AggregatedCurve",
    "ControlPoint",
    "DAggerModelTrainer",
    "GaussianMLP",
    "LearningCurve",
    "LinearGaussianLearner",
    "MLflowModelLearningTracker",
    "ModelLearningTracker",
    "ProbabilisticEnsembleLearner",
    "ProbabilisticEnsembleTransition",
    "RoundResult",
    "TransitionBatch",
    "TransitionDataset",
    "TransitionModelLearner",
    "aggregate_curves",
    "best_point",
    "block_indices",
    "collect_random_preset_episode",
    "curve_summaries",
    "curve_table_markdown",
    "curves_by_method",
    "evaluate_control",
    "evaluate_model",
    "load_learning_curves",
    "load_round_models",
    "metrics_table_csv",
    "metrics_table_markdown",
    "run_metrics",
    "log_study_comparison",
    "round_table_csv",
    "round_table_markdown",
    "run_learning_curves",
    "save_learning_curves",
    "write_reports",

    "held_out_log_likelihood",
    "horizon_drift_ratio",
    "preset_ranking_agreement",
]
