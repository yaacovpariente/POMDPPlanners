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

Functions:
    block_indices: Flat indices of named channels in a schema.
    collect_random_preset_episode: One held-random-action exploration rollout.
    evaluate_model: The three diagnostics for a fitted model.
"""

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
from POMDPPlanners.training.model_learning.transition_dataset import (
    TransitionBatch,
    TransitionDataset,
    block_indices,
)

__all__ = [
    "DAggerModelTrainer",
    "GaussianMLP",
    "LinearGaussianLearner",
    "ProbabilisticEnsembleLearner",
    "ProbabilisticEnsembleTransition",
    "RoundResult",
    "TransitionBatch",
    "TransitionDataset",
    "TransitionModelLearner",
    "block_indices",
    "collect_random_preset_episode",
    "evaluate_model",
    "held_out_log_likelihood",
    "horizon_drift_ratio",
    "preset_ranking_agreement",
]
