# SPDX-License-Identifier: MIT

"""The three numbers that say whether a fitted model is worth planning with.

One-step prediction error is the obvious measure and the wrong one. A model can
be excellent on one step and useless over a horizon, because errors compound;
and it can be mediocre everywhere yet plan perfectly, because the planner only
needs the *ranking* of its action presets to be right. The decision-aware model
learning literature makes this precise -- what matters is the error weighted by
how much it moves the value, not the error itself. These three are the
computable stand-ins.

**Held-out log-likelihood** is the loss the fit is guaranteed against, evaluated
where the fit could not see it. It is the only one of the three the training
procedure optimizes directly, which is exactly why it is not sufficient on its
own.

**Horizon drift over claimed noise** catches the failure that quietly ruins a
risk-sensitive planner: a model whose predictions are wrong by far more than the
uncertainty it reports. A ratio near 1 means the model's error is inside its own
error bars, so a belief built from it is honest. A ratio of 5 means the planner
is confidently searching a world that does not exist.

**Preset-ranking agreement** is the closest cheap proxy for what the planner
actually does with the model. If the model and the world disagree about which
preset is best from a state, no amount of likelihood will save the decision.

Reading them together is the point. Likelihood improving while ranking agreement
stays flat means the fit is buying accuracy the planner cannot spend -- worth
knowing before committing to a long evaluation.

One shared precondition. Two of the three roll the model and the world forward
side by side, so both must speak the same state vector. When the dataset slices
a block out of a wider state -- which it should, whenever the state carries a
latent the fit must not touch -- the ``world`` passed here has to be a view over
that same block, not the full-width world. Passing the full-width world does not
raise; it silently compares vectors of different meaning.

Functions:
    held_out_log_likelihood: Mean log-density of unseen successors.
    horizon_drift_ratio: Open-loop drift over the horizon, in units of claimed noise.
    preset_ranking_agreement: How often model and world pick the same best preset.
    evaluate_model: All three, as one metrics mapping.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from POMDPPlanners.core.environment import TransitionModel
from POMDPPlanners.training.model_learning.transition_dataset import TransitionBatch


def held_out_log_likelihood(model: TransitionModel, batch: TransitionBatch) -> float:
    """Mean log-density the model assigns to successors it was not fitted on.

    Args:
        model: The fitted transition.
        batch: Held-out transitions.

    Returns:
        Mean log-density per transition, or ``nan`` for an empty batch.
    """
    if len(batch) == 0:
        return float("nan")
    total = 0.0
    for state, action, next_state in zip(batch.states, batch.actions, batch.next_states):
        total += float(model.log_probability(state, action, next_state[None, :])[0])
    return total / len(batch)


def horizon_drift_ratio(
    model: TransitionModel,
    world: Any,
    start_states: Sequence[Any],
    action_presets: Any,
    horizon: int,
    rng: np.random.Generator,
) -> float:
    """Open-loop drift over ``horizon`` steps, divided by the noise the model claims.

    Both the model and the world are rolled forward under the *same* action
    sequence from the same start state, and the gap between where they end up is
    compared against the spread the model itself predicts over that horizon. The
    ratio, not the raw drift, is the useful number: a drift of half a metre is
    fine for a model that says it is unsure to within a metre, and alarming for
    one that claims centimetres.

    Args:
        model: The fitted transition, operating on the fitted state block.
        world: The true world, operating on the same block.
        start_states: States to roll out from.
        action_presets: Table of action vectors.
        horizon: Steps to roll forward -- use the planner's own depth.
        rng: Source of the action sequence and the model's own sampling.

    Returns:
        Mean ratio across start states. Values near 1 mean the model's error sits
        inside its own error bars; well above 1 means it is overconfident.
    """
    presets = np.asarray(action_presets, dtype=float)
    ratios: List[float] = []
    for start in start_states:
        actions = presets[rng.integers(len(presets), size=horizon)]
        world_state = np.asarray(start, dtype=float).ravel()
        model_state = world_state.copy()
        # The model's own claimed spread, accumulated as the standard deviation
        # of the per-step samples it draws along the way.
        claimed = 0.0
        for action in actions:
            world_state = np.asarray(
                world.sample_next_state(world_state, action), dtype=float
            ).ravel()
            draws = np.atleast_2d(model.sample_next_state(model_state, action, n_samples=16))
            claimed += float(np.mean(np.std(draws, axis=0)))
            model_state = draws[int(rng.integers(draws.shape[0]))]
        drift = float(np.linalg.norm(world_state - model_state))
        ratios.append(drift / max(claimed, 1e-9))
    return float(np.mean(ratios)) if ratios else float("nan")


def preset_ranking_agreement(
    model: TransitionModel,
    world: Any,
    reward_model: Any,
    start_states: Sequence[Any],
    action_presets: Any,
    num_samples: int = 32,
) -> float:
    """Fraction of start states where model and world rank the same preset best.

    Each preset is scored by its mean one-step reward under the model and under
    the world, using the *same* reward model for both -- the objective is our
    design, not something the fit is allowed to change, so holding it fixed is
    what makes the comparison about dynamics.

    Args:
        model: The fitted transition.
        world: The true world.
        reward_model: Anything exposing ``reward(state, action, next_state)``.
        start_states: States to compare rankings at.
        action_presets: Table of action vectors.
        num_samples: Successors drawn per preset when averaging the reward.

    Returns:
        Agreement in ``[0, 1]``, or ``nan`` when no start states are given.
    """
    presets = np.asarray(action_presets, dtype=float)
    if len(start_states) == 0:
        return float("nan")
    agreements: List[bool] = []
    for start in start_states:
        state = np.asarray(start, dtype=float).ravel()
        model_scores = [
            _mean_reward(reward_model, state, action, model, num_samples) for action in presets
        ]
        world_scores = [
            _mean_reward(reward_model, state, action, world, num_samples) for action in presets
        ]
        agreements.append(int(np.argmax(model_scores)) == int(np.argmax(world_scores)))
    return float(np.mean(agreements))


def evaluate_model(
    model: TransitionModel,
    world: Any,
    holdout: TransitionBatch,
    action_presets: Any,
    horizon: int,
    reward_model: Optional[Any] = None,
    num_start_states: int = 16,
    seed: int = 0,
) -> Dict[str, float]:
    """Compute all three diagnostics for one round's model.

    Args:
        model: The fitted transition.
        world: The true world.
        holdout: Held-out transitions; their source states double as the start
            states for the other two diagnostics, so every number describes the
            same region of the state space.
        action_presets: Table of action vectors.
        horizon: Steps for the drift measurement -- the planner's own depth.
        reward_model: Reward used for the ranking check. ``None`` skips it.
        num_start_states: How many held-out states to roll out from.
        seed: Seed for the action sequences and sampling.

    Returns:
        A mapping with ``held_out_log_likelihood``, ``horizon_drift_ratio`` and,
        when a reward model is supplied, ``preset_ranking_agreement``.
    """
    rng = np.random.default_rng(seed)
    metrics = {"held_out_log_likelihood": held_out_log_likelihood(model, holdout)}
    if len(holdout) == 0:
        metrics["horizon_drift_ratio"] = float("nan")
        return metrics

    count = min(num_start_states, len(holdout))
    chosen = rng.choice(len(holdout), size=count, replace=False)
    start_states = [holdout.states[index] for index in chosen]
    metrics["horizon_drift_ratio"] = horizon_drift_ratio(
        model, world, start_states, action_presets, horizon, rng
    )
    if reward_model is not None:
        metrics["preset_ranking_agreement"] = preset_ranking_agreement(
            model, world, reward_model, start_states, action_presets
        )
    return metrics


def _mean_reward(
    reward_model: Any,
    state: np.ndarray,
    action: np.ndarray,
    transition: Any,
    num_samples: int,
) -> float:
    """Mean one-step reward of ``action`` at ``state`` under one transition."""
    successors = np.atleast_2d(transition.sample_next_state(state, action, num_samples))
    return float(
        np.mean([reward_model.reward(state, action, successor) for successor in successors])
    )
