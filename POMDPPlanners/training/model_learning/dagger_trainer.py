# SPDX-License-Identifier: MIT

"""The round loop: plan with the current model, record what happened, refit.

A model fitted once on a fixed batch of rollouts is accurate where that batch was
collected and unconstrained everywhere else. The planner then does the one thing
guaranteed to expose that: it seeks out whatever the model says is cheap, which
includes regions no rollout visited and the model is optimistic about by
accident. Performance can be arbitrarily bad while training error is tiny. Ross
& Bagnell (ICML 2012) prove the fix -- collect under the learner's own policy,
aggregate, refit -- holds even when the true system is outside the model class,
which is always our situation.

Three details decide whether the loop works, and none of them is optional.

**Half the data comes from exploration, every round.** The paper singles this out
as its difference from earlier iterative methods: without a fixed balance,
exploration rows become a shrinking fraction of the buffer, the fit stops paying
for them, and the loop settles into a model that is excellent along the current
trajectory and blind beside it. Earlier methods that used exploration data only
in the first round plateau, measurably.

**The fit aggregates.** Round n refits on every transition from rounds 1..n, not
on round n's alone. That is the "no-regret" part; dropping it turns the loop into
a sequence of unrelated fits that can oscillate.

**Every round is kept.** The guarantee is on the best model in the sequence, or
their mixture -- not on the last one. Keeping only the final model discards the
object the theory is about.

Two things the loop deliberately does not do. It does not learn the reward: the
objective is our design, not a fact about the world, and holding it fixed is what
makes a model comparison about dynamics. And it does not claim its guarantee
covers a risk-sensitive objective -- the bound is stated for expected discounted
cost, and a nested CVaR is not that.

Classes:
    RoundResult: What one round produced.
    DAggerModelTrainer: The round loop.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from POMDPPlanners.core.environment import TransitionModel
from POMDPPlanners.training.model_learning.diagnostics import evaluate_model
from POMDPPlanners.training.model_learning.exploration import (
    DEFAULT_HOLD_STEPS,
    collect_random_preset_episode,
)
from POMDPPlanners.training.model_learning.learners import TransitionModelLearner
from POMDPPlanners.training.model_learning.transition_dataset import TransitionDataset
from POMDPPlanners.utils.logger import get_logger

#: Fraction of each round's episodes drawn from the exploration policy. The
#: paper's value, and the one detail it calls out as its own contribution over
#: prior iterative methods -- see the module docstring.
EXPLORATION_FRACTION = 0.5


@dataclass
class RoundResult:
    """What one round of the loop produced.

    Attributes:
        round_index: 1-based round number.
        model: The model fitted at the end of this round, on all data so far.
        dataset_size: Total transitions the fit saw.
        source_counts: Transitions contributed by each source, cumulative.
        training_metrics: Per-epoch training metrics from the fit.
        diagnostics: Held-out diagnostics for this round's model.
    """

    round_index: int
    model: TransitionModel
    dataset_size: int
    source_counts: Dict[str, int] = field(default_factory=dict)
    training_metrics: Dict[str, List[float]] = field(default_factory=dict)
    diagnostics: Dict[str, float] = field(default_factory=dict)


class DAggerModelTrainer:
    """Alternates collecting transitions and refitting a transition model.

    Args:
        world: The true world, stepped forward to generate data. Must expose
            ``sample_next_state``, ``is_terminal`` and ``initial_state_dist``.
            It works in the world's own full-width state; the dataset does the
            slicing.
        diagnostics_world: A view of the same world over the *fitted block*, used
            by the diagnostics that roll the model and the world forward side by
            side. ``None`` reuses ``world``, which is correct only when the
            dataset fits the whole state. The two cannot be one object whenever a
            latent is being carried, and passing the full-width world by mistake
            does not raise -- it compares vectors of different meaning -- so the
            seam is explicit rather than inferred.
        learner: The fitting procedure to use each round.
        dataset: The store transitions accumulate in. Its ``state_indices``
            decide which channels the fit ever sees, which is how a carried
            latent is kept out of it.
        action_presets: Table of action vectors the exploration policy draws from.
        planner_rollout_fn: Called as ``fn(model, round_index, num_episodes)`` and
            expected to run the planner against ``model`` in the true world,
            returning one ``(states, actions, next_states)`` triple per episode.
            Injected rather than built here because how a planner is run --
            budget, parallelism, caching -- is the caller's business. ``None``
            skips the on-policy half, which makes the loop a plain batch fit and
            forfeits the guarantee; useful only for testing the plumbing.
        initial_model: The model round 1's planner searches. Supplying the
            calibrated analytic model starts the loop warm, so no round is spent
            with the planner driving a model fitted on nothing.
        num_rounds: Rounds to run.
        episodes_per_round: Episodes collected per round, split by
            :data:`EXPLORATION_FRACTION`.
        steps_per_episode: Steps attempted per exploration episode.
        hold_steps: Control steps an exploration action is held for.
        horizon: Steps used for the drift diagnostic; use the planner's depth.
        reward_model: Reward used for the ranking diagnostic. ``None`` skips it.
        seed: Seed for exploration draws and diagnostics.
        logger: Optional logger.

    Raises:
        ValueError: If ``num_rounds`` or ``episodes_per_round`` is not positive.

    Example:
        Fit against a world with no planner in the loop, for wiring checks::

            trainer = DAggerModelTrainer(
                world=world,
                learner=LinearGaussianLearner(),
                dataset=TransitionDataset(state_indices=robot_indices),
                action_presets=presets,
                planner_rollout_fn=None,
                num_rounds=2,
            )
            rounds = trainer.run()
    """

    def __init__(
        self,
        world: Any,
        learner: TransitionModelLearner,
        dataset: TransitionDataset,
        action_presets: Any,
        diagnostics_world: Optional[Any] = None,
        planner_rollout_fn: Optional[Callable[[TransitionModel, int, int], Sequence[Any]]] = None,
        initial_model: Optional[TransitionModel] = None,
        num_rounds: int = 5,
        episodes_per_round: int = 40,
        steps_per_episode: int = 40,
        hold_steps: int = DEFAULT_HOLD_STEPS,
        horizon: int = 40,
        reward_model: Optional[Any] = None,
        seed: int = 0,
        logger: Optional[Any] = None,
    ) -> None:
        if num_rounds <= 0:
            raise ValueError(f"num_rounds must be positive, got {num_rounds}")
        if episodes_per_round <= 0:
            raise ValueError(f"episodes_per_round must be positive, got {episodes_per_round}")
        self.world = world
        self.diagnostics_world = world if diagnostics_world is None else diagnostics_world
        self.learner = learner
        self.dataset = dataset
        self.action_presets = np.asarray(action_presets, dtype=float)
        self.planner_rollout_fn = planner_rollout_fn
        self.initial_model = initial_model
        self.num_rounds = num_rounds
        self.episodes_per_round = episodes_per_round
        self.steps_per_episode = steps_per_episode
        self.hold_steps = hold_steps
        self.horizon = horizon
        self.reward_model = reward_model
        self.seed = seed
        self._logger = logger or get_logger(__name__)
        self._rounds: List[RoundResult] = []

    @property
    def rounds(self) -> List[RoundResult]:
        """Every round's result, in order. All are kept, not just the last."""
        return list(self._rounds)

    def run(self) -> List[RoundResult]:
        """Run the loop and return one result per round.

        Returns:
            The rounds, in order. The caller picks among them -- typically by
            evaluating each, since the guarantee is on the best of the sequence.
        """
        current_model = self.initial_model
        for round_index in range(1, self.num_rounds + 1):
            explore_episodes, planner_episodes = self._split(round_index, current_model)
            for states, actions, next_states in explore_episodes:
                self.dataset.add_episode(states, actions, next_states, source="exploration")
            for states, actions, next_states in planner_episodes:
                self.dataset.add_episode(states, actions, next_states, source="planner")

            current_model = self.learner.fit(self.dataset)
            result = RoundResult(
                round_index=round_index,
                model=current_model,
                dataset_size=len(self.dataset),
                source_counts=self.dataset.counts_by_source(),
                training_metrics=self.learner.training_metrics(),
                diagnostics=evaluate_model(
                    model=current_model,
                    world=self.diagnostics_world,
                    holdout=self.dataset.holdout_batch(),
                    action_presets=self.action_presets,
                    horizon=self.horizon,
                    reward_model=self.reward_model,
                    seed=self.seed + round_index,
                ),
            )
            self._rounds.append(result)
            self._logger.info(
                "round %d/%d: %d transitions (%s), %s",
                round_index,
                self.num_rounds,
                result.dataset_size,
                result.source_counts,
                result.diagnostics,
            )
        return self.rounds

    def _split(self, round_index: int, model: Optional[TransitionModel]) -> Any:
        """Collect this round's two halves.

        The exploration half always runs. The planner half runs only once a model
        exists to plan with; in round 1 with no initial model there is nothing to
        drive, so the whole round's budget goes to exploration rather than being
        halved and half of it discarded, and the balance holds from round 2 on.
        """
        can_plan = self.planner_rollout_fn is not None and model is not None
        num_explore = (
            max(1, int(round(self.episodes_per_round * EXPLORATION_FRACTION)))
            if can_plan
            else self.episodes_per_round
        )
        num_planner = self.episodes_per_round - num_explore

        rng = np.random.default_rng(self.seed + 1000 * round_index)
        explore_episodes = [
            collect_random_preset_episode(
                world=self.world,
                action_presets=self.action_presets,
                num_steps=self.steps_per_episode,
                rng=rng,
                hold_steps=self.hold_steps,
            )
            for _ in range(num_explore)
        ]

        planner_episodes: List[Any] = []
        if self.planner_rollout_fn is not None and model is not None and num_planner > 0:
            planner_episodes = list(self.planner_rollout_fn(model, round_index, num_planner))
        return explore_episodes, planner_episodes
