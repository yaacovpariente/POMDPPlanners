"""PolicyTrainer: orchestrates the collect-then-train loop for trainable policies.

This module provides a single concrete trainer that works with any policy
implementing the :class:`~POMDPPlanners.core.policy.TrainablePolicy` mixin.
It replaces the duplicated ``fit()`` methods that previously lived on
:class:`~POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero.BetaZero`
and its subclasses.

Classes:
    PolicyTrainer: Concrete trainer implementing the collect-then-train loop.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from POMDPPlanners.core.policy import TrainablePolicy
from POMDPPlanners.training.callbacks import TrainerCallback
from POMDPPlanners.utils.logger import get_logger


class PolicyTrainer:
    """Orchestrates offline policy-iteration training for trainable policies.

    The trainer alternates between two phases for ``num_iterations`` rounds:

    1. **Collect** — run episodes using the policy (MCTS-based or batched
       network-only) and store transitions in the policy's replay buffer.
    2. **Train** — call the policy's ``train_step()`` hook to update the
       network on the buffered data.

    Callbacks (:class:`~POMDPPlanners.training.callbacks.TrainerCallback`)
    are fired at well-defined points in the loop and can be used for early
    stopping, model checkpointing, or Optuna integration.

    Attributes:
        policy: A policy that implements :class:`TrainablePolicy`.
        initial_belief_fn: Callable returning a fresh initial belief.
        num_iterations: Number of collect-then-train rounds.
        episodes_per_iteration: Episodes to collect per round.
        episode_length: Maximum steps per episode.
        verbose: Whether to log progress information.
        batched_collection: Use fast network-only rollouts instead of MCTS.
        callbacks: Sequence of :class:`TrainerCallback` instances.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> from POMDPPlanners.utils.action_samplers import DiscreteActionSampler
        >>> from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero import BetaZero
        >>> from POMDPPlanners.training import PolicyTrainer
        >>>
        >>> env = TigerPOMDP(discount_factor=0.95)
        >>> sampler = DiscreteActionSampler(env.get_actions())
        >>> planner = BetaZero(
        ...     environment=env, discount_factor=0.95, depth=2,
        ...     name="BZ", action_sampler=sampler, n_simulations=5,
        ...     state_dim=1, training_epochs=2, training_batch_size=4,
        ... )
        >>> belief_fn = lambda: get_initial_belief(env, n_particles=5)
        >>> trainer = PolicyTrainer(
        ...     policy=planner, initial_belief_fn=belief_fn,
        ...     num_iterations=1, episodes_per_iteration=2, episode_length=5,
        ...     verbose=False,
        ... )
        >>> metrics = trainer.train()
        >>> "total_loss" in metrics
        True
    """

    def __init__(
        self,
        policy: TrainablePolicy,
        initial_belief_fn: Callable[[], Any],
        num_iterations: int = 10,
        episodes_per_iteration: int = 50,
        episode_length: int = 100,
        verbose: bool = True,
        batched_collection: bool = False,
        callbacks: Optional[Sequence[TrainerCallback]] = None,
    ):
        if not isinstance(policy, TrainablePolicy):
            raise TypeError(f"policy must implement TrainablePolicy, got {type(policy).__name__}")
        self.policy = policy
        self.initial_belief_fn = initial_belief_fn
        self.num_iterations = num_iterations
        self.episodes_per_iteration = episodes_per_iteration
        self.episode_length = episode_length
        self.verbose = verbose
        self.batched_collection = batched_collection
        self.callbacks: Sequence[TrainerCallback] = callbacks or ()

    @property
    def logger(self) -> logging.Logger:
        return get_logger(name="training.PolicyTrainer")

    # ── Public API ────────────────────────────────────────────────────

    def train(self) -> Dict[str, List[float]]:
        """Run the collect-then-train loop.

        Returns:
            Dictionary mapping metric keys (from ``policy.get_metric_keys()``)
            to lists of per-iteration loss values.
        """
        all_metrics = self._init_metrics()
        self._fire_on_train_begin()

        for iteration in range(self.num_iterations):
            self._fire_on_iteration_begin(iteration)
            self._collect_data(iteration)
            self._fire_on_collection_end(iteration)

            if self.policy.buffer_size() == 0:
                self._log_empty_buffer(iteration)
                continue

            metrics = self.policy.train_step()
            self._append_metrics(all_metrics, metrics)
            self._log_iteration(iteration, metrics)

            if self._fire_on_iteration_end(iteration, metrics):
                break

        self._fire_on_train_end(all_metrics)
        return all_metrics

    # ── Collection logic ──────────────────────────────────────────────

    def _collect_data(self, iteration: int) -> None:
        self._fire_on_collection_begin(iteration)

        if self.batched_collection:
            self.policy.collect_episodes_batched(
                initial_belief_fn=self.initial_belief_fn,
                n_episodes=self.episodes_per_iteration,
                episode_length=self.episode_length,
            )
        else:
            self._collect_episodes_mcts()

    def _collect_episodes_mcts(self) -> None:
        from POMDPPlanners.simulations.episodes import (  # pylint: disable=import-outside-toplevel
            EpisodeRunner,
        )

        self.policy.begin_collecting()
        for _ in range(self.episodes_per_iteration):
            self.policy.prepare_episode()
            runner = EpisodeRunner(
                environment=self.policy.environment,  # type: ignore[attr-defined]
                policy=self.policy,  # type: ignore[arg-type]
                initial_belief=self.initial_belief_fn(),
                num_steps=self.episode_length,
            )
            history = runner.run()
            self.policy.finalize_episode(history)
        self.policy.end_collecting()

    # ── Metric helpers ────────────────────────────────────────────────

    def _init_metrics(self) -> Dict[str, List[float]]:
        return {key: [] for key in self.policy.get_metric_keys()}

    @staticmethod
    def _append_metrics(all_metrics: Dict[str, List[float]], new: Dict[str, List[float]]) -> None:
        for key in all_metrics:
            if key in new and new[key]:
                all_metrics[key].append(new[key][-1])

    # ── Callback dispatch ─────────────────────────────────────────────

    def _fire_on_train_begin(self) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(self)

    def _fire_on_train_end(self, all_metrics: Dict[str, List[float]]) -> None:
        for cb in self.callbacks:
            cb.on_train_end(self, all_metrics)

    def _fire_on_iteration_begin(self, iteration: int) -> None:
        for cb in self.callbacks:
            cb.on_iteration_begin(self, iteration)

    def _fire_on_iteration_end(self, iteration: int, metrics: Dict[str, List[float]]) -> bool:
        for cb in self.callbacks:
            if cb.on_iteration_end(self, iteration, metrics):
                return True
        return False

    def _fire_on_collection_begin(self, iteration: int) -> None:
        for cb in self.callbacks:
            cb.on_collection_begin(self, iteration)

    def _fire_on_collection_end(self, iteration: int) -> None:
        for cb in self.callbacks:
            cb.on_collection_end(self, iteration)

    # ── Logging ───────────────────────────────────────────────────────

    def _log_empty_buffer(self, iteration: int) -> None:
        if self.verbose:
            self.logger.info(
                "Iteration %d: no training data collected, skipping training",
                iteration,
            )

    def _log_iteration(self, iteration: int, metrics: Dict[str, List[float]]) -> None:
        if not self.verbose:
            return
        parts = [f"Iteration {iteration}:"]
        for key in self.policy.get_metric_keys():
            vals = metrics.get(key, [])
            val = vals[-1] if vals else float("nan")
            parts.append(f"{key}={val:.4f}")
        self.logger.info(", ".join(parts))
