# SPDX-License-Identifier: MIT

"""Rollout transitions accumulated across rounds, sliced to a named state block.

The iterative model-learning loop refits on *every* transition seen so far, which
is what removes the train/test mismatch a one-shot fit carries (Ross & Bagnell,
ICML 2012). The package's existing training buffer cannot serve that: it is
built around iteration slots and drops the oldest, which is right for a policy
chasing a moving target and wrong for a model that should only ever see more
data.

Two rules are enforced here rather than left to the caller, because both fail
silently:

**The fit only sees named channels.** A persistent latent -- a hazard type fixed
at episode start -- is not a random variable of the dynamics, and a fit that
regresses over it flattens exactly the belief dispersion a risk-sensitive
planner grades. Slicing at dataset level means a learner cannot touch it by
accident.

**Held-out rows are split by episode, not by row.** Consecutive transitions of
one trajectory are near-duplicates; a row-wise split puts a row's own neighbours
in the training set and reports a held-out likelihood that flatters every model
equally, which is worse than no number at all.

Classes:
    TransitionBatch: States, actions and next states as parallel arrays.
    TransitionDataset: Growing store of transitions with an episode-wise holdout.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class TransitionBatch:
    """A set of transitions as parallel arrays.

    Attributes:
        states: ``(N, dim)`` source states, already sliced to the fitted block.
        actions: ``(N, action_dim)`` applied action vectors.
        next_states: ``(N, dim)`` resulting states, sliced the same way.
    """

    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray

    def __len__(self) -> int:
        """Number of transitions in the batch."""
        return int(self.states.shape[0])

    @property
    def deltas(self) -> np.ndarray:
        """``next_states - states``, the target a learner should predict.

        Regressing onto the change rather than the absolute successor removes the
        identity part of the map, which is most of it over one control step, and
        leaves the residual the model is actually being asked to learn.
        """
        return self.next_states - self.states


class TransitionDataset:
    """Growing store of ``(state, action, next_state)`` rows with an episode-wise holdout.

    Args:
        state_indices: Flat-vector indices of the block to fit, typically from
            ``IsaacChannelSchema.indices_of(...)``. ``None`` keeps the whole
            state, which is only right when the state has no carried latent.
        holdout_fraction: Fraction of *episodes* reserved for evaluation.
        seed: Seed for the train/holdout episode assignment. The assignment is a
            property of the episode index, so re-adding the same episodes in the
            same order reproduces the same split.

    Raises:
        ValueError: If ``holdout_fraction`` is outside ``[0, 1)``.

    Example:
        >>> import numpy as np
        >>> dataset = TransitionDataset(state_indices=np.array([0, 1]))
        >>> states = np.array([[0.0, 0.0, 9.0], [1.0, 0.0, 9.0]])
        >>> actions = np.array([[1.0], [1.0]])
        >>> next_states = np.array([[1.0, 0.0, 9.0], [2.0, 0.0, 9.0]])
        >>> _ = dataset.add_episode(states, actions, next_states, source="exploration")
        >>> len(dataset)
        2
        >>> dataset.training_batch().states.shape
        (2, 2)
    """

    def __init__(
        self,
        state_indices: Optional[ArrayLike] = None,
        holdout_fraction: float = 0.2,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= holdout_fraction < 1.0:
            raise ValueError(f"holdout_fraction must be in [0, 1), got {holdout_fraction}")
        self._indices = None if state_indices is None else np.asarray(state_indices, dtype=int)
        self._holdout_fraction = float(holdout_fraction)
        self._rng = np.random.default_rng(seed)
        self._episodes: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        """Total number of stored transitions, across every round."""
        return sum(int(episode["states"].shape[0]) for episode in self._episodes)

    @property
    def num_episodes(self) -> int:
        """Number of episodes contributed so far."""
        return len(self._episodes)

    def counts_by_source(self) -> Dict[str, int]:
        """Transitions contributed by each named source, e.g. exploration vs planner.

        The loop's correctness depends on the two sources staying balanced, so the
        count has to be observable rather than assumed.
        """
        counts: Dict[str, int] = {}
        for episode in self._episodes:
            counts[episode["source"]] = counts.get(episode["source"], 0) + int(
                episode["states"].shape[0]
            )
        return counts

    def add_episode(
        self,
        states: ArrayLike,
        actions: ArrayLike,
        next_states: ArrayLike,
        source: str,
    ) -> int:
        """Append one episode's transitions.

        Args:
            states: ``(N, full_dim)`` source states, full width.
            actions: ``(N, action_dim)`` applied actions.
            next_states: ``(N, full_dim)`` resulting states, full width.
            source: Where the episode came from, e.g. ``"exploration"`` or
                ``"planner"``.

        Returns:
            The number of transitions added.

        Raises:
            ValueError: If the three arrays disagree in length.
        """
        states_2d = np.atleast_2d(np.asarray(states, dtype=float))
        actions_2d = np.atleast_2d(np.asarray(actions, dtype=float))
        next_2d = np.atleast_2d(np.asarray(next_states, dtype=float))
        if not states_2d.shape[0] == actions_2d.shape[0] == next_2d.shape[0]:
            raise ValueError(
                "states, actions and next_states must have equal length, got "
                f"{states_2d.shape[0]}, {actions_2d.shape[0]}, {next_2d.shape[0]}"
            )
        if states_2d.shape[0] == 0:
            return 0
        if self._indices is not None:
            states_2d = states_2d[:, self._indices]
            next_2d = next_2d[:, self._indices]
        self._episodes.append(
            {
                "states": states_2d,
                "actions": actions_2d,
                "next_states": next_2d,
                "source": source,
                "holdout": bool(self._rng.random() < self._holdout_fraction),
            }
        )
        return int(states_2d.shape[0])

    def add_history(self, history: Any, source: str) -> int:
        """Append the transitions of one recorded episode.

        The terminal bookkeeping step carries no action and no successor, and a
        step whose successor is a post-reset observation is not a transition of
        the system at all -- the simulator auto-resets inside ``step()``, so that
        successor belongs to a fresh episode metres away. Both are dropped.

        Args:
            history: A :class:`~POMDPPlanners.core.simulation.history.History`.
            source: Where the episode came from.

        Returns:
            The number of usable transitions added.
        """
        states: List[Any] = []
        actions: List[Any] = []
        next_states: List[Any] = []
        for step in history.history:
            if step.action is None or step.next_state is None:
                continue
            states.append(np.asarray(step.state, dtype=float).ravel())
            actions.append(np.asarray(step.action, dtype=float).ravel())
            next_states.append(np.asarray(step.next_state, dtype=float).ravel())
        if not states:
            return 0
        return self.add_episode(states, actions, next_states, source=source)

    def training_batch(self) -> TransitionBatch:
        """Every transition from episodes not held out."""
        return self._batch(holdout=False)

    def holdout_batch(self) -> TransitionBatch:
        """Every transition from held-out episodes."""
        return self._batch(holdout=True)

    def _batch(self, holdout: bool) -> TransitionBatch:
        selected = [episode for episode in self._episodes if episode["holdout"] is holdout]
        if not selected:
            width = self._width()
            empty_state = np.zeros((0, width), dtype=float)
            return TransitionBatch(empty_state, np.zeros((0, self._action_width())), empty_state)
        return TransitionBatch(
            np.concatenate([episode["states"] for episode in selected]),
            np.concatenate([episode["actions"] for episode in selected]),
            np.concatenate([episode["next_states"] for episode in selected]),
        )

    def _width(self) -> int:
        if self._indices is not None:
            return int(self._indices.size)
        return int(self._episodes[0]["states"].shape[1]) if self._episodes else 0

    def _action_width(self) -> int:
        return int(self._episodes[0]["actions"].shape[1]) if self._episodes else 0


def block_indices(schema: Any, channels: Sequence[str]) -> np.ndarray:
    """Flat-vector indices of ``channels`` in ``schema``.

    A thin pass-through so a caller building a dataset does not have to import
    the Isaac schema module for one call, and so the intent -- "fit only these
    channels" -- reads at the call site.

    Args:
        schema: Anything exposing ``indices_of(names)``, e.g.
            ``IsaacChannelSchema``.
        channels: Channel names to fit.

    Returns:
        The concatenated flat indices, in the order given.
    """
    return np.asarray(schema.indices_of(channels), dtype=int)
